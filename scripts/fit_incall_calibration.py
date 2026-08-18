#!/usr/bin/env python3
"""Fit a serving stack's two calibrations from ONE gold run.

WHAT THIS IS FOR
----------------
A new serving stack (a vLLM H200, say) can read verdicts and margins on day
one, but neither number means anything downstream until two SEPARATE artifacts
exist for it. They are fitted on different quantities, keyed differently, and
either can exist without the other:

  BELIEF PROFILE   keyed (model, prompt_sha256)
      the two verdict log-LRs. Without it EVERY belief is hard gate --
      measured ECE 0.237 against 0.045 calibrated, with nothing downstream
      able to tell which it got.

  IN-CALL ISOTONIC keyed (model, served_model_id)
      delta_logit -> p_hat, which becomes the additive weight of evidence.
      Without it a margin is an uninterpretable raw number and every row
      keeps its verdict weight.

Both come out of the same gold run, which is the point of this script: one pass
over gold on the target stack, two artifacts, no second corpus read.

WHY THE ISOTONIC CANNOT BE BORROWED
-----------------------------------
delta_logit magnitudes are a property of the STACK, not the weights. The same
model in-process and over HTTP correlates at r=0.955 yet differs 2.4x in range
and disagrees in SIGN on 10% of rows, and the shipped probe isotonic (knots
-1.70..+1.61) maps a typical in-call margin of |13| to exactly 1.0. A borrowed
curve does not error; it returns saturated confidence that looks ordinary.

WHAT IT REFUSES TO DO
---------------------
It does not register anything. It writes an artifact and PRINTS the two
registry edits for a human to make, because both are ship decisions: the
belief profile changes every belief the reader produces, and the isotonic
decides whether logits reach belief at all. It also reports a held-out
comparison rather than assuming the margin helps -- on this codebase's own
history, a signal with real within-verdict AUROC (deliberation length, 0.56 to
0.72 across 15 arms) delivered NO held-out gain. The gate is held-out or
nothing.

USAGE
    python scripts/fit_incall_calibration.py \
        --run data/results/vllm_verdict_only_gold.jsonl \
        --model vllm-gemma-4-26b --served-model-id google/gemma-4-26B-A4B-it \
        --out data/probe_battery/incall_calibration_vllm.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

# Imported, not spelled: the loader validates the artifact's probe_ids against
# the same constant, and a local string literal here is how a fitted artifact
# came to be unloadable by the code meant to consume it.
from indra_belief.probes.reader import IN_CALL_PROBE_ID as PROBE_ID  # noqa: E402


def load_rows(path: Path) -> list[dict]:
    """Gold rows carrying BOTH a gold label and a measured margin.

    A row missing either is excluded and counted rather than imputed: a margin
    the reader could not produce is not a margin of zero, and treating it as one
    would drag the fitted curve toward the middle.
    """
    rows, skipped = [], {"no_gold": 0, "no_margin": 0, "no_verdict": 0}
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            gold = row.get("gold_correct")
            margin = row.get("probe_delta_logit")
            verdict = row.get("verdict")
            if gold is None:
                skipped["no_gold"] += 1
                continue
            if verdict not in {"correct", "incorrect"}:
                skipped["no_verdict"] += 1
                continue
            if not isinstance(margin, (int, float)) or isinstance(margin, bool):
                skipped["no_margin"] += 1
                continue
            rows.append({
                "record_id": str(row.get("source_hash") or row.get("row_index")),
                "gold": bool(gold),
                "verdict": verdict,
                "margin": float(margin),
            })
    return rows, skipped


def load_rows_from_shards(input_dir: Path, results_dir: Path, labels_path: Path,
                         limit: int | None = None) -> tuple[list[dict], dict]:
    """Rows read from what the PRODUCTION path actually produced.

    The alternative -- a gold-eval jsonl -- is fitted on a curated population
    read by a different script. That leaves POPULATION skew untouched even now
    that prompt skew is zero, and population skew is not a detail:
    ``fit_prevalence`` is baked into the artifact and anchors every weight
    (weight = logit(p_hat) - logit(fit_prevalence)). A curve fitted at
    prevalence 0.513 and applied to a corpus at 0.70 displaces every weight by
    +0.88 log-odds, and the isotonic's knots sit where the FIT population's
    margins fell, not where the corpus's do.

    Joining on ``source_hash``: it is the evidence identity the shard job, the
    scored cell and the curation all carry, so no positional assumption is
    needed. A label with no scored row and a scored row with no label are both
    counted, never silently paired.
    """
    import gzip as _gzip

    def _runner():
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "_shard_runner_for_fit", ROOT / "scripts/run_vllm_processed_shards.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    # KEYED ON THE PAIR, not the evidence. One Evidence can support several
    # statements, so source_hash alone is not unique -- MEASURED on
    # eval_curation_v1: 1606 rows, 1604 distinct source_hash, and one of the two
    # duplicates carries DISAGREEING gold labels. A dict keyed on source_hash
    # resolves that curator disagreement by file order, silently, which is the
    # same defect shape as the paired-by-file-order bug in the v2 gold.
    #
    # The label is a judgement about a CLAIM against an EVIDENCE, so the pair is
    # the unit, and a pair whose labels conflict is dropped rather than decided
    # by whichever line came last.
    rows: list[dict] = []
    skipped = {"no_gold": 0, "no_margin": 0, "no_verdict": 0, "unscored": 0,
               "no_results_file": 0, "conflicting_labels": 0, "duplicate_pair": 0}
    seen_pairs: set[tuple[str, str]] = set()
    labels: dict[tuple[str, str], bool] = {}
    conflicting: set[tuple[str, str]] = set()
    with labels_path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            gold = row.get("gold_correct")
            if gold is None and row.get("gold") in {"correct", "incorrect"}:
                gold = row["gold"] == "correct"
            source_hash = row.get("source_hash")
            stmt_hash = row.get("matches_hash", row.get("stmt_hash"))
            if gold is None or source_hash in (None, "") or stmt_hash in (None, ""):
                continue
            key = (str(stmt_hash), str(source_hash))
            if key in labels and labels[key] != bool(gold):
                conflicting.add(key)
            labels[key] = bool(gold)
    for key in conflicting:
        labels.pop(key, None)
    skipped["conflicting_labels"] = len(conflicting)

    for shard in sorted(Path(input_dir).glob("grounded-*.jsonl.gz")):
        index = int(re.search(r"grounded-(\d+)", shard.name).group(1))
        # Resolved, not reconstructed: output names carry the scoring run's
        # --limit, and rebuilding the unlimited name misses every shard and
        # reports "0 usable rows" as though the data were bad rather than
        # unfound.
        results = _runner().resolve_results_path(Path(results_dir), index)
        if results is None:
            skipped["no_results_file"] += 1
            continue
        with _gzip.open(results, "rt", encoding="utf-8") as fh:
            verdicts = json.load(fh)
        with _gzip.open(shard, "rt", encoding="utf-8") as fh:
            for line in fh:
                job = json.loads(line)
                source_hash = str(job.get("source_hash"))
                stmt_hash = str(job.get("stmt_hash"))
                key = (stmt_hash, source_hash)
                if key not in labels:
                    skipped["no_gold"] += 1
                    continue
                cell = (verdicts.get(stmt_hash) or {}).get(source_hash)
                if not cell:
                    skipped["unscored"] += 1
                    continue
                if cell.get("verdict") not in {"correct", "incorrect"}:
                    skipped["no_verdict"] += 1
                    continue
                margin = cell.get("probe_delta_logit")
                if not isinstance(margin, (int, float)) or isinstance(margin, bool):
                    skipped["no_margin"] += 1
                    continue
                # The record id is the PAIR too. The combiner requires unique
                # ids and refuses duplicates outright, so a bare source_hash
                # crashes the fit the moment one evidence supports two
                # statements -- which it did, on the first full-corpus run.
                # One READING per pair. The combiner requires unique record ids
                # and refuses duplicates outright, and a pair fitted twice would
                # weigh that evidence double in the curve. The corpus legitimately
                # repeats a pair (eval_curation_v1: 1606 entries, 1604 distinct
                # pairs), and the scoring path is deterministic -- verified
                # 290/290 identical margins across two runs -- so the second copy
                # carries no new information.
                if key in seen_pairs:
                    skipped["duplicate_pair"] += 1
                    continue
                seen_pairs.add(key)
                rows.append({"record_id": f"{stmt_hash}:{source_hash}",
                             "gold": labels[key],
                             "source_api": job.get("source_api"),
                             "verdict": cell["verdict"], "margin": float(margin)})
                if limit and len(rows) >= limit:
                    return rows, skipped
    return rows, skipped


def split(rows: list[dict], holdout_frac: float, seed: int):
    """Deterministic split on a hash of the record id.

    Hash-based, not random-shuffle: re-running with the same gold must produce
    the same split, or the held-out number is a lottery ticket that can be
    re-drawn until it passes.
    """
    fit, held = [], []
    for row in rows:
        digest = hashlib.sha256(f"{seed}:{row['record_id']}".encode()).digest()
        bucket = int.from_bytes(digest[:4], "big") / 2**32
        (held if bucket < holdout_frac else fit).append(row)
    return fit, held


def belief_profile(rows: list[dict]) -> dict:
    """The verdict log-LRs, from the 2x2 of verdict against gold.

    cc/ci/ic/ii are the counts the registry stores, so they are reported rather
    than only the derived rates -- a rate cannot be re-derived into a count, and
    the count is what makes a later refit auditable.
    """
    cc = sum(1 for r in rows if r["verdict"] == "correct" and r["gold"])
    ci = sum(1 for r in rows if r["verdict"] == "correct" and not r["gold"])
    ic = sum(1 for r in rows if r["verdict"] == "incorrect" and r["gold"])
    ii = sum(1 for r in rows if r["verdict"] == "incorrect" and not r["gold"])
    sens = cc / max(cc + ic, 1)
    fpr = ci / max(ci + ii, 1)
    prevalence = (cc + ic) / max(len(rows), 1)
    if min(sens, fpr) <= 0 or max(sens, fpr) >= 1:
        raise SystemExit(
            f"degenerate confusion (cc={cc} ci={ci} ic={ic} ii={ii}); a log-LR "
            "would be infinite. More gold, or a reader that is not perfect."
        )
    return {
        "counts": {"cc": cc, "ci": ci, "ic": ic, "ii": ii},
        "sensitivity": sens,
        "false_positive_rate": fpr,
        "log_lr_confirm": math.log(sens / fpr),
        "log_lr_reject": math.log((1 - sens) / (1 - fpr)),
        "prior_logodds": math.log(prevalence / (1 - prevalence)) if 0 < prevalence < 1 else 0.0,
        "n": len(rows),
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", help="gold-eval jsonl with probe_delta_logit")
    # The production-path alternative. Same builder, same runner, same prompt and
    # transport as the 60M run, and fitted on the population it will score.
    ap.add_argument("--from-shards", action="store_true",
                    help="fit from scored SHARDS instead of a gold-eval jsonl")
    ap.add_argument("--input-dir", help="prepared shards (with --from-shards)")
    ap.add_argument("--results-dir", help="scored shards (with --from-shards)")
    ap.add_argument("--gold", help="labels jsonl, joined on source_hash")
    ap.add_argument("--model", required=True)
    ap.add_argument("--served-model-id", required=True)
    ap.add_argument("--variant", default="verdict_only")
    ap.add_argument("--out", required=True, help="isotonic artifact path")
    ap.add_argument("--holdout-frac", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--reseeds", type=int, default=10,
                    help="re-run the whole fit+evaluate under this many splits. "
                         "One split is one lottery draw; this codebase has twice "
                         "shipped a signal that a different partition dissolved")
    args = ap.parse_args()

    from indra_belief.calibration_gate import gate_decision
    from indra_belief.metrics import auroc, brier_murphy, ece
    from indra_belief.probe_combiner import fit_combiner
    from indra_belief.scorers.monolithic import scorer as mono

    if args.from_shards:
        missing = [f for f in ("input_dir", "results_dir", "gold")
                   if not getattr(args, f)]
        if missing:
            raise SystemExit(f"--from-shards needs --{', --'.join(m.replace('_','-') for m in missing)}")
        rows, skipped = load_rows_from_shards(
            Path(args.input_dir), Path(args.results_dir), Path(args.gold))
        source = f"shards {args.results_dir}"
    elif args.run:
        rows, skipped = load_rows(Path(args.run))
        source = f"gold-eval run {args.run}"
    else:
        raise SystemExit("pass --run, or --from-shards with --input-dir/--results-dir/--gold")
    print(f"  fitting on: {source}")
    if len(rows) < 100:
        if skipped.get("no_results_file"):
            raise SystemExit(
                f"no scored output was found for {skipped['no_results_file']} "
                "shard(s). Either the scoring run has not been done, or this "
                "directory holds two generations of the same shard and the "
                "ambiguity was refused rather than guessed."
            )
        raise SystemExit(
            f"only {len(rows)} usable rows (skipped {skipped}); a curve fitted on "
            "this little would be noise wearing a calibration's clothes"
        )
    fit_rows, held_rows = split(rows, args.holdout_frac, args.seed)
    print(f"  rows={len(rows)} fit={len(fit_rows)} holdout={len(held_rows)} "
          f"skipped={skipped}", flush=True)

    # ---- 1. the belief profile, from the WHOLE set --------------------------
    # Fitted on everything on purpose: it is a property of the reader, it is
    # what the registry stores as counts, and it is evaluated by the ship gate
    # rather than by this script's holdout.
    profile = belief_profile(rows)
    print(f"\n  [belief profile] {profile['counts']} "
          f"sens={profile['sensitivity']:.4f} fpr={profile['false_positive_rate']:.4f}")
    print(f"    log_lr_confirm={profile['log_lr_confirm']:+.4f} "
          f"log_lr_reject={profile['log_lr_reject']:+.4f}")

    # ---- 2+3. fit and evaluate under MANY splits ---------------------------
    # One split is one draw. This codebase has twice shipped a signal that a
    # different partition dissolved (deliberation length; 15 of 16 probes), and
    # in both cases the in-sample picture looked fine. So the gate reads the
    # DISTRIBUTION over splits, not a single lucky one, and reports the worst.
    # Only available from the shard path; a gold-eval jsonl carries no
    # source_api, and an absent mapping means the floor is simply not applied
    # rather than applied wrongly with a default.
    source_api_of = {r["record_id"]: r.get("source_api") for r in rows
                     if r.get("source_api")}

    def evaluate(seed: int):
        fit_rows, held_rows = split(rows, args.holdout_frac, seed)
        if len(held_rows) < 30 or len({r["gold"] for r in held_rows}) < 2:
            return None
        combiner = fit_combiner(
            np.asarray([[r["margin"]] for r in fit_rows], dtype=float),
            np.asarray([r["gold"] for r in fit_rows], dtype=bool),
            probe_ids=[PROBE_ID],
            record_ids=[r["record_id"] for r in fit_rows],
        )
        held_y = np.asarray([r["gold"] for r in held_rows], dtype=bool)
        p_hat = combiner.score(
            np.asarray([[r["margin"]] for r in held_rows], dtype=float),
            record_ids=[r["record_id"] for r in held_rows], probe_ids=[PROBE_ID])

        # REPLACEMENT, not addition. statement_belief uses probe_weight INSTEAD
        # OF verdict_weight on a row carrying a measurement
        # (statement_belief.py:176); it never sums them, because the margin
        # already encodes the verdict and adding both double-counts it.
        #
        # This was wrong here first, and the wrong version looked BETTER on the
        # metric being gated: summing gave +0.0748 AUROC while TRIPLING ECE
        # (0.0374 -> 0.1241). A gate that scores a composition production never
        # runs is not a gate.
        lc, lr_ = profile["log_lr_confirm"], profile["log_lr_reject"]
        incumbent = np.asarray(
            [lc if r["verdict"] == "correct" else lr_ for r in held_rows])
        base = float(combiner.fit_prevalence)
        eps = 1e-6
        candidate = np.log(np.clip(p_hat, eps, 1 - eps)
                           / np.clip(1 - p_hat, eps, 1 - eps)) - math.log(base / (1 - base))
        # FLOORED, because production floors. `probe_weight` applies
        # max(weight, source_logodds) to a confirming reading so a generic
        # reader cannot drag a well-curated source below its own track record
        # (evidence_weights.py:106). Scoring the un-floored weight is the same
        # error as scoring the additive composition: a gate must evaluate the
        # arithmetic that runs.
        #
        # MEASURED on this gold set the floor binds on 0 of 74 held-out rows, so
        # it changes nothing here -- but the corpus source mix is not the gold
        # source mix, and a gate that is right by accident is not right.
        if source_api_of:
            from indra_belief.evidence_weights import probe_weight, source_logodds_for
            from indra_belief.noise_model import RECALIBRATED_PRIORS

            candidate = np.asarray([
                probe_weight(float(w), source_logodds_for(
                    source_api_of.get(r["record_id"]), RECALIBRATED_PRIORS))
                for w, r in zip(candidate, held_rows)
            ])

        # Paired bootstrap: the two scores are read off the SAME rows, so an
        # unpaired interval would be far too wide.
        rng = np.random.default_rng(seed)
        n = len(held_rows)
        deltas = []
        for _ in range(1000):
            idx = rng.integers(0, n, n)
            if len(set(held_y[idx].tolist())) < 2:
                continue
            deltas.append(auroc(candidate[idx], held_y[idx])
                          - auroc(incumbent[idx], held_y[idx]))
        lo = float(np.percentile(deltas, 2.5)) if deltas else float("nan")

        def probs(x):
            return 1.0 / (1.0 + np.exp(-(x + profile["prior_logodds"])))
        p_inc, p_can = probs(incumbent), probs(candidate)
        labels = held_y.tolist()
        bi = brier_murphy(p_inc, held_y)
        bc = brier_murphy(p_can, held_y)
        return {
            "seed": seed, "n": n,
            "rel_inc": bi["reliability"], "rel_can": bc["reliability"],
            "res_inc": bi["resolution"], "res_can": bc["resolution"],
            "auroc_inc": auroc(incumbent, held_y), "auroc_can": auroc(candidate, held_y),
            "ci_low": lo,
            "brier_inc": bi["brier"], "brier_can": bc["brier"],
            "ece_inc": ece(list(zip(p_inc.tolist(), labels))),
            "ece_can": ece(list(zip(p_can.tolist(), labels))),
            "combiner": combiner,
        }

    results = [r for r in (evaluate(args.seed + i) for i in range(max(args.reseeds, 1)))
               if r is not None]
    if not results:
        raise SystemExit("no split produced an evaluable holdout")

    print(f"\n  [held-out, {len(results)} splits]  (candidate REPLACES the verdict weight)")
    print(f"    {'seed':>5}{'n':>6}{'AUROC inc':>11}{'AUROC can':>11}{'CI low':>9}"
          f"{'Brier inc':>11}{'Brier can':>11}{'ECE can':>9}")
    for r in results:
        print(f"    {r['seed']:>5}{r['n']:>6}{r['auroc_inc']:>11.4f}{r['auroc_can']:>11.4f}"
              f"{r['ci_low']:>+9.4f}{r['brier_inc']:>11.4f}{r['brier_can']:>11.4f}"
              f"{r['ece_can']:>9.4f}")

    med = lambda k: float(np.median([r[k] for r in results]))  # noqa: E731
    worst_lo = min(r["ci_low"] for r in results)
    n_pass = sum(1 for r in results
                 if gate_decision(r["ci_low"], r["brier_inc"], r["brier_can"],
                                  r["rel_inc"], r["rel_can"],
                                  r["res_inc"], r["res_can"])["pass"])
    print(f"\n    median CI low {med('ci_low'):+.4f}   WORST {worst_lo:+.4f}")
    print(f"    median Brier  {med('brier_inc'):.4f} -> {med('brier_can'):.4f}")
    print(f"    median ECE    {med('ece_inc'):.4f} -> {med('ece_can'):.4f}")
    print(f"    splits passing both halves: {n_pass}/{len(results)}")

    # Gated on the MEDIAN, with the worst split reported beside it. Gating on the
    # best would be split-shopping; gating on the worst would let one unlucky
    # partition veto a real effect.
    g = gate_decision(med("ci_low"), med("brier_inc"), med("brier_can"),
                      med("rel_inc"), med("rel_can"), med("res_inc"), med("res_can"))
    discriminates, scores_better, verdict_go = g["ranking"], g["scoring"], g["pass"]
    # THE SHIPPED CURVE IS REFITTED ON ALL ROWS. Previously this shipped
    # `results[0]["combiner"]` -- seed 0's fit split -- while the decision was
    # made from the median over every split, so the artifact that went out was
    # not the artifact that was gated, and it was fitted on 70% of the data for
    # no reason. The reseeds establish that the RELATIONSHIP generalises; once
    # that is established the best estimate of it uses everything.
    #
    # The cost is that every fitted row is in-sample, so the combiner will
    # refuse to score them later. That is correct and it is small: the fit set
    # is the gold subset, and against 60M corpus rows only those are refused.
    combiner = fit_combiner(
        np.asarray([[r["margin"]] for r in rows], dtype=float),
        np.asarray([r["gold"] for r in rows], dtype=bool),
        probe_ids=[PROBE_ID],
        record_ids=[r["record_id"] for r in rows],
    )
    b_inc, b_can = med("brier_inc"), med("brier_can")
    e_inc, e_can = med("ece_inc"), med("ece_can")

    print(f"\n  GATE: {'PASS' if verdict_go else 'FAIL'}")
    print(f"    ranking   {'PASS' if discriminates else 'FAIL'}  "
          f"median AUROC CI {'excludes' if discriminates else 'spans'} 0")
    print(f"    scoring   {'PASS' if scores_better else 'FAIL'}  "
          f"median Brier {'improves' if scores_better else 'does not improve'} "
          f"({b_inc:.4f} -> {b_can:.4f})")
    if n_pass < len(results):
        print(f"    NOTE      {len(results) - n_pass} split(s) did NOT pass. The effect "
              "is partition-sensitive; more gold is the remedy.")
    ratio = "inf" if g["ratio"] == float("inf") else f"{g['ratio']:.1f}x"
    print(f"    trade     {'ok  ' if g['favourable'] else 'POOR'}  (diagnostic, not gated) "
          f"reliability cost {g['reliability_cost']:+.4f}, resolution gain "
          f"{g['resolution_gain']:+.4f} ({ratio})")
    if e_can > e_inc:
        print(f"    NOTE      ECE rises {e_inc:.4f} -> {e_can:.4f}. That is the "
              "reliability cost above, priced against the resolution gain rather "
              "than vetoed -- see gate_decision for why the ECE rule in "
              "fit_probe_belief_model.py does not transfer to this incumbent.")
    if not verdict_go:
        print("    On this codebase's history, a signal that ranks better without "
              "scoring better is exactly how deliberation length looked before it "
              "delivered no held-out gain.")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(combiner.to_dict(), indent=2, sort_keys=True) + "\n")

    prompt_sha = hashlib.sha256(
        mono.VARIANTS[args.variant].system_prompt.encode("utf-8")).hexdigest()
    print(f"\n  wrote {out}")
    print("\n  ---- registry edits, for a human to make ----")
    print(f"  1. probes/calibration.py  _SENTENCE_CALIBRATIONS:")
    print(f'       ("{args.model}", "{args.served_model_id}"): "{out.name}",')
    print(f"  2. calibration_constants.py  a profile for "
          f"({args.model}, prompt {prompt_sha[:12]}) with counts "
          f"{profile['counts']}")
    print("\n  Neither is applied here: both are ship decisions. The first "
          "decides whether logits reach belief at all; the second changes every "
          "belief this reader produces.")
    return 0 if verdict_go else 2


if __name__ == "__main__":
    raise SystemExit(main())
