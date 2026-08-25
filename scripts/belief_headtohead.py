"""Head-to-head: LLM statement belief vs the text-miner baseline, on gold.

For every production-readable statement covered by both the gold and a model
run, we compute:
  - belief_llm   = the hard-gate fallback (recalibrated source priors)
  - belief_llm_soft = the fitted reader's configuration-specific hybrid log-odds score
  - belief_recal = INDRA parametric belief recomputed from source_counts (no text read)
  - belief_indra = same, under INDRA default priors
  - belief_stored = INDRA belief as written on the statement (incl. propagation)
and score each against the calibration arc's conservative statement gold
(correct only when every curated evidence is correct): AUROC (positive=correct)
and 8-bin ECE, overall and split by evidence depth. Gold joins only on the exact
``(matches_hash, source_hash)`` pair, statements group on the run's
``stmt_hash``, and ``no_text``/unread evidence follows the production
``statement_belief`` behavior. Everything is on the SAME readable covered
statement set so the comparison is paired. We also report the tiered
verdict_statement's error-detection at statement grain.

    python scripts/belief_headtohead.py \
        --gold data/benchmark/eval_curation_v1.jsonl \
        --run  data/results/eval_curation_v1_gemma.jsonl \
        --label remote-gemma-4-26b \
        --model remote-gemma-4-26b --out-json data/results/belief_headtohead_gemma.json \
        --out-md reports/belief_headtohead_gemma.md
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from indra_belief.metrics import auroc as _canonical_auroc
from indra_belief.metrics import confusion_pr, ece  # noqa: E402
from indra_belief.noise_model import (  # noqa: E402
    RECALIBRATED_PRIORS,
    compute_edge_reliability_from_counts,
)
from indra_belief.indra_priors import (  # noqa: E402
    INDRA_DEFAULT_PRIOR_RESOURCE,
    INDRA_DEFAULT_PRIORS,
    INDRA_DEFAULT_PRIORS_SHA256,
    with_benchmark_recalibration,
)
from indra_belief.statement_belief import statement_belief  # noqa: E402
from indra_belief.calibration_constants import (  # noqa: E402
    calibration_for_run,
    reader_configuration_for_run,
)
from indra_belief.curation import aggregate_gold, is_gold_correct  # noqa: E402
from indra_belief.results import GoldMap, load_gold_map  # noqa: E402

_HASH_MASK = (1 << 64) - 1

# The benchmark only overrides sources it measured. All other sources retain
# the installed INDRA default instead of collapsing to noise_model's generic
# unknown-source fallback.
RECALIBRATED_WITH_INDRA_DEFAULTS = with_benchmark_recalibration(RECALIBRATED_PRIORS)


def ukey(x):
    try:
        return int(x) & _HASH_MASK
    except (ValueError, TypeError):
        return None


def run_statement_key(value) -> tuple[str | None, int | None]:
    """Return the production grouping key and its unsigned matches hash.

    Scored runs persist ``stmt_hash`` as a hexadecimal string. The original
    string is the statement-grain key; its integer form is used only for the
    exact gold-pair lookup.
    """
    if value is None:
        return None, None
    display = str(value)
    try:
        return display, int(display, 16) & _HASH_MASK
    except (ValueError, TypeError):
        return display, None


def load_exact_gold(path: str) -> tuple[GoldMap, dict[tuple[int, int], dict], int]:
    """Load aggregated gold plus one deterministic payload per exact pair.

    ``load_gold_map`` owns multi-curator any-incorrect-wins aggregation. This
    analysis deliberately reads only ``GoldMap.by_pair``; the truth-consistent
    source-only compatibility fallback is inappropriate when run ``stmt_hash``
    provides the authoritative statement provenance.
    """
    gold_map = load_gold_map(path)
    payload_by_pair: dict[tuple[int, int], dict] = {}
    n_rows = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n_rows += 1
            row = json.loads(line)
            mh = ukey(row.get("matches_hash"))
            sh = ukey(row.get("source_hash"))
            if mh is not None and sh is not None:
                payload_by_pair.setdefault((mh, sh), row)
    if not gold_map.by_pair:
        raise ValueError(
            f"{path} has no usable (matches_hash, source_hash) gold pairs; "
            "source-hash fallback is intentionally disabled"
        )
    return gold_map, payload_by_pair, n_rows


def auroc(scored: list[tuple[float, bool]]) -> float | None:
    """AUROC with positive class = label True. None if either class is empty.

    The rank (Mann-Whitney) body this used to carry was byte-identical to
    `indra_belief.metrics.auroc`, so the estimator now lives there and this is
    the adapter: unzip the pairs, and map the canonical `nan`-on-degenerate back
    to the `None` this module's callers (and its JSON output, where nan is not
    valid) have always seen.
    """
    if not scored:
        return None
    scores = [s for s, _ in scored]
    labels = [bool(lab) for _, lab in scored]
    if not any(labels) or all(labels):
        return None
    return float(_canonical_auroc(scores, labels))


def discrimination(rows: list[dict], key: str) -> dict:
    scored = [(r[key], r["gold_correct"]) for r in rows if isinstance(r.get(key), (int, float))]
    if not scored:
        return {"n": 0, "auroc": None, "ece": None}
    a = auroc(scored)
    return {
        "n": len(scored),
        "auroc": round(a, 4) if a is not None else None,
        "ece": round(ece(scored), 4),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default="data/benchmark/eval_curation_v1.jsonl")
    ap.add_argument("--run", default="data/results/eval_curation_v1_gemma.jsonl")
    ap.add_argument("--label", default="model")
    ap.add_argument("--model", default=None,
                    help="reader model name for exact fitted-profile lookup (default: --label)")
    ap.add_argument("--out-json", required=True,
                    help="explicit provenance-specific JSON artifact path")
    ap.add_argument("--out-md", required=True,
                    help="explicit provenance-specific Markdown artifact path")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing JSON artifact with different inputs")
    args = ap.parse_args()

    # Exact pair lookup only. load_gold_map aggregates repeated curator rows for
    # each pair with the shared conservative any-incorrect-wins rule.
    gold, gold_payload, n_gold_rows = load_exact_gold(args.gold)

    # Join run rows to exact gold pairs; group at production run stmt_hash grain.
    by_stmt: dict[str, dict] = defaultdict(
        lambda: {"rows": [], "tags": [], "source_counts": None, "belief_stored": None}
    )
    n_run = n_joined = n_unmatched = n_invalid_key = 0
    for line in open(args.run):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        n_run += 1
        stmt_key, mh = run_statement_key(d.get("stmt_hash"))
        if mh is None:
            # Same statement identity, TWO encodings: run_rasmachine_monolithic.py
            # persists it as `stmt_hash` (16-char hex); run_vllm_gold_eval.py
            # persists `matches_hash` (already unsigned) and writes NO stmt_hash
            # at all. Without this fallback a provider run joins to NOTHING and
            # this script emitted F1=0.000 with exit 0 — a fabricated
            # measurement, not an absent one. The same repair already lives in
            # scripts/calibration_ship_gate.py:228-233.
            mh = ukey(d.get("matches_hash"))
            if mh is not None:
                stmt_key = str(mh)
        sh = ukey(d.get("source_hash"))
        if stmt_key is None or mh is None or sh is None:
            n_invalid_key += 1
            continue
        pair = (mh, sh)
        pair_gold = gold.by_pair.get(pair)
        g = gold_payload.get(pair)
        if pair_gold is None or g is None:
            n_unmatched += 1
            continue
        n_joined += 1
        s = by_stmt[stmt_key]
        s["rows"].append({
            "source_api": d.get("source_api") or g.get("source_api"),
            "verdict": d.get("verdict"),
            "confidence": d.get("confidence"),
            "tier": d.get("tier"),
            "evidence_text": g.get("evidence_text"),
            "evidence_hash": d.get("evidence_hash"),
        })
        # Pair gold is already a conservative rollup across all curator rows.
        s["tags"].append(pair_gold["verdict"])
        if s["source_counts"] is None:
            counts = d.get("source_counts") or g.get("source_counts")
            if isinstance(counts, dict):
                s["source_counts"] = counts
        if s["belief_stored"] is None:
            # The scored run carries the statement's actual INDRA belief even
            # when a lightweight external gold file omits statement metadata.
            stored = d.get("belief")
            if not isinstance(stored, (int, float)):
                stored = g.get("belief")
            if isinstance(stored, (int, float)):
                s["belief_stored"] = stored

    # per-statement beliefs on the covered set
    reader_configuration = reader_configuration_for_run(args.run, args.model or args.label)
    calib = calibration_for_run(args.run, args.model or args.label)
    stmts: list[dict] = []
    flagged_pairs = []   # statement-grain error detection (positive class = incorrect)
    det_hard = {"tp": 0, "fp": 0}  # deterministic-only hard-flag precision
    n_undefined = n_no_text = n_parse_fail = n_null_source = 0
    n_dedup_groups = n_readable = 0
    for h, s in by_stmt.items():
        statement_gold = aggregate_gold(s["tags"])
        if statement_gold is None:
            continue
        sb = statement_belief(s["rows"], RECALIBRATED_WITH_INDRA_DEFAULTS)
        n_no_text += sb.n_no_text
        n_parse_fail += sb.n_parse_fail
        n_null_source += sb.n_null_source
        n_dedup_groups += sb.n_dedup_groups
        n_readable += sb.n_correct + sb.n_incorrect
        # Production semantics: no readable measurement means undefined belief
        # and review routing, not fabricated support. Exclude that statement from
        # every arm so the head-to-head remains paired.
        if sb.belief is None:
            n_undefined += 1
            continue
        gold_correct = is_gold_correct(statement_gold)
        belief_soft = (statement_belief(
            s["rows"], RECALIBRATED_WITH_INDRA_DEFAULTS, soft=calib
        ).belief if calib else None)
        sc = s["source_counts"] or {}
        stmts.append({
            "stmt_hash": h,
            "depth": sb.n_evidence,
            "gold_correct": gold_correct,
            "belief_llm": sb.belief,
            "belief_llm_soft": belief_soft,
            "belief_recal": compute_edge_reliability_from_counts(
                sc, RECALIBRATED_WITH_INDRA_DEFAULTS
            ) if sc else None,
            "belief_indra": compute_edge_reliability_from_counts(
                sc, INDRA_DEFAULT_PRIORS
            ) if sc else None,
            "belief_stored": s["belief_stored"],
            "verdict_statement": sb.verdict_statement,
        })
        # error detection: flag = verdict_statement != correct ; positive = gold-incorrect
        flag = sb.verdict_statement != "correct"
        flagged_pairs.append((not gold_correct, flag))
        if sb.verdict_statement == "incorrect":  # deterministic hard flag
            if not gold_correct:
                det_hard["tp"] += 1
            else:
                det_hard["fp"] += 1

    singles = [s for s in stmts if s["depth"] == 1]
    multis = [s for s in stmts if s["depth"] > 1]

    keys = ["belief_llm", "belief_llm_soft", "belief_recal", "belief_indra", "belief_stored"]
    if not calib:
        keys.remove("belief_llm_soft")
    disc = {
        k: {"all": discrimination(stmts, k),
            "single_evidence": discrimination(singles, k),
            "multi_evidence": discrimination(multis, k)}
        for k in keys
    }

    ed = confusion_pr(flagged_pairs)
    det_prec = det_hard["tp"] / (det_hard["tp"] + det_hard["fp"]) if (det_hard["tp"] + det_hard["fp"]) else None
    vcounts = defaultdict(int)
    for s in stmts:
        vcounts[s["verdict_statement"]] += 1

    artifact = {
        "schema_version": 1,
        "label": args.label,
        "model": args.model or args.label,
        "gold_source": args.gold,
        "run_source": args.run,
        "input_sha256": {
            "gold": hashlib.sha256(Path(args.gold).read_bytes()).hexdigest(),
            "run": hashlib.sha256(Path(args.run).read_bytes()).hexdigest(),
            "indra_default_priors": INDRA_DEFAULT_PRIORS_SHA256,
        },
        "indra_default_priors": {
            "resource": INDRA_DEFAULT_PRIOR_RESOURCE,
            "sha256": INDRA_DEFAULT_PRIORS_SHA256,
            "n_declared_sources": len(INDRA_DEFAULT_PRIORS.declared_sources),
            "n_complete_sources": len(INDRA_DEFAULT_PRIORS),
            "incomplete_sources": sorted(INDRA_DEFAULT_PRIORS.incomplete_sources),
        },
        "reader_configuration": reader_configuration,
        "calibration": calib,
        "evaluation_basis": {
            "join": "exact unsigned (matches_hash, source_hash); no source-hash fallback",
            "gold_pair": "multi-curator any-incorrect-wins via indra_belief.results.load_gold_map",
            "statement_key": "run stmt_hash",
            "statement_gold": "any-incorrect-wins across exactly joined evidence pairs",
            "statement_gold_note": (
                "conservative evaluation/review proxy; mixed evidence is not literally "
                "repeated measurement of one shared latent statement truth"
            ),
            "calibrated_score": (
                "hybrid log-odds: confusion-derived reader log-LRs plus a separately "
                "fitted source-reliability floor on confirmations; not a pure posterior"
            ),
            "default_priors": (
                "read at import from indra/resources/default_belief_probs.json; "
                "recalibrated sources override that complete installed table"
            ),
            "readability": (
                "indra_belief.statement_belief production de-dup/no_text semantics; "
                "all-unread statements excluded from every arm"
            ),
        },
        "coverage": {"run_rows": n_run, "joined_to_gold": n_joined, "statements": len(stmts),
                     "gold_rows": n_gold_rows, "gold_exact_pairs": len(gold.by_pair),
                     "exact_pair_misses": n_unmatched, "invalid_run_keys": n_invalid_key,
                     "grouped_statements": len(by_stmt),
                     "undefined_statements_excluded": n_undefined,
                     "single_evidence": len(singles), "multi_evidence": len(multis),
                     "gold_correct": sum(1 for s in stmts if s["gold_correct"]),
                     "post_dedup_groups": n_dedup_groups, "readable_evidence": n_readable,
                     "no_text": n_no_text, "parse_fail": n_parse_fail,
                     "null_source": n_null_source},
        "belief_discrimination": disc,
        "statement_error_detection": {
            "flag_rule": "verdict_statement != correct (review or incorrect)",
            "positive_class": "gold-incorrect (conservative any-incorrect-wins rollup)",
            "confusion": ed,
            "deterministic_hard_flag_precision": (round(det_prec, 4) if det_prec is not None else None),
            "verdict_statement_counts": dict(vcounts),
        },
    }
    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    if out_json.exists() and not args.force:
        try:
            prior = json.loads(out_json.read_text())
        except (json.JSONDecodeError, OSError):
            prior = {}
        if prior.get("input_sha256") not in (None, artifact["input_sha256"]):
            raise SystemExit(
                f"refusing to overwrite {out_json} with different input hashes; "
                "choose a new provenance-specific name or pass --force"
            )
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w") as f:
        json.dump(artifact, f, indent=2)

    label = {"belief_llm": f"LLM hard-gate belief ({args.label})",
             "belief_llm_soft": f"LLM hybrid log-odds ({args.label})",
             "belief_recal": "text-miner belief · recalibrated priors",
             "belief_indra": "text-miner belief · INDRA priors",
             "belief_stored": "INDRA stored belief (w/ propagation)"}
    L = [f"# Belief head-to-head — {args.label} vs text-miner baseline\n",
         f"Gold `{args.gold}` · run `{args.run}`  ",
         f"Configuration `{args.model or args.label}`. Exact `(matches_hash, source_hash)` join: "
         f"{n_joined}/{n_run} run rows; grouped by run `stmt_hash` → **{len(stmts)} readable "
         f"statements** ({len(singles)} single, {len(multis)} multi; "
         f"{artifact['coverage']['gold_correct']} gold-correct).  ",
         f"Excluded all-unread statements: {n_undefined}; post-dedup `no_text`: {n_no_text}. "
         "Pair and statement gold both use conservative any-incorrect-wins (an evaluation/"
         "review proxy, not literal latent truth for mixed evidence). "
         + ("The ship-approved calibrated arm is a hybrid of reader log-LRs and a "
            "separately fitted source-reliability floor, not a pure Bayesian posterior.\n"
            if calib else
            "No ship-approved profile matches this exact model+prompt configuration; "
            "the calibrated arm is unavailable and production uses the hard fallback.\n"),
         "## Belief discrimination (statement grain, positive = correct)\n",
         "| belief | subset | n | AUROC | ECE |", "|---|---|--:|--:|--:|"]
    for k in keys:
        for subset in ("all", "single_evidence", "multi_evidence"):
            d = disc[k][subset]
            au = f"{d['auroc']:.3f}" if d["auroc"] is not None else "—"
            ec = f"{d['ece']:.3f}" if d["ece"] is not None else "—"
            L.append(f"| {label[k]} | {subset} | {d['n']} | {au} | {ec} |")
    L.append("\n## Statement-grain error detection (positive = gold-incorrect)\n")
    L.append(f"Flag rule: `verdict_statement != correct`. "
             f"P={ed['p']:.3f} R={ed['r']:.3f} F1={ed['f1']:.3f} "
             f"(tp={ed['tp']} fp={ed['fp']} fn={ed['fn']} tn={ed['tn']}).  ")
    L.append(f"Deterministic hard-flag (`verdict_statement == incorrect`) precision: "
             f"{det_prec:.3f}." if det_prec is not None else "Deterministic hard-flag precision: n/a.")
    L.append(f"verdict_statement counts: {dict(vcounts)}.\n")
    with out_md.open("w") as f:
        f.write("\n".join(L))

    if n_joined == 0:
        raise SystemExit(
            f"[{args.label}] REFUSING to emit a measurement: 0 of {n_run} run rows "
            f"joined to gold ({n_unmatched} unmatched, {n_invalid_key} invalid key). "
            "Every metric below would be computed over an empty set and would print "
            "as 0.000 rather than as absent. Check the run and the gold share a "
            "statement key: monolithic runs carry `stmt_hash`, vLLM gold-eval runs "
            "carry `matches_hash`."
        )

    # console
    print(f"[{args.label}] coverage {n_joined}/{n_run} → {len(stmts)} statements "
          f"(single={len(singles)} multi={len(multis)})\n")
    print(f"{'belief':<42} {'AUROC':>7} {'ECE':>7}")
    for k in keys:
        d = disc[k]["all"]
        au = f"{d['auroc']:.3f}" if d["auroc"] is not None else "—"
        ec = f"{d['ece']:.3f}" if d["ece"] is not None else "—"
        print(f"  {label[k]:<40} {au:>7} {ec:>7}")
    print(f"\nstatement error-detection (flag != correct): "
          f"P={ed['p']:.3f} R={ed['r']:.3f} F1={ed['f1']:.3f}")
    print(f"deterministic hard-flag precision: {det_prec:.3f}" if det_prec is not None else "")
    print(f"verdict_statement counts: {dict(vcounts)}")
    print(f"\nwrote {args.out_json}\nwrote {args.out_md}")


if __name__ == "__main__":
    main()
