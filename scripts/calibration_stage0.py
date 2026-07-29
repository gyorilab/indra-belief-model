"""Historical Calibration Stage C0 — reliability curve, zero model change.

The decision gate (G0) for the whole calibration arc. See
``research/calibration_task_hypergraph.md``. Touches NO production code path —
it joins the existing eval_curation_v1 run outputs to gold and asks three
questions:

  C0.1  Verdict-conditional error anchors per reader (rand_corr / rand_rej) —
        the two parameters Stage C1 would fit. Confidence axis is reported but
        NOT fit (it has collapsed; see the printed mix).
  C0.2  Tier-1 per-EVIDENCE reliability of the persisted grid ``score`` as a
        forecast of ``gold == correct`` — reliability bins + ECE + Brier
        (Murphy decomposition). Expected to be degenerate (~2 occupied bins)
        because the grid score is near-binary once confidence collapses.
  C0.3  Tier-2 per-STATEMENT raw-belief headroom — the DECISIVE test. Aggregate
        the per-evidence verdicts through the production noise model
        (``compute_gated_belief`` + RECALIBRATED_PRIORS, hard gate
        ``included = verdict != 'incorrect'``) to a per-statement belief, then
        measure AUROC / AUPRC vs per-statement gold (any-incorrect-wins).

G0 read: if the per-statement belief AUPRC has no headroom over the base rate
(and AUROC ~ 0.5), a monotone post-hoc map cannot manufacture discrimination —
the lever is upstream (reader / grounding), and Stages C1-C3 should not fire.

Join is the canonical (matches_hash, source_hash) PAIR used by
eval_curation_compare.py (source_hash alone is not unique — version-skew lesson).
This retained baseline groups Tier-2 by the historical gold ``pa_hash`` grain;
the current production/formal surfaces use run ``stmt_hash`` and live in
``results.py`` / ``calibration_ship_gate.py``.

    PYTHONPATH=src python scripts/calibration_stage0.py \
        --gold data/benchmark/eval_curation_v1.jsonl \
        --run data/results/eval_curation_v1_medpsy.jsonl --name MedPsy-4B \
        --run data/results/eval_curation_v1_gemma.jsonl  --name gemma-26B \
        --out data/results/calibration_stage0.md \
        --json data/results/calibration_stage0.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from indra_belief.curation import aggregate_gold, is_gold_correct  # noqa: E402
from indra_belief.metrics import (  # noqa: E402
    BINS_8,
    auroc,
    auprc,
    brier_murphy,
    ece,
    reliability_bins,
)
from indra_belief.noise_model import (  # noqa: E402
    RECALIBRATED_PRIORS,
    compute_gated_belief,
)

MASK = (1 << 64) - 1


def umask(x) -> int:
    return int(x) & MASK


def load_jsonl(p: str | Path) -> list[dict]:
    return [json.loads(l) for l in open(p) if l.strip()]


# ---- gold-join trio: parallels eval_curation_compare.py, DELIBERATELY DIVERGES
# This build_gold_index / gold_for / join_model trio is NOT a lazy fork of
# eval_curation_compare's same-named trio and must not be unified into it. Its
# Tier-2 per-statement path needs three behaviors the leaner compare trio
# intentionally omits:
#   1. any-incorrect-wins multi-curator collapse (aggregate_gold) in
#      _collapse_gold_rows / build_gold_index (compare keeps last-write-wins,
#      raw rows in by_sh);
#   2. a PERMISSIVE source-only fallback in gold_for — fire whenever every gold
#      row sharing the source_hash agrees on truth class (len(classes) == 1);
#      compare instead requires exactly one candidate (len(cand) == 1);
#   3. pair-dedup in join_model that RAISES on conflicting scored verdicts for a
#      duplicated pair (compare does no dedup).
# The two build_gold_index collapses were MEASURED byte-equal on both real golds
# today (collapse_diff_pairs=0 on eval_curation_v1 n=1606 + external_curator_gold
# _v1 n=578), so the divergence is a latent-drift guard, not an active mismatch.
# -----------------------------------------------------------------------------
def _collapse_gold_rows(rows: list[dict]) -> dict | None:
    """Collapse multi-curator rows without a last-write-wins label.

    The canonical conservative rule is any-incorrect-wins. The representative
    row retains the original provenance fields and carries ``all_tags`` so a
    caller can still see that the pair was multiply curated.
    """
    tags = [r.get("tag") or r.get("gold") for r in rows]
    tags = [tag for tag in tags if tag is not None]
    verdict = aggregate_gold(tags)
    if verdict is None:
        return None
    out = dict(rows[0])
    out["tag"] = verdict
    if any("gold" in r for r in rows):
        out["gold"] = verdict
    out["all_tags"] = tags
    out["n_gold_rows"] = len(rows)
    return out


def build_gold_index(gold_rows: list[dict]):
    grouped: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for r in gold_rows:
        grouped[(umask(r["matches_hash"]), umask(r["source_hash"]))].append(r)

    by_pair: dict[tuple[int, int], dict] = {}
    by_sh: dict[int, list[dict]] = defaultdict(list)
    for pair, rows in grouped.items():
        collapsed = _collapse_gold_rows(rows)
        if collapsed is None:
            continue
        by_pair[pair] = collapsed
        by_sh[pair[1]].append(collapsed)
    return by_pair, by_sh


def gold_for(scored: dict, by_pair, by_sh) -> dict | None:
    sh = umask(scored["source_hash"])
    stmt_hash_hex = scored.get("stmt_hash")
    mh = int(stmt_hash_hex, 16) if stmt_hash_hex else None
    if mh is not None and (mh, sh) in by_pair:
        return by_pair[(mh, sh)]
    cand = by_sh.get(sh, [])
    if not cand:
        return None
    # Source-only fallback is allowed only when every statement context sharing
    # the source hash has one truth class. Conflicting contexts are ambiguous.
    classes = {is_gold_correct(r["tag"]) for r in cand}
    return cand[0] if len(classes) == 1 else None


def join_model(scored_rows, by_pair, by_sh):
    joined, parse_null, missed = [], 0, 0
    seen_pairs: dict[tuple[int | None, int], str | None] = {}
    for s in scored_rows:
        g = gold_for(s, by_pair, by_sh)
        if g is None:
            missed += 1
            continue
        sh = umask(s["source_hash"])
        stmt_hash_hex = s.get("stmt_hash")
        mh = int(stmt_hash_hex, 16) if stmt_hash_hex else g.get("matches_hash")
        pair = (umask(mh) if mh is not None else None, sh)
        verdict = s.get("verdict")
        if pair in seen_pairs:
            if seen_pairs[pair] != verdict:
                raise ValueError(
                    f"conflicting scored verdicts for duplicate pair {pair}: "
                    f"{seen_pairs[pair]!r} vs {verdict!r}"
                )
            continue
        seen_pairs[pair] = verdict
        if s.get("verdict") is None:
            parse_null += 1
            continue
        joined.append((g, s))
    return joined, parse_null, missed


# ---- AUROC / AUPRC / Brier / reliability_bins now live in indra_belief.metrics
#      (lifted there so results.py + the calibration scripts share ONE definition;
#      re-imported above). They are re-exported as module attributes so downstream
#      `c0.auroc` / `c0.brier_murphy` (calibration_stage1, calibration_ship_gate)
#      keep working unchanged.


# ---- per-statement aggregation (Tier-2) -------------------------------------
def per_statement_belief(joined) -> list[dict]:
    """Aggregate joined (gold, scored) rows to one row per statement.

    belief = compute_gated_belief over the statement's scored evidences with
    the production hard gate (included = verdict != 'incorrect') and
    RECALIBRATED_PRIORS. gold = any-incorrect-wins over the statement's tags.
    """
    # Group into statements by pa_hash (the preassembled-statement unit the rest
    # of the calibration arc uses, C2.4) — not matches_hash, which is finer.
    by_stmt: dict[int, dict] = defaultdict(lambda: {"ev": [], "tags": [], "indra_belief": None})
    for g, s in joined:
        mh = g.get("pa_hash") or g.get("matches_hash")
        rec = by_stmt[mh]
        rec["ev"].append({
            "source_api": s.get("source_api") or g.get("source_api"),
            "included": s.get("verdict") != "incorrect",
        })
        rec["tags"].append(g["tag"])
        if rec["indra_belief"] is None:
            rec["indra_belief"] = s.get("belief")
    rows = []
    for mh, rec in by_stmt.items():
        gold = aggregate_gold(rec["tags"])
        if gold is None:
            continue
        res = compute_gated_belief(rec["ev"], priors=RECALIBRATED_PRIORS)
        rows.append({
            "pa_hash": mh,
            "belief": res.belief,
            "parametric_only": res.parametric_only,
            "indra_belief": rec["indra_belief"],
            "n_ev": len(rec["ev"]),
            "gold_correct": is_gold_correct(gold),
        })
    return rows


# ---- per-model analysis ------------------------------------------------------
def analyze(name: str, joined, parse_null, missed) -> dict:
    # C0.1 — verdict-conditional anchors
    cells = defaultdict(int)  # (verdict, gold_correct) -> n
    conf = defaultdict(int)
    for g, s in joined:
        gc = is_gold_correct(g["tag"])
        cells[(s["verdict"], gc)] += 1
        conf[s.get("confidence")] += 1
    cc = cells[("correct", True)]
    ci = cells[("correct", False)]
    ic = cells[("incorrect", True)]
    ii = cells[("incorrect", False)]
    rand_corr = ci / (ci + cc) if (ci + cc) else float("nan")   # P(gold=incorrect|verdict=correct)
    rand_rej = ic / (ic + ii) if (ic + ii) else float("nan")    # P(gold=correct|verdict=incorrect)

    # C0.2 — Tier-1 per-evidence reliability of the grid score
    ev_pairs = [(s.get("score") if s.get("score") is not None else 0.5,
                 is_gold_correct(g["tag"])) for g, s in joined]
    ev_scores = [p for p, _ in ev_pairs]
    ev_labels = [y for _, y in ev_pairs]
    t1 = {
        "ece": ece(ev_pairs),
        "auroc": auroc(ev_scores, ev_labels),
        "auprc_correct": auprc(ev_scores, ev_labels),
        "base_rate_correct": float(np.mean(ev_labels)),
        "brier": brier_murphy(ev_scores, ev_labels),
        "bins": reliability_bins(ev_scores, ev_labels),
        "n": len(ev_pairs),
    }

    # C0.3 — Tier-2 per-statement raw-belief headroom (DECISIVE)
    stmt = per_statement_belief(joined)
    bel = [r["belief"] for r in stmt]
    lab = [r["gold_correct"] for r in stmt]
    indra = [r["indra_belief"] for r in stmt if r["indra_belief"] is not None]
    indra_lab = [r["gold_correct"] for r in stmt if r["indra_belief"] is not None]
    singles = [r for r in stmt if r["n_ev"] == 1]
    multi = [r for r in stmt if r["n_ev"] > 1]

    def headroom(scores, labels):
        if not scores:
            return None
        labels_b = np.asarray(labels, bool)
        return {
            "n": len(scores),
            "base_rate_correct": float(labels_b.mean()),
            "auroc": auroc(scores, labels),
            "auprc_correct": auprc(scores, labels),
            "ece": ece(list(zip(scores, [bool(x) for x in labels]))),
            "brier": brier_murphy(scores, labels),
            "mean_belief_when_correct": float(np.mean([s for s, y in zip(scores, labels) if y])) if any(labels) else None,
            "mean_belief_when_incorrect": float(np.mean([s for s, y in zip(scores, labels) if not y])) if not all(labels) else None,
        }

    t2 = {
        "all": headroom(bel, lab),
        "bins": reliability_bins(bel, lab),
        "singletons": headroom([r["belief"] for r in singles], [r["gold_correct"] for r in singles]),
        "multi_evidence": headroom([r["belief"] for r in multi], [r["gold_correct"] for r in multi]),
        "indra_prior_reference": headroom(indra, indra_lab),
        "n_statements": len(stmt),
        "n_singletons": len(singles),
        "n_multi": len(multi),
    }

    # G0 verdict per model: headroom = AUROC meaningfully > 0.5 AND AUPRC > base rate
    a = t2["all"]
    headroom_auroc = (a["auroc"] - 0.5) if a else float("nan")
    headroom_auprc = (a["auprc_correct"] - a["base_rate_correct"]) if a else float("nan")
    go = bool(a and headroom_auroc > 0.03 and headroom_auprc > 0.02)

    return {
        "name": name,
        "joined": len(joined), "parse_null": parse_null, "missed": missed,
        "confidence_mix": dict(conf),
        "anchors": {
            "cells": {"correct_correct": cc, "correct_incorrect": ci,
                      "incorrect_correct": ic, "incorrect_incorrect": ii},
            "rand_corr": rand_corr, "rand_rej": rand_rej,
        },
        "tier1_per_evidence": t1,
        "tier2_per_statement": t2,
        "g0": {"headroom_auroc": headroom_auroc, "headroom_auprc": headroom_auprc, "go": go},
    }


# ---- rendering ---------------------------------------------------------------
def _fmt(x, nd=3):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x:.{nd}f}"


def reliability_ascii(bins, width=24) -> list[str]:
    """One row per occupied bin: range, n, predicted vs empirical on a 0..1 axis."""
    lines = ["    bin            n     pred   emp    0" + " " * (width - 2) + "1"]
    for b in bins:
        if not b["n"]:
            continue
        pred, emp = b["mean_pred"], b["empirical"]
        row = [" "] * width
        pp = min(width - 1, int(round(pred * (width - 1))))
        ee = min(width - 1, int(round(emp * (width - 1))))
        row[pp] = "P"
        row[ee] = "E" if ee != pp else "X"
        lines.append(f"    [{b['lo']:.2f},{b['hi']:.2f}) {b['n']:>5}  {pred:.3f}  {emp:.3f}  |{''.join(row)}|")
    return lines


def render_md(results: list[dict], args) -> str:
    L = []
    e = L.append
    e("# Historical Calibration Stage C0 — reliability curve (zero model change)")
    e("")
    e("Diagnostic for **G0** (go/no-go). No production code path touched. "
      "Per-evidence join on the canonical (matches_hash, source_hash) pair; "
      "per-statement belief = `compute_gated_belief` over scored evidences with "
      "the production hard gate (`included = verdict != incorrect`) + "
      "RECALIBRATED_PRIORS; per-statement gold = any-incorrect-wins. This retained "
      "C0 artifact uses the historical gold `pa_hash` statement grain; current "
      "production/formal metrics use run `stmt_hash` in `results.py` and "
      "`calibration_ship_gate.py`. "
      "Generated by `scripts/calibration_stage0.py`.")
    e("")
    e("## G0 — go/no-go (does per-statement raw belief have headroom?)")
    e("")
    e("| reader | stmts | base rate | belief AUROC | Δ vs 0.5 | belief AUPRC | Δ vs base | verdict |")
    e("|---|---|---|---|---|---|---|---|")
    for r in results:
        a = r["tier2_per_statement"]["all"]
        g = r["g0"]
        verdict = "**GO**" if g["go"] else "**NO-GO**"
        e(f"| {r['name']} | {a['n']} | {_fmt(a['base_rate_correct'])} | {_fmt(a['auroc'])} | "
          f"{_fmt(g['headroom_auroc'],3)} | {_fmt(a['auprc_correct'])} | {_fmt(g['headroom_auprc'],3)} | {verdict} |")
    e("")
    e("> GO criterion: AUROC − 0.5 > 0.03 AND AUPRC − base_rate > 0.02. A NO-GO means a "
      "monotone post-hoc map cannot manufacture discrimination — the lever is upstream "
      "(reader / grounding), and Stages C1–C3 should not fire (D8).")
    e("")

    for r in results:
        e(f"## {r['name']}")
        e("")
        e(f"Joined {r['joined']} evidence rows (parse-null {r['parse_null']}, unmatched {r['missed']}). "
          f"Confidence mix: {r['confidence_mix']} — **degenerate; not fit (C0.1)**.")
        e("")
        an = r["anchors"]; c = an["cells"]
        e("### C0.1 — verdict-conditional anchors (the two params C1 would fit)")
        e("")
        e(f"- confusion (verdict × gold): correct/correct {c['correct_correct']}, "
          f"correct/incorrect {c['correct_incorrect']}, incorrect/correct {c['incorrect_correct']}, "
          f"incorrect/incorrect {c['incorrect_incorrect']}")
        e(f"- **rand_corr** = P(gold=incorrect | verdict=correct) = **{_fmt(an['rand_corr'])}** "
          f"(a confirmed read is wrong this often)")
        e(f"- **rand_rej** = P(gold=correct | verdict=incorrect) = **{_fmt(an['rand_rej'])}** "
          f"(a rejected read is wrong this often)")
        e("")
        t1 = r["tier1_per_evidence"]
        e("### C0.2 — Tier-1 per-evidence reliability of the grid `score`")
        e("")
        e(f"- ECE {_fmt(t1['ece'])} · Brier {_fmt(t1['brier']['brier'])} "
          f"(reliability {_fmt(t1['brier']['reliability'])} − resolution {_fmt(t1['brier']['resolution'])} "
          f"+ uncertainty {_fmt(t1['brier']['uncertainty'])}) · AUROC {_fmt(t1['auroc'])} · "
          f"AUPRC(correct) {_fmt(t1['auprc_correct'])} (base {_fmt(t1['base_rate_correct'])}) · n {t1['n']}")
        e("")
        e("```")
        L.extend(reliability_ascii(t1["bins"]))
        e("```")
        e("")
        t2 = r["tier2_per_statement"]
        a = t2["all"]
        e("### C0.3 — Tier-2 per-statement raw-belief (DECISIVE)")
        e("")
        e(f"- statements {t2['n_statements']} ({t2['n_singletons']} singletons, {t2['n_multi']} multi-evidence)")
        e(f"- **AUROC {_fmt(a['auroc'])} · AUPRC(correct) {_fmt(a['auprc_correct'])}** "
          f"(base {_fmt(a['base_rate_correct'])}) · ECE {_fmt(a['ece'])} · Brier {_fmt(a['brier']['brier'])} "
          f"(resolution {_fmt(a['brier']['resolution'])})")
        e(f"- saturation check: mean belief when correct {_fmt(a['mean_belief_when_correct'])} vs "
          f"when incorrect {_fmt(a['mean_belief_when_incorrect'])}")
        for sub in ("singletons", "multi_evidence", "indra_prior_reference"):
            h = t2[sub]
            if h:
                e(f"  - {sub}: n {h['n']}, AUROC {_fmt(h['auroc'])}, AUPRC {_fmt(h['auprc_correct'])} "
                  f"(base {_fmt(h['base_rate_correct'])}), ECE {_fmt(h['ece'])}")
        e("")
        e("```")
        L.extend(reliability_ascii(t2["bins"]))
        e("```")
        e("")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default="data/benchmark/eval_curation_v1.jsonl")
    ap.add_argument("--run", action="append", default=[], help="run output jsonl (repeatable)")
    ap.add_argument("--name", action="append", default=[], help="display name per --run (repeatable)")
    ap.add_argument("--out", default="data/results/calibration_stage0.md")
    ap.add_argument("--json", default="data/results/calibration_stage0.json")
    args = ap.parse_args()

    if not args.run:
        args.run = ["data/results/eval_curation_v1_medpsy.jsonl",
                    "data/results/eval_curation_v1_gemma.jsonl"]
        args.name = ["MedPsy-4B", "gemma-26B"]
    names = args.name + [f"run{i}" for i in range(len(args.name), len(args.run))]

    gold = load_jsonl(ROOT / args.gold)
    by_pair, by_sh = build_gold_index(gold)

    results = []
    for path, name in zip(args.run, names):
        rows = load_jsonl(ROOT / path)
        joined, pn, miss = join_model(rows, by_pair, by_sh)
        results.append(analyze(name, joined, pn, miss))

    md = render_md(results, args)
    (ROOT / args.out).write_text(md)
    (ROOT / args.json).write_text(json.dumps(results, indent=2, default=float))
    print(md)
    print(f"\nWrote {args.out} and {args.json}")


if __name__ == "__main__":
    main()
