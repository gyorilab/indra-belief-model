"""Head-to-head: LLM statement belief vs the text-miner baseline, on gold.

The "replace" make-or-break. For every statement covered by both the gold and a
model run, we compute:
  - belief_llm   = statement_belief() — the LLM-gated noisy-OR (recalibrated priors)
  - belief_recal = INDRA parametric belief recomputed from source_counts (no text read)
  - belief_indra = same, under INDRA default priors
  - belief_stored = INDRA belief as written on the statement (incl. propagation)
and score each against the majority-vote statement gold: AUROC (positive=correct)
and 8-bin ECE, overall and split by evidence depth. Everything is on the SAME
covered statement set so the comparison is paired. We also report the tiered
verdict_statement's error-detection at statement grain.

    python scripts/belief_headtohead.py \
        --gold data/benchmark/eval_curation_v1.jsonl \
        --run  data/results/eval_curation_v1_gemma.jsonl \
        --label gemma-26b
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from indra_belief.metrics import confusion_pr, ece  # noqa: E402
from indra_belief.noise_model import (  # noqa: E402
    INDRA_PRIORS,
    RECALIBRATED_PRIORS,
    compute_edge_reliability_from_counts,
)
from indra_belief.statement_belief import statement_belief  # noqa: E402
from indra_belief.calibration_constants import calibration_for  # noqa: E402
from indra_belief.curation import is_gold_correct  # noqa: E402

_HASH_MASK = (1 << 64) - 1


def ukey(x):
    try:
        return int(x) & _HASH_MASK
    except (ValueError, TypeError):
        return None


def auroc(scored: list[tuple[float, bool]]) -> float | None:
    pos = sum(1 for _, lab in scored if lab)
    neg = len(scored) - pos
    if not pos or not neg:
        return None
    ordered = sorted(scored, key=lambda x: x[0])
    ranks = [0.0] * len(ordered)
    i = 0
    while i < len(ordered):
        j = i
        while j + 1 < len(ordered) and ordered[j + 1][0] == ordered[i][0]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[k] = avg
        i = j + 1
    rank_sum_pos = sum(r for r, (_, lab) in zip(ranks, ordered) if lab)
    return (rank_sum_pos - pos * (pos + 1) / 2.0) / (pos * neg)


def discrimination(rows: list[dict], key: str) -> dict:
    scored = [(r[key], r["gold_correct"]) for r in rows if isinstance(r.get(key), (int, float))]
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
    ap.add_argument("--model", default=None, help="reader model name for soft-weight lookup (default: --label)")
    ap.add_argument("--out-json", default="data/results/belief_headtohead.json")
    ap.add_argument("--out-md", default="reports/belief_headtohead.md")
    args = ap.parse_args()

    # gold indexed by source_hash (the evidence-pair join key)
    gold: dict[int, dict] = {}
    for line in open(args.gold):
        line = line.strip()
        if not line:
            continue
        g = json.loads(line)
        k = ukey(g.get("source_hash"))
        if k is not None:
            gold[k] = g

    # join run rows to gold; group per statement (matches_hash)
    by_stmt: dict[str, dict] = defaultdict(
        lambda: {"rows": [], "golds": [], "source_counts": None, "belief_stored": None}
    )
    n_run = n_joined = 0
    for line in open(args.run):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        n_run += 1
        g = gold.get(ukey(d.get("source_hash")))
        if g is None:
            continue
        n_joined += 1
        h = str(g.get("matches_hash"))
        s = by_stmt[h]
        s["rows"].append({
            "source_api": d.get("source_api") or g.get("source_api"),
            "verdict": d.get("verdict"),
            "confidence": d.get("confidence"),
            "tier": d.get("tier"),
            "evidence_text": g.get("evidence_text"),
            "evidence_hash": d.get("evidence_hash"),
        })
        # Canonical gold field is `tag`; some eval sets also carry a `gold` alias.
        # Use the shared predicate so this is portable across datasets (holdout_cc
        # has `tag` but no `gold` — reading `gold` alone labels everything incorrect).
        s["golds"].append(is_gold_correct(g.get("gold") or g.get("tag")))
        if s["source_counts"] is None and isinstance(g.get("source_counts"), dict):
            s["source_counts"] = g["source_counts"]
        if s["belief_stored"] is None and isinstance(g.get("belief"), (int, float)):
            s["belief_stored"] = g["belief"]

    # per-statement beliefs on the covered set
    calib = calibration_for(args.model or args.label)
    stmts: list[dict] = []
    flagged_pairs = []   # statement-grain error detection (positive class = incorrect)
    det_hard = {"tp": 0, "fp": 0}  # deterministic-only hard-flag precision
    for h, s in by_stmt.items():
        n = len(s["golds"])
        gold_correct = sum(s["golds"]) * 2 > n
        sb = statement_belief(s["rows"], RECALIBRATED_PRIORS)
        belief_soft = (statement_belief(s["rows"], RECALIBRATED_PRIORS, soft=calib).belief
                       if calib else None)
        sc = s["source_counts"] or {}
        stmts.append({
            "matches_hash": h,
            "depth": n,
            "gold_correct": gold_correct,
            "belief_llm": sb.belief,
            "belief_llm_soft": belief_soft,
            "belief_recal": compute_edge_reliability_from_counts(sc, RECALIBRATED_PRIORS) if sc else None,
            "belief_indra": compute_edge_reliability_from_counts(sc, INDRA_PRIORS) if sc else None,
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
        "label": args.label,
        "gold_source": args.gold,
        "run_source": args.run,
        "calibration": calib,
        "coverage": {"run_rows": n_run, "joined_to_gold": n_joined, "statements": len(stmts),
                     "single_evidence": len(singles), "multi_evidence": len(multis),
                     "gold_correct": sum(1 for s in stmts if s["gold_correct"])},
        "belief_discrimination": disc,
        "statement_error_detection": {
            "flag_rule": "verdict_statement != correct (review or incorrect)",
            "positive_class": "gold-incorrect (majority vote)",
            "confusion": ed,
            "deterministic_hard_flag_precision": (round(det_prec, 4) if det_prec is not None else None),
            "verdict_statement_counts": dict(vcounts),
        },
    }
    with open(args.out_json, "w") as f:
        json.dump(artifact, f, indent=2)

    label = {"belief_llm": f"LLM gated belief ({args.label})",
             "belief_llm_soft": f"LLM gated + calibrated soft ({args.label})",
             "belief_recal": "text-miner belief · recalibrated priors",
             "belief_indra": "text-miner belief · INDRA priors",
             "belief_stored": "INDRA stored belief (w/ propagation)"}
    L = [f"# Belief head-to-head — {args.label} vs text-miner baseline\n",
         f"Gold `{args.gold}` · run `{args.run}`  ",
         f"Coverage: {n_joined}/{n_run} run rows joined to gold → **{len(stmts)} statements** "
         f"({len(singles)} single, {len(multis)} multi; {artifact['coverage']['gold_correct']} gold-correct).\n",
         "## Belief discrimination (statement grain, positive = correct)\n",
         "| belief | subset | n | AUROC | ECE |", "|---|---|--:|--:|--:|"]
    for k in keys:
        for subset in ("all", "single_evidence", "multi_evidence"):
            d = disc[k][subset]
            au = f"{d['auroc']:.3f}" if d["auroc"] is not None else "—"
            L.append(f"| {label[k]} | {subset} | {d['n']} | {au} | {d['ece']:.3f} |")
    L.append("\n## Statement-grain error detection (positive = gold-incorrect)\n")
    L.append(f"Flag rule: `verdict_statement != correct`. "
             f"P={ed['p']:.3f} R={ed['r']:.3f} F1={ed['f1']:.3f} "
             f"(tp={ed['tp']} fp={ed['fp']} fn={ed['fn']} tn={ed['tn']}).  ")
    L.append(f"Deterministic hard-flag (`verdict_statement == incorrect`) precision: "
             f"{det_prec:.3f}." if det_prec is not None else "Deterministic hard-flag precision: n/a.")
    L.append(f"verdict_statement counts: {dict(vcounts)}.\n")
    with open(args.out_md, "w") as f:
        f.write("\n".join(L))

    # console
    print(f"[{args.label}] coverage {n_joined}/{n_run} → {len(stmts)} statements "
          f"(single={len(singles)} multi={len(multis)})\n")
    print(f"{'belief':<42} {'AUROC':>7} {'ECE':>7}")
    for k in keys:
        d = disc[k]["all"]
        au = f"{d['auroc']:.3f}" if d["auroc"] is not None else "—"
        print(f"  {label[k]:<40} {au:>7} {d['ece']:>7.3f}")
    print(f"\nstatement error-detection (flag != correct): "
          f"P={ed['p']:.3f} R={ed['r']:.3f} F1={ed['f1']:.3f}")
    print(f"deterministic hard-flag precision: {det_prec:.3f}" if det_prec is not None else "")
    print(f"verdict_statement counts: {dict(vcounts)}")
    print(f"\nwrote {args.out_json}\nwrote {args.out_md}")


if __name__ == "__main__":
    main()
