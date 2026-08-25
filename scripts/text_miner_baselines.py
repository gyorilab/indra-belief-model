"""Record the text-mining-reader baselines on the balanced curation gold.

This is the number-to-beat for the LLM statement belief ("replace" track): before
we can claim an LLM-gated belief is *better* than INDRA's source-reliability
belief, we record exactly what the text miners deliver on our gold.

Two recorded baselines:

  1. PER-READER RELIABILITY (evidence-pair grain) — for each source_api, the
     fraction of its curated evidence the human gold marks `correct`. This is the
     empirical per-reader accuracy on THIS gold; we line it up against the implied
     single-read accuracy of INDRA's default priors (1 - rand - syst) and the
     recalibrated priors, so we can see whether the assembly-benchmark
     recalibration holds on the curation eval.

  2. BELIEF DISCRIMINATION (statement grain) — INDRA's parametric belief is a
     pure function of source + evidence count (no text read). We roll the gold
     up to statements with the calibration arc's conservative
     any-incorrect-wins rule and measure how well that belief separates gold-correct from
     gold-incorrect statements: AUROC (positive class = correct) and the 8-bin
     ECE. We report the belief as STORED on the gold rows, and as recomputed from
     source_counts under both INDRA's installed defaults and the benchmark
     overrides layered on those defaults.

Outputs a recorded artifact pair: data/results/text_miner_baselines.json (machine)
and reports/text_miner_baselines.md (human). Nothing here reads an LLM verdict —
it is the pre-LLM floor.

    python scripts/text_miner_baselines.py [--gold data/benchmark/eval_curation_v1.jsonl]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from indra_belief.metrics import auroc as _canonical_auroc
from indra_belief.metrics import ece  # noqa: E402
from indra_belief.curation import aggregate_gold, is_gold_correct  # noqa: E402
from indra_belief.noise_model import (  # noqa: E402
    RECALIBRATED_PRIORS,
    _DEFAULT_PRIOR,
    compute_edge_reliability_from_counts,
)
from indra_belief.indra_priors import (  # noqa: E402
    INDRA_DEFAULT_PRIOR_RESOURCE,
    INDRA_DEFAULT_PRIORS,
    INDRA_DEFAULT_PRIORS_SHA256,
    with_benchmark_recalibration,
)

# Text-mining readers (the "text miners"); everything else is a curated DB.
TEXT_MINERS = {"reach", "sparser", "trips", "medscan", "rlimsp", "eidos", "cwms", "isi"}

# Recalibration is an override set, not a replacement source registry.
RECALIBRATED_WITH_INDRA_DEFAULTS = with_benchmark_recalibration(RECALIBRATED_PRIORS)


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


def implied_accuracy(priors: dict[str, tuple[float, float]], src: str) -> float:
    """Single-read accuracy the priors imply for a source: 1 - rand - syst."""
    rand, syst = priors.get(src.lower(), _DEFAULT_PRIOR)
    return 1.0 - rand - syst


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default="data/benchmark/eval_curation_v1.jsonl")
    ap.add_argument("--out-json", default="data/results/text_miner_baselines.json")
    ap.add_argument("--out-md", default="reports/text_miner_baselines.md")
    args = ap.parse_args()

    pairs: list[dict] = []
    with open(args.gold) as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))

    n_pairs = len(pairs)
    n_correct_pairs = sum(1 for p in pairs if p.get("gold") == "correct")

    # ── (1) per-reader reliability at evidence-pair grain ────────────────────
    by_src: dict[str, list[bool]] = defaultdict(list)
    for p in pairs:
        src = (p.get("source_api") or "unknown").lower()
        by_src[src].append(p.get("gold") == "correct")

    per_source = []
    for src in sorted(by_src, key=lambda s: -len(by_src[s])):
        labels = by_src[src]
        n = len(labels)
        acc = sum(labels) / n
        per_source.append({
            "source_api": src,
            "is_text_miner": src in TEXT_MINERS,
            "n_pairs": n,
            "gold_accuracy": round(acc, 4),
            "implied_acc_indra": round(implied_accuracy(INDRA_DEFAULT_PRIORS, src), 4),
            "implied_acc_recalibrated": round(
                implied_accuracy(RECALIBRATED_WITH_INDRA_DEFAULTS, src), 4
            ),
        })

    # ── (2) belief discrimination at statement grain ─────────────────────────
    # Roll pairs up to statements under the same conservative gold used by the
    # calibration/metrics path: one incorrect evidence makes the rollup incorrect.
    by_stmt: dict[str, dict] = defaultdict(lambda: {"tags": [], "belief": None, "source_counts": None})
    for p in pairs:
        h = str(p.get("matches_hash"))
        s = by_stmt[h]
        s["tags"].append(p.get("gold") or p.get("tag"))
        if s["belief"] is None and isinstance(p.get("belief"), (int, float)):
            s["belief"] = p["belief"]
        if s["source_counts"] is None and isinstance(p.get("source_counts"), dict):
            s["source_counts"] = p["source_counts"]

    stmts = []
    for h, s in by_stmt.items():
        n = len(s["tags"])
        gold_correct = is_gold_correct(aggregate_gold(s["tags"]))
        sc = s["source_counts"] or {}
        stmts.append({
            "matches_hash": h,
            "depth": n,
            "gold_correct": gold_correct,
            "belief_stored": s["belief"],
            "belief_indra": compute_edge_reliability_from_counts(
                sc, INDRA_DEFAULT_PRIORS
            ) if sc else None,
            "belief_recal": compute_edge_reliability_from_counts(
                sc, RECALIBRATED_WITH_INDRA_DEFAULTS
            ) if sc else None,
        })

    def discrimination(rows: list[dict], key: str) -> dict:
        scored = [(r[key], r["gold_correct"]) for r in rows if isinstance(r.get(key), (int, float))]
        return {
            "n": len(scored),
            "n_correct": sum(1 for _, lab in scored if lab),
            "auroc": (round(auroc(scored), 4) if auroc(scored) is not None else None),
            "ece": round(ece([(sc, lab) for sc, lab in scored]), 4),
        }

    singles = [s for s in stmts if s["depth"] == 1]
    multis = [s for s in stmts if s["depth"] > 1]

    belief_baselines = {}
    for key in ("belief_stored", "belief_indra", "belief_recal"):
        belief_baselines[key] = {
            "all": discrimination(stmts, key),
            "single_evidence": discrimination(singles, key),
            "multi_evidence": discrimination(multis, key),
        }

    artifact = {
        "gold_source": args.gold,
        "indra_default_priors": {
            "resource": INDRA_DEFAULT_PRIOR_RESOURCE,
            "sha256": INDRA_DEFAULT_PRIORS_SHA256,
            "n_declared_sources": len(INDRA_DEFAULT_PRIORS.declared_sources),
            "n_complete_sources": len(INDRA_DEFAULT_PRIORS),
            "incomplete_sources": sorted(INDRA_DEFAULT_PRIORS.incomplete_sources),
        },
        "evidence_pairs": {
            "n": n_pairs,
            "n_correct": n_correct_pairs,
            "n_incorrect": n_pairs - n_correct_pairs,
            "balance": round(n_correct_pairs / n_pairs, 4),
        },
        "statements": {
            "n": len(stmts),
            "n_gold_correct": sum(1 for s in stmts if s["gold_correct"]),
            "n_single_evidence": len(singles),
            "n_multi_evidence": len(multis),
        },
        "per_source_reliability": per_source,
        "belief_discrimination": belief_baselines,
        "notes": [
            "positive class for AUROC = gold-correct (belief should rank correct > incorrect)",
            "statement gold = conservative any-incorrect-wins over evidence-pair gold",
            "belief_stored = INDRA belief as written on the statement (incl. supports-graph propagation)",
            "belief_indra / belief_recal = recomputed from source_counts (no propagation)",
            "recalibrated entries override installed INDRA defaults; unmeasured sources keep those defaults",
            "this is the pre-LLM floor: no text is read; belief is a function of source + count only",
        ],
    }

    with open(args.out_json, "w") as f:
        json.dump(artifact, f, indent=2)

    # ── human-readable recorded baseline ─────────────────────────────────────
    L = []
    L.append("# Text-miner baselines (pre-LLM floor)\n")
    L.append(f"Gold: `{args.gold}`  \n")
    ep = artifact["evidence_pairs"]
    st = artifact["statements"]
    L.append(f"Evidence pairs: **{ep['n']}** ({ep['n_correct']} correct / {ep['n_incorrect']} incorrect, "
             f"balance {ep['balance']:.1%}).  ")
    L.append(f"Statements (any-incorrect-wins rollup): **{st['n']}** "
             f"({st['n_gold_correct']} gold-correct; {st['n_single_evidence']} single-evidence, "
             f"{st['n_multi_evidence']} multi-evidence).\n")

    L.append("## Per-reader reliability (evidence-pair grain)\n")
    L.append("Empirical gold accuracy vs the single-read accuracy implied by the priors "
             "(`1 − rand − syst`).\n")
    L.append("| source | text-miner | n | gold acc | implied (INDRA) | implied (recal) |")
    L.append("|---|---|--:|--:|--:|--:|")
    for r in per_source:
        L.append(f"| {r['source_api']} | {'yes' if r['is_text_miner'] else ''} | {r['n_pairs']} | "
                 f"{r['gold_accuracy']:.3f} | {r['implied_acc_indra']:.3f} | "
                 f"{r['implied_acc_recalibrated']:.3f} |")

    L.append("\n## Belief discrimination (statement grain)\n")
    L.append("How well a *no-text* parametric belief separates gold-correct from gold-incorrect "
             "statements. AUROC positive class = correct; ECE is the 8-bin scheme. This is the bar "
             "the LLM statement belief must clear.\n")
    label = {"belief_stored": "stored INDRA belief (w/ propagation)",
             "belief_indra": "recomputed · INDRA priors",
             "belief_recal": "recomputed · recalibrated priors"}
    L.append("| belief | subset | n | n correct | AUROC | ECE |")
    L.append("|---|---|--:|--:|--:|--:|")
    for key in ("belief_stored", "belief_indra", "belief_recal"):
        for subset in ("all", "single_evidence", "multi_evidence"):
            d = belief_baselines[key][subset]
            au = f"{d['auroc']:.3f}" if d["auroc"] is not None else "—"
            L.append(f"| {label[key]} | {subset} | {d['n']} | {d['n_correct']} | {au} | {d['ece']:.3f} |")
    L.append("\n> AUROC ≈ 0.5 means the belief carries no signal for correctness on balanced gold — "
             "expected for a count-driven prior, since evidence count does not track truth. "
             "The LLM gate has to beat these numbers to justify replacement.\n")

    with open(args.out_md, "w") as f:
        f.write("\n".join(L))

    # ── console summary ──────────────────────────────────────────────────────
    print(f"gold pairs={ep['n']} balance={ep['balance']:.1%}  statements={st['n']} "
          f"(single={st['n_single_evidence']} multi={st['n_multi_evidence']})")
    print("\nbelief discrimination (statement grain, positive=correct):")
    for key in ("belief_stored", "belief_indra", "belief_recal"):
        d = belief_baselines[key]["all"]
        au = f"{d['auroc']:.3f}" if d["auroc"] is not None else "—"
        print(f"  {label[key]:<38} AUROC={au}  ECE={d['ece']:.3f}  (n={d['n']})")
    print(f"\nwrote {args.out_json}\nwrote {args.out_md}")


if __name__ == "__main__":
    main()
