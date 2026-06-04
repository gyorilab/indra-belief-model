#!/usr/bin/env python3
"""Freeze a stratified queue of inter-model VERDICT DISAGREEMENTS for adjudication.

Given two run exports of the same corpus (e.g. gemma vs medpsy), find the
(statement, evidence) rows where the two models gave *opposite* verdicts, and
draw a stratified, cluster-controlled sample to adjudicate by hand. The question
the sample answers: when the models disagree, which one does a blinded human
side with — i.e. is the more-lenient model right, or rubber-stamping?

Design (mirrors build_review_queue):
  * Stratify by DIRECTION (a_correct_b_incorrect vs a_incorrect_b_correct) — the
    headline asymmetry — equal-n per direction so each gets a usable CI, with
    Horvitz-Thompson weights frozen in for a population-reweighted overall.
  * Two-stage: statements are the PSU; <= --cap evidence rows per statement,
    GLOBALLY, so a mega-Complex can't dominate the disagreement sample.
  * bucket / stmt_type / source_api are recorded as cross-cuts, not allocators.

The queue carries each model's verdict (for stratification + later analysis) but
the /adjudicate loader never sends them to the client before the human commits
their own blinded verdict.

Usage::

    python scripts/build_disagreement_queue.py \
        --run-a data/exports/rasmachine_belief \
        --run-b data/exports/6aeedd3b76c74f06817b44353c8e91a8
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from indra_belief.curation import CurationIndex, load_index  # noqa: E402
from indra_belief.sampling import (  # noqa: E402
    item_id as _make_item_id,
    two_stage_sample,
    wilson_halfwidth as _wilson_halfwidth,
)

DIRECTIONS = ["a_correct_b_incorrect", "a_incorrect_b_correct"]

DEFAULT_CURATIONS = "data/benchmark/rasmachine_curations.jsonl"


def _load(export_dir):
    with open(os.path.join(export_dir, "per_evidence.jsonl")) as f:
        rows = {(r["stmt_hash"], r["evidence_hash"]): r
                for r in (json.loads(line) for line in f if line.strip())}
    with open(os.path.join(export_dir, "export_meta.json")) as f:
        meta = json.load(f)
    return rows, meta


def _is_curated(row, gold: CurationIndex):
    """Does this export row carry an INDRA curation? Delegates the hex->int
    bridge to the canonical curation module."""
    return gold.is_curated(row.get("indra_matches_hash"), row.get("source_hash"))


def _item_id(ra, rb, sh, eh):
    return _make_item_id(ra, rb, sh, eh)


# curated-first priority: scarce INDRA-curated disagreements are drawn first and
# cap-exempt so every gradable-against-gold row enters the queue.
def _curated_priority(r):
    return bool(r.get("_curated"))


def build(args):
    A, meta_a = _load(args.run_a)
    B, meta_b = _load(args.run_b)
    run_a, run_b = meta_a.get("run_id"), meta_b.get("run_id")
    model_a, model_b = meta_a.get("model"), meta_b.get("model")
    gold = load_index(args.curations)
    rng = random.Random(args.seed)

    # disagreements: both scored, opposite verdicts
    disagreements = []
    for k in set(A) & set(B):
        va, vb = A[k]["verdict"], B[k]["verdict"]
        if va in ("correct", "incorrect") and vb in ("correct", "incorrect") and va != vb:
            direction = "a_correct_b_incorrect" if va == "correct" else "a_incorrect_b_correct"
            disagreements.append({
                "stmt_hash": k[0], "evidence_hash": k[1],
                "stmt_type": A[k].get("stmt_type"), "source_api": A[k].get("source_api"),
                "bucket_a": A[k].get("bucket"), "bucket_b": B[k].get("bucket"),
                "verdict_a": va, "verdict_b": vb, "direction": direction,
                # curation-priority flag (bridge hex->int via run-A row)
                "_curated": _is_curated(A[k], gold),
            })

    pop = Counter(d["direction"] for d in disagreements)
    curated_pop = Counter(d["direction"] for d in disagreements if d["_curated"])
    total = sum(pop.values())
    weights = {dirn: pop[dirn] / total for dirn in pop}

    by_dir = defaultdict(list)
    for d in disagreements:
        by_dir[d["direction"]].append(d)

    taken_per_stmt = defaultdict(int)
    queue = []
    achieved = Counter()
    for dirn in DIRECTIONS:
        picked = two_stage_sample(by_dir.get(dirn, []), args.n_per_direction, args.cap,
                                   taken_per_stmt, rng, priority=_curated_priority)
        for d in picked:
            queue.append({
                "item_id": _item_id(run_a, run_b, d["stmt_hash"], d["evidence_hash"]),
                "stmt_hash": d["stmt_hash"], "evidence_hash": d["evidence_hash"],
                "run_a": run_a, "run_b": run_b, "model_a": model_a, "model_b": model_b,
                "stratum": f"dir:{dirn}",
                "direction": dirn, "stratum_weight": weights.get(dirn),
                "bucket_a": d["bucket_a"], "stmt_type": d["stmt_type"], "source_api": d["source_api"],
                # model verdicts — for stratification + analysis; NOT shown pre-commit
                "verdict_a": d["verdict_a"], "verdict_b": d["verdict_b"],
                # has an INDRA curation → blinded verdict is gradable against gold
                "curated": bool(d.get("_curated")),
            })
        achieved[dirn] = len(picked)
    achieved_curated = Counter(it["direction"] for it in queue if it.get("curated"))

    # annotator + double-label assignment
    rng.shuffle(queue)
    n_double = round(args.double_frac * len(queue)) if args.double_annotator else 0
    for i, it in enumerate(queue):
        primary = args.annotators[i % len(args.annotators)]
        anns = [primary]
        it["double"] = i < n_double
        if it["double"] and args.double_annotator and args.double_annotator != primary:
            anns.append(args.double_annotator)
        it["annotators"] = anns
    queue.sort(key=lambda it: it["item_id"])

    n_stmts = len({it["stmt_hash"] for it in queue})
    meta = {
        "kind": "disagreement",
        "seed": args.seed,
        "run_a": run_a, "run_b": run_b, "model_a": model_a, "model_b": model_b,
        "export_a": args.run_a, "export_b": args.run_b,
        "params": {"n_per_direction": args.n_per_direction, "cap": args.cap,
                   "annotators": args.annotators, "double_annotator": args.double_annotator,
                   "double_frac": args.double_frac},
        "population_disagreements": dict(pop),
        "curated_population": dict(curated_pop),
        "ht_weights": weights,
        "achieved_per_direction": dict(achieved),
        "achieved_curated_per_direction": dict(achieved_curated),
        "curations_file": args.curations,
        "totals": {"items": len(queue), "distinct_statements": n_stmts,
                   "max_items_per_statement": max(Counter(it["stmt_hash"] for it in queue).values()) if queue else 0,
                   "double_labeled": n_double,
                   "curated_in_queue": sum(1 for it in queue if it.get("curated")),
                   "curated_population_total": sum(curated_pop.values())},
        "estimand_note": (
            "Blinded human verdict on each disagreement; derive each model's "
            "win-rate (human sided with it) per direction, reweight by ht_weights "
            "for an overall, bootstrap over stmt_hash for cluster-robust CIs."
        ),
    }
    return queue, meta


def _summary(queue, meta):
    L = [f"disagreement queue · {meta['model_a']} (A) vs {meta['model_b']} (B) · seed {meta['seed']}",
         f"items={meta['totals']['items']}  distinct statements={meta['totals']['distinct_statements']}  "
         f"max/stmt={meta['totals']['max_items_per_statement']}  double={meta['totals']['double_labeled']}",
         f"curated in queue={meta['totals']['curated_in_queue']} / {meta['totals']['curated_population_total']} "
         f"curated disagreements in population (gradable vs gold)",
         "",
         f"{'direction':<28}{'pop':>9}{'curated':>9}{'share':>8}{'sampled':>9}{'cur·in·q':>10}{'wilson ±':>10}"]
    for dirn in DIRECTIONS:
        popn = meta["population_disagreements"].get(dirn, 0)
        curn = meta["curated_population"].get(dirn, 0)
        curq = meta["achieved_curated_per_direction"].get(dirn, 0)
        L.append(f"{dirn:<28}{popn:>9}{curn:>9}{meta['ht_weights'].get(dirn,0):>7.1%}"
                 f"{meta['achieved_per_direction'].get(dirn,0):>9}{curq:>10}{_wilson_halfwidth(meta['achieved_per_direction'].get(dirn,0)):>9.1%}")
    styp = Counter(it["stmt_type"] for it in queue)
    L.append("\nstmt_type cross-cut: " + ", ".join(f"{k}:{v}" for k, v in styp.most_common(6)))
    buc = Counter(it["bucket_a"] for it in queue)
    L.append("model-A bucket cross-cut: " + ", ".join(f"{k}:{v}" for k, v in buc.most_common(6)))
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-a", required=True, help="export dir of model A")
    ap.add_argument("--run-b", required=True, help="export dir of model B")
    ap.add_argument("--n-per-direction", type=int, default=120)
    ap.add_argument("--cap", type=int, default=3)
    ap.add_argument("--annotators", nargs="+", default=["ann1"])
    ap.add_argument("--double-annotator", default=None)
    ap.add_argument("--double-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=20260603)
    ap.add_argument("--curations", default=DEFAULT_CURATIONS,
                    help="INDRA curations jsonl; curated disagreements are drawn FIRST so the "
                         "blinded verdicts are gradable against gold. Pass '' to disable priority.")
    ap.add_argument("--out", default="data/truth/queue_disagree.jsonl")
    args = ap.parse_args()

    queue, meta = build(args)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        for it in queue:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    with open(args.out.replace(".jsonl", ".meta.json"), "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(_summary(queue, meta))
    print(f"\nwrote {args.out}  ({len(queue)} items)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
