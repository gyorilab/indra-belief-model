#!/usr/bin/env python3
"""Build a frozen, stratified human-in-the-loop review queue from a run export.

Implements the sampling design for estimating LLM-scorer vs. human *agreement*
(there is no gold for the monolithic run — the human is the reference standard):

  * Two-stage cluster sample. Statements are the primary sampling unit; at most
    `--cap` (default 3) evidence rows are drawn per statement *across the whole
    queue*. This bounds the mega-Complex (one statement holds 3,462 rows / 7.3%
    of the stream) to 3 reads and keeps the design effect ≈2 instead of ≈3.2.
  * Stratify by the report `bucket` (the only variable that partitions the *kind*
    of judgment). Equal-n over the substantive buckets so each gets a usable
    per-stratum CI; tiny telemetry buckets get a small audit-n. Population shares
    (Horvitz-Thompson weights) are frozen in so an overall estimate can be
    reweighted later. `confidence` is dropped (degenerate); `stmt_type` /
    `source_api` are reporting cross-cuts, not allocators.
  * A `--source-floor` guarantees the long-tail readers (isi, trips, …) appear.
  * Annotators + a double-labeled subset (for Cohen's kappa) are assigned here.

The queue is the sample design, auditable before any human time is spent. The
review app (viewer /review) consumes (stmt_hash, evidence_hash), strips the model
verdict/score/reasoning/bucket, and shows only statement + sentence for a blinded
first human verdict. NOTHING here is shown to the annotator pre-commit.

Usage (rough pass)::

    python scripts/build_review_queue.py --pass rough \
        --annotators ann1 --double-annotator ann2 --double-frac 0.15
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
from collections import Counter, defaultdict

# 6 substantive buckets get equal allocation; the 2 telemetry/stub buckets get a
# small fixed audit-n (we want to confirm routing, not a tight rate).
SUBSTANTIVE = [
    "semantic_correct", "semantic_incorrect", "hedged_evidence",
    "reader_hallucination", "no_evidence", "incomplete_claim",
]
AUDIT = ["row_error", "placeholder_text"]


def _load_export(export_dir: str):
    with open(os.path.join(export_dir, "per_evidence.jsonl")) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    with open(os.path.join(export_dir, "export_meta.json")) as f:
        meta = json.load(f)
    return rows, meta


def _item_id(run_id: str, stmt_hash: str, evidence_hash: str) -> str:
    h = hashlib.sha1(f"{run_id}|{stmt_hash}|{evidence_hash}".encode()).hexdigest()
    return h[:12]


def _wilson_halfwidth(n: int, p: float = 0.5, z: float = 1.96) -> float:
    """Wilson 95% interval half-width at the given p (worst case p=0.5)."""
    if n <= 0:
        return float("nan")
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    # report the (slightly asymmetric) interval's larger half-width
    return max(centre - (centre - margin), (centre + margin) - centre, margin)


def two_stage_sample(rows, target, cap, taken_per_stmt, rng):
    """Draw up to `target` rows: shuffle statements (PSUs), take <= remaining-cap
    rows each (rows shuffled), respecting the GLOBAL per-statement cap so a single
    statement never contributes more than `cap` items to the whole queue."""
    by_stmt = defaultdict(list)
    for r in rows:
        by_stmt[r["stmt_hash"]].append(r)
    stmts = list(by_stmt.keys())
    rng.shuffle(stmts)
    picked = []
    for h in stmts:
        if len(picked) >= target:
            break
        remaining_cap = cap - taken_per_stmt.get(h, 0)
        if remaining_cap <= 0:
            continue
        srows = by_stmt[h][:]
        rng.shuffle(srows)
        take = min(remaining_cap, len(srows), target - len(picked))
        for r in srows[:take]:
            picked.append(r)
            taken_per_stmt[h] += 1
    return picked


def build_queue(args) -> tuple[list[dict], dict]:
    rows, export_meta = _load_export(args.export)
    run_id = export_meta.get("run_id", "unknown")
    model = export_meta.get("model", "unknown")
    rng = random.Random(args.seed)

    # population bucket shares -> Horvitz-Thompson weights
    pop = Counter(r.get("bucket") for r in rows)
    total = sum(pop.values())
    weights = {b: pop[b] / total for b in pop}

    by_bucket = defaultdict(list)
    for r in rows:
        by_bucket[r.get("bucket")].append(r)

    taken_per_stmt: dict[str, int] = defaultdict(int)
    queue: list[dict] = []

    def emit(r, stratum, supplemental=False):
        queue.append({
            "item_id": _item_id(run_id, r["stmt_hash"], r["evidence_hash"]),
            "stmt_hash": r["stmt_hash"], "evidence_hash": r["evidence_hash"],
            "stmt_i": r.get("stmt_i"), "evidence_i": r.get("evidence_i"),
            "stratum": stratum,
            "bucket": r.get("bucket"),
            "stmt_type": r.get("stmt_type"),
            "source_api": r.get("source_api"),
            "stratum_weight": weights.get(r.get("bucket")),
            "supplemental": supplemental,
        })

    # equal-n over substantive buckets, audit-n over telemetry buckets
    achieved = Counter()
    for b in SUBSTANTIVE + AUDIT:
        tgt = args.audit_n if b in AUDIT else args.n_per_bucket
        picked = two_stage_sample(by_bucket.get(b, []), tgt, args.cap, taken_per_stmt, rng)
        for r in picked:
            emit(r, f"bucket:{b}")
        achieved[b] = len(picked)

    # source-api floor: make sure the long tail is represented (supplemental)
    src_have = Counter(it["source_api"] for it in queue)
    chosen = {(it["stmt_hash"], it["evidence_hash"]) for it in queue}
    src_supp = Counter()
    if args.source_floor > 0:
        all_src = {r.get("source_api") for r in rows}
        for src in sorted(s for s in all_src if s):
            need = args.source_floor - src_have.get(src, 0)
            if need <= 0:
                continue
            pool = [r for r in rows if r.get("source_api") == src
                    and (r["stmt_hash"], r["evidence_hash"]) not in chosen]
            picked = two_stage_sample(pool, need, args.cap, taken_per_stmt, rng)
            for r in picked:
                emit(r, f"source_floor:{src}", supplemental=True)
                chosen.add((r["stmt_hash"], r["evidence_hash"]))
            src_supp[src] = len(picked)

    # annotator + double-label assignment (shuffle for balanced primary load)
    rng.shuffle(queue)
    n_double = round(args.double_frac * len(queue)) if args.double_annotator else 0
    for i, it in enumerate(queue):
        primary = args.annotators[i % len(args.annotators)]
        anns = [primary]
        it["double"] = i < n_double
        if it["double"] and args.double_annotator and args.double_annotator != primary:
            anns.append(args.double_annotator)
        it["annotators"] = anns

    # re-sort by item_id for a stable on-disk order (assignment already frozen)
    queue.sort(key=lambda it: it["item_id"])

    n_stmts = len({it["stmt_hash"] for it in queue})
    max_per_stmt = max(Counter(it["stmt_hash"] for it in queue).values()) if queue else 0
    meta = {
        "pass": args.pass_name,
        "seed": args.seed,
        "run_id": run_id,
        "model": model,
        "export": args.export,
        "params": {
            "n_per_bucket": args.n_per_bucket, "audit_n": args.audit_n,
            "cap": args.cap, "source_floor": args.source_floor,
            "double_frac": args.double_frac,
            "annotators": args.annotators, "double_annotator": args.double_annotator,
        },
        "population_bucket_counts": dict(pop),
        "ht_weights": weights,
        "achieved_per_bucket": dict(achieved),
        "achieved_supplemental_per_source": dict(src_supp),
        "totals": {
            "items": len(queue),
            "distinct_statements": n_stmts,
            "max_items_per_statement": max_per_stmt,
            "double_labeled": n_double,
        },
        "estimand_note": (
            "Agreement against a blinded human, NOT accuracy vs gold. Report "
            "semantic-only + raw-stream + input-quality separately; reweight "
            "per-bucket rates by ht_weights for a population estimate; bootstrap "
            "over stmt_hash for cluster-robust CIs."
        ),
    }
    return queue, meta


def _summary(queue, meta) -> str:
    lines = [
        f"pass={meta['pass']} seed={meta['seed']} run={meta['run_id'][:8]} model={meta['model']}",
        f"items={meta['totals']['items']}  distinct statements={meta['totals']['distinct_statements']}  "
        f"max items/stmt={meta['totals']['max_items_per_statement']} (cap {meta['params']['cap']})  "
        f"double-labeled={meta['totals']['double_labeled']}",
        "",
        f"{'stratum':<22}{'pop share':>10}{'sampled':>9}{'wilson ±':>10}",
    ]
    for b in SUBSTANTIVE + AUDIT:
        n = meta["achieved_per_bucket"].get(b, 0)
        w = meta["ht_weights"].get(b, 0)
        hw = _wilson_halfwidth(n)
        lines.append(f"{b:<22}{w:>9.1%}{n:>9}{hw:>9.1%}")
    if meta["achieved_supplemental_per_source"]:
        supp = ", ".join(f"{k}+{v}" for k, v in sorted(meta["achieved_supplemental_per_source"].items()))
        lines.append(f"\nsource-floor supplements: {supp}")
    # overall reweighted precision (per-bucket equal-n -> HT variance; cluster-inflated)
    var = sum((meta["ht_weights"].get(b, 0) ** 2) * (0.25 / max(1, meta["achieved_per_bucket"].get(b, 0)))
              for b in SUBSTANTIVE)
    deff = 1.7  # ~ at cap=3, rho~0.49; rough guide, not exact
    overall_hw = 1.96 * math.sqrt(var) * math.sqrt(deff)
    lines.append(f"\noverall (HT-reweighted, substantive buckets, ~DEFF {deff}): ±{overall_hw:.1%}")
    src = Counter(it["source_api"] for it in queue)
    lines.append("source coverage: " + ", ".join(f"{k}:{v}" for k, v in src.most_common()))
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--export", default="data/exports/rasmachine_belief", help="run export dir")
    ap.add_argument("--pass", dest="pass_name", default="rough", choices=("rough", "robust"))
    ap.add_argument("--n-per-bucket", type=int, default=None, help="default 96 rough / 384 robust")
    ap.add_argument("--audit-n", type=int, default=30)
    ap.add_argument("--cap", type=int, default=3, help="max evidence rows per statement (global)")
    ap.add_argument("--source-floor", type=int, default=6, help="min items per source_api (supplemental)")
    ap.add_argument("--annotators", nargs="+", default=["ann1"], help="primary annotator id(s)")
    ap.add_argument("--double-annotator", default=None, help="id who re-labels a subset for kappa")
    ap.add_argument("--double-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=20260602)
    ap.add_argument("--out", default=None, help="default data/truth/queue_<pass>.jsonl")
    args = ap.parse_args()
    if args.n_per_bucket is None:
        args.n_per_bucket = 384 if args.pass_name == "robust" else 96

    queue, meta = build_queue(args)

    out = args.out or os.path.join("data", "truth", f"queue_{args.pass_name}.jsonl")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        for it in queue:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    meta_path = out.replace(".jsonl", ".meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(_summary(queue, meta))
    print(f"\nwrote {out}  ({len(queue)} items)\nwrote {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
