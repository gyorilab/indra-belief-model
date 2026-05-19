"""Build holdout_cc — a 500-record stratified curator-audited holdout
from holdout_large, excluding eval_set_v4 contamination.

Strategy:
  1. Load eval_set_v4 source_hashes (the contamination guard set).
  2. Load holdout_large; drop records whose source_hash appears in eval_set_v4.
  3. Bucket the remaining records by (stmt_type, gold_tag).
  4. Stratified sample 500 records:
     - stmt_type quotas: target the natural proportions in holdout_large
       but cap any single bucket at 35% (Complex was 35% — at cap).
     - gold_tag quotas: aim for at least 30% correct (positives) and ≥5
       records each in the top-8 error classes (grounding, no_relation,
       wrong_relation, entity_boundaries, act_vs_amt, polarity, other,
       hypothesis).
  5. Write data/benchmark/holdout_cc.jsonl.

Per CC0 (research/cc_phase_new_holdout_requirements.md): n ≥ 500
discriminates +0.01 F1 architectural wins at α=0.05 power=0.8 via
McNemar test; ≤5% hedged + ≤2% reader-error required for the
ceiling-protection threshold.
"""
from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).parent.parent

# Fixed seed for reproducibility — the holdout composition is
# load-bearing for cross-phase comparisons.
RANDOM_SEED = 20260519
TARGET_N = 500


def load_eval_set_v4_hashes() -> set[int]:
    """Return the source_hashes in eval_set_v4 (contamination guard)."""
    hashes: set[int] = set()
    for line in open(ROOT / "data/benchmark/eval_set_v4.jsonl"):
        r = json.loads(line)
        hashes.add(r["source_hash"])
    return hashes


def load_holdout_large() -> list[dict]:
    return [json.loads(l) for l in open(ROOT / "data/benchmark/holdout_large.jsonl")]


def stratified_sample(pool: list[dict], target: int, seed: int) -> list[dict]:
    """Sample `target` records from `pool`, balancing across (stmt_type, tag).

    Approach: round-robin draw from non-empty buckets, prioritizing
    buckets that are under-represented relative to a target distribution.
    """
    rng = random.Random(seed)

    # Bucket by (stmt_type, tag) — fine-grained stratification.
    buckets: dict[tuple, list[dict]] = defaultdict(list)
    for r in pool:
        key = (r.get("stmt_type", "?"), r.get("tag", "?"))
        buckets[key].append(r)

    # Shuffle each bucket so we draw varied records.
    for k in buckets:
        rng.shuffle(buckets[k])

    # Stmt-type cap: no single type > 35% of final set.
    stmt_cap = int(target * 0.35)
    # Per-stmt_type counts so far
    stmt_counts: Counter = Counter()

    # Tag-target counts: aim for at least the listed minimums.
    tag_min = {
        "correct":               int(target * 0.40),  # ~200
        "grounding":             int(target * 0.10),  # ~50
        "no_relation":           int(target * 0.08),  # ~40
        "wrong_relation":        int(target * 0.06),  # ~30
        "entity_boundaries":     int(target * 0.05),  # ~25
        "act_vs_amt":            int(target * 0.05),  # ~25
        "polarity":              int(target * 0.04),  # ~20
        "other":                 int(target * 0.04),  # ~20
        "hypothesis":            int(target * 0.02),  # ~10
        "negative_result":       int(target * 0.02),  # ~10
        "mod_site":              5,
        "agent_conditions":      5,
    }
    tag_counts: Counter = Counter()

    selected: list[dict] = []

    # Pass 1: fulfill tag minimums first (don't blow stmt cap)
    bucket_keys_by_tag: dict[str, list[tuple]] = defaultdict(list)
    for key in buckets:
        bucket_keys_by_tag[key[1]].append(key)
    for tag in bucket_keys_by_tag:
        rng.shuffle(bucket_keys_by_tag[tag])

    for tag, min_n in tag_min.items():
        while tag_counts[tag] < min_n:
            # Try each (stmt_type, tag) bucket round-robin
            drew_one = False
            for key in bucket_keys_by_tag.get(tag, []):
                if not buckets[key]:
                    continue
                stmt_type = key[0]
                if stmt_counts[stmt_type] >= stmt_cap:
                    continue
                rec = buckets[key].pop()
                selected.append(rec)
                tag_counts[tag] += 1
                stmt_counts[stmt_type] += 1
                drew_one = True
                if tag_counts[tag] >= min_n:
                    break
            if not drew_one:
                # Pool exhausted for this tag — accept what we have
                break

    # Pass 2: fill remaining slots with random records honoring stmt_cap.
    remaining_pool: list[dict] = []
    for key, records in buckets.items():
        for r in records:
            remaining_pool.append(r)
    rng.shuffle(remaining_pool)

    for rec in remaining_pool:
        if len(selected) >= target:
            break
        stmt_type = rec.get("stmt_type", "?")
        if stmt_counts[stmt_type] >= stmt_cap:
            continue
        selected.append(rec)
        stmt_counts[stmt_type] += 1
        tag_counts[rec.get("tag", "?")] += 1

    return selected[:target]


def report(records: list[dict]) -> None:
    print(f"\nSelected: {len(records)} records")
    print("\nTag distribution:")
    for t, n in Counter(r.get("tag", "?") for r in records).most_common():
        print(f"  {t:25s} {n:4d}  ({n/len(records)*100:.1f}%)")
    print("\nStmt-type distribution:")
    for t, n in Counter(r.get("stmt_type", "?") for r in records).most_common():
        print(f"  {t:25s} {n:4d}  ({n/len(records)*100:.1f}%)")
    print("\nSource-API distribution:")
    for s, n in Counter(r.get("source_api", "?") for r in records).most_common():
        print(f"  {s:15s} {n:4d}  ({n/len(records)*100:.1f}%)")


def main() -> None:
    print(f"random seed = {RANDOM_SEED}")
    eval4_hashes = load_eval_set_v4_hashes()
    print(f"eval_set_v4 source_hashes (contamination guard): {len(eval4_hashes)}")

    large = load_holdout_large()
    print(f"holdout_large total: {len(large)}")

    # Apply contamination guard
    clean_pool = [r for r in large if r["source_hash"] not in eval4_hashes]
    dropped = len(large) - len(clean_pool)
    print(f"dropped (eval_v4 contamination): {dropped}")
    print(f"clean pool: {len(clean_pool)}")

    selected = stratified_sample(clean_pool, TARGET_N, RANDOM_SEED)
    report(selected)

    out_path = ROOT / "data/benchmark/holdout_cc.jsonl"
    with open(out_path, "w") as f:
        for r in selected:
            f.write(json.dumps(r) + "\n")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
