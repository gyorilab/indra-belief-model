"""Carve a BALANCED, type-de-skewed gold out of rasmachine_v2_gold.jsonl.

The eval gold is 1:1 correct/incorrect and must not be dominated by Complex. Two
cuts are emitted so the cost is visible:

  * stratified  — stratified 1:1 by stmt_type (build_curation_eval's blessed rule:
                  per type take min(#c,#i) of EACH class, so the type marginal is
                  IDENTICAL across classes — Complex can't masquerade as a label
                  effect). Largest clean n.
  * capped      — same, but each type's per-class count is capped at the 2nd-largest
                  type's, so Complex is a genuine minority (true "no skew").

One statements file already exists (rasmachine_v2_statements.json, 304 ev); score
it ONCE and evaluate against whichever gold subset — the join keeps only its pairs.

    PYTHONPATH=src .venv/bin/python scripts/build_v2_balanced.py
"""
from __future__ import annotations

import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from build_curation_eval import stratified_balanced  # noqa: E402  (blessed 1:1-by-type rule)

GOLD = ROOT / "data" / "benchmark" / "rasmachine_v2_gold.jsonl"
OUT_STRAT = ROOT / "data" / "benchmark" / "rasmachine_v2_balanced_gold.jsonl"
OUT_CAP = ROOT / "data" / "benchmark" / "rasmachine_v2_balanced_capped_gold.jsonl"


def complex_share(rows) -> str:
    n = len(rows)
    nc = sum(1 for r in rows if r["stmt_type"] == "Complex")
    ni = sum(1 for r in rows if r["gold"] == "incorrect")
    return f"n={n}  ({ni} inc / {n-ni} cor)  Complex={nc} ({100*nc/max(n,1):.0f}%)"


def capped(rows, rng):
    """Stratified 1:1 by type but cap each type's per-class take at the
    2nd-largest type's incorrect count, so no single type dominates."""
    by_type = defaultdict(lambda: {"correct": [], "incorrect": []})
    for r in rows:
        by_type[r["stmt_type"]][r["gold"]].append(r)
    # per-type balanced availability = min(#c, #i)
    avail = {t: min(len(c["correct"]), len(c["incorrect"])) for t, c in by_type.items()}
    pos = sorted((v for v in avail.values() if v > 0), reverse=True)
    cap = pos[1] if len(pos) > 1 else (pos[0] if pos else 0)  # 2nd-largest
    picked = []
    for t, c in by_type.items():
        m = min(len(c["correct"]), len(c["incorrect"]), cap)
        if m == 0:
            continue
        rng.shuffle(c["correct"]); rng.shuffle(c["incorrect"])
        picked += c["correct"][:m] + c["incorrect"][:m]
    return picked, cap


def main():
    rows = [json.loads(l) for l in open(GOLD) if l.strip()]
    print(f"full v2 gold: {complex_share(rows)}")

    rng = random.Random(20260630)
    strat, report = stratified_balanced(rows, rng)
    with open(OUT_STRAT, "w") as f:
        for r in strat:
            f.write(json.dumps(r, default=str) + "\n")
    print(f"\n[stratified 1:1 by type] -> {OUT_STRAT.name}")
    print(f"  {complex_share(strat)}")
    print(f"  per-type each-class: {report['per_type_pairs_each_class']}")
    print(f"  dropped (single-class types): {report['types_dropped']}")

    cap_rows, cap = capped(rows, random.Random(20260630))
    with open(OUT_CAP, "w") as f:
        for r in cap_rows:
            f.write(json.dumps(r, default=str) + "\n")
    print(f"\n[capped: Complex a minority, per-type cap={cap}] -> {OUT_CAP.name}")
    print(f"  {complex_share(cap_rows)}")
    bt = Counter(r["stmt_type"] for r in cap_rows)
    print(f"  by type: {dict(bt.most_common())}")

    nonc_inc = sum(1 for r in rows if r["gold"] == "incorrect" and r["stmt_type"] != "Complex")
    print(f"\nBINDING CONSTRAINT: only {nonc_inc} non-Complex INCORRECT pairs exist in "
          f"all of Ben's corpus curations — that, not total volume, caps a de-skewed "
          f"balanced eval.")


if __name__ == "__main__":
    main()
