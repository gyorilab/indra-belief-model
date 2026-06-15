#!/usr/bin/env python3
"""Recompute Figure-5 numbers (aggregation-policy lever) from the COMPLETE
rasmachine scoring run, and print old-vs-new for every cited value.

Figure 5 partitions "touched" statements into:
  all-correct  : every scored evidence verdict == 'correct'
  mixed        : at least one 'correct' AND at least one 'incorrect'
  all-incorrect: no scored evidence is 'correct' (every scored evidence 'incorrect')
where "touched" = statement (stmt_hash) with >=1 scored evidence (verdict in
{'correct','incorrect'}).

Policies:
  any-correct% = (all_correct + mixed) / touched     (snapshot 48.0%)
  all-correct% = all_correct / touched               (snapshot 23.1%)
  mixed%       = mixed / touched                      (snapshot 24.9%)

Dedup by (stmt_i, evidence_i), keep last occurrence.
"""
import json
from collections import defaultdict

SRC = "data/results/rasmachine_mono_gemma_remote_direct.jsonl"

def main():
    # dedup by (stmt_i, evidence_i), keep last
    rows = {}
    with open(SRC) as f:
        for line in f:
            d = json.loads(line)
            rows[(d["stmt_i"], d["evidence_i"])] = d

    print(f"total unique rows (after dedup): {len(rows)}")

    # per-statement tally of scored verdicts
    # stmt -> [n_correct, n_incorrect]
    per_stmt = defaultdict(lambda: [0, 0])
    for d in rows.values():
        v = d.get("verdict")
        if v == "correct":
            per_stmt[d["stmt_hash"]][0] += 1
        elif v == "incorrect":
            per_stmt[d["stmt_hash"]][1] += 1
        # null/other verdicts do not count as "scored"

    # touched = >=1 scored evidence
    touched = [(nc, ni) for (nc, ni) in per_stmt.values() if (nc + ni) > 0]
    n_touched = len(touched)

    all_correct = sum(1 for nc, ni in touched if ni == 0 and nc > 0)
    all_incorrect = sum(1 for nc, ni in touched if nc == 0 and ni > 0)
    mixed = sum(1 for nc, ni in touched if nc > 0 and ni > 0)

    assert all_correct + all_incorrect + mixed == n_touched, "partition mismatch"

    any_correct_n = all_correct + mixed
    any_correct_pct = 100 * any_correct_n / n_touched
    all_correct_pct = 100 * all_correct / n_touched
    mixed_pct = 100 * mixed / n_touched
    all_incorrect_pct = 100 * all_incorrect / n_touched
    ratio = any_correct_pct / all_correct_pct

    # ---- old (snapshot) values for comparison ----
    old = {
        "n_touched": 7349,
        "all_correct_n": 1700,
        "mixed_n": 1828,
        "all_incorrect_n": 3821,
        "any_correct_n": 3529,
        "any_correct_pct": 48.0,
        "all_correct_pct": 23.1,
        "mixed_pct": 24.9,
        "all_incorrect_pct": 52.0,
    }

    def line(label, oldv, newv, fmt="{}"):
        print(f"  {label:<22} old={fmt.format(oldv):>10}   new={fmt.format(newv):>10}")

    print("\n=== Figure 5: touched-statement partition (old snapshot vs new complete) ===")
    line("n touched", old["n_touched"], n_touched)
    line("all-correct n", old["all_correct_n"], all_correct)
    line("mixed n", old["mixed_n"], mixed)
    line("all-incorrect n", old["all_incorrect_n"], all_incorrect)
    line("any-correct n", old["any_correct_n"], any_correct_n)
    print()
    line("any-correct %", old["any_correct_pct"], any_correct_pct, "{:.1f}")
    line("all-correct %", old["all_correct_pct"], all_correct_pct, "{:.1f}")
    line("mixed %", old["mixed_pct"], mixed_pct, "{:.1f}")
    line("all-incorrect %", old["all_incorrect_pct"], all_incorrect_pct, "{:.1f}")
    print(f"\n  any/all ratio (the '2x' swing): {ratio:.2f}x")

    # exact rounded values to splice into the report
    print("\n=== values to splice (rounded as report formats them) ===")
    print(f"  n touched           = {n_touched:,}")
    print(f"  all-correct seg     = {all_correct:,} · {all_correct_pct:.1f}%")
    print(f"  mixed seg           = {mixed:,} · {mixed_pct:.1f}%")
    print(f"  all-incorrect seg   = {all_incorrect:,} · {all_incorrect_pct:.1f}%")
    print(f"  any-correct policy  = {any_correct_pct:.1f}%  (n={any_correct_n:,})")
    print(f"  all-correct policy  = {all_correct_pct:.1f}%  (n={all_correct:,})")
    print(f"  flex weights        = all:{all_correct_pct:.1f} mixed:{mixed_pct:.1f} none:{all_incorrect_pct:.1f}")

if __name__ == "__main__":
    main()
