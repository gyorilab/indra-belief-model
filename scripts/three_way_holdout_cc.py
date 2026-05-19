"""Three-way comparison on holdout_cc: AA (decomposed) vs CC (extract-bind)
vs Monolithic (single-call).

Same McNemar machinery as cc_holdout_cc_compare.py, extended to three
arms. Pairwise McNemar tests on (AA vs Mono), (CC vs Mono), and
(AA vs CC); the latter is the previous result repeated for context.

Output: data/results/three_way_holdout_cc_profile.md
"""
from __future__ import annotations

import json
import math
import statistics
from collections import Counter
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).parent.parent


def load(p: Path) -> dict:
    return {json.loads(l)["source_hash"]: json.loads(l) for l in open(p)}


def confusion(run: dict, shared: list, src: dict) -> dict:
    tp = fp = fn = tn = 0
    for h in shared:
        gold = src[h]["tag"] == "correct"
        pred = run[h]["verdict"] == "correct"
        if pred and gold: tp += 1
        elif pred: fp += 1
        elif gold: fn += 1
        else: tn += 1
    n = tp + fp + fn + tn
    p = tp / (tp + fp) if (tp + fp) else 0
    r = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * p * r / (p + r) if (p + r) else 0
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, p=p, r=r, f1=f1,
                acc=(tp + tn) / n if n else 0)


def mcnemar_exact_pvalue(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = max(b, c)
    tail = sum(math.comb(n, i) for i in range(k, n + 1)) / (2 ** n)
    return min(2 * tail, 1.0)


def pairwise_mcnemar(run_a: dict, run_b: dict, shared: list, src: dict) -> dict:
    """Compute discordant pairs and McNemar p between two runs.
    `b` = A-only-right, `c` = B-only-right."""
    both_right = both_wrong = b = c = 0
    for h in shared:
        gold = src[h]["tag"] == "correct"
        a_right = (run_a[h]["verdict"] == "correct") == gold
        b_right = (run_b[h]["verdict"] == "correct") == gold
        if a_right and b_right:
            both_right += 1
        elif a_right and not b_right:
            b += 1
        elif not a_right and b_right:
            c += 1
        else:
            both_wrong += 1
    p = mcnemar_exact_pvalue(b, c)
    return dict(both_right=both_right, both_wrong=both_wrong,
                a_only=b, b_only=c, p=p)


def ece(rows, src) -> float:
    bins = [(0, 0.05), (0.05, 0.20), (0.20, 0.35), (0.35, 0.50),
            (0.50, 0.65), (0.65, 0.80), (0.80, 0.95), (0.95, 1.001)]
    rows_list = list(rows)
    n_all = len(rows_list)
    if not n_all:
        return 0.0
    tot = 0.0
    for lo, hi in bins:
        bin_rows = [r for r in rows_list if lo <= (r.get("score") or 0.5) < hi]
        if not bin_rows:
            continue
        mean_pred = statistics.mean(r.get("score") or 0.5 for r in bin_rows)
        empirical = sum(
            1 for r in bin_rows
            if src[r["source_hash"]]["tag"] == "correct"
        ) / len(bin_rows)
        tot += abs(mean_pred - empirical) * len(bin_rows) / n_all
    return tot


def emit_table(out, header: list, rows: list) -> None:
    out.write("| " + " | ".join(str(c) for c in header) + " |\n")
    out.write("|" + "|".join(["---"] * len(header)) + "|\n")
    for row in rows:
        out.write("| " + " | ".join(str(c) for c in row) + " |\n")
    out.write("\n")


def main():
    out_path = ROOT / "data/results/three_way_holdout_cc_profile.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    src = {json.loads(l)["source_hash"]: json.loads(l)
           for l in open(ROOT / "data/benchmark/holdout_cc.jsonl")}
    aa = load(ROOT / "data/results/aa_holdout_cc/holdout_cc.jsonl")
    cc = load(ROOT / "data/results/cc_holdout_cc/holdout_cc.jsonl")
    mono = load(ROOT / "data/results/monolithic_holdout_cc/holdout_cc.jsonl")

    shared = sorted(set(src) & set(aa) & set(cc) & set(mono))
    print(f"shared records: {len(shared)}")

    c_aa = confusion(aa, shared, src)
    c_cc = confusion(cc, shared, src)
    c_mn = confusion(mono, shared, src)

    with open(out_path, "w") as out:
        out.write(f"# 3-way comparison on holdout_cc (n={len(shared)})\n\n")
        out.write(
            "**Holdout:** 500-record stratified, no eval_set_v4 contamination, "
            "hedged 0.40% + reader-error 1.60% (CC0 gates PASS, ceiling 98%).\n\n"
            "**Arms:**\n"
            "- **AA (decomposed)**: four-probe (subject_role, object_role, "
            "relation_axis, scope) + adjudicator decision table. Production "
            "at commit `f5ab653`.\n"
            "- **CC (decomposed + extract-bind)**: relation_axis replaced "
            "with LLM extraction + deterministic bind-check.\n"
            "- **Monolithic**: single deterministic LLM call per (Stmt, Ev) "
            "(temp=0.1) with type-adaptive contrastive few-shots.\n\n"
        )

        out.write("## Binary metrics\n\n")
        emit_table(
            out,
            ["metric", "AA", "CC", "Monolithic"],
            [
                ["**F1**", f"**{c_aa['f1']:.3f}**", f"**{c_cc['f1']:.3f}**",
                 f"**{c_mn['f1']:.3f}**"],
                ["accuracy", f"{c_aa['acc']:.3f}", f"{c_cc['acc']:.3f}",
                 f"{c_mn['acc']:.3f}"],
                ["precision", f"{c_aa['p']:.3f}", f"{c_cc['p']:.3f}",
                 f"{c_mn['p']:.3f}"],
                ["recall", f"{c_aa['r']:.3f}", f"{c_cc['r']:.3f}",
                 f"{c_mn['r']:.3f}"],
                ["TP", c_aa['tp'], c_cc['tp'], c_mn['tp']],
                ["FP", c_aa['fp'], c_cc['fp'], c_mn['fp']],
                ["FN", c_aa['fn'], c_cc['fn'], c_mn['fn']],
                ["TN", c_aa['tn'], c_cc['tn'], c_mn['tn']],
            ],
        )

        out.write("## Pairwise McNemar's exact tests\n\n")
        runs = {"AA": aa, "CC": cc, "Mono": mono}
        rows = []
        for (na, ra), (nb, rb) in combinations(runs.items(), 2):
            r = pairwise_mcnemar(ra, rb, shared, src)
            winner = na if r["a_only"] > r["b_only"] else nb if r["b_only"] > r["a_only"] else "tie"
            sig = "SIG" if r["p"] < 0.05 else ""
            rows.append([
                f"{na} vs {nb}",
                r["both_right"], r["both_wrong"],
                f"{r['a_only']} ({na})",
                f"{r['b_only']} ({nb})",
                f"{r['p']:.4f}",
                winner, sig,
            ])
        emit_table(out, ["pair", "both right", "both wrong",
                         "a-only-right", "b-only-right",
                         "p-value", "preferred", "alpha=0.05"], rows)

        out.write("## Per-gold-tag correct-call rate\n\n")
        tag_dist = Counter(src[h]["tag"] for h in shared)
        rows = []
        for tag, n_total in tag_dist.most_common():
            hashes = [h for h in shared if src[h]["tag"] == tag]

            def rate(run_by):
                n = sum(1 for h in hashes
                        if (run_by[h]["verdict"] == "correct") == (tag == "correct"))
                return f"{n}/{n_total} = {n/n_total:.0%}"
            rows.append([tag, n_total, rate(aa), rate(cc), rate(mono)])
        emit_table(out, ["gold_tag", "n", "AA", "CC", "Monolithic"], rows)

        out.write("## Per-stmt_type accuracy\n\n")
        stmt_dist = Counter(src[h]["stmt_type"] for h in shared)
        rows = []
        for stype, n_total in stmt_dist.most_common():
            hashes = [h for h in shared if src[h]["stmt_type"] == stype]

            def rate(run_by):
                n = sum(1 for h in hashes
                        if (run_by[h]["verdict"] == "correct")
                        == (src[h]["tag"] == "correct"))
                return f"{n}/{n_total} = {n/n_total:.0%}"
            rows.append([stype, n_total, rate(aa), rate(cc), rate(mono)])
        emit_table(out, ["stmt_type", "n", "AA", "CC", "Monolithic"], rows)

        out.write("## Calibration (ECE)\n\n")
        out.write(f"- AA:         {ece([aa[h] for h in shared], src):.3f}\n")
        out.write(f"- CC:         {ece([cc[h] for h in shared], src):.3f}\n")
        out.write(f"- Monolithic: {ece([mono[h] for h in shared], src):.3f}\n\n")

        out.write("## Headline\n\n")
        winner_arch, winner_f1 = max(
            [("AA", c_aa["f1"]), ("CC", c_cc["f1"]), ("Mono", c_mn["f1"])],
            key=lambda x: x[1],
        )
        out.write(f"Best F1: **{winner_arch} = {winner_f1:.3f}**\n\n")

    print(f"wrote {out_path}")
    print()
    print(f"AA:   F1={c_aa['f1']:.3f}  acc={c_aa['acc']:.3f}  TP={c_aa['tp']} FP={c_aa['fp']} FN={c_aa['fn']}")
    print(f"CC:   F1={c_cc['f1']:.3f}  acc={c_cc['acc']:.3f}  TP={c_cc['tp']} FP={c_cc['fp']} FN={c_cc['fn']}")
    print(f"Mono: F1={c_mn['f1']:.3f}  acc={c_mn['acc']:.3f}  TP={c_mn['tp']} FP={c_mn['fp']} FN={c_mn['fn']}")
    print()
    print("Pairwise McNemar:")
    for (na, ra), (nb, rb) in combinations(runs.items(), 2):
        r = pairwise_mcnemar(ra, rb, shared, src)
        print(f"  {na} vs {nb}: b={r['a_only']} c={r['b_only']} p={r['p']:.4f}")


if __name__ == "__main__":
    main()
