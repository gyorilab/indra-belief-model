"""Compare AA vs CC on holdout_cc with McNemar's significance test.

The holdout_cc was commissioned (per research/cc_phase_new_holdout_
requirements.md) to discriminate +0.01 F1 architectural wins from
LLM-sampling variance. n = 500 stratified records, no eval_set_v4
contamination, ≤5% hedged + ≤2% reader-error per CC0 audit gates.

This script:
  1. Loads AA + CC verdicts joined with gold tags.
  2. Computes F1/P/R/acc with 4-way contingency.
  3. Runs McNemar's exact test on discordant pairs:
       b = (AA correct, CC wrong)
       c = (AA wrong, CC correct)
       under H_0 (no difference) b ~ Binomial(b+c, 0.5).
     A p-value < 0.05 with c > b means CC is a significant improvement.
  4. Per-stmt_type, per-gold_tag, per-source_api breakdowns.
  5. ECE comparison.

Output: data/results/cc_holdout_cc_profile.md
"""
from __future__ import annotations

import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).parent.parent


def load_jsonl(p: Path) -> dict:
    return {json.loads(l)["source_hash"]: json.loads(l) for l in open(p)}


def confusion(run: dict, shared: list, src: dict) -> dict:
    tp = fp = fn = tn = 0
    for h in shared:
        gold = src[h]["tag"] == "correct"
        pred = run[h]["verdict"] == "correct"
        if pred and gold:
            tp += 1
        elif pred:
            fp += 1
        elif gold:
            fn += 1
        else:
            tn += 1
    n = tp + fp + fn + tn
    p = tp / (tp + fp) if (tp + fp) else 0
    r = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * p * r / (p + r) if (p + r) else 0
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, p=p, r=r, f1=f1,
                acc=(tp + tn) / n if n else 0)


def mcnemar_exact_pvalue(b: int, c: int) -> float:
    """Exact (mid-p) McNemar test on discordant pairs.

    H_0: P(AA-correct, CC-wrong) = P(AA-wrong, CC-correct).
    Two-sided p-value: 2 × P(X ≥ max(b,c)) under Binomial(b+c, 0.5).
    Returns 1.0 when b == c == 0.
    """
    n = b + c
    if n == 0:
        return 1.0
    k = max(b, c)
    # Right-tail probability under Binom(n, 0.5)
    tail = sum(math.comb(n, i) for i in range(k, n + 1)) / (2 ** n)
    return min(2 * tail, 1.0)


def ece(rows: Iterable[dict], src: dict) -> float:
    """Expected Calibration Error on the standard 8-bin scheme."""
    bins = [(0, 0.05), (0.05, 0.20), (0.20, 0.35), (0.35, 0.50),
            (0.50, 0.65), (0.65, 0.80), (0.80, 0.95), (0.95, 1.001)]
    rows_list = list(rows)
    n_all = len(rows_list)
    if n_all == 0:
        return 0.0
    tot = 0.0
    for lo, hi in bins:
        bin_rows = [
            r for r in rows_list
            if lo <= (r.get("score") or 0.5) < hi
        ]
        if not bin_rows:
            continue
        mean_pred = statistics.mean(r.get("score") or 0.5 for r in bin_rows)
        empirical = sum(
            1 for r in bin_rows
            if src[r["source_hash"]]["tag"] == "correct"
        ) / len(bin_rows)
        tot += abs(mean_pred - empirical) * len(bin_rows) / n_all
    return tot


def emit_table(out, header: list[str], rows: list[list]) -> None:
    out.write("| " + " | ".join(str(c) for c in header) + " |\n")
    out.write("|" + "|".join(["---"] * len(header)) + "|\n")
    for row in rows:
        out.write("| " + " | ".join(str(c) for c in row) + " |\n")
    out.write("\n")


def main() -> None:
    out_path = ROOT / "data/results/cc_holdout_cc_profile.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    src = {
        json.loads(l)["source_hash"]: json.loads(l)
        for l in open(ROOT / "data/benchmark/holdout_cc.jsonl")
    }
    aa = load_jsonl(ROOT / "data/results/aa_holdout_cc/holdout_cc.jsonl")
    cc = load_jsonl(ROOT / "data/results/cc_holdout_cc/holdout_cc.jsonl")

    shared = sorted(set(src) & set(aa) & set(cc))
    print(f"shared records: {len(shared)}")

    c_aa = confusion(aa, shared, src)
    c_cc = confusion(cc, shared, src)

    # McNemar on correctness disagreement
    aa_correct_cc_wrong = 0  # b
    aa_wrong_cc_correct = 0  # c
    both_correct = 0
    both_wrong = 0
    for h in shared:
        gold = src[h]["tag"] == "correct"
        aa_match = (aa[h]["verdict"] == "correct") == gold
        cc_match = (cc[h]["verdict"] == "correct") == gold
        if aa_match and cc_match:
            both_correct += 1
        elif aa_match and not cc_match:
            aa_correct_cc_wrong += 1
        elif not aa_match and cc_match:
            aa_wrong_cc_correct += 1
        else:
            both_wrong += 1

    p_value = mcnemar_exact_pvalue(aa_correct_cc_wrong, aa_wrong_cc_correct)

    with open(out_path, "w") as out:
        out.write(f"# CC vs AA on holdout_cc (n={len(shared)})\n\n")
        out.write(
            "**Holdout:** 500-record stratified, no eval_set_v4 contamination, "
            "hedged 0.40% + reader-error 1.60% (CC0 gates PASS, ceiling 98%).\n\n"
            "**Phases compared:**\n"
            "- **AA** (production, commit `f5ab653`): closed-set "
            "relation_axis with BINDING CRITERION + COUNTER-EXAMPLES.\n"
            "- **CC** (this implementation): extract-then-bind-check — LLM "
            "extracts structured relation tuple, deterministic Python "
            "bind-check verdicts via alias resolution + axis taxonomy + "
            "sign reconciliation + binding-axis symmetry.\n\n"
        )

        # ==== Binary metrics ====
        out.write("## Dim 1 — Binary metrics\n\n")
        emit_table(
            out,
            ["metric", "AA", "CC", "Δ"],
            [
                ["**F1**", f"**{c_aa['f1']:.3f}**", f"**{c_cc['f1']:.3f}**",
                 f"**{c_cc['f1']-c_aa['f1']:+.3f}**"],
                ["accuracy", f"{c_aa['acc']:.3f}", f"{c_cc['acc']:.3f}",
                 f"{c_cc['acc']-c_aa['acc']:+.3f}"],
                ["precision", f"{c_aa['p']:.3f}", f"{c_cc['p']:.3f}",
                 f"{c_cc['p']-c_aa['p']:+.3f}"],
                ["recall", f"{c_aa['r']:.3f}", f"{c_cc['r']:.3f}",
                 f"{c_cc['r']-c_aa['r']:+.3f}"],
                ["TP", c_aa['tp'], c_cc['tp'],
                 f"{c_cc['tp']-c_aa['tp']:+d}"],
                ["FP", c_aa['fp'], c_cc['fp'],
                 f"{c_cc['fp']-c_aa['fp']:+d}"],
                ["FN", c_aa['fn'], c_cc['fn'],
                 f"{c_cc['fn']-c_aa['fn']:+d}"],
                ["TN", c_aa['tn'], c_cc['tn'],
                 f"{c_cc['tn']-c_aa['tn']:+d}"],
            ],
        )

        # ==== McNemar's test ====
        out.write("## Dim 2 — McNemar's exact test\n\n")
        out.write(f"- Both correct (concordant):       {both_correct}\n")
        out.write(f"- Both wrong (concordant):         {both_wrong}\n")
        out.write(f"- AA correct, CC wrong (b):        **{aa_correct_cc_wrong}**\n")
        out.write(f"- AA wrong, CC correct (c):        **{aa_wrong_cc_correct}**\n")
        out.write(f"- Discordant pairs (b+c):          {aa_correct_cc_wrong + aa_wrong_cc_correct}\n")
        out.write(f"- McNemar two-sided p-value:       **{p_value:.4f}**\n")
        winner = "CC" if aa_wrong_cc_correct > aa_correct_cc_wrong else "AA"
        sig = "SIGNIFICANT" if p_value < 0.05 else "NOT significant at α=0.05"
        out.write(f"- Difference direction:            {winner} preferred\n")
        out.write(f"- Verdict:                         **{sig}**\n\n")

        # ==== Per-gold-tag ====
        out.write("## Dim 3 — Per-gold-tag correct-call rate\n\n")
        tag_dist = Counter(src[h]["tag"] for h in shared)
        rows: list[list] = []
        for tag, n_total in tag_dist.most_common():
            tag_hashes = [h for h in shared if src[h]["tag"] == tag]

            def rate(run_by):
                n = sum(
                    1 for h in tag_hashes
                    if (run_by[h]["verdict"] == "correct") == (tag == "correct")
                )
                return f"{n}/{n_total} = {n/n_total:.0%}"

            rows.append([tag, n_total, rate(aa), rate(cc)])
        emit_table(out, ["gold_tag", "n", "AA", "CC"], rows)

        # ==== Per-stmt_type ====
        out.write("## Dim 4 — Per-stmt_type accuracy\n\n")
        stmt_dist = Counter(src[h]["stmt_type"] for h in shared)
        rows = []
        for stype, n_total in stmt_dist.most_common():
            stype_hashes = [h for h in shared if src[h]["stmt_type"] == stype]

            def rate(run_by):
                n = sum(
                    1 for h in stype_hashes
                    if (run_by[h]["verdict"] == "correct") == (src[h]["tag"] == "correct")
                )
                return f"{n}/{n_total} = {n/n_total:.0%}"

            rows.append([stype, n_total, rate(aa), rate(cc)])
        emit_table(out, ["stmt_type", "n", "AA", "CC"], rows)

        # ==== Per-source_api ====
        out.write("## Dim 5 — Per-source_api accuracy\n\n")
        src_dist = Counter(src[h]["source_api"] for h in shared)
        rows = []
        for sa, n_total in src_dist.most_common():
            sa_hashes = [h for h in shared if src[h]["source_api"] == sa]

            def rate(run_by):
                n = sum(
                    1 for h in sa_hashes
                    if (run_by[h]["verdict"] == "correct") == (src[h]["tag"] == "correct")
                )
                return f"{n}/{n_total} = {n/n_total:.0%}"

            rows.append([sa, n_total, rate(aa), rate(cc)])
        emit_table(out, ["source_api", "n", "AA", "CC"], rows)

        # ==== ECE ====
        out.write("## Dim 6 — Calibration (ECE)\n\n")
        aa_rows = [aa[h] for h in shared]
        cc_rows = [cc[h] for h in shared]
        out.write(f"- AA ECE: **{ece(aa_rows, src):.3f}**\n")
        out.write(f"- CC ECE: **{ece(cc_rows, src):.3f}**\n\n")

        # ==== Decision ====
        out.write("## Decision\n\n")
        if p_value < 0.05 and aa_wrong_cc_correct > aa_correct_cc_wrong:
            out.write("**Ship CC to origin/main.** Significant improvement "
                      f"over AA at α=0.05 (p={p_value:.4f}); architectural "
                      f"redesign validated.\n")
        elif p_value < 0.05 and aa_correct_cc_wrong > aa_wrong_cc_correct:
            out.write("**Do not ship CC.** Significant REGRESSION vs AA at "
                      f"α=0.05 (p={p_value:.4f}); architectural redesign "
                      f"under-performs on holdout_cc.\n")
        else:
            out.write(f"**Inconclusive.** p={p_value:.4f} — not significant "
                      f"at α=0.05. Discordant pairs: AA-only {aa_correct_cc_wrong} "
                      f"vs CC-only {aa_wrong_cc_correct}.\n")

    print(f"\nwrote {out_path}")
    print(f"\nAA: F1={c_aa['f1']:.3f}  acc={c_aa['acc']:.3f}  TP={c_aa['tp']} FP={c_aa['fp']} FN={c_aa['fn']} TN={c_aa['tn']}")
    print(f"CC: F1={c_cc['f1']:.3f}  acc={c_cc['acc']:.3f}  TP={c_cc['tp']} FP={c_cc['fp']} FN={c_cc['fn']} TN={c_cc['tn']}")
    print(f"McNemar p={p_value:.4f}; b={aa_correct_cc_wrong}, c={aa_wrong_cc_correct}")


if __name__ == "__main__":
    main()
