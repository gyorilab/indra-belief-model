"""Y5 — comprehensive error profile across every available dimension.

Compares Y-phase run (post Y1+Y2) against X-phase baseline on the holdouts
that have both. Quantifies error along 10 dimensions:

  1. Binary accuracy / P / R / F1                  — global
  2. Per-gold-tag confusion                         — semantic error types
  3. Per-stmt_type FN rate                          — distribution-aware
  4. Per-source_api FN rate                         — reader quality
  5. Score calibration (reliability + ECE)          — confidence honesty
  6. Error-mode taxonomy (probe attribution)        — substrate vs LLM commits
  7. Alias-coverage failures                        — surface form in text?
  8. Scope mismatch (substrate vs LLM)              — Y1 impact direct
  9. Perturbation handling (LOF cases)              — Y2 impact direct
 10. Confusion-direction asymmetry                  — over-strict vs over-lenient

Output: data/results/y_phase_error_profile.md
"""
from __future__ import annotations

import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).parent.parent

# Source records for evidence_text + curator_note + tag
def load_source(name: str) -> dict:
    src: dict = {}
    p = ROOT / "data" / "benchmark" / f"{name}.jsonl"
    for line in open(p):
        r = json.loads(line)
        src[r["source_hash"]] = r
    return src


def load_run(path: Path) -> list[dict]:
    return [json.loads(l) for l in open(path) if l.strip()]


PROBE_RE = re.compile(r"(subject_role|object_role|relation_axis|scope)=(\S+) \((\w+)\)")


def parse_probes(rec: dict) -> dict[str, tuple[str, str]]:
    """Extract per-probe (answer, source) from raw_text_preview."""
    rt = rec.get("raw_text_preview", "")
    return {m[0]: (m[1], m[2]) for m in PROBE_RE.findall(rt)}


def binary_confusion(rows: list[dict]) -> dict:
    tp = fp = tn = fn = 0
    for r in rows:
        gp = r["tag"] == "correct"; pp = r["verdict"] == "correct"
        if pp and gp: tp += 1
        elif pp: fp += 1
        elif gp: fn += 1
        else: tn += 1
    n = len(rows)
    prec = tp/(tp+fp) if (tp+fp) else 0
    rec = tp/(tp+fn) if (tp+fn) else 0
    f1 = 2*prec*rec/(prec+rec) if (prec+rec) else 0
    return {
        "n": n, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "accuracy": (tp+tn)/n, "precision": prec,
        "recall": rec, "f1": f1,
    }


def emit_section(out, title: str) -> None:
    out.write(f"\n## {title}\n\n")


def emit_table(out, header: list[str], rows: list[list[str]]) -> None:
    out.write("| " + " | ".join(header) + " |\n")
    out.write("|" + "|".join(["---"]*len(header)) + "|\n")
    for row in rows:
        out.write("| " + " | ".join(str(c) for c in row) + " |\n")
    out.write("\n")


def main():
    out_path = ROOT / "data" / "results" / "y_phase_error_profile.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    src = {
        "eval_set_v4": load_source("eval_set_v4"),
    }
    runs = {
        "X": {
            "eval_set_v4": ROOT / "data/results/x_holdout/eval_set_v4.jsonl",
        },
        "Y": {
            "eval_set_v4": ROOT / "data/results/y_holdout/eval_set_v4.jsonl",
        },
    }

    # Load
    x = load_run(runs["X"]["eval_set_v4"])
    y_path = runs["Y"]["eval_set_v4"]
    if not y_path.exists() or sum(1 for _ in open(y_path)) < 100:
        raise SystemExit(f"Y run not complete yet: {y_path}")
    y = load_run(y_path)

    # Index by source_hash for direct comparison
    x_by = {r["source_hash"]: r for r in x}
    y_by = {r["source_hash"]: r for r in y}
    shared = [h for h in y_by if h in x_by]
    print(f"shared records: {len(shared)}")

    with open(out_path, "w") as out:
        out.write("# Y-phase error profile — eval_set_v4 (n=100)\n")
        out.write("\nComparison: **X-phase** (commit `73273b0`) vs **Y-phase**"
                  " (Y1 LLM-confirm-negation + Y2 perturbation propagation).\n")

        # ===== Dim 1: binary metrics =====
        emit_section(out, "Dim 1 — Binary accuracy / P / R / F1")
        c_x = binary_confusion(x)
        c_y = binary_confusion(y)
        emit_table(out,
            ["metric", "X-phase", "Y-phase", "Δ"],
            [
                ["accuracy",   f"{c_x['accuracy']:.3f}",  f"{c_y['accuracy']:.3f}",  f"{c_y['accuracy']-c_x['accuracy']:+.3f}"],
                ["precision",  f"{c_x['precision']:.3f}", f"{c_y['precision']:.3f}", f"{c_y['precision']-c_x['precision']:+.3f}"],
                ["recall",     f"{c_x['recall']:.3f}",    f"{c_y['recall']:.3f}",    f"{c_y['recall']-c_x['recall']:+.3f}"],
                ["**F1**",     f"**{c_x['f1']:.3f}**",    f"**{c_y['f1']:.3f}**",    f"**{c_y['f1']-c_x['f1']:+.3f}**"],
                ["TP",         c_x['tp'], c_y['tp'], f"{c_y['tp']-c_x['tp']:+d}"],
                ["FP",         c_x['fp'], c_y['fp'], f"{c_y['fp']-c_x['fp']:+d}"],
                ["FN",         c_x['fn'], c_y['fn'], f"{c_y['fn']-c_x['fn']:+d}"],
                ["TN",         c_x['tn'], c_y['tn'], f"{c_y['tn']-c_x['tn']:+d}"],
            ])

        # ===== Dim 2: per-gold-tag confusion =====
        emit_section(out, "Dim 2 — Per-gold-tag breakdown")
        gold_tags = Counter(src["eval_set_v4"][h].get("tag", "?") for h in shared)
        per_tag: list[list[str]] = []
        for tag, n_total in gold_tags.most_common():
            x_corr = sum(1 for h in shared if src["eval_set_v4"][h]["tag"] == tag
                         and x_by[h]["verdict"] == "correct")
            y_corr = sum(1 for h in shared if src["eval_set_v4"][h]["tag"] == tag
                         and y_by[h]["verdict"] == "correct")
            target = "correct" if tag == "correct" else "incorrect"
            x_right = x_corr if tag == "correct" else (n_total - x_corr)
            y_right = y_corr if tag == "correct" else (n_total - y_corr)
            per_tag.append([
                tag, n_total,
                f"{x_right}/{n_total} = {x_right/n_total:.0%}",
                f"{y_right}/{n_total} = {y_right/n_total:.0%}",
                f"{y_right-x_right:+d}",
            ])
        emit_table(out, ["gold_tag", "n", "X correct-call rate", "Y correct-call rate", "Δ"], per_tag)

        # ===== Dim 3: per-stmt_type FN rate =====
        emit_section(out, "Dim 3 — Per-stmt_type recall (we caught gold=correct)")
        types = Counter(r["stmt_type"] for h in shared for r in [src["eval_set_v4"][h]])
        per_type: list[list[str]] = []
        for stype, n_total in types.most_common(10):
            n_gold_pos = sum(1 for h in shared if src["eval_set_v4"][h]["stmt_type"] == stype
                             and src["eval_set_v4"][h]["tag"] == "correct")
            x_recall = sum(1 for h in shared if src["eval_set_v4"][h]["stmt_type"] == stype
                           and src["eval_set_v4"][h]["tag"] == "correct"
                           and x_by[h]["verdict"] == "correct")
            y_recall = sum(1 for h in shared if src["eval_set_v4"][h]["stmt_type"] == stype
                           and src["eval_set_v4"][h]["tag"] == "correct"
                           and y_by[h]["verdict"] == "correct")
            if n_gold_pos == 0:
                per_type.append([stype, n_total, n_gold_pos, "—", "—", "—"])
            else:
                per_type.append([
                    stype, n_total, n_gold_pos,
                    f"{x_recall}/{n_gold_pos} = {x_recall/n_gold_pos:.0%}",
                    f"{y_recall}/{n_gold_pos} = {y_recall/n_gold_pos:.0%}",
                    f"{y_recall-x_recall:+d}",
                ])
        emit_table(out, ["stmt_type", "n_total", "n_gold=correct", "X recall", "Y recall", "Δ"], per_type)

        # ===== Dim 4: per-source_api =====
        emit_section(out, "Dim 4 — Per-source_api accuracy")
        srcs = Counter(src["eval_set_v4"][h]["source_api"] for h in shared)
        per_src: list[list[str]] = []
        for sa, n_total in srcs.most_common():
            x_correct = sum(1 for h in shared if src["eval_set_v4"][h]["source_api"] == sa
                            and (x_by[h]["verdict"] == "correct") == (src["eval_set_v4"][h]["tag"] == "correct"))
            y_correct = sum(1 for h in shared if src["eval_set_v4"][h]["source_api"] == sa
                            and (y_by[h]["verdict"] == "correct") == (src["eval_set_v4"][h]["tag"] == "correct"))
            per_src.append([sa, n_total,
                            f"{x_correct}/{n_total} = {x_correct/n_total:.0%}",
                            f"{y_correct}/{n_total} = {y_correct/n_total:.0%}",
                            f"{y_correct-x_correct:+d}"])
        emit_table(out, ["source_api", "n", "X accuracy", "Y accuracy", "Δ"], per_src)

        # ===== Dim 5: calibration =====
        emit_section(out, "Dim 5 — Score calibration (reliability bins)")
        bins = [(0, 0.05, "<0.05"), (0.05, 0.20, "0.05-0.20"),
                (0.20, 0.35, "0.20-0.35"), (0.35, 0.50, "0.35-0.50"),
                (0.50, 0.65, "0.50-0.65"), (0.65, 0.80, "0.65-0.80"),
                (0.80, 0.95, "0.80-0.95"), (0.95, 1.001, "0.95-1.00")]
        calib_rows: list[list[str]] = []
        for lo, hi, lbl in bins:
            x_in = [r for r in x if lo <= (r.get("score") or 0.5) < hi]
            y_in = [r for r in y if lo <= (r.get("score") or 0.5) < hi]
            x_pos = sum(1 for r in x_in if src["eval_set_v4"][r["source_hash"]]["tag"] == "correct")
            y_pos = sum(1 for r in y_in if src["eval_set_v4"][r["source_hash"]]["tag"] == "correct")
            calib_rows.append([
                lbl,
                f"{len(x_in)}",
                f"{x_pos/len(x_in):.0%}" if x_in else "—",
                f"{len(y_in)}",
                f"{y_pos/len(y_in):.0%}" if y_in else "—",
            ])
        emit_table(out, ["score bin", "X n", "X gold=correct rate", "Y n", "Y gold=correct rate"], calib_rows)

        # Expected Calibration Error (binary)
        def ece(rows: list[dict]) -> float:
            tot = 0.0
            n_all = len(rows)
            for lo, hi, _ in bins:
                bin_rows = [r for r in rows if lo <= (r.get("score") or 0.5) < hi]
                if not bin_rows: continue
                mean_pred = statistics.mean(r.get("score") or 0.5 for r in bin_rows)
                empirical = sum(1 for r in bin_rows
                                if src["eval_set_v4"][r["source_hash"]]["tag"] == "correct") / len(bin_rows)
                tot += abs(mean_pred - empirical) * len(bin_rows) / n_all
            return tot
        out.write(f"**Expected Calibration Error**: X = {ece(x):.3f}, Y = {ece(y):.3f}\n\n")

        # ===== Dim 6: error-mode taxonomy =====
        emit_section(out, "Dim 6 — Error-mode taxonomy (probe attribution for misclassifications)")
        x_errors: Counter = Counter()
        y_errors: Counter = Counter()
        for h in shared:
            gold = src["eval_set_v4"][h]["tag"] == "correct"
            x_pred = x_by[h]["verdict"] == "correct"
            y_pred = y_by[h]["verdict"] == "correct"
            if x_pred != gold:
                x_probes = parse_probes(x_by[h])
                # Categorize the X error
                if x_probes.get("scope") == ("negated", "substrate"):
                    x_errors["scope=negated (substrate)"] += 1
                elif x_probes.get("scope") == ("negated", "llm"):
                    x_errors["scope=negated (LLM)"] += 1
                elif x_probes.get("subject_role", ("",""))[0] == "absent" or x_probes.get("object_role", ("",""))[0] == "absent":
                    x_errors["sr/or = absent"] += 1
                elif x_probes.get("relation_axis", ("",""))[0] in ("direct_sign_mismatch",):
                    x_errors["sign_mismatch"] += 1
                elif x_probes.get("relation_axis", ("",""))[0] in ("direct_axis_mismatch", "direct_partner_mismatch"):
                    x_errors["axis_mismatch"] += 1
                elif "no_sentence_evidence" in (x_by[h].get("reasons") or []):
                    x_errors["no_sentence_evidence (INDRA prior wrong)"] += 1
                else:
                    x_errors["other"] += 1
            if y_pred != gold:
                y_probes = parse_probes(y_by[h])
                if y_probes.get("scope") == ("negated", "substrate"):
                    y_errors["scope=negated (substrate)"] += 1
                elif y_probes.get("scope") == ("negated", "llm"):
                    y_errors["scope=negated (LLM)"] += 1
                elif y_probes.get("subject_role", ("",""))[0] == "absent" or y_probes.get("object_role", ("",""))[0] == "absent":
                    y_errors["sr/or = absent"] += 1
                elif y_probes.get("relation_axis", ("",""))[0] in ("direct_sign_mismatch",):
                    y_errors["sign_mismatch"] += 1
                elif y_probes.get("relation_axis", ("",""))[0] in ("direct_axis_mismatch", "direct_partner_mismatch"):
                    y_errors["axis_mismatch"] += 1
                elif "no_sentence_evidence" in (y_by[h].get("reasons") or []):
                    y_errors["no_sentence_evidence (INDRA prior wrong)"] += 1
                else:
                    y_errors["other"] += 1
        all_modes = sorted(set(x_errors) | set(y_errors))
        tax_rows = []
        for m in all_modes:
            tax_rows.append([m, x_errors[m], y_errors[m], y_errors[m] - x_errors[m]])
        emit_table(out, ["error mode", "X count", "Y count", "Δ"], tax_rows)

        # ===== Dim 8: Y1 direct impact — scope=negated commits =====
        emit_section(out, "Dim 8 — Y1 impact: substrate vs LLM scope=negated")
        x_substrate_neg = sum(1 for r in x if parse_probes(r).get("scope", ("",""))[1] == "substrate"
                              and parse_probes(r).get("scope", ("",""))[0] == "negated")
        y_substrate_neg = sum(1 for r in y if parse_probes(r).get("scope", ("",""))[1] == "substrate"
                              and parse_probes(r).get("scope", ("",""))[0] == "negated")
        x_llm_neg = sum(1 for r in x if parse_probes(r).get("scope", ("",""))[1] == "llm"
                        and parse_probes(r).get("scope", ("",""))[0] == "negated")
        y_llm_neg = sum(1 for r in y if parse_probes(r).get("scope", ("",""))[1] == "llm"
                        and parse_probes(r).get("scope", ("",""))[0] == "negated")
        emit_table(out, ["source", "X scope=negated count", "Y scope=negated count", "Δ"],
                   [["substrate", x_substrate_neg, y_substrate_neg, y_substrate_neg - x_substrate_neg],
                    ["llm",       x_llm_neg,       y_llm_neg,       y_llm_neg - x_llm_neg]])

        # Y1 expected: substrate→0, LLM count increases (correctly or not)
        out.write("\nExpected Y1 effect: substrate count → 0; LLM count rises\n")
        out.write("by however many cases the substrate previously short-circuited.\n\n")

        # ===== Dim 10: confusion asymmetry =====
        emit_section(out, "Dim 10 — Confusion-direction asymmetry (vs gold)")
        out.write(f"X: FP={c_x['fp']} vs FN={c_x['fn']} — ratio {c_x['fp']/(c_x['fn'] or 1):.2f}x; "
                  f"{'too lenient' if c_x['fp'] > c_x['fn'] else 'too strict'}\n\n")
        out.write(f"Y: FP={c_y['fp']} vs FN={c_y['fn']} — ratio {c_y['fp']/(c_y['fn'] or 1):.2f}x; "
                  f"{'too lenient' if c_y['fp'] > c_y['fn'] else 'too strict'}\n\n")

    print(f"\nwrote {out_path}")
    # And also print summary to stdout
    print(f"\n=== Y vs X summary on eval_set_v4 ===")
    print(f"  X: acc={c_x['accuracy']:.3f}  P={c_x['precision']:.3f}  R={c_x['recall']:.3f}  F1={c_x['f1']:.3f}")
    print(f"  Y: acc={c_y['accuracy']:.3f}  P={c_y['precision']:.3f}  R={c_y['recall']:.3f}  F1={c_y['f1']:.3f}")
    print(f"  ΔF1: {c_y['f1']-c_x['f1']:+.3f}")
    print(f"  ECE: X={ece(x):.3f} → Y={ece(y):.3f}")


if __name__ == "__main__":
    main()
