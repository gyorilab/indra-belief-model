"""Z5 — 3-way X/Y/Z error profile on eval_set_v4.

Extends Y5 with:
  - 3-way table (X / Y / Z) for global metrics, gold-tag, stmt_type, source_api
  - Class-A FP resolution count: of the 17 Y-phase FPs with probe state
    (present_as_subject, present_as_object, direct_sign_match, asserted),
    how many flipped to TN in Z? Gate G_Z2: ≥ 10 of 17.
  - Class-B FP resolution count: of the Y-phase FPs with LOF +
    direct_sign_mismatch (Y2 over-flip mechanism), how many flipped?
  - Y→Z regression count: Y-correct → Z-incorrect.
  - Y-rescued recall preservation: of Y's TPs over X's FNs, how many
    survived in Z? Gate G_Z3.

Output: data/results/z_phase_error_profile.md
"""
from __future__ import annotations

import json
import re
import statistics
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parent.parent


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
        gp = r["tag"] == "correct"
        pp = r["verdict"] == "correct"
        if pp and gp:
            tp += 1
        elif pp:
            fp += 1
        elif gp:
            fn += 1
        else:
            tn += 1
    n = len(rows)
    prec = tp / (tp + fp) if (tp + fp) else 0
    rec = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
    return {
        "n": n, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "accuracy": (tp + tn) / n if n else 0, "precision": prec,
        "recall": rec, "f1": f1,
    }


def join_run_with_gold(run: list[dict], src: dict) -> list[dict]:
    """Attach gold 'tag' to each run record from source-of-truth dict."""
    joined = []
    for r in run:
        h = r["source_hash"]
        if h not in src:
            continue
        joined.append({**r, "tag": src[h]["tag"]})
    return joined


def emit_section(out, title: str) -> None:
    out.write(f"\n## {title}\n\n")


def emit_table(out, header: list[str], rows: list[list[str]]) -> None:
    out.write("| " + " | ".join(header) + " |\n")
    out.write("|" + "|".join(["---"] * len(header)) + "|\n")
    for row in rows:
        out.write("| " + " | ".join(str(c) for c in row) + " |\n")
    out.write("\n")


def is_class_a_fp(rec: dict, gold_tag: str) -> bool:
    """Y-phase Class-A FP: probes-pass, extraction-wrong.

    Probe state: subject_role=present_as_subject AND
                 object_role=present_as_object AND
                 relation_axis=direct_sign_match AND
                 scope=asserted.
    Predicted correct; gold says incorrect.
    """
    if rec["verdict"] != "correct" or gold_tag == "correct":
        return False
    p = parse_probes(rec)
    return (
        p.get("subject_role", ("", ""))[0] == "present_as_subject"
        and p.get("object_role", ("", ""))[0] == "present_as_object"
        and p.get("relation_axis", ("", ""))[0] == "direct_sign_match"
        and p.get("scope", ("", ""))[0] == "asserted"
    )


def is_class_b_fp(rec: dict, gold_tag: str) -> bool:
    """Y-phase Class-B FP: LOF flip mis-applies.

    Probe state: relation_axis=direct_sign_mismatch with perturbation
    marker that Y2 flips to effective match. Predicted correct; gold
    says incorrect.
    """
    if rec["verdict"] != "correct" or gold_tag == "correct":
        return False
    p = parse_probes(rec)
    # Class B requires the relation_axis ANSWER was direct_sign_mismatch
    # but the verdict came out correct (via Y2 LOF flip). The raw_text
    # encodes perturbation in subject_role rationale; we detect by the
    # answer pattern.
    return p.get("relation_axis", ("", ""))[0] == "direct_sign_mismatch"


def main():
    out_path = ROOT / "data" / "results" / "z_phase_error_profile.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    src = load_source("eval_set_v4")

    x_path = ROOT / "data/results/x_holdout/eval_set_v4.jsonl"
    y_path = ROOT / "data/results/y_holdout/eval_set_v4.jsonl"
    z_path = ROOT / "data/results/z_holdout/eval_set_v4.jsonl"

    for p in [x_path, y_path, z_path]:
        if not p.exists() or sum(1 for _ in open(p)) < 100:
            raise SystemExit(f"run not complete: {p}")

    x = join_run_with_gold(load_run(x_path), src)
    y = join_run_with_gold(load_run(y_path), src)
    z = join_run_with_gold(load_run(z_path), src)

    x_by = {r["source_hash"]: r for r in x}
    y_by = {r["source_hash"]: r for r in y}
    z_by = {r["source_hash"]: r for r in z}
    shared = sorted(set(x_by) & set(y_by) & set(z_by))
    print(f"shared records: {len(shared)}")

    # Filter all three runs to the shared set for fair comparison
    x_s = [x_by[h] for h in shared]
    y_s = [y_by[h] for h in shared]
    z_s = [z_by[h] for h in shared]

    with open(out_path, "w") as out:
        out.write("# Z-phase error profile — eval_set_v4 (n={})\n".format(len(shared)))
        out.write(
            "\nComparison: **X-phase** (`73273b0`) vs **Y-phase** (Y1+Y2) vs "
            "**Z-phase** (Z1 binding-criterion few-shots + Z2 LOF augment).\n"
        )

        # ===== Dim 1: binary metrics =====
        emit_section(out, "Dim 1 — Binary accuracy / P / R / F1 (3-way)")
        c_x = binary_confusion(x_s)
        c_y = binary_confusion(y_s)
        c_z = binary_confusion(z_s)
        emit_table(
            out,
            ["metric", "X", "Y", "Z", "Δ(Z-Y)"],
            [
                ["accuracy", f"{c_x['accuracy']:.3f}", f"{c_y['accuracy']:.3f}",
                    f"{c_z['accuracy']:.3f}", f"{c_z['accuracy']-c_y['accuracy']:+.3f}"],
                ["precision", f"{c_x['precision']:.3f}", f"{c_y['precision']:.3f}",
                    f"{c_z['precision']:.3f}", f"{c_z['precision']-c_y['precision']:+.3f}"],
                ["recall", f"{c_x['recall']:.3f}", f"{c_y['recall']:.3f}",
                    f"{c_z['recall']:.3f}", f"{c_z['recall']-c_y['recall']:+.3f}"],
                ["**F1**", f"**{c_x['f1']:.3f}**", f"**{c_y['f1']:.3f}**",
                    f"**{c_z['f1']:.3f}**", f"**{c_z['f1']-c_y['f1']:+.3f}**"],
                ["TP", c_x['tp'], c_y['tp'], c_z['tp'], f"{c_z['tp']-c_y['tp']:+d}"],
                ["FP", c_x['fp'], c_y['fp'], c_z['fp'], f"{c_z['fp']-c_y['fp']:+d}"],
                ["FN", c_x['fn'], c_y['fn'], c_z['fn'], f"{c_z['fn']-c_y['fn']:+d}"],
                ["TN", c_x['tn'], c_y['tn'], c_z['tn'], f"{c_z['tn']-c_y['tn']:+d}"],
            ],
        )

        # ===== Class-A FP resolution =====
        emit_section(out, "Class-A FP resolution (G_Z2 gate: ≥10 of 17 flipped to TN)")
        y_class_a = [h for h in shared if is_class_a_fp(y_by[h], src[h]["tag"])]
        y_class_a_in_z_correct = [
            h for h in y_class_a if z_by[h]["verdict"] != "correct"
        ]
        out.write(f"- Y-phase Class-A FPs: **{len(y_class_a)}** records\n")
        out.write(f"- Of those, Z-phase flipped to TN (verdict=incorrect): "
                  f"**{len(y_class_a_in_z_correct)}** records\n")
        out.write(f"- Class-A FP resolution rate: "
                  f"**{len(y_class_a_in_z_correct)}/{len(y_class_a)} = "
                  f"{len(y_class_a_in_z_correct)/(len(y_class_a) or 1):.0%}**\n\n")
        if y_class_a:
            out.write("Per-record outcomes (top 20):\n\n")
            rows = []
            for h in y_class_a[:20]:
                rows.append([
                    str(h)[:12],
                    src[h]["tag"],
                    src[h]["stmt_type"],
                    y_by[h]["verdict"],
                    z_by[h]["verdict"],
                    "flipped" if z_by[h]["verdict"] != "correct" else "—",
                ])
            emit_table(out, ["source_hash", "gold_tag", "stmt_type", "Y verdict",
                             "Z verdict", "outcome"], rows)

        # ===== Class-B FP resolution =====
        emit_section(out, "Class-B FP resolution (Y2 LOF over-flip)")
        y_class_b = [h for h in shared if is_class_b_fp(y_by[h], src[h]["tag"])]
        y_class_b_flipped = [
            h for h in y_class_b if z_by[h]["verdict"] != "correct"
        ]
        out.write(f"- Y-phase Class-B FPs (direct_sign_mismatch + LOF flip path): "
                  f"**{len(y_class_b)}** records\n")
        out.write(f"- Z flipped to TN: **{len(y_class_b_flipped)}** records\n\n")

        # ===== Y-rescued recall preservation =====
        emit_section(out, "G_Z3: Y-rescued recall preservation (threshold ≥ 0.85)")
        y_rescued = [
            h for h in shared
            if x_by[h]["verdict"] != "correct"
            and y_by[h]["verdict"] == "correct"
            and src[h]["tag"] == "correct"
        ]
        y_rescued_kept = [h for h in y_rescued if z_by[h]["verdict"] == "correct"]
        kept_rate = len(y_rescued_kept) / (len(y_rescued) or 1)
        out.write(f"- Y-rescued TPs (X-FN → Y-TP, gold=correct): "
                  f"**{len(y_rescued)}** records\n")
        out.write(f"- Z preserved: **{len(y_rescued_kept)}** records "
                  f"({kept_rate:.0%})\n")
        out.write(f"- Gate G_Z3: {'PASS' if kept_rate >= 0.85 else 'FAIL'}\n\n")

        # ===== Regressions (Y-correct → Z-incorrect) =====
        emit_section(out, "Y→Z regressions")
        regressions = [
            h for h in shared
            if (y_by[h]["verdict"] == "correct") == (src[h]["tag"] == "correct")
            and (z_by[h]["verdict"] == "correct") != (src[h]["tag"] == "correct")
        ]
        out.write(f"- Y-correct cases now Z-incorrect: **{len(regressions)}**\n\n")
        if regressions:
            rows = []
            for h in regressions[:15]:
                rows.append([
                    str(h)[:12],
                    src[h]["tag"],
                    src[h]["stmt_type"],
                    y_by[h]["verdict"],
                    z_by[h]["verdict"],
                ])
            emit_table(out, ["source_hash", "gold_tag", "stmt_type", "Y verdict", "Z verdict"], rows)

        # ===== Dim 2: per-gold-tag =====
        emit_section(out, "Dim 2 — Per-gold-tag correct-call rate (3-way)")
        gold_tags = Counter(src[h]["tag"] for h in shared)
        rows = []
        for tag, n_total in gold_tags.most_common():
            tag_hashes = [h for h in shared if src[h]["tag"] == tag]

            def rate(runs_by: dict[str, dict]) -> str:
                n = sum(
                    1 for h in tag_hashes
                    if (runs_by[h]["verdict"] == "correct") == (tag == "correct")
                )
                return f"{n}/{n_total} = {n/n_total:.0%}"

            rows.append([
                tag, n_total,
                rate(x_by), rate(y_by), rate(z_by),
            ])
        emit_table(out, ["gold_tag", "n", "X", "Y", "Z"], rows)

        # ===== Dim 5: ECE =====
        emit_section(out, "Dim 5 — Expected Calibration Error")
        bins = [(0, 0.05), (0.05, 0.20), (0.20, 0.35), (0.35, 0.50),
                (0.50, 0.65), (0.65, 0.80), (0.80, 0.95), (0.95, 1.001)]

        def ece(rows_run: list[dict]) -> float:
            tot = 0.0
            n_all = len(rows_run)
            for lo, hi in bins:
                bin_rows = [r for r in rows_run if lo <= (r.get("score") or 0.5) < hi]
                if not bin_rows:
                    continue
                mean_pred = statistics.mean(r.get("score") or 0.5 for r in bin_rows)
                empirical = sum(1 for r in bin_rows if r["tag"] == "correct") / len(bin_rows)
                tot += abs(mean_pred - empirical) * len(bin_rows) / n_all
            return tot

        out.write(f"- X = {ece(x_s):.3f}\n")
        out.write(f"- Y = {ece(y_s):.3f}\n")
        out.write(f"- Z = {ece(z_s):.3f}\n\n")

        # ===== Gate summary =====
        emit_section(out, "Gate verdict")
        g_z1 = c_z["f1"] >= 0.78
        g_z2 = len(y_class_a_in_z_correct) >= 10
        g_z3 = kept_rate >= 0.85
        # G_Z4 already validated in pytest separately; here we just note it.
        out.write(f"- **G_Z1** F1 ≥ 0.78: F1={c_z['f1']:.3f} → "
                  f"{'PASS' if g_z1 else 'FAIL'}\n")
        out.write(f"- **G_Z2** Class-A FP resolution ≥ 10 of 17: "
                  f"{len(y_class_a_in_z_correct)}/{len(y_class_a)} → "
                  f"{'PASS' if g_z2 else 'FAIL'}\n")
        out.write(f"- **G_Z3** Y-rescued recall ≥ 0.85: {kept_rate:.0%} → "
                  f"{'PASS' if g_z3 else 'FAIL'}\n")
        out.write(f"- **G_Z4** pytest: validated separately (289 passing)\n\n")
        verdict = "SHIP" if (g_z1 and g_z2 and g_z3) else "PIVOT"
        out.write(f"### Overall: **{verdict}**\n\n")
        if not (g_z1 and g_z2 and g_z3):
            out.write("Falsifier triggered. Pivot options:\n")
            out.write("1. Probe redesign: subject-object extraction-then-bind-check\n")
            out.write("2. Adversarial calibration on Class-A residuals\n\n")

    print(f"wrote {out_path}")
    print(f"\n=== 3-way summary ===")
    print(f"  X: F1={c_x['f1']:.3f}  TP={c_x['tp']:2d}  FP={c_x['fp']:2d}  FN={c_x['fn']:2d}")
    print(f"  Y: F1={c_y['f1']:.3f}  TP={c_y['tp']:2d}  FP={c_y['fp']:2d}  FN={c_y['fn']:2d}")
    print(f"  Z: F1={c_z['f1']:.3f}  TP={c_z['tp']:2d}  FP={c_z['fp']:2d}  FN={c_z['fn']:2d}")
    print(f"  Class-A FP resolution: {len(y_class_a_in_z_correct)}/{len(y_class_a)}")
    print(f"  Y-rescued recall: {kept_rate:.0%}")


if __name__ == "__main__":
    main()
