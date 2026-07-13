"""Derive the slide-8 recalibration reliability figure from the CURRENT belief model.

Reads the held-out gold ``holdout_cc`` + its gemma run, computes per-statement
(count belief = INDRA parametric, ignores verdicts) vs (recalibrated belief = the
shipped hybrid log-odds model), and emits:
  - ECE + AUROC for both arms
  - reliability bins (mean predicted, fraction actually right, count) for both
  - the verdict clusters (reject / confirm) with their gold hit-rate
  - paste-ready SVG <circle> coords in the slide's plot frame

Plot frame (matches slides.md slide 8): x: belief 0..1 -> px 58..372 ;
y: actually-right 0..1 -> px 330..18. Dot radius ~ 1.5*sqrt(count), min 3.

Run: python scripts/deck_recalibration_fig.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from calibration_ship_gate import statements_for_run
from indra_belief.statement_belief import statement_belief
from indra_belief.noise_model import RECALIBRATED_PRIORS
from indra_belief.calibration_constants import calibration_for_run
from indra_belief.metrics import auroc, ece, reliability_bins, BINS_8

GOLD = ROOT / "data/results/cc_holdout_cc/holdout_cc.jsonl"
RUN = ROOT / "data/results/holdout_cc_gemma.jsonl"

# plot frame (px) from slides.md slide 8
X0, X1 = 58, 372          # belief 0 .. 1
Y0, Y1 = 330, 18          # actually-right 0 .. 1
px = lambda b: X0 + b * (X1 - X0)
py = lambda a: Y0 + a * (Y1 - Y0)
rad = lambda n: round(max(3.0, 1.5 * math.sqrt(n)), 1)


def load():
    statements, diagnostics = statements_for_run(RUN, GOLD)
    if diagnostics["n_ambiguous_rows"]:
        raise ValueError(f"unsafe holdout source-hash gold join: {diagnostics}")
    return statements


def main():
    calib = calibration_for_run(RUN, "remote-gemma-4-26b")
    if calib is None:
        raise ValueError(f"no ship-approved exact configuration profile for {RUN}")
    stmts = []
    for s in load():
        lab = 1 if s["gold_correct"] else 0
        res = statement_belief(s["ev"], RECALIBRATED_PRIORS, soft=calib)
        recal = res.belief
        count = s["stored_belief"]  # ORIGINAL stored INDRA count belief (the "before")
        if recal is None or count is None:
            continue
        verds = [r["verdict"] for r in s["ev"]]
        scored = [v for v in verds if v in ("correct", "incorrect")]
        stmts.append({"recal": recal, "count": count, "gold": lab,
                      "single": len(set(scored)) == 1,
                      "cluster": scored[0] if len(set(scored)) == 1 else "mixed"})

    def arm(key, label, width01=False):
        scores = [x[key] for x in stmts]
        labels = [x["gold"] for x in stmts]
        pairs = list(zip(scores, [bool(g) for g in labels]))
        print(f"\n== {label} ==  n={len(scores)}  "
              f"ECE={ece(pairs):.3f}  AUROC={auroc(scores, labels):.3f}")
        pts = []
        if width01:
            # 0.1-width bins (the count-belief plot binning, shared with slide 7)
            N = len(scores)
            for i in range(10):
                lo, hi = i / 10, (i + 1) / 10
                grp = [(s, l) for s, l in zip(scores, labels)
                       if (lo <= s < hi or (i == 9 and s >= hi - 1e-9))]
                if not grp:
                    continue
                mb = sum(s for s, _ in grp) / len(grp)
                fr = sum(l for _, l in grp) / len(grp)
                pts.append((mb, fr, len(grp)))
                print(f"   [{lo:.1f},{hi:.1f})  belief~{mb:.3f}  right={fr:.2f}  n={len(grp)}")
            dom = [(s, l) for s, l in zip(scores, labels) if s >= 0.9]
            print(f"   DOMINANT bin [0.9,1.0): {len(dom)} statements, "
                  f"{sum(l for _, l in dom) / len(dom) * 100:.0f}% right")
        else:
            for b in reliability_bins(scores, labels, BINS_8):
                if not b["n"]:
                    continue
                pts.append((b["mean_pred"], b["empirical"], b["n"]))
                print(f"   belief~{b['mean_pred']:.3f}  right={b['empirical']:.2f}  n={b['n']}")
        return pts

    count_pts = arm("count", "COUNT belief (before / amber)", width01=True)
    arm("recal", "RECALIBRATED belief (after / cyan) — value bins, for ECE/AUROC only")

    # For the PLOT, the recalibrated belief is bimodal-by-verdict; show it as its
    # three honest verdict groups (reject / mixed / confirm), each a real point
    # (mean belief, actual frac right, count) — no 1-2-statement binning noise.
    print("\n== recalibrated verdict groups (cyan dots for the plot) ==")
    def group(members, name):
        mb = sum(x["recal"] for x in members) / len(members)
        fr = sum(x["gold"] for x in members) / len(members)
        print(f"   {name:16s} belief~{mb:.3f}  {fr*100:.0f}% right  (n={len(members)})")
        return (mb, fr, len(members))
    rej = [x for x in stmts if x["single"] and x["cluster"] == "incorrect"]
    con = [x for x in stmts if x["single"] and x["cluster"] == "correct"]
    mix = [x for x in stmts if not x["single"]]
    recal_groups = [group(rej, "reject"), group(mix, "mixed"), group(con, "confirm")]
    # count arm: drop negligible (<3-statement) bins so the amber cloud is clean
    count_plot = [p for p in count_pts if p[2] >= 3]

    def svg(pts, fill, op, big_stroke=False):
        out = []
        for mp, fr, cnt in pts:
            r = rad(cnt)
            stroke = ' stroke="#7fe3f0" stroke-width="1"' if (big_stroke and r > 12) else ""
            out.append(f'  <circle cx="{px(mp):.1f}" cy="{py(fr):.1f}" r="{r}" fill="{fill}" opacity="{op}"{stroke}/>')
        return "\n".join(out)

    print("\n== SVG (paste into slide 8) ==")
    print("<!-- BEFORE: count belief (amber, overconfident — below the line) -->")
    print(svg(count_plot, "#f59e0b", "0.22"))
    print("<!-- AFTER: recalibrated (cyan) — reject / mixed / confirm groups -->")
    print(svg(recal_groups, "#4ecadf", "0.75", big_stroke=True))


if __name__ == "__main__":
    main()
