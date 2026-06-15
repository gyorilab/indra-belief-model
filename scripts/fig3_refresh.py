#!/usr/bin/env python3
"""Recompute ALL of Figure 3's numbers (joint distribution hexbin + marginals +
the "agreement table" sub-section) from the COMPLETE rasmachine scoring run.

Figure 3 = joint distribution of RasMachine belief (X) vs our mean-aggregated
belief (Y), hexbin + marginals, plus the agreement table right after it.

Filter (bucket-aware, per the brief):
  - classify each evidence row into a bucket (reuse classify()/split_preview()
    from fig1_drilldown_data.py).
  - "non-artifact" buckets = {semantic_correct, semantic_incorrect, hedged_evidence}.
  - keep a statement if it has >=1 non-artifact evidence.
  - OUR belief = aggregate over that statement's NON-artifact evidence ONLY:
      mean policy   = arithmetic mean of `score`
      noisy-OR      = 1 - prod(1 - score)
  - RasMachine belief = the `belief` field (constant within a stmt_hash).

Prints old-vs-new for every number Figure 3 cites.
"""
import json, math, sys
from collections import defaultdict
from statistics import mean

sys.path.insert(0, "scripts")
from fig1_drilldown_data import classify, split_preview  # noqa: E402
from fig_utils import SRC, NON_ARTIFACT  # noqa: E402, F401


def pearson(xs, ys):
    n = len(xs)
    mx, my = mean(xs), mean(ys)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx == 0 or syy == 0:
        return float("nan")
    return sxy / math.sqrt(sxx * syy)


def main():
    # dedup by (stmt_i, evidence_i), keep last
    rows = {}
    with open(SRC) as f:
        for line in f:
            d = json.loads(line)
            rows[(d["stmt_i"], d["evidence_i"])] = d

    n_unique = len(rows)

    # group rows by stmt_hash; collect non-artifact scores + belief
    by_stmt = defaultdict(lambda: {"belief": None, "na_scores": [], "all_scores": []})
    scored_stmts = set()
    for d in rows.values():
        sh = d["stmt_hash"]
        if isinstance(d.get("score"), (int, float)):
            scored_stmts.add(sh)
            by_stmt[sh]["all_scores"].append(d["score"])
        ev, reasoning = split_preview(d.get("raw_text_preview"), d.get("text_len") or 0)
        b = classify(d, ev, reasoning)
        if b in NON_ARTIFACT and isinstance(d.get("score"), (int, float)):
            by_stmt[sh]["na_scores"].append(d["score"])
        if by_stmt[sh]["belief"] is None and isinstance(d.get("belief"), (int, float)):
            by_stmt[sh]["belief"] = d["belief"]

    # ---- FILTERED set: statements with >=1 non-artifact scored evidence ----
    filt = []  # (ras_belief, our_mean, our_noisyor)
    for sh, agg in by_stmt.items():
        na = agg["na_scores"]
        if not na or agg["belief"] is None:
            continue
        our_mean = sum(na) / len(na)
        prod = 1.0
        for s in na:
            prod *= (1.0 - s)
        our_nor = 1.0 - prod
        filt.append((agg["belief"], our_mean, our_nor))

    n_filt = len(filt)
    ras = [r[0] for r in filt]
    omean = [r[1] for r in filt]
    onor = [r[2] for r in filt]

    r_mean = pearson(ras, omean)
    r_nor = pearson(ras, onor)
    min_ras = min(ras)
    max_ras = max(ras)

    print(f"unique rows (dedup):          {n_unique}   (brief says 47,434)")
    print(f"scored statements:            {len(scored_stmts)}   (brief says 8,716)")
    print()
    print(f"=== FILTER: >=1 non-artifact ({sorted(NON_ARTIFACT)}) evidence ===")
    print(f"n filtered statements:        new={n_filt}        old(snapshot)=4,830")
    print(f"Pearson(RasMachine, our-mean):    new={r_mean:+.4f}   old(SVG/figcap)=+0.155")
    print(f"Pearson(RasMachine, noisy-OR):    new={r_nor:+.4f}   old(methodology)=+0.212")
    print(f"min RasMachine belief:        new={min_ras:.4f}    old claim='no statement below 0.86'")
    print(f"max RasMachine belief:        new={max_ras:.4f}")

    # bimodality at extremes (our mean): mass in [0,0.06) and [0.94,1.0]
    lo = sum(1 for v in omean if v < 0.06)
    hi = sum(1 for v in omean if v >= 0.94)
    mid = n_filt - lo - hi
    print()
    print(f"our-mean bimodality:  lo[0,0.06)={lo} ({100*lo/n_filt:.1f}%)  "
          f"hi[0.94,1.0]={hi} ({100*hi/n_filt:.1f}%)  mid={mid} ({100*mid/n_filt:.1f}%)")

    # ---- AGREEMENT TABLE: bin each predictor at 0.50 ----
    # rows: RasMachine >=0.50 / <0.50 ; cols: ours >=0.50 / <0.50
    rge_oge = sum(1 for b, m, _ in filt if b >= 0.50 and m >= 0.50)
    rge_olt = sum(1 for b, m, _ in filt if b >= 0.50 and m < 0.50)
    rlt_oge = sum(1 for b, m, _ in filt if b < 0.50 and m >= 0.50)
    rlt_olt = sum(1 for b, m, _ in filt if b < 0.50 and m < 0.50)
    rge_tot = rge_oge + rge_olt
    rlt_tot = rlt_oge + rlt_olt
    print()
    print("=== AGREEMENT TABLE (threshold 0.50) ===")
    print(f"                       ours>=0.50   ours<0.50   total")
    print(f"RasMachine>=0.50   new={rge_oge:>8} {rge_olt:>11} {rge_tot:>8}   old=3,053 / 1,559 / 4,612")
    print(f"RasMachine<0.50    new={rlt_oge:>8} {rlt_olt:>11} {rlt_tot:>8}   old=0 / 0 / 0")
    pct_olt = 100 * rge_olt / rge_tot if rge_tot else 0
    print(f"'confidently high to RasM, low to ours': new={rge_olt} ({pct_olt:.1f}%)   old=1,559 (33.8%)")

    # ---- HEXBIN GEOMETRY: X bins zoomed to [0.857,1.0] in 4 cols, Y bins 0..1 in 18 rows ----
    # X edges from the SVG ticks: cols span [0.857,1.0] -> 4 equal cols.
    # Recreate the exact bin layout used by the snapshot to compare cell counts.
    print()
    print("=== HEXBIN cell counts (4 X-cols x 18 Y-rows) ===")
    XLO, XHI, NX = 0.857, 1.0, 4
    NY = 18
    xedges = [XLO + (XHI - XLO) * i / NX for i in range(NX + 1)]
    yedges = [i / NY for i in range(NY + 1)]
    grid = defaultdict(int)
    for b, m, _ in filt:
        # X col
        if b < XLO:
            cx = 0
        else:
            cx = min(int((b - XLO) / (XHI - XLO) * NX), NX - 1)
        cy = min(int(m * NY), NY - 1)
        grid[(cx, cy)] += 1
    # marginal of ours (Y)
    ymarg = [0] * NY
    for (cx, cy), c in grid.items():
        ymarg[cy] += c
    # print non-trivial cells per column for the SVG <title> updates
    # column data-cx maps: col0->24, col1->25, col2->26, col3->27 in the SVG
    CX_MAP = {0: 24, 1: 25, 2: 26, 3: 27}
    XLABEL = {0: "[0.857, 0.893]", 1: "[0.893, 0.929]",
              2: "[0.929, 0.964]", 3: "[0.964, 1.000]"}
    YLABEL = []
    for i in range(NY):
        lo_e, hi_e = yedges[i], yedges[i + 1]
        YLABEL.append(f"[{lo_e:.2f}, {hi_e:.2f}]")
    for cx in range(NX):
        print(f"\n-- column cx={CX_MAP[cx]} RasMachine in {XLABEL[cx]} --")
        for cy in range(NY):
            c = grid.get((cx, cy), 0)
            if c:
                print(f"   ours {YLABEL[cy]:<16} cy={cy:<2} n={c}")
    print("\n-- Y marginal (ours) bars --")
    for cy in range(NY):
        print(f"   ours {YLABEL[cy]:<16} cy={cy:<2} n={ymarg[cy]}")

    # the two labeled clusters in the SVG
    print()
    print("=== labeled clusters in SVG ===")
    print(f"  disagreement (cx=24, cy=0)  RasM[0.857,0.893] ours[0.00,0.06]: new={grid.get((0,0),0)}  old=723")
    print(f"  agreement    (cx=24, cy=17) RasM[0.857,0.893] ours[0.94,1.00]: new={grid.get((0,17),0)}  old=1000")

    # fraction of unit interval RasMachine uses (label '14%')
    frac = (1.0 - min_ras) * 100
    print()
    print(f"RasMachine uses rightmost {frac:.0f}% of unit interval  (old label '14%')")


if __name__ == "__main__":
    main()
