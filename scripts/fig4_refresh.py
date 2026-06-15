#!/usr/bin/env python3
"""Recompute Figure 4 (per-source belief distributions) from the COMPLETE
rasmachine scoring run.

Figure 4 shows, for five sources (the five that appear in the SVG, ordered by
our-mean descending), two stacked histograms over [0,1]:
  - RasMachine per-statement belief  (field `belief`, constant within stmt_hash)
  - OUR per-statement belief = arithmetic mean of `score` over the statement's
    scored evidences (mean aggregator)

A statement's source = the source_api of its evidences. Bars are log(1+count)
scaled to a shared global_max; a vertical tick marks each predictor's mean.

This script dedups by (stmt_i, evidence_i) keeping LAST, groups by stmt_hash,
and reproduces every number baked into the figure: per-source n, RasMachine
mean, our mean, per-bin counts (the <title> tooltips), the global_max / tick
values, and the log(1+count) bar geometry. Prints old-vs-new for each.
"""
import json
import math
import sys
from collections import defaultdict

sys.path.insert(0, "scripts")
from fig1_drilldown_data import classify, split_preview  # noqa: E402
from fig_utils import SRC, NON_ARTIFACT  # noqa: E402, F401

# Five sources shown in the SVG, in the report's display order (our-mean desc).
# We recompute their order too and confirm it matches.
SVG_SOURCES = ["bel", "reach", "signor", "sparser", "eidos"]
NBINS = 20  # 20 bins of width 0.05 over [0,1]

# ---- geometry constants copied verbatim from the existing SVG ----
PLOT_LEFT = 108.0
PLOT_W = 502.0
PLOT_H = 32.0
BAR_W = 24.60
BIN_PX = PLOT_W / NBINS  # 25.1


def pearson(xs, ys):
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx == 0 or syy == 0:
        return float("nan")
    return sxy / math.sqrt(sxx * syy)


def bin_index(x):
    """Bin a value in [0,1] into 0..19; clamp 1.0 into the last bin."""
    i = int(x * NBINS)
    if i >= NBINS:
        i = NBINS - 1
    if i < 0:
        i = 0
    return i


def main():
    # ---- load + dedup (keep last) ----
    rows = {}
    with open(SRC) as f:
        for line in f:
            d = json.loads(line)
            rows[(d["stmt_i"], d["evidence_i"])] = d

    # ---- group by stmt_hash ----
    # per statement: source_api (mode of its evidences -> in practice all same),
    # RasMachine belief (constant), and our mean = mean(score) over scored evs.
    stmt_belief = {}          # stmt_hash -> rasmachine belief
    stmt_scores = defaultdict(list)   # stmt_hash -> [score,...] (non-null)
    stmt_source = {}          # stmt_hash -> source_api
    for d in rows.values():
        h = d["stmt_hash"]
        if h not in stmt_belief and isinstance(d.get("belief"), (int, float)):
            stmt_belief[h] = d["belief"]
        # a statement's source is its evidences' source_api
        if h not in stmt_source:
            stmt_source[h] = d.get("source_api")
        sc = d.get("score")
        if isinstance(sc, (int, float)):
            stmt_scores[h].append(sc)

    # A "scored statement" has >=1 evidence with a numeric score.
    scored_stmts = [h for h in stmt_belief if stmt_scores.get(h)]
    print(f"unique deduped rows: {len(rows)}")
    print(f"statements with a belief: {len(stmt_belief)}")
    print(f"scored statements (>=1 numeric score): {len(scored_stmts)}\n")

    # ---- per-source aggregation ----
    # source -> list of (rasm_belief, our_mean)
    per_src = defaultdict(list)
    for h in scored_stmts:
        src = stmt_source[h]
        rasm = stmt_belief[h]
        our = sum(stmt_scores[h]) / len(stmt_scores[h])
        per_src[src].append((rasm, our))

    # compute our-mean per source and the descending order
    src_stats = {}
    for src, pairs in per_src.items():
        n = len(pairs)
        rasm_mean = sum(p[0] for p in pairs) / n
        our_mean = sum(p[1] for p in pairs) / n
        src_stats[src] = {"n": n, "rasm_mean": rasm_mean, "our_mean": our_mean}

    order_desc = sorted(src_stats, key=lambda s: -src_stats[s]["our_mean"])
    print("ALL sources by our-mean (descending):")
    for s in order_desc:
        st = src_stats[s]
        print(f"  {s:<10} n={st['n']:>5}  RasM mean={st['rasm_mean']:.4f}  our mean={st['our_mean']:.4f}")
    print()

    # ---- histograms + bar geometry for the five SVG sources ----
    # First compute global_max = max bin count across the displayed sources/both
    # predictors (this drives the log scale and the legend ticks).
    hist = {}  # src -> {"rasm": [20], "our": [20]}
    for src in SVG_SOURCES:
        rb = [0] * NBINS
        ob = [0] * NBINS
        for rasm, our in per_src[src]:
            rb[bin_index(rasm)] += 1
            ob[bin_index(our)] += 1
        hist[src] = {"rasm": rb, "our": ob}

    global_max = 0
    for src in SVG_SOURCES:
        global_max = max(global_max, max(hist[src]["rasm"]), max(hist[src]["our"]))
    print(f"global_max (max bin count over displayed sources/predictors): {global_max}")
    print(f"  OLD global_max = 1139\n")

    logden = math.log(1 + global_max)

    def bar_geom(count, plot_top):
        h = math.log(1 + count) / logden * PLOT_H
        y = (plot_top + PLOT_H) - h
        return round(y, 2), round(h, 2)

    # ---- print per-source detail (means + every nonzero bin = a <title>) ----
    BIN_EDGES = [(round(i * 0.05, 2), round((i + 1) * 0.05, 2)) for i in range(NBINS)]
    OLD = {  # snapshot values baked into the SVG, for old-vs-new
        "bel":     {"n": 80,   "rasm": 0.925, "our": 0.739},
        "reach":   {"n": 1192, "rasm": 0.942, "our": 0.735},
        "signor":  {"n": 193,  "rasm": 0.964, "our": 0.675},
        "sparser": {"n": 2872, "rasm": 0.925, "our": 0.566},
        "eidos":   {"n": 475,  "rasm": 0.885, "our": 0.428},
    }
    print("=" * 78)
    print("PER-SOURCE (old snapshot -> new complete):")
    print("=" * 78)
    for src in SVG_SOURCES:
        st = src_stats[src]
        old = OLD[src]
        print(f"\n### {src}")
        print(f"  n       : {old['n']:>6} -> {st['n']:>6}")
        print(f"  RasM mean: {old['rasm']:.3f} -> {st['rasm_mean']:.3f}")
        print(f"  our  mean: {old['our']:.3f} -> {st['our_mean']:.3f}")
        # mean-tick x positions
        rasm_tick_x = PLOT_LEFT + st["rasm_mean"] * PLOT_W
        our_tick_x = PLOT_LEFT + st["our_mean"] * PLOT_W
        print(f"  RasM tick x = {rasm_tick_x:.2f}   our tick x = {our_tick_x:.2f}")
        print(f"  RasMachine nonzero bins (bin -> n):")
        for i in range(NBINS):
            c = hist[src]["rasm"][i]
            if c:
                lo, hi = BIN_EDGES[i]
                print(f"    [{lo:.2f}, {hi:.2f}] n = {c}")
        print(f"  ours nonzero bins (bin -> n):")
        for i in range(NBINS):
            c = hist[src]["our"][i]
            if c:
                lo, hi = BIN_EDGES[i]
                print(f"    [{lo:.2f}, {hi:.2f}] n = {c}")

    # ---- figcaption range + descending sequence ----
    rasm_means = [src_stats[s]["rasm_mean"] for s in SVG_SOURCES]
    print("\n" + "=" * 78)
    print("FIGCAPTION SUMMARY")
    print("=" * 78)
    print(f"RasMachine means across the 5 shown sources: "
          f"min={min(rasm_means):.3f} max={max(rasm_means):.3f}  (OLD range 0.89-0.96)")
    our_seq = [src_stats[s]["our_mean"] for s in SVG_SOURCES]
    print("our-mean descending sequence (SVG source order "
          f"{SVG_SOURCES}):")
    print("  OLD: 0.74 -> 0.74 -> 0.68 -> 0.57 -> 0.43")
    print("  NEW: " + " -> ".join(f"{m:.2f}" for m in our_seq))
    print(f"display order matches our-mean-desc? "
          f"{SVG_SOURCES == [s for s in order_desc if s in SVG_SOURCES]}")

    # ---- legend tick values ----
    # SVG legend ticks: [1, 10, 100, 1000, global_max]
    print(f"\nlegend ticks OLD: [1, 10, 100, 1000, 1139]")
    print(f"legend ticks NEW: [1, 10, 100, 1000, {global_max}]")

    # =====================================================================
    # SUB-TABLE (section 4 intro, the "n stmts / RasM mean / Ours mean / Δ /
    # Pearson" table) — this is the BUCKET-AWARE filtered set (same filter as
    # Figs 1-3): keep a statement with >=1 non-artifact scored evidence, and
    # aggregate OUR belief over its non-artifact evidence only. NOTE this is a
    # DIFFERENT statement set from the figure histograms above (which use ALL
    # scored evidence). The table rows are reach / biogrid / signor / bel /
    # sparser / eidos.
    # =====================================================================
    by_stmt = defaultdict(lambda: {"belief": None, "na": [], "src": None})
    for d in rows.values():
        sh = d["stmt_hash"]
        if by_stmt[sh]["src"] is None:
            by_stmt[sh]["src"] = d.get("source_api")
        if by_stmt[sh]["belief"] is None and isinstance(d.get("belief"), (int, float)):
            by_stmt[sh]["belief"] = d["belief"]
        ev, reasoning = split_preview(d.get("raw_text_preview"), d.get("text_len") or 0)
        b = classify(d, ev, reasoning)
        if b in NON_ARTIFACT and isinstance(d.get("score"), (int, float)):
            by_stmt[sh]["na"].append(d["score"])

    tbl = defaultdict(list)  # src -> [(ras, our_mean)]
    for sh, agg in by_stmt.items():
        if not agg["na"] or agg["belief"] is None:
            continue
        om = sum(agg["na"]) / len(agg["na"])
        tbl[agg["src"]].append((agg["belief"], om))

    OLD_TBL = {  # (n, rasm, ours, delta, pearson) as printed in the report table
        "reach":   (1024, 0.936, 0.735, "−0.20", "+0.05"),
        "biogrid": (143,  0.998, 0.772, "−0.23", "+0.13"),
        "signor":  (150,  0.959, 0.642, "−0.32", "+0.18"),
        "bel":     (74,   0.927, 0.712, "−0.22", "+0.03"),
        "sparser": (2786, 0.924, 0.561, "−0.36", "+0.10"),
        "eidos":   (397,  0.885, 0.421, "−0.46", "+0.08"),
    }
    print("\n" + "=" * 78)
    print("SUB-TABLE (bucket-aware filtered set; OLD -> NEW):")
    print("=" * 78)
    print(f"{'source':<10}{'n':>6}{'n_new':>7}  {'RasM':>6}{'RasM_new':>9}  "
          f"{'ours':>6}{'ours_new':>9}  {'Δnew':>7}  {'r_old':>6}{'r_new':>7}")
    for src in ["reach", "biogrid", "signor", "bel", "sparser", "eidos"]:
        pairs = tbl.get(src, [])
        if not pairs:
            print(f"{src:<10}  (no rows)")
            continue
        n = len(pairs)
        rm = sum(p[0] for p in pairs) / n
        om = sum(p[1] for p in pairs) / n
        r = pearson([p[0] for p in pairs], [p[1] for p in pairs])
        o = OLD_TBL[src]
        print(f"{src:<10}{o[0]:>6}{n:>7}  {o[1]:>6.3f}{rm:>9.3f}  "
              f"{o[2]:>6.3f}{om:>9.3f}  {om - rm:>7.2f}  {o[4]:>6}{r:>+7.2f}")


if __name__ == "__main__":
    main()
