#!/usr/bin/env python3
"""Recompute Figure-6 (score distribution + verdict x confidence grid) numbers
from the complete rasmachine scoring run, and print old-vs-new for every value
the report cites.

score is a deterministic map from (verdict, confidence):
  correct/high   = 0.95
  correct/medium = 0.80
  correct/low    = 0.65
  incorrect/low  = 0.35
  incorrect/medium = 0.20
  incorrect/high = 0.05
  null/parse-fail = 0.50

Dedup by (stmt_i, evidence_i), keep last. Cross-tab scored rows by
(verdict, confidence) -> 7 score values. Also reproduces the §6 confidence
table (high/medium/low counts + %) which uses non-null-confidence denominator.
"""
import json
from collections import Counter

SRC = "data/results/rasmachine_mono_gemma_remote_direct.jsonl"

# (verdict, confidence) -> score value, in the SVG's left-to-right order
SCORE_OF = {
    ("incorrect", "high"):   0.05,
    ("incorrect", "medium"): 0.20,
    ("incorrect", "low"):    0.35,
    (None, None):            0.50,   # null / parse fail
    ("correct", "low"):      0.65,
    ("correct", "medium"):   0.80,
    ("correct", "high"):     0.95,
}
# canonical bar order (matches SVG x positions)
BAR_ORDER = [
    (0.05, "incorrect", "high"),
    (0.20, "incorrect", "medium"),
    (0.35, "incorrect", "low"),
    (0.50, "null",      "parse fail"),
    (0.65, "correct",   "low"),
    (0.80, "correct",   "medium"),
    (0.95, "correct",   "high"),
]

# snapshot (91%) values baked into the report
SNAP = {
    0.05: 26000, 0.20: 289, 0.35: 119, 0.50: 209,
    0.65: 2, 0.80: 197, 0.95: 20618,
}
SNAP_CONF = {"high": 42067, "medium": 481, "low": 72}

def score_for(d):
    v = d.get("verdict")
    c = d.get("confidence")
    if v is None:
        return 0.50
    return SCORE_OF.get((v, c))

def main():
    rows = {}
    with open(SRC) as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            rows[(d["stmt_i"], d["evidence_i"])] = d

    total = len(rows)

    # cross-tab score value -> count
    score_ct = Counter()
    conf_ct = Counter()        # confidence (incl. None) over all rows
    vc_ct = Counter()          # (verdict, confidence)
    score_field_mismatch = 0
    for d in rows.values():
        s = score_for(d)
        # sanity: compare derived score against stored score field when present
        sf = d.get("score")
        if sf is not None and s is not None and abs(sf - s) > 1e-9:
            score_field_mismatch += 1
        score_ct[s] += 1
        conf_ct[d.get("confidence")] += 1
        vc_ct[(d.get("verdict"), d.get("confidence"))] += 1

    print(f"total unique (stmt_i,evidence_i) rows: {total}")
    print(f"derived-vs-stored score-field mismatches: {score_field_mismatch}\n")

    # ---- 7 score-value cells ----
    print(f"{'score':>6} {'verdict':>10} {'conf':>7} {'new_n':>8} {'snap_n':>8} {'new_%':>7}")
    for sc, vlabel, clabel in BAR_ORDER:
        n = score_ct.get(sc, 0)
        pct = 100 * n / total
        print(f"{sc:>6} {vlabel:>10} {clabel:>7} {n:>8} {SNAP.get(sc):>8} {pct:>6.1f}%")

    # explicit (verdict, confidence) raw cross-tab for transparency
    print("\nraw (verdict, confidence) cross-tab:")
    for key in sorted(vc_ct, key=lambda k: (str(k[0]), str(k[1]))):
        print(f"  {str(key):<28} {vc_ct[key]:>8}")

    # ---- two dominant cells ----
    n_05 = score_ct.get(0.05, 0)
    n_95 = score_ct.get(0.95, 0)
    big2 = n_05 + n_95
    print(f"\ntwo big cells: incorrect/high(0.05)={n_05} (snap 26,000), "
          f"correct/high(0.95)={n_95} (snap 20,618)")
    print(f"  big-2 sum = {big2}  -> {100*big2/total:.1f}% of population "
          f"(snap 98.3%)")

    # ---- other five cells ----
    other5 = total - big2
    print(f"  other five cells together = {other5} (snap 607)")

    # ---- high-confidence share (denominator = rows with non-null confidence) ----
    n_high = conf_ct.get("high", 0)
    n_med = conf_ct.get("medium", 0)
    n_low = conf_ct.get("low", 0)
    n_none = conf_ct.get(None, 0)
    nonnull = n_high + n_med + n_low
    print(f"\nconfidence counts: high={n_high} medium={n_med} low={n_low} "
          f"null={n_none}  (non-null denom={nonnull}, total={total})")
    print(f"  high-conf % (denom non-null) = {100*n_high/nonnull:.1f}% (snap 98.7%)")
    print(f"  high-conf % (denom total)    = {100*n_high/total:.1f}%")

    # §6 table uses non-null denominator
    print("\n§6 confidence table (denominator = non-null confidence rows):")
    for label, n in [("high", n_high), ("medium", n_med), ("low", n_low)]:
        print(f"  {label:<7} {n:>8} ({SNAP_CONF[label]:>6} snap)  "
              f"{100*n/nonnull:.1f}%")

    # ---- log-scale bar geometry (to regenerate SVG bar heights if needed) ----
    # From the SVG axis ticks: y maps log10(count) linearly.
    # tick anchors (count -> y of baseline-of-text -> the gridline y):
    #   1 -> 264.32, 10 -> 225.75, 100 -> 175.58, 1000 -> 123.69,
    #   10000 -> 71.62, 26000 -> 50.00 ; baseline y=280.
    # Fit y = a + b*log10(n) using the clean decade ticks (10..10000).
    import math
    pts = [(1, 264.32), (10, 225.75), (100, 175.58),
           (1000, 123.69), (10000, 71.62)]
    # linear regression on (log10 n, y)
    xs = [math.log10(n) for n, _ in pts]
    ys = [y for _, y in pts]
    mx = sum(xs) / len(xs); my = sum(ys) / len(ys)
    b = sum((x-mx)*(y-my) for x, y in zip(xs, ys)) / sum((x-mx)**2 for x in xs)
    a = my - b*mx
    print(f"\nlog-scale fit: y = {a:.3f} + {b:.3f}*log10(n)  (baseline y=280)")
    print("new bar geometry (x kept from original SVG):")
    XPOS = {0.05: 83.1, 0.20: 176.4, 0.35: 269.7, 0.50: 363.0,
            0.65: 456.3, 0.80: 549.6, 0.95: 642.9}
    for sc, vlabel, clabel in BAR_ORDER:
        n = score_ct.get(sc, 0)
        y = a + b*math.log10(max(n, 1))
        h = 280 - y
        print(f"  score={sc:<5} n={n:<7} y={y:7.2f} h={h:7.2f}  "
              f"(label y={y-5:.1f})")

if __name__ == "__main__":
    main()
