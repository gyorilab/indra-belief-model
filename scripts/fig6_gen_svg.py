#!/usr/bin/env python3
"""Regenerate Figure 6 with two toggleable views (the 'confidence-collapse lever').

  view "score" : the 7 score-value bars (verdict x confidence -> score), replicating the
                 original layout, with per-bar data hooks for the calibration card.
  view "conf"  : the confidence marginals (high / medium / low / null) — same log axis —
                 showing ~98% of rows are 'high', i.e. the confidence axis is degenerate,
                 which is *why* the score collapses to two values.

Counts recomputed from the jsonl. Geometry replicates the existing SVG exactly:
  log scale height(c, gmax) = log10(1+c)/log10(1+gmax) * 230, baseline y=280, top y=50.
"""
import json, math
from collections import Counter

SRC = "data/results/rasmachine_mono_gemma_remote_direct.jsonl"
GRID = {('correct', 'high'): 0.95, ('correct', 'medium'): 0.80, ('correct', 'low'): 0.65,
        ('incorrect', 'low'): 0.35, ('incorrect', 'medium'): 0.20, ('incorrect', 'high'): 0.05}
FF_SANS = "-apple-system,BlinkMacSystemFont,system-ui,sans-serif"
FF_MONO = "ui-monospace,SF Mono,monospace"
BASE_Y, TOP_RANGE = 280.0, 230.0
GOLD92, GOLD55, GOLD35 = "rgb(168 130 80 / 0.92)", "rgb(168 130 80 / 0.55)", "rgb(168 130 80 / 0.35)"
PLUM = "rgb(130 110 138 / 0.7)"


def load():
    rows = {}
    with open(SRC) as f:
        for line in f:
            d = json.loads(line)
            rows[(d["stmt_i"], d["evidence_i"])] = d
    score = Counter(); conf = Counter()
    for d in rows.values():
        v, c = d.get("verdict"), d.get("confidence")
        if v is None or c is None:
            score[0.50] += 1; conf["null"] += 1
        else:
            score[GRID.get((v, c), 0.50)] += 1; conf[c] += 1
    return len(rows), score, conf


def main():
    total, score, conf = load()
    S = []; a = S.append

    def hy(c, gmax):
        return math.log10(1 + c) / math.log10(1 + gmax) * TOP_RANGE

    def comma(n):
        return f"{n:,}"

    def yaxis(gmax, ticks):
        out = []
        for t in ticks:
            y = BASE_Y - hy(t, gmax)
            out.append(f'<line x1="66" y1="{y:.2f}" x2="70" y2="{y:.2f}" stroke="#b8b6b0" stroke-width="0.8"/>')
            out.append(f'<text x="63" y="{y + 3.2:.2f}" text-anchor="end" font-family="{FF_MONO}" font-size="9" font-feature-settings="\'tnum\'" fill="#8a8a8a">{comma(t)}</text>')
            out.append(f'<line x1="70" y1="{y:.2f}" x2="692" y2="{y:.2f}" stroke="#d8d6d0" stroke-width="0.4" stroke-dasharray="1 3" opacity="0.6"/>')
        out.append(f'<line x1="70" y1="{BASE_Y}" x2="692" y2="{BASE_Y}" stroke="#1a1a1a" stroke-width="1"/>')
        return "\n".join(out)

    def bar(cx, c, gmax, fill, stroke, sw, hooks, title, top_label_weight="500"):
        h = hy(c, gmax); y = BASE_Y - h; x = cx - 18
        hk = "".join(f' {k}="{v}"' for k, v in hooks.items())
        out = [f'<rect class="fig6-bar"{hk} x="{x:.1f}" y="{y:.2f}" width="36" height="{h:.2f}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"><title>{title}</title></rect>']
        out.append(f'<text x="{cx:.1f}" y="{y - 5:.2f}" text-anchor="middle" font-family="{FF_MONO}" font-size="10" font-feature-settings="\'tnum\'" fill="#1a1a1a" font-weight="{top_label_weight}">{comma(c)}</text>')
        return "\n".join(out)

    def xlabel(cx, lines):
        out = [f'<line x1="{cx:.1f}" y1="{BASE_Y}" x2="{cx:.1f}" y2="{BASE_Y+4}" stroke="#1a1a1a" stroke-width="1"/>']
        ys = [296, 310, 322]
        for i, (txt, fnt, fill, style) in enumerate(lines):
            st = f' font-style="{style}"' if style else ""
            out.append(f'<text x="{cx:.1f}" y="{ys[i]}" text-anchor="middle" font-family="{fnt}" font-size="{"10.5" if i==0 else "9"}" {"font-weight=\'700\'" if i==0 else ""} fill="{fill}"{st}>{txt}</text>')
        return "\n".join(out)

    a('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 720 380" class="fig6-svg" role="img" aria-label="Score distribution and confidence mix (log scale)">')
    a(f'<text x="20" y="165.0" text-anchor="middle" font-family="{FF_SANS}" font-size="11" fill="#1a1a1a" font-weight="600" transform="rotate(-90 20 165.0)">number of rows (log)</text>')

    # ---- view: score (7 bars) ----
    a('<g class="fig6-view" data-view="score">')
    a(f'<text x="70" y="14" font-family="{FF_SANS}" font-size="10" fill="#8a8a8a" font-style="italic" letter-spacing="0.04em">distribution of the score field · all {comma(total)} scored rows · y is log-scaled</text>')
    a(f'<text x="692" y="14" text-anchor="end" font-family="{FF_SANS}" font-size="11" font-weight="700" fill="#6b4423">98.3% of rows in 2 of 7 buckets</text>')
    a(yaxis(26000, [1, 10, 100, 1000, 10000, 26000]))
    SCORE_BARS = [
        (101.1, score[0.05], "0.05", "incorrect", "high", GOLD92, "#6b4423", 0.8, "600"),
        (194.4, score[0.20], "0.2", "incorrect", "medium", GOLD55, "#6b4423", 0.5, "500"),
        (287.7, score[0.35], "0.35", "incorrect", "low", GOLD55, "#6b4423", 0.5, "500"),
        (381.0, score[0.50], "0.5", "null", "parse fail", PLUM, "#7a6a82", 0.5, "500"),
        (474.3, score[0.65], "0.65", "correct", "low", GOLD35, "#6b4423", 0.5, "500"),
        (567.6, score[0.80], "0.8", "correct", "medium", GOLD55, "#6b4423", 0.5, "500"),
        (660.9, score[0.95], "0.95", "correct", "high", GOLD92, "#6b4423", 0.8, "600"),
    ]
    for cx, c, sc, v, cf, fill, stroke, sw, w in SCORE_BARS:
        pct = round(1000 * c / total) / 10
        hooks = {"data-score": sc, "data-verdict": v, "data-confidence": cf, "data-count": str(c), "data-pct": str(pct)}
        a(bar(cx, c, 26000, fill, stroke, sw, hooks, f"score = {sc} · {v}/{cf} · n = {comma(c)} ({pct}%)", w))
        vfill = "#7a6a82" if v == "null" else "#5a5a5a"
        a(xlabel(cx, [(sc, FF_MONO, "#1a1a1a", None), (v, FF_SANS, vfill, None), (cf, FF_SANS, "#8a8a8a", "italic")]))
    a(f'<text x="381.0" y="358" text-anchor="middle" font-family="{FF_SANS}" font-size="11" fill="#5a5a5a" font-style="italic">score · verdict × confidence</text>')
    a('</g>')

    # ---- view: conf (4 bars: high/medium/low/null) ----
    a('<g class="fig6-view" data-view="conf" style="display:none">')
    gmaxc = conf["high"]
    a(f'<text x="70" y="14" font-family="{FF_SANS}" font-size="10" fill="#8a8a8a" font-style="italic" letter-spacing="0.04em">the same rows grouped by <tspan font-weight="600">confidence</tspan> · y is log-scaled</text>')
    a(f'<text x="692" y="14" text-anchor="end" font-family="{FF_SANS}" font-size="11" font-weight="700" fill="#6b4423">{round(1000*conf["high"]/total)/10}% of rows say “high”</text>')
    a(yaxis(gmaxc, [1, 10, 100, 1000, 10000, gmaxc]))
    CONF_BARS = [
        (160.0, conf["high"], "high", GOLD92, "#6b4423", 0.8, "600"),
        (320.0, conf["medium"], "medium", GOLD55, "#6b4423", 0.5, "500"),
        (480.0, conf["low"], "low", GOLD35, "#6b4423", 0.5, "500"),
        (614.0, conf["null"], "null", PLUM, "#7a6a82", 0.5, "500"),
    ]
    for cx, c, lbl, fill, stroke, sw, w in CONF_BARS:
        pct = round(1000 * c / total) / 10
        a(bar(cx, c, gmaxc, fill, stroke, sw, {"data-confbar": lbl, "data-count": str(c)}, f"confidence = {lbl} · n = {comma(c)} ({pct}%)", w))
        lf = "#7a6a82" if lbl == "null" else "#1a1a1a"
        a(xlabel(cx, [(lbl, FF_SANS, lf, None), (f"{pct}%", FF_MONO, "#5a5a5a", None), ("", FF_SANS, "#8a8a8a", None)]))
    a(f'<text x="381.0" y="358" text-anchor="middle" font-family="{FF_SANS}" font-size="11" fill="#5a5a5a" font-style="italic">confidence level · the axis the score depends on</text>')
    a('</g>')

    a('</svg>')
    out = "\n".join(S) + "\n"
    with open("/tmp/fig6.svg", "w") as f:
        f.write(out)
    assert sum(score.values()) == total == sum(conf.values()), "count mismatch"
    print(f"total {total} | score cells {dict(sorted(score.items()))} | conf {dict(conf)}")
    print(f"wrote /tmp/fig6.svg ({len(out)} bytes)")


if __name__ == "__main__":
    main()
