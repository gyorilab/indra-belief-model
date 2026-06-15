#!/usr/bin/env python3
"""Regenerate the Figure-4 SVG from the COMPLETE run, preserving the EXACT
viewBox/dimensions/colours/fonts/classes/interactivity of the original.

The original SVG geometry was reverse-engineered and reproduced verbatim:
  - viewBox 0 0 720 790, class fig4-svg, same header legend block.
  - five source panels, 140px vertical pitch, RasMachine row above ours row.
  - log(1+count) bars vs a shared global_max; 20 bins of width 0.05 over [0,1].
  - mean tick = line + circle at x = 108 + mean*502.
  - per-bar <title> tooltips, fills #7a8a9a (RasMachine) / #a08560 (ours)
    (matched by the report CSS hover rule — must stay byte-identical).

Sources shown: the same five as the snapshot (bel, reach, signor, sparser,
eidos), re-sorted by our-mean DESCENDING (the panel order is data-driven and
the subtitle says "ordered by our mean, descending"). OUR per-statement belief
= mean of `score` over ALL the statement's scored evidences (the figure's
definition, per the brief).
"""
import json
import math
from collections import defaultdict

SRC = "data/results/rasmachine_mono_gemma_remote_direct.jsonl"
SHOWN = ["bel", "reach", "signor", "sparser", "eidos"]
NBINS = 20

PLOT_LEFT = 108.0
PLOT_RIGHT = 610.0
PLOT_W = 502.0
PLOT_H = 32.0
BAR_W = 24.60
BIN_PX = PLOT_W / NBINS  # 25.1
FF_SANS = "-apple-system,BlinkMacSystemFont,system-ui,sans-serif"
FF_MONO = "ui-monospace,SF Mono,monospace"


def bin_index(x):
    i = int(x * NBINS)
    return min(max(i, 0), NBINS - 1)


def load():
    rows = {}
    with open(SRC) as f:
        for line in f:
            d = json.loads(line)
            rows[(d["stmt_i"], d["evidence_i"])] = d
    belief, scores, source = {}, defaultdict(list), {}
    for d in rows.values():
        h = d["stmt_hash"]
        if h not in belief and isinstance(d.get("belief"), (int, float)):
            belief[h] = d["belief"]
        if h not in source:
            source[h] = d.get("source_api")
        if isinstance(d.get("score"), (int, float)):
            scores[h].append(d["score"])
    per_src = defaultdict(list)
    for h, b in belief.items():
        if not scores.get(h):
            continue
        per_src[source[h]].append((b, sum(scores[h]) / len(scores[h])))
    return per_src


def fmt(v):
    """Trim trailing-zero floats the way the SVG does (e.g. 25.1 -> '25.10')."""
    return f"{v:.2f}"


def main():
    per_src = load()

    stats = {}
    hist = {}
    for src in SHOWN:
        pairs = per_src[src]
        n = len(pairs)
        rm = sum(p[0] for p in pairs) / n
        om = sum(p[1] for p in pairs) / n
        rb = [0] * NBINS
        ob = [0] * NBINS
        for b, o in pairs:
            rb[bin_index(b)] += 1
            ob[bin_index(o)] += 1
        stats[src] = {"n": n, "rasm": rm, "our": om}
        hist[src] = {"rasm": rb, "our": ob}

    order = sorted(SHOWN, key=lambda s: -stats[s]["our"])
    gmax = 0
    for src in SHOWN:
        gmax = max(gmax, max(hist[src]["rasm"]), max(hist[src]["our"]))
    den = math.log(1 + gmax)

    def yh(count, top):
        h = math.log(1 + count) / den * PLOT_H
        return (top + PLOT_H) - h, h

    def comma(n):
        return f"{n:,}"

    S = []
    a = S.append

    # --- legend / header (geometry identical to original; ticks use gmax) ---
    a('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 720 790" class="fig4-svg" role="img" aria-label="Per-source belief distributions, log-scaled">')
    a(f'<text x="108" y="14" font-family="{FF_SANS}" font-size="10" fill="#8a8a8a" font-style="italic" letter-spacing="0.04em">distributions of belief by source · y is log-scaled · ordered by our mean, descending</text>')
    a(f'<text x="104" y="18" text-anchor="end" font-family="{FF_SANS}" font-size="10" fill="#1a1a1a" font-weight="600">log scale →</text>')
    a(f'<text x="104" y="30" text-anchor="end" font-family="{FF_SANS}" font-size="9" fill="#8a8a8a" font-style="italic">how to read</text>')
    a('<line x1="108" y1="56" x2="386" y2="56" stroke="#5a5a5a" stroke-width="0.7"/>')
    legend = [(1, 108, 127.0, "1"), (10, 168, 187.0, "10"), (100, 228, 247.0, "100"),
              (1000, 288, 307.0, "1,000")]
    for cnt, rx, tx, lbl in legend:
        h = math.log(1 + cnt) / den * PLOT_H
        ry = 56 - h
        a(f'<rect x="{rx}" y="{ry:.2f}" width="38" height="{h:.2f}" fill="#5a5a5a" opacity="0.55"/>')
        a(f'<text x="{tx}" y="69" text-anchor="middle" font-family="{FF_MONO}" font-size="10" font-feature-settings="\'tnum\'" fill="#1a1a1a" font-weight="600">{lbl}</text>')
        a(f'<text x="{tx}" y="81" text-anchor="middle" font-family="{FF_SANS}" font-size="8" fill="#8a8a8a">stmts</text>')
    a(f'<text x="408" y="43.0" font-family="{FF_SANS}" font-size="9.5" fill="#8a8a8a" font-style="italic">↑ a bar at this height<tspan x="408" dy="11">means this count</tspan></text>')

    # --- per-source panels ---
    GRID = [1, 10, 100, 1000]   # decade ladder; gmax tick dropped (it collided with 1,000 on the log scale)
    GRID_LBL = {1: "1", 10: "10", 100: "100", 1000: "1,000", gmax: comma(gmax)}
    label_y0 = 108  # bel label baseline in original; pitch 140
    for pi, src in enumerate(order):
        st = stats[src]
        h_hist = hist[src]
        label_y = label_y0 + pi * 140
        rasm_top = label_y + 8           # bg rect top
        our_top = rasm_top + 36
        a(f'<g class="fig4-panel" data-source="{src}">')
        # separator line above panels 2..5 (original draws it at rasm_top - 50? )
        # original: separator y = label_y - 22 region. Observed lines at y=226,366,506,646
        # = (label_y for reach 248)-22 = 226. So sep at label_y-22, for pi>=1.
        if pi >= 1:
            a(f'<line x1="58" y1="{label_y - 22:.1f}" x2="714" y2="{label_y - 22:.1f}" stroke="#d8d6d0" stroke-width="0.5" stroke-dasharray="1 2"/>')
        a(f'<text x="104" y="{label_y}" text-anchor="end" font-family="{FF_SANS}" font-size="14" font-weight="700" fill="#1a1a1a">{src}</text>')
        a(f'<text x="714" y="{label_y}" text-anchor="end" font-family="{FF_MONO}" font-size="10" font-feature-settings="\'tnum\'" fill="#5a5a5a">n = {comma(st["n"])}</text>')

        for which, top, color, mu, lbl_text, lbl_y_off in [
            ("rasm", rasm_top, "#7a8a9a", st["rasm"], "RasMachine", 19.5),
            ("our", our_top, "#a08560", st["our"], "ours", 19.5),
        ]:
            # bg rect
            a(f'<rect x="108" y="{top}" width="502" height="32" fill="#fafaf7" stroke="#d8d6d0" stroke-width="0.5"/>')
            # gridlines on both rows; numeric labels once per panel (RasMachine row) — the log
            # scale is shared, so ambient gray labels recede and the bars stay the figure.
            for cnt in GRID:
                gh = math.log(1 + cnt) / den * PLOT_H
                gy = (top + PLOT_H) - gh
                a(f'<line x1="108" y1="{gy:.2f}" x2="610" y2="{gy:.2f}" stroke="#d8d6d0" stroke-width="0.4" stroke-dasharray="1 2" opacity="0.7"/>')
                if which == "rasm":
                    a(f'<line x1="104" y1="{gy:.2f}" x2="108" y2="{gy:.2f}" stroke="#b8b6b0" stroke-width="0.8"/>')
                    a(f'<text x="102" y="{gy + 3.2:.2f}" text-anchor="end" font-family="{FF_MONO}" font-size="8" font-feature-settings="\'tnum\'" fill="#8a8a8a">{GRID_LBL[cnt]}</text>')
            # bars
            for i in range(NBINS):
                c = h_hist[which][i]
                if not c:
                    continue
                bx = PLOT_LEFT + i * BIN_PX
                by, bh = yh(c, top)
                lo, hi = round(i * 0.05, 2), round((i + 1) * 0.05, 2)
                tag = "RasMachine" if which == "rasm" else "ours"
                a(f'<rect class="fig4-bar" data-source="{src}" data-pred="{which}" data-bin="{i}" x="{bx:.2f}" y="{by:.2f}" width="{BAR_W:.2f}" height="{bh:.2f}" fill="{color}"><title>{tag} · bin [{lo:.2f}, {hi:.2f}] · n = {c}</title></rect>')
            # mean tick
            tx = PLOT_LEFT + mu * PLOT_W
            a(f'<line x1="{tx:.2f}" y1="{top - 2}" x2="{tx:.2f}" y2="{top + 34}" stroke="#1a1a1a" stroke-width="1.3" opacity="0.85"/>')
            a(f'<circle cx="{tx:.2f}" cy="{top + 16.0}" r="2.5" fill="#1a1a1a" opacity="0.9"/>')
            a(f'<text x="76" y="{top + lbl_y_off}" text-anchor="end" font-family="{FF_SANS}" font-size="10.5" fill="{color}" font-weight="600">{lbl_text}</text>')
            a(f'<text x="714" y="{top + lbl_y_off}" text-anchor="end" font-family="{FF_MONO}" font-size="10" font-feature-settings="\'tnum\'" fill="#5a5a5a">μ = {mu:.3f}</text>')

        # x-axis ticks under ours row
        xaxis_y = our_top + 36
        for xv, xpx in [(0, 108.0), (0.25, 233.5), (0.5, 359.0), (0.75, 484.5), (1.0, 610.0)]:
            lbl = "0" if xv == 0 else ("1.0" if xv == 1.0 else str(xv))
            a(f'<line x1="{xpx:.2f}" y1="{xaxis_y}" x2="{xpx:.2f}" y2="{xaxis_y + 3}" stroke="#5a5a5a" stroke-width="0.6"/>')
            a(f'<text x="{xpx:.2f}" y="{xaxis_y + 13}" text-anchor="middle" font-family="{FF_MONO}" font-size="9" font-feature-settings="\'tnum\'" fill="#8a8a8a">{lbl}</text>')
        a('</g>')

    a('</svg>')
    out = "\n".join(S) + "\n"
    with open("/tmp/fig4.svg", "w") as f:
        f.write(out)
    print(f"global_max = {gmax}")
    print(f"ticks: {GRID}")
    print(f"panel order (our-mean desc): {order}")
    print("our-mean sequence: " + " -> ".join(f"{stats[s]['our']:.2f}" for s in order))
    print(f"wrote /tmp/fig4.svg ({len(out)} bytes)")


if __name__ == "__main__":
    main()
