#!/usr/bin/env python3
"""Shareable, interactive scatterplots: error-detection F1 over cost, and over
model size — rasmachine_v1 substrate, BEDROCK models + medpsy only.

Built with Plotly into ONE self-contained HTML (plotly.js inlined → opens in any
browser offline, no deps). One title, two panels, Plotly's native look:
  - decade log x-axis, padded so edge dots aren't clipped; Pareto staircase
  - 95% bootstrap CI as error bars; hover gives exact F1 [CI] + cost/size
  - one distinct colour per model + a shared legend, so every dot (frontier-
    labelled or not) is identifiable; labels are the MODEL NAME only
  - frontier points are named in place (anchored up-left off the rising trace)

Run:  python scripts/frontier_report.py   ->  reports/frontier_scatter.html
"""
from __future__ import annotations

import glob
import json
import math
import os
from html import escape

import plotly.colors as pcolors
import plotly.graph_objects as go

SUBSTRATE = os.environ.get("FRONTIER_SUBSTRATE", "rasmachine_v1_statements.json")
OUT = os.environ.get("FRONTIER_OUT", "reports/frontier_scatter.html")
SUBSTRATE_LABEL = os.environ.get("FRONTIER_LABEL", "rasmachine_v1")

INK, INK_MUTED, INK_FAINT = "#1a1a1a", "#6a6a6a", "#727272"
PAPER, RULE, ACCENT = "#fdfcf8", "#e6e2d6", "#7d2a1a"
MONO = "ui-monospace, 'SF Mono', Menlo, monospace"
SERIF = "'Iowan Old Style', Georgia, serif"


# ── data + metrics ──────────────────────────────────────────────────────────
def _strip_host(m: str) -> str:
    for p in ("bedrock-", "remote-", "local-", "google-"):
        if m.startswith(p):
            return m[len(p):]
    return m


def _f1(pairs) -> float:
    tp = fp = fn = 0
    for g, p in pairs:
        if p and g:
            tp += 1
        elif p:
            fp += 1
        elif g:
            fn += 1
    pr = tp / (tp + fp) if tp + fp else 0.0
    rc = tp / (tp + fn) if tp + fn else 0.0
    return 2 * pr * rc / (pr + rc) if pr + rc else 0.0


def _seed(run_id: str) -> int:
    h = 0
    for ch in run_id:
        h = (h * 31 + ord(ch)) & 0xFFFFFFFF
    return h


def _f1_ci(pairs, run_id: str, B: int = 1000):
    """error-detection F1 + deterministic bootstrap 95% CI (seeded by run)."""
    n = len(pairs)
    if n == 0:
        return 0.0, 0.0, 0.0, 0
    s = (_seed(run_id) ^ 0x9E3779B9) & 0xFFFFFFFF
    def rnd():
        nonlocal s
        s = (s * 1664525 + 1013904223) & 0xFFFFFFFF
        return s / 2**32
    f1s = []
    for _ in range(B):
        samp = [pairs[int(rnd() * n)] for _ in range(n)]
        f1s.append(_f1(samp))
    f1s.sort()
    return _f1(pairs), f1s[int(0.025 * B)], f1s[min(B - 1, math.ceil(0.975 * B) - 1)], n


def load_runs():
    runs = []
    for meta_path in sorted(glob.glob("data/exports/*/export_meta.json")):
        d = json.load(open(meta_path))
        if not (d.get("generated_from") or {}).get("corpus", "").endswith(SUBSTRATE):
            continue
        model = d.get("model", "")
        if not (model.startswith("bedrock-") or "medpsy" in model):  # bedrock + medpsy ONLY
            continue
        ddir = os.path.dirname(meta_path)
        pairs = []
        for line in open(os.path.join(ddir, "per_evidence.jsonl")):
            if not line.strip():
                continue
            e = json.loads(line)
            g = e.get("gold")
            if not g:
                continue
            pairs.append((g.get("verdict") == "incorrect", e.get("verdict") == "incorrect"))
        f1, lo, hi, n = _f1_ci(pairs, d["run_id"])
        c = d.get("cost") or {}
        mm = d.get("model_meta") or {}
        size_known = mm.get("status") == "known" and mm.get("total_b") is not None
        runs.append({
            "name": _strip_host(model),
            "f1": f1, "f1_lo": lo, "f1_hi": hi, "n_gold": n,
            "usd_per_1k": c.get("usd_per_1k_evidence") if c.get("status") not in (None, "unavailable") else None,
            "cost_estimated": c.get("status") == "estimated",
            "total_b": mm.get("total_b") if size_known else None,
            "size_estimated": bool(mm.get("estimated")) if size_known else False,
        })
    return sorted(runs, key=lambda r: -r["f1"])


# ── figure ──────────────────────────────────────────────────────────────────
from plotly.subplots import make_subplots

AXES = {
    "cost": dict(
        xval=lambda r: r["usd_per_1k"],
        xtitle="cost — USD per 1k evidence (log)",
        tickfmt=lambda v: (f"${v:g}" if v >= 1 else f"${v:.2f}".rstrip("0").rstrip(".")),
        disp=lambda r: f"${r['usd_per_1k']:.2f}/1k",
    ),
    "size": dict(
        xval=lambda r: r["total_b"],
        xtitle="model size — total parameters (log)",
        tickfmt=lambda v: (f"{v/1000:g}T" if v >= 1000 else f"{v:g}B"),
        disp=lambda r: (f"{r['total_b']/1000:g}T" if r["total_b"] >= 1000 else f"{r['total_b']:g}B"),
    ),
}
FRONTIER = "firebrick"
LABEL = "#222"
PALETTE = pcolors.qualitative.Dark24  # 24 distinct hues; 17 models fit


def add_panel(fig, runs, axis, row, color_of, yrange, show_legend):
    spec = AXES[axis]
    plot = [r for r in runs if spec["xval"](r) and spec["xval"](r) > 0]
    xs = [spec["xval"](r) for r in plot]
    lmin, lmax = math.floor(math.log10(min(xs))), math.ceil(math.log10(max(xs)))
    if lmax == lmin:
        lmax += 1
    for r in plot:
        rx_, rf = spec["xval"](r), r["f1"]
        r["_front"] = not any(
            s is not r and spec["xval"](s) <= rx_ and s["f1"] >= rf
            and (spec["xval"](s) < rx_ or s["f1"] > rf)
            for s in plot
        )
    front = sorted([r for r in plot if r["_front"]], key=lambda r: spec["xval"](r))

    # Pareto staircase (step line, flat ends), as a legend item.
    if front:
        fx = [10**lmin] + [spec["xval"](r) for r in front] + [10**lmax]
        fy = [front[0]["f1"]] + [r["f1"] for r in front] + [front[-1]["f1"]]
        fig.add_trace(go.Scatter(x=fx, y=fy, mode="lines",
                                 line=dict(color=FRONTIER, width=1.6, shape="hv"),
                                 name="Pareto frontier", legendgroup="_pareto",
                                 showlegend=show_legend, hoverinfo="skip"), row=row, col=1)

    # Frontier labels: anchor UP-LEFT (off the right-rising staircase, so the trace
    # never crosses the text and edge points don't clip); near the left edge flip to
    # the right; flip a near-tie's label below so adjacent same-F1 labels don't
    # collide. (Plotly has no label de-collision; everything else is in the legend.)
    pos_of, placed = {}, []
    for r in front:
        lx, y = math.log10(spec["xval"](r)), r["f1"]
        horiz = "right" if (lx - lmin) / (lmax - lmin) < 0.12 else "left"
        pos = f"top {horiz}"
        if any(abs(lx - plx) < 0.2 and abs(y - py) < 0.025 and pos == ppos for plx, py, ppos in placed):
            pos = f"bottom {horiz}"
        placed.append((lx, y, pos))
        pos_of[id(r)] = pos

    # one trace per model → a per-model colour legend that identifies every dot,
    # labelled or not. Frontier dots are larger + named in place.
    for r in plot:
        fr = r["_front"]
        fig.add_trace(go.Scatter(
            x=[spec["xval"](r)], y=[r["f1"]],
            mode="markers+text" if fr else "markers",
            text=[r["name"]] if fr else None,
            textposition=pos_of.get(id(r), "top left") if fr else None,
            textfont=dict(size=10, color=LABEL),
            marker=dict(color=color_of[r["name"]], size=12 if fr else 8,
                        line=dict(color="#fff", width=1 if fr else 0.5)),
            error_y=dict(type="data", symmetric=False,
                         array=[r["f1_hi"] - r["f1"]], arrayminus=[r["f1"] - r["f1_lo"]],
                         color="rgba(0,0,0,0.18)", thickness=1, width=0),
            name=r["name"], legendgroup=r["name"], showlegend=show_legend,
            customdata=[[r["name"], f"[{r['f1_lo']:.2f}–{r['f1_hi']:.2f}]", spec["disp"](r)]],
            hovertemplate="<b>%{customdata[0]}</b><br>error-F1 %{y:.2f} %{customdata[1]}"
                          "<br>%{customdata[2]}<extra></extra>",
        ), row=row, col=1)

    tickvals = [10**p for p in range(lmin, lmax + 1)]
    fig.update_xaxes(type="log", range=[lmin - 0.1, lmax + 0.1], tickvals=tickvals,
                     ticktext=[spec["tickfmt"](v) for v in tickvals],
                     title_text=spec["xtitle"], row=row, col=1)
    fig.update_yaxes(range=list(yrange), title_text="error-detection F1", row=row, col=1)


def main():
    runs = load_runs()
    n = max((r["n_gold"] for r in runs), default=0)
    color_of = {r["name"]: PALETTE[i % len(PALETTE)] for i, r in enumerate(runs)}
    ylo = max(0.0, min(r["f1_lo"] for r in runs) - 0.02)
    yhi = min(1.0, max(r["f1_hi"] for r in runs) + 0.02)
    fig = make_subplots(rows=2, cols=1, vertical_spacing=0.17)
    add_panel(fig, runs, "cost", 1, color_of, (ylo, yhi), show_legend=True)
    add_panel(fig, runs, "size", 2, color_of, (ylo, yhi), show_legend=False)
    fig.update_layout(
        template="plotly_white", width=1080, height=900,
        title_text=f"Error-detection F1 over cost and model size — {SUBSTRATE_LABEL} (gold n={n})",
        title_x=0.5, margin=dict(l=70, r=240, t=70, b=60),
        legend=dict(font=dict(size=10), x=1.015, xanchor="left", y=1, yanchor="top",
                    itemsizing="constant", tracegroupgap=2,
                    title=dict(text="model (hover for detail)", font=dict(size=11))),
    )
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.write_html(OUT, include_plotlyjs="inline", full_html=True,
                   config=dict(displaylogo=False, responsive=True))
    print(f"wrote {OUT}  ·  {len(runs)} models  ·  gold n={n}")


if __name__ == "__main__":
    main()
