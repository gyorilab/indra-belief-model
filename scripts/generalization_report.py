#!/usr/bin/env python3
"""Standalone Plotly: error-detection F1 ACROSS dataset size (the generalization
dimension), the companion to frontier_report.py's cost/size Pareto plate.

frontier_report reads ONE substrate (cost × F1). This reads the SAME models ACROSS
substrates — each model a line through its (gold_n, error-F1) points, x-ordered by
gold size (log). The story it makes standalone-shareable: a model rated high on a
small, single-curator gold (rasmachine_v1, n=60) settles LOWER on a large, de-biased
one (external-578) — the small-gold mirage — while its 95% bootstrap CI NARROWS as n
grows (precision). Reuses frontier_report's export loader, F1/CI, theme + palette so
the two plates read as one family.

HONEST CAVEAT (rendered in the subtitle): substrates differ in COMPOSITION as well as
size, so a drop is "more AND less curator-captured data", not a within-set learning
curve (which would be flat — a fixed sample can't move its own estimate).

    python scripts/generalization_report.py            # -> reports/generalization_scatter.html
"""
from __future__ import annotations

import glob
import json
import math
import os
from collections import defaultdict

import plotly.colors as pcolors
import plotly.graph_objects as go

from frontier_report import _f1, _f1_ci, _strip_host, ACCENT, INK, INK_MUTED, MONO, PAPER, RULE, SERIF

OUT = os.environ.get("GEN_OUT", "reports/generalization_scatter.html")
PALETTE = pcolors.qualitative.Dark24


def _sub_label(corpus: str) -> str:
    base = os.path.basename(corpus)
    return base.replace("_statements.json", "").replace(".json", "")


def load_series():
    """Cross-substrate: model -> sorted list of points {n, f1, lo, hi, sub}. Folds
    repeat runs of a model on a substrate (mean F1; CI widened to the rep union)."""
    # (model, substrate) -> list of per-run (f1, lo, hi, n). GEN_MAX_N drops golds
    # larger than the cap (e.g. =600 keeps our 60 → 587, drops the separate
    # eval_curation_v1 n=1606 benchmark so the x-axis is only our purpose-built golds).
    cap = float(os.environ.get("GEN_MAX_N", "inf"))
    cells: dict[tuple[str, str], list[tuple[float, float, float, int]]] = defaultdict(list)
    sub_n: dict[str, int] = {}
    for meta_path in sorted(glob.glob("data/exports/*/export_meta.json")):
        d = json.load(open(meta_path))
        model = d.get("model", "")
        if not (model.startswith("bedrock-") or "medpsy" in model or model.startswith("remote-")):
            continue
        corpus = (d.get("generated_from") or {}).get("corpus", "")
        if not corpus:
            continue
        sub = _sub_label(corpus)
        ddir = os.path.dirname(meta_path)
        pe = os.path.join(ddir, "per_evidence.jsonl")
        if not os.path.exists(pe):
            continue
        pairs = []
        for line in open(pe):
            if not line.strip():
                continue
            e = json.loads(line)
            g = e.get("gold")
            if not g:
                continue
            pairs.append((g.get("verdict") == "incorrect", e.get("verdict") == "incorrect"))
        if not pairs:
            continue
        f1, lo, hi, n = _f1_ci(pairs, d["run_id"])
        if n > cap:
            continue
        cells[(model, sub)].append((f1, lo, hi, n))
        sub_n[sub] = max(sub_n.get(sub, 0), n)

    series: dict[str, list[dict]] = defaultdict(list)
    for (model, sub), reps in cells.items():
        f1 = sum(r[0] for r in reps) / len(reps)
        series[model].append({
            "sub": sub, "n": max(r[3] for r in reps),
            "f1": f1, "lo": min(r[1] for r in reps), "hi": max(r[2] for r in reps),
            "reps": len(reps),
        })
    out = {}
    for model, pts in series.items():
        if len(pts) < 2:
            continue
        pts.sort(key=lambda p: p["n"])
        out[model] = pts
    # steepest droppers first (most negative delta)
    return dict(sorted(out.items(), key=lambda kv: kv[1][-1]["f1"] - kv[1][0]["f1"])), sub_n


def main():
    series, sub_n = load_series()
    if not series:
        raise SystemExit("no cross-substrate models found in data/exports/")
    color_of = {m: PALETTE[i % len(PALETTE)] for i, m in enumerate(series)}
    # Disambiguate cross-host duplicates (bedrock- vs remote- gemma-4-26b both strip
    # to "gemma-4-26b"): keep the full model string on BOTH so no two legend rows /
    # lines read identically (mirrors the viewer's FrontierStrip rule).
    bare_counts: dict[str, int] = defaultdict(int)
    for m in series:
        bare_counts[_strip_host(m)] += 1
    disp = {m: (m if bare_counts[_strip_host(m)] > 1 else _strip_host(m)) for m in series}
    all_n = sorted({p["n"] for pts in series.values() for p in pts})
    all_f1 = [p["f1"] for pts in series.values() for p in pts]
    all_lo = [p["lo"] for pts in series.values() for p in pts]
    all_hi = [p["hi"] for pts in series.values() for p in pts]
    ylo, yhi = max(0, min(all_lo) - 0.02), min(1, max(all_hi) + 0.02)

    fig = go.Figure()
    for model, pts in series.items():
        xs = [p["n"] for p in pts]
        ys = [p["f1"] for p in pts]
        name = disp[model]
        delta = pts[-1]["f1"] - pts[0]["f1"]
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines+markers",
            line=dict(color=color_of[model], width=1.7),
            marker=dict(color=color_of[model], size=7, line=dict(color="#fff", width=0.8)),
            error_y=dict(type="data", symmetric=False,
                         array=[p["hi"] - p["f1"] for p in pts],
                         arrayminus=[p["f1"] - p["lo"] for p in pts],
                         color="rgba(0,0,0,0.16)", thickness=1, width=3),
            name=name,
            legendgroup=name,
            customdata=[[name, p["sub"], p["n"], f"[{p['lo']:.2f}–{p['hi']:.2f}]", f"{delta:+.3f}"] for p in pts],
            hovertemplate="<b>%{customdata[0]}</b> · %{customdata[1]}"
                          "<br>error-F1 %{y:.3f} %{customdata[3]} · gold n=%{customdata[2]}"
                          "<br>Δ over range %{customdata[4]}<extra></extra>",
        ))

    # substrate guide columns + n labels at the foot
    tick_text = []
    for n in all_n:
        sub = next((s for s, v in sub_n.items() if v == n), "")
        tick_text.append(f"n={n}<br>{sub}")
        fig.add_vline(x=n, line=dict(color="#bbb", width=0.6, dash="dot"))

    n_models = len(series)
    n_drop = sum(1 for pts in series.values() if pts[-1]["f1"] - pts[0]["f1"] < 0)
    fig.update_xaxes(
        type="log", tickvals=all_n, ticktext=tick_text,
        title_text="gold benchmark · size n (log) — larger & less curator-captured →",
        range=[math.log10(min(all_n)) - 0.08, math.log10(max(all_n)) + 0.08],
        gridcolor=RULE, zeroline=False,
    )
    fig.update_yaxes(range=[ylo, yhi], title_text="error-detection F1", gridcolor=RULE)
    fig.update_layout(
        template="plotly_white", width=1040, height=620, paper_bgcolor=PAPER, plot_bgcolor=PAPER,
        font=dict(family=MONO, size=12, color=INK),
        title=dict(
            text=(f"Error-detection F1 across dataset size — {n_models} models, "
                  f"{n_drop} fall as the gold grows<br>"
                  f"<span style='font-size:12px;color:{INK_MUTED}'>"
                  "the small-gold mirage: a model rated high on a small, single-curator gold settles "
                  "lower on a large, de-biased one. CI narrows with n (precision). "
                  "x mixes size with composition — not a within-set learning curve.</span>"),
            x=0.5, xanchor="center", font=dict(family=SERIF, size=18),
        ),
        margin=dict(l=70, r=180, t=96, b=78),
        legend=dict(font=dict(size=10), x=1.015, xanchor="left", y=1, yanchor="top",
                    itemsizing="constant", tracegroupgap=1,
                    title=dict(text="model (click to isolate)", font=dict(size=11))),
        hoverlabel=dict(font_family=MONO),
    )
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.write_html(OUT, include_plotlyjs="inline", full_html=True,
                   config=dict(displaylogo=False, responsive=True))
    print(f"wrote {OUT}  ·  {n_models} models across {len(all_n)} gold sizes (n={min(all_n)}–{max(all_n)})  ·  {n_drop} fall")


if __name__ == "__main__":
    main()
