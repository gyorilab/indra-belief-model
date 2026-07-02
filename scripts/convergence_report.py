#!/usr/bin/env python3
"""Standalone Plotly: how reliable is an error-F1 estimate at each eval size?

At each subsample size n, draw MANY real estimates (score a random size-n subsample of
the 587 gold, no replacement, repeat). Each dot is a value you genuinely might have
reported from an eval that size. The cloud is wide and unreliable when the eval is
small and collapses onto the true score as it grows — that collapse is the convergence.
There is NO trend line: the model is fixed, so a "centre" would be a flat, meaningless
constant. The only honest line is the dashed full-gold score the cloud converges TO.

Two views (CONV_MODE):
  overlay  a few chosen models on one axis — shows WHEN models become distinguishable
           (clouds overlap at small n, pull apart as n grows). Default models include
           gemma-4-26b + gemma-4-31b (which overlap → tied) and gemma-4-e2b (the floor).
  grid     small multiples — one panel per model we tested, ordered best→worst. Scales
           to all 13 without the haze of overlaid clouds; compare terminal height
           (score/rank) and cloud width (reliability) across the grid.

    python scripts/convergence_report.py                 # overlay -> reports/convergence_scatter.html
    CONV_MODE=grid python scripts/convergence_report.py  # grid    -> reports/convergence_grid.html
"""
from __future__ import annotations

import glob
import json
import math
import os
import sys

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from eval_curation_compare import build_gold_index, join_model  # noqa: E402
from frontier_report import _f1, _strip_host, ACCENT, INK, INK_MUTED, MONO, PAPER, RULE, SERIF  # noqa: E402
from indra_belief.curation import is_gold_correct  # noqa: E402
from indra_belief.model_meta import model_size  # noqa: E402

MODE = os.environ.get("CONV_MODE", "overlay")
_DEFOUT = {"grid": "convergence_grid", "summary": "convergence_summary"}.get(MODE, "convergence_scatter")
OUT = os.environ.get("CONV_OUT", f"reports/{_DEFOUT}.html")

# Run stems are host-stripped aliases; bridge the four whose size-registry key differs.
SIZE_ALIAS = {
    "nemotron-nano-30b": "nemotron-nano-3-30b",
    "nemotron-super-120b": "nemotron-super-3-120b",
    "qwen3-235b": "qwen3-235b-a22b",
    "qwen3-coder-480b": "qwen3-coder-480b-a35b",
}


def get_size(model: str):
    return model_size(SIZE_ALIAS.get(model, model)).get("total_b")


def fmt_b(b):
    if b is None:
        return "?"
    return f"{b / 1000:g}T" if b >= 1000 else f"{b:g}B"
GOLD = os.path.join(ROOT, "data", "benchmark", "external_curator_gold_v1.jsonl")
RNG = np.random.default_rng(20260630)
# the focused overlay set (well-spread + the tied top pair)
OVERLAY = [("gemma-4-31b", "#1f77b4", "31B — value pick"),
           ("gemma-4-26b", "#2a9d3a", "26B — production"),
           ("gemma-4-e2b", "#d6336c", "e2b — cheap floor")]


def all_pairs():
    gold = [json.loads(l) for l in open(GOLD) if l.strip()]
    by_pair, by_sh = build_gold_index(gold)
    out = {}
    for run in sorted(glob.glob(os.path.join(ROOT, "data/results/external_curator_v1_*.jsonl"))):
        if "progress" in run:
            continue
        scored = [json.loads(l) for l in open(run) if l.strip()]
        joined, _, _ = join_model(scored, by_pair, by_sh)
        pairs = np.array([(not is_gold_correct(g["tag"]), s["verdict"] == "incorrect") for g, s in joined], bool)
        if len(pairs) < 200:
            continue
        name = os.path.basename(run).replace("external_curator_v1_bedrock-", "").replace(
            "external_curator_v1_", "").replace(".jsonl", "")
        out[name] = pairs
    return out


def cloud(pairs: np.ndarray, n: int, k: int):
    m = len(pairs)
    n = min(n, m)
    return [_f1([(bool(g), bool(p)) for g, p in pairs[RNG.choice(m, n, replace=False)]]) for _ in range(k)]


def true_f1(pairs):
    return _f1([(bool(g), bool(p)) for g, p in pairs])


def ci(pairs, reps=2000):
    """Bootstrap 95% CI of error-F1 over the full pairs — the residual uncertainty
    at full n that the convergence collapses onto."""
    m = len(pairs)
    vals = np.array([_f1([(bool(g), bool(p)) for g, p in pairs[RNG.integers(0, m, m)]]) for _ in range(reps)])
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


# ── overlay: a few models, one axis ─────────────────────────────────────────
def render_overlay(data):
    pulled = [(n, c, r, data[n]) for n, c, r in OVERLAY if n in data]
    full = min(len(p) for *_, p in pulled)
    grid = [20, 40, 70, 120, 200, 300, 420, full]
    fig = go.Figure()
    for i, (name, color, role, pairs) in enumerate(pulled):
        t = true_f1(pairs)
        fig.add_hline(y=t, line=dict(color=color, width=1, dash="dash"), opacity=0.5,
                      annotation_text=f"  {name} full = {t:.3f}", annotation_position="right",
                      annotation_font=dict(color=color, size=10, family=MONO))
        off = [0.95, 1.0, 1.05][i]
        xs, ys = [], []
        for n in grid:
            for v in cloud(pairs, n, 80):
                ys.append(v); xs.append(n * off * (1 + RNG.uniform(-0.018, 0.018)))
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="markers",
                                 marker=dict(color=color, size=4, opacity=0.22),
                                 name=f"{name} ({role})", legendgroup=name,
                                 hovertemplate=f"<b>{name}</b><br>eval n≈%{{x:.0f}}: F1 %{{y:.3f}}<extra></extra>"))
    fig.update_xaxes(type="log", tickvals=grid, ticktext=[str(n) for n in grid],
                     title_text="eval size n — random subsample of the 587 gold →", gridcolor=RULE, zeroline=False)
    fig.update_yaxes(title_text="error-detection F1 you might measure", gridcolor=RULE, range=[0.4, 1.0])
    fig.update_layout(
        template="plotly_white", width=1040, height=620, paper_bgcolor=PAPER, plot_bgcolor=PAPER,
        font=dict(family=MONO, size=12, color=INK),
        title=dict(text=("How reliable is an error-F1 estimate at each eval size?<br>"
                         f"<span style='font-size:12px;color:{INK_MUTED}'>each dot = one score from a random "
                         "subsample that size. Small eval → huge spread, the gemmas overlap (tied / "
                         "indistinguishable). Big eval → clouds collapse onto the true score (dashed).</span>"),
                   x=0.5, xanchor="center", font=dict(family=SERIF, size=18)),
        margin=dict(l=70, r=190, t=96, b=70),
        legend=dict(font=dict(size=11), x=1.015, xanchor="left", y=0.5, yanchor="middle", itemsizing="constant"),
        hoverlabel=dict(font_family=MONO))
    return fig


# ── grid: one small-multiple per model ──────────────────────────────────────
def render_grid(data):
    order = sorted(data, key=lambda k: (get_size(k) or 1e9, -true_f1(data[k])))  # by parameter size
    full = min(len(p) for p in data.values())
    grid = [20, 50, 120, 300, full]
    cols = 4
    rows = math.ceil(len(order) / cols)
    titles = [f"{_strip_host(m)} {fmt_b(get_size(m))} · F1 {true_f1(data[m]):.3f}" for m in order]
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=titles,
                        shared_yaxes=True, vertical_spacing=0.09, horizontal_spacing=0.03)
    for i, model in enumerate(order):
        r, c = i // cols + 1, i % cols + 1
        pairs = data[model]
        t = true_f1(pairs)
        xs, ys = [], []
        for n in grid:
            for v in cloud(pairs, n, 35):
                ys.append(v); xs.append(n * (1 + RNG.uniform(-0.05, 0.05)))
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="markers",
                                 marker=dict(color=INK, size=2.4, opacity=0.16), showlegend=False,
                                 hovertemplate=f"<b>{model}</b><br>n≈%{{x:.0f}}: F1 %{{y:.3f}}<extra></extra>"),
                      row=r, col=c)
        fig.add_hline(y=t, line=dict(color=ACCENT, width=1, dash="dash"), opacity=0.7, row=r, col=c)
    fig.update_xaxes(type="log", tickvals=[20, 120, full], ticktext=["20", "120", str(full)],
                     gridcolor=RULE, zeroline=False, tickfont=dict(size=8))
    fig.update_yaxes(range=[0.4, 1.0], gridcolor=RULE, tickfont=dict(size=8))
    for ann in fig.layout.annotations:  # subplot titles → smaller, mono
        ann.font = dict(size=10, family=MONO, color=INK)
    fig.update_layout(
        template="plotly_white", width=1180, height=190 * rows + 110, paper_bgcolor=PAPER, plot_bgcolor=PAPER,
        font=dict(family=MONO, size=11, color=INK),
        title=dict(text=("Eval convergence, one panel per model — dot = a score from a random subsample, "
                         "dashed = full-gold score<br>"
                         f"<span style='font-size:12px;color:{INK_MUTED}'>every cloud collapses onto its "
                         "dashed line as the eval grows (convergence); the dashed height is the model's score, "
                         "the cloud width its reliability. Ordered by parameter size (small→large) — note score "
                         "does NOT track size.</span>"),
                   x=0.5, xanchor="center", font=dict(family=SERIF, size=17)),
        margin=dict(l=44, r=20, t=104, b=40), hoverlabel=dict(font_family=MONO))
    return fig


# ── summary: every model on ONE plane — score + CI, size-ordered ────────────
def render_summary(data):
    order = sorted(data, key=lambda k: (get_size(k) or 0))  # ascending size → big on top
    rows = [(m, true_f1(data[m]), *ci(data[m]), get_size(m)) for m in order]
    best = max(r[1] for r in rows)
    best_lo, best_hi = next((lo, hi) for (_, f, lo, hi, _s) in rows if f == best)
    ys = list(range(len(rows)))
    tied = [r[3] >= best_lo for r in rows]  # CI reaches into the leader's interval

    fig = go.Figure()
    # leader's 95% CI as a shaded vertical band: any dot/whisker overlapping it is
    # statistically tied with the best model — the whole "top cluster" point, at a glance.
    fig.add_vrect(x0=best_lo, x1=best_hi, fillcolor=ACCENT, opacity=0.08, line_width=0,
                  annotation_text="leader's 95% CI → overlap = tied", annotation_position="top",
                  annotation_font=dict(size=10, color=ACCENT, family=MONO))
    fig.add_trace(go.Scatter(
        x=[r[1] for r in rows], y=ys, mode="markers",
        marker=dict(color=[ACCENT if t else INK for t in tied], size=9, line=dict(color="#fff", width=1)),
        error_x=dict(type="data", symmetric=False, array=[r[3] - r[1] for r in rows],
                     arrayminus=[r[1] - r[2] for r in rows], color="rgba(0,0,0,0.32)", thickness=1.3, width=5),
        customdata=[[_strip_host(r[0]), fmt_b(r[4]), f"[{r[2]:.3f}–{r[3]:.3f}]"] for r in rows],
        hovertemplate="<b>%{customdata[0]}</b> · %{customdata[1]}<br>error-F1 %{x:.3f} %{customdata[2]}<extra></extra>",
        showlegend=False))
    fig.update_yaxes(tickvals=ys, ticktext=[f"{_strip_host(r[0])}  {fmt_b(r[4])}" for r in rows],
                     tickfont=dict(size=10), gridcolor=RULE, zeroline=False)
    fig.update_xaxes(title_text="error-detection F1 on the full 587 gold (dot) · 95% CI (whisker)",
                     gridcolor=RULE, zeroline=False,
                     range=[min(r[2] for r in rows) - 0.02, max(r[3] for r in rows) + 0.02])
    fig.update_layout(
        template="plotly_white", width=940, height=560, paper_bgcolor=PAPER, plot_bgcolor=PAPER,
        font=dict(family=MONO, size=12, color=INK),
        title=dict(text=("All models on one plane — converged score & uncertainty, smallest→largest<br>"
                         f"<span style='font-size:12px;color:{INK_MUTED}'>dot = full-gold error-F1, whisker = "
                         "95% CI. Models whose CI overlaps the leader's band (shaded) are statistically TIED — "
                         "the top cluster. Reading up the size axis, score does not climb: bigger ≠ better.</span>"),
                   x=0.5, xanchor="center", font=dict(family=SERIF, size=18)),
        margin=dict(l=170, r=40, t=92, b=56), hoverlabel=dict(font_family=MONO))
    return fig


def main():
    data = all_pairs()
    if not data:
        raise SystemExit("no external_curator_v1 runs found")
    fig = {"grid": render_grid, "summary": render_summary}.get(MODE, render_overlay)(data)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.write_html(OUT, include_plotlyjs="inline", full_html=True,
                   config=dict(displaylogo=False, responsive=True))
    print(f"wrote {OUT}  ·  mode={MODE} · {len(data)} models available")


if __name__ == "__main__":
    main()
