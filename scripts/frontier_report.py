#!/usr/bin/env python3
"""Shareable pair frontier plus a separate published-paper method landscape.

Built with Plotly into ONE self-contained HTML (plotly.js inlined → opens in any
browser offline, no deps). The first two panels retain the legacy evidence-pair
error-F1 frontier.  When the pinned paper-method artifact is present, a third
panel shows the paper's statement-level fold-mean trapezoidal PR areas.

The third panel is deliberately not overlaid on the pair frontier: its unit,
positive class, metric and uncertainty are different.  It is a published
reference landscape until all systems have shared-statement predictions.

Plot grammar:
  - decade log x-axis, padded so edge dots aren't clipped; Pareto staircase
  - 95% bootstrap CI as error bars; hover gives exact F1 [CI] + cost/size
  - one distinct colour per model + a shared legend, so every dot (frontier-
    labelled or not) is identifiable; labels are the MODEL NAME only
  - frontier points are named in place (anchored up-left off the rising trace)
  - paper points use horizontal fold-SD bars (explicitly not confidence intervals)

Run:  python scripts/frontier_report.py   ->  reports/frontier_scatter.html
"""
from __future__ import annotations

import glob
import json
import math
import os
from html import escape
from pathlib import Path

import plotly.colors as pcolors
import plotly.graph_objects as go

SUBSTRATE = os.environ.get("FRONTIER_SUBSTRATE", "rasmachine_v1_statements.json")
OUT = os.environ.get("FRONTIER_OUT", "reports/frontier_scatter.html")
SUBSTRATE_LABEL = os.environ.get("FRONTIER_LABEL", "rasmachine_v1")
PAPER_METHODS_PATH = Path(
    os.environ.get(
        "FRONTIER_PAPER_METHODS",
        "data/benchmark/indra_paper_2023_published_method_metrics.json",
    )
)

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


def load_paper_methods(path: Path = PAPER_METHODS_PATH):
    """Load the checksum-pinned published reference points, fail closed."""
    if not path.is_file():
        return None
    artifact = json.loads(path.read_text(encoding="utf-8"))
    if artifact.get("schema_version") != 1 or artifact.get("artifact_kind") != (
        "indra_assembly_paper_published_method_metrics"
    ):
        raise ValueError(f"unsupported paper method artifact: {path}")
    contract = artifact.get("metric_contract") or {}
    required_contract = {
        "unit": "assembled_statement",
        "uncertainty_is_confidence_interval": False,
        "metric_is_pooled_average_precision": False,
        "directly_comparable_to_pair_error_f1": False,
    }
    for key, expected in required_contract.items():
        if contract.get(key) != expected:
            raise ValueError(f"paper method contract mismatch for {key}: {path}")
    methods = artifact.get("methods")
    if not isinstance(methods, list) or len(methods) != artifact.get("method_count"):
        raise ValueError(f"paper method count mismatch: {path}")
    if len(methods) != 59 or len({row.get("method_id") for row in methods}) != 59:
        raise ValueError(f"expected 59 uniquely identified paper methods: {path}")
    for row in methods:
        mean = row.get("fold_mean_trapezoidal_pr_auc")
        sd = row.get("fold_population_sd")
        if (
            not isinstance(mean, (int, float))
            or not 0 <= mean <= 1
            or not isinstance(sd, (int, float))
            or sd < 0
            or row.get("fold_count") != 10
        ):
            raise ValueError(f"invalid paper method estimate: {row.get('method_id')}")
    return artifact


def _paper_family(method: str) -> str:
    if method.startswith("Belief Orig"):
        return "original belief"
    if method.startswith("RF "):
        return "random forest"
    if method.startswith("Log LR"):
        return "logistic regression"
    if method.startswith("SVC"):
        return "SVC"
    if method.startswith("KNN"):
        return "KNN"
    raise ValueError(f"unknown paper method family: {method}")


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
WIDTH, HEIGHT, MARGIN_L, MARGIN_R = 1180, 900, 70, 265
PAPER_FAMILY_ORDER = [
    "KNN",
    "SVC",
    "original belief",
    "logistic regression",
    "random forest",
]
PAPER_FAMILY_COLORS = {
    "KNN": "#8c8c8c",
    "SVC": "#6b6ecf",
    "original belief": "#7d2a1a",
    "logistic regression": "#2a6f97",
    "random forest": "#2f7d4a",
}
PAPER_SUBPLOT_TITLE = (
    "Published methods only · statement PR area · mixed eligible sets"
    "<br><sup>Not comparable with pair F1 above; 0.917 and 0.942 are not a paired delta</sup>"
)


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

    # one trace per model → a per-model colour legend that identifies every dot,
    # labelled or not. Frontier dots are larger; slight opacity + a white halo keep
    # coincident dots legible instead of one hiding the other.
    for r in plot:
        fr = r["_front"]
        fig.add_trace(go.Scatter(
            x=[spec["xval"](r)], y=[r["f1"]],
            mode="markers",
            marker=dict(color=color_of[r["name"]], size=11 if fr else 7,
                        opacity=0.9, line=dict(color="#fff", width=1.3 if fr else 0.6)),
            error_y=dict(type="data", symmetric=False,
                         array=[r["f1_hi"] - r["f1"]], arrayminus=[r["f1"] - r["f1_lo"]],
                         color="rgba(0,0,0,0.16)", thickness=1, width=0),
            name=r["name"], legendgroup=r["name"], showlegend=show_legend,
            customdata=[[r["name"], f"[{r['f1_lo']:.2f}–{r['f1_hi']:.2f}]", spec["disp"](r)]],
            hovertemplate="<b>%{customdata[0]}</b><br>error-F1 %{y:.2f} %{customdata[1]}"
                          "<br>%{customdata[2]}<extra></extra>",
        ), row=row, col=1)

    # Frontier labels as leader-line callouts. Plotly's textposition has no collision
    # handling, and the compressed 403 frontier packs several points into a narrow
    # band. An isolated point is labelled in place just above its dot; a run of points
    # too close to label side-by-side is FANNED along one row above, packed left→right
    # in x-order (so leaders never cross) and offset onto its own leader.
    span = (lmax - lmin) + 0.2
    px_per_log = max(1.0, (WIDTH - MARGIN_L - MARGIN_R) / span)

    def _wlog(name):  # approx rendered label width, in log-x units
        return (len(name) * 6.4) / px_per_log

    def _label(r, ax_px, ay_px, anchor):
        fig.add_annotation(
            x=math.log10(spec["xval"](r)), y=r["f1"], text=r["name"], row=row, col=1,
            showarrow=True, arrowhead=0, arrowwidth=0.7, arrowcolor="rgba(0,0,0,0.3)",
            ax=ax_px, ay=ay_px, xanchor=anchor, yanchor="bottom",
            font=dict(size=10, color=LABEL), bgcolor="rgba(253,252,248,0.82)",
        )

    pts = sorted(front, key=lambda s: spec["xval"](s))
    clusters, cur = [], (pts[:1])
    for r in pts[1:]:
        gap = math.log10(spec["xval"](r)) - math.log10(spec["xval"](cur[-1]))
        if gap < 0.5 * (_wlog(cur[-1]["name"]) + _wlog(r["name"])):
            cur.append(r)
        else:
            clusters.append(cur)
            cur = [r]
    if cur:
        clusters.append(cur)

    for cl in clusters:
        if len(cl) == 1:
            r = cl[0]
            xlog = math.log10(spec["xval"](r))
            at_left = (xlog - lmin) / (lmax - lmin) < 0.12
            _label(r, 16 if at_left else 0, -22, "left" if at_left else "center")
        else:  # fan: pack labels rightward from the leftmost dot, one row above
            acc = min(math.log10(spec["xval"](r)) for r in cl)
            for r in cl:
                w = _wlog(r["name"])
                target = acc + w / 2
                acc += w * 1.05
                _label(r, (target - math.log10(spec["xval"](r))) * px_per_log, -30, "center")

    tickvals = [10**p for p in range(lmin, lmax + 1)]
    fig.update_xaxes(type="log", range=[lmin - 0.1, lmax + 0.1], tickvals=tickvals,
                     ticktext=[spec["tickfmt"](v) for v in tickvals],
                     title_text=spec["xtitle"], row=row, col=1)
    fig.update_yaxes(range=list(yrange), title_text="error-detection F1", row=row, col=1)


def add_paper_panel(fig, artifact, row: int):
    """Draw all published paper configurations without claiming direct parity."""
    methods = artifact["methods"]
    plotted: dict[str, tuple[list[dict], list[float]]] = {}
    for family in PAPER_FAMILY_ORDER:
        family_rows = sorted(
            [item for item in methods if _paper_family(item["method"]) == family],
            key=lambda item: (
                item["fold_mean_trapezoidal_pr_auc"],
                item["method_id"],
            ),
        )
        count = len(family_rows)
        offsets = [0.0] if count == 1 else [
            -0.24 + (0.48 * index / (count - 1)) for index in range(count)
        ]
        plotted[family] = (family_rows, offsets)

    for family_index, family in enumerate(PAPER_FAMILY_ORDER):
        family_rows, offsets = plotted[family]
        x = [item["fold_mean_trapezoidal_pr_auc"] for item in family_rows]
        y = [family_index + offset for offset in offsets]
        sd = [item["fold_population_sd"] for item in family_rows]
        symbols = [
            "circle" if item["table_id"] == "paper_table_6" else "diamond-open"
            for item in family_rows
        ]
        sizes = [8 if item["table_id"] == "paper_table_6" else 7 for item in family_rows]
        customdata = [
            [
                item["method"],
                item["fold_population_sd"],
                item["fold_count"],
                "Table 6" if item["table_id"] == "paper_table_6" else "not in Table 6",
                item["row"],
            ]
            for item in family_rows
        ]
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="markers",
                marker=dict(
                    color=PAPER_FAMILY_COLORS[family],
                    symbol=symbols,
                    size=sizes,
                    opacity=0.88,
                    line=dict(color="#fff", width=0.7),
                ),
                error_x=dict(
                    type="data",
                    symmetric=True,
                    array=sd,
                    color="rgba(0,0,0,0.16)",
                    thickness=0.8,
                    width=0,
                ),
                name=f"paper · {family}",
                legendgroup=f"paper:{family}",
                # The y-axis already names each family. Keeping paper families
                # out of the pair-model legend prevents a false shared race.
                showlegend=False,
                customdata=customdata,
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>published fold-mean trapezoidal PR area "
                    "%{x:.3f}<br>fold population SD %{customdata[1]:.3f} "
                    "(not a CI)<br>%{customdata[3]}, row %{customdata[4]} · "
                    "%{customdata[2]} folds<extra></extra>"
                ),
            ),
            row=row,
            col=1,
        )

    baseline = next(item for item in methods if item["method"] == "Belief Orig - readers")
    baseline_rows, baseline_offsets = plotted["original belief"]
    baseline_index = baseline_rows.index(baseline)
    baseline_y = PAPER_FAMILY_ORDER.index("original belief") + baseline_offsets[baseline_index]
    best = min(
        (
            item
            for item in methods
            if item["fold_mean_trapezoidal_pr_auc"]
            == max(row_["fold_mean_trapezoidal_pr_auc"] for row_ in methods)
        ),
        key=lambda item: (item["fold_population_sd"], item["method_id"]),
    )
    best_rows, best_offsets = plotted[_paper_family(best["method"])]
    best_index = best_rows.index(best)
    best_y = PAPER_FAMILY_ORDER.index(_paper_family(best["method"])) + best_offsets[best_index]
    for item, y, label, ay in (
        (baseline, baseline_y, "original belief · readers n=1,676 · 0.917", 28),
        (best, best_y, "best RF · all sources n=1,689 · 0.942", 28),
    ):
        fig.add_annotation(
            x=item["fold_mean_trapezoidal_pr_auc"],
            y=y,
            text=label,
            row=row,
            col=1,
            showarrow=True,
            arrowhead=0,
            arrowwidth=0.7,
            arrowcolor="rgba(0,0,0,0.35)",
            ax=0,
            ay=ay,
            font=dict(size=10, color=LABEL),
            bgcolor="rgba(253,252,248,0.88)",
        )

    fig.add_annotation(
        x=0.5,
        y=-0.16,
        xref=f"x{row} domain" if row > 1 else "x domain",
        yref=f"y{row} domain" if row > 1 else "y domain",
        text=(
            "The headline anchors use different statement subsets and folds. "
            "Direct model deltas require a shared eligible-set panel."
        ),
        showarrow=False,
        xanchor="center",
        yanchor="top",
        font=dict(size=10, color=LABEL),
    )

    lower = min(
        item["fold_mean_trapezoidal_pr_auc"] - item["fold_population_sd"]
        for item in methods
    )
    upper = max(
        item["fold_mean_trapezoidal_pr_auc"] + item["fold_population_sd"]
        for item in methods
    )
    fig.update_xaxes(
        range=[max(0, lower - 0.006), min(1, upper + 0.006)],
        tickformat=".3f",
        title_text=(
            "published fold-mean trapezoidal PR area · horizontal bars are fold SD, not 95% CI"
        ),
        row=row,
        col=1,
    )
    fig.update_yaxes(
        range=[-0.55, len(PAPER_FAMILY_ORDER) - 0.45],
        tickmode="array",
        tickvals=list(range(len(PAPER_FAMILY_ORDER))),
        ticktext=PAPER_FAMILY_ORDER,
        title_text="paper method family",
        row=row,
        col=1,
    )


def main():
    runs = load_runs()
    paper_methods = load_paper_methods()
    n = max((r["n_gold"] for r in runs), default=0)
    color_of = {r["name"]: PALETTE[i % len(PALETTE)] for i, r in enumerate(runs)}
    ylo = max(0.0, min(r["f1_lo"] for r in runs) - 0.02)
    yhi = min(1.0, max(r["f1_hi"] for r in runs) + 0.02)
    row_count = 3 if paper_methods else 2
    subplot_titles = [
        "Evidence-pair frontier · cost",
        "Evidence-pair frontier · model size",
    ]
    if paper_methods:
        subplot_titles.append(PAPER_SUBPLOT_TITLE)
    fig = make_subplots(
        rows=row_count,
        cols=1,
        vertical_spacing=0.10 if paper_methods else 0.17,
        row_heights=[0.28, 0.28, 0.44] if paper_methods else None,
        subplot_titles=subplot_titles,
    )
    add_panel(fig, runs, "cost", 1, color_of, (ylo, yhi), show_legend=True)
    add_panel(fig, runs, "size", 2, color_of, (ylo, yhi), show_legend=False)
    if paper_methods:
        add_paper_panel(fig, paper_methods, 3)
    title_suffix = " · paper reference below" if paper_methods else ""
    fig.update_layout(
        template="plotly_white", width=WIDTH, height=1320 if paper_methods else HEIGHT,
        title_text=(
            f"Pair-triage diagnostic — {SUBSTRATE_LABEL} (curated pairs n={n})"
            f"{title_suffix}"
        ),
        title_x=0.5, margin=dict(l=MARGIN_L, r=MARGIN_R, t=70, b=60),
        legend=dict(font=dict(size=10), x=1.015, xanchor="left", y=1, yanchor="top",
                    itemsizing="constant", tracegroupgap=2,
                    title=dict(text="pair model", font=dict(size=11))),
    )
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.write_html(OUT, include_plotlyjs="inline", full_html=True,
                   config=dict(displaylogo=False, responsive=True))
    paper_count = paper_methods["method_count"] if paper_methods else 0
    print(
        f"wrote {OUT}  ·  {len(runs)} pair models  ·  curated pairs n={n}  ·  "
        f"{paper_count} paper reference methods"
    )


if __name__ == "__main__":
    main()
