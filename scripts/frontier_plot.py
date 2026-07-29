#!/usr/bin/env python3
"""Render ``frontier_table.py`` JSON as a self-contained Plotly HTML figure.

The renderer is deliberately calculation-free: Pareto membership, confidence
intervals, coverage, and normalized costs all come from the audited table JSON.

    PYTHONPATH=src .venv/bin/python scripts/frontier_plot.py \
        data/results/representative_indra_expanded_403_20260717_frontier.json \
        --output reports/representative_indra_expanded_403_20260717_frontier.html
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    ROOT
    / "data/results/representative_indra_expanded_403_20260717_frontier.json"
)
DEFAULT_OUTPUT = (
    ROOT
    / "reports/representative_indra_expanded_403_20260717_frontier.html"
)

PAPER = "#fdfcf8"
INK = "#252321"
MUTED = "#6b6863"
RULE = "#ded9cf"
FRONTIER = "#9d2f24"
ELIGIBLE = "#386b8c"
ESTIMATE = "#b27320"
INELIGIBLE = "#8b8b88"


def _number(value: Any, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError, OverflowError):
        return "—"


def _percent(value: Any) -> str:
    try:
        return f"{float(value):.1%}"
    except (TypeError, ValueError, OverflowError):
        return "—"


def _money(value: Any, *, estimate: bool = False, suffix: str = "") -> str:
    try:
        prefix = "~" if estimate else ""
        return f"{prefix}${float(value):.2f}{suffix}"
    except (TypeError, ValueError, OverflowError):
        return "—"


def _is_estimate(row: dict[str, Any]) -> bool:
    return bool(row.get("cost_is_estimate") or row.get("cost_basis") == "estimate")


def _formal_names(report: dict[str, Any]) -> set[str]:
    return set((report.get("pareto") or {}).get("formal_frontier") or [])


def _is_formal_pareto(row: dict[str, Any], names: set[str]) -> bool:
    return bool(row.get("pareto_point") is True or row.get("name") in names)


def _category(row: dict[str, Any], formal_names: set[str]) -> str:
    if row.get("frontier_eligible") and _is_formal_pareto(row, formal_names):
        return "Formal Pareto"
    if _is_estimate(row):
        return "Estimated / proxy cost"
    if row.get("frontier_eligible"):
        return "Eligible, dominated"
    return "Formally ineligible"


STYLES = {
    "Formal Pareto": dict(color=FRONTIER, symbol="diamond", size=13),
    "Eligible, dominated": dict(color=ELIGIBLE, symbol="circle", size=10),
    "Estimated / proxy cost": dict(
        color=ESTIMATE, symbol="diamond-open", size=12
    ),
    "Formally ineligible": dict(color=INELIGIBLE, symbol="x-open", size=11),
}


def _customdata(row: dict[str, Any], denominator: int) -> list[str]:
    estimate = _is_estimate(row)
    n = row.get("n")
    coverage = (
        f"{n}/{denominator} ({_percent(row.get('coverage_fraction'))})"
        if n is not None
        else "—"
    )
    ci = (
        f"[{_number(row.get('lo'))}, {_number(row.get('hi'))}]"
        if row.get("lo") is not None and row.get("hi") is not None
        else "unavailable"
    )
    reasons = ", ".join(row.get("frontier_exclusion_reasons") or []) or "none"
    return [
        str(row.get("name") or "unnamed"),
        ci,
        _number(row.get("precision")),
        _number(row.get("recall")),
        _number(row.get("accuracy")),
        coverage,
        _money(row.get("usd_1k"), estimate=estimate, suffix=" / 1k"),
        _money(row.get("cost_usd"), estimate=estimate, suffix=" / run"),
        str(row.get("cost_basis") or "unavailable"),
        "yes" if row.get("frontier_eligible") else "no",
        reasons,
    ]


HOVER = (
    "<b>%{customdata[0]}</b><br>"
    "Error-detection F1 %{y:.3f} &nbsp; 95% CI %{customdata[1]}<br>"
    "Precision %{customdata[2]} &nbsp; Recall %{customdata[3]} &nbsp; "
    "Accuracy %{customdata[4]}<br>"
    "Valid coverage %{customdata[5]}<br>"
    "Cost %{customdata[6]} &nbsp; %{customdata[7]} "
    "(%{customdata[8]})<br>"
    "Formal-frontier eligible: %{customdata[9]}<br>"
    "Exclusions: %{customdata[10]}<extra></extra>"
)


def build_figure(report: dict[str, Any]) -> go.Figure:
    """Build a figure without recomputing metrics or Pareto membership."""
    models = report.get("models")
    if not isinstance(models, list):
        raise ValueError("frontier report must contain a models list")
    denominator = int(report.get("denominator") or 0)
    if denominator <= 0:
        raise ValueError("frontier report denominator must be positive")

    plotted = [
        row
        for row in models
        if isinstance(row, dict)
        and isinstance(row.get("f1"), (int, float))
        and isinstance(row.get("usd_1k"), (int, float))
        and float(row["usd_1k"]) > 0
    ]
    if not plotted:
        raise ValueError("frontier report has no models with positive cost and F1")

    formal_names = _formal_names(report)
    formal = sorted(
        [
            row
            for row in plotted
            if row.get("frontier_eligible")
            and _is_formal_pareto(row, formal_names)
        ],
        key=lambda row: float(row["usd_1k"]),
    )
    fig = go.Figure()

    if len(formal) >= 2:
        fig.add_trace(
            go.Scatter(
                x=[row["usd_1k"] for row in formal],
                y=[row["f1"] for row in formal],
                mode="lines",
                line=dict(color=FRONTIER, width=2.2, shape="hv"),
                name="Formal Pareto frontier",
                hoverinfo="skip",
                legendrank=1,
            )
        )

    for legendrank, category in enumerate(STYLES, 2):
        rows = [row for row in plotted if _category(row, formal_names) == category]
        if not rows:
            continue
        style = STYLES[category]
        is_front = category == "Formal Pareto"
        fig.add_trace(
            go.Scatter(
                x=[row["usd_1k"] for row in rows],
                y=[row["f1"] for row in rows],
                mode="markers+text" if is_front else "markers",
                text=[str(row["name"]) for row in rows] if is_front else None,
                textposition=(
                    ["top center" if index % 2 == 0 else "bottom center"
                     for index, _ in enumerate(rows)]
                    if is_front
                    else None
                ),
                textfont=dict(color=INK, size=11),
                cliponaxis=False,
                marker=dict(
                    **style,
                    line=dict(color=PAPER if is_front else style["color"], width=1.2),
                ),
                error_y=dict(
                    type="data",
                    symmetric=False,
                    array=[
                        round(max(0.0, float(row["hi"]) - float(row["f1"])), 12)
                        if row.get("hi") is not None
                        else 0.0
                        for row in rows
                    ],
                    arrayminus=[
                        round(max(0.0, float(row["f1"]) - float(row["lo"])), 12)
                        if row.get("lo") is not None
                        else 0.0
                        for row in rows
                    ],
                    color="rgba(45,43,40,0.32)",
                    thickness=1.2,
                    width=3,
                ),
                customdata=[_customdata(row, denominator) for row in rows],
                hovertemplate=HOVER,
                name=category,
                legendrank=legendrank,
            )
        )

    baseline = report.get("always_incorrect_baseline") or {}
    baseline_f1 = baseline.get("f1")
    xs = [float(row["usd_1k"]) for row in plotted]
    if isinstance(baseline_f1, (int, float)):
        baseline_hover = (
            "<b>Always incorrect</b><br>"
            f"Error-detection F1 {_number(baseline_f1)}<br>"
            f"Precision {_number(baseline.get('precision'))} &nbsp; "
            f"Recall {_number(baseline.get('recall'))} &nbsp; "
            f"Accuracy {_number(baseline.get('accuracy'))}<br>"
            f"n={baseline.get('n') or '—'}<extra></extra>"
        )
        fig.add_trace(
            go.Scatter(
                x=[min(xs), max(xs)],
                y=[baseline_f1, baseline_f1],
                mode="lines",
                line=dict(color=MUTED, width=1.5, dash="dot"),
                name="Always-incorrect baseline",
                hovertemplate=baseline_hover,
                legendrank=20,
            )
        )

    y_values = [float(row["f1"]) for row in plotted]
    y_values += [float(row["lo"]) for row in plotted if row.get("lo") is not None]
    y_values += [float(row["hi"]) for row in plotted if row.get("hi") is not None]
    if isinstance(baseline_f1, (int, float)):
        y_values.append(float(baseline_f1))
    y_min = max(0.0, min(y_values) - 0.04)
    y_max = min(1.0, max(y_values) + 0.04)
    if y_max - y_min < 0.12:
        midpoint = (y_min + y_max) / 2
        y_min, y_max = max(0.0, midpoint - 0.06), min(1.0, midpoint + 0.06)

    omitted = [
        str(row.get("name") or "unnamed")
        for row in models
        if isinstance(row, dict) and row not in plotted
    ]
    caveat = (report.get("pareto") or {}).get("price_caveat") or ""
    footer = (
        "Formal Pareto uses cost and F1 point estimates; 95% CIs are descriptive. "
        "~ marks estimated/proxy cost, excluded from the formal frontier."
    )
    if omitted:
        footer += " Unplottable (missing F1 or positive cost): " + ", ".join(omitted) + "."
    if caveat:
        footer += " " + str(caveat).rstrip(".") + "."

    label = str(report.get("label") or "frontier")
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor=PAPER,
        plot_bgcolor=PAPER,
        width=1100,
        height=690,
        title=dict(
            text=(
                f"Cost vs error-detection F1 — {label}"
                f"<br><sup>evaluation denominator n={denominator}; vertical bars are 95% bootstrap CIs</sup>"
            ),
            x=0.5,
            xanchor="center",
            font=dict(color=INK, size=21),
        ),
        margin=dict(l=80, r=245, t=95, b=125),
        legend=dict(
            x=1.02,
            xanchor="left",
            y=1,
            yanchor="top",
            title=dict(text="model status"),
            font=dict(size=11, color=INK),
            itemsizing="constant",
        ),
        hoverlabel=dict(bgcolor="white", font=dict(color=INK, size=12)),
        annotations=[
            dict(
                x=0,
                y=-0.22,
                xref="paper",
                yref="paper",
                xanchor="left",
                yanchor="top",
                align="left",
                showarrow=False,
                text=footer,
                font=dict(size=10, color=MUTED),
            )
        ],
    )
    fig.update_xaxes(
        type="log",
        title_text="normalized cost (USD per 1,000 benchmark rows, log scale)",
        gridcolor=RULE,
        zeroline=False,
        tickprefix="$",
    )
    fig.update_yaxes(
        title_text="error-detection F1",
        range=[y_min, y_max],
        gridcolor=RULE,
        zeroline=False,
        tickformat=".2f",
    )
    return fig


def write_html(report: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    build_figure(report).write_html(
        output,
        include_plotlyjs="inline",
        full_html=True,
        div_id="cost-performance-frontier",
        config={"displaylogo": False, "responsive": True},
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("-o", "--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = json.loads(args.input.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise ValueError("frontier JSON root must be an object")
    write_html(report, args.output)
    print(f"wrote {args.output} · {len(report.get('models') or [])} model rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
