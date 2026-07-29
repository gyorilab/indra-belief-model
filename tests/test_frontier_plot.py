"""Focused rendering contract for the table-driven frontier plot."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import frontier_plot  # noqa: E402


def _model(name, cost, f1, *, eligible=True, pareto=False, estimate=False):
    return {
        "name": name,
        "f1": f1,
        "lo": f1 - 0.03,
        "hi": f1 + 0.02,
        "precision": f1 - 0.01,
        "recall": f1 + 0.01,
        "accuracy": f1 - 0.02,
        "n": 403 if eligible else 390,
        "coverage_fraction": 1.0 if eligible else 390 / 403,
        "usd_1k": cost,
        "cost_usd": cost * 0.403,
        "cost_basis": "estimate" if estimate else "list",
        "cost_is_estimate": estimate,
        "frontier_eligible": eligible and not estimate,
        "frontier_exclusion_reasons": [] if eligible and not estimate else ["cost_not_repo_list_rate"],
        "pareto_point": pareto,
    }


def _report():
    return {
        "schema_version": 1,
        "label": "expanded-403",
        "denominator": 403,
        "pareto": {
            "formal_frontier": ["cheap", "best"],
            "price_caveat": "Repository normalization, not invoice reconciliation",
        },
        "always_incorrect_baseline": {
            "f1": 0.67,
            "precision": 0.51,
            "recall": 1.0,
            "accuracy": 0.51,
            "n": 403,
        },
        "models": [
            _model("cheap", 0.2, 0.70, pareto=True),
            _model("best", 4.0, 0.84, pareto=True),
            _model("dominated", 1.0, 0.72),
            _model("proxy", 0.1, 0.73, eligible=False, estimate=True),
            _model("partial", 0.7, 0.71, eligible=False),
        ],
    }


def test_frontier_plot_encodes_audited_status_ci_baseline_and_hover(tmp_path):
    report = _report()
    fig = frontier_plot.build_figure(report)
    traces = {trace.name: trace for trace in fig.data}

    assert list(traces["Formal Pareto frontier"].x) == [0.2, 4.0]
    assert traces["Formal Pareto frontier"].line.shape == "hv"
    assert traces["Formal Pareto"].marker.symbol == "diamond"
    assert list(traces["Formal Pareto"].error_y.array) == [0.02, 0.02]
    assert traces["Estimated / proxy cost"].marker.symbol == "diamond-open"
    assert traces["Formally ineligible"].marker.symbol == "x-open"
    assert list(traces["Always-incorrect baseline"].y) == [0.67, 0.67]
    assert "Precision" in traces["Formal Pareto"].hovertemplate
    assert "Valid coverage" in traces["Formal Pareto"].hovertemplate
    assert "Cost" in traces["Formal Pareto"].hovertemplate

    output = tmp_path / "frontier.html"
    frontier_plot.write_html(report, output)
    html = output.read_text(encoding="utf-8")
    assert "cost-performance-frontier" in html
    assert "Plotly.newPlot" in html
    assert "expanded-403" in html
    assert "Always-incorrect baseline" in html
    assert len(html) > 1_000_000  # plotly.js is embedded, not fetched at view time
