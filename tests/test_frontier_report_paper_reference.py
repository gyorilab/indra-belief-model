"""The legacy pair frontier may show paper methods only as a separate reference."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from plotly.subplots import make_subplots


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import frontier_report as report  # noqa: E402


PAPER_METHODS = ROOT / "data/benchmark/indra_paper_2023_published_method_metrics.json"


def test_loads_pinned_paper_reference_without_cross_metric_claims():
    artifact = report.load_paper_methods(PAPER_METHODS)
    assert artifact is not None
    assert artifact["method_count"] == 59
    contract = artifact["metric_contract"]
    assert contract["unit"] == "assembled_statement"
    assert contract["directly_comparable_to_pair_error_f1"] is False
    assert contract["uncertainty_is_confidence_interval"] is False
    assert "Not comparable with pair F1 above" in report.PAPER_SUBPLOT_TITLE
    assert "not a paired delta" in report.PAPER_SUBPLOT_TITLE


def test_paper_panel_has_family_traces_sd_bars_and_reference_annotations():
    artifact = report.load_paper_methods(PAPER_METHODS)
    figure = make_subplots(rows=1, cols=1)
    report.add_paper_panel(figure, artifact, 1)

    assert len(figure.data) == 5
    assert {trace.name for trace in figure.data} == {
        "paper · KNN",
        "paper · SVC",
        "paper · original belief",
        "paper · logistic regression",
        "paper · random forest",
    }
    assert sum(len(trace.x) for trace in figure.data) == 59
    assert all(trace.error_x.type == "data" for trace in figure.data)
    assert all(trace.showlegend is False for trace in figure.data)
    labels = {annotation.text for annotation in figure.layout.annotations}
    assert {
        "original belief · readers n=1,676 · 0.917",
        "best RF · all sources n=1,689 · 0.942",
    } <= labels
    assert any("different statement subsets and folds" in label for label in labels)
    assert "fold SD, not 95% CI" in figure.layout.xaxis.title.text


def test_paper_reference_loader_fails_closed_on_metric_conflation(tmp_path: Path):
    artifact = json.loads(PAPER_METHODS.read_text())
    artifact["metric_contract"]["directly_comparable_to_pair_error_f1"] = True
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(artifact))
    with pytest.raises(ValueError, match="contract mismatch"):
        report.load_paper_methods(bad)


def test_missing_paper_reference_is_optional(tmp_path: Path):
    assert report.load_paper_methods(tmp_path / "missing.json") is None
