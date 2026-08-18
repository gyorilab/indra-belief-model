"""Fitting a stack's calibration must not be able to flatter itself.

Every guard here exists because the failure it prevents produces a NUMBER
rather than an error, and a number is what someone ships.

  * a missing margin imputed as 0.0 drags the curve toward the middle and looks
    like a well-behaved fit;
  * a random-shuffle split lets a failing gate be re-drawn until it passes;
  * a degenerate confusion yields an infinite log-LR, which propagates as inf
    or nan into every belief;
  * a gate reading the point estimate instead of the interval passes noise --
    which is precisely how the deliberation-length signal looked (real
    within-verdict AUROC 0.56-0.72 across 15 arms, no held-out gain).
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location(
        "_fit_under_test", ROOT / "scripts/fit_incall_calibration.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


fit = _load()

from indra_belief.calibration_gate import gate_decision as GATE  # noqa: E402


def _write(tmp_path, rows):
    path = tmp_path / "run.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


# ── row loading ───────────────────────────────────────────────────────────────

def test_a_missing_margin_is_excluded_not_imputed(tmp_path):
    """A margin the reader could not produce is not a margin of zero."""
    rows, skipped = fit.load_rows(_write(tmp_path, [
        {"source_hash": "1", "gold_correct": True, "verdict": "correct",
         "probe_delta_logit": 3.0},
        {"source_hash": "2", "gold_correct": True, "verdict": "correct",
         "probe_delta_logit": None},
    ]))
    assert len(rows) == 1
    assert skipped["no_margin"] == 1
    assert all(r["margin"] != 0.0 for r in rows)


def test_rows_without_gold_or_verdict_are_counted_separately(tmp_path):
    _, skipped = fit.load_rows(_write(tmp_path, [
        {"source_hash": "1", "gold_correct": None, "verdict": "correct",
         "probe_delta_logit": 1.0},
        {"source_hash": "2", "gold_correct": True, "verdict": None,
         "probe_delta_logit": 1.0},
    ]))
    assert skipped["no_gold"] == 1 and skipped["no_verdict"] == 1


def test_a_boolean_margin_is_not_mistaken_for_a_number(tmp_path):
    """bool is an int in Python; True would silently become a margin of 1.0."""
    rows, skipped = fit.load_rows(_write(tmp_path, [
        {"source_hash": "1", "gold_correct": True, "verdict": "correct",
         "probe_delta_logit": True},
    ]))
    assert rows == [] and skipped["no_margin"] == 1


# ── the split ─────────────────────────────────────────────────────────────────

def _rows(n):
    return [{"record_id": str(i), "gold": i % 2 == 0, "verdict": "correct",
             "margin": float(i)} for i in range(n)]


def test_the_split_is_deterministic_across_runs():
    """A re-drawable split turns a held-out gate into a lottery."""
    a = fit.split(_rows(500), 0.3, 0)
    b = fit.split(_rows(500), 0.3, 0)
    assert [r["record_id"] for r in a[1]] == [r["record_id"] for r in b[1]]


def test_the_split_is_disjoint_and_covers_every_row():
    rows = _rows(500)
    f, h = fit.split(rows, 0.3, 0)
    assert len(f) + len(h) == len(rows)
    assert not (set(r["record_id"] for r in f) & set(r["record_id"] for r in h))


def test_the_holdout_is_roughly_the_requested_size():
    _, h = fit.split(_rows(2000), 0.3, 0)
    assert 0.25 < len(h) / 2000 < 0.35


def test_a_different_seed_gives_a_different_split():
    """Determinism must come from the seed, not from ignoring it."""
    _, a = fit.split(_rows(500), 0.3, 0)
    _, b = fit.split(_rows(500), 0.3, 1)
    assert [r["record_id"] for r in a] != [r["record_id"] for r in b]


# ── the belief profile ────────────────────────────────────────────────────────

def test_the_profile_reports_counts_not_only_rates():
    """A rate cannot be re-derived into a count, and the count is what makes a
    later refit auditable."""
    rows = ([{"verdict": "correct", "gold": True}] * 80
            + [{"verdict": "correct", "gold": False}] * 20
            + [{"verdict": "incorrect", "gold": True}] * 10
            + [{"verdict": "incorrect", "gold": False}] * 90)
    profile = fit.belief_profile(rows)
    assert profile["counts"] == {"cc": 80, "ci": 20, "ic": 10, "ii": 90}
    assert profile["sensitivity"] == pytest.approx(80 / 90)
    assert profile["false_positive_rate"] == pytest.approx(20 / 110)
    assert profile["log_lr_confirm"] > 0 > profile["log_lr_reject"], (
        "a confirming read must raise the odds and a rejecting one lower them"
    )


def test_a_degenerate_confusion_refuses_rather_than_emitting_infinity():
    """A perfect reader gives fpr=0 and an infinite log-LR, which would
    propagate as inf/nan into every belief it touches."""
    rows = ([{"verdict": "correct", "gold": True}] * 50
            + [{"verdict": "incorrect", "gold": False}] * 50)
    with pytest.raises(SystemExit, match="degenerate"):
        fit.belief_profile(rows)


# ── the run refuses to fit on too little ──────────────────────────────────────

def test_too_few_rows_refuses_to_produce_a_curve(tmp_path):
    import subprocess

    path = _write(tmp_path, [
        {"source_hash": str(i), "gold_correct": i % 2 == 0, "verdict": "correct",
         "probe_delta_logit": float(i)} for i in range(20)
    ])
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts/fit_incall_calibration.py"),
         "--run", str(path), "--model", "m", "--served-model-id", "s",
         "--out", str(tmp_path / "c.json")],
        cwd=ROOT, capture_output=True, text=True,
        env={**__import__("os").environ, "PYTHONPATH": "src"},
    )
    assert proc.returncode != 0
    assert "noise wearing a calibration" in (proc.stdout + proc.stderr)
    assert not (tmp_path / "c.json").exists(), "a refused fit still wrote an artifact"


# ── the gate ──────────────────────────────────────────────────────────────────
#
# The gate got this wrong once, in the direction that matters: it scored a
# candidate ADDED to the verdict weight, while statement_belief REPLACES the
# verdict weight with it (statement_belief.py:176). The wrong composition looked
# better on the metric being gated on -- +0.0748 AUROC -- while tripling ECE
# (0.0374 -> 0.1241). A gate that evaluates a composition production never runs
# is not a gate, and the number it produces is the most dangerous kind: real,
# reproducible, and about the wrong thing.

def test_ranking_alone_does_not_pass():
    """The incumbent takes two distinct values, so AUROC over huge ties flatters
    any continuous score. Deliberation length cleared exactly this bar."""
    g = GATE(0.05, 0.140, 0.150)
    assert g["ranking"] and not g["scoring"] and not g["pass"]


def test_scoring_alone_does_not_pass():
    g = GATE(-0.01, 0.150, 0.140)
    assert g["scoring"] and not g["ranking"] and not g["pass"]


def test_a_ci_touching_zero_is_not_a_pass():
    """Strictly greater. A CI whose lower bound is 0 has not excluded it."""
    assert GATE(0.0, 0.150, 0.128)["ranking"] is False


def test_an_equal_brier_is_not_an_improvement():
    assert GATE(0.03, 0.150, 0.150)["scoring"] is False


def test_a_candidate_that_also_improves_calibration_needs_no_trade():
    """Nothing was traded away, so the favourability test has nothing to judge."""
    g = GATE(0.03, 0.150, 0.128,
             reliability_incumbent=0.008, reliability_candidate=0.002,
             resolution_incumbent=0.120, resolution_candidate=0.140)
    assert g["favourable"] and g["ratio"] == float("inf") and g["pass"]


def test_a_favourable_calibration_trade_passes():
    """The measured case: +0.0067 reliability cost buys +0.0242 resolution.
    The ECE rule in fit_probe_belief_model.py would refuse this 3.6:1 trade,
    because it was written against an incumbent at ECE 0.2137 where demanding
    better calibration was free."""
    g = GATE(0.03, 0.129, 0.108,
             reliability_incumbent=0.0015, reliability_candidate=0.0082,
             resolution_incumbent=0.1217, resolution_candidate=0.1459)
    assert g["favourable"] and g["pass"]
    assert g["ratio"] > 3.0


def test_an_unfavourable_trade_fails_even_when_brier_improves():
    """The reason Brier alone is not the whole gate. A net-positive average can
    hide a large calibration regression, and a consumer thresholding on belief
    feels reliability directly -- it does not get to enjoy the average."""
    g = GATE(0.03, 0.150, 0.149,
             reliability_incumbent=0.001, reliability_candidate=0.060,
             resolution_incumbent=0.120, resolution_candidate=0.180)
    assert g["scoring"], "Brier did improve"
    assert not g["favourable"], "a 1.0x trade must not clear a 2.0x bar"
    assert not g["pass"]


def test_the_favourability_bar_is_a_parameter_not_a_magic_number():
    args = dict(reliability_incumbent=0.001, reliability_candidate=0.011,
                resolution_incumbent=0.100, resolution_candidate=0.130)
    assert GATE(0.03, 0.150, 0.128, **args, min_favourability=3.0)["favourable"]
    assert not GATE(0.03, 0.150, 0.128, **args, min_favourability=5.0)["favourable"]
