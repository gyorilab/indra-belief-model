"""Regression: the monolithic scorer's few-shot example bank must actually load.

A path bug (or a missing/misplaced data file) silently strips every type-specific
few-shot example, degrading scoring quality with no error. These tests pin the bank
path and assert it is populated, including coverage for non-base statement types.
"""
from indra_belief.scorers.monolithic import scorer


def test_example_bank_path_exists():
    assert scorer._EXAMPLE_BANK_PATH.exists(), (
        f"few-shot example bank missing at {scorer._EXAMPLE_BANK_PATH}"
    )


def test_raw_bank_non_empty():
    assert scorer._RAW_BANK, "_RAW_BANK loaded empty — bank failed to load"


def test_type_bank_non_empty():
    assert scorer._TYPE_BANK, "_TYPE_BANK is empty — no type → pairs mapping built"


def test_non_base_type_has_coverage():
    # Translocation and IncreaseAmount are non-base (beyond Activation/Inhibition/
    # Phosphorylation) types that must carry their own few-shot pairs.
    for stmt_type in ("Translocation", "IncreaseAmount"):
        pairs = scorer._TYPE_BANK.get(stmt_type)
        assert pairs, f"no example-bank coverage for {stmt_type}"
