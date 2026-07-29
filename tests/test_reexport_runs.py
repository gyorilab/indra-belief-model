"""Contracts for mapping scored corpus files back to their gold artifacts."""
from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from reexport_runs import CORPUS_GOLD  # noqa: E402


def test_representative_expanded_corpus_maps_to_unique_pair_gold():
    assert CORPUS_GOLD["representative_indra_expanded_403_20260717_statements.json"] == (
        "data/benchmark/representative_indra_curations_400.jsonl"
    )
