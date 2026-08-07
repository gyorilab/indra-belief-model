from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_representative_scoring_corpus import build_corpus  # noqa: E402



# Reads the gitignored local artifact trees; skipped only when they are WHOLLY
# absent (CI, a fresh checkout). A PARTIAL tree is a failure in
# tests/test_local_artifacts.py, never a skip here.
import _local_artifacts as _artifacts

pytestmark = _artifacts.requires()

def test_frozen_representative_corpus_round_trips_all_exact_pairs():
    statements, meta = build_corpus(
        ROOT / "data/benchmark/representative_indra_curations_400.jsonl",
        ROOT / "data/corpora/cogex_evidence_sample.jsonl",
    )

    assert len(statements) == 403
    assert meta["counts"] == {
        "gold_rows": 403,
        "unique_exact_pairs": 403,
        "statements": 403,
        "evidences": 403,
        "gold_correct": 199,
        "gold_incorrect": 204,
    }
    assert meta["validation"] == {
        "statement_hashes_reproduced": 403,
        "source_hashes_reproduced": 403,
        "exact_pair_join": True,
    }


def test_builder_rejects_duplicate_gold_pairs(tmp_path: Path):
    source = json.loads(
        (ROOT / "data/benchmark/representative_indra_curations_400.jsonl")
        .read_text()
        .splitlines()[0]
    )
    gold = tmp_path / "gold.jsonl"
    gold.write_text(json.dumps(source) + "\n" + json.dumps(source) + "\n")

    with pytest.raises(ValueError, match="duplicate exact pairs"):
        build_corpus(gold, ROOT / "data/corpora/cogex_evidence_sample.jsonl")
