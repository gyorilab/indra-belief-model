from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from indra.belief import SimpleScorer
from indra.statements import Activation, Agent, Evidence


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import score_current_indra_hierarchy_paper as adapter  # noqa: E402


def test_reader_hierarchy_projection_filters_direct_and_inherited_database_evidence() -> None:
    reach_direct = Evidence(source_api="reach", text="A activates B.")
    signor_direct = Evidence(source_api="signor", text="SIGNOR A activates B.")
    statement = Activation(
        Agent("A"), Agent("B"), evidence=[reach_direct, signor_direct]
    )
    reach_extra = Evidence(source_api="sparser", text="A induces B.")
    signor_extra = Evidence(source_api="signor", text="SIGNOR A induces B.")

    actual = adapter._score_reader_projection(
        [statement], [[reach_extra, signor_extra]]
    )
    projected = adapter._project_statement(statement, set(adapter.bayes.READER_SOURCES))
    expected = SimpleScorer().score_statements([projected], [[reach_extra]])

    assert actual == expected
    assert projected.evidence == [reach_direct]


def test_pickle_descriptor_requires_exact_release_digest_and_graph_designation(
    tmp_path: Path,
) -> None:
    path = tmp_path / "corpus.pkl"
    path.write_bytes(b"frozen-object-graph")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {
        "files": [
            {
                "filename": "indra_benchmark_corpus.pkl",
                "bytes": path.stat().st_size,
                "sha256": digest,
                "canonical_for_historical_object_and_refinement_parity": True,
            }
        ]
    }

    descriptor = adapter._pickle_descriptor(path, manifest)
    assert descriptor["sha256"] == digest
    assert descriptor["verification"] == "pass_before_trusted_unpickle"

    bad = json.loads(json.dumps(manifest))
    bad["files"][0]["sha256"] = "0" * 64
    try:
        adapter._pickle_descriptor(path, bad)
    except adapter.ContractError:
        pass
    else:  # pragma: no cover - explicit failure message is clearer than helper import
        raise AssertionError("a mismatched pickle digest was accepted")

