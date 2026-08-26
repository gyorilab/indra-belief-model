"""Behavioral pins for ``src/indra_belief/data/corpus.py::CorpusIndex.get``."""

from __future__ import annotations

import json

from indra.statements import Activation, Agent, Evidence, stmts_to_json

from indra_belief.data.corpus import CorpusIndex


SHARED_HASH = 4_242_424_242
SHARED_TEXT = "A shared sentence supports two distinct interpretations."


def _evidence() -> Evidence:
    evidence = Evidence(source_api="test", text=SHARED_TEXT)
    evidence.source_hash = SHARED_HASH
    return evidence


def _index(tmp_path) -> CorpusIndex:
    first = Activation(
        Agent("MAPK1"), Agent("JUN"), evidence=[_evidence()]
    )
    second = Activation(
        Agent("AKT1"), Agent("FOXO3"), evidence=[_evidence()]
    )
    corpus_path = tmp_path / "synthetic_corpus.json"
    corpus_path.write_text(
        json.dumps(stmts_to_json([first, second])), encoding="utf-8"
    )
    return CorpusIndex(corpus_path)


def test_exact_agent_match_selects_nonfirst_statement_for_shared_source_hash(
    tmp_path,
):
    index = _index(tmp_path)

    result = index.get(SHARED_HASH, "AKT1", "FOXO3")

    assert result is not None
    statement, evidence = result
    assert [agent.name for agent in statement.agent_list()] == ["AKT1", "FOXO3"]
    assert evidence.text == SHARED_TEXT


def test_unmatched_agent_names_fall_back_to_first_statement_with_shared_evidence(
    tmp_path,
):
    """Pin the insertion-order fallback and its measured blast radius.

    It fired at 0/1606 on eval_curation_v1, 0/578 on
    external_curator_gold_v1, and 0/1084 on v2.
    """
    index = _index(tmp_path)

    result = index.get(SHARED_HASH, "NOBODY", "MISSING")

    assert result is not None
    statement, evidence = result
    assert [agent.name for agent in statement.agent_list()] == ["MAPK1", "JUN"]
    assert evidence.text == SHARED_TEXT


def test_unknown_source_hash_returns_none_without_fallback(tmp_path):
    index = _index(tmp_path)

    assert index.get(SHARED_HASH + 1, "AKT1", "FOXO3") is None
