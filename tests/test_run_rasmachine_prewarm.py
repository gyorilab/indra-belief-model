from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_rasmachine_monolithic import (  # noqa: E402
    _build_parser,
    _load_done_keys,
    _prewarm_grounding,
    _unmetered_client,
)
from indra_belief.data.entity import GroundedEntity  # noqa: E402
from indra.ontology.bio import bio_ontology  # noqa: E402


def test_grounding_prewarm_resolves_only_first_real_agent(monkeypatch):
    seen: list[str] = []
    ontology_queries: list[tuple[str, str]] = []

    def fake_resolve(cls, name, raw_text=None):
        seen.append(name)
        return cls(name=name, raw_text=raw_text)

    monkeypatch.setattr(GroundedEntity, "resolve", classmethod(fake_resolve))
    monkeypatch.setattr(
        bio_ontology,
        "get_children",
        lambda namespace, identifier: ontology_queries.append((namespace, identifier)) or [],
    )
    stmts = [
        SimpleNamespace(agent_list=lambda: [None, SimpleNamespace(name="?")]),
        SimpleNamespace(
            agent_list=lambda: [SimpleNamespace(name="EGFR"), SimpleNamespace(name="AKT1")]
        ),
    ]

    assert _prewarm_grounding(stmts) is True
    assert seen == ["EGFR"]
    assert ontology_queries == [("FPLX", "ERK")]


def test_grounding_prewarm_skips_corpus_without_real_agents(monkeypatch):
    monkeypatch.setattr(
        GroundedEntity,
        "resolve",
        classmethod(lambda cls, name, raw_text=None: (_ for _ in ()).throw(AssertionError())),
    )

    assert _prewarm_grounding([SimpleNamespace(agent_list=lambda: [None])]) is False


def test_resume_uses_latest_row_and_retries_errors_and_abstentions(tmp_path):
    output = tmp_path / "run.jsonl"
    rows = [
        {"stmt_i": 0, "evidence_i": 0, "row_status": "scored", "verdict": "correct"},
        {"stmt_i": 1, "evidence_i": 0, "row_status": "error", "verdict": None},
        {"stmt_i": 2, "evidence_i": 0, "row_status": "scored", "verdict": None},
        {"stmt_i": 0, "evidence_i": 0, "row_status": "scored", "verdict": "incorrect"},
    ]
    output.write_text("".join(json.dumps(row) + "\n" for row in rows))
    done, verdicts, errors, parser_nulls, retryable_errors = _load_done_keys(output)
    assert done == {(0, 0)}
    assert verdicts == {"incorrect": 1}
    assert (errors, parser_nulls, retryable_errors) == (1, 1, 1)


def test_generic_runner_rejects_provider_backed_models():
    with pytest.raises(ValueError, match="comparison run"):
        _unmetered_client("bedrock-gemma-4-e2b")


def test_generic_parser_has_no_spend_or_authorization_flags():
    with pytest.raises(SystemExit):
        _build_parser().parse_args(["--spend-ledger", "ledger.ndjson"])
