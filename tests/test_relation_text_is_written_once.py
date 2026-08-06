"""The relation-path text has one owner and still reproduces frozen bytes."""
from __future__ import annotations

import itertools
import json
import re
from pathlib import Path

import pytest

from indra_belief.comparison import replay
from indra_belief.comparison.replay import ReplayIndex, prompt_sha256
from indra_belief.prepared_execution import (
    relation_mismatch_note,
    relation_user_message,
)
from indra_belief.scorers.monolithic import _prompts_relation
from test_prepared_execution_goldens import _replay_index


_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src" / "indra_belief"
_GOLDEN = _ROOT / "tests" / "goldens" / "prepared_execution_goldens.json"
_SUBSTRATE = _ROOT / "data" / "comparison" / "grounding_replay"
_MANIFEST = _SUBSTRATE / "manifest.json"

requires_substrate = pytest.mark.skipif(
    not _MANIFEST.exists(),
    reason="data/comparison/grounding_replay absent (gitignored published artifact)",
)


class _RelationClient:
    def __init__(self, payload: str) -> None:
        self.payload = payload

    def call(self, **_kwargs):
        class Response:
            pass

        response = Response()
        response.content = self.payload
        response.raw_text = self.payload
        return response


def _grounding_shapes(name: str) -> tuple[dict | None, ...]:
    return (
        None,
        {"aliases": []},
        {"aliases": [name]},
        {"aliases": [name.upper(), "Syn1"]},
        {"aliases": [f"a{i}" for i in range(12)]},
    )


def test_the_relation_literals_appear_once_in_src():
    expected_owners = {
        "Relation nature (resolved):": "prepared_execution.py",
        "a gene FUSION / chimeric construct (one molecule)": "prepared_execution.py",
        "What relationship does the sentence assert between": "prepared_execution.py",
        "No evidence sentence — accepted by default (database-sourced).": "verdict.py",
    }
    sources = tuple(_SRC.rglob("*.py"))
    for literal, owner in expected_owners.items():
        matches = [path for path in sources if literal in path.read_text(encoding="utf-8")]
        assert [path.name for path in matches] == [owner], (literal, matches)

    # The shared constant contains only scalars; per-call state stays fresh.
    index = ReplayIndex({}, (), {}, {}, {}, {}, {}, ())
    first = index.deterministic_result({"route": "no_text"})
    second = index.deterministic_result({"route": "no_text"})
    assert first["call_log"] == second["call_log"] == []
    assert first["call_log"] is not second["call_log"]


def test_live_and_batch_relation_text_agree(monkeypatch):
    monkeypatch.setattr(_prompts_relation, "_gilda", lambda: None)
    natures = (
        "fusionconstruct",
        "fusion_construct",
        "signaling_cascade",
        "co_binding_third",
        "topic_or_aim",
        "other",
        "physical_binding",
        "weird_unknown",
        "",
        "  ",
    )
    spans = ("", "acts upstream", "x" * 200, 'has "quotes"', "em — dash — span")
    pairs = (("EGFR", "GRB2"), ("A", "B"), ("α-synuclein", "Tau"))

    note_count = 0
    for nature, span, (subject, object_) in itertools.product(natures, spans, pairs):
        payload = json.dumps({"nature": nature, "span": span})
        live = _prompts_relation.resolve_relation_nature(
            subject,
            object_,
            "Complex",
            "some evidence text",
            _RelationClient(payload),
        )
        batch = replay._relation_note(payload, subject, object_)
        normalized = re.sub(r"[^a-z]", "", nature.lower()) or None
        shared = relation_mismatch_note(normalized, span, subject, object_)
        assert live == batch == shared, (nature, span, subject, object_)
        note_count += 1
    assert note_count == 150
    assert "not direct physical binding" in relation_mismatch_note(
        "weirdunknown", "unknown relation", "A", "B"
    )
    assert relation_mismatch_note("physicalbinding", "binds", "A", "B") == ""

    texts = ("", "short text", "y" * 200, 'text with "quotes" and — dash')
    user_count = 0
    for subject, object_ in pairs:
        subject_groundings = _grounding_shapes(subject)
        object_groundings = _grounding_shapes(object_)
        relation_aliases = {
            **{f"subject-{i}": grounding
               for i, grounding in enumerate(subject_groundings)},
            **{f"object-{i}": grounding
               for i, grounding in enumerate(object_groundings)},
        }
        system_ref = "relation-system"
        system = "RELATION SYSTEM WITNESS"
        index = ReplayIndex(
            {}, (), {system_ref: system}, {}, {}, {}, relation_aliases, ()
        )
        for subject_i, subject_grounding in enumerate(subject_groundings):
            for object_i, object_grounding in enumerate(object_groundings):
                for text in texts:
                    shared = relation_user_message(
                        subject, object_, text, subject_grounding, object_grounding
                    )
                    row = {
                        "relation_system_ref": system_ref,
                        "relation_alias_refs": {
                            "subject": f"subject-{subject_i}",
                            "object": f"object-{object_i}",
                        },
                        "evidence_metadata": {"text": text},
                        "subject_name": subject,
                        "object_name": object_,
                        "relation_prompt_sha256": prompt_sha256(
                            system, [{"role": "user", "content": shared}]
                        ),
                    }
                    _, messages = index.relation_request(row)
                    batch = messages[0]["content"]
                    live = _prompts_relation._user_message(
                        subject, object_, text, subject_grounding, object_grounding
                    )
                    assert live == batch == shared
                    user_count += 1
    assert user_count == 300


def test_the_frozen_golden_notes_still_derive():
    relation_notes = json.loads(_GOLDEN.read_text(encoding="utf-8"))["relation_notes"]
    span = "acts upstream in a shared pathway"
    assert relation_mismatch_note(
        "signalingcascade", span, "WT1", "ZNF224"
    ) == relation_notes["0:0"]
    assert relation_mismatch_note(
        "signalingcascade", span, "VCL", "SORBS1"
    ) == relation_notes["2:0"]


@requires_substrate
def test_every_frozen_relation_prompt_still_hydrates():
    rows: dict[str, dict] = {}
    with (_SUBSTRATE / "executions.jsonl").open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            row = json.loads(line)
            if row.get("relation_prompt_sha256"):
                rows[str(line_number)] = row

    assert len(rows) == 17257
    index = _replay_index(_SUBSTRATE, rows)
    hydrated = 0
    for row in rows.values():
        index.relation_request(row)
        hydrated += 1
    assert hydrated == 17257
