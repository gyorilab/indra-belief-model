"""The live and batch producers build ONE request value — proven, not assumed.

Before ``indra_belief.prepared_execution`` there was nothing to assert this
against: the live scorer assembled its request in ``scorer._build_messages`` +
``ScoringRecord.format_user_message``, the batch replay assembled its own in
``ReplayIndex.main_request`` + ``ReplayIndex._record``, and the only evidence
they agreed was that a paid run had not yet failed its digest check. The two
halves could not be compared because neither produced a value — each spoke
straight to ``client.call``.

Now both produce a ``PreparedExecution``, so the claim is testable
element-for-element: same system, same every message, same transport params, on
the plain route, the tool route, and the relation-note path.

The second half is the cheap decisive gate on the refactor itself. The frozen
substrate ``data/comparison/grounding_replay/manifest.json`` commits to the
digests of the prompt components the LIVE renderer produces — 2 main system
prompts, 12 per-statement-type few-shot prefixes, and 1 relation system prompt.
If a byte of the assembly moved, one of those 15 stops reproducing. The substrate
is a gitignored published artifact, so those tests skip where it is absent; the
parity half above runs everywhere.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from indra_belief.hashing import canonical_sha256
from indra_belief.prepared_execution import (
    ExecutionBody,
    PreparedCall,
    PreparedExecution,
    assert_replay_digests,
    prepare_from_record,
    prepare_from_replay_row,
    prompt_sha256,
)

_ROOT = Path(__file__).resolve().parents[1]
_MANIFEST = _ROOT / "data" / "comparison" / "grounding_replay" / "manifest.json"

requires_substrate = pytest.mark.skipif(
    not _MANIFEST.exists(),
    reason="data/comparison/grounding_replay absent (gitignored published artifact)",
)

_EVIDENCE_TEXT = "PRKAA1 phosphorylates ACACA at Ser79 in cultured hepatocytes."
_LOOKUPS = ("PRKAA1 -> HGNC:9376 (query is a known alias: True)",
            "ACACA -> HGNC:84 (query is a known alias: True)")
_NOTE = "Relation nature (resolved): the evidence asserts a cascade."


class _Profile:
    """A minimal ScoringProfile. The real one is `scorer.ScoringVariant`; using a
    stub here keeps the parity claim about the PRODUCERS rather than about which
    prompt happens to be the default."""

    name = "parity_profile"
    system_prompt = "SYSTEM PROMPT BODY"
    structured = True

    @staticmethod
    def render_example(example):
        return f"U:{example['claim']}", f"A:{example['verdict']}"


_EXAMPLES = (
    {"claim": "A [Phosphorylation] B", "verdict": "correct"},
    {"claim": "C [Phosphorylation] D", "verdict": "incorrect"},
)
_GUIDANCE = "\n\nEXTERNAL LOOKUP CONTEXT — guidance body.\n"


def _record():
    """A real ScoringRecord with entity resolution suppressed.

    ``__post_init__`` grounds through Gilda; the pair below has no reader
    ``raw_text`` and no entities, so every downstream renderer — format_claim,
    format_entity_context, format_provenance, _abbreviation_alias_lines — still
    runs for real and returns the empty parts this comparison needs.
    """
    from indra.statements import Agent, Evidence, Phosphorylation

    from indra_belief.data.scoring_record import ScoringRecord

    statement = Phosphorylation(Agent("PRKAA1"), Agent("ACACA"),
                                residue="S", position="79")
    evidence = Evidence(source_api="reach", text=_EVIDENCE_TEXT)
    original = ScoringRecord.resolve_entities
    ScoringRecord.resolve_entities = lambda self: None
    try:
        record = ScoringRecord(statement=statement, evidence=evidence)
    finally:
        ScoringRecord.resolve_entities = original
    record.subject_entity = record.object_entity = None
    return record


def _pair(route: str, lookups: tuple[str, ...] = ()):
    """The same request, built once from a live record and once from a row.

    The row's fields are taken from the record's own rendered parts — that is
    exactly the relationship the real substrate has to the live scorer, and it is
    what makes an inequality below a real divergence rather than a typo.
    """
    record = _record()
    live = prepare_from_record(
        record, _Profile(), route=route, examples=_EXAMPLES, lookups=lookups,
        lookup_guidance=_GUIDANCE, max_tokens=4096,
    )
    prefix = [dict(message) for message in live.prefix]
    system = live.system
    row = {
        "route": route,
        "claim": record.format_claim(),
        "entity_context": record.format_entity_context(),
        "abbreviation_lines": record._abbreviation_alias_lines(),
        "provenance": "",
        "evidence_metadata": {"text": record.evidence_text},
        "lookup_refs": [f"ref{index}" for index in range(len(lookups))],
        "main_system_ref": hashlib.sha256(system.encode("utf-8")).hexdigest(),
        "main_message_prefix_ref": canonical_sha256(prefix),
    }
    batch = prepare_from_replay_row(
        row,
        systems={row["main_system_ref"]: system},
        prefixes={row["main_message_prefix_ref"]: prefix},
        lookups={f"ref{index}": text for index, text in enumerate(lookups)},
        max_tokens=4096,
    )
    return record, row, live, batch


# --------------------------------------------------------------------------
# One value, two producers
# --------------------------------------------------------------------------

@pytest.mark.parametrize("route,lookups,note", [
    ("plain", (), ""),
    ("plain", (), _NOTE),
    ("tool", _LOOKUPS, ""),
    ("tool", _LOOKUPS, _NOTE),
])
def test_live_and_batch_producers_agree_call_for_call(route, lookups, note):
    _, _, live, batch = _pair(route, lookups)
    live_calls, batch_calls = live.calls(note), batch.calls(note)
    assert len(live_calls) == len(batch_calls) == 1
    for mine, theirs in zip(live_calls, batch_calls):
        assert mine.system == theirs.system
        assert len(mine.messages) == len(theirs.messages)
        for a, b in zip(mine.messages, theirs.messages):
            assert dict(a) == dict(b)
        assert mine.kind == theirs.kind
        assert mine.max_tokens == theirs.max_tokens
        assert mine.temperature == theirs.temperature
        assert mine.response_format == theirs.response_format
        assert mine.reasoning_effort == theirs.reasoning_effort
        assert mine.prompt_sha256() == theirs.prompt_sha256()
        assert mine.client_kwargs() == theirs.client_kwargs()


def test_the_body_is_the_same_five_parts_on_both_sides():
    record, row, live, batch = _pair("plain")
    assert live.body == batch.body
    assert live.body == ExecutionBody(
        claim=row["claim"], entity_context="", abbreviation_lines=(),
        provenance="", evidence_text=_EVIDENCE_TEXT,
    )
    rendered = live.body.render()
    assert rendered.startswith("CLAIM: ")
    assert rendered.endswith(f'EVIDENCE: "{_EVIDENCE_TEXT}"')
    # The claim carries the modification site, so this is not a degenerate body.
    assert "@S79" in row["claim"]


def test_the_splice_happens_once_and_in_one_order():
    _, _, live, _ = _pair("tool", _LOOKUPS)
    user = live.calls(_NOTE)[-1].messages[-1]["content"]
    base = live.body.render()
    assert user == (base + "\n\n" + _NOTE + "\n\nEntity database lookups:\n"
                    + "\n".join(_LOOKUPS))
    # An empty note inserts nothing at all — the substrate's
    # `empty_note_inserts_prefix: false` coordinate.
    assert live.calls("")[-1].messages[-1]["content"] == (
        base + "\n\nEntity database lookups:\n" + "\n".join(_LOOKUPS))


def test_tool_route_system_is_the_plain_system_plus_the_guidance():
    _, _, plain, _ = _pair("plain")
    _, _, tool, _ = _pair("tool", _LOOKUPS)
    assert tool.system == plain.system + _GUIDANCE
    assert plain.calls()[-1].kind == "monolithic"
    assert tool.calls()[-1].kind == "monolithic_tool_context"


def test_transport_params_travel_with_the_call():
    _, _, live, _ = _pair("plain")
    kwargs = live.calls()[-1].client_kwargs()
    assert set(kwargs) == {"system", "messages", "max_tokens", "temperature", "kind"}
    assert kwargs["max_tokens"] == 4096 and kwargs["temperature"] == 0.1
    assert isinstance(kwargs["messages"], list)
    assert all(isinstance(message, dict) for message in kwargs["messages"])
    # Every call recomputes its own provenance digest.
    assert live.calls()[-1].prompt_sha256() == prompt_sha256(
        kwargs["system"], kwargs["messages"])


def test_the_relation_sub_call_leads_the_topology():
    relation = PreparedCall(
        kind="relation_nature", system="REL", messages=({"role": "user", "content": "q"},),
        max_tokens=3000, temperature=0.1, response_format={"type": "json_object"},
        reasoning_effort="none",
    )
    _, _, live, _ = _pair("plain")
    with_relation = PreparedExecution(
        route=live.route, system=live.system, body=live.body, prefix=live.prefix,
        max_tokens=live.max_tokens, relation=relation,
    )
    assert with_relation.call_topology == ("relation_nature", "monolithic")
    assert with_relation.calls(_NOTE)[0] is relation
    assert with_relation.calls(_NOTE)[-1] == live.calls(_NOTE)[-1]
    assert with_relation.calls()[-1].client_kwargs()["kind"] == "monolithic"
    # The relation call carries the two transport fields the main call omits.
    assert set(relation.client_kwargs()) == {
        "system", "messages", "max_tokens", "temperature", "response_format",
        "reasoning_effort", "kind",
    }


def test_a_deterministic_route_reaches_no_model():
    _, _, live, _ = _pair("plain")
    deterministic = PreparedExecution(
        route="no_text", system=live.system, body=live.body, prefix=live.prefix,
    )
    assert deterministic.calls() == ()
    assert deterministic.call_topology == ()


# --------------------------------------------------------------------------
# The digest contract, held once
# --------------------------------------------------------------------------

def test_assert_replay_digests_accepts_and_rejects_the_no_note_path():
    _, row, _, batch = _pair("plain")
    row["main_prompt_base_sha256"] = batch.calls()[-1].prompt_sha256()
    assert_replay_digests(batch, row)  # the committed request reproduces
    row["main_prompt_base_sha256"] = "0" * 64
    with pytest.raises(Exception, match="hydrated main prompt digest differs"):
        assert_replay_digests(batch, row)


def test_assert_replay_digests_checks_the_note_insertion_coordinates():
    _, row, _, batch = _pair("plain")
    row["relation_note_insertion"] = {
        "message_index": len(batch.prefix),
        "role": "user",
        "utf8_byte_offset": len(batch.body.render().encode("utf-8")),
        "prefix_if_nonempty": "\n\n",
        "empty_note_inserts_prefix": False,
    }
    assert_replay_digests(batch, row, relation_note=_NOTE)
    row["relation_note_insertion"]["utf8_byte_offset"] += 1
    with pytest.raises(Exception, match="relation-note insertion coordinates differ"):
        assert_replay_digests(batch, row, relation_note=_NOTE)


def test_an_absent_component_ref_is_a_replay_error():
    _, row, _, _ = _pair("plain")
    with pytest.raises(Exception, match="main prompt references an absent component"):
        prepare_from_replay_row(row, systems={}, prefixes={}, lookups={})


# --------------------------------------------------------------------------
# The frozen substrate digests the live renderer must still reproduce
# --------------------------------------------------------------------------

def _contract() -> dict:
    return json.loads(_MANIFEST.read_text(encoding="utf-8"))["generation_contract"]


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@requires_substrate
def test_live_renderer_reproduces_the_two_frozen_system_digests():
    from indra_belief.scorers.monolithic import scorer as S

    contract = _contract()
    variant = S.VARIANTS[contract["mono_variant"]]
    assert _sha(variant.system_prompt) == contract["plain_main_system_ref"]
    assert _sha(variant.system_prompt + S._LOOKUP_GUIDANCE) == \
        contract["tool_main_system_ref"]


@requires_substrate
def test_live_renderer_reproduces_the_twelve_frozen_prefix_digests():
    from indra_belief.scorers.monolithic import scorer as S

    contract = _contract()
    variant = S.VARIANTS[contract["mono_variant"]]
    refs = contract["message_prefix_refs_by_statement_type"]
    assert len(refs) == 12
    mismatched = []
    for statement_type, ref in sorted(refs.items()):
        execution = S._prepare(_record(), _select_for(S, statement_type),
                               route="plain", variant=variant)
        if canonical_sha256([dict(m) for m in execution.prefix]) != ref:
            mismatched.append(statement_type)
    assert mismatched == []


def _select_for(scorer_module, statement_type):
    return scorer_module._select_examples(statement_type)


@requires_substrate
def test_live_relation_system_reproduces_the_frozen_relation_digest():
    from indra_belief.scorers.monolithic import _prompts_relation

    assert _sha(_prompts_relation._RELATION_SYSTEM) == _contract()["relation_system_ref"]
