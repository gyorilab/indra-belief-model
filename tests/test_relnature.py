"""Unit tests for the relation-nature step (resolve_relation_nature).

Gilda is monkeypatched so the decision logic and alias-hint injection are tested
deterministically on the dev machine without the gilda package.
"""
import json

import pytest

from indra_belief.scorers.monolithic import _prompts_relation as R


class _Resp:
    def __init__(self, content):
        self.content = content
        self.raw_text = content


class _Client:
    """Returns a thinking-model-style response carrying the given JSON payload,
    and records the last user message so alias injection can be asserted."""
    def __init__(self, payload: dict):
        self._payload = json.dumps(payload)
        self.last_user = None

    def call(self, **kw):
        self.last_user = kw["messages"][-1]["content"]
        return _Resp("reasoning about it...\n</think>\n" + self._payload)


class _FakeGT:
    def __init__(self, groundings: dict):
        self._g = groundings

    def entity_grounding(self, name):
        return self._g.get(name)


_G = {
    "SMARCB1": {"db": "HGNC", "id": "11103", "name": "SMARCB1", "aliases": ["SMARCB1", "INI1", "BAF47"]},
    "APP": {"db": "HGNC", "id": "620", "name": "APP", "aliases": ["APP"]},
}


@pytest.fixture
def gilda(monkeypatch):
    def _install(groundings=None):
        gt = _FakeGT(groundings if groundings is not None else _G)
        monkeypatch.setattr(R, "_gilda", lambda: gt)
        return gt
    return _install


def _run(payload, subj="SMARCB1", obj="APP", text="some evidence.", client=None):
    client = client or _Client(payload)
    note = R.resolve_relation_nature(subj, obj, "Complex", text, client)
    return note, client


def test_physical_binding_no_note(gilda):
    gilda()
    note, _ = _run({"nature": "physical_binding", "span": "X binds Y"})
    assert note == ""


@pytest.mark.parametrize("nature,frag", [
    ("fusion_construct", "FUSION"),
    ("signaling_cascade", "signaling"),
    ("co_binding_third", "THIRD"),
    ("topic_or_aim", "title/topic"),
    ("other", "not a direct physical"),
])
def test_disqualifiers_emit_note(nature, frag, gilda):
    gilda()
    note, _ = _run({"nature": nature, "span": "s"})
    assert "unsupported" in note and "MISMATCH" in note and frag in note


def test_unparseable_nature_no_note(gilda):
    gilda()
    assert _run({"span": "no nature field"})[0] == ""          # missing nature
    assert _run({"nature": None, "span": "x"})[0] == ""        # null nature


def test_out_of_taxonomy_nature_is_nonbinding(gilda):
    gilda()
    note, _ = _run({"nature": "weird_unlisted", "span": "s"})
    assert "unsupported" in note  # unknown -> default non-binding label, still rejects


def test_alias_hints_injected_into_prompt(gilda):
    gilda()
    _, client = _run({"nature": "physical_binding", "span": "s"})
    assert "also known as" in client.last_user
    assert "INI1" in client.last_user  # SMARCB1's alias surfaced to the model


def test_whitespace_nature_no_note(gilda):
    gilda()
    assert _run({"nature": "   ", "span": "x"})[0] == ""   # punctuation/whitespace -> None -> no note
    assert _run({"nature": "!!", "span": "x"})[0] == ""


def test_self_complex(gilda):
    gilda()
    # homodimer: subj == obj; non-binding nature still rejects, physical binding accepts
    note, _ = _run({"nature": "topic_or_aim", "span": "the MET-MET interaction"},
                   subj="MET", obj="MET", text="The significance of the MET-MET interaction.")
    assert "unsupported" in note
    assert _run({"nature": "physical_binding", "span": "homodimerizes"},
                subj="MET", obj="MET", text="MET homodimerizes upon ligand binding.")[0] == ""


def test_non_hgnc_grounding_only_primary_name(gilda):
    # FPLX/CHEBI entity: entity_grounding returns just the primary name (no get_names)
    gilda({"AKT": {"db": "FPLX", "id": "AKT", "name": "AKT", "aliases": ["AKT"]},
           "FOXO1": {"db": "HGNC", "id": "3819", "name": "FOXO1", "aliases": ["FOXO1"]}})
    note, client = _run({"nature": "physical_binding", "span": "AKT binds FOXO1"},
                        subj="AKT", obj="FOXO1", text="AKT binds FOXO1.")
    assert note == ""  # classification proceeds; degrades to bare name for the family entity


def test_non_complex_and_empty_skip(gilda):
    gilda()
    assert R.resolve_relation_nature("A", "B", "Phosphorylation", "A phosphorylates B.", _Client({})) == ""
    assert R.resolve_relation_nature("A", "B", "Complex", "   ", _Client({})) == ""


def test_degrades_without_gilda(monkeypatch):
    monkeypatch.setattr(R, "_gilda", lambda: None)
    # no alias hints, but classification still works
    note = R.resolve_relation_nature("A", "B", "Complex", "A and B in a pathway.",
                                     _Client({"nature": "signaling_cascade", "span": "pathway"}))
    assert "signaling" in note
    ok = R.resolve_relation_nature("A", "B", "Complex", "A binds B.",
                                   _Client({"nature": "physical_binding", "span": "binds"}))
    assert ok == ""


def test_client_exception_degrades(monkeypatch):
    monkeypatch.setattr(R, "_gilda", lambda: None)
    class _Boom:
        def call(self, **kw):
            raise RuntimeError("transport down")
    assert R.resolve_relation_nature("A", "B", "Complex", "A binds B.", _Boom()) == ""
