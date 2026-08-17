"""Frozen contract for the monolithic scorer's variant selection.

``scorer.py`` used to resolve ``MONO_VARIANT`` into a module global at import
and branch on it in seven places, so the profile could only be changed by
re-importing the module under a mutated environment and the baseline branch was
unreachable from a test. C0-profile-identity replaces that global with an
immutable ``ScoringVariant`` threaded through the scoring entry points.

This module is the proof the replacement changed nothing. The fixture
``tests/fixtures/monolithic_variant_golden.json`` was captured from a PRISTINE
``git worktree`` of HEAD, before ``scorer.py`` was touched, by running the very
same ``_probe()`` below as a script under ``PYTHONPATH=<pristine>/src``. Each of
the five ``MONO_VARIANT`` values gets its own subprocess, because the pre-change
code could only be switched at import.

What ``_probe`` records is the COMPLETE set of surfaces the variant controlled:
the active system prompt, the assembled messages (with and without a relation
note), ``_relation_note``, the parse of a model reply,
``_stamp_committed_justification`` and — the ones that matter —
``_score_single`` / ``_score_with_tools`` driven with a recording client, so the
``system`` and the full ``messages`` production puts on the wire are frozen byte
for byte, together with the call topology (whether the relation-nature sub-call
fires at all).

The ``parse_verdict`` entry is now read off ``indra_belief.verdict``.
K2-one-parser removed ``scorer._parse_verdict`` — a dispatcher that picked one
of two parsers off the variant — so the profile no longer selects a reader at
all. The values below are unchanged across all five ``MONO_VARIANT`` settings,
which is the proof: the same four replies read the same way whichever profile
produced them, and they read the way the pre-refactor capture did.

Hermeticity: no network, no Gilda, no INDRA. ``indra_belief.tools.gilda_tools``
is replaced by a stub module — installed BOTH in ``sys.modules`` and as the
attribute on the parent package, because ``_prompts_relation._gilda()`` reaches
it by attribute and would otherwise get the real one. The records are plain
stubs rather than ``ScoringRecord`` (whose ``__post_init__`` grounds live).

Regeneration is NOT automatic and must never be done from the modified tree: the
fixture's whole value is that it predates the refactor. Recapturing it would be
re-baselining, not testing. As of K2 the probe cannot run against the pristine
worktree either — it reads the parse off ``indra_belief.verdict``, which does
not exist there — so the fixture is now frozen outright. That is the intended
end state for a value nothing is allowed to move.

The tests below split into two families. The frozen-golden family compares the
whole probed surface, byte for byte, against the pre-refactor capture; it drives
the PRIVATE helpers ``_score_single`` / ``_score_with_tools``. The second family
(everything built on ``_wire``) gates the PUBLIC entry points — ``score``,
``score_statement``, ``score_evidence`` — on the narrower property the golden
cannot see, because the golden was captured when a process had exactly one
profile: the variant handed to a CALL is the variant that reaches the wire on
that call. That is what makes two profiles in one process possible, and it is
the property a ``score()`` that accepted ``variant=`` and quietly dropped it
would violate while every golden still matched.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import types
from contextlib import contextmanager
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_FIXTURE = _ROOT / "tests" / "fixtures" / "monolithic_variant_golden.json"

# The five MONO_VARIANT values. "" is README.md:65-66's documented baseline
# switch; "bogus_typo" is the unrecognized-value path, which resolves to the
# same baseline.
# Deliberately EXCLUDES disconfirm_relnature_rf_noconf. The golden is a frozen
# pre-refactor capture that cannot be regenerated from this tree, so a variant
# added afterwards has no honest entry in it — inventing one would capture from
# the modified tree and turn the gate into a tautology. The new variant is
# covered instead by _VARIANT_KEYS, which asserts registry completeness.
_PROBED_ENVS = ("", "disconfirm", "disconfirm_relnature", "disconfirm_relnature_rf",
                "bogus_typo")
_BASELINE_ENVS = ("", "bogus_typo")
_RELNATURE_ENVS = ("disconfirm_relnature", "disconfirm_relnature_rf",
                   "disconfirm_relnature_rf_noconf")

_MODULE = "indra_belief.scorers.monolithic.scorer"


# --------------------------------------------------------------------------
# Hermetic seams
# --------------------------------------------------------------------------

_MISSING = object()


@contextmanager
def _fake_gilda():
    """Stub ``indra_belief.tools.gilda_tools`` for the duration of a probe.

    Two entry points must be covered: ``_score_with_tools`` ->
    ``_format_entity_lookups`` imports ``lookup_gene_executor`` from the module
    path (sys.modules), while ``_prompts_relation._gilda()`` does
    ``from indra_belief.tools import gilda_tools``, which resolves through the
    PARENT PACKAGE ATTRIBUTE once the real module has been imported. Patching
    only sys.modules would silently reach live Gilda on the second path.
    """
    import indra_belief.tools as tools_pkg  # empty __init__; imports nothing

    name = "indra_belief.tools.gilda_tools"
    fake = types.ModuleType(name)
    fake.entity_grounding = lambda entity_name: None
    fake.lookup_gene_executor = lambda payload: "STUB LOOKUP"

    prev_module = sys.modules.get(name, _MISSING)
    prev_attr = getattr(tools_pkg, "gilda_tools", _MISSING)
    sys.modules[name] = fake
    tools_pkg.gilda_tools = fake
    try:
        yield fake
    finally:
        if prev_module is _MISSING:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = prev_module
        if prev_attr is _MISSING:
            if hasattr(tools_pkg, "gilda_tools"):
                delattr(tools_pkg, "gilda_tools")
        else:
            tools_pkg.gilda_tools = prev_attr


class _StubEntity:
    """Only the two attributes ``_format_entity_lookups`` reads."""

    def __init__(self, name: str, raw_text: str) -> None:
        self.name, self.raw_text = name, raw_text


class _StubRecord:
    """Stands in for ScoringRecord — its ``__post_init__`` grounds via Gilda.

    K1-prepared-execution replaced ``ScoringRecord.format_user_message`` (one
    joined string) with ``execution_body`` (the five parts, joined once by
    ``ExecutionBody.render``). The stub follows the interface; the rendered
    bytes are unchanged, which is what the frozen fixture below proves.
    """

    def __init__(self, stmt_type, subject, object_, text, claim, entities=False):
        self.stmt_type, self.subject, self.object = stmt_type, subject, object_
        self.evidence_text, self._claim = text, claim
        if entities:
            self.subject_entity = _StubEntity(subject, subject.lower())
            self.object_entity = _StubEntity(object_, object_.lower())
        else:
            self.subject_entity = self.object_entity = None

    def execution_body(self):
        from indra_belief.prepared_execution import ExecutionBody

        return ExecutionBody(claim=self._claim, evidence_text=self.evidence_text)

    # The two members `score()` reads that the private helpers never did:
    # scorer.py:581 (`record.tier1_auto_reject()`, the deterministic Tier-1
    # gate) and scorer.py:588 (`record.format_provenance()`). Answering "no
    # auto-reject, no provenance" is what carries a probe past Tier 1 and into
    # the Tier-2 LLM call whose bytes the variant controls. Both are inert on
    # the `_dump()` script path, which never reaches `score()`.

    def tier1_auto_reject(self):
        return None

    def format_provenance(self) -> str:
        return ""


def _records() -> list[_StubRecord]:
    """Three statement types -> three different few-shot banks. Only the first
    carries no grounded entities, so the tool route's lookup block is captured
    both absent (0) and present (1, 2)."""
    return [
        _StubRecord(
            "Phosphorylation", "AAA1", "BBB2",
            "AAA1 phosphorylates BBB2 at the activation loop in vitro.",
            "AAA1 phosphorylates BBB2 [Phosphorylation]",
        ),
        _StubRecord(
            "Complex", "CCC3", "DDD4",
            "CCC3 and DDD4 act downstream of the same receptor.",
            "CCC3 binds DDD4 [Complex]",
            entities=True,
        ),
        _StubRecord(
            "Translocation", "EEE5", "FFF6",
            "EEE5 relocalizes to the nucleus upon FFF6 stimulation.",
            "EEE5 translocates [Translocation]",
            entities=True,
        ),
    ]


_MAIN_REPLY = (
    '{"relation_check": "the sentence states the relation directly", '
    '"support": "the sentence states it", "objection": null, '
    '"verdict": "correct", "confidence": "high"}'
)
_RELATION_REPLY = '{"nature":"cascade","span":"downstream of"}'
_RELATION_KIND = "relation_nature"


class _StubResponse:
    def __init__(self, content: str, raw_text: str | None = None, tokens: int = 7):
        self.content = content
        self.raw_text = content if raw_text is None else raw_text
        self.tokens = tokens
        self.reasoning_trace: dict = {}


class _StubClient:
    """Records every call; answers canned. The relation-nature sub-call is
    answered on its own ``kind`` so a single client serves both call sites."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def call(self, **kwargs):
        self.calls.append({
            "system": kwargs.get("system"),
            "messages": kwargs.get("messages"),
            "kwargs": {k: v for k, v in sorted(kwargs.items())
                       if k not in ("system", "messages")},
        })
        content = (_RELATION_REPLY if kwargs.get("kind") == _RELATION_KIND
                   else _MAIN_REPLY)
        return _StubResponse(content)

    def pop_call_log(self) -> list:
        return []


def _parse_cases() -> list[_StubResponse]:
    """Four responses covering every parser branch on both sides of the fork:
    structured JSON in content; structured JSON only in raw_text; a baseline
    Reason+JSON body; and garbage."""
    return [
        _StubResponse(_MAIN_REPLY),
        _StubResponse("", raw_text=_MAIN_REPLY),
        _StubResponse('Reason: the sentence states it\n'
                      '{"verdict": "correct", "confidence": "high"}'),
        _StubResponse("no verdict anywhere in this text"),
    ]


# --------------------------------------------------------------------------
# The probe
# --------------------------------------------------------------------------

def _probe(mod, variant=None) -> dict:
    """Every surface ``MONO_VARIANT`` controlled, as plain JSON-able data.

    ``variant`` is passed through to the scorer only when given, so one probe
    body serves both the ambient module default and the post-C0 in-process
    injection path.

    ``build_messages`` is now read off ``_prepare(...).calls(note)``, since
    K1-prepared-execution replaced ``_build_messages`` with the one request
    value, and ``parse_verdict`` off ``indra_belief.verdict.parse_response``,
    since K2-one-parser removed the per-variant parser dispatcher. Same
    messages, same parse, same bytes — that equality is the point of the frozen
    fixture, so the key names are kept and the values must not move.
    """
    kw = {} if variant is None else {"variant": variant}

    def parsed_pair(response) -> list:
        """(verdict, confidence) for one reply, read the way production reads it.

        No variant is passed: post-K2 there is one parser and no profile selects
        it. Unlike the ``ACTIVE_SYSTEM_PROMPT`` branch below, no fallback to the
        retired per-variant dispatcher is kept — calling a symbol this node
        deleted is exactly what the node forbids. The consequence is stated
        plainly: this probe no longer runs against a pre-K2 checkout, so the
        fixture is frozen rather than re-derivable, and its values are asserted
        below instead.
        """
        from indra_belief.verdict import parse_response

        read = parse_response(response)
        return [None, None] if read is None else [read.label, read.confidence]

    def messages(record, note: str = "") -> list[dict]:
        return [dict(m) for m in mod._prepare(record, **kw).calls(note)[-1].messages]

    if variant is not None:
        system = variant.system_prompt
    elif hasattr(mod, "DEFAULT_VARIANT"):
        system = mod.DEFAULT_VARIANT.system_prompt
    else:
        # A pre-C0 checkout, where the profile was the module global
        # ACTIVE_SYSTEM_PROMPT. This branch is dead against this tree and is
        # kept only so the fixture can be re-derived from the worktree it was
        # captured in.
        system = mod.ACTIVE_SYSTEM_PROMPT

    records = _records()
    payload: dict = {
        "system_prompt_sha256": hashlib.sha256(system.encode("utf-8")).hexdigest(),
        "system_prompt_len": len(system),
        "build_messages": [messages(r) for r in records],
        "build_messages_with_note": messages(records[0], "NOTE-X"),
        "relation_note": [mod._relation_note(_StubClient(), r, **kw) for r in records],
        "parse_verdict": [parsed_pair(resp) for resp in _parse_cases()],
    }

    stamped = []
    for resp in _parse_cases():
        mod._stamp_committed_justification(resp, **kw)
        stamped.append(resp.reasoning_trace)
    payload["stamp"] = stamped

    for name, fn in (("score_single", mod._score_single),
                     ("score_with_tools", mod._score_with_tools)):
        entries = []
        for record in records:
            client = _StubClient()
            result = fn(client, record, 64, **kw)
            entries.append({"result": result, "calls": client.calls})
        payload[name] = entries

    return payload


def _canonical(payload) -> str:
    return json.dumps(payload, sort_keys=True, ensure_ascii=True, indent=2)


def _dump() -> None:
    """Script entry point — one process, one MONO_VARIANT value."""
    import importlib

    with _fake_gilda():
        mod = importlib.import_module(_MODULE)
        payload = _probe(mod)
    payload["variant_env"] = os.environ.get("MONO_VARIANT", "<unset>")
    sys.stdout.write(_canonical(payload))


if __name__ == "__main__":
    if "--dump" not in sys.argv:
        raise SystemExit("usage: MONO_VARIANT=<value> python "
                         "tests/test_monolithic_variant_profile.py --dump")
    _dump()
    raise SystemExit(0)  # stdout carries the payload and nothing else


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------

import pytest  # noqa: E402  (kept below the script entry point on purpose)

_EXPECTED: dict = (json.loads(_FIXTURE.read_text(encoding="utf-8"))
                   if _FIXTURE.exists() else {})


def _probe_in_subprocess(env_value: str) -> dict:
    """Re-run the probe against the CURRENT tree, one process per value.

    A subprocess is not decoration: the fixture entries were produced by
    exactly this command line against the pristine tree, so a same-shape
    re-run is the only comparison that means anything.
    """
    env = dict(os.environ)
    env["MONO_VARIANT"] = env_value
    env["PYTHONPATH"] = str(_ROOT / "src")
    proc = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--dump"],
        env=env, capture_output=True, text=True, cwd=str(_ROOT),
    )
    assert proc.returncode == 0, f"probe failed for MONO_VARIANT={env_value!r}:\n{proc.stderr}"
    return json.loads(proc.stdout)


@pytest.fixture(scope="module")
def golden() -> dict:
    assert _EXPECTED, (
        f"golden fixture missing: {_FIXTURE}. It is a pre-refactor capture and "
        "cannot be regenerated from this tree."
    )
    return _EXPECTED


@pytest.mark.parametrize("env_value", _PROBED_ENVS)
def test_golden_diff_is_zero(env_value, golden):
    """The gate. Everything the variant controls, re-measured against the
    modified tree and compared to the pre-change capture."""
    want = golden[env_value]
    got = _probe_in_subprocess(env_value)
    moved = sorted(f for f in set(got) | set(want) if got.get(f) != want.get(f))
    assert not moved, (
        f"variant-controlled surface moved for MONO_VARIANT={env_value!r}: "
        f"{moved} differ from the pre-refactor golden"
    )
    assert _canonical(got) == _canonical(want)


def test_golden_covers_every_probed_value(golden):
    assert sorted(golden) == sorted(_PROBED_ENVS)
    for env_value in _PROBED_ENVS:
        assert golden[env_value]["variant_env"] == env_value


def test_in_process_variant_injection_matches_the_golden():
    """The new capability: select a non-default variant with NO environment
    manipulation and no re-import. Impossible before C0 — the profile was fixed
    at the moment the module was first imported."""
    from indra_belief.scorers.monolithic import scorer as S

    # The ambient module default is something else; the injected value wins.
    assert S.DEFAULT_VARIANT.name != "disconfirm"
    with _fake_gilda():
        got = _probe(S, variant=S.VARIANTS["disconfirm"])
    want = {k: v for k, v in _EXPECTED["disconfirm"].items() if k != "variant_env"}
    moved = sorted(f for f in set(got) | set(want) if got.get(f) != want.get(f))
    assert not moved, f"in-process injection diverges from the golden: {moved}"
    assert _canonical(got) == _canonical(want)


def test_in_process_injection_leaves_the_module_default_alone():
    """Injection is an argument, not a mutation: the import-time default is
    unchanged afterwards, which is what keeps a score attributable to one
    prompt for the life of a run."""
    from indra_belief.scorers.monolithic import scorer as S

    before = S.DEFAULT_VARIANT
    with _fake_gilda():
        _probe(S, variant=S.VARIANTS[""])
    assert S.DEFAULT_VARIANT is before


def test_variant_registry_shape():
    from indra_belief.scorers.monolithic import scorer as S

    assert sorted(S.VARIANTS) == ["", "disconfirm", "disconfirm_relnature",
                                  "disconfirm_relnature_rf",
                                  "disconfirm_relnature_rf_noconf"]
    baseline = S.VARIANTS[""]
    assert baseline.structured is False
    assert baseline.resolve_relation_nature is None
    # K2-one-parser: a profile no longer carries a parser at all. `structured`
    # says what the profile ASKS the model for, not who reads the answer.
    assert not hasattr(baseline, "parse_structured")
    assert not hasattr(baseline, "derive_verdict")

    disconfirm = S.VARIANTS["disconfirm"]
    assert disconfirm.structured is True
    assert disconfirm.resolve_relation_nature is None

    for name in _RELNATURE_ENVS:
        variant = S.VARIANTS[name]
        assert variant.structured is True
        assert variant.resolve_relation_nature is not None

    # The two structured prompts are genuinely different, and reasoning-first
    # is the one the default resolves to.
    assert S.VARIANTS["disconfirm_relnature_rf"].system_prompt != \
        S.VARIANTS["disconfirm"].system_prompt
    assert S.DEFAULT_VARIANT is S.VARIANTS[S.DEFAULT_VARIANT_NAME]
    assert S.DEFAULT_VARIANT_NAME == "disconfirm_relnature_rf"


def test_variant_is_immutable():
    from indra_belief.scorers.monolithic import scorer as S

    with pytest.raises(Exception):
        S.DEFAULT_VARIANT.system_prompt = "tampered"


def test_default_variant_prompt_matches_the_calibration_constant():
    """Load-bearing and, until now, unguarded: the reader calibration profiles
    in calibration_constants.py are keyed on (model, prompt_sha256), so the
    shipped constant IS the default variant's system prompt. If they drift, the
    calibration silently applies to a prompt that no longer exists.

    No matching assertion is made for BASELINE_PROMPT_SHA256 — measured, the
    baseline SYSTEM_PROMPT hashes to c6845ab46c0b... and that constant refers to
    a different, historical persisted prompt.
    """
    from indra_belief import calibration_constants
    from indra_belief.scorers.monolithic import scorer as S

    digest = hashlib.sha256(S.DEFAULT_VARIANT.system_prompt.encode("utf-8")).hexdigest()
    assert digest == calibration_constants.REASONING_FIRST_PROMPT_SHA256


def test_unrecognized_variant_falls_back_to_baseline_with_a_warning(caplog):
    """A typo used to select the baseline prompt in total silence. It still
    selects the baseline — the scoring behaviour is unchanged — but it now says
    so, which is the whole difference between a fallback and a bug."""
    import logging

    from indra_belief.scorers.monolithic import scorer as S

    with caplog.at_level(logging.WARNING, logger=S.log.name):
        resolved = S.variant_from_env({"MONO_VARIANT": "bogus_typo"})
    assert resolved is S.VARIANTS[""]
    assert any("bogus_typo" in r.getMessage() for r in caplog.records), caplog.text


def test_empty_variant_selects_baseline_silently(caplog):
    """README.md:65-66 documents MONO_VARIANT="" as the intended baseline
    switch, so it must stay quiet."""
    import logging

    from indra_belief.scorers.monolithic import scorer as S

    with caplog.at_level(logging.WARNING, logger=S.log.name):
        resolved = S.variant_from_env({"MONO_VARIANT": "   "})
    assert resolved is S.VARIANTS[""]
    assert caplog.records == []


def test_variant_from_env_normalizes_and_defaults():
    from indra_belief.scorers.monolithic import scorer as S

    assert S.variant_from_env({}) is S.VARIANTS[S.DEFAULT_VARIANT_NAME]
    assert S.variant_from_env({"MONO_VARIANT": "  DISCONFIRM  "}) is S.VARIANTS["disconfirm"]


def test_score_entry_points_accept_a_variant():
    """score() and score_statement() are the public seams; the keyword has to
    reach them, not just the private helpers.

    ``_parse_verdict`` is absent from the list below because K2-one-parser
    deleted it: it dispatched on the variant to pick one of two parsers, and
    there is now one parser that no profile selects. The case it covered is not
    dropped — ``test_the_parser_is_not_variant_selected`` asserts the stronger
    property that replaced it.
    """
    import inspect

    from indra_belief.scorers.monolithic import scorer as S
    from indra_belief.scorers import monolithic as pkg

    for fn in (S.score, S.score_statement, S._score_single, S._score_with_tools,
               S._prepare, S._stamp_committed_justification,
               S._relation_note, pkg.score_evidence):
        parameter = inspect.signature(fn).parameters.get("variant")
        assert parameter is not None, fn.__qualname__
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY, fn.__qualname__
        assert parameter.default is None, fn.__qualname__


def test_the_parser_is_not_variant_selected():
    """What replaced ``_parse_verdict``'s entry in the list above.

    The old dispatcher took ``variant=`` precisely so it could branch on it, so
    "it accepts a variant" was the property worth pinning. The property worth
    pinning now is the opposite one: the scorer holds no parser to select, and
    the five profiles read an identical reply identically — which the frozen
    ``parse_verdict`` values already show, case by case, across all five.
    """
    from indra_belief.scorers.monolithic import scorer as S
    from indra_belief import verdict as V

    assert not hasattr(S, "_parse_verdict")
    for variant in S.VARIANTS.values():
        assert not hasattr(variant, "parse_structured")
        assert not hasattr(variant, "derive_verdict")
    reply = _StubResponse('{"verdict": "incorrect", "confidence": "low"}')
    read = V.parse_response(reply)
    assert (read.label, read.confidence) == ("incorrect", "low")
    assert not hasattr(read, "score")
    assert all(entry["parse_verdict"] == _EXPECTED[""]["parse_verdict"]
               for entry in _EXPECTED.values())


# --------------------------------------------------------------------------
# The public seam: the variant handed to a CALL is the one that reaches the wire
#
# The golden above drives `_score_single` / `_score_with_tools` directly. These
# gate `score`, `score_statement` and `score_evidence` — the seams a caller
# actually holds — where the keyword has one more layer to survive.
# --------------------------------------------------------------------------

# The registry's keys. Unlike `_PROBED_ENVS` these are the four REAL profiles:
# "bogus_typo" is an env-string that resolves to the baseline, not an entry.
_VARIANT_KEYS = ("", "disconfirm", "disconfirm_relnature", "disconfirm_relnature_rf",
                 "disconfirm_relnature_rf_noconf")


def _wire(invoke, *, variant=_MISSING):
    """Drive one public entry point once and report what reached the wire.

    `invoke(client, **kw)` adapts the entry point's own argument order; `kw`
    carries `variant=` only when one was asked for, so the same helper covers
    the ambient-default control arm.

    Returns `(kinds, main_call)`. `kinds` is the call topology in order — the
    relation-nature sub-call fires FIRST on the relnature profiles, so the
    scoring call is identified by its `kind`, never as `calls[0]`.
    """
    client = _StubClient()
    invoke(client, **({} if variant is _MISSING else {"variant": variant}))
    kinds = [call["kwargs"]["kind"] for call in client.calls]
    main = [call for call in client.calls
            if call["kwargs"]["kind"] != _RELATION_KIND]
    assert len(main) == 1, f"expected exactly one scoring call, saw {kinds}"
    return kinds, main[0]


def test_score_puts_the_requested_variant_on_the_wire():
    """The discriminating gate: same record, two profiles, one process.

    Asserting only that the two arms DIFFER would pass on a `score()` that
    swapped them, so each arm is pinned to the identity of the profile it
    asked for. The third arm passes no variant at all: it is what a `score()`
    that accepted `variant=` and dropped it would record on all three arms, and
    it matches neither of the other two — so the assertions above it are
    load-bearing rather than accidentally true.
    """
    from indra_belief.scorers.monolithic import scorer as S

    before = S.DEFAULT_VARIANT
    record = _records()[0]  # entities=False -> the plain non-tool route

    def drive(client, **kw):
        return S.score(client, record, 64, **kw)

    with _fake_gilda():
        _, baseline = _wire(drive, variant=S.VARIANTS[""])
        _, disconfirm = _wire(drive, variant=S.VARIANTS["disconfirm"])
        _, ambient = _wire(drive)

    assert baseline["system"] != disconfirm["system"]
    # The few-shot renderer is variant-selected too, so the whole assembled
    # body must move, not only the system prompt.
    assert baseline["messages"] != disconfirm["messages"]

    assert baseline["system"] is S.VARIANTS[""].system_prompt
    assert disconfirm["system"] is S.VARIANTS["disconfirm"].system_prompt

    assert ambient["system"] is S.DEFAULT_VARIANT.system_prompt
    assert ambient["system"] != baseline["system"]
    assert ambient["system"] != disconfirm["system"]

    # Passing a variant is an argument, never a mutation.
    assert S.DEFAULT_VARIANT is before


@pytest.mark.parametrize("name", _VARIANT_KEYS)
def test_every_registered_variant_reaches_the_wire_through_score(name):
    """Coverage of the registry, one profile per case.

    Only FOUR distinct system prompts exist across the five profiles —
    `VARIANTS["disconfirm"].system_prompt is VARIANTS["disconfirm_relnature"]
    .system_prompt`. So the assertion is identity to the REQUESTED profile;
    claiming four distinct prompts would go red on correct code. What separates
    that pair is the call topology, gated by the next test.
    """
    from indra_belief.scorers.monolithic import scorer as S

    assert sorted(S.VARIANTS) == sorted(_VARIANT_KEYS)
    record = _records()[0]
    with _fake_gilda():
        kinds, main = _wire(lambda client, **kw: S.score(client, record, 64, **kw),
                            variant=S.VARIANTS[name])

    assert kinds == ["monolithic"]
    assert main["kwargs"]["kind"] == "monolithic"
    assert main["system"] is S.VARIANTS[name].system_prompt


def test_the_variant_selects_the_call_topology_not_only_the_prompt():
    """The second behavioural axis: whether the relation-nature sub-call fires.

    The `[Complex]` claim is required — measured, no profile fires the sub-call
    on the Phosphorylation record, so this test would be vacuous on it. The
    assertion is on the recorded `kind` sequence and not on the note text:
    `_RELATION_REPLY` names nature "cascade", which production logs as
    `unrecognized nature 'cascade'; treating as non-binding`. That log is
    expected and harmless — the sub-call still fires, and it is the firing that
    is being gated.
    """
    from indra_belief.scorers.monolithic import scorer as S

    record = _StubRecord(
        "Complex", "CCC3", "DDD4",
        "CCC3 and DDD4 act downstream of the same receptor.",
        "CCC3 binds DDD4 [Complex]",
    )
    with _fake_gilda():
        seen = {
            name: _wire(lambda client, **kw: S.score(client, record, 64, **kw),
                        variant=S.VARIANTS[name])[0]
            for name in _VARIANT_KEYS
        }

    for name in _RELNATURE_ENVS:
        assert seen[name] == [_RELATION_KIND, "monolithic"], seen
    assert seen[""] == ["monolithic"], seen
    assert seen["disconfirm"] == ["monolithic"], seen


def test_the_delegate_entry_points_pass_the_variant_through(monkeypatch):
    """`score_statement` and `score_evidence` — the seams the API layer holds.

    Both build a real `ScoringRecord` from an INDRA Statement, which grounds
    live and is orthogonal to what is gated here, so the constructor is
    replaced by the stub. `score_statement` resolves `ScoringRecord` out of
    module globals at call time, and one patch reaches both entry points
    because `score_evidence` delegates to `_score_evidence_monolithic`, which
    IS `scorer.score_statement`.
    """
    from indra_belief.scorers.monolithic import scorer as S
    from indra_belief.scorers import monolithic as pkg

    record = _records()[0]
    monkeypatch.setattr(S, "ScoringRecord", lambda **kw: record)

    with _fake_gilda():
        for entry in (S.score_statement, pkg.score_evidence):
            for name in ("", "disconfirm"):
                _, main = _wire(
                    lambda client, **kw: entry(None, None, client, **kw),
                    variant=S.VARIANTS[name],
                )
                assert main["system"] is S.VARIANTS[name].system_prompt, (
                    f"{entry.__name__} dropped variant {name!r}"
                )
