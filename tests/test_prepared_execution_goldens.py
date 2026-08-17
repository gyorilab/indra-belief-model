"""Frozen contract for the two request assemblers and the deterministic routes.

Two code paths build the same LLM request for one (Statement, Evidence) pair:

  live   ``indra_belief.scorers.monolithic.scorer`` — ``score``, ``_prepare``,
         ``_score_single`` / ``_score_with_tools``
  batch  ``indra_belief.comparison.replay`` — ``ReplayIndex.prepare``,
         ``ReplayIndex.deterministic_result``

K1-prepared-execution collapsed the two request assemblers into one
``indra_belief.prepared_execution.PreparedExecution``; both entry points above
now produce or consume that single value. This module is the only proof that the
collapse changed nothing, so it freezes, per ``MONO_VARIANT`` profile and per
route: the system digest, the message roles, the canonical
``replay.prompt_sha256`` request digest, and the full final user message — plus
the deterministic-route result dict from BOTH sides.

OBSERVED, NOT MIRRORED. The live half is captured by driving the real entry
point — ``scorer.score(client, record, max_tokens)`` — with a RECORDING FAKE
CLIENT that appends every ``client.call(...)`` to a log and returns a canned
reply. Nothing here re-derives the assembly rule, so a change to what
``_score_single`` / ``_score_with_tools`` actually pass (the system halves, the
lookup block, ``temperature``, ``kind``, ``max_tokens``) moves a frozen digest.
The route is likewise an OUTCOME, not a parameter: the returned ``tier`` and the
recorded ``kind`` are frozen and cross-checked against the substrate row's
``route``, so ``scorer.py:539-556`` is covered rather than bypassed. The
relation-nature note is an output too — the two relnature profiles reach
``_relation_note`` -> ``resolve_relation_nature`` for real, and the note text the
goldens carry is the one production emitted from a canned non-binding reply.

Everything here is a measurement of the CURRENT state. Two live/batch
divergences and one silent profile fallthrough are recorded as fact, not fixed;
resolving them belongs to the refactor nodes, and changing ``src/`` here would
destroy the baseline they are measured against.

Hermeticity: no network, no real LLM. The two Gilda seams
(``entity._cached_ground`` / ``._cached_get_names``) and the
``tools.gilda_tools.gilda`` module handle are replaced by a frozen lookup table
checked into the fixture — the module handle covers both ``lookup_gene_executor``
(the tool route's lookup block) and ``entity_grounding`` (the relation-nature
step's alias list). Note that ``ScoringRecord._abbreviation_alias_lines``
(scoring_record.py:281-305) wraps its whole body in ``except Exception: return
[]`` — a stub that *raises* on an unknown key is therefore SWALLOWED and
silently corrupts the golden. Hermeticity is proven positively instead: the stub
RECORDS unknown keys and the recorded list is asserted empty, and the rendered
abbreviation lines are asserted equal to the substrate's stored value.

EXCLUDED, deliberately: rows whose rendered claim carries a modification site
(``@S217``) or agent annotations (``SRC (bound to TLR4) [Activation] PLA2``) —
1225 of 33413 rows (3.67%). ``residue`` / ``position`` / ``Agent.mods`` /
``bound_conditions`` are not stored in the replay substrate, so rebuilding those
claims would mean parsing them back out of the rendered ``claim``, i.e. deriving
the golden's input from its own output. The six pinned rows all reconstruct from
structured fields alone.

Regeneration (never automatic — a diff here is a behaviour change):

    GOLDEN_REGEN=1 PYTHONPATH=src .venv/bin/python -m pytest -q \
        tests/test_prepared_execution_goldens.py
"""
from __future__ import annotations

import copy
import importlib
import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path

import pytest

from conftest import _MockResponse

from indra_belief.comparison.replay import ReplayError, ReplayIndex, prompt_sha256
from indra_belief.hashing import sha256_bytes, sha256_file
from indra_belief.prepared_execution import assert_replay_digests

import indra_belief.data.entity as _entity_mod
import indra_belief.tools.gilda_tools as _gilda_tools
from indra_belief.data.entity import GroundedEntity
from indra_belief.data.scoring_record import ScoringRecord

_ROOT = Path(__file__).resolve().parents[1]
_GOLDENS = Path(__file__).resolve().parent / "goldens" / "prepared_execution_goldens.json"

# The batch substrate the goldens pin. data/comparison ships the
# disconfirm_relnature_rf run: its plain main system ref is byte-identical to
# the live rf ACTIVE_SYSTEM_PROMPT, so live<->batch parity is assertable; it is
# the only run carrying relation_aliases.jsonl and relation-note rows (17,235 of
# 33,361). data/comparison_verdict_only pins two extra rows because its prompt
# components are emitted by scripts/build_verdict_only_replay.py:177 rather than
# by the live scorer — that script is a third assembler and must keep working.
_SUBSTRATE = _ROOT / "data" / "comparison" / "grounding_replay"
_VO_SUBSTRATE = _ROOT / "data" / "comparison_verdict_only" / "grounding_replay"
_WORKLOAD = "unique_exact_pairs_primary"

_TABLES = ("manifest.json", "executions.jsonl", "entities.jsonl", "lookups.jsonl")
_RF_TABLES = _TABLES + ("relation_aliases.jsonl",)

# (stmt_i, evidence_i) -> what the row exercises.
#   0:0    plain, no abbreviations, no lookups   WT1 + ZNF224 [Complex]
#   8:0    plain + one abbreviation line         TET2 [Inhibition] IL6
#   2:0    tool + two gilda lookup refs          VCL + SORBS1 [Complex]
#   2:41   no_text                               VCL + SORBS1 [Complex]
#   2:31   deterministic_mismatch                VCL + SORBS1 [Complex]
#   909:0  deterministic_pseudogene              HSP90B2P + PLG [Complex]
_PINNED = ("0:0", "8:0", "2:0", "2:41", "2:31", "909:0")
_VO_PINNED = ("0:0", "2:0")
# The substrate's route for each callable row. NOT an input to the capture: the
# live half runs score() and the returned tier / recorded call kind are compared
# back to this, so scorer.py:539-556 (needs_tool_use) is under test.
_LIVE_ROUTES = {"0:0": "plain", "8:0": "plain", "2:0": "tool"}
_ROUTE_TIER = {"plain": "llm_comprehension", "tool": "llm_tool_use"}
_ROUTE_KIND = {"plain": "monolithic", "tool": "monolithic_tool_context"}
_MAIN_KINDS = frozenset(_ROUTE_KIND.values())
_RELATION_KIND = "relation_nature"
_DETERMINISTIC = ("2:41", "2:31", "909:0")

# Threaded through score(client, record, max_tokens) so the frozen kwargs prove
# it reaches the main call and does NOT reach the relation call (which keeps
# resolve_relation_nature's own max_tokens=3000 default).
_MAX_TOKENS = 4096

# Only the row fields main_request / deterministic_result / the record builder
# read. Model outputs are never among them.
_ROW_FIELDS = (
    "abbreviation_lines", "claim", "entity_context", "entity_refs", "evidence_i",
    "evidence_metadata", "lookup_refs", "main_message_prefix_ref",
    "main_prompt_base_sha256", "main_system_ref", "object_name", "provenance",
    "relation_note_insertion", "route", "statement_type", "stmt_i",
    "subject_name", "workload",
)

# MONO_VARIANT profiles. The last two are the point: scorer.py:62 matches only
# the three "disconfirm*" values and scorer.py:81 falls through to the baseline
# SYSTEM_PROMPT, so "verdict_only" — the profile the shipped
# data/comparison_verdict_only run is named for — and any typo BOTH silently
# select the baseline prompt. Frozen as fact; not fixed here.
_VARIANTS = (
    "", "disconfirm", "disconfirm_relnature", "disconfirm_relnature_rf",
    "verdict_only", "typo_xyz",
)
_BASELINE_EQUIVALENT = ("", "verdict_only", "typo_xyz")
# scorer.py:78-79 imports resolve_relation_nature for these two only; scorer.py:90
# returns "" for every other profile without touching the client.
_RELNATURE_VARIANTS = ("disconfirm_relnature", "disconfirm_relnature_rf")
# The [Complex] rows. 8:0 is Inhibition, so
# `_prompts_relation.resolve_relation_nature` returns
# before the client call even under a relnature profile.
_NOTE_ROWS = ("0:0", "2:0")

# Canned model replies. These are INPUTS to production, not stand-ins for its
# output: the relation-nature note the goldens freeze is whatever
# _prompts_relation.resolve_relation_nature builds from the reply below, and the
# main reply exists so the verdict parse and _stamp_committed_justification run
# for real on every profile (structured and baseline both read it).
_MAIN_REPLY = (
    '{"support": "the sentence states a direct bind between the two entities", '
    '"objection": "none", "verdict": "correct", "confidence": "high"}'
)
# nature=physical_binding -> resolve_relation_nature returns "" (no note), which
# is the state the substrate's main_prompt_base_sha256 was generated under.
_REL_BIND_REPLY = '{"nature": "physical_binding", "span": "binds to"}'
# nature=signaling_cascade -> a real note, formatted by
# `prepared_execution.relation_mismatch_note`.
_REL_CASCADE_REPLY = (
    '{"nature": "signaling_cascade", "span": "acts upstream in a shared pathway"}'
)
_ARBITRARY_NOTE = "an arbitrary string no stored digest constrains"

# Digest prefixes measured BEFORE the capture channel moved from a hand-written
# mirror of the assembly rule to observation of the real score() call. They are
# repeated here, outside the fixture, so a regeneration cannot quietly re-baseline
# them: the three rf no-note values are the substrate's own
# main_prompt_base_sha256, and the baseline value is the silent-fallthrough
# identity that C0 must deliberately break.
_MEASURED = {
    "rf_plain_0_0": "778060f43554",
    "rf_plain_8_0": "89192b2c5850",
    "rf_tool_2_0": "f794e9a60a28",
    "baseline_plain_0_0": "e4705f8991f4",
    "disconfirm_plain_0_0": "9b843fc3cdc1",
}

_README = (
    "G0-goldens — frozen live/batch request + deterministic-result contract. "
    "Captured 2026-08-01 with gilda 1.6.1, indra 1.24.0, python 3.13.7, against "
    "data/comparison/grounding_replay (mono_variant disconfirm_relnature_rf) and "
    "data/comparison_verdict_only/grounding_replay. "
    "These values are the contract C0/K1/K2 refactor against. A diff here is a "
    "behaviour change, not a test to update."
)
_REGENERATE = (
    "GOLDEN_REGEN=1 PYTHONPATH=src .venv/bin/python -m pytest -q "
    "tests/test_prepared_execution_goldens.py"
)

_HAS_SUBSTRATE = all((_SUBSTRATE / name).exists() for name in _RF_TABLES)
_HAS_VO_SUBSTRATE = all((_VO_SUBSTRATE / name).exists() for name in _TABLES)

requires_substrate = pytest.mark.skipif(
    not _HAS_SUBSTRATE,
    reason="data/comparison/grounding_replay absent (gitignored published artifact)",
)
requires_vo_substrate = pytest.mark.skipif(
    not _HAS_VO_SUBSTRATE,
    reason="data/comparison_verdict_only/grounding_replay absent (gitignored)",
)


def _substrate_digests() -> dict[str, str]:
    """sha256 of every substrate file this module opens.

    Content, not `git status`: .gitignore:102 ignores data/comparison/ and
    .gitignore:123 ignores data/comparison_verdict_only/*, so a git check could
    never observe a mutation to either directory.
    """
    out: dict[str, str] = {}
    for base, names in ((_SUBSTRATE, _RF_TABLES), (_VO_SUBSTRATE, _TABLES)):
        for name in names:
            path = base / name
            if path.exists():
                out[str(path.relative_to(_ROOT))] = sha256_file(path)
    return out


# Taken at import, before this module reads a single substrate byte.
_DIGESTS_AT_IMPORT = _substrate_digests()

_EXPECTED: dict = json.loads(_GOLDENS.read_text(encoding="utf-8")) if _GOLDENS.exists() else {}


# --------------------------------------------------------------------------
# Frozen Gilda seam
# --------------------------------------------------------------------------

class _FrozenTerm:
    __slots__ = ("db", "id", "entry_name")

    def __init__(self, db: str, ident: str, entry_name: str) -> None:
        self.db, self.id, self.entry_name = db, ident, entry_name


class _FrozenMatch:
    __slots__ = ("term", "score")

    def __init__(self, db: str, ident: str, entry_name: str, score: float) -> None:
        self.term, self.score = _FrozenTerm(db, ident, entry_name), score


class _Grounder:
    """Stands in for gilda. ``table=None`` records from live gilda (regen only).

    Unknown keys are RECORDED, not raised: scoring_record.py:281-305 swallows
    every exception from the abbreviation path, so a raising stub degrades the
    golden in silence instead of failing the test.
    """

    def __init__(self, table: dict | None) -> None:
        self.recording = table is None
        self._ground = {} if self.recording else dict(table["ground"])
        self._names = {} if self.recording else dict(table["get_names"])
        self.misses: list[str] = []

    def ground(self, name):
        key = str(name)
        if self.recording:
            import gilda
            hits = gilda.ground(key) or []
            self._ground.setdefault(
                key, [[m.term.db, str(m.term.id), m.term.entry_name, float(m.score)]
                      for m in hits]
            )
            return hits
        if key not in self._ground:
            self.misses.append(f"ground:{key}")
            return []
        return [_FrozenMatch(*row) for row in self._ground[key]]

    def get_names(self, db, db_id):
        key = f"{db}\t{db_id}"
        if self.recording:
            import gilda
            names = list(gilda.get_names(db, db_id) or [])
            self._names.setdefault(key, names)
            return names
        if key not in self._names:
            self.misses.append(f"get_names:{key}")
            return []
        return list(self._names[key])

    def as_table(self) -> dict:
        return {"ground": self._ground, "get_names": self._names}


@contextmanager
def _gilda_seams(grounder: _Grounder):
    """Swap the TWO Gilda entry points, clearing both lru_caches either side.

    It was three. `tools.gilda_tools` held its own module-level `import gilda`
    and called `gilda.ground` / `gilda.get_names` directly, so it needed its own
    patch. haohangyan's scale_up branch routed both call sites through
    `entity._cached_ground` / `entity._cached_get_names` instead — for
    performance, since `gilda.get_names` is an unmemoized scan of the whole index
    and `execute_lookup_gene` calls it up to four times per lookup — and the
    side effect is that this seam gets SMALLER: there is now one set of cached
    entry points, and patching it reaches both modules.

    Verified rather than assumed: no unrouted `gilda.ground` / `gilda.get_names`
    call remains in `tools/gilda_tools.py`; the two textual hits there are a
    docstring and a log message.
    """
    orig_ground = _entity_mod._cached_ground
    orig_names = _entity_mod._cached_get_names
    _entity_mod._cached_ground = grounder.ground
    _entity_mod._cached_get_names = grounder.get_names
    orig_ground.cache_clear()
    orig_names.cache_clear()
    try:
        yield grounder
    finally:
        _entity_mod._cached_ground = orig_ground
        _entity_mod._cached_get_names = orig_names
        orig_ground.cache_clear()
        orig_names.cache_clear()


# --------------------------------------------------------------------------
# Profile switching
# --------------------------------------------------------------------------

def _load_scorer(variant: str):
    """The profile is read at import (scorer.py:61), so re-import to change it."""
    os.environ["MONO_VARIANT"] = variant
    for name in [n for n in sys.modules if n.startswith("indra_belief.scorers.monolithic")]:
        del sys.modules[name]
    return importlib.import_module("indra_belief.scorers.monolithic.scorer")


@contextmanager
def _profile_switching():
    """Restore the original MONO_VARIANT and re-import once at teardown, so the
    module state every other test sees is the state it would have had."""
    original = os.environ.get("MONO_VARIANT")
    try:
        yield _load_scorer
    finally:
        if original is None:
            os.environ.pop("MONO_VARIANT", None)
        else:
            os.environ["MONO_VARIANT"] = original
        for name in [n for n in sys.modules if n.startswith("indra_belief.scorers.monolithic")]:
            del sys.modules[name]
        importlib.import_module("indra_belief.scorers.monolithic.scorer")


# --------------------------------------------------------------------------
# Substrate loading (read-only) and record reconstruction
# --------------------------------------------------------------------------

def _jsonl(path: Path, needles: tuple[bytes, ...] | None = None) -> list[dict]:
    rows: list[dict] = []
    with path.open("rb") as handle:
        for line in handle:
            if not line.strip():
                continue
            if needles is not None and not any(n in line for n in needles):
                continue
            rows.append(json.loads(line))
    return rows


def _pinned_rows(base: Path, keys) -> dict[str, dict]:
    wanted = {tuple(int(part) for part in key.split(":")) for key in keys}
    needles = tuple(b'"stmt_i":%d,' % stmt for stmt, _ in wanted)
    out: dict[str, dict] = {}
    for row in _jsonl(base / "executions.jsonl", needles):
        if row.get("workload") != _WORKLOAD:
            continue
        key = (int(row["stmt_i"]), int(row["evidence_i"]))
        if key in wanted:
            out[f"{key[0]}:{key[1]}"] = {f: row.get(f) for f in _ROW_FIELDS}
    missing = set(keys) - set(out)
    if missing:
        raise AssertionError(f"pinned rows absent from {base}: {sorted(missing)}")
    return out


def _pinned_entities(base: Path, rows: dict[str, dict]) -> dict[str, dict]:
    refs = {
        str(ref)
        for row in rows.values()
        for ref in (row.get("entity_refs") or {}).values()
        if ref
    }
    needles = tuple(f'"{ref}"'.encode() for ref in refs)
    out = {
        row["entity_key_sha256"]: row
        for row in _jsonl(base / "entities.jsonl", needles)
        if row["entity_key_sha256"] in refs
    }
    missing = refs - set(out)
    if missing:
        raise AssertionError(f"entity rows absent from {base}: {sorted(missing)}")
    return out


def _replay_index(base: Path, rows: dict[str, dict]) -> ReplayIndex:
    """Positional construction — ReplayIndex.load() needs a RunPlan we do not have.

    Mirrors replay.py:236-277: systems keyed by sha256, prefixes keyed by the
    canonical digest of their message list, entities by entity_key_sha256,
    lookups by lookup_key_sha256 -> row["formatted"].
    """
    manifest = json.loads((base / "manifest.json").read_text(encoding="utf-8"))
    components = manifest["prompt_components"]
    systems = {item["sha256"]: item["text"] for item in components.get("main_systems", [])}
    relation = components.get("relation_system")
    if relation is not None:
        systems[str(relation["sha256"])] = str(relation["text"])
    prefixes = {
        item["sha256"]: tuple(dict(message) for message in item["messages"])
        for item in components.get("main_message_prefixes", [])
    }
    entities = {row["entity_key_sha256"]: row for row in _jsonl(base / "entities.jsonl")}
    lookups = {
        row["lookup_key_sha256"]: str(row["formatted"])
        for row in _jsonl(base / "lookups.jsonl")
    }
    alias_path = base / "relation_aliases.jsonl"
    relation_aliases = (
        {row["relation_alias_key_sha256"]: row.get("grounding")
         for row in _jsonl(alias_path)}
        if alias_path.exists() else {}
    )
    return ReplayIndex(manifest, (), systems, prefixes, entities, lookups,
                       relation_aliases, tuple(rows.values()))


def _batch_request(index: ReplayIndex, row: dict,
                   relation_note: str = "") -> tuple[str, list[dict[str, str]]]:
    """What ``ReplayIndex.main_request`` returned, through its replacement.

    K1-prepared-execution split that method in two — ``ReplayIndex.prepare``
    resolves the row's component refs, ``assert_replay_digests`` holds the two
    checks it used to make inline — so the batch half is now driven through both.
    Dropping the assert here would silently retire the digest contract these
    goldens exist to hold, so it is called on every path, exactly as before.
    """
    execution = index.prepare(row)
    assert_replay_digests(execution, row, relation_note=relation_note)
    call = execution.calls(relation_note)[-1]
    return call.system, [dict(message) for message in call.messages]


def _build_statement(row: dict):
    import indra.statements as statements

    subject = statements.Agent(str(row["subject_name"]))
    object_ = statements.Agent(str(row["object_name"]))
    stmt_type = str(row["statement_type"])
    if stmt_type == "Complex":
        return statements.Complex([subject, object_])
    return getattr(statements, stmt_type)(subject, object_)


def _build_record(row: dict, entities: dict[str, dict]) -> ScoringRecord:
    """Rebuild the live ScoringRecord from stored structured fields.

    ``__post_init__`` calls ``resolve_entities`` (scoring_record.py:37-38), which
    would ground live; the pre-resolved GroundedEntity rows from the substrate are
    installed instead. Everything downstream — format_claim, format_entity_context,
    format_provenance, _abbreviation_alias_lines, should_auto_reject,
    execution_body — still runs for real.
    """
    from indra.statements import Evidence

    metadata = row["evidence_metadata"]
    evidence = Evidence(
        source_api=metadata.get("source_api"),
        pmid=metadata.get("pmid"),
        text=metadata.get("text"),
        annotations={
            "found_by": metadata.get("found_by"),
            "agents": {"raw_text": list(metadata.get("raw_text") or [])},
        },
        epistemics={"direct": metadata.get("is_direct")},
    )
    original = ScoringRecord.resolve_entities
    ScoringRecord.resolve_entities = lambda self: None
    try:
        record = ScoringRecord(statement=_build_statement(row), evidence=evidence)
    finally:
        ScoringRecord.resolve_entities = original
    refs = row.get("entity_refs") or {}
    for side, attribute in (("subject", "subject_entity"), ("object", "object_entity")):
        ref = refs.get(side)
        stored = entities.get(str(ref)) if ref else None
        setattr(record, attribute, None if stored is None else GroundedEntity(
            **{k: v for k, v in stored.items() if k != "entity_key_sha256"}
        ))
    return record


class _RaisingClient:
    """Any LLM call on a deterministic route is a test failure, not a mock."""

    def call(self, *args, **kwargs):
        raise AssertionError("an LLM call was reached on a deterministic route")

    def pop_call_log(self):
        return []


class _RecordingClient:
    """Captures what production passed to ``client.call`` and answers canned.

    This is the whole point of the module: the goldens are taken off THESE
    recorded arguments, produced by ``score()`` -> ``_score_single`` /
    ``_score_with_tools`` / ``resolve_relation_nature``, so nothing in the test
    re-derives the assembly rule. ``pop_call_log`` returns ``[]`` — the real
    ModelClient's log is a network artefact, and returning it would put transport
    detail into a request golden (score() stamps it as ``call_log``).
    """

    def __init__(self, relation_reply: str) -> None:
        self.calls: list[dict] = []
        self._relation_reply = relation_reply

    def call(self, system=None, messages=None, **kwargs):
        self.calls.append({"system": system, "messages": messages,
                           "kwargs": dict(kwargs)})
        content = (self._relation_reply if kwargs.get("kind") == _RELATION_KIND
                   else _MAIN_REPLY)
        response = _MockResponse(content=content, raw_text=content)
        # scorer._stamp_committed_justification (scorer.py:328-347) is a no-op
        # unless the response carries a dict trace; giving it one makes the
        # structured variants run that branch for real.
        response.reasoning_trace = {}
        return response

    def pop_call_log(self):
        return []


# --------------------------------------------------------------------------
# Capture
# --------------------------------------------------------------------------

def _describe(system: str, messages: list[dict]) -> dict:
    return {
        "system_sha256": sha256_bytes(system.encode("utf-8")),
        "system_len": len(system),
        "messages_len": len(messages),
        "message_roles": [message["role"] for message in messages],
        "prompt_sha256": prompt_sha256(system, messages),
        "final_user_message": messages[-1]["content"],
    }


def _live_request(mod, record, relation_reply: str) -> dict:
    """Run the REAL entry point and freeze what production put on the wire.

    ``mod.score`` is the only assembler invoked. The call log is partitioned on
    the ``kind`` production stamped: exactly one main call
    (``monolithic`` / ``monolithic_tool_context``) plus zero or more
    relation-nature calls. Which of the two main kinds appears — and the ``tier``
    the result reports — is an OBSERVATION; nothing here selects a route.
    """
    client = _RecordingClient(relation_reply)
    result = mod.score(client, record, _MAX_TOKENS)
    main = [call for call in client.calls if call["kwargs"].get("kind") in _MAIN_KINDS]
    relation = [call for call in client.calls
                if call["kwargs"].get("kind") == _RELATION_KIND]
    if len(main) != 1:
        raise AssertionError(f"expected exactly one main call, recorded {len(main)}")
    if len(main) + len(relation) != len(client.calls):
        raise AssertionError(
            "unrecognised call kinds recorded: "
            f"{sorted({str(c['kwargs'].get('kind')) for c in client.calls})}"
        )
    entry = _describe(main[0]["system"], main[0]["messages"])
    entry["main_call_kwargs"] = main[0]["kwargs"]
    entry["tier"] = result["tier"]
    entry["call_log"] = result["call_log"]
    entry["relation_calls"] = [
        dict(_describe(call["system"], call["messages"]), kwargs=call["kwargs"])
        for call in relation
    ]
    return entry


def _capture(inputs: dict, table: dict | None, index: ReplayIndex | None,
             vo_index: ReplayIndex | None) -> tuple[dict, _Grounder]:
    rows = inputs["rows"]
    entities = inputs["entities"]
    grounder = _Grounder(table)
    data: dict = {"profiles": {}, "live_requests": {}, "relation_notes": {}}

    with _gilda_seams(grounder), _profile_switching() as load_scorer:
        for variant in _VARIANTS:
            mod = load_scorer(variant)
            # C0-profile-identity replaced the module globals (_VARIANT,
            # _STRUCTURED_VARIANTS, _variant_render, ACTIVE_SYSTEM_PROMPT) with
            # one frozen DEFAULT_VARIANT value. `resolved_variant` was the raw
            # env string the module read, which the module no longer retains;
            # _load_scorer set it immediately before the re-import, so it is
            # `variant`. Every other field is still read off the module.
            active = mod.DEFAULT_VARIANT
            data["profiles"][variant] = {
                "resolved_variant": variant,
                "structured": active.structured,
                "renderer": active.render_example.__name__,
                "renderer_module": active.render_example.__module__.rsplit(".", 1)[-1],
                "system_sha256": sha256_bytes(active.system_prompt.encode("utf-8")),
                "system_len": len(active.system_prompt),
                "lookup_guidance_sha256": sha256_bytes(mod._LOOKUP_GUIDANCE.encode("utf-8")),
                "lookup_guidance_len": len(mod._LOOKUP_GUIDANCE),
            }
            cases: dict[str, dict] = {}
            # A physical_binding reply makes resolve_relation_nature return ""
            # (`prepared_execution.relation_mismatch_note`) — the no-note state
            # the substrate's main_prompt_base_sha256 was generated under.
            for key, route in _LIVE_ROUTES.items():
                record = _build_record(rows[key], entities)
                cases[f"{route}@{key}"] = _live_request(mod, record, _REL_BIND_REPLY)
            if mod.DEFAULT_VARIANT.resolve_relation_nature is not None:
                # A non-binding reply makes production build a real note and
                # scorer.py:287-288 append "\n\n" + note to the last user message.
                for key in _NOTE_ROWS:
                    record = _build_record(rows[key], entities)
                    cases[f"{_LIVE_ROUTES[key]}+note@{key}"] = _live_request(
                        mod, record, _REL_CASCADE_REPLY)
            data["live_requests"][variant] = cases

        # Deterministic routes under the shipped default profile. These three
        # return before any request is assembled (scorer.py:504, :520;
        # replay.py:567), so the result dict is the whole deliverable.
        mod = load_scorer("disconfirm_relnature_rf")

        # The note text is production's, not a literal: scorer._relation_note
        # (:87) -> resolve_relation_nature, from the same canned reply the
        # +note cases above ran under. One source, reused by the batch half.
        for key in _NOTE_ROWS:
            record = _build_record(rows[key], entities)
            data["relation_notes"][key] = mod._relation_note(
                _RecordingClient(_REL_CASCADE_REPLY), record)

        data["deterministic_results"] = {}
        for key in _DETERMINISTIC:
            record = _build_record(rows[key], entities)
            live = mod.score(_RaisingClient(), record)
            entry = {"route": rows[key]["route"], "live": live}
            if index is not None:
                batch = index.deterministic_result(rows[key])
                entry["batch"] = batch
                entry["live_only_keys"] = sorted(set(live) - set(batch))
                entry["batch_only_keys"] = sorted(set(batch) - set(live))
                entry["value_differences"] = {
                    field: {"live": live[field], "batch": batch[field]}
                    for field in sorted(set(live) & set(batch))
                    if live[field] != batch[field]
                }
            data["deterministic_results"][key] = entry

        # Rendered-vs-stored equality for the substrate-derived record fields.
        data["record_fields"] = {}
        for key in _PINNED:
            record = _build_record(rows[key], entities)
            data["record_fields"][key] = {
                "claim": record.format_claim(),
                "entity_context": record.format_entity_context(),
                "abbreviation_lines": record._abbreviation_alias_lines(),
                "provenance": record.format_provenance(),
                "user_message": record.execution_body().render(),
            }

        if index is not None:
            # assert_replay_digests checks the hydrated request against
            # main_prompt_base_sha256 on the no-note path and raises ReplayError
            # on drift; reaching the next line at all is that assertion.
            requests: dict[str, dict] = {}
            for key in _LIVE_ROUTES:
                system, messages = _batch_request(index, rows[key])
                requests[key] = _describe(system, messages)
                requests[key]["stored_main_prompt_base_sha256"] = \
                    rows[key]["main_prompt_base_sha256"]
            for key in _NOTE_ROWS:
                system, messages = _batch_request(
                    index, rows[key], relation_note=data["relation_notes"][key])
                noted = _describe(system, messages)
                noted["relation_note_insertion"] = rows[key]["relation_note_insertion"]
                requests[f"{key}+note"] = noted
            data["batch_requests"] = requests

        if vo_index is not None:
            vo_rows = inputs["verdict_only_rows"]
            vo: dict[str, dict] = {}
            for key in _VO_PINNED:
                system, messages = _batch_request(vo_index, vo_rows[key])
                vo[key] = _describe(system, messages)
                vo[key]["stored_main_prompt_base_sha256"] = \
                    vo_rows[key]["main_prompt_base_sha256"]
            data["verdict_only_batch_requests"] = vo

    return data, grounder


def _read_inputs() -> dict:
    rows = _pinned_rows(_SUBSTRATE, _PINNED)
    inputs = {
        "substrate": str(_SUBSTRATE.relative_to(_ROOT)),
        "workload": _WORKLOAD,
        "rows": rows,
        "entities": _pinned_entities(_SUBSTRATE, rows),
    }
    if _HAS_VO_SUBSTRATE:
        inputs["verdict_only_substrate"] = str(_VO_SUBSTRATE.relative_to(_ROOT))
        inputs["verdict_only_rows"] = _pinned_rows(_VO_SUBSTRATE, _VO_PINNED)
    return inputs


class _Capture:
    def __init__(self, data: dict, misses: list[str], index, vo_index) -> None:
        self.data, self.misses, self.index, self.vo_index = data, misses, index, vo_index


@pytest.fixture(scope="module")
def capture():
    regen = os.environ.get("GOLDEN_REGEN") == "1"
    if regen and not _HAS_SUBSTRATE:
        raise AssertionError("GOLDEN_REGEN=1 requires data/comparison/grounding_replay")
    if not regen and not _EXPECTED:
        raise AssertionError(f"golden fixture missing: {_GOLDENS} (see _regenerate)")

    inputs = _read_inputs() if regen else copy.deepcopy(_EXPECTED["_inputs"])
    index = _replay_index(_SUBSTRATE, inputs["rows"]) if _HAS_SUBSTRATE else None
    vo_index = (
        _replay_index(_VO_SUBSTRATE, inputs["verdict_only_rows"])
        if _HAS_VO_SUBSTRATE and "verdict_only_rows" in inputs else None
    )

    table = None if regen else _EXPECTED["_gilda_table"]
    if regen:
        # Pass one records the table from live gilda; pass two proves the frozen
        # table alone is sufficient, so the shipped fixture is never
        # gilda-dependent.
        _, recorder = _capture(inputs, None, index, vo_index)
        table = recorder.as_table()

    data, grounder = _capture(inputs, table, index, vo_index)
    data["_readme"] = _README
    data["_regenerate"] = _REGENERATE
    data["_inputs"] = inputs
    data["_gilda_table"] = table

    if regen:
        _GOLDENS.parent.mkdir(parents=True, exist_ok=True)
        _GOLDENS.write_text(
            json.dumps(data, sort_keys=True, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    yield _Capture(data, grounder.misses, index, vo_index)

    assert _substrate_digests() == _DIGESTS_AT_IMPORT, (
        "a data/comparison* substrate file changed while this module ran"
    )


@pytest.fixture(scope="module")
def expected(capture) -> dict:
    return json.loads(_GOLDENS.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# Fixture shape
# --------------------------------------------------------------------------

def test_fixture_is_sorted_json_with_provenance(expected):
    raw = _GOLDENS.read_text(encoding="utf-8")
    assert raw == json.dumps(expected, sort_keys=True, indent=2, ensure_ascii=False) + "\n"
    assert expected["_readme"] == _README
    assert "gilda 1.6.1" in expected["_readme"]
    assert "indra 1.24.0" in expected["_readme"]
    assert "2026-08-01" in expected["_readme"]
    assert (
        "A diff here is a behaviour change, not a test to update."
        in expected["_readme"]
    )
    assert expected["_regenerate"] == _REGENERATE


# --------------------------------------------------------------------------
# Profile matrix
# --------------------------------------------------------------------------

def test_profile_matrix(capture, expected):
    assert capture.data["profiles"] == expected["profiles"]
    assert sorted(capture.data["profiles"]) == sorted(_VARIANTS)


def test_unrecognised_variant_selects_the_baseline_prompt(capture):
    """The VARIANTS registry holds only ("", "disconfirm", "disconfirm_relnature",
    "disconfirm_relnature_rf"); every other value falls back to the baseline. So
    "verdict_only" — the profile the shipped data/comparison_verdict_only run is
    named for — and a nonsense value like "typo_xyz" both produce the BASELINE
    prompt. C0-profile-identity made that fallback LOG (variant_from_env warns on
    an unrecognized non-empty value, and tests/test_monolithic_variant_profile.py
    pins the warning); the prompt identity below is deliberately unchanged, which
    is why none of these digests moved."""
    profiles = capture.data["profiles"]
    requests = capture.data["live_requests"]

    def prompt_identity(variant: str) -> dict:
        # resolved_variant is the raw env string and is the ONLY thing that
        # differs — nothing downstream of scorer.py:62 reads it.
        return {k: v for k, v in profiles[variant].items() if k != "resolved_variant"}

    baseline = profiles[""]
    for variant in _BASELINE_EQUIVALENT:
        assert prompt_identity(variant) == prompt_identity(""), variant
        assert profiles[variant]["resolved_variant"] == variant
        # Full recorded request, not just the system half: same digests, same
        # kwargs, same (empty) relation-call list.
        assert requests[variant] == requests[""], variant
        assert requests[variant]["plain@0:0"]["prompt_sha256"].startswith(
            _MEASURED["baseline_plain_0_0"]), variant
        assert requests[variant]["plain@0:0"]["relation_calls"] == []
    assert baseline["structured"] is False
    assert baseline["renderer_module"] == "_prompts"
    # ... and the three recognised values are genuinely different.
    assert profiles["disconfirm"]["system_sha256"] != baseline["system_sha256"]
    assert profiles["disconfirm_relnature_rf"]["system_sha256"] != \
        profiles["disconfirm"]["system_sha256"]


# --------------------------------------------------------------------------
# Live request goldens
# --------------------------------------------------------------------------

def test_live_request_goldens(capture, expected):
    """The gate. ``capture.data`` is what ``score()`` just put on the wire; a
    whole-dict compare reports it as a truncated diff, so the comparison is
    walked per (profile, case) and the moved FIELDS are named — a change to
    either call site (scorer.py:360 plain, :461 tool) says which digest moved."""
    got, want = capture.data["live_requests"], expected["live_requests"]
    assert sorted(got) == sorted(want)
    for variant in sorted(want):
        assert sorted(got[variant]) == sorted(want[variant]), variant
        for case in sorted(want[variant]):
            mine, frozen = got[variant][case], want[variant][case]
            moved = sorted(f for f in set(mine) | set(frozen)
                           if mine.get(f) != frozen.get(f))
            assert not moved, (
                f"live request moved for profile {variant!r} case {case!r}: "
                f"fields {moved} differ from the frozen golden"
            )
            assert mine == frozen, (variant, case)


def test_measured_digests_did_not_move(expected):
    """Swapping the capture channel (mirror -> observation of the real
    ``score()`` call) re-baselined NOTHING on the no-note path: these are the
    values measured before the swap, and the three rf ones are the substrate's
    own ``main_prompt_base_sha256``. Asserted without the substrate so a fresh
    checkout still proves it."""
    rf = expected["live_requests"]["disconfirm_relnature_rf"]
    assert rf["plain@0:0"]["prompt_sha256"].startswith(_MEASURED["rf_plain_0_0"])
    assert rf["plain@8:0"]["prompt_sha256"].startswith(_MEASURED["rf_plain_8_0"])
    assert rf["tool@2:0"]["prompt_sha256"].startswith(_MEASURED["rf_tool_2_0"])
    assert expected["live_requests"][""]["plain@0:0"]["prompt_sha256"].startswith(
        _MEASURED["baseline_plain_0_0"])
    assert expected["live_requests"]["disconfirm"]["plain@0:0"][
        "prompt_sha256"].startswith(_MEASURED["disconfirm_plain_0_0"])


def test_live_request_coverage(expected):
    notes = expected["relation_notes"]
    for variant in _VARIANTS:
        cases = expected["live_requests"][variant]
        for key, route in _LIVE_ROUTES.items():
            entry = cases[f"{route}@{key}"]
            assert len(entry["prompt_sha256"]) == 64
            assert entry["message_roles"][-1] == "user"
            assert entry["final_user_message"].startswith("CLAIM: ")
        has_note = "plain+note@0:0" in cases
        assert has_note == (variant in _RELNATURE_VARIANTS)
        if has_note:
            assert cases["plain+note@0:0"]["final_user_message"] == \
                cases["plain@0:0"]["final_user_message"] + "\n\n" + notes["0:0"]
            assert cases["plain+note@0:0"]["prompt_sha256"] != \
                cases["plain@0:0"]["prompt_sha256"]


def test_observed_route_agrees_with_the_substrate(expected):
    """The route is an outcome of scorer.py:539-556, not a test parameter: the
    ``tier`` score() returned and the ``kind`` production stamped on the call are
    both recorded, and both must agree with the row's stored route. A change to
    ``needs_tool_use`` moves one of them."""
    for variant in _VARIANTS:
        cases = expected["live_requests"][variant]
        for key, route in _LIVE_ROUTES.items():
            entry = cases[f"{route}@{key}"]
            assert route == expected["_inputs"]["rows"][key]["route"], key
            assert entry["tier"] == _ROUTE_TIER[route], (variant, key)
            assert entry["main_call_kwargs"]["kind"] == _ROUTE_KIND[route], (variant, key)
        for key in _NOTE_ROWS:
            noted = cases.get(f"{_LIVE_ROUTES[key]}+note@{key}")
            if noted is not None:
                assert noted["tier"] == _ROUTE_TIER[_LIVE_ROUTES[key]]
                assert noted["main_call_kwargs"]["kind"] == _ROUTE_KIND[_LIVE_ROUTES[key]]


def test_main_call_kwargs_are_frozen(expected):
    """The non-prompt half of the call. ``max_tokens`` is the value score() was
    given, proving it is threaded through (scorer.py:359-365, :460-466);
    ``temperature`` pins the 0.1 both call sites hard-code."""
    for variant in _VARIANTS:
        for name, entry in expected["live_requests"][variant].items():
            assert entry["main_call_kwargs"] == {
                "kind": entry["main_call_kwargs"]["kind"],
                "max_tokens": _MAX_TOKENS,
                "temperature": 0.1,
            }, (variant, name)
            # score() stamps the client's call log onto the result; the fake's is
            # empty, so a request golden carries no transport detail.
            assert entry["call_log"] == []


def test_relation_call_counts_and_kwargs(expected):
    """The relation-nature step is what distinguishes ``disconfirm`` from
    ``disconfirm_relnature``: on the no-note request they are byte-identical, so
    the call COUNT is the only discriminator. Row 8:0 is Inhibition and returns
    in ``_prompts_relation.resolve_relation_nature`` without a call, even under
    a relnature profile.
    The kwargs pin ``reasoning_effort='none'`` and the json_object response
    format in ``_prompts_relation.resolve_relation_nature``, which nothing else
    constrains."""
    for variant in _VARIANTS:
        relnature = variant in _RELNATURE_VARIANTS
        for name, entry in expected["live_requests"][variant].items():
            row = name.split("@", 1)[1]
            wanted = 1 if (relnature and row in _NOTE_ROWS) else 0
            assert len(entry["relation_calls"]) == wanted, (variant, name)
            for call in entry["relation_calls"]:
                assert call["kwargs"] == {
                    "kind": _RELATION_KIND,
                    "max_tokens": 3000,
                    "temperature": 0.1,
                    "response_format": {"type": "json_object"},
                    "reasoning_effort": "none",
                }, (variant, name)
                assert call["message_roles"] == ["user"]
                assert call["final_user_message"].startswith("Entities: ")
    # ... and the two profiles really are indistinguishable without it.
    plain_disconfirm = expected["live_requests"]["disconfirm"]["plain@0:0"]
    plain_relnature = expected["live_requests"]["disconfirm_relnature"]["plain@0:0"]
    assert plain_disconfirm["prompt_sha256"] == plain_relnature["prompt_sha256"]
    assert plain_disconfirm["relation_calls"] == []
    assert len(plain_relnature["relation_calls"]) == 1


def test_relation_note_is_production_output(expected):
    """The note is emitted by ``prepared_execution.relation_mismatch_note``
    from a canned non-binding reply, and is a SUFFIX of what production actually
    put in the plain+note user message — so the fixture cannot drift from the
    formatter. On the tool route it lands BEFORE the lookup block."""
    notes = expected["relation_notes"]
    assert sorted(notes) == sorted(_NOTE_ROWS)
    for key in _NOTE_ROWS:
        note = notes[key]
        assert note.startswith("Relation nature (resolved): the evidence asserts ")
        assert "a signaling/regulatory cascade (functional, not physical binding)" in note
        assert '"acts upstream in a shared pathway"' in note
        rows = expected["_inputs"]["rows"][key]
        assert f"between {rows['subject_name']} and {rows['object_name']}" in note
    for variant in _RELNATURE_VARIANTS:
        cases = expected["live_requests"][variant]
        assert cases["plain+note@0:0"]["final_user_message"].endswith("\n\n" + notes["0:0"])
        tool = cases["tool+note@2:0"]["final_user_message"]
        assert "\n\n" + notes["2:0"] + "\n\nEntity database lookups:" in tool


def test_tool_route_appends_the_lookup_block(expected):
    for variant in _VARIANTS:
        cases = expected["live_requests"][variant]
        plain, tool = cases["plain@0:0"], cases["tool@2:0"]
        # scorer.py:461 — the tool system is the plain system + _LOOKUP_GUIDANCE.
        assert tool["system_len"] == plain["system_len"] + \
            expected["profiles"][variant]["lookup_guidance_len"]
        assert tool["system_sha256"] != plain["system_sha256"]
        assert "Entity database lookups:" in tool["final_user_message"]


def test_record_fields_render_as_stored(capture, expected):
    assert capture.data["record_fields"] == expected["record_fields"]
    for key in _PINNED:
        rendered = expected["record_fields"][key]
        row = expected["_inputs"]["rows"][key]
        assert rendered["claim"] == row["claim"]
        assert rendered["entity_context"] == (row["entity_context"] or "")
        assert rendered["abbreviation_lines"] == (row["abbreviation_lines"] or [])
        assert rendered["provenance"] == (row["provenance"] or "")


# --------------------------------------------------------------------------
# Batch request goldens + live<->batch parity
# --------------------------------------------------------------------------

@requires_substrate
def test_batch_request_goldens(capture, expected):
    assert capture.data["batch_requests"] == expected["batch_requests"]


@requires_substrate
def test_live_batch_request_parity(capture, expected):
    """The rf profile's live prompt, the batch replay's prompt and the digest
    stored at generation time are one and the same for all three callable
    routes — including the abbreviation line and the live gilda lookup block."""
    live = expected["live_requests"]["disconfirm_relnature_rf"]
    batch = expected["batch_requests"]
    for key, route in _LIVE_ROUTES.items():
        stored = expected["_inputs"]["rows"][key]["main_prompt_base_sha256"]
        assert live[f"{route}@{key}"]["prompt_sha256"] == batch[key]["prompt_sha256"] == stored
        assert live[f"{route}@{key}"]["final_user_message"] == batch[key]["final_user_message"]
        assert live[f"{route}@{key}"]["system_sha256"] == batch[key]["system_sha256"]
    # The note path converges too — and nothing on the row asserts that, so it
    # is asserted here (see test_batch_note_path_is_digest_unchecked).
    for live_case, batch_case in (("plain+note@0:0", "0:0+note"),
                                  ("tool+note@2:0", "2:0+note")):
        assert live[live_case]["prompt_sha256"] == batch[batch_case]["prompt_sha256"]
        assert live[live_case]["final_user_message"] == batch[batch_case]["final_user_message"]
    # main_request self-checks the digest on the no-note path (replay.py:417)
    # and did not raise, which the capture above already exercised.
    assert capture.index is not None


@requires_substrate
def test_batch_note_path_is_digest_unchecked(capture, expected):
    """replay.py:410-418 takes the insertion-coordinate branch whenever a
    relation note is set and never compares a digest — no stored field
    constrains the 17,235 note-carrying rows. The goldens therefore record the
    note-case digest THEMSELVES, and assert the insertion coordinates, so the
    branch has a contract it did not have before."""
    index, rows = capture.index, expected["_inputs"]["rows"]
    notes = expected["relation_notes"]
    for key in ("0:0", "2:0"):
        entry = expected["batch_requests"][f"{key}+note"]
        assert entry["relation_note_insertion"] == rows[key]["relation_note_insertion"]
        insertion = entry["relation_note_insertion"]
        assert insertion["role"] == "user"
        assert insertion["prefix_if_nonempty"] == "\n\n"
        assert insertion["empty_note_inserts_prefix"] is False
        assert insertion["message_index"] == entry["messages_len"] - 1
        assert entry["prompt_sha256"] != rows[key]["main_prompt_base_sha256"]

        # An arbitrary note is accepted: nothing on the row constrains it.
        # (On the tool route the note lands BEFORE the lookup block —
        # PreparedExecution.calls appends the note first, then the lookups.)
        system, messages = _batch_request(index, rows[key], relation_note=_ARBITRARY_NOTE)
        assert "\n\n" + _ARBITRARY_NOTE in messages[-1]["content"]
        assert prompt_sha256(system, messages) != entry["prompt_sha256"]

        # The coordinates, by contrast, ARE checked — so a refactor cannot drop
        # them silently. Tampered with the SAME note production emitted, so the
        # only thing that moved is the coordinate.
        tampered = copy.deepcopy(rows[key])
        tampered["relation_note_insertion"]["utf8_byte_offset"] += 1
        with pytest.raises(ReplayError):
            _batch_request(index, tampered, relation_note=notes[key])


@requires_vo_substrate
def test_verdict_only_substrate_requests(capture, expected):
    """data/comparison_verdict_only's prompt components are emitted by
    scripts/build_verdict_only_replay.py:177, not by the live scorer: its plain
    main system is 5781a5842d (3671 chars) and no MONO_VARIANT value produces
    it, because "verdict_only" silently resolves to the baseline c6845ab46c
    (3411 chars). Pinned so K1 cannot break that third assembler."""
    assert capture.data["verdict_only_batch_requests"] == \
        expected["verdict_only_batch_requests"]
    vo = expected["verdict_only_batch_requests"]
    baseline = expected["profiles"]["verdict_only"]["system_sha256"]
    assert vo["0:0"]["system_sha256"] != baseline
    for key in _VO_PINNED:
        assert vo[key]["prompt_sha256"] == vo[key]["stored_main_prompt_base_sha256"]


# --------------------------------------------------------------------------
# Deterministic routes — result dicts from both sides
# --------------------------------------------------------------------------

def test_deterministic_result_goldens(capture, expected):
    assert sorted(expected["deterministic_results"]) == sorted(_DETERMINISTIC)
    for key, entry in capture.data["deterministic_results"].items():
        golden = expected["deterministic_results"][key]
        # Without the substrate only the live half is captured; it is asserted
        # everywhere, and the key sets must agree wherever the batch half exists.
        assert entry == {k: v for k, v in golden.items() if k in entry}, key
        if capture.index is not None:
            assert set(entry) == set(golden), key


def test_no_llm_call_on_a_deterministic_route():
    with pytest.raises(AssertionError, match="LLM call was reached"):
        _RaisingClient().call(system="s", messages=[])


@requires_substrate
def test_divergence_a_no_text_live_only_keys(expected):
    """Live scorer.py:514-515 emits selected_example_ids / selected_examples on
    the no_text route; batch replay.py:555-560 (_result) has no such keys. No
    shared value differs. K1-prepared-execution owns the reconciliation."""
    entry = expected["deterministic_results"]["2:41"]
    assert entry["route"] == "no_text"
    # `weight_of_evidence` joined this set when the scoring boundary began
    # persisting the probe's additive weight beside its probability. It is
    # live-only for the same reason `score_error` is: batch replay rebuilds a
    # result from a recorded row and never re-reads the model.
    assert entry["live_only_keys"] == [
        "probe_delta_logit", "score_error", "selected_example_ids",
        "selected_examples", "weight_of_evidence",
    ]
    assert entry["batch_only_keys"] == []
    assert entry["value_differences"] == {}
    assert entry["live"]["selected_example_ids"] == []
    assert entry["live"]["selected_examples"] == []
    # deterministic_mismatch has no divergence at all.
    mismatch = expected["deterministic_results"]["2:31"]
    assert mismatch["live_only_keys"] == [
        "probe_delta_logit", "score_error", "weight_of_evidence"]
    assert mismatch["batch_only_keys"] == []
    assert mismatch["value_differences"] == {}


@requires_substrate
def test_divergence_b_pseudogene_raw_text_prefix(expected):
    """Row 909:0 — the ONLY difference is the raw_text prefix, and the cause is
    branch-selection order, not a dropped clause.

    Live ``GroundedEntity.should_auto_reject`` takes the MISMATCH branch at
    entity.py:309-311 and returns before reaching the AMBIGUOUS+pseudogene
    branch at entity.py:327, so the "{name} is a pseudogene. " clause is emitted
    by NEITHER side. Batch replay.py:466-467 rebuilds the string from the row's
    route and therefore labels it "Pseudogene mapping". The tier agrees on both
    sides. K2-one-parser owns the reconciliation."""
    entry = expected["deterministic_results"]["909:0"]
    assert entry["route"] == "deterministic_pseudogene"
    assert entry["live_only_keys"] == [
        "probe_delta_logit", "score_error", "weight_of_evidence"]
    assert entry["batch_only_keys"] == []
    assert sorted(entry["value_differences"]) == ["raw_text"]
    live = entry["value_differences"]["raw_text"]["live"]
    batch = entry["value_differences"]["raw_text"]["batch"]
    assert live.startswith('Grounding mismatch: "HSP" independently grounds to ')
    assert batch.startswith('Pseudogene mapping: "HSP" independently grounds to ')
    assert live.split(": ", 1)[1] == batch.split(": ", 1)[1]
    assert "is a pseudogene." not in live
    assert "is a pseudogene." not in batch
    assert entry["live"]["tier"] == entry["batch"]["tier"] == "deterministic_pseudogene"


# --------------------------------------------------------------------------
# Hermeticity and substrate integrity
# --------------------------------------------------------------------------

def test_gilda_seam_saw_no_unknown_key(capture, expected):
    """Positive proof, not a raising stub: scoring_record.py:281-305 wraps the
    abbreviation path in `except Exception: return []`, so a stub that raises is
    swallowed and silently empties the golden. The stub records misses instead,
    and the rendered abbreviation lines are asserted against the stored value."""
    assert capture.misses == []
    assert expected["record_fields"]["8:0"]["abbreviation_lines"] == \
        expected["_inputs"]["rows"]["8:0"]["abbreviation_lines"]
    assert len(expected["record_fields"]["8:0"]["abbreviation_lines"]) == 1
    assert sorted(expected["_gilda_table"]) == ["get_names", "ground"]
    assert expected["_gilda_table"]["ground"]


def test_gilda_seams_are_restored(capture):
    """Both seams are `lru_cache` wrappers again after the capture.

    The third assertion — `_gilda_tools.gilda.__name__ == "gilda"` — is gone
    with the module-level import it checked; `tools.gilda_tools` now reaches
    Gilda through the two cached entry points below, so restoring them restores
    it. `test_gilda_tools_has_no_unrouted_gilda_call` is what keeps that true.
    """
    assert hasattr(_entity_mod._cached_ground, "cache_clear")
    assert hasattr(_entity_mod._cached_get_names, "cache_clear")


def test_gilda_tools_has_no_unrouted_gilda_call():
    """The seam is only two-part while every call goes through the cache.

    If `tools/gilda_tools.py` regains a direct `gilda.ground` / `gilda.get_names`
    call, the goldens would silently ground against the REAL index instead of the
    fixture grounder — passing or failing for reasons that have nothing to do
    with the code under test. That is the failure this pins, and it is why the
    seam could be simplified rather than merely patched around.
    """
    import ast
    import inspect

    source = inspect.getsource(_gilda_tools)
    unrouted = [
        node.func.attr
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "gilda"
    ]
    assert unrouted == [], (
        f"tools/gilda_tools.py calls gilda.{unrouted} directly again; either route "
        "it through entity._cached_* or restore the third patch point in _gilda_seams"
    )


def test_mono_variant_is_restored(capture, expected):
    """The capture re-imports the scorer six times; the suite must not notice."""
    import indra_belief.scorers.monolithic.scorer as scorer

    variant = os.environ.get("MONO_VARIANT", "disconfirm_relnature_rf").strip().lower()
    assert scorer.DEFAULT_VARIANT is scorer.variant_from_env()
    profile = expected["profiles"].get(variant, expected["profiles"][""])
    assert sha256_bytes(scorer.DEFAULT_VARIANT.system_prompt.encode("utf-8")) == \
        profile["system_sha256"]


@requires_substrate
def test_inputs_match_the_substrate(capture, expected):
    """The fixture carries its own record inputs so the live goldens assert
    everywhere; this proves those inputs are the substrate's, byte for byte."""
    assert expected["_inputs"]["rows"] == _pinned_rows(_SUBSTRATE, _PINNED)
    assert expected["_inputs"]["entities"] == \
        _pinned_entities(_SUBSTRATE, expected["_inputs"]["rows"])
    if _HAS_VO_SUBSTRATE:
        assert expected["_inputs"]["verdict_only_rows"] == \
            _pinned_rows(_VO_SUBSTRATE, _VO_PINNED)


@requires_substrate
def test_substrate_files_are_byte_identical(capture):
    """Content, not `git status`: .gitignore:102 and :123 ignore both substrate
    directories, so a git-status guard could never observe a mutation."""
    assert _substrate_digests() == _DIGESTS_AT_IMPORT
    assert set(_DIGESTS_AT_IMPORT) >= {
        f"data/comparison/grounding_replay/{name}" for name in _RF_TABLES
    }
