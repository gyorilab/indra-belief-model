"""One prepared execution — the whole LLM request for one (Statement, Evidence) pair.

One INDRA (Statement, Evidence) pair becomes one LLM request. That request used
to be assembled in THREE places — the live scorer, the batch replay, and the
verdict-only substrate generator — which agreed byte-for-byte only by
maintenance. This module is the single owner: two PRODUCERS build one VALUE, and
every consumer reads it.

    prepare_from_record(record, profile, ...)   live  — a resolved ScoringRecord
    prepare_from_replay_row(row, ...)           batch — a frozen substrate row
      -> PreparedExecution -> .calls(relation_note) -> (PreparedCall, ...)

Two producers is correct: their inputs genuinely differ (a live record resolves
entities through Gilda and renders few-shot examples at call time; a replay row
carries pre-rendered parts and resolves its prompt components by sha256 ref).
Two ASSEMBLERS was the defect.

THE ONE STRUCTURAL CONSTRAINT: the relation-nature note is only known AFTER the
relation sub-call returns, so a PreparedExecution cannot be a frozen message
list. `calls()` is the ONLY splice — it appends the note and then the entity
lookup block, in that order, and no caller may re-splice.

Provenance. `PreparedCall.prompt_sha256()` recomputes the canonical request
digest from the call itself, and `PreparedExecution.profile_id` names the prompt
profile the request was built under, so a score stays attributable to an exact
prompt. `assert_replay_digests` is the batch side of that contract: it holds the
two checks the frozen substrate commits to, defined once.

Home. Top level of the package, next to `hashing.py` / `metrics.py` /
`curation.py` / `sampling.py`, because both `scorers.monolithic` and
`comparison.replay` consume it and neither may own it.

Relation text. `relation_user_message` and `relation_mismatch_note` are the only
copies of the relation sub-call question and mismatch-note rendering. A byte
change in either function moves the prompt on the live and batch paths at once.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Protocol, Sequence

from indra_belief.comparison.contracts import ContractError
from indra_belief.hashing import canonical_sha256

# The main call's `kind` is a function of the route and nothing else. A route
# absent from this mapping is deterministic — it reaches no model at all.
MAIN_CALL_KINDS: Mapping[str, str] = {
    "plain": "monolithic",
    "tool": "monolithic_tool_context",
}
RELATION_CALL_KIND = "relation_nature"

_RELATION_NATURE_LABELS: Mapping[str, str] = MappingProxyType({
    "fusionconstruct": "a gene FUSION / chimeric construct (one molecule)",
    "signalingcascade": "a signaling/regulatory cascade (functional, not physical binding)",
    "cobindingthird": "co-binding to a shared THIRD entity (not each other)",
    "topicoraim": "only a title/topic phrase or an aim/methods clause (not an asserted result)",
    "other": "not a direct physical interaction",
})

# Which reader turns the model's reply back into (verdict, confidence). It
# travels with the request that produced the reply, and it is a CONSTANT: this
# was the seam K2-one-parser hung its parser from, and K2's answer is that there
# is exactly one — `indra_belief.verdict` reads every reply, on the live path and
# on the batch replay, under every profile. It was two values only because a
# profile used to select its own parser.
PARSER_ID = "indra_belief.verdict"

_LOOKUP_BLOCK_HEADER = "Entity database lookups:"


def relation_user_message(
    subject: str,
    object_: str,
    text: str,
    subject_grounding: Mapping[str, Any] | None = None,
    object_grounding: Mapping[str, Any] | None = None,
) -> str:
    """Render the one relation sub-call user message used by both paths."""
    def _entity(name: str, grounding: Mapping[str, Any] | None) -> str:
        if not grounding:
            return name
        aliases = [
            alias for alias in grounding["aliases"] if alias.lower() != name.lower()
        ][:6]
        return (
            f"{name} (also known as: {', '.join(aliases)})"
            if aliases else name
        )

    return (
        f'Entities: {_entity(subject, subject_grounding)}, '
        f'{_entity(object_, object_grounding)}\nSentence: "{text}"\n'
        f"What relationship does the sentence assert between {subject} and {object_}?"
    )


def relation_mismatch_note(
    nature: str | None,
    span: Any,
    subject: str,
    object_: str,
) -> str:
    """Render a relation mismatch from an already-normalized nature key."""
    if nature in (None, "", "physicalbinding"):
        return ""
    clipped_span = str(span)[:160]
    label = _RELATION_NATURE_LABELS.get(nature, "not direct physical binding")
    return (
        "Relation nature (resolved): the evidence asserts %s%s. A [Complex] claim requires a "
        "stated DIRECT PHYSICAL BIND between %s and %s — that is a grounding MISMATCH here, so "
        "the [Complex] extraction is unsupported." % (
            label,
            (' — "%s"' % clipped_span if clipped_span else ""),
            subject,
            object_,
        )
    )


class ReplayError(ContractError):
    """A frozen replay substrate no longer reproduces its committed request.

    Defined here rather than in `comparison.replay` because `assert_replay_digests`
    — the only place the digest contract is enforced — lives here now.
    `comparison.replay` re-exports the name, so every existing
    `except ReplayError` / `pytest.raises(ReplayError)` still binds this class.
    """


class ScoringProfile(Protocol):
    """The half of `scorers.monolithic.ScoringVariant` a request needs.

    Structural, not imported: `scorers.monolithic.scorer` imports THIS module, so
    the dependency may not run the other way.
    """

    name: str
    system_prompt: str

    def render_example(self, example: Mapping[str, Any]) -> tuple[str, str]: ...

    @property
    def structured(self) -> bool: ...


def prompt_sha256(system: str, messages: Sequence[Mapping[str, Any]]) -> str:
    """Canonical digest of one request. The value `main_prompt_base_sha256` and
    `relation_prompt_sha256` are committed against in the replay substrate."""
    return canonical_sha256({"system": system, "messages": list(messages)})


@dataclass(frozen=True)
class ExecutionBody:
    """The five parts of the user message, unjoined.

    The join below is the one that used to live twice — at
    `data/scoring_record.py` (live) and `comparison/replay.py` (batch). It is
    copied character-for-character from those two, which were already byte-equal;
    every frozen digest in `data/comparison/grounding_replay/manifest.json`
    depends on it.
    """

    claim: str
    entity_context: str = ""
    abbreviation_lines: tuple[str, ...] = ()
    provenance: str = ""
    evidence_text: str = ""

    def render(self) -> str:
        parts = [f"CLAIM: {self.claim}"]
        if self.entity_context:
            parts.append(self.entity_context)
        if self.abbreviation_lines:
            parts.append("In-text abbreviations:\n" + "\n".join(self.abbreviation_lines))
        if self.provenance:
            parts.append(self.provenance)
        parts.append(f'EVIDENCE: "{self.evidence_text}"')
        return "\n".join(parts)


@dataclass(frozen=True)
class PreparedCall:
    """One model-client call, fully determined.

    Everything `ModelClient.call` needs and nothing it does not: `client_kwargs()`
    omits the two optional transport fields when unset, so the plain main call
    passes exactly the five arguments it has always passed.
    """

    kind: str
    system: str
    messages: tuple[Mapping[str, str], ...]
    max_tokens: int | None = None
    temperature: float = 0.1
    response_format: Mapping[str, Any] | None = None
    reasoning_effort: str | None = None

    def prompt_sha256(self) -> str:
        """This call's canonical request digest, recomputed from the call."""
        return prompt_sha256(self.system, self.messages)

    def client_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "system": self.system,
            "messages": [dict(message) for message in self.messages],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        if self.response_format is not None:
            kwargs["response_format"] = dict(self.response_format)
        if self.reasoning_effort is not None:
            kwargs["reasoning_effort"] = self.reasoning_effort
        kwargs["kind"] = self.kind
        return kwargs


@dataclass(frozen=True)
class PreparedExecution:
    """Everything one (Statement, Evidence) pair puts on the wire.

    `route` is the scoring route the pair took: "plain" / "tool" reach the model,
    while "no_text" / "deterministic_mismatch" / "deterministic_pseudogene" are
    answered without one and therefore have an empty call topology.

    `relation` is the relation-nature sub-call when the substrate carries one. It
    is populated on the batch side only — the live scorer's relation step is owned
    end to end by `scorers.monolithic._prompts_relation.resolve_relation_nature`,
    which issues its own call.
    """

    route: str
    system: str
    body: ExecutionBody
    prefix: tuple[Mapping[str, str], ...] = ()
    lookups: tuple[str, ...] = ()
    max_tokens: int | None = None
    temperature: float = 0.1
    profile_id: str = ""
    parser_id: str = PARSER_ID
    relation: PreparedCall | None = None

    def calls(self, relation_note: str = "") -> tuple[PreparedCall, ...]:
        """The ordered call topology, with the ONE note-then-lookups splice.

        The relation sub-call comes first when there is one; the main call is
        always last. Both current sites spliced in exactly this order — the note
        joined with "\\n\\n", then the lookup block — and the substrate's
        `relation_note_insertion` coordinates commit to it.
        """
        if self.route not in MAIN_CALL_KINDS:
            return ()
        user = self.body.render()
        if relation_note:
            user += "\n\n" + relation_note
        if self.lookups:
            user += "\n\n" + _LOOKUP_BLOCK_HEADER + "\n" + "\n".join(self.lookups)
        main = PreparedCall(
            kind=MAIN_CALL_KINDS[self.route],
            system=self.system,
            messages=(*(dict(message) for message in self.prefix),
                      {"role": "user", "content": user}),
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        return (main,) if self.relation is None else (self.relation, main)

    @property
    def call_topology(self) -> tuple[str, ...]:
        return tuple(call.kind for call in self.calls())


def prepare_from_record(
    record: Any,
    profile: ScoringProfile,
    *,
    route: str,
    examples: Iterable[Mapping[str, Any]],
    lookups: Sequence[str] = (),
    lookup_guidance: str = "",
    max_tokens: int | None = None,
    temperature: float = 0.1,
) -> PreparedExecution:
    """Build the request for a live `ScoringRecord`.

    Route selection and few-shot SELECTION stay in `scorers.monolithic.scorer` —
    they read that module's type bank and its grounding flags — so the chosen
    `route` and `examples` are passed in. Rendering the chosen examples, composing
    the tool-route system prompt, and the body join are assembly, and live here.
    """
    prefix: list[Mapping[str, str]] = []
    for example in examples:
        user, assistant = profile.render_example(example)
        prefix.append({"role": "user", "content": user})
        prefix.append({"role": "assistant", "content": assistant})
    return PreparedExecution(
        route=route,
        # The tool route's system prompt is the profile's plus the lookup
        # guidance — the rule `_score_with_tools` used to apply at its call site.
        system=(profile.system_prompt + lookup_guidance) if route == "tool"
        else profile.system_prompt,
        body=record.execution_body(),
        prefix=tuple(prefix),
        lookups=tuple(lookups),
        max_tokens=max_tokens,
        temperature=temperature,
        profile_id=profile.name,
        parser_id=PARSER_ID,
    )


def prepare_from_replay_row(
    row: Mapping[str, Any],
    *,
    systems: Mapping[str, str],
    prefixes: Mapping[str, Sequence[Mapping[str, str]]],
    lookups: Mapping[str, str],
    max_tokens: int | None = None,
    temperature: float = 0.1,
    profile_id: str = "",
    relation: PreparedCall | None = None,
) -> PreparedExecution:
    """Build the request for a frozen replay row.

    `systems` / `prefixes` / `lookups` are the ReplayIndex's content-addressed
    component tables; resolving a ref against them is the index's remaining job,
    and a ref that misses raises the same ReplayError it always did.
    """
    route = str(row["route"])
    try:
        system = systems[str(row["main_system_ref"])]
        prefix = tuple(dict(message) for message in prefixes[str(row["main_message_prefix_ref"])])
        blocks = tuple(lookups[str(ref)] for ref in row.get("lookup_refs", []))
        body = ExecutionBody(
            claim=str(row["claim"]),
            entity_context=str(row["entity_context"]) if row.get("entity_context") else "",
            abbreviation_lines=tuple(row.get("abbreviation_lines") or ()),
            provenance=str(row["provenance"]) if row.get("provenance") else "",
            evidence_text=str(row["evidence_metadata"]["text"]),
        )
    except (KeyError, TypeError) as exc:
        raise ReplayError("main prompt references an absent component") from exc
    return PreparedExecution(
        route=route,
        system=system,
        body=body,
        prefix=prefix,
        lookups=blocks,
        max_tokens=max_tokens,
        temperature=temperature,
        profile_id=profile_id,
        parser_id=PARSER_ID,
        relation=relation,
    )


def assert_replay_digests(execution: PreparedExecution, row: Mapping[str, Any],
                          *, relation_note: str = "") -> None:
    """The two checks the frozen substrate commits its main prompt to.

    Without a note, the hydrated request must reproduce `main_prompt_base_sha256`
    byte for byte. With one, no stored digest can constrain the result — the note
    is a model output — so what is checked instead is WHERE the note went:
    `relation_note_insertion` pins the message index, the role, the UTF-8 byte
    offset it starts at, and the "\\n\\n" prefix rule.
    """
    if relation_note:
        insertion = row.get("relation_note_insertion")
        if not isinstance(insertion, Mapping) or (
            insertion.get("message_index") != len(execution.prefix)
            or insertion.get("role") != "user"
            or insertion.get("utf8_byte_offset") != len(execution.body.render().encode("utf-8"))
            or insertion.get("prefix_if_nonempty") != "\n\n"
            or insertion.get("empty_note_inserts_prefix") is not False
        ):
            raise ReplayError("relation-note insertion coordinates differ")
    elif execution.calls()[-1].prompt_sha256() != row.get("main_prompt_base_sha256"):
        raise ReplayError("hydrated main prompt digest differs")
