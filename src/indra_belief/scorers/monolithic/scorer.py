"""Monolithic single-call belief scorer — sibling to the decomposed
four-probe pipeline.

Two-tier architecture (single LLM call per (Statement, Evidence)):
  Tier 1: Deterministic grounding check (GroundedEntity.should_auto_reject)
    - MISMATCH → auto-reject
    - PSEUDOGENE + AMBIGUOUS → auto-reject
    - AMBIGUOUS → LLM judges with grounding context
  Tier 2: LLM text comprehension
    - base system prompt + adaptive contrastive examples (14 per record,
      retrieved by statement type via _TYPE_BANK + _TYPE_ADJACENCY)
    - Entity context injected from ScoringRecord
    - Output: a single JSON verdict, read by `indra_belief.verdict`

The decomposed sibling lives in indra_belief.scorers.probes.*.
Selection between architectures is via the CLI `--arch` flag in
indra_belief.scorers.scorer. The entry points here are `score(client, record)`
and `score_statement(stmt, ev, client)`; the canonical `score_evidence` name
the decomposed path also exposes is a delegate over `score_statement` in this
package's `__init__`.

The scoring profile — prompt, few-shot renderer, output contract, and whether
the relation-nature step fires — is a `ScoringVariant` value. The profile does
NOT select a parser: `indra_belief.verdict` reads every reply, on this path and
on the batch replay alike. It defaults to
`DEFAULT_VARIANT` (resolved from MONO_VARIANT once, at import) and can be
overridden per call with `variant=`.

Run:
    PYTHONPATH=src python -m indra_belief.scorers.scorer --arch monolithic ...
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

log = logging.getLogger(__name__)

from indra_belief.data.scoring_record import ScoringRecord
from indra_belief.model_client import ModelClient
from indra_belief.prepared_execution import PreparedExecution, prepare_from_record

# CorpusIndex is imported lazily in main() — the benchmark harness is the
# only consumer. Keeping it out of the module-level import chain means
# `from indra_belief import ModelResponse` (for typing) doesn't pay the
# cost of pulling in the INDRA corpus index.

from indra_belief.scorers.monolithic._prompts import (
    SYSTEM_PROMPT,
    CONTRASTIVE_EXAMPLES as _ALL_EXAMPLES,
    render_example as _render_example,
)
from indra_belief.scorers.monolithic._prompts_disconfirm import (
    DISCONFIRM_SYSTEM_PROMPT,
    REASONFIRST_SYSTEM_PROMPT,
    REASONFIRST_NOCONF_SYSTEM_PROMPT,
    render_example as _render_example_disconfirm,
    render_example_reasonfirst as _render_example_reasonfirst,
    render_example_reasonfirst_noconf as _render_example_reasonfirst_noconf,
)
from indra_belief.scorers.monolithic._prompts_verdict_only import (
    VERDICT_ONLY_SYSTEM_PROMPT,
    render_example as _render_example_verdict_only,
)
from indra_belief.scorers.monolithic._prompts_relation import resolve_relation_nature
from indra_belief.verdict import NO_TEXT_RESULT, parse_response

# --- Scoring variants ---
# A variant is the whole scoring profile — prompt, few-shot renderer, and the
# two optional halves — as ONE immutable value, so a caller can select it per
# call instead of per process. Every prompt module a variant needs is imported
# eagerly rather than behind the branch that used to pick one: measured, the
# two extra modules cost 3.8 ms and 12 sys.modules entries on the baseline
# path, and pull in no gilda and no INDRA.


@dataclass(frozen=True)
class ScoringVariant:
    """One scoring profile: the prompt, the few-shot renderer, and the two
    optional halves.

    `structured` says whether the profile's OUTPUT CONTRACT asks the model for a
    `support`/`objection` justification alongside the verdict — so it decides
    whether there is a committed justification to stamp, and nothing else. It
    used to be derived from a pair of per-variant parser callables; a profile no
    longer selects a parser, because `indra_belief.verdict` reads every reply.
    `resolve_relation_nature` is the focused [Complex] second step.
    """

    name: str
    system_prompt: str
    render_example: Callable[[dict], tuple[str, str]]
    structured: bool = False
    resolve_relation_nature: Callable[..., str] | None = None
    # Width of the top-logprob window to request on this variant's SCORING call.
    # Set only where the output contract puts the verdict label FIRST, so its
    # margin is readable from the response we were making anyway. A prompt that
    # deliberates in its answer fields lands the label ~56 tokens deep and reads
    # +22.50 — saturated, and a scan would still return a number that looks fine.
    in_call_label_logprobs: int | None = None
    # Transport-level reasoning suppression. "none" is translated by ModelClient
    # into chat_template_kwargs {"enable_thinking": False} on permissive
    # backends. REQUIRED alongside an in-call label read: registering a
    # verdict-first PROMPT is not enough, because with the thinking channel open
    # the model spends its budget reasoning and may emit no answer at all — the
    # first live check of this variant returned empty content and no margin.
    reasoning_effort: str | None = None
    # Sampling temperature for this variant's scoring call, overriding the 0.1
    # both call sites otherwise hard-code. REQUIRED with an in-call label read:
    # ModelClient refuses top_logprobs above temperature 0, because the reported
    # argmax token stream diverges from the sampled text there and the verdict
    # POSITION stops being trustworthy. So an in-call variant needs all three to
    # agree — verdict-first prompt, no reasoning channel, temperature 0 — and
    # missing any one of them fails loudly rather than returning a bad number.
    temperature: float | None = None


VARIANTS: dict[str, ScoringVariant] = {
    variant.name: variant
    for variant in (
        # baseline prompt, byte-for-byte — README.md documents MONO_VARIANT=""
        # as the switch onto it.
        ScoringVariant(
            name="",
            system_prompt=SYSTEM_PROMPT,
            render_example=_render_example,
        ),
        # commit-first prompt: support + objection, then the verdict.
        ScoringVariant(
            name="disconfirm",
            system_prompt=DISCONFIRM_SYSTEM_PROMPT,
            render_example=_render_example_disconfirm,
            structured=True,
        ),
        # base disconfirm + a focused relation-nature step that rejects
        # [Complex] claims whose evidence is not a direct physical bind.
        ScoringVariant(
            name="disconfirm_relnature",
            system_prompt=DISCONFIRM_SYSTEM_PROMPT,
            render_example=_render_example_disconfirm,
            structured=True,
            resolve_relation_nature=resolve_relation_nature,
        ),
        # reasoning-first disconfirm + the relation-nature step (the default).
        ScoringVariant(
            name="disconfirm_relnature_rf",
            system_prompt=REASONFIRST_SYSTEM_PROMPT,
            render_example=_render_example_reasonfirst,
            structured=True,
            resolve_relation_nature=resolve_relation_nature,
        ),
        # W2b: the default with verbalized confidence removed. Its prompt hashes
        # differently, so it carries its OWN calibration profile and cannot
        # borrow the default's — see REASONFIRST_NOCONF_SYSTEM_PROMPT. Not the
        # default until its profile is fitted and gated.
        # Verdict FIRST, no reasoning channel, no intermediate fields — so the
        # label's margin comes out of the scoring call itself. MEASURED n=80 on
        # MLX against the second-call probe: AUROC 0.8734 vs 0.7237, and
        # within-verdict 0.7814 vs 0.6856. The free read is also the better one.
        # It trades deliberation, previously measured at -0.0689 err-F1 on the
        # 26B, which is the open question this variant exists to settle.
        ScoringVariant(
            name="verdict_only",
            system_prompt=VERDICT_ONLY_SYSTEM_PROMPT,
            render_example=_render_example_verdict_only,
            structured=True,
            in_call_label_logprobs=128,
            reasoning_effort="none",
            temperature=0.0,
        ),
        ScoringVariant(
            name="disconfirm_relnature_rf_noconf",
            system_prompt=REASONFIRST_NOCONF_SYSTEM_PROMPT,
            render_example=_render_example_reasonfirst_noconf,
            structured=True,
            resolve_relation_nature=resolve_relation_nature,
        ),
    )
}

DEFAULT_VARIANT_NAME = "disconfirm_relnature_rf"  # the validated default


def variant_from_env(env: Mapping[str, str] | None = None) -> ScoringVariant:
    """Resolve the MONO_VARIANT environment variable to a variant.

    An unrecognized NON-EMPTY value falls back to the baseline and now says so.
    The fallback itself is unchanged; only the silence is. A plausible way to
    reach it: `data/comparison_verdict_only/grounding_replay/manifest.json`
    labels itself `mono_variant: "verdict_only"`, which is not a key here — a
    reader reproducing that run from its own label would get the baseline
    prompt with no signal. (That run itself was NOT scored that way; its
    prompts come from scripts/build_verdict_only_replay.py, not from this
    module.) `""` stays silent: README.md documents it as the intended
    baseline switch.
    """
    source = os.environ if env is None else env
    raw = source.get("MONO_VARIANT", DEFAULT_VARIANT_NAME).strip().lower()
    variant = VARIANTS.get(raw)
    if variant is None:
        if raw:
            log.warning(
                "MONO_VARIANT=%r is not a known variant (%s); falling back to the "
                "baseline prompt", raw, ", ".join(repr(k) for k in VARIANTS),
            )
        variant = VARIANTS[""]
    return variant


# Resolved ONCE, at import. A per-call environment read would let a mutated
# environ switch prompts mid-run, and a score has to stay attributable to an
# exact prompt+model. In-process selection is the `variant=` argument below.
DEFAULT_VARIANT = variant_from_env()


def _relation_note(client, record, *, variant: ScoringVariant | None = None) -> str:
    """Relation-nature rejection note for [Complex] claims (relnature variants);
    empty for other variants or when the evidence supports a direct physical bind."""
    variant = variant or DEFAULT_VARIANT
    if variant.resolve_relation_nature is None:
        return ""
    try:
        return variant.resolve_relation_nature(
            record.subject, record.object, record.stmt_type, record.evidence_text, client)
    except Exception as e:
        log.warning(
            "relation-nature step failed for %r->%r (%s): %s",
            getattr(record, "subject", None),
            getattr(record, "object", None),
            getattr(record, "stmt_type", None),
            e,
        )
        return ""

ROOT = Path(__file__).resolve().parents[4]

# Provenance is injected only when grounding is flagged; selective injection avoids context overhead.

# --- Adaptive few-shot selection ---
# Seven contrastive pairs (14 examples) per record — a balance between
# example coverage and leaving prompt budget for the model's own
# reasoning. More examples dilute attention; fewer lose type coverage.

# Type-specific example bank (loaded from JSON)
# Bank keys can be exact types ("Activation") or sub-keys ("Activation_no_relation")
_EXAMPLE_BANK_PATH = Path(__file__).resolve().parents[2] / "data" / "example_bank.json"
_RAW_BANK: dict[str, list[dict]] = {}
if _EXAMPLE_BANK_PATH.exists():
    with open(_EXAMPLE_BANK_PATH) as _f:
        _RAW_BANK = json.load(_f)
else:
    # A missing bank silently strips all few-shot examples — degrading scoring
    # quality without any error. Surface it loudly so packaging/path bugs are caught.
    log.warning(
        "few-shot example bank not found at %s — scorer will run with NO type-specific "
        "examples; check package data layout",
        _EXAMPLE_BANK_PATH,
    )

# Build type → list of pairs mapping from bank
# Keys like "Activation_no_relation" contribute to "Activation"
# Known INDRA statement types (any legal base_type must be one of these).
# Sub-keys use the pattern "{StmtType}_{errorPattern}" — e.g. "Activation_family".
_KNOWN_TYPES = {
    "Activation", "Inhibition", "Phosphorylation", "Dephosphorylation",
    "Autophosphorylation", "Acetylation", "Deacetylation", "Methylation",
    "Demethylation", "Ubiquitination", "Deubiquitination", "Translocation",
    "Complex", "IncreaseAmount", "DecreaseAmount", "Conversion",
    "GtpActivation", "Gef", "Gap",
}

_TYPE_BANK: dict[str, list[list[dict]]] = {}
for key, pair in _RAW_BANK.items():
    # Match the longest known type that the key starts with (handles
    # IncreaseAmount_foo → IncreaseAmount, Activation_family → Activation).
    base_type = next(
        (t for t in sorted(_KNOWN_TYPES, key=len, reverse=True)
         if key == t or key.startswith(t + "_")),
        key,  # fallback: unrecognized key routes to itself
    )
    _TYPE_BANK.setdefault(base_type, []).append(pair)

# Map base examples into pairs by their statement type
# Statement type is the bracketed suffix of a claim, e.g. "... [Activation]".
# Sentinel for a claim with no parseable [Type] bracket — keeps a malformed
# claim from raising IndexError; such pairs route to this bucket instead.
_UNKNOWN_STMT_TYPE = "Unknown"
_CLAIM_TYPE_RE = re.compile(r"\[([^\]]+)\]")


def _claim_stmt_type(claim: str) -> str:
    """Extract the bracketed statement type from a claim string.

    Returns the contents of the first ``[...]`` (stripped), matching the
    legacy ``claim.split('[')[1].split(']')[0].strip()`` on well-formed
    claims, but falls back to ``_UNKNOWN_STMT_TYPE`` instead of raising
    IndexError when no bracket is present.
    """
    m = _CLAIM_TYPE_RE.search(claim)
    return m.group(1).strip() if m else _UNKNOWN_STMT_TYPE


_BASE_PAIRS: dict[str, list[list[dict]]] = {}
for i in range(0, len(_ALL_EXAMPLES), 2):
    stype = _claim_stmt_type(_ALL_EXAMPLES[i]["claim"])
    _BASE_PAIRS.setdefault(stype, []).append([_ALL_EXAMPLES[i], _ALL_EXAMPLES[i + 1]])

# Universal pairs — patterns that apply to all statement types
_UNIVERSAL_PAIRS = [
    _ALL_EXAMPLES[4:6],    # Pair 3: logical inversion (AGER/MMP2, TP53/MDM2)
    _ALL_EXAMPLES[6:8],    # Pair 4: hedging scope (MYB/PPID)
]

# Which types are commonly confused with each other?
_TYPE_ADJACENCY = {
    "Phosphorylation": ["Dephosphorylation", "Autophosphorylation"],
    "Dephosphorylation": ["Phosphorylation", "Inhibition"],
    "Activation": ["IncreaseAmount", "Inhibition"],
    "Inhibition": ["DecreaseAmount", "Activation"],
    "IncreaseAmount": ["Activation", "DecreaseAmount"],
    "DecreaseAmount": ["IncreaseAmount", "Inhibition"],
    "Complex": ["Activation"],
    "Autophosphorylation": ["Phosphorylation"],
    "Translocation": [],
    "Ubiquitination": [],
    "Acetylation": ["Deacetylation"],
}

TARGET_PAIRS = 7  # balances type coverage against the model's reasoning budget


def _select_examples(stmt_type: str) -> list[dict]:
    """Select 7 contrastive pairs (14 examples) for a record's statement type.

    Priority:
    1. Own type pair(s) — from bank (may have multiple sub-keys) and/or base
    2. Adjacent type pairs — types commonly confused with this one
    3. Universal patterns — logical inversion, hedging scope
    4. Fill from remaining base pairs
    """
    selected: list[list[dict]] = []
    used_claims: set[str] = set()

    def _add_pair(pair: list[dict]) -> bool:
        key = pair[0]["claim"]
        if key in used_claims or len(selected) >= TARGET_PAIRS:
            return False
        selected.append(pair)
        used_claims.add(key)
        return True

    # 1. Own type from bank (may have multiple pairs from sub-keys)
    for pair in _TYPE_BANK.get(stmt_type, []):
        _add_pair(pair)

    # 1b. Own type from base
    for pair in _BASE_PAIRS.get(stmt_type, []):
        _add_pair(pair)

    # 2. Adjacent types
    for adj_type in _TYPE_ADJACENCY.get(stmt_type, []):
        for pair in _TYPE_BANK.get(adj_type, []):
            _add_pair(pair)
        for pair in _BASE_PAIRS.get(adj_type, []):
            _add_pair(pair)

    # 3. Universal patterns
    for pair in _UNIVERSAL_PAIRS:
        _add_pair(pair)

    # 4. Fill remaining from base pairs
    for i in range(0, len(_ALL_EXAMPLES), 2):
        _add_pair([_ALL_EXAMPLES[i], _ALL_EXAMPLES[i + 1]])

    # Flatten pairs into example list
    examples = []
    for pair in selected[:TARGET_PAIRS]:
        examples.extend(pair)
    return examples


def _example_id(ex: dict) -> str:
    """Stable identifier for a static contrastive example."""
    payload = json.dumps({
        "claim": ex.get("claim"),
        "evidence": ex.get("evidence"),
        "verdict": ex.get("verdict"),
        "confidence": ex.get("confidence"),
    }, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def _example_trace_rows(examples: list[dict]) -> list[dict]:
    """Compact selected-example provenance for persisted trace output."""
    return [
        {
            "id": _example_id(ex),
            "claim": ex.get("claim"),
            "verdict": ex.get("verdict"),
            "confidence": ex.get("confidence"),
        }
        for ex in examples
    ]


_LOOKUP_GUIDANCE = """

EXTERNAL LOOKUP CONTEXT — an "Entity database lookups:" block is included
below for claim entities that may be ambiguous. It shows the top Gilda
candidates for each entity — a proper gene, a chemical, a protein family,
a MeSH concept, or a pseudogene. Use this to ground the evidence correctly.

How to read it:
- If the top candidate is a gene family (FPLX) and the evidence mentions
  a specific member, that is still a valid claim per rule 2.
- If the top candidate is a chemical, lipid (CHEBI), or MeSH concept, the
  "gene" in the claim may be a conflation with a non-protein entity. If
  the evidence describes actions on that non-gene entity (e.g. a lipid
  receptor extracting as the lipid itself), treat it as a grounding error.
- If the top candidate is a pseudogene, the claim is likely wrong unless
  the evidence explicitly describes pseudogene transcripts.
- If "query is a known alias" is False, and the top candidate is a
  different gene from the claim, this is a grounding error.

Use the lookups to refine your verdict. Do NOT emit TOOL_CALL — the
lookups are already done for you.
"""


def _prepare(record: ScoringRecord, examples: list[dict] | None = None, *,
             route: str = "plain", lookups: Sequence[str] = (),
             max_tokens: int | None = None, temperature: float = 0.1,
             variant: ScoringVariant | None = None) -> PreparedExecution:
    """Bind this module's few-shot selection to the one request value.

    Everything the REQUEST is — the rendered few-shot prefix, the tool-route
    system prompt, the body join, and the note-then-lookups splice — belongs to
    `prepare_from_record`. What stays here is what genuinely reads this module:
    which contrastive examples this statement type gets, and the lookup-guidance
    block that composes the tool system prompt.
    """
    variant = variant or DEFAULT_VARIANT
    examples = examples if examples is not None else _select_examples(record.stmt_type)
    return prepare_from_record(
        record, variant, route=route, examples=examples, lookups=lookups,
        lookup_guidance=_LOOKUP_GUIDANCE, max_tokens=max_tokens,
        temperature=temperature,
    )


def _stamp_committed_justification(response, *,
                                   variant: ScoringVariant | None = None) -> None:
    """Record the model's committed `support`/`objection` into the response's
    reasoning trace, so a downstream interface can present the model's own
    justification uniformly across backends. Structured variants only (baseline
    does not ask for support/objection). The trace dict is the SAME object the
    model client appended to the call log, so this mutation travels with the
    persisted record. No-op (and crash-proof) when there's no structured trace.

    A reply the parser cannot read still stamps the key, with both fields None:
    the interface distinguishes "no justification committed" from "no trace
    recorded", and a justification without a verdict is not a commitment."""
    variant = variant or DEFAULT_VARIANT
    if not variant.structured:
        return
    trace = getattr(response, "reasoning_trace", None)
    if not isinstance(trace, dict):
        return
    parsed = parse_response(response)
    trace["committed_justification"] = {
        "support": None if parsed is None else parsed.support,
        "objection": None if parsed is None else parsed.objection,
        "source": "answer_json",
    }


def _call_kwargs(execution, note, variant) -> dict:
    """The scoring call's kwargs, asking for label logprobs where they are free.

    A variant whose output contract emits the verdict FIRST can have its label
    margin read from this response — no second probe request. Only such variants
    set ``in_call_label_logprobs``; for every other variant the kwargs are byte
    unchanged, so the plain path still passes exactly what it always passed.
    """
    kwargs = execution.calls(note)[-1].client_kwargs()
    resolved = variant or DEFAULT_VARIANT
    width = getattr(resolved, "in_call_label_logprobs", None)
    if width:
        kwargs["top_logprobs"] = width
    effort = getattr(resolved, "reasoning_effort", None)
    if effort:
        kwargs["reasoning_effort"] = effort
    temperature = getattr(resolved, "temperature", None)
    if temperature is not None:
        kwargs["temperature"] = temperature
    return kwargs


def _in_call_margin(response, variant) -> float | None:
    """The label's log-odds from THIS response, for variants that can supply it.

    Only variants whose output contract emits the verdict first set
    ``in_call_label_logprobs``; for everything else the scan would find the
    label ~56 tokens deep behind deliberative answer fields, where it reads
    +22.50 — saturated, and a number that looks fine. So this returns None
    unless the variant declared the read, rather than trying opportunistically.
    """
    resolved = variant or DEFAULT_VARIANT
    if not getattr(resolved, "in_call_label_logprobs", None):
        return None
    from indra_belief.probes.reader import label_margin_from_logprobs

    try:
        return label_margin_from_logprobs(getattr(response, "logprobs", None))
    except Exception:
        return None


def _score_single(
    client: ModelClient,
    record: ScoringRecord,
    max_tokens: int | None,
    temperature: float = 0.1,
    *,
    variant: ScoringVariant | None = None,
    margin_out: dict | None = None,
) -> dict:
    """Single LLM call for Tier 2 (+ optional relation-nature note). Returns result dict."""
    variant = variant or DEFAULT_VARIANT
    examples = _select_examples(record.stmt_type)
    note = _relation_note(client, record, variant=variant)
    execution = _prepare(record, examples, route="plain", max_tokens=max_tokens,
                         temperature=temperature, variant=variant)
    response = client.call(**_call_kwargs(execution, note, variant))
    parsed = parse_response(response)
    _stamp_committed_justification(response, variant=variant)
    selected_examples = _example_trace_rows(examples)
    # Side channel, NOT a new key in the returned dict. The variant-behaviour
    # golden captures these two functions' RETURN SHAPE and is a pre-refactor
    # capture with no regeneration path, so widening it would mean hand-editing
    # a fixture whose whole value is that it predates the change. The margin is
    # probe output; it belongs in the fields replace_sentence_score writes.
    if margin_out is not None:
        margin_out["in_call_label_margin"] = _in_call_margin(response, variant)
    return {
        "verdict": None if parsed is None else parsed.label,
        "confidence": None if parsed is None else parsed.confidence,
        "raw_text": response.raw_text,
        "tokens": response.tokens,
        "selected_example_ids": [ex["id"] for ex in selected_examples],
        "selected_examples": selected_examples,
    }


def _format_entity_lookups(record: ScoringRecord) -> list[str]:
    """Pre-compute gilda lookups for ambiguous entities. Returns one formatted
    line per looked-up entity, or [] when neither entity benefits from lookup.

    The "Entity database lookups:" header and the join belong to
    `PreparedExecution.calls` — the batch replay carries the same per-entity
    lines as separate content-addressed rows, so the block is assembled once,
    there, from the same shape on both sides.

    Looks up `raw_text` (the ambiguous mention the reader extracted), not
    `name` (the already-resolved canonical symbol). Looking up the canonical
    just confirms Gilda's existing decision; the raw_text is what actually
    needs disambiguation.
    """
    import logging
    from indra_belief.tools.gilda_tools import lookup_gene_executor

    log = logging.getLogger(__name__)
    lines: list[str] = []
    seen: set[str] = set()
    for entity in (record.subject_entity, record.object_entity):
        if not entity or not entity.name or entity.name == "?":
            continue
        # Prefer raw_text (the ambiguous mention that triggered the flag).
        # Fall back to name when raw_text is missing or identical.
        lookup_target = entity.raw_text or entity.name
        # Avoid duplicate lookups (autophosphorylation, same text twice)
        if lookup_target in seen:
            continue
        seen.add(lookup_target)
        try:
            result = lookup_gene_executor({"entity_name": lookup_target})
        except Exception as e:
            log.warning("lookup_gene failed for %r: %s", lookup_target, e)
            continue
        lines.append(result)
    return lines


def _score_with_tools(
    client: ModelClient,
    record: ScoringRecord,
    max_tokens: int | None,
    *,
    variant: ScoringVariant | None = None,
    margin_out: dict | None = None,
) -> dict:
    """Tier 2 with pre-computed entity lookups. For records where grounding
    is flagged or entity symbols are short/ambiguous, gilda lookups are
    executed deterministically and injected into the prompt. The model
    does not need to decide whether to call the tool — the external
    signal is always present.
    """
    variant = variant or DEFAULT_VARIANT
    lookups = _format_entity_lookups(record)
    examples = _select_examples(record.stmt_type)
    note = _relation_note(client, record, variant=variant)
    execution = _prepare(record, examples, route="tool", lookups=lookups,
                         max_tokens=max_tokens, variant=variant)
    response = client.call(**_call_kwargs(execution, note, variant))
    parsed = parse_response(response)
    _stamp_committed_justification(response, variant=variant)
    selected_examples = _example_trace_rows(examples)
    # Side channel, NOT a new key in the returned dict. The variant-behaviour
    # golden captures these two functions' RETURN SHAPE and is a pre-refactor
    # capture with no regeneration path, so widening it would mean hand-editing
    # a fixture whose whole value is that it predates the change. The margin is
    # probe output; it belongs in the fields replace_sentence_score writes.
    if margin_out is not None:
        margin_out["in_call_label_margin"] = _in_call_margin(response, variant)
    return {
        "verdict": None if parsed is None else parsed.label,
        "confidence": None if parsed is None else parsed.confidence,
        "raw_text": response.raw_text,
        "tokens": response.tokens,
        "selected_example_ids": [ex["id"] for ex in selected_examples],
        "selected_examples": selected_examples,
    }


def _score_categorical(
    client: ModelClient,
    record: ScoringRecord,
    max_tokens: int | None = None,
    *,
    variant: ScoringVariant | None = None,
    margin_out: dict | None = None,
) -> dict:
    """Produce categorical audit output for one extraction.

    Two-tier:
      Tier 1: deterministic grounding auto-reject (mismatch/pseudogene).
      Tier 2: single LLM call (temp=0.1) — tool-use path when grounding
              is flagged; otherwise straight comprehension.

    `variant` selects the scoring profile (prompt, few-shot renderer, parser,
    relation-nature step); it defaults to DEFAULT_VARIANT, resolved from
    MONO_VARIANT at import.

    Returns dict with: score, verdict, confidence, raw_text, tokens,
    tier, grounding_status, provenance_triggered.
    """
    variant = variant or DEFAULT_VARIANT
    _pop = getattr(client, "pop_call_log", lambda: [])
    _pop()

    # --- Tier 0: no evidence sentence -> correct-by-default ---
    # Database-sourced evidence (tas, biogrid, ...) often carries no sentence.
    # With nothing to read it is accepted by default rather than fed an empty
    # prompt the LLM would reject from nothing; anything WITH text falls through
    # to the LLM below. Interim handling: text-less verification by the LLM (from
    # grounding + the asserted relation) is planned and will replace this branch.
    if not (record.evidence_text or "").strip():
        return {
            # There is no sentence to probe, so calibrated score availability is
            # explicitly absent even though the categorical default is correct.
            **NO_TEXT_RESULT,
            "selected_example_ids": [],
            "selected_examples": [],
            "call_log": _pop(),
        }

    # --- Tier 1: Deterministic auto-reject ---
    reject = record.tier1_auto_reject()
    if reject:
        reject["call_log"] = _pop()
        return reject

    # AMBIGUOUS entities skip the intermediate decision path and proceed directly to full comprehension scoring.

    provenance_triggered = bool(record.format_provenance())

    # Determine grounding status now — it picks the Tier-2 path.
    flagged = any(
        e.has_grounding_signal
        for e in (record.subject_entity, record.object_entity)
        if e
    )
    # Pre-computed lookups only fire for flagged grounding: a short-symbol
    # soft-flag adds nothing, since Gilda is the same oracle that blessed the
    # all_match records, so its lookups would return the same ranking. Tool-use
    # only helps where Gilda already flagged a mismatch or ambiguity.
    needs_tool_use = flagged
    grounding_status = "flagged" if flagged else "all_match"

    # --- Tier 2: single LLM call (deterministic, temp=0.1) ---
    if needs_tool_use:
        result = _score_with_tools(client, record, max_tokens, variant=variant,
                                   margin_out=margin_out)
        verdict = result["verdict"]
        confidence = result["confidence"]
        total_tokens = result["tokens"]
        raw = result["raw_text"]
        tier = "llm_tool_use"
    else:
        result = _score_single(client, record, max_tokens, variant=variant,
                               margin_out=margin_out)
        verdict = result["verdict"]
        confidence = result["confidence"]
        total_tokens = result["tokens"]
        raw = result["raw_text"]
        tier = "llm_comprehension"
    call_log = _pop()

    return {
        # The public score() boundary replaces this placeholder with the
        # calibrated direct-probe probability when it is available.
        "score": None,
        "verdict": verdict,
        "confidence": confidence,
        "raw_text": raw,
        "tokens": total_tokens,
        "tier": tier,
        "grounding_status": grounding_status,
        "provenance_triggered": provenance_triggered,
        "selected_example_ids": result.get("selected_example_ids", []),
        "selected_examples": result.get("selected_examples", []),
        "call_log": call_log,
    }


def score(
    client: ModelClient,
    record: ScoringRecord,
    max_tokens: int | None = None,
    *,
    variant: ScoringVariant | None = None,
    extra_probe_call: bool = False,
) -> dict:
    """Score one extraction and emit one calibrated sentence probability.

    The categorical verdict remains an audit output.  The numeric ``score`` is
    replaced at this canonical boundary by the persisted direct-probe
    calibration; an unsupported client, empty sentence, fitted-row leakage
    guard, or probe failure yields ``None`` rather than a categorical midpoint.
    """

    # Collected out-of-band so the two inner functions keep the exact return
    # shape the variant-behaviour golden froze.
    margin: dict = {}
    categorical = _score_categorical(
        client,
        record,
        max_tokens=max_tokens,
        variant=variant,
        margin_out=margin,
    )
    from indra_belief.probes.calibration import replace_sentence_score

    try:
        record_id = f"f{record.source_hash}"
    except Exception:
        record_id = None

    return replace_sentence_score(
        {**categorical, **{k: v for k, v in margin.items() if v is not None}},
        {
            "subject": record.subject,
            "object": record.object,
            "stmt_type": record.stmt_type,
            "evidence_text": record.evidence_text,
        },
        client,
        record_id=record_id,
        extra_probe_call=extra_probe_call,
    )


def score_statement(
    statement,
    evidence,
    client: ModelClient,
    *,
    max_tokens: int | None = None,
    variant: ScoringVariant | None = None,
) -> dict:
    """Score a single INDRA Statement + Evidence pair.

    Single deterministic LLM call per (Statement, Evidence) at temp=0.1
    (with a tool-use variant when grounding is flagged).

    Args:
        statement: An `indra.statements.Statement` instance. Binary types
            (Phosphorylation, Activation, …), SelfModification
            (Autophosphorylation, Transphosphorylation), Complex (any
            arity), and Translocation are rendered correctly.
        evidence: An `indra.statements.Evidence`. Tier-1 grounding
            verification only runs when `evidence.annotations["agents"]
            ["raw_text"]` is populated (i.e., produced by an NLP reader).
            For manually-constructed Evidence, verification is skipped
            and scoring is driven entirely by the LLM tier.
        client: A `ModelClient` configured for the chosen backend.
        max_tokens: Per-generation token limit. Default 12000.
        variant: Scoring profile to use. Defaults to DEFAULT_VARIANT.

    Returns:
        A dict with keys:
            score            calibrated sentence probability when a serving
                             boundary can perform the fitted direct-probe read;
                             otherwise **None**. This categorical call never
                             derives a number from verdict/confidence.
            verdict          "correct" | "incorrect" | None (parse failure)
            confidence       "high" | "medium" | "low" | None
            tier             which scoring path produced the verdict
            grounding_status "all_match" | "flagged"
            provenance_triggered bool
            tokens           completion tokens consumed
            raw_text         decision trace (for debugging)

    Callers should handle `verdict is None` explicitly; it denotes a
    parse failure, not a neutral judgement.
    """
    record = ScoringRecord(statement=statement, evidence=evidence)
    return score(client, record, max_tokens=max_tokens, variant=variant)


def main():
    import argparse
    from indra_belief.data.corpus import CorpusIndex

    parser = argparse.ArgumentParser(description="Evidence quality scorer (INDRA native)")
    # Default model backend is overridable via IBR_MODEL; unchanged default
    # ('gemma-remote') keeps a transient backend from being hard-pinned here.
    parser.add_argument("--model", default=os.environ.get("IBR_MODEL", "gemma-remote"))
    parser.add_argument("--holdout", default=str(ROOT / "data" / "benchmark" / "holdout.jsonl"))
    parser.add_argument("--output", default=str(ROOT / "data" / "results" / "scorer_output.jsonl"))
    parser.add_argument("--max-tokens", type=int, default=12000)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from existing output file (skip scored records)")
    args = parser.parse_args()

    # Load corpus and build records
    index = CorpusIndex()
    records = index.build_records(args.holdout)
    if args.limit:
        records = records[:args.limit]

    # Resume support: skip already-scored records
    scored_hashes = set()
    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.exists():
            with open(resume_path) as f:
                for lineno, line in enumerate(f, start=1):
                    try:
                        r = json.loads(line)
                        scored_hashes.add(r.get("source_hash"))
                    except json.JSONDecodeError:
                        # Skip corrupt NDJSON line; surface it for data-integrity visibility.
                        log.warning(
                            "resume: skipping corrupt JSON line %d in %s: %r",
                            lineno, resume_path, line.rstrip("\n")[:200],
                        )
            print(f"Resuming: {len(scored_hashes)} records already scored")

    print(f"\nScorer: {len(records)} records, model={args.model}")

    client = ModelClient(args.model)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.resume else "w"
    out_fh = open(output_path, mode)

    correct = 0
    total_parsed = 0
    tier_counts = {}
    t_start = time.time()

    for i, record in enumerate(records):
        if record.source_hash in scored_hashes:
            continue

        result = score(client, record, args.max_tokens)

        gt_correct = record.tag == "correct"
        llm_correct = (result["verdict"] == "correct") if result["verdict"] else None

        if llm_correct is not None:
            total_parsed += 1
            if llm_correct == gt_correct:
                correct += 1

        tier = result.get("tier", "?")
        tier_counts[tier] = tier_counts.get(tier, 0) + 1

        result.update({
            "source_hash": record.source_hash,
            "tag": record.tag or "",
            "subject": record.subject,
            "stmt_type": record.stmt_type,
            "object": record.object,
        })

        r_save = {k: v for k, v in result.items() if k != "raw_text"}
        r_save["raw_text_preview"] = result.get("raw_text", "")  # full output — no cap
        out_fh.write(json.dumps(r_save) + "\n")
        out_fh.flush()

        acc = correct / total_parsed * 100 if total_parsed > 0 else 0
        mark = "✓" if (llm_correct == gt_correct) else ("✗" if llm_correct is not None else "?")
        tier_short = {
            "deterministic_mismatch": "T1:MSMATCH",
            "deterministic_pseudogene": "T1:PSEUDO",
            "ambiguous_then_llm": "T1→T2",
            "llm_comprehension": "T2:LLM",
        }.get(tier, tier)
        print(f"  [{i+1:3d}/{len(records)}] {mark} {record.subject:>10s} [{record.stmt_type:>15s}] {record.object:10s} "
              f"→ {result['verdict'] or 'PARSE':>9s} [{tier_short:10s}] acc={acc:.1f}%")

    out_fh.close()

    print(f"\n{'='*70}")
    print(f"RESULTS: {correct}/{total_parsed} = {correct/max(total_parsed,1)*100:.1f}%")
    print(f"Tier breakdown: {tier_counts}")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
