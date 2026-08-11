"""The no-reasoning probe battery as data plus minimal rendering.

This package is separate from :mod:`indra_belief.scorers.probes`, the existing
four-probe decision pipeline.  Every probe here terminates in the same closed
``correct`` / ``incorrect`` token decision; that shared label set is what makes
:func:`indra_belief.logprobs.label_probability` applicable.  This module has no
model client, parser, scoring logic, or metric.

Orientation is deliberately a consumer-side contract.  A2 stores each column's
raw ``p_raw`` from ``label_probability``--P(token ``correct``) at the label
position--for every probe, including flipped probes.  :func:`oriented_p` is
applied exactly once by the B1 consumer, never by the runner.  For
``pol.verdict_flipped``, token ``correct`` means that the proposition "the
extraction is incorrect" is true, so ``1 - p_raw`` restores P(claim is good)
and makes the polarity pair directly comparable.

The declaration is consumed by the combiner/evaluator track in ``.venv`` while
the MLX runner lives in the separate ``~/.venvs/mlx-serve`` environment.  Heavy
model or numerical imports here would couple those two tracks.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from string import Formatter

from indra_belief.hashing import canonical_sha256


LABELS: tuple[str, str] = ("correct", "incorrect")
# Verified 2026-08-09 with this command (wrapped with shell continuations):
# ~/.venvs/mlx-serve/bin/python -c "from transformers import AutoTokenizer; \
# t=AutoTokenizer.from_pretrained('mlx-community/gemma-4-26b-a4b-it-8bit', \
# local_files_only=True); print(t.encode('correct', add_special_tokens=False), \
# t.encode('incorrect', add_special_tokens=False)); print( \
# t.convert_ids_to_tokens([19448]), t.convert_ids_to_tokens([111863]))"
# Result: [19448] [111863], then ['correct'] ['incorrect']; both are single tokens.
# A2 re-asserts the ids against its live tokenizer as defence in depth.
LABEL_TOKEN_IDS: tuple[int, int] = (19448, 111863)

PROBE_FAMILIES: tuple[str, ...] = ("taxonomy", "polarity", "perturbation")

# Three content fields shared by FIT and TEST, plus evidence_text, which TEST
# obtains through the belief_benchmark source_hash join.  source_hash and tag
# are join and label keys respectively; neither is rendered.
RENDER_FIELDS: frozenset[str] = frozenset(
    {"subject", "object", "stmt_type", "evidence_text"}
)

TAXONOMY_TAGS: tuple[str, ...] = (
    "correct",  # FIT n=803
    "no_relation",  # FIT n=225
    "grounding",  # FIT n=166
    "wrong_relation",  # FIT n=145
    "other",  # FIT n=62
    "act_vs_amt",  # FIT n=61
    "entity_boundaries",  # FIT n=40
    "hypothesis",  # FIT n=36
    "polarity",  # FIT n=36
    "negative_result",  # FIT n=27
    "mod_site",  # FIT n=4
    "agent_conditions",  # FIT n=1
)


@dataclass(frozen=True)
class Probe:
    """One stable, closed-label question in the ordered battery."""

    id: str
    family: str
    system: str
    user_template: str
    prefill_suffix: str
    orientation: str
    targets: tuple[str, ...]


_DECISION_SYSTEM = (
    "Assess one proposition about a biomedical text-mining extraction against "
    "the supplied claim and evidence. Decide silently. Return only the JSON "
    'verdict value: "correct" when the answer to QUESTION is yes, or '
    '"incorrect" when it is no. Add no analysis, prose, or additional fields.'
)

_PARAPHRASED_DECISION_SYSTEM = (
    "Judge whether one proposition about a biomedical extraction is supported "
    "by the supplied claim and evidence. Work silently and supply only the JSON "
    'verdict value: "correct" for yes and "incorrect" for no. Include no '
    "analysis, commentary, or other fields."
)

_PREFILL_SUFFIX = '{"verdict":"'

_ANCHOR_USER_TEMPLATE = (
    "CLAIM: {subject} [{stmt_type}] {object}\n"
    "EVIDENCE: {evidence_text}\n"
    "QUESTION: Is this extraction correct?"
)


# Taxonomy policy: agent_conditions gets no probe because FIT n=1, and other
# gets none because it is a residue bucket rather than a failure mode. mod_site
# is declared but REPORT-ONLY (FIT n=4): no combiner can learn a weight from
# four positive examples.
PROBES: tuple[Probe, ...] = (
    Probe(
        id="tax.relation_present",
        family="taxonomy",
        system=_DECISION_SYSTEM,
        user_template=(
            "CLAIM: {subject} [{stmt_type}] {object}\n"
            "EVIDENCE: {evidence_text}\n"
            "QUESTION: Does the evidence state any relation between the claim's "
            "subject and object?"
        ),
        prefill_suffix=_PREFILL_SUFFIX,
        orientation="direct",
        targets=("no_relation",),
    ),
    Probe(
        id="tax.subject_grounded",
        family="taxonomy",
        system=_DECISION_SYSTEM,
        user_template=(
            "CLAIM: {subject} [{stmt_type}] {object}\n"
            "EVIDENCE: {evidence_text}\n"
            "QUESTION: Is the claim's SUBJECT actually named or unambiguously "
            "aliased in the evidence?"
        ),
        prefill_suffix=_PREFILL_SUFFIX,
        orientation="direct",
        targets=("grounding", "entity_boundaries"),
    ),
    Probe(
        id="tax.object_grounded",
        family="taxonomy",
        system=_DECISION_SYSTEM,
        user_template=(
            "CLAIM: {subject} [{stmt_type}] {object}\n"
            "EVIDENCE: {evidence_text}\n"
            "QUESTION: Is the claim's OBJECT actually named or unambiguously "
            "aliased in the evidence?"
        ),
        prefill_suffix=_PREFILL_SUFFIX,
        orientation="direct",
        targets=("grounding", "entity_boundaries"),
    ),
    Probe(
        id="tax.relation_type",
        family="taxonomy",
        system=_DECISION_SYSTEM,
        user_template=(
            "CLAIM: {subject} [{stmt_type}] {object}\n"
            "EVIDENCE: {evidence_text}\n"
            "QUESTION: Is the relation described by the evidence the same type "
            "as the claim's relation?"
        ),
        prefill_suffix=_PREFILL_SUFFIX,
        orientation="direct",
        targets=("wrong_relation",),
    ),
    Probe(
        id="tax.activity_vs_amount",
        family="taxonomy",
        system=_DECISION_SYSTEM,
        user_template=(
            "CLAIM: {subject} [{stmt_type}] {object}\n"
            "EVIDENCE: {evidence_text}\n"
            "QUESTION: Does the evidence support the claim's activity-versus-"
            "amount reading?"
        ),
        prefill_suffix=_PREFILL_SUFFIX,
        orientation="direct",
        targets=("act_vs_amt",),
    ),
    Probe(
        id="tax.direction_polarity",
        family="taxonomy",
        system=_DECISION_SYSTEM,
        user_template=(
            "CLAIM: {subject} [{stmt_type}] {object}\n"
            "EVIDENCE: {evidence_text}\n"
            "QUESTION: Does the evidence support the relation in the claim's "
            "direction and sign?"
        ),
        prefill_suffix=_PREFILL_SUFFIX,
        orientation="direct",
        targets=("polarity",),
    ),
    Probe(
        id="tax.assertion_not_hypothesis",
        family="taxonomy",
        system=_DECISION_SYSTEM,
        user_template=(
            "CLAIM: {subject} [{stmt_type}] {object}\n"
            "EVIDENCE: {evidence_text}\n"
            "QUESTION: Is the relation asserted rather than merely "
            "hypothesised, proposed, or tested?"
        ),
        prefill_suffix=_PREFILL_SUFFIX,
        orientation="direct",
        targets=("hypothesis",),
    ),
    Probe(
        id="tax.not_negative_result",
        family="taxonomy",
        system=_DECISION_SYSTEM,
        user_template=(
            "CLAIM: {subject} [{stmt_type}] {object}\n"
            "EVIDENCE: {evidence_text}\n"
            "QUESTION: Is the evidence a positive finding rather than a negation "
            "or failed detection?"
        ),
        prefill_suffix=_PREFILL_SUFFIX,
        orientation="direct",
        targets=("negative_result",),
    ),
    Probe(
        id="tax.mod_site_match",
        family="taxonomy",
        system=_DECISION_SYSTEM,
        user_template=(
            "CLAIM: {subject} [{stmt_type}] {object}\n"
            "EVIDENCE: {evidence_text}\n"
            "QUESTION: If the claim names a residue and position, does the "
            "evidence state that same site?"
        ),
        prefill_suffix=_PREFILL_SUFFIX,
        orientation="direct",
        targets=("mod_site",),
    ),
    # Fixed anchor for the polarity and perturbation families. This is also a
    # previously killed arm: single no-reasoning prefill p_raw reached 0.678
    # AUROC versus the incumbent's 0.748, so it must never be reported alone as
    # the battery's answer.
    Probe(
        id="pol.verdict_direct",
        family="polarity",
        system=_DECISION_SYSTEM,
        user_template=_ANCHOR_USER_TEMPLATE,
        prefill_suffix=_PREFILL_SUFFIX,
        orientation="direct",
        targets=(),
    ),
    Probe(
        id="pol.verdict_flipped",
        family="polarity",
        system=_DECISION_SYSTEM,
        user_template=(
            "CLAIM: {subject} [{stmt_type}] {object}\n"
            "EVIDENCE: {evidence_text}\n"
            "QUESTION: Is this extraction incorrect?"
        ),
        prefill_suffix=_PREFILL_SUFFIX,
        orientation="flipped",
        targets=(),
    ),
    Probe(
        id="pol.relation_direct",
        family="polarity",
        system=_DECISION_SYSTEM,
        user_template=(
            "CLAIM: {subject} [{stmt_type}] {object}\n"
            "EVIDENCE: {evidence_text}\n"
            "QUESTION: Does the evidence state the claimed relation?"
        ),
        prefill_suffix=_PREFILL_SUFFIX,
        orientation="direct",
        targets=("no_relation",),
    ),
    Probe(
        id="pol.relation_flipped",
        family="polarity",
        system=_DECISION_SYSTEM,
        user_template=(
            "CLAIM: {subject} [{stmt_type}] {object}\n"
            "EVIDENCE: {evidence_text}\n"
            "QUESTION: Is the claimed relation absent from the evidence?"
        ),
        prefill_suffix=_PREFILL_SUFFIX,
        orientation="flipped",
        targets=("no_relation",),
    ),
    Probe(
        id="perturb.paraphrase",
        family="perturbation",
        system=_PARAPHRASED_DECISION_SYSTEM,
        user_template=_ANCHOR_USER_TEMPLATE,
        prefill_suffix=_PREFILL_SUFFIX,
        orientation="direct",
        targets=(),
    ),
    Probe(
        id="perturb.evidence_first",
        family="perturbation",
        system=_DECISION_SYSTEM,
        user_template=(
            "EVIDENCE: {evidence_text}\n"
            "CLAIM: {subject} [{stmt_type}] {object}\n"
            "QUESTION: Is this extraction correct?"
        ),
        prefill_suffix=_PREFILL_SUFFIX,
        orientation="direct",
        targets=(),
    ),
    Probe(
        id="perturb.field_order",
        family="perturbation",
        system=_DECISION_SYSTEM,
        user_template=(
            "CLAIM FIELDS:\n"
            "OBJECT: {object}\n"
            "TYPE: {stmt_type}\n"
            "SUBJECT: {subject}\n"
            "EVIDENCE: {evidence_text}\n"
            "QUESTION: Is this extraction correct?"
        ),
        prefill_suffix=_PREFILL_SUFFIX,
        orientation="direct",
        targets=(),
    ),
)

# THE frozen JSONL column order for A2 and feature-matrix column order for B1.
PROBE_IDS: tuple[str, ...] = tuple(probe.id for probe in PROBES)


def probe_by_id(probe_id: str) -> Probe:
    """Return the probe with ``probe_id``; raise KeyError when it is unknown."""
    for probe in PROBES:
        if probe.id == probe_id:
            return probe
    raise KeyError(f"unknown probe id: {probe_id}")


def probes_in_family(family: str) -> tuple[Probe, ...]:
    """Return probes in frozen order for one declared family."""
    if family not in PROBE_FAMILIES:
        raise ValueError(f"unknown probe family: {family}")
    return tuple(probe for probe in PROBES if probe.family == family)


def required_fields(probe: Probe) -> frozenset[str]:
    """Parse the record field names used by ``probe.user_template``."""
    return frozenset(
        field_name
        for _, field_name, _, _ in Formatter().parse(probe.user_template)
        if field_name is not None
    )


def render(
    probe: Probe,
    record: Mapping[str, object],
) -> tuple[str, str, str]:
    """Render ``(system, user, prefill_suffix)`` from only required fields."""
    values: dict[str, object] = {}
    for field_name in sorted(required_fields(probe)):
        try:
            values[field_name] = record[field_name]
        except KeyError:
            raise KeyError(f"missing render field: {field_name}") from None
    user = probe.user_template.format_map(values)
    return probe.system, user, probe.prefill_suffix


def oriented_p(p_raw: float, orientation: str) -> float:
    """Map raw P(token ``correct``) to P(claim is good) exactly once."""
    if orientation == "direct":
        return p_raw
    if orientation == "flipped":
        return 1.0 - p_raw
    raise ValueError(f"unknown probe orientation: {orientation}")


def battery_digest() -> str:
    """Return the battery's order-sensitive content digest.

    Order sensitivity is by design: reordering probes changes A2's JSONL column
    contract and B1's feature-matrix column contract.
    """
    payload = {
        "labels": LABELS,
        "label_token_ids": LABEL_TOKEN_IDS,
        "probes": [
            (
                probe.id,
                probe.family,
                probe.system,
                probe.user_template,
                probe.prefill_suffix,
                probe.orientation,
                probe.targets,
            )
            for probe in PROBES
        ],
    }
    return canonical_sha256(payload)
