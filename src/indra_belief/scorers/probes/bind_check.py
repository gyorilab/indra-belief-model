"""Deterministic bind-check: extracted relation tuple → relation_axis answer.

The extract-then-bind-check architecture separates two concerns the
legacy `relation_axis` LLM probe was conflating:

  (1) WHAT does the evidence assert?  (Stage 1 — LLM extraction)
  (2) DOES it match the claim?         (Stage 2 — this module)

Stage 2 is deterministic Python: axis taxonomy + alias resolution +
sign reconciliation → one of the 8 RelationAxisAnswer values.

The CC-phase rationale: at 27B-Gemma scale, asking an LLM to
simultaneously extract and classify produces brittle decisions that
cluster around adversarial vs counter-example prompt phrasings.
Splitting the question and resolving the comparison in Python
eliminates the prompt-saturation problem (where 3 adversarial
few-shots dominate 1 positive shot via attention competition) and
makes the answer-set semantics auditable.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


# Mapping from claim's typed axis to the set of evidence axis_subtypes
# that count as "same axis" for the bind-check.
#
# INDRA encodes axis-with-sign via Statement type; here we factor that:
# - Phosphorylation accepts {phosphorylation, transphosphorylation,
#   autophosphorylation}; Dephosphorylation only matches the inverse;
#   etc.
# - Activation/Inhibition share axis=activity; sign distinguishes.
# - IncreaseAmount/DecreaseAmount share axis=amount; sign distinguishes.
# - Complex requires axis=binding with partner_type=protein.
# - Translocation requires axis=localization.
_AXIS_TAXONOMY: dict[str, frozenset[str]] = {
    "activity": frozenset({
        "activity", "activation", "inhibition", "regulation",
        "signal", "signaling", "stimulation", "suppression",
    }),
    "modification": frozenset({
        "modification",
        # Phosphorylation family — including the indirect-but-direct
        # variants the old closed-set probe couldn't represent.
        "phosphorylation", "transphosphorylation", "autophosphorylation",
        "dephosphorylation",
        # Other PTMs
        "acetylation", "deacetylation",
        "methylation", "demethylation",
        "ubiquitination", "deubiquitination",
        "sumoylation", "desumoylation",
        "glycosylation", "deglycosylation",
        "hydroxylation",
        "palmitoylation",
        "myristoylation",
        "ribosylation",
        "farnesylation",
        "geranylgeranylation",
    }),
    "amount": frozenset({
        "amount", "expression", "transcription", "translation",
        "secretion", "release", "production",
        "increase", "decrease",
        "stabilization", "destabilization", "degradation",
    }),
    "binding": frozenset({
        "binding", "complex", "interaction", "association",
    }),
    "localization": frozenset({
        "localization", "translocation", "transport", "import",
        "export", "sequestration",
    }),
}


# Subtype → sign default. When the extracted subtype implies a sign
# (e.g., "destabilization" implies negative for amount), and the LLM
# left sign=unknown, we can recover the sign from the subtype.
# Empty mapping for ambiguous subtypes.
_SUBTYPE_SIGN_DEFAULTS: dict[str, str] = {
    "phosphorylation": "positive",
    "transphosphorylation": "positive",
    "autophosphorylation": "positive",
    "dephosphorylation": "negative",
    "acetylation": "positive",
    "deacetylation": "negative",
    "methylation": "positive",
    "demethylation": "negative",
    "ubiquitination": "positive",
    "deubiquitination": "negative",
    "sumoylation": "positive",
    "desumoylation": "negative",
    "activation": "positive",
    "stimulation": "positive",
    "inhibition": "negative",
    "suppression": "negative",
    "increase": "positive",
    "stabilization": "positive",
    "decrease": "negative",
    "destabilization": "negative",
    "degradation": "negative",
}


# INDRA stmt_type → (axis_category, sign) factoring. Used to read the
# claim's typed axis from the stmt_type when claim_meta provides one.
_STMT_TYPE_TO_AXIS_SIGN: dict[str, tuple[str, str]] = {
    "Phosphorylation":          ("modification", "positive"),
    "Dephosphorylation":        ("modification", "negative"),
    "Acetylation":              ("modification", "positive"),
    "Deacetylation":            ("modification", "negative"),
    "Methylation":              ("modification", "positive"),
    "Demethylation":            ("modification", "negative"),
    "Ubiquitination":           ("modification", "positive"),
    "Deubiquitination":         ("modification", "negative"),
    "Sumoylation":              ("modification", "positive"),
    "Desumoylation":            ("modification", "negative"),
    "Autophosphorylation":      ("modification", "positive"),
    "Transphosphorylation":     ("modification", "positive"),
    "Activation":               ("activity", "positive"),
    "Inhibition":               ("activity", "negative"),
    "IncreaseAmount":           ("amount", "positive"),
    "DecreaseAmount":           ("amount", "negative"),
    "Complex":                  ("binding", "neutral"),
    "Translocation":            ("localization", "neutral"),
}


@dataclass(frozen=True)
class ExtractedRelation:
    """Stage-1 LLM extraction output. All fields can be None/unknown when
    evidence underdetermines."""
    subject_acts_on_object: str          # see _SUBJECT_ACTS values below
    object_extracted: str | None
    intermediate: str | None
    axis: str                            # category: modification, activity, ...
    subtype: str                         # specific verb
    sign: str                            # positive, negative, neutral, unknown
    perturbation: str                    # LOF, GOF, none
    scope: str                           # asserted, hedged, negated, abstain
    partner_type: str                    # protein, DNA, ..., unknown
    rationale: str = ""


# Closed set for subject_acts_on_object — describes how the claim
# subject relates to the claim object in the evidence.
_SUBJECT_ACTS = frozenset({
    "directly",            # claim subject directly acts on claim object
    "via_named_chain",     # via a named intermediate
    "via_unnamed_chain",   # chain marker present but no extractable intermediate
    "on_other_entity",     # claim subject acts on a different entity
    "is_target",           # claim subject is the TARGET (role-swap)
    "absent",              # claim subject not asserted to act
    "abstain",             # underdetermined
})


# Closed set for axis (Stage-1 extraction; broader than RelationAxisAnswer).
_AXIS_VALUES = frozenset({
    "modification", "activity", "amount", "binding", "localization", "unknown",
})

_SIGN_VALUES = frozenset({"positive", "negative", "neutral", "unknown"})
_PERTURBATION_VALUES = frozenset({"LOF", "GOF", "none"})
_SCOPE_VALUES = frozenset({"asserted", "hedged", "negated", "abstain"})
_PARTNER_TYPE_VALUES = frozenset({
    "protein", "DNA", "RNA", "lipid", "membrane",
    "complex", "small_molecule", "metal", "unknown",
})


# Greek letter normalization — words and unicode glyphs both map to
# Latin shortform. Mirrors context_builder._GREEK_TO_LATIN to keep
# alias-matching consistent across substrate and bind-check layers.
_GREEK_GLYPHS: dict[str, str] = {
    "α": "a", "β": "b", "γ": "g", "δ": "d", "ε": "e",
    "ζ": "z", "η": "h", "θ": "q", "ι": "i", "κ": "k",
    "λ": "l", "μ": "m", "ν": "n", "ξ": "x", "ο": "o",
    "π": "p", "ρ": "r", "σ": "s", "τ": "t", "υ": "u",
    "φ": "f", "χ": "c", "ψ": "y", "ω": "w",
    "Α": "a", "Β": "b", "Γ": "g", "Δ": "d", "Ε": "e",
    "Ζ": "z", "Η": "h", "Θ": "q", "Ι": "i", "Κ": "k",
    "Λ": "l", "Μ": "m", "Ν": "n", "Ξ": "x", "Ο": "o",
    "Π": "p", "Ρ": "r", "Σ": "s", "Τ": "t", "Υ": "u",
    "Φ": "f", "Χ": "c", "Ψ": "y", "Ω": "w",
}
_GREEK_WORDS: dict[str, str] = {
    "alpha": "a", "beta": "b", "gamma": "g", "delta": "d",
    "epsilon": "e", "zeta": "z", "eta": "h", "theta": "q",
    "iota": "i", "kappa": "k", "lambda": "l", "mu": "m",
    "nu": "n", "xi": "x", "omicron": "o", "pi": "p",
    "rho": "r", "sigma": "s", "tau": "t", "upsilon": "u",
    "phi": "f", "chi": "c", "psi": "y", "omega": "w",
}

# Common biological suffix-qualifiers the LLM may include after an
# entity name (e.g., "NF-κB response", "p53 activity", "EGFR levels").
# Tokens equal to these are dropped during alias matching.
_ENTITY_SUFFIX_QUALIFIERS = frozenset({
    "response", "activity", "expression", "levels", "signaling",
    "pathway", "complex", "receptor", "protein", "family",
    "function", "function", "interaction", "binding",
})


def _norm(s: str | None) -> str:
    """Normalize an entity surface form for alias matching.

    Lowercase, Greek-letter transliteration (both single glyphs and
    spelled-out words), strip non-alphanumerics. Result is a compact
    alphanumeric string suitable for equality checks.
    """
    if not s:
        return ""
    out = s.lower()
    for glyph, latin in _GREEK_GLYPHS.items():
        out = out.replace(glyph, latin)
    for word, latin in _GREEK_WORDS.items():
        out = out.replace(word, latin)
    return "".join(ch for ch in out if ch.isalnum())


def _tokens(s: str | None) -> list[str]:
    """Split surface form into normalized alphanumeric tokens.

    Splits on whitespace and punctuation EXCEPT hyphens — many entity
    names embed hyphens ("NF-κB", "NF-kB", "alpha-Dgk", "TNF-α") and
    splitting on them would shatter the entity. Normalizes each token
    via _norm. Empty tokens and recognized suffix-qualifiers (response,
    activity, etc.) are dropped.
    """
    if not s:
        return []
    import re
    parts = re.split(r"[\s_/,()\[\].:;]+", s)
    out: list[str] = []
    for p in parts:
        norm = _norm(p)
        if not norm:
            continue
        if norm in _ENTITY_SUFFIX_QUALIFIERS:
            continue
        out.append(norm)
    return out


def _matches_alias(extracted_name: str | None,
                   alias_set: frozenset[str]) -> bool:
    """True iff `extracted_name` matches any alias.

    Match priority:
      1. Exact normalized equality.
      2. Token-level: any normalized token of extracted_name equals
         a normalized alias (handles "NF-κB response" → tokens ["nfkb",
         "response"]; "response" drops via _ENTITY_SUFFIX_QUALIFIERS;
         "nfkb" matches alias "NF-kappaB" → "nfkb").
      3. Suffix/prefix containment for chemical-prefix patterns like
         "alpha-Dgk" vs "DGK" (≥3 chars both sides).
    """
    if not extracted_name:
        return False
    norm_extracted = _norm(extracted_name)
    if not norm_extracted:
        return False
    # Tier 1: exact normalized match
    for alias in alias_set:
        if _norm(alias) == norm_extracted:
            return True
    # Tier 2: token-level match
    extracted_tokens = _tokens(extracted_name)
    if extracted_tokens:
        for alias in alias_set:
            norm_alias = _norm(alias)
            if not norm_alias or len(norm_alias) < 2:
                continue
            if norm_alias in extracted_tokens:
                return True
        # Also: alias tokens vs extracted tokens (handles "NF-kappaB" →
        # ["nfkappab"] which post-Greek-normalization becomes ["nfkb"]
        # via _norm anyway, but defensive against multi-token aliases).
        for alias in alias_set:
            alias_tokens = _tokens(alias)
            for at in alias_tokens:
                if at and len(at) >= 3 and at in extracted_tokens:
                    return True
    # Tier 3: suffix/prefix containment (e.g., "alpha-Dgk" vs "DGK")
    for alias in alias_set:
        norm_alias = _norm(alias)
        if len(norm_alias) >= 3 and len(norm_extracted) >= 3:
            if norm_alias.endswith(norm_extracted) or \
               norm_extracted.endswith(norm_alias):
                return True
    return False


def _resolve_effective_sign(claim_sign: str,
                            extracted_sign: str,
                            perturbation: str) -> str:
    """Apply LOF/GOF perturbation to the extracted sign.

    Claim sign is what the claim asserts (positive/negative/neutral).
    Extracted sign is what evidence asserts (after the LLM extracts).
    Perturbation on the claim subject inverts the evidence semantics:
    LOF on subject means the evidence describes the INVERSE of what the
    subject normally does. So if evidence says "X knockdown reduces Y",
    extracted sign is negative; perturbation=LOF says "subject is under
    LOF"; effective sign of X's normal effect on Y is POSITIVE.
    """
    if perturbation == "LOF":
        if extracted_sign == "positive":
            return "negative"
        if extracted_sign == "negative":
            return "positive"
    return extracted_sign


def bind_check(
    claim_meta: dict,
    extracted: ExtractedRelation,
) -> tuple[str, str]:
    """Deterministic comparison of extracted relation tuple vs claim.

    Returns (relation_axis_answer, rationale).

    `claim_meta` must include:
      - stmt_type:        INDRA Statement type string (e.g. "Phosphorylation")
      - subject_aliases:  frozenset[str] for claim subject
      - object_aliases:   frozenset[str] for claim object
      - claim_axis:       "modification" | "activity" | "amount" | ...
      - claim_sign:       "positive" | "negative" | "neutral"
      - binding_admissible: frozenset[str] of partner_type strings the
                            claim's stmt_type admits (e.g., {"protein"}
                            for Complex). Empty if no gate.
    """
    # Abstain on scope=abstain (extraction underdetermined)
    if extracted.scope == "abstain":
        return "abstain", "extraction underdetermined"

    # Stage A: subject role check
    sao = extracted.subject_acts_on_object
    if sao == "absent" or sao == "abstain":
        return "abstain", f"subject_acts_on_object={sao}"

    claim_axis = claim_meta.get("claim_axis", "")
    subj_aliases = claim_meta.get("subject_aliases", frozenset())
    obj_aliases = claim_meta.get("object_aliases", frozenset())
    extracted_obj = extracted.object_extracted

    # Role-swap handling.
    # For NON-BINDING axes, claim subject as target → no_relation.
    # For BINDING axis, role-swap is symmetric (Complex(X,Y) ≡ Complex(Y,X)
    # per doctrine §5.3): if the OTHER side of the binding event alias-
    # matches the claim object, accept.
    if sao == "is_target":
        if claim_axis != "binding":
            return "no_relation", "claim subject is target (role-swap), non-binding axis"
        # Binding axis: extracted_obj names the entity binding the claim
        # subject. Accept if it alias-matches the claim object.
        if extracted_obj and _matches_alias(extracted_obj, obj_aliases):
            sao = "directly"  # treat as symmetric direct binding
        else:
            return "no_relation", (
                f"is_target on binding axis, but other party '{extracted_obj}' "
                f"doesn't alias-match claim object")

    # Stage B: object identity check.
    # The "object_extracted" field is what the subject is asserted to
    # act on. Multiple promotion paths handle compound nouns, binding
    # symmetry, and on-other-entity → via_named_chain promotion.

    if sao == "on_other_entity":
        # First, check whether the LLM was wrong about "other": the
        # extracted entity may still alias-match the claim object (the
        # LLM saw a less-common surface form or a compound noun).
        if _matches_alias(extracted_obj, obj_aliases):
            sao = "directly"
        # Binding-axis symmetry: extracted entity may be the claim
        # subject named in a role-swapped position (e.g., "IGF-IR is
        # bound by insulin" with claim Complex(IGF1R, INS) — extracted
        # object would be "insulin" matching INS).
        elif claim_axis == "binding" and \
                _matches_alias(extracted_obj, subj_aliases):
            sao = "directly"
        # Promote to via_named_chain when intermediate links to obj
        elif extracted.intermediate and _matches_alias(
                extracted.intermediate, obj_aliases):
            sao = "via_named_chain"
        else:
            return "no_relation", (
                f"subject acts on '{extracted_obj}', not claim object")

    if sao == "directly" and extracted_obj is not None:
        # Accept if extracted object alias-matches claim object OR
        # (for binding axis) claim subject (symmetric).
        if _matches_alias(extracted_obj, obj_aliases):
            pass
        elif claim_axis == "binding" and \
                _matches_alias(extracted_obj, subj_aliases):
            pass  # binding symmetric — extracted is the OTHER party
        else:
            return "no_relation", (
                f"extracted object '{extracted_obj}' doesn't match claim object")

    # Stage C: chain handling
    if sao == "via_named_chain":
        # Causal claims accept indirect chains per §5.6; direct claims
        # (Complex, Phosphorylation, Translocation) require contact.
        stmt_type = claim_meta.get("stmt_type", "")
        if stmt_type in {"Activation", "Inhibition",
                         "IncreaseAmount", "DecreaseAmount"}:
            return "via_mediator", (
                f"chain via {extracted.intermediate}")
        # Direct claims with a chain → abstain in the legacy semantics
        # (allows substrate-fallback to rescue if CATALOG matches).
        return "via_mediator", (
            f"chain via {extracted.intermediate} (caller-judged)")

    if sao == "via_unnamed_chain":
        return "via_mediator_partial", "chain markers but no extractable intermediate"

    # Stage D: axis match. At this point sao=="directly" and the object
    # alias-matched.
    claim_axis = claim_meta.get("claim_axis", "")
    extracted_axis = extracted.axis
    extracted_subtype = (extracted.subtype or "").lower()

    # Special: if claim_axis is "modification" the subtype must also fit
    # the claim's stmt_type (Phosphorylation accepts phosphorylation /
    # transphosphorylation / autophosphorylation; Acetylation accepts
    # acetylation; etc.). Dephosphorylation does NOT match Phosphorylation
    # — it's a sign flip on the same axis category.
    stmt_type = claim_meta.get("stmt_type", "")
    if claim_axis == "modification":
        claim_subtype_family = _modification_subtype_family(stmt_type)
        if claim_subtype_family and extracted_subtype:
            if extracted_subtype in claim_subtype_family:
                # Same modification family → match continues to sign check
                pass
            elif extracted_subtype in _AXIS_TAXONOMY.get("modification", frozenset()):
                # Different modification subtype → axis mismatch
                return "direct_axis_mismatch", (
                    f"claim modification subtype family={sorted(claim_subtype_family)} "
                    f"vs extracted={extracted_subtype}")
            elif extracted_axis != "modification" and extracted_axis != "unknown":
                return "direct_axis_mismatch", (
                    f"claim axis=modification vs extracted={extracted_axis}")
        elif extracted_axis != claim_axis and extracted_axis != "unknown":
            return "direct_axis_mismatch", (
                f"claim axis={claim_axis} vs extracted={extracted_axis}")
    elif extracted_axis != claim_axis and extracted_axis != "unknown":
        # For activity / amount / binding / localization, the high-level
        # axis must match.
        return "direct_axis_mismatch", (
            f"claim axis={claim_axis} vs extracted={extracted_axis}")

    # Stage E: binding partner-type check (only for binding axis claims).
    if claim_axis == "binding":
        admissible = claim_meta.get("binding_admissible", frozenset())
        partner = extracted.partner_type or "unknown"
        if admissible and partner != "unknown" and partner not in admissible:
            return "direct_partner_mismatch", (
                f"binding partner={partner} not admissible "
                f"(admissible={sorted(admissible)})")

    # Stage F: sign reconciliation with LOF/GOF.
    claim_sign = claim_meta.get("claim_sign", "neutral")
    extracted_sign = extracted.sign
    # Recover sign from subtype if extraction was "unknown"
    if extracted_sign == "unknown" and extracted_subtype in _SUBTYPE_SIGN_DEFAULTS:
        extracted_sign = _SUBTYPE_SIGN_DEFAULTS[extracted_subtype]

    effective_sign = _resolve_effective_sign(
        claim_sign, extracted_sign, extracted.perturbation
    )

    # Neutral axes (binding, translocation, modification family with
    # implicit positive sign) don't need strict sign comparison.
    if claim_sign == "neutral" or effective_sign == "neutral" \
            or effective_sign == "unknown":
        # No usable sign comparison → match by axis alone
        return "direct_sign_match", (
            f"axis match (sign unknown / neutral)")

    if effective_sign == claim_sign:
        return "direct_sign_match", (
            f"axis match, sign match (effective_sign={effective_sign})")

    return "direct_sign_mismatch", (
        f"axis match, sign mismatch "
        f"(claim={claim_sign}, effective={effective_sign})")


def _modification_subtype_family(stmt_type: str) -> frozenset[str]:
    """Return the family of valid extraction subtypes for a modification
    Statement type. Phosphorylation accepts trans/autophosphorylation;
    Dephosphorylation only matches dephosphorylation; etc."""
    families: dict[str, frozenset[str]] = {
        "Phosphorylation": frozenset({
            "phosphorylation", "transphosphorylation", "autophosphorylation",
        }),
        "Autophosphorylation": frozenset({
            "phosphorylation", "autophosphorylation",
        }),
        "Transphosphorylation": frozenset({
            "phosphorylation", "transphosphorylation",
        }),
        "Dephosphorylation": frozenset({"dephosphorylation"}),
        "Acetylation": frozenset({"acetylation"}),
        "Deacetylation": frozenset({"deacetylation"}),
        "Methylation": frozenset({"methylation"}),
        "Demethylation": frozenset({"demethylation"}),
        "Ubiquitination": frozenset({"ubiquitination"}),
        "Deubiquitination": frozenset({"deubiquitination"}),
        "Sumoylation": frozenset({"sumoylation"}),
        "Desumoylation": frozenset({"desumoylation"}),
    }
    return families.get(stmt_type, frozenset())


def stmt_type_to_axis_sign(stmt_type: str) -> tuple[str, str]:
    """Map INDRA Statement type to (axis_category, claim_sign).
    Returns ('unknown', 'neutral') for unmapped types."""
    return _STMT_TYPE_TO_AXIS_SIGN.get(stmt_type, ("unknown", "neutral"))
