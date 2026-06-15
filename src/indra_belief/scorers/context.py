"""Pre-resolved knowledge that every decomposed sub-call needs.

Before this module existed, parse_evidence ran blind: it received a raw
evidence sentence and was expected to extract assertions whose entity
names matched the claim's canonical names. When the sentence used an
alias for an entity, parse_evidence emitted those literal mentions and
grounding rejected the binding — producing absent_relationship false
negatives. Alias-blocked extraction was a leading contributor to that
error class.

EvidenceContext is the single shared resolution object: built once per
(Statement, Evidence) pair, threaded through every sub-call, and
queried — never bypassed. The pattern this commits to:

    no sub-call runs without context

If a sub-call's prompt could plausibly benefit from a piece of
pre-resolvable knowledge that another sub-call has, that knowledge
belongs here, not in a downstream patch.

Construction is pure deterministic Python. No LLM calls. Gilda is the
only external dependency, and Gilda lookups are cached by
GroundedEntity.resolve. Cost: ~5ms per (Statement, Evidence).

------------------------------------------------------------------------
DOCTRINE: deterministic substrate expansion under transformer
          scale constraints
------------------------------------------------------------------------
At this model scale the LLM is reliable for local extraction within a
single sentence but unreliable for: multi-hop reasoning (chains),
counting / enumeration (sites), world-knowledge priors (cytokine vs
kinase), cross-sentence antecedents, and noun-phrase relation
extraction (nominalization). Each LLM call carries JSON-mode reasoning
degradation, latency, and attention budget cost; parse_evidence's
prompt is at the edge of in-context-learning saturation.

The principle: move signal from the LLM to the deterministic
substrate wherever a regex, a Gilda lookup, or a canonical map can
encode the rule reliably. These additions introduce no new LLM
sub-call. Each addition is justified by a specific scale-driven
failure mode; each detector is regex- or Gilda-bounded so correctness
is provable in tests. The LLM is asked only to do what it's actually
good at — paraphrase recognition and semantic interpretation.

------------------------------------------------------------------------
DOCTRINE: relation surface-form expansion
------------------------------------------------------------------------
Substrate expansion covered entity surface forms (alias map), site
surface forms (detected_sites), and ontology classes. But the parser
remained solo on RELATION extraction, and absent_relationship false
negatives still dominated: the parser failed to emit an assertion the
adjudicator could bind even though the evidence sentence supported the
claim. The leading classes were nominalization, Greek/hyphen alias
variants, perturbation language, cascade/chain, sparse family
expansion, and reverse binding.

Diagnosis: the LLM has high recall on ONE canonical surface form per
relation (e.g. "X phosphorylates Y") and degrades on alternates
("phosphorylation of Y by X", "X-induced Y phosphorylation",
"X-dependent phosphorylation of Y") under attention pressure as the
prompt grows.

The principle: relations are surface forms of canonical
(axis, sign, X, Y, site?) tuples — exactly as entities are surface
forms of canonical referents — so the SAME substrate-expansion
pattern applies. A regex catalog detects cataloged surface variants
deterministically; the adjudicator falls back to substrate-detected
relations when the parser yields no matching assertion. Coverage
becomes recall(LLM) ∪ recall(regex). A confidence cap on substrate
matches (medium ceiling) preserves parser-confirmed evidence at
high — regex never outranks the LLM on shared inputs.

This expansion introduces no new LLM sub-call. The perturbation and
hedge markers add small prompt nudges (analogous to the chain-signal
and nominalization hints); the rest is pure ctx-and-adjudicator work.

------------------------------------------------------------------------
DOCTRINE: break verifier asymmetry (claim-aware extraction)
------------------------------------------------------------------------
Even after substrate covered every layer the LLM was unreliable on
(relation surface forms, perturbation markers, hedge anchors,
cascade terminals), the decomposed pipeline still trailed the
monolithic baseline on the same model. The gap traced to
architectural lossy compression:
  (a) Schema-strict extraction discards multi-word/dashed/parenthetical
      surface forms when the family backfill doesn't carry the spelled
      written form.
  (b) Claim-blind extraction picks the sentence-initial subject under
      attention pressure on long passages, missing the relevant role
      when it's nested in a participial chain.
  (c) Adjudicator can't recover from the parser binding the wrong agent
      when an inhibitor surface form is present.

The principle: verifier asymmetry — the doctrine that parse_evidence
runs claim-blind to prevent rationalization — is load-bearing at
LARGE scale (where attention isn't a bottleneck) but LOSSY at this
scale (where the parser drops relevant relations under prompt
pressure). The fix breaks the asymmetry CONTROLLEDLY: the parser
receives the claim entities as TOPICS OF INTEREST (an attention focus
cue, not a verdict prompt). The contract becomes claim-AWARE-but-not-
claim-CONFIRMATORY:
  - Schema unchanged: parser still emits structured assertions with
    literal signs, sites, perturbation flags, negation, and
    claim_status.
  - The hint says WHAT TO LOOK FOR (so attention doesn't drop the
    relevant relation), not WHAT TO REPORT (literal extraction is
    enforced, including assertions=[] when the sentence asserts no
    topic-relevant relation).
  - The few-shot curriculum is unchanged — examples remain
    claim-blind in form so the parser learns "extract literally"
    rather than "produce evidence supporting topic".
  - Calibration probes negative-claim records (where the gold
    answer is "incorrect" because the sentence doesn't assert the
    relation) to verify the rationalization rate stays near zero.

This is a CONCESSION on doctrine, not a cleanup. Verifier asymmetry
remains the right principle at scales where attention isn't the
bottleneck; it isn't here.

------------------------------------------------------------------------
DOCTRINE: parse_evidence is one-shot
------------------------------------------------------------------------
The verifier-asymmetry break originally landed alongside a multi-tier
retry pattern in parse_evidence (informed retry, decomposition retry).
That interacted catastrophically with a degraded endpoint: each
timed-out call sat at the full timeout before the retry layer kicked
in, then retried, multiplying latency per record on degraded paths and
stalling whole runs.

The retries are removed. Failure semantics:
  - Transport (TimeoutError, ConnectionError): return None (abstain)
  - Malformed JSON / schema violation: return None (abstain)
  - Valid JSON, zero assertions: return commitment as-is — the
    adjudicator's substrate-fallback bind decides whether the
    regex relation catalog catches what the parser missed
  - Valid commitment: return as-is

Why this is safe to remove:
  1. The substrate now covers the failure classes the retries were
     compensating for: zero-assertion on long sentences, multi-clause
     confusion, nominalization, chain attribution.
  2. At low temperature the model is nearly deterministic — re-asking
     the same question rarely produces meaningfully different output.
  3. Reliability comes from STRUCTURAL redundancy (parse + grounding
     + adjudicate cross-check), not sample-level retry. Self-consistency
     voting was likewise removed for the same reason.
  4. Per-clause parse is preserved — that's structural decomposition
     over different inputs, not retry on the same input.

The clean-removal commitment ("leave nothing behind"): no flag, no
shim, no commented-out blocks. The deprecated retry helpers and the
previous_content / extra_user_message parameters of _attempt_parse are
deleted. A test fails noisily if any future refactor re-introduces
them.

------------------------------------------------------------------------
FIELD INVENTORY (god-object discipline)
------------------------------------------------------------------------
Every field documents (a) the failure class it addresses and (b) the
consumer sub-call that reads it. If a field's failure class is later
resolved by a different mechanism, retire the field — no orphans.

Field                            Failure class                    Consumer
-------------------------------- -------------------------------- ----------------
aliases                          Alias-blocked extraction         parse_evidence,
                                                                  adjudicate
families                         Generic-class actor handling     adjudicate
is_pseudogene                    Pseudogene auto-reject           adjudicate
clauses                          Multi-sentence parse failures    parse_evidence
binding_admissible               Complex-vs-DNA-binding ambiguity adjudicate
acceptable_sites                 Multi-site claim matching        adjudicate
stmt_type / is_complex / etc.    stmt-type predicate cache        adjudicate
has_chain_signal                 Indirect-chain extraction        parse_evidence,
                                                                  adjudicate
chain_intermediate_candidates    Indirect-chain extraction        parse_evidence
subject_class / object_class     Cytokine/ligand subject          adjudicate
subject_precision /              FPLX-FPLX bilateral ambiguity    adjudicate
  object_precision
nominalized_relations            Nominalization extraction        parse_evidence
detected_sites                   Multi-site counting              adjudicate
detected_relations               Relation surface-form expansion  adjudicate
cascade_terminals                Pathway-listing terminal rule    adjudicate
subject_perturbation_marker /    Perturbation-flag inversion      adjudicate
  object_perturbation_marker     (LOF/GOF surface form)
explicit_hedge_markers           Hypothesis claim_status downgrade parse_evidence,
                                                                  adjudicate

Notes on the relation-surface-form additions:
  * detected_relations is the load-bearing recovery path for the
    absent_relationship false negatives. Each capture is
    alias-validated (agent_text + target_text in ctx.aliases) before
    inclusion.
  * cascade_terminals is set-valued; consumer matches when claim
    subject ∈ cascade_terminals AND parser extracted the downstream
    effect on claim object via the cascade.
  * subject_perturbation_marker is a tri-state (None | LOF | GOF);
    None means "no detected perturbation language anchored to
    subject" — the adjudicator's existing inversion logic still
    runs on parser-emitted perturbation when present.
  * explicit_hedge_markers is informational at the parse layer and
    a downgrade trigger at the adjudicate layer (claim_status
    'asserted' → 'hedged' when markers anchor near the relation
    predicate).

Alias normalization is INTERNAL — it patches the aliases dict
construction (Greek↔Latin, hyphen-strip, apostrophe-strip) and the
adjudicate alias-bind helper. No new field surfaces; the change
is observable only in match-rates on Greek/hyphen-variant aliases.

FPLX backfill is INTERNAL — it patches the aliases dict
post-Gilda-resolution with a static roster for under-expanded
families. No new field; the change is observable in
ctx.aliases[<family>] cardinality and in ctx.families coverage.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


# Domain of binding for Complex/IncreaseAmount/Translocation gating. The
# parse_evidence sub-call extracts which of these the binding partner is;
# the adjudicator rejects when the extracted domain isn't in
# ctx.binding_admissible. Complex demands {"protein"} (DNA-binding and
# membrane-binding both look like X-binds-Y on the surface); other
# stmt-types broaden admissibility.
BindingDomain = Literal[
    "protein",          # protein-protein interaction
    "dna",              # DNA / promoter / regulatory element binding
    "rna",              # RNA binding
    "lipid",            # lipid / membrane lipid
    "complex",          # binding to a multi-protein complex
    "membrane",         # cellular membrane
    "small_molecule",   # ligand / drug
    "none",             # no binding described (catalytic, regulatory non-binding)
]


@dataclass(frozen=True)
class DetectedRelation:
    """One regex-detected canonical relation surface form.

    Constructed by context_builder._detect_relations from the
    relation_patterns catalog; consumed by adjudicate's substrate-
    fallback bind.

    Fields:
      axis, sign: the canonical (axis, sign) tuple this surface form maps to.
      agent_canonical: claim-relevant canonical name the agent_text
        alias-binds to (e.g., evidence "Plk1" → "PLK1"). When the
        capture binds to multiple canonical names, the entry-with-this-
        canonical is repeated; the alias-binding logic is in the
        builder.
      target_canonical: same shape, for target.
      site: normalized site (Sxx) when the catalog pattern captured
        one; None otherwise.
      pattern_id: which pattern fired (telemetry + debugging).
      span: char offsets in evidence_text for trace rendering.
    """
    axis: str
    sign: str
    agent_canonical: str
    target_canonical: str
    site: str | None
    pattern_id: str
    span: tuple[int, int]


@dataclass(frozen=True)
class EvidenceContext:
    """Pre-resolved knowledge threaded through every decomposed sub-call.

    Built once at the top of score_evidence_decomposed via
    `from_statement_and_evidence(stmt, ev)`. Read-only after construction
    (frozen). All fields are populated at construction time; no field is
    added or removed mid-pipeline.
    """

    # ---- Alias resolution (was: built post-parse in _build_alias_map) ----
    # Keyed by canonical entity name. Each value is the set of accepted
    # synonyms — including the canonical itself, Gilda-derived aliases,
    # all_names from HGNC/FPLX, and family_members for FPLX entries.
    # Synonyms shorter than 2 chars are dropped (over-match risk).
    #
    # Threading point: injected into parse_evidence prompt as KNOWN
    # MENTIONS so the parser extracts directly to canonical names rather
    # than literal mentions. Also consumed by adjudicate's binding check
    # (where it lives today).
    aliases: dict[str, frozenset[str]] = field(default_factory=dict)

    # ---- Family memberships (FPLX) ----
    # Keyed by canonical name; value is the family-member set when the
    # entity is a family (FPLX-grounded). Used by adjudicate for the
    # generic-class actor rule — direct lookup instead of repeated
    # entity resolution.
    families: dict[str, frozenset[str]] = field(default_factory=dict)

    # ---- Pseudogene flags ----
    # Pre-collected canonical names that are flagged as pseudogenes.
    # Used by adjudicate's auto-reject path (with the evidence-text
    # exception for "pseudogene transcript" / "lncRNA" cases).
    is_pseudogene: frozenset[str] = frozenset()

    # ---- Complexity-driven clause split ----
    # Length 1 when the evidence sentence is simple (≤40 tokens, ≤2
    # subordinators, no coordinated objects); length >1 when split.
    # parse_evidence runs once per clause when len > 1 and the
    # adjudicator unions assertions across clauses. A single-element
    # `(evidence_text,)` means no split.
    clauses: tuple[str, ...] = ()

    # ---- Binding admissibility ----
    # The set of BindingDomain values acceptable for this statement's
    # type. Complex statements admit {"protein"}; broad-binding types
    # admit a wider set. Empty set means "no binding gate" (catalytic
    # statements where the assertion's binding_partner_type is "none").
    binding_admissible: frozenset[str] = frozenset()

    # ---- Multi-site tolerance ----
    # Sites the claim accepts as a match. For a single-site claim
    # (residue=S, position=102), this is {"S102"}. Multi-site claims
    # populate the full set. Empty means "no site requirement".
    acceptable_sites: frozenset[str] = frozenset()

    # ---- Statement-derived helpers ----
    # Cached predicates so sub-calls don't re-introspect stmt.type.
    stmt_type: str = ""
    is_complex: bool = False
    is_modification: bool = False
    is_translocation: bool = False

    # ---- indirect-chain signal detection ----
    # When the evidence text contains chain markers ("thereby",
    # "leads to", "is mediated by", "via", ...), this flag is set so
    # parse_evidence appends a conditional nudge to the user message
    # asking it to populate `intermediates` if applicable. The
    # adjudicator emits a `chain_extraction_gap` informational reason
    # when a parse missed intermediates despite the signal.
    #
    # Multi-hop reasoning is unreliable at this model scale; pattern
    # markers are high-precision regex targets — a better signal source
    # than asking the LLM to discover them under attention pressure.
    has_chain_signal: bool = False
    chain_intermediate_candidates: tuple[str, ...] = ()

    # ---- subject upstream-attribution anchor ----
    # True when the evidence text contains an explicit
    # "X-induced/mediated/driven Y" or "induced/mediated/driven by X"
    # construction where X matches the claim subject (or any of its
    # aliases). Used to gate the cytokine bypass: a cytokine class
    # label alone is insufficient — the bypass only fires when the
    # text also explicitly nominates the cytokine as the upstream
    # actor. Catches "IL6-induced STAT3 phosphorylation" but rejects
    # "IL6 is downstream of TNF in the cascade leading to STAT3
    # phosphorylation" (no IL6-anchored upstream construction).
    subject_has_upstream_anchor: bool = False

    # ---- subject/object semantic class (Gilda-derived) ----
    # Maps each agent to a typed class via Gilda db_ns + ontology
    # subset. Adjudicate auto-fires upstream_attribution when the
    # subject is cytokine/ligand/mirna and the claim axis matches
    # downstream pathway semantics — without requiring parse_evidence
    # to recognize cytokine subjects as a special case.
    #
    # World-knowledge priors are a retrieval task; Gilda is purpose-
    # built for it. The LLM should not be re-deriving this from prompt
    # context.
    subject_class: str = "unknown"
    object_class: str = "unknown"

    # ---- precision class for bilateral-ambiguity guard ----
    # "specific" / "family" / "ambiguous_alias" / "unknown" per agent.
    # When BOTH agents are family OR ambiguous_alias, adjudicate
    # downgrades correct/high → correct/medium with informational
    # reason `bilateral_ambiguity_downgrade`. Asymmetric calibration
    # guard, mirroring presupposition handling.
    subject_precision: str = "unknown"
    object_precision: str = "unknown"

    # ---- pre-detected nominalized relations ----
    # Regex captures of "X-induced/mediated/driven Y of Z" patterns.
    # parse_evidence receives these as a hint so it can emit standard
    # verbal assertions; the LLM is good at the verbal mapping but
    # poor at finding nominalizations in long passages under attention
    # pressure.
    nominalized_relations: tuple[str, ...] = ()

    # ---- regex-detected modification sites ----
    # Normalized canonical site form ("S102", "T461", "Y732").
    # adjudicate's site_check unions parser-extracted sites with this
    # set so multi-site evidence ("X phosphorylates Y at S152, S156,
    # and S163") matches even when the parser drops sites past the
    # first under coordination pressure.
    detected_sites: frozenset[str] = frozenset()

    # ---- deterministic perturbation-marker detection ----
    # Tri-state per-side flags. None = no marker detected; LOF = the
    # entity surface form is anchored to a loss-of-function construct
    # ("X inhibitor", "knockdown of X", "X siRNA"); GOF = anchored to
    # a gain-of-function construct ("overexpression of X",
    # "constitutively active X").
    #
    # The adjudicator inverts effective_sign when parser-emitted
    # perturbation is 'none' but a marker is detected — closing false
    # negatives like "inhibiting X blocked Y phosphorylation" and false
    # positives like "X inhibitors induced Y acetylation".
    #
    # Why deterministic: the parser misses these flags a meaningful
    # fraction of the time at this model scale. Substrate detection
    # makes the inversion gate reliable without re-asking the LLM.
    subject_perturbation_marker: str | None = None
    object_perturbation_marker: str | None = None

    # ---- explicit hedge markers ----
    # Set of hedge phrases detected in evidence text near a claim
    # entity ('may', 'could', 'might', 'we hypothesize', 'is thought
    # to', 'appears to', 'putatively', 'likely'). The adjudicator
    # downgrades parser-emitted claim_status='asserted' → 'hedged'
    # when this set is non-empty AND a hedge marker anchors within
    # 50 chars of any claim entity alias mention. Closes false
    # positives like "X may activate Y" that the parser took as
    # asserted.
    explicit_hedge_markers: frozenset[str] = frozenset()

    # ---- cascade-terminal detection ----
    # The set of entities that appear as the LAST element of a
    # pathway/signaling/cascade listing in evidence text (e.g.,
    # "HER2/PI3K/AKT/NFkappaB pathway" yields cascade_terminals=
    # {"NFkappaB"}). Adjudicate accepts cascade-terminal claims
    # at confidence=medium with reason 'cascade_terminal_match' when
    # the claim subject is in this set AND the claim object is
    # mentioned in evidence — bridging "X is upstream in the
    # cascade" claims that the parser missed because cascade-listing
    # syntax doesn't trigger its verbal-extraction patterns.
    cascade_terminals: frozenset[str] = frozenset()

    # ---- regex-detected canonical relation surface forms ----
    # Each entry is a DetectedRelation: (axis, sign, agent_canonical,
    # target_canonical, site|None, pattern_id, span). Captures from
    # the relation_patterns catalog are alias-validated AND
    # claim-relevance-filtered before inclusion: at least one of
    # {agent_text, target_text} must alias-bind to a claim entity
    # (subject or any object). This drops irrelevant relations
    # mentioned elsewhere in the evidence.
    #
    # Adjudicate consults this as a substrate-fallback when the
    # parser yields no matching assertion: a detected relation whose
    # agent_canonical aliases to claim.subject AND target_canonical
    # aliases to claim.objects[0] AND axis matches (with axis bridges)
    # AND sign matches MATCHES at confidence=medium with reason
    # 'regex_substrate_match'. Confidence ceiling prevents regex from
    # outranking parser-confirmed evidence.
    detected_relations: tuple["DetectedRelation", ...] = ()

    # ---- claim entities for TOPICS OF INTEREST hint ----
    # Sourced from the statement at ctx-build time. Injected into the
    # parse_evidence user message as an attention focus, NOT a verdict
    # prompt. Doctrine break: parse_evidence is no longer claim-blind;
    # it is claim-AWARE-but-not-claim-CONFIRMATORY. The parser still
    # emits literal extractions with literal signs; the claim narrows
    # attention to ensure relevant relations aren't dropped under
    # prompt pressure on long sentences.
    #
    # Why this isn't rationalization: the topic hint tells the parser
    # WHAT TO LOOK FOR, not WHAT TO REPORT. Negative-claim records
    # remain valid outputs (assertions=[] when the sentence doesn't
    # assert the topic relation). Calibration verifies the
    # rationalization rate stays near zero.
    #
    # claim_subject: the canonical name of the first claim entity
    # (parse_claim's subject — first member for Complex, first agent
    # otherwise).
    # claim_objects: the canonical names of the remaining claim
    # entities (parse_claim's objects — members[1:] for Complex,
    # agents[1:] for everything else; (subject,) for SelfModification).
    claim_subject: str = ""
    claim_objects: tuple[str, ...] = ()

    # ---- INDRA KG signal (lazy-populated by orchestrator) ----
    # Set to one of:
    #   None — no KG lookup performed (tests, KG disabled, or coverage gap)
    #   {"kind": "same_axis", "max_belief": float, "count": int,
    #    "total_evidence": int} — curated triple of claim's axis exists
    #   {"kind": "diff_axis", "count": int} — curated triples exist but
    #    only on a different axis (informational; never used to override)
    #
    # Adjudicator confidence policy (KG modifier):
    #   - on verdict=correct AND kind=same_axis AND max_belief >= 0.5:
    #     boost confidence one tier (low → medium, medium → high).
    #   - never affects verdict.
    # The line is asymmetric by design: KG presence boosts; KG absence
    # is silent. Coverage gaps are not negative signal. (Letting absence
    # subtract confidence regressed accuracy.)
    kg_signal: dict | None = None

    @classmethod
    def from_statement_and_evidence(cls, stmt, evidence) -> "EvidenceContext":
        """Build an EvidenceContext from an INDRA Statement and Evidence.

        Delegates to context_builder.build_context. The pipeline
        degrades gracefully: an empty alias map means parse_evidence
        sees no KNOWN MENTIONS block; a single clause means no split.
        """
        from indra_belief.scorers.context_builder import build_context
        return build_context(stmt, evidence)
