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
    - Output: single JSON verdict extracted via extract_verdict()

The decomposed sibling lives in indra_belief.scorers.probes.*.
Selection between architectures is via the CLI `--arch` flag in
indra_belief.scorers.scorer. This module is invoked through
`score_evidence(stmt, ev, client)` and `score_statement(stmt, client)`
defined at the bottom — same public signatures as the decomposed path.

Run:
    PYTHONPATH=src python -m indra_belief.scorers.scorer --arch monolithic ...
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from indra_belief.data.scoring_record import ScoringRecord
from indra_belief.model_client import ModelClient

# CorpusIndex is imported lazily in main() — the benchmark harness is the
# only consumer. Keeping it out of the module-level import chain means
# `from indra_belief import ModelResponse` (for typing) doesn't pay the
# cost of pulling in the INDRA corpus index.

from indra_belief.scorers.monolithic._prompts import (
    SYSTEM_PROMPT,
    CONTRASTIVE_EXAMPLES as _ALL_EXAMPLES,
    render_example as _render_example,
    extract_verdict,
    verdict_to_score,
)

ROOT = Path(__file__).resolve().parents[4]

# Note: provenance is injected only when grounding is flagged. Full-
# population provenance regresses accuracy — attention dilution from the
# extra context outweighs the disambiguation benefit on records where
# grounding is unambiguous. See scoring_record.format_user_message.

# --- Adaptive few-shot selection ---
# Seven contrastive pairs (14 examples) per record — a balance between
# example coverage and leaving prompt budget for the model's own
# reasoning. More examples dilute attention; fewer lose type coverage.

# Type-specific example bank (loaded from JSON)
# Bank keys can be exact types ("Activation") or sub-keys ("Activation_no_relation")
_EXAMPLE_BANK_PATH = Path(__file__).parent.parent / "data" / "example_bank.json"
_RAW_BANK: dict[str, list[dict]] = {}
if _EXAMPLE_BANK_PATH.exists():
    with open(_EXAMPLE_BANK_PATH) as _f:
        _RAW_BANK = json.load(_f)

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
_BASE_PAIRS: dict[str, list[list[dict]]] = {}
for i in range(0, len(_ALL_EXAMPLES), 2):
    stype = _ALL_EXAMPLES[i]["claim"].split("[")[1].split("]")[0].strip()
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

TARGET_PAIRS = 7  # reduced from 9 — frees ~20% attention for reasoning


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


def _build_messages(record: ScoringRecord, examples: list[dict] | None = None) -> list[dict]:
    """Build the contrastive-example + user-message conversation for a record."""
    examples = examples if examples is not None else _select_examples(record.stmt_type)
    messages: list[dict] = []
    for ex in examples:
        u, a = _render_example(ex)
        messages.append({"role": "user", "content": u})
        messages.append({"role": "assistant", "content": a})
    messages.append({"role": "user", "content": record.format_user_message()})
    return messages


def _parse_verdict(response) -> tuple[str | None, str | None]:
    """Extract verdict from a model response.

    Strategy:
      1. Try parsing `response.content` (final assistant message). For
         separate-reasoning models (Gemma-4), this is CoT-free — guards
         against hypothetical JSON in reasoning being picked up by
         extract_verdict's last-match logic.
      2. If content is empty OR yields no verdict, fall back to `raw_text`
         (reasoning + content joined). Recovers truncated responses where
         the verdict is in reasoning and final content is incomplete.

    The two-step fallback is load-bearing: truncation at max_tokens
    (finish_reason="length") can produce a non-empty content that lacks
    the JSON verdict. Without the fallback, such records silently
    collapse to (None, None) → score 0.5.
    """
    if response.content:
        verdict, confidence = extract_verdict(response.content)
        if verdict is not None:
            return verdict, confidence
    # Fall back to full raw_text (includes reasoning).
    return extract_verdict(response.raw_text)


def _score_single(
    client: ModelClient,
    record: ScoringRecord,
    max_tokens: int | None,
    temperature: float = 0.1,
) -> dict:
    """Single LLM call for Tier 2. Returns result dict."""
    examples = _select_examples(record.stmt_type)
    response = client.call(
        system=SYSTEM_PROMPT,
        messages=_build_messages(record, examples),
        max_tokens=max_tokens,
        temperature=temperature,
        kind="monolithic",
    )
    verdict, confidence = _parse_verdict(response)
    selected_examples = _example_trace_rows(examples)
    return {
        "verdict": verdict,
        "confidence": confidence,
        "raw_text": response.raw_text,
        "tokens": response.tokens,
        "selected_example_ids": [ex["id"] for ex in selected_examples],
        "selected_examples": selected_examples,
    }


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


def _format_entity_lookups(record: ScoringRecord) -> str:
    """Pre-compute gilda lookups for ambiguous entities. Returns a
    formatted block suitable for prompt injection, or "" when neither
    entity benefits from lookup.

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
    if not lines:
        return ""
    return "Entity database lookups:\n" + "\n".join(lines)


def _score_with_tools(
    client: ModelClient,
    record: ScoringRecord,
    max_tokens: int | None,
) -> dict:
    """Tier 2 with pre-computed entity lookups. For records where grounding
    is flagged or entity symbols are short/ambiguous, gilda lookups are
    executed deterministically and injected into the prompt. The model
    does not need to decide whether to call the tool — the external
    signal is always present.
    """
    lookup_ctx = _format_entity_lookups(record)
    examples = _select_examples(record.stmt_type)
    messages = _build_messages(record, examples)
    if lookup_ctx:
        # Augment the user message (last message) with the lookup block.
        augmented = messages[-1]["content"] + "\n\n" + lookup_ctx
        messages[-1] = {"role": "user", "content": augmented}

    response = client.call(
        system=SYSTEM_PROMPT + _LOOKUP_GUIDANCE,
        messages=messages,
        max_tokens=max_tokens,
        temperature=0.1,
        kind="monolithic_tool_context",
    )
    verdict, confidence = _parse_verdict(response)
    selected_examples = _example_trace_rows(examples)
    return {
        "verdict": verdict,
        "confidence": confidence,
        "raw_text": response.raw_text,
        "tokens": response.tokens,
        "selected_example_ids": [ex["id"] for ex in selected_examples],
        "selected_examples": selected_examples,
    }


def score(
    client: ModelClient,
    record: ScoringRecord,
    max_tokens: int | None = None,
) -> dict:
    """Score a single extraction with a single deterministic LLM call.

    Two-tier:
      Tier 1: deterministic grounding auto-reject (mismatch/pseudogene).
      Tier 2: single LLM call (temp=0.1) — tool-use variant when grounding
              is flagged; otherwise straight comprehension.

    Returns dict with: score, verdict, confidence, raw_text, tokens,
    tier, grounding_status, provenance_triggered.
    """
    _pop = getattr(client, "pop_call_log", lambda: [])
    _pop()

    # --- Tier 1: Deterministic auto-reject ---
    reject = record.tier1_auto_reject()
    if reject:
        reject["call_log"] = _pop()
        return reject

    # AMBIGUOUS entities go directly to Tier 2 — the intermediate AMBIGUOUS
    # LLM was 64% accurate at scale (barely better than coin flip) and added
    # an extra LLM call that primed the model with grounding-focused evaluation
    # before the comprehension evaluation.

    provenance_triggered = bool(record.format_provenance())

    # Determine grounding status now — it picks the Tier-2 path.
    flagged = any(
        e.has_grounding_signal
        for e in (record.subject_entity, record.object_entity)
        if e
    )
    # Pre-computed-lookups only fire for flagged grounding. The short-symbol
    # soft-flag was considered but rejected: Gilda is the same oracle that
    # blessed all_match records, so its lookups return the same misranking
    # that caused the FP. Tool-use can only help where Gilda already
    # flagged a mismatch or ambiguity.
    needs_tool_use = flagged
    grounding_status = "flagged" if flagged else "all_match"

    # --- Tier 2: single LLM call (deterministic, temp=0.1) ---
    if needs_tool_use:
        result = _score_with_tools(client, record, max_tokens)
        verdict = result["verdict"]
        confidence = result["confidence"]
        total_tokens = result["tokens"]
        raw = f"[TIER 2 TOOL-USE]\n{result['raw_text']}"
        tier = "llm_tool_use"
    else:
        result = _score_single(client, record, max_tokens)
        verdict = result["verdict"]
        confidence = result["confidence"]
        total_tokens = result["tokens"]
        raw = f"[TIER 2 LLM]\n{result['raw_text']}"
        tier = "llm_comprehension"
    call_log = _pop()

    return {
        "score": verdict_to_score(verdict, confidence),
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


def score_statement(
    statement,
    evidence,
    client: ModelClient,
    *,
    max_tokens: int | None = None,
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

    Returns:
        A dict with keys:
            score            float in [0, 1]; 0.95=correct/high … 0.05=incorrect/high.
                             Returns 0.5 when verdict cannot be parsed.
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
    return score(client, record, max_tokens=max_tokens)


def main():
    import argparse
    from indra_belief.data.corpus import CorpusIndex

    parser = argparse.ArgumentParser(description="Evidence quality scorer (INDRA native)")
    parser.add_argument("--model", default="gemma-remote")
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
                for line in f:
                    try:
                        r = json.loads(line)
                        scored_hashes.add(r.get("source_hash"))
                    except json.JSONDecodeError:
                        pass
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
        r_save["raw_text_preview"] = result.get("raw_text", "")[:500]
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
