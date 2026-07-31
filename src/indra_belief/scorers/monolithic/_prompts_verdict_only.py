"""Verdict-only scoring variant — the scorer with its reasoning scaffolding removed.

Every other monolithic variant makes the model deliberate IN ITS ANSWER before
committing: `disconfirm` emits `support` then `objection` then the verdict;
`disconfirm_relnature_rf` (the production default) additionally emits a leading
`relation_check` acceptance clause. Turning the provider's chain-of-thought off
does not touch any of that — the deliberation is in the output contract, not in
a hidden reasoning channel.

This variant removes it. The model emits the verdict and nothing else:

    {"verdict": "correct" | "incorrect", "confidence": "high" | "medium" | "low"}

WHAT IS HELD FIXED. The rule body (rules 1-7) is reused BYTE-IDENTICALLY from
``DISCONFIRM_SYSTEM_PROMPT`` via the same ``.split("HOW TO DECIDE")[0]`` seam
``REASONFIRST_SYSTEM_PROMPT`` uses. Only the decision procedure and the output
contract differ. So a paired run against the reasoning-first arm isolates the
scaffolding and nothing else — same rules, same user message, same evidence.

ONE JUDGMENT CALL, stated rather than buried. The reasoning-first prompt's
closing disposition is phrased in terms of a field this variant deletes ("a
licensing form named in relation_check is grounds to ACCEPT"). It cannot be
copied verbatim. It is translated to the same acceptance-leaning stance
expressed without the field, rather than reverting to `disconfirm`'s stricter
"Default to incorrect" — reverting would silently reintroduce the
over-rejection that the reasoning-first variant was built to fix, and would
confound "no scaffolding" with "stricter disposition".

The relation-nature second call is NOT part of this variant: it is a separate
deliberation step, and a run built on this prompt drops it (call topology
becomes a single monolithic call per execution).
"""
from __future__ import annotations

import json

from indra_belief.scorers.monolithic._prompts_disconfirm import (
    DISCONFIRM_SYSTEM_PROMPT,
    parse_structured as _structured_parse,
)

VERDICT_ONLY_SYSTEM_PROMPT = DISCONFIRM_SYSTEM_PROMPT.split("HOW TO DECIDE")[0] + """\
HOW TO DECIDE (judge silently; emit ONLY the verdict):
- Apply the rules above to the CLAIM and the EVIDENCE and commit directly.
  * correct — the evidence explicitly and directly supports the exact claim,
    including a form an explicit rule licenses: an activity/amount form (rule 3),
    a miRNA inverse (rule 3), a family member (rule 2), an indirect or inverse
    statement that still asserts the direction, or a biological process/phenotype
    object (e.g. proliferation, migration).
  * incorrect — no evidence span states this relation between these two entities,
    or a defeater stands: negation, hypothesis-only, methods/aim framing, wrong
    direction, amount-vs-activity with no licensing form, no-direct-relation, or a
    grounding mismatch.
  Treat the claim as incorrect ONLY when there is no licensing form AND no direct
  span; a rule-licensed form is grounds to ACCEPT.
- Do NOT explain, quote, justify, restate the rules, or emit any field other than
  the two below. No preamble, no commentary, no reasoning.

Output JSON ONLY, exactly these two keys:
{"verdict": "correct" | "incorrect", "confidence": "high" | "medium" | "low"}\
"""


def render_example(ex: dict) -> tuple[str, str]:
    """Render a base contrastive example in verdict-only form.

    The user half matches the other variants exactly except for the trailing
    output-shape hint, so the few-shots differ from the reasoning-first bank in
    precisely the way the system prompt does. The assistant half carries no
    reason, support, objection or relation_check — a few-shot that showed any of
    them would teach the very scaffolding this variant removes.
    """
    user = (
        f"CLAIM: {ex['claim']}\n"
        f"EVIDENCE: {ex['evidence']}\n\n"
        f'Output JSON: {{"verdict": ..., "confidence": ...}}'
    )
    assistant = json.dumps(
        {"verdict": ex["verdict"], "confidence": ex.get("confidence", "high")},
        ensure_ascii=True,
    )
    return user, assistant


def parse_structured(text: str) -> dict:
    """Reuse the structured parser. It already scans for the LAST brace-balanced
    object carrying a `verdict` and tolerates absent support/objection, so a
    two-key answer parses without a variant-specific reader. Keeping one parser
    also means a model that ignores the instruction and emits extra fields anyway
    is still scored, and the extra fields are visible in the persisted raw text
    rather than silently dropping the row to a parse failure."""
    return _structured_parse(text)


def derive_verdict(parsed: dict) -> tuple[str | None, str | None, str]:
    """Pass the model's committed verdict through unchanged — same no-override
    doctrine as the other variants (see _prompts_disconfirm.derive_verdict)."""
    verdict, confidence = parsed.get("verdict"), parsed.get("confidence")
    if verdict is None:
        return None, None, "parse_null"
    return verdict, (confidence or "medium"), "model"
