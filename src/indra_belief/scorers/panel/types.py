"""Panel types. The atomic unit is an Objection — a single, committed,
confident reason an extraction is WRONG.

NO ABSTENTION. A detector returns an Objection (a committed defect) or None.
None means "no defect found," NOT "I'm unsure" — uncertainty resolves to the
accept-default, never to a third state. There is no abstain answer and no
abstain verdict anywhere in this arch (that is the deliberate break from the old
decomposed adjudicator, whose sub-1.0 abstain factors compounded into an 81%
over-reject)."""
from __future__ import annotations

from dataclasses import dataclass

CONFIDENCE_RANK = {"high": 3, "medium": 2, "low": 1}


@dataclass(frozen=True)
class Objection:
    """One detector's committed finding that the extraction is wrong."""
    kind: str          # detector name: grounding | relation_exist | axis | assertion | direction
    defect: str        # machine code: no_relation | fusion_not_complex | amount_not_activity | ...
    confidence: str    # high | medium | low
    source: str        # substrate | llm
    rationale: str = ""
