"""Objection-panel scorer.

Design:

  The disposition is the lever, not knowledge: "accept unless you can COMMIT a
  specific, confident objection." This module factors that disposition into a
  panel of high-precision, single-purpose objection detectors — one per root
  failure category — and ADJUDICATES by accept-unless-objected:

      verdict = incorrect  iff  >=1 detector commits a CONFIDENT objection
      verdict = correct    otherwise   (abstain/uncertain contributes NOTHING)

  Each detector gets ONLY its context, framed AS its question — non-dilutive and
  disconfirming-by-construction:
    - grounding      : Gilda mismatch (substrate) + narrow conflation LLM check
    - relation_exist : syntactic X<->Y binding vs co-mention/third-party/fusion
    - axis           : amount-change vs activity-change (act_vs_amt, wrong_relation)
    - assertion      : asserted finding vs hedged/title/methods (hypothesis)
    - direction      : subject->object direction + sign (polarity)

  Surface-detectable defects go to deterministic substrate; only the
  irreducibly-semantic calls (axis, assertion) use a focused LLM. Each
  detector's commit-threshold is calibrated for PRECISION.
"""
from indra_belief.scorers.panel.orchestrator import score_via_panel  # noqa: F401
