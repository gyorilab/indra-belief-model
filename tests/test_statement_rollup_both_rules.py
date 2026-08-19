"""The metrics export reports statement gold under BOTH rollup rules.

WHY
---
Statement gold is not observed. Curators label EVIDENCES; a statement label is
manufactured by a rollup rule, and the two defensible rules disagree:

  any-incorrect-wins  correct iff EVERY curated evidence is correct
                      -> a curation-quality question; falls with evidence count
  any-correct-wins    correct iff ANY curated evidence is correct
                      -> what a noisy-OR belief claims; rises with evidence count

MEASURED on holdout_large_fit (n=2533): r(n_evidence, gold) is -0.104 under the
first rule and +0.055 under the second, while r(n_evidence, INDRA belief) is
+0.241. So a statement-grain metric computed under the shipped rule is partly
reporting a definitional mismatch rather than model quality.

The export therefore carries both and names each. It does NOT switch, because
the rules do not favour one arm consistently: the noisy-OR-matched rule improves
calibrated ECE on holdout_large_fit (0.1772 -> 0.1334 at 4+ evidences) and
worsens it on eval_curation_v1 (0.0462 -> 0.0540).

This file pins the contract and the two structural invariants that make the pair
interpretable. It deliberately does NOT pin either rule's metric values — those
are properties of a run, and freezing them here would turn a reporting contract
into a golden test that fights every legitimate re-measurement.
"""
from __future__ import annotations

import pytest

from indra_belief.results import METRICS_SCHEMA_VERSION, _statement_gold_any_correct


def test_the_alternative_rollup_rule_is_any_correct_wins():
    assert _statement_gold_any_correct(["correct"]) == "correct"
    assert _statement_gold_any_correct(["grounding"]) == "incorrect"
    # the discriminating case: one good evidence among bad ones
    assert _statement_gold_any_correct(["grounding", "correct"]) == "correct"
    # ...where the shipped rule says the opposite
    from indra_belief.curation import aggregate_gold

    assert aggregate_gold(["grounding", "correct"]) == "incorrect"
    assert _statement_gold_any_correct([]) is None, "uncurated is not a verdict"


@pytest.mark.parametrize("tags", [["correct"], ["grounding"], ["polarity"]])
def test_single_evidence_statements_are_identical_under_both_rules(tags):
    """The internal check that any divergence is a real multi-evidence effect.

    With one curated evidence, "all" and "any" coincide by construction. If this
    ever fails, the two rules differ somewhere they cannot, and every
    multi-evidence comparison built on them is suspect.
    """
    from indra_belief.curation import aggregate_gold

    assert aggregate_gold(tags) == _statement_gold_any_correct(tags)


def test_schema_version_advertises_the_new_block():
    assert METRICS_SCHEMA_VERSION >= 4, (
        "tiers.stmt.by_rollup was added at v4; a consumer pinning v3 must be able "
        "to tell that the block exists"
    )


def _fake_stmt_block(rows):
    """Run the export's statement block over synthetic rows.

    Uses the real _metric_block so the arms are the shipped computation, not a
    reimplementation.
    """
    from indra_belief.results import _metric_block

    labels = [r["gold_correct"] for r in rows]
    alt = [r["gold_correct_any"] for r in rows]
    return (
        _metric_block([r["hard"] for r in rows], labels, tau=0.5),
        _metric_block([r["hard"] for r in rows], alt, tau=0.5),
    )


def test_the_two_rules_can_only_disagree_on_multi_evidence_statements():
    """The invariant the export reports as `disagreement`.

    A single-evidence statement has one tag, so both rules read it the same way.
    Any relabelling therefore has to come from a multi-evidence statement, and
    n_statements_relabelled <= n_multi_evidence is a structural fact rather than
    an empirical observation about one corpus.
    """
    from indra_belief.curation import aggregate_gold

    corpus = [
        ["correct"],                      # single, agrees
        ["grounding"],                    # single, agrees
        ["correct", "correct"],           # multi, agrees
        ["correct", "grounding"],         # multi, DISAGREES
        ["grounding", "polarity"],        # multi, agrees (both incorrect)
    ]
    relabelled = sum(
        1 for tags in corpus
        if aggregate_gold(tags) != _statement_gold_any_correct(tags)
    )
    multi = sum(1 for tags in corpus if len(tags) > 1)
    assert relabelled == 1
    assert relabelled <= multi
    for tags in corpus:
        if len(tags) == 1:
            assert aggregate_gold(tags) == _statement_gold_any_correct(tags)
