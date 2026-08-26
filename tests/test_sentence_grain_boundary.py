"""The calibrated sentence probability must never reach statement belief.

WHY THIS EXISTS
---------------
The logit-derived per-sentence probability was measured at both grains:

    per sentence  +0.020 AUROC, 95% CI [+0.007, +0.033]   -> a real gain
    per statement +0.004 AUROC, 95% CI spans 0            -> no gain
                  and ECE worsens 0.0199 -> 0.0388        -> an active loss

So the value is shipped at sentence grain only. Today that holds by
construction: ``replace_sentence_score`` writes ``score`` (and ``score_error``),
while ``statement_belief`` builds its number from ``verdict`` / ``source_api`` /
``tier`` plus the fitted profile. The two sets are disjoint.

Nothing enforced that. A single ``row.get("score")`` added to the belief math
would silently double the statement-grain calibration error, and every existing
test would still pass -- they pin values, and the values would simply move
together. This file pins the BOUNDARY instead.

DELIBERATELY STRUCTURAL, NOT NUMERIC
------------------------------------
No assertion here references a calibration constant, a prompt SHA, or an
expected belief value. A guard written against today's numbers would fail the
moment the reader is recalibrated (a new prompt SHA re-fits every log-LR), and
a guard that fails for a legitimate reason is a guard someone edits until it
stops complaining. The invariant is "belief does not depend on the probe's
output", which is true before and after any refit.
"""
from __future__ import annotations

import pytest

from indra_belief.noise_model import RECALIBRATED_PRIORS
from indra_belief.statement_belief import statement_belief

# The fields `replace_sentence_score` writes (probes/calibration.py). `score`
# carries the calibrated p_hat; `score_error` carries its failure.
#
# `weight_of_evidence` — the same reading as an additive weight — is
# deliberately NOT in this set because it is CONSUMABLE. Under AUTO it engages
# by default wherever it was actually measured; `probe_weights=False` still
# refuses it, and a bare hard gate never engages it.
PROBE_OUTPUT_FIELDS = ("score", "score_error")

# Everything `replace_sentence_score` WRITES. Kept separate from
# PROBE_OUTPUT_FIELDS above because the two sets answer different questions:
# what the producer emits, versus what belief must never read unasked.
# `weight_of_evidence` is in this set and not that one — it is consumable under
# AUTO wherever it was actually measured.
PROBE_WRITTEN_FIELDS = ("score", "score_error", "weight_of_evidence",
                        "probe_delta_logit")

# A synthetic profile, NOT a registered one: the boundary is a property of the
# code path, not of any particular fitted reader, and a synthetic profile keeps
# this file valid across recalibration.
SYNTHETIC_PROFILE = {
    "prior_logodds": 0.25,
    "log_lr_confirm": 1.5,
    "log_lr_reject": -1.25,
}

BASE_ROWS = [
    {"source_api": "reach", "verdict": "correct", "tier": 1,
     "evidence_text": "A phosphorylates B.", "evidence_hash": "e1"},
    {"source_api": "reach", "verdict": "incorrect", "tier": 1,
     "evidence_text": "B binds C.", "evidence_hash": "e2"},
    {"source_api": "sparser", "verdict": "correct", "tier": 2,
     "evidence_text": "C activates D.", "evidence_hash": "e3"},
    {"source_api": "signor", "verdict": None, "tier": 1,
     "evidence_text": "D inhibits E.", "evidence_hash": "e4"},
]


class _RecordingRow(dict):
    """A row that remembers which keys were actually read.

    This is what makes the guard self-updating: it does not compare against a
    hand-maintained list of consumed fields (which would drift), it observes the
    real read set of the real call.
    """

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.read: set[str] = set()

    def __getitem__(self, key):
        self.read.add(key)
        return super().__getitem__(key)

    def get(self, key, default=None):
        self.read.add(key)
        return super().get(key, default)


def _rows_with(score_value, cls=dict):
    return [cls({**r, "score": score_value, "score_error": None}) for r in BASE_ROWS]


def _rows_with_ell(ell_value):
    return [{**r, "weight_of_evidence": ell_value} for r in BASE_ROWS]


@pytest.mark.parametrize("soft", [None, SYNTHETIC_PROFILE], ids=["hard", "calibrated"])
def test_statement_belief_never_reads_the_probe_output(soft):
    """The read set of a real belief computation excludes the probe's fields."""
    rows = _rows_with(0.9, cls=_RecordingRow)
    statement_belief(rows, RECALIBRATED_PRIORS, soft=soft)
    seen: set[str] = set()
    for r in rows:
        seen |= r.read
    assert seen, "recorded no field reads at all - the probe is not observing the call"
    leaked = seen.intersection(PROBE_OUTPUT_FIELDS)
    assert not leaked, (
        f"statement_belief read {sorted(leaked)} from its evidence rows. The "
        "calibrated sentence probability is sentence-grain only: at statement "
        "grain it gains nothing (CI spans 0) and worsens ECE 0.0199 -> 0.0388."
    )


@pytest.mark.parametrize("soft", [None, SYNTHETIC_PROFILE], ids=["hard", "calibrated"])
def test_belief_is_invariant_to_the_probe_output(soft):
    """Belief is byte-identical across adversarial probe values.

    The read-set test above catches a direct read. This catches an indirect one
    -- a helper that copies the row, a dict merge, anything that launders the
    field past a `.get`.
    """
    def belief_for(v):
        sb = statement_belief(_rows_with(v), RECALIBRATED_PRIORS, soft=soft)
        return sb.belief

    reference = belief_for(None)
    for value in (0.0, 0.5, 1.0, 0.9999, -3.0, "not-a-number"):
        assert belief_for(value) == reference, (
            f"statement belief moved when the sentence score was {value!r}; "
            "the probe's output must not reach statement grain"
        )


def test_the_probe_writes_only_fields_belief_ignores():
    """The other half of the boundary, asserted at the producer.

    Pins the contract from `replace_sentence_score`'s side, so widening what it
    writes cannot quietly reintroduce a path this file does not model.
    """
    from indra_belief.probes.calibration import replace_sentence_score

    original = {"verdict": "correct", "source_api": "reach", "tier": 1}
    # enabled=False exercises the unavailable branch without a client or network.
    enriched = replace_sentence_score(
        original, {}, client=None, record_id="sh1", enabled=False
    )
    written = {k for k in enriched if k not in original or enriched[k] != original.get(k)}
    assert written <= set(PROBE_WRITTEN_FIELDS), (
        f"replace_sentence_score now writes {sorted(written)}; anything outside "
        f"{sorted(PROBE_WRITTEN_FIELDS)} needs checking against the belief read set"
    )
    assert "probe_delta_logit" in enriched, (
        "the RAW reading must be persisted even when unavailable — it is what "
        "lets a new serving stack fit its own calibration from its first run "
        "instead of needing a second pass"
    )
    assert "weight_of_evidence" in enriched, (
        "the additive weight must be persisted even when unavailable — without "
        "the key, statement_belief(probe_weights=True) is a silent no-op and "
        "an absent reading is indistinguishable from an unprobed run"
    )
    for field in ("verdict", "source_api", "tier"):
        assert enriched[field] == original[field], f"the probe overwrote {field}"


# ── the opt-in half: explicitly requested, the probe DOES reach belief ────────

def test_a_measured_weight_engages_by_default():
    """CONTRACT CHANGE, made deliberately: the default now USES a measured weight.

    This test previously asserted the opposite — that a row carrying
    `weight_of_evidence` was ignored unless a flag was set. That invariant was
    traded away on purpose. Flag-gated, the logit path was inert by
    construction: nothing in the codebase set the flag, so a measurement we paid
    a model call for sat unread on every row that had one.

    What is GIVEN UP: a consumer's numbers move the first time a probe-capable
    client scores their corpus, with no change at their call site. That is the
    real cost and it is why `StatementBelief.weighting` exists — the number
    names the rule that produced it, so the shift is attributable rather than
    mysterious.

    What is KEPT: `score` and `score_error` still never reach belief (the tests
    above), a row without a measured weight is untouched, and `probe_weights=
    False` still refuses. Only the additive weight, only where it was actually
    measured.
    """
    from indra_belief.statement_belief import statement_belief

    plain = statement_belief(BASE_ROWS, RECALIBRATED_PRIORS, soft=SYNTHETIC_PROFILE)
    carrying = statement_belief(_rows_with_ell(3.0), RECALIBRATED_PRIORS,
                                soft=SYNTHETIC_PROFILE)
    assert carrying.belief != plain.belief, (
        "a measured weight was present and ignored — the logit path is inert again"
    )
    assert carrying.weighting == "probe_weight"
    assert plain.weighting == "verdict_weight"


def test_a_row_without_a_measured_weight_is_untouched():
    """AUTO must not change anything it has no measurement for."""
    from indra_belief.statement_belief import statement_belief

    off = statement_belief(BASE_ROWS, RECALIBRATED_PRIORS, soft=SYNTHETIC_PROFILE,
                           probe_weights=False)
    auto = statement_belief(BASE_ROWS, RECALIBRATED_PRIORS, soft=SYNTHETIC_PROFILE)
    assert auto.belief == off.belief
    assert auto.weighting == "verdict_weight"


def test_auto_never_engages_on_the_hard_gate():
    """No fitted profile means no anchor for the measured weight, and no verdict
    weight to fall back to per row. AUTO stays silent rather than raising."""
    from indra_belief.statement_belief import statement_belief

    sb = statement_belief(_rows_with_ell(3.0), RECALIBRATED_PRIORS, soft=None)
    assert sb.weighting == "hard_gate"


def test_probe_weights_can_still_be_refused():
    from indra_belief.statement_belief import statement_belief

    sb = statement_belief(_rows_with_ell(3.0), RECALIBRATED_PRIORS,
                          soft=SYNTHETIC_PROFILE, probe_weights=False)
    assert sb.weighting == "verdict_weight"


def test_probe_weights_reach_belief_when_asked_and_say_so():
    from indra_belief.statement_belief import statement_belief

    weak = statement_belief(_rows_with_ell(0.2), RECALIBRATED_PRIORS,
                            soft=SYNTHETIC_PROFILE, probe_weights=True)
    strong = statement_belief(_rows_with_ell(3.0), RECALIBRATED_PRIORS,
                              soft=SYNTHETIC_PROFILE, probe_weights=True)
    assert strong.belief > weak.belief, "the magnitude must carry"
    assert strong.weighting == "probe_weight", "the scalar must name the rule that made it"


def test_enabling_probe_weights_without_probe_data_changes_nothing():
    """The safety property that makes the flag deployable ahead of a probe run."""
    from indra_belief.statement_belief import statement_belief

    off = statement_belief(BASE_ROWS, RECALIBRATED_PRIORS, soft=SYNTHETIC_PROFILE)
    on = statement_belief(BASE_ROWS, RECALIBRATED_PRIORS, soft=SYNTHETIC_PROFILE,
                          probe_weights=True)
    assert on.belief == off.belief


def test_probe_weights_refuse_the_hard_gate():
    """The weight is calibrated against a fitted reader's base rate; without one the
    per-row fallback has no verdict weight to fall back TO."""
    import pytest as _pytest
    from indra_belief.statement_belief import statement_belief

    with _pytest.raises(ValueError, match="requires a fitted reader profile"):
        statement_belief(_rows_with_ell(1.0), RECALIBRATED_PRIORS,
                         soft=None, probe_weights=True)
