"""Reading a probe and having a calibration for it are separate questions.

WHY THIS EXISTS
---------------
``supports_sentence_calibration`` used to be one identity check pinned to
``local-gemma-4-26b`` on the MLX model id. That fused two unrelated questions:

  * CAPABILITY  — can this client physically produce a ``delta_logit``?
  * PROFILE     — is there an isotonic map fitted for this serving stack?

Fusing them made the remedy unreachable. Fitting a calibration for a new
serving stack requires reading raw delta_logits ON that stack, and an
identity-pinned gate forbids exactly that. A capable vLLM instance could not be
read, so it could never be calibrated, so it could never become readable.

The pin was protecting something real, and this file pins it separately rather
than discarding it: delta_logit magnitudes are substrate-specific (the same
weights in-process vs over HTTP correlate at r=0.955 but differ 2.4x in range
and disagree in sign on 10% of rows). So an unregistered client must be
readable WITHOUT inheriting another stack's map.

The safety property is therefore: capable-but-unregistered yields NO number,
never a number computed from someone else's isotonic.
"""
from __future__ import annotations

import pytest

from indra_belief.probes import calibration as C


class _Client:
    """Minimal stand-in carrying only what the gates actually inspect."""

    _guard = None

    def __init__(self, model_name, model_id, top_k, backend="openai_compat"):
        self.model_name = model_name
        self.backend = backend
        self.config = {"model_id": model_id, "max_top_logprobs": top_k}


FITTED = _Client(C.CALIBRATION_MODEL, C.CALIBRATION_MODEL_ID, 1024)
# Same weights, different serving stack — the case the old pin could not express.
CAPABLE_UNFITTED = _Client("vllm-local", "google/gemma-4-26B-A4B-it", 1024)
INCAPABLE = _Client("ollama-local", "gemma3:27b", None)
TOO_NARROW = _Client("vllm-local", "google/gemma-4-26B-A4B-it", 20)


def test_the_fitted_client_is_both_readable_and_calibrated():
    assert C.probe_reading_supported(FITTED)
    assert C.supports_sentence_calibration(FITTED)
    assert C.sentence_calibration_path_for(FITTED) is not None


def test_a_capable_but_unregistered_client_is_readable():
    """The bootstrap path. Without this, no new substrate can ever be fitted."""
    assert C.probe_reading_supported(CAPABLE_UNFITTED), (
        "a client declaring enough top_logprobs must be readable even with no "
        "fitted calibration - otherwise fitting one is impossible"
    )


def test_a_capable_but_unregistered_client_is_not_calibrated():
    assert not C.supports_sentence_calibration(CAPABLE_UNFITTED)
    assert C.sentence_calibration_path_for(CAPABLE_UNFITTED) is None


def test_an_unregistered_client_never_borrows_another_stacks_map(monkeypatch):
    """The safety property the old identity pin existed to enforce.

    Absence of a fitted map must produce NO number. A calibrated probability
    computed from a different substrate's isotonic would be wrong by 2.4x in
    range and sign-flipped on ~10% of rows, and would look entirely ordinary.

    `read_probe` is stubbed deliberately. Without the stub these stand-in
    clients cannot issue a request at all, so `score` comes back None whatever
    the registry says and the test proves nothing — it passed unchanged when the
    registry was mutated to fall back to the shipped artifact. With a reading
    that always succeeds, the ONLY thing that can withhold a score is the
    missing registry entry, which is the property under test.
    """
    from indra_belief.probes.reader import ProbeReading

    monkeypatch.setattr(
        C, "read_probe", lambda record, client: ProbeReading(p_raw=0.9, delta_logit=2.0)
    )

    fitted = C.replace_sentence_score(
        {"verdict": "correct"}, {"evidence_text": "A binds B."},
        FITTED, record_id="sh-fitted", extra_probe_call=True,
    )
    assert isinstance(fitted["score"], float), (
        "the stub must produce a real score on the FITTED client, otherwise the "
        "negative case below is vacuous again"
    )

    out = C.replace_sentence_score(
        {"verdict": "correct"}, {"evidence_text": "A binds B."},
        CAPABLE_UNFITTED, record_id="sh1", extra_probe_call=True,
    )
    assert out["score"] is None, (
        "an unregistered serving stack produced a calibrated score; it can only "
        "have come from another stack's fitted map"
    )


@pytest.mark.parametrize("client,label", [(INCAPABLE, "no top_logprobs"),
                                          (TOO_NARROW, "top_logprobs below the measured floor")])
def test_incapable_clients_are_not_readable(client, label):
    assert not C.probe_reading_supported(client), label
    assert not C.supports_sentence_calibration(client)


def test_the_registry_is_keyed_on_the_served_id_not_just_the_name():
    """Same registry name, different served weights, must not resolve.

    The served id is in the key because that is what identifies the stack; a
    registry entry repointed at a different model would otherwise keep matching.
    """
    impostor = _Client(C.CALIBRATION_MODEL, "some-other/quantization", 1024)
    assert C.sentence_calibration_path_for(impostor) is None
    assert not C.supports_sentence_calibration(impostor)


# ── an artifact must be SCORABLE, not merely loadable ─────────────────────────
#
# The loader was widened to accept either route's probe id; the SCORING call was
# not widened with it and kept passing the direct probe's id as a constant. So a
# freshly fitted in-call artifact loaded cleanly and then raised "X column order
# does not match probe_ids" on every row it was asked to score. The corpus
# bridge's bare `except` turned that into a counter, and 1930 tests passed --
# because the only round-trip test stopped at load.

def test_an_in_call_artifact_can_be_scored_not_just_loaded(tmp_path):
    """The regression. Load is half a round trip; use is the other half."""
    import json

    import numpy as np

    from indra_belief.probe_combiner import fit_combiner
    from indra_belief.probes.calibration import _calibration_at, calibrate_probe
    from indra_belief.probes.reader import IN_CALL_PROBE_ID, ProbeReading

    rng = np.random.default_rng(0)
    margins = np.concatenate([rng.normal(6, 3, 60), rng.normal(-6, 3, 60)])
    labels = np.array([True] * 60 + [False] * 60)
    combiner = fit_combiner(
        margins.reshape(-1, 1), labels,
        probe_ids=[IN_CALL_PROBE_ID],
        record_ids=[f"fit-{i}" for i in range(len(margins))],
    )
    path = tmp_path / "incall.json"
    path.write_text(json.dumps(combiner.to_dict()))

    loaded = _calibration_at(path)
    assert loaded.probe_ids == (IN_CALL_PROBE_ID,)

    reading = calibrate_probe(
        ProbeReading(p_raw=float("nan"), delta_logit=8.0),
        record_id="never-seen", calibration=loaded,
    )
    assert 0.0 <= reading.p_hat <= 1.0
    assert isinstance(reading.weight_of_evidence, float)


def test_the_shipped_direct_artifact_still_scores():
    """The other half of the widening: accepting a second id must not break the
    one that was already there."""
    from indra_belief.probes.calibration import (
        DEFAULT_CALIBRATION_PATH, _calibration_at, calibrate_probe,
    )
    from indra_belief.probes.reader import DIRECT_PROBE_ID, ProbeReading

    cal = _calibration_at(DEFAULT_CALIBRATION_PATH)
    assert cal.probe_ids == (DIRECT_PROBE_ID,)
    reading = calibrate_probe(
        ProbeReading(p_raw=float("nan"), delta_logit=1.0),
        record_id="never-seen", calibration=cal,
    )
    assert 0.0 <= reading.p_hat <= 1.0
