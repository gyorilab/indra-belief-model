"""The forced-verdict probe is a strict, serving-callable measurement."""
from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from indra_belief.probes import (
    ProbeReadError,
    ProbeTopKError,
    read_probe,
)
from indra_belief.probes.battery import probe_by_id, render


RECORD = {
    "subject": "MAPK1",
    "object": "JUN",
    "stmt_type": "Activation",
    "evidence_text": "MAPK1 activates JUN in stimulated cells.",
}


def _alternative(token: str, probability: float) -> SimpleNamespace:
    return SimpleNamespace(token=token, logprob=math.log(probability))


def _log_alternative(token: str, logprob: float) -> SimpleNamespace:
    return SimpleNamespace(token=token, logprob=logprob)


def _response(alternatives: list[SimpleNamespace]) -> SimpleNamespace:
    position = SimpleNamespace(
        token=alternatives[0].token if alternatives else "",
        logprob=alternatives[0].logprob if alternatives else None,
        top_logprobs=alternatives,
    )
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                logprobs=SimpleNamespace(content=[position]),
            )
        ]
    )


class _FakeCompletions:
    def __init__(self, response: SimpleNamespace) -> None:
        self.response = response
        self.request: dict | None = None

    def create(self, **kwargs):
        self.request = kwargs
        return self.response


class _FakeClient:
    backend = "openai_compat"

    def __init__(self, alternatives: list[SimpleNamespace]) -> None:
        self.config = {
            "model_id": "test/model",
            "max_top_logprobs": 1024,
            "timeout": 300,
        }
        self.completions = _FakeCompletions(_response(alternatives))
        self._client = SimpleNamespace(
            chat=SimpleNamespace(completions=self.completions)
        )


class _WallTimedFakeClient(_FakeClient):
    def __init__(self, alternatives: list[SimpleNamespace]) -> None:
        super().__init__(alternatives)
        self.wall_timeout: int | float | None = None

    def _invoke_with_wall_timeout(self, function, timeout, **kwargs):
        self.wall_timeout = timeout
        return function(**kwargs)


def test_read_probe_forces_position_zero_and_returns_both_measurements():
    client = _FakeClient(
        [
            _alternative("correct", 0.6),
            _alternative("incorrect", 0.2),
            _alternative("maybe", 0.2),
        ]
    )

    reading = read_probe(RECORD, client)

    assert reading.p_raw == pytest.approx(0.75)
    assert reading.delta_logit == pytest.approx(math.log(0.6) - math.log(0.2))
    assert reading.label_log_odds == reading.delta_logit
    assert tuple(reading) == pytest.approx((0.75, math.log(3.0)))

    probe = probe_by_id("pol.verdict_direct")
    system, user, prefill = render(probe, RECORD)
    assert client.completions.request == {
        "model": "test/model",
        "max_tokens": 1,
        "temperature": 0.0,
        "logprobs": True,
        "top_logprobs": 1024,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": prefill},
        ],
        "extra_body": {
            "chat_template_kwargs": {"enable_thinking": False},
            "continue_final_message": True,
            "add_generation_prompt": False,
        },
        "timeout": 300,
    }


@pytest.mark.parametrize(
    "alternatives",
    [
        [_alternative("correct", 0.9), _alternative("maybe", 0.1)],
        [_alternative("incorrect", 0.9), _alternative("maybe", 0.1)],
    ],
    ids=["incorrect-outside-window", "correct-outside-window"],
)
def test_read_probe_raises_when_either_label_falls_outside_top_k(alternatives):
    client = _FakeClient(alternatives)

    with pytest.raises(ProbeTopKError, match="outside.*top-1024"):
        read_probe(RECORD, client)


@pytest.mark.parametrize("alternatives", [[], [_alternative("maybe", 1.0)]])
def test_read_probe_raises_on_an_empty_or_label_free_distribution(alternatives):
    client = _FakeClient(alternatives)

    with pytest.raises(ProbeReadError):
        read_probe(RECORD, client)


def test_read_probe_keeps_log_odds_exact_when_label_masses_underflow():
    client = _FakeClient(
        [
            _log_alternative("correct", -800.0),
            _log_alternative("incorrect", -801.0),
        ]
    )

    reading = read_probe(RECORD, client)

    assert reading.p_raw == pytest.approx(1.0 / (1.0 + math.exp(-1.0)))
    assert reading.delta_logit == 1.0


def test_read_probe_uses_only_literal_labels_for_both_measurements():
    client = _FakeClient(
        [
            _alternative("in", 0.7),
            _alternative("correct", 0.2),
            _alternative("incorrect", 0.1),
        ]
    )

    reading = read_probe(RECORD, client)

    assert reading.p_raw == pytest.approx(2.0 / 3.0)
    assert reading.delta_logit == pytest.approx(math.log(2.0))


def test_read_probe_uses_model_clients_hard_wall_timeout():
    client = _WallTimedFakeClient(
        [_alternative("correct", 0.6), _alternative("incorrect", 0.4)]
    )

    read_probe(RECORD, client)

    assert client.wall_timeout == 300
    assert client.completions.request is not None


def test_read_probe_never_unwraps_a_guarded_client():
    inner = _FakeClient(
        [_alternative("correct", 0.6), _alternative("incorrect", 0.4)]
    )
    guarded = SimpleNamespace(
        _guard=object(),
        _client=inner,
        backend=inner.backend,
        config=inner.config,
    )

    with pytest.raises(ProbeReadError, match="guarded model clients"):
        read_probe(RECORD, guarded)

    assert inner.completions.request is None


def _calibrated_client():
    from indra_belief.probes.calibration import (
        CALIBRATION_MODEL,
        CALIBRATION_MODEL_ID,
    )

    return SimpleNamespace(
        model_name=CALIBRATION_MODEL,
        backend="openai_compat",
        config={
            "model_id": CALIBRATION_MODEL_ID,
            "max_top_logprobs": 1024,
        },
    )


def _scoring_record(source_hash=987654321, evidence_text=RECORD["evidence_text"]):
    return SimpleNamespace(
        subject=RECORD["subject"],
        object=RECORD["object"],
        stmt_type=RECORD["stmt_type"],
        evidence_text=evidence_text,
        source_hash=source_hash,
    )


def test_canonical_monolithic_score_replaces_the_only_numeric_field(monkeypatch):
    from indra_belief.probes import calibration
    from indra_belief.probes.calibration import CalibratedProbeReading
    from indra_belief.scorers.monolithic import scorer

    monkeypatch.setattr(
        scorer,
        "_score_categorical",
        lambda *args, **kwargs: {
            "score": 0.95,
            "verdict": "correct",
            "confidence": "high",
        },
    )
    reads: list[dict] = []
    monkeypatch.setattr(
        calibration,
        "read_probe",
        lambda record, client: reads.append(dict(record))
        or calibration.ProbeReading(p_raw=0.75, delta_logit=1.0),
    )
    ids: list[str] = []
    monkeypatch.setattr(
        calibration,
        "calibrate_probe",
        lambda reading, *, record_id, calibration=None: ids.append(record_id)
        or CalibratedProbeReading(p_hat=0.617, weight_of_evidence=0.4),
    )

    result = scorer.score(_calibrated_client(), _scoring_record())

    assert result["score"] == 0.617
    assert result["score_error"] is None
    assert result["verdict"] == "correct"
    assert "calibrated_probability" not in result
    assert ids == ["f987654321"]
    assert reads == [RECORD]


def test_canonical_monolithic_score_is_none_when_profile_is_unavailable(monkeypatch):
    from indra_belief.scorers.monolithic import scorer

    monkeypatch.setattr(
        scorer,
        "_score_categorical",
        lambda *args, **kwargs: {"score": 0.05, "verdict": "incorrect"},
    )

    result = scorer.score(SimpleNamespace(config={}), _scoring_record())

    assert result["score"] is None
    assert result["score_error"] is None
    assert result["verdict"] == "incorrect"


def test_canonical_monolithic_score_obeys_fitted_row_leakage_guard(monkeypatch):
    from indra_belief.probes import calibration
    from indra_belief.scorers.monolithic import scorer

    monkeypatch.setattr(
        scorer,
        "_score_categorical",
        lambda *args, **kwargs: {"score": None, "verdict": "correct"},
    )
    monkeypatch.setattr(
        calibration,
        "read_probe",
        lambda record, client: calibration.ProbeReading(
            p_raw=0.75,
            delta_logit=1.0,
        ),
    )
    fit_id = next(iter(calibration.load_calibration().fit_record_ids))

    result = scorer.score(
        _calibrated_client(),
        _scoring_record(source_hash=int(fit_id[1:])),
    )

    assert result["score"] is None
    assert result["score_error"].startswith("InSampleError:")


def test_empty_sentence_never_runs_the_calibrated_probe(monkeypatch):
    from indra_belief.probes import calibration
    from indra_belief.scorers.monolithic import scorer

    monkeypatch.setattr(
        scorer,
        "_score_categorical",
        lambda *args, **kwargs: {"score": None, "verdict": "correct"},
    )
    monkeypatch.setattr(
        calibration,
        "read_probe",
        lambda *args, **kwargs: pytest.fail("empty sentences cannot be probed"),
    )

    result = scorer.score(_calibrated_client(), _scoring_record(evidence_text=""))

    assert result["score"] is None
    assert result["score_error"] is None


def test_calibration_loader_rejects_direct_probe_content_drift(monkeypatch):
    from indra_belief.probes import calibration

    monkeypatch.setattr(calibration, "probe_digest", lambda probe_id: "0" * 64)

    assert not calibration.supports_sentence_calibration(_calibrated_client())
    with pytest.raises(ValueError, match="does not match the fitted calibration"):
        calibration.load_calibration()
