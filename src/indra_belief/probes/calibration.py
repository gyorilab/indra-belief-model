"""Sentence-grain calibration for the direct verdict probe.

``ProbeReading.delta_logit`` is a log-odds measurement, not a probability.
This module is the apply boundary for the fitted mapping from that measurement
to ``p_hat = P(the reading is correct)``.  The persisted model is the existing
:class:`indra_belief.probe_combiner.FrozenCombiner` with one feature; no second
isotonic implementation lives here.

The calibration was fitted at the sentence/evidence grain.  It must not be
used as a statement-belief update.  Consumers that need additive evidence can
use ``ell``, the calibrated log-odds relative to the fit-set base rate::

    ell = logit(p_hat) - logit(base_rate)

Endpoint probabilities are clipped by the combiner's shared ``to_logit``
policy, because an isotonic model can legitimately emit zero or one.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple

import numpy as np

from indra_belief.probe_combiner import (
    LOGIT_EPS,
    FrozenCombiner,
    to_logit,
)
from indra_belief.probes.battery import probe_digest
from indra_belief.probes.reader import DIRECT_PROBE_ID, ProbeReading, read_probe


CALIBRATION_FILENAME = "sentence_probe_calibration.json"
CALIBRATION_MODEL = "local-gemma-4-26b"
CALIBRATION_MODEL_ID = "mlx-community/gemma-4-26b-a4b-it-8bit"
CALIBRATION_PROBE_DIGEST = (
    "2aa7729f9b4f5e897c6e99baf25956c710c1a36f4f49dfd7f89b4fc747d641ed"
)
SENTENCE_SCORE_CONTRACT_VERSION = 1
SENTENCE_SCORE_KIND = "calibrated_probability_correct"
DEFAULT_CALIBRATION_PATH = (
    Path(__file__).resolve().parents[3]
    / "data"
    / "probe_battery"
    / CALIBRATION_FILENAME
)
CALIBRATED_PROBE_IDS = (DIRECT_PROBE_ID,)


def _validate_probe_profile() -> None:
    current = probe_digest(DIRECT_PROBE_ID)
    if current != CALIBRATION_PROBE_DIGEST:
        raise ValueError(
            "direct sentence probe content does not match the fitted calibration "
            f"profile: expected {CALIBRATION_PROBE_DIGEST}, got {current}"
        )


def supports_sentence_calibration(client) -> bool:
    """Whether ``client`` exactly matches the persisted calibration profile."""

    config = getattr(client, "config", None)
    if not isinstance(config, Mapping):
        return False
    top_k = config.get("max_top_logprobs")
    try:
        _validate_probe_profile()
    except ValueError:
        return False
    return (
        getattr(client, "_guard", None) is None
        and getattr(client, "model_name", None) == CALIBRATION_MODEL
        and getattr(client, "backend", "openai_compat") == "openai_compat"
        and config.get("model_id") == CALIBRATION_MODEL_ID
        and isinstance(top_k, int)
        and not isinstance(top_k, bool)
        and top_k >= 2
    )


def calibrated_sentence_reading(
    record: Mapping[str, object],
    client,
    *,
    record_id: str,
) -> CalibratedProbeReading | None:
    """Read and calibrate one evidence sentence at the scoring boundary."""

    evidence_text = str(record.get("evidence_text") or "")
    if not evidence_text:
        return None
    reading = read_probe(record, client)
    return calibrate_probe(reading, record_id=record_id)


def replace_sentence_score(
    result: Mapping[str, object],
    record: Mapping[str, object],
    client,
    *,
    record_id: str | None,
    enabled: bool | None = None,
) -> dict[str, object]:
    """Replace the sole sentence score with calibrated ``p_hat`` or ``None``.

    Categorical output remains available when the independent probe or
    calibration fails.  There is intentionally no verdict/confidence fallback:
    absence of a calibrated probability has exactly one representation.
    """

    enriched = dict(result)
    enriched["score"] = None
    enriched["score_error"] = None
    available = supports_sentence_calibration(client) if enabled is None else enabled
    if not available:
        return enriched
    if not record_id:
        enriched["score_error"] = (
            "ValueError: calibrated sentence score requires a source-hash identity"
        )
        return enriched
    try:
        calibrated = calibrated_sentence_reading(
            record,
            client,
            record_id=record_id,
        )
        if calibrated is not None:
            enriched["score"] = calibrated.p_hat
    except Exception as exc:
        enriched["score_error"] = f"{type(exc).__name__}: {exc}"
    return enriched


class CalibratedProbeReading(NamedTuple):
    """A calibrated correctness probability and its weight of evidence."""

    p_hat: float
    ell: float

    @property
    def weight_of_evidence(self) -> float:
        """Descriptive alias for ``ell``."""

        return self.ell


def load_calibration(
    path: str | Path = DEFAULT_CALIBRATION_PATH,
) -> FrozenCombiner:
    """Reload and validate the persisted sentence-probe calibration."""

    _validate_probe_profile()
    artifact_path = Path(path)
    with artifact_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    calibration = FrozenCombiner.from_dict(payload)
    if calibration.probe_ids != CALIBRATED_PROBE_IDS:
        raise ValueError(
            "sentence probe calibration must contain exactly "
            f"{CALIBRATED_PROBE_IDS!r}, got {calibration.probe_ids!r}"
        )
    return calibration


@lru_cache(maxsize=1)
def _default_calibration() -> FrozenCombiner:
    """Load the shipped frozen model once per serving process."""

    return load_calibration()


def _calibration_or_default(
    calibration: FrozenCombiner | None,
) -> FrozenCombiner:
    _validate_probe_profile()
    resolved = _default_calibration() if calibration is None else calibration
    if not isinstance(resolved, FrozenCombiner):
        raise TypeError("calibration must be a FrozenCombiner")
    if resolved.probe_ids != CALIBRATED_PROBE_IDS:
        raise ValueError(
            "sentence probe calibration must contain exactly "
            f"{CALIBRATED_PROBE_IDS!r}, got {resolved.probe_ids!r}"
        )
    return resolved


def calibrated_probabilities(
    delta_logits,
    *,
    record_ids,
    calibration: FrozenCombiner | None = None,
) -> np.ndarray:
    """Map a batch of direct-probe log-odds to calibrated probabilities.

    ``record_ids`` is mandatory so the underlying ``FrozenCombiner`` can
    refuse rows used to fit its isotonic map.  The returned values, unlike the
    input ``delta_logits``, are probabilities and are safe to use for ECE or
    Brier scoring.
    """

    try:
        values = np.asarray(delta_logits, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "delta_logits must be coercible to a float vector"
        ) from exc
    if values.ndim != 1:
        raise ValueError("delta_logits must have shape (n,)")

    model = _calibration_or_default(calibration)
    return model.score(
        values.reshape(-1, 1),
        record_ids=record_ids,
        probe_ids=CALIBRATED_PROBE_IDS,
    )


def calibrated_probability(
    delta_logit: float,
    *,
    record_id: str,
    calibration: FrozenCombiner | None = None,
) -> float:
    """Map one direct-probe log-odds measurement to ``p_hat``."""

    scores = calibrated_probabilities(
        [delta_logit],
        record_ids=(record_id,),
        calibration=calibration,
    )
    return float(scores[0])


def weight_of_evidence(
    p_hat,
    base_rate,
    *,
    eps: float = LOGIT_EPS,
):
    """Return ``logit(p_hat) - logit(base_rate)`` with finite endpoints.

    Scalars produce a ``float`` and array-like inputs produce an ``ndarray``.
    Validation and clipping are delegated to the combiner's canonical
    :func:`to_logit`, keeping the evidence convention identical everywhere.
    """

    evidence = np.asarray(
        to_logit(p_hat, eps=eps) - to_logit(base_rate, eps=eps),
        dtype=float,
    )
    if evidence.ndim == 0:
        return float(evidence)
    return evidence


def calibrate_probe(
    reading: ProbeReading,
    *,
    record_id: str,
    calibration: FrozenCombiner | None = None,
) -> CalibratedProbeReading:
    """Calibrate a ``ProbeReading`` and expose its additive evidence form."""

    if not isinstance(reading, ProbeReading):
        raise TypeError("reading must be a ProbeReading")
    model = _calibration_or_default(calibration)
    p_hat = calibrated_probability(
        reading.delta_logit,
        record_id=record_id,
        calibration=model,
    )
    ell = weight_of_evidence(p_hat, model.fit_prevalence)
    return CalibratedProbeReading(p_hat=p_hat, ell=ell)


# ``calibrate_reading`` reads naturally beside ``read_probe`` while the more
# explicit spelling above keeps the domain visible at call sites.
calibrate_reading = calibrate_probe


__all__ = [
    "CALIBRATED_PROBE_IDS",
    "CALIBRATION_FILENAME",
    "CALIBRATION_MODEL",
    "CALIBRATION_MODEL_ID",
    "CALIBRATION_PROBE_DIGEST",
    "SENTENCE_SCORE_CONTRACT_VERSION",
    "SENTENCE_SCORE_KIND",
    "DEFAULT_CALIBRATION_PATH",
    "CalibratedProbeReading",
    "calibrate_probe",
    "calibrate_reading",
    "calibrated_probabilities",
    "calibrated_probability",
    "calibrated_sentence_reading",
    "load_calibration",
    "replace_sentence_score",
    "supports_sentence_calibration",
    "weight_of_evidence",
]
