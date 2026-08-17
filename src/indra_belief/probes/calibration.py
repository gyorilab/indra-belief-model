"""Sentence-grain calibration for the direct verdict probe.

``ProbeReading.delta_logit`` is a log-odds measurement, not a probability.
This module is the apply boundary for the fitted mapping from that measurement
to ``p_hat = P(the reading is correct)``.  The persisted model is the existing
:class:`indra_belief.probe_combiner.FrozenCombiner` with one feature; no second
isotonic implementation lives here.

The calibration was fitted at the sentence/evidence grain.  It must not be
used as a statement-belief update.  Consumers that need additive evidence can
use ``weight_of_evidence`` — how far this one read moves the belief, in
log-odds relative to the fit-set base rate::

    weight_of_evidence = logit(p_hat) - logit(base_rate)

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


# Measured on the fit corpus: the LOSING label lands at rank 42/83/168 of the
# top-k window. A client that declares less headroom than this can physically
# issue the probe but will lose a label on a large fraction of rows, so it is
# not a production reading client. `read_probe` keeps its own mechanical floor
# of 2 and raises ProbeTopKError per row, which is what a FITTING run wants.
MIN_PROBE_TOP_LOGPROBS = 256

# Serving identity -> fitted artifact. The key includes the SERVED model id, not
# just the registry name, because delta_logit magnitudes are substrate-specific:
# the same weights read in-process and over HTTP correlate at r=0.955 but differ
# 2.4x in range and disagree in sign on 10% of rows. An isotonic map fitted on
# one serving stack is therefore not valid on another, exactly as a reader
# confusion profile is not valid across prompts.
#
# To add a substrate: read raw delta_logits on it (which
# `probe_reading_supported` now permits without a calibration), fit an isotonic,
# ship the artifact, and add one row here. Adding a row is the whole change.
_SENTENCE_CALIBRATIONS: dict[tuple[str, str], str] = {
    (CALIBRATION_MODEL, CALIBRATION_MODEL_ID): CALIBRATION_FILENAME,
}


def probe_reading_supported(client) -> bool:
    """Whether ``client`` can produce a ``delta_logit`` at all.

    A CAPABILITY question, deliberately separate from whether a calibration
    exists for this client. Fusing the two made the remedy unreachable: fitting
    a calibration for a new serving stack requires reading raw delta_logits on
    that stack, which an identity-pinned gate forbids.
    """

    config = getattr(client, "config", None)
    if not isinstance(config, Mapping):
        return False
    top_k = config.get("max_top_logprobs")
    return (
        getattr(client, "_guard", None) is None
        and getattr(client, "backend", "openai_compat") == "openai_compat"
        and isinstance(top_k, int)
        and not isinstance(top_k, bool)
        and top_k >= MIN_PROBE_TOP_LOGPROBS
    )


def sentence_calibration_path_for(client) -> Path | None:
    """The fitted artifact for this client's exact serving identity, or None."""

    config = getattr(client, "config", None)
    if not isinstance(config, Mapping):
        return None
    key = (getattr(client, "model_name", None), config.get("model_id"))
    filename = _SENTENCE_CALIBRATIONS.get(key)  # type: ignore[arg-type]
    if filename is None:
        return None
    return DEFAULT_CALIBRATION_PATH.parent / filename


def supports_sentence_calibration(client) -> bool:
    """Whether ``client`` can be read AND has a calibration fitted for it.

    The production gate: both halves must hold before a calibrated probability
    is emitted. An uncalibrated but capable client reads `False` here and still
    reads `True` from :func:`probe_reading_supported`.
    """

    if not probe_reading_supported(client):
        return False
    if sentence_calibration_path_for(client) is None:
        return False
    try:
        _validate_probe_profile()
    except ValueError:
        return False
    return True


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
    # Resolve the artifact for THIS client's serving identity rather than always
    # the shipped default, so a second registered substrate uses its own fitted
    # map instead of silently borrowing another stack's.
    artifact = sentence_calibration_path_for(client)
    if artifact is None:
        return None
    reading = read_probe(record, client)
    return calibrate_probe(
        reading, record_id=record_id, calibration=_calibration_at(artifact)
    )


def replace_sentence_score(
    result: Mapping[str, object],
    record: Mapping[str, object],
    client,
    *,
    record_id: str | None,
    enabled: bool | None = None,
) -> dict[str, object]:
    """Replace the sole sentence score with calibrated ``p_hat`` or ``None``.

    Also persists ``weight_of_evidence`` — the same reading as an additive
    log-odds weight, the form ``statement_belief(probe_weights=True)`` consumes.
    Without it that flag was a silent no-op on every real run: the belief path
    looked for a field the scorer never wrote and fell back to verdict weights
    for every row.

    Categorical output remains available when the independent probe or
    calibration fails.  There is intentionally no verdict/confidence fallback:
    absence of a calibrated probability has exactly one representation, and both
    fields go to ``None`` together because they come from one reading.
    """

    enriched = dict(result)
    enriched["score"] = None
    enriched["score_error"] = None
    # The same reading in additive form. PERSISTED rather than re-derived,
    # although `weight_of_evidence(score, fit_prevalence)` would reproduce it
    # exactly: the anchor is a property of the artifact that produced the score,
    # and only this function knows which artifact that was. Deriving downstream
    # would mean re-resolving the calibration per row to recover an anchor we
    # are holding right here. One float, written once, removes that lookup and
    # the ambiguity with it.
    enriched["weight_of_evidence"] = None
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
            enriched["weight_of_evidence"] = calibrated.weight_of_evidence
    except Exception as exc:
        enriched["score_error"] = f"{type(exc).__name__}: {exc}"
    return enriched


class CalibratedProbeReading(NamedTuple):
    """A calibrated correctness probability and its weight of evidence."""

    p_hat: float
    weight_of_evidence: float


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


@lru_cache(maxsize=8)
def _calibration_at(path: Path) -> FrozenCombiner:
    """Load and cache one fitted artifact per serving identity.

    Keyed by path rather than a single global slot, because more than one
    substrate can be registered at once and each has its own isotonic map.
    """

    return load_calibration(path)


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
    weight = weight_of_evidence(p_hat, model.fit_prevalence)
    return CalibratedProbeReading(p_hat=p_hat, weight_of_evidence=weight)


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
