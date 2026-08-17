"""No-reasoning probe declarations, distinct from ``scorers.probes``.

``indra_belief.scorers.probes`` is the pre-existing 11-module, four-probe
decision pipeline.  This package instead holds the two-label logit battery and
its serving-callable forced-verdict reader.  Calibration conveniences are
loaded lazily so importing the shared battery declaration remains NumPy-free.
"""

from importlib import import_module

from indra_belief.probes.reader import (
    DIRECT_PROBE_ID,
    ProbeReadError,
    ProbeReading,
    ProbeTopKError,
    read_probe,
)

_CALIBRATION_EXPORTS = (
    "CALIBRATED_PROBE_IDS",
    "CALIBRATION_FILENAME",
    "CALIBRATION_MODEL",
    "CALIBRATION_MODEL_ID",
    "CALIBRATION_PROBE_DIGEST",
    "DEFAULT_CALIBRATION_PATH",
    "CalibratedProbeReading",
    "calibrate_probe",
    "calibrate_reading",
    "calibrated_probabilities",
    "calibrated_probability",
    "load_calibration",
    "replace_sentence_score",
    "supports_sentence_calibration",
    "weight_of_evidence",
)

__all__ = [
    *_CALIBRATION_EXPORTS,
    "DIRECT_PROBE_ID",
    "ProbeReadError",
    "ProbeReading",
    "ProbeTopKError",
    "read_probe",
]


def __getattr__(name: str):
    if name in _CALIBRATION_EXPORTS:
        calibration = import_module("indra_belief.probes.calibration")
        return getattr(calibration, name)
    raise AttributeError(name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
