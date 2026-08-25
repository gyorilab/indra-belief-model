"""Fit and freeze multivariate probe combination and monotone calibration.

This module is a fitted map from each record's vector of probe log-odds to one
scalar, followed by a monotone calibration of that scalar.  ``raw_logit`` is
the rank-preserving axis for AUROC and dAUROC, while ``score`` is the calibrated
axis for ECE and Brier.  Isotonic calibration collapses distinct values, so any
average precision computed on ``score`` must be reported beside a measured
``len(np.unique(...))``; that count is measured separately on every split and
is never assumed from another split.

This two-axis split is load-bearing rather than stylistic.  ``metrics.ece``
scores out-of-unit-interval inputs as perfect: the measured result is
``ece([(5.0, True), (-3.0, False), (9.0, False), (-7.0, True)]) == 0.0``.
Consequently, ``raw_logit`` must never be handed to ``ece()``.  The G0 caution
in ``scripts/calibration_stage0.py`` is also binding: "a monotone post-hoc map
cannot manufacture discrimination -- the lever is upstream".  Here the
discrimination comes from the multivariate logistic stage, which can create an
axis the inputs lack individually; the isotonic stage only calibrates that
axis, which is why the two stages are exposed separately.

This is the k-feature generalisation of
``calibration_constants.py::profile_from_confusion``.  It deliberately does
not import, modify, or replace that shipped production belief calibration: the
2x2-confusion map is closed-form by design.  Nothing here is wired into the
belief math, and this module neither imports nor mutates
``src/indra_belief/noise_model.py``.

The isotonic knots are learned from out-of-fold decisions, whereas
``raw_logit`` uses the full-data logistic refit, so the two sit on slightly
different scales.  This is the standard
``CalibratedClassifierCV(method='isotonic', ensemble=False)`` construction,
named explicitly as the chosen design rather than an accidental mismatch.
For the default lbfgs solver, ``random_state`` is inert; determinism comes from
the deterministic solver and the fixed fold assignment, not from seeding the
logistic regression.

Finally, the in-sample refusal is only as sound as the caller's record-id
scheme: a caller who mints fresh ids for fit rows evades it.  The column check
likewise validates the caller's declared label order, so a caller who permutes
the matrix while still declaring the fit order is undetectable -- the same
limit as sklearn's ``feature_names_in_``.  These checks claim no more than that.
"""

from __future__ import annotations

from dataclasses import dataclass
from operator import index as _index

import numpy as np

from indra_belief.hashing import canonical_sha256


__all__ = [
    "NEUTRAL_LOGIT",
    "LOGIT_EPS",
    "InSampleError",
    "to_logit",
    "fit_combiner",
    "FrozenCombiner",
]


NEUTRAL_LOGIT: float = 0.0  # p_raw = 0.5, "this probe said nothing"
LOGIT_EPS: float = 1e-6  # clip bound; logit(1-1e-6) ~= +13.8155


class InSampleError(ValueError):
    """Raised when an apply call contains a row used to fit the combiner."""


def _float_matrix(value, *, name: str) -> np.ndarray:
    try:
        matrix = np.asarray(value, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be coercible to a float matrix") from exc
    if matrix.ndim != 2:
        raise ValueError(f"{name} must have shape (n, k)")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} contains non-finite values")
    return matrix


def _bool_labels(value) -> np.ndarray:
    try:
        numeric = np.asarray(value, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("labels must be coercible to a boolean vector") from exc
    if numeric.ndim != 1:
        raise ValueError("labels must have shape (n,)")
    if not np.all(np.isfinite(numeric)):
        raise ValueError("labels contains non-finite values")
    return numeric.astype(bool)


def _normalise_id_sequence(value, *, name: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be a sequence of strings")
    try:
        normalised = tuple(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be a sequence of strings") from exc
    if not all(isinstance(item, str) for item in normalised):
        raise ValueError(f"{name} must contain only strings")
    if len(set(normalised)) != len(normalised):
        raise ValueError(f"{name} contains duplicates")
    return normalised


def _normalise_probe_ids(probe_ids) -> tuple[str, ...]:
    return _normalise_id_sequence(probe_ids, name="probe_ids")


def _normalise_record_ids(record_ids, *, n_rows: int) -> tuple[str, ...]:
    normalised = _normalise_id_sequence(record_ids, name="record_ids")
    if any(record_id == "" for record_id in normalised):
        raise ValueError("record_ids must not contain empty strings")
    if len(normalised) != n_rows:
        raise ValueError(
            "record_ids length does not match row count: "
            f"{len(normalised)} != {n_rows}"
        )
    return normalised


def _integer_parameter(value, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer")
    try:
        return int(_index(value))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be an integer") from exc


def to_logit(p, *, eps: float = LOGIT_EPS) -> np.ndarray:
    """Return clipped log-odds for scalar or array-like probabilities."""

    try:
        probabilities = np.asarray(p, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("p must be coercible to finite probabilities") from exc
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("p contains non-finite values")
    if np.any((probabilities < 0.0) | (probabilities > 1.0)):
        raise ValueError("p contains values outside [0, 1]")

    try:
        epsilon = float(eps)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("eps must be a finite number in (0, 0.5)") from exc
    if not np.isfinite(epsilon) or not 0.0 < epsilon < 0.5:
        raise ValueError("eps must be a finite number in (0, 0.5)")

    clipped = np.clip(probabilities, epsilon, 1.0 - epsilon)
    return np.asarray(np.log(clipped / (1.0 - clipped)), dtype=float)


@dataclass(frozen=True)
class FrozenCombiner:
    """A JSON-freezable, apply-only probe combiner containing learned numbers."""

    probe_ids: tuple[str, ...]
    coef: tuple[float, ...]
    intercept: float
    iso_x: tuple[float, ...]
    iso_y: tuple[float, ...]
    fit_record_ids: frozenset[str]
    fit_fingerprint: str
    n_fit: int
    fit_prevalence: float

    def _validated_X(self, X, probe_ids) -> np.ndarray:
        matrix = _float_matrix(X, name="X")
        if matrix.shape[1] != len(self.coef):
            raise ValueError(
                "X feature count does not match coef: "
                f"{matrix.shape[1]} != {len(self.coef)}"
            )
        received_probe_ids = _normalise_id_sequence(probe_ids, name="probe_ids")
        if received_probe_ids != self.probe_ids:
            raise ValueError(
                "X column order does not match probe_ids: "
                f"expected {self.probe_ids!r}, received {received_probe_ids!r}"
            )
        return matrix

    def _check_not_in_sample(self, record_ids: tuple[str, ...]) -> None:
        offending = [
            index
            for index, record_id in enumerate(record_ids)
            if record_id in self.fit_record_ids
        ]
        if offending:
            first = offending[0]
            raise InSampleError(
                f"X contains {len(offending)} in-sample row(s); "
                f"first offending index is {first}; "
                f"first offending record id is {record_ids[first]!r}"
            )

    def _raw_logit_unchecked(self, X: np.ndarray) -> np.ndarray:
        return X @ np.asarray(self.coef, dtype=float) + self.intercept

    def raw_logit(self, X, *, record_ids, probe_ids) -> np.ndarray:
        """Return the uncalibrated, rank-preserving combined axis."""

        matrix = self._validated_X(X, probe_ids)
        ids_of_records = _normalise_record_ids(
            record_ids, n_rows=matrix.shape[0]
        )
        self._check_not_in_sample(ids_of_records)
        return self._raw_logit_unchecked(matrix)

    def score(self, X, *, record_ids, probe_ids) -> np.ndarray:
        """Return the monotone isotonic probability, clipped to ``[0, 1]``."""

        matrix = self._validated_X(X, probe_ids)
        ids_of_records = _normalise_record_ids(
            record_ids, n_rows=matrix.shape[0]
        )
        self._check_not_in_sample(ids_of_records)
        raw = self._raw_logit_unchecked(matrix)
        calibrated = np.interp(
            raw,
            np.asarray(self.iso_x, dtype=float),
            np.asarray(self.iso_y, dtype=float),
            left=self.iso_y[0],
            right=self.iso_y[-1],
        )
        return np.clip(calibrated, 0.0, 1.0)

    def to_dict(self) -> dict:
        """Return a strict-JSON-safe versioned representation."""

        return {
            "version": 2,
            "probe_ids": list(self.probe_ids),
            "coef": list(self.coef),
            "intercept": self.intercept,
            "iso_x": list(self.iso_x),
            "iso_y": list(self.iso_y),
            "fit_record_ids": sorted(self.fit_record_ids),
            "fit_fingerprint": self.fit_fingerprint,
            "n_fit": self.n_fit,
            "fit_prevalence": self.fit_prevalence,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "FrozenCombiner":
        """Validate and reconstruct a frozen combiner at the apply boundary."""

        if not isinstance(d, dict):
            raise ValueError("serialized combiner must be a dict")
        version = d.get("version")
        if type(version) is not int or version != 2:
            raise ValueError(f"version must be 2, got {version!r}")

        expected_keys = {
            "version",
            "probe_ids",
            "coef",
            "intercept",
            "iso_x",
            "iso_y",
            "fit_record_ids",
            "fit_fingerprint",
            "n_fit",
            "fit_prevalence",
        }
        missing = sorted(expected_keys - set(d))
        extra = sorted(set(d) - expected_keys)
        if missing or extra:
            raise ValueError(
                f"serialized combiner keys invalid; missing={missing}, extra={extra}"
            )

        if not isinstance(d["probe_ids"], list):
            raise ValueError("probe_ids must be a JSON list")
        probe_ids = _normalise_probe_ids(d["probe_ids"])
        if not probe_ids:
            raise ValueError("probe_ids must not be empty")

        coef = _validated_float_list(d["coef"], name="coef")
        iso_x = _validated_float_list(d["iso_x"], name="iso_x")
        iso_y = _validated_float_list(d["iso_y"], name="iso_y")
        intercept = _validated_float_scalar(d["intercept"], name="intercept")
        fit_prevalence = _validated_float_scalar(
            d["fit_prevalence"], name="fit_prevalence"
        )

        if len(coef) != len(probe_ids):
            raise ValueError("coef length does not match probe_ids length")
        if not iso_x:
            raise ValueError("iso_x and iso_y knot arrays must not be empty")
        if len(iso_x) != len(iso_y):
            raise ValueError("iso_x and iso_y lengths do not match")
        if any(right <= left for left, right in zip(iso_x, iso_x[1:])):
            raise ValueError("iso_x must be strictly ascending")
        if any(right < left for left, right in zip(iso_y, iso_y[1:])):
            raise ValueError("iso_y must be non-decreasing")
        if any(value < 0.0 or value > 1.0 for value in iso_y):
            raise ValueError("iso_y must stay inside [0, 1]")

        record_ids_value = d["fit_record_ids"]
        if not isinstance(record_ids_value, list) or not record_ids_value:
            raise ValueError("fit_record_ids must be a non-empty JSON list")
        if not all(
            isinstance(record_id, str) and record_id
            for record_id in record_ids_value
        ):
            raise ValueError("fit_record_ids must contain non-empty strings")
        fit_record_ids = frozenset(record_ids_value)
        if len(fit_record_ids) != len(record_ids_value):
            raise ValueError("fit_record_ids must not contain duplicates")

        fit_fingerprint = d["fit_fingerprint"]
        if not _is_sha256(fit_fingerprint):
            raise ValueError("fit_fingerprint must be a lowercase sha256 string")

        n_fit = d["n_fit"]
        if type(n_fit) is not int or n_fit <= 0:
            raise ValueError("n_fit must be a positive integer")
        if len(fit_record_ids) > n_fit:
            raise ValueError("fit_record_ids count cannot exceed n_fit")
        if not 0.0 < fit_prevalence < 1.0:
            raise ValueError("fit_prevalence must be inside (0, 1)")

        return cls(
            probe_ids=probe_ids,
            coef=coef,
            intercept=intercept,
            iso_x=iso_x,
            iso_y=iso_y,
            fit_record_ids=fit_record_ids,
            fit_fingerprint=fit_fingerprint,
            n_fit=n_fit,
            fit_prevalence=fit_prevalence,
        )


def _validated_float_list(value, *, name: str) -> tuple[float, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a JSON list")
    if any(isinstance(item, bool) for item in value):
        raise ValueError(f"{name} must contain finite numbers")
    try:
        normalised = tuple(float(item) for item in value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must contain finite numbers") from exc
    if not all(np.isfinite(item) for item in normalised):
        raise ValueError(f"{name} contains non-finite values")
    return normalised


def _validated_float_scalar(value, *, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    try:
        normalised = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not np.isfinite(normalised):
        raise ValueError(f"{name} must be finite")
    return normalised


def _is_sha256(value) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def fit_combiner(
    logits,
    labels,
    *,
    probe_ids,
    record_ids,
    n_splits=5,
    C=1.0,
    seed=0,
) -> "FrozenCombiner":
    """Cross-fit isotonic calibration, refit logistic weights, and freeze."""

    # Function scope is deliberate: applying a frozen scorer requires numpy only.
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold

    matrix = _float_matrix(logits, name="logits")
    target = _bool_labels(labels)
    ids = _normalise_probe_ids(probe_ids)
    ids_of_records = _normalise_record_ids(
        record_ids, n_rows=matrix.shape[0]
    )

    if matrix.shape[0] != target.shape[0]:
        raise ValueError("logits and labels row counts do not match")
    if matrix.shape[1] != len(ids):
        raise ValueError(
            "logits feature count does not match probe_ids: "
            f"{matrix.shape[1]} != {len(ids)}"
        )
    if matrix.shape[1] == 0:
        raise ValueError("logits and probe_ids must contain at least one feature")

    folds = _integer_parameter(n_splits, name="n_splits")
    if folds < 2:
        raise ValueError("n_splits must be at least 2")

    if isinstance(C, (bool, np.bool_)):
        raise ValueError("C must be a finite positive number")
    try:
        regularisation = float(C)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("C must be a finite positive number") from exc
    if not np.isfinite(regularisation) or regularisation <= 0.0:
        raise ValueError("C must be a finite positive number")

    random_seed = _integer_parameter(seed, name="seed")
    if not 0 <= random_seed < 2**32:
        raise ValueError("seed must be in [0, 2**32)")

    classes, class_counts = np.unique(target, return_counts=True)
    if classes.size != 2:
        raise ValueError("labels must contain both boolean classes")
    minority_count = int(class_counts.min())
    if minority_count < folds:
        raise ValueError(
            "labels smaller-class count must be at least n_splits: "
            f"{minority_count} < {folds}"
        )

    # sklearn.calibration.CalibratedClassifierCV was considered and rejected:
    # it retains fitted sklearn estimators (so cannot be JSON-frozen and
    # round-tripped), exposes no separated rank-preserving axis, and carries no
    # in-sample guard -- the three things this node exists to provide.  The
    # estimator, splitter, and PAVA implementation below still all come from
    # sklearn; hashing likewise stays in indra_belief.hashing.
    splitter = StratifiedKFold(
        n_splits=folds,
        shuffle=True,
        random_state=random_seed,
    )
    oof_decisions = np.empty(matrix.shape[0], dtype=float)
    oof_seen = np.zeros(matrix.shape[0], dtype=np.uint8)
    for train_indices, held_out_indices in splitter.split(matrix, target):
        fold_model = LogisticRegression(
            C=regularisation,
            max_iter=1000,
            random_state=random_seed,
        )
        fold_model.fit(matrix[train_indices], target[train_indices])
        oof_decisions[held_out_indices] = fold_model.decision_function(
            matrix[held_out_indices]
        )
        oof_seen[held_out_indices] += 1
    if not np.all(oof_seen == 1):  # pragma: no cover - sklearn splitter contract
        raise RuntimeError("cross-fit did not produce exactly one decision per row")

    isotonic = IsotonicRegression(
        out_of_bounds="clip",
        y_min=0.0,
        y_max=1.0,
    )
    isotonic.fit(oof_decisions, target)

    full_model = LogisticRegression(
        C=regularisation,
        max_iter=1000,
        random_state=random_seed,
    )
    full_model.fit(matrix, target)

    fit_fingerprint = canonical_sha256(
        {
            "probe_ids": list(ids),
            "record_ids": sorted(ids_of_records),
            "n": int(matrix.shape[0]),
            "n_splits": folds,
            "C": regularisation,
            "seed": random_seed,
        }
    )

    return FrozenCombiner(
        probe_ids=ids,
        coef=tuple(float(value) for value in full_model.coef_[0]),
        intercept=float(full_model.intercept_[0]),
        iso_x=tuple(float(value) for value in isotonic.X_thresholds_),
        iso_y=tuple(float(value) for value in isotonic.y_thresholds_),
        fit_record_ids=frozenset(ids_of_records),
        fit_fingerprint=fit_fingerprint,
        n_fit=int(matrix.shape[0]),
        fit_prevalence=float(target.mean()),
    )
