import json
from dataclasses import FrozenInstanceError, fields

import numpy as np
import pytest
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from indra_belief.probe_combiner import (
    LOGIT_EPS,
    NEUTRAL_LOGIT,
    FrozenCombiner,
    InSampleError,
    fit_combiner,
    to_logit,
)


PROBE_IDS = ("signal_plus_noise", "nuisance_1", "nuisance_2", "nuisance_3")


def _ids(prefix, n):
    return tuple(f"{prefix}-{index:04d}" for index in range(n))


def _denoising_problem():
    rng = np.random.default_rng(0)

    def make_block(n):
        latent = rng.normal(size=n)
        nuisance = rng.normal(size=(n, 3))
        logits = np.column_stack(
            (latent + nuisance.sum(axis=1), nuisance)
        )
        return logits, latent > 0.0

    return (*make_block(160), *make_block(200))


def _fit_denoising_combiner():
    X_fit, y_fit, X_holdout, y_holdout = _denoising_problem()
    combiner = fit_combiner(
        X_fit,
        y_fit,
        probe_ids=PROBE_IDS,
        record_ids=_ids("fit", len(X_fit)),
    )
    return combiner, X_fit, y_fit, X_holdout, y_holdout


def test_combined_raw_logit_outranks_every_single_column_on_holdout():
    combiner, _, _, X_holdout, y_holdout = _fit_denoising_combiner()

    raw = combiner.raw_logit(
        X_holdout,
        record_ids=_ids("hold", len(X_holdout)),
        probe_ids=PROBE_IDS,
    )
    combined_auc = roc_auc_score(y_holdout, raw)
    best_single_auc = max(
        max(
            roc_auc_score(y_holdout, X_holdout[:, column]),
            roc_auc_score(y_holdout, -X_holdout[:, column]),
        )
        for column in range(X_holdout.shape[1])
    )

    assert combined_auc > best_single_auc


def test_score_is_bounded_and_monotone_in_raw_logit():
    combiner, _, _, X_holdout, _ = _fit_denoising_combiner()

    apply_kwargs = {
        "record_ids": _ids("hold", len(X_holdout)),
        "probe_ids": PROBE_IDS,
    }
    raw = combiner.raw_logit(X_holdout, **apply_kwargs)
    score = combiner.score(X_holdout, **apply_kwargs)
    order = np.argsort(raw)

    assert np.all((score >= 0.0) & (score <= 1.0))
    assert np.all(np.diff(score[order]) >= -1e-12)


def test_numpy_interpolation_exactly_matches_isotonic_including_tails():
    combiner, _, _, _, _ = _fit_denoising_combiner()
    targets = np.r_[
        combiner.iso_x[0] - 1.0,
        np.linspace(combiner.iso_x[0], combiner.iso_x[-1], 257),
        combiner.iso_x[-1] + 1.0,
    ]
    strongest = int(np.argmax(np.abs(combiner.coef)))
    X_probe = np.zeros((targets.size, len(combiner.coef)))
    X_probe[:, strongest] = (
        targets - combiner.intercept
    ) / combiner.coef[strongest]

    apply_kwargs = {
        "record_ids": _ids("probe", len(X_probe)),
        "probe_ids": PROBE_IDS,
    }
    raw_probe = combiner.raw_logit(X_probe, **apply_kwargs)
    assert raw_probe[0] < combiner.iso_x[0]
    assert raw_probe[-1] > combiner.iso_x[-1]

    reference = IsotonicRegression(out_of_bounds="clip").fit(
        np.asarray(combiner.iso_x),
        np.asarray(combiner.iso_y),
    ).predict(raw_probe)
    actual = combiner.score(X_probe, **apply_kwargs)

    np.testing.assert_array_equal(actual, reference)
    assert np.max(np.abs(actual - reference)) == 0.0


def test_both_apply_axes_refuse_fit_rows_and_score_disjoint_rows():
    combiner, X_fit, _, X_holdout, _ = _fit_denoising_combiner()
    fit_ids = _ids("fit", len(X_fit))
    hold_ids = _ids("hold", len(X_holdout))
    message = rf"{len(X_fit)} in-sample row\(s\).*index is 0"

    with pytest.raises(InSampleError, match=message):
        combiner.raw_logit(
            X_fit, record_ids=fit_ids, probe_ids=PROBE_IDS
        )
    with pytest.raises(InSampleError, match=message):
        combiner.score(X_fit, record_ids=fit_ids, probe_ids=PROBE_IDS)

    mixed = np.vstack((X_holdout[0], X_fit[7], X_holdout[1], X_fit[12]))
    mixed_ids = (hold_ids[0], fit_ids[7], hold_ids[1], fit_ids[12])
    with pytest.raises(
        InSampleError, match=r"2 in-sample row\(s\).*index is 1"
    ):
        combiner.score(
            mixed, record_ids=mixed_ids, probe_ids=PROBE_IDS
        )

    k = 3
    batch = X_holdout[:7]
    batch_ids = list(_ids("batch", len(batch)))
    batch_ids[k] = fit_ids[19]
    planted_message = rf"1 in-sample row\(s\).*index is {k}"
    for apply in (combiner.raw_logit, combiner.score):
        with pytest.raises(InSampleError, match=planted_message):
            apply(
                batch,
                record_ids=batch_ids,
                probe_ids=PROBE_IDS,
            )

    assert combiner.raw_logit(
        X_holdout, record_ids=hold_ids, probe_ids=PROBE_IDS
    ).shape == (len(X_holdout),)
    assert combiner.score(
        X_holdout, record_ids=hold_ids, probe_ids=PROBE_IDS
    ).shape == (len(X_holdout),)


def test_identical_fit_values_are_allowed_under_fresh_record_ids():
    combiner, X_fit, _, _, _ = _fit_denoising_combiner()
    apply_kwargs = {
        "record_ids": _ids("fresh", len(X_fit)),
        "probe_ids": PROBE_IDS,
    }

    raw = combiner.raw_logit(X_fit.copy(), **apply_kwargs)
    score = combiner.score(X_fit.copy(), **apply_kwargs)

    assert raw.shape == (len(X_fit),)
    assert score.shape == (len(X_fit),)
    assert np.all(np.isfinite(raw))
    assert np.all(np.isfinite(score))


def test_fit_is_deterministic_for_fixed_inputs_and_seed():
    _, X_fit, y_fit, X_holdout, _ = _fit_denoising_combiner()
    fit_kwargs = {
        "probe_ids": PROBE_IDS,
        "record_ids": _ids("fit", len(X_fit)),
        "seed": 17,
    }
    first = fit_combiner(X_fit, y_fit, **fit_kwargs)
    second = fit_combiner(X_fit, y_fit, **fit_kwargs)
    apply_kwargs = {
        "record_ids": _ids("hold", len(X_holdout)),
        "probe_ids": PROBE_IDS,
    }

    assert first.fit_fingerprint == second.fit_fingerprint
    assert first.coef == second.coef
    np.testing.assert_array_equal(
        first.raw_logit(X_holdout, **apply_kwargs),
        second.raw_logit(X_holdout, **apply_kwargs),
    )
    np.testing.assert_array_equal(
        first.score(X_holdout, **apply_kwargs),
        second.score(X_holdout, **apply_kwargs),
    )


def test_json_round_trip_preserves_outputs_provenance_and_guard():
    combiner, X_fit, _, X_holdout, _ = _fit_denoising_combiner()
    payload = json.loads(json.dumps(combiner.to_dict(), allow_nan=False))
    restored = FrozenCombiner.from_dict(payload)
    fit_kwargs = {
        "record_ids": _ids("fit", len(X_fit)),
        "probe_ids": PROBE_IDS,
    }
    hold_kwargs = {
        "record_ids": _ids("hold", len(X_holdout)),
        "probe_ids": PROBE_IDS,
    }

    assert payload["version"] == 2
    assert payload["fit_record_ids"] == sorted(payload["fit_record_ids"])
    assert restored.fit_fingerprint == combiner.fit_fingerprint
    np.testing.assert_array_equal(
        restored.raw_logit(X_holdout, **hold_kwargs),
        combiner.raw_logit(X_holdout, **hold_kwargs),
    )
    np.testing.assert_array_equal(
        restored.score(X_holdout, **hold_kwargs),
        combiner.score(X_holdout, **hold_kwargs),
    )
    with pytest.raises(InSampleError):
        restored.raw_logit(X_fit, **fit_kwargs)
    with pytest.raises(InSampleError):
        restored.score(X_fit, **fit_kwargs)

    legacy_guard_key = "_".join(("fit", "row", "digests"))
    old_shape = json.loads(json.dumps(payload))
    old_shape[legacy_guard_key] = old_shape.pop("fit_record_ids")
    old_shape["version"] = 1
    with pytest.raises(ValueError, match="version"):
        FrozenCombiner.from_dict(old_shape)

    old_shape["version"] = 2
    with pytest.raises(ValueError, match="keys invalid"):
        FrozenCombiner.from_dict(old_shape)


def test_to_logit_clips_endpoints_and_validates_before_clipping():
    lower = to_logit(0.0).item()
    upper = to_logit(1.0).item()

    assert NEUTRAL_LOGIT == 0.0
    assert LOGIT_EPS == 1e-6
    assert np.isfinite(lower)
    assert np.isfinite(upper)
    assert np.isclose(lower, -upper)
    np.testing.assert_allclose(
        to_logit([0.25, 0.5, 0.75]),
        [-np.log(3.0), NEUTRAL_LOGIT, np.log(3.0)],
    )

    for invalid in (1.5, -0.1, np.nan, np.inf, "not-a-probability"):
        with pytest.raises(ValueError, match="p"):
            to_logit(invalid)
    for invalid_eps in (0.0, 0.5, np.nan):
        with pytest.raises(ValueError, match="eps"):
            to_logit(0.5, eps=invalid_eps)


def test_fit_combiner_rejects_each_required_invalid_input():
    rng = np.random.default_rng(11)
    X = rng.normal(size=(12, 2))
    y = np.asarray([False, True] * 6)

    nonfinite_X = X.copy()
    nonfinite_X[0, 0] = np.nan
    nonfinite_y = y.astype(float)
    nonfinite_y[0] = np.nan
    one_class = np.zeros(12, dtype=bool)
    too_small = np.asarray([False] * 10 + [True] * 2)

    cases = [
        (nonfinite_X, y, ("a", "b"), {}, "logits"),
        (X, nonfinite_y, ("a", "b"), {}, "labels"),
        (X, y, ("a",), {}, "probe_ids"),
        (X, y, ("same", "same"), {}, "probe_ids"),
        (X, one_class, ("a", "b"), {}, "labels"),
        (X, too_small, ("a", "b"), {"n_splits": 3}, "n_splits"),
    ]
    for bad_X, bad_y, probe_ids, kwargs, message in cases:
        with pytest.raises(ValueError, match=message):
            fit_combiner(
                bad_X,
                bad_y,
                probe_ids=probe_ids,
                record_ids=_ids("fit", len(bad_X)),
                **kwargs,
            )


def test_fit_combiner_rejects_malformed_shapes_and_parameters():
    rng = np.random.default_rng(12)
    X = rng.normal(size=(12, 2))
    y = np.asarray([False, True] * 6)

    cases = [
        (
            np.full((12, 2), "not-a-logit", dtype=object),
            y,
            ("a", "b"),
            {},
            "logits",
        ),
        (X.ravel(), y, ("a", "b"), {}, "logits"),
        (X, ["not-a-label"] * 12, ("a", "b"), {}, "labels"),
        (X, y[:, None], ("a", "b"), {}, "labels"),
        (X[:-1], y, ("a", "b"), {}, "row counts"),
        (np.empty((12, 0)), y, (), {}, "at least one feature"),
        (X, y, "ab", {}, "probe_ids"),
        (X, y, None, {}, "probe_ids"),
        (X, y, ("a", 2), {}, "probe_ids"),
        (X, y, ("a", "b"), {"n_splits": 1}, "n_splits"),
        (X, y, ("a", "b"), {"n_splits": True}, "n_splits"),
        (X, y, ("a", "b"), {"n_splits": 2.5}, "n_splits"),
        (X, y, ("a", "b"), {"C": 0.0}, "C"),
        (X, y, ("a", "b"), {"C": True}, "C"),
        (X, y, ("a", "b"), {"C": "not-C"}, "C"),
        (X, y, ("a", "b"), {"C": np.nan}, "C"),
        (X, y, ("a", "b"), {"seed": -1}, "seed"),
        (X, y, ("a", "b"), {"seed": True}, "seed"),
        (X, y, ("a", "b"), {"seed": 1.5}, "seed"),
        (X, y, ("a", "b"), {"seed": 2**32}, "seed"),
    ]
    for bad_X, bad_y, probe_ids, kwargs, message in cases:
        with pytest.raises(ValueError, match=message):
            fit_combiner(
                bad_X,
                bad_y,
                probe_ids=probe_ids,
                record_ids=_ids("fit", len(bad_X)),
                **kwargs,
            )


def test_apply_validates_shape_feature_count_finiteness_and_column_order():
    combiner, _, _, X_holdout, _ = _fit_denoising_combiner()
    hold_ids = _ids("hold", len(X_holdout))

    with pytest.raises(ValueError, match="X"):
        combiner.score(
            X_holdout[0], record_ids=hold_ids[:1], probe_ids=PROBE_IDS
        )
    with pytest.raises(ValueError, match="X"):
        combiner.raw_logit(
            X_holdout[0], record_ids=hold_ids[:1], probe_ids=PROBE_IDS
        )
    with pytest.raises(ValueError, match="feature count"):
        combiner.raw_logit(
            X_holdout[:, :-1], record_ids=hold_ids, probe_ids=PROBE_IDS
        )
    nonfinite = X_holdout.copy()
    nonfinite[0, 0] = np.inf
    with pytest.raises(ValueError, match="X"):
        combiner.score(
            nonfinite, record_ids=hold_ids, probe_ids=PROBE_IDS
        )
    with pytest.raises(ValueError, match="X"):
        combiner.raw_logit(
            nonfinite, record_ids=hold_ids, probe_ids=PROBE_IDS
        )

    rows = X_holdout[:9]
    row_ids = _ids("order-check", len(rows))
    assert np.all(
        np.isfinite(
            combiner.raw_logit(
                rows, record_ids=row_ids, probe_ids=PROBE_IDS
            )
        )
    )
    assert np.all(
        np.isfinite(
            combiner.score(rows, record_ids=row_ids, probe_ids=PROBE_IDS)
        )
    )

    permutation = (1, 0, 2, 3)
    permuted_probe_ids = tuple(PROBE_IDS[index] for index in permutation)
    for apply in (combiner.raw_logit, combiner.score):
        with pytest.raises(ValueError, match="column order"):
            apply(
                rows[:, permutation],
                record_ids=row_ids,
                probe_ids=permuted_probe_ids,
            )


def test_required_id_arguments_cannot_be_omitted():
    combiner, X_fit, y_fit, X_holdout, _ = _fit_denoising_combiner()
    hold_ids = _ids("hold", len(X_holdout))

    for apply in (combiner.raw_logit, combiner.score):
        with pytest.raises(TypeError):
            apply(X_holdout, probe_ids=PROBE_IDS)
        with pytest.raises(TypeError):
            apply(X_holdout, record_ids=hold_ids)

    with pytest.raises(TypeError):
        fit_combiner(X_fit, y_fit, probe_ids=PROBE_IDS)


def test_record_id_validation_is_enforced_at_fit_and_apply():
    rng = np.random.default_rng(20260810)
    X = rng.normal(size=(12, 2))
    y = np.asarray([False, True] * 6)
    probe_ids = ("a", "b")
    valid_fit_ids = _ids("fit", len(X))
    combiner = fit_combiner(
        X,
        y,
        probe_ids=probe_ids,
        record_ids=valid_fit_ids,
    )

    non_string = list(_ids("fresh", len(X)))
    non_string[4] = 7
    empty = list(_ids("fresh", len(X)))
    empty[4] = ""
    invalid_record_ids = (
        "bare-record-id",
        ("duplicate",) * len(X),
        _ids("short", len(X) - 1),
        tuple(non_string),
        tuple(empty),
    )

    for bad_record_ids in invalid_record_ids:
        with pytest.raises(ValueError, match="record_ids"):
            fit_combiner(
                X,
                y,
                probe_ids=probe_ids,
                record_ids=bad_record_ids,
            )
        for apply in (combiner.raw_logit, combiner.score):
            with pytest.raises(ValueError, match="record_ids"):
                apply(
                    X,
                    record_ids=bad_record_ids,
                    probe_ids=probe_ids,
                )


def test_from_dict_rejects_invalid_calibration_payloads():
    combiner, _, _, _, _ = _fit_denoising_combiner()
    payload = combiner.to_dict()

    def clone():
        return json.loads(json.dumps(payload))

    invalid = []

    bad = clone()
    bad["version"] = 1
    invalid.append((bad, "version"))

    bad = clone()
    bad["iso_x"][1] = bad["iso_x"][0]
    invalid.append((bad, "iso_x"))

    bad = clone()
    bad["iso_y"] = [1.0] * len(bad["iso_y"])
    bad["iso_y"][-1] = 0.0
    invalid.append((bad, "iso_y"))

    bad = clone()
    bad["iso_y"].pop()
    invalid.append((bad, "iso_x and iso_y lengths"))

    bad = clone()
    bad["iso_x"] = []
    bad["iso_y"] = []
    invalid.append((bad, "must not be empty"))

    bad = clone()
    bad["coef"].pop()
    invalid.append((bad, "coef length.*probe_ids"))

    bad = clone()
    bad["fit_record_ids"] = []
    invalid.append((bad, "fit_record_ids.*non-empty"))

    bad = clone()
    bad["fit_record_ids"][0] = ""
    invalid.append((bad, "fit_record_ids.*non-empty strings"))

    bad = clone()
    bad["fit_record_ids"][0] = 1
    invalid.append((bad, "fit_record_ids.*non-empty strings"))

    bad = clone()
    bad["fit_record_ids"][1] = bad["fit_record_ids"][0]
    invalid.append((bad, "fit_record_ids.*duplicates"))

    bad = clone()
    bad["fit_record_ids"] = list(_ids("serialized", bad["n_fit"] + 1))
    invalid.append((bad, "fit_record_ids.*exceed n_fit"))

    for bad_payload, message in invalid:
        with pytest.raises(ValueError, match=message):
            FrozenCombiner.from_dict(bad_payload)


def test_cross_fit_knots_differ_from_in_sample_isotonic_knots():
    combiner, X_fit, y_fit, _, _ = _fit_denoising_combiner()
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    oof_decisions = np.empty(len(X_fit))
    for train_indices, held_out_indices in splitter.split(X_fit, y_fit):
        fold_model = LogisticRegression(C=1.0, max_iter=1000, random_state=0)
        fold_model.fit(X_fit[train_indices], y_fit[train_indices])
        oof_decisions[held_out_indices] = fold_model.decision_function(
            X_fit[held_out_indices]
        )
    expected_cross_fit = IsotonicRegression(
        out_of_bounds="clip",
        y_min=0.0,
        y_max=1.0,
    ).fit(oof_decisions, y_fit)

    np.testing.assert_array_equal(combiner.iso_x, expected_cross_fit.X_thresholds_)
    np.testing.assert_array_equal(combiner.iso_y, expected_cross_fit.y_thresholds_)

    full_data_decisions = (
        X_fit @ np.asarray(combiner.coef) + combiner.intercept
    )
    in_sample = IsotonicRegression(
        out_of_bounds="clip",
        y_min=0.0,
        y_max=1.0,
    ).fit(full_data_decisions, y_fit)

    stored_x = np.asarray(combiner.iso_x)
    stored_y = np.asarray(combiner.iso_y)
    same_knots = (
        stored_x.shape == in_sample.X_thresholds_.shape
        and stored_y.shape == in_sample.y_thresholds_.shape
        and np.allclose(stored_x, in_sample.X_thresholds_, rtol=0.0, atol=1e-12)
        and np.allclose(stored_y, in_sample.y_thresholds_, rtol=0.0, atol=1e-12)
    )

    assert not same_knots


def test_cross_fitted_calibration_does_not_separate_pinned_pure_noise():
    rng = np.random.default_rng(20260809)
    X_fit = rng.normal(size=(400, 4))
    y_fit = rng.integers(0, 2, size=400).astype(bool)
    X_holdout = rng.normal(size=(1000, 4))
    y_holdout = rng.integers(0, 2, size=1000).astype(bool)
    combiner = fit_combiner(
        X_fit,
        y_fit,
        probe_ids=PROBE_IDS,
        record_ids=_ids("fit", len(X_fit)),
    )

    held_out_auc = roc_auc_score(
        y_holdout,
        combiner.score(
            X_holdout,
            record_ids=_ids("hold", len(X_holdout)),
            probe_ids=PROBE_IDS,
        ),
    )

    assert abs(held_out_auc - 0.5) < 0.10


def test_fit_record_id_remains_guarded_when_feature_values_change():
    rng = np.random.default_rng(77)
    X_fit = rng.normal(size=(40, 2))
    y_fit = np.asarray([False, True] * 20)
    fit_ids = _ids("fit", len(X_fit))
    probe_ids = ("a", "b")
    combiner = fit_combiner(
        X_fit,
        y_fit,
        probe_ids=probe_ids,
        record_ids=fit_ids,
        n_splits=4,
    )

    changed_row = X_fit[[0]].copy() + 100.0
    assert not np.array_equal(changed_row, X_fit[[0]])

    for apply in (combiner.raw_logit, combiner.score):
        with pytest.raises(
            InSampleError, match=r"1 in-sample row\(s\).*index is 0"
        ):
            apply(
                changed_row,
                record_ids=(fit_ids[0],),
                probe_ids=probe_ids,
            )


def test_frozen_combiner_contains_only_declared_plain_fields_and_has_no_fit():
    combiner, X_fit, y_fit, _, _ = _fit_denoising_combiner()

    assert tuple(field.name for field in fields(combiner)) == (
        "probe_ids",
        "coef",
        "intercept",
        "iso_x",
        "iso_y",
        "fit_record_ids",
        "fit_fingerprint",
        "n_fit",
        "fit_prevalence",
    )
    assert not hasattr(combiner, "fit")
    assert isinstance(combiner.probe_ids, tuple)
    assert isinstance(combiner.coef, tuple)
    assert isinstance(combiner.intercept, float)
    assert isinstance(combiner.iso_x, tuple)
    assert isinstance(combiner.iso_y, tuple)
    assert isinstance(combiner.fit_record_ids, frozenset)
    assert isinstance(combiner.fit_fingerprint, str)
    assert isinstance(combiner.n_fit, int)
    assert isinstance(combiner.fit_prevalence, float)
    assert combiner.n_fit == len(X_fit)
    assert combiner.fit_prevalence == float(y_fit.mean())
    assert issubclass(InSampleError, ValueError)
    with pytest.raises(FrozenInstanceError):
        combiner.intercept = 0.0
