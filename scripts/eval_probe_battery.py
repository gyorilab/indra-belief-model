"""Held-out evaluation for a candidate probe-battery score.

This evaluator uses ``indra_belief.metrics.auprc``.  The sibling scripts
``compute_deployed_baseline_replication.py`` and
``compute_belief_model_ladder.py`` both carry comments banning that estimator
as order-dependent per the 2026-07-25 finding.  That ban is stale: the fix is
regression-locked by
``tests/test_metrics.py::test_auprc_is_order_invariant`` and
``tests/test_metrics.py::test_auprc_matches_sklearn_average_precision``, and was
re-measured today.  This node deliberately does not edit those two scripts.

Historical incumbent exports may have very low score resolution. This
evaluator reads the persisted incumbent measurement only; it never recreates a
missing value from categorical verdict/confidence.

Timing is a measurement only when a number was recorded.  An absent key, a
JSON ``null``, or a call log in which no call carries ``duration_s`` produces
``None``.  An explicit ``0.0`` is a measurement and stays ``0.0``; this
discriminator prevents missing cost from masquerading as a measured zero.
Whole-run seconds per record is emitted only when every record is timed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from indra_belief.curation import is_gold_correct
from indra_belief.metrics import (
    BINS_8,
    auprc,
    auroc,
    brier_murphy,
    ece,
    reliability_bins,
)
ROOT = Path(__file__).resolve().parents[1]
# NOTE: a module-level `AP_REQUIRES_DISTINCT_COUNT = True` used to sit here. It
# was read by nothing — a symbol shaped like a compliance switch for the
# "AP only beside its distinct count" invariant that enforced no such thing.
# The invariant IS enforced, by tests/test_eval_probe_battery.py's AST lock on
# the single auprc call site plus the assertions that both keys ship together.
# Deleted rather than wired: a decorative guard is worse than none, because it
# answers the question "is this enforced?" with a lie.
_MISSING = object()
_REQUIRED = object()
_OWNED_TOP_LEVEL_KEYS = (
    "schema_version",
    "kind",
    "n",
    "base_rate_correct",
    "candidate",
    "incumbent",
    "paired_bootstrap",
    "cost",
    "gate",
    "verdict",
)


class PairingError(ValueError):
    """The score and gold rows do not form a bijection."""


@dataclass(frozen=True)
class Pairs:
    """Candidate and incumbent observations aligned on the same records."""

    record_ids: list[str]
    cluster_ids: list[str]
    cluster_field: str
    labels: list[bool]
    candidate: list[float]
    incumbent: list[float]
    candidate_seconds: list[float | None]
    incumbent_seconds: list[float | None]
    incumbent_untimed_calls: int = 0


def _resolve_input_path(path: str | Path) -> Path:
    resolved = Path(path)
    return resolved if resolved.is_absolute() else ROOT / resolved


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(
        _resolve_input_path(path).read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"JSONL row {line_number} in {path} is not an object")
        rows.append(row)
    return rows


def _alias_value(
    row: dict[str, Any], canonical: str, alias: str | None = None, *, default=_REQUIRED
):
    if canonical in row:
        return row[canonical]
    if alias is not None and alias in row:
        return row[alias]
    if default is not _REQUIRED:
        return default
    names = canonical if alias is None else f"{canonical} (alias {alias})"
    raise ValueError(f"score row is missing required field {names}")


def _score_record_id(row: dict[str, Any]) -> str:
    if "record_id" in row:
        record_id = row["record_id"]
        if not isinstance(record_id, str):
            raise PairingError("record_id must be a string")
        return record_id
    if "row_index" in row and "source_hash" in row:
        return f"{row['row_index']}:{row['source_hash']}"
    raise PairingError(
        "score row has no record id: record_id_missing=1 "
        "(expected record_id or row_index plus source_hash)"
    )


def _assert_bijection(expected_ids: Sequence[str], score_ids: Sequence[str]) -> None:
    counts = Counter(score_ids)
    duplicate = sum(count - 1 for count in counts.values() if count > 1)
    expected_set = set(expected_ids)
    score_set = set(score_ids)
    missing = expected_set - score_set
    extra = score_set - expected_set
    length_mismatch = int(len(expected_ids) != len(score_ids))
    if missing or extra or duplicate or length_mismatch:
        raise PairingError(
            "pairing failed: "
            f"gold={len(expected_ids)} scores={len(score_ids)} "
            f"missing={len(missing)} extra={len(extra)} duplicate={duplicate} "
            f"length_mismatch={length_mismatch}"
        )


def _candidate_seconds(row: dict[str, Any]) -> float | None:
    value = _alias_value(
        row, "candidate_seconds", "elapsed_s_battery", default=_MISSING
    )
    if value is _MISSING or value is None:
        return None
    return float(value)


def _score_incumbent_seconds(row: dict[str, Any]):
    if "incumbent_seconds" in row:
        return row["incumbent_seconds"]
    if "elapsed_s_incumbent" in row:
        return row["elapsed_s_incumbent"]
    return _MISSING


def _call_log_seconds(
    call_log: Sequence[dict[str, Any]] | None,
) -> tuple[float | None, int]:
    """Return the timed-call lower bound and number of untimed calls."""
    durations: list[float] = []
    untimed_calls = 0
    for call in call_log or ():
        value = call.get("duration_s", _MISSING)
        if value is _MISSING or value is None:
            untimed_calls += 1
            continue
        durations.append(float(value))
    if not durations:
        return None, untimed_calls
    return float(sum(durations)), untimed_calls


def load_pairs(scores_path: str | Path, gold_path: str | Path | None = None) -> Pairs:
    """Load a bijectively paired score artifact in PAIR or SELF mode.

    In PAIR mode, gold-file ordinal is part of the key.  On holdout_cc, 34
    source hashes repeat over 81 of 500 rows and 10 of those groups disagree on
    ``(verdict, confidence)``, so a source-hash join would silently pick one of
    two different incumbent scores.  The ordinal key is injective by
    construction, and no row is ever dropped to make a join work.

    In SELF mode, D1's combined score artifact supplies labels, both scores,
    timings, and the ``row_index`` plus ``source_hash`` ordinal key itself.
    """
    score_rows = _read_jsonl(scores_path)
    if gold_path is None:
        for row_index, row in enumerate(score_rows):
            if "source_hash" not in row:
                raise PairingError(
                    "SELF score row "
                    f"{row_index} is missing required field source_hash"
                )
    score_ids = [_score_record_id(row) for row in score_rows]

    if gold_path is None:
        _assert_bijection(score_ids, score_ids)
        record_ids: list[str] = []
        cluster_ids: list[str] = []
        labels: list[bool] = []
        candidate: list[float] = []
        incumbent: list[float] = []
        candidate_seconds: list[float | None] = []
        incumbent_seconds: list[float | None] = []
        for record_id, row in zip(score_ids, score_rows):
            gold_correct = _alias_value(row, "gold_correct")
            if not isinstance(gold_correct, bool):
                raise ValueError(
                    f"gold_correct for {record_id} must be a JSON boolean"
                )
            record_ids.append(record_id)
            cluster_ids.append(str(row["source_hash"]))
            labels.append(gold_correct)
            candidate.append(
                float(_alias_value(row, "candidate_score", "battery_score"))
            )
            incumbent.append(float(_alias_value(row, "incumbent_score")))
            candidate_seconds.append(_candidate_seconds(row))
            incumbent_value = _score_incumbent_seconds(row)
            incumbent_seconds.append(
                None
                if incumbent_value is _MISSING or incumbent_value is None
                else float(incumbent_value)
            )
        return Pairs(
            record_ids=record_ids,
            cluster_ids=cluster_ids,
            cluster_field="source_hash",
            labels=labels,
            candidate=candidate,
            incumbent=incumbent,
            candidate_seconds=candidate_seconds,
            incumbent_seconds=incumbent_seconds,
        )

    gold_rows = _read_jsonl(gold_path)
    expected_ids = [f"{i}:{row['source_hash']}" for i, row in enumerate(gold_rows)]
    _assert_bijection(expected_ids, score_ids)
    score_by_id = dict(zip(score_ids, score_rows))

    record_ids = []
    cluster_ids = []
    labels = []
    candidate = []
    incumbent = []
    candidate_seconds = []
    incumbent_seconds = []
    incumbent_untimed_calls = 0
    for record_id, gold_row in zip(expected_ids, gold_rows):
        score_row = score_by_id[record_id]
        record_ids.append(record_id)
        cluster_ids.append(str(gold_row["source_hash"]))
        labels.append(is_gold_correct(gold_row.get("tag")))
        candidate.append(
            float(_alias_value(score_row, "candidate_score", "battery_score"))
        )
        candidate_seconds.append(_candidate_seconds(score_row))

        incumbent_score = score_row.get("incumbent_score")
        if incumbent_score is None:
            raise ValueError(
                "incumbent_score is required; missing scores are not derived "
                f"from verdict/confidence (record_id={record_id})"
            )
        incumbent.append(float(incumbent_score))

        incumbent_value = _score_incumbent_seconds(score_row)
        if incumbent_value is _MISSING:
            incumbent_value, untimed_calls = _call_log_seconds(
                gold_row.get("call_log")
            )
            incumbent_untimed_calls += untimed_calls
        incumbent_seconds.append(
            None if incumbent_value is None else float(incumbent_value)
        )

    return Pairs(
        record_ids=record_ids,
        cluster_ids=cluster_ids,
        cluster_field="source_hash",
        labels=labels,
        candidate=candidate,
        incumbent=incumbent,
        candidate_seconds=candidate_seconds,
        incumbent_seconds=incumbent_seconds,
        incumbent_untimed_calls=incumbent_untimed_calls,
    )


def _refuse_single_class(labels: Sequence[bool]) -> None:
    n_pos = int(sum(bool(value) for value in labels))
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError(
            f"AUROC requires both classes: n_pos={n_pos} n_neg={n_neg}"
        )


def _ap_with_distinct(scores: Sequence[float], labels: Sequence[bool]) -> dict:
    """Return AP welded to the number of score thresholds that produced it."""
    return {"auprc": float(auprc(scores, labels)), "distinct_scores": len(set(scores))}


def score_block(
    scores: Sequence[float],
    labels: Sequence[bool],
    *,
    name: str,
    seconds_per_record: float | None,
) -> dict:
    """Compute one arm's metrics, with calibration refused off the unit interval.

    A low-resolution incumbent can flatter a continuous arm in AP comparisons,
    so its distinct-threshold count is produced by the same helper call.

    Re-measured today, ``metrics.BINS_8`` spans [0.0, 1.001), and out-of-range
    rows leave the numerator while remaining in ``n_all``:
    ``ece([(5.0, True), (-3.0, False), (9.0, False), (-7.0, True)])`` returns
    0.0, a perfect-calibration report for a score that is not a probability.
    Calibration metrics are therefore refused outside [0, 1].  AUROC and AP
    remain valid because they are rank statistics.
    """
    score_values = [float(value) for value in scores]
    label_values = [bool(value) for value in labels]
    if len(score_values) != len(label_values):
        raise ValueError(
            f"score/label length mismatch: scores={len(score_values)} "
            f"labels={len(label_values)}"
        )
    if not score_values:
        raise ValueError("cannot score an empty label vector")

    _refuse_single_class(label_values)
    ap = _ap_with_distinct(score_values, label_values)
    in_unit_interval = all(0.0 <= value <= 1.0 for value in score_values)
    if in_unit_interval:
        calibration = brier_murphy(score_values, label_values, bins=BINS_8)
        ece_value = float(ece(zip(score_values, label_values), bins=BINS_8))
        brier = float(calibration["brier"])
        reliability = float(calibration["reliability"])
        resolution = float(calibration["resolution"])
        bins = reliability_bins(score_values, label_values, bins=BINS_8)
    else:
        ece_value = None
        brier = None
        reliability = None
        resolution = None
        bins = None

    return {
        "name": name,
        "n": len(score_values),
        **ap,
        "in_unit_interval": in_unit_interval,
        "auroc": float(auroc(score_values, label_values)),
        "ece": ece_value,
        "brier": brier,
        "reliability": reliability,
        "resolution": resolution,
        "reliability_bins": bins,
        "seconds_per_record": (
            None if seconds_per_record is None else float(seconds_per_record)
        ),
        "base_rate": float(sum(label_values) / len(label_values)),
    }


def paired_bootstrap_delta_auroc(
    labels: Sequence[bool],
    candidate: Sequence[float],
    incumbent: Sequence[float],
    *,
    cluster_ids: Sequence[str],
    cluster_field: str,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict:
    """Clustered percentile CI for paired candidate-minus-incumbent AUROC.

    This local implementation deliberately streams sorted-key cluster draws.
    It is not shared with the frozen correlation analyzer, whose helper
    materializes every draw and whose exact sequence is baked into an existing
    artifact. Keeping the implementations local avoids changing that result.
    """
    y = np.asarray(labels, dtype=bool)
    high = np.asarray(candidate, dtype=float)
    low = np.asarray(incumbent, dtype=float)
    if high.size != y.size or low.size != y.size or len(cluster_ids) != y.size:
        raise ValueError(
            f"bootstrap length mismatch: labels={y.size} "
            f"candidate={high.size} incumbent={low.size} "
            f"cluster_ids={len(cluster_ids)}"
        )

    members: dict[str, list[int]] = {}
    for row_index, cluster_id in enumerate(cluster_ids):
        members.setdefault(cluster_id, []).append(row_index)
    ids = sorted(members)
    groups = [
        np.asarray(members[cluster_id], dtype=np.int64) for cluster_id in ids
    ]
    n_clusters = len(groups)
    if n_clusters == 0:
        raise ValueError("bootstrap requires at least one cluster")

    rng = np.random.default_rng(seed)
    deltas = np.empty(n_boot, dtype=float)
    n_valid = 0
    for _ in range(n_boot):
        drawn = rng.integers(0, n_clusters, size=n_clusters)
        idx = np.concatenate(
            [groups[int(group_index)] for group_index in drawn]
        )
        yb = y[idx]
        s = int(yb.sum())
        if s == 0 or s == yb.size:
            continue
        deltas[n_valid] = auroc(high[idx], yb) - auroc(low[idx], yb)
        n_valid += 1
    if n_valid < n_boot // 2:
        raise AssertionError(
            f"only {n_valid}/{n_boot} bootstrap resamples were valid"
        )
    valid = deltas[:n_valid]
    ci95_low, ci95_high = np.percentile(valid, [2.5, 97.5])
    result = {
        "delta_auroc": float(auroc(high, y) - auroc(low, y)),
        "ci95_low": float(ci95_low),
        "ci95_high": float(ci95_high),
        "p_delta_gt_0": float(np.mean(valid > 0.0)),
        "n_valid_resamples": int(n_valid),
        "n_bootstrap": int(n_boot),
        "seed": int(seed),
        "resampling_unit": "cluster",
        "cluster_field": cluster_field,
        "n_clusters": int(n_clusters),
        "max_cluster_multiplicity": int(max(group.size for group in groups)),
    }
    assert {"delta_auroc", "ci95_low", "ci95_high"} <= result.keys()
    return result


def _complete_timing_mean(
    seconds: Sequence[float | None],
) -> tuple[float | None, int]:
    timed = [float(value) for value in seconds if value is not None]
    if not seconds or len(timed) != len(seconds):
        return None, len(timed)
    return float(sum(timed) / len(seconds)), len(timed)


def evaluate(pairs: Pairs, *, n_boot: int = 2000, seed: int = 0) -> dict:
    """Evaluate both arms and apply only the held-out delta-AUROC CI gate.

    The holdout file is tag-sorted: its first 100 rows are all ``tag=correct``.
    Because ``metrics.auroc`` returns NaN when a class is absent, a prefix run
    must be refused before it can serialize a NaN-poisoned decision.
    """
    lengths = {
        "record_ids": len(pairs.record_ids),
        "cluster_ids": len(pairs.cluster_ids),
        "labels": len(pairs.labels),
        "candidate": len(pairs.candidate),
        "incumbent": len(pairs.incumbent),
        "candidate_seconds": len(pairs.candidate_seconds),
        "incumbent_seconds": len(pairs.incumbent_seconds),
    }
    if len(set(lengths.values())) != 1:
        raise PairingError(f"Pairs fields have unequal lengths: {lengths}")
    n = len(pairs.labels)
    n_pos = int(sum(bool(value) for value in pairs.labels))
    _refuse_single_class(pairs.labels)

    candidate_s, candidate_records_timed = _complete_timing_mean(
        pairs.candidate_seconds
    )
    incumbent_s, incumbent_records_timed = _complete_timing_mean(
        pairs.incumbent_seconds
    )
    candidate_block = score_block(
        pairs.candidate,
        pairs.labels,
        name="candidate",
        seconds_per_record=candidate_s,
    )
    incumbent_block = score_block(
        pairs.incumbent,
        pairs.labels,
        name="incumbent",
        seconds_per_record=incumbent_s,
    )
    bootstrap = paired_bootstrap_delta_auroc(
        pairs.labels,
        pairs.candidate,
        pairs.incumbent,
        cluster_ids=pairs.cluster_ids,
        cluster_field=pairs.cluster_field,
        n_boot=n_boot,
        seed=seed,
    )
    passed = bool(bootstrap["ci95_low"] > 0.0)
    return {
        "schema_version": 1,
        "kind": "probe_battery_decision",
        "n": n,
        "base_rate_correct": float(n_pos / n),
        "candidate": candidate_block,
        "incumbent": incumbent_block,
        "paired_bootstrap": bootstrap,
        "cost": {
            "candidate_s_per_record": candidate_s,
            "candidate_records_timed": candidate_records_timed,
            "incumbent_s_per_record": incumbent_s,
            "incumbent_records_timed": incumbent_records_timed,
            "incumbent_untimed_calls": pairs.incumbent_untimed_calls,
            "n": n,
            # REPLACEMENT semantics: what you save if the candidate REPLACES the
            # incumbent. Correct for a replacement arm; MEANINGLESS for an
            # additive one, where you pay both. An additive arm carrying a
            # `speedup_x` of 4.09 beside a GO reads as a cost win when the arm
            # actually costs 1.24x MORE, so the two ratios ship side by side and
            # the consumer is told which semantics apply.
            "speedup_x_if_replacement": float(incumbent_s / candidate_s)
            if (
                candidate_s is not None
                and incumbent_s is not None
                and candidate_s > 0.0
            )
            else None,
            # ADDITIVE semantics: you run the incumbent AND the candidate.
            "combined_s_per_record": (
                float(candidate_s + incumbent_s)
                if candidate_s is not None and incumbent_s is not None
                else None
            ),
            "cost_ratio_vs_incumbent_if_additive": (
                float((candidate_s + incumbent_s) / incumbent_s)
                if (
                    candidate_s is not None
                    and incumbent_s is not None
                    and incumbent_s > 0.0
                )
                else None
            ),
            "arm_semantics_note": (
                "An arm whose candidate score CONSUMES the incumbent score as a "
                "feature is ADDITIVE: read cost_ratio_vs_incumbent_if_additive, "
                "not speedup_x_if_replacement. An arm built only from probe "
                "features is a REPLACEMENT: read speedup_x_if_replacement."
            ),
        },
        "gate": {
            "rule": "held-out paired ci95_low(delta AUROC) > 0",
            "passed": passed,
        },
        "verdict": "GO" if passed else "NO-GO",
    }


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(_resolve_input_path(path).read_bytes()).hexdigest()


def _input_descriptor(path: str | Path) -> dict[str, str]:
    return {"path": str(path), "sha256": _sha256(path)}


def write_decision(
    path: str | Path,
    decision: dict,
    *,
    scores_path: str | Path,
    gold_path: str | Path | None = None,
) -> None:
    """Write a reproducible decision artifact with hashed inputs."""
    artifact = dict(decision)
    inputs = {"scores": _input_descriptor(scores_path)}
    if gold_path is not None:
        inputs["gold"] = _input_descriptor(gold_path)
    artifact["inputs"] = inputs
    artifact["generated_utc"] = datetime.now(timezone.utc).isoformat()
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _artifact_inputs(artifact: dict) -> tuple[list[tuple[str, dict]], str | None, str | None]:
    inputs = artifact.get("inputs")
    if isinstance(inputs, dict) and inputs:
        entries = list(inputs.items())
        scores = inputs.get("scores")
        gold = inputs.get("gold")
        scores_path = scores.get("path") if isinstance(scores, dict) else None
        gold_path = gold.get("path") if isinstance(gold, dict) else None
        return entries, scores_path, gold_path

    scores_file = artifact.get("scores_file")
    if isinstance(scores_file, dict):
        return [("scores_file", scores_file)], scores_file.get("path"), None
    return [], None, None


def _compare_recomputed(
    recorded: Any, recomputed: Any, field: str, differences: list[str]
) -> int:
    if isinstance(recomputed, dict):
        if not isinstance(recorded, dict):
            differences.append(
                f"{field}: recorded={recorded!r} recomputed={recomputed!r}"
            )
            return 1
        compared = 0
        for key, value in recomputed.items():
            child = f"{field}.{key}" if field else key
            if key not in recorded:
                differences.append(f"{child}: recorded=<missing> recomputed={value!r}")
                continue
            compared += _compare_recomputed(
                recorded[key], value, child, differences
            )
        return compared

    if isinstance(recomputed, list):
        if not isinstance(recorded, list) or len(recorded) != len(recomputed):
            differences.append(
                f"{field}: recorded={recorded!r} recomputed={recomputed!r}"
            )
            return 1
        return sum(
            _compare_recomputed(old, new, f"{field}[{i}]", differences)
            for i, (old, new) in enumerate(zip(recorded, recomputed))
        )

    if recorded != recomputed:
        differences.append(
            f"{field}: recorded={recorded!r} recomputed={recomputed!r}"
        )
    return 1


def reproduce(path: str | Path) -> int:
    """Re-hash inputs and exactly recompute B2-owned decision fields.

    Unknown top-level fields are deliberately ignored so D1 can attach split,
    combiner, kill-gate, sensitivity, and provenance metadata without changing
    this evaluator.
    """
    try:
        artifact = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        print(f"artifact: unable to read {path}: {error}", file=sys.stderr)
        print("compared_fields=0")
        return 1
    if not isinstance(artifact, dict):
        print("artifact: decision JSON root must be an object", file=sys.stderr)
        print("compared_fields=0")
        return 1

    entries, scores_path, gold_path = _artifact_inputs(artifact)
    digest_differences = []
    for name, entry in entries:
        if not isinstance(entry, dict) or not entry.get("path") or not entry.get("sha256"):
            digest_differences.append(
                f"{name}: input descriptor must contain path and sha256"
            )
            continue
        try:
            actual = _sha256(entry["path"])
        except OSError as error:
            digest_differences.append(f"{name}.sha256: unable to hash: {error}")
            continue
        if actual != entry["sha256"]:
            digest_differences.append(
                f"{name}.sha256: recorded={entry['sha256']!r} recomputed={actual!r}"
            )

    if not entries or scores_path is None:
        digest_differences.append(
            "inputs: no usable scores input in inputs or scores_file"
        )
    if digest_differences:
        for difference in digest_differences:
            print(difference, file=sys.stderr)
        print("compared_fields=0")
        return 1

    bootstrap = artifact.get("paired_bootstrap")
    if not isinstance(bootstrap, dict):
        bootstrap = artifact.get("bootstrap_rows")
    if not isinstance(bootstrap, dict):
        print("paired_bootstrap: missing bootstrap parameters", file=sys.stderr)
        print("compared_fields=0")
        return 1
    if "resampling_unit" not in bootstrap:
        print(
            "paired_bootstrap: artifact does not declare its resampling unit",
            file=sys.stderr,
        )
        print("compared_fields=0")
        return 1
    if "cluster_field" not in bootstrap:
        print(
            "paired_bootstrap: artifact does not declare its cluster field",
            file=sys.stderr,
        )
        print("compared_fields=0")
        return 1
    try:
        pairs = load_pairs(scores_path, gold_path)
        recomputed = evaluate(
            pairs,
            n_boot=int(bootstrap["n_bootstrap"]),
            seed=int(bootstrap["seed"]),
        )
    except (KeyError, OSError, ValueError, AssertionError, json.JSONDecodeError) as error:
        print(f"recompute: {type(error).__name__}: {error}", file=sys.stderr)
        print("compared_fields=0")
        return 1

    differences: list[str] = []
    compared = 0
    for key in _OWNED_TOP_LEVEL_KEYS:
        if key not in artifact:
            differences.append(
                f"{key}: recorded=<missing> recomputed={recomputed[key]!r}"
            )
            continue
        compared += _compare_recomputed(
            artifact[key], recomputed[key], key, differences
        )

    print(f"compared_fields={compared}")
    for difference in differences:
        print(difference, file=sys.stderr)
    if compared == 0:
        print("reproduce: no B2-owned fields were compared", file=sys.stderr)
        return 1
    return 1 if differences else 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reproduce", type=Path)
    parser.add_argument("--scores", type=Path)
    parser.add_argument("--gold", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)

    if args.reproduce is not None:
        if args.scores is not None or args.gold is not None or args.out is not None:
            parser.error("--reproduce cannot be combined with --scores, --gold, or --out")
        return reproduce(args.reproduce)
    if args.scores is None or args.out is None:
        parser.error("provide --reproduce PATH or --scores PATH --out PATH [--gold PATH]")

    pairs = load_pairs(args.scores, args.gold)
    decision = evaluate(pairs)
    write_decision(
        args.out, decision, scores_path=args.scores, gold_path=args.gold
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
