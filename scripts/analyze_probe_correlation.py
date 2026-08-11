#!/usr/bin/env python3
"""FIT-split correlation kill gate for the no-reasoning probe battery.

This script performs no inference.  Its primary path depends only on numpy and
the repository's canonical gold/AUROC/rank helpers.  ``--assert-artifact`` is a
deliberately independent scipy/sklearn recomputation of a real, powered result.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from indra_belief.curation import is_gold_correct  # noqa: E402
from indra_belief.metrics import _rankdata_avg, auroc  # noqa: E402


DEFAULT_ARTIFACT = ROOT / "data/probe_battery/killgate.json"
DEFAULT_PROBES = ROOT / "data/probe_battery/probes_fit.jsonl"
GOLD = ROOT / "data/benchmark/eval_curation_v1.jsonl"

KILL_RULE = {
  "bootstrap_resamples": 2000,
  "bootstrap_seed": 20260809,
  "cluster_field": "pa_hash",
  "k1_max_probe_spearman_vs_base_ci_low_abs": 0.90,
  "k2_max_delta_auroc_vs_base_ci_high": 0.02,
  "k3_min_abs_pairwise_spearman": 0.90,
  "k3_pc1_explained_variance_ci_low": 0.95,
  "power_max_base_auroc_se": 0.035,
  "power_max_class_imbalance": 0.10,
  "power_min_clusters": 350,
  "power_min_rows": 500,
}
KILL_RULE_SHA_FROZEN = "b23952b83b98f9cc8bf8ed7e438b4192f454df8ff4b349905a13bfd80a014650"

assert (
    hashlib.sha256(
        json.dumps(KILL_RULE, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    == KILL_RULE_SHA_FROZEN
)

SCHEMA_VERSION = 1
CLIP_EPS = 1e-6
SCALAR_SOURCE = (
    "delta_logit when present; otherwise "
    "logit(clip(p_raw, 1e-6, 1-1e-6))"
)
SCALAR_NOTE = (
    "delta_logit preserves saturated ordering: on smoke_fit.jsonl, "
    "logit(clip(p_raw, 1e-6, 1-1e-6)) would collapse 22/64 readings "
    "(34.4%) at the clip bounds. KILL_RULE is unchanged, and non-clipped "
    "delta_logit/logit(p_raw) ranks are asserted identical."
)


class AnalysisError(ValueError):
    """A named input, integrity, power, or reproduction failure."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path, *, kind: str) -> list[dict[str, Any]]:
    if not path.is_file():
        raise AnalysisError(f"{kind} is missing: {path}")
    values: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise AnalysisError(
                    f"{kind} contains an empty line at line {line_number}: {path}"
                )
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise AnalysisError(
                    f"{kind} has invalid JSON at line {line_number}: {exc.msg}"
                ) from exc
            if not isinstance(value, dict):
                raise AnalysisError(
                    f"{kind} line {line_number} must be a JSON object"
                )
            values.append(value)
    if not values:
        raise AnalysisError(f"{kind} is empty: {path}")
    return values


def _load_gold(path: Path) -> list[dict[str, Any]]:
    return _read_jsonl(path, kind="gold JSONL")


def _missing_field(*, context: str, field: str) -> AnalysisError:
    return AnalysisError(f"{context} missing required field {field!r}")


def _validate_probe_parts(
    manifest: dict[str, Any],
    records: Sequence[dict[str, Any]],
    *,
    source: str,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    if manifest.get("_manifest") is not True:
        raise AnalysisError(f"{source} first line must have '_manifest' is True")
    if manifest.get("split") != "fit":
        raise AnalysisError(
            f"{source} manifest split must be 'fit', got {manifest.get('split')!r}"
        )
    if "probe_meta" not in manifest:
        raise _missing_field(context=f"{source} manifest", field="probe_meta")
    raw_meta = manifest["probe_meta"]
    if not isinstance(raw_meta, dict) or not raw_meta:
        raise AnalysisError(f"{source} manifest field 'probe_meta' must be non-empty")

    probe_meta: dict[str, dict[str, Any]] = {}
    for probe_id, raw in raw_meta.items():
        if not isinstance(probe_id, str) or not probe_id:
            raise AnalysisError(f"{source} probe_meta contains an invalid probe id")
        if not isinstance(raw, dict):
            raise AnalysisError(f"probe_meta[{probe_id!r}] must be an object")
        for field in ("family", "is_base"):
            if field not in raw:
                raise _missing_field(
                    context=f"probe_meta[{probe_id!r}]", field=field
                )
        if not isinstance(raw["family"], str):
            raise AnalysisError(f"probe_meta[{probe_id!r}] field 'family' must be str")
        if not isinstance(raw["is_base"], bool):
            raise AnalysisError(
                f"probe_meta[{probe_id!r}] field 'is_base' must be bool"
            )
        probe_meta[probe_id] = {
            "family": raw["family"],
            "is_base": raw["is_base"],
        }

    bases = [probe_id for probe_id, meta in probe_meta.items() if meta["is_base"]]
    if len(bases) != 1:
        raise AnalysisError(
            f"probe_meta must declare exactly one is_base: true probe; got {len(bases)}"
        )
    if len(probe_meta) < 2:
        raise AnalysisError("probe_meta must declare a base and at least one other probe")

    seen_indices: set[int] = set()
    checked: list[dict[str, Any]] = []
    declared = set(probe_meta)
    for ordinal, record in enumerate(records, start=2):
        if record.get("_manifest") is True:
            raise AnalysisError(
                f"{source} contains more than one '_manifest' line (line {ordinal})"
            )
        for field in ("row_index", "source_hash", "pa_hash", "probes"):
            if field not in record:
                raise _missing_field(context=f"probe record line {ordinal}", field=field)
        row_index = record["row_index"]
        if isinstance(row_index, bool) or not isinstance(row_index, int):
            raise AnalysisError(
                f"probe record line {ordinal} field 'row_index' must be an integer"
            )
        if row_index in seen_indices:
            raise AnalysisError(f"duplicate row_index {row_index} in {source}")
        seen_indices.add(row_index)
        if not isinstance(record["source_hash"], str) or not record["source_hash"]:
            raise AnalysisError(
                f"probe record row_index {row_index} field 'source_hash' must be str"
            )
        if not isinstance(record["pa_hash"], str) or not record["pa_hash"]:
            raise AnalysisError(
                f"probe record row_index {row_index} field 'pa_hash' must be str"
            )
        probes = record["probes"]
        if not isinstance(probes, dict):
            raise AnalysisError(
                f"probe record row_index {row_index} field 'probes' must be an object"
            )
        missing_probes = sorted(declared - set(probes))
        if missing_probes:
            raise AnalysisError(
                f"probe record row_index {row_index} field 'probes' missing probe "
                f"{missing_probes[0]!r}"
            )
        extra_probes = sorted(set(probes) - declared)
        if extra_probes:
            raise AnalysisError(
                f"probe record row_index {row_index} field 'probes' has undeclared "
                f"probe {extra_probes[0]!r}"
            )
        for probe_id in sorted(probe_meta):
            value = probes[probe_id]
            if not isinstance(value, dict):
                raise AnalysisError(
                    f"probe record row_index {row_index} probe {probe_id!r} "
                    "must be an object"
                )
            for field in (
                "p_raw",
                "status",
                "both_observed",
                "precision_limited",
            ):
                if field not in value:
                    raise _missing_field(
                        context=(
                            f"probe record row_index {row_index} probe {probe_id!r}"
                        ),
                        field=field,
                    )
            if not isinstance(value["status"], str):
                raise AnalysisError(
                    f"probe record row_index {row_index} probe {probe_id!r} "
                    "field 'status' must be str"
                )
            for field in ("both_observed", "precision_limited"):
                if not isinstance(value[field], bool):
                    raise AnalysisError(
                        f"probe record row_index {row_index} probe {probe_id!r} "
                        f"field {field!r} must be bool"
                    )
            p_raw = value["p_raw"]
            if p_raw is not None:
                if (
                    isinstance(p_raw, bool)
                    or not isinstance(p_raw, (int, float))
                    or not math.isfinite(float(p_raw))
                    or not 0.0 <= float(p_raw) <= 1.0
                ):
                    raise AnalysisError(
                        f"probe record row_index {row_index} probe {probe_id!r} "
                        "field 'p_raw' must be null or a finite number in [0,1]"
                    )
            delta = value.get("delta_logit")
            if delta is not None and (
                isinstance(delta, bool)
                or not isinstance(delta, (int, float))
                or not math.isfinite(float(delta))
            ):
                raise AnalysisError(
                    f"probe record row_index {row_index} probe {probe_id!r} "
                    "field 'delta_logit' must be finite when present"
                )
            if value["status"] == "ok" and p_raw is None:
                raise AnalysisError(
                    f"probe record row_index {row_index} probe {probe_id!r} "
                    "field 'p_raw' is null while status is 'ok'"
                )
        elapsed = record.get("elapsed_s")
        if elapsed is not None and (
            isinstance(elapsed, bool)
            or not isinstance(elapsed, (int, float))
            or not math.isfinite(float(elapsed))
            or float(elapsed) < 0.0
        ):
            raise AnalysisError(
                f"probe record row_index {row_index} field 'elapsed_s' must be "
                "null or a finite non-negative number"
            )
        checked.append(record)
    checked.sort(key=lambda row: row["row_index"])
    return probe_meta, checked


def _load_probe_jsonl(
    path: Path,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], list[dict[str, Any]]]:
    values = _read_jsonl(path, kind="probe JSONL")
    manifest, records = values[0], values[1:]
    probe_meta, checked = _validate_probe_parts(
        manifest, records, source=str(path)
    )
    return manifest, probe_meta, checked


def _join_records(
    gold_rows: Sequence[dict[str, Any]],
    records: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Join only by row_index, then cross-check both provenance hashes."""
    joined: list[dict[str, Any]] = []
    for record in records:
        row_index = record["row_index"]
        if row_index < 0 or row_index >= len(gold_rows):
            raise AnalysisError(
                f"probe record row_index {row_index} is outside gold row range "
                f"0..{len(gold_rows) - 1}"
            )
        gold = gold_rows[row_index]
        for field in ("pa_hash", "source_hash", "tag"):
            if field not in gold:
                raise _missing_field(context=f"gold row_index {row_index}", field=field)
        for field in ("pa_hash", "source_hash"):
            if str(gold[field]) != str(record[field]):
                raise AnalysisError(
                    f"row_index {row_index} {field} mismatch: "
                    f"gold={gold[field]!r}, probe={record[field]!r}"
                )
        gold_correct = bool(is_gold_correct(gold["tag"]))
        if "gold_correct" in record:
            supplied = record["gold_correct"]
            if not isinstance(supplied, bool):
                raise AnalysisError(
                    f"row_index {row_index} field 'gold_correct' must be bool"
                )
            if supplied != gold_correct:
                raise AnalysisError(
                    f"row_index {row_index} gold_correct mismatch: "
                    f"gold tag {gold['tag']!r} gives {gold_correct}, probe gives {supplied}"
                )
        joined.append(
            {
                "row_index": row_index,
                "source_hash": str(record["source_hash"]),
                "pa_hash": str(record["pa_hash"]),
                "tag": gold["tag"],
                "gold_correct": gold_correct,
                "elapsed_s": record.get("elapsed_s"),
                "probes": record["probes"],
            }
        )
    joined.sort(key=lambda row: row["row_index"])
    return joined


def _row_gold(row: dict[str, Any]) -> bool:
    if "gold_correct" in row:
        return bool(row["gold_correct"])
    if "tag" not in row:
        raise _missing_field(context="sample row", field="tag")
    return bool(is_gold_correct(row["tag"]))


def select_killgate_sample(
    rows: Sequence[dict[str, Any]], *, n_rows: int, seed: int
) -> list[dict[str, Any]]:
    """Select a deterministic, gold-stratified, shuffled row-grain sample."""
    if isinstance(n_rows, bool) or not isinstance(n_rows, int) or n_rows <= 0:
        raise AnalysisError("n_rows must be a positive integer")
    if n_rows > len(rows):
        raise AnalysisError(
            f"cannot sample n_rows={n_rows} from only {len(rows)} rows"
        )
    positives = [index for index, row in enumerate(rows) if _row_gold(row)]
    negatives = [index for index, row in enumerate(rows) if not _row_gold(row)]
    if not positives or not negatives:
        raise AnalysisError("gold-stratified sampling requires both gold classes")

    # Deliberately do not use sampling.two_stage_sample: its per-statement cap
    # changes the cluster distribution against which the >=350-cluster floor was
    # simulated.  Membership is selected at plain ROW grain, never as a prefix.
    raw_positive = n_rows * len(positives) / len(rows)
    target_positive = int(math.floor(raw_positive + 0.5))
    minimum_positive = max(0, n_rows - len(negatives))
    maximum_positive = min(len(positives), n_rows)
    if n_rows >= 2:
        minimum_positive = max(minimum_positive, 1)
        maximum_positive = min(maximum_positive, n_rows - 1)
    target_positive = min(
        maximum_positive, max(minimum_positive, target_positive)
    )
    target_negative = n_rows - target_positive

    rng = np.random.default_rng(seed)
    selected = list(rng.permutation(positives)[:target_positive])
    selected.extend(rng.permutation(negatives)[:target_negative])
    selected = list(rng.permutation(selected))
    return [rows[int(index)] for index in selected]


def _logit_clipped(value: float) -> float:
    clipped = min(max(float(value), CLIP_EPS), 1.0 - CLIP_EPS)
    return math.log(clipped) - math.log1p(-clipped)


def _snap_logit_roundoff(
    delta: np.ndarray, transformed: np.ndarray
) -> np.ndarray:
    """Remove only machine-roundoff differences from a logit round trip."""
    scale = np.maximum(1.0, np.maximum(np.abs(delta), np.abs(transformed)))
    tolerance = 8.0 * np.finfo(float).eps * scale
    return np.where(np.abs(delta - transformed) <= tolerance, delta, transformed)


def _prepare_data(
    rows: Sequence[dict[str, Any]],
    probe_meta: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    if not rows:
        raise AnalysisError("probe input has no data records")
    probe_ids = sorted(probe_meta)
    base_ids = [probe_id for probe_id in probe_ids if probe_meta[probe_id]["is_base"]]
    if len(base_ids) != 1:
        raise AnalysisError(
            f"probe_meta must declare exactly one is_base: true probe; got {len(base_ids)}"
        )
    base_id = base_ids[0]
    labels = np.asarray([bool(row["gold_correct"]) for row in rows], dtype=bool)
    clusters = np.asarray([str(row["pa_hash"]) for row in rows], dtype=object)
    tags = np.asarray([str(row["tag"]) for row in rows], dtype=object)
    scalars: dict[str, np.ndarray] = {}
    ok_masks: dict[str, np.ndarray] = {}
    accounting: dict[str, dict[str, Any]] = {}

    for probe_id in probe_ids:
        scores = np.full(len(rows), np.nan, dtype=float)
        ok_mask = np.zeros(len(rows), dtype=bool)
        statuses: Counter[str] = Counter()
        both_count = 0
        limited_count = 0
        n_would_clip = 0
        rank_delta: list[float] = []
        rank_logit: list[float] = []
        for index, row in enumerate(rows):
            value = row["probes"][probe_id]
            status = value["status"]
            statuses[status] += 1
            both_count += int(value["both_observed"])
            limited_count += int(value["precision_limited"])
            p_raw_value = value["p_raw"]
            delta_raw = value.get("delta_logit")
            clipped_logit: float | None = None
            if p_raw_value is not None:
                p_raw = float(p_raw_value)
                clipped_logit = _logit_clipped(p_raw)
                if delta_raw is not None:
                    delta = float(delta_raw)
                    if abs(clipped_logit - delta) > 1e-9:
                        n_would_clip += 1
                    if CLIP_EPS <= p_raw <= 1.0 - CLIP_EPS:
                        rank_delta.append(delta)
                        rank_logit.append(math.log(p_raw) - math.log1p(-p_raw))
            if status != "ok":
                continue
            if delta_raw is not None:
                score = float(delta_raw)
            else:
                assert clipped_logit is not None
                score = clipped_logit
            scores[index] = score
            ok_mask[index] = True

        rank_identity = True
        if rank_delta:
            delta_array = np.asarray(rank_delta, dtype=float)
            logit_array = np.asarray(rank_logit, dtype=float)
            logit_array = _snap_logit_roundoff(delta_array, logit_array)
            rank_identity = bool(
                np.array_equal(
                    _rankdata_avg(delta_array),
                    _rankdata_avg(logit_array),
                )
            )
        if not rank_identity:
            raise AnalysisError(
                f"probe {probe_id!r} delta_logit and non-clipped logit(p_raw) "
                "are not rank-identical"
            )
        n_ok = int(ok_mask.sum())
        scalars[probe_id] = scores
        ok_masks[probe_id] = ok_mask
        accounting[probe_id] = {
            "family": probe_meta[probe_id]["family"],
            "is_base": bool(probe_meta[probe_id]["is_base"]),
            "n_ok": n_ok,
            "n_status_not_ok": {
                status: int(count)
                for status, count in sorted(statuses.items())
                if status != "ok"
            },
            "frac_both_observed": float(both_count / len(rows)),
            "frac_precision_limited": float(limited_count / len(rows)),
            "n_distinct_values": int(len(np.unique(scores[ok_mask]))),
            "n_would_clip": int(n_would_clip),
            "non_clipped_rank_identity": rank_identity,
            "n_non_clipped_rank_compared": int(len(rank_delta)),
        }

    elapsed = [row.get("elapsed_s") for row in rows]
    timed = [float(value) for value in elapsed if value is not None]
    complete_timing = len(timed) == len(rows)
    cost = {
        "mean_elapsed_s": (
            float(np.mean(timed)) if complete_timing and timed else None
        ),
        "median_elapsed_s": (
            float(np.median(timed)) if complete_timing and timed else None
        ),
        "n_timed": int(len(timed)),
        "n_missing": int(len(rows) - len(timed)),
    }
    return {
        "rows": list(rows),
        "probe_ids": probe_ids,
        "base_id": base_id,
        "labels": labels,
        "clusters": clusters,
        "tags": tags,
        "scalars": scalars,
        "ok_masks": ok_masks,
        "accounting": accounting,
        "cost": cost,
    }


def _cluster_bootstrap_indices(
    clusters: np.ndarray, *, n_boot: int, seed: int
) -> list[np.ndarray]:
    """Generate one shared set of full-group cluster-bootstrap row indices."""
    members: dict[str, list[int]] = {}
    for index, cluster in enumerate(clusters):
        members.setdefault(str(cluster), []).append(index)
    cluster_ids = sorted(members)
    if not cluster_ids:
        raise AnalysisError("cannot bootstrap zero clusters")
    arrays = {
        cluster_id: np.asarray(members[cluster_id], dtype=np.int64)
        for cluster_id in cluster_ids
    }
    rng = np.random.default_rng(seed)
    draws: list[np.ndarray] = []
    for _ in range(n_boot):
        selected = rng.integers(0, len(cluster_ids), size=len(cluster_ids))
        draws.append(
            np.concatenate([arrays[cluster_ids[int(index)]] for index in selected])
        )
    return draws


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    left = np.asarray(a, dtype=float)
    right = np.asarray(b, dtype=float)
    if (
        left.size < 2
        or right.size != left.size
        or len(np.unique(left)) <= 1
        or len(np.unique(right)) <= 1
    ):
        return float("nan")
    # Mid-ranks, not argsort-of-argsort.  The naive form breaks ties by position
    # and makes the coefficient depend on input order.
    return float(
        np.corrcoef(_rankdata_avg(left), _rankdata_avg(right))[0, 1]
    )


def _masked_draw(draw: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return draw[mask[draw]]


def _auc_for_indices(
    scores: np.ndarray, labels: np.ndarray, indices: np.ndarray
) -> float:
    if indices.size < 2:
        return float("nan")
    y = labels[indices]
    if bool(y.all()) or bool((~y).all()):
        return float("nan")
    return float(auroc(scores[indices], y))


def _collect_bootstrap(
    draws: Sequence[np.ndarray],
    statistic: Callable[[np.ndarray], float],
) -> np.ndarray:
    values: list[float] = []
    for draw in draws:
        value = float(statistic(draw))
        if math.isfinite(value):
            values.append(value)
    return np.asarray(values, dtype=float)


def _require_half(values: np.ndarray, *, family: str, n_boot: int) -> None:
    if len(values) < n_boot // 2:
        raise AnalysisError(
            f"only {len(values)}/{n_boot} cluster-bootstrap resamples were "
            f"valid for {family}"
        )


def _ci_summary(
    point_name: str,
    point: float,
    values: np.ndarray,
    *,
    family: str,
    require_half: bool = True,
) -> dict[str, Any]:
    n_boot = int(KILL_RULE["bootstrap_resamples"])
    if require_half:
        _require_half(values, family=family, n_boot=n_boot)
    if len(values) >= n_boot // 2:
        low, high = np.percentile(values, [2.5, 97.5])
        ci_low: float | None = float(low)
        ci_high: float | None = float(high)
    else:
        ci_low = None
        ci_high = None
    return {
        point_name: float(point) if math.isfinite(point) else None,
        "ci95_low": ci_low,
        "ci95_high": ci_high,
        "n_valid_resamples": int(len(values)),
        "n_bootstrap": n_boot,
        "seed": int(KILL_RULE["bootstrap_seed"]),
    }


def _pc_ratios_svd(matrix: np.ndarray) -> np.ndarray | None:
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 1:
        return None
    std = values.std(axis=0, ddof=0)
    scale = np.where(std > 0.0, std, 1.0)
    standardized = (values - values.mean(axis=0)) / scale
    singular = np.linalg.svd(standardized, full_matrices=False, compute_uv=False)
    variances = singular * singular
    total = float(variances.sum())
    if not math.isfinite(total) or total <= 0.0:
        return None
    return variances / total


def _rho_matrix(data: dict[str, Any]) -> dict[str, Any]:
    probe_ids = data["probe_ids"]
    values: list[list[float | None]] = []
    counts: list[list[int]] = []
    for left_id in probe_ids:
        row_values: list[float | None] = []
        row_counts: list[int] = []
        for right_id in probe_ids:
            mask = data["ok_masks"][left_id] & data["ok_masks"][right_id]
            rho = _spearman(
                data["scalars"][left_id][mask],
                data["scalars"][right_id][mask],
            )
            row_values.append(float(rho) if math.isfinite(rho) else None)
            row_counts.append(int(mask.sum()))
        values.append(row_values)
        counts.append(row_counts)
    return {"probe_ids": probe_ids, "values": values, "n_pairwise": counts}


def _advisory_tag_aurocs(
    data: dict[str, Any],
    probe_id: str,
    draws: Sequence[np.ndarray],
) -> dict[str, Any]:
    tags = data["tags"]
    labels = data["labels"]
    scores = data["scalars"][probe_id]
    ok = data["ok_masks"][probe_id]
    blocks: dict[str, Any] = {}
    for tag in sorted({str(value) for value in tags if not is_gold_correct(str(value))}):
        mask = ok & ((tags == "correct") | (tags == tag))
        point_indices = np.flatnonzero(mask)
        point = _auc_for_indices(scores, labels, point_indices)
        samples = _collect_bootstrap(
            draws,
            lambda draw, mask=mask: _auc_for_indices(
                scores, labels, _masked_draw(draw, mask)
            ),
        )
        block = _ci_summary(
            "auroc",
            point,
            samples,
            family=f"advisory AUROC {probe_id} correct-vs-{tag}",
            require_half=math.isfinite(point),
        )
        block["n_rows"] = int(mask.sum())
        blocks[tag] = block
    return {
        "advisory": True,
        "definition": (
            "For each non-correct tag, AUROC is computed on rows tagged "
            "correct or that tag; this block gates nothing."
        ),
        "tags": blocks,
    }


def _apply_verdict(
    *,
    data: dict[str, Any],
    rho: dict[str, Any],
    pc1: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], str, str]:
    base_id = data["base_id"]
    nonbase = [probe_id for probe_id in data["probe_ids"] if probe_id != base_id]
    k1_threshold = float(
        KILL_RULE["k1_max_probe_spearman_vs_base_ci_low_abs"]
    )
    k1_per_probe: dict[str, Any] = {}
    realized_nonconstant: list[float] = []
    for probe_id in nonbase:
        distinct = data["accounting"][probe_id]["n_distinct_values"]
        spearman_block = data["probe_results"][probe_id]["spearman_vs_base"]
        if distinct <= 1:
            satisfies = True
            reason = "zero_variance"
            realized = None
        else:
            realized = spearman_block["ci_low_abs"]
            satisfies = realized is not None and realized >= k1_threshold
            reason = None
            if realized is not None:
                realized_nonconstant.append(float(realized))
        k1_per_probe[probe_id] = {
            "realized_ci_low_abs": realized,
            "threshold": k1_threshold,
            "boolean": bool(satisfies),
            "redundant_reason": reason,
        }
    k1_boolean = all(block["boolean"] for block in k1_per_probe.values())
    k1 = {
        "realized_min_ci_low_abs": (
            min(realized_nonconstant) if realized_nonconstant else None
        ),
        "threshold": k1_threshold,
        "boolean": bool(k1_boolean),
        "per_probe": k1_per_probe,
    }

    k2_threshold = float(KILL_RULE["k2_max_delta_auroc_vs_base_ci_high"])
    delta_highs = {
        probe_id: data["probe_results"][probe_id]["delta_auroc_vs_base"][
            "ci95_high"
        ]
        for probe_id in nonbase
    }
    finite_highs = [float(value) for value in delta_highs.values() if value is not None]
    k2_realized = max(finite_highs) if len(finite_highs) == len(nonbase) else None
    k2_boolean = k2_realized is not None and k2_realized <= k2_threshold
    k2 = {
        "realized_max_ci95_high": k2_realized,
        "threshold": k2_threshold,
        "boolean": bool(k2_boolean),
        "per_probe_ci95_high": delta_highs,
    }

    off_diagonal: list[float] = []
    undefined_pair = False
    for i in range(len(rho["probe_ids"])):
        for j in range(i + 1, len(rho["probe_ids"])):
            value = rho["values"][i][j]
            if value is None:
                undefined_pair = True
            else:
                off_diagonal.append(abs(float(value)))
    min_abs_rho = (
        None if undefined_pair or not off_diagonal else min(off_diagonal)
    )
    pc_low = pc1["ci95_low"]
    pc_threshold = float(KILL_RULE["k3_pc1_explained_variance_ci_low"])
    rho_threshold = float(KILL_RULE["k3_min_abs_pairwise_spearman"])
    k3_boolean = (
        pc_low is not None
        and pc_low >= pc_threshold
        and min_abs_rho is not None
        and min_abs_rho >= rho_threshold
    )
    k3 = {
        "realized_pc1_ci95_low": pc_low,
        "pc1_threshold": pc_threshold,
        "realized_min_abs_pairwise_spearman": min_abs_rho,
        "spearman_threshold": rho_threshold,
        "boolean": bool(k3_boolean),
    }

    no_go = bool((k1_boolean and k2_boolean) or k3_boolean)
    verdict = "NO-GO" if no_go else "GO"
    if k1_boolean and k2_boolean and k3_boolean:
        reason = "NO-GO: K1 and K2 fired, and K3 also fired."
    elif k1_boolean and k2_boolean:
        reason = "NO-GO: K1 and K2 fired; K3 did not fire."
    elif k3_boolean:
        reason = "NO-GO: K3 fired; the combined K1-and-K2 condition did not fire."
    else:
        reason = "GO: neither the combined K1-and-K2 condition nor K3 fired."
    return k1, k2, k3, verdict, reason


def _artifact_skeleton(
    *,
    data: dict[str, Any],
    inputs: dict[str, Any],
    generated_utc: str,
    power: dict[str, Any],
    powered: bool,
    power_failure: list[dict[str, Any]],
) -> dict[str, Any]:
    probes = {
        probe_id: dict(data["accounting"][probe_id])
        for probe_id in data["probe_ids"]
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_utc": generated_utc,
        "split": "fit",
        "in_sample": True,
        "inputs": inputs,
        "rule": KILL_RULE,
        "rule_sha256": KILL_RULE_SHA_FROZEN,
        "scalar_source": SCALAR_SOURCE,
        "scalar_note": SCALAR_NOTE,
        "powered": powered,
        "power": power,
        "power_failure": power_failure,
        "base_probe_id": data["base_id"],
        "probes": probes,
        "rho_matrix": None,
        "pc1_explained_variance": None,
        "n_effective_dimensions": None,
        "cost": data["cost"],
        "k1": None,
        "k2": None,
        "k3": None,
        "verdict": None,
        "verdict_reason": (
            "Power gate failed; no verdict was issued." if not powered else None
        ),
    }


def _analyze_joined(
    rows: Sequence[dict[str, Any]],
    probe_meta: dict[str, dict[str, Any]],
    *,
    inputs: dict[str, Any],
    generated_utc: str | None = None,
) -> dict[str, Any]:
    data = _prepare_data(rows, probe_meta)
    n_boot = int(KILL_RULE["bootstrap_resamples"])
    bootstrap_seed = int(KILL_RULE["bootstrap_seed"])
    draws = _cluster_bootstrap_indices(
        data["clusters"], n_boot=n_boot, seed=bootstrap_seed
    )

    n_rows = len(rows)
    n_clusters = len({str(value) for value in data["clusters"]})
    n_pos = int(data["labels"].sum())
    n_neg = n_rows - n_pos
    class_imbalance = abs(n_pos - n_neg) / n_rows
    base_id = data["base_id"]
    base_mask = data["ok_masks"][base_id]
    base_scores = data["scalars"][base_id]
    base_point = _auc_for_indices(
        base_scores, data["labels"], np.flatnonzero(base_mask)
    )
    base_samples = _collect_bootstrap(
        draws,
        lambda draw: _auc_for_indices(
            base_scores, data["labels"], _masked_draw(draw, base_mask)
        ),
    )
    base_se = (
        float(np.std(base_samples, ddof=1)) if len(base_samples) >= 2 else None
    )

    row_pass = n_rows >= int(KILL_RULE["power_min_rows"])
    cluster_pass = n_clusters >= int(KILL_RULE["power_min_clusters"])
    imbalance_pass = class_imbalance <= float(
        KILL_RULE["power_max_class_imbalance"]
    )
    se_pass = (
        base_se is not None
        and math.isfinite(base_se)
        and len(base_samples) >= n_boot // 2
        and base_se <= float(KILL_RULE["power_max_base_auroc_se"])
    )
    power = {
        "n_rows": int(n_rows),
        "n_clusters": int(n_clusters),
        "class_imbalance": float(class_imbalance),
        "base_auroc": float(base_point) if math.isfinite(base_point) else None,
        "base_auroc_cluster_bootstrap_se": base_se,
        "base_auroc_n_valid_resamples": int(len(base_samples)),
        "n_bootstrap": n_boot,
        "seed": bootstrap_seed,
        "thresholds": {
            "power_min_rows": int(KILL_RULE["power_min_rows"]),
            "power_min_clusters": int(KILL_RULE["power_min_clusters"]),
            "power_max_class_imbalance": float(
                KILL_RULE["power_max_class_imbalance"]
            ),
            "power_max_base_auroc_se": float(
                KILL_RULE["power_max_base_auroc_se"]
            ),
        },
    }
    power_failure: list[dict[str, Any]] = []
    for failed, name, operator, threshold, realized in (
        (
            not row_pass,
            "power_min_rows",
            ">=",
            int(KILL_RULE["power_min_rows"]),
            int(n_rows),
        ),
        (
            not cluster_pass,
            "power_min_clusters",
            ">=",
            int(KILL_RULE["power_min_clusters"]),
            int(n_clusters),
        ),
        (
            not imbalance_pass,
            "power_max_class_imbalance",
            "<=",
            float(KILL_RULE["power_max_class_imbalance"]),
            float(class_imbalance),
        ),
        (
            not se_pass,
            "power_max_base_auroc_se",
            "<=",
            float(KILL_RULE["power_max_base_auroc_se"]),
            base_se,
        ),
    ):
        if failed:
            power_failure.append(
                {
                    "threshold_name": name,
                    "operator": operator,
                    "threshold": threshold,
                    "realized": realized,
                }
            )
    powered = not power_failure
    artifact = _artifact_skeleton(
        data=data,
        inputs=inputs,
        generated_utc=generated_utc or _utc_now(),
        power=power,
        powered=powered,
        power_failure=power_failure,
    )
    if not powered:
        return artifact

    _require_half(base_samples, family="base AUROC", n_boot=n_boot)
    probe_results: dict[str, dict[str, Any]] = {}
    for probe_id in data["probe_ids"]:
        scores = data["scalars"][probe_id]
        mask = data["ok_masks"][probe_id]
        point_indices = np.flatnonzero(mask)
        point_auc = _auc_for_indices(scores, data["labels"], point_indices)
        if probe_id == base_id:
            auc_samples = base_samples
        else:
            auc_samples = _collect_bootstrap(
                draws,
                lambda draw, scores=scores, mask=mask: _auc_for_indices(
                    scores, data["labels"], _masked_draw(draw, mask)
                ),
            )
        auc_block = _ci_summary(
            "auroc",
            point_auc,
            auc_samples,
            family=f"AUROC {probe_id}",
            require_half=math.isfinite(point_auc),
        )
        if probe_id == base_id:
            spearman_block = None
            delta_block = None
        else:
            paired_mask = mask & data["ok_masks"][base_id]
            paired_indices = np.flatnonzero(paired_mask)
            point_rho = _spearman(
                scores[paired_indices], base_scores[paired_indices]
            )
            rho_samples = _collect_bootstrap(
                draws,
                lambda draw, scores=scores, paired_mask=paired_mask: (
                    lambda idx: _spearman(scores[idx], base_scores[idx])
                )(_masked_draw(draw, paired_mask)),
            )
            if not math.isfinite(point_rho):
                spearman_block = _ci_summary(
                    "spearman",
                    point_rho,
                    rho_samples,
                    family=f"Spearman {probe_id} vs base",
                    require_half=False,
                )
                spearman_block["ci_low_abs"] = None
            else:
                _require_half(
                    rho_samples,
                    family=f"Spearman {probe_id} vs base",
                    n_boot=n_boot,
                )
                spearman_block = _ci_summary(
                    "spearman",
                    point_rho,
                    rho_samples,
                    family=f"Spearman {probe_id} vs base",
                )
                spearman_block["ci_low_abs"] = float(
                    np.percentile(np.abs(rho_samples), 2.5)
                )
            spearman_block["n_paired"] = int(paired_mask.sum())

            point_delta = _auc_for_indices(
                scores, data["labels"], paired_indices
            ) - _auc_for_indices(base_scores, data["labels"], paired_indices)
            delta_samples = _collect_bootstrap(
                draws,
                lambda draw, scores=scores, paired_mask=paired_mask: (
                    lambda idx: (
                        _auc_for_indices(scores, data["labels"], idx)
                        - _auc_for_indices(base_scores, data["labels"], idx)
                    )
                )(_masked_draw(draw, paired_mask)),
            )
            delta_block = _ci_summary(
                "delta_auroc",
                point_delta,
                delta_samples,
                family=f"paired delta AUROC {probe_id} vs base",
                require_half=math.isfinite(point_delta),
            )
            delta_block["n_paired"] = int(paired_mask.sum())

        probe_results[probe_id] = {
            "auroc_vs_gold": auc_block,
            "spearman_vs_base": spearman_block,
            "delta_auroc_vs_base": delta_block,
            "auroc_vs_gold_by_tag": _advisory_tag_aurocs(
                data, probe_id, draws
            ),
        }

    data["probe_results"] = probe_results
    rho = _rho_matrix(data)
    complete_mask = np.ones(n_rows, dtype=bool)
    for probe_id in data["probe_ids"]:
        complete_mask &= data["ok_masks"][probe_id]
    complete_indices = np.flatnonzero(complete_mask)
    complete_matrix = np.column_stack(
        [data["scalars"][probe_id][complete_indices] for probe_id in data["probe_ids"]]
    )
    point_ratios = _pc_ratios_svd(complete_matrix)
    point_pc1 = float(point_ratios[0]) if point_ratios is not None else float("nan")

    def pc1_for_draw(draw: np.ndarray) -> float:
        indices = _masked_draw(draw, complete_mask)
        matrix = np.column_stack(
            [data["scalars"][probe_id][indices] for probe_id in data["probe_ids"]]
        )
        ratios = _pc_ratios_svd(matrix)
        return float(ratios[0]) if ratios is not None else float("nan")

    pc_samples = _collect_bootstrap(draws, pc1_for_draw)
    pc1 = _ci_summary(
        "explained_variance_ratio",
        point_pc1,
        pc_samples,
        family="PC1 explained variance",
        require_half=point_ratios is not None,
    )
    pc1.update(
        {
            "n_rows_complete": int(complete_mask.sum()),
            "n_rows_dropped": int((~complete_mask).sum()),
            "n_components": (
                int(len(point_ratios)) if point_ratios is not None else 0
            ),
        }
    )
    if point_ratios is None:
        n_effective = None
    else:
        n_effective = int(
            np.searchsorted(np.cumsum(point_ratios), 0.95, side="left") + 1
        )

    for probe_id in data["probe_ids"]:
        artifact["probes"][probe_id].update(probe_results[probe_id])
    artifact["rho_matrix"] = rho
    artifact["pc1_explained_variance"] = pc1
    artifact["n_effective_dimensions"] = n_effective
    k1, k2, k3, verdict, reason = _apply_verdict(data=data, rho=rho, pc1=pc1)
    artifact["k1"] = k1
    artifact["k2"] = k2
    artifact["k3"] = k3
    artifact["verdict"] = verdict
    artifact["verdict_reason"] = reason
    return artifact


def _input_block(
    *,
    gold_path: Path,
    probes_path: Path,
    rows: Sequence[dict[str, Any]],
    subsampled: bool,
    sample_seed: int,
) -> dict[str, Any]:
    return {
        "gold_path": str(gold_path.resolve()),
        "gold_sha256": _sha256_file(gold_path),
        "probe_jsonl": str(probes_path.resolve()),
        "probe_jsonl_sha256": _sha256_file(probes_path),
        "n_rows": int(len(rows)),
        "n_clusters": int(len({str(row["pa_hash"]) for row in rows})),
        "subsampled": bool(subsampled),
        "seed": int(sample_seed),
    }


def _analyze_file(
    *, probes_path: Path, max_rows: int | None, sample_seed: int
) -> dict[str, Any]:
    gold_rows = _load_gold(GOLD)
    _manifest, probe_meta, records = _load_probe_jsonl(probes_path)
    joined = _join_records(gold_rows, records)
    # The use of --max-rows is provenance even when it happens to name the
    # complete current input, so stamp it as a sampling operation consistently.
    subsampled = max_rows is not None
    if max_rows is not None:
        if max_rows <= 0:
            raise AnalysisError("--max-rows must be a positive integer")
        if max_rows > len(joined):
            raise AnalysisError(
                f"--max-rows={max_rows} exceeds the {len(joined)} available rows"
            )
        joined = select_killgate_sample(
            joined, n_rows=max_rows, seed=sample_seed
        )
        joined.sort(key=lambda row: row["row_index"])
    inputs = _input_block(
        gold_path=GOLD,
        probes_path=probes_path,
        rows=joined,
        subsampled=subsampled,
        sample_seed=sample_seed,
    )
    return _analyze_joined(joined, probe_meta, inputs=inputs)


def _write_artifact(path: Path, artifact: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        artifact, indent=2, sort_keys=True, allow_nan=False
    ) + "\n"
    path.write_text(encoded, encoding="utf-8")


def _synthetic_meta() -> dict[str, dict[str, Any]]:
    return {
        "base": {"family": "polarity", "is_base": True},
        "probe_a": {"family": "synthetic", "is_base": False},
        "probe_b": {"family": "synthetic", "is_base": False},
        "probe_c": {"family": "synthetic", "is_base": False},
    }


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def _synthetic_rows(
    *, n_rows: int, n_clusters: int, mode: str, seed: int
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    labels = np.arange(n_rows) % 2 == 0
    signed = np.where(labels, 1.0, -1.0)
    if mode == "monotone":
        base = 1.4 * signed + rng.normal(0.0, 1.0, size=n_rows)
        deltas = {
            "base": base,
            "probe_a": 0.85 * base + rng.normal(0.0, 1e-7, size=n_rows),
            "probe_b": 1.15 * base + rng.normal(0.0, 1e-7, size=n_rows),
            "probe_c": 1.35 * base + rng.normal(0.0, 1e-7, size=n_rows),
        }
    elif mode == "independent":
        deltas = {
            "base": 0.30 * signed + rng.normal(0.0, 1.0, size=n_rows),
            "probe_a": 1.15 * signed + rng.normal(0.0, 1.0, size=n_rows),
            "probe_b": 1.00 * signed + rng.normal(0.0, 1.0, size=n_rows),
            "probe_c": 0.90 * signed + rng.normal(0.0, 1.0, size=n_rows),
        }
    else:
        raise AssertionError(f"unknown synthetic mode {mode!r}")

    rows: list[dict[str, Any]] = []
    for index in range(n_rows):
        probe_values = {}
        for probe_id, values in deltas.items():
            delta = float(values[index])
            probe_values[probe_id] = {
                "p_raw": _sigmoid(delta),
                "status": "ok",
                "both_observed": True,
                "precision_limited": False,
                "delta_logit": delta,
            }
        rows.append(
            {
                "row_index": index,
                "source_hash": str(100_000 + index),
                "pa_hash": f"cluster-{index % n_clusters:04d}",
                "tag": "correct" if labels[index] else "grounding",
                "gold_correct": bool(labels[index]),
                "elapsed_s": 5.0,
                "probes": probe_values,
            }
        )
    return rows


def _selftest_inputs(name: str, rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {
        "gold_path": f"<selftest:{name}:gold>",
        "gold_sha256": "0" * 64,
        "probe_jsonl": f"<selftest:{name}:probes>",
        "probe_jsonl_sha256": "0" * 64,
        "n_rows": len(rows),
        "n_clusters": len({str(row["pa_hash"]) for row in rows}),
        "subsampled": False,
        "seed": int(KILL_RULE["bootstrap_seed"]),
    }


def _selftest_hash_and_saturation() -> tuple[int, float]:
    deltas_base = np.asarray([-2.0, 0.0, 2.0, 14.5, 16.0])
    # Reverse the two saturated probe readings.  delta_logit retains that
    # ordering (rho=0.9), while clipping both probes creates tied top ranks
    # (rho=1.0), so this fixture detects a regression to the clipped scalar.
    deltas_probe = np.asarray([-1.0, 1.0, 3.0, 17.0, 15.5])
    gold_rows: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    meta = {
        "base": {"family": "polarity", "is_base": True},
        "probe": {"family": "synthetic", "is_base": False},
    }
    for index, (base_delta, probe_delta) in enumerate(
        zip(deltas_base, deltas_probe)
    ):
        tag = "correct" if index % 2 == 0 else "grounding"
        source_hash = 9000 + index
        pa_hash = 7000 + index
        gold_rows.append(
            {"source_hash": source_hash, "pa_hash": pa_hash, "tag": tag}
        )
        probes = {}
        for probe_id, delta in (("base", base_delta), ("probe", probe_delta)):
            probes[probe_id] = {
                "p_raw": _sigmoid(float(delta)),
                "status": "ok",
                "both_observed": True,
                "precision_limited": abs(float(delta)) > 10.0,
                "delta_logit": float(delta),
            }
        records.append(
            {
                "row_index": index,
                "source_hash": str(source_hash),
                "pa_hash": str(pa_hash),
                "gold_correct": bool(is_gold_correct(tag)),
                "elapsed_s": None,
                "probes": probes,
            }
        )
    manifest = {"_manifest": True, "split": "fit", "probe_meta": meta}
    checked_meta, checked_records = _validate_probe_parts(
        manifest, records, source="selftest saturation fixture"
    )
    joined = _join_records(gold_rows, checked_records)
    prepared = _prepare_data(joined, checked_meta)
    rho = _rho_matrix(prepared)
    expected = _spearman(deltas_base, deltas_probe)
    realized = rho["values"][0][1]
    if realized is None or abs(float(realized) - expected) > 1e-12:
        raise AssertionError("delta_logit rho did not preserve the unsaturated ranking")
    if not np.array_equal(prepared["scalars"]["base"], deltas_base):
        raise AssertionError("base scalar did not use delta_logit")
    if not np.array_equal(prepared["scalars"]["probe"], deltas_probe):
        raise AssertionError("probe scalar did not use delta_logit")
    clipped_rho = _spearman(
        np.asarray([_logit_clipped(_sigmoid(float(x))) for x in deltas_base]),
        np.asarray([_logit_clipped(_sigmoid(float(x))) for x in deltas_probe]),
    )
    if abs(expected - clipped_rho) <= 1e-6:
        raise AssertionError("saturation fixture does not distinguish clipped scalars")
    n_would_clip = sum(
        block["n_would_clip"] for block in prepared["accounting"].values()
    )
    if n_would_clip <= 0:
        raise AssertionError("saturation fixture did not exercise clipping")

    mismatched = [dict(record) for record in checked_records]
    mismatched[0] = dict(mismatched[0])
    mismatched[0]["source_hash"] = "genuinely-wrong"
    try:
        _join_records(gold_rows, mismatched)
    except AnalysisError as exc:
        message = str(exc)
        if "row_index 0" not in message or "source_hash" not in message:
            raise AssertionError("hash mismatch did not name row_index and field") from exc
    else:
        raise AssertionError("genuinely mismatched source_hash was accepted")
    return n_would_clip, float(realized)


def _run_selftest() -> int:
    meta = _synthetic_meta()
    monotone_rows = _synthetic_rows(
        n_rows=600, n_clusters=450, mode="monotone", seed=11
    )
    monotone = _analyze_joined(
        monotone_rows,
        meta,
        inputs=_selftest_inputs("monotone", monotone_rows),
        generated_utc="2000-01-01T00:00:00+00:00",
    )
    assert monotone["powered"] is True
    assert monotone["verdict"] == "NO-GO"
    print("selftest monotone-copy: powered=true verdict=NO-GO PASS")

    independent_rows = _synthetic_rows(
        n_rows=600, n_clusters=450, mode="independent", seed=29
    )
    independent = _analyze_joined(
        independent_rows,
        meta,
        inputs=_selftest_inputs("independent", independent_rows),
        generated_utc="2000-01-01T00:00:00+00:00",
    )
    assert independent["powered"] is True
    assert independent["verdict"] == "GO"
    print("selftest independent-signal: powered=true verdict=GO PASS")

    underpowered_rows = _synthetic_rows(
        n_rows=120, n_clusters=100, mode="independent", seed=31
    )
    underpowered = _analyze_joined(
        underpowered_rows,
        meta,
        inputs=_selftest_inputs("underpowered", underpowered_rows),
        generated_utc="2000-01-01T00:00:00+00:00",
    )
    simulated_exit = 0 if underpowered["powered"] else 1
    assert underpowered["powered"] is False
    assert underpowered["verdict"] is None
    assert simulated_exit != 0
    print("selftest underpowered: powered=false verdict=null simulated_exit=1 PASS")

    n_would_clip, rho = _selftest_hash_and_saturation()
    print(
        "selftest normalized-hash+saturation: "
        f"n_would_clip={n_would_clip} rho={rho:.12g} rank_identity=true PASS"
    )
    print("SELFTEST PASS (4/4)")
    return 0


def _required_artifact_keys() -> set[str]:
    return {
        "schema_version",
        "generated_utc",
        "split",
        "in_sample",
        "inputs",
        "rule",
        "rule_sha256",
        "scalar_source",
        "scalar_note",
        "powered",
        "power",
        "power_failure",
        "base_probe_id",
        "probes",
        "rho_matrix",
        "pc1_explained_variance",
        "n_effective_dimensions",
        "cost",
        "k1",
        "k2",
        "k3",
        "verdict",
        "verdict_reason",
    }


def _require_artifact_object(
    value: Any, *, path: str, fields: Sequence[str] = ()
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise AnalysisError(f"artifact schema incomplete: {path} must be an object")
    missing = sorted(set(fields) - set(value))
    if missing:
        raise AnalysisError(
            f"artifact schema incomplete: {path} missing {missing[0]!r}"
        )
    return value


def _load_artifact_for_assert(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise AnalysisError(f"artifact is missing: {path}")
    if path.stat().st_size == 0:
        raise AnalysisError(f"artifact is empty: {path}")
    try:
        artifact = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise AnalysisError(f"artifact is invalid JSON: {exc.msg}") from exc
    if not isinstance(artifact, dict):
        raise AnalysisError("artifact root must be an object")
    missing = sorted(_required_artifact_keys() - set(artifact))
    if missing:
        raise AnalysisError(f"artifact schema incomplete: missing {missing[0]!r}")
    if artifact["schema_version"] != SCHEMA_VERSION:
        raise AnalysisError(
            f"artifact schema_version mismatch: {artifact['schema_version']!r}"
        )
    if artifact["split"] != "fit" or artifact["in_sample"] is not True:
        raise AnalysisError("artifact must carry split='fit' and in_sample=true")
    if artifact["rule_sha256"] != KILL_RULE_SHA_FROZEN:
        raise AnalysisError("artifact rule_sha256 does not match the frozen rule sha")
    if artifact["rule"] != KILL_RULE:
        raise AnalysisError("artifact rule does not deep-equal KILL_RULE")
    derived = hashlib.sha256(
        json.dumps(artifact["rule"], sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if derived != KILL_RULE_SHA_FROZEN:
        raise AnalysisError("artifact rule content does not derive the frozen rule sha")
    return artifact


def _independent_prepare(
    rows: Sequence[dict[str, Any]],
    probe_meta: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Rebuild scalar arrays and accounting without primary-path helpers."""
    from scipy.stats import rankdata

    if not rows:
        raise AnalysisError("independent recompute has no selected rows")
    probe_ids = sorted(probe_meta)
    base_ids = [
        probe_id for probe_id in probe_ids if probe_meta[probe_id]["is_base"] is True
    ]
    if len(base_ids) != 1:
        raise AnalysisError(
            "independent recompute requires exactly one is_base: true probe"
        )
    labels = np.asarray(
        [bool(is_gold_correct(str(row["tag"]))) for row in rows], dtype=bool
    )
    clusters = np.asarray([str(row["pa_hash"]) for row in rows], dtype=object)
    tags = np.asarray([str(row["tag"]) for row in rows], dtype=object)
    scalars: dict[str, np.ndarray] = {}
    ok_masks: dict[str, np.ndarray] = {}
    accounting: dict[str, dict[str, Any]] = {}

    for probe_id in probe_ids:
        scores = np.full(len(rows), np.nan, dtype=float)
        ok = np.zeros(len(rows), dtype=bool)
        statuses: Counter[str] = Counter()
        n_both = 0
        n_limited = 0
        n_would_clip = 0
        delta_for_rank: list[float] = []
        logit_for_rank: list[float] = []
        for row_number, row in enumerate(rows):
            reading = row["probes"][probe_id]
            status = str(reading["status"])
            statuses[status] += 1
            n_both += int(reading["both_observed"] is True)
            n_limited += int(reading["precision_limited"] is True)
            raw = reading["p_raw"]
            delta_value = reading.get("delta_logit")
            clipped_value: float | None = None
            if raw is not None:
                probability = float(raw)
                bounded = min(1.0 - 1e-6, max(1e-6, probability))
                clipped_value = math.log(bounded / (1.0 - bounded))
                if delta_value is not None:
                    delta = float(delta_value)
                    if abs(clipped_value - delta) > 1e-9:
                        n_would_clip += 1
                    if 1e-6 <= probability <= 1.0 - 1e-6:
                        delta_for_rank.append(delta)
                        logit_for_rank.append(
                            math.log(probability / (1.0 - probability))
                        )
            if status != "ok":
                continue
            scores[row_number] = (
                float(delta_value)
                if delta_value is not None
                else float(clipped_value)
            )
            ok[row_number] = True

        rank_identity = True
        if delta_for_rank:
            delta_array = np.asarray(delta_for_rank, dtype=float)
            logit_array = np.asarray(logit_for_rank, dtype=float)
            logit_array = _snap_logit_roundoff(delta_array, logit_array)
            rank_identity = bool(
                np.array_equal(
                    rankdata(delta_array, method="average"),
                    rankdata(logit_array, method="average"),
                )
            )
        if not rank_identity:
            raise AnalysisError(
                f"independent recompute found non-identical non-clipped ranks "
                f"for probe {probe_id!r}"
            )
        scalars[probe_id] = scores
        ok_masks[probe_id] = ok
        accounting[probe_id] = {
            "family": probe_meta[probe_id]["family"],
            "is_base": bool(probe_meta[probe_id]["is_base"]),
            "n_ok": int(ok.sum()),
            "n_status_not_ok": {
                status: int(count)
                for status, count in sorted(statuses.items())
                if status != "ok"
            },
            "frac_both_observed": float(n_both / len(rows)),
            "frac_precision_limited": float(n_limited / len(rows)),
            "n_distinct_values": int(len(np.unique(scores[ok]))),
            "n_would_clip": int(n_would_clip),
            "non_clipped_rank_identity": rank_identity,
            "n_non_clipped_rank_compared": int(len(delta_for_rank)),
        }

    timings = [row.get("elapsed_s") for row in rows]
    observed_timings = [float(value) for value in timings if value is not None]
    timing_complete = len(observed_timings) == len(rows)
    cost = {
        "mean_elapsed_s": (
            float(np.mean(observed_timings))
            if timing_complete and observed_timings
            else None
        ),
        "median_elapsed_s": (
            float(np.median(observed_timings))
            if timing_complete and observed_timings
            else None
        ),
        "n_timed": int(len(observed_timings)),
        "n_missing": int(len(rows) - len(observed_timings)),
    }
    return {
        "rows": list(rows),
        "probe_ids": probe_ids,
        "base_id": base_ids[0],
        "labels": labels,
        "clusters": clusters,
        "tags": tags,
        "scalars": scalars,
        "ok_masks": ok_masks,
        "accounting": accounting,
        "cost": cost,
    }


def _independent_cluster_draws(
    clusters: np.ndarray, *, n_boot: int, seed: int
) -> list[np.ndarray]:
    # Separate implementation for the gate: the RNG call sequence intentionally
    # matches the primary path, one group-index vector per resample.
    grouped: dict[str, list[int]] = {}
    for row_number in range(len(clusters)):
        grouped.setdefault(str(clusters[row_number]), []).append(row_number)
    names = sorted(grouped.keys())
    group_rows = [np.asarray(grouped[name], dtype=np.int64) for name in names]
    generator = np.random.default_rng(seed)
    output: list[np.ndarray] = []
    for _resample in range(n_boot):
        choices = generator.integers(0, len(names), size=len(names))
        output.append(np.concatenate([group_rows[int(choice)] for choice in choices]))
    return output


def _independent_recompute(
    data: dict[str, Any], draws: Sequence[np.ndarray]
) -> dict[str, Any]:
    # These imports are deliberately lazy and exclusive to --assert-artifact.
    from scipy.stats import spearmanr
    from sklearn.decomposition import PCA
    from sklearn.metrics import roc_auc_score

    labels = data["labels"]
    base_id = data["base_id"]
    base_scores = data["scalars"][base_id]
    base_mask = data["ok_masks"][base_id]
    n_boot = int(KILL_RULE["bootstrap_resamples"])

    def sk_auc(scores: np.ndarray, indices: np.ndarray) -> float:
        if indices.size < 2:
            return float("nan")
        y = labels[indices]
        if bool(y.all()) or bool((~y).all()):
            return float("nan")
        return float(roc_auc_score(y, scores[indices]))

    def scipy_rho(left: np.ndarray, right: np.ndarray) -> float:
        if (
            left.size < 2
            or right.size != left.size
            or len(np.unique(left)) <= 1
            or len(np.unique(right)) <= 1
        ):
            return float("nan")
        return float(spearmanr(left, right).statistic)

    def collect(function: Callable[[np.ndarray], float]) -> np.ndarray:
        results: list[float] = []
        for bootstrap_rows in draws:
            result = float(function(bootstrap_rows))
            if math.isfinite(result):
                results.append(result)
        return np.asarray(results, dtype=float)

    def summary(point: float, samples: np.ndarray) -> dict[str, Any]:
        normalized_point: float | None = (
            float(point) if math.isfinite(point) else None
        )
        if len(samples) < n_boot // 2:
            if normalized_point is not None:
                raise AnalysisError(
                    "numeric disagreement: only "
                    f"{len(samples)}/{n_boot} independent cluster-bootstrap "
                    "resamples were valid for a defined point statistic"
                )
            return {
                "point": normalized_point,
                "low": None,
                "high": None,
                "n_valid": int(len(samples)),
                "n_bootstrap": n_boot,
                "seed": int(KILL_RULE["bootstrap_seed"]),
            }
        low, high = np.percentile(samples, [2.5, 97.5])
        return {
            "point": normalized_point,
            "low": float(low),
            "high": float(high),
            "n_valid": int(len(samples)),
            "n_bootstrap": n_boot,
            "seed": int(KILL_RULE["bootstrap_seed"]),
        }

    base_indices = np.flatnonzero(base_mask)
    base_point = sk_auc(base_scores, base_indices)
    base_samples = collect(
        lambda draw: sk_auc(base_scores, draw[base_mask[draw]])
    )
    n_rows = int(len(labels))
    n_clusters = int(len({str(value) for value in data["clusters"]}))
    n_positive = int(labels.sum())
    class_imbalance = float(abs(n_positive - (n_rows - n_positive)) / n_rows)
    base_se = (
        float(np.std(base_samples, ddof=1)) if len(base_samples) >= 2 else None
    )
    power = {
        "n_rows": n_rows,
        "n_clusters": n_clusters,
        "class_imbalance": class_imbalance,
        "base_auroc": float(base_point) if math.isfinite(base_point) else None,
        "base_se": base_se,
        "base_n_valid": int(len(base_samples)),
        "n_bootstrap": n_boot,
        "seed": int(KILL_RULE["bootstrap_seed"]),
        "thresholds": {
            "power_min_rows": int(KILL_RULE["power_min_rows"]),
            "power_min_clusters": int(KILL_RULE["power_min_clusters"]),
            "power_max_class_imbalance": float(
                KILL_RULE["power_max_class_imbalance"]
            ),
            "power_max_base_auroc_se": float(
                KILL_RULE["power_max_base_auroc_se"]
            ),
        },
    }
    power_failures: list[str] = []
    if n_rows < int(KILL_RULE["power_min_rows"]):
        power_failures.append("power_min_rows")
    if n_clusters < int(KILL_RULE["power_min_clusters"]):
        power_failures.append("power_min_clusters")
    if class_imbalance > float(KILL_RULE["power_max_class_imbalance"]):
        power_failures.append("power_max_class_imbalance")
    if (
        base_se is None
        or not math.isfinite(base_se)
        or len(base_samples) < n_boot // 2
        or base_se > float(KILL_RULE["power_max_base_auroc_se"])
    ):
        power_failures.append("power_max_base_auroc_se")
    power["failures"] = power_failures
    power["powered"] = not power_failures

    rho_values: list[list[float | None]] = []
    rho_counts: list[list[int]] = []
    for left_id in data["probe_ids"]:
        row: list[float | None] = []
        count_row: list[int] = []
        for right_id in data["probe_ids"]:
            mask = data["ok_masks"][left_id] & data["ok_masks"][right_id]
            value = scipy_rho(
                data["scalars"][left_id][mask], data["scalars"][right_id][mask]
            )
            row.append(float(value) if math.isfinite(value) else None)
            count_row.append(int(mask.sum()))
        rho_values.append(row)
        rho_counts.append(count_row)

    probes: dict[str, Any] = {}
    for probe_id in data["probe_ids"]:
        scores = data["scalars"][probe_id]
        ok = data["ok_masks"][probe_id]
        auc_point = sk_auc(scores, np.flatnonzero(ok))
        if probe_id == base_id:
            auc_samples = base_samples
        else:
            auc_samples = collect(lambda draw, scores=scores, ok=ok: sk_auc(scores, draw[ok[draw]]))
        result: dict[str, Any] = {"auroc": summary(auc_point, auc_samples)}
        if probe_id != base_id:
            paired = ok & base_mask
            point_rows = np.flatnonzero(paired)
            rho_point = scipy_rho(scores[point_rows], base_scores[point_rows])
            rho_samples = collect(
                lambda draw, scores=scores, paired=paired: (
                    lambda idx: scipy_rho(scores[idx], base_scores[idx])
                )(draw[paired[draw]])
            )
            rho_summary = summary(rho_point, rho_samples)
            rho_summary["ci_low_abs"] = (
                float(np.percentile(np.abs(rho_samples), 2.5))
                if len(rho_samples) >= n_boot // 2
                else None
            )
            rho_summary["n_paired"] = int(paired.sum())
            delta_point = sk_auc(scores, point_rows) - sk_auc(base_scores, point_rows)
            delta_samples = collect(
                lambda draw, scores=scores, paired=paired: (
                    lambda idx: sk_auc(scores, idx) - sk_auc(base_scores, idx)
                )(draw[paired[draw]])
            )
            result["rho"] = rho_summary
            result["delta"] = summary(delta_point, delta_samples)
            result["n_paired"] = int(paired.sum())

        advisory: dict[str, Any] = {}
        for tag in sorted(
            {str(value) for value in data["tags"] if not is_gold_correct(str(value))}
        ):
            tag_mask = ok & ((data["tags"] == "correct") | (data["tags"] == tag))
            tag_point = sk_auc(scores, np.flatnonzero(tag_mask))
            tag_samples = collect(
                lambda draw, scores=scores, tag_mask=tag_mask: sk_auc(
                    scores, draw[tag_mask[draw]]
                )
            )
            advisory[tag] = summary(tag_point, tag_samples)
            advisory[tag]["n_rows"] = int(tag_mask.sum())
        result["advisory"] = advisory
        probes[probe_id] = result

    complete = np.ones(len(labels), dtype=bool)
    for probe_id in data["probe_ids"]:
        complete &= data["ok_masks"][probe_id]

    def sklearn_ratios(indices: np.ndarray) -> np.ndarray | None:
        if indices.size < 2:
            return None
        matrix = np.column_stack(
            [data["scalars"][probe_id][indices] for probe_id in data["probe_ids"]]
        )
        std = matrix.std(axis=0, ddof=0)
        scale = np.where(std > 0.0, std, 1.0)
        standardized = (matrix - matrix.mean(axis=0)) / scale
        if not np.any(np.var(standardized, axis=0) > 0.0):
            return None
        return np.asarray(
            PCA(svd_solver="full").fit(standardized).explained_variance_ratio_,
            dtype=float,
        )

    point_ratios = sklearn_ratios(np.flatnonzero(complete))
    point_pc1 = float(point_ratios[0]) if point_ratios is not None else float("nan")
    pc_samples = collect(
        lambda draw: (
            lambda ratios: float(ratios[0]) if ratios is not None else float("nan")
        )(sklearn_ratios(draw[complete[draw]]))
    )
    pc = summary(point_pc1, pc_samples)
    pc.update(
        {
            "n_rows_complete": int(complete.sum()),
            "n_rows_dropped": int((~complete).sum()),
            "n_components": (
                int(len(point_ratios)) if point_ratios is not None else 0
            ),
        }
    )
    n_effective = (
        int(np.searchsorted(np.cumsum(point_ratios), 0.95, side="left") + 1)
        if point_ratios is not None
        else None
    )

    nonbase = [probe_id for probe_id in data["probe_ids"] if probe_id != base_id]
    k1_threshold = float(
        KILL_RULE["k1_max_probe_spearman_vs_base_ci_low_abs"]
    )
    k1_each: list[bool] = []
    k1_per_probe: dict[str, Any] = {}
    k1_nonconstant: list[float] = []
    for probe_id in nonbase:
        if data["accounting"][probe_id]["n_distinct_values"] <= 1:
            realized = None
            satisfied = True
            reason = "zero_variance"
        else:
            realized = probes[probe_id]["rho"]["ci_low_abs"]
            satisfied = realized is not None and realized >= k1_threshold
            reason = None
            if realized is not None:
                k1_nonconstant.append(float(realized))
        k1_each.append(bool(satisfied))
        k1_per_probe[probe_id] = {
            "realized_ci_low_abs": realized,
            "threshold": k1_threshold,
            "boolean": bool(satisfied),
            "redundant_reason": reason,
        }
    k1_boolean = all(k1_each)
    k1 = {
        "realized_min_ci_low_abs": (
            min(k1_nonconstant) if k1_nonconstant else None
        ),
        "threshold": k1_threshold,
        "boolean": bool(k1_boolean),
        "per_probe": k1_per_probe,
    }
    k2_threshold = float(KILL_RULE["k2_max_delta_auroc_vs_base_ci_high"])
    delta_by_probe = {
        probe_id: probes[probe_id]["delta"]["high"] for probe_id in nonbase
    }
    delta_highs = list(delta_by_probe.values())
    k2_realized = (
        max(float(value) for value in delta_highs)
        if all(value is not None for value in delta_highs)
        else None
    )
    k2_boolean = (
        k2_realized is not None and k2_realized <= k2_threshold
    )
    k2 = {
        "realized_max_ci95_high": k2_realized,
        "threshold": k2_threshold,
        "boolean": bool(k2_boolean),
        "per_probe_ci95_high": delta_by_probe,
    }
    offdiag: list[float] = []
    undefined = False
    for i in range(len(rho_values)):
        for j in range(i + 1, len(rho_values)):
            value = rho_values[i][j]
            if value is None:
                undefined = True
            else:
                offdiag.append(abs(float(value)))
    min_abs = None if undefined or not offdiag else min(offdiag)
    pc_threshold = float(KILL_RULE["k3_pc1_explained_variance_ci_low"])
    rho_threshold = float(KILL_RULE["k3_min_abs_pairwise_spearman"])
    k3_boolean = (
        pc["low"] is not None
        and pc["low"] >= pc_threshold
        and min_abs is not None
        and min_abs >= rho_threshold
    )
    k3 = {
        "realized_pc1_ci95_low": pc["low"],
        "pc1_threshold": pc_threshold,
        "realized_min_abs_pairwise_spearman": min_abs,
        "spearman_threshold": rho_threshold,
        "boolean": bool(k3_boolean),
    }
    verdict = "NO-GO" if ((k1_boolean and k2_boolean) or k3_boolean) else "GO"
    if k1_boolean and k2_boolean and k3_boolean:
        verdict_reason = "NO-GO: K1 and K2 fired, and K3 also fired."
    elif k1_boolean and k2_boolean:
        verdict_reason = "NO-GO: K1 and K2 fired; K3 did not fire."
    elif k3_boolean:
        verdict_reason = (
            "NO-GO: K3 fired; the combined K1-and-K2 condition did not fire."
        )
    else:
        verdict_reason = (
            "GO: neither the combined K1-and-K2 condition nor K3 fired."
        )
    return {
        "power": power,
        "rho_values": rho_values,
        "rho_counts": rho_counts,
        "probes": probes,
        "pc": pc,
        "n_effective": n_effective,
        "k1": k1,
        "k2": k2,
        "k3": k3,
        "verdict": verdict,
        "verdict_reason": verdict_reason,
    }


def _numeric_delta(
    expected: Any,
    realized: Any,
    *,
    path: str,
    tolerance: float,
) -> float:
    if expected is None or realized is None:
        if expected is realized:
            return 0.0
        raise AnalysisError(
            f"numeric disagreement at {path}: artifact={expected!r}, "
            f"recomputed={realized!r}"
        )
    delta = abs(float(expected) - float(realized))
    if not math.isfinite(delta) or delta > tolerance:
        raise AnalysisError(
            f"numeric disagreement at {path}: |delta|={delta:.12g} > {tolerance}"
        )
    return delta


def _assert_artifact() -> int:
    artifact = _load_artifact_for_assert(DEFAULT_ARTIFACT)
    inputs = artifact["inputs"]
    required_inputs = {
        "gold_path",
        "gold_sha256",
        "probe_jsonl",
        "probe_jsonl_sha256",
        "n_rows",
        "n_clusters",
        "subsampled",
        "seed",
    }
    if not isinstance(inputs, dict):
        raise AnalysisError("artifact schema incomplete: 'inputs' must be an object")
    missing_inputs = sorted(required_inputs - set(inputs))
    if missing_inputs:
        raise AnalysisError(
            f"artifact schema incomplete: inputs missing {missing_inputs[0]!r}"
        )
    gold_path = Path(inputs["gold_path"])
    probes_path = Path(inputs["probe_jsonl"])
    for path, recorded, name in (
        (gold_path, inputs["gold_sha256"], "gold_path"),
        (probes_path, inputs["probe_jsonl_sha256"], "probe_jsonl"),
    ):
        if not path.is_file():
            raise AnalysisError(f"input sha drift: {name} is missing: {path}")
        current = _sha256_file(path)
        if current != recorded:
            raise AnalysisError(
                f"input sha drift for {name}: recorded={recorded}, current={current}"
            )
    if artifact["powered"] is not True:
        raise AnalysisError("artifact powered == false; no verdict can be reproduced")
    if artifact["verdict"] not in {"GO", "NO-GO"}:
        raise AnalysisError("artifact verdict must be GO or NO-GO when powered")
    if artifact["scalar_source"] != SCALAR_SOURCE:
        raise AnalysisError("artifact scalar_source does not match the frozen method")
    if artifact["scalar_note"] != SCALAR_NOTE:
        raise AnalysisError("artifact scalar_note does not match the saturation note")

    gold_rows = _load_gold(gold_path)
    _manifest, probe_meta, records = _load_probe_jsonl(probes_path)
    joined = _join_records(gold_rows, records)
    if inputs["subsampled"] is True:
        joined = select_killgate_sample(
            joined, n_rows=int(inputs["n_rows"]), seed=int(inputs["seed"])
        )
        joined.sort(key=lambda row: row["row_index"])
    elif len(joined) != int(inputs["n_rows"]):
        raise AnalysisError(
            "numeric disagreement: non-subsampled input row count changed"
        )
    data = _independent_prepare(joined, probe_meta)
    if len(joined) != int(inputs["n_rows"]):
        raise AnalysisError("numeric disagreement: selected n_rows changed")
    n_clusters = len({str(value) for value in data["clusters"]})
    if n_clusters != int(inputs["n_clusters"]):
        raise AnalysisError("numeric disagreement: selected n_clusters changed")
    if artifact["base_probe_id"] != data["base_id"]:
        raise AnalysisError("numeric disagreement: base_probe_id changed")
    if not isinstance(artifact["probes"], dict):
        raise AnalysisError("artifact schema incomplete: probes must be an object")
    if set(artifact["probes"]) != set(data["probe_ids"]):
        raise AnalysisError("numeric disagreement: artifact probe ids changed")

    stored_power = _require_artifact_object(
        artifact["power"],
        path="power",
        fields=(
            "n_rows",
            "n_clusters",
            "class_imbalance",
            "base_auroc",
            "base_auroc_cluster_bootstrap_se",
            "base_auroc_n_valid_resamples",
            "n_bootstrap",
            "seed",
            "thresholds",
        ),
    )
    if stored_power["n_bootstrap"] != int(KILL_RULE["bootstrap_resamples"]):
        raise AnalysisError("numeric disagreement: power.n_bootstrap")
    if stored_power["seed"] != int(KILL_RULE["bootstrap_seed"]):
        raise AnalysisError("numeric disagreement: power.seed")
    draws = _independent_cluster_draws(
        data["clusters"],
        n_boot=int(stored_power["n_bootstrap"]),
        seed=int(stored_power["seed"]),
    )
    recomputed = _independent_recompute(data, draws)
    maxima = {
        "power": 0.0,
        "auroc_point": 0.0,
        "auroc_ci": 0.0,
        "rho_point": 0.0,
        "rho_ci": 0.0,
        "delta_auroc_point": 0.0,
        "delta_auroc_ci": 0.0,
        "pc1_point": 0.0,
        "pc1_ci": 0.0,
        "accounting": 0.0,
        "cost": 0.0,
        "gate": 0.0,
    }

    def compare(
        family: str,
        expected: Any,
        realized: Any,
        *,
        path: str,
        tolerance: float,
    ) -> None:
        maxima[family] = max(
            maxima[family],
            _numeric_delta(
                expected, realized, path=path, tolerance=tolerance
            ),
        )

    def compare_bootstrap_metadata(
        stored: dict[str, Any], calculated: dict[str, Any], *, path: str
    ) -> None:
        for stored_name, calculated_name in (
            ("n_valid_resamples", "n_valid"),
            ("n_bootstrap", "n_bootstrap"),
            ("seed", "seed"),
        ):
            if stored.get(stored_name) != calculated.get(calculated_name):
                raise AnalysisError(
                    f"numeric disagreement: {path}.{stored_name}"
                )

    for field in ("n_rows", "n_clusters", "n_bootstrap", "seed"):
        if stored_power.get(field) != recomputed["power"][field]:
            raise AnalysisError(f"numeric disagreement: power.{field}")
    compare(
        "power",
        stored_power.get("class_imbalance"),
        recomputed["power"]["class_imbalance"],
        path="power.class_imbalance",
        tolerance=1e-8,
    )
    if stored_power.get("thresholds") != recomputed["power"]["thresholds"]:
        raise AnalysisError("numeric disagreement: power.thresholds")
    if recomputed["power"]["powered"] is not True:
        failures = ", ".join(recomputed["power"]["failures"])
        raise AnalysisError(
            f"powered disagreement: independent power gate failed ({failures})"
        )
    if artifact["power_failure"] != []:
        raise AnalysisError("powered artifact has a non-empty power_failure list")
    compare(
        "power",
        stored_power.get("base_auroc"),
        recomputed["power"]["base_auroc"],
        path="power.base_auroc",
        tolerance=1e-8,
    )
    compare(
        "power",
        stored_power.get("base_auroc_cluster_bootstrap_se"),
        recomputed["power"]["base_se"],
        path="power.base_auroc_cluster_bootstrap_se",
        tolerance=1e-8,
    )
    if (
        stored_power.get("base_auroc_n_valid_resamples")
        != recomputed["power"]["base_n_valid"]
    ):
        raise AnalysisError("numeric disagreement: base AUROC valid-resample count")

    rho_artifact = artifact["rho_matrix"]
    rho_artifact = _require_artifact_object(
        rho_artifact,
        path="rho_matrix",
        fields=("probe_ids", "values", "n_pairwise"),
    )
    if rho_artifact.get("probe_ids") != data["probe_ids"]:
        raise AnalysisError("numeric disagreement: rho_matrix probe_ids changed")
    matrix_size = len(data["probe_ids"])
    for field in ("values", "n_pairwise"):
        matrix = rho_artifact[field]
        if (
            not isinstance(matrix, list)
            or len(matrix) != matrix_size
            or any(not isinstance(row, list) or len(row) != matrix_size for row in matrix)
        ):
            raise AnalysisError(
                f"artifact schema incomplete: rho_matrix.{field} must be square"
            )
    for i, probe_id in enumerate(data["probe_ids"]):
        for j, other_id in enumerate(data["probe_ids"]):
            compare(
                "rho_point",
                rho_artifact["values"][i][j],
                recomputed["rho_values"][i][j],
                path=f"rho_matrix[{probe_id}][{other_id}]",
                tolerance=1e-8,
            )
            if rho_artifact["n_pairwise"][i][j] != recomputed["rho_counts"][i][j]:
                raise AnalysisError(
                    f"numeric disagreement: rho pair count {probe_id}/{other_id}"
                )

    for probe_id in data["probe_ids"]:
        stored = _require_artifact_object(
            artifact["probes"][probe_id],
            path=f"probes.{probe_id}",
            fields=(
                "family",
                "is_base",
                "n_ok",
                "n_status_not_ok",
                "frac_both_observed",
                "frac_precision_limited",
                "n_distinct_values",
                "n_would_clip",
                "non_clipped_rank_identity",
                "n_non_clipped_rank_compared",
                "auroc_vs_gold",
                "spearman_vs_base",
                "delta_auroc_vs_base",
                "auroc_vs_gold_by_tag",
            ),
        )
        calculated = recomputed["probes"][probe_id]
        for field in (
            "family",
            "is_base",
            "n_ok",
            "n_status_not_ok",
            "n_distinct_values",
            "n_would_clip",
            "non_clipped_rank_identity",
            "n_non_clipped_rank_compared",
        ):
            if stored[field] != data["accounting"][probe_id][field]:
                raise AnalysisError(
                    f"numeric disagreement: probes.{probe_id}.{field}"
                )
        for field in ("frac_both_observed", "frac_precision_limited"):
            compare(
                "accounting",
                stored[field],
                data["accounting"][probe_id][field],
                path=f"probes.{probe_id}.{field}",
                tolerance=1e-8,
            )
        stored_auc = _require_artifact_object(
            stored["auroc_vs_gold"],
            path=f"probes.{probe_id}.auroc_vs_gold",
            fields=(
                "auroc",
                "ci95_low",
                "ci95_high",
                "n_valid_resamples",
                "n_bootstrap",
                "seed",
            ),
        )
        independent_auc = calculated["auroc"]
        compare_bootstrap_metadata(
            stored_auc,
            independent_auc,
            path=f"probes.{probe_id}.auroc_vs_gold",
        )
        compare(
            "auroc_point",
            stored_auc["auroc"],
            independent_auc["point"],
            path=f"probes.{probe_id}.auroc_vs_gold.auroc",
            tolerance=1e-8,
        )
        for stored_key, other_key in (("ci95_low", "low"), ("ci95_high", "high")):
            compare(
                "auroc_ci",
                stored_auc[stored_key],
                independent_auc[other_key],
                path=f"probes.{probe_id}.auroc_vs_gold.{stored_key}",
                tolerance=5e-3,
            )
        if probe_id != data["base_id"]:
            stored_rho = _require_artifact_object(
                stored["spearman_vs_base"],
                path=f"probes.{probe_id}.spearman_vs_base",
                fields=(
                    "spearman",
                    "ci95_low",
                    "ci95_high",
                    "ci_low_abs",
                    "n_valid_resamples",
                    "n_bootstrap",
                    "seed",
                    "n_paired",
                ),
            )
            independent_rho = calculated["rho"]
            compare_bootstrap_metadata(
                stored_rho,
                independent_rho,
                path=f"probes.{probe_id}.spearman_vs_base",
            )
            compare(
                "rho_point",
                stored_rho["spearman"],
                independent_rho["point"],
                path=f"probes.{probe_id}.spearman_vs_base.spearman",
                tolerance=1e-8,
            )
            for stored_key, other_key in (
                ("ci95_low", "low"),
                ("ci95_high", "high"),
                ("ci_low_abs", "ci_low_abs"),
            ):
                compare(
                    "rho_ci",
                    stored_rho[stored_key],
                    independent_rho[other_key],
                    path=f"probes.{probe_id}.spearman_vs_base.{stored_key}",
                    tolerance=5e-3,
                )
            if stored_rho.get("n_paired") != independent_rho["n_paired"]:
                raise AnalysisError(
                    f"numeric disagreement: probes.{probe_id}."
                    "spearman_vs_base.n_paired"
                )
            stored_delta = _require_artifact_object(
                stored["delta_auroc_vs_base"],
                path=f"probes.{probe_id}.delta_auroc_vs_base",
                fields=(
                    "delta_auroc",
                    "ci95_low",
                    "ci95_high",
                    "n_valid_resamples",
                    "n_bootstrap",
                    "seed",
                    "n_paired",
                ),
            )
            independent_delta = calculated["delta"]
            compare_bootstrap_metadata(
                stored_delta,
                independent_delta,
                path=f"probes.{probe_id}.delta_auroc_vs_base",
            )
            compare(
                "delta_auroc_point",
                stored_delta["delta_auroc"],
                independent_delta["point"],
                path=f"probes.{probe_id}.delta_auroc_vs_base.delta_auroc",
                tolerance=1e-8,
            )
            for stored_key, other_key in (("ci95_low", "low"), ("ci95_high", "high")):
                compare(
                    "delta_auroc_ci",
                    stored_delta[stored_key],
                    independent_delta[other_key],
                    path=f"probes.{probe_id}.delta_auroc_vs_base.{stored_key}",
                    tolerance=5e-3,
                )
            if stored_delta["n_paired"] != calculated["n_paired"]:
                raise AnalysisError(
                    f"numeric disagreement: probes.{probe_id}.n_paired"
                )
        else:
            if stored["spearman_vs_base"] is not None:
                raise AnalysisError(
                    "artifact schema incomplete: base spearman_vs_base must be null"
                )
            if stored["delta_auroc_vs_base"] is not None:
                raise AnalysisError(
                    "artifact schema incomplete: base delta_auroc_vs_base must be null"
                )
        stored_advisory = _require_artifact_object(
            stored["auroc_vs_gold_by_tag"],
            path=f"probes.{probe_id}.auroc_vs_gold_by_tag",
            fields=("advisory", "definition", "tags"),
        )
        advisory = stored_advisory["tags"]
        if not isinstance(advisory, dict):
            raise AnalysisError(
                f"artifact schema incomplete: probes.{probe_id}."
                "auroc_vs_gold_by_tag.tags must be an object"
            )
        if stored_advisory.get("advisory") is not True:
            raise AnalysisError(
                f"numeric disagreement: advisory marker for probe {probe_id}"
            )
        if set(advisory) != set(calculated["advisory"]):
            raise AnalysisError(
                f"numeric disagreement: advisory tags for probe {probe_id}"
            )
        for tag, tag_stored in advisory.items():
            tag_stored = _require_artifact_object(
                tag_stored,
                path=f"probes.{probe_id}.auroc_vs_gold_by_tag.tags.{tag}",
                fields=(
                    "auroc",
                    "ci95_low",
                    "ci95_high",
                    "n_valid_resamples",
                    "n_bootstrap",
                    "seed",
                    "n_rows",
                ),
            )
            tag_independent = calculated["advisory"][tag]
            compare_bootstrap_metadata(
                tag_stored,
                tag_independent,
                path=f"probes.{probe_id}.advisory.{tag}",
            )
            if tag_stored.get("n_rows") != tag_independent["n_rows"]:
                raise AnalysisError(
                    f"numeric disagreement: probes.{probe_id}.advisory.{tag}.n_rows"
                )
            compare(
                "auroc_point",
                tag_stored["auroc"],
                tag_independent["point"],
                path=f"probes.{probe_id}.advisory.{tag}.auroc",
                tolerance=1e-8,
            )
            for stored_key, other_key in (("ci95_low", "low"), ("ci95_high", "high")):
                compare(
                    "auroc_ci",
                    tag_stored[stored_key],
                    tag_independent[other_key],
                    path=f"probes.{probe_id}.advisory.{tag}.{stored_key}",
                    tolerance=5e-3,
                )

    stored_pc = _require_artifact_object(
        artifact["pc1_explained_variance"],
        path="pc1_explained_variance",
        fields=(
            "explained_variance_ratio",
            "ci95_low",
            "ci95_high",
            "n_valid_resamples",
            "n_bootstrap",
            "seed",
            "n_rows_complete",
            "n_rows_dropped",
            "n_components",
        ),
    )
    compare_bootstrap_metadata(
        stored_pc, recomputed["pc"], path="pc1_explained_variance"
    )
    compare(
        "pc1_point",
        stored_pc["explained_variance_ratio"],
        recomputed["pc"]["point"],
        path="pc1_explained_variance.explained_variance_ratio",
        tolerance=1e-8,
    )
    for stored_key, other_key in (("ci95_low", "low"), ("ci95_high", "high")):
        compare(
            "pc1_ci",
            stored_pc[stored_key],
            recomputed["pc"][other_key],
            path=f"pc1_explained_variance.{stored_key}",
            tolerance=5e-3,
        )
    if artifact["n_effective_dimensions"] != recomputed["n_effective"]:
        raise AnalysisError("numeric disagreement: n_effective_dimensions")
    for field in ("n_rows_complete", "n_rows_dropped", "n_components"):
        if stored_pc.get(field) != recomputed["pc"][field]:
            raise AnalysisError(
                f"numeric disagreement: pc1_explained_variance.{field}"
            )

    stored_cost = _require_artifact_object(
        artifact["cost"],
        path="cost",
        fields=("mean_elapsed_s", "median_elapsed_s", "n_timed", "n_missing"),
    )
    for field in ("n_timed", "n_missing"):
        if stored_cost.get(field) != data["cost"][field]:
            raise AnalysisError(f"numeric disagreement: cost.{field}")
    for field in ("mean_elapsed_s", "median_elapsed_s"):
        compare(
            "cost",
            stored_cost.get(field),
            data["cost"][field],
            path=f"cost.{field}",
            tolerance=1e-8,
        )

    stored_k1 = _require_artifact_object(
        artifact["k1"],
        path="k1",
        fields=(
            "realized_min_ci_low_abs",
            "threshold",
            "boolean",
            "per_probe",
        ),
    )
    independent_k1 = recomputed["k1"]
    if set(stored_k1.get("per_probe", {})) != set(
        independent_k1["per_probe"]
    ):
        raise AnalysisError("verdict disagreement: k1 per-probe ids changed")
    compare(
        "gate",
        stored_k1.get("realized_min_ci_low_abs"),
        independent_k1["realized_min_ci_low_abs"],
        path="k1.realized_min_ci_low_abs",
        tolerance=5e-3,
    )
    compare(
        "gate",
        stored_k1.get("threshold"),
        independent_k1["threshold"],
        path="k1.threshold",
        tolerance=1e-8,
    )
    for probe_id, independent_block in independent_k1["per_probe"].items():
        stored_block = _require_artifact_object(
            stored_k1["per_probe"][probe_id],
            path=f"k1.per_probe.{probe_id}",
            fields=(
                "realized_ci_low_abs",
                "threshold",
                "boolean",
                "redundant_reason",
            ),
        )
        compare(
            "gate",
            stored_block.get("realized_ci_low_abs"),
            independent_block["realized_ci_low_abs"],
            path=f"k1.per_probe.{probe_id}.realized_ci_low_abs",
            tolerance=5e-3,
        )
        compare(
            "gate",
            stored_block.get("threshold"),
            independent_block["threshold"],
            path=f"k1.per_probe.{probe_id}.threshold",
            tolerance=1e-8,
        )
        for field in ("boolean", "redundant_reason"):
            if stored_block.get(field) != independent_block[field]:
                raise AnalysisError(
                    f"verdict disagreement: k1.per_probe.{probe_id}.{field}"
                )

    stored_k2 = _require_artifact_object(
        artifact["k2"],
        path="k2",
        fields=(
            "realized_max_ci95_high",
            "threshold",
            "boolean",
            "per_probe_ci95_high",
        ),
    )
    independent_k2 = recomputed["k2"]
    if set(stored_k2.get("per_probe_ci95_high", {})) != set(
        independent_k2["per_probe_ci95_high"]
    ):
        raise AnalysisError("verdict disagreement: k2 per-probe ids changed")
    for field in ("realized_max_ci95_high", "threshold"):
        compare(
            "gate",
            stored_k2.get(field),
            independent_k2[field],
            path=f"k2.{field}",
            tolerance=5e-3 if field.startswith("realized") else 1e-8,
        )
    for probe_id, independent_value in independent_k2[
        "per_probe_ci95_high"
    ].items():
        compare(
            "gate",
            stored_k2["per_probe_ci95_high"].get(probe_id),
            independent_value,
            path=f"k2.per_probe_ci95_high.{probe_id}",
            tolerance=5e-3,
        )

    stored_k3 = _require_artifact_object(
        artifact["k3"],
        path="k3",
        fields=(
            "realized_pc1_ci95_low",
            "pc1_threshold",
            "realized_min_abs_pairwise_spearman",
            "spearman_threshold",
            "boolean",
        ),
    )
    independent_k3 = recomputed["k3"]
    for field, tolerance in (
        ("realized_pc1_ci95_low", 5e-3),
        ("pc1_threshold", 1e-8),
        ("realized_min_abs_pairwise_spearman", 1e-8),
        ("spearman_threshold", 1e-8),
    ):
        compare(
            "gate",
            stored_k3.get(field),
            independent_k3[field],
            path=f"k3.{field}",
            tolerance=tolerance,
        )
    for name, stored_gate, independent_gate in (
        ("k1", stored_k1, independent_k1),
        ("k2", stored_k2, independent_k2),
        ("k3", stored_k3, independent_k3),
    ):
        if stored_gate.get("boolean") != independent_gate["boolean"]:
            raise AnalysisError(f"verdict disagreement: {name} boolean changed")
    if artifact["verdict"] != recomputed["verdict"]:
        raise AnalysisError(
            f"verdict disagreement: artifact={artifact['verdict']}, "
            f"recomputed={recomputed['verdict']}"
        )
    if artifact["verdict_reason"] != recomputed["verdict_reason"]:
        raise AnalysisError("verdict disagreement: verdict_reason changed")

    for family in (
        "power",
        "auroc_point",
        "auroc_ci",
        "rho_point",
        "rho_ci",
        "delta_auroc_point",
        "delta_auroc_ci",
        "pc1_point",
        "pc1_ci",
        "accounting",
        "cost",
        "gate",
    ):
        print(f"assert max_abs_delta {family}={maxima[family]:.12g}")
    nonredundant = sum(
        not block["boolean"] for block in artifact["k1"]["per_probe"].values()
    )
    print(
        f"summary verdict={artifact['verdict']} n_rows={inputs['n_rows']} "
        f"n_clusters={inputs['n_clusters']} "
        "base_auroc_cluster_bootstrap_se="
        f"{artifact['power']['base_auroc_cluster_bootstrap_se']:.12g} "
        f"non_redundant_probes={nonredundant}"
    )
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--selftest", action="store_true", help="run four in-memory tests"
    )
    mode.add_argument(
        "--assert-artifact",
        action="store_true",
        help="independently reproduce DEFAULT_ARTIFACT",
    )
    parser.add_argument("--probes", type=Path, default=DEFAULT_PROBES)
    parser.add_argument("--out", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument(
        "--seed", type=int, default=int(KILL_RULE["bootstrap_seed"])
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.selftest:
            return _run_selftest()
        if args.assert_artifact:
            return _assert_artifact()
        artifact = _analyze_file(
            probes_path=args.probes,
            max_rows=args.max_rows,
            sample_seed=args.seed,
        )
        if (
            not artifact["powered"]
            and args.probes.resolve() != DEFAULT_PROBES.resolve()
            and args.out.resolve() == DEFAULT_ARTIFACT.resolve()
        ):
            raise AnalysisError(
                "an underpowered non-default --probes input requires an explicit "
                "non-default --out; refusing to create the canonical killgate "
                "artifact"
            )
        _write_artifact(args.out, artifact)
        if not artifact["powered"]:
            failures = ", ".join(
                block["threshold_name"] for block in artifact["power_failure"]
            )
            print(
                f"powered=false verdict=null power_failure={failures} "
                f"artifact={args.out}"
            )
            return 1
        print(
            f"verdict={artifact['verdict']} n_rows={artifact['inputs']['n_rows']} "
            f"n_clusters={artifact['inputs']['n_clusters']} artifact={args.out}"
        )
        return 0
    except (
        AnalysisError,
        AssertionError,
        AttributeError,
        IndexError,
        KeyError,
        TypeError,
        ValueError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
