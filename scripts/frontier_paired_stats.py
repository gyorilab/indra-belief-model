#!/usr/bin/env python3
"""Paired uncertainty for adjacent points on a frozen cost/F1 frontier.

This script consumes the audited JSON emitted by ``frontier_table.py``.  It
does not recompute the frontier or mutate scoring runs.  Instead, it verifies
the gold/run SHA-256 digests recorded in that JSON, reconstructs the latest
logical attempt for every benchmark pair, and compares each adjacent formal
Pareto point on the same rows.

The performance contrast is error-detection F1(B) - error-detection F1(A),
where A is the cheaper point and B is the next more expensive point.  Sampling
uncertainty is a paired percentile bootstrap over exact statement/evidence
pairs.  The null test independently swaps A/B predictions within each pair;
its Monte Carlo p-value uses the standard plus-one correction. Holm-adjusted
p-values annotate multiplicity across the selected adjacent comparisons; because
the same outcomes selected the frontier, they are exploratory rather than valid
post-selection family-wise error guarantees.

Example (after the strict frontier JSON has been frozen)::

    PYTHONPATH=src .venv/bin/python scripts/frontier_paired_stats.py \
      data/results/representative_indra_expanded_403_20260717_frontier.json \
      --bootstrap-samples 20000 --permutations 20000 --seed 20260717 \
      --output-json \
        data/results/representative_indra_expanded_403_20260717_paired.json \
      --output-md \
        data/results/representative_indra_expanded_403_20260717_paired.md

These are conditional, post-selection comparisons.  They quantify benchmark-
row sampling uncertainty for fixed run outputs; they do not include run-to-run
model variation, uncertainty in benchmark provenance, price uncertainty, or
the fact that the frontier itself was selected on the same rows.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from eval_curation_compare import MASK, bootstrap_errf1  # noqa: E402
from frontier_table import VALID_VERDICTS, canonicalize_attempts  # noqa: E402
from indra_belief.curation import is_gold_correct  # noqa: E402

DEFAULT_BOOTSTRAP_SAMPLES = 20_000
DEFAULT_PERMUTATIONS = 20_000
DEFAULT_SEED = 20_260_717
DEFAULT_ALPHA = 0.05


def _read_snapshot(path: Path) -> tuple[bytes, str]:
    """Read one immutable byte snapshot and return it with its digest."""
    payload = path.read_bytes()
    return payload, hashlib.sha256(payload).hexdigest()


def _decode_json(payload: bytes, path: Path) -> dict[str, Any]:
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _decode_jsonl(payload: bytes, path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"invalid UTF-8 in {path}: {exc}") from exc
    for line_no, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON in {path}:{line_no}: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"expected a JSON object in {path}:{line_no}")
        rows.append(value)
    return rows


def _resolve_report_path(value: Any, *, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"frontier JSON is missing {field}")
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path)


def _exact_pair(row: dict[str, Any]) -> tuple[int, int]:
    try:
        return int(row["matches_hash"]) & MASK, int(row["source_hash"]) & MASK
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError("gold row lacks a valid matches_hash/source_hash pair") from exc


def _scored_exact_pair(row: dict[str, Any]) -> tuple[int, int]:
    """Return the literal pair carried by a scored row, with no source fallback."""
    try:
        stmt_hash = row.get("stmt_hash")
        if stmt_hash is not None:
            if isinstance(stmt_hash, bool):
                raise ValueError
            matches_hash = (
                int(stmt_hash.strip(), 16)
                if isinstance(stmt_hash, str)
                else int(stmt_hash)
            )
        else:
            matches_hash_raw = row["matches_hash"]
            if isinstance(matches_hash_raw, bool):
                raise ValueError
            matches_hash = int(matches_hash_raw)
        source_hash = row["source_hash"]
        if isinstance(source_hash, bool):
            raise ValueError
        return matches_hash & MASK, int(source_hash) & MASK
    except (KeyError, TypeError, ValueError, OverflowError, AttributeError) as exc:
        raise ValueError(
            "scored row lacks a valid literal stmt_hash/matches_hash and source_hash pair"
        ) from exc


def _checked_snapshot(
    path: Path, expected: Any, *, field: str
) -> tuple[bytes, str]:
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError(f"frontier JSON is missing a valid {field}")
    payload, actual = _read_snapshot(path)
    if actual != expected:
        raise ValueError(
            f"SHA-256 mismatch for {path}: frontier {field}={expected}, current={actual}"
        )
    return payload, actual


def _stable_seed(base_seed: int, cheaper: str, expensive: str, purpose: str) -> int:
    payload = f"{base_seed}\0{cheaper}\0{expensive}\0{purpose}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _f1_many(gold: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Vectorized error-F1 for a matrix whose final axis is benchmark rows."""
    tp = np.count_nonzero(pred & gold, axis=-1)
    fp = np.count_nonzero(pred & ~gold, axis=-1)
    fn = np.count_nonzero(~pred & gold, axis=-1)
    denominator = 2 * tp + fp + fn
    return np.divide(
        2 * tp,
        denominator,
        out=np.zeros_like(denominator, dtype=float),
        where=denominator != 0,
    )


def paired_permutation_errf1(
    gold_err: Iterable[bool],
    pred_err_a: Iterable[bool],
    pred_err_b: Iterable[bool],
    *,
    n_perm: int = DEFAULT_PERMUTATIONS,
    seed: int = DEFAULT_SEED,
    batch_size: int = 2_048,
) -> dict[str, Any]:
    """Two-sided paired randomization test for ``|F1(B) - F1(A)|``.

    The prediction labels are independently swapped within each benchmark row
    under the exchangeability null.  ``(hits + 1) / (n_perm + 1)`` prevents a
    finite Monte Carlo run from reporting an impossible p-value of zero.
    """
    if n_perm <= 0:
        raise ValueError("n_perm must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    ge = np.asarray(list(gold_err), dtype=bool)
    pa = np.asarray(list(pred_err_a), dtype=bool)
    pb = np.asarray(list(pred_err_b), dtype=bool)
    if not len(ge) or ge.shape != pa.shape or ge.shape != pb.shape:
        raise ValueError("paired arrays must be non-empty and have identical shapes")

    observed = float(abs(_f1_many(ge, pb) - _f1_many(ge, pa)))
    rng = np.random.default_rng(seed)
    hits = 0
    completed = 0
    while completed < n_perm:
        batch = min(batch_size, n_perm - completed)
        swap = rng.integers(0, 2, size=(batch, len(ge)), dtype=np.int8).astype(bool)
        pa_perm = np.where(swap, pb, pa)
        pb_perm = np.where(swap, pa, pb)
        delta = np.abs(_f1_many(ge, pb_perm) - _f1_many(ge, pa_perm))
        hits += int(np.count_nonzero(delta >= observed))
        completed += batch

    return {
        "observed_absolute_delta": observed,
        "hits": hits,
        "permutations": n_perm,
        "p_value": (hits + 1) / (n_perm + 1),
        "minimum_attainable_p": 1 / (n_perm + 1),
    }


def _holm_adjust(p_values: list[float]) -> list[float]:
    """Holm step-down adjustment for a fixed family of p-values."""
    m = len(p_values)
    if not m:
        return []
    order = sorted(range(m), key=lambda index: p_values[index])
    adjusted = [0.0] * m
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (m - rank) * p_values[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def _prediction_map(
    run_path: Path,
    *,
    attempts: list[dict[str, Any]],
    gold_pairs: set[tuple[int, int]],
) -> tuple[dict[tuple[int, int], bool], int, int]:
    canonical, retry_rows = canonicalize_attempts(attempts)
    predictions: dict[tuple[int, int], bool] = {}
    for row in canonical:
        key = _scored_exact_pair(row)
        if key not in gold_pairs:
            raise ValueError(f"unmatched literal statement/source pair {key} in {run_path}")
        verdict = row.get("verdict")
        if verdict not in VALID_VERDICTS:
            raise ValueError(f"invalid/null verdict in canonical rows of {run_path}")
        if key in predictions:
            raise ValueError(f"duplicate canonical gold pair {key} in {run_path}")
        predictions[key] = verdict == "incorrect"
    missing = gold_pairs - predictions.keys()
    extra = predictions.keys() - gold_pairs
    if missing or extra:
        raise ValueError(
            f"{run_path} does not have exact complete gold coverage "
            f"(missing={len(missing)}, extra={len(extra)})"
        )
    return predictions, len(attempts), retry_rows


def build_paired_report(
    frontier_path: Path,
    *,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    permutations: int = DEFAULT_PERMUTATIONS,
    seed: int = DEFAULT_SEED,
    alpha: float = DEFAULT_ALPHA,
) -> dict[str, Any]:
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    if permutations <= 0:
        raise ValueError("permutations must be positive")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between zero and one")

    frontier_path = frontier_path.resolve()
    frontier_bytes, frontier_sha = _read_snapshot(frontier_path)
    frontier = _decode_json(frontier_bytes, frontier_path)
    if frontier.get("require_valid_coverage") is not True:
        raise ValueError(
            "paired frontier inference requires a report built with "
            "--require-valid-coverage"
        )
    denominator = frontier.get("denominator")
    if not isinstance(denominator, int) or denominator <= 0:
        raise ValueError("frontier denominator must be a positive integer")
    metric = frontier.get("metric") or {}
    if (
        metric.get("name") != "error_detection_f1"
        or metric.get("positive_class") != "curator label is incorrect"
        or metric.get("positive_prediction") != "model verdict is incorrect"
    ):
        raise ValueError("frontier JSON does not declare the expected error-F1 semantics")

    gold_info = frontier.get("gold") or {}
    gold_path = _resolve_report_path(gold_info.get("path"), field="gold.path")
    gold_bytes, gold_sha = _checked_snapshot(
        gold_path, gold_info.get("sha256"), field="gold.sha256"
    )
    gold_rows = _decode_jsonl(gold_bytes, gold_path)
    gold_order = [_exact_pair(row) for row in gold_rows]
    if len(gold_rows) != denominator or len(set(gold_order)) != denominator:
        raise ValueError(
            "gold must contain denominator unique exact (matches_hash, source_hash) pairs"
        )
    gold_pairs = set(gold_order)
    unique_statements = len({pair[0] for pair in gold_order})

    model_rows = frontier.get("models")
    if not isinstance(model_rows, list):
        raise ValueError("frontier JSON is missing its models list")
    rows_by_name: dict[str, dict[str, Any]] = {}
    for row in model_rows:
        if not isinstance(row, dict) or not isinstance(row.get("name"), str):
            raise ValueError("invalid model row in frontier JSON")
        if row["name"] in rows_by_name:
            raise ValueError(f"duplicate model name in frontier JSON: {row['name']}")
        rows_by_name[row["name"]] = row

    names = (frontier.get("pareto") or {}).get("formal_frontier")
    if not isinstance(names, list) or len(names) < 2 or not all(
        isinstance(name, str) for name in names
    ):
        raise ValueError("formal frontier must contain at least two named models")
    try:
        formal_rows = [rows_by_name[name] for name in names]
    except KeyError as exc:
        raise ValueError(f"formal frontier names an absent model: {exc.args[0]}") from exc
    formal_rows.sort(key=lambda row: (float(row["usd_1k"]), row["name"]))

    predictions: dict[str, dict[tuple[int, int], bool]] = {}
    run_audit: dict[str, dict[str, Any]] = {}
    for row in formal_rows:
        name = row["name"]
        if not row.get("frontier_eligible"):
            raise ValueError(f"formal frontier model is not marked eligible: {name}")
        if row.get("run_status") != "completed":
            raise ValueError(f"formal frontier model run is not completed: {name}")
        if (
            not row.get("run_coverage_complete")
            or not row.get("coverage_complete")
            or row.get("n") != denominator
        ):
            raise ValueError(f"formal frontier model lacks valid full coverage: {name}")
        if (
            row.get("parse_nulls")
            or row.get("row_errors")
            or row.get("unmatched_rows")
            or row.get("invalid_json_lines")
        ):
            raise ValueError(f"formal frontier model has invalid rows: {name}")
        run_path = _resolve_report_path(row.get("run_path"), field=f"{name}.run_path")
        run_bytes, run_sha = _checked_snapshot(
            run_path, row.get("run_sha256"), field=f"{name}.run_sha256"
        )
        pred, attempt_rows, retry_rows = _prediction_map(
            run_path,
            attempts=_decode_jsonl(run_bytes, run_path),
            gold_pairs=gold_pairs,
        )
        observed_f1 = float(
            _f1_many(
                np.asarray(
                    [not is_gold_correct(row_["tag"]) for row_ in gold_rows],
                    dtype=bool,
                ),
                np.asarray([pred[key] for key in gold_order], dtype=bool),
            )
        )
        if not math.isclose(observed_f1, float(row["f1"]), abs_tol=1e-12):
            raise ValueError(
                f"recomputed F1 for {name} ({observed_f1}) differs from frontier "
                f"point ({row['f1']})"
            )
        predictions[name] = pred
        run_audit[name] = {
            "path": row["run_path"],
            "sha256": run_sha,
            "attempt_rows": attempt_rows,
            "retry_rows": retry_rows,
            "canonical_valid_rows": denominator,
        }

    gold_err = [not is_gold_correct(row["tag"]) for row in gold_rows]
    comparisons: list[dict[str, Any]] = []
    for cheaper, expensive in zip(formal_rows, formal_rows[1:]):
        a_name = cheaper["name"]
        b_name = expensive["name"]
        pred_a = [predictions[a_name][key] for key in gold_order]
        pred_b = [predictions[b_name][key] for key in gold_order]
        boot_seed = _stable_seed(seed, a_name, b_name, "bootstrap")
        perm_seed = _stable_seed(seed, a_name, b_name, "permutation")
        boot = bootstrap_errf1(
            gold_err,
            pred_a,
            pred_b,
            n_boot=bootstrap_samples,
            seed=boot_seed,
        )
        permutation = paired_permutation_errf1(
            gold_err,
            pred_a,
            pred_b,
            n_perm=permutations,
            seed=perm_seed,
        )
        delta_cost = float(expensive["usd_1k"]) - float(cheaper["usd_1k"])
        delta_f1 = float(boot["delta"])
        comparisons.append(
            {
                "cheaper": a_name,
                "more_expensive": b_name,
                "n": denominator,
                "f1_cheaper": float(boot["f1_a"]),
                "f1_more_expensive": float(boot["f1_b"]),
                "delta_f1_more_expensive_minus_cheaper": delta_f1,
                "delta_f1_ci_95": [float(x) for x in boot["ci_delta"]],
                "f1_cheaper_ci_95_on_paired_resamples": [
                    float(x) for x in boot["ci_a"]
                ],
                "f1_more_expensive_ci_95_on_paired_resamples": [
                    float(x) for x in boot["ci_b"]
                ],
                "verdict_disagreements": sum(a != b for a, b in zip(pred_a, pred_b)),
                "permutation_p_raw": float(permutation["p_value"]),
                "permutation_hits": int(permutation["hits"]),
                "bootstrap_seed": boot_seed,
                "permutation_seed": perm_seed,
                "usd_per_1k_cheaper": float(cheaper["usd_1k"]),
                "usd_per_1k_more_expensive": float(expensive["usd_1k"]),
                "delta_usd_per_1k": delta_cost,
                "incremental_usd_per_1k_per_0_01_f1": (
                    delta_cost * 0.01 / delta_f1 if delta_f1 > 0 else None
                ),
            }
        )

    adjusted = _holm_adjust([row["permutation_p_raw"] for row in comparisons])
    for row, p_holm in zip(comparisons, adjusted):
        row["permutation_p_holm"] = p_holm
        row["raw_p_at_or_below_alpha_exploratory"] = (
            row["permutation_p_raw"] <= alpha
        )
        row["holm_p_at_or_below_alpha_exploratory"] = p_holm <= alpha
        lo, hi = row["delta_f1_ci_95"]
        row["delta_ci_excludes_zero"] = lo > 0 or hi < 0

    return {
        "schema_version": 1,
        "label": frontier.get("label"),
        "source_frontier": {
            "path": _display_path(frontier_path),
            "sha256": frontier_sha,
        },
        "gold": {
            "path": gold_info.get("path"),
            "sha256": gold_sha,
            "rows": len(gold_rows),
            "unique_exact_pairs": len(gold_pairs),
            "unique_statement_hashes": unique_statements,
        },
        "denominator": denominator,
        "formal_frontier_cost_order": [row["name"] for row in formal_rows],
        "method": {
            "metric": "error-detection F1",
            "positive_class": "curator label is incorrect",
            "positive_prediction": "model verdict is incorrect",
            "contrast": "next-more-expensive minus cheaper adjacent Pareto point",
            "resampling_unit": "exact (matches_hash, source_hash) pair",
            "resampling_unit_equals_statement": unique_statements == denominator,
            "bootstrap": "paired percentile 95% CI",
            "bootstrap_samples": bootstrap_samples,
            "permutation": (
                "two-sided paired within-row prediction swap; Monte Carlo plus-one correction"
            ),
            "permutations": permutations,
            "multiplicity": (
                "Holm adjustment across selected adjacent comparisons; descriptive only, "
                "not post-selection family-wise error control"
            ),
            "alpha": alpha,
            "base_seed": seed,
            "inference_scope": (
                "conditional on the frozen benchmark rows and fixed run outputs; post-selection "
                "because the same rows define the point frontier"
            ),
            "not_included": [
                "model run-to-run variation",
                "benchmark selection/provenance uncertainty",
                "price uncertainty",
                "independent validation after frontier selection",
            ],
        },
        "run_audit": run_audit,
        "comparisons": comparisons,
    }


def render_markdown(report: dict[str, Any]) -> str:
    alpha = report["method"]["alpha"]
    lines = [
        f"# Paired uncertainty on the formal frontier — {report.get('label') or 'unnamed'}",
        "",
        (
            f"The formal point frontier, cheapest to most expensive, is "
            f"**{' → '.join(report['formal_frontier_cost_order'])}**. Each contrast below is "
            "the next-more-expensive model minus its cheaper neighbor on the same frozen rows."
        ),
        "",
        "| adjacent step | n | error-F1 (cheap → costly) | ΔF1 [paired 95% CI] | Δ$/1k | paired permutation p | Holm p |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["comparisons"]:
        lo, hi = row["delta_f1_ci_95"]
        lines.append(
            f"| {row['cheaper']} → {row['more_expensive']} | {row['n']} | "
            f"{row['f1_cheaper']:.3f} → {row['f1_more_expensive']:.3f} | "
            f"{row['delta_f1_more_expensive_minus_cheaper']:+.3f} "
            f"[{lo:+.3f}, {hi:+.3f}] | {row['delta_usd_per_1k']:+.2f} | "
            f"{row['permutation_p_raw']:.4f} | {row['permutation_p_holm']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
        ]
    )
    for row in report["comparisons"]:
        contrast = f"{row['more_expensive']} over {row['cheaper']}"
        delta = row["delta_f1_more_expensive_minus_cheaper"]
        if row["holm_p_at_or_below_alpha_exploratory"]:
            statement = (
                f"the exploratory Holm-adjusted p is at or below α for the observed "
                f"{'gain' if delta > 0 else 'loss'} on these rows; selection on the same "
                "outcomes prevents a confirmatory claim"
            )
        else:
            statement = (
                "the observed difference is unresolved after Holm correction; this is not "
                "evidence that the models are equivalent"
            )
        lines.append(f"- **{contrast}:** {statement} (reference α={alpha:.2f}).")

    scope = report["method"]["inference_scope"]
    excluded = ", ".join(report["method"]["not_included"])
    unit = report["method"]["resampling_unit"]
    if report["method"]["resampling_unit_equals_statement"]:
        unit += " (one unique statement hash per pair in this benchmark)"
    lines.extend(
        [
            "",
            "## Method and scope",
            "",
            f"The resampling unit is the {unit}. The paired bootstrap uses "
            f"{report['method']['bootstrap_samples']:,} resamples; the two-sided paired "
            f"randomization test uses {report['method']['permutations']:,} swaps with a "
            "plus-one Monte Carlo correction. Holm adjustment covers only the adjacent "
            "comparisons in this table; it is a descriptive multiplicity annotation and "
            "does not restore family-wise error control after outcome-based frontier selection.",
            "",
            f"Inference is {scope}. It does not include {excluded}. Consequently, these tests "
            "are an uncertainty annotation on the point-estimate frontier, not a replacement "
            "for an independently validated model-selection claim.",
            "",
        ]
    )
    return "\n".join(lines)


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("frontier_json", type=Path)
    parser.add_argument(
        "--bootstrap-samples", type=int, default=DEFAULT_BOOTSTRAP_SAMPLES
    )
    parser.add_argument("--permutations", type=int, default=DEFAULT_PERMUTATIONS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-md", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_paired_report(
        args.frontier_json,
        bootstrap_samples=args.bootstrap_samples,
        permutations=args.permutations,
        seed=args.seed,
        alpha=args.alpha,
    )
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    markdown = render_markdown(report)
    if args.output_json:
        _write(args.output_json, payload)
        print(f"wrote {args.output_json}")
    if args.output_md:
        _write(args.output_md, markdown)
        print(f"wrote {args.output_md}")
    if not args.output_json and not args.output_md:
        print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
