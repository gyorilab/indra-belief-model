"""E1: prediction--rejection curves for the shipped local Gemma belief.

This is a deterministic replay over stored reader verdicts.  It makes no model
calls.  The JSONL ``belief`` field predates the enabled local-Gemma calibration,
so the shipped scalar is recomputed from those verdicts through the enabled
profile, exactly as ``calibration_ship_gate.score_statements`` does.

The Vashurin et al. prediction--rejection curve has rejection fraction on the
x-axis and average quality of the retained predictions on the y-axis.  Here the
quality atom is binary statement correctness.  In addition to that curve and
its PRR-normalized area, this artifact carries the operational view: how many
gold errors a curator is expected to remove at each review budget.

The shipped belief has large ties.  Scores are rounded to 12 decimal places for
tie grouping, which only coalesces machine-roundoff variants in these runs.
Every cut through a tie is evaluated analytically as uniform selection within
the tie.  An exact-float sensitivity is emitted beside the primary result.

Usage:
    PYTHONPATH=src .venv/bin/python \
      scripts/compute_shipped_belief_rejection_curve.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from xml.sax.saxutils import escape as xml_escape


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from calibration_ship_gate import statements_for_run  # noqa: E402
from indra_belief.calibration_constants import (  # noqa: E402
    calibration_for_run,
    reader_configuration_for_run,
)
from indra_belief.noise_model import RECALIBRATED_PRIORS  # noqa: E402
from indra_belief.statement_belief import statement_belief  # noqa: E402


SCORE_DECIMALS = 12
REVIEW_FRACTIONS = (0.05, 0.10, 0.20, 0.30, 0.50)
PAPER_URL = "https://aclanthology.org/2025.tacl-1.11/"
IMPLEMENTATION_URL = (
    "https://lm-polygraph.readthedocs.io/en/latest/_modules/"
    "lm_polygraph/ue_metrics/pred_rej_area.html"
)

SPLITS = (
    {
        "id": "fit_eval_curation_v1",
        "role": "profile fit split (descriptive, not independent)",
        "run": "data/results/eval_curation_v1_local-gemma-4-26b.jsonl",
        "gold": "data/benchmark/eval_curation_v1.jsonl",
    },
    {
        "id": "validation_external_curator_v1",
        "role": "independent validation split",
        "run": "data/results/external_curator_v1_local-gemma-4-26b.jsonl",
        "gold": "data/benchmark/external_curator_gold_v1.jsonl",
    },
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _file_record(relative_path: str) -> dict:
    path = ROOT / relative_path
    return {
        "path": relative_path,
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
        "physical_jsonl_rows": sum(1 for line in path.open() if line.strip()),
    }


def _profile_record(profile: dict, configuration: dict) -> dict:
    return {
        "profile_id": profile.get("profile_id"),
        "reader_model": profile.get("reader_model"),
        "deployment_status": profile.get("deployment_status"),
        "reader_configuration": configuration.get("id"),
        "configuration_status": configuration.get("status"),
        "fit_run": profile.get("fit_run"),
        "fit_gold": profile.get("fit_gold"),
        "prior_logodds": profile.get("prior_logodds"),
        "log_lr_confirm": profile.get("log_lr_confirm"),
        "log_lr_reject": profile.get("log_lr_reject"),
    }


def _score_split(spec: dict) -> tuple[list[dict], dict, dict, dict]:
    """Join, roll up, and deterministically replay the shipped belief."""
    run_path = ROOT / spec["run"]
    statements, join = statements_for_run(spec["run"], spec["gold"])
    configuration = reader_configuration_for_run(run_path)
    profile = calibration_for_run(run_path)
    if configuration.get("status") != "identified":
        raise ValueError(f"unidentified reader configuration for {spec['run']}: {configuration}")
    if profile is None:
        raise ValueError(f"no ship-approved calibration profile for {spec['run']}")
    if profile.get("deployment_status") != "enabled":
        raise ValueError(f"profile is not shipped/enabled for {spec['run']}: {profile}")

    scored: list[dict] = []
    undefined: list[dict] = []
    for stmt in statements:
        # Match calibration_ship_gate.score_statements: an observation enters the
        # belief metric only when both the fallback and calibrated paths define a
        # scalar.  The calibrated result is the currently shipped scalar.
        fallback = statement_belief(stmt["ev"], RECALIBRATED_PRIORS)
        shipped = statement_belief(stmt["ev"], RECALIBRATED_PRIORS, soft=profile)
        if fallback.belief is None or shipped.belief is None:
            undefined.append(
                {
                    "gold_correct": bool(stmt["gold_correct"]),
                    "route": shipped.verdict_statement,
                }
            )
            continue
        belief = float(shipped.belief)
        if not math.isfinite(belief) or not 0.0 <= belief <= 1.0:
            raise ValueError(f"invalid shipped belief {belief!r} for {stmt['statement_key']}")
        scored.append(
            {
                "statement_key": stmt["statement_key"],
                "belief": belief,
                "gold_correct": bool(stmt["gold_correct"]),
            }
        )

    census = {
        "grouped_statements": len(statements),
        "scored_statements": len(scored),
        "correct_statements": sum(row["gold_correct"] for row in scored),
        "incorrect_statements": sum(not row["gold_correct"] for row in scored),
        "undefined_belief_statements": len(undefined),
        "undefined_correct": sum(row["gold_correct"] for row in undefined),
        "undefined_incorrect": sum(not row["gold_correct"] for row in undefined),
        "undefined_routes": dict(sorted(Counter(row["route"] for row in undefined).items())),
    }
    census["grouped_correct_statements"] = (
        census["correct_statements"] + census["undefined_correct"]
    )
    census["grouped_incorrect_statements"] = (
        census["incorrect_statements"] + census["undefined_incorrect"]
    )
    if census["scored_statements"] + census["undefined_belief_statements"] != len(statements):
        raise AssertionError("scored + undefined does not reconcile to grouped statements")
    return scored, census, join, _profile_record(profile, configuration)


def _score_key(score: float, decimals: int | None) -> float:
    return score if decimals is None else round(score, decimals)


def _tie_blocks(rows: list[dict], decimals: int | None) -> list[dict]:
    groups: dict[float, list[dict]] = defaultdict(list)
    for row in rows:
        groups[_score_key(row["belief"], decimals)].append(row)
    blocks = []
    for score in sorted(groups):
        members = groups[score]
        errors = sum(not row["gold_correct"] for row in members)
        raw = [row["belief"] for row in members]
        blocks.append(
            {
                "belief": score,
                "statements": len(members),
                "errors": errors,
                "correct": len(members) - errors,
                "error_rate": errors / len(members),
                "raw_belief_min": min(raw),
                "raw_belief_max": max(raw),
            }
        )
    return blocks


def _allocation(blocks: list[dict], review_count: int) -> dict:
    """Expected, minimum, and maximum errors caught at an exact budget.

    Whole lower-belief blocks are deterministic.  If the budget cuts through a
    tie, the expected count is hypergeometric expectation; the bounds are the
    attainable extremes without pretending belief supplies an order in the tie.
    """
    total = sum(block["statements"] for block in blocks)
    if not 0 <= review_count <= total:
        raise ValueError(f"review_count {review_count} outside [0, {total}]")

    remaining = review_count
    expected = minimum = maximum = 0.0
    reviewed_before = errors_before = 0
    boundary = None
    for block in blocks:
        n = block["statements"]
        e = block["errors"]
        c = block["correct"]
        if remaining >= n:
            expected += e
            minimum += e
            maximum += e
            remaining -= n
            reviewed_before += n
            errors_before += e
            continue
        if remaining > 0:
            take = remaining
            expected += take * e / n
            minimum += max(0, take - c)
            maximum += min(take, e)
            boundary = {
                "belief": block["belief"],
                "block_statements": n,
                "block_errors": e,
                "reviewed_from_block": take,
                "statements_strictly_below": reviewed_before,
                "errors_strictly_below": errors_before,
            }
            remaining = 0
        break
    if remaining:
        raise AssertionError("tie blocks did not cover the requested review budget")
    return {
        "errors_removed_expected": float(expected),
        "errors_removed_min": int(minimum),
        "errors_removed_max": int(maximum),
        "boundary_tie": boundary,
    }


def _budget_count(n: int, fraction: float) -> int:
    # floor keeps the queue at or below the stated budget.  The epsilon prevents
    # a decimal such as 0.30 that is mathematically integral from losing one row.
    return int(math.floor(n * fraction + 1e-12))


def _analyze_order(rows: list[dict], decimals: int | None, *, full_curve: bool) -> dict:
    n = len(rows)
    errors = sum(not row["gold_correct"] for row in rows)
    correct = n - errors
    if n == 0 or errors == 0 or correct == 0:
        raise ValueError("prediction--rejection analysis requires both gold classes")

    blocks = _tie_blocks(rows, decimals)
    points = []
    model_quality = []
    random_quality = correct / n
    oracle_quality = []
    capture_y = []
    for k in range(n + 1):
        allocation = _allocation(blocks, k)
        caught = allocation["errors_removed_expected"]
        remaining_count = n - k
        remaining_errors = errors - caught
        rejected_fraction = k / n
        error_capture = caught / errors
        capture_y.append(error_capture)
        point = {
            "reviewed_statements": k,
            "rejection_fraction": rejected_fraction,
            "retained_statements": remaining_count,
            "errors_removed_expected": caught,
            "errors_removed_min": allocation["errors_removed_min"],
            "errors_removed_max": allocation["errors_removed_max"],
            "fraction_of_defined_belief_errors_removed": error_capture,
            "remaining_errors_expected": remaining_errors,
            "retained_error_rate_expected": (
                remaining_errors / remaining_count if remaining_count else None
            ),
            "retained_quality_expected": (
                1.0 - remaining_errors / remaining_count if remaining_count else None
            ),
            "random": {
                "errors_removed_expected": errors * rejected_fraction,
                "fraction_of_defined_belief_errors_removed": rejected_fraction,
                "retained_quality_expected": random_quality if remaining_count else None,
            },
            "oracle": {
                "errors_removed": min(k, errors),
                "fraction_of_defined_belief_errors_removed": min(k, errors) / errors,
                "retained_quality": (
                    1.0 - (errors - min(k, errors)) / remaining_count
                    if remaining_count
                    else None
                ),
            },
            "boundary_tie": allocation["boundary_tie"],
        }
        if remaining_count:
            model_quality.append(point["retained_quality_expected"])
            oracle_quality.append(point["oracle"]["retained_quality"])
        if full_curve:
            points.append(point)

    # LM-Polygraph's full-rejection convention is a discrete mean over the N
    # non-empty retained sets (k=0,...,N-1), rather than inventing quality at k=N.
    auc_model = sum(model_quality) / n
    auc_random = random_quality
    auc_oracle = sum(oracle_quality) / n
    prr_denominator = auc_oracle - auc_random
    if prr_denominator <= 0:
        raise AssertionError("oracle AUC must exceed random AUC")

    # Trapezoidal area of the operational cumulative-error-capture curve.  It is
    # useful to operators, but is deliberately not labeled prediction-rejection
    # AUC; random rejection is the diagonal and therefore has area exactly 0.5.
    capture_auc = sum(
        (capture_y[i] + capture_y[i + 1]) / (2 * n) for i in range(n)
    )
    oracle_capture_y = [min(k, errors) / errors for k in range(n + 1)]
    oracle_capture_auc = sum(
        (oracle_capture_y[i] + oracle_capture_y[i + 1]) / (2 * n)
        for i in range(n)
    )

    budget_points = []
    for requested in REVIEW_FRACTIONS:
        k = _budget_count(n, requested)
        allocation = _allocation(blocks, k)
        caught = allocation["errors_removed_expected"]
        remaining_count = n - k
        budget_points.append(
            {
                "requested_review_fraction": requested,
                "reviewed_statements": k,
                "actual_review_fraction": k / n,
                "retained_statements": remaining_count,
                "errors_removed_expected": caught,
                "errors_removed_min": allocation["errors_removed_min"],
                "errors_removed_max": allocation["errors_removed_max"],
                "fraction_of_defined_belief_errors_removed": caught / errors,
                "random_errors_removed_expected": errors * k / n,
                "random_fraction_of_defined_belief_errors_removed": k / n,
                "error_removal_lift_over_random": (
                    (caught / errors) / (k / n) if k else None
                ),
                "remaining_errors_expected": errors - caught,
                "retained_error_rate_expected": (errors - caught) / remaining_count,
                "retained_quality_expected": 1.0 - (errors - caught) / remaining_count,
                "boundary_tie": allocation["boundary_tie"],
            }
        )

    max_span = max(block["raw_belief_max"] - block["raw_belief_min"] for block in blocks)
    result = {
        "tie_policy": {
            "score_decimals": decimals,
            "description": (
                "exact IEEE-754 equality"
                if decimals is None
                else (
                    f"round belief to {decimals} decimals for tie grouping; "
                    "uniform expected selection within a boundary tie"
                )
            ),
        },
        "tie_diagnostics": {
            "distinct_beliefs": len(blocks),
            "statements_in_non_singleton_ties": sum(
                block["statements"] for block in blocks if block["statements"] > 1
            ),
            "largest_tie": max(block["statements"] for block in blocks),
            "max_raw_belief_span_within_group": max_span,
        },
        "areas": {
            "prediction_rejection_auc": auc_model,
            "random_prediction_rejection_auc": auc_random,
            "oracle_prediction_rejection_auc": auc_oracle,
            "prediction_rejection_ratio_prr": (
                (auc_model - auc_random) / prr_denominator
            ),
            "auc_gain_over_random": auc_model - auc_random,
            "beats_random": bool(auc_model > auc_random),
            "convention": (
                "mean retained quality over k=0..N-1 reviewed; binary quality="
                "is_gold_correct; higher is better"
            ),
            "error_removal_curve_auc": capture_auc,
            "random_error_removal_curve_auc": 0.5,
            "oracle_error_removal_curve_auc": oracle_capture_auc,
            "error_removal_auc_gain_over_random": capture_auc - 0.5,
        },
        "review_budget_points": budget_points,
        "belief_blocks": blocks,
    }
    if full_curve:
        result["prediction_rejection_curve"] = points

    if abs(auc_random - correct / n) > 1e-15:
        raise AssertionError("random AUC does not equal base accuracy")
    if abs(capture_y[-1] - 1.0) > 1e-15:
        raise AssertionError("full review did not remove every error")
    if any(a > b + 1e-15 for a, b in zip(capture_y, capture_y[1:])):
        raise AssertionError("error-removal curve is not monotone")
    return result


def _plain_10_percent(split: dict) -> dict:
    point = next(
        p for p in split["review_budget_points"]
        if p["requested_review_fraction"] == 0.10
    )
    errors = split["census"]["incorrect_statements"]
    return {
        "split": split["id"],
        "reviewed_statements": point["reviewed_statements"],
        "scored_statements": split["census"]["scored_statements"],
        "actual_review_fraction": point["actual_review_fraction"],
        "errors_removed_expected": point["errors_removed_expected"],
        "errors_removed_attainable_range": [
            point["errors_removed_min"],
            point["errors_removed_max"],
        ],
        "errors_before_review": errors,
        "errors_in_full_grouped_panel": split["census"][
            "grouped_incorrect_statements"
        ],
        "fraction_of_defined_belief_errors_removed": point[
            "fraction_of_defined_belief_errors_removed"
        ],
        "random_errors_removed_expected": point["random_errors_removed_expected"],
        "random_fraction_of_defined_belief_errors_removed": point[
            "random_fraction_of_defined_belief_errors_removed"
        ],
        "remaining_errors_expected": point["remaining_errors_expected"],
        "retained_statements": point["retained_statements"],
        "undefined_belief_statements": split["census"][
            "undefined_belief_statements"
        ],
        "undefined_incorrect": split["census"]["undefined_incorrect"],
        "undefined_routes": split["census"]["undefined_routes"],
        "boundary_tie": point["boundary_tie"],
    }


def build_artifact() -> dict:
    split_results = []
    profile_ids = set()
    for spec in SPLITS:
        rows, census, join, profile = _score_split(spec)
        primary = _analyze_order(rows, SCORE_DECIMALS, full_curve=True)
        exact = _analyze_order(rows, None, full_curve=False)
        profile_ids.add(profile["profile_id"])
        split_results.append(
            {
                "id": spec["id"],
                "role": spec["role"],
                "inputs": {
                    "run": _file_record(spec["run"]),
                    "gold": _file_record(spec["gold"]),
                },
                "reader_profile": profile,
                "join": join,
                "census": census,
                "tie_policy": primary["tie_policy"],
                "tie_diagnostics": {
                    **primary["tie_diagnostics"],
                    "exact_float_distinct_beliefs": exact["tie_diagnostics"][
                        "distinct_beliefs"
                    ],
                },
                "areas": primary["areas"],
                "review_budget_points": primary["review_budget_points"],
                "belief_blocks": primary["belief_blocks"],
                "prediction_rejection_curve": primary["prediction_rejection_curve"],
                "exact_float_sensitivity": {
                    "purpose": (
                        "shows the effect of preserving sub-1e-12 machine-roundoff "
                        "score distinctions; no input or statement-key tie order is used"
                    ),
                    "tie_diagnostics": exact["tie_diagnostics"],
                    "areas": exact["areas"],
                    "review_budget_points": exact["review_budget_points"],
                },
            }
        )
    if len(profile_ids) != 1:
        raise AssertionError(f"splits resolved different shipped profiles: {profile_ids}")

    artifact = {
        "artifact": "shipped_belief_prediction_rejection_curve",
        "schema_version": 1,
        "generated_by": "scripts/compute_shipped_belief_rejection_curve.py",
        "no_new_inference": True,
        "question": (
            "If a curator reviews the least-confident X% of statements, what "
            "fraction of errors in the defined-belief curve population is removed, "
            "and how many statements is that?"
        ),
        "method": {
            "score": (
                "current shipped hybrid calibrated belief, deterministically replayed "
                "from stored evidence verdicts via the enabled configuration profile"
            ),
            "stored_run_belief_used": False,
            "quality": "1 iff indra_belief.curation.is_gold_correct(statement_gold)",
            "statement_gold": "group by run stmt_hash; any-incorrect-wins",
            "join_and_exclusion_protocol": (
                "calibration_ship_gate.statements_for_run and score_statements semantics"
            ),
            "ordering": "lowest belief reviewed first",
            "budget_rounding": "floor(N * requested_fraction), so budget is not exceeded",
            "ties": (
                f"belief rounded to {SCORE_DECIMALS} decimals only for tie grouping; "
                "expected uniform selection within a cut tie, with attainable bounds"
            ),
            "undefined_beliefs": (
                "excluded from the scalar curve and disclosed separately, matching the "
                "ship gate; their shipped route is reported in each split census"
            ),
            "random_baseline": (
                "analytic uniform random rejection: retained quality stays at base "
                "accuracy and expected fraction of errors removed equals rejection fraction"
            ),
            "oracle_baseline": "reject every gold-incorrect statement before any correct one",
            "prediction_rejection_auc": (
                "discrete mean of retained average quality for k=0..N-1, matching "
                "LM-Polygraph's full-rejection convention"
            ),
            "prr": "(AUC_shipped - AUC_random) / (AUC_oracle - AUC_random)",
            "reference": PAPER_URL,
            "reference_implementation": IMPLEMENTATION_URL,
        },
        "curator_hours": {
            "estimated": False,
            "reason": (
                "No measured human per-statement review time is available in the "
                "repository; model inference latency is not curator labor. Counts are reported."
            ),
        },
        "splits": split_results,
    }
    artifact["plain_language_10_percent"] = [
        _plain_10_percent(split) for split in split_results
    ]
    return artifact


def _pct(value: float, digits: int = 1) -> str:
    return f"{100 * value:.{digits}f}%"


def _num(value: float, digits: int = 1) -> str:
    return f"{value:.{digits}f}"


def render_markdown(artifact: dict) -> str:
    lines: list[str] = []
    add = lines.append
    add("# E1 — What does the shipped-belief review queue buy?")
    add("")
    add(
        "This is a deterministic replay of the **shipped hybrid calibrated belief** "
        "from stored reader verdicts; it makes no new model calls. The stored JSONL "
        "`belief` field is not used because it predates the enabled calibration."
    )
    add("")
    add("## Plain answer at a 10% review budget")
    add("")
    for answer in artifact["plain_language_10_percent"]:
        label = "Fit" if answer["split"].startswith("fit_") else "Independent validation"
        add(
            f"- **{label}:** review {answer['reviewed_statements']} of "
            f"{answer['scored_statements']} statements "
            f"({_pct(answer['actual_review_fraction'])}). That removes an expected "
            f"{_num(answer['errors_removed_expected'])} of "
            f"{answer['errors_before_review']} defined-belief errors "
            f"({_pct(answer['fraction_of_defined_belief_errors_removed'])}), versus "
            f"{_num(answer['random_errors_removed_expected'])} "
            f"({_pct(answer['random_fraction_of_defined_belief_errors_removed'])}) "
            f"under random "
            f"review. It leaves an expected {_num(answer['remaining_errors_expected'])} "
            f"errors among {answer['retained_statements']} unreviewed statements."
        )
    add("")
    add(
        "Those are **tie-neutral expectations**, not guaranteed integer catches. Both "
        "10% cuts land inside a large equal-belief cell, so belief alone cannot choose "
        "which tied statements come first; the attainable error counts are shown below."
    )
    add("")
    add(
        "The percentages are conditional on statements with a defined belief, matching "
        "the ship-gate curve population. Outside that ranking, the product already routes "
        + " and ".join(
            f"{answer['undefined_belief_statements']} {('fit' if answer['split'].startswith('fit_') else 'validation')} "
            f"undefined-belief statements to review ({answer['undefined_incorrect']} gold "
            f"{'error' if answer['undefined_incorrect'] == 1 else 'errors'})"
            for answer in artifact["plain_language_10_percent"]
        )
        + "; no artificial scalar rank is assigned to them."
    )
    add("")
    add("## Prediction–rejection area and random baseline")
    add("")
    add(
        "The cited prediction–rejection curve plots correctness among the retained "
        "statements as low-belief statements are progressively reviewed. PRR normalizes "
        "the shipped curve between random and oracle rejection."
    )
    add("")
    add("| split | scored | errors | shipped AUC | random AUC | oracle AUC | PRR | beats random? |")
    add("|---|---:|---:|---:|---:|---:|---:|:---:|")
    for split in artifact["splits"]:
        census, areas = split["census"], split["areas"]
        add(
            f"| {split['id']} | {census['scored_statements']} | "
            f"{census['incorrect_statements']} | "
            f"{areas['prediction_rejection_auc']:.6f} | "
            f"{areas['random_prediction_rejection_auc']:.6f} | "
            f"{areas['oracle_prediction_rejection_auc']:.6f} | "
            f"{areas['prediction_rejection_ratio_prr']:.6f} | "
            f"{'yes' if areas['beats_random'] else 'no'} |"
        )
    add("")
    add(
        "The separate cumulative error-removal AUCs are "
        + "; ".join(
            f"{split['id']} {split['areas']['error_removal_curve_auc']:.6f}"
            for split in artifact["splits"]
        )
        + ", against random = 0.500000. This operational curve is not mislabeled as "
        "the Vashurin prediction–rejection AUC."
    )
    add("")
    add("## Review-budget operating points")
    add("")
    for split in artifact["splits"]:
        census = split["census"]
        add(f"### {split['id']}")
        add("")
        add(
            "| target | reviewed | retained | expected errors removed (attainable range) | "
            "% of defined-belief errors removed | random | errors left |"
        )
        add("|---:|---:|---:|---:|---:|---:|---:|")
        for point in split["review_budget_points"]:
            add(
                f"| {_pct(point['requested_review_fraction'], 0)} | "
                f"{point['reviewed_statements']} ({_pct(point['actual_review_fraction'])}) | "
                f"{point['retained_statements']} | {_num(point['errors_removed_expected'])} "
                f"({point['errors_removed_min']}–{point['errors_removed_max']}) | "
                f"{_pct(point['fraction_of_defined_belief_errors_removed'])} | "
                f"{_pct(point['random_fraction_of_defined_belief_errors_removed'])} | "
                f"{_num(point['remaining_errors_expected'])} |"
            )
        add("")
        add(
            f"Census: {census['grouped_statements']} gold-grouped statements; "
            f"{census['scored_statements']} defined beliefs; "
            f"{census['undefined_belief_statements']} undefined and excluded exactly as "
            f"the ship gate does (routes: {census['undefined_routes']})."
        )
        add("")
        td = split["tie_diagnostics"]
        add(
            f"Tie census: {td['distinct_beliefs']} numerical-tie-normalized belief cells "
            f"({td['exact_float_distinct_beliefs']} exact float values); "
            f"largest cell {td['largest_tie']} statements; "
            f"{td['statements_in_non_singleton_ties']} statements are in non-singleton cells."
        )
        add("")
    add("## Interpretation limits")
    add("")
    add(
        "- The fit split is descriptive because it supplied the shipped reader profile; "
        "the external-curator split is the independent check."
    )
    add(
        "- Scores are rounded to 12 decimals only to collapse machine-roundoff variants. "
        "The JSON includes exact-float sensitivity results. No file-order or statement-key "
        "secondary ranking is credited to belief."
    )
    add(
        "- No measured human review time is available, so converting statement counts to "
        "curator-hours would be invented. Counts are reported instead."
    )
    add("")
    add(
        f"Method reference: [Vashurin et al. (2025)]({PAPER_URL}); "
        f"[LM-Polygraph area implementation]({IMPLEMENTATION_URL})."
    )
    add("")
    add("Generated by `scripts/compute_shipped_belief_rejection_curve.py`.")
    return "\n".join(lines) + "\n"


def _polyline(points: list[dict], key_path: tuple[str, ...], x0: float, y0: float,
              width: float, height: float) -> str:
    coords = []
    for point in points:
        value = point
        for key in key_path:
            value = value[key]
        if value is None:
            continue
        x = x0 + width * point["rejection_fraction"]
        y = y0 + height * (1.0 - float(value))
        coords.append(f"{x:.2f},{y:.2f}")
    return " ".join(coords)


def render_svg(artifact: dict) -> str:
    width, height = 1120, 520
    plot_w, plot_h = 460, 340
    top = 92
    panel_x = (78, 630)
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfaf7"/>',
        '<style>text{font-family:Inter,Arial,sans-serif;fill:#20252b}'
        '.axis{stroke:#68717b;stroke-width:1}.grid{stroke:#d9dde1;stroke-width:1}'
        '.model{fill:none;stroke:#2457a6;stroke-width:3}'
        '.random{fill:none;stroke:#7a7f87;stroke-width:2;stroke-dasharray:7 6}'
        '.oracle{fill:none;stroke:#27815c;stroke-width:2;stroke-dasharray:2 5}</style>',
        '<text x="50" y="38" font-size="24" font-weight="700">Shipped-belief prediction–rejection curve</text>',
        '<text x="50" y="64" font-size="13">Retained statement correctness as the lowest-belief statements go to review · tie-neutral expectation</text>',
    ]
    for idx, split in enumerate(artifact["splits"]):
        x0 = panel_x[idx]
        points = split["prediction_rejection_curve"]
        for tick in (0.0, 0.25, 0.5, 0.75, 1.0):
            x = x0 + plot_w * tick
            y = top + plot_h * (1 - tick)
            out.append(f'<line class="grid" x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_h}"/>')
            out.append(f'<line class="grid" x1="{x0}" y1="{y:.2f}" x2="{x0 + plot_w}" y2="{y:.2f}"/>')
            out.append(f'<text x="{x:.2f}" y="{top + plot_h + 21}" text-anchor="middle" font-size="11">{int(tick * 100)}%</text>')
            out.append(f'<text x="{x0 - 12}" y="{y + 4:.2f}" text-anchor="end" font-size="11">{tick:.2f}</text>')
        out.append(f'<line class="axis" x1="{x0}" y1="{top + plot_h}" x2="{x0 + plot_w}" y2="{top + plot_h}"/>')
        out.append(f'<line class="axis" x1="{x0}" y1="{top}" x2="{x0}" y2="{top + plot_h}"/>')
        out.append(
            f'<polyline class="oracle" points="{_polyline(points, ("oracle", "retained_quality"), x0, top, plot_w, plot_h)}"/>'
        )
        out.append(
            f'<polyline class="random" points="{_polyline(points, ("random", "retained_quality_expected"), x0, top, plot_w, plot_h)}"/>'
        )
        out.append(
            f'<polyline class="model" points="{_polyline(points, ("retained_quality_expected",), x0, top, plot_w, plot_h)}"/>'
        )
        title = xml_escape(split["id"])
        areas = split["areas"]
        out.append(f'<text x="{x0}" y="{top - 28}" font-size="15" font-weight="700">{title}</text>')
        out.append(
            f'<text x="{x0}" y="{top - 9}" font-size="12">AUC {areas["prediction_rejection_auc"]:.3f} · '
            f'random {areas["random_prediction_rejection_auc"]:.3f} · PRR {areas["prediction_rejection_ratio_prr"]:.3f}</text>'
        )
        out.append(f'<text x="{x0 + plot_w / 2}" y="{top + plot_h + 46}" text-anchor="middle" font-size="12">Statements sent to review</text>')
    out.extend(
        [
            '<text x="18" y="275" transform="rotate(-90 18 275)" text-anchor="middle" font-size="12">Correctness among retained statements</text>',
            '<line class="model" x1="385" y1="489" x2="421" y2="489"/><text x="429" y="493" font-size="12">shipped belief</text>',
            '<line class="random" x1="535" y1="489" x2="571" y2="489"/><text x="579" y="493" font-size="12">random</text>',
            '<line class="oracle" x1="655" y1="489" x2="691" y2="489"/><text x="699" y="493" font-size="12">oracle</text>',
            '</svg>',
        ]
    )
    return "\n".join(out) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-json", default="data/results/shipped_belief_rejection_curve.json"
    )
    parser.add_argument(
        "--out-md", default="data/results/shipped_belief_rejection_curve.md"
    )
    parser.add_argument(
        "--out-svg", default="data/results/shipped_belief_rejection_curve.svg"
    )
    args = parser.parse_args()

    artifact = build_artifact()
    outputs = {
        ROOT / args.out_json: json.dumps(artifact, indent=2) + "\n",
        ROOT / args.out_md: render_markdown(artifact),
        ROOT / args.out_svg: render_svg(artifact),
    }
    for path, content in outputs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)

    for answer in artifact["plain_language_10_percent"]:
        print(
            f"{answer['split']}: review {answer['reviewed_statements']}/"
            f"{answer['scored_statements']}; expected errors removed "
            f"{answer['errors_removed_expected']:.2f}/"
            f"{answer['errors_before_review']} "
            f"({100 * answer['fraction_of_defined_belief_errors_removed']:.2f}%), random "
            f"{100 * answer['random_fraction_of_defined_belief_errors_removed']:.2f}%"
        )
    for split in artifact["splits"]:
        areas = split["areas"]
        print(
            f"{split['id']}: AUC={areas['prediction_rejection_auc']:.6f}, "
            f"random={areas['random_prediction_rejection_auc']:.6f}, "
            f"oracle={areas['oracle_prediction_rejection_auc']:.6f}, "
            f"PRR={areas['prediction_rejection_ratio_prr']:.6f}"
        )
    print("Wrote " + ", ".join(str(path.relative_to(ROOT)) for path in outputs))


if __name__ == "__main__":
    main()
