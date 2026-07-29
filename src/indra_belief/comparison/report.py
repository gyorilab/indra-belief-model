"""Validate the comparison metrics artifact and render Markdown and HTML."""

from __future__ import annotations

import hashlib
import html
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from .error_review import CANONICAL_PROTOCOL_SHA256, HUMAN_ATTESTATION, REPORT_KIND


METRICS_KIND = "indra_statement_belief_comparison"
LITERATURE_KIND = "indra_assembly_paper_published_method_metrics"
ERROR_REVIEW_FIELDS = {
    "artifact_kind",
    "status",
    "panel_id",
    "arm_id",
    "model_id",
    "packet_id",
    "evaluated_statements",
    "threshold_errors",
    "error_types",
    "human_classifications",
    "review",
    "defensibility",
    "dimensions",
    "taxonomy_refinements",
    "adjudications",
    "provenance",
}
SUMMARY_FIELDS = {"count", "denominator", "proportion"}
ERROR_TYPES = ("false_positive", "false_negative")
HUMAN_CLASSIFICATIONS = ("supports_claim", "rejects_claim", "indeterminate")
ERROR_REVIEW_BINDING_FILES = (
    "spec",
    "bundle_manifest",
    "gold",
    "predictions",
    "execution_ledger",
)


class ReportError(ValueError):
    """The supplied artifacts cannot support a comparison report."""


def _load_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReportError(f"cannot read {label} at {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ReportError(f"{label} must be a JSON object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, staged_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(staged_name, path)
    except Exception:
        try:
            os.unlink(staged_name)
        except FileNotFoundError:
            pass
        raise


def _number(value: Any, *, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReportError(f"{context} must be numeric")
    result = float(value)
    if not (-float("inf") < result < float("inf")):
        raise ReportError(f"{context} must be finite")
    return result


def _mapping(value: Any, *, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ReportError(f"{context} must be an object")
    return value


def _exact_fields(value: Mapping[str, Any], expected: set[str], *, context: str) -> None:
    if set(value) != expected:
        missing = sorted(expected - set(value))
        extra = sorted(set(value) - expected)
        raise ReportError(f"{context} fields differ: missing={missing}; unexpected={extra}")


def _count(value: Any, *, context: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ReportError(f"{context} must be an integer >= {minimum}")
    return value


def _sha256_text(value: Any, *, context: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ReportError(f"{context} must be a lowercase SHA-256")
    return value


def _commitment_sha256(value: Any, *, context: str) -> str:
    descriptor = _mapping(value, context=context)
    if not {"sha256", "bytes"} <= set(descriptor) or set(descriptor) - {
        "sha256",
        "bytes",
        "rows",
    }:
        raise ReportError(f"{context} is not a path-free file commitment")
    _count(descriptor["bytes"], context=f"{context}.bytes")
    if "rows" in descriptor:
        _count(descriptor["rows"], context=f"{context}.rows")
    return _sha256_text(descriptor["sha256"], context=f"{context}.sha256")


def _review_count_summary(
    value: Any, *, denominator: int, context: str
) -> dict[str, int | float | None]:
    row = _mapping(value, context=context)
    _exact_fields(row, SUMMARY_FIELDS, context=context)
    count = _count(row["count"], context=f"{context}.count")
    declared = _count(row["denominator"], context=f"{context}.denominator")
    if declared != denominator or count > denominator:
        raise ReportError(f"{context} does not use the required denominator")
    expected = None if denominator == 0 else count / denominator
    proportion = row["proportion"]
    if expected is None:
        if proportion is not None:
            raise ReportError(f"{context}.proportion must be null for a zero denominator")
    elif (
        isinstance(proportion, bool)
        or not isinstance(proportion, (int, float))
        or abs(float(proportion) - expected) > 1e-12
    ):
        raise ReportError(f"{context}.proportion is inconsistent")
    return {"count": count, "denominator": denominator, "proportion": expected}


def _canonical_error_review(value: Mapping[str, Any], *, index: int) -> dict[str, Any]:
    context = f"error_reviews[{index}]"
    review = _mapping(value, context=context)
    _exact_fields(review, ERROR_REVIEW_FIELDS, context=context)
    if review["artifact_kind"] != REPORT_KIND or review["status"] != "complete":
        raise ReportError(f"{context} is not a completed canonical error review")
    panel_id = review["panel_id"]
    if panel_id not in {"paper_all_source", "paper_readers"}:
        raise ReportError(f"{context}.panel_id is not a canonical paper panel")
    evaluated = _count(
        review["evaluated_statements"],
        context=f"{context}.evaluated_statements",
        minimum=1,
    )
    threshold = _review_count_summary(
        review["threshold_errors"],
        denominator=evaluated,
        context=f"{context}.threshold_errors",
    )
    total = int(threshold["count"])

    error_types = _mapping(review["error_types"], context=f"{context}.error_types")
    _exact_fields(error_types, set(ERROR_TYPES), context=f"{context}.error_types")
    parsed_error_types: dict[str, Any] = {}
    error_total = 0
    for error_type in ERROR_TYPES:
        raw = _mapping(error_types[error_type], context=f"{context}.error_types.{error_type}")
        _exact_fields(
            raw,
            SUMMARY_FIELDS | {"defensible", "non_defensible"},
            context=f"{context}.error_types.{error_type}",
        )
        summary = _review_count_summary(
            {key: raw[key] for key in SUMMARY_FIELDS},
            denominator=total,
            context=f"{context}.error_types.{error_type}",
        )
        stratum_total = int(summary["count"])
        defensible = _review_count_summary(
            raw["defensible"],
            denominator=stratum_total,
            context=f"{context}.error_types.{error_type}.defensible",
        )
        non_defensible = _review_count_summary(
            raw["non_defensible"],
            denominator=stratum_total,
            context=f"{context}.error_types.{error_type}.non_defensible",
        )
        if int(defensible["count"]) + int(non_defensible["count"]) != stratum_total:
            raise ReportError(f"{context}.error_types.{error_type} does not partition its errors")
        parsed_error_types[error_type] = {
            **summary,
            "defensible": defensible,
            "non_defensible": non_defensible,
        }
        error_total += stratum_total
    if error_total != total:
        raise ReportError(f"{context}.error_types do not cover every threshold error")

    raw_classes = _mapping(
        review["human_classifications"], context=f"{context}.human_classifications"
    )
    _exact_fields(
        raw_classes,
        set(HUMAN_CLASSIFICATIONS),
        context=f"{context}.human_classifications",
    )
    classifications = {
        name: _review_count_summary(
            raw_classes[name],
            denominator=total,
            context=f"{context}.human_classifications.{name}",
        )
        for name in HUMAN_CLASSIFICATIONS
    }
    if sum(int(row["count"]) for row in classifications.values()) != total:
        raise ReportError(f"{context}.human_classifications do not cover every error")

    raw_defensibility = _mapping(
        review["defensibility"], context=f"{context}.defensibility"
    )
    expected_defensibility = {
        "denominator",
        "defensible",
        "non_defensible",
        "system_supported_defensible",
        "indeterminate_ambiguity_defensible",
        "unresolved",
    }
    _exact_fields(
        raw_defensibility,
        expected_defensibility,
        context=f"{context}.defensibility",
    )
    if raw_defensibility["denominator"] != "all_threshold_errors":
        raise ReportError(f"{context}.defensibility denominator differs")
    defensibility = {
        key: _review_count_summary(
            raw_defensibility[key],
            denominator=total,
            context=f"{context}.defensibility.{key}",
        )
        for key in expected_defensibility - {"denominator"}
    }
    defensible_count = int(defensibility["defensible"]["count"])
    if (
        defensible_count + int(defensibility["non_defensible"]["count"]) != total
        or int(defensibility["system_supported_defensible"]["count"])
        + int(defensibility["indeterminate_ambiguity_defensible"]["count"])
        != defensible_count
        or int(defensibility["indeterminate_ambiguity_defensible"]["count"])
        != int(classifications["indeterminate"]["count"])
        or int(defensibility["unresolved"]["count"]) != 0
    ):
        raise ReportError(f"{context}.defensibility split does not reconcile")

    raw_dimensions = _mapping(review["dimensions"], context=f"{context}.dimensions")
    _exact_fields(
        raw_dimensions,
        {"multiple_dimensions_per_case", "denominator", "rows"},
        context=f"{context}.dimensions",
    )
    if (
        raw_dimensions["multiple_dimensions_per_case"] is not True
        or raw_dimensions["denominator"] != "all_threshold_errors"
        or not isinstance(raw_dimensions["rows"], list)
    ):
        raise ReportError(f"{context}.dimensions contract differs")
    dimensions: list[dict[str, Any]] = []
    seen_dimensions: set[str] = set()
    for row_index, raw_value in enumerate(raw_dimensions["rows"]):
        row_context = f"{context}.dimensions.rows[{row_index}]"
        raw = _mapping(raw_value, context=row_context)
        _exact_fields(
            raw,
            SUMMARY_FIELDS | {"dimension", "by_judgment", "by_error_type"},
            context=row_context,
        )
        name = raw["dimension"]
        if not isinstance(name, str) or not name or name in seen_dimensions:
            raise ReportError(f"{row_context}.dimension is invalid or duplicated")
        seen_dimensions.add(name)
        summary = _review_count_summary(
            {key: raw[key] for key in SUMMARY_FIELDS},
            denominator=total,
            context=row_context,
        )
        by_judgment = _mapping(raw["by_judgment"], context=f"{row_context}.by_judgment")
        by_error_type = _mapping(raw["by_error_type"], context=f"{row_context}.by_error_type")
        _exact_fields(by_judgment, {"defensible", "non_defensible"}, context=f"{row_context}.by_judgment")
        _exact_fields(by_error_type, set(ERROR_TYPES), context=f"{row_context}.by_error_type")
        for key, raw_count in (*by_judgment.items(), *by_error_type.items()):
            _count(raw_count, context=f"{row_context}.{key}")
        if sum(int(value) for value in by_judgment.values()) != int(summary["count"]):
            raise ReportError(f"{row_context}.by_judgment does not reconcile")
        if sum(int(value) for value in by_error_type.values()) != int(summary["count"]):
            raise ReportError(f"{row_context}.by_error_type does not reconcile")
        dimensions.append(
            {
                "dimension": name,
                **summary,
                "by_judgment": dict(by_judgment),
                "by_error_type": dict(by_error_type),
            }
        )

    raw_review = _mapping(review["review"], context=f"{context}.review")
    _exact_fields(
        raw_review,
        {
            "reviewer_pseudonyms",
            "resolver_pseudonym",
            "exact_agreement",
            "disagreement_count",
            "resolved_by_resolver_count",
            "classification_reliability",
            "human_attestation",
        },
        context=f"{context}.review",
    )
    reviewers = raw_review["reviewer_pseudonyms"]
    if (
        not isinstance(reviewers, list)
        or len(reviewers) != 2
        or len(set(reviewers)) != 2
        or any(not isinstance(name, str) or not name for name in reviewers)
        or raw_review["human_attestation"] != HUMAN_ATTESTATION
    ):
        raise ReportError(f"{context}.review does not identify two attested human reviewers")
    disagreement_count = _count(
        raw_review["disagreement_count"], context=f"{context}.review.disagreement_count"
    )
    resolved_count = _count(
        raw_review["resolved_by_resolver_count"],
        context=f"{context}.review.resolved_by_resolver_count",
    )
    agreement = _review_count_summary(
        raw_review["exact_agreement"],
        denominator=total,
        context=f"{context}.review.exact_agreement",
    )
    if (
        disagreement_count != resolved_count
        or disagreement_count + int(agreement["count"]) != total
    ):
        raise ReportError(f"{context}.review disagreement resolution does not reconcile")
    resolver = raw_review["resolver_pseudonym"]
    if disagreement_count:
        if (
            not isinstance(resolver, str)
            or not resolver
            or resolver.casefold() in {str(name).casefold() for name in reviewers}
        ):
            raise ReportError(f"{context}.review resolver must be a distinct human")
    elif resolver is not None:
        raise ReportError(f"{context}.review resolver must be null without disagreements")

    adjudications = review["adjudications"]
    if not isinstance(review["taxonomy_refinements"], list) or not isinstance(
        adjudications, list
    ) or len(adjudications) != total:
        raise ReportError(f"{context} review detail arrays are malformed or incomplete")
    observed_classes = {name: 0 for name in HUMAN_CLASSIFICATIONS}
    observed_errors = {
        name: {"count": 0, "defensible": 0, "non_defensible": 0}
        for name in ERROR_TYPES
    }
    observed_judgments = {"defensible": 0, "non_defensible": 0}
    observed_system_supported = 0
    observed_ambiguity = 0
    observed_resolver = 0
    seen_cases: set[str] = set()
    adjudication_fields = {
        "case_id",
        "error_type",
        "human_classification",
        "judgment",
        "defensibility_basis",
        "dimensions",
        "comment",
        "decision_source",
    }
    for row_index, raw_value in enumerate(adjudications):
        row_context = f"{context}.adjudications[{row_index}]"
        row = _mapping(raw_value, context=row_context)
        _exact_fields(row, adjudication_fields, context=row_context)
        case_id = row["case_id"]
        error_type = row["error_type"]
        classification = row["human_classification"]
        if not isinstance(case_id, str) or not case_id or case_id in seen_cases:
            raise ReportError(f"{row_context}.case_id is invalid or duplicated")
        if error_type not in ERROR_TYPES or classification not in HUMAN_CLASSIFICATIONS:
            raise ReportError(f"{row_context} error type or classification is invalid")
        seen_cases.add(case_id)
        expected_judgment = (
            "defensible"
            if classification == "indeterminate"
            or (error_type == "false_positive" and classification == "supports_claim")
            or (error_type == "false_negative" and classification == "rejects_claim")
            else "non_defensible"
        )
        expected_basis = (
            "indeterminate_ambiguity"
            if classification == "indeterminate"
            else (
                "human_matches_system"
                if expected_judgment == "defensible"
                else "human_matches_reference"
            )
        )
        if row["judgment"] != expected_judgment or row["defensibility_basis"] != expected_basis:
            raise ReportError(f"{row_context} judgment or defensibility basis is inconsistent")
        if (
            not isinstance(row["dimensions"], list)
            or not row["dimensions"]
            or len(set(row["dimensions"])) != len(row["dimensions"])
            or any(not isinstance(item, str) or not item for item in row["dimensions"])
        ):
            raise ReportError(f"{row_context}.dimensions are invalid")
        if row["decision_source"] not in {"reviewer_agreement", "resolver"}:
            raise ReportError(f"{row_context}.decision_source is invalid")
        if row["decision_source"] == "resolver":
            observed_resolver += 1
        observed_classes[classification] += 1
        observed_errors[error_type]["count"] += 1
        observed_errors[error_type][expected_judgment] += 1
        observed_judgments[expected_judgment] += 1
        observed_system_supported += int(expected_basis == "human_matches_system")
        observed_ambiguity += int(expected_basis == "indeterminate_ambiguity")
    if observed_resolver != disagreement_count:
        raise ReportError(f"{context}.review resolver decisions do not reconcile")
    for name in HUMAN_CLASSIFICATIONS:
        if observed_classes[name] != int(classifications[name]["count"]):
            raise ReportError(f"{context}.human_classifications differ from adjudications")
    for error_type in ERROR_TYPES:
        expected = parsed_error_types[error_type]
        if any(
            observed_errors[error_type][key]
            != int(expected[key]["count"] if key != "count" else expected["count"])
            for key in ("count", "defensible", "non_defensible")
        ):
            raise ReportError(f"{context}.error_types differ from adjudications")
    if (
        observed_judgments["defensible"] != int(defensibility["defensible"]["count"])
        or observed_judgments["non_defensible"]
        != int(defensibility["non_defensible"]["count"])
        or observed_system_supported
        != int(defensibility["system_supported_defensible"]["count"])
        or observed_ambiguity
        != int(defensibility["indeterminate_ambiguity_defensible"]["count"])
    ):
        raise ReportError(f"{context}.defensibility differs from adjudications")

    provenance = _mapping(review["provenance"], context=f"{context}.provenance")
    protocol = _mapping(provenance.get("protocol"), context=f"{context}.provenance.protocol")
    if protocol.get("sha256") != CANONICAL_PROTOCOL_SHA256:
        raise ReportError(f"{context} does not bind the canonical error-review protocol")
    arm_id = review["arm_id"]
    model_id = review["model_id"]
    if not isinstance(arm_id, str) or not arm_id or not isinstance(model_id, str) or not model_id:
        raise ReportError(f"{context} arm_id and model_id must be non-empty text")
    comparison_inputs = _mapping(
        provenance.get("comparison_inputs"),
        context=f"{context}.provenance.comparison_inputs",
    )
    for key, expected in (
        ("panel_id", panel_id),
        ("arm_id", arm_id),
        ("model_id", model_id),
    ):
        if comparison_inputs.get(key) != expected:
            raise ReportError(
                f"{context}.provenance.comparison_inputs.{key} differs from the review"
            )
    comparison_files = _mapping(
        comparison_inputs.get("files"),
        context=f"{context}.provenance.comparison_inputs.files",
    )
    comparison_sha256s = {
        name: _commitment_sha256(
            comparison_files.get(name),
            context=f"{context}.provenance.comparison_inputs.files.{name}",
        )
        for name in ERROR_REVIEW_BINDING_FILES
    }
    return {
        "panel_id": panel_id,
        "arm_id": arm_id,
        "model_id": model_id,
        "comparison_sha256s": comparison_sha256s,
        "evaluated_statements": evaluated,
        "threshold_errors": threshold,
        "error_types": parsed_error_types,
        "human_classifications": classifications,
        "defensibility": defensibility,
        "dimensions": dimensions,
    }


def _estimate(value: Any, *, context: str) -> tuple[float, tuple[float, float] | None]:
    if not isinstance(value, Mapping):
        raise ReportError(f"{context} must be an estimate object")
    point = _number(value.get("estimate"), context=f"{context}.estimate")
    interval = value.get("ci95")
    if interval is None:
        return point, None
    if not isinstance(interval, list) or len(interval) != 2:
        raise ReportError(f"{context}.ci95 must contain two values")
    lower = _number(interval[0], context=f"{context}.ci95[0]")
    upper = _number(interval[1], context=f"{context}.ci95[1]")
    if lower > upper:
        raise ReportError(f"{context}.ci95 is reversed")
    return point, (lower, upper)


def _metric(arm: Mapping[str, Any], path: Sequence[str]) -> tuple[float, tuple[float, float] | None]:
    value: Any = arm.get("metrics")
    for key in path:
        if not isinstance(value, Mapping) or key not in value:
            raise ReportError(f"arm {arm.get('arm_id')!r} is missing metric {'.'.join(path)}")
        value = value[key]
    return _estimate(value, context=f"{arm.get('arm_id')}.metrics.{'.'.join(path)}")


def _fmt_estimate(value: tuple[float, tuple[float, float] | None], digits: int = 3) -> str:
    point, interval = value
    if interval is None:
        return f"{point:.{digits}f}"
    return f"{point:.{digits}f} [{interval[0]:.{digits}f}, {interval[1]:.{digits}f}]"


def _fmt_cost(arm: Mapping[str, Any]) -> str:
    cost = arm.get("cost")
    if not isinstance(cost, Mapping) or cost.get("status") != "available":
        return "—"
    value = cost.get("usd_per_1k_statements_upper", cost.get("usd_per_1k_statements"))
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "—"
    return f"${float(value):.3f}"


def _fmt_total_cost(arm: Mapping[str, Any]) -> str:
    cost = arm.get("cost")
    if not isinstance(cost, Mapping) or cost.get("status") != "available":
        return "—"
    value = cost.get("inference_usd_upper", cost.get("inference_usd_total"))
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "—"
    return f"${float(value):.4f}"


def _pareto_flag(arm: Mapping[str, Any], *, costed_arm_count: int) -> str:
    pareto = arm.get("pareto")
    if not isinstance(pareto, Mapping) or pareto.get("status") != "available":
        return "—"
    if costed_arm_count == 1:
        return "only costed arm"
    point = "point" if pareto.get("point_pareto") is True else ""
    robust = "robust" if pareto.get("uncertainty_pareto") is True else ""
    return ", ".join(item for item in (point, robust) if item) or "dominated"


def _validate_metrics(value: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    if value.get("artifact_kind") != METRICS_KIND:
        raise ReportError(f"metrics artifact_kind must be {METRICS_KIND!r}")
    substrates = value.get("substrates")
    if not isinstance(substrates, list) or not substrates:
        raise ReportError("metrics must contain at least one substrate")
    seen: set[str] = set()
    for panel_index, panel in enumerate(substrates):
        if not isinstance(panel, Mapping):
            raise ReportError(f"substrates[{panel_index}] must be an object")
        panel_id = panel.get("substrate_id")
        if not isinstance(panel_id, str) or not panel_id or panel_id in seen:
            raise ReportError(f"substrates[{panel_index}] has an invalid or duplicate id")
        seen.add(panel_id)
        arms = panel.get("arms")
        if not isinstance(arms, list) or not arms:
            raise ReportError(f"substrate {panel_id!r} has no arms")
        arm_ids: set[str] = set()
        for arm in arms:
            if not isinstance(arm, Mapping):
                raise ReportError(f"substrate {panel_id!r} contains a non-object arm")
            arm_id = arm.get("arm_id")
            if not isinstance(arm_id, str) or not arm_id or arm_id in arm_ids:
                raise ReportError(f"substrate {panel_id!r} has an invalid or duplicate arm id")
            arm_ids.add(arm_id)
            _metric(arm, ("fold_mean_trapezoidal_pr_auc",))
            _metric(arm, ("pooled_average_precision",))
            _metric(arm, ("auroc",))
            _metric(arm, ("brier",))
            _metric(arm, ("log_loss",))
            _metric(arm, ("calibration", "ece"))
            _metric(arm, ("calibration", "intercept"))
            _metric(arm, ("calibration", "slope"))
        audit = panel.get("released_label_audit")
        if not isinstance(audit, Mapping):
            raise ReportError(f"substrate {panel_id!r} is missing its released-label audit")
        sensitivity = panel.get("strict_e0_resolved_sensitivity")
        if not isinstance(sensitivity, Mapping):
            raise ReportError(f"substrate {panel_id!r} is missing strict-E0 sensitivity")
        sensitivity_arms = sensitivity.get("arms")
        if not isinstance(sensitivity_arms, list) or not sensitivity_arms:
            raise ReportError(f"substrate {panel_id!r} strict-E0 sensitivity has no arms")
        if any(
            not isinstance(arm, Mapping) or "cost" in arm or "pareto" in arm
            for arm in sensitivity_arms
        ):
            raise ReportError(
                f"substrate {panel_id!r} strict-E0 sensitivity duplicates cost or Pareto"
            )
    return substrates


def _md_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    def clean(value: str) -> str:
        return value.replace("|", "\\|").replace("\n", " ")

    rendered = ["| " + " | ".join(clean(item) for item in headers) + " |"]
    rendered.append("| " + " | ".join("---" for _ in headers) + " |")
    rendered.extend("| " + " | ".join(clean(item) for item in row) + " |" for row in rows)
    return "\n".join(rendered)


def _html_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    head = "".join(f"<th>{html.escape(item)}</th>" for item in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(item)}</td>" for item in row) + "</tr>"
        for row in rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _arm_rows(panel: Mapping[str, Any]) -> list[list[str]]:
    rows: list[list[str]] = []
    costed_arm_count = sum(
        1
        for arm in panel["arms"]
        if isinstance(arm.get("cost"), Mapping) and arm["cost"].get("status") == "available"
    )
    for arm in panel["arms"]:
        threshold = arm.get("metrics", {}).get("threshold", {})
        threshold_metrics = threshold.get("metrics", {}) if isinstance(threshold, Mapping) else {}
        f1 = threshold_metrics.get("f1") if isinstance(threshold_metrics, Mapping) else None
        f1_text = _fmt_estimate(_estimate(f1, context=f"{arm['arm_id']}.threshold.f1")) if f1 else "—"
        rows.append(
            [
                str(arm.get("label", arm["arm_id"])),
                str(arm.get("family", "")),
                _fmt_estimate(_metric(arm, ("fold_mean_trapezoidal_pr_auc",))),
                _fmt_estimate(_metric(arm, ("pooled_average_precision",))),
                _fmt_estimate(_metric(arm, ("auroc",))),
                _fmt_estimate(_metric(arm, ("brier",))),
                _fmt_estimate(_metric(arm, ("log_loss",))),
                _fmt_estimate(_metric(arm, ("calibration", "ece"))),
                _fmt_estimate(_metric(arm, ("calibration", "intercept"))),
                _fmt_estimate(_metric(arm, ("calibration", "slope"))),
                f1_text,
                _fmt_total_cost(arm),
                _fmt_cost(arm),
                _pareto_flag(arm, costed_arm_count=costed_arm_count),
            ]
        )
    return rows


ARM_HEADERS = (
    "Arm",
    "Family",
    "Paper PR-AUC",
    "Average precision",
    "AUROC",
    "Brier",
    "Log loss",
    "ECE",
    "Cal. intercept",
    "Cal. slope",
    "F1",
    "Total USD",
    "USD / 1k",
    "Pareto",
)


def _audit_text(panel: Mapping[str, Any]) -> str:
    audit = panel["released_label_audit"]
    released = audit.get("released")
    strict = audit.get("strict_e0")
    assumption = audit.get("released_negative_assumption")
    if not all(isinstance(value, Mapping) for value in (released, strict, assumption)):
        raise ReportError(f"{panel.get('substrate_id')} has a malformed released-label audit")
    share = _number(
        assumption.get("share_of_released_negatives"),
        context="released-negative assumption share",
    )
    return (
        f"Released target rule: {audit.get('released_label_rule')}. Strict E0 rule: "
        f"{audit.get('strict_e0_rule')}. The primary panel retains all "
        f"{released.get('statements')} released binary targets "
        f"({released.get('positive')} positive, {released.get('negative')} negative). "
        f"Strict E0 resolves {strict.get('resolved')} "
        f"({strict.get('positive')} positive, {strict.get('negative')} negative) and leaves "
        f"{strict.get('unresolved')} unresolved. Those unresolved rows are "
        f"{share:.1%} of released negatives; they are not silently presented as complete "
        "evidence-level truth."
    )


def _paired_rows(value: Mapping[str, Any]) -> list[list[str]]:
    comparisons = value.get("comparisons")
    if not isinstance(comparisons, list):
        raise ReportError("paired comparisons must be an array")
    rows: list[list[str]] = []
    for row in comparisons:
        if not isinstance(row, Mapping) or row.get("metric") != "fold_mean_trapezoidal_pr_auc":
            continue
        estimate = _estimate(row.get("delta"), context="paired PR-AUC delta")
        rows.append(
            [
                str(row.get("a_arm_id")),
                str(row.get("b_arm_id")),
                _fmt_estimate(estimate),
                str(row.get("better_when")),
            ]
        )
    return rows


def _structured_cost_rows(panel: Mapping[str, Any]) -> list[list[str]]:
    rows: list[list[str]] = []
    for arm in panel["arms"]:
        cost = arm.get("cost")
        if not isinstance(cost, Mapping) or cost.get("status") != "available":
            continue

        def numeric(name: str) -> str:
            value = cost.get(name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                return "—"
            return f"{float(value):.6f}"

        token_text = (
            f"{cost.get('input_tokens')} / {cost.get('output_tokens')}"
            if cost.get("token_accounting_complete") is True
            else "incomplete"
        )
        scope = cost.get("scope")
        excluded = scope.get("excluded_cost_categories") if isinstance(scope, Mapping) else None
        rules = (
            f"retries={cost.get('includes_retries')}; relation subcalls="
            f"{cost.get('includes_relation_subcalls')}; excluded="
            f"{', '.join(str(value) for value in excluded) if isinstance(excluded, list) else '—'}"
        )
        pricing = cost.get("pricing")
        tariff = pricing.get("tariff") if isinstance(pricing, Mapping) else None
        price_contract = (
            f"{pricing.get('provider', '—')} / {pricing.get('provider_model_id', '—')} · "
            f"${tariff.get('input_usd_per_million', '—')} input / "
            f"${tariff.get('output_usd_per_million', '—')} output per 1M tokens · "
            f"{pricing.get('region', '—')} · {pricing.get('pricing_mode', '—')} · "
            f"{pricing.get('resolved_service_tier', '—')} "
            f"(requested {pricing.get('service_tier_request', '—')}) · "
            f"retrieved {cost.get('price_date', '—')} · {cost.get('price_source', '—')}"
            if isinstance(pricing, Mapping) and isinstance(tariff, Mapping)
            else "—"
        )
        projection = (
            f"{cost.get('projection', '—')} · shared run {cost.get('shared_run_id', '—')} · "
            f"counterfactual={cost.get('counterfactual_run_cost', '—')} · "
            f"additive across panels={cost.get('additive_across_panels', '—')} · "
            f"comparability={cost.get('cost_comparability_id', '—')}"
        )
        rows.append(
            [
                str(arm.get("label", arm.get("arm_id"))),
                str(cost.get("view_id")),
                str(cost.get("basis")),
                f"${numeric('inference_usd_lower')}",
                f"${numeric('inference_usd_upper')}",
                f"${numeric('usd_per_1k_statements_lower')}",
                f"${numeric('usd_per_1k_statements_upper')}",
                f"{cost.get('provider_measured_call_count', '—')} / "
                f"{cost.get('conservative_call_count', '—')}",
                token_text,
                price_contract,
                projection,
                rules,
            ]
        )
    return rows


COST_HEADERS = (
    "Arm",
    "Cost view",
    "Basis",
    "Total lower",
    "Total upper",
    "USD/1k lower",
    "USD/1k upper",
    "Measured / reserved calls",
    "Input / output tokens",
    "Provider tariff and service tier",
    "Projection and comparability",
    "Accounting rules",
)


def _literature_rows(literature: Mapping[str, Any], *, limit: int = 12) -> list[list[str]]:
    if literature.get("artifact_kind") != LITERATURE_KIND:
        raise ReportError(f"literature artifact_kind must be {LITERATURE_KIND!r}")
    methods = literature.get("methods")
    if not isinstance(methods, list):
        raise ReportError("literature methods must be an array")
    parsed: list[tuple[str, float, Any]] = []
    for row in methods:
        if not isinstance(row, Mapping) or not isinstance(row.get("method"), str):
            raise ReportError("literature contains an invalid method row")
        score = _number(row.get("fold_mean_trapezoidal_pr_auc"), context="literature PR-AUC")
        parsed.append((row["method"], score, row.get("fold_population_sd")))
    parsed.sort(key=lambda item: item[1], reverse=True)
    selected = parsed[:limit]
    for item in parsed:
        if item[0] == "Belief Orig - readers" and item not in selected:
            selected.append(item)
    return [
        [name, f"{score:.3f}", "—" if sd is None else f"{float(sd):.3f}"]
        for name, score, sd in selected
    ]


def _fmt_review_proportion(value: Any) -> str:
    return "—" if value is None else f"{float(value):.1%}"


def _error_review_sections(
    reviews: Sequence[Mapping[str, Any]],
    *,
    metrics: Mapping[str, Any],
    panels: Sequence[Mapping[str, Any]],
) -> tuple[list[str], list[str]]:
    if len(reviews) != 2:
        raise ReportError("error reviews must contain exactly the two canonical paper panels")
    parsed = [_canonical_error_review(value, index=index) for index, value in enumerate(reviews)]
    by_panel = {row["panel_id"]: row for row in parsed}
    if set(by_panel) != {"paper_all_source", "paper_readers"}:
        raise ReportError("error reviews must contain one all-source and one reader report")

    metrics_provenance = _mapping(metrics.get("provenance"), context="metrics.provenance")
    source_manifest_sha256 = _sha256_text(
        metrics_provenance.get("source_manifest_sha256"),
        context="metrics.provenance.source_manifest_sha256",
    )
    panels_by_id = {str(panel["substrate_id"]): panel for panel in panels}
    for index, review in enumerate(parsed):
        context = f"error_reviews[{index}]"
        panel = panels_by_id[review["panel_id"]]
        arm = next(
            (candidate for candidate in panel["arms"] if candidate.get("arm_id") == review["arm_id"]),
            None,
        )
        if arm is None:
            raise ReportError(f"{context} reviewed arm is absent from the exact metrics panel")
        contract = _mapping(panel.get("contract"), context=f"{context} metrics panel contract")
        arm_provenance = _mapping(
            arm.get("provenance"), context=f"{context} metrics arm provenance"
        )
        cost = _mapping(arm.get("cost"), context=f"{context} metrics arm cost")
        if cost.get("status") != "available":
            raise ReportError(f"{context} reviewed arm has no available execution ledger")
        expected_sha256s = {
            "spec": source_manifest_sha256,
            "bundle_manifest": _sha256_text(
                arm_provenance.get("implementation_digest"),
                context=f"{context} metrics arm implementation_digest",
            ),
            "gold": _sha256_text(
                contract.get("gold_sha256"),
                context=f"{context} metrics panel gold_sha256",
            ),
            "predictions": _sha256_text(
                arm_provenance.get("predictions_sha256"),
                context=f"{context} metrics arm predictions_sha256",
            ),
            "execution_ledger": _sha256_text(
                cost.get("ledger_sha256"),
                context=f"{context} metrics arm ledger_sha256",
            ),
        }
        labels = {
            "spec": "comparison spec",
            "bundle_manifest": "bundle",
            "gold": "gold",
            "predictions": "prediction",
            "execution_ledger": "execution-ledger",
        }
        for name in ERROR_REVIEW_BINDING_FILES:
            if review["comparison_sha256s"][name] != expected_sha256s[name]:
                raise ReportError(
                    f"{context} {labels[name]} digest differs from the supplied metrics artifact"
                )

    markdown = ["", "## Human error review"]
    html_sections = ["<h2>Human error review</h2>"]
    explanation = (
        "Every threshold error was classified independently by two blinded human reviewers, "
        "with disagreements resolved by a distinct human resolver. System-supported defensible "
        "errors are cases where the final human classification supports the model over the released "
        "reference; indeterminate-ambiguity defensible errors remain genuinely ambiguous and must "
        "not be interpreted as confirmed model correctness."
    )
    markdown.extend(["", explanation])
    html_sections.append(f"<p>{html.escape(explanation)}</p>")
    labels = {
        "paper_all_source": "Paper all-source",
        "paper_readers": "Paper readers",
    }
    for panel_id in ("paper_all_source", "paper_readers"):
        row = by_panel[panel_id]
        threshold = row["threshold_errors"]
        defensibility = row["defensibility"]
        summary = (
            f"{threshold['count']} of {row['evaluated_statements']} evaluated statements were "
            f"threshold errors ({_fmt_review_proportion(threshold['proportion'])}). Of those, "
            f"{defensibility['defensible']['count']} were defensible "
            f"({_fmt_review_proportion(defensibility['defensible']['proportion'])}): "
            f"{defensibility['system_supported_defensible']['count']} system-supported and "
            f"{defensibility['indeterminate_ambiguity_defensible']['count']} ambiguity-defensible. "
            f"{defensibility['non_defensible']['count']} were non-defensible "
            f"({_fmt_review_proportion(defensibility['non_defensible']['proportion'])})."
        )
        markdown.extend(["", f"### {labels[panel_id]}", "", summary])
        html_sections.extend(
            [f"<h3>{html.escape(labels[panel_id])}</h3>", f"<p>{html.escape(summary)}</p>"]
        )
        error_rows = []
        for error_type in ERROR_TYPES:
            error = row["error_types"][error_type]
            error_rows.append(
                [
                    error_type.replace("_", " "),
                    str(error["count"]),
                    _fmt_review_proportion(error["proportion"]),
                    f"{error['defensible']['count']} ({_fmt_review_proportion(error['defensible']['proportion'])})",
                    f"{error['non_defensible']['count']} ({_fmt_review_proportion(error['non_defensible']['proportion'])})",
                ]
            )
        error_headers = ("Error type", "Count", "Share of errors", "Defensible", "Non-defensible")
        markdown.extend(["", _md_table(error_headers, error_rows)])
        html_sections.append(_html_table(error_headers, error_rows))

        classification_rows = [
            [
                name.replace("_", " "),
                str(row["human_classifications"][name]["count"]),
                _fmt_review_proportion(row["human_classifications"][name]["proportion"]),
            ]
            for name in HUMAN_CLASSIFICATIONS
        ]
        classification_headers = ("Final human classification", "Count", "Share of errors")
        markdown.extend(["", _md_table(classification_headers, classification_rows)])
        html_sections.append(_html_table(classification_headers, classification_rows))

        dimension_rows = [
            [
                dimension["dimension"],
                str(dimension["count"]),
                _fmt_review_proportion(dimension["proportion"]),
                str(dimension["by_judgment"]["defensible"]),
                str(dimension["by_judgment"]["non_defensible"]),
                str(dimension["by_error_type"]["false_positive"]),
                str(dimension["by_error_type"]["false_negative"]),
            ]
            for dimension in row["dimensions"]
        ]
        if dimension_rows:
            dimension_headers = (
                "Error dimension",
                "Count",
                "Share of errors",
                "Defensible",
                "Non-defensible",
                "FP",
                "FN",
            )
            markdown.extend(["", _md_table(dimension_headers, dimension_rows)])
            html_sections.append(_html_table(dimension_headers, dimension_rows))
    return markdown, html_sections


def build_report(
    metrics: Mapping[str, Any],
    *,
    literature: Mapping[str, Any] | None = None,
    error_reviews: Sequence[Mapping[str, Any]] | None = None,
) -> tuple[str, str]:
    """Return Markdown and HTML backed by the same validated metrics object."""

    panels = _validate_metrics(metrics)
    provenance = metrics.get("provenance")
    resamples = provenance.get("bootstrap_resamples") if isinstance(provenance, Mapping) else None
    diagnostic_note = (
        f"DIAGNOSTIC ONLY: this artifact uses {resamples:,} paired bootstrap resamples. "
        "Publication requires 10,000; no parity, superiority, or deployment claim is released."
        if isinstance(resamples, int) and resamples < 10_000
        else None
    )
    markdown: list[str] = [
        "# INDRA belief-system comparison",
        "",
        *([f"> **{diagnostic_note}**", ""] if diagnostic_note else []),
        "Direct statement-level results use the same frozen gold and statement IDs within each panel. "
        "Intervals are paired within panel; Pareto sets never mix panels. "
        "All-source and reader LLM costs are views of the same paid run and must not be added.",
    ]
    html_sections: list[str] = [
        "<h1>INDRA belief-system comparison</h1>",
        *(
            [f"<aside class='diagnostic'><strong>{html.escape(diagnostic_note)}</strong></aside>"]
            if diagnostic_note
            else []
        ),
        "<p>Direct statement-level results use the same frozen gold and statement IDs within each panel. "
        "Intervals are paired within panel; Pareto sets never mix panels. "
        "All-source and reader LLM costs are views of the same paid run and must not be added.</p>",
    ]

    for panel in panels:
        label = str(panel.get("label", panel["substrate_id"]))
        summary = (
            f"n={panel.get('n_evaluable')}; positives={panel.get('n_positive')}; "
            f"negatives={panel.get('n_negative')}"
        )
        rows = _arm_rows(panel)
        audit_text = _audit_text(panel)
        markdown.extend(
            ["", f"## {label}", "", summary, "", audit_text, "", "### Primary released-label results", "", _md_table(ARM_HEADERS, rows)]
        )
        html_sections.extend(
            [
                f"<h2>{html.escape(label)}</h2>",
                f"<p>{html.escape(summary)}</p>",
                f"<p>{html.escape(audit_text)}</p>",
                "<h3>Primary released-label results</h3>",
                _html_table(ARM_HEADERS, rows),
            ]
        )
        cost_rows = _structured_cost_rows(panel)
        if cost_rows:
            markdown.extend(["", "### Structured inference cost", "", _md_table(COST_HEADERS, cost_rows)])
            html_sections.extend(["<h3>Structured inference cost</h3>", _html_table(COST_HEADERS, cost_rows)])
        paired_rows = _paired_rows(panel)
        if paired_rows:
            paired_note = "Delta is second arm minus first arm on the paper-compatible mean-fold trapezoidal PR-AUC."
            headers = ("First arm", "Second arm", "Paired Δ PR-AUC", "Better when")
            markdown.extend(["", "### Paired primary deltas", "", paired_note, "", _md_table(headers, paired_rows)])
            html_sections.extend(["<h3>Paired primary deltas</h3>", f"<p>{html.escape(paired_note)}</p>", _html_table(headers, paired_rows)])

        sensitivity = panel["strict_e0_resolved_sensitivity"]
        sensitivity_summary = (
            f"Fixed resolved-only sensitivity: n={sensitivity.get('n_evaluable')}; "
            f"positives={sensitivity.get('n_positive')}; negatives={sensitivity.get('n_negative')}; "
            f"excluded unresolved={sensitivity.get('excluded_unresolved')}. Cost and Pareto are not "
            "recomputed or duplicated for this selected subset."
        )
        sensitivity_rows = [row[:11] for row in _arm_rows(sensitivity)]
        markdown.extend(
            ["", "### Strict E0 resolved-only sensitivity", "", sensitivity_summary, "", _md_table(ARM_HEADERS[:11], sensitivity_rows)]
        )
        html_sections.extend(
            ["<h3>Strict E0 resolved-only sensitivity</h3>", f"<p>{html.escape(sensitivity_summary)}</p>", _html_table(ARM_HEADERS[:11], sensitivity_rows)]
        )
        sensitivity_paired_rows = _paired_rows(sensitivity)
        if sensitivity_paired_rows:
            headers = ("First arm", "Second arm", "Paired Δ PR-AUC", "Better when")
            markdown.extend(["", "#### Paired strict-sensitivity deltas", "", _md_table(headers, sensitivity_paired_rows)])
            html_sections.extend(["<h4>Paired strict-sensitivity deltas</h4>", _html_table(headers, sensitivity_paired_rows)])
        excluded = panel.get("excluded_arms")
        if isinstance(excluded, list) and excluded:
            excluded_rows = [
                [str(row.get("label", row.get("arm_id", ""))), str(row.get("reason", ""))]
                for row in excluded
                if isinstance(row, Mapping)
            ]
            markdown.extend(["", "### Exclusions", "", _md_table(("Arm", "Reason"), excluded_rows)])
            html_sections.extend(["<h3>Exclusions</h3>", _html_table(("Arm", "Reason"), excluded_rows)])

    if literature is not None:
        rows = _literature_rows(literature)
        note = (
            "The 2023 table is contextual: it uses the paper's reported fold summaries. "
            "Only arms re-evaluated on the shared statement-level gold are direct paired comparisons."
        )
        markdown.extend(["", "## 2023 published landscape", "", note, "", _md_table(("Method", "PR-AUC", "Fold SD"), rows)])
        html_sections.extend(["<h2>2023 published landscape</h2>", f"<p>{html.escape(note)}</p>", _html_table(("Method", "PR-AUC", "Fold SD"), rows)])

    if error_reviews is not None:
        review_markdown, review_html = _error_review_sections(
            error_reviews,
            metrics=metrics,
            panels=panels,
        )
        markdown.extend(review_markdown)
        html_sections.extend(review_html)

    css = """
body{font:15px/1.45 system-ui,sans-serif;max-width:1500px;margin:2rem auto;padding:0 1rem;color:#17212b}
.diagnostic{border:2px solid #9b2c2c;background:#fff5f5;padding:.8rem 1rem;margin:1rem 0;color:#742a2a}
table{border-collapse:collapse;width:100%;margin:1rem 0 2rem;font-variant-numeric:tabular-nums}
th,td{border-bottom:1px solid #d8dee4;padding:.45rem .55rem;text-align:left;vertical-align:top}
th{position:sticky;top:0;background:#f6f8fa} h1,h2,h3{line-height:1.15}
""".strip()
    html_doc = (
        "<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' "
        f"content='width=device-width,initial-scale=1'><title>INDRA belief comparison</title><style>{css}</style>"
        "</head><body>" + "".join(html_sections) + "</body></html>\n"
    )
    return "\n".join(markdown).rstrip() + "\n", html_doc


def render_reports(
    metrics_path: Path,
    *,
    markdown_path: Path,
    html_path: Path,
    manifest_path: Path,
    literature_path: Path | None = None,
    error_review_paths: Sequence[Path] = (),
) -> dict[str, Any]:
    metrics = _load_object(metrics_path, label="metrics")
    literature = _load_object(literature_path, label="literature") if literature_path else None
    error_reviews = [
        _load_object(path, label=f"error review {index}")
        for index, path in enumerate(error_review_paths)
    ]
    markdown, html_doc = build_report(
        metrics,
        literature=literature,
        error_reviews=error_reviews or None,
    )
    _atomic_write(markdown_path, markdown.encode("utf-8"))
    _atomic_write(html_path, html_doc.encode("utf-8"))
    manifest = {
        "kind": "comparison_report_bundle",
        "inputs": {
            "metrics": {"path": str(metrics_path), "sha256": _sha256(metrics_path)},
            "literature": (
                {"path": str(literature_path), "sha256": _sha256(literature_path)}
                if literature_path
                else None
            ),
            "error_reviews": [
                {"path": str(path), "sha256": _sha256(path)}
                for path in error_review_paths
            ],
        },
        "outputs": {
            "markdown": {"path": str(markdown_path), "sha256": _sha256(markdown_path)},
            "html": {"path": str(html_path), "sha256": _sha256(html_path)},
        },
    }
    payload = json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False).encode("utf-8") + b"\n"
    _atomic_write(manifest_path, payload)
    return manifest
