"""Cost x error-detection frontier on curator gold.

With no arguments this reproduces the historical external-curator analysis:

    PYTHONPATH=src .venv/bin/python scripts/frontier_table.py

The inputs are parameterized so the same calculation can be applied to a new
benchmark without copying this script.  A scoring output is append-only: retry
attempts can therefore repeat a logical ``(stmt_i, evidence_i)`` row.  Metrics
use the latest attempt for each logical row, while cost intentionally includes
the calls from *every* attempt.

The formal point-Pareto frontier is deliberately narrow: complete Bedrock runs
whose calls all resolve through the repository's ``list`` rate bucket.  Some
entries in that bucket have documented second-party sources; this is a
normalized cost landscape, not an invoice reconciliation.  Non-Bedrock runs
with proxy costs (notably MedPsy) remain in the report and in a separately
labelled reference frontier, but are not silently mixed into the formal cost
comparison.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from eval_curation_compare import build_gold_index, gold_for  # noqa: E402
from indra_belief.corpus.cost import price_for  # noqa: E402
from indra_belief.curation import is_gold_correct  # noqa: E402
from indra_belief.metrics import confusion_pr  # noqa: E402

DEFAULT_GOLD = ROOT / "data" / "benchmark" / "external_curator_gold_v1.jsonl"
DEFAULT_RUNS_GLOB = str(ROOT / "data" / "results" / "external_curator_v1_*.jsonl")
DEFAULT_LABEL = "external-578"
DEFAULT_DENOMINATOR = 587
DEFAULT_BOOTSTRAP_SAMPLES = 2_000
DEFAULT_SEED = 20260630
MASK = (1 << 64) - 1
VALID_VERDICTS = frozenset({"correct", "incorrect"})


def _sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    try:
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        return None
    return digest.hexdigest()


def _display_path(path: Path) -> str:
    """Prefer a stable repository-relative path in serialized reports."""
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path)


def _load_jsonl(path: Path, *, strict: bool) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    invalid = 0
    with path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                if strict:
                    raise ValueError(f"invalid JSON in {path}:{line_no}: {exc}") from exc
                invalid += 1
                continue
            if not isinstance(value, dict):
                if strict:
                    raise ValueError(f"expected an object in {path}:{line_no}")
                invalid += 1
                continue
            rows.append(value)
    return rows, invalid


def _hash_int(value: Any, *, hexadecimal_string: bool = False) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        if isinstance(value, str):
            raw = value.strip()
            if not raw:
                return None
            base = 16 if hexadecimal_string and not raw.startswith("-") else 10
            if raw.lower().startswith(("0x", "-0x")):
                base = 16
            return int(raw, base) & MASK
        return int(value) & MASK
    except (TypeError, ValueError, OverflowError):
        return None


def _logical_row_key(row: dict[str, Any], ordinal: int) -> tuple[Any, ...]:
    """Identify a retry without collapsing legitimate duplicate evidence rows.

    Runner indices are the authoritative grain.  Older outputs can lack those
    indices, so an exact statement/evidence hash pair is the safe fallback.
    Source hash alone is intentionally *not* used because one sentence can
    support multiple statements.  Unidentifiable rows remain distinct.
    """
    if row.get("stmt_i") is not None and row.get("evidence_i") is not None:
        try:
            return ("index", int(row["stmt_i"]), int(row["evidence_i"]))
        except (TypeError, ValueError, OverflowError):
            pass

    if row.get("stmt_hash") is not None:
        mh = _hash_int(row.get("stmt_hash"), hexadecimal_string=True)
    else:
        mh = _hash_int(row.get("matches_hash"))
    sh = _hash_int(row.get("source_hash"))
    if mh is not None and sh is not None:
        return ("pair", mh, sh)
    return ("unidentified", ordinal)


def canonicalize_attempts(
    rows: Iterable[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    """Keep the latest append attempt, retaining the first-seen corpus order."""
    latest: dict[tuple[Any, ...], tuple[int, dict[str, Any]]] = {}
    n_rows = 0
    for ordinal, row in enumerate(rows):
        n_rows += 1
        key = _logical_row_key(row, ordinal)
        first_ordinal = latest[key][0] if key in latest else ordinal
        latest[key] = (first_ordinal, row)
    canonical = [row for _, row in sorted(latest.values(), key=lambda item: item[0])]
    return canonical, n_rows - len(canonical)


def _f1(pairs: Iterable[tuple[bool, bool]]) -> float:
    return float(confusion_pr([(bool(g), bool(p)) for g, p in pairs])["f1"])


def _bootstrap_ci(
    pairs: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> tuple[float, float] | None:
    if not len(pairs) or samples <= 0:
        return None
    # Reset per model.  Complete models with the same corpus then receive the
    # same resample indices, and a model's interval cannot change merely because
    # another file was added to the glob.
    rng = np.random.default_rng(seed)
    boot = np.array(
        [_f1(pairs[rng.integers(0, len(pairs), len(pairs))]) for _ in range(samples)],
        dtype=float,
    )
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(lo), float(hi)


def _read_meta(run: Path) -> dict[str, Any]:
    meta_path = run.with_suffix(".meta.json")
    if not meta_path.exists():
        return {}
    try:
        value = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _run_identity(run: Path, meta: dict[str, Any]) -> tuple[str, str | None, str]:
    served_model = str(meta["model"]) if meta.get("model") else None
    if served_model:
        name = served_model.removeprefix("bedrock-")
    else:
        name = run.stem
        name = name.replace("external_curator_v1_bedrock-", "")
        name = name.replace("external_curator_v1_", "")

    stem = run.stem.lower()
    is_bedrock = bool(
        (served_model and served_model.lower().startswith("bedrock-"))
        or "_bedrock-" in stem
    )
    return name, served_model, "bedrock" if is_bedrock else "non_bedrock"


def _nonnegative_number(value: Any) -> float:
    if isinstance(value, bool):
        return 0.0
    try:
        return max(0.0, float(value or 0))
    except (TypeError, ValueError, OverflowError):
        return 0.0


def _cost_summary(all_attempt_rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    prompt_tokens = 0.0
    output_tokens = 0.0
    call_count = 0
    known_cost = 0.0
    bases: set[str] = set()
    model_ids: Counter[str] = Counter()
    unpriced_calls = 0

    for row in all_attempt_rows:
        for call in row.get("call_log") or []:
            if not isinstance(call, dict):
                unpriced_calls += 1
                continue
            call_count += 1
            ti = _nonnegative_number(call.get("prompt_tokens"))
            to = _nonnegative_number(call.get("out_tokens"))
            prompt_tokens += ti
            output_tokens += to
            model_id = str(call.get("model_id") or "").strip()
            if model_id:
                model_ids[model_id] += 1
            price = price_for(model_id) if model_id else None
            if price is None:
                unpriced_calls += 1
                continue
            in_per_m, out_per_m, basis = price
            bases.add(basis)
            known_cost += (ti * in_per_m + to * out_per_m) / 1_000_000

    if not call_count:
        basis: str | None = None
        cost: float | None = None
    elif unpriced_calls:
        basis = "unavailable"
        cost = None
    elif len(bases) == 1:
        basis = next(iter(bases))
        cost = known_cost
    else:
        basis = "mixed"
        cost = known_cost

    return {
        "call_count": call_count,
        "prompt_tokens": int(prompt_tokens),
        "output_tokens": int(output_tokens),
        "model_ids": dict(sorted(model_ids.items())),
        "price_bases": sorted(bases),
        "unpriced_calls": unpriced_calls,
        "known_priced_cost_usd": known_cost if call_count else None,
        "cost_basis": basis,
        "cost_is_estimate": "estimate" in bases,
        "cost_usd": cost,
    }


def _score_run(
    run: Path,
    *,
    by_pair: dict,
    by_source: dict,
    denominator: int,
    bootstrap_samples: int,
    seed: int,
    require_valid_coverage: bool,
) -> tuple[dict[str, Any], list[bool]]:
    attempts, invalid_json_lines = _load_jsonl(run, strict=False)
    canonical, retry_rows = canonicalize_attempts(attempts)
    meta = _read_meta(run)
    meta_path = run.with_suffix(".meta.json")
    name, served_model, serving_scope = _run_identity(run, meta)
    input_raw = meta.get("input")
    input_path = Path(str(input_raw)) if input_raw else None
    if input_path is not None and not input_path.is_absolute():
        input_path = ROOT / input_path

    joined: list[tuple[dict[str, Any], dict[str, Any]]] = []
    parse_nulls = 0
    row_errors = 0
    unmatched = 0
    for scored in canonical:
        gold = gold_for(scored, by_pair, by_source)
        if gold is None:
            unmatched += 1
            continue
        if scored.get("row_status") == "error":
            row_errors += 1
        if scored.get("verdict") not in VALID_VERDICTS:
            parse_nulls += 1
            continue
        joined.append((gold, scored))

    pairs = np.asarray(
        [
            (not is_gold_correct(gold["tag"]), scored["verdict"] == "incorrect")
            for gold, scored in joined
        ],
        dtype=bool,
    )
    error_labels = [bool(pair[0]) for pair in pairs]
    metric = confusion_pr([(bool(g), bool(p)) for g, p in pairs]) if len(pairs) else None
    ci = _bootstrap_ci(pairs, samples=bootstrap_samples, seed=seed)
    cost = _cost_summary(attempts)
    usd_1k = (
        cost["cost_usd"] / denominator * 1_000
        if cost["cost_usd"] is not None
        else None
    )

    matched_rows = len(joined) + parse_nulls
    run_coverage_complete = matched_rows == denominator
    metric_coverage_complete = len(joined) == denominator
    exclusions: list[str] = []
    if not run_coverage_complete:
        exclusions.append("incomplete_run_coverage")
    if require_valid_coverage and not metric_coverage_complete:
        exclusions.append("incomplete_valid_coverage")
    if metric is None:
        exclusions.append("metric_unavailable")
    if unmatched:
        exclusions.append("unmatched_rows")
    if invalid_json_lines:
        exclusions.append("invalid_json_lines")
    if serving_scope != "bedrock":
        exclusions.append("non_bedrock_serving_scope")
    if cost["cost_usd"] is None:
        exclusions.append("cost_unavailable")
    elif cost["cost_basis"] != "list":
        exclusions.append("cost_not_repo_list_rate")

    result = {
        "name": name,
        "served_model": served_model,
        "serving_scope": serving_scope,
        "run_id": meta.get("run_id"),
        "run_status": meta.get("status"),
        "architecture": meta.get("arch"),
        "input_path": _display_path(input_path) if input_path else None,
        "input_sha256": _sha256_file(input_path) if input_path else None,
        "run_path": _display_path(run),
        "run_sha256": _sha256_file(run),
        "meta_path": _display_path(meta_path) if meta_path.exists() else None,
        "meta_sha256": _sha256_file(meta_path),
        "attempt_rows": len(attempts),
        "canonical_rows": len(canonical),
        "retry_rows": retry_rows,
        "invalid_json_lines": invalid_json_lines,
        "n": len(joined),
        "matched_rows": matched_rows,
        "run_coverage_fraction": matched_rows / denominator,
        "run_coverage_complete": run_coverage_complete,
        "coverage_fraction": len(joined) / denominator,
        "coverage_complete": metric_coverage_complete,
        "missing_valid_rows": max(0, denominator - len(joined)),
        "parse_nulls": parse_nulls,
        "row_errors": row_errors,
        "unmatched_rows": unmatched,
        "f1": float(metric["f1"]) if metric else None,
        "precision": float(metric["p"]) if metric else None,
        "recall": float(metric["r"]) if metric else None,
        "accuracy": float(metric["acc"]) if metric else None,
        "tp": int(metric["tp"]) if metric else None,
        "fp": int(metric["fp"]) if metric else None,
        "fn": int(metric["fn"]) if metric else None,
        "tn": int(metric["tn"]) if metric else None,
        "lo": ci[0] if ci else None,
        "hi": ci[1] if ci else None,
        **cost,
        "usd_1k": usd_1k,
        "frontier_eligible": not exclusions,
        "frontier_exclusion_reasons": exclusions,
        "pareto_point": None,
        "reference_frontier_eligible": False,
        "reference_pareto_point": None,
    }
    return result, error_labels


def _mark_pareto(rows: list[dict[str, Any]], *, marker: str, eligibility: str) -> None:
    eligible = [
        row
        for row in rows
        if row[eligibility] and row["f1"] is not None and row["usd_1k"] is not None
    ]
    for row in eligible:
        row[marker] = not any(
            other is not row
            and other["usd_1k"] <= row["usd_1k"]
            and other["f1"] >= row["f1"]
            and (
                other["usd_1k"] < row["usd_1k"]
                or other["f1"] > row["f1"]
            )
            for other in eligible
        )


def _always_incorrect_baseline(
    rows: list[dict[str, Any]],
    labels_by_run: dict[str, list[bool]],
    gold_rows: list[dict[str, Any]],
    denominator: int,
) -> dict[str, Any]:
    labels: list[bool] | None = None
    source: str | None = None
    for row in rows:
        candidate = labels_by_run[row["run_path"]]
        if row["coverage_complete"] and len(candidate) == denominator:
            labels = candidate
            source = row["run_path"]
            break
    if labels is None and len(gold_rows) == denominator:
        labels = [not is_gold_correct(row["tag"]) for row in gold_rows]
        source = "gold_rows"
    if labels is None:
        return {
            "name": "always-incorrect",
            "n": None,
            "precision": None,
            "recall": None,
            "f1": None,
            "accuracy": None,
            "source": None,
            "note": "unavailable: no complete run and gold-row count differs from denominator",
        }

    metric = confusion_pr([(label, True) for label in labels])
    return {
        "name": "always-incorrect",
        "n": len(labels),
        "precision": float(metric["p"]),
        "recall": float(metric["r"]),
        "f1": float(metric["f1"]),
        "accuracy": float(metric["acc"]),
        "source": source,
        "note": "reference baseline only; excluded from the model Pareto frontier",
    }


def build_report(
    *,
    gold_path: Path,
    run_paths: Iterable[Path],
    label: str,
    denominator: int,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    seed: int = DEFAULT_SEED,
    require_valid_coverage: bool = False,
) -> dict[str, Any]:
    if denominator <= 0:
        raise ValueError("denominator must be positive")
    if bootstrap_samples < 0:
        raise ValueError("bootstrap_samples must be non-negative")

    gold_rows, _ = _load_jsonl(gold_path, strict=True)
    by_pair, by_source = build_gold_index(gold_rows)
    rows: list[dict[str, Any]] = []
    labels_by_run: dict[str, list[bool]] = {}
    for run in sorted({Path(path) for path in run_paths}, key=lambda path: str(path)):
        scored, labels = _score_run(
            run,
            by_pair=by_pair,
            by_source=by_source,
            denominator=denominator,
            bootstrap_samples=bootstrap_samples,
            seed=seed,
            require_valid_coverage=require_valid_coverage,
        )
        rows.append(scored)
        labels_by_run[scored["run_path"]] = labels

    rows.sort(
        key=lambda row: (
            row["f1"] is None,
            -(row["f1"] if row["f1"] is not None else 0.0),
            row["name"],
        )
    )

    # Estimated prices are useful context, but the formal frontier is restricted
    # to complete Bedrock/list-price runs.  The reference frontier makes the
    # sensitivity to including proxy prices explicit rather than implicit.
    for row in rows:
        row["reference_frontier_eligible"] = bool(
            row["run_coverage_complete"]
            and (not require_valid_coverage or row["coverage_complete"])
            and not row["unmatched_rows"]
            and not row["invalid_json_lines"]
            and row["cost_usd"] is not None
            and row["cost_basis"] in {"list", "estimate"}
        )
    _mark_pareto(rows, marker="pareto_point", eligibility="frontier_eligible")
    _mark_pareto(
        rows,
        marker="reference_pareto_point",
        eligibility="reference_frontier_eligible",
    )

    formal = [row for row in rows if row["frontier_eligible"]]
    formal_front = sorted(
        [row for row in formal if row["pareto_point"]], key=lambda row: row["usd_1k"]
    )
    reference_front = sorted(
        [row for row in rows if row["reference_pareto_point"]],
        key=lambda row: row["usd_1k"],
    )
    best = max(formal, key=lambda row: row["f1"], default=None)
    value = max(
        formal,
        key=lambda row: row["f1"] / max(row["usd_1k"], 1e-12),
        default=None,
    )
    baseline = _always_incorrect_baseline(rows, labels_by_run, gold_rows, denominator)

    unique_gold_pairs = len(
        {
            (_hash_int(row.get("matches_hash")), _hash_int(row.get("source_hash")))
            for row in gold_rows
        }
    )
    coverage_rule = (
        "complete valid metric coverage (no parser nulls)"
        if require_valid_coverage
        else "complete benchmark-row coverage (parser nulls reported separately)"
    )
    return {
        "schema_version": 1,
        "label": label,
        "gold": {
            "path": _display_path(gold_path),
            "sha256": _sha256_file(gold_path),
            "rows": len(gold_rows),
            "unique_exact_pairs": unique_gold_pairs,
        },
        "denominator": denominator,
        "require_valid_coverage": require_valid_coverage,
        "metric": {
            "name": "error_detection_f1",
            "positive_class": "curator label is incorrect",
            "positive_prediction": "model verdict is incorrect",
            "bootstrap_samples": bootstrap_samples,
            "bootstrap_seed": seed,
            "ci": "unclustered row bootstrap, percentile 95%",
        },
        "pareto": {
            "method": "point estimates; confidence intervals are descriptive, not a dominance rule",
            "cost_axis": "USD per 1,000 denominator rows",
            "formal_eligibility": (
                f"{coverage_rule}; no unmatched/invalid rows; Bedrock serving scope; all "
                "calls resolved by the repository's list-rate bucket"
            ),
            "price_caveat": (
                "Repository rate-table normalization, not invoice reconciliation; the cost "
                "module documents second-party sources for some list-bucket entries"
            ),
            "formal_frontier": [row["name"] for row in formal_front],
            "reference_frontier_including_estimates": [
                row["name"] for row in reference_front
            ],
        },
        "always_incorrect_baseline": baseline,
        "models": rows,
        "best_f1": (
            {"name": best["name"], "f1": best["f1"], "usd_1k": best["usd_1k"]}
            if best
            else None
        ),
        "value_champion": (
            {"name": value["name"], "f1": value["f1"], "usd_1k": value["usd_1k"]}
            if value
            else None
        ),
        "formal_fleet_cost_usd": sum(
            row["cost_usd"] for row in formal if row["cost_usd"] is not None
        ),
    }


def _number(value: float | None, digits: int = 3) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def _money(value: float | None, *, estimate: bool = False) -> str:
    if value is None:
        return "—"
    return f"{'~' if estimate else ''}${value:.2f}"


def render_text(report: dict[str, Any]) -> str:
    lines = [
        f"=== cost x error-F1 frontier — {report['label']} "
        f"(denominator={report['denominator']}, gold_rows={report['gold']['rows']}) ===",
        (
            f"{'model':>26} {'coverage':>10} {'null':>5} {'error-F1 [95%]':>22} "
            f"{'P':>6} {'R':>6} {'acc':>6} {'$/run':>9} {'$/1k ev':>9} "
            f"{'basis':>10} {'Pareto':>7}"
        ),
    ]
    for row in report["models"]:
        interval = (
            f"{row['f1']:.3f} [{row['lo']:.3f},{row['hi']:.3f}]"
            if row["f1"] is not None and row["lo"] is not None
            else "—"
        )
        coverage = f"{row['n']}/{report['denominator']}"
        estimated = bool(row["cost_is_estimate"])
        star = "★" if row["pareto_point"] else ""
        lines.append(
            f"{row['name']:>26} {coverage:>10} {row['parse_nulls']:>5} "
            f"{interval:>22} {_number(row['precision']):>6} {_number(row['recall']):>6} "
            f"{_number(row['accuracy']):>6} {_money(row['cost_usd'], estimate=estimated):>9} "
            f"{_money(row['usd_1k'], estimate=estimated):>9} "
            f"{str(row['cost_basis'] or '—'):>10} {star:>7}"
        )

    baseline = report["always_incorrect_baseline"]
    lines.extend(
        [
            "",
            (
                "  always-incorrect baseline: "
                f"F1={_number(baseline['f1'])}, P={_number(baseline['precision'])}, "
                f"R={_number(baseline['recall'])}, acc={_number(baseline['accuracy'])}"
            ),
        ]
    )
    if report["best_f1"]:
        best = report["best_f1"]
        value = report["value_champion"]
        lines.extend(
            [
                f"  best F1:        {best['name']} {best['f1']:.3f} (${best['usd_1k']:.2f}/1k)",
                f"  value champion: {value['name']} {value['f1']:.3f} (${value['usd_1k']:.2f}/1k)",
            ]
        )
    lines.append(
        "  formal point-Pareto frontier: "
        + (", ".join(report["pareto"]["formal_frontier"]) or "none")
    )
    if (
        report["pareto"]["reference_frontier_including_estimates"]
        != report["pareto"]["formal_frontier"]
    ):
        lines.append(
            "  reference frontier incl. estimates: "
            + ", ".join(report["pareto"]["reference_frontier_including_estimates"])
        )
    lines.append("  ~ = proxy/estimated cost; estimates are excluded from the formal frontier")
    lines.append("  costs use the repository rate table; see its source notes (not invoice reconciliation)")
    return "\n".join(lines) + "\n"


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        f"# Cost × error-detection frontier — {report['label']}",
        "",
        (
            f"Gold: `{report['gold']['path']}` ({report['gold']['rows']} rows; "
            f"{report['gold']['unique_exact_pairs']} unique exact pairs). "
            f"Evaluation denominator: **{report['denominator']}**."
        ),
        "",
        (
            "Positive class is curator-flagged incorrect; positive prediction is "
            "`verdict == \"incorrect\"`. Intervals are deterministic unclustered "
            f"row-bootstrap 95% CIs ({report['metric']['bootstrap_samples']:,} resamples; "
            f"seed {report['metric']['bootstrap_seed']})."
        ),
        "",
        "| Model | Valid coverage | Nulls | Error F1 [95% CI] | P | R | Accuracy | Cost/run | Cost/1k | Basis | Formal point-Pareto |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|:---:|",
    ]
    for row in report["models"]:
        interval = (
            f"{row['f1']:.3f} [{row['lo']:.3f}, {row['hi']:.3f}]"
            if row["f1"] is not None and row["lo"] is not None
            else "—"
        )
        estimated = bool(row["cost_is_estimate"])
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["name"]).replace("|", "\\|"),
                    f"{row['n']}/{report['denominator']} ({row['coverage_fraction']:.1%})",
                    str(row["parse_nulls"]),
                    interval,
                    _number(row["precision"]),
                    _number(row["recall"]),
                    _number(row["accuracy"]),
                    _money(row["cost_usd"], estimate=estimated),
                    _money(row["usd_1k"], estimate=estimated),
                    str(row["cost_basis"] or "—"),
                    "★" if row["pareto_point"] else ("ineligible" if not row["frontier_eligible"] else ""),
                ]
            )
            + " |"
        )

    baseline = report["always_incorrect_baseline"]
    lines.extend(
        [
            "",
            (
                f"Always-incorrect reference: F1 **{_number(baseline['f1'])}**, "
                f"precision {_number(baseline['precision'])}, recall {_number(baseline['recall'])}, "
                f"accuracy {_number(baseline['accuracy'])}. It is a baseline, not a frontier candidate."
            ),
            "",
            "Formal point-Pareto frontier: **"
            + (", ".join(report["pareto"]["formal_frontier"]) or "none")
            + "**.",
            "",
            (
                "Reference frontier when estimate-priced runs are admitted: **"
                + (
                    ", ".join(report["pareto"]["reference_frontier_including_estimates"])
                    or "none"
                )
                + "**."
            ),
            "",
            (
                "The Pareto rule uses F1 and cost point estimates only; confidence intervals are "
                "descriptive. `~$` marks proxy/estimated cost. Formal eligibility: "
                + report["pareto"]["formal_eligibility"]
                + ". Retry calls are included in cost even though only the latest retry is scored."
            ),
            "",
            "Pricing caveat: " + report["pareto"]["price_caveat"] + ".",
        ]
    )

    excluded = [row for row in report["models"] if not row["frontier_eligible"]]
    if excluded:
        lines.extend(["", "## Formal-frontier exclusions", ""])
        for row in excluded:
            lines.append(
                f"- `{row['name']}`: " + ", ".join(row["frontier_exclusion_reasons"])
            )
    return "\n".join(lines) + "\n"


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--runs-glob", default=DEFAULT_RUNS_GLOB)
    parser.add_argument("--label", default=DEFAULT_LABEL)
    parser.add_argument(
        "--denominator",
        type=int,
        default=DEFAULT_DENOMINATOR,
        help="expected number of scoreable evidence rows (historical default: 587)",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=DEFAULT_BOOTSTRAP_SAMPLES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--require-valid-coverage",
        action="store_true",
        help=(
            "require n_valid == denominator (and therefore zero parser nulls) for "
            "formal/reference Pareto eligibility; historical default permits explicit nulls"
        ),
    )
    parser.add_argument("--json-out", "--output-json", dest="json_out", type=Path)
    parser.add_argument("--md-out", "--output-md", dest="md_out", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    run_paths = [
        Path(path)
        for path in glob.glob(args.runs_glob)
        if "progress" not in Path(path).name
    ]
    report = build_report(
        gold_path=args.gold,
        run_paths=run_paths,
        label=args.label,
        denominator=args.denominator,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
        require_valid_coverage=args.require_valid_coverage,
    )
    sys.stdout.write(render_text(report))
    if args.json_out:
        _write(args.json_out, json.dumps(report, indent=2, sort_keys=True) + "\n")
    if args.md_out:
        _write(args.md_out, render_markdown(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
