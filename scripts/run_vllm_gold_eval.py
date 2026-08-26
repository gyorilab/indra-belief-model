"""Run the monolithic scorer against human gold through a vLLM endpoint.

The runner is deliberately evidence-pair/grading oriented:

* one input JSONL row is one stable job (``row_index``);
* results are appended as each request completes;
* resume skips valid verdicts and retries parser-null/error rows by default;
* accuracy, coverage, strict end-to-end accuracy, and the confusion matrix are
  recomputed from the latest attempt for every row;
* output metadata pins the input hash, model registry entry, prompt variant,
  and active system-prompt hash.

``--variant baseline`` selects the baseline prompt path. The production default,
``disconfirm_relnature_rf``, uses a different prompt path.

Examples
--------
Small smoke test::

    PYTHONPATH=src python scripts/run_vllm_gold_eval.py \
      --gold data/benchmark/eval_curation_v1.jsonl \
      --model vllm-gemma-4-26b --variant baseline --workers 8 --limit 20

Resume the same output (the default behavior)::

    PYTHONPATH=src python scripts/run_vllm_gold_eval.py \
      --gold data/benchmark/eval_curation_v1.jsonl \
      --model vllm-gemma-4-26b --variant baseline --workers 16

Run the legacy large holdout::

    PYTHONPATH=src python scripts/run_vllm_gold_eval.py \
      --gold data/benchmark/holdout_large.jsonl \
      --model vllm-gemma-4-26b --variant baseline --workers 16 \
      --output data/results/vllm_baseline_holdout_large.jsonl
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import signal
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

VALID_VERDICTS = {"correct", "incorrect"}
STOP_REQUESTED = False


def _request_stop(signum, _frame) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print(
        f"\nreceived signal {signum}; stopping submission and draining in-flight requests",
        file=sys.stderr,
        flush=True,
    )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as fh:
        for lineno, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON: {exc}") from exc
    return rows


def _gold_label(row: dict[str, Any]) -> str:
    """Return the binary human-gold label for one benchmark row."""
    explicit = row.get("gold")
    if explicit in VALID_VERDICTS:
        return str(explicit)
    return "correct" if row.get("tag") == "correct" else "incorrect"


def load_latest_attempts(path: Path) -> tuple[dict[int, dict[str, Any]], int]:
    """Load the latest valid JSON object for each stable ``row_index``."""
    latest: dict[int, dict[str, Any]] = {}
    corrupt_lines = 0
    if not path.exists():
        return latest, corrupt_lines
    with path.open() as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                row_index = int(row["row_index"])
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                corrupt_lines += 1
                continue
            latest[row_index] = row
    return latest, corrupt_lines


def _is_done(
    row: dict[str, Any],
    *,
    retry_parser_nulls: bool,
    retry_errors: bool,
) -> bool:
    if row.get("verdict") in VALID_VERDICTS:
        return True
    status = row.get("row_status")
    if status == "unmatched":
        return True
    if status == "error":
        return not retry_errors
    if status == "parser_null":
        return not retry_parser_nulls
    return False


def compute_metrics(
    gold_rows: list[dict[str, Any]],
    latest: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    """Compute coverage-aware metrics from the latest attempt per input row."""
    total = len(gold_rows)
    valid: list[tuple[bool, bool]] = []
    status_counts: Counter[str] = Counter()
    tier_counts: Counter[str] = Counter()
    correct_predictions = 0

    for row_index, gold_row in enumerate(gold_rows):
        out = latest.get(row_index)
        if out is None:
            status_counts["not_attempted"] += 1
            continue
        status_counts[str(out.get("row_status") or "unknown")] += 1
        if out.get("tier"):
            tier_counts[str(out["tier"])] += 1
        verdict = out.get("verdict")
        if verdict not in VALID_VERDICTS:
            continue
        gold_correct = _gold_label(gold_row) == "correct"
        pred_correct = verdict == "correct"
        valid.append((gold_correct, pred_correct))
        correct_predictions += int(gold_correct == pred_correct)

    tp = sum(g and p for g, p in valid)
    fp = sum((not g) and p for g, p in valid)
    fn = sum(g and (not p) for g, p in valid)
    tn = sum((not g) and (not p) for g, p in valid)
    parsed = len(valid)
    gold_counts = Counter(_gold_label(row) for row in gold_rows)

    return {
        "total_input": total,
        "gold": dict(gold_counts),
        "valid_verdicts": parsed,
        "correct_predictions": correct_predictions,
        "accuracy_on_verdicts": correct_predictions / parsed if parsed else 0.0,
        "coverage": parsed / total if total else 0.0,
        # Every missing/error/parser-null row counts as wrong here.
        "strict_end_to_end_accuracy": correct_predictions / total if total else 0.0,
        "confusion_positive_correct": {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        },
        "status_counts": dict(status_counts),
        "tier_counts": dict(tier_counts),
    }


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def _ensure_append_boundary(path: Path) -> None:
    """Keep a crash-truncated final JSONL fragment separate from new rows."""
    if not path.exists() or path.stat().st_size == 0:
        return
    with path.open("rb") as fh:
        fh.seek(-1, os.SEEK_END)
        ends_with_newline = fh.read(1) == b"\n"
    if not ends_with_newline:
        with path.open("ab") as fh:
            fh.write(b"\n")


def _format_percent(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def _print_metrics(metrics: dict[str, Any]) -> None:
    print("\nEvaluation summary")
    print(f"  total input:              {metrics['total_input']}")
    print(f"  valid verdicts:           {metrics['valid_verdicts']}")
    print(f"  correct predictions:      {metrics['correct_predictions']}")
    print(
        "  accuracy on verdicts:    "
        f"{_format_percent(metrics['accuracy_on_verdicts'])}"
    )
    print(f"  coverage:                 {_format_percent(metrics['coverage'])}")
    print(
        "  strict end-to-end acc:   "
        f"{_format_percent(metrics['strict_end_to_end_accuracy'])}"
    )
    print(f"  status counts:            {metrics['status_counts']}")
    print(
        "  confusion (+ = correct): "
        f"{metrics['confusion_positive_correct']}"
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gold",
        default=str(ROOT / "data" / "benchmark" / "eval_curation_v1.jsonl"),
        help="Human-gold JSONL; one row per Statement/Evidence evaluation item.",
    )
    parser.add_argument(
        "--corpus",
        default=str(
            ROOT / "data" / "benchmark" / "indra_benchmark_corpus.json.gz"
        ),
        help="Full INDRA benchmark corpus used to recover native objects/db_refs.",
    )
    parser.add_argument("--model", default="vllm-gemma-4-26b")
    parser.add_argument(
        "--base-url",
        default=None,
        help="Override the selected registry entry's OpenAI-compatible base URL.",
    )
    parser.add_argument(
        "--served-model-id",
        default=None,
        help="Override the model name sent to the OpenAI-compatible endpoint.",
    )
    parser.add_argument(
        "--variant",
        choices=(
            "baseline",
            "disconfirm",
            "disconfirm_relnature",
            "disconfirm_relnature_rf",
            "disconfirm_relnature_rf_noconf",
            "verdict_only",
        ),
        default="baseline",
        help="baseline is the closest current-code path to historical v12; "
        "disconfirm_relnature_rf is the current production default.",
    )
    parser.add_argument("--require-calibrated", action="store_true",
                        help="refuse the run unless this model+prompt resolves a "
                             "ship-approved calibration profile")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--retry-sleep-s", type=float, default=2.0)
    parser.add_argument(
        "--output",
        default=str(ROOT / "data" / "results" / "vllm_gold_eval.jsonl"),
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume an output file. Valid verdicts are skipped.",
    )
    parser.add_argument(
        "--retry-parser-nulls",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--retry-errors",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--progress-every", type=int, default=20)
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.workers < 1:
        raise SystemExit("--workers must be >= 1")
    if args.retries < 0:
        raise SystemExit("--retries must be >= 0")
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be >= 1")
    if args.progress_every < 1:
        raise SystemExit("--progress-every must be >= 1")


def _iter_pending_jobs(
    gold_rows: list[dict[str, Any]],
    done_indices: set[int],
) -> Iterable[tuple[int, dict[str, Any]]]:
    for row_index, row in enumerate(gold_rows):
        if row_index not in done_indices:
            yield row_index, row


def main() -> int:
    args = _build_parser().parse_args()
    _validate_args(args)
    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)

    # The scorer chooses its prompt variant at import time.
    os.environ["MONO_VARIANT"] = (
        "" if args.variant == "baseline" else args.variant
    )

    from indra_belief.data.corpus import CorpusIndex
    from indra_belief.model_client import (
        LOCAL_MODELS, ModelClient, canonical_model_name,
    )

    # Resolved before the lookup, so a renamed entry stays reachable under its
    # old name and an aliased run is byte-identical to a canonical one -- the
    # calibration profile is keyed on the canonical name.
    args.model = canonical_model_name(args.model)
    from indra_belief.scorers.monolithic import scorer as mono

    if args.model not in LOCAL_MODELS:
        raise SystemExit(
            f"--model {args.model!r} is not a local/OpenAI-compatible registry "
            "entry. Add the vLLM endpoint to LOCAL_MODELS first."
        )
    if args.base_url:
        LOCAL_MODELS[args.model]["base_url"] = args.base_url.rstrip("/")
    if args.served_model_id:
        LOCAL_MODELS[args.model]["model_id"] = args.served_model_id

    gold_path = Path(args.gold).resolve()
    corpus_path = Path(args.corpus).resolve()
    output_path = Path(args.output).resolve()
    meta_path = output_path.with_suffix(".meta.json")
    summary_path = output_path.with_suffix(".summary.json")

    gold_rows = _read_jsonl(gold_path)
    if args.limit is not None:
        gold_rows = gold_rows[: args.limit]
    input_sha256 = _sha256(gold_path)
    # The prompt is per-VARIANT, so the fingerprint must be read from the variant
    # actually selected. `mono.ACTIVE_SYSTEM_PROMPT` was a single module-level
    # global; it no longer exists, and once variants landed it could not have been
    # right anyway — every variant would have recorded the same sha. The registry
    # keyed by `--variant` is the same one the scorer dispatches on.
    prompt_sha256 = hashlib.sha256(
        mono.VARIANTS[args.variant].system_prompt.encode("utf-8")
    ).hexdigest()

    # Say out loud whether this run will be calibrated. An unfitted (model,
    # prompt) pair silently falls back to the hard gate, and at corpus scale
    # that produces ECE 0.237 numbers indistinguishable from ECE 0.045 ones.
    from indra_belief.calibration_constants import calibration_banner

    calibrated, banner = calibration_banner(args.model, prompt_sha256)
    print(banner, flush=True)
    if args.require_calibrated and not calibrated:
        raise SystemExit(
            "refusing to run: --require-calibrated was passed and this "
            "model+prompt pair has no ship-approved profile"
        )

    expected_identity = {
        "gold_path": str(gold_path),
        "gold_sha256": input_sha256,
        "corpus_path": str(corpus_path),
        "model": args.model,
        "base_url": LOCAL_MODELS[args.model].get("base_url"),
        "model_id": LOCAL_MODELS[args.model].get("model_id"),
        "variant": args.variant,
        "prompt_sha256": prompt_sha256,
    }

    if output_path.exists() and not args.resume:
        raise SystemExit(
            f"{output_path} already exists; use --resume or choose a new --output"
        )
    if output_path.exists():
        if not meta_path.exists():
            raise SystemExit(
                f"{output_path} exists without {meta_path}; refusing unsafe resume"
            )
        old_meta = json.loads(meta_path.read_text())
        mismatches = {
            key: (old_meta.get(key), value)
            for key, value in expected_identity.items()
            if old_meta.get(key) != value
        }
        if mismatches:
            raise SystemExit(
                "resume configuration does not match existing run: "
                + json.dumps(mismatches, sort_keys=True)
            )

    latest, corrupt_lines = load_latest_attempts(output_path)
    done_indices = {
        row_index
        for row_index, row in latest.items()
        if row_index < len(gold_rows)
        and _is_done(
            row,
            retry_parser_nulls=args.retry_parser_nulls,
            retry_errors=args.retry_errors,
        )
    }
    attempts_by_index = Counter(
        {
            row_index: int(row.get("attempt") or 1)
            for row_index, row in latest.items()
        }
    )

    meta = {
        **expected_identity,
        "status": "running",
        "started_or_resumed_at": _now(),
        "output": str(output_path),
        "summary": str(summary_path),
        "total_input": len(gold_rows),
        "already_done": len(done_indices),
        "workers": args.workers,
        "max_tokens": args.max_tokens,
        "retries": args.retries,
        "retry_parser_nulls": args.retry_parser_nulls,
        "retry_errors": args.retry_errors,
        "corrupt_output_lines_seen": corrupt_lines,
    }
    _write_json_atomic(meta_path, meta)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _ensure_append_boundary(output_path)

    print(
        f"gold={gold_path.name} rows={len(gold_rows)} already_done={len(done_indices)}"
    )
    print(
        f"model={args.model} model_id={LOCAL_MODELS[args.model].get('model_id')} "
        f"variant={args.variant} workers={args.workers}"
    )
    print(f"prompt_sha256={prompt_sha256}")
    print(f"loading corpus index from {corpus_path} ...", flush=True)
    index = CorpusIndex(corpus_path)
    index.load()

    client = ModelClient(args.model)

    def resolve_job(item: tuple[int, dict[str, Any]]):
        row_index, gold_row = item
        resolved = index.get(
            gold_row["source_hash"],
            gold_row.get("subject", ""),
            gold_row.get("object", ""),
        )
        return row_index, gold_row, resolved

    pending_resolved = [
        resolve_job(item)
        for item in _iter_pending_jobs(gold_rows, done_indices)
    ]

    def score_job(job):
        row_index, gold_row, resolved = job
        attempt_start = time.perf_counter()
        if resolved is None:
            return {
                "row_index": row_index,
                "source_hash": gold_row.get("source_hash"),
                "pa_hash": gold_row.get("pa_hash"),
                "matches_hash": gold_row.get("matches_hash"),
                "stmt_type": gold_row.get("stmt_type"),
                "subject": gold_row.get("subject"),
                "object": gold_row.get("object"),
                "tag": gold_row.get("tag"),
                "gold": _gold_label(gold_row),
                "gold_correct": _gold_label(gold_row) == "correct",
                "row_status": "unmatched",
                "verdict": None,
                "prediction_correct": None,
                "error": "Statement/Evidence pair not found in benchmark corpus",
                "latency_s": 0.0,
                "attempts_this_invocation": 0,
            }

        stmt, evidence = resolved
        last_error: Exception | None = None
        last_result: dict[str, Any] | None = None
        attempts = 0
        for attempt in range(args.retries + 1):
            attempts += 1
            try:
                result = mono.score_statement(
                    stmt,
                    evidence,
                    client,
                    max_tokens=args.max_tokens,
                )
                last_result = result
                if result.get("verdict") in VALID_VERDICTS:
                    break
            except Exception as exc:  # recorded per row; resume can retry it
                last_error = exc
            if attempt < args.retries:
                time.sleep(args.retry_sleep_s)

        elapsed = round(time.perf_counter() - attempt_start, 3)
        base = {
            "row_index": row_index,
            "source_hash": gold_row.get("source_hash"),
            "pa_hash": gold_row.get("pa_hash"),
            "matches_hash": gold_row.get("matches_hash"),
            "stmt_type": gold_row.get("stmt_type"),
            "subject": gold_row.get("subject"),
            "object": gold_row.get("object"),
            "tag": gold_row.get("tag"),
            "gold": _gold_label(gold_row),
            "gold_correct": _gold_label(gold_row) == "correct",
            "latency_s": elapsed,
            "attempts_this_invocation": attempts,
        }
        if last_result is not None:
            verdict = last_result.get("verdict")
            return {
                **base,
                "row_status": (
                    "scored" if verdict in VALID_VERDICTS else "parser_null"
                ),
                "verdict": verdict,
                "prediction_correct": (
                    (verdict == "correct") == base["gold_correct"]
                    if verdict in VALID_VERDICTS
                    else None
                ),
                "score": last_result.get("score"),
                "confidence": last_result.get("confidence"),
                # The RAW in-call margin. This row list is a whitelist, so a
                # field absent here is DROPPED -- and a gold run whose margins
                # were dropped cannot fit the isotonic that is the entire reason
                # to do a gold run on a new serving stack. Persisted even when
                # None, so "the reader could not produce one" is distinguishable
                # from "nobody asked".
                "probe_delta_logit": last_result.get("probe_delta_logit"),
                "tier": last_result.get("tier"),
                "grounding_status": last_result.get("grounding_status"),
                "provenance_triggered": last_result.get("provenance_triggered"),
                "tokens": last_result.get("tokens"),
                # The per-call token/model ledger, for the SAME whitelist reason as
                # probe_delta_logit above. ``scorers/monolithic/scorer.py::_score_categorical`` returns it and this row list
                # dropped it, so every row scored through this runner was UNPRICEABLE:
                # ``scripts/frontier_table.py::_cost_summary`` and ``scripts/cost_report.py::tally``
                # both read `call_log`, and
                # with it absent that same `_cost_summary` sets cost=None, so the model
                # cannot be placed on the cost x err-F1 frontier at all. That is what
                # made the compare loop complete for local/self-hosted readers and
                # incomplete for provider-hosted ones. Persisted even when empty, so
                # "this reader reported no calls" stays distinguishable from "nobody
                # recorded any".
                "call_log": last_result.get("call_log") or [],
                "selected_example_ids": last_result.get(
                    "selected_example_ids", []
                ),
                "raw_text": last_result.get("raw_text") or "",
                "error": None,
            }
        assert last_error is not None
        return {
            **base,
            "row_status": "error",
            "verdict": None,
            "prediction_correct": None,
            "score": None,
            "confidence": None,
            "probe_delta_logit": None,
            "tier": "row_error",
            "grounding_status": None,
            "provenance_triggered": None,
            "tokens": None,
            "selected_example_ids": [],
            "raw_text": "",
            "error": f"{type(last_error).__name__}: {last_error}",
        }

    completed_this_invocation = 0
    started = time.perf_counter()
    jobs = iter(pending_resolved)
    window = max(args.workers, args.workers * 2)

    with output_path.open("a", buffering=1) as out_fh:
        with cf.ThreadPoolExecutor(
            max_workers=args.workers, thread_name_prefix="vllm-gold"
        ) as pool:
            inflight: set[cf.Future] = set()

            def submit_next() -> bool:
                try:
                    job = next(jobs)
                except StopIteration:
                    return False
                inflight.add(pool.submit(score_job, job))
                return True

            for _ in range(window):
                if not submit_next():
                    break

            while inflight:
                finished, _ = cf.wait(
                    inflight, return_when=cf.FIRST_COMPLETED
                )
                for future in finished:
                    inflight.remove(future)
                    row = future.result()
                    row_index = int(row["row_index"])
                    attempts_by_index[row_index] += 1
                    row["attempt"] = attempts_by_index[row_index]
                    row["completed_at"] = _now()
                    out_fh.write(json.dumps(row, default=str) + "\n")
                    out_fh.flush()
                    latest[row_index] = row
                    completed_this_invocation += 1

                    if (
                        completed_this_invocation == 1
                        or completed_this_invocation % args.progress_every == 0
                    ):
                        elapsed = time.perf_counter() - started
                        metrics = compute_metrics(gold_rows, latest)
                        print(
                            f"[{metrics['valid_verdicts']}/{len(gold_rows)} valid] "
                            f"acc={_format_percent(metrics['accuracy_on_verdicts'])} "
                            f"coverage={_format_percent(metrics['coverage'])} "
                            f"rate={completed_this_invocation / max(elapsed, 1e-9):.3f} rows/s",
                            flush=True,
                        )

                    if not STOP_REQUESTED:
                        submit_next()

    metrics = compute_metrics(gold_rows, latest)
    retryable = sum(
        1
        for row_index, row in latest.items()
        if row_index < len(gold_rows)
        and not _is_done(
            row,
            retry_parser_nulls=args.retry_parser_nulls,
            retry_errors=args.retry_errors,
        )
    )
    status = (
        "stopped"
        if STOP_REQUESTED
        else ("completed_with_retryable_failures" if retryable else "completed")
    )
    summary = {
        **expected_identity,
        "status": status,
        "finished_at": _now(),
        "duration_this_invocation_s": round(
            time.perf_counter() - started, 3
        ),
        "completed_this_invocation": completed_this_invocation,
        "retryable_rows": retryable,
        "metrics": metrics,
    }
    _write_json_atomic(summary_path, summary)
    meta.update(summary)
    _write_json_atomic(meta_path, meta)
    _print_metrics(metrics)
    print(f"\nresults: {output_path}")
    print(f"summary: {summary_path}")
    return 130 if STOP_REQUESTED else (2 if retryable else 0)


if __name__ == "__main__":
    raise SystemExit(main())
