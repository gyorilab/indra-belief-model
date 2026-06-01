"""Resumable monolithic scoring run for the rasmachine corpus.

This is intentionally file-backed rather than DuckDB-backed during inference:
each completed (statement index, evidence index) lands as one JSONL row, and
resume skips rows already present in the output file. If the process is
interrupted, rerunning the same command continues from the next missing row.

Example:
  PYTHONPATH=src .venv/bin/python scripts/run_rasmachine_monolithic.py \
    --model gemma-remote \
    --input data/corpora/latest_statements_rasmachine.json \
    --output data/results/rasmachine_mono_gemma_remote_direct.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from indra.statements import stmts_from_json  # noqa: E402

from indra_belief.model_client import ModelClient  # noqa: E402
from indra_belief.scorers.monolithic.scorer import (  # noqa: E402
    score_statement as score_evidence_monolithic,
)


DEFAULT_INPUT = ROOT / "data" / "corpora" / "latest_statements_rasmachine.json"
DEFAULT_OUTPUT = ROOT / "data" / "results" / "rasmachine_mono_gemma_remote_direct.jsonl"

STOP_REQUESTED = False


def _request_stop(signum, _frame) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print(f"received signal {signum}; will stop after current evidence", file=sys.stderr)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _hex64(value: int | str | None) -> str | None:
    if value is None:
        return None
    try:
        return f"{int(value) & ((1 << 64) - 1):016x}"
    except Exception:
        return str(value)


def _evidence_hash(ev) -> str:
    try:
        return _hex64(ev.get_source_hash()) or ""
    except Exception:
        import hashlib

        payload = f"{ev.source_api}|{ev.source_id}|{ev.pmid}|{ev.text}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _stmt_hash(stmt) -> str:
    try:
        return _hex64(stmt.get_hash(shallow=True)) or ""
    except Exception:
        return ""


def _statement_label(stmt) -> tuple[str, str]:
    agents = list(stmt.agent_list() or [])
    names = [a.name for a in agents if a is not None]
    if not names:
        return "?", "?"
    if len(names) == 1:
        return names[0], "?"
    return names[0], names[1]


def _load_done_keys(
    output_path: Path,
    retry_parser_nulls: bool = True,
    retry_row_errors: bool = True,
) -> tuple[set[tuple[int, int]], Counter, int, int, int]:
    done: set[tuple[int, int]] = set()
    verdicts: Counter = Counter()
    row_errors = 0
    retryable_parser_nulls = 0
    retryable_row_errors = 0
    if not output_path.exists():
        return done, verdicts, row_errors, retryable_parser_nulls, retryable_row_errors

    latest_by_key: dict[tuple[int, int], dict[str, Any]] = {}
    with open(output_path) as fh:
        for line in fh:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "stmt_i" not in row or "evidence_i" not in row:
                continue
            key = (int(row["stmt_i"]), int(row["evidence_i"]))
            latest_by_key[key] = row

    for key, row in latest_by_key.items():
        is_row_error = row.get("row_status") == "error"
        is_parser_null = row.get("verdict") is None and not is_row_error
        if retry_row_errors and is_row_error:
            retryable_row_errors += 1
            row_errors += 1
            continue
        if retry_parser_nulls and is_parser_null:
            retryable_parser_nulls += 1
            continue
        done.add(key)
        verdicts[row.get("verdict") or "None"] += 1
        if is_row_error:
            row_errors += 1
    return done, verdicts, row_errors, retryable_parser_nulls, retryable_row_errors


def _load_or_create_run_id(meta_path: Path, requested: str | None) -> str:
    if requested:
        return requested
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
            if meta.get("run_id"):
                return str(meta["run_id"])
        except Exception:
            pass
    return uuid.uuid4().hex


def _write_meta(meta_path: Path, payload: dict[str, Any]) -> None:
    tmp = meta_path.with_suffix(meta_path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(meta_path)


def _progress(fh, event: str, **payload) -> None:
    payload = {"event": event, "ts": _now(), **payload}
    fh.write(json.dumps(payload, default=str) + "\n")
    fh.flush()


def _preserve_call_log(call_log: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Preserve full model telemetry without trimming prompts or outputs."""
    return call_log


def _optional_positive_int(value: str | None) -> int | None:
    if value is None:
        return None
    lowered = str(value).strip().lower()
    if lowered in {"", "none", "null", "default", "unlimited", "uncapped"}:
        return None
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative or 'none'")
    return parsed


def _maybe_truncate(value: str, limit: int | None) -> str:
    if limit is None:
        return value
    return value[:limit]


def _score_with_retries(
    stmt, ev, client, max_tokens: int | None, retries: int, retry_sleep_s: float
) -> dict:
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            return score_evidence_monolithic(stmt, ev, client, max_tokens=max_tokens)
        except Exception as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(retry_sleep_s)
    assert last_error is not None
    raise last_error


def _stmt_meta(stmt) -> dict[str, Any]:
    """Per-statement fields shared by every evidence row of that statement."""
    subj, obj = _statement_label(stmt)
    return {
        "stmt_hash": _stmt_hash(stmt),
        "subject": subj,
        "object": obj,
        "stmt_type": type(stmt).__name__,
        "belief": getattr(stmt, "belief", None),
    }


def _scored_row(run_id, stmt_i, ev_i, ev, sm, result, latency_s, call_summary) -> dict:
    """Build a scored row. MUST stay byte-identical between serial and
    concurrent paths so the resumable JSONL output has one schema."""
    raw_text = result.get("raw_text") or ""
    return {
        "run_id": run_id,
        "row_status": "scored",
        "stmt_i": stmt_i,
        "evidence_i": ev_i,
        "source_hash": ev.get_source_hash(),
        "stmt_hash": sm["stmt_hash"],
        "evidence_hash": _evidence_hash(ev),
        "stmt_type": sm["stmt_type"],
        "subject": sm["subject"],
        "object": sm["object"],
        "source_api": ev.source_api or "",
        "pmid": ev.pmid,
        "belief": sm["belief"],
        "text_len": len(ev.text or ""),
        "verdict": result.get("verdict"),
        "score": result.get("score"),
        "confidence": result.get("confidence"),
        "tier": result.get("tier"),
        "grounding_status": result.get("grounding_status"),
        "provenance_triggered": result.get("provenance_triggered"),
        "latency_s": round(latency_s, 3),
        "tokens": result.get("tokens"),
        "call_log": call_summary,
        "error": result.get("error"),
        "raw_text_preview": raw_text,
    }


def _error_row(run_id, stmt_i, ev_i, ev, sm, error_text, elapsed, call_summary) -> dict:
    """Build an error row. MUST stay byte-identical between serial and
    concurrent paths."""
    return {
        "run_id": run_id,
        "row_status": "error",
        "stmt_i": stmt_i,
        "evidence_i": ev_i,
        "source_hash": ev.get_source_hash(),
        "stmt_hash": sm["stmt_hash"],
        "evidence_hash": _evidence_hash(ev),
        "stmt_type": sm["stmt_type"],
        "subject": sm["subject"],
        "object": sm["object"],
        "source_api": ev.source_api or "",
        "pmid": ev.pmid,
        "belief": sm["belief"],
        "text_len": len(ev.text or ""),
        "verdict": None,
        "score": None,
        "confidence": None,
        "tier": "row_error",
        "grounding_status": None,
        "provenance_triggered": None,
        "latency_s": round(elapsed, 3),
        "tokens": None,
        "call_log": call_summary,
        "error": error_text,
        "raw_text_preview": "",
    }


def _run_concurrent(
    *,
    stmts,
    done_keys,
    client,
    args,
    run_id,
    meta,
    meta_path,
    out_fh,
    prog_fh,
    verdicts,
    total_evidences,
    recorded_row_errors_start,
) -> int:
    """Concurrent scoring path (args.workers > 1).

    Design: only the LLM call runs on worker threads; ALL file I/O,
    done_keys/verdict bookkeeping, error-policy decisions, and progress
    emission stay single-threaded on this (main) thread. Rows are built
    with the same `_scored_row`/`_error_row` helpers the serial path uses,
    so the JSONL schema is identical and resume stays compatible.

    Worker telemetry is captured inside the worker via `pop_call_log()`
    because ModelClient's call log is thread-local — popping it on the
    main thread would read an empty log. A bounded in-flight window
    (workers * 4) caps memory and lets us stop/drain cleanly on signal.
    """
    import concurrent.futures as cf

    def pending_items():
        for stmt_i, stmt in enumerate(stmts):
            evidences = list(getattr(stmt, "evidence", None) or [])
            if not evidences:
                continue
            sm = _stmt_meta(stmt)  # one hash per statement, not per evidence
            for ev_i, ev in enumerate(evidences):
                if (stmt_i, ev_i) in done_keys:
                    continue
                yield (stmt_i, ev_i, ev, sm)

    def work(item):
        stmt_i, ev_i, ev, sm = item
        # Resolve the live statement/evidence objects on the worker.
        stmt = stmts[stmt_i]
        t_row = time.time()
        try:
            result = _score_with_retries(
                stmt, ev, client, args.max_tokens, args.retries, args.retry_sleep_s
            )
            latency_s = time.time() - t_row
            # Scorer returns the drained call log under result["call_log"];
            # popping the client again would read empty (see serial path).
            call_summary = _preserve_call_log(result.get("call_log") or [])
            return ("ok", stmt_i, ev_i, ev, sm, result, latency_s, call_summary)
        except Exception as exc:  # noqa: BLE001 — recorded per row-error policy
            elapsed = time.time() - t_row
            # On exception the scorer never returned a dict; the partial log
            # (if any) is still on the client's thread-local — pop it here.
            call_summary = _preserve_call_log(client.pop_call_log())
            return ("err", stmt_i, ev_i, ev, sm, exc, elapsed, call_summary)

    recorded_row_errors = recorded_row_errors_start
    completed_this_invocation = 0
    t0 = time.time()
    items = pending_items()
    window = max(1, args.workers) * 4

    with cf.ThreadPoolExecutor(
        max_workers=args.workers, thread_name_prefix="score"
    ) as ex:
        inflight: set = set()

        def submit_next() -> bool:
            try:
                item = next(items)
            except StopIteration:
                return False
            inflight.add(ex.submit(work, item))
            return True

        for _ in range(window):
            if not submit_next():
                break

        stopping = False
        while inflight:
            completed, _ = cf.wait(inflight, return_when=cf.FIRST_COMPLETED)
            for fut in completed:
                inflight.discard(fut)
                kind, stmt_i, ev_i, ev, sm, payload, dur, call_summary = fut.result()
                key = (stmt_i, ev_i)

                if kind == "ok":
                    row = _scored_row(
                        run_id, stmt_i, ev_i, ev, sm, payload, dur, call_summary
                    )
                else:
                    exc = payload
                    error_text = f"{type(exc).__name__}: {exc}"
                    cap_remaining = (
                        args.max_recorded_errors is None
                        or recorded_row_errors < args.max_recorded_errors
                    )
                    if args.row_error_policy == "record" and cap_remaining:
                        recorded_row_errors += 1
                        row = _error_row(
                            run_id, stmt_i, ev_i, ev, sm, error_text, dur, call_summary
                        )
                        _progress(
                            prog_fh,
                            "row_error_recorded",
                            run_id=run_id,
                            stmt_i=stmt_i,
                            evidence_i=ev_i,
                            stmt_hash=sm["stmt_hash"],
                            evidence_hash=_evidence_hash(ev),
                            elapsed_s=round(dur, 3),
                            recorded_row_errors=recorded_row_errors,
                            error=_maybe_truncate(error_text, args.error_preview_chars),
                        )
                    else:
                        limit_reason = (
                            "recorded row error limit reached"
                            if args.row_error_policy == "record"
                            else "row error policy is fail"
                        )
                        if args.row_error_policy == "record":
                            error_text = f"{error_text} ({limit_reason})"
                        event_name = (
                            "fatal_row_error_limit"
                            if args.row_error_policy == "record"
                            else "fatal_row_error"
                        )
                        _progress(
                            prog_fh,
                            event_name,
                            run_id=run_id,
                            stmt_i=stmt_i,
                            evidence_i=ev_i,
                            stmt_hash=sm["stmt_hash"],
                            evidence_hash=_evidence_hash(ev),
                            elapsed_s=round(dur, 3),
                            recorded_row_errors=recorded_row_errors,
                            error=_maybe_truncate(error_text, args.error_preview_chars),
                        )
                        meta.update({
                            "status": "failed",
                            "failed_at": _now(),
                            "failed_key": {"stmt_i": stmt_i, "evidence_i": ev_i},
                            "error": error_text,
                            "completed_this_invocation": completed_this_invocation,
                            "recorded_row_errors": recorded_row_errors,
                        })
                        _write_meta(meta_path, meta)
                        # Abandon in-flight futures; their keys were never
                        # written, so resume re-scores them next invocation.
                        for f in inflight:
                            f.cancel()
                        return 2

                out_fh.write(json.dumps(row, default=str) + "\n")
                out_fh.flush()
                done_keys.add(key)
                completed_this_invocation += 1
                verdicts[row.get("verdict") or "None"] += 1

                if (
                    completed_this_invocation == 1
                    or completed_this_invocation % args.progress_every == 0
                ):
                    elapsed = time.time() - t0
                    done_total = len(done_keys)
                    rate = completed_this_invocation / elapsed if elapsed > 0 else 0.0
                    remaining = total_evidences - done_total
                    eta_s = remaining / rate if rate > 0 else None
                    _progress(
                        prog_fh,
                        "progress",
                        run_id=run_id,
                        done_total=done_total,
                        total_evidences=total_evidences,
                        completed_this_invocation=completed_this_invocation,
                        rate_ev_per_s=round(rate, 5),
                        eta_s=round(eta_s, 1) if eta_s is not None else None,
                        latest={"stmt_i": stmt_i, "evidence_i": ev_i},
                        verdicts=dict(verdicts),
                        recorded_row_errors=recorded_row_errors,
                        workers=args.workers,
                    )

                # Refill the window unless we're stopping or drained.
                if not STOP_REQUESTED and not stopping:
                    submit_next()

            if STOP_REQUESTED and not stopping:
                stopping = True  # stop submitting; let in-flight drain

        if STOP_REQUESTED:
            meta.update({
                "status": "stopped",
                "stopped_at": _now(),
                "completed_this_invocation": completed_this_invocation,
                "recorded_row_errors": recorded_row_errors,
            })
            _write_meta(meta_path, meta)
            _progress(
                prog_fh,
                "stopped",
                run_id=run_id,
                completed_this_invocation=completed_this_invocation,
            )
            return 130

    elapsed = time.time() - t0
    meta.update({
        "status": "completed",
        "completed_at": _now(),
        "completed_total": len(done_keys),
        "completed_this_invocation": completed_this_invocation,
        "duration_this_invocation_s": round(elapsed, 3),
        "verdicts": dict(verdicts),
        "recorded_row_errors": recorded_row_errors,
    })
    _write_meta(meta_path, meta)
    _progress(
        prog_fh,
        "done",
        run_id=run_id,
        completed_total=len(done_keys),
        total_evidences=total_evidences,
        duration_this_invocation_s=round(elapsed, 3),
        verdicts=dict(verdicts),
        recorded_row_errors=recorded_row_errors,
    )
    return 0


def main() -> int:
    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)

    parser = argparse.ArgumentParser(description="Run monolithic rasmachine scoring with JSONL resume.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--model", default="gemma-remote")
    parser.add_argument(
        "--max-tokens",
        type=_optional_positive_int,
        default=None,
        help="Per-generation token limit. Omit or pass 'none' to use the model backend ceiling.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", action="store_false", dest="resume")
    parser.add_argument("--retry-parser-nulls", action="store_true", default=True)
    parser.add_argument("--keep-parser-nulls", action="store_false", dest="retry_parser_nulls")
    parser.add_argument("--retry-row-errors", action="store_true", default=True)
    parser.add_argument("--keep-row-errors", action="store_false", dest="retry_row_errors")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--retry-sleep-s", type=float, default=10.0)
    parser.add_argument("--row-error-policy", choices=("fail", "record"), default="fail")
    parser.add_argument(
        "--max-recorded-errors",
        type=_optional_positive_int,
        default=None,
        help="Optional cap on recorded row errors before aborting. Default: unlimited "
        "(pass an integer to opt in; 'none' is equivalent to omitting).",
    )
    parser.add_argument("--error-preview-chars", type=_optional_positive_int, default=None)
    parser.add_argument("--progress-every", type=int, default=5)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Concurrent scoring workers. 1 = serial (default, battle-tested). "
        ">1 dispatches LLM calls through a thread pool while keeping all file "
        "I/O and bookkeeping single-threaded on the main thread. Match the "
        "serving backend's slot count (the llm-gateway llama container runs "
        "-np 4, so --workers 4 saturates it).",
    )
    # NOTE: raw_text (the model's full reasoning) is recorded UNCAPPED. The former
    # --raw-preview-chars flag was removed — clipping it silently discarded generated
    # output and is no longer permitted.
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    progress_path = output_path.with_suffix(".progress.ndjson")
    meta_path = output_path.with_suffix(".meta.json")
    run_id = _load_or_create_run_id(meta_path, args.run_id)

    if args.resume:
        (
            done_keys,
            existing_verdicts,
            existing_row_errors,
            retryable_parser_nulls,
            retryable_row_errors,
        ) = _load_done_keys(
            output_path,
            retry_parser_nulls=args.retry_parser_nulls,
            retry_row_errors=args.retry_row_errors,
        )
    else:
        (
            done_keys,
            existing_verdicts,
            existing_row_errors,
            retryable_parser_nulls,
            retryable_row_errors,
        ) = (
            set(),
            Counter(),
            0,
            0,
            0,
        )

    with open(input_path) as fh:
        stmt_json = json.load(fh)
    stmts = stmts_from_json(stmt_json)
    if args.limit is not None:
        stmts = stmts[: args.limit]

    total_evidences = sum(len(getattr(stmt, "evidence", None) or []) for stmt in stmts)
    pending_total = sum(
        1
        for stmt_i, stmt in enumerate(stmts)
        for ev_i, _ev in enumerate(getattr(stmt, "evidence", None) or [])
        if (stmt_i, ev_i) not in done_keys
    )

    meta = {
        "run_id": run_id,
        "status": "running",
        "model": args.model,
        "max_tokens": args.max_tokens,
        "input": str(input_path),
        "output": str(output_path),
        "progress": str(progress_path),
        "started_or_resumed_at": _now(),
        "total_statements": len(stmts),
        "total_evidences": total_evidences,
        "already_completed": len(done_keys),
        "pending_at_start": pending_total,
        "row_error_policy": args.row_error_policy,
        "max_recorded_errors": args.max_recorded_errors,
        "existing_row_errors": existing_row_errors,
        "retry_parser_nulls": args.retry_parser_nulls,
        "retryable_parser_nulls": retryable_parser_nulls,
        "retry_row_errors": args.retry_row_errors,
        "retryable_row_errors": retryable_row_errors,
        "pid": os.getpid(),
    }
    _write_meta(meta_path, meta)

    print(
        f"run_id={run_id} model={args.model} total_ev={total_evidences} "
        f"already_done={len(done_keys)} pending={pending_total}",
        file=sys.stderr,
        flush=True,
    )

    client = ModelClient(args.model)
    verdicts = Counter(existing_verdicts)
    recorded_row_errors = existing_row_errors
    completed_this_invocation = 0
    t0 = time.time()

    with open(output_path, "a", buffering=1) as out_fh, open(progress_path, "a", buffering=1) as prog_fh:
        _progress(
            prog_fh,
            "started_or_resumed",
            run_id=run_id,
            model=args.model,
            total_evidences=total_evidences,
            already_completed=len(done_keys),
            pending=pending_total,
            retryable_parser_nulls=retryable_parser_nulls,
            retryable_row_errors=retryable_row_errors,
            pid=os.getpid(),
        )

        if args.workers > 1:
            return _run_concurrent(
                stmts=stmts,
                done_keys=done_keys,
                client=client,
                args=args,
                run_id=run_id,
                meta=meta,
                meta_path=meta_path,
                out_fh=out_fh,
                prog_fh=prog_fh,
                verdicts=verdicts,
                total_evidences=total_evidences,
                recorded_row_errors_start=recorded_row_errors,
            )

        for stmt_i, stmt in enumerate(stmts):
            evidences = list(getattr(stmt, "evidence", None) or [])
            if not evidences:
                continue

            stmt_hash = _stmt_hash(stmt)
            subj, obj = _statement_label(stmt)
            stmt_type = type(stmt).__name__
            belief = getattr(stmt, "belief", None)

            for ev_i, ev in enumerate(evidences):
                if STOP_REQUESTED:
                    meta.update({
                        "status": "stopped",
                        "stopped_at": _now(),
                        "completed_this_invocation": completed_this_invocation,
                        "recorded_row_errors": recorded_row_errors,
                    })
                    _write_meta(meta_path, meta)
                    _progress(
                        prog_fh,
                        "stopped",
                        run_id=run_id,
                        completed_this_invocation=completed_this_invocation,
                    )
                    return 130

                key = (stmt_i, ev_i)
                if key in done_keys:
                    continue

                ev_hash = _evidence_hash(ev)
                text = ev.text or ""
                t_row = time.time()
                try:
                    result = _score_with_retries(
                        stmt, ev, client, args.max_tokens, args.retries, args.retry_sleep_s
                    )
                    latency_s = time.time() - t_row
                    # The monolithic scorer drains the thread-local call log
                    # internally and returns it under result["call_log"]; a
                    # separate client.pop_call_log() here would read empty.
                    call_summary = _preserve_call_log(result.get("call_log") or [])
                    raw_text = result.get("raw_text") or ""
                    row = {
                        "run_id": run_id,
                        "row_status": "scored",
                        "stmt_i": stmt_i,
                        "evidence_i": ev_i,
                        "source_hash": ev.get_source_hash(),
                        "stmt_hash": stmt_hash,
                        "evidence_hash": ev_hash,
                        "stmt_type": stmt_type,
                        "subject": subj,
                        "object": obj,
                        "source_api": ev.source_api or "",
                        "pmid": ev.pmid,
                        "belief": belief,
                        "text_len": len(text),
                        "verdict": result.get("verdict"),
                        "score": result.get("score"),
                        "confidence": result.get("confidence"),
                        "tier": result.get("tier"),
                        "grounding_status": result.get("grounding_status"),
                        "provenance_triggered": result.get("provenance_triggered"),
                        "latency_s": round(latency_s, 3),
                        "tokens": result.get("tokens"),
                        "call_log": call_summary,
                        "error": result.get("error"),
                        # Full model output — no cap. Field name kept for downstream compat.
                        "raw_text_preview": raw_text,
                    }
                except Exception as exc:
                    elapsed = time.time() - t_row
                    error_text = f"{type(exc).__name__}: {exc}"
                    cap_remaining = (
                        args.max_recorded_errors is None
                        or recorded_row_errors < args.max_recorded_errors
                    )
                    if args.row_error_policy == "record" and cap_remaining:
                        recorded_row_errors += 1
                        call_summary = _preserve_call_log(client.pop_call_log())
                        row = {
                            "run_id": run_id,
                            "row_status": "error",
                            "stmt_i": stmt_i,
                            "evidence_i": ev_i,
                            "source_hash": ev.get_source_hash(),
                            "stmt_hash": stmt_hash,
                            "evidence_hash": ev_hash,
                            "stmt_type": stmt_type,
                            "subject": subj,
                            "object": obj,
                            "source_api": ev.source_api or "",
                            "pmid": ev.pmid,
                            "belief": belief,
                            "text_len": len(text),
                            "verdict": None,
                            "score": None,
                            "confidence": None,
                            "tier": "row_error",
                            "grounding_status": None,
                            "provenance_triggered": None,
                            "latency_s": round(elapsed, 3),
                            "tokens": None,
                            "call_log": call_summary,
                            "error": error_text,
                            "raw_text_preview": "",
                        }
                        _progress(
                            prog_fh,
                            "row_error_recorded",
                            run_id=run_id,
                            stmt_i=stmt_i,
                            evidence_i=ev_i,
                            stmt_hash=stmt_hash,
                            evidence_hash=ev_hash,
                            elapsed_s=round(elapsed, 3),
                            recorded_row_errors=recorded_row_errors,
                            error=_maybe_truncate(error_text, args.error_preview_chars),
                        )
                    else:
                        limit_reason = (
                            "recorded row error limit reached"
                            if args.row_error_policy == "record"
                            else "row error policy is fail"
                        )
                        if args.row_error_policy == "record":
                            error_text = f"{error_text} ({limit_reason})"
                        client.pop_call_log()
                        event_name = (
                            "fatal_row_error_limit"
                            if args.row_error_policy == "record"
                            else "fatal_row_error"
                        )
                        _progress(
                            prog_fh,
                            event_name,
                            run_id=run_id,
                            stmt_i=stmt_i,
                            evidence_i=ev_i,
                            stmt_hash=stmt_hash,
                            evidence_hash=ev_hash,
                            elapsed_s=round(elapsed, 3),
                            recorded_row_errors=recorded_row_errors,
                            error=_maybe_truncate(error_text, args.error_preview_chars),
                        )
                        meta.update({
                            "status": "failed",
                            "failed_at": _now(),
                            "failed_key": {"stmt_i": stmt_i, "evidence_i": ev_i},
                            "error": error_text,
                            "completed_this_invocation": completed_this_invocation,
                            "recorded_row_errors": recorded_row_errors,
                        })
                        _write_meta(meta_path, meta)
                        return 2

                out_fh.write(json.dumps(row, default=str) + "\n")
                out_fh.flush()
                done_keys.add(key)
                completed_this_invocation += 1
                verdicts[row.get("verdict") or "None"] += 1

                if (
                    completed_this_invocation == 1
                    or completed_this_invocation % args.progress_every == 0
                ):
                    elapsed = time.time() - t0
                    done_total = len(done_keys)
                    rate = completed_this_invocation / elapsed if elapsed > 0 else 0.0
                    remaining = total_evidences - done_total
                    eta_s = remaining / rate if rate > 0 else None
                    _progress(
                        prog_fh,
                        "progress",
                        run_id=run_id,
                        done_total=done_total,
                        total_evidences=total_evidences,
                        completed_this_invocation=completed_this_invocation,
                        rate_ev_per_s=round(rate, 5),
                        eta_s=round(eta_s, 1) if eta_s is not None else None,
                        latest={"stmt_i": stmt_i, "evidence_i": ev_i},
                        verdicts=dict(verdicts),
                        recorded_row_errors=recorded_row_errors,
                    )

        elapsed = time.time() - t0
        meta.update({
            "status": "completed",
            "completed_at": _now(),
            "completed_total": len(done_keys),
            "completed_this_invocation": completed_this_invocation,
            "duration_this_invocation_s": round(elapsed, 3),
            "verdicts": dict(verdicts),
            "recorded_row_errors": recorded_row_errors,
        })
        _write_meta(meta_path, meta)
        _progress(
            prog_fh,
            "done",
            run_id=run_id,
            completed_total=len(done_keys),
            total_evidences=total_evidences,
            duration_this_invocation_s=round(elapsed, 3),
            verdicts=dict(verdicts),
            recorded_row_errors=recorded_row_errors,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
