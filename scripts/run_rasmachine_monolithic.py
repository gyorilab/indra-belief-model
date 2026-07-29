#!/usr/bin/env python3
"""Score an INDRA statement corpus with an unmetered local/self-hosted model.

Results are appended one evidence row at a time.  Re-running the same command
skips completed keys and retries prior parser abstentions or recorded row
errors by default.  Metered provider comparison runs intentionally do not enter
through this utility; ``python -m indra_belief.comparison run`` is their sole
entry point and owns spend accounting.
"""

from __future__ import annotations

import argparse
from collections import Counter
import concurrent.futures as futures
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import tempfile
import time
import uuid
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from indra.statements import stmts_from_json  # noqa: E402

from indra_belief.model_client import (  # noqa: E402
    LOCAL_MODELS,
    ModelClient,
    canonical_model_name,
)
from indra_belief.scorers.monolithic import scorer as monolithic_scorer  # noqa: E402


DEFAULT_INPUT = ROOT / "data/corpora/latest_statements_rasmachine.json"
DEFAULT_OUTPUT = ROOT / "data/results/rasmachine_mono_gemma_remote_direct.jsonl"
STOP_REQUESTED = False
_ARCH = "monolithic"


def _request_stop(signum: int, _frame: Any) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print(
        f"received signal {signum}; draining active rows before stopping",
        file=sys.stderr,
    )


def _now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def _score_one(stmt: Any, evidence: Any, client: ModelClient, max_tokens: int | None):
    if _ARCH == "panel":
        from indra_belief.scorers.panel import score_via_panel

        return score_via_panel(stmt, evidence, client)
    if _ARCH == "decomposed":
        from indra_belief.scorers.probes.orchestrator import score_via_probes

        return score_via_probes(stmt, evidence, client)
    return monolithic_scorer.score_statement(
        stmt, evidence, client, max_tokens=max_tokens
    )


def _optional_positive_int(value: str | None) -> int | None:
    if value is None or value.casefold() in {"none", "null"}:
        return None
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive or 'none'")
    return parsed


def _hex64(value: Any) -> str:
    try:
        return f"{int(value) & ((1 << 64) - 1):016x}"
    except Exception:
        return str(value or "")


def _evidence_hash(evidence: Any) -> str:
    try:
        return _hex64(evidence.get_source_hash())
    except Exception:
        value = "|".join(
            str(item or "")
            for item in (
                getattr(evidence, "source_api", None),
                getattr(evidence, "source_id", None),
                getattr(evidence, "pmid", None),
                getattr(evidence, "text", None),
            )
        )
        return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def _evidence_json_sha256(evidence: Any) -> str:
    raw = json.dumps(
        evidence.to_json(),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _statement_hash(stmt: Any) -> str:
    try:
        return _hex64(stmt.get_hash(shallow=True))
    except Exception:
        return ""


def _statement_label(stmt: Any) -> tuple[str, str]:
    names = [
        str(agent.name)
        for agent in (stmt.agent_list() or [])
        if agent is not None and getattr(agent, "name", None)
    ]
    if not names:
        return "?", "?"
    return names[0], names[1] if len(names) > 1 else "?"


def _statement_metadata(stmt: Any) -> dict[str, Any]:
    subject, object_ = _statement_label(stmt)
    return {
        "stmt_hash": _statement_hash(stmt),
        "paper_statement_hash": str(stmt.get_hash(shallow=True)),
        "subject": subject,
        "object": object_,
        "stmt_type": type(stmt).__name__,
        "belief": getattr(stmt, "belief", None),
    }


def _load_done_keys(
    output_path: Path,
    retry_parser_nulls: bool = True,
    retry_row_errors: bool = True,
) -> tuple[set[tuple[int, int]], Counter, int, int, int]:
    """Return terminal keys and resume counters from the latest row per key."""

    if not output_path.exists():
        return set(), Counter(), 0, 0, 0
    latest: dict[tuple[int, int], dict[str, Any]] = {}
    with output_path.open(encoding="utf-8", errors="strict") as stream:
        for line in stream:
            try:
                row = json.loads(line)
                key = (int(row["stmt_i"]), int(row["evidence_i"]))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
            latest[key] = row
    done: set[tuple[int, int]] = set()
    verdicts: Counter = Counter()
    row_errors = parser_nulls = retryable_errors = 0
    for key, row in latest.items():
        is_error = row.get("row_status") == "error"
        is_parser_null = row.get("verdict") is None and not is_error
        row_errors += int(is_error)
        if is_error and retry_row_errors:
            retryable_errors += 1
            continue
        if is_parser_null and retry_parser_nulls:
            parser_nulls += 1
            continue
        done.add(key)
        verdicts[row.get("verdict") or "None"] += 1
    return done, verdicts, row_errors, parser_nulls, retryable_errors


def _load_or_create_run_id(meta_path: Path, requested: str | None) -> str:
    if requested:
        return requested
    if meta_path.exists():
        try:
            value = json.loads(meta_path.read_text(encoding="utf-8")).get("run_id")
            if isinstance(value, str) and value:
                return value
        except (OSError, json.JSONDecodeError):
            pass
    return uuid.uuid4().hex


def _write_meta(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, sort_keys=True, indent=2, default=str)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _progress(stream: Any, event: str, **value: Any) -> None:
    stream.write(
        json.dumps(
            {"event": event, "recorded_at": _now(), **value},
            sort_keys=True,
            default=str,
        )
        + "\n"
    )
    stream.flush()


def _prewarm_grounding(stmts: Iterable[Any]) -> bool:
    """Initialize the shared grounder and ontology before worker fan-out."""

    from indra_belief.data.entity import GroundedEntity

    for stmt in stmts:
        for agent in stmt.agent_list() or []:
            name = getattr(agent, "name", None)
            if not name or name == "?":
                continue
            GroundedEntity.resolve(name)
            try:
                from indra.ontology.bio import bio_ontology

                bio_ontology.get_children("FPLX", "ERK")
            except Exception:
                pass
            return True
    return False


def _base_row(
    run_id: str,
    stmt_i: int,
    evidence_i: int,
    evidence: Any,
    statement: dict[str, Any],
    latency_s: float,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "stmt_i": stmt_i,
        "evidence_i": evidence_i,
        "source_hash": evidence.get_source_hash(),
        "stmt_hash": statement["stmt_hash"],
        "paper_statement_hash": statement["paper_statement_hash"],
        "evidence_hash": _evidence_hash(evidence),
        "evidence_json_sha256": _evidence_json_sha256(evidence),
        "stmt_type": statement["stmt_type"],
        "subject": statement["subject"],
        "object": statement["object"],
        "source_api": evidence.source_api or "",
        "pmid": evidence.pmid,
        "belief": statement["belief"],
        "text_len": len(evidence.text or ""),
        "latency_s": round(latency_s, 3),
    }


def _scored_row(
    run_id: str,
    stmt_i: int,
    evidence_i: int,
    evidence: Any,
    statement: dict[str, Any],
    result: dict[str, Any],
    latency_s: float,
) -> dict[str, Any]:
    return {
        **_base_row(run_id, stmt_i, evidence_i, evidence, statement, latency_s),
        "row_status": "scored",
        "verdict": result.get("verdict"),
        "score": result.get("score"),
        "confidence": result.get("confidence"),
        "tier": result.get("tier"),
        "grounding_status": result.get("grounding_status"),
        "provenance_triggered": result.get("provenance_triggered"),
        "tokens": result.get("tokens"),
        "call_log": list(result.get("call_log") or []),
        "error": result.get("error"),
        "raw_text_preview": str(result.get("raw_text") or ""),
    }


def _error_row(
    run_id: str,
    stmt_i: int,
    evidence_i: int,
    evidence: Any,
    statement: dict[str, Any],
    error: BaseException,
    latency_s: float,
    call_log: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        **_base_row(run_id, stmt_i, evidence_i, evidence, statement, latency_s),
        "row_status": "error",
        "verdict": None,
        "score": None,
        "confidence": None,
        "tier": "row_error",
        "grounding_status": None,
        "provenance_triggered": None,
        "tokens": None,
        "call_log": call_log,
        "error": f"{type(error).__name__}: {error}",
        "raw_text_preview": "",
    }


def _score_with_retries(
    stmt: Any,
    evidence: Any,
    client: ModelClient,
    *,
    max_tokens: int | None,
    retries: int,
    retry_sleep_s: float,
) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            return _score_one(stmt, evidence, client, max_tokens)
        except Exception as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(retry_sleep_s)
    assert last_error is not None
    raise last_error


def _unmetered_client(model: str) -> ModelClient:
    canonical = canonical_model_name(model)
    config = LOCAL_MODELS.get(canonical)
    if canonical.startswith("claude-") or config is None:
        raise ValueError(
            "this utility accepts configured local/self-hosted models only; "
            "use the comparison CLI for provider-backed runs"
        )
    base_url = str(config.get("base_url") or "")
    if config.get("api_key_env") or base_url.startswith("https://") or str(
        config.get("backend") or ""
    ).startswith("bedrock_"):
        raise ValueError(
            "provider-backed models must run through python -m "
            "indra_belief.comparison run"
        )
    return ModelClient(canonical)


def _items(stmts: list[Any], done: set[tuple[int, int]]):
    for stmt_i, stmt in enumerate(stmts):
        metadata = _statement_metadata(stmt)
        for evidence_i, evidence in enumerate(stmt.evidence or []):
            if (stmt_i, evidence_i) not in done:
                yield stmt_i, evidence_i, stmt, evidence, metadata


def _append_row(stream: Any, row: dict[str, Any]) -> None:
    stream.write(json.dumps(row, sort_keys=True, ensure_ascii=True, default=str) + "\n")
    stream.flush()


def _ensure_append_boundary(path: Path) -> None:
    if not path.exists() or path.stat().st_size == 0:
        return
    with path.open("rb+") as stream:
        stream.seek(-1, os.SEEK_END)
        if stream.read(1) != b"\n":
            stream.seek(0, os.SEEK_END)
            stream.write(b"\n")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Resumable local/self-hosted scoring over an INDRA corpus."
    )
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output", "--out", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--model", default="remote-gemma-4-26b")
    parser.add_argument(
        "--arch", choices=("monolithic", "decomposed", "panel"), default="monolithic"
    )
    parser.add_argument("--max-tokens", type=_optional_positive_int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--retry-parser-nulls", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--retry-row-errors", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--run-id")
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--retry-sleep-s", type=float, default=10.0)
    parser.add_argument("--row-error-policy", choices=("fail", "record"), default="fail")
    parser.add_argument("--max-recorded-errors", type=_optional_positive_int)
    parser.add_argument("--error-preview-chars", type=_optional_positive_int)
    parser.add_argument("--progress-every", type=int, default=5)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--export", action=argparse.BooleanOptionalAction, default=True)
    return parser


def _run(args: argparse.Namespace) -> int:
    global _ARCH
    _ARCH = args.arch
    if args.retries < 0 or args.retry_sleep_s < 0:
        raise ValueError("retry values must be nonnegative")
    if args.workers < 1 or args.progress_every < 1:
        raise ValueError("workers and progress-every must be positive")
    if args.limit is not None and args.limit < 0:
        raise ValueError("limit must be nonnegative")

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path = output_path.with_suffix(".meta.json")
    progress_path = output_path.with_suffix(".progress.ndjson")
    run_id = _load_or_create_run_id(meta_path, args.run_id) if args.resume else (
        args.run_id or uuid.uuid4().hex
    )
    if args.resume:
        done, verdicts, existing_errors, parser_nulls, retryable_errors = _load_done_keys(
            output_path,
            retry_parser_nulls=args.retry_parser_nulls,
            retry_row_errors=args.retry_row_errors,
        )
        _ensure_append_boundary(output_path)
        output_mode = "a"
    else:
        done, verdicts, existing_errors, parser_nulls, retryable_errors = (
            set(),
            Counter(),
            0,
            0,
            0,
        )
        output_mode = "w"

    statement_json = json.loads(input_path.read_text(encoding="utf-8"))
    stmts = list(stmts_from_json(statement_json))
    if args.limit is not None:
        stmts = stmts[: args.limit]
    total = sum(len(stmt.evidence or []) for stmt in stmts)
    pending = total - len(done)
    if args.workers > 1:
        _prewarm_grounding(stmts)
    client = _unmetered_client(args.model)
    meta: dict[str, Any] = {
        "run_id": run_id,
        "status": "running",
        "started_at": _now(),
        "model": canonical_model_name(args.model),
        "architecture": args.arch,
        "input": str(input_path),
        "output": str(output_path),
        "total_evidences": total,
        "completed_before_start": len(done),
        "pending_at_start": pending,
        "retryable_parser_nulls": parser_nulls,
        "retryable_row_errors": retryable_errors,
        "workers": args.workers,
    }
    _write_meta(meta_path, meta)

    def work(item):
        stmt_i, evidence_i, stmt, evidence, statement = item
        started = time.monotonic()
        try:
            result = _score_with_retries(
                stmt,
                evidence,
                client,
                max_tokens=args.max_tokens,
                retries=args.retries,
                retry_sleep_s=args.retry_sleep_s,
            )
            return (
                stmt_i,
                evidence_i,
                _scored_row(
                    run_id,
                    stmt_i,
                    evidence_i,
                    evidence,
                    statement,
                    result,
                    time.monotonic() - started,
                ),
                None,
            )
        except Exception as exc:
            call_log = list(getattr(client, "pop_call_log", lambda: [])() or [])
            return (
                stmt_i,
                evidence_i,
                _error_row(
                    run_id,
                    stmt_i,
                    evidence_i,
                    evidence,
                    statement,
                    exc,
                    time.monotonic() - started,
                    call_log,
                ),
                exc,
            )

    completed = 0
    recorded_errors = existing_errors
    fatal: tuple[int, int, BaseException] | None = None
    started = time.monotonic()
    with output_path.open(output_mode, encoding="utf-8") as output, progress_path.open(
        "a", encoding="utf-8"
    ) as progress:
        _progress(progress, "started", **meta)
        iterator = iter(_items(stmts, done))
        executor = futures.ThreadPoolExecutor(
            max_workers=args.workers, thread_name_prefix="rasmachine"
        )
        active: set[futures.Future] = set()
        exhausted = False

        def fill() -> None:
            nonlocal exhausted
            while not exhausted and not STOP_REQUESTED and fatal is None and len(active) < args.workers * 2:
                try:
                    active.add(executor.submit(work, next(iterator)))
                except StopIteration:
                    exhausted = True

        fill()
        try:
            while active:
                ready, active = futures.wait(active, return_when=futures.FIRST_COMPLETED)
                for future in ready:
                    stmt_i, evidence_i, row, error = future.result()
                    if error is not None:
                        can_record = args.row_error_policy == "record" and (
                            args.max_recorded_errors is None
                            or recorded_errors < args.max_recorded_errors
                        )
                        if not can_record:
                            fatal = (stmt_i, evidence_i, error)
                            continue
                        recorded_errors += 1
                    _append_row(output, row)
                    done.add((stmt_i, evidence_i))
                    completed += 1
                    verdicts[row.get("verdict") or "None"] += 1
                    if completed == 1 or completed % args.progress_every == 0:
                        elapsed = time.monotonic() - started
                        rate = completed / elapsed if elapsed else 0.0
                        _progress(
                            progress,
                            "progress",
                            run_id=run_id,
                            done_total=len(done),
                            total_evidences=total,
                            completed_this_invocation=completed,
                            rate_ev_per_s=round(rate, 5),
                            eta_s=round((total - len(done)) / rate, 1) if rate else None,
                            latest={"stmt_i": stmt_i, "evidence_i": evidence_i},
                            verdicts=dict(verdicts),
                            recorded_row_errors=recorded_errors,
                        )
                fill()
        finally:
            executor.shutdown(wait=True, cancel_futures=True)

        elapsed = time.monotonic() - started
        if fatal is not None:
            stmt_i, evidence_i, error = fatal
            preview = f"{type(error).__name__}: {error}"
            if args.error_preview_chars is not None:
                preview = preview[: args.error_preview_chars]
            meta.update(
                status="failed",
                failed_at=_now(),
                failed_key={"stmt_i": stmt_i, "evidence_i": evidence_i},
                error=preview,
                completed_this_invocation=completed,
                recorded_row_errors=recorded_errors,
            )
            _progress(progress, "failed", **meta)
            _write_meta(meta_path, meta)
            return 2
        status = "stopped" if STOP_REQUESTED else "completed"
        meta.update(
            status=status,
            completed_at=_now() if status == "completed" else None,
            stopped_at=_now() if status == "stopped" else None,
            completed_total=len(done),
            completed_this_invocation=completed,
            duration_this_invocation_s=round(elapsed, 3),
            verdicts=dict(verdicts),
            recorded_row_errors=recorded_errors,
        )
        _progress(progress, status, **meta)
        _write_meta(meta_path, meta)

    if args.export and meta["status"] == "completed":
        try:
            from indra_belief.results import write_run_export

            exported = write_run_export(str(output_path))
            print(
                f"export: data/exports/{exported['run_id']}/ "
                f"({exported['counts']['unique_evidence_rows']} evidence rows)"
            )
        except Exception as exc:
            print(f"export skipped ({type(exc).__name__}): {exc}", file=sys.stderr)
    return 130 if STOP_REQUESTED else 0


def main(argv: list[str] | None = None) -> int:
    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        return _run(args)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
