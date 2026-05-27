"""Viewer-spawned worker (U3 of the pipeline-in-viewer hypergraph).

The SvelteKit viewer invokes this module via
    python -m indra_belief.worker <verb> [--args...]

Subcommands:
    ingest                 — load an INDRA-Statement JSON (or .json.gz) into corpus.duckdb
    register-truth-set     — load a JSONL of tagged records as a truth_set
    estimate-cost          — per-model cost estimate for a corpus
    score                  — run score_corpus end-to-end with SSE-style progress

Output convention: each non-fatal status event is a single JSON object
written to stdout followed by a newline + flush, so the viewer endpoint
can stream events line-by-line for SSE-style progress. Final events
include `event: "done"` with summary fields. Failures raise; stderr
carries the traceback for the endpoint to surface.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time
import uuid


def emit(event: dict) -> None:
    """Write a single newline-terminated JSON event to stdout."""
    sys.stdout.write(json.dumps(event))
    sys.stdout.write("\n")
    sys.stdout.flush()


PENDING_WRITER_LOCK_STALE_SECONDS = 10 * 60
WRITER_LOCK_KINDS = {"paired_score", "single_score", "ingest", "truth_set", "repair"}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _writer_lock_path(db: str) -> Path:
    return Path(db).resolve().parent / "viewer_state" / "writer_lock.json"


def _process_is_alive(pid: int | None) -> bool:
    if pid is None or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except PermissionError:
        return True
    except OSError:
        return False


def _parse_iso(value: object | None) -> float | None:
    if not value:
        return None
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def _read_writer_lock(path: Path) -> dict | None:
    if not path.exists():
        return None
    def malformed(reason: str) -> dict:
        try:
            updated = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
            at = updated.isoformat().replace("+00:00", "Z")
        except OSError:
            at = _now_iso()
        return {
            "kind": "malformed",
            "token": "__malformed_writer_lock__",
            "pid": None,
            "label": "malformed writer lock",
            "source_dump_id": None,
            "dataset_path": None,
            "pair_id": None,
            "architecture": None,
            "model": None,
            "started_at": at,
            "updated_at": at,
            "malformed_reason": reason,
        }

    try:
        lock = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return malformed("writer_lock.json is unreadable JSON")
    if not isinstance(lock, dict):
        return malformed("writer_lock.json must contain an object")
    kind = lock.get("kind")
    if not isinstance(kind, str) or kind not in WRITER_LOCK_KINDS:
        return malformed("writer_lock.json is missing or has invalid kind")
    if not isinstance(lock.get("token"), str) or not lock.get("token"):
        return malformed("writer_lock.json is missing or has invalid token")
    if _parse_iso(lock.get("started_at")) is None:
        return malformed("writer_lock.json is missing or has invalid started_at")
    if _parse_iso(lock.get("updated_at")) is None:
        return malformed("writer_lock.json is missing or has invalid updated_at")
    pid = lock.get("pid")
    if pid is not None and (not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0):
        return malformed("writer_lock.json has invalid pid")
    if lock.get("architecture") not in (None, "monolithic", "decomposed"):
        return malformed("writer_lock.json has invalid architecture")
    for key in ("label", "source_dump_id", "dataset_path", "pair_id", "model"):
        if lock.get(key) is not None and not isinstance(lock.get(key), str):
            return malformed(f"writer_lock.json has invalid {key}")
    return lock


def _lock_is_stale(lock: dict) -> bool:
    if lock.get("kind") == "malformed":
        return False
    pid = lock.get("pid")
    if isinstance(pid, int):
        return not _process_is_alive(pid)
    updated = _parse_iso(lock.get("updated_at"))
    if updated is None:
        return True
    return time.time() - updated > PENDING_WRITER_LOCK_STALE_SECONDS


def _clear_stale_writer_lock(path: Path) -> None:
    lock = _read_writer_lock(path)
    if lock and _lock_is_stale(lock):
        try:
            path.unlink()
        except OSError:
            pass


def _write_json_exclusive(path: Path, value: dict) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
    except FileExistsError:
        return False
    with os.fdopen(fd, "w") as fh:
        json.dump(value, fh, indent=2)
    return True


def _writer_lock_conflict_text(lock: dict) -> str:
    if lock.get("kind") == "malformed":
        detail = f" ({lock.get('malformed_reason')})" if lock.get("malformed_reason") else ""
        return (
            f"writer_lock_malformed: DuckDB writer lock state is malformed{detail}; "
            "reads and writes pause until viewer_state/writer_lock.json is repaired "
            "or removed after confirming no writer is active"
        )
    arch = f" {lock.get('architecture')}" if lock.get("architecture") else ""
    pair = f" pair {lock.get('pair_id')}" if lock.get("pair_id") else ""
    pid = f" pid {lock.get('pid')}" if lock.get("pid") is not None else ""
    return (
        f"DuckDB writer is busy with {lock.get('kind')}{arch}{pair}{pid}; "
        "wait for it to finish or cancel it from the dashboard"
    )


def _writer_lock_block_code(lock: dict | None) -> str:
    return "writer_lock_malformed" if lock and lock.get("kind") == "malformed" else "writer_lock_busy"


@contextmanager
def worker_writer_lock(
    *,
    db: str,
    kind: str,
    label: str,
    source_dump_id: str | None = None,
    dataset_path: str | None = None,
    pair_id: str | None = None,
    architecture: str | None = None,
    model: str | None = None,
):
    """Participate in the viewer's DuckDB writer-lock sidecar.

    Viewer-spawned workers receive INDRA_VIEWER_WRITER_LOCK_TOKEN and must
    validate that token instead of acquiring a second lock. Direct CLI workers
    acquire and clear their own lock, so they cannot silently write around the
    UI's queue/repair state.
    """
    path = _writer_lock_path(db)
    inherited_token = os.environ.get("INDRA_VIEWER_WRITER_LOCK_TOKEN")
    if inherited_token:
        lock = _read_writer_lock(path)
        if lock and lock.get("kind") == "malformed":
            message = _writer_lock_conflict_text(lock)
            emit({"event": "blocked", "code": "writer_lock_malformed", "message": message})
            raise RuntimeError(message)
        if not lock or lock.get("token") != inherited_token or _lock_is_stale(lock):
            message = "viewer writer lock token is missing, stale, or does not match this worker"
            emit({"event": "blocked", "code": "writer_lock_invalid", "message": message})
            raise RuntimeError(message)
        yield
        return

    _clear_stale_writer_lock(path)
    existing = _read_writer_lock(path)
    if existing and not _lock_is_stale(existing):
        message = _writer_lock_conflict_text(existing)
        emit({"event": "blocked", "code": _writer_lock_block_code(existing), "message": message})
        raise RuntimeError(message)

    token = uuid.uuid4().hex
    at = _now_iso()
    lock = {
        "kind": kind,
        "token": token,
        "pid": os.getpid(),
        "label": label,
        "source_dump_id": source_dump_id,
        "dataset_path": dataset_path,
        "pair_id": pair_id,
        "architecture": architecture,
        "model": model,
        "started_at": at,
        "updated_at": at,
    }
    if not _write_json_exclusive(path, lock):
        current = _read_writer_lock(path)
        message = _writer_lock_conflict_text(current or {"kind": "unknown"})
        emit({"event": "blocked", "code": _writer_lock_block_code(current), "message": message})
        raise RuntimeError(message)
    try:
        yield
    finally:
        current = _read_writer_lock(path)
        if current and current.get("token") == token:
            try:
                path.unlink()
            except OSError:
                pass


def _load_stmts_from_any(path: str):
    """Load INDRA Statements from `.json` or `.json.gz` transparently.

    For .gz files we stream-decompress + json.load into memory. The 438MB
    benchmark corpus expands to ~3GB; this is acceptable for an explicit
    user action but would not scale to a server hot path.
    """
    from indra.statements import stmts_from_json, stmts_from_json_file

    if path.endswith(".gz"):
        import gzip

        with gzip.open(path, "rt", encoding="utf-8") as fh:
            return stmts_from_json(json.load(fh))
    return stmts_from_json_file(path)


def do_ingest(args: argparse.Namespace) -> int:
    import duckdb

    from indra_belief.corpus import apply_schema, ingest_statements

    with worker_writer_lock(db=args.db, kind="ingest", label="dataset ingest",
                            source_dump_id=args.source_dump_id,
                            dataset_path=args.path):
        emit({"event": "started", "verb": "ingest", "path": args.path})
        t0 = time.time()
        con = duckdb.connect(args.db)
        try:
            apply_schema(con)
            stmts = _load_stmts_from_any(args.path)
            n_total = len(stmts)
            emit({"event": "loaded", "n_statements": n_total})

            # Emit progress every 100 stmts for the first 1k, then every 500,
            # then every 2500 — keeps SSE legible across 3GB-corpus ingests
            # without spamming on small ones.
            def _on_progress(n: int) -> None:
                if n <= 1000 and n % 100 == 0:
                    step = True
                elif n <= 10_000 and n % 500 == 0:
                    step = True
                elif n % 2500 == 0:
                    step = True
                else:
                    step = False
                if step:
                    emit({
                        "event": "progress",
                        "n_statements_done": n,
                        "n_statements_total": n_total,
                    })

            ingest_statements(
                con,
                stmts,
                source_dump_id=args.source_dump_id,
                on_progress=_on_progress,
            )
            emit({
                "event": "done",
                "n_statements": n_total,
                "duration_s": round(time.time() - t0, 2),
            })
            return 0
        finally:
            con.close()


def do_estimate_cost(args: argparse.Namespace) -> int:
    """Read a JSON of INDRA Statements, estimate cost per supported model."""
    from indra_belief.corpus.cost import MODEL_PRICES_PER_M_TOKENS, estimate_cost

    architecture = getattr(args, "arch", "decomposed")
    probe_step_filter = [
        part.strip()
        for part in (getattr(args, "probe_step_filter", None) or "").split(",")
        if part.strip()
    ] or None
    probe_only = bool(getattr(args, "probe_only", False))
    emit({
        "event": "started",
        "verb": "estimate-cost",
        "path": args.path,
        "architecture": architecture,
        "scoring_mode": "probe_only" if probe_only else "aggregate",
    })
    t0 = time.time()
    stmts = _load_stmts_from_any(args.path)
    estimates: list[dict] = []
    for model_id in MODEL_PRICES_PER_M_TOKENS.keys():
        e = estimate_cost(
            stmts,
            model_id=model_id,
            architecture=architecture,
            probe_step_filter=probe_step_filter,
            probe_only=probe_only,
        )
        estimates.append({
            "model_id": model_id,
            "architecture": architecture,
            "scoring_mode": "probe_only" if probe_only else "aggregate",
            "probe_step_filter": probe_step_filter or [],
            "cost_usd": e["cost_usd"],
            "n_stmts": e["n_stmts"],
            "n_evidences_est": e["n_evidences_est"],
            "n_llm_calls_est": e["n_llm_calls_est"],
        })
    emit({
        "event": "done",
        "architecture": architecture,
        "scoring_mode": "probe_only" if probe_only else "aggregate",
        "probe_step_filter": probe_step_filter or [],
        "n_statements": len(stmts),
        "estimates": estimates,
        "duration_s": round(time.time() - t0, 2),
    })
    return 0


def do_score(args: argparse.Namespace) -> int:
    """Ingest (idempotent) + score a corpus end-to-end. Emits per-evidence progress."""
    import duckdb

    from indra_belief.corpus import (
        apply_schema,
        ingest_statements,
        score_corpus,
    )
    from indra_belief.model_client import ModelClient

    run_id = getattr(args, "run_id", None) or uuid.uuid4().hex
    probe_step_filter = [
        part.strip()
        for part in (getattr(args, "probe_step_filter", None) or "").split(",")
        if part.strip()
    ] or None
    probe_only = bool(getattr(args, "probe_only", False))
    with worker_writer_lock(db=args.db,
                            kind="paired_score" if getattr(args, "paired_run_group_id", None) else "single_score",
                            label=f"{args.arch} score",
                            source_dump_id=args.source_dump_id,
                            dataset_path=args.path,
                            pair_id=getattr(args, "paired_run_group_id", None),
                            architecture=args.arch,
                            model=args.model):
        emit({
            "event": "started",
            "verb": "score",
            "run_id": run_id,
            "path": args.path,
            "model": args.model,
            "scorer_version": args.scorer_version,
            "architecture": args.arch,
            "scoring_mode": "probe_only" if probe_only else "aggregate",
            "parent_run_id": args.parent_run_id,
        })
        t0 = time.time()
        con = duckdb.connect(args.db)
        try:
            apply_schema(con)
            stmts = _load_stmts_from_any(args.path)
            # Count evidences so the viewer's progress bar has a real denominator
            # rather than a fabricated multiplier.
            n_evidences = sum(
                len(getattr(s, "evidence", None) or []) for s in stmts
            )
            emit({
                "event": "loaded",
                "n_statements": len(stmts),
                "n_evidences": n_evidences,
                "architecture": args.arch,
                "scoring_mode": "probe_only" if probe_only else "aggregate",
            })

            if args.skip_ingest:
                emit({
                    "event": "ingest_skipped",
                    "reason": "statements already exist in corpus",
                })
            else:
                # Idempotent ingest — safe to call even if the user already clicked
                # [ingest] before. INSERT OR REPLACE on the natural key (stmt_hash).
                ingest_statements(con, stmts, source_dump_id=args.source_dump_id)
                emit({"event": "ingested"})

            client = ModelClient(args.model)
            evidences_done = [0]

            def on_ev(stmt_hash: str, ev_hash: str, result: dict) -> None:
                evidences_done[0] += 1
                n = evidences_done[0]
                # Emit progress: every evidence for the first 5, every 5 thereafter,
                # then every 25. Keeps the SSE stream legible.
                if n <= 5 or (n <= 50 and n % 5 == 0) or n % 25 == 0:
                    emit({
                        "event": "progress",
                        "n_evidences_done": n,
                        "latest_stmt_hash": stmt_hash,
                        "architecture": args.arch,
                        "cost_so_far_usd": result.get("_cost_so_far_usd"),
                        "cost_cap_usd": result.get("_cost_cap_usd"),
                        "cost_increment_usd": result.get("_cost_actual_increment_usd"),
                    })

            run_id = score_corpus(
                con,
                stmts,
                run_id=run_id,
                client=client,
                scorer_version=args.scorer_version,
                model_id_default=args.model,
                architecture=args.arch,
                paired_run_group_id=getattr(args, "paired_run_group_id", None),
                parent_run_id=getattr(args, "parent_run_id", None),
                decompose=(args.arch == "decomposed"),
                cost_threshold_usd=args.cost_threshold_usd,
                probe_step_filter=probe_step_filter,
                probe_only=probe_only,
                on_evidence=on_ev,
            )
            emit({
                "event": "done",
                "run_id": run_id,
                "architecture": args.arch,
                "scoring_mode": "probe_only" if probe_only else "aggregate",
                "paired_run_group_id": getattr(args, "paired_run_group_id", None),
                "parent_run_id": getattr(args, "parent_run_id", None),
                "n_statements": len(stmts),
                "n_evidences_done": evidences_done[0],
                "duration_s": round(time.time() - t0, 2),
                "cost_cap_usd": args.cost_threshold_usd,
            })
            return 0
        finally:
            con.close()


def do_register_truth_set(args: argparse.Namespace) -> int:
    import duckdb

    from indra_belief.corpus import apply_schema
    from indra_belief.corpus.loader import _hex

    with worker_writer_lock(db=args.db, kind="truth_set",
                            label="truth-set registration",
                            dataset_path=args.path):
        emit({
            "event": "started",
            "verb": "register-truth-set",
            "path": args.path,
            "truth_set_id": args.truth_set_id,
        })
        t0 = time.time()
        con = duckdb.connect(args.db)
        try:
            apply_schema(con)
            n_loaded = 0
            n_skipped = 0
            n_missing_target = 0
            n_missing_relation_target = 0
            n_missing_field = 0
            skipped_examples: list[str] = []
            # Track distinct target_ids so we can honestly report "100 rows
            # processed -> 98 unique evidence targets"; track label keys
            # separately because benchmark source_hash rows can legitimately
            # apply to different matches_hash/statement contexts.
            distinct_targets: set[str] = set()
            distinct_label_keys: set[tuple[str, str | None, str]] = set()
            n_prior_labels = int(con.execute(
                "SELECT COUNT(*) FROM truth_label WHERE truth_set_id=?",
                [args.truth_set_id],
            ).fetchone()[0])

            # Replace truth-set labels transactionally: either the latest file
            # becomes the active set, or the prior active set remains intact.
            con.execute("BEGIN TRANSACTION")
            try:
                con.execute(
                    "DELETE FROM metric WHERE truth_set_id=?",
                    [args.truth_set_id],
                )
                con.execute(
                    "DELETE FROM truth_label WHERE truth_set_id=?",
                    [args.truth_set_id],
                )
                con.execute(
                    "DELETE FROM truth_set WHERE id=?",
                    [args.truth_set_id],
                )
                con.execute(
                    "INSERT INTO truth_set (id, name, description) VALUES (?, ?, ?)",
                    [args.truth_set_id, args.truth_set_name,
                     f"loaded from {args.path} via worker"],
                )

                with open(args.path) as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            rec = json.loads(line)
                        except json.JSONDecodeError:
                            n_skipped += 1
                            continue

                        value = rec.get(args.field)
                        if value is None:
                            n_missing_field += 1
                            if len(skipped_examples) < 3:
                                skipped_examples.append(
                                    f"no `{args.field}` on record"
                                )
                            continue

                        relation_target_id: str | None = None
                        # Resolve target_id from INDRA hashes. Evidence-level
                        # benchmark rows often carry matches_hash too; retain it
                        # as statement context so the same source_hash can be
                        # judged differently under different statements.
                        if args.target_kind == "evidence":
                            raw = rec.get(args.target_hash_field or "source_hash")
                            relation_raw = rec.get("matches_hash")
                            if relation_raw is not None:
                                try:
                                    relation_target_id = _hex(int(relation_raw))
                                except (TypeError, ValueError):
                                    n_missing_relation_target += 1
                                    continue
                        elif args.target_kind == "stmt":
                            raw = rec.get(args.target_hash_field or "matches_hash")
                        else:
                            raise ValueError(
                                f"unsupported target-kind {args.target_kind}"
                            )
                        if raw is None:
                            n_missing_target += 1
                            continue

                        try:
                            target_id = _hex(int(raw))
                        except (TypeError, ValueError):
                            n_missing_target += 1
                            continue

                        # Idempotent write on the contextual natural key.
                        con.execute(
                            "DELETE FROM truth_label "
                            "WHERE truth_set_id=? AND target_kind=? AND target_id=? "
                            "AND field=? AND COALESCE(relation_target_id, '') = COALESCE(?, '')",
                            [args.truth_set_id, args.target_kind, target_id,
                             args.field, relation_target_id],
                        )
                        con.execute(
                            "INSERT INTO truth_label "
                            "(truth_set_id, target_kind, target_id, relation_target_id, field, value_text, provenance) "
                            "VALUES (?, ?, ?, ?, ?, ?, ?)",
                            [args.truth_set_id, args.target_kind, target_id,
                             relation_target_id, args.field, str(value), args.path],
                        )
                        n_loaded += 1
                        distinct_targets.add(target_id)
                        distinct_label_keys.add((target_id, relation_target_id, args.field))
                        if n_loaded % 200 == 0:
                            emit({
                                "event": "progress",
                                "n_loaded": n_loaded,
                                "n_missing_target": n_missing_target,
                                "n_missing_relation_target": n_missing_relation_target,
                                "n_missing_field": n_missing_field,
                            })
                con.execute("COMMIT")
            except Exception:
                try:
                    con.execute("ROLLBACK")
                except Exception:
                    pass
                raise

            # Optionally re-compute validity for the latest succeeded run so the
            # viewer's validity panel grows a P/R/F1 row automatically. Without
            # this, the new truth_set is registered but invisible until the next
            # score_corpus.
            recomputed_run_id: str | None = None
            if args.recompute_latest_validity:
                from indra_belief.corpus import compute_validity

                r = con.execute(
                    "SELECT run_id FROM score_run WHERE status='succeeded' "
                    "ORDER BY started_at DESC LIMIT 1"
                ).fetchone()
                if r:
                    recomputed_run_id = r[0]
                    try:
                        compute_validity(con, recomputed_run_id)
                        emit({
                            "event": "validity_recomputed",
                            "run_id": recomputed_run_id,
                        })
                    except Exception as e:
                        emit({
                            "event": "validity_recompute_failed",
                            "run_id": recomputed_run_id,
                            "error": str(e),
                        })

            emit({
                "event": "done",
                "n_loaded": n_loaded,
                "n_unique_targets": len(distinct_targets),
                "n_unique_labels": len(distinct_label_keys),
                "n_replaced_labels": n_prior_labels,
                "n_missing_target": n_missing_target,
                "n_missing_relation_target": n_missing_relation_target,
                "n_missing_field": n_missing_field,
                "n_skipped": n_skipped,
                "skipped_examples": skipped_examples,
                "recomputed_run_id": recomputed_run_id,
                "duration_s": round(time.time() - t0, 2),
            })
            return 0
        finally:
            con.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="indra_belief.worker",
        description="viewer-spawned worker for ingest / truth-set / score verbs",
    )
    sub = parser.add_subparsers(dest="cmd")

    p_ingest = sub.add_parser("ingest")
    p_ingest.add_argument("--db", required=True)
    p_ingest.add_argument("--path", required=True)
    p_ingest.add_argument("--source-dump-id", required=True)

    p_est = sub.add_parser("estimate-cost")
    p_est.add_argument("--path", required=True)
    p_est.add_argument("--arch", choices=("decomposed", "monolithic"),
                       default="decomposed",
                       help="scoring architecture to estimate")
    p_est.add_argument("--probe-step-filter", default=None,
                       help="comma-separated decomposed probe step kinds to estimate")
    p_est.add_argument("--probe-only", action="store_true",
                       help="estimate only selected decomposed probe steps")

    p_score = sub.add_parser("score")
    p_score.add_argument("--db", required=True)
    p_score.add_argument("--path", required=True)
    p_score.add_argument("--source-dump-id", required=True)
    p_score.add_argument("--model", required=True)
    p_score.add_argument("--scorer-version", required=True)
    p_score.add_argument("--arch", choices=("decomposed", "monolithic"),
                         default="decomposed",
                         help="scoring architecture to run")
    p_score.add_argument("--paired-run-group-id", default=None,
                         help="optional id shared by paired architecture runs")
    p_score.add_argument("--parent-run-id", default=None,
                         help="optional baseline run id for repair/rerun loops")
    p_score.add_argument("--run-id", default=None,
                         help="optional preassigned 32-hex run id")
    p_score.add_argument("--skip-ingest", action="store_true",
                         help="score statements from --path without writing them into statement/evidence tables")
    p_score.add_argument("--cost-threshold-usd", type=float, default=None,
                         help="abort if estimated cost exceeds this dollar cap")
    p_score.add_argument("--probe-step-filter", default=None,
                         help="comma-separated decomposed probe step kinds to materialize as native rows")
    p_score.add_argument("--probe-only", action="store_true",
                         help="resolve only selected decomposed probe steps; merge deterministically when parent trace is available")

    p_truth = sub.add_parser("register-truth-set")
    p_truth.add_argument("--db", required=True)
    p_truth.add_argument("--path", required=True)
    p_truth.add_argument("--truth-set-id", required=True)
    p_truth.add_argument("--truth-set-name", required=True)
    p_truth.add_argument("--target-kind", required=True,
                         choices=["stmt", "evidence"])
    p_truth.add_argument("--field", required=True,
                         help="record field whose value becomes the label")
    p_truth.add_argument("--target-hash-field", default=None,
                         help="record field carrying the INDRA hash; "
                              "defaults to source_hash for evidence, matches_hash for stmt")
    p_truth.add_argument("--recompute-latest-validity", action="store_true",
                         help="after registering, re-run compute_validity for "
                              "the latest succeeded run so the new truth_set's "
                              "P/R/F1 row appears in the viewer")

    args = parser.parse_args(argv)

    if args.cmd == "ingest":
        return do_ingest(args)
    if args.cmd == "estimate-cost":
        return do_estimate_cost(args)
    if args.cmd == "score":
        return do_score(args)
    if args.cmd == "register-truth-set":
        return do_register_truth_set(args)

    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
