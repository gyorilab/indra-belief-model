"""Stream processed INDRA statements into Gilda-prepared evidence shards.

This is Stage A of the full-corpus monolithic pipeline. It does not call an
LLM. The input is the gzip TSV emitted by ``export_assembly.py``:

    statement_hash<TAB>statement_json

The file is read one row at a time. Statements with no evidence and evidence
objects with no text are skipped. Every remaining (Statement, Evidence) pair
is grounded once, checked by deterministic Tier 1, rendered into the
monolithic user message, and written to an atomic gzip JSONL shard.

Defaults are tailored to the intended HPC layout:

    input:  /scratch/h.yan/data/processed_statements.tsv.gz
    output: /scratch/h.yan/data/processed_grounding_shards/

Resume is automatic. The manifest advances only after a complete input row is
durably represented in committed shards, so a statement's evidence list is
never split across a checkpoint boundary. A persistent SQLite cache prevents
repeating Gilda work across input rows and resumed invocations.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import signal
import sqlite3
import sys
import time
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

DEFAULT_INPUT = Path("/scratch/h.yan/data/processed_statements.tsv.gz")
DEFAULT_OUTPUT_DIR = Path("/scratch/h.yan/data/processed_grounding_shards")
SCHEMA_VERSION = 1
STOP_REQUESTED = False


def _request_stop(signum, _frame) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print(
        f"\nreceived signal {signum}; finishing the current input row",
        file=sys.stderr,
        flush=True,
    )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _configure_csv_field_limit() -> None:
    """Allow large statement JSON fields on platforms with a small CSV limit."""
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def iter_processed_rows(
    path: Path,
    *,
    start_row: int = 0,
) -> Iterable[tuple[int, int, dict[str, Any]]]:
    """Yield ``(zero-based input row, statement hash, statement JSON)``."""
    _configure_csv_field_limit()
    with gzip.open(path, "rt", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh, delimiter="\t")
        for row_index, row in enumerate(reader):
            if row_index < start_row:
                continue
            if len(row) != 2:
                raise ValueError(
                    f"{path}: input row {row_index + 1} has {len(row)} columns; "
                    "expected statement_hash<TAB>statement_json"
                )
            stmt_hash_text, stmt_json_text = row
            try:
                stmt_hash = int(stmt_hash_text)
                stmt_json = json.loads(stmt_json_text)
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"{path}: invalid input row {row_index + 1}: {exc}"
                ) from exc
            if not isinstance(stmt_json, dict):
                raise ValueError(
                    f"{path}: statement JSON on row {row_index + 1} "
                    "is not an object"
                )
            yield row_index, stmt_hash, stmt_json


def text_evidence_items(
    stmt_json: dict[str, Any],
) -> tuple[list[tuple[int, dict[str, Any]]], Counter[str]]:
    """Return evidence entries with non-empty text plus filtering counters."""
    counts: Counter[str] = Counter()
    evidences = stmt_json.get("evidence") or []
    if not isinstance(evidences, list):
        raise ValueError("statement evidence is not a list")
    if not evidences:
        counts["statements_without_evidence"] = 1
        return [], counts

    selected: list[tuple[int, dict[str, Any]]] = []
    for evidence_index, evidence_json in enumerate(evidences):
        if not isinstance(evidence_json, dict):
            counts["malformed_evidence"] += 1
            continue
        text = evidence_json.get("text")
        if not isinstance(text, str) or not text.strip():
            counts["evidences_without_text"] += 1
            continue
        selected.append((evidence_index, evidence_json))
    return selected, counts


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    )
    tmp.replace(path)


def _code_fingerprint() -> str:
    """Pin the preparation logic that materially affects shard contents."""
    paths = [
        Path(__file__),
        ROOT / "src" / "indra_belief" / "data" / "entity.py",
        ROOT / "src" / "indra_belief" / "data" / "scoring_record.py",
        ROOT / "src" / "indra_belief" / "tools" / "gilda_tools.py",
    ]
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(ROOT)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


class GroundingCache:
    """Persistent cache for GroundedEntity snapshots and lookup text."""

    def __init__(self, path: Path, namespace: str):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.namespace = namespace
        self.connection = sqlite3.connect(path)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=NORMAL")
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS entity_grounding (
                cache_key TEXT PRIMARY KEY,
                payload TEXT NOT NULL
            )
            """
        )
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS lookup_context (
                lookup_target TEXT PRIMARY KEY,
                payload TEXT NOT NULL
            )
            """
        )
        self.connection.commit()
        self.hits = 0
        self.misses = 0
        self.lookup_hits = 0
        self.lookup_misses = 0
        self._writes_since_commit = 0

    @staticmethod
    def _cache_key(*parts: str | None) -> str:
        return json.dumps(
            parts,
            ensure_ascii=False,
            separators=(",", ":"),
        )

    def resolve_entity(self, name: str, raw_text: str | None, entity_cls):
        key = self._cache_key(self.namespace, name, raw_text)
        row = self.connection.execute(
            "SELECT payload FROM entity_grounding WHERE cache_key = ?",
            (key,),
        ).fetchone()
        if row is not None:
            self.hits += 1
            return entity_cls(**json.loads(row[0]))

        self.misses += 1
        entity = entity_cls.resolve(name, raw_text)
        payload = json.dumps(asdict(entity), ensure_ascii=False)
        self.connection.execute(
            "INSERT OR REPLACE INTO entity_grounding(cache_key, payload) "
            "VALUES (?, ?)",
            (key, payload),
        )
        self._after_write()
        return entity

    def lookup(
        self,
        target: str,
        compute: Callable[[str], str],
    ) -> str:
        lookup_key = self._cache_key(self.namespace, target)
        row = self.connection.execute(
            "SELECT payload FROM lookup_context WHERE lookup_target = ?",
            (lookup_key,),
        ).fetchone()
        if row is not None:
            self.lookup_hits += 1
            return str(row[0])

        self.lookup_misses += 1
        payload = compute(target)
        self.connection.execute(
            "INSERT OR REPLACE INTO lookup_context(lookup_target, payload) "
            "VALUES (?, ?)",
            (lookup_key, payload),
        )
        self._after_write()
        return payload

    def _after_write(self) -> None:
        self._writes_since_commit += 1
        if self._writes_since_commit >= 500:
            self.connection.commit()
            self._writes_since_commit = 0

    def commit(self) -> None:
        self.connection.commit()
        self._writes_since_commit = 0

    def close(self) -> None:
        self.commit()
        self.connection.close()


class AtomicShardWriter:
    """Write one gzip JSONL shard and publish it with an atomic rename."""

    def __init__(self, output_dir: Path, shard_index: int, compresslevel: int):
        self.output_dir = output_dir
        self.shard_index = shard_index
        self.compresslevel = compresslevel
        self.count = 0
        self._fh = None
        self.final_path = output_dir / f"grounded-{shard_index:06d}.jsonl.gz"
        self.tmp_path = output_dir / f".grounded-{shard_index:06d}.jsonl.gz.tmp"

    def write(self, row: dict[str, Any]) -> None:
        if self._fh is None:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            if self.tmp_path.exists():
                self.tmp_path.unlink()
            self._fh = gzip.open(
                self.tmp_path,
                "wt",
                encoding="utf-8",
                compresslevel=self.compresslevel,
            )
        self._fh.write(
            json.dumps(row, ensure_ascii=False, default=str, separators=(",", ":"))
            + "\n"
        )
        self.count += 1

    def commit(self) -> dict[str, Any] | None:
        if self._fh is None:
            return None
        self._fh.close()
        self._fh = None
        self.tmp_path.replace(self.final_path)
        return {
            "index": self.shard_index,
            "path": self.final_path.name,
            "jobs": self.count,
        }

    def abort(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None
        if self.tmp_path.exists():
            self.tmp_path.unlink()


def _entity_inputs(statement, evidence) -> tuple[
    tuple[str, str | None] | None,
    tuple[str, str | None] | None,
]:
    """Mirror ScoringRecord's subject/object and raw-text mapping."""
    from indra.statements import SelfModification

    agents = statement.agent_list()
    subject = agents[0].name if agents and agents[0] else "?"
    if isinstance(statement, SelfModification):
        obj = subject
    elif len(agents) > 1 and agents[1]:
        obj = agents[1].name
    else:
        obj = "?"

    raw_texts = (
        evidence.annotations.get("agents", {}).get("raw_text") or []
    )
    clean_raw_texts = [value for value in raw_texts if value is not None]
    subject_raw = clean_raw_texts[0] if clean_raw_texts else None
    object_raw = clean_raw_texts[1] if len(clean_raw_texts) > 1 else None

    subject_input = None if subject == "?" else (subject, subject_raw)
    object_input = None if obj == "?" else (obj, object_raw)
    return subject_input, object_input


def _compact_entity(entity) -> dict[str, Any] | None:
    if entity is None:
        return None
    fields = (
        "name",
        "raw_text",
        "canonical",
        "db",
        "db_id",
        "aliases",
        "is_family",
        "family_members",
        "description",
        "is_pseudogene",
        "verification_status",
        "verification_note",
        "gilda_score",
        "is_low_confidence",
        "is_known_alias",
        "competing_candidates",
        "text_top_name",
    )
    return {field: getattr(entity, field) for field in fields}


def _lookup_context(record, cache: GroundingCache) -> str:
    from indra_belief.tools.gilda_tools import lookup_gene_executor

    lines: list[str] = []
    seen: set[str] = set()
    for entity in (record.subject_entity, record.object_entity):
        if not entity or not entity.name or entity.name == "?":
            continue
        target = entity.raw_text or entity.name
        if target in seen:
            continue
        seen.add(target)
        lines.append(
            cache.lookup(
                target,
                lambda value: lookup_gene_executor({"entity_name": value}),
            )
        )
    return "Entity database lookups:\n" + "\n".join(lines) if lines else ""


def prepare_statement_jobs(
    *,
    input_row_index: int,
    stmt_hash: int,
    stmt_json: dict[str, Any],
    selected_evidence: list[tuple[int, dict[str, Any]]],
    cache: GroundingCache,
) -> list[dict[str, Any]]:
    """Deserialize one statement and prepare all selected evidence jobs."""
    from indra.statements import stmt_from_json

    from indra_belief.data.entity import GroundedEntity
    from indra_belief.data.scoring_record import ScoringRecord

    statement = stmt_from_json(stmt_json)
    if statement is None:
        raise ValueError("INDRA could not deserialize statement JSON")

    jobs: list[dict[str, Any]] = []
    for evidence_index, evidence_json in selected_evidence:
        if evidence_index >= len(statement.evidence):
            raise ValueError(
                f"evidence index {evidence_index} is missing after deserialization"
            )
        evidence = statement.evidence[evidence_index]
        subject_input, object_input = _entity_inputs(statement, evidence)
        subject_entity = (
            cache.resolve_entity(*subject_input, GroundedEntity)
            if subject_input
            else None
        )
        object_entity = (
            cache.resolve_entity(*object_input, GroundedEntity)
            if object_input
            else None
        )
        record = ScoringRecord(
            statement=statement,
            evidence=evidence,
            subject_entity=subject_entity,
            object_entity=object_entity,
        )

        tier1_result = record.tier1_auto_reject()
        flagged = any(
            entity.has_grounding_signal
            for entity in (record.subject_entity, record.object_entity)
            if entity
        )
        needs_llm = tier1_result is None
        user_message = None
        lookup_context = ""
        if needs_llm:
            user_message = record.format_user_message()
            if flagged:
                lookup_context = _lookup_context(record, cache)
                if lookup_context:
                    user_message += "\n\n" + lookup_context

        source_hash = evidence_json.get("source_hash")
        if source_hash is None:
            source_hash = evidence.get_source_hash()

        jobs.append(
            {
                "schema_version": SCHEMA_VERSION,
                "job_id": f"{input_row_index}:{evidence_index}",
                "input_row_index": input_row_index,
                "evidence_index": evidence_index,
                "stmt_hash": stmt_hash,
                "statement_uuid": getattr(statement, "uuid", None),
                "source_hash": source_hash,
                "source_api": evidence.source_api or "",
                "pmid": evidence.pmid,
                "stmt_type": record.stmt_type,
                "subject": record.subject,
                "object": record.object,
                "claim": record.format_claim(),
                "evidence_text": record.evidence_text,
                "subject_grounding": _compact_entity(record.subject_entity),
                "object_grounding": _compact_entity(record.object_entity),
                "grounding_status": (
                    tier1_result.get("grounding_status")
                    if tier1_result
                    else ("flagged" if flagged else "all_match")
                ),
                "lookup_guidance_required": bool(lookup_context),
                "needs_llm": needs_llm,
                "tier1_result": tier1_result,
                "user_message": user_message,
            }
        )
    return jobs


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--cache",
        default=None,
        help="SQLite Gilda cache; defaults to OUTPUT_DIR/gilda_cache.sqlite3.",
    )
    parser.add_argument("--shard-size", type=int, default=50_000)
    parser.add_argument("--compresslevel", type=int, default=6)
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--limit-input-rows",
        type=int,
        default=None,
        help="Stop at this absolute input-row count; useful for a smoke test.",
    )
    parser.add_argument(
        "--on-error",
        choices=("fail", "skip"),
        default="fail",
        help="Fail loudly or skip malformed/deserialization rows.",
    )
    parser.add_argument("--progress-every", type=int, default=10_000)
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.shard_size < 1:
        raise SystemExit("--shard-size must be >= 1")
    if not 0 <= args.compresslevel <= 9:
        raise SystemExit("--compresslevel must be between 0 and 9")
    if args.limit_input_rows is not None and args.limit_input_rows < 1:
        raise SystemExit("--limit-input-rows must be >= 1")
    if args.progress_every < 1:
        raise SystemExit("--progress-every must be >= 1")


def main() -> int:
    args = _build_parser().parse_args()
    _validate_args(args)
    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)

    input_path = Path(args.input).resolve()
    output_dir = Path(args.output_dir).resolve()
    cache_path = (
        Path(args.cache).resolve()
        if args.cache
        else output_dir / "gilda_cache.sqlite3"
    )
    manifest_path = output_dir / "manifest.json"

    if not input_path.exists():
        raise SystemExit(f"input does not exist: {input_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    stat = input_path.stat()
    identity = {
        "schema_version": SCHEMA_VERSION,
        "input": str(input_path),
        "input_size": stat.st_size,
        "input_mtime_ns": stat.st_mtime_ns,
        "preparation_code_sha256": _code_fingerprint(),
        "shard_size": args.shard_size,
        "compresslevel": args.compresslevel,
    }

    if manifest_path.exists():
        if not args.resume:
            raise SystemExit(
                f"{manifest_path} exists; use --resume or a new --output-dir"
            )
        manifest = json.loads(manifest_path.read_text())
        mismatches = {
            key: (manifest.get(key), value)
            for key, value in identity.items()
            if manifest.get(key) != value
        }
        if mismatches:
            raise SystemExit(
                "resume configuration/input mismatch: "
                + json.dumps(mismatches, sort_keys=True)
            )
    else:
        existing_shards = list(output_dir.glob("grounded-*.jsonl.gz"))
        if existing_shards:
            raise SystemExit(
                f"{output_dir} has shards but no manifest; refusing unsafe resume"
            )
        manifest = {
            **identity,
            "status": "new",
            "created_at": _now(),
            "input_rows_consumed": 0,
            "next_shard_index": 0,
            "shards": [],
            "counters": {},
        }
        _write_json_atomic(manifest_path, manifest)

    start_row = int(manifest.get("input_rows_consumed") or 0)
    shard_index = int(manifest.get("next_shard_index") or 0)
    counters: Counter[str] = Counter(manifest.get("counters") or {})
    writer = AtomicShardWriter(output_dir, shard_index, args.compresslevel)
    cache = GroundingCache(cache_path, identity["preparation_code_sha256"])
    started = time.perf_counter()
    last_consumed = start_row
    durable_consumed = start_row
    durable_counters: Counter[str] = counters.copy()
    rows_this_invocation = 0

    def save_manifest(status: str) -> None:
        manifest.update(
            {
                **identity,
                "status": status,
                "updated_at": _now(),
                "input_rows_consumed": durable_consumed,
                "next_shard_index": writer.shard_index,
                "counters": dict(durable_counters),
                "cache": str(cache_path),
                "cache_stats_this_invocation": {
                    "entity_hits": cache.hits,
                    "entity_misses": cache.misses,
                    "lookup_hits": cache.lookup_hits,
                    "lookup_misses": cache.lookup_misses,
                },
            }
        )
        _write_json_atomic(manifest_path, manifest)

    def commit_current_shard() -> None:
        nonlocal writer, durable_consumed, durable_counters
        shard_meta = writer.commit()
        if shard_meta is None:
            return
        manifest.setdefault("shards", []).append(shard_meta)
        counters["shards_written"] += 1
        cache.commit()
        durable_consumed = last_consumed
        durable_counters = counters.copy()
        next_index = writer.shard_index + 1
        writer = AtomicShardWriter(output_dir, next_index, args.compresslevel)
        save_manifest("running")

    print(f"input={input_path}")
    print(f"output_dir={output_dir}")
    print(f"resume_from_input_row={start_row:,} shard_index={shard_index}")
    print(f"gilda_cache={cache_path}", flush=True)

    try:
        for input_row_index, stmt_hash, stmt_json in iter_processed_rows(
            input_path,
            start_row=start_row,
        ):
            if (
                args.limit_input_rows is not None
                and input_row_index >= args.limit_input_rows
            ):
                break

            row_counts: Counter[str] = Counter(input_rows_seen=1)
            try:
                selected, filter_counts = text_evidence_items(stmt_json)
                row_counts.update(filter_counts)
                if selected:
                    row_jobs = prepare_statement_jobs(
                        input_row_index=input_row_index,
                        stmt_hash=stmt_hash,
                        stmt_json=stmt_json,
                        selected_evidence=selected,
                        cache=cache,
                    )
                else:
                    row_jobs = []
            except Exception as exc:
                if args.on_error == "fail":
                    raise RuntimeError(
                        f"failed at input row {input_row_index + 1}, "
                        f"stmt_hash={stmt_hash}"
                    ) from exc
                row_counts["error_rows_skipped"] += 1
                row_jobs = []
                print(
                    f"warning: skipped input row {input_row_index + 1} "
                    f"stmt_hash={stmt_hash}: {type(exc).__name__}: {exc}",
                    file=sys.stderr,
                )

            # Keep every input statement's evidence jobs on the same side of a
            # checkpoint. A single unusually large statement may exceed the
            # nominal shard size, but it remains resume-safe.
            if (
                writer.count > 0
                and writer.count + len(row_jobs) > args.shard_size
            ):
                commit_current_shard()

            for job in row_jobs:
                writer.write(job)
                row_counts["evidence_jobs_written"] += 1
                if job["needs_llm"]:
                    row_counts["needs_llm"] += 1
                else:
                    row_counts["tier1_resolved"] += 1

            counters.update(row_counts)
            last_consumed = input_row_index + 1
            rows_this_invocation += 1

            if (
                rows_this_invocation == 1
                or rows_this_invocation % args.progress_every == 0
            ):
                elapsed = time.perf_counter() - started
                print(
                    f"rows={last_consumed:,} "
                    f"jobs={counters['evidence_jobs_written']:,} "
                    f"needs_llm={counters['needs_llm']:,} "
                    f"tier1={counters['tier1_resolved']:,} "
                    f"no_evidence={counters['statements_without_evidence']:,} "
                    f"no_text={counters['evidences_without_text']:,} "
                    f"rate={rows_this_invocation / max(elapsed, 1e-9):.1f} rows/s",
                    flush=True,
                )

            if STOP_REQUESTED:
                break

        commit_current_shard()
        cache.commit()
        # Rows that produced no jobs still need a durable resume checkpoint.
        durable_consumed = last_consumed
        durable_counters = counters.copy()
        if STOP_REQUESTED:
            status = "stopped"
        elif (
            args.limit_input_rows is not None
            and last_consumed >= args.limit_input_rows
        ):
            status = "limited"
        else:
            status = "completed"
        save_manifest(status)
    except BaseException:
        writer.abort()
        cache.commit()
        save_manifest("failed")
        raise
    finally:
        cache.close()

    print("\nPreparation summary")
    print(f"  status:                       {manifest['status']}")
    print(f"  input rows consumed:          {last_consumed:,}")
    print(
        "  statements without evidence: "
        f"{counters['statements_without_evidence']:,}"
    )
    print(
        f"  evidence without text:        {counters['evidences_without_text']:,}"
    )
    print(f"  evidence jobs written:        {counters['evidence_jobs_written']:,}")
    print(f"  deterministic Tier 1:         {counters['tier1_resolved']:,}")
    print(f"  needs vLLM:                   {counters['needs_llm']:,}")
    print(f"  shards:                       {counters['shards_written']:,}")
    print(f"  manifest:                     {manifest_path}")
    return 130 if STOP_REQUESTED else 0


if __name__ == "__main__":
    raise SystemExit(main())
