"""DuckDB schema for the INDRA belief-rescoring corpus.

Specified by Phase 2.1 of `research/rasmachine_task_graph.md`. The schema
must survive scorer-architecture iterations: every scorer step is its own
row keyed by `(stmt_hash, scorer_version, step_kind)`, append-only with
versioned migrations layered alongside (never overwriting).

Truth-set support is foundational (D7). `truth_set` registers labelled
sources (INDRA published belief, INDRA epistemics flags, gold pool, etc.)
and `truth_label` is a polymorphic attachment that can target any of:
statement, evidence, agent, scorer_step, supports_edge.

Hashes (per Phase 1.5 lock):
  - `stmt_hash` = `Statement.get_hash(shallow=True)` 14-nibble hex, stored as VARCHAR
  - `extraction_hash` = `Statement.get_hash(shallow=False)` 16-nibble, identifies
    a specific dump's statement-with-evidence
  - `evidence_hash` = `Evidence.source_hash` (INDRA content-addressed; NOT
    `matches_key()` — that has dict-ordering instability). The same evidence
    hash may appear under multiple statements, so statement membership is
    authoritative in `statement_evidence`; `evidence` stores the canonical
    payload keyed by evidence hash for legacy/detail joins.
  - `agent_hash`, `step_hash` = stable hashes computed at write time
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import duckdb

log = logging.getLogger(__name__)

SCHEMA_VERSION = 5


# Step kinds emitted by the composed scorer (per Phase 1.2 catalog of
# the 9-step pipeline).
SCORER_STEP_KINDS = (
    "parse_claim",
    "build_context",
    "substrate_route",
    "subject_role_probe",
    "object_role_probe",
    "relation_axis_probe",
    "scope_probe",
    "grounding",
    "adjudicate",
)

# Target kinds for polymorphic truth labels (per Phase 1.3 catalog).
TRUTH_TARGET_KINDS = (
    "stmt",
    "evidence",
    "agent",
    "scorer_step",
    "supports_edge",
)


_DDL = f"""
-- ─── INDRA-native object tables ────────────────────────────────────────
CREATE TABLE IF NOT EXISTS statement (
    stmt_hash         VARCHAR PRIMARY KEY,        -- Statement.get_hash(shallow=True)
    extraction_hash   VARCHAR,                    -- Statement.get_hash(shallow=False)
    indra_uuid        VARCHAR,                    -- per-extraction UUID
    indra_type        VARCHAR NOT NULL,           -- discriminator: Phosphorylation, Activation, …
    indra_belief      DOUBLE,                     -- INDRA's published belief ∈ [0,1]
    supports_count    INTEGER NOT NULL DEFAULT 0,
    supported_by_count INTEGER NOT NULL DEFAULT 0,
    raw_json          JSON NOT NULL,              -- canonical full Statement.to_json()
    loaded_at         TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    source_dump_id    VARCHAR                     -- which corpus ingest brought this in
);
CREATE INDEX IF NOT EXISTS idx_statement_indra_type ON statement(indra_type);
CREATE INDEX IF NOT EXISTS idx_statement_source_dump ON statement(source_dump_id);

CREATE TABLE IF NOT EXISTS evidence (
    evidence_hash     VARCHAR PRIMARY KEY,        -- Evidence.source_hash
    source_api        VARCHAR,                    -- reach, biopax, biogrid, signor, …
    source_id         VARCHAR,
    pmid              VARCHAR,
    text              TEXT,
    -- Top-level epistemics columns (G1: query-friendly mirror of epistemics_json):
    is_direct         BOOLEAN,
    is_negated        BOOLEAN,
    is_curated        BOOLEAN,
    epistemics_json   JSON,
    annotations_json  JSON,
    raw_json          JSON NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_evidence_source_api ON evidence(source_api);
CREATE INDEX IF NOT EXISTS idx_evidence_pmid ON evidence(pmid);

-- `statement_evidence` is the authoritative contextual membership table.
-- `evidence_index` reflects ingest order from `raw_json.evidence[]` for live
-- ingest rows. Rows materialized by `statement_evidence_backfill_v1` from
-- legacy `evidence` rows use evidence_hash lexicographic ordering (the
-- original ingest order is unrecoverable from `evidence` alone). Treat the
-- column as a stable per-(stmt_hash, evidence_hash) identifier on backfilled
-- corpora; new ingests get the literal array position.
CREATE TABLE IF NOT EXISTS statement_evidence (
    stmt_hash         VARCHAR NOT NULL,
    evidence_hash     VARCHAR NOT NULL,
    evidence_index    INTEGER NOT NULL DEFAULT 0,
    source_dump_id    VARCHAR,
    PRIMARY KEY (stmt_hash, evidence_hash)
);
CREATE INDEX IF NOT EXISTS idx_statement_evidence_stmt ON statement_evidence(stmt_hash);
CREATE INDEX IF NOT EXISTS idx_statement_evidence_evidence ON statement_evidence(evidence_hash);

-- Agent participation in a statement. Same agent (by agent_hash) can appear
-- in many statements; PK is the participation, not the identity.
CREATE TABLE IF NOT EXISTS agent (
    stmt_hash         VARCHAR NOT NULL,           -- FK statement.stmt_hash
    agent_hash        VARCHAR NOT NULL,           -- canonical agent identity (matches_key hash)
    role              VARCHAR NOT NULL,           -- subj, obj, enz, sub, member, …
    role_index        INTEGER NOT NULL DEFAULT 0, -- ordinal within role (e.g. members[0..N])
    name              VARCHAR NOT NULL,
    db_refs_json      JSON NOT NULL,
    mods_json         JSON,
    mutations_json    JSON,
    bound_conditions_json JSON,
    activity_json     JSON,
    location          VARCHAR,
    PRIMARY KEY (stmt_hash, agent_hash, role, role_index)
);
CREATE INDEX IF NOT EXISTS idx_agent_stmt ON agent(stmt_hash);
CREATE INDEX IF NOT EXISTS idx_agent_hash ON agent(agent_hash);
CREATE INDEX IF NOT EXISTS idx_agent_name ON agent(name);

-- INDRA's supports/supported_by are UUIDs in the JSON dump, NOT live refs.
-- Reconstruction is a separate post-ingest pass; this table holds the edges.
CREATE TABLE IF NOT EXISTS supports_edge (
    from_stmt_hash    VARCHAR NOT NULL,
    to_stmt_hash      VARCHAR NOT NULL,
    kind              VARCHAR NOT NULL,           -- 'supports' or 'supported_by'
    source_dump_id    VARCHAR,
    PRIMARY KEY (from_stmt_hash, to_stmt_hash, kind)
);
CREATE INDEX IF NOT EXISTS idx_supports_from ON supports_edge(from_stmt_hash);
CREATE INDEX IF NOT EXISTS idx_supports_to ON supports_edge(to_stmt_hash);

-- ─── Truth-set registry + polymorphic labels ───────────────────────────
CREATE TABLE IF NOT EXISTS truth_set (
    id            VARCHAR PRIMARY KEY,            -- e.g., 'indra_published_belief', 'gold_pool_v15'
    name          VARCHAR NOT NULL,
    source        VARCHAR,                        -- 'indra_evidence_epistemics', 'project_annotators', …
    version       VARCHAR,
    loaded_at     TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    description   TEXT
);

CREATE SEQUENCE IF NOT EXISTS truth_label_id_seq;
CREATE TABLE IF NOT EXISTS truth_label (
    label_id            BIGINT PRIMARY KEY DEFAULT nextval('truth_label_id_seq'),
    truth_set_id        VARCHAR NOT NULL,         -- FK truth_set.id
    target_kind         VARCHAR NOT NULL,         -- stmt | evidence | agent | scorer_step | supports_edge
    target_id           VARCHAR NOT NULL,         -- the relevant *_hash
    relation_target_id  VARCHAR,                  -- optional context hash: stmt_hash for evidence labels, supports_edge.to_stmt_hash for edge labels
    field               VARCHAR NOT NULL,         -- e.g., 'belief', 'is_direct', 'verdict', 'subj_grounding'
    value_text          VARCHAR,                  -- scalar value as text
    value_json          JSON,                     -- non-scalar value (e.g., grounding correction)
    confidence          DOUBLE,                   -- annotator confidence if known
    provenance          VARCHAR                   -- 'biogrid_v4.4_curated', 'annotator_eric', …
);
-- App-level uniqueness on the natural key is enforced in `loader.py` / the
-- worker via DELETE+INSERT on (truth_set_id, target_kind, target_id,
-- relation_target_id, field). DuckDB's INSERT OR REPLACE / ON
-- CONFLICT requires the conflict target to be a UNIQUE/PK constraint —
-- the natural key is an index, not a constraint, because this table
-- uses a surrogate label_id PK. The DELETE-then-INSERT pattern is the
-- equivalent. Locked by `test_load_truth_labels_is_idempotent` and
-- `test_re_ingest_idempotent` in tests/test_corpus_loader.py.
CREATE INDEX IF NOT EXISTS idx_truth_label_natural ON truth_label(truth_set_id, target_kind, target_id, field);
CREATE INDEX IF NOT EXISTS idx_truth_label_natural_context ON truth_label(truth_set_id, target_kind, target_id, relation_target_id, field);
CREATE INDEX IF NOT EXISTS idx_truth_label_target ON truth_label(target_kind, target_id);
CREATE INDEX IF NOT EXISTS idx_truth_label_set ON truth_label(truth_set_id);
CREATE INDEX IF NOT EXISTS idx_truth_label_field ON truth_label(field);

-- ─── Scorer traces ─────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS score_run (
    run_id             VARCHAR PRIMARY KEY,
    scorer_version     VARCHAR NOT NULL,          -- git commit hash of indra-belief
    indra_version      VARCHAR NOT NULL,          -- indra.__version__ at run time
    architecture       VARCHAR NOT NULL DEFAULT 'unknown', -- decomposed | monolithic | mixed | unknown
    paired_run_group_id VARCHAR,                   -- shared id for paired architecture experiments
    parent_run_id      VARCHAR,                    -- baseline run for repair / rerun loops
    repair_kind        VARCHAR,                    -- 'rerun' | 'review_only' | NULL; typed lineage for Phase 4
    model_id_default   VARCHAR,                   -- LLM identifier (per-step may override)
    started_at         TIMESTAMP NOT NULL,
    finished_at        TIMESTAMP,
    n_stmts            INTEGER,
    status             VARCHAR NOT NULL,          -- running | succeeded | failed | canceled | pre_started_cancelled | crashed_at_startup
    cost_estimate_usd  DOUBLE,
    cost_actual_usd    DOUBLE,
    reviewed_at        TIMESTAMP,
    reviewed_by        VARCHAR,
    review_status      VARCHAR,
    review_notes       TEXT,
    terminated_by      VARCHAR,                   -- user | worker_error | janitor | system
    termination_reason TEXT,
    notes              TEXT
);
CREATE INDEX IF NOT EXISTS idx_score_run_parent_run_id ON score_run(parent_run_id);
CREATE INDEX IF NOT EXISTS idx_score_run_paired_group ON score_run(paired_run_group_id);
CREATE INDEX IF NOT EXISTS idx_score_run_status ON score_run(status);

CREATE TABLE IF NOT EXISTS scorer_step (
    step_hash             VARCHAR PRIMARY KEY,    -- composite hash over (run_id, stmt_hash, evidence_hash, scorer_version, model_id, architecture, step_kind, input_payload)
    stmt_hash             VARCHAR NOT NULL,
    evidence_hash         VARCHAR,                -- nullable: parse_claim is statement-level, no evidence
    run_id                VARCHAR,                -- FK score_run.run_id
    scorer_version        VARCHAR NOT NULL,
    architecture          VARCHAR NOT NULL DEFAULT 'decomposed',
    model_id              VARCHAR,                -- nullable for substrate-only steps
    step_kind             VARCHAR NOT NULL,       -- enumerated in SCORER_STEP_KINDS
    is_substrate_answered BOOLEAN,                -- true for substrate-answered probe rows; null for non-probe steps
    input_payload_json    JSON,
    output_json           JSON,                   -- shape varies per step_kind; see scorer-output catalog
    latency_ms            INTEGER,
    prompt_tokens         INTEGER,
    out_tokens            INTEGER,
    finish_reason         VARCHAR,
    error                 VARCHAR,
    started_at            TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_scorer_step_stmt ON scorer_step(stmt_hash);
CREATE INDEX IF NOT EXISTS idx_scorer_step_evidence ON scorer_step(evidence_hash);
CREATE INDEX IF NOT EXISTS idx_scorer_step_run ON scorer_step(run_id);
CREATE INDEX IF NOT EXISTS idx_scorer_step_kind ON scorer_step(step_kind);
CREATE INDEX IF NOT EXISTS idx_scorer_step_version ON scorer_step(scorer_version);
CREATE INDEX IF NOT EXISTS idx_scorer_step_arch ON scorer_step(architecture);

-- ─── Append-only reviewer corrections / repair backlog ─────────────────
CREATE SEQUENCE IF NOT EXISTS scorer_step_correction_id_seq;
CREATE TABLE IF NOT EXISTS scorer_step_correction (
    correction_id       BIGINT PRIMARY KEY DEFAULT nextval('scorer_step_correction_id_seq'),
    step_hash           VARCHAR NOT NULL,          -- FK scorer_step.step_hash
    run_id              VARCHAR NOT NULL,          -- denormalized for fast run/cohort repair views
    architecture         VARCHAR NOT NULL DEFAULT 'unknown',
    stmt_hash           VARCHAR NOT NULL,
    evidence_hash       VARCHAR,
    correction_kind     VARCHAR NOT NULL,          -- repair_candidate | label_correction | reviewer_note
    status              VARCHAR NOT NULL DEFAULT 'open', -- open | resolved | rejected | superseded
    reviewer            VARCHAR,
    note                TEXT,
    value_json          JSON,                      -- structured correction payload when known
    parent_correction_id BIGINT,                   -- typed repair-rerun lineage projection
    child_run_id        VARCHAR,                   -- typed repair-rerun lineage projection
    repair_source_dump_id VARCHAR,                 -- typed repair-rerun lineage projection
    source_route        VARCHAR,                   -- cohort/workbench URL that produced the task
    source_filters_json JSON,                      -- exact cohort filters at creation
    created_at          TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_scorer_step_correction_step ON scorer_step_correction(step_hash);
CREATE INDEX IF NOT EXISTS idx_scorer_step_correction_run ON scorer_step_correction(run_id);
CREATE INDEX IF NOT EXISTS idx_scorer_step_correction_status ON scorer_step_correction(status);
CREATE INDEX IF NOT EXISTS idx_scorer_step_correction_parent_correction ON scorer_step_correction(parent_correction_id);
CREATE INDEX IF NOT EXISTS idx_scorer_step_correction_child_run ON scorer_step_correction(child_run_id);
CREATE INDEX IF NOT EXISTS idx_scorer_step_correction_repair_source_dump ON scorer_step_correction(repair_source_dump_id);

-- ─── Metrics ───────────────────────────────────────────────────────────
CREATE SEQUENCE IF NOT EXISTS metric_id_seq;
CREATE TABLE IF NOT EXISTS metric (
    metric_id     BIGINT PRIMARY KEY DEFAULT nextval('metric_id_seq'),
    run_id        VARCHAR NOT NULL,               -- FK score_run.run_id
    truth_set_id  VARCHAR,                        -- nullable: no-truth-mode metrics omit it
    metric_name   VARCHAR NOT NULL,               -- e.g., 'indra_belief_calibration', 'parse_claim_precision'
    value         DOUBLE NOT NULL,
    slice_json    JSON,                           -- e.g., {{"step": "grounding", "source_api": "reach"}}
    computed_at   TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_metric_run ON metric(run_id);
CREATE INDEX IF NOT EXISTS idx_metric_truthset ON metric(truth_set_id);
CREATE INDEX IF NOT EXISTS idx_metric_natural ON metric(run_id, metric_name);

-- ─── Schema version tracking ───────────────────────────────────────────
CREATE TABLE IF NOT EXISTS schema_meta (
    key    VARCHAR PRIMARY KEY,
    value  VARCHAR NOT NULL
);
"""


def apply_schema(con: "duckdb.DuckDBPyConnection") -> None:
    """Apply the corpus schema to a DuckDB connection (idempotent).

    Records `SCHEMA_VERSION` in the `schema_meta` table. Existing tables
    are preserved (CREATE IF NOT EXISTS). Future migrations append new
    tables / columns rather than rewriting; that's the append-only
    contract Phase 2.6 will enforce.
    """
    # Old DuckDB files may already have `score_run` / `scorer_step` without
    # newly added columns. Add those columns before running `_DDL`, because
    # `_DDL` also creates indexes on the new columns.
    _apply_additive_migrations(con, backfill=False)
    con.execute(_DDL)
    _apply_additive_migrations(con, backfill=True)
    con.execute(
        "INSERT OR REPLACE INTO schema_meta (key, value) VALUES ('schema_version', ?)",
        [str(SCHEMA_VERSION)],
    )
    # The janitor writes to score_run, so a READ_ONLY connection (e.g., a
    # future viewer/admin script that opens read-only and still calls
    # apply_schema by mistake) would crash here. Log and continue rather
    # than blocking apply_schema for the whole process.
    try:
        reconcile_stale_running_runs(con)
    except Exception as e:
        log.warning("startup janitor skipped: %s", e)
    log.info("corpus schema applied (version=%s)", SCHEMA_VERSION)


# A score_run that has been in 'running' status for longer than this threshold
# is assumed to be from a process that crashed/was killed between worker spawn
# and tombstone. Default 48h: the full Rasmachine corpus monolithic-scoring
# at 27B can run 6-12h, and a paused-laptop scenario can extend a live run
# further; 48h is the conservative ceiling. Override via the
# `INDRA_STALE_RUN_HOURS` env var (used by `apply_schema()`) for ops with
# shorter SLAs. A future heartbeat column will replace this with precise
# liveness detection.
def _stale_running_threshold_hours_default() -> int:
    raw = os.environ.get("INDRA_STALE_RUN_HOURS", "48")
    try:
        parsed = int(raw)
        return max(1, parsed)
    except ValueError:
        return 48


STALE_RUNNING_THRESHOLD_HOURS = _stale_running_threshold_hours_default()


def reconcile_stale_running_runs(
    con: "duckdb.DuckDBPyConnection",
    *,
    threshold_hours: int = STALE_RUNNING_THRESHOLD_HOURS,
) -> list[str]:
    """Tombstone score_run rows stuck in 'running' from a crashed prior worker.

    Returns the list of run_ids that were tombstoned to `'crashed_at_startup'`.
    A row is considered stale when its `started_at` is older than
    `threshold_hours` and its status is still `'running'` — at that point we
    assume the worker that owned the row did not survive to flip the status,
    e.g. the host rebooted between worker spawn and tombstone.

    Safety: a live run with `started_at` within the threshold is never touched.
    The threshold needs to be wider than the largest plausible scoring time on
    the largest corpus; bump `STALE_RUNNING_THRESHOLD_HOURS` if a real run
    legitimately exceeds the default. A future Phase will add a heartbeat
    column so the janitor can detect crashes precisely instead of by age.
    """
    if not _table_exists(con, "score_run"):
        return []
    candidate_rows = con.execute(
        f"""
        SELECT run_id
          FROM score_run
         WHERE status = 'running'
           AND started_at IS NOT NULL
           AND started_at < CURRENT_TIMESTAMP - INTERVAL '{int(threshold_hours)} hours'
        """
    ).fetchall()
    if not candidate_rows:
        return []
    run_ids = [str(row[0]) for row in candidate_rows]
    placeholders = ",".join(["?"] * len(run_ids))
    con.execute(
        f"""
        UPDATE score_run
           SET status = 'crashed_at_startup',
               finished_at = COALESCE(finished_at, CURRENT_TIMESTAMP),
               terminated_by = COALESCE(terminated_by, 'janitor'),
               termination_reason = COALESCE(
                 termination_reason,
                 'startup janitor: row was in running state for more than '
                   || ? || ' hours; assuming the owning worker process did not survive'
               )
         WHERE run_id IN ({placeholders})
        """,
        [str(threshold_hours), *run_ids],
    )
    log.warning(
        "startup janitor tombstoned %d stale running run(s): %s",
        len(run_ids),
        ", ".join(run_ids[:8]) + ("…" if len(run_ids) > 8 else ""),
    )
    return run_ids


def _apply_additive_migrations(
    con: "duckdb.DuckDBPyConnection",
    *,
    backfill: bool = True,
) -> None:
    """Apply append-only schema additions for existing DuckDB files.

    `CREATE TABLE IF NOT EXISTS` does not add columns to an already-created
    table. The viewer and worker routinely reopen long-lived corpus files, so
    every new column must be backfilled here as well as declared in `_DDL`.
    """
    additions = (
        ("score_run", "architecture", "VARCHAR DEFAULT 'unknown'"),
        ("score_run", "paired_run_group_id", "VARCHAR"),
        ("score_run", "parent_run_id", "VARCHAR"),
        ("score_run", "reviewed_at", "TIMESTAMP"),
        ("score_run", "reviewed_by", "VARCHAR"),
        ("score_run", "review_status", "VARCHAR"),
        ("score_run", "review_notes", "TEXT"),
        ("score_run", "terminated_by", "VARCHAR"),
        ("score_run", "termination_reason", "TEXT"),
        ("score_run", "repair_kind", "VARCHAR"),
        ("scorer_step", "architecture", "VARCHAR DEFAULT 'decomposed'"),
        ("scorer_step_correction", "architecture", "VARCHAR DEFAULT 'unknown'"),
        ("scorer_step_correction", "parent_correction_id", "BIGINT"),
        ("scorer_step_correction", "child_run_id", "VARCHAR"),
        ("scorer_step_correction", "repair_source_dump_id", "VARCHAR"),
    )
    for table, column, ddl in additions:
        if not _table_exists(con, table):
            continue
        try:
            con.execute(f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {column} {ddl}")
        except Exception as e:
            log.warning("schema migration add %s.%s failed: %s", table, column, e)

    if not backfill:
        return
    if _table_exists(con, "scorer_step_correction"):
        flag = con.execute(
            "SELECT value FROM schema_meta WHERE key = 'repair_lineage_backfill_v1'"
        ).fetchone()
        if not (flag and flag[0] == "done"):
            try:
                con.execute(
                    """
                    UPDATE scorer_step_correction
                       SET parent_correction_id = COALESCE(
                             parent_correction_id,
                             TRY_CAST(json_extract_string(value_json, '$.parent_correction_id') AS BIGINT)
                           ),
                           child_run_id = COALESCE(
                             NULLIF(child_run_id, ''),
                             json_extract_string(value_json, '$.child_run_id')
                           ),
                           repair_source_dump_id = COALESCE(
                             NULLIF(repair_source_dump_id, ''),
                             json_extract_string(value_json, '$.source_dump_id')
                           )
                     WHERE value_json IS NOT NULL
                       AND (
                         (
                           parent_correction_id IS NULL
                           AND json_extract_string(value_json, '$.parent_correction_id') IS NOT NULL
                         )
                         OR (
                           (child_run_id IS NULL OR child_run_id = '')
                           AND json_extract_string(value_json, '$.child_run_id') IS NOT NULL
                         )
                         OR (
                           (repair_source_dump_id IS NULL OR repair_source_dump_id = '')
                           AND json_extract_string(value_json, '$.source_dump_id') IS NOT NULL
                         )
                       )
                    """
                )
                con.execute(
                    "INSERT OR REPLACE INTO schema_meta (key, value) VALUES "
                    "('repair_lineage_backfill_v1', 'done')"
                )
            except Exception as e:
                log.warning("schema correction repair lineage backfill failed: %s", e)

    if _table_exists(con, "scorer_step_correction") and _table_exists(con, "scorer_step"):
        try:
            con.execute(
                """
                UPDATE scorer_step_correction AS c
                   SET architecture = COALESCE(NULLIF(ss.architecture, ''), c.architecture, 'unknown')
                  FROM scorer_step AS ss
                 WHERE c.step_hash = ss.step_hash
                   AND (c.architecture IS NULL OR c.architecture = '' OR c.architecture = 'unknown')
                """
            )
        except Exception as e:
            log.warning("schema correction architecture backfill failed: %s", e)

    if _table_exists(con, "statement_evidence") and _table_exists(con, "evidence"):
        _run_backfill_with_retry(
            con,
            flag_key="statement_evidence_backfill_v1",
            statements=[_statement_evidence_backfill_sql(con)],
            label="statement_evidence backfill",
        )
        _drop_legacy_evidence_stmt_hash(con)

    if not (
        _table_exists(con, "schema_meta")
        and _table_exists(con, "score_run")
        and _table_exists(con, "scorer_step")
    ):
        return
    flag = con.execute(
        "SELECT value FROM schema_meta WHERE key = 'architecture_backfill_v2'"
    ).fetchone()
    if flag and flag[0] == "done":
        return

    # Existing rows were produced by the decomposed viewer/worker path unless
    # an old experiment encoded monolithic in scorer_version/notes. Keep the
    # backfill conservative and visible: monolithic-looking rows are marked
    # monolithic; otherwise decomposed rows with scorer_step evidence are
    # decomposed; empty legacy rows remain unknown.
    try:
        con.execute(
            """
            UPDATE score_run
               SET architecture = CASE
                   WHEN lower(COALESCE(scorer_version, '')) LIKE '%mono%'
                     OR lower(COALESCE(notes, '')) LIKE '%monolithic%'
                     THEN 'monolithic'
                   WHEN run_id IN (SELECT DISTINCT run_id FROM scorer_step)
                     THEN 'decomposed'
                   ELSE COALESCE(NULLIF(architecture, ''), 'unknown')
               END
             WHERE architecture IS NULL OR architecture = '' OR architecture = 'unknown'
            """
        )
        con.execute(
            """
            UPDATE scorer_step AS ss
               SET architecture = COALESCE(NULLIF(sr.architecture, ''), 'decomposed')
              FROM score_run AS sr
             WHERE ss.run_id = sr.run_id
               AND (ss.architecture IS NULL OR ss.architecture = '' OR ss.architecture = 'decomposed')
            """
        )
        con.execute(
            "INSERT OR REPLACE INTO schema_meta (key, value) VALUES "
            "('architecture_backfill_v2', 'done')"
        )
    except Exception as e:
        log.warning("schema architecture backfill failed: %s", e)


_BACKFILL_MAX_ATTEMPTS = 3


def _column_exists(
    con: "duckdb.DuckDBPyConnection", table: str, column: str
) -> bool:
    row = con.execute(
        """
        SELECT COUNT(*)
          FROM information_schema.columns
         WHERE table_name = ? AND column_name = ?
        """,
        [table, column],
    ).fetchone()
    return bool(row and row[0])


def _statement_evidence_backfill_sql(con: "duckdb.DuckDBPyConnection") -> str:
    """Choose the legacy or stmt_hash-less source for the backfill.

    Until `evidence_stmt_hash_drop_v1` runs, `evidence.stmt_hash` is the
    only place legacy membership lives, so it must be the backfill source.
    After the drop migration, the backfill flag is already `done` and this
    function is not exercised on the live path; the safe variant returns an
    empty insert in that case so a wiped-flag scenario does not crash.
    """
    if _column_exists(con, "evidence", "stmt_hash"):
        return """
            INSERT OR REPLACE INTO statement_evidence
                (stmt_hash, evidence_hash, evidence_index, source_dump_id)
            SELECT e.stmt_hash,
                   e.evidence_hash,
                   ROW_NUMBER() OVER (
                     PARTITION BY e.stmt_hash ORDER BY e.evidence_hash
                   ) - 1 AS evidence_index,
                   s.source_dump_id
              FROM evidence e
              LEFT JOIN statement s ON s.stmt_hash = e.stmt_hash
             WHERE e.stmt_hash IS NOT NULL
               AND e.evidence_hash IS NOT NULL
            """
    # Legacy column already dropped — nothing to re-derive. Membership lives
    # entirely in statement_evidence at this point.
    return "SELECT 1 WHERE FALSE"


def _attempt_counter_key(flag_key: str) -> str:
    return f"{flag_key}_attempts"


def _run_backfill_with_retry(
    con: "duckdb.DuckDBPyConnection",
    *,
    flag_key: str,
    statements: list[str] | tuple[str, ...],
    label: str,
    use_transaction: bool = True,
) -> None:
    """Run one or more backfill statements with a retry counter.

    When `use_transaction=True` (default) the statements + flag write commit
    atomically. Set `use_transaction=False` for DDL sequences that DuckDB
    refuses to evaluate inside a transaction (e.g., DROP INDEX followed by
    ALTER TABLE DROP COLUMN — the dropped index is not visible to the column
    drop's constraint check until commit, so the column drop fails).
    A failure increments `flag_key_attempts`; after `_BACKFILL_MAX_ATTEMPTS`
    the helper writes a distinct `'failed'` flag and stops retrying so a
    deterministic poison-failure surfaces to operators instead of looping
    forever.
    """
    flag_row = con.execute(
        "SELECT value FROM schema_meta WHERE key = ?", [flag_key]
    ).fetchone()
    flag_val = flag_row[0] if flag_row else None
    if flag_val == "done" or flag_val == "failed":
        return
    attempts_row = con.execute(
        "SELECT value FROM schema_meta WHERE key = ?",
        [_attempt_counter_key(flag_key)],
    ).fetchone()
    try:
        attempts = int(attempts_row[0]) if attempts_row and attempts_row[0] else 0
    except (TypeError, ValueError):
        attempts = 0
    if attempts >= _BACKFILL_MAX_ATTEMPTS:
        try:
            con.execute(
                "INSERT OR REPLACE INTO schema_meta (key, value) VALUES (?, ?)",
                [flag_key, "failed"],
            )
        except Exception as e:
            log.warning("%s: failed to set 'failed' marker: %s", label, e)
        log.warning(
            "%s: giving up after %d attempts; set %s='failed'",
            label,
            attempts,
            flag_key,
        )
        return
    try:
        if use_transaction:
            con.execute("BEGIN")
        for stmt in statements:
            stripped = stmt.strip()
            if stripped:
                con.execute(stripped)
        con.execute(
            "INSERT OR REPLACE INTO schema_meta (key, value) VALUES (?, ?)",
            [flag_key, "done"],
        )
        if use_transaction:
            con.execute("COMMIT")
    except Exception as e:
        if use_transaction:
            try:
                con.execute("ROLLBACK")
            except Exception:
                pass
        try:
            con.execute(
                "INSERT OR REPLACE INTO schema_meta (key, value) VALUES (?, ?)",
                [_attempt_counter_key(flag_key), str(attempts + 1)],
            )
        except Exception as bump_err:
            log.warning(
                "%s: failed to bump attempt counter to %d: %s",
                label,
                attempts + 1,
                bump_err,
            )
        log.warning(
            "%s: failed (attempt %d/%d): %s",
            label,
            attempts + 1,
            _BACKFILL_MAX_ATTEMPTS,
            e,
        )


def _drop_legacy_evidence_stmt_hash(con: "duckdb.DuckDBPyConnection") -> None:
    """Drop the lossy `evidence.stmt_hash` column once membership migration is safe.

    Safety: the column is only dropped if statement_evidence has at least one
    row for every distinct evidence.stmt_hash; otherwise membership would be
    lost. The flag is `evidence_stmt_hash_drop_v1`. Wrapped in the same
    transactional retry as other backfills.
    """
    if not _column_exists(con, "evidence", "stmt_hash"):
        return
    flag_row = con.execute(
        "SELECT value FROM schema_meta WHERE key = 'evidence_stmt_hash_drop_v1'"
    ).fetchone()
    if flag_row and flag_row[0] in ("done", "failed"):
        return
    # Membership-completeness check: every legacy (evidence_hash, stmt_hash)
    # must have a corresponding statement_evidence row. If any are missing
    # the drop is unsafe.
    missing = con.execute(
        """
        SELECT COUNT(*)
          FROM evidence e
         WHERE e.stmt_hash IS NOT NULL
           AND e.evidence_hash IS NOT NULL
           AND NOT EXISTS (
             SELECT 1 FROM statement_evidence se
              WHERE se.evidence_hash = e.evidence_hash
                AND se.stmt_hash    = e.stmt_hash
           )
        """
    ).fetchone()
    if missing and missing[0]:
        log.warning(
            "evidence_stmt_hash_drop_v1: %d legacy rows missing in "
            "statement_evidence; refusing to drop column",
            missing[0],
        )
        return
    # DuckDB's ALTER TABLE DROP COLUMN refuses if any index references a column
    # positioned *after* the one being dropped. Drop all evidence indexes first,
    # drop the column, then recreate the indexes we still want on the new
    # column layout. This must run *outside* a transaction: a dropped index is
    # not visible to ALTER TABLE's constraint check until commit, so the drop
    # would otherwise see the stale index and refuse the column drop.
    _run_backfill_with_retry(
        con,
        flag_key="evidence_stmt_hash_drop_v1",
        statements=[
            "DROP INDEX IF EXISTS idx_evidence_stmt",
            "DROP INDEX IF EXISTS idx_evidence_source_api",
            "DROP INDEX IF EXISTS idx_evidence_pmid",
            "ALTER TABLE evidence DROP COLUMN stmt_hash",
            "CREATE INDEX IF NOT EXISTS idx_evidence_source_api ON evidence(source_api)",
            "CREATE INDEX IF NOT EXISTS idx_evidence_pmid ON evidence(pmid)",
        ],
        label="evidence.stmt_hash column drop",
        use_transaction=False,
    )


def _table_exists(con: "duckdb.DuckDBPyConnection", table: str) -> bool:
    row = con.execute(
        """
        SELECT COUNT(*)
          FROM information_schema.tables
         WHERE table_name = ?
        """,
        [table],
    ).fetchone()
    return bool(row and row[0])


def schema_ddl() -> str:
    """Return the canonical DDL string. Useful for inspection / dumping."""
    return _DDL
