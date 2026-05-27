/**
 * DuckDB connection helper.
 *
 * The viewer reads from a single .duckdb file produced by the corpus loader.
 * Path resolves from the VIEWER_DUCKDB_PATH env var, falling back to a default
 * inside the project's data/ folder.
 */

import { DuckDBInstance, type DuckDBConnection } from '@duckdb/node-api';
import { existsSync, statSync } from 'node:fs';
import { resolve } from 'node:path';
import { error } from '@sveltejs/kit';
import {
	computeAttributions,
	summarizeAcrossEvidences,
	type ProbeAttribution,
	type ProbeKind,
	type ProbeOutput,
	type ProbeConfidence,
	type ProbeSource
} from './probeAttribution';
import type { PairedLedgerRole, PairedMetricKind, PanelApplicabilityKind } from './pairedMetricKinds';
import {
	cleanTraceSnapshot,
	cleanTraceStateFilter,
	INVALID_TRACE_STATE,
	INVALID_TRACE_SNAPSHOT,
	isTraceFidelityState,
	zeroTraceCounts,
	type TraceFidelityFilter,
	type TraceFidelityState
} from './traceState';
import { traceEvidenceCteForRun } from './traceStateSql';
import {
	REPAIR_RERUN_QUEUED_INTENT_LOCK_MINUTES,
	REPAIR_RERUN_TERMINAL_STATUS_SQL,
	repairCandidateAvailablePredicateSql,
	repairRerunChildRunIdSql,
	repairRerunParentCorrectionIdNumberSql,
	repairRerunParentCorrectionIdStringSql,
	repairRerunSourceDumpIdSql,
	type RepairRerunLineageSqlOptions
} from './repairRerunSql';
import {
	nonTraceEvidenceCohortWhereClauses,
	nonTraceStatementCohortWhereClauses,
	RUN_COHORT_LIMIT,
	traceStateCohortWhereClauses
} from './cohorts/sql';
import {
	aggregateEvidenceCohortBaseSql,
	aggregateEvidenceCohortCountSql,
	aggregateEvidenceCohortRowsSql,
	statementCohortCte,
	statementCohortRowsSql,
	traceCohortBaseSql,
	traceCohortCountSql,
	traceCohortCte,
	traceCohortRowsSql
} from './cohorts/queries';
import {
	statementEvidenceCountSql,
	statementEvidenceSourceStratumSql
} from './cohorts/evidenceMembership';
import {
	aggregateEvidenceEmptyPredicates,
	emptyCohortDiagnostics,
	runScorerStepCounts,
	statementEmptyPredicates,
	traceEvidenceEmptyPredicates,
	type RunCohortEmptyDiagnostic
} from './cohorts/diagnostics';
import { validateRunCohortFilterValues } from './server/runCohortContract';
import type { RunCohortFilters } from './cohorts/types';

export type { ProbeAttribution, ProbeKind, ProbeOutput } from './probeAttribution';
export type { TraceFidelityState } from './traceState';
export type { RunCohortFilters } from './cohorts/types';
export type { RunCohortEmptyDiagnostic } from './cohorts/diagnostics';
export type {
	PairedLedgerRole,
	PairedMetricKind,
	PairedMetricKindFamily,
	PanelApplicabilityKind
} from './pairedMetricKinds';

const DEFAULT_DB_PATH = resolve(
	process.cwd(),
	'..',
	'data',
	'corpus.duckdb'
);

let _instance: DuckDBInstance | null = null;
let _resolvedPath = '';
let _instanceMtimeMs = 0;
let _instanceCtimeMs = 0;
let _instanceSize = 0;
const DUCKDB_LOCK_FRAGMENTS = ['Could not set lock', 'Conflicting lock'] as const;

// Serialized/refcounted DuckDB connection manager. Closes the cycle 51-vintage
// race: two concurrent `connect()` calls both saw `_instance == null`, both
// called `DuckDBInstance.create()`, one win + one leak; the leaked instance
// kept an OS-level read lock so writer-spawn endpoints got 503 storms.
//
// _instanceCreationPromise serializes acquisition: in-flight `connect()` calls
// share the same create() promise. _activeReaderCount is a refcount of open
// reader connections; on close we wait for it to drain (bounded by
// CLOSE_INSTANCE_DRAIN_TIMEOUT_MS) before destroying the instance.
// _closePending blocks new connects from landing on a soon-to-be-destroyed
// handle. We use a counter (not a Set) so a caller that forgets to call
// `disconnectSync` does not hard-reference the connection object and block GC.
let _instanceCreationPromise: Promise<DuckDBInstance> | null = null;
let _activeReaderCount = 0;
let _closePending = false;
let _closeWaiters: Array<() => void> = [];

// C2 of deferred hypergraph: drainer ceiling. The forced-close path
// after this timeout can destroy in-flight reader handles. Operators
// with slow-corpus reads can extend the timeout via the
// `INDRA_VIEWER_CLOSE_DRAIN_MS` env var; otherwise the default 5 s
// applies. After timeout the drainer logs a warning naming the leaked
// reader count and proceeds with closeSync — readers will surface a
// destroyed-handle error on their next read, which the wrapper below
// re-throws as a typed `reader_lease_expired` for clean UI handling.
const CLOSE_INSTANCE_DRAIN_TIMEOUT_MS = (() => {
	const raw = process.env.INDRA_VIEWER_CLOSE_DRAIN_MS;
	if (!raw) return 5_000;
	const parsed = Number.parseInt(raw, 10);
	return Number.isFinite(parsed) && parsed > 0 ? parsed : 5_000;
})();
const CLOSE_INSTANCE_POLL_MS = 25;

/**
 * Thrown by reads that fire against a DuckDB instance the connection
 * manager force-closed. SSR routes should catch and surface as 503.
 */
export class ReaderLeaseExpiredError extends Error {
	readonly code = 'reader_lease_expired';
	constructor(originalMessage: string) {
		super(`reader_lease_expired: ${originalMessage}`);
		this.name = 'ReaderLeaseExpiredError';
	}
}

function delay(ms: number): Promise<void> {
	return new Promise((r) => setTimeout(r, ms));
}

function notifyCloseDone(): void {
	const waiters = _closeWaiters;
	_closeWaiters = [];
	for (const w of waiters) {
		try {
			w();
		} catch {
			// waiter callbacks must not throw into here
		}
	}
}

function registerConnection(con: DuckDBConnection): DuckDBConnection {
	_activeReaderCount += 1;
	const proto = Object.getPrototypeOf(con);
	const originalDisconnect = (con as unknown as { disconnectSync?: () => void }).disconnectSync
		?? proto?.disconnectSync;
	let released = false;
	const release = () => {
		if (released) return;
		released = true;
		// Floor at 0 so a stray double-call doesn't poison the count.
		_activeReaderCount = Math.max(0, _activeReaderCount - 1);
	};
	if (typeof originalDisconnect === 'function') {
		// C7 of deferred hypergraph: patch lives on the instance (own
		// property) AND falls back to the prototype's disconnectSync if
		// the binding closes through an internal path that doesn't go
		// through the instance method we patched. The release() is
		// idempotent (released flag + Math.max floor) so a binding
		// internal path that bypasses the patch is safe — it leaves the
		// counter pinned, but the next legitimate disconnectSync call
		// (or a forced closeInstance drain after timeout) releases it.
		(con as unknown as { disconnectSync: () => void }).disconnectSync = function patchedDisconnect() {
			release();
			try {
				originalDisconnect.call(this);
			} catch {
				// best-effort — handle may already be gone
			}
		};
		// Also schedule a FinalizationRegistry cleanup so a connection
		// that gets garbage-collected without an explicit disconnect
		// (caller forgot in an exception path) still releases the
		// counter. Best-effort — Node does not guarantee finalizer
		// timing, but worst case is identical to the no-finalizer
		// version (the slot eventually expires through closeInstance's
		// drain timeout).
		try {
			_connectionFinalizers.register(con, release);
		} catch {
			// FinalizationRegistry not available in some runtimes
		}
	} else {
		// Without disconnectSync we can't observe close, so don't pin the
		// counter; assume the caller's lifecycle is short and release now.
		release();
	}
	return con;
}

// Single shared FinalizationRegistry so connections garbage-collected
// without explicit `disconnectSync()` still release the refcount.
const _connectionFinalizers = new FinalizationRegistry<() => void>((heldValue) => {
	try {
		heldValue();
	} catch {
		// finalizer callbacks must not throw
	}
});

export function dbPath(): string {
	if (_resolvedPath) return _resolvedPath;
	const env = process.env.VIEWER_DUCKDB_PATH;
	_resolvedPath = env ? resolve(env) : DEFAULT_DB_PATH;
	return _resolvedPath;
}

export function dbExists(): boolean {
	return existsSync(dbPath());
}

export async function connect(): Promise<DuckDBConnection> {
	const path = dbPath();
	// File-stat invalidation: when the Python loader replaces / rewrites
	// the .duckdb file, our cached READ_ONLY instance was mmap'd against
	// the old contents and serves stale data. Re-instantiate when the
	// file signature changes (or the file was created since the cached
	// instance opened).
	let currentMtime = 0;
	let currentCtime = 0;
	let currentSize = 0;
	let statAvailable = false;
	try {
		const stat = statSync(path);
		currentMtime = stat.mtimeMs;
		currentCtime = stat.ctimeMs;
		currentSize = stat.size;
		statAvailable = true;
	} catch {
		// File may not exist yet
	}

	// If a close is in flight, wait for it; otherwise new connects could land
	// on a soon-to-be-destroyed instance.
	while (_closePending) {
		await new Promise<void>((r) => _closeWaiters.push(r));
	}

	const staleByFileStat =
		_instance != null &&
		statAvailable &&
		(currentMtime !== _instanceMtimeMs ||
			currentCtime !== _instanceCtimeMs ||
			currentSize !== _instanceSize);
	if (staleByFileStat) {
		// closeInstance() is the serialized path that handles in-flight readers.
		// Bypass it only when our cached instance is provably stale and no in-
		// flight create is running; otherwise we'd leak by recreating without
		// waiting.
		await closeInstance();
	}

	if (!_instance && !_instanceCreationPromise) {
		// READ_ONLY mode. DuckDB's file-level lock is held at the *instance*
		// level: a process holding any open instance (even READ_ONLY) blocks
		// another process from opening the same file for writing. Endpoints
		// that spawn a Python writer (ingest / score) MUST call
		// closeInstance() before spawning so the worker can acquire its lock.
		// The next connect() lazy-reopens. Dashboard reads issued while a
		// writer holds the lock surface as a typed 503 (caught by
		// +error.svelte and rendered as a friendly waiting screen) rather
		// than a raw 500 with a DuckDB stack trace.
		_instanceCreationPromise = (async () => {
			try {
				const inst = await DuckDBInstance.create(path, { access_mode: 'READ_ONLY' });
				// C6 of deferred hypergraph: re-stat AFTER the file is open
				// so a concurrent rewrite that landed during our initial
				// stat-then-create window invalidates the cached signature.
				// The next connect() will re-detect stale-by-stat and
				// re-open against the actually-current contents.
				try {
					const restat = statSync(path);
					currentMtime = restat.mtimeMs;
					currentCtime = restat.ctimeMs;
					currentSize = restat.size;
				} catch {
					// File may have been removed between create() and re-stat;
					// surface the original create's contents as-is.
				}
				_instance = inst;
				_instanceMtimeMs = currentMtime;
				_instanceCtimeMs = currentCtime;
				_instanceSize = currentSize;
				return inst;
			} finally {
				_instanceCreationPromise = null;
			}
		})();
	}

	try {
		const inst = await (_instanceCreationPromise ?? Promise.resolve(_instance as DuckDBInstance));
		const con = await inst.connect();
		return registerConnection(con);
	} catch (e) {
		const msg = (e as Error)?.message ?? String(e);
		if (DUCKDB_LOCK_FRAGMENTS.some((fragment) => msg.includes(fragment))) {
			throw error(
				503,
				{
					code: 'writer_in_progress',
					message:
						'writer_in_progress: an ingest or score worker is holding the DuckDB write lock. Wait for it to finish, then reload.'
				}
			);
		}
		throw e;
	}
}

/**
 * Release the cached READ_ONLY instance so a Python writer subprocess can
 * acquire the file lock. Drains in-flight reader connections first (bounded
 * by `CLOSE_INSTANCE_DRAIN_TIMEOUT_MS`); after the timeout it closes anyway
 * — leaking is worse than blocking writer spawn forever.
 */
export async function closeInstance(): Promise<void> {
	if (!_instance && !_instanceCreationPromise) return;
	_closePending = true;
	try {
		// Wait for any in-flight create() to finish so we have a real handle
		// to close (and we don't race the assignment in the create promise).
		if (_instanceCreationPromise) {
			try {
				await _instanceCreationPromise;
			} catch {
				// Creation failed; nothing to close.
			}
		}
		// Snapshot the instance now so a concurrent closeInstance() racing
		// with a fresh `connect()` after our drain finishes does not
		// closeSync() the NEW instance someone else just opened.
		const targetInstance = _instance;
		const deadline = Date.now() + CLOSE_INSTANCE_DRAIN_TIMEOUT_MS;
		while (_activeReaderCount > 0 && Date.now() < deadline) {
			await delay(CLOSE_INSTANCE_POLL_MS);
		}
		if (_activeReaderCount > 0) {
			console.warn(
				`closeInstance: forced close after ${CLOSE_INSTANCE_DRAIN_TIMEOUT_MS}ms ` +
					`with ${_activeReaderCount} reader(s) still active; ` +
					`their next read may surface a destroyed-handle error`
			);
		}
		if (targetInstance) {
			try {
				(targetInstance as unknown as { closeSync?: () => void }).closeSync?.();
			} catch {
				// best-effort — instance may already be closed
			}
		}
		// Only reset shared state if our target is still the active one. A
		// concurrent close that already swapped in a new instance must keep
		// its bookkeeping intact.
		if (_instance === targetInstance) {
			_instance = null;
			_instanceMtimeMs = 0;
			_instanceCtimeMs = 0;
			_instanceSize = 0;
			_activeReaderCount = 0;
		}
	} finally {
		_closePending = false;
		notifyCloseDone();
	}
}

export interface CorpusOverview {
	dbPath: string;
	dbExists: boolean;
	statementCount: number;
	evidenceCount: number;
	agentCount: number;
	supportsEdgeCount: number;
	truthLabelCount: number;
	truthSets: Array<{ id: string; name: string; rowCount: number }>;
	sourceDumps: Array<{ source_dump_id: string | null; n: number }>;
	indraTypes: Array<{ indra_type: string; n: number }>;
	scorerRuns: Array<{
		run_id: string;
		scorer_version: string;
		architecture: string;
		paired_run_group_id: string | null;
		started_at: string;
		status: string;
		terminated_by: string | null;
		termination_reason: string | null;
		n_stmts: number | null;
		cost_estimate_usd: number | null;
		mae: number | null;
		bias: number | null;
		hasIndraExport: boolean;
		hasCardExport: boolean;
	}>;
	latestValidity: LatestValidity | null;
}

export interface LatestValidity {
	run_id: string;
	scorer_version: string;
	architecture: string;
	verdicts: Array<{ verdict: string; n: number; cohort_href: string }>;
	calibration: {
		mae: number | null;
		rmse: number | null;
		bias: number | null;
		n_stmts: number | null;
		cohort_href: string;
		applicability: PanelApplicabilityKind;
	};
	inter_evidence_consistency: {
		mean_stdev: number | null;
		n_multi_ev: number | null;
		cohort_href: string | null;
		applicability: PanelApplicabilityKind;
		not_defined_reason: string | null;
	};
	supports_graph_delta: number | null;
	supports_graph_href: string | null;
	supports_graph_not_defined_reason: string | null;
	byIndraType: StratumRow[];
	bySourceApi: StratumRow[];
	truthPresent: TruthPresentRow[];
	confidenceCalibration: ConfidenceCalibrationRow[];
	metricTaxonomy: PanelApplicabilityRow[];
}

export interface TruthPresentRow {
	truth_set_id: string;
	step_kind: string;
	precision: number | null;
	recall: number | null;
	f1: number | null;
	n_compared: number;
	tp: number;
	fp: number;
	fn: number;
	tn: number;
	n_gold_labels: number | null;
	n_applicable_gold_labels: number | null;
	n_scored_evidences: number | null;
	gold_fields: string[];
	positive_gold_label: string;
	negative_gold_rule: string;
	unavailable_reason: string | null;
	cohort_href: string;
	truth_href: string;
	applicability: PanelApplicabilityKind;
}

export type TruthSetOverlapStatus =
	| 'compared'
	| 'no_aggregate_verdict'
	| 'context_mismatch'
	| 'not_scored_in_run'
	| 'not_in_corpus';

export interface TruthSetOverlapRow {
	label_id: number;
	target_id: string;
	relation_target_id: string | null;
	field: string;
	value_text: string | null;
	provenance: string | null;
	corpus_stmt_hash: string | null;
	source_api: string | null;
	evidence_text: string | null;
	same_evidence_scored_rows: number;
	matched_step_rows: number;
	matched_verdict_rows: number;
	matched_no_verdict_rows: number;
	sample_matched_stmt_hash: string | null;
	sample_our_verdict: string | null;
	sample_scored_stmt_hash: string | null;
	status: TruthSetOverlapStatus;
	statement_href: string | null;
}

export interface TruthSetOverlapDetail {
	run_id: string;
	truth_set_id: string;
	truth_set_name: string;
	architecture: string;
	status: string;
	metric_step_kind: string;
	n_gold_labels: number;
	n_applicable_gold_labels: number;
	n_metric_compared_rows: number;
	n_scored_evidences: number;
	n_compared_labels: number;
	n_no_aggregate_verdict: number;
	n_context_mismatch: number;
	n_not_scored_in_run: number;
	n_not_in_corpus: number;
	positive_gold_label: string;
	negative_gold_rule: string;
	gold_fields: string[];
	cohort_href: string | null;
	rows: TruthSetOverlapRow[];
	row_limit: number;
	status_row_limit: number;
}

export const TRUTH_SET_OVERLAP_ROW_LIMIT = 200;
export const TRUTH_SET_OVERLAP_ROWS_PER_STATUS = 40;

export interface StratumRow {
	value: string;
	n: number;
	mae: number;
	bias: number;
	cohort_href: string;
	applicability: PanelApplicabilityKind;
}

export interface ConfidenceCalibrationRow {
	family: string;
	confidence: string;
	n: number;
	mean_score: number | null;
	mean_indra_belief: number | null;
	mae: number | null;
	bias: number | null;
	cohort_href: string;
	applicability: PanelApplicabilityKind;
}

export const EMPTY_OVERVIEW: Omit<CorpusOverview, 'dbPath' | 'dbExists'> = {
	statementCount: 0,
	evidenceCount: 0,
	agentCount: 0,
	supportsEdgeCount: 0,
	truthLabelCount: 0,
	truthSets: [],
	sourceDumps: [],
	indraTypes: [],
	scorerRuns: [],
	latestValidity: null
};

export const CORE_OVERVIEW_REQUIRED_COLUMNS: Record<string, string[]> = {
	statement: ['stmt_hash', 'indra_type', 'indra_belief', 'source_dump_id'],
	evidence: ['evidence_hash', 'source_api'],
	statement_evidence: ['stmt_hash', 'evidence_hash'],
	agent: ['stmt_hash', 'agent_hash', 'role', 'name'],
	supports_edge: ['from_stmt_hash', 'to_stmt_hash', 'kind'],
	truth_set: ['id', 'name'],
	truth_label: ['label_id', 'truth_set_id', 'target_kind', 'target_id', 'field'],
	scorer_step: ['run_id', 'stmt_hash', 'evidence_hash', 'step_kind', 'output_json'],
	score_run: [
		'run_id',
		'scorer_version',
		'architecture',
		'paired_run_group_id',
		'started_at',
		'status',
		'terminated_by',
		'termination_reason',
		'n_stmts',
		'cost_estimate_usd'
	],
	metric: ['run_id', 'truth_set_id', 'metric_name', 'value', 'slice_json']
};

export function cohortHref(
	run_id: string,
	params: Record<string, string | number | boolean | null | undefined> = {}
): string {
	const sp = new URLSearchParams();
	for (const [k, v] of Object.entries(params)) {
		if (k === 'run_id') continue;
		if (v === null || v === undefined || v === '') continue;
		sp.set(k, String(v));
	}
	const query = sp.toString();
	return query ? `/runs/${run_id}/cohort?${query}` : `/runs/${run_id}/cohort`;
}


// ---- B4 cohort orchestration re-exports (see $lib/server/cohorts/) ----
//
// RUNTIME-INITIALIZATION CYCLE WARNING: each re-exported module imports
// helpers back from `$lib/db` (the same file). The cycle is currently safe
// because no `server/cohorts/*` file uses `$lib/db` symbols at module
// TOP-LEVEL — every usage is inside function bodies, by which time db.ts
// has finished evaluating. Adding a top-level `const X = SOME_DB_CONST * 2`
// inside any cohort module will produce a TDZ ReferenceError at server
// startup. If you need top-level state, move the dependency into a leaf
// module (no $lib/db imports at top-level of cohort files).
export { getLatestValidity } from './server/cohorts/validity';
export { getTruthSetOverlapDetail } from './server/cohorts/truthSet';
export { getStatementMatrix } from './server/cohorts/statementMatrix';
export { getRunCohort } from './server/cohorts/runCohort';
export { getRunRepairBacklog, getRepairRerunDetail } from './server/cohorts/repairCohort';
export { getPairedWorkbench } from './server/cohorts/workbench';
export { getRunNarrative, getHeuristicCoverage, getTraceFidelity, getResidualDistribution, getFindings } from './server/cohorts/runOverview/index';
export { getFocusStatement, getProbeAttribution, getStatementDetail } from './server/cohorts/statement';
export { getCorpusOverview } from './server/cohorts/corpusOverview';
// ---- end re-exports ----

export async function scalar(con: DuckDBConnection, sql: string): Promise<number> {
	try {
		const reader = await con.runAndReadAll(sql);
		const rows = reader.getRowObjects();
		if (rows.length === 0) return 0;
		const v = Object.values(rows[0])[0];
		return typeof v === 'bigint' ? Number(v) : Number(v ?? 0);
	} catch {
		return 0;
	}
}

// Lenient count helper for best-effort optional projections. Do not use this
// where a failed query would make broken data look legitimately absent.

export async function readScalar(con: DuckDBConnection, sql: string): Promise<number> {
	const reader = await con.runAndReadAll(sql);
	const rows = reader.getRowObjects();
	if (rows.length === 0) return 0;
	const v = Object.values(rows[0])[0];
	return typeof v === 'bigint' ? Number(v) : Number(v ?? 0);
}

export async function tableExists(con: DuckDBConnection, tableName: string): Promise<boolean> {
	const n = await readScalar(
		con,
		`SELECT COUNT(*)
		   FROM information_schema.tables
		  WHERE lower(table_name)=lower('${sqlQuote(tableName)}')`
	);
	return n > 0;
}

/**
 * Normalize DuckDB result values:
 *  - bigint → Number (UI works in JS numbers)
 *  - LIST of STRUCT → plain array of plain objects
 *    DuckDB wraps `list(struct(...))` as `{items: [{entries: {...}}]}` via
 *    @duckdb/node-api. Strip the wrapper here so call sites can treat the
 *    column as a plain array of objects.
 *  - STRUCT (non-list) → plain object (strip `entries`)
 */
export function normalizeDuckValue(v: unknown): unknown {
	if (typeof v === 'bigint') return Number(v);
	if (v && typeof v === 'object') {
		const obj = v as Record<string, unknown>;
		if ('items' in obj && Array.isArray(obj.items) && Object.keys(obj).length === 1) {
			return obj.items.map((it) => normalizeDuckValue(it));
		}
		if ('entries' in obj && obj.entries && typeof obj.entries === 'object' && Object.keys(obj).length === 1) {
			const e = obj.entries as Record<string, unknown>;
			const out: Record<string, unknown> = {};
			for (const [k, val] of Object.entries(e)) out[k] = normalizeDuckValue(val);
			return out;
		}
	}
	return v;
}

// Lenient row helper for best-effort optional projections. Use readRows for
// dashboard summary facts whose projection failures must be visible.
export async function rows<T = Record<string, unknown>>(
	con: DuckDBConnection,
	sql: string
): Promise<T[]> {
	try {
		const reader = await con.runAndReadAll(sql);
		return reader.getRowObjects().map((row) => {
			const out: Record<string, unknown> = {};
			for (const [k, v] of Object.entries(row)) {
				out[k] = normalizeDuckValue(v);
			}
			return out as T;
		});
	} catch {
		return [];
	}
}

export async function repairRerunLineageSqlOptions(
	con: DuckDBConnection
): Promise<RepairRerunLineageSqlOptions> {
	const n = await scalar(
		con,
		`SELECT COUNT(*)
		   FROM information_schema.columns
		  WHERE table_name='scorer_step_correction'
		    AND column_name IN ('parent_correction_id', 'child_run_id', 'repair_source_dump_id')`
	);
	return { typedLineage: n === 3 };
}

export async function resolveTraceSnapshotStartedAt(
	con: DuckDBConnection,
	run_id: string,
	requestedSnapshotStartedAt: string | null | undefined,
	forceLatest: boolean
): Promise<string | null> {
	const cleaned = cleanTraceSnapshot(requestedSnapshotStartedAt);
	if (cleaned === INVALID_TRACE_SNAPSHOT) {
		throw error(400, {
			message: `invalid trace_snapshot: ${requestedSnapshotStartedAt}`,
			code: 'invalid_trace_snapshot'
		});
	}
	if (!forceLatest && !cleaned) return null;
	const qr = run_id.replace(/'/g, "''");
	const requestedSnapshot = cleaned ? `TIMESTAMP '${cleaned.replace(/'/g, "''")}'` : 'NULL';
	const snapshotRows = await rows<{ snapshot_started_at: string | null }>(
		con,
		`SELECT
		   CASE
		     WHEN ${requestedSnapshot} IS NULL THEN MAX(started_at)
		     WHEN MAX(started_at) IS NULL THEN ${requestedSnapshot}
		     WHEN ${requestedSnapshot} > MAX(started_at) THEN MAX(started_at)
		     ELSE ${requestedSnapshot}
		   END::VARCHAR AS snapshot_started_at
		 FROM scorer_step
		 WHERE run_id='${qr}'
		   AND evidence_hash IS NOT NULL`
	);
	return snapshotRows[0]?.snapshot_started_at ?? null;
}

export interface StatementMatrixRow {
	stmt_hash: string;
	run_id: string | null;
	indra_type: string;
	indra_belief: number | null;
	agent_names: string;
	n_evidences: number;
	supports_count: number;
	supported_by_count: number;
	source_apis: string;
	source_dump_id: string | null;
	is_curated_any: number;
	our_belief: number | null;
	belief_delta: number | null;
}

export interface RunCohortRow {
	run_id: string;
	architecture: string;
	stmt_hash: string;
	evidence_hash: string | null;
	indra_type: string;
	agent_names: string;
	source_api: string | null;
	source_stratum?: string | null;
	representative_evidence_hash?: string | null;
	pmid: string | null;
	text: string | null;
	indra_belief: number | null;
	score: number | null;
	residual: number | null;
	verdict: string | null;
	confidence: string | null;
	trace_state: string | null;
	step_kind: string | null;
	latency_ms: number | null;
	prompt_tokens: number | null;
	out_tokens: number | null;
	n_evidences_for_stmt: number;
	has_supports: boolean;
}

export interface RunCohort {
	run_id: string;
	architecture: string;
	status: string;
	filters: RunCohortFilters;
	grain: 'evidence' | 'statement';
	rows: RunCohortRow[];
	totalRows: number;
	limit: number;
	emptyDiagnostics: RunCohortEmptyDiagnostic[];
}

export interface RunRepairCandidateRow {
	correction_id: number;
	step_hash: string;
	run_id: string;
	architecture: string;
	stmt_hash: string;
	evidence_hash: string | null;
	correction_kind: string;
	status: string;
	reviewer: string | null;
	note: string | null;
	value_json: string | null;
	source_route: string | null;
	source_filters_json: string | null;
	created_at: string;
	step_kind: string | null;
	suspected_step_kind: string | null;
	probe_coverage: string | null;
	missing_probe_slots: string | null;
	n_substrate_route: number | null;
	n_subject_role_probe: number | null;
	n_object_role_probe: number | null;
	n_relation_axis_probe: number | null;
	n_scope_probe: number | null;
	probe_slot_review_slots: string | null;
	probe_slot_review_note: string | null;
	probe_slot_review_reviewer: string | null;
	probe_slot_reviewed_at: string | null;
	probe_slot_review_count: number;
	severity: string | null;
	reviewer_hypothesis: string | null;
	indra_type: string | null;
	agent_names: string;
	source_api: string | null;
	pmid: string | null;
	text: string | null;
	verdict: string | null;
	confidence: string | null;
	score: number | null;
	residual: number | null;
}

export interface RunRepairBacklog {
	run_id: string;
	architecture: string;
	openCount: number;
	rows: RunRepairCandidateRow[];
	reruns: RepairRerunComparison[];
	activeRerunIntents: RepairRerunActiveIntent[];
	activeRerunIntentCount: number;
	recoverableRerunIntents: RepairRerunRecoveryIntent[];
	recoverableRerunIntentCount: number;
	recoverableRerunIntentOffset: number;
	recoverableRerunIntentLimit: number;
	limit: number;
}

export interface RunRepairBacklogOptions {
	recoveryOffset?: number | null;
	recoveryLimit?: number | null;
}

export interface RepairRerunRecoveryIntent {
	child_run_id: string;
	architecture: string;
	status: string;
	source_dump_id: string;
	correction_ids: number[];
	uncovered_correction_ids: number[];
	n_candidates: number;
	n_missing_markers: number;
	n_uncovered_candidates: number;
	first_intent_at: string;
	last_intent_at: string;
}

export interface RepairRerunActiveIntent {
	child_run_id: string;
	architecture: string;
	status: string;
	source_dump_id: string;
	correction_ids: number[];
	n_candidates: number;
	has_child_run: boolean;
	first_intent_at: string;
	last_intent_at: string;
}

export interface RepairRerunComparison {
	run_id: string;
	architecture: string;
	status: string;
	started_at: string | null;
	model_id_default: string | null;
	cost_estimate_usd: number | null;
	cost_actual_usd: number | null;
	n_overlap_evidences: number;
	n_score_evidences: number;
	n_verdict_evidences: number;
	parent_mae: number | null;
	child_mae: number | null;
	parent_bias: number | null;
	child_bias: number | null;
	verdicts_moved_total: number;
	verdicts_moved_to_correct: number;
	verdicts_moved_to_incorrect: number;
	n_candidate_evidences: number;
	n_parent_aggregate_candidates: number;
	n_child_covered_candidates: number;
	n_new_child_aggregate_candidates: number;
	candidate_lanes: RepairRerunCandidateLane[];
	not_defined_reason: string | null;
}

export interface RepairRerunDetail extends RepairRerunComparison {
	parent_run_id: string;
	child_run_id: string;
	parent_architecture: string;
	candidate_lane_total: number;
	candidate_lane_offset: number;
	candidate_lane_limit: number;
}

export interface RepairRerunCandidateLane {
	correction_id: number;
	stmt_hash: string;
	evidence_hash: string | null;
	indra_type: string | null;
	source_api: string | null;
	suspected_step_kind: string | null;
	parent_verdict: string | null;
	child_verdict: string | null;
	parent_score: number | null;
	child_score: number | null;
	indra_belief: number | null;
	parent_abs_error: number | null;
	child_abs_error: number | null;
	abs_error_delta: number | null;
	movement: string;
}

export interface PairRunSummary {
	run_id: string;
	architecture: string;
	scorer_version: string;
	model_id_default: string | null;
	started_at: string;
	finished_at: string | null;
	status: string;
	n_stmts: number | null;
	n_evidences: number;
	duration_s: number | null;
	cost_estimate_usd: number | null;
	cost_actual_usd: number | null;
}

export interface PairedOverlapStats {
	monolithic_evidences: number;
	decomposed_evidences: number;
	monolithic_comparable_evidences: number;
	decomposed_comparable_evidences: number;
	overlap_evidences: number;
	overlap_statements: number;
	monolithic_only_evidences: number;
	decomposed_only_evidences: number;
	monolithic_true_nonoverlap_evidences: number;
	decomposed_true_nonoverlap_evidences: number;
	monolithic_overlap_pct: number | null;
	decomposed_overlap_pct: number | null;
	monolithic_step_error_evidences: number;
	decomposed_step_error_evidences: number;
	monolithic_nonaggregate_step_error_evidences: number;
	decomposed_nonaggregate_step_error_evidences: number;
	monolithic_missing_aggregate_evidences: number;
	decomposed_missing_aggregate_evidences: number;
	monolithic_nonverdict_aggregate_evidences: number;
	decomposed_nonverdict_aggregate_evidences: number;
}

export interface PairedComparableMetrics {
	n_overlap: number;
	n_truth_overlap: number;
	verdict_agreement_n: number;
	verdict_agreement_rate: number | null;
	both_correct_n: number;
	monolithic_only_correct_n: number;
	decomposed_only_correct_n: number;
	both_incorrect_n: number;
	verdict_label_pairs: PairedVerdictPairRow[];
	monolithic_score_mean: number | null;
	decomposed_score_mean: number | null;
	monolithic_mae: number | null;
	decomposed_mae: number | null;
	monolithic_bias: number | null;
	decomposed_bias: number | null;
	mean_score_delta: number | null;
	monolithic_latency_mean_ms: number | null;
	decomposed_latency_mean_ms: number | null;
	monolithic_latency_observed_n: number;
	decomposed_latency_observed_n: number;
	monolithic_tokens_total: number;
	decomposed_tokens_total: number;
	monolithic_tokens_observed_n: number;
	decomposed_tokens_observed_n: number;
}

export interface PairedVerdictPairRow {
	monolithic_verdict: string;
	decomposed_verdict: string;
	support_cell: 'both_supported' | 'monolithic_only' | 'decomposed_only' | 'neither_supported';
	n: number;
}

export interface PairedResourceFrontierArch {
	architecture: 'monolithic' | 'decomposed';
	run_id: string;
	run_cost_usd: number | null;
	run_cost_basis: 'actual' | 'estimate' | 'missing';
	n_evidences: number;
	cost_per_evidence_usd: number | null;
	duration_s: number | null;
	wall_seconds_per_evidence: number | null;
	clean_overlap_n: number;
	clean_overlap_latency_mean_ms: number | null;
	clean_overlap_latency_observed_n: number;
	clean_overlap_tokens_total: number;
	clean_overlap_tokens_observed_n: number;
	clean_overlap_tokens_per_observed_evidence: number | null;
	truth_overlap_n: number;
	mae: number | null;
}

export interface PairedResourceFrontier {
	monolithic: PairedResourceFrontierArch;
	decomposed: PairedResourceFrontierArch;
	spend_scope: string;
	latency_scope: string;
	quality_scope: string;
	not_defined_reason: string | null;
}

export interface PairedExampleRow {
	stmt_hash: string;
	evidence_hash: string;
	indra_type: string;
	agent_names: string;
	text: string | null;
	source_api: string | null;
	pmid: string | null;
	indra_belief: number | null;
	monolithic_score: number | null;
	decomposed_score: number | null;
	monolithic_verdict: string | null;
	decomposed_verdict: string | null;
	monolithic_error: number | null;
	decomposed_error: number | null;
	abs_error_delta: number | null;
	monolithic_href: string;
	decomposed_href: string;
	excluded_side: string | null;
	excluded_reason: string | null;
}

export interface PairedArchProbeRow {
	name: string;
	n: number;
	substrate_n: number;
	error_n: number;
}

export interface PairedArchTierRow {
	tier: string;
	n: number;
	mean_score: number | null;
}

export interface PairedDenominatorRow {
	key: string;
	panel: string;
	applicability: PanelApplicabilityKind;
	metric_kind: PairedMetricKind;
	ledger_role: PairedLedgerRole;
	parent_key: string | null;
	unit: string;
	denominator_n: number | null;
	monolithic_n: number | null;
	decomposed_n: number | null;
	overlap_n: number | null;
	excluded_n: number | null;
	reason: string;
}

export interface PairedWorkbench {
	pair_id: string;
	runs: PairRunSummary[];
	monolithic: PairRunSummary | null;
	decomposed: PairRunSummary | null;
	overlap: PairedOverlapStats | null;
	comparable: PairedComparableMetrics | null;
	resource_frontier: PairedResourceFrontier | null;
	denominator_ledger: PairedDenominatorRow[];
	exemplars: {
		monolithic_wins: PairedExampleRow[];
		decomposed_wins: PairedExampleRow[];
		verdict_disagreements: PairedExampleRow[];
		mutual_failures: PairedExampleRow[];
		monolithic_only: PairedExampleRow[];
		decomposed_only: PairedExampleRow[];
		excluded_by_integrity: PairedExampleRow[];
	};
	arch_conditioned: {
		monolithic_tiers: PairedArchTierRow[];
		decomposed_probes: PairedArchProbeRow[];
	};
	not_defined_reason: string | null;
}

import { sqlQuote } from './cohorts/sqlEscape';
export { sqlQuote };

export function timestampMs(raw: string | null | undefined): number | null {
	if (!raw) return null;
	const normalized = raw.includes('T') ? raw : raw.replace(' ', 'T');
	const ms = Date.parse(normalized);
	return Number.isFinite(ms) ? ms : null;
}

export function durationSeconds(started_at: string | null | undefined, finished_at: string | null | undefined): number | null {
	const start = timestampMs(started_at);
	const finish = timestampMs(finished_at);
	if (start == null || finish == null || finish < start) return null;
	return (finish - start) / 1000;
}

export function safeRate(numerator: number | null | undefined, denominator: number | null | undefined): number | null {
	if (numerator == null || denominator == null || denominator <= 0 || Number.isNaN(numerator)) return null;
	return numerator / denominator;
}

// Strict row helper: SQL/projection failures reject and should surface through
// SvelteKit error handling when the UI would otherwise tell a false absence.
// C2: if a forced-close happened during a long-running read, the
// binding's error message typically contains one of the fragments
// below. Detect and re-throw as ReaderLeaseExpiredError so SSR routes
// can surface a typed 503 rather than the raw stack.
const LEASE_EXPIRED_FRAGMENTS = [
	'connection has been closed',
	'connection is closed',
	'instance is closed',
	'connection was closed',
	'Could not bind to closed connection'
];

function rethrowAsLeaseExpiredIfClosed(err: unknown): never {
	const message = err instanceof Error ? err.message : String(err);
	const lower = message.toLowerCase();
	if (LEASE_EXPIRED_FRAGMENTS.some((frag) => lower.includes(frag.toLowerCase()))) {
		throw new ReaderLeaseExpiredError(message);
	}
	throw err;
}

export async function readRows<T = Record<string, unknown>>(
	con: DuckDBConnection,
	sql: string
): Promise<T[]> {
	try {
		const reader = await con.runAndReadAll(sql);
		return reader.getRowObjects().map((row) => {
			const out: Record<string, unknown> = {};
			for (const [k, v] of Object.entries(row)) {
				out[k] = normalizeDuckValue(v);
			}
			return out as T;
		});
	} catch (err) {
		rethrowAsLeaseExpiredIfClosed(err);
	}
}

export async function assertCoreOverviewSchema(con: DuckDBConnection): Promise<void> {
	const requiredTables = Object.keys(CORE_OVERVIEW_REQUIRED_COLUMNS);
	const tableList = requiredTables.map((t) => `'${sqlQuote(t)}'`).join(', ');
	const presentRows = await readRows<{ table_name: string; column_name: string }>(
		con,
		`SELECT lower(table_name) AS table_name,
		        lower(column_name) AS column_name
		   FROM information_schema.columns
		  WHERE lower(table_name) IN (${tableList})`
	);
	const present = new Set(presentRows.map((r) => `${r.table_name}.${r.column_name}`));
	const missing: string[] = [];
	for (const [table, columns] of Object.entries(CORE_OVERVIEW_REQUIRED_COLUMNS)) {
		for (const column of columns) {
			if (!present.has(`${table}.${column}`)) missing.push(`${table}.${column}`);
		}
	}
	if (missing.length > 0) {
		const shown = missing.slice(0, 16).join(', ');
		const suffix = missing.length > 16 ? `, +${missing.length - 16} more` : '';
		// Recovery depends on connect() seeing the repaired file signature
		// change; otherwise restart the viewer to drop the cached instance.
		throw error(500, {
			code: 'corrupt_corpus_schema',
			message: `corrupt_corpus_schema: missing required DuckDB columns: ${shown}${suffix}`
		});
	}
}

export function truthSetOverlapCte(run_id: string, truth_set_id: string, step_kind: string): string {
	const qr = sqlQuote(run_id);
	const qt = sqlQuote(truth_set_id);
	const qs = sqlQuote(step_kind);
	return `
		WITH labels AS (
			SELECT tl.label_id,
			       tl.target_id,
			       tl.relation_target_id,
			       tl.field,
			       tl.value_text,
			       tl.provenance,
			       e.stmt_hash AS corpus_stmt_hash,
			       e.source_api,
			       e.text AS evidence_text
			FROM truth_label tl
			LEFT JOIN evidence e ON e.evidence_hash = tl.target_id
			WHERE tl.truth_set_id = '${qt}'
			  AND tl.target_kind = 'evidence'
			  AND tl.field IN ('verdict', 'tag')
		),
		agg_steps AS (
			SELECT step_hash,
			       stmt_hash,
			       evidence_hash,
			       json_extract_string(output_json, '$.verdict') AS our_verdict
			FROM scorer_step
			WHERE run_id = '${qr}'
			  AND step_kind = '${qs}'
			  AND evidence_hash IS NOT NULL
		),
		same_rollup AS (
			SELECT l.label_id,
			       COUNT(DISTINCT same.step_hash) AS same_evidence_scored_rows,
			       MIN(same.stmt_hash) AS sample_scored_stmt_hash
			FROM labels l
			LEFT JOIN agg_steps same
			  ON same.evidence_hash = l.target_id
			GROUP BY l.label_id
		),
		matched_rollup AS (
			SELECT l.label_id,
			       COUNT(DISTINCT matched.step_hash) AS matched_step_rows,
			       COUNT(DISTINCT CASE WHEN matched.our_verdict IS NOT NULL THEN matched.step_hash END) AS matched_verdict_rows,
			       COUNT(DISTINCT CASE WHEN matched.step_hash IS NOT NULL AND matched.our_verdict IS NULL THEN matched.step_hash END) AS matched_no_verdict_rows,
			       MIN(CASE WHEN matched.our_verdict IS NOT NULL THEN matched.stmt_hash END) AS sample_matched_stmt_hash,
			       MIN(CASE WHEN matched.our_verdict IS NOT NULL THEN matched.our_verdict END) AS sample_our_verdict
			FROM labels l
			LEFT JOIN agg_steps matched
			  ON matched.evidence_hash = l.target_id
			 AND (
				l.relation_target_id IS NULL
				OR l.relation_target_id = matched.stmt_hash
			 )
			GROUP BY l.label_id
		),
		label_rollup AS (
			SELECT l.label_id,
			       l.target_id,
			       l.relation_target_id,
			       l.field,
			       l.value_text,
			       l.provenance,
			       l.corpus_stmt_hash,
			       l.source_api,
			       l.evidence_text,
			       COALESCE(same.same_evidence_scored_rows, 0) AS same_evidence_scored_rows,
			       COALESCE(matched.matched_step_rows, 0) AS matched_step_rows,
			       COALESCE(matched.matched_verdict_rows, 0) AS matched_verdict_rows,
			       COALESCE(matched.matched_no_verdict_rows, 0) AS matched_no_verdict_rows,
			       matched.sample_matched_stmt_hash,
			       matched.sample_our_verdict,
			       same.sample_scored_stmt_hash
			FROM labels l
			LEFT JOIN same_rollup same ON same.label_id = l.label_id
			LEFT JOIN matched_rollup matched ON matched.label_id = l.label_id
		),
		classified_labels AS (
			SELECT *,
			       CASE
			         WHEN corpus_stmt_hash IS NULL THEN 'not_in_corpus'
			         WHEN matched_verdict_rows > 0 THEN 'compared'
			         WHEN matched_step_rows > 0 THEN 'no_aggregate_verdict'
			         WHEN relation_target_id IS NOT NULL AND same_evidence_scored_rows > 0 THEN 'context_mismatch'
			         ELSE 'not_scored_in_run'
			       END AS status
			FROM label_rollup
		),
		scorer_verdicts AS (
			SELECT '${qs}' AS step_kind,
			       stmt_hash,
			       evidence_hash,
			       json_extract_string(output_json, '$.verdict') AS our_verdict
			FROM scorer_step
			WHERE run_id = '${qr}'
			  AND step_kind = '${qs}'
			  AND json_extract(output_json, '$.verdict') IS NOT NULL
			  AND evidence_hash IS NOT NULL
		),
		gold_candidates AS (
			SELECT target_id AS evidence_hash,
			       relation_target_id,
			       field AS gold_field,
			       value_text AS gold_verdict,
			       CASE WHEN field = 'verdict' THEN 0 ELSE 1 END AS field_rank
			FROM truth_label
			WHERE truth_set_id = '${qt}'
			  AND target_kind = 'evidence'
			  AND field IN ('verdict', 'tag')
		),
		selected_gold AS (
			SELECT step_kind, our_verdict, gold_verdict, gold_field
			FROM (
				SELECT sv.step_kind,
				       sv.stmt_hash,
				       sv.evidence_hash,
				       sv.our_verdict,
				       gc.gold_verdict,
				       gc.gold_field,
				       row_number() OVER (
				         PARTITION BY sv.step_kind, sv.stmt_hash, sv.evidence_hash
				         ORDER BY
				           CASE WHEN gc.relation_target_id IS NOT NULL THEN 0 ELSE 1 END,
				           gc.field_rank,
				           gc.gold_verdict
				       ) AS rn
				FROM scorer_verdicts sv
				JOIN gold_candidates gc
				  ON gc.evidence_hash = sv.evidence_hash
				 AND (
					gc.relation_target_id IS NULL
					OR gc.relation_target_id = sv.stmt_hash
				 )
			)
			WHERE rn = 1
		)
	`;
}

export const REPAIR_RECOVERY_PAGE_LIMIT = 20;

export interface StatementDetail {
	stmt_hash: string;
	indra_type: string;
	indra_belief: number | null;
	supports_count: number;
	supported_by_count: number;
	source_dump_id: string | null;
	raw_json: string;
	selected_run_id: string | null;
	selected_architecture: string | null;
	selected_scorer_version: string | null;
	selected_run_status: string | null;
	available_runs: StatementRunOption[];
	agents: AgentRow[];
	evidences: EvidenceRow[];
	truth_labels: TruthLabelRow[];
	registered_truth_sets: string[];
	supports_edges: SupportsEdgeRow[];
	scorer_steps: ScorerStepRow[];
}

export interface ScorerStepRow {
	step_hash: string;
	run_id: string | null;
	evidence_hash: string | null;
	scorer_version: string;
	architecture: string | null;
	model_id: string | null;
	step_kind: string;
	is_substrate_answered: boolean | null;
	input_payload_json: string | null;
	output_json: string;
	latency_ms: number | null;
	prompt_tokens: number | null;
	out_tokens: number | null;
	finish_reason: string | null;
	error: string | null;
	started_at: string | null;
}

export interface StatementRunOption {
	run_id: string;
	architecture: string | null;
	scorer_version: string;
	model_id_default: string | null;
	started_at: string;
	status: string;
}

export interface AgentRow {
	agent_hash: string;
	role: string;
	role_index: number;
	name: string;
	db_refs_json: string;
	mods_json: string | null;
	location: string | null;
}

export interface EvidenceRow {
	evidence_hash: string;
	source_api: string | null;
	source_id: string | null;
	pmid: string | null;
	text: string | null;
	is_direct: boolean | null;
	is_negated: boolean | null;
	is_curated: boolean | null;
	epistemics_json: string | null;
}

export interface TruthLabelRow {
	truth_set_id: string;
	target_kind: string;
	target_id: string;
	field: string;
	value_text: string | null;
	value_json: string | null;
	provenance: string | null;
}

export interface SupportsEdgeRow {
	from_stmt_hash: string;
	to_stmt_hash: string;
	kind: string;
}

export interface RunNarrative {
	run_id: string;
	prev_run_id: string | null;
	summary_sentence: string;
	comparison_blocked_reason: string | null;
	mae_delta: number | null;
	bias_delta: number | null;
	verdicts_moved_total: number;
	verdicts_moved_to_correct: number;
	verdicts_moved_to_incorrect: number;
	verdict_crossings: Array<{ stmt_hash: string; prev_verdict: string; curr_verdict: string }>;
}

export interface ProbeCoverageRow {
	probe: ProbeKind;
	total: number;
	substrate_n: number;
	llm_n: number;
	abstain_n: number;
	notrun_n: number;
}

export type HeuristicCoverageTraceDiagnosticKind =
	| 'not_applicable'
	| 'no_aggregate_evidence'
	| 'aggregate_only_trace'
	| 'native_steps_without_probe_slots'
	| 'probe_rows_present';

export type HeuristicCoverageTraceLifecycleKind =
	| 'not_applicable'
	| 'probe_records_present'
	| 'running_snapshot'
	| 'interrupted_run'
	| 'succeeded_without_cause_provenance'
	| 'unknown_lifecycle';

export interface HeuristicCoverageTraceDiagnostic {
	kind: HeuristicCoverageTraceDiagnosticKind;
	n_aggregate_evidences: number;
	n_nonaggregate_steps: number;
	n_substrate_route_steps: number;
	n_probe_steps: number;
	n_grounding_steps: number;
	n_adjudicate_steps: number;
	run_status: string;
	scorer_version: string | null;
	started_at: string | null;
	terminated_by: string | null;
	termination_reason: string | null;
	lifecycle_kind: HeuristicCoverageTraceLifecycleKind;
	message: string;
	lifecycle_message: string;
}

export interface HeuristicCoverage {
	run_id: string;
	architecture: string;
	applicability: PanelApplicabilityKind;
	not_defined_reason: string | null;
	/** Aggregate verdict evidence rows; supports architecture-blind verdict/cost/latency summaries. */
	n_evidences: number;
	/** Evidence rows with decomposed substrate/probe slots; supports four-probe coverage rates. */
	n_probe_evidences: number;
	per_probe: ProbeCoverageRow[];
	trace_diagnostic: HeuristicCoverageTraceDiagnostic;
	/** % of evidences where every invoked probe was substrate-answered. */
	all_substrate_rate: number;
	/** % of evidences where at least one probe didn't run (short-circuit). */
	short_circuited_rate: number;
	/** Mean count of probes that fired (substrate + LLM, not "not run") per evidence. */
	mean_invoked_probes: number;
}

export interface TraceFidelityRow {
	stmt_hash: string;
	evidence_hash: string;
	state: TraceFidelityState;
	captured_steps: string[];
	missing_native_steps: string[];
	note: string;
}

export interface PanelApplicabilityRow {
	panel: string;
	applicability: PanelApplicabilityKind;
	reason: string;
	cohort_href: string | null;
}

export interface TraceFidelitySummary {
	run_id: string;
	architecture: string;
	n_evidences: number;
	counts: Record<TraceFidelityState, number>;
	native_grammar: string[];
	limitations: string[];
	rows: TraceFidelityRow[];
	detail_offset: number;
	detail_limit: number;
	detail_snapshot_started_at: string | null;
	panel_applicability: PanelApplicabilityRow[];
}

export interface TraceFidelityOptions {
	detailOffset?: number;
	detailLimit?: number;
	detailSnapshotStartedAt?: string | null;
}

export const TRACE_FIDELITY_DETAIL_LIMIT = 30;

export interface ResidualDistribution {
	run_id: string;
	bins: number[];
	n_total: number;
	mean_residual: number | null;
}

export type FindingKind =
	| 'biggest_disagreement'
	| 'probe_split'
	| 'verdict_regression'
	| 'verdict_recovery'
	| 'low_confidence_high_stakes';

export interface FindingRow {
	kind: FindingKind;
	stmt_hash: string;
	indra_type: string;
	agents: Array<{ role: string; name: string }>;
	our_score: number | null;
	indra_score: number | null;
	n_evidences: number;
	why_text: string;
	/** Sort key used to pick the top-K. Always > 0 for ranking purposes. */
	rank_value: number;
}

export interface Findings {
	run_id: string;
	prev_run_id: string | null;
	biggest_disagreement: FindingRow[];
	probe_split: FindingRow[];
	verdict_regression: FindingRow[];
	verdict_recovery: FindingRow[];
	low_confidence_high_stakes: FindingRow[];
}

export const FINDING_K = 5;

export interface FocusStatement {
	run_id: string;
	stmt: {
		stmt_hash: string;
		indra_type: string;
		agents: Array<{ role: string; name: string }>;
	};
	our_score: number | null;
	indra_score: number | null;
	probes: ProbeAttribution[];
	evidences: Array<{ evidence_hash: string; source_api: string | null; text: string | null }>;
	n_evidences: number;
	why_this_one: string;
}

export const PROBE_KINDS: Record<string, ProbeKind> = {
	subject_role_probe: 'subject_role',
	object_role_probe: 'object_role',
	relation_axis_probe: 'relation_axis',
	scope_probe: 'scope'
};

