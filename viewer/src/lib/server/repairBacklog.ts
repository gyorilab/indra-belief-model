import { randomUUID } from 'node:crypto';
import { DuckDBInstance, type DuckDBConnection } from '@duckdb/node-api';
import { closeInstance, dbExists, dbPath, resolveTraceSnapshotStartedAt, type RunCohortFilters } from '$lib/db';
import {
	cleanTraceSnapshot,
	cleanTraceStateFilter,
	INVALID_TRACE_SNAPSHOT,
	INVALID_TRACE_STATE
} from '$lib/traceState';
import { traceEvidenceCtesWithRunMeta } from '$lib/traceStateSql';
import {
	nonTraceEvidenceCohortWhereClauses,
	nonTraceStatementCohortWhereClauses,
	RUN_COHORT_LIMIT,
	traceStateCohortWhereClauses
} from '$lib/runCohortSql';
import { REPAIR_RERUN_LINEAGE_DDL } from '$lib/repairRerunSql';
import { validateRunCohortFilterValues } from '$lib/server/runCohortContract';
import {
	activePairedWorkflowStates,
	activeWriterLock,
	acquireWriterLock,
	clearWriterLockToken,
	writerLockConflictCode,
	writerLockConflictText
} from '$lib/server/pairedState';

export interface MaterializeRepairCohortInput {
	run_id: string;
	filters: RunCohortFilters;
	source_route: string;
	expected_run_status?: string | null;
	reviewer?: string | null;
	note?: string | null;
	estimate_token?: string | null;
	require_estimate_token?: boolean;
}

export interface MaterializeRepairCohortResult {
	run_id: string;
	grain: 'evidence' | 'statement';
	filters: RunCohortFilters;
	inspected: number;
	created: number;
	skipped_existing: number;
	skipped_duplicate_selection: number;
	limit: number;
}

export interface EstimateRepairCohortResult {
	run_id: string;
	grain: 'evidence' | 'statement';
	filters: RunCohortFilters;
	inspected: number;
	unique_selected: number;
	would_create: number;
	skipped_existing: number;
	skipped_duplicate_selection: number;
	limit: number;
	estimate_token: string;
	estimate_expires_at: string;
}

export class RepairCohortHttpError extends Error {
	constructor(
		public status: number,
		public code: string,
		message: string
	) {
		super(message);
		this.name = 'RepairCohortHttpError';
	}
}

export class RepairCohortInputError extends RepairCohortHttpError {
	constructor(code: string, message: string) {
		super(400, code, message);
		this.name = 'RepairCohortInputError';
	}
}

export class RepairCohortBusyError extends RepairCohortHttpError {
	constructor(code: string, message: string) {
		super(409, code, message);
		this.name = 'RepairCohortBusyError';
	}
}

export class RepairCohortConflictError extends RepairCohortHttpError {
	constructor(
		public expected: string,
		public actual: string
	) {
		super(
			409,
			'run_status_changed',
			`run status changed from ${expected} to ${actual}; refresh this cohort before creating repair candidates`
		);
		this.name = 'RepairCohortConflictError';
	}
}

export const CORRECTION_DDL = `
CREATE SEQUENCE IF NOT EXISTS scorer_step_correction_id_seq;
CREATE TABLE IF NOT EXISTS scorer_step_correction (
    correction_id       BIGINT PRIMARY KEY DEFAULT nextval('scorer_step_correction_id_seq'),
    step_hash           VARCHAR NOT NULL,
    run_id              VARCHAR NOT NULL,
    architecture         VARCHAR NOT NULL DEFAULT 'unknown',
    stmt_hash           VARCHAR NOT NULL,
    evidence_hash       VARCHAR,
    correction_kind     VARCHAR NOT NULL,
    status              VARCHAR NOT NULL DEFAULT 'open',
    reviewer            VARCHAR,
    note                TEXT,
    value_json          JSON,
    parent_correction_id BIGINT,
    child_run_id        VARCHAR,
    repair_source_dump_id VARCHAR,
    materialize_batch_id VARCHAR,
    source_route        VARCHAR,
    source_filters_json JSON,
    created_at          TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);
ALTER TABLE scorer_step_correction ADD COLUMN IF NOT EXISTS architecture VARCHAR DEFAULT 'unknown';
ALTER TABLE scorer_step_correction ADD COLUMN IF NOT EXISTS materialize_batch_id VARCHAR;
CREATE INDEX IF NOT EXISTS idx_scorer_step_correction_step ON scorer_step_correction(step_hash);
CREATE INDEX IF NOT EXISTS idx_scorer_step_correction_run ON scorer_step_correction(run_id);
CREATE INDEX IF NOT EXISTS idx_scorer_step_correction_status ON scorer_step_correction(status);
CREATE INDEX IF NOT EXISTS idx_scorer_step_correction_batch ON scorer_step_correction(materialize_batch_id);
${REPAIR_RERUN_LINEAGE_DDL}
	`;
const REPAIR_ESTIMATE_TTL_MS = 5 * 60 * 1000;
const repairEstimateTokens = new Map<string, { fingerprint: string; expiresAt: number }>();

function sqlQuote(s: string): string {
	return s.replace(/'/g, "''");
}

function stableJson(value: unknown): string {
	if (value === null || typeof value !== 'object') return JSON.stringify(value);
	if (Array.isArray(value)) return `[${value.map(stableJson).join(',')}]`;
	return `{${Object.entries(value as Record<string, unknown>)
		.sort(([a], [b]) => a.localeCompare(b))
		.map(([k, v]) => `${JSON.stringify(k)}:${stableJson(v)}`)
		.join(',')}}`;
}

function repairEstimateFingerprint(
	input: MaterializeRepairCohortInput,
	filters: RunCohortFilters
): string {
	return stableJson({
		run_id: input.run_id,
		source_route: input.source_route,
		expected_run_status: input.expected_run_status ?? null,
		reviewer: input.reviewer || '',
		filters
	});
}

function pruneRepairEstimateTokens(now = Date.now()): void {
	for (const [token, value] of repairEstimateTokens.entries()) {
		if (value.expiresAt <= now) repairEstimateTokens.delete(token);
	}
}

function issueRepairEstimateToken(fingerprint: string): { token: string; expiresAt: number } {
	const now = Date.now();
	pruneRepairEstimateTokens(now);
	const token = randomUUID();
	const expiresAt = now + REPAIR_ESTIMATE_TTL_MS;
	repairEstimateTokens.set(token, { fingerprint, expiresAt });
	return { token, expiresAt };
}

function consumeRepairEstimateToken(token: string | null | undefined, fingerprint: string): void {
	const now = Date.now();
	pruneRepairEstimateTokens(now);
	if (!token) {
		throw new RepairCohortInputError(
			'repair_estimate_required',
			'estimate this repair cohort before creating append-only repair candidates'
		);
	}
	const record = repairEstimateTokens.get(token);
	if (!record) {
		throw new RepairCohortHttpError(
			409,
			'repair_estimate_expired',
			'repair estimate expired or was already used; estimate this cohort again'
		);
	}
	if (record.fingerprint !== fingerprint) {
		throw new RepairCohortHttpError(
			409,
			'repair_estimate_stale',
			'repair estimate no longer matches this cohort; estimate this cohort again'
		);
	}
	repairEstimateTokens.delete(token);
}

async function openReadOnlyRepairInstance(): Promise<DuckDBInstance> {
	try {
		return await DuckDBInstance.create(dbPath(), { access_mode: 'READ_ONLY' });
	} catch (e) {
		const msg = (e as Error)?.message ?? String(e);
		if (msg.includes('Could not set lock') || msg.includes('Conflicting lock')) {
			throw new RepairCohortBusyError(
				'duckdb_writer_active',
				'DuckDB is locked by an active writer; retry the repair estimate after the writer finishes'
			);
		}
		throw e;
	}
}

async function correctionTableExists(con: DuckDBConnection): Promise<boolean> {
	const reader = await con.runAndReadAll(
		`SELECT COUNT(*) AS n
		 FROM information_schema.tables
		 WHERE table_name='scorer_step_correction'`
	);
	return Number(reader.getRowObjects()[0]?.n ?? 0) > 0;
}

async function runStatus(con: DuckDBConnection, run_id: string): Promise<string> {
	const reader = await con.runAndReadAll(
		`SELECT COALESCE(status, 'unknown') AS status
		 FROM score_run
		 WHERE run_id='${sqlQuote(run_id)}'
		 LIMIT 1`
	);
	return String(reader.getRowObjects()[0]?.status ?? 'unknown');
}

async function canonicalizeRepairFilters(
	con: DuckDBConnection,
	run_id: string,
	filters: RunCohortFilters,
	expectedRunStatus?: string | null
): Promise<RunCohortFilters> {
	const traceState = cleanTraceStateFilter(filters.trace_state);
	if (traceState === INVALID_TRACE_STATE) {
		throw new RepairCohortInputError('invalid_trace_state', `invalid trace_state: ${filters.trace_state}`);
	}
	const traceSnapshot = cleanTraceSnapshot(filters.trace_snapshot);
	if (traceSnapshot === INVALID_TRACE_SNAPSHOT) {
		throw new RepairCohortInputError('invalid_trace_snapshot', `invalid trace_snapshot: ${filters.trace_snapshot}`);
	}
	const probeCoverageCohort = filters.probe_coverage === 'present';
	const tracePlaneCohort = Boolean(traceState || probeCoverageCohort);
	if (traceSnapshot && !tracePlaneCohort) {
		throw new RepairCohortInputError('trace_plane_required', 'trace_snapshot requires trace_state or probe_coverage');
	}
	const status = await runStatus(con, run_id);
	if (expectedRunStatus && status && expectedRunStatus !== status) {
		throw new RepairCohortConflictError(expectedRunStatus, status);
	}
	if (!tracePlaneCohort) {
		if (status !== 'succeeded') {
			throw new RepairCohortHttpError(
				409,
				'repair_requires_succeeded_run',
				`aggregate and statement repair cohorts require a succeeded run; run ${run_id.slice(0, 8)} is ${status}`
			);
		}
		return filters;
	}
	if (status === 'running' && !traceSnapshot) {
		throw new RepairCohortInputError(
			'trace_snapshot_required',
			'running trace-plane repair requires trace_snapshot; refresh the cohort and retry from the pinned route'
		);
	}
	// This status read stays coherent with viewer-mediated selection/inserts
	// because repair holds the viewer writer lock used by viewer-spawned score
	// workers. External CLI writers must use pinned snapshots and status guards.
	const canonicalTraceSnapshot = await resolveTraceSnapshotStartedAt(
		con,
		run_id,
		traceSnapshot,
		status === 'running'
	);
	return {
		...filters,
		grain: 'evidence',
		trace_state: traceState,
		trace_snapshot: canonicalTraceSnapshot
	};
}

function selectSql(run_id: string, filters: RunCohortFilters): string {
	const grain = filters.grain === 'statement' ? 'statement' : 'evidence';
	const traceState = cleanTraceStateFilter(filters.trace_state);
	if (traceState === INVALID_TRACE_STATE) {
		throw new RepairCohortInputError('invalid_trace_state', `invalid trace_state: ${filters.trace_state}`);
	}
	const traceSnapshot = cleanTraceSnapshot(filters.trace_snapshot);
	if (traceSnapshot === INVALID_TRACE_SNAPSHOT) {
		throw new RepairCohortInputError('invalid_trace_snapshot', `invalid trace_snapshot: ${filters.trace_snapshot}`);
	}
	const probeCoverageCohort = filters.probe_coverage === 'present';
	const tracePlaneCohort = Boolean(traceState || probeCoverageCohort);
	if (tracePlaneCohort) {
		const where = traceStateCohortWhereClauses(filters, traceSnapshot);
		const whereSql = where.join(' AND ');
		return `
			WITH ${traceEvidenceCtesWithRunMeta(run_id, traceSnapshot)}
			SELECT
			   ts.step_hash,
			   ts.run_id,
			   COALESCE(ts.architecture, 'unknown') AS architecture,
			   ts.stmt_hash,
			   ts.evidence_hash,
			   ts.step_kind,
			   ts.trace_state,
			   ${probeCoverageCohort ? "'present'" : 'NULL'}::VARCHAR AS probe_coverage,
			   ts.n_substrate_route,
			   ts.n_subject_role_probe,
			   ts.n_object_role_probe,
			   ts.n_relation_axis_probe,
			   ts.n_scope_probe,
			   ts.verdict,
			   ts.confidence,
			   ts.score,
			   CASE
			     WHEN s.indra_belief IS NOT NULL AND ts.score IS NOT NULL
			     THEN ts.score - s.indra_belief
			     ELSE NULL
			   END AS residual
			FROM trace_rows ts
			LEFT JOIN statement s ON s.stmt_hash = ts.stmt_hash
			LEFT JOIN evidence e ON e.evidence_hash = ts.evidence_hash
			WHERE ${whereSql}
			ORDER BY ABS(COALESCE(ts.score - s.indra_belief, 0)) DESC,
			         ts.stmt_hash,
			         ts.evidence_hash
			LIMIT ${RUN_COHORT_LIMIT}`;
	}
	if (grain === 'evidence') {
		const whereSql = nonTraceEvidenceCohortWhereClauses(run_id, filters).join(' AND ');
		return `
			SELECT
			   ss.step_hash,
			   ss.run_id,
			   COALESCE(ss.architecture, 'unknown') AS architecture,
			   ss.stmt_hash,
			   ss.evidence_hash,
			   ss.step_kind,
			   NULL::VARCHAR AS trace_state,
			   NULL::VARCHAR AS probe_coverage,
			   NULL::BIGINT AS n_substrate_route,
			   NULL::BIGINT AS n_subject_role_probe,
			   NULL::BIGINT AS n_object_role_probe,
			   NULL::BIGINT AS n_relation_axis_probe,
			   NULL::BIGINT AS n_scope_probe,
			   json_extract_string(ss.output_json, '$.verdict') AS verdict,
			   json_extract_string(ss.output_json, '$.confidence') AS confidence,
			   CAST(json_extract(ss.output_json, '$.score') AS DOUBLE) AS score,
			   CASE
			     WHEN s.indra_belief IS NOT NULL AND json_extract(ss.output_json, '$.score') IS NOT NULL
			     THEN CAST(json_extract(ss.output_json, '$.score') AS DOUBLE) - s.indra_belief
			     ELSE NULL
			   END AS residual
			FROM scorer_step ss
			JOIN statement s ON s.stmt_hash = ss.stmt_hash
			LEFT JOIN evidence e ON e.evidence_hash = ss.evidence_hash
			WHERE ${whereSql}
			ORDER BY ABS(COALESCE(
			           CASE
			             WHEN s.indra_belief IS NOT NULL AND json_extract(ss.output_json, '$.score') IS NOT NULL
			             THEN CAST(json_extract(ss.output_json, '$.score') AS DOUBLE) - s.indra_belief
			             ELSE NULL
			           END,
			           0
			         )) DESC,
			         ss.stmt_hash,
			         ss.evidence_hash
			LIMIT ${RUN_COHORT_LIMIT}`;
	}

	const qr = sqlQuote(run_id);
	const where = nonTraceStatementCohortWhereClauses(filters);
	const whereSql = where.join(' AND ');
	const evidenceWhereSql = [
		...nonTraceEvidenceCohortWhereClauses(run_id, filters),
		`json_extract(ss.output_json, '$.score') IS NOT NULL`
	].join(' AND ');
	return `
		WITH stmt_scores AS (
			SELECT
			   ss.run_id,
			   s.stmt_hash,
			   s.indra_type,
			   (SELECT MIN(e.source_api) FROM statement_evidence se JOIN evidence e ON e.evidence_hash=se.evidence_hash WHERE se.stmt_hash=s.stmt_hash) AS source_api,
			   (SELECT MIN(e.source_api) FROM statement_evidence se JOIN evidence e ON e.evidence_hash=se.evidence_hash WHERE se.stmt_hash=s.stmt_hash) AS source_stratum,
			   s.indra_belief,
			   AVG(CAST(json_extract(ss.output_json, '$.score') AS DOUBLE)) AS score,
			   CASE
			     WHEN s.indra_belief IS NOT NULL AND AVG(CAST(json_extract(ss.output_json, '$.score') AS DOUBLE)) IS NOT NULL
			     THEN AVG(CAST(json_extract(ss.output_json, '$.score') AS DOUBLE)) - s.indra_belief
			     ELSE NULL
			   END AS residual,
			   COUNT(*) AS n_evidences_for_stmt,
			   (s.supports_count > 0 OR s.supported_by_count > 0) AS has_supports
			 FROM scorer_step ss
			 JOIN statement s ON s.stmt_hash = ss.stmt_hash
			 WHERE ss.run_id='${qr}'
			   AND ss.step_kind='aggregate'
			   AND json_extract(ss.output_json, '$.score') IS NOT NULL
			 GROUP BY ss.run_id, s.stmt_hash, s.indra_type, s.indra_belief,
			          s.supports_count, s.supported_by_count
		),
		selected_statements AS (
			SELECT
			   stmt_hash,
			   ROW_NUMBER() OVER (ORDER BY ABS(COALESCE(residual, 0)) DESC, stmt_hash) AS display_rank
			FROM stmt_scores
			WHERE ${whereSql}
			ORDER BY ABS(COALESCE(residual, 0)) DESC, stmt_hash
			LIMIT ${RUN_COHORT_LIMIT}
		)
		SELECT
		   ss.step_hash,
		   ss.run_id,
		   COALESCE(ss.architecture, 'unknown') AS architecture,
		   ss.stmt_hash,
		   ss.evidence_hash,
		   ss.step_kind,
		   NULL::VARCHAR AS trace_state,
		   NULL::VARCHAR AS probe_coverage,
		   NULL::BIGINT AS n_substrate_route,
		   NULL::BIGINT AS n_subject_role_probe,
		   NULL::BIGINT AS n_object_role_probe,
		   NULL::BIGINT AS n_relation_axis_probe,
		   NULL::BIGINT AS n_scope_probe,
		   json_extract_string(ss.output_json, '$.verdict') AS verdict,
		   json_extract_string(ss.output_json, '$.confidence') AS confidence,
		   CAST(json_extract(ss.output_json, '$.score') AS DOUBLE) AS score,
		   CASE
		     WHEN s.indra_belief IS NOT NULL AND json_extract(ss.output_json, '$.score') IS NOT NULL
		     THEN CAST(json_extract(ss.output_json, '$.score') AS DOUBLE) - s.indra_belief
		     ELSE NULL
		   END AS residual
		FROM scorer_step ss
		JOIN selected_statements st ON st.stmt_hash=ss.stmt_hash
		JOIN statement s ON s.stmt_hash=ss.stmt_hash
		LEFT JOIN evidence e ON e.evidence_hash=ss.evidence_hash
		WHERE ${evidenceWhereSql}
		ORDER BY st.display_rank, ss.evidence_hash`;
	}

async function createRepairSelected(
	con: DuckDBConnection,
	run_id: string,
	filters: RunCohortFilters
): Promise<void> {
	await con.run('DROP TABLE IF EXISTS repair_selected');
	await con.run(
		`CREATE TEMP TABLE repair_selected AS
		 SELECT
		   selected.*,
		   ROW_NUMBER() OVER (
		     PARTITION BY selected.step_hash
		     ORDER BY ABS(COALESCE(selected.residual, 0)) DESC,
		              selected.stmt_hash,
		              selected.evidence_hash,
		              selected.step_kind
		   ) AS repair_candidate_rank
		 FROM (${selectSql(run_id, filters)}) selected`
	);
}

async function repairSelectionCounts(
	con: DuckDBConnection,
	sourceRoute: string,
	reviewerKey: string,
	hasCorrectionTable: boolean
): Promise<{
	inspected: number;
	uniqueSelected: number;
	skippedExisting: number;
	skippedDuplicateSelection: number;
	wouldCreate: number;
}> {
	const countReader = await con.runAndReadAll('SELECT COUNT(*) AS n FROM repair_selected');
	const inspected = Number(countReader.getRowObjects()[0]?.n ?? 0);
	const uniqueReader = await con.runAndReadAll(
		`SELECT COUNT(*) AS n FROM repair_selected WHERE repair_candidate_rank=1`
	);
	const uniqueSelected = Number(uniqueReader.getRowObjects()[0]?.n ?? 0);
	const skippedDuplicateSelection = inspected - uniqueSelected;
	let skippedExisting = 0;
	if (hasCorrectionTable) {
		const existingReader = await con.runAndReadAll(
			`SELECT COUNT(*) AS n
			 FROM repair_selected rs
			 WHERE rs.repair_candidate_rank=1
			   AND EXISTS (
				SELECT 1
				FROM scorer_step_correction rr
				WHERE rr.step_hash=rs.step_hash
				  AND rr.correction_kind='repair_candidate'
				  AND rr.status='open'
				  AND COALESCE(rr.source_route, '')=?
				  AND COALESCE(rr.reviewer, '')=?
			 )`,
			[sourceRoute, reviewerKey]
		);
		skippedExisting = Number(existingReader.getRowObjects()[0]?.n ?? 0);
	}
	return {
		inspected,
		uniqueSelected,
		skippedExisting,
		skippedDuplicateSelection,
		wouldCreate: Math.max(0, uniqueSelected - skippedExisting)
	};
}

export async function estimateRepairCohort(
	input: MaterializeRepairCohortInput
): Promise<EstimateRepairCohortResult> {
	if (!dbExists()) throw new RepairCohortHttpError(404, 'corpus_db_missing', 'corpus DuckDB does not exist');
	const activePair = activePairedWorkflowStates()[0] ?? null;
	if (activePair) {
		throw new RepairCohortBusyError(
			'paired_workflow_active',
			`paired workflow ${activePair.pair_id} is already ${activePair.status}; wait, cancel it, or inspect ${activePair.href}`
		);
	}
	const activeLock = activeWriterLock();
	if (activeLock) throw new RepairCohortBusyError(writerLockConflictCode(activeLock), writerLockConflictText(activeLock));

	const instance = await openReadOnlyRepairInstance();
	const con = await instance.connect();
	try {
		const filters = await canonicalizeRepairFilters(
			con,
			input.run_id,
			input.filters,
			input.expected_run_status
		);
		await validateRunCohortFilterValues(con, input.run_id, filters);
		const tracePlaneCohort = Boolean(filters.trace_state || filters.probe_coverage === 'present');
		const grain = tracePlaneCohort ? 'evidence' : filters.grain === 'statement' ? 'statement' : 'evidence';
		const reviewerKey = input.reviewer || '';
		await createRepairSelected(con, input.run_id, filters);
		const counts = await repairSelectionCounts(
			con,
			input.source_route,
			reviewerKey,
			await correctionTableExists(con)
		);
		const fingerprint = repairEstimateFingerprint(input, filters);
		const estimateToken = issueRepairEstimateToken(fingerprint);
		return {
			run_id: input.run_id,
			grain,
			filters,
			inspected: counts.inspected,
			unique_selected: counts.uniqueSelected,
			would_create: counts.wouldCreate,
			skipped_existing: counts.skippedExisting,
			skipped_duplicate_selection: counts.skippedDuplicateSelection,
			limit: RUN_COHORT_LIMIT,
			estimate_token: estimateToken.token,
			estimate_expires_at: new Date(estimateToken.expiresAt).toISOString()
		};
	} finally {
		con.disconnectSync?.();
		instance.closeSync();
	}
}

export async function materializeRepairCohort(
	input: MaterializeRepairCohortInput
): Promise<MaterializeRepairCohortResult> {
	if (!dbExists()) throw new RepairCohortHttpError(404, 'corpus_db_missing', 'corpus DuckDB does not exist');
	const activePair = activePairedWorkflowStates()[0] ?? null;
	if (activePair) {
		throw new RepairCohortBusyError(
			'paired_workflow_active',
			`paired workflow ${activePair.pair_id} is already ${activePair.status}; wait, cancel it, or inspect ${activePair.href}`
		);
	}
	const activeLock = activeWriterLock();
	if (activeLock) throw new RepairCohortBusyError(writerLockConflictCode(activeLock), writerLockConflictText(activeLock));
	const writerLock = acquireWriterLock({
		kind: 'repair',
		label: 'repair backlog',
		source_dump_id: null,
		dataset_path: dbPath(),
		pid: process.pid
	});
	if (!writerLock) {
		const lock = activeWriterLock();
		throw new RepairCohortBusyError(
			writerLockConflictCode(lock),
			lock ? writerLockConflictText(lock) : 'DuckDB writer lock is busy; retry after the active worker finishes'
		);
	}

	let instance: DuckDBInstance | null = null;
	let con: DuckDBConnection | null = null;
	try {
		closeInstance();
		instance = await DuckDBInstance.create(dbPath());
		con = await instance.connect();
		await con.run(CORRECTION_DDL);
		const filters = await canonicalizeRepairFilters(
			con,
			input.run_id,
			input.filters,
			input.expected_run_status
			);
			await validateRunCohortFilterValues(con, input.run_id, filters);
			if (input.require_estimate_token) {
				consumeRepairEstimateToken(
					input.estimate_token,
					repairEstimateFingerprint(input, filters)
				);
			}
			const tracePlaneCohort = Boolean(filters.trace_state || filters.probe_coverage === 'present');
			const grain = tracePlaneCohort ? 'evidence' : filters.grain === 'statement' ? 'statement' : 'evidence';
			const filtersJson = JSON.stringify(filters);
			const reviewer = input.reviewer || null;
			const reviewerKey = input.reviewer || '';
			const note = input.note || null;
			const stepKind = filters.step_kind ?? null;
			await createRepairSelected(con, input.run_id, filters);
			const counts = await repairSelectionCounts(con, input.source_route, reviewerKey, true);
			const inspected = counts.inspected;
			const skipped_existing = counts.skippedExisting;
			const skipped_duplicate_selection = counts.skippedDuplicateSelection;
			const materializeBatchId = randomUUID();
			let created = 0;
		await con.run('BEGIN TRANSACTION');
		try {
			await con.run(
				`INSERT INTO scorer_step_correction
				 (step_hash, run_id, architecture, stmt_hash, evidence_hash,
				  correction_kind, status, reviewer, note, value_json,
				  materialize_batch_id, source_route, source_filters_json)
				 SELECT
				   rs.step_hash,
				   rs.run_id,
				   rs.architecture,
				   rs.stmt_hash,
				   rs.evidence_hash,
				   'repair_candidate',
				   'open',
				   ?,
				   ?,
				   json_object(
				     'kind', 'cohort_repair_candidate',
				     'grain', ?,
				     'suspected_step_kind',
				        CASE
				          WHEN rs.probe_coverage='present' THEN 'probe_coverage'
				          WHEN rs.trace_state IS NOT NULL THEN
				            CASE WHEN rs.trace_state='step_error' THEN COALESCE(rs.step_kind, 'step_error') ELSE rs.trace_state END
				          WHEN CAST(? AS VARCHAR) IS NOT NULL AND CAST(? AS VARCHAR) <> 'aggregate' THEN CAST(? AS VARCHAR)
				          ELSE 'aggregate'
				        END,
				     'severity', 'untriaged',
				     'materialize_batch_id', ?,
				     'reviewer_hypothesis', ?,
				     'observed', json_object(
				        'verdict', rs.verdict,
				        'confidence', rs.confidence,
				        'score', rs.score,
				        'residual', rs.residual,
				        'trace_state', rs.trace_state,
				        'step_kind', rs.step_kind,
				        'probe_coverage', rs.probe_coverage,
				        'probe_counts', json_object(
				          'substrate_route', rs.n_substrate_route,
				          'subject_role_probe', rs.n_subject_role_probe,
				          'object_role_probe', rs.n_object_role_probe,
				          'relation_axis_probe', rs.n_relation_axis_probe,
				          'scope_probe', rs.n_scope_probe
				        ),
				        'missing_probe_slots',
				          CONCAT(
				            CASE WHEN rs.probe_coverage='present' AND COALESCE(rs.n_substrate_route, 0)=0 THEN 'substrate_route,' ELSE '' END,
				            CASE WHEN rs.probe_coverage='present' AND COALESCE(rs.n_subject_role_probe, 0)=0 THEN 'subject_role_probe,' ELSE '' END,
				            CASE WHEN rs.probe_coverage='present' AND COALESCE(rs.n_object_role_probe, 0)=0 THEN 'object_role_probe,' ELSE '' END,
				            CASE WHEN rs.probe_coverage='present' AND COALESCE(rs.n_relation_axis_probe, 0)=0 THEN 'relation_axis_probe,' ELSE '' END,
				            CASE WHEN rs.probe_coverage='present' AND COALESCE(rs.n_scope_probe, 0)=0 THEN 'scope_probe,' ELSE '' END
				          )
				     ),
				     'source_route', ?
				   )::JSON,
				   ?,
				   ?,
				   CAST(? AS JSON)
				 FROM repair_selected rs
				 WHERE rs.repair_candidate_rank=1
				   AND NOT EXISTS (
					SELECT 1
					FROM scorer_step_correction rr
					WHERE rr.step_hash=rs.step_hash
					  AND rr.correction_kind='repair_candidate'
					  AND rr.status='open'
					  AND COALESCE(rr.source_route, '')=?
					  AND COALESCE(rr.reviewer, '')=?
				 )`,
				[
					reviewer,
					note,
					grain,
					stepKind,
					stepKind,
					stepKind,
					materializeBatchId,
					note,
					input.source_route,
					materializeBatchId,
					input.source_route,
					filtersJson,
					input.source_route,
					reviewerKey
				]
			);
			const actualCreatedReader = await con.runAndReadAll(
				`SELECT COUNT(*) AS n
				 FROM scorer_step_correction
				 WHERE correction_kind='repair_candidate'
				   AND materialize_batch_id=?`,
				[materializeBatchId]
			);
			created = Number(actualCreatedReader.getRowObjects()[0]?.n ?? 0);
				if (created + skipped_existing + skipped_duplicate_selection !== inspected) {
					throw new RepairCohortHttpError(
						409,
						'repair_count_changed',
						`repair materialization count mismatch: created=${created}, skipped_existing=${skipped_existing}, ` +
							`skipped_duplicate_selection=${skipped_duplicate_selection}, inspected=${inspected}`
					);
			}
			await con.run('COMMIT');
		} catch (e) {
			await con.run('ROLLBACK');
			throw e;
		}
		return {
			run_id: input.run_id,
			grain,
			filters,
			inspected,
			created,
			skipped_existing,
			skipped_duplicate_selection,
			limit: RUN_COHORT_LIMIT
		};
		} finally {
			try {
				con?.disconnectSync?.();
			} catch {
				// Preserve the writer-lock cleanup path even if native close fails.
			}
			try {
				instance?.closeSync();
			} catch {
				// Preserve the writer-lock cleanup path even if native close fails.
			}
			clearWriterLockToken(writerLock.token);
		}
	}
