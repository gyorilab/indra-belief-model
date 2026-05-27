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
} from './runCohortSql';
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
} from './runCohortQueries';
import {
	statementEvidenceCountSql,
	statementEvidenceSourceStratumSql
} from './evidenceMembershipSql';
import {
	aggregateEvidenceEmptyPredicates,
	emptyCohortDiagnostics,
	runScorerStepCounts,
	statementEmptyPredicates,
	traceEvidenceEmptyPredicates,
	type RunCohortEmptyDiagnostic
} from './runCohortDiagnostics';
import { validateRunCohortFilterValues } from './server/runCohortContract';
import type { RunCohortFilters } from './runCohortTypes';

export type { ProbeAttribution, ProbeKind, ProbeOutput } from './probeAttribution';
export type { TraceFidelityState } from './traceState';
export type { RunCohortFilters } from './runCohortTypes';
export type { RunCohortEmptyDiagnostic } from './runCohortDiagnostics';
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
	if (
		_instance &&
		statAvailable &&
		(currentMtime !== _instanceMtimeMs ||
			currentCtime !== _instanceCtimeMs ||
			currentSize !== _instanceSize)
	) {
		try {
			(_instance as unknown as { closeSync?: () => void }).closeSync?.();
		} catch {
			// best-effort
		}
		_instance = null;
	}
	if (!_instance) {
		// READ_ONLY mode. DuckDB's file-level lock is held at the *instance*
		// level: a process holding any open instance (even READ_ONLY) blocks
		// another process from opening the same file for writing. Endpoints
		// that spawn a Python writer (ingest / score) MUST call
		// closeInstance() before spawning so the worker can acquire its lock.
		// The next connect() lazy-reopens. Dashboard reads issued while a
		// writer holds the lock surface as a typed 503 (caught by
		// +error.svelte and rendered as a friendly waiting screen) rather
		// than a raw 500 with a DuckDB stack trace.
		try {
			_instance = await DuckDBInstance.create(path, { access_mode: 'READ_ONLY' });
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
		_instanceMtimeMs = currentMtime;
		_instanceCtimeMs = currentCtime;
		_instanceSize = currentSize;
	}
	return _instance.connect();
}

/**
 * Release the cached READ_ONLY instance so a Python writer subprocess can
 * acquire the file lock. Call before spawning ingest / score workers; the
 * next `connect()` will lazily reopen. Cheap and idempotent.
 */
export function closeInstance(): void {
	if (!_instance) return;
	try {
		(_instance as unknown as { closeSync?: () => void }).closeSync?.();
	} catch {
		// best-effort — instance may already be closed
	}
	_instance = null;
	_instanceMtimeMs = 0;
	_instanceCtimeMs = 0;
	_instanceSize = 0;
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

const EMPTY_OVERVIEW: Omit<CorpusOverview, 'dbPath' | 'dbExists'> = {
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

const CORE_OVERVIEW_REQUIRED_COLUMNS: Record<string, string[]> = {
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

function cohortHref(
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

async function getLatestValidity(con: DuckDBConnection): Promise<LatestValidity | null> {
	const runs = await readRows<{
		run_id: string;
		scorer_version: string;
		architecture: string | null;
		paired_run_group_id: string | null;
	}>(
		con,
		`SELECT run_id, scorer_version, architecture, paired_run_group_id FROM score_run
		 WHERE status = 'succeeded'
		 ORDER BY started_at DESC LIMIT 1`
	);
	if (runs.length === 0) return null;
	const { run_id, scorer_version } = runs[0];
	const architecture = runs[0].architecture ?? 'unknown';
	const pairedGroup = runs[0].paired_run_group_id ?? null;

	type TruthPresentRaw = Omit<TruthPresentRow, 'cohort_href' | 'truth_href' | 'applicability' | 'gold_fields'> & {
		gold_fields_json: string | null;
	};
	const latestValidityReads = [
		readRows<{ verdict: string; n: number }>(
			con,
			`SELECT replace(replace(metric_name, 'verdict_share.', ''), '"', '') AS verdict,
			        CAST(json_extract(slice_json, '$.n') AS BIGINT) AS n
			 FROM metric
			 WHERE run_id = '${run_id.replace(/'/g, "''")}' AND metric_name LIKE 'verdict_share.%'
			 ORDER BY n DESC`
		),
		readRows<{ value: number }>(
			con,
			`SELECT value FROM metric
			 WHERE run_id = '${run_id.replace(/'/g, "''")}'
			   AND metric_name = 'indra_belief_calibration.mae' AND truth_set_id = 'indra_published_belief' LIMIT 1`
		),
		readRows<{ value: number }>(
			con,
			`SELECT value FROM metric
			 WHERE run_id = '${run_id.replace(/'/g, "''")}'
			   AND metric_name = 'indra_belief_calibration.rmse' AND truth_set_id = 'indra_published_belief' LIMIT 1`
		),
		readRows<{ value: number }>(
			con,
			`SELECT value FROM metric
			 WHERE run_id = '${run_id.replace(/'/g, "''")}'
			   AND metric_name = 'indra_belief_calibration.bias' AND truth_set_id = 'indra_published_belief' LIMIT 1`
		),
		readRows<{ n: number }>(
			con,
			`SELECT CAST(json_extract(slice_json, '$.n_stmts') AS BIGINT) AS n
			 FROM metric
			 WHERE run_id = '${run_id.replace(/'/g, "''")}'
			   AND metric_name = 'indra_belief_calibration.mae' AND truth_set_id = 'indra_published_belief' LIMIT 1`
		),
		readRows<{ value: number; n: number }>(
			con,
			`SELECT value, CAST(json_extract(slice_json, '$.n_multi_evidence_stmts') AS BIGINT) AS n
			 FROM metric
			 WHERE run_id = '${run_id.replace(/'/g, "''")}'
			   AND metric_name = 'inter_evidence_consistency.mean_stdev' LIMIT 1`
		),
		readRows<{ value: number; unavailable_reason: string | null }>(
			con,
			`SELECT value, json_extract_string(slice_json, '$.unavailable_reason') AS unavailable_reason
			 FROM metric
			 WHERE run_id = '${run_id.replace(/'/g, "''")}'
			   AND metric_name = 'supports_graph_plausibility.delta' LIMIT 1`
		),
		readRows<Omit<StratumRow, 'cohort_href' | 'applicability'>>(
			con,
			`WITH mae_rows AS (
				SELECT json_extract(slice_json, '$.value')::VARCHAR AS value,
				       CAST(json_extract(slice_json, '$.n') AS BIGINT) AS n,
				       value AS mae
				FROM metric
				WHERE run_id = '${run_id.replace(/'/g, "''")}'
				  AND metric_name = 'indra_belief_calibration_by_type.mae' AND truth_set_id = 'indra_published_belief'
			),
			bias_rows AS (
				SELECT json_extract(slice_json, '$.value')::VARCHAR AS value,
				       value AS bias
				FROM metric
				WHERE run_id = '${run_id.replace(/'/g, "''")}'
				  AND metric_name = 'indra_belief_calibration_by_type.bias' AND truth_set_id = 'indra_published_belief'
			)
			SELECT replace(m.value, '"', '') AS value, m.n, m.mae, COALESCE(b.bias, 0) AS bias
			FROM mae_rows m LEFT JOIN bias_rows b USING(value)
			ORDER BY m.mae DESC`
		),
		readRows<Omit<StratumRow, 'cohort_href' | 'applicability'>>(
			con,
			`WITH mae_rows AS (
				SELECT json_extract(slice_json, '$.value')::VARCHAR AS value,
				       CAST(json_extract(slice_json, '$.n') AS BIGINT) AS n,
				       value AS mae
				FROM metric
				WHERE run_id = '${run_id.replace(/'/g, "''")}'
				  AND metric_name = 'indra_belief_calibration_by_source.mae' AND truth_set_id = 'indra_published_belief'
			),
			bias_rows AS (
				SELECT json_extract(slice_json, '$.value')::VARCHAR AS value,
				       value AS bias
				FROM metric
				WHERE run_id = '${run_id.replace(/'/g, "''")}'
				  AND metric_name = 'indra_belief_calibration_by_source.bias' AND truth_set_id = 'indra_published_belief'
			)
			SELECT replace(m.value, '"', '') AS value, m.n, m.mae, COALESCE(b.bias, 0) AS bias
			FROM mae_rows m LEFT JOIN bias_rows b USING(value)
			ORDER BY m.mae DESC`
		),
		readRows<TruthPresentRaw>(
			con,
			`WITH p AS (
				SELECT truth_set_id,
				       replace(json_extract(slice_json, '$.step_kind')::VARCHAR, '"', '') AS step_kind,
				       value AS precision,
				       CAST(json_extract(slice_json, '$.n_compared') AS BIGINT) AS n_compared,
				       CAST(json_extract(slice_json, '$.tp') AS BIGINT) AS tp,
				       CAST(json_extract(slice_json, '$.fp') AS BIGINT) AS fp,
				       CAST(json_extract(slice_json, '$.fn') AS BIGINT) AS fn,
				       COALESCE(
				         CAST(json_extract(slice_json, '$.tn') AS BIGINT),
				         CAST(json_extract(slice_json, '$.n_compared') AS BIGINT)
				           - CAST(json_extract(slice_json, '$.tp') AS BIGINT)
				           - CAST(json_extract(slice_json, '$.fp') AS BIGINT)
				           - CAST(json_extract(slice_json, '$.fn') AS BIGINT)
				       ) AS tn,
				       CAST(json_extract(slice_json, '$.n_gold_labels') AS BIGINT) AS n_gold_labels,
				       CAST(json_extract(slice_json, '$.n_applicable_gold_labels') AS BIGINT) AS n_applicable_gold_labels,
				       CAST(json_extract(slice_json, '$.n_scored_evidences') AS BIGINT) AS n_scored_evidences,
				       json_extract(slice_json, '$.gold_fields')::VARCHAR AS gold_fields_json,
				       COALESCE(json_extract_string(slice_json, '$.positive_gold_label'), 'correct') AS positive_gold_label,
				       COALESCE(json_extract_string(slice_json, '$.negative_gold_rule'), 'any value != ''correct''') AS negative_gold_rule,
				       json_extract_string(slice_json, '$.unavailable_reason') AS unavailable_reason
				FROM metric
				WHERE run_id = '${run_id.replace(/'/g, "''")}'
				  AND metric_name LIKE 'truth_present.%.precision'
			),
			r AS (
				SELECT truth_set_id,
				       replace(json_extract(slice_json, '$.step_kind')::VARCHAR, '"', '') AS step_kind,
				       value AS recall
				FROM metric
				WHERE run_id = '${run_id.replace(/'/g, "''")}'
				  AND metric_name LIKE 'truth_present.%.recall'
			),
			f AS (
				SELECT truth_set_id,
				       replace(json_extract(slice_json, '$.step_kind')::VARCHAR, '"', '') AS step_kind,
				       value AS f1
				FROM metric
				WHERE run_id = '${run_id.replace(/'/g, "''")}'
				  AND metric_name LIKE 'truth_present.%.f1'
			),
			measured AS (
				SELECT p.truth_set_id, p.step_kind,
				       p.precision, COALESCE(r.recall, 0) AS recall, COALESCE(f.f1, 0) AS f1,
				       p.n_compared, p.tp, p.fp, p.fn, p.tn,
				       p.n_gold_labels,
				       p.n_applicable_gold_labels,
				       p.n_scored_evidences,
				       p.gold_fields_json, p.positive_gold_label, p.negative_gold_rule,
				       p.unavailable_reason
				FROM p
				LEFT JOIN r USING (truth_set_id, step_kind)
				LEFT JOIN f USING (truth_set_id, step_kind)
			)
			SELECT * FROM measured
			ORDER BY truth_set_id, step_kind`
		),
		readRows<Omit<ConfidenceCalibrationRow, 'cohort_href' | 'applicability'>>(
			con,
			`WITH aggregate_rows AS (
				SELECT
				   ss.stmt_hash,
				   s.indra_type,
				   CAST(json_extract(ss.output_json, '$.score') AS DOUBLE) AS score,
				   s.indra_belief AS indra_belief,
				   COALESCE(json_extract_string(ss.output_json, '$.confidence'), 'unknown') AS confidence
				FROM scorer_step ss
				JOIN statement s ON s.stmt_hash = ss.stmt_hash
				WHERE ss.run_id = '${run_id.replace(/'/g, "''")}'
				  AND ss.step_kind = 'aggregate'
				  AND json_extract(ss.output_json, '$.score') IS NOT NULL
				  AND s.indra_belief IS NOT NULL
			),
			grouped AS (
				SELECT
				   'all' AS family,
				   confidence,
				   COUNT(*) AS n,
				   AVG(score) AS mean_score,
				   AVG(indra_belief) AS mean_indra_belief,
				   AVG(ABS(score - indra_belief)) AS mae,
				   AVG(score - indra_belief) AS bias
				FROM aggregate_rows
				GROUP BY confidence
				UNION ALL
				SELECT
				   'indra_type:' || indra_type AS family,
				   confidence,
				   COUNT(*) AS n,
				   AVG(score) AS mean_score,
				   AVG(indra_belief) AS mean_indra_belief,
				   AVG(ABS(score - indra_belief)) AS mae,
				   AVG(score - indra_belief) AS bias
				FROM aggregate_rows
				GROUP BY indra_type, confidence
				HAVING COUNT(*) >= 2
			)
			SELECT family, confidence, n, mean_score, mean_indra_belief, mae, bias
			FROM grouped
			ORDER BY CASE family WHEN 'all' THEN 0 ELSE 1 END, family, confidence`
		)
	] as const;
	const latestSettled = await Promise.allSettled(latestValidityReads);
	const latestRejected = latestSettled.find((r): r is PromiseRejectedResult => r.status === 'rejected');
	if (latestRejected) throw latestRejected.reason;
	const [verdictsRaw, mae, rmse, bias, calN, consistency, supports, byIndraTypeRaw, bySourceApiRaw, truthPresentRaw, confidenceCalibrationRaw] =
		latestSettled.map((r) => (r as PromiseFulfilledResult<unknown>).value) as [
			Array<{ verdict: string; n: number }>,
			Array<{ value: number }>,
			Array<{ value: number }>,
			Array<{ value: number }>,
			Array<{ n: number }>,
			Array<{ value: number; n: number }>,
			Array<{ value: number; unavailable_reason: string | null }>,
			Array<Omit<StratumRow, 'cohort_href' | 'applicability'>>,
			Array<Omit<StratumRow, 'cohort_href' | 'applicability'>>,
			TruthPresentRaw[],
			Array<Omit<ConfidenceCalibrationRow, 'cohort_href' | 'applicability'>>
		];

	const cleanVal = (rs: { value: number }[]) => {
		if (rs.length === 0) return null;
		const v = rs[0].value;
		return typeof v === 'number' && !Number.isNaN(v) ? v : null;
	};
	const jsonStringArray = (raw: string | null | undefined): string[] => {
		if (!raw) return [];
		try {
			const parsed = JSON.parse(raw) as unknown;
			return Array.isArray(parsed)
				? parsed.filter((v): v is string => typeof v === 'string')
				: [];
		} catch {
			return [];
		}
	};
	const verdicts = verdictsRaw.map((r) => ({
		...r,
		cohort_href: cohortHref(run_id, { verdict: r.verdict })
	}));
	const byIndraType = byIndraTypeRaw.map((r) => ({
		...r,
		cohort_href: cohortHref(run_id, { grain: 'statement', type: r.value, indra_belief_present: true }),
		applicability: 'arch_blind' as PanelApplicabilityKind
	}));
	const bySourceApi = bySourceApiRaw.map((r) => ({
		...r,
		cohort_href: cohortHref(run_id, { grain: 'statement', source_stratum: r.value, indra_belief_present: true }),
		applicability: 'arch_blind' as PanelApplicabilityKind
	}));
	const truthPresent = truthPresentRaw.map(({ gold_fields_json, ...r }) => ({
		...r,
		gold_fields: jsonStringArray(gold_fields_json),
		cohort_href: cohortHref(run_id, {
			truth_set: r.truth_set_id,
			step_kind: r.step_kind,
			verdict_present: true
		}),
		truth_href: `/runs/${run_id}/truth-sets/${encodeURIComponent(r.truth_set_id)}?step_kind=${encodeURIComponent(r.step_kind)}`,
		applicability: 'arch_blind' as PanelApplicabilityKind
	}));
	const confidenceCalibration = confidenceCalibrationRaw.map((r) => {
		const typeFilter = r.family.startsWith('indra_type:')
			? r.family.slice('indra_type:'.length)
			: null;
		return {
			...r,
			cohort_href: cohortHref(run_id, {
				confidence: r.confidence,
				type: typeFilter,
				score_present: true,
				indra_belief_present: true
			}),
			applicability: 'arch_blind' as PanelApplicabilityKind
		};
	});
	const consistencyRow = consistency[0] ?? null;
	const supportsValue = cleanVal(supports);
	const supportsReason = supports[0]?.unavailable_reason ?? null;

	return {
		run_id,
		scorer_version,
		architecture,
		verdicts,
		calibration: {
			mae: cleanVal(mae),
			rmse: cleanVal(rmse),
			bias: cleanVal(bias),
			n_stmts: calN[0]?.n ?? null,
			cohort_href: cohortHref(run_id, { grain: 'statement', indra_belief_present: true }),
			applicability: 'arch_blind'
		},
		inter_evidence_consistency: {
			mean_stdev: consistencyRow && !Number.isNaN(consistencyRow.value)
				? consistencyRow.value
				: null,
			n_multi_ev: consistencyRow?.n ?? null,
			cohort_href: consistencyRow && !Number.isNaN(consistencyRow.value)
				? cohortHref(run_id, { grain: 'statement', multi_evidence: true })
				: null,
			applicability: consistencyRow && !Number.isNaN(consistencyRow.value)
				? 'arch_blind'
				: 'not_defined',
			not_defined_reason: consistencyRow && !Number.isNaN(consistencyRow.value)
				? null
				: 'needs statements with more than one scored evidence'
		},
		supports_graph_delta: supportsValue,
		supports_graph_href: supportsValue == null ? null : cohortHref(run_id, { grain: 'statement', supports_compare: true }),
		supports_graph_not_defined_reason: supportsValue == null
			? (supportsReason ?? 'needs both supports-rich and non-supports evidence buckets')
			: null,
		byIndraType,
		bySourceApi,
		truthPresent,
		confidenceCalibration,
		metricTaxonomy: [
			{
				panel: 'verdict distribution',
				applicability: 'arch_blind',
				reason: 'aggregate verdict fields exist for both scorer architectures; cohorts stay scoped to one run architecture',
				cohort_href: cohortHref(run_id, { verdict_present: true })
			},
			{
				panel: 'gold-pool comparison',
				applicability: 'arch_blind',
				reason: 'evidence-level verdict/tag gold applies to aggregate rows from either architecture; values stay scoped to one run',
				cohort_href: cohortHref(run_id)
			},
			{
				panel: 'INDRA prior calibration',
				applicability: 'arch_blind',
				reason: 'mean score residuals can be computed for any run with INDRA belief anchors',
				cohort_href: cohortHref(run_id, { grain: 'statement', indra_belief_present: true })
			},
			{
				panel: 'confidence calibration by family',
				applicability: 'arch_blind',
				reason: 'confidence buckets are aggregate fields on both architectures and are measured separately inside the selected architecture',
				cohort_href: cohortHref(run_id, { score_present: true, indra_belief_present: true })
			},
			{
				panel: 'decomposed probe health',
				applicability: architecture === 'decomposed' ? 'arch_conditioned' : 'not_defined',
				reason: architecture === 'decomposed'
					? 'probe rows are native to decomposed scoring'
					: 'monolithic scoring does not emit decomposed probe rows',
				cohort_href: architecture === 'decomposed' ? cohortHref(run_id, { trace_fidelity: 'native_decomposed' }) : null
			},
			{
				panel: 'paired deltas',
				applicability: pairedGroup ? 'paired_only' : 'not_defined',
				reason: pairedGroup
					? `architecture deltas require overlap accounting inside paired group ${pairedGroup}`
					: 'architecture deltas require an overlap-first paired workbench',
				cohort_href: pairedGroup ? `/pairs/${pairedGroup}` : null
			}
		]
	};
}

async function scalar(con: DuckDBConnection, sql: string): Promise<number> {
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

async function readScalar(con: DuckDBConnection, sql: string): Promise<number> {
	const reader = await con.runAndReadAll(sql);
	const rows = reader.getRowObjects();
	if (rows.length === 0) return 0;
	const v = Object.values(rows[0])[0];
	return typeof v === 'bigint' ? Number(v) : Number(v ?? 0);
}

async function tableExists(con: DuckDBConnection, tableName: string): Promise<boolean> {
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
function normalizeDuckValue(v: unknown): unknown {
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
async function rows<T = Record<string, unknown>>(
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

async function repairRerunLineageSqlOptions(
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

function sqlQuote(s: string): string {
	return s.replace(/'/g, "''");
}

function timestampMs(raw: string | null | undefined): number | null {
	if (!raw) return null;
	const normalized = raw.includes('T') ? raw : raw.replace(' ', 'T');
	const ms = Date.parse(normalized);
	return Number.isFinite(ms) ? ms : null;
}

function durationSeconds(started_at: string | null | undefined, finished_at: string | null | undefined): number | null {
	const start = timestampMs(started_at);
	const finish = timestampMs(finished_at);
	if (start == null || finish == null || finish < start) return null;
	return (finish - start) / 1000;
}

function safeRate(numerator: number | null | undefined, denominator: number | null | undefined): number | null {
	if (numerator == null || denominator == null || denominator <= 0 || Number.isNaN(numerator)) return null;
	return numerator / denominator;
}

// Strict row helper: SQL/projection failures reject and should surface through
// SvelteKit error handling when the UI would otherwise tell a false absence.
async function readRows<T = Record<string, unknown>>(
	con: DuckDBConnection,
	sql: string
): Promise<T[]> {
	const reader = await con.runAndReadAll(sql);
	return reader.getRowObjects().map((row) => {
		const out: Record<string, unknown> = {};
		for (const [k, v] of Object.entries(row)) {
			out[k] = normalizeDuckValue(v);
		}
		return out as T;
	});
}

async function assertCoreOverviewSchema(con: DuckDBConnection): Promise<void> {
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

function truthSetOverlapCte(run_id: string, truth_set_id: string, step_kind: string): string {
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

export async function getTruthSetOverlapDetail(
	run_id: string,
	truth_set_id: string,
	step_kind = 'aggregate'
): Promise<TruthSetOverlapDetail | null> {
	if (!dbExists()) return null;
	const con = await connect();
	try {
		const qr = sqlQuote(run_id);
		const qt = sqlQuote(truth_set_id);
		const runRows = await readRows<{
			run_id: string;
			architecture: string | null;
			status: string;
		}>(
			con,
			`SELECT run_id, architecture, status
			 FROM score_run
			 WHERE run_id='${qr}'`
		);
		if (runRows.length === 0) return null;
		const truthRows = await readRows<{ id: string; name: string | null }>(
			con,
			`SELECT id, name FROM truth_set WHERE id='${qt}'`
		);
		if (truthRows.length === 0) return null;

		const cte = truthSetOverlapCte(run_id, truth_set_id, step_kind);
		const summaryRows = await readRows<{
			n_gold_labels: number;
			n_applicable_gold_labels: number;
			n_metric_compared_rows: number;
			n_scored_evidences: number;
			n_compared_labels: number;
			n_no_aggregate_verdict: number;
			n_context_mismatch: number;
			n_not_scored_in_run: number;
			n_not_in_corpus: number;
			gold_fields_json: string | null;
		}>(
			con,
			`${cte}
			SELECT
			  COALESCE((SELECT COUNT(*) FROM classified_labels), 0) AS n_gold_labels,
			  COALESCE((SELECT SUM(CASE WHEN matched_verdict_rows > 0 THEN 1 ELSE 0 END) FROM classified_labels), 0) AS n_applicable_gold_labels,
			  COALESCE((SELECT COUNT(*) FROM selected_gold), 0) AS n_metric_compared_rows,
			  COALESCE((SELECT COUNT(*) FROM scorer_verdicts), 0) AS n_scored_evidences,
			  COALESCE((SELECT SUM(CASE WHEN status = 'compared' THEN 1 ELSE 0 END) FROM classified_labels), 0) AS n_compared_labels,
			  COALESCE((SELECT SUM(CASE WHEN status = 'no_aggregate_verdict' THEN 1 ELSE 0 END) FROM classified_labels), 0) AS n_no_aggregate_verdict,
			  COALESCE((SELECT SUM(CASE WHEN status = 'context_mismatch' THEN 1 ELSE 0 END) FROM classified_labels), 0) AS n_context_mismatch,
			  COALESCE((SELECT SUM(CASE WHEN status = 'not_scored_in_run' THEN 1 ELSE 0 END) FROM classified_labels), 0) AS n_not_scored_in_run,
			  COALESCE((SELECT SUM(CASE WHEN status = 'not_in_corpus' THEN 1 ELSE 0 END) FROM classified_labels), 0) AS n_not_in_corpus,
			  COALESCE(
			    (SELECT to_json(list(DISTINCT gold_field ORDER BY gold_field))::VARCHAR FROM selected_gold),
			    (SELECT COALESCE(to_json(list(DISTINCT field ORDER BY field))::VARCHAR, '[]') FROM classified_labels),
			    '[]'
			  ) AS gold_fields_json`
		);
		const detailRows = await readRows<Omit<TruthSetOverlapRow, 'statement_href'> & { status: TruthSetOverlapStatus }>(
			con,
			`${cte},
			ranked_labels AS (
				SELECT *,
				       row_number() OVER (
				         PARTITION BY status
				         ORDER BY field, target_id, relation_target_id NULLS LAST
				       ) AS status_rank
				FROM classified_labels
			)
			SELECT label_id,
			       target_id,
			       relation_target_id,
			       field,
			       value_text,
			       provenance,
			       corpus_stmt_hash,
			       source_api,
			       evidence_text,
			       same_evidence_scored_rows,
			       matched_step_rows,
			       matched_verdict_rows,
			       matched_no_verdict_rows,
			       sample_matched_stmt_hash,
			       sample_our_verdict,
			       sample_scored_stmt_hash,
			       status
			FROM ranked_labels
			WHERE status_rank <= ${TRUTH_SET_OVERLAP_ROWS_PER_STATUS}
			ORDER BY CASE status
			           WHEN 'compared' THEN 0
			           WHEN 'no_aggregate_verdict' THEN 1
			           WHEN 'context_mismatch' THEN 2
			           WHEN 'not_scored_in_run' THEN 3
			           ELSE 4
			         END,
			         field,
			         target_id,
			         relation_target_id NULLS LAST
			LIMIT ${TRUTH_SET_OVERLAP_ROW_LIMIT}`
		);
		const s = summaryRows[0] ?? {
			n_gold_labels: 0,
			n_applicable_gold_labels: 0,
			n_metric_compared_rows: 0,
			n_scored_evidences: 0,
			n_compared_labels: 0,
			n_no_aggregate_verdict: 0,
			n_context_mismatch: 0,
			n_not_scored_in_run: 0,
			n_not_in_corpus: 0,
			gold_fields_json: '[]'
		};
		let goldFields: string[] = [];
		try {
			const parsed = JSON.parse(s.gold_fields_json ?? '[]') as unknown;
			goldFields = Array.isArray(parsed)
				? parsed.filter((v): v is string => typeof v === 'string')
				: [];
		} catch {
			goldFields = [];
		}
		return {
			run_id,
			truth_set_id,
			truth_set_name: truthRows[0].name ?? truth_set_id,
			architecture: runRows[0].architecture ?? 'unknown',
			status: runRows[0].status,
			metric_step_kind: step_kind,
			n_gold_labels: Number(s.n_gold_labels ?? 0),
			n_applicable_gold_labels: Number(s.n_applicable_gold_labels ?? 0),
			n_metric_compared_rows: Number(s.n_metric_compared_rows ?? 0),
			n_scored_evidences: Number(s.n_scored_evidences ?? 0),
			n_compared_labels: Number(s.n_compared_labels ?? 0),
			n_no_aggregate_verdict: Number(s.n_no_aggregate_verdict ?? 0),
			n_context_mismatch: Number(s.n_context_mismatch ?? 0),
			n_not_scored_in_run: Number(s.n_not_scored_in_run ?? 0),
			n_not_in_corpus: Number(s.n_not_in_corpus ?? 0),
			positive_gold_label: 'correct',
			negative_gold_rule: "any value != 'correct'",
			gold_fields: goldFields,
			cohort_href: step_kind === 'aggregate' && Number(s.n_metric_compared_rows ?? 0) > 0
				? cohortHref(run_id, { truth_set: truth_set_id, step_kind, verdict_present: true })
				: null,
			rows: detailRows.map((r) => ({
				...r,
				label_id: Number(r.label_id),
				same_evidence_scored_rows: Number(r.same_evidence_scored_rows ?? 0),
				matched_step_rows: Number(r.matched_step_rows ?? 0),
				matched_verdict_rows: Number(r.matched_verdict_rows ?? 0),
				matched_no_verdict_rows: Number(r.matched_no_verdict_rows ?? 0),
				statement_href: r.sample_matched_stmt_hash
					? `/statements/${r.sample_matched_stmt_hash}?run_id=${run_id}`
					: r.corpus_stmt_hash
						? `/statements/${r.corpus_stmt_hash}?run_id=${run_id}`
						: null
			})),
			row_limit: TRUTH_SET_OVERLAP_ROW_LIMIT,
			status_row_limit: TRUTH_SET_OVERLAP_ROWS_PER_STATUS
		};
	} finally {
		con.disconnectSync?.();
	}
}

export async function getStatementMatrix(): Promise<StatementMatrixRow[]> {
	if (!dbExists()) return [];
	const con = await connect();
	try {
		// Latest succeeded run, used to surface our_belief + belief_delta
		const latestRun = await rows<{ run_id: string }>(
			con,
			`SELECT run_id FROM score_run
			 WHERE status = 'succeeded' ORDER BY started_at DESC LIMIT 1`
		);
		const runId = latestRun[0]?.run_id ?? null;
		const runJoin = runId
			? `LEFT JOIN (
					SELECT stmt_hash,
					       AVG(CAST(json_extract(output_json, '$.score') AS DOUBLE)) AS our_belief
					FROM scorer_step
					WHERE run_id = '${runId.replace(/'/g, "''")}'
					  AND step_kind = 'aggregate'
					  AND json_extract(output_json, '$.score') IS NOT NULL
					GROUP BY stmt_hash
				) ours ON ours.stmt_hash = s.stmt_hash`
			: 'LEFT JOIN (SELECT NULL::VARCHAR AS stmt_hash, NULL::DOUBLE AS our_belief WHERE FALSE) ours ON FALSE';

		const MATRIX_LIMIT = 50_000;
		const matrixRows = await rows<StatementMatrixRow>(
			con,
			`SELECT
			   s.stmt_hash,
			   ${runId ? `'${runId.replace(/'/g, "''")}'` : 'NULL'} AS run_id,
			   s.indra_type,
			   s.indra_belief,
			   COALESCE(string_agg(DISTINCT a.name, ', ' ORDER BY a.name), '') AS agent_names,
			   COUNT(DISTINCT e.evidence_hash) AS n_evidences,
			   s.supports_count,
			   s.supported_by_count,
			   COALESCE(string_agg(DISTINCT e.source_api, ',' ORDER BY e.source_api), '') AS source_apis,
			   s.source_dump_id,
			   MAX(CASE WHEN e.is_curated THEN 1 ELSE 0 END) AS is_curated_any,
			   ours.our_belief AS our_belief,
			   CASE WHEN ours.our_belief IS NOT NULL AND s.indra_belief IS NOT NULL
			        THEN ours.our_belief - s.indra_belief
			        ELSE NULL END AS belief_delta
			 FROM statement s
			 LEFT JOIN statement_evidence se ON se.stmt_hash = s.stmt_hash
			 LEFT JOIN evidence e ON e.evidence_hash = se.evidence_hash
			 LEFT JOIN agent a ON a.stmt_hash = s.stmt_hash
			 ${runJoin}
			 GROUP BY s.stmt_hash, s.indra_type, s.indra_belief,
			          s.supports_count, s.supported_by_count, s.source_dump_id,
			          ours.our_belief
			 ORDER BY ABS(COALESCE(ours.our_belief - s.indra_belief, 0)) DESC, s.stmt_hash
			 LIMIT ${MATRIX_LIMIT}`
		);
		// Warn at the ceiling — silent truncation lies about coverage. At
		// 50K, a corpus larger than rasmachine has been loaded and the
		// matrix should grow server-side pagination rather than truncate.
		if (matrixRows.length >= MATRIX_LIMIT) {
			console.warn(
				`getStatementMatrix: hit LIMIT ${MATRIX_LIMIT} — corpus ` +
				`exceeds matrix ceiling, results truncated. Add server-side ` +
				`pagination if this becomes routine.`
			);
		}
		return matrixRows;
	} finally {
		con.disconnectSync?.();
	}
}

export async function getRunCohort(
	run_id: string,
	filters: RunCohortFilters = {}
): Promise<RunCohort | null> {
	if (!dbExists()) return null;
	const con = await connect();
	try {
		const qr = sqlQuote(run_id);
		const runRows = await rows<{ architecture: string | null; status: string | null }>(
			con,
			`SELECT architecture, status FROM score_run WHERE run_id='${qr}'`
		);
		if (runRows.length === 0) return null;
		await validateRunCohortFilterValues(con, run_id, filters);
		const architecture = runRows[0].architecture ?? 'unknown';
		const status = runRows[0].status ?? 'unknown';
			const traceState = cleanTraceStateFilter(filters.trace_state);
			if (traceState === INVALID_TRACE_STATE) {
				throw error(400, {
					code: 'invalid_trace_state',
					message: `invalid trace_state: ${filters.trace_state}`
				});
			}
			const probeCoverageCohort = filters.probe_coverage === 'present';
			const tracePlaneCohort = Boolean(traceState || probeCoverageCohort);
			const traceSnapshot = cleanTraceSnapshot(filters.trace_snapshot);
			if (traceSnapshot === INVALID_TRACE_SNAPSHOT) {
				throw error(400, {
					code: 'invalid_trace_snapshot',
					message: `invalid trace_snapshot: ${filters.trace_snapshot}`
				});
			}
			const grain: 'evidence' | 'statement' = filters.grain === 'statement' && !tracePlaneCohort ? 'statement' : 'evidence';
			const canonicalTraceSnapshot = tracePlaneCohort
				? await resolveTraceSnapshotStartedAt(con, run_id, traceSnapshot, status === 'running')
				: null;
			const cohortFilters: RunCohortFilters = tracePlaneCohort
				? { ...filters, grain: 'evidence', trace_snapshot: canonicalTraceSnapshot }
				: filters;

		if (grain === 'statement') {
			const where = nonTraceStatementCohortWhereClauses(filters);
			const stmtCte = statementCohortCte(run_id, architecture, filters);
			const whereSql = where.join(' AND ');
			const totalRows = await scalar(con, `${stmtCte} SELECT COUNT(*) FROM stmt_scores WHERE ${whereSql}`);
			const limit = RUN_COHORT_LIMIT;
			const cohortRows = await rows<RunCohortRow>(
				con,
				statementCohortRowsSql(stmtCte, whereSql, limit)
			);
			const statementEmpty = statementEmptyPredicates(cohortFilters);
			const emptyDiagnostics = totalRows === 0
				? await emptyCohortDiagnostics(
					con,
					`${stmtCte}, cohort_base AS (SELECT * FROM stmt_scores)`,
					'statement',
					cohortFilters,
					statementEmpty.predicates,
					statementEmpty.diagnostics,
					'this run',
					status,
					await runScorerStepCounts(con, run_id)
				)
				: [];
			return {
				run_id,
				architecture,
				status,
				filters: cohortFilters,
				grain,
				rows: cohortRows,
				totalRows,
				limit,
				emptyDiagnostics
			};
		}

			if (tracePlaneCohort) {
				const where = traceStateCohortWhereClauses(cohortFilters, canonicalTraceSnapshot);

			const traceCte = traceCohortCte(run_id, architecture, status, canonicalTraceSnapshot);
			const whereSql = where.join(' AND ');
			const totalRows = await scalar(
				con,
				traceCohortCountSql(traceCte, whereSql)
			);
			const limit = RUN_COHORT_LIMIT;
			const cohortRows = await rows<RunCohortRow>(
				con,
				traceCohortRowsSql(traceCte, architecture, whereSql, limit)
			);
			const traceEmpty = traceEvidenceEmptyPredicates(cohortFilters, canonicalTraceSnapshot);
			const emptyDiagnostics = totalRows === 0
				? await emptyCohortDiagnostics(
					con,
					traceCohortBaseSql(traceCte),
					'trace evidence',
					cohortFilters,
					traceEmpty.predicates,
					traceEmpty.diagnostics,
					canonicalTraceSnapshot ? 'this run snapshot' : 'this run',
					status,
					await runScorerStepCounts(con, run_id)
				)
				: [];
			return {
				run_id,
				architecture,
				status,
				filters: cohortFilters,
				grain,
				rows: cohortRows,
				totalRows,
				limit,
				emptyDiagnostics
			};
		}

		const where = nonTraceEvidenceCohortWhereClauses(run_id, filters);

		const whereSql = where.join(' AND ');
		const totalRows = await scalar(
			con,
			aggregateEvidenceCohortCountSql(whereSql)
		);
		const limit = RUN_COHORT_LIMIT;
		const cohortRows = await rows<RunCohortRow>(
			con,
			aggregateEvidenceCohortRowsSql(architecture, whereSql, limit)
		);
		const aggregateEmpty = aggregateEvidenceEmptyPredicates(cohortFilters);
		const emptyDiagnostics = totalRows === 0
			? await emptyCohortDiagnostics(
				con,
				aggregateEvidenceCohortBaseSql(run_id),
				'aggregate evidence',
				cohortFilters,
				aggregateEmpty.predicates,
				aggregateEmpty.diagnostics,
				'this run',
				status,
				await runScorerStepCounts(con, run_id)
			)
			: [];

		return {
			run_id,
			architecture,
			status,
			filters: cohortFilters,
			grain,
			rows: cohortRows,
			totalRows,
			limit,
			emptyDiagnostics
		};
	} finally {
		con.disconnectSync?.();
	}
}

const REPAIR_RECOVERY_PAGE_LIMIT = 20;

function boundedNonNegativeInteger(value: number | null | undefined, fallback: number): number {
	const n = Number(value);
	return Number.isInteger(n) && n >= 0 ? n : fallback;
}

export async function getRunRepairBacklog(
	run_id: string,
	options: RunRepairBacklogOptions = {}
): Promise<RunRepairBacklog | null> {
	if (!dbExists()) return null;
	const recoveryOffset = boundedNonNegativeInteger(options.recoveryOffset, 0);
	const recoveryLimit = Math.min(
		100,
		Math.max(1, boundedNonNegativeInteger(options.recoveryLimit, REPAIR_RECOVERY_PAGE_LIMIT))
	);
	const con = await connect();
	try {
		const qr = sqlQuote(run_id);
		const runRows = await rows<{ architecture: string | null }>(
			con,
			`SELECT architecture FROM score_run WHERE run_id='${qr}'`
		);
		if (runRows.length === 0) return null;
		const architecture = runRows[0].architecture ?? 'unknown';
		const reruns = await getRepairRerunComparisons(con, run_id, architecture);
		const hasCorrectionTable = await scalar(
			con,
			`SELECT COUNT(*)
			   FROM information_schema.tables
			  WHERE table_name='scorer_step_correction'`
		);
		if (hasCorrectionTable === 0) {
			return {
				run_id,
				architecture,
				openCount: 0,
				rows: [],
				reruns,
				activeRerunIntents: [],
				activeRerunIntentCount: 0,
				recoverableRerunIntents: [],
				recoverableRerunIntentCount: 0,
				recoverableRerunIntentOffset: recoveryOffset,
				recoverableRerunIntentLimit: recoveryLimit,
				limit: RUN_COHORT_LIMIT
			};
		}
		const hasArchitectureColumn = await scalar(
			con,
			`SELECT COUNT(*)
			   FROM information_schema.columns
			  WHERE table_name='scorer_step_correction'
			    AND column_name='architecture'`
		);
		const lineageSqlOptions = await repairRerunLineageSqlOptions(con);
		const correctionArchitectureSql = hasArchitectureColumn > 0 ? 'c.architecture' : 'NULL::VARCHAR';
		const intentArchitectureSql = hasArchitectureColumn > 0 ? 'i.architecture' : `'${sqlQuote(architecture)}'`;
		const recoverableRerunIntentsResult = await getRecoverableRepairRerunIntents(
			con,
			run_id,
			architecture,
			intentArchitectureSql,
			lineageSqlOptions,
			recoveryOffset,
			recoveryLimit
		);
		const activeRerunIntents = await getActiveRepairRerunIntents(
			con,
			run_id,
			architecture,
			intentArchitectureSql,
			lineageSqlOptions
		);

		const availablePredicate = repairCandidateAvailablePredicateSql('c', lineageSqlOptions);
		const slotReviewParentCorrectionIdSql = repairRerunParentCorrectionIdNumberSql('r', lineageSqlOptions);
		const openCount = await scalar(
			con,
			`SELECT COUNT(*)
			   FROM scorer_step_correction c
			  WHERE c.run_id='${qr}'
			    AND c.status='open'
			    AND c.correction_kind='repair_candidate'
			    AND ${availablePredicate}`
		);
		const limit = RUN_COHORT_LIMIT;
		const repairRows = await rows<RunRepairCandidateRow>(
			con,
			`WITH ags AS (
				SELECT stmt_hash,
				       string_agg(name, ', ' ORDER BY
				         CASE role
				           WHEN 'subj' THEN 0 WHEN 'enz' THEN 0
				           WHEN 'obj' THEN 1 WHEN 'sub' THEN 1
				           WHEN 'member' THEN 2 ELSE 3 END,
				         role_index) AS agent_names
				FROM agent
				GROUP BY stmt_hash
			),
			latest_probe_slot_review AS (
				SELECT
				   parent_correction_id,
				   probe_slot_review_slots,
				   probe_slot_review_note,
				   probe_slot_review_reviewer,
				   probe_slot_reviewed_at,
				   probe_slot_review_count
				FROM (
					SELECT
					   ${slotReviewParentCorrectionIdSql} AS parent_correction_id,
					   json_extract_string(r.value_json, '$.selected_probe_slots') AS probe_slot_review_slots,
					   COALESCE(json_extract_string(r.value_json, '$.reviewer_note'), r.note) AS probe_slot_review_note,
					   r.reviewer AS probe_slot_review_reviewer,
					   r.created_at::VARCHAR AS probe_slot_reviewed_at,
					   COUNT(*) OVER (PARTITION BY ${slotReviewParentCorrectionIdSql}) AS probe_slot_review_count,
					   ROW_NUMBER() OVER (
					     PARTITION BY ${slotReviewParentCorrectionIdSql}
					     ORDER BY r.created_at DESC, r.correction_id DESC
					   ) AS rn
					 FROM scorer_step_correction r
					 WHERE r.run_id='${qr}'
					   AND r.correction_kind='probe_slot_review'
				)
				WHERE rn=1
			)
			SELECT
			   c.correction_id,
			   c.step_hash,
			   c.run_id,
			   COALESCE(${correctionArchitectureSql}, ss.architecture, '${sqlQuote(architecture)}') AS architecture,
			   c.stmt_hash,
			   c.evidence_hash,
			   c.correction_kind,
			   c.status,
			   c.reviewer,
			   c.note,
			   CAST(c.value_json AS VARCHAR) AS value_json,
			   c.source_route,
			   CAST(c.source_filters_json AS VARCHAR) AS source_filters_json,
			   c.created_at::VARCHAR AS created_at,
			   ss.step_kind,
			   json_extract_string(c.value_json, '$.suspected_step_kind') AS suspected_step_kind,
			   json_extract_string(c.value_json, '$.observed.probe_coverage') AS probe_coverage,
			   json_extract_string(c.value_json, '$.observed.missing_probe_slots') AS missing_probe_slots,
			   CAST(json_extract(c.value_json, '$.observed.probe_counts.substrate_route') AS BIGINT) AS n_substrate_route,
			   CAST(json_extract(c.value_json, '$.observed.probe_counts.subject_role_probe') AS BIGINT) AS n_subject_role_probe,
			   CAST(json_extract(c.value_json, '$.observed.probe_counts.object_role_probe') AS BIGINT) AS n_object_role_probe,
			   CAST(json_extract(c.value_json, '$.observed.probe_counts.relation_axis_probe') AS BIGINT) AS n_relation_axis_probe,
			   CAST(json_extract(c.value_json, '$.observed.probe_counts.scope_probe') AS BIGINT) AS n_scope_probe,
			   pr.probe_slot_review_slots,
			   pr.probe_slot_review_note,
			   pr.probe_slot_review_reviewer,
			   pr.probe_slot_reviewed_at,
			   COALESCE(pr.probe_slot_review_count, 0) AS probe_slot_review_count,
			   json_extract_string(c.value_json, '$.severity') AS severity,
			   json_extract_string(c.value_json, '$.reviewer_hypothesis') AS reviewer_hypothesis,
			   s.indra_type,
			   COALESCE(ags.agent_names, '') AS agent_names,
			   e.source_api,
			   e.pmid,
			   e.text,
			   COALESCE(json_extract_string(c.value_json, '$.observed.verdict'), json_extract_string(ss.output_json, '$.verdict')) AS verdict,
			   COALESCE(json_extract_string(c.value_json, '$.observed.confidence'), json_extract_string(ss.output_json, '$.confidence')) AS confidence,
			   COALESCE(CAST(json_extract(c.value_json, '$.observed.score') AS DOUBLE), CAST(json_extract(ss.output_json, '$.score') AS DOUBLE)) AS score,
			   COALESCE(
			     CAST(json_extract(c.value_json, '$.observed.residual') AS DOUBLE),
			     CASE
			       WHEN s.indra_belief IS NOT NULL AND json_extract(ss.output_json, '$.score') IS NOT NULL
			       THEN CAST(json_extract(ss.output_json, '$.score') AS DOUBLE) - s.indra_belief
			       ELSE NULL
			     END
			   ) AS residual
			 FROM scorer_step_correction c
			 LEFT JOIN scorer_step ss ON ss.step_hash = c.step_hash
			 LEFT JOIN statement s ON s.stmt_hash = c.stmt_hash
			 LEFT JOIN evidence e ON e.evidence_hash = c.evidence_hash
			 LEFT JOIN ags ON ags.stmt_hash = c.stmt_hash
			 LEFT JOIN latest_probe_slot_review pr ON pr.parent_correction_id=c.correction_id
			 WHERE c.run_id='${qr}'
			   AND c.status='open'
			   AND c.correction_kind='repair_candidate'
			   AND ${availablePredicate}
			 ORDER BY CASE c.status WHEN 'open' THEN 0 ELSE 1 END,
			          c.created_at DESC,
			          c.correction_id DESC
			 LIMIT ${limit}`
		);
		return {
			run_id,
			architecture,
			openCount,
			rows: repairRows,
			reruns,
			activeRerunIntents: activeRerunIntents.rows,
			activeRerunIntentCount: activeRerunIntents.totalCount,
			recoverableRerunIntents: recoverableRerunIntentsResult.rows,
			recoverableRerunIntentCount: recoverableRerunIntentsResult.totalCount,
			recoverableRerunIntentOffset: recoverableRerunIntentsResult.offset,
			recoverableRerunIntentLimit: recoveryLimit,
			limit
		};
	} finally {
		con.disconnectSync?.();
	}
}

async function getActiveRepairRerunIntents(
	con: DuckDBConnection,
	parent_run_id: string,
	parent_architecture: string,
	intentArchitectureSql: string,
	lineageSqlOptions: RepairRerunLineageSqlOptions
): Promise<{ rows: RepairRerunActiveIntent[]; totalCount: number }> {
	const qp = sqlQuote(parent_run_id);
	const qa = sqlQuote(parent_architecture);
	const intentChildRunIdSql = repairRerunChildRunIdSql('i', lineageSqlOptions);
	const intentSourceDumpIdSql = repairRerunSourceDumpIdSql('i', lineageSqlOptions);
	const intentParentCorrectionIdSql = repairRerunParentCorrectionIdNumberSql('i', lineageSqlOptions);
	const toNumberArray = (value: unknown): number[] => Array.isArray(value)
		? value.map((id) => Number(id)).filter((id) => Number.isInteger(id) && id > 0)
		: [];
	const rawRows = await rows<{
		child_run_id: string | null;
		architecture: string | null;
		status: string | null;
		source_dump_id: string | null;
		correction_ids: unknown;
		n_candidates: number;
		has_child_run: number;
		total_groups: number;
		first_intent_at: string;
		last_intent_at: string;
	}>(
		con,
		`WITH active_intent AS (
			 SELECT
			   ${intentChildRunIdSql} AS child_run_id,
			   ${intentSourceDumpIdSql} AS source_dump_id,
			   COALESCE(
			     json_extract_string(i.value_json, '$.architecture'),
			     NULLIF(${intentArchitectureSql}, 'unknown'),
			     sr.architecture,
			     '${qa}'
			   ) AS architecture,
			   ${intentParentCorrectionIdSql} AS parent_correction_id,
			   COALESCE(sr.status, 'queued') AS status,
			   CASE WHEN sr.run_id IS NULL THEN 0 ELSE 1 END AS has_child_run,
			   i.created_at
			  FROM scorer_step_correction i
			  LEFT JOIN score_run sr
			    ON sr.run_id=${intentChildRunIdSql}
			   AND sr.parent_run_id='${qp}'
			 WHERE i.run_id='${qp}'
			   AND i.correction_kind='rerun_intent'
			   AND COALESCE(json_extract_string(i.value_json, '$.parent_run_id'), i.run_id)='${qp}'
			   AND (
			        (
			          sr.run_id IS NOT NULL
			          AND COALESCE(sr.status, 'running') NOT IN (${REPAIR_RERUN_TERMINAL_STATUS_SQL})
			        )
			        OR (
			          sr.run_id IS NULL
			          AND i.created_at >= CURRENT_TIMESTAMP - INTERVAL '${REPAIR_RERUN_QUEUED_INTENT_LOCK_MINUTES} minutes'
			        )
			   )
		),
		active_open AS (
			SELECT ai.*
			  FROM active_intent ai
			  JOIN scorer_step_correction c
			    ON c.correction_id=ai.parent_correction_id
			   AND c.run_id='${qp}'
			   AND c.correction_kind='repair_candidate'
			   AND c.status='open'
			 WHERE ai.child_run_id IS NOT NULL
			   AND ai.source_dump_id IS NOT NULL
			   AND ai.parent_correction_id IS NOT NULL
		)
		,
		grouped AS (
			 SELECT child_run_id,
			        architecture,
			        status,
			        source_dump_id,
			        MAX(has_child_run) AS has_child_run,
			        LIST(parent_correction_id ORDER BY parent_correction_id) AS correction_ids,
			        COUNT(*) AS n_candidates,
			        MIN(created_at)::VARCHAR AS first_intent_at,
			        MAX(created_at)::VARCHAR AS last_intent_at
			   FROM active_open
			  GROUP BY child_run_id, architecture, status, source_dump_id
		)
		 SELECT *,
		        COUNT(*) OVER () AS total_groups
		   FROM grouped
		  ORDER BY last_intent_at DESC, child_run_id, source_dump_id
		  LIMIT 20`
	);
	return {
		rows: rawRows.map((row) => ({
			child_run_id: String(row.child_run_id ?? ''),
			architecture: String(row.architecture ?? parent_architecture),
			status: String(row.status ?? 'running'),
			source_dump_id: String(row.source_dump_id ?? ''),
			correction_ids: toNumberArray(row.correction_ids),
			n_candidates: Number(row.n_candidates ?? 0),
			has_child_run: Number(row.has_child_run ?? 0) > 0,
			first_intent_at: row.first_intent_at,
			last_intent_at: row.last_intent_at
		})),
		totalCount: Number(rawRows[0]?.total_groups ?? 0)
	};
}

async function getRecoverableRepairRerunIntents(
	con: DuckDBConnection,
	parent_run_id: string,
	parent_architecture: string,
	intentArchitectureSql: string,
	lineageSqlOptions: RepairRerunLineageSqlOptions,
	offset: number,
	limit: number
): Promise<{ rows: RepairRerunRecoveryIntent[]; totalCount: number; offset: number }> {
	const qp = sqlQuote(parent_run_id);
	const qa = sqlQuote(parent_architecture);
	const intentChildRunIdSql = repairRerunChildRunIdSql('i', lineageSqlOptions);
	const intentSourceDumpIdSql = repairRerunSourceDumpIdSql('i', lineageSqlOptions);
	const intentParentCorrectionIdSql = repairRerunParentCorrectionIdNumberSql('i', lineageSqlOptions);
	const markerParentCorrectionIdSql = repairRerunParentCorrectionIdStringSql('rc', lineageSqlOptions);
	const markerChildRunIdSql = repairRerunChildRunIdSql('rc', lineageSqlOptions);
	const toNumberArray = (value: unknown): number[] => Array.isArray(value)
		? value.map((id) => Number(id)).filter((id) => Number.isInteger(id) && id > 0)
		: [];
	const baseCte = `WITH intent_raw AS (
		 SELECT
		   ${intentChildRunIdSql} AS child_run_id,
		   ${intentSourceDumpIdSql} AS source_dump_id,
		   COALESCE(
		     json_extract_string(i.value_json, '$.architecture'),
		     NULLIF(${intentArchitectureSql}, 'unknown'),
		     sr.architecture,
		     '${qa}'
		   ) AS architecture,
		   COALESCE(json_extract_string(i.value_json, '$.scoring_mode'), 'aggregate') AS scoring_mode,
		   COALESCE(json_extract_string(i.value_json, '$.probe_step_filter_csv'), '') AS probe_step_filter_csv,
		   ${intentParentCorrectionIdSql} AS parent_correction_id,
		   sr.status,
		   i.created_at
		  FROM scorer_step_correction i
		  JOIN score_run sr
		    ON sr.run_id=${intentChildRunIdSql}
		   AND sr.parent_run_id='${qp}'
		   AND sr.status='succeeded'
		 WHERE i.run_id='${qp}'
		   AND i.correction_kind='rerun_intent'
		   AND COALESCE(json_extract_string(i.value_json, '$.parent_run_id'), i.run_id)='${qp}'
	),
	intent AS (
		SELECT child_run_id,
		       source_dump_id,
		       architecture,
		       scoring_mode,
		       probe_step_filter_csv,
		       parent_correction_id,
		       status,
		       MIN(created_at) AS first_intent_at,
		       MAX(created_at) AS last_intent_at
		  FROM intent_raw
		 WHERE child_run_id IS NOT NULL
		   AND source_dump_id IS NOT NULL
		   AND parent_correction_id IS NOT NULL
		 GROUP BY child_run_id, source_dump_id, architecture, scoring_mode, probe_step_filter_csv, parent_correction_id, status
	),
	missing_marker AS (
		SELECT *
		  FROM intent i
		 WHERE NOT EXISTS (
		     SELECT 1
		       FROM scorer_step_correction rc
		      WHERE (
		             rc.correction_kind='rerun_child'
		          OR (
		             rc.correction_kind='rerun_uncovered'
		             AND ${markerChildRunIdSql}=i.child_run_id
		          )
		        )
		        AND ${markerParentCorrectionIdSql}=CAST(i.parent_correction_id AS VARCHAR)
		   )
	),
	coverage AS (
		SELECT i.*,
		       CASE
		         WHEN i.scoring_mode='probe_only'
		          AND c.evidence_hash IS NOT NULL
		          AND i.probe_step_filter_csv <> ''
		          AND (POSITION('subject_role_probe' IN i.probe_step_filter_csv)=0 OR EXISTS (
		                 SELECT 1 FROM scorer_step child
		                  WHERE child.run_id=i.child_run_id
		                    AND child.step_kind='subject_role_probe'
		                    AND child.stmt_hash=c.stmt_hash
		                    AND child.evidence_hash=c.evidence_hash
		              ))
		          AND (POSITION('object_role_probe' IN i.probe_step_filter_csv)=0 OR EXISTS (
		                 SELECT 1 FROM scorer_step child
		                  WHERE child.run_id=i.child_run_id
		                    AND child.step_kind='object_role_probe'
		                    AND child.stmt_hash=c.stmt_hash
		                    AND child.evidence_hash=c.evidence_hash
		              ))
		          AND (POSITION('relation_axis_probe' IN i.probe_step_filter_csv)=0 OR EXISTS (
		                 SELECT 1 FROM scorer_step child
		                  WHERE child.run_id=i.child_run_id
		                    AND child.step_kind='relation_axis_probe'
		                    AND child.stmt_hash=c.stmt_hash
		                    AND child.evidence_hash=c.evidence_hash
		              ))
		          AND (POSITION('scope_probe' IN i.probe_step_filter_csv)=0 OR EXISTS (
		                 SELECT 1 FROM scorer_step child
		                  WHERE child.run_id=i.child_run_id
		                    AND child.step_kind='scope_probe'
		                    AND child.stmt_hash=c.stmt_hash
		                    AND child.evidence_hash=c.evidence_hash
		              )) THEN 1
		         WHEN i.scoring_mode<>'probe_only' AND c.evidence_hash IS NOT NULL AND cs.step_hash IS NOT NULL THEN 1
		         WHEN c.evidence_hash IS NULL
		          AND i.scoring_mode<>'probe_only'
		          AND EXISTS (
		            SELECT 1 FROM statement_evidence pe
		             WHERE pe.stmt_hash=c.stmt_hash
		          )
		          AND NOT EXISTS (
		            SELECT 1
		              FROM statement_evidence pe
		             WHERE pe.stmt_hash=c.stmt_hash
		               AND NOT EXISTS (
		                 SELECT 1
		                   FROM scorer_step child
		                  WHERE child.run_id=i.child_run_id
		                    AND child.step_kind='aggregate'
		                    AND child.stmt_hash=pe.stmt_hash
		                    AND child.evidence_hash=pe.evidence_hash
		               )
		          ) THEN 1
		         ELSE 0
		       END AS child_covers
		  FROM missing_marker i
		  JOIN scorer_step_correction c
		    ON c.correction_id=i.parent_correction_id
		   AND c.run_id='${qp}'
		   AND c.correction_kind='repair_candidate'
		  LEFT JOIN scorer_step cs
		    ON cs.run_id=i.child_run_id
		   AND cs.step_kind='aggregate'
		   AND cs.stmt_hash=c.stmt_hash
		   AND cs.evidence_hash=c.evidence_hash
	),
	grouped AS (
		SELECT child_run_id,
		       architecture,
		       status,
		       source_dump_id,
		       LIST(parent_correction_id ORDER BY parent_correction_id) FILTER (WHERE child_covers=1) AS correction_ids,
		       LIST(parent_correction_id ORDER BY parent_correction_id) FILTER (WHERE child_covers=0) AS uncovered_correction_ids,
		       SUM(child_covers) AS n_candidates,
		       COUNT(*) AS n_missing_markers,
		       SUM(CASE WHEN child_covers=0 THEN 1 ELSE 0 END) AS n_uncovered_candidates,
		       MIN(first_intent_at)::VARCHAR AS first_intent_at,
		       MAX(last_intent_at)::VARCHAR AS last_intent_at
		  FROM coverage
		 GROUP BY child_run_id, architecture, status, source_dump_id
	)`;
	const rawRows = await rows<{
		child_run_id: string | null;
		architecture: string | null;
		status: string | null;
		source_dump_id: string | null;
		correction_ids: unknown;
		uncovered_correction_ids: unknown;
		n_candidates: number;
		n_missing_markers: number;
		n_uncovered_candidates: number;
		total_groups: number;
		page_offset: number;
		first_intent_at: string;
		last_intent_at: string;
	}>(
		con,
		`${baseCte},
		numbered AS (
			SELECT *,
			       COUNT(*) OVER () AS total_groups,
			       ROW_NUMBER() OVER (
			         ORDER BY last_intent_at DESC, child_run_id, source_dump_id
			       ) AS row_number
			  FROM grouped
		),
		windowed AS (
			SELECT *,
			       CASE
			         WHEN ${offset} >= total_groups
			         THEN CAST(FLOOR((CAST(total_groups AS DOUBLE) - 1) / ${limit}) AS BIGINT) * ${limit}
			         ELSE CAST(FLOOR(CAST(${offset} AS DOUBLE) / ${limit}) AS BIGINT) * ${limit}
			       END AS page_offset
			  FROM numbered
		)
		 SELECT child_run_id,
		       architecture,
		       status,
		       source_dump_id,
		       correction_ids,
		       uncovered_correction_ids,
		       n_candidates,
		       n_missing_markers,
		       n_uncovered_candidates,
		       total_groups,
		       page_offset,
		       first_intent_at,
		       last_intent_at
		  FROM windowed
		 WHERE row_number > page_offset
		   AND row_number <= page_offset + ${limit}
		 ORDER BY row_number`
	);
	const mapped = rawRows
		.map((row) => ({
			child_run_id: String(row.child_run_id ?? ''),
			architecture: String(row.architecture ?? parent_architecture),
			status: String(row.status ?? 'succeeded'),
			source_dump_id: String(row.source_dump_id ?? ''),
			correction_ids: toNumberArray(row.correction_ids),
			uncovered_correction_ids: toNumberArray(row.uncovered_correction_ids),
			n_candidates: Number(row.n_candidates ?? 0),
			n_missing_markers: Number(row.n_missing_markers ?? 0),
			n_uncovered_candidates: Number(row.n_uncovered_candidates ?? 0),
			first_intent_at: row.first_intent_at,
			last_intent_at: row.last_intent_at
		}));
	return {
		rows: mapped,
		totalCount: Number(rawRows[0]?.total_groups ?? 0),
		offset: Number(rawRows[0]?.page_offset ?? 0)
	};
}

async function getRepairRerunComparisons(
	con: DuckDBConnection,
	parent_run_id: string,
	parent_architecture: string
): Promise<RepairRerunComparison[]> {
	const qp = sqlQuote(parent_run_id);
	const hasCorrectionTable = await scalar(
		con,
		`SELECT COUNT(*)
		   FROM information_schema.tables
		  WHERE table_name='scorer_step_correction'`
	);
	const lineageSqlOptions = hasCorrectionTable > 0
		? await repairRerunLineageSqlOptions(con)
		: { typedLineage: false };
	const markerParentCorrectionIdSql = repairRerunParentCorrectionIdNumberSql('m', lineageSqlOptions);
	const markerChildRunIdSql = repairRerunChildRunIdSql('m', lineageSqlOptions);
	const childRows = await rows<{
		run_id: string;
		architecture: string | null;
		status: string;
		started_at: string | null;
		model_id_default: string | null;
		cost_estimate_usd: number | null;
		cost_actual_usd: number | null;
	}>(
		con,
		`SELECT run_id,
		        COALESCE(architecture, 'unknown') AS architecture,
		        status,
		        started_at::VARCHAR AS started_at,
		        model_id_default,
		        cost_estimate_usd,
		        cost_actual_usd
		   FROM score_run
		  WHERE parent_run_id='${qp}'
		  ORDER BY started_at DESC NULLS LAST, run_id
		  LIMIT 20`
	);
	const out: RepairRerunComparison[] = [];
	for (const child of childRows) {
		const childArch = child.architecture ?? 'unknown';
		const base = {
			run_id: child.run_id,
			architecture: childArch,
			status: child.status,
			started_at: child.started_at,
			model_id_default: child.model_id_default,
			cost_estimate_usd: child.cost_estimate_usd == null ? null : Number(child.cost_estimate_usd),
			cost_actual_usd: child.cost_actual_usd == null ? null : Number(child.cost_actual_usd)
		};
		if (childArch !== parent_architecture) {
			out.push({
				...base,
				n_overlap_evidences: 0,
				n_score_evidences: 0,
				n_verdict_evidences: 0,
				parent_mae: null,
				child_mae: null,
				parent_bias: null,
				child_bias: null,
				verdicts_moved_total: 0,
				verdicts_moved_to_correct: 0,
				verdicts_moved_to_incorrect: 0,
				n_candidate_evidences: 0,
				n_parent_aggregate_candidates: 0,
				n_child_covered_candidates: 0,
				n_new_child_aggregate_candidates: 0,
				candidate_lanes: [],
				not_defined_reason: `before/after repair deltas are not defined across architectures (${parent_architecture} vs ${childArch}); use a paired workbench for architecture comparison`
			});
			continue;
		}
		const qc = sqlQuote(child.run_id);
		let candidateSummary = {
			n_candidate_evidences: 0,
			n_parent_aggregate_candidates: 0,
			n_child_covered_candidates: 0,
			n_new_child_aggregate_candidates: 0
		};
		let candidateLanes: RepairRerunCandidateLane[] = [];
		if (hasCorrectionTable > 0) {
			const candidateSummaryRows = await rows<typeof candidateSummary>(
				con,
				`WITH candidate AS (
					SELECT
					   c.correction_id,
					   c.stmt_hash,
					   c.evidence_hash,
					   p.step_hash AS parent_step_hash,
					   child.step_hash AS child_step_hash
					 FROM scorer_step_correction m
					 JOIN scorer_step_correction c
					   ON c.correction_id=${markerParentCorrectionIdSql}
					  AND c.correction_kind='repair_candidate'
					  AND c.run_id='${qp}'
					 LEFT JOIN scorer_step p
					   ON p.run_id='${qp}'
					  AND p.step_kind='aggregate'
					  AND p.stmt_hash=c.stmt_hash
					  AND p.evidence_hash=c.evidence_hash
					 LEFT JOIN scorer_step child
					   ON child.run_id='${qc}'
					  AND child.step_kind='aggregate'
					  AND child.stmt_hash=c.stmt_hash
					  AND child.evidence_hash=c.evidence_hash
					 WHERE m.correction_kind='rerun_child'
					   AND m.run_id='${qp}'
					   AND ${markerChildRunIdSql}='${qc}'
				)
				SELECT
				   COUNT(*) AS n_candidate_evidences,
				   SUM(CASE WHEN parent_step_hash IS NOT NULL THEN 1 ELSE 0 END) AS n_parent_aggregate_candidates,
				   SUM(CASE WHEN child_step_hash IS NOT NULL THEN 1 ELSE 0 END) AS n_child_covered_candidates,
				   SUM(CASE WHEN parent_step_hash IS NULL AND child_step_hash IS NOT NULL THEN 1 ELSE 0 END) AS n_new_child_aggregate_candidates
				 FROM candidate`
			);
			candidateSummary = {
				n_candidate_evidences: Number(candidateSummaryRows[0]?.n_candidate_evidences ?? 0),
				n_parent_aggregate_candidates: Number(candidateSummaryRows[0]?.n_parent_aggregate_candidates ?? 0),
				n_child_covered_candidates: Number(candidateSummaryRows[0]?.n_child_covered_candidates ?? 0),
				n_new_child_aggregate_candidates: Number(candidateSummaryRows[0]?.n_new_child_aggregate_candidates ?? 0)
			};
			const candidateRows = await rows<{
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
				movement_rank: number;
			}>(
				con,
				`WITH candidate AS (
					SELECT
					   c.correction_id,
					   c.stmt_hash,
					   c.evidence_hash,
					   json_extract_string(c.value_json, '$.suspected_step_kind') AS suspected_step_kind,
					   s.indra_type,
					   s.indra_belief,
					   e.source_api,
					   CAST(json_extract(p.output_json, '$.score') AS DOUBLE) AS parent_score,
					   CAST(json_extract(child.output_json, '$.score') AS DOUBLE) AS child_score,
					   json_extract_string(p.output_json, '$.verdict') AS parent_verdict,
					   json_extract_string(child.output_json, '$.verdict') AS child_verdict
					 FROM scorer_step_correction m
					 JOIN scorer_step_correction c
					   ON c.correction_id=${markerParentCorrectionIdSql}
					  AND c.correction_kind='repair_candidate'
					  AND c.run_id='${qp}'
					 LEFT JOIN scorer_step p
					   ON p.run_id='${qp}'
					  AND p.step_kind='aggregate'
					  AND p.stmt_hash=c.stmt_hash
					  AND p.evidence_hash=c.evidence_hash
					 LEFT JOIN scorer_step child
					   ON child.run_id='${qc}'
					  AND child.step_kind='aggregate'
					  AND child.stmt_hash=c.stmt_hash
					  AND child.evidence_hash=c.evidence_hash
					 LEFT JOIN statement s ON s.stmt_hash=c.stmt_hash
					 LEFT JOIN evidence e ON e.evidence_hash=c.evidence_hash
					 WHERE m.correction_kind='rerun_child'
					   AND m.run_id='${qp}'
					   AND ${markerChildRunIdSql}='${qc}'
				),
				scored AS (
					SELECT *,
					   CASE WHEN indra_belief IS NOT NULL AND parent_score IS NOT NULL THEN ABS(parent_score - indra_belief) ELSE NULL END AS parent_abs_error,
					   CASE WHEN indra_belief IS NOT NULL AND child_score IS NOT NULL THEN ABS(child_score - indra_belief) ELSE NULL END AS child_abs_error,
					   CASE
					     WHEN indra_belief IS NOT NULL AND parent_score IS NOT NULL AND child_score IS NOT NULL
					     THEN ABS(child_score - indra_belief) - ABS(parent_score - indra_belief)
					     ELSE NULL
					   END AS abs_error_delta,
					   CASE
					     WHEN parent_score IS NULL AND child_score IS NOT NULL THEN 0
					     WHEN parent_verdict IS NOT NULL AND child_verdict IS NOT NULL AND parent_verdict <> 'correct' AND child_verdict='correct' THEN 1
					     WHEN parent_verdict IS NOT NULL AND child_verdict IS NOT NULL AND parent_verdict <> 'incorrect' AND child_verdict='incorrect' THEN 2
					     WHEN indra_belief IS NOT NULL AND parent_score IS NOT NULL AND child_score IS NOT NULL AND ABS(child_score - indra_belief) < ABS(parent_score - indra_belief) THEN 3
					     WHEN indra_belief IS NOT NULL AND parent_score IS NOT NULL AND child_score IS NOT NULL AND ABS(child_score - indra_belief) > ABS(parent_score - indra_belief) THEN 4
					     ELSE 9
					   END AS movement_rank
					FROM candidate
				)
				SELECT *
				FROM scored
				ORDER BY movement_rank,
				         ABS(COALESCE(abs_error_delta, 0)) DESC,
				         correction_id
				LIMIT 8`
			);
			candidateLanes = candidateRows.map((row) => {
				const parentScore = row.parent_score == null ? null : Number(row.parent_score);
				const childScore = row.child_score == null ? null : Number(row.child_score);
				const parentAbs = row.parent_abs_error == null ? null : Number(row.parent_abs_error);
				const childAbs = row.child_abs_error == null ? null : Number(row.child_abs_error);
				const absDelta = row.abs_error_delta == null ? null : Number(row.abs_error_delta);
				let movement = 'unchanged';
				if (parentScore == null && childScore != null) movement = 'new_child_aggregate';
				else if (row.parent_verdict && row.child_verdict && row.parent_verdict !== 'correct' && row.child_verdict === 'correct') movement = 'verdict_to_correct';
				else if (row.parent_verdict && row.child_verdict && row.parent_verdict !== 'incorrect' && row.child_verdict === 'incorrect') movement = 'verdict_to_incorrect';
				else if (absDelta != null && absDelta < 0) movement = 'score_improved';
				else if (absDelta != null && absDelta > 0) movement = 'score_regressed';
				return {
					correction_id: Number(row.correction_id),
					stmt_hash: String(row.stmt_hash),
					evidence_hash: row.evidence_hash == null ? null : String(row.evidence_hash),
					indra_type: row.indra_type == null ? null : String(row.indra_type),
					source_api: row.source_api == null ? null : String(row.source_api),
					suspected_step_kind: row.suspected_step_kind == null ? null : String(row.suspected_step_kind),
					parent_verdict: row.parent_verdict == null ? null : String(row.parent_verdict),
					child_verdict: row.child_verdict == null ? null : String(row.child_verdict),
					parent_score: parentScore,
					child_score: childScore,
					indra_belief: row.indra_belief == null ? null : Number(row.indra_belief),
					parent_abs_error: parentAbs,
					child_abs_error: childAbs,
					abs_error_delta: absDelta,
					movement
				};
			});
		}
		const metricRows = await rows<{
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
		}>(
			con,
			`WITH overlap AS (
				SELECT
				   p.stmt_hash,
				   p.evidence_hash,
				   CAST(json_extract(p.output_json, '$.score') AS DOUBLE) AS parent_score,
				   CAST(json_extract(c.output_json, '$.score') AS DOUBLE) AS child_score,
				   json_extract_string(p.output_json, '$.verdict') AS parent_verdict,
				   json_extract_string(c.output_json, '$.verdict') AS child_verdict,
				   s.indra_belief
				 FROM scorer_step p
				 JOIN scorer_step c
				   ON c.stmt_hash=p.stmt_hash
				  AND c.evidence_hash=p.evidence_hash
				  AND c.step_kind='aggregate'
				 JOIN statement s ON s.stmt_hash=p.stmt_hash
				 WHERE p.run_id='${qp}'
				   AND c.run_id='${qc}'
				   AND p.step_kind='aggregate'
				   AND p.evidence_hash IS NOT NULL
			)
			SELECT
			   COUNT(*) AS n_overlap_evidences,
			   SUM(CASE WHEN indra_belief IS NOT NULL AND parent_score IS NOT NULL AND child_score IS NOT NULL THEN 1 ELSE 0 END) AS n_score_evidences,
			   SUM(CASE WHEN parent_verdict IS NOT NULL AND child_verdict IS NOT NULL THEN 1 ELSE 0 END) AS n_verdict_evidences,
			   AVG(CASE WHEN indra_belief IS NOT NULL AND parent_score IS NOT NULL AND child_score IS NOT NULL THEN ABS(parent_score - indra_belief) ELSE NULL END) AS parent_mae,
			   AVG(CASE WHEN indra_belief IS NOT NULL AND parent_score IS NOT NULL AND child_score IS NOT NULL THEN ABS(child_score - indra_belief) ELSE NULL END) AS child_mae,
			   AVG(CASE WHEN indra_belief IS NOT NULL AND parent_score IS NOT NULL AND child_score IS NOT NULL THEN parent_score - indra_belief ELSE NULL END) AS parent_bias,
			   AVG(CASE WHEN indra_belief IS NOT NULL AND parent_score IS NOT NULL AND child_score IS NOT NULL THEN child_score - indra_belief ELSE NULL END) AS child_bias,
			   SUM(CASE WHEN parent_verdict IS NOT NULL AND child_verdict IS NOT NULL AND parent_verdict <> child_verdict THEN 1 ELSE 0 END) AS verdicts_moved_total,
			   SUM(CASE WHEN parent_verdict IS NOT NULL AND child_verdict IS NOT NULL AND parent_verdict <> 'correct' AND child_verdict='correct' THEN 1 ELSE 0 END) AS verdicts_moved_to_correct,
			   SUM(CASE WHEN parent_verdict IS NOT NULL AND child_verdict IS NOT NULL AND parent_verdict <> 'incorrect' AND child_verdict='incorrect' THEN 1 ELSE 0 END) AS verdicts_moved_to_incorrect
			 FROM overlap`
		);
		const m = metricRows[0] ?? null;
		const nOverlap = Number(m?.n_overlap_evidences ?? 0);
		out.push({
			...base,
			n_overlap_evidences: nOverlap,
			n_score_evidences: Number(m?.n_score_evidences ?? 0),
			n_verdict_evidences: Number(m?.n_verdict_evidences ?? 0),
			parent_mae: m?.parent_mae == null ? null : Number(m.parent_mae),
			child_mae: m?.child_mae == null ? null : Number(m.child_mae),
			parent_bias: m?.parent_bias == null ? null : Number(m.parent_bias),
			child_bias: m?.child_bias == null ? null : Number(m.child_bias),
			verdicts_moved_total: Number(m?.verdicts_moved_total ?? 0),
			verdicts_moved_to_correct: Number(m?.verdicts_moved_to_correct ?? 0),
			verdicts_moved_to_incorrect: Number(m?.verdicts_moved_to_incorrect ?? 0),
			...candidateSummary,
			candidate_lanes: candidateLanes,
			not_defined_reason: nOverlap > 0 || candidateSummary.n_child_covered_candidates > 0
				? null
				: 'before/after repair deltas need overlapping aggregate evidence rows between parent and child run'
		});
	}
	return out;
}

export async function getRepairRerunDetail(
	parent_run_id: string,
	child_run_id: string,
	options: { offset?: number | null; limit?: number | null } = {}
): Promise<RepairRerunDetail | null> {
	if (!dbExists()) return null;
	const offset = boundedNonNegativeInteger(options.offset, 0);
	const limit = Math.min(200, Math.max(1, boundedNonNegativeInteger(options.limit, 100)));
	const con = await connect();
	try {
		const qp = sqlQuote(parent_run_id);
		const qc = sqlQuote(child_run_id);
		const runRows = await rows<{
			parent_architecture: string | null;
			run_id: string;
			architecture: string | null;
			status: string;
			started_at: string | null;
			model_id_default: string | null;
			cost_estimate_usd: number | null;
			cost_actual_usd: number | null;
		}>(
			con,
			`SELECT
			   COALESCE(p.architecture, 'unknown') AS parent_architecture,
			   c.run_id,
			   COALESCE(c.architecture, 'unknown') AS architecture,
			   c.status,
			   c.started_at::VARCHAR AS started_at,
			   c.model_id_default,
			   c.cost_estimate_usd,
			   c.cost_actual_usd
			 FROM score_run p
			 JOIN score_run c
			   ON c.parent_run_id=p.run_id
			  AND c.run_id='${qc}'
			 WHERE p.run_id='${qp}'
			 LIMIT 1`
		);
		const child = runRows[0];
		if (!child) return null;
		const parentArchitecture = child.parent_architecture ?? 'unknown';
		const childArch = child.architecture ?? 'unknown';
		const base = {
			run_id: child.run_id,
			parent_run_id,
			child_run_id: child.run_id,
			parent_architecture: parentArchitecture,
			architecture: childArch,
			status: child.status,
			started_at: child.started_at,
			model_id_default: child.model_id_default,
			cost_estimate_usd: child.cost_estimate_usd == null ? null : Number(child.cost_estimate_usd),
			cost_actual_usd: child.cost_actual_usd == null ? null : Number(child.cost_actual_usd),
			candidate_lane_offset: offset,
			candidate_lane_limit: limit
		};
		if (childArch !== parentArchitecture) {
			return {
				...base,
				n_overlap_evidences: 0,
				n_score_evidences: 0,
				n_verdict_evidences: 0,
				parent_mae: null,
				child_mae: null,
				parent_bias: null,
				child_bias: null,
				verdicts_moved_total: 0,
				verdicts_moved_to_correct: 0,
				verdicts_moved_to_incorrect: 0,
				n_candidate_evidences: 0,
				n_parent_aggregate_candidates: 0,
				n_child_covered_candidates: 0,
				n_new_child_aggregate_candidates: 0,
				candidate_lanes: [],
				candidate_lane_total: 0,
				not_defined_reason: `before/after repair deltas are not defined across architectures (${parentArchitecture} vs ${childArch}); use a paired workbench for architecture comparison`
			};
		}

		let candidateSummary = {
			n_candidate_evidences: 0,
			n_parent_aggregate_candidates: 0,
			n_child_covered_candidates: 0,
			n_new_child_aggregate_candidates: 0
		};
		let candidateLanes: RepairRerunCandidateLane[] = [];
		const hasCorrectionTable = await scalar(
			con,
			`SELECT COUNT(*)
			   FROM information_schema.tables
			  WHERE table_name='scorer_step_correction'`
		);
		if (hasCorrectionTable > 0) {
			const lineageSqlOptions = await repairRerunLineageSqlOptions(con);
			const markerParentCorrectionIdSql = repairRerunParentCorrectionIdNumberSql('m', lineageSqlOptions);
			const markerChildRunIdSql = repairRerunChildRunIdSql('m', lineageSqlOptions);
			const candidateSummaryRows = await rows<typeof candidateSummary>(
				con,
				`WITH candidate AS (
					SELECT
					   c.correction_id,
					   c.stmt_hash,
					   c.evidence_hash,
					   p.step_hash AS parent_step_hash,
					   child.step_hash AS child_step_hash
					 FROM scorer_step_correction m
					 JOIN scorer_step_correction c
					   ON c.correction_id=${markerParentCorrectionIdSql}
					  AND c.correction_kind='repair_candidate'
					  AND c.run_id='${qp}'
					 LEFT JOIN scorer_step p
					   ON p.run_id='${qp}'
					  AND p.step_kind='aggregate'
					  AND p.stmt_hash=c.stmt_hash
					  AND p.evidence_hash=c.evidence_hash
					 LEFT JOIN scorer_step child
					   ON child.run_id='${qc}'
					  AND child.step_kind='aggregate'
					  AND child.stmt_hash=c.stmt_hash
					  AND child.evidence_hash=c.evidence_hash
					 WHERE m.correction_kind='rerun_child'
					   AND m.run_id='${qp}'
					   AND ${markerChildRunIdSql}='${qc}'
				)
				SELECT
				   COUNT(*) AS n_candidate_evidences,
				   SUM(CASE WHEN parent_step_hash IS NOT NULL THEN 1 ELSE 0 END) AS n_parent_aggregate_candidates,
				   SUM(CASE WHEN child_step_hash IS NOT NULL THEN 1 ELSE 0 END) AS n_child_covered_candidates,
				   SUM(CASE WHEN parent_step_hash IS NULL AND child_step_hash IS NOT NULL THEN 1 ELSE 0 END) AS n_new_child_aggregate_candidates
				 FROM candidate`
			);
			candidateSummary = {
				n_candidate_evidences: Number(candidateSummaryRows[0]?.n_candidate_evidences ?? 0),
				n_parent_aggregate_candidates: Number(candidateSummaryRows[0]?.n_parent_aggregate_candidates ?? 0),
				n_child_covered_candidates: Number(candidateSummaryRows[0]?.n_child_covered_candidates ?? 0),
				n_new_child_aggregate_candidates: Number(candidateSummaryRows[0]?.n_new_child_aggregate_candidates ?? 0)
			};
			const candidateRows = await rows<{
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
				movement_rank: number;
			}>(
				con,
				`WITH candidate AS (
					SELECT
					   c.correction_id,
					   c.stmt_hash,
					   c.evidence_hash,
					   json_extract_string(c.value_json, '$.suspected_step_kind') AS suspected_step_kind,
					   s.indra_type,
					   s.indra_belief,
					   e.source_api,
					   CAST(json_extract(p.output_json, '$.score') AS DOUBLE) AS parent_score,
					   CAST(json_extract(child.output_json, '$.score') AS DOUBLE) AS child_score,
					   json_extract_string(p.output_json, '$.verdict') AS parent_verdict,
					   json_extract_string(child.output_json, '$.verdict') AS child_verdict
					 FROM scorer_step_correction m
					 JOIN scorer_step_correction c
					   ON c.correction_id=${markerParentCorrectionIdSql}
					  AND c.correction_kind='repair_candidate'
					  AND c.run_id='${qp}'
					 LEFT JOIN scorer_step p
					   ON p.run_id='${qp}'
					  AND p.step_kind='aggregate'
					  AND p.stmt_hash=c.stmt_hash
					  AND p.evidence_hash=c.evidence_hash
					 LEFT JOIN scorer_step child
					   ON child.run_id='${qc}'
					  AND child.step_kind='aggregate'
					  AND child.stmt_hash=c.stmt_hash
					  AND child.evidence_hash=c.evidence_hash
					 LEFT JOIN statement s ON s.stmt_hash=c.stmt_hash
					 LEFT JOIN evidence e ON e.evidence_hash=c.evidence_hash
					 WHERE m.correction_kind='rerun_child'
					   AND m.run_id='${qp}'
					   AND ${markerChildRunIdSql}='${qc}'
				),
				scored AS (
					SELECT *,
					   CASE WHEN indra_belief IS NOT NULL AND parent_score IS NOT NULL THEN ABS(parent_score - indra_belief) ELSE NULL END AS parent_abs_error,
					   CASE WHEN indra_belief IS NOT NULL AND child_score IS NOT NULL THEN ABS(child_score - indra_belief) ELSE NULL END AS child_abs_error,
					   CASE
					     WHEN indra_belief IS NOT NULL AND parent_score IS NOT NULL AND child_score IS NOT NULL
					     THEN ABS(child_score - indra_belief) - ABS(parent_score - indra_belief)
					     ELSE NULL
					   END AS abs_error_delta,
					   CASE
					     WHEN parent_score IS NULL AND child_score IS NOT NULL THEN 0
					     WHEN parent_verdict IS NOT NULL AND child_verdict IS NOT NULL AND parent_verdict <> 'correct' AND child_verdict='correct' THEN 1
					     WHEN parent_verdict IS NOT NULL AND child_verdict IS NOT NULL AND parent_verdict <> 'incorrect' AND child_verdict='incorrect' THEN 2
					     WHEN indra_belief IS NOT NULL AND parent_score IS NOT NULL AND child_score IS NOT NULL AND ABS(child_score - indra_belief) < ABS(parent_score - indra_belief) THEN 3
					     WHEN indra_belief IS NOT NULL AND parent_score IS NOT NULL AND child_score IS NOT NULL AND ABS(child_score - indra_belief) > ABS(parent_score - indra_belief) THEN 4
					     ELSE 9
					   END AS movement_rank
					FROM candidate
				)
				SELECT *
				FROM scored
				ORDER BY movement_rank,
				         ABS(COALESCE(abs_error_delta, 0)) DESC,
				         correction_id
				LIMIT ${limit}
				OFFSET ${offset}`
			);
			candidateLanes = candidateRows.map((row) => {
				const parentScore = row.parent_score == null ? null : Number(row.parent_score);
				const childScore = row.child_score == null ? null : Number(row.child_score);
				const parentAbs = row.parent_abs_error == null ? null : Number(row.parent_abs_error);
				const childAbs = row.child_abs_error == null ? null : Number(row.child_abs_error);
				const absDelta = row.abs_error_delta == null ? null : Number(row.abs_error_delta);
				let movement = 'unchanged';
				if (parentScore == null && childScore != null) movement = 'new_child_aggregate';
				else if (row.parent_verdict && row.child_verdict && row.parent_verdict !== 'correct' && row.child_verdict === 'correct') movement = 'verdict_to_correct';
				else if (row.parent_verdict && row.child_verdict && row.parent_verdict !== 'incorrect' && row.child_verdict === 'incorrect') movement = 'verdict_to_incorrect';
				else if (absDelta != null && absDelta < 0) movement = 'score_improved';
				else if (absDelta != null && absDelta > 0) movement = 'score_regressed';
				return {
					correction_id: Number(row.correction_id),
					stmt_hash: String(row.stmt_hash),
					evidence_hash: row.evidence_hash == null ? null : String(row.evidence_hash),
					indra_type: row.indra_type == null ? null : String(row.indra_type),
					source_api: row.source_api == null ? null : String(row.source_api),
					suspected_step_kind: row.suspected_step_kind == null ? null : String(row.suspected_step_kind),
					parent_verdict: row.parent_verdict == null ? null : String(row.parent_verdict),
					child_verdict: row.child_verdict == null ? null : String(row.child_verdict),
					parent_score: parentScore,
					child_score: childScore,
					indra_belief: row.indra_belief == null ? null : Number(row.indra_belief),
					parent_abs_error: parentAbs,
					child_abs_error: childAbs,
					abs_error_delta: absDelta,
					movement
				};
			});
		}
		const metricRows = await rows<{
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
		}>(
			con,
			`WITH overlap AS (
				SELECT
				   p.stmt_hash,
				   p.evidence_hash,
				   CAST(json_extract(p.output_json, '$.score') AS DOUBLE) AS parent_score,
				   CAST(json_extract(c.output_json, '$.score') AS DOUBLE) AS child_score,
				   json_extract_string(p.output_json, '$.verdict') AS parent_verdict,
				   json_extract_string(c.output_json, '$.verdict') AS child_verdict,
				   s.indra_belief
				 FROM scorer_step p
				 JOIN scorer_step c
				   ON c.stmt_hash=p.stmt_hash
				  AND c.evidence_hash=p.evidence_hash
				  AND c.step_kind='aggregate'
				 JOIN statement s ON s.stmt_hash=p.stmt_hash
				 WHERE p.run_id='${qp}'
				   AND c.run_id='${qc}'
				   AND p.step_kind='aggregate'
				   AND p.evidence_hash IS NOT NULL
			)
			SELECT
			   COUNT(*) AS n_overlap_evidences,
			   SUM(CASE WHEN indra_belief IS NOT NULL AND parent_score IS NOT NULL AND child_score IS NOT NULL THEN 1 ELSE 0 END) AS n_score_evidences,
			   SUM(CASE WHEN parent_verdict IS NOT NULL AND child_verdict IS NOT NULL THEN 1 ELSE 0 END) AS n_verdict_evidences,
			   AVG(CASE WHEN indra_belief IS NOT NULL AND parent_score IS NOT NULL AND child_score IS NOT NULL THEN ABS(parent_score - indra_belief) ELSE NULL END) AS parent_mae,
			   AVG(CASE WHEN indra_belief IS NOT NULL AND parent_score IS NOT NULL AND child_score IS NOT NULL THEN ABS(child_score - indra_belief) ELSE NULL END) AS child_mae,
			   AVG(CASE WHEN indra_belief IS NOT NULL AND parent_score IS NOT NULL AND child_score IS NOT NULL THEN parent_score - indra_belief ELSE NULL END) AS parent_bias,
			   AVG(CASE WHEN indra_belief IS NOT NULL AND parent_score IS NOT NULL AND child_score IS NOT NULL THEN child_score - indra_belief ELSE NULL END) AS child_bias,
			   SUM(CASE WHEN parent_verdict IS NOT NULL AND child_verdict IS NOT NULL AND parent_verdict <> child_verdict THEN 1 ELSE 0 END) AS verdicts_moved_total,
			   SUM(CASE WHEN parent_verdict IS NOT NULL AND child_verdict IS NOT NULL AND parent_verdict <> 'correct' AND child_verdict='correct' THEN 1 ELSE 0 END) AS verdicts_moved_to_correct,
			   SUM(CASE WHEN parent_verdict IS NOT NULL AND child_verdict IS NOT NULL AND parent_verdict <> 'incorrect' AND child_verdict='incorrect' THEN 1 ELSE 0 END) AS verdicts_moved_to_incorrect
			 FROM overlap`
		);
		const m = metricRows[0] ?? null;
		const nOverlap = Number(m?.n_overlap_evidences ?? 0);
		return {
			...base,
			n_overlap_evidences: nOverlap,
			n_score_evidences: Number(m?.n_score_evidences ?? 0),
			n_verdict_evidences: Number(m?.n_verdict_evidences ?? 0),
			parent_mae: m?.parent_mae == null ? null : Number(m.parent_mae),
			child_mae: m?.child_mae == null ? null : Number(m.child_mae),
			parent_bias: m?.parent_bias == null ? null : Number(m.parent_bias),
			child_bias: m?.child_bias == null ? null : Number(m.child_bias),
			verdicts_moved_total: Number(m?.verdicts_moved_total ?? 0),
			verdicts_moved_to_correct: Number(m?.verdicts_moved_to_correct ?? 0),
			verdicts_moved_to_incorrect: Number(m?.verdicts_moved_to_incorrect ?? 0),
			...candidateSummary,
			candidate_lanes: candidateLanes,
			candidate_lane_total: candidateSummary.n_candidate_evidences,
			not_defined_reason: nOverlap > 0 || candidateSummary.n_child_covered_candidates > 0
				? null
				: 'before/after repair deltas need overlapping aggregate evidence rows between parent and child run'
		};
	} finally {
		con.disconnectSync?.();
	}
}

export async function getPairedWorkbench(pair_id: string): Promise<PairedWorkbench | null> {
	if (!dbExists()) return null;
	const con = await connect();
	try {
		const qp = sqlQuote(pair_id);
		const runsRaw = await rows<PairRunSummary>(
			con,
			`SELECT
			   sr.run_id,
			   COALESCE(sr.architecture, 'unknown') AS architecture,
				   sr.scorer_version,
			   sr.model_id_default,
			   sr.started_at::VARCHAR AS started_at,
			   sr.finished_at::VARCHAR AS finished_at,
			   sr.status,
			   sr.n_stmts,
			   (SELECT COUNT(DISTINCT evidence_hash)
			    FROM scorer_step
			    WHERE run_id=sr.run_id AND step_kind='aggregate' AND evidence_hash IS NOT NULL) AS n_evidences,
			   sr.cost_estimate_usd,
			   sr.cost_actual_usd
			 FROM score_run sr
			 WHERE sr.paired_run_group_id='${qp}'
			 ORDER BY CASE sr.status WHEN 'succeeded' THEN 0 ELSE 1 END,
			          sr.started_at DESC`
		);
		if (runsRaw.length === 0) return null;
		const runs = runsRaw.map((r) => ({
			...r,
			n_stmts: r.n_stmts == null ? null : Number(r.n_stmts),
			n_evidences: Number(r.n_evidences ?? 0),
			finished_at: r.finished_at ?? null,
			duration_s: durationSeconds(r.started_at, r.finished_at),
			cost_estimate_usd: r.cost_estimate_usd == null ? null : Number(r.cost_estimate_usd),
			cost_actual_usd: r.cost_actual_usd == null ? null : Number(r.cost_actual_usd)
		}));
		const pick = (arch: string) =>
			runs.find((r) => r.architecture === arch && r.status === 'succeeded')
			?? runs.find((r) => r.architecture === arch)
			?? null;
		const monolithic = pick('monolithic');
		const decomposed = pick('decomposed');

		const empty: PairedWorkbench = {
			pair_id,
			runs,
			monolithic,
			decomposed,
			overlap: null,
			comparable: null,
			resource_frontier: null,
			denominator_ledger: [],
			exemplars: {
				monolithic_wins: [],
				decomposed_wins: [],
				verdict_disagreements: [],
				mutual_failures: [],
				monolithic_only: [],
				decomposed_only: [],
				excluded_by_integrity: []
			},
			arch_conditioned: {
				monolithic_tiers: [],
				decomposed_probes: []
			},
			not_defined_reason: null
		};
		if (!monolithic || !decomposed) {
			return {
				...empty,
				not_defined_reason: 'paired workbench needs one monolithic run and one decomposed run in this paired_run_group_id'
			};
		}
		if (monolithic.status !== 'succeeded' || decomposed.status !== 'succeeded') {
			return {
				...empty,
				not_defined_reason: 'paired workbench compares only succeeded runs; one side is still running, cancelled, or failed'
			};
		}

		const qm = sqlQuote(monolithic.run_id);
		const qd = sqlQuote(decomposed.run_id);
		const comparableSideCte = (alias: string, runId: string) => `
			${alias}_integrity AS (
				SELECT stmt_hash,
				       evidence_hash,
				       SUM(CASE WHEN step_kind='aggregate' THEN 1 ELSE 0 END) AS n_aggregate,
				       SUM(CASE WHEN step_kind='aggregate'
				                  AND json_extract_string(output_json, '$.verdict') IS NOT NULL THEN 1 ELSE 0 END) AS n_verdict_aggregate,
				       SUM(CASE
				         WHEN step_kind='aggregate'
				              AND (
				                NULLIF(TRIM(COALESCE(error, '')), '') IS NOT NULL
				                OR json_extract(output_json, '$.error') IS NOT NULL
				              ) THEN 1
				         ELSE 0
				       END) AS n_aggregate_step_errors,
				       SUM(CASE
				         WHEN step_kind<>'aggregate'
				              AND (
				                NULLIF(TRIM(COALESCE(error, '')), '') IS NOT NULL
				                OR json_extract(output_json, '$.error') IS NOT NULL
				              ) THEN 1
				         ELSE 0
				       END) AS n_nonaggregate_step_errors
				FROM scorer_step
				WHERE run_id='${runId}' AND evidence_hash IS NOT NULL
				GROUP BY stmt_hash, evidence_hash
			),
			${alias}_aggregate_candidates AS (
				SELECT ss.stmt_hash, ss.evidence_hash,
				       CAST(json_extract(ss.output_json, '$.score') AS DOUBLE) AS score,
				       json_extract_string(ss.output_json, '$.verdict') AS verdict,
				       json_extract_string(ss.output_json, '$.confidence') AS confidence,
				       ss.latency_ms,
				       ss.prompt_tokens,
				       ss.out_tokens,
				       ROW_NUMBER() OVER (
				         PARTITION BY ss.stmt_hash, ss.evidence_hash
				         ORDER BY ss.started_at DESC, ss.step_hash DESC
				       ) AS rn
				FROM scorer_step ss
				JOIN ${alias}_integrity i USING (stmt_hash, evidence_hash)
				WHERE ss.run_id='${runId}'
				  AND ss.step_kind='aggregate'
				  AND ss.evidence_hash IS NOT NULL
				  AND i.n_aggregate_step_errors = 0
				  AND i.n_verdict_aggregate > 0
				  AND NULLIF(TRIM(COALESCE(ss.error, '')), '') IS NULL
				  AND json_extract(ss.output_json, '$.error') IS NULL
				  AND json_extract_string(ss.output_json, '$.verdict') IS NOT NULL
			),
			${alias} AS (
				SELECT stmt_hash, evidence_hash, score, verdict, confidence, latency_ms, prompt_tokens, out_tokens
				FROM ${alias}_aggregate_candidates
				WHERE rn = 1
			)`;
		const pairCte = `WITH
			${comparableSideCte('m', qm)},
			${comparableSideCte('d', qd)},
			j AS (
				SELECT
				   m.stmt_hash,
				   m.evidence_hash,
				   m.score AS monolithic_score,
				   d.score AS decomposed_score,
				   m.verdict AS monolithic_verdict,
				   d.verdict AS decomposed_verdict,
				   m.confidence AS monolithic_confidence,
				   d.confidence AS decomposed_confidence,
				   m.latency_ms AS monolithic_latency_ms,
				   d.latency_ms AS decomposed_latency_ms,
				   m.prompt_tokens AS monolithic_prompt_tokens,
				   d.prompt_tokens AS decomposed_prompt_tokens,
				   m.out_tokens AS monolithic_out_tokens,
				   d.out_tokens AS decomposed_out_tokens
				FROM m JOIN d USING (stmt_hash, evidence_hash)
			)`;
		const rawAggregateCte = (alias: string, runId: string) => `
			${alias}_raw_candidates AS (
				SELECT stmt_hash,
				       evidence_hash,
				       CAST(json_extract(output_json, '$.score') AS DOUBLE) AS score,
				       json_extract_string(output_json, '$.verdict') AS verdict,
				       ROW_NUMBER() OVER (
				         PARTITION BY stmt_hash, evidence_hash
				         ORDER BY started_at DESC, step_hash DESC
				       ) AS rn
				FROM scorer_step
				WHERE run_id='${runId}'
				  AND step_kind='aggregate'
				  AND evidence_hash IS NOT NULL
			),
			${alias}_raw AS (
				SELECT stmt_hash, evidence_hash, score, verdict
				FROM ${alias}_raw_candidates
				WHERE rn = 1
			)`;

		const overlapRows = await rows<PairedOverlapStats>(
			con,
			`WITH
			 ${comparableSideCte('m', qm)},
			 ${comparableSideCte('d', qd)},
			 o AS (
				SELECT m.stmt_hash, m.evidence_hash FROM m JOIN d USING (stmt_hash, evidence_hash)
			 )
			 SELECT
			   (SELECT COUNT(*) FROM m_integrity WHERE n_aggregate > 0) AS monolithic_evidences,
			   (SELECT COUNT(*) FROM d_integrity WHERE n_aggregate > 0) AS decomposed_evidences,
			   (SELECT COUNT(*) FROM m) AS monolithic_comparable_evidences,
			   (SELECT COUNT(*) FROM d) AS decomposed_comparable_evidences,
			   (SELECT COUNT(*) FROM o) AS overlap_evidences,
			   (SELECT COUNT(DISTINCT stmt_hash) FROM o) AS overlap_statements,
			   (SELECT COUNT(*) FROM m LEFT JOIN d USING (stmt_hash, evidence_hash) WHERE d.evidence_hash IS NULL) AS monolithic_only_evidences,
			   (SELECT COUNT(*) FROM d LEFT JOIN m USING (stmt_hash, evidence_hash) WHERE m.evidence_hash IS NULL) AS decomposed_only_evidences,
			   (SELECT COUNT(*) FROM m LEFT JOIN d_integrity di USING (stmt_hash, evidence_hash) WHERE di.evidence_hash IS NULL) AS monolithic_true_nonoverlap_evidences,
			   (SELECT COUNT(*) FROM d LEFT JOIN m_integrity mi USING (stmt_hash, evidence_hash) WHERE mi.evidence_hash IS NULL) AS decomposed_true_nonoverlap_evidences,
			   CASE WHEN (SELECT COUNT(*) FROM m) > 0
			        THEN (SELECT COUNT(*) FROM o)::DOUBLE / (SELECT COUNT(*) FROM m)
			        ELSE NULL END AS monolithic_overlap_pct,
			   CASE WHEN (SELECT COUNT(*) FROM d) > 0
			        THEN (SELECT COUNT(*) FROM o)::DOUBLE / (SELECT COUNT(*) FROM d)
			        ELSE NULL END AS decomposed_overlap_pct,
			   (SELECT COUNT(*) FROM m_integrity WHERE n_aggregate_step_errors > 0) AS monolithic_step_error_evidences,
			   (SELECT COUNT(*) FROM d_integrity WHERE n_aggregate_step_errors > 0) AS decomposed_step_error_evidences,
			   (SELECT COUNT(*) FROM m_integrity WHERE n_nonaggregate_step_errors > 0) AS monolithic_nonaggregate_step_error_evidences,
			   (SELECT COUNT(*) FROM d_integrity WHERE n_nonaggregate_step_errors > 0) AS decomposed_nonaggregate_step_error_evidences,
			   (SELECT COUNT(*) FROM m_integrity WHERE n_aggregate = 0) AS monolithic_missing_aggregate_evidences,
			   (SELECT COUNT(*) FROM d_integrity WHERE n_aggregate = 0) AS decomposed_missing_aggregate_evidences,
			   (SELECT COUNT(*) FROM m_integrity WHERE n_aggregate > 0 AND n_aggregate_step_errors = 0 AND n_verdict_aggregate = 0) AS monolithic_nonverdict_aggregate_evidences,
			   (SELECT COUNT(*) FROM d_integrity WHERE n_aggregate > 0 AND n_aggregate_step_errors = 0 AND n_verdict_aggregate = 0) AS decomposed_nonverdict_aggregate_evidences`
		);
		const toNumberOrNull = (v: unknown): number | null => v == null ? null : Number(v);
		const toNumber = (v: unknown): number => Number(v ?? 0);
		const overlapRaw = overlapRows[0] ?? null;
		const overlap: PairedOverlapStats | null = overlapRaw
			? {
				monolithic_evidences: toNumber(overlapRaw.monolithic_evidences),
				decomposed_evidences: toNumber(overlapRaw.decomposed_evidences),
				monolithic_comparable_evidences: toNumber(overlapRaw.monolithic_comparable_evidences),
				decomposed_comparable_evidences: toNumber(overlapRaw.decomposed_comparable_evidences),
				overlap_evidences: toNumber(overlapRaw.overlap_evidences),
				overlap_statements: toNumber(overlapRaw.overlap_statements),
				monolithic_only_evidences: toNumber(overlapRaw.monolithic_only_evidences),
				decomposed_only_evidences: toNumber(overlapRaw.decomposed_only_evidences),
				monolithic_true_nonoverlap_evidences: toNumber(overlapRaw.monolithic_true_nonoverlap_evidences),
				decomposed_true_nonoverlap_evidences: toNumber(overlapRaw.decomposed_true_nonoverlap_evidences),
				monolithic_overlap_pct: toNumberOrNull(overlapRaw.monolithic_overlap_pct),
				decomposed_overlap_pct: toNumberOrNull(overlapRaw.decomposed_overlap_pct),
				monolithic_step_error_evidences: toNumber(overlapRaw.monolithic_step_error_evidences),
				decomposed_step_error_evidences: toNumber(overlapRaw.decomposed_step_error_evidences),
				monolithic_nonaggregate_step_error_evidences: toNumber(overlapRaw.monolithic_nonaggregate_step_error_evidences),
				decomposed_nonaggregate_step_error_evidences: toNumber(overlapRaw.decomposed_nonaggregate_step_error_evidences),
				monolithic_missing_aggregate_evidences: toNumber(overlapRaw.monolithic_missing_aggregate_evidences),
				decomposed_missing_aggregate_evidences: toNumber(overlapRaw.decomposed_missing_aggregate_evidences),
				monolithic_nonverdict_aggregate_evidences: toNumber(overlapRaw.monolithic_nonverdict_aggregate_evidences),
				decomposed_nonverdict_aggregate_evidences: toNumber(overlapRaw.decomposed_nonverdict_aggregate_evidences)
			}
			: null;

		const comparableRows = await rows<PairedComparableMetrics>(
			con,
			`${pairCte}
			 SELECT
			   COUNT(*) AS n_overlap,
			   SUM(CASE WHEN s.indra_belief IS NOT NULL THEN 1 ELSE 0 END) AS n_truth_overlap,
			   SUM(CASE WHEN monolithic_verdict = decomposed_verdict THEN 1 ELSE 0 END) AS verdict_agreement_n,
			   CASE WHEN COUNT(*) > 0
			        THEN SUM(CASE WHEN monolithic_verdict = decomposed_verdict THEN 1 ELSE 0 END)::DOUBLE / COUNT(*)
			        ELSE NULL END AS verdict_agreement_rate,
			   SUM(CASE WHEN monolithic_verdict='correct' AND decomposed_verdict='correct' THEN 1 ELSE 0 END) AS both_correct_n,
			   SUM(CASE WHEN monolithic_verdict='correct' AND decomposed_verdict <> 'correct' THEN 1 ELSE 0 END) AS monolithic_only_correct_n,
			   SUM(CASE WHEN monolithic_verdict <> 'correct' AND decomposed_verdict='correct' THEN 1 ELSE 0 END) AS decomposed_only_correct_n,
			   SUM(CASE WHEN monolithic_verdict <> 'correct' AND decomposed_verdict <> 'correct' THEN 1 ELSE 0 END) AS both_incorrect_n,
			   AVG(monolithic_score) AS monolithic_score_mean,
			   AVG(decomposed_score) AS decomposed_score_mean,
			   AVG(CASE WHEN s.indra_belief IS NOT NULL AND monolithic_score IS NOT NULL
			            THEN ABS(monolithic_score - s.indra_belief) ELSE NULL END) AS monolithic_mae,
			   AVG(CASE WHEN s.indra_belief IS NOT NULL AND decomposed_score IS NOT NULL
			            THEN ABS(decomposed_score - s.indra_belief) ELSE NULL END) AS decomposed_mae,
			   AVG(CASE WHEN s.indra_belief IS NOT NULL AND monolithic_score IS NOT NULL
			            THEN monolithic_score - s.indra_belief ELSE NULL END) AS monolithic_bias,
			   AVG(CASE WHEN s.indra_belief IS NOT NULL AND decomposed_score IS NOT NULL
			            THEN decomposed_score - s.indra_belief ELSE NULL END) AS decomposed_bias,
			   AVG(CASE WHEN monolithic_score IS NOT NULL AND decomposed_score IS NOT NULL
			            THEN decomposed_score - monolithic_score ELSE NULL END) AS mean_score_delta,
			   AVG(monolithic_latency_ms) AS monolithic_latency_mean_ms,
			   AVG(decomposed_latency_ms) AS decomposed_latency_mean_ms,
			   SUM(CASE WHEN monolithic_latency_ms IS NOT NULL THEN 1 ELSE 0 END) AS monolithic_latency_observed_n,
			   SUM(CASE WHEN decomposed_latency_ms IS NOT NULL THEN 1 ELSE 0 END) AS decomposed_latency_observed_n,
			   SUM(COALESCE(monolithic_prompt_tokens, 0) + COALESCE(monolithic_out_tokens, 0)) AS monolithic_tokens_total,
			   SUM(COALESCE(decomposed_prompt_tokens, 0) + COALESCE(decomposed_out_tokens, 0)) AS decomposed_tokens_total,
			   SUM(CASE WHEN monolithic_prompt_tokens IS NOT NULL OR monolithic_out_tokens IS NOT NULL THEN 1 ELSE 0 END) AS monolithic_tokens_observed_n,
			   SUM(CASE WHEN decomposed_prompt_tokens IS NOT NULL OR decomposed_out_tokens IS NOT NULL THEN 1 ELSE 0 END) AS decomposed_tokens_observed_n
			 FROM j
			 LEFT JOIN statement s ON s.stmt_hash=j.stmt_hash`
		);
		const comparableRaw = comparableRows[0] ?? null;
		const comparableOverlap = toNumber(comparableRaw?.n_overlap);
		const verdictPairRows = comparableOverlap > 0
			? (await rows<PairedVerdictPairRow>(
				con,
				`${pairCte}
				 SELECT
				   monolithic_verdict,
				   decomposed_verdict,
				   CASE
				     WHEN monolithic_verdict='correct' AND decomposed_verdict='correct' THEN 'both_supported'
				     WHEN monolithic_verdict='correct' AND decomposed_verdict <> 'correct' THEN 'monolithic_only'
				     WHEN monolithic_verdict <> 'correct' AND decomposed_verdict='correct' THEN 'decomposed_only'
				     ELSE 'neither_supported'
				   END AS support_cell,
				   COUNT(*) AS n
				 FROM j
				 GROUP BY monolithic_verdict, decomposed_verdict, support_cell
				 ORDER BY
				   CASE
				     WHEN monolithic_verdict='correct' AND decomposed_verdict='correct' THEN 0
				     WHEN monolithic_verdict='correct' AND decomposed_verdict <> 'correct' THEN 1
				     WHEN monolithic_verdict <> 'correct' AND decomposed_verdict='correct' THEN 2
				     ELSE 3
				   END,
				   n DESC,
				   monolithic_verdict,
				   decomposed_verdict`
			)).map((r) => ({
				...r,
				n: toNumber(r.n)
			}))
			: [];
		const comparable: PairedComparableMetrics | null = comparableRaw && comparableOverlap > 0
			? {
				n_overlap: comparableOverlap,
				n_truth_overlap: toNumber(comparableRaw.n_truth_overlap),
				verdict_agreement_n: toNumber(comparableRaw.verdict_agreement_n),
				verdict_agreement_rate: toNumberOrNull(comparableRaw.verdict_agreement_rate),
				both_correct_n: toNumber(comparableRaw.both_correct_n),
				monolithic_only_correct_n: toNumber(comparableRaw.monolithic_only_correct_n),
				decomposed_only_correct_n: toNumber(comparableRaw.decomposed_only_correct_n),
				both_incorrect_n: toNumber(comparableRaw.both_incorrect_n),
				verdict_label_pairs: verdictPairRows,
				monolithic_score_mean: toNumberOrNull(comparableRaw.monolithic_score_mean),
				decomposed_score_mean: toNumberOrNull(comparableRaw.decomposed_score_mean),
				monolithic_mae: toNumberOrNull(comparableRaw.monolithic_mae),
				decomposed_mae: toNumberOrNull(comparableRaw.decomposed_mae),
				monolithic_bias: toNumberOrNull(comparableRaw.monolithic_bias),
				decomposed_bias: toNumberOrNull(comparableRaw.decomposed_bias),
				mean_score_delta: toNumberOrNull(comparableRaw.mean_score_delta),
				monolithic_latency_mean_ms: toNumberOrNull(comparableRaw.monolithic_latency_mean_ms),
				decomposed_latency_mean_ms: toNumberOrNull(comparableRaw.decomposed_latency_mean_ms),
				monolithic_latency_observed_n: toNumber(comparableRaw.monolithic_latency_observed_n),
				decomposed_latency_observed_n: toNumber(comparableRaw.decomposed_latency_observed_n),
				monolithic_tokens_total: toNumber(comparableRaw.monolithic_tokens_total),
				decomposed_tokens_total: toNumber(comparableRaw.decomposed_tokens_total),
				monolithic_tokens_observed_n: toNumber(comparableRaw.monolithic_tokens_observed_n),
				decomposed_tokens_observed_n: toNumber(comparableRaw.decomposed_tokens_observed_n)
			}
			: null;
		const resourceFrontierArch = (
			architecture: 'monolithic' | 'decomposed',
			run: PairRunSummary,
			latencyMeanMs: number | null | undefined,
			latencyObservedN: number | null | undefined,
			tokensTotal: number | null | undefined,
			tokensObservedN: number | null | undefined,
			mae: number | null | undefined
		): PairedResourceFrontierArch => {
			const actualCost = run.cost_actual_usd == null ? null : Number(run.cost_actual_usd);
			const estimateCost = run.cost_estimate_usd == null ? null : Number(run.cost_estimate_usd);
			const runCost = actualCost ?? estimateCost;
			return {
				architecture,
				run_id: run.run_id,
				run_cost_usd: runCost,
				run_cost_basis: actualCost != null ? 'actual' : estimateCost != null ? 'estimate' : 'missing',
				n_evidences: run.n_evidences,
				cost_per_evidence_usd: safeRate(runCost, run.n_evidences),
				duration_s: run.duration_s,
				wall_seconds_per_evidence: safeRate(run.duration_s, run.n_evidences),
				clean_overlap_n: comparable?.n_overlap ?? 0,
				clean_overlap_latency_mean_ms: latencyMeanMs == null ? null : Number(latencyMeanMs),
				clean_overlap_latency_observed_n: Number(latencyObservedN ?? 0),
				clean_overlap_tokens_total: Number(tokensTotal ?? 0),
				clean_overlap_tokens_observed_n: Number(tokensObservedN ?? 0),
				clean_overlap_tokens_per_observed_evidence: safeRate(Number(tokensTotal ?? 0), Number(tokensObservedN ?? 0)),
				truth_overlap_n: comparable?.n_truth_overlap ?? 0,
				mae: mae == null ? null : Number(mae)
			};
		};
		const resource_frontier: PairedResourceFrontier | null = comparable
			? {
				monolithic: resourceFrontierArch(
					'monolithic',
					monolithic,
					comparable.monolithic_latency_mean_ms,
					comparable.monolithic_latency_observed_n,
					comparable.monolithic_tokens_total,
					comparable.monolithic_tokens_observed_n,
					comparable.monolithic_mae
				),
				decomposed: resourceFrontierArch(
					'decomposed',
					decomposed,
					comparable.decomposed_latency_mean_ms,
					comparable.decomposed_latency_observed_n,
					comparable.decomposed_tokens_total,
					comparable.decomposed_tokens_observed_n,
					comparable.decomposed_mae
				),
				spend_scope: 'whole-run spend; denominator is each side aggregate evidence count, not clean overlap',
				latency_scope: 'clean shared aggregate verdict evidence with per-row telemetry counts shown',
				quality_scope: comparable.n_truth_overlap > 0
					? 'MAE over truth-anchored clean overlap'
					: 'quality not defined: clean overlap has no INDRA belief anchors',
				not_defined_reason: null
			}
			: null;

		const exampleSelect = (whereExtra: string, orderBy: string) => `${pairCte},
			ags AS (
				SELECT stmt_hash,
				       string_agg(name, ', ' ORDER BY
				         CASE role
				           WHEN 'subj' THEN 0 WHEN 'enz' THEN 0
				           WHEN 'obj' THEN 1 WHEN 'sub' THEN 1
				           WHEN 'member' THEN 2 ELSE 3 END,
				         role_index) AS agent_names
				FROM agent
				GROUP BY stmt_hash
			),
			base AS (
				SELECT
				   j.stmt_hash,
				   j.evidence_hash,
				   s.indra_type,
				   COALESCE(ags.agent_names, '') AS agent_names,
				   e.text,
				   e.source_api,
				   e.pmid,
				   s.indra_belief,
				   j.monolithic_score,
				   j.decomposed_score,
				   j.monolithic_verdict,
				   j.decomposed_verdict,
				   CASE WHEN s.indra_belief IS NOT NULL AND j.monolithic_score IS NOT NULL
				        THEN ABS(j.monolithic_score - s.indra_belief) ELSE NULL END AS monolithic_error,
				   CASE WHEN s.indra_belief IS NOT NULL AND j.decomposed_score IS NOT NULL
				        THEN ABS(j.decomposed_score - s.indra_belief) ELSE NULL END AS decomposed_error,
				   CASE WHEN s.indra_belief IS NOT NULL AND j.monolithic_score IS NOT NULL AND j.decomposed_score IS NOT NULL
				        THEN ABS(j.decomposed_score - s.indra_belief) - ABS(j.monolithic_score - s.indra_belief)
				        ELSE NULL END AS abs_error_delta,
				   '/statements/' || j.stmt_hash || '?run_id=${qm}' AS monolithic_href,
				   '/statements/' || j.stmt_hash || '?run_id=${qd}' AS decomposed_href,
				   NULL::VARCHAR AS excluded_side,
				   NULL::VARCHAR AS excluded_reason
				 FROM j
				 JOIN statement s ON s.stmt_hash=j.stmt_hash
				 LEFT JOIN evidence e ON e.evidence_hash=j.evidence_hash
				 LEFT JOIN ags ON ags.stmt_hash=j.stmt_hash
			)
			SELECT * FROM base
			 WHERE ${whereExtra}
			 ORDER BY ${orderBy}, stmt_hash
			 LIMIT 8`;

		const nonOverlapSelect = (side: 'monolithic' | 'decomposed') => {
			const currentRun = side === 'monolithic' ? qm : qd;
			const otherRun = side === 'monolithic' ? qd : qm;
			const scoreColumns = side === 'monolithic'
				? `br.score AS monolithic_score,
				   NULL::DOUBLE AS decomposed_score,
				   br.verdict AS monolithic_verdict,
				   NULL::VARCHAR AS decomposed_verdict,
				   CASE WHEN s.indra_belief IS NOT NULL AND br.score IS NOT NULL
				        THEN ABS(br.score - s.indra_belief) ELSE NULL END AS monolithic_error,
				   NULL::DOUBLE AS decomposed_error`
				: `NULL::DOUBLE AS monolithic_score,
				   br.score AS decomposed_score,
				   NULL::VARCHAR AS monolithic_verdict,
				   br.verdict AS decomposed_verdict,
				   NULL::DOUBLE AS monolithic_error,
				   CASE WHEN s.indra_belief IS NOT NULL AND br.score IS NOT NULL
				        THEN ABS(br.score - s.indra_belief) ELSE NULL END AS decomposed_error`;
			return `WITH
				${comparableSideCte('br', currentRun)},
				${comparableSideCte('other', otherRun)},
				ags AS (
					SELECT stmt_hash,
					       string_agg(name, ', ' ORDER BY
					         CASE role
					           WHEN 'subj' THEN 0 WHEN 'enz' THEN 0
					           WHEN 'obj' THEN 1 WHEN 'sub' THEN 1
					           WHEN 'member' THEN 2 ELSE 3 END,
					         role_index) AS agent_names
					FROM agent
					GROUP BY stmt_hash
				),
				base AS (
					SELECT
					   br.stmt_hash,
					   br.evidence_hash,
					   s.indra_type,
					   COALESCE(ags.agent_names, '') AS agent_names,
					   e.text,
					   e.source_api,
					   e.pmid,
					   s.indra_belief,
					   ${scoreColumns},
					   NULL::DOUBLE AS abs_error_delta,
					   '/statements/' || br.stmt_hash || '?run_id=${qm}' AS monolithic_href,
					   '/statements/' || br.stmt_hash || '?run_id=${qd}' AS decomposed_href,
					   NULL::VARCHAR AS excluded_side,
					   NULL::VARCHAR AS excluded_reason
					FROM br
					JOIN statement s ON s.stmt_hash=br.stmt_hash
					LEFT JOIN evidence e ON e.evidence_hash=br.evidence_hash
					LEFT JOIN ags ON ags.stmt_hash=br.stmt_hash
					LEFT JOIN other_integrity oi
					  ON oi.stmt_hash=br.stmt_hash
					 AND oi.evidence_hash=br.evidence_hash
					WHERE oi.evidence_hash IS NULL
				)
				SELECT * FROM base
				ORDER BY COALESCE(monolithic_error, decomposed_error, 0) DESC, stmt_hash
				LIMIT 8`;
		};
		const excludedByIntegritySelect = `WITH
			${comparableSideCte('m', qm)},
			${comparableSideCte('d', qd)},
			${rawAggregateCte('mr', qm)},
			${rawAggregateCte('dr', qd)},
			keys AS (
				SELECT stmt_hash, evidence_hash FROM m_integrity
				UNION
				SELECT stmt_hash, evidence_hash FROM d_integrity
			),
			ags AS (
				SELECT stmt_hash,
				       string_agg(name, ', ' ORDER BY
				         CASE role
				           WHEN 'subj' THEN 0 WHEN 'enz' THEN 0
				           WHEN 'obj' THEN 1 WHEN 'sub' THEN 1
				           WHEN 'member' THEN 2 ELSE 3 END,
				         role_index) AS agent_names
				FROM agent
				GROUP BY stmt_hash
			),
			classified AS (
				SELECT
				   k.stmt_hash,
				   k.evidence_hash,
				   COALESCE(s.indra_type, 'unknown') AS indra_type,
				   COALESCE(ags.agent_names, '') AS agent_names,
				   e.text,
				   e.source_api,
				   e.pmid,
				   s.indra_belief,
				   mr.score AS monolithic_score,
				   dr.score AS decomposed_score,
				   mr.verdict AS monolithic_verdict,
				   dr.verdict AS decomposed_verdict,
				   CASE WHEN s.indra_belief IS NOT NULL AND mr.score IS NOT NULL
				        THEN ABS(mr.score - s.indra_belief) ELSE NULL END AS monolithic_error,
				   CASE WHEN s.indra_belief IS NOT NULL AND dr.score IS NOT NULL
				        THEN ABS(dr.score - s.indra_belief) ELSE NULL END AS decomposed_error,
				   CASE WHEN s.indra_belief IS NOT NULL AND mr.score IS NOT NULL AND dr.score IS NOT NULL
				        THEN ABS(dr.score - s.indra_belief) - ABS(mr.score - s.indra_belief)
				        ELSE NULL END AS abs_error_delta,
				   '/statements/' || k.stmt_hash || '?run_id=${qm}' AS monolithic_href,
				   '/statements/' || k.stmt_hash || '?run_id=${qd}' AS decomposed_href,
				   CASE
				     WHEN mi.n_aggregate_step_errors > 0 THEN '[M] aggregate step error'
				     WHEN mi.evidence_hash IS NOT NULL AND mi.n_aggregate = 0 THEN '[M] missing aggregate'
				     WHEN mi.n_aggregate > 0 AND mi.n_aggregate_step_errors = 0 AND mi.n_verdict_aggregate = 0 THEN '[M] aggregate lacks verdict'
				     ELSE NULL
				   END AS monolithic_reason,
				   CASE
				     WHEN di.n_aggregate_step_errors > 0 THEN '[D] aggregate step error'
				     WHEN di.evidence_hash IS NOT NULL AND di.n_aggregate = 0 THEN '[D] missing aggregate'
				     WHEN di.n_aggregate > 0 AND di.n_aggregate_step_errors = 0 AND di.n_verdict_aggregate = 0 THEN '[D] aggregate lacks verdict'
				     ELSE NULL
				   END AS decomposed_reason
				FROM keys k
				LEFT JOIN m_integrity mi USING (stmt_hash, evidence_hash)
				LEFT JOIN d_integrity di USING (stmt_hash, evidence_hash)
				LEFT JOIN mr_raw mr USING (stmt_hash, evidence_hash)
				LEFT JOIN dr_raw dr USING (stmt_hash, evidence_hash)
				LEFT JOIN statement s ON s.stmt_hash=k.stmt_hash
				LEFT JOIN evidence e ON e.evidence_hash=k.evidence_hash
				LEFT JOIN ags ON ags.stmt_hash=k.stmt_hash
			)
			SELECT
			   stmt_hash,
			   evidence_hash,
			   indra_type,
			   agent_names,
			   text,
			   source_api,
			   pmid,
			   indra_belief,
			   monolithic_score,
			   decomposed_score,
			   monolithic_verdict,
			   decomposed_verdict,
			   monolithic_error,
			   decomposed_error,
			   abs_error_delta,
			   monolithic_href,
			   decomposed_href,
			   CASE
			     WHEN monolithic_reason IS NOT NULL AND decomposed_reason IS NOT NULL THEN '[M]/[D]'
			     WHEN monolithic_reason IS NOT NULL THEN '[M]'
			     ELSE '[D]'
			   END AS excluded_side,
			   CASE
			     WHEN monolithic_reason IS NOT NULL AND decomposed_reason IS NOT NULL THEN monolithic_reason || '; ' || decomposed_reason
			     ELSE COALESCE(monolithic_reason, decomposed_reason)
			   END AS excluded_reason
			FROM classified
			WHERE monolithic_reason IS NOT NULL OR decomposed_reason IS NOT NULL
			ORDER BY
			  CASE
			    WHEN COALESCE(monolithic_reason, '') LIKE '%step error%' OR COALESCE(decomposed_reason, '') LIKE '%step error%' THEN 0
			    WHEN COALESCE(monolithic_reason, '') LIKE '%missing aggregate%' OR COALESCE(decomposed_reason, '') LIKE '%missing aggregate%' THEN 1
			    ELSE 2
			  END,
			  stmt_hash
			LIMIT 8`;

		const [
			monolithicWins,
			decomposedWins,
			verdictDisagreements,
			mutualFailures,
			monolithicOnly,
			decomposedOnly,
			excludedByIntegrity,
			monoTiers,
			decomposedProbes
		] = await Promise.all([
			rows<PairedExampleRow>(
				con,
				exampleSelect(
					`abs_error_delta IS NOT NULL AND abs_error_delta > 0`,
					`abs_error_delta DESC`
				)
			),
			rows<PairedExampleRow>(
				con,
				exampleSelect(
					`abs_error_delta IS NOT NULL AND abs_error_delta < 0`,
					`abs_error_delta ASC`
				)
			),
			rows<PairedExampleRow>(
				con,
				exampleSelect(
					`COALESCE(monolithic_verdict, '') <> COALESCE(decomposed_verdict, '')`,
					`ABS(COALESCE(abs_error_delta, 0)) DESC`
				)
			),
			rows<PairedExampleRow>(
				con,
				exampleSelect(
					`monolithic_error >= 0.25 AND decomposed_error >= 0.25`,
					`(monolithic_error + decomposed_error) DESC`
				)
			),
			rows<PairedExampleRow>(con, nonOverlapSelect('monolithic')),
			rows<PairedExampleRow>(con, nonOverlapSelect('decomposed')),
			rows<PairedExampleRow>(con, excludedByIntegritySelect),
			rows<PairedArchTierRow>(
				con,
				`SELECT
				   COALESCE(json_extract_string(output_json, '$.tier'), 'unknown') AS tier,
				   COUNT(*) AS n,
				   AVG(CAST(json_extract(output_json, '$.score') AS DOUBLE)) AS mean_score
				 FROM scorer_step
				 WHERE run_id='${qm}' AND step_kind='aggregate'
				 GROUP BY tier
				 ORDER BY n DESC, tier`
			),
			rows<PairedArchProbeRow>(
				con,
				`SELECT
				   step_kind AS name,
				   COUNT(*) AS n,
				   SUM(CASE WHEN is_substrate_answered THEN 1 ELSE 0 END) AS substrate_n,
				   SUM(CASE WHEN error IS NOT NULL THEN 1 ELSE 0 END) AS error_n
				 FROM scorer_step
				 WHERE run_id='${qd}'
				   AND step_kind IN ('subject_role_probe','object_role_probe','relation_axis_probe','scope_probe','substrate_route')
				 GROUP BY step_kind
				 ORDER BY CASE step_kind
				   WHEN 'substrate_route' THEN 0
				   WHEN 'subject_role_probe' THEN 1
				   WHEN 'object_role_probe' THEN 2
				   WHEN 'relation_axis_probe' THEN 3
				   WHEN 'scope_probe' THEN 4
				   ELSE 5 END`
			)
		]);

		const withNumericExamples = (rs: PairedExampleRow[]) => rs.map((r) => ({
			...r,
			indra_belief: r.indra_belief == null ? null : Number(r.indra_belief),
			monolithic_score: r.monolithic_score == null ? null : Number(r.monolithic_score),
			decomposed_score: r.decomposed_score == null ? null : Number(r.decomposed_score),
			monolithic_error: r.monolithic_error == null ? null : Number(r.monolithic_error),
			decomposed_error: r.decomposed_error == null ? null : Number(r.decomposed_error),
			abs_error_delta: r.abs_error_delta == null ? null : Number(r.abs_error_delta)
		}));
		const numericTiers = monoTiers.map((r) => ({
			...r,
			n: toNumber(r.n),
			mean_score: toNumberOrNull(r.mean_score)
		}));
		const numericProbes = decomposedProbes.map((r) => ({
			...r,
			n: toNumber(r.n),
			substrate_n: toNumber(r.substrate_n),
			error_n: toNumber(r.error_n)
		}));
		const notDefinedReason = !overlap
			? 'paired workbench needs overlap accounting before comparable metrics are defined'
			: overlap.overlap_evidences === 0
				? 'comparison not defined: monolithic and decomposed runs share zero non-error aggregate verdict evidence rows'
				: comparable
					? null
					: 'comparison not defined: paired aggregate rows exist, but no comparable metric denominator could be formed';
		const monolithicIntegrityExclusions = overlap
			? overlap.monolithic_step_error_evidences +
				overlap.monolithic_missing_aggregate_evidences +
				overlap.monolithic_nonverdict_aggregate_evidences
			: 0;
		const decomposedIntegrityExclusions = overlap
			? overlap.decomposed_step_error_evidences +
				overlap.decomposed_missing_aggregate_evidences +
				overlap.decomposed_nonverdict_aggregate_evidences
			: 0;
		const integrityExclusions = monolithicIntegrityExclusions + decomposedIntegrityExclusions;
		const traceWarnings = overlap
			? overlap.monolithic_nonaggregate_step_error_evidences +
				overlap.decomposed_nonaggregate_step_error_evidences
			: 0;
		const probeRows = numericProbes.reduce((sum, r) => sum + r.n, 0);
		const tierRows = numericTiers.reduce((sum, r) => sum + r.n, 0);
		const denominatorLedger: PairedDenominatorRow[] = overlap
			? [
					{
						key: 'side_aggregate_evidence_rows',
						panel: 'overlap accounting',
						applicability: 'paired_only',
						metric_kind: 'denominator_base',
						ledger_role: 'root',
						parent_key: null,
						unit: 'side aggregate evidence rows',
						denominator_n: null,
						monolithic_n: overlap.monolithic_evidences,
						decomposed_n: overlap.decomposed_evidences,
						overlap_n: overlap.overlap_evidences,
						excluded_n: integrityExclusions,
						reason: 'Starts from persisted aggregate rows on each side before any paired metric is allowed to share a denominator.'
					},
					{
						key: 'clean_shared_aggregate_verdict_evidence',
						panel: 'clean shared paired base',
						applicability: comparable ? 'paired_only' : 'not_defined',
						metric_kind: 'denominator_base',
						ledger_role: 'base',
						parent_key: 'side_aggregate_evidence_rows',
						unit: 'clean shared aggregate verdict evidence',
						denominator_n: comparable?.n_overlap ?? null,
						monolithic_n: comparable?.n_overlap ?? null,
						decomposed_n: comparable?.n_overlap ?? null,
						overlap_n: overlap.overlap_evidences,
						excluded_n: integrityExclusions,
						reason: comparable
							? 'Only evidence with non-error aggregate verdict rows on both architectures enters any clean shared paired metric.'
							: (notDefinedReason ?? 'comparison not defined for the paired clean-overlap denominator')
					},
					{
						key: 'exact_label_agreement_metric',
						panel: '[M] vs [D] exact verdict label match',
						applicability: comparable ? 'paired_only' : 'not_defined',
						metric_kind: 'arch_arch_exact_label',
						ledger_role: 'metric',
						parent_key: 'clean_shared_aggregate_verdict_evidence',
						unit: 'clean shared aggregate verdict evidence',
						denominator_n: comparable?.n_overlap ?? null,
						monolithic_n: comparable?.n_overlap ?? null,
						decomposed_n: comparable?.n_overlap ?? null,
						overlap_n: overlap.overlap_evidences,
						excluded_n: integrityExclusions,
						reason: comparable
							? 'Compares exact aggregate verdict labels between architectures. It is inter-architecture agreement, not accuracy.'
							: (notDefinedReason ?? 'exact label agreement needs clean shared aggregate verdict rows')
					},
					{
						key: 'support_state_recode_metric',
						panel: 'support-state matrix and exact-label taxonomy',
						applicability: comparable ? 'paired_only' : 'not_defined',
						metric_kind: 'arch_arch_support_recode',
						ledger_role: 'metric',
						parent_key: 'clean_shared_aggregate_verdict_evidence',
						unit: 'clean shared aggregate verdict evidence',
						denominator_n: comparable?.n_overlap ?? null,
						monolithic_n: comparable?.n_overlap ?? null,
						decomposed_n: comparable?.n_overlap ?? null,
						overlap_n: overlap.overlap_evidences,
						excluded_n: integrityExclusions,
						reason: comparable
							? 'Recodes exact labels into supported/not-supported buckets, then shows the exact labels inside each bucket.'
							: (notDefinedReason ?? 'support-state recode needs clean shared aggregate verdict rows')
					},
					{
						key: 'score_posture_metric',
						panel: 'mean score movement',
						applicability: comparable ? 'paired_only' : 'not_defined',
						metric_kind: 'arch_arch_score_posture',
						ledger_role: 'metric',
						parent_key: 'clean_shared_aggregate_verdict_evidence',
						unit: 'scores over clean shared aggregate verdict evidence',
						denominator_n: comparable?.n_overlap ?? null,
						monolithic_n: comparable?.n_overlap ?? null,
						decomposed_n: comparable?.n_overlap ?? null,
						overlap_n: overlap.overlap_evidences,
						excluded_n: integrityExclusions,
						reason: comparable
							? 'Compares mean score posture between architectures. It is not a winner and not an INDRA residual.'
							: (notDefinedReason ?? 'score posture needs clean shared aggregate verdict rows')
					},
					{
						key: 'resource_counter_metric',
						panel: 'latency and token counters',
						applicability: comparable ? 'paired_only' : 'not_defined',
						metric_kind: 'arch_arch_resource',
						ledger_role: 'metric',
						parent_key: 'clean_shared_aggregate_verdict_evidence',
						unit: 'clean shared aggregate verdict evidence',
						denominator_n: comparable?.n_overlap ?? null,
						monolithic_n: comparable?.n_overlap ?? null,
						decomposed_n: comparable?.n_overlap ?? null,
						overlap_n: overlap.overlap_evidences,
						excluded_n: integrityExclusions,
						reason: comparable
							? 'Compares resource counters over the clean shared denominator; resource direction is not residual quality.'
							: (notDefinedReason ?? 'resource counters need clean shared aggregate verdict rows')
					},
					{
						key: 'truth_anchored_overlap_evidence',
						panel: 'INDRA MAE and bias',
						applicability: comparable && comparable.n_truth_overlap > 0 ? 'paired_only' : 'not_defined',
						metric_kind: 'arch_indra_residual',
						ledger_role: 'metric',
						parent_key: 'clean_shared_aggregate_verdict_evidence',
						unit: 'truth-anchored overlap evidence',
						denominator_n: comparable?.n_truth_overlap ?? null,
						monolithic_n: comparable?.n_truth_overlap ?? null,
						decomposed_n: comparable?.n_truth_overlap ?? null,
						overlap_n: comparable?.n_overlap ?? overlap.overlap_evidences,
						excluded_n: comparable ? comparable.n_overlap - comparable.n_truth_overlap : null,
						reason: comparable && comparable.n_truth_overlap > 0
							? 'Residual metrics compare each architecture to INDRA belief, not to the other architecture.'
							: 'Residual metrics need clean overlap rows with INDRA belief anchors.'
					},
					{
						key: 'side_evidence_integrity_exclusions',
						panel: 'integrity exclusions',
						applicability: 'paired_only',
						metric_kind: 'integrity_gate',
						ledger_role: 'gate',
						parent_key: 'side_aggregate_evidence_rows',
						unit: 'side-evidence exclusions',
						denominator_n: integrityExclusions,
						monolithic_n: monolithicIntegrityExclusions,
						decomposed_n: decomposedIntegrityExclusions,
						overlap_n: null,
						excluded_n: integrityExclusions,
						reason: 'Aggregate step errors, missing aggregates, and aggregate rows without verdicts are withheld from comparable metrics but remain inspectable.'
					},
					{
						key: 'clean_true_nonoverlap_evidence',
						panel: 'true non-overlap exemplars',
						applicability: 'paired_only',
						metric_kind: 'paired_nonoverlap',
						ledger_role: 'gate',
						parent_key: 'side_aggregate_evidence_rows',
						unit: 'clean evidence rows',
						denominator_n: overlap.monolithic_true_nonoverlap_evidences + overlap.decomposed_true_nonoverlap_evidences,
						monolithic_n: overlap.monolithic_true_nonoverlap_evidences,
						decomposed_n: overlap.decomposed_true_nonoverlap_evidences,
						overlap_n: 0,
						excluded_n: overlap.monolithic_true_nonoverlap_evidences + overlap.decomposed_true_nonoverlap_evidences,
						reason: 'These lanes only include clean rows where the other architecture has no persisted rows for the evidence, so they stay outside paired deltas.'
					},
					{
						key: 'monolithic_aggregate_rows',
						panel: 'native monolithic tier path',
						applicability: 'arch_conditioned',
						metric_kind: 'architecture_native',
						ledger_role: 'native',
						parent_key: null,
						unit: 'monolithic aggregate rows',
						denominator_n: tierRows,
						monolithic_n: tierRows,
						decomposed_n: null,
						overlap_n: null,
						excluded_n: monolithicIntegrityExclusions,
						reason: 'Tier diagnostics are native to monolithic aggregate rows and are not converted into decomposed probe grammar.'
					},
					{
						key: 'decomposed_native_step_rows',
						panel: 'native decomposed probe health',
						applicability: 'arch_conditioned',
						metric_kind: 'architecture_native',
						ledger_role: 'native',
						parent_key: null,
						unit: 'decomposed native step rows',
						denominator_n: probeRows,
						monolithic_n: null,
						decomposed_n: probeRows,
						overlap_n: null,
						excluded_n: overlap.decomposed_nonaggregate_step_error_evidences,
						reason: 'Probe health is measured over decomposed native step rows, not over the paired clean-overlap denominator.'
					},
					{
						key: 'nonaggregate_step_error_evidence',
						panel: 'non-aggregate trace warnings',
						applicability: traceWarnings > 0 ? 'arch_conditioned' : 'not_defined',
						metric_kind: 'architecture_native_trace_health',
						ledger_role: 'native',
						parent_key: null,
						unit: 'evidence with non-aggregate step errors',
						denominator_n: traceWarnings,
						monolithic_n: overlap.monolithic_nonaggregate_step_error_evidences,
						decomposed_n: overlap.decomposed_nonaggregate_step_error_evidences,
						overlap_n: null,
						excluded_n: 0,
						reason: traceWarnings > 0
							? 'These warnings describe native trace health outside the aggregate comparable gate.'
							: 'No non-aggregate step errors are persisted for either architecture in this pair.'
				}
			]
			: [];

		return {
			pair_id,
			runs,
			monolithic,
			decomposed,
			overlap,
			comparable,
			resource_frontier,
			denominator_ledger: denominatorLedger,
			exemplars: {
				monolithic_wins: withNumericExamples(monolithicWins),
				decomposed_wins: withNumericExamples(decomposedWins),
				verdict_disagreements: withNumericExamples(verdictDisagreements),
				mutual_failures: withNumericExamples(mutualFailures),
				monolithic_only: withNumericExamples(monolithicOnly),
				decomposed_only: withNumericExamples(decomposedOnly),
				excluded_by_integrity: withNumericExamples(excludedByIntegrity)
			},
			arch_conditioned: {
				monolithic_tiers: numericTiers,
				decomposed_probes: numericProbes
			},
			not_defined_reason: notDefinedReason
		};
	} finally {
		con.disconnectSync?.();
	}
}

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

async function getRunTraceIntegrityCounts(
	con: DuckDBConnection,
	run_id: string
): Promise<{ missing_aggregate: number; step_error: number; aggregate_evidences: number }> {
	const qr = sqlQuote(run_id);
	const rs = await rows<{
		missing_aggregate: number | null;
		step_error: number | null;
		aggregate_evidences: number | null;
	}>(
		con,
		`WITH evidence_steps AS (
			SELECT evidence_hash,
			       SUM(CASE WHEN step_kind='aggregate' THEN 1 ELSE 0 END) AS n_aggregate,
			       SUM(CASE
			         WHEN error IS NOT NULL OR json_extract(output_json, '$.error') IS NOT NULL THEN 1
			         ELSE 0
			       END) AS n_step_errors
			FROM scorer_step
			WHERE run_id='${qr}'
			  AND evidence_hash IS NOT NULL
			GROUP BY evidence_hash
		)
		SELECT
		   SUM(CASE WHEN n_aggregate = 0 AND n_step_errors = 0 THEN 1 ELSE 0 END) AS missing_aggregate,
		   SUM(CASE WHEN n_step_errors > 0 THEN 1 ELSE 0 END) AS step_error,
		   SUM(CASE WHEN n_aggregate > 0 THEN 1 ELSE 0 END) AS aggregate_evidences
		FROM evidence_steps`
	);
	const row = rs[0] ?? { missing_aggregate: 0, step_error: 0, aggregate_evidences: 0 };
	return {
		missing_aggregate: Number(row.missing_aggregate ?? 0),
		step_error: Number(row.step_error ?? 0),
		aggregate_evidences: Number(row.aggregate_evidences ?? 0)
	};
}

/**
 * Run-over-run narrative for the dashboard's runs feed and the /runs/[id]
 * page. Auto-comparison uses the most recent same-architecture predecessor;
 * cross-architecture comparison is only valid when a caller names it
 * explicitly. When there is no predecessor, returns a self-summary.
 */
export async function getRunNarrative(
	run_id: string,
	explicit_prev_run_id?: string | null
): Promise<RunNarrative | null> {
	if (!dbExists()) return null;
	const con = await connect();
	try {
		const qr = run_id.replace(/'/g, "''");
		const currentRows = await rows<{ status: string | null }>(
			con,
			`SELECT status FROM score_run WHERE run_id='${qr}'`
		);
		if (currentRows.length === 0 || currentRows[0].status !== 'succeeded') return null;
		// If caller named a specific predecessor (e.g. from the compare-runs
		// route), use it as-is; otherwise auto-find the most-recent succeeded
		// run that started before this one.
		let prev_run_id: string | null = null;
		if (explicit_prev_run_id !== undefined) {
			prev_run_id = explicit_prev_run_id;
		} else {
			const prevRows = await rows<{ run_id: string }>(
				con,
				`SELECT run_id FROM score_run
				 WHERE status='succeeded'
				   AND architecture = (SELECT architecture FROM score_run WHERE run_id='${qr}')
				   AND started_at < (SELECT started_at FROM score_run WHERE run_id='${qr}')
				 ORDER BY started_at DESC LIMIT 1`
			);
			prev_run_id = prevRows[0]?.run_id ?? null;
		}

		let comparison_blocked_reason: string | null = null;
		if (prev_run_id) {
			const qpArch = prev_run_id.replace(/'/g, "''");
			const archRows = await rows<{ curr_arch: string | null; prev_arch: string | null }>(
				con,
				`SELECT
				   (SELECT architecture FROM score_run WHERE run_id='${qr}') AS curr_arch,
				   (SELECT architecture FROM score_run WHERE run_id='${qpArch}') AS prev_arch`
			);
			const currArch = archRows[0]?.curr_arch ?? null;
			const prevArch = archRows[0]?.prev_arch ?? null;
			if (currArch && prevArch && currArch !== prevArch) {
				comparison_blocked_reason =
					`run-page deltas are not defined across architectures (${currArch} vs ${prevArch}); use a paired workbench that compares overlap explicitly`;
				prev_run_id = null;
			}
		}
		if (!comparison_blocked_reason) {
			const currentIntegrity = await getRunTraceIntegrityCounts(con, run_id);
			if (currentIntegrity.step_error > 0) {
				comparison_blocked_reason =
					`comparison not defined: ${currentIntegrity.step_error} evidence row${currentIntegrity.step_error === 1 ? '' : 's'} in this succeeded run has an explicit scorer step error`;
				prev_run_id = null;
			} else if (currentIntegrity.missing_aggregate > 0) {
				comparison_blocked_reason =
					`comparison not defined: ${currentIntegrity.missing_aggregate} evidence row${currentIntegrity.missing_aggregate === 1 ? '' : 's'} in this succeeded run has persisted trace steps but no aggregate verdict`;
				prev_run_id = null;
			}
		}
		if (!comparison_blocked_reason && prev_run_id) {
			const prevIntegrity = await getRunTraceIntegrityCounts(con, prev_run_id);
			if (prevIntegrity.step_error > 0) {
				comparison_blocked_reason =
					`comparison not defined: predecessor ${prev_run_id.slice(0, 8)} has ${prevIntegrity.step_error} evidence row${prevIntegrity.step_error === 1 ? '' : 's'} with an explicit scorer step error`;
				prev_run_id = null;
			} else if (prevIntegrity.missing_aggregate > 0) {
				comparison_blocked_reason =
					`comparison not defined: predecessor ${prev_run_id.slice(0, 8)} has ${prevIntegrity.missing_aggregate} evidence row${prevIntegrity.missing_aggregate === 1 ? '' : 's'} with persisted trace steps but no aggregate verdict`;
				prev_run_id = null;
			}
		}
		if (!comparison_blocked_reason && prev_run_id) {
			const qp = sqlQuote(prev_run_id);
			const overlapRows = await rows<{ n: number }>(
				con,
				`WITH curr AS (
					SELECT DISTINCT stmt_hash
					FROM scorer_step
					WHERE run_id='${qr}'
					  AND step_kind='aggregate'
					  AND error IS NULL
					  AND json_extract(output_json, '$.error') IS NULL
				),
				prev AS (
					SELECT DISTINCT stmt_hash
					FROM scorer_step
					WHERE run_id='${qp}'
					  AND step_kind='aggregate'
					  AND error IS NULL
					  AND json_extract(output_json, '$.error') IS NULL
				)
				SELECT COUNT(*) AS n
				FROM curr JOIN prev USING(stmt_hash)`
			);
			if (Number(overlapRows[0]?.n ?? 0) === 0) {
				comparison_blocked_reason =
					`comparison not defined: this run and predecessor ${prev_run_id.slice(0, 8)} share zero completed aggregate statement verdicts`;
				prev_run_id = null;
			}
		}

		const calRow = await rows<{ mae: number | null; bias: number | null }>(
			con,
			`SELECT
			   (SELECT value FROM metric WHERE run_id='${qr}' AND metric_name='indra_belief_calibration.mae' AND truth_set_id='indra_published_belief' LIMIT 1) AS mae,
			   (SELECT value FROM metric WHERE run_id='${qr}' AND metric_name='indra_belief_calibration.bias' AND truth_set_id='indra_published_belief' LIMIT 1) AS bias`
		);
		const mae = calRow[0]?.mae ?? null;
		const bias = calRow[0]?.bias ?? null;

		let mae_delta: number | null = null;
		let bias_delta: number | null = null;
		const crossings: Array<{ stmt_hash: string; prev_verdict: string; curr_verdict: string }> = [];

		if (prev_run_id) {
			const qp = prev_run_id.replace(/'/g, "''");
			const prevCalRow = await rows<{ mae: number | null; bias: number | null }>(
				con,
				`SELECT
				   (SELECT value FROM metric WHERE run_id='${qp}' AND metric_name='indra_belief_calibration.mae' AND truth_set_id='indra_published_belief' LIMIT 1) AS mae,
				   (SELECT value FROM metric WHERE run_id='${qp}' AND metric_name='indra_belief_calibration.bias' AND truth_set_id='indra_published_belief' LIMIT 1) AS bias`
			);
			const prev_mae = prevCalRow[0]?.mae ?? null;
			const prev_bias = prevCalRow[0]?.bias ?? null;
			if (mae != null && prev_mae != null) mae_delta = mae - prev_mae;
			if (bias != null && prev_bias != null) bias_delta = bias - prev_bias;

				const verdictRows = await rows<{ stmt_hash: string; curr_verdict: string; prev_verdict: string }>(
					con,
					`WITH curr AS (
						SELECT stmt_hash,
						       CASE
						         WHEN SUM(CASE WHEN json_extract_string(output_json, '$.verdict')='incorrect' THEN 1 ELSE 0 END) > 0 THEN 'incorrect'
						         WHEN SUM(CASE WHEN json_extract_string(output_json, '$.verdict')='abstain' THEN 1 ELSE 0 END) > 0 THEN 'abstain'
						         WHEN SUM(CASE WHEN json_extract_string(output_json, '$.verdict')='correct' THEN 1 ELSE 0 END) = COUNT(*) THEN 'correct'
						         WHEN SUM(CASE WHEN json_extract_string(output_json, '$.verdict')='correct' THEN 1 ELSE 0 END) > 0 THEN 'mixed'
						         ELSE 'unknown'
						       END AS verdict
						FROM scorer_step WHERE run_id='${qr}' AND step_kind='aggregate' GROUP BY stmt_hash
					),
					prev AS (
						SELECT stmt_hash,
						       CASE
						         WHEN SUM(CASE WHEN json_extract_string(output_json, '$.verdict')='incorrect' THEN 1 ELSE 0 END) > 0 THEN 'incorrect'
						         WHEN SUM(CASE WHEN json_extract_string(output_json, '$.verdict')='abstain' THEN 1 ELSE 0 END) > 0 THEN 'abstain'
						         WHEN SUM(CASE WHEN json_extract_string(output_json, '$.verdict')='correct' THEN 1 ELSE 0 END) = COUNT(*) THEN 'correct'
						         WHEN SUM(CASE WHEN json_extract_string(output_json, '$.verdict')='correct' THEN 1 ELSE 0 END) > 0 THEN 'mixed'
						         ELSE 'unknown'
						       END AS verdict
						FROM scorer_step WHERE run_id='${qp}' AND step_kind='aggregate' GROUP BY stmt_hash
					)
				SELECT curr.stmt_hash AS stmt_hash,
				       curr.verdict AS curr_verdict,
				       prev.verdict AS prev_verdict
				FROM curr JOIN prev USING(stmt_hash)
				WHERE curr.verdict <> prev.verdict`
			);
			for (const r of verdictRows) {
				crossings.push({ stmt_hash: r.stmt_hash, prev_verdict: r.prev_verdict, curr_verdict: r.curr_verdict });
			}
		}

		const moved_to_correct = crossings.filter((c) => c.curr_verdict === 'correct').length;
		const moved_to_incorrect = crossings.filter((c) => c.curr_verdict === 'incorrect').length;

		const fmtSigned3 = (n: number) => `${n >= 0 ? '+' : '−'}${Math.abs(n).toFixed(3)}`;
		let summary_sentence: string;
		if (comparison_blocked_reason) {
			summary_sentence = comparison_blocked_reason;
		} else if (!prev_run_id) {
			summary_sentence =
				mae != null
					? `first comparable run · MAE ${mae.toFixed(3)} · bias ${bias != null ? fmtSigned3(bias) : '—'}`
					: 'first run, no calibration data';
		} else {
			const parts: string[] = [];
			if (mae_delta != null) parts.push(`MAE ${fmtSigned3(mae_delta)}`);
			if (bias_delta != null) parts.push(`bias ${fmtSigned3(bias_delta)}`);
			if (crossings.length > 0) {
				parts.push(
					`${crossings.length} verdict${crossings.length === 1 ? '' : 's'} moved (` +
						`${moved_to_correct} to correct, ${moved_to_incorrect} to incorrect)`
				);
			} else {
				parts.push('no verdicts moved');
			}
			summary_sentence = parts.join(' · ');
		}

		return {
			run_id,
			prev_run_id,
			summary_sentence,
			comparison_blocked_reason,
			mae_delta,
			bias_delta,
			verdicts_moved_total: crossings.length,
			verdicts_moved_to_correct: moved_to_correct,
			verdicts_moved_to_incorrect: moved_to_incorrect,
			verdict_crossings: crossings
		};
	} finally {
		con.disconnectSync?.();
	}
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

function stepOrder(kind: string): number {
	return (
		{
			parse_claim: 0,
			build_context: 1,
			substrate_route: 2,
			subject_role_probe: 3,
			object_role_probe: 4,
			relation_axis_probe: 5,
			scope_probe: 6,
			grounding: 7,
			adjudicate: 8,
			aggregate: 9
		} as Record<string, number>
	)[kind] ?? 99;
}

function heuristicLifecycleMessage(
	architecture: string,
	counts: Omit<HeuristicCoverageTraceDiagnostic, 'kind' | 'message' | 'lifecycle_kind' | 'lifecycle_message'>
): { lifecycle_kind: HeuristicCoverageTraceLifecycleKind; lifecycle_message: string } {
	const status = counts.run_status || 'unknown';
	const termination = [counts.terminated_by, counts.termination_reason].filter(Boolean).join(' · ');
	if (architecture !== 'decomposed') {
		return {
			lifecycle_kind: 'not_applicable',
			lifecycle_message: 'run lifecycle does not define four-probe trace cause for this architecture'
		};
	}
	if (counts.n_substrate_route_steps > 0 || counts.n_probe_steps > 0) {
		return {
			lifecycle_kind: 'probe_records_present',
			lifecycle_message: 'probe rows are present, so substrate/LLM coverage is measured from persisted trace slots'
		};
	}
	if (status === 'running') {
		return {
			lifecycle_kind: 'running_snapshot',
			lifecycle_message:
				'run lifecycle is running; this page is a persisted snapshot and probe absence can still change as the worker appends rows'
		};
	}
	if (status === 'failed' || status === 'canceled' || status === 'cancelled' || status === 'aborted') {
		return {
			lifecycle_kind: 'interrupted_run',
			lifecycle_message:
				`run lifecycle is ${status}${termination ? ` (${termination})` : ''}; probe absence is in an interrupted-run context, so coverage is not a completed trace-health measurement`
		};
	}
	if (status === 'succeeded') {
		return {
			lifecycle_kind: 'succeeded_without_cause_provenance',
			lifecycle_message:
				'run lifecycle is succeeded, so the stored run row does not support user cancellation as the explanation; importer, worker-log, and migration provenance are still absent, so root cause remains unclassified'
		};
	}
	return {
		lifecycle_kind: 'unknown_lifecycle',
		lifecycle_message:
			`run lifecycle is ${status}; without a recognized terminal/running state, the UI cannot classify why probe records are absent`
	};
}

function heuristicTraceDiagnostic(
	architecture: string,
	counts: Omit<HeuristicCoverageTraceDiagnostic, 'kind' | 'message' | 'lifecycle_kind' | 'lifecycle_message'>
): HeuristicCoverageTraceDiagnostic {
	const lifecycle = heuristicLifecycleMessage(architecture, counts);
	if (architecture !== 'decomposed') {
		return {
			...counts,
			...lifecycle,
			kind: 'not_applicable',
			message: 'four-probe trace diagnostics are not defined for this architecture'
		};
	}
		if (counts.n_substrate_route_steps > 0 || counts.n_probe_steps > 0) {
			return {
				...counts,
				...lifecycle,
				kind: 'probe_rows_present',
				message: counts.n_aggregate_evidences === 0
					? 'persisted decomposed substrate/probe rows exist before aggregate verdict rows; the coverage table is measured from trace records while the aggregate verdict denominator remains 0'
					: 'persisted decomposed substrate/probe rows exist, so the coverage table is measured from trace records'
			};
		}
		if (counts.n_aggregate_evidences === 0) {
			return {
				...counts,
				...lifecycle,
				kind: 'no_aggregate_evidence',
				message: 'no aggregate evidence rows are finalized for this run yet'
			};
		}
	if (counts.n_nonaggregate_steps > 0) {
		return {
			...counts,
			...lifecycle,
			kind: 'native_steps_without_probe_slots',
			message:
				`persisted scorer_step rows include ${counts.n_nonaggregate_steps} decomposed native step row${counts.n_nonaggregate_steps === 1 ? '' : 's'}, but 0 substrate_route/probe rows; probe health is unavailable until those slot records exist`
		};
	}
	return {
		...counts,
		...lifecycle,
		kind: 'aggregate_only_trace',
		message:
			`persisted scorer_step rows show ${counts.n_aggregate_evidences} aggregate evidence row${counts.n_aggregate_evidences === 1 ? '' : 's'} and 0 decomposed native step rows; the DB cannot distinguish legacy/imported aggregate-only data from failed trace persistence`
	};
}

/**
 * Per-probe substrate/LLM/abstain coverage for a single run.
 *
 * Reads two sources and unions:
 *   1. substrate_route rows expose per-probe slot results.
 *   2. Individual {probe}_probe rows are written when substrate answers
 *      OR when LLM fires its own probe step.
 *
 * Coverage = the "final" source per (evidence, probe), with substrate-route's
 * slot taking precedence (it's authoritative for what substrate decided).
 * Missing data on both sides means "probe did not run for this evidence" —
 * counted in `notrun_n`, not silently dropped.
 */
export async function getHeuristicCoverage(run_id: string): Promise<HeuristicCoverage | null> {
	if (!dbExists()) return null;
	const con = await connect();
	try {
		const qr = run_id.replace(/'/g, "''");
		const runRows = await rows<{
			architecture: string | null;
			scorer_version: string | null;
			started_at: string | null;
			status: string | null;
			terminated_by: string | null;
			termination_reason: string | null;
			n_evidences: number;
			n_nonaggregate_steps: number;
			n_substrate_route_steps: number;
			n_probe_steps: number;
			n_grounding_steps: number;
			n_adjudicate_steps: number;
		}>(
			con,
			`SELECT
			   architecture,
			   scorer_version,
			   started_at::VARCHAR AS started_at,
			   status,
			   terminated_by,
			   termination_reason,
			   (SELECT COUNT(DISTINCT evidence_hash)
			    FROM scorer_step
			    WHERE run_id='${qr}' AND step_kind='aggregate') AS n_evidences,
			   (SELECT COUNT(*)
			    FROM scorer_step
			    WHERE run_id='${qr}'
			      AND evidence_hash IS NOT NULL
			      AND step_kind<>'aggregate') AS n_nonaggregate_steps,
			   (SELECT COUNT(*)
			    FROM scorer_step
			    WHERE run_id='${qr}'
			      AND evidence_hash IS NOT NULL
			      AND step_kind='substrate_route') AS n_substrate_route_steps,
			   (SELECT COUNT(*)
			    FROM scorer_step
			    WHERE run_id='${qr}'
			      AND evidence_hash IS NOT NULL
			      AND step_kind IN ('subject_role_probe','object_role_probe','relation_axis_probe','scope_probe')) AS n_probe_steps,
			   (SELECT COUNT(*)
			    FROM scorer_step
			    WHERE run_id='${qr}'
			      AND evidence_hash IS NOT NULL
			      AND step_kind='grounding') AS n_grounding_steps,
			   (SELECT COUNT(*)
			    FROM scorer_step
			    WHERE run_id='${qr}'
			      AND evidence_hash IS NOT NULL
			      AND step_kind='adjudicate') AS n_adjudicate_steps
			 FROM score_run
			 WHERE run_id='${qr}'`
		);
		if (runRows.length === 0) return null;
		const architecture = runRows[0].architecture ?? 'unknown';
		const trace_diagnostic = heuristicTraceDiagnostic(architecture, {
			n_aggregate_evidences: Number(runRows[0].n_evidences ?? 0),
			n_nonaggregate_steps: Number(runRows[0].n_nonaggregate_steps ?? 0),
			n_substrate_route_steps: Number(runRows[0].n_substrate_route_steps ?? 0),
			n_probe_steps: Number(runRows[0].n_probe_steps ?? 0),
			n_grounding_steps: Number(runRows[0].n_grounding_steps ?? 0),
			n_adjudicate_steps: Number(runRows[0].n_adjudicate_steps ?? 0),
			run_status: runRows[0].status ?? 'unknown',
			scorer_version: runRows[0].scorer_version ?? null,
			started_at: runRows[0].started_at ?? null,
			terminated_by: runRows[0].terminated_by ?? null,
			termination_reason: runRows[0].termination_reason ?? null
		});
			if (architecture !== 'decomposed') {
				return {
					run_id,
					architecture,
					applicability: 'not_defined',
					not_defined_reason:
						'four-probe substrate/LLM coverage is defined only for decomposed runs; monolithic runs use a single aggregate tier grammar',
					n_evidences: runRows[0].n_evidences ?? 0,
					n_probe_evidences: 0,
					per_probe: [],
					trace_diagnostic,
					all_substrate_rate: 0,
				short_circuited_rate: 0,
				mean_invoked_probes: 0
			};
			}
			const probes: ProbeKind[] = ['subject_role', 'object_role', 'relation_axis', 'scope'];
			const probeNames = probes.map((p) => `SELECT '${p}' AS probe`).join(' UNION ALL ');
			const slotUnions = probes
				.map(
					(p) => `
					SELECT evidence_hash, '${p}' AS probe,
				       json_extract_string(output_json, '$.${p}.source') AS substrate_source,
				       json_extract_string(output_json, '$.${p}.answer') AS substrate_answer
					FROM scorer_step
					WHERE run_id='${qr}' AND step_kind='substrate_route'`
				)
				.join(' UNION ALL ');
			const probeCoverageCtes = `
				probe_names AS (${probeNames}),
				substrate_slots AS (${slotUnions}),
				individual_probes AS (
					SELECT evidence_hash,
					       replace(step_kind, '_probe', '') AS probe,
					       json_extract_string(output_json, '$.source') AS llm_source,
					       json_extract_string(output_json, '$.answer') AS llm_answer
					FROM scorer_step
					WHERE run_id='${qr}'
					  AND step_kind IN ('subject_role_probe','object_role_probe','relation_axis_probe','scope_probe')
				),
				probe_evidence AS (
					SELECT DISTINCT evidence_hash FROM substrate_slots
					UNION
					SELECT DISTINCT evidence_hash FROM individual_probes
				),
				probe_keys AS (
					SELECT pe.evidence_hash, pn.probe
					FROM probe_evidence pe
					CROSS JOIN probe_names pn
				),
				joined AS (
					SELECT pk.evidence_hash, pk.probe,
					       CASE
					         WHEN ss.substrate_source = 'substrate' THEN 'substrate'
					         WHEN ip.llm_source = 'llm' THEN 'llm'
					         WHEN ip.llm_source = 'abstain' OR ip.llm_answer = 'abstain' THEN 'abstain'
					         WHEN ip.llm_source IS NOT NULL THEN ip.llm_source
					         ELSE NULL
					       END AS final_source
					FROM probe_keys pk
					LEFT JOIN substrate_slots ss
					  ON ss.evidence_hash = pk.evidence_hash AND ss.probe = pk.probe
					LEFT JOIN individual_probes ip
					  ON ip.evidence_hash = pk.evidence_hash AND ip.probe = pk.probe
				)`;
			const sql = `
				WITH ${probeCoverageCtes},
				per_evidence AS (
					SELECT evidence_hash,
					       SUM(CASE WHEN final_source = 'substrate' THEN 1 ELSE 0 END) AS substrate_count,
				       SUM(CASE WHEN final_source = 'llm' THEN 1 ELSE 0 END) AS llm_count,
				       SUM(CASE WHEN final_source = 'abstain' THEN 1 ELSE 0 END) AS abstain_count,
				       SUM(CASE WHEN final_source IS NULL THEN 1 ELSE 0 END) AS notrun_count
				FROM joined
				GROUP BY evidence_hash
			)
			SELECT
				probe,
				COUNT(*) AS total,
				SUM(CASE WHEN final_source='substrate' THEN 1 ELSE 0 END) AS substrate_n,
				SUM(CASE WHEN final_source='llm' THEN 1 ELSE 0 END) AS llm_n,
				SUM(CASE WHEN final_source='abstain' THEN 1 ELSE 0 END) AS abstain_n,
				SUM(CASE WHEN final_source IS NULL THEN 1 ELSE 0 END) AS notrun_n
			FROM joined
			GROUP BY probe
			ORDER BY probe`;
		const perProbeRaw = await rows<{
			probe: string;
			total: number;
			substrate_n: number;
			llm_n: number;
			abstain_n: number;
			notrun_n: number;
			}>(con, sql);
			const per_probe: ProbeCoverageRow[] = perProbeRaw.map((r) => ({
				probe: r.probe as ProbeKind,
				total: Number(r.total ?? 0),
				substrate_n: Number(r.substrate_n ?? 0),
				llm_n: Number(r.llm_n ?? 0),
				abstain_n: Number(r.abstain_n ?? 0),
				notrun_n: Number(r.notrun_n ?? 0)
			}));

		const aggRows = await rows<{
			n_evidences: number;
			all_substrate: number;
			short_circuited: number;
			mean_invoked: number;
		}>(
				con,
				`WITH ${probeCoverageCtes},
				per_evidence AS (
					SELECT evidence_hash,
					       SUM(CASE WHEN final_source='substrate' THEN 1 ELSE 0 END) AS substrate_count,
				       SUM(CASE WHEN final_source='llm' THEN 1 ELSE 0 END) AS llm_count,
				       SUM(CASE WHEN final_source='abstain' THEN 1 ELSE 0 END) AS abstain_count,
				       SUM(CASE WHEN final_source IS NULL THEN 1 ELSE 0 END) AS notrun_count
				FROM joined
				GROUP BY evidence_hash
			)
			SELECT
				COUNT(*) AS n_evidences,
				SUM(CASE WHEN llm_count=0 AND abstain_count=0 THEN 1 ELSE 0 END) AS all_substrate,
				SUM(CASE WHEN notrun_count > 0 THEN 1 ELSE 0 END) AS short_circuited,
				AVG(substrate_count + llm_count + abstain_count) AS mean_invoked
			FROM per_evidence`
		);

			const agg = aggRows[0] ?? { n_evidences: 0, all_substrate: 0, short_circuited: 0, mean_invoked: 0 };
			const n_evidences = Number(runRows[0].n_evidences ?? 0);
			const n_probe_evidences = Number(agg.n_evidences ?? 0);
			return {
				run_id,
				architecture,
				applicability: 'arch_conditioned',
				not_defined_reason: null,
				n_evidences,
				n_probe_evidences,
				per_probe,
				trace_diagnostic,
				all_substrate_rate: n_probe_evidences > 0 ? Number(agg.all_substrate ?? 0) / n_probe_evidences : 0,
				short_circuited_rate: n_probe_evidences > 0 ? Number(agg.short_circuited ?? 0) / n_probe_evidences : 0,
				mean_invoked_probes: Number(agg.mean_invoked ?? 0)
			};
	} finally {
		con.disconnectSync?.();
	}
}

export async function getTraceFidelity(
	run_id: string,
	options: TraceFidelityOptions = {}
): Promise<TraceFidelitySummary | null> {
	if (!dbExists()) return null;
	const con = await connect();
	try {
		const qr = run_id.replace(/'/g, "''");
		const runRows = await rows<{ architecture: string | null; paired_run_group_id: string | null; status: string | null }>(
			con,
			`SELECT architecture, paired_run_group_id, status
			 FROM score_run
			 WHERE run_id='${qr}'`
		);
		if (runRows.length === 0) return null;
		const architecture = runRows[0].architecture ?? 'unknown';
		const pairedGroup = runRows[0].paired_run_group_id ?? null;
		const runStatus = runRows[0].status ?? 'unknown';
		const requestedDetailLimit = Number.isFinite(options.detailLimit)
			? Math.floor(options.detailLimit as number)
			: TRACE_FIDELITY_DETAIL_LIMIT;
		const detailLimit = Math.min(100, Math.max(1, requestedDetailLimit));
		const detailSnapshotStartedAt = await resolveTraceSnapshotStartedAt(
			con,
			run_id,
			options.detailSnapshotStartedAt,
			runStatus === 'running'
		);
		const summaryRows = await rows<{ trace_state: string; n: number | bigint }>(
			con,
			`WITH ${traceEvidenceCteForRun(run_id, architecture, runStatus, detailSnapshotStartedAt)}
			 SELECT trace_state, COUNT(*) AS n
			 FROM trace_rows
			 GROUP BY trace_state`
		);

		const counts = zeroTraceCounts();
		for (const row of summaryRows) {
			if (!isTraceFidelityState(row.trace_state)) {
				throw new Error(`trace CTE emitted invalid trace_state: ${row.trace_state}`);
			}
			counts[row.trace_state] = Number(row.n ?? 0);
		}
		const nEvidences = Object.values(counts).reduce((sum, n) => sum + n, 0);
		const requestedOffset = Math.max(0, Math.floor(options.detailOffset ?? 0));
		const requestedPageOffset = Math.floor(requestedOffset / detailLimit) * detailLimit;
		const maxOffset = nEvidences > 0 ? Math.floor((nEvidences - 1) / detailLimit) * detailLimit : 0;
		const detailOffset = Math.min(requestedPageOffset, maxOffset);
		const traceRows = await rows<{
			stmt_hash: string | null;
			evidence_hash: string;
			trace_state: string;
			captured_step_kinds: string | null;
			n_parse_claim: number | bigint | null;
			n_build_context: number | bigint | null;
			n_substrate_route: number | bigint | null;
			n_subject_role_probe: number | bigint | null;
			n_object_role_probe: number | bigint | null;
			n_relation_axis_probe: number | bigint | null;
			n_scope_probe: number | bigint | null;
			n_grounding: number | bigint | null;
			n_adjudicate: number | bigint | null;
			has_tier: number | bigint | null;
			has_grounding_status: number | bigint | null;
			has_parse_state: number | bigint | null;
			has_call_log: number | bigint | null;
			has_raw_text: number | bigint | null;
			has_selected_examples: number | bigint | null;
		}>(
			con,
			`WITH ${traceEvidenceCteForRun(run_id, architecture, runStatus, detailSnapshotStartedAt)}
			 SELECT stmt_hash,
			        evidence_hash,
			        trace_state,
			        captured_step_kinds,
			        n_parse_claim,
			        n_build_context,
			        n_substrate_route,
			        n_subject_role_probe,
			        n_object_role_probe,
			        n_relation_axis_probe,
			        n_scope_probe,
			        n_grounding,
			        n_adjudicate,
			        has_tier,
			        has_grounding_status,
			        has_parse_state,
			        has_call_log,
			        has_raw_text,
			        has_selected_examples
			 FROM trace_rows
			 ORDER BY evidence_hash
			 LIMIT ${detailLimit} OFFSET ${detailOffset}`
		);

		const rowsOut: TraceFidelityRow[] = [];
		const present = (v: number | bigint | null | undefined) => Number(v ?? 0) > 0;
		for (const traceRow of traceRows) {
			const evidence_hash = traceRow.evidence_hash;
			const captured_steps = (traceRow.captured_step_kinds ?? '')
				.split(',')
				.filter(Boolean)
				.sort((a, b) => stepOrder(a) - stepOrder(b));
			if (!isTraceFidelityState(traceRow.trace_state)) {
				throw new Error(`trace CTE emitted invalid trace_state: ${traceRow.trace_state}`);
			}
			const state = traceRow.trace_state;
			let missing_native_steps: string[] = [];
			let note = '';

			if (state === 'step_error') {
				note = 'one or more persisted scorer steps carries an error';
			} else if (state === 'missing_aggregate') {
				note = 'no aggregate row was persisted for this evidence in a succeeded run';
				missing_native_steps.push('aggregate');
			} else if (state === 'terminated_inflight') {
				note = runStatus === 'running'
					? 'this evidence has persisted pre-aggregate steps and may still reach aggregate'
					: 'this evidence was in flight when the run terminated; no aggregate verdict was finalized';
				missing_native_steps.push('aggregate');
			} else if (architecture === 'decomposed') {
				const core = ['parse_claim', 'build_context', 'substrate_route'];
				const hasCore = {
					parse_claim: present(traceRow.n_parse_claim),
					build_context: present(traceRow.n_build_context),
					substrate_route: present(traceRow.n_substrate_route)
				};
				const coreMissing = core.filter((k) => !hasCore[k as keyof typeof hasCore]);
				const probeSteps = [
					'subject_role_probe',
					'object_role_probe',
					'relation_axis_probe',
					'scope_probe'
				];
				const hasProbe = {
					subject_role_probe: present(traceRow.n_subject_role_probe),
					object_role_probe: present(traceRow.n_object_role_probe),
					relation_axis_probe: present(traceRow.n_relation_axis_probe),
					scope_probe: present(traceRow.n_scope_probe)
				};
				const probeMissing = probeSteps.filter((k) => !hasProbe[k as keyof typeof hasProbe]);
				const fullTail = ['grounding', 'adjudicate'];
				const hasTail = {
					grounding: present(traceRow.n_grounding),
					adjudicate: present(traceRow.n_adjudicate)
				};
				const tailMissing = fullTail.filter((k) => !hasTail[k as keyof typeof hasTail]);
				if (state === 'aggregate_only') {
					missing_native_steps = [...core, ...probeSteps, ...fullTail];
					note = 'only the final aggregate dict is present; decomposed substeps were not persisted for this run';
				} else if (state === 'full') {
					note = 'native decomposed trace rows include all named probe events through adjudication';
				} else {
					missing_native_steps = [...coreMissing, ...probeMissing, ...tailMissing];
					note = probeMissing.length > 0
						? 'decomposed trace rows are visible, but one or more named probe event rows is absent; route slots alone do not prove full probe-event persistence'
						: tailMissing.length > 0
							? 'deterministic and probe event rows are visible, while later grounding/adjudication remains collapsed into the aggregate row'
							: 'some native decomposed substeps are visible, but the trace is not complete';
				}
			} else if (architecture === 'monolithic') {
				const hasTier = present(traceRow.has_tier);
				const hasGroundingStatus = present(traceRow.has_grounding_status);
				const hasParseState = present(traceRow.has_parse_state);
				const hasCallLog = present(traceRow.has_call_log);
				const hasRawText = present(traceRow.has_raw_text);
				const hasSelectedExamples = present(traceRow.has_selected_examples);
				if (state === 'full') {
					note = 'monolithic tier, grounding, call, examples, and parse fields are captured';
				} else if (state === 'partial') {
					missing_native_steps = [
						...(hasTier ? [] : ['tier']),
						...(hasGroundingStatus ? [] : ['grounding_status']),
						...(hasParseState ? [] : ['verdict_or_score']),
						...(hasCallLog ? [] : ['call_log']),
						...(hasRawText ? [] : ['raw_text']),
						...(hasSelectedExamples ? [] : ['selected_examples'])
					];
					note = 'monolithic native aggregate fields are present, but not every diagnostic field is persisted';
				} else {
					missing_native_steps = ['tier', 'grounding_status', 'verdict_or_score', 'call_log', 'raw_text', 'selected_examples'];
					note = 'the final dict exists, but monolithic native diagnostic fields are absent';
				}
			} else {
				missing_native_steps = ['architecture'];
				note = 'run architecture is unknown, so only aggregate presence can be interpreted';
			}

			rowsOut.push({
				stmt_hash: traceRow.stmt_hash ?? '',
				evidence_hash,
				state,
				captured_steps,
				missing_native_steps,
				note
			});
		}

		const native_grammar = architecture === 'monolithic'
			? ['tier-1 grounding gate', 'tier-2 comprehension/tool path', 'model call log', 'response parse', 'verdict bucket']
			: architecture === 'decomposed'
				? ['parse claim', 'build context', 'substrate route', 'subject-role probe', 'object-role probe', 'relation-axis probe', 'scope probe', 'grounding/adjudicate', 'aggregate verdict']
				: ['aggregate verdict'];
		const limitations = architecture === 'monolithic'
			? ['selected example IDs are expected on monolithic LLM-tier aggregate rows; deterministic tier-1 rows may have no example prompt', 'decomposed probe slots are not part of this architecture']
			: architecture === 'decomposed'
				? ['full fidelity requires the four named probe event rows; substrate-route slots alone are partial trace evidence', 'LLM-escalated probe details and adjudication may still be collapsed into aggregate call_log rows']
				: ['architecture is unknown; trace fidelity is limited to aggregate row presence'];

		const panel_applicability: PanelApplicabilityRow[] = [
			{
				panel: 'verdict, belief, cost, latency',
				applicability: 'arch_blind',
				reason: 'these fields are read from aggregate scorer_step rows and apply to both architectures',
				cohort_href: `/runs/${run_id}`
			},
			{
				panel: 'four-probe substrate/LLM coverage',
				applicability: architecture === 'decomposed' ? 'arch_conditioned' : 'not_defined',
				reason: architecture === 'decomposed'
					? 'this run uses the decomposed four-probe grammar'
					: 'monolithic runs do not emit subject/object/relation/scope probe slots',
				cohort_href: architecture === 'decomposed' ? `/runs/${run_id}` : null
			},
			{
				panel: 'native monolithic tier path',
				applicability: architecture === 'monolithic' ? 'arch_conditioned' : 'not_defined',
				reason: architecture === 'monolithic'
					? 'this run emits tier and grounding-path fields on aggregate rows'
					: 'decomposed runs route through deterministic/probe/adjudication stages instead',
				cohort_href: architecture === 'monolithic' ? `/runs/${run_id}` : null
			},
			{
				panel: 'paired architecture deltas',
				applicability: pairedGroup ? 'paired_only' : 'not_defined',
				reason: pairedGroup
					? `requires overlap inside paired group ${pairedGroup}`
					: 'requires a shared paired_run_group_id before deltas are meaningful',
				cohort_href: pairedGroup ? `/pairs/${pairedGroup}` : null
			}
		];

		return {
			run_id,
			architecture,
			n_evidences: nEvidences,
			counts,
			native_grammar,
			limitations,
			rows: rowsOut,
			detail_offset: detailOffset,
			detail_limit: detailLimit,
			detail_snapshot_started_at: detailSnapshotStartedAt,
			panel_applicability
		};
	} finally {
		con.disconnectSync?.();
	}
}

export interface ResidualDistribution {
	run_id: string;
	bins: number[];
	n_total: number;
	mean_residual: number | null;
}

/**
 * Histogram of (our_belief − indra_belief) for the latest succeeded run.
 * Always 11 bins on [-1, +1]; bin 5 (index 5) is the zero-centered bucket.
 */
export async function getResidualDistribution(run_id?: string): Promise<ResidualDistribution | null> {
	if (!dbExists()) return null;
	const con = await connect();
	try {
		let resolved = run_id ?? null;
		if (!resolved) {
			const r = await rows<{ run_id: string }>(
				con,
				`SELECT run_id FROM score_run WHERE status='succeeded' ORDER BY started_at DESC LIMIT 1`
			);
			resolved = r[0]?.run_id ?? null;
		}
		if (!resolved) return null;
		const qr = resolved.replace(/'/g, "''");
		const residualRows = await rows<{ residual: number }>(
			con,
			`WITH ours AS (
				SELECT stmt_hash,
				       AVG(CAST(json_extract(output_json, '$.score') AS DOUBLE)) AS our
				FROM scorer_step
				WHERE run_id='${qr}' AND step_kind='aggregate'
				  AND json_extract(output_json, '$.score') IS NOT NULL
				GROUP BY stmt_hash
			)
			SELECT (ours.our - s.indra_belief) AS residual
			FROM statement s
			JOIN ours ON ours.stmt_hash = s.stmt_hash
			WHERE s.indra_belief IS NOT NULL AND ours.our IS NOT NULL`
		);
		const bins = new Array(11).fill(0);
		let sum = 0;
		for (const r of residualRows) {
			const v = r.residual;
			if (typeof v !== 'number' || Number.isNaN(v)) continue;
			sum += v;
			const clamped = Math.max(-1, Math.min(1, v));
			const idx = Math.min(10, Math.max(0, Math.floor((clamped + 1) * 5.5)));
			bins[idx] += 1;
		}
		return {
			run_id: resolved,
			bins,
			n_total: residualRows.length,
			mean_residual: residualRows.length > 0 ? sum / residualRows.length : null
		};
	} finally {
		con.disconnectSync?.();
	}
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

const FINDING_K = 5;

/**
 * Rank the latest run's statements along five "interesting" axes. Each lane
 * returns up to K rows with the same shape; the UI renders them as compact
 * BeliefPrimitive cards. Empty lanes are normal — e.g. no prev_run means no
 * regressions can be computed.
 */
export async function getFindings(): Promise<Findings | null> {
	if (!dbExists()) return null;
	const con = await connect();
	try {
		const latestRows = await rows<{ run_id: string; architecture: string }>(
			con,
			`SELECT run_id, architecture FROM score_run
			 WHERE status='succeeded'
			 ORDER BY started_at DESC LIMIT 1`
		);
		if (latestRows.length === 0) return null;
		const run_id = latestRows[0].run_id;
		const qa = (latestRows[0].architecture ?? 'unknown').replace(/'/g, "''");
		const prevRows = await rows<{ run_id: string }>(
			con,
			`SELECT run_id FROM score_run
			 WHERE status='succeeded'
			   AND architecture='${qa}'
			   AND started_at < (SELECT started_at FROM score_run WHERE run_id='${run_id.replace(/'/g, "''")}')
			 ORDER BY started_at DESC LIMIT 1`
		);
		const prev_run_id = prevRows[0]?.run_id ?? null;
		const qr = run_id.replace(/'/g, "''");
		const qp = prev_run_id ? prev_run_id.replace(/'/g, "''") : null;

		const baseSelect = (whereExtra: string, orderBy: string) => `
				WITH ours AS (
					SELECT stmt_hash,
					       AVG(CAST(json_extract(output_json, '$.score') AS DOUBLE)) AS our,
					       CASE
					         WHEN SUM(CASE WHEN json_extract_string(output_json, '$.verdict')='incorrect' THEN 1 ELSE 0 END) > 0 THEN 'incorrect'
					         WHEN SUM(CASE WHEN json_extract_string(output_json, '$.verdict')='abstain' THEN 1 ELSE 0 END) > 0 THEN 'abstain'
					         WHEN SUM(CASE WHEN json_extract_string(output_json, '$.verdict')='correct' THEN 1 ELSE 0 END) = COUNT(*) THEN 'correct'
					         WHEN SUM(CASE WHEN json_extract_string(output_json, '$.verdict')='correct' THEN 1 ELSE 0 END) > 0 THEN 'mixed'
					         ELSE 'unknown'
					       END AS verdict
					FROM scorer_step
				WHERE run_id = '${qr}'
				  AND step_kind = 'aggregate'
				GROUP BY stmt_hash
			),
			ev_n AS (
				SELECT stmt_hash, COUNT(*) AS n_evidences FROM statement_evidence GROUP BY stmt_hash
			),
			ags AS (
				SELECT stmt_hash,
				       list({role: role, name: name}
				            ORDER BY CASE role
				              WHEN 'subj' THEN 0 WHEN 'enz' THEN 0
				              WHEN 'obj' THEN 1 WHEN 'sub' THEN 1
				              WHEN 'member' THEN 2 ELSE 3 END,
				            role_index) AS agents
				FROM agent GROUP BY stmt_hash
			)
			SELECT s.stmt_hash, s.indra_type, s.indra_belief AS indra_score,
			       ours.our AS our_score, ours.verdict AS verdict,
			       COALESCE(ev_n.n_evidences, 0) AS n_evidences,
			       ags.agents AS agents
			FROM statement s
			JOIN ours ON ours.stmt_hash = s.stmt_hash
			LEFT JOIN ev_n ON ev_n.stmt_hash = s.stmt_hash
			LEFT JOIN ags ON ags.stmt_hash = s.stmt_hash
			${whereExtra}
			ORDER BY ${orderBy}
			LIMIT ${FINDING_K}`;

		type Row = {
			stmt_hash: string;
			indra_type: string;
			indra_score: number | null;
			our_score: number | null;
			verdict: string | null;
			n_evidences: number;
			agents: Array<{ role: string; name: string }> | null;
		};

		const [disagreeRows, probeSplitRows, lowConfRows, prevAggRows] = await Promise.all([
			rows<Row>(
				con,
				baseSelect(
					`WHERE s.indra_belief IS NOT NULL AND ours.our IS NOT NULL`,
					`ABS(ours.our - s.indra_belief) DESC, s.stmt_hash`
				)
			),
			rows<Row & { probe_stdev: number }>(
				con,
				`WITH probe_votes AS (
					SELECT stmt_hash,
					       CASE
					         WHEN step_kind = 'subject_role_probe' AND json_extract_string(output_json, '$.answer') = 'present_as_subject' THEN 1.0
					         WHEN step_kind = 'object_role_probe'  AND json_extract_string(output_json, '$.answer') = 'present_as_object'  THEN 1.0
					         WHEN step_kind = 'relation_axis_probe' AND json_extract_string(output_json, '$.answer') = 'direct_sign_match' THEN 1.0
					         WHEN step_kind = 'scope_probe' AND json_extract_string(output_json, '$.answer') = 'asserted' THEN 1.0
					         WHEN step_kind = 'scope_probe' AND json_extract_string(output_json, '$.answer') = 'negated'  THEN -1.0
					         WHEN json_extract_string(output_json, '$.answer') IN ('absent','present_as_decoy','direct_sign_mismatch','direct_axis_mismatch','no_relation') THEN -1.0
					         ELSE 0
					       END AS vote
					FROM scorer_step
					WHERE run_id = '${qr}'
					  AND step_kind IN ('subject_role_probe','object_role_probe','relation_axis_probe','scope_probe')
				),
				stdevs AS (
					SELECT stmt_hash, STDDEV_POP(vote) AS probe_stdev
					FROM probe_votes GROUP BY stmt_hash
				)
				SELECT s.stmt_hash, s.indra_type, s.indra_belief AS indra_score,
					       (SELECT AVG(CAST(json_extract(output_json, '$.score') AS DOUBLE))
					        FROM scorer_step WHERE run_id='${qr}'
					          AND step_kind='aggregate' AND stmt_hash=s.stmt_hash) AS our_score,
					       (SELECT CASE
					          WHEN SUM(CASE WHEN json_extract_string(output_json, '$.verdict')='incorrect' THEN 1 ELSE 0 END) > 0 THEN 'incorrect'
					          WHEN SUM(CASE WHEN json_extract_string(output_json, '$.verdict')='abstain' THEN 1 ELSE 0 END) > 0 THEN 'abstain'
					          WHEN SUM(CASE WHEN json_extract_string(output_json, '$.verdict')='correct' THEN 1 ELSE 0 END) = COUNT(*) THEN 'correct'
					          WHEN SUM(CASE WHEN json_extract_string(output_json, '$.verdict')='correct' THEN 1 ELSE 0 END) > 0 THEN 'mixed'
					          ELSE 'unknown'
					        END
					        FROM scorer_step WHERE run_id='${qr}'
					          AND step_kind='aggregate' AND stmt_hash=s.stmt_hash) AS verdict,
				       (SELECT COUNT(*) FROM statement_evidence WHERE stmt_hash=s.stmt_hash) AS n_evidences,
				       (SELECT list({role: role, name: name}
				                    ORDER BY CASE role
				                      WHEN 'subj' THEN 0 WHEN 'enz' THEN 0
				                      WHEN 'obj' THEN 1 WHEN 'sub' THEN 1
				                      WHEN 'member' THEN 2 ELSE 3 END,
				                    role_index)
				        FROM agent WHERE stmt_hash=s.stmt_hash) AS agents,
				       stdevs.probe_stdev AS probe_stdev
				FROM statement s
				JOIN stdevs ON stdevs.stmt_hash = s.stmt_hash
				WHERE stdevs.probe_stdev > 0
				ORDER BY stdevs.probe_stdev DESC, s.stmt_hash
				LIMIT ${FINDING_K}`
			),
			rows<Row>(
				con,
				baseSelect(
					`WHERE ours.our BETWEEN 0.4 AND 0.6
					  AND COALESCE(ev_n.n_evidences, 0) >= 3`,
					`ev_n.n_evidences DESC, ABS(ours.our - 0.5) ASC, s.stmt_hash`
				)
			),
			prev_run_id
				? rows<{ stmt_hash: string; verdict: string }>(
						con,
							`SELECT stmt_hash,
							        CASE
							          WHEN SUM(CASE WHEN json_extract_string(output_json, '$.verdict')='incorrect' THEN 1 ELSE 0 END) > 0 THEN 'incorrect'
							          WHEN SUM(CASE WHEN json_extract_string(output_json, '$.verdict')='abstain' THEN 1 ELSE 0 END) > 0 THEN 'abstain'
							          WHEN SUM(CASE WHEN json_extract_string(output_json, '$.verdict')='correct' THEN 1 ELSE 0 END) = COUNT(*) THEN 'correct'
							          WHEN SUM(CASE WHEN json_extract_string(output_json, '$.verdict')='correct' THEN 1 ELSE 0 END) > 0 THEN 'mixed'
							          ELSE 'unknown'
							        END AS verdict
							 FROM scorer_step WHERE run_id='${qp}' AND step_kind='aggregate'
							 GROUP BY stmt_hash`
					)
				: Promise.resolve([])
		]);

		const prevVerdict = new Map(prevAggRows.map((r) => [r.stmt_hash, r.verdict]));
		const allCurrent = await rows<Row>(
			con,
			baseSelect('WHERE 1=1', 's.stmt_hash')
		);
		const verdictMoved = (curr: string | null, prev: string | undefined, dir: 'down' | 'up') => {
			if (!curr || !prev) return false;
			if (dir === 'down') return prev === 'correct' && curr === 'incorrect';
			return prev === 'incorrect' && curr === 'correct';
		};
		const regressions: Row[] = [];
		const recoveries: Row[] = [];
		for (const r of allCurrent) {
			const pv = prevVerdict.get(r.stmt_hash);
			if (verdictMoved(r.verdict, pv, 'down')) regressions.push(r);
			if (verdictMoved(r.verdict, pv, 'up')) recoveries.push(r);
		}

		const toFinding = (
			kind: FindingKind,
			r: Row,
			rank: number,
			why: string
		): FindingRow => ({
			kind,
			stmt_hash: r.stmt_hash,
			indra_type: r.indra_type,
			agents: r.agents ?? [],
			our_score: r.our_score,
			indra_score: r.indra_score,
			n_evidences: r.n_evidences,
			rank_value: rank,
			why_text: why
		});

		const fmtSigned = (n: number) => `${n >= 0 ? '+' : '−'}${Math.abs(n).toFixed(2)}`;

		return {
			run_id,
			prev_run_id,
			biggest_disagreement: disagreeRows.map((r) => {
				const d = (r.our_score ?? 0) - (r.indra_score ?? 0);
				return toFinding(
					'biggest_disagreement',
					r,
					Math.abs(d),
					`Δ ${fmtSigned(d)} vs INDRA · n_ev=${r.n_evidences}`
				);
			}),
			probe_split: probeSplitRows.map((r) =>
				toFinding(
					'probe_split',
					r,
					r.probe_stdev,
					`probe stdev ${r.probe_stdev.toFixed(2)} · disagreement across the four probes`
				)
			),
			verdict_regression: regressions
				.slice(0, FINDING_K)
				.map((r) =>
					toFinding(
						'verdict_regression',
						r,
						1,
						`verdict moved correct → incorrect since prev run`
					)
				),
			verdict_recovery: recoveries
				.slice(0, FINDING_K)
				.map((r) =>
					toFinding(
						'verdict_recovery',
						r,
						1,
						`verdict moved incorrect → correct since prev run`
					)
				),
			low_confidence_high_stakes: lowConfRows.map((r) =>
				toFinding(
					'low_confidence_high_stakes',
					r,
					r.n_evidences,
					`belief ${(r.our_score ?? 0).toFixed(2)} (mid-range) · n_ev=${r.n_evidences} ≥ 3`
				)
			)
		};
	} finally {
		con.disconnectSync?.();
	}
}

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

/**
 * Pick a focus statement to lead the dashboard with. Defaults to the
 * highest-|Δ vs INDRA| in the latest succeeded run; can be deep-linked to a
 * specific stmt_hash. Returns null if there's no scoring data yet.
 */
export async function getFocusStatement(
	focus_hash?: string,
	run_id?: string
): Promise<FocusStatement | null> {
	if (!dbExists()) return null;
	const con = await connect();
	try {
		let resolvedRun = run_id ?? null;
		if (!resolvedRun) {
			const r = await rows<{ run_id: string }>(
				con,
				`SELECT run_id FROM score_run WHERE status='succeeded' ORDER BY started_at DESC LIMIT 1`
			);
			resolvedRun = r[0]?.run_id ?? null;
		}
		if (!resolvedRun) return null;

		let resolvedHash = focus_hash ?? null;
		let whyKind: 'biggest_delta' | 'requested' = 'biggest_delta';
		let biggestDelta: number | null = null;
		if (!resolvedHash) {
			const r = await rows<{ stmt_hash: string; delta: number }>(
				con,
				`WITH ours AS (
					SELECT stmt_hash,
					       AVG(CAST(json_extract(output_json, '$.score') AS DOUBLE)) AS our
					FROM scorer_step
					WHERE run_id = '${resolvedRun.replace(/'/g, "''")}'
					  AND step_kind = 'aggregate'
					  AND json_extract(output_json, '$.score') IS NOT NULL
					GROUP BY stmt_hash
				)
				SELECT s.stmt_hash AS stmt_hash,
				       ours.our - s.indra_belief AS delta
				FROM statement s
				JOIN ours ON ours.stmt_hash = s.stmt_hash
				WHERE s.indra_belief IS NOT NULL
				ORDER BY ABS(ours.our - s.indra_belief) DESC, s.stmt_hash
				LIMIT 1`
			);
			if (r.length === 0) return null;
			resolvedHash = r[0].stmt_hash;
			biggestDelta = r[0].delta;
		} else {
			whyKind = 'requested';
		}

		const stmtRows = await rows<{
			stmt_hash: string;
			indra_type: string;
			indra_belief: number | null;
			our_score: number | null;
			n_evidences: number;
		}>(
			con,
			`WITH ours AS (
				SELECT stmt_hash,
				       AVG(CAST(json_extract(output_json, '$.score') AS DOUBLE)) AS our
				FROM scorer_step
				WHERE run_id = '${resolvedRun.replace(/'/g, "''")}'
				  AND step_kind = 'aggregate'
				  AND json_extract(output_json, '$.score') IS NOT NULL
				GROUP BY stmt_hash
			)
			SELECT s.stmt_hash, s.indra_type, s.indra_belief,
			       ours.our AS our_score,
			       (SELECT COUNT(*) FROM statement_evidence WHERE stmt_hash = s.stmt_hash) AS n_evidences
			FROM statement s
			LEFT JOIN ours ON ours.stmt_hash = s.stmt_hash
			WHERE s.stmt_hash = '${resolvedHash.replace(/'/g, "''")}'`
		);
		if (stmtRows.length === 0) return null;
		const s = stmtRows[0];

		const [agents, evidences, probes] = await Promise.all([
			rows<{ role: string; name: string }>(
				con,
				`SELECT role, name FROM agent
				 WHERE stmt_hash = '${resolvedHash.replace(/'/g, "''")}'
				 ORDER BY CASE role
				   WHEN 'subj' THEN 0 WHEN 'enz' THEN 0
				   WHEN 'obj' THEN 1 WHEN 'sub' THEN 1
				   WHEN 'member' THEN 2 ELSE 3 END, role_index`
			),
			rows<{ evidence_hash: string; source_api: string | null; text: string | null }>(
				con,
				`SELECT e.evidence_hash, e.source_api, e.text
				 FROM statement_evidence se
				 JOIN evidence e ON e.evidence_hash = se.evidence_hash
				 WHERE se.stmt_hash = '${resolvedHash.replace(/'/g, "''")}'
				 ORDER BY (e.text IS NULL), length(e.text) DESC LIMIT 3`
			),
			getProbeAttribution(resolvedRun, resolvedHash)
		]);

		const evPlural = s.n_evidences === 1 ? '' : 's';
		const evText = `${s.n_evidences} evidence${evPlural}`;
		// `why_this_one` only renders when the system chose the focus editorially
		// (largest |Δ|). When the user deep-linked to a stmt_hash, the URL
		// already explains why they're here — silencing avoids self-evident
		// bookkeeping ("opened via deep-link").
		const whyText = whyKind === 'biggest_delta'
			? `the largest disagreement with INDRA in this run · ${evText}`
			: '';

		return {
			run_id: resolvedRun,
			stmt: { stmt_hash: s.stmt_hash, indra_type: s.indra_type, agents },
			our_score: s.our_score,
			indra_score: s.indra_belief,
			probes,
			evidences,
			n_evidences: s.n_evidences,
			why_this_one: whyText
		};
	} finally {
		con.disconnectSync?.();
	}
}

const PROBE_KINDS: Record<string, ProbeKind> = {
	subject_role_probe: 'subject_role',
	object_role_probe: 'object_role',
	relation_axis_probe: 'relation_axis',
	scope_probe: 'scope'
};

/**
 * For a (run_id, stmt_hash), return the four probes' contributions to the
 * final score. When `evidence_hash` is supplied, returns the per-evidence
 * view; otherwise picks the highest-confidence answer per probe across all
 * evidences for that statement. Pure logic lives in probeAttribution.ts.
 */
export async function getProbeAttribution(
	run_id: string,
	stmt_hash: string,
	evidence_hash?: string
): Promise<ProbeAttribution[]> {
	if (!dbExists()) return [];
	const con = await connect();
	try {
		const evClause = evidence_hash
			? `AND evidence_hash = '${evidence_hash.replace(/'/g, "''")}'`
			: '';
		const [stepRows, substrateRows] = await Promise.all([
			rows<{
				step_kind: string;
				evidence_hash: string | null;
				answer: string | null;
				confidence: string | null;
				source: string | null;
				rationale: string | null;
			}>(
				con,
				`SELECT
				   step_kind,
				   evidence_hash,
				   json_extract_string(output_json, '$.answer') AS answer,
				   json_extract_string(output_json, '$.confidence') AS confidence,
				   json_extract_string(output_json, '$.source') AS source,
				   json_extract_string(output_json, '$.rationale') AS rationale
				 FROM scorer_step
				 WHERE run_id = '${run_id.replace(/'/g, "''")}'
				   AND stmt_hash = '${stmt_hash.replace(/'/g, "''")}'
				   AND step_kind IN ('subject_role_probe', 'object_role_probe', 'relation_axis_probe', 'scope_probe')
				   ${evClause}`
			),
			rows<{ evidence_hash: string | null; output_json: string }>(
				con,
				`SELECT evidence_hash, output_json::VARCHAR AS output_json
				 FROM scorer_step
				 WHERE run_id = '${run_id.replace(/'/g, "''")}'
				   AND stmt_hash = '${stmt_hash.replace(/'/g, "''")}'
				   AND step_kind = 'substrate_route'
				   ${evClause}`
			)
		]);

		const probeOutputs: ProbeOutput[] = [];
		const seen = new Set<string>();
		for (const r of stepRows) {
			const probe = PROBE_KINDS[r.step_kind];
			if (!probe) continue;
			probeOutputs.push({
				probe,
				evidence_hash: r.evidence_hash,
				answer: r.answer,
				confidence: (r.confidence as ProbeConfidence) ?? null,
				source: (r.source as ProbeSource) ?? null,
				rationale: r.rationale
			});
			seen.add(`${r.evidence_hash ?? ''}|${probe}`);
		}

		for (const sr of substrateRows) {
			let parsed: Record<string, { source?: string; answer?: string; confidence?: string }> | null = null;
			try {
				parsed = JSON.parse(sr.output_json);
			} catch {
				continue;
			}
			if (!parsed) continue;
			const probesInRow: ProbeKind[] = ['subject_role', 'object_role', 'relation_axis', 'scope'];
			for (const probe of probesInRow) {
				if (seen.has(`${sr.evidence_hash ?? ''}|${probe}`)) continue;
				const slot = parsed[probe];
				if (!slot || !slot.answer) continue;
				probeOutputs.push({
					probe,
					evidence_hash: sr.evidence_hash,
					answer: slot.answer ?? null,
					confidence: (slot.confidence as ProbeConfidence) ?? null,
					source: (slot.source as ProbeSource) ?? 'substrate',
					rationale: `substrate-resolved (no LLM call)`
				});
				seen.add(`${sr.evidence_hash ?? ''}|${probe}`);
			}
		}

		if (evidence_hash) {
			return computeAttributions(probeOutputs);
		}
		return computeAttributions(summarizeAcrossEvidences(probeOutputs));
	} finally {
		con.disconnectSync?.();
	}
}

export async function getStatementDetail(stmt_hash: string, run_id?: string | null): Promise<StatementDetail | null> {
	if (!dbExists()) return null;
	const con = await connect();
	try {
		const qStmt = stmt_hash.replace(/'/g, "''");
		const stmtRows = await rows<{
			stmt_hash: string;
			indra_type: string;
			indra_belief: number | null;
			supports_count: number;
			supported_by_count: number;
			source_dump_id: string | null;
			raw_json: string;
		}>(
			con,
			`SELECT stmt_hash, indra_type, indra_belief, supports_count,
			        supported_by_count, source_dump_id, raw_json::VARCHAR AS raw_json
			 FROM statement WHERE stmt_hash = '${qStmt}'`
		);
		if (stmtRows.length === 0) return null;
		const s = stmtRows[0];

		const available_runs = await rows<StatementRunOption>(
			con,
			`SELECT DISTINCT
			   sr.run_id,
			   sr.architecture,
			   sr.scorer_version,
			   sr.model_id_default,
			   sr.started_at::VARCHAR AS started_at,
			   sr.status
			 FROM score_run sr
			 JOIN scorer_step ss ON ss.run_id = sr.run_id
			 WHERE ss.stmt_hash = '${qStmt}'
			 ORDER BY sr.started_at DESC`
		);
		const requestedRun = run_id ? run_id.replace(/'/g, "''") : null;
		const selectedRun =
			(requestedRun
				? available_runs.find((r) => r.run_id === requestedRun)
				: available_runs.find((r) => r.status === 'succeeded') ?? available_runs[0]) ?? null;
		const scorerStepRunClause = selectedRun
			? `AND run_id = '${selectedRun.run_id.replace(/'/g, "''")}'`
			: 'AND 1=0';

		const [agents, evidences, truth_labels, registered_truth_sets_rows, supports_edges, scorer_steps] = await Promise.all([
			rows<AgentRow>(
				con,
				`SELECT agent_hash, role, role_index, name,
				        db_refs_json::VARCHAR AS db_refs_json,
				        mods_json::VARCHAR AS mods_json,
				        location
				 FROM agent
				 WHERE stmt_hash = '${qStmt}'
				 ORDER BY
				   CASE role
				     WHEN 'subj' THEN 0
				     WHEN 'enz'  THEN 0
				     WHEN 'obj'  THEN 1
				     WHEN 'sub'  THEN 1
				     WHEN 'member' THEN 2
				     ELSE 3
				   END,
				   role_index`
			),
			rows<EvidenceRow>(
				con,
				`SELECT e.evidence_hash, e.source_api, e.source_id, e.pmid, e.text,
				        e.is_direct, e.is_negated, e.is_curated,
				        e.epistemics_json::VARCHAR AS epistemics_json
				 FROM statement_evidence se
				 JOIN evidence e ON e.evidence_hash = se.evidence_hash
				 WHERE se.stmt_hash = '${qStmt}'
				 ORDER BY e.source_api, e.evidence_hash`
			),
			rows<TruthLabelRow>(
				con,
				`SELECT truth_set_id, target_kind, target_id, field, value_text,
				        value_json::VARCHAR AS value_json, provenance
				 FROM truth_label
				 WHERE (target_kind = 'stmt' AND target_id = '${qStmt}')
				    OR (target_kind = 'evidence' AND target_id IN (
				          SELECT evidence_hash FROM statement_evidence WHERE stmt_hash = '${qStmt}'))
				    OR (target_kind = 'agent' AND target_id IN (
				          SELECT agent_hash FROM agent WHERE stmt_hash = '${qStmt}'))
				 ORDER BY truth_set_id, field`
			),
			rows<{ id: string }>(
				con,
				'SELECT id FROM truth_set ORDER BY id'
			),
			rows<SupportsEdgeRow>(
				con,
				`SELECT from_stmt_hash, to_stmt_hash, kind
				 FROM supports_edge
				 WHERE from_stmt_hash = '${qStmt}'
				 ORDER BY kind, to_stmt_hash`
			),
			rows<ScorerStepRow>(
				con,
				`SELECT step_hash, run_id, evidence_hash, scorer_version,
				        architecture, model_id, step_kind, is_substrate_answered,
				        input_payload_json::VARCHAR AS input_payload_json,
				        output_json::VARCHAR AS output_json,
				        latency_ms, prompt_tokens, out_tokens, finish_reason, error,
				        started_at::VARCHAR AS started_at
				 FROM scorer_step
				 WHERE stmt_hash = '${qStmt}' ${scorerStepRunClause}
				 ORDER BY evidence_hash,
				   CASE step_kind
				     WHEN 'parse_claim' THEN 0
				     WHEN 'build_context' THEN 1
				     WHEN 'substrate_route' THEN 2
				     WHEN 'subject_role_probe' THEN 3
				     WHEN 'object_role_probe' THEN 4
				     WHEN 'relation_axis_probe' THEN 5
				     WHEN 'scope_probe' THEN 6
				     WHEN 'grounding' THEN 7
				     WHEN 'adjudicate' THEN 8
				     WHEN 'aggregate' THEN 9
				     ELSE 99
				   END,
				   started_at DESC,
				   step_hash DESC`
			)
		]);

		return {
			...s,
			selected_run_id: selectedRun?.run_id ?? null,
			selected_architecture: selectedRun?.architecture ?? null,
			selected_scorer_version: selectedRun?.scorer_version ?? null,
			selected_run_status: selectedRun?.status ?? null,
			available_runs,
			agents,
			evidences,
			truth_labels,
			registered_truth_sets: registered_truth_sets_rows.map((r) => r.id),
			supports_edges,
			scorer_steps
		};
	} finally {
		con.disconnectSync?.();
	}
}

export async function getCorpusOverview(): Promise<CorpusOverview> {
	const path = dbPath();
	if (!dbExists()) {
		return { dbPath: path, dbExists: false, ...EMPTY_OVERVIEW };
	}

	const con = await connect();
	try {
		await assertCoreOverviewSchema(con);
		const overviewReads = [
			readScalar(con, 'SELECT COUNT(*) FROM statement') as Promise<number>,
			readScalar(con, 'SELECT COUNT(*) FROM evidence') as Promise<number>,
			readScalar(con, 'SELECT COUNT(*) FROM agent') as Promise<number>,
			readScalar(con, 'SELECT COUNT(*) FROM supports_edge') as Promise<number>,
			readScalar(con, 'SELECT COUNT(*) FROM truth_label') as Promise<number>,
			readRows<{ id: string; name: string; rowCount: number }>(
				con,
				`SELECT ts.id AS id, ts.name AS name,
				        COUNT(tl.label_id) AS "rowCount"
				 FROM truth_set ts
				 LEFT JOIN truth_label tl ON tl.truth_set_id = ts.id
				 GROUP BY ts.id, ts.name
				 ORDER BY "rowCount" DESC, ts.id`
			) as Promise<Array<{ id: string; name: string; rowCount: number }>>,
			readRows<{ source_dump_id: string | null; n: number }>(
				con,
				`SELECT source_dump_id, COUNT(*) AS n
				 FROM statement
				 GROUP BY source_dump_id
				 ORDER BY n DESC
				 LIMIT 16`
			) as Promise<Array<{ source_dump_id: string | null; n: number }>>,
			readRows<{ indra_type: string; n: number }>(
				con,
				`SELECT indra_type, COUNT(*) AS n
				 FROM statement
				 GROUP BY indra_type
				 ORDER BY n DESC
				 LIMIT 32`
			) as Promise<Array<{ indra_type: string; n: number }>>,
			readRows<{
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
			}>(
				con,
				`SELECT
				   sr.run_id,
				   sr.scorer_version,
				   sr.architecture,
				   sr.paired_run_group_id,
				   sr.started_at::VARCHAR AS started_at,
				   sr.status,
				   sr.terminated_by,
				   sr.termination_reason,
				   sr.n_stmts,
				   sr.cost_estimate_usd,
				   (SELECT value FROM metric WHERE run_id = sr.run_id
				    AND metric_name = 'indra_belief_calibration.mae' AND truth_set_id = 'indra_published_belief' LIMIT 1) AS mae,
				   (SELECT value FROM metric WHERE run_id = sr.run_id
				    AND metric_name = 'indra_belief_calibration.bias' AND truth_set_id = 'indra_published_belief' LIMIT 1) AS bias
				 FROM score_run sr
				 ORDER BY sr.started_at DESC
				 LIMIT 8`
			) as Promise<Array<{
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
			}>>,
			getLatestValidity(con) as Promise<LatestValidity | null>
		] as const;
		const settled = await Promise.allSettled(overviewReads);
		const firstRejected = settled.find((r): r is PromiseRejectedResult => r.status === 'rejected');
		if (firstRejected) throw firstRejected.reason;
		const [
			statementCount,
			evidenceCount,
			agentCount,
			supportsEdgeCount,
			truthLabelCount,
			truthSets,
			sourceDumps,
			indraTypes,
			scorerRuns,
			latestValidity
		] = settled.map((r) => (r as PromiseFulfilledResult<unknown>).value) as [
			number,
			number,
			number,
			number,
			number,
			Array<{ id: string; name: string; rowCount: number }>,
			Array<{ source_dump_id: string | null; n: number }>,
			Array<{ indra_type: string; n: number }>,
			Array<{
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
			}>,
			LatestValidity | null
		];

		// Pre-flight check: which exports already exist on disk? Avoids
		// dashboard offering ↓ links that 404 because export wasn't run.
		const exportDir = resolve(process.cwd(), '..', 'data', 'exports');
		const enrichedRuns = scorerRuns.map((r) => ({
			...r,
			hasIndraExport: existsSync(resolve(exportDir, `${r.run_id}_indra.json`)),
			hasCardExport: existsSync(resolve(exportDir, `${r.run_id}_card.json`)),
		}));

		return {
			dbPath: path,
			dbExists: true,
			statementCount,
			evidenceCount,
			agentCount,
			supportsEdgeCount,
			truthLabelCount,
			truthSets,
			sourceDumps,
			indraTypes,
			scorerRuns: enrichedRuns,
			latestValidity
		};
	} finally {
		// connections are short-lived; the DuckDBInstance is reused
		con.disconnectSync?.();
	}
}
