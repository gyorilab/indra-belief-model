// Auto-extracted from $lib/db.ts on 2026-05-27 to satisfy B4 of the
// active goal (cohort orchestration extraction).
//
// All callers continue to import these symbols from $lib/db; db.ts
// re-exports each function from the corresponding $lib/server/cohorts/*
// module. Shared internal helpers (sqlQuote, rows, readRows, scalar,
// tableExists, normalizeDuckValue, timestampMs, durationSeconds,
// safeRate, repairRerunLineageSqlOptions, etc.) and connection
// management (connect, closeInstance, dbPath, dbExists) still live in
// $lib/db; this file imports them from there.

import { closeInstance, connect, dbExists, dbPath, durationSeconds, normalizeDuckValue, readRows, repairRerunLineageSqlOptions, resolveTraceSnapshotStartedAt, rows, safeRate, scalar, sqlQuote, tableExists, timestampMs } from '$lib/db';
import type { RunCohort, RunCohortRow } from '$lib/db';
import { aggregateEvidenceEmptyPredicates, emptyCohortDiagnostics, runScorerStepCounts, statementEmptyPredicates, traceEvidenceEmptyPredicates } from '$lib/cohorts/diagnostics';
import { aggregateEvidenceCohortBaseSql, aggregateEvidenceCohortCountSql, aggregateEvidenceCohortRowsSql, statementCohortCte, statementCohortRowsSql, traceCohortBaseSql, traceCohortCountSql, traceCohortCte, traceCohortRowsSql } from '$lib/cohorts/queries';
import { RUN_COHORT_LIMIT, nonTraceEvidenceCohortWhereClauses, nonTraceStatementCohortWhereClauses, traceStateCohortWhereClauses } from '$lib/cohorts/sql';
import type { RunCohortFilters } from '$lib/cohorts/types';
import { validateRunCohortFilterValues } from '$lib/server/runCohortContract';
import { INVALID_TRACE_SNAPSHOT, INVALID_TRACE_STATE, cleanTraceSnapshot, cleanTraceStateFilter } from '$lib/traceState';
import { error } from '@sveltejs/kit';

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
