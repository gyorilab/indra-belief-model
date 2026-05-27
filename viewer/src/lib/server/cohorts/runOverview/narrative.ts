// Split from $lib/server/cohorts/runOverview.ts on 2026-05-27
// in response to brutalist HIGH finding (one file, multiple surfaces).
// Original db.ts callers continue to import via $lib/db (re-export
// chain: db.ts -> runOverview/index.ts -> this file).

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

import { FINDING_K, TRACE_FIDELITY_DETAIL_LIMIT, closeInstance, connect, dbExists, dbPath, durationSeconds, normalizeDuckValue, readRows, repairRerunLineageSqlOptions, resolveTraceSnapshotStartedAt, rows, safeRate, scalar, sqlQuote, tableExists, timestampMs } from '$lib/db';
import type { Findings, HeuristicCoverage, ResidualDistribution, RunNarrative } from '$lib/db';
import type { ProbeKind } from '$lib/probeAttribution';
import { isTraceFidelityState, zeroTraceCounts } from '$lib/traceState';
import { traceEvidenceCteForRun } from '$lib/traceStateSql';
import type { DuckDBConnection } from '@duckdb/node-api';
import { error } from '@sveltejs/kit';

import type { HeuristicCoverageTraceDiagnostic, HeuristicCoverageTraceLifecycleKind, ProbeCoverageRow, TraceFidelityOptions, TraceFidelitySummary, TraceFidelityRow, PanelApplicabilityRow, FindingKind, FindingRow } from '$lib/db';
export async function getRunTraceIntegrityCounts(
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

