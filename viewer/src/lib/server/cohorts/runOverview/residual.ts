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

