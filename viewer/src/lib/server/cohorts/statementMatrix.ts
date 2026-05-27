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

import { closeInstance, connect, dbExists, dbPath, durationSeconds, normalizeDuckValue, readRows, repairRerunLineageSqlOptions, rows, safeRate, scalar, sqlQuote, tableExists, timestampMs } from '$lib/db';
import type { StatementMatrixRow } from '$lib/db';

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
