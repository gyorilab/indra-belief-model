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

import { EMPTY_OVERVIEW, assertCoreOverviewSchema, closeInstance, connect, dbExists, dbPath, durationSeconds, normalizeDuckValue, readRows, readScalar, repairRerunLineageSqlOptions, rows, safeRate, scalar, sqlQuote, tableExists, timestampMs } from '$lib/db';
import type { CorpusOverview, LatestValidity } from '$lib/db';

import { getLatestValidity } from '$lib/server/cohorts/validity';
import { resolve } from 'node:path';
import { existsSync } from 'node:fs';
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
