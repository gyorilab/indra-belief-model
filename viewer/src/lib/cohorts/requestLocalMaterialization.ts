// B2+B3 of deferred hypergraph: request-local materialized cohort
// tables for SSR routes. SSR handlers that read the same filtered
// projection multiple times (e.g., paired workbench's overlap +
// comparable + arch-conditioned panels) can populate a per-request
// TEMPORARY TABLE once and read from it for each panel — avoiding
// the rebuild cost on every cell.
//
// DuckDB TEMPORARY TABLEs are scoped to the owning connection; the
// SSR route must use ONE shared connection across the reads that
// access the materialized view. The lifetime ends when the
// connection is released, so a `connect()` per request +
// `disconnectSync()` at the end is the correct shape.
//
// Usage (example in $lib/server/cohorts/workbench.ts):
//
//   const con = await connect();
//   try {
//     await materializeRequestLocalTable(con, 'mw_overlap', `
//       SELECT stmt_hash, evidence_hash
//         FROM scorer_step
//        WHERE run_id IN ('${monoRun}', '${decompRun}')
//          AND step_kind = 'aggregate'
//     `);
//     const overlapRows = await rows(con, 'SELECT COUNT(*) FROM mw_overlap');
//     const verdictRows = await rows(con,
//       `SELECT ... FROM mw_overlap JOIN scorer_step USING (run_id, ...)`);
//     // additional reads share the materialized projection
//   } finally {
//     con.disconnectSync?.();
//   }
//
// Naming: prefix temp tables with the route shorthand (e.g., `mw_` for
// monolithic workbench, `cohort_` for runs/[run_id]/cohort) so debug
// inspection makes provenance obvious.
import type { DuckDBConnection } from '@duckdb/node-api';

export async function materializeRequestLocalTable(
	con: DuckDBConnection,
	name: string,
	sql: string
): Promise<void> {
	// DuckDB validates identifiers; the caller is responsible for ensuring
	// `name` is a stable hard-coded string, not user input.
	if (!/^[a-zA-Z_][a-zA-Z0-9_]*$/.test(name)) {
		throw new Error(`materializeRequestLocalTable: invalid table name ${name}`);
	}
	await con.run(`CREATE OR REPLACE TEMPORARY TABLE ${name} AS ${sql}`);
}

export async function dropRequestLocalTable(
	con: DuckDBConnection,
	name: string
): Promise<void> {
	if (!/^[a-zA-Z_][a-zA-Z0-9_]*$/.test(name)) {
		throw new Error(`dropRequestLocalTable: invalid table name ${name}`);
	}
	await con.run(`DROP TABLE IF EXISTS ${name}`);
}
