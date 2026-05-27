import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { DuckDBInstance } from '@duckdb/node-api';
import assert from 'node:assert/strict';
import { createServer } from 'vite';

// Phase 2 gate: concurrent reader+writer test for the DuckDB connection
// manager in `viewer/src/lib/db.ts`. Verifies:
//   1. concurrent `connect()` calls share one underlying instance (no leak)
//   2. `closeInstance()` drains in-flight reader connections before destroying
//   3. a new `connect()` after `closeInstance()` lazy-reopens
// Cycle 51's chronic db.ts singleton race surfaced as `Could not set lock`
// when a writer spawned while readers were live; this test pins the new
// serialized/refcounted behavior.

const tmp = mkdtempSync(join(tmpdir(), 'idb-connmgr-'));
const dbFile = join(tmp, 'corpus.duckdb');
{
	const seed = await DuckDBInstance.create(dbFile);
	const con = await seed.connect();
	await con.run('CREATE TABLE schema_meta (key VARCHAR, value VARCHAR)');
	await con.run('CREATE TABLE statement (stmt_hash VARCHAR, indra_type VARCHAR, indra_belief DOUBLE, source_dump_id VARCHAR)');
	await con.run('CREATE TABLE evidence (evidence_hash VARCHAR, source_api VARCHAR)');
	await con.run('CREATE TABLE statement_evidence (stmt_hash VARCHAR, evidence_hash VARCHAR, evidence_index INTEGER, source_dump_id VARCHAR)');
	await con.run('CREATE TABLE agent (stmt_hash VARCHAR, agent_hash VARCHAR, role VARCHAR, name VARCHAR)');
	await con.run('CREATE TABLE supports_edge (from_stmt_hash VARCHAR, to_stmt_hash VARCHAR, kind VARCHAR)');
	await con.run('CREATE TABLE truth_set (id VARCHAR, name VARCHAR)');
	await con.run('CREATE TABLE truth_label (label_id BIGINT, truth_set_id VARCHAR, target_kind VARCHAR, target_id VARCHAR, field VARCHAR)');
	await con.run("INSERT INTO statement VALUES ('s1', 'Activation', 0.7, 'demo')");
	await con.run("INSERT INTO evidence VALUES ('e1', 'reach')");
	await con.run("INSERT INTO statement_evidence VALUES ('s1', 'e1', 0, 'demo')");
	await con.run(
		"CREATE TABLE scorer_step (step_hash VARCHAR, run_id VARCHAR, stmt_hash VARCHAR, evidence_hash VARCHAR, step_kind VARCHAR, output_json JSON)"
	);
	await con.run(
		"CREATE TABLE score_run (run_id VARCHAR, scorer_version VARCHAR, architecture VARCHAR, paired_run_group_id VARCHAR, started_at TIMESTAMP, status VARCHAR, terminated_by VARCHAR, termination_reason VARCHAR, n_stmts BIGINT, cost_estimate_usd DOUBLE)"
	);
	await con.run(
		"CREATE TABLE metric (run_id VARCHAR, truth_set_id VARCHAR, metric_name VARCHAR, value DOUBLE, slice_json JSON)"
	);
	con.disconnectSync?.();
	seed.closeSync();
}

process.env.VIEWER_DUCKDB_PATH = dbFile;

const server = await createServer({
	configFile: './vite.config.ts',
	server: { middlewareMode: true, hmr: { port: 30_000 + (process.pid % 10_000) } },
	logLevel: 'error'
});

try {
	const db = await server.ssrLoadModule('/src/lib/db.ts');

	// 1. Concurrent connect() calls reuse the same in-flight instance creation.
	const connections = await Promise.all([
		db.connect(),
		db.connect(),
		db.connect(),
		db.connect(),
		db.connect()
	]);
	assert.equal(connections.length, 5);
	// All connections should be usable independently.
	for (const con of connections) {
		const reader = await con.runAndReadAll('SELECT COUNT(*) AS n FROM statement');
		assert.equal(Number(reader.getRowObjects()[0].n), 1);
	}

	// 2. closeInstance() must wait for in-flight reads before destroying.
	//    Issue several long-ish queries concurrently with a closeInstance() and
	//    verify the close completes after the reads, not before.
	const longReads = connections.map((con) =>
		con.runAndReadAll(
			"WITH RECURSIVE c(i) AS (SELECT 1 UNION ALL SELECT i+1 FROM c WHERE i<1000) SELECT COUNT(*) AS n FROM c"
		)
	);
	const closePromise = db.closeInstance();
	const readResults = await Promise.all(longReads);
	await closePromise;
	for (const res of readResults) {
		assert.equal(Number(res.getRowObjects()[0].n), 1000);
	}
	for (const con of connections) {
		con.disconnectSync?.();
	}

	// 3. A new connect() after close lazy-reopens.
	const reopened = await db.connect();
	const reader = await reopened.runAndReadAll('SELECT COUNT(*) AS n FROM statement');
	assert.equal(Number(reader.getRowObjects()[0].n), 1);
	reopened.disconnectSync?.();

	// 4. closeInstance() on a fresh handle still works.
	await db.closeInstance();

	console.log('db connection manager tests passed');
} finally {
	await server.close();
	try {
		rmSync(tmp, { recursive: true, force: true });
	} catch {
		// best-effort
	}
}
