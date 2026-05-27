import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { DuckDBInstance } from '@duckdb/node-api';
import assert from 'node:assert/strict';
import { createServer } from 'vite';

// B2+B3 of deferred hypergraph: verify the request-local
// materialization helper:
//  - creates a TEMPORARY TABLE on the supplied connection
//  - is visible to subsequent reads on the SAME connection
//  - is NOT visible to a separate connection (DuckDB temp tables are
//    connection-scoped)
//  - drops on dropRequestLocalTable
//  - rejects unsafe table names

const tmp = mkdtempSync(join(tmpdir(), 'idb-rlm-'));
const dbFile = join(tmp, 'corpus.duckdb');
{
	const seed = await DuckDBInstance.create(dbFile);
	const con = await seed.connect();
	await con.run('CREATE TABLE scorer_step (run_id VARCHAR, stmt_hash VARCHAR, evidence_hash VARCHAR, step_kind VARCHAR)');
	await con.run(`INSERT INTO scorer_step VALUES
		('r1', 's1', 'e1', 'aggregate'),
		('r1', 's2', 'e2', 'aggregate'),
		('r1', 's3', 'e3', 'aggregate')`);
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
	const mod = await server.ssrLoadModule('/src/lib/cohorts/requestLocalMaterialization.ts');
	// Use a fresh DuckDB instance directly — connect()/closeInstance()
	// from $lib/db lives in module state and would conflict across
	// repeated tests. The helper is connection-agnostic.
	const inst = await DuckDBInstance.create(dbFile);
	const conA = await inst.connect();

	await mod.materializeRequestLocalTable(conA, 'cohort_test',
		"SELECT * FROM scorer_step WHERE step_kind='aggregate'");
	const r1 = await conA.runAndReadAll('SELECT COUNT(*) AS n FROM cohort_test');
	assert.equal(Number(r1.getRowObjects()[0].n), 3);

	// Same connection — table still visible after multiple reads.
	const r2 = await conA.runAndReadAll(
		"SELECT stmt_hash FROM cohort_test WHERE evidence_hash='e2'"
	);
	assert.equal(r2.getRowObjects()[0].stmt_hash, 's2');

	// A DIFFERENT connection on the same instance must NOT see the temp table.
	const conB = await inst.connect();
	let visibleOnConB = true;
	try {
		await conB.runAndReadAll('SELECT COUNT(*) FROM cohort_test');
	} catch {
		visibleOnConB = false;
	}
	assert.equal(visibleOnConB, false,
		'temp tables must be connection-scoped, not visible to sibling connections');
	conB.disconnectSync?.();

	// dropRequestLocalTable removes it
	await mod.dropRequestLocalTable(conA, 'cohort_test');
	let droppedReadFailed = true;
	try {
		await conA.runAndReadAll('SELECT COUNT(*) FROM cohort_test');
		droppedReadFailed = false;
	} catch {
		// expected — table is gone
	}
	assert.equal(droppedReadFailed, true);

	// Reject unsafe table names.
	for (const bad of ['1evil', "evil; DROP TABLE", 'evil"name', '']) {
		await assert.rejects(
			() => mod.materializeRequestLocalTable(conA, bad, 'SELECT 1'),
			/invalid table name/
		);
	}

	conA.disconnectSync?.();
	inst.closeSync();

	console.log('request-local materialization tests passed');
} finally {
	await server.close();
	try {
		rmSync(tmp, { recursive: true, force: true });
	} catch {
		// best-effort
	}
}
