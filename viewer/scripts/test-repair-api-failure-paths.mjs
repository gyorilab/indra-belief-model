import assert from 'node:assert/strict';
import { mkdtempSync, mkdirSync, rmSync, unlinkSync, existsSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createServer } from 'vite';

// D1 of deferred hypergraph: API-level tests for the typed repair-API
// failure codes that prior cycles' brutalist reviews tracked
// (paired_workflow_active, corpus_db_missing, repair_cohort_failed).

const RUN_ID = 'a'.repeat(32);
const tmp = mkdtempSync(join(tmpdir(), 'idb-d1-'));
const dataDir = join(tmp, 'data');
mkdirSync(dataDir, { recursive: true });
const dbFile = join(dataDir, 'corpus.duckdb');
// Pin the path BEFORE loading any module so db.ts's cached _resolvedPath
// stays constant across the tests below — only the on-disk presence
// varies.
process.env.VIEWER_DUCKDB_PATH = dbFile;

const server = await createServer({
	configFile: './vite.config.ts',
	server: { middlewareMode: true, hmr: { port: 30_000 + (process.pid % 10_000) } },
	logLevel: 'error'
});

try {
	const repairBacklog = await server.ssrLoadModule('/src/lib/server/repairBacklog.ts');
	const pairedState = await server.ssrLoadModule('/src/lib/server/pairedState.ts');

	// ---- Test 1: corpus_db_missing (file is absent) ----
	if (existsSync(dbFile)) unlinkSync(dbFile);
	try {
		await repairBacklog.estimateRepairCohort({
			run_id: RUN_ID,
			filters: {},
			source_route: '/runs/test/cohort',
			expected_run_status: 'succeeded'
		});
		throw new Error('estimateRepairCohort should have thrown corpus_db_missing');
	} catch (err) {
		assert.equal(err.code, 'corpus_db_missing',
			`expected corpus_db_missing, got code=${err?.code} msg=${err?.message}`);
		assert.equal(err.status, 404);
	}
	try {
		await repairBacklog.materializeRepairCohort({
			run_id: RUN_ID,
			filters: {},
			source_route: '/runs/test/cohort',
			expected_run_status: 'succeeded'
		});
		throw new Error('materializeRepairCohort should have thrown corpus_db_missing');
	} catch (err) {
		assert.equal(err.code, 'corpus_db_missing');
		assert.equal(err.status, 404);
	}

	// ---- Test 2: paired_workflow_active ----
	// Create the DB file (so corpus_db_missing no longer fires) and a
	// paired_workflow_state.json sidecar so activePairedWorkflowStates()
	// returns a busy state.
	const { DuckDBInstance } = await import('@duckdb/node-api');
	{
		const inst = await DuckDBInstance.create(dbFile);
		const con = await inst.connect();
		await con.run('CREATE TABLE score_run (run_id VARCHAR)');
		con.disconnectSync?.();
		inst.closeSync();
	}
	pairedState.createPairedWorkflowState({
		pair_id: 'pair_d1_test',
		source_dump_id: 'd1_smoke',
		dataset_path: join(dataDir, 'fake_corpus.json'),
		model: 'mock',
		scorer_version: 'd1-test',
		total_cost_threshold_usd: 1.0,
		caps: { monolithic: 0.5, decomposed: 0.5 }
	});

	try {
		await repairBacklog.estimateRepairCohort({
			run_id: RUN_ID,
			filters: {},
			source_route: '/runs/test/cohort',
			expected_run_status: 'succeeded'
		});
		throw new Error('estimateRepairCohort should have thrown paired_workflow_active');
	} catch (err) {
		assert.equal(err.code, 'paired_workflow_active',
			`expected paired_workflow_active, got code=${err?.code}`);
		assert.equal(err.status, 409);
	}

	// ---- Test 3: repair_cohort_failed fallback path exists in route ----
	const fs = await import('node:fs');
	const routeSrc = fs.readFileSync(
		new URL('../src/routes/api/repairs/cohort/+server.ts', import.meta.url),
		'utf-8'
	);
	assert.match(routeSrc, /code: 'repair_cohort_failed'/,
		'route +server.ts must surface repair_cohort_failed as the 500 fallback');
	assert.match(routeSrc, /status: 500/,
		'fallback must be a 500');

	console.log('repair API failure-path tests passed');
} finally {
	await server.close();
	try {
		rmSync(tmp, { recursive: true, force: true });
	} catch {
		// best-effort
	}
}
