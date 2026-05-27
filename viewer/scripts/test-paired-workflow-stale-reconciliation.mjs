import assert from 'node:assert/strict';
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createServer } from 'vite';
import { DuckDBInstance } from '@duckdb/node-api';

const dir = await mkdtemp(join(tmpdir(), 'indra-stale-pair-'));
const dbFile = join(dir, 'stale.duckdb');
process.env.VIEWER_DUCKDB_PATH = dbFile;

const instance = await DuckDBInstance.create(dbFile);
const con = await instance.connect();
await con.run('CREATE TABLE statement (stmt_hash VARCHAR)');
con.disconnectSync?.();
instance.closeSync();

const server = await createServer({
	configFile: './vite.config.ts',
	server: { middlewareMode: true, hmr: { port: 30_000 + (process.pid % 10_000) } },
	logLevel: 'error'
});

try {
	const pairedState = await server.ssrLoadModule('/src/lib/server/pairedState.ts');
	const stale = pairedState.createPairedWorkflowState({
		pair_id: 'cycle92_stale_pair',
		source_dump_id: 'cycle92',
		dataset_path: join(dir, 'input.json'),
		model: 'test-model',
		scorer_version: 'test-v1',
		total_cost_threshold_usd: 0.02,
		caps: { monolithic: 0.01, decomposed: 0.01 }
	});
	const staleUpdatedAt = new Date(Date.now() - 11 * 60 * 1000).toISOString();
	pairedState.updatePairedWorkflowState(stale.pair_id, (state) => ({
		...state,
		status: 'running',
		architectures: {
			monolithic: {
				...state.architectures.monolithic,
				status: 'running',
				pid: 999_999_999,
				run_id: 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa',
				n_evidences_done: 3,
				n_evidences_total: 10,
				started_at: staleUpdatedAt,
				updated_at: staleUpdatedAt
			},
			decomposed: {
				...state.architectures.decomposed,
				status: 'queued',
				updated_at: staleUpdatedAt
			}
		}
	}));
	const staleStatePath = join(dir, 'viewer_state', 'paired', `${stale.pair_id}.json`);
	const staleRaw = JSON.parse(await readFile(staleStatePath, 'utf-8'));
	staleRaw.updated_at = staleUpdatedAt;
	await writeFile(staleStatePath, JSON.stringify(staleRaw, null, 2));
	const activeBefore = pairedState.activePairedWorkflowStates();
	assert.equal(activeBefore.some((state) => state.pair_id === stale.pair_id), false);
	const reconciled = pairedState.readPairedWorkflowState(stale.pair_id);
	assert.equal(reconciled.status, 'failed');
	assert.match(reconciled.termination_reason, /stale/);
	assert.equal(reconciled.architectures.monolithic.status, 'failed');
	assert.equal(reconciled.architectures.monolithic.pid, null);
	assert.match(reconciled.architectures.monolithic.error, /without mutating any scorer output/);
	assert.equal(reconciled.architectures.decomposed.status, 'blocked');
	assert.match(reconciled.architectures.decomposed.error, /queued architecture never started/);

	const fresh = pairedState.createPairedWorkflowState({
		pair_id: 'cycle92_fresh_pair',
		source_dump_id: 'cycle92',
		dataset_path: join(dir, 'input.json'),
		model: 'test-model',
		scorer_version: 'test-v1',
		total_cost_threshold_usd: 0.02,
		caps: { monolithic: 0.01, decomposed: 0.01 }
	});
	assert.equal(pairedState.pairedWorkflowStaleReason(fresh), null);
	assert.equal(pairedState.reconcileStalePairedWorkflowState(fresh.pair_id).status, 'queued');

	console.log('paired workflow stale reconciliation tests passed');
} finally {
	await server.close();
	await rm(dir, { recursive: true, force: true });
}
