import assert from 'node:assert/strict';
import { createServer as createHttpServer } from 'node:http';
import { readFileSync } from 'node:fs';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createServer } from 'vite';
import { DuckDBInstance } from '@duckdb/node-api';

const RUN = '00000000000000000000000000000111';
const dir = await mkdtemp(join(tmpdir(), 'indra-run-zero-scorer-'));
const dbFile = join(dir, 'zero-scorer.duckdb');
process.env.VIEWER_DUCKDB_PATH = dbFile;
process.env.INDRA_VIEWER_SUPPRESS_WRITER_GUARD_LOG = '1';

const runPageServer = readFileSync(new URL('../src/routes/runs/[run_id]/+page.server.ts', import.meta.url), 'utf-8');
const runPage = readFileSync(new URL('../src/routes/runs/[run_id]/+page.svelte', import.meta.url), 'utf-8');
assert.match(runPageServer, /n_scorer_steps/);
assert.match(runPage, /run output integrity warning/);
assert.match(runPage, /no persisted <code>scorer_step<\/code> rows/);

const instance = await DuckDBInstance.create(dbFile);
const con = await instance.connect();
try {
	await con.run(`CREATE TABLE score_run (
		run_id VARCHAR,
		scorer_version VARCHAR,
		architecture VARCHAR,
		paired_run_group_id VARCHAR,
		indra_version VARCHAR,
		model_id_default VARCHAR,
		started_at TIMESTAMP,
		finished_at TIMESTAMP,
		status VARCHAR,
		terminated_by VARCHAR,
		termination_reason VARCHAR,
		n_stmts BIGINT,
		cost_estimate_usd DOUBLE,
		cost_actual_usd DOUBLE
	)`);
	await con.run(`CREATE TABLE scorer_step (
		step_hash VARCHAR,
		run_id VARCHAR,
		stmt_hash VARCHAR,
		evidence_hash VARCHAR,
		architecture VARCHAR,
		model_id VARCHAR,
		step_kind VARCHAR,
		output_json JSON,
		error VARCHAR,
		latency_ms BIGINT,
		prompt_tokens BIGINT,
		out_tokens BIGINT,
		started_at TIMESTAMP
	)`);
	await con.run(`CREATE TABLE metric (
		run_id VARCHAR,
		truth_set_id VARCHAR,
		metric_name VARCHAR,
		value DOUBLE,
		slice_json JSON
	)`);
	await con.run(`INSERT INTO score_run VALUES (
		'${RUN}',
		'zero-scorer-test',
		'decomposed',
		NULL,
		'test-indra',
		'smoke-local',
		TIMESTAMP '2026-05-26 00:00:00',
		TIMESTAMP '2026-05-26 00:00:03',
		'succeeded',
		NULL,
		NULL,
		2,
		0.01,
		0.00
	)`);
} finally {
	con.disconnectSync?.();
	instance.closeSync();
}

const server = await createServer({
	configFile: './vite.config.ts',
	server: { middlewareMode: true, hmr: { port: 30_000 + (process.pid % 10_000) } },
	logLevel: 'error',
	optimizeDeps: { disabled: true },
	environments: {
		client: {
			optimizeDeps: { disabled: true }
		}
	}
});

const httpServer = createHttpServer(server.middlewares);
await new Promise((resolve, reject) => {
	httpServer.once('error', reject);
	httpServer.listen(0, '127.0.0.1', () => {
		httpServer.off('error', reject);
		resolve();
	});
});
const address = httpServer.address();
const baseUrl = `http://127.0.0.1:${address.port}`;

try {
	const response = await fetch(`${baseUrl}/runs/${RUN}`, { headers: { connection: 'close' } });
	const html = await response.text();
	assert.equal(response.status, 200);
	assert.match(html, /run output integrity/);
	assert.match(html, /marked\s+<strong[^>]*>succeeded<\/strong>\s+with\s+2\s+target statements/);
	assert.match(html, /no persisted <code[^>]*>scorer_step<\/code> rows/);
	assert.match(html, /Trace, repair, and comparison panels are not evidence of completed scoring/);
	console.log('run zero scorer integrity tests passed');
} finally {
	httpServer.closeIdleConnections?.();
	httpServer.closeAllConnections?.();
	await new Promise((resolve) => httpServer.close(resolve));
	await server.close();
	await rm(dir, { recursive: true, force: true });
}
