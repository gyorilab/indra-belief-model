import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createServer } from 'vite';
import { DuckDBInstance } from '@duckdb/node-api';

const FULL_RUN = 'tf000000000000000000000000000101';
const PARTIAL_RUN = 'tf000000000000000000000000000102';
const dir = await mkdtemp(join(tmpdir(), 'indra-monolithic-selected-examples-'));
const dbFile = join(dir, 'monolithic-selected-examples.duckdb');
process.env.VIEWER_DUCKDB_PATH = dbFile;
process.env.INDRA_VIEWER_SUPPRESS_WRITER_GUARD_LOG = '1';

const statementPage = readFileSync(new URL('../src/routes/statements/[stmt_hash]/+page.svelte', import.meta.url), 'utf-8');
const dbSource = readFileSync(new URL('../src/lib/db.ts', import.meta.url), 'utf-8');

assert.match(statementPage, /selectedExamplesForOutput/);
assert.match(statementPage, /selected monolithic contrastive examples/);
assert.match(statementPage, /selected example IDs captured/);
assert.match(statementPage, /selected example IDs are captured for/);
assert.match(dbSource, /selected example IDs are expected on monolithic LLM-tier aggregate rows/);

const instance = await DuckDBInstance.create(dbFile);
const con = await instance.connect();
try {
	await con.run(`CREATE TABLE score_run (
		run_id VARCHAR,
		architecture VARCHAR,
		paired_run_group_id VARCHAR,
		status VARCHAR
	)`);
	await con.run(`CREATE TABLE scorer_step (
		step_hash VARCHAR,
		run_id VARCHAR,
		stmt_hash VARCHAR,
		evidence_hash VARCHAR,
		architecture VARCHAR,
		step_kind VARCHAR,
		output_json JSON,
		error VARCHAR,
		started_at TIMESTAMP,
		latency_ms BIGINT,
		prompt_tokens BIGINT,
		out_tokens BIGINT
	)`);
	await con.run(`INSERT INTO score_run VALUES
		('${FULL_RUN}', 'monolithic', NULL, 'succeeded'),
		('${PARTIAL_RUN}', 'monolithic', NULL, 'succeeded')`);
	await con.run(`INSERT INTO scorer_step VALUES
		('m_full', '${FULL_RUN}', 'stmt_mono_full', 'ev_mono_full', 'monolithic', 'aggregate',
		 '{"score":0.8,"verdict":"correct","confidence":"high","tier":"llm_comprehension","grounding_status":"all_match","call_log":[{"kind":"monolithic"}],"raw_text":"trace","selected_example_ids":["abc123def456"],"selected_examples":[{"id":"abc123def456","claim":"A [Activation] B","verdict":"correct","confidence":"high"}]}'::JSON,
		 NULL, TIMESTAMP '2026-05-26 00:00:01', 10, 20, 5),
		('m_partial', '${PARTIAL_RUN}', 'stmt_mono_partial', 'ev_mono_partial', 'monolithic', 'aggregate',
		 '{"score":0.7,"verdict":"correct","confidence":"medium","tier":"llm_comprehension","grounding_status":"all_match","call_log":[{"kind":"monolithic"}],"raw_text":"trace"}'::JSON,
		 NULL, TIMESTAMP '2026-05-26 00:00:02', 10, 20, 5)`);
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

try {
	const db = await server.ssrLoadModule('/src/lib/db.ts');
	const full = await db.getTraceFidelity(FULL_RUN);
	assert.ok(full);
	assert.equal(full.architecture, 'monolithic');
	assert.equal(full.counts.full, 1);
	assert.equal(full.counts.partial, 0);
	assert.deepEqual(full.rows[0].missing_native_steps, []);

	const partial = await db.getTraceFidelity(PARTIAL_RUN);
	assert.ok(partial);
	assert.equal(partial.architecture, 'monolithic');
	assert.equal(partial.counts.full, 0);
	assert.equal(partial.counts.partial, 1);
	assert.ok(partial.rows[0].missing_native_steps.includes('selected_examples'));
	assert.match(partial.limitations.join(' '), /selected example IDs are expected/);

	console.log('monolithic selected example trace tests passed');
} finally {
	await server.close();
	await rm(dir, { recursive: true, force: true });
}
