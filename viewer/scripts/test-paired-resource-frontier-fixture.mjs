import assert from 'node:assert/strict';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createServer } from 'vite';
import { DuckDBInstance } from '@duckdb/node-api';

const PAIR = 'pair_resource_frontier';
const MONO = 'mono_frontier_run';
const DECOMP = 'decomp_frontier_run';

const dir = await mkdtemp(join(tmpdir(), 'indra-paired-frontier-'));
const dbFile = join(dir, 'frontier.duckdb');
process.env.VIEWER_DUCKDB_PATH = dbFile;
process.env.INDRA_VIEWER_SUPPRESS_WRITER_GUARD_LOG = '1';

const instance = await DuckDBInstance.create(dbFile);
const con = await instance.connect();
try {
	await con.run(`CREATE TABLE score_run (
		run_id VARCHAR,
		scorer_version VARCHAR,
		architecture VARCHAR,
		paired_run_group_id VARCHAR,
		model_id_default VARCHAR,
		started_at TIMESTAMP,
		finished_at TIMESTAMP,
		status VARCHAR,
		n_stmts BIGINT,
		cost_estimate_usd DOUBLE,
		cost_actual_usd DOUBLE
	)`);
	await con.run(`CREATE TABLE statement (
		stmt_hash VARCHAR,
		indra_type VARCHAR,
		indra_belief DOUBLE
	)`);
	await con.run(`CREATE TABLE evidence (
		evidence_hash VARCHAR,
		stmt_hash VARCHAR,
		source_api VARCHAR,
		pmid VARCHAR,
		text VARCHAR
	)`);
	await con.run(`CREATE TABLE agent (
		stmt_hash VARCHAR,
		agent_hash VARCHAR,
		role VARCHAR,
		name VARCHAR,
		role_index BIGINT
	)`);
	await con.run(`CREATE TABLE scorer_step (
		step_hash VARCHAR,
		run_id VARCHAR,
		scorer_version VARCHAR,
		architecture VARCHAR,
		stmt_hash VARCHAR,
		evidence_hash VARCHAR,
		step_kind VARCHAR,
		is_substrate_answered BOOLEAN,
		output_json JSON,
		latency_ms BIGINT,
		prompt_tokens BIGINT,
		out_tokens BIGINT,
		error VARCHAR,
		started_at TIMESTAMP
	)`);
	await con.run(`INSERT INTO score_run VALUES
		('${MONO}', 'frontier-v1', 'monolithic', '${PAIR}', 'demo-model', TIMESTAMP '2026-05-26 00:00:00', TIMESTAMP '2026-05-26 00:01:00', 'succeeded', 3, 0.07, 0.06),
		('${DECOMP}', 'frontier-v1', 'decomposed', '${PAIR}', 'demo-model', TIMESTAMP '2026-05-26 00:02:00', TIMESTAMP '2026-05-26 00:02:40', 'succeeded', 2, 0.05, 0.04)`);
	await con.run(`INSERT INTO statement VALUES
		('stmt_a', 'Activation', 0.80),
		('stmt_b', 'Inhibition', 0.20),
		('stmt_c', 'Complex', 0.50)`);
	await con.run(`INSERT INTO evidence VALUES
		('ev_a', 'stmt_a', 'reach', '1', 'shared evidence a'),
		('ev_b', 'stmt_b', 'reach', '2', 'shared evidence b'),
		('ev_c', 'stmt_c', 'reach', '3', 'monolithic only evidence')`);
	await con.run(`INSERT INTO agent VALUES
		('stmt_a', 'agent_a', 'subj', 'A', 0),
		('stmt_a', 'agent_b', 'obj', 'B', 1),
		('stmt_b', 'agent_c', 'subj', 'C', 0),
		('stmt_b', 'agent_d', 'obj', 'D', 1)`);
	await con.run(`INSERT INTO scorer_step VALUES
		('m_a', '${MONO}', 'frontier-v1', 'monolithic', 'stmt_a', 'ev_a', 'aggregate', NULL, '{"score":0.80,"verdict":"correct","confidence":"high","tier":"direct"}'::JSON, 100, 10, 5, NULL, TIMESTAMP '2026-05-26 00:00:10'),
		('m_b', '${MONO}', 'frontier-v1', 'monolithic', 'stmt_b', 'ev_b', 'aggregate', NULL, '{"score":0.30,"verdict":"incorrect","confidence":"medium","tier":"direct"}'::JSON, 200, 20, 10, NULL, TIMESTAMP '2026-05-26 00:00:20'),
		('m_c', '${MONO}', 'frontier-v1', 'monolithic', 'stmt_c', 'ev_c', 'aggregate', NULL, '{"score":0.50,"verdict":"abstain","confidence":"low","tier":"fallback"}'::JSON, 400, 30, 20, NULL, TIMESTAMP '2026-05-26 00:00:30'),
		('d_a', '${DECOMP}', 'frontier-v1', 'decomposed', 'stmt_a', 'ev_a', 'aggregate', NULL, '{"score":0.75,"verdict":"correct","confidence":"high"}'::JSON, 300, 40, 10, NULL, TIMESTAMP '2026-05-26 00:02:10'),
		('d_b', '${DECOMP}', 'frontier-v1', 'decomposed', 'stmt_b', 'ev_b', 'aggregate', NULL, '{"score":0.60,"verdict":"correct","confidence":"medium"}'::JSON, 400, 35, 15, NULL, TIMESTAMP '2026-05-26 00:02:20')`);
} finally {
	con.disconnectSync?.();
	instance.closeSync();
}

const server = await createServer({
	configFile: './vite.config.ts',
	server: { middlewareMode: true, hmr: { port: 30_000 + (process.pid % 10_000) } },
	logLevel: 'error'
});

try {
	const db = await server.ssrLoadModule('/src/lib/db.ts');
	const workbench = await db.getPairedWorkbench(PAIR);
	assert.ok(workbench);
	assert.ok(workbench.resource_frontier);
	assert.equal(workbench.resource_frontier.spend_scope, 'whole-run spend; denominator is each side aggregate evidence count, not clean overlap');
	assert.equal(workbench.resource_frontier.latency_scope, 'clean shared aggregate verdict evidence with per-row telemetry counts shown');
	assert.equal(workbench.resource_frontier.quality_scope, 'MAE over truth-anchored clean overlap');

	const mono = workbench.resource_frontier.monolithic;
	const decomp = workbench.resource_frontier.decomposed;
	assert.equal(mono.n_evidences, 3);
	assert.equal(decomp.n_evidences, 2);
	assert.equal(mono.clean_overlap_n, 2);
	assert.equal(decomp.clean_overlap_n, 2);
	assert.equal(mono.clean_overlap_latency_mean_ms, 150);
	assert.equal(decomp.clean_overlap_latency_mean_ms, 350);
	assert.equal(mono.clean_overlap_latency_observed_n, 2);
	assert.equal(decomp.clean_overlap_latency_observed_n, 2);
	assert.equal(mono.clean_overlap_tokens_total, 45);
	assert.equal(decomp.clean_overlap_tokens_total, 100);
	assert.equal(mono.clean_overlap_tokens_observed_n, 2);
	assert.equal(decomp.clean_overlap_tokens_observed_n, 2);
	assert.equal(mono.clean_overlap_tokens_per_observed_evidence, 22.5);
	assert.equal(decomp.clean_overlap_tokens_per_observed_evidence, 50);
	assert.equal(mono.duration_s, 60);
	assert.equal(decomp.duration_s, 40);
	assert.equal(mono.wall_seconds_per_evidence, 20);
	assert.equal(decomp.wall_seconds_per_evidence, 20);
	assert.equal(mono.run_cost_basis, 'actual');
	assert.equal(decomp.run_cost_basis, 'actual');
	assert.equal(mono.cost_per_evidence_usd, 0.02);
	assert.equal(decomp.cost_per_evidence_usd, 0.02);
	assert.equal(mono.truth_overlap_n, 2);
	assert.equal(decomp.truth_overlap_n, 2);
	assert.ok(Math.abs(mono.mae - 0.05) < 0.0000001);
	assert.ok(Math.abs(decomp.mae - 0.225) < 0.0000001);

	const resourceLedger = workbench.denominator_ledger.find((row) => row.key === 'resource_counter_metric');
	assert.ok(resourceLedger);
	assert.equal(resourceLedger.denominator_n, 2);
	assert.equal(resourceLedger.unit, 'clean shared aggregate verdict evidence');

	console.log('paired resource frontier fixture tests passed');
} finally {
	await server.close();
	await rm(dir, { recursive: true, force: true });
}
