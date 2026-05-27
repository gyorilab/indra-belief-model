import assert from 'node:assert/strict';
import { createServer as createHttpServer } from 'node:http';
import { readFileSync } from 'node:fs';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createServer } from 'vite';
import { DuckDBInstance } from '@duckdb/node-api';

const RUN = '00000000000000000000000000000112';
const dir = await mkdtemp(join(tmpdir(), 'indra-run-denominators-'));
const dbFile = join(dir, 'denominators.duckdb');
process.env.VIEWER_DUCKDB_PATH = dbFile;
process.env.INDRA_VIEWER_SUPPRESS_WRITER_GUARD_LOG = '1';

const runPage = readFileSync(new URL('../src/routes/runs/[run_id]/+page.svelte', import.meta.url), 'utf-8');
const dbSource = readFileSync(new URL('../src/lib/db.ts', import.meta.url), 'utf-8');
const coverageSource = readFileSync(new URL('../src/lib/components/HeuristicCoverage.svelte', import.meta.url), 'utf-8');
const repairSource = readFileSync(new URL('../src/lib/server/repairBacklog.ts', import.meta.url), 'utf-8');
assert.match(runPage, /denominatorRows/);
assert.match(runPage, /denominator boundaries/);
assert.match(runPage, /probe coverage evidence/);
assert.match(runPage, /probe_coverage=present/);
assert.match(dbSource, /n_probe_evidences/);
assert.match(coverageSource, /probe coverage denominator:/);
assert.match(coverageSource, /probe_coverage=present/);
assert.match(repairSource, /suspected_step_kind[\s\S]*probe_coverage/);

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
		await con.run(`CREATE TABLE statement (
			stmt_hash VARCHAR,
			indra_type VARCHAR,
			indra_belief DOUBLE,
			supports_count BIGINT,
			supported_by_count BIGINT
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
			name VARCHAR,
			role VARCHAR,
			role_index BIGINT
		)`);
	await con.run(`INSERT INTO score_run VALUES (
		'${RUN}',
		'denominator-boundary-test',
		'decomposed',
		NULL,
		'test-indra',
		'smoke-local',
		TIMESTAMP '2026-05-26 00:00:00',
		TIMESTAMP '2026-05-26 00:00:03',
		'failed',
		'worker_error',
		'aggregate writer exited early',
		2,
		0.02,
		0.01
	)`);
		await con.run(`INSERT INTO scorer_step VALUES
		('s112_parse', '${RUN}', 'stmt_denominator', 'ev_denominator', 'decomposed', 'smoke-local', 'parse_claim', '{"claim":"A activates B"}'::JSON, NULL, 10, 0, 0, TIMESTAMP '2026-05-26 00:00:01'),
		('s112_context', '${RUN}', 'stmt_denominator', 'ev_denominator', 'decomposed', 'smoke-local', 'build_context', '{"context":"partial"}'::JSON, NULL, 10, 0, 0, TIMESTAMP '2026-05-26 00:00:02'),
		('s112_route', '${RUN}', 'stmt_denominator', 'ev_denominator', 'decomposed', 'smoke-local', 'substrate_route', '{"subject_role":{"source":"substrate","answer":"yes"},"object_role":{"source":"needs_llm","answer":null},"relation_axis":{"source":"needs_llm","answer":null},"scope":{"source":null,"answer":null}}'::JSON, NULL, 10, 0, 0, TIMESTAMP '2026-05-26 00:00:03'),
		('s112_object', '${RUN}', 'stmt_denominator', 'ev_denominator', 'decomposed', 'smoke-local', 'object_role_probe', '{"source":"llm","answer":"yes"}'::JSON, NULL, 10, 8, 3, TIMESTAMP '2026-05-26 00:00:04'),
			('s112_relation', '${RUN}', 'stmt_denominator', 'ev_denominator', 'decomposed', 'smoke-local', 'relation_axis_probe', '{"source":"abstain","answer":"abstain"}'::JSON, NULL, 10, 8, 3, TIMESTAMP '2026-05-26 00:00:05')`);
		await con.run(`INSERT INTO statement VALUES (
			'stmt_denominator',
			'Activation',
			0.40,
			0,
			0
		)`);
		await con.run(`INSERT INTO evidence VALUES (
			'ev_denominator',
			'stmt_denominator',
			'test_source',
			'1',
			'probe coverage denominator fixture'
		)`);
		await con.run(`INSERT INTO agent VALUES
			('stmt_denominator', 'A', 'subj', 0),
			('stmt_denominator', 'B', 'obj', 1)`);
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
	assert.match(html, /denominator boundaries/);
	assert.match(html, /These panels intentionally do not share one denominator/);
	assert.match(html, /target statements[\s\S]*?<td[^>]*class="num[^"]*"[^>]*>2<\/td>/);
	assert.match(html, /persisted scorer steps[\s\S]*?<td[^>]*class="num[^"]*"[^>]*>5<\/td>/);
	assert.match(html, /trace evidence states[\s\S]*?<td[^>]*class="num[^"]*"[^>]*>1<\/td>/);
	assert.match(html, /aggregate verdict evidence[\s\S]*?<td[^>]*class="num[^"]*"[^>]*>0<\/td>/);
	assert.match(html, /probe coverage evidence[\s\S]*?<td[^>]*class="num[^"]*"[^>]*>1<\/td>/);
	assert.match(
		html,
		/href="\/runs\/00000000000000000000000000000112\/cohort\?probe_coverage=present"/
	);
	assert.match(html, /probe coverage denominator:/);
	assert.match(html, /1 evidence row with persisted substrate\/probe slots/);
	assert.match(html, /aggregate verdict denominator:\s*0 evidence rows/);
	assert.match(html, /On average the system invoked 3\.00 of 4 probes per probe-covered evidence/);
	assert.doesNotMatch(html, /no probe records persisted for this run yet/);
	const cohortResponse = await fetch(`${baseUrl}/runs/${RUN}/cohort?probe_coverage=present`, {
		headers: { connection: 'close' }
	});
	const cohortHtml = await cohortResponse.text();
	assert.equal(cohortResponse.status, 200);
	assert.match(cohortHtml, /1 trace evidence row in run/);
	assert.match(cohortHtml, /probe_coverage[\s\S]{0,300}present/);
	assert.match(cohortHtml, /probe coverage evidence/);
	assert.match(cohortHtml, /exact denominator used by the four-probe coverage panel/);
	assert.match(
		cohortHtml,
		/Repair candidates attach to the representative persisted probe step and preserve missing-slot facts/
	);
	assert.match(
		cohortHtml,
		/repair writes one candidate per probe-covered evidence row, with observed probe counts and missing probe slots preserved/
	);
	assert.match(cohortHtml, /terminated_inflight/);
	assert.match(cohortHtml, /relation_axis_probe/);
	console.log('run denominator boundary tests passed');
} finally {
	httpServer.closeIdleConnections?.();
	httpServer.closeAllConnections?.();
	await new Promise((resolve) => httpServer.close(resolve));
	await server.close();
	await rm(dir, { recursive: true, force: true });
}
