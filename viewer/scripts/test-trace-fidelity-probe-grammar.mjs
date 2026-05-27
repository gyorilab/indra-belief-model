import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createServer } from 'vite';
import { DuckDBInstance } from '@duckdb/node-api';

const RUN = 'tf000000000000000000000000000001';
const dir = await mkdtemp(join(tmpdir(), 'indra-trace-fidelity-probes-'));
const dbFile = join(dir, 'trace-fidelity.duckdb');
process.env.VIEWER_DUCKDB_PATH = dbFile;
process.env.INDRA_VIEWER_SUPPRESS_WRITER_GUARD_LOG = '1';

const sqlSource = readFileSync(new URL('../src/lib/traceStateSql.ts', import.meta.url), 'utf-8');
const dbSource = readFileSync(new URL('../src/lib/db.ts', import.meta.url), 'utf-8');

assert.match(sqlSource, /n_subject_role_probe/);
assert.match(sqlSource, /n_object_role_probe/);
assert.match(sqlSource, /n_relation_axis_probe/);
assert.match(sqlSource, /n_scope_probe/);
assert.match(
	sqlSource,
	/n_subject_role_probe > 0 AND n_object_role_probe > 0[\s\S]*n_relation_axis_probe > 0 AND n_scope_probe > 0/
);
assert.match(dbSource, /subject-role probe/);
assert.match(dbSource, /object-role probe/);
assert.match(dbSource, /relation-axis probe/);
assert.match(dbSource, /scope probe/);
assert.match(dbSource, /route slots alone do not prove full probe-event persistence/);
assert.match(dbSource, /full fidelity requires the four named probe event rows/);

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
		('${RUN}', 'decomposed', NULL, 'succeeded')`);

	let i = 0;
	async function insertStep(evidence, stepKind) {
		i += 1;
		const stmt = evidence === 'ev_full_probe_trace' ? 'stmt_full_probe_trace' : 'stmt_route_slot_only';
		const payload = stepKind === 'aggregate'
			? '{"score":0.7,"verdict":"correct","confidence":"high"}'
			: '{"ok":true}';
		await con.run(`INSERT INTO scorer_step VALUES
			('step_${evidence}_${stepKind}', '${RUN}', '${stmt}', '${evidence}', 'decomposed', '${stepKind}', '${payload}'::JSON, NULL, TIMESTAMP '2026-05-26 00:00:${String(i).padStart(2, '0')}', 1, NULL, NULL)`);
	}

	for (const stepKind of [
		'parse_claim',
		'build_context',
		'substrate_route',
		'subject_role_probe',
		'object_role_probe',
		'relation_axis_probe',
		'scope_probe',
		'grounding',
		'adjudicate',
		'aggregate'
	]) {
		await insertStep('ev_full_probe_trace', stepKind);
	}
	for (const stepKind of [
		'parse_claim',
		'build_context',
		'substrate_route',
		'grounding',
		'adjudicate',
		'aggregate'
	]) {
		await insertStep('ev_route_slot_only', stepKind);
	}
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
	const fidelity = await db.getTraceFidelity(RUN);
	assert.ok(fidelity);
	assert.equal(fidelity.architecture, 'decomposed');
	assert.equal(fidelity.counts.full, 1);
	assert.equal(fidelity.counts.partial, 1);
	assert.equal(fidelity.counts.aggregate_only, 0);
	assert.deepEqual(
		fidelity.native_grammar.filter((part) => part.includes('probe')),
		['subject-role probe', 'object-role probe', 'relation-axis probe', 'scope probe']
	);
	assert.match(fidelity.limitations.join(' '), /substrate-route slots alone are partial trace evidence/);

	const fullRow = fidelity.rows.find((row) => row.evidence_hash === 'ev_full_probe_trace');
	assert.ok(fullRow);
	assert.equal(fullRow.state, 'full');
	assert.deepEqual(fullRow.missing_native_steps, []);
	assert.match(fullRow.note, /all named probe events/);
	assert.ok(fullRow.captured_steps.includes('subject_role_probe'));
	assert.ok(fullRow.captured_steps.includes('scope_probe'));

	const partialRow = fidelity.rows.find((row) => row.evidence_hash === 'ev_route_slot_only');
	assert.ok(partialRow);
	assert.equal(partialRow.state, 'partial');
	assert.deepEqual(partialRow.missing_native_steps, [
		'subject_role_probe',
		'object_role_probe',
		'relation_axis_probe',
		'scope_probe'
	]);
	assert.match(partialRow.note, /route slots alone do not prove full probe-event persistence/);

	console.log('trace fidelity probe grammar tests passed');
} finally {
	await server.close();
	await rm(dir, { recursive: true, force: true });
}
