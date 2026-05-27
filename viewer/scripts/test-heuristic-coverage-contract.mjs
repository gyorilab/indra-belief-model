import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createServer } from 'vite';
import { DuckDBInstance } from '@duckdb/node-api';

const DECOMP = 'heuristic_decomp_missing_probe_rows';
const DECOMP_PARTIAL = 'heuristic_decomp_native_no_probe_rows';
const DECOMP_PROBE_PARTIAL = 'heuristic_decomp_probe_rows_no_aggregate';
const MONO = 'heuristic_mono_native';
const dir = await mkdtemp(join(tmpdir(), 'indra-heuristic-coverage-'));
const dbFile = join(dir, 'heuristic.duckdb');
process.env.VIEWER_DUCKDB_PATH = dbFile;
process.env.INDRA_VIEWER_SUPPRESS_WRITER_GUARD_LOG = '1';

const source = readFileSync(new URL('../src/lib/components/HeuristicCoverage.svelte', import.meta.url), 'utf-8');
const cohortSource = readFileSync(new URL('../src/routes/runs/[run_id]/cohort/+page.svelte', import.meta.url), 'utf-8');

assert.match(source, /const probeRecordsAbsent = \$derived/);
assert.match(source, /coverage\.n_probe_evidences > 0 && totalProbeSlots > 0/);
assert.match(source, /coverage\.n_evidences > 0 &&\s*!probeCoverageAvailable/s);
assert.match(source, /const verb = coverage\.n_evidences === 1 \? 'exists' : 'exist'/);
assert.match(source, /Probe coverage is unavailable, not zero/);
assert.match(source, /probe coverage denominator:/);
assert.match(source, /aggregate verdict denominator:/);
assert.match(source, /class="cov-missing-probes" role="note"/);
assert.match(source, /probe records absent for this run state/);
assert.match(source, /This is not a zero-LLM or zero-probe result/);
assert.match(source, /href=\{`\/runs\/\$\{coverage\.run_id\}\/cohort\?trace_state=aggregate_only`\}/);
assert.match(source, /open aggregate-only trace cohort/);
assert.match(source, /persisted trace observation/);
assert.match(source, /coverage\.trace_diagnostic\.message/);
assert.match(source, /run lifecycle boundary/);
assert.match(source, /coverage\.trace_diagnostic\.lifecycle_message/);
assert.match(source, /n_substrate_route_steps \+ coverage\.trace_diagnostic\.n_probe_steps/);
assert.match(source, /coverage\.trace_diagnostic\.run_status/);
assert.match(source, /coverage\.trace_diagnostic\.scorer_version \?\? 'not captured'/);
assert.match(source, /\.cov-trace-diagnostic/);
assert.match(source, /\.cov-missing-probes/);
assert.match(source, /\.cov-missing-kicker/);
assert.match(source, /\.cov-diagnostic-link/);

assert.match(cohortSource, /TRACE_STATE_LABELS, TRACE_STATE_NOTES/);
assert.match(cohortSource, /traceStateMeaning/);
assert.match(cohortSource, /makes probe coverage unavailable here, not zero/);
assert.match(cohortSource, /does not prove whether the cause is legacy import, worker failure, migration skew/);
assert.match(cohortSource, /class="trace-state-note" role="note"/);
assert.match(cohortSource, /\.trace-state-note/);

const instance = await DuckDBInstance.create(dbFile);
const con = await instance.connect();
try {
	await con.run(`CREATE TABLE score_run (
		run_id VARCHAR,
		scorer_version VARCHAR,
		architecture VARCHAR,
		started_at TIMESTAMP,
		status VARCHAR,
		terminated_by VARCHAR,
		termination_reason VARCHAR
	)`);
	await con.run(`CREATE TABLE scorer_step (
		run_id VARCHAR,
		step_kind VARCHAR,
		evidence_hash VARCHAR,
		output_json JSON
	)`);
		await con.run(`INSERT INTO score_run VALUES
			('${DECOMP}', 'heuristic-contract-v1', 'decomposed', TIMESTAMP '2026-05-26 00:00:00', 'succeeded', NULL, NULL),
			('${DECOMP_PARTIAL}', 'heuristic-contract-v1', 'decomposed', TIMESTAMP '2026-05-26 00:01:00', 'failed', 'worker_error', 'probe writer exited early'),
			('${DECOMP_PROBE_PARTIAL}', 'heuristic-contract-v1', 'decomposed', TIMESTAMP '2026-05-26 00:01:30', 'failed', 'worker_error', 'aggregate writer exited early'),
			('${MONO}', 'heuristic-contract-v1', 'monolithic', TIMESTAMP '2026-05-26 00:02:00', 'succeeded', NULL, NULL)`);
		await con.run(`INSERT INTO scorer_step VALUES
			('${DECOMP}', 'aggregate', 'ev_decomp_missing_probe', '{"score":0.8,"verdict":"correct","confidence":"high"}'::JSON),
			('${DECOMP_PARTIAL}', 'parse_claim', 'ev_decomp_partial_probe', '{"claim":"A activates B"}'::JSON),
			('${DECOMP_PARTIAL}', 'build_context', 'ev_decomp_partial_probe', '{"context":"partial"}'::JSON),
			('${DECOMP_PARTIAL}', 'aggregate', 'ev_decomp_partial_probe', '{"score":0.7,"verdict":"correct","confidence":"medium"}'::JSON),
			('${DECOMP_PROBE_PARTIAL}', 'substrate_route', 'ev_decomp_probe_partial', '{"subject_role":{"source":"substrate","answer":"yes"},"object_role":{"source":"needs_llm","answer":null},"relation_axis":{"source":"needs_llm","answer":null},"scope":{"source":null,"answer":null}}'::JSON),
			('${DECOMP_PROBE_PARTIAL}', 'object_role_probe', 'ev_decomp_probe_partial', '{"source":"llm","answer":"yes"}'::JSON),
			('${DECOMP_PROBE_PARTIAL}', 'relation_axis_probe', 'ev_decomp_probe_partial', '{"source":"abstain","answer":"abstain"}'::JSON),
			('${MONO}', 'aggregate', 'ev_mono', '{"score":0.6,"verdict":"correct","confidence":"medium"}'::JSON)`);
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

	const decompCoverage = await db.getHeuristicCoverage(DECOMP);
	assert.ok(decompCoverage);
	assert.equal(decompCoverage.architecture, 'decomposed');
		assert.equal(decompCoverage.applicability, 'arch_conditioned');
		assert.equal(decompCoverage.n_evidences, 1);
		assert.equal(decompCoverage.n_probe_evidences, 0);
		assert.deepEqual(decompCoverage.per_probe, []);
	assert.equal(decompCoverage.mean_invoked_probes, 0);
	assert.equal(decompCoverage.all_substrate_rate, 0);
	assert.equal(decompCoverage.short_circuited_rate, 0);
	assert.equal(decompCoverage.trace_diagnostic.kind, 'aggregate_only_trace');
	assert.equal(decompCoverage.trace_diagnostic.n_aggregate_evidences, 1);
	assert.equal(decompCoverage.trace_diagnostic.n_nonaggregate_steps, 0);
	assert.equal(decompCoverage.trace_diagnostic.n_substrate_route_steps, 0);
	assert.equal(decompCoverage.trace_diagnostic.n_probe_steps, 0);
	assert.equal(decompCoverage.trace_diagnostic.run_status, 'succeeded');
	assert.equal(decompCoverage.trace_diagnostic.scorer_version, 'heuristic-contract-v1');
	assert.equal(decompCoverage.trace_diagnostic.lifecycle_kind, 'succeeded_without_cause_provenance');
	assert.match(decompCoverage.trace_diagnostic.message, /0 decomposed native step rows/);
	assert.match(decompCoverage.trace_diagnostic.message, /cannot distinguish/);
	assert.match(decompCoverage.trace_diagnostic.lifecycle_message, /does not support user cancellation/);
	assert.match(decompCoverage.trace_diagnostic.lifecycle_message, /root cause remains unclassified/);

	const partialCoverage = await db.getHeuristicCoverage(DECOMP_PARTIAL);
	assert.ok(partialCoverage);
	assert.equal(partialCoverage.architecture, 'decomposed');
		assert.equal(partialCoverage.applicability, 'arch_conditioned');
		assert.equal(partialCoverage.n_evidences, 1);
		assert.equal(partialCoverage.n_probe_evidences, 0);
		assert.deepEqual(partialCoverage.per_probe, []);
	assert.equal(partialCoverage.trace_diagnostic.kind, 'native_steps_without_probe_slots');
	assert.equal(partialCoverage.trace_diagnostic.n_nonaggregate_steps, 2);
	assert.equal(partialCoverage.trace_diagnostic.n_substrate_route_steps, 0);
	assert.equal(partialCoverage.trace_diagnostic.n_probe_steps, 0);
	assert.equal(partialCoverage.trace_diagnostic.run_status, 'failed');
	assert.equal(partialCoverage.trace_diagnostic.terminated_by, 'worker_error');
	assert.equal(partialCoverage.trace_diagnostic.lifecycle_kind, 'interrupted_run');
	assert.match(partialCoverage.trace_diagnostic.message, /2 decomposed native step rows/);
	assert.match(partialCoverage.trace_diagnostic.message, /probe health is unavailable/);
		assert.match(partialCoverage.trace_diagnostic.lifecycle_message, /interrupted-run context/);
		assert.match(partialCoverage.trace_diagnostic.lifecycle_message, /worker_error/);

		const probePartialCoverage = await db.getHeuristicCoverage(DECOMP_PROBE_PARTIAL);
		assert.ok(probePartialCoverage);
		assert.equal(probePartialCoverage.architecture, 'decomposed');
		assert.equal(probePartialCoverage.applicability, 'arch_conditioned');
		assert.equal(probePartialCoverage.n_evidences, 0);
		assert.equal(probePartialCoverage.n_probe_evidences, 1);
		assert.equal(probePartialCoverage.per_probe.length, 4);
		const byProbe = Object.fromEntries(probePartialCoverage.per_probe.map((row) => [row.probe, row]));
		assert.deepEqual(byProbe.subject_role, {
			probe: 'subject_role',
			total: 1,
			substrate_n: 1,
			llm_n: 0,
			abstain_n: 0,
			notrun_n: 0
		});
		assert.deepEqual(byProbe.object_role, {
			probe: 'object_role',
			total: 1,
			substrate_n: 0,
			llm_n: 1,
			abstain_n: 0,
			notrun_n: 0
		});
		assert.deepEqual(byProbe.relation_axis, {
			probe: 'relation_axis',
			total: 1,
			substrate_n: 0,
			llm_n: 0,
			abstain_n: 1,
			notrun_n: 0
		});
		assert.deepEqual(byProbe.scope, {
			probe: 'scope',
			total: 1,
			substrate_n: 0,
			llm_n: 0,
			abstain_n: 0,
			notrun_n: 1
		});
		assert.equal(probePartialCoverage.mean_invoked_probes, 3);
		assert.equal(probePartialCoverage.all_substrate_rate, 0);
		assert.equal(probePartialCoverage.short_circuited_rate, 1);
		assert.equal(probePartialCoverage.trace_diagnostic.kind, 'probe_rows_present');
		assert.equal(probePartialCoverage.trace_diagnostic.n_aggregate_evidences, 0);
		assert.equal(probePartialCoverage.trace_diagnostic.n_substrate_route_steps, 1);
		assert.equal(probePartialCoverage.trace_diagnostic.n_probe_steps, 2);

		const monoCoverage = await db.getHeuristicCoverage(MONO);
	assert.ok(monoCoverage);
	assert.equal(monoCoverage.architecture, 'monolithic');
	assert.equal(monoCoverage.applicability, 'not_defined');
		assert.match(monoCoverage.not_defined_reason, /defined only for decomposed runs/);
		assert.equal(monoCoverage.n_evidences, 1);
		assert.equal(monoCoverage.n_probe_evidences, 0);
		assert.equal(monoCoverage.trace_diagnostic.kind, 'not_applicable');
	assert.equal(monoCoverage.trace_diagnostic.lifecycle_kind, 'not_applicable');

	console.log('heuristic coverage contract tests passed');
} finally {
	await server.close();
	await rm(dir, { recursive: true, force: true });
}
