import assert from 'node:assert/strict';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createServer } from 'vite';
import { DuckDBInstance } from '@duckdb/node-api';

const RUN = 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa';
const EMPTY_RUN = 'bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb';
const TRACE_RUN = 'cccccccccccccccccccccccccccccccc';
const TRACE_NO_PARSE_RUN = 'dddddddddddddddddddddddddddddddd';
const SUBSTEP_ONLY_RUN = 'eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee';
const NO_SCORE_AGG_RUN = 'ffffffffffffffffffffffffffffffff';
const RUNNING_EMPTY_RUN = '11111111111111111111111111111111';
const RUNNING_SUBSTEP_RUN = '22222222222222222222222222222222';
const RUNNING_NO_SCORE_RUN = '33333333333333333333333333333333';
const RUNNING_ORPHAN_AGG_RUN = '44444444444444444444444444444444';
const TRACE_NO_EVIDENCE_HASH_RUN = '55555555555555555555555555555555';
const TRACE_FIDELITY_SPLIT_RUN = '66666666666666666666666666666666';

function findKind(cohort, kind) {
	return cohort.emptyDiagnostics.find((d) => d.kind === kind) ?? null;
}

async function insertStep(con, {
	run_id,
	step_hash,
	stmt_hash,
	evidence_hash,
	architecture = 'decomposed',
	step_kind,
	output = {},
	error = null,
	started_at = '2026-01-01 00:00:00'
}) {
	await con.run(
		`INSERT INTO scorer_step
		 (step_hash, run_id, stmt_hash, evidence_hash, architecture, model_id,
		  step_kind, output_json, error, latency_ms, prompt_tokens, out_tokens, started_at)
		 VALUES (?, ?, ?, ?, ?, 'smoke-local', ?, ?::JSON, ?, NULL, NULL, NULL, CAST(? AS TIMESTAMP))`,
		[
			step_hash,
			run_id,
			stmt_hash,
			evidence_hash,
			architecture,
			step_kind,
			JSON.stringify(output),
			error,
			started_at
		]
	);
}

const dir = await mkdtemp(join(tmpdir(), 'indra-cohort-diagnostics-'));
const dbFile = join(dir, 'diagnostics.duckdb');
process.env.VIEWER_DUCKDB_PATH = dbFile;

const instance = await DuckDBInstance.create(dbFile);
const con = await instance.connect();
try {
	await con.run('CREATE TABLE score_run (run_id VARCHAR, architecture VARCHAR, status VARCHAR)');
	await con.run(`CREATE TABLE statement (
		stmt_hash VARCHAR,
		indra_type VARCHAR,
		indra_belief DOUBLE,
		supports_count BIGINT,
		supported_by_count BIGINT
	)`);
	await con.run(`CREATE TABLE evidence (
		evidence_hash VARCHAR,
		source_api VARCHAR,
		pmid VARCHAR,
		text VARCHAR
	)`);
	await con.run(`CREATE TABLE statement_evidence (
		stmt_hash VARCHAR,
		evidence_hash VARCHAR,
		evidence_index INTEGER DEFAULT 0,
		source_dump_id VARCHAR
	)`);
	await con.run(`CREATE TABLE agent (
		stmt_hash VARCHAR,
		name VARCHAR,
		role VARCHAR,
		role_index BIGINT
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
	await con.run('CREATE TABLE truth_set (id VARCHAR)');
	await con.run(`CREATE TABLE truth_label (
		truth_set_id VARCHAR,
		target_kind VARCHAR,
		target_id VARCHAR,
		relation_target_id VARCHAR,
		field VARCHAR
	)`);

	await con.run(`INSERT INTO score_run VALUES
		('${RUN}', 'decomposed', 'succeeded'),
		('${EMPTY_RUN}', 'decomposed', 'succeeded'),
		('${TRACE_RUN}', 'decomposed', 'succeeded'),
		('${TRACE_NO_PARSE_RUN}', 'decomposed', 'succeeded'),
		('${SUBSTEP_ONLY_RUN}', 'decomposed', 'succeeded'),
		('${NO_SCORE_AGG_RUN}', 'decomposed', 'succeeded'),
		('${RUNNING_EMPTY_RUN}', 'decomposed', 'running'),
		('${RUNNING_SUBSTEP_RUN}', 'decomposed', 'running'),
		('${RUNNING_NO_SCORE_RUN}', 'decomposed', 'running'),
		('${RUNNING_ORPHAN_AGG_RUN}', 'decomposed', 'running'),
		('${TRACE_NO_EVIDENCE_HASH_RUN}', 'decomposed', 'succeeded'),
		('${TRACE_FIDELITY_SPLIT_RUN}', 'decomposed', 'succeeded')`);
	await con.run(`INSERT INTO statement VALUES
		('stmt1', 'Ubiquitination', 0.80, 0, 0),
		('stmt2', 'Activation', 0.20, 0, 0),
		('stmt3', 'Activation', 0.50, 0, 0),
		('stmt4', 'Complex', 0.50, 0, 0),
		('stmt5', 'MixedSource', 0.40, 0, 0),
		('stmt6', 'SourceBiasCase', 0.50, 0, 0),
		('stmt7', 'SplitFilterCase', 0.50, 0, 0),
		('stmt8', 'TraceFidelitySplitCase', 0.50, 0, 0)`);
	await con.run(`INSERT INTO evidence VALUES
		('ev1', 'paired_smoke', NULL, 'ubiquitination evidence'),
		('ev2', 'paired_smoke', NULL, 'activation evidence'),
		('ev3', 'paired_smoke', NULL, 'trace-only evidence'),
		('ev4', 'cycle40', NULL, 'source exists globally'),
		('ev5a', 'alpha_source', 'PMID-A', 'unscored representative-stratum evidence'),
		('ev5z', 'zz_source', 'PMID-Z', 'scored source evidence'),
		('ev6a', 'alpha_source', 'PMID-6A', 'higher-residual alpha scored evidence'),
		('ev6z', 'zz_source', 'PMID-6Z', 'lower-residual filtered zz scored evidence'),
		('ev7a', 'alpha_source', 'PMID-7A', 'correct alpha evidence'),
		('ev7z', 'zz_source', 'PMID-7Z', 'incorrect zz evidence'),
		('ev8a', 'paired_smoke', 'PMID-8A', 'scored aggregate-only evidence'),
		('ev8b', 'paired_smoke', 'PMID-8B', 'unscored nonaggregate sibling evidence')`);
	await con.run(`INSERT INTO statement_evidence (stmt_hash, evidence_hash, evidence_index) VALUES
		('stmt1', 'ev1', 0),
		('stmt2', 'ev2', 0),
		('stmt3', 'ev3', 0),
		('stmt4', 'ev4', 0),
		('stmt5', 'ev5a', 0), ('stmt5', 'ev5z', 1),
		('stmt6', 'ev6a', 0), ('stmt6', 'ev6z', 1),
		('stmt7', 'ev7a', 0), ('stmt7', 'ev7z', 1),
		('stmt8', 'ev8a', 0), ('stmt8', 'ev8b', 1)`);
	await con.run(`INSERT INTO agent VALUES
		('stmt1', 'A', 'subj', 0),
		('stmt2', 'B', 'subj', 0),
		('stmt3', 'C', 'subj', 0),
		('stmt5', 'E', 'subj', 0),
		('stmt6', 'F', 'subj', 0),
		('stmt7', 'G', 'subj', 0),
		('stmt8', 'H', 'subj', 0)`);
	await con.run("INSERT INTO truth_set VALUES ('gold')");

	await insertStep(con, {
		run_id: RUN,
		step_hash: 'step_agg_1',
		stmt_hash: 'stmt1',
		evidence_hash: 'ev1',
		step_kind: 'aggregate',
		output: { score: 0.9, verdict: 'correct', confidence: 'high' }
	});
	await insertStep(con, {
		run_id: RUN,
		step_hash: 'step_agg_2',
		stmt_hash: 'stmt2',
		evidence_hash: 'ev2',
		step_kind: 'aggregate',
		output: { score: 0.1, verdict: 'incorrect', confidence: 'high' }
	});
	await insertStep(con, {
		run_id: RUN,
		step_hash: 'step_agg_5z',
		stmt_hash: 'stmt5',
		evidence_hash: 'ev5z',
		step_kind: 'aggregate',
		output: { score: 0.7, verdict: 'correct', confidence: 'high' }
	});
	await insertStep(con, {
		run_id: RUN,
		step_hash: 'step_agg_6a',
		stmt_hash: 'stmt6',
		evidence_hash: 'ev6a',
		step_kind: 'aggregate',
		output: { score: 0.0, verdict: 'incorrect', confidence: 'low' }
	});
	await insertStep(con, {
		run_id: RUN,
		step_hash: 'step_agg_6z',
		stmt_hash: 'stmt6',
		evidence_hash: 'ev6z',
		step_kind: 'aggregate',
		output: { score: 0.55, verdict: 'correct', confidence: 'high' }
	});
	await insertStep(con, {
		run_id: RUN,
		step_hash: 'step_parse_6a',
		stmt_hash: 'stmt6',
		evidence_hash: 'ev6a',
		step_kind: 'parse_claim',
		output: { parsed: true }
	});
	await insertStep(con, {
		run_id: RUN,
		step_hash: 'step_agg_7a',
		stmt_hash: 'stmt7',
		evidence_hash: 'ev7a',
		step_kind: 'aggregate',
		output: { score: 0.7, verdict: 'correct', confidence: 'high' }
	});
	await insertStep(con, {
		run_id: RUN,
		step_hash: 'step_agg_7z',
		stmt_hash: 'stmt7',
		evidence_hash: 'ev7z',
		step_kind: 'aggregate',
		output: { score: 0.2, verdict: 'incorrect', confidence: 'low' }
	});
	await insertStep(con, {
		run_id: RUN,
		step_hash: 'step_parse_1',
		stmt_hash: 'stmt1',
		evidence_hash: 'ev1',
		step_kind: 'parse_claim',
		output: { parsed: true }
	});
	await insertStep(con, {
		run_id: RUN,
		step_hash: 'step_agg_8a',
		stmt_hash: 'stmt8',
		evidence_hash: 'ev8a',
		step_kind: 'aggregate',
		output: { score: 0.6, verdict: 'correct', confidence: 'high' }
	});
	await insertStep(con, {
		run_id: RUN,
		step_hash: 'step_parse_8b',
		stmt_hash: 'stmt8',
		evidence_hash: 'ev8b',
		step_kind: 'parse_claim',
		output: { parsed: true }
	});
	await insertStep(con, {
		run_id: TRACE_FIDELITY_SPLIT_RUN,
		step_hash: 'split_trace_agg_8a',
		stmt_hash: 'stmt8',
		evidence_hash: 'ev8a',
		step_kind: 'aggregate',
		output: { score: 0.6, verdict: 'correct', confidence: 'high' }
	});
	await insertStep(con, {
		run_id: TRACE_FIDELITY_SPLIT_RUN,
		step_hash: 'split_trace_parse_8b',
		stmt_hash: 'stmt8',
		evidence_hash: 'ev8b',
		step_kind: 'parse_claim',
		output: { parsed: true }
	});
	await insertStep(con, {
		run_id: TRACE_RUN,
		step_hash: 'trace_parse_1',
		stmt_hash: 'stmt3',
		evidence_hash: 'ev3',
		step_kind: 'parse_claim',
		output: { parsed: true }
	});
	await insertStep(con, {
		run_id: TRACE_NO_PARSE_RUN,
		step_hash: 'trace_build_1',
		stmt_hash: 'stmt4',
		evidence_hash: 'ev4',
		step_kind: 'build_context',
		output: { built: true }
	});
	await insertStep(con, {
		run_id: TRACE_NO_EVIDENCE_HASH_RUN,
		step_hash: 'trace_no_evidence_hash_parse',
		stmt_hash: 'stmt1',
		evidence_hash: null,
		step_kind: 'parse_claim',
		output: { parsed: true }
	});
	await insertStep(con, {
		run_id: SUBSTEP_ONLY_RUN,
		step_hash: 'substep_only_parse',
		stmt_hash: 'stmt1',
		evidence_hash: 'ev1',
		step_kind: 'parse_claim',
		output: { parsed: true }
	});
	await insertStep(con, {
		run_id: RUNNING_SUBSTEP_RUN,
		step_hash: 'running_substep_only_parse',
		stmt_hash: 'stmt1',
		evidence_hash: 'ev1',
		step_kind: 'parse_claim',
		output: { parsed: true }
	});
	await insertStep(con, {
		run_id: NO_SCORE_AGG_RUN,
		step_hash: 'no_score_aggregate',
		stmt_hash: 'stmt2',
		evidence_hash: 'ev2',
		step_kind: 'aggregate',
		output: { verdict: 'abstain', confidence: 'low' }
	});
	await insertStep(con, {
		run_id: RUNNING_NO_SCORE_RUN,
		step_hash: 'running_no_score_aggregate',
		stmt_hash: 'stmt2',
		evidence_hash: 'ev2',
		step_kind: 'aggregate',
		output: { verdict: 'abstain', confidence: 'low' }
	});
	await insertStep(con, {
		run_id: RUNNING_ORPHAN_AGG_RUN,
		step_hash: 'running_orphan_aggregate',
		stmt_hash: 'stmt_missing',
		evidence_hash: 'ev_orphan',
		step_kind: 'aggregate',
		output: { score: 0.7, verdict: 'correct', confidence: 'high' }
	});
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
	const { getRunCohort, closeInstance } = await server.ssrLoadModule('/src/lib/db.ts');

	let cohort = await getRunCohort(RUN, { source: 'cycle40' });
	assert.equal(cohort.totalRows, 0);
	assert.equal(findKind(cohort, 'absent_in_run')?.filter, 'source');
	assert.match(findKind(cohort, 'absent_in_run')?.message ?? '', /this run has no aggregate evidence rows with source=cycle40/);

	cohort = await getRunCohort(RUN, { type: 'Ubiquitination', verdict: 'incorrect' });
	assert.equal(cohort.totalRows, 0);
	assert.match(findKind(cohort, 'no_intersection')?.message ?? '', /each active filter appears in this run/);

	cohort = await getRunCohort(RUN, {
		type: 'Ubiquitination',
		trace_fidelity: 'native_decomposed'
	});
	assert.equal(cohort.totalRows, 1);
	assert.deepEqual(cohort.rows.map((r) => r.evidence_hash), ['ev1']);
	assert.deepEqual(cohort.emptyDiagnostics, []);

	cohort = await getRunCohort(RUN, {
		grain: 'statement',
		type: 'Ubiquitination',
		trace_fidelity: 'native_decomposed'
	});
	assert.equal(cohort.totalRows, 1);
	assert.deepEqual(cohort.rows.map((r) => r.stmt_hash), ['stmt1']);
	assert.deepEqual(cohort.emptyDiagnostics, []);

	cohort = await getRunCohort(RUN, { grain: 'statement', type: 'MixedSource', source: 'zz_source' });
	assert.equal(cohort.totalRows, 1);
	assert.deepEqual(cohort.rows.map((r) => r.stmt_hash), ['stmt5']);
	assert.equal(cohort.rows[0].source_api, 'zz_source');
	assert.equal(cohort.rows[0].source_stratum, 'alpha_source');
	assert.equal(cohort.rows[0].representative_evidence_hash, 'ev5z');
	assert.equal(cohort.rows[0].pmid, 'PMID-Z');
	assert.match(cohort.rows[0].text, /scored source evidence/);
	assert.deepEqual(cohort.emptyDiagnostics, []);

	cohort = await getRunCohort(RUN, { grain: 'statement', type: 'SourceBiasCase', source: 'zz_source' });
	assert.equal(cohort.totalRows, 1);
	assert.deepEqual(cohort.rows.map((r) => r.stmt_hash), ['stmt6']);
	assert.equal(cohort.rows[0].source_api, 'zz_source');
	assert.equal(cohort.rows[0].representative_evidence_hash, 'ev6z');
	assert.equal(cohort.rows[0].pmid, 'PMID-6Z');
	assert.match(cohort.rows[0].text, /filtered zz scored evidence/);

	cohort = await getRunCohort(RUN, {
		grain: 'statement',
		type: 'SourceBiasCase',
		source: 'alpha_source',
		trace_fidelity: 'native_decomposed'
	});
	assert.equal(cohort.totalRows, 1);
	assert.deepEqual(cohort.rows.map((r) => r.stmt_hash), ['stmt6']);
	assert.equal(cohort.rows[0].source_api, 'alpha_source');
	assert.equal(cohort.rows[0].representative_evidence_hash, 'ev6a');

	cohort = await getRunCohort(RUN, {
		grain: 'statement',
		type: 'SourceBiasCase',
		source: 'zz_source',
		trace_fidelity: 'native_decomposed'
	});
	assert.equal(cohort.totalRows, 0);
	assert.match(findKind(cohort, 'no_intersection')?.message ?? '', /one scored evidence row satisfying them together/);

	cohort = await getRunCohort(RUN, {
		grain: 'statement',
		type: 'SourceBiasCase',
		trace_fidelity: 'aggregate_only'
	});
	assert.equal(cohort.totalRows, 1);
	assert.deepEqual(cohort.rows.map((r) => r.stmt_hash), ['stmt6']);
	assert.equal(cohort.rows[0].source_api, 'zz_source');
	assert.equal(cohort.rows[0].representative_evidence_hash, 'ev6z');

	cohort = await getRunCohort(RUN, {
		grain: 'statement',
		type: 'SourceBiasCase',
		verdict: 'incorrect',
		trace_fidelity: 'aggregate_only'
	});
	assert.equal(cohort.totalRows, 0);
	assert.match(findKind(cohort, 'no_intersection')?.message ?? '', /one scored evidence row satisfying them together/);
	assert.notEqual(findKind(cohort, 'absent_in_run')?.filter, 'trace_fidelity');

	cohort = await getRunCohort(RUN, {
		grain: 'statement',
		type: 'SplitFilterCase',
		source: 'zz_source',
		verdict: 'correct'
	});
	assert.equal(cohort.totalRows, 0);
	assert.match(findKind(cohort, 'no_intersection')?.message ?? '', /one scored evidence row satisfying them together/);

	cohort = await getRunCohort(RUN, {
		grain: 'statement',
		type: 'SplitFilterCase',
		source: 'zz_source',
		verdict: 'incorrect'
	});
	assert.equal(cohort.totalRows, 1);
	assert.deepEqual(cohort.rows.map((r) => r.stmt_hash), ['stmt7']);
	assert.equal(cohort.rows[0].source_api, 'zz_source');
	assert.equal(cohort.rows[0].representative_evidence_hash, 'ev7z');
	assert.equal(cohort.rows[0].pmid, 'PMID-7Z');
	assert.match(cohort.rows[0].text, /incorrect zz evidence/);
	assert.ok(Math.abs(cohort.rows[0].score - 0.45) < 1e-9);
	assert.ok(Math.abs(cohort.rows[0].residual - -0.05) < 1e-9);

	cohort = await getRunCohort(RUN, {
		grain: 'statement',
		type: 'TraceFidelitySplitCase',
		trace_fidelity: 'aggregate_only'
	});
	assert.equal(cohort.totalRows, 1);
	assert.deepEqual(cohort.rows.map((r) => r.stmt_hash), ['stmt8']);

	cohort = await getRunCohort(TRACE_FIDELITY_SPLIT_RUN, {
		trace_fidelity: 'aggregate_only'
	});
	assert.equal(cohort.totalRows, 1);
	assert.deepEqual(cohort.rows.map((r) => r.evidence_hash), ['ev8a']);

	cohort = await getRunCohort(TRACE_FIDELITY_SPLIT_RUN, {
		trace_fidelity: 'native_decomposed'
	});
	assert.equal(cohort.totalRows, 0);
	assert.equal(findKind(cohort, 'absent_in_run')?.filter, 'trace_fidelity');
	assert.match(findKind(cohort, 'absent_in_run')?.message ?? '', /trace_fidelity=native_decomposed/);

	cohort = await getRunCohort(TRACE_FIDELITY_SPLIT_RUN, {
		grain: 'statement',
		trace_fidelity: 'aggregate_only'
	});
	assert.equal(cohort.totalRows, 1);
	assert.deepEqual(cohort.rows.map((r) => r.stmt_hash), ['stmt8']);

	cohort = await getRunCohort(TRACE_FIDELITY_SPLIT_RUN, {
		grain: 'statement',
		trace_fidelity: 'native_decomposed'
	});
	assert.equal(cohort.totalRows, 0);
	assert.equal(findKind(cohort, 'absent_in_run')?.filter, 'trace_fidelity');
	assert.match(findKind(cohort, 'absent_in_run')?.message ?? '', /trace_fidelity=native_decomposed/);

	cohort = await getRunCohort(RUN, { grain: 'statement', source: 'cycle40' });
	assert.equal(cohort.totalRows, 0);
	assert.equal(findKind(cohort, 'absent_in_run')?.filter, 'source');
	assert.match(findKind(cohort, 'absent_in_run')?.message ?? '', /scored aggregate evidence from that source/);
	assert.match(findKind(cohort, 'absent_in_run')?.message ?? '', /use evidence grain/);

	cohort = await getRunCohort(RUN, { grain: 'statement', source_stratum: 'zz_source' });
	assert.equal(cohort.totalRows, 0);
	assert.equal(findKind(cohort, 'absent_in_run')?.filter, 'source_stratum');
	assert.match(findKind(cohort, 'absent_in_run')?.message ?? '', /statement-level stratum/);
	assert.match(findKind(cohort, 'absent_in_run')?.message ?? '', /statement-level minimum evidence source/);

	cohort = await getRunCohort(RUN, { step_kind: 'parse_claim' });
	assert.equal(cohort.totalRows, 0);
	assert.equal(findKind(cohort, 'not_applicable')?.filter, 'step_kind');

	cohort = await getRunCohort(EMPTY_RUN, {});
	assert.equal(cohort.totalRows, 0);
	assert.equal(cohort.emptyDiagnostics.length, 1);
	assert.equal(findKind(cohort, 'empty_run')?.kind, 'empty_run');
	assert.match(findKind(cohort, 'empty_run')?.message ?? '', /no scorer_step rows/);
	assert.doesNotMatch(findKind(cohort, 'empty_run')?.message ?? '', /yet/);

	cohort = await getRunCohort(SUBSTEP_ONLY_RUN, {});
	assert.equal(cohort.totalRows, 0);
	assert.equal(findKind(cohort, 'empty_run')?.kind, 'empty_run');
	assert.match(findKind(cohort, 'empty_run')?.message ?? '', /persisted substep row/);
	assert.match(findKind(cohort, 'empty_run')?.message ?? '', /no aggregate evidence scorer rows/);

	cohort = await getRunCohort(EMPTY_RUN, { grain: 'statement' });
	assert.equal(cohort.totalRows, 0);
	assert.equal(findKind(cohort, 'empty_run')?.kind, 'empty_run');
	assert.match(findKind(cohort, 'empty_run')?.message ?? '', /no scorer_step rows/);
	assert.doesNotMatch(findKind(cohort, 'empty_run')?.message ?? '', /yet/);

	cohort = await getRunCohort(SUBSTEP_ONLY_RUN, { grain: 'statement' });
	assert.equal(cohort.totalRows, 0);
	assert.equal(findKind(cohort, 'empty_run')?.kind, 'empty_run');
	assert.match(findKind(cohort, 'empty_run')?.message ?? '', /persisted substep row/);
	assert.match(findKind(cohort, 'empty_run')?.message ?? '', /statement cohorts summarize aggregate rollups/);

	cohort = await getRunCohort(NO_SCORE_AGG_RUN, { grain: 'statement' });
	assert.equal(cohort.totalRows, 0);
	assert.equal(findKind(cohort, 'empty_run')?.kind, 'empty_run');
	assert.match(findKind(cohort, 'empty_run')?.message ?? '', /aggregate scorer row/);
	assert.match(findKind(cohort, 'empty_run')?.message ?? '', /none have numeric scores/);

	cohort = await getRunCohort(RUNNING_NO_SCORE_RUN, { grain: 'statement' });
	assert.equal(cohort.totalRows, 0);
	assert.equal(findKind(cohort, 'empty_run')?.kind, 'empty_run');
	assert.match(findKind(cohort, 'empty_run')?.message ?? '', /still running/);
	assert.match(findKind(cohort, 'empty_run')?.message ?? '', /none have numeric scores/);

	cohort = await getRunCohort(NO_SCORE_AGG_RUN, {});
	assert.equal(cohort.totalRows, 1);
	assert.deepEqual(cohort.emptyDiagnostics, []);

	cohort = await getRunCohort(RUNNING_ORPHAN_AGG_RUN, {});
	assert.equal(cohort.totalRows, 0);
	assert.equal(findKind(cohort, 'empty_run')?.kind, 'empty_run');
	assert.match(findKind(cohort, 'empty_run')?.message ?? '', /still running/);
	assert.match(findKind(cohort, 'empty_run')?.message ?? '', /aggregate scorer row/);
	assert.match(findKind(cohort, 'empty_run')?.message ?? '', /none join to statement rows/);

	cohort = await getRunCohort(RUNNING_ORPHAN_AGG_RUN, { grain: 'statement' });
	assert.equal(cohort.totalRows, 0);
	assert.equal(findKind(cohort, 'empty_run')?.kind, 'empty_run');
	assert.match(findKind(cohort, 'empty_run')?.message ?? '', /still running/);
	assert.match(findKind(cohort, 'empty_run')?.message ?? '', /scored aggregate rows do not join to statements/);

	cohort = await getRunCohort(TRACE_NO_EVIDENCE_HASH_RUN, { trace_state: 'missing_aggregate' });
	assert.equal(cohort.totalRows, 0);
	assert.equal(findKind(cohort, 'empty_run')?.kind, 'empty_run');
	assert.match(findKind(cohort, 'empty_run')?.message ?? '', /scorer_step row/);
	assert.match(findKind(cohort, 'empty_run')?.message ?? '', /no trace evidence rows in this trace-state plane/);

	cohort = await getRunCohort(RUNNING_EMPTY_RUN, {});
	assert.equal(cohort.totalRows, 0);
	assert.equal(findKind(cohort, 'empty_run')?.kind, 'empty_run');
	assert.match(findKind(cohort, 'empty_run')?.message ?? '', /still running/);
	assert.match(findKind(cohort, 'empty_run')?.message ?? '', /no scorer_step rows yet/);

	cohort = await getRunCohort(RUNNING_SUBSTEP_RUN, {});
	assert.equal(cohort.totalRows, 0);
	assert.equal(findKind(cohort, 'empty_run')?.kind, 'empty_run');
	assert.match(findKind(cohort, 'empty_run')?.message ?? '', /still running/);
	assert.match(findKind(cohort, 'empty_run')?.message ?? '', /persisted substep row/);

	cohort = await getRunCohort(TRACE_RUN, {
		trace_state: 'missing_aggregate',
		trace_snapshot: '2026-01-01 00:00:00',
		type: 'Ubiquitination'
	});
	assert.equal(cohort.totalRows, 0);
	assert.match(findKind(cohort, 'absent_in_run')?.message ?? '', /this run snapshot has no trace evidence rows with type=Ubiquitination/);

	cohort = await getRunCohort(TRACE_NO_PARSE_RUN, {
		trace_state: 'missing_aggregate',
		trace_snapshot: '2026-01-01 00:00:00',
		step_kind: 'parse_claim'
	});
	assert.equal(cohort.totalRows, 0);
	assert.equal(findKind(cohort, 'absent_in_run')?.filter, 'step_kind');
	assert.match(findKind(cohort, 'absent_in_run')?.message ?? '', /this run snapshot has no trace evidence rows with step_kind=parse_claim/);

	closeInstance();
	console.log('cohort diagnostic tests passed');
} finally {
	await server.close();
	await rm(dir, { recursive: true, force: true });
}
