import assert from 'node:assert/strict';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createServer } from 'vite';
import { DuckDBInstance } from '@duckdb/node-api';

const RUN = 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa';
const dir = await mkdtemp(join(tmpdir(), 'indra-denominator-routes-'));
const dbFile = join(dir, 'routes.duckdb');
process.env.VIEWER_DUCKDB_PATH = dbFile;
process.env.INDRA_VIEWER_SUPPRESS_WRITER_GUARD_LOG = '1';

function hrefSearch(href) {
	return new URL(href, 'http://localhost').searchParams;
}

const instance = await DuckDBInstance.create(dbFile);
const con = await instance.connect();
try {
	await con.run(`CREATE TABLE statement (
		stmt_hash VARCHAR,
		indra_type VARCHAR,
		indra_belief DOUBLE,
		source_dump_id VARCHAR,
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
		agent_hash VARCHAR,
		role VARCHAR,
		name VARCHAR,
		role_index BIGINT
	)`);
	await con.run('CREATE TABLE supports_edge (from_stmt_hash VARCHAR, to_stmt_hash VARCHAR, kind VARCHAR)');
	await con.run('CREATE TABLE truth_set (id VARCHAR, name VARCHAR)');
	await con.run(`CREATE TABLE truth_label (
		label_id BIGINT,
		truth_set_id VARCHAR,
		target_kind VARCHAR,
		target_id VARCHAR,
		field VARCHAR,
		relation_target_id VARCHAR
	)`);
	await con.run(`CREATE TABLE score_run (
		run_id VARCHAR,
		scorer_version VARCHAR,
		architecture VARCHAR,
		paired_run_group_id VARCHAR,
		started_at TIMESTAMP,
		status VARCHAR,
		terminated_by VARCHAR,
		termination_reason VARCHAR,
		n_stmts BIGINT,
		cost_estimate_usd DOUBLE
	)`);
	await con.run(`CREATE TABLE scorer_step (
		run_id VARCHAR,
		architecture VARCHAR,
		stmt_hash VARCHAR,
		evidence_hash VARCHAR,
		step_kind VARCHAR,
		output_json JSON,
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
	await con.run(`INSERT INTO score_run VALUES
		('${RUN}', 'test-v1', 'monolithic', NULL, TIMESTAMP '2026-05-25 00:00:00', 'succeeded', NULL, NULL, 3, 0.0)`);
	await con.run(`INSERT INTO statement VALUES
		('stmt_anchor', 'Activation', 0.80, 'demo', 0, 0),
		('stmt_unanchored', 'Activation', NULL, 'demo', 0, 0),
		('stmt_no_score', 'Activation', 0.25, 'demo', 0, 0)`);
	await con.run(`INSERT INTO evidence VALUES
		('ev_anchor', 'reach', NULL, 'anchored scored evidence'),
		('ev_unanchored', 'reach', NULL, 'unanchored scored evidence'),
		('ev_no_score', 'reach', NULL, 'anchored verdict without score')`);
	await con.run(`INSERT INTO statement_evidence (stmt_hash, evidence_hash, evidence_index) VALUES
		('stmt_anchor', 'ev_anchor', 0),
		('stmt_unanchored', 'ev_unanchored', 0),
		('stmt_no_score', 'ev_no_score', 0)`);
	await con.run(`INSERT INTO scorer_step VALUES
		('${RUN}', 'monolithic', 'stmt_anchor', 'ev_anchor', 'aggregate', '{"score":0.90,"verdict":"correct","confidence":"high"}'::JSON, NULL, NULL, NULL, TIMESTAMP '2026-05-25 00:00:01'),
		('${RUN}', 'monolithic', 'stmt_unanchored', 'ev_unanchored', 'aggregate', '{"score":0.70,"verdict":"correct","confidence":"high"}'::JSON, NULL, NULL, NULL, TIMESTAMP '2026-05-25 00:00:02'),
		('${RUN}', 'monolithic', 'stmt_no_score', 'ev_no_score', 'aggregate', '{"verdict":"correct","confidence":"high"}'::JSON, NULL, NULL, NULL, TIMESTAMP '2026-05-25 00:00:03')`);
	await con.run(`INSERT INTO metric VALUES
		('${RUN}', NULL, 'verdict_share.correct', 1.0, '{"n":3}'::JSON),
		('${RUN}', 'indra_published_belief', 'indra_belief_calibration.mae', 0.10, '{"n_stmts":1}'::JSON),
		('${RUN}', 'indra_published_belief', 'indra_belief_calibration.rmse', 0.10, '{"n_stmts":1}'::JSON),
		('${RUN}', 'indra_published_belief', 'indra_belief_calibration.bias', 0.10, '{"n_stmts":1}'::JSON)`);
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
	const { cohortFiltersFromSearchParams } = await server.ssrLoadModule('/src/lib/server/runCohortContract.ts');
	const overview = await db.getCorpusOverview();
	const validity = overview.latestValidity;
	assert.ok(validity);

	assert.equal(
		validity.calibration.cohort_href,
		`/runs/${RUN}/cohort?grain=statement&indra_belief_present=true`
	);
	const calibrationCohort = await db.getRunCohort(
		RUN,
		cohortFiltersFromSearchParams(hrefSearch(validity.calibration.cohort_href))
	);
	assert.equal(calibrationCohort.totalRows, validity.calibration.n_stmts);
	assert.deepEqual(calibrationCohort.rows.map((r) => r.stmt_hash), ['stmt_anchor']);

	const confidence = validity.confidenceCalibration.find((r) => r.family === 'all' && r.confidence === 'high');
	assert.ok(confidence);
	assert.equal(confidence.n, 1);
	assert.equal(
		confidence.cohort_href,
		`/runs/${RUN}/cohort?confidence=high&score_present=true&indra_belief_present=true`
	);
	const confidenceCohort = await db.getRunCohort(
		RUN,
		cohortFiltersFromSearchParams(hrefSearch(confidence.cohort_href))
	);
	assert.equal(confidenceCohort.totalRows, confidence.n);
	assert.deepEqual(confidenceCohort.rows.map((r) => r.evidence_hash), ['ev_anchor']);

	const verdictMetric = validity.metricTaxonomy.find((row) => row.panel === 'verdict distribution');
	assert.ok(verdictMetric);
	assert.equal(verdictMetric.cohort_href, `/runs/${RUN}/cohort?verdict_present=true`);
	const verdictCohort = await db.getRunCohort(
		RUN,
		cohortFiltersFromSearchParams(hrefSearch(verdictMetric.cohort_href))
	);
	assert.equal(verdictCohort.totalRows, validity.verdicts.reduce((sum, row) => sum + row.n, 0));

	console.log('summary denominator route-contract tests passed');
} finally {
	await server.close();
	await rm(dir, { recursive: true, force: true });
}
