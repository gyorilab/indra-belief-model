import assert from 'node:assert/strict';
import { readFile, mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createServer } from 'vite';
import { DuckDBInstance } from '@duckdb/node-api';

const RUN_ID = 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa';

const dir = await mkdtemp(join(tmpdir(), 'indra-truth-overlap-'));
const dbFile = join(dir, 'overlap.duckdb');
process.env.VIEWER_DUCKDB_PATH = dbFile;

const instance = await DuckDBInstance.create(dbFile);
const con = await instance.connect();
try {
	await con.run(`
		CREATE TABLE score_run (
			run_id VARCHAR,
			scorer_version VARCHAR,
			architecture VARCHAR,
			paired_run_group_id VARCHAR,
			started_at TIMESTAMP,
			status VARCHAR,
			terminated_by VARCHAR,
			termination_reason VARCHAR,
			n_stmts INTEGER,
			cost_estimate_usd DOUBLE
		)
	`);
	await con.run(`
		CREATE TABLE truth_set (
			id VARCHAR,
			name VARCHAR
		)
	`);
	await con.run(`
		CREATE TABLE truth_label (
			label_id BIGINT,
			truth_set_id VARCHAR,
			target_kind VARCHAR,
			target_id VARCHAR,
			relation_target_id VARCHAR,
			field VARCHAR,
			value_text VARCHAR,
			provenance VARCHAR
		)
	`);
	await con.run(`
		CREATE TABLE statement (
			stmt_hash VARCHAR,
			indra_type VARCHAR,
			indra_belief DOUBLE,
			supports_count INTEGER,
			supported_by_count INTEGER,
			source_dump_id VARCHAR
		)
	`);
	await con.run(`
		CREATE TABLE evidence (
			evidence_hash VARCHAR,
			stmt_hash VARCHAR,
			source_api VARCHAR,
			text TEXT
		)
	`);
	await con.run(`
		CREATE TABLE agent (
			stmt_hash VARCHAR,
			agent_hash VARCHAR,
			role VARCHAR,
			name VARCHAR
		)
	`);
	await con.run(`
		CREATE TABLE supports_edge (
			from_stmt_hash VARCHAR,
			to_stmt_hash VARCHAR,
			kind VARCHAR
		)
	`);
	await con.run(`
		CREATE TABLE scorer_step (
			step_hash VARCHAR,
			stmt_hash VARCHAR,
			evidence_hash VARCHAR,
			run_id VARCHAR,
			step_kind VARCHAR,
			output_json JSON
		)
	`);
	await con.run(`
		CREATE TABLE metric (
			run_id VARCHAR,
			truth_set_id VARCHAR,
			metric_name VARCHAR,
			value DOUBLE,
			slice_json JSON
		)
	`);

	await con.run(`INSERT INTO score_run VALUES ('${RUN_ID}', 'test', 'decomposed', NULL, TIMESTAMP '2026-05-25 12:00:00', 'succeeded', NULL, NULL, 1, 0.01)`);
	await con.run(`
		INSERT INTO truth_set VALUES
		('gold', 'Gold overlap fixture'),
		('miss', 'Missing overlap fixture')
	`);
	await con.run(`
		INSERT INTO statement VALUES
		('stmt_compared', 'Activation', 0.8, 0, 0, 'fixture_dump'),
		('stmt_gold_context', 'Activation', 0.8, 0, 0, 'fixture_dump'),
		('stmt_other_context', 'Activation', 0.8, 0, 0, 'fixture_dump'),
		('stmt_no_verdict', 'Activation', 0.8, 0, 0, 'fixture_dump'),
		('stmt_unscored', 'Activation', 0.8, 0, 0, 'fixture_dump')
	`);
	await con.run(`
		INSERT INTO evidence VALUES
		('ev_compared', 'stmt_compared', 'reach', 'compared evidence'),
		('ev_context', 'stmt_gold_context', 'reach', 'contextual evidence'),
		('ev_no_verdict', 'stmt_no_verdict', 'biopax', 'no verdict evidence'),
		('ev_unscored', 'stmt_unscored', 'signor', 'unscored evidence')
	`);
	await con.run(`
		INSERT INTO truth_label VALUES
		(1, 'gold', 'evidence', 'ev_compared', NULL, 'tag', 'correct', 'fixture'),
		(2, 'gold', 'evidence', 'ev_context', 'stmt_gold_context', 'tag', 'correct', 'fixture'),
		(3, 'gold', 'evidence', 'ev_missing', NULL, 'tag', 'correct', 'fixture'),
		(4, 'gold', 'evidence', 'ev_no_verdict', NULL, 'tag', 'incorrect', 'fixture'),
		(5, 'gold', 'evidence', 'ev_unscored', NULL, 'tag', 'correct', 'fixture'),
		(6, 'miss', 'evidence', 'ev_unscored', NULL, 'tag', 'correct', 'fixture')
	`);
	await con.run(`
		INSERT INTO scorer_step VALUES
		('step_compared', 'stmt_compared', 'ev_compared', '${RUN_ID}', 'aggregate', '{"verdict":"correct"}'),
		('step_context', 'stmt_other_context', 'ev_context', '${RUN_ID}', 'aggregate', '{"verdict":"correct"}'),
		('step_no_verdict', 'stmt_no_verdict', 'ev_no_verdict', '${RUN_ID}', 'aggregate', '{}')
	`);
	const measuredSlice = JSON.stringify({
		step_kind: 'aggregate',
		n_compared: 1,
		tp: 1,
		fp: 0,
		fn: 0,
		tn: 0,
		n_gold_labels: 5,
		n_applicable_gold_labels: 1,
		n_scored_evidences: 2,
		gold_fields: ['tag'],
		positive_gold_label: 'correct',
		negative_gold_rule: 'any value != correct'
	});
	for (const metric of ['precision', 'recall', 'f1']) {
		await con.run(`INSERT INTO metric VALUES ('${RUN_ID}', 'gold', 'truth_present.aggregate.${metric}', 1.0, '${measuredSlice}')`);
	}
	const unavailableSlice = JSON.stringify({
		step_kind: 'aggregate',
		n_compared: 0,
		tp: 0,
		fp: 0,
		fn: 0,
		tn: 0,
		n_gold_labels: 1,
		n_applicable_gold_labels: 0,
		n_scored_evidences: 2,
		gold_fields: ['tag'],
		positive_gold_label: 'correct',
		negative_gold_rule: 'any value != correct',
		unavailable_reason: 'no scored aggregate evidence rows overlap this truth_set'
	});
	for (const metric of ['precision', 'recall', 'f1']) {
		await con.run(`INSERT INTO metric VALUES ('${RUN_ID}', 'miss', 'truth_present.aggregate.${metric}', 0.0, '${unavailableSlice}')`);
	}
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
	const detail = await db.getTruthSetOverlapDetail(RUN_ID, 'gold');
	assert.equal(detail.run_id, RUN_ID);
	assert.equal(detail.truth_set_id, 'gold');
	assert.equal(detail.n_gold_labels, 5);
	assert.equal(detail.n_applicable_gold_labels, 1);
	assert.equal(detail.n_metric_compared_rows, 1);
	assert.equal(detail.n_scored_evidences, 2);
	assert.equal(detail.n_compared_labels, 1);
	assert.equal(detail.n_no_aggregate_verdict, 1);
	assert.equal(detail.n_context_mismatch, 1);
	assert.equal(detail.n_not_scored_in_run, 1);
	assert.equal(detail.n_not_in_corpus, 1);
	assert.equal(detail.cohort_href, `/runs/${RUN_ID}/cohort?truth_set=gold&step_kind=aggregate&verdict_present=true`);

	const statuses = new Map(detail.rows.map((row) => [row.target_id, row.status]));
	assert.equal(statuses.get('ev_compared'), 'compared');
	assert.equal(statuses.get('ev_context'), 'context_mismatch');
	assert.equal(statuses.get('ev_missing'), 'not_in_corpus');
	assert.equal(statuses.get('ev_no_verdict'), 'no_aggregate_verdict');
	assert.equal(statuses.get('ev_unscored'), 'not_scored_in_run');
	const missingDetail = await db.getTruthSetOverlapDetail(RUN_ID, 'miss');
	assert.equal(missingDetail.n_metric_compared_rows, 0);
	assert.equal(missingDetail.cohort_href, null);
	assert.equal(missingDetail.rows[0].status, 'not_scored_in_run');

	const route = await server.ssrLoadModule('/src/routes/runs/[run_id]/truth-sets/[truth_set_id]/+page.server.ts');
	const loaded = await route.load({
		params: { run_id: RUN_ID, truth_set_id: 'gold' },
		url: new URL(`http://localhost/runs/${RUN_ID}/truth-sets/gold?step_kind=aggregate`)
	});
	assert.equal(loaded.detail.n_metric_compared_rows, 1);

	const overview = await db.getCorpusOverview();
	const goldValidity = overview.latestValidity.truthPresent.find((row) => row.truth_set_id === 'gold');
	const missValidity = overview.latestValidity.truthPresent.find((row) => row.truth_set_id === 'miss');
	assert.ok(goldValidity);
	assert.ok(missValidity);
	assert.equal(goldValidity.truth_href, `/runs/${RUN_ID}/truth-sets/gold?step_kind=aggregate`);
	assert.equal(
		goldValidity.cohort_href,
		`/runs/${RUN_ID}/cohort?truth_set=gold&step_kind=aggregate&verdict_present=true`
	);
	assert.equal(missValidity.truth_href, `/runs/${RUN_ID}/truth-sets/miss?step_kind=aggregate`);
	assert.equal(missValidity.n_compared, 0);
	const broadCohort = await db.getRunCohort(RUN_ID, { truth_set: 'gold', step_kind: 'aggregate' });
	const exactCohort = await db.getRunCohort(RUN_ID, { truth_set: 'gold', step_kind: 'aggregate', verdict_present: true });
	assert.equal(broadCohort.totalRows, 2);
	assert.equal(exactCohort.totalRows, 1);

	const validitySource = await readFile('src/lib/components/Validity.svelte', 'utf8');
	assert.match(validitySource, /href=\{g\.truth_href\}/);
	assert.match(validitySource, /href=\{g\.cohort_href\}/);

	db.closeInstance();
	console.log('truth-set overlap route tests passed');
} finally {
	await server.close();
	await rm(dir, { recursive: true, force: true });
}
