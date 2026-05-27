import assert from 'node:assert/strict';
import { createServer as createHttpServer } from 'node:http';
import { mkdir, mkdtemp, readFile, rename, rm, unlink, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createServer } from 'vite';
import { DuckDBInstance } from '@duckdb/node-api';

const RUN = 'eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee';
const RUNNING_RUN = 'ffffffffffffffffffffffffffffffff';

const repairPageSource = await readFile(new URL('../src/routes/runs/[run_id]/repairs/+page.svelte', import.meta.url), 'utf8');
const scoreRouteSource = await readFile(new URL('../src/routes/api/runs/score/+server.ts', import.meta.url), 'utf8');
const scoringSource = await readFile(new URL('../../src/indra_belief/corpus/scoring.py', import.meta.url), 'utf8');
const workerSource = await readFile(new URL('../../src/indra_belief/worker.py', import.meta.url), 'utf8');
assert.match(repairPageSource, /probe_step_filter: probeStepFilterForCorpus\(corpus\)/);
assert.match(repairPageSource, /probe_only: probeOnlyForCorpus\(corpus\)/);
assert.match(repairPageSource, /deterministic aggregate re-adjudication/);
assert.match(repairPageSource, /not fresh aggregate LLM calls/);
assert.match(scoreRouteSource, /--probe-step-filter/);
assert.match(scoreRouteSource, /--probe-only/);
assert.match(workerSource, /probe_step_filter=probe_step_filter/);
assert.match(workerSource, /probe_only=probe_only/);
assert.match(scoringSource, /probe_step_filter/);
assert.match(scoringSource, /probe_only/);
assert.match(scoringSource, /PROBE_STEP_KIND_TO_TRACE_KEY/);

async function insertStep(con, {
	run_id = RUN,
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

async function correctionRows(dbFile, sourceRoute) {
	const instance = await DuckDBInstance.create(dbFile);
	const con = await instance.connect();
	try {
			const reader = await con.runAndReadAll(
				`SELECT
				   correction_id,
				   stmt_hash,
				   evidence_hash,
				   step_hash,
			   json_extract_string(value_json, '$.observed.trace_state') AS trace_state,
			   json_extract_string(value_json, '$.observed.probe_coverage') AS probe_coverage,
			   json_extract_string(value_json, '$.observed.missing_probe_slots') AS missing_probe_slots,
			   json_extract_string(value_json, '$.suspected_step_kind') AS suspected_step_kind
			 FROM scorer_step_correction
			 WHERE source_route=?
			   AND correction_kind='repair_candidate'
			 ORDER BY stmt_hash, evidence_hash, step_hash`,
			[sourceRoute]
		);
			return reader.getRowObjects().map((r) => ({
				correction_id: Number(r.correction_id),
				stmt_hash: String(r.stmt_hash),
				evidence_hash: r.evidence_hash == null ? null : String(r.evidence_hash),
				step_hash: String(r.step_hash),
			trace_state: r.trace_state == null ? null : String(r.trace_state),
			probe_coverage: r.probe_coverage == null ? null : String(r.probe_coverage),
			missing_probe_slots: r.missing_probe_slots == null ? null : String(r.missing_probe_slots),
			suspected_step_kind: r.suspected_step_kind == null ? null : String(r.suspected_step_kind)
		}));
	} finally {
		con.disconnectSync?.();
		instance.closeSync();
	}
}

async function probeSlotReviewRows(dbFile, parentCorrectionId) {
	const instance = await DuckDBInstance.create(dbFile);
	const con = await instance.connect();
	try {
		const reader = await con.runAndReadAll(
			`SELECT
			   correction_id,
			   parent_correction_id,
			   status,
			   reviewer,
			   note,
			   json_extract_string(value_json, '$.selected_probe_slots') AS selected_probe_slots,
			   json_extract_string(value_json, '$.missing_probe_slots') AS missing_probe_slots
			 FROM scorer_step_correction
			 WHERE correction_kind='probe_slot_review'
			   AND parent_correction_id=?
			 ORDER BY correction_id`,
			[parentCorrectionId]
		);
		return reader.getRowObjects().map((r) => ({
			correction_id: Number(r.correction_id),
			parent_correction_id: Number(r.parent_correction_id),
			status: String(r.status),
			reviewer: r.reviewer == null ? null : String(r.reviewer),
			note: r.note == null ? null : String(r.note),
			selected_probe_slots: r.selected_probe_slots == null ? null : String(r.selected_probe_slots),
			missing_probe_slots: r.missing_probe_slots == null ? null : String(r.missing_probe_slots)
		}));
	} finally {
		con.disconnectSync?.();
		instance.closeSync();
	}
}

async function statementRawJson(dbFile, stmtHash) {
	const instance = await DuckDBInstance.create(dbFile);
	const con = await instance.connect();
	try {
		const reader = await con.runAndReadAll(
			`SELECT CAST(raw_json AS VARCHAR) AS raw_json
			   FROM statement
			  WHERE stmt_hash=?`,
			[stmtHash]
		);
		const row = reader.getRowObjects()[0];
		assert.ok(row, `statement fixture ${stmtHash} exists`);
		return String(row.raw_json);
	} finally {
		con.disconnectSync?.();
		instance.closeSync();
	}
}

async function updateStatementRawJson(dbFile, stmtHash, rawJson) {
	const instance = await DuckDBInstance.create(dbFile);
	const con = await instance.connect();
	try {
		await con.run(
			`UPDATE statement
			    SET raw_json=?::JSON
			  WHERE stmt_hash=?`,
			[rawJson, stmtHash]
		);
	} finally {
		con.disconnectSync?.();
		instance.closeSync();
	}
}

async function insertStatementScopeCorrection(dbFile, sourceRoute) {
	const instance = await DuckDBInstance.create(dbFile);
	const con = await instance.connect();
	try {
		await con.run(
			`INSERT INTO scorer_step_correction
			 (step_hash, run_id, architecture, stmt_hash, evidence_hash,
			  correction_kind, status, reviewer, note, value_json,
			  source_route, source_filters_json)
			 VALUES (?, ?, 'decomposed', 'stmt1', NULL, 'repair_candidate', 'open', 'parity-test',
			         'legacy statement-scope repair candidate', ?::JSON, ?, ?::JSON)`,
			[
				'stmt_scope_stmt1',
				RUN,
				JSON.stringify({
					kind: 'cohort_repair_candidate',
					grain: 'statement',
					suspected_step_kind: 'aggregate',
					severity: 'untriaged',
					observed: { residual: 0.51 },
					source_route: sourceRoute
				}),
				sourceRoute,
				JSON.stringify({ grain: 'statement', fixture: 'legacy_statement_scope' })
			]
		);
		const reader = await con.runAndReadAll(
			`SELECT correction_id
			   FROM scorer_step_correction
			  WHERE source_route=?
			  ORDER BY correction_id DESC
			  LIMIT 1`,
			[sourceRoute]
		);
		return Number(reader.getRowObjects()[0]?.correction_id);
	} finally {
		con.disconnectSync?.();
		instance.closeSync();
	}
}

async function insertRepairChildRun(dbFile, childRunId, status = 'succeeded', evidenceHashes = ['ev1', 'ev2', 'ev5']) {
	const instance = await DuckDBInstance.create(dbFile);
	const con = await instance.connect();
	try {
		await con.run(
			`INSERT INTO score_run
			 (run_id, scorer_version, indra_version, architecture, status, parent_run_id, started_at, model_id_default)
			 VALUES (?, 'repair-parity', 'test-indra', 'decomposed', ?, ?, CAST('2026-01-01 00:20:00' AS TIMESTAMP), 'smoke-local')`,
			[childRunId, status, RUN]
		);
		for (const evidenceHash of evidenceHashes) {
			await insertStep(con, {
				run_id: childRunId,
				step_hash: `child_${childRunId.slice(0, 6)}_${evidenceHash}`,
				stmt_hash: 'stmt1',
				evidence_hash: evidenceHash,
				step_kind: 'aggregate',
				output: { score: 0.88, verdict: 'correct', confidence: 'high' },
				started_at: '2026-01-01 00:20:00'
			});
		}
	} finally {
		con.disconnectSync?.();
		instance.closeSync();
	}
}

async function updateRepairChildRunStatus(dbFile, childRunId, status) {
	const instance = await DuckDBInstance.create(dbFile);
	const con = await instance.connect();
	try {
		await con.run(
			`UPDATE score_run
			    SET status=?
			  WHERE run_id=?`,
			[status, childRunId]
		);
	} finally {
		con.disconnectSync?.();
		instance.closeSync();
	}
}

async function scoreRunStatus(dbFile, runId) {
	const instance = await DuckDBInstance.create(dbFile);
	const con = await instance.connect();
	try {
		const reader = await con.runAndReadAll(
			`SELECT status
			   FROM score_run
			  WHERE run_id=?`,
			[runId]
		);
		return reader.getRowObjects()[0]?.status == null
			? null
			: String(reader.getRowObjects()[0].status);
	} finally {
		con.disconnectSync?.();
		instance.closeSync();
	}
}

async function ageRepairIntent(dbFile, childRunId, createdAt) {
	const instance = await DuckDBInstance.create(dbFile);
	const con = await instance.connect();
	try {
		await con.run(
			`UPDATE scorer_step_correction
			    SET created_at=CAST(? AS TIMESTAMP)
			  WHERE correction_kind='rerun_intent'
			    AND child_run_id=?`,
			[createdAt, childRunId]
		);
	} finally {
		con.disconnectSync?.();
		instance.closeSync();
	}
}

async function insertRecoverableIntentPageFixtures(dbFile, count) {
	const instance = await DuckDBInstance.create(dbFile);
	const con = await instance.connect();
	try {
		for (let i = 0; i < count; i += 1) {
			const childRunId = (2000 + i).toString(16).padStart(32, '0');
			const sourceDumpId = `page_recovery_${String(i).padStart(2, '0')}`;
			await con.run(
				`INSERT INTO score_run
				 (run_id, scorer_version, indra_version, architecture, status, parent_run_id, started_at, model_id_default)
				 VALUES (?, 'repair-parity', 'test-indra', 'decomposed', 'succeeded', ?, CAST(? AS TIMESTAMP), 'smoke-local')`,
				[childRunId, RUN, `2026-01-01 00:${String(21 + i).padStart(2, '0')}:00`]
			);
			await insertStep(con, {
				run_id: childRunId,
				step_hash: `page_child_${String(i).padStart(2, '0')}_ev1`,
				stmt_hash: 'stmt1',
				evidence_hash: 'ev1',
				step_kind: 'aggregate',
				output: { score: 0.81, verdict: 'correct', confidence: 'high' },
				started_at: `2026-01-01 01:${String(i).padStart(2, '0')}:00`
			});
			const sourceRoute = `/runs/${RUN}/cohort?case=recovery-page-${i}`;
			await con.run(
				`INSERT INTO scorer_step_correction
				 (step_hash, run_id, architecture, stmt_hash, evidence_hash,
				  correction_kind, status, reviewer, note, value_json,
				  source_route, source_filters_json)
				 VALUES (?, ?, 'decomposed', 'stmt1', 'ev1',
				         'repair_candidate', 'open', 'parity-test',
				         'synthetic recovery pagination candidate', ?::JSON, ?, NULL)`,
				[
					`page_parent_${String(i).padStart(2, '0')}`,
					RUN,
					JSON.stringify({
						kind: 'cohort_repair_candidate',
						grain: 'evidence',
						suspected_step_kind: 'aggregate',
						severity: 'untriaged',
						source_route: sourceRoute
					}),
					sourceRoute
				]
			);
			const reader = await con.runAndReadAll(
				`SELECT correction_id
				   FROM scorer_step_correction
				  WHERE source_route=?
				  ORDER BY correction_id DESC
				  LIMIT 1`,
				[sourceRoute]
			);
			const correctionId = Number(reader.getRowObjects()[0]?.correction_id);
			await con.run(
				`INSERT INTO scorer_step_correction
				 (step_hash, run_id, architecture, stmt_hash, evidence_hash,
				  correction_kind, status, reviewer, note, value_json,
				  parent_correction_id, child_run_id, repair_source_dump_id,
				  source_route, source_filters_json, created_at)
				 VALUES (?, ?, 'decomposed', 'stmt1', 'ev1',
				         'rerun_intent', 'open', 'viewer',
				         'synthetic recovery pagination intent', ?::JSON, ?, ?, ?, ?, NULL, CAST(? AS TIMESTAMP))`,
				[
					`page_parent_${String(i).padStart(2, '0')}`,
					RUN,
					JSON.stringify({
						kind: 'repair_rerun_intent',
						parent_correction_id: correctionId,
						parent_run_id: RUN,
						child_run_id: childRunId,
						source_dump_id: sourceDumpId,
						architecture: 'decomposed'
					}),
					correctionId,
					childRunId,
					sourceDumpId,
					`/runs/${childRunId}`,
					'2026-01-01 02:00:00'
				]
			);
		}
	} finally {
		con.disconnectSync?.();
		instance.closeSync();
	}
}

async function repairMarkerCounts(dbFile, childRunId) {
	const instance = await DuckDBInstance.create(dbFile);
	const con = await instance.connect();
	try {
		const reader = await con.runAndReadAll(
			`SELECT correction_kind, COUNT(*) AS n
			   FROM scorer_step_correction
			  WHERE correction_kind IN ('rerun_child', 'rerun_collateral')
			    AND json_extract_string(value_json, '$.child_run_id')=?
			  GROUP BY correction_kind
			  ORDER BY correction_kind`,
			[childRunId]
		);
		return Object.fromEntries(
			reader.getRowObjects().map((row) => [String(row.correction_kind), Number(row.n)])
		);
	} finally {
		con.disconnectSync?.();
		instance.closeSync();
	}
}

async function repairLineageRows(dbFile, correctionKind, childRunId) {
	const instance = await DuckDBInstance.create(dbFile);
	const con = await instance.connect();
	try {
		const reader = await con.runAndReadAll(
			`SELECT correction_kind,
			        parent_correction_id,
			        child_run_id,
			        repair_source_dump_id
			   FROM scorer_step_correction
			  WHERE correction_kind=?
			    AND child_run_id=?
			  ORDER BY correction_id`,
			[correctionKind, childRunId]
		);
		return reader.getRowObjects().map((row) => ({
			correction_kind: String(row.correction_kind),
			parent_correction_id: row.parent_correction_id == null ? null : Number(row.parent_correction_id),
			child_run_id: row.child_run_id == null ? null : String(row.child_run_id),
			repair_source_dump_id: row.repair_source_dump_id == null ? null : String(row.repair_source_dump_id)
		}));
	} finally {
		con.disconnectSync?.();
		instance.closeSync();
	}
}

function escapeRegExp(s) {
	return s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

function sorted(values) {
	return [...values].sort();
}

function assertUniqueEvidence(rows, label) {
	const ids = rows.map((r) => `${r.stmt_hash}:${r.evidence_hash ?? ''}`);
	assert.equal(new Set(ids).size, ids.length, `${label}: no duplicate statement/evidence repair candidates`);
}

const dir = await mkdtemp(join(tmpdir(), 'indra-cohort-repair-parity-'));
const dbFile = join(dir, 'repair-parity.duckdb');
process.env.VIEWER_DUCKDB_PATH = dbFile;

const instance = await DuckDBInstance.create(dbFile);
const con = await instance.connect();
try {
	await con.run(`CREATE TABLE score_run (
		run_id VARCHAR PRIMARY KEY,
		scorer_version VARCHAR NOT NULL,
		indra_version VARCHAR NOT NULL,
		architecture VARCHAR NOT NULL DEFAULT 'unknown',
		paired_run_group_id VARCHAR,
		parent_run_id VARCHAR,
		model_id_default VARCHAR,
		started_at TIMESTAMP NOT NULL,
		finished_at TIMESTAMP,
		n_stmts INTEGER,
		status VARCHAR NOT NULL,
		cost_estimate_usd DOUBLE,
		cost_actual_usd DOUBLE,
		reviewed_at TIMESTAMP,
		reviewed_by VARCHAR,
		review_status VARCHAR,
		review_notes TEXT,
		terminated_by VARCHAR,
		termination_reason TEXT,
		notes TEXT
	)`);
		await con.run(`CREATE TABLE statement (
			stmt_hash VARCHAR,
			indra_type VARCHAR,
			indra_belief DOUBLE,
			supports_count BIGINT,
			supported_by_count BIGINT,
			raw_json JSON
		)`);
	await con.run(`CREATE TABLE evidence (
		evidence_hash VARCHAR,
		stmt_hash VARCHAR,
		source_api VARCHAR,
		pmid VARCHAR,
		text VARCHAR
	)`);
	await con.run(`CREATE TABLE statement_evidence (
		stmt_hash VARCHAR,
		evidence_hash VARCHAR
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

	await con.run(`INSERT INTO score_run
		(run_id, scorer_version, indra_version, architecture, status, parent_run_id, started_at, model_id_default)
		VALUES
		('${RUN}', 'repair-parity', 'test-indra', 'decomposed', 'succeeded', NULL, CAST('2026-01-01 00:00:00' AS TIMESTAMP), 'smoke-local'),
		('${RUNNING_RUN}', 'repair-parity', 'test-indra', 'decomposed', 'running', NULL, CAST('2026-01-01 00:10:00' AS TIMESTAMP), 'smoke-local')`);
		await con.run(`INSERT INTO statement VALUES
			('stmt1', 'Activation', 0.40, 1, 0, '{"type":"Activation","matches_hash":"stmt1","evidence":[{"source_hash":"ev1"},{"source_hash":"ev2"},{"source_hash":"ev5"}]}'::JSON),
			('stmt2', 'Activation', 0.30, 0, 0, '{"type":"Activation","matches_hash":"stmt2","evidence":[{"source_hash":"ev3"}]}'::JSON),
			('stmt3', 'Inhibition', 0.70, 0, 0, '{"type":"Inhibition","matches_hash":"stmt3","evidence":[{"source_hash":"ev4"},{"source_hash":"ev_trace_snapshot"},{"source_hash":"ev_trace_error"},{"source_hash":"ev_trace_error_mixed"},{"source_hash":"ev_running_snapshot"}]}'::JSON),
			('stmt4', 'Complex', 0.55, 1, 0, '{"type":"Complex","matches_hash":"stmt4","evidence":[]}'::JSON),
			('stmt5', 'Activation', 0.25, 0, 0, '{"type":"Activation","matches_hash":"stmt5","evidence":[{"source_hash":"ev_trace_missing"},{"source_hash":"ev_probe_partial"},{"source_hash":"ev_running_missing"}]}'::JSON),
			('stmt_mixed_source', 'MixedSourceCase', 0.45, 0, 0, '{"type":"MixedSourceCase","matches_hash":"stmt_mixed_source","evidence":[{"source_hash":"ev_mixed_a"},{"source_hash":"ev_mixed_z"}]}'::JSON),
			('stmt_split_filters', 'SplitFilterCase', 0.50, 0, 0, '{"type":"SplitFilterCase","matches_hash":"stmt_split_filters","evidence":[{"source_hash":"ev_split_a"},{"source_hash":"ev_split_z"}]}'::JSON),
			('stmt_dup', 'DuplicateCase', 0.50, 0, 0, '{"type":"DuplicateCase","matches_hash":"stmt_dup","evidence":[{"source_hash":"aa_dup_less"},{"source_hash":"zz_dup_salient"}]}'::JSON)`);
	await con.run(`INSERT INTO evidence VALUES
		('ev1', 'stmt1', 'reach', NULL, 'activation evidence with native trace'),
		('ev2', 'stmt1', 'reach', NULL, 'second activation evidence'),
		('ev5', 'stmt1', 'reach', NULL, 'null-score aggregate evidence'),
		('ev3', 'stmt2', 'sparser', NULL, 'aggregate-only activation evidence'),
		('ev4', 'stmt3', 'reach', NULL, 'inhibition evidence'),
		('ev_trace_missing', 'stmt5', 'reach', NULL, 'trace missing aggregate evidence'),
		('ev_trace_snapshot', 'stmt3', 'reach', NULL, 'snapshot missing aggregate evidence'),
		('ev_trace_error', 'stmt3', 'reach', NULL, 'trace step error evidence'),
		('ev_trace_error_mixed', 'stmt3', 'reach', NULL, 'trace mixed step error evidence'),
		('ev_probe_partial', 'stmt5', 'reach', NULL, 'partial probe coverage evidence'),
		('ev_running_missing', 'stmt5', 'reach', NULL, 'running trace missing evidence'),
		('ev_running_snapshot', 'stmt3', 'reach', NULL, 'running snapshot trace evidence'),
		('ev_mixed_a', 'stmt_mixed_source', 'alpha_source', 'PMID-A', 'unscored representative-stratum evidence'),
		('ev_mixed_z', 'stmt_mixed_source', 'zz_source', 'PMID-Z', 'scored mixed-source evidence'),
		('ev_split_a', 'stmt_split_filters', 'alpha_source', 'PMID-SA', 'correct split alpha evidence'),
		('ev_split_z', 'stmt_split_filters', 'zz_source', 'PMID-SZ', 'incorrect split zz evidence'),
		('aa_dup_less', 'stmt_dup', 'dup_source', NULL, 'duplicate shared-step low-residual evidence'),
		('zz_dup_salient', 'stmt_dup', 'dup_source', NULL, 'duplicate shared-step high-residual evidence')`);
	await con.run(`INSERT INTO statement_evidence
		SELECT DISTINCT stmt_hash, evidence_hash FROM evidence`);
	await con.run(`INSERT INTO agent VALUES
		('stmt1', 'A', 'subj', 0),
		('stmt2', 'B', 'subj', 0),
		('stmt3', 'C', 'subj', 0),
		('stmt4', 'D', 'subj', 0),
		('stmt5', 'E', 'subj', 0),
		('stmt_mixed_source', 'M', 'subj', 0),
		('stmt_split_filters', 'S', 'subj', 0),
		('stmt_dup', 'F', 'subj', 0)`);
	await con.run("INSERT INTO truth_set VALUES ('gold')");
	await con.run("INSERT INTO truth_label VALUES ('gold', 'evidence', 'ev1', NULL, 'tag')");
	await con.run("INSERT INTO truth_label VALUES ('gold', 'evidence', 'ev_trace_missing', NULL, 'tag')");

	await insertStep(con, {
		step_hash: 'agg_ev1',
		stmt_hash: 'stmt1',
		evidence_hash: 'ev1',
		step_kind: 'aggregate',
		output: { score: 0.91, verdict: 'correct', confidence: 'high' }
	});
	await insertStep(con, {
		step_hash: 'agg_ev2',
		stmt_hash: 'stmt1',
		evidence_hash: 'ev2',
		step_kind: 'aggregate',
		output: { score: 0.20, verdict: 'incorrect', confidence: 'low' }
	});
	await insertStep(con, {
		step_hash: 'agg_ev3',
		stmt_hash: 'stmt2',
		evidence_hash: 'ev3',
		step_kind: 'aggregate',
		output: { score: 0.63, verdict: 'correct', confidence: 'high' }
	});
	await insertStep(con, {
		step_hash: 'agg_ev4',
		stmt_hash: 'stmt3',
		evidence_hash: 'ev4',
		step_kind: 'aggregate',
		output: { score: 0.50, verdict: 'abstain', confidence: 'medium' }
	});
	await insertStep(con, {
		step_hash: 'agg_mixed_z',
		stmt_hash: 'stmt_mixed_source',
		evidence_hash: 'ev_mixed_z',
		step_kind: 'aggregate',
		output: { score: 0.80, verdict: 'correct', confidence: 'high' }
	});
	await insertStep(con, {
		step_hash: 'agg_split_a',
		stmt_hash: 'stmt_split_filters',
		evidence_hash: 'ev_split_a',
		step_kind: 'aggregate',
		output: { score: 0.70, verdict: 'correct', confidence: 'high' }
	});
	await insertStep(con, {
		step_hash: 'agg_split_z',
		stmt_hash: 'stmt_split_filters',
		evidence_hash: 'ev_split_z',
		step_kind: 'aggregate',
		output: { score: 0.20, verdict: 'incorrect', confidence: 'low' }
	});
	await insertStep(con, {
		step_hash: 'agg_ev5_no_score',
		stmt_hash: 'stmt1',
		evidence_hash: 'ev5',
		step_kind: 'aggregate',
		output: { verdict: 'correct', confidence: 'high' }
	});
	// Same-step fanout is intentionally malformed at the scorer_step layer.
	// It protects repair materialization against duplicated selection rows from
	// joins or fixture drift without relying on row-by-row insertion semantics.
	await insertStep(con, {
		step_hash: 'agg_dup_shared',
		stmt_hash: 'stmt_dup',
		evidence_hash: 'aa_dup_less',
		step_kind: 'aggregate',
		output: { score: 0.55, verdict: 'incorrect', confidence: 'medium' }
	});
	await insertStep(con, {
		step_hash: 'agg_dup_shared',
		stmt_hash: 'stmt_dup',
		evidence_hash: 'zz_dup_salient',
		step_kind: 'aggregate',
		output: { score: 0.95, verdict: 'incorrect', confidence: 'high' }
	});
	await insertStep(con, {
		step_hash: 'parse_ev1',
		stmt_hash: 'stmt1',
		evidence_hash: 'ev1',
		step_kind: 'parse_claim',
		output: { parsed: true }
	});
	await insertStep(con, {
		step_hash: 'parse_missing',
		stmt_hash: 'stmt5',
		evidence_hash: 'ev_trace_missing',
		step_kind: 'parse_claim',
		output: { parsed: true },
		started_at: '2026-01-01 00:01:00'
	});
	await insertStep(con, {
		step_hash: 'parse_snapshot',
		stmt_hash: 'stmt3',
		evidence_hash: 'ev_trace_snapshot',
		step_kind: 'parse_claim',
		output: { parsed: true },
		started_at: '2026-01-01 00:02:00'
	});
	await insertStep(con, {
		step_hash: 'agg_snapshot_late',
		stmt_hash: 'stmt3',
		evidence_hash: 'ev_trace_snapshot',
		step_kind: 'aggregate',
		output: { score: 0.35, verdict: 'incorrect', confidence: 'low' },
		started_at: '2026-01-01 00:10:00'
	});
	await insertStep(con, {
		step_hash: 'build_error',
		stmt_hash: 'stmt3',
		evidence_hash: 'ev_trace_error',
		step_kind: 'build_context',
		output: { error: 'context build failed' },
		error: 'context build failed',
		started_at: '2026-01-01 00:03:00'
	});
	await insertStep(con, {
		step_hash: 'parse_error_mixed',
		stmt_hash: 'stmt3',
		evidence_hash: 'ev_trace_error_mixed',
		step_kind: 'parse_claim',
		output: { error: 'parse failed' },
		error: 'parse failed',
		started_at: '2026-01-01 00:04:00'
	});
	await insertStep(con, {
		step_hash: 'build_healthy_mixed',
		stmt_hash: 'stmt3',
		evidence_hash: 'ev_trace_error_mixed',
		step_kind: 'build_context',
		output: { built: true },
		started_at: '2026-01-01 00:05:00'
	});
	await insertStep(con, {
		step_hash: 'route_probe_partial',
		stmt_hash: 'stmt5',
		evidence_hash: 'ev_probe_partial',
		step_kind: 'substrate_route',
		output: {
			subject_role: { source: 'substrate', answer: 'yes' },
			object_role: { source: 'needs_llm', answer: null },
			relation_axis: { source: 'needs_llm', answer: null },
			scope: { source: null, answer: null }
		},
		started_at: '2026-01-01 00:06:00'
	});
	await insertStep(con, {
		step_hash: 'subject_probe_partial',
		stmt_hash: 'stmt5',
		evidence_hash: 'ev_probe_partial',
		step_kind: 'subject_role_probe',
		output: { source: 'substrate', answer: 'yes' },
		started_at: '2026-01-01 00:07:00'
	});
	await insertStep(con, {
		run_id: RUNNING_RUN,
		step_hash: 'parse_running_missing',
		stmt_hash: 'stmt5',
		evidence_hash: 'ev_running_missing',
		step_kind: 'parse_claim',
		output: { parsed: true },
		started_at: '2026-01-01 00:01:00'
	});
	await insertStep(con, {
		run_id: RUNNING_RUN,
		step_hash: 'parse_running_snapshot',
		stmt_hash: 'stmt3',
		evidence_hash: 'ev_running_snapshot',
		step_kind: 'parse_claim',
		output: { parsed: true },
		started_at: '2026-01-01 00:02:00'
	});
	await insertStep(con, {
		run_id: RUNNING_RUN,
		step_hash: 'agg_running_late',
		stmt_hash: 'stmt3',
		evidence_hash: 'ev_running_snapshot',
		step_kind: 'aggregate',
		output: { score: 0.31, verdict: 'incorrect', confidence: 'low' },
		started_at: '2026-01-01 00:10:00'
	});
	for (let i = 0; i < 501; i += 1) {
		const evidenceHash = `ev_many_${String(i).padStart(3, '0')}`;
		await con.run(
			'INSERT INTO evidence VALUES (?, ?, ?, NULL, ?)',
			[evidenceHash, 'stmt4', 'reach', `complex evidence ${i}`]
		);
		await insertStep(con, {
			step_hash: `agg_many_${String(i).padStart(3, '0')}`,
			stmt_hash: 'stmt4',
			evidence_hash: evidenceHash,
			step_kind: 'aggregate',
			output: { score: 0.40 + (i % 20) / 100, verdict: 'correct', confidence: 'high' }
		});
	}
} finally {
	con.disconnectSync?.();
	instance.closeSync();
}

const server = await createServer({
	configFile: './vite.config.ts',
	server: { middlewareMode: true, hmr: false },
	logLevel: 'error'
});
const httpServer = createHttpServer(server.middlewares);
await new Promise((resolve, reject) => {
	httpServer.once('error', reject);
	httpServer.listen(0, '127.0.0.1', () => {
		httpServer.off('error', reject);
		resolve();
	});
});
const httpAddress = httpServer.address();
const baseUrl = `http://127.0.0.1:${httpAddress.port}`;

try {
	const {
		getRunCohort,
		getRunRepairBacklog,
		getRepairRerunDetail,
		closeInstance
	} = await server.ssrLoadModule('/src/lib/db.ts');
		const {
			estimateRepairCohort,
			materializeRepairCohort,
			RepairCohortConflictError,
			RepairCohortHttpError,
			RepairCohortInputError
		} = await server.ssrLoadModule('/src/lib/server/repairBacklog.ts');
			const {
				exportRepairRerunCorpus,
				recordRepairRerunIntent,
				recordRepairRerunChildAfterScore,
				recordRepairRerunUncovered,
				tombstoneStaleRepairRerunChild
			} = await server.ssrLoadModule('/src/lib/server/repairRerun.ts');
	const {
		acquireWriterLock,
		clearWriterLockToken,
		createPairedWorkflowState
	} = await server.ssrLoadModule('/src/lib/server/pairedState.ts');
		const { POST: postRepairCohort } = await server.ssrLoadModule('/src/routes/api/repairs/cohort/+server.ts');
		const { POST: postRepairEstimate } = await server.ssrLoadModule('/src/routes/api/repairs/cohort/estimate/+server.ts');
		const { POST: postProbeSlotReview } = await server.ssrLoadModule('/src/routes/api/repairs/probe-slots/+server.ts');
		const { GET: getWriterLock } = await server.ssrLoadModule('/src/routes/api/writer-lock/+server.ts');
		const { load: loadDashboard } = await server.ssrLoadModule('/src/routes/+page.server.ts');

	async function postRaw(body) {
		const response = await postRepairCohort({
			request: new Request('http://localhost/api/repairs/cohort', {
				method: 'POST',
				headers: { 'content-type': 'application/json' },
				body
			})
		});
		return {
			status: response.status,
			body: await response.json()
		};
	}

		async function postRepair(body) {
			return postRaw(JSON.stringify(body));
		}

		async function postEstimateRaw(body) {
			const response = await postRepairEstimate({
				request: new Request('http://localhost/api/repairs/cohort/estimate', {
					method: 'POST',
					headers: { 'content-type': 'application/json' },
					body
				})
			});
			return {
				status: response.status,
				body: await response.json()
			};
		}

		async function postProbeSlots(body) {
			const response = await postProbeSlotReview({
				request: new Request('http://localhost/api/repairs/probe-slots', {
					method: 'POST',
					headers: { 'content-type': 'application/json' },
					body: JSON.stringify(body)
				})
			});
			return {
				status: response.status,
				body: await response.json()
			};
		}

	async function getWriterLockSnapshot() {
		const response = await getWriterLock({});
		return {
			status: response.status,
			body: await response.json()
		};
	}

	async function loadDashboardSnapshot() {
		return await loadDashboard({
			url: new URL('http://localhost/')
		});
	}

	async function pageHtml(path) {
		const { status, body } = await pageResponse(path);
		assert.equal(status, 200);
		return body;
	}

	async function pageResponse(path, options = {}) {
		const ctrl = new AbortController();
		const timeout = setTimeout(() => ctrl.abort(), 5000);
		const response = await fetch(`${baseUrl}${path}`, {
			headers: { connection: 'close' },
			redirect: options.redirect ?? 'follow',
			signal: ctrl.signal
		});
		clearTimeout(timeout);
		const body = await response.text();
		return { status: response.status, body, location: response.headers.get('location'), url: response.url };
	}

	async function repairPageHtml(runId = RUN) {
		return pageHtml(`/runs/${runId}/repairs`);
	}

	async function dashboardPageHtml() {
		return pageHtml('/');
	}

	function assertPublicWriterLockShape(lock) {
		assert.deepEqual(
			Object.keys(lock).sort(),
			['architecture', 'kind', 'label', 'malformed_reason', 'model', 'pair_id', 'pid', 'started_at', 'updated_at'].sort()
		);
	}

	function sectionByAriaLabel(html, label) {
		const pattern = new RegExp(`<section\\b[^>]*aria-label="${escapeRegExp(label)}"[\\s\\S]*?<\\/section>`);
		const match = html.match(pattern);
		assert.ok(match, `rendered page exposes section ${label}`);
		return match[0];
	}

	function repairActiveCancelButton(html) {
		const buttons = [...html.matchAll(/<button\b[^>]*>[\s\S]*?<\/button>/g)].map((m) => m[0]);
		const matchingButtons = buttons.filter((b) => /\saria-label="mark repair child [^"]+ canceled"/.test(b));
		assert.equal(matchingButtons.length, 1, 'rendered repairs page exposes exactly one active child cancel button');
		const button = matchingButtons[0];
		return button;
	}

	function buttonByAriaLabel(html, label) {
		const buttons = [...html.matchAll(/<button\b[^>]*>[\s\S]*?<\/button>/g)].map((m) => m[0]);
		const matchingButtons = buttons.filter((b) => b.includes(`aria-label="${label}"`));
		assert.equal(matchingButtons.length, 1, `rendered page exposes exactly one ${label} button`);
		return matchingButtons[0];
	}

	function buttonIsDisabled(button) {
		const openTag = button.match(/^<button\b[^>]*>/)?.[0] ?? '';
		return /\sdisabled(?:[=\s>]|$)/.test(openTag);
	}

	function repairWarningText(html, contains) {
		const paragraphs = [...html.matchAll(/<p class="repair-warning[^"]*">([\s\S]*?)<\/p>/g)]
			.map((m) => m[1]
				.replace(/<!--[\s\S]*?-->/g, '')
				.replace(/<[^>]*>/g, ' ')
				.replace(/\s+/g, ' ')
				.trim());
		const paragraph = paragraphs.find((p) => p.includes(contains));
		assert.ok(paragraph, `rendered repairs page has warning text containing ${contains}`);
		return paragraph;
	}

		async function postEstimate(body) {
			return postEstimateRaw(JSON.stringify(body));
		}

	async function assertRepeatSkipped(name, {
		run_id = RUN,
		filters,
		sourceRoute,
		expectedRunStatus,
		expectedInspected,
		expectedCorrections,
		expectedDuplicateSelection = 0
		}) {
			const repeatEstimate = await estimateRepairCohort({
				run_id,
				filters,
				source_route: sourceRoute,
				expected_run_status: expectedRunStatus,
				reviewer: 'parity-test'
			});
			assert.equal(repeatEstimate.inspected, expectedInspected, `${name}: estimate still inspects the visible cohort`);
			assert.equal(
				repeatEstimate.would_create,
				0,
				`${name}: estimate reports no new candidates after repeat route is already open`
			);
			assert.equal(
				repeatEstimate.skipped_existing,
				expectedInspected - expectedDuplicateSelection,
				`${name}: estimate reports existing open candidates before repeat write`
			);
			assert.equal(
				repeatEstimate.skipped_duplicate_selection,
				expectedDuplicateSelection,
				`${name}: estimate reports collapsed in-selection duplicates before repeat write`
			);
			const repeat = await materializeRepairCohort({
				run_id,
				filters,
				source_route: sourceRoute,
			expected_run_status: expectedRunStatus,
			reviewer: 'parity-test'
		});
		assert.equal(repeat.inspected, expectedInspected, `${name}: repeat still inspects the visible cohort`);
		assert.equal(repeat.created, 0, `${name}: repeat creates no duplicate candidates`);
		assert.equal(
			repeat.skipped_existing,
			expectedInspected - expectedDuplicateSelection,
			`${name}: repeat skips existing candidates in bulk`
		);
		assert.equal(
			repeat.skipped_duplicate_selection,
			expectedDuplicateSelection,
			`${name}: repeat reports collapsed in-selection duplicates separately`
		);
		assert.equal(
			(await correctionRows(dbFile, sourceRoute)).length,
			expectedCorrections,
			`${name}: repeat leaves the append-only correction set unchanged`
		);
	}

	async function assertEvidenceParity(name, filters) {
		const cohort = await getRunCohort(RUN, filters);
		const sourceRoute = `/runs/${RUN}/cohort?case=${name}`;
		const result = await materializeRepairCohort({
			run_id: RUN,
			filters,
			source_route: sourceRoute,
			reviewer: 'parity-test'
		});
		const corrections = await correctionRows(dbFile, sourceRoute);
		assert.equal(result.inspected, cohort.rows.length, `${name}: inspected count follows display rows`);
		assert.equal(result.created, corrections.length, `${name}: created count follows inserted corrections`);
		assert.equal(result.skipped_existing, 0, `${name}: fresh route has no existing repair skips`);
		assert.equal(result.skipped_duplicate_selection, 0, `${name}: fresh route has no collapsed duplicates`);
		assertUniqueEvidence(corrections, name);
		assert.deepEqual(
			sorted(corrections.map((r) => r.evidence_hash)),
			sorted(cohort.rows.map((r) => r.evidence_hash)),
			`${name}: repair evidence set follows display evidence set`
		);
		await assertRepeatSkipped(name, {
			filters,
			sourceRoute,
			expectedInspected: cohort.rows.length,
			expectedCorrections: corrections.length
		});
	}

	async function assertTraceParity(name, filters, options = {}) {
		const runId = options.run_id ?? RUN;
		const expectedRunStatus = options.expected_run_status ?? 'succeeded';
		const expectedSuspectedByEvidence = options.expected_suspected_by_evidence ?? {};
		const cohort = await getRunCohort(runId, filters);
		const sourceRoute = `/runs/${runId}/cohort?case=${name}`;
		const result = await materializeRepairCohort({
			run_id: runId,
			filters,
			source_route: sourceRoute,
			expected_run_status: expectedRunStatus,
			reviewer: 'parity-test'
		});
		const corrections = await correctionRows(dbFile, sourceRoute);
		assert.equal(result.grain, 'evidence', `${name}: trace repair canonicalizes to evidence grain`);
		assert.equal(result.inspected, cohort.rows.length, `${name}: inspected count follows display rows`);
		assert.equal(result.created, corrections.length, `${name}: created count follows inserted trace corrections`);
		assert.equal(result.skipped_existing, 0, `${name}: fresh trace route has no existing repair skips`);
		assert.equal(result.skipped_duplicate_selection, 0, `${name}: fresh trace route has no collapsed duplicates`);
		assertUniqueEvidence(corrections, name);
		assert.deepEqual(
			sorted(corrections.map((r) => r.evidence_hash)),
			sorted(cohort.rows.map((r) => r.evidence_hash)),
			`${name}: repair evidence set follows display trace evidence set`
		);
		assert.deepEqual(
			[...new Set(corrections.map((r) => r.trace_state))],
			[filters.trace_state],
			`${name}: repair provenance preserves the visible trace_state`
		);
		for (const row of corrections) {
			if (filters.trace_state === 'step_error') {
				assert.ok(
					Object.hasOwn(expectedSuspectedByEvidence, row.evidence_hash),
					`${name}: step_error expected suspected step must be explicit for ${row.evidence_hash}`
				);
			}
			const expectedSuspectedStep = expectedSuspectedByEvidence[row.evidence_hash] ?? filters.trace_state;
			assert.equal(
				row.suspected_step_kind,
				expectedSuspectedStep,
				`${name}: suspected step follows the trace repair provenance contract for ${row.evidence_hash}`
			);
		}
		await assertRepeatSkipped(name, {
			run_id: runId,
			filters,
			sourceRoute,
			expectedRunStatus,
			expectedInspected: cohort.rows.length,
			expectedCorrections: corrections.length
		});
	}

	async function assertProbeCoverageParity() {
		const filters = { probe_coverage: 'present' };
		const cohort = await getRunCohort(RUN, filters);
		const sourceRoute = `/runs/${RUN}/cohort?case=probe-coverage-repair`;
		assert.deepEqual(
			sorted(cohort.rows.map((r) => r.evidence_hash)),
			['ev_probe_partial'],
			'probe coverage cohort fixture selects the persisted probe-covered evidence'
		);
		const estimate = await estimateRepairCohort({
			run_id: RUN,
			filters,
			source_route: sourceRoute,
			expected_run_status: 'succeeded',
			reviewer: 'parity-test'
		});
		assert.equal(estimate.grain, 'evidence', 'probe coverage repair estimates evidence grain');
		assert.equal(estimate.inspected, cohort.rows.length, 'probe coverage estimate follows display rows');
		assert.equal(estimate.would_create, 1, 'probe coverage estimate creates one candidate');
		const result = await materializeRepairCohort({
			run_id: RUN,
			filters,
			source_route: sourceRoute,
			expected_run_status: 'succeeded',
			reviewer: 'parity-test',
			estimate_token: estimate.estimate_token,
			require_estimate_token: true
		});
		const corrections = await correctionRows(dbFile, sourceRoute);
		assert.equal(result.grain, 'evidence', 'probe coverage repair materializes evidence grain');
		assert.equal(result.inspected, cohort.rows.length, 'probe coverage materialization follows display rows');
		assert.equal(result.created, 1, 'probe coverage materializes one repair candidate');
		assert.equal(corrections.length, 1, 'probe coverage correction row is persisted append-only');
		const row = corrections[0];
		assert.equal(row.evidence_hash, 'ev_probe_partial');
		assert.equal(row.trace_state, 'missing_aggregate');
		assert.equal(row.probe_coverage, 'present');
		assert.equal(row.suspected_step_kind, 'probe_coverage');
		assert.match(row.missing_probe_slots, /object_role_probe/);
		assert.match(row.missing_probe_slots, /relation_axis_probe/);
		assert.match(row.missing_probe_slots, /scope_probe/);
		assert.doesNotMatch(row.missing_probe_slots, /subject_role_probe/);
		const backlog = await getRunRepairBacklog(RUN);
		const backlogRow = backlog.rows.find((r) => r.correction_id === row.correction_id);
		assert.ok(backlogRow, 'probe coverage repair candidate appears in the backlog');
		assert.equal(backlogRow.probe_coverage, 'present');
		assert.equal(backlogRow.suspected_step_kind, 'probe_coverage');
		assert.equal(backlogRow.n_substrate_route, 1);
		assert.equal(backlogRow.n_subject_role_probe, 1);
		assert.equal(backlogRow.n_object_role_probe, 0);
		assert.equal(backlogRow.n_relation_axis_probe, 0);
		assert.equal(backlogRow.n_scope_probe, 0);
		const repairsHtml = await repairPageHtml(RUN);
		assert.match(repairsHtml, /probe repair facts/);
		assert.match(repairsHtml, /probe coverage/);
		assert.match(repairsHtml, /object_role_probe/);
		assert.match(repairsHtml, /relation_axis_probe/);
		assert.match(repairsHtml, /scope_probe/);
		assert.match(repairsHtml, /subject_role_probe<\/code>\s*1/);
		assert.match(repairsHtml, /probe slot review/);
		assert.match(repairsHtml, /record slot review/);
		const invalidSlotReview = await postProbeSlots({
			run_id: RUN,
			correction_id: row.correction_id,
			selected_slots: ['subject_role_probe'],
			note: 'should not target a present slot'
		});
		assert.equal(invalidSlotReview.status, 400);
		assert.match(invalidSlotReview.body.message, /not missing/);
		const slotReview = await postProbeSlots({
			run_id: RUN,
			correction_id: row.correction_id,
			selected_slots: ['object_role_probe', 'scope_probe'],
			note: 'target object and scope probes'
		});
		assert.equal(slotReview.status, 200);
		assert.equal(slotReview.body.recorded, 1);
		assert.deepEqual(slotReview.body.selected_slots, ['object_role_probe', 'scope_probe']);
		const slotReviewRows = await probeSlotReviewRows(dbFile, row.correction_id);
		assert.equal(slotReviewRows.length, 1, 'probe slot review is append-only');
		assert.equal(slotReviewRows[0].status, 'recorded');
		assert.equal(slotReviewRows[0].parent_correction_id, row.correction_id);
		assert.equal(slotReviewRows[0].selected_probe_slots, 'object_role_probe,scope_probe');
		assert.equal(slotReviewRows[0].note, 'target object and scope probes');
		const reviewedBacklog = await getRunRepairBacklog(RUN);
		const reviewedBacklogRow = reviewedBacklog.rows.find((r) => r.correction_id === row.correction_id);
		assert.ok(reviewedBacklogRow, 'probe slot reviewed candidate remains visible in backlog');
		assert.equal(reviewedBacklogRow.probe_slot_review_count, 1);
		assert.equal(reviewedBacklogRow.probe_slot_review_slots, 'object_role_probe,scope_probe');
		assert.equal(reviewedBacklogRow.probe_slot_review_note, 'target object and scope probes');
		const reviewedRepairsHtml = await repairPageHtml(RUN);
		assert.match(reviewedRepairsHtml, /target object and scope probes/);
		assert.match(reviewedRepairsHtml, /object_role_probe, scope_probe/);
		const probeRerunCorpus = await exportRepairRerunCorpus({
			run_id: RUN,
			correction_ids: [row.correction_id]
		});
		assert.equal(probeRerunCorpus.n_evidences, 3);
		assert.equal(probeRerunCorpus.n_raw_json_evidences, 3);
		assert.equal(probeRerunCorpus.n_table_evidences, 3);
		assert.equal(probeRerunCorpus.evidence_count_validated, true);
		assert.equal(
			probeRerunCorpus.n_probe_slot_reviewed_candidates,
			1,
			'probe slot review is counted in repair rerun corpus metadata'
		);
		assert.deepEqual(
			probeRerunCorpus.probe_slot_reviews.map((r) => ({
				correction_id: r.correction_id,
				selected_slots: r.selected_slots,
				note: r.note,
				review_count: r.review_count
			})),
			[{
				correction_id: row.correction_id,
				selected_slots: ['object_role_probe', 'scope_probe'],
				note: 'target object and scope probes',
				review_count: 1
			}],
			'latest probe slot review travels with rerun corpus metadata'
		);
		assert.deepEqual(
			probeRerunCorpus.probe_slot_counts,
			{ object_role_probe: 1, scope_probe: 1 },
			'probe slot review counts aggregate selected slots for preflight display'
		);
		const probeRerunMeta = JSON.parse(
			await readFile(join(dir, 'repair_reruns', `${probeRerunCorpus.source_dump_id}.meta.json`), 'utf8')
		);
		assert.equal(probeRerunMeta.n_raw_json_evidences, 3);
		assert.equal(probeRerunMeta.n_table_evidences, 3);
		assert.equal(probeRerunMeta.evidence_count_validated, true);
		assert.equal(probeRerunMeta.n_probe_slot_reviewed_candidates, 1);
		assert.deepEqual(probeRerunMeta.probe_slot_counts, { object_role_probe: 1, scope_probe: 1 });
		assert.deepEqual(probeRerunMeta.probe_slot_reviews[0].selected_slots, ['object_role_probe', 'scope_probe']);
		assert.equal(probeRerunMeta.scoring_mode, 'probe_only');
		assert.deepEqual(probeRerunMeta.probe_step_filter, ['object_role_probe', 'scope_probe']);
		await assertRepeatSkipped('probe-coverage-repair', {
			filters,
			sourceRoute,
			expectedRunStatus: 'succeeded',
			expectedInspected: cohort.rows.length,
			expectedCorrections: corrections.length
		});
	}

	await assertEvidenceParity('evidence-filtered', {
		type: 'Activation',
		source: 'reach',
		verdict: 'correct'
	});
	await assertEvidenceParity('native-decomposed', {
		trace_fidelity: 'native_decomposed'
	});
	await assertEvidenceParity('aggregate-only', {
		trace_fidelity: 'aggregate_only',
		verdict: 'correct'
	});

	const zeroCohort = await getRunCohort(RUN, { step_kind: 'parse_claim' });
	const zeroRoute = `/runs/${RUN}/cohort?case=non-applicable-step`;
	const zeroResult = await materializeRepairCohort({
		run_id: RUN,
		filters: { step_kind: 'parse_claim' },
		source_route: zeroRoute,
		reviewer: 'parity-test'
	});
	assert.equal(zeroCohort.totalRows, 0);
	assert.equal(zeroResult.inspected, 0);
	assert.equal(zeroResult.created, 0);
	assert.equal(zeroResult.skipped_existing, 0);
	assert.equal(zeroResult.skipped_duplicate_selection, 0);
	assert.deepEqual(await correctionRows(dbFile, zeroRoute), []);

		const duplicateFilters = {
			type: 'DuplicateCase'
		};
		const duplicateCohort = await getRunCohort(RUN, duplicateFilters);
		const duplicateRoute = `/runs/${RUN}/cohort?case=duplicate-selection`;
		const duplicateEstimate = await estimateRepairCohort({
			run_id: RUN,
			filters: duplicateFilters,
			source_route: duplicateRoute,
			reviewer: 'parity-test'
		});
		assert.equal(duplicateEstimate.inspected, 2);
		assert.equal(duplicateEstimate.unique_selected, 1);
		assert.equal(duplicateEstimate.would_create, 1);
		assert.equal(duplicateEstimate.skipped_existing, 0);
		assert.equal(duplicateEstimate.skipped_duplicate_selection, 1);
		const duplicateResult = await materializeRepairCohort({
			run_id: RUN,
			filters: duplicateFilters,
		source_route: duplicateRoute,
		reviewer: 'parity-test'
	});
	const duplicateCorrections = await correctionRows(dbFile, duplicateRoute);
	assert.equal(duplicateCohort.rows.length, 2);
	assert.equal(duplicateResult.inspected, 2);
	assert.equal(duplicateResult.created, 1);
	assert.equal(duplicateResult.skipped_existing, 0);
	assert.equal(duplicateResult.skipped_duplicate_selection, 1);
	assert.deepEqual(duplicateCorrections.map((r) => r.evidence_hash), ['zz_dup_salient']);
	await assertRepeatSkipped('duplicate-selection', {
		filters: duplicateFilters,
		sourceRoute: duplicateRoute,
		expectedInspected: 2,
		expectedCorrections: duplicateCorrections.length,
		expectedDuplicateSelection: 1
	});

		const duplicateStatementFilters = {
			grain: 'statement',
			type: 'DuplicateCase'
		};
		const duplicateStatementCohort = await getRunCohort(RUN, duplicateStatementFilters);
		const duplicateStatementRoute = `/runs/${RUN}/cohort?case=duplicate-statement-selection`;
		const duplicateStatementEstimate = await estimateRepairCohort({
			run_id: RUN,
			filters: duplicateStatementFilters,
			source_route: duplicateStatementRoute,
			reviewer: 'parity-test'
		});
		assert.equal(duplicateStatementEstimate.inspected, 2);
		assert.equal(duplicateStatementEstimate.unique_selected, 1);
		assert.equal(duplicateStatementEstimate.would_create, 1);
		assert.equal(duplicateStatementEstimate.skipped_existing, 0);
		assert.equal(duplicateStatementEstimate.skipped_duplicate_selection, 1);
		const duplicateStatementResult = await materializeRepairCohort({
			run_id: RUN,
			filters: duplicateStatementFilters,
		source_route: duplicateStatementRoute,
		reviewer: 'parity-test'
	});
	const duplicateStatementCorrections = await correctionRows(dbFile, duplicateStatementRoute);
	assert.deepEqual(duplicateStatementCohort.rows.map((r) => r.stmt_hash), ['stmt_dup']);
	assert.equal(duplicateStatementResult.inspected, 2);
	assert.equal(duplicateStatementResult.created, 1);
	assert.equal(duplicateStatementResult.skipped_existing, 0);
	assert.equal(duplicateStatementResult.skipped_duplicate_selection, 1);
	assert.deepEqual(duplicateStatementCorrections.map((r) => r.evidence_hash), ['zz_dup_salient']);
	await assertRepeatSkipped('duplicate-statement-selection', {
		filters: duplicateStatementFilters,
		sourceRoute: duplicateStatementRoute,
		expectedInspected: 2,
		expectedCorrections: duplicateStatementCorrections.length,
		expectedDuplicateSelection: 1
	});

	const statementFilters = {
		grain: 'statement',
		type: 'Activation',
		multi_evidence: true,
		supports: true
	};
	const statementCohort = await getRunCohort(RUN, statementFilters);
	const statementRoute = `/runs/${RUN}/cohort?case=statement-filtered`;
	const statementResult = await materializeRepairCohort({
		run_id: RUN,
		filters: statementFilters,
		source_route: statementRoute,
		reviewer: 'parity-test'
	});
	const statementCorrections = await correctionRows(dbFile, statementRoute);
	assert.deepEqual(statementCohort.rows.map((r) => r.stmt_hash), ['stmt1']);
	assert.equal(statementResult.inspected, 2);
	assert.equal(statementResult.created, statementCorrections.length);
	assert.equal(statementResult.skipped_existing, 0);
	assert.equal(statementResult.skipped_duplicate_selection, 0);
		assert.deepEqual([...new Set(statementCorrections.map((r) => r.stmt_hash))], ['stmt1']);
		assert.deepEqual(sorted(statementCorrections.map((r) => r.evidence_hash)), ['ev1', 'ev2']);
		assertUniqueEvidence(statementCorrections, 'statement-filtered');
		const statementRerunCorpus = await exportRepairRerunCorpus({
			run_id: RUN,
			correction_ids: statementCorrections.map((r) => r.correction_id)
		});
		assert.equal(statementRerunCorpus.n_candidates, 2);
		assert.equal(statementRerunCorpus.n_statements, 1);
		assert.equal(statementRerunCorpus.n_evidences, 3);
		assert.equal(statementRerunCorpus.n_raw_json_evidences, 3);
		assert.equal(statementRerunCorpus.n_table_evidences, 3);
		assert.equal(statementRerunCorpus.evidence_count_validated, true);
		const statementRerunMeta = JSON.parse(
			await readFile(join(dir, 'repair_reruns', `${statementRerunCorpus.source_dump_id}.meta.json`), 'utf8')
		);
		assert.equal(statementRerunMeta.n_raw_json_evidences, 3);
		assert.equal(statementRerunMeta.n_table_evidences, 3);
		assert.equal(statementRerunMeta.evidence_count_validated, true);
		const originalStmt1RawJson = await statementRawJson(dbFile, 'stmt1');
		try {
			await updateStatementRawJson(
				dbFile,
				'stmt1',
				'{"type":"Activation","matches_hash":"stmt1","evidence":[{"source_hash":"ev1"}]}'
			);
			await assert.rejects(
				() => exportRepairRerunCorpus({
					run_id: RUN,
					correction_ids: statementCorrections.map((r) => r.correction_id)
				}),
				/evidence denominator mismatch.*stmt1.*raw_json has 1 evidence row.*normalized evidence rows have 3/
			);
		} finally {
			await updateStatementRawJson(dbFile, 'stmt1', originalStmt1RawJson);
		}
			assert.equal(statementRerunCorpus.n_selected_evidence_candidates, 2);
			assert.equal(statementRerunCorpus.n_statement_scope_candidates, 0);
			assert.equal(statementRerunCorpus.n_scope_expansion_evidences, 1);
			assert.equal(statementRerunCorpus.n_collateral_evidences, 1);
			const legacyStatementScopeRoute = `/runs/${RUN}/cohort?case=legacy-statement-scope-rerun`;
			const legacyStatementScopeCorrectionId = await insertStatementScopeCorrection(
				dbFile,
				legacyStatementScopeRoute
			);
			const legacyStatementScopeRerunCorpus = await exportRepairRerunCorpus({
				run_id: RUN,
				correction_ids: [legacyStatementScopeCorrectionId]
			});
			assert.equal(legacyStatementScopeRerunCorpus.n_candidates, 1);
			assert.equal(legacyStatementScopeRerunCorpus.n_statements, 1);
			assert.equal(legacyStatementScopeRerunCorpus.n_evidences, 3);
			assert.equal(legacyStatementScopeRerunCorpus.n_selected_evidence_candidates, 0);
			assert.equal(legacyStatementScopeRerunCorpus.n_statement_scope_candidates, 1);
			assert.equal(legacyStatementScopeRerunCorpus.n_scope_expansion_evidences, 0);
			assert.equal(legacyStatementScopeRerunCorpus.n_collateral_evidences, 3);
			const mixedStatementScopeRerunCorpus = await exportRepairRerunCorpus({
				run_id: RUN,
				correction_ids: [
					legacyStatementScopeCorrectionId,
					...statementCorrections.map((r) => r.correction_id)
				]
			});
			assert.equal(mixedStatementScopeRerunCorpus.n_candidates, 3);
			assert.equal(mixedStatementScopeRerunCorpus.n_statements, 1);
			assert.equal(mixedStatementScopeRerunCorpus.n_evidences, 3);
			assert.equal(mixedStatementScopeRerunCorpus.n_selected_evidence_candidates, 2);
			assert.equal(mixedStatementScopeRerunCorpus.n_statement_scope_candidates, 1);
			assert.equal(mixedStatementScopeRerunCorpus.n_scope_expansion_evidences, 0);
			assert.equal(mixedStatementScopeRerunCorpus.n_collateral_evidences, 1);
			const repairChildRun = '11111111111111111111111111111111';
			await insertRepairChildRun(dbFile, repairChildRun);
			await assert.rejects(
				() => recordRepairRerunIntent({
					parent_run_id: RUN,
					child_run_id: '14141414141414141414141414141414',
					architecture: 'decomposed',
					source_dump_id: statementRerunCorpus.source_dump_id,
					correction_ids: statementCorrections.map((r) => r.correction_id),
					path: `${statementRerunCorpus.path}.wrong`
				}),
				/corpus path/
			);
			const repairIntent = await recordRepairRerunIntent({
				parent_run_id: RUN,
				child_run_id: repairChildRun,
				architecture: 'decomposed',
				source_dump_id: statementRerunCorpus.source_dump_id,
				correction_ids: statementCorrections.map((r) => r.correction_id),
				path: statementRerunCorpus.path
			});
			assert.equal(repairIntent.recorded, 2);
			const intentLineageRows = await repairLineageRows(dbFile, 'rerun_intent', repairChildRun);
			assert.deepEqual(
				intentLineageRows.map((row) => row.parent_correction_id),
				statementCorrections.map((r) => r.correction_id).sort((a, b) => a - b)
			);
			assert.ok(intentLineageRows.every((row) => row.child_run_id === repairChildRun));
			assert.ok(intentLineageRows.every((row) => row.repair_source_dump_id === statementRerunCorpus.source_dump_id));
			const intentBacklog = await getRunRepairBacklog(RUN);
			const recoverableIntent = intentBacklog.recoverableRerunIntents.find(
				(row) => row.child_run_id === repairChildRun
			);
			assert.ok(recoverableIntent, 'succeeded rerun_intent without child markers is recoverable after reload');
			assert.equal(recoverableIntent.source_dump_id, statementRerunCorpus.source_dump_id);
			assert.equal(recoverableIntent.status, 'succeeded');
			assert.equal(recoverableIntent.n_candidates, 2);
			assert.equal(recoverableIntent.n_missing_markers, 2);
			assert.equal(recoverableIntent.n_uncovered_candidates, 0);
			assert.deepEqual(
				recoverableIntent.correction_ids,
				statementCorrections.map((r) => r.correction_id).sort((a, b) => a - b)
			);
			assert.deepEqual(recoverableIntent.uncovered_correction_ids, []);
			await assert.rejects(
				() => exportRepairRerunCorpus({
					run_id: RUN,
					correction_ids: statementCorrections.map((r) => r.correction_id)
				}),
				/already rerun/
			);
			await assert.rejects(
				() => recordRepairRerunIntent({
					parent_run_id: RUN,
					child_run_id: '13131313131313131313131313131313',
					architecture: 'decomposed',
					source_dump_id: statementRerunCorpus.source_dump_id,
					correction_ids: statementCorrections.map((r) => r.correction_id)
				}),
				/stale, closed, already consumed, or already have an active child run/
			);
			await unlink(join(dir, 'repair_reruns', `${statementRerunCorpus.source_dump_id}.meta.json`));
			const completedRepair = await recordRepairRerunChildAfterScore({
				parent_run_id: RUN,
				child_run_id: repairChildRun,
				architecture: 'decomposed',
				source_dump_id: statementRerunCorpus.source_dump_id,
				correction_ids: statementCorrections.map((r) => r.correction_id),
				path: statementRerunCorpus.path
			});
			assert.equal(completedRepair.recorded, 2);
			assert.equal(completedRepair.collateral_recorded, 1);
			assert.deepEqual(await repairMarkerCounts(dbFile, repairChildRun), {
				rerun_child: 2,
				rerun_collateral: 1
			});
			const childLineageRows = await repairLineageRows(dbFile, 'rerun_child', repairChildRun);
			assert.deepEqual(
				childLineageRows.map((row) => row.parent_correction_id),
				statementCorrections.map((r) => r.correction_id).sort((a, b) => a - b)
			);
			assert.ok(childLineageRows.every((row) => row.child_run_id === repairChildRun));
			assert.ok(childLineageRows.every((row) => row.repair_source_dump_id === statementRerunCorpus.source_dump_id));
			const collateralLineageRows = await repairLineageRows(dbFile, 'rerun_collateral', repairChildRun);
			assert.deepEqual(collateralLineageRows.map((row) => row.parent_correction_id), [null]);
			assert.ok(collateralLineageRows.every((row) => row.child_run_id === repairChildRun));
			assert.ok(collateralLineageRows.every((row) => row.repair_source_dump_id === statementRerunCorpus.source_dump_id));
			const completedBacklog = await getRunRepairBacklog(RUN);
			assert.equal(
				completedBacklog.recoverableRerunIntents.find((row) => row.child_run_id === repairChildRun),
				undefined,
				'recoverable intent disappears after append-only child markers are recorded'
			);
			const completedRerun = completedBacklog.reruns.find((row) => row.run_id === repairChildRun);
			assert.ok(completedRerun, 'completed repair child appears in before/after rerun comparisons');
			assert.equal(completedRerun.n_candidate_evidences, statementCorrections.length);
			assert.equal(completedRerun.n_child_covered_candidates, statementCorrections.length);
			assert.ok(
				completedRerun.candidate_lanes.some((lane) => lane.movement === 'verdict_to_correct'),
				'before/after candidate lanes expose verdict movement'
			);
			const completedDetail = await getRepairRerunDetail(RUN, repairChildRun);
			assert.ok(completedDetail, 'completed repair child has a full before/after detail contract');
			assert.equal(completedDetail.candidate_lane_total, statementCorrections.length);
			assert.equal(completedDetail.candidate_lanes.length, statementCorrections.length);
			assert.ok(
				completedDetail.candidate_lanes.some((lane) => lane.movement === 'verdict_to_correct'),
				'full before/after detail exposes verdict movement'
			);
			const pagedDetail = await getRepairRerunDetail(RUN, repairChildRun, { limit: 1, offset: 1 });
			assert.ok(pagedDetail, 'completed repair child detail supports offset pagination');
			assert.equal(pagedDetail.candidate_lane_total, statementCorrections.length);
			assert.equal(pagedDetail.candidate_lanes.length, 1);
			assert.equal(pagedDetail.candidate_lane_offset, 1);
			const completedRepairHtml = await repairPageHtml(RUN);
			assert.match(completedRepairHtml, /candidate before after repair lanes/);
			assert.match(completedRepairHtml, /repair candidates/);
			assert.match(completedRepairHtml, /verdict to correct/);
			assert.match(completedRepairHtml, /full repair comparison/);
			const completedRepairDetailHtml = await pageHtml(`/runs/${RUN}/repairs/${repairChildRun}`);
			assert.match(completedRepairDetailHtml, /full repair comparison/);
			assert.match(completedRepairDetailHtml, /full repair candidate before after lanes/);
			assert.match(completedRepairDetailHtml, /showing[\s\S]*1[\s\S]*-[\s\S]*2[\s\S]*of[\s\S]*2[\s\S]*repair candidates/);
			assert.match(completedRepairDetailHtml, /verdict to correct/);
			const pagedRepairDetailHtml = await pageHtml(`/runs/${RUN}/repairs/${repairChildRun}?limit=1`);
			assert.match(pagedRepairDetailHtml, /showing[\s\S]*1[\s\S]*-[\s\S]*1[\s\S]*of[\s\S]*2[\s\S]*repair candidates/);
			assert.match(pagedRepairDetailHtml, /next/);
			const completedRepairAgain = await recordRepairRerunChildAfterScore({
				parent_run_id: RUN,
				child_run_id: repairChildRun,
				architecture: 'decomposed',
				source_dump_id: statementRerunCorpus.source_dump_id,
				correction_ids: statementCorrections.map((r) => r.correction_id),
				path: statementRerunCorpus.path
			});
			assert.equal(completedRepairAgain.recorded, 0);
			assert.equal(completedRepairAgain.skipped_existing, 2);
			assert.equal(completedRepairAgain.collateral_recorded, 0);
			assert.equal(completedRepairAgain.collateral_skipped_existing, 1);
			await assertRepeatSkipped('statement-filtered', {
				filters: statementFilters,
				sourceRoute: statementRoute,
			expectedInspected: 2,
		expectedCorrections: statementCorrections.length
	});

	const statementVerdictFilters = {
		grain: 'statement',
		type: 'Activation',
		multi_evidence: true,
		verdict: 'correct'
	};
	const statementVerdictCohort = await getRunCohort(RUN, statementVerdictFilters);
	const statementVerdictRoute = `/runs/${RUN}/cohort?case=statement-verdict-filtered`;
	const statementVerdictResult = await materializeRepairCohort({
		run_id: RUN,
		filters: statementVerdictFilters,
		source_route: statementVerdictRoute,
		reviewer: 'parity-test'
	});
	const statementVerdictCorrections = await correctionRows(dbFile, statementVerdictRoute);
	assert.deepEqual(statementVerdictCohort.rows.map((r) => r.stmt_hash), ['stmt1']);
	assert.equal(statementVerdictResult.inspected, 1);
	assert.equal(statementVerdictResult.created, statementVerdictCorrections.length);
	assert.equal(statementVerdictResult.skipped_existing, 0);
	assert.equal(statementVerdictResult.skipped_duplicate_selection, 0);
	assert.deepEqual(statementVerdictCorrections.map((r) => r.evidence_hash), ['ev1']);
	assertUniqueEvidence(statementVerdictCorrections, 'statement-verdict-filtered');
	await assertRepeatSkipped('statement-verdict-filtered', {
		filters: statementVerdictFilters,
		sourceRoute: statementVerdictRoute,
		expectedInspected: 1,
		expectedCorrections: statementVerdictCorrections.length
	});
	const inProgressRepairCorpus = await exportRepairRerunCorpus({
		run_id: RUN,
		correction_ids: statementVerdictCorrections.map((r) => r.correction_id)
	});
	const queuedRepairChildRun = '19191919191919191919191919191919';
	const queuedRepairIntent = await recordRepairRerunIntent({
		parent_run_id: RUN,
		child_run_id: queuedRepairChildRun,
		architecture: 'decomposed',
		source_dump_id: inProgressRepairCorpus.source_dump_id,
		correction_ids: statementVerdictCorrections.map((r) => r.correction_id),
		path: inProgressRepairCorpus.path
	});
	assert.equal(queuedRepairIntent.recorded, 1);
	const queuedBacklog = await getRunRepairBacklog(RUN);
	assert.equal(queuedBacklog.activeRerunIntentCount, 1);
	assert.equal(queuedBacklog.activeRerunIntents[0].child_run_id, queuedRepairChildRun);
	assert.equal(queuedBacklog.activeRerunIntents[0].status, 'queued');
	assert.equal(queuedBacklog.activeRerunIntents[0].has_child_run, false);
	await assert.rejects(
		() => tombstoneStaleRepairRerunChild({
			parent_run_id: RUN,
			child_run_id: queuedRepairChildRun,
			architecture: 'decomposed',
			source_dump_id: inProgressRepairCorpus.source_dump_id,
			correction_ids: statementVerdictCorrections.map((r) => r.correction_id),
			path: inProgressRepairCorpus.path
		}),
		/queued repair intents expire automatically/
	);
	await assert.rejects(
		() => exportRepairRerunCorpus({
			run_id: RUN,
			correction_ids: statementVerdictCorrections.map((r) => r.correction_id)
		}),
		/rerun in progress/
	);
	await ageRepairIntent(dbFile, queuedRepairChildRun, '2025-01-01 00:00:00');
	const expiredQueuedBacklog = await getRunRepairBacklog(RUN);
	assert.equal(expiredQueuedBacklog.activeRerunIntentCount, 0);
	assert.equal(
		expiredQueuedBacklog.rows.some((row) => row.correction_id === statementVerdictCorrections[0].correction_id),
		true,
		'old queued intent without a child run releases candidate back to the spendable queue'
	);
	const activeRepairChildRun = '17171717171717171717171717171717';
	await insertRepairChildRun(dbFile, activeRepairChildRun, 'running', []);
	const activeRepairIntent = await recordRepairRerunIntent({
		parent_run_id: RUN,
		child_run_id: activeRepairChildRun,
		architecture: 'decomposed',
		source_dump_id: inProgressRepairCorpus.source_dump_id,
		correction_ids: statementVerdictCorrections.map((r) => r.correction_id),
		path: inProgressRepairCorpus.path
	});
	assert.equal(activeRepairIntent.recorded, 1);
	const activeBacklog = await getRunRepairBacklog(RUN);
	assert.equal(activeBacklog.activeRerunIntentCount, 1);
	assert.equal(activeBacklog.activeRerunIntents[0].child_run_id, activeRepairChildRun);
	assert.deepEqual(
		activeBacklog.activeRerunIntents[0].correction_ids,
		statementVerdictCorrections.map((r) => r.correction_id)
	);
	assert.equal(
		activeBacklog.rows.some((row) => row.correction_id === statementVerdictCorrections[0].correction_id),
		false,
		'active rerun intents hide candidates from the spendable queue'
	);
	let activeRepairHtml = await repairPageHtml(RUN);
	assert.match(activeRepairHtml, /<section[^>]*aria-label="active repair reruns locking candidates"/);
	let activeRepairSection = sectionByAriaLabel(activeRepairHtml, 'active repair reruns locking candidates');
	assert.equal(
		buttonIsDisabled(repairActiveCancelButton(activeRepairSection)),
		false,
		'active repair child cancel button is enabled at first paint when no writer lock is active'
	);
	assert.equal(
		buttonIsDisabled(buttonByAriaLabel(activeRepairHtml, 'preview repair rerun cost')),
		false,
		'repair preflight preview is enabled at first paint when no writer lock is active'
	);
	await assert.rejects(
		() => exportRepairRerunCorpus({
			run_id: RUN,
			correction_ids: statementVerdictCorrections.map((r) => r.correction_id)
		}),
		/rerun in progress/
	);
	await assert.rejects(
		() => recordRepairRerunIntent({
			parent_run_id: RUN,
			child_run_id: '18181818181818181818181818181818',
			architecture: 'decomposed',
			source_dump_id: inProgressRepairCorpus.source_dump_id,
			correction_ids: statementVerdictCorrections.map((r) => r.correction_id),
			path: inProgressRepairCorpus.path
		}),
		/active child run/
	);
	const tombstoneBusyLock = acquireWriterLock({
		kind: 'repair',
		label: 'parity tombstone busy lock',
		source_dump_id: inProgressRepairCorpus.source_dump_id,
		dataset_path: dbFile,
		architecture: 'decomposed',
		pid: process.pid
	});
	assert.ok(tombstoneBusyLock);
	try {
		const lockedRepairPage = await pageResponse(`/runs/${RUN}/repairs`);
		assert.equal(lockedRepairPage.status, 503);
		assert.match(lockedRepairPage.body, /writer in progress/);
		assert.match(lockedRepairPage.body, /repair/);
		assert.doesNotMatch(lockedRepairPage.body, new RegExp(escapeRegExp(tombstoneBusyLock.token)));
		assert.doesNotMatch(lockedRepairPage.body, new RegExp(escapeRegExp(dbFile)));
		await assert.rejects(
			() => tombstoneStaleRepairRerunChild({
				parent_run_id: RUN,
				child_run_id: activeRepairChildRun,
				architecture: 'decomposed',
				source_dump_id: inProgressRepairCorpus.source_dump_id,
				correction_ids: statementVerdictCorrections.map((r) => r.correction_id),
				path: inProgressRepairCorpus.path
			}),
			/DuckDB writer|already active|busy/
		);
	} finally {
		clearWriterLockToken(tombstoneBusyLock.token);
	}
	const tombstonedChild = await tombstoneStaleRepairRerunChild({
		parent_run_id: RUN,
		child_run_id: activeRepairChildRun,
		architecture: 'decomposed',
		source_dump_id: inProgressRepairCorpus.source_dump_id,
		correction_ids: statementVerdictCorrections.map((r) => r.correction_id),
		path: inProgressRepairCorpus.path
	});
	assert.equal(tombstonedChild.canceled, true);
	assert.equal(tombstonedChild.status, 'canceled');
	assert.deepEqual(
		tombstonedChild.correction_ids,
		statementVerdictCorrections.map((r) => r.correction_id)
	);
	assert.equal(await scoreRunStatus(dbFile, activeRepairChildRun), 'canceled');
	const tombstonedAgain = await tombstoneStaleRepairRerunChild({
		parent_run_id: RUN,
		child_run_id: activeRepairChildRun,
		architecture: 'decomposed',
		source_dump_id: inProgressRepairCorpus.source_dump_id,
		correction_ids: statementVerdictCorrections.map((r) => r.correction_id),
		path: inProgressRepairCorpus.path
	});
	assert.equal(tombstonedAgain.canceled, false);
	assert.equal(tombstonedAgain.status, 'canceled');
	const reopenedBacklog = await getRunRepairBacklog(RUN);
	assert.equal(reopenedBacklog.activeRerunIntentCount, 0);
	assert.equal(
		reopenedBacklog.rows.some((row) => row.correction_id === statementVerdictCorrections[0].correction_id),
		true,
		'tombstoned stale child intent releases candidate back to the spendable queue'
	);
	const uncoveredRepairCorpus = await exportRepairRerunCorpus({
		run_id: RUN,
		correction_ids: statementVerdictCorrections.map((r) => r.correction_id)
	});
	const failedRepairChildRun = '12121212121212121212121212121212';
	await insertRepairChildRun(dbFile, failedRepairChildRun, 'failed');
	await recordRepairRerunIntent({
		parent_run_id: RUN,
		child_run_id: failedRepairChildRun,
		architecture: 'decomposed',
		source_dump_id: uncoveredRepairCorpus.source_dump_id,
		correction_ids: statementVerdictCorrections.map((r) => r.correction_id),
		path: uncoveredRepairCorpus.path
	});
	await assert.rejects(
		() => recordRepairRerunChildAfterScore({
			parent_run_id: RUN,
			child_run_id: failedRepairChildRun,
			architecture: 'decomposed',
			source_dump_id: uncoveredRepairCorpus.source_dump_id,
			correction_ids: statementVerdictCorrections.map((r) => r.correction_id),
			path: uncoveredRepairCorpus.path
		}),
		/status=succeeded/
	);
	const uncoveredRepairChildRun = '16161616161616161616161616161616';
	await insertRepairChildRun(dbFile, uncoveredRepairChildRun, 'succeeded', []);
	await recordRepairRerunIntent({
		parent_run_id: RUN,
		child_run_id: uncoveredRepairChildRun,
		architecture: 'decomposed',
		source_dump_id: uncoveredRepairCorpus.source_dump_id,
		correction_ids: statementVerdictCorrections.map((r) => r.correction_id),
		path: uncoveredRepairCorpus.path
	});
	const uncoveredBacklog = await getRunRepairBacklog(RUN);
	const uncoveredIntent = uncoveredBacklog.recoverableRerunIntents.find(
		(row) => row.child_run_id === uncoveredRepairChildRun
	);
	assert.ok(uncoveredIntent, 'succeeded intent with missing child coverage remains visible after reload');
	assert.equal(uncoveredIntent.n_candidates, 0);
	assert.equal(uncoveredIntent.n_missing_markers, 1);
	assert.equal(uncoveredIntent.n_uncovered_candidates, 1);
	assert.deepEqual(uncoveredIntent.correction_ids, []);
	assert.deepEqual(
		uncoveredIntent.uncovered_correction_ids,
		statementVerdictCorrections.map((r) => r.correction_id)
	);
	await assert.rejects(
		() => recordRepairRerunChildAfterScore({
			parent_run_id: RUN,
			child_run_id: uncoveredRepairChildRun,
			architecture: 'decomposed',
			source_dump_id: uncoveredRepairCorpus.source_dump_id,
			correction_ids: statementVerdictCorrections.map((r) => r.correction_id),
			path: uncoveredRepairCorpus.path
		}),
		/does not cover selected repair candidates/
	);
	const releasedUncovered = await recordRepairRerunUncovered({
		parent_run_id: RUN,
		child_run_id: uncoveredRepairChildRun,
		architecture: 'decomposed',
		source_dump_id: uncoveredRepairCorpus.source_dump_id,
		correction_ids: statementVerdictCorrections.map((r) => r.correction_id),
		path: uncoveredRepairCorpus.path
	});
	assert.equal(releasedUncovered.recorded, 1);
	const uncoveredLineageRows = await repairLineageRows(dbFile, 'rerun_uncovered', uncoveredRepairChildRun);
	assert.deepEqual(
		uncoveredLineageRows.map((row) => row.parent_correction_id),
		statementVerdictCorrections.map((r) => r.correction_id)
	);
	assert.ok(uncoveredLineageRows.every((row) => row.child_run_id === uncoveredRepairChildRun));
	assert.ok(uncoveredLineageRows.every((row) => row.repair_source_dump_id === uncoveredRepairCorpus.source_dump_id));
	const releasedBacklog = await getRunRepairBacklog(RUN);
	assert.equal(
		releasedBacklog.recoverableRerunIntents.find((row) => row.child_run_id === uncoveredRepairChildRun),
		undefined,
		'uncovered release removes the dead recovery row'
	);
	const releasedExport = await exportRepairRerunCorpus({
		run_id: RUN,
		correction_ids: statementVerdictCorrections.map((r) => r.correction_id)
	});
	assert.equal(releasedExport.n_candidates, 1);
	await insertRecoverableIntentPageFixtures(dbFile, 22);
	const recoveryPageOne = await getRunRepairBacklog(RUN);
	assert.equal(recoveryPageOne.recoverableRerunIntentCount, 22);
	assert.equal(recoveryPageOne.recoverableRerunIntentOffset, 0);
	assert.equal(recoveryPageOne.recoverableRerunIntentLimit, 20);
	assert.equal(recoveryPageOne.recoverableRerunIntents.length, 20);
	const recoveryPageTwo = await getRunRepairBacklog(RUN, { recoveryOffset: 20 });
	assert.equal(recoveryPageTwo.recoverableRerunIntentCount, 22);
	assert.equal(recoveryPageTwo.recoverableRerunIntentOffset, 20);
	assert.equal(recoveryPageTwo.recoverableRerunIntentLimit, 20);
	assert.equal(recoveryPageTwo.recoverableRerunIntents.length, 2);
	assert.notEqual(
		recoveryPageOne.recoverableRerunIntents[0].child_run_id,
		recoveryPageTwo.recoverableRerunIntents[0].child_run_id,
		'recovery pages expose different child run groups'
	);
	const recoveryPageClamped = await getRunRepairBacklog(RUN, { recoveryOffset: 999 });
	assert.equal(recoveryPageClamped.recoverableRerunIntentCount, 22);
	assert.equal(recoveryPageClamped.recoverableRerunIntentOffset, 20);
	assert.equal(recoveryPageClamped.recoverableRerunIntents.length, 2);
	const recoveryPageNonBoundary = await getRunRepairBacklog(RUN, { recoveryOffset: 5 });
	assert.equal(recoveryPageNonBoundary.recoverableRerunIntentCount, 22);
	assert.equal(recoveryPageNonBoundary.recoverableRerunIntentOffset, 0);
	assert.equal(recoveryPageNonBoundary.recoverableRerunIntents.length, 20);

	const statementTruthFilters = {
		grain: 'statement',
		truth_set: 'gold'
	};
	const statementTruthCohort = await getRunCohort(RUN, statementTruthFilters);
	const statementTruthRoute = `/runs/${RUN}/cohort?case=statement-truth-filtered`;
	const statementTruthResult = await materializeRepairCohort({
		run_id: RUN,
		filters: statementTruthFilters,
		source_route: statementTruthRoute,
		reviewer: 'parity-test'
	});
	const statementTruthCorrections = await correctionRows(dbFile, statementTruthRoute);
	assert.deepEqual(statementTruthCohort.rows.map((r) => r.stmt_hash), ['stmt1']);
	assert.equal(statementTruthResult.inspected, 1);
	assert.equal(statementTruthResult.created, statementTruthCorrections.length);
	assert.equal(statementTruthResult.skipped_existing, 0);
	assert.equal(statementTruthResult.skipped_duplicate_selection, 0);
	assert.deepEqual(statementTruthCorrections.map((r) => r.evidence_hash), ['ev1']);
	assertUniqueEvidence(statementTruthCorrections, 'statement-truth-filtered');
	await assertRepeatSkipped('statement-truth-filtered', {
		filters: statementTruthFilters,
		sourceRoute: statementTruthRoute,
		expectedInspected: 1,
		expectedCorrections: statementTruthCorrections.length
	});

	const statementSourceFilters = {
		grain: 'statement',
		type: 'MixedSourceCase',
		source: 'zz_source'
	};
	const statementSourceCohort = await getRunCohort(RUN, statementSourceFilters);
	const statementSourceRoute = `/runs/${RUN}/cohort?case=statement-source-filtered`;
	const statementSourceResult = await materializeRepairCohort({
		run_id: RUN,
		filters: statementSourceFilters,
		source_route: statementSourceRoute,
		reviewer: 'parity-test'
	});
	const statementSourceCorrections = await correctionRows(dbFile, statementSourceRoute);
	assert.deepEqual(statementSourceCohort.rows.map((r) => r.stmt_hash), ['stmt_mixed_source']);
	assert.equal(statementSourceCohort.rows[0].source_api, 'zz_source');
	assert.equal(statementSourceCohort.rows[0].source_stratum, 'alpha_source');
	assert.equal(statementSourceCohort.rows[0].representative_evidence_hash, 'ev_mixed_z');
	assert.equal(statementSourceCohort.rows[0].pmid, 'PMID-Z');
	assert.match(statementSourceCohort.rows[0].text, /scored mixed-source evidence/);
	assert.equal(statementSourceResult.inspected, 1);
	assert.equal(statementSourceResult.created, statementSourceCorrections.length);
	assert.equal(statementSourceResult.skipped_existing, 0);
	assert.equal(statementSourceResult.skipped_duplicate_selection, 0);
	assert.deepEqual(statementSourceCorrections.map((r) => r.evidence_hash), ['ev_mixed_z']);
	assertUniqueEvidence(statementSourceCorrections, 'statement-source-filtered');
	await assertRepeatSkipped('statement-source-filtered', {
		filters: statementSourceFilters,
		sourceRoute: statementSourceRoute,
		expectedInspected: 1,
		expectedCorrections: statementSourceCorrections.length
	});

	const statementSourceStratumHitFilters = {
		grain: 'statement',
		type: 'MixedSourceCase',
		source_stratum: 'alpha_source'
	};
	const statementSourceStratumHitCohort = await getRunCohort(RUN, statementSourceStratumHitFilters);
	const statementSourceStratumHitRoute = `/runs/${RUN}/cohort?case=statement-source-stratum-hit`;
	const statementSourceStratumHitResult = await materializeRepairCohort({
		run_id: RUN,
		filters: statementSourceStratumHitFilters,
		source_route: statementSourceStratumHitRoute,
		reviewer: 'parity-test'
	});
	const statementSourceStratumHitCorrections = await correctionRows(dbFile, statementSourceStratumHitRoute);
	assert.deepEqual(statementSourceStratumHitCohort.rows.map((r) => r.stmt_hash), ['stmt_mixed_source']);
	assert.equal(statementSourceStratumHitCohort.rows[0].source_api, 'zz_source');
	assert.equal(statementSourceStratumHitCohort.rows[0].source_stratum, 'alpha_source');
	assert.equal(statementSourceStratumHitCohort.rows[0].representative_evidence_hash, 'ev_mixed_z');
	assert.equal(statementSourceStratumHitCohort.rows[0].pmid, 'PMID-Z');
	assert.match(statementSourceStratumHitCohort.rows[0].text, /scored mixed-source evidence/);
	assert.equal(statementSourceStratumHitResult.inspected, 1);
	assert.equal(statementSourceStratumHitResult.created, statementSourceStratumHitCorrections.length);
	assert.equal(statementSourceStratumHitResult.skipped_existing, 0);
	assert.equal(statementSourceStratumHitResult.skipped_duplicate_selection, 0);
	assert.deepEqual(statementSourceStratumHitCorrections.map((r) => r.evidence_hash), ['ev_mixed_z']);
	assertUniqueEvidence(statementSourceStratumHitCorrections, 'statement-source-stratum-hit');
	await assertRepeatSkipped('statement-source-stratum-hit', {
		filters: statementSourceStratumHitFilters,
		sourceRoute: statementSourceStratumHitRoute,
		expectedInspected: 1,
		expectedCorrections: statementSourceStratumHitCorrections.length
	});

	const statementSplitNoMatchFilters = {
		grain: 'statement',
		type: 'SplitFilterCase',
		source: 'zz_source',
		verdict: 'correct'
	};
	const statementSplitNoMatchCohort = await getRunCohort(RUN, statementSplitNoMatchFilters);
	const statementSplitNoMatchRoute = `/runs/${RUN}/cohort?case=statement-split-no-match`;
	const statementSplitNoMatchResult = await materializeRepairCohort({
		run_id: RUN,
		filters: statementSplitNoMatchFilters,
		source_route: statementSplitNoMatchRoute,
		reviewer: 'parity-test'
	});
	assert.equal(statementSplitNoMatchCohort.totalRows, 0);
	assert.equal(statementSplitNoMatchResult.inspected, 0);
	assert.equal(statementSplitNoMatchResult.created, 0);
	assert.equal(statementSplitNoMatchResult.skipped_existing, 0);
	assert.equal(statementSplitNoMatchResult.skipped_duplicate_selection, 0);
	assert.deepEqual(await correctionRows(dbFile, statementSplitNoMatchRoute), []);

	const statementSplitMatchFilters = {
		grain: 'statement',
		type: 'SplitFilterCase',
		source: 'zz_source',
		verdict: 'incorrect'
	};
	const statementSplitMatchCohort = await getRunCohort(RUN, statementSplitMatchFilters);
	const statementSplitMatchRoute = `/runs/${RUN}/cohort?case=statement-split-match`;
	const statementSplitMatchResult = await materializeRepairCohort({
		run_id: RUN,
		filters: statementSplitMatchFilters,
		source_route: statementSplitMatchRoute,
		reviewer: 'parity-test'
	});
	const statementSplitMatchCorrections = await correctionRows(dbFile, statementSplitMatchRoute);
	assert.deepEqual(statementSplitMatchCohort.rows.map((r) => r.stmt_hash), ['stmt_split_filters']);
	assert.equal(statementSplitMatchCohort.rows[0].source_api, 'zz_source');
	assert.equal(statementSplitMatchCohort.rows[0].representative_evidence_hash, 'ev_split_z');
	assert.equal(statementSplitMatchCohort.rows[0].pmid, 'PMID-SZ');
	assert.match(statementSplitMatchCohort.rows[0].text, /incorrect split zz evidence/);
	assert.equal(statementSplitMatchResult.inspected, 1);
	assert.equal(statementSplitMatchResult.created, statementSplitMatchCorrections.length);
	assert.equal(statementSplitMatchResult.skipped_existing, 0);
	assert.equal(statementSplitMatchResult.skipped_duplicate_selection, 0);
	assert.deepEqual(statementSplitMatchCorrections.map((r) => r.evidence_hash), ['ev_split_z']);
	assertUniqueEvidence(statementSplitMatchCorrections, 'statement-split-match');
	await assertRepeatSkipped('statement-split-match', {
		filters: statementSplitMatchFilters,
		sourceRoute: statementSplitMatchRoute,
		expectedInspected: 1,
		expectedCorrections: statementSplitMatchCorrections.length
	});

	const statementTraceAggregateMatchFilters = {
		grain: 'statement',
		type: 'Activation',
		source: 'reach',
		verdict: 'incorrect',
		trace_fidelity: 'aggregate_only'
	};
	const statementTraceAggregateMatchCohort = await getRunCohort(RUN, statementTraceAggregateMatchFilters);
	const statementTraceAggregateMatchRoute = `/runs/${RUN}/cohort?case=statement-trace-aggregate-match`;
	const statementTraceAggregateMatchResult = await materializeRepairCohort({
		run_id: RUN,
		filters: statementTraceAggregateMatchFilters,
		source_route: statementTraceAggregateMatchRoute,
		reviewer: 'parity-test'
	});
	const statementTraceAggregateMatchCorrections = await correctionRows(dbFile, statementTraceAggregateMatchRoute);
	assert.deepEqual(statementTraceAggregateMatchCohort.rows.map((r) => r.stmt_hash), ['stmt1']);
	assert.equal(statementTraceAggregateMatchCohort.rows[0].representative_evidence_hash, 'ev2');
	assert.equal(statementTraceAggregateMatchResult.inspected, 1);
	assert.equal(statementTraceAggregateMatchResult.created, statementTraceAggregateMatchCorrections.length);
	assert.equal(statementTraceAggregateMatchResult.skipped_existing, 0);
	assert.equal(statementTraceAggregateMatchResult.skipped_duplicate_selection, 0);
	assert.deepEqual(statementTraceAggregateMatchCorrections.map((r) => r.evidence_hash), ['ev2']);
	assertUniqueEvidence(statementTraceAggregateMatchCorrections, 'statement-trace-aggregate-match');
	await assertRepeatSkipped('statement-trace-aggregate-match', {
		filters: statementTraceAggregateMatchFilters,
		sourceRoute: statementTraceAggregateMatchRoute,
		expectedInspected: 1,
		expectedCorrections: statementTraceAggregateMatchCorrections.length
	});

	const statementTraceAggregateNoMatchFilters = {
		grain: 'statement',
		type: 'Activation',
		source: 'reach',
		verdict: 'correct',
		trace_fidelity: 'aggregate_only'
	};
	const statementTraceAggregateNoMatchCohort = await getRunCohort(RUN, statementTraceAggregateNoMatchFilters);
	const statementTraceAggregateNoMatchRoute = `/runs/${RUN}/cohort?case=statement-trace-aggregate-no-match`;
	const statementTraceAggregateNoMatchResult = await materializeRepairCohort({
		run_id: RUN,
		filters: statementTraceAggregateNoMatchFilters,
		source_route: statementTraceAggregateNoMatchRoute,
		reviewer: 'parity-test'
	});
	assert.equal(statementTraceAggregateNoMatchCohort.totalRows, 0);
	assert.equal(statementTraceAggregateNoMatchResult.inspected, 0);
	assert.equal(statementTraceAggregateNoMatchResult.created, 0);
	assert.equal(statementTraceAggregateNoMatchResult.skipped_existing, 0);
	assert.equal(statementTraceAggregateNoMatchResult.skipped_duplicate_selection, 0);
	assert.deepEqual(await correctionRows(dbFile, statementTraceAggregateNoMatchRoute), []);

	const statementTraceNativeMatchFilters = {
		grain: 'statement',
		type: 'Activation',
		source: 'reach',
		verdict: 'correct',
		trace_fidelity: 'native_decomposed'
	};
	const statementTraceNativeMatchCohort = await getRunCohort(RUN, statementTraceNativeMatchFilters);
	const statementTraceNativeMatchRoute = `/runs/${RUN}/cohort?case=statement-trace-native-match`;
	const statementTraceNativeMatchResult = await materializeRepairCohort({
		run_id: RUN,
		filters: statementTraceNativeMatchFilters,
		source_route: statementTraceNativeMatchRoute,
		reviewer: 'parity-test'
	});
	const statementTraceNativeMatchCorrections = await correctionRows(dbFile, statementTraceNativeMatchRoute);
	assert.deepEqual(statementTraceNativeMatchCohort.rows.map((r) => r.stmt_hash), ['stmt1']);
	assert.equal(statementTraceNativeMatchCohort.rows[0].representative_evidence_hash, 'ev1');
	assert.equal(statementTraceNativeMatchResult.inspected, 1);
	assert.equal(statementTraceNativeMatchResult.created, statementTraceNativeMatchCorrections.length);
	assert.equal(statementTraceNativeMatchResult.skipped_existing, 0);
	assert.equal(statementTraceNativeMatchResult.skipped_duplicate_selection, 0);
	assert.deepEqual(statementTraceNativeMatchCorrections.map((r) => r.evidence_hash), ['ev1']);
	assertUniqueEvidence(statementTraceNativeMatchCorrections, 'statement-trace-native-match');
	await assertRepeatSkipped('statement-trace-native-match', {
		filters: statementTraceNativeMatchFilters,
		sourceRoute: statementTraceNativeMatchRoute,
		expectedInspected: 1,
		expectedCorrections: statementTraceNativeMatchCorrections.length
	});

	const statementTraceNativeNoMatchFilters = {
		grain: 'statement',
		type: 'Activation',
		source: 'reach',
		verdict: 'incorrect',
		trace_fidelity: 'native_decomposed'
	};
	const statementTraceNativeNoMatchCohort = await getRunCohort(RUN, statementTraceNativeNoMatchFilters);
	const statementTraceNativeNoMatchRoute = `/runs/${RUN}/cohort?case=statement-trace-native-no-match`;
	const statementTraceNativeNoMatchResult = await materializeRepairCohort({
		run_id: RUN,
		filters: statementTraceNativeNoMatchFilters,
		source_route: statementTraceNativeNoMatchRoute,
		reviewer: 'parity-test'
	});
	assert.equal(statementTraceNativeNoMatchCohort.totalRows, 0);
	assert.equal(statementTraceNativeNoMatchResult.inspected, 0);
	assert.equal(statementTraceNativeNoMatchResult.created, 0);
	assert.equal(statementTraceNativeNoMatchResult.skipped_existing, 0);
	assert.equal(statementTraceNativeNoMatchResult.skipped_duplicate_selection, 0);
	assert.deepEqual(await correctionRows(dbFile, statementTraceNativeNoMatchRoute), []);

	const statementSourceStratumFilters = {
		grain: 'statement',
		source_stratum: 'zz_source'
	};
	const statementSourceStratumCohort = await getRunCohort(RUN, statementSourceStratumFilters);
	const statementSourceStratumRoute = `/runs/${RUN}/cohort?case=statement-source-stratum-filtered`;
	const statementSourceStratumResult = await materializeRepairCohort({
		run_id: RUN,
		filters: statementSourceStratumFilters,
		source_route: statementSourceStratumRoute,
		reviewer: 'parity-test'
	});
	assert.equal(statementSourceStratumCohort.totalRows, 0);
	assert.equal(statementSourceStratumResult.inspected, 0);
	assert.equal(statementSourceStratumResult.created, 0);
	assert.equal(statementSourceStratumResult.skipped_existing, 0);
	assert.equal(statementSourceStratumResult.skipped_duplicate_selection, 0);
	assert.deepEqual(await correctionRows(dbFile, statementSourceStratumRoute), []);

		const manyFilters = {
			grain: 'statement',
			type: 'Complex'
		};
		const manyCohort = await getRunCohort(RUN, manyFilters);
		const manyRoute = `/runs/${RUN}/cohort?case=statement-many-evidence`;
		const manyEstimate = await estimateRepairCohort({
			run_id: RUN,
			filters: manyFilters,
			source_route: manyRoute,
			reviewer: 'parity-test'
		});
		assert.equal(manyEstimate.inspected, 501);
		assert.equal(manyEstimate.unique_selected, 501);
		assert.equal(manyEstimate.would_create, 501);
		assert.equal(manyEstimate.skipped_existing, 0);
		assert.equal(manyEstimate.skipped_duplicate_selection, 0);
		const manyResult = await materializeRepairCohort({
			run_id: RUN,
			filters: manyFilters,
		source_route: manyRoute,
		reviewer: 'parity-test'
	});
	const manyCorrections = await correctionRows(dbFile, manyRoute);
	assert.deepEqual(manyCohort.rows.map((r) => r.stmt_hash), ['stmt4']);
	assert.equal(manyResult.inspected, 501);
	assert.equal(manyResult.created, manyCorrections.length);
	assert.equal(manyResult.skipped_existing, 0);
	assert.equal(manyResult.skipped_duplicate_selection, 0);
	assert.equal(manyCorrections.length, 501);
	assertUniqueEvidence(manyCorrections, 'statement-many-evidence');
	await assertRepeatSkipped('statement-many-evidence', {
		filters: manyFilters,
		sourceRoute: manyRoute,
		expectedInspected: 501,
		expectedCorrections: manyCorrections.length
	});

	await assertTraceParity('trace-missing-filtered', {
		trace_state: 'missing_aggregate',
		step_kind: 'parse_claim',
		type: 'Activation',
		source: 'reach',
		truth_set: 'gold'
	});

	await assertTraceParity('trace-snapshot-filtered', {
		trace_state: 'missing_aggregate',
		trace_snapshot: '2026-01-01 00:05:00',
		step_kind: 'parse_claim',
		type: 'Inhibition',
		source: 'reach'
	});

	await assertTraceParity('trace-step-error-filtered', {
		trace_state: 'step_error',
		step_kind: 'build_context',
		source: 'reach'
	}, {
		expected_suspected_by_evidence: {
			ev_trace_error: 'build_context',
			ev_trace_error_mixed: 'parse_claim'
		}
	});

	await assertTraceParity('trace-running-terminated-filtered', {
		trace_state: 'terminated_inflight',
		trace_snapshot: '2026-01-01 00:05:00',
		step_kind: 'parse_claim',
		source: 'reach'
	}, {
		run_id: RUNNING_RUN,
		expected_run_status: 'running'
	});
	await assertProbeCoverageParity();

	const succeededNoSnapshotPath = `/runs/${RUN}/cohort?trace_state=missing_aggregate&step_kind=parse_claim`;
	const succeededNoSnapshotRoute = await pageResponse(succeededNoSnapshotPath, { redirect: 'manual' });
	assert.equal(succeededNoSnapshotRoute.status, 200, 'succeeded trace-state cohort without snapshot renders directly');
	assert.equal(succeededNoSnapshotRoute.location, null, 'succeeded trace-state cohort without snapshot is not redirected to a pin');
	assert.match(succeededNoSnapshotRoute.body, /1 trace evidence row/, 'succeeded no-snapshot route renders the visible trace population');
	assert.doesNotMatch(succeededNoSnapshotRoute.body, /snapshot pinned/, 'succeeded no-snapshot route does not fabricate a pinned snapshot');

	const succeededNoSnapshotApiSource = `/runs/${RUN}/cohort?trace_state=missing_aggregate&step_kind=parse_claim&case=api-succeeded-nosnapshot`;
	const succeededNoSnapshotEstimate = await postEstimate({
		run_id: RUN,
		filters: { trace_state: 'missing_aggregate', step_kind: 'parse_claim' },
		source_route: succeededNoSnapshotApiSource,
		expected_run_status: 'succeeded',
		reviewer: 'parity-test'
	});
	assert.equal(succeededNoSnapshotEstimate.status, 200);
	assert.equal(succeededNoSnapshotEstimate.body.inspected, 1);
	assert.equal(succeededNoSnapshotEstimate.body.filters.trace_snapshot, null, 'estimate preserves the unpinned finished-run cohort');
	const succeededNoSnapshotRepair = await postRepair({
		run_id: RUN,
		filters: { trace_state: 'missing_aggregate', step_kind: 'parse_claim' },
		source_route: succeededNoSnapshotApiSource,
		expected_run_status: 'succeeded',
		reviewer: 'parity-test',
		estimate_token: succeededNoSnapshotEstimate.body.estimate_token
	});
	assert.equal(succeededNoSnapshotRepair.status, 200);
	assert.equal(succeededNoSnapshotRepair.body.inspected, 1);
	assert.equal(succeededNoSnapshotRepair.body.created, 1);
	assert.equal(succeededNoSnapshotRepair.body.filters.trace_snapshot, null, 'repair API preserves the same unpinned finished-run cohort');

	await assert.rejects(
		() => materializeRepairCohort({
			run_id: RUN,
			filters: { trace_state: 'missing_aggregate' },
			source_route: `/runs/${RUN}/cohort?case=trace-status-conflict`,
			expected_run_status: 'running',
			reviewer: 'parity-test'
		}),
		(e) => e instanceof RepairCohortConflictError && e.expected === 'running' && e.actual === 'succeeded'
	);

	await assert.rejects(
		() => materializeRepairCohort({
			run_id: RUN,
			filters: { verdict: 'correct' },
			source_route: `/runs/${RUN}/cohort?case=nontrace-status-conflict`,
			expected_run_status: 'running',
			reviewer: 'parity-test'
		}),
		(e) => e instanceof RepairCohortConflictError && e.expected === 'running' && e.actual === 'succeeded'
	);

	await assert.rejects(
		() => estimateRepairCohort({
			run_id: RUNNING_RUN,
			filters: { verdict: 'incorrect' },
			source_route: `/runs/${RUNNING_RUN}/cohort?case=nontrace-running-direct`,
			expected_run_status: 'running',
			reviewer: 'parity-test'
		}),
		(e) => e instanceof RepairCohortHttpError
			&& e.status === 409
			&& e.code === 'repair_requires_succeeded_run'
	);

	await assert.rejects(
		() => materializeRepairCohort({
			run_id: RUNNING_RUN,
			filters: { trace_state: 'terminated_inflight', step_kind: 'parse_claim' },
			source_route: `/runs/${RUNNING_RUN}/cohort?case=trace-running-unpinned`,
			expected_run_status: 'running',
			reviewer: 'parity-test'
		}),
		(e) => e instanceof RepairCohortInputError
			&& e.status === 400
			&& e.code === 'trace_snapshot_required'
			&& /requires trace_snapshot/.test(e.message)
	);

	const runningNonTraceRoute = await pageResponse(`/runs/${RUNNING_RUN}/cohort?verdict=incorrect`);
	assert.equal(runningNonTraceRoute.status, 200);
	assert.match(runningNonTraceRoute.body, /aggregate and statement repair backlogs require a completed run/);
	assert.match(
		runningNonTraceRoute.body,
		/<button\b[^>]*disabled[^>]*>\s*estimate repair write\s*<\/button>/,
		'running aggregate repair estimate button is disabled'
	);

	let apiResponse = await postRepair({
		run_id: 'not-a-run',
		filters: {}
	});
	assert.equal(apiResponse.status, 400);
	assert.equal(apiResponse.body.code, 'invalid_run_id');

	apiResponse = await postRaw('{not-json');
	assert.equal(apiResponse.status, 400);
	assert.equal(apiResponse.body.code, 'invalid_json');

		apiResponse = await postRepair({
			run_id: RUN,
			filters: { trace_state: 'missing_aggregate' }
		});
		assert.equal(apiResponse.status, 400);
		assert.equal(apiResponse.body.code, 'expected_run_status_required');

		apiResponse = await postEstimate({
			run_id: RUN,
			filters: { verdict: 'correct' }
		});
		assert.equal(apiResponse.status, 400);
		assert.equal(apiResponse.body.code, 'expected_run_status_required');

		apiResponse = await postRepair({
			run_id: RUN,
			filters: { verdict: 'correct' }
		});
		assert.equal(apiResponse.status, 400);
	assert.equal(apiResponse.body.code, 'expected_run_status_required');

		apiResponse = await postRepair({
			run_id: RUN,
			expected_run_status: 'done-ish',
			filters: {}
		});
		assert.equal(apiResponse.status, 400);
		assert.equal(apiResponse.body.code, 'invalid_expected_run_status');

		apiResponse = await postRepair({
			run_id: RUN,
			filters: { verdict: 'correct' },
			expected_run_status: 'succeeded',
			source_route: `/runs/${RUN}/cohort?case=api-no-estimate-token`
		});
		assert.equal(apiResponse.status, 400);
		assert.equal(apiResponse.body.code, 'repair_estimate_required');

		const apiEstimateRoute = `/runs/${RUN}/cohort?case=api-estimate-duplicate`;
		apiResponse = await postEstimate({
			run_id: RUN,
			filters: { type: 'DuplicateCase' },
			expected_run_status: 'succeeded',
			source_route: apiEstimateRoute
		});
		assert.equal(apiResponse.status, 200);
		assert.equal(apiResponse.body.inspected, 2);
		assert.equal(apiResponse.body.unique_selected, 1);
		assert.equal(apiResponse.body.would_create, 1);
		assert.equal(apiResponse.body.skipped_existing, 0);
		assert.equal(apiResponse.body.skipped_duplicate_selection, 1);
		assert.deepEqual(await correctionRows(dbFile, apiEstimateRoute), []);
		const apiEstimateToken = apiResponse.body.estimate_token;
		assert.equal(typeof apiEstimateToken, 'string');
		apiResponse = await postRepair({
			run_id: RUN,
			filters: { type: 'DuplicateCase' },
			expected_run_status: 'succeeded',
			source_route: apiEstimateRoute,
			estimate_token: apiEstimateToken
		});
		assert.equal(apiResponse.status, 200);
		assert.equal(apiResponse.body.inspected, 2);
		assert.equal(apiResponse.body.created, 1);
		assert.equal(apiResponse.body.skipped_duplicate_selection, 1);
		assert.equal((await correctionRows(dbFile, apiEstimateRoute)).length, 1);
		apiResponse = await postRepair({
			run_id: RUN,
			filters: { type: 'DuplicateCase' },
			expected_run_status: 'succeeded',
			source_route: apiEstimateRoute,
			estimate_token: apiEstimateToken
		});
		assert.equal(apiResponse.status, 409);
		assert.equal(apiResponse.body.code, 'repair_estimate_expired');

		const apiStaleEstimateRoute = `/runs/${RUN}/cohort?case=api-estimate-stale`;
		apiResponse = await postEstimate({
			run_id: RUN,
			filters: { type: 'DuplicateCase' },
			expected_run_status: 'succeeded',
			source_route: apiStaleEstimateRoute
		});
		const apiStaleToken = apiResponse.body.estimate_token;
		apiResponse = await postRepair({
			run_id: RUN,
			filters: { verdict: 'correct' },
			expected_run_status: 'succeeded',
			source_route: apiStaleEstimateRoute,
			estimate_token: apiStaleToken
		});
		assert.equal(apiResponse.status, 409);
		assert.equal(apiResponse.body.code, 'repair_estimate_stale');

		apiResponse = await postRepair({
			run_id: RUN,
			filters: { not_a_filter: 'x' }
	});
	assert.equal(apiResponse.status, 400);
	assert.equal(apiResponse.body.code, 'unknown_cohort_filter');

	apiResponse = await postRepair({
		run_id: RUN,
		filters: { trace_state: 'bad_trace' },
		expected_run_status: 'succeeded'
	});
	assert.equal(apiResponse.status, 400);
	assert.equal(apiResponse.body.code, 'invalid_trace_state');

	apiResponse = await postRepair({
		run_id: RUN,
		filters: { trace_state: 'missing_aggregate', trace_snapshot: 'not-a-date' },
		expected_run_status: 'succeeded'
	});
	assert.equal(apiResponse.status, 400);
	assert.equal(apiResponse.body.code, 'invalid_trace_snapshot');

	apiResponse = await postRepair({
		run_id: RUN,
		filters: { trace_snapshot: '2026-01-01 00:01:00' },
		expected_run_status: 'succeeded'
	});
	assert.equal(apiResponse.status, 400);
	assert.equal(apiResponse.body.code, 'trace_plane_required');

	apiResponse = await postRepair({
		run_id: RUN,
		filters: { type: 'BogusType' },
		expected_run_status: 'succeeded',
		source_route: `/runs/${RUN}/cohort?case=unknown-type-api`
	});
	assert.equal(apiResponse.status, 400);
	assert.equal(apiResponse.body.code, 'unknown_type_filter');

	apiResponse = await postRepair({
		run_id: RUNNING_RUN,
		filters: { trace_state: 'terminated_inflight', step_kind: 'parse_claim' },
		expected_run_status: 'running',
		source_route: `/runs/${RUNNING_RUN}/cohort?case=trace-running-unpinned-api`,
		reviewer: 'parity-test'
	});
	assert.equal(apiResponse.status, 400);
	assert.equal(apiResponse.body.code, 'trace_snapshot_required');

	apiResponse = await postRepair({
		run_id: RUNNING_RUN,
		filters: {
			trace_state: 'terminated_inflight',
			trace_snapshot: '2026-01-01 00:05:00',
			step_kind: 'parse_claim'
		},
		expected_run_status: 'succeeded',
		source_route: `/runs/${RUNNING_RUN}/cohort?case=trace-running-stale-api`,
		reviewer: 'parity-test'
	});
	assert.equal(apiResponse.status, 409);
	assert.equal(apiResponse.body.code, 'run_status_changed');
	assert.equal(apiResponse.body.actual_run_status, 'running');

	apiResponse = await postRepair({
		run_id: RUN,
		filters: { verdict: 'correct' },
		expected_run_status: 'running',
		source_route: `/runs/${RUN}/cohort?case=nontrace-status-conflict-api`,
		reviewer: 'parity-test'
	});
	assert.equal(apiResponse.status, 409);
	assert.equal(apiResponse.body.code, 'run_status_changed');
	assert.equal(apiResponse.body.actual_run_status, 'succeeded');

	apiResponse = await postEstimate({
		run_id: RUNNING_RUN,
		filters: { verdict: 'incorrect' },
		expected_run_status: 'running',
		source_route: `/runs/${RUNNING_RUN}/cohort?case=nontrace-running-estimate-api`,
		reviewer: 'parity-test'
	});
	assert.equal(apiResponse.status, 409);
	assert.equal(apiResponse.body.code, 'repair_requires_succeeded_run');
	apiResponse = await postRepair({
		run_id: RUNNING_RUN,
		filters: { verdict: 'incorrect' },
		expected_run_status: 'running',
		source_route: `/runs/${RUNNING_RUN}/cohort?case=nontrace-running-materialize-api`,
		reviewer: 'parity-test'
	});
	assert.equal(apiResponse.status, 409);
	assert.equal(apiResponse.body.code, 'repair_requires_succeeded_run');

	let writerLockSnapshot = await getWriterLockSnapshot();
	assert.equal(writerLockSnapshot.status, 200);
	assert.equal(writerLockSnapshot.body.ok, true);
	assert.equal(writerLockSnapshot.body.writerLock, null);

	const lock = acquireWriterLock({
		kind: 'ingest',
		label: 'parity test lock',
		dataset_path: dbFile,
		pid: process.pid
	});
	assert.ok(lock);
	try {
		apiResponse = await postRepair({
			run_id: RUN,
			filters: { verdict: 'correct' },
			expected_run_status: 'succeeded',
			source_route: `/runs/${RUN}/cohort?case=writer-lock-api`,
			reviewer: 'parity-test'
		});
		assert.equal(apiResponse.status, 409);
		assert.equal(apiResponse.body.code, 'writer_lock_busy');
		writerLockSnapshot = await getWriterLockSnapshot();
		assert.equal(writerLockSnapshot.status, 200);
		assert.equal(writerLockSnapshot.body.writerLock.kind, 'ingest');
		assert.equal(writerLockSnapshot.body.writerLock.label, 'parity test lock');
		assert.equal(writerLockSnapshot.body.writerLock.pid, process.pid);
		assertPublicWriterLockShape(writerLockSnapshot.body.writerLock);
		await assert.rejects(
			() => loadDashboardSnapshot(),
			(err) => {
				assert.equal(err?.status, 503);
				assert.equal(err?.body?.code, 'writer_in_progress');
				assert.match(String(err?.body?.message ?? ''), /ingest/);
				assert.doesNotMatch(String(err?.body?.message ?? ''), new RegExp(escapeRegExp(lock.token)));
				assert.doesNotMatch(String(err?.body?.message ?? ''), new RegExp(escapeRegExp(dbFile)));
				return true;
			}
		);
		const dashboardLockPage = await pageResponse('/');
		assert.equal(dashboardLockPage.status, 503);
		assert.match(dashboardLockPage.body, /writer in progress/);
		assert.doesNotMatch(dashboardLockPage.body, new RegExp(escapeRegExp(lock.token)));
	} finally {
		clearWriterLockToken(lock.token);
	}
	writerLockSnapshot = await getWriterLockSnapshot();
	assert.equal(writerLockSnapshot.body.writerLock, null);

	const malformedWriterLockPath = join(dir, 'viewer_state', 'writer_lock.json');
	await mkdir(join(dir, 'viewer_state'), { recursive: true });
	await writeFile(malformedWriterLockPath, JSON.stringify({
		kind: 'repair',
		token: 'malformed-repair-test',
		started_at: '2026-05-26T12:00:00.000Z',
		updated_at: 'not-a-time'
	}));
	apiResponse = await postEstimate({
		run_id: RUN,
		filters: { verdict: 'correct' },
		expected_run_status: 'succeeded',
		source_route: `/runs/${RUN}/cohort?case=malformed-writer-lock-estimate`,
		reviewer: 'parity-test'
	});
	assert.equal(apiResponse.status, 409);
	assert.equal(apiResponse.body.code, 'writer_lock_malformed');
	assert.match(apiResponse.body.message, /writer_lock\.json/);
	apiResponse = await postRepair({
		run_id: RUN,
		filters: { verdict: 'correct' },
		expected_run_status: 'succeeded',
		source_route: `/runs/${RUN}/cohort?case=malformed-writer-lock-materialize`,
		reviewer: 'parity-test'
	});
	assert.equal(apiResponse.status, 409);
	assert.equal(apiResponse.body.code, 'writer_lock_malformed');
	assert.match(apiResponse.body.message, /writer_lock\.json/);
	await unlink(malformedWriterLockPath);
	writerLockSnapshot = await getWriterLockSnapshot();
	assert.equal(writerLockSnapshot.body.writerLock, null);

	const activePairId = 'repairactivepair000000000000000001';
	createPairedWorkflowState({
		pair_id: activePairId,
		source_dump_id: 'repair-parity-source',
		dataset_path: dbFile,
		model: 'smoke-local',
		scorer_version: 'repair-parity',
		total_cost_threshold_usd: 0.05,
		caps: { monolithic: 0.025, decomposed: 0.025 }
	});
	try {
		apiResponse = await postEstimate({
			run_id: RUN,
			filters: { verdict: 'correct' },
			expected_run_status: 'succeeded',
			source_route: `/runs/${RUN}/cohort?case=active-pair-estimate`,
			reviewer: 'parity-test'
		});
		assert.equal(apiResponse.status, 409);
		assert.equal(apiResponse.body.code, 'paired_workflow_active');
		assert.match(apiResponse.body.message, new RegExp(activePairId));
		apiResponse = await postRepair({
			run_id: RUN,
			filters: { verdict: 'correct' },
			expected_run_status: 'succeeded',
			source_route: `/runs/${RUN}/cohort?case=active-pair-materialize`,
			reviewer: 'parity-test'
		});
		assert.equal(apiResponse.status, 409);
		assert.equal(apiResponse.body.code, 'paired_workflow_active');
		assert.match(apiResponse.body.message, new RegExp(activePairId));
	} finally {
		await unlink(join(dir, 'viewer_state', 'paired', `${activePairId}.json`));
	}

	closeInstance();
	const missingDbFile = `${dbFile}.missing`;
	await rename(dbFile, missingDbFile);
	try {
		apiResponse = await postEstimate({
			run_id: RUN,
			filters: { verdict: 'correct' },
			expected_run_status: 'succeeded',
			source_route: `/runs/${RUN}/cohort?case=missing-db-estimate`,
			reviewer: 'parity-test'
		});
		assert.equal(apiResponse.status, 404);
		assert.equal(apiResponse.body.code, 'corpus_db_missing');
		apiResponse = await postRepair({
			run_id: RUN,
			filters: { verdict: 'correct' },
			expected_run_status: 'succeeded',
			source_route: `/runs/${RUN}/cohort?case=missing-db-materialize`,
			reviewer: 'parity-test'
		});
		assert.equal(apiResponse.status, 404);
		assert.equal(apiResponse.body.code, 'corpus_db_missing');
	} finally {
		await rename(missingDbFile, dbFile);
	}

	const fallbackRoute = `/runs/${RUN}/cohort?case=generic-repair-failure`;
	apiResponse = await postEstimate({
		run_id: RUN,
		filters: { verdict: 'correct' },
		expected_run_status: 'succeeded',
		source_route: fallbackRoute,
		reviewer: 'parity-test'
	});
	assert.equal(apiResponse.status, 200);
	const fallbackToken = apiResponse.body.estimate_token;
	const fallbackInstance = await DuckDBInstance.create(dbFile);
	const fallbackCon = await fallbackInstance.connect();
	try {
		await fallbackCon.run('DROP TABLE statement');
	} finally {
		fallbackCon.disconnectSync?.();
		fallbackInstance.closeSync();
	}
	apiResponse = await postRepair({
		run_id: RUN,
		filters: { verdict: 'correct' },
		expected_run_status: 'succeeded',
		source_route: fallbackRoute,
		reviewer: 'parity-test',
		estimate_token: fallbackToken
	});
	assert.equal(apiResponse.status, 500);
	assert.equal(apiResponse.body.code, 'repair_cohort_failed');

	closeInstance();
	console.log('cohort repair parity tests passed');
} finally {
	httpServer.closeIdleConnections?.();
	httpServer.closeAllConnections?.();
	await new Promise((resolve) => httpServer.close(resolve));
	await server.close();
	await rm(dir, { recursive: true, force: true });
}
