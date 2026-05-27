import assert from 'node:assert/strict';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createServer } from 'vite';
import { DuckDBInstance } from '@duckdb/node-api';

function httpMessage(e) {
	return e?.body?.message ?? e?.message ?? String(e);
}

function assertHttp400(fn, pattern) {
	try {
		fn();
	} catch (e) {
		assert.equal(e.status, 400);
		assert.match(httpMessage(e), pattern);
		return;
	}
	assert.fail('expected HttpError 400');
}

async function assertRejectsHttp400(fn, pattern) {
	try {
		await fn();
	} catch (e) {
		assert.equal(e.status, 400);
		assert.match(httpMessage(e), pattern);
		return;
	}
	assert.fail('expected HttpError 400');
}

const server = await createServer({
	configFile: './vite.config.ts',
	server: { middlewareMode: true, hmr: { port: 30_000 + (process.pid % 10_000) } },
	logLevel: 'error'
});

try {
	const {
		cohortFiltersFromRecord,
		cohortFiltersFromSearchParams,
		expectedRunStatus,
		validateRunCohortFilterValues
	} = await server.ssrLoadModule('/src/lib/server/runCohortContract.ts');

	const traceFilters = cohortFiltersFromSearchParams(new URLSearchParams(
		'trace_state=missing_aggregate&grain=evidence&trace_snapshot=2026-05-24T16:42:00&multi_evidence=yes&supports=0'
	));
	assert.deepEqual(traceFilters, {
		grain: 'evidence',
		verdict: null,
		verdict_present: false,
		score_present: false,
		indra_belief_present: false,
		confidence: null,
		type: null,
		source: null,
		source_stratum: null,
		truth_set: null,
		step_kind: null,
			trace_fidelity: null,
			trace_state: 'missing_aggregate',
			trace_snapshot: '2026-05-24 16:42:00',
			probe_coverage: null,
			multi_evidence: true,
		supports: false,
		supports_compare: false
	});

	assert.deepEqual(cohortFiltersFromRecord({
		verdict: 'correct',
		confidence: 'high',
		multi_evidence: 'no',
		supports_compare: true
	}), {
		grain: null,
		verdict: 'correct',
		verdict_present: false,
		score_present: false,
		indra_belief_present: false,
		confidence: 'high',
		type: null,
		source: null,
		source_stratum: null,
		truth_set: null,
		step_kind: null,
			trace_fidelity: null,
			trace_state: null,
			trace_snapshot: null,
			probe_coverage: null,
			multi_evidence: false,
		supports: false,
		supports_compare: true
	});

	assert.deepEqual(cohortFiltersFromSearchParams(new URLSearchParams(
		'grain=statement&score_present=true&indra_belief_present=1&verdict_present=yes'
	)), {
		grain: 'statement',
		verdict: null,
		verdict_present: true,
		score_present: true,
		indra_belief_present: true,
		confidence: null,
		type: null,
		source: null,
		source_stratum: null,
		truth_set: null,
		step_kind: null,
			trace_fidelity: null,
			trace_state: null,
			trace_snapshot: null,
			probe_coverage: null,
			multi_evidence: false,
		supports: false,
		supports_compare: false
	});

	assertHttp400(
		() => cohortFiltersFromSearchParams(new URLSearchParams('run_id=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa')),
		/unknown cohort filter: run_id/
	);
		assertHttp400(
			() => cohortFiltersFromSearchParams(new URLSearchParams('trace_state=missing_aggregate&grain=statement')),
			/trace-plane cohorts require grain=evidence/
		);
		assertHttp400(
			() => cohortFiltersFromSearchParams(new URLSearchParams('probe_coverage=present&grain=statement')),
			/trace-plane cohorts require grain=evidence/
		);
	assertHttp400(
		() => cohortFiltersFromSearchParams(new URLSearchParams('verdict=maybe')),
		/invalid verdict: maybe/
	);
	assertHttp400(
		() => cohortFiltersFromSearchParams(new URLSearchParams('confidence=certain')),
		/invalid confidence: certain/
	);
	assertHttp400(
		() => cohortFiltersFromSearchParams(new URLSearchParams('multi_evidence=maybe')),
		/invalid multi_evidence: maybe/
	);
	assertHttp400(
		() => cohortFiltersFromSearchParams(new URLSearchParams('indra_belief_present=maybe')),
		/invalid indra_belief_present: maybe/
	);
	assertHttp400(
		() => cohortFiltersFromRecord({ verdict: 7 }),
		/verdict must be a string/
	);
		assertHttp400(
			() => cohortFiltersFromRecord({ trace_state: 'missing_aggregate', trace_fidelity: 'aggregate_only' }),
			/trace_state and trace_fidelity cannot be combined/
		);
		assert.deepEqual(cohortFiltersFromSearchParams(new URLSearchParams(
			'probe_coverage=present&trace_snapshot=2026-05-24T16:42:00'
		)), {
			grain: 'evidence',
			verdict: null,
			verdict_present: false,
			score_present: false,
			indra_belief_present: false,
			confidence: null,
			type: null,
			source: null,
			source_stratum: null,
			truth_set: null,
			step_kind: null,
			trace_fidelity: null,
			trace_state: null,
			trace_snapshot: '2026-05-24 16:42:00',
			probe_coverage: 'present',
			multi_evidence: false,
			supports: false,
			supports_compare: false
		});
		assertHttp400(
			() => cohortFiltersFromSearchParams(new URLSearchParams('probe_coverage=absent')),
			/invalid probe_coverage: absent/
		);
		assert.equal(expectedRunStatus('succeeded'), 'succeeded');
	assert.equal(expectedRunStatus(null), null);
	assertHttp400(
		() => expectedRunStatus('done-ish'),
		/expected_run_status must be one of/
	);
	assertHttp400(
		() => cohortFiltersFromRecord({ type: 'x'.repeat(257) }),
		/type must be 256 characters or fewer/
	);

	const dir = await mkdtemp(join(tmpdir(), 'indra-cohort-contract-'));
	const dbFile = join(dir, 'contract.duckdb');
	const instance = await DuckDBInstance.create(dbFile);
	const con = await instance.connect();
	try {
		await con.run('CREATE TABLE statement (stmt_hash VARCHAR, indra_type VARCHAR)');
		await con.run('CREATE TABLE evidence (evidence_hash VARCHAR, stmt_hash VARCHAR, source_api VARCHAR)');
		await con.run('CREATE TABLE truth_set (id VARCHAR)');
		await con.run('CREATE TABLE scorer_step (run_id VARCHAR, step_kind VARCHAR)');
		await con.run("INSERT INTO statement VALUES ('stmt1', 'Activation')");
		await con.run("INSERT INTO evidence VALUES ('ev1', 'stmt1', 'reach')");
		await con.run("INSERT INTO truth_set VALUES ('gold')");
		await con.run("INSERT INTO scorer_step VALUES ('aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa', 'aggregate')");
		await con.run("INSERT INTO scorer_step VALUES ('aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa', 'parse_claim')");

		await validateRunCohortFilterValues(con, 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa', {
			grain: 'statement',
			verdict: 'correct',
			confidence: 'high',
			type: 'Activation',
			source: 'reach',
			source_stratum: 'reach',
			truth_set: 'gold',
			step_kind: 'parse_claim'
		});
		await assertRejectsHttp400(
			() => validateRunCohortFilterValues(con, 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa', { type: 'BogusType' }),
			/unknown type filter: BogusType/
		);
		await assertRejectsHttp400(
			() => validateRunCohortFilterValues(con, 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa', { source: 'bogus_source' }),
			/unknown source filter: bogus_source/
		);
		await assertRejectsHttp400(
			() => validateRunCohortFilterValues(con, 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa', { source_stratum: 'bogus_source' }),
			/unknown source_stratum filter: bogus_source/
		);
		await assertRejectsHttp400(
			() => validateRunCohortFilterValues(con, 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa', { truth_set: 'bogus_truth' }),
			/unknown truth_set filter: bogus_truth/
		);
		await assertRejectsHttp400(
			() => validateRunCohortFilterValues(con, 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa', { step_kind: 'grounding' }),
			/unknown step_kind filter: grounding/
		);
		await assertRejectsHttp400(
			() => validateRunCohortFilterValues(con, 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa', { grain: 'thread' }),
			/invalid grain: thread/
		);
		await assertRejectsHttp400(
			() => validateRunCohortFilterValues(con, 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa', {
				trace_state: 'missing_aggregate',
				trace_fidelity: 'aggregate_only'
			}),
			/trace_state and trace_fidelity cannot be combined/
		);
			await assertRejectsHttp400(
				() => validateRunCohortFilterValues(con, 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa', {
					trace_snapshot: '2026-05-24 16:42:00'
				}),
				/trace_snapshot requires trace_state or probe_coverage/
			);
	} finally {
		con.disconnectSync?.();
		instance.closeSync();
		await rm(dir, { recursive: true, force: true });
	}

	console.log('cohort contract tests passed');
} finally {
	await server.close();
}
