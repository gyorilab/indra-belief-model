import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { mkdir, mkdtemp, rm, writeFile } from 'node:fs/promises';
import { createServer as createHttpServer } from 'node:http';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createServer } from 'vite';
import { DuckDBInstance } from '@duckdb/node-api';

const RUN_ID = 'bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb';
const dir = await mkdtemp(join(tmpdir(), 'indra-latest-validity-error-'));
const dbFile = join(dir, 'broken-validity.duckdb');
const pairedInputFile = join(dir, 'paired-input.json');
process.env.VIEWER_DUCKDB_PATH = dbFile;
process.env.INDRA_VIEWER_SUPPRESS_WRITER_GUARD_LOG = '1';
await writeFile(pairedInputFile, '[]');

function errorMessage(err) {
	return String(err?.body?.message ?? err?.message ?? err);
}

function assertCodedError(err, code, detailPattern) {
	assert.equal(err?.body?.code, code);
	assert.match(errorMessage(err), new RegExp(code));
	assert.match(errorMessage(err), detailPattern);
	return true;
}

async function holdWriterLock(path) {
	const child = spawn(
		process.execPath,
		[
			'--input-type=module',
			'-e',
			`
				import { DuckDBInstance } from '@duckdb/node-api';
				const instance = await DuckDBInstance.create(${JSON.stringify(path)});
				const con = await instance.connect();
				console.log('ready');
				process.on('SIGTERM', () => {
					try { con.disconnectSync?.(); } catch {}
					try { instance.closeSync(); } catch {}
					process.exit(0);
				});
				setInterval(() => {}, 1000);
			`
		],
		{ cwd: process.cwd(), env: process.env, stdio: ['ignore', 'pipe', 'pipe'] }
	);
	let stdout = '';
	let stderr = '';
	await new Promise((resolve, reject) => {
		const timeout = setTimeout(
			() => reject(new Error(`writer-lock child did not become ready; stderr=${stderr}`)),
			5000
		);
		child.stdout.setEncoding('utf8');
		child.stderr.setEncoding('utf8');
		child.stdout.on('data', (chunk) => {
			stdout += chunk;
			if (stdout.includes('ready')) {
				clearTimeout(timeout);
				resolve();
			}
		});
		child.stderr.on('data', (chunk) => {
			stderr += chunk;
		});
		child.once('exit', (code, signal) => {
			clearTimeout(timeout);
			reject(new Error(`writer-lock child exited early: code=${code} signal=${signal} stderr=${stderr}`));
		});
	});
	return {
		async close() {
			if (child.exitCode !== null) return;
			child.kill('SIGTERM');
			await new Promise((resolve) => child.once('exit', resolve));
		}
	};
}

const instance = await DuckDBInstance.create(dbFile);
const con = await instance.connect();
try {
	await con.run(`
		CREATE TABLE statement (
			stmt_hash VARCHAR,
			indra_type VARCHAR,
			source_dump_id VARCHAR
		)
	`);
	await con.run(`
		CREATE TABLE evidence (
			evidence_hash VARCHAR,
			stmt_hash VARCHAR,
			source_api VARCHAR
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
			field VARCHAR
		)
	`);
} finally {
	con.disconnectSync?.();
	instance.closeSync();
}

const server = await createServer({
	configFile: './vite.config.ts',
	server: { middlewareMode: true, hmr: { port: 30_000 + (process.pid % 10_000) } },
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
	const db = await server.ssrLoadModule('/src/lib/db.ts');
	const route = await server.ssrLoadModule('/src/routes/+page.server.ts');
	const runRoute = await server.ssrLoadModule('/src/routes/runs/[run_id]/+page.server.ts');
	const cohortRoute = await server.ssrLoadModule('/src/routes/runs/[run_id]/cohort/+page.server.ts');
	const repairsRoute = await server.ssrLoadModule('/src/routes/runs/[run_id]/repairs/+page.server.ts');
	const truthOverlapRoute = await server.ssrLoadModule('/src/routes/runs/[run_id]/truth-sets/[truth_set_id]/+page.server.ts');
	const statementsRoute = await server.ssrLoadModule('/src/routes/statements/+page.server.ts');
	const statementRoute = await server.ssrLoadModule('/src/routes/statements/[stmt_hash]/+page.server.ts');
	const pairRoute = await server.ssrLoadModule('/src/routes/pairs/[pair_id]/+page.server.ts');
	const ingestRoute = await server.ssrLoadModule('/src/routes/api/datasets/ingest/+server.ts');
	const scoreRoute = await server.ssrLoadModule('/src/routes/api/runs/score/+server.ts');
	const scorePairedRoute = await server.ssrLoadModule('/src/routes/api/runs/score-paired/+server.ts');
	const truthSetRoute = await server.ssrLoadModule('/src/routes/api/truth-sets/+server.ts');
	const repairRerunEstimateRoute = await server.ssrLoadModule('/src/routes/api/repairs/rerun/estimate/+server.ts');
	const pairedState = await server.ssrLoadModule('/src/lib/server/pairedState.ts');

	const writerActionCases = () => [
		[
			'ingest',
			ingestRoute.POST,
			'http://localhost/api/datasets/ingest',
			{ path: pairedInputFile, source_dump_id: 'cycle85_pair' }
		],
		[
			'single score',
			scoreRoute.POST,
			'http://localhost/api/runs/score',
			{
				path: pairedInputFile,
				source_dump_id: 'cycle85_pair',
				model: 'test-model',
				scorer_version: 'test-v1',
				arch: 'decomposed',
				cost_threshold_usd: 0.01
			}
		],
		[
			'truth set',
			truthSetRoute.POST,
			'http://localhost/api/truth-sets',
			{
				path: pairedInputFile,
				truth_set_id: 'cycle85_truth',
				truth_set_name: 'cycle85 truth',
				target_kind: 'evidence',
				field: 'tag'
			}
		]
	];

	function postEvent(url, body) {
		return {
			request: new Request(url, {
				method: 'POST',
				headers: { 'content-type': 'application/json' },
				body: JSON.stringify(body)
			})
		};
	}

	async function assertWriterActionHttpConflict(expectedCode, detailPattern) {
		for (const [label, , url, body] of writerActionCases()) {
			const endpointUrl = `${baseUrl}${new URL(url).pathname}`;
			const response = await fetch(endpointUrl, {
				method: 'POST',
				headers: { 'content-type': 'application/json' },
				body: JSON.stringify(body)
			});
			assert.equal(response.status, 409, String(label));
			const responseBody = await response.json();
			assert.equal(responseBody.code, expectedCode, String(label));
			assert.match(String(responseBody.message), detailPattern, String(label));
			assert.doesNotMatch(String(responseBody.message), /score_run/i, String(label));
		}
	}

	const writerLock = await holdWriterLock(dbFile);
	try {
		await assert.rejects(
			() => db.getCorpusOverview(),
			(err) => assertCodedError(err, 'writer_in_progress', /write lock/i)
		);
		const writerResponse = await fetch(baseUrl);
		assert.equal(writerResponse.status, 503);
		const writerHtml = await writerResponse.text();
		assert.match(writerHtml, /writer in progress/);
		assert.doesNotMatch(writerHtml, /corpus schema mismatch/);
		assert.doesNotMatch(writerHtml, /\b\d+\s+statements?\b/i);
	} finally {
		await writerLock.close();
	}
	db.closeInstance();

	const sidecarLock = pairedState.acquireWriterLock({
		kind: 'single_score',
		label: 'cycle85 sidecar writer lock',
		architecture: 'decomposed',
		model: 'test-model',
		pid: process.pid
	});
	assert.ok(sidecarLock);
	try {
		const guardedLoads = [
			['dashboard', () => route.load({ url: new URL('http://localhost/') })],
			['run detail', () => runRoute.load({ params: { run_id: RUN_ID }, url: new URL(`http://localhost/runs/${RUN_ID}`) })],
			['run cohort', () => cohortRoute.load({ params: { run_id: RUN_ID }, url: new URL(`http://localhost/runs/${RUN_ID}/cohort`) })],
			['run repairs', () => repairsRoute.load({ params: { run_id: RUN_ID }, url: new URL(`http://localhost/runs/${RUN_ID}/repairs`) })],
			[
				'truth overlap',
				() => truthOverlapRoute.load({
					params: { run_id: RUN_ID, truth_set_id: 'cycle85_truth' },
					url: new URL(`http://localhost/runs/${RUN_ID}/truth-sets/cycle85_truth?step_kind=aggregate`)
				})
			],
			['statement list', () => statementsRoute.load({ url: new URL('http://localhost/statements') })],
			[
				'statement detail',
				() => statementRoute.load({
					params: { stmt_hash: 'abcdef0123456789' },
					url: new URL('http://localhost/statements/abcdef0123456789')
				})
			],
			[
				'pair workbench',
				() => pairRoute.load({
					params: { pair_id: 'cycle85_pair' },
					cookies: { get: () => undefined },
					url: new URL('http://localhost/pairs/cycle85_pair')
				})
			]
		];
		for (const [label, load] of guardedLoads) {
			await assert.rejects(load, (err) => assertCodedError(err, 'writer_in_progress', /single_score/), String(label));
		}
		await assert.rejects(
			() => scorePairedRoute.POST({
				request: new Request('http://localhost/api/runs/score-paired', {
					method: 'POST',
					headers: { 'content-type': 'application/json' },
					body: JSON.stringify({
						path: pairedInputFile,
						source_dump_id: 'cycle85_pair',
						model: 'test-model',
						scorer_version: 'test-v1',
						paired_run_group_id: 'cycle85_pair',
						monolithic_cost_threshold_usd: 0.01,
						decomposed_cost_threshold_usd: 0.01,
						total_cost_threshold_usd: 0.02
					})
				})
			}),
			(err) => {
				assert.equal(err?.status, 409);
				assert.equal(err?.body?.code, 'writer_in_progress');
				assert.match(errorMessage(err), /single_score/);
				assert.doesNotMatch(errorMessage(err), /score_run/i);
				return true;
			}
		);
		const endpointResponse = await fetch(`${baseUrl}/api/runs/score-paired`, {
			method: 'POST',
			headers: { 'content-type': 'application/json' },
			body: JSON.stringify({
				path: pairedInputFile,
				source_dump_id: 'cycle85_pair',
				model: 'test-model',
				scorer_version: 'test-v1',
				paired_run_group_id: 'cycle85_pair_http',
				monolithic_cost_threshold_usd: 0.01,
				decomposed_cost_threshold_usd: 0.01,
				total_cost_threshold_usd: 0.02
			})
		});
		assert.equal(endpointResponse.status, 409);
		const endpointBody = await endpointResponse.json();
		assert.equal(endpointBody.code, 'writer_in_progress');
		assert.match(String(endpointBody.message), /single_score/);
		for (const [label, handler, url, body] of writerActionCases()) {
			await assert.rejects(
				() => handler(postEvent(url, body)),
				(err) => {
					assert.equal(err?.status, 409, String(label));
					assert.equal(err?.body?.code, 'writer_lock_busy', String(label));
					assert.match(errorMessage(err), /single_score/, String(label));
					assert.doesNotMatch(errorMessage(err), /score_run/i, String(label));
					return true;
				},
				String(label)
			);
		}
		await assertWriterActionHttpConflict('writer_lock_busy', /single_score/);
		await assert.rejects(
			() => runRoute.load({ params: { run_id: RUN_ID }, url: new URL(`http://localhost/runs/${RUN_ID}?trace_page=bad`) }),
			(err) => {
				assert.notEqual(err?.body?.code, 'writer_in_progress');
				assert.match(errorMessage(err), /trace_page/);
				return true;
			}
		);
		const sidecarResponse = await fetch(baseUrl);
		assert.equal(sidecarResponse.status, 503);
		const sidecarHtml = await sidecarResponse.text();
		assert.match(sidecarHtml, /writer in progress/);
		assert.match(sidecarHtml, /single_score/);
		assert.doesNotMatch(sidecarHtml, /corpus schema mismatch/);
		assert.doesNotMatch(sidecarHtml, /\b\d+\s+statements?\b/i);
	} finally {
		pairedState.clearWriterLockToken(sidecarLock.token);
	}

	const malformedWriterLockPath = join(dir, 'viewer_state', 'writer_lock.json');
	await mkdir(join(dir, 'viewer_state'), { recursive: true });
	await writeFile(malformedWriterLockPath, '{not-json');
	const malformedLock = pairedState.activePublicWriterLock();
	assert.equal(malformedLock?.kind, 'malformed');
	assert.match(String(malformedLock?.malformed_reason), /unreadable JSON/);
	await assert.rejects(
		() => route.load({ url: new URL('http://localhost/') }),
		(err) => assertCodedError(err, 'writer_lock_malformed', /writer_lock\.json/)
	);
	await assert.rejects(
		() => scorePairedRoute.POST({
			request: new Request('http://localhost/api/runs/score-paired', {
				method: 'POST',
				headers: { 'content-type': 'application/json' },
				body: JSON.stringify({
					path: pairedInputFile,
					source_dump_id: 'cycle85_pair',
					model: 'test-model',
					scorer_version: 'test-v1',
					paired_run_group_id: 'cycle87_malformed_lock_pair',
					monolithic_cost_threshold_usd: 0.01,
					decomposed_cost_threshold_usd: 0.01,
					total_cost_threshold_usd: 0.02
				})
			})
		}),
		(err) => {
			assert.equal(err?.status, 409);
			assert.equal(err?.body?.code, 'writer_lock_malformed');
			assert.match(errorMessage(err), /writer_lock\.json/);
			assert.doesNotMatch(errorMessage(err), /score_run/i);
			return true;
		}
	);
	for (const [label, handler, url, body] of writerActionCases()) {
		await assert.rejects(
			() => handler(postEvent(url, body)),
			(err) => {
				assert.equal(err?.status, 409, String(label));
				assert.equal(err?.body?.code, 'writer_lock_malformed', String(label));
				assert.match(errorMessage(err), /writer_lock\.json/, String(label));
				assert.doesNotMatch(errorMessage(err), /score_run/i, String(label));
				return true;
			},
			String(label)
		);
	}
	await assertWriterActionHttpConflict('writer_lock_malformed', /writer_lock\.json/);
	await assert.rejects(
		() => repairRerunEstimateRoute.POST({
			request: new Request('http://localhost/api/repairs/rerun/estimate', {
				method: 'POST',
				headers: { 'content-type': 'application/json' },
				body: JSON.stringify({
					run_id: RUN_ID,
					arch: 'decomposed'
				})
			})
		}),
		(err) => {
			assert.equal(err?.status, 409);
			assert.equal(err?.body?.code, 'writer_lock_malformed');
			assert.match(errorMessage(err), /writer_lock\.json/);
			assert.doesNotMatch(errorMessage(err), /repair rerun/i);
			return true;
		}
	);
	const malformedResponse = await fetch(baseUrl);
	assert.equal(malformedResponse.status, 503);
	const malformedHtml = await malformedResponse.text();
	assert.match(malformedHtml, /writer lock needs repair/);
	assert.doesNotMatch(malformedHtml, /corpus schema mismatch/);
	assert.equal(pairedState.acquireWriterLock({
		kind: 'single_score',
		label: 'cycle87 must not overwrite malformed lock',
		pid: process.pid
	}), null);
	await rm(malformedWriterLockPath, { force: true });
	await writeFile(malformedWriterLockPath, JSON.stringify({
		kind: 'single_score',
		token: 'cycle87-invalid-time',
		started_at: '2026-05-25T12:00:00.000Z',
		updated_at: 'not-a-time'
	}));
	const invalidTimeLock = pairedState.activePublicWriterLock();
	assert.equal(invalidTimeLock?.kind, 'malformed');
	assert.match(String(invalidTimeLock?.malformed_reason), /updated_at/);
	assert.equal(pairedState.acquireWriterLock({
		kind: 'single_score',
		label: 'cycle87 must not stale-delete invalid timestamp',
		pid: process.pid
	}), null);
	assert.equal(pairedState.updateWriterLock('__malformed_writer_lock__', { label: 'cycle87 sentinel hijack' }), null);
	pairedState.clearWriterLockToken('__malformed_writer_lock__');
	assert.equal(pairedState.activePublicWriterLock()?.kind, 'malformed');
	await rm(malformedWriterLockPath, { force: true });

	await assert.rejects(
		() => scorePairedRoute.POST({
			request: new Request('http://localhost/api/runs/score-paired', {
				method: 'POST',
				headers: { 'content-type': 'application/json' },
				body: JSON.stringify({
					path: pairedInputFile,
					source_dump_id: 'cycle85_pair',
					model: 'test-model',
					scorer_version: 'test-v1',
					paired_run_group_id: 'cycle85_missing_score_run_table',
					monolithic_cost_threshold_usd: 0.01,
					decomposed_cost_threshold_usd: 0.01,
					total_cost_threshold_usd: 0.02
				})
			})
		}),
		(err) => {
			assert.equal(err?.status, 503);
			assert.equal(err?.body?.code, 'corrupt_corpus_schema');
			assert.match(errorMessage(err), /score_run/);
			return true;
		}
	);

	pairedState.createPairedWorkflowState({
		pair_id: 'cycle85_active_pair',
		source_dump_id: 'cycle85_pair',
		dataset_path: pairedInputFile,
		model: 'test-model',
		scorer_version: 'test-v1',
		total_cost_threshold_usd: 0.02,
		caps: { monolithic: 0.01, decomposed: 0.01 }
	});
	await writeFile(malformedWriterLockPath, '{not-json');
	for (const [label, handler, url, body] of writerActionCases()) {
		await assert.rejects(
			() => handler(postEvent(url, body)),
			(err) => {
				assert.equal(err?.status, 409, String(label));
				assert.equal(err?.body?.code, 'writer_lock_malformed', String(label));
				assert.match(errorMessage(err), /writer_lock\.json/, String(label));
				return true;
			},
			`${label} mixed malformed lock and active pair`
		);
	}
	await assertWriterActionHttpConflict('writer_lock_malformed', /writer_lock\.json/);
	await rm(malformedWriterLockPath, { force: true });
	for (const [label, handler, url, body] of writerActionCases()) {
		await assert.rejects(
			() => handler(postEvent(url, body)),
			(err) => {
				assert.equal(err?.status, 409, String(label));
				assert.equal(err?.body?.code, 'paired_workflow_active', String(label));
				assert.match(errorMessage(err), /cycle85_active_pair/, String(label));
				assert.doesNotMatch(errorMessage(err), /score_run/i, String(label));
				return true;
			},
			String(label)
		);
	}
	await assertWriterActionHttpConflict('paired_workflow_active', /cycle85_active_pair/);
	await assert.rejects(
		() => scorePairedRoute.POST({
			request: new Request('http://localhost/api/runs/score-paired', {
				method: 'POST',
				headers: { 'content-type': 'application/json' },
				body: JSON.stringify({
					path: pairedInputFile,
					source_dump_id: 'cycle85_pair',
					model: 'test-model',
					scorer_version: 'test-v1',
					paired_run_group_id: 'cycle85_new_pair',
					monolithic_cost_threshold_usd: 0.01,
					decomposed_cost_threshold_usd: 0.01,
					total_cost_threshold_usd: 0.02
				})
			})
			}),
			(err) => {
				assert.equal(err?.status, 409);
				assert.equal(err?.body?.code, 'paired_workflow_active');
			assert.match(errorMessage(err), /cycle85_active_pair/);
			assert.doesNotMatch(errorMessage(err), /score_run/i);
			return true;
		}
	);
	pairedState.updatePairedWorkflowState('cycle85_active_pair', (state) => ({
		...state,
		status: 'succeeded',
		finished_at: '2026-05-25T12:10:00.000Z',
		architectures: {
			monolithic: {
				...state.architectures.monolithic,
				status: 'succeeded',
				finished_at: '2026-05-25T12:05:00.000Z'
			},
			decomposed: {
				...state.architectures.decomposed,
				status: 'succeeded',
				finished_at: '2026-05-25T12:10:00.000Z'
			}
		}
	}));
	pairedState.createPairedWorkflowState({
		pair_id: 'cycle86_failed_queued_pair',
		source_dump_id: 'cycle85_pair',
		dataset_path: pairedInputFile,
		model: 'test-model',
		scorer_version: 'test-v1',
		total_cost_threshold_usd: 0.02,
		caps: { monolithic: 0.01, decomposed: 0.01 }
	});
	const terminalBlockedState = pairedState.updatePairedWorkflowState('cycle86_failed_queued_pair', (state) => ({
		...state,
		status: 'failed',
		finished_at: '2026-05-25T12:11:00.000Z',
		termination_reason: 'writer lock blocked before decomposed started',
		architectures: {
			...state.architectures,
			monolithic: {
				...state.architectures.monolithic,
				status: 'blocked',
				error: 'writer lock blocked before worker spawn',
				finished_at: '2026-05-25T12:11:00.000Z'
			}
		}
	}));
	assert.ok(terminalBlockedState);
	assert.equal(pairedState.pairedWorkflowIsActive(terminalBlockedState), false);
	assert.equal(
		pairedState.activePairedWorkflowStates().some((state) => state.pair_id === 'cycle86_failed_queued_pair'),
		false
	);
	await assert.rejects(
		() => scorePairedRoute.POST({
			request: new Request('http://localhost/api/runs/score-paired', {
				method: 'POST',
				headers: { 'content-type': 'application/json' },
				body: JSON.stringify({
					path: pairedInputFile,
					source_dump_id: 'cycle85_pair',
					model: 'test-model',
					scorer_version: 'test-v1',
					paired_run_group_id: 'cycle85_active_pair',
					monolithic_cost_threshold_usd: 0.01,
					decomposed_cost_threshold_usd: 0.01,
					total_cost_threshold_usd: 0.02
				})
			})
			}),
			(err) => {
				assert.equal(err?.status, 409);
				assert.equal(err?.body?.code, 'paired_workflow_state_exists');
				assert.match(errorMessage(err), /paired workflow state already exists/);
			assert.doesNotMatch(errorMessage(err), /score_run/i);
			return true;
		}
	);

	const runInstance = await DuckDBInstance.create(dbFile);
	const runCon = await runInstance.connect();
	try {
		await runCon.run(`
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
		await runCon.run(`
			INSERT INTO score_run VALUES (
				'${RUN_ID}',
				'test',
				'decomposed',
				NULL,
				TIMESTAMP '2026-05-25 12:00:00',
				'succeeded',
				NULL,
				NULL,
				1,
				0.01
			)
		`);
		await runCon.run(`
			INSERT INTO score_run VALUES (
				'cccccccccccccccccccccccccccccccc',
				'test',
				'monolithic',
				'cycle85_taken_pair',
				TIMESTAMP '2026-05-25 12:01:00',
				'succeeded',
				NULL,
				NULL,
				1,
				0.01
			)
		`);
	} finally {
		runCon.disconnectSync?.();
		runInstance.closeSync();
	}

	await assert.rejects(
		() => scorePairedRoute.POST({
			request: new Request('http://localhost/api/runs/score-paired', {
				method: 'POST',
				headers: { 'content-type': 'application/json' },
				body: JSON.stringify({
					path: pairedInputFile,
					source_dump_id: 'cycle85_pair',
					model: 'test-model',
					scorer_version: 'test-v1',
					paired_run_group_id: 'cycle85_taken_pair',
					monolithic_cost_threshold_usd: 0.01,
					decomposed_cost_threshold_usd: 0.01,
					total_cost_threshold_usd: 0.02
				})
			})
		}),
		(err) => {
			assert.equal(err?.status, 409);
			assert.equal(err?.body?.code, 'paired_group_id_taken');
			assert.match(errorMessage(err), /fresh pair id/);
			return true;
		}
	);

	await assert.rejects(
		() => db.getCorpusOverview(),
		(err) => assertCodedError(err, 'corrupt_corpus_schema', /metric/i)
	);
	await assert.rejects(
		() => route.load({ url: new URL('http://localhost/') }),
		(err) => assertCodedError(err, 'corrupt_corpus_schema', /metric/i)
	);
	const corruptResponse = await fetch(baseUrl);
	assert.equal(corruptResponse.status, 500);
	const corruptHtml = await corruptResponse.text();
	assert.match(corruptHtml, /corpus schema mismatch/);
	assert.doesNotMatch(corruptHtml, /writer in progress/);
	assert.doesNotMatch(corruptHtml, /\b\d+\s+statements?\b/i);
	db.closeInstance();

	const updateInstance = await DuckDBInstance.create(dbFile);
	const updateCon = await updateInstance.connect();
	try {
		await updateCon.run(`UPDATE score_run SET status='running' WHERE run_id='${RUN_ID}'`);
	} finally {
		updateCon.disconnectSync?.();
		updateInstance.closeSync();
	}

	await assert.rejects(
		() => db.getCorpusOverview(),
		(err) => assertCodedError(err, 'corrupt_corpus_schema', /metric/i)
	);
	db.closeInstance();

	const coreBrokenInstance = await DuckDBInstance.create(dbFile);
	const coreBrokenCon = await coreBrokenInstance.connect();
	try {
		await coreBrokenCon.run(`
			CREATE TABLE metric (
				run_id VARCHAR,
				truth_set_id VARCHAR,
				metric_name VARCHAR,
				value DOUBLE,
				slice_json JSON
			)
		`);
		await coreBrokenCon.run('DROP TABLE statement');
	} finally {
		coreBrokenCon.disconnectSync?.();
		coreBrokenInstance.closeSync();
	}

	await assert.rejects(
		() => db.getCorpusOverview(),
		(err) => assertCodedError(err, 'corrupt_corpus_schema', /statement/i)
	);
	await assert.rejects(
		() => route.load({ url: new URL('http://localhost/') }),
		(err) => assertCodedError(err, 'corrupt_corpus_schema', /statement/i)
	);
	db.closeInstance();
	console.log('latest validity query-error tests passed');
} finally {
	await new Promise((resolve) => httpServer.close(resolve));
	await server.close();
	await rm(dir, { recursive: true, force: true });
}
