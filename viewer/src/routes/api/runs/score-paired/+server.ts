/**
 * POST /api/runs/score-paired
 *
 * Runs monolithic and decomposed scorers under one paired_run_group_id.
 * The endpoint owns serial execution so DuckDB has only one writer process
 * at a time, while the client receives a single SSE stream with visible
 * queued/running/done/canceled/crashed states per architecture.
 */
import { spawn, type ChildProcessWithoutNullStreams } from 'node:child_process';
import { randomUUID } from 'node:crypto';
import { existsSync } from 'node:fs';
import { resolve } from 'node:path';
import { error } from '@sveltejs/kit';
import { closeInstance, connect, dbExists, dbPath } from '$lib/db';
import { PAIRED_RUN_CONFLICT } from '$lib/pairedRunConflicts';
import { assertPathUnderData } from '$lib/pathGuard';
import {
	activePairedWorkflowStates,
	activeWriterLock,
	acquireWriterLock,
	clearWriterLockToken,
	createPairedWorkflowState,
	pairedWorkflowStateFileExists,
	readPairedWorkflowState,
	updatePairedWorkflowState,
	updateWriterLock,
	writerLockConflictText
} from '$lib/server/pairedState';
import type { WriterLockState } from '$lib/server/pairedState';
import { terminateChildProcessWithEscalation } from '$lib/server/childProcess';
import { markScoreRunCanceled } from '$lib/server/scoreRunLifecycle';
import type { RequestHandler } from './$types';

type Architecture = 'monolithic' | 'decomposed';

const SOURCE_DUMP_RE = /^[a-z][a-z0-9_-]{1,63}$/i;
const MODEL_RE = /^[a-z0-9][a-z0-9_.\-/:]{1,63}$/i;
const SCORER_VERSION_RE = /^[a-z0-9][a-z0-9_.\-]{1,63}$/i;
const PAIRED_RUN_GROUP_RE = /^[a-z0-9][a-z0-9_.\-]{1,63}$/i;
const ARCH_ORDER: Architecture[] = ['monolithic', 'decomposed'];

function workflowConflict(code: string, message: string): never {
	throw error(409, { code, message });
}

function writerLockConflictCode(lock: WriterLockState): string {
	return lock.malformed_reason
		? PAIRED_RUN_CONFLICT.writerLockMalformed
		: PAIRED_RUN_CONFLICT.writerInProgress;
}

function pythonBin(): string {
	if (process.env.PYTHON_BIN) return process.env.PYTHON_BIN;
	const repoRoot = resolve(process.cwd(), '..');
	const venv = resolve(repoRoot, '.venv', 'bin', 'python');
	if (existsSync(venv)) return venv;
	return 'python3';
}

function repoRoot(): string {
	return resolve(process.cwd(), '..');
}

function numberField(body: Record<string, unknown>, name: string): number | null {
	const v = body[name];
	return typeof v === 'number' && Number.isFinite(v) ? v : null;
}

function pairId(): string {
	return `pair_${new Date().toISOString().replace(/[-:TZ.]/g, '').slice(0, 14)}_${randomUUID().replace(/-/g, '').slice(0, 8)}`;
}

async function pairedGroupExists(id: string): Promise<boolean> {
	if (!dbExists()) return false;
	const con = await connect();
	try {
		const reader = await con.runAndReadAll(
			`SELECT COUNT(*) AS n FROM score_run WHERE paired_run_group_id='${id.replace(/'/g, "''")}'`
		);
		const rows = reader.getRowObjects();
		return Number(rows[0]?.n ?? 0) > 0;
	} catch (e) {
		const maybeHttp = e as { status?: unknown; body?: unknown };
		if (typeof maybeHttp.status === 'number' && maybeHttp.body) throw e;
		const message = (e as Error)?.message ?? String(e);
		if (/Table with name score_run does not exist/i.test(message)) {
			throw error(503, {
				code: 'corrupt_corpus_schema',
				message: 'corrupt_corpus_schema: missing required DuckDB table score_run; paired run identity cannot be checked safely'
			});
		}
		throw error(503, {
			code: 'paired_group_check_failed',
			message: `paired_group_check_failed: could not check paired_run_group_id before scoring: ${message.slice(0, 200)}`
		});
	} finally {
		con.disconnectSync?.();
	}
}

export const POST: RequestHandler = async (event) => {
	const request = event.request;
	const body = (await request.json()) as Record<string, unknown>;
	const path = body.path as string | undefined;
	const source_dump_id = body.source_dump_id as string | undefined;
	const model = body.model as string | undefined;
	const scorer_version = body.scorer_version as string | undefined;
	const requestedPair = body.paired_run_group_id as string | undefined;
	const monolithic_cap = numberField(body, 'monolithic_cost_threshold_usd');
	const decomposed_cap = numberField(body, 'decomposed_cost_threshold_usd');
	const total_cap = numberField(body, 'total_cost_threshold_usd');

	const safePath = assertPathUnderData(path);
	if (!source_dump_id || !SOURCE_DUMP_RE.test(source_dump_id))
		throw error(400, 'source_dump_id must match /^[a-z][a-z0-9_-]{1,63}$/i');
	if (!model || !MODEL_RE.test(model))
		throw error(400, 'model required (and must be safe shell-token)');
	if (!scorer_version || !SCORER_VERSION_RE.test(scorer_version))
		throw error(400, 'scorer_version required');
	if (requestedPair != null && !PAIRED_RUN_GROUP_RE.test(requestedPair))
		throw error(400, 'paired_run_group_id must be a safe token');
	if (monolithic_cap == null || monolithic_cap <= 0)
		throw error(400, 'monolithic_cost_threshold_usd must be positive');
	if (decomposed_cap == null || decomposed_cap <= 0)
		throw error(400, 'decomposed_cost_threshold_usd must be positive');
	if (total_cap == null || total_cap <= 0)
		throw error(400, 'total_cost_threshold_usd must be positive');
	if (monolithic_cap + decomposed_cap > total_cap + 1e-9)
		throw error(400, 'per-architecture caps must not exceed total_cost_threshold_usd');

	const activePair = activePairedWorkflowStates()[0] ?? null;
	if (activePair) {
		workflowConflict(
			PAIRED_RUN_CONFLICT.pairedWorkflowActive,
			`paired workflow ${activePair.pair_id} is already ${activePair.status}; wait, cancel it, or inspect ${activePair.href}`
		);
	}
	const activeLock = activeWriterLock();
	if (activeLock) {
		workflowConflict(writerLockConflictCode(activeLock), writerLockConflictText(activeLock));
	}
	const paired_run_group_id = requestedPair ?? pairId();
	if (pairedWorkflowStateFileExists(paired_run_group_id)) {
		workflowConflict(
			PAIRED_RUN_CONFLICT.pairedWorkflowStateExists,
			'paired workflow state already exists; use a fresh pair id or inspect the existing pair'
		);
	}
	if (await pairedGroupExists(paired_run_group_id)) {
		workflowConflict(
			PAIRED_RUN_CONFLICT.pairedGroupIdTaken,
			'paired_run_group_id already exists; use a fresh pair id for append-only history'
		);
	}
	let launchWriterLock: WriterLockState | null = acquireWriterLock({
		kind: 'paired_score',
		label: 'paired workflow launch',
		pair_id: paired_run_group_id,
		source_dump_id,
		dataset_path: safePath,
		architecture: null,
		model,
		pid: process.pid
	});
	if (!launchWriterLock) {
		const lock = activeWriterLock();
		workflowConflict(
			lock ? writerLockConflictCode(lock) : PAIRED_RUN_CONFLICT.writerInProgress,
			lock ? writerLockConflictText(lock) : 'DuckDB writer lock is busy'
		);
	}
	const activePairAfterLaunchLock = activePairedWorkflowStates()[0] ?? null;
	if (activePairAfterLaunchLock) {
		clearWriterLockToken(launchWriterLock.token);
		launchWriterLock = null;
		workflowConflict(
			PAIRED_RUN_CONFLICT.pairedWorkflowActive,
			`paired workflow ${activePairAfterLaunchLock.pair_id} is already ${activePairAfterLaunchLock.status}; wait, cancel it, or inspect ${activePairAfterLaunchLock.href}`
		);
	}
	const caps: Record<Architecture, number> = {
		monolithic: monolithic_cap,
		decomposed: decomposed_cap
	};
	try {
		createPairedWorkflowState({
			pair_id: paired_run_group_id,
			source_dump_id,
			dataset_path: safePath,
			model,
			scorer_version,
			total_cost_threshold_usd: total_cap,
			caps
			});
	} catch (e) {
		clearWriterLockToken(launchWriterLock.token);
		launchWriterLock = null;
		workflowConflict(
			PAIRED_RUN_CONFLICT.pairedWorkflowStateExists,
			(e as Error).message || 'paired workflow state already exists'
		);
	}
	const py = pythonBin();
	const root = repoRoot();

	let cancelFromStream: (() => void) | null = null;
	const stream = new ReadableStream<Uint8Array>({
		start(controller) {
			const encoder = new TextEncoder();
			let closed = false;
			let activeChild: ChildProcessWithoutNullStreams | null = null;
			let aborted = false;
			const terminalBlock: { current: { code: string; message: string } | null } = { current: null };
			let stopActiveWriterHeartbeat: (() => void) | null = null;
			const runIds: Partial<Record<Architecture, string>> = {};

			const writeEvent = (obj: unknown) => {
				if (closed) return;
				try {
					controller.enqueue(encoder.encode(`data: ${JSON.stringify(obj)}\n\n`));
				} catch {
					closed = true;
				}
			};
			let reservedWriterLock: WriterLockState | null = launchWriterLock;
			const releaseReservedWriterLock = () => {
				if (!reservedWriterLock) return;
				clearWriterLockToken(reservedWriterLock.token);
				reservedWriterLock = null;
				launchWriterLock = null;
			};
			const cleanup = () => {
				if (closed) return;
				closed = true;
				stopActiveWriterHeartbeat?.();
				stopActiveWriterHeartbeat = null;
				request.signal.removeEventListener('abort', onAbort);
				try { controller.close(); } catch { /* stream already closed */ }
			};
			const onAbort = () => {
				if (aborted) return;
				aborted = true;
				updatePairedWorkflowState(paired_run_group_id, (s) => {
					for (const arch of ARCH_ORDER) {
						const a = s.architectures[arch];
						if (a.status !== 'succeeded') {
							s.architectures[arch] = {
								...a,
								status: 'canceled',
								finished_at: new Date().toISOString(),
								error: 'client_disconnected'
							};
						}
					}
					return {
						...s,
						status: 'canceled',
						finished_at: new Date().toISOString(),
						termination_reason: 'client_disconnected'
					};
				});
				if (activeChild) {
					terminateChildProcessWithEscalation(activeChild);
				} else {
					releaseReservedWriterLock();
				}
				for (const arch of ARCH_ORDER) {
					if (!runIds[arch]) {
						writeEvent({ event: 'arch_state', architecture: arch, state: 'canceled', reason: 'client_disconnected' });
					}
				}
				writeEvent({ event: 'paired_canceled', paired_run_group_id, reason: 'client_disconnected' });
				cleanup();
			};
			cancelFromStream = onAbort;
			request.signal.addEventListener('abort', onAbort);
			if (request.signal.aborted) onAbort();

				const runArch = (architecture: Architecture): Promise<boolean> => new Promise((resolveP) => { (async () => {
					// Must stay before spawn: onAbort can run between serial
					// architecture lanes when activeChild is null.
					if (aborted) {
						releaseReservedWriterLock();
						resolveP(false);
						return;
					}
					const writerLockInput = {
						kind: 'paired_score',
						label: `paired ${architecture} score`,
						pair_id: paired_run_group_id,
						source_dump_id,
						dataset_path: safePath,
						architecture,
						model,
						pid: null
					} as const;
					let writerLock: WriterLockState | null = null;
					if (reservedWriterLock) {
						const reserved = reservedWriterLock;
						reservedWriterLock = null;
						launchWriterLock = null;
						writerLock = updateWriterLock(reserved.token, writerLockInput);
						if (!writerLock) {
							writerLock = acquireWriterLock(writerLockInput);
						}
					} else {
						writerLock = acquireWriterLock(writerLockInput);
					}
					if (!writerLock) {
						const lock = activeWriterLock();
						const message = lock ? writerLockConflictText(lock) : 'DuckDB writer lock is busy';
						const code = lock ? writerLockConflictCode(lock) : PAIRED_RUN_CONFLICT.writerInProgress;
						const skippedMessage = `not started because paired workflow was blocked: ${message}`;
						const blockedReasons: Partial<Record<Architecture, string>> = {};
						terminalBlock.current = { code, message };
						updatePairedWorkflowState(paired_run_group_id, (s) => {
							const at = new Date().toISOString();
							const architectures = { ...s.architectures };
							for (const arch of ARCH_ORDER) {
								const current = architectures[arch];
								if (current.status === 'succeeded' || current.status === 'failed' || current.status === 'canceled') continue;
								const reason = arch === architecture ? message : skippedMessage;
								blockedReasons[arch] = reason;
								architectures[arch] = {
									...current,
									status: 'blocked',
									error: reason,
									finished_at: at,
									updated_at: at
								};
							}
							return {
								...s,
								status: 'failed',
								finished_at: at,
								termination_reason: message,
								architectures
							};
						});
						for (const arch of ARCH_ORDER) {
							const reason = blockedReasons[arch];
							if (!reason) continue;
							writeEvent({
								event: 'arch_state',
								architecture: arch,
								state: 'blocked',
								code,
								reason
							});
						}
						resolveP(false);
						return;
					}
					if (aborted) {
						clearWriterLockToken(writerLock.token);
						resolveP(false);
						return;
					}
					const runId = randomUUID().replace(/-/g, '');
					runIds[architecture] = runId;
					const args = [
						'-m', 'indra_belief.worker',
						'score',
						'--db', dbPath(),
						'--path', safePath,
						'--source-dump-id', source_dump_id,
						'--model', model,
						'--scorer-version', scorer_version,
						'--arch', architecture,
						'--paired-run-group-id', paired_run_group_id,
						'--run-id', runId,
						'--cost-threshold-usd', caps[architecture].toString()
					];
					updatePairedWorkflowState(paired_run_group_id, (s) => {
						const at = new Date().toISOString();
						return {
							...s,
							status: 'running',
							started_at: s.started_at ?? at,
							architectures: {
								...s.architectures,
								[architecture]: {
									...s.architectures[architecture],
									status: 'loading',
									run_id: s.architectures[architecture].run_id ?? runId,
									started_at: s.architectures[architecture].started_at ?? at,
									updated_at: at
								}
							}
						};
					});
					writeEvent({
						event: 'arch_state',
						architecture,
						state: 'running',
						cost_threshold_usd: caps[architecture]
					});
					try {
						await closeInstance();
					} catch (err) {
						const message = err instanceof Error ? err.message : String(err);
						clearWriterLockToken(writerLock.token);
						updatePairedWorkflowState(paired_run_group_id, (s) => ({
							...s,
							status: 'failed',
							architectures: {
								...s.architectures,
								[architecture]: {
									...s.architectures[architecture],
									status: 'failed',
									pid: null,
									finished_at: new Date().toISOString(),
									error: message,
									updated_at: new Date().toISOString()
								}
							}
						}));
						writeEvent({ event: 'arch_state', architecture, state: 'crashed', error: message });
						resolveP(false);
						return;
					}
					activeChild = spawn(py, args, {
					cwd: root,
					env: {
						...process.env,
						PYTHONPATH: resolve(root, 'src'),
						INDRA_VIEWER_WRITER_LOCK_TOKEN: writerLock.token
					}
				});
				let writerHeartbeat: ReturnType<typeof setInterval> | null = null;
				const stopWriterHeartbeat = () => {
					if (writerHeartbeat) clearInterval(writerHeartbeat);
					writerHeartbeat = null;
				};
				stopActiveWriterHeartbeat = stopWriterHeartbeat;
				const childPid = activeChild.pid ?? null;
				if (childPid != null) {
					const at = new Date().toISOString();
					updateWriterLock(writerLock.token, { pid: childPid, updated_at: at });
					writerHeartbeat = setInterval(() => {
						updateWriterLock(writerLock.token, {});
					}, 5000);
					writerHeartbeat.unref?.();
					updatePairedWorkflowState(paired_run_group_id, (s) => ({
						...s,
						architectures: {
							...s.architectures,
							[architecture]: {
								...s.architectures[architecture],
								run_id: s.architectures[architecture].run_id ?? runId,
								pid: childPid,
								updated_at: at
							}
						}
					}));
				}

				let stdoutBuf = '';
				let stderrBuf = '';
				const STDERR_CAP = 64 * 1024;
				let sawDone = false;

				activeChild.stdout.on('data', (chunk: Buffer) => {
					stdoutBuf += chunk.toString('utf-8');
					let nl: number;
					while ((nl = stdoutBuf.indexOf('\n')) >= 0) {
						const line = stdoutBuf.slice(0, nl).trim();
						stdoutBuf = stdoutBuf.slice(nl + 1);
						if (!line) continue;
						let workerEvent: Record<string, unknown>;
						try {
							workerEvent = JSON.parse(line);
						} catch {
							writeEvent({ event: 'arch_stdout_raw', architecture, line });
							continue;
						}
						const t = workerEvent.event;
						if (t === 'started' && typeof workerEvent.run_id === 'string') {
							runIds[architecture] = String(workerEvent.run_id);
							updatePairedWorkflowState(paired_run_group_id, (s) => ({
								...s,
								architectures: {
									...s.architectures,
									[architecture]: {
										...s.architectures[architecture],
										run_id: String(workerEvent.run_id),
										updated_at: new Date().toISOString()
									}
								}
							}));
							writeEvent({ event: 'arch_started', architecture, run_id: workerEvent.run_id });
						} else if (t === 'loaded') {
							updatePairedWorkflowState(paired_run_group_id, (s) => ({
								...s,
								status: 'running',
								architectures: {
									...s.architectures,
									[architecture]: {
										...s.architectures[architecture],
										status: 'running',
										n_evidences_total: typeof workerEvent.n_evidences === 'number' ? workerEvent.n_evidences : null,
										updated_at: new Date().toISOString()
									}
								}
							}));
							writeEvent({
								event: 'arch_loaded',
								architecture,
								n_statements: workerEvent.n_statements,
								n_evidences: workerEvent.n_evidences
							});
						} else if (t === 'ingested') {
							writeEvent({ event: 'arch_ingested', architecture });
						} else if (t === 'progress') {
							updatePairedWorkflowState(paired_run_group_id, (s) => ({
								...s,
								architectures: {
									...s.architectures,
									[architecture]: {
										...s.architectures[architecture],
										status: 'running',
										n_evidences_done: Number(workerEvent.n_evidences_done ?? 0),
										cost_so_far_usd: typeof workerEvent.cost_so_far_usd === 'number' ? workerEvent.cost_so_far_usd : s.architectures[architecture].cost_so_far_usd,
										latest_stmt_hash: typeof workerEvent.latest_stmt_hash === 'string' ? workerEvent.latest_stmt_hash : null,
										updated_at: new Date().toISOString()
									}
								}
							}));
							writeEvent({
								event: 'arch_progress',
								architecture,
								n_evidences_done: workerEvent.n_evidences_done,
								cost_so_far_usd: workerEvent.cost_so_far_usd,
								cost_cap_usd: workerEvent.cost_cap_usd,
								latest_stmt_hash: workerEvent.latest_stmt_hash
							});
						} else if (t === 'done') {
							sawDone = true;
							runIds[architecture] = String(workerEvent.run_id);
							updatePairedWorkflowState(paired_run_group_id, (s) => ({
								...s,
								architectures: {
									...s.architectures,
									[architecture]: {
										...s.architectures[architecture],
										status: 'succeeded',
										pid: null,
										run_id: String(workerEvent.run_id),
										n_evidences_done: Number(workerEvent.n_evidences_done ?? 0),
										duration_s: typeof workerEvent.duration_s === 'number' ? workerEvent.duration_s : null,
										finished_at: new Date().toISOString(),
										error: null,
										updated_at: new Date().toISOString()
									}
								}
							}));
							writeEvent({
								event: 'arch_done',
								architecture,
								run_id: workerEvent.run_id,
								paired_run_group_id,
								n_statements: workerEvent.n_statements,
								n_evidences_done: workerEvent.n_evidences_done,
								duration_s: workerEvent.duration_s
							});
						} else {
							writeEvent({ event: 'arch_worker_event', architecture, worker_event: workerEvent });
						}
					}
				});
				activeChild.stderr.on('data', (chunk: Buffer) => {
					if (stderrBuf.length >= STDERR_CAP) return;
					stderrBuf += chunk.toString('utf-8');
					if (stderrBuf.length > STDERR_CAP) {
						stderrBuf = stderrBuf.slice(0, STDERR_CAP) + '\n...[stderr truncated]';
					}
				});
				activeChild.on('exit', (code, signal) => {
					activeChild = null;
					stopWriterHeartbeat();
					if (stopActiveWriterHeartbeat === stopWriterHeartbeat) stopActiveWriterHeartbeat = null;
					const canceled = aborted || signal === 'SIGTERM' || signal === 'SIGKILL';
					const releaseWriterLock = () => {
						clearWriterLockToken(writerLock.token);
					};
					if (canceled) {
						const runId = runIds[architecture] ?? null;
						const currentState = readPairedWorkflowState(paired_run_group_id);
						const cancelReason = aborted
							? 'client_disconnected'
							: currentState?.termination_reason ?? 'worker_terminated';
						const tombstone = runId
							? markScoreRunCanceled({
								run_id: runId,
								scorer_version,
								architecture,
								model,
								paired_run_group_id,
								reason: cancelReason
							}).catch((err) => {
								const message = err instanceof Error ? err.message : String(err);
								console.error('arch_cancel_tombstone_failed', err);
								updatePairedWorkflowState(paired_run_group_id, (s) => ({
									...s,
									architectures: {
										...s.architectures,
										[architecture]: {
											...s.architectures[architecture],
											error: `cancel_tombstone_failed: ${message}`,
											updated_at: new Date().toISOString()
										}
									}
								}));
								writeEvent({
									event: 'arch_cancel_tombstone_failed',
									architecture,
									run_id: runId,
									error: message
								});
							})
							: Promise.resolve();
						void tombstone.finally(() => {
							releaseWriterLock();
							updatePairedWorkflowState(paired_run_group_id, (s) => ({
								...s,
								status: 'canceled',
								termination_reason: cancelReason,
								finished_at: new Date().toISOString(),
								architectures: {
									...s.architectures,
									[architecture]: {
										...s.architectures[architecture],
										status: 'canceled',
										pid: null,
										finished_at: new Date().toISOString(),
										error: s.architectures[architecture].error ?? cancelReason,
										updated_at: new Date().toISOString()
									}
								}
							}));
							writeEvent({ event: 'arch_state', architecture, state: 'canceled', signal, reason: cancelReason });
							resolveP(false);
						});
						return;
					}
					releaseWriterLock();
					if (code !== 0 || !sawDone) {
						updatePairedWorkflowState(paired_run_group_id, (s) => ({
							...s,
							status: 'failed',
							architectures: {
								...s.architectures,
								[architecture]: {
									...s.architectures[architecture],
									status: 'failed',
									pid: null,
									finished_at: new Date().toISOString(),
									error: stderrBuf.slice(0, 4096) || `exit ${code ?? 'unknown'}`,
									updated_at: new Date().toISOString()
								}
							}
						}));
						writeEvent({
							event: 'arch_state',
							architecture,
							state: 'crashed',
							exit_code: code,
							signal,
							stderr: stderrBuf.slice(0, 4096)
						});
						resolveP(false);
						return;
					}
					writeEvent({ event: 'arch_state', architecture, state: 'done', run_id: runIds[architecture] });
					resolveP(true);
				});
				activeChild.on('error', (err) => {
					activeChild = null;
					stopWriterHeartbeat();
					if (stopActiveWriterHeartbeat === stopWriterHeartbeat) stopActiveWriterHeartbeat = null;
					clearWriterLockToken(writerLock.token);
					updatePairedWorkflowState(paired_run_group_id, (s) => ({
						...s,
						status: 'failed',
						architectures: {
							...s.architectures,
							[architecture]: {
								...s.architectures[architecture],
								status: 'failed',
								pid: null,
								finished_at: new Date().toISOString(),
								error: err.message,
								updated_at: new Date().toISOString()
							}
						}
					}));
					writeEvent({ event: 'arch_state', architecture, state: 'crashed', error: err.message });
					resolveP(false);
				});
			})().catch((err) => {
				console.error('runArch unexpected error', err);
				resolveP(false);
			}); });

			(async () => {
				writeEvent({
					event: 'paired_started',
					paired_run_group_id,
					model,
					scorer_version,
					order: ARCH_ORDER,
					total_cost_threshold_usd: total_cap
				});
				for (const arch of ARCH_ORDER) {
					writeEvent({ event: 'arch_state', architecture: arch, state: 'queued', cost_threshold_usd: caps[arch] });
					}

					let allOk = true;
					for (const arch of ARCH_ORDER) {
						const ok = await runArch(arch);
						if (!ok) {
							allOk = false;
							const currentState = readPairedWorkflowState(paired_run_group_id);
							if (aborted || currentState?.status === 'canceled') {
								writeEvent({
									event: 'paired_canceled',
									paired_run_group_id,
									monolithic_run_id: runIds.monolithic,
									decomposed_run_id: runIds.decomposed,
									reason: currentState?.termination_reason ?? 'client_disconnected',
									href: `/pairs/${paired_run_group_id}`
								});
								writeEvent({ event: 'channel_closed' });
								cleanup();
								return;
							}
							if (terminalBlock.current) break;
						}
					}

					if (!allOk) {
						updatePairedWorkflowState(paired_run_group_id, (s) => ({
							...s,
							status: 'failed',
							finished_at: new Date().toISOString(),
							termination_reason: terminalBlock.current?.message ?? 'one_or_more_architectures_failed'
						}));
						writeEvent({
							event: 'paired_failed',
							paired_run_group_id,
							monolithic_run_id: runIds.monolithic,
							decomposed_run_id: runIds.decomposed,
							code: terminalBlock.current?.code,
							error: terminalBlock.current?.message,
							href: `/pairs/${paired_run_group_id}`
						});
						writeEvent({ event: 'channel_closed' });
						cleanup();
						return;
					}
				updatePairedWorkflowState(paired_run_group_id, (s) => ({
					...s,
					status: 'succeeded',
					finished_at: new Date().toISOString(),
					termination_reason: null
				}));
				writeEvent({
					event: 'paired_done',
					paired_run_group_id,
					monolithic_run_id: runIds.monolithic,
					decomposed_run_id: runIds.decomposed,
					href: `/pairs/${paired_run_group_id}`
				});
				writeEvent({ event: 'channel_closed' });
				cleanup();
			})().catch((err) => {
				updatePairedWorkflowState(paired_run_group_id, (s) => ({
					...s,
					status: 'failed',
					finished_at: new Date().toISOString(),
					termination_reason: (err as Error).message
				}));
				writeEvent({ event: 'paired_failed', paired_run_group_id, error: (err as Error).message });
				writeEvent({ event: 'channel_closed' });
				cleanup();
			});
		},
		cancel() {
			if (cancelFromStream) cancelFromStream();
		}
	});

	return new Response(stream, {
		headers: {
			'content-type': 'text/event-stream',
			'cache-control': 'no-cache, no-transform',
			'x-accel-buffering': 'no'
		}
	});
};
