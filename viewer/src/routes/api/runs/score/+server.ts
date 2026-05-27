/**
 * POST /api/runs/score — ingest (idempotent) + score a corpus end-to-end.
 *
 * Spawns `python -m indra_belief.worker score ...` and streams stdout events
 * as text/event-stream (SSE) so the viewer can render per-evidence progress.
 *
 * Body:
 *   {
 *     path: string,
 *     source_dump_id: string,
 *     model: string,                  // e.g. "claude-sonnet-4-6"
 *     scorer_version: string,         // e.g. "prod-v1"
 *     arch?: "decomposed" | "monolithic",
 *     parent_run_id?: string,         // repair/rerun baseline
 *     skip_ingest?: boolean,          // true for DB-derived repair subsets
 *     cost_threshold_usd?: number     // hard cap; worker aborts above this
 *   }
 *
 * Returns: SSE stream. Each event is one of:
 *   data: {"event": "started", ...}
 *   data: {"event": "loaded", "n_statements": N}
 *   data: {"event": "ingested"}
 *   data: {"event": "progress", "n_evidences_done": N, "latest_stmt_hash": "..."}
 *   data: {"event": "done", "run_id": "...", ...}
 *   data: {"event": "error", ...}
 * Terminated by `data: {"event": "channel_closed"}` and an end-of-stream.
 */
import { spawn } from 'node:child_process';
import { randomUUID } from 'node:crypto';
import { resolve } from 'node:path';
import { existsSync } from 'node:fs';
import { closeInstance, dbPath } from '$lib/db';
import { assertPathUnderData } from '$lib/pathGuard';
import {
	activePairedWorkflowStates,
	activeWriterLock,
	acquireWriterLock,
	clearWriterLockToken,
	updateWriterLock,
	writerLockConflictPayload
} from '$lib/server/pairedState';
import { WRITER_ACTION_CONFLICT } from '$lib/writerActionConflicts';
import { terminateChildProcessWithEscalation } from '$lib/server/childProcess';
import { recordRepairRerunChildAfterScore, recordRepairRerunIntent } from '$lib/server/repairRerun';
import { markScoreRunCanceled } from '$lib/server/scoreRunLifecycle';
import { error } from '@sveltejs/kit';
import type { RequestHandler } from './$types';

const SOURCE_DUMP_RE = /^[a-z][a-z0-9_-]{1,63}$/i;
const MODEL_RE = /^[a-z0-9][a-z0-9_.\-/:]{1,63}$/i;
const SCORER_VERSION_RE = /^[a-z0-9][a-z0-9_.\-]{1,63}$/i;
const ARCH_RE = /^(decomposed|monolithic)$/;
const PAIRED_RUN_GROUP_RE = /^[a-z0-9][a-z0-9_.\-]{1,63}$/i;
const RUN_ID_RE = /^[a-f0-9]{32}$/i;
const PROBE_STEP_KINDS = new Set([
	'subject_role_probe',
	'object_role_probe',
	'relation_axis_probe',
	'scope_probe'
]);

function positiveIntegerIds(raw: unknown): number[] {
	return Array.isArray(raw)
		? Array.from(new Set(
			raw
				.map((id) => Number(id))
				.filter((id) => Number.isInteger(id) && id > 0)
		))
		: [];
}

function probeStepFilter(raw: unknown): string[] {
	if (!Array.isArray(raw)) return [];
	const out: string[] = [];
	const seen = new Set<string>();
	for (const value of raw) {
		const step = String(value ?? '').trim();
		if (!step) continue;
		if (!PROBE_STEP_KINDS.has(step)) {
			throw error(400, `unknown decomposed probe step: ${step}`);
		}
		if (!seen.has(step)) {
			seen.add(step);
			out.push(step);
		}
	}
	return out;
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

export const POST: RequestHandler = async (event) => {
	const request = event.request;
	const body = (await request.json()) as Record<string, unknown>;
	const path = body.path as string | undefined;
	const source_dump_id = body.source_dump_id as string | undefined;
	const model = body.model as string | undefined;
	const scorer_version = body.scorer_version as string | undefined;
	const arch = (body.arch as string | undefined) ?? 'decomposed';
	const paired_run_group_id = body.paired_run_group_id as string | undefined;
		const parent_run_id = body.parent_run_id as string | undefined;
		const skip_ingest = body.skip_ingest === true;
		const cost_threshold_usd = body.cost_threshold_usd as number | undefined;
		const repair_rerun = body.repair_rerun as Record<string, unknown> | undefined;

	const safePath = assertPathUnderData(path);
	if (!source_dump_id || !SOURCE_DUMP_RE.test(source_dump_id))
		throw error(400, 'source_dump_id must match /^[a-z][a-z0-9_-]{1,63}$/i');
	if (!model || !MODEL_RE.test(model))
		throw error(400, 'model required (and must be safe shell-token)');
	if (!scorer_version || !SCORER_VERSION_RE.test(scorer_version))
		throw error(400, 'scorer_version required');
	if (!ARCH_RE.test(arch))
		throw error(400, 'arch must be decomposed or monolithic');
	if (paired_run_group_id != null && !PAIRED_RUN_GROUP_RE.test(paired_run_group_id))
		throw error(400, 'paired_run_group_id must be a safe token');
		if (parent_run_id != null && !RUN_ID_RE.test(parent_run_id))
			throw error(400, 'parent_run_id must be 32 hex chars');
		if (cost_threshold_usd != null && (typeof cost_threshold_usd !== 'number' || cost_threshold_usd <= 0))
			throw error(400, 'cost_threshold_usd must be positive number');
		let repairRerunCorrectionIds: number[] | null = null;
		let repairRerunProbeStepFilter: string[] = [];
		let repairRerunProbeOnly = false;
		if (repair_rerun != null) {
			if (typeof repair_rerun !== 'object' || Array.isArray(repair_rerun)) {
				throw error(400, 'repair_rerun must be an object');
			}
			repairRerunCorrectionIds = positiveIntegerIds(repair_rerun.correction_ids);
			repairRerunProbeStepFilter = probeStepFilter(repair_rerun.probe_step_filter);
			repairRerunProbeOnly = repair_rerun.probe_only === true;
			if (repairRerunCorrectionIds.length === 0) {
				throw error(400, 'repair_rerun.correction_ids must include at least one positive integer');
			}
			if (!parent_run_id) {
				throw error(400, 'repair_rerun completion requires parent_run_id');
			}
			if (!skip_ingest) {
				throw error(400, 'repair_rerun completion requires skip_ingest=true');
			}
			if (repairRerunProbeStepFilter.length > 0 && arch !== 'decomposed') {
				throw error(400, 'repair_rerun.probe_step_filter is only valid for decomposed repair reruns');
			}
			if (repairRerunProbeOnly && arch !== 'decomposed') {
				throw error(400, 'repair_rerun.probe_only is only valid for decomposed repair reruns');
			}
			if (repairRerunProbeOnly && repairRerunProbeStepFilter.length === 0) {
				throw error(400, 'repair_rerun.probe_only requires probe_step_filter');
			}
		}
	const activeLock = activeWriterLock();
	if (activeLock?.kind === 'malformed') {
		throw error(409, writerLockConflictPayload(activeLock));
	}
	const activePair = activePairedWorkflowStates()[0] ?? null;
	if (activePair) {
		throw error(409, {
			code: WRITER_ACTION_CONFLICT.pairedWorkflowActive,
			message: `paired workflow ${activePair.pair_id} is already ${activePair.status}; wait, cancel it, or inspect ${activePair.href}`
		});
	}
	if (activeLock) {
		throw error(409, writerLockConflictPayload(activeLock));
	}
	const writerLock = acquireWriterLock({
		kind: 'single_score',
		label: `${arch} score`,
		source_dump_id,
		dataset_path: safePath,
		architecture: arch as 'decomposed' | 'monolithic',
		model,
		pid: null
	});
	if (!writerLock) {
		const lock = activeWriterLock();
		throw error(409, writerLockConflictPayload(lock));
	}

	const args = [
		'-m', 'indra_belief.worker',
		'score',
		'--db', dbPath(),
		'--path', safePath,
		'--source-dump-id', source_dump_id,
		'--model', model,
		'--scorer-version', scorer_version,
		'--arch', arch
	];
	if (paired_run_group_id != null) {
		args.push('--paired-run-group-id', paired_run_group_id);
	}
	if (parent_run_id != null) {
		args.push('--parent-run-id', parent_run_id);
	}
	if (skip_ingest) {
		args.push('--skip-ingest');
	}
		if (cost_threshold_usd != null) {
		args.push('--cost-threshold-usd', cost_threshold_usd.toString());
	}
	if (repairRerunProbeStepFilter.length > 0) {
		args.push('--probe-step-filter', repairRerunProbeStepFilter.join(','));
	}
	if (repairRerunProbeOnly) {
		args.push('--probe-only');
	}
		const preassignedRunId = randomUUID().replace(/-/g, '');
		args.push('--run-id', preassignedRunId);

		if (preassignedRunId && parent_run_id && repairRerunCorrectionIds) {
			try {
				await recordRepairRerunIntent({
					parent_run_id,
					child_run_id: preassignedRunId,
					architecture: arch as 'decomposed' | 'monolithic',
					source_dump_id,
					correction_ids: repairRerunCorrectionIds,
					path: safePath,
					scoring_mode: repairRerunProbeOnly ? 'probe_only' : 'aggregate',
					probe_step_filter: repairRerunProbeStepFilter
				});
			} catch (err) {
				clearWriterLockToken(writerLock.token);
				throw error(400, err instanceof Error ? err.message : String(err));
			}
		}

		const py = pythonBin();

	// Release the viewer's cached READ_ONLY DuckDB instance so the worker can
	// acquire the file lock. Next dashboard read will lazy-reopen.
	try {
		closeInstance();
	} catch (e) {
		clearWriterLockToken(writerLock.token);
		throw e;
	}

	// SSE stream. Each line of worker stdout is already JSON; we wrap as SSE events.
	let cancelFromStream: (() => void) | null = null;
	const stream = new ReadableStream<Uint8Array>({
		start(controller) {
			const encoder = new TextEncoder();
			let streamClosed = false;
			let aborted = false;
			let cancelTombstonePromise: Promise<void> | null = null;
				let runId: string | null = preassignedRunId;
			const writeEvent = (obj: unknown) => {
				if (streamClosed) return;
				try {
					controller.enqueue(encoder.encode(`data: ${JSON.stringify(obj)}\n\n`));
				} catch {
					streamClosed = true;
				}
			};
			let writerHeartbeat: ReturnType<typeof setInterval> | null = null;
			const stopWriterHeartbeat = () => {
				if (writerHeartbeat) clearInterval(writerHeartbeat);
				writerHeartbeat = null;
			};

			const child = spawn(py, args, {
				cwd: repoRoot(),
				env: {
					...process.env,
					PYTHONPATH: resolve(repoRoot(), 'src'),
					INDRA_VIEWER_WRITER_LOCK_TOKEN: writerLock.token
				}
			});
			if (child.pid != null) {
				updateWriterLock(writerLock.token, { pid: child.pid });
				writerHeartbeat = setInterval(() => {
					updateWriterLock(writerLock.token, {});
				}, 5000);
				writerHeartbeat.unref?.();
			}

			// Idempotent terminal cleanup. Both child.on('exit') and
			// child.on('error') route here so the abort listener and stream
			// controller are released exactly once, regardless of which
			// terminal event the child fires (or whether one fires at all
			// because of a synchronous spawn failure).
			let terminated = false;
				const cleanup = () => {
					if (terminated) return;
					terminated = true;
					streamClosed = true;
					stopWriterHeartbeat();
					event.request.signal.removeEventListener('abort', onAbort);
					try { controller.close(); } catch { /* already closed */ }
				};

				// U5.5: if the client disconnects (closed tab, browser-side
				// AbortController.abort()), ask the worker to stop and escalate
				// after a short grace window. This prevents further loop spend,
				// but an already accepted provider request may still bill.
			const onAbort = () => {
				if (aborted) return;
				aborted = true;
				terminateChildProcessWithEscalation(child);
				writeEvent({ event: 'canceled', run_id: runId, reason: 'client_disconnected' });
				cleanup();
			};
			cancelFromStream = onAbort;
			event.request.signal.addEventListener('abort', onAbort);
			if (event.request.signal.aborted) onAbort();

			let stdoutBuf = '';
			let stderrBuf = '';
			// Bound stderr accumulation — score runs can last an hour on a
			// large corpus; an unfiltered Python warning loop would otherwise
			// grow stderrBuf unbounded.
			const STDERR_CAP = 64 * 1024;

			child.stdout.on('data', (chunk: Buffer) => {
				stdoutBuf += chunk.toString('utf-8');
				let nl: number;
				while ((nl = stdoutBuf.indexOf('\n')) >= 0) {
					const line = stdoutBuf.slice(0, nl).trim();
					stdoutBuf = stdoutBuf.slice(nl + 1);
					if (!line) continue;
					try {
						const workerEvent = JSON.parse(line);
						if (
							workerEvent?.event === 'started' &&
							typeof workerEvent.run_id === 'string' &&
							RUN_ID_RE.test(workerEvent.run_id)
						) {
							runId = workerEvent.run_id;
						}
						writeEvent(workerEvent);
					} catch {
						writeEvent({ event: 'stdout_raw', line });
					}
				}
			});
			child.stderr.on('data', (chunk: Buffer) => {
				if (stderrBuf.length >= STDERR_CAP) return;
				stderrBuf += chunk.toString('utf-8');
				if (stderrBuf.length > STDERR_CAP) {
					stderrBuf = stderrBuf.slice(0, STDERR_CAP) + '\n…[stderr truncated]';
				}
			});
				child.on('exit', (code, signal) => {
					stopWriterHeartbeat();
					const canceled = aborted || signal === 'SIGTERM' || signal === 'SIGKILL';
					const cancelReason = aborted ? 'client_disconnected' : 'worker_terminated';
					if (canceled && runId) {
					cancelTombstonePromise = markScoreRunCanceled({
						run_id: runId,
						scorer_version,
						architecture: arch,
						model,
						paired_run_group_id,
						parent_run_id,
						reason: cancelReason
					}).catch((err) => {
						console.error('cancel_tombstone_failed', err);
						writeEvent({
							event: 'cancel_tombstone_failed',
							run_id: runId,
							error: err instanceof Error ? err.message : String(err)
							});
						});
					}
					const afterTombstone = () => {
						clearWriterLockToken(writerLock.token);
					if (canceled) {
						writeEvent({ event: 'canceled', run_id: runId, reason: cancelReason, signal });
					}
					if (code !== 0 && !canceled) {
						writeEvent({
							event: 'error',
							exit_code: code,
							signal,
							stderr: stderrBuf.slice(0, 4096)
						});
					}
						writeEvent({ event: 'channel_closed', exit_code: code ?? -1, signal });
						cleanup();
					};
					if (code === 0 && !canceled && runId && parent_run_id && repairRerunCorrectionIds) {
						void recordRepairRerunChildAfterScore({
							parent_run_id,
							child_run_id: runId,
							architecture: arch as 'decomposed' | 'monolithic',
							source_dump_id,
							correction_ids: repairRerunCorrectionIds,
							path: safePath,
							scoring_mode: repairRerunProbeOnly ? 'probe_only' : 'aggregate',
							probe_step_filter: repairRerunProbeStepFilter
						}).then((result) => {
							writeEvent({ event: 'repair_rerun_completed', ...result });
						}).catch((err) => {
							writeEvent({
								event: 'repair_rerun_completion_failed',
								run_id: runId,
								error: err instanceof Error ? err.message : String(err)
							});
						}).finally(afterTombstone);
						return;
					}
					if (cancelTombstonePromise) {
						void cancelTombstonePromise.finally(afterTombstone);
						return;
					}
				afterTombstone();
			});
			child.on('error', (err) => {
				stopWriterHeartbeat();
				clearWriterLockToken(writerLock.token);
				writeEvent({ event: 'spawn_error', error: err.message });
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
