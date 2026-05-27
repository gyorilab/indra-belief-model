/**
 * POST /api/datasets/ingest — ingest an INDRA Statement JSON (or .json.gz)
 * into corpus.duckdb.
 *
 * U7.1: was synchronous (drain stdout, return JSON) — broke down on the
 * 460MB benchmark .gz where parse + DB-write takes minutes with no signal
 * to the user. Now streams SSE the same way /api/runs/score does, with
 * AbortController-driven SIGTERM/SIGKILL so closing the tab kills the
 * worker.
 *
 * Body:
 *   {
 *     path: string,            // absolute path to JSON or .json.gz
 *     source_dump_id: string   // e.g. "rasmachine_2026-05-11"
 *   }
 *
 * Returns: SSE stream. Events:
 *   data: {"event": "started", ...}
 *   data: {"event": "loaded", "n_statements": N}
 *   data: {"event": "progress", "n_statements_done": N, "n_statements_total": M}
 *   data: {"event": "done", "n_statements": N, "duration_s": ...}
 *   data: {"event": "error", ...}
 * Terminated by `data: {"event": "channel_closed"}`.
 */
import { spawn } from 'node:child_process';
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
import { error } from '@sveltejs/kit';
import type { RequestHandler } from './$types';

const SOURCE_DUMP_RE = /^[a-z][a-z0-9_-]{1,63}$/i;

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

	const safePath = assertPathUnderData(path);
	if (!source_dump_id || !SOURCE_DUMP_RE.test(source_dump_id))
		throw error(400, 'source_dump_id must match /^[a-z][a-z0-9_-]{1,63}$/i');
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
		kind: 'ingest',
		label: 'dataset ingest',
		source_dump_id,
		dataset_path: safePath,
		pid: null
	});
	if (!writerLock) {
		const lock = activeWriterLock();
		throw error(409, writerLockConflictPayload(lock));
	}

	const args = [
		'-m', 'indra_belief.worker',
		'ingest',
		'--db', dbPath(),
		'--path', safePath,
		'--source-dump-id', source_dump_id
	];
	const py = pythonBin();

	// Release the viewer's cached READ_ONLY DuckDB instance so the Python
	// writer can acquire the file lock. Next dashboard read will lazy-reopen.
	try {
		closeInstance();
	} catch (e) {
		clearWriterLockToken(writerLock.token);
		throw e;
	}

	let cancelFromStream: (() => void) | null = null;
	const stream = new ReadableStream<Uint8Array>({
		start(controller) {
			const encoder = new TextEncoder();
			let streamClosed = false;
			let aborted = false;
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

			const onAbort = () => {
				if (aborted) return;
				aborted = true;
				terminateChildProcessWithEscalation(child);
				writeEvent({ event: 'canceled', reason: 'client_disconnected' });
				cleanup();
			};
			cancelFromStream = onAbort;
			event.request.signal.addEventListener('abort', onAbort);
			if (event.request.signal.aborted) onAbort();

			let stdoutBuf = '';
			let stderrBuf = '';
			// Bound stderr accumulation. A chatty worker (verbose warnings on a
			// multi-minute ingest) would otherwise grow this unbounded.
			const STDERR_CAP = 64 * 1024;

			child.stdout.on('data', (chunk: Buffer) => {
				stdoutBuf += chunk.toString('utf-8');
				let nl: number;
				while ((nl = stdoutBuf.indexOf('\n')) >= 0) {
					const line = stdoutBuf.slice(0, nl).trim();
					stdoutBuf = stdoutBuf.slice(nl + 1);
					if (!line) continue;
					try {
						writeEvent(JSON.parse(line));
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
				clearWriterLockToken(writerLock.token);
				if (code !== 0 && signal !== 'SIGTERM' && signal !== 'SIGKILL') {
					writeEvent({
						event: 'error',
						exit_code: code,
						signal,
						stderr: stderrBuf.slice(0, 4096)
					});
				}
				writeEvent({ event: 'channel_closed', exit_code: code ?? -1, signal });
				cleanup();
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
