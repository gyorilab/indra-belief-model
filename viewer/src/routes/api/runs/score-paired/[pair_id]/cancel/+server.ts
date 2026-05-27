import { error, json } from '@sveltejs/kit';
import {
	clearWriterLock,
	readPairedWorkflowState,
	updatePairedWorkflowState,
	type PairedArchitecture
} from '$lib/server/pairedState';
import { markScoreRunCanceled } from '$lib/server/scoreRunLifecycle';
import type { RequestHandler } from './$types';

const PAIR_ID_RE = /^[A-Za-z0-9][A-Za-z0-9_.-]{1,120}$/;
const ARCHES: PairedArchitecture[] = ['monolithic', 'decomposed'];
const CANCEL_REASON = 'canceled_by_user';

function processIsAlive(pid: number): boolean {
	try {
		process.kill(pid, 0);
		return true;
	} catch (e) {
		return (e as NodeJS.ErrnoException).code === 'EPERM';
	}
}

async function waitForExit(pid: number, timeoutMs = 4500): Promise<boolean> {
	const started = Date.now();
	while (Date.now() - started < timeoutMs) {
		if (!processIsAlive(pid)) return true;
		await new Promise((resolve) => setTimeout(resolve, 100));
	}
	return !processIsAlive(pid);
}

export const POST: RequestHandler = async ({ params }) => {
	if (!PAIR_ID_RE.test(params.pair_id)) {
		throw error(400, 'invalid pair_id');
	}
	const state = readPairedWorkflowState(params.pair_id);
	if (!state) {
		throw error(404, 'paired workflow state not found');
	}
	updatePairedWorkflowState(params.pair_id, (s) => {
		const finished = new Date().toISOString();
		for (const architecture of ARCHES) {
			const a = s.architectures[architecture];
			if (a.status !== 'succeeded') {
				s.architectures[architecture] = {
					...a,
					status: 'canceled',
					pid: null,
					finished_at: a.finished_at ?? finished,
					error: CANCEL_REASON,
					updated_at: finished
				};
			}
		}
		return {
			...s,
			status: 'canceled',
			finished_at: s.finished_at ?? finished,
			termination_reason: CANCEL_REASON
		};
	});
	const killed: Array<{ architecture: PairedArchitecture; pid: number; signal: string; ok: boolean; error?: string }> = [];
	const tombstones: Array<Promise<{ architecture: PairedArchitecture; run_id: string | null; ok: boolean; error?: string }>> = [];
	for (const architecture of ARCHES) {
		const archState = state.architectures[architecture];
		const pid = archState.pid;
		if (pid != null) {
			try {
				process.kill(pid, 'SIGTERM');
				killed.push({ architecture, pid, signal: 'SIGTERM', ok: true });
				const forceTimer = setTimeout(() => {
					try {
						process.kill(pid, 0);
						process.kill(pid, 'SIGKILL');
					} catch {
						// already exited
					}
				}, 2000);
				forceTimer.unref?.();
			} catch (e) {
				killed.push({ architecture, pid, signal: 'SIGTERM', ok: false, error: (e as Error).message });
			}
		}
		if (archState.run_id || pid != null) {
			tombstones.push((async () => {
				try {
					if (pid != null) {
						const exited = await waitForExit(pid);
						if (!exited) {
							throw new Error(`worker pid ${pid} still alive after cancel timeout`);
						}
					}
					if (archState.run_id) {
						await markScoreRunCanceled({
							run_id: archState.run_id,
							scorer_version: state.scorer_version,
							architecture,
							model: state.model,
							paired_run_group_id: params.pair_id,
							reason: CANCEL_REASON
						});
					}
					return { architecture, run_id: archState.run_id, ok: true };
				} catch (e) {
					const message = (e as Error).message;
					updatePairedWorkflowState(params.pair_id, (s) => ({
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
					return { architecture, run_id: archState.run_id, ok: false, error: message };
				} finally {
					if (pid == null || !processIsAlive(pid)) {
						clearWriterLock(params.pair_id, architecture, pid);
					}
				}
			})());
		}
	}
	return json({ ok: true, pair_id: params.pair_id, killed, tombstones: await Promise.all(tombstones) });
};
