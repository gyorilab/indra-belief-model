import type { ChildProcess } from 'node:child_process';

export interface TerminateChildProcessOptions {
	graceMs?: number;
	graceSignal?: NodeJS.Signals;
	forceSignal?: NodeJS.Signals;
}

function childHasExited(child: ChildProcess): boolean {
	return child.exitCode !== null || child.signalCode !== null;
}

export function terminateChildProcessWithEscalation(
	child: ChildProcess,
	options: TerminateChildProcessOptions = {}
): void {
	// This deliberately targets the child process only. If the Python worker
	// ever starts subprocesses for model calls, switch the spawner to an
	// isolated process group and signal that group instead.
	const graceMs = options.graceMs ?? 2000;
	const graceSignal = options.graceSignal ?? 'SIGTERM';
	const forceSignal = options.forceSignal ?? 'SIGKILL';
	let observedExit = childHasExited(child);
	let forceTimer: ReturnType<typeof setTimeout> | null = null;

	const cleanup = () => {
		observedExit = true;
		if (forceTimer) {
			clearTimeout(forceTimer);
			forceTimer = null;
		}
		child.off('exit', cleanup);
		child.off('close', cleanup);
	};

	if (observedExit) return;

	child.once('exit', cleanup);
	child.once('close', cleanup);

	const signal = (signalName: NodeJS.Signals) => {
		if (observedExit || childHasExited(child)) {
			cleanup();
			return false;
		}
		try {
			const delivered = child.kill(signalName);
			if (!delivered) cleanup();
			return delivered;
		} catch {
			cleanup();
			return false;
		}
	};

	if (!signal(graceSignal)) return;

	forceTimer = setTimeout(() => {
		signal(forceSignal);
		cleanup();
	}, Math.max(0, graceMs));
	forceTimer.unref?.();
}
