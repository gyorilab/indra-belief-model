import { spawn } from 'node:child_process';
import { existsSync } from 'node:fs';
import { resolve } from 'node:path';
import { error, json } from '@sveltejs/kit';
import { exportRepairRerunCorpus } from '$lib/server/repairRerun';
import { activeWriterLock, writerLockConflictCode, writerLockConflictText } from '$lib/server/pairedState';
import type { RequestHandler } from './$types';

const RUN_ID_RE = /^[a-f0-9]{32}$/i;
const ARCH_RE = /^(decomposed|monolithic)$/;

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

async function estimateCost(
	path: string,
	arch: string,
	scoringMode: string,
	probeStepFilter: string[]
): Promise<{
	exitCode: number;
	events: Array<Record<string, unknown>>;
	stderr: string;
}> {
	const args = ['-m', 'indra_belief.worker', 'estimate-cost', '--path', path, '--arch', arch];
	if (scoringMode === 'probe_only') {
		args.push('--probe-only', '--probe-step-filter', probeStepFilter.join(','));
	}
	const py = pythonBin();
	const events: Array<Record<string, unknown>> = [];
	let stderr = '';
	const exitCode: number = await new Promise((resolveP) => {
		const child = spawn(py, args, {
			cwd: repoRoot(),
			env: { ...process.env, PYTHONPATH: resolve(repoRoot(), 'src') }
		});
		let stdout = '';
		child.stdout.on('data', (chunk: Buffer) => {
			stdout += chunk.toString('utf-8');
			let nl: number;
			while ((nl = stdout.indexOf('\n')) >= 0) {
				const line = stdout.slice(0, nl).trim();
				stdout = stdout.slice(nl + 1);
				if (!line) continue;
				try {
					events.push(JSON.parse(line));
				} catch {
					events.push({ event: 'stdout_raw', line });
				}
			}
		});
		child.stderr.on('data', (chunk: Buffer) => {
			stderr += chunk.toString('utf-8');
		});
		child.on('exit', (code) => resolveP(code ?? -1));
		child.on('error', (err) => {
			stderr += `\nspawn error: ${err.message}`;
			resolveP(-1);
		});
	});
	return { exitCode, events, stderr };
}

export const POST: RequestHandler = async ({ request }) => {
	const body = (await request.json()) as Record<string, unknown>;
	const run_id = body.run_id;
	if (typeof run_id !== 'string' || !RUN_ID_RE.test(run_id)) {
		throw error(400, 'run_id must be 32 hex chars');
	}
	const arch = (body.arch as string | undefined) ?? 'decomposed';
	if (!ARCH_RE.test(arch)) throw error(400, 'arch must be decomposed or monolithic');
	const correction_ids = Array.isArray(body.correction_ids)
		? body.correction_ids.map((id) => Number(id)).filter((id) => Number.isInteger(id) && id > 0)
		: null;
	if (Array.isArray(body.correction_ids) && correction_ids?.length === 0) {
		throw error(400, 'select at least one repair candidate');
	}

	const activeLock = activeWriterLock();
	if (activeLock) {
		throw error(409, {
			code: writerLockConflictCode(activeLock),
			message: writerLockConflictText(activeLock)
		});
	}

	try {
		const corpus = await exportRepairRerunCorpus({ run_id, correction_ids });
		if (arch !== corpus.architecture) {
			throw new Error(`repair rerun architecture ${arch} does not match parent run architecture ${corpus.architecture}; use the paired workbench for cross-architecture comparison`);
		}
		const estimate = await estimateCost(corpus.path, arch, corpus.scoring_mode, corpus.probe_step_filter);
		if (estimate.exitCode !== 0) {
			return json({
				ok: false,
				corpus,
				exit_code: estimate.exitCode,
				events: estimate.events,
				stderr: estimate.stderr.slice(0, 4096)
			}, { status: 500 });
		}
		const summary = estimate.events.find((e) => e.event === 'done') ?? null;
		return json({ ok: true, corpus, summary });
	} catch (e) {
		throw error(400, (e as Error).message || 'repair rerun estimate failed');
	}
};
