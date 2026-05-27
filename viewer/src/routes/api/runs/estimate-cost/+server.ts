/**
 * POST /api/runs/estimate-cost — preflight cost estimate for a dataset.
 *
 * Spawns `python -m indra_belief.worker estimate-cost --path <p> --arch <a>`
 * (no DB writes).
 * Returns per-model cost estimates so the viewer can render the cost preflight
 * panel before the user commits to scoring.
 */
import { spawn } from 'node:child_process';
import { resolve } from 'node:path';
import { existsSync } from 'node:fs';
import { dbPath } from '$lib/db';
import { assertPathUnderData } from '$lib/pathGuard';
import { json } from '@sveltejs/kit';
import type { RequestHandler } from './$types';

const ARCH_RE = /^(decomposed|monolithic)$/;
const PROBE_STEP_KINDS = new Set([
	'subject_role_probe',
	'object_role_probe',
	'relation_axis_probe',
	'scope_probe'
]);

function probeStepFilter(raw: unknown): string[] {
	if (!Array.isArray(raw)) return [];
	const out: string[] = [];
	const seen = new Set<string>();
	for (const value of raw) {
		const step = String(value ?? '').trim();
		if (!step) continue;
		if (!PROBE_STEP_KINDS.has(step)) {
			throw new Error(`unknown decomposed probe step: ${step}`);
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

export const POST: RequestHandler = async ({ request }) => {
	const body = (await request.json()) as Record<string, unknown>;
	const path = body.path as string | undefined;
	const arch = (body.arch as string | undefined) ?? 'decomposed';
	const probe_only = body.probe_only === true;
	let probe_step_filter: string[];
	try {
		probe_step_filter = probeStepFilter(body.probe_step_filter);
	} catch (err) {
		return json({ ok: false, stderr: err instanceof Error ? err.message : String(err) }, { status: 400 });
	}
	const safePath = assertPathUnderData(path);
	if (!ARCH_RE.test(arch)) {
		return json({ ok: false, stderr: 'arch must be decomposed or monolithic' }, { status: 400 });
	}
	if (probe_only && arch !== 'decomposed') {
		return json({ ok: false, stderr: 'probe_only is only valid for decomposed estimates' }, { status: 400 });
	}
	if (probe_only && probe_step_filter.length === 0) {
		return json({ ok: false, stderr: 'probe_only requires probe_step_filter' }, { status: 400 });
	}

	const args = ['-m', 'indra_belief.worker', 'estimate-cost', '--path', safePath, '--arch', arch];
	if (probe_only) {
		args.push('--probe-only', '--probe-step-filter', probe_step_filter.join(','));
	}
	const py = pythonBin();
	const events: Array<Record<string, unknown>> = [];
	let stderrBuf = '';

	const exitCode: number = await new Promise((resolveP) => {
		const child = spawn(py, args, {
			cwd: repoRoot(),
			env: { ...process.env, PYTHONPATH: resolve(repoRoot(), 'src') }
		});
		let stdoutBuf = '';
		child.stdout.on('data', (chunk: Buffer) => {
			stdoutBuf += chunk.toString('utf-8');
			let nl: number;
			while ((nl = stdoutBuf.indexOf('\n')) >= 0) {
				const line = stdoutBuf.slice(0, nl).trim();
				stdoutBuf = stdoutBuf.slice(nl + 1);
				if (!line) continue;
				try {
					events.push(JSON.parse(line));
				} catch {
					events.push({ event: 'stdout_raw', line });
				}
			}
		});
		child.stderr.on('data', (chunk: Buffer) => {
			stderrBuf += chunk.toString('utf-8');
		});
		child.on('exit', (code) => resolveP(code ?? -1));
		child.on('error', (err) => {
			stderrBuf += `\nspawn error: ${err.message}`;
			resolveP(-1);
		});
	});

	if (exitCode !== 0) {
		return json({ ok: false, exit_code: exitCode, events, stderr: stderrBuf.slice(0, 4096) }, { status: 500 });
	}
	const done = events.find((e) => e.event === 'done') ?? null;
	return json({ ok: true, summary: done });
};
