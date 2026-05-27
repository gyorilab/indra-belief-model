import assert from 'node:assert/strict';
import { chmod, mkdir, mkdtemp, rm, writeFile } from 'node:fs/promises';
import { existsSync, realpathSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createServer } from 'vite';

function sleep(ms) {
	return new Promise((resolve) => setTimeout(resolve, ms));
}

async function waitFor(predicate, description, timeoutMs = 5000) {
	const deadline = Date.now() + timeoutMs;
	let lastValue = null;
	while (Date.now() < deadline) {
		lastValue = await predicate();
		if (lastValue) return lastValue;
		await sleep(50);
	}
	throw new Error(`${description} did not become true; last=${JSON.stringify(lastValue)}`);
}

const dir = await mkdtemp(join(tmpdir(), 'indra-truth-set-lifecycle-'));
const dataDir = join(dir, 'data');
const benchmarkDir = join(dataDir, 'benchmark');
const dbFile = join(dataDir, 'corpus.duckdb');
const stubPython = join(dir, 'python-worker-stub.mjs');
const previousDbPath = process.env.VIEWER_DUCKDB_PATH;
const previousPython = process.env.PYTHON_BIN;
let ctrl = null;
let responsePromise = null;

process.env.VIEWER_DUCKDB_PATH = dbFile;
process.env.PYTHON_BIN = stubPython;

const server = await createServer({
	configFile: './vite.config.ts',
	server: { middlewareMode: true, hmr: { port: 31_000 + (process.pid % 10_000) } },
	logLevel: 'error'
});

try {
	await mkdir(benchmarkDir, { recursive: true });

	const { POST } = await server.ssrLoadModule('/src/routes/api/truth-sets/+server.ts');
	const { readWriterLock } = await server.ssrLoadModule('/src/lib/server/pairedState.ts');

	async function runAbortCase(label, stubBody) {
		const truthFile = join(benchmarkDir, `${label}.jsonl`);
		await writeFile(
			truthFile,
			`${JSON.stringify({ matches_hash: '123456789', tag: 'correct', text: `truth lifecycle fixture ${label}` })}\n`
		);
		await writeFile(
			stubPython,
			`#!/usr/bin/env node
process.stdout.write(JSON.stringify({ event: 'started', label: ${JSON.stringify(label)} }) + '\\n');
${stubBody}
setInterval(() => {}, 1000);
`,
			{ mode: 0o755 }
		);
		await chmod(stubPython, 0o755);

		ctrl = new AbortController();
		const request = new Request('http://localhost/api/truth-sets', {
			method: 'POST',
			headers: { 'content-type': 'application/json' },
			body: JSON.stringify({
				path: truthFile,
				truth_set_id: `truth_${label}`,
				truth_set_name: `truth ${label} fixture`,
				target_kind: 'evidence',
				field: 'tag'
			}),
			signal: ctrl.signal
		});
		responsePromise = POST({ request });

		const activeLock = await waitFor(
			() => {
				const lock = readWriterLock();
				return lock?.kind === 'truth_set' && lock.pid != null ? lock : null;
			},
			`truth-set writer lock for ${label}`
		);
		assert.equal(activeLock.dataset_path, realpathSync(truthFile), 'truth-set lock records the registered file path');

		ctrl.abort();
		const response = await responsePromise;
		assert.equal(response.status, 499, `aborted truth-set registration returns client-disconnected status for ${label}`);
		const body = await response.json();
		assert.equal(body.ok, false);
		assert.equal(body.error, 'client_disconnected');

		await waitFor(() => readWriterLock() == null, `truth-set writer lock cleanup for ${label}`);
		assert.equal(existsSync(join(dataDir, 'viewer_state', 'writer_lock.json')), false, 'writer lock file is removed after abort');
		ctrl = null;
		responsePromise = null;
	}

	await runAbortCase('cooperative', "process.on('SIGTERM', () => setTimeout(() => process.exit(143), 25));");
	await runAbortCase('stubborn', "process.on('SIGTERM', () => {});");
	console.log('truth-set lifecycle tests passed');
} finally {
	ctrl?.abort();
	if (responsePromise) {
		await Promise.race([
			responsePromise.catch(() => null),
			sleep(1000)
		]);
	}
	await server.close();
	if (previousDbPath == null) delete process.env.VIEWER_DUCKDB_PATH;
	else process.env.VIEWER_DUCKDB_PATH = previousDbPath;
	if (previousPython == null) delete process.env.PYTHON_BIN;
	else process.env.PYTHON_BIN = previousPython;
	await rm(dir, { recursive: true, force: true });
}
