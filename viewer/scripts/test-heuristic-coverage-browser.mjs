import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { existsSync } from 'node:fs';
import { mkdir, mkdtemp, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createServer } from 'vite';
import { DuckDBInstance } from '@duckdb/node-api';

const DECOMP = 'de000000000000000000000000000101';
const MONO = 'de000000000000000000000000000102';
const chromePath = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const artifactRoot = join(process.env.TMPDIR ?? tmpdir(), 'indra-agent-trace-hypergraph');
const profileDir = join(tmpdir(), `indra-heuristic-coverage-chrome-${process.pid}-${Date.now()}`);
const cdpPort = 9600 + Math.floor(Math.random() * 300);
const viewports = [
	{ name: 'desktop', width: 1280, height: 900, mobile: false },
	{ name: 'mobile', width: 390, height: 900, mobile: true }
];

function sleep(ms) {
	return new Promise((resolve) => setTimeout(resolve, ms));
}

async function pollJson(url, label, timeoutMs = 15000) {
	const deadline = Date.now() + timeoutMs;
	let lastError = null;
	while (Date.now() < deadline) {
		try {
			const response = await fetch(url);
			if (response.ok) return await response.json();
			lastError = new Error(`${label} returned HTTP ${response.status}`);
		} catch (err) {
			lastError = err;
		}
		await sleep(150);
	}
	throw lastError ?? new Error(`${label} did not become available`);
}

function connect(wsUrl) {
	const ws = new WebSocket(wsUrl);
	let nextId = 1;
	const pending = new Map();
	const eventWaiters = new Map();

	ws.addEventListener('message', (event) => {
		const msg = JSON.parse(String(event.data));
		if (msg.id && pending.has(msg.id)) {
			const { resolve: resolvePending, reject } = pending.get(msg.id);
			pending.delete(msg.id);
			if (msg.error) reject(new Error(`${msg.error.message}${msg.error.data ? `: ${msg.error.data}` : ''}`));
			else resolvePending(msg.result ?? {});
			return;
		}
		if (msg.method && eventWaiters.has(msg.method)) {
			const waiters = eventWaiters.get(msg.method);
			eventWaiters.delete(msg.method);
			for (const resolveWaiter of waiters) resolveWaiter(msg.params ?? {});
		}
	});

	const open = new Promise((resolveOpen, reject) => {
		ws.addEventListener('open', resolveOpen, { once: true });
		ws.addEventListener('error', () => reject(new Error('CDP websocket failed to open')), { once: true });
	});

	return {
		open,
		send(method, params = {}) {
			const id = nextId++;
			ws.send(JSON.stringify({ id, method, params }));
			return new Promise((resolveSend, reject) => {
				pending.set(id, { resolve: resolveSend, reject });
			});
		},
		waitEvent(method, timeoutMs = 10000) {
			return new Promise((resolveWaiter, reject) => {
				let timeout = null;
				const resolveOnce = (params) => {
					if (timeout) clearTimeout(timeout);
					resolveWaiter(params);
				};
				const waiters = eventWaiters.get(method) ?? [];
				waiters.push(resolveOnce);
				eventWaiters.set(method, waiters);
				timeout = setTimeout(() => {
					const active = eventWaiters.get(method) ?? [];
					const remaining = active.filter((fn) => fn !== resolveOnce);
					if (remaining.length > 0) eventWaiters.set(method, remaining);
					else eventWaiters.delete(method);
					reject(new Error(`Timed out waiting for ${method}`));
				}, timeoutMs);
			});
		},
		close() {
			ws.close();
		}
	};
}

async function evaluate(client, expression) {
	const result = await client.send('Runtime.evaluate', {
		expression,
		awaitPromise: true,
		returnByValue: true
	});
	if (result.exceptionDetails) {
		const exception = result.exceptionDetails.exception;
		const detail = exception?.description ?? exception?.value ?? result.exceptionDetails.text ?? 'Runtime evaluation failed';
		throw new Error(String(detail));
	}
	return result.result?.value;
}

async function waitFor(client, expression, label, timeoutMs = 10000) {
	const deadline = Date.now() + timeoutMs;
	let lastValue = null;
	while (Date.now() < deadline) {
		lastValue = await evaluate(client, `Boolean(${expression})`);
		if (lastValue) return;
		await sleep(100);
	}
	throw new Error(`${label} did not become true; last=${lastValue}`);
}

async function screenshot(client, name) {
	const image = await client.send('Page.captureScreenshot', { format: 'png', fromSurface: true });
	const path = join(artifactRoot, `${name}.png`);
	await writeFile(path, image.data, 'base64');
	return path;
}

async function createFixtureDb(dbFile) {
	const instance = await DuckDBInstance.create(dbFile);
	const con = await instance.connect();
	try {
		await con.run(`CREATE TABLE score_run (
			run_id VARCHAR,
			scorer_version VARCHAR,
			architecture VARCHAR,
			paired_run_group_id VARCHAR,
			parent_run_id VARCHAR,
			indra_version VARCHAR,
			model_id_default VARCHAR,
			started_at TIMESTAMP,
			finished_at TIMESTAMP,
			status VARCHAR,
			terminated_by VARCHAR,
			termination_reason VARCHAR,
			n_stmts BIGINT,
			cost_estimate_usd DOUBLE,
			cost_actual_usd DOUBLE
		)`);
		await con.run(`CREATE TABLE statement (
			stmt_hash VARCHAR,
			indra_type VARCHAR,
			indra_belief DOUBLE,
			supports_count BIGINT,
			supported_by_count BIGINT,
			source_dump_id VARCHAR
		)`);
		await con.run(`CREATE TABLE evidence (
			evidence_hash VARCHAR,
			stmt_hash VARCHAR,
			source_api VARCHAR,
			pmid VARCHAR,
			text VARCHAR
		)`);
		await con.run(`CREATE TABLE agent (
			stmt_hash VARCHAR,
			agent_hash VARCHAR,
			role VARCHAR,
			name VARCHAR,
			role_index BIGINT
		)`);
		await con.run(`CREATE TABLE metric (
			run_id VARCHAR,
			metric_name VARCHAR,
			truth_set_id VARCHAR,
			value DOUBLE
		)`);
		await con.run(`CREATE TABLE scorer_step (
			step_hash VARCHAR,
			run_id VARCHAR,
			scorer_version VARCHAR,
			architecture VARCHAR,
			stmt_hash VARCHAR,
			evidence_hash VARCHAR,
			step_kind VARCHAR,
			is_substrate_answered BOOLEAN,
			input_payload_json JSON,
			output_json JSON,
			latency_ms BIGINT,
			prompt_tokens BIGINT,
			out_tokens BIGINT,
			finish_reason VARCHAR,
			error VARCHAR,
			started_at TIMESTAMP
		)`);
		await con.run(`INSERT INTO score_run VALUES
			('${DECOMP}', 'heuristic-browser-v1', 'decomposed', NULL, NULL, 'indra-fixture', 'demo-model', TIMESTAMP '2026-05-26 00:00:00', TIMESTAMP '2026-05-26 00:00:20', 'succeeded', NULL, NULL, 1, 0.04, 0.03),
			('${MONO}', 'heuristic-browser-v1', 'monolithic', NULL, NULL, 'indra-fixture', 'demo-model', TIMESTAMP '2026-05-26 00:01:00', TIMESTAMP '2026-05-26 00:01:10', 'succeeded', NULL, NULL, 1, 0.02, 0.02)`);
		await con.run(`INSERT INTO statement VALUES
			('stmt_heuristic_decomp', 'Activation', 0.80, 0, 0, 'heuristic-browser'),
			('stmt_heuristic_mono', 'Inhibition', 0.30, 0, 0, 'heuristic-browser')`);
		await con.run(`INSERT INTO evidence VALUES
			('ev_heuristic_decomp', 'stmt_heuristic_decomp', 'heuristic_fixture', '1', 'decomposed aggregate-only evidence'),
			('ev_heuristic_mono', 'stmt_heuristic_mono', 'heuristic_fixture', '2', 'monolithic native evidence')`);
		await con.run(`INSERT INTO agent VALUES
			('stmt_heuristic_decomp', 'agent_h_a', 'subj', 'A', 0),
			('stmt_heuristic_decomp', 'agent_h_b', 'obj', 'B', 1),
			('stmt_heuristic_mono', 'agent_h_c', 'subj', 'C', 0),
			('stmt_heuristic_mono', 'agent_h_d', 'obj', 'D', 1)`);
		await con.run(`INSERT INTO scorer_step VALUES
			('d_heuristic_aggregate', '${DECOMP}', 'heuristic-browser-v1', 'decomposed', 'stmt_heuristic_decomp', 'ev_heuristic_decomp', 'aggregate', NULL, NULL, '{"score":0.82,"verdict":"correct","confidence":"high"}'::JSON, 120, 12, 6, NULL, NULL, TIMESTAMP '2026-05-26 00:00:10'),
			('m_heuristic_aggregate', '${MONO}', 'heuristic-browser-v1', 'monolithic', 'stmt_heuristic_mono', 'ev_heuristic_mono', 'aggregate', NULL, NULL, '{"score":0.32,"verdict":"incorrect","confidence":"medium","tier":"direct","grounding_status":"grounded","call_log":[],"selected_example_ids":[]}'::JSON, 90, 10, 4, NULL, NULL, TIMESTAMP '2026-05-26 00:01:05')`);
	} finally {
		con.disconnectSync?.();
		instance.closeSync();
	}
}

function absenceAuditExpression(viewportName) {
	return `(() => {
		const failures = [];
		const text = (el) => (el?.textContent ?? '').replace(/\\s+/g, ' ').trim();
		const rect = (el) => {
			if (!el) return null;
			const r = el.getBoundingClientRect();
			return { left: r.left, right: r.right, top: r.top, bottom: r.bottom, width: r.width, height: r.height };
		};
		const insideViewport = (name, r) => {
			if (!r) {
				failures.push(name + ' missing');
				return;
			}
			if (r.left < -1 || r.right > window.innerWidth + 1) failures.push(name + ' escapes viewport');
			if (r.width < 1 || r.height < 1) failures.push(name + ' is not visible');
		};
		const pageText = text(document.body);
		const coverage = document.querySelector('.coverage');
		const missing = document.querySelector('.cov-missing-probes');
		const link = document.querySelector('.cov-missing-probes .cov-diagnostic-link');
		const traceTable = document.querySelector('.tf-counts');
		const overflow = document.documentElement.scrollWidth - window.innerWidth;
		if (overflow > 1) failures.push('page has horizontal overflow');
		if (!pageText.includes('decomposed')) failures.push('run architecture missing');
		if (!pageText.includes('1 aggregate evidence row exists')) failures.push('aggregate denominator summary missing');
		if (!pageText.includes('Probe coverage is unavailable, not zero')) failures.push('unavailable-not-zero summary missing');
		if (!pageText.includes('This is not a zero-LLM or zero-probe result')) failures.push('zero-vs-unavailable note missing');
		if (!pageText.includes('persisted trace observation')) failures.push('trace observation block missing');
		if (!pageText.includes('0 decomposed native step rows')) failures.push('aggregate-only trace observation missing');
		if (!pageText.includes('DB cannot distinguish legacy/imported aggregate-only data from failed trace persistence')) failures.push('cause-boundary copy missing');
		if (!pageText.includes('aggregate evidences 1')) failures.push('aggregate evidence diagnostic fact missing');
		if (!pageText.includes('native steps 0')) failures.push('native-step diagnostic fact missing');
		if (!pageText.includes('route/probe rows 0')) failures.push('route/probe diagnostic fact missing');
		if (!pageText.includes('run lifecycle boundary')) failures.push('run lifecycle boundary missing');
		if (!pageText.includes('run lifecycle is succeeded')) failures.push('succeeded lifecycle message missing');
		if (!pageText.includes('does not support user cancellation as the explanation')) failures.push('lifecycle cause-boundary copy missing');
		if (!pageText.includes('root cause remains unclassified')) failures.push('root-cause unclassified copy missing');
		if (!pageText.includes('run status succeeded')) failures.push('run-status diagnostic fact missing');
		if (!pageText.includes('scorer version heuristic-browser-v1')) failures.push('scorer-version diagnostic fact missing');
		if (!pageText.includes('termination none recorded')) failures.push('termination diagnostic fact missing');
		if (!pageText.includes('open aggregate-only trace cohort')) failures.push('diagnostic cohort action missing');
		if (!pageText.includes('aggregate only')) failures.push('trace-fidelity aggregate-only state missing');
		if (document.querySelector('.coverage .cov-table')) failures.push('probe table rendered despite missing probe records');
		if (document.querySelector('.coverage .cov-empty')) failures.push('generic empty state rendered instead of absence note');
		const expectedHref = '/runs/${DECOMP}/cohort?trace_state=aggregate_only';
		if (link?.getAttribute('href') !== expectedHref) failures.push('diagnostic link href mismatch');
		insideViewport('coverage panel', rect(coverage));
		insideViewport('missing-probes note', rect(missing));
		insideViewport('diagnostic link', rect(link));
		insideViewport('trace table', rect(traceTable));
		if (missing && missing.scrollWidth > missing.clientWidth + 1) failures.push('missing-probes note text overflows');
		return {
			viewport: ${JSON.stringify(viewportName)},
			url: location.href,
			browserViewport: {
				width: window.innerWidth,
				height: window.innerHeight,
				scrollWidth: document.documentElement.scrollWidth
			},
			coverage: rect(coverage),
			missing: rect(missing),
			link: rect(link),
			traceTable: rect(traceTable),
			failures
		};
	})()`;
}

function monolithicAuditExpression() {
	return `(() => {
		const failures = [];
		const text = (el) => (el?.textContent ?? '').replace(/\\s+/g, ' ').trim();
		const pageText = text(document.body);
		if (!pageText.includes('monolithic')) failures.push('run architecture missing');
		if (!pageText.includes('not defined for this architecture')) failures.push('not-defined state missing');
		if (!pageText.includes('four-probe pillbar belongs to decomposed scoring only')) failures.push('monolithic native grammar note missing');
		if (!pageText.includes('1 aggregate evidence row can still support architecture-blind verdict, cost, and latency metrics')) failures.push('monolithic aggregate denominator footnote missing');
		if (document.querySelector('.cov-missing-probes')) failures.push('monolithic rendered decomposed absence state');
		if (document.querySelector('.coverage .cov-table')) failures.push('monolithic rendered decomposed probe table');
		return { url: location.href, failures };
	})()`;
}

function cohortAuditExpression() {
	return `(() => {
		const failures = [];
		const text = (el) => (el?.textContent ?? '').replace(/\\s+/g, ' ').trim();
		const pageText = text(document.body);
		if (!pageText.includes('1 trace evidence row in run')) failures.push('cohort count missing');
		if (!pageText.includes('trace_stateaggregate_only')) failures.push('trace_state filter chip missing');
		if (!pageText.includes('Trace-state cohorts are grouped from persisted scorer steps')) failures.push('trace-state cohort note missing');
		if (!pageText.includes('trace state: aggregate only')) failures.push('trace-state meaning label missing');
		if (!pageText.includes('These rows have aggregate verdicts but no decomposed native/probe rows in the trace plane')) failures.push('aggregate-only cohort cause-boundary text missing');
		if (!pageText.includes('makes probe coverage unavailable here, not zero')) failures.push('aggregate-only cohort unavailable-not-zero text missing');
		if (!pageText.includes('does not prove whether the cause is legacy import, worker failure, migration skew')) failures.push('aggregate-only cohort root-cause boundary missing');
		if (!pageText.includes('aggregate_only') && !pageText.includes('aggregate only')) failures.push('aggregate-only row state missing');
		if (!pageText.includes('ev_heur')) failures.push('expected evidence row missing');
		if (document.documentElement.scrollWidth - window.innerWidth > 1) failures.push('cohort page has horizontal overflow');
		return { url: location.href, failures };
	})()`;
}

async function main() {
	if (!existsSync(chromePath)) {
		throw new Error(`Chrome not found at ${chromePath}`);
	}

	const dir = await mkdtemp(join(tmpdir(), 'indra-heuristic-coverage-browser-'));
	const dbFile = join(dir, 'heuristic-browser.duckdb');
	process.env.VIEWER_DUCKDB_PATH = dbFile;
	process.env.INDRA_VIEWER_SUPPRESS_WRITER_GUARD_LOG = '1';

	let vite = null;
	let chrome = null;
	let client = null;
	let primaryError = null;

	try {
		await mkdir(artifactRoot, { recursive: true });
		await createFixtureDb(dbFile);
		vite = await createServer({
			configFile: './vite.config.ts',
			optimizeDeps: {
				disabled: true
			},
			environments: {
				client: {
					optimizeDeps: {
						disabled: true
					}
				}
			},
			server: {
				host: '127.0.0.1',
				port: 0,
				strictPort: false,
				hmr: false
			},
			logLevel: 'error'
		});
		await vite.listen();
		const address = vite.httpServer?.address();
		if (!address || typeof address === 'string') throw new Error('Vite did not expose a TCP address');
		const baseUrl = `http://127.0.0.1:${address.port}`;

		chrome = spawn(chromePath, [
			`--user-data-dir=${profileDir}`,
			`--remote-debugging-port=${cdpPort}`,
			'--headless=new',
			'--disable-gpu',
			'--disable-background-networking',
			'--no-first-run',
			'--no-default-browser-check',
			'about:blank'
		], { stdio: ['ignore', 'ignore', 'pipe'] });

		await pollJson(`http://127.0.0.1:${cdpPort}/json/version`, 'CDP version');
		const target = await pollJson(`http://127.0.0.1:${cdpPort}/json/new?${encodeURIComponent('about:blank')}`, 'CDP target')
			.catch(async () => {
				const targets = await pollJson(`http://127.0.0.1:${cdpPort}/json/list`, 'CDP target list');
				return targets.find((item) => item.type === 'page') ?? targets[0];
			});
		if (!target?.webSocketDebuggerUrl) throw new Error('No page websocket URL returned by Chrome');
		client = connect(target.webSocketDebuggerUrl);
		await client.open;
		await client.send('Page.enable');
		await client.send('Runtime.enable');

		const summary = { run_id: DECOMP, checks: [], screenshots: [] };
		for (const viewport of viewports) {
			await client.send('Emulation.setDeviceMetricsOverride', {
				width: viewport.width,
				height: viewport.height,
				deviceScaleFactor: 1,
				mobile: viewport.mobile
			});
			const load = client.waitEvent('Page.loadEventFired', 20000).catch(() => null);
			await client.send('Page.navigate', { url: `${baseUrl}/runs/${DECOMP}` });
			await load;
			await waitFor(client, `document.querySelector('.cov-missing-probes .cov-diagnostic-link')`, `${viewport.name} heuristic absence render`);
			await evaluate(client, `new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve)))`);
			const audit = await evaluate(client, absenceAuditExpression(viewport.name));
			summary.checks.push(audit);
			summary.screenshots.push(await screenshot(client, `heuristic-coverage-absence-${viewport.name}`));
		}

		await client.send('Emulation.setDeviceMetricsOverride', {
			width: 1280,
			height: 900,
			deviceScaleFactor: 1,
			mobile: false
		});
		let load = client.waitEvent('Page.loadEventFired', 20000).catch(() => null);
		await client.send('Page.navigate', { url: `${baseUrl}/runs/${MONO}` });
		await load;
		await waitFor(client, `document.querySelector('.cov-na')`, 'monolithic not-defined render');
		const monoAudit = await evaluate(client, monolithicAuditExpression());
		summary.checks.push({ viewport: 'desktop-monolithic', ...monoAudit });

		load = client.waitEvent('Page.loadEventFired', 20000).catch(() => null);
		await client.send('Page.navigate', { url: `${baseUrl}/runs/${DECOMP}/cohort?trace_state=aggregate_only` });
		await load;
		await waitFor(client, `document.querySelector('.trace-state')`, 'aggregate-only cohort render');
		const cohortAudit = await evaluate(client, cohortAuditExpression());
		summary.checks.push({ viewport: 'desktop-cohort', ...cohortAudit });

		const failures = summary.checks.flatMap((check) => check.failures.map((failure) => `${check.viewport}: ${failure}`));
		const result = { ...summary, failures };
		await writeFile(join(artifactRoot, 'heuristic_coverage_browser_summary.json'), JSON.stringify(result, null, 2));
		assert.deepEqual(failures, []);
		console.log('heuristic coverage browser tests passed');
	} catch (err) {
		primaryError = err;
	} finally {
		client?.close();
		if (chrome) {
			chrome.kill('SIGTERM');
			await sleep(300);
			if (chrome.exitCode === null) chrome.kill('SIGKILL');
		}
		if (vite) {
			await sleep(1500);
			await vite.close();
		}
		await rm(profileDir, { recursive: true, force: true });
		await rm(dir, { recursive: true, force: true });
	}
	if (primaryError) throw primaryError;
}

main().catch((err) => {
	console.error(err);
	process.exitCode = 1;
});
