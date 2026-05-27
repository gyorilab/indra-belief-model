import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { existsSync } from 'node:fs';
import { mkdir, mkdtemp, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createServer } from 'vite';
import { DuckDBInstance } from '@duckdb/node-api';

const PAIR = 'pair_resource_frontier_browser';
const MONO = 'mono_frontier_browser_run';
const DECOMP = 'decomp_frontier_browser_run';
const chromePath = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const artifactRoot = join(process.env.TMPDIR ?? tmpdir(), 'indra-agent-trace-hypergraph');
const profileDir = join(tmpdir(), `indra-frontier-chrome-${process.pid}-${Date.now()}`);
const cdpPort = 9900 + Math.floor(Math.random() * 250);
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

async function createFixtureDb(dbFile) {
	const instance = await DuckDBInstance.create(dbFile);
	const con = await instance.connect();
	try {
		await con.run(`CREATE TABLE score_run (
			run_id VARCHAR,
			scorer_version VARCHAR,
			architecture VARCHAR,
			paired_run_group_id VARCHAR,
			model_id_default VARCHAR,
			started_at TIMESTAMP,
			finished_at TIMESTAMP,
			status VARCHAR,
			n_stmts BIGINT,
			cost_estimate_usd DOUBLE,
			cost_actual_usd DOUBLE
		)`);
		await con.run(`CREATE TABLE statement (
			stmt_hash VARCHAR,
			indra_type VARCHAR,
			indra_belief DOUBLE
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
		await con.run(`CREATE TABLE scorer_step (
			step_hash VARCHAR,
			run_id VARCHAR,
			scorer_version VARCHAR,
			architecture VARCHAR,
			stmt_hash VARCHAR,
			evidence_hash VARCHAR,
			step_kind VARCHAR,
			is_substrate_answered BOOLEAN,
			output_json JSON,
			latency_ms BIGINT,
			prompt_tokens BIGINT,
			out_tokens BIGINT,
			error VARCHAR,
			started_at TIMESTAMP
		)`);
		await con.run(`INSERT INTO score_run VALUES
			('${MONO}', 'frontier-browser-v1', 'monolithic', '${PAIR}', 'demo-model', TIMESTAMP '2026-05-26 00:00:00', TIMESTAMP '2026-05-26 00:01:00', 'succeeded', 3, 0.07, 0.06),
			('${DECOMP}', 'frontier-browser-v1', 'decomposed', '${PAIR}', 'demo-model', TIMESTAMP '2026-05-26 00:02:00', TIMESTAMP '2026-05-26 00:02:40', 'succeeded', 2, 0.05, 0.04)`);
		await con.run(`INSERT INTO statement VALUES
			('stmt_frontier_a', 'Activation', 0.80),
			('stmt_frontier_b', 'Inhibition', 0.20),
			('stmt_frontier_c', 'Complex', 0.50)`);
		await con.run(`INSERT INTO evidence VALUES
			('ev_frontier_a', 'stmt_frontier_a', 'reach', '1', 'shared evidence a'),
			('ev_frontier_b', 'stmt_frontier_b', 'reach', '2', 'shared evidence b'),
			('ev_frontier_c', 'stmt_frontier_c', 'reach', '3', 'monolithic only evidence')`);
		await con.run(`INSERT INTO agent VALUES
			('stmt_frontier_a', 'agent_a', 'subj', 'A', 0),
			('stmt_frontier_a', 'agent_b', 'obj', 'B', 1),
			('stmt_frontier_b', 'agent_c', 'subj', 'C', 0),
			('stmt_frontier_b', 'agent_d', 'obj', 'D', 1)`);
		await con.run(`INSERT INTO scorer_step VALUES
			('m_frontier_a', '${MONO}', 'frontier-browser-v1', 'monolithic', 'stmt_frontier_a', 'ev_frontier_a', 'aggregate', NULL, '{"score":0.80,"verdict":"correct","confidence":"high","tier":"direct"}'::JSON, 100, 10, 5, NULL, TIMESTAMP '2026-05-26 00:00:10'),
			('m_frontier_b', '${MONO}', 'frontier-browser-v1', 'monolithic', 'stmt_frontier_b', 'ev_frontier_b', 'aggregate', NULL, '{"score":0.30,"verdict":"incorrect","confidence":"medium","tier":"direct"}'::JSON, 200, 20, 10, NULL, TIMESTAMP '2026-05-26 00:00:20'),
			('m_frontier_c', '${MONO}', 'frontier-browser-v1', 'monolithic', 'stmt_frontier_c', 'ev_frontier_c', 'aggregate', NULL, '{"score":0.50,"verdict":"abstain","confidence":"low","tier":"fallback"}'::JSON, 400, 30, 20, NULL, TIMESTAMP '2026-05-26 00:00:30'),
			('d_frontier_a', '${DECOMP}', 'frontier-browser-v1', 'decomposed', 'stmt_frontier_a', 'ev_frontier_a', 'aggregate', NULL, '{"score":0.75,"verdict":"correct","confidence":"high"}'::JSON, 300, 40, 10, NULL, TIMESTAMP '2026-05-26 00:02:10'),
			('d_frontier_b', '${DECOMP}', 'frontier-browser-v1', 'decomposed', 'stmt_frontier_b', 'ev_frontier_b', 'aggregate', NULL, '{"score":0.60,"verdict":"correct","confidence":"medium"}'::JSON, 400, 35, 15, NULL, TIMESTAMP '2026-05-26 00:02:20')`);
	} finally {
		con.disconnectSync?.();
		instance.closeSync();
	}
}

async function createFixtureWorkflowState(dir) {
	const stateDir = join(dir, 'viewer_state', 'paired');
	await mkdir(stateDir, { recursive: true });
	const startedAt = new Date(Date.now() - 60_000).toISOString();
	const updatedAt = new Date(Date.now() - 35_000).toISOString();
	const state = {
		pair_id: PAIR,
		status: 'running',
		source_dump_id: 'frontier-browser-source',
		dataset_path: '/tmp/frontier-browser-source.json',
		model: 'demo-model',
		scorer_version: 'frontier-browser-v1',
		total_cost_threshold_usd: 0.12,
		href: `/pairs/${PAIR}`,
		created_at: startedAt,
		updated_at: updatedAt,
		started_at: startedAt,
		finished_at: null,
		termination_reason: null,
		architectures: {
			monolithic: {
				architecture: 'monolithic',
				status: 'running',
				pid: process.pid,
				run_id: MONO,
				cost_threshold_usd: 0.07,
				cost_so_far_usd: 0.03,
				n_evidences_done: 1,
				n_evidences_total: 4,
				latest_stmt_hash: 'stmt_frontier_a',
				started_at: startedAt,
				finished_at: null,
				duration_s: null,
				error: null,
				updated_at: updatedAt
			},
			decomposed: {
				architecture: 'decomposed',
				status: 'queued',
				pid: null,
				run_id: null,
				cost_threshold_usd: 0.05,
				cost_so_far_usd: null,
				n_evidences_done: 0,
				n_evidences_total: null,
				latest_stmt_hash: null,
				started_at: null,
				finished_at: null,
				duration_s: null,
				error: null,
				updated_at: startedAt
			}
		}
	};
	await writeFile(join(stateDir, `${PAIR}.json`), JSON.stringify(state, null, 2));
}

async function screenshot(client, name) {
	const image = await client.send('Page.captureScreenshot', { format: 'png', fromSurface: true });
	const path = join(artifactRoot, `${name}.png`);
	await writeFile(path, image.data, 'base64');
	return path;
}

function auditExpression(viewportName) {
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
		const noTextOverflow = (selector, label) => {
			for (const [index, el] of Array.from(document.querySelectorAll(selector)).entries()) {
				const r = rect(el);
				insideViewport(label + ' ' + index, r);
				if (el.scrollWidth > el.clientWidth + 1) failures.push(label + ' ' + index + ' text overflows');
			}
		};
		const frontier = document.querySelector('#resource-frontier');
		const frontierText = text(frontier);
		const workflowConsole = document.querySelector('#workflow-console');
		const workflowText = text(workflowConsole);
		const runAxis = document.querySelector('.run-axis');
		const runAxisText = text(runAxis);
		const monoPane = document.querySelector('.run-pane-monolithic');
		const decompPane = document.querySelector('.run-pane-decomposed');
		const axisLabel = document.querySelector('.axis-label');
		const monoRunLink = document.querySelector('.run-pane-monolithic .run-link');
		const decompRunLink = document.querySelector('.run-pane-decomposed .run-link');
		const arches = Array.from(document.querySelectorAll('#resource-frontier .frontier-arch'));
		const mono = document.querySelector('#resource-frontier .frontier-arch-monolithic');
		const decomp = document.querySelector('#resource-frontier .frontier-arch-decomposed');
		const scopes = Array.from(document.querySelectorAll('#resource-frontier .frontier-scopes p'));
		const latency = document.querySelector('#metric-mean-latency');
		const tokens = document.querySelector('#metric-tokens');
		const metricGrid = document.querySelector('.metric-grid');
		const overflow = document.documentElement.scrollWidth - window.innerWidth;

		if (overflow > 1) failures.push('page has horizontal overflow');
		if (!workflowConsole) failures.push('workflow console missing');
		if (!workflowText.includes('workflow console running')) failures.push('workflow console did not load running sidecar state');
		if (!workflowText.includes('started') || !workflowText.includes('updated')) failures.push('workflow console clock text missing');
		if (!workflowText.includes('1/4 ev')) failures.push('workflow console progress fact missing');
		if (!workflowText.includes('$0.03/$0.07')) failures.push('workflow console spend/cap fact missing');
		if (!workflowText.includes('eta')) failures.push('workflow console eta missing');
		if (!workflowText.includes('no progress >30s')) failures.push('workflow console stall state missing');
		if (document.querySelectorAll('#workflow-console .workflow-progress-rail[role="progressbar"]').length !== 2) failures.push('workflow progress rails missing');
		if (!runAxis) failures.push('run axis missing');
		if (!runAxisText.includes('[M] monolithic')) failures.push('monolithic lane header is not architecture-first');
		if (!runAxisText.includes('[D] decomposed')) failures.push('decomposed lane header is not architecture-first');
		if (!runAxisText.includes('base denominator')) failures.push('run axis missing base denominator link');
		if (!runAxisText.includes('2 shared') || !runAxisText.includes('1 only') || !runAxisText.includes('0 outside')) failures.push('run axis partition counts missing');
		if (!runAxisText.includes(${JSON.stringify(MONO.slice(0, 8))}) || !runAxisText.includes(${JSON.stringify(DECOMP.slice(0, 8))})) failures.push('run axis short run ids missing');
		if (runAxisText.includes(${JSON.stringify(MONO)}) || runAxisText.includes(${JSON.stringify(DECOMP)})) failures.push('run axis exposes full run ids as visible text');
		if (monoRunLink?.getAttribute('title') !== ${JSON.stringify(MONO)}) failures.push('monolithic full run id is not preserved as title');
		if (decompRunLink?.getAttribute('title') !== ${JSON.stringify(DECOMP)}) failures.push('decomposed full run id is not preserved as title');
		if (!frontier) failures.push('resource frontier section missing');
		if (arches.length !== 2) failures.push('frontier does not render two architecture cards');
		if (!frontierText.includes('whole-run spend beside clean-overlap counters')) failures.push('frontier heading scope missing');
		if (!frontierText.includes('2/2 rows report latency')) failures.push('frontier latency telemetry coverage missing');
		if (!frontierText.includes('2/2 rows report tokens')) failures.push('frontier token telemetry coverage missing');
		if (!frontierText.includes('lower resource use is not a quality signal')) failures.push('frontier resource-not-quality guardrail missing');
		if (scopes.length !== 4) failures.push('frontier scope rail should have four cells');
		for (const [index, scope] of scopes.entries()) insideViewport('frontier scope ' + index, rect(scope));
		insideViewport('workflow console', rect(workflowConsole));
		insideViewport('run axis', rect(runAxis));
		insideViewport('monolithic run pane', rect(monoPane));
		insideViewport('decomposed run pane', rect(decompPane));
		insideViewport('run axis label', rect(axisLabel));
		insideViewport('frontier', rect(frontier));
		insideViewport('latency metric', rect(latency));
		insideViewport('tokens metric', rect(tokens));

		const monoPaneRect = rect(monoPane);
		const axisLabelRect = rect(axisLabel);
		const decompPaneRect = rect(decompPane);
		const monoRect = rect(mono);
		const decompRect = rect(decomp);
		if (${JSON.stringify(viewportName)} === 'desktop') {
			if (monoPaneRect && axisLabelRect && decompPaneRect && !(Math.abs(monoPaneRect.top - decompPaneRect.top) < 8 && monoPaneRect.right <= axisLabelRect.left + 1 && axisLabelRect.right <= decompPaneRect.left + 1)) {
				failures.push('desktop run axis lanes are not left-center-right');
			}
			if (monoRect && decompRect && !(Math.abs(monoRect.top - decompRect.top) < 8 && monoRect.right <= decompRect.left + 1)) {
				failures.push('desktop frontier cards are not side-by-side');
			}
		} else {
			if (monoPaneRect && axisLabelRect && decompPaneRect && !(axisLabelRect.top >= monoPaneRect.bottom - 1 && decompPaneRect.top >= axisLabelRect.bottom - 1)) {
				failures.push('mobile run axis lanes are not stacked');
			}
			if (monoRect && decompRect && !(decompRect.top >= monoRect.bottom - 1)) {
				failures.push('mobile frontier cards are not stacked');
			}
			const metricRect = rect(metricGrid);
			if (metricRect && metricRect.width > window.innerWidth + 1) failures.push('mobile metric grid wider than viewport');
		}

		const latencyText = text(latency);
		const tokenText = text(tokens);
		if (!latencyText.includes('2/2 rows report latency') || !latencyText.includes('not a quality signal')) failures.push('detailed latency card lost coverage or guardrail text');
		if (!tokenText.includes('2/2 rows report tokens') || !tokenText.includes('missing telemetry is not counted as zero-token evidence')) failures.push('detailed token card lost coverage or missing-telemetry text');
		noTextOverflow('#workflow-console .workflow-arch-facts dd, #workflow-console .workflow-clock', 'workflow value');
		noTextOverflow('.run-axis .run-link code, .run-axis .run-facts dd, .run-axis .massbar-key dd', 'run axis value');
		noTextOverflow('#resource-frontier .frontier-row dd', 'frontier value');
		noTextOverflow('#metric-mean-latency dd, #metric-tokens dd', 'resource metric value');
		return {
			viewport: ${JSON.stringify(viewportName)},
			url: location.href,
			browserViewport: {
				width: window.innerWidth,
				height: window.innerHeight,
				scrollWidth: document.documentElement.scrollWidth
			},
			workflowConsole: rect(workflowConsole),
			runAxis: rect(runAxis),
			runMonolithic: monoPaneRect,
			runDecomposed: decompPaneRect,
			frontier: rect(frontier),
			monolithic: monoRect,
			decomposed: decompRect,
			latency: rect(latency),
			tokens: rect(tokens),
			failures
		};
	})()`;
}

async function main() {
	if (!existsSync(chromePath)) {
		throw new Error(`Chrome not found at ${chromePath}`);
	}

	const dir = await mkdtemp(join(tmpdir(), 'indra-pair-frontier-browser-'));
	const dbFile = join(dir, 'frontier-browser.duckdb');
	process.env.VIEWER_DUCKDB_PATH = dbFile;
	process.env.INDRA_VIEWER_SUPPRESS_WRITER_GUARD_LOG = '1';

	let vite = null;
	let chrome = null;
	let client = null;
	let primaryError = null;

	try {
		await mkdir(artifactRoot, { recursive: true });
		await createFixtureDb(dbFile);
		await createFixtureWorkflowState(dir);
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

		const summary = { pair: PAIR, checks: [], screenshots: [] };
		for (const viewport of viewports) {
			await client.send('Emulation.setDeviceMetricsOverride', {
				width: viewport.width,
				height: viewport.height,
				deviceScaleFactor: 1,
				mobile: viewport.mobile
			});
			const load = client.waitEvent('Page.loadEventFired', 20000).catch(() => null);
			await client.send('Page.navigate', { url: `${baseUrl}/pairs/${PAIR}#resource-frontier` });
			await load;
			await waitFor(client, `document.querySelector('#resource-frontier .frontier-arch-decomposed')`, `${viewport.name} frontier render`);
			await evaluate(client, `new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve)))`);
			const audit = await evaluate(client, auditExpression(viewport.name));
			summary.checks.push(audit);
			summary.screenshots.push(await screenshot(client, `paired-resource-frontier-${viewport.name}`));
		}

		const failures = summary.checks.flatMap((check) => check.failures.map((failure) => `${check.viewport}: ${failure}`));
		const result = { ...summary, failures };
		await writeFile(join(artifactRoot, 'paired_resource_frontier_browser_summary.json'), JSON.stringify(result, null, 2));
		assert.deepEqual(failures, []);
		console.log('paired resource frontier browser tests passed');
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
