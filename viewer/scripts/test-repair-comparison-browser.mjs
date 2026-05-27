import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { existsSync } from 'node:fs';
import { mkdir, mkdtemp, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createServer } from 'vite';
import { DuckDBInstance } from '@duckdb/node-api';

const PARENT = 'aa000000000000000000000000000118';
const CHILD = 'bb000000000000000000000000000118';
const chromePath = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const artifactRoot = join(process.env.TMPDIR ?? tmpdir(), 'indra-agent-trace-hypergraph');
const profileDir = join(tmpdir(), `indra-repair-comparison-chrome-${process.pid}-${Date.now()}`);
const cdpPort = 9900 + Math.floor(Math.random() * 300);
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
			const { resolve, reject } = pending.get(msg.id);
			pending.delete(msg.id);
			if (msg.error) reject(new Error(`${msg.error.message}${msg.error.data ? `: ${msg.error.data}` : ''}`));
			else resolve(msg.result ?? {});
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
			return new Promise((resolve, reject) => {
				pending.set(id, { resolve, reject });
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
			indra_version VARCHAR,
			architecture VARCHAR,
			paired_run_group_id VARCHAR,
			parent_run_id VARCHAR,
			model_id_default VARCHAR,
			started_at TIMESTAMP,
			finished_at TIMESTAMP,
			n_stmts BIGINT,
			status VARCHAR,
			cost_estimate_usd DOUBLE,
			cost_actual_usd DOUBLE,
			terminated_by VARCHAR,
			termination_reason VARCHAR
		)`);
		await con.run(`CREATE TABLE statement (
			stmt_hash VARCHAR,
			indra_type VARCHAR,
			indra_belief DOUBLE,
			supports_count BIGINT,
			supported_by_count BIGINT,
			raw_json JSON
		)`);
		await con.run(`CREATE TABLE evidence (
			evidence_hash VARCHAR,
			stmt_hash VARCHAR,
			source_api VARCHAR,
			pmid VARCHAR,
			text VARCHAR
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
		await con.run(`CREATE TABLE scorer_step_correction (
			correction_id BIGINT,
			run_id VARCHAR,
			step_hash VARCHAR,
			stmt_hash VARCHAR,
			evidence_hash VARCHAR,
			correction_kind VARCHAR,
			status VARCHAR,
			severity VARCHAR,
			note TEXT,
			reviewer VARCHAR,
			value_json JSON,
			created_at TIMESTAMP,
			parent_correction_id BIGINT,
			child_run_id VARCHAR,
			repair_source_dump_id VARCHAR
		)`);
		await con.run(`INSERT INTO score_run VALUES
			('${PARENT}', 'repair-browser-v1', 'indra-fixture', 'decomposed', NULL, NULL, 'demo-model', TIMESTAMP '2026-05-26 00:00:00', TIMESTAMP '2026-05-26 00:00:20', 3, 'succeeded', 0.06, 0.05, NULL, NULL),
			('${CHILD}', 'repair-browser-v1-child', 'indra-fixture', 'decomposed', NULL, '${PARENT}', 'demo-model', TIMESTAMP '2026-05-26 00:03:00', TIMESTAMP '2026-05-26 00:03:35', 3, 'succeeded', 0.07, 0.06, NULL, NULL)`);
		await con.run(`INSERT INTO statement VALUES
			('stmt_repair_a', 'Activation', 0.80, 0, 0, '{"type":"Activation"}'::JSON),
			('stmt_repair_b', 'Inhibition', 0.20, 0, 0, '{"type":"Inhibition"}'::JSON),
			('stmt_repair_c', 'Complex', 0.55, 0, 0, '{"type":"Complex"}'::JSON)`);
		await con.run(`INSERT INTO evidence VALUES
			('ev_repair_a', 'stmt_repair_a', 'reach', '1', 'candidate A evidence'),
			('ev_repair_b', 'stmt_repair_b', 'sparser', '2', 'candidate B evidence'),
			('ev_repair_c', 'stmt_repair_c', 'reach', '3', 'candidate C evidence')`);
		await con.run(`INSERT INTO scorer_step VALUES
			('parent_agg_a', '${PARENT}', 'repair-browser-v1', 'decomposed', 'stmt_repair_a', 'ev_repair_a', 'aggregate', NULL, '{"score":0.20,"verdict":"incorrect","confidence":"low"}'::JSON, 100, 10, 5, NULL, TIMESTAMP '2026-05-26 00:00:10'),
			('parent_agg_b', '${PARENT}', 'repair-browser-v1', 'decomposed', 'stmt_repair_b', 'ev_repair_b', 'aggregate', NULL, '{"score":0.82,"verdict":"correct","confidence":"high"}'::JSON, 120, 12, 6, NULL, TIMESTAMP '2026-05-26 00:00:11'),
			('child_agg_a', '${CHILD}', 'repair-browser-v1-child', 'decomposed', 'stmt_repair_a', 'ev_repair_a', 'aggregate', NULL, '{"score":0.82,"verdict":"correct","confidence":"high"}'::JSON, 90, 9, 4, NULL, TIMESTAMP '2026-05-26 00:03:10'),
			('child_agg_b', '${CHILD}', 'repair-browser-v1-child', 'decomposed', 'stmt_repair_b', 'ev_repair_b', 'aggregate', NULL, '{"score":0.10,"verdict":"incorrect","confidence":"high"}'::JSON, 110, 11, 5, NULL, TIMESTAMP '2026-05-26 00:03:11'),
			('child_agg_c', '${CHILD}', 'repair-browser-v1-child', 'decomposed', 'stmt_repair_c', 'ev_repair_c', 'aggregate', NULL, '{"score":0.52,"verdict":"correct","confidence":"medium"}'::JSON, 130, 13, 6, NULL, TIMESTAMP '2026-05-26 00:03:12')`);
		await con.run(`INSERT INTO scorer_step_correction VALUES
			(101, '${PARENT}', 'parent_agg_a', 'stmt_repair_a', 'ev_repair_a', 'repair_candidate', 'open', 'high', 'candidate A', 'browser-test', '{"suspected_step_kind":"aggregate"}'::JSON, TIMESTAMP '2026-05-26 00:01:00', NULL, NULL, NULL),
			(102, '${PARENT}', 'parent_agg_b', 'stmt_repair_b', 'ev_repair_b', 'repair_candidate', 'open', 'high', 'candidate B', 'browser-test', '{"suspected_step_kind":"aggregate"}'::JSON, TIMESTAMP '2026-05-26 00:01:01', NULL, NULL, NULL),
			(103, '${PARENT}', NULL, 'stmt_repair_c', 'ev_repair_c', 'repair_candidate', 'open', 'medium', 'candidate C', 'browser-test', '{"suspected_step_kind":"aggregate"}'::JSON, TIMESTAMP '2026-05-26 00:01:02', NULL, NULL, NULL),
			(201, '${PARENT}', NULL, 'stmt_repair_a', 'ev_repair_a', 'rerun_child', 'done', NULL, NULL, 'browser-test', '{"parent_correction_id":101,"child_run_id":"${CHILD}","source_dump_id":"repair-browser"}'::JSON, TIMESTAMP '2026-05-26 00:04:00', 101, '${CHILD}', 'repair-browser'),
			(202, '${PARENT}', NULL, 'stmt_repair_b', 'ev_repair_b', 'rerun_child', 'done', NULL, NULL, 'browser-test', '{"parent_correction_id":102,"child_run_id":"${CHILD}","source_dump_id":"repair-browser"}'::JSON, TIMESTAMP '2026-05-26 00:04:01', 102, '${CHILD}', 'repair-browser'),
			(203, '${PARENT}', NULL, 'stmt_repair_c', 'ev_repair_c', 'rerun_child', 'done', NULL, NULL, 'browser-test', '{"parent_correction_id":103,"child_run_id":"${CHILD}","source_dump_id":"repair-browser"}'::JSON, TIMESTAMP '2026-05-26 00:04:02', 103, '${CHILD}', 'repair-browser')`);
	} finally {
		con.disconnectSync?.();
		instance.closeSync();
	}
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
		const pageText = text(document.body);
		const summary = document.querySelector('[aria-label="full repair comparison summary"]');
		const lanes = document.querySelector('[aria-label="full repair candidate before after lanes"]');
		const tableWrap = document.querySelector('.table-wrap');
		const table = document.querySelector('table');
		const pager = document.querySelector('[aria-label="repair comparison pagination"]');
		const rows = Array.from(document.querySelectorAll('tbody tr'));
		const next = Array.from(document.querySelectorAll('.pager a')).find((a) => text(a) === 'next');
		const previousText = Array.from(document.querySelectorAll('.pager span')).find((span) => text(span) === 'previous');
		const overflow = document.documentElement.scrollWidth - window.innerWidth;
		if (overflow > 1) failures.push('page has horizontal overflow');
		if (!pageText.includes('full repair comparison')) failures.push('title missing');
		if (!pageText.includes(${JSON.stringify(PARENT.slice(0, 8))}) || !pageText.includes(${JSON.stringify(CHILD.slice(0, 8))})) failures.push('parent/child anchors missing');
		if (!pageText.includes('decomposed to decomposed')) failures.push('architecture comparison missing');
		if (!pageText.includes('2 ev')) failures.push('overlap denominator missing');
		if (!pageText.includes('3 / 3 covered')) failures.push('candidate coverage missing');
		if (!pageText.includes('1 new aggregate')) failures.push('new child aggregate count missing');
		if (!pageText.includes('showing 1-2 of 3 repair candidates')) failures.push('page count missing');
		if (!pageText.includes('verdict to correct')) failures.push('verdict-to-correct movement missing');
		if (!pageText.includes('new child aggregate')) failures.push('new child aggregate movement missing');
		if (!pageText.includes('#101') || !pageText.includes('#103')) failures.push('visible candidate ids missing');
		if (pageText.includes('#102')) failures.push('third-ranked candidate leaked onto first page');
		if (rows.length !== 2) failures.push('first page should render exactly two candidate rows');
		if (!next || !next.getAttribute('href')?.includes('offset=2')) failures.push('next pagination link missing offset');
		if (!previousText) failures.push('disabled previous state missing');
		insideViewport('summary', rect(summary));
		insideViewport('lanes', rect(lanes));
		insideViewport('table wrapper', rect(tableWrap));
		insideViewport('pager', rect(pager));
		if (tableWrap && table) {
			const wrapRect = rect(tableWrap);
			const tableRect = rect(table);
			if (${JSON.stringify(viewportName)} === 'desktop' && tableWrap.scrollWidth > tableWrap.clientWidth + 1) failures.push('desktop candidate table scrolls horizontally');
			if (${JSON.stringify(viewportName)} === 'mobile' && tableRect.width <= wrapRect.width) failures.push('mobile candidate table is not contained as a scrollable comparison plane');
		}
		noTextOverflow('h1, dl dd, .pager a, .pager span', 'summary/control text');
		return {
			viewport: ${JSON.stringify(viewportName)},
			url: location.href,
			browserViewport: {
				width: window.innerWidth,
				height: window.innerHeight,
				scrollWidth: document.documentElement.scrollWidth
			},
			summary: rect(summary),
			lanes: rect(lanes),
			tableWrap: rect(tableWrap),
			pager: rect(pager),
			rowCount: rows.length,
			failures
		};
	})()`;
}

function secondPageAuditExpression() {
	return `(() => {
		const failures = [];
		const text = (el) => (el?.textContent ?? '').replace(/\\s+/g, ' ').trim();
		const pageText = text(document.body);
		const rows = Array.from(document.querySelectorAll('tbody tr'));
		const previous = Array.from(document.querySelectorAll('.pager a')).find((a) => text(a) === 'previous');
		const nextText = Array.from(document.querySelectorAll('.pager span')).find((span) => text(span) === 'next');
		if (!pageText.includes('showing 3-3 of 3 repair candidates')) failures.push('second page count missing');
		if (!pageText.includes('verdict to incorrect')) failures.push('verdict-to-incorrect movement missing on second page');
		if (!pageText.includes('#102')) failures.push('third-ranked candidate missing on second page');
		if (pageText.includes('#101') || pageText.includes('#103')) failures.push('first-page candidates leaked onto second page');
		if (rows.length !== 1) failures.push('second page should render exactly one candidate row');
		if (!previous || previous.getAttribute('href') !== '/runs/${PARENT}/repairs/${CHILD}?limit=2') failures.push('previous pagination link missing');
		if (!nextText) failures.push('disabled next state missing');
		return { url: location.href, failures };
	})()`;
}

async function main() {
	if (!existsSync(chromePath)) {
		throw new Error(`Chrome not found at ${chromePath}`);
	}

	const dir = await mkdtemp(join(tmpdir(), 'indra-repair-comparison-browser-'));
	const dbFile = join(dir, 'repair-comparison-browser.duckdb');
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

		const summary = { parent_run_id: PARENT, child_run_id: CHILD, checks: [], screenshots: [] };
		for (const viewport of viewports) {
			await client.send('Emulation.setDeviceMetricsOverride', {
				width: viewport.width,
				height: viewport.height,
				deviceScaleFactor: 1,
				mobile: viewport.mobile
			});
			const load = client.waitEvent('Page.loadEventFired', 20000).catch(() => null);
			await client.send('Page.navigate', { url: `${baseUrl}/runs/${PARENT}/repairs/${CHILD}?limit=2` });
			await load;
			await waitFor(client, `document.querySelector('[aria-label="full repair candidate before after lanes"] tbody tr')`, `${viewport.name} repair comparison render`);
			await evaluate(client, `new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve)))`);
			const audit = await evaluate(client, auditExpression(viewport.name));
			summary.checks.push(audit);
			summary.screenshots.push(await screenshot(client, `repair-comparison-${viewport.name}`));
		}

		await client.send('Emulation.setDeviceMetricsOverride', {
			width: 1280,
			height: 900,
			deviceScaleFactor: 1,
			mobile: false
		});
		const load = client.waitEvent('Page.loadEventFired', 20000).catch(() => null);
		await client.send('Page.navigate', { url: `${baseUrl}/runs/${PARENT}/repairs/${CHILD}?offset=2&limit=2` });
		await load;
		await waitFor(client, `document.querySelector('[aria-label="repair comparison pagination"]')`, 'second page repair comparison render');
		const secondPageAudit = await evaluate(client, secondPageAuditExpression());
		summary.checks.push({ viewport: 'desktop-second-page', ...secondPageAudit });

		const failures = summary.checks.flatMap((check) => check.failures.map((failure) => `${check.viewport}: ${failure}`));
		const result = { ...summary, failures };
		await writeFile(join(artifactRoot, 'repair_comparison_browser_summary.json'), JSON.stringify(result, null, 2));
		assert.deepEqual(failures, []);
		console.log('repair comparison browser tests passed');
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
