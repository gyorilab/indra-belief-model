import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { existsSync } from 'node:fs';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createServer } from 'vite';
import { DuckDBInstance } from '@duckdb/node-api';

const PAIR = 'pair_export_card_browser';
const MONO = 'mono_export_card_browser';
const DECOMP = 'decomp_export_card_browser';
const chromePath = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const profileDir = join(tmpdir(), `indra-pair-card-chrome-${process.pid}-${Date.now()}`);
const cdpPort = 9800 + Math.floor(Math.random() * 300);

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

function exportPayloadExpression() {
	return `(() => {
		const payload = (download) => {
			const link = Array.from(document.querySelectorAll('a[download]'))
				.find((a) => a.getAttribute('download') === download);
			if (!link) throw new Error('missing export link: ' + download);
			const href = link.getAttribute('href') ?? '';
			const comma = href.indexOf(',');
			if (!href.startsWith('data:') || comma < 0) throw new Error('invalid data URI for ' + download);
			return {
				download,
				media: href.slice(5, comma),
				body: decodeURIComponent(href.slice(comma + 1))
			};
		};
		return {
			json: payload(${JSON.stringify(`${PAIR}-comparison-card.json`)}),
			markdown: payload(${JSON.stringify(`${PAIR}-comparison-card.md`)}),
			schema: payload(${JSON.stringify(`${PAIR}-comparison-card.schema.json`)})
		};
	})()`;
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
			('${MONO}', 'card-v1', 'monolithic', '${PAIR}', 'demo-model', TIMESTAMP '2026-05-26 00:00:00', TIMESTAMP '2026-05-26 00:01:00', 'succeeded', 2, 0.08, 0.06),
			('${DECOMP}', 'card-v1', 'decomposed', '${PAIR}', 'demo-model', TIMESTAMP '2026-05-26 00:02:00', TIMESTAMP '2026-05-26 00:02:45', 'succeeded', 2, 0.05, 0.04)`);
		await con.run(`INSERT INTO statement VALUES
			('stmt_export_a', 'Activation', 0.80),
			('stmt_export_b', 'Inhibition', 0.20)`);
		await con.run(`INSERT INTO evidence VALUES
			('ev_export_a', 'stmt_export_a', 'reach', '1', 'rendered export evidence a'),
			('ev_export_b', 'stmt_export_b', 'reach', '2', 'rendered export evidence b')`);
		await con.run(`INSERT INTO agent VALUES
			('stmt_export_a', 'agent_a', 'subj', 'A', 0),
			('stmt_export_a', 'agent_b', 'obj', 'B', 1),
			('stmt_export_b', 'agent_c', 'subj', 'C', 0),
			('stmt_export_b', 'agent_d', 'obj', 'D', 1)`);
		await con.run(`INSERT INTO scorer_step VALUES
			('m_export_a', '${MONO}', 'card-v1', 'monolithic', 'stmt_export_a', 'ev_export_a', 'aggregate', NULL, '{"score":0.80,"verdict":"correct","confidence":"high","tier":"direct"}'::JSON, 100, 10, 5, NULL, TIMESTAMP '2026-05-26 00:00:10'),
			('m_export_b', '${MONO}', 'card-v1', 'monolithic', 'stmt_export_b', 'ev_export_b', 'aggregate', NULL, '{"score":0.30,"verdict":"incorrect","confidence":"medium","tier":"direct"}'::JSON, 200, 20, 10, NULL, TIMESTAMP '2026-05-26 00:00:20'),
			('d_export_a_probe', '${DECOMP}', 'card-v1', 'decomposed', 'stmt_export_a', 'ev_export_a', 'subject_role_probe', TRUE, '{"answer":"A"}'::JSON, 50, 5, 2, NULL, TIMESTAMP '2026-05-26 00:02:05'),
			('d_export_a', '${DECOMP}', 'card-v1', 'decomposed', 'stmt_export_a', 'ev_export_a', 'aggregate', NULL, '{"score":0.75,"verdict":"correct","confidence":"high"}'::JSON, 300, 40, 10, NULL, TIMESTAMP '2026-05-26 00:02:10'),
			('d_export_b', '${DECOMP}', 'card-v1', 'decomposed', 'stmt_export_b', 'ev_export_b', 'aggregate', NULL, '{"score":0.60,"verdict":"correct","confidence":"medium"}'::JSON, 400, 35, 15, NULL, TIMESTAMP '2026-05-26 00:02:20')`);
	} finally {
		con.disconnectSync?.();
		instance.closeSync();
	}
}

async function main() {
	if (!existsSync(chromePath)) {
		throw new Error(`Chrome not found at ${chromePath}`);
	}

	const dir = await mkdtemp(join(tmpdir(), 'indra-pair-card-browser-'));
	const dbFile = join(dir, 'pair-card.duckdb');
	process.env.VIEWER_DUCKDB_PATH = dbFile;
	process.env.INDRA_VIEWER_SUPPRESS_WRITER_GUARD_LOG = '1';

	let vite = null;
	let chrome = null;
	let client = null;
	let primaryError = null;

	try {
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
		await client.send('Emulation.setDeviceMetricsOverride', {
			width: 1280,
			height: 900,
			deviceScaleFactor: 1,
			mobile: false
		});

		const load = client.waitEvent('Page.loadEventFired', 20000).catch(() => null);
		await client.send('Page.navigate', { url: `${baseUrl}/pairs/${PAIR}` });
		await load;
		await waitFor(
			client,
			`(() => {
				try {
					const payloads = ${exportPayloadExpression()};
					const card = JSON.parse(payloads.json.body);
					return card.generated_at !== 'client-pending' && card.status === 'defined';
				} catch {
					return false;
				}
			})()`,
			'hydrated rendered comparison-card exports'
		);

		const payloads = await evaluate(client, exportPayloadExpression());
		assert.equal(payloads.json.download, `${PAIR}-comparison-card.json`);
		assert.equal(payloads.json.media, 'application/json;charset=utf-8');
		assert.equal(payloads.markdown.download, `${PAIR}-comparison-card.md`);
		assert.equal(payloads.markdown.media, 'text/markdown;charset=utf-8');
		assert.equal(payloads.schema.download, `${PAIR}-comparison-card.schema.json`);
		assert.equal(payloads.schema.media, 'application/schema+json;charset=utf-8');

		const card = JSON.parse(payloads.json.body);
		const schema = JSON.parse(payloads.schema.body);
		assert.equal(card.schema_version, 'indra_pair_comparison_card_v1');
		assert.equal(schema.properties.schema_version.const, card.schema_version);
		assert.ok(schema.$id.endsWith('/indra_pair_comparison_card_v1.schema.json'));
		assert.deepEqual([...schema.required].sort(), Object.keys(card).sort());
		assert.notEqual(card.generated_at, 'client-pending', 'rendered export should be hydrated before assertion');
		assert.match(card.generated_at, /^\d{4}-\d{2}-\d{2}T/);
		assert.equal(card.pair_id, PAIR);
		assert.equal(card.pair_href, `/pairs/${PAIR}`);
		assert.equal(card.status, 'defined');
		assert.equal(card.workflow_status, 'idle');
		assert.equal(card.runs.monolithic.run_id, MONO);
		assert.equal(card.runs.decomposed.run_id, DECOMP);
		assert.equal(card.overlap.overlap_evidences, 2);
		assert.equal(card.comparable.verdict_agreement_n, 1);
		assert.equal(card.resource_frontier.monolithic.clean_overlap_latency_observed_n, 2);
		assert.equal(card.resource_frontier.decomposed.clean_overlap_tokens_observed_n, 2);
		assert.equal(card.denominator_ledger.find((row) => row.key === 'resource_counter_metric')?.denominator_n, 2);
		assert.ok(card.guardrails.includes('Lower resource use is not a quality signal.'));
		assert.ok(card.guardrails.includes('Architecture-native diagnostics are not converted into the other architecture grammar.'));

		assert.match(payloads.markdown.body, new RegExp(`^# Paired Architecture Comparison: ${PAIR}`, 'm'));
		assert.match(payloads.markdown.body, /\| exact verdict agreement \| - \| - \| 1\/2 exact labels \| clean shared aggregate verdict evidence \|/);
		assert.match(payloads.markdown.body, /resource_counter_metric/);
		assert.match(payloads.markdown.body, /Lower resource use is not a quality signal/);
		assert.match(payloads.markdown.body, /Monolithic tier rows: 2/);
		assert.match(payloads.markdown.body, /Decomposed probe rows: 1/);
		assert.doesNotMatch(payloads.markdown.body, /client-pending/);
		console.log('paired comparison card browser export tests passed');
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
