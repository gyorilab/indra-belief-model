import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { existsSync } from 'node:fs';
import { mkdir, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { createServer } from 'vite';

const chromePath = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const fixtureRoot = resolve(process.cwd(), '..', 'data', 'corpora');
const benchmarkRoot = resolve(process.cwd(), '..', 'data', 'benchmark');
const fixtures = [
	{ role: 'ingest', filename: `zz_cycle76_cancel_ingest_${process.pid}.json`, hash: '987654321001' },
	{ role: 'score', filename: `zz_cycle76_cancel_score_${process.pid}.json`, hash: '987654321002' },
	{ role: 'paired', filename: `zz_cycle76_cancel_paired_${process.pid}.json`, hash: '987654321003' },
	{ role: 'truth_set', filename: `zz_cycle78_cancel_truth_${process.pid}.jsonl`, hash: '987654321004' }
];
const profileDir = join(tmpdir(), `indra-dashboard-cancel-chrome-${process.pid}-${Date.now()}`);
const cdpPort = 9700 + Math.floor(Math.random() * 400);

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
	const bindingCalls = [];

	ws.addEventListener('message', (event) => {
		const msg = JSON.parse(String(event.data));
		if (msg.id && pending.has(msg.id)) {
			const { resolve: resolvePending, reject } = pending.get(msg.id);
			pending.delete(msg.id);
			if (msg.error) reject(new Error(`${msg.error.message}${msg.error.data ? `: ${msg.error.data}` : ''}`));
			else resolvePending(msg.result ?? {});
			return;
		}
		if (msg.method) {
			if (msg.method === 'Runtime.bindingCalled') {
				bindingCalls.push(msg.params ?? {});
			}
			if (eventWaiters.has(msg.method)) {
				const waiters = eventWaiters.get(msg.method);
				eventWaiters.delete(msg.method);
				for (const resolveWaiter of waiters) resolveWaiter(msg.params ?? {});
			}
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
				const waiters = eventWaiters.get(method) ?? [];
				waiters.push(resolveWaiter);
				eventWaiters.set(method, waiters);
				setTimeout(() => {
					const active = eventWaiters.get(method) ?? [];
					const remaining = active.filter((fn) => fn !== resolveWaiter);
					if (remaining.length > 0) eventWaiters.set(method, remaining);
					else eventWaiters.delete(method);
					reject(new Error(`Timed out waiting for ${method}`));
				}, timeoutMs);
			});
		},
		close() {
			ws.close();
		},
		bindingPayloads(name) {
			return bindingCalls
				.filter((call) => call.name === name)
				.map((call) => {
					try {
						return JSON.parse(call.payload);
					} catch {
						return { raw: call.payload };
					}
				});
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

function fixtureDocument(hash, label) {
	return JSON.stringify({
		statements: [
			{
				type: 'Activation',
				matches_hash: hash,
				belief: 0.42,
				evidence: [
					{
						source_api: 'cancel_smoke',
						text: `dashboard cancellation fixture ${label}`
					}
				],
				subj: { name: 'A' },
				obj: { name: 'B' }
			}
		]
	});
}

function truthSetFixtureLine(hash, label) {
	return `${JSON.stringify({
		matches_hash: hash,
		tag: 'correct',
		source_api: 'cancel_smoke',
		text: `dashboard cancellation benchmark ${label}`
	})}\n`;
}

function installFetchHarnessExpression(label) {
	return `(() => {
		const label = ${JSON.stringify(label)};
		const record = (event) => {
			const payload = { label, ...event };
			try { window.__indraCancelSmokeRecord?.(JSON.stringify(payload)); } catch {}
			return payload;
		};
		const state = {
			requests: [],
			aborts: [],
			cancelPosts: [],
			estimates: []
		};
		const originalFetch = window.fetch.bind(window);
		const jsonResponse = (body, status = 200) => Promise.resolve(new Response(JSON.stringify(body), {
			status,
			headers: { 'content-type': 'application/json' }
		}));
		const pendingRun = (kind, init = {}) => {
			state.requests.push(record({ type: 'request', kind, keepalive: Boolean(init.keepalive) }));
			return new Promise((resolve, reject) => {
				const abort = () => {
					state.aborts.push(record({ type: 'abort', kind }));
					reject(new DOMException('Aborted', 'AbortError'));
				};
				if (init.signal?.aborted) {
					abort();
					return;
				}
				init.signal?.addEventListener('abort', abort, { once: true });
			});
		};
		window.fetch = (input, init = {}) => {
			const raw = typeof input === 'string' ? input : input.url;
			const url = new URL(raw, location.href);
			if (url.pathname === '/api/runs/estimate-cost') {
				const body = typeof init.body === 'string' ? JSON.parse(init.body) : {};
				const arch = body.arch === 'monolithic' ? 'monolithic' : 'decomposed';
				state.estimates.push({ arch });
				return jsonResponse({
					summary: {
						estimates: [{
							model_id: 'cancel-smoke-model',
							architecture: arch,
							cost_usd: arch === 'monolithic' ? 0.02 : 0.04,
							n_stmts: 1,
							n_evidences_est: 1,
							n_llm_calls_est: arch === 'monolithic' ? 1 : 4
						}]
					}
				});
			}
			if (url.pathname === '/api/datasets/ingest') return pendingRun('ingest', init);
			if (url.pathname === '/api/truth-sets') return pendingRun('truth_set', init);
			if (url.pathname === '/api/runs/score') return pendingRun('score', init);
			if (url.pathname === '/api/runs/score-paired') return pendingRun('paired', init);
			if (/^\\/api\\/runs\\/score-paired\\/[^/]+\\/cancel$/.test(url.pathname)) {
				state.cancelPosts.push(record({ type: 'paired_cancel', path: url.pathname, method: init.method ?? 'GET', keepalive: Boolean(init.keepalive) }));
				return jsonResponse({ ok: true, pair_id: url.pathname.split('/').at(-2), killed: [], tombstones: [] });
			}
			return originalFetch(input, init);
		};
		window.__indraCancelSmoke = state;
		window.__indraDispatchPagehide = (persisted) => {
			const ev = new PageTransitionEvent('pagehide', { persisted });
			window.dispatchEvent(ev);
		};
		window.addEventListener('pagehide', (event) => {
			record({ type: 'pagehide', persisted: event.persisted });
		}, { capture: true });
		return true;
	})()`;
}

function rowButtonClickExpression(filename, buttonPattern) {
	return `(() => {
		const rows = Array.from(document.querySelectorAll('.ds-row'));
		const row = rows.find((r) => r.querySelector('.ds-name')?.textContent?.trim() === ${JSON.stringify(filename)});
		if (!row) throw new Error('dataset row not found: ${filename}');
		const pattern = new RegExp(${JSON.stringify(buttonPattern)});
		const button = Array.from(row.querySelectorAll('button')).find((b) => pattern.test(b.textContent ?? ''));
		if (!button) throw new Error('button not found for ${filename}: ${buttonPattern}');
		if (button.disabled) throw new Error('button disabled for ${filename}: ' + button.textContent);
		button.click();
		return true;
	})()`;
}

function rowTextExpression(filename) {
	return `(() => {
		const row = Array.from(document.querySelectorAll('.ds-row')).find((r) => r.querySelector('.ds-name')?.textContent?.trim() === ${JSON.stringify(filename)});
		return (row?.textContent ?? '').replace(/\\s+/g, ' ').trim();
	})()`;
}

async function navigateToDashboard(client, baseUrl, label = 'dashboard') {
	const load = client.waitEvent('Page.loadEventFired', 20000).catch(() => null);
	await client.send('Page.navigate', { url: `${baseUrl}/?cancel_smoke=${encodeURIComponent(label)}#datasets` });
	await load;
	await waitFor(
		client,
		`Array.from(document.querySelectorAll('.ds-name')).some((el) => el.textContent.trim() === ${JSON.stringify(fixtures[2].filename)}) && Array.from(document.querySelectorAll('.ds-name')).some((el) => el.textContent.trim() === ${JSON.stringify(fixtures[3].filename)})`,
		'fixture dataset rows'
	);
	await evaluate(client, `new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve)))`);
	await sleep(500);
}

function fixtureForRole(role) {
	const fixture = fixtures.find((f) => f.role === role);
	if (!fixture) throw new Error(`missing fixture for role ${role}`);
	return fixture;
}

async function startDashboardStream(client, role) {
	const fixture = fixtureForRole(role);
	if (role === 'ingest') {
		await evaluate(client, rowButtonClickExpression(fixture.filename, 'ingest into corpus'));
		await waitFor(client, `${rowTextExpression(fixture.filename)}.includes('parsing')`, 'ingest running state');
		return;
	}
	if (role === 'score') {
		await evaluate(client, rowButtonClickExpression(fixture.filename, 'preview .*scoring cost'));
		await waitFor(client, `${rowTextExpression(fixture.filename)}.includes('spend up to')`, 'single score preflight');
		await evaluate(client, rowButtonClickExpression(fixture.filename, 'spend up to'));
		await waitFor(client, `${rowTextExpression(fixture.filename)}.includes('spending on cancel-smoke-model')`, 'single score running state');
		return;
	}
	if (role === 'paired') {
		await evaluate(client, rowButtonClickExpression(fixture.filename, 'preview paired'));
		await waitFor(client, `${rowTextExpression(fixture.filename)}.includes('start pair with')`, 'paired preflight');
		await evaluate(client, rowButtonClickExpression(fixture.filename, 'start pair'));
		await waitFor(client, `${rowTextExpression(fixture.filename)}.includes('cancel paired run')`, 'paired running state');
		return;
	}
	if (role === 'truth_set') {
		await evaluate(client, rowButtonClickExpression(fixture.filename, 'register `tag` as truth_set'));
		await waitFor(client, `${rowTextExpression(fixture.filename)}.includes('registering truth_set')`, 'truth-set register running state');
		return;
	}
	throw new Error(`unknown stream role ${role}`);
}

function runningTextPattern(role) {
	if (role === 'ingest') return /parsing/;
	if (role === 'score') return /spending on cancel-smoke-model/;
	if (role === 'paired') return /cancel paired run/;
	if (role === 'truth_set') return /registering truth_set/;
	throw new Error(`unknown stream role ${role}`);
}

async function assertPageExitMessage(client, role) {
	const fixture = fixtureForRole(role);
	const text = rowTextExpression(fixture.filename);
	if (role === 'paired') {
		await waitFor(
			client,
			`${text}.includes('cancelled pair') && ${text}.includes('local stream closed because the page closed or navigated away')`,
			'paired page-exit message'
		);
		return;
	}
	if (role === 'truth_set') {
		await waitFor(
			client,
			`${text}.includes('local stream closed because the page closed or navigated away') && ${text}.includes('truth_set registration')`,
			'truth-set page-exit message'
		);
		return;
	}
	await waitFor(
		client,
		`${text}.includes('local stream closed because the page closed or navigated away')`,
		`${role} page-exit message`
	);
}

function recordedEvents(client, label) {
	return client.bindingPayloads('__indraCancelSmokeRecord').filter((event) => event.label === label);
}

async function waitForRecordedEvents(client, label, predicate, description, timeoutMs = 10000) {
	const deadline = Date.now() + timeoutMs;
	while (Date.now() < deadline) {
		const events = recordedEvents(client, label);
		if (predicate(events)) return events;
		await sleep(100);
	}
	throw new Error(`${description} did not become true; events=${JSON.stringify(recordedEvents(client, label))}`);
}

async function main() {
	if (!existsSync(chromePath)) {
		throw new Error(`Chrome not found at ${chromePath}`);
	}

	let vite = null;
	let chrome = null;
	let client = null;
	let primaryError = null;
	let fixtureLeftovers = [];

	try {
		await mkdir(fixtureRoot, { recursive: true });
		await mkdir(benchmarkRoot, { recursive: true });
		for (const f of fixtures.slice(0, 3)) {
			await writeFile(join(fixtureRoot, f.filename), fixtureDocument(f.hash, f.role));
		}
		await writeFile(join(benchmarkRoot, fixtures[3].filename), truthSetFixtureLine(fixtures[3].hash, fixtures[3].role));

		vite = await createServer({
			configFile: './vite.config.ts',
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
			await client.send('Runtime.addBinding', { name: '__indraCancelSmokeRecord' });
			await client.send('Emulation.setDeviceMetricsOverride', {
				width: 1280,
				height: 900,
				deviceScaleFactor: 1,
				mobile: false
			});
			const roles = ['ingest', 'score', 'paired', 'truth_set'];
			for (const role of roles) {
				const label = `synthetic-pagehide-${role}`;
				const fixture = fixtureForRole(role);
				await navigateToDashboard(client, baseUrl, label);
				await evaluate(client, installFetchHarnessExpression(label));
				await startDashboardStream(client, role);

				await evaluate(client, `window.__indraDispatchPagehide(true)`);
				await sleep(250);
				const bfcacheState = await evaluate(client, `window.__indraCancelSmoke`);
				assert.deepEqual(bfcacheState.aborts, [], `BFCache pagehide must not abort ${role}`);
				assert.deepEqual(bfcacheState.cancelPosts, [], `BFCache pagehide must not emit paired cancel POST for ${role}`);
				assert.match(await evaluate(client, rowTextExpression(fixture.filename)), runningTextPattern(role));

				await evaluate(client, `window.__indraDispatchPagehide(false)`);
				await waitFor(client, `window.__indraCancelSmoke.aborts.length === 1`, `destructive pagehide aborts ${role}`);
				if (role === 'paired') {
					await waitFor(client, `window.__indraCancelSmoke.cancelPosts.length === 1`, 'destructive pagehide sends one paired cancel POST');
				}
				await assertPageExitMessage(client, role);
				await evaluate(client, `window.__indraDispatchPagehide(false)`);
				await sleep(250);
				const finalState = await evaluate(client, `window.__indraCancelSmoke`);
				assert.deepEqual(
					finalState.aborts.map((r) => r.kind),
					[role],
					`destructive pagehide aborts ${role} exactly once`
				);
				assert.deepEqual(
					finalState.requests.map((r) => r.kind),
					[role],
					`test harness observed the active ${role} request`
				);
				if (role === 'paired') {
					assert.equal(finalState.cancelPosts.length, 1, 'paired cancel POST is not duplicated by a second pagehide');
					assert.equal(finalState.cancelPosts[0].method, 'POST', 'paired page-exit cancel uses POST');
					assert.equal(finalState.cancelPosts[0].keepalive, true, 'paired page-exit cancel uses keepalive');
					assert.deepEqual(
						finalState.estimates.map((r) => r.arch).sort(),
						['decomposed', 'monolithic'],
						'paired cost preflight exercised both architecture estimates'
					);
				} else {
					assert.equal(finalState.cancelPosts.length, 0, `${role} page-exit does not emit paired cancel POST`);
					if (role === 'score') {
						assert.deepEqual(finalState.estimates.map((r) => r.arch), ['decomposed'], 'single-score cost preflight exercised decomposed estimate');
					}
				}
			}

			const blankBeforeRealNavigation = client.waitEvent('Page.loadEventFired', 20000).catch(() => null);
			await client.send('Page.navigate', { url: 'about:blank' });
			await blankBeforeRealNavigation;
			for (const role of roles) {
				const label = `route-navigation-${role}`;
				await navigateToDashboard(client, baseUrl, label);
				await evaluate(client, installFetchHarnessExpression(label));
				await startDashboardStream(client, role);
				const unexpectedFullLoad = client.waitEvent('Page.loadEventFired', 1500)
					.then(() => true)
					.catch(() => false);
				await evaluate(client, `document.querySelector('a.nav-link')?.click()`);
				await waitFor(client, `location.pathname === '/statements'`, `dashboard route navigation for ${role}`);
				assert.equal(await unexpectedFullLoad, false, `route navigation for ${role} stays inside SvelteKit and does not perform a full page load`);
				const navigationEvents = await waitForRecordedEvents(
					client,
					label,
					(events) =>
						events.filter((event) => event.type === 'abort').length === 1 &&
						events.filter((event) => event.type === 'paired_cancel').length === (role === 'paired' ? 1 : 0),
					`route navigation abort/cancel binding events for ${role}`
				);
				assert.equal(
					navigationEvents.filter((event) => event.type === 'pagehide').length,
					0,
					`route navigation for ${role} does not use the pagehide path`
				);
				assert.deepEqual(
					navigationEvents.filter((event) => event.type === 'abort').map((event) => event.kind),
					[role],
					`route navigation aborts ${role} once`
				);
				const realCancelEvents = navigationEvents.filter((event) => event.type === 'paired_cancel');
				if (role === 'paired') {
					assert.equal(realCancelEvents[0].method, 'POST', 'route navigation paired cancel uses POST');
					assert.equal(realCancelEvents[0].keepalive, true, 'route navigation paired cancel uses keepalive');
				}
				await sleep(300);
				const settledNavigationEvents = recordedEvents(client, label);
				assert.equal(
					settledNavigationEvents.filter((event) => event.type === 'pagehide').length,
					0,
					`route navigation for ${role} still has no late pagehide records`
				);
				assert.deepEqual(
					settledNavigationEvents.filter((event) => event.type === 'abort').map((event) => event.kind),
					[role],
					`route navigation for ${role} does not emit late duplicate aborts`
				);
				assert.equal(
					settledNavigationEvents.filter((event) => event.type === 'paired_cancel').length,
					role === 'paired' ? 1 : 0,
					`route navigation for ${role} does not emit late duplicate paired cancel`
				);
			}
		} catch (err) {
		primaryError = err;
	} finally {
		client?.close();
		if (chrome) {
			chrome.kill('SIGTERM');
			await sleep(300);
			if (chrome.exitCode === null) chrome.kill('SIGKILL');
		}
		if (vite) await vite.close();
		await rm(profileDir, { recursive: true, force: true });
		for (const f of fixtures.slice(0, 3)) {
			await rm(join(fixtureRoot, f.filename), { force: true });
		}
		await rm(join(benchmarkRoot, fixtures[3].filename), { force: true });
		fixtureLeftovers = [
			...fixtures.slice(0, 3).map((f) => join(fixtureRoot, f.filename)),
			join(benchmarkRoot, fixtures[3].filename)
		]
			.filter((path) => existsSync(path));
	}
	if (primaryError) throw primaryError;
	assert.deepEqual(fixtureLeftovers, [], 'temporary dashboard cancel fixtures are removed');
	console.log('dashboard cancel browser smoke passed');
}

main().catch((err) => {
	console.error(err);
	process.exitCode = 1;
});
