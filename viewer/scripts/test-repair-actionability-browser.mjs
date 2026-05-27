import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { existsSync } from 'node:fs';
import { chmod, mkdir, mkdtemp, readdir, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createServer } from 'vite';
import { DuckDBInstance } from '@duckdb/node-api';

const RUN = 'cc000000000000000000000000000121';
const chromePath = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const artifactRoot = join(process.env.TMPDIR ?? tmpdir(), 'indra-agent-trace-hypergraph');
const profileDir = join(tmpdir(), `indra-repair-actionability-chrome-${process.pid}-${Date.now()}`);
const cdpPort = 10200 + Math.floor(Math.random() * 300);
const reviewNote = 'browser clicked object and scope probes';
const ACTIVE_CHILD = 'dd000000000000000000000000000123';
const TOMBSTONE_SOURCE_DUMP = 'repair_tombstone_browser';
const LAUNCH_MODEL = 'browser-repair-smoke';
const CANCEL_MODEL = 'browser-repair-cancel-smoke';
const RECOVERY_CHILD = 'ee000000000000000000000000000124';
const RECOVERY_SOURCE_DUMP = 'repair_recovery_browser';
const statementRawJson = {
	type: 'Activation',
	subj: { name: 'BRAF', db_refs: {} },
	obj: { name: 'MAPK1', db_refs: {} },
	obj_activity: 'activity',
	belief: 0.35,
	evidence: [{
		source_api: 'reach',
		pmid: 'PMID-BROWSER',
		text: 'browser actionability probe evidence',
		text_refs: { PMID: 'PMID-BROWSER' },
		source_hash: 121121
	}],
	id: 'repair-actionability-statement',
	matches_hash: '121121'
};
const tombstoneStatementRawJson = {
	type: 'Activation',
	subj: { name: 'EGFR', db_refs: {} },
	obj: { name: 'ERK', db_refs: {} },
	obj_activity: 'activity',
	belief: 0.62,
	evidence: [{
		source_api: 'reach',
		pmid: 'PMID-TOMBSTONE',
		text: 'EGFR activates ERK in the tombstone fixture.',
		text_refs: { PMID: 'PMID-TOMBSTONE' },
		source_hash: 123123
	}],
	id: 'repair-tombstone-statement',
	matches_hash: '123123'
};
const recoveryStatementRawJson = {
	type: 'Activation',
	subj: { name: 'MEK1', db_refs: {} },
	obj: { name: 'ERK2', db_refs: {} },
	obj_activity: 'activity',
	belief: 0.44,
	evidence: [{
		source_api: 'reach',
		pmid: 'PMID-RECOVERY',
		text: 'MEK1 activates ERK2 in the recovery fixture.',
		text_refs: { PMID: 'PMID-RECOVERY' },
		source_hash: 124124
	}],
	id: 'repair-recovery-statement',
	matches_hash: '124124'
};

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
			run_id VARCHAR PRIMARY KEY,
			scorer_version VARCHAR NOT NULL,
			indra_version VARCHAR NOT NULL,
			architecture VARCHAR NOT NULL DEFAULT 'unknown',
			paired_run_group_id VARCHAR,
			parent_run_id VARCHAR,
			model_id_default VARCHAR,
			started_at TIMESTAMP NOT NULL,
			finished_at TIMESTAMP,
			n_stmts INTEGER,
			status VARCHAR NOT NULL,
			cost_estimate_usd DOUBLE,
			cost_actual_usd DOUBLE,
			reviewed_at TIMESTAMP,
			reviewed_by VARCHAR,
			review_status VARCHAR,
			review_notes TEXT,
			terminated_by VARCHAR,
			termination_reason TEXT,
			notes TEXT
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
		await con.run(`CREATE TABLE agent (
			stmt_hash VARCHAR,
			name VARCHAR,
			role VARCHAR,
			role_index BIGINT
		)`);
		await con.run(`CREATE TABLE scorer_step (
			step_hash VARCHAR,
			run_id VARCHAR,
			stmt_hash VARCHAR,
			evidence_hash VARCHAR,
			architecture VARCHAR,
			model_id VARCHAR,
			step_kind VARCHAR,
			output_json JSON,
			error VARCHAR,
			latency_ms BIGINT,
			prompt_tokens BIGINT,
			out_tokens BIGINT,
			started_at TIMESTAMP
		)`);
		await con.run(`CREATE SEQUENCE scorer_step_correction_id_seq START 1000`);
		await con.run(`CREATE TABLE scorer_step_correction (
			correction_id BIGINT PRIMARY KEY DEFAULT nextval('scorer_step_correction_id_seq'),
			step_hash VARCHAR NOT NULL,
			run_id VARCHAR NOT NULL,
			architecture VARCHAR NOT NULL DEFAULT 'unknown',
			stmt_hash VARCHAR NOT NULL,
			evidence_hash VARCHAR,
			correction_kind VARCHAR NOT NULL,
			status VARCHAR NOT NULL DEFAULT 'open',
			reviewer VARCHAR,
			note TEXT,
			value_json JSON,
			parent_correction_id BIGINT,
			child_run_id VARCHAR,
			repair_source_dump_id VARCHAR,
			materialize_batch_id VARCHAR,
			source_route VARCHAR,
			source_filters_json JSON,
			created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
		)`);
		await con.run(`INSERT INTO score_run
			(run_id, scorer_version, indra_version, architecture, status, parent_run_id, started_at, finished_at, n_stmts, model_id_default)
			VALUES
			('${RUN}', 'repair-actionability', 'test-indra', 'decomposed', 'succeeded', NULL, TIMESTAMP '2026-05-26 01:00:00', TIMESTAMP '2026-05-26 01:00:30', 2, 'smoke-local'),
			('${ACTIVE_CHILD}', 'repair-actionability-child', 'test-indra', 'decomposed', 'running', '${RUN}', TIMESTAMP '2026-05-26 01:10:00', NULL, 1, 'smoke-local'),
			('${RECOVERY_CHILD}', 'repair-actionability-recovery-child', 'test-indra', 'decomposed', 'succeeded', '${RUN}', TIMESTAMP '2026-05-26 01:12:00', TIMESTAMP '2026-05-26 01:12:30', 1, 'smoke-local')`);
		await con.run(
			`INSERT INTO statement VALUES
			 ('stmt_probe_browser', 'Activation', 0.35, 0, 0, ?::JSON)`,
			[JSON.stringify(statementRawJson)]
		);
		await con.run(
			`INSERT INTO statement VALUES
			 ('stmt_tombstone_browser', 'Activation', 0.62, 0, 0, ?::JSON)`,
			[JSON.stringify(tombstoneStatementRawJson)]
		);
		await con.run(
			`INSERT INTO statement VALUES
			 ('stmt_recovery_browser', 'Activation', 0.44, 0, 0, ?::JSON)`,
			[JSON.stringify(recoveryStatementRawJson)]
		);
		await con.run(`INSERT INTO evidence VALUES
			('ev_probe_browser', 'stmt_probe_browser', 'reach', 'PMID-BROWSER', 'browser actionability probe evidence'),
			('ev_tombstone_browser', 'stmt_tombstone_browser', 'reach', 'PMID-TOMBSTONE', 'browser actionability tombstone evidence'),
			('ev_recovery_browser', 'stmt_recovery_browser', 'reach', 'PMID-RECOVERY', 'browser actionability recovery evidence')`);
		await con.run(`INSERT INTO agent VALUES
			('stmt_probe_browser', 'BRAF', 'subj', 0),
			('stmt_probe_browser', 'MAPK1', 'obj', 1),
			('stmt_tombstone_browser', 'EGFR', 'subj', 0),
			('stmt_tombstone_browser', 'ERK', 'obj', 1),
			('stmt_recovery_browser', 'MEK1', 'subj', 0),
			('stmt_recovery_browser', 'ERK2', 'obj', 1)`);
		await con.run(`INSERT INTO scorer_step VALUES
			('route_probe_browser', '${RUN}', 'stmt_probe_browser', 'ev_probe_browser', 'decomposed', 'smoke-local', 'substrate_route',
			 '{"subject_role":{"source":"substrate","answer":"yes"},"object_role":{"source":"needs_llm","answer":null},"relation_axis":{"source":"needs_llm","answer":null},"scope":{"source":null,"answer":null}}'::JSON,
			 NULL, 40, 4, 2, TIMESTAMP '2026-05-26 01:00:05'),
			('subject_probe_browser', '${RUN}', 'stmt_probe_browser', 'ev_probe_browser', 'decomposed', 'smoke-local', 'subject_role_probe',
			 '{"source":"substrate","answer":"present_as_subject"}'::JSON,
			 NULL, 45, 4, 2, TIMESTAMP '2026-05-26 01:00:06'),
			('agg_tombstone_browser', '${RUN}', 'stmt_tombstone_browser', 'ev_tombstone_browser', 'decomposed', 'smoke-local', 'aggregate',
			 '{"score":0.20,"verdict":"incorrect","confidence":"low"}'::JSON,
			 NULL, 50, 5, 2, TIMESTAMP '2026-05-26 01:00:07'),
			('agg_recovery_browser', '${RUN}', 'stmt_recovery_browser', 'ev_recovery_browser', 'decomposed', 'smoke-local', 'aggregate',
			 '{"score":0.30,"verdict":"incorrect","confidence":"low"}'::JSON,
			 NULL, 52, 5, 2, TIMESTAMP '2026-05-26 01:00:08'),
			('agg_recovery_child_browser', '${RECOVERY_CHILD}', 'stmt_recovery_browser', 'ev_recovery_browser', 'decomposed', 'smoke-local', 'aggregate',
			 '{"score":0.82,"verdict":"correct","confidence":"medium","fixture":"repair_actionability_recovery"}'::JSON,
			 NULL, 54, 6, 2, TIMESTAMP '2026-05-26 01:12:10')`);
		await con.run(
			`INSERT INTO scorer_step_correction
			 (correction_id, step_hash, run_id, architecture, stmt_hash, evidence_hash,
			  correction_kind, status, reviewer, note, value_json, source_route, source_filters_json, created_at)
			 VALUES
			 (121, 'route_probe_browser', '${RUN}', 'decomposed', 'stmt_probe_browser', 'ev_probe_browser',
			  'repair_candidate', 'open', 'browser-test', 'probe candidate before browser review', ?::JSON,
			  '/runs/${RUN}/cohort?probe_coverage=present', '{"probe_coverage":"present"}'::JSON,
			  TIMESTAMP '2026-05-26 01:01:00')`,
			[JSON.stringify({
				suspected_step_kind: 'probe_coverage',
				severity: 'medium',
				reviewer_hypothesis: 'Browser reviewer should target missing object and scope slots',
				observed: {
					trace_state: 'missing_aggregate',
					probe_coverage: 'present',
					missing_probe_slots: 'object_role_probe,relation_axis_probe,scope_probe',
					probe_counts: {
						substrate_route: 1,
						subject_role_probe: 1,
						object_role_probe: 0,
						relation_axis_probe: 0,
						scope_probe: 0
					}
				}
			})]
		);
		await con.run(
			`INSERT INTO scorer_step_correction
			 (correction_id, step_hash, run_id, architecture, stmt_hash, evidence_hash,
			  correction_kind, status, reviewer, note, value_json, source_route, source_filters_json, created_at)
			 VALUES
			 (122, 'agg_tombstone_browser', '${RUN}', 'decomposed', 'stmt_tombstone_browser', 'ev_tombstone_browser',
			  'repair_candidate', 'open', 'browser-test', 'active child should be tombstoned from browser', ?::JSON,
			  '/runs/${RUN}/cohort?case=tombstone-browser', '{"case":"tombstone-browser"}'::JSON,
			  TIMESTAMP '2026-05-26 01:01:30')`,
			[JSON.stringify({
				suspected_step_kind: 'aggregate',
				severity: 'high',
				reviewer_hypothesis: 'Browser reviewer should be able to release stale active child lock',
				observed: {
					trace_state: 'native_decomposed',
					verdict: 'incorrect',
					confidence: 'low',
					score: 0.2,
					residual: -0.42
				}
			})]
		);
		await con.run(
			`INSERT INTO scorer_step_correction
			 (correction_id, step_hash, run_id, architecture, stmt_hash, evidence_hash,
			  correction_kind, status, reviewer, note, value_json,
			  parent_correction_id, child_run_id, repair_source_dump_id,
			  source_route, source_filters_json, created_at)
			 VALUES
			 (123, 'agg_tombstone_browser', '${RUN}', 'decomposed', 'stmt_tombstone_browser', 'ev_tombstone_browser',
			  'rerun_intent', 'open', 'viewer', 'active child lock for browser tombstone', ?::JSON,
			  122, '${ACTIVE_CHILD}', '${TOMBSTONE_SOURCE_DUMP}',
			  '/runs/${ACTIVE_CHILD}', NULL, TIMESTAMP '2026-05-26 01:02:00')`,
			[JSON.stringify({
				kind: 'repair_rerun_intent',
				parent_correction_id: 122,
				parent_run_id: RUN,
				child_run_id: ACTIVE_CHILD,
				source_dump_id: TOMBSTONE_SOURCE_DUMP,
				architecture: 'decomposed'
			})]
		);
		await con.run(
			`INSERT INTO scorer_step_correction
			 (correction_id, step_hash, run_id, architecture, stmt_hash, evidence_hash,
			  correction_kind, status, reviewer, note, value_json, source_route, source_filters_json, created_at)
			 VALUES
			 (124, 'agg_recovery_browser', '${RUN}', 'decomposed', 'stmt_recovery_browser', 'ev_recovery_browser',
			  'repair_candidate', 'open', 'browser-test', 'completed child should be finalized from browser', ?::JSON,
			  '/runs/${RUN}/cohort?case=recovery-browser', '{"case":"recovery-browser"}'::JSON,
			  TIMESTAMP '2026-05-26 01:03:00')`,
			[JSON.stringify({
				suspected_step_kind: 'aggregate',
				severity: 'medium',
				reviewer_hypothesis: 'Browser reviewer should be able to finalize succeeded child markers without rescoring',
				observed: {
					trace_state: 'native_decomposed',
					verdict: 'incorrect',
					confidence: 'low',
					score: 0.3,
					residual: -0.14
				}
			})]
		);
		await con.run(
			`INSERT INTO scorer_step_correction
			 (correction_id, step_hash, run_id, architecture, stmt_hash, evidence_hash,
			  correction_kind, status, reviewer, note, value_json,
			  parent_correction_id, child_run_id, repair_source_dump_id,
			  source_route, source_filters_json, created_at)
			 VALUES
			 (125, 'agg_recovery_browser', '${RUN}', 'decomposed', 'stmt_recovery_browser', 'ev_recovery_browser',
			  'rerun_intent', 'open', 'viewer', 'completed child awaiting marker recovery', ?::JSON,
			  124, '${RECOVERY_CHILD}', '${RECOVERY_SOURCE_DUMP}',
			  '/runs/${RECOVERY_CHILD}', NULL, TIMESTAMP '2026-05-26 01:03:30')`,
			[JSON.stringify({
				kind: 'repair_rerun_intent',
				parent_correction_id: 124,
				parent_run_id: RUN,
				child_run_id: RECOVERY_CHILD,
				source_dump_id: RECOVERY_SOURCE_DUMP,
				architecture: 'decomposed'
			})]
		);
	} finally {
		con.disconnectSync?.();
		instance.closeSync();
	}
}

async function probeSlotRows(dbFile) {
	const instance = await DuckDBInstance.create(dbFile);
	const con = await instance.connect();
	try {
		const reader = await con.runAndReadAll(
			`SELECT
			   correction_id,
			   parent_correction_id,
			   correction_kind,
			   status,
			   reviewer,
			   note,
			   json_extract_string(value_json, '$.selected_probe_slots') AS selected_probe_slots,
			   json_extract_string(value_json, '$.reviewer_note') AS reviewer_note
			 FROM scorer_step_correction
			WHERE correction_kind='probe_slot_review'
			ORDER BY correction_id`
		);
		return reader.getRowObjects().map((row) => ({
			correction_id: Number(row.correction_id),
			parent_correction_id: Number(row.parent_correction_id),
			correction_kind: String(row.correction_kind),
			status: String(row.status),
			reviewer: row.reviewer == null ? null : String(row.reviewer),
			note: row.note == null ? null : String(row.note),
			selected_probe_slots: row.selected_probe_slots == null ? null : String(row.selected_probe_slots),
			reviewer_note: row.reviewer_note == null ? null : String(row.reviewer_note)
		}));
	} finally {
		con.disconnectSync?.();
		instance.closeSync();
	}
}

async function repairStateRows(dbFile) {
	const instance = await DuckDBInstance.create(dbFile);
	const con = await instance.connect();
	try {
		const statusReader = await con.runAndReadAll(
			`SELECT run_id, status, terminated_by, termination_reason
			   FROM score_run
			  WHERE run_id IN (?, ?)
			  ORDER BY run_id`,
			[RUN, ACTIVE_CHILD]
		);
		const correctionReader = await con.runAndReadAll(
			`SELECT correction_id, correction_kind, status, parent_correction_id, child_run_id, repair_source_dump_id
			   FROM scorer_step_correction
			  WHERE correction_id IN (122, 123)
			     OR parent_correction_id=122
			  ORDER BY correction_id`
		);
		return {
			runs: statusReader.getRowObjects().map((row) => ({
				run_id: String(row.run_id),
				status: String(row.status),
				terminated_by: row.terminated_by == null ? null : String(row.terminated_by),
				termination_reason: row.termination_reason == null ? null : String(row.termination_reason)
			})),
			corrections: correctionReader.getRowObjects().map((row) => ({
				correction_id: Number(row.correction_id),
				correction_kind: String(row.correction_kind),
				status: String(row.status),
				parent_correction_id: row.parent_correction_id == null ? null : Number(row.parent_correction_id),
				child_run_id: row.child_run_id == null ? null : String(row.child_run_id),
				repair_source_dump_id: row.repair_source_dump_id == null ? null : String(row.repair_source_dump_id)
			}))
		};
	} finally {
		con.disconnectSync?.();
		instance.closeSync();
	}
}

async function repairLaunchRows(dbFile) {
	const instance = await DuckDBInstance.create(dbFile);
	const con = await instance.connect();
	try {
		const runReader = await con.runAndReadAll(
			`SELECT run_id, parent_run_id, status, architecture, model_id_default, cost_actual_usd,
			        terminated_by, termination_reason
			   FROM score_run
			  WHERE parent_run_id=?
			    AND run_id<>?
			  ORDER BY started_at, run_id`,
			[RUN, ACTIVE_CHILD]
		);
		const correctionReader = await con.runAndReadAll(
			`SELECT correction_id, correction_kind, status, parent_correction_id, child_run_id, repair_source_dump_id
			   FROM scorer_step_correction
			  WHERE correction_id IN (121, 122, 124, 125)
			     OR parent_correction_id IN (121, 122, 124)
			  ORDER BY correction_id`
		);
		const stepReader = await con.runAndReadAll(
			`SELECT run_id, stmt_hash, evidence_hash, step_kind, json_extract_string(output_json, '$.fixture') AS fixture
			   FROM scorer_step
			  WHERE (stmt_hash='stmt_probe_browser' AND evidence_hash='ev_probe_browser')
			     OR (stmt_hash='stmt_recovery_browser' AND evidence_hash='ev_recovery_browser')
			  ORDER BY run_id, step_kind`
		);
		return {
			runs: runReader.getRowObjects().map((row) => ({
				run_id: String(row.run_id),
				parent_run_id: row.parent_run_id == null ? null : String(row.parent_run_id),
				status: String(row.status),
				architecture: String(row.architecture),
				model_id_default: row.model_id_default == null ? null : String(row.model_id_default),
				cost_actual_usd: row.cost_actual_usd == null ? null : Number(row.cost_actual_usd),
				terminated_by: row.terminated_by == null ? null : String(row.terminated_by),
				termination_reason: row.termination_reason == null ? null : String(row.termination_reason)
			})),
			corrections: correctionReader.getRowObjects().map((row) => ({
				correction_id: Number(row.correction_id),
				correction_kind: String(row.correction_kind),
				status: String(row.status),
				parent_correction_id: row.parent_correction_id == null ? null : Number(row.parent_correction_id),
				child_run_id: row.child_run_id == null ? null : String(row.child_run_id),
				repair_source_dump_id: row.repair_source_dump_id == null ? null : String(row.repair_source_dump_id)
			})),
			steps: stepReader.getRowObjects().map((row) => ({
				run_id: String(row.run_id),
				stmt_hash: String(row.stmt_hash),
				evidence_hash: String(row.evidence_hash),
				step_kind: String(row.step_kind),
				fixture: row.fixture == null ? null : String(row.fixture)
			}))
		};
	} finally {
		con.disconnectSync?.();
		instance.closeSync();
	}
}

function repairPageAuditExpression(label) {
	return `(() => {
		const failures = [];
		const text = (el) => (el?.textContent ?? '').replace(/\\s+/g, ' ').trim();
		const form = document.querySelector('form.probe-slot-editor');
		const probeFacts = document.querySelector('[aria-label^="probe repair facts"]');
		const head = document.querySelector('.repair-head');
		const panel = document.querySelector('[aria-label="rerun repair cohort"]');
		const pageText = text(document.body);
		const overflow = document.documentElement.scrollWidth - window.innerWidth;
		if (overflow > 1) failures.push('page has horizontal overflow');
		if (!pageText.includes('repair backlog')) failures.push('repair title missing');
		if (!probeFacts) failures.push('probe facts missing');
		if (!pageText.includes('probe slot review')) failures.push('probe slot editor missing');
		if (!pageText.includes('object_role_probe')) failures.push('object slot missing');
		if (!pageText.includes('relation_axis_probe')) failures.push('relation slot missing');
		if (!pageText.includes('scope_probe')) failures.push('scope slot missing');
		for (const [name, el] of [['head', head], ['rerun panel', panel], ['probe slot form', form]]) {
			if (!el) {
				failures.push(name + ' missing');
				continue;
			}
			const rect = el.getBoundingClientRect();
			if (rect.width < 1 || rect.height < 1) failures.push(name + ' invisible');
			if (rect.left < -1 || rect.right > window.innerWidth + 1) failures.push(name + ' escapes viewport');
		}
		return { viewport: ${JSON.stringify(label)}, failures };
	})()`;
}

function submitProbeSlotReviewExpression() {
	return `(() => {
		const form = document.querySelector('form.probe-slot-editor');
		if (!form) throw new Error('probe slot form missing');
		const byValue = (value) => form.querySelector('input[name="probe_slot"][value="' + value + '"]');
		for (const slot of ['object_role_probe', 'relation_axis_probe', 'scope_probe']) {
			const input = byValue(slot);
			if (!input) throw new Error(slot + ' checkbox missing');
		}
		for (const slot of ['object_role_probe', 'scope_probe']) {
			const input = byValue(slot);
			if (!input.checked) input.click();
		}
		const relation = byValue('relation_axis_probe');
		if (relation.checked) relation.click();
		const textarea = form.querySelector('textarea[name="probe_slot_note"]');
		if (!textarea) throw new Error('probe slot note textarea missing');
		textarea.value = ${JSON.stringify(reviewNote)};
		textarea.dispatchEvent(new Event('input', { bubbles: true }));
		const button = form.querySelector('button[type="submit"]');
		if (!button) throw new Error('submit button missing');
		button.click();
		return {
			selected: Array.from(form.querySelectorAll('input[name="probe_slot"]:checked')).map((input) => input.value),
			note: textarea.value
		};
	})()`;
}

function reviewedStateExpression() {
	return `(() => {
		const form = document.querySelector('form.probe-slot-editor');
		const byValue = (value) => form?.querySelector('input[name="probe_slot"][value="' + value + '"]');
		const latest = Array.from(document.querySelectorAll('.probe-slot-editor-actions .muted')).map((el) => el.textContent ?? '').join(' ');
		const textarea = form?.querySelector('textarea[name="probe_slot_note"]');
		const fieldset = form?.querySelector('fieldset');
		const button = form?.querySelector('button[type="submit"]');
		return {
			pageText: (document.body.textContent ?? '').replace(/\\s+/g, ' ').trim(),
			latest: latest.replace(/\\s+/g, ' ').trim(),
			note: textarea?.value ?? null,
			buttonText: button?.textContent?.replace(/\\s+/g, ' ').trim() ?? null,
			buttonDisabled: Boolean(button?.disabled),
			fieldsetDisabled: Boolean(fieldset?.disabled),
			objectChecked: Boolean(byValue('object_role_probe')?.checked),
			relationChecked: Boolean(byValue('relation_axis_probe')?.checked),
			scopeChecked: Boolean(byValue('scope_probe')?.checked),
			failures: []
		};
	})()`;
}

function clickRepairPreflightExpression() {
	return `(() => {
		const button = document.querySelector('button[aria-label="preview repair rerun cost"]');
		if (!button) throw new Error('repair rerun cost preview button missing');
		if (button.disabled) throw new Error('repair rerun cost preview button disabled');
		button.click();
		return { buttonText: button.textContent?.replace(/\\s+/g, ' ').trim() ?? null };
	})()`;
}

function repairPreflightExpression(label) {
	return `(() => {
		const failures = [];
		const pageText = (document.body.textContent ?? '').replace(/\\s+/g, ' ').trim();
		const panel = document.querySelector('[aria-label="rerun repair cohort"]');
		const tableWrap = document.querySelector('.cost-table-wrap');
		const table = document.querySelector('.cost-table-wrap table');
		const rows = Array.from(document.querySelectorAll('.cost-table-wrap tbody tr'));
		const runButtons = rows.flatMap((row) => Array.from(row.querySelectorAll('button.spend')));
		const overflow = document.documentElement.scrollWidth - window.innerWidth;
		if (overflow > 1) failures.push('preflight page has horizontal overflow');
		if (!panel) failures.push('rerun panel missing after preflight');
		if (!table) failures.push('cost table missing after preflight');
		if (rows.length === 0) failures.push('cost estimate rows missing');
		if (runButtons.length === 0) failures.push('spend launch buttons missing');
		if (!pageText.includes('1 candidates')) failures.push('candidate count missing from preflight');
		if (!pageText.includes('1 candidate statements')) failures.push('statement count missing from preflight');
		if (!pageText.includes('1 evidences')) failures.push('evidence count missing from preflight');
		if (!pageText.includes('Evidence denominator validated')) failures.push('evidence denominator validation copy missing');
		if (!pageText.includes('probe-slot reviewed candidate')) failures.push('probe slot review metadata missing from preflight');
		if (!pageText.includes('object_role_probe 1')) failures.push('object slot count missing from preflight');
		if (!pageText.includes('scope_probe 1')) failures.push('scope slot count missing from preflight');
		if (!pageText.includes('This launch will run only the reviewed probes')) failures.push('probe-only worker boundary copy missing');
		if (!pageText.includes('deterministic aggregate re-adjudication')) failures.push('probe-only merge copy missing');
		if (!pageText.includes('not fresh aggregate LLM calls')) failures.push('probe-only aggregate boundary copy missing');
		if (!pageText.includes('Spend cap uses the selected')) failures.push('spend cap copy missing');
		if (!pageText.includes('revise selection or note')) failures.push('revision affordance missing');
		for (const [name, el] of [['panel', panel], ['cost table wrapper', tableWrap]]) {
			if (!el) continue;
			const rect = el.getBoundingClientRect();
			if (rect.width < 1 || rect.height < 1) failures.push(name + ' invisible');
			if (rect.left < -1 || rect.right > window.innerWidth + 1) failures.push(name + ' escapes viewport');
		}
		if (${JSON.stringify(label)}.includes('mobile') && tableWrap && table && table.getBoundingClientRect().width <= tableWrap.getBoundingClientRect().width) {
			failures.push('mobile cost table is not preserved as a scrollable cost plane');
		}
		return {
			viewport: ${JSON.stringify(label)},
			costRowCount: rows.length,
			runButtonCount: runButtons.length,
			failures
		};
	})()`;
}

function activeTombstoneAuditExpression(label) {
	return `(() => {
		const failures = [];
		const pageText = (document.body.textContent ?? '').replace(/\\s+/g, ' ').trim();
		const repairHeadText = (document.querySelector('.repair-head')?.textContent ?? '').replace(/\\s+/g, ' ').trim();
		const tableText = Array.from(document.querySelectorAll('tbody tr')).map((row) => row.textContent ?? '').join(' ').replace(/\\s+/g, ' ').trim();
		const panel = document.querySelector('[aria-label="active repair reruns locking candidates"]');
		const button = document.querySelector('button[aria-label="mark repair child ${ACTIVE_CHILD.slice(0, 8)} canceled"]');
		const overflow = document.documentElement.scrollWidth - window.innerWidth;
		if (overflow > 1) failures.push('active tombstone page has horizontal overflow');
		if (!panel) failures.push('active rerun panel missing');
		if (!button) failures.push('mark child canceled button missing');
		if (button?.disabled) failures.push('mark child canceled button disabled');
		if (!pageText.includes('active reruns locking candidates')) failures.push('active rerun heading missing');
		if (!pageText.includes('1 locked candidate')) failures.push('locked candidate count missing');
		if (!pageText.includes(${JSON.stringify(TOMBSTONE_SOURCE_DUMP)})) failures.push('source dump id missing');
		if (!repairHeadText.includes('1 open repair candidate')) failures.push('available candidate count before tombstone is wrong');
		if (tableText.includes('#122')) failures.push('locked candidate leaked into spendable queue before tombstone');
		if (panel) {
			const rect = panel.getBoundingClientRect();
			if (rect.width < 1 || rect.height < 1) failures.push('active rerun panel invisible');
			if (rect.left < -1 || rect.right > window.innerWidth + 1) failures.push('active rerun panel escapes viewport');
		}
		return { viewport: ${JSON.stringify(label)}, failures };
	})()`;
}

function clickTombstoneExpression() {
	return `(() => {
		const button = document.querySelector('button[aria-label="mark repair child ${ACTIVE_CHILD.slice(0, 8)} canceled"]');
		if (!button) throw new Error('mark child canceled button missing');
		if (button.disabled) throw new Error('mark child canceled button disabled');
		button.click();
		return { buttonText: button.textContent?.replace(/\\s+/g, ' ').trim() ?? null };
	})()`;
}

function tombstonedAuditExpression(label) {
	return `(() => {
		const failures = [];
		const pageText = (document.body.textContent ?? '').replace(/\\s+/g, ' ').trim();
		const repairHeadText = (document.querySelector('.repair-head')?.textContent ?? '').replace(/\\s+/g, ' ').trim();
		const tableText = Array.from(document.querySelectorAll('tbody tr')).map((row) => row.textContent ?? '').join(' ').replace(/\\s+/g, ' ').trim();
		const panel = document.querySelector('[aria-label="active repair reruns locking candidates"]');
		const overflow = document.documentElement.scrollWidth - window.innerWidth;
		if (overflow > 1) failures.push('tombstoned page has horizontal overflow');
		if (panel) failures.push('active rerun panel remained after tombstone');
		if (!repairHeadText.includes('2 open repair candidates')) failures.push('available candidate count after tombstone is wrong');
		if (!tableText.includes('#122')) failures.push('tombstoned candidate did not return to spendable queue');
		if (!tableText.includes('Browser reviewer should be able to release stale active child lock')) failures.push('tombstoned candidate review text missing from spendable queue');
		if (pageText.includes('mark child canceled')) failures.push('tombstone action remained visible after success');
		return { viewport: ${JSON.stringify(label)}, failures };
	})()`;
}

function selectOnlyRepairCandidateExpression(correctionId) {
	return `(() => {
		const failures = [];
		const desired = String(${JSON.stringify(correctionId)});
		const inputs = Array.from(document.querySelectorAll('input[aria-label^="select repair candidate "]'));
		if (inputs.length === 0) throw new Error('repair candidate selection inputs missing');
		for (const input of inputs) {
			const match = (input.getAttribute('aria-label') ?? '').match(/select repair candidate (\\d+)/);
			const id = match?.[1] ?? '';
			const shouldCheck = id === desired;
			if (input.disabled) failures.push('candidate #' + id + ' selection disabled');
			if (input.checked !== shouldCheck) input.click();
		}
		const selected = inputs
			.filter((input) => input.checked)
			.map((input) => (input.getAttribute('aria-label') ?? '').match(/(\\d+)/)?.[1] ?? '');
		return { selected, failures };
	})()`;
}

function clickRepairSpendExpression(modelId = LAUNCH_MODEL) {
	return `(() => {
		const buttons = Array.from(document.querySelectorAll('button.spend'));
		const button = buttons.find((btn) => (btn.textContent ?? '').includes(${JSON.stringify(modelId)})) ?? buttons[0];
		if (!button) throw new Error('repair rerun spend button missing');
		if (button.disabled) throw new Error('repair rerun spend button disabled');
		const text = button.textContent?.replace(/\\s+/g, ' ').trim() ?? null;
		button.click();
		return { buttonText: text };
	})()`;
}

function repairScoringExpression(label, modelId = LAUNCH_MODEL) {
	return `(() => {
		const failures = [];
		const pageText = (document.body.textContent ?? '').replace(/\\s+/g, ' ').trim();
		const panel = document.querySelector('[aria-label="rerun repair cohort"]');
		const overflow = document.documentElement.scrollWidth - window.innerWidth;
		if (overflow > 1) failures.push('repair scoring page has horizontal overflow');
		if (!panel) failures.push('rerun panel missing during scoring');
		if (!pageText.includes('scoring')) failures.push('scoring verb missing after spend launch');
		if (!pageText.includes(${JSON.stringify(modelId)})) failures.push('model missing during scoring');
		if (!pageText.includes('spent')) failures.push('spend progress missing during scoring');
		if (!pageText.includes('child')) failures.push('child run link missing during scoring');
		if (!pageText.includes('cancel')) failures.push('cancel affordance missing during scoring');
		if (panel) {
			const rect = panel.getBoundingClientRect();
			if (rect.width < 1 || rect.height < 1) failures.push('rerun panel invisible during scoring');
			if (rect.left < -1 || rect.right > window.innerWidth + 1) failures.push('rerun panel escapes viewport during scoring');
		}
		return { viewport: ${JSON.stringify(label)}, failures };
	})()`;
}

function repairCancelPreflightExpression(label) {
	return `(() => {
		const failures = [];
		const pageText = (document.body.textContent ?? '').replace(/\\s+/g, ' ').trim();
		const panel = document.querySelector('[aria-label="rerun repair cohort"]');
		const rows = Array.from(document.querySelectorAll('.cost-table-wrap tbody tr'));
		const cancelButton = rows.flatMap((row) => Array.from(row.querySelectorAll('button.spend'))).find((btn) => (btn.textContent ?? '').includes(${JSON.stringify(CANCEL_MODEL)}));
		const overflow = document.documentElement.scrollWidth - window.innerWidth;
		if (overflow > 1) failures.push('cancel preflight page has horizontal overflow');
		if (!panel) failures.push('rerun panel missing before cancel launch');
		if (rows.length === 0) failures.push('cost estimate rows missing before cancel launch');
		if (!cancelButton) failures.push('cancel smoke spend button missing');
		if (!pageText.includes('1 candidates')) failures.push('candidate count missing before cancel launch');
		if (!pageText.includes('Spend cap uses the selected')) failures.push('spend cap copy missing before cancel launch');
		return { viewport: ${JSON.stringify(label)}, costRowCount: rows.length, failures };
	})()`;
}

function clickRepairCancelExpression() {
	return `(() => {
		const button = Array.from(document.querySelectorAll('button.inline-cancel')).find((btn) => (btn.textContent ?? '').includes('cancel'));
		if (!button) throw new Error('repair rerun cancel button missing');
		if (button.disabled) throw new Error('repair rerun cancel button disabled');
		const text = button.textContent?.replace(/\\s+/g, ' ').trim() ?? null;
		button.click();
		return { buttonText: text };
	})()`;
}

function repairCancelRequestedExpression(label) {
	return `(() => {
		const failures = [];
		const pageText = (document.body.textContent ?? '').replace(/\\s+/g, ' ').trim();
		const overflow = document.documentElement.scrollWidth - window.innerWidth;
		if (overflow > 1) failures.push('repair cancel requested page has horizontal overflow');
		if (!pageText.includes('cancel requested')) failures.push('cancel request acknowledgement missing');
		if (!pageText.includes('no repair candidates were marked rerun-complete')) failures.push('no-completion-marker copy missing');
		if (!pageText.includes('child')) failures.push('canceled child link missing from cancel acknowledgement');
		return { viewport: ${JSON.stringify(label)}, failures };
	})()`;
}

function repairCancelRehydratedExpression(label) {
	return `(() => {
		const failures = [];
		const pageText = (document.body.textContent ?? '').replace(/\\s+/g, ' ').trim();
		const historyText = (document.querySelector('[aria-label="repair rerun comparisons"]')?.textContent ?? '').replace(/\\s+/g, ' ').trim();
		const repairHeadText = (document.querySelector('.repair-head')?.textContent ?? '').replace(/\\s+/g, ' ').trim();
		const tableText = Array.from(document.querySelectorAll('tbody tr')).map((row) => row.textContent ?? '').join(' ').replace(/\\s+/g, ' ').trim();
		const overflow = document.documentElement.scrollWidth - window.innerWidth;
		if (overflow > 1) failures.push('repair cancel rehydrated page has horizontal overflow');
		if (!pageText.includes('cancel requested')) failures.push('cancel acknowledgement disappeared after rehydration');
		if (!historyText.includes(${JSON.stringify(CANCEL_MODEL)})) failures.push('canceled child model missing from rerun history after rehydration');
		if (!historyText.includes('canceled')) failures.push('canceled terminal state missing from rerun history after rehydration');
		if (!repairHeadText.includes('1 open repair candidate')) failures.push('open repair count after cancel is wrong');
		if (!tableText.includes('#122')) failures.push('canceled candidate did not remain spendable');
		if (tableText.includes('#121')) failures.push('previously completed candidate returned after cancel');
		if (pageText.includes('active reruns locking candidates')) failures.push('active lock panel remained after cancel');
		return { viewport: ${JSON.stringify(label)}, failures };
	})()`;
}

function completedRecoveryAuditExpression(label) {
	return `(() => {
		const failures = [];
		const pageText = (document.body.textContent ?? '').replace(/\\s+/g, ' ').trim();
		const panel = document.querySelector('[aria-label="completed repair reruns awaiting markers"]');
		const finalizeButton = Array.from(panel?.querySelectorAll('button') ?? []).find((btn) => (btn.textContent ?? '').includes('finalize markers'));
		const releaseButton = Array.from(panel?.querySelectorAll('button') ?? []).find((btn) => (btn.textContent ?? '').includes('release uncovered'));
		const overflow = document.documentElement.scrollWidth - window.innerWidth;
		if (overflow > 1) failures.push('completed recovery page has horizontal overflow');
		if (!panel) failures.push('completed recovery panel missing');
		if (!pageText.includes('completed reruns awaiting markers')) failures.push('completed recovery heading missing');
		if (!pageText.includes('Finalizing writes only append-only completion markers')) failures.push('append-only marker copy missing');
		if (!pageText.includes(${JSON.stringify(RECOVERY_SOURCE_DUMP)})) failures.push('recovery source dump missing');
		if (!pageText.includes('1 finalizable marker')) failures.push('finalizable marker count missing');
		if (!pageText.includes('1 missing total')) failures.push('missing marker count missing');
		if (!finalizeButton) failures.push('finalize markers button missing');
		if (finalizeButton?.disabled) failures.push('finalize markers button disabled');
		if (!releaseButton) failures.push('release uncovered button missing');
		if (releaseButton && !releaseButton.disabled) failures.push('release uncovered should be disabled for fully covered recovery');
		if (panel) {
			const rect = panel.getBoundingClientRect();
			if (rect.width < 1 || rect.height < 1) failures.push('completed recovery panel invisible');
			if (rect.left < -1 || rect.right > window.innerWidth + 1) failures.push('completed recovery panel escapes viewport');
		}
		return { viewport: ${JSON.stringify(label)}, failures };
	})()`;
}

function clickFinalizeRecoveryExpression() {
	return `(() => {
		const panel = document.querySelector('[aria-label="completed repair reruns awaiting markers"]');
		if (!panel) throw new Error('completed recovery panel missing');
		const button = Array.from(panel.querySelectorAll('button')).find((btn) => (btn.textContent ?? '').includes('finalize markers'));
		if (!button) throw new Error('finalize markers button missing');
		if (button.disabled) throw new Error('finalize markers button disabled');
		const text = button.textContent?.replace(/\\s+/g, ' ').trim() ?? null;
		button.click();
		return { buttonText: text };
	})()`;
}

function recoveryFinalizedExpression(label) {
	return `(() => {
		const failures = [];
		const pageText = (document.body.textContent ?? '').replace(/\\s+/g, ' ').trim();
		const repairHeadText = (document.querySelector('.repair-head')?.textContent ?? '').replace(/\\s+/g, ' ').trim();
		const tableText = Array.from(document.querySelectorAll('tbody tr')).map((row) => row.textContent ?? '').join(' ').replace(/\\s+/g, ' ').trim();
		const overflow = document.documentElement.scrollWidth - window.innerWidth;
		if (overflow > 1) failures.push('recovery finalized page has horizontal overflow');
		if (document.querySelector('[aria-label="completed repair reruns awaiting markers"]')) failures.push('completed recovery panel remained after finalization');
		if (!repairHeadText.includes('1 open repair candidate')) failures.push('open repair count after recovery finalization is wrong');
		if (!tableText.includes('#122')) failures.push('spendable canceled candidate missing after recovery finalization');
		if (tableText.includes('#124')) failures.push('finalized recovery candidate appeared in spendable queue');
		if (!pageText.includes(${JSON.stringify(RECOVERY_CHILD.slice(0, 8))})) failures.push('recovered child missing from rerun history');
		if (!pageText.includes('repair candidates')) failures.push('repair rerun coverage summary missing after recovery finalization');
		return { viewport: ${JSON.stringify(label)}, failures };
	})()`;
}

function repairLaunchDoneExpression(label) {
	return `(() => {
		const failures = [];
		const pageText = (document.body.textContent ?? '').replace(/\\s+/g, ' ').trim();
		const repairHeadText = (document.querySelector('.repair-head')?.textContent ?? '').replace(/\\s+/g, ' ').trim();
		const tableText = Array.from(document.querySelectorAll('tbody tr')).map((row) => row.textContent ?? '').join(' ').replace(/\\s+/g, ' ').trim();
		const overflow = document.documentElement.scrollWidth - window.innerWidth;
		if (overflow > 1) failures.push('repair launch done page has horizontal overflow');
		if (!pageText.includes('child run')) failures.push('done child run text missing');
		if (!pageText.includes(${JSON.stringify(LAUNCH_MODEL)})) failures.push('done model text missing');
		if (!pageText.includes('1 repair candidates now have append-only rerun markers')) failures.push('append-only completion marker count missing');
		if (!repairHeadText.includes('1 open repair candidate')) failures.push('open repair count after launch is wrong');
		if (tableText.includes('#121')) failures.push('consumed launched candidate remained in spendable queue');
		if (!tableText.includes('#122')) failures.push('unlaunched candidate missing from spendable queue after launch');
		if (pageText.includes('active reruns locking candidates')) failures.push('active lock panel remained after launched child completed');
		return { viewport: ${JSON.stringify(label)}, failures };
	})()`;
}

async function repairRerunMetaFiles(dir) {
	const root = join(dir, 'repair_reruns');
	const names = await readdir(root).catch(() => []);
	const metaNames = names.filter((name) => name.endsWith('.meta.json')).sort();
	const out = [];
	for (const name of metaNames) {
		out.push(JSON.parse(await readFile(join(root, name), 'utf8')));
	}
	return out;
}

async function writeTombstoneMeta(dir) {
	const root = join(dir, 'repair_reruns');
	await mkdir(root, { recursive: true });
	const corpusPath = join(root, `${TOMBSTONE_SOURCE_DUMP}.json`);
	await writeFile(corpusPath, `[\n${JSON.stringify(tombstoneStatementRawJson)}\n]\n`);
	await writeFile(join(root, `${TOMBSTONE_SOURCE_DUMP}.meta.json`), JSON.stringify({
		parent_run_id: RUN,
		architecture: 'decomposed',
		source_dump_id: TOMBSTONE_SOURCE_DUMP,
		path: corpusPath,
		export_scope: 'candidate_statements',
		n_candidates: 1,
		n_statements: 1,
		n_evidences: 1,
		correction_ids: [122],
		requested_correction_ids: [122],
		dropped_correction_ids: [],
		n_selected_evidence_candidates: 1,
		n_statement_scope_candidates: 0,
		n_scope_expansion_evidences: 0,
		n_collateral_evidences: 0,
		n_probe_slot_reviewed_candidates: 0,
		probe_slot_reviews: [],
		probe_slot_counts: {},
		max_correction_ids: 500,
		truncated: false
	}, null, 2));
}

async function writeRecoveryMeta(dir) {
	const root = join(dir, 'repair_reruns');
	await mkdir(root, { recursive: true });
	const corpusPath = join(root, `${RECOVERY_SOURCE_DUMP}.json`);
	await writeFile(corpusPath, `[\n${JSON.stringify(recoveryStatementRawJson)}\n]\n`);
	await writeFile(join(root, `${RECOVERY_SOURCE_DUMP}.meta.json`), JSON.stringify({
		parent_run_id: RUN,
		architecture: 'decomposed',
		source_dump_id: RECOVERY_SOURCE_DUMP,
		path: corpusPath,
		export_scope: 'candidate_statements',
		n_candidates: 1,
		n_statements: 1,
		n_evidences: 1,
		correction_ids: [124],
		requested_correction_ids: [124],
		dropped_correction_ids: [],
		n_selected_evidence_candidates: 1,
		n_statement_scope_candidates: 0,
		n_scope_expansion_evidences: 0,
		n_collateral_evidences: 0,
		n_probe_slot_reviewed_candidates: 0,
		probe_slot_reviews: [],
		probe_slot_counts: {},
		max_correction_ids: 500,
		truncated: false
	}, null, 2));
}

async function writeFakeWorkerPython(dir) {
	const fakePath = join(dir, 'fake-python-worker.mjs');
	await writeFile(fakePath, `#!/usr/bin/env node
import { createRequire } from 'node:module';

const requireBase = process.env.INDRA_FAKE_WORKER_REQUIRE_BASE;
if (!requireBase) throw new Error('INDRA_FAKE_WORKER_REQUIRE_BASE missing');
const require = createRequire(requireBase);
const { DuckDBInstance } = await import(require.resolve('@duckdb/node-api'));

const args = process.argv.slice(2);
const workerIndex = args.indexOf('indra_belief.worker');
const verb = workerIndex >= 0 ? args[workerIndex + 1] : args[0];

function emit(event) {
	process.stdout.write(JSON.stringify(event) + '\\n');
}

function option(name) {
	const idx = args.indexOf(name);
	return idx >= 0 ? args[idx + 1] : null;
}

function sleep(ms) {
	return new Promise((resolve) => setTimeout(resolve, ms));
}

function sourceDumpToStepPrefix(sourceDumpId) {
	return String(sourceDumpId ?? 'repair').replace(/[^a-z0-9_]/gi, '_').slice(0, 40);
}

async function withConnection(dbFile, fn) {
	const instance = await DuckDBInstance.create(dbFile);
	const con = await instance.connect();
	try {
		return await fn(con);
	} finally {
		con.disconnectSync?.();
		instance.closeSync();
	}
}

async function estimateCost() {
	const arch = option('--arch') ?? 'decomposed';
	const probeOnly = args.includes('--probe-only');
	const probeStepFilter = (option('--probe-step-filter') ?? '').split(',').filter(Boolean);
	emit({ event: 'started', verb: 'estimate-cost', fixture: 'repair-actionability-browser' });
	emit({
		event: 'done',
		scoring_mode: probeOnly ? 'probe_only' : 'aggregate',
		probe_step_filter: probeStepFilter,
		estimates: [
			{
				model_id: ${JSON.stringify(LAUNCH_MODEL)},
				architecture: arch,
				scoring_mode: probeOnly ? 'probe_only' : 'aggregate',
				probe_step_filter: probeStepFilter,
				cost_usd: 0.002,
				n_stmts: 1,
				n_evidences_est: 1,
				n_llm_calls_est: probeOnly ? probeStepFilter.length : 1
			},
			{
				model_id: ${JSON.stringify(CANCEL_MODEL)},
				architecture: arch,
				scoring_mode: probeOnly ? 'probe_only' : 'aggregate',
				probe_step_filter: probeStepFilter,
				cost_usd: 0.003,
				n_stmts: 1,
				n_evidences_est: 1,
				n_llm_calls_est: probeOnly ? probeStepFilter.length : 1
			}
		]
	});
}

async function score() {
	const dbFile = option('--db');
	const runId = option('--run-id');
	const parentRunId = option('--parent-run-id');
	const sourceDumpId = option('--source-dump-id');
	const model = option('--model') ?? ${JSON.stringify(LAUNCH_MODEL)};
	const scorerVersion = option('--scorer-version') ?? 'repair-actionability-browser';
	const arch = option('--arch') ?? 'decomposed';
	const probeOnly = args.includes('--probe-only');
	const probeStepFilter = (option('--probe-step-filter') ?? '').split(',').filter(Boolean);
	if (!dbFile || !runId || !parentRunId || !sourceDumpId) {
		throw new Error('fake score worker missing db/run/parent/source arguments');
	}

	emit({
		event: 'started',
		verb: 'score',
		run_id: runId,
		model,
		scorer_version: scorerVersion,
		architecture: arch,
		parent_run_id: parentRunId
	});
	await withConnection(dbFile, async (con) => {
		await con.run(
			\`INSERT INTO score_run
			 (run_id, scorer_version, indra_version, architecture, status, parent_run_id,
			  started_at, finished_at, n_stmts, model_id_default, cost_estimate_usd, cost_actual_usd)
			 VALUES (?, ?, 'test-indra', ?, 'running', ?, CURRENT_TIMESTAMP, NULL, 1, ?, 0.002, NULL)\`,
			[runId, scorerVersion, arch, parentRunId, model]
		);
	});
	emit({ event: 'loaded', n_statements: 1, n_evidences: 1, architecture: arch });
	if (model === ${JSON.stringify(CANCEL_MODEL)}) {
		await sleep(60000);
		return;
	}
	await sleep(800);
	await withConnection(dbFile, async (con) => {
		const prefix = sourceDumpToStepPrefix(sourceDumpId);
		if (probeOnly) {
			for (const step of probeStepFilter) {
				await con.run(
					\`INSERT INTO scorer_step
					 (step_hash, run_id, stmt_hash, evidence_hash, architecture, model_id, step_kind,
					  output_json, error, latency_ms, prompt_tokens, out_tokens, started_at)
					 VALUES (?, ?, 'stmt_probe_browser', 'ev_probe_browser', ?, ?, ?,
					         ?::JSON, NULL, 25, 5, 2, CURRENT_TIMESTAMP)\`,
					[
						\`\${prefix}_\${step}\`,
						runId,
						arch,
						model,
						step,
						JSON.stringify({
							kind: step.replace('_probe', ''),
							answer: step === 'scope_probe' ? 'asserted' : 'present_as_object',
							source: 'llm',
							confidence: 'medium',
							fixture: 'repair_actionability_probe_only'
						})
					]
				);
			}
			await con.run(
				\`INSERT INTO scorer_step
				 (step_hash, run_id, stmt_hash, evidence_hash, architecture, model_id, step_kind,
				  output_json, error, latency_ms, prompt_tokens, out_tokens, started_at)
				 VALUES (?, ?, 'stmt_probe_browser', 'ev_probe_browser', ?, ?, 'aggregate',
				         ?::JSON, NULL, 0, NULL, NULL, CURRENT_TIMESTAMP)\`,
				[
					\`\${prefix}_probe_merge_aggregate\`,
					runId,
					arch,
					model,
					JSON.stringify({
						score: 0.64,
						verdict: 'correct',
						confidence: 'low',
						tier: 'decomposed_probe_repair_merge',
						fixture: 'repair_actionability_probe_merge',
						repair_merge: {
							aggregate_llm_call: false,
							probe_step_filter: probeStepFilter
						}
					})
				]
			);
		} else {
			await con.run(
				\`INSERT INTO scorer_step
				 (step_hash, run_id, stmt_hash, evidence_hash, architecture, model_id, step_kind,
				  output_json, error, latency_ms, prompt_tokens, out_tokens, started_at)
				 VALUES (?, ?, 'stmt_probe_browser', 'ev_probe_browser', ?, ?, 'aggregate',
				         ?::JSON, NULL, 25, 5, 2, CURRENT_TIMESTAMP)\`,
				[
					\`\${prefix}_aggregate\`,
					runId,
					arch,
					model,
					JSON.stringify({
						score: 0.78,
						verdict: 'correct',
						confidence: 'medium',
						fixture: 'repair_actionability_launch'
					})
				]
			);
		}
		await con.run(
			\`UPDATE score_run
			    SET status='succeeded',
			        finished_at=CURRENT_TIMESTAMP,
			        cost_actual_usd=0.002
			  WHERE run_id=?\`,
			[runId]
		);
	});
	emit({
		event: 'progress',
		n_evidences_done: 1,
		latest_stmt_hash: 'stmt_probe_browser',
		architecture: arch,
		cost_so_far_usd: 0.002,
		cost_cap_usd: 0.01,
		cost_increment_usd: 0.002
	});
	await sleep(150);
	emit({
		event: 'done',
		run_id: runId,
		architecture: arch,
		parent_run_id: parentRunId,
		n_statements: 1,
		n_evidences_done: 1,
		duration_s: 0.9,
		cost_cap_usd: 0.01
	});
}

if (verb === 'estimate-cost') {
	await estimateCost();
} else if (verb === 'score') {
	await score();
} else {
	throw new Error(\`unsupported fake worker verb: \${verb}\`);
}
`);
	await chmod(fakePath, 0o755);
	return fakePath;
}

async function main() {
	if (!existsSync(chromePath)) {
		throw new Error(`Chrome not found at ${chromePath}`);
	}

	const dir = await mkdtemp(join(tmpdir(), 'indra-repair-actionability-browser-'));
	const dbFile = join(dir, 'repair-actionability-browser.duckdb');
	process.env.VIEWER_DUCKDB_PATH = dbFile;
	process.env.INDRA_VIEWER_SUPPRESS_WRITER_GUARD_LOG = '1';

	let vite = null;
	let chrome = null;
	let client = null;
	let primaryError = null;

	try {
		await mkdir(artifactRoot, { recursive: true });
		await createFixtureDb(dbFile);
		await writeTombstoneMeta(dir);
		await writeRecoveryMeta(dir);
		process.env.PYTHON_BIN = await writeFakeWorkerPython(dir);
		process.env.INDRA_FAKE_WORKER_REQUIRE_BASE = join(process.cwd(), 'package.json');
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

		const summary = { run_id: RUN, checks: [], screenshots: [] };
		await client.send('Emulation.setDeviceMetricsOverride', {
			width: 1280,
			height: 900,
			deviceScaleFactor: 1,
			mobile: false
		});
		const load = client.waitEvent('Page.loadEventFired', 20000).catch(() => null);
		await client.send('Page.navigate', { url: `${baseUrl}/runs/${RUN}/repairs` });
		await load;
		await waitFor(client, `document.querySelector('form.probe-slot-editor button[type="submit"]:not([disabled])')`, 'desktop hydrated repair action form');
		await evaluate(client, `new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve)))`);
		summary.checks.push(await evaluate(client, repairPageAuditExpression('desktop-before-action')));
		const submitResult = await evaluate(client, submitProbeSlotReviewExpression());
		assert.deepEqual(submitResult.selected, ['object_role_probe', 'scope_probe']);
		assert.equal(submitResult.note, reviewNote);
		try {
			await waitFor(
				client,
				`(() => {
					const latest = Array.from(document.querySelectorAll('.probe-slot-editor-actions .muted')).map((el) => el.textContent ?? '').join(' ');
					const note = document.querySelector('form.probe-slot-editor textarea[name="probe_slot_note"]')?.value ?? '';
					return latest.includes('object_role_probe, scope_probe') && note === ${JSON.stringify(reviewNote)};
				})()`,
				'probe slot review saved through browser'
			);
		} catch (err) {
			const diagnostic = await evaluate(client, `(() => {
				const form = document.querySelector('form.probe-slot-editor');
				const latest = Array.from(document.querySelectorAll('.probe-slot-editor-actions .muted')).map((el) => el.textContent ?? '').join(' ');
				const error = document.querySelector('.probe-slot-editor .repair-error')?.textContent ?? null;
				const textarea = form?.querySelector('textarea[name="probe_slot_note"]');
				const button = form?.querySelector('button[type="submit"]');
				return {
					location: location.href,
					latest: latest.replace(/\\s+/g, ' ').trim(),
					error: error?.replace(/\\s+/g, ' ').trim() ?? null,
					note: textarea?.value ?? null,
					buttonText: button?.textContent?.replace(/\\s+/g, ' ').trim() ?? null,
					buttonDisabled: Boolean(button?.disabled),
					bodyText: (document.body.textContent ?? '').replace(/\\s+/g, ' ').trim().slice(0, 1200)
				};
			})()`);
			console.error(JSON.stringify(diagnostic, null, 2));
			throw err;
		}
		const reviewed = await evaluate(client, reviewedStateExpression());
		if (!reviewed.latest.includes('object_role_probe, scope_probe')) reviewed.failures.push('latest review slots not visible');
		if (reviewed.note !== reviewNote) reviewed.failures.push('review note not rehydrated into textarea');
		if (!reviewed.objectChecked || !reviewed.scopeChecked || reviewed.relationChecked) {
			reviewed.failures.push('reviewed checkbox state did not rehydrate selected slots only');
		}
		if (reviewed.fieldsetDisabled || reviewed.buttonDisabled || reviewed.buttonText !== 'record slot review') {
			reviewed.failures.push('review form did not leave working state after save');
		}
		summary.checks.push({ viewport: 'desktop-after-action', ...reviewed });
		summary.screenshots.push(await screenshot(client, 'repair-actionability-probe-review-desktop'));
		const desktopPreviewClick = await evaluate(client, clickRepairPreflightExpression());
		assert.equal(desktopPreviewClick.buttonText, 'preview rerun cost');
		await waitFor(client, `document.querySelector('.cost-table-wrap tbody tr')`, 'desktop repair rerun cost preflight');
		const desktopPreflight = await evaluate(client, repairPreflightExpression('desktop-rerun-preflight'));
		summary.checks.push(desktopPreflight);
		summary.screenshots.push(await screenshot(client, 'repair-actionability-rerun-preflight-desktop'));

		await client.send('Emulation.setDeviceMetricsOverride', {
			width: 390,
			height: 900,
			deviceScaleFactor: 1,
			mobile: true
		});
		const mobileLoad = client.waitEvent('Page.loadEventFired', 20000).catch(() => null);
		await client.send('Page.navigate', { url: `${baseUrl}/runs/${RUN}/repairs` });
		await mobileLoad;
		await waitFor(client, `document.querySelector('form.probe-slot-editor textarea[name="probe_slot_note"]')?.value === ${JSON.stringify(reviewNote)}`, 'mobile reviewed repair action form');
		await waitFor(client, `document.querySelector('form.probe-slot-editor button[type="submit"]:not([disabled])')`, 'mobile hydrated repair action form');
		summary.checks.push(await evaluate(client, repairPageAuditExpression('mobile-after-action')));
		summary.screenshots.push(await screenshot(client, 'repair-actionability-probe-review-mobile'));
		const mobilePreviewClick = await evaluate(client, clickRepairPreflightExpression());
		assert.equal(mobilePreviewClick.buttonText, 'preview rerun cost');
		await waitFor(client, `document.querySelector('.cost-table-wrap tbody tr')`, 'mobile repair rerun cost preflight');
		const mobilePreflight = await evaluate(client, repairPreflightExpression('mobile-rerun-preflight'));
		summary.checks.push(mobilePreflight);
		summary.screenshots.push(await screenshot(client, 'repair-actionability-rerun-preflight-mobile'));

		const rows = await probeSlotRows(dbFile);
		assert.deepEqual(rows.map((row) => ({
			parent_correction_id: row.parent_correction_id,
			correction_kind: row.correction_kind,
			status: row.status,
			reviewer: row.reviewer,
			note: row.note,
			selected_probe_slots: row.selected_probe_slots,
			reviewer_note: row.reviewer_note
		})), [{
			parent_correction_id: 121,
			correction_kind: 'probe_slot_review',
			status: 'recorded',
			reviewer: 'viewer',
			note: reviewNote,
			selected_probe_slots: 'object_role_probe,scope_probe',
			reviewer_note: reviewNote
		}]);
		summary.persisted_probe_slot_reviews = rows;
		const metas = await repairRerunMetaFiles(dir);
		assert.equal(metas.length, 4, 'fixture tombstone/recovery metas plus desktop and mobile preflight corpora exist');
		for (const meta of metas) {
			if (meta.source_dump_id === TOMBSTONE_SOURCE_DUMP || meta.source_dump_id === RECOVERY_SOURCE_DUMP) continue;
			assert.equal(meta.parent_run_id, RUN);
			assert.equal(meta.architecture, 'decomposed');
			assert.equal(meta.n_candidates, 1);
			assert.equal(meta.n_statements, 1);
			assert.equal(meta.n_evidences, 1);
			assert.equal(meta.n_raw_json_evidences, 1);
			assert.equal(meta.n_table_evidences, 1);
			assert.equal(meta.evidence_count_validated, true);
			assert.equal(meta.n_probe_slot_reviewed_candidates, 1);
			assert.deepEqual(meta.probe_slot_counts, { object_role_probe: 1, scope_probe: 1 });
			assert.deepEqual(meta.probe_slot_reviews[0].selected_slots, ['object_role_probe', 'scope_probe']);
			assert.equal(meta.scoring_mode, 'probe_only');
			assert.deepEqual(meta.probe_step_filter, ['object_role_probe', 'scope_probe']);
		}
		summary.repair_rerun_meta = metas.map((meta) => ({
			source_dump_id: meta.source_dump_id,
			n_candidates: meta.n_candidates,
			n_statements: meta.n_statements,
			n_evidences: meta.n_evidences,
			n_raw_json_evidences: meta.n_raw_json_evidences,
			n_table_evidences: meta.n_table_evidences,
			evidence_count_validated: meta.evidence_count_validated,
			n_probe_slot_reviewed_candidates: meta.n_probe_slot_reviewed_candidates,
			probe_slot_counts: meta.probe_slot_counts,
			scoring_mode: meta.scoring_mode,
			probe_step_filter: meta.probe_step_filter
		}));
		const repairPreflightMetas = metas.filter((meta) =>
			meta.source_dump_id !== TOMBSTONE_SOURCE_DUMP &&
			meta.source_dump_id !== RECOVERY_SOURCE_DUMP
		);
		assert.equal(repairPreflightMetas.length, 2, 'desktop and mobile preflight each export a fresh repair rerun corpus');

		await client.send('Emulation.setDeviceMetricsOverride', {
			width: 1280,
			height: 900,
			deviceScaleFactor: 1,
			mobile: false
		});
		const tombstoneLoad = client.waitEvent('Page.loadEventFired', 20000).catch(() => null);
		await client.send('Page.navigate', { url: `${baseUrl}/runs/${RUN}/repairs` });
		await tombstoneLoad;
		await waitFor(client, `document.querySelector('form.probe-slot-editor button[type="submit"]:not([disabled])')`, 'desktop hydrated repair page before tombstone');
		await waitFor(client, `document.querySelector('button[aria-label="mark repair child ${ACTIVE_CHILD.slice(0, 8)} canceled"]:not([disabled])')`, 'desktop active child tombstone button');
		const activeTombstoneAudit = await evaluate(client, activeTombstoneAuditExpression('desktop-active-tombstone-before'));
		summary.checks.push(activeTombstoneAudit);
		summary.screenshots.push(await screenshot(client, 'repair-actionability-active-tombstone-before'));
		const tombstoneClick = await evaluate(client, clickTombstoneExpression());
		assert.equal(tombstoneClick.buttonText, 'mark child canceled');
		await waitFor(
			client,
			`(() => {
				const text = (document.body.textContent ?? '').replace(/\\s+/g, ' ');
				const repairHeadText = (document.querySelector('.repair-head')?.textContent ?? '').replace(/\\s+/g, ' ');
				const tableText = Array.from(document.querySelectorAll('tbody tr')).map((row) => row.textContent ?? '').join(' ').replace(/\\s+/g, ' ');
				return !document.querySelector('[aria-label="active repair reruns locking candidates"]') && repairHeadText.includes('2 open repair candidates') && tableText.includes('#122');
			})()`,
			'active repair child tombstoned through browser'
		);
		const tombstonedAudit = await evaluate(client, tombstonedAuditExpression('desktop-active-tombstone-after'));
		summary.checks.push(tombstonedAudit);
		summary.screenshots.push(await screenshot(client, 'repair-actionability-active-tombstone-after'));
		const repairState = await repairStateRows(dbFile);
		const activeChildRun = repairState.runs.find((row) => row.run_id === ACTIVE_CHILD);
		assert.equal(activeChildRun?.status, 'canceled');
		assert.equal(activeChildRun?.terminated_by, 'user');
		assert.equal(activeChildRun?.termination_reason, 'stale_repair_child_released_from_repair_ui');
		const activeIntent = repairState.corrections.find((row) => row.correction_id === 123);
		assert.equal(activeIntent?.correction_kind, 'rerun_intent');
		assert.equal(activeIntent?.status, 'open');
		assert.equal(activeIntent?.parent_correction_id, 122);
		assert.equal(activeIntent?.child_run_id, ACTIVE_CHILD);
		assert.equal(activeIntent?.repair_source_dump_id, TOMBSTONE_SOURCE_DUMP);
		summary.tombstone_state = repairState;

		const selectedLaunch = await evaluate(client, selectOnlyRepairCandidateExpression(121));
		assert.deepEqual(selectedLaunch.selected, ['121']);
		assert.deepEqual(selectedLaunch.failures, []);
		const launchPreviewClick = await evaluate(client, clickRepairPreflightExpression());
		assert.equal(launchPreviewClick.buttonText, 'preview rerun cost');
		await waitFor(client, `document.querySelector('.cost-table-wrap tbody tr')`, 'desktop repair launch cost preflight');
		const launchPreflight = await evaluate(client, repairPreflightExpression('desktop-rerun-launch-preflight'));
		summary.checks.push(launchPreflight);
		summary.screenshots.push(await screenshot(client, 'repair-actionability-rerun-launch-preflight'));
		const spendClick = await evaluate(client, clickRepairSpendExpression());
		assert.match(spendClick.buttonText ?? '', /browser-repair-smoke/);
		await waitFor(
			client,
			`(() => {
				const text = (document.body.textContent ?? '').replace(/\\s+/g, ' ');
				return text.includes('scoring') && text.includes(${JSON.stringify(LAUNCH_MODEL)}) && text.includes('child');
			})()`,
			'repair rerun entered scoring state through browser'
		);
		const scoringAudit = await evaluate(client, repairScoringExpression('desktop-rerun-launch-scoring'));
		summary.checks.push(scoringAudit);
		summary.screenshots.push(await screenshot(client, 'repair-actionability-rerun-launch-scoring'));
		try {
			await waitFor(
				client,
				`(() => {
					const text = (document.body.textContent ?? '').replace(/\\s+/g, ' ');
					const repairHeadText = (document.querySelector('.repair-head')?.textContent ?? '').replace(/\\s+/g, ' ');
					const tableText = Array.from(document.querySelectorAll('tbody tr')).map((row) => row.textContent ?? '').join(' ').replace(/\\s+/g, ' ');
					return text.includes('append-only rerun markers') && repairHeadText.includes('1 open repair candidate') && !tableText.includes('#121') && tableText.includes('#122');
				})()`,
				'repair rerun completed and rehydrated through browser',
				15000
			);
		} catch (err) {
			const diagnostic = await evaluate(client, `(() => {
				const text = (document.body.textContent ?? '').replace(/\\s+/g, ' ').trim();
				const repairHeadText = (document.querySelector('.repair-head')?.textContent ?? '').replace(/\\s+/g, ' ').trim();
				const tableText = Array.from(document.querySelectorAll('tbody tr')).map((row) => row.textContent ?? '').join(' ').replace(/\\s+/g, ' ').trim();
				return {
					location: location.href,
					repairHeadText,
					tableText,
					rerunPanelText: (document.querySelector('[aria-label="rerun repair cohort"]')?.textContent ?? '').replace(/\\s+/g, ' ').trim(),
					bodyText: text.slice(0, 1800)
				};
			})()`);
			console.error(JSON.stringify({
				launch_completion_wait_failed: diagnostic,
				launch_state: await repairLaunchRows(dbFile)
			}, null, 2));
			throw err;
		}
		const launchDoneAudit = await evaluate(client, repairLaunchDoneExpression('desktop-rerun-launch-done'));
		summary.checks.push(launchDoneAudit);
		summary.screenshots.push(await screenshot(client, 'repair-actionability-rerun-launch-done'));
		const launchState = await repairLaunchRows(dbFile);
		const launchedRun = launchState.runs.find((row) => row.model_id_default === LAUNCH_MODEL);
		assert.ok(launchedRun, 'launched repair child run persisted');
		assert.match(launchedRun.run_id, /^[a-f0-9]{32}$/);
		assert.equal(launchedRun.parent_run_id, RUN);
		assert.equal(launchedRun.status, 'succeeded');
		assert.equal(launchedRun.architecture, 'decomposed');
		assert.equal(launchedRun.model_id_default, LAUNCH_MODEL);
		assert.equal(launchedRun.cost_actual_usd, 0.002);
		const launchCandidate = launchState.corrections.find((row) => row.correction_id === 121);
		const launchIntent = launchState.corrections.find((row) => row.correction_kind === 'rerun_intent' && row.parent_correction_id === 121);
		const launchMarker = launchState.corrections.find((row) => row.correction_kind === 'rerun_child' && row.parent_correction_id === 121);
		assert.equal(launchCandidate?.correction_kind, 'repair_candidate');
		assert.equal(launchCandidate?.status, 'open');
		assert.equal(launchIntent?.status, 'open');
		assert.equal(launchIntent?.child_run_id, launchedRun.run_id);
		assert.equal(launchMarker?.status, 'resolved');
		assert.equal(launchMarker?.child_run_id, launchedRun.run_id);
		assert.equal(launchMarker?.repair_source_dump_id, launchIntent?.repair_source_dump_id);
		const launchedAggregate = launchState.steps.find((row) => row.run_id === launchedRun.run_id && row.step_kind === 'aggregate');
		assert.equal(launchedAggregate?.fixture, 'repair_actionability_probe_merge');
		const launchedProbeRows = launchState.steps.filter((row) => row.run_id === launchedRun.run_id && row.fixture === 'repair_actionability_probe_only');
		assert.deepEqual(
			launchedProbeRows.map((row) => row.step_kind).sort(),
			['object_role_probe', 'scope_probe'],
			'probe-only launch materializes reviewed native probe rows'
		);
		const metasAfterLaunch = await repairRerunMetaFiles(dir);
		assert.equal(metasAfterLaunch.length, 5, 'fixture tombstone/recovery metas plus three browser preflight corpora exist after launch');
		summary.launch_state = launchState;

		const selectedCancel = await evaluate(client, selectOnlyRepairCandidateExpression(122));
		assert.deepEqual(selectedCancel.selected, ['122']);
		assert.deepEqual(selectedCancel.failures, []);
		const cancelPreviewClick = await evaluate(client, clickRepairPreflightExpression());
		assert.equal(cancelPreviewClick.buttonText, 'preview rerun cost');
		await waitFor(client, `document.querySelector('.cost-table-wrap tbody tr')`, 'desktop repair cancel cost preflight');
		const cancelPreflight = await evaluate(client, repairCancelPreflightExpression('desktop-rerun-cancel-preflight'));
		summary.checks.push(cancelPreflight);
		summary.screenshots.push(await screenshot(client, 'repair-actionability-rerun-cancel-preflight'));
		const cancelSpendClick = await evaluate(client, clickRepairSpendExpression(CANCEL_MODEL));
		assert.match(cancelSpendClick.buttonText ?? '', /browser-repair-cancel-smoke/);
		await waitFor(
			client,
			`(() => {
				const text = (document.body.textContent ?? '').replace(/\\s+/g, ' ');
				return text.includes('scoring') && text.includes(${JSON.stringify(CANCEL_MODEL)}) && text.includes('child') && text.includes('cancel');
			})()`,
			'repair rerun entered cancelable scoring state through browser'
		);
		const cancelScoringAudit = await evaluate(client, repairScoringExpression('desktop-rerun-cancel-scoring', CANCEL_MODEL));
		summary.checks.push(cancelScoringAudit);
		summary.screenshots.push(await screenshot(client, 'repair-actionability-rerun-cancel-scoring'));
		const cancelClick = await evaluate(client, clickRepairCancelExpression());
		assert.equal(cancelClick.buttonText, 'cancel');
		await waitFor(
			client,
			`(() => {
				const text = (document.body.textContent ?? '').replace(/\\s+/g, ' ');
				return text.includes('cancel requested') && text.includes('no repair candidates were marked rerun-complete');
			})()`,
			'repair rerun cancel request acknowledged through browser'
		);
		const cancelRequestedAudit = await evaluate(client, repairCancelRequestedExpression('desktop-rerun-cancel-requested'));
		summary.checks.push(cancelRequestedAudit);
		summary.screenshots.push(await screenshot(client, 'repair-actionability-rerun-cancel-requested'));
		try {
			await waitFor(
				client,
				`(() => {
					const text = (document.body.textContent ?? '').replace(/\\s+/g, ' ');
					const historyText = (document.querySelector('[aria-label="repair rerun comparisons"]')?.textContent ?? '').replace(/\\s+/g, ' ');
					const repairHeadText = (document.querySelector('.repair-head')?.textContent ?? '').replace(/\\s+/g, ' ');
					const tableText = Array.from(document.querySelectorAll('tbody tr')).map((row) => row.textContent ?? '').join(' ').replace(/\\s+/g, ' ');
					return text.includes('cancel requested') &&
						historyText.includes(${JSON.stringify(CANCEL_MODEL)}) &&
						historyText.includes('canceled') &&
						repairHeadText.includes('1 open repair candidate') &&
						tableText.includes('#122') &&
						!tableText.includes('#121') &&
						!document.querySelector('[aria-label="active repair reruns locking candidates"]');
				})()`,
				'repair rerun canceled and rehydrated through browser',
				15000
			);
		} catch (err) {
			const diagnostic = await evaluate(client, `(() => {
				return {
					location: location.href,
					repairHeadText: (document.querySelector('.repair-head')?.textContent ?? '').replace(/\\s+/g, ' ').trim(),
					rerunPanelText: (document.querySelector('[aria-label="rerun repair cohort"]')?.textContent ?? '').replace(/\\s+/g, ' ').trim(),
					historyText: (document.querySelector('[aria-label="repair rerun comparisons"]')?.textContent ?? '').replace(/\\s+/g, ' ').trim(),
					tableText: Array.from(document.querySelectorAll('tbody tr')).map((row) => row.textContent ?? '').join(' ').replace(/\\s+/g, ' ').trim(),
					bodyText: (document.body.textContent ?? '').replace(/\\s+/g, ' ').trim().slice(0, 2200)
				};
			})()`);
			console.error(JSON.stringify({
				cancel_rehydration_wait_failed: diagnostic,
				cancel_state: await repairLaunchRows(dbFile)
			}, null, 2));
			throw err;
		}
		const cancelRehydratedAudit = await evaluate(client, repairCancelRehydratedExpression('desktop-rerun-cancel-rehydrated'));
		summary.checks.push(cancelRehydratedAudit);
		summary.screenshots.push(await screenshot(client, 'repair-actionability-rerun-cancel-rehydrated'));
		const cancelState = await repairLaunchRows(dbFile);
		const canceledRun = cancelState.runs.find((row) => row.model_id_default === CANCEL_MODEL);
		assert.ok(canceledRun, 'canceled repair child run persisted');
		assert.match(canceledRun.run_id, /^[a-f0-9]{32}$/);
		assert.equal(canceledRun.parent_run_id, RUN);
		assert.equal(canceledRun.status, 'canceled');
		assert.equal(canceledRun.architecture, 'decomposed');
		assert.equal(canceledRun.terminated_by, 'user');
		assert.equal(canceledRun.termination_reason, 'client_disconnected');
		const canceledCandidate = cancelState.corrections.find((row) => row.correction_id === 122);
		const canceledIntent = cancelState.corrections.find((row) =>
			row.correction_kind === 'rerun_intent' &&
			row.parent_correction_id === 122 &&
			row.child_run_id === canceledRun.run_id
		);
		const canceledMarker = cancelState.corrections.find((row) =>
			row.correction_kind === 'rerun_child' &&
			row.parent_correction_id === 122 &&
			row.child_run_id === canceledRun.run_id
		);
		assert.equal(canceledCandidate?.correction_kind, 'repair_candidate');
		assert.equal(canceledCandidate?.status, 'open');
		assert.equal(canceledIntent?.status, 'open');
		assert.equal(canceledMarker, undefined, 'canceled repair child must not record rerun_child markers');
		assert.equal(cancelState.steps.some((row) => row.run_id === canceledRun.run_id), false, 'canceled fake child must not write aggregate steps');
		const metasAfterCancel = await repairRerunMetaFiles(dir);
		assert.equal(metasAfterCancel.length, 6, 'fixture tombstone/recovery metas plus four browser preflight corpora exist after cancel preflight');
		summary.cancel_state = cancelState;

		await waitFor(
			client,
			`(() => {
				const panel = document.querySelector('[aria-label="completed repair reruns awaiting markers"]');
				const button = Array.from(panel?.querySelectorAll('button') ?? []).find((btn) => (btn.textContent ?? '').includes('finalize markers'));
				return Boolean(button && !button.disabled);
			})()`,
			'completed recovery finalize button enabled'
		);
		const recoveryAudit = await evaluate(client, completedRecoveryAuditExpression('desktop-completed-recovery-before'));
		summary.checks.push(recoveryAudit);
		summary.screenshots.push(await screenshot(client, 'repair-actionability-completed-recovery-before'));
		const finalizeClick = await evaluate(client, clickFinalizeRecoveryExpression());
		assert.equal(finalizeClick.buttonText, 'finalize markers');
		await waitFor(
			client,
			`(() => {
				const repairHeadText = (document.querySelector('.repair-head')?.textContent ?? '').replace(/\\s+/g, ' ');
				const tableText = Array.from(document.querySelectorAll('tbody tr')).map((row) => row.textContent ?? '').join(' ').replace(/\\s+/g, ' ');
				const pageText = (document.body.textContent ?? '').replace(/\\s+/g, ' ');
				return !document.querySelector('[aria-label="completed repair reruns awaiting markers"]') &&
					repairHeadText.includes('1 open repair candidate') &&
					tableText.includes('#122') &&
					!tableText.includes('#124') &&
					pageText.includes(${JSON.stringify(RECOVERY_CHILD.slice(0, 8))});
			})()`,
			'completed repair child markers finalized through browser',
			15000
		);
		const recoveryFinalizedAudit = await evaluate(client, recoveryFinalizedExpression('desktop-completed-recovery-after'));
		summary.checks.push(recoveryFinalizedAudit);
		summary.screenshots.push(await screenshot(client, 'repair-actionability-completed-recovery-after'));
		const recoveryState = await repairLaunchRows(dbFile);
		const recoveryChild = recoveryState.runs.find((row) => row.run_id === RECOVERY_CHILD);
		assert.equal(recoveryChild?.status, 'succeeded');
		assert.equal(recoveryChild?.parent_run_id, RUN);
		assert.equal(recoveryChild?.architecture, 'decomposed');
		const recoveryCandidate = recoveryState.corrections.find((row) => row.correction_id === 124);
		const recoveryIntent = recoveryState.corrections.find((row) => row.correction_kind === 'rerun_intent' && row.parent_correction_id === 124);
		const recoveryMarker = recoveryState.corrections.find((row) => row.correction_kind === 'rerun_child' && row.parent_correction_id === 124);
		assert.equal(recoveryCandidate?.correction_kind, 'repair_candidate');
		assert.equal(recoveryCandidate?.status, 'open');
		assert.equal(recoveryIntent?.status, 'open');
		assert.equal(recoveryIntent?.child_run_id, RECOVERY_CHILD);
		assert.equal(recoveryMarker?.status, 'resolved');
		assert.equal(recoveryMarker?.child_run_id, RECOVERY_CHILD);
		assert.equal(recoveryMarker?.repair_source_dump_id, RECOVERY_SOURCE_DUMP);
		const recoveryAggregate = recoveryState.steps.find((row) => row.run_id === RECOVERY_CHILD && row.step_kind === 'aggregate');
		assert.equal(recoveryAggregate?.fixture, 'repair_actionability_recovery');
		const metasAfterRecovery = await repairRerunMetaFiles(dir);
		assert.equal(metasAfterRecovery.length, 6, 'marker recovery writes no new repair-rerun corpus');
		summary.recovery_state = recoveryState;

		const failures = summary.checks.flatMap((check) => (check.failures ?? []).map((failure) => `${check.viewport}: ${failure}`));
		const result = { ...summary, failures };
		await writeFile(join(artifactRoot, 'repair_actionability_browser_summary.json'), JSON.stringify(result, null, 2));
		assert.deepEqual(failures, []);
		console.log('repair actionability browser tests passed');
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
