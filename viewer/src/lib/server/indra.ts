/**
 * Live INDRA DB transport for the /curate route — sample evidence to curate, and
 * submit a human curation back. SERVER-ONLY ($lib/server, never bundled to the
 * browser).
 *
 * Sampling is UNIFORM over the tracked CoGEx reservoir manifest: pick a random
 * exact evidence-row key (5,000 rows reservoir-sampled uniformly from CoGEx),
 * then materialize that statement UNBIASED by matches_hash via
 * POST /statements/from_hashes (public) and select the evidence whose source_hash
 * matches the pool row. No agent-pool bias, no from_agents.
 *
 * Two endpoints (verified against db.indra.bio):
 *   POST /statements/from_hashes?format=json-js&ev_limit  body {hashes:[<int>]}
 *        → INDRA JSON: results{} keyed by matches_hash + belief_scores +
 *        evidence_counts; each statement carries inline evidence. PUBLIC, no auth.
 *   POST /curation/submit/<matches_hash>   body {tag, text, ev_hash:<int>, source}
 *        Authenticated by the CURATOR's own INDRA JWT (forwarded as the
 *        access_token_cookie). No shared api_key, no self-declared email — INDRA
 *        attributes the curation to the verified user. JWT plumbing: session.ts.
 *
 * HASH PRECISION: INDRA hashes are 64-bit ints that overflow JS Number. The REST
 * API returns them as quoted strings; we keep them as strings everywhere and
 * inject ev_hash into the POST body as a bare integer literal — never Number().
 */
import { existsSync, readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { CURATION_TAGS } from '$lib/data/curation';
import { getDataset, type Dataset } from './datasets';
import {
	curatorPairKeys,
	exactPairKey,
	parseCurationHistory,
	poolPairOf,
	unseenPoolLines,
	type CurationHistoryRow
} from './curation-history';
import { reservedPairKeys, tryReserveDraw } from './curation-draw-ledger';
import { indraBaseUrl } from './session';
import type { Claim, EvidenceAgent, EvidenceSample } from '$lib/data/types';

/** Per-request budget for live INDRA calls. Without a signal, undici waits ~5 min
 *  for a stalled response, which would hang the SSR load across all retries. */
const REQUEST_TIMEOUT_MS = 30_000;
// ── statement rendering (INDRA JSON → subject/relation/object) ───────────────────

/** Humanised relation verbs for the common INDRA statement types; anything else
 *  falls back to the CamelCase type spaced into words. */
const RELATION: Record<string, string> = {
	Complex: 'binds',
	Activation: 'activates',
	Inhibition: 'inhibits',
	IncreaseAmount: 'increases amount of',
	DecreaseAmount: 'decreases amount of',
	Phosphorylation: 'phosphorylates',
	Dephosphorylation: 'dephosphorylates',
	Ubiquitination: 'ubiquitinates',
	Deubiquitination: 'deubiquitinates',
	Acetylation: 'acetylates',
	Methylation: 'methylates',
	Sumoylation: 'sumoylates',
	Glycosylation: 'glycosylates',
	Ribosylation: 'ribosylates',
	Hydroxylation: 'hydroxylates',
	Farnesylation: 'farnesylates',
	Palmitoylation: 'palmitoylates',
	Gef: 'is a GEF for',
	Gap: 'is a GAP for',
	GtpActivation: 'activates (GTP)',
	Translocation: 'translocates',
	ActiveForm: 'has active form',
	Conversion: 'converts'
};

function agentName(a: unknown): string | null {
	if (!a) return null;
	if (typeof a === 'string') return a;
	const o = a as { name?: string; db_refs?: Record<string, string> };
	return o.name ?? o.db_refs?.HGNC ?? o.db_refs?.TEXT ?? null;
}

function dbRefs(a: unknown): Record<string, string> {
	if (!a || typeof a === 'string') return {};
	const refs = (a as { db_refs?: Record<string, unknown> }).db_refs ?? {};
	const out: Record<string, string> = {};
	for (const [k, v] of Object.entries(refs)) {
		if (v != null) out[k] = String(v);
	}
	return out;
}

function agentEntries(stmt: Record<string, unknown>): Array<{ role: EvidenceAgent['role']; agent: unknown }> {
	if (Array.isArray(stmt.members)) return stmt.members.map((agent) => ({ role: 'member', agent }));
	const out: Array<{ role: EvidenceAgent['role']; agent: unknown }> = [];
	const subj = stmt.enz ?? stmt.subj ?? stmt.gef ?? stmt.agent ?? stmt.sub_obj;
	const obj = stmt.sub ?? stmt.obj ?? stmt.obj_to ?? stmt.ras;
	if (subj) out.push({ role: 'subject', agent: subj });
	if (obj) out.push({ role: 'object', agent: obj });
	if (!out.length) {
		const a = stmt.agent ?? stmt.sub_obj;
		if (a) out.push({ role: 'agent', agent: a });
	}
	return out;
}

function rawAgentTexts(ev: Record<string, unknown>): string[] {
	const ann = ev.annotations as { agents?: { raw_text?: unknown } } | undefined;
	const raw = ann?.agents?.raw_text;
	return Array.isArray(raw) ? raw.map((x) => (x == null ? '' : String(x))) : [];
}

function evidenceAgents(stmt: Record<string, unknown>, ev: Record<string, unknown>): EvidenceAgent[] {
	const raws = rawAgentTexts(ev);
	return agentEntries(stmt)
		.map(({ role, agent }, i) => {
			const name = agentName(agent);
			if (!name) return null;
			const rawText = raws[i] && raws[i] !== name ? raws[i] : (raws[i] || null);
			return { role, name, rawText, dbRefs: dbRefs(agent) };
		})
		.filter((a): a is EvidenceAgent => !!a);
}

function humanType(t: string): string {
	return RELATION[t] ?? t.replace(/([a-z0-9])([A-Z])/g, '$1 $2').toLowerCase();
}

/** Render one INDRA statement dict to a subject/relation/object triple. Handles
 *  the common shapes (members, enz/sub, subj/obj, single agent) and degrades to
 *  a bare agent list for anything exotic. */
export function renderStatement(stmt: Record<string, unknown>): Claim {
	const type = (stmt.type as string) ?? 'Statement';
	const relation = humanType(type);

	if (Array.isArray(stmt.members)) {
		const names = stmt.members.map(agentName).filter((n): n is string => !!n);
		const subject = names[0] ?? '?';
		const object = names.slice(1).join(', ') || '—';
		return { subject, relation, object, full: `${type}(${names.join(', ')})` };
	}

	const subj = agentName(stmt.enz ?? stmt.subj ?? stmt.gef ?? stmt.agent ?? stmt.sub_obj);
	let obj = agentName(stmt.sub ?? stmt.obj ?? stmt.obj_to ?? stmt.ras);
	// modification site, when present, sharpens the object (e.g. "MAPK1 (T185)")
	const residue = stmt.residue as string | undefined;
	const position = stmt.position as string | undefined;
	if (obj && (residue || position)) obj = `${obj} (${residue ?? ''}${position ?? ''})`;

	const subject = subj ?? '?';
	const object = obj ?? '—';
	const full = obj ? `${subject} ${relation} ${object}` : `${type}(${subject})`;
	return { subject, relation, object, full };
}

// ── sampling ────────────────────────────────────────────────────────────────

const HASH_RE = /^-?\d+$/;

function pick<T>(xs: readonly T[]): T {
	return xs[Math.floor(Math.random() * xs.length)];
}

/** The load-bearing part of one local CoGEx pool line: the statement + evidence
 *  hashes that select the pair to materialize from the DB (which then supplies the
 *  assembled claim, agents, belief, text). Both are 64-bit ints that OVERFLOW JS
 *  Number, so they are kept as EXACT-DIGIT STRINGS — never JSON.parse'd (which
 *  would silently round e.g. …258 → …256 and break the by-hash join). */
/** Resolve a dataset's file (e.g. 'corpora/…') under the viewer's DATA_DIR via the
 *  same pattern runs.ts uses (viewer/ cwd → ../data). dataset.file already carries
 *  the 'corpora/…' prefix. */
function dataPath(file: string): string {
	return resolve(process.cwd(), '..', 'data', file);
}

const _lineCache = new Map<string, string[]>();

/** Read + cache a JSONL's non-empty lines, keyed by resolved path (one dataset
 *  file per key). Reads each file once per server process; throws if the file is
 *  missing (caught upstream → sampleError, same graceful UX as a live DB failure). */
function linesOf(path: string): string[] {
	const cached = _lineCache.get(path);
	if (cached) return cached;
	const txt = readFileSync(path, 'utf8');
	const lines = txt.split('\n').filter((l) => l.trim().length > 0);
	_lineCache.set(path, lines);
	return lines;
}

/** Materialize a statement (and all its evidence) UNBIASED by its matches_hash.
 *  A generous ev_limit ensures the target source_hash's evidence is in the set.
 *  stmtHash is HASH_RE-validated (digits only), so injecting it as a bare integer
 *  literal in the body is precision-safe — JSON.stringify of a Number would round
 *  the 64-bit hash. */
async function fetchFromHashes(
	baseUrl: string,
	stmtHash: string,
	evLimit: number
): Promise<Record<string, unknown> | null> {
	const url = `${baseUrl}/statements/from_hashes?format=json-js&ev_limit=${evLimit}`;
	const r = await fetch(url, {
		method: 'POST',
		headers: { 'Content-Type': 'application/json', 'User-Agent': 'indra-belief-curate/1' },
		body: `{"hashes":[${stmtHash}]}`,
		signal: AbortSignal.timeout(REQUEST_TIMEOUT_MS)
	});
	if (!r.ok) return null;
	return (await r.json()) as Record<string, unknown>;
}

/** Assemble an EvidenceSample from an already-selected (statement, evidence) pair.
 *  Shared by the pool sampler (belief/evCount from the DB response) and the inline
 *  sampler (belief/evCount from the statement itself). `datasetId` records which
 *  universe the draw came from — threaded to the /curate frame header. Hashes are
 *  passed through as STRINGS, never Number(). */
function buildSample(
	stmt: Record<string, unknown>,
	ev: Record<string, unknown>,
	belief: number | null,
	evCount: number,
	datasetId: string,
	drawToken: string
): EvidenceSample {
	const refs = (ev.text_refs as Record<string, string>) ?? {};
	const pmid = (ev.pmid as string) ?? refs.PMID ?? null;
	return {
		matchesHash: stmt.matches_hash as string,
		sourceHash: ev.source_hash as string,
		text: (ev.text as string).trim(),
		pmid: pmid ? String(pmid) : null,
		pmcid: refs.PMCID ? String(refs.PMCID) : null,
		sourceApi: (ev.source_api as string) ?? null,
		stmtType: (stmt.type as string) ?? 'Statement',
		belief,
		claim: renderStatement(stmt),
		agents: evidenceAgents(stmt, ev),
		dataset: datasetId,
		drawToken,
		evCount
	};
}

/** A usable curation sentence: long enough and not publisher-redacted. Shared by
 *  both samplers so the card never renders an empty or credential-limited body. */
function usableText(text: unknown): text is string {
	if (typeof text !== 'string' || text.trim().length < 12) return false;
	return !/MISSING\/INVALID CREDENTIALS|limited to \d+ char/i.test(text);
}

/** Sample one (statement, evidence) pair from a dataset, dispatching on its kind:
 *   'cogex-pool'        → uniform-random pool line materialized via the live DB.
 *   'inline-statements' → random statement × random evidence from a local JSONL.
 *  Unknown/empty datasetId falls back to the representative draw (DATASETS[0]). */
/** Sample one (statement, evidence) pair UNIFORMLY at random from a dataset — ONE
 *  strategy for every dataset kind. Pick a random pool line, resolve it into a
 *  candidate (the ONLY kind-specific step, `resolverFor`), apply the shared guards
 *  (exact-digit STRING hashes + a usable, non-redacted sentence), and retry another
 *  random line on any dead-end (version skew, text-less DB rows ~10-16%, redacted
 *  text) up to maxAttempts. New dataset kind = one new branch in resolverFor only. */
export async function sampleEvidence(
	datasetId?: string | null,
	curatedKeys: ReadonlySet<string> = new Set(),
	curatorEmail = '',
	maxAttempts = 8
): Promise<EvidenceSample> {
	const ds = getDataset(datasetId);
	if (!curatorEmail.trim()) throw new Error('cannot sample without an authenticated curator identity');
	const path = dataPath(ds.file);
	if (!existsSync(path)) {
		throw new Error(ds.provisioning ?? `dataset frame is not provisioned: ${ds.file}`);
	}
	const lines = linesOf(path);
	// Completed curations come from the shared INDRA history; all cards previously
	// drawn (including skips and unsubmitted cards) come from the persistent local
	// reservation ledger. Their union is the exact without-replacement boundary.
	const blockedKeys = new Set(curatedKeys);
	for (const key of reservedPairKeys(curatorEmail, ds.id)) blockedKeys.add(key);
	const candidates = ds.kind === 'cogex-pool' ? unseenPoolLines(lines, blockedKeys) : [...lines];
	if (!candidates.length) throw new Error(`every evidence pair in ${ds.id} has already been curated`);
	const resolve = resolverFor(ds, blockedKeys);
	let lastErr = 'no usable evidence found';
	for (let i = 0; i < maxAttempts && candidates.length; i++) {
		try {
			// Remove the attempted candidate locally too, so retries in this request
			// cannot redraw the same dead-end line.
			const index = Math.floor(Math.random() * candidates.length);
			const [line] = candidates.splice(index, 1);
			const r = await resolve(line);
			if (!r) continue; // dead-end draw — retry another random line
			const mh = r.stmt.matches_hash;
			const sh = r.ev.source_hash;
			if (typeof mh !== 'string' || !HASH_RE.test(mh)) continue;
			if (typeof sh !== 'string' || !HASH_RE.test(sh)) continue;
			const key = exactPairKey(mh, sh);
			if (!key || blockedKeys.has(key)) continue;
			if (!usableText(r.ev.text)) continue;
			// Exclusive creation is the cross-request race boundary. If another tab or
			// process reserved this candidate while it materialized, retry a distinct
			// candidate rather than returning a duplicate draw.
			const drawToken = tryReserveDraw(curatorEmail, ds.id, key);
			if (!drawToken) continue;
			return buildSample(r.stmt, r.ev, r.belief, r.evCount, ds.id, drawToken);
		} catch (e) {
			lastErr = e instanceof Error ? e.message : String(e);
		}
	}
	throw new Error(`could not sample evidence from ${ds.id} (${lastErr})`);
}

/** A resolved candidate: the (statement, evidence) pair plus the belief +
 *  evidence-count the dataset's materialization exposes. `null` ⇒ dead-end, retry. */
interface Resolved {
	stmt: Record<string, unknown>;
	ev: Record<string, unknown>;
	belief: number | null;
	evCount: number;
}

/** The ONLY per-dataset-kind step: turn one random pool line into a materialized
 *  candidate. Everything around it (the uniform draw, the guards, the retry loop,
 *  buildSample) is shared in sampleEvidence.
 *   - 'inline-statements' — the line IS the statement (evidence inline, hashes already
 *       quoted strings by build_curate_pool.py): no network, skew-free; pick one of
 *       its evidences.
 *   - 'cogex-pool' — the line carries only hashes: materialize the statement UNBIASED
 *       by matches_hash via /statements/from_hashes, then select the row's evidence
 *       (null on version skew / re-hashed evidence → retry). */
function resolverFor(
	ds: Dataset,
	curatedKeys: ReadonlySet<string>
): (line: string) => Promise<Resolved | null> {
	if (ds.kind === 'inline-statements') {
		return async (line) => {
			const stmt = JSON.parse(line) as Record<string, unknown>;
			const mh = typeof stmt.matches_hash === 'string' ? stmt.matches_hash : null;
			const evs = (Array.isArray(stmt.evidence) ? (stmt.evidence as Record<string, unknown>[]) : []).filter(
				(ev) => {
					const sh = typeof ev.source_hash === 'string' ? ev.source_hash : null;
					const key = exactPairKey(mh, sh);
					return key != null && !curatedKeys.has(key);
				}
			);
			if (!evs.length) return null;
			const belief = typeof stmt.belief === 'number' ? (stmt.belief as number) : null;
			return { stmt, ev: pick(evs), belief, evCount: evs.length };
		};
	}
	const baseUrl = indraBaseUrl();
	return async (line) => {
		const pair = poolPairOf(line);
		if (!pair) return null;
		const resp = await fetchFromHashes(baseUrl, pair.stmtHash, 10_000);
		if (!resp) return null; // INDRA DB error — retry another row
		const results = resp.results as Record<string, Record<string, unknown>> | undefined;
		const stmt = results?.[pair.stmtHash];
		if (!stmt) return null; // version skew: hash didn't materialize — retry
		const evs = Array.isArray(stmt.evidence) ? (stmt.evidence as Record<string, unknown>[]) : [];
		const ev = evs.find((e) => String(e.source_hash) === pair.sourceHash);
		if (!ev) return null; // evidence re-hashed / dropped by ev_limit — retry
		const mh = stmt.matches_hash as string;
		const beliefScores = (resp.belief_scores as Record<string, number>) ?? {};
		const evCounts = (resp.evidence_counts as Record<string, number>) ?? {};
		const belief =
			typeof beliefScores[mh] === 'number' ? beliefScores[mh] : ((stmt.belief as number) ?? null);
		const evCount = typeof evCounts[mh] === 'number' ? evCounts[mh] : 0;
		return { stmt, ev, belief, evCount };
	};
}

// ── submission ──────────────────────────────────────────────────────────────

export interface SubmitArgs {
	matchesHash: string;
	sourceHash: string;
	dataset: string;
	tag: string;
	text: string;
	/** The curator's INDRA JWT (from their session). INDRA authenticates it and
	 *  attributes the curation to that user — no shared key, no declared email. */
	jwt: string;
}

export interface SubmitResult {
	ok: boolean;
	id?: number;
	result?: string;
	status?: number;
	error?: string;
}

export interface CuratorStats {
	status: 'available' | 'unavailable';
	total: number;
	correct: number;
	incorrect: number;
	reason?: string;
}

export interface CuratorContext {
	stats: CuratorStats;
	curatedKeys: ReadonlySet<string>;
}

let _curatorStatsCache: {
	email: string;
	until: number;
	stats: CuratorStats;
	curatedKeys: Set<string>;
} | null = null;
let _indraApiKey: string | null | undefined;

function indraApiKey(): string | null {
	if (_indraApiKey !== undefined) return _indraApiKey;
	let key = process.env.INDRA_DB_REST_API_KEY ?? '';
	if (!key) {
		try {
			const txt = readFileSync(resolve(process.cwd(), '..', '.env'), 'utf8');
			for (const raw of txt.split('\n')) {
				const m = raw.match(/^\s*INDRA_DB_REST_API_KEY\s*=(.*)$/);
				if (!m) continue;
				let v = m[1].trim();
				if (v.length >= 2 && ((v[0] === '"' && v.at(-1) === '"') || (v[0] === "'" && v.at(-1) === "'")))
					v = v.slice(1, -1);
				key = v;
				break;
			}
		} catch {
			/* no repo env file */
		}
	}
	_indraApiKey = key || null;
	return _indraApiKey;
}

async function fetchCurationListRows(jwt: string): Promise<CurationHistoryRow[]> {
	let r: Response;
	const key = indraApiKey();
	const url = key
		? `${indraBaseUrl()}/curation/list?api_key=${encodeURIComponent(key)}`
		: `${indraBaseUrl()}/curation/list`;
	const headers: Record<string, string> = { 'User-Agent': 'indra-belief-curate/1' };
	// The definitive stats interface is the API-keyed list-all endpoint. Do not
	// mix a curator JWT into that request; an invalid/expired cookie can change the
	// auth path. JWT is only a fallback when no server API key is configured.
	if (!key) headers.Cookie = `access_token_cookie=${jwt}`;
	try {
		r = await fetch(url, { headers, signal: AbortSignal.timeout(REQUEST_TIMEOUT_MS) });
	} catch (e) {
		throw new Error(`could not load curation counts: ${e instanceof Error ? e.message : String(e)}`);
	}
	if (!r.ok) {
		throw new Error(`could not load curation counts: HTTP ${r.status}`);
	}

	const text = await r.text();
	try {
		return parseCurationHistory(text);
	} catch {
		const prefix = text.replace(/\s+/g, ' ').slice(0, 80);
		throw new Error(`curation counts were not JSON (${r.headers.get('content-type') ?? 'no content-type'}; ${prefix})`);
	}
}

/** Count the signed-in curator's full INDRA curation history. Confirmed
 *  interface: keyed GET /curation/list returns a JSON array whose rows contain
 *  at least {curator, tag}; correct is tag === "correct", every other tag is an
 *  incorrect/error flag. */
export async function getCuratorContext(
	jwt: string,
	email: string,
	forceRefresh = false
): Promise<CuratorContext> {
	if (!jwt || !email) {
		return {
			stats: { status: 'unavailable', total: 0, correct: 0, incorrect: 0, reason: 'not signed in' },
			curatedKeys: new Set()
		};
	}
	const now = Date.now();
	if (!forceRefresh && _curatorStatsCache && _curatorStatsCache.email === email && _curatorStatsCache.until > now) {
		return { stats: _curatorStatsCache.stats, curatedKeys: _curatorStatsCache.curatedKeys };
	}

	let rows: CurationHistoryRow[];
	try {
		rows = await fetchCurationListRows(jwt);
	} catch (e) {
		return {
			stats: {
				status: 'unavailable',
				total: 0,
				correct: 0,
				incorrect: 0,
				reason: e instanceof Error ? e.message : String(e)
			},
			curatedKeys: new Set()
		};
	}

	const emailKey = email.toLowerCase();
	const mine = rows.filter((row) => row.curator.toLowerCase() === emailKey);
	const correct = mine.filter((row) => row.tag === 'correct').length;
	const stats = {
		status: 'available' as const,
		total: mine.length,
		correct,
		incorrect: mine.length - correct
	};
	const curatedKeys = curatorPairKeys(rows, email);
	_curatorStatsCache = { email, until: now + 60_000, stats, curatedKeys };
	return { stats, curatedKeys };
}

export async function getCuratorStats(jwt: string, email: string): Promise<CuratorStats> {
	return (await getCuratorContext(jwt, email)).stats;
}

export function noteCuratorSubmission(
	email: string,
	tag: string,
	matchesHash: string,
	sourceHash: string
): void {
	if (!_curatorStatsCache || _curatorStatsCache.email !== email || _curatorStatsCache.stats.status !== 'available') {
		return;
	}
	const stats = _curatorStatsCache.stats;
	const curatedKeys = new Set(_curatorStatsCache.curatedKeys);
	const key = exactPairKey(matchesHash, sourceHash);
	if (key) curatedKeys.add(key);
	_curatorStatsCache = {
		email,
		until: _curatorStatsCache.until,
		curatedKeys,
		stats: {
			...stats,
			total: stats.total + 1,
			correct: stats.correct + (tag === 'correct' ? 1 : 0),
			incorrect: stats.incorrect + (tag === 'correct' ? 0 : 1)
		}
	};
}

/** Submit one human curation to the live INDRA DB under the curator's own JWT. The
 *  statement hash goes in the URL path verbatim; ev_hash is injected as a bare
 *  integer literal so the 64-bit value survives intact (JSON.stringify of a Number
 *  would corrupt it). The JWT is forwarded as INDRA's access_token_cookie; INDRA
 *  derives the curator email from it, so we send no email and no api_key. */
export async function submitCuration(a: SubmitArgs): Promise<SubmitResult> {
	const baseUrl = indraBaseUrl();
	if (!a.jwt) return { ok: false, error: 'not signed in' };
	if (!HASH_RE.test(a.matchesHash)) return { ok: false, error: 'invalid statement hash' };
	if (!HASH_RE.test(a.sourceHash)) return { ok: false, error: 'invalid evidence hash' };
	const dataset = getDataset(a.dataset);
	if (!a.dataset || dataset.id !== a.dataset) return { ok: false, error: 'invalid curation dataset' };
	// Validate the tag against the canonical vocabulary server-side — the radio
	// list in the page is browser-only; this is the trust boundary, so a
	// non-canonical tag must not reach the shared gold DB.
	if (!CURATION_TAGS.includes(a.tag)) return { ok: false, error: 'invalid curation tag' };

	const url = `${baseUrl}/curation/submit/${a.matchesHash}`;
	// Hand-built JSON so ev_hash is a bare integer literal (precision-safe). The
	// string fields are JSON.stringify'd individually for correct escaping.
	const body =
		`{"tag":${JSON.stringify(a.tag)},` +
		`"text":${JSON.stringify(a.text ?? '')},` +
		`"ev_hash":${a.sourceHash},` +
		`"source":${JSON.stringify(`indra-belief viewer/${dataset.id}`)}}`;

	let r: Response;
	try {
		r = await fetch(url, {
			method: 'POST',
			headers: {
				'Content-Type': 'application/json',
				// the curator's INDRA session — INDRA authenticates + attributes to them
				Cookie: `access_token_cookie=${a.jwt}`
			},
			body,
			signal: AbortSignal.timeout(REQUEST_TIMEOUT_MS)
		});
	} catch (e) {
		return { ok: false, error: `network error: ${e instanceof Error ? e.message : String(e)}` };
	}

	let data: { result?: string; reason?: string; message?: string; ref?: { id?: number } } = {};
	try {
		data = await r.json();
	} catch {
		/* non-JSON error body */
	}
	if (!r.ok) {
		return { ok: false, status: r.status, error: data.reason || data.message || `HTTP ${r.status}` };
	}
	return { ok: true, id: data.ref?.id, result: data.result };
}
