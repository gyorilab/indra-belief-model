/**
 * Live INDRA DB transport for the /curate route — sample evidence to curate, and
 * submit a human curation back. SERVER-ONLY ($lib/server, never bundled to the
 * browser).
 *
 * Two endpoints (verified against db.indra.bio):
 *   GET  /statements/from_agents?agent=…&format=json-js&ev_limit&max_stmts&offset
 *        → INDRA JSON; statements carry inline evidence. PUBLIC, no auth.
 *   POST /curation/submit/<matches_hash>   body {tag, text, ev_hash:<int>, source}
 *        Authenticated by the CURATOR's own INDRA JWT (forwarded as the
 *        access_token_cookie). No shared api_key, no self-declared email — INDRA
 *        attributes the curation to the verified user. JWT plumbing: session.ts.
 *
 * HASH PRECISION: INDRA hashes are 64-bit ints that overflow JS Number. The REST
 * API returns them as quoted strings; we keep them as strings everywhere and
 * inject ev_hash into the POST body as a bare integer literal — never Number().
 */
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { AGENT_POOL } from './agents';
import { CURATION_TAGS } from '$lib/data/curation';
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

async function fetchFromAgent(
	baseUrl: string,
	agent: string,
	offset: number,
	evLimit: number,
	maxStmts: number
): Promise<Record<string, unknown> | null> {
	const url =
		`${baseUrl}/statements/from_agents?agent=${encodeURIComponent(agent)}` +
		`&format=json-js&ev_limit=${evLimit}&max_stmts=${maxStmts}&offset=${offset}`;
	const r = await fetch(url, {
		headers: { 'User-Agent': 'indra-belief-curate/1' },
		signal: AbortSignal.timeout(REQUEST_TIMEOUT_MS)
	});
	if (!r.ok) return null;
	return (await r.json()) as Record<string, unknown>;
}

interface Pair {
	stmt: Record<string, unknown>;
	ev: Record<string, unknown>;
}

/** Flatten a from_agents response into (statement, evidence) pairs that have a
 *  usable sentence and valid string hashes. */
function pairsOf(resp: Record<string, unknown>): Pair[] {
	const statements = (resp.statements as Record<string, Record<string, unknown>>) ?? {};
	const out: Pair[] = [];
	for (const stmt of Object.values(statements)) {
		const mh = stmt.matches_hash;
		if (typeof mh !== 'string' || !HASH_RE.test(mh)) continue;
		const evs = Array.isArray(stmt.evidence) ? (stmt.evidence as Record<string, unknown>[]) : [];
		for (const ev of evs) {
			const sh = ev.source_hash;
			const text = ev.text;
			if (typeof sh !== 'string' || !HASH_RE.test(sh)) continue;
			if (typeof text !== 'string' || text.trim().length < 12) continue;
			// Skip publisher-redacted text — INDRA truncates restricted full-text
			// (e.g. Elsevier) to ~200 chars and tacks on a credentials placeholder.
			// A curator can't judge an extraction against a chopped sentence.
			if (/MISSING\/INVALID CREDENTIALS|limited to \d+ char/i.test(text)) continue;
			out.push({ stmt, ev });
		}
	}
	return out;
}

function toSample(resp: Record<string, unknown>, pair: Pair, agentQuery: string): EvidenceSample {
	const { stmt, ev } = pair;
	const mh = stmt.matches_hash as string;
	const beliefScores = (resp.belief_scores as Record<string, number>) ?? {};
	const evCounts = (resp.evidence_counts as Record<string, number>) ?? {};
	const refs = (ev.text_refs as Record<string, string>) ?? {};
	const pmid = (ev.pmid as string) ?? refs.PMID ?? null;
	return {
		matchesHash: mh,
		sourceHash: ev.source_hash as string,
		text: (ev.text as string).trim(),
		pmid: pmid ? String(pmid) : null,
		pmcid: refs.PMCID ? String(refs.PMCID) : null,
		sourceApi: (ev.source_api as string) ?? null,
		stmtType: (stmt.type as string) ?? 'Statement',
		belief: typeof beliefScores[mh] === 'number' ? beliefScores[mh] : ((stmt.belief as number) ?? null),
		claim: renderStatement(stmt),
		agents: evidenceAgents(stmt, ev),
		agentQuery,
		evCount: typeof evCounts[mh] === 'number' ? evCounts[mh] : 0
	};
}

/** Sample one (statement, evidence) pair broadly from the live DB: random agent
 *  from the pool × random offset × random statement × random evidence. Retries a
 *  few agents/offsets if a draw dead-ends (sparse agent or offset past the end),
 *  with the final attempt pinned to offset 0 so a pool agent always yields. */
export async function sampleEvidence(maxAttempts = 8): Promise<EvidenceSample> {
	const baseUrl = indraBaseUrl();
	let lastErr = 'no evidence found';
	for (let i = 0; i < maxAttempts; i++) {
		const agent = pick(AGENT_POOL);
		const offset = i === maxAttempts - 1 ? 0 : Math.floor(Math.random() * 90);
		try {
			const resp = await fetchFromAgent(baseUrl, agent, offset, 8, 20);
			if (!resp) {
				lastErr = `INDRA DB error for ${agent}`;
				continue;
			}
			const pairs = pairsOf(resp);
			if (pairs.length === 0) continue;
			return toSample(resp, pick(pairs), agent);
		} catch (e) {
			lastErr = e instanceof Error ? e.message : String(e);
		}
	}
	throw new Error(`could not sample evidence from INDRA DB (${lastErr})`);
}

// ── submission ──────────────────────────────────────────────────────────────

export interface SubmitArgs {
	matchesHash: string;
	sourceHash: string;
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

let _curatorStatsCache: { email: string; until: number; stats: CuratorStats } | null = null;
let _indraApiKey: string | null | undefined;
type CurationListRow = { curator: string; tag: string };

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

async function fetchCurationListRows(jwt: string): Promise<CurationListRow[]> {
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
	let rows: unknown;
	try {
		rows = JSON.parse(text);
	} catch {
		const prefix = text.replace(/\s+/g, ' ').slice(0, 80);
		throw new Error(`curation counts were not JSON (${r.headers.get('content-type') ?? 'no content-type'}; ${prefix})`);
	}
	if (!Array.isArray(rows)) {
		throw new Error('unexpected curation counts payload');
	}
	return rows
		.filter((row): row is { curator?: unknown; tag?: unknown } => row != null && typeof row === 'object')
		.map((row) => ({
			curator: typeof row.curator === 'string' ? row.curator : '',
			tag: typeof row.tag === 'string' ? row.tag : ''
		}));
}

/** Count the signed-in curator's full INDRA curation history. Confirmed
 *  interface: keyed GET /curation/list returns a JSON array whose rows contain
 *  at least {curator, tag}; correct is tag === "correct", every other tag is an
 *  incorrect/error flag. */
export async function getCuratorStats(jwt: string, email: string): Promise<CuratorStats> {
	if (!jwt || !email) return { status: 'unavailable', total: 0, correct: 0, incorrect: 0, reason: 'not signed in' };
	const now = Date.now();
	if (_curatorStatsCache && _curatorStatsCache.email === email && _curatorStatsCache.until > now) {
		return _curatorStatsCache.stats;
	}

	let rows: CurationListRow[];
	try {
		rows = await fetchCurationListRows(jwt);
	} catch (e) {
		return {
			status: 'unavailable',
			total: 0,
			correct: 0,
			incorrect: 0,
			reason: e instanceof Error ? e.message : String(e)
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
	_curatorStatsCache = { email, until: now + 60_000, stats };
	return stats;
}

export function noteCuratorSubmission(email: string, tag: string): void {
	if (!_curatorStatsCache || _curatorStatsCache.email !== email || _curatorStatsCache.stats.status !== 'available') {
		return;
	}
	const stats = _curatorStatsCache.stats;
	_curatorStatsCache = {
		email,
		until: _curatorStatsCache.until,
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
		`"source":"indra-belief viewer"}`;

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
