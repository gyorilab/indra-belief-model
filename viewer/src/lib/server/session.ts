/**
 * Viewer authentication — per-curator, backed by INDRA's own login.
 *
 * Model: a curator logs in with their INDRA account; we proxy db.indra.bio/login,
 * take the returned JWT (access_token_cookie), and seal it into our OWN httpOnly
 * cookie. Curations are then submitted under that user's JWT (see indra.ts), so
 * INDRA itself authenticates and attributes them — there is no shared api_key on
 * the write path, and the curator identity is verified, not self-declared.
 *
 * The cookie carries the JWT (a bearer token, integrity self-evident to INDRA)
 * plus an HMAC over it keyed by VIEWER_SESSION_SECRET, so the app only accepts
 * sessions it issued. Confidentiality is provided by httpOnly + Secure transport,
 * not by encrypting at rest (if the cookie leaks, the JWT leaks regardless — the
 * same property as INDRA's own cookie). SERVER-ONLY ($lib/server).
 */
import { createHmac, timingSafeEqual } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import type { Cookies } from '@sveltejs/kit';

export const SESSION_COOKIE = 'vsession';
const REQUEST_TIMEOUT_MS = 10_000;

// ── config (env first, then the repo-root .env) ─────────────────────────────

let _secret: string | null = null;
let _baseUrl: string | null = null;

function loadEnv(): { secret: string; baseUrl: string } {
	if (_secret !== null && _baseUrl !== null) return { secret: _secret, baseUrl: _baseUrl };
	let secret = process.env.VIEWER_SESSION_SECRET ?? '';
	let baseUrl = process.env.INDRA_DB_REST_URL ?? '';
	if (!secret || !baseUrl) {
		try {
			const txt = readFileSync(resolve(process.cwd(), '..', '.env'), 'utf8');
			for (const raw of txt.split('\n')) {
				const m = raw.match(/^\s*([A-Z_][A-Z0-9_]*)\s*=(.*)$/);
				if (!m) continue;
				let v = m[2].trim();
				if (v.length >= 2 && ((v[0] === '"' && v.at(-1) === '"') || (v[0] === "'" && v.at(-1) === "'")))
					v = v.slice(1, -1);
				if (m[1] === 'VIEWER_SESSION_SECRET' && !secret) secret = v;
				else if (m[1] === 'INDRA_DB_REST_URL' && !baseUrl) baseUrl = v;
			}
		} catch {
			/* fall through to the guard below */
		}
	}
	_secret = secret;
	_baseUrl = (baseUrl || 'https://db.indra.bio').replace(/\/+$/, '');
	return { secret: _secret, baseUrl: _baseUrl };
}

/** Fail closed: a missing/short secret must stop auth, never silently weaken it. */
function requireSecret(): string {
	const { secret } = loadEnv();
	if (!secret || secret.length < 32) {
		throw new Error(
			'VIEWER_SESSION_SECRET is missing or too short (need ≥32 chars) — refusing to run auth insecurely. Add it to the repo-root .env.'
		);
	}
	return secret;
}

export function indraBaseUrl(): string {
	return loadEnv().baseUrl;
}

// ── cookie sealing (HMAC-signed, not encrypted — see header) ─────────────────

function b64url(buf: Buffer): string {
	return buf.toString('base64url');
}

function sign(value: string): string {
	return b64url(createHmac('sha256', requireSecret()).update(value).digest());
}

/** Seal a raw INDRA JWT into the cookie value: `<jwt>.<hmac(jwt)>`. */
export function sealSession(jwt: string): string {
	return `${jwt}.${sign(jwt)}`;
}

export interface SessionData {
	jwt: string;
	email: string;
	exp: number; // unix seconds, from the JWT
}

function decodeJwtIdentity(jwt: string): { email: string; exp: number } | null {
	const parts = jwt.split('.');
	if (parts.length !== 3) return null;
	try {
		const payload = JSON.parse(Buffer.from(parts[1], 'base64url').toString('utf8'));
		const email = payload?.identity?.email;
		const exp = payload?.exp;
		if (typeof email !== 'string' || typeof exp !== 'number') return null;
		return { email, exp };
	} catch {
		return null;
	}
}

/** Verify + decode a sealed cookie. Returns null on any tamper, malformed token,
 *  or expiry — callers treat null as "logged out". */
export function readSession(cookieValue: string | undefined): SessionData | null {
	if (!cookieValue) return null;
	const i = cookieValue.lastIndexOf('.');
	if (i <= 0) return null;
	const jwt = cookieValue.slice(0, i);
	const mac = cookieValue.slice(i + 1);
	let expected: string;
	try {
		expected = sign(jwt);
	} catch {
		return null; // no/short secret → fail closed
	}
	const a = Buffer.from(mac);
	const b = Buffer.from(expected);
	if (a.length !== b.length || !timingSafeEqual(a, b)) return null;
	const id = decodeJwtIdentity(jwt);
	if (!id) return null;
	if (id.exp * 1000 <= Date.now()) return null; // expired
	return { jwt, email: id.email, exp: id.exp };
}

/** Cookie options. `secure` tracks the actual transport so it works on local
 *  http while being Secure under https in deployment. */
export function sessionCookieOptions(url: URL, maxAgeSeconds: number) {
	return {
		path: '/',
		httpOnly: true,
		sameSite: 'lax' as const,
		secure: url.protocol === 'https:',
		maxAge: maxAgeSeconds
	};
}

export function clearSession(cookies: Cookies, url: URL): void {
	cookies.delete(SESSION_COOKIE, { path: '/', secure: url.protocol === 'https:' });
}

// ── login rate limit (best-effort, per-instance) ────────────────────────────

/** Fixed-window login throttle keyed by caller (IP + email). PER-PROCESS ONLY —
 *  a multi-instance / serverless deploy needs a shared store (e.g. Redis) keyed
 *  the same way, and getClientAddress() needs ADDRESS_HEADER set behind a proxy.
 *  This raises the bar against local credential-stuffing and against the viewer
 *  being abused as an IP-masking auth proxy into INDRA. */
const LOGIN_MAX = 10;
const LOGIN_WINDOW_MS = 5 * 60_000;
const _loginHits = new Map<string, { n: number; resetAt: number }>();

export function loginThrottled(key: string): boolean {
	const now = Date.now();
	const e = _loginHits.get(key);
	if (!e || e.resetAt <= now) {
		_loginHits.set(key, { n: 1, resetAt: now + LOGIN_WINDOW_MS });
		return false;
	}
	e.n += 1;
	return e.n > LOGIN_MAX;
}

// ── INDRA login proxy ───────────────────────────────────────────────────────

export interface LoginResult {
	ok: boolean;
	jwt?: string;
	email?: string;
	exp?: number;
	error?: string;
}

/** Authenticate against INDRA and return its JWT. We never persist the password. */
export async function loginToIndra(email: string, password: string): Promise<LoginResult> {
	const { baseUrl } = loadEnv();
	if (!email || !password) return { ok: false, error: 'email and password are required' };
	let r: Response;
	try {
		r = await fetch(`${baseUrl}/login`, {
			method: 'POST',
			headers: { 'Content-Type': 'application/json' },
			body: JSON.stringify({ email, password }),
			signal: AbortSignal.timeout(REQUEST_TIMEOUT_MS)
		});
	} catch (e) {
		return { ok: false, error: `could not reach INDRA: ${e instanceof Error ? e.message : String(e)}` };
	}
	// INDRA returns the JWT in a Set-Cookie (access_token_cookie), not the body.
	const setCookies = r.headers.getSetCookie?.() ?? [];
	let jwt = '';
	for (const c of setCookies) {
		const m = c.match(/^access_token_cookie=([^;]+)/);
		if (m) {
			jwt = m[1];
			break;
		}
	}
	if (!r.ok || !jwt) {
		let reason = `login failed (HTTP ${r.status})`;
		try {
			const body = await r.json();
			if (body?.reason) reason = String(body.reason);
		} catch {
			/* keep default */
		}
		return { ok: false, error: r.status === 401 ? 'invalid email or password' : reason };
	}
	const id = decodeJwtIdentity(jwt);
	return { ok: true, jwt, email: id?.email ?? email, exp: id?.exp };
}
