/**
 * /login — authenticate a curator against INDRA and seal the returned JWT into a
 * session cookie. Public route (allow-listed in hooks.server.ts).
 */
import { fail, redirect } from '@sveltejs/kit';
import type { PageServerLoad, Actions } from './$types';
import {
	SESSION_COOKIE,
	loginThrottled,
	loginToIndra,
	sealSession,
	sessionCookieOptions
} from '$lib/server/session';

/** Only permit same-origin relative redirects — never an absolute/protocol URL.
 *  Prefix checks alone are insufficient: browsers strip TAB/CR/LF from URLs, so
 *  "/\t/evil.com" collapses to "//evil.com" (scheme-relative → off-origin) and
 *  CR/LF would corrupt the Location header. So: reject all control chars, then
 *  resolve against a sentinel origin and require the result to stay on it. */
function safeRedirect(target: string | null): string {
	if (!target || target[0] !== '/') return '/';
	// eslint-disable-next-line no-control-regex
	if (/[\u0000-\u001f\u007f]/.test(target)) return '/';
	try {
		const SENTINEL = 'http://localhost.invalid';
		const u = new URL(target, SENTINEL);
		if (u.origin !== SENTINEL) return '/';
		return u.pathname + u.search + u.hash;
	} catch {
		return '/';
	}
}

export const load: PageServerLoad = async ({ locals, url }) => {
	// already authenticated → bounce to where they were headed
	if (locals.user) throw redirect(303, safeRedirect(url.searchParams.get('redirectTo')));
	return { redirectTo: url.searchParams.get('redirectTo') ?? '' };
};

export const actions: Actions = {
	default: async ({ request, cookies, url, getClientAddress }) => {
		const fd = await request.formData();
		const email = String(fd.get('email') ?? '').trim();
		const password = String(fd.get('password') ?? '');
		const redirectTo = safeRedirect(String(fd.get('redirectTo') ?? '') || null);

		if (!email || !password) {
			return fail(400, { error: 'email and password are required', email });
		}

		if (loginThrottled(`${getClientAddress()}|${email.toLowerCase()}`)) {
			return fail(429, { error: 'too many attempts — wait a few minutes and try again', email });
		}

		const res = await loginToIndra(email, password);
		if (!res.ok || !res.jwt) {
			return fail(401, { error: res.error ?? 'login failed', email });
		}

		const maxAge = res.exp ? Math.max(0, res.exp - Math.floor(Date.now() / 1000)) : 60 * 60 * 24 * 30;
		cookies.set(SESSION_COOKIE, sealSession(res.jwt), sessionCookieOptions(url, maxAge));
		throw redirect(303, redirectTo);
	}
};
