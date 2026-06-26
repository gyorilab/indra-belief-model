/**
 * Auth gate for the whole viewer. Every request: read + verify the sealed session
 * cookie into `locals`. If it resolves to no user and the request targets a real
 * route other than the public ones (/login, /logout), redirect to /login.
 *
 * Gating on `event.route.id` (non-null) means static assets and the SvelteKit
 * runtime (route.id === null) are never redirected — only pages/actions/endpoints
 * are protected, so the login page can still load its own assets.
 */
import { error, redirect, type Handle } from '@sveltejs/kit';
import { SESSION_COOKIE, readSession } from '$lib/server/session';

const PUBLIC_ROUTES = new Set(['/login', '/logout']);
const UNSAFE_METHODS = new Set(['POST', 'PUT', 'PATCH', 'DELETE']);

export const handle: Handle = async ({ event, resolve }) => {
	const session = readSession(event.cookies.get(SESSION_COOKIE));
	event.locals.session = session;
	event.locals.user = session ? { email: session.email } : null;

	const routeId = event.route.id;

	// CSRF for the PUBLIC auth actions (/login, /logout): they need no pre-existing
	// cookie, so SvelteKit's sameSite cookie defense doesn't cover them — and its
	// built-in Origin check is compiled out under `vite dev`. Enforce same-origin
	// here ourselves, in every run mode. (Gated routes are already safe: their
	// session cookie is sameSite=lax and isn't sent on a cross-site request.)
	if (UNSAFE_METHODS.has(event.request.method) && routeId && PUBLIC_ROUTES.has(routeId)) {
		const origin = event.request.headers.get('origin');
		if (origin !== event.url.origin) throw error(403, 'cross-site request forbidden');
	}

	if (!session && routeId && !PUBLIC_ROUTES.has(routeId)) {
		const dest = event.url.pathname + event.url.search;
		throw redirect(303, `/login?redirectTo=${encodeURIComponent(dest)}`);
	}

	return resolve(event);
};
