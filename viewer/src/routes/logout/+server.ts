/** POST /logout — clear the session cookie and return to the login page. POST
 *  only (SvelteKit's origin check guards it); public route in hooks.server.ts. */
import { redirect } from '@sveltejs/kit';
import type { RequestHandler } from './$types';
import { clearSession } from '$lib/server/session';

export const POST: RequestHandler = ({ cookies, url }) => {
	clearSession(cookies, url);
	throw redirect(303, '/login');
};
