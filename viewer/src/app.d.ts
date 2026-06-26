// See https://svelte.dev/docs/kit/types#app.d.ts
// for information about these interfaces
declare global {
	namespace App {
		interface Error {
			code?: string;
			message: string;
		}
		interface Locals {
			/** The authenticated curator, or null. Derived from the sealed session
			 *  cookie on every request in hooks.server.ts. */
			user: { email: string } | null;
			/** Full session incl. the INDRA JWT — server-only, forwarded on writes.
			 *  Never returned to the browser (only `user` is). */
			session: { jwt: string; email: string; exp: number } | null;
		}
		// interface PageData {}
		// interface PageState {}
		// interface Platform {}
	}
}

export {};
