/**
 * /curate — sample a random (statement, evidence) pair from the live INDRA DB and
 * let a signed-in curator submit a curation back to it. Human-in-the-loop only:
 * nothing is auto-judged. The whole viewer is behind auth (hooks.server.ts), so a
 * request here always carries an authenticated session; the curation is submitted
 * under that curator's own INDRA JWT (locals.session.jwt), never a shared key.
 *
 * Verified live contracts:
 *   sample → GET  db.indra.bio/statements/from_agents (public)
 *   submit → POST db.indra.bio/curation/submit/<matches_hash> (curator JWT cookie)
 */
import { fail } from '@sveltejs/kit';
import type { PageServerLoad, Actions } from './$types';
import { sampleEvidence, submitCuration } from '$lib/server/indra';
import { indraBaseUrl } from '$lib/server/session';

export const load: PageServerLoad = async () => {
	const dbHost = indraBaseUrl();
	try {
		return { dbHost, sample: await sampleEvidence(), sampleError: null };
	} catch (e) {
		return { dbHost, sample: null, sampleError: e instanceof Error ? e.message : String(e) };
	}
};

export const actions: Actions = {
	/** Draw a fresh random sample (used by "sample another" + post-submit "next"). */
	sample: async () => {
		try {
			return { sampled: await sampleEvidence() };
		} catch (e) {
			return fail(502, { sampleError: e instanceof Error ? e.message : String(e) });
		}
	},

	/** Submit the signed-in curator's curation to the live INDRA DB under their JWT. */
	submit: async ({ request, locals }) => {
		if (!locals.session) return fail(401, { submitError: 'your session has expired — sign in again' });
		const fd = await request.formData();
		const args = {
			matchesHash: String(fd.get('matches_hash') ?? ''),
			sourceHash: String(fd.get('source_hash') ?? ''),
			tag: String(fd.get('tag') ?? ''),
			text: String(fd.get('text') ?? ''),
			jwt: locals.session.jwt
		};
		const res = await submitCuration(args);
		if (!res.ok) {
			return fail(res.status ?? 400, { submitError: res.error ?? 'submission failed', tag: args.tag });
		}
		return {
			submitted: {
				id: res.id ?? null,
				result: res.result ?? 'success',
				tag: args.tag,
				matchesHash: args.matchesHash,
				sourceHash: args.sourceHash
			}
		};
	}
};
