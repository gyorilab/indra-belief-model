/**
 * /curate — sample a random (statement, evidence) pair from the live INDRA DB and
 * let a signed-in curator submit a curation back to it. Human-in-the-loop only:
 * nothing is auto-judged. The whole viewer is behind auth (hooks.server.ts), so a
 * request here always carries an authenticated session; the curation is submitted
 * under that curator's own INDRA JWT (locals.session.jwt), never a shared key.
 *
 * Verified live contracts:
 *   sample → uniform-random line from data/corpora/cogex_evidence_sample.jsonl,
 *            materialized unbiased by matches_hash via
 *            POST db.indra.bio/statements/from_hashes (public)
 *   submit → POST db.indra.bio/curation/submit/<matches_hash> (curator JWT cookie)
 */
import { fail } from '@sveltejs/kit';
import type { PageServerLoad, Actions } from './$types';
import { getCuratorStats, noteCuratorSubmission, sampleEvidence, submitCuration } from '$lib/server/indra';
import { datasetsForClient } from '$lib/server/datasets';
import { indraBaseUrl } from '$lib/server/session';

export const load: PageServerLoad = async ({ locals, url }) => {
	const dbHost = indraBaseUrl();
	const datasets = datasetsForClient();
	const datasetId = url.searchParams.get('dataset');
	const stats = locals.session
		? await getCuratorStats(locals.session.jwt, locals.session.email)
		: { status: 'unavailable' as const, total: 0, correct: 0, incorrect: 0, reason: 'not signed in' };
	try {
		return { dbHost, stats, datasets, sample: await sampleEvidence(datasetId), sampleError: null };
	} catch (e) {
		return { dbHost, stats, datasets, sample: null, sampleError: e instanceof Error ? e.message : String(e) };
	}
};

export const actions: Actions = {
	/** Draw a fresh random sample from the chosen dataset (used by "sample another",
	 *  the post-submit auto-advance, and a dataset switch in the selector rail). */
	sample: async ({ request }) => {
		const fd = await request.formData();
		const datasetId = String(fd.get('dataset') ?? '') || null;
		try {
			return { sampled: await sampleEvidence(datasetId) };
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
		noteCuratorSubmission(locals.session.email, args.tag);
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
