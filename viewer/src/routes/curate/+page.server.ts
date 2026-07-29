/**
 * /curate — sample a random (statement, evidence) pair from the live INDRA DB and
 * let a signed-in curator submit a curation back to it. Human-in-the-loop only:
 * nothing is auto-judged. The whole viewer is behind auth (hooks.server.ts), so a
 * request here always carries an authenticated session; the curation is submitted
 * under that curator's own INDRA JWT (locals.session.jwt), never a shared key.
 *
 * Verified live contracts:
 *   sample → uniform-random unseen line from the tracked 5k reservoir manifest,
 *            materialized unbiased by matches_hash via
 *            POST db.indra.bio/statements/from_hashes (public)
 *   submit → POST db.indra.bio/curation/submit/<matches_hash> (curator JWT cookie)
 */
import { fail } from '@sveltejs/kit';
import type { PageServerLoad, Actions } from './$types';
import { getCuratorContext, noteCuratorSubmission, sampleEvidence, submitCuration } from '$lib/server/indra';
import { datasetsForClient } from '$lib/server/datasets';
import { indraBaseUrl } from '$lib/server/session';
import { exactPairKey } from '$lib/server/curation-history';
import { CURATION_TAGS } from '$lib/data/curation';
import {
	claimDrawReservation,
	commitDrawReservation,
	releaseDrawClaim,
	submissionFailureIsDefinitive
} from '$lib/server/curation-draw-ledger';

export const load: PageServerLoad = async ({ locals, url }) => {
	const dbHost = indraBaseUrl();
	const datasets = datasetsForClient();
	const datasetId = url.searchParams.get('dataset');
	const context = locals.session
		? await getCuratorContext(locals.session.jwt, locals.session.email)
		: {
				stats: { status: 'unavailable' as const, total: 0, correct: 0, incorrect: 0, reason: 'not signed in' },
				curatedKeys: new Set<string>()
			};
	const stats = context.stats;
	if (stats.status === 'unavailable') {
		return { dbHost, stats, datasets, sample: null, sampleError: stats.reason ?? 'curation history unavailable' };
	}
	try {
		return {
			dbHost,
			stats,
			datasets,
			sample: await sampleEvidence(datasetId, context.curatedKeys, locals.session?.email ?? ''),
			sampleError: null
		};
	} catch (e) {
		return { dbHost, stats, datasets, sample: null, sampleError: e instanceof Error ? e.message : String(e) };
	}
};

export const actions: Actions = {
	/** Draw a fresh random sample from the chosen dataset (used by "sample another",
	 *  the post-submit auto-advance, and a dataset switch in the selector rail). */
	sample: async ({ request, locals }) => {
		if (!locals.session) return fail(401, { sampleError: 'your session has expired — sign in again' });
		const fd = await request.formData();
		const datasetId = String(fd.get('dataset') ?? '') || null;
		const context = await getCuratorContext(locals.session.jwt, locals.session.email);
		if (context.stats.status === 'unavailable') {
			return fail(503, {
				sampleError: context.stats.reason ?? 'cannot load curation history; refusing to sample with replacement'
			});
		}
		try {
			return { sampled: await sampleEvidence(datasetId, context.curatedKeys, locals.session.email) };
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
			dataset: String(fd.get('dataset') ?? ''),
			tag: String(fd.get('tag') ?? ''),
			text: String(fd.get('text') ?? ''),
			jwt: locals.session.jwt
		};
		const drawToken = String(fd.get('draw_token') ?? '');
		const pairKey = exactPairKey(args.matchesHash, args.sourceHash);
		if (!datasetsForClient().some((dataset) => dataset.id === args.dataset)) {
			return fail(400, { submitError: 'invalid curation dataset', tag: args.tag });
		}
		if (!CURATION_TAGS.includes(args.tag)) {
			return fail(400, { submitError: 'invalid curation tag', tag: args.tag });
		}
		if (
			!pairKey ||
			!claimDrawReservation(locals.session.email, args.dataset, pairKey, drawToken)
		) {
			return fail(409, {
				submitError: 'this pair has no open draw reservation or was already submitted',
				tag: args.tag
			});
		}
		// A fresh shared-history read closes the ordinary stale-tab/process window.
		// The local exclusive claim above closes concurrent submissions on a shared
		// ledger filesystem; a multi-instance deploy must share that storage.
		const fresh = await getCuratorContext(locals.session.jwt, locals.session.email, true);
		if (fresh.stats.status === 'unavailable') {
			releaseDrawClaim(locals.session.email, args.dataset, pairKey);
			return fail(503, {
				submitError: fresh.stats.reason ?? 'cannot verify that this pair is still uncurated',
				tag: args.tag
			});
		}
		if (fresh.curatedKeys.has(pairKey)) {
			commitDrawReservation(locals.session.email, args.dataset, pairKey, drawToken);
			return fail(409, { submitError: 'this exact pair was already curated', tag: args.tag });
		}
		const res = await submitCuration(args);
		if (!res.ok) {
			if (submissionFailureIsDefinitive(res.status)) {
				releaseDrawClaim(locals.session.email, args.dataset, pairKey);
				return fail(res.status ?? 400, {
					submitError: res.error ?? 'submission was rejected',
					tag: args.tag
				});
			}
			// A timeout/network/5xx response can arrive after INDRA committed. Re-read
			// shared history once; if the pair is present, close the reservation as a
			// reconciled success. Otherwise keep the claim fail-closed—never retry an
			// upstream write whose outcome is unknown.
			const reconciled = await getCuratorContext(locals.session.jwt, locals.session.email, true);
			if (reconciled.stats.status === 'available' && reconciled.curatedKeys.has(pairKey)) {
				commitDrawReservation(locals.session.email, args.dataset, pairKey, drawToken);
				return {
					submitted: {
						id: null,
						result: 'confirmed in INDRA history after an ambiguous response',
						tag: args.tag,
						matchesHash: args.matchesHash,
						sourceHash: args.sourceHash
					}
				};
			}
			return fail(502, {
				submitError:
					'INDRA may have accepted this curation; the pair is locked pending history/operator reconciliation',
				tag: args.tag
			});
		}
		try {
			commitDrawReservation(locals.session.email, args.dataset, pairKey, drawToken);
		} catch (error) {
			// INDRA already accepted the curation. Keep the exclusive claim in place so
			// a retry cannot duplicate it, and surface the ledger repair in server logs.
			console.error('could not mark curation draw committed', error);
		}
		noteCuratorSubmission(locals.session.email, args.tag, args.matchesHash, args.sourceHash);
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
