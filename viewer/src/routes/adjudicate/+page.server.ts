import { fail, redirect } from '@sveltejs/kit';
import {
	appendLabel,
	annotatorsFromMeta,
	hasQueue,
	loadMeta,
	nextBlinded,
	progress,
	revealPair
} from '$lib/data/adjudicate';
import type { Actions, PageServerLoad } from './$types';

const VERDICTS = ['correct', 'incorrect', 'abstain'];

export const load: PageServerLoad = async ({ url }) => {
	const annotator = url.searchParams.get('annotator') || 'ann1';
	const meta = loadMeta();
	const next = nextBlinded(annotator);
	// only the blinded item leaves the server — neither model's call is sent yet
	return {
		annotator,
		annotators: annotatorsFromMeta(),
		ready: hasQueue(),
		models: meta ? { a: meta.model_a, b: meta.model_b } : null,
		progress: progress(annotator),
		item: next?.blinded ?? null
	};
};

export const actions: Actions = {
	// Phase 1 → record the blinded verdict, THEN reveal both models' calls.
	commit: async ({ request }) => {
		const fd = await request.formData();
		const item_id = String(fd.get('item_id') ?? '');
		const human_verdict = String(fd.get('human_verdict') ?? '');
		if (!item_id || !VERDICTS.includes(human_verdict)) {
			return fail(400, { error: 'pick your verdict first' });
		}
		const pair = revealPair(item_id);
		if (!pair) return fail(404, { error: 'item not found' });
		return { revealed: true, item_id, human_verdict, a: pair.a, b: pair.b, gold: pair.gold };
	},

	// Phase 2 → append the adjudication, advance.
	submit: async ({ request, url }) => {
		const annotator = url.searchParams.get('annotator') || 'ann1';
		const fd = await request.formData();
		const ok = appendLabel(annotator, {
			item_id: String(fd.get('item_id') ?? ''),
			human_verdict: String(fd.get('human_verdict') ?? ''),
			reasoning_a: (fd.get('reasoning_a') as string) || null,
			reasoning_b: (fd.get('reasoning_b') as string) || null,
			ambiguous: fd.get('ambiguous') === 'on',
			notes: (fd.get('notes') as string) || ''
		});
		if (!ok) return fail(400, { error: 'could not save adjudication' });
		throw redirect(303, `/adjudicate?annotator=${encodeURIComponent(annotator)}`);
	}
};
