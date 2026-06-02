import { fail, redirect } from '@sveltejs/kit';
import {
	appendLabel,
	loadQueueMeta,
	nextBlinded,
	progress,
	reveal,
	listPasses
} from '$lib/data/review';
import type { Actions, PageServerLoad } from './$types';

const AXIS_A = ['faithful', 'unfaithful', 'no_text', 'cant_tell'];
const AXIS_B = ['correct', 'incorrect', 'abstain'];

function resolvePass(p: string | null): string {
	const passes = listPasses();
	if (p && passes.includes(p)) return p;
	return passes[0] ?? 'rough';
}

export const load: PageServerLoad = async ({ url }) => {
	const pass = resolvePass(url.searchParams.get('pass'));
	const annotator = url.searchParams.get('annotator') || 'ann1';
	const meta = loadQueueMeta(pass);
	const next = nextBlinded(pass, annotator);
	// NOTE: only the blinded item leaves the server here — no verdict/reasoning.
	return {
		pass,
		annotator,
		passes: listPasses(),
		annotators: meta
			? [...new Set([...(meta as any).params?.annotators ?? [], (meta as any).params?.double_annotator].filter(Boolean))]
			: [annotator],
		run: meta ? { run_id: meta.run_id, model: meta.model } : null,
		progress: progress(pass, annotator),
		item: next?.blinded ?? null
	};
};

export const actions: Actions = {
	// Phase 1 → reveal the model's call AFTER the human verdict is captured.
	commit: async ({ request, url }) => {
		const pass = resolvePass(url.searchParams.get('pass'));
		const fd = await request.formData();
		const item_id = String(fd.get('item_id') ?? '');
		const axis_a = String(fd.get('axis_a_faithful') ?? '');
		const axis_b = String(fd.get('axis_b_human_verdict') ?? '');
		if (!item_id || !AXIS_A.includes(axis_a) || !AXIS_B.includes(axis_b)) {
			return fail(400, { error: 'pick faithfulness + your verdict' });
		}
		const model = reveal(pass, item_id);
		if (!model) return fail(404, { error: 'item not found' });
		return { revealed: true, item_id, axis_a, axis_b, model };
	},

	// Phase 2 → append the full multi-axis label, advance to the next item.
	submit: async ({ request, url }) => {
		const pass = resolvePass(url.searchParams.get('pass'));
		const annotator = url.searchParams.get('annotator') || 'ann1';
		const fd = await request.formData();
		const ok = appendLabel(pass, annotator, {
			item_id: String(fd.get('item_id') ?? ''),
			axis_a_faithful: String(fd.get('axis_a_faithful') ?? ''),
			axis_b_human_verdict: String(fd.get('axis_b_human_verdict') ?? ''),
			axis_c_reasoning: (fd.get('axis_c_reasoning') as string) || null,
			axis_d_failure: (fd.get('axis_d_failure') as string) || null,
			notes: (fd.get('notes') as string) || ''
		});
		if (!ok) return fail(400, { error: 'could not save label' });
		throw redirect(303, `/review?pass=${pass}&annotator=${encodeURIComponent(annotator)}`);
	}
};
