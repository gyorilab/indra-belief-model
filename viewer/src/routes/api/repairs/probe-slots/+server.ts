import { json } from '@sveltejs/kit';
import { recordProbeSlotReview, PROBE_REPAIR_SLOTS } from '$lib/server/repairProbeSlots';
import { RUN_ID_RE } from '$lib/server/runCohortContract';
import type { RequestHandler } from './$types';

const PROBE_REPAIR_SLOT_SET = new Set<string>(PROBE_REPAIR_SLOTS);

export const POST: RequestHandler = async ({ request }) => {
	try {
		const body = (await request.json().catch(() => {
			throw new Error('request body must be JSON');
		})) as Record<string, unknown>;
		const run_id = body.run_id;
		if (typeof run_id !== 'string' || !RUN_ID_RE.test(run_id)) {
			return json({ message: 'run_id must be 32 hex chars', code: 'invalid_run_id' }, { status: 400 });
		}
		const correction_id = Number(body.correction_id);
		if (!Number.isInteger(correction_id) || correction_id <= 0) {
			return json({ message: 'correction_id must be a positive integer', code: 'invalid_correction_id' }, { status: 400 });
		}
		const selected_slots = Array.isArray(body.selected_slots)
			? body.selected_slots.map((slot) => String(slot))
			: [];
		const unknown = selected_slots.filter((slot) => !PROBE_REPAIR_SLOT_SET.has(slot));
		if (unknown.length > 0) {
			return json({ message: `unknown probe slot: ${unknown[0]}`, code: 'invalid_probe_slot' }, { status: 400 });
		}
		const result = await recordProbeSlotReview({
			run_id,
			correction_id,
			selected_slots,
			note: typeof body.note === 'string' ? body.note : null,
			reviewer: typeof body.reviewer === 'string' ? body.reviewer : 'viewer'
		});
		return json({ ok: true, ...result });
	} catch (e) {
		const message = (e as Error)?.message ?? 'probe slot review failed';
		const status = message.includes('busy') || message.includes('paired workflow') ? 409 : 400;
		return json({ message, code: status === 409 ? 'writer_lock_busy' : 'probe_slot_review_failed' }, { status });
	}
};
