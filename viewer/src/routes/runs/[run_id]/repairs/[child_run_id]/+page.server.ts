import { error } from '@sveltejs/kit';
import { getRepairRerunDetail } from '$lib/db';
import { assertNoActiveWriter } from '$lib/server/writerReadGuard';
import type { PageServerLoad } from './$types';

const RUN_ID_RE = /^[a-f0-9]{32}$/i;

function nonNegativeInt(value: string | null): number {
	if (!value) return 0;
	const n = Number(value);
	return Number.isInteger(n) && n >= 0 ? n : 0;
}

function positiveInt(value: string | null, fallback: number): number {
	if (!value) return fallback;
	const n = Number(value);
	return Number.isInteger(n) && n > 0 ? n : fallback;
}

export const load: PageServerLoad = async ({ params, url }) => {
	if (!RUN_ID_RE.test(params.run_id)) {
		throw error(400, 'invalid run_id: must be 32 hex chars');
	}
	if (!RUN_ID_RE.test(params.child_run_id)) {
		throw error(400, 'invalid child_run_id: must be 32 hex chars');
	}
	assertNoActiveWriter();
	const detail = await getRepairRerunDetail(params.run_id, params.child_run_id, {
		offset: nonNegativeInt(url.searchParams.get('offset')),
		limit: positiveInt(url.searchParams.get('limit'), 100)
	});
	if (!detail) throw error(404, `repair child ${params.child_run_id} not found for parent ${params.run_id}`);
	return { detail };
};
