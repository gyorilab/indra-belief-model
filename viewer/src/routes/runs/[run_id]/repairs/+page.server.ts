import { error } from '@sveltejs/kit';
import { getRunRepairBacklog } from '$lib/db';
import type { PublicWriterLockState } from '$lib/server/pairedState';
import { assertNoActiveWriter } from '$lib/server/writerReadGuard';
import type { PageServerLoad } from './$types';

const RUN_ID_RE = /^[a-f0-9]{32}$/i;

function nonNegativeInt(value: string | null): number {
	if (!value) return 0;
	const n = Number(value);
	return Number.isInteger(n) && n >= 0 ? n : 0;
}

export const load: PageServerLoad = async ({ params, url }) => {
	if (!RUN_ID_RE.test(params.run_id)) {
		throw error(400, 'invalid run_id: must be 32 hex chars');
	}
	assertNoActiveWriter();
	const backlog = await getRunRepairBacklog(params.run_id, {
		recoveryOffset: nonNegativeInt(url.searchParams.get('recovery_offset'))
	});
	if (!backlog) throw error(404, `run_id ${params.run_id} not found`);
	return {
		backlog,
		writerLock: null as PublicWriterLockState | null
	};
};
