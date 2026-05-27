import { error, redirect } from '@sveltejs/kit';
import { getRunCohort } from '$lib/db';
import {
	cohortFiltersFromSearchParams,
	RUN_ID_RE
} from '$lib/server/runCohortContract';
import { assertNoActiveWriter } from '$lib/server/writerReadGuard';
import type { PageServerLoad } from './$types';

export const load: PageServerLoad = async ({ params, url }) => {
	if (!RUN_ID_RE.test(params.run_id)) {
		throw error(400, 'invalid run_id: must be 32 hex chars');
	}
	const filters = cohortFiltersFromSearchParams(url.searchParams);
	assertNoActiveWriter();
	const cohort = await getRunCohort(params.run_id, filters);
	if (!cohort) throw error(404, `run_id ${params.run_id} not found`);
	const canonicalSnapshot = cohort.filters.trace_snapshot ?? null;
	if ((filters.trace_state || filters.probe_coverage) && filters.trace_snapshot !== canonicalSnapshot) {
		const sp = new URLSearchParams(url.searchParams);
		sp.set('grain', 'evidence');
		if (canonicalSnapshot) sp.set('trace_snapshot', canonicalSnapshot);
		else sp.delete('trace_snapshot');
		throw redirect(302, `/runs/${params.run_id}/cohort?${sp.toString()}`);
	}
	return { cohort };
};
