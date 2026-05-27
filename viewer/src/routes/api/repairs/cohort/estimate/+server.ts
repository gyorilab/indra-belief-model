import { json } from '@sveltejs/kit';
import {
	estimateRepairCohort,
	RepairCohortConflictError,
	RepairCohortHttpError,
	RepairCohortInputError
} from '$lib/server/repairBacklog';
import {
	cohortFiltersFromRecord,
	expectedRunStatus,
	RUN_ID_RE
} from '$lib/server/runCohortContract';
import type { RequestHandler } from './$types';

export const POST: RequestHandler = async ({ request }) => {
	try {
		const body = (await request.json().catch(() => {
			throw new RepairCohortInputError('invalid_json', 'request body must be JSON');
		})) as Record<string, unknown>;
		const run_id = body.run_id;
		if (typeof run_id !== 'string' || !RUN_ID_RE.test(run_id)) {
			throw new RepairCohortInputError('invalid_run_id', 'run_id must be 32 hex chars');
		}
		const source_route = typeof body.source_route === 'string'
			? body.source_route.slice(0, 512)
			: `/runs/${run_id}/cohort`;
		const expected_run_status = expectedRunStatus(body.expected_run_status);
		const reviewer = typeof body.reviewer === 'string' ? body.reviewer.slice(0, 120) : null;
		const note = typeof body.note === 'string' ? body.note.slice(0, 1000) : null;
		const filters = cohortFiltersFromRecord(body.filters);
		if (!expected_run_status) {
			throw new RepairCohortInputError('expected_run_status_required', 'cohort repair estimate requires expected_run_status');
		}
		const result = await estimateRepairCohort({
			run_id,
			filters,
			source_route,
			expected_run_status,
			reviewer,
			note
		});
		return json({ ok: true, ...result });
	} catch (e) {
		const httpError = e as { status?: number; body?: { message?: string; code?: string }; message?: string };
		const message = httpError.body?.message || httpError.message || 'repair backlog estimate failed';
		if (e instanceof RepairCohortConflictError) {
			return json(
				{
					message,
					code: e.code,
					expected_run_status: e.expected,
					actual_run_status: e.actual
				},
				{ status: 409 }
			);
		}
		if (e instanceof RepairCohortHttpError) {
			return json({ message, code: e.code }, { status: e.status });
		}
		if (httpError.status && httpError.status >= 400 && httpError.status < 500) {
			const code = typeof httpError.body?.code === 'string' ? httpError.body.code : 'invalid_cohort_request';
			return json({ message, code }, { status: httpError.status });
		}
		return json({ message, code: 'repair_cohort_estimate_failed' }, { status: 500 });
	}
};
