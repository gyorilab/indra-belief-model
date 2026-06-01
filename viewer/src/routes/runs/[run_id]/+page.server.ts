import { error } from '@sveltejs/kit';
import {
	getOverview,
	getResidualDistribution,
	getValidity,
	type RunSummary
} from '$lib/data/queries';
import type { PageServerLoad } from './$types';

export const load: PageServerLoad = async ({ params }) => {
	const validity = getValidity(params.run_id);
	if (!validity) throw error(404, `run_id ${params.run_id} not found`);

	const residual = getResidualDistribution(params.run_id);

	// Run-level metadata (model, dates, counts, bucket mix) lives on the
	// RunSummary; resolve it from the registry by id/prefix.
	const overview = getOverview();
	const run: RunSummary =
		overview.runs.find((r) => r.run_id === validity.run_id) ??
		({
			run_id: validity.run_id,
			model: validity.model,
			status: null,
			generated_date: null,
			n_statements: 0,
			n_evidences: 0,
			bucket_counts: {}
		} satisfies RunSummary);

	return { run, validity, residual };
};
