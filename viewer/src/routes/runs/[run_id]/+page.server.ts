import { error } from '@sveltejs/kit';
import {
	getOverview,
	getResidualDistribution,
	getRunCalibration,
	getValidity,
	type RunSummary,
	type Tier
} from '$lib/data/queries';
import type { PageServerLoad } from './$types';

export const load: PageServerLoad = async ({ params, url }) => {
	const validity = getValidity(params.run_id);
	if (!validity) throw error(404, `run_id ${params.run_id} not found`);

	const residual = getResidualDistribution(params.run_id);
	const calibration = getRunCalibration(params.run_id);

	// The one reactive lever: Tier toggle ?tier=ev|stmt. Default ev (Tier-1
	// per-evidence — the realized score; Tier-2 is the rolled-up belief).
	const tierParam = url.searchParams.get('tier');
	const tier: Tier = tierParam === 'stmt' ? 'stmt' : 'ev';

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
			export_schema_version: null,
			source_run: null,
			provenance: null,
			n_statements: 0,
			n_evidences: 0,
			bucket_counts: {},
			cost: null,
			soft_calibration: undefined
		} satisfies RunSummary);

	return { run, validity, residual, calibration, tier };
};
