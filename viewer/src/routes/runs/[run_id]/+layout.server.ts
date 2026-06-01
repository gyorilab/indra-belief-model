import type { LayoutServerLoad } from './$types';

export const load: LayoutServerLoad = async ({ params }) => {
	// The cohort / repairs / truth-set lenses were cut with the DuckDB layer;
	// the monolithic JSONL exports carry no probe traces or truth labels.
	return { run_id: params.run_id ?? '' };
};
