import type { LayoutServerLoad } from './$types';

export const load: LayoutServerLoad = async ({ params }) => {
	// The JSONL exports carry no probe traces or truth labels, so there are no
	// cohort / repairs / truth-set lenses here.
	return { run_id: params.run_id ?? '' };
};
