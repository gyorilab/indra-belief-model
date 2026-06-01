import { getRuns, type RunSummary } from '$lib/data/queries';
import type { PageServerLoad } from './$types';

export const load: PageServerLoad = async () => {
	const runs: RunSummary[] = getRuns();
	return { runs };
};
