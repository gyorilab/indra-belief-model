import { frontier, generalization } from '$lib/data/queries';
import { loadBeliefComparison } from '$lib/server/belief-comparison';
import { loadPaperMethodLandscape } from '$lib/server/paper-method-landscape';
import type { PageServerLoad } from './$types';

export const load: PageServerLoad = async ({ url }) => {
	const substrate = url.searchParams.get('substrate');
	const f = frontier(substrate);
	const gen = generalization();
	const beliefComparison = loadBeliefComparison();
	const paperMethodLandscape = loadPaperMethodLandscape();
	return { frontier: f, generalization: gen, beliefComparison, paperMethodLandscape };
};
