import { frontier } from '$lib/data/queries';
import type { PageServerLoad } from './$types';

export const load: PageServerLoad = async ({ url }) => {
	const substrate = url.searchParams.get('substrate');
	const f = frontier(substrate);
	return { frontier: f };
};
