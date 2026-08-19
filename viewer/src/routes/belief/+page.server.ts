import { loadBeliefVsIndra } from '$lib/server/belief-vs-indra';
import type { PageServerLoad } from './$types';

export const load: PageServerLoad = async () => ({ beliefVsIndra: loadBeliefVsIndra() });
