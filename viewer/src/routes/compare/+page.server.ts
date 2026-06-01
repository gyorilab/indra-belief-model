import { compareRuns, getRuns, type Comparison, type RunSummary } from '$lib/data/queries';
import type { PageServerLoad } from './$types';

const RUN_ID_RE = /^[a-f0-9]{32}$/i;

function pickRun(runs: RunSummary[], requested: string | null): string | null {
	if (requested && RUN_ID_RE.test(requested) && runs.some((r) => r.run_id === requested)) {
		return requested;
	}
	return null;
}

export const load: PageServerLoad = async ({ url }) => {
	const runs = getRuns();

	// Default a/b to the two newest runs (getRuns returns newest-first).
	const defaultA = runs[0]?.run_id ?? null;
	const defaultB = runs[1]?.run_id ?? null;

	const a = pickRun(runs, url.searchParams.get('a')) ?? defaultA;
	const b = pickRun(runs, url.searchParams.get('b')) ?? defaultB;

	let comparison: Comparison | null = null;
	if (a && b && a !== b) {
		comparison = compareRuns(a, b);
	}

	return { runs, comparison, selectedA: a, selectedB: b };
};
