import { getStatementMatrix } from '$lib/data/queries';
import type { PageServerLoad } from './$types';

// The monolithic JSONL layer has no run-compare or DuckDB params; default to
// the latest run. getStatementMatrix() returns { run_id, rows } or null when
// no run has been exported yet.
export const load: PageServerLoad = async () => {
	const result = getStatementMatrix();
	if (!result) {
		return { matrix: [], run_id: null as string | null };
	}
	return { matrix: result.rows, run_id: result.run_id };
};
