import { error } from '@sveltejs/kit';
import { getPairedWorkbench, type PairedWorkbench } from '$lib/db';
import { reconcileStalePairedWorkflowState } from '$lib/server/pairedState';
import { assertNoActiveWriter } from '$lib/server/writerReadGuard';
import type { PageServerLoad } from './$types';

const PAIR_ID_RE = /^[A-Za-z0-9][A-Za-z0-9_.-]{1,120}$/;
const FOLD_COOKIE = 'indra_pair_exemplar_folds_v2';
const MAX_SERVER_FOLD_KEYS = 80;

type StoredFoldState = { open?: unknown };

function parseInitialFoldKeys(raw: string | undefined): string[] {
	if (!raw) return [];
	try {
		const parsed = JSON.parse(decodeURIComponent(raw)) as StoredFoldState;
		if (!Array.isArray(parsed.open)) return [];
		return parsed.open
			.filter((key): key is string => typeof key === 'string' && key.length > 0 && key.length < 500)
			.slice(0, MAX_SERVER_FOLD_KEYS);
	} catch {
		return [];
	}
}

export const load: PageServerLoad = async ({ params, cookies }) => {
	if (!PAIR_ID_RE.test(params.pair_id)) {
		throw error(400, 'invalid pair_id: must be a safe paired_run_group_id token');
	}
	const initial_open_pattern_keys = parseInitialFoldKeys(cookies.get(FOLD_COOKIE));
	assertNoActiveWriter();
	const workflow = reconcileStalePairedWorkflowState(params.pair_id);
	const workbench = await getPairedWorkbench(params.pair_id) as PairedWorkbench | null;
	if (!workbench && !workflow) throw error(404, `paired_run_group_id ${params.pair_id} not found`);
	const emptyWorkbench: PairedWorkbench = {
		pair_id: params.pair_id,
		runs: [],
		monolithic: null,
		decomposed: null,
		overlap: null,
		comparable: null,
		resource_frontier: null,
		denominator_ledger: [],
		exemplars: {
			monolithic_wins: [],
			decomposed_wins: [],
			verdict_disagreements: [],
			mutual_failures: [],
			monolithic_only: [],
			decomposed_only: [],
			excluded_by_integrity: []
		},
		arch_conditioned: {
			monolithic_tiers: [],
			decomposed_probes: []
		},
		not_defined_reason: workflow
			? `paired workflow is ${workflow.status}; comparison is not defined until score_run rows exist for both architectures`
			: 'paired workflow state is unavailable'
	};
	return { workbench: workbench ?? emptyWorkbench, workflow, initial_open_pattern_keys };
};
