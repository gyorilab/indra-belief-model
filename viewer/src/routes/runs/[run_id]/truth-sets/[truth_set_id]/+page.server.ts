import { error } from '@sveltejs/kit';
import { getTruthSetOverlapDetail } from '$lib/db';
import { RUN_ID_RE } from '$lib/server/runCohortContract';
import { assertNoActiveWriter } from '$lib/server/writerReadGuard';
import type { PageServerLoad } from './$types';

const STEP_KIND_RE = /^[A-Za-z0-9_:-]{1,128}$/;
const KNOWN_STEP_KINDS = new Set([
	'aggregate',
	'parse_claim',
	'build_context',
	'substrate_route',
	'subject_role_probe',
	'object_role_probe',
	'relation_axis_probe',
	'scope_probe',
	'grounding',
	'adjudicate'
]);

export const load: PageServerLoad = async ({ params, url }) => {
	if (!RUN_ID_RE.test(params.run_id)) {
		throw error(400, 'invalid run_id: must be 32 hex chars');
	}
	if (!params.truth_set_id || params.truth_set_id.length > 256) {
		throw error(400, 'invalid truth_set_id');
	}
	const stepKind = url.searchParams.get('step_kind') ?? 'aggregate';
	if (!STEP_KIND_RE.test(stepKind) || !KNOWN_STEP_KINDS.has(stepKind)) {
		throw error(400, 'invalid step_kind');
	}
	assertNoActiveWriter();
	const detail = await getTruthSetOverlapDetail(params.run_id, params.truth_set_id, stepKind);
	if (!detail) {
		throw error(404, `truth_set ${params.truth_set_id} was not found for run ${params.run_id}`);
	}
	return { detail };
};
