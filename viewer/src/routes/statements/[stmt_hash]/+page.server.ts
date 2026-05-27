import { error } from '@sveltejs/kit';
import { getProbeAttribution, getStatementDetail } from '$lib/db';
import { assertNoActiveWriter } from '$lib/server/writerReadGuard';
import type { PageServerLoad } from './$types';

// stmt_hash is INDRA's `Statement.get_hash(shallow=True)` — 16 hex nibbles
// (see corpus/loader.py::_hex). Reject anything else at the gate as a
// defense-in-depth (SQL escaping in db.ts is the inner layer).
const STMT_HASH_RE = /^[a-f0-9]{16}$/i;
const RUN_ID_RE = /^[a-f0-9]{32}$/i;

export const load: PageServerLoad = async ({ params, url }) => {
	if (!STMT_HASH_RE.test(params.stmt_hash)) {
		throw error(400, `invalid stmt_hash: must be 16 hex chars`);
	}
	const runParam = url.searchParams.get('run_id');
	if (runParam && !RUN_ID_RE.test(runParam)) {
		throw error(400, `invalid run_id: must be 32 hex chars`);
	}
	const compareRunParam = url.searchParams.get('compare_run_id');
	if (compareRunParam && !RUN_ID_RE.test(compareRunParam)) {
		throw error(400, `invalid compare_run_id: must be 32 hex chars`);
	}
	assertNoActiveWriter();
	const detail = await getStatementDetail(params.stmt_hash, runParam);
	if (!detail) {
		throw error(404, `statement ${params.stmt_hash} not found`);
	}

	const compare_detail = compareRunParam
		? await getStatementDetail(params.stmt_hash, compareRunParam)
		: null;
	if (compareRunParam && compare_detail?.selected_run_id !== compareRunParam) {
		throw error(404, `statement ${params.stmt_hash} has no scorer rows for compare_run_id ${compareRunParam}`);
	}

	let probes: Awaited<ReturnType<typeof getProbeAttribution>> = [];
	if (detail.selected_run_id && detail.selected_architecture === 'decomposed') {
		probes = await getProbeAttribution(detail.selected_run_id, params.stmt_hash);
	}

	return { detail, compare_detail, probes };
};
