import { error } from '@sveltejs/kit';
import { getStatementDetail } from '$lib/data/queries';
import type { PageServerLoad } from './$types';

// stmt_hash is INDRA's `Statement.get_hash(shallow=True)` — 16 hex nibbles
// (see corpus/loader.py::_hex). Reject anything else at the gate.
const STMT_HASH_RE = /^[a-f0-9]{16}$/i;

export const load: PageServerLoad = async ({ params }) => {
	if (!STMT_HASH_RE.test(params.stmt_hash)) {
		throw error(400, `invalid stmt_hash: must be 16 hex chars`);
	}
	const detail = getStatementDetail(params.stmt_hash);
	if (!detail) {
		throw error(404, `statement ${params.stmt_hash} not found`);
	}
	return { detail };
};
