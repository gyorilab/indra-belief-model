import { getStatementMatrix } from '$lib/db';
import { assertNoActiveWriter } from '$lib/server/writerReadGuard';
import type { PageServerLoad } from './$types';

export const load: PageServerLoad = async () => {
	assertNoActiveWriter();
	const rows = await getStatementMatrix();
	return { rows };
};
