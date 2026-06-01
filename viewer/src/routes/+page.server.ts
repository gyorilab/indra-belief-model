import {
	getOverview,
	getFocus,
	getValidity,
	getResidualDistribution,
	getFindings
} from '$lib/data/queries';
import type { PageServerLoad } from './$types';

const STMT_HASH_RE = /^[a-f0-9]{16}$/i;

export const load: PageServerLoad = ({ url }) => {
	const focusParam = url.searchParams.get('focus');
	const focusHash = focusParam && STMT_HASH_RE.test(focusParam) ? focusParam : undefined;

	return {
		overview: getOverview(),
		focus: getFocus(undefined, focusHash),
		validity: getValidity(),
		residuals: getResidualDistribution(),
		findings: getFindings()
	};
};
