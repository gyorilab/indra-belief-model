/** Root layout load — expose the authenticated curator (email only) to every page
 *  and to SiteNav. The JWT in locals.session is never returned to the browser. */
import type { LayoutServerLoad } from './$types';

export const load: LayoutServerLoad = async ({ locals }) => {
	return { user: locals.user };
};
