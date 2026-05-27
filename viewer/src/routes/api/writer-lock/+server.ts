import { json } from '@sveltejs/kit';
import { activePublicWriterLock } from '$lib/server/pairedState';
import type { RequestHandler } from './$types';

export const GET: RequestHandler = async () => {
	return json({
		ok: true,
		writerLock: activePublicWriterLock()
	});
};
