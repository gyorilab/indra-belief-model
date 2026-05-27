import { error, json } from '@sveltejs/kit';
import { recordRepairRerunChild } from '$lib/server/repairRerun';
import type { RequestHandler } from './$types';

const RUN_ID_RE = /^[a-f0-9]{32}$/i;
const ARCH_RE = /^(decomposed|monolithic)$/;
const SOURCE_DUMP_RE = /^[a-z][a-z0-9_-]{1,63}$/i;

export const POST: RequestHandler = async ({ request }) => {
	const body = (await request.json()) as Record<string, unknown>;
	const parent_run_id = body.parent_run_id;
	const child_run_id = body.child_run_id;
	const architecture = body.architecture;
	const source_dump_id = body.source_dump_id;
	if (typeof parent_run_id !== 'string' || !RUN_ID_RE.test(parent_run_id)) {
		throw error(400, 'parent_run_id must be 32 hex chars');
	}
	if (typeof child_run_id !== 'string' || !RUN_ID_RE.test(child_run_id)) {
		throw error(400, 'child_run_id must be 32 hex chars');
	}
	if (typeof architecture !== 'string' || !ARCH_RE.test(architecture)) {
		throw error(400, 'architecture must be decomposed or monolithic');
	}
	if (typeof source_dump_id !== 'string' || !SOURCE_DUMP_RE.test(source_dump_id)) {
		throw error(400, 'source_dump_id must be a safe source dump token');
	}
	const correction_ids = Array.isArray(body.correction_ids)
		? body.correction_ids.map((id) => Number(id)).filter((id) => Number.isInteger(id) && id > 0)
		: [];
	if (correction_ids.length === 0) {
		throw error(400, 'correction_ids required');
	}
	try {
		const result = await recordRepairRerunChild({
			parent_run_id,
			child_run_id,
			architecture: architecture as 'decomposed' | 'monolithic',
			source_dump_id,
			correction_ids
		});
		return json({ ok: true, ...result });
	} catch (e) {
		const message = (e as Error).message || 'repair rerun completion failed';
		const status = message.includes('DuckDB writer') || message.includes('paired workflow') ? 409 : 400;
		throw error(status, message);
	}
};
