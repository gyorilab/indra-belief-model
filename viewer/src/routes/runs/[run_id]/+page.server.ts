import { error, redirect } from '@sveltejs/kit';
import {
	connect,
	dbExists,
	getHeuristicCoverage,
	getRunNarrative,
	getTraceFidelity,
	TRACE_FIDELITY_DETAIL_LIMIT,
	type HeuristicCoverage,
	type RunNarrative,
	type TraceFidelitySummary
} from '$lib/db';
import { cleanTraceSnapshot, INVALID_TRACE_SNAPSHOT } from '$lib/traceState';
import { assertNoActiveWriter } from '$lib/server/writerReadGuard';
import type { PageServerLoad } from './$types';

const RUN_ID_RE = /^[a-f0-9]{32}$/i;
const MAX_TRACE_PAGE = 1_000_000;

function positiveIntParam(value: string | null, name: string): number {
	if (value == null || value === '') return 1;
	if (!/^[1-9]\d*$/.test(value)) {
		throw error(400, `${name} must be a positive integer`);
	}
	const parsed = Number(value);
	if (!Number.isSafeInteger(parsed) || parsed > MAX_TRACE_PAGE) {
		throw error(400, `${name} must be a positive integer no greater than ${MAX_TRACE_PAGE}`);
	}
	return parsed;
}

function traceSnapshotParam(value: string | null): string | null {
	const cleaned = cleanTraceSnapshot(value);
	if (cleaned === INVALID_TRACE_SNAPSHOT) {
		throw error(400, 'trace_snapshot must be a scorer_step timestamp');
	}
	return cleaned;
}

function runHref(
	runId: string,
	compareTo: string | undefined,
	tracePage: number,
	traceSnapshot: string | null,
	anchorDetails: boolean
): string {
	// Keep canonicalization scoped to URL params this route owns.
	const sp = new URLSearchParams();
	if (compareTo) sp.set('compare_to', compareTo);
	if (tracePage > 1) sp.set('trace_page', String(tracePage));
	if (traceSnapshot) sp.set('trace_snapshot', traceSnapshot);
	const qs = sp.toString();
	return `/runs/${runId}${qs ? `?${qs}` : ''}${anchorDetails ? '#trace-fidelity-details' : ''}`;
}

export interface RunMeta {
	run_id: string;
	scorer_version: string;
	architecture: string;
	paired_run_group_id: string | null;
	indra_version: string | null;
	model_id_default: string | null;
	started_at: string;
	status: string;
	terminated_by: string | null;
	termination_reason: string | null;
	n_stmts: number | null;
	n_evidences: number | null;
	n_scorer_steps: number | null;
	cost_estimate_usd: number | null;
	cost_actual_usd: number | null;
}

export interface AllRunsRow {
	run_id: string;
	scorer_version: string;
	architecture: string;
	started_at: string;
	status: string;
}

async function getRunMeta(run_id: string): Promise<RunMeta | null> {
	if (!dbExists()) return null;
	const con = await connect();
	try {
		// `score_run` doesn't carry n_evidences directly — count from scorer_step.
		const qr = run_id.replace(/'/g, "''");
		const reader = await con.runAndReadAll(
			`SELECT
			   sr.run_id, sr.scorer_version, sr.architecture, sr.paired_run_group_id,
			   sr.indra_version, sr.model_id_default,
			   sr.started_at::VARCHAR AS started_at, sr.status, sr.n_stmts,
			   sr.terminated_by, sr.termination_reason,
			   (SELECT COUNT(*)
			    FROM scorer_step WHERE run_id = sr.run_id) AS n_scorer_steps,
			   (SELECT COUNT(DISTINCT evidence_hash)
			    FROM scorer_step WHERE run_id = sr.run_id) AS n_evidences,
			   sr.cost_estimate_usd, sr.cost_actual_usd
			 FROM score_run sr WHERE sr.run_id='${qr}'`
		);
		const rs = reader.getRowObjects();
		if (rs.length === 0) return null;
		const r = rs[0];
		return {
			run_id: r.run_id as string,
			scorer_version: r.scorer_version as string,
			architecture: (r.architecture as string | null) ?? 'unknown',
			paired_run_group_id: (r.paired_run_group_id as string | null) ?? null,
			indra_version: (r.indra_version as string | null) ?? null,
			model_id_default: (r.model_id_default as string | null) ?? null,
			started_at: r.started_at as string,
			status: r.status as string,
			terminated_by: (r.terminated_by as string | null) ?? null,
			termination_reason: (r.termination_reason as string | null) ?? null,
			n_stmts: r.n_stmts != null ? Number(r.n_stmts) : null,
			n_evidences: r.n_evidences != null ? Number(r.n_evidences) : null,
			n_scorer_steps: r.n_scorer_steps != null ? Number(r.n_scorer_steps) : null,
			cost_estimate_usd: r.cost_estimate_usd != null ? Number(r.cost_estimate_usd) : null,
			cost_actual_usd: r.cost_actual_usd != null ? Number(r.cost_actual_usd) : null
		};
	} finally {
		con.disconnectSync?.();
	}
}

async function listSucceededRuns(): Promise<AllRunsRow[]> {
	if (!dbExists()) return [];
	const con = await connect();
	try {
		const reader = await con.runAndReadAll(
			`SELECT run_id, scorer_version, architecture, started_at::VARCHAR AS started_at, status
			 FROM score_run
			 WHERE status='succeeded'
			 ORDER BY started_at DESC`
		);
		return reader.getRowObjects().map((r) => ({
			run_id: r.run_id as string,
			scorer_version: r.scorer_version as string,
			architecture: ((r.architecture as string | null) ?? 'unknown') as string,
			started_at: r.started_at as string,
			status: r.status as string
		}));
	} finally {
		con.disconnectSync?.();
	}
}

export const load: PageServerLoad = async ({ params, url }) => {
	if (!RUN_ID_RE.test(params.run_id)) {
		throw error(400, 'invalid run_id: must be 32 hex chars (UUID hex)');
	}
	// Optional ?compare_to=<run_id> overrides the auto-found predecessor.
	const compareToParam = url.searchParams.get('compare_to');
	const explicitPrev =
		compareToParam && RUN_ID_RE.test(compareToParam) ? compareToParam : undefined;
	const tracePage = positiveIntParam(url.searchParams.get('trace_page'), 'trace_page');
	const traceSnapshot = traceSnapshotParam(url.searchParams.get('trace_snapshot'));
	assertNoActiveWriter();
	const meta = await getRunMeta(params.run_id);
	if (!meta) throw error(404, `run_id ${params.run_id} not found`);

	const narrativePromise = meta.status === 'succeeded'
		? getRunNarrative(params.run_id, explicitPrev) as Promise<RunNarrative | null>
		: Promise.resolve(null);
	const [narrative, coverage, fidelity, allRuns] = await Promise.all([
		narrativePromise,
		getHeuristicCoverage(params.run_id) as Promise<HeuristicCoverage | null>,
		getTraceFidelity(params.run_id, {
			detailOffset: (tracePage - 1) * TRACE_FIDELITY_DETAIL_LIMIT,
			detailLimit: TRACE_FIDELITY_DETAIL_LIMIT,
			detailSnapshotStartedAt: traceSnapshot
		}) as Promise<TraceFidelitySummary | null>,
		listSucceededRuns()
	]);

	if (fidelity) {
		const canonicalTracePage = Math.floor(fidelity.detail_offset / fidelity.detail_limit) + 1;
		const canonicalSnapshot = fidelity.detail_snapshot_started_at;
		const shouldPinSnapshot = traceSnapshot !== null || (meta.status === 'running' && tracePage > 1);
		const snapshotMismatch = shouldPinSnapshot && traceSnapshot !== canonicalSnapshot;
		if (tracePage !== canonicalTracePage || snapshotMismatch) {
			throw redirect(
				302,
				runHref(
					params.run_id,
					explicitPrev,
					canonicalTracePage,
					shouldPinSnapshot ? canonicalSnapshot : null,
					url.searchParams.has('trace_page')
				)
			);
		}
	}

	return {
		meta,
		narrative,
		coverage,
		fidelity,
		allRuns,
		compareToParam: explicitPrev ?? null
	};
};
