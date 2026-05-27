import {
	getCorpusOverview,
	getFindings,
	getFocusStatement,
	getHeuristicCoverage,
	getResidualDistribution,
	getRunNarrative,
	type HeuristicCoverage,
	type RunNarrative
} from '$lib/db';
import {
	datasetShape,
	getDatasets,
	getDatasetIngestStatus,
	type DatasetDescriptor,
	type DatasetShape,
	type IngestStatus
} from '$lib/datasets';
import { listPairedWorkflowStates, pairedWorkflowIsActive, reconcileStalePairedWorkflowStates } from '$lib/server/pairedState';
import type { PublicWriterLockState } from '$lib/server/pairedState';
import { assertNoActiveWriter } from '$lib/server/writerReadGuard';
import type { PageServerLoad } from './$types';

const STMT_HASH_RE = /^[a-f0-9]{16}$/i;

export const load: PageServerLoad = async ({ url }) => {
	const focusParam = url.searchParams.get('focus');
	const focusHash = focusParam && STMT_HASH_RE.test(focusParam) ? focusParam : undefined;
	assertNoActiveWriter();
	reconcileStalePairedWorkflowStates();
	const pairedWorkflows = listPairedWorkflowStates(8).map((p) => ({
		pair_id: p.pair_id,
		status: p.status,
		source_dump_id: p.source_dump_id,
		model: p.model,
		scorer_version: p.scorer_version,
		total_cost_threshold_usd: p.total_cost_threshold_usd,
		href: p.href,
		created_at: p.created_at,
		updated_at: p.updated_at,
		started_at: p.started_at,
		finished_at: p.finished_at,
		termination_reason: p.termination_reason,
		architectures: p.architectures,
		is_active: pairedWorkflowIsActive(p)
	}));

	const [overview, focus, findings, residuals] = await Promise.all([
		getCorpusOverview(),
		getFocusStatement(focusHash),
		getFindings(),
		getResidualDistribution()
	]);

	const narratives: Record<string, RunNarrative> = {};
	const narrativeRows = await Promise.all(
		overview.scorerRuns
			.filter((r) => r.status === 'succeeded')
			.map(async (r) => [r.run_id, await getRunNarrative(r.run_id)] as const)
	);
	for (const [rid, n] of narrativeRows) {
		if (n) narratives[rid] = n;
	}

	let coverage: HeuristicCoverage | null = null;
	if (overview.latestValidity?.run_id) {
		coverage = await getHeuristicCoverage(overview.latestValidity.run_id);
	}

	// U2: filesystem-discovered datasets + lazy shape preview + ingest status.
	const baseDescriptors = getDatasets();
	const datasets: Array<DatasetDescriptor & { shape: DatasetShape; ingest: IngestStatus | null }> = await Promise.all(
		baseDescriptors.map(async (d) => ({
			...d,
			shape: datasetShape(d),
			ingest: await getDatasetIngestStatus(d)
		}))
	);

	return {
		overview,
		focus,
		findings,
		residuals,
		narratives,
		coverage,
		datasets,
		pairedWorkflows,
		writerLock: null as PublicWriterLockState | null
	};
};
