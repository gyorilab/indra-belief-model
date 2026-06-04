import {
	getRuns,
	compareAnatomy,
	stratifyCell,
	cohortForCell,
	evidenceSideBySide,
	goldPerformance,
	type RunSummary,
	type CompareAnatomy,
	type CellStratification,
	type Cohort,
	type SideBySideEvidence,
	type CompareCell,
	type StratAxis,
	type GoldFilter,
	type GoldGranularity,
	type GoldPerformance
} from '$lib/data/queries';
import type { PageServerLoad } from './$types';

const RUN_ID_RE = /^[a-f0-9]{32}$/i;
const HASH_RE = /^[a-f0-9]{1,32}$/i;
const CELLS: CompareCell[] = ['acbc', 'acbi', 'aibc', 'aibi'];
const AXES: StratAxis[] = ['source_api', 'stmt_type', 'grounding_status', 'bucket_a', 'bucket_b'];
const GOLD_FILTERS: GoldFilter[] = ['any', 'match', 'fp', 'fn', 'disagree'];

function pickRun(runs: RunSummary[], requested: string | null): string | null {
	if (requested && RUN_ID_RE.test(requested) && runs.some((r) => r.run_id === requested)) {
		return requested;
	}
	return null;
}

export const load: PageServerLoad = async ({ url }) => {
	const runs = getRuns();
	const q = url.searchParams;

	// Default a/b to the two newest runs (getRuns returns newest-first).
	const a = pickRun(runs, q.get('a')) ?? runs[0]?.run_id ?? null;
	const b = pickRun(runs, q.get('b')) ?? runs[1]?.run_id ?? null;

	const semanticOnly = q.get('sem') === '1';
	const cellParam = q.get('cell');
	const cell: CompareCell | null = cellParam && CELLS.includes(cellParam as CompareCell) ? (cellParam as CompareCell) : null;
	const axisParam = q.get('axis');
	const axis: StratAxis | null = axisParam && AXES.includes(axisParam as StratAxis) ? (axisParam as StratAxis) : null;
	const axisValue = q.get('val');
	const ev = q.get('ev'); // "<stmt_hash>.<evidence_hash>"
	const goldMode = q.get('mode') === 'gold';
	const goldParam = q.get('gold');
	const goldFilter: GoldFilter | null =
		goldParam && GOLD_FILTERS.includes(goldParam as GoldFilter) ? (goldParam as GoldFilter) : null;
	const granularity: GoldGranularity = q.get('gran') === 'statement' ? 'statement' : 'evidence';

	let anatomy: CompareAnatomy | null = null;
	let stratification: CellStratification | null = null;
	let cohort: Cohort | null = null;
	let sideBySide: SideBySideEvidence | null = null;
	let goldPerf: GoldPerformance | null = null;

	if (a && b && a !== b) {
		anatomy = compareAnatomy(a, b);
		if (goldMode) goldPerf = goldPerformance(a, b, granularity);

		// L3: a specific evidence is selected → side-by-side reasoning.
		if (ev) {
			const dot = ev.indexOf('.');
			if (dot > 0) {
				const sh = ev.slice(0, dot);
				const eh = ev.slice(dot + 1);
				if (HASH_RE.test(sh) && HASH_RE.test(eh)) {
					sideBySide = evidenceSideBySide(a, b, sh, eh);
				}
			}
		}

		// L1 + L2: a cell is selected → stratify (default axis source_api) + cohort.
		if (cell) {
			const stratAxis: StratAxis = axis ?? 'source_api';
			stratification = stratifyCell(a, b, cell, stratAxis, semanticOnly);
			cohort = cohortForCell(a, b, cell, {
				axis: axis,
				axisValue: axis ? axisValue : null,
				semanticOnly,
				goldFilter
			});
		}
	}

	return {
		runs,
		selectedA: a,
		selectedB: b,
		semanticOnly,
		goldMode,
		goldFilter,
		granularity,
		goldPerf,
		cell,
		axis,
		axisValue,
		ev,
		anatomy,
		stratification,
		cohort,
		sideBySide
	};
};
