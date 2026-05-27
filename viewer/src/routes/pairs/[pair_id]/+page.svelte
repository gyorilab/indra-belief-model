<script lang="ts">
	import { invalidateAll } from '$app/navigation';
	import { browser } from '$app/environment';
	import { page } from '$app/state';
	import { onMount } from 'svelte';
	import type { PageData } from './$types';
	import type { PairedDenominatorRow, PairedExampleRow, PairedResourceFrontierArch, PairedVerdictPairRow, PairRunSummary } from '$lib/db';
	import LedgerKindChip from '$lib/components/LedgerKindChip.svelte';
	import LedgerRoleMark from '$lib/components/LedgerRoleMark.svelte';
	import LedgerScopeChip from '$lib/components/LedgerScopeChip.svelte';
	import { fmtBelief, fmtDelta, shortHash, verdictDisplay } from '$lib/format';
	import { pairedMetricKindFamily, panelApplicabilityClass } from '$lib/pairedMetricKinds';
	import {
		buildPairedComparisonCard,
		pairedComparisonCardJson,
		pairedComparisonCardMarkdown,
		pairedComparisonCardSchemaJson
	} from '$lib/pairedComparisonCard';

	let { data }: { data: PageData } = $props();
	const w = $derived(data.workbench);
	const workflow = $derived(data.workflow);
	const WORKFLOW_ARCHES = ['monolithic', 'decomposed'] as const;
	type ArchitectureLane = (typeof WORKFLOW_ARCHES)[number];
	type MetricWinner = ArchitectureLane | 'tie' | 'none';
	type WorkflowState = NonNullable<PageData['workflow']>;
	type WorkflowArchState = WorkflowState['architectures'][ArchitectureLane];

	type ExemplarKey =
		| 'monolithic_wins'
		| 'decomposed_wins'
		| 'verdict_disagreements'
		| 'mutual_failures'
		| 'monolithic_only'
		| 'decomposed_only'
		| 'excluded_by_integrity';

	type ExemplarLane = { key: ExemplarKey; title: string; scope: string };
	type ExemplarFoldGroup =
		| { kind: 'single'; key: string; row: PairedExampleRow }
		| { kind: 'pattern'; key: string; rows: PairedExampleRow[]; sample: PairedExampleRow };
	type ExemplarLaneCount = { key: ExemplarKey; title: string; count: number; unit: string };
	type ExemplarSectionSummary = {
		distinctN: number;
		positionsN: number;
		extraPositionN: number;
		crossListedN: number;
		laneCounts: ExemplarLaneCount[];
	};
	type ExemplarSortMode = 'impact' | 'statement' | 'type' | 'verdict';
	type ExemplarFilterMode = 'all' | 'verdict_split' | 'truth_anchored' | 'nonoverlap';
	type ExemplarControlOption<T extends string> = { mode: T; label: string };
	type StoredFoldState = { open?: string[] };
	type VerdictPairGroup = {
		support_cell: string;
		rows: PairedVerdictPairRow[];
		n: number;
		exactMatches: number;
		exactDivergences: number;
	};
	type PairedOutsidePolicy = {
		hidden: boolean;
		hasUnknown: boolean;
		rowCount: number;
		total: number;
	};

	const EXEMPLAR_LANES: ExemplarLane[] = [
		{ key: 'excluded_by_integrity', title: 'excluded by integrity', scope: 'not compared; one side has an aggregate error, missing aggregate, or aggregate without a verdict' },
		{ key: 'monolithic_wins', title: '[M] closer to INDRA', scope: 'overlap only; positive error delta means [D] was farther from INDRA' },
		{ key: 'decomposed_wins', title: '[D] closer to INDRA', scope: 'overlap only; negative error delta means [D] was closer to INDRA' },
		{ key: 'verdict_disagreements', title: 'verdict disagreements', scope: 'overlap only; same evidence, different aggregate verdict' },
		{ key: 'mutual_failures', title: 'mutual failures', scope: 'overlap only; both scores are far from INDRA belief' },
		{ key: 'monolithic_only', title: '[M] true non-overlap examples', scope: 'not compared; [D] has no persisted rows for this evidence in the paired run' },
		{ key: 'decomposed_only', title: '[D] true non-overlap examples', scope: 'not compared; [M] has no persisted rows for this evidence in the paired run' }
	];
	const EXEMPLAR_LANE_ANCHORS: Record<ExemplarKey, string> = {
		excluded_by_integrity: 'lane-excluded-by-integrity',
		monolithic_wins: 'lane-m-closer-to-indra',
		decomposed_wins: 'lane-d-closer-to-indra',
		verdict_disagreements: 'lane-verdict-disagreements',
		mutual_failures: 'lane-mutual-failures',
		monolithic_only: 'lane-m-true-nonoverlap',
		decomposed_only: 'lane-d-true-nonoverlap'
	};
	const EXEMPLAR_SORT_OPTIONS: ExemplarControlOption<ExemplarSortMode>[] = [
		{ mode: 'impact', label: 'impact' },
		{ mode: 'verdict', label: 'verdict' },
		{ mode: 'type', label: 'type' },
		{ mode: 'statement', label: 'statement' }
	];
	const EXEMPLAR_FILTER_OPTIONS: ExemplarControlOption<ExemplarFilterMode>[] = [
		{ mode: 'all', label: 'all' },
		{ mode: 'verdict_split', label: 'verdict split' },
		{ mode: 'truth_anchored', label: 'truth anchored' },
		{ mode: 'nonoverlap', label: 'non-overlap' }
	];
	const FOLD_COOKIE = 'indra_pair_exemplar_folds_v2';
	const MAX_FOLD_COOKIE_BYTES = 3500;
	const WORKFLOW_STALL_MS = 30_000;
	let tickNow = $state(Date.now());
	const exemplarLaneMemberships = $derived.by(() => {
		const memberships = new Map<string, ExemplarLane[]>();
		for (const lane of EXEMPLAR_LANES) {
			for (const row of w.exemplars[lane.key]) {
				const identity = exemplarIdentity(row);
				const lanes = memberships.get(identity);
				if (lanes) {
					if (!lanes.some((member) => member.key === lane.key)) lanes.push(lane);
				} else {
					memberships.set(identity, [lane]);
				}
			}
		}
		return memberships;
	});

	const visibleIntegrityExemplarLanes = $derived.by(() =>
		EXEMPLAR_LANES.filter((lane) => lane.key === 'excluded_by_integrity' && w.exemplars[lane.key].length > 0)
	);
	const visibleComparisonExemplarLanes = $derived.by(() =>
		EXEMPLAR_LANES.filter((lane) => lane.key !== 'excluded_by_integrity' && w.exemplars[lane.key].length > 0)
	);
	const emptyComparisonExemplarLanes = $derived.by(() =>
		EXEMPLAR_LANES.filter((lane) => lane.key !== 'excluded_by_integrity' && w.exemplars[lane.key].length === 0)
	);
	const exemplarSortMode = $derived(normalizeExemplarSort(page.url.searchParams.get('exemplar_sort')));
	const exemplarFilterMode = $derived(normalizeExemplarFilter(page.url.searchParams.get('exemplar_filter')));
	const activeComparisonExemplarLanes = $derived.by(() =>
		visibleComparisonExemplarLanes.filter((lane) => sortedExemplarRows(lane.key).length > 0)
	);
	const hiddenByFilterComparisonExemplarLanes = $derived.by(() =>
		visibleComparisonExemplarLanes.filter(
			(lane) => exemplarFilterMode !== 'all' && sortedExemplarRows(lane.key).length === 0
		)
	);
	let openPatternKeys = $state<Set<string>>(new Set());
	let foldStateLoaded = $state(false);
	let comparisonCardGeneratedAt = $state('client-pending');
	const effectiveOpenPatternKeys = $derived.by(() =>
		foldStateLoaded ? openPatternKeys : initialServerFoldState()
	);
	const visiblePatternKeys = $derived.by(() => {
		const keys = new Set<string>();
		for (const lane of activeComparisonExemplarLanes) {
			for (const group of foldedExemplarGroups(lane.key)) {
				if (group.kind === 'pattern') keys.add(patternStorageKey(lane.key, group.key));
			}
		}
		return keys;
	});
	const comparisonExemplarSummary = $derived.by((): ExemplarSectionSummary => {
		const identities = new Set<string>();
		const laneCounts: ExemplarLaneCount[] = [];
		let positionsN = 0;
		for (const lane of activeComparisonExemplarLanes) {
			const rows = sortedExemplarRows(lane.key);
			if (rows.length === 0) continue;
			positionsN += rows.length;
			laneCounts.push({
				key: lane.key,
				title: lane.title,
				count: rows.length,
				unit: exemplarLaneUnit(lane.key)
			});
			for (const row of rows) identities.add(exemplarIdentity(row));
		}
		let crossListedN = 0;
		for (const identity of identities) {
			const visibleMemberships = activeComparisonExemplarLanes.filter((lane) =>
				sortedExemplarRows(lane.key).some((row) => exemplarIdentity(row) === identity)
			);
			if (visibleMemberships.length > 1) crossListedN += 1;
		}
		return {
			distinctN: identities.size,
			positionsN,
			extraPositionN: Math.max(0, positionsN - identities.size),
			crossListedN,
			laneCounts
		};
	});
	const visibleFoldPatternCount = $derived.by(() => {
		return visiblePatternKeys.size;
	});
	const visibleOpenFoldCount = $derived.by(() => {
		let count = 0;
		for (const key of visiblePatternKeys) {
			if (effectiveOpenPatternKeys.has(key)) count += 1;
		}
		return count;
	});
	const hiddenOpenFoldCount = $derived.by(() =>
		Math.max(0, effectiveOpenPatternKeys.size - visibleOpenFoldCount)
	);
	const hiddenOpenFoldLanes = $derived.by(() => {
		const lanes: ExemplarLane[] = [];
		const seen = new Set<ExemplarKey>();
		for (const key of effectiveOpenPatternKeys) {
			if (visiblePatternKeys.has(key)) continue;
			const lane = foldLaneFromStorageKey(key);
			if (!lane || seen.has(lane.key)) continue;
			seen.add(lane.key);
			lanes.push(lane);
		}
		return lanes;
	});
	const totalOpenFoldCount = $derived.by(() => effectiveOpenPatternKeys.size);

	onMount(() => {
		comparisonCardGeneratedAt = new Date().toISOString();
		openPatternKeys = readStoredFoldState();
		openPatternForCurrentHash();
		foldStateLoaded = true;
		const tickHandle = setInterval(() => { tickNow = Date.now(); }, 1000);
		const handleHashChange = () => openPatternForCurrentHash();
		window.addEventListener('hashchange', handleHashChange);
		return () => {
			window.removeEventListener('hashchange', handleHashChange);
			clearInterval(tickHandle);
		};
	});
		function isArchNativeLedgerRow(row: PairedDenominatorRow): boolean {
			return row.key === 'monolithic_aggregate_rows'
				|| row.key === 'decomposed_native_step_rows'
				|| row.key === 'nonaggregate_step_error_evidence';
		}

	const pairedDenominatorRows = $derived.by(() =>
		w.denominator_ledger.filter((row) => !isArchNativeLedgerRow(row))
	);
	const archNativeLedgerRows = $derived.by(() =>
		w.denominator_ledger.filter((row) => isArchNativeLedgerRow(row))
	);
	const pairedOutsidePolicy = $derived.by((): PairedOutsidePolicy => {
		let hasUnknown = false;
		let total = 0;
		for (const row of pairedDenominatorRows) {
			if (row.excluded_n == null || Number.isNaN(row.excluded_n)) {
				hasUnknown = true;
			} else if (row.excluded_n > 0) {
				total += row.excluded_n;
			}
		}
		return {
			hidden: pairedDenominatorRows.length > 0 && !hasUnknown && total === 0,
			hasUnknown,
			rowCount: pairedDenominatorRows.length,
			total
		};
	});
	const showPairedOutsideColumn = $derived.by(() => !pairedOutsidePolicy.hidden);
	const overlapIntegrityText = $derived.by(() => {
		const o = w.overlap;
		if (!o) return '';
		const sideText = (
			mark: string,
			stepError: number,
			missingAggregate: number,
			nonverdictAggregate: number
		) => {
			const bits: string[] = [];
			if (stepError > 0) bits.push(`${stepError.toLocaleString()} step-error`);
			if (missingAggregate > 0) bits.push(`${missingAggregate.toLocaleString()} missing-aggregate`);
			if (nonverdictAggregate > 0) bits.push(`${nonverdictAggregate.toLocaleString()} non-verdict aggregate`);
			return bits.length > 0 ? `${mark} ${bits.join(', ')}` : null;
		};
		const parts = [
			sideText('[M]', o.monolithic_step_error_evidences, o.monolithic_missing_aggregate_evidences, o.monolithic_nonverdict_aggregate_evidences),
			sideText('[D]', o.decomposed_step_error_evidences, o.decomposed_missing_aggregate_evidences, o.decomposed_nonverdict_aggregate_evidences)
		].filter(Boolean);
		return parts.length > 0 ? `Excluded from comparable metrics: ${parts.join('; ')}.` : '';
	});
	const overlapTraceWarningText = $derived.by(() => {
		const o = w.overlap;
		if (!o) return '';
		const parts: string[] = [];
		if (o.monolithic_nonaggregate_step_error_evidences > 0) {
			parts.push(`[M] ${o.monolithic_nonaggregate_step_error_evidences.toLocaleString()} non-aggregate step-error`);
		}
		if (o.decomposed_nonaggregate_step_error_evidences > 0) {
			parts.push(`[D] ${o.decomposed_nonaggregate_step_error_evidences.toLocaleString()} non-aggregate step-error`);
		}
		return parts.length > 0
			? `Trace warnings outside the comparable aggregate gate: ${parts.join('; ')}.`
			: '';
	});

	function archMark(architecture: string | null | undefined): string {
		if (architecture === 'monolithic') return '[M]';
		if (architecture === 'decomposed') return '[D]';
		return '[?]';
	}

	function architectureLabel(architecture: string | null | undefined, fallback = 'unknown'): string {
		if (architecture === 'monolithic' || architecture === 'decomposed') return architecture;
		return fallback;
	}

	function runStatusClass(status: string | null | undefined): string {
		return (status ?? 'missing').toLowerCase().replace(/[^a-z0-9_-]+/g, '-') || 'unknown';
	}

	function fmtCost(c: number | null | undefined): string {
		if (c == null) return '-';
		if (c < 0.01) return '<$0.01';
		if (c < 100) return '$' + c.toFixed(2);
		return '$' + c.toFixed(0);
	}

	function fmtUnitCost(c: number | null | undefined): string {
		if (c == null || Number.isNaN(c)) return '-';
		if (c === 0) return '$0';
		if (c < 0.0001) return '<$0.0001';
		if (c < 0.01) return '$' + c.toFixed(4);
		if (c < 1) return '$' + c.toFixed(3);
		return fmtCost(c);
	}

	function runCost(r: PairRunSummary): string {
		if (r.status === 'succeeded') return fmtCost(r.cost_actual_usd ?? r.cost_estimate_usd);
		if (r.cost_actual_usd == null) return 'not finalized';
		return `${fmtCost(r.cost_actual_usd)} partial`;
	}

	function fmtRatio(n: number | null | undefined, d: number | null | undefined): string {
		return `${fmtCount(n)}/${fmtCount(d)}`;
	}

	function fmtMaybe(n: number | null | undefined, digits = 3): string {
		return n == null || Number.isNaN(n) ? '-' : n.toFixed(digits);
	}

	function fmtAbsMaybe(n: number | null | undefined, digits = 3): string {
		return n == null || Number.isNaN(n) ? '-' : Math.abs(n).toFixed(digits);
	}

	function scoreShiftDirection(n: number | null | undefined): string {
		if (n == null || Number.isNaN(n) || Math.abs(n) < 0.0000001) return 'no mean-score movement';
		return n < 0 ? '[D] lower than [M]' : '[D] higher than [M]';
	}

	function scoreAxisPercent(n: number | null | undefined): number | null {
		if (n == null || Number.isNaN(n)) return null;
		return Math.max(0, Math.min(1, n)) * 100;
	}

	function scoreAxisStyle(monolithic: number | null | undefined, decomposed: number | null | undefined): string {
		const m = scoreAxisPercent(monolithic);
		const d = scoreAxisPercent(decomposed);
		if (m == null || d == null) {
			return '--m-pos: 50%; --d-pos: 50%;';
		}
		return `--m-pos: ${m.toFixed(3)}%; --d-pos: ${d.toFixed(3)}%;`;
	}

	function fmtMs(n: number | null | undefined): string {
		if (n == null || Number.isNaN(n)) return '-';
		if (n < 1000) return `${n.toFixed(0)} ms`;
		return `${(n / 1000).toFixed(1)} s`;
	}

	function fmtSeconds(n: number | null | undefined): string {
		if (n == null || Number.isNaN(n)) return '-';
		if (n < 60) return `${n.toFixed(1)}s`;
		if (n < 3600) return `${(n / 60).toFixed(1)}m`;
		return `${(n / 3600).toFixed(1)}h`;
	}

	function fmtEta(ms: number | null): string {
		if (ms == null || !Number.isFinite(ms) || ms < 0) return 'eta pending';
		const s = Math.round(ms / 1000);
		if (s < 60) return `eta ${s}s`;
		return `eta ${Math.floor(s / 60)}m ${s % 60}s`;
	}

	function parseTimeMs(value: string | null | undefined): number | null {
		if (!value) return null;
		const ms = Date.parse(value);
		return Number.isFinite(ms) ? ms : null;
	}

	function fmtAge(ms: number | null): string {
		if (ms == null || !Number.isFinite(ms) || ms < 0) return 'time -';
		const s = Math.max(0, Math.round(ms / 1000));
		if (s < 60) return `${s}s`;
		if (s < 3600) return `${Math.floor(s / 60)}m ${s % 60}s`;
		const h = Math.floor(s / 3600);
		const m = Math.floor((s % 3600) / 60);
		return `${h}h ${m}m`;
	}

	function since(value: string | null | undefined): number | null {
		const ms = parseTimeMs(value);
		return ms == null ? null : tickNow - ms;
	}

	function fmtCount(n: number | null | undefined): string {
		return n == null || Number.isNaN(n) ? '-' : n.toLocaleString();
	}

	function fmtRate(n: number | null | undefined, maximumFractionDigits = 1): string {
		return n == null || Number.isNaN(n)
			? '-'
			: n.toLocaleString(undefined, { maximumFractionDigits });
	}

	function metricRate(numerator: number | null | undefined, denominator: number | null | undefined): number | null {
		if (numerator == null || denominator == null || denominator <= 0 || Number.isNaN(numerator)) return null;
		return numerator / denominator;
	}

	function telemetryCoverageText(
		observed: number | null | undefined,
		total: number | null | undefined,
		kind: 'latency' | 'tokens'
	): string {
		return `${fmtCount(observed)}/${fmtCount(total)} rows report ${kind}`;
	}

	function telemetryComplete(observed: number | null | undefined, total: number | null | undefined): boolean {
		return observed != null && total != null && total > 0 && observed === total;
	}

	function resourceTelemetryWinner(
		monolithicValue: number | null | undefined,
		decomposedValue: number | null | undefined,
		monolithicObserved: number | null | undefined,
		decomposedObserved: number | null | undefined,
		total: number | null | undefined
	): MetricWinner {
		if (!telemetryComplete(monolithicObserved, total) || !telemetryComplete(decomposedObserved, total)) return 'none';
		return metricWinnerByPositiveLower(monolithicValue, decomposedValue);
	}

	function fmtLedgerCount(n: number | null | undefined): string {
		return n == null || Number.isNaN(n) ? 'n/a' : n.toLocaleString();
	}

	function metricWinnerByLower(
		monolithic: number | null | undefined,
		decomposed: number | null | undefined
	): MetricWinner {
		if (monolithic == null || decomposed == null || Number.isNaN(monolithic) || Number.isNaN(decomposed)) return 'none';
		if (Math.abs(monolithic - decomposed) < 0.0000001) return 'tie';
		return monolithic < decomposed ? 'monolithic' : 'decomposed';
	}

	function metricWinnerByAbsLower(
		monolithic: number | null | undefined,
		decomposed: number | null | undefined
	): MetricWinner {
		if (monolithic == null || decomposed == null || Number.isNaN(monolithic) || Number.isNaN(decomposed)) return 'none';
		const monolithicAbs = Math.abs(monolithic);
		const decomposedAbs = Math.abs(decomposed);
		if (Math.abs(monolithicAbs - decomposedAbs) < 0.0000001) return 'tie';
		return monolithicAbs < decomposedAbs ? 'monolithic' : 'decomposed';
	}

	function metricWinnerByPositiveLower(
		monolithic: number | null | undefined,
		decomposed: number | null | undefined
	): MetricWinner {
		if (
			monolithic == null ||
			decomposed == null ||
			Number.isNaN(monolithic) ||
			Number.isNaN(decomposed) ||
			monolithic <= 0 ||
			decomposed <= 0
		) return 'none';
		return metricWinnerByLower(monolithic, decomposed);
	}

	function metricRowClass(arch: ArchitectureLane, winner: MetricWinner): string {
		return `metric-row metric-row-${arch}${winner === arch ? ' metric-row-winner' : ''}${winner === 'tie' ? ' metric-row-tie' : ''}`;
	}

	function metricWinnerLabel(arch: ArchitectureLane, winner: MetricWinner, label: string): string {
		if (winner === arch) return label;
		if (winner === 'tie') return 'tie';
		return '';
	}

	function zeroMetricLabel(value: number | null | undefined): string {
		return value === 0 ? 'not ranked' : '';
	}

	function lowerRatioLabel(
		arch: ArchitectureLane,
		winner: MetricWinner,
		monolithic: number | null | undefined,
		decomposed: number | null | undefined,
		label: string
	): string {
		const base = metricWinnerLabel(arch, winner, label);
		if (!base || monolithic == null || decomposed == null || monolithic <= 0 || decomposed <= 0) return base;
		const ratio = Math.max(monolithic, decomposed) / Math.min(monolithic, decomposed);
		if (!Number.isFinite(ratio) || ratio < 1.05) return base;
		return `${base} · ${ratio.toFixed(1)}x`;
	}

	function frontierRows(): PairedResourceFrontierArch[] {
		if (!w.resource_frontier) return [];
		return [w.resource_frontier.monolithic, w.resource_frontier.decomposed];
	}

	function frontierRowClass(arch: ArchitectureLane, winner: MetricWinner): string {
		return `frontier-row frontier-row-${arch}${winner === arch ? ' frontier-row-winner' : ''}${winner === 'tie' ? ' frontier-row-tie' : ''}`;
	}

	function frontierCostBasis(row: PairedResourceFrontierArch): string {
		if (row.run_cost_basis === 'missing') return 'cost missing';
		return `${row.run_cost_basis} ${fmtCost(row.run_cost_usd)}`;
	}

	function frontierQuality(row: PairedResourceFrontierArch): string {
		return row.truth_overlap_n > 0 ? fmtMaybe(row.mae) : 'not defined';
	}

	function supportedCount(arch: ArchitectureLane): number {
		if (!w.comparable) return 0;
		return arch === 'monolithic'
			? w.comparable.both_correct_n + w.comparable.monolithic_only_correct_n
			: w.comparable.both_correct_n + w.comparable.decomposed_only_correct_n;
	}

	function supportCellLabel(cell: string): string {
		if (cell === 'both_supported') return 'both supported';
		if (cell === 'monolithic_only') return '[M] only';
		if (cell === 'decomposed_only') return '[D] only';
		return 'neither supported';
	}

	function supportCellClass(cell: string): string {
		return `verdict-taxonomy-${cell.replace(/_/g, '-')}`;
	}

	function exactPairClass(pair: PairedVerdictPairRow): string {
		return pair.monolithic_verdict === pair.decomposed_verdict ? 'exact-match' : 'exact-divergence';
	}

	function exactPairLabel(pair: PairedVerdictPairRow): string {
		return pair.monolithic_verdict === pair.decomposed_verdict ? 'exact match' : 'exact divergent';
	}

	function supportCellSort(cell: string): number {
		if (cell === 'both_supported') return 0;
		if (cell === 'monolithic_only') return 1;
		if (cell === 'decomposed_only') return 2;
		if (cell === 'neither_supported') return 3;
		return 4;
	}

	function groupVerdictPairs(rows: PairedVerdictPairRow[]): VerdictPairGroup[] {
		const grouped = new Map<string, PairedVerdictPairRow[]>();
		for (const row of rows) {
			grouped.set(row.support_cell, [...(grouped.get(row.support_cell) ?? []), row]);
		}
		return [...grouped.entries()]
			.sort(([a], [b]) => supportCellSort(a) - supportCellSort(b))
			.map(([support_cell, groupRows]) => {
				const exactMatches = groupRows.reduce(
					(sum, r) => sum + (r.monolithic_verdict === r.decomposed_verdict ? r.n : 0),
					0
				);
				const n = groupRows.reduce((sum, r) => sum + r.n, 0);
				return {
					support_cell,
					rows: groupRows,
					n,
					exactMatches,
					exactDivergences: n - exactMatches
				};
			});
	}

		function ledgerRowClass(row: PairedDenominatorRow): string {
			const family = pairedMetricKindFamily(row.metric_kind);
			const scope = panelApplicabilityClass(row.applicability);
			return `ledger-row ledger-row-${scope} ledger-key-${row.key} ledger-row-family-${family} ledger-tree-role-${row.ledger_role}${row.parent_key ? ' ledger-row-child' : ''}`;
		}

		function ledgerAnchor(key: string): string {
			return `denom-${key.replace(/_/g, '-')}`;
		}

		function ledgerHref(key: string): string {
			return `#${ledgerAnchor(key)}`;
		}

		function ledgerParentLabel(key: string | null): string {
			if (key === 'side_aggregate_evidence_rows') return 'side aggregate base';
			if (key === 'clean_shared_aggregate_verdict_evidence') return 'clean shared paired base';
			return key ? key.replace(/_/g, ' ') : '';
		}

		function ledgerParentIsContinued(rows: PairedDenominatorRow[], index: number): boolean {
			const parent = rows[index]?.parent_key;
			return Boolean(parent && index > 0 && rows[index - 1]?.parent_key === parent);
		}

		function ledgerParentText(rows: PairedDenominatorRow[], index: number): string {
			const parent = rows[index]?.parent_key ?? null;
			if (!parent) return '';
			return ledgerParentIsContinued(rows, index) ? 'same parent' : `under ${ledgerParentLabel(parent)}`;
		}

		function ledgerCountClass(row: PairedDenominatorRow, column: 'panel' | 'monolithic' | 'decomposed' | 'shared' | 'outside'): string {
			const inherited = row.ledger_role === 'metric' && column === 'shared';
			const metricPopulation =
				row.ledger_role === 'metric' &&
				(column === 'panel' || column === 'monolithic' || column === 'decomposed');
			return `num${inherited ? ' ledger-count-inherited' : ''}${metricPopulation ? ' ledger-count-kind' : ''}`;
		}

	function ledgerCountText(
		row: PairedDenominatorRow,
		value: number | null,
		column: 'panel' | 'monolithic' | 'decomposed' | 'shared' | 'outside'
	): string {
			const formatted = fmtLedgerCount(value);
			if (value == null || row.ledger_role !== 'metric') return formatted;
			if (column === 'panel') return `metric n=${formatted}`;
			if (column === 'monolithic' || column === 'decomposed') return `side n=${formatted}`;
			if (column === 'shared') return `base n=${formatted}`;
		if (column === 'outside' && value === 0) return 'none';
		return formatted;
	}

	function outsidePolicyText(policy: PairedOutsidePolicy): string {
		if (policy.hidden) {
			return `column-policy: outside hidden; outside = 0 across ${fmtCount(policy.rowCount)} paired rows.`;
		}
		if (policy.hasUnknown) {
			return 'column-policy: outside shown because at least one paired outside value is not defined.';
		}
		return `column-policy: outside shown; total outside mass ${fmtCount(policy.total)} across ${fmtCount(policy.rowCount)} paired rows.`;
	}

		function archNativeSide(row: PairedDenominatorRow): string {
		if (row.monolithic_n != null && row.decomposed_n == null) return `[M] ${fmtLedgerCount(row.monolithic_n)}`;
		if (row.decomposed_n != null && row.monolithic_n == null) return `[D] ${fmtLedgerCount(row.decomposed_n)}`;
		return `[M] ${fmtLedgerCount(row.monolithic_n)} / [D] ${fmtLedgerCount(row.decomposed_n)}`;
	}

		function workflowIsActive(): boolean {
			return workflow?.status === 'queued' || workflow?.status === 'running';
		}

		function workflowConsoleStatus(): string {
			return workflow?.status ?? 'idle';
		}

		function workflowConsoleText(): string {
			if (!workflow) {
				return 'No live paired workflow sidecar is attached to this pair; the comparison below is from persisted score_run rows.';
			}
			if (workflow.status === 'queued') return 'Queued for the DuckDB writer lock; no architecture has started spending yet.';
			if (workflow.status === 'running') return 'Live paired workflow; one architecture runs at a time under the DuckDB writer lock.';
			if (workflow.status === 'succeeded') return 'Paired workflow completed; rerun starts from a fresh cost preflight and appends new score_run rows.';
			if (workflow.status === 'canceled') return 'Paired workflow was canceled; any completed child runs remain append-only and inspectable.';
			return 'Paired workflow failed; inspect child run state before rerunning from the same source.';
		}

		function workflowRerunHref(): string {
			if (!workflow) return '/#datasets';
			const params = new URLSearchParams();
			params.set('pair_source', workflow.dataset_path);
			params.set('pair_model', workflow.model);
			params.set('pair_scorer', workflow.scorer_version);
			return `/?${params.toString()}#datasets`;
		}

	async function cancelWorkflow(): Promise<void> {
		if (!workflow) return;
		await fetch(`/api/runs/score-paired/${workflow.pair_id}/cancel`, { method: 'POST' });
		await invalidateAll();
	}

	function runPaneLabel(r: PairRunSummary | null, fallbackArch: ArchitectureLane): string {
		if (!r) return `${fallbackArch} architecture lane missing run`;
		return `${architectureLabel(r.architecture, fallbackArch)} architecture lane ${r.status} run ${shortHash(r.run_id)}`;
	}

	function foldStorageKey(): string {
		return foldStorageKeyForPair(w.pair_id);
	}

	function foldStorageKeyForPair(pairId: string): string {
		return `indra:pair:${pairId}:exemplar-folds:v2`;
	}

	function patternStorageKey(laneKey: ExemplarKey, patternKey: string): string {
		return `${laneKey}\u001f${patternKey}`;
	}

	function foldLaneFromStorageKey(key: string): ExemplarLane | null {
		const separator = key.indexOf('\u001f');
		if (separator < 0) return null;
		const laneKey = key.slice(0, separator);
		return EXEMPLAR_LANES.find((lane) => lane.key === laneKey) ?? null;
	}

	function readStoredFoldState(): Set<string> {
		return readStoredFoldStateForPair(w.pair_id);
	}

	function initialServerFoldState(): Set<string> {
		return new Set(data.initial_open_pattern_keys ?? []);
	}

	function readStoredFoldStateForPair(pairId: string): Set<string> {
		if (!browser) return initialServerFoldState();
		try {
			const raw = window.localStorage.getItem(foldStorageKeyForPair(pairId));
			const parsed = parseStoredFoldState(raw);
			if (parsed) return parsed;
		} catch {
			// Fall through to the SSR-readable cookie mirror.
		}
		return readStoredFoldCookie();
	}

	function parseStoredFoldState(raw: string | null | undefined): Set<string> | null {
		if (!raw) return null;
		try {
			const parsed = JSON.parse(raw) as StoredFoldState;
			if (!Array.isArray(parsed.open)) return null;
			return new Set(parsed.open.filter((key): key is string => typeof key === 'string'));
		} catch {
			return null;
		}
	}

	function foldCookiePath(): string {
		return `/pairs/${w.pair_id}`;
	}

	function readStoredFoldCookie(): Set<string> {
		if (!browser) return initialServerFoldState();
		const cookie = document.cookie
			.split('; ')
			.find((item) => item.startsWith(`${FOLD_COOKIE}=`));
		if (!cookie) return new Set();
		const encoded = cookie.slice(FOLD_COOKIE.length + 1);
		return parseStoredFoldState(decodeURIComponent(encoded)) ?? new Set();
	}

	function writeStoredFoldCookie(keys: Set<string>): void {
		if (!browser) return;
		const path = foldCookiePath();
		if (keys.size === 0) {
			document.cookie = `${FOLD_COOKIE}=; Max-Age=0; Path=${path}; SameSite=Lax`;
			return;
		}
		// Paint hint only: pair-scoped plaintext cookie for trusted local/loopback deployments.
		// localStorage remains authoritative when the full fold set is too large for a cookie.
		const open: string[] = [];
		for (const key of keys) {
			const next = [...open, key];
			const encoded = encodeURIComponent(JSON.stringify({ open: next, truncated: next.length < keys.size }));
			if (encoded.length > MAX_FOLD_COOKIE_BYTES) break;
			open.push(key);
		}
		if (open.length === 0) {
			document.cookie = `${FOLD_COOKIE}=; Max-Age=0; Path=${path}; SameSite=Lax`;
			return;
		}
		const encoded = encodeURIComponent(JSON.stringify({ open, truncated: open.length < keys.size }));
		document.cookie = `${FOLD_COOKIE}=${encoded}; Max-Age=31536000; Path=${path}; SameSite=Lax`;
	}

	function writeStoredFoldState(keys: Set<string>): void {
		if (!browser) return;
		try {
			if (keys.size === 0) {
				window.localStorage.removeItem(foldStorageKey());
			} else {
				window.localStorage.setItem(foldStorageKey(), JSON.stringify({ open: Array.from(keys) }));
			}
		} catch {
			// Local storage can be unavailable in private or restricted contexts.
		}
		try {
			writeStoredFoldCookie(keys);
		} catch {
			// Cookie mirroring is an SSR paint optimization; local fold memory still works without it.
		}
	}

	function setOpenPatternKey(key: string, open: boolean): void {
		const next = new Set(openPatternKeys);
		if (open) next.add(key);
		else next.delete(key);
		openPatternKeys = next;
		writeStoredFoldState(next);
	}

	function openPattern(laneKey: ExemplarKey, patternKey: string): void {
		const key = patternStorageKey(laneKey, patternKey);
		if (openPatternKeys.has(key)) return;
		setOpenPatternKey(key, true);
	}

	function patternKeyForRow(laneKey: ExemplarKey, row: PairedExampleRow): string | null {
		const patternKey = exemplarPatternKey(row);
		return foldedExemplarGroups(laneKey).some(
			(group) => group.kind === 'pattern' && group.key === patternKey
		) ? patternKey : null;
	}

	function openPatternForRow(laneKey: ExemplarKey, row: PairedExampleRow | null | undefined): void {
		if (!row) return;
		const patternKey = patternKeyForRow(laneKey, row);
		if (patternKey) openPattern(laneKey, patternKey);
	}

	function openPatternForCurrentHash(): void {
		if (!browser) return;
		const hash = window.location.hash.slice(1);
		if (!hash) return;
		for (const lane of activeComparisonExemplarLanes) {
			for (const row of sortedExemplarRows(lane.key)) {
				if (exemplarRowAnchor(row, lane.key) === hash) {
					openPatternForRow(lane.key, row);
					return;
				}
			}
		}
	}

	function updatePatternOpen(laneKey: ExemplarKey, patternKey: string, event: Event): void {
		if (!foldStateLoaded) return;
		const target = event.currentTarget as HTMLDetailsElement | null;
		if (!target) return;
		const key = patternStorageKey(laneKey, patternKey);
		setOpenPatternKey(key, target.open);
	}

	function resetVisibleOpenPatterns(): void {
		const next = new Set(openPatternKeys);
		for (const key of visiblePatternKeys) next.delete(key);
		openPatternKeys = next;
		writeStoredFoldState(next);
	}

	function resetAllOpenPatterns(): void {
		const next = new Set<string>();
		openPatternKeys = next;
		writeStoredFoldState(next);
	}

	function normalizeExemplarSort(value: string | null): ExemplarSortMode {
		if (value === 'statement' || value === 'type' || value === 'verdict') return value;
		return 'impact';
	}

	function normalizeExemplarFilter(value: string | null): ExemplarFilterMode {
		if (value === 'verdict_split' || value === 'truth_anchored' || value === 'nonoverlap') return value;
		return 'all';
	}

	function exemplarModeHref(
		param: 'exemplar_sort' | 'exemplar_filter',
		mode: ExemplarSortMode | ExemplarFilterMode,
		defaultMode: ExemplarSortMode | ExemplarFilterMode,
		hash = 'comparison-exemplars'
	): string {
		const params = new URLSearchParams(page.url.searchParams);
		if (mode === defaultMode) params.delete(param);
		else params.set(param, mode);
		const query = params.toString();
		return `${page.url.pathname}${query ? `?${query}` : ''}#${hash}`;
	}

	function controlOptionLabel<T extends string>(options: ExemplarControlOption<T>[], mode: T): string {
		return options.find((option) => option.mode === mode)?.label ?? mode;
	}

	function exemplarRows(key: ExemplarKey): PairedExampleRow[] {
		return w.exemplars[key];
	}

	function laneTitle(key: ExemplarKey): string {
		return EXEMPLAR_LANES.find((lane) => lane.key === key)?.title ?? key;
	}

	function exemplarImpact(r: PairedExampleRow, laneKey: ExemplarKey): number {
		if (laneKey === 'excluded_by_integrity') {
			const reason = r.excluded_reason ?? '';
			if (reason.includes('step error')) return 3;
			if (reason.includes('missing aggregate')) return 2;
			if (reason.includes('lacks verdict')) return 1;
			return 0;
		}
		if (laneKey === 'mutual_failures') {
			return (r.monolithic_error ?? 0) + (r.decomposed_error ?? 0);
		}
		if (laneKey === 'monolithic_only' || laneKey === 'decomposed_only') {
			return Math.max(r.monolithic_error ?? 0, r.decomposed_error ?? 0);
		}
		return Math.abs(r.abs_error_delta ?? 0);
	}

	function compareText(a: string | null | undefined, b: string | null | undefined): number {
		return (a ?? '').localeCompare(b ?? '');
	}

	function exemplarTiebreak(a: PairedExampleRow, b: PairedExampleRow): number {
		return compareText(a.stmt_hash, b.stmt_hash) || compareText(a.evidence_hash, b.evidence_hash);
	}

	function compareExemplarRows(a: PairedExampleRow, b: PairedExampleRow, laneKey: ExemplarKey): number {
		if (exemplarSortMode === 'statement') return exemplarTiebreak(a, b);
		if (exemplarSortMode === 'type') {
			return compareText(a.indra_type, b.indra_type) ||
				compareText(a.agent_names, b.agent_names) ||
				exemplarTiebreak(a, b);
		}
		if (exemplarSortMode === 'verdict') {
			return compareText(verdictDisplay(a.monolithic_verdict), verdictDisplay(b.monolithic_verdict)) ||
				compareText(verdictDisplay(a.decomposed_verdict), verdictDisplay(b.decomposed_verdict)) ||
				(exemplarImpact(b, laneKey) - exemplarImpact(a, laneKey)) ||
				exemplarTiebreak(a, b);
		}
		return (exemplarImpact(b, laneKey) - exemplarImpact(a, laneKey)) || exemplarTiebreak(a, b);
	}

	function rowMatchesExemplarFilter(r: PairedExampleRow, laneKey: ExemplarKey): boolean {
		if (laneKey === 'excluded_by_integrity' || exemplarFilterMode === 'all') return true;
		if (exemplarFilterMode === 'nonoverlap') {
			return laneKey === 'monolithic_only' || laneKey === 'decomposed_only';
		}
		if (laneKey === 'monolithic_only' || laneKey === 'decomposed_only') return false;
		if (exemplarFilterMode === 'verdict_split') return isVerdictSplit(r);
		return r.indra_belief != null && (r.monolithic_error != null || r.decomposed_error != null);
	}

	function sortedExemplarRows(key: ExemplarKey): PairedExampleRow[] {
		return exemplarRows(key)
			.filter((row) => rowMatchesExemplarFilter(row, key))
			.sort((a, b) => compareExemplarRows(a, b, key));
	}

	function exemplarRowIndex(r: PairedExampleRow, laneKey: ExemplarKey): number {
		const identity = exemplarIdentity(r);
		return sortedExemplarRows(laneKey).findIndex((row) => exemplarIdentity(row) === identity);
	}

	function adjacentExemplarRow(r: PairedExampleRow, laneKey: ExemplarKey, offset: -1 | 1): PairedExampleRow | null {
		const rows = sortedExemplarRows(laneKey);
		const index = rows.findIndex((row) => exemplarIdentity(row) === exemplarIdentity(r));
		if (index < 0) return null;
		return rows[index + offset] ?? null;
	}

	function exemplarRowPositionText(r: PairedExampleRow, laneKey: ExemplarKey): string {
		const index = exemplarRowIndex(r, laneKey);
		if (index < 0) return '';
		return `${(index + 1).toLocaleString()}/${sortedExemplarRows(laneKey).length.toLocaleString()}`;
	}

	function firstExemplarLinkLabel(): string {
		return exemplarSortMode === 'impact' ? 'worst' : 'first';
	}

	function csvCell(value: string | number | null | undefined): string {
		const text = value == null ? '' : String(value);
		return /[",\n\r]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
	}

	function exemplarCsvRecord(laneKey: ExemplarKey, r: PairedExampleRow): Array<string | number | null | undefined> {
		return [
			laneTitle(laneKey),
			r.stmt_hash,
			r.evidence_hash,
			r.indra_type,
			r.agent_names,
			r.source_api,
			r.indra_belief,
			r.monolithic_score,
			r.monolithic_verdict,
			r.monolithic_error,
			r.decomposed_score,
			r.decomposed_verdict,
			r.decomposed_error,
			r.abs_error_delta,
			r.excluded_side,
			r.excluded_reason,
			r.monolithic_href,
			r.decomposed_href,
			r.text
		];
	}

	function exemplarCsv(entries: Array<{ laneKey: ExemplarKey; row: PairedExampleRow }>): string {
		const header = [
			'lane',
			'stmt_hash',
			'evidence_hash',
			'indra_type',
			'agents',
			'source_api',
			'indra_belief',
			'monolithic_score',
			'monolithic_verdict',
			'monolithic_error',
			'decomposed_score',
			'decomposed_verdict',
			'decomposed_error',
			'abs_error_delta',
			'excluded_side',
			'excluded_reason',
			'monolithic_href',
			'decomposed_href',
			'evidence_text'
		];
		const lines = [
			header,
			...entries.map(({ laneKey, row }) => exemplarCsvRecord(laneKey, row))
		];
		return `${lines.map((line) => line.map(csvCell).join(',')).join('\n')}\n`;
	}

	function exemplarCsvHref(csv: string): string {
		return `data:text/csv;charset=utf-8,${encodeURIComponent(csv)}`;
	}

	function exemplarCsvFilename(key?: ExemplarKey): string {
		const suffix = key ? key.replace(/_/g, '-') : 'comparison-exemplars';
		return `${w.pair_id}-${suffix}.csv`;
	}

	function comparisonCardSnapshot() {
		return buildPairedComparisonCard(w, {
			generated_at: comparisonCardGeneratedAt,
			pair_href: `/pairs/${w.pair_id}`,
			workflow_status: workflowConsoleStatus()
		});
	}

	function comparisonCardDataHref(kind: 'json' | 'md' | 'schema'): string {
		const card = comparisonCardSnapshot();
		const body = kind === 'json'
			? pairedComparisonCardJson(card)
			: kind === 'md'
				? pairedComparisonCardMarkdown(card)
				: pairedComparisonCardSchemaJson();
		const media = kind === 'json'
			? 'application/json'
			: kind === 'md'
				? 'text/markdown'
				: 'application/schema+json';
		return `data:${media};charset=utf-8,${encodeURIComponent(body)}`;
	}

	function comparisonCardFilename(kind: 'json' | 'md' | 'schema'): string {
		if (kind === 'schema') return `${w.pair_id}-comparison-card.schema.json`;
		return `${w.pair_id}-comparison-card.${kind}`;
	}

	function statementTraceCompareHref(row: PairedExampleRow): string | null {
		if (!w.monolithic?.run_id || !w.decomposed?.run_id) return null;
		const params = new URLSearchParams();
		params.set('run_id', w.monolithic.run_id);
		params.set('compare_run_id', w.decomposed.run_id);
		return `/statements/${row.stmt_hash}?${params.toString()}#compare-evidence-${row.evidence_hash}`;
	}

	function laneCsv(laneKey: ExemplarKey): string {
		return exemplarCsv(sortedExemplarRows(laneKey).map((row) => ({ laneKey, row })));
	}

	function comparisonCsv(): string {
		const entries = activeComparisonExemplarLanes.flatMap((lane) =>
			sortedExemplarRows(lane.key).map((row) => ({ laneKey: lane.key, row }))
		);
		return exemplarCsv(entries);
	}

	function stablePatternNumber(value: number | null | undefined, digits: number): string {
		return value == null || Number.isNaN(value) ? 'na' : value.toFixed(digits);
	}

	function stablePatternToken(value: string | null | undefined): string {
		return encodeURIComponent(value ?? '');
	}

	function exemplarPatternKey(r: PairedExampleRow): string {
		return [
			'v2',
			`indra:${stablePatternNumber(r.indra_belief, 2)}`,
			`m:${stablePatternNumber(r.monolithic_score, 2)}:${stablePatternToken(r.monolithic_verdict)}:${stablePatternNumber(r.monolithic_error, 3)}`,
			`d:${stablePatternNumber(r.decomposed_score, 2)}:${stablePatternToken(r.decomposed_verdict)}:${stablePatternNumber(r.decomposed_error, 3)}`,
			`delta:${stablePatternNumber(r.abs_error_delta, 2)}`,
			`excluded:${stablePatternToken(r.excluded_side)}:${stablePatternToken(r.excluded_reason)}`
		].join('|');
	}

	function exemplarPatternSummary(r: PairedExampleRow): string {
		const parts = [
			`INDRA ${fmtBelief(r.indra_belief)}`,
			`[M] ${fmtBelief(r.monolithic_score)} ${verdictDisplay(r.monolithic_verdict)} err ${errorLabel(r.monolithic_error)}`,
			`[D] ${fmtBelief(r.decomposed_score)} ${verdictDisplay(r.decomposed_verdict)} err ${errorLabel(r.decomposed_error)}`,
			`delta ${fmtDelta(r.abs_error_delta)}`
		];
		if (r.excluded_side || r.excluded_reason) {
			parts.push(`${r.excluded_side ?? 'excluded'} ${r.excluded_reason ?? ''}`.trim());
		}
		return parts.join(' · ');
	}

	function exemplarPatternDiversity(rows: PairedExampleRow[]): string {
		const types = Array.from(new Set(rows.map((row) => row.indra_type).filter(Boolean)));
		if (types.length === 0) return '';
		const shown = types.slice(0, 3).join(', ');
		const more = types.length > 3 ? ` +${types.length - 3} more` : '';
		return `types: ${shown}${more}`;
	}

	function foldedExemplarGroups(key: ExemplarKey): ExemplarFoldGroup[] {
		const rows = sortedExemplarRows(key);
		if (key === 'excluded_by_integrity' || rows.length < 3) {
			return rows.map((row) => ({ kind: 'single' as const, key: exemplarIdentity(row), row }));
		}
		const buckets = new Map<string, PairedExampleRow[]>();
		for (const row of rows) {
			const pattern = exemplarPatternKey(row);
			const bucket = buckets.get(pattern);
			if (bucket) bucket.push(row);
			else buckets.set(pattern, [row]);
		}
		const groups: ExemplarFoldGroup[] = [];
		for (const [pattern, bucket] of buckets) {
			if (bucket.length >= 3) {
				groups.push({ kind: 'pattern', key: pattern, rows: bucket, sample: bucket[0] });
			} else {
				for (const row of bucket) {
					groups.push({ kind: 'single', key: exemplarIdentity(row), row });
				}
			}
		}
		return groups;
	}

	function exemplarIdentity(r: PairedExampleRow): string {
		return `${r.stmt_hash}:${r.evidence_hash}`;
	}

	function exemplarRowAnchor(r: PairedExampleRow, laneKey: ExemplarKey): string {
		return `exemplar-${laneKey}-${r.stmt_hash.slice(0, 8)}-${r.evidence_hash.slice(0, 8)}`;
	}

	function patternShouldOpen(laneKey: ExemplarKey, patternKey: string): boolean {
		return effectiveOpenPatternKeys.has(patternStorageKey(laneKey, patternKey));
	}

	function laneAnchor(key: ExemplarKey): string {
		return EXEMPLAR_LANE_ANCHORS[key];
	}

	function exemplarLaneUnit(key: ExemplarKey): string {
		if (key === 'monolithic_wins' || key === 'decomposed_wins' || key === 'mutual_failures') return 'truth-anchored overlap';
		if (key === 'verdict_disagreements') return 'clean shared overlap';
		if (key === 'monolithic_only' || key === 'decomposed_only') return 'clean non-overlap';
		return 'integrity exclusions';
	}

	function exemplarLaneCountText(key: ExemplarKey): string {
		const shown = sortedExemplarRows(key).length;
		const total = exemplarRows(key).length;
		const prefix = exemplarFilterMode === 'all' || key === 'excluded_by_integrity'
			? `n=${shown.toLocaleString()}`
			: `${shown.toLocaleString()}/${total.toLocaleString()} shown`;
		return `${prefix} · ${exemplarLaneUnit(key)}`;
	}

	function isCloserLane(key: ExemplarKey): boolean {
		return key === 'monolithic_wins' || key === 'decomposed_wins';
	}

	function isVerdictSplit(r: PairedExampleRow): boolean {
		return r.monolithic_verdict !== r.decomposed_verdict;
	}

	function exemplarRowClass(r: PairedExampleRow, laneKey: ExemplarKey): string {
		const classes = ['example-row'];
		if (r.excluded_reason) classes.push(integrityTone(r.excluded_reason));
		if (isCloserLane(laneKey) && isVerdictSplit(r)) classes.push('verdict-split-row');
		return classes.join(' ');
	}

	function relatedExemplarLanes(r: PairedExampleRow, currentKey: ExemplarKey): ExemplarLane[] {
		return (exemplarLaneMemberships.get(exemplarIdentity(r)) ?? []).filter((lane) => lane.key !== currentKey);
	}

	function exemplarLaneReason(r: PairedExampleRow, key: ExemplarKey): string {
		if (key === 'verdict_disagreements') {
			return `[M]=${verdictDisplay(r.monolithic_verdict)} / [D]=${verdictDisplay(r.decomposed_verdict)}`;
		}
		if (key === 'monolithic_wins' || key === 'decomposed_wins') {
			return `delta ${fmtDelta(r.abs_error_delta)}`;
		}
		if (key === 'mutual_failures') {
			return `err [M] ${errorLabel(r.monolithic_error)} / [D] ${errorLabel(r.decomposed_error)}`;
		}
		if (key === 'monolithic_only') return '[D] missing';
		if (key === 'decomposed_only') return '[M] missing';
		return r.excluded_reason ?? 'integrity gated';
	}

	function errorLabel(v: number | null | undefined): string {
		return v == null || Number.isNaN(v) ? 'not scored' : v.toFixed(3);
	}

	function missingNumber(v: number | null | undefined): boolean {
		return v == null || Number.isNaN(v);
	}

	function integrityTone(reason: string | null | undefined): string {
		return reason?.includes('step error') ? 'integrity-error' : 'integrity-omission';
	}

	function massPct(n: number): string {
		if (!w.overlap) return '0%';
		const total =
			w.overlap.monolithic_only_evidences +
			w.overlap.overlap_evidences +
			w.overlap.decomposed_only_evidences +
			excludedMass();
		return total > 0 ? `${(n / total) * 100}%` : '0%';
	}

		function excludedMass(): number {
		if (!w.overlap) return 0;
		return w.overlap.monolithic_step_error_evidences +
			w.overlap.decomposed_step_error_evidences +
			w.overlap.monolithic_missing_aggregate_evidences +
			w.overlap.decomposed_missing_aggregate_evidences +
				w.overlap.monolithic_nonverdict_aggregate_evidences +
				w.overlap.decomposed_nonverdict_aggregate_evidences;
		}

		function massbarLabel(): string {
			if (!w.overlap) return 'overlap massbar';
			return `clean overlap mass: ${w.overlap.monolithic_only_evidences.toLocaleString()} monolithic-only, ${w.overlap.overlap_evidences.toLocaleString()} shared, ${w.overlap.decomposed_only_evidences.toLocaleString()} decomposed-only, ${excludedMass().toLocaleString()} outside comparable metrics`;
		}

	function workflowClockText(): string {
		if (!workflow) return '';
		const basis = workflow.finished_at ?? workflow.started_at ?? workflow.created_at;
		const verb = workflow.finished_at ? 'finished' : workflow.started_at ? 'started' : 'created';
		const parts = [`${verb} ${fmtAge(since(basis))} ago`, `updated ${fmtAge(since(workflow.updated_at))} ago`];
		if (workflow.termination_reason) parts.push(workflow.termination_reason.slice(0, 120));
		return parts.join(' · ');
	}

	function workflowArchProgressText(a: WorkflowArchState): string {
		const total = a.n_evidences_total;
		return total != null ? `${fmtCount(a.n_evidences_done)}/${fmtCount(total)} ev` : `${fmtCount(a.n_evidences_done)} ev`;
	}

	function workflowArchSpendText(a: WorkflowArchState): string {
		const spent = a.cost_so_far_usd == null ? 'pending' : fmtCost(a.cost_so_far_usd);
		return `${spent}/${fmtCost(a.cost_threshold_usd)}`;
	}

	function workflowArchEtaText(a: WorkflowArchState): string {
		if (a.status !== 'running') return a.status === 'queued' || a.status === 'loading' ? 'eta pending' : '-';
		const started = parseTimeMs(a.started_at);
		const total = a.n_evidences_total;
		const done = a.n_evidences_done;
		if (started == null || total == null || total <= 0 || done <= 0) return 'eta pending';
		return fmtEta(((tickNow - started) / Math.max(done, 1)) * (total - done));
	}

	function workflowArchStallText(a: WorkflowArchState): string {
		if (a.status !== 'running') return '';
		const elapsed = since(a.updated_at);
		return elapsed != null && elapsed > WORKFLOW_STALL_MS ? 'no progress >30s' : '';
	}

	function workflowArchUpdatedText(a: WorkflowArchState): string {
		return `${fmtAge(since(a.updated_at))} ago`;
	}

	function workflowArchProgressPercent(a: WorkflowArchState): number {
		const total = a.n_evidences_total;
		if (total == null || total <= 0) return a.status === 'succeeded' ? 100 : 0;
		return Math.max(0, Math.min(100, (a.n_evidences_done / total) * 100));
	}

	function workflowArchProgressStyle(a: WorkflowArchState): string {
		return `--workflow-progress: ${workflowArchProgressPercent(a).toFixed(2)}%;`;
	}

	function workflowArchMeterMax(a: WorkflowArchState): number {
		return Math.max(a.n_evidences_total ?? a.n_evidences_done ?? 0, 1);
	}

	function workflowArchClass(a: WorkflowArchState): string {
		return `workflow-arch workflow-arch-${a.status}${workflowArchStallText(a) ? ' workflow-arch-stalled' : ''}`;
	}

	function workflowArchText(arch: 'monolithic' | 'decomposed'): string {
		const a = workflow?.architectures[arch];
		if (!a) return 'no workflow state';
		if (a.status === 'queued') return `queued · cap ${fmtCost(a.cost_threshold_usd)}`;
		if (a.status === 'loading') return `loading corpus${a.pid ? ` · pid ${a.pid}` : ''} · cap ${fmtCost(a.cost_threshold_usd)}`;
		if (a.status === 'running') {
			const stall = workflowArchStallText(a);
			return `${workflowArchProgressText(a)} · spent ${workflowArchSpendText(a)} · ${workflowArchEtaText(a)}${a.latest_stmt_hash ? ` · latest ${a.latest_stmt_hash.slice(0, 8)}` : ''}${stall ? ` · ${stall}` : ''}${a.pid ? ` · pid ${a.pid}` : ''}`;
		}
		if (a.status === 'succeeded') return `done${a.run_id ? ` · run ${a.run_id.slice(0, 8)}` : ''}${a.duration_s != null ? ` · ${fmtSeconds(a.duration_s)}` : ''}`;
		if (a.status === 'canceled') return a.error ? `canceled · ${a.error.slice(0, 120)}` : 'canceled';
		if (a.status === 'blocked') return 'blocked';
		return a.error ? `failed · ${a.error.slice(0, 120)}` : 'failed';
	}
</script>

<svelte:head><title>{w.pair_id} pair · INDRA Belief</title></svelte:head>

<header>
	<div class="crumb">
		<a href="/">corpus</a><span class="sep"> / </span><strong>pair {w.pair_id}</strong>
	</div>
	<div class="meta">
		<span>{w.runs.length} run{w.runs.length === 1 ? '' : 's'}</span>
	</div>
</header>

<main id="main">
		<section class="pair-head">
			<h1>paired architecture workbench</h1>
			<p>
				<code>{w.pair_id}</code> keeps monolithic and decomposed scoring in one spatial locus.
				The workbench separates overlap evidence from architecture-native diagnostics so deltas are not inferred across non-isomorphic traces.
			</p>
				<nav class="pair-actions" aria-label="paired workflow actions">
					<a class="pair-action-primary" href={workflowRerunHref()}>{workflow ? 'clone pair preflight' : 'new paired run'}</a>
					<a href="/">corpus dashboard</a>
					<a href="#workflow-console">workflow state</a>
					<a href={comparisonCardDataHref('json')} download={comparisonCardFilename('json')}>json card</a>
					<a href={comparisonCardDataHref('md')} download={comparisonCardFilename('md')}>markdown card</a>
					<a href={comparisonCardDataHref('schema')} download={comparisonCardFilename('schema')}>json schema</a>
				</nav>
			</section>

	{#snippet exemplarRow(r: PairedExampleRow, laneKey: ExemplarKey)}
		{@const relatedLanes = relatedExemplarLanes(r, laneKey)}
		{@const previousRow = adjacentExemplarRow(r, laneKey, -1)}
		{@const nextRow = adjacentExemplarRow(r, laneKey, 1)}
		<article id={exemplarRowAnchor(r, laneKey)} class={exemplarRowClass(r, laneKey)}>
			<div class="example-main">
				<a href={`/statements/${r.stmt_hash}`} class="stmt-link"><code>{shortHash(r.stmt_hash)}</code></a>
				<span class="type">{r.indra_type}</span>
				<span class="agents" title={r.agent_names}>{r.agent_names}</span>
				{#if r.source_api}<span class="source">{r.source_api}</span>{/if}
				{#if r.excluded_side}<span class="excluded-side">{r.excluded_side}</span>{/if}
				{#if isCloserLane(laneKey) && isVerdictSplit(r)}<span class="verdict-split-mark">verdict split</span>{/if}
			</div>
			{#if r.excluded_reason}<p class="integrity-reason">{r.excluded_reason}</p>{/if}
			<p class="evidence">{r.text ?? 'no evidence text persisted'}</p>
			{#if relatedLanes.length > 0}
				<div class="lane-memberships" aria-label="same evidence also appears in">
					<span>also in</span>
					{#each relatedLanes as related}
						<span class="lane-membership">
							<a href={`#${laneAnchor(related.key)}`}>{related.title}</a>
							<em>{exemplarLaneReason(r, related.key)}</em>
						</span>
					{/each}
				</div>
			{/if}
			<dl class="example-metrics">
				<div><dt>INDRA</dt><dd>{fmtBelief(r.indra_belief)}</dd></div>
				<div><dt>[M]</dt><dd>{fmtBelief(r.monolithic_score)} <span class={`verdict-chip verdict-${r.monolithic_verdict ?? 'missing'}`}>{verdictDisplay(r.monolithic_verdict)}</span> <em class:metric-missing={missingNumber(r.monolithic_error)}>err {errorLabel(r.monolithic_error)}</em></dd></div>
				<div><dt>[D]</dt><dd>{fmtBelief(r.decomposed_score)} <span class={`verdict-chip verdict-${r.decomposed_verdict ?? 'missing'}`}>{verdictDisplay(r.decomposed_verdict)}</span> <em class:metric-missing={missingNumber(r.decomposed_error)}>err {errorLabel(r.decomposed_error)}</em></dd></div>
				<div class="delta-metric"><dt>error delta</dt><dd>{fmtDelta(r.abs_error_delta)}</dd></div>
			</dl>
			<div class="example-links">
				<a href={`#${exemplarRowAnchor(r, laneKey)}`}>row anchor</a>
				{#if statementTraceCompareHref(r)}
					<a href={statementTraceCompareHref(r)}>compare traces</a>
				{/if}
				<a href={r.monolithic_href}>open [M] trace</a>
				<a href={r.decomposed_href}>open [D] trace</a>
			</div>
			{#if sortedExemplarRows(laneKey).length > 1}
				<nav class="example-stepper" aria-label={`${laneTitle(laneKey)} row stepper`}>
					{#if previousRow}
						<a href={`#${exemplarRowAnchor(previousRow, laneKey)}`} onclick={() => openPatternForRow(laneKey, previousRow)}>prev</a>
					{:else}
						<span class="step-disabled">prev</span>
					{/if}
					<span class="step-position">{exemplarRowPositionText(r, laneKey)}</span>
					{#if nextRow}
						<a href={`#${exemplarRowAnchor(nextRow, laneKey)}`} onclick={() => openPatternForRow(laneKey, nextRow)}>next</a>
					{:else}
						<span class="step-disabled">next</span>
					{/if}
				</nav>
			{/if}
		</article>
	{/snippet}

	{#snippet exemplarLane(lane: ExemplarLane)}
		{@const laneRows = sortedExemplarRows(lane.key)}
		<section id={laneAnchor(lane.key)} class={`example-lane example-lane-${lane.key}`}>
			<div class="lane-head">
				<div class="lane-head-main">
					<h3>{lane.title} <span class="lane-n">{exemplarLaneCountText(lane.key)}</span></h3>
					<p>{lane.scope}</p>
				</div>
				<nav class="lane-tools" aria-label={`${lane.title} lane tools`}>
					{#if laneRows[0]}
						<a href={`#${exemplarRowAnchor(laneRows[0], lane.key)}`} onclick={() => openPatternForRow(lane.key, laneRows[0])}>{firstExemplarLinkLabel()}</a>
						<a href={exemplarCsvHref(laneCsv(lane.key))} download={exemplarCsvFilename(lane.key)}>csv lane</a>
					{/if}
				</nav>
			</div>
			{#if laneRows.length > 0}
				<details class="lane-table">
					<summary>table</summary>
					<div class="table-wrap">
						<table>
							<thead>
								<tr>
									<th>row</th>
									<th>stmt</th>
									<th>type</th>
									<th>[M]</th>
									<th>[D]</th>
									<th>delta</th>
									<th>basis</th>
								</tr>
							</thead>
							<tbody>
								{#each laneRows as r, index}
									<tr>
										<td class="num">{index + 1}</td>
										<td><a href={`#${exemplarRowAnchor(r, lane.key)}`} onclick={() => openPatternForRow(lane.key, r)}><code>{shortHash(r.stmt_hash)}</code></a></td>
										<td>{r.indra_type}</td>
										<td>{fmtBelief(r.monolithic_score)} {verdictDisplay(r.monolithic_verdict)}</td>
										<td>{fmtBelief(r.decomposed_score)} {verdictDisplay(r.decomposed_verdict)}</td>
										<td class="delta-cell">{fmtDelta(r.abs_error_delta)}</td>
										<td>{r.excluded_reason ?? exemplarLaneReason(r, lane.key)}</td>
									</tr>
								{/each}
							</tbody>
						</table>
					</div>
				</details>
			{/if}
			<div class="example-list">
				{#each foldedExemplarGroups(lane.key) as group (group.key)}
					{#if group.kind === 'pattern'}
						<details
							class="example-pattern"
							open={patternShouldOpen(lane.key, group.key)}
							ontoggle={(event) => updatePatternOpen(lane.key, group.key, event)}
						>
							<summary>
								<strong>{group.rows.length.toLocaleString()} rows render to this metric tuple (display precision)</strong>
								<span>{exemplarPatternSummary(group.sample)}</span>
								{#if exemplarPatternDiversity(group.rows)}
									<em>{exemplarPatternDiversity(group.rows)}</em>
								{/if}
							</summary>
							<div class="example-list example-list-folded">
								{#each group.rows as r}
									{@render exemplarRow(r, lane.key)}
								{/each}
							</div>
						</details>
					{:else}
						{@render exemplarRow(group.row, lane.key)}
					{/if}
				{/each}
			</div>
		</section>
	{/snippet}

			<section id="workflow-console" class={`workflow-state workflow-state-${workflowConsoleStatus()}`} aria-label="paired workflow console">
				<div class="workflow-head">
					<h2>workflow console <span>{workflowConsoleStatus()}</span></h2>
					<nav class="workflow-actions" aria-label="workflow controls">
						<a class="workflow-rerun" href={workflowRerunHref()}>{workflow ? 'clone source preflight' : 'new paired run'}</a>
						{#if workflowIsActive()}
							<button class="workflow-cancel" type="button" onclick={cancelWorkflow}>cancel workflow</button>
						{/if}
					</nav>
				</div>
				<p class="scope-note">{workflowConsoleText()}</p>
				{#if workflow}
					<p class="workflow-clock">{workflowClockText()}</p>
					<dl class="workflow-facts">
						<div><dt>source</dt><dd title={workflow.dataset_path}>{workflow.source_dump_id}</dd></div>
						<div><dt>model</dt><dd>{workflow.model}</dd></div>
						<div><dt>scorer</dt><dd>{workflow.scorer_version}</dd></div>
						<div><dt>cap</dt><dd>{fmtCost(workflow.total_cost_threshold_usd)}</dd></div>
					</dl>
					{#if workflow.termination_reason}
						<p class="workflow-terminal">{workflow.termination_reason}</p>
					{/if}
				<div class="workflow-arches">
					{#each WORKFLOW_ARCHES as arch}
						{@const archState = workflow.architectures[arch]}
						<div class={workflowArchClass(archState)} aria-label={`${arch} workflow ${archState.status}`}>
							<div class="workflow-arch-head">
								<strong>{archMark(arch)} {arch}</strong>
								<span>{archState.status}</span>
							</div>
							<div
								class="workflow-progress-rail"
								role="progressbar"
								aria-label={`${arch} workflow progress ${workflowArchProgressText(archState)}`}
								aria-valuemin="0"
								aria-valuemax={workflowArchMeterMax(archState)}
								aria-valuenow={archState.n_evidences_done}
								style={workflowArchProgressStyle(archState)}
							><i></i></div>
							<dl class="workflow-arch-facts">
								<div><dt>progress</dt><dd>{workflowArchProgressText(archState)}</dd></div>
								<div><dt>spend</dt><dd>{workflowArchSpendText(archState)}</dd></div>
								<div><dt>eta</dt><dd>{workflowArchEtaText(archState)}</dd></div>
								<div><dt>updated</dt><dd>{workflowArchUpdatedText(archState)}</dd></div>
							</dl>
							<em>{workflowArchText(arch)}</em>
					</div>
					{/each}
				</div>
				{:else}
					<p class="workflow-idle-note">Use the preflight handoff to choose a corpus and re-estimate [M]/[D] costs; this page never starts API spend without that preview.</p>
				{/if}
			</section>

		<section class="run-axis" aria-label="paired run axis">
		<div class="run-pane run-pane-monolithic" aria-label={runPaneLabel(w.monolithic, 'monolithic')}>
			<div class="run-pane-head">
				<h2><span>{archMark(w.monolithic?.architecture ?? 'monolithic')}</span> {architectureLabel(w.monolithic?.architecture, 'monolithic')}</h2>
				{#if w.monolithic}
					<span class={`run-status run-status-${runStatusClass(w.monolithic.status)}`}>{w.monolithic.status}</span>
				{/if}
			</div>
			{#if w.monolithic}
				<a href={`/runs/${w.monolithic.run_id}`} class="run-link" title={w.monolithic.run_id}><span>run</span><code>{shortHash(w.monolithic.run_id)}</code></a>
				<dl class="run-facts">
					<div><dt>model</dt><dd>{w.monolithic.model_id_default ?? '-'}</dd></div>
					<div><dt>evidence</dt><dd>{fmtCount(w.monolithic.n_evidences)}</dd></div>
					<div><dt>cost</dt><dd>{runCost(w.monolithic)}</dd></div>
					<div><dt>duration</dt><dd>{fmtSeconds(w.monolithic.duration_s)}</dd></div>
				</dl>
			{:else}
				<p class="not-defined">No monolithic run has this paired group id.</p>
			{/if}
		</div>
			<div class="axis-label" role="group" aria-label={massbarLabel()}>
				{#if w.overlap}
					<strong>{fmtCount(w.overlap.overlap_evidences)}</strong>
					<span>clean shared evidence</span>
					<a class="axis-denominator-link" href={ledgerHref('clean_shared_aggregate_verdict_evidence')}>base denominator</a>
					<div class="massbar">
						<i class="massbar-m" style={`width: ${massPct(w.overlap.monolithic_only_evidences)}`}></i>
					<i class="massbar-o" style={`width: ${massPct(w.overlap.overlap_evidences)}`}></i>
						<i class="massbar-d" style={`width: ${massPct(w.overlap.decomposed_only_evidences)}`}></i>
						<i class="massbar-x" style={`width: ${massPct(excludedMass())}`}></i>
					</div>
					<dl class="massbar-key" aria-label="overlap partition counts">
						<div class="key-m"><dt>[M]</dt><dd>{fmtCount(w.overlap.monolithic_only_evidences)} only</dd></div>
						<div class="key-o"><dt>both</dt><dd>{fmtCount(w.overlap.overlap_evidences)} shared</dd></div>
						<div class="key-d"><dt>[D]</dt><dd>{fmtCount(w.overlap.decomposed_only_evidences)} only</dd></div>
						<div class="key-x"><dt>out</dt><dd>{fmtCount(excludedMass())} outside</dd></div>
					</dl>
				{:else}
					<span>overlap</span>
				{/if}
		</div>
		<div class="run-pane run-pane-decomposed" aria-label={runPaneLabel(w.decomposed, 'decomposed')}>
			<div class="run-pane-head">
				<h2><span>{archMark(w.decomposed?.architecture ?? 'decomposed')}</span> {architectureLabel(w.decomposed?.architecture, 'decomposed')}</h2>
				{#if w.decomposed}
					<span class={`run-status run-status-${runStatusClass(w.decomposed.status)}`}>{w.decomposed.status}</span>
				{/if}
			</div>
			{#if w.decomposed}
				<a href={`/runs/${w.decomposed.run_id}`} class="run-link" title={w.decomposed.run_id}><span>run</span><code>{shortHash(w.decomposed.run_id)}</code></a>
				<dl class="run-facts">
					<div><dt>model</dt><dd>{w.decomposed.model_id_default ?? '-'}</dd></div>
					<div><dt>evidence</dt><dd>{fmtCount(w.decomposed.n_evidences)}</dd></div>
					<div><dt>cost</dt><dd>{runCost(w.decomposed)}</dd></div>
					<div><dt>duration</dt><dd>{fmtSeconds(w.decomposed.duration_s)}</dd></div>
				</dl>
			{:else}
				<p class="not-defined">No decomposed run has this paired group id.</p>
			{/if}
		</div>
	</section>

	{#if w.overlap}
		<section class="overlap" aria-label="overlap accounting">
			<h2>overlap accounting first</h2>
				<p class="scope-note">
					{w.overlap.overlap_evidences.toLocaleString()} non-error aggregate verdict rows overlap across {w.overlap.overlap_statements.toLocaleString()} statements.
					Comparable metrics use this clean overlap only by default.
				</p>
				<p class="overlap-ledger-note">
					<a class="ledger-ref" href={ledgerHref('side_aggregate_evidence_rows')}>source -> overlap accounting</a>
					<a class="ledger-ref" href={ledgerHref('clean_shared_aggregate_verdict_evidence')}>base -> paired metrics</a>
				</p>
			{#if overlapIntegrityText}
				<p class="integrity-note">{overlapIntegrityText}</p>
			{/if}
			{#if overlapTraceWarningText}
				<p class="trace-warning-note">{overlapTraceWarningText}</p>
			{/if}
		</section>
	{/if}

	{#if visibleIntegrityExemplarLanes.length > 0}
		<section class="exemplars integrity-exemplars" aria-label="integrity exclusion examples">
			<h2>integrity exclusions</h2>
			<p class="scope-note">Rows counted above but withheld from comparable metrics because one side cannot provide a clean aggregate verdict.</p>
			{#each visibleIntegrityExemplarLanes as lane}
				{@render exemplarLane(lane)}
			{/each}
		</section>
	{/if}

		{#if w.denominator_ledger.length > 0}
			<section class="denominator-ledger" aria-label="panel denominator ledger">
				<h2>denominator ledger <span>: panel populations</span></h2>
				<p class="scope-note">Tree order is <LedgerKindChip kind="denominator_base" variant="inline" /> -> metric kind -> panel population; counts with different units or kinds are not interchangeable.</p>
				{#if pairedDenominatorRows.length > 0}
					<h3>paired denominators</h3>
					<div class="table-wrap">
						<table class:no-outside={!showPairedOutsideColumn}>
							<caption class="ledger-column-policy">{outsidePolicyText(pairedOutsidePolicy)}</caption>
							<thead>
								<tr>
									<th>node</th>
									<th>scope</th>
									<th>metric kind</th>
									<th>unit</th>
									<th class="num">panel n</th>
									<th class="num">[M]</th>
									<th class="num">[D]</th>
									<th class="num">shared</th>
									{#if showPairedOutsideColumn}<th class="num">outside</th>{/if}
									<th>basis</th>
								</tr>
							</thead>
						<tbody>
							{#each pairedDenominatorRows as row, i}
								<tr id={ledgerAnchor(row.key)} class={ledgerRowClass(row)}>
										<td class="ledger-panel-cell">
											<LedgerRoleMark role={row.ledger_role} />
											<span class="ledger-panel-title">{row.panel}</span>
											{#if ledgerParentText(pairedDenominatorRows, i)}<span class={`ledger-parent ${ledgerParentIsContinued(pairedDenominatorRows, i) ? 'ledger-parent-continued' : ''}`}>{ledgerParentText(pairedDenominatorRows, i)}</span>{/if}
											<a class="ledger-back" href="#comparable-metrics">back to metric callsite</a>
										</td>
										<td><LedgerScopeChip applicability={row.applicability} /></td>
										<td><LedgerKindChip kind={row.metric_kind} /></td>
										<td>{row.unit}</td>
										<td class={ledgerCountClass(row, 'panel')}>{ledgerCountText(row, row.denominator_n, 'panel')}</td>
										<td class={ledgerCountClass(row, 'monolithic')}>{ledgerCountText(row, row.monolithic_n, 'monolithic')}</td>
										<td class={ledgerCountClass(row, 'decomposed')}>{ledgerCountText(row, row.decomposed_n, 'decomposed')}</td>
										<td class={ledgerCountClass(row, 'shared')}>{ledgerCountText(row, row.overlap_n, 'shared')}</td>
										{#if showPairedOutsideColumn}<td class={ledgerCountClass(row, 'outside')}>{ledgerCountText(row, row.excluded_n, 'outside')}</td>{/if}
										<td class="ledger-basis">{row.reason}</td>
									</tr>
								{/each}
							</tbody>
						</table>
					</div>
				{/if}
				{#if archNativeLedgerRows.length > 0}
					<h3>architecture-native populations</h3>
					<p class="native-ledger-note">Not part of the paired tree; native units remain architecture-conditioned.</p>
					<div class="table-wrap">
						<table>
							<thead>
								<tr>
									<th>native panel</th>
									<th>scope</th>
									<th>metric kind</th>
									<th>native unit</th>
									<th class="num">n</th>
									<th>native side</th>
									<th class="num">issue count</th>
									<th>basis</th>
								</tr>
							</thead>
						<tbody>
							{#each archNativeLedgerRows as row, i}
								<tr id={ledgerAnchor(row.key)} class={ledgerRowClass(row)}>
										<td class="ledger-panel-cell">
											<LedgerRoleMark role={row.ledger_role} />
											<span class="ledger-panel-title">{row.panel}</span>
											{#if ledgerParentText(archNativeLedgerRows, i)}<span class={`ledger-parent ${ledgerParentIsContinued(archNativeLedgerRows, i) ? 'ledger-parent-continued' : ''}`}>{ledgerParentText(archNativeLedgerRows, i)}</span>{/if}
											<a class="ledger-back" href="#comparable-metrics">back to metric callsite</a>
										</td>
										<td><LedgerScopeChip applicability={row.applicability} /></td>
										<td><LedgerKindChip kind={row.metric_kind} /></td>
										<td>{row.unit}</td>
										<td class="num">{fmtLedgerCount(row.denominator_n)}</td>
										<td>{archNativeSide(row)}</td>
										<td class="num">{fmtLedgerCount(row.excluded_n)}</td>
										<td class="ledger-basis">{row.reason}</td>
									</tr>
								{/each}
							</tbody>
						</table>
					</div>
				{/if}
			</section>
		{/if}

		{#if w.resource_frontier}
			{@const frontier = w.resource_frontier}
			{@const frontierCostWinner = metricWinnerByPositiveLower(frontier.monolithic.cost_per_evidence_usd, frontier.decomposed.cost_per_evidence_usd)}
			{@const frontierWallWinner = metricWinnerByPositiveLower(frontier.monolithic.wall_seconds_per_evidence, frontier.decomposed.wall_seconds_per_evidence)}
			{@const frontierLatencyWinner = resourceTelemetryWinner(
				frontier.monolithic.clean_overlap_latency_mean_ms,
				frontier.decomposed.clean_overlap_latency_mean_ms,
				frontier.monolithic.clean_overlap_latency_observed_n,
				frontier.decomposed.clean_overlap_latency_observed_n,
				frontier.monolithic.clean_overlap_n
			)}
			{@const frontierTokenWinner = resourceTelemetryWinner(
				frontier.monolithic.clean_overlap_tokens_per_observed_evidence,
				frontier.decomposed.clean_overlap_tokens_per_observed_evidence,
				frontier.monolithic.clean_overlap_tokens_observed_n,
				frontier.decomposed.clean_overlap_tokens_observed_n,
				frontier.monolithic.clean_overlap_n
			)}
			{@const frontierMaeWinner = metricWinnerByLower(frontier.monolithic.mae, frontier.decomposed.mae)}
			<section id="resource-frontier" class="resource-frontier" aria-label="cost latency frontier">
				<div class="frontier-head">
					<h2>cost/latency frontier <span>whole-run spend beside clean-overlap counters</span></h2>
					<p class="scope-note">Spend and wall clock use each side's full run. Latency, tokens, and MAE use the clean shared overlap shown in the denominator ledger.</p>
				</div>
				<div class="frontier-grid">
					{#each frontierRows() as row}
						<article class={`frontier-arch frontier-arch-${row.architecture}`}>
							<h3>{archMark(row.architecture)} {row.architecture}</h3>
							<dl class="frontier-rows">
								<div class={frontierRowClass(row.architecture, frontierCostWinner)}>
									<dt>whole-run cost/ev</dt>
									<dd>{fmtUnitCost(row.cost_per_evidence_usd)} <em>{frontierCostBasis(row)} over {fmtCount(row.n_evidences)} ev</em></dd>
								</div>
								<div class={frontierRowClass(row.architecture, frontierWallWinner)}>
									<dt>wall time/ev</dt>
									<dd>{fmtSeconds(row.wall_seconds_per_evidence)} <em>run {fmtSeconds(row.duration_s)}</em></dd>
								</div>
								<div class={frontierRowClass(row.architecture, frontierLatencyWinner)}>
									<dt>clean latency</dt>
									<dd>{fmtMs(row.clean_overlap_latency_mean_ms)} <em>{fmtCount(row.clean_overlap_latency_observed_n)}/{fmtCount(row.clean_overlap_n)} rows report latency</em></dd>
								</div>
								<div class={frontierRowClass(row.architecture, frontierTokenWinner)}>
									<dt>tokens/reported ev</dt>
									<dd>{fmtRate(row.clean_overlap_tokens_per_observed_evidence)} <em>{fmtCount(row.clean_overlap_tokens_observed_n)}/{fmtCount(row.clean_overlap_n)} rows report tokens · {fmtCount(row.clean_overlap_tokens_total)} total</em></dd>
								</div>
								<div class={frontierRowClass(row.architecture, frontierMaeWinner)}>
									<dt>MAE</dt>
									<dd>{frontierQuality(row)} <em>truth n={fmtCount(row.truth_overlap_n)}</em></dd>
								</div>
							</dl>
						</article>
					{/each}
				</div>
				<div class="frontier-scopes" aria-label="frontier denominator scopes">
					<p><strong>spend</strong> {frontier.spend_scope}</p>
					<p><strong>latency</strong> {frontier.latency_scope} · <a class="ledger-ref" href={ledgerHref('resource_counter_metric')}>resource denominator</a></p>
					<p><strong>quality</strong> {frontier.quality_scope} · <a class="ledger-ref" href={ledgerHref('truth_anchored_overlap_evidence')}>truth denominator</a></p>
					<p><strong>guardrail</strong> lower resource use is not a quality signal; frontier rows show tradeoff posture, not a scalar winner.</p>
				</div>
			</section>
		{/if}

	{#if w.not_defined_reason}
		<section class="not-defined-block">
			<h2>comparison not defined</h2>
			<p>{w.not_defined_reason}.</p>
			{#if w.runs.length > 0}
				<div class="table-wrap">
					<table>
						<thead><tr><th>run</th><th>arch</th><th>status</th><th>started</th><th class="num">evidence</th></tr></thead>
						<tbody>
							{#each w.runs as r}
								<tr>
									<td><a href={`/runs/${r.run_id}`}><code>{shortHash(r.run_id)}</code></a></td>
									<td>{archMark(r.architecture)} {r.architecture}</td>
									<td>{r.status}</td>
									<td>{r.started_at}</td>
									<td class="num">{fmtCount(r.n_evidences)}</td>
								</tr>
							{/each}
						</tbody>
					</table>
				</div>
			{/if}
		</section>
	{:else if w.comparable}
		{@const verdictMonolithicSupported = supportedCount('monolithic')}
		{@const verdictDecomposedSupported = supportedCount('decomposed')}
		{@const verdictSupportAgreementN = w.comparable.both_correct_n + w.comparable.both_incorrect_n}
		{@const verdictSupportSplitN = w.comparable.monolithic_only_correct_n + w.comparable.decomposed_only_correct_n}
		{@const verdictRecodingDivergenceN = Math.max(0, verdictSupportAgreementN - w.comparable.verdict_agreement_n)}
		{@const verdictMonolithicUnsupported = Math.max(0, w.comparable.n_overlap - verdictMonolithicSupported)}
		{@const verdictDecomposedUnsupported = Math.max(0, w.comparable.n_overlap - verdictDecomposedSupported)}
		{@const verdictPairGroups = groupVerdictPairs(w.comparable.verdict_label_pairs)}
		{@const maeWinner = metricWinnerByLower(w.comparable.monolithic_mae, w.comparable.decomposed_mae)}
				{@const biasWinner = metricWinnerByAbsLower(w.comparable.monolithic_bias, w.comparable.decomposed_bias)}
				{@const latencyWinner = resourceTelemetryWinner(
					w.comparable.monolithic_latency_mean_ms,
					w.comparable.decomposed_latency_mean_ms,
					w.comparable.monolithic_latency_observed_n,
					w.comparable.decomposed_latency_observed_n,
					w.comparable.n_overlap
				)}
				{@const monolithicTokensPerObserved = metricRate(w.comparable.monolithic_tokens_total, w.comparable.monolithic_tokens_observed_n)}
				{@const decomposedTokensPerObserved = metricRate(w.comparable.decomposed_tokens_total, w.comparable.decomposed_tokens_observed_n)}
				{@const tokensWinner = resourceTelemetryWinner(
					monolithicTokensPerObserved,
					decomposedTokensPerObserved,
					w.comparable.monolithic_tokens_observed_n,
					w.comparable.decomposed_tokens_observed_n,
					w.comparable.n_overlap
				)}

			<section id="comparable-metrics" class="comparable" aria-label="comparable metrics">
			<h2>shared metrics by kind <span>overlap only · not interchangeable</span></h2>
			<section id="metric-verdict-labels" class="verdict-fusion" aria-label={`verdict label agreement over ${fmtCount(w.comparable.n_overlap)} clean shared verdict rows`}>
				<span id="verdict-outcomes" class="legacy-anchor" aria-hidden="true"></span>
				<div class="verdict-fusion-head">
					<div>
						<div class="fusion-kicker">arch-blind · clean shared verdict rows</div>
						<h3>[M] vs [D] verdict agreement <span>n={fmtCount(w.comparable.n_overlap)} · exact labels plus support state</span></h3>
						<p class="verdict-definition">Exact aggregate labels are counted only in the match rate. The matrix recodes every non-supported label into not supported.</p>
						<p class="verdict-guardrail">inter-architecture agreement · not accuracy · no gold label</p>
					</div>
				</div>
				<div class="verdict-comparison-strip" aria-label="exact label agreement versus recoded support-state agreement">
					<div class="verdict-compare-term">
						<span>exact label match</span>
						<strong>{fmtRatio(w.comparable.verdict_agreement_n, w.comparable.n_overlap)}</strong>
					</div>
					<span class:verdict-operator-diverges={verdictRecodingDivergenceN > 0} class="verdict-operator">recode</span>
					<div class="verdict-compare-term verdict-recode-term">
						<span>recoded merge</span>
						<strong>+{fmtCount(verdictRecodingDivergenceN)}</strong>
					</div>
					<span class="verdict-operator">=</span>
					<div class="verdict-compare-term">
						<span>same support state</span>
						<strong>{fmtRatio(verdictSupportAgreementN, w.comparable.n_overlap)}</strong>
					</div>
					{#if verdictRecodingDivergenceN > 0}
						<p>{fmtCount(verdictRecodingDivergenceN)} exact-divergent row{verdictRecodingDivergenceN === 1 ? '' : 's'} fold into same-state agreement after recoding</p>
					{/if}
				</div>
				<div class="verdict-fusion-grid">
					<section class="verdict-marginals" aria-label="support-state marginals by architecture">
						<h4>matrix marginals</h4>
						<div class="marginal-row marginal-row-monolithic">
							<span class="marginal-label">[M] supported row</span>
							<strong>{fmtRatio(verdictMonolithicSupported, w.comparable.n_overlap)}</strong>
							<em>{fmtCount(w.comparable.both_correct_n)} both + {fmtCount(w.comparable.monolithic_only_correct_n)} [M] only</em>
							<em>{fmtRatio(verdictMonolithicUnsupported, w.comparable.n_overlap)} [M] non-supported exact labels below</em>
						</div>
						<div class="marginal-row marginal-row-decomposed">
							<span class="marginal-label">[D] supported column</span>
							<strong>{fmtRatio(verdictDecomposedSupported, w.comparable.n_overlap)}</strong>
							<em>{fmtCount(w.comparable.both_correct_n)} both + {fmtCount(w.comparable.decomposed_only_correct_n)} [D] only</em>
							<em>{fmtRatio(verdictDecomposedUnsupported, w.comparable.n_overlap)} [D] non-supported exact labels below</em>
						</div>
						<p>{fmtRatio(verdictSupportAgreementN, w.comparable.n_overlap)} same support state · {fmtRatio(verdictSupportSplitN, w.comparable.n_overlap)} split support state</p>
					</section>
					<section class="verdict-matrix-panel" aria-label="joint support-state outcome matrix">
						<h4>joint outcome matrix</h4>
						<div class="outcome-matrix-scroll">
							<div class="outcome-matrix" role="table" aria-label="2x2 verdict support matrix">
								<div class="outcome-corner" aria-hidden="true"></div>
								<div class="outcome-axis" role="columnheader">[D] supported</div>
								<div class="outcome-axis" role="columnheader">[D] not supported</div>
								<div class="outcome-axis outcome-axis-row" role="rowheader">[M] supported</div>
								<div class="outcome-cell outcome-cell-diagonal" role="cell"><strong>{fmtCount(w.comparable.both_correct_n)}</strong><span>both supported</span></div>
								<div class="outcome-cell outcome-cell-split outcome-cell-m-only" role="cell"><strong>{fmtCount(w.comparable.monolithic_only_correct_n)}</strong><span>[M] only</span></div>
								<div class="outcome-axis outcome-axis-row" role="rowheader">[M] not supported</div>
								<div class="outcome-cell outcome-cell-split outcome-cell-d-only" role="cell"><strong>{fmtCount(w.comparable.decomposed_only_correct_n)}</strong><span>[D] only</span></div>
								<div class="outcome-cell outcome-cell-diagonal" role="cell"><strong>{fmtCount(w.comparable.both_incorrect_n)}</strong><span>neither supported</span></div>
							</div>
						</div>
						<p class="outcome-note"><span><i class="legend-swatch legend-diagonal"></i>diagonal support-state agreement</span><span><i class="legend-swatch legend-m-only"></i>[M]-only split</span><span class:legend-zero={w.comparable.decomposed_only_correct_n === 0}><i class="legend-swatch legend-d-only"></i>[D]-only split{w.comparable.decomposed_only_correct_n === 0 ? ' · 0' : ''}</span></p>
					</section>
				</div>
				<section class="verdict-taxonomy" aria-label="exact verdict labels inside support-state buckets">
					<h4>exact labels inside support-state buckets</h4>
					<div class="verdict-taxonomy-grid">
						{#each verdictPairGroups as group}
							<section class={`verdict-taxonomy-group ${supportCellClass(group.support_cell)}`}>
								<div class="verdict-taxonomy-group-head">
									<span>{supportCellLabel(group.support_cell)}</span>
									<strong>n={fmtCount(group.n)}</strong>
									<em>{group.exactDivergences > 0 ? `${fmtCount(group.exactDivergences)} exact-divergent` : 'exact match'}</em>
								</div>
								<div class="verdict-taxonomy-group-rows">
									{#each group.rows as pair}
										<div class={`verdict-taxonomy-row ${exactPairClass(pair)}`}>
											<span class="taxonomy-cell">{exactPairLabel(pair)}</span>
											<span class="taxonomy-arch">[M] {verdictDisplay(pair.monolithic_verdict)}</span>
											<span class="taxonomy-arch">[D] {verdictDisplay(pair.decomposed_verdict)}</span>
											<strong>{fmtCount(pair.n)}</strong>
										</div>
									{/each}
								</div>
							</section>
						{/each}
					</div>
					<p>Group headers are recoded support-state buckets; child rows keep exact aggregate labels.</p>
				</section>
				<p class="verdict-fusion-note">
					<span class="metric-n">n={fmtCount(w.comparable.n_overlap)}</span> · <a class="ledger-ref" href={ledgerHref('exact_label_agreement_metric')}>kind -> <LedgerKindChip kind="arch_arch_exact_label" variant="inline" /></a> · <a class="ledger-ref" href={ledgerHref('support_state_recode_metric')}>kind -> <LedgerKindChip kind="arch_arch_support_recode" variant="inline" /></a> · <a class="ledger-ref" href={ledgerHref('clean_shared_aggregate_verdict_evidence')}>base -> clean shared verdict rows</a>
				</p>
			</section>
				<div class="metric-grid">
					<article id="metric-mae-vs-indra" class="metric">
						<h3>MAE vs INDRA <span>n={fmtCount(w.comparable.n_truth_overlap)} · truth anchored · kind: <LedgerKindChip kind="arch_indra_residual" variant="inline" /></span></h3>
						<dl class="metric-rows" aria-label="MAE by architecture">
							<div class={metricRowClass('monolithic', maeWinner)}>
								<dt>[M]</dt>
								<dd>{fmtMaybe(w.comparable.monolithic_mae)} {#if metricWinnerLabel('monolithic', maeWinner, 'closer')}<em>{metricWinnerLabel('monolithic', maeWinner, 'closer')}</em>{/if}</dd>
							</div>
							<div class={metricRowClass('decomposed', maeWinner)}>
								<dt>[D]</dt>
								<dd>{fmtMaybe(w.comparable.decomposed_mae)} {#if metricWinnerLabel('decomposed', maeWinner, 'closer')}<em>{metricWinnerLabel('decomposed', maeWinner, 'closer')}</em>{/if}</dd>
							</div>
						</dl>
						<p>lower error is closer to INDRA</p>
						<p><span class="metric-n">n={fmtCount(w.comparable.n_truth_overlap)}</span> · <a class="ledger-ref" href={ledgerHref('truth_anchored_overlap_evidence')}>kind -> <LedgerKindChip kind="arch_indra_residual" variant="inline" /></a> · base -> truth-anchored overlap</p>
					</article>
					<article id="metric-bias-vs-indra" class="metric">
						<h3>bias vs INDRA <span>n={fmtCount(w.comparable.n_truth_overlap)} · truth anchored · kind: <LedgerKindChip kind="arch_indra_residual" variant="inline" /></span></h3>
						<dl class="metric-rows" aria-label="bias by architecture">
							<div class={metricRowClass('monolithic', biasWinner)}>
								<dt>[M]</dt>
								<dd>{fmtDelta(w.comparable.monolithic_bias)} {#if metricWinnerLabel('monolithic', biasWinner, 'closer to 0')}<em>{metricWinnerLabel('monolithic', biasWinner, 'closer to 0')}</em>{/if}</dd>
							</div>
							<div class={metricRowClass('decomposed', biasWinner)}>
								<dt>[D]</dt>
								<dd>{fmtDelta(w.comparable.decomposed_bias)} {#if metricWinnerLabel('decomposed', biasWinner, 'closer to 0')}<em>{metricWinnerLabel('decomposed', biasWinner, 'closer to 0')}</em>{/if}</dd>
							</div>
						</dl>
						<p>positive means above INDRA</p>
						<p><span class="metric-n">n={fmtCount(w.comparable.n_truth_overlap)}</span> · <a class="ledger-ref" href={ledgerHref('truth_anchored_overlap_evidence')}>kind -> <LedgerKindChip kind="arch_indra_residual" variant="inline" /></a> · base -> truth-anchored overlap</p>
					</article>
					<article id="metric-mean-latency" class="metric">
						<h3>mean latency <span>n={fmtCount(w.comparable.n_overlap)} · clean shared · kind: <LedgerKindChip kind="arch_arch_resource" variant="inline" /></span></h3>
						<dl class="metric-rows" aria-label="mean latency by architecture">
							<div class={metricRowClass('monolithic', latencyWinner)}>
								<dt>[M]</dt>
								<dd>{fmtMs(w.comparable.monolithic_latency_mean_ms)} {#if zeroMetricLabel(w.comparable.monolithic_latency_mean_ms)}<em>{zeroMetricLabel(w.comparable.monolithic_latency_mean_ms)}</em>{:else if lowerRatioLabel('monolithic', latencyWinner, w.comparable.monolithic_latency_mean_ms, w.comparable.decomposed_latency_mean_ms, 'faster')}<em>{lowerRatioLabel('monolithic', latencyWinner, w.comparable.monolithic_latency_mean_ms, w.comparable.decomposed_latency_mean_ms, 'faster')}</em>{/if}<em>{telemetryCoverageText(w.comparable.monolithic_latency_observed_n, w.comparable.n_overlap, 'latency')}</em></dd>
							</div>
							<div class={metricRowClass('decomposed', latencyWinner)}>
								<dt>[D]</dt>
								<dd>{fmtMs(w.comparable.decomposed_latency_mean_ms)} {#if zeroMetricLabel(w.comparable.decomposed_latency_mean_ms)}<em>{zeroMetricLabel(w.comparable.decomposed_latency_mean_ms)}</em>{:else if lowerRatioLabel('decomposed', latencyWinner, w.comparable.monolithic_latency_mean_ms, w.comparable.decomposed_latency_mean_ms, 'faster')}<em>{lowerRatioLabel('decomposed', latencyWinner, w.comparable.monolithic_latency_mean_ms, w.comparable.decomposed_latency_mean_ms, 'faster')}</em>{/if}<em>{telemetryCoverageText(w.comparable.decomposed_latency_observed_n, w.comparable.n_overlap, 'latency')}</em></dd>
							</div>
						</dl>
						<p>{w.comparable.monolithic_latency_mean_ms === 0 || w.comparable.decomposed_latency_mean_ms === 0 ? '0 ms is treated as telemetry floor, not a speed win' : 'aggregate scorer-step latency; missing telemetry is not counted as zero-latency evidence'}</p>
						<p><span class="metric-n">n={fmtCount(w.comparable.n_overlap)}</span> · <a class="ledger-ref" href={ledgerHref('resource_counter_metric')}>kind -> <LedgerKindChip kind="arch_arch_resource" variant="inline" /></a> · base -> clean shared verdict rows · not a quality signal</p>
					</article>
					<article id="metric-tokens" class="metric">
						<h3>tokens/reported evidence <span>n={fmtCount(w.comparable.n_overlap)} · clean shared · kind: <LedgerKindChip kind="arch_arch_resource" variant="inline" /></span></h3>
						<dl class="metric-rows" aria-label="token use by architecture">
							<div class={metricRowClass('monolithic', tokensWinner)}>
								<dt>[M]</dt>
								<dd>{fmtRate(monolithicTokensPerObserved)} {#if lowerRatioLabel('monolithic', tokensWinner, monolithicTokensPerObserved, decomposedTokensPerObserved, 'leaner')}<em>{lowerRatioLabel('monolithic', tokensWinner, monolithicTokensPerObserved, decomposedTokensPerObserved, 'leaner')}</em>{/if}<em>{telemetryCoverageText(w.comparable.monolithic_tokens_observed_n, w.comparable.n_overlap, 'tokens')} · {fmtCount(w.comparable.monolithic_tokens_total)} total</em></dd>
							</div>
							<div class={metricRowClass('decomposed', tokensWinner)}>
								<dt>[D]</dt>
								<dd>{fmtRate(decomposedTokensPerObserved)} {#if lowerRatioLabel('decomposed', tokensWinner, monolithicTokensPerObserved, decomposedTokensPerObserved, 'leaner')}<em>{lowerRatioLabel('decomposed', tokensWinner, monolithicTokensPerObserved, decomposedTokensPerObserved, 'leaner')}</em>{/if}<em>{telemetryCoverageText(w.comparable.decomposed_tokens_observed_n, w.comparable.n_overlap, 'tokens')} · {fmtCount(w.comparable.decomposed_tokens_total)} total</em></dd>
							</div>
						</dl>
						<p>prompt + output tokens; missing telemetry is not counted as zero-token evidence</p>
						<p><span class="metric-n">n={fmtCount(w.comparable.n_overlap)}</span> · <a class="ledger-ref" href={ledgerHref('resource_counter_metric')}>kind -> <LedgerKindChip kind="arch_arch_resource" variant="inline" /></a> · base -> clean shared verdict rows · not a quality signal</p>
					</article>
				</div>
			<section id="metric-mean-score-shift" class="metric-shift" aria-label="non-directional score movement">
				<div class="shift-kicker">non-directional · calibration posture</div>
				<div class="shift-head">
					<h3>mean score movement <span>n={fmtCount(w.comparable.n_overlap)} · clean shared · kind: <LedgerKindChip kind="arch_arch_score_posture" variant="inline" /></span></h3>
				</div>
				<div
					class="score-axis"
					style={scoreAxisStyle(w.comparable.monolithic_score_mean, w.comparable.decomposed_score_mean)}
					aria-label={`mean score posture on 0 to 1 belief scale: [M] ${fmtMaybe(w.comparable.monolithic_score_mean)}, [D] ${fmtMaybe(w.comparable.decomposed_score_mean)}`}
				>
					<p class="score-axis-guardrail">Not residual quality; use <a href="#metric-mae-vs-indra">MAE</a> and <a href="#metric-bias-vs-indra">bias</a> for truth-anchored error.</p>
					<div class="score-axis-scale" aria-hidden="true">
						<span>0.00 no belief</span>
						<span>0.50 uncertain</span>
						<span>1.00 full belief</span>
					</div>
					<div class="score-axis-rail">
						<span class="score-axis-mid" aria-hidden="true"></span>
						<span class="score-axis-dot score-axis-dot-m" aria-hidden="true"></span>
						<span class="score-axis-dot score-axis-dot-d" aria-hidden="true"></span>
					</div>
					<div class="score-axis-legend" aria-hidden="true">
						<span class="score-axis-legend-item"><i>[M]</i><strong>{fmtMaybe(w.comparable.monolithic_score_mean)}</strong></span>
						<span class="score-axis-legend-item"><i>[D]</i><strong>{fmtMaybe(w.comparable.decomposed_score_mean)}</strong></span>
					</div>
				</div>
				<dl class="metric-rows shift-rows" aria-label="mean score by architecture">
					<div class={metricRowClass('monolithic', 'none')}>
						<dt>[M]</dt>
						<dd>{fmtMaybe(w.comparable.monolithic_score_mean)}</dd>
					</div>
					<div class={metricRowClass('decomposed', 'none')}>
						<dt>[D]</dt>
						<dd>{fmtMaybe(w.comparable.decomposed_score_mean)}</dd>
					</div>
				</dl>
				<p><span class="metric-posture">shift magnitude {fmtAbsMaybe(w.comparable.mean_score_delta)}</span> · {scoreShiftDirection(w.comparable.mean_score_delta)}; not a winner and not an INDRA residual</p>
				<p><span class="metric-n">n={fmtCount(w.comparable.n_overlap)}</span> · <a class="ledger-ref" href={ledgerHref('score_posture_metric')}>kind -> <LedgerKindChip kind="arch_arch_score_posture" variant="inline" /></a> · base -> scores over clean shared verdict rows</p>
			</section>
		</section>

		<section class="arch-conditioned" aria-label="architecture-conditioned diagnostics">
			<h2>architecture-conditioned diagnostics</h2>
			<p class="scope-note">
				These panels are native to one architecture and are not converted into each other's grammar; their row populations are diagnostic, not the clean overlap denominator above.
			</p>
			<div class="arch-columns">
				<section class="arch-panel">
					<h3>[M] native tier path</h3>
					{#if w.arch_conditioned.monolithic_tiers.length > 0}
						<table>
							<thead><tr><th>tier</th><th class="num">rows</th><th class="num">mean score</th></tr></thead>
							<tbody>
								{#each w.arch_conditioned.monolithic_tiers as t}
									<tr><td>{t.tier}</td><td class="num">{fmtCount(t.n)}</td><td class="num">{fmtBelief(t.mean_score)}</td></tr>
								{/each}
							</tbody>
						</table>
					{:else}
						<p class="not-defined">Tier diagnostics are not persisted for this monolithic run.</p>
					{/if}
				</section>
				<section class="arch-panel">
					<h3>[D] native probe health</h3>
					{#if w.arch_conditioned.decomposed_probes.length > 0}
						<table>
							<thead><tr><th>step</th><th class="num">rows</th><th class="num">substrate</th><th class="num">errors</th></tr></thead>
							<tbody>
								{#each w.arch_conditioned.decomposed_probes as p}
									<tr><td>{p.name}</td><td class="num">{fmtCount(p.n)}</td><td class="num">{fmtCount(p.substrate_n)}</td><td class="num">{fmtCount(p.error_n)}</td></tr>
								{/each}
							</tbody>
						</table>
					{:else}
						<p class="not-defined">Probe diagnostics are not defined unless the decomposed scorer emits native probe rows.</p>
					{/if}
				</section>
			</div>
		</section>

	{/if}

	{#if visibleComparisonExemplarLanes.length > 0}
		<section id="comparison-exemplars" class="exemplars" aria-label="comparison exemplar lanes">
			<h2>comparison exemplar lanes</h2>
			<div class="exemplar-controls" aria-label="comparison exemplar controls">
				<nav class="segmented-control" aria-label="exemplar order">
					<span>order</span>
					{#each EXEMPLAR_SORT_OPTIONS as option}
						<a
							class:active={option.mode === exemplarSortMode}
							aria-current={option.mode === exemplarSortMode ? 'true' : undefined}
							href={exemplarModeHref('exemplar_sort', option.mode, 'impact')}
						>{option.label}</a>
					{/each}
				</nav>
				<nav class="segmented-control" aria-label="exemplar focus">
					<span>focus</span>
					{#each EXEMPLAR_FILTER_OPTIONS as option}
						<a
							class:active={option.mode === exemplarFilterMode}
							aria-current={option.mode === exemplarFilterMode ? 'true' : undefined}
							href={exemplarModeHref('exemplar_filter', option.mode, 'all')}
						>{option.label}</a>
					{/each}
				</nav>
				<a
					class="exemplar-export"
					href={exemplarCsvHref(comparisonCsv())}
					download={exemplarCsvFilename()}
				>csv all</a>
				<nav class="fold-memory" aria-label="exemplar fold memory">
					<span class="fold-memory-status">
						{#if visibleFoldPatternCount > 0}
							{visibleOpenFoldCount.toLocaleString()} of {visibleFoldPatternCount.toLocaleString()} groups expanded
						{:else}
							no repeated groups in view
						{/if}
						{#if hiddenOpenFoldCount > 0}
							<span class="fold-memory-outside">
								· {hiddenOpenFoldCount.toLocaleString()} outside current focus
								{#if hiddenOpenFoldLanes.length > 0}
									{#each hiddenOpenFoldLanes as lane}
										<a href={exemplarModeHref('exemplar_filter', 'all', 'all', laneAnchor(lane.key))}>{lane.title}</a>
									{/each}
								{/if}
							</span>
						{/if}
					</span>
					<button
						type="button"
						onclick={resetVisibleOpenPatterns}
						disabled={visibleOpenFoldCount === 0}
					>reset visible</button>
					<button
						type="button"
						onclick={resetAllOpenPatterns}
						disabled={hiddenOpenFoldCount === 0}
					>reset all</button>
				</nav>
			</div>
			<div class="exemplar-summary" aria-label="comparison exemplar accounting">
				<div class="exemplar-summary-scope">
					current view · focus {controlOptionLabel(EXEMPLAR_FILTER_OPTIONS, exemplarFilterMode)} · order {controlOptionLabel(EXEMPLAR_SORT_OPTIONS, exemplarSortMode)}
				</div>
				<dl class="exemplar-summary-math">
					<div class="summary-primary">
						<dt>distinct rows</dt>
						<dd>{comparisonExemplarSummary.distinctN.toLocaleString()}</dd>
					</div>
					<div>
						<dt>lane positions</dt>
						<dd>{comparisonExemplarSummary.positionsN.toLocaleString()}</dd>
					</div>
					<div>
						<dt>extra positions</dt>
						<dd>{comparisonExemplarSummary.extraPositionN.toLocaleString()}</dd>
					</div>
					<div>
						<dt>cross-listed rows</dt>
						<dd>{comparisonExemplarSummary.crossListedN.toLocaleString()}</dd>
					</div>
				</dl>
				{#if comparisonExemplarSummary.laneCounts.length > 0}
					<nav class="exemplar-summary-lanes" aria-label="active exemplar lane counts">
						<span>lane mix</span>
						{#each comparisonExemplarSummary.laneCounts as lane}
							<a class="exemplar-summary-lane" href={`#${laneAnchor(lane.key)}`}>
								<strong>{lane.count.toLocaleString()}</strong>
								<span>{lane.title}</span>
								<em>{lane.unit}</em>
							</a>
						{/each}
						{#if emptyComparisonExemplarLanes.length > 0}
							<span class="exemplar-summary-empty">{emptyComparisonExemplarLanes.length.toLocaleString()} lane type{emptyComparisonExemplarLanes.length === 1 ? '' : 's'} empty</span>
						{/if}
					</nav>
				{/if}
			</div>
			{#if emptyComparisonExemplarLanes.length > 0}
				<p class="empty-lane">No rows in lanes: {emptyComparisonExemplarLanes.map((lane) => lane.title).join(', ')}.</p>
			{/if}
			{#if hiddenByFilterComparisonExemplarLanes.length > 0}
				<p class="empty-lane">Hidden by focus: {hiddenByFilterComparisonExemplarLanes.map((lane) => lane.title).join(', ')}.</p>
			{/if}
			{#if activeComparisonExemplarLanes.length > 0}
				{#each activeComparisonExemplarLanes as lane}
					{@render exemplarLane(lane)}
				{/each}
			{:else}
				<p class="not-defined">No exemplar rows match the current focus.</p>
			{/if}
		</section>
	{/if}
</main>

<style>
		:global(:root) {
			--ink: #1a1a1a;
			--ink-muted: #6a6a6a;
			--ink-faint: #727272;
			--paper: #fdfcf8;
			--rule: #e6e2d6;
			--accent: #7d2a1a;
			--ok-green: #2a6f2a;
			--mass-m: #5f6b82;
			--mass-d: #6d7d3c;
			--mass-shared: var(--ink);
			--mass-out: var(--accent);
			--ledger-chip-native-bg: #f7f4ec;
			--ledger-chip-danger-ink: #9f2d20;
			--ledger-chip-danger-paper: #fff;
			--mono: ui-monospace, 'SF Mono', 'JetBrains Mono', Menlo, monospace;
			--serif: 'Iowan Old Style', 'Source Serif Pro', Georgia, serif;
		}
		:global(html) {
			scroll-behavior: smooth;
		}
		:global(html, body) {
			background: var(--paper);
		color: var(--ink);
		font-family: var(--serif);
		font-size: 16px;
		line-height: 1.45;
		margin: 0;
	}
	header {
		display: flex;
		justify-content: space-between;
		align-items: baseline;
		gap: 1rem;
		padding: 0.6rem 1.5rem;
		border-bottom: 1px solid var(--rule);
		font-family: var(--mono);
		font-size: 0.78rem;
		color: var(--ink-muted);
	}
	.crumb a {
		color: var(--ink-muted);
		text-decoration: none;
	}
	.crumb a:hover,
	a:hover {
		text-decoration: underline;
	}
	.crumb strong,
	code {
		color: var(--ink);
	}
	.sep,
	.meta,
	.scope-note,
	.not-defined,
	.empty-lane,
	.integrity-note,
	.trace-warning-note {
		color: var(--ink-muted);
	}
	a {
		color: var(--accent);
		text-decoration: none;
	}
	main {
		max-width: 1400px;
		margin: 0 auto;
		padding: 1.5rem 1.5rem 4rem;
	}
	h1,
	h2,
	h3 {
		font-family: var(--serif);
		font-weight: 400;
		margin: 0;
	}
	h1 {
		font-size: 1.7rem;
	}
	h2 {
		font-size: 1.2rem;
		margin-bottom: 0.45rem;
	}
	h2 span {
		font-family: var(--mono);
		font-size: 0.72rem;
		color: var(--ink-muted);
	}
	h3 {
		font-size: 1rem;
	}
		.pair-head {
			margin-bottom: 1.4rem;
		}
		.pair-head p {
			max-width: 46rem;
			margin: 0.35rem 0 0;
			color: var(--ink-muted);
		}
		.pair-actions {
			display: flex;
			flex-wrap: wrap;
			gap: 0.85rem;
			margin-top: 0.9rem;
			font-family: var(--mono);
			font-size: 0.78rem;
		}
		.pair-actions a,
		.pair-actions button {
			border: 0;
			border-bottom: 1px solid var(--ink);
			border-radius: 0;
			background: transparent;
			color: var(--ink);
			cursor: pointer;
			font: inherit;
			padding: 0;
		}
		.pair-actions .pair-action-primary {
			border: 1px solid var(--ink);
			padding: 0.08rem 0.35rem;
			text-decoration: none;
		}
			.workflow-state {
				border-top: 1px solid var(--ink);
				border-left: 3px solid var(--rule);
				padding: 0.8rem 0 0 0.7rem;
				margin-bottom: 1.6rem;
				scroll-margin-top: 4rem;
			}
			.workflow-state:target {
				outline: 2px solid var(--ink);
				outline-offset: 0.35rem;
			}
	.workflow-state-queued {
			border-left-color: var(--ink-muted);
			border-left-style: dashed;
		}
	.workflow-state-running {
			border-left-color: var(--ink);
			border-left-width: 4px;
			animation: workflow-pulse 1.8s ease-in-out infinite;
		}
		.workflow-state-idle {
			border-left-color: var(--rule);
		}
		.workflow-state-succeeded {
			border-left-color: var(--ok-green);
		}
		.workflow-state-failed,
		.workflow-state-crashed {
			border-left-color: var(--accent);
			border-left-width: 4px;
		}
		.workflow-state-canceled {
			border-left-color: var(--ink-muted);
			border-left-style: double;
		}
		@keyframes workflow-pulse {
			0%, 100% { border-left-color: var(--ink); }
			50% { border-left-color: var(--ink-muted); }
		}
		.workflow-head {
			display: flex;
			justify-content: space-between;
			gap: 1rem;
			align-items: baseline;
		}
		.workflow-actions {
			display: flex;
			flex-wrap: wrap;
			gap: 0.7rem;
			font-family: var(--mono);
			font-size: 0.76rem;
		}
		.workflow-actions a,
		.workflow-actions button {
			border: 0;
			border-bottom: 1px solid var(--ink);
			border-radius: 0;
			background: transparent;
			color: var(--ink);
			cursor: pointer;
			font: inherit;
			padding: 0;
		}
		.workflow-actions .workflow-cancel {
			border: 1px solid var(--accent);
			color: var(--accent);
			padding: 0.08rem 0.35rem;
		}
		.workflow-actions .workflow-rerun {
			border: 1px solid var(--ink);
			padding: 0.08rem 0.35rem;
			text-decoration: none;
		}
		.workflow-actions .workflow-cancel:hover {
			background: #f4e3dd;
		}
		.workflow-facts {
			display: grid;
			grid-template-columns: repeat(4, minmax(0, 1fr));
			gap: 0.6rem;
			margin: 0.65rem 0 0;
		}
		.workflow-facts dd {
			overflow: hidden;
			text-overflow: ellipsis;
			white-space: nowrap;
		}
		.workflow-clock {
			margin: 0.35rem 0 0;
			font-family: var(--mono);
			font-size: 0.72rem;
			color: var(--ink-muted);
		}
		.workflow-terminal,
		.workflow-idle-note {
			margin: 0.45rem 0 0;
			color: var(--ink-muted);
			font-family: var(--mono);
			font-size: 0.76rem;
		}
		.workflow-arches {
			display: grid;
			grid-template-columns: repeat(2, minmax(0, 1fr));
		gap: 0.7rem;
		margin-top: 0.6rem;
	}
	.workflow-arch {
		display: grid;
		grid-template-columns: 1fr;
		gap: 0.45rem;
		border-top: 1px solid var(--rule);
		padding-top: 0.55rem;
		font-family: var(--mono);
		font-size: 0.76rem;
		min-width: 0;
	}
	.workflow-arch-head {
		display: flex;
		align-items: baseline;
		justify-content: space-between;
		gap: 0.55rem;
	}
	.workflow-arch span {
		border: 1px solid currentColor;
		padding: 0 0.25rem;
		text-transform: lowercase;
	}
	.workflow-arch-queued span,
	.workflow-arch-loading span {
		border-style: dashed;
		color: var(--ink-muted);
	}
	.workflow-arch-running span {
		border-width: 2px;
		color: var(--ink);
		animation: workflow-pulse 1.8s ease-in-out infinite;
	}
	.workflow-arch-succeeded span {
		color: var(--ok-green);
	}
	.workflow-arch-failed span {
		color: var(--accent);
	}
	.workflow-arch-canceled span,
	.workflow-arch-blocked span {
		color: var(--ink-muted);
	}
	.workflow-arch-canceled span {
		text-decoration: line-through;
	}
	.workflow-arch-stalled .workflow-progress-rail {
		border-color: var(--accent);
	}
	.workflow-progress-rail {
		height: 0.5rem;
		background: var(--rule);
		border: 1px solid var(--rule);
	}
	.workflow-progress-rail i {
		display: block;
		width: var(--workflow-progress);
		min-width: 0;
		max-width: 100%;
		height: 100%;
		background: var(--ink);
	}
	.workflow-arch-failed .workflow-progress-rail i {
		background: var(--accent);
	}
	.workflow-arch-canceled .workflow-progress-rail i,
	.workflow-arch-blocked .workflow-progress-rail i {
		background: var(--ink-muted);
	}
	.workflow-arch-facts {
		display: grid;
		grid-template-columns: repeat(4, minmax(0, 1fr));
		gap: 0.45rem;
		margin: 0;
	}
	.workflow-arch-facts dt,
	.workflow-arch-facts dd {
		font-size: 0.62rem;
		line-height: 1.15;
	}
	.workflow-arch em {
		font-style: normal;
		color: var(--ink-muted);
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}
	.run-axis {
		display: grid;
		grid-template-columns: minmax(0, 1fr) auto minmax(0, 1fr);
		gap: 1rem;
		align-items: stretch;
		margin-bottom: 2rem;
	}
	.run-pane {
		border-top: 1px solid var(--ink);
		padding-top: 0.65rem;
		min-width: 0;
	}
	.run-pane-head {
		display: flex;
		align-items: flex-start;
		justify-content: space-between;
		gap: 0.75rem;
	}
	.run-pane-head h2 {
		display: flex;
		flex-wrap: wrap;
		gap: 0.45rem;
		align-items: baseline;
		margin: 0;
	}
	.run-pane-head h2 span {
		font-family: var(--mono);
		font-size: 0.72em;
		color: var(--ink-muted);
	}
	.run-status {
		flex: 0 0 auto;
		border: 1px solid var(--rule);
		padding: 0.12rem 0.35rem;
		font-family: var(--mono);
		font-size: 0.65rem;
		line-height: 1.2;
		color: var(--ink-muted);
		text-transform: lowercase;
	}
	.run-status-succeeded,
	.run-status-running {
		border-color: var(--ink);
		color: var(--ink);
	}
	.run-status-failed {
		border-color: var(--accent);
		color: var(--accent);
	}
	.run-status-canceled,
	.run-status-blocked {
		color: var(--ink-muted);
	}
	.run-status-canceled {
		text-decoration: line-through;
	}
	.run-link {
		display: inline-flex;
		align-items: baseline;
		gap: 0.35rem;
		margin-top: 0.45rem;
		font-family: var(--mono);
		font-size: 0.72rem;
		color: var(--ink-muted);
	}
	.run-link span {
		color: var(--ink-muted);
	}
	.run-link code,
	.stmt-link code {
		font-family: var(--mono);
		overflow-wrap: anywhere;
	}
	.run-facts,
	.example-metrics {
		display: grid;
		grid-template-columns: repeat(4, minmax(0, 1fr));
		gap: 0.65rem;
		margin: 0.65rem 0 0;
	}
	dt {
		font-family: var(--mono);
		font-size: 0.68rem;
		color: var(--ink-muted);
		text-transform: lowercase;
	}
	dd {
		margin: 0.1rem 0 0;
		font-family: var(--mono);
		font-size: 0.82rem;
		font-variant-numeric: tabular-nums;
		overflow-wrap: anywhere;
	}
		.axis-label {
			align-self: center;
		font-family: var(--mono);
		font-size: 0.72rem;
		color: var(--ink-muted);
		border-top: 1px solid var(--rule);
		border-bottom: 1px solid var(--rule);
			padding: 0.25rem 0.4rem;
			min-width: 10rem;
			text-align: center;
		}
	.axis-label strong {
		display: block;
		color: var(--ink);
		font-size: 1.1rem;
		font-variant-numeric: tabular-nums;
	}
	.axis-label span {
		display: block;
	}
	.axis-denominator-link {
		display: inline-block;
		margin-top: 0.15rem;
		color: var(--ink);
		text-decoration-color: var(--rule);
	}
		.massbar {
			display: flex;
			height: 0.75rem;
			margin-top: 0.35rem;
			background: var(--rule);
			border: 1px solid var(--rule);
		}
		.massbar i {
			display: block;
			min-width: 0;
		}
		.massbar-m,
		.massbar-d {
			background: var(--mass-m);
		}
		.massbar-o {
			background: var(--mass-shared);
		}
		.massbar-d {
			background: var(--mass-d);
		}
		.massbar-x {
			background: repeating-linear-gradient(
				45deg,
				var(--mass-out) 0,
				var(--mass-out) 1px,
				transparent 1px,
				transparent 4px
			);
		}
		.massbar-key {
			display: grid;
			grid-template-columns: repeat(4, minmax(0, 1fr));
			gap: 0.25rem;
			margin: 0.25rem 0 0;
		}
		.massbar-key div {
			min-width: 0;
		}
		.massbar-key dt,
		.massbar-key dd {
			font-size: 0.58rem;
			line-height: 1.1;
			overflow: hidden;
			text-overflow: ellipsis;
			white-space: normal;
		}
		.massbar-key dd {
			color: var(--ink-muted);
			margin-top: 0;
		}
		.massbar-key dt::before {
			content: '';
			display: inline-block;
			width: 0.58rem;
			height: 0.58rem;
			margin-right: 0.2rem;
			vertical-align: -0.08rem;
			background: var(--rule);
		}
		.massbar-key .key-m dt::before {
			background: var(--mass-m);
		}
		.massbar-key .key-o dt::before {
			background: var(--mass-shared);
		}
		.massbar-key .key-d dt::before {
			background: var(--mass-d);
		}
		.massbar-key .key-x dt::before {
			background: repeating-linear-gradient(
				45deg,
				var(--mass-out) 0,
				var(--mass-out) 1px,
				transparent 1px,
				transparent 4px
			);
			border: 1px solid var(--rule);
			box-sizing: border-box;
		}
	.not-defined-block,
	.overlap,
	.denominator-ledger,
	.resource-frontier,
	.comparable,
	.arch-conditioned,
	.exemplars {
		border-top: 2px solid var(--ink);
		padding-top: 1rem;
		margin-top: 1.8rem;
	}
	.scope-note {
		margin: 0 0 0.85rem;
	}
		.integrity-note {
			margin: 0.8rem 0 0;
			border-left: 2px solid var(--accent);
			padding-left: 0.55rem;
			font-family: var(--mono);
			font-size: 0.78rem;
		}
		.overlap-ledger-note {
			display: flex;
			flex-wrap: wrap;
			gap: 0.7rem;
			margin: 0.2rem 0 0;
			font-family: var(--mono);
			font-size: 0.78rem;
		}
	.trace-warning-note {
		margin: 0.35rem 0 0;
		border-left: 2px dotted var(--rule);
		padding-left: 0.55rem;
		font-family: var(--mono);
		font-size: 0.78rem;
	}
		.resource-frontier {
			scroll-margin-top: 4rem;
		}
		.resource-frontier:target {
			outline: 2px solid var(--ink);
			outline-offset: 0.35rem;
		}
		.frontier-head h2 span {
			display: block;
			margin-top: 0.08rem;
			font-family: var(--mono);
			font-size: 0.68rem;
			color: var(--ink-muted);
		}
		.frontier-grid {
			display: grid;
			grid-template-columns: repeat(2, minmax(0, 1fr));
			gap: 1rem;
			align-items: start;
		}
		.frontier-arch {
			min-width: 0;
			border-left: 3px solid var(--rule);
			padding-left: 0.7rem;
		}
		.frontier-arch-monolithic {
			border-left-color: var(--mass-m);
		}
		.frontier-arch-decomposed {
			border-left-color: var(--mass-d);
		}
		.frontier-arch h3 {
			margin-bottom: 0.45rem;
			font-size: 1rem;
		}
		.frontier-rows {
			display: grid;
			gap: 0.26rem;
			margin: 0;
		}
		.frontier-row {
			display: grid;
			grid-template-columns: minmax(7.5rem, 0.8fr) minmax(0, 1fr);
			gap: 0.55rem;
			align-items: baseline;
			border-left: 2px solid transparent;
			padding-left: 0.35rem;
			font-family: var(--mono);
			font-variant-numeric: tabular-nums;
		}
		.frontier-row-monolithic {
			border-left-color: var(--mass-m);
		}
		.frontier-row-decomposed {
			border-left-color: var(--mass-d);
		}
		.frontier-row-winner {
			background: rgba(26, 26, 26, 0.035);
			border-left-width: 4px;
		}
		.frontier-row-tie {
			border-left-color: var(--ink-muted);
			border-left-style: dotted;
		}
		.frontier-row dd {
			margin: 0;
			color: var(--ink);
			font-size: 0.95rem;
		}
		.frontier-row dd em {
			color: var(--ink-muted);
			display: block;
			font-size: 0.68rem;
			font-style: normal;
			line-height: 1.2;
		}
		.frontier-scopes {
			display: grid;
			grid-template-columns: repeat(4, minmax(0, 1fr));
			gap: 0.55rem;
			margin-top: 0.8rem;
			font-family: var(--mono);
			font-size: 0.72rem;
			color: var(--ink-muted);
		}
		.frontier-scopes p {
			margin: 0;
			border-top: 1px solid var(--rule);
			padding-top: 0.3rem;
			min-width: 0;
		}
		.frontier-scopes strong {
			color: var(--ink);
			display: block;
			font-weight: 600;
			text-transform: lowercase;
		}
		.metric-grid {
			display: grid;
			grid-template-columns: repeat(3, minmax(0, 1fr));
			gap: 0.8rem;
		}
			.metric {
				border-left: 2px solid var(--rule);
				padding-left: 0.65rem;
				min-width: 0;
			}
			.metric h3 {
				font-family: var(--serif);
				font-size: 0.98rem;
				font-weight: 400;
				margin: 0 0 0.35rem;
			}
			.metric h3 span {
				display: block;
				margin-top: 0.08rem;
				font-family: var(--mono);
				font-size: 0.68rem;
				color: var(--ink-muted);
			}
			.metric p {
				display: block;
				margin-top: 0.15rem;
				font-style: normal;
				color: var(--ink-muted);
				font-size: 0.78rem;
			}
			.metric-rows {
				display: grid;
				gap: 0.18rem;
				margin: 0;
			}
			.metric-row {
				display: grid;
				grid-template-columns: 2.3rem minmax(0, 1fr);
				gap: 0.45rem;
				align-items: baseline;
				font-family: var(--mono);
				font-variant-numeric: tabular-nums;
				border-left: 3px solid transparent;
				padding-left: 0.3rem;
			}
			.metric-row-monolithic {
				border-left-color: var(--mass-m);
			}
			.metric-row-decomposed {
				border-left-color: var(--mass-d);
			}
			.metric-row dt {
				color: var(--ink-muted);
				margin: 0;
			}
			.metric-row dd {
				margin: 0;
				color: var(--ink);
				font-size: 1.02rem;
				overflow-wrap: anywhere;
			}
			.metric-row dd em {
				font-size: 0.72rem;
				font-style: normal;
				margin-left: 0.25rem;
				color: var(--ink-muted);
				white-space: nowrap;
			}
			.metric-row dd em + em {
				display: block;
				margin-left: 0;
				white-space: normal;
			}
			.metric-row-winner {
				background: rgba(26, 26, 26, 0.035);
				border-left-width: 4px;
			}
			.metric-row-winner dt,
			.metric-row-winner dd {
				font-weight: 600;
			}
			.metric-row-winner dd em {
				color: var(--ink);
			}
			.metric-row-tie {
				border-left-color: var(--ink-muted);
				border-left-style: dotted;
			}
			.metric-result {
				color: var(--ink);
				font-family: var(--mono);
				font-variant-numeric: tabular-nums;
			}
			.verdict-fusion {
				position: relative;
				border-top: 2px solid var(--ink);
				border-bottom: 1px solid var(--rule);
				margin-bottom: 1rem;
				padding: 0.8rem 0 0.9rem;
				scroll-margin-top: 4rem;
			}
			.legacy-anchor {
				position: absolute;
				top: -4rem;
				display: block;
				width: 1px;
				height: 1px;
				overflow: hidden;
			}
			.verdict-fusion-head {
				display: flex;
				justify-content: space-between;
				gap: 1rem;
				align-items: start;
				margin-bottom: 0.7rem;
			}
			.fusion-kicker {
				margin-bottom: 0.2rem;
				color: var(--ink);
				font-family: var(--mono);
				font-size: 0.68rem;
				letter-spacing: 0;
				text-transform: uppercase;
			}
			.verdict-fusion h3,
			.verdict-fusion h4 {
				font-family: var(--serif);
				font-weight: 400;
				margin: 0;
			}
			.verdict-fusion h3 {
				font-size: 1rem;
			}
			.verdict-fusion h3 span {
				display: block;
				margin-top: 0.08rem;
				font-family: var(--mono);
				font-size: 0.68rem;
				color: var(--ink-muted);
			}
			.verdict-definition {
				max-width: 42rem;
				margin: 0.35rem 0 0;
				color: var(--ink);
				font-size: 0.82rem;
			}
			.verdict-guardrail {
				margin: 0.3rem 0 0;
				color: var(--ink-muted);
				font-family: var(--mono);
				font-size: 0.72rem;
			}
			.verdict-fusion h4 {
				margin-bottom: 0.45rem;
				font-size: 0.9rem;
			}
			.verdict-comparison-strip {
				display: grid;
				grid-template-columns: minmax(0, 1fr) auto minmax(7rem, auto) auto minmax(0, 1fr);
				gap: 0.65rem;
				align-items: center;
				margin: 0 0 0.9rem;
				border: 1px solid var(--ink);
				padding: 0.55rem 0.65rem;
				font-family: var(--mono);
				font-variant-numeric: tabular-nums;
			}
			.verdict-compare-term {
				min-width: 0;
			}
			.verdict-compare-term span {
				display: block;
				color: var(--ink-muted);
				font-size: 0.7rem;
			}
			.verdict-compare-term strong {
				display: block;
				font-size: 1.1rem;
				font-weight: 600;
			}
			.verdict-recode-term {
				text-align: center;
			}
			.verdict-operator {
				border: 1px solid var(--rule);
				color: var(--ink-muted);
				padding: 0.08rem 0.38rem;
				font-size: 0.72rem;
				font-weight: 700;
				text-align: center;
				text-transform: uppercase;
			}
			.verdict-operator-diverges {
				border-color: var(--ink);
				color: var(--ink);
			}
			.verdict-comparison-strip p {
				margin: 0;
				color: var(--ink);
				font-size: 0.76rem;
			}
			.verdict-fusion-grid {
				display: grid;
				grid-template-columns: minmax(15rem, 0.85fr) minmax(0, 1.4fr);
				gap: 1rem;
				align-items: start;
			}
			.verdict-marginals,
			.verdict-matrix-panel {
				min-width: 0;
			}
			.marginal-row {
				display: grid;
				grid-template-columns: minmax(8rem, 1fr) auto;
				gap: 0.45rem;
				align-items: baseline;
				margin-top: 0.35rem;
				border-left: 3px solid var(--rule);
				padding-left: 0.45rem;
				font-family: var(--mono);
				font-size: 0.78rem;
				font-variant-numeric: tabular-nums;
			}
			.marginal-row-monolithic {
				border-left-color: var(--mass-m);
			}
			.marginal-row-decomposed {
				border-left-color: var(--mass-d);
			}
			.marginal-label {
				color: var(--ink);
			}
			.marginal-row strong {
				font-weight: 600;
			}
			.marginal-row em {
				grid-column: 1 / -1;
				color: var(--ink-muted);
				font-style: normal;
				white-space: nowrap;
			}
			.verdict-marginals p,
			.verdict-fusion-note {
				margin: 0.45rem 0 0;
				color: var(--ink-muted);
				font-size: 0.78rem;
			}
			.verdict-marginals p {
				font-family: var(--mono);
			}
			.verdict-taxonomy {
				margin-top: 0.9rem;
				border-top: 1px dotted var(--rule);
				padding-top: 0.65rem;
			}
			.verdict-taxonomy-grid {
				display: grid;
				gap: 0.45rem;
			}
			.verdict-taxonomy-group {
				border-left: 3px solid var(--rule);
				padding-left: 0.5rem;
			}
			.verdict-taxonomy-group-head {
				display: grid;
				grid-template-columns: minmax(8rem, 1fr) auto minmax(7rem, auto);
				gap: 0.55rem;
				align-items: baseline;
				font-family: var(--mono);
				font-size: 0.74rem;
				font-variant-numeric: tabular-nums;
			}
			.verdict-taxonomy-group-head span {
				color: var(--ink);
				font-weight: 600;
			}
			.verdict-taxonomy-group-head strong {
				font-weight: 600;
			}
			.verdict-taxonomy-group-head em {
				color: var(--ink-muted);
				font-style: normal;
				text-align: right;
				white-space: nowrap;
			}
			.verdict-taxonomy-group-rows {
				display: grid;
				gap: 0.22rem;
				margin-top: 0.22rem;
			}
			.verdict-taxonomy-row {
				display: grid;
				grid-template-columns: minmax(8rem, 0.9fr) repeat(2, minmax(0, 1fr)) auto;
				gap: 0.55rem;
				align-items: baseline;
				border-left: 1px solid var(--rule);
				padding: 0.16rem 0 0.16rem 0.45rem;
				font-family: var(--mono);
				font-size: 0.76rem;
				font-variant-numeric: tabular-nums;
			}
			.verdict-taxonomy-both-supported {
				border-left-color: var(--ink-muted);
				background:
					repeating-linear-gradient(
						135deg,
						rgba(26, 26, 26, 0.045) 0,
						rgba(26, 26, 26, 0.045) 4px,
						transparent 4px,
						transparent 8px
					);
			}
			.verdict-taxonomy-neither-supported {
				border-left-color: var(--ink-muted);
				background:
					repeating-linear-gradient(
						135deg,
						rgba(26, 26, 26, 0.045) 0,
						rgba(26, 26, 26, 0.045) 4px,
						transparent 4px,
						transparent 8px
					);
			}
			.verdict-taxonomy-monolithic-only {
				border-left-color: var(--mass-m);
			}
			.verdict-taxonomy-decomposed-only {
				border-left-color: var(--mass-d);
			}
			.verdict-taxonomy-row.exact-divergence {
				border-left-style: dotted;
			}
			.verdict-taxonomy-row.exact-divergence .taxonomy-cell {
				color: var(--ink);
			}
			.taxonomy-cell {
				color: var(--ink-muted);
				white-space: nowrap;
			}
			.taxonomy-arch {
				min-width: 0;
				overflow-wrap: anywhere;
			}
			.verdict-taxonomy-row strong {
				font-weight: 600;
			}
			.verdict-taxonomy p {
				margin: 0.4rem 0 0;
				color: var(--ink-muted);
				font-size: 0.78rem;
			}
			.metric-shift {
				border-top: 2px solid var(--ink);
				border-bottom: 1px dotted var(--rule);
				margin-top: 1.15rem;
				padding: 0.75rem 0;
				scroll-margin-top: 4rem;
			}
			.metric:target,
			.metric-shift:target,
			.verdict-fusion:target {
				outline: 2px solid var(--ink);
				outline-offset: 0.35rem;
			}
			.shift-kicker {
				margin-bottom: 0.2rem;
				color: var(--ink);
				font-family: var(--mono);
				font-size: 0.68rem;
				letter-spacing: 0;
				text-transform: uppercase;
			}
			.shift-head {
				display: block;
				margin-bottom: 0.45rem;
			}
			.shift-head h3 {
				font-family: var(--serif);
				font-size: 0.98rem;
				font-weight: 400;
				margin: 0;
			}
			.shift-head h3 span {
				display: block;
				margin-top: 0.08rem;
				font-family: var(--mono);
				font-size: 0.68rem;
				color: var(--ink-muted);
			}
			.shift-head p,
			.metric-shift p {
				margin: 0.15rem 0 0;
				color: var(--ink-muted);
				font-size: 0.78rem;
			}
			.shift-head a {
				color: var(--ink);
				text-decoration: underline;
				text-decoration-thickness: 1px;
				text-underline-offset: 2px;
			}
			.score-axis {
				--m-pos: 50%;
				--d-pos: 50%;
				max-width: 38rem;
				margin: 0.55rem 0 0.5rem;
				font-family: var(--mono);
				font-variant-numeric: tabular-nums;
			}
			.score-axis-guardrail {
				margin: 0 0 0.4rem;
				border-left: 2px solid var(--ink);
				padding-left: 0.45rem;
				color: var(--ink);
				font-size: 0.72rem;
			}
			.score-axis-guardrail a {
				color: var(--ink);
				text-decoration: underline;
				text-decoration-thickness: 1px;
				text-underline-offset: 2px;
			}
			.score-axis-scale {
				display: grid;
				grid-template-columns: repeat(3, minmax(0, 1fr));
				color: var(--ink-muted);
				font-size: 0.64rem;
				line-height: 1.2;
			}
			.score-axis-scale span:nth-child(2) {
				color: var(--ink);
				font-weight: 600;
				text-align: center;
			}
			.score-axis-scale span:nth-child(3) {
				text-align: right;
			}
			.score-axis-rail {
				position: relative;
				height: 1.7rem;
				margin: 0.08rem 0 0.15rem;
			}
			.score-axis-rail::before {
				content: '';
				position: absolute;
				left: 0;
				right: 0;
				top: 0.82rem;
				border-top: 2px solid var(--ink);
			}
			.score-axis-rail::after {
				content: '';
				position: absolute;
				left: 0;
				right: 0;
				top: 0.62rem;
				height: 0.42rem;
				background:
					repeating-linear-gradient(
						to right,
						var(--rule) 0,
						var(--rule) 1px,
						transparent 1px,
						transparent 10%
					);
				pointer-events: none;
			}
			.score-axis-mid {
				position: absolute;
				left: 50%;
				top: 0.15rem;
				height: 1.35rem;
				border-left: 2px solid var(--ink);
			}
			.score-axis-dot {
				display: block;
				position: absolute;
				top: 0.54rem;
				width: 0.58rem;
				height: 0.58rem;
				border: 1px solid var(--ink);
				background: var(--paper);
				transform: translateX(-50%);
			}
			.score-axis-dot-m {
				left: clamp(0.35rem, var(--m-pos), calc(100% - 0.35rem));
			}
			.score-axis-dot-d {
				left: clamp(0.35rem, var(--d-pos), calc(100% - 0.35rem));
			}
			.score-axis-legend {
				display: flex;
				flex-wrap: wrap;
				gap: 0.4rem;
				align-items: center;
				font-size: 0.72rem;
			}
			.score-axis-legend-item {
				display: inline-flex;
				gap: 0.28rem;
				align-items: baseline;
				border: 1px solid var(--rule);
				padding: 0.05rem 0.28rem;
				white-space: nowrap;
			}
			.score-axis-legend-item i {
				color: var(--ink-muted);
				font-style: normal;
			}
			.score-axis-legend-item strong {
				color: var(--ink);
				font-weight: 600;
			}
			.shift-rows {
				grid-template-columns: repeat(2, minmax(0, 1fr));
				gap: 0.5rem;
				max-width: 34rem;
			}
			.metric-posture {
				border: 1px solid var(--ink-muted);
				color: var(--ink);
				font-family: var(--mono);
				font-variant-numeric: tabular-nums;
				padding: 0 0.28rem;
			}
			.example-metrics dd span,
			.example-metrics dd em {
				color: var(--ink-muted);
				font-style: normal;
			}
		.metric-n {
			color: var(--ink);
			font-family: var(--mono);
			font-variant-numeric: tabular-nums;
			white-space: nowrap;
		}
			.outcome-matrix-scroll {
				position: relative;
				overflow-x: auto;
				border-top: 1px solid var(--ink);
				border-left: 1px solid var(--ink);
				box-shadow: inset -0.9rem 0 0.8rem -0.8rem rgba(26, 26, 26, 0.45);
				scrollbar-gutter: stable;
			}
			.outcome-matrix {
				display: grid;
				grid-template-columns: minmax(6.8rem, 0.75fr) repeat(2, minmax(0, 1fr));
				min-width: 30rem;
			}
			.outcome-corner,
			.outcome-axis,
			.outcome-cell {
				border-right: 1px solid var(--rule);
				border-bottom: 1px solid var(--rule);
				padding: 0.45rem 0.55rem;
				min-width: 0;
			}
			.outcome-axis {
				font-family: var(--mono);
				font-size: 0.72rem;
				font-weight: 600;
				color: var(--ink);
				text-transform: uppercase;
			}
			.outcome-axis-row {
				color: var(--ink);
			}
			.outcome-cell strong {
				display: block;
				font-family: var(--mono);
				font-size: 1.05rem;
				font-variant-numeric: tabular-nums;
			}
			.outcome-cell span {
				display: block;
				margin-top: 0.1rem;
				color: var(--ink-muted);
				font-size: 0.82rem;
			}
			.outcome-cell-diagonal {
				background:
					repeating-linear-gradient(
						135deg,
						rgba(26, 26, 26, 0.08) 0,
						rgba(26, 26, 26, 0.08) 4px,
						rgba(26, 26, 26, 0.025) 4px,
						rgba(26, 26, 26, 0.025) 8px
					);
				box-shadow: inset 0 0 0 1px rgba(26, 26, 26, 0.12);
			}
			.outcome-cell-split {
				background: rgba(253, 252, 248, 0.45);
			}
			.outcome-cell-m-only {
				border-left: 3px solid var(--mass-m);
			}
			.outcome-cell-d-only {
				border-left: 3px solid var(--mass-d);
			}
			.outcome-note {
				display: flex;
				flex-wrap: wrap;
				gap: 0.8rem;
				margin: 0.45rem 0 0;
				font-family: var(--mono);
				font-size: 0.76rem;
				color: var(--ink-muted);
			}
			.outcome-note span {
				display: inline-flex;
				gap: 0.3rem;
				align-items: center;
			}
			.outcome-note .legend-zero {
				opacity: 0.62;
			}
			.outcome-note .legend-zero .legend-swatch {
				background:
					repeating-linear-gradient(
						45deg,
						rgba(26, 26, 26, 0.06) 0,
						rgba(26, 26, 26, 0.06) 2px,
						transparent 2px,
						transparent 5px
					);
			}
			.legend-swatch {
				display: inline-block;
				width: 0.75rem;
				height: 0.75rem;
				border: 1px solid var(--ink-muted);
			}
			.legend-diagonal {
				background:
					repeating-linear-gradient(
						135deg,
						rgba(26, 26, 26, 0.08) 0,
						rgba(26, 26, 26, 0.08) 3px,
						rgba(26, 26, 26, 0.025) 3px,
						rgba(26, 26, 26, 0.025) 6px
					);
			}
			.legend-m-only {
				border-left: 3px solid var(--mass-m);
			}
			.legend-d-only {
				border-left: 3px solid var(--mass-d);
			}
	.arch-columns {
		display: grid;
		grid-template-columns: repeat(2, minmax(0, 1fr));
		gap: 1.5rem;
	}
	.arch-panel {
		min-width: 0;
	}
	.table-wrap {
		overflow-x: auto;
		border-top: 1px solid var(--ink);
	}
	table {
		width: 100%;
		border-collapse: collapse;
		font-family: var(--mono);
		font-size: 0.78rem;
		font-variant-numeric: tabular-nums;
	}
	th,
	td {
		text-align: left;
		vertical-align: top;
		padding: 0.45rem 0.7rem 0.45rem 0;
		border-bottom: 1px dotted var(--rule);
	}
	th {
		color: var(--ink-muted);
		font-size: 0.68rem;
		font-weight: 500;
		text-transform: lowercase;
	}
		.num {
			text-align: right;
		}
		.denominator-ledger h3 {
			margin: 1rem 0 0.35rem;
			font-family: var(--mono);
			font-size: 0.78rem;
			font-weight: 600;
			text-transform: lowercase;
		}
		.denominator-ledger table {
			min-width: 76rem;
		}
		.denominator-ledger table.no-outside {
			min-width: 69rem;
		}
		.ledger-column-policy {
			caption-side: top;
			text-align: left;
			padding: 0 0 0.45rem;
			color: var(--ink-muted);
			font-family: var(--mono);
			font-size: 0.72rem;
			text-transform: lowercase;
		}
		.native-ledger-note {
			margin: -0.1rem 0 0.45rem;
			color: var(--ink-muted);
			font-family: var(--mono);
			font-size: 0.72rem;
		}
		.ledger-row-family-arch-arch td:first-child {
			border-left: 3px solid var(--ink);
			padding-left: 0.45rem;
		}
		.ledger-row-family-arch-indra td:first-child {
			border-left: 3px solid var(--accent);
			padding-left: 0.45rem;
		}
		.ledger-row-family-unknown td:first-child {
			outline: 2px solid var(--ledger-chip-danger-ink);
			outline-offset: -2px;
		}
		.ledger-panel-cell {
			min-width: 15rem;
			position: relative;
		}
		.ledger-panel-title {
			overflow-wrap: anywhere;
		}
		.ledger-parent {
			color: var(--ink-muted);
			display: block;
			font-family: var(--mono);
			font-size: 0.66rem;
			margin-top: 0.1rem;
		}
		.ledger-parent-continued {
			font-style: italic;
		}
		.ledger-tree-role-root td:first-child {
			border-left: 4px solid var(--ink);
			font-weight: 600;
			padding-left: 0.75rem;
		}
		.ledger-tree-role-base td:first-child {
			border-left: 2px dotted var(--ink);
			padding-left: 1.5rem;
		}
		.ledger-tree-role-metric td:first-child {
			padding-left: 2.65rem;
		}
		.ledger-tree-role-metric td:first-child::before {
			border-left: 1px solid var(--rule);
			content: '';
			position: absolute;
			top: 0;
			bottom: 0;
			left: 1.65rem;
		}
		.ledger-tree-role-gate td:first-child {
			border-left: 2px solid var(--rule);
			padding-left: 1.5rem;
		}
		.ledger-tree-role-native td:first-child {
			border-left: 2px solid var(--rule);
			padding-left: 0.75rem;
		}
		.ledger-count-kind {
			color: var(--ink);
			font-weight: 600;
			white-space: nowrap;
		}
		.ledger-count-inherited {
			color: var(--ink-muted);
			font-style: italic;
			font-weight: 400;
			white-space: nowrap;
		}
		.ledger-row {
			scroll-margin-top: 4rem;
		}
		.ledger-row:target {
			background: #efe6ca;
			outline: 2px solid var(--ink);
			outline-offset: -2px;
		}
		.ledger-key-clean_true_nonoverlap_evidence td:first-child {
			border-left: 2px solid var(--ink);
			font-weight: 600;
			padding-left: 0.45rem;
		}
		.ledger-basis {
			min-width: 18rem;
			max-width: 28rem;
			color: var(--ink-muted);
			overflow-wrap: normal;
		}
		.ledger-ref {
			color: var(--ink);
			text-decoration: underline;
			text-decoration-thickness: 1px;
			text-underline-offset: 2px;
		}
		.ledger-back {
			display: none;
			margin-top: 0.2rem;
			font-family: var(--mono);
			font-size: 0.68rem;
		}
		.ledger-row:target .ledger-back {
			display: block;
		}
		.example-lane {
			border-left: 3px solid var(--rule);
			border-top: 1px solid var(--ink);
			padding-top: 1rem;
			padding-left: 0.75rem;
			margin-top: 0.8rem;
			scroll-margin-top: 4rem;
		}
		.example-lane:target {
			outline: 2px solid var(--ink);
			outline-offset: 0.35rem;
		}
	.empty-lane,
	.not-defined {
		border-left: 2px solid var(--ink);
		padding-left: 0.55rem;
		font-family: var(--mono);
		font-size: 0.78rem;
	}
	.not-defined::before {
		content: 'n/d ';
		color: var(--ink);
		font-weight: 600;
	}
	.lane-head {
		display: flex;
		justify-content: space-between;
		gap: 1rem;
		align-items: flex-start;
		margin-bottom: 0.5rem;
	}
	.lane-head-main {
		min-width: 0;
	}
	.lane-head h3 {
		display: flex;
		gap: 0.45rem;
		align-items: baseline;
		flex-wrap: wrap;
	}
	.lane-n {
		color: var(--ink-muted);
		font-family: var(--mono);
		font-size: 0.7rem;
		font-weight: 400;
	}
	.lane-head p {
		margin: 0;
		color: var(--ink-muted);
		font-size: 0.84rem;
	}
	.lane-tools {
		display: flex;
		flex-wrap: wrap;
		gap: 0.35rem;
		justify-content: flex-end;
		font-family: var(--mono);
		font-size: 0.72rem;
		white-space: nowrap;
	}
	.lane-tools a,
	.exemplar-export,
	.fold-memory button {
		border: 1px solid var(--rule);
		color: var(--ink);
		font-family: var(--mono);
		font-size: 0.72rem;
		line-height: 1.35;
		padding: 0.08rem 0.34rem;
	}
	.exemplar-export {
		border-color: var(--ink);
		background: transparent;
		color: var(--ink);
	}
	.lane-tools a:hover,
	.lane-tools a:focus-visible,
	.exemplar-export:hover,
	.exemplar-export:focus-visible {
		border-color: var(--ink);
		outline: none;
	}
	.exemplar-summary {
		display: grid;
		gap: 0.5rem;
		border-left: 2px solid var(--ink);
		padding-left: 0.55rem;
		color: var(--ink-muted);
		font-family: var(--mono);
		font-size: 0.78rem;
		overflow-wrap: anywhere;
	}
	.exemplar-summary-scope {
		color: var(--ink-muted);
		font-size: 0.68rem;
		text-transform: lowercase;
	}
	.exemplar-summary-math {
		display: grid;
		grid-template-columns: minmax(8rem, 1.35fr) repeat(3, minmax(6.5rem, 1fr));
		gap: 0.45rem;
		margin: 0;
	}
	.exemplar-summary-math div {
		min-width: 0;
		border-top: 1px solid var(--rule);
		padding-top: 0.22rem;
	}
	.exemplar-summary-math .summary-primary {
		border-top-color: var(--ink);
	}
	.exemplar-summary-math dt {
		font-size: 0.64rem;
	}
	.exemplar-summary-math dd {
		margin: 0.04rem 0 0;
		color: var(--ink);
		font-size: 1rem;
	}
	.exemplar-summary-math .summary-primary dd {
		font-size: 1.35rem;
		font-weight: 600;
		line-height: 1.05;
	}
	.exemplar-summary-lanes {
		display: flex;
		flex-wrap: wrap;
		gap: 0.35rem;
		align-items: stretch;
	}
	.exemplar-summary-lanes > span {
		align-self: center;
		color: var(--ink-muted);
		font-size: 0.68rem;
	}
	.exemplar-summary-lane {
		display: inline-grid;
		grid-template-columns: auto minmax(0, 1fr) auto;
		flex: 1 1 15rem;
		column-gap: 0.35rem;
		row-gap: 0;
		max-width: 18rem;
		border: 1px solid var(--rule);
		padding: 0.12rem 0.36rem;
		color: var(--ink);
	}
	.exemplar-summary-lane::after {
		content: '->';
		grid-column: 3;
		grid-row: 1 / 3;
		align-self: center;
		color: var(--accent);
		font-size: 0.68rem;
	}
	.exemplar-summary-lane:hover,
	.exemplar-summary-lane:focus-visible {
		border-color: var(--ink);
		outline: none;
		text-decoration: none;
	}
	.exemplar-summary-lane strong {
		font-weight: 600;
		font-variant-numeric: tabular-nums;
	}
	.exemplar-summary-lane span,
	.exemplar-summary-lane em {
		min-width: 0;
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}
	.exemplar-summary-lane em {
		grid-column: 2;
		color: var(--ink-muted);
		font-size: 0.64rem;
		font-style: normal;
	}
	.exemplar-summary-empty {
		align-self: stretch;
		border: 1px dashed var(--rule);
		color: var(--ink-muted);
		display: inline-flex;
		align-items: center;
		padding: 0.12rem 0.36rem;
		font-size: 0.68rem;
	}
	.exemplar-controls {
		display: flex;
		flex-wrap: wrap;
		gap: 1.35rem;
		align-items: center;
		margin: 0.45rem 0 0.55rem;
	}
	.fold-memory {
		display: inline-grid;
		grid-template-columns: minmax(0, 1fr) 7.8rem 6rem;
		align-items: stretch;
		border: 1px solid var(--rule);
		width: min(100%, 36rem);
		margin-left: auto;
		font-family: var(--mono);
		font-size: 0.72rem;
		line-height: 1.35;
	}
	.fold-memory-status {
		display: flex;
		align-items: center;
		flex-wrap: wrap;
		gap: 0.1rem 0.34rem;
		color: var(--ink-muted);
		font-size: 0.66rem;
		min-width: 0;
		min-height: 2rem;
		padding: 0.2rem 0.42rem;
	}
	.fold-memory-outside {
		display: inline-flex;
		align-items: center;
		flex-wrap: wrap;
		gap: 0.28rem;
	}
	.fold-memory-outside a {
		color: var(--ink);
		text-decoration: underline;
		text-underline-offset: 0.12em;
	}
	.fold-memory button {
		background: transparent;
		border: 0;
		border-left: 1px solid var(--rule);
		border-radius: 0;
		cursor: pointer;
		font: inherit;
		min-height: 2rem;
		min-width: 0;
		padding: 0.2rem 0.42rem;
		white-space: nowrap;
	}
	.fold-memory button:hover,
	.fold-memory button:focus-visible {
		background: var(--ink);
		color: var(--paper);
		outline: none;
	}
	.fold-memory button:disabled {
		background: transparent;
		color: var(--ink-muted);
		cursor: not-allowed;
		opacity: 0.38;
		text-decoration: line-through;
	}
	.fold-memory button:disabled:hover,
	.fold-memory button:disabled:focus-visible {
		background: transparent;
		color: var(--ink-muted);
	}
	.segmented-control {
		display: inline-flex;
		flex-wrap: wrap;
		align-items: stretch;
		border: 1px solid var(--rule);
		font-family: var(--mono);
		font-size: 0.72rem;
		line-height: 1.35;
	}
	.segmented-control span,
	.segmented-control a {
		padding: 0.12rem 0.38rem;
	}
	.segmented-control span {
		background: transparent;
		color: var(--ink-muted);
		border-right: 1px solid var(--rule);
		font-size: 0.64rem;
		text-transform: uppercase;
	}
	.segmented-control a {
		color: var(--ink);
		border-right: 1px solid var(--rule);
	}
	.segmented-control a:last-child {
		border-right: 0;
	}
	.segmented-control a.active {
		background: var(--ink);
		color: var(--paper);
	}
	.example-list {
		display: grid;
		gap: 0.75rem;
	}
	.lane-table {
		margin: 0.25rem 0 0.75rem;
		border-left: 1px dotted var(--rule);
		padding-left: 0.6rem;
	}
	.lane-table summary {
		cursor: pointer;
		font-family: var(--mono);
		font-size: 0.78rem;
		color: var(--ink);
		border-bottom: 1px dotted currentColor;
		width: fit-content;
	}
	.lane-table summary:hover,
	.lane-table summary:focus-visible {
		color: var(--ink);
		outline: 1px solid var(--ink);
		outline-offset: 0.2rem;
	}
	.lane-table table {
		min-width: 58rem;
	}
	.lane-table td {
		vertical-align: top;
	}
	.example-pattern {
		border-left: 2px solid var(--ink-muted);
		padding-left: 0.75rem;
		min-width: 0;
	}
	.example-pattern summary {
		display: grid;
		grid-template-columns: auto minmax(0, 1fr);
		gap: 0.18rem 0.5rem;
		align-items: start;
		cursor: pointer;
		list-style: none;
	}
	.example-pattern summary:hover,
	.example-pattern summary:focus-visible {
		outline: 1px solid var(--ink);
		outline-offset: 0.25rem;
	}
	.example-pattern summary::-webkit-details-marker {
		display: none;
	}
	.example-pattern summary::before {
		content: '[>]';
		font-family: var(--mono);
		font-size: 0.78rem;
		color: var(--ink);
	}
	.example-pattern[open] summary::before {
		content: '[v]';
	}
	.example-pattern summary strong {
		font-family: var(--mono);
		font-size: 0.78rem;
		font-weight: 600;
		line-height: 1.35;
	}
	.example-pattern summary span {
		grid-column: 2;
		color: var(--ink-muted);
		font-family: var(--mono);
		font-size: 0.74rem;
		line-height: 1.45;
		overflow-wrap: anywhere;
	}
	.example-pattern summary em {
		grid-column: 2;
		color: var(--ink-muted);
		font-family: var(--mono);
		font-size: 0.72rem;
		font-style: normal;
		line-height: 1.35;
	}
	.example-pattern[open] summary {
		margin-bottom: 0.65rem;
	}
	.example-list-folded {
		border-left: 1px dotted var(--rule);
		padding-left: 0.6rem;
	}
	.example-row {
		border-left: 2px solid var(--rule);
		padding-left: 0.75rem;
		min-width: 0;
	}
	.example-row:target {
		outline: 2px solid var(--ink);
		outline-offset: 0.35rem;
	}
	.verdict-split-row {
		border-left-style: double;
	}
	.example-lane-excluded_by_integrity .example-row {
		border-left-color: var(--ink-muted);
	}
	.example-lane-excluded_by_integrity .example-row.integrity-error {
		border-left-color: var(--accent);
	}
	.example-main {
		display: flex;
		gap: 0.55rem;
		align-items: baseline;
		flex-wrap: wrap;
		font-family: var(--mono);
		font-size: 0.78rem;
	}
	.type {
		color: var(--ok-green);
	}
	.agents {
		max-width: 32rem;
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}
	.source {
		color: var(--ink-muted);
	}
	.excluded-side {
		border: 1px solid var(--ink-muted);
		color: var(--ink-muted);
		padding: 0 0.28rem;
		font-size: 0.68rem;
	}
	.verdict-split-mark {
		border: 1px solid var(--accent);
		color: var(--accent);
		padding: 0 0.28rem;
		font-size: 0.68rem;
	}
	.integrity-error .excluded-side {
		border-color: var(--accent);
		color: var(--accent);
	}
	.integrity-reason {
		margin: 0.25rem 0 0;
		color: var(--ink-muted);
		font-family: var(--mono);
		font-size: 0.76rem;
	}
		.integrity-error .integrity-reason {
			color: var(--accent);
		}
		.lane-memberships {
			display: flex;
			flex-wrap: wrap;
			gap: 0.35rem;
			align-items: baseline;
			margin-top: 0.28rem;
			font-family: var(--mono);
			font-size: 0.72rem;
			color: var(--ink-muted);
		}
		.lane-memberships a {
			border-bottom: 1px solid var(--accent);
			color: var(--ink);
			line-height: 1.35;
		}
		.lane-memberships a::before {
			content: '-> ';
			color: var(--accent);
		}
		.lane-membership {
			display: inline-flex;
			flex-wrap: wrap;
			gap: 0.15rem;
			align-items: baseline;
		}
		.lane-memberships em {
			color: var(--ink-muted);
			font-style: normal;
		}
		.lane-memberships em::before {
			content: ' · ';
		}
		.evidence {
			max-width: 58rem;
			margin: 0.3rem 0 0;
		color: var(--ink);
		overflow-wrap: anywhere;
	}
	.example-links {
		display: flex;
		gap: 0.8rem;
		flex-wrap: wrap;
		margin-top: 0.45rem;
		font-family: var(--mono);
		font-size: 0.76rem;
	}
	.example-stepper {
		display: flex;
		align-items: baseline;
		gap: 0.45rem;
		margin-top: 0.35rem;
		font-family: var(--mono);
		font-size: 0.72rem;
		color: var(--ink-muted);
	}
	.example-stepper a {
		border-bottom: 1px dotted currentColor;
		color: var(--ink);
	}
	.example-stepper .step-position {
		min-width: 3.8rem;
		text-align: center;
	}
	.example-stepper .step-disabled {
		color: var(--ink-faint);
	}
	.delta-metric dd,
	.delta-cell {
		color: var(--accent);
		font-weight: 700;
	}
	.verdict-chip {
		display: inline-block;
		border: 1px solid var(--rule);
		padding: 0 0.25rem;
		font-family: var(--mono);
		font-size: 0.7rem;
		line-height: 1.35;
	}
	.metric-missing {
		color: var(--ink-muted);
		font-style: normal;
	}
	.example-metrics dd .verdict-chip {
		color: var(--ink-muted);
	}
	.example-metrics dd .verdict-chip.verdict-correct {
		border-color: var(--ink);
		color: var(--ink);
	}
	.example-metrics dd .verdict-chip.verdict-incorrect {
		border-color: var(--accent);
		color: var(--accent);
	}
	.example-metrics dd .verdict-chip.verdict-abstain,
	.example-metrics dd .verdict-chip.verdict-missing {
		color: var(--ink-muted);
	}
	@media (max-width: 760px) {
		header {
			align-items: flex-start;
			flex-direction: column;
			gap: 0.25rem;
		}
		main {
			padding: 1rem 1rem 3rem;
		}
		.run-axis,
		.arch-columns,
		.frontier-grid,
		.frontier-scopes {
			grid-template-columns: 1fr;
		}
		.axis-label {
			justify-self: start;
			}
				.run-facts,
				.metric-grid,
				.example-metrics,
				.workflow-facts,
				.workflow-arch-facts,
			.workflow-arches,
			.exemplar-summary-math {
					grid-template-columns: repeat(2, minmax(0, 1fr));
			}
		.workflow-arch {
			grid-template-columns: 1fr;
		}
		.workflow-arch em {
			white-space: normal;
		}
		.lane-head {
			display: block;
		}
		.verdict-fusion-head {
			display: block;
		}
		.verdict-fusion-grid {
			grid-template-columns: 1fr;
		}
		.verdict-comparison-strip {
			grid-template-columns: 1fr;
		}
		.verdict-recode-term {
			text-align: left;
		}
		.verdict-operator {
			justify-self: start;
		}
		.verdict-comparison-strip p {
			grid-column: 1 / -1;
			border-top: 1px dotted var(--rule);
			padding-top: 0.35rem;
		}
		.shift-head {
			display: block;
		}
		.agents {
			max-width: 100%;
			white-space: normal;
		}
		.fold-memory {
			grid-template-columns: 1fr 1fr;
			width: 100%;
			margin-left: 0;
		}
		.fold-memory-status {
			grid-column: 1 / -1;
			border-bottom: 1px solid var(--rule);
			min-height: 1.8rem;
		}
		.fold-memory button {
			min-height: 2.6rem;
			white-space: normal;
		}
		.fold-memory button:first-of-type {
			border-left: 0;
		}
	}
		@media (max-width: 460px) {
					.run-facts,
					.metric-grid,
				.example-metrics,
				.workflow-facts,
				.workflow-arch-facts,
				.workflow-arches,
				.shift-rows,
				.frontier-row,
				.exemplar-summary-math {
					grid-template-columns: 1fr;
				}
				.exemplar-summary-lane,
				.exemplar-summary-empty {
					flex-basis: 100%;
					max-width: none;
				}
				.marginal-row {
					grid-template-columns: 1fr;
				}
				.verdict-taxonomy-row {
					grid-template-columns: 1fr auto;
				}
				.taxonomy-cell {
					grid-column: 1 / -1;
					white-space: normal;
				}
				.taxonomy-arch {
					grid-column: 1;
				}
		}
</style>
