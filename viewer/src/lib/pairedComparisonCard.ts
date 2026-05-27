import type {
	PairRunSummary,
	PairedComparableMetrics,
	PairedDenominatorRow,
	PairedOverlapStats,
	PairedResourceFrontier,
	PairedWorkbench
} from './db';
import pairedComparisonCardSchema from './pairedComparisonCard.schema.json';

type Architecture = 'monolithic' | 'decomposed';

export const PAIRED_COMPARISON_CARD_SCHEMA_VERSION = 'indra_pair_comparison_card_v1';
export { pairedComparisonCardSchema };

export interface PairedComparisonCardRun {
	present: boolean;
	run_id: string | null;
	architecture: Architecture;
	status: string | null;
	model_id_default: string | null;
	started_at: string | null;
	finished_at: string | null;
	n_evidences: number | null;
	cost_estimate_usd: number | null;
	cost_actual_usd: number | null;
	duration_s: number | null;
}

export interface PairedComparisonCard {
	schema_version: typeof PAIRED_COMPARISON_CARD_SCHEMA_VERSION;
	generated_at: string;
	pair_id: string;
	pair_href: string;
	status: 'defined' | 'not_defined';
	not_defined_reason: string | null;
	workflow_status: string | null;
	guardrails: string[];
	runs: Record<Architecture, PairedComparisonCardRun>;
	overlap: PairedOverlapStats | null;
	comparable: PairedComparableMetrics | null;
	resource_frontier: PairedResourceFrontier | null;
	denominator_ledger: PairedDenominatorRow[];
	architecture_native: PairedWorkbench['arch_conditioned'];
}

export interface BuildPairedComparisonCardOptions {
	generated_at: string;
	pair_href?: string;
	workflow_status?: string | null;
}

function runCard(architecture: Architecture, run: PairRunSummary | null): PairedComparisonCardRun {
	return {
		present: Boolean(run),
		run_id: run?.run_id ?? null,
		architecture,
		status: run?.status ?? null,
		model_id_default: run?.model_id_default ?? null,
		started_at: run?.started_at ?? null,
		finished_at: run?.finished_at ?? null,
		n_evidences: run?.n_evidences ?? null,
		cost_estimate_usd: run?.cost_estimate_usd ?? null,
		cost_actual_usd: run?.cost_actual_usd ?? null,
		duration_s: run?.duration_s ?? null
	};
}

export function buildPairedComparisonCard(
	workbench: PairedWorkbench,
	options: BuildPairedComparisonCardOptions
): PairedComparisonCard {
	return {
		schema_version: PAIRED_COMPARISON_CARD_SCHEMA_VERSION,
		generated_at: options.generated_at,
		pair_id: workbench.pair_id,
		pair_href: options.pair_href ?? `/pairs/${encodeURIComponent(workbench.pair_id)}`,
		status: workbench.comparable ? 'defined' : 'not_defined',
		not_defined_reason: workbench.not_defined_reason,
		workflow_status: options.workflow_status ?? null,
		guardrails: [
			'Paired metrics default to clean shared aggregate verdict evidence.',
			'Whole-run spend and wall clock do not share the clean-overlap denominator.',
			'Lower resource use is not a quality signal.',
			'Architecture-native diagnostics are not converted into the other architecture grammar.',
			'Not-defined panels are explicit contract states, not zero values.'
		],
		runs: {
			monolithic: runCard('monolithic', workbench.monolithic),
			decomposed: runCard('decomposed', workbench.decomposed)
		},
		overlap: workbench.overlap,
		comparable: workbench.comparable,
		resource_frontier: workbench.resource_frontier,
		denominator_ledger: workbench.denominator_ledger,
		architecture_native: workbench.arch_conditioned
	};
}

export function pairedComparisonCardJson(card: PairedComparisonCard): string {
	return `${JSON.stringify(card, null, 2)}\n`;
}

export function pairedComparisonCardSchemaJson(): string {
	return `${JSON.stringify(pairedComparisonCardSchema, null, 2)}\n`;
}

function mdCell(value: unknown): string {
	if (value === null || value === undefined || value === '') return '-';
	return String(value).replace(/\|/g, '\\|').replace(/\n/g, ' ');
}

function mdNumber(value: number | null | undefined, digits = 3): string {
	if (value === null || value === undefined || Number.isNaN(value)) return '-';
	if (Number.isInteger(value)) return value.toLocaleString();
	return value.toLocaleString(undefined, { maximumFractionDigits: digits });
}

function mdCost(value: number | null | undefined): string {
	if (value === null || value === undefined || Number.isNaN(value)) return '-';
	if (value === 0) return '$0';
	if (value < 0.01) return '<$0.01';
	if (value < 100) return `$${value.toFixed(2)}`;
	return `$${value.toFixed(0)}`;
}

function runCostBasis(run: PairedComparisonCardRun): string {
	if (run.cost_actual_usd != null) return `${mdCost(run.cost_actual_usd)} actual`;
	if (run.cost_estimate_usd != null) return `${mdCost(run.cost_estimate_usd)} estimate`;
	return '-';
}

function markdownTable(headers: string[], rows: string[][]): string {
	return [
		`| ${headers.map(mdCell).join(' | ')} |`,
		`| ${headers.map(() => '---').join(' | ')} |`,
		...rows.map((row) => `| ${row.map(mdCell).join(' | ')} |`)
	].join('\n');
}

export function pairedComparisonCardMarkdown(card: PairedComparisonCard): string {
	const lines: string[] = [];
	lines.push(`# Paired Architecture Comparison: ${card.pair_id}`);
	lines.push('');
	lines.push(`Generated: ${card.generated_at}`);
	lines.push(`Pair: ${card.pair_href}`);
	lines.push(`Status: ${card.status}${card.not_defined_reason ? ` (${card.not_defined_reason})` : ''}`);
	if (card.workflow_status) lines.push(`Workflow: ${card.workflow_status}`);
	lines.push('');
	lines.push('## Guardrails');
	for (const guardrail of card.guardrails) lines.push(`- ${guardrail}`);
	lines.push('');
	lines.push('## Runs');
	lines.push(markdownTable(
		['arch', 'run', 'status', 'model', 'evidence', 'cost', 'duration_s'],
		(['monolithic', 'decomposed'] as const).map((arch) => {
			const run = card.runs[arch];
			return [
				arch === 'monolithic' ? '[M]' : '[D]',
				run.run_id ?? 'missing',
				run.status ?? '-',
				run.model_id_default ?? '-',
				mdNumber(run.n_evidences, 0),
				runCostBasis(run),
				mdNumber(run.duration_s, 1)
			];
		})
	));
	lines.push('');
	lines.push('## Overlap');
	if (card.overlap) {
		lines.push(markdownTable(
			['clean shared evidence', 'shared statements', '[M] only', '[D] only', 'integrity outside'],
			[[
				mdNumber(card.overlap.overlap_evidences, 0),
				mdNumber(card.overlap.overlap_statements, 0),
				mdNumber(card.overlap.monolithic_only_evidences, 0),
				mdNumber(card.overlap.decomposed_only_evidences, 0),
				mdNumber(
					card.overlap.monolithic_step_error_evidences +
						card.overlap.decomposed_step_error_evidences +
						card.overlap.monolithic_missing_aggregate_evidences +
						card.overlap.decomposed_missing_aggregate_evidences +
						card.overlap.monolithic_nonverdict_aggregate_evidences +
						card.overlap.decomposed_nonverdict_aggregate_evidences,
					0
				)
			]]
		));
	} else {
		lines.push('Not defined: overlap accounting is unavailable.');
	}
	lines.push('');
	lines.push('## Comparable Metrics');
	if (card.comparable) {
		lines.push(markdownTable(
			['metric', '[M]', '[D]', 'paired value', 'scope'],
			[
				['exact verdict agreement', '-', '-', `${mdNumber(card.comparable.verdict_agreement_n, 0)}/${mdNumber(card.comparable.n_overlap, 0)} exact labels`, 'clean shared aggregate verdict evidence'],
				['MAE vs INDRA', mdNumber(card.comparable.monolithic_mae), mdNumber(card.comparable.decomposed_mae), '-', `truth overlap n=${mdNumber(card.comparable.n_truth_overlap, 0)}`],
				['bias vs INDRA', mdNumber(card.comparable.monolithic_bias), mdNumber(card.comparable.decomposed_bias), '-', `truth overlap n=${mdNumber(card.comparable.n_truth_overlap, 0)}`],
				['mean score', mdNumber(card.comparable.monolithic_score_mean), mdNumber(card.comparable.decomposed_score_mean), '-', `clean overlap n=${mdNumber(card.comparable.n_overlap, 0)}`],
				['mean latency ms', mdNumber(card.comparable.monolithic_latency_mean_ms, 1), mdNumber(card.comparable.decomposed_latency_mean_ms, 1), `${mdNumber(card.comparable.monolithic_latency_observed_n, 0)}/${mdNumber(card.comparable.n_overlap, 0)} [M], ${mdNumber(card.comparable.decomposed_latency_observed_n, 0)}/${mdNumber(card.comparable.n_overlap, 0)} [D] reported`, 'clean shared aggregate verdict evidence with telemetry coverage'],
				['token rows reported', mdNumber(card.comparable.monolithic_tokens_observed_n, 0), mdNumber(card.comparable.decomposed_tokens_observed_n, 0), '-', `clean overlap n=${mdNumber(card.comparable.n_overlap, 0)}`]
			]
		));
	} else {
		lines.push(`Not defined: ${card.not_defined_reason ?? 'comparable metrics need clean shared overlap.'}`);
	}
	lines.push('');
	if (card.resource_frontier) {
		lines.push('## Resource Frontier');
		lines.push(`Spend scope: ${card.resource_frontier.spend_scope}`);
		lines.push(`Latency scope: ${card.resource_frontier.latency_scope}`);
		lines.push(`Quality scope: ${card.resource_frontier.quality_scope}`);
		lines.push(markdownTable(
			['arch', 'cost/ev', 'wall s/ev', 'latency ms', 'latency rows', 'tokens/reported ev', 'token rows', 'MAE'],
			(['monolithic', 'decomposed'] as const).map((arch) => {
				const row = card.resource_frontier![arch];
				return [
					arch === 'monolithic' ? '[M]' : '[D]',
					mdCost(row.cost_per_evidence_usd),
					mdNumber(row.wall_seconds_per_evidence, 1),
					mdNumber(row.clean_overlap_latency_mean_ms, 1),
					`${mdNumber(row.clean_overlap_latency_observed_n, 0)}/${mdNumber(row.clean_overlap_n, 0)}`,
					mdNumber(row.clean_overlap_tokens_per_observed_evidence, 1),
					`${mdNumber(row.clean_overlap_tokens_observed_n, 0)}/${mdNumber(row.clean_overlap_n, 0)}`,
					mdNumber(row.mae)
				];
			})
		));
		lines.push('');
	}
	lines.push('## Denominator Ledger');
	if (card.denominator_ledger.length > 0) {
		lines.push(markdownTable(
			['key', 'panel', 'scope', 'kind', 'unit', 'panel n', 'shared', 'outside'],
			card.denominator_ledger.map((row) => [
				row.key,
				row.panel,
				row.applicability,
				row.metric_kind,
				row.unit,
				mdNumber(row.denominator_n, 0),
				mdNumber(row.overlap_n, 0),
				mdNumber(row.excluded_n, 0)
			])
		));
	} else {
		lines.push('No denominator ledger rows are defined for this pair state.');
	}
	lines.push('');
	lines.push('## Architecture-Native Diagnostics');
	lines.push(`Monolithic tier rows: ${mdNumber(card.architecture_native.monolithic_tiers.reduce((sum, row) => sum + row.n, 0), 0)}`);
	lines.push(`Decomposed probe rows: ${mdNumber(card.architecture_native.decomposed_probes.reduce((sum, row) => sum + row.n, 0), 0)}`);
	lines.push('');
	return `${lines.join('\n')}\n`;
}
