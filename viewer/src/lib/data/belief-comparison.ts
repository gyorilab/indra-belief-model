/**
 * Fail-closed schema for the canonical assembled-statement comparison artifact.
 *
 * There is no generic `auprc` field: the historical paper
 * estimand (mean per-fold trapezoidal PR area) and pooled average precision are
 * distinct metrics.  Every plotted arm repeats the exact evaluation contract,
 * has complete prediction coverage, and carries its own prediction/cost
 * provenance. The artifact also carries explicit non-runnable configuration
 * exclusions, exact-decimal cost intervals, and evidence-identity retry
 * accounting. The artifact is accepted only when both canonical paper panels
 * pass together.
 */

export const BELIEF_COMPARISON_KIND = 'indra_statement_belief_comparison';
export const BELIEF_PREDICTION_UNIT = 'assembled_statement';
export const BELIEF_GOLD_RULE = 'released_paper_observed_positive_else_negative';
export const BELIEF_STRICT_GOLD_RULE = 'strict_e0_resolved_only';
export const BELIEF_POSITIVE_CLASS = 'correct_statement';
export const BELIEF_PARETO_METRIC = 'fold_mean_trapezoidal_pr_auc';
export const BELIEF_PANEL_IDS = ['paper_all_source', 'paper_readers'] as const;
export const BELIEF_PRIMARY_METRICS = [
	'fold_mean_trapezoidal_pr_auc',
	'pooled_average_precision',
	'auroc',
	'brier',
	'log_loss',
	'calibration_ece',
	'calibration_intercept_abs_error',
	'calibration_slope_abs_error'
] as const;
export const BELIEF_THRESHOLD_METRICS = [
	'threshold_accuracy',
	'threshold_precision',
	'threshold_recall',
	'threshold_f1'
] as const;

export type BeliefPanelId = (typeof BELIEF_PANEL_IDS)[number];
export type BeliefArmFamily = 'paper' | 'current' | 'llm';
export type BeliefMetricKey =
	| (typeof BELIEF_PRIMARY_METRICS)[number]
	| (typeof BELIEF_THRESHOLD_METRICS)[number];
export type BeliefBetterWhen = 'higher' | 'lower';
export type BeliefCostBasis =
	| 'provider_measured_observed'
	| 'mixed_conservative_upper_bound';
export type BeliefCostProjection = 'all_executions' | 'observed_execution_subset';

export interface BeliefPricingTariff {
	input_usd_per_million: string;
	output_usd_per_million: string;
	pricing_basis: string;
}

export interface BeliefPricing {
	cost_comparability_id: string;
	currency: 'USD';
	provider: string;
	provider_model_id: string;
	pricing_mode: 'on_demand';
	region: string;
	resolved_service_tier: 'standard';
	retrieved_on: string;
	service_tier_request: 'default';
	source_url: string;
	tariff: BeliefPricingTariff;
	unit: 'per_million_tokens';
}

export interface BeliefContractIdentity {
	prediction_unit: typeof BELIEF_PREDICTION_UNIT;
	gold_rule: typeof BELIEF_GOLD_RULE | typeof BELIEF_STRICT_GOLD_RULE;
	substrate_sha256: string;
	gold_sha256: string;
	evaluation_set_sha256: string;
}

export interface BeliefEstimate {
	estimate: number;
	ci95: [number, number];
	method: string;
	resamples: number;
	valid_resamples: number;
}

export interface BeliefFoldEstimate {
	fold_id: number;
	n: number;
	positive: number;
	negative: number;
	estimate: number;
}

export interface BeliefFoldedEstimate extends BeliefEstimate {
	fold_estimates: BeliefFoldEstimate[];
	fold_population_sd: number;
}

export interface BeliefReliabilityBin {
	bin_index: number;
	lower: number;
	upper: number;
	upper_inclusive: boolean;
	n: number;
	mean_prediction: number | null;
	observed_fraction: number | null;
}

export interface BeliefCalibrationMetrics {
	ece: BeliefEstimate;
	intercept: BeliefEstimate;
	slope: BeliefEstimate;
	intercept_abs_error: BeliefEstimate;
	slope_abs_error: BeliefEstimate;
	reliability_bins: BeliefReliabilityBin[];
}

export interface BeliefThresholdAvailable {
	status: 'available';
	value: number;
	operator: 'greater_than_or_equal';
	source_path: string;
	source_sha256: string;
	frozen_at: string;
	reason: null;
	confusion: { tp: number; fp: number; fn: number; tn: number };
	metrics: {
		accuracy: BeliefEstimate;
		precision: BeliefEstimate;
		recall: BeliefEstimate;
		f1: BeliefEstimate;
	};
}

export interface BeliefThresholdUnavailable {
	status: 'unavailable';
	value: null;
	operator: null;
	source_path: null;
	source_sha256: null;
	frozen_at: null;
	reason: string;
	confusion: null;
	metrics: null;
}

export type BeliefThreshold = BeliefThresholdAvailable | BeliefThresholdUnavailable;

export interface BeliefArmMetrics {
	fold_mean_trapezoidal_pr_auc: BeliefFoldedEstimate;
	pooled_average_precision: BeliefEstimate;
	auroc: BeliefEstimate;
	brier: BeliefEstimate;
	log_loss: BeliefEstimate;
	calibration: BeliefCalibrationMetrics;
	threshold: BeliefThreshold;
}

export interface BeliefCoverage {
	eligible: number;
	predicted: number;
	invalid: number;
	fraction: number;
}

export interface BeliefCostAvailable {
	status: 'available';
	record_type: 'evidence_execution';
	inference_usd_total: number;
	inference_usd_total_exact: string;
	usd_per_1k_statements: number;
	provider_measured_usd_total: number;
	provider_measured_usd_total_exact: string;
	conservative_reserved_usd_total: number;
	conservative_reserved_usd_total_exact: string;
	inference_usd_lower: number;
	inference_usd_lower_exact: string;
	inference_usd_upper: number;
	inference_usd_upper_exact: string;
	usd_per_1k_statements_lower: number;
	usd_per_1k_statements_upper: number;
	basis: BeliefCostBasis;
	view_id: string;
	includes_retries: true;
	includes_relation_subcalls: true;
	denominator: { statements: number; evidence_executions: number };
	scope: {
		included_cost_categories: ['provider_inference_calls'];
		excluded_cost_categories: ['training', 'local_aggregation', 'feature_materialization', 'upstream_reading'];
	};
	execution_count: number;
	attempt_count: number;
	retry_attempt_count: number;
	successful_attempt_count: number;
	error_attempt_count: number;
	provider_measured_call_count: number;
	conservative_call_count: number;
	input_tokens: number | null;
	output_tokens: number | null;
	token_accounting_complete: boolean;
	ledger_path: string;
	ledger_sha256: string;
	price_source: string;
	price_date: string;
	cost_comparability_id: string;
	pricing: BeliefPricing;
	projection: BeliefCostProjection;
	counterfactual_run_cost: false;
	shared_run_id: string;
	additive_across_panels: false;
	reason: null;
}

export interface BeliefCostUnavailable {
	status: 'unavailable';
	record_type: null;
	inference_usd_total: null;
	inference_usd_total_exact: null;
	usd_per_1k_statements: null;
	provider_measured_usd_total: null;
	provider_measured_usd_total_exact: null;
	conservative_reserved_usd_total: null;
	conservative_reserved_usd_total_exact: null;
	inference_usd_lower: null;
	inference_usd_lower_exact: null;
	inference_usd_upper: null;
	inference_usd_upper_exact: null;
	usd_per_1k_statements_lower: null;
	usd_per_1k_statements_upper: null;
	basis: 'unavailable';
	view_id: null;
	includes_retries: null;
	includes_relation_subcalls: null;
	denominator: null;
	scope: null;
	execution_count: null;
	attempt_count: null;
	retry_attempt_count: null;
	successful_attempt_count: null;
	error_attempt_count: null;
	provider_measured_call_count: null;
	conservative_call_count: null;
	input_tokens: null;
	output_tokens: null;
	token_accounting_complete: null;
	ledger_path: null;
	ledger_sha256: null;
	price_source: null;
	price_date: null;
	cost_comparability_id: null;
	pricing: null;
	projection: null;
	counterfactual_run_cost: null;
	shared_run_id: null;
	additive_across_panels: null;
	reason: string;
}

export type BeliefCost = BeliefCostAvailable | BeliefCostUnavailable;

export interface BeliefArmParetoAvailable {
	status: 'available';
	view_id: string;
	basis: BeliefCostBasis;
	point_pareto: boolean;
	uncertainty_pareto: boolean;
	reason: null;
}

export interface BeliefArmParetoUnavailable {
	status: 'unavailable';
	view_id: null;
	basis: null;
	point_pareto: null;
	uncertainty_pareto: null;
	reason: string;
}

export type BeliefArmPareto = BeliefArmParetoAvailable | BeliefArmParetoUnavailable;

export interface BeliefArmProvenance {
	implementation: string;
	implementation_digest: string;
	training_data_sha256: string | null;
	environment: string;
	notes: string | null;
	predictions_path: string;
	predictions_sha256: string;
}

export interface BeliefArm {
	arm_id: string;
	label: string;
	family: BeliefArmFamily;
	contract: BeliefContractIdentity;
	coverage: BeliefCoverage;
	metrics: BeliefArmMetrics;
	released_label_error_strata: BeliefReleasedLabelErrorStrata | null;
	cost: BeliefCost;
	pareto: BeliefArmPareto;
	provenance: BeliefArmProvenance;
}

export interface BeliefErrorStratum {
	statements: number;
	tp: number;
	fp: number;
	fn: number;
	tn: number;
	errors: number;
}

export interface BeliefReleasedLabelErrorStrata {
	strict_e0_resolved: BeliefErrorStratum;
	released_negative_assumption: BeliefErrorStratum;
}

export interface BeliefReleasedLabelAudit {
	released_label_rule: string;
	strict_e0_rule: string;
	released: { statements: number; positive: number; negative: number };
	strict_e0: {
		resolved: number;
		positive: number;
		negative: number;
		unresolved: number;
		ordered_statement_id_sha256: string;
	};
	released_negative_assumption: {
		statements: number;
		share_of_released_negatives: number;
		ordered_statement_id_sha256: string;
	};
}

export interface BeliefSensitivityArm {
	arm_id: string;
	label: string;
	family: BeliefArmFamily;
	contract: BeliefContractIdentity;
	coverage: BeliefCoverage;
	metrics: BeliefArmMetrics;
}

export interface BeliefStrictSensitivity {
	analysis_scope: 'fixed_resolved_only_sensitivity';
	selection_rule: string;
	contract: BeliefContractIdentity;
	gold_path: string;
	n_evaluable: number;
	n_positive: number;
	n_negative: number;
	excluded_unresolved: number;
	pr_summary_contract: {
		fold_mean_trapezoidal_pr_auc: string;
		pooled_average_precision: string;
		fold_count: number;
	};
	arms: BeliefSensitivityArm[];
	comparisons: BeliefPairedComparison[];
}

export interface BeliefPairedComparison {
	a_arm_id: string;
	b_arm_id: string;
	metric: BeliefMetricKey;
	direction: 'b_minus_a';
	better_when: BeliefBetterWhen;
	contract: BeliefContractIdentity;
	delta: BeliefEstimate;
	resamples: number;
	method: string;
}

export interface BeliefParetoAuditRow {
	candidate_arm_id: string;
	challenger_arm_id: string;
	candidate_cost_per_1k_interval: [number, number];
	challenger_cost_per_1k_interval: [number, number];
	challenger_minus_candidate_cost_per_1k: number;
	cost_interval_definitely_not_worse: boolean;
	challenger_minus_candidate_performance: number;
	performance_delta_ci95: [number, number];
	point_dominates: boolean;
	uncertainty_dominates: boolean;
}

export interface BeliefParetoView {
	view_id: string;
	basis: BeliefCostBasis | 'mixed';
	eligible_arm_ids: string[];
	point_frontier_arm_ids: string[];
	uncertainty_frontier_arm_ids: string[];
	audit: BeliefParetoAuditRow[];
}

export interface BeliefPareto {
	objective_metric: typeof BELIEF_PARETO_METRIC;
	performance_direction: 'higher_is_better';
	cost_axis: 'usd_per_1k_statements_upper';
	point_rule: string;
	uncertainty_rule: string;
	views: BeliefParetoView[];
}

export interface BeliefExcludedArm {
	arm_id: string;
	label: string;
	family: BeliefArmFamily;
	status: 'excluded';
	reason: string;
	required_artifact: string;
	provenance: string;
}

export interface BeliefSubstrate {
	substrate_id: BeliefPanelId;
	lane: 'paper';
	label: string;
	analysis_scope: 'primary';
	released_label_audit: BeliefReleasedLabelAudit;
	contract: BeliefContractIdentity;
	substrate_manifest_path: string;
	gold_path: string;
	positive_class: typeof BELIEF_POSITIVE_CLASS;
	n_evaluable: number;
	n_positive: number;
	n_negative: number;
	pr_summary_contract: {
		fold_mean_trapezoidal_pr_auc: string;
		pooled_average_precision: string;
		fold_count: number;
	};
	arms: BeliefArm[];
	excluded_arms: BeliefExcludedArm[];
	comparisons: BeliefPairedComparison[];
	pareto: BeliefPareto;
	strict_e0_resolved_sensitivity: BeliefStrictSensitivity;
}

export interface BeliefArtifactProvenance {
	metrics_code_sha256: string;
	source_manifest_sha256: string;
	source_manifest_path: string;
	scorer_registry: { path: string; sha256: string; bytes: number };
	bootstrap_seed: number;
	bootstrap_resamples: number;
	bootstrap_rng: string;
	ci_level: 0.95;
	log_loss_epsilon: number;
	calibration_bin_edges: number[];
	minimum_valid_bootstrap_fraction: number;
	evaluation_set_digest_method: string;
	runtime: { python: string; numpy: string; scikit_learn: string };
}

export interface BeliefArtifactAvailable {
	status: 'available';
	frozen_at: string;
	provenance: BeliefArtifactProvenance;
	panels: [BeliefSubstrate, BeliefSubstrate];
	reasons: [];
}

export interface BeliefArtifactUnavailable {
	status: 'unavailable';
	frozen_at: string | null;
	provenance: BeliefArtifactProvenance | null;
	panels: [];
	reasons: string[];
}

export type BeliefArtifactValidation = BeliefArtifactAvailable | BeliefArtifactUnavailable;

type UnknownRecord = Record<string, unknown>;

class SchemaError extends Error {}

const SHA256_RE = /^[a-f0-9]{64}$/i;
const IDENTIFIER_RE = /^[a-z0-9][a-z0-9._:-]*$/i;
const DECIMAL_RE = /^(?:0|[1-9][0-9]*)(?:\.[0-9]+)?$/;
const DATE_RE = /^\d{4}-\d{2}-\d{2}$/;
const COST_KEYS = [
	'status',
	'record_type',
	'inference_usd_total',
	'inference_usd_total_exact',
	'usd_per_1k_statements',
	'provider_measured_usd_total',
	'provider_measured_usd_total_exact',
	'conservative_reserved_usd_total',
	'conservative_reserved_usd_total_exact',
	'inference_usd_lower',
	'inference_usd_lower_exact',
	'inference_usd_upper',
	'inference_usd_upper_exact',
	'usd_per_1k_statements_lower',
	'usd_per_1k_statements_upper',
	'basis',
	'view_id',
	'includes_retries',
	'includes_relation_subcalls',
	'denominator',
	'scope',
	'execution_count',
	'attempt_count',
	'retry_attempt_count',
	'successful_attempt_count',
	'error_attempt_count',
	'provider_measured_call_count',
	'conservative_call_count',
	'input_tokens',
	'output_tokens',
	'token_accounting_complete',
	'ledger_path',
	'ledger_sha256',
	'price_source',
	'price_date',
	'cost_comparability_id',
	'pricing',
	'projection',
	'counterfactual_run_cost',
	'shared_run_id',
	'additive_across_panels',
	'reason'
] as const;
const THRESHOLD_KEYS = [
	'status',
	'value',
	'operator',
	'source_path',
	'source_sha256',
	'frozen_at',
	'reason',
	'confusion',
	'metrics'
] as const;

function fail(context: string, message: string): never {
	throw new SchemaError(`${context}: ${message}`);
}

function record(value: unknown, context: string): UnknownRecord {
	if (value == null || typeof value !== 'object' || Array.isArray(value)) {
		fail(context, 'expected an object');
	}
	return value as UnknownRecord;
}

function exactKeys(value: UnknownRecord, keys: readonly string[], context: string): void {
	const expected = new Set(keys);
	const missing = keys.filter((key) => !(key in value));
	const extra = Object.keys(value).filter((key) => !expected.has(key));
	if (missing.length || extra.length) {
		fail(
			context,
			[
				missing.length ? `missing fields ${missing.join(', ')}` : '',
				extra.length ? `unexpected fields ${extra.join(', ')}` : ''
			]
				.filter(Boolean)
				.join('; ')
		);
	}
}

function string(value: unknown, context: string): string {
	if (typeof value !== 'string' || value.trim() === '') fail(context, 'expected a non-empty string');
	return value;
}

function optionalString(value: unknown, context: string): string | null {
	return value === null ? null : string(value, context);
}

function identifier(value: unknown, context: string): string {
	const parsed = string(value, context);
	if (!IDENTIFIER_RE.test(parsed)) fail(context, 'expected a stable identifier');
	return parsed;
}

function sha256(value: unknown, context: string): string {
	if (typeof value !== 'string' || !SHA256_RE.test(value)) fail(context, 'expected a SHA-256 digest');
	return value.toLowerCase();
}

function exactNonnegativeDecimal(value: unknown, context: string): string {
	const parsed = string(value, context);
	if (!DECIMAL_RE.test(parsed) || !Number.isFinite(Number(parsed))) {
		fail(context, 'expected a canonical non-negative fixed-point decimal string');
	}
	return parsed;
}

function finite(value: unknown, context: string): number {
	if (typeof value !== 'number' || !Number.isFinite(value)) fail(context, 'expected a finite number');
	return value;
}

function ranged(value: unknown, context: string, lower: number, upper: number): number {
	const parsed = finite(value, context);
	if (parsed < lower || parsed > upper) fail(context, `outside [${lower}, ${upper}]`);
	return parsed;
}

function integer(value: unknown, context: string, minimum = 0): number {
	const parsed = finite(value, context);
	if (!Number.isInteger(parsed) || parsed < minimum) fail(context, `expected an integer >= ${minimum}`);
	return parsed;
}

function boolean(value: unknown, context: string): boolean {
	if (typeof value !== 'boolean') fail(context, 'expected boolean');
	return value;
}

function isoTimestamp(value: unknown, context: string): string {
	const parsed = string(value, context);
	if (!parsed.includes('T') || !Number.isFinite(Date.parse(parsed))) fail(context, 'expected an ISO-8601 timestamp');
	return parsed;
}

function exactStringArray(value: unknown, context: string): string[] {
	if (!Array.isArray(value)) fail(context, 'expected an array');
	const parsed = value.map((item, index) => identifier(item, `${context}[${index}]`));
	if (new Set(parsed).size !== parsed.length) fail(context, 'duplicate identifiers');
	return parsed;
}

function sameMembers(got: string[], want: string[]): boolean {
	return got.length === want.length && [...got].sort().every((value, index) => value === [...want].sort()[index]);
}

function contractFingerprint(contract: BeliefContractIdentity): string {
	return [
		contract.prediction_unit,
		contract.gold_rule,
		contract.substrate_sha256,
		contract.gold_sha256,
		contract.evaluation_set_sha256
	].join('|');
}

function parseContract(
	value: unknown,
	context: string,
	expectedGoldRule: BeliefContractIdentity['gold_rule'] = BELIEF_GOLD_RULE
): BeliefContractIdentity {
	const obj = record(value, context);
	exactKeys(
		obj,
		['prediction_unit', 'gold_rule', 'substrate_sha256', 'gold_sha256', 'evaluation_set_sha256'],
		context
	);
	if (obj.prediction_unit !== BELIEF_PREDICTION_UNIT) fail(`${context}.prediction_unit`, `must be ${BELIEF_PREDICTION_UNIT}`);
	if (obj.gold_rule !== expectedGoldRule) fail(`${context}.gold_rule`, `must be ${expectedGoldRule}`);
	return {
		prediction_unit: BELIEF_PREDICTION_UNIT,
		gold_rule: expectedGoldRule,
		substrate_sha256: sha256(obj.substrate_sha256, `${context}.substrate_sha256`),
		gold_sha256: sha256(obj.gold_sha256, `${context}.gold_sha256`),
		evaluation_set_sha256: sha256(obj.evaluation_set_sha256, `${context}.evaluation_set_sha256`)
	};
}

function requireContract(got: BeliefContractIdentity, want: BeliefContractIdentity, context: string): void {
	if (contractFingerprint(got) !== contractFingerprint(want)) fail(context, 'contract identity does not match panel');
}

function parseEstimate(
	value: unknown,
	context: string,
	resamples: number,
	minimumValidFraction: number,
	range: [number, number] | null
): BeliefEstimate {
	const obj = record(value, context);
	exactKeys(obj, ['estimate', 'ci95', 'method', 'resamples', 'valid_resamples'], context);
	const estimate = finite(obj.estimate, `${context}.estimate`);
	if (!Array.isArray(obj.ci95) || obj.ci95.length !== 2) fail(`${context}.ci95`, 'expected [lower, upper]');
	const lower = finite(obj.ci95[0], `${context}.ci95[0]`);
	const upper = finite(obj.ci95[1], `${context}.ci95[1]`);
	if (lower > upper) fail(`${context}.ci95`, 'lower bound exceeds upper bound');
	if (range && (estimate < range[0] || estimate > range[1] || lower < range[0] || upper > range[1])) {
		fail(context, `estimate or CI outside [${range[0]}, ${range[1]}]`);
	}
	if (integer(obj.resamples, `${context}.resamples`, 1) !== resamples) fail(`${context}.resamples`, 'does not match artifact bootstrap count');
	const valid = integer(obj.valid_resamples, `${context}.valid_resamples`, 1);
	if (valid > resamples) fail(`${context}.valid_resamples`, 'exceeds requested resamples');
	if (valid < Math.ceil(resamples * minimumValidFraction)) fail(`${context}.valid_resamples`, 'below artifact minimum-valid fraction');
	const method = string(obj.method, `${context}.method`);
	if (!method.includes('stratified independently within') || !method.includes('frozen fold')) {
		fail(`${context}.method`, 'must disclose the stratified-within-frozen-fold paired bootstrap');
	}
	return {
		estimate,
		ci95: [lower, upper],
		method,
		resamples,
		valid_resamples: valid
	};
}

function parseFoldedEstimate(value: unknown, context: string, resamples: number, minimumValidFraction: number, n: number, foldCount: number): BeliefFoldedEstimate {
	const obj = record(value, context);
	exactKeys(
		obj,
		['estimate', 'ci95', 'method', 'resamples', 'valid_resamples', 'fold_estimates', 'fold_population_sd'],
		context
	);
	const base = parseEstimate(
		{
			estimate: obj.estimate,
			ci95: obj.ci95,
			method: obj.method,
			resamples: obj.resamples,
			valid_resamples: obj.valid_resamples
		},
		context,
		resamples,
		minimumValidFraction,
		[0, 1]
	);
	if (!Array.isArray(obj.fold_estimates) || obj.fold_estimates.length !== foldCount) fail(`${context}.fold_estimates`, `expected ${foldCount} folds`);
	const seen = new Set<number>();
	const folds = obj.fold_estimates.map((raw, index) => {
		const foldContext = `${context}.fold_estimates[${index}]`;
		const fold = record(raw, foldContext);
		exactKeys(fold, ['fold_id', 'n', 'positive', 'negative', 'estimate'], foldContext);
		const foldId = integer(fold.fold_id, `${foldContext}.fold_id`);
		if (seen.has(foldId)) fail(foldContext, `duplicate fold_id ${foldId}`);
		seen.add(foldId);
		const foldN = integer(fold.n, `${foldContext}.n`, 1);
		const positive = integer(fold.positive, `${foldContext}.positive`, 1);
		const negative = integer(fold.negative, `${foldContext}.negative`, 1);
		if (positive + negative !== foldN) fail(foldContext, 'class counts do not sum to fold n');
		return { fold_id: foldId, n: foldN, positive, negative, estimate: ranged(fold.estimate, `${foldContext}.estimate`, 0, 1) };
	});
	if (folds.reduce((sum, fold) => sum + fold.n, 0) !== n) fail(`${context}.fold_estimates`, 'fold counts do not sum to panel n');
	const mean = folds.reduce((sum, fold) => sum + fold.estimate, 0) / folds.length;
	if (Math.abs(mean - base.estimate) > 1e-12) fail(context, 'estimate is not arithmetic mean of fold estimates');
	const sd = finite(obj.fold_population_sd, `${context}.fold_population_sd`);
	const expectedSd = Math.sqrt(folds.reduce((sum, fold) => sum + (fold.estimate - mean) ** 2, 0) / folds.length);
	if (sd < 0 || Math.abs(sd - expectedSd) > 1e-12) fail(`${context}.fold_population_sd`, 'does not match fold estimates');
	return { ...base, fold_estimates: folds, fold_population_sd: sd };
}

function parseReliabilityBins(value: unknown, context: string, edges: number[], n: number): BeliefReliabilityBin[] {
	if (!Array.isArray(value) || value.length !== edges.length - 1) fail(context, `expected ${edges.length - 1} frozen bins`);
	const bins = value.map((raw, index) => {
		const binContext = `${context}[${index}]`;
		const obj = record(raw, binContext);
		exactKeys(obj, ['bin_index', 'lower', 'upper', 'upper_inclusive', 'n', 'mean_prediction', 'observed_fraction'], binContext);
		if (integer(obj.bin_index, `${binContext}.bin_index`) !== index) fail(`${binContext}.bin_index`, 'not in frozen order');
		const lower = ranged(obj.lower, `${binContext}.lower`, 0, 1);
		const upper = ranged(obj.upper, `${binContext}.upper`, 0, 1);
		if (lower !== edges[index] || upper !== edges[index + 1]) fail(binContext, 'bounds do not match artifact bin edges');
		const upperInclusive = boolean(obj.upper_inclusive, `${binContext}.upper_inclusive`);
		if (upperInclusive !== (index === edges.length - 2)) fail(`${binContext}.upper_inclusive`, 'only final bin may include upper edge');
		const count = integer(obj.n, `${binContext}.n`);
		let meanPrediction: number | null;
		let observedFraction: number | null;
		if (count === 0) {
			if (obj.mean_prediction !== null || obj.observed_fraction !== null) fail(binContext, 'empty bin values must be null');
			meanPrediction = null;
			observedFraction = null;
		} else {
			meanPrediction = ranged(obj.mean_prediction, `${binContext}.mean_prediction`, lower, upper);
			observedFraction = ranged(obj.observed_fraction, `${binContext}.observed_fraction`, 0, 1);
		}
		return { bin_index: index, lower, upper, upper_inclusive: upperInclusive, n: count, mean_prediction: meanPrediction, observed_fraction: observedFraction };
	});
	if (bins.reduce((sum, bin) => sum + bin.n, 0) !== n) fail(context, 'bin counts do not sum to panel n');
	return bins;
}

function parseThreshold(value: unknown, context: string, resamples: number, minimumValidFraction: number, n: number): BeliefThreshold {
	const obj = record(value, context);
	exactKeys(obj, THRESHOLD_KEYS, context);
	if (obj.status === 'unavailable') {
		for (const key of ['value', 'operator', 'source_path', 'source_sha256', 'frozen_at', 'confusion', 'metrics']) {
			if (obj[key] !== null) fail(`${context}.${key}`, 'must be null when threshold is unavailable');
		}
		return { status: 'unavailable', value: null, operator: null, source_path: null, source_sha256: null, frozen_at: null, reason: string(obj.reason, `${context}.reason`), confusion: null, metrics: null };
	}
	if (obj.status !== 'available') fail(`${context}.status`, 'expected available or unavailable');
	if (obj.operator !== 'greater_than_or_equal') fail(`${context}.operator`, 'must be greater_than_or_equal');
	if (obj.reason !== null) fail(`${context}.reason`, 'must be null when threshold is available');
	const confusionObj = record(obj.confusion, `${context}.confusion`);
	exactKeys(confusionObj, ['tp', 'fp', 'fn', 'tn'], `${context}.confusion`);
	const confusion = {
		tp: integer(confusionObj.tp, `${context}.confusion.tp`),
		fp: integer(confusionObj.fp, `${context}.confusion.fp`),
		fn: integer(confusionObj.fn, `${context}.confusion.fn`),
		tn: integer(confusionObj.tn, `${context}.confusion.tn`)
	};
	if (confusion.tp + confusion.fp + confusion.fn + confusion.tn !== n) fail(`${context}.confusion`, 'counts do not sum to panel n');
	const metricObj = record(obj.metrics, `${context}.metrics`);
	exactKeys(metricObj, ['accuracy', 'precision', 'recall', 'f1'], `${context}.metrics`);
	return {
		status: 'available',
		value: ranged(obj.value, `${context}.value`, 0, 1),
		operator: 'greater_than_or_equal',
		source_path: string(obj.source_path, `${context}.source_path`),
		source_sha256: sha256(obj.source_sha256, `${context}.source_sha256`),
		frozen_at: isoTimestamp(obj.frozen_at, `${context}.frozen_at`),
		reason: null,
		confusion,
		metrics: {
			accuracy: parseEstimate(metricObj.accuracy, `${context}.metrics.accuracy`, resamples, minimumValidFraction, [0, 1]),
			precision: parseEstimate(metricObj.precision, `${context}.metrics.precision`, resamples, minimumValidFraction, [0, 1]),
			recall: parseEstimate(metricObj.recall, `${context}.metrics.recall`, resamples, minimumValidFraction, [0, 1]),
			f1: parseEstimate(metricObj.f1, `${context}.metrics.f1`, resamples, minimumValidFraction, [0, 1])
		}
	};
}

function parseMetrics(value: unknown, context: string, resamples: number, minimumValidFraction: number, edges: number[], n: number, foldCount: number): BeliefArmMetrics {
	const obj = record(value, context);
	exactKeys(obj, ['fold_mean_trapezoidal_pr_auc', 'pooled_average_precision', 'auroc', 'brier', 'log_loss', 'calibration', 'threshold'], context);
	const calibrationObj = record(obj.calibration, `${context}.calibration`);
	exactKeys(calibrationObj, ['ece', 'intercept', 'slope', 'intercept_abs_error', 'slope_abs_error', 'reliability_bins'], `${context}.calibration`);
	const intercept = parseEstimate(calibrationObj.intercept, `${context}.calibration.intercept`, resamples, minimumValidFraction, null);
	const slope = parseEstimate(calibrationObj.slope, `${context}.calibration.slope`, resamples, minimumValidFraction, null);
	const interceptAbsError = parseEstimate(calibrationObj.intercept_abs_error, `${context}.calibration.intercept_abs_error`, resamples, minimumValidFraction, [0, Infinity]);
	const slopeAbsError = parseEstimate(calibrationObj.slope_abs_error, `${context}.calibration.slope_abs_error`, resamples, minimumValidFraction, [0, Infinity]);
	if (Math.abs(interceptAbsError.estimate - Math.abs(intercept.estimate)) > 1e-12) fail(`${context}.calibration.intercept_abs_error`, 'point estimate must equal |intercept|');
	if (Math.abs(slopeAbsError.estimate - Math.abs(slope.estimate - 1)) > 1e-12) fail(`${context}.calibration.slope_abs_error`, 'point estimate must equal |slope - 1|');
	return {
		fold_mean_trapezoidal_pr_auc: parseFoldedEstimate(obj.fold_mean_trapezoidal_pr_auc, `${context}.fold_mean_trapezoidal_pr_auc`, resamples, minimumValidFraction, n, foldCount),
		pooled_average_precision: parseEstimate(obj.pooled_average_precision, `${context}.pooled_average_precision`, resamples, minimumValidFraction, [0, 1]),
		auroc: parseEstimate(obj.auroc, `${context}.auroc`, resamples, minimumValidFraction, [0, 1]),
		brier: parseEstimate(obj.brier, `${context}.brier`, resamples, minimumValidFraction, [0, 1]),
		log_loss: parseEstimate(obj.log_loss, `${context}.log_loss`, resamples, minimumValidFraction, [0, Infinity]),
		calibration: {
			ece: parseEstimate(calibrationObj.ece, `${context}.calibration.ece`, resamples, minimumValidFraction, [0, 1]),
			intercept,
			slope,
			intercept_abs_error: interceptAbsError,
			slope_abs_error: slopeAbsError,
			reliability_bins: parseReliabilityBins(calibrationObj.reliability_bins, `${context}.calibration.reliability_bins`, edges, n)
		},
		threshold: parseThreshold(obj.threshold, `${context}.threshold`, resamples, minimumValidFraction, n)
	};
}

function parseCoverage(value: unknown, context: string, n: number): BeliefCoverage {
	const obj = record(value, context);
	exactKeys(obj, ['eligible', 'predicted', 'invalid', 'fraction'], context);
	const coverage = {
		eligible: integer(obj.eligible, `${context}.eligible`, 1),
		predicted: integer(obj.predicted, `${context}.predicted`),
		invalid: integer(obj.invalid, `${context}.invalid`),
		fraction: ranged(obj.fraction, `${context}.fraction`, 0, 1)
	};
	if (coverage.eligible !== n || coverage.predicted !== n || coverage.invalid !== 0 || coverage.fraction !== 1) fail(context, 'formal panel requires predicted = eligible = panel n, invalid = 0, fraction = 1');
	return coverage;
}

function parsePricing(value: unknown, context: string): BeliefPricing {
	const obj = record(value, context);
	exactKeys(
		obj,
		[
			'cost_comparability_id',
			'currency',
			'provider',
			'provider_model_id',
			'pricing_mode',
			'region',
			'resolved_service_tier',
			'retrieved_on',
			'service_tier_request',
			'source_url',
			'tariff',
			'unit'
		],
		context
	);
	if (obj.currency !== 'USD') fail(`${context}.currency`, 'must be USD');
	if (obj.pricing_mode !== 'on_demand') fail(`${context}.pricing_mode`, 'must be on_demand');
	if (obj.resolved_service_tier !== 'standard') fail(`${context}.resolved_service_tier`, 'must be standard');
	if (obj.service_tier_request !== 'default') fail(`${context}.service_tier_request`, 'must be default');
	if (obj.unit !== 'per_million_tokens') fail(`${context}.unit`, 'must be per_million_tokens');
	const sourceUrl = string(obj.source_url, `${context}.source_url`);
	if (!sourceUrl.startsWith('https://')) fail(`${context}.source_url`, 'must be an HTTPS pricing source');
	const retrievedOn = string(obj.retrieved_on, `${context}.retrieved_on`);
	if (!DATE_RE.test(retrievedOn)) fail(`${context}.retrieved_on`, 'expected YYYY-MM-DD');
	const tariffObj = record(obj.tariff, `${context}.tariff`);
	exactKeys(
		tariffObj,
		['input_usd_per_million', 'output_usd_per_million', 'pricing_basis'],
		`${context}.tariff`
	);
	return {
		cost_comparability_id: identifier(
			obj.cost_comparability_id,
			`${context}.cost_comparability_id`
		),
		currency: 'USD',
		provider: string(obj.provider, `${context}.provider`),
		provider_model_id: string(obj.provider_model_id, `${context}.provider_model_id`),
		pricing_mode: 'on_demand',
		region: string(obj.region, `${context}.region`),
		resolved_service_tier: 'standard',
		retrieved_on: retrievedOn,
		service_tier_request: 'default',
		source_url: sourceUrl,
		tariff: {
			input_usd_per_million: exactNonnegativeDecimal(
				tariffObj.input_usd_per_million,
				`${context}.tariff.input_usd_per_million`
			),
			output_usd_per_million: exactNonnegativeDecimal(
				tariffObj.output_usd_per_million,
				`${context}.tariff.output_usd_per_million`
			),
			pricing_basis: string(tariffObj.pricing_basis, `${context}.tariff.pricing_basis`)
		},
		unit: 'per_million_tokens'
	};
}

function parseCost(value: unknown, context: string, n: number): BeliefCost {
	const obj = record(value, context);
	exactKeys(obj, COST_KEYS, context);
	if (obj.status === 'unavailable') {
		const expectedNull = COST_KEYS.filter((key) => !['status', 'basis', 'reason'].includes(key));
		for (const key of expectedNull) if (obj[key] !== null) fail(`${context}.${key}`, 'must be null when cost is unavailable');
		if (obj.basis !== 'unavailable') fail(`${context}.basis`, 'must be unavailable');
		return {
			status: 'unavailable', record_type: null, inference_usd_total: null, inference_usd_total_exact: null, usd_per_1k_statements: null,
			provider_measured_usd_total: null, provider_measured_usd_total_exact: null, conservative_reserved_usd_total: null, conservative_reserved_usd_total_exact: null,
			inference_usd_lower: null, inference_usd_lower_exact: null, inference_usd_upper: null, inference_usd_upper_exact: null,
			usd_per_1k_statements_lower: null, usd_per_1k_statements_upper: null,
			basis: 'unavailable', view_id: null, includes_retries: null, includes_relation_subcalls: null, denominator: null, scope: null,
			execution_count: null,
			attempt_count: null, retry_attempt_count: null, successful_attempt_count: null, error_attempt_count: null,
			provider_measured_call_count: null, conservative_call_count: null,
			input_tokens: null, output_tokens: null, token_accounting_complete: null, ledger_path: null, ledger_sha256: null, price_source: null, price_date: null,
			cost_comparability_id: null, pricing: null, projection: null, counterfactual_run_cost: null,
			shared_run_id: null, additive_across_panels: null,
			reason: string(obj.reason, `${context}.reason`)
		};
	}
	if (obj.status !== 'available') fail(`${context}.status`, 'expected available or unavailable');
	if (!['provider_measured_observed', 'mixed_conservative_upper_bound'].includes(String(obj.basis))) fail(`${context}.basis`, 'unsupported cost basis');
	if (obj.includes_retries !== true) fail(`${context}.includes_retries`, 'available cost must include retries');
	if (obj.reason !== null) fail(`${context}.reason`, 'must be null when cost is available');
	if (obj.record_type !== 'evidence_execution') fail(`${context}.record_type`, 'expected evidence_execution');
	const total = ranged(obj.inference_usd_total, `${context}.inference_usd_total`, 0, Infinity);
	const totalExact = exactNonnegativeDecimal(obj.inference_usd_total_exact, `${context}.inference_usd_total_exact`);
	const close = (a: number, b: number): boolean => Math.abs(a - b) <= Math.max(1e-15, Math.max(Math.abs(a), Math.abs(b)) * 1e-12);
	const per1k = ranged(obj.usd_per_1k_statements, `${context}.usd_per_1k_statements`, 0, Infinity);
	const lower = ranged(obj.inference_usd_lower, `${context}.inference_usd_lower`, 0, Infinity);
	const lowerExact = exactNonnegativeDecimal(obj.inference_usd_lower_exact, `${context}.inference_usd_lower_exact`);
	const upper = ranged(obj.inference_usd_upper, `${context}.inference_usd_upper`, 0, Infinity);
	const upperExact = exactNonnegativeDecimal(obj.inference_usd_upper_exact, `${context}.inference_usd_upper_exact`);
	const lowerPer1k = ranged(obj.usd_per_1k_statements_lower, `${context}.usd_per_1k_statements_lower`, 0, Infinity);
	const upperPer1k = ranged(obj.usd_per_1k_statements_upper, `${context}.usd_per_1k_statements_upper`, 0, Infinity);
	if (!close(Number(totalExact), total) || !close(Number(lowerExact), lower) || !close(Number(upperExact), upper)) fail(context, 'numeric and exact-decimal costs disagree');
	if (!close(total, upper) || totalExact !== upperExact || !close(per1k, upperPer1k)) fail(context, 'point cost must equal the explicit upper endpoint');
	if (!close(lowerPer1k, (lower * 1000) / n) || !close(upperPer1k, (upper * 1000) / n)) fail(context, 'normalized cost interval does not reconcile to endpoints and panel n');
	if (lower > upper) fail(context, 'cost lower endpoint exceeds upper endpoint');
	const executionCount = integer(obj.execution_count, `${context}.execution_count`, 1);
	const attemptCount = integer(obj.attempt_count, `${context}.attempt_count`, executionCount);
	const successful = integer(obj.successful_attempt_count, `${context}.successful_attempt_count`);
	const error = integer(obj.error_attempt_count, `${context}.error_attempt_count`);
	if (successful + error !== attemptCount) fail(context, 'attempt status counts do not sum to attempt_count');
	const retry = integer(obj.retry_attempt_count, `${context}.retry_attempt_count`);
	const providerMeasured = integer(obj.provider_measured_call_count, `${context}.provider_measured_call_count`);
	const conservative = integer(obj.conservative_call_count, `${context}.conservative_call_count`);
	const costSubtotal = (numericRaw: unknown, exactRaw: unknown, field: string): [number, string] => {
		const numeric = ranged(numericRaw, `${context}.${field}`, 0, Infinity);
		const exact = exactNonnegativeDecimal(exactRaw, `${context}.${field}_exact`);
		if (!close(numeric, Number(exact))) fail(`${context}.${field}`, 'numeric and exact values disagree');
		return [numeric, exact];
	};
	const [providerMeasuredUsd, providerMeasuredUsdExact] = costSubtotal(obj.provider_measured_usd_total, obj.provider_measured_usd_total_exact, 'provider_measured_usd_total');
	const [conservativeReservedUsd, conservativeReservedUsdExact] = costSubtotal(obj.conservative_reserved_usd_total, obj.conservative_reserved_usd_total_exact, 'conservative_reserved_usd_total');
	if (attemptCount !== executionCount + retry) fail(context, 'evidence attempts must equal execution identities plus per-identity retries');
	if (successful !== executionCount || error !== retry) fail(context, 'evidence final-success and failed-retry counts do not reconcile by execution identity');
	if (!close(lower, providerMeasuredUsd) || !close(upper, providerMeasuredUsd + conservativeReservedUsd)) fail(context, 'cost endpoints do not reconcile to measured plus reserved subtotals');
	const expectedBasis = conservative > 0 ? 'mixed_conservative_upper_bound' : 'provider_measured_observed';
	if (obj.basis !== expectedBasis) fail(`${context}.basis`, 'does not follow call-level accounting provenance');
	if (obj.view_id !== 'provider-runtime-retry-inclusive') fail(`${context}.view_id`, 'evidence costs must share provider-runtime-retry-inclusive');
	if (obj.includes_relation_subcalls !== true) fail(`${context}.includes_relation_subcalls`, 'evidence cost must include relation subcalls');
	const denominatorObj = record(obj.denominator, `${context}.denominator`);
	exactKeys(denominatorObj, ['statements', 'evidence_executions'], `${context}.denominator`);
	const denominator = {
		statements: integer(denominatorObj.statements, `${context}.denominator.statements`, 1),
		evidence_executions: integer(denominatorObj.evidence_executions, `${context}.denominator.evidence_executions`, 1)
	};
	if (denominator.statements !== n || denominator.evidence_executions !== executionCount) fail(`${context}.denominator`, 'does not reconcile to panel and ledger counts');
	const scopeObj = record(obj.scope, `${context}.scope`);
	exactKeys(scopeObj, ['included_cost_categories', 'excluded_cost_categories'], `${context}.scope`);
	if (JSON.stringify(scopeObj.included_cost_categories) !== JSON.stringify(['provider_inference_calls']) || JSON.stringify(scopeObj.excluded_cost_categories) !== JSON.stringify(['training', 'local_aggregation', 'feature_materialization', 'upstream_reading'])) fail(`${context}.scope`, 'cost inclusion/exclusion categories drifted');
	const tokenComplete = boolean(obj.token_accounting_complete, `${context}.token_accounting_complete`);
	const parseToken = (raw: unknown, tokenContext: string): number | null => raw === null ? null : integer(raw, tokenContext);
	const inputTokens = parseToken(obj.input_tokens, `${context}.input_tokens`);
	const outputTokens = parseToken(obj.output_tokens, `${context}.output_tokens`);
	if (tokenComplete !== (inputTokens !== null && outputTokens !== null)) fail(context, 'token completeness marker disagrees with token totals');
	if (tokenComplete !== (conservative === 0)) fail(context, 'token completeness marker disagrees with conservative calls');
	const costComparabilityId = identifier(
		obj.cost_comparability_id,
		`${context}.cost_comparability_id`
	);
	const pricing = parsePricing(obj.pricing, `${context}.pricing`);
	if (pricing.cost_comparability_id !== costComparabilityId) {
		fail(context, 'top-level and pricing cost comparability IDs disagree');
	}
	const priceSource = string(obj.price_source, `${context}.price_source`);
	const priceDate = string(obj.price_date, `${context}.price_date`);
	if (priceSource !== pricing.source_url || priceDate !== pricing.retrieved_on) {
		fail(context, 'price source/date disagree with structured pricing provenance');
	}
	if (obj.projection !== 'all_executions' && obj.projection !== 'observed_execution_subset') {
		fail(`${context}.projection`, 'must be all_executions or observed_execution_subset');
	}
	if (obj.counterfactual_run_cost !== false) {
		fail(`${context}.counterfactual_run_cost`, 'must be false');
	}
	if (obj.additive_across_panels !== false) {
		fail(`${context}.additive_across_panels`, 'must be false');
	}
	return {
		status: 'available', record_type: 'evidence_execution', inference_usd_total: total, inference_usd_total_exact: totalExact, usd_per_1k_statements: per1k,
		provider_measured_usd_total: providerMeasuredUsd, provider_measured_usd_total_exact: providerMeasuredUsdExact,
		conservative_reserved_usd_total: conservativeReservedUsd, conservative_reserved_usd_total_exact: conservativeReservedUsdExact,
		inference_usd_lower: lower, inference_usd_lower_exact: lowerExact, inference_usd_upper: upper, inference_usd_upper_exact: upperExact,
		usd_per_1k_statements_lower: lowerPer1k, usd_per_1k_statements_upper: upperPer1k,
		basis: obj.basis as BeliefCostBasis, view_id: identifier(obj.view_id, `${context}.view_id`), includes_retries: true,
		includes_relation_subcalls: true, denominator,
		scope: { included_cost_categories: ['provider_inference_calls'], excluded_cost_categories: ['training', 'local_aggregation', 'feature_materialization', 'upstream_reading'] },
		execution_count: executionCount,
		attempt_count: attemptCount, retry_attempt_count: retry, successful_attempt_count: successful, error_attempt_count: error,
		provider_measured_call_count: providerMeasured, conservative_call_count: conservative,
		input_tokens: inputTokens, output_tokens: outputTokens, token_accounting_complete: tokenComplete,
		ledger_path: string(obj.ledger_path, `${context}.ledger_path`), ledger_sha256: sha256(obj.ledger_sha256, `${context}.ledger_sha256`),
		price_source: priceSource, price_date: priceDate,
		cost_comparability_id: costComparabilityId, pricing,
		projection: obj.projection,
		counterfactual_run_cost: false,
		shared_run_id: identifier(obj.shared_run_id, `${context}.shared_run_id`),
		additive_across_panels: false,
		reason: null
	};
}

function parseArmPareto(value: unknown, context: string, cost: BeliefCost): BeliefArmPareto {
	const obj = record(value, context);
	exactKeys(obj, ['status', 'view_id', 'basis', 'point_pareto', 'uncertainty_pareto', 'reason'], context);
	if (cost.status === 'unavailable') {
		if (obj.status !== 'unavailable' || obj.view_id !== null || obj.basis !== null || obj.point_pareto !== null || obj.uncertainty_pareto !== null) fail(context, 'cost-unavailable arm must be Pareto-unavailable');
		return { status: 'unavailable', view_id: null, basis: null, point_pareto: null, uncertainty_pareto: null, reason: string(obj.reason, `${context}.reason`) };
	}
	if (obj.status !== 'available' || obj.reason !== null) fail(context, 'cost-available arm must have available Pareto membership');
	if (obj.view_id !== cost.view_id || obj.basis !== cost.basis) fail(context, 'Pareto view/basis disagrees with arm cost');
	return { status: 'available', view_id: cost.view_id, basis: cost.basis, point_pareto: boolean(obj.point_pareto, `${context}.point_pareto`), uncertainty_pareto: boolean(obj.uncertainty_pareto, `${context}.uncertainty_pareto`), reason: null };
}

function parseArmProvenance(value: unknown, context: string): BeliefArmProvenance {
	const obj = record(value, context);
	exactKeys(obj, ['implementation', 'implementation_digest', 'training_data_sha256', 'environment', 'notes', 'predictions_path', 'predictions_sha256'], context);
	return {
		implementation: string(obj.implementation, `${context}.implementation`),
		implementation_digest: string(obj.implementation_digest, `${context}.implementation_digest`),
		training_data_sha256: obj.training_data_sha256 === null ? null : sha256(obj.training_data_sha256, `${context}.training_data_sha256`),
		environment: string(obj.environment, `${context}.environment`),
		notes: optionalString(obj.notes, `${context}.notes`),
		predictions_path: string(obj.predictions_path, `${context}.predictions_path`),
		predictions_sha256: sha256(obj.predictions_sha256, `${context}.predictions_sha256`)
	};
}

function parseComparison(value: unknown, context: string, contract: BeliefContractIdentity, armIds: Set<string>, resamples: number, minimumValidFraction: number): BeliefPairedComparison {
	const obj = record(value, context);
	exactKeys(obj, ['a_arm_id', 'b_arm_id', 'metric', 'direction', 'better_when', 'contract', 'delta', 'resamples', 'method'], context);
	const a = identifier(obj.a_arm_id, `${context}.a_arm_id`);
	const b = identifier(obj.b_arm_id, `${context}.b_arm_id`);
	if (a === b || !armIds.has(a) || !armIds.has(b)) fail(context, 'comparison arms must be two distinct arms in this panel');
	if (![...BELIEF_PRIMARY_METRICS, ...BELIEF_THRESHOLD_METRICS].includes(obj.metric as BeliefMetricKey)) fail(`${context}.metric`, 'unsupported metric');
	if (obj.direction !== 'b_minus_a') fail(`${context}.direction`, 'must be b_minus_a');
	if (obj.better_when !== 'higher' && obj.better_when !== 'lower') fail(`${context}.better_when`, 'unsupported direction');
	const higherMetrics = new Set<BeliefMetricKey>([
		'fold_mean_trapezoidal_pr_auc', 'pooled_average_precision', 'auroc',
		'threshold_accuracy', 'threshold_precision', 'threshold_recall', 'threshold_f1'
	]);
	const expectedBetterWhen: BeliefBetterWhen = higherMetrics.has(obj.metric as BeliefMetricKey) ? 'higher' : 'lower';
	if (obj.better_when !== expectedBetterWhen) fail(`${context}.better_when`, `must be ${expectedBetterWhen} for ${String(obj.metric)}`);
	requireContract(
		parseContract(obj.contract, `${context}.contract`, contract.gold_rule),
		contract,
		`${context}.contract`
	);
	if (integer(obj.resamples, `${context}.resamples`, 1) !== resamples) fail(`${context}.resamples`, 'does not match artifact bootstrap count');
	return {
		a_arm_id: a,
		b_arm_id: b,
		metric: obj.metric as BeliefMetricKey,
		direction: 'b_minus_a',
		better_when: obj.better_when as BeliefBetterWhen,
		contract,
		delta: parseEstimate(obj.delta, `${context}.delta`, resamples, minimumValidFraction, null),
		resamples,
		method: string(obj.method, `${context}.method`)
	};
}

function parseExcludedArms(
	value: unknown,
	context: string,
	evaluatedArmIds: Set<string>
): BeliefExcludedArm[] {
	if (!Array.isArray(value)) fail(context, 'expected an array');
	const seen = new Set<string>();
	return value.map((raw, index) => {
		const itemContext = `${context}[${index}]`;
		const obj = record(raw, itemContext);
		exactKeys(
			obj,
			['arm_id', 'label', 'family', 'status', 'reason', 'required_artifact', 'provenance'],
			itemContext
		);
		const armId = identifier(obj.arm_id, `${itemContext}.arm_id`);
		if (seen.has(armId)) fail(itemContext, `duplicate excluded arm_id ${armId}`);
		if (evaluatedArmIds.has(armId)) fail(itemContext, 'excluded arm is also evaluated');
		seen.add(armId);
		if (obj.family !== 'paper' && obj.family !== 'current' && obj.family !== 'llm') {
			fail(`${itemContext}.family`, 'expected paper, current, or llm');
		}
		if (obj.status !== 'excluded') fail(`${itemContext}.status`, 'must be excluded');
		return {
			arm_id: armId,
			label: string(obj.label, `${itemContext}.label`),
			family: obj.family,
			status: 'excluded',
			reason: string(obj.reason, `${itemContext}.reason`),
			required_artifact: string(obj.required_artifact, `${itemContext}.required_artifact`),
			provenance: string(obj.provenance, `${itemContext}.provenance`)
		};
	});
}
function parsePareto(
	value: unknown,
	context: string,
	arms: BeliefArm[],
	comparisons: BeliefPairedComparison[]
): BeliefPareto {
	const obj = record(value, context);
	exactKeys(obj, ['objective_metric', 'performance_direction', 'cost_axis', 'point_rule', 'uncertainty_rule', 'views'], context);
	if (obj.objective_metric !== BELIEF_PARETO_METRIC) fail(`${context}.objective_metric`, `must be ${BELIEF_PARETO_METRIC}`);
	if (obj.performance_direction !== 'higher_is_better') fail(`${context}.performance_direction`, 'must be higher_is_better');
	if (obj.cost_axis !== 'usd_per_1k_statements_upper') fail(`${context}.cost_axis`, 'must be usd_per_1k_statements_upper');
	if (!Array.isArray(obj.views)) fail(`${context}.views`, 'expected an array');
	const armById = new Map(arms.map((arm) => [arm.arm_id, arm]));
	const apComparison = (candidate: string, challenger: string): { estimate: number; ci95: [number, number] } => {
		const direct = comparisons.find(
			(row) =>
				row.metric === BELIEF_PARETO_METRIC &&
				row.a_arm_id === candidate &&
				row.b_arm_id === challenger
		);
		if (direct) return { estimate: direct.delta.estimate, ci95: direct.delta.ci95 };
		const reverse = comparisons.find(
			(row) =>
				row.metric === BELIEF_PARETO_METRIC &&
				row.a_arm_id === challenger &&
				row.b_arm_id === candidate
		);
		if (!reverse) fail(context, `missing ${BELIEF_PARETO_METRIC} delta for ${candidate} and ${challenger}`);
		return {
			estimate: -reverse.delta.estimate,
			ci95: [-reverse.delta.ci95[1], -reverse.delta.ci95[0]]
		};
	};
	const seenViews = new Set<string>();
	const views = obj.views.map((raw, index) => {
		const viewContext = `${context}.views[${index}]`;
		const view = record(raw, viewContext);
		exactKeys(view, ['view_id', 'basis', 'eligible_arm_ids', 'point_frontier_arm_ids', 'uncertainty_frontier_arm_ids', 'audit'], viewContext);
		const viewId = identifier(view.view_id, `${viewContext}.view_id`);
		if (seenViews.has(viewId)) fail(viewContext, `duplicate view_id ${viewId}`);
		seenViews.add(viewId);
		if (!['provider_measured_observed', 'mixed_conservative_upper_bound', 'mixed'].includes(String(view.basis))) fail(`${viewContext}.basis`, 'unsupported view basis');
		const basis = view.basis as BeliefCostBasis | 'mixed';
		const eligible = exactStringArray(view.eligible_arm_ids, `${viewContext}.eligible_arm_ids`);
		const expectedEligible = arms.filter((arm) => arm.cost.status === 'available' && arm.cost.view_id === viewId).map((arm) => arm.arm_id);
		if (!sameMembers(eligible, expectedEligible)) fail(viewContext, 'eligible arms do not match cost view and basis');
		const comparabilityIds = new Set(
			expectedEligible.map(
				(armId) => (armById.get(armId)!.cost as BeliefCostAvailable).cost_comparability_id
			)
		);
		if (comparabilityIds.size !== 1) {
			fail(viewContext, 'eligible arm costs do not share one cost comparability ID');
		}
		const eligibleBases = new Set(expectedEligible.map((armId) => (armById.get(armId)!.cost as BeliefCostAvailable).basis));
		const expectedBasis = eligibleBases.size === 1 ? [...eligibleBases][0] : 'mixed';
		if (basis !== expectedBasis) fail(`${viewContext}.basis`, 'does not summarize eligible arm cost bases');
		const point = exactStringArray(view.point_frontier_arm_ids, `${viewContext}.point_frontier_arm_ids`);
		const uncertainty = exactStringArray(view.uncertainty_frontier_arm_ids, `${viewContext}.uncertainty_frontier_arm_ids`);
		if (point.some((id) => !eligible.includes(id)) || uncertainty.some((id) => !eligible.includes(id))) fail(viewContext, 'frontier contains an ineligible arm');
		if (!Array.isArray(view.audit)) fail(`${viewContext}.audit`, 'expected an array');
		const audit = view.audit.map((auditRaw, auditIndex) => {
			const auditContext = `${viewContext}.audit[${auditIndex}]`;
			const row = record(auditRaw, auditContext);
			exactKeys(row, ['candidate_arm_id', 'challenger_arm_id', 'candidate_cost_per_1k_interval', 'challenger_cost_per_1k_interval', 'challenger_minus_candidate_cost_per_1k', 'cost_interval_definitely_not_worse', 'challenger_minus_candidate_performance', 'performance_delta_ci95', 'point_dominates', 'uncertainty_dominates'], auditContext);
			const candidate = identifier(row.candidate_arm_id, `${auditContext}.candidate_arm_id`);
			const challenger = identifier(row.challenger_arm_id, `${auditContext}.challenger_arm_id`);
			if (candidate === challenger || !eligible.includes(candidate) || !eligible.includes(challenger)) fail(auditContext, 'audit pair must contain distinct eligible arms');
			if (!Array.isArray(row.performance_delta_ci95) || row.performance_delta_ci95.length !== 2) fail(`${auditContext}.performance_delta_ci95`, 'expected [lower, upper]');
			const lower = finite(row.performance_delta_ci95[0], `${auditContext}.performance_delta_ci95[0]`);
			const upper = finite(row.performance_delta_ci95[1], `${auditContext}.performance_delta_ci95[1]`);
			if (lower > upper) fail(`${auditContext}.performance_delta_ci95`, 'lower bound exceeds upper');
			const costDelta = finite(
				row.challenger_minus_candidate_cost_per_1k,
				`${auditContext}.challenger_minus_candidate_cost_per_1k`
			);
			const performanceDelta = finite(
				row.challenger_minus_candidate_performance,
				`${auditContext}.challenger_minus_candidate_performance`
			);
			const candidateArm = armById.get(candidate)!;
			const challengerArm = armById.get(challenger)!;
			if (candidateArm.cost.status !== 'available' || challengerArm.cost.status !== 'available') {
				fail(auditContext, 'audit references an arm without available cost');
			}
			const parseCostInterval = (raw: unknown, intervalContext: string): [number, number] => {
				if (!Array.isArray(raw) || raw.length !== 2) fail(intervalContext, 'expected [lower, upper]');
				const interval: [number, number] = [
					ranged(raw[0], `${intervalContext}[0]`, 0, Infinity),
					ranged(raw[1], `${intervalContext}[1]`, 0, Infinity)
				];
				if (interval[0] > interval[1]) fail(intervalContext, 'lower endpoint exceeds upper');
				return interval;
			};
			const candidateCostInterval = parseCostInterval(row.candidate_cost_per_1k_interval, `${auditContext}.candidate_cost_per_1k_interval`);
			const challengerCostInterval = parseCostInterval(row.challenger_cost_per_1k_interval, `${auditContext}.challenger_cost_per_1k_interval`);
			const expectedCost =
				challengerArm.cost.usd_per_1k_statements_upper - candidateArm.cost.usd_per_1k_statements_upper;
			const expectedPerformance =
				challengerArm.metrics.fold_mean_trapezoidal_pr_auc.estimate -
				candidateArm.metrics.fold_mean_trapezoidal_pr_auc.estimate;
			const paired = apComparison(candidate, challenger);
			const tolerance = 1e-12;
			if (
				Math.abs(candidateCostInterval[0] - candidateArm.cost.usd_per_1k_statements_lower) > tolerance ||
				Math.abs(candidateCostInterval[1] - candidateArm.cost.usd_per_1k_statements_upper) > tolerance ||
				Math.abs(challengerCostInterval[0] - challengerArm.cost.usd_per_1k_statements_lower) > tolerance ||
				Math.abs(challengerCostInterval[1] - challengerArm.cost.usd_per_1k_statements_upper) > tolerance ||
				Math.abs(costDelta - expectedCost) > tolerance ||
				Math.abs(performanceDelta - expectedPerformance) > tolerance ||
				Math.abs(performanceDelta - paired.estimate) > tolerance ||
				Math.abs(lower - paired.ci95[0]) > tolerance ||
				Math.abs(upper - paired.ci95[1]) > tolerance
			) {
				fail(auditContext, 'cost/performance deltas do not match arm metrics and paired CI');
			}
			const pointCostNotWorse = challengerArm.cost.usd_per_1k_statements_upper <= candidateArm.cost.usd_per_1k_statements_upper;
			const intervalCostNotWorse = challengerArm.cost.usd_per_1k_statements_upper <= candidateArm.cost.usd_per_1k_statements_lower;
			const declaredIntervalCostNotWorse = boolean(row.cost_interval_definitely_not_worse, `${auditContext}.cost_interval_definitely_not_worse`);
			if (declaredIntervalCostNotWorse !== intervalCostNotWorse) fail(auditContext, 'cost interval comparison flag is inconsistent');
			const expectedPoint =
				pointCostNotWorse &&
				performanceDelta >= 0 &&
				(challengerArm.cost.usd_per_1k_statements_upper < candidateArm.cost.usd_per_1k_statements_upper ||
					performanceDelta > 0);
			const expectedUncertainty =
				intervalCostNotWorse &&
				lower >= 0 &&
				(challengerArm.cost.usd_per_1k_statements_upper < candidateArm.cost.usd_per_1k_statements_lower ||
					lower > 0);
			const pointDominates = boolean(row.point_dominates, `${auditContext}.point_dominates`);
			const uncertaintyDominates = boolean(
				row.uncertainty_dominates,
				`${auditContext}.uncertainty_dominates`
			);
			if (pointDominates !== expectedPoint || uncertaintyDominates !== expectedUncertainty) {
				fail(auditContext, 'dominance flags do not follow the declared point/uncertainty rules');
			}
			return {
				candidate_arm_id: candidate,
				challenger_arm_id: challenger,
				candidate_cost_per_1k_interval: candidateCostInterval,
				challenger_cost_per_1k_interval: challengerCostInterval,
				challenger_minus_candidate_cost_per_1k: costDelta,
				cost_interval_definitely_not_worse: declaredIntervalCostNotWorse,
				challenger_minus_candidate_performance: performanceDelta,
				performance_delta_ci95: [lower, upper] as [number, number],
				point_dominates: pointDominates,
				uncertainty_dominates: uncertaintyDominates
			};
		});
		const expectedAuditCount = eligible.length * Math.max(0, eligible.length - 1);
		const auditKeys = audit.map((row) => `${row.candidate_arm_id}|${row.challenger_arm_id}`);
		if (audit.length !== expectedAuditCount || new Set(auditKeys).size !== auditKeys.length) fail(`${viewContext}.audit`, 'must contain every ordered eligible arm pair exactly once');
		const expectedPointFrontier = eligible.filter(
			(armId) => !audit.some((row) => row.candidate_arm_id === armId && row.point_dominates)
		);
		const expectedUncertaintyFrontier = eligible.filter(
			(armId) =>
				!audit.some(
					(row) => row.candidate_arm_id === armId && row.uncertainty_dominates
				)
		);
		if (!sameMembers(point, expectedPointFrontier) || !sameMembers(uncertainty, expectedUncertaintyFrontier)) {
			fail(viewContext, 'frontier lists do not follow the ordered-pair dominance audit');
		}
		for (const armId of eligible) {
			const arm = armById.get(armId)!;
			if (arm.pareto.status !== 'available' || arm.pareto.point_pareto !== point.includes(armId) || arm.pareto.uncertainty_pareto !== uncertainty.includes(armId)) fail(viewContext, `arm ${armId} membership disagrees with frontier lists`);
		}
		return { view_id: viewId, basis, eligible_arm_ids: eligible, point_frontier_arm_ids: point, uncertainty_frontier_arm_ids: uncertainty, audit };
	});
	const expectedViews = new Set(arms.filter((arm) => arm.cost.status === 'available').map((arm) => arm.cost.status === 'available' ? arm.cost.view_id : ''));
	if (views.length !== expectedViews.size || views.some((view) => !expectedViews.has(view.view_id))) fail(`${context}.views`, 'does not cover every available cost view exactly once');
	return {
		objective_metric: BELIEF_PARETO_METRIC,
		performance_direction: 'higher_is_better',
		cost_axis: 'usd_per_1k_statements_upper',
		point_rule: string(obj.point_rule, `${context}.point_rule`),
		uncertainty_rule: string(obj.uncertainty_rule, `${context}.uncertainty_rule`),
		views
	};
}

function parseReleasedLabelAudit(
	value: unknown,
	context: string,
	n: number,
	positive: number,
	negative: number
): BeliefReleasedLabelAudit {
	const obj = record(value, context);
	exactKeys(
		obj,
		['released_label_rule', 'strict_e0_rule', 'released', 'strict_e0', 'released_negative_assumption'],
		context
	);
	const releasedObj = record(obj.released, `${context}.released`);
	exactKeys(releasedObj, ['statements', 'positive', 'negative'], `${context}.released`);
	const released = {
		statements: integer(releasedObj.statements, `${context}.released.statements`, 1),
		positive: integer(releasedObj.positive, `${context}.released.positive`, 1),
		negative: integer(releasedObj.negative, `${context}.released.negative`, 1)
	};
	if (released.statements !== n || released.positive !== positive || released.negative !== negative) {
		fail(`${context}.released`, 'counts differ from the primary released target');
	}
	const strictObj = record(obj.strict_e0, `${context}.strict_e0`);
	exactKeys(
		strictObj,
		['resolved', 'positive', 'negative', 'unresolved', 'ordered_statement_id_sha256'],
		`${context}.strict_e0`
	);
	const strictE0 = {
		resolved: integer(strictObj.resolved, `${context}.strict_e0.resolved`, 1),
		positive: integer(strictObj.positive, `${context}.strict_e0.positive`, 1),
		negative: integer(strictObj.negative, `${context}.strict_e0.negative`, 1),
		unresolved: integer(strictObj.unresolved, `${context}.strict_e0.unresolved`, 1),
		ordered_statement_id_sha256: sha256(
			strictObj.ordered_statement_id_sha256,
			`${context}.strict_e0.ordered_statement_id_sha256`
		)
	};
	if (
		strictE0.resolved + strictE0.unresolved !== n ||
		strictE0.positive + strictE0.negative !== strictE0.resolved ||
		strictE0.positive !== positive
	) {
		fail(`${context}.strict_e0`, 'strict resolved/unresolved counts do not reconcile');
	}
	const assumptionObj = record(
		obj.released_negative_assumption,
		`${context}.released_negative_assumption`
	);
	exactKeys(
		assumptionObj,
		['statements', 'share_of_released_negatives', 'ordered_statement_id_sha256'],
		`${context}.released_negative_assumption`
	);
	const assumption = {
		statements: integer(
			assumptionObj.statements,
			`${context}.released_negative_assumption.statements`,
			1
		),
		share_of_released_negatives: ranged(
			assumptionObj.share_of_released_negatives,
			`${context}.released_negative_assumption.share_of_released_negatives`,
			0,
			1
		),
		ordered_statement_id_sha256: sha256(
			assumptionObj.ordered_statement_id_sha256,
			`${context}.released_negative_assumption.ordered_statement_id_sha256`
		)
	};
	if (
		assumption.statements !== strictE0.unresolved ||
		Math.abs(assumption.share_of_released_negatives - assumption.statements / negative) > 1e-15
	) {
		fail(`${context}.released_negative_assumption`, 'cohort count/share do not reconcile');
	}
	return {
		released_label_rule: string(obj.released_label_rule, `${context}.released_label_rule`),
		strict_e0_rule: string(obj.strict_e0_rule, `${context}.strict_e0_rule`),
		released,
		strict_e0: strictE0,
		released_negative_assumption: assumption
	};
}

function parseReleasedErrorStrata(
	value: unknown,
	context: string,
	audit: BeliefReleasedLabelAudit,
	threshold: BeliefThreshold
): BeliefReleasedLabelErrorStrata | null {
	if (threshold.status === 'unavailable') {
		if (value !== null) fail(context, 'must be null when the arm has no frozen threshold');
		return null;
	}
	const obj = record(value, context);
	exactKeys(obj, ['strict_e0_resolved', 'released_negative_assumption'], context);
	const parseStratum = (raw: unknown, stratumContext: string, expectedN: number): BeliefErrorStratum => {
		const row = record(raw, stratumContext);
		exactKeys(row, ['statements', 'tp', 'fp', 'fn', 'tn', 'errors'], stratumContext);
		const parsed = {
			statements: integer(row.statements, `${stratumContext}.statements`, 1),
			tp: integer(row.tp, `${stratumContext}.tp`),
			fp: integer(row.fp, `${stratumContext}.fp`),
			fn: integer(row.fn, `${stratumContext}.fn`),
			tn: integer(row.tn, `${stratumContext}.tn`),
			errors: integer(row.errors, `${stratumContext}.errors`)
		};
		if (
			parsed.statements !== expectedN ||
			parsed.tp + parsed.fp + parsed.fn + parsed.tn !== expectedN ||
			parsed.errors !== parsed.fp + parsed.fn
		) {
			fail(stratumContext, 'stratum counts do not reconcile');
		}
		return parsed;
	};
	const resolved = parseStratum(
		obj.strict_e0_resolved,
		`${context}.strict_e0_resolved`,
		audit.strict_e0.resolved
	);
	const assumption = parseStratum(
		obj.released_negative_assumption,
		`${context}.released_negative_assumption`,
		audit.released_negative_assumption.statements
	);
	if (assumption.tp !== 0 || assumption.fn !== 0) {
		fail(`${context}.released_negative_assumption`, 'released-negative cohort must contain only negative labels');
	}
	const total = threshold.confusion;
	for (const key of ['tp', 'fp', 'fn', 'tn'] as const) {
		if (resolved[key] + assumption[key] !== total[key]) fail(context, 'strata do not sum to threshold confusion');
	}
	return { strict_e0_resolved: resolved, released_negative_assumption: assumption };
}

function requireCompleteComparisons(
	comparisons: BeliefPairedComparison[],
	arms: Array<{ arm_id: string; metrics: BeliefArmMetrics }>,
	context: string
): void {
	const expected = new Set<string>();
	for (let a = 0; a < arms.length; a++) {
		for (let b = a + 1; b < arms.length; b++) {
			for (const metric of BELIEF_PRIMARY_METRICS) expected.add(`${arms[a].arm_id}|${arms[b].arm_id}|${metric}`);
			if (arms[a].metrics.threshold.status === 'available' && arms[b].metrics.threshold.status === 'available') {
				for (const metric of BELIEF_THRESHOLD_METRICS) expected.add(`${arms[a].arm_id}|${arms[b].arm_id}|${metric}`);
			}
		}
	}
	const actual = comparisons.map((row) => `${row.a_arm_id}|${row.b_arm_id}|${row.metric}`);
	if (new Set(actual).size !== actual.length || !sameMembers(actual, [...expected])) {
		fail(context, 'must contain every arm-pair metric delta exactly once');
	}
}

function parseStrictSensitivity(
	value: unknown,
	context: string,
	provenance: BeliefArtifactProvenance,
	primaryContract: BeliefContractIdentity,
	audit: BeliefReleasedLabelAudit,
	primaryArms: BeliefArm[]
): BeliefStrictSensitivity {
	const obj = record(value, context);
	exactKeys(
		obj,
		['analysis_scope', 'selection_rule', 'contract', 'gold_path', 'n_evaluable', 'n_positive', 'n_negative', 'excluded_unresolved', 'pr_summary_contract', 'arms', 'comparisons'],
		context
	);
	if (obj.analysis_scope !== 'fixed_resolved_only_sensitivity') fail(`${context}.analysis_scope`, 'must be fixed_resolved_only_sensitivity');
	const n = integer(obj.n_evaluable, `${context}.n_evaluable`, 1);
	const positive = integer(obj.n_positive, `${context}.n_positive`, 1);
	const negative = integer(obj.n_negative, `${context}.n_negative`, 1);
	const excluded = integer(obj.excluded_unresolved, `${context}.excluded_unresolved`, 1);
	if (
		n !== audit.strict_e0.resolved ||
		positive !== audit.strict_e0.positive ||
		negative !== audit.strict_e0.negative ||
		excluded !== audit.strict_e0.unresolved ||
		positive + negative !== n
	) fail(context, 'strict sensitivity counts differ from released-label audit');
	const contract = parseContract(obj.contract, `${context}.contract`, BELIEF_STRICT_GOLD_RULE);
	if (contract.substrate_sha256 !== primaryContract.substrate_sha256) fail(`${context}.contract`, 'substrate digest differs from primary panel');
	if (contract.evaluation_set_sha256 !== audit.strict_e0.ordered_statement_id_sha256) fail(`${context}.contract`, 'strict evaluation digest differs from audit');
	const prObj = record(obj.pr_summary_contract, `${context}.pr_summary_contract`);
	exactKeys(prObj, ['fold_mean_trapezoidal_pr_auc', 'pooled_average_precision', 'fold_count'], `${context}.pr_summary_contract`);
	const foldCount = integer(prObj.fold_count, `${context}.pr_summary_contract.fold_count`, 1);
	if (foldCount !== 10) fail(`${context}.pr_summary_contract.fold_count`, 'strict sensitivity requires all ten frozen folds');
	if (!Array.isArray(obj.arms) || obj.arms.length !== primaryArms.length) fail(`${context}.arms`, 'must contain every primary arm exactly once');
	const armIds = new Set<string>();
	const arms = obj.arms.map((raw, armIndex) => {
		const armContext = `${context}.arms[${armIndex}]`;
		const armObj = record(raw, armContext);
		exactKeys(armObj, ['arm_id', 'label', 'family', 'contract', 'coverage', 'metrics'], armContext);
		const armId = identifier(armObj.arm_id, `${armContext}.arm_id`);
		if (armIds.has(armId)) fail(armContext, `duplicate arm_id ${armId}`);
		armIds.add(armId);
		const primary = primaryArms[armIndex];
		if (armId !== primary.arm_id || armObj.label !== primary.label || armObj.family !== primary.family) fail(armContext, 'identity/order differs from primary arms');
		requireContract(parseContract(armObj.contract, `${armContext}.contract`, BELIEF_STRICT_GOLD_RULE), contract, `${armContext}.contract`);
		return {
			arm_id: armId,
			label: string(armObj.label, `${armContext}.label`),
			family: armObj.family as BeliefArmFamily,
			contract,
			coverage: parseCoverage(armObj.coverage, `${armContext}.coverage`, n),
			metrics: parseMetrics(armObj.metrics, `${armContext}.metrics`, provenance.bootstrap_resamples, provenance.minimum_valid_bootstrap_fraction, provenance.calibration_bin_edges, n, foldCount)
		} satisfies BeliefSensitivityArm;
	});
	if (!Array.isArray(obj.comparisons)) fail(`${context}.comparisons`, 'expected an array');
	const comparisons = obj.comparisons.map((raw, index) => parseComparison(raw, `${context}.comparisons[${index}]`, contract, armIds, provenance.bootstrap_resamples, provenance.minimum_valid_bootstrap_fraction));
	requireCompleteComparisons(comparisons, arms, `${context}.comparisons`);
	return {
		analysis_scope: 'fixed_resolved_only_sensitivity',
		selection_rule: string(obj.selection_rule, `${context}.selection_rule`),
		contract,
		gold_path: string(obj.gold_path, `${context}.gold_path`),
		n_evaluable: n,
		n_positive: positive,
		n_negative: negative,
		excluded_unresolved: excluded,
		pr_summary_contract: {
			fold_mean_trapezoidal_pr_auc: string(prObj.fold_mean_trapezoidal_pr_auc, `${context}.pr_summary_contract.fold_mean_trapezoidal_pr_auc`),
			pooled_average_precision: string(prObj.pooled_average_precision, `${context}.pr_summary_contract.pooled_average_precision`),
			fold_count: foldCount
		},
		arms,
		comparisons
	};
}

function parsePanel(
	value: unknown,
	index: number,
	provenance: BeliefArtifactProvenance,
	expectedId: BeliefPanelId
): BeliefSubstrate {
	const context = `substrates[${index}]`;
	const obj = record(value, context);
	exactKeys(obj, ['substrate_id', 'lane', 'label', 'analysis_scope', 'released_label_audit', 'contract', 'substrate_manifest_path', 'gold_path', 'positive_class', 'n_evaluable', 'n_positive', 'n_negative', 'pr_summary_contract', 'arms', 'excluded_arms', 'comparisons', 'pareto', 'strict_e0_resolved_sensitivity'], context);
	const substrateId = identifier(obj.substrate_id, `${context}.substrate_id`);
	if (substrateId !== expectedId) fail(`${context}.substrate_id`, `expected canonical panel ${expectedId}`);
	if (obj.lane !== 'paper') fail(`${context}.lane`, 'canonical comparison only accepts the paper lane');
	if (obj.analysis_scope !== 'primary') fail(`${context}.analysis_scope`, 'canonical comparison only accepts primary scope');
	if (obj.positive_class !== BELIEF_POSITIVE_CLASS) fail(`${context}.positive_class`, `must be ${BELIEF_POSITIVE_CLASS}`);
	const n = integer(obj.n_evaluable, `${context}.n_evaluable`, 1);
	const positive = integer(obj.n_positive, `${context}.n_positive`, 1);
	const negative = integer(obj.n_negative, `${context}.n_negative`, 1);
	if (positive + negative !== n) fail(context, 'class counts do not sum to n_evaluable');
	const releasedLabelAudit = parseReleasedLabelAudit(
		obj.released_label_audit,
		`${context}.released_label_audit`,
		n,
		positive,
		negative
	);
	const contract = parseContract(obj.contract, `${context}.contract`);
	const prObj = record(obj.pr_summary_contract, `${context}.pr_summary_contract`);
	exactKeys(prObj, ['fold_mean_trapezoidal_pr_auc', 'pooled_average_precision', 'fold_count'], `${context}.pr_summary_contract`);
	const foldCount = integer(prObj.fold_count, `${context}.pr_summary_contract.fold_count`, 1);
	if (foldCount !== 10) fail(`${context}.pr_summary_contract.fold_count`, 'canonical paper panels require the frozen 10-fold protocol');
	if (!Array.isArray(obj.arms) || obj.arms.length === 0) fail(`${context}.arms`, 'expected a non-empty array');
	const armIds = new Set<string>();
	const arms = obj.arms.map((raw, armIndex) => {
		const armContext = `${context}.arms[${armIndex}]`;
		const armObj = record(raw, armContext);
		exactKeys(armObj, ['arm_id', 'label', 'family', 'contract', 'coverage', 'metrics', 'released_label_error_strata', 'cost', 'pareto', 'provenance'], armContext);
		const armId = identifier(armObj.arm_id, `${armContext}.arm_id`);
		if (armIds.has(armId)) fail(armContext, `duplicate arm_id ${armId}`);
		armIds.add(armId);
		if (armObj.family !== 'paper' && armObj.family !== 'current' && armObj.family !== 'llm') fail(`${armContext}.family`, 'expected paper, current, or llm');
		requireContract(parseContract(armObj.contract, `${armContext}.contract`), contract, `${armContext}.contract`);
		const cost = parseCost(armObj.cost, `${armContext}.cost`, n);
		const metrics = parseMetrics(armObj.metrics, `${armContext}.metrics`, provenance.bootstrap_resamples, provenance.minimum_valid_bootstrap_fraction, provenance.calibration_bin_edges, n, foldCount);
		return {
			arm_id: armId,
			label: string(armObj.label, `${armContext}.label`),
			family: armObj.family,
			contract,
			coverage: parseCoverage(armObj.coverage, `${armContext}.coverage`, n),
			metrics,
			released_label_error_strata: parseReleasedErrorStrata(
				armObj.released_label_error_strata,
				`${armContext}.released_label_error_strata`,
				releasedLabelAudit,
				metrics.threshold
			),
			cost,
			pareto: parseArmPareto(armObj.pareto, `${armContext}.pareto`, cost),
			provenance: parseArmProvenance(armObj.provenance, `${armContext}.provenance`)
		} satisfies BeliefArm;
	});
	for (const family of ['paper', 'current', 'llm'] as const) if (!arms.some((arm) => arm.family === family)) fail(`${context}.arms`, `missing required ${family} family`);
	const excludedArms = parseExcludedArms(obj.excluded_arms, `${context}.excluded_arms`, armIds);
	if (!Array.isArray(obj.comparisons)) fail(`${context}.comparisons`, 'expected an array');
	const comparisons = obj.comparisons.map((raw, comparisonIndex) => parseComparison(raw, `${context}.comparisons[${comparisonIndex}]`, contract, armIds, provenance.bootstrap_resamples, provenance.minimum_valid_bootstrap_fraction));
	requireCompleteComparisons(comparisons, arms, `${context}.comparisons`);
	const pareto = parsePareto(obj.pareto, `${context}.pareto`, arms, comparisons);
	const strictSensitivity = parseStrictSensitivity(
		obj.strict_e0_resolved_sensitivity,
		`${context}.strict_e0_resolved_sensitivity`,
		provenance,
		contract,
		releasedLabelAudit,
		arms
	);
	return {
		substrate_id: substrateId as BeliefPanelId,
		lane: 'paper',
		label: string(obj.label, `${context}.label`),
		analysis_scope: 'primary',
		released_label_audit: releasedLabelAudit,
		contract,
		substrate_manifest_path: string(obj.substrate_manifest_path, `${context}.substrate_manifest_path`),
		gold_path: string(obj.gold_path, `${context}.gold_path`),
		positive_class: BELIEF_POSITIVE_CLASS,
		n_evaluable: n,
		n_positive: positive,
		n_negative: negative,
		pr_summary_contract: {
			fold_mean_trapezoidal_pr_auc: string(prObj.fold_mean_trapezoidal_pr_auc, `${context}.pr_summary_contract.fold_mean_trapezoidal_pr_auc`),
			pooled_average_precision: string(prObj.pooled_average_precision, `${context}.pr_summary_contract.pooled_average_precision`),
			fold_count: foldCount
		},
		arms,
		excluded_arms: excludedArms,
		comparisons,
		pareto,
		strict_e0_resolved_sensitivity: strictSensitivity
	};
}
function parseProvenance(value: unknown, context: string): BeliefArtifactProvenance {
	const obj = record(value, context);
	exactKeys(obj, ['metrics_code_sha256', 'source_manifest_sha256', 'source_manifest_path', 'scorer_registry', 'bootstrap_seed', 'bootstrap_resamples', 'bootstrap_rng', 'ci_level', 'log_loss_epsilon', 'calibration_bin_edges', 'minimum_valid_bootstrap_fraction', 'evaluation_set_digest_method', 'runtime'], context);
	if (obj.ci_level !== 0.95) fail(`${context}.ci_level`, 'canonical comparison requires 0.95');
	if (!Array.isArray(obj.calibration_bin_edges) || obj.calibration_bin_edges.length < 3) fail(`${context}.calibration_bin_edges`, 'expected at least three edges');
	const edges = obj.calibration_bin_edges.map((edge, index) => ranged(edge, `${context}.calibration_bin_edges[${index}]`, 0, 1));
	if (edges[0] !== 0 || edges.at(-1) !== 1 || edges.some((edge, index) => index > 0 && edge <= edges[index - 1])) fail(`${context}.calibration_bin_edges`, 'must increase strictly from 0 to 1');
	const runtimeObj = record(obj.runtime, `${context}.runtime`);
	exactKeys(runtimeObj, ['python', 'numpy', 'scikit_learn'], `${context}.runtime`);
	const scorerRegistry = record(obj.scorer_registry, `${context}.scorer_registry`);
	exactKeys(scorerRegistry, ['path', 'sha256', 'bytes'], `${context}.scorer_registry`);
	return {
		metrics_code_sha256: sha256(obj.metrics_code_sha256, `${context}.metrics_code_sha256`),
		source_manifest_sha256: sha256(obj.source_manifest_sha256, `${context}.source_manifest_sha256`),
		source_manifest_path: string(obj.source_manifest_path, `${context}.source_manifest_path`),
		scorer_registry: {
			path: string(scorerRegistry.path, `${context}.scorer_registry.path`),
			sha256: sha256(scorerRegistry.sha256, `${context}.scorer_registry.sha256`),
			bytes: integer(scorerRegistry.bytes, `${context}.scorer_registry.bytes`, 1)
		},
		bootstrap_seed: integer(obj.bootstrap_seed, `${context}.bootstrap_seed`),
		bootstrap_resamples: integer(obj.bootstrap_resamples, `${context}.bootstrap_resamples`, 1),
		bootstrap_rng: string(obj.bootstrap_rng, `${context}.bootstrap_rng`),
		ci_level: 0.95,
		log_loss_epsilon: ranged(obj.log_loss_epsilon, `${context}.log_loss_epsilon`, Number.MIN_VALUE, 0.499999999999),
		calibration_bin_edges: edges,
		minimum_valid_bootstrap_fraction: ranged(obj.minimum_valid_bootstrap_fraction, `${context}.minimum_valid_bootstrap_fraction`, Number.MIN_VALUE, 1),
		evaluation_set_digest_method: string(obj.evaluation_set_digest_method, `${context}.evaluation_set_digest_method`),
		runtime: {
			python: string(runtimeObj.python, `${context}.runtime.python`),
			numpy: string(runtimeObj.numpy, `${context}.runtime.numpy`),
			scikit_learn: string(runtimeObj.scikit_learn, `${context}.runtime.scikit_learn`)
		}
	};
}

export function validateBeliefComparisonArtifact(raw: unknown): BeliefArtifactValidation {
	let frozenAt: string | null = null;
	let provenance: BeliefArtifactProvenance | null = null;
	try {
		const obj = record(raw, 'artifact');
		exactKeys(obj, ['artifact_kind', 'frozen_at', 'provenance', 'substrates'], 'artifact');
		if (obj.artifact_kind !== BELIEF_COMPARISON_KIND) fail('artifact.artifact_kind', `expected ${BELIEF_COMPARISON_KIND}`);
		frozenAt = isoTimestamp(obj.frozen_at, 'artifact.frozen_at');
		provenance = parseProvenance(obj.provenance, 'artifact.provenance');
		if (!Array.isArray(obj.substrates) || obj.substrates.length !== BELIEF_PANEL_IDS.length) {
			fail('artifact.substrates', `expected exactly the canonical panels ${BELIEF_PANEL_IDS.join(', ')}`);
		}
		const panels: [BeliefSubstrate, BeliefSubstrate] = [
			parsePanel(obj.substrates[0], 0, provenance, BELIEF_PANEL_IDS[0]),
			parsePanel(obj.substrates[1], 1, provenance, BELIEF_PANEL_IDS[1])
		];
		return {
			status: 'available',
			frozen_at: frozenAt,
			provenance,
			panels,
			reasons: []
		};
	} catch (error) {
		return { status: 'unavailable', frozen_at: frozenAt, provenance, panels: [], reasons: [error instanceof Error ? error.message : String(error)] };
	}
}
