/** Pure-function assertions for the canonical statement-belief artifact gate. */
import {
	BELIEF_COMPARISON_KIND,
	BELIEF_GOLD_RULE,
	BELIEF_STRICT_GOLD_RULE,
	BELIEF_PANEL_IDS,
	BELIEF_POSITIVE_CLASS,
	BELIEF_PREDICTION_UNIT,
	BELIEF_PRIMARY_METRICS,
	BELIEF_THRESHOLD_METRICS,
	validateBeliefComparisonArtifact
} from '../src/lib/data/belief-comparison.ts';

let failures = 0;
const RESAMPLES = 10_000;
const MINIMUM_VALID_FRACTION = 0.99;
const CALIBRATION_EDGES = Array.from({ length: 11 }, (_, index) => index / 10);

function eq(got, want, label) {
	if (got !== want) {
		failures++;
		console.error(`FAIL ${label}: got ${JSON.stringify(got)}, want ${JSON.stringify(want)}`);
	}
}

function ok(got, label) {
	if (!got) {
		failures++;
		console.error(`FAIL ${label}`);
	}
}

const SHA = {
	allSourceSubstrate: 'a'.repeat(64),
	allSourceGold: 'b'.repeat(64),
	allSourceEvaluation: 'c'.repeat(64),
	readersSubstrate: 'd'.repeat(64),
	readersGold: 'e'.repeat(64),
	readersEvaluation: 'f'.repeat(64),
	prediction: '1'.repeat(64),
	training: '2'.repeat(64),
	metrics: '3'.repeat(64),
	manifest: '4'.repeat(64),
	cost: '5'.repeat(64),
	threshold: '6'.repeat(64)
};
SHA.allSourceStrictGold = '7'.repeat(64);
SHA.allSourceStrictEvaluation = '8'.repeat(64);
SHA.readersStrictGold = '9'.repeat(64);
SHA.readersStrictEvaluation = '0'.repeat(64);


function contract(panelId = BELIEF_PANEL_IDS[0], strict = false) {
	const allSource = panelId === BELIEF_PANEL_IDS[0];
	return {
		prediction_unit: BELIEF_PREDICTION_UNIT,
		gold_rule: strict ? BELIEF_STRICT_GOLD_RULE : BELIEF_GOLD_RULE,
		substrate_sha256: allSource ? SHA.allSourceSubstrate : SHA.readersSubstrate,
		gold_sha256: strict
			? allSource ? SHA.allSourceStrictGold : SHA.readersStrictGold
			: allSource ? SHA.allSourceGold : SHA.readersGold,
		evaluation_set_sha256: strict
			? allSource ? SHA.allSourceStrictEvaluation : SHA.readersStrictEvaluation
			: allSource ? SHA.allSourceEvaluation : SHA.readersEvaluation
	};
}

function estimate(
	value,
	width = 0.02,
	method = 'paired statement bootstrap stratified independently within each frozen fold'
) {
	return {
		estimate: value,
		ci95: [value - width, value + width],
		method,
		resamples: RESAMPLES,
		valid_resamples: RESAMPLES
	};
}

function foldedEstimate(value, n = 100, positiveTotal = n * 0.6) {
	const offsets = Array.from({ length: 10 }, (_, index) => (index - 4.5) * 0.002);
	const foldN = n / 10;
	const positiveBase = Math.floor(positiveTotal / 10);
	const positiveRemainder = positiveTotal - positiveBase * 10;
	return {
		...estimate(value),
		fold_estimates: offsets.map((offset, foldId) => ({
			fold_id: foldId,
			n: foldN,
			positive: positiveBase + (foldId < positiveRemainder ? 1 : 0),
			negative: foldN - positiveBase - (foldId < positiveRemainder ? 1 : 0),
			estimate: value + offset
		})),
		fold_population_sd: Math.sqrt(
			offsets.reduce((sum, offset) => sum + offset ** 2, 0) / offsets.length
		)
	};
}

const BASE = { paper: 0.72, current: 0.8, llm: 0.84 };

function metricPoint(family, metric) {
	const base = BASE[family];
	if (metric === 'fold_mean_trapezoidal_pr_auc') return base - 0.01;
	if (metric === 'pooled_average_precision') return base;
	if (metric === 'auroc') return base + 0.02;
	if (metric === 'brier') return 1 - base;
	if (metric === 'log_loss') return 1.1 - base;
	if (metric === 'calibration_ece') return 0.1 - (base - 0.7) * 0.2;
	if (metric === 'calibration_intercept') return { paper: 0.1, current: 0.04, llm: 0.02 }[family];
	if (metric === 'calibration_slope') return { paper: 0.85, current: 0.94, llm: 0.98 }[family];
	if (metric === 'calibration_intercept_abs_error') return Math.abs(metricPoint(family, 'calibration_intercept'));
	if (metric === 'calibration_slope_abs_error') return Math.abs(metricPoint(family, 'calibration_slope') - 1);
	if (metric === 'threshold_accuracy') return base - 0.02;
	if (metric === 'threshold_precision') return base - 0.01;
	if (metric === 'threshold_recall') return base - 0.03;
	if (metric === 'threshold_f1') return base - 0.02;
	throw new Error(`unknown metric ${metric}`);
}

function betterWhen(metric) {
	if (['brier', 'log_loss', 'calibration_ece', 'calibration_intercept_abs_error', 'calibration_slope_abs_error'].includes(metric)) return 'lower';
	return 'higher';
}

function threshold(family, n) {
	return {
		status: 'available',
		value: 0.5,
		operator: 'greater_than_or_equal',
		source_path: `threshold-${family}.json`,
		source_sha256: SHA.threshold,
		frozen_at: '2026-07-01T00:00:00Z',
		reason: null,
		confusion: { tp: n * 0.5, fp: n * 0.1, fn: n * 0.1, tn: n * 0.3 },
		metrics: {
			accuracy: estimate(metricPoint(family, 'threshold_accuracy')),
			precision: estimate(metricPoint(family, 'threshold_precision')),
			recall: estimate(metricPoint(family, 'threshold_recall')),
			f1: estimate(metricPoint(family, 'threshold_f1'))
		}
	};
}

function arm(id, family, panelId = BELIEF_PANEL_IDS[0], n = 100, index = 0, strict = false, positiveTotal = n * 0.6) {
	const total = [0.01, 0.02, 0.1][index];
	const measured = family === 'llm' ? total * 0.8 : total;
	const reserved = family === 'llm' ? total * 0.2 : 0;
	const lower = measured;
	const costComparabilityId = 'fixture_provider_token_cost';
	const priceSource = 'https://example.test/pricing';
	const priceDate = '2026-07-21';
	return {
		arm_id: id,
		label: `${family} ${id}`,
		family,
		contract: contract(panelId, strict),
		coverage: { eligible: n, predicted: n, invalid: 0, fraction: 1 },
		metrics: {
			fold_mean_trapezoidal_pr_auc: foldedEstimate(
				metricPoint(family, 'fold_mean_trapezoidal_pr_auc'),
				n,
				positiveTotal
			),
			pooled_average_precision: estimate(metricPoint(family, 'pooled_average_precision')),
			auroc: estimate(metricPoint(family, 'auroc')),
			brier: estimate(metricPoint(family, 'brier'), 0.01),
			log_loss: estimate(metricPoint(family, 'log_loss'), 0.03),
			calibration: {
				ece: estimate(metricPoint(family, 'calibration_ece'), 0.01),
				intercept: estimate(metricPoint(family, 'calibration_intercept'), 0.1),
				slope: estimate(metricPoint(family, 'calibration_slope'), 0.1),
				intercept_abs_error: estimate(metricPoint(family, 'calibration_intercept_abs_error'), 0.01),
				slope_abs_error: estimate(metricPoint(family, 'calibration_slope_abs_error'), 0.01),
				reliability_bins: CALIBRATION_EDGES.slice(0, -1).map((lower, binIndex) => ({
					bin_index: binIndex,
					lower,
					upper: CALIBRATION_EDGES[binIndex + 1],
					upper_inclusive: binIndex === CALIBRATION_EDGES.length - 2,
					n: n / (CALIBRATION_EDGES.length - 1),
					mean_prediction: lower + 0.05,
					observed_fraction: lower + 0.04
				}))
			},
			threshold: threshold(family, n)
		},
		released_label_error_strata: strict ? undefined : {
			strict_e0_resolved: { statements: n - 20, tp: n * 0.5, fp: n * 0.08, fn: n * 0.1, tn: n * 0.12, errors: n * 0.18 },
			released_negative_assumption: { statements: 20, tp: 0, fp: n * 0.02, fn: 0, tn: 18, errors: n * 0.02 }
		},
		cost: {
			status: 'available',
			record_type: 'evidence_execution',
			inference_usd_total: total,
			inference_usd_total_exact: total.toFixed(2),
			usd_per_1k_statements: (total * 1000) / n,
			provider_measured_usd_total: measured,
			provider_measured_usd_total_exact: measured.toFixed(2),
			conservative_reserved_usd_total: reserved,
			conservative_reserved_usd_total_exact: reserved.toFixed(2),
			inference_usd_lower: lower,
			inference_usd_lower_exact: lower.toFixed(2),
			inference_usd_upper: total,
			inference_usd_upper_exact: total.toFixed(2),
			usd_per_1k_statements_lower: (lower * 1000) / n,
			usd_per_1k_statements_upper: (total * 1000) / n,
			basis: family === 'llm' ? 'mixed_conservative_upper_bound' : 'provider_measured_observed',
			view_id: 'provider-runtime-retry-inclusive',
			includes_retries: true,
			includes_relation_subcalls: true,
			denominator: { statements: n, evidence_executions: family === 'llm' ? n + 5 : n },
			scope: {
				included_cost_categories: ['provider_inference_calls'],
				excluded_cost_categories: ['training', 'local_aggregation', 'feature_materialization', 'upstream_reading']
			},
			execution_count: family === 'llm' ? n + 5 : n,
			attempt_count: family === 'llm' ? n + 6 : n,
			retry_attempt_count: family === 'llm' ? 1 : 0,
			successful_attempt_count: family === 'llm' ? n + 5 : n,
			error_attempt_count: family === 'llm' ? 1 : 0,
			provider_measured_call_count: family === 'llm' ? n + 5 : n,
			conservative_call_count: family === 'llm' ? 1 : 0,
			input_tokens: family === 'llm' ? null : n * 100,
			output_tokens: family === 'llm' ? null : n * 20,
			token_accounting_complete: family !== 'llm',
			ledger_path: `${id}.cost.jsonl`,
			ledger_sha256: SHA.cost,
			price_source: priceSource,
			price_date: priceDate,
			cost_comparability_id: costComparabilityId,
			pricing: {
				cost_comparability_id: costComparabilityId,
				currency: 'USD',
				provider: 'Fixture Provider',
				provider_model_id: `fixture.${family}`,
				pricing_mode: 'on_demand',
				region: 'fixture-region',
				resolved_service_tier: 'standard',
				retrieved_on: priceDate,
				service_tier_request: 'default',
				source_url: priceSource,
				tariff: {
					input_usd_per_million: ['0.04', '0.13', '1'][index],
					output_usd_per_million: ['0.08', '0.4', '3.2'][index],
					pricing_basis: 'list'
				},
				unit: 'per_million_tokens'
			},
			projection: panelId === BELIEF_PANEL_IDS[0] ? 'all_executions' : 'observed_execution_subset',
			counterfactual_run_cost: false,
			shared_run_id: `fixture_${family}_run`,
			additive_across_panels: false,
			reason: null
		},
		pareto: {
			status: 'available',
			view_id: 'provider-runtime-retry-inclusive',
			basis: family === 'llm' ? 'mixed_conservative_upper_bound' : 'provider_measured_observed',
			point_pareto: true,
			uncertainty_pareto: true,
			reason: null
		},
		provenance: {
			implementation: `implementation:${id}`,
			implementation_digest: `commit:${id}`,
			predictions_path: `${id}.predictions.jsonl`,
			predictions_sha256: SHA.prediction,
			training_data_sha256: family === 'paper' ? null : SHA.training,
			environment: 'test environment',
			notes: null
		}
	};
}

function comparisonRows(arms, panelId, strict = false) {
	const rows = [];
	for (let a = 0; a < arms.length; a++) {
		for (let b = a + 1; b < arms.length; b++) {
			for (const metric of [...BELIEF_PRIMARY_METRICS, ...BELIEF_THRESHOLD_METRICS]) {
				const delta = metricPoint(arms[b].family, metric) - metricPoint(arms[a].family, metric);
				rows.push({
					a_arm_id: arms[a].arm_id,
					b_arm_id: arms[b].arm_id,
					metric,
					direction: 'b_minus_a',
					better_when: betterWhen(metric),
					contract: contract(panelId, strict),
					delta: estimate(delta, 0.03),
					resamples: RESAMPLES,
					method: 'paired statement bootstrap percentile CI'
				});
			}
		}
	}
	return rows;
}

function pareto(arms) {
	const audit = [];
	for (const candidate of arms) {
		for (const challenger of arms) {
			if (candidate === challenger) continue;
			const performanceDelta =
				challenger.metrics.fold_mean_trapezoidal_pr_auc.estimate -
				candidate.metrics.fold_mean_trapezoidal_pr_auc.estimate;
			audit.push({
				candidate_arm_id: candidate.arm_id,
				challenger_arm_id: challenger.arm_id,
				candidate_cost_per_1k_interval: [
					candidate.cost.usd_per_1k_statements_lower,
					candidate.cost.usd_per_1k_statements_upper
				],
				challenger_cost_per_1k_interval: [
					challenger.cost.usd_per_1k_statements_lower,
					challenger.cost.usd_per_1k_statements_upper
				],
				challenger_minus_candidate_cost_per_1k:
					challenger.cost.usd_per_1k_statements_upper - candidate.cost.usd_per_1k_statements_upper,
				cost_interval_definitely_not_worse:
					challenger.cost.usd_per_1k_statements_upper <= candidate.cost.usd_per_1k_statements_lower,
				challenger_minus_candidate_performance: performanceDelta,
				performance_delta_ci95: [performanceDelta - 0.03, performanceDelta + 0.03],
				point_dominates: false,
				uncertainty_dominates: false
			});
		}
	}
	return {
		objective_metric: 'fold_mean_trapezoidal_pr_auc',
		performance_direction: 'higher_is_better',
		cost_axis: 'usd_per_1k_statements_upper',
		point_rule: 'weakly no worse in both axes and strictly better in one',
		uncertainty_rule: 'paired lower CI must be non-negative at no greater cost',
		views: [
			{
				view_id: 'provider-runtime-retry-inclusive',
				basis: 'mixed',
				eligible_arm_ids: arms.map((candidate) => candidate.arm_id),
				point_frontier_arm_ids: arms.map((candidate) => candidate.arm_id),
				uncertainty_frontier_arm_ids: arms.map((candidate) => candidate.arm_id),
				audit
			}
		]
	};
}

function panel(panelId = BELIEF_PANEL_IDS[0], n = 100) {
	const allSource = panelId === BELIEF_PANEL_IDS[0];
	const suffix = allSource ? 'all' : 'readers';
	const arms = [
		arm(`paper-${suffix}`, 'paper', panelId, n, 0),
		arm(`current-${suffix}`, 'current', panelId, n, 1),
		arm(`llm-${suffix}`, 'llm', panelId, n, 2)
	];
	const strictArms = [
		arm(`paper-${suffix}`, 'paper', panelId, 80, 0, true, 60),
		arm(`current-${suffix}`, 'current', panelId, 80, 1, true, 60),
		arm(`llm-${suffix}`, 'llm', panelId, 80, 2, true, 60)
	].map((candidate) => {
		delete candidate.released_label_error_strata;
		delete candidate.cost;
		delete candidate.pareto;
		delete candidate.provenance;
		return candidate;
	});
	const strictContract = contract(panelId, true);
	return {
		substrate_id: panelId,
		lane: 'paper',
		label: allSource ? 'Paper all-source panel' : 'Paper five-reader panel',
		analysis_scope: 'primary',
		released_label_audit: {
			released_label_rule: 'positive if any reviewed evidence is positive; released negative otherwise',
			strict_e0_rule: 'negative only if every exact evidence pair is reviewed negative',
			released: { statements: n, positive: 60, negative: 40 },
			strict_e0: {
				resolved: 80,
				positive: 60,
				negative: 20,
				unresolved: 20,
				ordered_statement_id_sha256: strictContract.evaluation_set_sha256
			},
			released_negative_assumption: {
				statements: 20,
				share_of_released_negatives: 0.5,
				ordered_statement_id_sha256: '6'.repeat(64)
			}
		},
		contract: contract(panelId),
		substrate_manifest_path: `${panelId}.substrate.manifest.json`,
		gold_path: `${panelId}.gold.jsonl`,
		positive_class: BELIEF_POSITIVE_CLASS,
		n_evaluable: n,
		n_positive: n * 0.6,
		n_negative: n * 0.4,
		pr_summary_contract: {
			fold_mean_trapezoidal_pr_auc: 'mean of per-fold precision_recall_curve + auc',
			pooled_average_precision: 'pooled average_precision_score',
			fold_count: 10
		},
		arms,
		excluded_arms: [
			{
				arm_id: `counts-scorer-${suffix}`,
				label: 'INDRA CountsScorer',
				family: 'current',
				status: 'excluded',
				reason: 'A frozen out-of-fold prediction artifact has not yet been materialized.',
					required_artifact: `${panelId}.counts-scorer.predictions.jsonl`,
				provenance: 'indra.belief.skl.SklearnScorer registry entry'
			}
		],
		comparisons: comparisonRows(arms, panelId),
		pareto: pareto(arms),
		strict_e0_resolved_sensitivity: {
			analysis_scope: 'fixed_resolved_only_sensitivity',
			selection_rule: 'exclude the exact strict-unresolved released-negative cohort',
			contract: strictContract,
			gold_path: `${panelId}.strict.gold.jsonl`,
			n_evaluable: 80,
			n_positive: 60,
			n_negative: 20,
			excluded_unresolved: 20,
			pr_summary_contract: {
				fold_mean_trapezoidal_pr_auc: 'mean of per-fold precision_recall_curve + auc',
				pooled_average_precision: 'pooled average_precision_score',
				fold_count: 10
			},
			arms: strictArms,
			comparisons: comparisonRows(strictArms, panelId, true)
		}
	};
}

function artifact() {
	return {
		artifact_kind: BELIEF_COMPARISON_KIND,
		frozen_at: '2026-07-17T00:00:00Z',
		provenance: {
			metrics_code_sha256: SHA.metrics,
			source_manifest_sha256: SHA.manifest,
			source_manifest_path: 'fixture.spec.json',
			scorer_registry: {
				path: 'fixture.scorers.json',
				sha256: SHA.manifest,
				bytes: 128
			},
			bootstrap_seed: 20260717,
			bootstrap_resamples: RESAMPLES,
			bootstrap_rng: 'numpy.random.Generator(PCG64)',
			ci_level: 0.95,
			log_loss_epsilon: 1e-6,
			calibration_bin_edges: CALIBRATION_EDGES,
			minimum_valid_bootstrap_fraction: MINIMUM_VALID_FRACTION,
			evaluation_set_digest_method: 'canonical ordered statement ID JSONL',
			runtime: { python: '3.14', numpy: '2.4', scikit_learn: '1.9' }
		},
		substrates: BELIEF_PANEL_IDS.map((panelId) => panel(panelId, 100))
	};
}

const valid = validateBeliefComparisonArtifact(artifact());
eq(valid.status, 'available', 'canonical paper artifact passes');
eq(valid.panels.length, 2, 'both canonical paper panels retained');
eq(valid.panels[0].substrate_id, BELIEF_PANEL_IDS[0], 'all-source panel ready');
eq(valid.panels[1].substrate_id, BELIEF_PANEL_IDS[1], 'reader panel ready');
function rejectsArtifact(candidate, label, reasonFragment) {
	const result = validateBeliefComparisonArtifact(candidate);
	eq(result.status, 'unavailable', `${label}: complete artifact gates`);
	eq(result.panels.length, 0, `${label}: no one-panel artifact is exposed`);
	ok(
		result.reasons.some((reason) => reason.includes(reasonFragment)),
		`${label}: gate reports ${reasonFragment}`
	);
}

for (const [field, badValue, label] of [
	['prediction_unit', 'statement_evidence_pair', 'mixed prediction unit'],
	['gold_rule', 'exact_pair', 'mixed gold rule'],
	['substrate_sha256', '9'.repeat(64), 'mixed substrate digest'],
	['gold_sha256', '8'.repeat(64), 'mixed gold digest'],
	['evaluation_set_sha256', '7'.repeat(64), 'mixed evaluation digest']
]) {
	const bad = artifact();
	bad.substrates[0].arms[2].contract[field] = badValue;
	rejectsArtifact(
		bad,
		label,
		field === 'prediction_unit'
			? `must be ${BELIEF_PREDICTION_UNIT}`
			: field === 'gold_rule'
				? `must be ${BELIEF_GOLD_RULE}`
				: 'contract identity does not match panel'
	);
}

const incomplete = artifact();
incomplete.substrates[1].arms[2].coverage.predicted = 99;
incomplete.substrates[1].arms[2].coverage.fraction = 0.99;
rejectsArtifact(incomplete, 'incomplete coverage', 'formal panel requires predicted');

const retriesOmitted = artifact();
retriesOmitted.substrates[0].arms[2].cost.includes_retries = false;
rejectsArtifact(retriesOmitted, 'cost excluding retries', 'available cost must include retries');

const malformedExactCost = artifact();
malformedExactCost.substrates[0].arms[2].cost.inference_usd_total_exact = '1e-1';
rejectsArtifact(malformedExactCost, 'exponent-form exact cost', 'fixed-point decimal string');

const mismatchedExactCost = artifact();
mismatchedExactCost.substrates[0].arms[2].cost.inference_usd_total_exact = '0.11';
rejectsArtifact(mismatchedExactCost, 'numeric/exact cost mismatch', 'numeric and exact-decimal costs disagree');

const badEvidenceRetry = artifact();
badEvidenceRetry.substrates[0].arms[2].cost.retry_attempt_count = 2;
rejectsArtifact(badEvidenceRetry, 'evidence retry reconciliation', 'execution identities plus per-identity retries');

const badEvidenceStatus = artifact();
badEvidenceStatus.substrates[0].arms[2].cost.successful_attempt_count -= 1;
rejectsArtifact(badEvidenceStatus, 'evidence status reconciliation', 'attempt status counts');

const missingEvidenceBasis = artifact();
missingEvidenceBasis.substrates[0].arms[2].cost.conservative_call_count = null;
rejectsArtifact(missingEvidenceBasis, 'evidence accounting basis', 'expected a finite number');

const falseCompleteEvidenceTokens = artifact();
falseCompleteEvidenceTokens.substrates[0].arms[2].cost.input_tokens = 100;
falseCompleteEvidenceTokens.substrates[0].arms[2].cost.output_tokens = 10;
falseCompleteEvidenceTokens.substrates[0].arms[2].cost.token_accounting_complete = true;
rejectsArtifact(falseCompleteEvidenceTokens, 'conservative token completeness', 'disagrees with conservative calls');

const wrongCostRecord = artifact();
wrongCostRecord.substrates[0].arms[0].cost.record_type = 'statement_attempt';
rejectsArtifact(wrongCostRecord, 'wrong cost record type', 'expected evidence_execution');

const pricingComparabilityMismatch = artifact();
pricingComparabilityMismatch.substrates[0].arms[2].cost.pricing.cost_comparability_id = 'different_cost_basis';
rejectsArtifact(
	pricingComparabilityMismatch,
	'pricing comparability mismatch',
	'top-level and pricing cost comparability IDs disagree'
);

const crossArmCostMismatch = artifact();
crossArmCostMismatch.substrates[0].arms[2].cost.cost_comparability_id = 'different_cost_basis';
crossArmCostMismatch.substrates[0].arms[2].cost.pricing.cost_comparability_id = 'different_cost_basis';
rejectsArtifact(
	crossArmCostMismatch,
	'Pareto cost comparability mismatch',
	'eligible arm costs do not share one cost comparability ID'
);

const malformedPricingShape = artifact();
malformedPricingShape.substrates[0].arms[2].cost.pricing.tariff.per_request = '0';
rejectsArtifact(malformedPricingShape, 'unexpected tariff field', 'unexpected fields per_request');

const mismatchedPricingSource = artifact();
mismatchedPricingSource.substrates[0].arms[2].cost.price_source = 'https://example.test/other-pricing';
rejectsArtifact(
	mismatchedPricingSource,
	'pricing source mismatch',
	'price source/date disagree with structured pricing provenance'
);

const counterfactualCost = artifact();
counterfactualCost.substrates[0].arms[2].cost.counterfactual_run_cost = true;
rejectsArtifact(counterfactualCost, 'counterfactual cost', 'must be false');

const additivePanelCost = artifact();
additivePanelCost.substrates[0].arms[2].cost.additive_across_panels = true;
rejectsArtifact(additivePanelCost, 'additive panel cost', 'must be false');

const nonCanonicalClaim = artifact();
nonCanonicalClaim.substrates[0].confirmatory_claim = { claim_id: 'old-branch' };
rejectsArtifact(nonCanonicalClaim, 'removed claim branch', 'unexpected fields');

const exclusionCollision = artifact();
exclusionCollision.substrates[0].excluded_arms[0].arm_id = 'current-all';
rejectsArtifact(exclusionCollision, 'evaluated/excluded arm collision', 'excluded arm is also evaluated');

const ambiguousPr = artifact();
ambiguousPr.substrates[0].arms[0].metrics.auprc = estimate(0.8);
rejectsArtifact(ambiguousPr, 'ambiguous auprc field', 'unexpected fields');

const badReliability = artifact();
badReliability.substrates[0].arms[0].metrics.calibration.reliability_bins[0].n = 49;
rejectsArtifact(badReliability, 'reliability counts', 'bin counts do not sum');

const missingFamily = artifact();
missingFamily.substrates[0].arms = missingFamily.substrates[0].arms.filter((candidate) => candidate.family !== 'llm');
rejectsArtifact(missingFamily, 'missing model family', 'missing required llm family');

const missingDelta = artifact();
missingDelta.substrates[0].comparisons.pop();
rejectsArtifact(missingDelta, 'incomplete paired-delta matrix', 'every arm-pair metric delta');

const mismatchedDelta = artifact();
mismatchedDelta.substrates[0].comparisons[0].contract.gold_sha256 = '8'.repeat(64);
rejectsArtifact(mismatchedDelta, 'paired delta contract', 'contract identity does not match panel');

const badPareto = artifact();
badPareto.substrates[0].pareto.views[0].eligible_arm_ids.pop();
rejectsArtifact(badPareto, 'Pareto eligibility', 'eligible arms do not match cost view');

const onePanel = artifact();
onePanel.substrates.pop();
rejectsArtifact(onePanel, 'one-panel compatibility', 'expected exactly the canonical panels');

const reversedPanels = artifact();
reversedPanels.substrates.reverse();
rejectsArtifact(reversedPanels, 'reordered panels', `expected canonical panel ${BELIEF_PANEL_IDS[0]}`);

const representativeLane = artifact();
representativeLane.substrates[0].lane = 'representative';
rejectsArtifact(representativeLane, 'representative lane', 'only accepts the paper lane');

const partialGold = artifact();
partialGold.substrates[0].released_label_audit.strict_e0.unresolved = 1;
rejectsArtifact(partialGold, 'inconsistent strict audit', 'strict resolved/unresolved counts');

const duplicatedSensitivityCost = artifact();
duplicatedSensitivityCost.substrates[0].strict_e0_resolved_sensitivity.arms[0].cost = { status: 'unavailable' };
rejectsArtifact(duplicatedSensitivityCost, 'sensitivity cost duplication', 'unexpected fields cost');

const unknownTopField = artifact();
unknownTopField.fabricated = true;
const unknownResult = validateBeliefComparisonArtifact(unknownTopField);
eq(unknownResult.status, 'unavailable', 'unknown top-level field fails strict schema');
eq(unknownResult.panels.length, 0, 'structural artifact failure exposes no panels');

const obsoleteSchemaField = artifact();
obsoleteSchemaField.schema_version = 1;
eq(validateBeliefComparisonArtifact(obsoleteSchemaField).status, 'unavailable', 'obsolete schema field fails closed');

const obsoleteProvenanceNotes = artifact();
obsoleteProvenanceNotes.provenance.notes = null;
rejectsArtifact(obsoleteProvenanceNotes, 'obsolete provenance notes', 'unexpected fields notes');

ok(valid.provenance?.bootstrap_resamples === RESAMPLES, 'bootstrap provenance retained');

if (failures) {
	console.error(`\n${failures} statement-belief contract assertion(s) failed`);
	process.exit(1);
}
console.log('statement-belief comparison contract assertions passed');
