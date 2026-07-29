/**
 * Pure-function assertions for the belief-model-ladder data contract.
 *
 * A NEW runner rather than another tail on `test-paper-literal-contract.mjs`:
 * that file is shared with a concurrently-running node, and two nodes appending
 * to the same tail collide. Everything here follows the same idiom — eq/ok
 * counters, a fixture builder, mutate-and-expect-throw helpers, exit 1 on any
 * failure — so the two read the same way.
 *
 * Covers: a well-formed fixture validates; the SHIPPED bytes validate and their
 * sha256 matches the run manifest's `output_sha256` entry; the display sort is
 * monotone in average precision; the kind -> hue mapping returns exactly the
 * tokens `paperArmColorVar` returns; the referent derivation picks the paper's
 * strongest literal arm and lands inside the shipped range; and every arithmetic
 * identity the figure's geometry depends on fails CLOSED when broken.
 */
import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';

import {
	BELIEF_LADDER_BASELINE_LABEL,
	BELIEF_LADDER_CAVEAT_COUNT,
	BELIEF_LADDER_PROSE_ANCHORS,
	BELIEF_LADDER_ENTRY_SPECS,
	BELIEF_LADDER_METRIC,
	BELIEF_LADDER_NOISY_OR_FORMULA,
	BELIEF_LADDER_PARITY_TOL,
	BELIEF_LADDER_VS_LLMS_BASENAME,
	BELIEF_LADDER_WRONG_NOISY_OR_FRAGMENT,
	beliefLadderColorVar,
	beliefLadderDisplayOrder,
	beliefLadderPaperKind,
	beliefLadderReferents,
	validateBeliefLadder
} from '../src/lib/data/paper-belief-ladder.ts';
import { PAPER_LITERAL_ARM_SPECS, paperArmColorVar } from '../src/lib/data/paper-literal.ts';

let failures = 0;

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

function close(got, want, label) {
	if (!(Math.abs(got - want) <= 1e-9)) {
		failures++;
		console.error(`FAIL ${label}: got ${got}, want ${want}`);
	}
}

// ---------------------------------------------------------------------------
// FIXTURE. Synthetic average precisions; every delta, range and gap is COMPUTED
// from them, so the fixture can never encode an identity it does not satisfy.
// ---------------------------------------------------------------------------
const FIXTURE_AP = [0.895, 0.897, 0.9, 0.905, 0.91, 0.94, 0.94, 0.939, 0.95, 0.949, 0.948, 0.92];
const FIXTURE_SIBLING = `data/results/fixture/${BELIEF_LADDER_VS_LLMS_BASENAME}`;
const FIXTURE_MANIFEST = 'data/results/fixture/manifest.json';
/** Ladder label -> the `point_metrics` key it is recorded under, for the sibling arms. */
const FIXTURE_SIBLING_KEYS = {
	'RF 2k-d13 + Type/#PMIDs/promoter': 'Paper literal RF+promoter',
	'Gemma 4 26B gate': 'Gemma 4 26B',
	'GLM-5 gate': 'GLM-5',
	'Gemma 4 31B gate': 'Gemma 4 31B',
	'Gemma 4 E2B gate': 'Gemma 4 E2B'
};
const FIXTURE_GATE_LABEL = 'Gemma 4 26B gate';
const FIXTURE_FEATURES_LABEL = 'CountsScorer RF, full features';
const FIXTURE_LITERAL_LABEL = 'RF 2k-d13 + Type/#PMIDs/promoter';
const FIXTURE_VARIANT_LABEL = 'BayesianScorer, source+subtype refit';

function fixtureApOf(label) {
	return FIXTURE_AP[BELIEF_LADDER_ENTRY_SPECS.findIndex((spec) => spec.label === label)];
}

function ladderFixture() {
	const baselineAp = fixtureApOf(BELIEF_LADDER_BASELINE_LABEL);
	const entries = BELIEF_LADDER_ENTRY_SPECS.map((spec, index) => {
		const average = FIXTURE_AP[index];
		const sibling = FIXTURE_SIBLING_KEYS[spec.label] !== undefined;
		return {
			label: spec.label,
			kind: spec.kind,
			average_precision: average,
			recorded_average_precision: average,
			disagreement_vs_recorded: 0.0,
			agrees_with_recorded: true,
			distinct_scores: 100 + index,
			scores_path: `data/results/fixture/${index}.jsonl`,
			scores_key: null,
			recorded_in: sibling ? FIXTURE_SIBLING : FIXTURE_MANIFEST,
			recorded_key: sibling ? FIXTURE_SIBLING_KEYS[spec.label] : `fixture_key_${index}`,
			// Every restatement is pinned to a fragment of the sentence it was written
			// for, so a pure-placeholder fixture would gate the loader before this
			// runner could exercise anything past it. Same splice as the caveats below.
			note: `fixture rung ${index}: ${BELIEF_LADDER_PROSE_ANCHORS.notes[spec.label]}`,
			delta_vs_noisy_or_baseline:
				spec.label === BELIEF_LADDER_BASELINE_LABEL ? 0.0 : average - baselineAp
		};
	});
	const apOf = (label) => fixtureApOf(label);
	const gateAp = apOf(FIXTURE_GATE_LABEL);
	const againstBest = {
		[FIXTURE_FEATURES_LABEL]: gateAp - apOf(FIXTURE_FEATURES_LABEL),
		[FIXTURE_LITERAL_LABEL]: gateAp - apOf(FIXTURE_LITERAL_LABEL)
	};
	const againstValues = Object.values(againstBest);

	return {
		artifact_kind: 'belief_model_ladder',
		schema_version: 1,
		metric: BELIEF_LADDER_METRIC,
		metric_source: 'fixture',
		noisy_or_formula: BELIEF_LADDER_NOISY_OR_FORMULA,
		panel: {
			n: 100,
			n_errors: 30,
			n_correct: 70,
			error_base_rate: 0.3,
			label: 'fixture.released_paper_correct',
			label_convention: 'fixture convention',
			negative_breakdown: {
				n_errors: 30,
				adjudication_safe_negatives: 20,
				flagged_label_is_adjudication_safe_false: 10
			},
			ordering: 'sorted(statement_id)'
		},
		baseline: {
			label: BELIEF_LADDER_BASELINE_LABEL,
			average_precision: baselineAp,
			why: 'fixture baseline reason',
			formula: BELIEF_LADDER_NOISY_OR_FORMULA
		},
		entries,
		delta_guardrails: {
			baseline_label: BELIEF_LADDER_BASELINE_LABEL,
			baseline_average_precision: baselineAp,
			engineered_features: {
				label: FIXTURE_FEATURES_LABEL,
				average_precision: apOf(FIXTURE_FEATURES_LABEL),
				delta_vs_noisy_or_baseline: apOf(FIXTURE_FEATURES_LABEL) - baselineAp
			},
			reading_gate: {
				label: FIXTURE_GATE_LABEL,
				average_precision: gateAp,
				delta_vs_noisy_or_baseline: gateAp - baselineAp,
				delta_vs_best_noisy_or_variant: {
					label: FIXTURE_VARIANT_LABEL,
					delta: gateAp - apOf(FIXTURE_VARIANT_LABEL)
				},
				delta_vs_best_paper_model: againstBest,
				delta_vs_best_paper_model_range: [
					Math.min(...againstValues),
					Math.max(...againstValues)
				]
			},
			flat_against_baseline: {
				'Hierarchy propagation': apOf('Hierarchy propagation') - baselineAp,
				'CountsScorer RF, source counts':
					apOf('CountsScorer RF, source counts') - baselineAp
			},
			reimplementation_proximity: {
				reimplemented_rf_full_features: apOf(FIXTURE_FEATURES_LABEL),
				paper_literal_rf_promoter: apOf(FIXTURE_LITERAL_LABEL),
				absolute_gap: Math.abs(apOf(FIXTURE_FEATURES_LABEL) - apOf(FIXTURE_LITERAL_LABEL)),
				status: 'consistency check ACROSS DIFFERENT CORPORA — not fidelity evidence',
				fidelity_evidence: {
					statistic: 'per-statement Pearson r, literal vs semantic port',
					value: 0.999,
					source: FIXTURE_SIBLING
				}
			}
		},
		caveats: BELIEF_LADDER_PROSE_ANCHORS.caveats.map(
			(anchor, i) => `fixture caveat ${i}: ${anchor}`
		),
		checks: {
			every_entry_covers_the_panel_exactly: true,
			gold_matches_hash_agrees_with_prediction_provenance: true,
			literal_arm_joins_on_matches_hash: true,
			baseline_delta_is_exactly_zero: true,
			recorded_value_agreement_tol: BELIEF_LADDER_PARITY_TOL,
			n_entries: BELIEF_LADDER_ENTRY_SPECS.length,
			n_entries_agreeing_with_recorded_value: BELIEF_LADDER_ENTRY_SPECS.length,
			n_entries_disagreeing_with_recorded_value: 0,
			same_fitted_model_pair: [FIXTURE_FEATURES_LABEL, 'HybridScorer, full features'],
			same_fitted_model_absolute_gap: 0.0,
			same_fitted_model_tol: BELIEF_LADDER_PARITY_TOL,
			note: 'fixture note'
		},
		provenance: {
			gold: 'data/results/fixture/gold.jsonl',
			scores: Object.fromEntries(entries.map((entry) => [entry.label, entry.scores_path])),
			recorded_values: Object.fromEntries(
				entries.map((entry) => [entry.label, { path: entry.recorded_in, key: entry.recorded_key }])
			),
			join: 'statement_id',
			generated_by: 'scripts/compute_belief_model_ladder.py'
		}
	};
}

function pointMetricsFixture() {
	const table = {
		'Paper literal RF+promoter': { pooled_average_precision: 0.939, fold_population_sd: 0.014 },
		'Paper literal RF+prom/avglen': { pooled_average_precision: 0.9395, fold_population_sd: 0.015 },
		'Paper semantic port RF+promoter': { pooled_average_precision: 0.939, fold_population_sd: 0.0137 },
		'Gemma 4 E2B': { pooled_average_precision: 0.92, fold_population_sd: 0.017 },
		'Gemma 4 26B': { pooled_average_precision: 0.95, fold_population_sd: 0.011 },
		'Gemma 4 31B': { pooled_average_precision: 0.948, fold_population_sd: 0.0111 },
		'GLM-5': { pooled_average_precision: 0.949, fold_population_sd: 0.0103 },
		'INDRA CoGEx hybrid': { pooled_average_precision: 0.9227, fold_population_sd: 0.023 }
	};
	return table;
}

const fixture = ladderFixture();
const parsedFixture = validateBeliefLadder(fixture);
eq(parsedFixture.entries.length, BELIEF_LADDER_ENTRY_SPECS.length, 'fixture validates with every rung');
eq(parsedFixture.baseline.label, BELIEF_LADDER_BASELINE_LABEL, 'fixture baseline is the marked rung');
eq(parsedFixture.caveats.length, BELIEF_LADDER_CAVEAT_COUNT, 'fixture carries every caveat');
eq(
	parsedFixture.entries.find((entry) => entry.isBaseline)?.deltaVsNoisyOrBaseline,
	0,
	'the fixture baseline rung is exactly zero'
);

const fixtureReferents = beliefLadderReferents(parsedFixture, pointMetricsFixture());
eq(
	fixtureReferents.referents.filter((referent) => referent.derived).length,
	1,
	'exactly one referent is derived from the sibling artifact'
);
eq(
	fixtureReferents.referents.find((referent) => referent.derived)?.armDisplay,
	'RF 2k-d13 + Type/#PMIDs/prom/avglen',
	'the derived referent renders the DISPLAY name, never the frozen join key'
);
eq(fixtureReferents.foldSd.nArms, Object.keys(FIXTURE_SIBLING_KEYS).length, 'every sibling rung carries a fold SD');

// ---------------------------------------------------------------------------
// ORDERING + HUE. Layout decisions, asserted so they cannot silently invert.
// ---------------------------------------------------------------------------
const fixtureOrder = beliefLadderDisplayOrder(parsedFixture);
ok(
	fixtureOrder.every(
		(entry, index) => index === 0 || fixtureOrder[index - 1].averagePrecision <= entry.averagePrecision
	),
	'display order is monotone non-decreasing in average precision'
);
eq(fixtureOrder.length, parsedFixture.entries.length, 'display order keeps every rung');

eq(beliefLadderPaperKind('paper-family'), 'paper', 'the paper family keeps the paper hue kind');
eq(beliefLadderPaperKind('paper-literal'), 'paper', 'the literal arm keeps the paper hue kind');
eq(beliefLadderPaperKind('reader-gate'), 'llm', 'the reader gates keep the llm hue kind');
for (const entry of parsedFixture.entries) {
	eq(
		beliefLadderColorVar(entry),
		paperArmColorVar(beliefLadderPaperKind(entry.kind)),
		`${entry.label}: hue is the page-wide token`
	);
}
ok(
	new Set(parsedFixture.entries.map((entry) => beliefLadderColorVar(entry))).size ===
		new Set(PAPER_LITERAL_ARM_SPECS.map((spec) => paperArmColorVar(spec.kind))).size,
	'the ladder introduces no colour token the page does not already use'
);

// ---------------------------------------------------------------------------
// SHIPPED-NUMBER PARITY. The figure must not be able to drift from the artifact.
// ---------------------------------------------------------------------------
const MODEL_DIR = new URL('../../data/results/indra_paper_literal_models_20260724/', import.meta.url);
const LADDER_NAME = 'belief_model_ladder.json';
const ladderBytes = readFileSync(new URL(LADDER_NAME, MODEL_DIR));
const shippedLadder = JSON.parse(ladderBytes.toString('utf8'));
const shippedVsLlms = JSON.parse(
	readFileSync(new URL(BELIEF_LADDER_VS_LLMS_BASENAME, MODEL_DIR), 'utf8')
);
const runManifest = JSON.parse(readFileSync(new URL('manifest.json', MODEL_DIR), 'utf8'));

const parsed = validateBeliefLadder(shippedLadder);
eq(parsed.entries.length, BELIEF_LADDER_ENTRY_SPECS.length, 'shipped ladder validates with every rung');
eq(
	parsed.entries.map((entry) => `${entry.label}|${entry.kind}`).join(' / '),
	BELIEF_LADDER_ENTRY_SPECS.map((spec) => `${spec.label}|${spec.kind}`).join(' / '),
	'shipped rungs are in the fixed presentation order'
);
eq(
	createHash('sha256').update(ladderBytes).digest('hex'),
	runManifest.output_sha256[LADDER_NAME],
	'ladder artifact sha256 matches the run manifest'
);
eq(runManifest.outputs.belief_model_ladder, LADDER_NAME, 'the manifest names the ladder output');
eq(parsed.noisyOrFormula, BELIEF_LADDER_NOISY_OR_FORMULA, 'the shipped noisy-OR form is the right one');
ok(
	!ladderBytes.toString('utf8').includes(BELIEF_LADDER_WRONG_NOISY_OR_FRAGMENT),
	'the wrong noisy-OR form appears nowhere in the shipped bytes'
);

// The shipped bar geometry: every bar IS ap - baseline, the baseline's is zero.
for (const entry of parsed.entries) {
	close(
		entry.deltaVsNoisyOrBaseline,
		entry.averagePrecision - parsed.baseline.averagePrecision,
		`${entry.label}: shipped delta is ap minus the baseline`
	);
}
eq(
	parsed.entries.find((entry) => entry.isBaseline)?.deltaVsNoisyOrBaseline,
	0,
	'the shipped baseline rung is exactly zero'
);

const shippedOrder = beliefLadderDisplayOrder(parsed);
ok(
	shippedOrder.every(
		(entry, index) => index === 0 || shippedOrder[index - 1].averagePrecision <= entry.averagePrecision
	),
	'shipped display order is monotone in average precision'
);

const referents = beliefLadderReferents(parsed, shippedVsLlms.point_metrics);
eq(referents.referents.length, 3, 'three named referents ride with the gate delta');
ok(
	referents.referents.every(
		(referent, index) => index === 0 || referents.referents[index - 1].delta <= referent.delta
	),
	'referents come back sorted delta ascending, conservative first'
);
const derived = referents.referents.find((referent) => referent.derived);
eq(
	derived?.armDisplay,
	'RF 2k-d13 + Type/#PMIDs/prom/avglen',
	'the derived referent is the strongest literal arm, rendered by display name'
);
const [rangeLow, rangeHigh] = parsed.guardrails.readingGate.deltaVsBestPaperModelRange;
ok(
	derived.delta >= rangeLow && derived.delta <= rangeHigh,
	'the derived referent lands inside the shipped against-their-best range'
);
ok(
	rangeHigh < parsed.guardrails.readingGate.deltaVsNoisyOrBaseline,
	'the against-their-best range is strictly tighter than the from-baseline delta'
);
eq(referents.foldSd.nArms, 5, 'five shipped rungs carry the paper own fold SD');
ok(
	referents.foldSd.min > 0 && referents.foldSd.max > referents.foldSd.min,
	'the shipped fold-SD range is a real spread'
);

// ---------------------------------------------------------------------------
// FAIL-CLOSED. Drift in any identity must THROW, not draw a wrong bar.
// ---------------------------------------------------------------------------
let mutations = 0;

function ladderRejects(mutate, label) {
	mutations++;
	const bad = structuredClone(shippedLadder);
	mutate(bad);
	let threw = false;
	try {
		validateBeliefLadder(bad);
	} catch {
		threw = true;
	}
	ok(threw, `${label}: ladder validation fails closed`);
}

function referentsRejects(mutate, label) {
	mutations++;
	const bad = structuredClone(shippedVsLlms.point_metrics);
	mutate(bad);
	let threw = false;
	try {
		beliefLadderReferents(parsed, bad);
	} catch {
		threw = true;
	}
	ok(threw, `${label}: referent derivation fails closed`);
}

ladderRejects((bad) => {
	bad.entries[0].delta_vs_noisy_or_baseline += 0.01;
}, 'a bar length that is not ap minus the baseline');
ladderRejects((bad) => {
	const index = bad.entries.findIndex((entry) => entry.label === BELIEF_LADDER_BASELINE_LABEL);
	bad.entries[index].delta_vs_noisy_or_baseline = Number.MIN_VALUE;
}, 'a baseline rung whose own delta is not exactly zero');
ladderRejects((bad) => bad.entries.reverse(), 'rungs out of the fixed presentation order');
ladderRejects((bad) => bad.entries.pop(), 'a dropped rung');
ladderRejects((bad) => bad.caveats.pop(), 'a dropped caveat');
ladderRejects((bad) => {
	bad.delta_guardrails.reading_gate.delta_vs_best_paper_model_range[0] -= 0.001;
}, 'a range that is not the min/max of its own map');
ladderRejects((bad) => {
	// A referent so weak that the range swallows the from-baseline delta.
	const gate = bad.delta_guardrails.reading_gate;
	const weakest = bad.entries.reduce((low, entry) =>
		entry.average_precision < low.average_precision ? entry : low
	);
	gate.delta_vs_best_paper_model[weakest.label] = gate.average_precision - weakest.average_precision;
	const values = Object.values(gate.delta_vs_best_paper_model);
	gate.delta_vs_best_paper_model_range = [Math.min(...values), Math.max(...values)];
}, 'an against-their-best range that is not tighter than the from-baseline delta');
ladderRejects((bad) => {
	bad.noisy_or_formula = 'belief = 1 - PROD (1-r_s)^n';
}, 'the wrong noisy-OR form in the formula field');
ladderRejects((bad) => {
	bad.caveats[0] = `${bad.caveats[0]} ${BELIEF_LADDER_WRONG_NOISY_OR_FRAGMENT}_s)^n`;
}, 'the wrong noisy-OR form smuggled into a caveat');
ladderRejects((bad) => {
	bad.schema_version = 2;
}, 'an unknown ladder schema version');
ladderRejects((bad) => {
	bad.metric = 'ap_points';
}, 'a renamed or rescaled metric');
ladderRejects((bad) => {
	bad.delta_guardrails.reimplementation_proximity.absolute_gap += 0.001;
}, 'a proximity gap that does not match its own two operands');
ladderRejects((bad) => {
	bad.checks.same_fitted_model_absolute_gap = 0.01;
}, 'a same-fitted-model pair that does not actually agree');
ladderRejects((bad) => {
	bad.delta_guardrails.engineered_features.average_precision += 0.001;
}, 'a guardrail that disagrees with the rung it names');
ladderRejects((bad) => {
	const flat = bad.delta_guardrails.flat_against_baseline;
	const key = Object.keys(flat)[0];
	flat[key] += 0.001;
}, 'a flat rung whose guardrail value is not its own delta');
ladderRejects((bad) => {
	bad.panel.negative_breakdown.adjudication_safe_negatives += 1;
}, 'a negative breakdown that does not sum to the panel errors');

referentsRejects((bad) => {
	// The sibling now claims a stronger literal arm than the gate, so the derived
	// delta goes negative and escapes the range the ladder itself shipped.
	bad['Paper literal RF+prom/avglen'].pooled_average_precision = 0.99;
}, 'a derived referent delta pushed outside the shipped range');
referentsRejects((bad) => {
	// ...and the other way: both paper arms weaker than the shipped range allows.
	bad['Paper literal RF+prom/avglen'].pooled_average_precision = 0.8;
	bad['Paper literal RF+promoter'].pooled_average_precision = 0.8;
}, 'a derived referent delta below the shipped range');
referentsRejects((bad) => {
	delete bad['GLM-5'];
}, 'a fold-SD key missing from the sibling point_metrics');

console.log(`${mutations} fail-closed mutation cases exercised`);
if (mutations < 10) {
	failures++;
	console.error('FAIL fewer than 10 fail-closed mutation cases');
}

if (failures) {
	console.error(`\n${failures} belief-ladder contract assertion(s) failed`);
	process.exit(1);
}
console.log('belief-model-ladder data contract assertions passed');
