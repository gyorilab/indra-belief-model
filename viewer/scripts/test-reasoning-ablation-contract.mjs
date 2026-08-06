/**
 * Pure-function assertions for the REASONING ABLATION data contract.
 *
 * Mirrors test-per-evidence-contract.mjs: the validator is exercised against the
 * REAL shipped artifact, then against mutations that must each gate the figure to
 * `unavailable` rather than render a wrong number.
 *
 * Why this file exists. This surface carries five claims that are each one field
 * flip away from being wrong:
 *   1. the two marks in a row are the SAME statements scored twice — licensed
 *      only by an EXACT reconciliation against the shipped statement scores AND
 *      by shipped-parity on the measurements, so either failing must gate;
 *   2. the verdict-only run has NO packaging digest behind it — so any claim that
 *      it does must gate, and `verdictOnlyBundled` must be structurally false;
 *   3. the two runs roll evidence up identically — so a differing rollup must
 *      gate, because otherwise a rollup change would read as a reading change;
 *   4. direction is decided from the two ENDPOINTS, never from a shipped flag and
 *      never from the sign of the point estimate — so a shipped standing that
 *      disagrees with its own interval must gate;
 *   5. right-anchored SVG text clips its LEADING glyphs in silence — so the
 *      gutter budgets must be DERIVED from the geometry and enforced.
 * Each is asserted below against the live artifact and against a mutation.
 *
 * Run: node --experimental-strip-types scripts/test-reasoning-ablation-contract.mjs
 */
import { readFileSync } from 'node:fs';

import { standingOfBounds } from '../src/lib/data/paper-interval.ts';
import {
	REASONING_ABLATION_ARTIFACT_KIND,
	REASONING_ABLATION_BENCHMARK_DISPLAY,
	REASONING_ABLATION_BENCHMARK_IDS,
	REASONING_ABLATION_GEOMETRY,
	REASONING_ABLATION_NAME_BUDGET_CHARS,
	REASONING_ABLATION_READOUT_BUDGET_CHARS,
	REASONING_ABLATION_SIDE_DISPLAY,
	REASONING_ABLATION_SIDE_IDS,
	REASONING_ABLATION_SIMULTANEOUS_SENTENCE,
	REASONING_ABLATION_STANDING_SENTENCE,
	fmt3,
	fmtSignedDelta,
	mostDecidedModel,
	reasoningAblationExtent,
	reasoningAblationNameFits,
	reasoningAblationReadoutFits,
	validatePaperReasoningAblation
} from '../src/lib/data/paper-reasoning-ablation.ts';

let failures = 0;
function ok(condition, label) {
	if (!condition) {
		console.error(`FAIL ${label}`);
		failures += 1;
	}
}
function eq(actual, expected, label) {
	ok(
		Object.is(actual, expected),
		`${label}: got ${JSON.stringify(actual)}, want ${JSON.stringify(expected)}`
	);
}
/** A mutation must gate. Passing one through is the failure mode this file hunts. */
function gates(mutate, label) {
	const copy = structuredClone(raw);
	mutate(copy);
	const result = validatePaperReasoningAblation(copy, { artifactPath: 'fixture' });
	ok(result.status === 'unavailable', `${label}: expected the figure to gate`);
	if (result.status === 'unavailable') {
		ok(result.figure === null, `${label}: a gated figure carries no geometry`);
		ok(typeof result.reason === 'string' && result.reason.length > 0, `${label}: carries a reason`);
	}
}

const ARTIFACT = new URL(
	'../../data/results/reasoning_ablation_20260805/reasoning_ablation.json',
	import.meta.url
);
const raw = JSON.parse(readFileSync(ARTIFACT, 'utf8'));

eq(raw.artifact_kind, REASONING_ABLATION_ARTIFACT_KIND, 'the artifact is the kind we accept');

const live = validatePaperReasoningAblation(raw, {
	artifactPath: 'fixture',
	artifactSha256: 'deadbeef'
});
eq(
	live.status,
	'ok',
	`the shipped artifact validates${live.status === 'ok' ? '' : `: ${live.reason}`}`
);
if (live.status !== 'ok') {
	console.error('reasoning-ablation contract: cannot continue without a valid figure');
	process.exit(1);
}
const figure = live.figure;

// -------------------------------------------------------------------------
console.log('\n(1) the two marks in a row are the same statements scored twice');
// -------------------------------------------------------------------------
eq(figure.shippedParityVerified, true, 'the figure asserts shipped parity structurally');
for (const model of figure.models) {
	for (const benchmark of model.benchmarks) {
		eq(
			benchmark.nPositive + benchmark.nNegative,
			benchmark.nEvaluable,
			`${model.id}/${benchmark.id}: the counts partition the benchmark`
		);
		for (const side of ['reasoning', 'verdictOnly']) {
			const confusion = benchmark[side].confusion;
			eq(
				confusion.tp + confusion.fp + confusion.fn + confusion.tn,
				benchmark.nEvaluable,
				`${model.id}/${benchmark.id}/${side}: every statement is counted once`
			);
			eq(
				confusion.tp + confusion.fn,
				benchmark.nPositive,
				`${model.id}/${benchmark.id}/${side}: the positive count is recovered`
			);
		}
	}
}
const executionCounts = new Set(figure.models.map((model) => model.evidence.nExecutions));
eq(executionCounts.size, 1, 'every model read the same number of evidence rows');

gates(
	(copy) => (copy.arms[0].reconciliation.paper_all_source.max_abs_diff = 1e-9),
	'a non-exact reconciliation'
);
gates(
	(copy) => (copy.arms[0].reconciliation.paper_all_source.n_exact -= 1),
	'a reconciliation that is exact on fewer statements than it scored'
);
gates(
	(copy) => (copy.arms[1].panels.paper_readers.shipped_parity_verified = false),
	'a benchmark whose thinking-first side was never checked against the published numbers'
);
gates((copy) => delete copy.arms[1].panels.paper_readers.shipped_parity_verified, 'a missing parity flag');
gates(
	(copy) => (copy.arms[0].panels.paper_all_source.n_positive += 1),
	'a benchmark whose parts no longer sum to its whole'
);
gates(
	(copy) => (copy.arms[0].panels.paper_all_source.reasoning.confusion.tn += 1),
	'a confusion count that no longer covers the benchmark exactly'
);
gates(
	(copy) => (copy.arms[0].evidence_grain.n_executions += 1),
	'models that did not read the same rows, so their rows are not paired'
);

// -------------------------------------------------------------------------
console.log('\n(1b) the DRAWN metric is the error class, and its counts close');
// -------------------------------------------------------------------------
for (const model of figure.models) {
	for (const benchmark of model.benchmarks) {
		const { ownCut, deployedCut, cutsAgree } = benchmark.errorClass;
		for (const [name, cut] of [
			['own/reasoning', ownCut.reasoning],
			['own/verdictOnly', ownCut.verdictOnly],
			['deployed/reasoning', deployedCut.reasoning],
			['deployed/verdictOnly', deployedCut.verdictOnly]
		]) {
			const where = `${model.id}/${benchmark.id}/${name}`;
			eq(cut.tp + cut.fn, benchmark.nNegative, `${where}: the error class closes on the benchmark`);
			eq(cut.tp + cut.fp, cut.flagged, `${where}: the review pile is what the cut flags`);
			eq(
				cut.tp + cut.fp + cut.fn + cut.tn,
				benchmark.nEvaluable,
				`${where}: every statement is counted once`
			);
			// F1 must BE the harmonic mean of the precision and recall beside it.
			const expected =
				cut.errorPrecision + cut.errorRecall === 0
					? 0
					: (2 * cut.errorPrecision * cut.errorRecall) / (cut.errorPrecision + cut.errorRecall);
			ok(Math.abs(cut.errorF1 - expected) < 1e-9, `${where}: F1 is its own precision and recall`);
		}
		// The deployed rule IS the deliberating side's own cut, by definition.
		eq(
			deployedCut.tau,
			ownCut.reasoning.tau,
			`${model.id}/${benchmark.id}: the deployed cut is the thinking side's own`
		);
		eq(
			cutsAgree,
			ownCut.verdictOnly.tau === deployedCut.tau,
			`${model.id}/${benchmark.id}: cutsAgree is derived, not asserted`
		);
		// When the two rules pick the same cut, the two deltas MUST coincide;
		// printing them as separate findings would double-count one measurement.
		if (cutsAgree) {
			ok(
				Math.abs(benchmark.errorF1OwnCutDelta.value - benchmark.errorF1DeployedCutDelta.value) <
					1e-12,
				`${model.id}/${benchmark.id}: identical cuts give identical margins`
			);
		}
	}
}
gates(
	(copy) => (copy.arms[0].panels.paper_all_source.error_class.own_cut.reasoning.tp += 1),
	'an error-class cut whose caught errors no longer close on the benchmark'
);
gates(
	(copy) => (copy.arms[0].panels.paper_all_source.error_class.own_cut.reasoning.flagged += 1),
	'a review pile that is not the cut’s own flag set'
);
gates(
	(copy) => (copy.arms[0].panels.paper_all_source.error_class.deployed_cut.tau = 0.42),
	'a deployed cut that is not the thinking side’s own'
);
gates((copy) => {
	copy.arms[0].panels.paper_all_source.error_class.deployed_cut.reasoning.error_f1 = 0.5;
}, 'the same side at the same cut disagreeing with itself across the two rules');

// The threshold rules must reach the figure, and the ORACLE one especially: a
// benchmark-fitted cutoff printed without it reads as one you could have picked
// in advance.
for (const half of ['decision', 'threshold', 'oracle']) {
	ok(
		figure.errorRules[half].plain.length > 0 &&
			figure.errorRules[half].plain !== figure.errorRules[half].shipped,
		`${half}: a plain restatement is authored, not the shipped sentence again`
	);
}
ok(
	figure.errorRules.oracle.shipped.includes('ON THIS PANEL'),
	'the oracle disclosure is carried in the artifact’s own words'
);
gates(
	(copy) => (copy.provenance.error_class.oracle_disclosure = 'cutoffs were chosen in advance'),
	'a reworded oracle disclosure whose restatement was written for a different sentence'
);
gates((copy) => delete copy.provenance.error_class, 'an artifact carrying no threshold rules at all');

// -------------------------------------------------------------------------
console.log('\n(2) the verdict-only run carries no packaging digest, and says so');
// -------------------------------------------------------------------------
eq(figure.bundler.verdictOnlyBundled, false, 'the figure states the run was never packaged');
ok(
	figure.bundler.error.shipped.includes('topology'),
	'the refusal is carried in the artifact’s own words'
);
for (const half of ['error', 'cause', 'consequence']) {
	ok(
		figure.bundler[half].plain.length > 0 && figure.bundler[half].plain !== figure.bundler[half].shipped,
		`${half}: a plain restatement is authored, not the shipped sentence again`
	);
}
gates((copy) => (copy.bundler_status.state = 'available'), 'a claim that the run was packaged');
gates(
	(copy) => (copy.arms[0].verdict_only.raw_attempts.sha256_matches_bundle_manifest = true),
	'a verdict-only side claiming a bundle digest it cannot have'
);
gates(
	(copy) => (copy.bundler_status.cause = 'the packaging step was not run'),
	'a reworded refusal whose plain restatement was written for a different sentence'
);

// -------------------------------------------------------------------------
console.log('\n(3) the two runs roll evidence up identically');
// -------------------------------------------------------------------------
eq(
	raw.provenance.aggregation_identical_across_runs,
	true,
	'the producer verified one rollup across both runs'
);
gates(
	(copy) => (copy.provenance.aggregation_identical_across_runs = false),
	'two runs that roll evidence up differently'
);
gates(
	(copy) => delete copy.provenance.aggregation_identical_across_runs,
	'a missing rollup-identity flag'
);

// -------------------------------------------------------------------------
console.log('\n(4) direction is read off the endpoints, never off a flag or a sign');
// -------------------------------------------------------------------------
for (const model of figure.models) {
	for (const benchmark of model.benchmarks) {
		for (const [name, delta] of [
			['average precision', benchmark.averagePrecisionDelta],
			['AUROC', benchmark.aurocDelta],
			['error-F1 own cut', benchmark.errorF1OwnCutDelta],
			['error-F1 deployed cut', benchmark.errorF1DeployedCutDelta]
		]) {
			eq(
				delta.standing,
				standingOfBounds(delta.low, delta.high),
				`${model.id}/${benchmark.id}/${name}: standing is re-derived from its own bounds`
			);
			ok(delta.low <= delta.high, `${model.id}/${benchmark.id}/${name}: the interval is ordered`);
			ok(
				typeof REASONING_ABLATION_STANDING_SENTENCE[delta.standing] === 'string',
				`${model.id}/${benchmark.id}/${name}: a sentence exists for this class`
			);
			// The correction for having run four models can only WIDEN.
			eq(
				delta.simultaneousStanding,
				standingOfBounds(delta.simultaneousLow, delta.simultaneousHigh),
				`${model.id}/${benchmark.id}/${name}: the widened standing is re-derived too`
			);
			ok(
				delta.simultaneousLow <= delta.low && delta.simultaneousHigh >= delta.high,
				`${model.id}/${benchmark.id}/${name}: the widened range contains the pointwise one`
			);
			ok(
				typeof REASONING_ABLATION_SIMULTANEOUS_SENTENCE[delta.simultaneousStanding] === 'string',
				`${model.id}/${benchmark.id}/${name}: a widened sentence exists for this class`
			);
			// A correction may never PROMOTE a claim the pointwise range left open.
			ok(
				!(delta.standing === 'not-significant' && delta.simultaneousStanding !== 'not-significant'),
				`${model.id}/${benchmark.id}/${name}: widening never settles what pointwise left open`
			);
			ok(delta.familySize >= 1, `${model.id}/${benchmark.id}/${name}: names a family size`);
		}
	}
}
// The two sentence records must be DISTINCT and both TOTAL: reusing the
// pointwise wording under a widened range is how an uncorrected margin gets
// reported as a corrected one.
for (const standing of ['ahead', 'behind', 'not-significant']) {
	ok(
		REASONING_ABLATION_SIMULTANEOUS_SENTENCE[standing] !==
			REASONING_ABLATION_STANDING_SENTENCE[standing],
		`the ${standing} class has its own widened wording`
	);
}
gates((copy) => {
	const delta = copy.arms[0].panels.paper_all_source.delta.error_f1_own_cut;
	delta.simultaneous.ci95 = [delta.ci95[0] / 2, delta.ci95[1] / 2];
	delta.simultaneous.standing = 'behind';
}, 'a widened range narrower than the pointwise one it corrects');
gates(
	(copy) =>
		(copy.arms[0].panels.paper_all_source.delta.error_f1_own_cut.simultaneous.standing = 'ahead'),
	'a widened standing that disagrees with its own interval'
);
// The record must be TOTAL: a missing class is a template with nothing to say.
for (const standing of ['ahead', 'behind', 'not-significant']) {
	ok(
		(REASONING_ABLATION_STANDING_SENTENCE[standing] ?? '').length > 0,
		`a sentence is written for the ${standing} class`
	);
}
// An interval lying entirely BELOW zero is the case the page's sixth
// sign-blindness regression got wrong. Assert it directly.
eq(standingOfBounds(-0.0375, -0.0069), 'behind', 'an interval below zero reads as behind');
eq(standingOfBounds(-0.0203, 0.0015), 'not-significant', 'an interval spanning zero is undecided');

gates(
	(copy) => (copy.arms[0].panels.paper_all_source.delta.auroc.standing = 'ahead'),
	'a shipped standing that disagrees with its own interval'
);
gates((copy) => {
	const bounds = copy.arms[0].panels.paper_all_source.delta.auroc.ci95;
	copy.arms[0].panels.paper_all_source.delta.auroc.ci95 = [bounds[1], bounds[0]];
}, 'an inverted interval');
gates(
	(copy) => (copy.arms[0].panels.paper_all_source.delta.auroc.value += 0.01),
	'a point difference that is not the two measurements differenced'
);
gates(
	(copy) => (copy.arms[0].panels.paper_all_source.delta.average_precision.value -= 0.01),
	'an average-precision difference that is not the two measurements differenced'
);

// -------------------------------------------------------------------------
console.log('\n(5) the gutter budgets are derived from the geometry and enforced');
// -------------------------------------------------------------------------
const G = REASONING_ABLATION_GEOMETRY;
eq(
	REASONING_ABLATION_NAME_BUDGET_CHARS,
	Math.floor(G.labelAnchorX / G.monoUnitsPerChar),
	'the name budget is re-derived from the left gutter'
);
eq(
	REASONING_ABLATION_READOUT_BUDGET_CHARS,
	Math.floor((G.width - G.readoutX) / G.readoutUnitsPerChar),
	'the readout budget is re-derived from the right gutter'
);
ok(G.plotLeft < G.plotRight, 'the plot has a positive span');
ok(G.labelAnchorX <= G.plotLeft, 'names are anchored outside the plot they label');
ok(G.readoutX >= G.plotRight, 'readouts are anchored outside the plot they annotate');

for (const model of figure.models) {
	ok(
		reasoningAblationNameFits(model.display),
		`${model.id}: "${model.display}" fits the left gutter`
	);
	for (const benchmark of model.benchmarks) {
		// The DRAWN readouts, built exactly as the component builds them.
		const delta = benchmark.errorF1OwnCutDelta;
		const range = `${fmtSignedDelta(delta.value)} [${fmtSignedDelta(delta.low)},${fmtSignedDelta(delta.high)}]`;
		ok(
			reasoningAblationReadoutFits(range),
			`${model.id}/${benchmark.id}: the range readout "${range}" fits the right gutter`
		);
		const values = `${fmt3(benchmark.errorClass.ownCut.reasoning.errorF1)} → ${fmt3(
			benchmark.errorClass.ownCut.verdictOnly.errorF1
		)}`;
		ok(
			reasoningAblationReadoutFits(values),
			`${model.id}/${benchmark.id}: the value readout "${values}" fits the right gutter`
		);
	}
}
// THE BUDGET IS NOT VACUOUS: a name one character over its budget must be
// rejected, and a model carrying one must gate the whole figure.
ok(
	!reasoningAblationNameFits('x'.repeat(REASONING_ABLATION_NAME_BUDGET_CHARS + 1)),
	'a name one character over budget does not fit'
);
ok(
	!reasoningAblationReadoutFits('x'.repeat(REASONING_ABLATION_READOUT_BUDGET_CHARS + 1)),
	'a readout one character over budget does not fit'
);
gates(
	(copy) => (copy.arms[0].display = 'x'.repeat(REASONING_ABLATION_NAME_BUDGET_CHARS + 1)),
	'a model name that would clip its leading glyphs'
);

// -------------------------------------------------------------------------
console.log('\n(6) shape, keys and the pure helpers');
// -------------------------------------------------------------------------
eq(REASONING_ABLATION_BENCHMARK_IDS.length, 2, 'two benchmarks are declared');
for (const id of REASONING_ABLATION_BENCHMARK_IDS) {
	ok(
		(REASONING_ABLATION_BENCHMARK_DISPLAY[id] ?? '').length > 0,
		`${id}: an on-screen name is written`
	);
}
for (const id of REASONING_ABLATION_SIDE_IDS) {
	ok((REASONING_ABLATION_SIDE_DISPLAY[id] ?? '').length > 0, `${id}: an on-screen name is written`);
}
for (const model of figure.models) {
	eq(
		model.benchmarks.length,
		REASONING_ABLATION_BENCHMARK_IDS.length,
		`${model.id}: carries every declared benchmark`
	);
	ok(model.reasoningCost.lower <= model.reasoningCost.upper, `${model.id}: cost bounds are ordered`);
	if (model.verdictOnlyCost) {
		ok(
			model.verdictOnlyCost.lower <= model.verdictOnlyCost.upper,
			`${model.id}: verdict-only cost bounds are ordered`
		);
	}
	const evidence = model.evidence;
	ok(
		evidence.toCorrect + evidence.toIncorrect <= evidence.nModelRead,
		`${model.id}: the changed readings do not exceed the readings taken`
	);
	ok(evidence.agreement >= 0 && evidence.agreement <= 1, `${model.id}: agreement is a share`);
}

gates((copy) => (copy.artifact_kind = 'something_else'), 'an artifact of another kind');
gates((copy) => (copy.arms = []), 'an artifact with no models');
gates((copy) => (copy.arms[1].arm_id = copy.arms[0].arm_id), 'repeated model keys');
gates((copy) => delete copy.arms[0].panels.paper_readers, 'a missing benchmark');
gates(
	(copy) => (copy.arms[0].evidence_grain.llm_tier.flips += 1),
	'a changed-reading total that is not its own two directions summed'
);
gates(
	(copy) => (copy.arms[0].evidence_grain.llm_tier.flips = copy.arms[0].evidence_grain.llm_tier.n + 1),
	'more changed readings than readings taken'
);
gates((copy) => (copy.arms[0].reasoning.cost = null), 'a shipped run with no dollar bounds');

// The extent helper must cover BOTH sides of every model, or the axis it sizes
// would cut a mark off the frame.
const extent = reasoningAblationExtent(figure.models, 'paper_all_source', (side) => side.auroc);
ok(extent !== null, 'the extent helper returns a range for the drawn benchmark');
for (const model of figure.models) {
	const benchmark = model.benchmarks.find((entry) => entry.id === 'paper_all_source');
	for (const value of [benchmark.reasoning.auroc, benchmark.verdictOnly.auroc]) {
		ok(
			value >= extent.min && value <= extent.max,
			`${model.id}: ${value} lies inside the extent the axis is sized from`
		);
	}
}
eq(reasoningAblationExtent([], 'paper_all_source', (side) => side.auroc), null, 'no models, no range');

// mostDecidedModel must return null when nothing is decided rather than an
// arbitrary first entry — "nothing is decided" is a real state on this surface.
const decided = mostDecidedModel(figure.models, 'paper_all_source', (b) => b.aurocDelta);
if (decided === null) {
	ok(
		figure.models.every(
			(model) =>
				model.benchmarks.find((entry) => entry.id === 'paper_all_source').aurocDelta.standing ===
				'not-significant'
		),
		'a null result means every range covers zero'
	);
} else {
	const benchmark = decided.benchmarks.find((entry) => entry.id === 'paper_all_source');
	ok(
		benchmark.aurocDelta.standing !== 'not-significant',
		'the most decided model is one whose range does not cover zero'
	);
}

if (failures > 0) {
	console.error(`\nreasoning-ablation contract: ${failures} assertion(s) failed`);
	process.exit(1);
}
console.log('\nreasoning-ablation contract: all assertions passed');
