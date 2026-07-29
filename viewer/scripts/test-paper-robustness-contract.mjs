/**
 * Pure-function assertions for the margin-robustness data contract.
 *
 * Mirrors test-tie-inflation-contract.mjs: the validator is exercised against the
 * REAL shipped artifact, then against mutations that must each gate the figure to
 * `unavailable` rather than render a wrong number.
 *
 * Why this file exists. This surface carries the page's most misreadable numbers:
 * a family-wise band that must never be mistaken for the primary interval, and a
 * panel that is OUR revision of the paper's labels and must never be mistaken for
 * the paper's data. Both confusions are one field flip away, so both fields gate,
 * and the gate is tested here rather than assumed.
 */
import { readFileSync } from 'node:fs';

import {
	PAPER_ROBUSTNESS_DISPLAY,
	PAPER_ROBUSTNESS_GEOMETRY,
	PAPER_ROBUSTNESS_LABEL_BUDGET_CHARS,
	PAPER_ROBUSTNESS_READOUT_BUDGET_CHARS,
	PAPER_ROBUSTNESS_SERIES,
	PAPER_ROBUSTNESS_SERIES_IDS,
	validatePaperRobustness
} from '../src/lib/data/paper-robustness.ts';
import { PAPER_LITERAL_ARM_SPECS } from '../src/lib/data/paper-literal.ts';

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
function close(actual, expected, tolerance, label) {
	ok(Math.abs(actual - expected) <= tolerance, `${label}: got ${actual}, want ~${expected}`);
}

const ARTIFACT = new URL(
	'../../data/results/indra_paper_literal_models_20260724/paper_margin_robustness.json',
	import.meta.url
);
const raw = JSON.parse(readFileSync(ARTIFACT, 'utf8'));

const live = validatePaperRobustness(raw, { artifactPath: 'fixture', artifactSha256: 'deadbeef' });
eq(live.status, 'ok', 'the shipped artifact validates');

if (live.status === 'ok') {
	const figure = live.figure;
	const lanes = figure.lanes;

	// ---- the family is exactly the arms the frozen plan staged -----------------
	eq(lanes.length, raw.multiplicity.family_size, 'every family member reaches the figure');
	eq(lanes.length, 4, 'four reader arms');
	eq(
		figure.multiplicity.runPlanStages.length,
		lanes.length,
		'one run-plan stage per drawn arm'
	);
	eq(figure.multiplicity.noDesignatedPrimaryArm, true, 'the plan designates no primary arm');

	// ---- display is DECOUPLED from the frozen join key -------------------------
	const specByArmId = new Map(PAPER_LITERAL_ARM_SPECS.map((spec) => [spec.id, spec]));
	for (const lane of lanes) {
		eq(lane.display, PAPER_ROBUSTNESS_DISPLAY[lane.id], `${lane.id}: canonical display name`);
		eq(lane.label, specByArmId.get(lane.id)?.label, `${lane.id}: frozen point_metrics join key`);
		ok(lane.display !== lane.label, `${lane.id}: display is not the frozen join key`);
	}
	eq(
		PAPER_ROBUSTNESS_DISPLAY['paper-rf-promoter'],
		'RF 2k-d13 + Type/#PMIDs/promoter',
		'the reference keeps the paper’s own name for its own model'
	);
	eq(figure.referenceDisplay, 'RF 2k-d13 + Type/#PMIDs/promoter', 'reference display');

	// ---- shipped fields are READ, only ×100 for the drawing --------------------
	for (const lane of lanes) {
		const shipped = raw.arms.find((arm) => arm.id === lane.id);
		ok(shipped !== undefined, `${lane.id}: is a real artifact arm`);
		if (!shipped) continue;
		close(lane.pointwise.deltaPts, shipped.primary.delta * 100, 1e-12, `${lane.id}: delta is READ`);
		close(lane.pointwise.lowPts, shipped.primary.ci95_low * 100, 1e-12, `${lane.id}: ci low is READ`);
		close(
			lane.pointwise.highPts,
			shipped.primary.ci95_high * 100,
			1e-12,
			`${lane.id}: ci high is READ`
		);
		close(
			lane.simultaneous.lowPts,
			shipped.primary.simultaneous_low * 100,
			1e-12,
			`${lane.id}: simultaneous low is READ`
		);
		close(
			lane.simultaneous.highPts,
			shipped.primary.simultaneous_high * 100,
			1e-12,
			`${lane.id}: simultaneous high is READ`
		);
		close(
			lane.sensitivity.deltaPts,
			shipped.sensitivity.delta * 100,
			1e-12,
			`${lane.id}: sensitivity delta is READ`
		);
		eq(lane.pGreaterThanZero, shipped.primary.p_greater_than_zero, `${lane.id}: P(>0) is READ`);
	}

	// ---- the drawing's own invariants -----------------------------------------
	for (const lane of lanes) {
		ok(
			lane.simultaneous.lowPts <= lane.pointwise.lowPts &&
				lane.simultaneous.highPts >= lane.pointwise.highPts,
			`${lane.id}: the simultaneous band contains the pointwise interval`
		);
		eq(
			lane.simultaneousAdverse === null,
			lane.simultaneous.standing !== 'not-significant',
			`${lane.id}: the overhang block is drawn exactly when the band straddles zero`
		);
		if (lane.simultaneousAdverse) {
			ok(
				lane.simultaneousAdverse.fromPts <= 0 && lane.simultaneousAdverse.toPts >= 0,
				`${lane.id}: the overhang block touches zero`
			);
		}
	}
	for (let i = 1; i < lanes.length; i += 1) {
		ok(
			lanes[i].primaryDelta <= lanes[i - 1].primaryDelta,
			'lanes are ordered by primary delta, descending'
		);
	}
	ok(figure.domainMinPts < 0 && figure.domainMaxPts > 0, 'the zero rule sits inside the axis');
	ok(figure.ticksPts.includes(0), 'zero is a labelled tick');

	// ---- the structure the figure exists to show -------------------------------
	// Stated as arithmetic rather than prose: two arms clear zero pointwise and
	// none of them clears it simultaneously, and the single negative arm clears it
	// on BOTH views. If a rerun changes this, the page's reading changes with it.
	const winners = lanes.filter((lane) => lane.pointwise.deltaPts > 0);
	const losers = lanes.filter((lane) => lane.pointwise.deltaPts < 0);
	eq(losers.length, 1, 'exactly one arm sits below zero');
	eq(losers[0].id, figure.doseResponse.smallestArmId, 'the arm below zero is the smallest arm');
	// AND THE CLASS CARRIES THE SIGN. The arm below zero is significantly BEHIND on
	// both views — under the deleted `excludesZero` boolean it was `true` here, the
	// identical value the two winning arms carried pointwise, which is exactly how
	// "clears zero" came to be printed over a loss six times.
	eq(losers[0].pointwise.standing, 'behind', 'the negative arm stands BEHIND pointwise');
	eq(
		losers[0].simultaneous.standing,
		'behind',
		'and BEHIND under the widened band — not merely "excluding zero"'
	);
	ok(
		winners.every((lane) => lane.simultaneous.standing === 'not-significant'),
		'no winning arm clears zero under the simultaneous band'
	);
	ok(
		winners.some((lane) => lane.pointwise.standing === 'ahead'),
		'at least one winning arm stands ahead pointwise — the primary result'
	);
	// THE TWO SIGNED CLASSES ARE DISTINGUISHABLE. One assertion, and the sign-blind
	// boolean could not have made it: the losing arm and the leading arm must not
	// carry the same pointwise value.
	ok(
		losers[0].pointwise.standing !==
			winners.find((lane) => lane.pointwise.standing !== 'not-significant')?.pointwise.standing,
		'a significantly behind arm and a significantly ahead arm are different values'
	);

	// ---- power: the half-width is the same order as the effect ------------------
	ok(figure.worstHalfWidthPts > 0 && figure.bestDeltaPts > 0, 'power scalars are present');
	ok(
		figure.worstHalfWidthPts > figure.bestDeltaPts * 0.5,
		'the panel is underpowered for an effect this size (half-width ~ effect)'
	);

	// ---- panel census ----------------------------------------------------------
	eq(figure.primaryPanel.isOurLabelRevision, false, 'the 1689 panel is the paper’s own labels');
	eq(figure.sensitivityPanel.isOurLabelRevision, true, 'the 1578 panel is OUR label revision');
	eq(
		figure.primaryPanel.nStatements - figure.sensitivityPanel.nStatements,
		figure.labelCompleteness.nDropped,
		'the dropped statements account for the whole panel difference'
	);
	eq(
		figure.primaryPanel.nPositive,
		figure.sensitivityPanel.nPositive,
		'the label-completeness check drops negatives only'
	);
	ok(
		figure.labelCompleteness.negativeFractionAfter < figure.labelCompleteness.negativeFractionBefore,
		'dropping label-incomplete negatives shifts the class balance'
	);
	ok(
		figure.multiplicity.criticalValue > figure.multiplicity.pointwiseNormalCriticalValue &&
			figure.multiplicity.criticalValue <= figure.multiplicity.bonferroniCriticalValue,
		'max-t sits strictly between pointwise and Bonferroni'
	);

	// ---- label budgets are derived from the geometry, then enforced ------------
	const G = PAPER_ROBUSTNESS_GEOMETRY;
	eq(
		PAPER_ROBUSTNESS_LABEL_BUDGET_CHARS,
		Math.floor(G.labelAnchorX / G.monoUnitsPerChar),
		'the lane-label budget is the measured gutter, not a guess'
	);
	eq(
		PAPER_ROBUSTNESS_READOUT_BUDGET_CHARS,
		Math.floor((G.width - G.readoutX) / G.readoutUnitsPerChar),
		'the readout budget is the measured gutter, not a guess'
	);
	for (const lane of lanes) {
		ok(
			lane.display.length <= PAPER_ROBUSTNESS_LABEL_BUDGET_CHARS,
			`${lane.id}: lane label fits its right-anchored gutter`
		);
		ok(
			lane.readoutPrimary.length <= PAPER_ROBUSTNESS_READOUT_BUDGET_CHARS,
			`${lane.id}: primary readout fits (${lane.readoutPrimary.length} chars)`
		);
		ok(
			lane.readoutSensitivity.length <= PAPER_ROBUSTNESS_READOUT_BUDGET_CHARS,
			`${lane.id}: sensitivity readout fits (${lane.readoutSensitivity.length} chars)`
		);
	}

	// ---- every series is separable without colour ------------------------------
	const strokes = new Set();
	const dashes = new Set();
	const shapes = new Set();
	for (const id of PAPER_ROBUSTNESS_SERIES_IDS) {
		const style = PAPER_ROBUSTNESS_SERIES[id];
		strokes.add(style.strokeVar);
		dashes.add(`${style.dash}|${style.strokeWidth}`);
		shapes.add(style.shape);
		ok(style.strokeVar.startsWith('var(--'), `${id}: stroke is a token, never a raw hex`);
	}
	eq(strokes.size, PAPER_ROBUSTNESS_SERIES_IDS.length, 'no two series share a hue');
	eq(dashes.size, PAPER_ROBUSTNESS_SERIES_IDS.length, 'no two series share a stroke pattern');
	eq(shapes.size, PAPER_ROBUSTNESS_SERIES_IDS.length, 'no two series share a mark shape');
}

// ---- fail-closed: each mutation must gate, not render ------------------------
let mutationCases = 0;
function mutated(mutate, label) {
	mutationCases += 1;
	const copy = JSON.parse(JSON.stringify(raw));
	mutate(copy);
	const result = validatePaperRobustness(copy, { artifactPath: 'fixture', artifactSha256: 'x' });
	eq(result.status, 'unavailable', `gates: ${label}`);
}

mutated((d) => {
	d.artifact_kind = 'something_else';
}, 'a different artifact kind');
mutated((d) => delete d.arms, 'no arms at all');
mutated((d) => d.arms.pop(), 'an arm disappearing from the family');
mutated((d) => {
	d.arms[0].id = 'gemma-4-42b';
}, 'an arm id with no canonical display name');
mutated((d) => {
	d.arms[0].label = 'Gemma 4 26B v2';
}, 'the frozen point_metrics join key drifting');

// The two framing flips. Either one, unchecked, would let a family-wise band or
// our own label revision be read as the primary result.
mutated((d) => {
	d.panels.primary.is_our_label_revision = true;
}, 'the paper-label panel being relabelled as our revision');
mutated((d) => {
	d.panels.sensitivity.is_our_label_revision = false;
}, 'our label revision being relabelled as the paper’s data');

mutated((d) => {
	d.multiplicity.critical_value = 1.5;
}, 'a simultaneous band narrower than pointwise');
mutated((d) => {
	d.multiplicity.critical_value = 3.4;
}, 'a max-t critical value above Bonferroni');
mutated((d) => {
	d.multiplicity.no_designated_primary_arm = false;
}, 'a plan that did designate an arm (the correction would need re-deriving)');

mutated((d) => {
	d.shipped_reconciliation.worst_residual_vs_shipped = 1e-3;
}, 'the pointwise half no longer reproducing the shipped head-to-head');

mutated((d) => {
	d.arms[0].primary.simultaneous_low = d.arms[0].primary.ci95_low + 1e-6;
}, 'a simultaneous band that does not contain its own pointwise interval');
mutated((d) => {
	d.arms[0].primary.excludes_zero_pointwise = !d.arms[0].primary.excludes_zero_pointwise;
}, 'an excludes-zero flag that disagrees with its own interval');
mutated((d) => {
	d.arms[0].primary.excludes_zero_simultaneous = !d.arms[0].primary.excludes_zero_simultaneous;
}, 'a simultaneous excludes-zero flag that disagrees with its own band');
mutated((d) => {
	d.arms[0].primary.ci95_low = d.arms[0].primary.ci95_high + 0.01;
}, 'an inverted interval');

mutated((d) => {
	d.label_completeness.n_dropped = 7;
}, 'a dropped-statement count that does not reconcile with the panels');
mutated((d) => {
	d.label_completeness.all_dropped_are_negative = false;
}, 'dropped statements that are no longer all negatives');
mutated((d) => {
	d.label_completeness.no_model_is_refit = false;
}, 'a sensitivity panel that refit a model');
mutated((d) => {
	d.panels.sensitivity.n_positive -= 1;
	d.panels.sensitivity.n_statements -= 1;
}, 'a label-completeness check that dropped a positive');

mutated((d) => {
	d.arms.reverse();
}, 'arms arriving in an order that would put a losing arm in the headline sentence');
mutated((d) => {
	d.dose_response.n_arms_with_negative_delta = 2;
}, 'a dose-response census that disagrees with the drawn arms');
mutated((d) => {
	d.dose_response.smallest_arm_id = 'glm-5';
}, 'the negative arm no longer being the smallest arm');

// ---- the component must not soften or reframe what it draws ------------------
const component = readFileSync(
	new URL('../src/lib/components/PaperRobustness.svelte', import.meta.url),
	'utf8'
);
ok(
	!/\{lane\.label\}/.test(component),
	'the component never renders the frozen join key — it renders `display`'
);
ok(
	/our label revision/i.test(component),
	'the component says in words that the smaller panel is OUR label revision'
);
ok(
	component.indexOf('primaryPanel.nStatements') < component.indexOf('sensitivityPanel.nStatements'),
	'the primary panel is named before the sensitivity panel'
);
ok(
	/primary result stands/i.test(component),
	'the component restates the primary result before either robustness view'
);
ok(
	!/\brecomputed?\b/i.test(component.replace(/never recomputed?|re-?measur\w*/gi, '')),
	'the component reads shipped fields rather than recomputing them'
);

// Reported so a runner that silently stopped mutating cannot still exit 0.
console.log(`${mutationCases} fail-closed mutation cases exercised`);
console.log(
	failures === 0
		? 'paper-robustness data contract assertions passed'
		: `${failures} paper-robustness contract assertion(s) failed`
);
process.exit(failures === 0 ? 0 : 1);
