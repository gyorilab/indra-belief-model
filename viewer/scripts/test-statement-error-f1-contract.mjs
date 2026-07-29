/**
 * Pure-function assertions for the STATEMENT-GRAIN ERROR-F1 data contract.
 *
 * Same shape as `test-tie-inflation-contract.mjs` and
 * `test-belief-ladder-contract.mjs`: eq/ok counters, the REAL shipped artifact
 * exercised through the loader, then a list of mutations that must each gate the
 * figure to `unavailable` rather than draw a wrong number — with the mutation
 * count printed so a runner that silently stopped mutating cannot still exit 0.
 *
 * WHAT THIS FILE IS FOR. `viewer/src/lib/data/paper-error-f1.ts` already gates
 * hard; nothing here re-implements its arithmetic. This runner PINS those gates
 * so they cannot regress, and pins the numbers the page argues from. The figure
 * names the one margin that survives multiplicity correction on the 2023 paper's
 * own panel, so a number that drifted here would corrupt the claim this whole
 * surface exists to make honestly — and a gate that quietly stopped firing would
 * let it drift in silence. Every gate below has already shipped broken once:
 *
 *   · the headline delta was ungated against the two error-F1 values panel A
 *     draws, and a +0.30 with a matching interval rendered beside the 0.7855 and
 *     0.6439 that refute it;
 *   · `error_precision := 0.95` with tp/fp/fn untouched rendered "the winning
 *     gates 75%–95%" and put the F1 tick outside its own precision–recall span;
 *   · deleting the out-of-family contender returned `ok` and drew five lanes,
 *     removing a DISCLOSED LOSS from the page.
 *
 * EVERY ASSERTION HERE MUST BE ABLE TO FAIL. The first revision of this runner
 * carried a "READ, not recomputed" block that compared each rendered field against
 * the shipped JSON float and claimed that "a loader that started deriving any of
 * these would drift in the last decimal and fail here". It would not. The loader's
 * own docblock records that its identities hold at residual 0.000e+00, so on the
 * REAL artifact a derived value and the shipped value are the same double: the
 * comparison passes either way. A reviewer disabled loader gates one at a time and
 * this runner stayed green. That block read as coverage and was not coverage —
 * worse than nothing, and the same failure mode as the contamination guard that
 * once shipped green while reading a field that did not exist.
 *
 * The block is now split into the two things it was conflating:
 *
 *   (a) A MAPPING PIN — the byte-identity loop. It proves the loader put each
 *       shipped field in the field the figure draws it from (no swapped precision
 *       and recall, no lane reading another lane's row). It cannot prove READ-ness
 *       and no longer claims to.
 *   (b) A FALSIFIABILITY PROBE TABLE — `probeReadField`, below the mutation list.
 *       Each field is perturbed ON ITS OWN and its outcome is DECLARED in advance
 *       as `gated` (with the reason) or `read-through`. A declared-gated field
 *       whose loader gate is deleted comes back `ok` and FAILS here; a declared
 *       read-through field the loader starts deriving comes back with the derived
 *       value rather than the perturbed one and FAILS here. Both directions of
 *       must-assert item 2 are decidable this way, and each probe names a specific
 *       loader gate that a reviewer can delete to watch this runner go red.
 *
 * WHERE READ-NESS IS NOT DECIDABLE, AND WHY THAT IS THE RIGHT ANSWER. For a field
 * pinned by an identity gate — `error_f1` against its own tp/fp/fn, `delta_error_f1`
 * against the two drawn F1s — "read" and "recomputed" are observationally
 * identical BY CONSTRUCTION: the gate is exactly the statement that the two agree.
 * There is no artifact that distinguishes them, so no test can. What is decidable,
 * and what the probe asserts, is that the gate FIRES. That is the stronger property
 * anyway: it holds for every future artifact, not just this one.
 *
 * SIGN-BLINDNESS IS THE REPEAT DEFECT, so it gets two cases the runner expects to
 * SURVIVE rather than gate. The `excludesZero` boolean that caused it — `ciLow > 0
 * || ciHigh < 0`, blind to sign by construction — has since been deleted from
 * every loader on /paper, but four separate clauses on this page branched on it
 * two-way first and called a loss a win or a tie. A mutation that pushes an arm
 * genuinely BELOW the reference must load `ok` and come back classified `behind`;
 * a runner that only checked "broken things gate" would pass while every loss
 * rendered as a tie. Those two cases are counted and reported separately.
 *
 * NO RENDERED SENTENCE IS PINNED. A prior test pinned exact prose and a
 * legitimate trim read as a regression. Intent only: which class the loader
 * assigns, which numbers it carries, which mutations take the figure down.
 *
 * Label gutters, geometry and the page-wide render invariants belong to
 * `test-paper-render-invariants.mjs`, which already sweeps this component; they
 * are deliberately not duplicated here.
 */
import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';

import {
	STATEMENT_ERROR_F1_ARTIFACT_KIND,
	STATEMENT_ERROR_F1_CONTENDER_ARM_IDS,
	STATEMENT_ERROR_F1_DRAWN_ARM_IDS,
	STATEMENT_ERROR_F1_PARITY_TOL,
	fmt4,
	fmtDelta,
	validateStatementErrorF1
} from '../src/lib/data/paper-error-f1.ts';
import { PAPER_LITERAL_ARM_SPECS, PAPER_LITERAL_REFERENCE_ARM_ID } from '../src/lib/data/paper-literal.ts';

let failures = 0;

function ok(condition, label) {
	if (!condition) {
		failures += 1;
		console.error(`FAIL ${label}`);
	}
}

function eq(got, want, label) {
	if (Object.is(got, want)) return;
	failures += 1;
	console.error(`FAIL ${label}: got ${JSON.stringify(got)}, want ${JSON.stringify(want)}`);
}

const MODEL_DIR = new URL('../../data/results/indra_paper_literal_models_20260724/', import.meta.url);
const ARTIFACT_NAME = 'statement_error_f1.json';
const artifactBytes = readFileSync(new URL(ARTIFACT_NAME, MODEL_DIR));
const raw = JSON.parse(artifactBytes.toString('utf8'));
const manifest = JSON.parse(readFileSync(new URL('manifest.json', MODEL_DIR), 'utf8'));
const sibling = JSON.parse(readFileSync(new URL('paper_literal_vs_llms.json', MODEL_DIR), 'utf8'));

/** The shipped bytes are the bytes the run signed. */
eq(
	createHash('sha256').update(artifactBytes).digest('hex'),
	manifest.output_sha256[ARTIFACT_NAME],
	'error-F1 artifact sha256 matches the run manifest'
);
eq(manifest.outputs.statement_error_f1, ARTIFACT_NAME, 'the manifest names the error-F1 output');
eq(raw.artifact_kind, STATEMENT_ERROR_F1_ARTIFACT_KIND, 'the shipped artifact declares its kind');

// ---------------------------------------------------------------------------
// 1. THE REAL SHIPPED ARTIFACT VALIDATES.
// ---------------------------------------------------------------------------
const live = validateStatementErrorF1(raw, {
	artifactPath: ARTIFACT_NAME,
	artifactSha256: 'deadbeef'
});
eq(live.status, 'ok', `the shipped artifact validates: ${live.reason ?? ''}`);

/** Raw arm record by canonical id, for the mapping pin below. */
function rawArm(payload, id) {
	if (id === PAPER_LITERAL_REFERENCE_ARM_ID) return payload.reference;
	const found = payload.arms.find((entry) => entry.key === id);
	if (!found) throw new Error(`no raw arm ${id}`);
	return found;
}

/** Every headline number this figure argues from, at the precision it is argued at. */
const EXPECTED_ERROR_F1 = {
	'paper-rf-promoter': '0.6439',
	'glm-5': '0.7855',
	'gemma-4-26b': '0.7746',
	'gemma-4-31b': '0.7699',
	'gemma-4-e2b': '0.6345'
};
/** The margins, and the intervals that decide them, from the GOAL's verified set. */
const EXPECTED_DELTA = {
	'glm-5': ['+0.1416', '+0.1090', '+0.1744'],
	'gemma-4-26b': ['+0.1307', '+0.0955', '+0.1655'],
	'gemma-4-31b': ['+0.1260', '+0.0912', '+0.1609'],
	'gemma-4-e2b': ['-0.0094', '-0.0381', '+0.0194']
};
/** The three arms whose SIMULTANEOUS band clears zero on the winning side. */
const EXPECTED_WINNERS = ['glm-5', 'gemma-4-26b', 'gemma-4-31b'];

/**
 * THE FIELDS THE FIGURE PRESENTS AS SHIPPED, as [rendered field, artifact key].
 *
 * Module scope on purpose: these lists feed BOTH halves of must-assert item 2 —
 * the mapping pin inside the live block below, and the falsifiability probe table
 * near the end of this file. The probe table requires a declared expectation for
 * every key in these lists and fails on any it does not recognise, so a field
 * added here cannot be silently left unprobed.
 */
const READ_POINT_FIELDS = [
	['tau', 'tau'],
	['flagged', 'flagged'],
	['errorPrecision', 'error_precision'],
	['errorRecall', 'error_recall'],
	['errorF1', 'error_f1'],
	['tp', 'tp'],
	['fp', 'fp'],
	['fn', 'fn'],
	['tn', 'tn'],
	['accuracy', 'accuracy'],
	['correctPrecision', 'correct_precision'],
	['correctRecall', 'correct_recall'],
	['correctF1', 'correct_f1'],
	['flagSetIsTheArmsZeroPile', 'flag_set_is_the_arms_zero_pile']
];
const READ_DELTA_FIELDS = [
	['delta', 'delta_error_f1'],
	['bootstrapMean', 'delta_bootstrap_mean'],
	['ciLow', 'ci95_low'],
	['ciHigh', 'ci95_high'],
	['bootstrapSe', 'bootstrap_se'],
	['tStatistic', 't_statistic'],
	['pGreaterThanZero', 'p_greater_than_zero'],
	['nValidResamples', 'n_valid_resamples'],
	['simLow', 'simultaneous_low'],
	['simHigh', 'simultaneous_high']
	// `excludes_zero_pointwise` / `excludes_zero_simultaneous` are still READ and
	// still gated against their own endpoints (see the mutation probes below), but
	// they are no longer CARRIED: a sign-blind boolean on a loader's output is the
	// shape that produced six directional regressions, so the loader emits a
	// three-way class instead and the flags stop at the validator.
];
const READ_MATCHED_FIELDS = [
	['targetErrorRecall', 'target_error_recall'],
	['referenceErrorRecallAtThisRow', 'reference_error_recall_at_this_row'],
	['referenceErrorF1AtThisRow', 'reference_error_f1_at_this_row'],
	['referenceRecallOvershoot', 'reference_recall_overshoot'],
	['deltaAtMatchedRecall', 'delta_error_f1_at_matched_recall'],
	['deltaEachSideAtItsOwnTargetCut', 'delta_error_f1_each_side_at_its_own_target_cut']
];

/**
 * A readout is pinned for the NUMBER it carries, never for its wording. An
 * earlier test in this repo pinned an exact rendered sentence and a legitimate
 * trim read as a regression; the intent is that the readout shows the shipped
 * value at the precision the page argues it at, and shows no OTHER number beside
 * it. Signed and unsigned four-decimal forms both match, so panel A's "F1 0.7855"
 * and panel B's "+0.1416" go through the same check.
 */
function readoutNumbers(rendered) {
	return rendered.match(/[+-]?\d+\.\d{4}/g) ?? [];
}

if (live.status === 'ok') {
	const figure = live.figure;
	const lanes = figure.lanes;
	const laneOf = (id) => lanes.find((lane) => lane.id === id);

	// -- the drawn set is exactly the frozen one, once each -------------------
	eq(lanes.length, STATEMENT_ERROR_F1_DRAWN_ARM_IDS.length, 'every frozen lane is drawn');
	eq(
		[...lanes.map((lane) => lane.id)].sort().join(','),
		[...STATEMENT_ERROR_F1_DRAWN_ARM_IDS].sort().join(','),
		'the drawn lanes are exactly the frozen drawn set'
	);
	eq(new Set(lanes.map((lane) => lane.id)).size, lanes.length, 'no lane is drawn twice');
	eq(figure.reference.id, PAPER_LITERAL_REFERENCE_ARM_ID, 'the reference lane is the paper RF');
	eq(figure.reference.delta, null, 'the reference carries no margin against itself');

	// -- 3. POINT VALUES ------------------------------------------------------
	for (const [id, want] of Object.entries(EXPECTED_ERROR_F1)) {
		const lane = laneOf(id);
		ok(lane !== undefined, `${id}: is drawn`);
		if (!lane) continue;
		eq(fmt4(lane.operating.errorF1), want, `${id}: shipped error-class F1`);
		// The rendered readout carries that same number and no other. Its WORDING
		// is not pinned: a trim of the "F1 " prefix is a legitimate edit, a
		// different number in the readout is not.
		eq(
			readoutNumbers(lane.readoutA).join(','),
			want,
			`${id}: panel-A readout carries the shipped F1, and only it`
		);
	}
	eq(laneOf('indra-cogex-hybrid') !== undefined, true, 'the out-of-family control is drawn');

	for (const [id, [delta, low, high]] of Object.entries(EXPECTED_DELTA)) {
		const lane = laneOf(id);
		if (!lane || !lane.delta) {
			failures += 1;
			console.error(`FAIL ${id}: has no margin`);
			continue;
		}
		eq(fmtDelta(lane.delta.delta), delta, `${id}: shipped margin against the paper RF`);
		eq(fmtDelta(lane.delta.ciLow), low, `${id}: pointwise 95% lower bound`);
		eq(fmtDelta(lane.delta.ciHigh), high, `${id}: pointwise 95% upper bound`);
		eq(
			readoutNumbers(lane.readoutB).join(','),
			delta,
			`${id}: panel-B readout carries the shipped margin, signed, and only it`
		);
	}
	// The reference's panel-B readout must carry NO margin — it is the baseline,
	// and a number there would read as a margin against itself. The intent is
	// "names it, quotes nothing"; the words it uses to name it are free.
	ok(
		figure.reference.readoutB.trim().length > 0,
		'the reference readout names the reference rather than sitting empty'
	);
	eq(
		readoutNumbers(figure.reference.readoutB).length,
		0,
		'and quotes no margin — the reference has none against itself'
	);
	eq(figure.multiplicity.criticalValue.toFixed(4), '2.3684', 'the max-t critical value is pinned');
	eq(figure.multiplicity.familySize, 4, 'the max-t family is the four reader gates');
	eq(figure.panel.n, 1689, 'the panel is the paper’s own 1,689 statements');
	eq(figure.panel.nErrors, 452, 'the panel carries its own error count');
	eq(figure.panel.nErrors + figure.panel.nCorrect, figure.panel.n, 'the panel closes');

	// -- 2a. THE MAPPING PIN, over every lane ---------------------------------
	// Byte identity against the shipped JSON floats. What this proves is that each
	// shipped field arrives in the field the figure DRAWS it from: precision not
	// swapped with recall, the matched cut not read off the headline cut, no lane
	// carrying another lane's row. It does NOT prove the value was read rather
	// than derived, and no longer says it does — the loader's identities hold at
	// residual 0.000e+00 on this artifact, so a derived value IS the shipped
	// double and this loop cannot tell the two apart. READ-ness, and the gates
	// that make it moot where it is undecidable, are the probe table's job; see
	// "READ-NESS AND ITS GATES" near the end of this file. These three lists are
	// its input, so a field added here is probed there or the runner fails.
	for (const lane of lanes) {
		const shipped = rawArm(raw, lane.id);
		eq(lane.distinctScores, shipped.distinct_scores, `${lane.id}: distinct_scores is drawn as shipped`);
		for (const [field, key] of READ_POINT_FIELDS) {
			eq(lane.operating[field], shipped.operating_point[key], `${lane.id}: ${key} is READ`);
			eq(
				lane.matched.point[field],
				shipped.matched_recall[key],
				`${lane.id}: matched ${key} is READ`
			);
			eq(
				lane.matched.referenceAtThisRowsRecall[field],
				shipped.matched_recall.reference_at_this_rows_recall[key],
				`${lane.id}: re-cut reference ${key} is READ`
			);
		}
		for (const [field, key] of READ_MATCHED_FIELDS) {
			eq(lane.matched[field], shipped.matched_recall[key], `${lane.id}: ${key} is READ`);
		}
		if (lane.delta) {
			for (const [field, key] of READ_DELTA_FIELDS) {
				eq(lane.delta[field], shipped[key], `${lane.id}: ${key} is READ`);
			}
		}
	}
	// The censuses the prose reads off are counted over those SHIPPED values.
	eq(
		figure.nWinsSimultaneously,
		raw.arms.filter((arm) => arm.excludes_zero_simultaneous === true && arm.delta_error_f1 > 0)
			.length,
		'the simultaneous-win census counts the shipped arms'
	);
	eq(
		figure.nWinsPointwise,
		raw.arms.filter((arm) => arm.excludes_zero_pointwise === true && arm.delta_error_f1 > 0).length,
		'the pointwise-win census counts the shipped arms'
	);
	eq(
		figure.multiplicity.nExcludingZeroSimultaneous,
		raw.multiplicity.n_excluding_zero_simultaneous,
		'the shipped exclusion count is READ'
	);

	// -- lanes are rank-interleaved, so the reference sits among the arms -----
	ok(
		lanes.every(
			(lane, index) => index === 0 || lanes[index - 1].operating.errorF1 >= lane.operating.errorF1
		),
		'lanes are ordered by shipped error-F1, descending'
	);
	const referenceRank = lanes.findIndex((lane) => lane.isReference);
	eq(referenceRank, 3, 'three arms rank above the paper RF and two below it');

	// -- 4. SIGN / DIRECTION: the disclosed loss is not marked as a win -------
	eq(figure.nWinsSimultaneously, 3, 'exactly three arms win under the family-wide band');
	eq(
		lanes
			.filter((lane) => lane.delta?.winsSimultaneously)
			.map((lane) => lane.id)
			.sort()
			.join(','),
		[...EXPECTED_WINNERS].sort().join(','),
		'the three winners are the three larger reader gates'
	);
	const e2b = laneOf('gemma-4-e2b');
	if (e2b?.delta) {
		ok(e2b.delta.delta < 0, 'Gemma 4 E2B’s margin is NEGATIVE');
		ok(e2b.delta.ciLow < 0 && e2b.delta.ciHigh > 0, 'its pointwise interval SPANS zero');
		ok(
			e2b.delta.simLow !== null &&
				e2b.delta.simHigh !== null &&
				e2b.delta.simLow < 0 &&
				e2b.delta.simHigh > 0,
			'its simultaneous band spans zero too'
		);
		eq(
			e2b.delta.simultaneousStanding,
			'not-significant',
			'E2B’s band is classed not-significant — it does not clear zero either way'
		);
		eq(e2b.delta.winsSimultaneously, false, 'E2B is NOT marked a simultaneous win');
		eq(e2b.delta.winsPointwise, false, 'E2B is NOT marked a pointwise win');
		eq(e2b.delta.standing, 'not-significant', 'E2B is classed not-significant, not behind');
		eq(e2b.delta.pointwiseStanding, 'not-significant', 'and the same pointwise');
		eq(e2b.delta.standingBasis, 'simultaneous', 'an in-family arm is judged on the band');
		eq(e2b.delta.standingLow, e2b.delta.simLow, 'the printed low bound is the deciding one');
		eq(e2b.delta.standingHigh, e2b.delta.simHigh, 'the printed high bound is the deciding one');
	} else {
		failures += 1;
		console.error('FAIL Gemma 4 E2B has no margin');
	}
	const cogex = laneOf('indra-cogex-hybrid');
	if (cogex?.delta) {
		eq(cogex.delta.simLow, null, 'the out-of-family control carries no band');
		eq(
			cogex.delta.simultaneousStanding,
			null,
			'and its band class is null, not not-significant: no band was measured'
		);
		eq(cogex.delta.standingBasis, 'pointwise', 'so it is judged pointwise');
		ok(cogex.delta.delta < 0, 'the out-of-family control is also behind on the point estimate');
		eq(cogex.delta.winsPointwise, false, 'and is NOT marked a win');
	} else {
		failures += 1;
		console.error('FAIL the out-of-family control has no margin');
	}
	// The invariant behind all of it, asserted over every lane rather than at the
	// one arm that exposed the defect: no lane can be a win with a margin <= 0,
	// and no lane can be `ahead` on an interval that is not strictly above zero.
	for (const lane of lanes) {
		if (!lane.delta) continue;
		const d = lane.delta;
		ok(!d.winsSimultaneously || d.delta > 0, `${lane.id}: a simultaneous win has a positive margin`);
		ok(!d.winsPointwise || d.delta > 0, `${lane.id}: a pointwise win has a positive margin`);
		ok(d.standing !== 'ahead' || (d.delta > 0 && d.standingLow > 0), `${lane.id}: ahead means above zero`);
		ok(d.standing !== 'behind' || (d.delta < 0 && d.standingHigh < 0), `${lane.id}: behind means below zero`);
		ok(
			d.standing !== 'not-significant' || (d.standingLow <= 0 && d.standingHigh >= 0),
			`${lane.id}: not-significant means the deciding interval contains zero`
		);
		ok(
			d.pointwiseStanding !== 'ahead' || (d.delta > 0 && d.ciLow > 0),
			`${lane.id}: pointwise ahead means the pointwise interval is above zero`
		);
		ok(
			d.pointwiseStanding !== 'behind' || (d.delta < 0 && d.ciHigh < 0),
			`${lane.id}: pointwise behind means the pointwise interval is below zero`
		);
		ok(d.delta >= d.ciLow && d.delta <= d.ciHigh, `${lane.id}: the marker sits inside its own bar`);
	}

	// -- 5. THE ORACLE DISCLOSURE TRAVELS WITH EVERY THRESHOLD-BASED NUMBER ---
	const rules = [
		figure.headlineThresholdRule,
		figure.matchedThresholdRule,
		figure.reconciliation.thresholdRule
	];
	eq(
		rules.map((rule) => rule.id).join(','),
		'best-f1,target-recall-60,review-queue',
		'all three tau rules reach the figure, each under its own id'
	);
	for (const rule of rules) {
		ok(rule.oracle.trim().length > 0, `${rule.id}: carries a non-empty oracle disclosure`);
		ok(rule.rule.trim().length > 0, `${rule.id}: carries a non-empty threshold rule`);
		ok(rule.name.trim().length > 0, `${rule.id}: is named on screen`);
	}
	// The oracle FAVOURS the paper's model, and the disclosure is only meaningful
	// if that is checkable: their search had far more candidate cuts than ours.
	const winnerCuts = lanes
		.filter((lane) => lane.delta?.winsSimultaneously)
		.map((lane) => lane.distinctScores);
	ok(
		figure.reference.distinctScores > Math.max(...winnerCuts),
		`the reference had more candidate cuts (${figure.reference.distinctScores}) than any winner (${Math.max(...winnerCuts)})`
	);
	ok(figure.caveats.length > 0, 'the figure carries its caveat list');

	// -- 6. JOIN KEYS ARE REAL ARTIFACT KEYS, DECOUPLED FROM WHAT IS DRAWN ----
	const siblingKeys = new Set(Object.keys(sibling.point_metrics));
	let distinctDisplays = 0;
	const coincident = [];
	for (const lane of lanes) {
		const spec = PAPER_LITERAL_ARM_SPECS.find((candidate) => candidate.id === lane.id);
		ok(spec !== undefined, `${lane.id}: is a canonical arm`);
		if (!spec) continue;
		eq(lane.label, spec.label, `${lane.id}: the join key is the canonical frozen label`);
		eq(lane.display, spec.display, `${lane.id}: the on-screen name is the canonical display`);
		ok(siblingKeys.has(lane.label), `${lane.id}: "${lane.label}" is a real point_metrics key`);
		if (lane.display === lane.label) coincident.push(lane.id);
		else distinctDisplays += 1;
	}
	// The invariant is that `label` (join key) and `display` (screen) are separate
	// fields, not that the two strings always differ: one canonical arm is named
	// the same on both sides. Pinning the census rather than asserting inequality
	// keeps the real rule enforced — if a second arm ever collapsed its display
	// onto its join key, this moves.
	eq(distinctDisplays, 5, 'five of the six lanes render a name that is not their join key');
	eq(coincident.join(','), 'indra-cogex-hybrid', 'exactly one canonical arm names both alike');
	// The reference's rendered name is emphatically not its frozen key.
	ok(
		figure.reference.display !== figure.reference.label,
		'the reference renders its method name, never its point_metrics key'
	);
	// Reconciliation joins on a THIRD key space; it must be the drawn lane's own.
	for (const row of figure.reconciliation.rows) {
		const lane = laneOf(row.id);
		ok(lane !== undefined, `reconciliation row ${row.id} is a drawn lane`);
		if (!lane) continue;
		eq(row.reviewQueueModelKey, lane.reviewQueueModelKey, `${row.id}: review-queue key is the lane’s`);
		eq(row.display, lane.display, `${row.id}: the row renders the lane’s display name`);
		eq(row.thisArtifactErrorF1, lane.operating.errorF1, `${row.id}: reconciled F1 is the drawn F1`);
	}
	ok(
		figure.reconciliation.worstResidual <= figure.reconciliation.tolerance,
		'the review-queue cross-check holds on the shipped artifact'
	);
	eq(figure.reconciliation.panelN, figure.panel.n, 'the review queue was scored on this panel');

	// -- the matched-recall block is named for what it holds ------------------
	// The unmatched column flipped E2B's sign; the artifact ships both and the
	// loader derives the disagreement, so pin that it is still detected.
	eq(figure.signFlipDisplays.length, 1, 'exactly one arm’s two matched deltas disagree in sign');
	// WHICH arm, not what it is called: the display string is pinned to the
	// canonical spec table a few blocks down, so naming it twice would only mean a
	// legitimate rename reads as a regression here.
	eq(
		figure.signFlipDisplays[0],
		laneOf('gemma-4-e2b')?.display,
		'and it is the smallest gate, identified by its lane rather than by its name'
	);
	if (e2b) {
		ok(e2b.matched.deltaAtMatchedRecall < 0, 'E2B’s recall-MATCHED delta is negative');
		ok(e2b.matched.deltaEachSideAtItsOwnTargetCut > 0, 'its unmatched delta is positive');
		eq(e2b.matched.signsDisagree, true, 'and the disagreement is derived, not narrated');
	}
	// The three-list split exists because one arm's two values are EQUAL.
	eq(
		figure.unmatchedLargerDisplays.length +
			figure.unmatchedEqualDisplays.length +
			figure.unmatchedSmallerDisplays.length,
		lanes.length - 1,
		'every contender lands in exactly one matched-vs-unmatched class'
	);
	ok(figure.unmatchedEqualDisplays.length > 0, 'the EQUAL class is non-empty, as the prose needs');
}

// ---------------------------------------------------------------------------
// FAIL-CLOSED. Each mutation must gate the figure — and gate on the RIGHT thing.
// A mutation that gates for an incidental reason is a false pass, so the reason
// is checked too.
// ---------------------------------------------------------------------------
let mutationCases = 0;

function gates(mutate, label, expected) {
	mutationCases += 1;
	const copy = structuredClone(raw);
	mutate(copy);
	const result = validateStatementErrorF1(copy, { artifactPath: 'fixture', artifactSha256: 'x' });
	if (result.status !== 'unavailable') {
		failures += 1;
		console.error(`FAIL gates: ${label} — the figure still rendered`);
		return;
	}
	if (expected !== undefined && !result.reason.includes(expected)) {
		failures += 1;
		console.error(
			`FAIL gates: ${label} — gated on the wrong thing: ${result.reason} (wanted "${expected}")`
		);
	}
}

/** The raw arm record inside a mutable copy. */
function armIn(copy, id) {
	return copy.arms.find((entry) => entry.key === id);
}

/**
 * A self-consistent confusion table at the given tp/fp, rates and all, so a
 * mutation can move an operating point WITHOUT tripping the rate-identity gate
 * — which is what makes the delta-identity gate testable on its own.
 */
function recut(point, tp, fp, n, nErrors) {
	const fn = nErrors - tp;
	const tn = n - tp - fp - fn;
	const precision = tp + fp > 0 ? tp / (tp + fp) : 0;
	const recall = tp + fn > 0 ? tp / (tp + fn) : 0;
	const f1 = precision + recall > 0 ? (2 * precision * recall) / (precision + recall) : 0;
	const cPrecision = tn + fn > 0 ? tn / (tn + fn) : 0;
	const cRecall = tn + fp > 0 ? tn / (tn + fp) : 0;
	const cF1 = cPrecision + cRecall > 0 ? (2 * cPrecision * cRecall) / (cPrecision + cRecall) : 0;
	return {
		...point,
		flagged: tp + fp,
		tp,
		fp,
		fn,
		tn,
		error_precision: precision,
		error_recall: recall,
		error_f1: f1,
		accuracy: (tp + tn) / n,
		correct_precision: cPrecision,
		correct_recall: cRecall,
		correct_f1: cF1
	};
}

// -- the drawn set: nothing may vanish, double, or sneak in ------------------
gates(
	(d) => {
		d.arms = d.arms.filter((arm) => arm.key !== 'indra-cogex-hybrid');
	},
	'the out-of-family contender deleted — a DISCLOSED LOSS removed from the page',
	// The whole clause, not the bare id: the id also appears in the family-member
	// and reconciliation reasons, so a fragment match could pass for the wrong gate.
	'the figure must draw "indra-cogex-hybrid"; it is missing'
);
gates(
	(d) => {
		d.arms = d.arms.filter((arm) => arm.key !== 'glm-5');
	},
	'a family member deleted',
	'the figure must draw "glm-5"; it is missing'
);
gates(
	(d) => {
		d.arms.push(structuredClone(armIn(d, 'glm-5')));
	},
	'the same arm drawn twice',
	'more than once'
);
gates(
	(d) => {
		const extra = structuredClone(armIn(d, 'glm-5'));
		extra.key = 'port-rf-promoter';
		extra.label = 'Paper semantic port RF+promoter';
		extra.display = 'Our port of RF + Type/#PMIDs/promoter';
		extra.in_max_t_family = false;
		extra.simultaneous_low = null;
		extra.simultaneous_high = null;
		extra.excludes_zero_simultaneous = null;
		d.arms.push(extra);
	},
	'an arm this figure does not draw appearing in the artifact',
	"is not one of this figure's contenders"
);

// -- intervals ---------------------------------------------------------------
gates((d) => delete armIn(d, 'glm-5').ci95_low, 'a missing interval bound', 'ci95_low');
gates(
	(d) => {
		const arm = armIn(d, 'glm-5');
		const low = arm.ci95_low;
		arm.ci95_low = arm.ci95_high;
		arm.ci95_high = low;
	},
	'an inverted interval (low above high)',
	'ci95_low must not exceed'
);
gates(
	(d) => {
		const arm = armIn(d, 'glm-5');
		arm.ci95_low = arm.delta_error_f1 + 0.01;
		arm.ci95_high = arm.delta_error_f1 + 0.02;
	},
	'a point estimate outside its own interval',
	'lies outside its own 95% interval'
);
gates(
	(d) => {
		armIn(d, 'glm-5').excludes_zero_pointwise = false;
	},
	'excludes_zero_pointwise contradicting its own endpoints',
	'excludes_zero_pointwise'
);
gates(
	(d) => {
		armIn(d, 'glm-5').excludes_zero_simultaneous = false;
	},
	'excludes_zero_simultaneous contradicting its own endpoints',
	'excludes_zero_simultaneous'
);
gates(
	(d) => {
		armIn(d, 'indra-cogex-hybrid').excludes_zero_simultaneous = false;
	},
	'an out-of-family arm reporting FALSE where it must report null',
	'must be null outside the max-t family'
);
gates(
	(d) => {
		armIn(d, 'glm-5').simultaneous_low = armIn(d, 'glm-5').ci95_low + 0.001;
	},
	'a simultaneous band narrower than the pointwise interval it must contain',
	'must contain the pointwise interval'
);

// -- the rate identity: rates must be the rates of the counts drawn beside them
gates(
	(d) => {
		armIn(d, 'glm-5').operating_point.error_precision = 0.95;
	},
	'error_precision := 0.95 with tp/fp/fn untouched',
	'operating_point.error_precision'
);
gates(
	(d) => {
		armIn(d, 'glm-5').operating_point.error_recall = 0.99;
	},
	'error_recall := 0.99 with the counts untouched',
	'operating_point.error_recall'
);
gates(
	(d) => {
		armIn(d, 'glm-5').operating_point.error_f1 = 0.99;
	},
	'error_f1 := 0.99 with the counts untouched',
	'operating_point.error_f1'
);
gates(
	(d) => {
		d.reference.operating_point.error_precision = 0.95;
	},
	'the REFERENCE rate detached from its own counts',
	'reference.operating_point.error_precision'
);
gates(
	(d) => {
		// tn alone: the flag set and the error count both still close, so this
		// reaches the coverage gate rather than an earlier one.
		armIn(d, 'glm-5').operating_point.tn += 1;
	},
	'a confusion cell that no longer covers the panel',
	'cover the panel'
);
gates(
	(d) => {
		armIn(d, 'glm-5').operating_point.flagged += 1;
	},
	'a flag set that is not tp + fp',
	'flagged must equal tp + fp'
);
gates(
	(d) => {
		const point = armIn(d, 'glm-5').operating_point;
		point.tp += 1;
		point.flagged += 1;
	},
	'a true-positive count that no longer closes on the panel’s errors',
	'tp + fn must equal'
);

// -- the headline identity: delta IS the difference of the two drawn F1s -----
gates(
	(d) => {
		const arm = armIn(d, 'glm-5');
		arm.delta_error_f1 = 0.3;
		arm.delta_bootstrap_mean = 0.3;
		arm.ci95_low = 0.27;
		arm.ci95_high = 0.33;
		arm.simultaneous_low = 0.26;
		arm.simultaneous_high = 0.34;
		arm.excludes_zero_pointwise = true;
		arm.excludes_zero_simultaneous = true;
	},
	'delta := +0.30 with a self-consistent interval and band',
	'drawn error-F1 minus the reference'
);
gates(
	(d) => {
		for (const arm of d.arms) {
			arm.delta_error_f1 += 0.1;
			arm.ci95_low += 0.1;
			arm.ci95_high += 0.1;
			if (arm.simultaneous_low !== null) {
				arm.simultaneous_low += 0.1;
				arm.simultaneous_high += 0.1;
			}
			arm.excludes_zero_pointwise = arm.ci95_low > 0 || arm.ci95_high < 0;
			arm.excludes_zero_simultaneous =
				arm.simultaneous_low === null
					? null
					: arm.simultaneous_low > 0 || arm.simultaneous_high < 0;
		}
		d.multiplicity.n_excluding_zero_simultaneous = d.arms.filter(
			(arm) => arm.excludes_zero_simultaneous === true
		).length;
	},
	'every margin shifted +0.10, turning the disclosed loss into a win',
	'drawn error-F1 minus the reference'
);
gates(
	(d) => {
		// The reference moved to a DIFFERENT but self-consistent cut, margins left
		// alone: the identity must be symmetric, not anchored on the arm side only.
		d.reference.operating_point = recut(d.reference.operating_point, 300, 200, d.panel.n, d.panel.n_errors);
	},
	'the reference re-cut while the margins stand — the identity is SYMMETRIC',
	'drawn error-F1 minus the reference'
);

// -- the matched-recall identity: the matched name holds the matched quantity
gates(
	(d) => {
		const matched = armIn(d, 'gemma-4-e2b').matched_recall;
		matched.delta_error_f1_at_matched_recall =
			matched.delta_error_f1_each_side_at_its_own_target_cut;
	},
	'the UNMATCHED subtraction shipped under the matched name — the original defect',
	'delta_error_f1_at_matched_recall'
);
gates(
	(d) => {
		armIn(d, 'glm-5').matched_recall.reference_error_f1_at_this_row += 0.01;
	},
	'a quoted re-cut summary that is not the re-cut row’s own F1',
	'reference_error_f1_at_this_row'
);
gates(
	(d) => {
		// The re-cut reference moved to a cut that catches FEWER errors than the arm
		// it is supposed to be matched to, with every dependent summary moved with
		// it so nothing else is inconsistent. Then it is not a matched comparison at
		// all — it is the reference judged at an easier recall — and the block's own
		// name would be false. A single-field probe cannot reach this gate, because
		// any single field it could move trips the rate identity first.
		const matched = armIn(d, 'glm-5').matched_recall;
		const row = matched.reference_at_this_rows_recall;
		const looser = recut(row, 380, row.fp, d.panel.n, d.panel.n_errors);
		matched.reference_at_this_rows_recall = looser;
		matched.reference_error_recall_at_this_row = looser.error_recall;
		matched.reference_error_f1_at_this_row = looser.error_f1;
		matched.delta_error_f1_at_matched_recall = matched.error_f1 - looser.error_f1;
	},
	'a “matched” reference re-cut BELOW the recall it is matched to',
	'does not reach this row'
);

// -- thresholds and their disclosures ---------------------------------------
gates(
	(d) => delete armIn(d, 'glm-5').operating_point.tau,
	'a threshold removed',
	'operating_point.tau'
);
gates(
	(d) => delete d.reference.matched_recall.tau,
	'the reference’s second threshold removed',
	'reference.matched_recall.tau'
);
gates((d) => delete d.oracle_disclosure, 'the oracle disclosure deleted', 'oracle_disclosure');
gates(
	(d) => {
		d.oracle_disclosure = '   ';
	},
	'the oracle disclosure blanked to whitespace',
	'oracle_disclosure'
);
gates((d) => delete d.threshold_rule, 'the headline threshold rule deleted', 'threshold_rule');
gates(
	(d) => delete d.matched_recall_rule,
	'the matched-cut threshold rule deleted',
	'matched_recall_rule'
);

// -- the panel ---------------------------------------------------------------
gates(
	(d) => {
		d.panel.n = 0;
	},
	'a zero-statement panel',
	'panel.n:'
);
gates(
	(d) => {
		d.panel.n_errors = 0;
	},
	'a panel with no errors to find',
	'panel.n_errors'
);
gates(
	(d) => {
		d.reconciliation.panel_matches.n = 1578;
	},
	'the review queue scored on a DIFFERENT panel',
	'not scored on this panel'
);
gates(
	(d) => {
		d.reconciliation.worst_residual = d.reconciliation.tolerance + 0.001;
	},
	'a review-queue cross-check that does not actually hold',
	'cross-check does not hold'
);
// THE SENTENCE'S OWN TWO QUANTITIES. The page promotes `worst_residual` into
// prose — "disagreeing by at most X error-F1" — and the only gate behind it used
// to be `worst_residual <= tolerance`, which is a different claim: it says the
// number is small, not that it is the disagreement. Both mutations below are
// SMALL and would have sailed through that check while the printed sentence
// described a residual no row exhibits. They gate on the derived comparison, so
// deleting either derivation in `paper-error-f1.ts` turns this runner red.
gates(
	(d) => {
		// The only row where the two rules land on different cuts, zeroed: the
		// sentence would then read "disagreeing by at most 0.000000".
		d.reconciliation.worst_residual = 0;
	},
	'a worst residual that is not the largest row disagreement',
	'is not the largest disagreement'
);
gates(
	(d) => {
		d.reconciliation.reference.residual = 0;
	},
	'a row residual that is not its own two error-F1 values',
	'is not this row’s own two error-F1 values'
);
gates(
	(d) => {
		// The queue side moved on its own: same shipped residual, different pair
		// of numbers under it. Still inside the tolerance, still false.
		d.reconciliation.reference.review_queue_error_f1 =
			d.reconciliation.reference.this_artifact_error_f1 - 0.001;
	},
	'a queue-side error-F1 the row’s own residual no longer describes',
	'is not this row’s own two error-F1 values'
);

// -- multiplicity, checks and provenance ------------------------------------
gates(
	(d) => {
		d.multiplicity.n_excluding_zero_simultaneous += 1;
	},
	'an exclusion count that disagrees with the arms',
	'n_excluding_zero_simultaneous'
);
gates(
	(d) => {
		d.multiplicity.max_t_critical_value = 1.5;
	},
	'a "simultaneous" band narrower than pointwise',
	'narrower than pointwise'
);
gates(
	(d) => {
		d.multiplicity.family[0] = 'port-rf-promoter';
	},
	'a declared family member that is not among the arms',
	'is not among the arms'
);
gates(
	(d) => {
		d.checks.reconciles_with_statement_review_queue = false;
	},
	'a weakened self-check',
	'checks.reconciles_with_statement_review_queue'
);
gates(
	(d) => {
		d.artifact_kind = 'statement_error_f2';
	},
	'a different artifact kind',
	'artifact_kind'
);
gates(
	(d) => {
		d.caveats = [];
	},
	'the caveat list emptied',
	'caveats'
);

// ---------------------------------------------------------------------------
// READ-NESS AND ITS GATES — the falsifiable half of must-assert item 2.
//
// The mapping pin above cannot fail when a loader gate is deleted, because on the
// real artifact every identity holds at residual 0.000e+00 and a derived value is
// the shipped double. This table can. Each field of `READ_POINT_FIELDS`,
// `READ_DELTA_FIELDS` and `READ_MATCHED_FIELDS` is perturbed ON ITS OWN, and the
// outcome is DECLARED here in advance:
//
//   · `gate: '<reason fragment>'` — the loader must refuse the artifact, for that
//     reason, in that context. Delete the gate and the probe sees the perturbed
//     value render instead, and this runner goes red. This is also the only
//     decidable form of "READ, not recomputed" for a gated field: a loader that
//     derived the value would be comparing a number against itself and could not
//     see the perturbation either.
//   · `gate: null` — the field is UNGATED, and the loader must render exactly what
//     was shipped. This is the direct test of READ-ness: a loader that started
//     deriving the field would return its own value, not the perturbed one.
//
// A declared-ungated field that starts gating also fails, deliberately: that is a
// real change in what the figure will accept, and the declaration is the record of
// it. Every key in the three lists must appear in a table below or the run fails,
// so the two halves cannot drift apart.
//
// WHAT THIS RUNNER DOES NOT COVER, measured rather than asserted. The whole file
// was run once per `fail(` site in the loader, each site neutralised in turn: 40
// of 61 take this runner red, 21 do not. Naming the 21, by the message they carry
// rather than by a line number that will move:
//
//   · TO `test-paper-render-invariants.mjs` (the page-wide sweep this component is
//     enrolled in): `budget()`'s "the gutter budget is N". That runner re-derives
//     the budgets from `STATEMENT_ERROR_F1_GEOMETRY` and measures the drawn labels
//     against them, which is a stronger check than a constant pinned here would be.
//     [1 gate]
//   · REDUNDANT WITH THE LIVE-DATA PINS ABOVE, which assert the same equality on
//     the shipped bytes: "n_errors + n_correct must equal n", "max-t cannot exceed
//     the Bonferroni critical value", "the zero rule must sit strictly inside the
//     margin axis", "the reference rule must sit inside the axis", "reconciled F1
//     is not the drawn F1", "review-queue join key drifted". Deleting any of these
//     changes nothing observable on THIS artifact, because the assertion above
//     already re-checks the property directly; they are fail-closed protection for
//     a future artifact, and the pins are the protection for this one. [6 gates]
//   · GENUINELY UNCOVERED — each needs an artifact this runner does not build, and
//     none of them is the last line of defence for a number on screen: "is not a
//     canonical paper-literal arm", "frozen join key drifted", "on-screen name
//     drifted", `reference.key` not the canonical reference, "multiplicity.family
//     expected a non-empty array", "must have family_size members",
//     "run_plan.stages expected a non-empty array", "arms expected a non-empty
//     array", "review_queue_model_key expected a string or null", "a declared
//     family member must be in the family", "the arms flagged in_max_t_family must
//     be exactly the declared family", "the reference lane was not drawn", and the
//     two reconciliation "is not a drawn lane" gates. [14 gates]
//
// RE-MEASURED, not carried forward, when the two reconciliation-residual gates
// below landed: the sweep was run again, all 61 sites neutralised one at a time,
// and it reads 40 red / 21 green with the 21 being exactly the ones named above.
// Both new sites are red. That means the totals for the loader WITHOUT them were
// 38 of 59, so the "40 of 61" this note used to carry named the right 21 but
// counted two sites the loader did not yet have; it is exact as of this revision.
// Re-run the sweep rather than adjusting these numbers by hand whenever a `fail(`
// site is added or removed — a hand-adjusted census is the same kind of unchecked
// claim this runner exists to refuse.
// ---------------------------------------------------------------------------
let readProbeCases = 0;
let gatedProbeCases = 0;
let readThroughProbeCases = 0;

/**
 * One field, perturbed alone, against a declared outcome.
 *
 * `rawOf` picks the containing record out of a mutable copy; `figureOf` picks the
 * record the component renders from. `perturb` returns the new value and must keep
 * the field's own type and domain, so the probe reaches the gate it names rather
 * than a type check upstream of it.
 */
function probeReadField(spec) {
	readProbeCases += 1;
	const copy = structuredClone(raw);
	const container = spec.rawOf(copy);
	const applied = spec.perturb(container[spec.key], container);
	container[spec.key] = applied;
	const result = validateStatementErrorF1(copy, { artifactPath: 'fixture', artifactSha256: 'x' });
	const where = `${spec.context}.${spec.key}`;

	if (spec.gate !== null) {
		gatedProbeCases += 1;
		if (result.status !== 'unavailable') {
			failures += 1;
			console.error(
				`FAIL probe ${where}: declared GATED, but the loader accepted the perturbed value — ` +
					'the gate that protects this field is gone'
			);
			return;
		}
		if (!result.reason.includes(spec.gate)) {
			failures += 1;
			console.error(
				`FAIL probe ${where}: gated on the wrong thing: ${result.reason} (wanted "${spec.gate}")`
			);
			return;
		}
		if (!result.reason.includes(spec.gateContext)) {
			failures += 1;
			console.error(
				`FAIL probe ${where}: gated somewhere else: ${result.reason} ` +
					`(wanted the reason to name "${spec.gateContext}")`
			);
		}
		return;
	}

	readThroughProbeCases += 1;
	if (result.status !== 'ok') {
		failures += 1;
		console.error(
			`FAIL probe ${where}: declared UNGATED, but the loader refused it: ${result.reason} — ` +
				'if a gate was added on purpose, declare it here'
		);
		return;
	}
	const rendered = spec.figureOf(result.figure)[spec.field];
	if (!Object.is(rendered, applied)) {
		failures += 1;
		console.error(
			`FAIL probe ${where}: shipped ${JSON.stringify(applied)}, rendered ` +
				`${JSON.stringify(rendered)} — this field is being DERIVED, not read`
		);
	}
}

const laneIn = (figure, id) => figure.lanes.find((lane) => lane.id === id);
/** The loader reports arm context by position, so the probe expects the same. */
const ARM_INDEX = Object.fromEntries(raw.arms.map((arm, index) => [arm.key, index]));

const down = (value) => value - 0.01;
const up = (value) => value + 0.01;
const plusOne = (value) => value + 1;
const negate = (value) => !value;

/**
 * Expectations for an operating point, shared by all four places one appears. The
 * headline cut's `tau` is cross-checked against the reconciliation table; the two
 * matched cuts' taus are quoted, not cross-checked, so each caller declares its own.
 */
function pointExpectations(tauGate, tauGateContext) {
	return {
		tau: { perturb: down, gate: tauGate, gateContext: tauGateContext },
		flagged: { perturb: plusOne, gate: 'flagged must equal tp + fp' },
		error_precision: { perturb: down, gate: 'own tp/fp/fn' },
		error_recall: { perturb: down, gate: 'own tp/fp/fn' },
		error_f1: { perturb: down, gate: 'own tp/fp/fn' },
		tp: { perturb: plusOne, gate: 'flagged must equal tp + fp' },
		fp: { perturb: plusOne, gate: 'flagged must equal tp + fp' },
		fn: { perturb: plusOne, gate: 'tp + fn must equal' },
		tn: { perturb: plusOne, gate: 'confusion table must cover the panel' },
		// The four rates below are read straight through: nothing on this figure
		// cross-checks them, so the probe proves they are the shipped numbers.
		accuracy: { perturb: down, gate: null },
		correct_precision: { perturb: down, gate: null },
		correct_recall: { perturb: down, gate: null },
		correct_f1: { perturb: down, gate: null },
		flag_set_is_the_arms_zero_pile: { perturb: negate, gate: null }
	};
}

const RECONCILED_TAU = 'reconciled tau is not the drawn tau';
const CONTAINS_POINTWISE = 'simultaneous band must contain the pointwise interval';

const PROBE_CONTEXTS = [
	{
		context: 'arms[glm-5].operating_point',
		gateContext: `arms[${ARM_INDEX['glm-5']}].operating_point`,
		rawOf: (d) => armIn(d, 'glm-5').operating_point,
		figureOf: (f) => laneIn(f, 'glm-5').operating,
		fields: READ_POINT_FIELDS,
		expect: pointExpectations(RECONCILED_TAU, 'reconciliation')
	},
	{
		context: 'reference.operating_point',
		gateContext: 'reference.operating_point',
		rawOf: (d) => d.reference.operating_point,
		figureOf: (f) => f.reference.operating,
		fields: READ_POINT_FIELDS,
		expect: pointExpectations(RECONCILED_TAU, 'reconciliation')
	},
	{
		context: 'arms[glm-5].matched_recall',
		gateContext: `arms[${ARM_INDEX['glm-5']}].matched_recall`,
		rawOf: (d) => armIn(d, 'glm-5').matched_recall,
		figureOf: (f) => laneIn(f, 'glm-5').matched.point,
		fields: READ_POINT_FIELDS,
		expect: pointExpectations(null, null)
	},
	{
		context: 'arms[glm-5].matched_recall.reference_at_this_rows_recall',
		gateContext: `arms[${ARM_INDEX['glm-5']}].matched_recall.reference_at_this_rows_recall`,
		rawOf: (d) => armIn(d, 'glm-5').matched_recall.reference_at_this_rows_recall,
		figureOf: (f) => laneIn(f, 'glm-5').matched.referenceAtThisRowsRecall,
		fields: READ_POINT_FIELDS,
		expect: pointExpectations(null, null)
	},
	{
		context: 'arms[glm-5].matched_recall',
		gateContext: `arms[${ARM_INDEX['glm-5']}].matched_recall`,
		rawOf: (d) => armIn(d, 'glm-5').matched_recall,
		figureOf: (f) => laneIn(f, 'glm-5').matched,
		fields: READ_MATCHED_FIELDS,
		expect: {
			// Raised above the recall the cut actually achieves, so the block's own
			// claim ("this cut delivers its target") stops being true.
			target_error_recall: { perturb: () => 0.99, gate: 'does not reach its own target recall' },
			reference_error_recall_at_this_row: {
				perturb: down,
				gate: 'reference_error_recall_at_this_row is not the re-cut row'
			},
			reference_error_f1_at_this_row: {
				perturb: down,
				gate: 'reference_error_f1_at_this_row is not the re-cut row'
			},
			reference_recall_overshoot: { perturb: up, gate: null },
			delta_error_f1_at_matched_recall: {
				perturb: up,
				gate: 'delta_error_f1_at_matched_recall'
			},
			delta_error_f1_each_side_at_its_own_target_cut: {
				perturb: up,
				gate: 'delta_error_f1_each_side_at_its_own_target_cut'
			}
		}
	},
	{
		context: 'arms[glm-5]',
		gateContext: `arms[${ARM_INDEX['glm-5']}]`,
		rawOf: (d) => armIn(d, 'glm-5'),
		figureOf: (f) => laneIn(f, 'glm-5').delta,
		fields: READ_DELTA_FIELDS,
		expect: {
			delta_error_f1: { perturb: up, gate: 'drawn error-F1 minus the reference' },
			delta_bootstrap_mean: { perturb: up, gate: null },
			// Either endpoint moved OUTWARD breaks the nesting the drawing asserts,
			// which is the first gate a widened pointwise interval meets.
			ci95_low: { perturb: down, gate: CONTAINS_POINTWISE },
			ci95_high: { perturb: up, gate: CONTAINS_POINTWISE },
			bootstrap_se: { perturb: up, gate: null },
			t_statistic: { perturb: up, gate: null },
			p_greater_than_zero: { perturb: (value) => value / 2, gate: null },
			n_valid_resamples: { perturb: (value) => value - 1, gate: null },
			// Pushed past its own upper bound, so it reaches the ordering gate rather
			// than the containment one below it.
			simultaneous_low: {
				perturb: (_value, arm) => arm.simultaneous_high + 0.01,
				gate: 'simultaneous_low must not exceed simultaneous_high'
			},
			simultaneous_high: { perturb: down, gate: CONTAINS_POINTWISE },
			excludes_zero_pointwise: {
				perturb: negate,
				gate: 'must equal ci95_low > 0 || ci95_high < 0'
			},
			excludes_zero_simultaneous: {
				perturb: negate,
				gate: 'must equal simultaneous_low > 0 || simultaneous_high < 0'
			}
		}
	}
];

for (const context of PROBE_CONTEXTS) {
	for (const [field, key] of context.fields) {
		const expectation = context.expect[key];
		if (expectation === undefined) {
			failures += 1;
			console.error(
				`FAIL probe ${context.context}.${key}: is drawn from the artifact but has no declared ` +
					'gated / read-through expectation'
			);
			continue;
		}
		probeReadField({
			context: context.context,
			gateContext: expectation.gateContext ?? context.gateContext,
			key,
			field,
			rawOf: context.rawOf,
			figureOf: context.figureOf,
			perturb: expectation.perturb,
			gate: expectation.gate
		});
	}
}

// The same discipline for the three fields outside those lists that the figure
// draws, and for the shapes a rendered number must have to be a number at all.
for (const extra of [
	{
		context: 'arms[glm-5]',
		gateContext: `arms[${ARM_INDEX['glm-5']}]`,
		key: 'distinct_scores',
		field: 'distinctScores',
		rawOf: (d) => armIn(d, 'glm-5'),
		figureOf: (f) => laneIn(f, 'glm-5'),
		perturb: plusOne,
		gate: null
	},
	{
		context: 'arms[indra-cogex-hybrid]',
		gateContext: `arms[${ARM_INDEX['indra-cogex-hybrid']}]`,
		key: 'simultaneous_low',
		field: 'simLow',
		rawOf: (d) => armIn(d, 'indra-cogex-hybrid'),
		figureOf: (f) => laneIn(f, 'indra-cogex-hybrid').delta,
		perturb: () => 0.1,
		gate: 'must carry no simultaneous band'
	},
	{
		context: 'arms[glm-5].operating_point [shape]',
		gateContext: `arms[${ARM_INDEX['glm-5']}].operating_point.accuracy`,
		key: 'accuracy',
		field: 'accuracy',
		rawOf: (d) => armIn(d, 'glm-5').operating_point,
		figureOf: (f) => laneIn(f, 'glm-5').operating,
		perturb: () => 1.5,
		gate: 'expected a number in [0, 1]'
	},
	{
		context: 'arms[glm-5].operating_point [shape]',
		gateContext: `arms[${ARM_INDEX['glm-5']}].operating_point.tn`,
		key: 'tn',
		field: 'tn',
		rawOf: (d) => armIn(d, 'glm-5').operating_point,
		figureOf: (f) => laneIn(f, 'glm-5').operating,
		perturb: () => -1,
		gate: 'expected a non-negative integer'
	},
	{
		context: 'arms[glm-5].operating_point [shape]',
		gateContext: `arms[${ARM_INDEX['glm-5']}].operating_point.flag_set_is_the_arms_zero_pile`,
		key: 'flag_set_is_the_arms_zero_pile',
		field: 'flagSetIsTheArmsZeroPile',
		rawOf: (d) => armIn(d, 'glm-5').operating_point,
		figureOf: (f) => laneIn(f, 'glm-5').operating,
		perturb: () => 'yes',
		gate: 'expected a boolean'
	},
	{
		context: 'arms[glm-5] [shape]',
		gateContext: `arms[${ARM_INDEX['glm-5']}].operating_point`,
		key: 'operating_point',
		field: 'operating',
		rawOf: (d) => armIn(d, 'glm-5'),
		figureOf: (f) => laneIn(f, 'glm-5'),
		perturb: () => 42,
		gate: 'expected an object'
	}
]) {
	probeReadField(extra);
}

// ---------------------------------------------------------------------------
// SIGN-BLINDNESS: the two mutations that must SURVIVE, correctly reclassified.
// The deleted `excludesZero` was `ciLow > 0 || ciHigh < 0` — sign-blind by
// construction — and four clauses here branched on it two-way and read a LOSS as
// a tie.
// A runner that only checks "broken things gate" passes while every loss renders
// as a win, so these are checked the other way round: the artifact stays valid
// and the CLASS has to move.
// ---------------------------------------------------------------------------
let reclassifyCases = 0;

function reclassifies(mutate, label, check) {
	reclassifyCases += 1;
	const copy = structuredClone(raw);
	mutate(copy);
	const result = validateStatementErrorF1(copy, { artifactPath: 'fixture', artifactSha256: 'x' });
	if (result.status !== 'ok') {
		failures += 1;
		console.error(`FAIL reclassify: ${label} — expected a valid artifact, got: ${result.reason}`);
		return;
	}
	check(result.figure, label);
}

// A pointwise interval moved entirely BELOW zero, the band still spanning it.
// This is the exact shape that rendered "pointwise […] does not [span zero], so
// the family-wide correction is what makes this a tie" without ever saying the
// interval was below zero.
reclassifies(
	(d) => {
		const arm = armIn(d, 'gemma-4-e2b');
		arm.ci95_low = -0.0381;
		arm.ci95_high = -0.002;
		arm.excludes_zero_pointwise = true;
	},
	'E2B’s pointwise interval pushed entirely below zero',
	(figure, label) => {
		const lane = figure.lanes.find((entry) => entry.id === 'gemma-4-e2b');
		eq(lane?.delta?.pointwiseStanding, 'behind', `${label}: pointwise standing reads BEHIND`);
		eq(lane?.delta?.standing, 'not-significant', `${label}: the band still says not-significant`);
		eq(lane?.delta?.winsPointwise, false, `${label}: an interval below zero is not a win`);
		eq(figure.nWinsPointwise, 3, `${label}: the pointwise census does not gain an arm`);
	}
);

// A winner pushed genuinely BEHIND the reference: a self-consistent worse cut,
// with the margin, both intervals and the reconciled row moved with it. Nothing
// is inconsistent, so nothing may gate — but the class must flip and the win
// must be given up.
//
// BOTH SIDES of the reconciled row move, not just this artifact's. The row's
// `residual` is the disagreement between the two threshold rules over ONE arm —
// the page prints its maximum as "disagreeing by at most X error-F1" — so a
// hypothetical where this artifact re-cuts GLM-5 to 0.38 while the review queue
// still measured 0.79 for the same arm is not a worse arm, it is a broken
// cross-check, and the loader now says so. Moving the queue side with it keeps
// the residual (and therefore the shipped `worst_residual`, still the
// reference's 0.007967) true, which is what this case needs to isolate the
// SIGN reclassification it is actually testing.
reclassifies(
	(d) => {
		const arm = armIn(d, 'glm-5');
		arm.operating_point = recut(arm.operating_point, 200, 400, d.panel.n, d.panel.n_errors);
		const delta = arm.operating_point.error_f1 - d.reference.operating_point.error_f1;
		arm.delta_error_f1 = delta;
		arm.delta_bootstrap_mean = delta;
		arm.ci95_low = delta - 0.03;
		arm.ci95_high = delta + 0.03;
		arm.simultaneous_low = delta - 0.05;
		arm.simultaneous_high = delta + 0.05;
		arm.t_statistic = -arm.t_statistic;
		arm.excludes_zero_pointwise = true;
		arm.excludes_zero_simultaneous = true;
		d.reconciliation.arms['glm-5'].this_artifact_error_f1 = arm.operating_point.error_f1;
		d.reconciliation.arms['glm-5'].review_queue_error_f1 = arm.operating_point.error_f1;
		d.reconciliation.arms['glm-5'].residual = 0;
	},
	'GLM-5 re-cut to a genuinely worse, self-consistent operating point',
	(figure, label) => {
		const lane = figure.lanes.find((entry) => entry.id === 'glm-5');
		eq(lane?.delta?.standing, 'behind', `${label}: the standing reads BEHIND, not not-significant`);
		eq(lane?.delta?.pointwiseStanding, 'behind', `${label}: and so does the pointwise class`);
		eq(lane?.delta?.winsSimultaneously, false, `${label}: an excluded-zero LOSS is not a win`);
		eq(lane?.delta?.winsPointwise, false, `${label}: nor pointwise`);
		eq(figure.nWinsSimultaneously, 2, `${label}: the win census drops to two`);
		ok(
			figure.lanes.findIndex((entry) => entry.isReference) < figure.lanes.findIndex((entry) => entry.id === 'glm-5'),
			`${label}: the lane order re-ranks it below the reference`
		);
	}
);

// ---------------------------------------------------------------------------
// SELF-GUARD. A runner that silently stopped mutating would otherwise exit 0.
// ---------------------------------------------------------------------------
console.log(`${mutationCases} fail-closed mutation cases exercised`);
console.log(`${reclassifyCases} sign-reclassification cases exercised`);
console.log(
	`${readProbeCases} read-vs-gated field probes exercised ` +
		`(${gatedProbeCases} declared gated, ${readThroughProbeCases} declared read-through)`
);
if (mutationCases < 30) {
	failures += 1;
	console.error(`FAIL fewer than 30 fail-closed mutation cases (${mutationCases})`);
}
if (reclassifyCases < 2) {
	failures += 1;
	console.error(`FAIL fewer than 2 sign-reclassification cases (${reclassifyCases})`);
}
// Every field of the three lists, in every context they appear in, plus the six
// extras. A probe table that silently stopped probing would otherwise exit 0 —
// which is the exact defect this section was added to fix.
const EXPECTED_PROBE_CASES =
	READ_POINT_FIELDS.length * 4 + READ_MATCHED_FIELDS.length + READ_DELTA_FIELDS.length + 6;
if (readProbeCases !== EXPECTED_PROBE_CASES) {
	failures += 1;
	console.error(
		`FAIL ${readProbeCases} read-vs-gated probes ran; the declared field lists and contexts ` +
			`require ${EXPECTED_PROBE_CASES}`
	);
}
// Both halves must be non-trivial: a table that declared everything read-through
// would assert nothing about the gates, and one that declared everything gated
// would never test READ-ness at all.
if (gatedProbeCases < 20 || readThroughProbeCases < 10) {
	failures += 1;
	console.error(
		`FAIL the probe table is one-sided (${gatedProbeCases} gated, ${readThroughProbeCases} read-through)`
	);
}
eq(
	STATEMENT_ERROR_F1_CONTENDER_ARM_IDS.length + 1,
	STATEMENT_ERROR_F1_DRAWN_ARM_IDS.length,
	'the drawn set is the contenders plus the one reference'
);
ok(STATEMENT_ERROR_F1_PARITY_TOL <= 1e-9, 'the identity tolerance is still decisive');

if (failures) {
	console.error(`\n${failures} statement-error-F1 contract assertion(s) failed`);
	process.exit(1);
}
console.log('statement-grain error-F1 data contract assertions passed');
