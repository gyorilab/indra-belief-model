/**
 * Pure-function assertions for the PER-EVIDENCE grain data contract.
 *
 * Mirrors test-paper-robustness-contract.mjs: the validator is exercised against
 * the REAL shipped artifact, then against mutations that must each gate the
 * figure to `unavailable` rather than render a wrong number.
 *
 * Why this file exists. This surface carries four claims that are each one field
 * flip away from being wrong:
 *   1. the two marks in a lane are the SAME model at two grains — licensed only
 *      by an EXACT statement-grain reconciliation, so a non-exact one must gate;
 *   2. the baselines are INDRA library code, NOT published paper methods — so a
 *      missing attribution must gate;
 *   3. the per-evidence panel is the same for every arm — so an unscored
 *      reviewed pair must be visible, never silently dropped;
 *   4. right-anchored SVG text clips its LEADING glyphs in silence — so the
 *      gutter budgets must be enforced by the builder, not by eyeballing.
 * Each is asserted below against the live artifact and against a mutation.
 *
 * Run: node --experimental-strip-types scripts/test-per-evidence-contract.mjs
 */
import { readFileSync } from 'node:fs';

import {
	PAPER_PER_EVIDENCE_ARTIFACT_KIND,
	PAPER_PER_EVIDENCE_CHANCE,
	PAPER_PER_EVIDENCE_DISPLAY,
	PAPER_PER_EVIDENCE_GEOMETRY,
	PAPER_PER_EVIDENCE_LABEL_BUDGET_CHARS,
	PAPER_PER_EVIDENCE_READOUT_BUDGET_CHARS,
	PAPER_PER_EVIDENCE_SERIES,
	PAPER_PER_EVIDENCE_SERIES_IDS,
	PAPER_PER_EVIDENCE_SOURCE_TICKS,
	chanceLabelFits,
	laneNameFits,
	readoutFits,
	validatePaperPerEvidence
} from '../src/lib/data/paper-per-evidence.ts';

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
/** A mutation must gate. Passing one through is the failure mode this file hunts. */
function gates(mutate, label) {
	const copy = structuredClone(raw);
	mutate(copy);
	const result = validatePaperPerEvidence(copy, { artifactPath: 'fixture' });
	ok(result.status === 'unavailable', `${label}: expected the figure to gate`);
	if (result.status === 'unavailable') {
		ok(result.figure === null, `${label}: a gated figure carries no geometry`);
		ok(typeof result.reason === 'string' && result.reason.length > 0, `${label}: carries a reason`);
	}
}

const ARTIFACT = new URL(
	'../../data/results/per_evidence_comparison_20260727/per_evidence_comparison.json',
	import.meta.url
);
const raw = JSON.parse(readFileSync(ARTIFACT, 'utf8'));

eq(raw.artifact_kind, PAPER_PER_EVIDENCE_ARTIFACT_KIND, 'the artifact is the kind we accept');

const live = validatePaperPerEvidence(raw, { artifactPath: 'fixture', artifactSha256: 'deadbeef' });
eq(live.status, 'ok', `the shipped artifact validates${live.status === 'ok' ? '' : `: ${live.reason}`}`);

if (live.status === 'ok') {
	const figure = live.figure;
	const lanes = figure.lanes;

	// ---- the panel is what the brief says it is -------------------------------
	eq(figure.nPositive + figure.nNegative, figure.nReviewedPairs, 'the panel census closes');
	ok(figure.nReviewedPairs > figure.nStatements, 'the evidence panel is larger than the statement panel');
	close(
		figure.powerRatio,
		figure.nReviewedPairs / figure.nStatements,
		1e-9,
		'the power ratio is the item ratio'
	);
	ok(figure.powerNote.length > 0, 'the power ratio ships with its clustering caveat');

	// ---- shipped fields are READ, never recomputed ----------------------------
	const armById = new Map(raw.arms.filter((arm) => arm.metrics).map((arm) => [arm.id, arm]));
	eq(lanes.length, armById.size, 'every arm carrying metrics reaches the figure');
	for (const lane of lanes) {
		const shipped = armById.get(lane.id);
		ok(shipped !== undefined, `${lane.id}: is a real artifact arm`);
		if (!shipped) continue;
		close(lane.evidence.value, shipped.metrics.auroc, 1e-12, `${lane.id}: evidence AUROC is READ`);
		close(
			lane.evidence.low,
			shipped.metrics.interval.auroc.ci95_low,
			1e-12,
			`${lane.id}: interval low is READ`
		);
		close(
			lane.evidence.high,
			shipped.metrics.interval.auroc.ci95_high,
			1e-12,
			`${lane.id}: interval high is READ`
		);
		close(
			lane.statementAuroc,
			shipped.statement_grain.auroc,
			1e-12,
			`${lane.id}: statement AUROC is READ`
		);
		close(
			lane.grainShift,
			shipped.statement_grain.auroc - shipped.metrics.auroc,
			1e-12,
			`${lane.id}: the grain shift is the difference of two READ fields`
		);
		close(
			lane.errorF1,
			shipped.metrics.error_detection.f1,
			1e-12,
			`${lane.id}: error-detection F1 is READ`
		);
		// The interval is drawn as a whisker THROUGH the mark; a mark off its own
		// whisker would be a drawing that contradicts itself.
		ok(
			lane.evidence.low <= lane.evidence.value && lane.evidence.value <= lane.evidence.high,
			`${lane.id}: the point estimate lies inside its own interval`
		);
	}

	// ---- display is DECOUPLED from the frozen join key ------------------------
	for (const lane of lanes) {
		eq(lane.display, PAPER_PER_EVIDENCE_DISPLAY[lane.id], `${lane.id}: canonical display name`);
		ok(lane.joinKey.length > 0, `${lane.id}: carries its frozen join key`);
		ok(lane.display !== lane.joinKey, `${lane.id}: display is not the frozen join key`);
	}

	// ---- MISATTRIBUTION BAN: no arm may render without provenance -------------
	for (const lane of lanes) {
		ok(lane.attribution.length > 0, `${lane.id}: carries an attribution`);
	}
	const bundled = lanes.find((lane) => lane.id === 'indra-default-source-prior');
	ok(bundled !== undefined, 'the bundled source prior is drawn');
	if (bundled) {
		ok(
			/not that arm|UNFITTED/i.test(bundled.attribution),
			'the bundled prior says it is not the paper’s own refit belief arm'
		);
	}
	for (const id of ['indra-bayes-source-oof', 'indra-bayes-subtype-oof']) {
		const lane = lanes.find((entry) => entry.id === id);
		ok(lane !== undefined, `${id} is drawn`);
		if (lane) {
			ok(
				/publishes NO/i.test(lane.attribution),
				`${id}: says the 2023 paper publishes no such arm`
			);
		}
	}
	for (const lane of lanes.filter((entry) => entry.kind === 'reader')) {
		ok(/NOT zero-shot/i.test(lane.attribution), `${lane.id}: says it is not zero-shot`);
	}
	// The paper's supervised rows cannot be drawn at this grain, and that exclusion
	// must ship as data rather than as prose someone can delete.
	ok(figure.coverage.excludedBaselines.length > 0, 'the statement-only exclusion ships as data');
	ok(
		figure.coverage.excludedBaselines.every(
			(entry) => entry.family.length > 0 && entry.reason.length > 0
		),
		'each exclusion carries a family and a reason'
	);

	// ---- the grain bridge is EXACT, or there is no figure ---------------------
	eq(figure.reaggregation.verified, true, 'the reconciliation is verified');
	eq(
		figure.reaggregation.nExact,
		figure.reaggregation.nStatements,
		'every statement reproduces exactly'
	);
	eq(figure.reaggregation.maxAbsDiff, 0, 'the reconciliation residual is exactly zero');
	ok(figure.twoGrainNote.length > 0, 'the two-grain caveat ships with the figure');
	ok(
		/not a causal increment|not paired/i.test(figure.twoGrainNote),
		'the two-grain caveat says the connector is not a paired increment'
	);

	// ---- no reviewed pair is silently dropped ---------------------------------
	eq(figure.coverage.unscoredPairs, 0, 'no reviewed pair lacks a reader verdict');
	eq(
		figure.coverage.reviewedPairs + figure.coverage.unreviewedPairs,
		figure.coverage.executedUniquePairs,
		'the coverage census closes against the execution map'
	);
	const tierTotal = figure.coverage.tierCensus.reduce((sum, row) => sum + row.pairs, 0);
	eq(tierTotal, figure.nReviewedPairs, 'the route census covers every reviewed pair');

	// ---- the per-source register ---------------------------------------------
	const censusTotal = figure.sourceRows.reduce((sum, row) => sum + row.reviewedPairs, 0);
	eq(censusTotal, figure.nReviewedPairs, 'the source census sums to the panel');
	for (const row of figure.sourceRows) {
		eq(
			row.positivePairs + row.negativePairs,
			row.reviewedPairs,
			`${row.source}: the row census closes`
		);
		close(
			row.observedCorrectFraction,
			row.positivePairs / row.reviewedPairs,
			1e-9,
			`${row.source}: the observed fraction is its own counts`
		);
	}
	ok(figure.sharedPrior !== null, 'the shared-prior block is present');
	if (figure.sharedPrior) {
		const shared = figure.sharedPrior;
		ok(shared.sources.length >= 2, 'a shared prior needs at least two sources');
		// The whole point: the sources sharing ONE prior do not behave alike.
		const rows = figure.sourceRows.filter((row) => shared.sources.includes(row.source));
		eq(rows.length, shared.sources.length, 'every shared-prior source is in the census');
		for (const row of rows) {
			close(
				row.bundledPriorAtOneEvidence,
				shared.sharedPrior,
				1e-12,
				`${row.source}: carries the shared prior exactly`
			);
		}
		const observed = rows.map((row) => row.observedCorrectFraction);
		close(shared.observedMin, Math.min(...observed), 1e-12, 'the shipped min is the census min');
		close(shared.observedMax, Math.max(...observed), 1e-12, 'the shipped max is the census max');
		ok(shared.observedMax - shared.observedMin > 0.1, 'the spread the one prior conflates is real');
		ok(shared.pValue < 0.001, 'the sources sharing one prior differ');
	}

	// ---- geometry and the label budgets ---------------------------------------
	for (const lane of lanes) {
		ok(
			lane.display.length <= PAPER_PER_EVIDENCE_LABEL_BUDGET_CHARS,
			`${lane.id}: lane name fits the right-anchored gutter`
		);
		ok(
			lane.readout.length <= PAPER_PER_EVIDENCE_READOUT_BUDGET_CHARS,
			`${lane.id}: readout fits the left-anchored gutter`
		);
	}
	// The budgets must be DERIVED from the geometry, not typed in beside it.
	const G = PAPER_PER_EVIDENCE_GEOMETRY;
	eq(
		PAPER_PER_EVIDENCE_LABEL_BUDGET_CHARS,
		Math.floor(G.labelAnchorX / G.monoUnitsPerChar),
		'the lane budget is the gutter divided by the measured 9px advance'
	);
	eq(
		PAPER_PER_EVIDENCE_READOUT_BUDGET_CHARS,
		Math.floor((G.width - G.readoutX) / G.readoutUnitsPerChar),
		'the readout budget is the gutter divided by the measured 8px advance'
	);
	close(G.readoutUnitsPerChar, (G.monoUnitsPerChar * 8) / 9, 1e-4, 'the 8px advance scales the 9px one');
	// The fit predicates are exercised at their boundary, because the artifact
	// alone can never produce an over-long name (the display table is canonical),
	// which would leave the one guard against a silent clip untested.
	ok(laneNameFits('x'.repeat(PAPER_PER_EVIDENCE_LABEL_BUDGET_CHARS)), 'a name at budget fits');
	ok(
		!laneNameFits('x'.repeat(PAPER_PER_EVIDENCE_LABEL_BUDGET_CHARS + 1)),
		'a name one char over budget does not fit'
	);
	ok(readoutFits('x'.repeat(PAPER_PER_EVIDENCE_READOUT_BUDGET_CHARS)), 'a readout at budget fits');
	ok(
		!readoutFits('x'.repeat(PAPER_PER_EVIDENCE_READOUT_BUDGET_CHARS + 1)),
		'a readout one char over budget does not fit'
	);
	// And every name the canonical table can produce clears it.
	for (const [id, display] of Object.entries(PAPER_PER_EVIDENCE_DISPLAY)) {
		ok(laneNameFits(display), `PAPER_PER_EVIDENCE_DISPLAY[${id}] fits the gutter`);
	}

	for (const lane of lanes) {
		ok(lane.evidenceY < lane.statementY, `${lane.id}: the evidence mark sits above the statement mark`);
		ok(
			Math.abs(lane.evidenceY - lane.y) <= G.laneHeight / 2 &&
				Math.abs(lane.statementY - lane.y) <= G.laneHeight / 2,
			`${lane.id}: both marks stay inside their lane`
		);
	}
	const domainSpan = figure.domainMax - figure.domainMin;
	ok(domainSpan > 0, 'the discrimination axis has a positive span');
	ok(figure.domainMin < PAPER_PER_EVIDENCE_CHANCE, 'the chance rule is inside the axis, not on its end');
	for (const lane of lanes) {
		ok(
			lane.evidence.low >= figure.domainMin && lane.evidence.high <= figure.domainMax,
			`${lane.id}: its interval is inside the axis`
		);
		ok(
			lane.statementAuroc >= figure.domainMin && lane.statementAuroc <= figure.domainMax,
			`${lane.id}: its statement mark is inside the axis`
		);
	}
	ok(figure.ticks.length >= 3, 'the axis carries readable ticks');
	ok(
		figure.ticks.every((tick, index) => index === 0 || tick > figure.ticks[index - 1]),
		'ticks ascend'
	);
	eq(figure.chanceLabelFits, chanceLabelFits(figure.chanceLabel, chanceXOf(figure)), 'the chance label fit is measured, not assumed');
	ok(PAPER_PER_EVIDENCE_SOURCE_TICKS.length >= 3, 'the probability register has its own ticks');

	// ---- ordering: bands, then per-evidence strength --------------------------
	const readers = lanes.filter((lane) => lane.kind === 'reader');
	const baselines = lanes.filter((lane) => lane.kind === 'baseline');
	ok(readers.length > 0 && baselines.length > 0, 'both bands are populated');
	eq(
		lanes.slice(0, readers.length).every((lane) => lane.kind === 'reader'),
		true,
		'readers occupy the first band'
	);
	for (const band of [readers, baselines]) {
		for (let i = 1; i < band.length; i += 1) {
			ok(
				band[i - 1].evidence.value >= band[i].evidence.value,
				`${band[i].id}: bands descend by per-evidence AUROC`
			);
		}
	}
	ok(
		figure.baselineBandY > figure.readerBandY && figure.sourceBandY > figure.baselineBandY,
		'the three band headers are stacked in reading order'
	);

	// ---- each series has its OWN (stroke, dash, shape) ------------------------
	const styles = PAPER_PER_EVIDENCE_SERIES_IDS.map((id) => PAPER_PER_EVIDENCE_SERIES[id]);
	eq(new Set(styles.map((style) => style.strokeVar)).size, styles.length, 'no two series share a stroke');
	eq(new Set(styles.map((style) => style.shape)).size, styles.length, 'no two series share a mark shape');
	eq(
		new Set(styles.map((style) => `${style.strokeVar}|${style.dash}`)).size,
		styles.length,
		'no two series share a (stroke, dash) pair'
	);
	ok(
		styles.every((style) => style.strokeVar.startsWith('var(--')),
		'every stroke is a token, never a raw hex'
	);
	ok(
		styles.every((style) => style.legend.length > 0),
		'every series carries a legend'
	);

	// ---- contamination is reported, not buried --------------------------------
	ok(
		figure.contamination.overlappingPairsSameClaim <= figure.contamination.overlappingPairs,
		'same-claim overlaps are a subset of overlaps'
	);
	if (figure.contamination.overlappingPairs > 0) {
		ok(figure.contamination.pairsKept !== null, 'an overlap ships with its sensitivity');
		ok(
			figure.contamination.maxAurocAbsShift !== null,
			'an overlap ships with how far it moves the metric'
		);
		eq(
			figure.contamination.pairsKept,
			figure.nReviewedPairs - figure.contamination.overlappingPairs,
			'the sensitivity panel is the primary panel minus the overlaps'
		);
	}
}

function chanceXOf(figure) {
	const G = PAPER_PER_EVIDENCE_GEOMETRY;
	return (
		G.plotLeft +
		((PAPER_PER_EVIDENCE_CHANCE - figure.domainMin) / (figure.domainMax - figure.domainMin)) *
			(G.plotRight - G.plotLeft)
	);
}

// ---- mutations: each must GATE, never render ---------------------------------
gates((copy) => {
	copy.artifact_kind = 'something_else';
}, 'a foreign artifact kind');

gates((copy) => {
	copy.reaggregation.arms[Object.keys(copy.reaggregation.arms)[0]].max_abs_diff = 1e-12;
}, 'a reconciliation that is close but not exact');

gates((copy) => {
	const key = Object.keys(copy.reaggregation.arms)[0];
	copy.reaggregation.arms[key].n_exact -= 1;
}, 'one statement failing to reproduce');

gates((copy) => {
	copy.reaggregation.verified = false;
}, 'an unverified reconciliation');

gates((copy) => {
	delete copy.arms.find((arm) => arm.id === 'indra-bayes-source-oof').attribution;
}, 'a baseline with no attribution');

gates((copy) => {
	copy.arms.find((arm) => arm.id === 'llm-glm-5').display = 'GLM 5 (renamed)';
}, 'a display name that drifts from the canonical table');

gates((copy) => {
	copy.arms.find((arm) => arm.id === 'llm-glm-5').metrics.n -= 1;
}, 'an arm scored on fewer pairs than the panel');

gates((copy) => {
	copy.arms.find((arm) => arm.id === 'llm-glm-5').metrics.interval.auroc.ci95_low =
		copy.arms.find((arm) => arm.id === 'llm-glm-5').metrics.auroc + 0.01;
}, 'a point estimate outside its own interval');

gates((copy) => {
	copy.coverage.per_arm[Object.keys(copy.coverage.per_arm)[0]].raw_attempts_sha256_matches_manifest =
		false;
}, 'a raw attempts file whose digest no longer matches its manifest');

gates((copy) => {
	copy.coverage.sources[0].positive_pairs += 1;
}, 'a source census that does not close');

gates((copy) => {
	copy.coverage.excluded_baselines = [];
}, 'dropping the statement-only exclusion');

gates((copy) => {
	copy.arms.find((arm) => arm.id === 'llm-glm-5').display =
		'GLM-5 with an extremely long display name that cannot fit the right-anchored gutter at all';
}, 'a lane name that drifts from the canonical table');

gates((copy) => {
	delete copy.statement_grain.note;
}, 'dropping the two-grain caveat');

gates((copy) => {
	copy.contamination.n_overlapping_pairs_same_claim = copy.contamination.n_overlapping_pairs + 1;
}, 'more same-claim overlaps than overlaps');

gates((copy) => {
	delete copy.contamination.sensitivity;
}, 'reporting overlaps with no sensitivity beside them');

gates((copy) => {
	copy.arms.find((arm) => arm.id === 'llm-glm-5').per_source = {};
}, 'an arm missing a per-source stratum');

if (failures > 0) {
	console.error(`\n${failures} assertion(s) failed`);
	process.exit(1);
}
console.log('per-evidence contract: all assertions passed');
