/**
 * Pure-function assertions for the EXTENDED TABLE 6 data contract.
 *
 * Modelled on `test-tie-inflation-contract.mjs`: the shipped validator is run
 * against the REAL artifact, then against mutations that must each gate the
 * figure to `unavailable` rather than draw a wrong ranked list.
 *
 * WHY THIS FILE EXISTS. This is the one figure that puts our arms INSIDE the
 * 2023 paper's own table, on their estimator, their folds and their labels. Three
 * separate claims carry it, and each is a different way to be wrong:
 *
 *   1. THE ORDER. Ranks 1–3 are ours and their best fitted model is rank 4. A
 *      figure whose whole point is an ordering must not be able to re-order
 *      quietly, so the anchors are pinned HERE, independently of the artifact's
 *      own `checks.expected_ranks` — and the two pins are then required to agree
 *      in both directions, so neither can drop an anchor alone.
 *
 *   2. THE LICENCE. Putting our rows in their table is licensed by the ≤0.0016
 *      agreement between the rows we RE-RAN and the values they printed. That
 *      licence covers 10 of these 20 rows. The other 10 (5 of ours, never
 *      published; 5 of theirs, printed only and never re-run) have no deviation
 *      at all, and an earlier draft of the lede claimed the bound over rows it
 *      could not cover. The scoping is asserted as a partition, not as prose.
 *
 *   3. THE OBJECTION. The paper's estimator is trapezoidal and hands coarse-scored
 *      arms area no threshold reaches. That gift is +0.0097..+0.0143 to our reader
 *      arms and −0.0008..+0.0006 to their models — and our OWN INDRA CoGEx hybrid,
 *      at 1,176 distinct scores, collects +0.0006 and sits with their models. The
 *      control is the whole argument that the effect tracks tie density rather
 *      than authorship, so it is checked against both groups, both ways.
 *
 * SIGN IS NEVER TAKEN ON MAGNITUDE. The paper-side gifts STRADDLE zero (7 of the
 * 13 scored rows are positive, 6 negative), so every cross-group comparison here
 * takes an absolute value on the paper side EXPLICITLY and asserts the reader
 * side is positive EXPLICITLY, rather than comparing raw extremes and reading a
 * separation out of a number whose sign it never checked. Sign-blindness has
 * shipped four times on this page; a bare `max()` over signed gifts is the same
 * defect wearing a different hat.
 *
 * Runs with `node --experimental-strip-types`. Reads two shipped files and the
 * component source. Writes nothing.
 */
import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';

import {
	PAPER_TABLE6_AGREEMENT_BOUND,
	PAPER_TABLE6_FAMILIES,
	PAPER_TABLE6_LLM_READER_FAMILY,
	PAPER_TABLE6_ORIGINS,
	PAPER_TABLE6_OUR_ARM_ID_BY_LABEL,
	validatePaperTable6Extended
} from '../src/lib/data/paper-table6-extended.ts';

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
/** Four decimals — the precision the claims in the GOAL and the lede are stated at. */
function round4(value) {
	return Number(value.toFixed(4));
}

const MODEL_DIR = new URL(
	'../../data/results/indra_paper_literal_models_20260724/',
	import.meta.url
);
const ARTIFACT_NAME = 'paper_table6_extended.json';
const bytes = readFileSync(new URL(ARTIFACT_NAME, MODEL_DIR));
const raw = JSON.parse(bytes.toString('utf8'));

// ---------------------------------------------------------------------------
// The anchors this node exists to pin. Written here as LABELS — the frozen join
// keys — never as display strings, which are the paper's own screen names and
// are free to be re-worded. `side` records which claim the anchor carries.
// ---------------------------------------------------------------------------
const ANCHORS = [
	{ label: 'ours_glm_5', rank: 1, side: 'ours' },
	{ label: 'ours_gemma_4_26b', rank: 2, side: 'ours' },
	{ label: 'ours_gemma_4_31b', rank: 3, side: 'ours' },
	{ label: 'paper_rf_prom_avglen', rank: 4, side: 'theirs' },
	{ label: 'paper_rf_promoter', rank: 5, side: 'theirs' },
	{ label: 'ours_gemma_4_e2b', rank: 11, side: 'ours' },
	{ label: 'paper_belief_orig', rank: 16, side: 'theirs' },
	{ label: 'paper_svc', rank: 20, side: 'theirs' }
];
const CONTROL_LABEL = 'ours_indra_cogex_hybrid';
const CONTROL_DISTINCT_SCORES = 1176;

// ---- the shipped bytes are the bytes the run signed --------------------------
// Cheap here, and the manifest is read-modify-written by several generators, so
// a lost update would otherwise surface only as a gated figure in the browser.
{
	const digest = createHash('sha256').update(bytes).digest('hex');
	const manifest = JSON.parse(readFileSync(new URL('manifest.json', MODEL_DIR), 'utf8'));
	eq(
		manifest?.output_sha256?.[ARTIFACT_NAME],
		digest,
		'the run manifest signs the extended-Table-6 bytes on disk'
	);
}

// ---- the real artifact validates --------------------------------------------
const live = validatePaperTable6Extended(raw, {
	artifactPath: 'fixture',
	artifactSha256: 'deadbeef'
});
eq(live.status, 'ok', `the shipped artifact validates (${live.reason ?? ''})`);

if (live.status === 'ok') {
	const figure = live.figure;
	const rows = figure.rows;
	const byLabel = new Map(rows.map((row) => [row.label, row]));
	const rawByLabel = new Map(raw.rows.map((row) => [row.label, row]));

	// -- 1. row count and a CONTIGUOUS rank sequence ---------------------------
	eq(rows.length, raw.n_rows, 'the drawn row count is the row count the artifact declares');
	eq(rows.length, 20, 'this table is the twenty-row "all sources, specific" block');
	const ranks = rows.map((row) => row.rank);
	eq(new Set(ranks).size, ranks.length, 'no rank is used twice');
	ok(
		ranks.every((rank, index) => rank === index + 1),
		`ranks run 1..${rows.length} in drawn order without a gap: ${ranks.join(',')}`
	);
	// Contiguity is not enough on its own: it is satisfied by any permutation that
	// was re-numbered. The ORDER must also be the order the artifact says it is.
	ok(
		rows.every((row, index) => index === 0 || rows[index - 1].foldMean >= row.foldMean),
		'the ranking is descending on the paper’s own fold-mean trapezoidal PR-AUC'
	);

	// -- 2. the anchors --------------------------------------------------------
	for (const anchor of ANCHORS) {
		const row = byLabel.get(anchor.label);
		ok(row !== undefined, `anchor "${anchor.label}" is a row in this table`);
		if (!row) continue;
		eq(row.rank, anchor.rank, `${row.display} holds rank ${anchor.rank}`);
		eq(
			row.origin === 'ours',
			anchor.side === 'ours',
			`${row.display} is on the ${anchor.side} side of the table`
		);
	}
	// The three leading ranks are ours, and their best model is the row after.
	eq(figure.leadingOurRanks, 3, 'three leading ranks are ours — the fact the banded chart hides');
	eq(figure.theirBestRank, 4, 'their table now starts at rank 4');
	ok(
		rows.slice(0, 3).every((row) => row.origin === 'ours' && row.family === PAPER_TABLE6_LLM_READER_FAMILY),
		'ranks 1–3 are all reader arms of ours'
	);
	// Ranks 4–5 are their two best, and are RFs. Derived from the rows rather than
	// asserted from the anchor list, so the two pins have to agree.
	const theirRows = rows.filter((row) => row.origin !== 'ours');
	eq(
		theirRows.slice(0, 2).map((row) => row.label).join(' | '),
		'paper_rf_prom_avglen | paper_rf_promoter',
		'their two best rows are the two RF+promoter variants'
	);
	ok(
		theirRows.slice(0, 2).every((row) => row.origin === 'paper_rerun' && row.family === 'paper_fitted_ml'),
		'both of their leading rows are fitted models we re-ran from their own code'
	);
	// Display-side sanity only — never a join. The paper's own row names for these
	// two begin "RF", and the anchor claim is stated as "their best RFs".
	ok(
		theirRows.slice(0, 2).every((row) => row.display.startsWith('RF ')),
		`their two leading rows are named as RFs on screen: ${theirRows
			.slice(0, 2)
			.map((row) => row.display)
			.join(' | ')}`
	);
	// The reference arm every paired margin on /paper is measured against is one of
	// the two, and is named as the reference exactly once.
	eq(
		rows.filter((row) => row.isReference).map((row) => row.label).join(''),
		'paper_rf_promoter',
		'exactly one row is the reference arm, and it is their RF+promoter'
	);

	// The artifact pins the same anchors independently. Required to agree BOTH
	// ways: a re-generated file that dropped an anchor from its own checks would
	// otherwise still satisfy this runner, and a runner that dropped one would
	// still satisfy the artifact.
	const shippedPins = new Map(raw.checks.expected_ranks.map((pin) => [pin.label, pin.rank]));
	eq(shippedPins.size, ANCHORS.length, 'the artifact pins exactly the anchors this runner pins');
	for (const anchor of ANCHORS) {
		eq(
			shippedPins.get(anchor.label),
			anchor.rank,
			`the artifact's own checks pin ${anchor.label} at rank ${anchor.rank}`
		);
	}

	// -- 3. the replication bound, SCOPED to the rows it can cover -------------
	const rerun = rows.filter((row) => row.origin === 'paper_rerun');
	const publishedOnly = rows.filter((row) => row.origin === 'paper_published_only');
	const ours = rows.filter((row) => row.origin === 'ours');
	eq(ours.length, 5, 'five rows are ours');
	eq(rerun.length, 10, 'ten rows are theirs, re-run here');
	eq(publishedOnly.length, 5, 'five rows are theirs, printed only');
	eq(
		ours.length + rerun.length + publishedOnly.length,
		rows.length,
		'the three origins partition the table'
	);
	eq(figure.reproduction.nRerunRows, rerun.length, 'the re-run census matches the drawn rows');
	ok(
		figure.reproduction.tolerance <= PAPER_TABLE6_AGREEMENT_BOUND,
		`the artifact's own tolerance (${figure.reproduction.tolerance}) is inside the ${PAPER_TABLE6_AGREEMENT_BOUND} bound`
	);
	for (const row of rerun) {
		ok(
			row.absDevVsPublished !== null && row.absDevVsPublished <= PAPER_TABLE6_AGREEMENT_BOUND,
			`${row.display} lands within ${PAPER_TABLE6_AGREEMENT_BOUND} of its published value (${row.absDevVsPublished})`
		);
	}
	ok(
		figure.reproduction.maxAbsDev <= PAPER_TABLE6_AGREEMENT_BOUND,
		`the worst re-run deviation (${figure.reproduction.maxAbsDev} on ${figure.reproduction.maxAbsDevDisplay}) is inside the bound`
	);
	// THE SCOPING, as a partition rather than as a sentence: a deviation exists on
	// the re-run rows and NOWHERE else, so the bound cannot be read over the table.
	eq(
		rows.filter((row) => row.absDevVsPublished !== null).length,
		rerun.length,
		'a published-vs-re-run deviation exists on exactly the rows we re-ran'
	);
	ok(
		rows.length > rerun.length,
		'the bound covers fewer rows than the table has — it is a scoped claim'
	);
	for (const row of [...ours, ...publishedOnly]) {
		eq(row.absDevVsPublished, null, `${row.display} was never re-run, so it carries no deviation`);
	}
	for (const row of ours) {
		eq(row.publishedMean, null, `${row.display} is ours and was never published`);
	}

	// -- 4. published-only rows carry NO tie-robust number ---------------------
	for (const row of publishedOnly) {
		eq(row.ap, null, `${row.display} has no score vector, so no average precision`);
		eq(row.tieGift, null, `${row.display} has no score vector, so no tie gift`);
		eq(row.apRank, null, `${row.display} has no score vector, so no AP rank`);
		eq(row.distinctScores, null, `${row.display} has no score vector, so no distinct-score count`);
		eq(row.rounded, true, `${row.display} exists only at the paper's three decimals`);
		eq(
			row.publishedMean,
			row.foldMean,
			`${row.display} IS its printed value; the two must be the same number`
		);
		ok(
			!/AP rank/.test(row.tieReadout),
			`${row.display} does not print an AP rank it cannot have: "${row.tieReadout}"`
		);
	}
	// A gift exists exactly where a score vector does — including on two rows of
	// THEIRS that we re-ran but whose out-of-fold scores were never released, so
	// "no tie-robust number" is not a property of being published-only.
	const scored = rows.filter((row) => row.ap !== null);
	eq(scored.length, 13, 'thirteen rows carry a score vector');
	eq(
		figure.reproduction.nRerunRowsWithScores,
		rerun.filter((row) => row.ap !== null).length,
		'the scored-re-run census matches the drawn rows'
	);
	ok(
		rerun.some((row) => row.ap === null),
		'some re-run rows also lack a score vector, so the guard is not just about publication'
	);
	for (const row of rows) {
		eq(
			row.tieGift === null,
			row.ap === null,
			`${row.display}: a tie gift exists exactly where an average precision does`
		);
	}

	// -- 5. the ± is a population fold SD, and nothing calls it an interval ----
	eq(figure.metric.uncertaintyIsConfidenceInterval, false, 'the ± is not a confidence interval');
	eq(figure.metric.uncertaintyIsDispersion, true, 'the artifact asserts the ± is dispersion');
	eq(figure.metric.metricIsPooledAveragePrecision, false, 'the ranking metric is not average precision');
	ok(
		/population standard deviation/i.test(figure.metric.uncertaintyField),
		`the ± is named as a population SD: "${figure.metric.uncertaintyField}"`
	);
	ok(
		/not a confidence interval/i.test(figure.metric.uncertaintyNote),
		'the uncertainty note says outright that the ± is not a confidence interval'
	);
	ok(
		rows.every((row) => row.foldCount === figure.nFolds && figure.nFolds === 10),
		'every row disperses over the same ten folds'
	);
	eq(figure.paperReportsNoTests, true, 'the paper reports no test alongside these numbers');
	eq(figure.convention.nPValues, 0, 'their table carries no p-value');
	eq(figure.convention.nConfidenceIntervals, 0, 'their table carries no confidence interval');
	eq(figure.convention.nMultiplicityCorrections, 0, 'their table carries no multiplicity correction');

	// -- 6. the tie gift: readers, paper models, and the control ---------------
	const readerRows = rows.filter((row) => row.family === PAPER_TABLE6_LLM_READER_FAMILY);
	const paperScored = rerun.filter((row) => row.tieGift !== null);
	eq(readerRows.length, 4, 'four reader arms of ours are in the table');
	eq(paperScored.length, 8, 'eight rows of theirs carry a gift');
	eq(figure.tie.readers.count, readerRows.length, 'the reader group summarises every reader row');
	eq(figure.tie.paperRerun.count, paperScored.length, 'their group summarises every scored row of theirs');

	// The published range, at the precision it is published at.
	eq(round4(figure.tie.readers.min), 0.0097, 'the readers’ smallest gift is +0.0097');
	eq(round4(figure.tie.readers.max), 0.0143, 'the readers’ largest gift is +0.0143');
	eq(round4(figure.tie.paperRerun.min), -0.0008, 'their smallest gift is −0.0008');
	eq(round4(figure.tie.paperRerun.max), 0.0006, 'their largest gift is +0.0006');

	// SIGN, ASSERTED — not inferred from a magnitude. Their gifts straddle zero, so
	// any cross-group comparison that reads a bare extreme is reading the wrong end.
	ok(
		readerRows.every((row) => row.tieGift !== null && row.tieGift > 0),
		'every reader arm is GIVEN area by the trapezoid — all four gifts are positive'
	);
	ok(
		paperScored.some((row) => row.tieGift < 0) && paperScored.some((row) => row.tieGift > 0),
		'their gifts straddle zero, so a magnitude-only comparison would be sign-blind'
	);

	// The separation, pairwise and absolute-valued on their side.
	for (const reader of readerRows) {
		for (const theirs of paperScored) {
			ok(
				reader.tieGift > 10 * Math.abs(theirs.tieGift),
				`${reader.display} (${reader.tieGift}) collects more than 10x |${theirs.display}| (${theirs.tieGift})`
			);
		}
	}
	// Tie density is the mechanism, so it separates the same way.
	ok(
		Math.max(...readerRows.map((row) => row.distinctScores)) <
			Math.min(...paperScored.map((row) => row.distinctScores)),
		'every reader arm emits fewer distinct scores than every scored row of theirs'
	);

	// THE CONTROL. Ours, and it sits with their models on both axes.
	const control = byLabel.get(CONTROL_LABEL);
	ok(control !== undefined, 'the INDRA CoGEx hybrid control is in the table');
	if (control) {
		eq(control.origin, 'ours', 'the control is one of our own arms — that is its whole force');
		ok(
			control.family !== PAPER_TABLE6_LLM_READER_FAMILY,
			'the control is not tagged as one of our reader arms'
		);
		eq(control.distinctScores, CONTROL_DISTINCT_SCORES, 'the control emits 1,176 distinct scores');
		eq(round4(control.tieGift), 0.0006, 'the control collects +0.0006');
		eq(figure.tie.control.display, control.display, 'the disclosure names the control row it draws');
		eq(figure.tie.control.tieGift, control.tieGift, 'the disclosure quotes the control’s own gift');
		// With their models, not with our readers — stated as both inequalities.
		ok(
			control.tieGift >= figure.tie.paperRerun.min && control.tieGift <= figure.tie.paperRerun.max,
			`the control's gift (${control.tieGift}) is inside their models' range`
		);
		ok(
			control.tieGift * 10 < Math.min(...readerRows.map((row) => row.tieGift)),
			'the control is an order of magnitude below the least-gifted reader arm'
		);
		ok(
			control.distinctScores > Math.max(...readerRows.map((row) => row.distinctScores)),
			'the control scores far finer than any reader arm'
		);
		// And the coarse/fine split the artifact draws puts it on their side.
		ok(
			control.distinctScores > figure.tie.separation.coarseMaxDistinctScores,
			'the density cut puts the control on the fine-scored side, with their models'
		);
		eq(
			figure.tie.separation.nCoarse,
			readerRows.length,
			'the coarse side is exactly our four reader arms'
		);
		eq(
			figure.tie.separation.nFine,
			paperScored.length + 1,
			'the fine side is their eight scored rows plus our control'
		);
	}

	// The correction is not cosmetic: it changes WHICH of our arms leads.
	ok(
		figure.tie.best.ourPaperMetricDisplay !== figure.tie.best.ourApDisplay,
		`the tie correction changes our leading arm (${figure.tie.best.ourPaperMetricDisplay} → ${figure.tie.best.ourApDisplay})`
	);
	ok(
		figure.tie.best.apMargin > 0 && figure.tie.best.apMargin < figure.tie.best.paperMetricMargin,
		`the lead survives the correction but shrinks (${figure.tie.best.paperMetricMargin} → ${figure.tie.best.apMargin})`
	);
	// Direction, per row, among the rows that HAVE both ranks: the gift buys E2B a
	// place among the thirteen scored rows and buys the control none.
	const scoredMetricRank = new Map(scored.map((row, index) => [row.label, index + 1]));
	eq(scoredMetricRank.get('ours_gemma_4_e2b'), 11, 'E2B is 11th of the scored rows on their metric');
	eq(byLabel.get('ours_gemma_4_e2b')?.apRank, 12, 'E2B is 12th once the tie gift is taken back');
	eq(scoredMetricRank.get(CONTROL_LABEL), 13, 'the control is last of the scored rows on their metric');
	eq(byLabel.get(CONTROL_LABEL)?.apRank, 13, 'and last once the tie gift is taken back — it moves nothing');
	// The gift-density correlation is signed and NEGATIVE: coarser scores, bigger
	// gift. A validator comparing |ρ| would accept the opposite claim.
	ok(figure.tie.spearman < 0, `the gift anti-correlates with score-vector width (ρ = ${figure.tie.spearman})`);
	ok(
		/mid-rank/i.test(figure.tie.spearmanMethod),
		'the correlation is named as a mid-rank Spearman, which is what it is'
	);

	// -- 7. label is the join key; display is the screen name ------------------
	for (const row of rows) {
		ok(row.display !== row.label, `${row.label}: display is decoupled from the join key`);
		ok(
			/^[a-z0-9_]+$/.test(row.label),
			`${row.label}: a join key is a stable slug, not a screen name`
		);
		ok(
			!/^[a-z0-9_]+$/.test(row.display),
			`"${row.display}": a screen name must not be mistakable for a join key`
		);
		// The key is the artifact's, unchanged: `display` may be re-worded freely,
		// the key may not move without a deliberate edit to the artifact.
		ok(rawByLabel.has(row.label), `${row.label} is a real key in the shipped artifact`);
		eq(
			rawByLabel.get(row.label)?.display,
			row.display,
			`${row.label}: the drawn name is the artifact's, not a rebuilt one`
		);
	}
	eq(new Set(rows.map((row) => row.label)).size, rows.length, 'every join key is unique');
	// Our rows additionally resolve through the pinned table to a canonical
	// paper-literal arm, so an arm added to this table must be re-pinned by hand.
	for (const row of ours) {
		ok(
			row.label in PAPER_TABLE6_OUR_ARM_ID_BY_LABEL,
			`${row.label} is pinned to a canonical paper-literal arm`
		);
		ok(
			row.headToHeadDisplay !== null && row.headToHeadDisplay.length > 0,
			`${row.display} resolves to its head-to-head name`
		);
	}
	// Nothing the figure renders may name a key. The validator enforces this over
	// the artifact's own prose; re-asserted here over what the rows actually carry.
	const renderedStrings = rows.flatMap((row) => [row.display, row.title, row.metricReadout, row.tieReadout]);
	for (const rendered of renderedStrings) {
		for (const label of rows.map((row) => row.label)) {
			ok(!rendered.includes(label), `a rendered string names the join key "${label}"`);
		}
	}
	// Every origin and family the table uses is inside its enum.
	ok(
		rows.every((row) => PAPER_TABLE6_ORIGINS.includes(row.origin)),
		'every row sits in the origin enum'
	);
	ok(
		rows.every((row) => PAPER_TABLE6_FAMILIES.includes(row.family)),
		'every row sits in the family enum'
	);
}

// ---------------------------------------------------------------------------
// Fail-closed: each mutation must gate the figure, not render it.
// `expect` is a regex the gate REASON must match, used where a mutation could
// plausibly trip an unrelated guard first — a gate credited for the wrong reason
// is a gate that will not fire when the real drift arrives.
// ---------------------------------------------------------------------------
let mutationCases = 0;
/** A floor, not an equality: adding cases is free, silently removing them fails. */
const MIN_MUTATION_CASES = 24;

function mutated(mutate, label, expect = null) {
	mutationCases += 1;
	const copy = JSON.parse(JSON.stringify(raw));
	mutate(copy);
	const result = validatePaperTable6Extended(copy, { artifactPath: 'fixture' });
	eq(result.status, 'unavailable', `gates: ${label}`);
	if (expect && result.status === 'unavailable') {
		ok(
			expect.test(result.reason),
			`gates for the right reason: ${label} — got "${result.reason}"`
		);
	}
}

const rowAt = (doc, label) => doc.rows.find((row) => row.label === label);

// -- the ranked list itself ---------------------------------------------------
mutated((d) => d.rows.splice(19, 1), 'a row disappearing');
mutated(
	(d) => {
		// …and the same deletion with every count made consistent with it, so the
		// contiguity check cannot be the thing that catches it.
		d.rows.splice(19, 1);
		d.n_rows = 19;
		d.checks.n_rows = 19;
	},
	'the last-ranked row deleted with its counts patched',
	/paper_svc/
);
mutated(
	(d) => {
		d.rows.splice(16, 1);
		d.n_rows = 19;
		d.checks.n_rows = 19;
	},
	'a gap opened in the middle of the rank sequence',
	/without a gap/
);
mutated((d) => {
	rowAt(d, 'ours_gemma_4_26b').rank = 1;
}, 'two rows claiming the same rank', /rank 1 is used more than once/);
mutated((d) => {
	rowAt(d, 'ours_glm_5').origin = 'independent_replication';
}, 'an origin outside the enum', /outside the origin enum/);
mutated((d) => {
	rowAt(d, 'ours_glm_5').family = 'best_in_class';
}, 'a family outside the enum', /outside the family enum/);
mutated((d) => {
	// Retagging one of THEIR rows as one of our readers would widen the group whose
	// tie range the figure quotes.
	rowAt(d, 'paper_rf_promoter').family = 'llm_reader';
}, 'one of their rows retagged as a reader arm of ours');
mutated((d) => {
	const row = rowAt(d, 'ours_glm_5');
	row.rank = 2;
	rowAt(d, 'ours_gemma_4_26b').rank = 1;
}, 'the top two ranks swapped without their scores moving', /scores below rank/);
mutated((d) => {
	d.checks.expected_ranks[0].rank = 2;
}, 'the artifact’s own anchor pin moved off the row it names', /expects rank 2/);

// -- the replication licence ---------------------------------------------------
mutated(
	(d) => {
		// Pushed past the bound CONSISTENTLY: the deviation and the published value
		// it is derived from move together, so the identity check cannot catch it
		// first and the bound itself has to.
		const row = rowAt(d, 'paper_rf_type_pmids');
		row.published_mean = row.fold_mean_trapezoidal_pr_auc - 0.002;
		row.abs_dev_vs_published = 0.002;
	},
	'a re-run row deviating past the 0.0016 agreement bound',
	/past the .* bound/
);
mutated((d) => {
	rowAt(d, 'paper_rf_type_pmids').abs_dev_vs_published = 0.0002;
}, 'a deviation that is not |re-run − published|', /must equal \|fold mean − published mean\|/);
mutated(
	(d) => {
		d.reproduction_fidelity.tolerance = 0.01;
		d.checks.max_abs_dev_vs_published_tolerance = 0.01;
	},
	'the artifact relaxing its own agreement bound',
	/is outside \(0, 0.0016\]/
);
mutated((d) => {
	// A deviation attached to a row that was never re-run: the exact overclaim the
	// scoping exists to prevent.
	rowAt(d, 'paper_svc').abs_dev_vs_published = 0.0004;
}, 'a printed-only row given a re-run deviation');
mutated((d) => {
	rowAt(d, 'ours_glm_5').published_mean = 0.9649;
}, 'one of our arms given a published value it never had');

// -- published-only rows have no tie-robust number -----------------------------
mutated((d) => {
	rowAt(d, 'paper_belief_orig').pooled_average_precision = 0.92;
}, 'a printed-only row given an average precision');
mutated(
	(d) => {
		// The consistent version: score vector, AP, gift, width and rank all supplied
		// so every pairwise presence check is satisfied. It must still gate, because
		// a row that was never re-run has no score vector for any of them to come from.
		const row = rowAt(d, 'paper_svc');
		row.has_out_of_fold_scores = true;
		row.pooled_average_precision = 0.894;
		row.tie_gift = row.fold_mean_trapezoidal_pr_auc - 0.894;
		row.distinct_scores = 900;
		row.pooled_ap_rank = 14;
	},
	'a printed-only row given a complete, self-consistent tie-robust block',
	/expected null/
);
mutated((d) => {
	rowAt(d, 'paper_knn').tie_gift = 0.001;
}, 'a printed-only row given a tie gift');

// -- the ± ---------------------------------------------------------------------
mutated((d) => delete rowAt(d, 'ours_glm_5').fold_population_sd, 'a fold SD removed');
mutated((d) => {
	rowAt(d, 'paper_belief_orig').fold_population_sd = 0.02;
}, 'a printed-only row drawing a ± that is not the one the paper printed');
mutated((d) => {
	d.metric_contract.uncertainty_is_dispersion_not_a_confidence_interval = false;
}, 'the artifact withdrawing its dispersion assertion', /expected true/);
mutated((d) => {
	d.metric_contract.uncertainty_is_confidence_interval = true;
}, 'the ± relabelled as a confidence interval', /expected false/);
mutated((d) => {
	d.paper_reporting_convention.n_confidence_intervals = 2;
}, 'the no-tests census contradicting the no-tests claim', /disagrees with its own census/);

// -- the tie disclosure ---------------------------------------------------------
mutated((d) => {
	const row = rowAt(d, 'ours_glm_5');
	row.tie_gift = row.tie_gift / 2;
}, 'a gift that is no longer the paper metric minus the tie-robust one');
mutated(
	(d) => {
		// The range narrowed by dropping the row that sets its floor.
		const group = d.tie_disclosure.llm_reader_arms;
		group.labels = group.labels.filter((label) => label !== 'ours_gemma_4_e2b');
		group.min = 0.01004894434644732;
		group.distinct_scores_min = 475;
	},
	'the reader tie range narrowed by dropping a reading model',
	// RE-WORDED 2026-07-29 with the loader's gate text. The assertion tests the
	// same INTENT it always did — the gate must name the class-completeness
	// failure (the group no longer covers every reading model in the table) and
	// not some unrelated guard tripping first. It quoted the old phrasing "rows
	// are one of our llm_reader arms", which named the group by whose it is and by
	// a frozen family tag; the gate reason renders on the page when the figure
	// gates, so the wording moved and this regex followed it.
	/rows are the reading models in this table/
);
mutated((d) => {
	d.tie_disclosure.spearman_gift_vs_distinct_scores =
		-d.tie_disclosure.spearman_gift_vs_distinct_scores;
}, 'the gift-vs-density correlation flipped in sign', /mid-rank Spearman/);
mutated((d) => {
	d.tie_disclosure.indra_cogex_hybrid.label = 'paper_rf_promoter';
	// Same re-wording, same intent: the control's whole force is that it is one of
	// the NEWLY SCORED rows and still collects nothing, so re-pointing it at a
	// published row must gate on exactly that fact.
}, 'the control re-pointed at a published row', /must be one of the newly scored rows/);
mutated(
	(d) => {
		// The control moved onto the readers' side of the density cut, with every
		// count patched to match. The separation claim is then false, and says so.
		const separation = d.tie_disclosure.separation;
		separation.coarse_max_distinct_scores = 1176;
		separation.n_coarse = 5;
		separation.n_fine = 8;
		separation.min_gift_among_coarse_scored_arms = 0.000636448388305344;
	},
	'the density cut slid to put the control with our readers',
	/overlap/
);
mutated((d) => {
	rowAt(d, 'ours_indra_cogex_hybrid').family = 'llm_reader';
}, 'the control retagged as a reader arm of ours');
mutated((d) => {
	rowAt(d, 'ours_glm_5').distinct_scores = d.n_statements + 1;
}, 'more distinct scores than the panel has statements', /exceeds the 1689 statements/);
mutated((d) => {
	rowAt(d, 'ours_gemma_4_26b').pooled_ap_rank = 2;
}, 'a collision in the tie-robust rank order');

// -- the join keys ---------------------------------------------------------------
mutated((d) => {
	const row = rowAt(d, 'ours_glm_5');
	row.display = row.label;
}, 'a display string collapsed onto its join key', /must differ from the join key/);
mutated((d) => {
	rowAt(d, 'ours_glm_5').label = 'ours_glm_five';
}, 'one of our rows re-keyed without being re-pinned', /re-pin it deliberately/);

ok(
	mutationCases >= MIN_MUTATION_CASES,
	`the mutation suite still has cases: ${mutationCases} < ${MIN_MUTATION_CASES}`
);

// ---------------------------------------------------------------------------
// The component must not contradict what the artifact asserts about itself.
// Source-level, because both defects are invisible to a render: a ± described as
// an interval reads exactly like a ± described as dispersion, and a reproduction
// sentence scoped to the wrong rows renders as a fact.
// ---------------------------------------------------------------------------
{
	const component = readFileSync(
		new URL('../src/lib/components/PaperTable6Extended.svelte', import.meta.url),
		'utf8'
	);
	// The words that actually reach the screen. `includes()` over raw source is
	// also satisfied by a code comment, which no reader sees; positive claims are
	// therefore asserted here and NEGATIVE ones stay on the raw source, where
	// they are the stricter of the two.
	let componentBody = component
		.replace(/<!--[\s\S]*?-->/g, ' ')
		.replace(/<script\b[^>]*>[\s\S]*?<\/script\s*>/gi, ' ')
		.replace(/<style\b[^>]*>[\s\S]*?<\/style\s*>/gi, ' ');
	// `{…}` first, so a `<` inside an expression cannot eat the markup after it.
	for (;;) {
		const next = componentBody.replace(/\{[^{}]*\}/g, ' ');
		if (next === componentBody) break;
		componentBody = next;
	}
	const componentText = componentBody.replace(/<[^>]+>/g, ' ').replace(/\s+/g, ' ');
	// The denial, in whatever wording carries it — singular or plural, with or
	// without the article.
	ok(
		/\bnot\s+(a\s+)?confidence intervals?\b/i.test(componentText),
		'the component states outright that the ± is not a confidence interval'
	);
	// Every mention of the phrase must be a DENIAL of it. Checked positionally
	// rather than by counting, so a second, affirming mention cannot hide behind
	// the first one's denial.
	for (const match of component.matchAll(/confidence intervals?/gi)) {
		const before = component.slice(Math.max(0, match.index - 16), match.index);
		ok(
			/\bnot\s+(a\s+)?$/i.test(before),
			`every "confidence interval" in the component is a denial; found "…${before}${match[0]}"`
		);
	}
	ok(!/\b95\s*%/.test(component), 'the component quotes no 95% coverage beside the ±');
	ok(!/\b1\.96\b/.test(component), 'the component applies no normal multiplier to the ±');
	// Saying what the ± is NOT is only half of it; the bar also has to be named
	// for what it IS. Tested as that claim rather than as the word "dispersion":
	// beside the denial, something MOVES, and it moves across a slice count read
	// off the figure rather than typed. A byte pin on one term makes a plain
	// rewrite of the same claim read as a regression, which is how this page has
	// lost a build before.
	const MOVES = /\b(sd|standard deviations?|deviations?|dispersion|spreads?|scatter|moves?|moved|moving|movement|varies|variation|swing|range)\b/i;
	ok(
		[...componentText.matchAll(/confidence intervals?/gi)].some((match) => {
			const around = componentText.slice(Math.max(0, match.index - 240), match.index + 240);
			return MOVES.test(around);
		}),
		'the component names what the bar measures, not only what it is not'
	);
	ok(
		/\{figure\.nFolds\}/.test(component),
		'…over a slice count read off the figure rather than typed'
	);
	// The reproduction sentence is scoped by construction: it counts the re-run
	// rows and the never-re-run rows separately off the drawn rows.
	ok(
		component.includes('reproduction.nRerunRows'),
		'the reproduction sentence counts the rows we actually re-ran'
	);
	ok(
		component.includes('countByOrigin.paper_published_only'),
		'…and states the never-re-run rows as their own count'
	);
	ok(
		!/every published row/i.test(component),
		'the component does not claim the agreement bound over rows it never re-ran'
	);
	// The gift stub is drawn in both directions on this data, so a one-sided legend
	// would be a false statement about six of the thirteen stubs.
	ok(
		/tieGift !== null && row\.tieGift > 0/.test(component) &&
			/tieGift !== null && row\.tieGift < 0/.test(component),
		'the gift legend counts both signs rather than describing one direction'
	);
	if (live.status === 'ok') {
		const gifts = live.figure.rows.map((row) => row.tieGift).filter((gift) => gift !== null);
		ok(
			gifts.some((gift) => gift > 0) && gifts.some((gift) => gift < 0),
			'both signs really do occur among the drawn gifts'
		);
	}
	// The frozen keys stay out of the drawing: `label` is an each-key at most.
	ok(
		!/\{row\.label\}/.test(component) && !/\{figure\.rows\[[^\]]*\]\.label\}/.test(component),
		'no join key is interpolated into a render position'
	);
}

console.log(`${mutationCases} fail-closed mutation cases exercised`);
console.log(
	failures === 0
		? 'extended Table 6 data contract assertions passed'
		: `${failures} extended Table 6 contract assertion(s) failed`
);
process.exit(failures === 0 ? 0 : 1);
