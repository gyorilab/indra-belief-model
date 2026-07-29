/**
 * Typed data contract for the STATEMENT-GRAIN ERROR-CLASS F1 surface: the margin
 * the 2023 INDRA paper's own panel actually supports, named rather than implied.
 *
 * WHAT THIS NODE ARGUES. The head-to-head on this page quotes pooled average
 * precision, which is the most CONSERVATIVE lens available — the panel is 73.2%
 * positive and the paper's own RF+promoter already scores AP 0.9412, so only
 * 0.0588 of that scale is left to win. The work a reader gate actually does is
 * pushing ERRORS DOWN, and at statement grain that is error-class F1. On the
 * paper's own 1,689 statements, their released labels and their own folds, the
 * three larger reader gates beat their RF by +0.126 to +0.142 with intervals that
 * exclude zero simultaneously; the smallest gate does not, and says so.
 *
 * THREE DIFFERENT TAU RULES LIVE IN ONE ARTIFACT, AND EACH NUMBER CARRIES ITS OWN.
 * A threshold-based number without its threshold rule is an over-claim wherever it
 * is written, and this artifact fits taus by three different rules:
 *
 *   · `best-f1`          the headline cut — the arm's own full-panel best-error-F1
 *                        threshold. Rule text: `threshold_rule`. Oracle text:
 *                        `oracle_disclosure`.
 *   · `target-recall-60` the matched-recall block's SECOND cut — each arm's
 *                        cheapest cut catching ≥60% of the panel's errors. Rule
 *                        text: `matched_recall_rule`, which also states that the
 *                        oracle applies to it by a DIFFERENT rule.
 *   · `review-queue`     the separately-ruled cut in
 *                        `statement_review_queue.json`, reached by target recall
 *                        under `belief <= tau`, quoted only inside the
 *                        reconciliation block. Rule text: `reconciliation.note`.
 *
 * `ErrorF1ThresholdRule` pairs each rendered group with the rule that produced its
 * tau, so the component cannot draw one group's number under another's disclosure.
 *
 * THE MATCHED-RECALL BLOCK SHIPS TWO DELTAS AND ONLY ONE OF THEM IS MATCHED.
 * `deltaAtMatchedRecall` re-cuts the REFERENCE at each row's ACHIEVED error recall
 * — that is the recall-matched comparison and the only one this surface may
 * present as such. `deltaEachSideAtItsOwnTargetCut` puts both sides at their own
 * 60% cut; because the arms' achieved recalls span 26.1 points, it flatters
 * whichever arm overshot furthest and it FLIPS Gemma 4 E2B's sign (−0.0059 matched
 * against +0.0350 unmatched). It is carried because it is a real quantity and
 * deleting it would lose information, it is named for exactly what it is, and
 * `signsDisagree` is derived so the component can point at the flip instead of
 * hoping a reader notices. An earlier revision of the artifact emitted only the
 * unmatched form under the matched name; that is the defect this module exists to
 * make un-renderable.
 *
 * SHIPPED FIELDS ARE READ, NEVER RECOMPUTED. Every precision, recall, F1, count,
 * tau, delta, interval bound, critical value and disclosure string is READ off
 * `data/results/indra_paper_literal_models_20260724/statement_error_f1.json`. The
 * loader derives NO metric. What it does derive is (a) drawing geometry, (b)
 * formatted display strings, (c) censuses and extrema OVER shipped values (how
 * many arms clear zero, the lowest and highest winning precision), and (d)
 * consistency GATES that compare shipped fields against each other and fail
 * closed. A gate may compute `arm.errorF1 − reference.errorF1`; nothing rendered
 * ever comes from that subtraction.
 *
 * EVERY RENDERED SCALAR IS TIED TO THE ONES DRAWN BESIDE IT. `parseOperatingPoint`
 * proves each of `error_precision`, `error_recall` and `error_f1` is the rate of
 * that row's own tp/fp/fn — without it, a precision of 0.95 beside untouched
 * counts rendered "the winning gates 75%–95%" and put the F1 tick outside the
 * precision–recall span. `parseDelta` proves the headline
 * `delta_error_f1` is the margin between the two error-F1 values panel A draws,
 * and `parseMatchedRecall` proves `delta_error_f1_at_matched_recall` is this
 * row's F1 minus the reference RE-CUT at this row's achieved recall. Both hold
 * byte-exactly on the shipped artifact. Without the first, a delta inflated to
 * +0.30 with a matching interval and band passed every other check and the lede
 * printed it beside the 0.7855 and 0.6439 that refute it. The reconciliation
 * block is held the same way: each row's `residual` is proved to be the gap
 * between the two error-F1 values that row prints, and `worst_residual` — which
 * the page promotes into prose as "disagreeing by at most X error-F1" — is proved
 * to be the largest of those. It was previously checked only against `tolerance`,
 * which says the number is small, not that it is the disagreement the sentence
 * names.
 *
 * THE DRAWN ARM SET IS FROZEN AGAINST THE CANONICAL SPEC TABLE, not read off the
 * artifact — see `STATEMENT_ERROR_F1_DRAWN_ARM_IDS`. And every margin carries a
 * THREE-WAY `standing` (ahead / not-significant / behind) with the interval that
 * decided it, so the component can select a sentence instead of asserting one —
 * plus a `pointwiseStanding` in the same three classes, because a clause that
 * PRINTS the pointwise interval beside a standing decided on the simultaneous
 * band needs its own sign, and a two-way boolean there is the same defect one
 * clause deeper. One helper, `standingOfBounds` in paper-literal.ts, decides
 * both — and every other interval on /paper.
 *
 * This module is import-safe on the client: typed shape plus a pure, fail-closed
 * validator. All filesystem work lives in `$lib/server/paper-error-f1`.
 */

import {
	PAPER_LITERAL_ARM_SPECS,
	PAPER_LITERAL_REFERENCE_ARM_ID,
	pairShippedProse,
	standingOfBounds,
	type AnchoredProse,
	type ShippedProse,
	type Standing
} from './paper-literal.ts';

/** The artifact kind this module will accept, and nothing else. */
export const STATEMENT_ERROR_F1_ARTIFACT_KIND = 'statement_error_f1';

/**
 * THE ARM SET THIS FIGURE MUST DRAW — frozen against the canonical spec table,
 * not read off the artifact's own `arms` array.
 *
 * Why it is not derived from the artifact. `multiplicity.family` is the max-t
 * family, which is FOUR arms; gating "every arm is present" against it left the
 * fifth contender ungated, so deleting `indra-cogex-hybrid` from the artifact
 * still returned `status:'ok'` and drew five lanes where six belong — quietly
 * removing a DISCLOSED LOSS from the page. The contenders are exactly the
 * canonical `kind: 'llm'` arms (the four reader gates plus the out-of-family
 * INDRA CoGEx hybrid control); the reference is the canonical paper RF. A
 * missing arm, an extra arm or a duplicate gates the figure.
 */
export const STATEMENT_ERROR_F1_CONTENDER_ARM_IDS: readonly string[] =
	PAPER_LITERAL_ARM_SPECS.filter((spec) => spec.kind === 'llm').map((spec) => spec.id);

/** Contenders plus the reference: every lane this figure draws. */
export const STATEMENT_ERROR_F1_DRAWN_ARM_IDS: readonly string[] = [
	PAPER_LITERAL_REFERENCE_ARM_ID,
	...STATEMENT_ERROR_F1_CONTENDER_ARM_IDS
];

/**
 * Float slack for the gates that check one shipped scalar against a difference of
 * two others. Both sides are IEEE doubles near 0.6–0.8 written out at 17
 * significant digits, so the achievable residual is ~1e-16; 1e-9 is decisive and
 * still tolerates the JSON round-trip.
 */
export const STATEMENT_ERROR_F1_PARITY_TOL = 1e-9;

/**
 * SVG geometry for the two-panel lane figure, exported so the label budgets below
 * are DERIVED from it rather than eyeballed, and so a contract runner can
 * re-derive them. Same discipline as `paper-robustness.ts`: right-anchored SVG
 * text that overruns its gutter loses its LEADING glyphs silently — no layout
 * error, no failing assertion, and the `<desc>` still reads correctly to a screen
 * reader, so review never sees it.
 *
 * Horizontal budget, re-measured against the FROZEN drawn set: the lane-label
 * gutter is 0 → `labelAnchorX` = 186 user units. At 9px the mono face advances
 * `monoUnitsPerChar` = 5.4186 units per character, so the gutter holds
 * ⌊186 / 5.4186⌋ = 34 characters. The six lanes this figure can draw
 * (`STATEMENT_ERROR_F1_DRAWN_ARM_IDS`) top out at the reference's own name,
 * "RF 2k-d13 + Type/#PMIDs/promoter", 32 characters = 173.40 units — 12.60 units
 * (2.3 characters) of headroom.
 *
 * TWO CANONICAL DISPLAYS DO NOT FIT, and both belong to arms this figure never
 * draws: "RF 2k-d13 + Type/#PMIDs/prom/avglen" (35 chars = 189.65 units, spec
 * `paper-rf-prom-avglen`) and "Our port of RF + Type/#PMIDs/promoter" (37 =
 * 200.49, spec `port-rf-promoter`). The earlier claim that 32 was the longest
 * name "this artifact can produce" was unenforced — any spec could have arrived
 * in `arms`. It is now enforced, and by TWO gates in a definite order, which is
 * what makes the arithmetic above a bound rather than an observation:
 *
 *   1. THE DRAWN-SET GATE fires first, and in practice it is the only one that
 *      ever fires. Neither spec is `kind: 'llm'`, so neither is a contender;
 *      an arm carrying either display that merely turns up in the artifact's
 *      `arms` is rejected as "not one of this figure's contenders" BEFORE any
 *      lane is built, so `budget()` is never reached.
 *   2. `budget()` is the backstop for the one path that gets past (1): promoting
 *      such a spec to `kind: 'llm'`, which puts it in the drawn set. Then a
 *      missing arm still gates at (1) ("the figure must draw …; it is missing"),
 *      and an arm actually supplied reaches lane construction, where `budget()`
 *      fails on the 35- or 37-character display.
 *
 * Either way the figure gates to `unavailable` and this text is never clipped,
 * because a right-anchored SVG string that overruns loses its LEADING glyphs in
 * silence and the `<desc>` beside it still reads correctly. Fitting them would
 * mean dropping the lane name to 8px
 * (37 × 4.8165 = 178.24 units), which collapses its contrast with the 7.5px tau
 * sub-label — so the budget stays measured at 9px and the arm set stays frozen.
 *
 * Vertical budget, worked: each lane is `laneHeight` = 34 units and carries two
 * right-anchored lines — the 9px name at baseline `laneY − 1` and the 7.5px tau
 * sub-label at baseline `laneY + 10`. Taking the usual 0.75em ascent / 0.21em
 * descent, the name occupies [laneY − 7.75, laneY + 0.89] and the sub-label
 * [laneY + 4.38, laneY + 11.58]: they clear each other by 3.49 units, the stack is
 * 19.33 units tall inside a 34-unit lane, and the next lane's name begins at
 * laneY + 26.25 — 14.68 units below this lane's sub-label. No clipping and no
 * inter-lane collision at any lane count.
 */
export const STATEMENT_ERROR_F1_GEOMETRY = {
	width: 960,
	/** Lane names are right-anchored here; usable gutter is 0 → 186 units. */
	labelAnchorX: 186,
	/** Panel A — precision, recall and F1 at the headline best-F1 cut. */
	panelALeft: 202,
	panelARight: 512,
	/** Panel A readouts are left-anchored here; usable gutter is 520 → 612. */
	panelAReadoutX: 520,
	panelAReadoutRight: 612,
	/** Panel B — Δ error-F1 against the paper's own model. */
	panelBLeft: 646,
	panelBRight: 866,
	/** Panel B readouts are left-anchored here; usable gutter is 874 → 960. */
	panelBReadoutX: 874,
	labelFontPx: 9,
	subFontPx: 7.5,
	readoutFontPx: 8,
	/** Measured advance of the mono face at 9px, in user units per character. */
	monoUnitsPerChar: 5.4186,
	/** The same face at 7.5px and 8px: 5.4186 × 7.5/9 and × 8/9. */
	subUnitsPerChar: 4.5155,
	readoutUnitsPerChar: 4.8165,
	laneHeight: 34,
	/** Baseline offsets inside a lane: name above, tau sub-label below. */
	nameOffset: -1,
	subOffset: 10,
	topPad: 56,
	axisPad: 66,
	/** Half-height of the heavy F1 tick in panel A. */
	f1TickHalf: 5,
	/** Half-height of an interval end cap in panel B. */
	simultaneousCap: 5,
	pointwiseCap: 3.5,
	/** A rule annotation flips to the other side rather than clip. */
	rulePad: 5
} as const;

/** ⌊186 / 5.4186⌋. */
export const STATEMENT_ERROR_F1_LABEL_BUDGET_CHARS = 34;
/**
 * The longest name this figure can DRAW, derived from the frozen drawn set so a
 * contract runner re-measures it instead of trusting the comment. Two canonical
 * displays are longer than the budget; neither is in this set, and adding one
 * gates the figure rather than clipping it.
 */
export const STATEMENT_ERROR_F1_LONGEST_DRAWN_LABEL: string = PAPER_LITERAL_ARM_SPECS.filter(
	(spec) => STATEMENT_ERROR_F1_DRAWN_ARM_IDS.includes(spec.id)
).reduce((longest, spec) => (spec.display.length > longest.length ? spec.display : longest), '');
/** ⌊186 / 4.5155⌋ for the 7.5px tau sub-label; longest shipped is "tau 0.8775". */
export const STATEMENT_ERROR_F1_SUB_BUDGET_CHARS = 41;
/** ⌊(612 − 520) / 4.8165⌋; longest shipped panel-A readout is "F1 0.7855". */
export const STATEMENT_ERROR_F1_READOUT_A_BUDGET_CHARS = 19;
/** ⌊(960 − 874) / 4.8165⌋; longest shipped panel-B readout is "reference". */
export const STATEMENT_ERROR_F1_READOUT_B_BUDGET_CHARS = 17;

/** Both panels snap their domain outward to this grid, and tick on it. */
export const STATEMENT_ERROR_F1_AXIS_GRID = 0.05;

export type ErrorF1SeriesId = 'precision' | 'recall' | 'f1' | 'pointwise' | 'simultaneous';

export const STATEMENT_ERROR_F1_SERIES_IDS: readonly ErrorF1SeriesId[] = [
	'precision',
	'recall',
	'f1',
	'pointwise',
	'simultaneous'
] as const;

export interface ErrorF1SeriesStyle {
	id: ErrorF1SeriesId;
	/** CSS custom property name, never a raw hex. */
	strokeVar: string;
	dash: string;
	strokeWidth: number;
	shape: 'square' | 'open-circle' | 'tick' | 'diamond' | 'bracket';
	panel: 'A' | 'B';
	legend: string;
}

/**
 * Each series carries its OWN (stroke token, dash, mark shape), so the figure
 * survives greyscale and colour-vision deficiency and no two marks in a panel
 * share a hue. Every stroke clears 3:1 against --paper #fdfcf8 (WCAG 1.4.11):
 * --ink #1a1a1a ≈ 16:1, --accent #7d2a1a = 9.2:1, --blocked #6f5a16 = 6.5:1,
 * --ink-muted #6a6a6a = 5.3:1.
 */
export const STATEMENT_ERROR_F1_SERIES: Record<ErrorF1SeriesId, ErrorF1SeriesStyle> = {
	precision: {
		id: 'precision',
		strokeVar: 'var(--accent)',
		dash: '',
		strokeWidth: 2,
		shape: 'square',
		panel: 'A',
		legend:
			'error precision — of the statements this model flags, the share that really are errors. This is the mechanism: it is where the margin comes from.'
	},
	recall: {
		id: 'recall',
		strokeVar: 'var(--ink-muted)',
		dash: '',
		strokeWidth: 1.4,
		shape: 'open-circle',
		panel: 'A',
		legend:
			'error recall — of the errors we already know about in these 1,689 statements, the share this model catches'
	},
	f1: {
		id: 'f1',
		strokeVar: 'var(--ink)',
		dash: '',
		strokeWidth: 2.4,
		shape: 'tick',
		panel: 'A',
		legend:
			'error-class F1 — their harmonic mean, so it always lands between the two marks; the span between them is what F1 alone hides'
	},
	pointwise: {
		id: 'pointwise',
		strokeVar: 'var(--accent)',
		dash: '',
		strokeWidth: 2.6,
		shape: 'diamond',
		panel: 'B',
		legend:
			'Δ error-F1 against the random forest from the 2023 INDRA assembly paper, with its pointwise 95% interval'
	},
	simultaneous: {
		id: 'simultaneous',
		strokeVar: 'var(--blocked)',
		dash: '4 2',
		strokeWidth: 1.4,
		shape: 'bracket',
		panel: 'B',
		legend:
			'the same margin, with its interval widened to cover all four reading models at once — the price of having run four and not one'
	}
};

/**
 * THE PLAIN HALF OF EVERY TWIN THIS MODULE EMITS.
 *
 * `statement_error_f1.json` is written for a referee. Its own words are the ones
 * a reader met on this page for seven review waves — "flag a statement as an
 * ERROR iff belief < tau", "each arm's OWN full-panel best-error-F1 cut",
 * "paired fold-stratified bootstrap over the paper's own out-of-fold fold
 * assignment" — because no scan of this repo can see a string that does not
 * exist until the file is read. Each restatement below says the SAME thing in
 * the words the biologist reading it already has. Nothing is softened: every
 * disclosure, every loss and every number in the shipped sentence survives into
 * its plain twin, which is why several of these are long.
 *
 * The shipped sentence is never deleted. It is parsed exactly as before and
 * travels beside its restatement as the audit trail.
 */
const ERROR_F1_PLAIN = {
	metric:
		'How well a model finds WRONG statements, one statement at a time: the balance ' +
		'between how often a flag is right and how many of the wrong statements it catches. ' +
		'Each model is measured at its own best cutoff.',
	decisionRule: 'A statement is flagged as wrong when its belief score falls below the cutoff.',
	positiveClassNote:
		'What is being counted is a WRONG statement, not a right one. The pair scored for each ' +
		'statement is: it is truly wrong when the label released with the 2023 paper says it is ' +
		'not correct, and it is called wrong when its belief falls below the cutoff. 73.2% of ' +
		'these 1,689 statements are right, so scoring the right ones flatters everything: a model ' +
		'that says “all correct” and never flags anything scores 0.845 on the right-statement ' +
		'version of this measure and 0.000 on the wrong-statement version, so the right-statement ' +
		'version must NOT be the headline. The right-statement precision, catch rate and score at ' +
		'the SAME cutoff are printed beside every row along with every raw count, so that view is ' +
		'recoverable rather than deleted.',
	thresholdRule:
		'Each model’s cutoff is drawn from the model’s OWN distinct scores: the one whose flagged ' +
		'set — every statement scoring strictly below it — makes its own wrong-statement score as ' +
		'high as it will go across all 1,689 statements. Where several cutoffs tie, the SMALLEST ' +
		'is taken, so the rule depends only on the scores and not on the order the rows arrived ' +
		'in. Every model is cut at its own number — no cutoff is shared, and none is carried ' +
		'across from one model to another.',
	oracleDisclosure:
		'Every cutoff here was chosen with the answers already in hand, on the same 1,689 ' +
		'statements it is then scored on. Nobody could have picked it before the curation was ' +
		'done, and none of it is checked on statements held back. That help is real, and it goes ' +
		'to the random forest rather than to the reading models: its scores are near-continuous, ' +
		'so the search ' +
		'had 1,546 candidate cutoffs to optimise over, against 475–498 for the three reading ' +
		'models that win and 420 for Gemma 4 E2B. The side handed the finest choice still loses ' +
		'by 0.1416 to GLM-5. Read this as a comparison at one chosen operating point, not as a ' +
		'result on held-back statements. It covers the headline cutoff ONLY; the matched-catch-' +
		'rate block uses a second cutoff, chosen by a different rule, disclosed separately.',
	matchedRecallRule:
		'This is a SECOND cutoff, and it is chosen with the answers in hand just like the ' +
		'headline one: the cheapest cut that still catches at least 60% of the 452 known wrong ' +
		'statements. The disclosure above therefore applies to it too — by a DIFFERENT rule ' +
		'(catch rate, not best score), which is why it is stated separately. Because each ' +
		'model’s scores come in a small number of steps, the catch rate a model actually ' +
		'achieves overshoots the 60% target by whatever its own steps allow, and across these ' +
		'models the achieved catch rates run from 0.6018 to 0.8628 — 26.1 points apart, over 6 ' +
		'different values in 6 rows. Subtracting one fixed comparison row from rows sitting at ' +
		'six different catch rates is not a like-for-like comparison and is not what this block ' +
		'reports. Read the MATCHED number: for each row the random forest is re-cut at that ' +
		'row’s own achieved catch rate, and the re-cut row is printed beside it so the match can ' +
		'be checked rather than taken on trust. The unmatched number — both sides at their own ' +
		'60% cut — is kept because it is a real quantity, and is named for what it is, because ' +
		'it flatters whichever model overshot furthest.',
	modalThresholdNote:
		'Gemma 4 26B, Gemma 4 31B and GLM-5 all land on the SAME cutoff, 0.6500 — the most common ' +
		'non-zero belief INDRA’s combination rule returns on these 1,689 statements, and the ' +
		'smallest it can return from a single surviving piece of evidence. No reading model’s ' +
		'score falls between 0 and that value, so at ' +
		'that cutoff the statements they flag are exactly the ones whose evidence they rejected ' +
		'outright. Their “best” cutoff is therefore not tuned at all: it is the block they had ' +
		'already thrown out, and being allowed to see the answers bought them nothing. Gemma 4 ' +
		'E2B lands at 0.8775; the random forest at 0.6541. This is stated because it is the ' +
		'strongest objection anyone could raise to these numbers, and it turns out to cut the ' +
		'other way.',
	bootstrapDesign:
		'The intervals come from re-drawing the statements at random with replacement, within ' +
		'each of the 10 folds — the 10 groups the statements were split into — assigned in 2023, ' +
		'so every redraw keeps that fold make-up. ONE ' +
		'redraw is shared by every model, so the models’ margins are drawn ' +
		'together and the widened interval carries how much they move together instead of ' +
		'assuming it away. The cutoffs are held FIXED at the values fitted over all 1,689 ' +
		'statements and are not re-chosen inside each redraw: the interval is on the margin at a ' +
		'stated operating point, not on the search that found the cutoff. Same seed, same number ' +
		'of redraws and same design as scripts/compute_paper_robustness.py and ' +
		'scripts/compute_statement_review_queue.py.',
	multiplicityNote:
		'The frozen run plan lists all four reading models and names none of them as the one to ' +
		'be confirmed, so an interval wide enough to cover all four at once is the fair thing to ' +
		'ask for. The four move together from redraw to redraw, so covering all four costs far ' +
		'less width than a Bonferroni correction would. INDRA CoGEx hybrid sits outside that ' +
		'group of four and is quoted with its own single-model interval only.',
	reconciliationNote:
		'TWO cutoff rules over one quantity, both recorded rather than one being trusted. The ' +
		'review-list figure reaches its cutoff by catch rate and flags every statement at or ' +
		'below it; this figure maximises the wrong-statement score and flags every statement ' +
		'strictly below. They are NOT independent derivations: same statements, same labels, ' +
		'same scores — and for the three larger reading models the two rules select the SAME ' +
		'statements (both land on the block that model rejected outright), so the gap is exactly ' +
		'zero. That is a cross-check of two pieces of code over one set of flags — worth ' +
		'recording, but it is not a second measurement and it cannot corroborate the flag set ' +
		'itself. For the random forest the two rules land on genuinely different cutoffs, which ' +
		'is the only row here where the gap carries information; it is the largest one, and it ' +
		'is why a tolerance is stated instead of assumed to be zero.',
	/**
	 * The four remaining shipped sentences this module reads. `multiplicity.method`
	 * was twinned nowhere while its own sibling `note` was — so the one line that
	 * says HOW one interval is made to cover four models reached a reader as
	 * "studentized max-t over the shared paired-bootstrap draws".
	 */
	multiplicityMethod:
		'The width is set by the largest standardised margin seen across the four reading models ' +
		'on each shared redraw, so ONE interval covers all four at once.',
	labelProvenance: 'the labels released with the 2023 paper, unmodified',
	panelOrdering:
		'the statements taken in the order of their sorted content hashes, the same order the ' +
		'average-precision head-to-head uses',
	referenceDescription:
		'a re-run of the released random-forest code from the 2023 INDRA assembly paper ' +
		'(sorgerlab/indra_assembly_paper), with every statement scored by a copy of the model ' +
		'that never saw it, on the 10 folds released with it'
} as const;

/**
 * `caveats[]` in shipped order, each pinned to its restatement by a verbatim
 * fragment. Nine sentences, none of them decorative: two of them are the losses.
 */
const ERROR_F1_CAVEAT_TWINS: readonly AnchoredProse[] = [
	{
		artifactAnchor: 'NEVER published a decision metric',
		plain:
			'The 2023 paper published no decision measure, no cutoff and no statistical test of ' +
			'any kind — only trapezoidal PR-AUC, averaged over 10 folds with a spread, across 59 ' +
			'rows. The wrong-statement score is a new derivation, on those same statements, those ' +
			'released labels and the folds that came with them. It was never reported, and it is ' +
			'not a number anyone lost on: it is a number nobody ran.'
	},
	{
		artifactAnchor: 'EVERY threshold here is fitted',
		plain:
			'EVERY cutoff here is chosen and scored on the same 1,689 statements — the headline one ' +
			'AND the catch-rate one, which is a second cutoff chosen by a second rule. Both are ' +
			'disclosed above, separately. On the headline cutoff the advantage that buys runs ' +
			'toward the random forest, which gets the finest-grained search, and away from the ' +
			'three reading models, whose best cutoff is simply the block they had already rejected.'
	},
	{
		artifactAnchor: 'DECISION metric at ONE operating point',
		plain:
			'A score at a chosen cutoff is a DECISION measure at ONE operating point. It does not ' +
			'supersede average precision or AUROC and it retracts neither: it answers a different ' +
			'question — how good is the list this model hands a curator — and both views stay on ' +
			'the page. The catch-rate block gives a second cutoff on the same models, so the ' +
			'headline is not the only operating point on record. Read its MATCHED margin, in which ' +
			'the random forest is re-cut at each row’s OWN achieved catch rate: at a 60% target the ' +
			'achieved catch rates span 26.1 points, so the unmatched subtraction kept beside it is ' +
			'larger for every model and flips Gemma 4 E2B’s sign. The matched margins are Gemma 4 ' +
			'E2B −0.0059, Gemma 4 26B +0.1391, Gemma 4 31B +0.1368, GLM-5 +0.1451. They are single ' +
			'values: no interval, no redraws and no correction for having run four models is ' +
			'computed at this second cutoff, so the tested claim remains the headline one.'
	},
	{
		artifactAnchor: 'ERROR is the positive class here BECAUSE',
		plain:
			'Wrong statements are what is counted here BECAUSE 73.2% of these statements are right. ' +
			'Scoring the right ones is dominated by that majority and would flatter every model; ' +
			'the right-statement precision, catch rate and score at the same cutoff are printed ' +
			'beside every row, so that view is recoverable, not deleted.'
	},
	{
		artifactAnchor: 'Gemma 4 E2B LOSES',
		plain:
			'Gemma 4 E2B LOSES: its margin is negative and its own interval includes zero. ' +
			'“Reading models beat the random forest at finding wrong statements” is therefore false ' +
			'as a blanket claim, and true of the three larger models only. The correction covers ' +
			'all four, not only the winners.'
	},
	{
		artifactAnchor: 'noisy-OR applied to the evidence the reader KEPT',
		plain:
			'The reading models are INDRA’s own combination rule applied to the evidence the model ' +
			'kept, so what is compared is belief models over a shared way of combining evidence, ' +
			'and the reading model’s contribution is the filtering. They are also not zero-shot: ' +
			'each call carries 14 hand-written example pairs.'
	},
	{
		artifactAnchor: '2023 feature matrix',
		plain:
			'The random forest is scored on the 2023 feature matrix and the folds that came with ' +
			'it, because that is what re-running the released code produces; the reading models ' +
			'are scored on current INDRA evidence. The models compare cleanly to each other on the ' +
			'same statements and the same labels, but only loosely to the table published in 2023.'
	},
	{
		artifactAnchor: 'INDRA CoGEx hybrid is carried for completeness',
		plain:
			'INDRA CoGEx hybrid is carried for completeness and is NOT inside the group of four the ' +
			'widened interval covers: its own bundle manifest calls it descriptive and ' +
			'non-confirmatory, and the frozen run plan lists the four reading models and nothing ' +
			'else. Its interval is a single-model one.'
	},
	{
		artifactAnchor: 'adds no data',
		plain:
			'This adds no data. It scores the same predictions the ordering head-to-head and the ' +
			'review list already score; the cross-check below pins it to the review list’s ' +
			'separately-ruled operating point to within 0.0080. Separately ruled, not independent: ' +
			'same statements, same labels, same scores, and on the three winning models the two ' +
			'rules select the same statements.'
	}
];

/** Which panel-fitted rule produced a tau, and the shipped text that discloses it. */
export type ErrorF1ThresholdRuleId = 'best-f1' | 'target-recall-60' | 'review-queue';

export interface ErrorF1ThresholdRule {
	id: ErrorF1ThresholdRuleId;
	/**
	 * On-screen name of the rule, in the reader's words rather than ours: a "cut"
	 * is a score cutoff, the "panel" is the 1,689 statements, an "arm" is a model.
	 * Ours, not the artifact's — `rule` and `oracle` below are the shipped bytes
	 * and are printed verbatim beside this.
	 */
	name: string;
	/** SHIPPED rule text — how this tau was chosen. Always `ruleProse.shipped`. */
	rule: string;
	/** SHIPPED oracle text — what it costs that it was chosen on this panel. */
	oracle: string;
	/**
	 * The same rule text with its plain restatement — `ruleProse.shipped === rule`,
	 * byte for byte. The flat field is kept so nothing that reads it today changes
	 * behaviour; a render site should print `ruleProse.plain` and leave `rule`
	 * behind the page's verification boundary.
	 */
	ruleProse: ShippedProse;
	/** The same for the disclosure — `oracleProse.shipped === oracle`. */
	oracleProse: ShippedProse;
}

/** One arm at one cut. Every field is READ; none is recomputed. */
export interface ErrorF1OperatingPoint {
	tau: number;
	flagged: number;
	errorPrecision: number;
	errorRecall: number;
	errorF1: number;
	tp: number;
	fp: number;
	fn: number;
	tn: number;
	accuracy: number;
	correctPrecision: number;
	correctRecall: number;
	correctF1: number;
	/** True when the cut's flag set is exactly the arm's own belief-0 block. */
	flagSetIsTheArmsZeroPile: boolean;
}

export interface ErrorF1MatchedRecall {
	targetErrorRecall: number;
	/** This arm at its OWN 60% cut. */
	point: ErrorF1OperatingPoint;
	/** The reference RE-CUT at this row's achieved error recall. */
	referenceAtThisRowsRecall: ErrorF1OperatingPoint;
	referenceErrorRecallAtThisRow: number;
	referenceErrorF1AtThisRow: number;
	referenceRecallOvershoot: number;
	/**
	 * THE MATCHED DELTA — the only one this surface may present as recall-matched.
	 * SHIPPED as `delta_error_f1_at_matched_recall`.
	 */
	deltaAtMatchedRecall: number;
	/**
	 * NOT recall-matched: both sides at their own 60% cut. SHIPPED as
	 * `delta_error_f1_each_side_at_its_own_target_cut`. Render it only under a name
	 * that says so.
	 */
	deltaEachSideAtItsOwnTargetCut: number;
	/** Derived comparison of the two SHIPPED deltas: do they disagree in sign? */
	signsDisagree: boolean;
}

/**
 * Where an arm stands against the reference — THREE-WAY, never two. The page's
 * one classifier, `standingOfBounds` in paper-literal.ts; re-exported here so a
 * consumer of this figure has one import to make.
 *
 * A two-way split on `!winsSimultaneously` merges "not significant" with
 * "significantly worse" and then has to assert one of them in prose; this page
 * shipped exactly that and rendered a −0.1416 margin whose interval sat entirely
 * below zero as "spans zero — level with the paper's model". The classes are
 * disjoint and exhaustive so the sentence can be selected by the data:
 *   · `ahead`            — the deciding interval is strictly above zero
 *   · `not-significant`  — it contains zero, whatever the point estimate's sign
 *   · `behind`           — it is strictly below zero
 */
export type { Standing };

/** The paired margin at the headline cut, with both bands. */
export interface ErrorF1Delta {
	delta: number;
	bootstrapMean: number;
	ciLow: number;
	ciHigh: number;
	bootstrapSe: number;
	tStatistic: number;
	pGreaterThanZero: number;
	nValidResamples: number;
	/** Null for an arm outside the max-t family; it is quoted pointwise only. */
	simLow: number | null;
	simHigh: number | null;
	/**
	 * The band's own three-way class, or NULL for an arm outside the max-t family
	 * — null, never 'not-significant', because "no band was measured" and "a band
	 * was measured and contains zero" are different claims.
	 */
	simultaneousStanding: Standing | null;
	/** The simultaneous band lies entirely ABOVE zero. Sign-aware by construction. */
	winsSimultaneously: boolean;
	/** The pointwise interval lies entirely ABOVE zero. Sign-aware by construction. */
	winsPointwise: boolean;
	/** ahead / not-significant / behind, decided on `standingBasis`. */
	standing: Standing;
	/**
	 * The SAME three-way class computed on the POINTWISE interval alone, for the
	 * clauses that print that interval beside a standing decided on the
	 * simultaneous band. A two-way `{#if excludesZeroPointwise}` there is the
	 * outer defect one clause deeper: it printed "pointwise […] does not [span
	 * zero], so the family-wide correction is what makes this a tie" for
	 * [−0.0381, −0.0020] and for [+0.1090, +0.1744] alike — the same words for
	 * opposite signs, with the sign itself never stated. Equal to `standing` when
	 * `standingBasis` is 'pointwise'.
	 */
	pointwiseStanding: Standing;
	/**
	 * WHICH interval decided `standing` — the simultaneous band for an arm inside
	 * the max-t family, the pointwise interval for one outside it, where no band
	 * exists. A sentence that classifies on one interval and prints the other is
	 * unfalsifiable on screen, so the bounds that decided the class travel with it
	 * and the component prints THESE.
	 */
	standingBasis: 'simultaneous' | 'pointwise';
	standingLow: number;
	standingHigh: number;
}

export interface ErrorF1Lane {
	id: string;
	/** FROZEN `point_metrics` join key. Never rendered — render `display`. */
	label: string;
	/** On-screen name, from the canonical arm spec. Rendered. */
	display: string;
	/** The arm's join key into `statement_review_queue.json`, or null. */
	reviewQueueModelKey: string | null;
	role: string;
	inMaxTFamily: boolean;
	isReference: boolean;
	/** Candidate cuts the best-F1 search had to choose from. The oracle's size. */
	distinctScores: number;
	operating: ErrorF1OperatingPoint;
	matched: ErrorF1MatchedRecall;
	/** Null on the reference lane: it is the baseline, not a contender. */
	delta: ErrorF1Delta | null;
	y: number;
	nameY: number;
	subY: number;
	subLabel: string;
	readoutA: string;
	readoutB: string;
	titleA: string;
	titleB: string;
}

export interface ErrorF1Panel {
	id: string;
	n: number;
	nErrors: number;
	nCorrect: number;
	errorBaseRate: number;
	correctBaseRate: number;
	labelField: string;
	labelProvenance: string;
	/** `labelProvenance` with its plain restatement — `shipped` is byte-identical. */
	labelProvenanceProse: ShippedProse;
	nFolds: number;
	ordering: string;
	/** `ordering` with its plain restatement — `shipped` is byte-identical. */
	orderingProse: ShippedProse;
}

export interface ErrorF1Multiplicity {
	family: string[];
	familySize: number;
	familyAlpha: number;
	method: string;
	/** `method` with its plain restatement — `methodProse.shipped === method`. */
	methodProse: ShippedProse;
	criticalValue: number;
	pointwiseNormalCriticalValue: number;
	bonferroniCriticalValue: number;
	nExcludingZeroSimultaneous: number;
	runPlanPath: string;
	runPlanSha256: string;
	runPlanStages: string[];
	note: string;
	/** `note` with its plain restatement — `noteProse.shipped === note`. */
	noteProse: ShippedProse;
}

/** One row of the cross-check against the separately-ruled review-queue cut. */
export interface ErrorF1ReconciliationRow {
	id: string;
	display: string;
	reviewQueueModelKey: string;
	reviewQueueTau: number;
	reviewQueueErrorPrecision: number;
	reviewQueueErrorRecall: number;
	reviewQueueErrorF1: number;
	reviewQueueTp: number;
	reviewQueueFp: number;
	thisArtifactTau: number;
	thisArtifactErrorF1: number;
	residual: number;
	sameFlagSet: boolean;
}

export interface ErrorF1Reconciliation {
	source: string;
	sha256: string;
	sourceTargetRecall: number;
	tolerance: number;
	worstResidual: number;
	panelN: number;
	panelNErrors: number;
	rows: ErrorF1ReconciliationRow[];
	note: string;
	/** `note` with its plain restatement — `noteProse.shipped === note`. */
	noteProse: ShippedProse;
	/** The review-queue tau is a THIRD rule; it travels with these numbers. */
	thresholdRule: ErrorF1ThresholdRule;
}

export interface ErrorF1Axis {
	left: number;
	right: number;
	min: number;
	max: number;
	ticks: number[];
}

/**
 * EVERY SHIPPED SENTENCE THIS FIGURE CARRIES, IN ONE PLACE, each with the plain
 * restatement a reader is handed instead.
 *
 * One bag rather than a plain twin scattered beside each flat field, because the
 * question a render site asks is "which of these strings do I have a plain half
 * for" and the answer should be a single list it cannot read past. Three of these
 * are ALSO reachable through the structure that owns them — `thresholdRule` and
 * `oracleDisclosure` through `headlineThresholdRule` / `matchedThresholdRule` /
 * `reconciliation.thresholdRule`, `multiplicityNote` through `multiplicity`,
 * `reconciliationNote` through `reconciliation` — and they are the SAME objects,
 * not copies, so the rule still travels with the number it governs.
 *
 * `oracleDisclosure` is one string for all three cutoff rules because the
 * artifact ships one: the disclosure covers the headline cut and says in its own
 * last sentence that the other two are disclosed separately.
 */
export interface ErrorF1Prose {
	metric: ShippedProse;
	decisionRule: ShippedProse;
	positiveClassNote: ShippedProse;
	/** The headline cut's rule. */
	thresholdRule: ShippedProse;
	/** The 60%-target cut's rule. */
	matchedRecallRule: ShippedProse;
	/** Shared by all three cutoff rules — the artifact ships exactly one. */
	oracleDisclosure: ShippedProse;
	modalThresholdNote: ShippedProse;
	bootstrapDesign: ShippedProse;
	multiplicityNote: ShippedProse;
	/** How ONE interval is made to cover all four reading models. */
	multiplicityMethod: ShippedProse;
	/** Whose labels these are, and that we did not touch them. */
	labelProvenance: ShippedProse;
	/** What order the statements are taken in. */
	panelOrdering: ShippedProse;
	/** What the comparison model is, and how it was scored. */
	referenceDescription: ShippedProse;
	/** Also the review-queue cut's own rule text. */
	reconciliationNote: ShippedProse;
	/** Index-aligned with `caveats`, pinned to it by a verbatim fragment. */
	caveats: ShippedProse[];
}

export interface ErrorF1Figure {
	lanes: ErrorF1Lane[];
	reference: ErrorF1Lane;
	panelA: ErrorF1Axis;
	panelB: ErrorF1Axis;
	height: number;

	/** Vertical rule in panel A at the reference's own error-F1. */
	referenceRuleValue: number;
	referenceRuleLabel: string;
	referenceRuleLabelFits: boolean;
	/** Vertical rule in panel B at zero. */
	zeroRuleLabel: string;
	zeroRuleLabelFits: boolean;

	/**
	 * Both right-anchored strings of every lane — the name and its tau sub-label —
	 * in drawn order, emitted VERBATIM into `<desc>`. The gutter arithmetic above
	 * says they fit; this says what they are even if a face substitution ever made
	 * them not fit.
	 */
	descLabels: string;

	metric: string;
	positiveClass: string;
	positiveClassNote: string;
	decisionRule: string;
	/** The headline cut's rule + oracle. Travels with every panel-A number. */
	headlineThresholdRule: ErrorF1ThresholdRule;
	/** The 60%-target cut's rule + oracle. Travels with every matched number. */
	matchedThresholdRule: ErrorF1ThresholdRule;
	modalThresholdNote: string;

	panel: ErrorF1Panel;
	multiplicity: ErrorF1Multiplicity;
	reconciliation: ErrorF1Reconciliation;

	referenceDescription: string;
	/** `referenceDescription` with its plain restatement — `shipped` is byte-identical. */
	referenceDescriptionProse: ShippedProse;
	referenceMethodString: string;

	seed: number;
	nBootstrap: number;
	bootstrapDesign: string;
	caveats: string[];
	/**
	 * The plain half of every string above that came off the artifact. Each flat
	 * field is byte-identical to its `prose` twin's `shipped`; the twin is the one
	 * to render.
	 */
	prose: ErrorF1Prose;
	metricImplementation: string;
	generatedBy: string;

	/** Censuses over SHIPPED values, so the prose cannot drift from the figure. */
	nWinsSimultaneously: number;
	nWinsPointwise: number;
	nInFamily: number;
	/**
	 * Lowest and highest error precision among the simultaneous winners. NULL, not
	 * a placeholder number, when nothing wins — the mechanism sentence must not be
	 * printable from an empty set.
	 */
	winnerPrecisionMin: number | null;
	winnerPrecisionMax: number | null;
	/** Arms whose two matched-recall deltas disagree in sign. */
	signFlipDisplays: string[];
	/**
	 * How the UNMATCHED column compares to the matched one, contender by
	 * contender. The prose asserted it was "larger for every arm"; on the shipped
	 * artifact INDRA CoGEx hybrid's two values are EQUAL to the last bit, because
	 * the reference re-cut at its achieved recall IS the reference's own 60% row.
	 * Three lists rather than one boolean, so the sentence is selected by the data
	 * and stays true if a rerun moves a row into the third case.
	 */
	unmatchedLargerDisplays: string[];
	unmatchedEqualDisplays: string[];
	unmatchedSmallerDisplays: string[];
	/** Arms in the family that share the modal cut and flag their own zero pile. */
	modalCutValue: number | null;
	modalCutDisplays: string[];
}

export interface StatementErrorF1Ok {
	status: 'ok';
	figure: ErrorF1Figure;
	reason: null;
	artifact_path: string;
	/**
	 * NULLABLE on the ok branch too. The server always supplies a digest; the pure
	 * client-side validate path does not, and coalescing that to '' printed an
	 * empty provenance line that read as a real, empty sha. Null is typed so the
	 * compiler finds every render site and each one states the absence.
	 */
	artifact_sha256: string | null;
}

export interface StatementErrorF1Unavailable {
	status: 'unavailable';
	figure: null;
	reason: string;
	artifact_path: string;
	artifact_sha256: string | null;
}

export type StatementErrorF1Load = StatementErrorF1Ok | StatementErrorF1Unavailable;

export interface StatementErrorF1Context {
	artifactPath?: string;
	artifactSha256?: string;
}

type UnknownRecord = Record<string, unknown>;

function fail(context: string, message: string): never {
	throw new Error(`${context}: ${message}`);
}

function record(value: unknown, context: string): UnknownRecord {
	if (value === null || typeof value !== 'object' || Array.isArray(value)) {
		fail(context, 'expected an object');
	}
	return value as UnknownRecord;
}

/** Any finite number — deltas and interval bounds are legitimately signed. */
function number(value: unknown, context: string): number {
	if (typeof value !== 'number' || !Number.isFinite(value)) {
		fail(context, 'expected a finite number');
	}
	return value;
}

function unit(value: unknown, context: string): number {
	const parsed = number(value, context);
	if (parsed < 0 || parsed > 1) fail(context, 'expected a number in [0, 1]');
	return parsed;
}

function positiveInteger(value: unknown, context: string): number {
	if (typeof value !== 'number' || !Number.isInteger(value) || value < 1) {
		fail(context, 'expected a positive integer');
	}
	return value;
}

function count(value: unknown, context: string): number {
	if (typeof value !== 'number' || !Number.isInteger(value) || value < 0) {
		fail(context, 'expected a non-negative integer');
	}
	return value;
}

function text(value: unknown, context: string): string {
	if (typeof value !== 'string' || value.trim().length === 0) {
		fail(context, 'expected a non-empty string');
	}
	return value;
}

function boolean(value: unknown, context: string): boolean {
	if (typeof value !== 'boolean') fail(context, 'expected a boolean');
	return value;
}

/**
 * Exactly `want`, or the figure gates. Used on the artifact's own self-checks:
 * a weakened rerun should take the figure down, never quietly change what a
 * disclosure on screen is claiming.
 */
function exactly(value: unknown, want: boolean, context: string): boolean {
	const parsed = boolean(value, context);
	if (parsed !== want) fail(context, `expected ${want}`);
	return parsed;
}

function budget(value: string, chars: number, context: string): string {
	if (value.length > chars) {
		fail(context, `"${value}" is ${value.length} chars; the gutter budget is ${chars}`);
	}
	return value;
}

/**
 * Signed, four decimals, ASCII sign. ASCII '+'/'-' on purpose: the readout
 * budgets are measured in characters against a MEASURED mono advance, and a
 * typographic minus that fell back to another face would be a different width —
 * silently breaking the one guard that keeps this text from clipping.
 */
export function fmtDelta(value: number): string {
	return `${value >= 0 ? '+' : '-'}${Math.abs(value).toFixed(4)}`;
}

/** Four decimals: the precision this artifact's F1 values are argued at. */
export function fmt4(value: number): string {
	return value.toFixed(4);
}

/** Whole percent, for the precision/recall sentences. */
export function fmtPct(value: number): string {
	return `${Math.round(value * 100)}%`;
}

/** Ticks on the shared grid across a domain, endpoints included. */
function ticksOf(min: number, max: number, grid: number): number[] {
	const out: number[] = [];
	const steps = Math.round((max - min) / grid);
	for (let i = 0; i <= steps; i += 1) out.push(Number((min + i * grid).toFixed(10)));
	return out;
}

/** Snap outward to the tick grid so no drawn mark lands flush on an axis end. */
function snapDomain(values: number[], grid: number): { min: number; max: number } {
	const min = Math.floor(Math.min(...values) / grid) * grid;
	const max = Math.ceil(Math.max(...values) / grid) * grid;
	return { min: Number(min.toFixed(10)), max: Number(max.toFixed(10)) };
}

/**
 * Parse one operating point. Precision, recall and F1 are READ — the rendered
 * values are the shipped ones — and each is GATED against the counts drawn
 * beside it, the same identity discipline `parseDelta` applies to the headline
 * margin.
 *
 * WHY THE RATES ARE GATED AND NOT MERELY READ. The count checks alone (the flag
 * set is tp+fp, the errors are tp+fn, the four cells cover the panel) are
 * self-consistent with ANY rates shipped next to them: `error_precision := 0.95`
 * with tp/fp/fn untouched passed every one of them, and the figure then rendered
 * "the winning gates 75%–95%" and drew the F1 tick OUTSIDE the precision–recall
 * span its own `<desc>` promises it always lands between — because F1 stayed
 * honest while precision did not. Panel A draws all three marks and the prose
 * quotes the precision range, so all three are tied to the counts.
 *
 * The definitions are `metrics.py::_pr_f1_acc`'s, zero-denominator convention
 * included (0, not NaN), so this gate agrees with the generator rather than
 * introducing a second convention. It holds byte-exactly on the shipped
 * artifact — residual 0.000e+00 across all 18 operating points, arms, matched
 * cuts and re-cut references alike — so it costs nothing.
 */
function parseOperatingPoint(
	raw: unknown,
	context: string,
	panelN: number,
	panelErrors: number
): ErrorF1OperatingPoint {
	const point = record(raw, context);
	// "a threshold removed" must gate: tau is required, finite and in range.
	const tau = unit(point.tau, `${context}.tau`);
	const flagged = count(point.flagged, `${context}.flagged`);
	const tp = count(point.tp, `${context}.tp`);
	const fp = count(point.fp, `${context}.fp`);
	const fn = count(point.fn, `${context}.fn`);
	const tn = count(point.tn, `${context}.tn`);
	if (tp + fp !== flagged) fail(context, 'flagged must equal tp + fp');
	if (tp + fn !== panelErrors) fail(context, 'tp + fn must equal the panel’s error count');
	if (tp + fp + fn + tn !== panelN) fail(context, 'the confusion table must cover the panel');

	const errorPrecision = unit(point.error_precision, `${context}.error_precision`);
	const errorRecall = unit(point.error_recall, `${context}.error_recall`);
	const errorF1 = unit(point.error_f1, `${context}.error_f1`);
	// THE RATE IDENTITY GATE. These three subtractions are never rendered; the
	// shipped rates are. They exist only to prove each shipped rate is the rate of
	// the counts this figure draws beside it.
	const fromCounts = tp + fp > 0 ? tp / (tp + fp) : 0;
	const recallFromCounts = tp + fn > 0 ? tp / (tp + fn) : 0;
	const f1FromCounts =
		fromCounts + recallFromCounts > 0
			? (2 * fromCounts * recallFromCounts) / (fromCounts + recallFromCounts)
			: 0;
	const rateIdentities: readonly [string, number, number][] = [
		['error_precision', errorPrecision, fromCounts],
		['error_recall', errorRecall, recallFromCounts],
		['error_f1', errorF1, f1FromCounts]
	];
	for (const [field, shipped, derived] of rateIdentities) {
		const residual = Math.abs(shipped - derived);
		if (residual > STATEMENT_ERROR_F1_PARITY_TOL) {
			fail(
				`${context}.${field}`,
				`is not this row's own tp/fp/fn (${shipped} against ${derived}; residual ` +
					`${residual.toExponential(3)} exceeds the ${STATEMENT_ERROR_F1_PARITY_TOL} tolerance)`
			);
		}
	}

	return {
		tau,
		flagged,
		errorPrecision,
		errorRecall,
		errorF1,
		tp,
		fp,
		fn,
		tn,
		accuracy: unit(point.accuracy, `${context}.accuracy`),
		correctPrecision: unit(point.correct_precision, `${context}.correct_precision`),
		correctRecall: unit(point.correct_recall, `${context}.correct_recall`),
		correctF1: unit(point.correct_f1, `${context}.correct_f1`),
		flagSetIsTheArmsZeroPile: boolean(
			point.flag_set_is_the_arms_zero_pile,
			`${context}.flag_set_is_the_arms_zero_pile`
		)
	};
}

/**
 * Parse a matched-recall block and PROVE, from the shipped fields alone, that the
 * delta named `delta_error_f1_at_matched_recall` really is the recall-matched
 * one. This is the gate that makes the earlier defect un-renderable: an artifact
 * that emitted the unmatched subtraction under the matched name would fail here
 * and take the whole figure down.
 *
 * `referenceMatchedF1` is the reference's own 60%-cut F1 — the single row the
 * UNMATCHED column subtracts from for every arm, regardless of the recall that
 * arm reached.
 */
function parseMatchedRecall(
	raw: unknown,
	context: string,
	panelN: number,
	panelErrors: number,
	referenceMatchedF1: number
): ErrorF1MatchedRecall {
	const block = record(raw, context);
	const target = unit(block.target_error_recall, `${context}.target_error_recall`);
	const point = parseOperatingPoint(block, context, panelN, panelErrors);
	const referenceAtThisRowsRecall = parseOperatingPoint(
		block.reference_at_this_rows_recall,
		`${context}.reference_at_this_rows_recall`,
		panelN,
		panelErrors
	);
	const referenceErrorRecallAtThisRow = unit(
		block.reference_error_recall_at_this_row,
		`${context}.reference_error_recall_at_this_row`
	);
	const referenceErrorF1AtThisRow = unit(
		block.reference_error_f1_at_this_row,
		`${context}.reference_error_f1_at_this_row`
	);
	const deltaAtMatchedRecall = number(
		block.delta_error_f1_at_matched_recall,
		`${context}.delta_error_f1_at_matched_recall`
	);
	const deltaEachSideAtItsOwnTargetCut = number(
		block.delta_error_f1_each_side_at_its_own_target_cut,
		`${context}.delta_error_f1_each_side_at_its_own_target_cut`
	);

	// The cut must actually deliver its own target.
	if (point.errorRecall < target) {
		fail(context, 'the target-recall cut does not reach its own target recall');
	}
	// The quoted summary fields must be the re-cut row's own numbers.
	if (referenceAtThisRowsRecall.errorRecall !== referenceErrorRecallAtThisRow) {
		fail(context, 'reference_error_recall_at_this_row is not the re-cut row’s recall');
	}
	if (referenceAtThisRowsRecall.errorF1 !== referenceErrorF1AtThisRow) {
		fail(context, 'reference_error_f1_at_this_row is not the re-cut row’s F1');
	}
	// A MATCHED reference must catch at least as many errors as the row it is
	// matched to; otherwise it is not the cheapest cut at that recall.
	if (referenceAtThisRowsRecall.errorRecall < point.errorRecall) {
		fail(context, 'the re-cut reference does not reach this row’s achieved error recall');
	}
	// THE GATE. These subtractions are never rendered; they exist only to prove
	// which quantity each shipped name holds.
	if (
		Math.abs(deltaAtMatchedRecall - (point.errorF1 - referenceAtThisRowsRecall.errorF1)) >
		STATEMENT_ERROR_F1_PARITY_TOL
	) {
		fail(
			`${context}.delta_error_f1_at_matched_recall`,
			'is not this row’s F1 minus the reference RE-CUT at this row’s achieved recall'
		);
	}
	if (
		Math.abs(deltaEachSideAtItsOwnTargetCut - (point.errorF1 - referenceMatchedF1)) >
		STATEMENT_ERROR_F1_PARITY_TOL
	) {
		fail(
			`${context}.delta_error_f1_each_side_at_its_own_target_cut`,
			'is not this row’s F1 minus the reference at the reference’s OWN target cut'
		);
	}

	return {
		targetErrorRecall: target,
		point,
		referenceAtThisRowsRecall,
		referenceErrorRecallAtThisRow,
		referenceErrorF1AtThisRow,
		referenceRecallOvershoot: number(
			block.reference_recall_overshoot,
			`${context}.reference_recall_overshoot`
		),
		deltaAtMatchedRecall,
		deltaEachSideAtItsOwnTargetCut,
		signsDisagree:
			Math.sign(deltaAtMatchedRecall) !== Math.sign(deltaEachSideAtItsOwnTargetCut)
	};
}

/**
 * Parse the headline margin, and PROVE it is the margin between the two numbers
 * this figure draws.
 *
 * THE IDENTITY GATE IS THE POINT OF THIS FUNCTION. Every other check here is
 * self-consistent with whatever delta is shipped: the point estimate against its
 * own interval, `excludes_zero` against its own endpoints, the simultaneous band
 * against the pointwise one. None of them ties `delta_error_f1` to
 * `arm.operating_point.error_f1 − reference.operating_point.error_f1`, so a delta
 * inflated to +0.30 with a matching interval and band passed every gate and the
 * lede read "beat it by +0.1260 to +0.3000" in the same sentence as the 0.7855
 * and 0.6439 that refute it. `parseMatchedRecall` has enforced exactly this class
 * of identity at 1e-9 for the SECONDARY matched delta since the first revision;
 * the primary number was the one left ungated. It holds byte-exactly on the
 * shipped artifact — residual 0.000e+00 for all five arms — so it costs nothing.
 */
function parseDelta(
	arm: UnknownRecord,
	context: string,
	inFamily: boolean,
	armErrorF1: number,
	referenceErrorF1: number
): ErrorF1Delta {
	const delta = number(arm.delta_error_f1, `${context}.delta_error_f1`);
	// THE GATE. This subtraction is never rendered; the shipped delta is. It exists
	// only to prove the shipped delta is the margin between the two drawn F1s.
	if (Math.abs(delta - (armErrorF1 - referenceErrorF1)) > STATEMENT_ERROR_F1_PARITY_TOL) {
		fail(
			`${context}.delta_error_f1`,
			'is not this arm’s drawn error-F1 minus the reference’s drawn error-F1'
		);
	}
	const ciLow = number(arm.ci95_low, `${context}.ci95_low`);
	const ciHigh = number(arm.ci95_high, `${context}.ci95_high`);
	if (ciLow > ciHigh) fail(context, 'ci95_low must not exceed ci95_high');
	// A point estimate outside its own interval means the two were computed over
	// different things; drawing the marker inside a bar it does not belong to
	// would misreport both.
	if (delta < ciLow || delta > ciHigh) {
		fail(context, 'delta_error_f1 lies outside its own 95% interval');
	}
	// The shipped flag is read and gated against its own endpoints — an artifact
	// whose flag disagrees with its bounds is corrupt. It goes no further than this
	// gate: what leaves `parseDelta` is the three-way class.
	if (
		boolean(arm.excludes_zero_pointwise, `${context}.excludes_zero_pointwise`) !==
		(ciLow > 0 || ciHigh < 0)
	) {
		fail(`${context}.excludes_zero_pointwise`, 'must equal ci95_low > 0 || ci95_high < 0');
	}

	let simLow: number | null = null;
	let simHigh: number | null = null;
	let simultaneousStanding: Standing | null = null;
	if (inFamily) {
		simLow = number(arm.simultaneous_low, `${context}.simultaneous_low`);
		simHigh = number(arm.simultaneous_high, `${context}.simultaneous_high`);
		if (simLow > simHigh) fail(context, 'simultaneous_low must not exceed simultaneous_high');
		// The drawing nests the pointwise interval INSIDE the simultaneous band, and
		// that nesting is the whole visual argument ("correcting for four arms widens
		// it"). A band that failed to contain its own pointwise interval would draw a
		// lie, so it gates instead.
		if (simLow > ciLow || simHigh < ciHigh) {
			fail(context, 'the simultaneous band must contain the pointwise interval');
		}
		if (
			boolean(arm.excludes_zero_simultaneous, `${context}.excludes_zero_simultaneous`) !==
			(simLow > 0 || simHigh < 0)
		) {
			fail(
				`${context}.excludes_zero_simultaneous`,
				'must equal simultaneous_low > 0 || simultaneous_high < 0'
			);
		}
		simultaneousStanding = standingOfBounds(simLow, simHigh);
	} else {
		// Outside the family the band is absent, not false: a `false` here would
		// read as "measured and did not clear", which is a different claim.
		if (arm.simultaneous_low !== null || arm.simultaneous_high !== null) {
			fail(context, 'an arm outside the max-t family must carry no simultaneous band');
		}
		if (arm.excludes_zero_simultaneous !== null) {
			fail(
				`${context}.excludes_zero_simultaneous`,
				'must be null outside the max-t family, never false'
			);
		}
	}

	// An arm in the family is judged on the FAMILY-WIDE band, which is the claim
	// the lede makes; one outside it has no band and is judged pointwise. Either
	// way the deciding bounds are carried so the same interval that chose the class
	// is the interval the sentence prints.
	const standingBasis: 'simultaneous' | 'pointwise' = inFamily ? 'simultaneous' : 'pointwise';
	const standingLow = standingBasis === 'simultaneous' ? (simLow as number) : ciLow;
	const standingHigh = standingBasis === 'simultaneous' ? (simHigh as number) : ciHigh;
	// ONE three-way classifier for the whole page, applied to whichever interval a
	// sentence prints. Every sign-aware class on this figure comes through
	// `standingOfBounds`, so there is a single place for the rule to be right and
	// no clause can quietly re-derive it two-way. The point estimate is gated above
	// to lie inside its own interval, and the band is gated to contain that
	// interval, so classifying on the endpoints and classifying on the sign of
	// `delta` agree here by construction.
	const standing = standingOfBounds(standingLow, standingHigh);
	const pointwiseStanding = standingOfBounds(ciLow, ciHigh);

	return {
		delta,
		bootstrapMean: number(arm.delta_bootstrap_mean, `${context}.delta_bootstrap_mean`),
		ciLow,
		ciHigh,
		bootstrapSe: number(arm.bootstrap_se, `${context}.bootstrap_se`),
		tStatistic: number(arm.t_statistic, `${context}.t_statistic`),
		pGreaterThanZero: unit(arm.p_greater_than_zero, `${context}.p_greater_than_zero`),
		nValidResamples: positiveInteger(arm.n_valid_resamples, `${context}.n_valid_resamples`),
		simLow,
		simHigh,
		simultaneousStanding,
		// SIGN-AWARE, and now unmistakably so: a win is the 'ahead' class and
		// nothing else. The predecessor was `delta > 0 && excludesZero…`, whose
		// second conjunct is sign-blind and once marked two significant LOSSES as
		// wins on this page when someone read it on its own.
		winsSimultaneously: simultaneousStanding === 'ahead',
		winsPointwise: pointwiseStanding === 'ahead',
		standing,
		pointwiseStanding,
		standingBasis,
		standingLow,
		standingHigh
	};
}

/**
 * The canonical arm spec for an artifact key, with BOTH frozen strings checked:
 * the artifact's `label` must still be the label the rest of the page joins this
 * arm on, and its `display` must still be the name the shipped bytes carry
 * (`spec.artifactDisplay`). Both gate rather than drawing a figure joined on one
 * arm and named for another. Reuses `PAPER_LITERAL_ARM_SPECS` — this page does
 * not get a third display table.
 *
 * The artifact's `display` is checked against `artifactDisplay`, NOT against the
 * on-screen `display`. Those two diverged when the reader arms stopped being
 * called gates on screen: the artifact is sha-pinned and still says "Gemma 4 26B
 * gate", the axis says "Gemma 4 26B reading", and the lane is drawn from
 * `spec.display` either way. Checking the shipped string against the shipped
 * expectation keeps this a live drift guard instead of a tautology.
 */
function specFor(arm: UnknownRecord, context: string) {
	const id = text(arm.key, `${context}.key`);
	const spec = PAPER_LITERAL_ARM_SPECS.find((candidate) => candidate.id === id);
	if (!spec) fail(`${context}.key`, `"${id}" is not a canonical paper-literal arm`);
	const label = text(arm.label, `${context}.label`);
	if (spec.label !== label) {
		fail(`${context}.label`, `frozen join key drifted: "${label}" vs "${spec.label}"`);
	}
	const display = text(arm.display, `${context}.display`);
	if (spec.artifactDisplay !== display) {
		fail(
			`${context}.display`,
			`shipped arm name drifted: "${display}" vs "${spec.artifactDisplay}"`
		);
	}
	return spec;
}

/**
 * The artifact's own self-checks, every one of which governs a claim this figure
 * puts on screen. Gated to `true` so a weakened rerun takes the figure down
 * rather than silently changing what a disclosure means.
 */
const REQUIRED_CHECKS = [
	'every_arm_covers_the_panel_exactly',
	'error_counts_close_on_the_panel',
	'flag_set_equals_tp_plus_fp',
	'matched_recall_cut_delivers_the_target',
	'matched_recall_reference_is_recut_at_each_rows_achieved_recall',
	'family_is_exactly_the_frozen_run_plan_stages',
	'max_t_band_is_between_pointwise_and_bonferroni',
	'review_queue_panel_is_this_panel',
	'reconciles_with_statement_review_queue',
	'family_is_split_into_winners_and_losers',
	'winning_arms_share_one_cut_and_it_is_their_zero_pile'
] as const;

function buildFigure(raw: UnknownRecord): ErrorF1Figure {
	const kind = text(raw.artifact_kind, 'artifact_kind');
	if (kind !== STATEMENT_ERROR_F1_ARTIFACT_KIND) {
		fail('artifact_kind', `expected ${STATEMENT_ERROR_F1_ARTIFACT_KIND}, got ${kind}`);
	}

	const checks = record(raw.checks, 'checks');
	for (const name of REQUIRED_CHECKS) exactly(checks[name], true, `checks.${name}`);

	// ---- panel --------------------------------------------------------------
	const panelRaw = record(raw.panel, 'panel');
	const n = positiveInteger(panelRaw.n, 'panel.n');
	const nErrors = positiveInteger(panelRaw.n_errors, 'panel.n_errors');
	const nCorrect = positiveInteger(panelRaw.n_correct, 'panel.n_correct');
	if (nErrors + nCorrect !== n) fail('panel', 'n_errors + n_correct must equal n');
	const labelProvenanceProse: ShippedProse = {
		shipped: text(panelRaw.label_provenance, 'panel.label_provenance'),
		plain: ERROR_F1_PLAIN.labelProvenance
	};
	const orderingProse: ShippedProse = {
		shipped: text(panelRaw.ordering, 'panel.ordering'),
		plain: ERROR_F1_PLAIN.panelOrdering
	};
	const panel: ErrorF1Panel = {
		id: text(panelRaw.id, 'panel.id'),
		n,
		nErrors,
		nCorrect,
		errorBaseRate: unit(panelRaw.error_base_rate, 'panel.error_base_rate'),
		correctBaseRate: unit(panelRaw.correct_base_rate, 'panel.correct_base_rate'),
		labelField: text(panelRaw.label_field, 'panel.label_field'),
		labelProvenance: labelProvenanceProse.shipped,
		labelProvenanceProse,
		nFolds: positiveInteger(panelRaw.n_folds, 'panel.n_folds'),
		ordering: orderingProse.shipped,
		orderingProse
	};

	// ---- the three tau rules, each with the text that discloses it -----------
	// Absent or empty disclosure gates the figure: this surface may not render a
	// threshold-based number without the rule that produced its threshold.
	// Each is parsed EXACTLY as before — same call, same fail-closed gate — and the
	// plain restatement authored in ERROR_F1_PLAIN is attached beside it. The flat
	// `rule` / `oracle` strings below stay byte-identical to `…Prose.shipped`.
	const thresholdRuleProse: ShippedProse = {
		shipped: text(raw.threshold_rule, 'threshold_rule'),
		plain: ERROR_F1_PLAIN.thresholdRule
	};
	const oracleProse: ShippedProse = {
		shipped: text(raw.oracle_disclosure, 'oracle_disclosure'),
		plain: ERROR_F1_PLAIN.oracleDisclosure
	};
	const matchedRuleProse: ShippedProse = {
		shipped: text(raw.matched_recall_rule, 'matched_recall_rule'),
		plain: ERROR_F1_PLAIN.matchedRecallRule
	};
	const modalThresholdNoteProse: ShippedProse = {
		shipped: text(raw.modal_threshold_note, 'modal_threshold_note'),
		plain: ERROR_F1_PLAIN.modalThresholdNote
	};
	const oracle = oracleProse.shipped;
	const modalThresholdNote = modalThresholdNoteProse.shipped;

	const headlineThresholdRule: ErrorF1ThresholdRule = {
		id: 'best-f1',
		name: 'the score cutoff that gives each model its own best error-F1 across all 1,689 statements',
		rule: thresholdRuleProse.shipped,
		oracle,
		ruleProse: thresholdRuleProse,
		oracleProse
	};
	const matchedThresholdRule: ErrorF1ThresholdRule = {
		id: 'target-recall-60',
		name: 'a second cutoff: the fewest statements a model can flag and still catch 60% or more of the known errors',
		rule: matchedRuleProse.shipped,
		oracle,
		ruleProse: matchedRuleProse,
		oracleProse
	};

	// ---- reference ----------------------------------------------------------
	const referenceRaw = record(raw.reference, 'reference');
	const referenceDescriptionProse: ShippedProse = {
		shipped: text(referenceRaw.description, 'reference.description'),
		plain: ERROR_F1_PLAIN.referenceDescription
	};
	const referenceSpec = specFor(referenceRaw, 'reference');
	if (referenceSpec.id !== PAPER_LITERAL_REFERENCE_ARM_ID) {
		fail('reference.key', `expected ${PAPER_LITERAL_REFERENCE_ARM_ID}, got ${referenceSpec.id}`);
	}
	const referenceOperating = parseOperatingPoint(
		referenceRaw.operating_point,
		'reference.operating_point',
		n,
		nErrors
	);
	const referenceMatchedRaw = record(referenceRaw.matched_recall, 'reference.matched_recall');
	// Parsed twice on purpose: once to learn the reference's OWN target-cut F1 (the
	// row the unmatched column subtracts from), then again through the full gate.
	const referenceMatchedF1 = unit(
		referenceMatchedRaw.error_f1,
		'reference.matched_recall.error_f1'
	);
	const referenceMatched = parseMatchedRecall(
		referenceMatchedRaw,
		'reference.matched_recall',
		n,
		nErrors,
		referenceMatchedF1
	);

	// ---- multiplicity -------------------------------------------------------
	const multiplicityRaw = record(raw.multiplicity, 'multiplicity');
	const familyRaw = multiplicityRaw.family;
	if (!Array.isArray(familyRaw) || familyRaw.length === 0) {
		fail('multiplicity.family', 'expected a non-empty array');
	}
	const family = familyRaw.map((entry, index) => text(entry, `multiplicity.family[${index}]`));
	const familySize = positiveInteger(multiplicityRaw.family_size, 'multiplicity.family_size');
	if (family.length !== familySize) fail('multiplicity.family', 'must have family_size members');
	const criticalValue = number(
		multiplicityRaw.max_t_critical_value,
		'multiplicity.max_t_critical_value'
	);
	const pointwiseZ = number(
		multiplicityRaw.pointwise_normal_critical_value,
		'multiplicity.pointwise_normal_critical_value'
	);
	const bonferroniZ = number(
		multiplicityRaw.bonferroni_critical_value,
		'multiplicity.bonferroni_critical_value'
	);
	// A "simultaneous" band narrower than the pointwise one, or wider than
	// Bonferroni, would mean the correction is not the correction it is labelled
	// as. Both are checkable, so both gate.
	if (!(criticalValue > pointwiseZ)) {
		fail('multiplicity.max_t_critical_value', 'a simultaneous band cannot be narrower than pointwise');
	}
	if (!(criticalValue <= bonferroniZ)) {
		fail('multiplicity.max_t_critical_value', 'max-t cannot exceed the Bonferroni critical value');
	}
	const multiplicityNoteProse: ShippedProse = {
		shipped: text(multiplicityRaw.note, 'multiplicity.note'),
		plain: ERROR_F1_PLAIN.multiplicityNote
	};
	const multiplicityMethodProse: ShippedProse = {
		shipped: text(multiplicityRaw.method, 'multiplicity.method'),
		plain: ERROR_F1_PLAIN.multiplicityMethod
	};
	const runPlan = record(multiplicityRaw.run_plan, 'multiplicity.run_plan');
	const stagesRaw = runPlan.stages;
	if (!Array.isArray(stagesRaw) || stagesRaw.length === 0) {
		fail('multiplicity.run_plan.stages', 'expected a non-empty array');
	}
	const multiplicity: ErrorF1Multiplicity = {
		family,
		familySize,
		familyAlpha: unit(multiplicityRaw.family_alpha, 'multiplicity.family_alpha'),
		method: multiplicityMethodProse.shipped,
		methodProse: multiplicityMethodProse,
		criticalValue,
		pointwiseNormalCriticalValue: pointwiseZ,
		bonferroniCriticalValue: bonferroniZ,
		nExcludingZeroSimultaneous: count(
			multiplicityRaw.n_excluding_zero_simultaneous,
			'multiplicity.n_excluding_zero_simultaneous'
		),
		runPlanPath: text(runPlan.path, 'multiplicity.run_plan.path'),
		runPlanSha256: text(runPlan.sha256, 'multiplicity.run_plan.sha256'),
		runPlanStages: stagesRaw.map((stage, index) =>
			text(stage, `multiplicity.run_plan.stages[${index}]`)
		),
		note: multiplicityNoteProse.shipped,
		noteProse: multiplicityNoteProse
	};

	// ---- arms ---------------------------------------------------------------
	const armsRaw = raw.arms;
	if (!Array.isArray(armsRaw) || armsRaw.length === 0) fail('arms', 'expected a non-empty array');

	interface Parsed {
		spec: (typeof PAPER_LITERAL_ARM_SPECS)[number];
		reviewQueueModelKey: string | null;
		role: string;
		inMaxTFamily: boolean;
		distinctScores: number;
		operating: ErrorF1OperatingPoint;
		matched: ErrorF1MatchedRecall;
		delta: ErrorF1Delta | null;
		isReference: boolean;
	}

	const parsed: Parsed[] = armsRaw.map((entry, index) => {
		const context = `arms[${index}]`;
		const arm = record(entry, context);
		const spec = specFor(arm, context);
		const inMaxTFamily = boolean(arm.in_max_t_family, `${context}.in_max_t_family`);
		const queueKey = arm.review_queue_model_key;
		if (queueKey !== null && typeof queueKey !== 'string') {
			fail(`${context}.review_queue_model_key`, 'expected a string or null');
		}
		const operating = parseOperatingPoint(
			arm.operating_point,
			`${context}.operating_point`,
			n,
			nErrors
		);
		return {
			spec,
			reviewQueueModelKey: queueKey as string | null,
			role: text(arm.role, `${context}.role`),
			inMaxTFamily,
			distinctScores: positiveInteger(arm.distinct_scores, `${context}.distinct_scores`),
			operating,
			matched: parseMatchedRecall(
				arm.matched_recall,
				`${context}.matched_recall`,
				n,
				nErrors,
				referenceMatchedF1
			),
			// The headline margin is gated against the two error-F1 values THIS
			// FIGURE DRAWS: the arm's own operating point and the reference's.
			delta: parseDelta(
				arm,
				context,
				inMaxTFamily,
				operating.errorF1,
				referenceOperating.errorF1
			),
			isReference: false
		};
	});

	// ---- the drawn set is frozen, so a deleted arm cannot just vanish ---------
	// Gating on `multiplicity.family` alone left the out-of-family contender
	// ungated: deleting it returned ok and drew five lanes, removing a disclosed
	// LOSS from the page. The set is checked both ways, and duplicates gate too —
	// a doubled arm would otherwise satisfy a set comparison while drawing twice.
	const byId = new Map(parsed.map((arm) => [arm.spec.id, arm]));
	if (byId.size !== parsed.length) fail('arms', 'the same arm appears more than once');
	for (const expected of STATEMENT_ERROR_F1_CONTENDER_ARM_IDS) {
		if (!byId.has(expected)) fail('arms', `the figure must draw "${expected}"; it is missing`);
	}
	for (const arm of parsed) {
		if (!STATEMENT_ERROR_F1_CONTENDER_ARM_IDS.includes(arm.spec.id)) {
			fail('arms', `"${arm.spec.id}" is not one of this figure's contenders`);
		}
	}

	// Every member of the declared family must actually be drawn, and only family
	// members may carry a simultaneous band. A missing arm gates.
	for (const member of family) {
		const found = byId.get(member);
		if (!found) fail('multiplicity.family', `family member "${member}" is not among the arms`);
		if (!found.inMaxTFamily) {
			fail(`arms[${member}].in_max_t_family`, 'a declared family member must be in the family');
		}
	}
	const inFamily = parsed.filter((arm) => arm.inMaxTFamily);
	if (inFamily.length !== family.length) {
		fail('arms', 'the arms flagged in_max_t_family must be exactly the declared family');
	}
	const nWinsSimultaneously = inFamily.filter((arm) => arm.delta?.winsSimultaneously).length;
	// "How many bands lie strictly one side of zero" — the artifact's own census,
	// asked in a form that cannot be mistaken for "how many won": both signed
	// classes count, and the question is spelled out rather than named after a
	// boolean.
	const nExcludingZero = inFamily.filter(
		(arm) => arm.delta !== null && arm.delta.simultaneousStanding !== null &&
			arm.delta.simultaneousStanding !== 'not-significant'
	).length;
	if (nExcludingZero !== multiplicity.nExcludingZeroSimultaneous) {
		fail(
			'multiplicity.n_excluding_zero_simultaneous',
			'disagrees with the arms that actually exclude zero simultaneously'
		);
	}

	// ---- lanes: the reference interleaved with the arms, by rank -------------
	const referenceLaneSeed: Parsed = {
		spec: referenceSpec,
		reviewQueueModelKey:
			typeof referenceRaw.review_queue_model_key === 'string'
				? referenceRaw.review_queue_model_key
				: null,
		role: 'reference',
		inMaxTFamily: false,
		distinctScores: positiveInteger(referenceRaw.distinct_scores, 'reference.distinct_scores'),
		operating: referenceOperating,
		matched: referenceMatched,
		delta: null,
		isReference: true
	};

	// RANK-INTERLEAVED, not banded: the reference sits at its own error-F1 among
	// the arms, so the three that beat it and the two that do not are read off the
	// order rather than off a caption. Ties break on the canonical id, so the
	// order is a function of the shipped values alone.
	const ordered = [referenceLaneSeed, ...parsed].sort(
		(a, b) => b.operating.errorF1 - a.operating.errorF1 || a.spec.id.localeCompare(b.spec.id)
	);

	const G = STATEMENT_ERROR_F1_GEOMETRY;
	const extentsA: number[] = [];
	const extentsB: number[] = [0];
	for (const arm of ordered) {
		extentsA.push(arm.operating.errorPrecision, arm.operating.errorRecall, arm.operating.errorF1);
		if (arm.delta) {
			extentsB.push(arm.delta.ciLow, arm.delta.ciHigh);
			if (arm.delta.simLow !== null) extentsB.push(arm.delta.simLow);
			if (arm.delta.simHigh !== null) extentsB.push(arm.delta.simHigh);
		}
	}
	const domainA = snapDomain(extentsA, STATEMENT_ERROR_F1_AXIS_GRID);
	const domainB = snapDomain(extentsB, STATEMENT_ERROR_F1_AXIS_GRID);
	if (!(domainB.min < 0 && domainB.max > 0)) {
		fail('arms', 'the zero rule must sit strictly inside the margin axis');
	}
	if (!(domainA.min < referenceOperating.errorF1 && referenceOperating.errorF1 < domainA.max)) {
		fail('reference.operating_point.error_f1', 'the reference rule must sit inside the axis');
	}

	const panelA: ErrorF1Axis = {
		left: G.panelALeft,
		right: G.panelARight,
		min: domainA.min,
		max: domainA.max,
		ticks: ticksOf(domainA.min, domainA.max, STATEMENT_ERROR_F1_AXIS_GRID)
	};
	const panelB: ErrorF1Axis = {
		left: G.panelBLeft,
		right: G.panelBRight,
		min: domainB.min,
		max: domainB.max,
		ticks: ticksOf(domainB.min, domainB.max, STATEMENT_ERROR_F1_AXIS_GRID)
	};

	let y = G.topPad;
	const lanes: ErrorF1Lane[] = ordered.map((arm) => {
		const laneY = y + G.laneHeight / 2;
		y += G.laneHeight;
		const context = `lane[${arm.spec.id}]`;
		const readoutB = arm.delta ? fmtDelta(arm.delta.delta) : 'reference';
		return {
			id: arm.spec.id,
			label: arm.spec.label,
			display: budget(
				arm.spec.display,
				STATEMENT_ERROR_F1_LABEL_BUDGET_CHARS,
				`${context}.display`
			),
			reviewQueueModelKey: arm.reviewQueueModelKey,
			role: arm.role,
			inMaxTFamily: arm.inMaxTFamily,
			isReference: arm.isReference,
			distinctScores: arm.distinctScores,
			operating: arm.operating,
			matched: arm.matched,
			delta: arm.delta,
			y: laneY,
			nameY: laneY + G.nameOffset,
			subY: laneY + G.subOffset,
			subLabel: budget(
				`cutoff ${fmt4(arm.operating.tau)}`,
				STATEMENT_ERROR_F1_SUB_BUDGET_CHARS,
				`${context}.subLabel`
			),
			readoutA: budget(
				`F1 ${fmt4(arm.operating.errorF1)}`,
				STATEMENT_ERROR_F1_READOUT_A_BUDGET_CHARS,
				`${context}.readoutA`
			),
			readoutB: budget(
				readoutB,
				STATEMENT_ERROR_F1_READOUT_B_BUDGET_CHARS,
				`${context}.readoutB`
			),
			titleA:
				`${arm.spec.display} at its own best-error-F1 score cutoff ${fmt4(arm.operating.tau)}: ` +
				`error precision ${fmt4(arm.operating.errorPrecision)}, ` +
				`error recall ${fmt4(arm.operating.errorRecall)}, ` +
				`error-class F1 ${fmt4(arm.operating.errorF1)}; ` +
				`${arm.operating.flagged} statements flagged, ${arm.operating.tp} of the ` +
				`${nErrors} known errors caught, ${arm.operating.fp} correct statements queued`,
			titleB: arm.delta
				? `${arm.spec.display}: ${fmtDelta(arm.delta.delta)} error-F1 against ` +
					`${referenceSpec.display}, pointwise 95% [${fmtDelta(arm.delta.ciLow)}, ` +
					`${fmtDelta(arm.delta.ciHigh)}]` +
					(arm.delta.simLow !== null && arm.delta.simHigh !== null
						? `, widened to cover all ${multiplicity.familySize} reading models at once [` +
							`${fmtDelta(arm.delta.simLow)}, ${fmtDelta(arm.delta.simHigh)}]`
						: ', pointwise only — this model is not one of the four the widened interval covers') +
					`, t ${arm.delta.tStatistic.toFixed(2)}`
				: `${arm.spec.display} is the reference: every margin in this figure is measured against it`
		};
	});

	const referenceLane = lanes.find((lane) => lane.isReference);
	if (!referenceLane) fail('reference', 'the reference lane was not drawn');

	// ---- reconciliation: a THIRD tau rule, carried with its own numbers ------
	const reconciliationRaw = record(raw.reconciliation, 'reconciliation');
	const tolerance = number(reconciliationRaw.tolerance, 'reconciliation.tolerance');
	const worstResidual = number(reconciliationRaw.worst_residual, 'reconciliation.worst_residual');
	// The reconciliation block claims this artifact pins to a separately-ruled
	// operating point. If it does not, the claim is false and the figure gates.
	if (!(worstResidual <= tolerance)) {
		fail(
			'reconciliation',
			`the review-queue cross-check does not hold (${worstResidual} > ${tolerance})`
		);
	}
	const panelMatches = record(reconciliationRaw.panel_matches, 'reconciliation.panel_matches');
	exactly(panelMatches.asserted, true, 'reconciliation.panel_matches.asserted');
	const reconciliationPanelN = positiveInteger(panelMatches.n, 'reconciliation.panel_matches.n');
	const reconciliationPanelErrors = positiveInteger(
		panelMatches.n_errors,
		'reconciliation.panel_matches.n_errors'
	);
	// Two populations silently compared is the failure mode here: the queue could
	// be regenerated on the 1578-row adjudication-safe panel and every residual
	// would still "reconcile".
	if (reconciliationPanelN !== n || reconciliationPanelErrors !== nErrors) {
		fail('reconciliation.panel_matches', 'the review queue was not scored on this panel');
	}

	const reconciliationNoteProse: ShippedProse = {
		shipped: text(reconciliationRaw.note, 'reconciliation.note'),
		plain: ERROR_F1_PLAIN.reconciliationNote
	};
	const reconciliationNote = reconciliationNoteProse.shipped;
	const displayById = new Map(lanes.map((lane) => [lane.id, lane.display]));
	function reconciliationRow(rawRow: unknown, id: string, context: string): ErrorF1ReconciliationRow {
		const row = record(rawRow, context);
		const display = displayById.get(id);
		if (display === undefined) fail(context, `"${id}" is not a drawn lane`);
		return {
			id,
			display,
			reviewQueueModelKey: text(row.review_queue_model_key, `${context}.review_queue_model_key`),
			reviewQueueTau: unit(row.review_queue_tau, `${context}.review_queue_tau`),
			reviewQueueErrorPrecision: unit(
				row.review_queue_error_precision,
				`${context}.review_queue_error_precision`
			),
			reviewQueueErrorRecall: unit(
				row.review_queue_error_recall,
				`${context}.review_queue_error_recall`
			),
			reviewQueueErrorF1: unit(row.review_queue_error_f1, `${context}.review_queue_error_f1`),
			reviewQueueTp: count(row.review_queue_tp, `${context}.review_queue_tp`),
			reviewQueueFp: count(row.review_queue_fp, `${context}.review_queue_fp`),
			thisArtifactTau: unit(row.this_artifact_tau, `${context}.this_artifact_tau`),
			thisArtifactErrorF1: unit(row.this_artifact_error_f1, `${context}.this_artifact_error_f1`),
			residual: number(row.residual, `${context}.residual`),
			sameFlagSet: boolean(row.same_flag_set, `${context}.same_flag_set`)
		};
	}
	const reconciliationArms = record(reconciliationRaw.arms, 'reconciliation.arms');
	const reconciliationRows: ErrorF1ReconciliationRow[] = [
		reconciliationRow(reconciliationRaw.reference, referenceSpec.id, 'reconciliation.reference'),
		...Object.keys(reconciliationArms).map((id) =>
			reconciliationRow(reconciliationArms[id], id, `reconciliation.arms[${id}]`)
		)
	];
	// Every reconciled row must be the same tau this figure draws for that lane,
	// or the cross-check is being reported against a cut nobody can see.
	for (const row of reconciliationRows) {
		const lane = lanes.find((candidate) => candidate.id === row.id);
		if (!lane) fail('reconciliation', `"${row.id}" is not a drawn lane`);
		if (lane.operating.tau !== row.thisArtifactTau) {
			fail('reconciliation', `${row.display}: reconciled tau is not the drawn tau`);
		}
		if (lane.operating.errorF1 !== row.thisArtifactErrorF1) {
			fail('reconciliation', `${row.display}: reconciled F1 is not the drawn F1`);
		}
		if (lane.reviewQueueModelKey !== row.reviewQueueModelKey) {
			fail('reconciliation', `${row.display}: review-queue join key drifted`);
		}
	}

	// `worst_residual` is PROMOTED INTO PROSE: the page's beat-8b lead-in reads
	// "two derivations of one finding under different threshold rules, disagreeing
	// by at most X error-F1". The only gate behind it was `worstResidual <=
	// tolerance` above, which asserts something else entirely — that the shipped
	// scalar is small, not that it IS the disagreement the sentence describes. A
	// `worst_residual` of 0 passed that gate while the reference row disagreed by
	// 0.008, and a row `residual` unrelated to its own two error-F1 values passed
	// it too. So the sentence's two quantities are re-derived here from the rows:
	// each row's residual is |its review-queue error-F1 − its error-F1 on this
	// artifact|, and the shipped worst is the largest of those.
	let derivedWorstResidual = 0;
	for (const row of reconciliationRows) {
		const derived = Math.abs(row.reviewQueueErrorF1 - row.thisArtifactErrorF1);
		if (Math.abs(derived - row.residual) > STATEMENT_ERROR_F1_PARITY_TOL) {
			fail(
				'reconciliation',
				`${row.display}: residual ${row.residual} is not this row’s own two error-F1 values ` +
					`(${row.reviewQueueErrorF1} vs ${row.thisArtifactErrorF1} differ by ${derived})`
			);
		}
		derivedWorstResidual = Math.max(derivedWorstResidual, derived);
	}
	if (Math.abs(derivedWorstResidual - worstResidual) > STATEMENT_ERROR_F1_PARITY_TOL) {
		fail(
			'reconciliation',
			`worst_residual ${worstResidual} is not the largest disagreement between the two ` +
				`threshold rules across the reconciled rows (${derivedWorstResidual})`
		);
	}

	const reconciliation: ErrorF1Reconciliation = {
		source: text(reconciliationRaw.source, 'reconciliation.source'),
		sha256: text(reconciliationRaw.sha256, 'reconciliation.sha256'),
		sourceTargetRecall: unit(
			reconciliationRaw.source_target_recall,
			'reconciliation.source_target_recall'
		),
		tolerance,
		worstResidual,
		panelN: reconciliationPanelN,
		panelNErrors: reconciliationPanelErrors,
		rows: reconciliationRows,
		note: reconciliationNote,
		noteProse: reconciliationNoteProse,
		thresholdRule: {
			id: 'review-queue',
			name: 'a third cutoff: the review queue’s own rule — lower the cutoff until the target share of known errors is caught, flagging every statement whose belief score sits at or below it',
			rule: reconciliationNote,
			oracle,
			ruleProse: reconciliationNoteProse,
			oracleProse
		}
	};

	// ---- censuses over shipped values, so prose cannot drift from the figure --
	const winners = lanes.filter((lane) => lane.delta?.winsSimultaneously);
	const winnerPrecisions = winners.map((lane) => lane.operating.errorPrecision);
	const contenderLanes = lanes.filter((lane) => !lane.isReference);
	const signFlipDisplays = contenderLanes
		.filter((lane) => lane.matched.signsDisagree)
		.map((lane) => lane.display);
	const unmatchedGap = (lane: ErrorF1Lane): number =>
		lane.matched.deltaEachSideAtItsOwnTargetCut - lane.matched.deltaAtMatchedRecall;
	const modalLanes = lanes.filter(
		(lane) => lane.inMaxTFamily && lane.operating.flagSetIsTheArmsZeroPile
	);
	const modalTaus = new Set(modalLanes.map((lane) => lane.operating.tau));
	// The modal-cut sentence claims ONE shared cut. If the arms that flag their own
	// zero pile stop agreeing on tau, the sentence is false, so it is not printed.
	const modalCutValue = modalTaus.size === 1 ? [...modalTaus][0] : null;

	const provenance = record(raw.provenance, 'provenance');
	const caveatsRaw = raw.caveats;
	if (!Array.isArray(caveatsRaw) || caveatsRaw.length === 0) {
		fail('caveats', 'expected a non-empty array');
	}
	const caveats = caveatsRaw.map((entry, index) => text(entry, `caveats[${index}]`));
	// Positional twins, so a reissued artifact that reorders or drops a caveat gates
	// the figure rather than printing restatement N under caveat N+1.
	const caveatProse = pairShippedProse(caveats, ERROR_F1_CAVEAT_TWINS, 'caveats');

	const metricProse: ShippedProse = {
		shipped: text(raw.metric, 'metric'),
		plain: ERROR_F1_PLAIN.metric
	};
	const decisionRuleProse: ShippedProse = {
		shipped: text(raw.decision_rule, 'decision_rule'),
		plain: ERROR_F1_PLAIN.decisionRule
	};
	const positiveClassNoteProse: ShippedProse = {
		shipped: text(raw.positive_class_note, 'positive_class_note'),
		plain: ERROR_F1_PLAIN.positiveClassNote
	};
	const bootstrapDesignProse: ShippedProse = {
		shipped: text(raw.bootstrap_design, 'bootstrap_design'),
		plain: ERROR_F1_PLAIN.bootstrapDesign
	};

	// ---- rule annotations, measured rather than assumed ----------------------
	const spanA = panelA.max - panelA.min;
	const referenceRuleX =
		panelA.left + ((referenceOperating.errorF1 - panelA.min) / spanA) * (panelA.right - panelA.left);
	const referenceRuleLabel = `${fmt4(referenceOperating.errorF1)} — the random forest`;
	const spanB = panelB.max - panelB.min;
	const zeroRuleX = panelB.left + ((0 - panelB.min) / spanB) * (panelB.right - panelB.left);
	const zeroRuleLabel = '0 — level with the random forest';
	// Both annotations float ABOVE the lanes at 8px, so their right bound is the
	// free space beside their panel, not the plot edge: panel A's runs until panel
	// B's readout gutter begins, panel B's until the viewBox ends. Each flips to the
	// other side of its rule rather than clipping — the same device
	// `PaperOwnMetric` and `PaperRobustness` use for their reference labels.

	return {
		lanes,
		reference: referenceLane,
		panelA,
		panelB,
		height: y + G.axisPad,

		referenceRuleValue: referenceOperating.errorF1,
		referenceRuleLabel,
		referenceRuleLabelFits:
			referenceRuleX + G.rulePad + referenceRuleLabel.length * G.readoutUnitsPerChar <=
			G.panelAReadoutRight,
		zeroRuleLabel,
		zeroRuleLabelFits:
			zeroRuleX + G.rulePad + zeroRuleLabel.length * G.readoutUnitsPerChar <= G.width,

		descLabels: lanes.map((lane) => `${lane.display}, ${lane.subLabel}`).join('; '),

		metric: metricProse.shipped,
		positiveClass: text(raw.positive_class, 'positive_class'),
		positiveClassNote: positiveClassNoteProse.shipped,
		decisionRule: decisionRuleProse.shipped,
		headlineThresholdRule,
		matchedThresholdRule,
		modalThresholdNote,

		panel,
		multiplicity,
		reconciliation,

		referenceDescription: referenceDescriptionProse.shipped,
		referenceDescriptionProse,
		referenceMethodString: text(referenceRaw.method_string, 'reference.method_string'),

		seed: positiveInteger(raw.seed, 'seed'),
		nBootstrap: positiveInteger(raw.n_bootstrap, 'n_bootstrap'),
		bootstrapDesign: bootstrapDesignProse.shipped,
		caveats,
		prose: {
			metric: metricProse,
			decisionRule: decisionRuleProse,
			positiveClassNote: positiveClassNoteProse,
			thresholdRule: thresholdRuleProse,
			matchedRecallRule: matchedRuleProse,
			oracleDisclosure: oracleProse,
			modalThresholdNote: modalThresholdNoteProse,
			bootstrapDesign: bootstrapDesignProse,
			multiplicityNote: multiplicityNoteProse,
			multiplicityMethod: multiplicity.methodProse,
			labelProvenance: panel.labelProvenanceProse,
			panelOrdering: panel.orderingProse,
			referenceDescription: referenceDescriptionProse,
			reconciliationNote: reconciliationNoteProse,
			caveats: caveatProse
		},
		metricImplementation: text(
			provenance.metric_implementation,
			'provenance.metric_implementation'
		),
		generatedBy: text(provenance.generated_by, 'provenance.generated_by'),

		nWinsSimultaneously,
		nWinsPointwise: lanes.filter((lane) => lane.delta?.winsPointwise).length,
		nInFamily: inFamily.length,
		winnerPrecisionMin: winnerPrecisions.length ? Math.min(...winnerPrecisions) : null,
		winnerPrecisionMax: winnerPrecisions.length ? Math.max(...winnerPrecisions) : null,
		signFlipDisplays,
		unmatchedLargerDisplays: contenderLanes
			.filter((lane) => unmatchedGap(lane) > 0)
			.map((lane) => lane.display),
		unmatchedEqualDisplays: contenderLanes
			.filter((lane) => unmatchedGap(lane) === 0)
			.map((lane) => lane.display),
		unmatchedSmallerDisplays: contenderLanes
			.filter((lane) => unmatchedGap(lane) < 0)
			.map((lane) => lane.display),
		modalCutValue,
		modalCutDisplays: modalLanes.map((lane) => lane.display)
	};
}

/**
 * Pure, fail-closed validator for the parsed `statement_error_f1.json`. Returns
 * `status:'ok'` with a drawable figure, or `status:'unavailable'` with a reason on
 * any shape or invariant drift. Never throws.
 */
export function validateStatementErrorF1(
	raw: unknown,
	context: StatementErrorF1Context = {}
): StatementErrorF1Load {
	const artifactPath = context.artifactPath ?? '';
	const artifactSha256 = context.artifactSha256 ?? null;
	try {
		return {
			status: 'ok',
			figure: buildFigure(record(raw, 'statement_error_f1')),
			reason: null,
			artifact_path: artifactPath,
			artifact_sha256: artifactSha256
		};
	} catch (error) {
		return {
			status: 'unavailable',
			figure: null,
			reason: error instanceof Error ? error.message : String(error),
			artifact_path: artifactPath,
			artifact_sha256: artifactSha256
		};
	}
}
