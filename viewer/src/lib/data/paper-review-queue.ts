/**
 * Typed data contract for the curator REVIEW QUEUE figure — /paper's beat 2.
 *
 * Source artifact: `data/results/indra_paper_literal_models_20260724/
 * statement_review_queue.json`, emitted by
 * `scripts/compute_statement_review_queue.py`. Every arm on the /paper panel
 * emits a belief in [0,1] for the same 1689 all-sources-specific statements;
 * turning that scalar into WORK needs one more decision the 2023 paper never
 * published — a threshold. The artifact makes that decision explicit and
 * identical across arms (flag as wrong iff belief <= tau; tau is the smallest of
 * the arm's own distinct scores whose flag set reaches the target recall of the
 * known errors), so the only remaining question is how much correct work a
 * curator wades through to get there.
 *
 * This module is import-safe on the client: typed shape, fixed arm table, and a
 * pure `validateReviewQueue()` that THROWS on any shape or arithmetic drift. The
 * server loader (`$lib/server/paper-review-queue`) wraps it in its own try/catch
 * and gates the whole load to `unavailable` rather than drawing a bar whose two
 * segments do not add up to its length.
 *
 * NOTHING here is hard-coded from the numbers: the panel size, the error count,
 * the target recall, every queue/caught/false-alarm count, the bootstrap interval
 * on the operational result, and the method caveats are all read off the
 * artifact. The only constants are the nine arm names (the fixed presentation
 * order, asserted against the artifact), their on-screen display and their
 * provenance.
 */

import {
	anchoredShippedProse,
	keyedShippedProse,
	paperArmColorVar,
	pairShippedProse,
	standingOfBounds,
	type AnchoredProse,
	type PaperArmKind,
	type ShippedProse,
	type Standing
} from './paper-literal.ts';

/** Arithmetic parity tolerance: queue/precision/recall identities. */
export const REVIEW_QUEUE_PARITY_TOL = 1e-9;

/**
 * The two arm families. FROZEN contract strings — the shipped viewer contract
 * keys on the literal `'paper-model'`, so this union cannot be renamed.
 *
 * Read it as "a belief model that scores by formula or by fit" versus "a reader
 * gate", NOT as an attribution: only ONE of the five belief models here is the
 * 2023 paper's. Whose a model actually is lives in `provenance`, and that is the
 * field any sentence about attribution must read.
 */
export type ReviewQueueArmKind = 'paper-model' | 'llm-gate';

/**
 * Whose model this is. The /paper audience wrote the paper, so an arm labelled
 * with the wrong lineage is the one error this figure cannot afford.
 *
 *   · `indra-served`         ships in `indra` and is computed for every
 *                            statement today. NOT the paper's MCMC-refit
 *                            "Belief Orig".
 *   · `paper-published`      the 2023 paper's own released model, re-run by us
 *                            from `sorgerlab/indra_assembly_paper` out of fold.
 *                            A research model: never deployed.
 *   · `our-reimplementation` ours, over INDRA's feature superset. Never
 *                            published by the paper, never served.
 *   · `reader-gate`          INDRA's served scorer over the evidence a reader
 *                            kept. Purely subtractive; not zero-shot.
 */
export type ReviewQueueProvenance =
	| 'indra-served'
	| 'paper-published'
	| 'our-reimplementation'
	| 'reader-gate';

/** One-line gloss per provenance, printed where the arm table is listed. */
export const REVIEW_QUEUE_PROVENANCE_GLOSS: Readonly<Record<ReviewQueueProvenance, string>> = {
	'indra-served': 'ships in indra; served for every statement today',
	'paper-published': 'released with the 2023 paper, re-run here on data it never learned from; never deployed',
	'our-reimplementation': 're-implemented over INDRA’s wider feature set; never published, never served',
	'reader-gate': 'INDRA’s served scorer over the evidence the reader kept'
} as const;

export interface ReviewQueueArmSpec {
	/** Exact `arms[].name` in statement_review_queue.json — the join key. */
	name: string;
	kind: ReviewQueueArmKind;
	/** Whose model this is; asserted against the artifact's own field. */
	provenance: ReviewQueueProvenance;
	/**
	 * What the reader sees. DECOUPLED from `name`, which is a FROZEN artifact join
	 * key (arms[].name, promotion_ceiling.per_arm_overlap, provenance.scores) and
	 * must not change. Every display here is the SAME string the belief-model
	 * ladder uses for the same model, so one model reads as one model across the
	 * page — and where an arm IS a published method, it is the paper's own Table
	 * name for it, U+002D hyphen and all.
	 *
	 * The four reader rows read "… reading" where `name` says "… gate". "Gate" was
	 * our word for the step where the model reads each piece of evidence and keeps
	 * or drops it, and it means nothing outside this repo; the name says what the
	 * step is. The keys keep "gate" because they address shipped bytes.
	 */
	display: string;
}

/**
 * The nine drawn arms in the artifact's FIXED presentation order: the two forms
 * of what INDRA SERVES, then the one model the 2023 paper actually published,
 * then the four reader gates, then our own two reimplementations — which are
 * neither served nor published and are carried for completeness, not as anyone's
 * baseline. `validateReviewQueue` requires the artifact's arm sequence to equal
 * this name sequence exactly; the FIGURE then re-orders by queue size ascending
 * for display, which is a layout decision, not a data one.
 *
 * The paper's own RF sits at index 2 rather than index 0 deliberately: it is the
 * STRONGEST belief model on this panel, and the shipped contract's negative
 * control for "the drawn comparator is the strongest" needs index 0 to be a
 * weaker one to be a real test rather than a vacuous one.
 */
export const REVIEW_QUEUE_ARM_SPECS: readonly ReviewQueueArmSpec[] = [
	{
		name: 'INDRA SimpleScorer, default priors',
		display: 'noisy-OR SimpleScorer (direct)',
		kind: 'paper-model',
		provenance: 'indra-served'
	},
	{
		name: 'INDRA SimpleScorer + hierarchy',
		display: 'noisy-OR + hierarchy propagation',
		kind: 'paper-model',
		provenance: 'indra-served'
	},
	{
		name: 'Paper RF 2k-d13 + Type/#PMIDs/promoter',
		display: 'RF 2k-d13 + Type/#PMIDs/promoter',
		kind: 'paper-model',
		provenance: 'paper-published'
	},
	{ name: 'Gemma 4 26B gate', display: 'Gemma 4 26B reading', kind: 'llm-gate', provenance: 'reader-gate' },
	{ name: 'GLM-5 gate', display: 'GLM-5 reading', kind: 'llm-gate', provenance: 'reader-gate' },
	{ name: 'Gemma 4 31B gate', display: 'Gemma 4 31B reading', kind: 'llm-gate', provenance: 'reader-gate' },
	{ name: 'Gemma 4 E2B gate', display: 'Gemma 4 E2B reading', kind: 'llm-gate', provenance: 'reader-gate' },
	{
		name: 'Our BayesianScorer, source+subtype refit',
		display: 'BayesianScorer, source+subtype refit',
		kind: 'paper-model',
		provenance: 'our-reimplementation'
	},
	{
		name: 'Our CountsScorer RF, full features',
		display: 'CountsScorer RF, full features',
		kind: 'paper-model',
		provenance: 'our-reimplementation'
	}
] as const;

/**
 * Hue, derived from the arm family rather than stored per arm. Storing a
 * `PaperArmKind` on every spec meant writing `paperKind: 'paper'` beside INDRA's
 * shipped SimpleScorer and beside our own ports — a misattribution sitting in
 * the type, invisible on screen but read by anyone maintaining the file. The
 * page only ever distinguishes belief model from reader gate by hue, so derive
 * it. Same function the ladder uses (`beliefLadderPaperKind`), same tokens.
 */
export function reviewQueuePaperKind(kind: ReviewQueueArmKind): PaperArmKind {
	return kind === 'llm-gate' ? 'llm' : 'paper';
}

/** The method caveats travel with the artifact; the figure prints them verbatim. */
export const REVIEW_QUEUE_CAVEAT_COUNT = 7;

/**
 * THE PLAIN HALF OF EVERY TWIN THIS MODULE EMITS.
 *
 * `statement_review_queue.json` carries the two sentences that made this page's
 * dialect problem visible — "tau = the smallest of the arm's own distinct scores
 * whose flag set reaches the target recall of known errors" and "Each arm's
 * threshold is chosen on THIS SAME PANEL". Both are on screen today, and neither
 * is a string any scan of this repo could see: they exist only once the file is
 * read. The restatements below say the same things in the reader's words. Every
 * disclosure survives — the oracle, the two losses, the tie coarseness and the
 * dependence on the 70% target are all still here, at the same strength.
 */
const REVIEW_QUEUE_PLAIN = {
	decisionRule:
		'A statement is flagged as wrong when its belief score is at or below the cutoff.',
	thresholdRule:
		'We lower each model’s score cutoff through the model’s OWN distinct scores until the ' +
		'statements it flags cover the target share of the errors we already know about, and stop ' +
		'at the first — that is, the smallest — cutoff that gets there.',
	operatingRule:
		'A curator with a budget of B reviews reads the B lowest-scoring statements. Where B falls ' +
		'inside a block of statements that share one score, the model cannot say which of them to ' +
		'read first, so that block contributes its errors in proportion, and the count is the ' +
		'EXPECTED number of errors found reading a random B-sized run from the top of the list. No ' +
		'arbitrary order is ever imposed on tied statements: the same statements scored in a ' +
		'different row order give the same curve. Reporting one arbitrary tie order instead would ' +
		'move the reading model’s own count by several errors and would be a property of the sort, ' +
		'not of the model.',
	oracleDisclosure:
		'The budget each comparison model needs is found by choosing its cutoff HERE, with these ' +
		'labels already in hand, to land exactly on the reading model’s error count. That is an ' +
		'ORACLE cutoff: it is chosen and scored on the same 1,689 statements, nobody could have ' +
		'had it before the curation was done, and it FAVOURS the comparison model. The reading ' +
		'model is handed no such help — it has NO cutoff at all. Its operating point is the block ' +
		'of statements whose evidence it rejected outright, which is not a cutoff anyone chose, ' +
		'could not have been tuned, and is the same block on any set of statements. The comparison ' +
		'is stated in this direction deliberately: the side handed the advantage still loses.',
	budgetSweepNote:
		'The advantage is the reading model’s expected catch minus the comparison model’s, at the ' +
		'same budget. It is a property of the budget, not of the models: it is negative at small ' +
		'budgets, peaks at the reading model’s own operating point, and decays back to zero once ' +
		'the budget covers every statement.',
	promotionCeilingWhy:
		'The reading models are INDRA’s own combination rule run over the evidence the reader ' +
		'KEPT, so a reader can only take belief away, never add it. Every true statement the ' +
		'unfiltered rule already scores below 0.9 is therefore beyond promotion by every reading ' +
		'model here, however well it reads. Verified in code: no reading model scores any of these ' +
		'statements above the unfiltered rule. The 0.9 bar is a stated illustration, not a fitted ' +
		'number, and the count only moves one way as the bar moves. NOTE: this count and the true ' +
		'statements a reader zeroes OVERLAP — the per-model overlap is carried beside it — so they ' +
		'are NOT additive.',
	/**
	 * The whole `error_recall_robustness` block. Five of these six sentences are on
	 * screen today under "how this is computed", which is a method note and not one
	 * of the boundaries marked "in the artifact's own words" — so they reached a
	 * reader with nothing beside them at all.
	 */
	/**
	 * The count is DERIVED in the shipped sentence ("the panel's known errors"),
	 * and this block reports two different sets of statements — the full one and
	 * the sensitivity one with 111 statements removed — so their known-wrong
	 * counts are not the same number. An earlier restatement hard-coded 452 here,
	 * which is the full set's count and is wrong for the other one.
	 */
	robustnessMetric:
		'How much of the known-wrong work a curator turns up on a fixed review budget: the ' +
		'expected share of the statements already known to be wrong IN THE SET BEING SCORED that ' +
		'is found by reading the B lowest-scoring statements, with statements sharing one score ' +
		'counted in proportion (the same rule the budget curve uses).',
	robustnessBudgetRule:
		'The budget is 0.25 of the statements being scored — a quarter of them — rounded down, so ' +
		'both sets of statements are asked the same question at the same share of their own size.',
	robustnessBootstrapDesign:
		'Statements are re-drawn at random with replacement WITHIN each of the 10 folds — the 10 groups ' +
		'the statements were split into — assigned in 2023, so every redraw keeps that fold make-up. ' +
		'ONE redraw is ' +
		'shared by every model, so the models’ margins are drawn together and the interval that ' +
		'covers all four at once carries how they move together rather than assuming they do not. ' +
		'Same seed, same number of redraws and the same design as ' +
		'scripts/compute_paper_robustness.py, which does this for the ordering margin on the same ' +
		'1,689 statements.',
	robustnessMultiplicityMethod:
		'The width is set by the largest standardised margin seen across the four reading models ' +
		'on each shared redraw, so ONE interval covers all four at once.',
	robustnessMultiplicityNote:
		'The frozen run plan lists all four reading models and names none of them as the one to be ' +
		'confirmed, so an interval wide enough to cover all four at once is the fair thing to ask ' +
		'for. The four move together from redraw to redraw, so covering all four costs far less ' +
		'width than a Bonferroni correction would.',
	robustnessLabelCompletenessNote:
		'These statements are labelled WRONG in the 2023 released labels and their evidence review ' +
		'was never finished. Dropping them revises those released labels, so this is a check to ' +
		'the side and never the headline result. It is not free either: it removes a quarter of ' +
		'the wrong statements and changes what this set of statements is a sample OF.'
} as const;

/**
 * The nine per-model notes, keyed by the FROZEN `arms[].name`. Keyed rather than
 * written under one `note:` key because one key here addresses nine different
 * sentences: four of the nine ship the SAME text, and a twin bound by key alone
 * would happily print the random forest's restatement under a reading model.
 * Each carries a verbatim fragment, so a reissued note gates instead.
 */
const REVIEW_QUEUE_ARM_NOTE_TWINS: Readonly<Record<string, AnchoredProse>> = {
	'INDRA SimpleScorer, default priors': {
		artifactAnchor: 'This is what INDRA serves today',
		plain:
			'INDRA’s shipped unfitted combination rule at its default per-source reliabilities, over ' +
			'the statement’s own evidence. This is what INDRA serves today. It is NOT the ' +
			'MCMC-refitted “Belief Orig” of the 2023 paper.'
	},
	'INDRA SimpleScorer + hierarchy': {
		artifactAnchor: 'propagated over the statement hierarchy',
		plain:
			'The same shipped scorer with belief passed up the statement hierarchy — direct evidence ' +
			'plus evidence from the more specific statements underneath it, negations excluded — ' +
			'which is the stronger of the two served forms on these 1,689 statements. The 2023 paper ' +
			'publishes no hierarchy scorer.'
	},
	'Paper RF 2k-d13 + Type/#PMIDs/promoter': {
		artifactAnchor: 'released random forest',
		plain:
			'The random forest released with the 2023 paper, re-run from ' +
			'sorgerlab/indra_assembly_paper on the feature matrix and the 10 folds released with ' +
			'it, with every statement scored by a copy that never saw it. A research model: it is ' +
			'not in indra and has never been served.'
	},
	'Gemma 4 26B gate': {
		artifactAnchor: 'the evidence the reader KEPT',
		plain: 'INDRA’s own combination rule applied to the evidence the reading model KEPT.'
	},
	'GLM-5 gate': {
		artifactAnchor: 'the evidence the reader KEPT',
		plain: 'INDRA’s own combination rule applied to the evidence the reading model KEPT.'
	},
	'Gemma 4 31B gate': {
		artifactAnchor: 'the evidence the reader KEPT',
		plain: 'INDRA’s own combination rule applied to the evidence the reading model KEPT.'
	},
	'Gemma 4 E2B gate': {
		artifactAnchor: 'the evidence the reader KEPT',
		plain: 'INDRA’s own combination rule applied to the evidence the reading model KEPT.'
	},
	'Our BayesianScorer, source+subtype refit': {
		artifactAnchor: 'reliabilities over INDRA',
		plain:
			'A refit of the per-source (and per-subtype) reliabilities over INDRA’s wider feature ' +
			'set, with every statement scored by a copy that never saw it. The 2023 paper publishes ' +
			'no Bayesian or subtype scorer; this one is fitted here.'
	},
	'Our CountsScorer RF, full features': {
		artifactAnchor: 'port of INDRA',
		plain:
			'A port of INDRA’s CountsScorer over its 77-feature superset, with every statement ' +
			'scored by a copy that never saw it. It is not the engineered feature set of the 2023 ' +
			'paper and must never be called the published random forest.'
	}
};

/** The reading models' shared definition of a belief of exactly zero. */
const REVIEW_QUEUE_ZERO_PILE_TWIN: AnchoredProse = {
	artifactAnchor: 'the reader rejected every piece of evidence it read',
	plain:
		'A belief of exactly 0.0 means the reading model rejected every piece of evidence it read: ' +
		'a source with no surviving evidence drops out of the combination entirely, and an empty ' +
		'combination is a belief of 0.'
};

/** What the reading design costs before any model runs. */
const REVIEW_QUEUE_CEILING_WHY_TWIN: AnchoredProse = {
	artifactAnchor: 'a reader can only remove belief, never add it',
	plain: REVIEW_QUEUE_PLAIN.promotionCeilingWhy
};

/** Where an equal-yield reference point comes from: nobody chose it. */
const REVIEW_QUEUE_ORIGIN_TWIN: AnchoredProse = {
	artifactAnchor: 'No threshold was chosen',
	plain:
		'the block of statements whose evidence the reading model rejected outright. No cutoff was ' +
		'chosen.'
};

/** `caveats[]` in shipped order, pinned to each sentence by a verbatim fragment. */
const REVIEW_QUEUE_CAVEAT_TWINS: readonly AnchoredProse[] = [
	{
		artifactAnchor: 'NEVER published a decision or threshold metric',
		plain:
			'The 2023 paper published no decision measure and no cutoff measure at all — only ' +
			'trapezoidal PR-AUC, averaged over 10 folds with a spread. This figure is a new ' +
			'derivation. Exactly one belief model here was published: RF 2k-d13 + ' +
			'Type/#PMIDs/promoter, the released code re-run so that every statement is scored by a ' +
			'model that never saw it. Two are INDRA’s shipped SimpleScorer, which is NOT the ' +
			'MCMC-refitted “Belief Orig” of the 2023 paper, and two are reimplementations over ' +
			'INDRA’s wider feature set that were never published. Nothing here was reported in 2023.'
	},
	{
		artifactAnchor: 'chosen on THIS SAME PANEL',
		plain:
			'Each model’s cutoff is chosen on THESE SAME 1,689 statements, to catch at least the ' +
			'target share of the errors. That is an operating-point choice, not a result on ' +
			'statements held back, and no model’s cutoff is checked out of sample. The models are ' +
			'asked for the same catch rate but do not all achieve it — one that cannot land exactly ' +
			'on the target overshoots it — so read the bars as a both-axes-at-once comparison ' +
			'(shorter list AND more errors caught), not as a like-for-like held at equal catch rate.'
	},
	{
		artifactAnchor: 'can only be operated at a few discrete points',
		plain:
			'The reading models can only be operated at a few discrete points. Their lowest score is ' +
			'an exact tie shared by hundreds of statements — every statement whose evidence the ' +
			'model rejected outright — and that one block is the whole flagged set here, so the ' +
			'cutoff cannot be tuned finely. Gemma 4 26B returns the same 462 statements at the 50%, ' +
			'60% and 70% targets; GLM-5 returns the same 541 at all four; the next reachable point ' +
			'for Gemma 4 26B jumps the list to 687 and drops precision to 58.2%, with nothing in ' +
			'between. That is the cost of the tied scores this page examines in the ' +
			'score-distribution figure below.'
	},
	{
		artifactAnchor: 'headline target of 70% is a choice',
		plain:
			'The headline target of 70% is a choice, and the ordering of the list depends on it: at ' +
			'an 80% target Gemma 4 26B’s list grows to 687, within 1 of RF 2k-d13 + ' +
			'Type/#PMIDs/promoter’s 688 — the margin has closed. What the figure shows is a property ' +
			'of this operating point, not of every operating point.'
	},
	{
		artifactAnchor: 'scored on CURRENT INDRA evidence',
		plain:
			'The four reading models and the four belief models that were never published are ' +
			'scored on CURRENT ' +
			'INDRA evidence. The random forest is scored on the 2023 feature matrix and the folds ' +
			'that came with it, because that is what re-running the released code produces. The ' +
			'models therefore compare cleanly to each other on the same statements and the same ' +
			'labels, but only loosely to the table published in 2023.'
	},
	{
		artifactAnchor: 'noisy-OR applied to the evidence the reader kept',
		plain:
			'The reading models use INDRA’s own combination rule over the evidence each model kept, ' +
			'so what is compared is belief models over a shared way of combining evidence; the ' +
			'reading model’s contribution is the filtering. They are also not zero-shot: each call ' +
			'carries 14 hand-written example pairs.'
	},
	{
		artifactAnchor: 'review-budget metric, not a ranking metric',
		plain:
			'The interval on the operational result is a review-budget measure, not an ordering ' +
			'measure. It is the one place on this page where a reading model is measured on what it ' +
			'actually does — decide — rather than on the ordering the 2023 paper reported, and the ' +
			'two do not move together: the ordering margin shrinks on the adjudication-safe set of ' +
			'statements while this one grows.'
	}
];

/**
 * SVG geometry for the budget-sweep panel, exported so the label budget below is
 * DERIVED from it rather than eyeballed. This page has shipped a silent
 * right-anchored clip before: SVG text that overruns its gutter loses glyphs with
 * no layout error and no test failure, and `<desc>` still reads the full string,
 * so the loss is invisible to both a reader and a screen-reader audit.
 */
export const REVIEW_QUEUE_SWEEP_GEOMETRY = {
	width: 900,
	height: 300,
	plotLeft: 46,
	plotRight: 700,
	plotTop: 46,
	plotBottom: 248,
	/** Series labels are LEFT-anchored here; usable rail is 708 → 900 units. */
	labelX: 708,
	labelFontPx: 9,
	/** Measured advance of the mono face at 9px, in user units per character. */
	monoUnitsPerChar: 5.4186
} as const;

/**
 * SERIES LABEL BUDGET: (900 − 708) = 192 units ÷ 5.4186 u/char at 9px = 35.43 →
 * 35 characters. Only the two DRAWN series are measured against this: today the
 * reader row, "Gemma 4 26B reading", at 19 (103.0 units) and the paper's own RF,
 * "RF 2k-d13 + Type/#PMIDs/promoter", at 32 (173.4 units, 18.6 units of slack).
 * `buildReviewQueueSweep` THROWS above the budget, so a longer display name gates
 * the sweep panel instead of quietly eating its trailing glyphs.
 */
export const REVIEW_QUEUE_SWEEP_LABEL_BUDGET_CHARS = 35;

/**
 * ARM GUTTER BUDGET for the bar panel: labels are right-anchored at x = 206 with
 * the gutter running 0 → 206, and 204 usable units ÷ 5.4186 = 37.6 → 37
 * characters. Every arm is measured against this, not just the drawn pair. The
 * longest the table can produce is "BayesianScorer, source+subtype refit" at 36
 * (195.1 units, 8.9 units of slack). Right-anchored overruns lose LEADING
 * glyphs, which reads as a different model name rather than as damage. Enforced
 * in the builder for the same reason as above.
 */
export const REVIEW_QUEUE_GUTTER_BUDGET_CHARS = 37;

/**
 * AXIS READOUT BUDGET: the sweep's y-tick labels are right-anchored at
 * plotLeft − 6 = 40 units, and the tick face is 8px → 5.4186 × 8/9 = 4.8165
 * u/char, so 40 ÷ 4.8165 = 8.3 → 8 characters. Today's longest is the panel's
 * own error count, "452". Enforced because the value comes from the artifact:
 * a differently-sized panel must gate the figure, not clip its axis.
 */
export const REVIEW_QUEUE_AXIS_BUDGET_CHARS = 8;

/** The aggregation the LLM zero-pile identity depends on (unfitted hard gate). */
export const REVIEW_QUEUE_REQUIRED_AGGREGATION = 'indra_default_hard_gate';

/** One arm at one target recall: the whole operating point, nothing derived. */
export interface ReviewQueuePoint {
	targetRecall: number;
	/** Flag as wrong iff belief <= tau. */
	tau: number;
	/** Statements a curator would have to read. queue = caught + falseAlarms. */
	queue: number;
	trueErrorsCaught: number;
	falseAlarms: number;
	/** caught / queue. */
	precision: number;
	/** caught / panel errors — always >= targetRecall (the search is a >= search). */
	recallAchieved: number;
	/** recallAchieved − targetRecall; large on the coarse LLM operating points. */
	recallOvershoot: number;
	queueShareOfPanel: number;
}

/**
 * The zero pile: belief exactly 0.0 == the reader rejected EVERY piece of
 * evidence it read. Present only on the LLM gates (a paper belief model has no
 * such decision), and only meaningful because the bundles use the unfitted hard
 * gate — under a fitted soft profile belief floors at sigmoid(prior), not 0.
 */
export interface ReviewQueueZeroPile {
	definition: string;
	/** `definition` with its plain restatement — `shipped` is byte-identical. */
	definitionProse: ShippedProse;
	size: number;
	trueErrors: number;
	falseAlarms: number;
	precision: number;
	/** trueErrors / panel errors. */
	shareOfAllErrors: number;
	/** True when the pile IS the arm's entire flag set at the headline target. */
	isWholeFlagSetAtHeadlineTarget: boolean;
	requiresNoThresholdTuning: boolean;
}

export interface ReviewQueueArm {
	name: string;
	/** On-screen name; see ReviewQueueArmSpec.display. */
	display: string;
	kind: ReviewQueueArmKind;
	/** Whose model this is. Asserted equal to the artifact's own field. */
	provenance: ReviewQueueProvenance;
	paperKind: PaperArmKind;
	modelKey: string;
	note: string;
	/** `note` with its plain restatement — `noteProse.shipped === note`. */
	noteProse: ShippedProse;
	distinctScores: number;
	/** The headline operating point — identical to the grid entry at that target. */
	operatingPoint: ReviewQueuePoint;
	/** Precision at every matched recall target, in target order. */
	grid: ReviewQueuePoint[];
	/**
	 * How many DISTINCT queue sizes the arm can produce across the whole target
	 * grid. 4 for the near-continuous paper models; 2 for the Gemmas; 1 for GLM-5,
	 * which has a single operating point across the entire grid. This is the field
	 * the coarseness caveat cites — the repeated precision is the symptom.
	 */
	distinctQueueSizesAcrossTargets: number;
	zeroPile: ReviewQueueZeroPile | null;
	scoresPath: string;
}

export interface ReviewQueuePanel {
	n: number;
	nErrors: number;
	nCorrect: number;
	errorBaseRate: number;
	label: string;
}

/**
 * The promotion ceiling: the loss the FORMULA has already taken before any
 * reader runs. The reader arms are INDRA's own unfitted noisy-OR over the evidence a
 * reader KEPT, so a reader can only remove belief — every true statement the
 * unfiltered noisy-OR already scores below the bar is unpromotable by every
 * reader arm here. The emitting script asserts exactly that on the arms
 * themselves; this side re-checks only that the count is a possible subset count
 * of the panel's correct statements.
 */
export interface ReviewQueuePromotionCeiling {
	/** The stated promotion bar, in (0, 1). Not fitted; the count is monotone in it. */
	threshold: number;
	/** The arm whose scores the ceiling is read off — the unfiltered noisy-OR. */
	referenceArm: string;
	/** `referenceArm` resolved to its display name; render THIS. */
	referenceArmDisplay: string;
	nTrue: number;
	nTrueBelowThreshold: number;
	/**
	 * Per reader arm: how its zero pile RELATES to the ceiling. These two costs
	 * overlap (a true statement on one weak evidence is both already under the bar
	 * and exactly what a reader zeroes), so they must never be presented as
	 * additive. Derived in the artifact; the page renders the union, never a sum.
	 */
	perArmOverlap: Record<string, ReviewQueueCeilingOverlap>;
	/** The artifact's own justification, printed verbatim where it is cited. */
	why: string;
	/** `why` with its plain restatement — `whyProse.shipped === why`. */
	whyProse: ShippedProse;
}

export interface ReviewQueueCeilingOverlap {
	nTrueZeroedByArm: number;
	nAlsoAlreadyBelowThreshold: number;
	nNewlyLostByArm: number;
	nTrueAffectedUnion: number;
	shareOfTrueAffected: number;
}

/**
 * One paper belief model measured against a reader gate's own untuned operating
 * point, on the two questions a curator actually asks: how many statements do I
 * have to read to find the same errors, and how many do I find for the same
 * reading. The threshold that lands the comparator on the reference yield is
 * chosen ON THIS PANEL with the labels in hand — an oracle, and one that favours
 * the comparator. `ReviewQueueEqualYield.oracleDisclosure` says so verbatim.
 */
export interface ReviewQueueEqualYieldComparator {
	/** Frozen `arms[].name` join key. */
	arm: string;
	/** Smallest budget whose expected catch reaches the reference arm's catch. */
	budgetForEqualYield: number;
	/** budgetForEqualYield − the reference budget. Signed: a better comparator is negative. */
	extraReviews: number;
	/** reference catch / budgetForEqualYield. */
	precisionAtEqualYield: number;
	/** Expected errors found at the REFERENCE arm's budget. Fractional where a tie block is cut. */
	errorsCaughtAtReferenceBudget: number;
	/** reference catch − errorsCaughtAtReferenceBudget. */
	shortfallAtReferenceBudget: number;
	/** Always true for a comparator: this is the oracle being disclosed per row. */
	thresholdFittedOnThisPanel: boolean;
}

/** A reader gate at its zero pile — the operating point nobody chose. */
export interface ReviewQueueEqualYieldReference {
	arm: string;
	budget: number;
	trueErrorsCaught: number;
	falseAlarms: number;
	precision: number;
	/** trueErrorsCaught / panel errors. */
	recall: number;
	/** Always false: no threshold was fitted to produce this point. */
	thresholdFittedOnThisPanel: boolean;
	origin: string;
	/** `origin` with its plain restatement — `originProse.shipped === origin`. */
	originProse: ShippedProse;
	isWholeFlagSetAtHeadlineTarget: boolean;
	comparators: ReviewQueueEqualYieldComparator[];
}

/**
 * Expected errors found against review budget, for every arm on the panel. The
 * figure draws two of these lines; the artifact carries all of them so the pair
 * cannot be a selection the page made and did not show.
 */
export interface ReviewQueueBudgetSweep {
	step: number;
	/** Strictly increasing, 0 → panel.n, including every reference budget exactly. */
	budgets: number[];
	/** arm name → expected errors caught, aligned to `budgets`. */
	errorsCaught: Record<string, number[]>;
	/** The reference arm's strongest comparator — fewest extra reviews. Derived, not named. */
	comparatorArm: string;
	/** reference − comparator, aligned to `budgets`. Negative at small budgets. */
	advantage: number[];
	firstPositiveBudget: number;
	peakBudget: number;
	peakAdvantage: number;
	/** First budget past the peak where the advantage has halved; null if it never does. */
	halfPeakDecayBudget: number | null;
	note: string;
	/** `note` with its plain restatement — `noteProse.shipped === note`. */
	noteProse: ShippedProse;
}

export interface ReviewQueueEqualYield {
	/** How a budget becomes a catch, including the tie rule. Printed verbatim. */
	operatingRule: string;
	/** That the comparator's threshold is fitted here, and that this favours it. */
	oracleDisclosure: string;
	/** `operatingRule` with its plain restatement — `shipped` is byte-identical. */
	operatingRuleProse: ShippedProse;
	/** `oracleDisclosure` with its plain restatement — `shipped` is byte-identical. */
	oracleDisclosureProse: ShippedProse;
	/** The gate the figure draws — the shortest whole-flag-set zero pile. */
	referenceArm: string;
	references: ReviewQueueEqualYieldReference[];
	budgetSweep: ReviewQueueBudgetSweep;
}

/**
 * ONE arm's error recall at a fixed review budget, against the paper's own RF.
 *
 * This is the operational result's own uncertainty, and it is a DIFFERENT
 * question from the ranking margin the rest of /paper reports: a review budget
 * asks what a curator finds, a ranking metric asks how the whole list is
 * ordered. The two do not move together, which is exactly why this block exists
 * rather than being inferred from the AP interval.
 */
export interface ReviewQueueRobustnessArm {
	/** Frozen `arms[].name` join key. */
	arm: string;
	/** Errors found in the budget / all panel errors, tie-fair. */
	errorRecall: number;
	/** errorRecall − the reference arm's. Asserted to reconstruct. */
	delta: number;
	deltaBootstrapMean: number;
	ci95Low: number;
	ci95High: number;
	bootstrapSe: number;
	pGreaterThanZero: number;
	/** Studentized max-t band over the whole family of reader gates. */
	simultaneousLow: number;
	simultaneousHigh: number;
	/**
	 * ahead / behind / not-significant on this arm's OWN interval, from its own
	 * endpoints. Replaces a sign-blind `excludesZeroPointwise`.
	 */
	pointwiseStanding: Standing;
	/**
	 * The one that answers a referee: the same three classes on the band widened
	 * over all four arms. The clause that prints it used to read
	 * `excludesZeroSimultaneous ? '(stays clear of zero)' : '(includes zero)'`,
	 * which says "stays clear" for a band lying entirely BELOW zero — the seventh
	 * sign-blindness occurrence, waiting.
	 */
	simultaneousStanding: Standing;
	nValidResamples: number;
}

/**
 * The clause printed after the widened band, one per class. Authored here rather
 * than in the component so the sweep that reads every string a `paper-*.ts`
 * module writes can see it, and TOTAL so the compiler demands a clause for each.
 *
 * The `behind` clause is unreachable on the shipped artifact — all eight bands
 * are above zero or across it. It is written anyway, because the two-way test it
 * replaces would have printed "stays clear of zero" for a band lying entirely
 * BELOW zero, and the whole point of the three-way class is that the case gets
 * words before it gets data.
 */
export const REVIEW_QUEUE_WIDENED_BAND_CLAUSE: Readonly<Record<Standing, string>> = {
	ahead: '(stays clear of zero)',
	behind: '(stays clear of zero, on the losing side — the random forest finds more)',
	'not-significant': '(includes zero)'
};

export interface ReviewQueueRobustnessPanel {
	id: string;
	role: 'primary' | 'sensitivity';
	nStatements: number;
	nErrors: number;
	/** floor(budgetShare x nStatements). Recomputed by the validator. */
	budget: number;
	budgetShareOfPanel: number;
	referenceErrorRecall: number;
	maxTCriticalValue: number;
	bonferroniCriticalValue: number;
	/** arm name → its row. Exactly the drawn reader gates. */
	arms: Record<string, ReviewQueueRobustnessArm>;
}

export interface ReviewQueueLabelCompleteness {
	field: string;
	nDropped: number;
	allDroppedAreNegative: boolean;
	droppedShareOfAllNegatives: number;
	droppedShareOfPanel: number;
	noModelIsRefit: boolean;
	note: string;
	/** `note` with its plain restatement — `noteProse.shipped === note`. */
	noteProse: ShippedProse;
}

/**
 * The six sentences the robustness block ships, each with the plain restatement
 * a reader is handed instead. They render today under "how this is computed" —
 * a method note, NOT one of the boundaries marked "in the artifact's own words"
 * — so before these twins existed they were the one place on this figure where
 * shipped wording reached a reader with nothing beside it.
 */
export interface ReviewQueueRobustnessProse {
	/** What "error recall at a fixed review budget" measures. */
	metric: ShippedProse;
	/** How the budget is set, and why it is a share rather than a count. */
	budgetRule: ShippedProse;
	/** How the interval is drawn, and what the shared redraw buys. */
	bootstrapDesign: ShippedProse;
	/** How ONE interval is made to cover all four reading models. */
	multiplicityMethod: ShippedProse;
	/** Why covering all four is the fair ask here. */
	multiplicityNote: ShippedProse;
	/** What the second, label-revised set of statements costs to look at. */
	labelCompletenessNote: ShippedProse;
}

export interface ReviewQueueErrorRecallRobustness {
	metric: string;
	budgetShare: number;
	budgetRule: string;
	/** The comparator: the paper's own published RF. */
	referenceArm: string;
	/** The gate the figure quotes — the equal-yield reference, derived upstream. */
	headlineArm: string;
	seed: number;
	nBootstrap: number;
	bootstrapDesign: string;
	/** Every reader gate, because the run plan designated none of them. */
	family: string[];
	familyAlpha: number;
	multiplicityMethod: string;
	pointwiseNormalCriticalValue: number;
	multiplicityNote: string;
	labelCompleteness: ReviewQueueLabelCompleteness;
	/** The paper's own labels, unmodified. The result. */
	primary: ReviewQueueRobustnessPanel;
	/** OUR revision of their labels. A check, never the headline. */
	sensitivity: ReviewQueueRobustnessPanel;
	/**
	 * The plain half of every string above. Each flat field is byte-identical to
	 * its twin's `shipped`; the twin is the one to render, and the shipped half
	 * belongs behind the page's verification boundary.
	 */
	prose: ReviewQueueRobustnessProse;
}

/**
 * Every shipped sentence this figure carries, in one place, each with the plain
 * restatement a reader is handed instead. The two under `equalYield` are the
 * SAME objects reachable through `equalYield.operatingRuleProse` /
 * `.oracleDisclosureProse`, not copies, so the disclosure still travels with the
 * numbers it governs.
 */
export interface ReviewQueueProse {
	decisionRule: ShippedProse;
	thresholdRule: ShippedProse;
	/** How a budget becomes a catch, including the tie rule. */
	operatingRule: ShippedProse;
	/** That the comparator's cutoff is fitted here, and that this favours it. */
	oracleDisclosure: ShippedProse;
	/** What the advantage curve is a property of. */
	budgetSweepNote: ShippedProse;
	/** What the reading design has already cost before any model runs. */
	promotionCeilingWhy: ShippedProse;
	/** Index-aligned with `caveats`, pinned to it by a verbatim fragment. */
	caveats: ShippedProse[];
	/**
	 * Index-aligned with `arms`, pinned to each model's own sentence. Nine models,
	 * nine notes, four of which ship identical text — so these are the SAME objects
	 * reachable through `arms[i].noteProse`, never copies.
	 */
	armNotes: ShippedProse[];
	/** The shared "belief 0.0 means it rejected everything" definition. */
	zeroPileDefinition: ShippedProse;
	/** Where the untuned operating point comes from. */
	equalYieldOrigin: ShippedProse;
	/**
	 * The whole `error_recall_robustness` block. The SAME objects reachable through
	 * `errorRecallRobustness.prose`, so the method note and the verification
	 * boundary can never show different restatements of one sentence.
	 */
	robustness: ReviewQueueRobustnessProse;
}

export interface ReviewQueue {
	decisionRule: string;
	thresholdRule: string;
	headlineTargetRecall: number;
	targetRecalls: number[];
	panel: ReviewQueuePanel;
	/** What the gate design costs before any reader runs. */
	promotionCeiling: ReviewQueuePromotionCeiling;
	arms: ReviewQueueArm[];
	/** Fixed yield instead of fixed threshold, plus the whole budget curve. */
	equalYield: ReviewQueueEqualYield;
	/** Is the operational gap real? Bootstrapped, max-t corrected, two panels. */
	errorRecallRobustness: ReviewQueueErrorRecallRobustness;
	/** The method caveats, verbatim from the artifact. */
	caveats: string[];
	/**
	 * The plain half of every string this figure took off the artifact. Each flat
	 * field above is byte-identical to its twin's `shipped`; the twin is the one to
	 * render, and the shipped half belongs behind the verification boundary.
	 */
	prose: ReviewQueueProse;
	generatedBy: string;
}

export interface ReviewQueueOk {
	status: 'ok';
	reason: null;
	artifact_path: string;
	artifact_sha256: string;
	queue: ReviewQueue;
}

export interface ReviewQueueUnavailable {
	status: 'unavailable';
	reason: string;
	artifact_path: string;
	artifact_sha256: string | null;
	queue: null;
}

export type ReviewQueueLoad = ReviewQueueOk | ReviewQueueUnavailable;

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

function finite(value: unknown, context: string): number {
	if (typeof value !== 'number' || !Number.isFinite(value)) {
		fail(context, 'expected a finite number');
	}
	return value;
}

function unit(value: unknown, context: string): number {
	const parsed = finite(value, context);
	if (parsed < 0 || parsed > 1) fail(context, 'expected a number in [0, 1]');
	return parsed;
}

function nonNegativeInteger(value: unknown, context: string): number {
	if (typeof value !== 'number' || !Number.isInteger(value) || value < 0) {
		fail(context, 'expected a non-negative integer');
	}
	return value;
}

function positiveInteger(value: unknown, context: string): number {
	const parsed = nonNegativeInteger(value, context);
	if (parsed < 1) fail(context, 'expected a positive integer');
	return parsed;
}

function boolean(value: unknown, context: string): boolean {
	if (typeof value !== 'boolean') fail(context, 'expected a boolean');
	return value;
}

function text(value: unknown, context: string): string {
	if (typeof value !== 'string' || value.length === 0) {
		fail(context, 'expected a non-empty string');
	}
	return value;
}

function close(got: number, want: number, context: string, message: string): void {
	if (Math.abs(got - want) > REVIEW_QUEUE_PARITY_TOL) fail(context, message);
}

/**
 * One (arm, target) cell. The three identities the bar geometry depends on are
 * enforced here, not assumed: the two segments ARE the bar length, the printed
 * precision IS caught/queue, and the achieved recall IS caught/errors and never
 * undershoots the target it was chosen to hit.
 */
function parsePoint(value: unknown, context: string, nErrors: number, nPanel: number): ReviewQueuePoint {
	const obj = record(value, context);
	const targetRecall = unit(obj.target_recall, `${context}.target_recall`);
	const tau = unit(obj.tau, `${context}.tau`);
	const queue = positiveInteger(obj.queue, `${context}.queue`);
	const trueErrorsCaught = nonNegativeInteger(obj.true_errors_caught, `${context}.true_errors_caught`);
	const falseAlarms = nonNegativeInteger(obj.false_alarms, `${context}.false_alarms`);
	const precision = unit(obj.precision, `${context}.precision`);
	const recallAchieved = unit(obj.recall_achieved, `${context}.recall_achieved`);

	if (trueErrorsCaught + falseAlarms !== queue) {
		fail(context, 'true_errors_caught + false_alarms must equal queue');
	}
	if (queue > nPanel) fail(`${context}.queue`, 'cannot exceed the panel size');
	if (trueErrorsCaught > nErrors) fail(`${context}.true_errors_caught`, 'cannot exceed the panel errors');
	close(precision, trueErrorsCaught / queue, `${context}.precision`, 'must equal true_errors_caught / queue');
	close(
		recallAchieved,
		trueErrorsCaught / nErrors,
		`${context}.recall_achieved`,
		'must equal true_errors_caught / the panel error count'
	);
	if (recallAchieved + REVIEW_QUEUE_PARITY_TOL < targetRecall) {
		fail(`${context}.recall_achieved`, 'must reach the target recall the threshold was chosen for');
	}
	return {
		targetRecall,
		tau,
		queue,
		trueErrorsCaught,
		falseAlarms,
		precision,
		recallAchieved,
		// Optional in the operating_point block; derived identically either way.
		recallOvershoot:
			obj.recall_overshoot === undefined
				? recallAchieved - targetRecall
				: finite(obj.recall_overshoot, `${context}.recall_overshoot`),
		queueShareOfPanel:
			obj.queue_share_of_panel === undefined
				? queue / nPanel
				: unit(obj.queue_share_of_panel, `${context}.queue_share_of_panel`)
	};
}

function samePoint(a: ReviewQueuePoint, b: ReviewQueuePoint): boolean {
	return (
		a.targetRecall === b.targetRecall &&
		a.tau === b.tau &&
		a.queue === b.queue &&
		a.trueErrorsCaught === b.trueErrorsCaught &&
		a.falseAlarms === b.falseAlarms &&
		a.precision === b.precision &&
		a.recallAchieved === b.recallAchieved
	);
}

function parseZeroPile(
	value: unknown,
	context: string,
	nErrors: number,
	operatingPoint: ReviewQueuePoint
): ReviewQueueZeroPile {
	const obj = record(value, context);
	const size = positiveInteger(obj.size, `${context}.size`);
	const trueErrors = nonNegativeInteger(obj.true_errors, `${context}.true_errors`);
	const falseAlarms = nonNegativeInteger(obj.false_alarms, `${context}.false_alarms`);
	const precision = unit(obj.precision, `${context}.precision`);
	const shareOfAllErrors = unit(obj.share_of_all_errors, `${context}.share_of_all_errors`);
	const isWholeFlagSetAtHeadlineTarget = boolean(
		obj.is_whole_flag_set_at_headline_target,
		`${context}.is_whole_flag_set_at_headline_target`
	);
	if (trueErrors + falseAlarms !== size) fail(context, 'true_errors + false_alarms must equal size');
	close(precision, trueErrors / size, `${context}.precision`, 'must equal true_errors / size');
	close(
		shareOfAllErrors,
		trueErrors / nErrors,
		`${context}.share_of_all_errors`,
		'must equal true_errors / the panel error count'
	);
	// The whole claim of the callout: the pile is not a tuned slice of the queue,
	// it IS the queue. If the artifact ever says otherwise, refuse to draw it.
	if (
		isWholeFlagSetAtHeadlineTarget &&
		(size !== operatingPoint.queue ||
			trueErrors !== operatingPoint.trueErrorsCaught ||
			falseAlarms !== operatingPoint.falseAlarms)
	) {
		fail(context, 'claims to be the whole flag set but does not equal the headline operating point');
	}
	// Parsed exactly as before; the plain restatement is pinned beside it, and the
	// flat string stays byte-identical to `definitionProse.shipped`.
	const definitionProse = anchoredShippedProse(
		text(obj.definition, `${context}.definition`),
		REVIEW_QUEUE_ZERO_PILE_TWIN,
		`${context}.definition`
	);
	return {
		definition: definitionProse.shipped,
		definitionProse,
		size,
		trueErrors,
		falseAlarms,
		precision,
		shareOfAllErrors,
		isWholeFlagSetAtHeadlineTarget,
		requiresNoThresholdTuning: boolean(
			obj.requires_no_threshold_tuning,
			`${context}.requires_no_threshold_tuning`
		)
	};
}

/**
 * The promotion ceiling, fail-closed. Three things must hold or the number is
 * not drawn: the bar is a real probability strictly inside (0, 1) (a bar of 0 or
 * 1 would make the count trivially empty or the whole panel), the count and the
 * true-statement total are integers, and the count cannot exceed the correct
 * statements it is a subset of. A missing or malformed block takes the loader's
 * `unavailable` path — the page never falls back to a default ceiling.
 */
function parseCeiling(
	value: unknown,
	context: string,
	panel: ReviewQueuePanel
): ReviewQueuePromotionCeiling {
	const obj = record(value, context);
	const threshold = unit(obj.threshold, `${context}.threshold`);
	if (threshold <= 0 || threshold >= 1) {
		fail(`${context}.threshold`, 'expected a promotion bar strictly inside (0, 1)');
	}
	const nTrue = positiveInteger(obj.n_true, `${context}.n_true`);
	const nTrueBelowThreshold = nonNegativeInteger(
		obj.n_true_below_threshold,
		`${context}.n_true_below_threshold`
	);
	if (nTrue !== panel.nCorrect) {
		fail(`${context}.n_true`, 'must be the panel correct-statement count');
	}
	if (nTrueBelowThreshold > panel.nCorrect) {
		fail(`${context}.n_true_below_threshold`, 'cannot exceed the panel correct-statement count');
	}
	const rawOverlap = record(obj.per_arm_overlap, `${context}.per_arm_overlap`);
	const perArmOverlap: Record<string, ReviewQueueCeilingOverlap> = {};
	for (const [arm, blk] of Object.entries(rawOverlap)) {
		const o = record(blk, `${context}.per_arm_overlap.${arm}`);
		const zeroed = nonNegativeInteger(o.n_true_zeroed_by_arm, `${context}.${arm}.n_true_zeroed_by_arm`);
		const both = nonNegativeInteger(o.n_also_already_below_threshold, `${context}.${arm}.n_also_already_below_threshold`);
		const fresh = nonNegativeInteger(o.n_newly_lost_by_arm, `${context}.${arm}.n_newly_lost_by_arm`);
		const union = nonNegativeInteger(o.n_true_affected_union, `${context}.${arm}.n_true_affected_union`);
		// The split must close and the union must be the ceiling plus the newly lost,
		// or the page could print a sum that the artifact does not support.
		if (both + fresh !== zeroed) {
			fail(`${context}.per_arm_overlap.${arm}`, 'overlap split does not close to the zeroed count');
		}
		if (union !== nTrueBelowThreshold + fresh) {
			fail(`${context}.per_arm_overlap.${arm}`, 'union is not the ceiling plus the newly-lost');
		}
		if (union > nTrue) {
			fail(`${context}.per_arm_overlap.${arm}`, 'union cannot exceed the true-statement count');
		}
		perArmOverlap[arm] = {
			nTrueZeroedByArm: zeroed,
			nAlsoAlreadyBelowThreshold: both,
			nNewlyLostByArm: fresh,
			nTrueAffectedUnion: union,
			shareOfTrueAffected: unit(o.share_of_true_affected, `${context}.${arm}.share_of_true_affected`)
		};
	}
	const whyProse = anchoredShippedProse(
		text(obj.why, `${context}.why`),
		REVIEW_QUEUE_CEILING_WHY_TWIN,
		`${context}.why`
	);
	return {
		threshold,
		referenceArm: text(obj.reference_arm, `${context}.reference_arm`),
		// The frozen key resolved to its on-screen name. The page rendered the raw
		// key here — the seventh instance of that leak — so resolve it once, at
		// the boundary, and gate if the key names no known arm.
		referenceArmDisplay: (() => {
			const key = text(obj.reference_arm, `${context}.reference_arm`);
			const spec = REVIEW_QUEUE_ARM_SPECS.find((candidate) => candidate.name === key);
			if (!spec) fail(`${context}.reference_arm`, `names no arm on this figure: ${key}`);
			return spec!.display;
		})(),
		nTrue,
		nTrueBelowThreshold,
		perArmOverlap,
		why: whyProse.shipped,
		whyProse
	};
}

/**
 * The equal-yield block, fail-closed and CROSS-CHECKED against the bars the
 * figure already draws. Nothing here is taken on trust:
 *
 *   · every reference must be a drawn LLM gate whose zero pile IS the budget and
 *     catch claimed, so the callout cannot cite an operating point that is not on
 *     the chart;
 *   · every comparator must be a drawn PAPER model, so the oracle disclosure
 *     cannot be attached to something that never got an oracle;
 *   · `extraReviews`, `precisionAtEqualYield` and `shortfallAtReferenceBudget`
 *     must reconstruct from their parts, so no printed delta can drift;
 *   · the sweep must start at 0, end at every error, never decrease, and never
 *     claim more errors than reviews;
 *   · the sweep and the equal-yield budget must BRACKET each other — the grid
 *     point below `budgetForEqualYield` catches strictly less than the reference,
 *     the grid point at or above it catches at least as much. The two numbers are
 *     derived independently in the script, so this is a real cross-derivation
 *     check rather than a restatement.
 */
function parseEqualYield(
	value: unknown,
	context: string,
	panel: ReviewQueuePanel,
	arms: ReviewQueueArm[]
): ReviewQueueEqualYield {
	const obj = record(value, context);
	const byName = new Map(arms.map((arm) => [arm.name, arm] as const));

	const rawRefs = obj.references;
	if (!Array.isArray(rawRefs) || rawRefs.length === 0) {
		fail(`${context}.references`, 'expected a non-empty array');
	}
	const references = rawRefs.map((entry, index): ReviewQueueEqualYieldReference => {
		const at = `${context}.references[${index}]`;
		const ref = record(entry, at);
		const arm = byName.get(text(ref.arm, `${at}.arm`));
		if (!arm) fail(`${at}.arm`, 'must name one of the drawn arms');
		if (arm.kind !== 'llm-gate' || arm.zeroPile === null) {
			fail(`${at}.arm`, 'a reference operating point is a reader gate zero pile');
		}
		const budget = positiveInteger(ref.budget, `${at}.budget`);
		const trueErrorsCaught = positiveInteger(ref.true_errors_caught, `${at}.true_errors_caught`);
		const falseAlarms = nonNegativeInteger(ref.false_alarms, `${at}.false_alarms`);
		// The whole claim of this block: the reference is the bar already drawn.
		if (
			budget !== arm.zeroPile.size ||
			trueErrorsCaught !== arm.zeroPile.trueErrors ||
			falseAlarms !== arm.zeroPile.falseAlarms
		) {
			fail(at, 'reference must be the arm’s own zero pile, exactly');
		}
		if (boolean(ref.threshold_fitted_on_this_panel, `${at}.threshold_fitted_on_this_panel`)) {
			fail(`${at}.threshold_fitted_on_this_panel`, 'the reference point is not fitted');
		}

		const rawComparators = ref.comparators;
		if (!Array.isArray(rawComparators) || rawComparators.length === 0) {
			fail(`${at}.comparators`, 'expected a non-empty array');
		}
		const comparators = rawComparators.map((cell, cellIndex): ReviewQueueEqualYieldComparator => {
			const ct = `${at}.comparators[${cellIndex}]`;
			const c = record(cell, ct);
			const other = byName.get(text(c.arm, `${ct}.arm`));
			if (!other) fail(`${ct}.arm`, 'must name one of the drawn arms');
			if (other.kind !== 'paper-model') {
				fail(`${ct}.arm`, 'only a paper belief model is given an oracle threshold');
			}
			if (!boolean(c.threshold_fitted_on_this_panel, `${ct}.threshold_fitted_on_this_panel`)) {
				fail(`${ct}.threshold_fitted_on_this_panel`, 'the comparator threshold IS fitted here');
			}
			const budgetForEqualYield = positiveInteger(
				c.budget_for_equal_yield,
				`${ct}.budget_for_equal_yield`
			);
			if (budgetForEqualYield > panel.n) fail(`${ct}.budget_for_equal_yield`, 'cannot exceed the panel');
			const extraReviews = finite(c.extra_reviews, `${ct}.extra_reviews`);
			if (extraReviews !== budgetForEqualYield - budget) {
				fail(`${ct}.extra_reviews`, 'must equal budget_for_equal_yield − the reference budget');
			}
			const precisionAtEqualYield = unit(c.precision_at_equal_yield, `${ct}.precision_at_equal_yield`);
			close(
				precisionAtEqualYield,
				trueErrorsCaught / budgetForEqualYield,
				`${ct}.precision_at_equal_yield`,
				'must equal the reference catch / budget_for_equal_yield'
			);
			const errorsCaughtAtReferenceBudget = finite(
				c.errors_caught_at_reference_budget,
				`${ct}.errors_caught_at_reference_budget`
			);
			if (
				errorsCaughtAtReferenceBudget < 0 ||
				errorsCaughtAtReferenceBudget > Math.min(budget, panel.nErrors) + REVIEW_QUEUE_PARITY_TOL
			) {
				fail(`${ct}.errors_caught_at_reference_budget`, 'cannot exceed the reviews or the errors');
			}
			const shortfallAtReferenceBudget = finite(
				c.shortfall_at_reference_budget,
				`${ct}.shortfall_at_reference_budget`
			);
			close(
				shortfallAtReferenceBudget,
				trueErrorsCaught - errorsCaughtAtReferenceBudget,
				`${ct}.shortfall_at_reference_budget`,
				'must equal the reference catch − the comparator catch at that budget'
			);
			return {
				arm: other.name,
				budgetForEqualYield,
				extraReviews,
				precisionAtEqualYield,
				errorsCaughtAtReferenceBudget,
				shortfallAtReferenceBudget,
				thresholdFittedOnThisPanel: true
			};
		});

		const originProse = anchoredShippedProse(
			text(ref.origin, `${at}.origin`),
			REVIEW_QUEUE_ORIGIN_TWIN,
			`${at}.origin`
		);
		return {
			arm: arm.name,
			budget,
			trueErrorsCaught,
			falseAlarms,
			precision: unit(ref.precision, `${at}.precision`),
			recall: unit(ref.recall, `${at}.recall`),
			thresholdFittedOnThisPanel: false,
			origin: originProse.shipped,
			originProse,
			isWholeFlagSetAtHeadlineTarget: boolean(
				ref.is_whole_flag_set_at_headline_target,
				`${at}.is_whole_flag_set_at_headline_target`
			),
			comparators
		};
	});

	const referenceArm = text(obj.reference_arm, `${context}.reference_arm`);
	const reference = references.find((entry) => entry.arm === referenceArm);
	if (!reference) fail(`${context}.reference_arm`, 'must name one of the references');
	if (!reference.isWholeFlagSetAtHeadlineTarget) {
		fail(`${context}.reference_arm`, 'the drawn reference must be the arm’s whole flag set');
	}
	// Structurally the same arm `reviewQueueCalloutArm` names, so the operational
	// callout and the zero-pile bar it points at can never come apart.
	if (
		references.some(
			(entry) => entry.isWholeFlagSetAtHeadlineTarget && entry.budget < reference.budget
		)
	) {
		fail(`${context}.reference_arm`, 'another gate operates at its own zero pile on a shorter queue');
	}

	// ---- the sweep -----------------------------------------------------------
	const sweepAt = `${context}.budget_sweep`;
	const sweepRaw = record(obj.budget_sweep, sweepAt);
	if (!Array.isArray(sweepRaw.budgets) || sweepRaw.budgets.length < 2) {
		fail(`${sweepAt}.budgets`, 'expected at least two budgets');
	}
	const budgets = sweepRaw.budgets.map((entry, index) =>
		nonNegativeInteger(entry, `${sweepAt}.budgets[${index}]`)
	);
	if (budgets[0] !== 0 || budgets[budgets.length - 1] !== panel.n) {
		fail(`${sweepAt}.budgets`, 'must run from 0 reviews to the whole panel');
	}
	budgets.forEach((budget, index) => {
		if (index > 0 && budget <= budgets[index - 1]) {
			fail(`${sweepAt}.budgets[${index}]`, 'must be strictly increasing');
		}
	});
	for (const entry of references) {
		if (!budgets.includes(entry.budget)) {
			fail(`${sweepAt}.budgets`, `must contain the ${entry.arm} operating point exactly`);
		}
	}

	const caughtRaw = record(sweepRaw.errors_caught, `${sweepAt}.errors_caught`);
	const errorsCaught: Record<string, number[]> = {};
	for (const arm of arms) {
		const at = `${sweepAt}.errors_caught["${arm.name}"]`;
		const row = caughtRaw[arm.name];
		if (!Array.isArray(row) || row.length !== budgets.length) {
			fail(at, `expected ${budgets.length} values, one per budget`);
		}
		const values = row.map((entry, index) => finite(entry, `${at}[${index}]`));
		if (values[0] !== 0) fail(`${at}[0]`, 'zero reviews find zero errors');
		close(
			values[values.length - 1],
			panel.nErrors,
			`${at}[last]`,
			'reviewing the whole panel must find every error'
		);
		values.forEach((value, index) => {
			if (index > 0 && value + REVIEW_QUEUE_PARITY_TOL < values[index - 1]) {
				fail(`${at}[${index}]`, 'a larger budget cannot find fewer errors');
			}
			if (value > Math.min(budgets[index], panel.nErrors) + REVIEW_QUEUE_PARITY_TOL) {
				fail(`${at}[${index}]`, 'cannot find more errors than reviews, or than the panel holds');
			}
		});
		errorsCaught[arm.name] = values;
	}
	if (Object.keys(caughtRaw).length !== arms.length) {
		fail(`${sweepAt}.errors_caught`, 'must carry every drawn arm and nothing else');
	}

	const comparatorArm = text(sweepRaw.comparator_arm, `${sweepAt}.comparator_arm`);
	const comparator = reference.comparators.find((entry) => entry.arm === comparatorArm);
	if (!comparator) fail(`${sweepAt}.comparator_arm`, 'must be a comparator of the reference arm');
	// Derived, never named: the drawn comparator is the STRONGEST one.
	if (reference.comparators.some((entry) => entry.budgetForEqualYield < comparator.budgetForEqualYield)) {
		fail(`${sweepAt}.comparator_arm`, 'another comparator needs fewer reviews for the same yield');
	}

	if (!Array.isArray(sweepRaw.advantage) || sweepRaw.advantage.length !== budgets.length) {
		fail(`${sweepAt}.advantage`, `expected ${budgets.length} values`);
	}
	const referenceRow = errorsCaught[reference.arm];
	const comparatorRow = errorsCaught[comparator.arm];
	const advantage = sweepRaw.advantage.map((entry, index) => {
		const parsed = finite(entry, `${sweepAt}.advantage[${index}]`);
		close(
			parsed,
			referenceRow[index] - comparatorRow[index],
			`${sweepAt}.advantage[${index}]`,
			'must equal the reference catch minus the comparator catch'
		);
		return parsed;
	});

	// The sweep and the equal-yield budget are derived independently; make them
	// agree or draw neither. The grid point below the budget must fall short, and
	// the first at or above it must not.
	const below = budgets.filter((budget) => budget < comparator.budgetForEqualYield).at(-1);
	const above = budgets.find((budget) => budget >= comparator.budgetForEqualYield);
	if (below !== undefined) {
		const caught = comparatorRow[budgets.indexOf(below)];
		if (caught + REVIEW_QUEUE_PARITY_TOL >= reference.trueErrorsCaught) {
			fail(`${sweepAt}`, 'the sweep reaches the reference yield below budget_for_equal_yield');
		}
	}
	if (above !== undefined) {
		const caught = comparatorRow[budgets.indexOf(above)];
		if (caught + REVIEW_QUEUE_PARITY_TOL < reference.trueErrorsCaught) {
			fail(`${sweepAt}`, 'the sweep has not reached the reference yield at budget_for_equal_yield');
		}
	}
	// The reference's own curve must pass exactly through the bar the figure draws.
	close(
		referenceRow[budgets.indexOf(reference.budget)],
		reference.trueErrorsCaught,
		`${sweepAt}.errors_caught["${reference.arm}"]`,
		'must pass through the reference operating point'
	);

	const peakBudget = nonNegativeInteger(sweepRaw.peak_budget, `${sweepAt}.peak_budget`);
	const peakIndex = budgets.indexOf(peakBudget);
	if (peakIndex < 0) fail(`${sweepAt}.peak_budget`, 'must be one of the swept budgets');
	const peakAdvantage = finite(sweepRaw.peak_advantage, `${sweepAt}.peak_advantage`);
	close(peakAdvantage, advantage[peakIndex], `${sweepAt}.peak_advantage`, 'must be the advantage there');
	if (advantage.some((value) => value > peakAdvantage + REVIEW_QUEUE_PARITY_TOL)) {
		fail(`${sweepAt}.peak_budget`, 'a larger advantage occurs at another budget');
	}
	const firstPositiveBudget = nonNegativeInteger(
		sweepRaw.first_positive_budget,
		`${sweepAt}.first_positive_budget`
	);
	const firstIndex = budgets.indexOf(firstPositiveBudget);
	if (firstIndex < 0 || advantage[firstIndex] <= 0) {
		fail(`${sweepAt}.first_positive_budget`, 'must be the first swept budget with a positive advantage');
	}
	if (advantage.slice(0, firstIndex).some((value) => value > 0)) {
		fail(`${sweepAt}.first_positive_budget`, 'an earlier budget already leads');
	}
	const halfPeakDecayBudget =
		sweepRaw.half_peak_decay_budget === null
			? null
			: nonNegativeInteger(sweepRaw.half_peak_decay_budget, `${sweepAt}.half_peak_decay_budget`);
	if (halfPeakDecayBudget !== null) {
		const decayIndex = budgets.indexOf(halfPeakDecayBudget);
		if (decayIndex <= peakIndex || advantage[decayIndex] > peakAdvantage / 2) {
			fail(`${sweepAt}.half_peak_decay_budget`, 'must be the first budget past the peak at half of it');
		}
	}

	// Parsed exactly as before; the plain restatement is attached beside it, and the
	// flat string stays byte-identical to `…Prose.shipped`.
	const operatingRuleProse: ShippedProse = {
		shipped: text(obj.operating_rule, `${context}.operating_rule`),
		plain: REVIEW_QUEUE_PLAIN.operatingRule
	};
	const oracleDisclosureProse: ShippedProse = {
		shipped: text(obj.oracle_disclosure, `${context}.oracle_disclosure`),
		plain: REVIEW_QUEUE_PLAIN.oracleDisclosure
	};
	const sweepNoteProse: ShippedProse = {
		shipped: text(sweepRaw.note, `${sweepAt}.note`),
		plain: REVIEW_QUEUE_PLAIN.budgetSweepNote
	};

	return {
		operatingRule: operatingRuleProse.shipped,
		oracleDisclosure: oracleDisclosureProse.shipped,
		operatingRuleProse,
		oracleDisclosureProse,
		referenceArm,
		references,
		budgetSweep: {
			step: positiveInteger(sweepRaw.step, `${sweepAt}.step`),
			budgets,
			errorsCaught,
			comparatorArm,
			advantage,
			firstPositiveBudget,
			peakBudget,
			peakAdvantage,
			halfPeakDecayBudget,
			note: sweepNoteProse.shipped,
			noteProse: sweepNoteProse
		}
	};
}

/**
 * One robustness panel, fail-closed. Nothing is taken on trust:
 *
 *   · the budget must be exactly `floor(share x n)` — recomputed here, so a
 *     hand-edited budget cannot move the interval off the point it is quoted at;
 *   · every delta must reconstruct as `errorRecall − referenceErrorRecall`;
 *   · the simultaneous band must CONTAIN its own point estimate (it is
 *     `centre ± crit x se`, so a band that excludes its centre is corrupt) and
 *     the two shipped `excludes_zero_*` flags must agree with their own endpoints
 *     — they gate here and go no further, because what the panel carries OUT is a
 *     three-way `Standing` and never a sign-blind boolean;
 *   · the max-t critical value must sit strictly above the pointwise normal
 *     quantile and no higher than Bonferroni — a "simultaneous" band narrower
 *     than a pointwise one is not a correction, it is a mistake.
 */
function parseRobustnessPanel(
	value: unknown,
	context: string,
	role: 'primary' | 'sensitivity',
	share: number,
	pointwiseZ: number,
	family: string[]
): ReviewQueueRobustnessPanel {
	const obj = record(value, context);
	if (obj.role !== role) fail(`${context}.role`, `expected ${role}`);
	const nStatements = positiveInteger(obj.n_statements, `${context}.n_statements`);
	const nErrors = positiveInteger(obj.n_errors, `${context}.n_errors`);
	if (nErrors >= nStatements) fail(`${context}.n_errors`, 'cannot be the whole panel');
	const budget = positiveInteger(obj.budget, `${context}.budget`);
	if (budget !== Math.floor(share * nStatements)) {
		fail(`${context}.budget`, 'must be floor(budget_share x n_statements)');
	}
	const maxT = finite(obj.max_t_critical_value, `${context}.max_t_critical_value`);
	const bonferroni = finite(obj.bonferroni_critical_value, `${context}.bonferroni_critical_value`);
	if (!(maxT > pointwiseZ) || !(maxT <= bonferroni)) {
		fail(
			`${context}.max_t_critical_value`,
			'a simultaneous band cannot be narrower than pointwise nor wider than Bonferroni'
		);
	}
	const referenceErrorRecall = unit(obj.reference_error_recall, `${context}.reference_error_recall`);

	const rawArms = record(obj.arms, `${context}.arms`);
	if (Object.keys(rawArms).length !== family.length) {
		fail(`${context}.arms`, `expected exactly the ${family.length} family arms`);
	}
	const arms: Record<string, ReviewQueueRobustnessArm> = {};
	for (const name of family) {
		const at = `${context}.arms["${name}"]`;
		const row = record(rawArms[name], at);
		const errorRecall = unit(row.error_recall, `${at}.error_recall`);
		const delta = finite(row.delta, `${at}.delta`);
		close(delta, errorRecall - referenceErrorRecall, `${at}.delta`, 'must equal this arm’s error recall − the reference’s');
		const low = finite(row.simultaneous_low, `${at}.simultaneous_low`);
		const high = finite(row.simultaneous_high, `${at}.simultaneous_high`);
		if (!(low < high)) fail(`${at}.simultaneous_low`, 'must be below simultaneous_high');
		if (delta < low - REVIEW_QUEUE_PARITY_TOL || delta > high + REVIEW_QUEUE_PARITY_TOL) {
			fail(at, 'the simultaneous band must contain its own point estimate');
		}
		const ci95Low = finite(row.ci95_low, `${at}.ci95_low`);
		const ci95High = finite(row.ci95_high, `${at}.ci95_high`);
		if (!(ci95Low < ci95High)) fail(`${at}.ci95_low`, 'must be below ci95_high');
		// A simultaneous band is the pointwise one widened; it can never be inside it.
		if (low > ci95Low + REVIEW_QUEUE_PARITY_TOL || high < ci95High - REVIEW_QUEUE_PARITY_TOL) {
			fail(at, 'the simultaneous band must contain the pointwise interval');
		}
		// Both shipped flags are still read and still gated against their own
		// endpoints; neither is carried forward. A consumer gets the three-way class.
		if (
			boolean(row.excludes_zero_simultaneous, `${at}.excludes_zero_simultaneous`) !==
			(low > 0 || high < 0)
		) {
			fail(`${at}.excludes_zero_simultaneous`, 'must agree with its own band endpoints');
		}
		if (
			boolean(row.excludes_zero_pointwise, `${at}.excludes_zero_pointwise`) !==
			(ci95Low > 0 || ci95High < 0)
		) {
			fail(`${at}.excludes_zero_pointwise`, 'must agree with its own interval endpoints');
		}
		arms[name] = {
			arm: name,
			errorRecall,
			delta,
			deltaBootstrapMean: finite(row.delta_bootstrap_mean, `${at}.delta_bootstrap_mean`),
			ci95Low,
			ci95High,
			bootstrapSe: finite(row.bootstrap_se, `${at}.bootstrap_se`),
			pGreaterThanZero: unit(row.p_greater_than_zero, `${at}.p_greater_than_zero`),
			simultaneousLow: low,
			simultaneousHigh: high,
			pointwiseStanding: standingOfBounds(ci95Low, ci95High),
			simultaneousStanding: standingOfBounds(low, high),
			nValidResamples: positiveInteger(row.n_valid_resamples, `${at}.n_valid_resamples`)
		};
	}

	return {
		id: text(obj.id, `${context}.id`),
		role,
		nStatements,
		nErrors,
		budget,
		budgetShareOfPanel: unit(obj.budget_share_of_panel, `${context}.budget_share_of_panel`),
		referenceErrorRecall,
		maxTCriticalValue: maxT,
		bonferroniCriticalValue: bonferroni,
		arms
	};
}

/**
 * The whole robustness block, plus the cross-derivation that makes it belong to
 * this figure rather than beside it: every arm's bootstrapped error recall must
 * equal the SWEPT catch at the same budget, divided by the panel errors. The two
 * are computed by different code paths in the emitting script (a block walk for
 * the curve, a sorted-prefix cut inside the bootstrap), so agreeing is a real
 * check — and it means the interval is quoted at a point the drawn curve
 * actually passes through.
 */
function parseRobustness(
	value: unknown,
	context: string,
	panel: ReviewQueuePanel,
	arms: ReviewQueueArm[],
	equalYield: ReviewQueueEqualYield
): ReviewQueueErrorRecallRobustness {
	const obj = record(value, context);
	const byName = new Map(arms.map((arm) => [arm.name, arm] as const));

	const referenceArm = text(obj.reference_arm, `${context}.reference_arm`);
	const reference = byName.get(referenceArm);
	if (!reference || reference.kind !== 'paper-model') {
		fail(`${context}.reference_arm`, 'must name a drawn belief model');
	}
	// The comparator has to be the arm whose model the paper actually published;
	// bootstrapping against one of our own ports would answer a different question.
	if (reference.provenance !== 'paper-published') {
		fail(`${context}.reference_arm`, 'the comparator must be the paper’s own published model');
	}
	const headlineArm = text(obj.headline_arm, `${context}.headline_arm`);
	const headline = byName.get(headlineArm);
	if (!headline || headline.kind !== 'llm-gate') {
		fail(`${context}.headline_arm`, 'must name a drawn reader gate');
	}
	// The quoted gate is the SAME arm the operational callout draws, or the two
	// halves of the claim would be about different models.
	if (headlineArm !== equalYield.referenceArm) {
		fail(`${context}.headline_arm`, 'must be the equal-yield reference arm');
	}

	const multiplicity = record(obj.multiplicity, `${context}.multiplicity`);
	const rawFamily = multiplicity.family;
	if (!Array.isArray(rawFamily) || rawFamily.length === 0) {
		fail(`${context}.multiplicity.family`, 'expected a non-empty array');
	}
	const family = rawFamily.map((entry, index) =>
		text(entry, `${context}.multiplicity.family[${index}]`)
	);
	const gates = arms.filter((arm) => arm.kind === 'llm-gate').map((arm) => arm.name);
	if (family.length !== gates.length || !gates.every((name) => family.includes(name))) {
		fail(
			`${context}.multiplicity.family`,
			'the corrected family must be exactly the drawn reader gates'
		);
	}
	if (!family.includes(headlineArm)) {
		fail(`${context}.headline_arm`, 'must be inside the corrected family');
	}

	const budgetShare = unit(obj.budget_share, `${context}.budget_share`);
	if (budgetShare <= 0 || budgetShare >= 1) {
		fail(`${context}.budget_share`, 'expected a share strictly inside (0, 1)');
	}
	const pointwiseZ = finite(
		multiplicity.pointwise_normal_critical_value,
		`${context}.multiplicity.pointwise_normal_critical_value`
	);
	const panels = record(obj.panels, `${context}.panels`);
	const primary = parseRobustnessPanel(
		panels.primary,
		`${context}.panels.primary`,
		'primary',
		budgetShare,
		pointwiseZ,
		family
	);
	const sensitivity = parseRobustnessPanel(
		panels.sensitivity,
		`${context}.panels.sensitivity`,
		'sensitivity',
		budgetShare,
		pointwiseZ,
		family
	);
	if (primary.nStatements !== panel.n || primary.nErrors !== panel.nErrors) {
		fail(`${context}.panels.primary`, 'the primary panel must be the figure’s own panel');
	}
	// The completeness check drops NEGATIVES only. Stated as an identity on the
	// two censuses rather than trusted from a flag: the positives must be equal.
	if (primary.nStatements - primary.nErrors !== sensitivity.nStatements - sensitivity.nErrors) {
		fail(`${context}.panels.sensitivity`, 'the sensitivity panel must drop negatives only');
	}
	if (sensitivity.nStatements >= primary.nStatements) {
		fail(`${context}.panels.sensitivity`, 'must be a strict subset of the primary panel');
	}

	// The cross-derivation: the interval is quoted at a point on the drawn curve.
	const sweep = equalYield.budgetSweep;
	const at = sweep.budgets.indexOf(primary.budget);
	if (at < 0) fail(`${context}.panels.primary.budget`, 'must be one of the swept budgets');
	close(
		sweep.errorsCaught[referenceArm][at] / panel.nErrors,
		primary.referenceErrorRecall,
		`${context}.panels.primary.reference_error_recall`,
		'must equal the reference arm’s swept catch at that budget'
	);
	for (const name of family) {
		close(
			sweep.errorsCaught[name][at] / panel.nErrors,
			primary.arms[name].errorRecall,
			`${context}.panels.primary.arms["${name}"].error_recall`,
			'must equal that arm’s swept catch at that budget'
		);
	}

	const completeness = record(obj.label_completeness, `${context}.label_completeness`);
	const nDropped = positiveInteger(completeness.n_dropped, `${context}.label_completeness.n_dropped`);
	if (nDropped !== primary.nErrors - sensitivity.nErrors) {
		fail(`${context}.label_completeness.n_dropped`, 'must be the negatives the sensitivity panel drops');
	}
	if (!boolean(completeness.all_dropped_are_negative, `${context}.label_completeness.all_dropped_are_negative`)) {
		fail(`${context}.label_completeness.all_dropped_are_negative`, 'must be true');
	}

	// Parsed exactly as before; every one of the six sentences now carries the
	// plain restatement a reader is handed, and each flat field below stays
	// byte-identical to its twin's `shipped`.
	const prose: ReviewQueueRobustnessProse = {
		metric: {
			shipped: text(obj.metric, `${context}.metric`),
			plain: REVIEW_QUEUE_PLAIN.robustnessMetric
		},
		budgetRule: {
			shipped: text(obj.budget_rule, `${context}.budget_rule`),
			plain: REVIEW_QUEUE_PLAIN.robustnessBudgetRule
		},
		bootstrapDesign: {
			shipped: text(obj.bootstrap_design, `${context}.bootstrap_design`),
			plain: REVIEW_QUEUE_PLAIN.robustnessBootstrapDesign
		},
		multiplicityMethod: {
			shipped: text(multiplicity.method, `${context}.multiplicity.method`),
			plain: REVIEW_QUEUE_PLAIN.robustnessMultiplicityMethod
		},
		multiplicityNote: {
			shipped: text(multiplicity.note, `${context}.multiplicity.note`),
			plain: REVIEW_QUEUE_PLAIN.robustnessMultiplicityNote
		},
		labelCompletenessNote: {
			shipped: text(completeness.note, `${context}.label_completeness.note`),
			plain: REVIEW_QUEUE_PLAIN.robustnessLabelCompletenessNote
		}
	};

	return {
		metric: prose.metric.shipped,
		budgetShare,
		budgetRule: prose.budgetRule.shipped,
		referenceArm,
		headlineArm,
		seed: nonNegativeInteger(obj.seed, `${context}.seed`),
		nBootstrap: positiveInteger(obj.n_bootstrap, `${context}.n_bootstrap`),
		bootstrapDesign: prose.bootstrapDesign.shipped,
		family,
		familyAlpha: unit(multiplicity.family_alpha, `${context}.multiplicity.family_alpha`),
		multiplicityMethod: prose.multiplicityMethod.shipped,
		pointwiseNormalCriticalValue: pointwiseZ,
		multiplicityNote: prose.multiplicityNote.shipped,
		labelCompleteness: {
			field: text(completeness.field, `${context}.label_completeness.field`),
			nDropped,
			allDroppedAreNegative: true,
			droppedShareOfAllNegatives: unit(
				completeness.dropped_share_of_all_negatives,
				`${context}.label_completeness.dropped_share_of_all_negatives`
			),
			droppedShareOfPanel: unit(
				completeness.dropped_share_of_panel,
				`${context}.label_completeness.dropped_share_of_panel`
			),
			noModelIsRefit: boolean(
				completeness.no_model_is_refit,
				`${context}.label_completeness.no_model_is_refit`
			),
			note: prose.labelCompletenessNote.shipped,
			noteProse: prose.labelCompletenessNote
		},
		primary,
		sensitivity,
		prose
	};
}

function parseArm(
	entry: unknown,
	index: number,
	headlineTarget: number,
	targetRecalls: number[],
	panel: ReviewQueuePanel
): ReviewQueueArm {
	const spec = REVIEW_QUEUE_ARM_SPECS[index];
	const context = `statement_review_queue.arms[${index}]`;
	const arm = record(entry, context);
	if (arm.name !== spec.name) {
		fail(`${context}.name`, `expected the fixed presentation order — ${spec.name}`);
	}
	if (arm.kind !== spec.kind) fail(`${context}.kind`, `expected ${spec.kind}`);
	// Attribution is the one thing this figure cannot get wrong in front of the
	// people who wrote the paper, so the artifact and the display table have to
	// agree about it before anything is drawn.
	if (arm.provenance !== spec.provenance) {
		fail(`${context}.provenance`, `expected ${spec.provenance}`);
	}

	const operatingPoint = parsePoint(
		arm.operating_point,
		`${context}.operating_point`,
		panel.nErrors,
		panel.n
	);
	if (operatingPoint.targetRecall !== headlineTarget) {
		fail(`${context}.operating_point.target_recall`, 'must be the headline target recall');
	}

	const rawGrid = arm.precision_at_matched_recall;
	if (!Array.isArray(rawGrid) || rawGrid.length !== targetRecalls.length) {
		fail(`${context}.precision_at_matched_recall`, `expected ${targetRecalls.length} entries`);
	}
	const grid = rawGrid.map((cell, cellIndex) =>
		parsePoint(
			cell,
			`${context}.precision_at_matched_recall[${cellIndex}]`,
			panel.nErrors,
			panel.n
		)
	);
	grid.forEach((cell, cellIndex) => {
		if (cell.targetRecall !== targetRecalls[cellIndex]) {
			fail(
				`${context}.precision_at_matched_recall[${cellIndex}].target_recall`,
				'must follow the artifact target_recalls in order'
			);
		}
	});
	// The bar draws the headline cell; the grid backs the coarseness caveat. They
	// must be the same measurement, or the caption and the figure disagree.
	const headlineCell = grid.find((cell) => cell.targetRecall === headlineTarget);
	if (!headlineCell || !samePoint(headlineCell, operatingPoint)) {
		fail(`${context}.operating_point`, 'must equal the grid cell at the headline target recall');
	}

	const distinctQueueSizesAcrossTargets = positiveInteger(
		arm.n_distinct_queue_sizes_across_targets,
		`${context}.n_distinct_queue_sizes_across_targets`
	);
	if (distinctQueueSizesAcrossTargets !== new Set(grid.map((cell) => cell.queue)).size) {
		fail(
			`${context}.n_distinct_queue_sizes_across_targets`,
			'must equal the number of distinct queue sizes in the grid'
		);
	}

	// A zero pile is an LLM-gate property. A paper belief model never rejects all
	// of its evidence, and claiming one would mean the artifact changed meaning.
	if (spec.kind === 'paper-model' && arm.zero_pile !== null) {
		fail(`${context}.zero_pile`, 'a paper belief model must not carry a zero pile');
	}
	const zeroPile =
		spec.kind === 'llm-gate'
			? parseZeroPile(arm.zero_pile, `${context}.zero_pile`, panel.nErrors, operatingPoint)
			: null;

	// The note is looked up by the FROZEN arm name, then pinned to its own text: a
	// model with no authored restatement gates the figure rather than reaching a
	// reader in the artifact's dialect, and four models ship the SAME sentence, so
	// the key alone is not enough to bind the right restatement to the right row.
	const noteProse = keyedShippedProse(
		spec.name,
		text(arm.note, `${context}.note`),
		REVIEW_QUEUE_ARM_NOTE_TWINS,
		`${context}.note`
	);

	return {
		name: spec.name,
		display: spec.display,
		kind: spec.kind,
		provenance: spec.provenance,
		paperKind: reviewQueuePaperKind(spec.kind),
		modelKey: text(arm.model_key, `${context}.model_key`),
		note: noteProse.shipped,
		noteProse,
		distinctScores: positiveInteger(arm.distinct_scores, `${context}.distinct_scores`),
		operatingPoint,
		grid,
		distinctQueueSizesAcrossTargets,
		zeroPile,
		scoresPath: text(arm.scores_path, `${context}.scores_path`)
	};
}

/**
 * Pure, fail-closed parse of `statement_review_queue.json`. THROWS on any drift —
 * shape, arm order, segment arithmetic, precision/recall identity, a zero pile
 * that is not the flag set it claims to be, a soft-gated bundle (which would
 * silently break the "belief 0 == rejected everything" identity), a failed check
 * flag, a caveat list that is not the six the figure is required to print, or a
 * promotion ceiling whose bar or count cannot be what it claims.
 */
export function validateReviewQueue(raw: unknown): ReviewQueue {
	const obj = record(raw, 'statement_review_queue');
	if (obj.artifact_kind !== 'statement_review_queue') {
		fail('statement_review_queue.artifact_kind', 'expected statement_review_queue');
	}
	// 2 added `equal_yield`: the same panel with the YIELD fixed instead of the
	// threshold, plus the whole budget curve behind it. 4 added the paper's OWN
	// released RF and INDRA's served hierarchy variant as arms (so the oracle
	// comparator is the paper's published model rather than our port of it), an
	// explicit `provenance` per arm, and `error_recall_robustness` — the
	// bootstrap interval on the operational result.
	//
	// 3 IS DELIBERATELY SKIPPED and must stay unknown forever. The shipped
	// contract runner (viewer/scripts/test-paper-literal-contract.mjs) uses the
	// literal `schema_version = 3` as its negative control for "an unknown
	// schema version fails closed". Shipping 3 would turn that assertion from a
	// real test into a silent pass, which is a worse outcome than a gap in the
	// numbering. No 3 was ever emitted.
	if (obj.schema_version !== 4) fail('statement_review_queue.schema_version', 'expected 4');

	const panelRaw = record(obj.panel, 'statement_review_queue.panel');
	const n = positiveInteger(panelRaw.n, 'statement_review_queue.panel.n');
	const nErrors = positiveInteger(panelRaw.n_errors, 'statement_review_queue.panel.n_errors');
	const nCorrect = positiveInteger(panelRaw.n_correct, 'statement_review_queue.panel.n_correct');
	if (nErrors + nCorrect !== n) {
		fail('statement_review_queue.panel', 'n_errors + n_correct must equal n');
	}
	const errorBaseRate = unit(panelRaw.error_base_rate, 'statement_review_queue.panel.error_base_rate');
	close(
		errorBaseRate,
		nErrors / n,
		'statement_review_queue.panel.error_base_rate',
		'must equal n_errors / n'
	);
	const panel: ReviewQueuePanel = {
		n,
		nErrors,
		nCorrect,
		errorBaseRate,
		label: text(panelRaw.label, 'statement_review_queue.panel.label')
	};

	const headlineTargetRecall = unit(
		obj.headline_target_recall,
		'statement_review_queue.headline_target_recall'
	);
	if (!Array.isArray(obj.target_recalls) || obj.target_recalls.length === 0) {
		fail('statement_review_queue.target_recalls', 'expected a non-empty array');
	}
	const targetRecalls = obj.target_recalls.map((value, index) =>
		unit(value, `statement_review_queue.target_recalls[${index}]`)
	);
	if (!targetRecalls.includes(headlineTargetRecall)) {
		fail('statement_review_queue.headline_target_recall', 'must appear in target_recalls');
	}

	if (!Array.isArray(obj.arms) || obj.arms.length !== REVIEW_QUEUE_ARM_SPECS.length) {
		fail('statement_review_queue.arms', `expected ${REVIEW_QUEUE_ARM_SPECS.length} arms in fixed order`);
	}
	const arms = obj.arms.map((entry, index) =>
		parseArm(entry, index, headlineTargetRecall, targetRecalls, panel)
	);

	const promotionCeiling = parseCeiling(
		obj.promotion_ceiling,
		'statement_review_queue.promotion_ceiling',
		panel
	);
	// The ceiling is read off one of the drawn arms, so it must name one of them.
	if (!arms.some((arm) => arm.name === promotionCeiling.referenceArm)) {
		fail(
			'statement_review_queue.promotion_ceiling.reference_arm',
			'must name one of the drawn arms'
		);
	}

	const equalYield = parseEqualYield(
		obj.equal_yield,
		'statement_review_queue.equal_yield',
		panel,
		arms
	);

	const errorRecallRobustness = parseRobustness(
		obj.error_recall_robustness,
		'statement_review_queue.error_recall_robustness',
		panel,
		arms,
		equalYield
	);

	// The method caveats are printed verbatim; a shorter list means one was dropped.
	if (!Array.isArray(obj.caveats) || obj.caveats.length !== REVIEW_QUEUE_CAVEAT_COUNT) {
		fail('statement_review_queue.caveats', `expected ${REVIEW_QUEUE_CAVEAT_COUNT} caveats`);
	}
	const caveats = obj.caveats.map((entry, index) =>
		text(entry, `statement_review_queue.caveats[${index}]`)
	);
	// Positional twins: a reissued artifact that reorders or rewrites a caveat gates
	// the figure rather than printing restatement N under caveat N+1.
	const caveatProse = pairShippedProse(
		caveats,
		REVIEW_QUEUE_CAVEAT_TWINS,
		'statement_review_queue.caveats'
	);

	// The script asserts these in Python and fails the build on a violation; hold
	// the same line here so a hand-edited artifact cannot slip a false one past.
	const checks = record(obj.checks, 'statement_review_queue.checks');
	for (const key of [
		'queue_equals_real_plus_wasted',
		'precision_equals_real_over_queue',
		'recall_achieved_at_least_target',
		'every_arm_covers_the_panel_exactly',
		'gold_matches_hash_agrees_with_prediction_provenance',
		'llm_zero_pile_is_the_minimum_score_tie_block',
		'equal_yield_budget_is_minimal',
		'equal_yield_reference_is_the_arms_own_zero_pile',
		'budget_sweep_is_monotone_and_closes_at_every_error',
		'two_catch_derivations_agree_at_every_swept_budget',
		'swept_catch_agrees_with_bootstrapped_error_recall',
		'paper_model_key_is_a_published_method_string'
	]) {
		if (boolean(checks[key], `statement_review_queue.checks.${key}`) !== true) {
			fail(`statement_review_queue.checks.${key}`, 'must be true');
		}
	}
	// "belief 0.0 == the reader rejected every piece of evidence" only holds under
	// the unfitted hard gate; a fitted soft profile floors belief at sigmoid(prior).
	if (checks.llm_bundles_use_unfitted_hard_gate !== REVIEW_QUEUE_REQUIRED_AGGREGATION) {
		fail(
			'statement_review_queue.checks.llm_bundles_use_unfitted_hard_gate',
			`expected ${REVIEW_QUEUE_REQUIRED_AGGREGATION}`
		);
	}

	const provenance = record(obj.provenance, 'statement_review_queue.provenance');

	const decisionRuleProse: ShippedProse = {
		shipped: text(obj.decision_rule, 'statement_review_queue.decision_rule'),
		plain: REVIEW_QUEUE_PLAIN.decisionRule
	};
	const thresholdRuleProse: ShippedProse = {
		shipped: text(obj.threshold_rule, 'statement_review_queue.threshold_rule'),
		plain: REVIEW_QUEUE_PLAIN.thresholdRule
	};

	return {
		decisionRule: decisionRuleProse.shipped,
		thresholdRule: thresholdRuleProse.shipped,
		headlineTargetRecall,
		targetRecalls,
		panel,
		promotionCeiling,
		arms,
		equalYield,
		errorRecallRobustness,
		caveats,
		prose: {
			decisionRule: decisionRuleProse,
			thresholdRule: thresholdRuleProse,
			operatingRule: equalYield.operatingRuleProse,
			oracleDisclosure: equalYield.oracleDisclosureProse,
			budgetSweepNote: equalYield.budgetSweep.noteProse,
			promotionCeilingWhy: promotionCeiling.whyProse,
			caveats: caveatProse,
			// The SAME objects the rows carry, in the artifact's own arm order, so a
			// verification boundary and a per-row note can never disagree.
			armNotes: arms.map((arm) => arm.noteProse),
			zeroPileDefinition: (() => {
				const withPile = arms.find((arm) => arm.zeroPile !== null);
				if (!withPile || withPile.zeroPile === null) {
					fail('statement_review_queue.arms', 'no reader gate carries a zero pile to define');
				}
				return withPile.zeroPile.definitionProse;
			})(),
			equalYieldOrigin: (() => {
				const reference = equalYield.references.find(
					(entry) => entry.arm === equalYield.referenceArm
				);
				if (!reference) {
					fail('statement_review_queue.equal_yield.reference_arm', 'must name one of the references');
				}
				return reference.originProse;
			})(),
			robustness: errorRecallRobustness.prose
		},
		generatedBy: text(provenance.generated_by, 'statement_review_queue.provenance.generated_by')
	};
}

/**
 * The robustness rows the figure quotes: the headline gate against the paper's
 * own RF, on the paper's labels and on the adjudication-safe panel, with the arm
 * records so the caption can print display names rather than join keys.
 *
 * Derived, never named: the gate is whichever arm the equal-yield block already
 * made the reference, so the interval and the operational callout can never come
 * apart. Returns null rather than a partial object — the caller gates.
 */
export function reviewQueueRobustnessHeadline(queue: ReviewQueue): {
	robustness: ReviewQueueErrorRecallRobustness;
	gate: ReviewQueueArm;
	reference: ReviewQueueArm;
	primary: ReviewQueueRobustnessArm;
	sensitivity: ReviewQueueRobustnessArm;
} | null {
	const robustness = queue.errorRecallRobustness;
	const gate = queue.arms.find((arm) => arm.name === robustness.headlineArm);
	const reference = queue.arms.find((arm) => arm.name === robustness.referenceArm);
	const primary = robustness.primary.arms[robustness.headlineArm];
	const sensitivity = robustness.sensitivity.arms[robustness.headlineArm];
	if (!gate || !reference || !primary || !sensitivity) return null;
	return { robustness, gate, reference, primary, sensitivity };
}

/** Shared hue resolution — the same token the other /paper panels use. */
export function reviewQueueColorVar(arm: ReviewQueueArm): string {
	return paperArmColorVar(arm.paperKind);
}

/**
 * Display order: shortest queue first, so the reader's eye runs from least work
 * to most. Ties (none today) fall back to the artifact's fixed arm order, which
 * keeps the layout deterministic between SSR and the client.
 */
export function reviewQueueDisplayOrder(queue: ReviewQueue): ReviewQueueArm[] {
	const rank = new Map(REVIEW_QUEUE_ARM_SPECS.map((spec, index) => [spec.name, index] as const));
	return [...queue.arms].sort(
		(a, b) =>
			a.operatingPoint.queue - b.operatingPoint.queue ||
			(rank.get(a.name) ?? 0) - (rank.get(b.name) ?? 0)
	);
}

/**
 * The arm the zero-pile callout names: among the arms whose zero pile IS their
 * entire flag set, the one with the shortest queue — i.e. the top bar in the
 * figure, which is what the reader is already looking at. Derived, never named
 * by hand, so the callout follows the data if the panel changes.
 */
/** One placed point on a swept series, in SVG user units. */
export interface ReviewQueueSweepPoint {
	budget: number;
	caught: number;
	x: number;
	y: number;
}

export interface ReviewQueueSweepSeries {
	arm: ReviewQueueArm;
	points: ReviewQueueSweepPoint[];
	/** `points` as an SVG polyline `points` string. */
	polyline: string;
	color: string;
	/** Left-anchored rail label, already checked against the gutter budget. */
	label: string;
	labelY: number;
}

export interface ReviewQueueSweepFigure {
	geometry: typeof REVIEW_QUEUE_SWEEP_GEOMETRY;
	reference: ReviewQueueSweepSeries;
	comparator: ReviewQueueSweepSeries;
	/** Filled polygons over the budgets where the reference leads. May be empty. */
	leadBands: string[];
	/** The same, where the COMPARATOR leads. Drawn, not omitted. */
	deficitBands: string[];
	xTicks: { value: number; x: number }[];
	yTicks: { value: number; y: number }[];
	/** The reference arm's own operating point, and the two deltas drawn from it. */
	marker: {
		budget: number;
		x: number;
		yReference: number;
		yComparator: number;
		caught: number;
		comparatorCaught: number;
		advantage: number;
	};
	/** The oracle bracket: same yield, further right. */
	equalYield: {
		budget: number;
		x: number;
		y: number;
		extraReviews: number;
	};
	maxBudget: number;
	maxErrors: number;
}

function budgetChars(label: string, chars: number, context: string): string {
	if (label.length > chars) {
		fail(context, `"${label}" is ${label.length} chars; the gutter budget is ${chars}`);
	}
	return label;
}

function r2(value: number): number {
	return Math.round(value * 100) / 100;
}

/**
 * Place the budget sweep. THROWS on any drift the validator cannot see —
 * an arm the sweep does not cover, a degenerate axis, or a display name that
 * would overrun its rail — so the caller gates the panel instead of drawing a
 * clipped label or a flat line.
 *
 * The lead bands are the region between the two series wherever the reference
 * leads, with the zero crossings interpolated, so the figure shows the advantage
 * opening and closing without a word of caption. `arms` display names are ALSO
 * budget-checked against the bar panel's left gutter here, because a rename is
 * the one change that breaks both figures at once and neither one loudly.
 */
export function buildReviewQueueSweep(queue: ReviewQueue): ReviewQueueSweepFigure {
	const g = REVIEW_QUEUE_SWEEP_GEOMETRY;
	const sweep = queue.equalYield.budgetSweep;
	const byName = new Map(queue.arms.map((arm) => [arm.name, arm] as const));

	for (const arm of queue.arms) {
		budgetChars(arm.display, REVIEW_QUEUE_GUTTER_BUDGET_CHARS, `arm[${arm.name}].display`);
	}

	const reference = byName.get(queue.equalYield.referenceArm);
	const comparator = byName.get(sweep.comparatorArm);
	if (!reference || !comparator) {
		fail('sweep', 'the sweep names an arm the panel does not draw');
	}
	const refPoint = queue.equalYield.references.find((entry) => entry.arm === reference.name);
	const cmpPoint = refPoint?.comparators.find((entry) => entry.arm === comparator.name);
	if (!refPoint || !cmpPoint) fail('sweep', 'the drawn pair has no equal-yield entry');

	const maxBudget = sweep.budgets[sweep.budgets.length - 1];
	const maxErrors = queue.panel.nErrors;
	if (maxBudget <= 0 || maxErrors <= 0) fail('sweep', 'degenerate axis range');

	const x = (budget: number) =>
		g.plotLeft + (budget / maxBudget) * (g.plotRight - g.plotLeft);
	const y = (caught: number) =>
		g.plotBottom - (caught / maxErrors) * (g.plotBottom - g.plotTop);

	const place = (arm: ReviewQueueArm, labelY: number): ReviewQueueSweepSeries => {
		const row = sweep.errorsCaught[arm.name];
		if (!row) fail('sweep', `no swept curve for ${arm.name}`);
		const points = row.map((caught, index) => ({
			budget: sweep.budgets[index],
			caught,
			x: r2(x(sweep.budgets[index])),
			y: r2(y(caught))
		}));
		return {
			arm,
			points,
			polyline: points.map((point) => `${point.x},${point.y}`).join(' '),
			color: reviewQueueColorVar(arm),
			label: budgetChars(
				arm.display,
				REVIEW_QUEUE_SWEEP_LABEL_BUDGET_CHARS,
				`sweep[${arm.name}].label`
			),
			labelY: r2(y(row[row.length - 1]))
		};
	};

	const referenceSeries = place(reference, 0);
	const comparatorSeries = place(comparator, 0);
	// Both series end at every error, so their rail labels would collide. Split
	// them around the shared endpoint rather than stacking them on it.
	const endY = referenceSeries.labelY;
	referenceSeries.labelY = r2(endY - 6);
	comparatorSeries.labelY = r2(endY + 9);

	/**
	 * The region between the two curves, split by who is ahead. The zero
	 * crossings are interpolated so a band closes on the curves rather than on
	 * the grid point after them, and a run CONTINUES through an exact tie —
	 * otherwise a single 0.0 in the converged tail chops one band into four for
	 * no reason a reader could see.
	 *
	 * Both signs are returned. The deficit band is what makes budget-dependence
	 * visible instead of asserted: at small budgets the paper model is ahead, and
	 * a reader who suspects a cherry-picked operating point can see it.
	 */
	const cross = (i: number, j: number) => {
		const a = sweep.advantage[i];
		const b = sweep.advantage[j];
		const t = a === b ? 0 : a / (a - b);
		const budget = sweep.budgets[i] + t * (sweep.budgets[j] - sweep.budgets[i]);
		const caught =
			referenceSeries.points[i].caught +
			t * (referenceSeries.points[j].caught - referenceSeries.points[i].caught);
		return `${r2(x(budget))},${r2(y(caught))}`;
	};
	const bandsWhere = (ahead: 1 | -1): string[] => {
		const out: string[] = [];
		let run: number[] = [];
		const flush = (endsAt: string | null) => {
			if (run.length === 0) return;
			const entry = run[0] > 0 ? cross(run[0] - 1, run[0]) : null;
			const top = run.map((i) => `${referenceSeries.points[i].x},${referenceSeries.points[i].y}`);
			const bottom = [...run]
				.reverse()
				.map((i) => `${comparatorSeries.points[i].x},${comparatorSeries.points[i].y}`);
			const head = entry === null ? [] : [entry];
			const tail = endsAt === null ? [] : [endsAt];
			out.push(`M ${[...head, ...top, ...tail, ...bottom].join(' L ')} Z`);
			run = [];
		};
		for (let i = 0; i < sweep.advantage.length; i += 1) {
			const value = sweep.advantage[i] * ahead;
			if (value > 0 || (value === 0 && run.length > 0)) {
				run.push(i);
				continue;
			}
			if (run.length) flush(cross(run[run.length - 1], i));
		}
		flush(null);
		return out;
	};
	const leadBands = bandsWhere(1);
	const deficitBands = bandsWhere(-1);

	// The reference arm's own budget carries its own emphasised tick, so a regular
	// tick landing within a label's width of it is dropped rather than overprinted.
	// A 4-digit label is 4 x 5.4186 = 21.7 units wide and both are centre-anchored,
	// so 24 units of separation is one clear label-width between their edges.
	const markerX = r2(x(refPoint.budget));
	const xTicks: { value: number; x: number }[] = [];
	const pushTick = (value: number) => {
		const at = r2(x(value));
		if (Math.abs(at - markerX) < 24) return;
		if (xTicks[xTicks.length - 1]?.value === value) return;
		xTicks.push({ value, x: at });
	};
	for (let value = 0; value <= maxBudget; value += 500) pushTick(value);
	// The panel end anchors the right of the axis, but only once.
	if (xTicks[xTicks.length - 1]?.value !== maxBudget) pushTick(maxBudget);
	const yTicks: { value: number; y: number }[] = [];
	for (let value = 0; value <= maxErrors; value += 100) yTicks.push({ value, y: r2(y(value)) });
	// The panel's own error count anchors the top of the axis, but only once.
	if (yTicks[yTicks.length - 1]?.value !== maxErrors) {
		yTicks.push({ value: maxErrors, y: r2(y(maxErrors)) });
	}
	for (const tick of yTicks) {
		budgetChars(String(tick.value), REVIEW_QUEUE_AXIS_BUDGET_CHARS, 'sweep.yTick');
	}

	return {
		geometry: g,
		reference: referenceSeries,
		comparator: comparatorSeries,
		leadBands,
		deficitBands,
		xTicks,
		yTicks,
		marker: {
			budget: refPoint.budget,
			x: markerX,
			yReference: r2(y(refPoint.trueErrorsCaught)),
			yComparator: r2(y(cmpPoint.errorsCaughtAtReferenceBudget)),
			caught: refPoint.trueErrorsCaught,
			comparatorCaught: cmpPoint.errorsCaughtAtReferenceBudget,
			advantage: cmpPoint.shortfallAtReferenceBudget
		},
		equalYield: {
			budget: cmpPoint.budgetForEqualYield,
			x: r2(x(cmpPoint.budgetForEqualYield)),
			y: r2(y(refPoint.trueErrorsCaught)),
			extraReviews: cmpPoint.extraReviews
		},
		maxBudget,
		maxErrors
	};
}

/**
 * The equal-yield entry the figure's callout draws: the reference arm's own row,
 * paired with the comparator the sweep draws. Derived from the artifact so a
 * change of reference arm moves the callout with it.
 */
export function reviewQueueEqualYieldPair(queue: ReviewQueue): {
	reference: ReviewQueueEqualYieldReference;
	comparator: ReviewQueueEqualYieldComparator;
	referenceArm: ReviewQueueArm;
	comparatorArm: ReviewQueueArm;
} | null {
	const reference = queue.equalYield.references.find(
		(entry) => entry.arm === queue.equalYield.referenceArm
	);
	if (!reference) return null;
	const comparator = reference.comparators.find(
		(entry) => entry.arm === queue.equalYield.budgetSweep.comparatorArm
	);
	if (!comparator) return null;
	const referenceArm = queue.arms.find((arm) => arm.name === reference.arm);
	const comparatorArm = queue.arms.find((arm) => arm.name === comparator.arm);
	if (!referenceArm || !comparatorArm) return null;
	return { reference, comparator, referenceArm, comparatorArm };
}

export function reviewQueueCalloutArm(queue: ReviewQueue): ReviewQueueArm | null {
	const candidates = queue.arms.filter(
		(arm) => arm.zeroPile !== null && arm.zeroPile.isWholeFlagSetAtHeadlineTarget
	);
	if (candidates.length === 0) return null;
	return candidates.reduce((best, arm) =>
		arm.operatingPoint.queue < best.operatingPoint.queue ? arm : best
	);
}
