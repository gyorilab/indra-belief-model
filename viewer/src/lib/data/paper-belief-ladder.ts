import { fail, record, number, unit, text, positiveInteger } from './paper-validate.ts';
/**
 * Typed data contract for the BELIEF-MODEL LADDER figure — /paper's beat 3.
 *
 * Source artifact: `data/results/indra_paper_literal_models_20260724/
 * belief_model_ladder.json`, emitted by `scripts/compute_belief_model_ladder.py`.
 * Every rung is pooled average precision on the SAME all-sources-specific panel
 * with the SAME released paper labels, so the 2023 paper's own belief-model
 * lineage, the literal released model, and the reader gates all sit on one axis
 * and one metric. The artifact's baseline is the unfitted noisy-OR — not the
 * paper's best model — because it is what BOTH the paper's engineered features
 * and the reading gate modify, and every delta the figure draws is measured from
 * exactly that.
 *
 * This module is import-safe on the client: typed shape, the fixed entry table,
 * and a pure `validateBeliefLadder()` that THROWS on any shape or ARITHMETIC
 * drift. The server loader (`$lib/server/paper-belief-ladder`) wraps it in its own
 * try/catch and gates the whole load to `unavailable` rather than drawing a rung
 * whose bar length disagrees with the number printed beside it.
 *
 * NOTHING here is hard-coded from the numbers: no average precision, no delta, no
 * panel count, no caveat text. The only constants are the twelve (label, kind)
 * pairs in the artifact's fixed order (the same list `tests/
 * test_viewer_paper_literal_contract.py` pins as `_LADDER_ENTRIES`), the baseline
 * label, the schema version, the metric name, the caveat count, the two noisy-OR
 * formula strings (the right one, and the wrong one that must never reappear), and
 * one arithmetic parity tolerance.
 *
 * Ordering: the artifact's entry order is the VALIDATED one and is asserted
 * verbatim. `beliefLadderDisplayOrder()` re-sorts by average precision ascending —
 * that is a LAYOUT decision made here, not a claim about the data.
 */

import {
	PAPER_LITERAL_ARM_SPECS,
	keyedShippedProse,
	pairShippedProse,
	paperArmColorVar,
	type AnchoredProse,
	type PaperArmKind,
	type ShippedProse
} from './paper-literal.ts';

/**
 * THE PLAIN HALF OF EVERY TWIN THIS MODULE EMITS.
 *
 * `belief_model_ladder.json` is where "pooled average precision", "out-of-fold
 * refit", "the paper's MCMC-refit Belief Orig" and "the reading gate" reach the
 * screen — twelve per-row notes, eight caveats and four header lines, none of
 * them visible to any scan of this repo because they only exist once the file is
 * read. Every disclosure survives the restatement at full strength: the two rows
 * that are ONE fitted model, the consistency check that is NOT fidelity
 * evidence, and which rows the 2023 paper never published are all still here.
 *
 * A restatement carries every CLAIM, every NUMBER, every NAMED MODEL and every
 * RESTRICTION of the sentence it restates. Plain language changes the words; it
 * never changes what is asserted, what is denied, or what is forbidden. The
 * third caveat is why this is written down: it ships the +0.0479 reading-step
 * figure together with the instruction never to quote that figure alone, and an
 * earlier restatement of it dropped the instruction and then quoted the figure
 * alone. `scripts/test-paper-prose-coverage.mjs` now fails on exactly that —
 * a numeral, a negation or a named model in the shipped half with no counterpart
 * in the plain half.
 */
const BELIEF_LADDER_PLAIN = {
	metricSource:
		'average precision computed over all 1,689 statements at once, tie-aware ' +
		'(scikit-learn’s average_precision_score)',
	labelConvention: '“wrong” means the label released with the 2023 paper says incorrect',
	panelOrdering: 'the statements taken in the order of their sorted statement ids',
	baselineWhy:
		'the unfitted combination rule is what BOTH the engineered statement-type / #PMIDs / ' +
		'promoter features and the reading step modify',
	proximityStatus: 'a consistency check ACROSS DIFFERENT CORPORA — not evidence of fidelity',
	fidelityStatistic:
		'the per-statement Pearson correlation between the released code and the semantic port of it',
	checksNote:
		'The checks are enforced in code; breaking one fails the build rather than being reported ' +
		'here as false. A disagreement with a recorded value is NOT one of them: this file carries ' +
		'a re-derived number, records the one the original run shipped beside it, and the ' +
		'script prints a warning.',
	noisyOrFormula:
		'one minus the product, over every source, of that source’s systematic error rate plus its ' +
		'random error rate raised to the number of evidence entries that source supplied',
	join:
		'joined on statement id (the statement_id of the frozen 2023 gold) for every row except ' +
		'the one that re-runs the released code, which joins on the content hash released with ' +
		'it; that hash is cross-checked for all of them'
} as const;

/**
 * The twelve per-row notes, keyed by the FROZEN `entries[].label`. Four of the
 * twelve ship the SAME sentence, so each is additionally pinned to a verbatim
 * fragment — a key alone could bind a reading model's restatement to a scorer.
 */
const BELIEF_LADDER_NOTE_TWINS: Readonly<Record<string, AnchoredProse>> = {
	'CountsScorer RF, source counts': {
		artifactAnchor: 'ONLY the per-source evidence counts',
		plain:
			'A supervised random forest given ONLY the per-source evidence counts the combination ' +
			'rule already sees.'
	},
	'Hierarchy propagation': {
		artifactAnchor: 'propagated over the statement hierarchy',
		plain:
			'The combination rule with belief passed up the statement hierarchy, so a statement also ' +
			'inherits evidence from the more specific statements underneath it.'
	},
	'noisy-OR SimpleScorer (direct)': {
		artifactAnchor: 'published source-prior noisy-OR',
		plain:
			'INDRA’s published per-source combination rule, unfitted, on current evidence: one minus ' +
			'the product, over every source, of that source’s systematic error rate plus its random ' +
			'error rate raised to the number of evidence entries that source supplied.'
	},
	'BayesianScorer, source refit': {
		artifactAnchor: 'refit of the per-source reliabilities',
		plain:
			'A refit of the per-source reliabilities, with every statement scored by a copy that ' +
			'never saw it.'
	},
	'BayesianScorer, source+subtype refit': {
		artifactAnchor: 'source AND source-subtype reliabilities',
		plain:
			'A refit of the source AND source-subtype reliabilities, with every statement scored by ' +
			'a copy that never saw it — the best combination-rule variant on this list, and it is ' +
			'newly fitted here: the 2023 paper publishes no subtype-resolved belief model.'
	},
	'CountsScorer RF, full features': {
		artifactAnchor: 'full CountsScorer feature set',
		plain:
			'A supervised random forest over INDRA’s full CountsScorer feature set, a SUPERSET of ' +
			'the statement-type / #PMIDs / promoter features engineered in the 2023 paper.'
	},
	'HybridScorer, full features': {
		artifactAnchor: 'not two independent results',
		plain:
			'The same fitted random forest served through the hybrid wrapper — two INDRA scorer ' +
			'classes over one model, not two independent results.'
	},
	'RF 2k-d13 + Type/#PMIDs/promoter': {
		artifactAnchor: 'run as-is on the released corpus',
		plain:
			'The code released with the 2023 paper, run as-is on the released corpus — a different ' +
			'codebase and a different snapshot of the evidence.'
	},
	'Gemma 4 26B gate': {
		artifactAnchor: 'the evidence the reader KEPT',
		plain:
			'INDRA’s own unfitted combination rule applied to the evidence the reading model KEPT — ' +
			'not the MCMC-refitted “Belief Orig” of the 2023 paper.'
	},
	'GLM-5 gate': {
		artifactAnchor: 'the evidence the reader KEPT',
		plain:
			'INDRA’s own unfitted combination rule applied to the evidence the reading model KEPT — ' +
			'not the MCMC-refitted “Belief Orig” of the 2023 paper.'
	},
	'Gemma 4 31B gate': {
		artifactAnchor: 'the evidence the reader KEPT',
		plain:
			'INDRA’s own unfitted combination rule applied to the evidence the reading model KEPT — ' +
			'not the MCMC-refitted “Belief Orig” of the 2023 paper.'
	},
	'Gemma 4 E2B gate': {
		artifactAnchor: 'the evidence the reader KEPT',
		plain:
			'INDRA’s own unfitted combination rule applied to the evidence the reading model KEPT — ' +
			'not the MCMC-refitted “Belief Orig” of the 2023 paper.'
	}
};

/** `caveats[]` in shipped order, pinned to each sentence by a verbatim fragment. */
const BELIEF_LADDER_CAVEAT_TWINS: readonly AnchoredProse[] = [
	{
		artifactAnchor: 'SAME 1689 all-source curated statements',
		plain:
			'Every value here is average precision over all 1,689 curated statements at once — the ' +
			'SAME statements, drawn from every source, for every row — with the SAME labels released ' +
			'with the 2023 paper, re-derived with scikit-learn’s tie-aware average_precision_score ' +
			'and then cross-checked against the value each original run recorded for itself.'
	},
	{
		artifactAnchor: 'It is the baseline because',
		plain:
			'The baseline is the unfitted SimpleScorer combination rule — one minus the product, ' +
			'over every source, of that source’s systematic error rate plus its random error rate ' +
			'raised to the number of evidence entries it supplied — at 0.9031 average precision over ' +
			'all statements at once. It is the baseline because it is what BOTH the ' +
			'engineered-feature random forest and the reading step modify — NOT because it is the ' +
			'best model published in 2023. And that random forest is not the published one: ' +
			'INDRA’s full CountsScorer feature set is a superset of the statement-type / #PMIDs / ' +
			'promoter features engineered for the 2023 paper.'
	},
	{
		artifactAnchor: 'is worth',
		plain:
			'From that baseline, INDRA’s full CountsScorer feature set — a superset of the ' +
			'statement-type / #PMIDs / promoter features engineered in the 2023 paper — is worth ' +
			'+0.0390, and the reading step is worth +0.0479 (Gemma 4 26B). But that +0.0479 is ' +
			'measured from the unfitted combination rule, the WEAKEST member of the 2023 ' +
			'lineage. Against the strongest ' +
			'combination-rule variant here (BayesianScorer, source+subtype refit) the same reading ' +
			'step is worth +0.0331; against the strongest model of all (the random forest with ' +
			'full features, 0.9422 re-implemented / 0.9412 literal) it is worth +0.0088 to +0.0098. ' +
			'Quote that range. NEVER quote the +0.0479 on its own.'
	},
	{
		artifactAnchor: 'flat against plain noisy-OR',
		plain:
			'Passing belief up the hierarchy (−0.0001) and a random forest trained on source counts ' +
			'alone (−0.0004) are level with the plain combination rule on these statements: the ' +
			'supervised gain comes from the features that are not counts, not from learning a better ' +
			'function of the source counts.'
	},
	{
		artifactAnchor: 'CONSISTENCY CHECK ACROSS DIFFERENT CORPORA',
		plain:
			'The re-implemented random forest lands at 0.9422 and the literal released model at ' +
			'0.9412, 0.0010 apart. That is a CONSISTENCY CHECK ACROSS DIFFERENT CORPORA — two ' +
			'codebases, two snapshots of the evidence, agreeing in one number — and it is NOT ' +
			'evidence that the re-implementation is faithful to the released code. The fidelity ' +
			'evidence is the per-statement Pearson r = 0.9994 between the released model and the ' +
			'semantic port of it, already reported on this page.'
	},
	{
		artifactAnchor: 'only some are models the paper published',
		plain:
			'The seven non-reading rows are re-implementations on indra 1.24.0 over current INDRA ' +
			'evidence, and only some of them are models the 2023 paper published: it has no ' +
			'Bayesian, subtype-resolved, hierarchy-propagated or HybridScorer model at all. The ' +
			'row that re-runs the released code is that 2023 code on the released corpus. The ' +
			'families compare cleanly to each other, and that literal row compares only loosely to ' +
			'them.'
	},
	{
		artifactAnchor: 'SAME fitted model reported twice',
		plain:
			'“CountsScorer RF, full features” and “HybridScorer, full features” are the SAME fitted ' +
			'model reported twice (agreeing to 0.000e+00), not two independent results. They appear ' +
			'as two rows because INDRA’s own scorer classes name them separately; the 2023 paper ' +
			'names neither.'
	},
	{
		artifactAnchor: 'contribution is the filtering',
		plain:
			'The reading rows are the unfitted noisy-OR over per-source reliabilities applied to ' +
			'the evidence the reading model kept, so this list compares belief models under one ' +
			'shared way of combining evidence; the reading model’s contribution is the filtering.'
	}
];

/**
 * The verbatim fragment each restated sentence must still contain, in shipped
 * order for the caveats and by frozen row label for the notes.
 *
 * EXPORTED because it is part of the CONTRACT, not an implementation detail: a
 * sentence this figure cannot restate is a sentence it may not print, and the
 * contract runner builds a synthetic artifact that has to satisfy that. Deriving
 * it from the twins means the runner cannot drift from the loader.
 */
export const BELIEF_LADDER_PROSE_ANCHORS = {
	caveats: BELIEF_LADDER_CAVEAT_TWINS.map((twin) => twin.artifactAnchor),
	notes: Object.fromEntries(
		Object.entries(BELIEF_LADDER_NOTE_TWINS).map(([label, twin]) => [label, twin.artifactAnchor])
	)
} as const;

/**
 * Arithmetic parity tolerance for every identity the figure's geometry depends
 * on. Matches the artifact's own `checks.recorded_value_agreement_tol` /
 * `checks.same_fitted_model_tol`; the only numeric constant in this file.
 */
export const BELIEF_LADDER_PARITY_TOL = 1e-12;

/** The artifact's three families: the paper's own lineage, its literal released model, the reader gates. */
export type BeliefLadderKind = 'paper-family' | 'paper-literal' | 'reader-gate';

export interface BeliefLadderEntrySpec {
	/** Exact `entries[].label` in belief_model_ladder.json — the join key. */
	label: string;
	/**
	 * On-screen name. DECOUPLED from `label`, which addresses shipped bytes
	 * (`provenance.scores[label]`, `provenance.recorded_values[label]`, and every
	 * guardrail referent) and must never change to suit a caption. Nothing on this
	 * page may render `label`; `beliefLadderDisplay()` is the only way to a name.
	 */
	display: string;
	kind: BeliefLadderKind;
}

/**
 * The twelve rungs in the artifact's FIXED order (the paper's family ascending,
 * then the literal released model, then the reader gates). `validateBeliefLadder`
 * requires the artifact's entry sequence to equal this (label, kind) sequence
 * exactly — a reordering or a dropped rung is a visible failure, never a silently
 * different figure.
 *
 * FIVE displays deliberately differ from their join keys, and the field split is
 * what makes that safe: the two are separate fields, so a rename on either side
 * cannot silently move the other.
 *
 *   · `Hierarchy propagation` IS the noisy-OR with belief propagated over the
 *     statement hierarchy (the artifact's own `note` says so), and that variant —
 *     not the direct one — is what INDRA actually serves. A bare "Hierarchy
 *     propagation" names the operation without naming the model it operates on.
 *   · The four reader rungs read "… reading" where their join key says "… gate".
 *     "Gate" was our word for the step where the model reads each piece of
 *     evidence and keeps or drops it; it means nothing to a reader outside this
 *     repo, so the name on the axis says what the step is. The keys still say
 *     "gate" and must: they address shipped bytes.
 */
export const BELIEF_LADDER_ENTRY_SPECS: readonly BeliefLadderEntrySpec[] = [
	{
		label: 'CountsScorer RF, source counts',
		display: 'CountsScorer RF, source counts',
		kind: 'paper-family'
	},
	{
		label: 'Hierarchy propagation',
		display: 'noisy-OR + hierarchy propagation',
		kind: 'paper-family'
	},
	{
		label: 'noisy-OR SimpleScorer (direct)',
		display: 'noisy-OR SimpleScorer (direct)',
		kind: 'paper-family'
	},
	{
		label: 'BayesianScorer, source refit',
		display: 'BayesianScorer, source refit',
		kind: 'paper-family'
	},
	{
		label: 'BayesianScorer, source+subtype refit',
		display: 'BayesianScorer, source+subtype refit',
		kind: 'paper-family'
	},
	{
		label: 'CountsScorer RF, full features',
		display: 'CountsScorer RF, full features',
		kind: 'paper-family'
	},
	{
		label: 'HybridScorer, full features',
		display: 'HybridScorer, full features',
		kind: 'paper-family'
	},
	{
		label: 'RF 2k-d13 + Type/#PMIDs/promoter',
		display: 'RF 2k-d13 + Type/#PMIDs/promoter',
		kind: 'paper-literal'
	},
	{ label: 'Gemma 4 26B gate', display: 'Gemma 4 26B reading', kind: 'reader-gate' },
	{ label: 'GLM-5 gate', display: 'GLM-5 reading', kind: 'reader-gate' },
	{ label: 'Gemma 4 31B gate', display: 'Gemma 4 31B reading', kind: 'reader-gate' },
	{ label: 'Gemma 4 E2B gate', display: 'Gemma 4 E2B reading', kind: 'reader-gate' }
] as const;

/**
 * SVG geometry the row-label budget is DERIVED from, exported so the budget below
 * is a computation rather than an eyeballed constant, and so the contract runner
 * can re-derive it. Mirrors the constants in `BeliefModelLadder.svelte`.
 */
export const BELIEF_LADDER_GEOMETRY = {
	width: 900,
	/** Row labels are right-anchored at LABEL_RIGHT − 2 = 228; the gutter is 0 → 228. */
	labelAnchorX: 228,
	/** The figure's own left gutter, kept clear so a long name never touches x = 0. */
	leftGutter: 12,
	/** Left edge of the plotted axis, and the furthest right the origin can sit. */
	plotLeft: 250,
	plotRight: 742,
	labelFontPx: 9,
	baselineTagFontPx: 8,
	/** Measured advance of the mono face at 9px, in user units per character. */
	monoUnitsPerChar: 5.4186,
	/** The same face at 8px: 5.4186 × 8/9. */
	tagUnitsPerChar: 4.8165,
	/** The baseline tag starts this far right of the origin rule. */
	baselineTagOffsetX: 5
} as const;

/**
 * ROW LABEL BUDGET (right-anchored SVG text loses its LEADING glyphs silently —
 * no layout error, no failing a11y check, because <desc> emits the full string
 * either way). 228 units ÷ 5.4186 u/char at 9px = 42.0 characters hard, and
 * (228 − 12) ÷ 5.4186 = 39.8 → 39 once the figure's own left gutter is respected.
 * The longest display this module ships is
 * "BayesianScorer, source+subtype refit" at 36 characters (195.1 units), leaving
 * 3 characters / 20.9 units of slack — measured and printed by
 * `viewer/scripts/test-paper-render-invariants.mjs`, not estimated here.
 * `validateBeliefLadder` THROWS above the budget, so a longer rung name gates the
 * figure to `unavailable` instead of quietly eating its first glyphs.
 *
 * Mirrored by `_LADDER_LABEL_MAX_CHARS` in tests/test_viewer_paper_literal_contract.py.
 */
export const BELIEF_LADDER_DISPLAY_BUDGET_CHARS = 39;

/**
 * BASELINE TAG BUDGET. The tag is LEFT-anchored at the origin rule, which moves
 * with the data, so this budget is checked against the WORST case the axis
 * permits: the origin at `plotRight`. (900 − 742 − 5) ÷ 4.8165 = 31.7 → 31
 * characters for the whole `display · 0.9031` string. Today's origin sits near
 * x = 272 (the two negative rungs are tiny), so the real headroom is far larger —
 * but the worst case is what a rerun could hand us, and `beliefLadderBaselineTag`
 * measures the tag against the ACTUAL origin it will be drawn at, so the figure
 * gates only when it would really clip.
 */
export const BELIEF_LADDER_BASELINE_TAG_WORST_CASE_CHARS = 31;

const DISPLAY_BY_LABEL: ReadonlyMap<string, string> = new Map(
	BELIEF_LADDER_ENTRY_SPECS.map((spec) => [spec.label, spec.display] as const)
);

/**
 * The ONLY route from a frozen join key to a name on the screen. Throws on an
 * unknown label rather than falling back to the key itself: a fallback is how a
 * join key reaches the screen looking like a display name, which is the exact
 * defect this indirection exists to prevent.
 */
export function beliefLadderDisplay(label: string): string {
	const display = DISPLAY_BY_LABEL.get(label);
	if (display === undefined) {
		fail('belief_ladder.display', `${label} is not a rung on this ladder, so it has no display name`);
	}
	return display;
}

/**
 * The origin tag, budget-checked against the origin it will actually be drawn at.
 * Returns null when it would run past the viewBox — the caller gates rather than
 * drawing a label whose trailing glyphs are outside the frame.
 */
export function beliefLadderBaselineTag(display: string, ap: string, zeroX: number): string | null {
	const g = BELIEF_LADDER_GEOMETRY;
	const tag = `${display} · ${ap}`;
	const room = g.width - (zeroX + g.baselineTagOffsetX);
	return tag.length * g.tagUnitsPerChar <= room ? tag : null;
}

/** The rung the axis origin sits on, and the rung every delta is measured from. */
export const BELIEF_LADDER_BASELINE_LABEL = 'noisy-OR SimpleScorer (direct)';

/** The one metric on the axis. Never renamed, never rescaled into hundredths. */
export const BELIEF_LADDER_METRIC = 'pooled_average_precision';

export const BELIEF_LADDER_ARTIFACT_KIND = 'belief_model_ladder';
export const BELIEF_LADDER_SCHEMA_VERSION = 1;

/** The method caveats travel with the artifact; the figure prints them verbatim. */
export const BELIEF_LADDER_CAVEAT_COUNT = 8;

/** INDRA's aggregation, exactly as the artifact states it. */
export const BELIEF_LADDER_NOISY_OR_FORMULA = 'belief = 1 - PROD_s (syst_s + rand_s^{n_s})';

/**
 * The WRONG form of the same aggregation. It was purged from two headers once and
 * must not reappear anywhere in the artifact — validation scans the whole payload
 * for this fragment.
 */
export const BELIEF_LADDER_WRONG_NOISY_OR_FRAGMENT = '1 - PROD (1-r';

/** The sibling artifact the referent derivation reads its fold SDs from. */
export const BELIEF_LADDER_VS_LLMS_BASENAME = 'paper_literal_vs_llms.json';

export interface BeliefLadderPanel {
	n: number;
	nErrors: number;
	nCorrect: number;
	errorBaseRate: number;
	/**
	 * The paper's own released label FIELD NAME — "error" means IT says incorrect.
	 * Not a display name and not an arm join key: it is printed inside <code> as
	 * the field it is. Named `labelField` so no sweep can mistake it for either.
	 */
	labelField: string;
	labelConvention: string;
	/** `labelConvention` with its plain restatement — `shipped` is byte-identical. */
	labelConventionProse: ShippedProse;
	/** Of the errors: the adjudication-safe negatives... */
	adjudicationSafeNegatives: number;
	/** ...and the ones flagged `label_is_adjudication_safe: false`. */
	flaggedNotAdjudicationSafe: number;
	ordering: string;
	/** `ordering` with its plain restatement — `shipped` is byte-identical. */
	orderingProse: ShippedProse;
}

export interface BeliefLadderBaseline {
	/** Frozen join key — never rendered. */
	label: string;
	/** On-screen name, resolved through `beliefLadderDisplay`. */
	display: string;
	averagePrecision: number;
	/** `formula` with its plain restatement — `shipped` is byte-identical. */
	formulaProse: ShippedProse;
	/** Why THIS is the baseline — printed as the axis rule's caption, verbatim. */
	why: string;
	/** `why` with its plain restatement — `whyProse.shipped === why`. */
	whyProse: ShippedProse;
	formula: string;
}

export interface BeliefLadderEntry {
	/** Frozen join key into `scores` / `recorded_values` — never rendered. */
	label: string;
	/** On-screen name, budget-checked against the axis gutter. */
	display: string;
	kind: BeliefLadderKind;
	/** Hue kind, resolved through the SAME `paperArmColorVar` the other /paper panels use. */
	paperKind: PaperArmKind;
	averagePrecision: number;
	/** What the run that produced the scores recorded, carried beside ours. */
	recordedAveragePrecision: number;
	disagreementVsRecorded: number;
	agreesWithRecorded: boolean;
	distinctScores: number;
	scoresPath: string;
	scoresKey: string | null;
	recordedIn: string;
	recordedKey: string;
	note: string;
	/** `note` with its plain restatement — `noteProse.shipped === note`. */
	noteProse: ShippedProse;
	/** The bar length: averagePrecision − baseline.averagePrecision. */
	deltaVsNoisyOrBaseline: number;
	isBaseline: boolean;
}

export interface BeliefLadderNamedDelta {
	/** Frozen join key — never rendered. */
	label: string;
	/** On-screen name, resolved through `beliefLadderDisplay`. */
	display: string;
	delta: number;
}

export interface BeliefLadderModifier {
	/** Frozen join key — never rendered. */
	label: string;
	/** On-screen name, resolved through `beliefLadderDisplay`. */
	display: string;
	averagePrecision: number;
	deltaVsNoisyOrBaseline: number;
}

export interface BeliefLadderGate extends BeliefLadderModifier {
	/** The gate against the best noisy-OR variant on this ladder (ours, not a published paper arm) — the middle rung. */
	deltaVsBestNoisyOrVariant: BeliefLadderNamedDelta;
	/** The gate against each of the paper's best MODELS, every referent named. */
	deltaVsBestPaperModel: BeliefLadderNamedDelta[];
	/** (min, max) of that map — the range that must ride with the headline delta. */
	deltaVsBestPaperModelRange: [number, number];
}

export interface BeliefLadderFidelityEvidence {
	statistic: string;
	/** `statistic` with its plain restatement — `shipped` is byte-identical. */
	statisticProse: ShippedProse;
	value: number;
	source: string;
}

export interface BeliefLadderProximity {
	reimplementedRfFullFeatures: number;
	paperLiteralRfPromoter: number;
	absoluteGap: number;
	/** The artifact's own words: a consistency check across corpora, NOT fidelity. */
	status: string;
	/** `status` with its plain restatement — `statusProse.shipped === status`. */
	statusProse: ShippedProse;
	/** Where the fidelity evidence actually lives — a different statistic entirely. */
	fidelityEvidence: BeliefLadderFidelityEvidence;
}

export interface BeliefLadderGuardrails {
	baselineLabel: string;
	baselineAveragePrecision: number;
	engineeredFeatures: BeliefLadderModifier;
	readingGate: BeliefLadderGate;
	/** The rungs that land where the baseline lands, with their signed deltas. */
	flatAgainstBaseline: BeliefLadderNamedDelta[];
	reimplementationProximity: BeliefLadderProximity;
}

export interface BeliefLadderChecks {
	nEntries: number;
	/** The two rows that are ONE fitted model reported twice in the paper's lineage. */
	sameFittedModelPair: [string, string];
	sameFittedModelAbsoluteGap: number;
	sameFittedModelTol: number;
	recordedValueAgreementTol: number;
	note: string;
	/** `note` with its plain restatement — `noteProse.shipped === note`. */
	noteProse: ShippedProse;
}

export interface BeliefLadderRecordedValue {
	path: string;
	key: string;
}

export interface BeliefLadder {
	metric: string;
	metricSource: string;
	/** `metricSource` with its plain restatement — `shipped` is byte-identical. */
	metricSourceProse: ShippedProse;
	noisyOrFormula: string;
	/** `noisyOrFormula` with its plain restatement — `shipped` is byte-identical. */
	noisyOrFormulaProse: ShippedProse;
	panel: BeliefLadderPanel;
	baseline: BeliefLadderBaseline;
	entries: BeliefLadderEntry[];
	guardrails: BeliefLadderGuardrails;
	/** The method caveats, verbatim from the artifact. Nothing may be dropped. */
	caveats: string[];
	checks: BeliefLadderChecks;
	gold: string;
	scores: Record<string, string>;
	recordedValues: Record<string, BeliefLadderRecordedValue>;
	join: string;
	/** `join` with its plain restatement — `joinProse.shipped === join`. */
	joinProse: ShippedProse;
	generatedBy: string;
	/**
	 * The plain half of every string this figure took off the artifact, in one
	 * place. Each of the twelve row notes and eight caveats is the SAME object the
	 * row carries, never a copy.
	 */
	prose: BeliefLadderProse;
}

/** Every shipped sentence this figure carries, each with its plain restatement. */
export interface BeliefLadderProse {
	metricSource: ShippedProse;
	/** The combining rule itself, in words rather than in product notation. */
	noisyOrFormula: ShippedProse;
	labelConvention: ShippedProse;
	panelOrdering: ShippedProse;
	/** Why the unfitted combination rule is the zero line. */
	baselineWhy: ShippedProse;
	/** That the two random forests agreeing is NOT evidence of fidelity. */
	proximityStatus: ShippedProse;
	/** Where the fidelity evidence actually lives — a different statistic. */
	fidelityStatistic: ShippedProse;
	checksNote: ShippedProse;
	join: ShippedProse;
	/** Index-aligned with `entries`, pinned to each row's own sentence. */
	entryNotes: ShippedProse[];
	/** Index-aligned with `caveats`, pinned to it by a verbatim fragment. */
	caveats: ShippedProse[];
}

/** One "the gate against X" statement, with X named and X's own value carried. */
export interface BeliefLadderReferent {
	/** X's on-screen name. Never a join key: `armLabel` was renamed to make that unmissable. */
	armDisplay: string;
	armAp: number;
	delta: number;
	/** True for the referent DERIVED from the sibling head-to-head artifact. */
	derived: boolean;
}

/**
 * The paper's own fold-to-fold dispersion, for the arms that carry one. It belongs
 * to THEIR trapezoidal estimator and is not an error bar on average precision —
 * the figure must say so wherever it prints this range.
 */
export interface BeliefLadderFoldSd {
	min: number;
	max: number;
	nArms: number;
}

export interface BeliefLadderReferents {
	/** The gate's on-screen name. */
	gateDisplay: string;
	gateAp: number;
	/** Sorted by delta ASCENDING, so the conservative number is structurally first. */
	referents: BeliefLadderReferent[];
	foldSd: BeliefLadderFoldSd;
}

export interface BeliefLadderOk {
	status: 'ok';
	reason: null;
	artifact_path: string;
	artifact_sha256: string;
	ladder: BeliefLadder;
	/**
	 * Derived by `beliefLadderReferents` in the loader from the ladder plus the
	 * sibling head-to-head artifact. On the ok branch it is always present: its
	 * derivation is inside the loader's try/catch, so a failure gates the panel.
	 */
	referents: BeliefLadderReferents;
}

export interface BeliefLadderUnavailable {
	status: 'unavailable';
	reason: string;
	artifact_path: string;
	artifact_sha256: string | null;
	ladder: null;
	referents: null;
}

export type BeliefLadderLoad = BeliefLadderOk | BeliefLadderUnavailable;

type UnknownRecord = Record<string, unknown>;



/** Any finite number (deltas are signed). */
function finite(value: unknown, context: string): number {
	if (typeof value !== 'number' || !Number.isFinite(value)) {
		fail(context, 'expected a finite number');
	}
	return value;
}



function boolean(value: unknown, context: string): boolean {
	if (typeof value !== 'boolean') fail(context, 'expected a boolean');
	return value;
}


function close(got: number, want: number, context: string, message: string): void {
	if (!(Math.abs(got - want) <= BELIEF_LADDER_PARITY_TOL)) fail(context, message);
}

function parsePanel(value: unknown): BeliefLadderPanel {
	const context = 'belief_model_ladder.panel';
	const obj = record(value, context);
	const n = positiveInteger(obj.n, `${context}.n`);
	const nErrors = positiveInteger(obj.n_errors, `${context}.n_errors`);
	const nCorrect = positiveInteger(obj.n_correct, `${context}.n_correct`);
	if (nErrors + nCorrect !== n) fail(context, 'n_errors + n_correct must equal n');
	const errorBaseRate = unit(obj.error_base_rate, `${context}.error_base_rate`);
	close(errorBaseRate, nErrors / n, `${context}.error_base_rate`, 'must equal n_errors / n');

	// The label convention the panel discloses: the errors split into the
	// adjudication-safe negatives and the ones flagged as not adjudication-safe.
	const breakdown = record(obj.negative_breakdown, `${context}.negative_breakdown`);
	if (breakdown.n_errors !== nErrors) {
		fail(`${context}.negative_breakdown.n_errors`, 'must equal the panel error count');
	}
	const adjudicationSafeNegatives = positiveInteger(
		breakdown.adjudication_safe_negatives,
		`${context}.negative_breakdown.adjudication_safe_negatives`
	);
	const flaggedNotAdjudicationSafe = positiveInteger(
		breakdown.flagged_label_is_adjudication_safe_false,
		`${context}.negative_breakdown.flagged_label_is_adjudication_safe_false`
	);
	if (adjudicationSafeNegatives + flaggedNotAdjudicationSafe !== nErrors) {
		fail(`${context}.negative_breakdown`, 'the two negative buckets must sum to n_errors');
	}
	// Parsed exactly as before; the restatement travels beside each, and the flat
	// string stays byte-identical to its twin's `shipped`.
	const labelConventionProse: ShippedProse = {
		shipped: text(obj.label_convention, `${context}.label_convention`),
		plain: BELIEF_LADDER_PLAIN.labelConvention
	};
	const orderingProse: ShippedProse = {
		shipped: text(obj.ordering, `${context}.ordering`),
		plain: BELIEF_LADDER_PLAIN.panelOrdering
	};

	return {
		n,
		nErrors,
		nCorrect,
		errorBaseRate,
		labelField: text(obj.label, `${context}.label`),
		labelConvention: labelConventionProse.shipped,
		labelConventionProse,
		adjudicationSafeNegatives,
		flaggedNotAdjudicationSafe,
		ordering: orderingProse.shipped,
		orderingProse
	};
}

function parseEntry(
	value: unknown,
	index: number,
	baselineAp: number,
	recordedTol: number
): BeliefLadderEntry {
	const spec = BELIEF_LADDER_ENTRY_SPECS[index];
	const context = `belief_model_ladder.entries[${index}]`;
	const obj = record(value, context);
	if (obj.label !== spec.label) {
		fail(`${context}.label`, `expected the fixed presentation order — ${spec.label}`);
	}
	if (obj.kind !== spec.kind) fail(`${context}.kind`, `expected ${spec.kind}`);
	// Looked up by the FROZEN row label, then pinned to its own text: a row with no
	// authored restatement gates the figure, and four rows ship the SAME sentence,
	// so the key alone cannot bind the right restatement to the right row.
	const noteProse = keyedShippedProse(
		spec.label,
		text(obj.note, `${context}.note`),
		BELIEF_LADDER_NOTE_TWINS,
		`${context}.note`
	);

	const averagePrecision = unit(obj.average_precision, `${context}.average_precision`);
	const recordedAveragePrecision = unit(
		obj.recorded_average_precision,
		`${context}.recorded_average_precision`
	);
	const disagreementVsRecorded = finite(
		obj.disagreement_vs_recorded,
		`${context}.disagreement_vs_recorded`
	);
	close(
		disagreementVsRecorded,
		averagePrecision - recordedAveragePrecision,
		`${context}.disagreement_vs_recorded`,
		'must equal average_precision minus recorded_average_precision'
	);
	const agreesWithRecorded = boolean(obj.agrees_with_recorded, `${context}.agrees_with_recorded`);
	if (agreesWithRecorded !== Math.abs(disagreementVsRecorded) <= recordedTol) {
		fail(`${context}.agrees_with_recorded`, 'must be the comparison it reports, not a claim');
	}

	// THE identity the bar geometry rests on: the bar IS ap − baseline.
	const deltaVsNoisyOrBaseline = finite(
		obj.delta_vs_noisy_or_baseline,
		`${context}.delta_vs_noisy_or_baseline`
	);
	close(
		deltaVsNoisyOrBaseline,
		averagePrecision - baselineAp,
		`${context}.delta_vs_noisy_or_baseline`,
		'must equal average_precision minus the marked baseline'
	);

	const isBaseline = spec.label === BELIEF_LADDER_BASELINE_LABEL;
	if (isBaseline) {
		if (deltaVsNoisyOrBaseline !== 0) {
			fail(`${context}.delta_vs_noisy_or_baseline`, 'the baseline rung must be exactly zero');
		}
		if (averagePrecision !== baselineAp) {
			fail(`${context}.average_precision`, 'must be the marked baseline average precision');
		}
	}

	// The budget is checked HERE, on the string that will actually be drawn, so an
	// over-long rung name gates the whole figure instead of losing its first
	// glyphs to the viewBox where no test and no screen reader can see it.
	if (spec.display.length > BELIEF_LADDER_DISPLAY_BUDGET_CHARS) {
		fail(
			`${context}.display`,
			`"${spec.display}" is ${spec.display.length} chars; the axis gutter budget is ${BELIEF_LADDER_DISPLAY_BUDGET_CHARS}`
		);
	}

	return {
		label: spec.label,
		display: spec.display,
		kind: spec.kind,
		paperKind: beliefLadderPaperKind(spec.kind),
		averagePrecision,
		recordedAveragePrecision,
		disagreementVsRecorded,
		agreesWithRecorded,
		distinctScores: positiveInteger(obj.distinct_scores, `${context}.distinct_scores`),
		scoresPath: text(obj.scores_path, `${context}.scores_path`),
		scoresKey: obj.scores_key === null ? null : text(obj.scores_key, `${context}.scores_key`),
		recordedIn: text(obj.recorded_in, `${context}.recorded_in`),
		recordedKey: text(obj.recorded_key, `${context}.recorded_key`),
		note: noteProse.shipped,
		noteProse,
		deltaVsNoisyOrBaseline,
		isBaseline
	};
}

function namedDelta(
	label: string,
	value: unknown,
	context: string,
	against: number,
	byLabel: Map<string, BeliefLadderEntry>
): BeliefLadderNamedDelta {
	const entry = byLabel.get(label);
	if (!entry) fail(context, `names ${label}, which is not a rung on this ladder`);
	const delta = finite(value, context);
	close(delta, against - entry.averagePrecision, context, `must equal the gate minus ${label}`);
	return { label, display: entry.display, delta };
}

function parseModifier(
	value: unknown,
	context: string,
	byLabel: Map<string, BeliefLadderEntry>
): BeliefLadderModifier {
	const obj = record(value, context);
	const label = text(obj.label, `${context}.label`);
	const entry = byLabel.get(label);
	if (!entry) fail(`${context}.label`, `names ${label}, which is not a rung on this ladder`);
	const averagePrecision = unit(obj.average_precision, `${context}.average_precision`);
	const deltaVsNoisyOrBaseline = finite(
		obj.delta_vs_noisy_or_baseline,
		`${context}.delta_vs_noisy_or_baseline`
	);
	// A guardrail that disagrees with the rung it names would put a different
	// number in the caption than the figure draws.
	close(
		averagePrecision,
		entry.averagePrecision,
		`${context}.average_precision`,
		`must equal the ${label} rung`
	);
	close(
		deltaVsNoisyOrBaseline,
		entry.deltaVsNoisyOrBaseline,
		`${context}.delta_vs_noisy_or_baseline`,
		`must equal the ${label} rung's own delta`
	);
	return { label, display: entry.display, averagePrecision, deltaVsNoisyOrBaseline };
}

function parseGate(
	value: unknown,
	context: string,
	byLabel: Map<string, BeliefLadderEntry>
): BeliefLadderGate {
	const base = parseModifier(value, context, byLabel);
	const obj = record(value, context);

	const variantRaw = record(obj.delta_vs_best_noisy_or_variant, `${context}.delta_vs_best_noisy_or_variant`);
	const deltaVsBestNoisyOrVariant = namedDelta(
		text(variantRaw.label, `${context}.delta_vs_best_noisy_or_variant.label`),
		variantRaw.delta,
		`${context}.delta_vs_best_noisy_or_variant.delta`,
		base.averagePrecision,
		byLabel
	);

	const againstBestRaw = record(obj.delta_vs_best_paper_model, `${context}.delta_vs_best_paper_model`);
	const deltaVsBestPaperModel = Object.entries(againstBestRaw).map(([label, delta]) =>
		namedDelta(label, delta, `${context}.delta_vs_best_paper_model[${label}]`, base.averagePrecision, byLabel)
	);
	if (deltaVsBestPaperModel.length === 0) {
		fail(`${context}.delta_vs_best_paper_model`, 'expected at least one named referent');
	}

	const rangeRaw = obj.delta_vs_best_paper_model_range;
	if (!Array.isArray(rangeRaw) || rangeRaw.length !== 2) {
		fail(`${context}.delta_vs_best_paper_model_range`, 'expected a (low, high) pair');
	}
	const low = finite(rangeRaw[0], `${context}.delta_vs_best_paper_model_range[0]`);
	const high = finite(rangeRaw[1], `${context}.delta_vs_best_paper_model_range[1]`);
	const deltas = deltaVsBestPaperModel.map((item) => item.delta);
	close(
		low,
		Math.min(...deltas),
		`${context}.delta_vs_best_paper_model_range[0]`,
		'must be the smallest against-their-best delta'
	);
	close(
		high,
		Math.max(...deltas),
		`${context}.delta_vs_best_paper_model_range[1]`,
		'must be the largest against-their-best delta'
	);
	// The whole reason both numbers ship: the against-their-best range is strictly
	// tighter than the delta measured from the weakest family member. If that ever
	// stops being true, the caption's framing is wrong and the figure goes dark.
	if (!(high < base.deltaVsNoisyOrBaseline)) {
		fail(
			`${context}.delta_vs_best_paper_model_range`,
			'must sit strictly below the delta measured from the baseline'
		);
	}

	return { ...base, deltaVsBestNoisyOrVariant, deltaVsBestPaperModel, deltaVsBestPaperModelRange: [low, high] };
}

function parseProximity(value: unknown, context: string): BeliefLadderProximity {
	const obj = record(value, context);
	const reimplementedRfFullFeatures = unit(
		obj.reimplemented_rf_full_features,
		`${context}.reimplemented_rf_full_features`
	);
	const paperLiteralRfPromoter = unit(
		obj.paper_literal_rf_promoter,
		`${context}.paper_literal_rf_promoter`
	);
	const absoluteGap = finite(obj.absolute_gap, `${context}.absolute_gap`);
	close(
		absoluteGap,
		Math.abs(reimplementedRfFullFeatures - paperLiteralRfPromoter),
		`${context}.absolute_gap`,
		'must equal the distance between its own two operands'
	);
	const evidence = record(obj.fidelity_evidence, `${context}.fidelity_evidence`);
	return {
		reimplementedRfFullFeatures,
		paperLiteralRfPromoter,
		absoluteGap,
		status: text(obj.status, `${context}.status`),
		statusProse: {
			shipped: text(obj.status, `${context}.status`),
			plain: BELIEF_LADDER_PLAIN.proximityStatus
		},
		fidelityEvidence: {
			statistic: text(evidence.statistic, `${context}.fidelity_evidence.statistic`),
			statisticProse: {
				shipped: text(evidence.statistic, `${context}.fidelity_evidence.statistic`),
				plain: BELIEF_LADDER_PLAIN.fidelityStatistic
			},
			value: unit(evidence.value, `${context}.fidelity_evidence.value`),
			source: text(evidence.source, `${context}.fidelity_evidence.source`)
		}
	};
}

function parseGuardrails(
	value: unknown,
	baseline: BeliefLadderBaseline,
	byLabel: Map<string, BeliefLadderEntry>
): BeliefLadderGuardrails {
	const context = 'belief_model_ladder.delta_guardrails';
	const obj = record(value, context);
	if (obj.baseline_label !== baseline.label) {
		fail(`${context}.baseline_label`, 'must be the ladder baseline');
	}
	const baselineAveragePrecision = unit(
		obj.baseline_average_precision,
		`${context}.baseline_average_precision`
	);
	close(
		baselineAveragePrecision,
		baseline.averagePrecision,
		`${context}.baseline_average_precision`,
		'must be the ladder baseline average precision'
	);

	const flatRaw = record(obj.flat_against_baseline, `${context}.flat_against_baseline`);
	const flatAgainstBaseline = Object.entries(flatRaw).map(([label, delta]) => {
		const entry = byLabel.get(label);
		const where = `${context}.flat_against_baseline[${label}]`;
		if (!entry) fail(where, `names ${label}, which is not a rung on this ladder`);
		close(finite(delta, where), entry.deltaVsNoisyOrBaseline, where, "must equal that rung's own delta");
		return { label, display: entry.display, delta: entry.deltaVsNoisyOrBaseline };
	});
	if (flatAgainstBaseline.length === 0) {
		fail(`${context}.flat_against_baseline`, 'expected at least one flat rung');
	}

	return {
		baselineLabel: baseline.label,
		baselineAveragePrecision,
		engineeredFeatures: parseModifier(obj.engineered_features, `${context}.engineered_features`, byLabel),
		readingGate: parseGate(obj.reading_gate, `${context}.reading_gate`, byLabel),
		flatAgainstBaseline,
		reimplementationProximity: parseProximity(
			obj.reimplementation_proximity,
			`${context}.reimplementation_proximity`
		)
	};
}

function parseChecks(value: unknown, byLabel: Map<string, BeliefLadderEntry>): BeliefLadderChecks {
	const context = 'belief_model_ladder.checks';
	const obj = record(value, context);
	const checksNoteProse: ShippedProse = {
		shipped: text(obj.note, `${context}.note`),
		plain: BELIEF_LADDER_PLAIN.checksNote
	};
	for (const key of [
		'every_entry_covers_the_panel_exactly',
		'gold_matches_hash_agrees_with_prediction_provenance',
		'literal_arm_joins_on_matches_hash',
		'baseline_delta_is_exactly_zero'
	]) {
		if (boolean(obj[key], `${context}.${key}`) !== true) fail(`${context}.${key}`, 'must be true');
	}
	const nEntries = positiveInteger(obj.n_entries, `${context}.n_entries`);
	if (nEntries !== BELIEF_LADDER_ENTRY_SPECS.length) {
		fail(`${context}.n_entries`, 'must equal the fixed rung count');
	}
	if (
		positiveInteger(
			obj.n_entries_agreeing_with_recorded_value,
			`${context}.n_entries_agreeing_with_recorded_value`
		) !== nEntries
	) {
		fail(`${context}.n_entries_agreeing_with_recorded_value`, 'must cover every rung');
	}

	const pairRaw = obj.same_fitted_model_pair;
	if (!Array.isArray(pairRaw) || pairRaw.length !== 2) {
		fail(`${context}.same_fitted_model_pair`, 'expected exactly two labels');
	}
	const first = text(pairRaw[0], `${context}.same_fitted_model_pair[0]`);
	const second = text(pairRaw[1], `${context}.same_fitted_model_pair[1]`);
	const firstEntry = byLabel.get(first);
	const secondEntry = byLabel.get(second);
	if (!firstEntry || !secondEntry) {
		fail(`${context}.same_fitted_model_pair`, 'both labels must be rungs on this ladder');
	}
	const sameFittedModelTol = finite(obj.same_fitted_model_tol, `${context}.same_fitted_model_tol`);
	const sameFittedModelAbsoluteGap = finite(
		obj.same_fitted_model_absolute_gap,
		`${context}.same_fitted_model_absolute_gap`
	);
	// "One fitted model reported twice" is a claim the figure BRACKETS. If the two
	// rows ever disagree, the bracket would be a lie, so refuse to draw it.
	if (
		Math.abs(firstEntry.averagePrecision - secondEntry.averagePrecision) > sameFittedModelTol ||
		Math.abs(sameFittedModelAbsoluteGap) > sameFittedModelTol
	) {
		fail(`${context}.same_fitted_model_pair`, 'the two rows do not agree to their own tolerance');
	}

	return {
		nEntries,
		sameFittedModelPair: [first, second],
		sameFittedModelAbsoluteGap,
		sameFittedModelTol,
		recordedValueAgreementTol: finite(
			obj.recorded_value_agreement_tol,
			`${context}.recorded_value_agreement_tol`
		),
		note: checksNoteProse.shipped,
		noteProse: checksNoteProse
	};
}

/**
 * Pure, fail-closed parse of `belief_model_ladder.json`. THROWS on any drift —
 * shape, rung order, the delta identity every bar length depends on, a baseline
 * whose own delta is not zero, a guardrail that disagrees with the rung it names,
 * an against-their-best range that is not the min/max of its own map or that is
 * not strictly tighter than the from-baseline delta, a proximity gap that does not
 * match its operands, a same-fitted-model pair that does not actually agree, a
 * dropped caveat, or either noisy-OR formula string being wrong.
 */
export function validateBeliefLadder(raw: unknown): BeliefLadder {
	const obj = record(raw, 'belief_model_ladder');
	if (obj.artifact_kind !== BELIEF_LADDER_ARTIFACT_KIND) {
		fail('belief_model_ladder.artifact_kind', `expected ${BELIEF_LADDER_ARTIFACT_KIND}`);
	}
	if (obj.schema_version !== BELIEF_LADDER_SCHEMA_VERSION) {
		fail('belief_model_ladder.schema_version', `expected ${BELIEF_LADDER_SCHEMA_VERSION}`);
	}
	if (obj.metric !== BELIEF_LADDER_METRIC) {
		fail('belief_model_ladder.metric', `expected ${BELIEF_LADDER_METRIC}`);
	}
	if (obj.noisy_or_formula !== BELIEF_LADDER_NOISY_OR_FORMULA) {
		fail('belief_model_ladder.noisy_or_formula', `expected ${BELIEF_LADDER_NOISY_OR_FORMULA}`);
	}
	// The wrong aggregation must not appear ANYWHERE in the payload — not in a
	// note, not in a caveat, not in a formula field.
	if (JSON.stringify(raw).includes(BELIEF_LADDER_WRONG_NOISY_OR_FRAGMENT)) {
		fail('belief_model_ladder', 'carries the wrong noisy-OR form somewhere in its payload');
	}

	const panel = parsePanel(obj.panel);

	const baselineRaw = record(obj.baseline, 'belief_model_ladder.baseline');
	if (baselineRaw.label !== BELIEF_LADDER_BASELINE_LABEL) {
		fail('belief_model_ladder.baseline.label', `expected ${BELIEF_LADDER_BASELINE_LABEL}`);
	}
	if (baselineRaw.formula !== BELIEF_LADDER_NOISY_OR_FORMULA) {
		fail('belief_model_ladder.baseline.formula', `expected ${BELIEF_LADDER_NOISY_OR_FORMULA}`);
	}
	const baselineWhyProse: ShippedProse = {
		shipped: text(baselineRaw.why, 'belief_model_ladder.baseline.why'),
		plain: BELIEF_LADDER_PLAIN.baselineWhy
	};
	const baseline: BeliefLadderBaseline = {
		label: BELIEF_LADDER_BASELINE_LABEL,
		display: beliefLadderDisplay(BELIEF_LADDER_BASELINE_LABEL),
		averagePrecision: unit(baselineRaw.average_precision, 'belief_model_ladder.baseline.average_precision'),
		why: baselineWhyProse.shipped,
		whyProse: baselineWhyProse,
		formula: BELIEF_LADDER_NOISY_OR_FORMULA,
		formulaProse: {
			shipped: BELIEF_LADDER_NOISY_OR_FORMULA,
			plain: BELIEF_LADDER_PLAIN.noisyOrFormula
		}
	};

	const checksRaw = record(obj.checks, 'belief_model_ladder.checks');
	const recordedTol = finite(
		checksRaw.recorded_value_agreement_tol,
		'belief_model_ladder.checks.recorded_value_agreement_tol'
	);

	if (!Array.isArray(obj.entries) || obj.entries.length !== BELIEF_LADDER_ENTRY_SPECS.length) {
		fail('belief_model_ladder.entries', `expected ${BELIEF_LADDER_ENTRY_SPECS.length} rungs in fixed order`);
	}
	const entries = obj.entries.map((entry, index) =>
		parseEntry(entry, index, baseline.averagePrecision, recordedTol)
	);
	const byLabel = new Map(entries.map((entry) => [entry.label, entry] as const));

	const guardrails = parseGuardrails(obj.delta_guardrails, baseline, byLabel);
	const checks = parseChecks(obj.checks, byLabel);

	if (!Array.isArray(obj.caveats) || obj.caveats.length !== BELIEF_LADDER_CAVEAT_COUNT) {
		fail('belief_model_ladder.caveats', `expected ${BELIEF_LADDER_CAVEAT_COUNT} caveats`);
	}
	const caveats = obj.caveats.map((entry, index) =>
		text(entry, `belief_model_ladder.caveats[${index}]`)
	);

	const provenance = record(obj.provenance, 'belief_model_ladder.provenance');
	const scoresRaw = record(provenance.scores, 'belief_model_ladder.provenance.scores');
	const recordedRaw = record(provenance.recorded_values, 'belief_model_ladder.provenance.recorded_values');
	const scores: Record<string, string> = {};
	const recordedValues: Record<string, BeliefLadderRecordedValue> = {};
	for (const entry of entries) {
		const where = `belief_model_ladder.provenance.scores[${entry.label}]`;
		const path = text(scoresRaw[entry.label], where);
		if (path !== entry.scoresPath) fail(where, "must equal the rung's own scores_path");
		scores[entry.label] = path;

		const recordedWhere = `belief_model_ladder.provenance.recorded_values[${entry.label}]`;
		const recorded = record(recordedRaw[entry.label], recordedWhere);
		const recordedPath = text(recorded.path, `${recordedWhere}.path`);
		const recordedKey = text(recorded.key, `${recordedWhere}.key`);
		if (recordedPath !== entry.recordedIn) fail(`${recordedWhere}.path`, "must equal the rung's recorded_in");
		if (recordedKey !== entry.recordedKey) fail(`${recordedWhere}.key`, "must equal the rung's recorded_key");
		recordedValues[entry.label] = { path: recordedPath, key: recordedKey };
	}

	const metricSourceProse: ShippedProse = {
		shipped: text(obj.metric_source, 'belief_model_ladder.metric_source'),
		plain: BELIEF_LADDER_PLAIN.metricSource
	};
	const joinProse: ShippedProse = {
		shipped: text(provenance.join, 'belief_model_ladder.provenance.join'),
		plain: BELIEF_LADDER_PLAIN.join
	};
	// Positional twins: a reissued artifact that reorders or rewrites a caveat
	// gates the figure rather than printing restatement N under caveat N+1.
	const caveatProse = pairShippedProse(
		caveats,
		BELIEF_LADDER_CAVEAT_TWINS,
		'belief_model_ladder.caveats'
	);

	return {
		metric: BELIEF_LADDER_METRIC,
		metricSource: metricSourceProse.shipped,
		metricSourceProse,
		noisyOrFormula: BELIEF_LADDER_NOISY_OR_FORMULA,
		noisyOrFormulaProse: baseline.formulaProse,
		panel,
		baseline,
		entries,
		guardrails,
		caveats,
		checks,
		gold: text(provenance.gold, 'belief_model_ladder.provenance.gold'),
		scores,
		recordedValues,
		join: joinProse.shipped,
		joinProse,
		generatedBy: text(provenance.generated_by, 'belief_model_ladder.provenance.generated_by'),
		prose: {
			metricSource: metricSourceProse,
			noisyOrFormula: baseline.formulaProse,
			labelConvention: panel.labelConventionProse,
			panelOrdering: panel.orderingProse,
			baselineWhy: baseline.whyProse,
			proximityStatus: guardrails.reimplementationProximity.statusProse,
			fidelityStatistic: guardrails.reimplementationProximity.fidelityEvidence.statisticProse,
			checksNote: checks.noteProse,
			join: joinProse,
			// The SAME objects the rows carry, in the artifact's own order.
			entryNotes: entries.map((entry) => entry.noteProse),
			caveats: caveatProse
		}
	};
}

/**
 * Kind → the page-wide hue convention. The ladder's three families map onto the
 * two hues the rest of /paper already uses: everything from the paper's lineage
 * (re-implemented family AND the literal released model) is `paper`, the reader
 * gates are `llm`. The literal row is distinguished by STROKE treatment and by
 * its own label — never by a new colour token.
 */
export function beliefLadderPaperKind(kind: BeliefLadderKind): PaperArmKind {
	return kind === 'reader-gate' ? 'llm' : 'paper';
}

/** Shared hue resolution — the same token the other /paper panels use. */
export function beliefLadderColorVar(entry: BeliefLadderEntry): string {
	return paperArmColorVar(entry.paperKind);
}

/**
 * Display order: lowest average precision first, so the axis reads bottom-up as a
 * ladder. This is a LAYOUT decision made here; the artifact's own entry order is
 * the validated one and is asserted untouched. Ties (the two rows that are one
 * fitted model reported twice) fall back to that artifact order, which keeps the
 * layout deterministic between SSR and the client.
 */
export function beliefLadderDisplayOrder(ladder: BeliefLadder): BeliefLadderEntry[] {
	const rank = new Map(BELIEF_LADDER_ENTRY_SPECS.map((spec, index) => [spec.label, index] as const));
	return [...ladder.entries].sort(
		(a, b) =>
			a.averagePrecision - b.averagePrecision ||
			(rank.get(a.label) ?? 0) - (rank.get(b.label) ?? 0)
	);
}

function pointMetricNumber(pointMetrics: unknown, key: string, field: string): number {
	const table = record(pointMetrics, 'paper_literal_vs_llms.point_metrics');
	const arm = record(table[key], `paper_literal_vs_llms.point_metrics[${key}]`);
	return unit(arm[field], `paper_literal_vs_llms.point_metrics[${key}].${field}`);
}

/**
 * Fold the sibling head-to-head artifact into the ladder, purely.
 *
 * Two things come back, neither of them typed by hand:
 *
 *  1. The gate's referents. The artifact ships its delta against two of the
 *     paper's best models; the paper's STRONGEST literal arm is derived here as
 *     the max pooled average precision over the `paper`-kind arms of
 *     `PAPER_LITERAL_ARM_SPECS` — its label is never written down in this file.
 *     The derived delta is ASSERTED to fall inside the artifact's own shipped
 *     `delta_vs_best_paper_model_range`; if it does not, the artifact and the
 *     sibling disagree and this THROWS rather than shipping a fourth number that
 *     contradicts the shipped range. Referents come back sorted delta ascending,
 *     so the conservative one is structurally first.
 *
 *  2. The fold-SD spread. For every rung whose recorded value lives in the
 *     head-to-head artifact, the paper's own fold-to-fold population SD is read
 *     off `point_metrics`. It belongs to THEIR trapezoidal estimator — it is not
 *     an error bar on average precision, and the figure must say so.
 *
 * THROWS on any missing key: a rung that cites the sibling artifact but has no
 * entry there means the two files have drifted apart.
 */
export function beliefLadderReferents(
	ladder: BeliefLadder,
	pointMetrics: unknown
): BeliefLadderReferents {
	const gate = ladder.guardrails.readingGate;
	const byLabel = new Map(ladder.entries.map((entry) => [entry.label, entry] as const));

	const shipped: BeliefLadderReferent[] = gate.deltaVsBestPaperModel.map((item) => {
		const entry = byLabel.get(item.label);
		if (!entry) fail('belief_model_ladder.delta_guardrails.reading_gate', `unknown referent ${item.label}`);
		// entry.display, never item.label: that string addresses shipped bytes.
		return {
			armDisplay: entry.display,
			armAp: entry.averagePrecision,
			delta: item.delta,
			derived: false
		};
	});

	// The paper's strongest LITERAL arm, derived — never named by hand here.
	const literalSpecs = PAPER_LITERAL_ARM_SPECS.filter((spec) => spec.kind === 'paper');
	if (literalSpecs.length === 0) fail('PAPER_LITERAL_ARM_SPECS', 'expected at least one paper arm');
	const strongest = literalSpecs
		.map((spec) => ({
			// display for the reader, label for the lookup: `label` is a frozen join
			// key into shipped point_metrics and must never reach the screen.
			armDisplay: spec.display,
			armAp: pointMetricNumber(pointMetrics, spec.label, BELIEF_LADDER_METRIC)
		}))
		.reduce((best, arm) => (arm.armAp > best.armAp ? arm : best));
	const derived: BeliefLadderReferent = {
		armDisplay: strongest.armDisplay,
		armAp: strongest.armAp,
		delta: gate.averagePrecision - strongest.armAp,
		derived: true
	};
	const [low, high] = gate.deltaVsBestPaperModelRange;
	if (
		derived.delta < low - BELIEF_LADDER_PARITY_TOL ||
		derived.delta > high + BELIEF_LADDER_PARITY_TOL
	) {
		fail(
			'belief_model_ladder.delta_guardrails.reading_gate.delta_vs_best_paper_model_range',
			`the gate against ${derived.armDisplay} falls outside the shipped range`
		);
	}

	const referents = [...shipped, derived].sort((a, b) => a.delta - b.delta);

	// The paper's own fold-to-fold dispersion, for the rungs that carry one.
	const sds = ladder.entries
		.filter((entry) => ladder.recordedValues[entry.label].path.endsWith(BELIEF_LADDER_VS_LLMS_BASENAME))
		.map((entry) =>
			pointMetricNumber(pointMetrics, ladder.recordedValues[entry.label].key, 'fold_population_sd')
		);
	if (sds.length === 0) {
		fail('belief_model_ladder.provenance.recorded_values', 'no rung cites the head-to-head artifact');
	}

	return {
		gateDisplay: gate.display,
		gateAp: gate.averagePrecision,
		referents,
		foldSd: { min: Math.min(...sds), max: Math.max(...sds), nArms: sds.length }
	};
}
