import { boolean, fail, nonNegativeInteger, number, positiveInteger, record, text, unit } from './paper-validate.ts';
/**
 * Typed data contract for the FRAMING CORRECTION panel — /paper's beat 2.
 *
 * Source artifacts, both from the same run directory:
 *   · `data/results/indra_paper_literal_models_20260724/framing_correction.json`,
 *     emitted by `scripts/compute_framing_correction.py` — legs (a) declaration,
 *     (b) subtractive and (c) reachable values, plus the permutation floor (c) is
 *     quoted against;
 *   · `data/results/indra_paper_literal_models_20260724/non_reading_control.json`,
 *     emitted by `scripts/compute_non_reading_control.py` — leg (d), the no-LLM
 *     control. The panel draws that leg straight from P1's file rather than
 *     restating its numbers anywhere else.
 *
 * The panel makes one thing unmistakable: the reader arm is not a rival belief
 * model. It is INDRA's own unfitted noisy-OR — `belief = 1 - PROD_s (syst_s +
 * rand_s^{n_s})` — run over the evidence a reader kept, so its only power is to
 * REMOVE factors from that product. It can never promote a statement the formula
 * under-scored.
 *
 * This module is import-safe on the client: typed shape, fixed arm table, and
 * pure `validateFramingCorrection()` / `validateNonReadingControl()` that THROW
 * on any shape or arithmetic drift. The server loader
 * (`$lib/server/paper-framing-correction`) wraps them and gates the WHOLE payload
 * to `unavailable` if either artifact fails — the panel's argument is not
 * partially true, so it is not partially drawn.
 *
 * NOTHING here is hard-coded from the numbers: the panel size, the error count
 * and its breakdown, every per-arm count, the tolerance, the search budget, the
 * permutation floor, the priors, and every control row are read off the
 * artifacts. The only constants are the four arm keys (the fixed presentation
 * order, asserted against the artifact), the aggregation name the whole argument
 * rests on, and the two formula strings — the correct one, which must appear
 * verbatim, and the wrong one, which must not appear at all.
 */

import {
	keyedShippedProse,
	pairShippedProse,
	paperArmColorVar,
	type AnchoredProse,
	type PaperArmKind,
	type ShippedProse
} from './paper-literal.ts';

/**
 * THE PLAIN HALF OF EVERY TWIN THESE TWO ARTIFACTS EMIT.
 *
 * `framing_correction.json` and `non_reading_control.json` are where "the reader
 * arm", "the unfitted hard gate", "pooled average precision" and "the paper's own
 * released label" reach the screen — none of them visible to any scan of this
 * repo, because the strings only exist once the files are read. Nothing is
 * softened: the 35 statements our own grounding rules zero, the de-dup scope
 * difference, the search budget and the enumeration restriction all survive at
 * full strength.
 */
const FRAMING_PLAIN = {
	question:
		'Is the reading model a rival belief model, or INDRA’s own unfitted combination rule run ' +
		'on the evidence a reader kept?',
	finding:
		'INDRA’s own unfitted combination rule, on a filtered evidence set. It can only take ' +
		'belief away, never add it.',
	labelConvention: '“wrong” means the label released with the 2023 paper says incorrect',
	declarationClaim:
		'All four reading bundles declare the unfitted hard reading step published in 2023, with no ' +
		'fitted reader profile, over the reliabilities in data/comparison/aggregation.json.',
	declarationDispatch:
		'statement_belief(soft=None) takes the unfitted path into ' +
		'noise_model.compute_gated_belief, which multiplies each source’s systematic error rate ' +
		'plus its random error rate raised to its surviving evidence count, over the sources in ' +
		'sorted order, and drops a source with no surviving evidence out of the product entirely. ' +
		'The two component digests below are the sha256 of those source files as they stand, so ' +
		'this is anchored to bytes rather than to a sentence.',
	subtractiveClaim:
		'Dropping evidence removes factors from a product of numbers below 1, so a reader can only ' +
		'lower belief. Checked, not assumed.',
	subtractiveBaseline: 'the raw combination rule over all evidence, with repeats kept',
	reachableClaim:
		'Every non-zero reading score is a value the published combining rule emits for some ' +
		'sub-collection of that statement’s evidence, counting repeats.',
	reachableDefinition:
		'the set of values one minus a product can take, where each source contributes either 1 ' +
		'(dropped entirely) or its systematic error rate plus its random error rate raised to any ' +
		'surviving count from 1 up to that statement’s own count for that source, with the ' +
		'reliabilities read from data/comparison/aggregation.json',
	nullBaselineQuestion:
		'How often does a reading score land on a value the formula can emit purely by chance? ' +
		'Such values are not rare, so “100% of them do” is only worth stating against this floor.',
	nullBaselineMethod:
		'The model’s own non-zero scores are shuffled across the statements whose value set is ' +
		'small enough to enumerate in full, and the same membership test is re-run against the ' +
		'RECEIVING statement’s set.',
	noisyOrFloorNote:
		'The lowest score the formula itself reaches on these 1,689 statements. It never reaches ' +
		'0, so a reading model’s block of zeroes is a different object: an empty product, not a ' +
		'low score.',
	controlQuestion:
		'Do the three NON-reading subtractions — removing duplicates, skipping evidence that ' +
		'carries no sentence text, and deterministic grounding rejects — account for the reading ' +
		'models’ gain?',
	controlFinding:
		'No. Applied with no model verdicts at all, they land BELOW the combination rule over ' +
		'every evidence entry, with no reading step applied.',
	controlMetric: 'average precision over all statements at once',
	controlMetricSource: 'scikit-learn’s average_precision_score, tie-aware',
	noisyOrFormula:
		'one minus the product, over every source, of that source’s systematic error rate plus its ' +
		'random error rate raised to the number of evidence entries that source supplied'
} as const;

/** `framing_correction.caveats[]`, pinned to each sentence by a verbatim fragment. */
const FRAMING_CAVEAT_TWINS: readonly AnchoredProse[] = [
	{
		artifactAnchor: 'Two different vectors can land on the same value',
		plain:
			'The reachable-value check asks whether that published combining rule CAN emit each score for ' +
			'SOME set of surviving counts, not whether it emitted it for the set the reading model ' +
			'actually produced. Two different sets can land on the same value. The first leg is what ' +
			'pins the combining rule itself; this leg shows the scores are consistent with it, and ' +
			'the chance floor below prices how much that consistency is worth.'
	},
	{
		artifactAnchor: 'budget-exhausted',
		plain:
			'The search is exhaustive inside its window but capped at 5,000,000 nodes per statement. ' +
			'Nothing here reached that cap (the worst statement used 1,155,323 nodes), and any ' +
			'statement that did would be reported as having run out of budget rather than counted ' +
			'as confirming anything.'
	},
	{
		artifactAnchor: 'LARGEST reachable sets',
		plain:
			'The chance floor is measured on the 1,635 of 1,689 statements whose value set can be ' +
			'enumerated outright, so it is measured against a complete set. The remaining statements ' +
			'have the LARGEST value sets, so including them could only push the floor up.'
	},
	{
		artifactAnchor: 'leaving an empty product',
		plain:
			'A belief of 0.0 is the reading model rejecting every piece of evidence it read, leaving ' +
			'an empty product. It is not a low score the formula assigned: SimpleScorer’s own lowest ' +
			'score on these statements is 0.65, which is one sentence.'
	}
];

/** The four rows of the non-reading control, keyed by their FROZEN row key. */
const CONTROL_ROW_TWINS: Readonly<Record<string, AnchoredProse>> = {
	raw: {
		artifactAnchor: 'no subtraction of any kind',
		plain:
			'The baseline with no reading step: all evidence, repeats kept, no subtraction of any ' +
			'kind.'
	},
	dedup_only: {
		artifactAnchor: 'One vote per UNIQUE',
		plain:
			'One vote per UNIQUE (statement, evidence) pair — the set of pairs the reading models ' +
			'were built on.'
	},
	dedup_plus_no_text: {
		artifactAnchor: 'drops evidence with no sentence',
		plain:
			'Also drops evidence with no sentence to read; the reading step never sees these.'
	},
	full_control: {
		artifactAnchor: 'no model verdict anywhere',
		plain:
			'All three non-reading subtractions, every readable pair accepted, and no model verdict ' +
			'anywhere.'
	}
};

/** `non_reading_control.caveats[]`, in shipped order. */
const CONTROL_CAVEAT_TWINS: readonly AnchoredProse[] = [
	{
		artifactAnchor: 'EXECUTION-MAP pass',
		plain:
			'This is the EXECUTION-MAP pass. It removes duplicates down to unique (statement, ' +
			'evidence) pairs as recorded in ' +
			'data/benchmark/indra_paper_unique_pairs_20260717_execution_map.jsonl. Production ' +
			'statement_belief collapses roughly 40 further near-duplicates within a source after ' +
			'text normalisation, which needs the production de-duplication pass to reproduce and is ' +
			'therefore NOT re-derived here; the memo-reported figures for that scope are carried ' +
			'separately and marked as memo-reported.'
	},
	{
		artifactAnchor: 'No model verd',
		plain:
			'Every row uses the combining rule published in 2023 and the reliabilities the ' +
			'reading bundles declare (data/comparison/aggregation.json): one minus the product, over ' +
			'the sources in sorted order, of each source’s systematic error rate plus its random ' +
			'error rate raised to its evidence count. No model verdict enters any row here.'
	},
	{
		artifactAnchor: 'not a defect being corrected',
		plain:
			'The de-duplication row is not a defect being corrected — it is a difference in scope ' +
			'between counting every evidence entry and counting each distinct (statement, evidence) ' +
			'pair once. It is priced here so it cannot be mistaken for either.'
	},
	{
		artifactAnchor: 'property of our grounding rules',
		plain:
			'35 statements lose ALL their evidence to the deterministic rejects and score a belief ' +
			'of 0 in the full control. That is a property of the grounding rules applied here, not of ' +
			'any reading model.'
	},
	{
		artifactAnchor: 'over-credits tied score distributions',
		plain:
			'Average precision here is scikit-learn’s tie-aware average_precision_score throughout; ' +
			'the trapezoidal PR-AUC the 2023 paper used is not used here, because it over-credits ' +
			'score distributions with many ties.'
	}
];

/** Arithmetic parity tolerance for the shares and rates the panel prints. */
export const FRAMING_PARITY_TOL = 1e-9;

/**
 * INDRA's aggregation, unfitted. The entire argument rests on it: under a fitted
 * reader profile belief floors at sigmoid(prior) rather than at an empty product,
 * and the arm would no longer be the paper's formula on kept evidence.
 */
export const FRAMING_REQUIRED_AGGREGATION = 'indra_default_hard_gate';

/** The noisy-OR, exactly as both artifacts state it. Anything else is drift. */
export const FRAMING_NOISY_OR_FORMULA = 'belief = 1 - PROD_s (syst_s + rand_s^{n_s})';

/**
 * The WRONG form. `1 - PROD (1-r_s)^n` is not INDRA's aggregation; it was already
 * purged from two headers and must not come back through an artifact. Rejected by
 * substring, so a reworded variant of the same error is still caught.
 */
export const FRAMING_WRONG_NOISY_OR_FRAGMENT = '1 - PROD (1-r';

/**
 * SVG geometry of the no-LLM control strip, exported so the label budget below is
 * DERIVED from it rather than eyeballed, and so the contract runner can re-derive
 * it. Mirrors the layout constants in `FramingCorrection.svelte`.
 */
export const FRAMING_CONTROL_GEOMETRY = {
	width: 900,
	/** Left edge of the plotted axis. */
	stripLeft: 350,
	stripRight: 820,
	/** Row labels are right-anchored 10 units inside that; the gutter is 0 → 340. */
	labelAnchorX: 340,
	labelFontPx: 8.5,
	/**
	 * Measured advance of the mono face at 8.5px. The page-wide measurement is
	 * 5.4186 user units per character at 9px; at 8.5px it scales linearly to
	 * 5.4186 × 8.5/9 = 5.1176.
	 */
	monoUnitsPerChar: 5.1176
} as const;

/**
 * CONTROL STRIP LABEL BUDGET. 340 units ÷ 5.1176 u/char at 8.5px = 66.4 → 66
 * characters. Right-anchored SVG text that overruns runs past x = 0 and is CLIPPED
 * by the viewBox — silently, with no layout error and no failing a11y check,
 * because the <desc> emits the full string either way. Two of the artifact's own
 * row labels were doing exactly that at the previous 300-unit gutter.
 *
 * The longest label the shipped artifact produces is
 * "de-dup + no-sentence + deterministic rejects, no LLM (control)" at 62
 * characters (317.3 units), leaving FOUR characters / 22.7 units of slack —
 * measured and printed by `viewer/scripts/test-paper-render-invariants.mjs`.
 * `validateNonReadingControl` THROWS above the budget, so a re-worded control row
 * gates the strip to `unavailable` instead of losing its leading glyphs.
 */
export const FRAMING_CONTROL_LABEL_BUDGET_CHARS = 66;

export interface FramingArmSpec {
	/** Exact key in `subtractive.arms` / `reachable_values.arms` — the join key. */
	key: string;
	/**
	 * Hue kind, resolved through the SAME `paperArmColorVar` the head-to-head,
	 * score-distribution and review-queue panels use, so an arm keeps one hue
	 * across the whole page.
	 */
	paperKind: PaperArmKind;
}

/**
 * The four reader arms in the artifact's FIXED presentation order. The labels are
 * NOT hard-coded here — they travel with the artifact — only the join keys and
 * the order do.
 */
export const FRAMING_ARM_SPECS: readonly FramingArmSpec[] = [
	{ key: 'gemma_4_26b', paperKind: 'llm' },
	{ key: 'glm_5', paperKind: 'llm' },
	{ key: 'gemma_4_31b', paperKind: 'llm' },
	{ key: 'gemma_4_e2b', paperKind: 'llm' }
] as const;

/** One reader bundle's declaration, as its own manifest states it. */
export interface FramingDeclarationArm {
	/** Frozen join key into the artifact's arm map — never rendered. */
	key: string;
	/** On-screen name, carried by the artifact. */
	display: string;
	manifestPath: string;
	implementation: string;
	aggregation: string;
	/** Must be null: a fitted profile would break the empty-product identity. */
	readerProfile: null;
	dedup: boolean;
	aggregationConfigSha256: string;
	noiseModelSha256: string;
	statementBeliefSha256: string;
}

/** A named source file and the sha256 the artifact checked it against. */
export interface FramingSourceDigest {
	path: string;
	sha256: string;
}

/** Sources that share one prior — the grouping the panel's one line reports. */
export interface FramingPriorGroup {
	rand: number;
	syst: number;
	sources: string[];
	nStatements: number;
}

export interface FramingDeclaration {
	claim: string;
	/** `claim` with its plain restatement — `claimProse.shipped === claim`. */
	claimProse: ShippedProse;
	dispatch: string;
	/** `dispatch` with its plain restatement — `shipped` is byte-identical. */
	dispatchProse: ShippedProse;
	requiredAggregation: string;
	aggregationConfig: FramingSourceDigest;
	noiseModelSource: FramingSourceDigest;
	statementBeliefSource: FramingSourceDigest;
	arms: FramingDeclarationArm[];
	/** Descending by member count: the first group is the readers' shared prior. */
	priorGroups: FramingPriorGroup[];
	nPanelSources: number;
}

/** One arm's subtractive check against the ungated noisy-OR on the same statement. */
export interface FramingSubtractiveArm {
	/** Frozen join key — never rendered. */
	key: string;
	/** On-screen name, carried by the artifact. */
	display: string;
	nStatements: number;
	/** The whole headline. Must be 0, or the panel refuses to draw. */
	nExceedingNoisyOr: number;
	nAtExactlyZero: number;
	nNonzero: number;
	maxBeliefAboveNoisyOr: number;
	scoresPath: string;
}

export interface FramingSubtractive {
	claim: string;
	/** `claim` with its plain restatement — `claimProse.shipped === claim`. */
	claimProse: ShippedProse;
	baseline: string;
	/** `baseline` with its plain restatement — `shipped` is byte-identical. */
	baselineProse: ShippedProse;
	baselineScores: string;
	arms: FramingSubtractiveArm[];
	nComparisons: number;
	nExceedingNoisyOr: number;
	maxBeliefAboveNoisyOr: number;
	crossCheckArtifact: string;
	crossCheckSha256: string;
}

/** One arm against the values the paper's formula can emit for each statement. */
export interface FramingReachableArm {
	/** Frozen join key — never rendered. */
	key: string;
	/** On-screen name, carried by the artifact. */
	display: string;
	nAtExactlyZero: number;
	nNonzero: number;
	/** Confirmed within the artifact's tolerance. */
	nConfirmedReachable: number;
	/** The tighter tier: bit-exact float equality in the canonical order. */
	nBitExact: number;
	/** Statements the search could not settle inside its budget. Named, never folded in. */
	nBudgetExhausted: number;
	/** Must be 0, or the panel refuses to draw. */
	nCounterexamples: number;
	shareConfirmed: number;
	shareBitExact: number;
	/** Permutation chance floor for this arm, over the enumerable statements. */
	permutedRateMean: number;
	permutedRateMin: number;
	permutedRateMax: number;
	nEnumerableNonzero: number;
}

export interface FramingNullBaseline {
	question: string;
	/** `question` with its plain restatement — `shipped` is byte-identical. */
	questionProse: ShippedProse;
	method: string;
	/** `method` with its plain restatement — `methodProse.shipped === method`. */
	methodProse: ShippedProse;
	nPermutations: number;
	seed: number;
	nStatementsEnumerable: number;
	nStatementsOnPanel: number;
	pooledMean: number;
	pooledMin: number;
	pooledMax: number;
}

export interface FramingReachable {
	claim: string;
	/** `claim` with its plain restatement — `claimProse.shipped === claim`. */
	claimProse: ShippedProse;
	definition: string;
	/** `definition` with its plain restatement — `shipped` is byte-identical. */
	definitionProse: ShippedProse;
	tolerance: number;
	nodeBudgetPerStatement: number;
	maxNodesUsed: number;
	sourceCountsPath: string;
	arms: FramingReachableArm[];
	nullBaseline: FramingNullBaseline;
	/** The formula's own lowest score on this panel; it never reaches 0. */
	noisyOrFloor: number;
	noisyOrFloorNote: string;
	/** `noisyOrFloorNote` with its plain restatement — `shipped` is byte-identical. */
	noisyOrFloorNoteProse: ShippedProse;
}

/** The label convention, disclosed once on the page from this block. */
export interface FramingNegativeBreakdown {
	nErrors: number;
	adjudicationSafeNegatives: number;
	flaggedNotAdjudicationSafe: number;
}

export interface FramingPanel {
	n: number;
	nErrors: number;
	nCorrect: number;
	errorBaseRate: number;
	/**
	 * The paper's own released label FIELD NAME, printed inside <code> as the field
	 * it is. Not a display name and not an arm join key — named `labelField` so a
	 * sweep for rendered join keys cannot confuse it with either.
	 */
	labelField: string;
	labelConvention: string;
	/** `labelConvention` with its plain restatement — `shipped` is byte-identical. */
	labelConventionProse: ShippedProse;
	negativeBreakdown: FramingNegativeBreakdown;
}

export interface FramingCorrection {
	question: string;
	finding: string;
	noisyOrFormula: string;
	/** `noisyOrFormula` with its plain restatement — `shipped` is byte-identical. */
	noisyOrFormulaProse: ShippedProse;
	aggregation: string;
	panel: FramingPanel;
	declaration: FramingDeclaration;
	subtractive: FramingSubtractive;
	reachable: FramingReachable;
	caveats: string[];
	generatedBy: string;
	provenance: Record<string, string>;
	/**
	 * The plain half of every string this figure took off the artifact. Each flat
	 * field above is byte-identical to its twin's `shipped`; the twin is the one to
	 * render, and the shipped half belongs behind the verification boundary. The
	 * nested ones are the SAME objects the blocks carry, never copies.
	 */
	prose: FramingCorrectionProse;
}

/** Every shipped sentence leg (a)–(c) carries, each with its restatement. */
export interface FramingCorrectionProse {
	question: ShippedProse;
	finding: ShippedProse;
	/** The combining rule itself, in words rather than in product notation. */
	noisyOrFormula: ShippedProse;
	labelConvention: ShippedProse;
	/** What the four reading bundles declare they ran. */
	declarationClaim: ShippedProse;
	/** Which code path that declaration dispatches into. */
	declarationDispatch: ShippedProse;
	/** Why a reader can only lower belief. */
	subtractiveClaim: ShippedProse;
	/** What the ungated baseline is. */
	subtractiveBaseline: ShippedProse;
	/** That every non-zero reading score is a value the formula can emit. */
	reachableClaim: ShippedProse;
	/** The set of values that formula can take. */
	reachableDefinition: ShippedProse;
	/** What the chance floor asks, and how it is measured. */
	nullBaselineQuestion: ShippedProse;
	nullBaselineMethod: ShippedProse;
	/** Why a belief of 0 is a different object from a low score. */
	noisyOrFloorNote: ShippedProse;
	/** Index-aligned with `caveats`, pinned to it by a verbatim fragment. */
	caveats: ShippedProse[];
}

// ---------------------------------------------------------------------------
// leg (d): the no-LLM control, read straight from P1's artifact
// ---------------------------------------------------------------------------

export interface NonReadingControlRow {
	/** Frozen join key (`baseline_row` / `control_row` address it) — never rendered. */
	key: string;
	/** On-screen name, carried by the artifact and budget-checked against the strip gutter. */
	display: string;
	weight: string;
	droppedRoutes: string[];
	nEvidenceScored: number;
	averagePrecision: number;
	deltaVsRawNoisyOr: number;
	note: string;
	/** `note` with its plain restatement — `noteProse.shipped === note`. */
	noteProse: ShippedProse;
}

export interface NonReadingControlContrast {
	/** Frozen join key — never rendered. */
	key: string;
	/** On-screen name, carried by the artifact and budget-checked against the strip gutter. */
	display: string;
	averagePrecision: number;
	deltaVsRawNoisyOr: number;
	deltaVsFullControl: number;
}

/** Every shipped sentence leg (d) carries, each with its restatement. */
export interface NonReadingControlProse {
	question: ShippedProse;
	finding: ShippedProse;
	metric: ShippedProse;
	metricSource: ShippedProse;
	/** The combining rule itself, in words rather than in product notation. */
	noisyOrFormula: ShippedProse;
	/** Index-aligned with `rows`, pinned to each row's own sentence. */
	rowNotes: ShippedProse[];
}

export interface NonReadingControl {
	question: string;
	/** `question` with its plain restatement — `shipped` is byte-identical. */
	questionProse: ShippedProse;
	finding: string;
	/** `finding` with its plain restatement — `shipped` is byte-identical. */
	findingProse: ShippedProse;
	metric: string;
	/** `metric` with its plain restatement — `shipped` is byte-identical. */
	metricProse: ShippedProse;
	metricSource: string;
	/** `metricSource` with its plain restatement — `shipped` is byte-identical. */
	metricSourceProse: ShippedProse;
	/** The plain half of every string above, plus the four row notes. */
	prose: NonReadingControlProse;
	/** `noisyOrFormula` with its plain restatement — `shipped` is byte-identical. */
	noisyOrFormulaProse: ShippedProse;
	noisyOrFormula: string;
	rows: NonReadingControlRow[];
	baselineRow: string;
	controlRow: string;
	controlMinusRaw: number;
	contrast: NonReadingControlContrast;
	generatedBy: string;
}

export interface FramingCorrectionOk {
	status: 'ok';
	reason: null;
	artifact_path: string;
	artifact_sha256: string;
	control_path: string;
	control_sha256: string;
	framing: FramingCorrection;
	control: NonReadingControl;
}

export interface FramingCorrectionUnavailable {
	status: 'unavailable';
	reason: string;
	artifact_path: string;
	artifact_sha256: string | null;
	control_path: string;
	control_sha256: string | null;
	framing: null;
	control: null;
}

export type FramingCorrectionLoad = FramingCorrectionOk | FramingCorrectionUnavailable;

type UnknownRecord = Record<string, unknown>;









function textList(value: unknown, context: string): string[] {
	if (!Array.isArray(value)) fail(context, 'expected an array');
	return value.map((entry, index) => text(entry, `${context}[${index}]`));
}

/**
 * A control-strip row name, checked against the gutter it is drawn into. THROWS
 * rather than clipping: the caller (the server loader) gates the whole panel, and
 * a gated panel is honest where a label missing its first six characters is not.
 */
function budgetedStripLabel(value: string, context: string): string {
	if (value.length > FRAMING_CONTROL_LABEL_BUDGET_CHARS) {
		fail(
			context,
			`"${value}" is ${value.length} chars; the control-strip gutter budget is ${FRAMING_CONTROL_LABEL_BUDGET_CHARS}`
		);
	}
	return value;
}

function close(got: number, want: number, context: string, message: string): void {
	if (Math.abs(got - want) > FRAMING_PARITY_TOL) fail(context, message);
}

/** The exact formula literal, on either artifact. Both forms are checked. */
function requireFormula(value: unknown, context: string): string {
	const formula = text(value, context);
	if (formula !== FRAMING_NOISY_OR_FORMULA) {
		fail(context, `expected the noisy-OR verbatim — ${FRAMING_NOISY_OR_FORMULA}`);
	}
	if (formula.includes(FRAMING_WRONG_NOISY_OR_FRAGMENT)) {
		fail(context, 'carries the wrong noisy-OR form');
	}
	return formula;
}

function parseDigest(value: unknown, context: string): FramingSourceDigest {
	const obj = record(value, context);
	return {
		path: text(obj.path, `${context}.path`),
		sha256: text(obj.sha256, `${context}.sha256`)
	};
}

function parseDeclarationArm(value: unknown, index: number, context: string): FramingDeclarationArm {
	const spec = FRAMING_ARM_SPECS[index];
	const obj = record(value, context);
	if (obj.arm !== spec.key) {
		fail(`${context}.arm`, `expected the fixed presentation order — ${spec.key}`);
	}
	// The two conditions the whole argument rests on. A bundle that ran anything
	// else is not the paper's aggregation, and the panel must not draw it.
	if (obj.aggregation !== FRAMING_REQUIRED_AGGREGATION) {
		fail(`${context}.aggregation`, `expected ${FRAMING_REQUIRED_AGGREGATION}`);
	}
	if (obj.reader_profile !== null) {
		fail(`${context}.reader_profile`, 'expected null — a fitted profile floors belief above 0');
	}
	// The declaration is only worth printing if the script actually checked it
	// against bytes rather than transcribing the manifest.
	if (boolean(obj.aggregation_config_sha256_matches, `${context}.aggregation_config_sha256_matches`) !== true) {
		fail(`${context}.aggregation_config_sha256_matches`, 'must be true');
	}
	if (
		boolean(
			obj.implementation_component_sha256_matches,
			`${context}.implementation_component_sha256_matches`
		) !== true
	) {
		fail(`${context}.implementation_component_sha256_matches`, 'must be true');
	}
	const components = record(
		obj.implementation_component_sha256,
		`${context}.implementation_component_sha256`
	);
	return {
		key: spec.key,
		display: text(obj.label, `${context}.label`),
		manifestPath: text(obj.manifest_path, `${context}.manifest_path`),
		implementation: text(obj.implementation, `${context}.implementation`),
		aggregation: FRAMING_REQUIRED_AGGREGATION,
		readerProfile: null,
		dedup: boolean(obj.dedup, `${context}.dedup`),
		aggregationConfigSha256: text(
			obj.declared_aggregation_config_sha256,
			`${context}.declared_aggregation_config_sha256`
		),
		noiseModelSha256: text(components.noise_model, `${context}.implementation_component_sha256.noise_model`),
		statementBeliefSha256: text(
			components.statement_belief,
			`${context}.implementation_component_sha256.statement_belief`
		)
	};
}

function parseDeclaration(value: unknown, context: string): FramingDeclaration {
	const obj = record(value, context);
	if (obj.required_aggregation !== FRAMING_REQUIRED_AGGREGATION) {
		fail(`${context}.required_aggregation`, `expected ${FRAMING_REQUIRED_AGGREGATION}`);
	}
	if (!Array.isArray(obj.arms) || obj.arms.length !== FRAMING_ARM_SPECS.length) {
		fail(`${context}.arms`, `expected ${FRAMING_ARM_SPECS.length} arms in fixed order`);
	}
	const arms = obj.arms.map((entry, index) =>
		parseDeclarationArm(entry, index, `${context}.arms[${index}]`)
	);

	const priorsRaw = record(obj.panel_priors, `${context}.panel_priors`);
	const nPanelSources = Object.keys(priorsRaw).length;
	if (nPanelSources < 1) fail(`${context}.panel_priors`, 'expected at least one source');

	if (!Array.isArray(obj.prior_groups) || obj.prior_groups.length === 0) {
		fail(`${context}.prior_groups`, 'expected a non-empty array');
	}
	const priorGroups = obj.prior_groups.map((entry, index) => {
		const groupContext = `${context}.prior_groups[${index}]`;
		const group = record(entry, groupContext);
		return {
			rand: unit(group.rand, `${groupContext}.rand`),
			syst: unit(group.syst, `${groupContext}.syst`),
			sources: textList(group.sources, `${groupContext}.sources`),
			nStatements: positiveInteger(group.n_statements, `${groupContext}.n_statements`)
		};
	});
	// The groups partition the panel's sources; if they do not, the "one prior for
	// N readers" line would be quoting an incomplete census.
	const grouped = priorGroups.reduce((total, group) => total + group.sources.length, 0);
	if (grouped !== nPanelSources) {
		fail(`${context}.prior_groups`, 'must partition the panel sources exactly');
	}

	const claimProse: ShippedProse = {
		shipped: text(obj.claim, `${context}.claim`),
		plain: FRAMING_PLAIN.declarationClaim
	};
	const dispatchProse: ShippedProse = {
		shipped: text(obj.dispatch, `${context}.dispatch`),
		plain: FRAMING_PLAIN.declarationDispatch
	};
	return {
		claim: claimProse.shipped,
		claimProse,
		dispatch: dispatchProse.shipped,
		dispatchProse,
		requiredAggregation: FRAMING_REQUIRED_AGGREGATION,
		aggregationConfig: parseDigest(obj.aggregation_config, `${context}.aggregation_config`),
		noiseModelSource: parseDigest(
			record(obj.implementation_sources, `${context}.implementation_sources`).noise_model,
			`${context}.implementation_sources.noise_model`
		),
		statementBeliefSource: parseDigest(
			record(obj.implementation_sources, `${context}.implementation_sources`).statement_belief,
			`${context}.implementation_sources.statement_belief`
		),
		arms,
		priorGroups,
		nPanelSources
	};
}

function parseSubtractive(value: unknown, context: string, panelN: number): FramingSubtractive {
	const obj = record(value, context);
	const armsRaw = record(obj.arms, `${context}.arms`);
	let totalExceeding = 0;
	const arms = FRAMING_ARM_SPECS.map((spec) => {
		const armContext = `${context}.arms.${spec.key}`;
		const arm = record(armsRaw[spec.key], armContext);
		const nStatements = positiveInteger(arm.n_statements, `${armContext}.n_statements`);
		if (nStatements !== panelN) fail(`${armContext}.n_statements`, 'must be the panel size');
		const nExceeding = nonNegativeInteger(arm.n_exceeding_noisy_or, `${armContext}.n_exceeding_noisy_or`);
		// THE headline. A reader that can raise belief is not the paper's formula
		// on kept evidence, and every claim resting on that is void.
		if (nExceeding !== 0) {
			fail(`${armContext}.n_exceeding_noisy_or`, 'a reader belief exceeds the noisy-OR');
		}
		const nAtExactlyZero = nonNegativeInteger(arm.n_at_exactly_zero, `${armContext}.n_at_exactly_zero`);
		const nNonzero = nonNegativeInteger(arm.n_nonzero, `${armContext}.n_nonzero`);
		if (nAtExactlyZero + nNonzero !== panelN) {
			fail(armContext, 'n_at_exactly_zero + n_nonzero must equal the panel size');
		}
		const maxAbove = number(arm.max_belief_above_noisy_or, `${armContext}.max_belief_above_noisy_or`);
		if (maxAbove > 0) {
			fail(`${armContext}.max_belief_above_noisy_or`, 'must not be positive');
		}
		totalExceeding += nExceeding;
		return {
			key: spec.key,
			display: text(arm.label, `${armContext}.label`),
			nStatements,
			nExceedingNoisyOr: nExceeding,
			nAtExactlyZero,
			nNonzero,
			maxBeliefAboveNoisyOr: maxAbove,
			scoresPath: text(arm.scores_path, `${armContext}.scores_path`)
		};
	});

	const nComparisons = positiveInteger(obj.n_comparisons, `${context}.n_comparisons`);
	if (nComparisons !== panelN * FRAMING_ARM_SPECS.length) {
		fail(`${context}.n_comparisons`, 'must be the panel size times the arm count');
	}
	const nExceedingTotal = nonNegativeInteger(obj.n_exceeding_noisy_or, `${context}.n_exceeding_noisy_or`);
	if (nExceedingTotal !== 0 || totalExceeding !== 0) {
		fail(`${context}.n_exceeding_noisy_or`, 'must be zero across every arm');
	}

	const crossCheck = record(obj.cross_check, `${context}.cross_check`);
	if (boolean(crossCheck.agrees, `${context}.cross_check.agrees`) !== true) {
		fail(`${context}.cross_check.agrees`, 'must be true');
	}

	const claimProse: ShippedProse = {
		shipped: text(obj.claim, `${context}.claim`),
		plain: FRAMING_PLAIN.subtractiveClaim
	};
	const baselineProse: ShippedProse = {
		shipped: text(obj.baseline, `${context}.baseline`),
		plain: FRAMING_PLAIN.subtractiveBaseline
	};
	return {
		claim: claimProse.shipped,
		claimProse,
		baseline: baselineProse.shipped,
		baselineProse,
		baselineScores: text(obj.baseline_scores, `${context}.baseline_scores`),
		arms,
		nComparisons,
		nExceedingNoisyOr: 0,
		maxBeliefAboveNoisyOr: number(obj.max_belief_above_noisy_or, `${context}.max_belief_above_noisy_or`),
		crossCheckArtifact: text(crossCheck.artifact, `${context}.cross_check.artifact`),
		crossCheckSha256: text(crossCheck.sha256, `${context}.cross_check.sha256`)
	};
}

function parseReachable(value: unknown, context: string, panelN: number): FramingReachable {
	const obj = record(value, context);
	const tolerance = number(obj.tolerance, `${context}.tolerance`);
	if (!(tolerance > 0)) fail(`${context}.tolerance`, 'expected a positive tolerance');

	const search = record(obj.search, `${context}.search`);
	const nodeBudget = positiveInteger(search.node_budget_per_statement, `${context}.search.node_budget_per_statement`);
	const maxNodes = positiveInteger(search.max_nodes_used, `${context}.search.max_nodes_used`);
	if (maxNodes > nodeBudget) {
		fail(`${context}.search.max_nodes_used`, 'cannot exceed the node budget it ran under');
	}

	const nullRaw = record(obj.null_baseline, `${context}.null_baseline`);
	const nullArmsRaw = record(nullRaw.arms, `${context}.null_baseline.arms`);
	const armsRaw = record(obj.arms, `${context}.arms`);

	const arms = FRAMING_ARM_SPECS.map((spec) => {
		const armContext = `${context}.arms.${spec.key}`;
		const arm = record(armsRaw[spec.key], armContext);
		const nAtExactlyZero = nonNegativeInteger(arm.n_at_exactly_zero, `${armContext}.n_at_exactly_zero`);
		const nNonzero = positiveInteger(arm.n_nonzero, `${armContext}.n_nonzero`);
		if (nAtExactlyZero + nNonzero !== panelN) {
			fail(armContext, 'n_at_exactly_zero + n_nonzero must equal the panel size');
		}
		const nConfirmed = nonNegativeInteger(arm.n_confirmed_reachable, `${armContext}.n_confirmed_reachable`);
		const nBitExact = nonNegativeInteger(arm.n_bit_exact, `${armContext}.n_bit_exact`);
		const nExhausted = nonNegativeInteger(arm.n_budget_exhausted, `${armContext}.n_budget_exhausted`);
		const nCounter = nonNegativeInteger(arm.n_counterexamples, `${armContext}.n_counterexamples`);
		// Every non-zero score is accounted for exactly once, in exactly one tier.
		if (nConfirmed + nExhausted + nCounter !== nNonzero) {
			fail(armContext, 'confirmed + budget-exhausted + counterexamples must equal n_nonzero');
		}
		// A counterexample means a score the paper's formula cannot emit. The panel
		// would be asserting the opposite, so it refuses to draw.
		if (nCounter !== 0) {
			fail(`${armContext}.n_counterexamples`, 'a non-zero score is not a value the formula can emit');
		}
		// The tighter tier is a subset of the looser one, never larger.
		if (nBitExact > nConfirmed) {
			fail(`${armContext}.n_bit_exact`, 'cannot exceed the confirmed count');
		}
		close(
			unit(arm.share_confirmed, `${armContext}.share_confirmed`),
			nConfirmed / nNonzero,
			`${armContext}.share_confirmed`,
			'must equal n_confirmed_reachable / n_nonzero'
		);
		close(
			unit(arm.share_bit_exact, `${armContext}.share_bit_exact`),
			nBitExact / nNonzero,
			`${armContext}.share_bit_exact`,
			'must equal n_bit_exact / n_nonzero'
		);

		const nullContext = `${context}.null_baseline.arms.${spec.key}`;
		const nullArm = record(nullArmsRaw[spec.key], nullContext);
		const permutedMean = unit(nullArm.permuted_rate_mean, `${nullContext}.permuted_rate_mean`);
		const permutedMin = unit(nullArm.permuted_rate_min, `${nullContext}.permuted_rate_min`);
		const permutedMax = unit(nullArm.permuted_rate_max, `${nullContext}.permuted_rate_max`);
		if (!(permutedMin <= permutedMean && permutedMean <= permutedMax)) {
			fail(nullContext, 'permuted rate mean must lie inside its own replication range');
		}

		return {
			key: spec.key,
			display: text(arm.label, `${armContext}.label`),
			nAtExactlyZero,
			nNonzero,
			nConfirmedReachable: nConfirmed,
			nBitExact,
			nBudgetExhausted: nExhausted,
			nCounterexamples: 0,
			shareConfirmed: nConfirmed / nNonzero,
			shareBitExact: nBitExact / nNonzero,
			permutedRateMean: permutedMean,
			permutedRateMin: permutedMin,
			permutedRateMax: permutedMax,
			nEnumerableNonzero: positiveInteger(
				nullArm.n_enumerable_nonzero,
				`${nullContext}.n_enumerable_nonzero`
			)
		};
	});

	const pooledMean = unit(nullRaw.pooled_permuted_rate_mean, `${context}.null_baseline.pooled_permuted_rate_mean`);
	const pooledMin = unit(nullRaw.pooled_permuted_rate_min, `${context}.null_baseline.pooled_permuted_rate_min`);
	const pooledMax = unit(nullRaw.pooled_permuted_rate_max, `${context}.null_baseline.pooled_permuted_rate_max`);
	if (!(pooledMin <= pooledMean && pooledMean <= pooledMax)) {
		fail(`${context}.null_baseline`, 'pooled permuted rate must lie inside its own range');
	}
	const nEnumerable = positiveInteger(
		nullRaw.n_statements_enumerable,
		`${context}.null_baseline.n_statements_enumerable`
	);
	if (nEnumerable > panelN) {
		fail(`${context}.null_baseline.n_statements_enumerable`, 'cannot exceed the panel');
	}

	const floorRaw = record(obj.noisy_or_floor_on_panel, `${context}.noisy_or_floor_on_panel`);
	const floor = unit(floorRaw.value, `${context}.noisy_or_floor_on_panel.value`);
	// The zero block only means "the reader rejected everything" while the formula
	// itself never emits 0 on this panel.
	if (!(floor > 0)) {
		fail(`${context}.noisy_or_floor_on_panel.value`, 'must be above zero');
	}

	const claimProse: ShippedProse = {
		shipped: text(obj.claim, `${context}.claim`),
		plain: FRAMING_PLAIN.reachableClaim
	};
	const definitionProse: ShippedProse = {
		shipped: text(obj.definition, `${context}.definition`),
		plain: FRAMING_PLAIN.reachableDefinition
	};
	const noisyOrFloorNoteProse: ShippedProse = {
		shipped: text(floorRaw.note, `${context}.noisy_or_floor_on_panel.note`),
		plain: FRAMING_PLAIN.noisyOrFloorNote
	};
	const nullQuestionProse: ShippedProse = {
		shipped: text(nullRaw.question, `${context}.null_baseline.question`),
		plain: FRAMING_PLAIN.nullBaselineQuestion
	};
	const nullMethodProse: ShippedProse = {
		shipped: text(nullRaw.method, `${context}.null_baseline.method`),
		plain: FRAMING_PLAIN.nullBaselineMethod
	};
	return {
		claim: claimProse.shipped,
		claimProse,
		definition: definitionProse.shipped,
		definitionProse,
		tolerance,
		nodeBudgetPerStatement: nodeBudget,
		maxNodesUsed: maxNodes,
		sourceCountsPath: text(obj.source_counts, `${context}.source_counts`),
		arms,
		nullBaseline: {
			question: nullQuestionProse.shipped,
			questionProse: nullQuestionProse,
			method: nullMethodProse.shipped,
			methodProse: nullMethodProse,
			nPermutations: positiveInteger(nullRaw.n_permutations, `${context}.null_baseline.n_permutations`),
			seed: positiveInteger(nullRaw.seed, `${context}.null_baseline.seed`),
			nStatementsEnumerable: nEnumerable,
			nStatementsOnPanel: panelN,
			pooledMean,
			pooledMin,
			pooledMax
		},
		noisyOrFloor: floor,
		noisyOrFloorNote: noisyOrFloorNoteProse.shipped,
		noisyOrFloorNoteProse
	};
}

/**
 * Pure, fail-closed parse of `framing_correction.json`. THROWS on any drift —
 * shape, arm order, an arm that did not run the unfitted hard gate, a bundle
 * carrying a fitted reader profile, a reader belief above the noisy-OR, a
 * counterexample to the reachable-value claim, counts that do not partition the
 * panel or the non-zero scores, a negative breakdown that does not sum, a failed
 * check flag, or a formula string that is not the noisy-OR verbatim.
 */
export function validateFramingCorrection(raw: unknown): FramingCorrection {
	const obj = record(raw, 'framing_correction');
	if (obj.artifact_kind !== 'framing_correction') {
		fail('framing_correction.artifact_kind', 'expected framing_correction');
	}
	if (obj.schema_version !== 1) fail('framing_correction.schema_version', 'expected 1');
	if (obj.aggregation !== FRAMING_REQUIRED_AGGREGATION) {
		fail('framing_correction.aggregation', `expected ${FRAMING_REQUIRED_AGGREGATION}`);
	}
	const noisyOrFormula = requireFormula(obj.noisy_or_formula, 'framing_correction.noisy_or_formula');

	const panelRaw = record(obj.panel, 'framing_correction.panel');
	const n = positiveInteger(panelRaw.n, 'framing_correction.panel.n');
	const nErrors = positiveInteger(panelRaw.n_errors, 'framing_correction.panel.n_errors');
	const nCorrect = positiveInteger(panelRaw.n_correct, 'framing_correction.panel.n_correct');
	if (nErrors + nCorrect !== n) {
		fail('framing_correction.panel', 'n_errors + n_correct must equal n');
	}
	const errorBaseRate = unit(panelRaw.error_base_rate, 'framing_correction.panel.error_base_rate');
	close(errorBaseRate, nErrors / n, 'framing_correction.panel.error_base_rate', 'must equal n_errors / n');

	// The label convention the page discloses exactly once. If the two parts do
	// not sum to the error count, the disclosure would be wrong, so refuse it.
	const breakdownRaw = record(panelRaw.negative_breakdown, 'framing_correction.panel.negative_breakdown');
	const safe = nonNegativeInteger(
		breakdownRaw.adjudication_safe_negatives,
		'framing_correction.panel.negative_breakdown.adjudication_safe_negatives'
	);
	const flagged = nonNegativeInteger(
		breakdownRaw.flagged_label_is_adjudication_safe_false,
		'framing_correction.panel.negative_breakdown.flagged_label_is_adjudication_safe_false'
	);
	const breakdownErrors = positiveInteger(
		breakdownRaw.n_errors,
		'framing_correction.panel.negative_breakdown.n_errors'
	);
	if (breakdownErrors !== nErrors || safe + flagged !== nErrors) {
		fail(
			'framing_correction.panel.negative_breakdown',
			'adjudication-safe + flagged must equal the panel error count'
		);
	}

	const labelConventionProse: ShippedProse = {
		shipped: text(panelRaw.label_convention, 'framing_correction.panel.label_convention'),
		plain: FRAMING_PLAIN.labelConvention
	};
	const panel: FramingPanel = {
		n,
		nErrors,
		nCorrect,
		errorBaseRate,
		labelField: text(panelRaw.label, 'framing_correction.panel.label'),
		labelConvention: labelConventionProse.shipped,
		labelConventionProse,
		negativeBreakdown: {
			nErrors,
			adjudicationSafeNegatives: safe,
			flaggedNotAdjudicationSafe: flagged
		}
	};

	const declaration = parseDeclaration(obj.declaration, 'framing_correction.declaration');
	const subtractive = parseSubtractive(obj.subtractive, 'framing_correction.subtractive', n);
	const reachable = parseReachable(obj.reachable_values, 'framing_correction.reachable_values', n);

	// The two legs measure the same arms on the same panel; the zero block is the
	// one number they share, so a disagreement means one of them is stale.
	subtractive.arms.forEach((arm, index) => {
		const other = reachable.arms[index];
		if (arm.key !== other.key || arm.nAtExactlyZero !== other.nAtExactlyZero) {
			fail(
				`framing_correction.reachable_values.arms.${other.key}.n_at_exactly_zero`,
				'must agree with the subtractive check on the same arm'
			);
		}
	});

	// The script asserts these in Python and fails the build on a violation; hold
	// the same line here so a hand-edited artifact cannot slip a false one past.
	const checks = record(obj.checks, 'framing_correction.checks');
	for (const key of [
		'every_manifest_declares_the_unfitted_hard_gate',
		'every_manifest_aggregation_config_sha_matches_the_tree',
		'every_manifest_component_digest_matches_the_source',
		'readers_never_exceed_the_noisy_or',
		'subtractive_agrees_with_non_reading_control',
		'every_nonzero_score_is_reachable',
		'zero_and_nonzero_partition_the_panel',
		'negative_breakdown_agrees_with_belief_model_ladder',
		'gold_matches_hash_agrees_with_prediction_provenance'
	]) {
		if (boolean(checks[key], `framing_correction.checks.${key}`) !== true) {
			fail(`framing_correction.checks.${key}`, 'must be true');
		}
	}

	const provenanceRaw = record(obj.provenance, 'framing_correction.provenance');
	const provenance: Record<string, string> = {};
	for (const [key, value] of Object.entries(provenanceRaw)) {
		provenance[key] = text(value, `framing_correction.provenance.${key}`);
	}

	const questionProse: ShippedProse = {
		shipped: text(obj.question, 'framing_correction.question'),
		plain: FRAMING_PLAIN.question
	};
	const findingProse: ShippedProse = {
		shipped: text(obj.finding, 'framing_correction.finding'),
		plain: FRAMING_PLAIN.finding
	};
	const caveats = textList(obj.caveats, 'framing_correction.caveats');
	// Positional twins: a reissued artifact that reorders or rewrites a caveat
	// gates the figure rather than printing restatement N under caveat N+1.
	const caveatProse = pairShippedProse(
		caveats,
		FRAMING_CAVEAT_TWINS,
		'framing_correction.caveats'
	);

	const noisyOrFormulaProse: ShippedProse = {
		shipped: noisyOrFormula,
		plain: FRAMING_PLAIN.noisyOrFormula
	};
	return {
		question: questionProse.shipped,
		finding: findingProse.shipped,
		noisyOrFormula,
		noisyOrFormulaProse,
		aggregation: FRAMING_REQUIRED_AGGREGATION,
		panel,
		declaration,
		subtractive,
		reachable,
		caveats,
		generatedBy: text(provenanceRaw.generated_by, 'framing_correction.provenance.generated_by'),
		provenance,
		prose: {
			question: questionProse,
			finding: findingProse,
			noisyOrFormula: noisyOrFormulaProse,
			labelConvention: panel.labelConventionProse,
			declarationClaim: declaration.claimProse,
			declarationDispatch: declaration.dispatchProse,
			subtractiveClaim: subtractive.claimProse,
			subtractiveBaseline: subtractive.baselineProse,
			reachableClaim: reachable.claimProse,
			reachableDefinition: reachable.definitionProse,
			nullBaselineQuestion: reachable.nullBaseline.questionProse,
			nullBaselineMethod: reachable.nullBaseline.methodProse,
			noisyOrFloorNote: reachable.noisyOrFloorNoteProse,
			caveats: caveatProse
		}
	};
}

/**
 * Pure, fail-closed parse of `non_reading_control.json` for leg (d). THROWS if the
 * control does not land below the ungated baseline it is drawn against — that
 * inequality IS the strip's claim, so a file asserting otherwise is not drawn.
 */
export function validateNonReadingControl(raw: unknown): NonReadingControl {
	const obj = record(raw, 'non_reading_control');
	if (obj.artifact_kind !== 'non_reading_control') {
		fail('non_reading_control.artifact_kind', 'expected non_reading_control');
	}
	if (obj.schema_version !== 1) fail('non_reading_control.schema_version', 'expected 1');
	if (obj.aggregation !== FRAMING_REQUIRED_AGGREGATION) {
		fail('non_reading_control.aggregation', `expected ${FRAMING_REQUIRED_AGGREGATION}`);
	}
	const noisyOrFormula = requireFormula(obj.noisy_or_formula, 'non_reading_control.noisy_or_formula');

	if (!Array.isArray(obj.rows) || obj.rows.length < 2) {
		fail('non_reading_control.rows', 'expected at least a baseline and a control row');
	}
	const rows = obj.rows.map((entry, index) => {
		const context = `non_reading_control.rows[${index}]`;
		const row = record(entry, context);
		const rowKey = text(row.key, `${context}.key`);
		// Looked up by the FROZEN row key, then pinned to its own text: a row with
		// no authored restatement gates the strip rather than reaching a reader in
		// the artifact's own wording.
		const noteProse = keyedShippedProse(
			rowKey,
			text(row.note, `${context}.note`),
			CONTROL_ROW_TWINS,
			`${context}.note`
		);
		return {
			key: rowKey,
			display: budgetedStripLabel(text(row.label, `${context}.label`), `${context}.label`),
			weight: text(row.weight, `${context}.weight`),
			droppedRoutes: Array.isArray(row.dropped_routes)
				? textList(row.dropped_routes, `${context}.dropped_routes`)
				: fail(`${context}.dropped_routes`, 'expected an array'),
			nEvidenceScored: positiveInteger(row.n_evidence_scored, `${context}.n_evidence_scored`),
			averagePrecision: unit(row.average_precision, `${context}.average_precision`),
			deltaVsRawNoisyOr: number(row.delta_vs_raw_noisy_or, `${context}.delta_vs_raw_noisy_or`),
			note: noteProse.shipped,
			noteProse
		};
	});

	const baselineRow = text(obj.baseline_row, 'non_reading_control.baseline_row');
	const controlRow = text(obj.control_row, 'non_reading_control.control_row');
	const baseline = rows.find((row) => row.key === baselineRow);
	const control = rows.find((row) => row.key === controlRow);
	if (!baseline) fail('non_reading_control.baseline_row', 'names a row that is not in rows');
	if (!control) fail('non_reading_control.control_row', 'names a row that is not in rows');

	// The strip's whole point: with no model verdict anywhere, the three
	// non-reading subtractions land BELOW the ungated noisy-OR.
	if (boolean(obj.control_lands_below_raw, 'non_reading_control.control_lands_below_raw') !== true) {
		fail('non_reading_control.control_lands_below_raw', 'must be true');
	}
	if (!(control.averagePrecision < baseline.averagePrecision)) {
		fail('non_reading_control.rows', 'the control must sit below the ungated baseline');
	}
	const controlMinusRaw = number(
		obj.control_minus_raw_average_precision,
		'non_reading_control.control_minus_raw_average_precision'
	);
	close(
		controlMinusRaw,
		control.averagePrecision - baseline.averagePrecision,
		'non_reading_control.control_minus_raw_average_precision',
		'must equal the control row minus the baseline row'
	);
	rows.forEach((row, index) => {
		close(
			row.deltaVsRawNoisyOr,
			row.averagePrecision - baseline.averagePrecision,
			`non_reading_control.rows[${index}].delta_vs_raw_noisy_or`,
			'must equal this row minus the baseline row'
		);
	});

	const contrastRaw = record(obj.contrast, 'non_reading_control.contrast');
	const contrastAp = unit(contrastRaw.average_precision, 'non_reading_control.contrast.average_precision');
	const contrast: NonReadingControlContrast = {
		key: text(contrastRaw.key, 'non_reading_control.contrast.key'),
		display: budgetedStripLabel(
			text(contrastRaw.label, 'non_reading_control.contrast.label'),
			'non_reading_control.contrast.label'
		),
		averagePrecision: contrastAp,
		deltaVsRawNoisyOr: number(
			contrastRaw.delta_vs_raw_noisy_or,
			'non_reading_control.contrast.delta_vs_raw_noisy_or'
		),
		deltaVsFullControl: number(
			contrastRaw.delta_vs_full_control,
			'non_reading_control.contrast.delta_vs_full_control'
		)
	};
	close(
		contrast.deltaVsRawNoisyOr,
		contrastAp - baseline.averagePrecision,
		'non_reading_control.contrast.delta_vs_raw_noisy_or',
		'must equal the contrast minus the baseline row'
	);
	close(
		contrast.deltaVsFullControl,
		contrastAp - control.averagePrecision,
		'non_reading_control.contrast.delta_vs_full_control',
		'must equal the contrast minus the control row'
	);

	const checks = record(obj.checks, 'non_reading_control.checks');
	for (const key of ['full_control_below_raw', 'readers_never_exceed_the_noisy_or']) {
		if (boolean(checks[key], `non_reading_control.checks.${key}`) !== true) {
			fail(`non_reading_control.checks.${key}`, 'must be true');
		}
	}

	const provenance = record(obj.provenance, 'non_reading_control.provenance');
	const questionProse: ShippedProse = {
		shipped: text(obj.question, 'non_reading_control.question'),
		plain: FRAMING_PLAIN.controlQuestion
	};
	const findingProse: ShippedProse = {
		shipped: text(obj.finding, 'non_reading_control.finding'),
		plain: FRAMING_PLAIN.controlFinding
	};
	const metricProse: ShippedProse = {
		shipped: text(obj.metric, 'non_reading_control.metric'),
		plain: FRAMING_PLAIN.controlMetric
	};
	const metricSourceProse: ShippedProse = {
		shipped: text(obj.metric_source, 'non_reading_control.metric_source'),
		plain: FRAMING_PLAIN.controlMetricSource
	};
	const controlFormulaProse: ShippedProse = {
		shipped: noisyOrFormula,
		plain: FRAMING_PLAIN.noisyOrFormula
	};
	return {
		question: questionProse.shipped,
		questionProse,
		finding: findingProse.shipped,
		findingProse,
		metric: metricProse.shipped,
		metricProse,
		metricSource: metricSourceProse.shipped,
		metricSourceProse,
		prose: {
			question: questionProse,
			finding: findingProse,
			metric: metricProse,
			metricSource: metricSourceProse,
			noisyOrFormula: controlFormulaProse,
			// The SAME objects the rows carry, in the artifact's own order.
			rowNotes: rows.map((row) => row.noteProse)
		},
		noisyOrFormula,
		noisyOrFormulaProse: controlFormulaProse,
		rows,
		baselineRow,
		controlRow,
		controlMinusRaw,
		contrast,
		generatedBy: text(provenance.generated_by, 'non_reading_control.provenance.generated_by')
	};
}

/**
 * The two artifacts must be describing the SAME run. The framing correction pins
 * the control's sha256 and re-derives its subtractive block; this is the viewer's
 * end of that pairing, so a panel can never show leg (b) from one run beside leg
 * (d) from another. THROWS on disagreement.
 */
export function crossCheckFramingAndControl(
	framing: FramingCorrection,
	control: NonReadingControl
): void {
	if (framing.noisyOrFormula !== control.noisyOrFormula) {
		fail('framing_correction', 'states a different noisy-OR from the non-reading control');
	}
	if (framing.subtractive.crossCheckArtifact.split('/').pop() !== 'non_reading_control.json') {
		fail('framing_correction.subtractive.cross_check.artifact', 'must name the non-reading control');
	}
}

/** Shared hue resolution — the same token the other /paper panels use. */
export function framingArmColorVar(index: number): string {
	return paperArmColorVar(FRAMING_ARM_SPECS[index]?.paperKind ?? 'llm');
}

/** The hue for the paper's own rows in the control strip. */
export function framingPaperColorVar(): string {
	return paperArmColorVar('paper');
}

/**
 * The prior group the "one prior for N readers" line names: the largest group of
 * sources sharing a single (rand, syst) pair. Derived, never named by hand.
 */
export function framingLargestPriorGroup(framing: FramingCorrection): FramingPriorGroup | null {
	if (framing.declaration.priorGroups.length === 0) return null;
	return framing.declaration.priorGroups.reduce((best, group) =>
		group.sources.length > best.sources.length ? group : best
	);
}

/**
 * The unresolved remainder across all arms — statements the reachable-value
 * search could not settle inside its budget. Zero today; if it is ever non-zero
 * the panel names it rather than printing an unearned 100%.
 */
export function framingUnresolvedTotal(framing: FramingCorrection): number {
	return framing.reachable.arms.reduce((total, arm) => total + arm.nBudgetExhausted, 0);
}
