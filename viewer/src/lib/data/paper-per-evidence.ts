import { boolean, budget, fail, nonNegativeInteger, number, positiveInteger, record, text, unit } from './paper-validate.ts';
/**
 * Typed data contract for the PER-EVIDENCE grain surface.
 *
 * WHAT THIS PAGE OTHERWISE MISSES. Every other figure on /paper is at STATEMENT
 * grain. The reader's native act is not a statement score: it emits one
 * correct/incorrect verdict per (statement, evidence) pair, and statement belief
 * is that verdict set pushed through INDRA's noisy-OR. Judging the system only on
 * the derived quantity measures the aggregation as much as the reading.
 *
 * THE FIGURE IS ONE PLATE WITH TWO MARKS PER ARM on a single AUROC axis: the
 * per-evidence mark (5,379 reviewed pairs) and the statement mark (1,689
 * statements), joined by a connector. Both come from shipped artifacts for the
 * SAME model, so the connector shows what aggregation does to discrimination.
 * It is NOT a paired increment: the two marks are measurements on two different
 * item populations at two different base rates (65.4% vs 73.2% positive), and
 * `twoGrainNote` carries that wording to every render site.
 *
 * MISATTRIBUTION BAN, enforced structurally. None of the baselines here is a
 * published paper method:
 *   - the bundled source prior is INDRA's UNFITTED default, not the paper's
 *     "Belief Orig", which refits per fold by MCMC;
 *   - the BayesianScorer arms are INDRA 1.24.0 library code refit out-of-fold on
 *     this panel's own curation counts — the 2023 paper publishes NO Bayesian and
 *     NO subtype arm;
 *   - the reader arms are NOT zero-shot (14 hand-authored demonstration pairs per
 *     call).
 * Every arm carries a non-empty `attribution` and the validator GATES the whole
 * figure if any is missing, so no consumer can draw a bar without its provenance.
 * The paper's own supervised Table 6 rows (RF / Log LR / KNN / SVC) cannot appear
 * at all — their features are statement-level aggregates and they emit no
 * per-evidence quantity. That exclusion is shipped as data, not as prose.
 *
 * SHIPPED FIELDS ARE READ, NEVER RECOMPUTED. Every AUROC, average precision,
 * interval bound, census count, chi-square and reconciliation residual comes off
 * `data/results/per_evidence_comparison_20260727/per_evidence_comparison.json`.
 * The only arithmetic here is axis placement and the consistency gates below,
 * which compare shipped fields against each other and fail closed.
 *
 * This module is import-safe on the client: typed shape plus a pure, fail-closed
 * validator. All filesystem work lives in `$lib/server/paper-per-evidence`.
 */

/** The artifact kind this module will accept, and nothing else. */
export const PAPER_PER_EVIDENCE_ARTIFACT_KIND = 'indra_per_evidence_comparison';

/**
 * On-screen names, keyed by arm id. DECOUPLED from the artifact's `join_key`,
 * which is the FROZEN directory / arm identifier addressing already-emitted data.
 * Render `display`, never `joinKey`.
 *
 * These are ours to write in English. `OOF` shipped here and reached the screen
 * as `INDRA Bayes source (OOF)`: an acronym for out-of-fold that a working
 * biologist has no way to resolve, sitting in a right-anchored lane gutter with
 * nowhere to gloss it. The figure's own basis note already says the plain thing —
 * "refitted on data they never learned from … the held-out fold" — so the lane
 * now says `held out` and the note carries the mechanism. `fold` itself is kept:
 * it is the field's standard term, and the basis note defines it in place.
 */
import {
	keyedShippedProse,
	standingOfBounds,
	type AnchoredProse,
	type ShippedProse,
	type Standing
} from './paper-literal.ts';

/**
 * THE PLAIN HALF OF EVERY TWIN THIS MODULE EMITS.
 *
 * `per_evidence_comparison.json` explains itself in seven per-model provenance
 * sentences plus five method notes, and every one of them shipped raw: "refit
 * out-of-fold", "the 2023 paper publishes NO Bayesian arm", "source-stratified
 * paired bootstrap". The restatements below keep every disclosure at full
 * strength — the clustering the interval does NOT model, the two populations
 * that are NOT paired, and the reason no trapezoidal number is quoted.
 */
const PER_EVIDENCE_PLAIN = {
	bootstrapDesign:
		'Evidence pairs are re-drawn at random with replacement WITHIN each source, and ONE redraw ' +
		'is shared by every model and every source, so the models are compared on the same draw.',
	contaminationGrain:
		'matched on the content hash of each reviewed pair — the corpus text of the evidence itself',
	powerNote:
		'The statement figure covers 1,689 items; this one covers 5,379 reviewed evidence pairs ' +
		'drawn from those same statements. Pairs belonging to one statement are not independent, ' +
		'so the real gain in resolution is below the raw 3.2x ratio, and the interval here ' +
		're-draws pairs within a source without modelling that grouping.',
	twoGrainNote:
		'A statement mark and an evidence mark are two measurements of ONE model over two ' +
		'different populations of items (1,689 statements, 73.2% of them right, against 5,379 ' +
		'evidence pairs, 65.4% of them right). The line between them shows what changes when ' +
		'INDRA’s combination rule turns per-evidence verdicts into one statement score; it is not ' +
		'a causal step up, and the two AUROCs are not paired.',
	estimatorNote:
		'deliberately absent: a reading model’s per-evidence score takes at most five distinct ' +
		'values, which is exactly the regime where joining adjacent points with straight lines ' +
		'inflates the number most',
	excludedBaselineFamily:
		'the supervised rows of Table 6 in the 2023 paper (random forest / logistic ' +
		'log-likelihood-ratio / k-nearest neighbours / support-vector classifier, with and without ' +
		'the statement-type, #PMIDs, promoter and mean-evidence-length features)',
	excludedBaselineReason:
		'statement-level only: every feature is an aggregate over the whole statement (#PMIDs, ' +
		'statement type, whether a promoter is mentioned, mean evidence length) and the models ' +
		'emit nothing per evidence pair, so there is no per-evidence score to draw'
} as const;

/**
 * The seven provenance sentences, keyed by the FROZEN arm id. Four of the seven
 * ship the SAME text, so each is additionally pinned to a verbatim fragment —
 * a key alone would let a scorer's restatement land on a reading model.
 */
const PER_EVIDENCE_ATTRIBUTION_TWINS: Readonly<Record<string, AnchoredProse>> = {
	'llm-gemma-4-e2b': {
		artifactAnchor: 'Our LLM evidence reader',
		plain:
			'A language model reading the evidence. NOT zero-shot: every call carries 14 hand-written ' +
			'example pairs. One verdict per (statement, evidence) pair; the statement’s belief is ' +
			'those verdicts pushed through INDRA’s own combination rule.'
	},
	'llm-gemma-4-26b': {
		artifactAnchor: 'Our LLM evidence reader',
		plain:
			'A language model reading the evidence. NOT zero-shot: every call carries 14 hand-written ' +
			'example pairs. One verdict per (statement, evidence) pair; the statement’s belief is ' +
			'those verdicts pushed through INDRA’s own combination rule.'
	},
	'llm-gemma-4-31b': {
		artifactAnchor: 'Our LLM evidence reader',
		plain:
			'A language model reading the evidence. NOT zero-shot: every call carries 14 hand-written ' +
			'example pairs. One verdict per (statement, evidence) pair; the statement’s belief is ' +
			'those verdicts pushed through INDRA’s own combination rule.'
	},
	'llm-glm-5': {
		artifactAnchor: 'Our LLM evidence reader',
		plain:
			'A language model reading the evidence. NOT zero-shot: every call carries 14 hand-written ' +
			'example pairs. One verdict per (statement, evidence) pair; the statement’s belief is ' +
			'those verdicts pushed through INDRA’s own combination rule.'
	},
	'indra-default-source-prior': {
		artifactAnchor: 'UNFITTED bundled default',
		plain:
			'INDRA’s library-default per-source reliabilities, scored at one piece of evidence. The ' +
			'belief model published in 2023 refits these fold by fold — the 10 groups the statements ' +
			'were split into — by MCMC; this is the UNFITTED bundled default and is not that model.'
	},
	'indra-bayes-source-oof': {
		artifactAnchor: 'publishes NO Bayesian arm',
		plain:
			'INDRA 1.24.0 BayesianScorer, with per-source reliabilities refit from these statements’ ' +
			'own curation counts and every pair scored by a copy that never saw it. The 2023 paper ' +
			'publishes NO Bayesian model; this is library code, not a published method.'
	},
	'indra-bayes-subtype-oof': {
		artifactAnchor: 'publishes NO subtype arm',
		plain:
			'INDRA 1.24.0 BayesianScorer, with per-(source, evidence subtype) reliabilities refit ' +
			'from these statements’ own curation counts and every pair scored by a copy that never ' +
			'saw it. The 2023 paper publishes NO subtype model; this is library code, not a ' +
			'published method.'
	}
};

export const PAPER_PER_EVIDENCE_DISPLAY: Readonly<Record<string, string>> = {
	'llm-gemma-4-e2b': 'Gemma 4 E2B',
	'llm-gemma-4-26b': 'Gemma 4 26B',
	'llm-gemma-4-31b': 'Gemma 4 31B',
	'llm-glm-5': 'GLM-5',
	'indra-default-source-prior': 'INDRA source prior (bundled)',
	'indra-bayes-source-oof': 'INDRA Bayes source (held out)',
	'indra-bayes-subtype-oof': 'INDRA Bayes subtype (held out)'
};

/**
 * What the ARTIFACT calls each arm — sha-pinned bytes, frozen, never edited from
 * here. The validator checks the shipped `display` against THIS table, so a
 * reissued artifact that renames an arm still takes the figure down, while the
 * on-screen name above stays ours to fix in plain English. Same split, and for
 * the same reason, as `PaperLiteralArmSpec.artifactDisplay`.
 */
export const PAPER_PER_EVIDENCE_ARTIFACT_DISPLAY: Readonly<Record<string, string>> = {
	'llm-gemma-4-e2b': 'Gemma 4 E2B',
	'llm-gemma-4-26b': 'Gemma 4 26B',
	'llm-gemma-4-31b': 'Gemma 4 31B',
	'llm-glm-5': 'GLM-5',
	'indra-default-source-prior': 'INDRA source prior (bundled)',
	'indra-bayes-source-oof': 'INDRA Bayes source (OOF)',
	'indra-bayes-subtype-oof': 'INDRA Bayes subtype (OOF)'
};

/**
 * SVG geometry. Exported so the label budgets below are DERIVED from it rather
 * than eyeballed: right-anchored SVG text that overruns its gutter loses its
 * LEADING glyphs silently — no layout error, no test failure, and a `<desc>`
 * hides it from review. Same shape as `paper-own-metric.ts`, which does this
 * correctly.
 */
export const PAPER_PER_EVIDENCE_GEOMETRY = {
	width: 920,
	plotLeft: 196,
	plotRight: 782,
	/** Lane names are right-anchored here; usable gutter is 0 → 184 units. */
	labelAnchorX: 184,
	/** Value readouts are left-anchored here; usable gutter is 790 → 920. */
	readoutX: 790,
	labelFontPx: 9,
	readoutFontPx: 8,
	/** Measured advance of the mono face at 9px, in user units per character. */
	monoUnitsPerChar: 5.4186,
	/** The same face at 8px: 5.4186 × 8/9. */
	readoutUnitsPerChar: 4.8165,
	laneHeight: 30,
	/** Row offsets inside a lane: evidence mark above, statement mark below. */
	evidenceOffset: -6,
	statementOffset: 7,
	bandHeaderHeight: 26,
	topPad: 16,
	/**
	 * Each register closes with its OWN axis rule and tick row. The two registers
	 * carry different quantities (AUROC above, probability below), so they may not
	 * share ticks and their grid lines may not cross into each other.
	 */
	axisRowHeight: 26,
	/** Room under the lower axis for the one shared axis caption. */
	axisPad: 22,
	/** Half-height of an interval end cap. */
	intervalCap: 3.5,
	/** Per-source strip: one row per source under the baseline band. */
	sourceRowHeight: 15,
	sourceHeaderHeight: 24
} as const;

/**
 * LANE NAME BUDGET: 184 units ÷ 5.4186 u/char at 9px = 33.9 → 33 characters.
 * The longest name this table can produce is "INDRA source prior (bundled)" at 28
 * characters (151.7 units), leaving 32 units of slack. `buildPaperPerEvidence`
 * FAILS if any lane name exceeds this, so a renamed arm gates the figure instead
 * of quietly eating its first glyphs.
 */
export const PAPER_PER_EVIDENCE_LABEL_BUDGET_CHARS = 33;

/**
 * READOUT BUDGET: (920 − 790) units ÷ 4.8165 u/char at 8px = 26.9 → 26
 * characters. Longest shipped readout is the two-grain line, "0.850 ev · 0.898
 * stmt" at 21 characters. Left-anchored, so an overrun clips the TRAILING glyphs;
 * budgeted and enforced all the same.
 */
export const PAPER_PER_EVIDENCE_READOUT_BUDGET_CHARS = 26;

/**
 * The chance rule's annotation is free-floating 8px text, so its fit is measured
 * against the plot's right edge and it flips sides rather than clip. Same device
 * as `PaperOwnMetric.svelte`'s reference label.
 */
export const PAPER_PER_EVIDENCE_CHANCE_LABEL_PAD = 6;

export type PaperPerEvidenceSeriesId = 'evidence-reader' | 'evidence-baseline' | 'statement';

export const PAPER_PER_EVIDENCE_SERIES_IDS: readonly PaperPerEvidenceSeriesId[] = [
	'evidence-reader',
	'evidence-baseline',
	'statement'
] as const;

export interface PaperPerEvidenceSeriesStyle {
	id: PaperPerEvidenceSeriesId;
	/** CSS custom property name, never a raw hex. */
	strokeVar: string;
	/** SVG stroke-dasharray; '' = solid. */
	dash: string;
	strokeWidth: number;
	shape: 'diamond' | 'bracket' | 'open-circle';
	legend: string;
}

/**
 * Each series carries its OWN (stroke token, dash, mark shape), so the figure
 * survives greyscale and colour-vision deficiency and no two series share a hue.
 * Every stroke clears 3:1 against --paper #fdfcf8 (WCAG 1.4.11): --accent
 * #7d2a1a = 9.2:1, --blocked #6f5a16 = 6.5:1, --ink-muted #6a6a6a = 5.3:1.
 *
 * The legends are TERSE on purpose. /paper counts visible words and this page is
 * at its budget; each series is described in full in the figure's `<desc>` (which
 * is not counted and is what a screen reader receives) and again in the
 * `<details>` beneath it, so the visible legend only has to name the series.
 */
export const PAPER_PER_EVIDENCE_SERIES: Record<
	PaperPerEvidenceSeriesId,
	PaperPerEvidenceSeriesStyle
> = {
	'evidence-reader': {
		id: 'evidence-reader',
		strokeVar: 'var(--accent)',
		dash: '',
		strokeWidth: 2.4,
		shape: 'diamond',
		legend: 'per evidence · reader'
	},
	'evidence-baseline': {
		id: 'evidence-baseline',
		strokeVar: 'var(--blocked)',
		dash: '4 2',
		strokeWidth: 1.6,
		shape: 'bracket',
		legend: 'per evidence · INDRA source prior'
	},
	statement: {
		id: 'statement',
		strokeVar: 'var(--ink-muted)',
		dash: '1.5 1.5',
		strokeWidth: 1.2,
		shape: 'open-circle',
		legend: 'same model · statement grain'
	}
};

/** AUROC at chance. Drawn as a rule because a constant-within-source score that
 *  is the SAME constant for every reader source can do no better. */
export const PAPER_PER_EVIDENCE_CHANCE = 0.5;

export interface PaperPerEvidenceInterval {
	value: number;
	low: number;
	high: number;
}

export interface PaperPerEvidenceSourceTick {
	source: string;
	auroc: number;
	reviewedPairs: number;
}

export interface PaperPerEvidenceLane {
	id: string;
	/** FROZEN artifact join key. Never rendered. */
	joinKey: string;
	/** On-screen name. Rendered. */
	display: string;
	kind: 'reader' | 'baseline';
	/** Provenance sentence. Non-empty by construction; gated, never defaulted. */
	attribution: string;
	/** `attribution` with its plain restatement — `shipped` is byte-identical. */
	attributionProse: ShippedProse;
	y: number;
	evidenceY: number;
	statementY: number;
	/** Per-evidence AUROC and its shipped bootstrap interval. */
	evidence: PaperPerEvidenceInterval;
	/** The same model's statement-grain AUROC. No interval: none is shipped. */
	statementAuroc: number;
	/** statement − evidence. Signed, and never described as an increment. */
	grainShift: number;
	/**
	 * PAIRED delta vs the reference arm, per-evidence AUROC, from the artifact's
	 * own bootstrap. Null for the reference itself. This is the figure's central
	 * inferential quantity — it shipped but was not being read, so the plate drew
	 * point estimates with no evidence they were separable.
	 */
	pairedDelta: { delta: number; ciLow: number; ciHigh: number; standing: Standing } | null;
	averagePrecisionIncorrect: number;
	errorF1: number;
	distinctScores: number;
	/** True where the score cannot vary within a source — the baselines' defect. */
	constantWithinSource: boolean;
	perSource: PaperPerEvidenceSourceTick[];
	readout: string;
	title: string;
}

export interface PaperPerEvidenceSourceRow {
	source: string;
	reviewedPairs: number;
	positivePairs: number;
	negativePairs: number;
	observedCorrectFraction: number;
	bundledPriorAtOneEvidence: number;
	metricRow: boolean;
	y: number;
}

export interface PaperPerEvidenceSharedPrior {
	sharedPrior: number;
	sources: string[];
	observedMin: number;
	observedMax: number;
	chi2: number;
	dof: number;
	pValue: number;
}

export interface PaperPerEvidenceCoverage {
	executedUniquePairs: number;
	reviewedPairs: number;
	unreviewedPairs: number;
	/** Reviewed pairs with no reader verdict. Shipped so a hole is visible. */
	unscoredPairs: number;
	tierCensus: { tier: string; pairs: number }[];
	excludedBaselines: {
		family: string;
		/** `family` with its plain restatement — `shipped` is byte-identical. */
		familyProse: ShippedProse;
		reason: string;
		/** `reason` with its plain restatement — `shipped` is byte-identical. */
		reasonProse: ShippedProse;
	}[];
}

export interface PaperPerEvidenceReaggregation {
	verified: boolean;
	nStatements: number;
	nExact: number;
	maxAbsDiff: number;
	aggregation: string;
}

export interface PaperPerEvidenceContamination {
	grain: string;
	/** `grain` with its plain restatement — `grainProse.shipped === grain`. */
	grainProse: ShippedProse;
	demonstrationSentences: number;
	reviewedPairsScanned: number;
	overlappingPairs: number;
	overlappingPairsSameClaim: number;
	maxAurocAbsShift: number | null;
	pairsKept: number | null;
	/** True when this grain finds leakage the (agent set, type) check does not. */
	disagreesWithAgentGrainCheck: boolean;
}

export interface PaperPerEvidenceFigure {
	lanes: PaperPerEvidenceLane[];
	sourceRows: PaperPerEvidenceSourceRow[];
	sharedPrior: PaperPerEvidenceSharedPrior | null;
	domainMin: number;
	domainMax: number;
	ticks: number[];
	height: number;
	readerBandY: number;
	baselineBandY: number;
	sourceBandY: number;
	/** Baseline of the AUROC register's own axis rule and tick row. */
	discriminationAxisY: number;
	/** Baseline of the probability register's own axis rule and tick row. */
	probabilityAxisY: number;
	chanceLabel: string;
	chanceLabelFits: boolean;
	referenceId: string;
	referenceDisplay: string;
	nReviewedPairs: number;
	nPositive: number;
	nNegative: number;
	negativeFraction: number;
	nStatements: number;
	statementPositiveRate: number;
	/** Evidence pairs per statement-panel item — the raw power ratio, uncorrected. */
	powerRatio: number;
	powerNote: string;
	/** `powerNote` with its plain restatement — `shipped` is byte-identical. */
	powerNoteProse: ShippedProse;
	twoGrainNote: string;
	/** `twoGrainNote` with its plain restatement — `shipped` is byte-identical. */
	twoGrainNoteProse: ShippedProse;
	coverage: PaperPerEvidenceCoverage;
	reaggregation: PaperPerEvidenceReaggregation;
	contamination: PaperPerEvidenceContamination;
	curators: { curator: string; reviewedPairs: number }[];
	seed: number;
	nBootstrap: number;
	bootstrapDesign: string;
	/** `bootstrapDesign` with its plain restatement — `shipped` is byte-identical. */
	bootstrapDesignProse: ShippedProse;
	estimatorNote: string;
	/** `estimatorNote` with its plain restatement — `shipped` is byte-identical. */
	estimatorNoteProse: ShippedProse;
	decisionThreshold: number;
	/**
	 * The plain half of every string this figure took off the artifact, in one
	 * place. The per-model sentences are the SAME objects the rows carry.
	 */
	prose: PaperPerEvidenceProse;
}

/** Every shipped sentence this figure carries, each with its restatement. */
export interface PaperPerEvidenceProse {
	bootstrapDesign: ShippedProse;
	/** What the contamination scan matched on. */
	contaminationGrain: ShippedProse;
	/** What the extra items buy, and what the interval does NOT model. */
	powerNote: ShippedProse;
	/** That the two marks are two populations, not a step up. */
	twoGrainNote: ShippedProse;
	/** Why no trapezoidal number is quoted here at all. */
	estimatorNote: ShippedProse;
	/** Index-aligned with `lanes`, pinned to each model's own sentence. */
	attributions: ShippedProse[];
	/** Index-aligned with `coverage.excludedBaselines`. */
	excludedBaselineFamilies: ShippedProse[];
	excludedBaselineReasons: ShippedProse[];
}

export interface PaperPerEvidenceOk {
	status: 'ok';
	figure: PaperPerEvidenceFigure;
	reason: null;
	artifact_path: string;
	/**
	 * NULLABLE on the ok branch too: the server always supplies a digest, the pure
	 * client-side validate path does not, and coalescing to '' would print an
	 * empty provenance line that reads as a real, empty sha.
	 */
	artifact_sha256: string | null;
}

export interface PaperPerEvidenceUnavailable {
	status: 'unavailable';
	figure: null;
	reason: string;
	artifact_path: string;
	artifact_sha256: string | null;
}

export type PaperPerEvidenceLoad = PaperPerEvidenceOk | PaperPerEvidenceUnavailable;

export interface PaperPerEvidenceContext {
	artifactPath?: string;
	artifactSha256?: string;
}

type UnknownRecord = Record<string, unknown>;



function array(value: unknown, context: string): unknown[] {
	if (!Array.isArray(value)) fail(context, 'expected an array');
	return value;
}







/** Exactly `want`, or the figure gates. Used where a flipped flag would present
 *  an unverified reconciliation as a verified one. */
function exactly(value: unknown, want: boolean, context: string): boolean {
	const parsed = boolean(value, context);
	if (parsed !== want) fail(context, `expected ${want}`);
	return parsed;
}

/** Three decimals: the precision this page prints AUROC and AP at. */
export function fmt3(value: number): string {
	return value.toFixed(3);
}

/**
 * Does a right-anchored lane name fit its gutter? MEASURED in user units against
 * the measured mono advance, and exported so the contract test exercises the same
 * function the builder calls rather than a re-typed copy of the same arithmetic.
 */
export function laneNameFits(value: string): boolean {
	const g = PAPER_PER_EVIDENCE_GEOMETRY;
	return value.length * g.monoUnitsPerChar <= g.labelAnchorX;
}

/** The same, for the left-anchored readout gutter at 8px. */
export function readoutFits(value: string): boolean {
	const g = PAPER_PER_EVIDENCE_GEOMETRY;
	return value.length * g.readoutUnitsPerChar <= g.width - g.readoutX;
}

/**
 * Does the chance-rule annotation fit to the right of its rule? Measured against
 * the MEASURED 8px mono advance, not guessed, and exported so the contract test
 * exercises the same function the component calls.
 */
export function chanceLabelFits(label: string, chanceX: number): boolean {
	const g = PAPER_PER_EVIDENCE_GEOMETRY;
	const width = label.length * g.readoutUnitsPerChar;
	return chanceX + PAPER_PER_EVIDENCE_CHANCE_LABEL_PAD + width <= g.plotRight;
}

function parseInterval(raw: unknown, value: number, context: string): PaperPerEvidenceInterval {
	const block = record(raw, context);
	const low = unit(block.ci95_low, `${context}.ci95_low`);
	const high = unit(block.ci95_high, `${context}.ci95_high`);
	if (low > high) fail(context, 'ci95_low must not exceed ci95_high');
	// The drawn interval is a whisker THROUGH its point mark. A bootstrap interval
	// that does not contain its own point estimate would draw a mark floating off
	// its own whisker, so it gates rather than renders.
	if (value < low || value > high) {
		fail(context, `point estimate ${value} lies outside its own interval [${low}, ${high}]`);
	}
	return { value, low, high };
}

/**
 * Discrimination-axis domain, snapped outward to the 0.05 grid over every value
 * the figure DRAWS — interval ends, both grains' marks, and the chance rule — with
 * one extra twentieth of slack on each side so no mark ever lands exactly on the
 * axis end (the chance rule sits at 0.5, which is a grid line, and without the
 * slack it would be drawn on top of the axis). Integer twentieths throughout:
 * float accumulation would drift the ticks.
 */
function onGrid(value: number): boolean {
	return Math.abs(value * 20 - Math.round(value * 20)) < 1e-9;
}

function domainOf(extents: number[]): { min: number; max: number; ticks: number[] } {
	let lo = Number.POSITIVE_INFINITY;
	let hi = Number.NEGATIVE_INFINITY;
	for (const value of extents) {
		lo = Math.min(lo, value);
		hi = Math.max(hi, value);
	}
	if (!Number.isFinite(lo) || !Number.isFinite(hi)) fail('domain', 'no marks to scale');
	// Slack is spent only where an extreme sits exactly ON a grid line, which is
	// the case a mark would be drawn on top of the axis end. Spending it
	// unconditionally would leave a whole empty tick of axis at the top.
	const minTwentieths = Math.max(0, Math.floor(lo * 20) - (onGrid(lo) ? 1 : 0));
	const maxTwentieths = Math.min(20, Math.ceil(hi * 20) + (onGrid(hi) ? 1 : 0));
	if (maxTwentieths - minTwentieths < 2) fail('domain', 'degenerate axis range');
	const ticks: number[] = [];
	for (let t = minTwentieths; t <= maxTwentieths; t += 1) ticks.push(t / 20);
	return { min: minTwentieths / 20, max: maxTwentieths / 20, ticks };
}

/**
 * The per-source register is a SECOND axis on the same plate, and it must be: it
 * carries P(correct at one evidence), not AUROC. Drawing a probability on a
 * discrimination axis would be a category error, so this register keeps its own
 * fixed [0, 1] domain and its own tick row, and the component scales it
 * separately.
 */
export const PAPER_PER_EVIDENCE_SOURCE_DOMAIN = { min: 0, max: 1 } as const;
export const PAPER_PER_EVIDENCE_SOURCE_TICKS = [0, 0.25, 0.5, 0.75, 1] as const;

/**
 * Validate the shipped artifact into figure geometry, or gate.
 *
 * Fail-closed by construction: every field is parsed, every cross-field identity
 * the drawing depends on is asserted, and a missing or drifted field throws so
 * the caller can render `unavailable`. There is no 0-placeholder anywhere in this
 * file — a zero here would render as a measurement.
 */
export function validatePaperPerEvidence(
	raw: unknown,
	context: PaperPerEvidenceContext = {}
): PaperPerEvidenceLoad {
	const artifactPath = context.artifactPath ?? 'per_evidence_comparison.json';
	const artifactSha256 = context.artifactSha256 ?? null;
	try {
		return {
			status: 'ok',
			figure: buildPaperPerEvidence(raw),
			reason: null,
			artifact_path: artifactPath,
			artifact_sha256: artifactSha256
		};
	} catch (error) {
		return {
			status: 'unavailable',
			figure: null,
			reason: String(error instanceof Error ? error.message : error),
			artifact_path: artifactPath,
			artifact_sha256: artifactSha256
		};
	}
}

export function buildPaperPerEvidence(raw: unknown): PaperPerEvidenceFigure {
	const root = record(raw, 'artifact');
	if (root.artifact_kind !== PAPER_PER_EVIDENCE_ARTIFACT_KIND) {
		fail('artifact.artifact_kind', `expected ${PAPER_PER_EVIDENCE_ARTIFACT_KIND}`);
	}

	const panel = record(root.panel, 'panel');
	const nReviewed = positiveInteger(panel.n_reviewed_pairs, 'panel.n_reviewed_pairs');
	const nPositive = positiveInteger(panel.n_positive, 'panel.n_positive');
	const nNegative = positiveInteger(panel.n_negative, 'panel.n_negative');
	if (nPositive + nNegative !== nReviewed) {
		fail('panel', 'n_positive + n_negative must equal n_reviewed_pairs');
	}
	const nStatements = positiveInteger(panel.n_statements, 'panel.n_statements');
	const nCovered = positiveInteger(
		panel.n_statements_with_reviewed_pair,
		'panel.n_statements_with_reviewed_pair'
	);
	if (nCovered > nStatements) fail('panel', 'more covered statements than statements');

	const statementGrain = record(root.statement_grain, 'statement_grain');
	const statementPositiveRate = unit(statementGrain.positive_rate, 'statement_grain.positive_rate');
	const twoGrainNote = text(statementGrain.note, 'statement_grain.note');
	const twoGrainNoteProse: ShippedProse = {
		shipped: twoGrainNote,
		plain: PER_EVIDENCE_PLAIN.twoGrainNote
	};

	const power = record(root.power, 'power');
	const powerRatio = number(power.ratio, 'power.ratio');
	const powerNote = text(power.note, 'power.note');
	const powerNoteProse: ShippedProse = { shipped: powerNote, plain: PER_EVIDENCE_PLAIN.powerNote };

	// The grain bridge is the figure's premise: if the recovered per-evidence
	// verdicts do not rebuild the shipped statement probabilities exactly, the two
	// marks on a lane are not the same run and the connector is a lie.
	const reaggRoot = record(root.reaggregation, 'reaggregation');
	exactly(reaggRoot.verified, true, 'reaggregation.verified');
	const reaggArms = record(reaggRoot.arms, 'reaggregation.arms');
	const reaggEntries = Object.entries(reaggArms);
	if (reaggEntries.length === 0) fail('reaggregation.arms', 'no arm was reconciled');
	let reaggExact = 0;
	let reaggTotal = 0;
	let reaggWorst = 0;
	let reaggAggregation = '';
	for (const [armId, value] of reaggEntries) {
		const block = record(value, `reaggregation.arms[${armId}]`);
		const total = positiveInteger(block.n_statements, `reaggregation.arms[${armId}].n_statements`);
		const exact = nonNegativeInteger(block.n_exact, `reaggregation.arms[${armId}].n_exact`);
		const worst = number(block.max_abs_diff, `reaggregation.arms[${armId}].max_abs_diff`);
		if (exact !== total) {
			fail(
				`reaggregation.arms[${armId}]`,
				`${exact}/${total} statements reproduce the shipped probability; the two grains are not the same run`
			);
		}
		if (worst !== 0) {
			fail(`reaggregation.arms[${armId}]`, `max_abs_diff must be exactly 0, got ${worst}`);
		}
		reaggExact += exact;
		reaggTotal += total;
		reaggWorst = Math.max(reaggWorst, Math.abs(worst));
		reaggAggregation = text(block.aggregation, `reaggregation.arms[${armId}].aggregation`);
	}

	// ---- per-source census --------------------------------------------------
	const coverage = record(root.coverage, 'coverage');
	const sourceRaw = array(coverage.sources, 'coverage.sources');
	if (sourceRaw.length === 0) fail('coverage.sources', 'no source census');
	const sourceCensus = sourceRaw.map((value, index) => {
		const block = record(value, `coverage.sources[${index}]`);
		const reviewed = positiveInteger(
			block.reviewed_pairs,
			`coverage.sources[${index}].reviewed_pairs`
		);
		const positive = nonNegativeInteger(
			block.positive_pairs,
			`coverage.sources[${index}].positive_pairs`
		);
		const negative = nonNegativeInteger(
			block.negative_pairs,
			`coverage.sources[${index}].negative_pairs`
		);
		if (positive + negative !== reviewed) {
			fail(`coverage.sources[${index}]`, 'positive + negative must equal reviewed_pairs');
		}
		return {
			source: text(block.source, `coverage.sources[${index}].source`),
			reviewedPairs: reviewed,
			positivePairs: positive,
			negativePairs: negative,
			observedCorrectFraction: unit(
				block.observed_correct_fraction,
				`coverage.sources[${index}].observed_correct_fraction`
			),
			bundledPriorAtOneEvidence: unit(
				block.bundled_prior_at_one_evidence,
				`coverage.sources[${index}].bundled_prior_at_one_evidence`
			),
			metricRow: boolean(block.metric_row, `coverage.sources[${index}].metric_row`),
			y: 0
		};
	});
	const totalReviewedBySource = sourceCensus.reduce((sum, row) => sum + row.reviewedPairs, 0);
	if (totalReviewedBySource !== nReviewed) {
		fail('coverage.sources', 'the source census does not sum to the reviewed panel');
	}

	// ---- arms ---------------------------------------------------------------
	const armsRaw = array(root.arms, 'arms');
	const referenceId = text(root.reference_arm_id, 'reference_arm_id');
	// The paired bootstrap the artifact already ships. Read, never recomputed.
	// Without this the plate drew point estimates with no evidence of separation.
	const pairedRaw = record(
		root.paired_delta_vs_reference,
		'paired_delta_vs_reference'
	) as Record<string, unknown>;
	const geometry = PAPER_PER_EVIDENCE_GEOMETRY;
	const extents: number[] = [PAPER_PER_EVIDENCE_CHANCE];
	interface Parsed {
		lane: Omit<PaperPerEvidenceLane, 'y' | 'evidenceY' | 'statementY'>;
	}
	const parsed: Parsed[] = [];
	for (const [index, value] of armsRaw.entries()) {
		const block = record(value, `arms[${index}]`);
		const id = text(block.id, `arms[${index}].id`);
		// An arm the compute script could not produce ships without metrics; it is
		// dropped from the drawing but never silently zeroed.
		if (block.metrics === undefined) continue;
		const kindRaw = text(block.kind, `arms[${id}].kind`);
		if (kindRaw !== 'reader' && kindRaw !== 'baseline') {
			fail(`arms[${id}].kind`, 'expected "reader" or "baseline"');
		}
		const display = text(block.display, `arms[${id}].display`);
		const canonical = PAPER_PER_EVIDENCE_DISPLAY[id];
		const artifactExpected = PAPER_PER_EVIDENCE_ARTIFACT_DISPLAY[id];
		if (canonical === undefined || artifactExpected === undefined) {
			fail(`arms[${id}]`, 'is not a known arm id');
		}
		// The gutter budget guards THIS module's own display table: the drawn name
		// is the canonical one, so a longer name added here — not there — is what
		// would clip its leading glyphs in silence.
		// The kit's argument order, with this figure's own predicate passed last.
		// Two `budget` signatures existed for a while — the kit's
		// (value, chars, context, fits?) and a local (value, fits, chars, context)
		// — which is precisely the transposition a reader cannot see and tsc can.
		budget(
			canonical,
			PAPER_PER_EVIDENCE_LABEL_BUDGET_CHARS,
			`PAPER_PER_EVIDENCE_DISPLAY[${id}]`,
			laneNameFits
		);
		// The drift guard, pointed at the artifact's OWN expected name rather than
		// at the on-screen one: renaming a lane in English must not gate the
		// figure, and a reissued artifact that renames an arm still must.
		if (display !== artifactExpected) {
			fail(
				`arms[${id}].display`,
				`artifact says "${display}", this module expects "${artifactExpected}"`
			);
		}
		// The misattribution ban, enforced as a gate rather than a caption. Looked up
		// by the FROZEN arm id and then pinned to its own text: four models ship the
		// SAME sentence, so the key alone cannot bind the right restatement to the
		// right row, and a model with none authored gates the figure.
		const attribution = text(block.attribution, `arms[${id}].attribution`);
		const attributionProse = keyedShippedProse(
			id,
			attribution,
			PER_EVIDENCE_ATTRIBUTION_TWINS,
			`arms[${id}].attribution`
		);

		const metrics = record(block.metrics, `arms[${id}].metrics`);
		const nPairs = positiveInteger(metrics.n, `arms[${id}].metrics.n`);
		if (nPairs !== nReviewed) {
			fail(`arms[${id}].metrics.n`, `scored ${nPairs} of ${nReviewed} reviewed pairs`);
		}
		const auroc = unit(metrics.auroc, `arms[${id}].metrics.auroc`);
		const intervals = record(metrics.interval, `arms[${id}].metrics.interval`);
		const evidence = parseInterval(intervals.auroc, auroc, `arms[${id}].metrics.interval.auroc`);
		const errorDetection = record(metrics.error_detection, `arms[${id}].metrics.error_detection`);

		const statement = record(block.statement_grain, `arms[${id}].statement_grain`);
		const statementAuroc = unit(statement.auroc, `arms[${id}].statement_grain.auroc`);
		const statementN = positiveInteger(
			statement.n_statements,
			`arms[${id}].statement_grain.n_statements`
		);
		if (statementN !== nStatements) {
			fail(`arms[${id}].statement_grain.n_statements`, 'disagrees with the statement panel');
		}

		const perSourceRaw = record(block.per_source, `arms[${id}].per_source`);
		const perSource: PaperPerEvidenceSourceTick[] = [];
		for (const row of sourceCensus) {
			if (!row.metricRow) continue;
			const entry = perSourceRaw[row.source];
			if (entry === undefined) fail(`arms[${id}].per_source`, `missing stratum ${row.source}`);
			const stratum = record(entry, `arms[${id}].per_source[${row.source}]`);
			const stratumN = positiveInteger(stratum.n, `arms[${id}].per_source[${row.source}].n`);
			if (stratumN !== row.reviewedPairs) {
				fail(
					`arms[${id}].per_source[${row.source}].n`,
					`${stratumN} pairs, but the census says ${row.reviewedPairs}`
				);
			}
			perSource.push({
				source: row.source,
				auroc: unit(stratum.auroc, `arms[${id}].per_source[${row.source}].auroc`),
				reviewedPairs: stratumN
			});
		}

		const readout = budget(
			`${fmt3(auroc)} ev · ${fmt3(statementAuroc)} stmt`,
			PAPER_PER_EVIDENCE_READOUT_BUDGET_CHARS,
			`arms[${id}].readout`,
			readoutFits
		);

		extents.push(evidence.low, evidence.high, auroc, statementAuroc);
		parsed.push({
			lane: {
				id,
				joinKey: text(block.join_key, `arms[${id}].join_key`),
				// The canonical on-screen name, not the artifact's. The artifact's is
				// checked above and never drawn.
				display: canonical,
				kind: kindRaw,
				attribution,
				attributionProse,
				evidence,
				statementAuroc,
				grainShift: statementAuroc - auroc,
				pairedDelta: ((): PaperPerEvidenceLane['pairedDelta'] => {
					if (id === referenceId) return null;
					const entry = pairedRaw[id];
					if (entry === undefined) {
						fail(`paired_delta_vs_reference[${id}]`, 'no paired delta for a non-reference arm');
					}
					const pooled = record(
						(entry as Record<string, unknown>).pooled,
						`paired_delta_vs_reference[${id}].pooled`
					);
					const a = record(pooled.auroc, `paired_delta_vs_reference[${id}].pooled.auroc`);
					const ctx = `paired_delta_vs_reference[${id}].pooled.auroc`;
					const delta = number(a.mean, `${ctx}.mean`);
					const ciLow = number(a.ci95_low, `${ctx}.ci95_low`);
					const ciHigh = number(a.ci95_high, `${ctx}.ci95_high`);
					if (ciLow > ciHigh) fail(ctx, 'ci95_low exceeds ci95_high');
					if (delta < ciLow || delta > ciHigh) {
						fail(ctx, 'the point estimate lies outside its own interval');
					}
					// The shipped flag is still read and still gated against its own
					// endpoints — a disagreement is a corrupt artifact and must gate. It
					// does not leave this parser: what travels is the three-way class.
					if (
						boolean(a.excludes_zero, `${ctx}.excludes_zero`) !==
						(ciLow > 0 || ciHigh < 0)
					) {
						fail(ctx, 'excludes_zero disagrees with its own endpoints');
					}
					return { delta, ciLow, ciHigh, standing: standingOfBounds(ciLow, ciHigh) };
				})(),
				averagePrecisionIncorrect: unit(
					metrics.average_precision_incorrect,
					`arms[${id}].metrics.average_precision_incorrect`
				),
				errorF1: unit(errorDetection.f1, `arms[${id}].metrics.error_detection.f1`),
				distinctScores: positiveInteger(
					metrics.distinct_scores,
					`arms[${id}].metrics.distinct_scores`
				),
				constantWithinSource: boolean(
					block.constant_within_source,
					`arms[${id}].constant_within_source`
				),
				perSource,
				readout,
				// `attributionProse.plain`, never `attribution`: this string is the row's
				// accessible title, so it is a READER surface. Built from the shipped
				// bytes it carried "out-of-fold", "arm" and "panel" onto the screen —
				// the exact defect the twins exist to close, one layer further out.
				title:
					`${canonical} — per evidence AUROC ${fmt3(auroc)} ` +
					`[${fmt3(evidence.low)}, ${fmt3(evidence.high)}] over ${nReviewed} reviewed pairs; ` +
					`the same model at statement grain scores ${fmt3(statementAuroc)} over ` +
					`${statementN} statements. ${attributionProse.plain}`
			}
		});
	}
	if (parsed.length === 0) fail('arms', 'no arm carries metrics');
	if (!parsed.some((entry) => entry.lane.id === referenceId)) {
		fail('reference_arm_id', `"${referenceId}" is not among the drawn arms`);
	}

	// Readers first, then baselines; strongest per-evidence AUROC at the top of
	// each band. Ties break on id so the order is stable across reruns.
	const byBand = (kind: 'reader' | 'baseline') =>
		parsed
			.filter((entry) => entry.lane.kind === kind)
			.sort(
				(a, b) => b.lane.evidence.value - a.lane.evidence.value || a.lane.id.localeCompare(b.lane.id)
			);
	const ordered = [...byBand('reader'), ...byBand('baseline')];
	if (byBand('reader').length === 0) fail('arms', 'no reader arm to draw');
	if (byBand('baseline').length === 0) fail('arms', 'no baseline arm to draw');

	let y = geometry.topPad;
	const readerBandY = y;
	y += geometry.bandHeaderHeight;
	const lanes: PaperPerEvidenceLane[] = [];
	let baselineBandY = 0;
	let placedBaselineHeader = false;
	for (const entry of ordered) {
		if (entry.lane.kind === 'baseline' && !placedBaselineHeader) {
			baselineBandY = y;
			y += geometry.bandHeaderHeight;
			placedBaselineHeader = true;
		}
		const centre = y + geometry.laneHeight / 2;
		lanes.push({
			...entry.lane,
			y: centre,
			evidenceY: centre + geometry.evidenceOffset,
			statementY: centre + geometry.statementOffset
		});
		y += geometry.laneHeight;
	}

	// The AUROC register closes here, with its own rule and ticks. Everything below
	// is on a different axis and the grid must not run through it.
	const discriminationAxisY = y + 6;
	y += geometry.axisRowHeight;

	const sourceBandY = y;
	y += geometry.sourceHeaderHeight;
	// NOTE: source values are deliberately NOT pushed into `extents`. They live on
	// the register's own probability axis; folding them into the discrimination
	// domain would silently scale two different quantities together.
	const sourceRows = sourceCensus.map((row) => {
		const placed = { ...row, y: y + geometry.sourceRowHeight / 2 };
		y += geometry.sourceRowHeight;
		return placed;
	});
	const probabilityAxisY = y + 6;
	y += geometry.axisRowHeight;

	// ---- the shared-prior block ---------------------------------------------
	const defect = record(root.shared_prior_defect, 'shared_prior_defect');
	const blocks = array(defect.blocks, 'shared_prior_defect.blocks');
	let sharedPrior: PaperPerEvidenceSharedPrior | null = null;
	if (blocks.length > 0) {
		const block = record(blocks[0], 'shared_prior_defect.blocks[0]');
		const sources = array(block.sources, 'shared_prior_defect.blocks[0].sources').map(
			(value, index) =>
				text(
					record(value, `shared_prior_defect.blocks[0].sources[${index}]`).source,
					`shared_prior_defect.blocks[0].sources[${index}].source`
				)
		);
		if (sources.length < 2) {
			fail('shared_prior_defect.blocks[0]', 'a shared prior needs at least two sources');
		}
		sharedPrior = {
			sharedPrior: unit(
				block.shared_prior_at_one_evidence,
				'shared_prior_defect.blocks[0].shared_prior_at_one_evidence'
			),
			sources,
			observedMin: unit(
				block.observed_correct_fraction_min,
				'shared_prior_defect.blocks[0].observed_correct_fraction_min'
			),
			observedMax: unit(
				block.observed_correct_fraction_max,
				'shared_prior_defect.blocks[0].observed_correct_fraction_max'
			),
			chi2: number(block.chi2, 'shared_prior_defect.blocks[0].chi2'),
			dof: positiveInteger(block.dof, 'shared_prior_defect.blocks[0].dof'),
			pValue: unit(block.p_value, 'shared_prior_defect.blocks[0].p_value')
		};
		if (sharedPrior.observedMin > sharedPrior.observedMax) {
			fail('shared_prior_defect.blocks[0]', 'observed min exceeds observed max');
		}
	}

	// ---- coverage census ----------------------------------------------------
	const perArm = record(coverage.per_arm, 'coverage.per_arm');
	let unscored = 0;
	const tierTally = new Map<string, number>();
	for (const [armId, value] of Object.entries(perArm)) {
		const block = record(value, `coverage.per_arm[${armId}]`);
		unscored = Math.max(
			unscored,
			nonNegativeInteger(block.reviewed_pairs_unscored, `coverage.per_arm[${armId}].reviewed_pairs_unscored`)
		);
		exactly(
			block.raw_attempts_sha256_matches_manifest,
			true,
			`coverage.per_arm[${armId}].raw_attempts_sha256_matches_manifest`
		);
		const tiers = record(block.tier_census_reviewed, `coverage.per_arm[${armId}].tier_census_reviewed`);
		for (const [tier, count] of Object.entries(tiers)) {
			const parsedCount = nonNegativeInteger(count, `coverage.per_arm[${armId}].tier_census_reviewed[${tier}]`);
			tierTally.set(tier, Math.max(tierTally.get(tier) ?? 0, parsedCount));
		}
	}
	const tierCensus = [...tierTally.entries()]
		.map(([tier, pairs]) => ({ tier, pairs }))
		.sort((a, b) => b.pairs - a.pairs || a.tier.localeCompare(b.tier));
	const excludedBaselines = array(coverage.excluded_baselines, 'coverage.excluded_baselines').map(
		(value, index) => {
			const block = record(value, `coverage.excluded_baselines[${index}]`);
			return {
				family: text(block.family, `coverage.excluded_baselines[${index}].family`),
				familyProse: {
					shipped: text(block.family, `coverage.excluded_baselines[${index}].family`),
					plain: PER_EVIDENCE_PLAIN.excludedBaselineFamily
				},
				reason: text(block.reason, `coverage.excluded_baselines[${index}].reason`),
				reasonProse: {
					shipped: text(block.reason, `coverage.excluded_baselines[${index}].reason`),
					plain: PER_EVIDENCE_PLAIN.excludedBaselineReason
				}
			};
		}
	);
	if (excludedBaselines.length === 0) {
		fail(
			'coverage.excluded_baselines',
			'the paper supervised rows cannot be drawn at this grain and the exclusion must ship as data'
		);
	}

	// ---- contamination ------------------------------------------------------
	const contaminationRaw = record(root.contamination, 'contamination');
	const sensitivity =
		contaminationRaw.sensitivity === undefined
			? null
			: record(contaminationRaw.sensitivity, 'contamination.sensitivity');
	const contamination: PaperPerEvidenceContamination = {
		grain: text(contaminationRaw.grain, 'contamination.grain'),
		grainProse: {
			shipped: text(contaminationRaw.grain, 'contamination.grain'),
			plain: PER_EVIDENCE_PLAIN.contaminationGrain
		},
		demonstrationSentences: positiveInteger(
			contaminationRaw.n_demonstration_sentences,
			'contamination.n_demonstration_sentences'
		),
		reviewedPairsScanned: positiveInteger(
			contaminationRaw.n_reviewed_pairs_scanned,
			'contamination.n_reviewed_pairs_scanned'
		),
		overlappingPairs: nonNegativeInteger(
			contaminationRaw.n_overlapping_pairs,
			'contamination.n_overlapping_pairs'
		),
		overlappingPairsSameClaim: nonNegativeInteger(
			contaminationRaw.n_overlapping_pairs_same_claim,
			'contamination.n_overlapping_pairs_same_claim'
		),
		maxAurocAbsShift: sensitivity
			? number(sensitivity.max_auroc_abs_shift, 'contamination.sensitivity.max_auroc_abs_shift')
			: null,
		pairsKept: sensitivity
			? positiveInteger(sensitivity.n_pairs_kept, 'contamination.sensitivity.n_pairs_kept')
			: null,
		disagreesWithAgentGrainCheck: boolean(
			contaminationRaw.disagrees_with_existing_agent_grain_check,
			'contamination.disagrees_with_existing_agent_grain_check'
		)
	};
	if (contamination.overlappingPairsSameClaim > contamination.overlappingPairs) {
		fail('contamination', 'same-claim overlaps cannot exceed total overlaps');
	}
	if (contamination.overlappingPairs > 0 && sensitivity === null) {
		fail('contamination', 'overlapping pairs are reported with no sensitivity beside them');
	}

	const curators = array(panel.curators, 'panel.curators').map((value, index) => {
		const block = record(value, `panel.curators[${index}]`);
		return {
			curator: text(block.curator, `panel.curators[${index}].curator`),
			reviewedPairs: positiveInteger(
				block.reviewed_pairs,
				`panel.curators[${index}].reviewed_pairs`
			)
		};
	});

	const scale = domainOf(extents);
	const chanceX =
		geometry.plotLeft +
		((PAPER_PER_EVIDENCE_CHANCE - scale.min) / (scale.max - scale.min)) *
			(geometry.plotRight - geometry.plotLeft);
	const chanceLabel = 'chance';
	const bootstrapDesignProse: ShippedProse = {
		shipped: text(root.bootstrap_design, 'bootstrap_design'),
		plain: PER_EVIDENCE_PLAIN.bootstrapDesign
	};
	const estimatorNoteProse: ShippedProse = {
		shipped: text(
			record(root.estimator_contract, 'estimator_contract').trapezoidal_pr_auc,
			'estimator_contract.trapezoidal_pr_auc'
		),
		plain: PER_EVIDENCE_PLAIN.estimatorNote
	};

	return {
		lanes,
		sourceRows,
		sharedPrior,
		domainMin: scale.min,
		domainMax: scale.max,
		ticks: scale.ticks,
		height: y + geometry.axisPad,
		readerBandY,
		baselineBandY,
		sourceBandY,
		discriminationAxisY,
		probabilityAxisY,
		chanceLabel,
		chanceLabelFits: chanceLabelFits(chanceLabel, chanceX),
		referenceId,
		referenceDisplay: PAPER_PER_EVIDENCE_DISPLAY[referenceId] ?? referenceId,
		nReviewedPairs: nReviewed,
		nPositive,
		nNegative,
		negativeFraction: unit(panel.negative_fraction, 'panel.negative_fraction'),
		nStatements,
		statementPositiveRate,
		powerRatio,
		powerNote,
		powerNoteProse,
		twoGrainNote,
		twoGrainNoteProse,
		coverage: {
			executedUniquePairs: positiveInteger(
				coverage.executed_unique_pairs,
				'coverage.executed_unique_pairs'
			),
			reviewedPairs: positiveInteger(coverage.reviewed_pairs, 'coverage.reviewed_pairs'),
			unreviewedPairs: nonNegativeInteger(coverage.unreviewed_pairs, 'coverage.unreviewed_pairs'),
			unscoredPairs: unscored,
			tierCensus,
			excludedBaselines
		},
		reaggregation: {
			verified: true,
			nStatements: reaggTotal,
			nExact: reaggExact,
			maxAbsDiff: reaggWorst,
			aggregation: reaggAggregation
		},
		contamination,
		curators,
		seed: nonNegativeInteger(root.seed, 'seed'),
		nBootstrap: positiveInteger(root.n_bootstrap, 'n_bootstrap'),
		bootstrapDesign: bootstrapDesignProse.shipped,
		bootstrapDesignProse,
		estimatorNote: estimatorNoteProse.shipped,
		estimatorNoteProse,
		decisionThreshold: unit(root.decision_threshold, 'decision_threshold'),
		prose: {
			bootstrapDesign: bootstrapDesignProse,
			contaminationGrain: contamination.grainProse,
			powerNote: powerNoteProse,
			twoGrainNote: twoGrainNoteProse,
			estimatorNote: estimatorNoteProse,
			// The SAME objects the rows carry, in the figure's own order.
			attributions: lanes.map((lane) => lane.attributionProse),
			excludedBaselineFamilies: excludedBaselines.map((entry) => entry.familyProse),
			excludedBaselineReasons: excludedBaselines.map((entry) => entry.reasonProse)
		}
	};
}
