/**
 * Typed data contract for the REASONING ABLATION surface.
 *
 * WHAT THIS FIGURE ANSWERS. Every reading model on this page was run twice over
 * the identical 33,361 evidence readings: once with the provider's private
 * deliberation on AND a prompt that told the model to argue with itself before
 * answering, once with BOTH removed so the model emits a verdict and nothing
 * else. The two runs pair one-to-one on the same statement and the same evidence,
 * so the difference between them is the deliberation and not the corpus, the
 * priors, the rollup or the score cutoff.
 *
 * IT IS TWO MEASUREMENTS AND THEY AGREE — ONCE THE RIGHT ONE IS TAKEN:
 *
 *   · at the grain the model actually works at — one reading of one piece of
 *     evidence — removing deliberation moves a large, LOPSIDED number of readings,
 *     and every model moves the same way: toward accepting. The deliberation was
 *     mostly producing rejections.
 *   · at the grain the benchmark scores — the assembled statement — finding wrong
 *     statements gets measurably worse on the three larger models.
 *
 * THIS FIGURE WAS FIRST BUILT ON AUROC AND THE AUROC VERSION SAID THE OPPOSITE.
 * On that axis three of the four models barely moved and the only decided margin
 * was the smallest model's. That was not a second finding, it was the wrong
 * instrument: the benchmark is 73.2% correct, so a ranking measure spends almost
 * all of its range on statements nobody disputes. Measured on the error class the
 * three larger models are the decided ones and the smallest is not — the exact
 * reverse — and Gemma 4 26B's margin is roughly SEVEN TIMES larger than the AUROC
 * reading of the same change. The ranking numbers are kept, in the detail table,
 * because they are true; they are just not what this page leads on and never were.
 *
 * WHY THIS SURFACE CANNOT CARRY A BUNDLE DIGEST, AND SAYS SO. `model-bundle`
 * REJECTS the verdict-only run: it derives each reading's expected provider-call
 * shape from the shared execution map, which still declares a second
 * relation-naming call for 17,235 of the 33,361 readings, and the verdict-only
 * substrate removed that call. So the verdict-only side has no bundle behind it,
 * and no figure on this page may imply otherwise. The artifact carries the
 * rejection VERBATIM under `bundler_status`, this module parses it fail-closed,
 * and `verdictOnlyBundled` is typed `false` so a render site cannot forget.
 *
 * What stands in for it is a reconciliation the artifact must pass before it is
 * written at all: the deliberating side is recomputed from its own raw readings
 * through the same rollup and must reproduce the SHIPPED statement probabilities
 * exactly, and the resulting average precision, AUROC, Brier, calibration error
 * and confusion counts must equal the shipped ones exactly. Both sides therefore
 * come from one code path, and that path is pinned to the numbers /paper already
 * reports. `validatePaperReasoningAblation` GATES on the shipped-parity flags and
 * on a zero reconciliation residual, so a drifted rerun takes the figure down
 * rather than drawing a comparison between two runs that are no longer the same.
 *
 * SHIPPED FIELDS ARE READ, NEVER RECOMPUTED. Every metric, interval bound,
 * transition count, token count and dollar bound comes off
 * `data/results/reasoning_ablation_20260805/reasoning_ablation.json`. The only
 * arithmetic here is axis placement and the consistency checks below, which
 * compare shipped fields against each other and fail closed.
 *
 * DIRECTION IS DECIDED HERE. Every signed claim goes through `standingOfBounds`
 * from `paper-interval.ts`, and the sentences that depend on it are total records
 * keyed by the three classes, so there is no two-way test to be sign-blind with.
 *
 * This module is import-safe on the client: typed shape plus a pure, fail-closed
 * validator. All filesystem work lives in `$lib/server/paper-reasoning-ablation`.
 */

import { standingOfBounds, type Standing } from './paper-interval.ts';
import { pairShippedProse, type AnchoredProse, type ShippedProse } from './paper-prose.ts';
import {
	budget,
	fail,
	nonNegativeInteger,
	number,
	positiveInteger,
	record,
	text,
	unit
} from './paper-validate.ts';

/** The artifact kind this module will accept, and nothing else. */
export const REASONING_ABLATION_ARTIFACT_KIND = 'indra_reasoning_ablation';

/**
 * The two benchmarks this page scores on. `paper_all_source` counts every source
 * behind a statement; `paper_readers` keeps only the five reading systems. Both
 * are frozen artifact keys — never rendered. Render `display`.
 */
export const REASONING_ABLATION_BENCHMARK_IDS = ['paper_all_source', 'paper_readers'] as const;
export type ReasoningAblationBenchmarkId = (typeof REASONING_ABLATION_BENCHMARK_IDS)[number];

/**
 * On-screen names for the two benchmarks, ours to write in English. The artifact
 * ships its own `display` for each, but that string is sha-pinned bytes; these
 * are the words a reader is handed.
 */
export const REASONING_ABLATION_BENCHMARK_DISPLAY: Readonly<
	Record<ReasoningAblationBenchmarkId, string>
> = {
	paper_all_source: 'every source',
	paper_readers: 'reading systems only'
};

/**
 * The two sides of the comparison. `reasoning` is the shipped run; `verdict_only`
 * is the 2026-07-31 re-run with the provider's deliberation and the prompt
 * scaffolding both removed.
 */
export const REASONING_ABLATION_SIDE_IDS = ['reasoning', 'verdict_only'] as const;
export type ReasoningAblationSideId = (typeof REASONING_ABLATION_SIDE_IDS)[number];

/** What each side is called on screen. No abbreviation a reader must resolve. */
export const REASONING_ABLATION_SIDE_DISPLAY: Readonly<
	Record<ReasoningAblationSideId, string>
> = {
	reasoning: 'thinking first',
	verdict_only: 'verdict only'
};

/**
 * SVG geometry, exported so the two label budgets below are DERIVED from it and
 * so the contract runner can re-derive them. Deliberately the SAME frame as the
 * per-evidence figure: a reader who has learned to read one row per model, marks
 * on a shared axis, name in the left gutter and readout in the right, does not
 * have to learn a second frame two figures later. Mirrors the constants in
 * `ReasoningAblation.svelte`.
 */
export const REASONING_ABLATION_GEOMETRY = {
	width: 920,
	plotLeft: 196,
	plotRight: 782,
	/** Model names are right-anchored here, into the gutter left of it. */
	labelAnchorX: 184,
	/** Readouts are left-anchored here, into the gutter right of it. */
	readoutX: 790,
	topPad: 46,
	rowHeight: 48,
	intervalCap: 3.5,
	labelFontPx: 9,
	/** Measured advance of the mono face at 9px, in user units per character. */
	monoUnitsPerChar: 5.4186,
	readoutFontPx: 8,
	/** The same face at 8px, where the readouts are set. */
	readoutUnitsPerChar: 4.8165
} as const;

/**
 * MODEL NAME FIT. Names are right-anchored into the gutter left of `labelAnchorX`,
 * so an overrun loses its LEADING glyphs silently — SVG text does not wrap and
 * does not warn, and the <desc> beside it still emits the full string. Budget
 * DERIVED from the gutter, not chosen.
 */
export const REASONING_ABLATION_NAME_BUDGET_CHARS = Math.floor(
	REASONING_ABLATION_GEOMETRY.labelAnchorX / REASONING_ABLATION_GEOMETRY.monoUnitsPerChar
);

/** The readout that trails each row, into the gutter right of `readoutX`. */
export const REASONING_ABLATION_READOUT_BUDGET_CHARS = Math.floor(
	(REASONING_ABLATION_GEOMETRY.width - REASONING_ABLATION_GEOMETRY.readoutX) /
		REASONING_ABLATION_GEOMETRY.readoutUnitsPerChar
);

export function reasoningAblationNameFits(display: string): boolean {
	return display.length <= REASONING_ABLATION_NAME_BUDGET_CHARS;
}

export function reasoningAblationReadoutFits(readout: string): boolean {
	return readout.length <= REASONING_ABLATION_READOUT_BUDGET_CHARS;
}

/** Three decimals, the page's convention for a metric on the unit scale. */
export function fmt3(value: number): string {
	return value.toFixed(3);
}

/** A signed difference, always carrying its sign so the direction is on screen. */
export function fmtSignedDelta(value: number): string {
	const rendered = Math.abs(value).toFixed(4);
	if (value > 0) return `+${rendered}`;
	if (value < 0) return `−${rendered}`;
	return rendered;
}

/**
 * The sentence a reader gets for each of the three interval classes. A TOTAL
 * record: the compiler demands one per class, so a two-way branch has nothing to
 * branch on and cannot be sign-blind.
 */
export const REASONING_ABLATION_STANDING_SENTENCE: Readonly<Record<Standing, string>> = {
	ahead: 'Removing the thinking step measurably helped this model.',
	behind: 'Removing the thinking step measurably hurt this model.',
	'not-significant':
		'The range covers zero: on these statements this model scores the same either way.'
};

/**
 * The same three classes read off the band that covers having run four models.
 * A separate record rather than a reused one: the pointwise sentence would
 * overstate a margin that only the uncorrected interval clears, which is the
 * position two of these four models are in.
 */
export const REASONING_ABLATION_SIMULTANEOUS_SENTENCE: Readonly<Record<Standing, string>> = {
	ahead: 'Still helped once the range is widened to cover having run four models.',
	behind: 'Still hurt once the range is widened to cover having run four models.',
	'not-significant':
		'Widened to cover having run four models, the range covers zero — this one is not settled.'
};

/** A shipped interval and where it sits relative to zero. */
export interface ReasoningAblationDelta {
	/** SHIPPED point difference, verdict-only minus thinking-first. */
	value: number;
	low: number;
	high: number;
	/** Decided HERE, from the two endpoints, never from the sign of `value`. */
	standing: Standing;
	/**
	 * The same margin widened to cover having run FOUR models, not one — a max-t
	 * band over the whole family. A pointwise interval understates how often at
	 * least one of four clears zero by chance, and two of these four clear it by
	 * less than 0.001 pointwise. Both standings are carried; a claim that names a
	 * direction should be read off `simultaneousStanding`.
	 */
	simultaneousLow: number;
	simultaneousHigh: number;
	simultaneousStanding: Standing;
	familySize: number;
	resamples: number;
	validResamples: number;
}

/** Counts at the score cutoff the shipped comparison froze. */
export interface ReasoningAblationConfusion {
	tp: number;
	fp: number;
	fn: number;
	tn: number;
	precision: number;
	recall: number;
	f1: number;
	accuracy: number;
}

/**
 * One cut, described on the ERROR class — the metric this page leads on.
 *
 * WHY THE ERROR CLASS AND NOT THE RANKING MEASURES. The benchmark is 73.2%
 * correct, and the paper's own random forest already scores 0.9412 average
 * precision, so only 0.0588 of that scale is left to win. What a reading model
 * is FOR is pushing errors down, and at statement grain that is error-class F1 —
 * the argument `paper-error-f1.ts` makes for the figure this one now sits beside.
 * Drawing this comparison on AUROC compressed it roughly sevenfold.
 *
 * `flagged` is `belief < tau`: the statements the model sends to a curator.
 */
export interface ReasoningAblationErrorCut {
	tau: number;
	/** Statements this cut sends for review. */
	flagged: number;
	/** Of those, the share that really are wrong. */
	errorPrecision: number;
	/** Of the wrong statements, the share this cut catches. */
	errorRecall: number;
	errorF1: number;
	tp: number;
	fp: number;
	fn: number;
	tn: number;
}

/**
 * The same comparison under the two threshold rules that both exist here, kept
 * side by side because the ablation READS DIFFERENTLY under each and a figure
 * that showed only one would be picking the flattering answer without saying so.
 *
 *   · `ownCut` — each side at its OWN best-error-F1 cut. The shipped headline
 *     rule, and an ORACLE: the cut is chosen on this benchmark with these labels
 *     already in hand. `oracleDisclosure` travels with every number under it.
 *   · `deployedCut` — both sides at the DELIBERATING side's cut. What a curator
 *     who swapped the model and left the cutoff alone actually gets. No oracle is
 *     available to the verdict-only side here.
 *
 * For the three larger models the two rules land on the same cut and agree. For
 * the smallest they do not, and the gap between them is what re-tuning the cutoff
 * buys back — which is a real finding, not a presentational choice.
 */
export interface ReasoningAblationErrorClass {
	ownCut: { reasoning: ReasoningAblationErrorCut; verdictOnly: ReasoningAblationErrorCut };
	deployedCut: {
		tau: number;
		reasoning: ReasoningAblationErrorCut;
		verdictOnly: ReasoningAblationErrorCut;
	};
	/** True when the two rules choose the same cut for both sides. */
	cutsAgree: boolean;
}

/** One side's statement-grain measurement on one benchmark. */
export interface ReasoningAblationSideMetrics {
	averagePrecision: number;
	auroc: number;
	brier: number;
	/** Expected calibration error over the shipped frozen bins. Lower is better. */
	ece: number;
	confusion: ReasoningAblationConfusion;
}

export interface ReasoningAblationBenchmark {
	id: ReasoningAblationBenchmarkId;
	display: string;
	nEvaluable: number;
	nPositive: number;
	nNegative: number;
	/** The score cutoff both sides are counted at. */
	threshold: number;
	reasoning: ReasoningAblationSideMetrics;
	verdictOnly: ReasoningAblationSideMetrics;
	errorClass: ReasoningAblationErrorClass;
	/** THE DRAWN MARGIN: error-class F1, each side at its own best cut. */
	errorF1OwnCutDelta: ReasoningAblationDelta;
	/** The same margin with the cutoff left where it was. */
	errorF1DeployedCutDelta: ReasoningAblationDelta;
	averagePrecisionDelta: ReasoningAblationDelta;
	aurocDelta: ReasoningAblationDelta;
}

/**
 * What changed at the grain the model actually works at. `toCorrect` and
 * `toIncorrect` count only readings BOTH runs answered with the model — the
 * deterministic readings are replayed, not asked, and cannot move.
 */
export interface ReasoningAblationEvidenceGrain {
	nExecutions: number;
	reasoningCorrect: number;
	verdictOnlyCorrect: number;
	/** Readings both runs put to the model. */
	nModelRead: number;
	/** Rejected while thinking, accepted without. */
	toCorrect: number;
	/** Accepted while thinking, rejected without. */
	toIncorrect: number;
	/** Share of model-read readings that did not change. */
	agreement: number;
}

/** Dollar bounds for one side. Lower is provider-measured; upper adds reserved. */
export interface ReasoningAblationCost {
	lower: number;
	upper: number;
}

export interface ReasoningAblationModel {
	/** Frozen artifact key. Never rendered — render `display`. */
	id: string;
	display: string;
	evidence: ReasoningAblationEvidenceGrain;
	benchmarks: ReasoningAblationBenchmark[];
	reasoningCost: ReasoningAblationCost;
	/** Null when the run's spend sweep was skipped; never coalesced to a number. */
	verdictOnlyCost: ReasoningAblationCost | null;
	/** SHIPPED output-token totals, the direct evidence deliberation was off. */
	reasoningOutputTokens: number | null;
	verdictOnlyOutputTokens: number | null;
}

/**
 * The rules behind every error-class number on this surface, each verbatim from
 * the artifact beside the plain restatement a reader gets.
 *
 * A threshold-fitted number without its threshold rule is an over-claim wherever
 * it is written, and the oracle disclosure is not optional decoration: every cut
 * under `ownCut` was chosen on this benchmark with these labels in hand.
 */
export interface ReasoningAblationErrorRules {
	decision: ShippedProse;
	threshold: ShippedProse;
	oracle: ShippedProse;
}

/** The bundler's refusal, verbatim, beside the plain restatement of it. */
export interface ReasoningAblationBundlerStatus {
	/** Always `false` on this surface. Typed so a render site cannot forget. */
	verdictOnlyBundled: false;
	error: ShippedProse;
	cause: ShippedProse;
	consequence: ShippedProse;
}

/**
 * The counts every model on this figure shares, lifted to figure level BECAUSE
 * they are shared. A render site that reached for `models[0]?.evidence…` would
 * have to coalesce the optional away, and a coalesced count prints as a fact —
 * the defect class this page has already shipped once as `ece: 0`. The validator
 * gates when the models disagree on any of these, so by the time the figure
 * exists they are one number, not four.
 */
export interface ReasoningAblationCensus {
	/** Evidence readings each model performed, in each run. */
	nExecutions: number;
	/** Of those, the ones both runs actually put to the model. */
	nModelRead: number;
	/** Statements scored, per benchmark. */
	statements: Record<ReasoningAblationBenchmarkId, number>;
	/** Of those, the ones the paper's labels call wrong — what this figure hunts. */
	errors: Record<ReasoningAblationBenchmarkId, number>;
}

export interface ReasoningAblationFigure {
	frozenAt: string;
	bundler: ReasoningAblationBundlerStatus;
	errorRules: ReasoningAblationErrorRules;
	models: ReasoningAblationModel[];
	census: ReasoningAblationCensus;
	/** Every model's deliberating side reproduced the shipped numbers exactly. */
	shippedParityVerified: true;
	resamples: number;
	seed: number;
}

export interface ReasoningAblationOk {
	status: 'ok';
	reason: null;
	figure: ReasoningAblationFigure;
	artifact_path: string;
	/**
	 * NULLABLE on the ok branch too: the server always supplies a digest, the pure
	 * client-side validate path does not, and coalescing to '' would print an empty
	 * provenance line that reads as a real, empty digest.
	 */
	artifact_sha256: string | null;
}

export interface ReasoningAblationUnavailable {
	status: 'unavailable';
	reason: string;
	figure: null;
	artifact_path: string;
	artifact_sha256: string | null;
}

export type ReasoningAblationLoad = ReasoningAblationOk | ReasoningAblationUnavailable;

export interface ReasoningAblationContext {
	artifactPath?: string;
	artifactSha256?: string;
}

/**
 * PLAIN RESTATEMENTS of the three sha-pinned sentences this surface prints. Kept
 * in one block so the whole of what a reader is told about the refusal can be
 * read at once. Each is pinned to a verbatim fragment of the sentence it
 * restates, so a reissued artifact that reworded one gates instead of printing a
 * restatement under the wrong sentence.
 */
const BUNDLER_TWINS: Readonly<Record<'error' | 'cause' | 'consequence', AnchoredProse>> = {
	error: {
		artifactAnchor: 'final provider-call topology differs',
		plain: 'The packaging step refuses this run.'
	},
	cause: {
		artifactAnchor: 'shared execution map',
		plain:
			'It checks how many calls each reading should have made against a shared list ' +
			'that still expects a second call for 17,235 of the 33,361 readings. The ' +
			'verdict-only run removed that second call, so every one of them looks wrong to it.'
	},
	consequence: {
		artifactAnchor: 'computed here',
		plain:
			'So the verdict-only side carries no packaging digest. Both sides were instead ' +
			'recomputed together from their own raw readings, and the thinking-first side had ' +
			'to reproduce the numbers already published on this page exactly before either was written.'
	}
};

/**
 * PLAIN RESTATEMENTS of the three sha-pinned threshold sentences. Kept beside the
 * bundler twins so the whole of what a reader is told about method can be read in
 * one place. The oracle sentence is the one that matters most: without it, a
 * benchmark-fitted cutoff reads as a cutoff you could have picked in advance.
 */
const ERROR_RULE_TWINS: Readonly<Record<'decision' | 'threshold' | 'oracle', AnchoredProse>> = {
	decision: {
		artifactAnchor: 'belief < tau',
		plain: 'A statement is sent for review when its score falls below the cutoff.'
	},
	threshold: {
		artifactAnchor: 'best-error-F1 cut',
		plain:
			'Each model gets the cutoff that finds the most errors it can on these same ' +
			'1,689 statements, and ties go to the lowest such cutoff.'
	},
	oracle: {
		artifactAnchor: 'chosen ON THIS PANEL',
		plain:
			'Those cutoffs were picked with the answers already in hand, on the same statements ' +
			'they are then scored on. You could not have chosen them before curating, and none ' +
			'of them is checked on data it did not see.'
	}
};

type UnknownRecord = Record<string, unknown>;

function parseDelta(raw: unknown, context: string): ReasoningAblationDelta {
	const obj = record(raw, context);
	const bounds = obj.ci95;
	if (!Array.isArray(bounds) || bounds.length !== 2) {
		fail(`${context}.ci95`, 'expected a two-element interval');
	}
	const low = number(bounds[0], `${context}.ci95[0]`);
	const high = number(bounds[1], `${context}.ci95[1]`);
	if (low > high) fail(`${context}.ci95`, 'the interval is inverted');
	const value = number(obj.value, `${context}.value`);
	// The artifact ships its own standing; it is RECOMPUTED here from the two
	// endpoints and gated against the shipped one. A figure that trusted the
	// shipped flag would inherit any sign-blindness in the producer.
	const derived = standingOfBounds(low, high);
	const shipped = text(obj.standing, `${context}.standing`);
	if (shipped !== derived) {
		fail(`${context}.standing`, `ships ${shipped} but its own interval reads ${derived}`);
	}
	const simultaneous = record(obj.simultaneous, `${context}.simultaneous`);
	const wide = simultaneous.ci95;
	if (!Array.isArray(wide) || wide.length !== 2) {
		fail(`${context}.simultaneous.ci95`, 'expected a two-element interval');
	}
	const wideLow = number(wide[0], `${context}.simultaneous.ci95[0]`);
	const wideHigh = number(wide[1], `${context}.simultaneous.ci95[1]`);
	if (wideLow > wideHigh) fail(`${context}.simultaneous.ci95`, 'the interval is inverted');
	// A correction for having run four models can only WIDEN. A "simultaneous"
	// band narrower than its own pointwise interval is not a correction, and it
	// would let a claim be promoted by the very step meant to restrain it.
	if (wideLow > low || wideHigh < high) {
		fail(
			`${context}.simultaneous.ci95`,
			'is narrower than the pointwise interval it is supposed to widen'
		);
	}
	const wideDerived = standingOfBounds(wideLow, wideHigh);
	const wideShipped = text(simultaneous.standing, `${context}.simultaneous.standing`);
	if (wideShipped !== wideDerived) {
		fail(
			`${context}.simultaneous.standing`,
			`ships ${wideShipped} but its own interval reads ${wideDerived}`
		);
	}
	return {
		value,
		low,
		high,
		standing: derived,
		simultaneousLow: wideLow,
		simultaneousHigh: wideHigh,
		simultaneousStanding: wideDerived,
		familySize: positiveInteger(
			simultaneous.family_size,
			`${context}.simultaneous.family_size`
		),
		resamples: positiveInteger(obj.resamples, `${context}.resamples`),
		validResamples: positiveInteger(obj.valid_resamples, `${context}.valid_resamples`)
	};
}

function parseConfusion(raw: unknown, context: string): ReasoningAblationConfusion {
	const obj = record(raw, context);
	return {
		tp: nonNegativeInteger(obj.tp, `${context}.tp`),
		fp: nonNegativeInteger(obj.fp, `${context}.fp`),
		fn: nonNegativeInteger(obj.fn, `${context}.fn`),
		tn: nonNegativeInteger(obj.tn, `${context}.tn`),
		precision: unit(obj.precision, `${context}.precision`),
		recall: unit(obj.recall, `${context}.recall`),
		f1: unit(obj.f1, `${context}.f1`),
		accuracy: unit(obj.accuracy, `${context}.accuracy`)
	};
}

function parseErrorCut(
	raw: unknown,
	context: string,
	nNegative: number
): ReasoningAblationErrorCut {
	const obj = record(raw, context);
	const tp = nonNegativeInteger(obj.tp, `${context}.tp`);
	const fp = nonNegativeInteger(obj.fp, `${context}.fp`);
	const fn = nonNegativeInteger(obj.fn, `${context}.fn`);
	const tn = nonNegativeInteger(obj.tn, `${context}.tn`);
	const flagged = nonNegativeInteger(obj.flagged, `${context}.flagged`);
	// The error class must CLOSE on the benchmark's own error count, and the
	// review pile must be exactly what the cut flags. Either failing means the
	// counts are not this cut's counts.
	if (tp + fn !== nNegative) {
		fail(`${context}`, 'the caught and missed errors do not sum to the benchmark’s errors');
	}
	if (tp + fp !== flagged) {
		fail(`${context}.flagged`, 'is not the errors caught plus the correct statements flagged');
	}
	return {
		tau: unit(obj.tau, `${context}.tau`),
		flagged,
		errorPrecision: unit(obj.error_precision, `${context}.error_precision`),
		errorRecall: unit(obj.error_recall, `${context}.error_recall`),
		errorF1: unit(obj.error_f1, `${context}.error_f1`),
		tp,
		fp,
		fn,
		tn
	};
}

function parseErrorClass(
	raw: unknown,
	context: string,
	nNegative: number
): ReasoningAblationErrorClass {
	const obj = record(raw, context);
	const own = record(obj.own_cut, `${context}.own_cut`);
	const deployed = record(obj.deployed_cut, `${context}.deployed_cut`);
	const ownCut = {
		reasoning: parseErrorCut(own.reasoning, `${context}.own_cut.reasoning`, nNegative),
		verdictOnly: parseErrorCut(own.verdict_only, `${context}.own_cut.verdict_only`, nNegative)
	};
	const deployedTau = unit(deployed.tau, `${context}.deployed_cut.tau`);
	const deployedCut = {
		tau: deployedTau,
		reasoning: parseErrorCut(deployed.reasoning, `${context}.deployed_cut.reasoning`, nNegative),
		verdictOnly: parseErrorCut(
			deployed.verdict_only,
			`${context}.deployed_cut.verdict_only`,
			nNegative
		)
	};
	// The deployed rule IS the deliberating side's own cut. If those disagree the
	// two rules are not the two rules this figure names, and the "what re-tuning
	// buys back" reading below would be measuring something else.
	if (deployedTau !== ownCut.reasoning.tau) {
		fail(
			`${context}.deployed_cut.tau`,
			'is not the thinking-first side’s own cut, which is what this rule is defined as'
		);
	}
	if (deployedCut.reasoning.errorF1 !== ownCut.reasoning.errorF1) {
		fail(
			`${context}.deployed_cut.reasoning`,
			'differs from the same side at the same cut under the other rule'
		);
	}
	return {
		ownCut,
		deployedCut,
		cutsAgree: ownCut.verdictOnly.tau === deployedTau
	};
}

function parseSide(raw: unknown, context: string): ReasoningAblationSideMetrics {
	const obj = record(raw, context);
	return {
		averagePrecision: unit(obj.average_precision, `${context}.average_precision`),
		auroc: unit(obj.auroc, `${context}.auroc`),
		brier: unit(obj.brier, `${context}.brier`),
		ece: unit(obj.ece, `${context}.ece`),
		confusion: parseConfusion(obj.confusion, `${context}.confusion`)
	};
}

function parseBenchmark(
	id: ReasoningAblationBenchmarkId,
	raw: unknown,
	context: string
): ReasoningAblationBenchmark {
	const obj = record(raw, context);
	if (obj.shipped_parity_verified !== true) {
		fail(
			`${context}.shipped_parity_verified`,
			'the thinking-first side was not verified against the shipped comparison'
		);
	}
	const nEvaluable = positiveInteger(obj.n_evaluable, `${context}.n_evaluable`);
	const nPositive = positiveInteger(obj.n_positive, `${context}.n_positive`);
	const nNegative = positiveInteger(obj.n_negative, `${context}.n_negative`);
	if (nPositive + nNegative !== nEvaluable) {
		fail(`${context}.n_evaluable`, 'is not its own positive and negative counts');
	}
	const reasoning = parseSide(obj.reasoning, `${context}.reasoning`);
	const verdictOnly = parseSide(obj.verdict_only, `${context}.verdict_only`);
	for (const [side, metrics] of [
		['reasoning', reasoning],
		['verdict_only', verdictOnly]
	] as const) {
		const confusion = metrics.confusion;
		const total = confusion.tp + confusion.fp + confusion.fn + confusion.tn;
		if (total !== nEvaluable) {
			fail(`${context}.${side}.confusion`, 'does not count every statement exactly once');
		}
		if (confusion.tp + confusion.fn !== nPositive) {
			fail(`${context}.${side}.confusion`, 'does not recover this benchmark’s positive count');
		}
	}
	const errorClass = parseErrorClass(obj.error_class, `${context}.error_class`, nNegative);
	const delta = record(obj.delta, `${context}.delta`);
	const averagePrecisionDelta = parseDelta(
		delta.average_precision,
		`${context}.delta.average_precision`
	);
	const aurocDelta = parseDelta(delta.auroc, `${context}.delta.auroc`);
	const errorF1OwnCutDelta = parseDelta(
		delta.error_f1_own_cut,
		`${context}.delta.error_f1_own_cut`
	);
	const errorF1DeployedCutDelta = parseDelta(
		delta.error_f1_deployed_cut,
		`${context}.delta.error_f1_deployed_cut`
	);
	for (const [name, shippedDelta, after, before] of [
		[
			'error_f1_own_cut',
			errorF1OwnCutDelta,
			errorClass.ownCut.verdictOnly.errorF1,
			errorClass.ownCut.reasoning.errorF1
		],
		[
			'error_f1_deployed_cut',
			errorF1DeployedCutDelta,
			errorClass.deployedCut.verdictOnly.errorF1,
			errorClass.deployedCut.reasoning.errorF1
		]
	] as const) {
		if (Math.abs(shippedDelta.value - (after - before)) > 1e-9) {
			fail(
				`${context}.delta.${name}.value`,
				'is not this rule’s verdict-only error-F1 minus its thinking-first one'
			);
		}
	}
	// The shipped point difference must BE the difference of the two shipped
	// measurements. Both are doubles near 0.95, so the achievable residual is
	// ~1e-15; 1e-9 is decisive and still tolerates the JSON round-trip.
	for (const [name, shippedDelta, after, before] of [
		[
			'average_precision',
			averagePrecisionDelta,
			verdictOnly.averagePrecision,
			reasoning.averagePrecision
		],
		['auroc', aurocDelta, verdictOnly.auroc, reasoning.auroc]
	] as const) {
		if (Math.abs(shippedDelta.value - (after - before)) > 1e-9) {
			fail(
				`${context}.delta.${name}.value`,
				'is not this benchmark’s verdict-only measurement minus its thinking-first one'
			);
		}
	}
	return {
		id,
		display: REASONING_ABLATION_BENCHMARK_DISPLAY[id],
		nEvaluable,
		nPositive,
		nNegative,
		threshold: unit(obj.threshold, `${context}.threshold`),
		reasoning,
		verdictOnly,
		errorClass,
		errorF1OwnCutDelta,
		errorF1DeployedCutDelta,
		averagePrecisionDelta,
		aurocDelta
	};
}

function parseEvidenceGrain(raw: unknown, context: string): ReasoningAblationEvidenceGrain {
	const obj = record(raw, context);
	const nExecutions = positiveInteger(obj.n_executions, `${context}.n_executions`);
	const reasoning = record(obj.reasoning, `${context}.reasoning`);
	const verdictOnly = record(obj.verdict_only, `${context}.verdict_only`);
	const modelTier = record(obj.llm_tier, `${context}.llm_tier`);
	const nModelRead = positiveInteger(modelTier.n, `${context}.llm_tier.n`);
	const toCorrect = nonNegativeInteger(modelTier.to_correct, `${context}.llm_tier.to_correct`);
	const toIncorrect = nonNegativeInteger(
		modelTier.to_incorrect,
		`${context}.llm_tier.to_incorrect`
	);
	const flips = nonNegativeInteger(modelTier.flips, `${context}.llm_tier.flips`);
	if (toCorrect + toIncorrect !== flips) {
		fail(`${context}.llm_tier.flips`, 'is not its own two directions summed');
	}
	if (flips > nModelRead) {
		fail(`${context}.llm_tier.flips`, 'exceeds the readings the model was asked for');
	}
	return {
		nExecutions,
		reasoningCorrect: nonNegativeInteger(reasoning.correct, `${context}.reasoning.correct`),
		verdictOnlyCorrect: nonNegativeInteger(
			verdictOnly.correct,
			`${context}.verdict_only.correct`
		),
		nModelRead,
		toCorrect,
		toIncorrect,
		agreement: unit(modelTier.agreement, `${context}.llm_tier.agreement`)
	};
}

function parseCost(raw: unknown, context: string): ReasoningAblationCost | null {
	if (raw === null || raw === undefined) return null;
	const obj = record(raw, context);
	const lower = number(obj.inference_usd_lower, `${context}.inference_usd_lower`);
	const upper = number(obj.inference_usd_upper, `${context}.inference_usd_upper`);
	if (lower < 0 || upper < lower) fail(context, 'the dollar bounds are inverted or negative');
	return { lower, upper };
}

function optionalTokens(raw: UnknownRecord, key: string, context: string): number | null {
	const value = raw[key];
	if (value === null || value === undefined) return null;
	return nonNegativeInteger(value, `${context}.${key}`);
}

function parseModel(raw: unknown, context: string): ReasoningAblationModel {
	const obj = record(raw, context);
	const id = text(obj.arm_id, `${context}.arm_id`);
	const display = budget(
		text(obj.display, `${context}.display`),
		REASONING_ABLATION_NAME_BUDGET_CHARS,
		`${context}.display`,
		reasoningAblationNameFits
	);

	// Both sides must have rebuilt the shipped statement probabilities EXACTLY.
	const reconciliation = record(obj.reconciliation, `${context}.reconciliation`);
	for (const benchmarkId of REASONING_ABLATION_BENCHMARK_IDS) {
		const block = record(
			reconciliation[benchmarkId],
			`${context}.reconciliation.${benchmarkId}`
		);
		const nStatements = positiveInteger(
			block.n_statements,
			`${context}.reconciliation.${benchmarkId}.n_statements`
		);
		const nExact = positiveInteger(
			block.n_exact,
			`${context}.reconciliation.${benchmarkId}.n_exact`
		);
		const maxAbs = number(
			block.max_abs_diff,
			`${context}.reconciliation.${benchmarkId}.max_abs_diff`
		);
		if (nExact !== nStatements || maxAbs !== 0) {
			fail(
				`${context}.reconciliation.${benchmarkId}`,
				'the thinking-first side does not rebuild the published statement scores exactly'
			);
		}
	}

	const benchmarksRaw = record(obj.panels, `${context}.panels`);
	const benchmarks = REASONING_ABLATION_BENCHMARK_IDS.map((benchmarkId) =>
		parseBenchmark(benchmarkId, benchmarksRaw[benchmarkId], `${context}.panels.${benchmarkId}`)
	);

	const reasoningSide = record(obj.reasoning, `${context}.reasoning`);
	const verdictOnlySide = record(obj.verdict_only, `${context}.verdict_only`);
	const reasoningCost = parseCost(reasoningSide.cost, `${context}.reasoning.cost`);
	if (reasoningCost === null) {
		fail(`${context}.reasoning.cost`, 'the shipped run always carries its own dollar bounds');
	}
	if (verdictOnlySide.raw_attempts !== undefined) {
		const attempts = record(verdictOnlySide.raw_attempts, `${context}.verdict_only.raw_attempts`);
		if (attempts.sha256_matches_bundle_manifest !== false) {
			fail(
				`${context}.verdict_only.raw_attempts.sha256_matches_bundle_manifest`,
				'the verdict-only run has no bundle, so this cannot be true'
			);
		}
	}
	const verdictOnlyCostRaw = verdictOnlySide.cost;
	const verdictOnlyCost = parseCost(verdictOnlyCostRaw ?? null, `${context}.verdict_only.cost`);

	return {
		id,
		display,
		evidence: parseEvidenceGrain(obj.evidence_grain, `${context}.evidence_grain`),
		benchmarks,
		reasoningCost,
		verdictOnlyCost,
		reasoningOutputTokens: null,
		verdictOnlyOutputTokens:
			verdictOnlyCostRaw === null || verdictOnlyCostRaw === undefined
				? null
				: optionalTokens(
						record(verdictOnlyCostRaw, `${context}.verdict_only.cost`),
						'output_tokens',
						`${context}.verdict_only.cost`
					)
	};
}

/**
 * The model whose statement-grain change is most decided, on the named benchmark
 * and the named measurement. Ties break on the id so the choice is deterministic
 * across reloads. Returns null when nothing is decided, which is a real state on
 * this surface and not an error.
 */
export function mostDecidedModel(
	models: ReasoningAblationModel[],
	benchmarkId: ReasoningAblationBenchmarkId,
	pick: (benchmark: ReasoningAblationBenchmark) => ReasoningAblationDelta
): ReasoningAblationModel | null {
	const decided = models.filter((model) => {
		const benchmark = model.benchmarks.find((entry) => entry.id === benchmarkId);
		return benchmark !== undefined && pick(benchmark).standing !== 'not-significant';
	});
	if (decided.length === 0) return null;
	return decided
		.slice()
		.sort((a, b) => {
			const left = pick(a.benchmarks.find((entry) => entry.id === benchmarkId)!);
			const right = pick(b.benchmarks.find((entry) => entry.id === benchmarkId)!);
			return Math.abs(right.value) - Math.abs(left.value) || a.id.localeCompare(b.id);
		})[0];
}

/** Inclusive [min, max] of a measurement over both sides of every model. */
export function reasoningAblationExtent(
	models: ReasoningAblationModel[],
	benchmarkId: ReasoningAblationBenchmarkId,
	pick: (side: ReasoningAblationSideMetrics) => number
): { min: number; max: number } | null {
	const values: number[] = [];
	for (const model of models) {
		const benchmark = model.benchmarks.find((entry) => entry.id === benchmarkId);
		if (!benchmark) continue;
		values.push(pick(benchmark.reasoning), pick(benchmark.verdictOnly));
	}
	if (values.length === 0) return null;
	return { min: Math.min(...values), max: Math.max(...values) };
}

/**
 * Pure, fail-closed validator. Returns the figure or `status:'unavailable'` with
 * a reason on any shape drift, any unverified shipped-parity flag, any non-zero
 * reconciliation residual, and any claim that the verdict-only run was packaged.
 * Never throws.
 */
export function validatePaperReasoningAblation(
	raw: unknown,
	context: ReasoningAblationContext = {}
): ReasoningAblationLoad {
	const artifactPath = context.artifactPath ?? '';
	const artifactSha256 = context.artifactSha256 ?? null;
	try {
		const obj = record(raw, 'reasoning_ablation');
		const kind = text(obj.artifact_kind, 'reasoning_ablation.artifact_kind');
		if (kind !== REASONING_ABLATION_ARTIFACT_KIND) {
			fail('reasoning_ablation.artifact_kind', `expected ${REASONING_ABLATION_ARTIFACT_KIND}`);
		}

		const bundlerRaw = record(obj.bundler_status, 'reasoning_ablation.bundler_status');
		if (text(bundlerRaw.state, 'reasoning_ablation.bundler_status.state') !== 'rejected') {
			fail(
				'reasoning_ablation.bundler_status.state',
				'this surface exists because the packaging step refuses the run; a different state is a different figure'
			);
		}
		const [error, cause, consequence] = pairShippedProse(
			[
				text(bundlerRaw.error, 'reasoning_ablation.bundler_status.error'),
				text(bundlerRaw.cause, 'reasoning_ablation.bundler_status.cause'),
				text(bundlerRaw.consequence, 'reasoning_ablation.bundler_status.consequence')
			],
			[BUNDLER_TWINS.error, BUNDLER_TWINS.cause, BUNDLER_TWINS.consequence],
			'reasoning_ablation.bundler_status'
		);

		const provenance = record(obj.provenance, 'reasoning_ablation.provenance');
		const errorRulesRaw = record(
			provenance.error_class,
			'reasoning_ablation.provenance.error_class'
		);
		const [decisionRule, thresholdRule, oracleRule] = pairShippedProse(
			[
				text(errorRulesRaw.decision_rule, 'reasoning_ablation.provenance.error_class.decision_rule'),
				text(
					errorRulesRaw.threshold_rule,
					'reasoning_ablation.provenance.error_class.threshold_rule'
				),
				text(
					errorRulesRaw.oracle_disclosure,
					'reasoning_ablation.provenance.error_class.oracle_disclosure'
				)
			],
			[ERROR_RULE_TWINS.decision, ERROR_RULE_TWINS.threshold, ERROR_RULE_TWINS.oracle],
			'reasoning_ablation.provenance.error_class'
		);
		if (provenance.aggregation_identical_across_runs !== true) {
			fail(
				'reasoning_ablation.provenance.aggregation_identical_across_runs',
				'the two runs roll evidence up differently, so a rollup change would be indistinguishable from a reading change'
			);
		}
		const bootstrap = record(provenance.bootstrap, 'reasoning_ablation.provenance.bootstrap');

		const modelsRaw = obj.arms;
		if (!Array.isArray(modelsRaw) || modelsRaw.length === 0) {
			fail('reasoning_ablation.arms', 'expected a non-empty array');
		}
		const models = modelsRaw.map((entry, index) =>
			parseModel(entry, `reasoning_ablation.arms[${index}]`)
		);
		const ids = new Set(models.map((model) => model.id));
		if (ids.size !== models.length) fail('reasoning_ablation.arms', 'model keys repeat');

		// EVERY shared count is gated here, and the figure carries the single value
		// rather than leaving a render site to pick one model and coalesce.
		const executions = new Set(models.map((model) => model.evidence.nExecutions));
		if (executions.size !== 1) {
			fail(
				'reasoning_ablation.arms',
				'the models did not all read the same number of evidence rows, so they are not paired'
			);
		}
		const modelRead = new Set(models.map((model) => model.evidence.nModelRead));
		if (modelRead.size !== 1) {
			fail(
				'reasoning_ablation.arms',
				'the models were not all asked for the same readings, so their changed-reading counts are not comparable'
			);
		}
		const statements = {} as Record<ReasoningAblationBenchmarkId, number>;
		const errors = {} as Record<ReasoningAblationBenchmarkId, number>;
		for (const benchmarkId of REASONING_ABLATION_BENCHMARK_IDS) {
			for (const [name, target, pick] of [
				['n_evaluable', statements, (b: ReasoningAblationBenchmark) => b.nEvaluable],
				['n_negative', errors, (b: ReasoningAblationBenchmark) => b.nNegative]
			] as const) {
				const counts = new Set(
					models.map((model) => pick(model.benchmarks.find((entry) => entry.id === benchmarkId)!))
				);
				if (counts.size !== 1) {
					fail(
						`reasoning_ablation.arms[*].panels.${benchmarkId}.${name}`,
						'the models were not all scored on the same statements'
					);
				}
				target[benchmarkId] = [...counts][0];
			}
		}

		return {
			status: 'ok',
			reason: null,
			artifact_path: artifactPath,
			artifact_sha256: artifactSha256,
			figure: {
				frozenAt: text(obj.frozen_at, 'reasoning_ablation.frozen_at'),
				bundler: { verdictOnlyBundled: false, error, cause, consequence },
				errorRules: { decision: decisionRule, threshold: thresholdRule, oracle: oracleRule },
				models,
				census: {
					nExecutions: [...executions][0],
					nModelRead: [...modelRead][0],
					statements,
					errors
				},
				shippedParityVerified: true,
				resamples: positiveInteger(
					bootstrap.resamples,
					'reasoning_ablation.provenance.bootstrap.resamples'
				),
				seed: positiveInteger(bootstrap.seed, 'reasoning_ablation.provenance.bootstrap.seed')
			}
		};
	} catch (error) {
		return {
			status: 'unavailable',
			reason: error instanceof Error ? error.message : String(error),
			figure: null,
			artifact_path: artifactPath,
			artifact_sha256: artifactSha256
		};
	}
}
