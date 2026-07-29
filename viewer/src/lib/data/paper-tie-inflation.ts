/**
 * Typed data contract for the TIE-INFLATION explainer that sits directly beneath
 * the paper's-own-metric comparison.
 *
 * WHAT THIS NODE ARGUES. The 2023 paper summarises each method as a per-fold
 * `sklearn.precision_recall_curve` followed by `auc(recall, precision)` — a
 * TRAPEZOIDAL integral. Between two adjacent precision-recall operating points
 * the trapezoid draws a straight chord, but no threshold realises any interior
 * point on that chord: crossing a block of TIED scores admits every tied
 * statement at once, so the achievable curve STEPS. Average precision integrates
 * that step (`Σ ΔR·P_new`); the trapezoid integrates the chord
 * (`Σ ΔR·(P_old+P_new)/2`). Their difference is exactly the sum of the triangles
 * between chord and step, `Σ ΔR·ΔP/2`, and it grows with how tied an arm's
 * scores are. The paper's random forests emit a near-continuous score and are
 * untouched by this; OUR reader arms pile hundreds of statements on a handful of
 * values and are flattered by it. This figure argues AGAINST our own
 * best-looking number, which is its entire point.
 *
 * SHIPPED FIELDS ARE READ, NEVER RECOMPUTED. `trapezoidal_minus_ap_inflation`
 * and `distinct_scores` come off `paper_literal_vs_llms.json` verbatim. The one
 * derived scalar here is `sameEstimatorInflation` — a difference of two OTHER
 * shipped scalars (`pooled_trapezoidal_pr_auc − pooled_average_precision`),
 * carried because the shipped headline differences a FOLD-MEAN trapezoid against
 * a POOLED average precision, and a paper author will rightly ask how much of
 * the gap is that mismatch rather than tie interpolation. Both are shown.
 *
 * This module is import-safe on the client: typed shape plus pure, fail-closed
 * validation. All filesystem work and all curve geometry live in
 * `$lib/server/paper-tie-inflation`.
 */

import {
	PAPER_LITERAL_ARM_SPECS,
	PAPER_LITERAL_REFERENCE_ARM_ID,
	type PaperArmKind
} from './paper-literal.ts';

/**
 * The one LLM-kind arm that is not a reading arm: a model bundle carried under
 * `kind: 'llm'`. Named the same way ScoreDistribution and PaperLiteralComparison
 * name it, so "reader arms" means the same set on every panel of this page.
 */
export const NON_READER_LLM_ARM_ID = 'indra-cogex-hybrid';

/**
 * Σ of the per-segment triangles must reproduce `pooled_trapezoidal_pr_auc −
 * pooled_average_precision` for every arm we draw. Both shipped values are
 * doubles at ~0.95, so the achievable residual is ~1e-15; 1e-9 is a decisive
 * guard that still tolerates the artifact's 17-significant-digit round-trip.
 * A residual above this gates the whole load to `unavailable` — the drawn
 * triangles are a claim about a shipped number, so they are never drawn unless
 * they reconstruct it.
 */
export const TIE_RECONCILIATION_TOLERANCE = 1e-9;

/**
 * SVG geometry of the tie-ness scatter (figure 2), exported so the label budget
 * below is DERIVED from it and so the contract runner can re-derive it. Mirrors
 * the `F2` constants in `TieInflation.svelte`.
 */
export const TIE_SCATTER_GEOMETRY = {
	width: 760,
	left: 62,
	right: 726,
	/** Each label sits this far to the side of its mark. */
	labelOffsetX: 9,
	labelFontPx: 9,
	/** Measured advance of the mono face at 9px, in user units per character. */
	monoUnitsPerChar: 5.4186
} as const;

/**
 * POINT LABEL FIT. Figure 2 direct-labels every arm at its own mark, on the side
 * AWAY from the nearer plot edge. That keeps the long paper method names off the
 * frame — but "keeps" was, until now, an argument in a comment rather than a
 * check, and SVG text that overruns the viewBox is clipped with no error and no
 * failing a11y assertion (the <desc> still emits the full string).
 *
 * This returns true when the label fits the gutter it is actually anchored into:
 * `end`-anchored labels grow LEFT toward `left`, `start`-anchored labels grow
 * RIGHT toward `width`. Measured headroom on today's data is 6.0× on the worst
 * case ("Our port of RF + Type/#PMIDs/promoter", 37 chars, 200.5 units, into a
 * 568.8-unit gutter). The component gates figure 2 when any label fails.
 */
export function tieScatterLabelFits(
	display: string,
	labelX: number,
	anchor: 'start' | 'end'
): boolean {
	const g = TIE_SCATTER_GEOMETRY;
	const width = display.length * g.monoUnitsPerChar;
	return anchor === 'end' ? labelX - width >= g.left : labelX + width <= g.width;
}

export interface TieInflationArm {
	id: string;
	/** Frozen `point_metrics` join key. Never rendered — render `display`. */
	label: string;
	/** On-screen name; the paper's own Table name where the arm is theirs. */
	display: string;
	kind: PaperArmKind;
	/** True for the LLM reading arms only (excludes the CoGEx model bundle). */
	isReader: boolean;
	/** SHIPPED `pooled_average_precision` — the tie-robust lens we quote. */
	ap: number;
	/** SHIPPED `fold_mean_trapezoidal_pr_auc` — the paper's own estimator. */
	foldMeanTrapezoidal: number;
	/** SHIPPED `pooled_trapezoidal_pr_auc` — same estimator, pooled not folded. */
	pooledTrapezoidal: number;
	/** SHIPPED `trapezoidal_minus_ap_inflation`. Read, never recomputed. */
	inflation: number;
	/**
	 * `pooledTrapezoidal − ap`: the like-for-like check on the shipped headline,
	 * which mixes a fold-mean against a pooled number. Differencing two shipped
	 * scalars, not a re-derivation from predictions.
	 */
	sameEstimatorInflation: number;
	/** SHIPPED `distinct_scores` — the tie-ness axis. Read, never recomputed. */
	distinctScores: number;
}

export interface TiePoint {
	recall: number;
	precision: number;
}

/**
 * The single tied block that contributes the most interpolated area to the
 * featured reader arm: every statement in it carries the identical score, so no
 * threshold can separate them and the whole block is admitted at once.
 */
export interface TieBlock {
	/** The tied score itself. */
	score: number;
	size: number;
	nTrue: number;
	nFalse: number;
	/** Operating point immediately before the block is admitted. */
	from: TiePoint;
	/** Operating point immediately after — the block's whole cost lands here. */
	to: TiePoint;
	/** ΔR·ΔP/2, the triangle between the chord and the achievable step. */
	area: number;
	/** area / (pooledTrapezoidal − ap) for this arm. */
	shareOfArmInflation: number;
}

/** One arm's real, uncapped stepped PR geometry inside the featured window. */
export interface TieFeaturedArm {
	id: string;
	display: string;
	kind: PaperArmKind;
	/** Real PR vertices with recall ≥ the window start, ascending in recall. */
	vertices: TiePoint[];
	/** Σ triangles over the segments that lie wholly inside the window. */
	windowInflation: number;
	/** Σ triangles over the arm's whole curve == pooledTrapezoidal − ap. */
	totalInflation: number;
	/** |Σ triangles − (pooledTrapezoidal − ap)|; gated at TIE_RECONCILIATION_TOLERANCE. */
	reconciliationResidual: number;
	/** Achievable (stepped) precision at the window's mid-recall probe. */
	midStepPrecision: number;
	/** Precision the trapezoid's chord credits at that same recall. */
	midChordPrecision: number;
}

export interface TieFeatured {
	/** The reader arm with the largest shipped inflation. */
	reader: TieFeaturedArm;
	/** The paired-delta reference arm: the paper's own RF, same panel. */
	reference: TieFeaturedArm;
	block: TieBlock;
	/** Recall window both arms are drawn over: the block's own span. */
	recallFrom: number;
	recallTo: number;
	/** Precision extent over that window across BOTH arms, for a shared axis. */
	precisionMin: number;
	precisionMax: number;
	/** Recall at which the chord-vs-step probe is read. */
	midRecall: number;
}

/**
 * The featured reader's paired margin over the reference arm, on both
 * estimators. Both numbers are shipped `paired_delta_vs_paper_literal` deltas —
 * the same 1,689 statements, the same folds — so the pair is like-for-like and
 * the shrinkage between them is the interpolation being given back.
 */
export interface TieMargin {
	armDisplay: string;
	referenceDisplay: string;
	/** Δ on the paper's own fold-mean trapezoidal PR-AUC. */
	trapezoidal: number;
	/** Δ on tie-robust pooled average precision — the number we quote. */
	ap: number;
}

export interface TieInflationOk {
	status: 'ok';
	reason: null;
	artifact_path: string;
	/**
	 * NULLABLE on the ok branch too. The server always supplies a digest; the pure
	 * client-side validate path (the contract runner) does not, and coalescing that
	 * to '' printed an empty provenance line that read as a real, empty sha. Null
	 * is typed so the compiler finds every render site and each one says so.
	 */
	artifact_sha256: string | null;
	nStatements: number;
	/** Statements carrying the paper's released-correct label; null pre-geometry. */
	nPositives: number | null;
	arms: TieInflationArm[];
	/** Real curve geometry; null only on the pure client-side validate path. */
	featured: TieFeatured | null;
	margin: TieMargin | null;
	generatedNote: string;
}

export interface TieInflationUnavailable {
	status: 'unavailable';
	reason: string;
	artifact_path: string;
	artifact_sha256: string | null;
	nStatements: null;
	nPositives: null;
	arms: [];
	featured: null;
	margin: null;
	generatedNote: null;
}

export type TieInflationLoad = TieInflationOk | TieInflationUnavailable;

export interface TieInflationContext {
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

/** Signed finite number: inflation deltas legitimately go negative. */
function number(value: unknown, context: string): number {
	if (typeof value !== 'number' || !Number.isFinite(value)) fail(context, 'expected a finite number');
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

function parseArm(spec: (typeof PAPER_LITERAL_ARM_SPECS)[number], pointMetrics: UnknownRecord): TieInflationArm {
	const context = `point_metrics[${spec.label}]`;
	const point = record(pointMetrics[spec.label], context);
	const ap = unit(point.pooled_average_precision, `${context}.pooled_average_precision`);
	const pooledTrapezoidal = unit(
		point.pooled_trapezoidal_pr_auc,
		`${context}.pooled_trapezoidal_pr_auc`
	);
	return {
		id: spec.id,
		label: spec.label,
		display: spec.display,
		kind: spec.kind,
		isReader: spec.kind === 'llm' && spec.id !== NON_READER_LLM_ARM_ID,
		ap,
		foldMeanTrapezoidal: unit(
			point.fold_mean_trapezoidal_pr_auc,
			`${context}.fold_mean_trapezoidal_pr_auc`
		),
		pooledTrapezoidal,
		// SHIPPED. Deliberately not derived from the two fields above, even though
		// the artifact's own generator defines it as fold-mean minus pooled AP:
		// reading it keeps this panel pinned to the emitted number.
		inflation: number(
			point.trapezoidal_minus_ap_inflation,
			`${context}.trapezoidal_minus_ap_inflation`
		),
		sameEstimatorInflation: pooledTrapezoidal - ap,
		distinctScores: positiveInteger(point.distinct_scores, `${context}.distinct_scores`)
	};
}

/**
 * The arm the mechanism figure features: the READER arm whose shipped inflation
 * is largest — i.e. the arm the paper's own metric flatters most, which is also
 * (deliberately) our best-looking arm on that metric. Ties break on the id so the
 * choice is deterministic across reloads. Pure, so the choice is testable without
 * touching the filesystem.
 */
export function featuredReaderArm(arms: TieInflationArm[]): TieInflationArm | null {
	const readers = arms.filter((arm) => arm.isReader);
	if (readers.length === 0) return null;
	return readers
		.slice()
		.sort((a, b) => b.inflation - a.inflation || a.id.localeCompare(b.id))[0];
}

/** The paired-delta baseline: the paper's own RF, reproduced on this panel. */
export function tieReferenceArm(arms: TieInflationArm[]): TieInflationArm | null {
	return arms.find((arm) => arm.id === PAPER_LITERAL_REFERENCE_ARM_ID) ?? null;
}

/** Inclusive [min, max] of a field over a subset; null when the subset is empty. */
export function tieRange(
	arms: TieInflationArm[],
	pick: (arm: TieInflationArm) => number
): { min: number; max: number } | null {
	if (arms.length === 0) return null;
	const values = arms.map(pick);
	return { min: Math.min(...values), max: Math.max(...values) };
}

/**
 * Pure, fail-closed validator for the scalar half of the panel. Returns the eight
 * canonical arms with their shipped inflation/tie-ness scalars and the featured
 * reader's paired margin, or `status:'unavailable'` with a reason on any shape
 * drift. Curve geometry is left null for the server loader to fill. Never throws.
 */
export function validateTieInflation(
	raw: unknown,
	context: TieInflationContext = {}
): TieInflationLoad {
	const artifactPath = context.artifactPath ?? '';
	const artifactSha256 = context.artifactSha256 ?? null;
	try {
		const obj = record(raw, 'paper_literal_vs_llms');
		const nStatements = positiveInteger(obj.n_statements, 'paper_literal_vs_llms.n_statements');
		const pointMetrics = record(obj.point_metrics, 'paper_literal_vs_llms.point_metrics');
		const pairedDelta = record(
			obj.paired_delta_vs_paper_literal,
			'paper_literal_vs_llms.paired_delta_vs_paper_literal'
		);

		const arms = PAPER_LITERAL_ARM_SPECS.map((spec) => parseArm(spec, pointMetrics));
		const reader = featuredReaderArm(arms);
		const reference = tieReferenceArm(arms);
		if (!reader) fail('paper_literal_vs_llms.point_metrics', 'no reader arm is present');
		if (!reference) {
			fail('paper_literal_vs_llms.point_metrics', 'the paired-delta reference arm is missing');
		}

		const deltaContext = `paired_delta_vs_paper_literal[${reader.label}]`;
		const delta = record(pairedDelta[reader.label], deltaContext);
		const trapDelta = record(
			delta.fold_mean_trapezoidal_pr_auc,
			`${deltaContext}.fold_mean_trapezoidal_pr_auc`
		);
		const apDelta = record(
			delta.pooled_average_precision,
			`${deltaContext}.pooled_average_precision`
		);

		return {
			status: 'ok',
			reason: null,
			artifact_path: artifactPath,
			artifact_sha256: artifactSha256,
			nStatements,
			nPositives: null,
			arms,
			featured: null,
			margin: {
				armDisplay: reader.display,
				referenceDisplay: reference.display,
				trapezoidal: number(trapDelta.delta, `${deltaContext}.fold_mean_trapezoidal_pr_auc.delta`),
				ap: number(apDelta.delta, `${deltaContext}.pooled_average_precision.delta`)
			},
			generatedNote: `Trapezoidal-vs-average-precision tie inflation over ${nStatements} all-sources-specific statements.`
		};
	} catch (error) {
		return {
			status: 'unavailable',
			reason: error instanceof Error ? error.message : String(error),
			artifact_path: artifactPath,
			artifact_sha256: artifactSha256,
			nStatements: null,
			nPositives: null,
			arms: [],
			featured: null,
			margin: null,
			generatedNote: null
		};
	}
}
