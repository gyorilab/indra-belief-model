import { boolean, budget, fail, number, positiveInteger, record, text, unit } from './paper-validate.ts';
/**
 * Typed data contract for the ROBUSTNESS surface: how the head-to-head margin
 * behaves under two standard checks that a pointwise interval cannot answer.
 *
 * WHAT IS PRIMARY. The 1689-statement panel with the 2023 RELEASED labels
 * (`paper_replication_policy.released_paper_correct`, unmodified) is the primary
 * result everywhere on this surface. It is the result the head-to-head ships and
 * the one the page leads with; nothing here replaces it, softens it, or is
 * substituted for it.
 *
 * WHAT IS ADDED. Two clearly-labelled robustness views of that same margin:
 *
 *  (A) A SIMULTANEOUS band over the four reader arms. `data/comparison/run_plan
 *      .json` was frozen before the comparison was generated and stages all four
 *      arms without designating one as confirmatory, so we cannot claim a
 *      pre-registered single arm and a family-wise band is a fair ask. It is a
 *      studentized max-t band computed from the SAME paired-bootstrap draws, so
 *      the arms' correlation is carried rather than assumed away — which is why
 *      it costs ~2.30 standard errors instead of Bonferroni's ~2.50.
 *
 *  (B) A LABEL-COMPLETENESS sensitivity on the 1578 statements whose labels are
 *      adjudication-safe. The 111 dropped statements are NEGATIVE in the labels
 *      released in 2023, so dropping them is OUR revision of those. It is a
 *      sensitivity check, never the primary result, and it is not free: it removes
 *      a quarter of all negatives and moves the panel from 26.8% to 21.6%
 *      negative, changing what the panel is a sample of.
 *
 * SHIPPED FIELDS ARE READ, NEVER RECOMPUTED. Every delta, interval, band bound,
 * critical value and census count comes off
 * `data/results/indra_paper_literal_models_20260724/paper_margin_robustness.json`.
 * The only arithmetic here is the ×100 conversion into AP points for the drawing
 * (the same unit `paper-ap-decomposition.ts` draws in) and the consistency gates
 * below, which compare shipped fields against each other and fail closed.
 *
 * This module is import-safe on the client: typed shape plus a pure, fail-closed
 * validator. All filesystem work lives in `$lib/server/paper-robustness`.
 */

import {
	PAPER_LITERAL_ARM_SPECS,
	standingOfBounds,
	type ShippedProse,
	type Standing
} from './paper-literal.ts';

/**
 * THE PLAIN HALF OF EVERY TWIN THIS MODULE EMITS.
 *
 * `paper_margin_robustness.json` is where "paired fold-stratified bootstrap over
 * the paper's own out-of-fold fold assignment" reaches the screen, and where the
 * headline measure introduces itself as "pooled average precision". Both are
 * runtime strings off a sha-pinned file, so they are translated here rather than
 * rewritten there. Every restatement keeps the whole of its sentence: the
 * sensitivity check is still OUR revision of the published labels, and still
 * never the headline. A restatement also drops the possessive that names a
 * published method — the 2023 paper is this lab's own prior work, so "the
 * paper's folds" is "the 10 folds" and nothing is lost by dropping the owner.
 *
 * FOLD, OUT-OF-FOLD and CROSS-VALIDATION SURVIVE TRANSLATION. They are the
 * field's terms, not this page's, and the audience published on them; an earlier
 * pass replaced them with "slice", which taught nobody anything. What a
 * restatement owes the reader is a GLOSS at first use, and `referenceDescription`
 * carries it — it is the first of these strings the robustness table prints, and
 * it is also the only place the fitted/reading asymmetry can be stated beside the
 * estimator it applies to.
 */
const ROBUSTNESS_PLAIN = {
	metric:
		'Average precision, computed over all 1,689 statements at once rather than fold by ' +
		'fold (scikit-learn’s average_precision_score).',
	bootstrapDesign:
		'Statements are re-drawn at random with replacement WITHIN each of the 10 folds assigned ' +
		'in the 2023 paper, so every redraw keeps that fold make-up. ONE redraw is ' +
		'shared by every model, and each margin is that model minus the re-run random forest on ' +
		'the SAME redraw. Imported from scripts/compare_paper_literal_vs_llms.py.',
	multiplicityNote:
		'The frozen run plan lists all four reading models and names none of them as the one to ' +
		'be confirmed, so we cannot claim a single model was nominated in advance, and the ' +
		'interval has to be wide enough to cover all four at once. The four ' +
		'order the statements very similarly, so covering all four costs far less width than a ' +
		'Bonferroni correction would.',
	labelCompletenessNote:
		'These 111 statements are labelled WRONG in the labels released in 2023. Dropping them is ' +
		'OUR revision of those published labels, so this is a check to the side and never the ' +
		'headline result. It also removes a quarter of all the wrong statements and shifts the mix ' +
		'of right and wrong, so it changes what this set of statements is a sample OF, not only how ' +
		'good its labels are.',
	doseResponseBasis:
		'Gemma 4 E2B is the edge variant of its family and the smallest model here; the other ' +
		'three are larger. No finer size ordering is claimed — GLM-5’s parameter count is not ' +
		'published. The checkable part is the sign pattern: exactly one model is negative, and it ' +
		'is the smallest.',
	// Opens lower-case on purpose: the robustness table sets it after an em dash,
	// as the continuation of "every margin is measured against <name> — …".
	referenceDescription:
		'the random-forest code released with the 2023 INDRA assembly paper ' +
		'(sorgerlab/indra_assembly_paper), re-run here under 10-fold cross-validation. ' +
		// THE GLOSS, AND WHY IT IS ONE UNBROKEN LINE. This is the first of these
		// sentences the robustness table prints, so it is where `fold` is defined for
		// that surface — and the guard's definition test cannot cross a full stop,
		// which the harvest inserts between concatenated chunks. Split this clause
		// across two chunks and the gloss stops counting as one.
		'The folds are the 10 groups the statements were split into, and every statement is ' +
		'scored out-of-fold — by a copy of the forest that never trained on it. Nothing is held ' +
		'back as a separate test set. The reading models are never trained and never see a label, ' +
		'so nothing has to be held back from them either: each is scored once over all 1,689 ' +
		'statements and then given the same fold numbers, purely so the identical estimator ' +
		'applies to both sides.',
	/**
	 * `multiplicity.method` was twinned nowhere while its sibling `note` was, so
	 * the one line saying HOW one interval is made to cover four models reached a
	 * reader as "studentized max-t over the shared paired-bootstrap draws".
	 */
	multiplicityMethod:
		'The width is set by the largest standardised margin seen across the four reading models ' +
		'on each shared redraw, so ONE interval covers all four at once.',
	primaryLabelProvenance: 'the labels released with the 2023 paper, unmodified',
	sensitivityLabelProvenance:
		'the labels released with the 2023 paper, with 111 statements REMOVED by us — every one ' +
		'of them labelled wrong in those released labels, and every one with an evidence review ' +
		'that was never finished'
} as const;

/**
 * On-screen names, keyed by arm id. DECOUPLED from the artifact's `label`, which
 * is a FROZEN `point_metrics` join key addressing already-emitted data. Render
 * `display`, never `label` — that invariant has leaked five times on this page,
 * so `validatePaperRobustness` additionally checks each artifact `label` against
 * the canonical spec for the same id and gates on a mismatch.
 */
export const PAPER_ROBUSTNESS_DISPLAY: Readonly<Record<string, string>> = {
	'paper-rf-promoter': 'RF 2k-d13 + Type/#PMIDs/promoter',
	'gemma-4-26b': 'Gemma 4 26B gate',
	'glm-5': 'GLM-5 gate',
	'gemma-4-31b': 'Gemma 4 31B gate',
	'gemma-4-e2b': 'Gemma 4 E2B gate'
};

/** The artifact kind this module will accept, and nothing else. */
export const PAPER_ROBUSTNESS_ARTIFACT_KIND = 'paper_margin_robustness';

/**
 * SVG geometry. Exported so the label budgets below are derived from it rather
 * than eyeballed: right-anchored SVG text that overruns its gutter loses its
 * LEADING glyphs silently — no layout error, no test failure, and `<desc>` hides
 * it from review. Copied in shape from `paper-own-metric.ts`, which does this
 * correctly.
 */
export const PAPER_ROBUSTNESS_GEOMETRY = {
	width: 920,
	plotLeft: 162,
	plotRight: 800,
	/** Lane labels are right-anchored here; usable gutter is 0 → 150 units. */
	labelAnchorX: 150,
	/** Readouts are left-anchored here; usable gutter is 808 → 920. */
	readoutX: 808,
	labelFontPx: 9,
	readoutFontPx: 8,
	/** Measured advance of the mono face at 9px, in user units per character. */
	monoUnitsPerChar: 5.4186,
	/** Measured advance of the mono face at 8px. */
	readoutUnitsPerChar: 4.8165,
	laneHeight: 54,
	/** Row offsets inside a lane: the primary result above, sensitivity below. */
	primaryOffset: -10,
	sensitivityOffset: 13,
	topPad: 34,
	axisPad: 56,
	/** Half-height of an interval's end cap. */
	simultaneousCap: 5,
	pointwiseCap: 3.5,
	sensitivityCap: 2.5
} as const;

/**
 * LANE LABEL BUDGET: 150 units ÷ 5.4186 u/char at 9px = 27.6 → 27 characters.
 * The longest label the artifact can produce is "Gemma 4 26B reading" at 19
 * characters (103.0 units), leaving 47 units of slack. The reference model's own
 * name, "RF 2k-d13 + Type/#PMIDs/promoter" (32 chars), is NOT a lane label — it
 * annotates the zero rule, whose fit is measured separately in the builder.
 * `buildPaperRobustness` FAILS if any lane label exceeds this, so a renamed arm
 * gates the figure instead of quietly eating its first glyphs.
 */
export const PAPER_ROBUSTNESS_LABEL_BUDGET_CHARS = 27;

/**
 * READOUT BUDGET: (920 − 808) units ÷ 4.8165 u/char at 8px = 23.2 → 23
 * characters. Longest shipped readout is the primary line, "+0.98 pts  P(>0)
 * 0.986" at 22 characters. Left-anchored, so an overrun clips the TRAILING
 * glyphs; budgeted and enforced all the same.
 */
export const PAPER_ROBUSTNESS_READOUT_BUDGET_CHARS = 23;

/**
 * The zero rule's annotation is free-floating text at 8px, so its fit is
 * measured against the right edge and it flips to the other side rather than
 * clip. Same device as `PaperOwnMetric.svelte`'s reference label.
 */
export const PAPER_ROBUSTNESS_ZERO_LABEL_PAD = 6;

export type PaperRobustnessSeriesId = 'pointwise' | 'simultaneous' | 'sensitivity';

export const PAPER_ROBUSTNESS_SERIES_IDS: readonly PaperRobustnessSeriesId[] = [
	'pointwise',
	'simultaneous',
	'sensitivity'
] as const;

export interface PaperRobustnessSeriesStyle {
	id: PaperRobustnessSeriesId;
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
 */
export const PAPER_ROBUSTNESS_SERIES: Record<
	PaperRobustnessSeriesId,
	PaperRobustnessSeriesStyle
> = {
	pointwise: {
		id: 'pointwise',
		strokeVar: 'var(--accent)',
		dash: '',
		strokeWidth: 2.6,
		shape: 'diamond',
		legend: 'PRIMARY — 1689 statements, the labels released in 2023 unmodified; point estimate and pointwise 95% interval'
	},
	simultaneous: {
		id: 'simultaneous',
		strokeVar: 'var(--blocked)',
		dash: '4 2',
		strokeWidth: 1.4,
		shape: 'bracket',
		legend:
			'the same 1689 statements, with the interval widened to cover all four reading models at once — the price of having run four and not one'
	},
	sensitivity: {
		id: 'sensitivity',
		strokeVar: 'var(--ink-muted)',
		dash: '1.5 1.5',
		strokeWidth: 1.2,
		shape: 'open-circle',
		legend: 'SENSITIVITY — our label revision: the 1578 adjudication-safe statements'
	}
};

/** One interval, already converted to AP points (ΔAP × 100) for drawing. */
export interface PaperRobustnessInterval {
	deltaPts: number;
	lowPts: number;
	highPts: number;
	/**
	 * ahead / behind / not-significant, from this interval's own endpoints via the
	 * page's one classifier. It replaces a sign-blind `excludesZero` boolean; see
	 * the block above `standingOfBounds` in paper-literal.ts for why that boolean
	 * no longer exists anywhere on /paper.
	 */
	standing: Standing;
}

/**
 * The part of an interval that lies on the far side of zero from its own point
 * estimate — i.e. the amount by which the interval fails to exclude zero, drawn
 * as a filled block against the zero rule. It is what separates "grazes zero"
 * from "does not clear zero" at a glance, and it is a SUBSET of its own series
 * rather than a new one, so it introduces no new hue. Null when the interval
 * excludes zero outright.
 */
export interface PaperRobustnessAdverse {
	fromPts: number;
	toPts: number;
}

function adverseOf(interval: PaperRobustnessInterval): PaperRobustnessAdverse | null {
	// Drawn only when the interval straddles zero. "Significant either way" is the
	// question here, and asking it as `!== 'not-significant'` says so; the old
	// `excludesZero` said it in a word that reads as a direction.
	if (interval.standing !== 'not-significant') return null;
	return interval.deltaPts >= 0
		? { fromPts: Math.min(interval.lowPts, 0), toPts: 0 }
		: { fromPts: 0, toPts: Math.max(interval.highPts, 0) };
}

export interface PaperRobustnessLane {
	id: string;
	/** FROZEN artifact join key. Never rendered. */
	label: string;
	/** On-screen name. Rendered. */
	display: string;
	y: number;
	/** Baseline y for the primary row (point + pointwise + simultaneous). */
	primaryY: number;
	/** Baseline y for the sensitivity row. */
	sensitivityY: number;
	pointwise: PaperRobustnessInterval;
	simultaneous: PaperRobustnessInterval;
	sensitivity: PaperRobustnessInterval;
	/** How far the simultaneous band reaches past zero; null when it excludes zero. */
	simultaneousAdverse: PaperRobustnessAdverse | null;
	/** Bootstrap P(delta > 0) on the primary panel — shipped, not derived. */
	pGreaterThanZero: number;
	sensitivityPGreaterThanZero: number;
	/** Raw AP-scale values, for the inspect table (never for the drawing). */
	primaryAp: number;
	sensitivityAp: number;
	primaryDelta: number;
	sensitivityDelta: number;
	bootstrapSe: number;
	readoutPrimary: string;
	readoutSensitivity: string;
	title: string;
}

export interface PaperRobustnessPanel {
	id: string;
	role: string;
	nStatements: number;
	nPositive: number;
	nNegative: number;
	negativeFraction: number;
	referenceAp: number;
	labelField: string;
	labelProvenance: string;
	/** `labelProvenance` with its plain restatement — `shipped` is byte-identical. */
	labelProvenanceProse: ShippedProse;
	isOurLabelRevision: boolean;
}

export interface PaperRobustnessLabelCompleteness {
	field: string;
	nDropped: number;
	droppedShareOfAllNegatives: number;
	negativeFractionBefore: number;
	negativeFractionAfter: number;
	allDroppedAreNegative: boolean;
	allDroppedHaveIncompleteEvidenceReview: boolean;
	noModelIsRefit: boolean;
	note: string;
	/** `note` with its plain restatement — `noteProse.shipped === note`. */
	noteProse: ShippedProse;
}

export interface PaperRobustnessMultiplicity {
	familySize: number;
	method: string;
	/** `method` with its plain restatement — `methodProse.shipped === method`. */
	methodProse: ShippedProse;
	familyAlpha: number;
	criticalValue: number;
	pointwiseNormalCriticalValue: number;
	bonferroniCriticalValue: number;
	scoreSpearmanMin: number;
	scoreSpearmanMax: number;
	noDesignatedPrimaryArm: boolean;
	runPlanPath: string;
	runPlanSha256: string;
	runPlanFrozenAt: string;
	runPlanStages: string[];
	note: string;
	/** `note` with its plain restatement — `noteProse.shipped === note`. */
	noteProse: ShippedProse;
}

export interface PaperRobustnessDoseResponse {
	smallestArmId: string;
	smallestArmDisplay: string;
	nNegative: number;
	nPositive: number;
	onlyNegativeArmIsTheSmallest: boolean;
	basis: string;
	/** `basis` with its plain restatement — `basisProse.shipped === basis`. */
	basisProse: ShippedProse;
}

export interface PaperRobustnessFigure {
	lanes: PaperRobustnessLane[];
	domainMinPts: number;
	domainMaxPts: number;
	ticksPts: number[];
	height: number;
	/** Text annotating the zero rule, and whether it fits to its right. */
	zeroLabel: string;
	zeroLabelFits: boolean;
	referenceDisplay: string;
	referenceDescription: string;
	primaryPanel: PaperRobustnessPanel;
	sensitivityPanel: PaperRobustnessPanel;
	labelCompleteness: PaperRobustnessLabelCompleteness;
	multiplicity: PaperRobustnessMultiplicity;
	doseResponse: PaperRobustnessDoseResponse;
	/** Widest pointwise half-width on the primary panel, in AP points. */
	worstHalfWidthPts: number;
	/** Largest positive primary delta, in AP points — the effect being measured. */
	bestDeltaPts: number;
	metric: string;
	seed: number;
	nBootstrap: number;
	bootstrapDesign: string;
	/**
	 * The plain half of every string this figure took off the artifact. Each flat
	 * field is byte-identical to its twin's `shipped`; the twin is the one to
	 * render, and the shipped half belongs behind the verification boundary. The
	 * four that live inside a structure — the two notes, the dose-response basis and
	 * each panel's label provenance — are the SAME objects reachable there, not
	 * copies.
	 */
	prose: PaperRobustnessProse;
	/** Worst |recomputed − shipped| across every reproduced pointwise field. */
	shippedResidual: number;
	shippedTolerance: number;
}

/** Every shipped sentence this figure carries, each with its plain restatement. */
export interface PaperRobustnessProse {
	metric: ShippedProse;
	bootstrapDesign: ShippedProse;
	referenceDescription: ShippedProse;
	multiplicityNote: ShippedProse;
	/** How ONE interval is made to cover all four reading models. */
	multiplicityMethod: ShippedProse;
	labelCompletenessNote: ShippedProse;
	doseResponseBasis: ShippedProse;
	primaryLabelProvenance: ShippedProse;
	sensitivityLabelProvenance: ShippedProse;
}

export interface PaperRobustnessOk {
	status: 'ok';
	figure: PaperRobustnessFigure;
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

export interface PaperRobustnessUnavailable {
	status: 'unavailable';
	figure: null;
	reason: string;
	artifact_path: string;
	artifact_sha256: string | null;
}

export type PaperRobustnessLoad = PaperRobustnessOk | PaperRobustnessUnavailable;

export interface PaperRobustnessContext {
	artifactPath?: string;
	artifactSha256?: string;
}

type UnknownRecord = Record<string, unknown>;








/** Exactly `want`, or the figure gates. Used where a false value would relabel
 *  the sensitivity panel as the primary one. */
function exactly(value: unknown, want: boolean, context: string): boolean {
	const parsed = boolean(value, context);
	if (parsed !== want) fail(context, `expected ${want}`);
	return parsed;
}

function pts(value: number): number {
	return value * 100;
}

/**
 * Signed AP points at two decimals. ASCII '+'/'-' on purpose: the readout budget
 * is measured in characters against a MEASURED mono advance, and a typographic
 * minus that fell back to another face would be a different width — silently
 * breaking the one guard that keeps this text from clipping.
 */
function fmtPts(value: number): string {
	return `${value >= 0 ? '+' : '-'}${Math.abs(value).toFixed(2)}`;
}

/** Three decimals: the precision this page prints probabilities at. */
export function fmt3(value: number): string {
	return value.toFixed(3);
}


function parsePanel(raw: unknown, context: string, expectRevision: boolean): PaperRobustnessPanel {
	const panel = record(raw, context);
	// The two panels ship DIFFERENT provenance sentences, so the restatement is
	// selected by the same flag that decides which panel this is — the one the
	// figure already gates on. There is no third case to fall through to.
	const labelProvenanceProse: ShippedProse = {
		shipped: text(panel.label_provenance, `${context}.label_provenance`),
		plain: expectRevision
			? ROBUSTNESS_PLAIN.sensitivityLabelProvenance
			: ROBUSTNESS_PLAIN.primaryLabelProvenance
	};
	const n = positiveInteger(panel.n_statements, `${context}.n_statements`);
	const nPositive = positiveInteger(panel.n_positive, `${context}.n_positive`);
	const nNegative = positiveInteger(panel.n_negative, `${context}.n_negative`);
	if (nPositive + nNegative !== n) {
		fail(context, 'n_positive + n_negative must equal n_statements');
	}
	return {
		id: text(panel.id, `${context}.id`),
		role: text(panel.role, `${context}.role`),
		nStatements: n,
		nPositive,
		nNegative,
		negativeFraction: unit(panel.negative_fraction, `${context}.negative_fraction`),
		referenceAp: unit(
			panel.reference_average_precision,
			`${context}.reference_average_precision`
		),
		labelField: text(panel.label_field, `${context}.label_field`),
		labelProvenance: labelProvenanceProse.shipped,
		labelProvenanceProse,
		// The framing is load-bearing, so it is GATED rather than displayed: if the
		// artifact ever stops calling the 1578 panel our own label revision, the
		// figure disappears instead of presenting a revision as the released data.
		isOurLabelRevision: exactly(
			panel.is_our_label_revision,
			expectRevision,
			`${context}.is_our_label_revision`
		)
	};
}

interface ParsedSide {
	ap: number;
	delta: number;
	ciLow: number;
	ciHigh: number;
	simLow: number;
	simHigh: number;
	pGreater: number;
	se: number;
	/**
	 * The shipped `excludes_zero_*` flags are still READ and still gated against
	 * their own endpoints below — an artifact whose flag disagrees with its bounds
	 * is corrupt and must gate. They are simply not carried forward: what leaves
	 * this parser is the three-way class, so no consumer can branch two ways on a
	 * sign-blind boolean.
	 */
	pointwiseStanding: Standing;
	simultaneousStanding: Standing;
}

function parseSide(raw: unknown, context: string): ParsedSide {
	const side = record(raw, context);
	const ciLow = number(side.ci95_low, `${context}.ci95_low`);
	const ciHigh = number(side.ci95_high, `${context}.ci95_high`);
	if (ciLow > ciHigh) fail(context, 'ci95_low must not exceed ci95_high');
	const simLow = number(side.simultaneous_low, `${context}.simultaneous_low`);
	const simHigh = number(side.simultaneous_high, `${context}.simultaneous_high`);
	if (simLow > simHigh) fail(context, 'simultaneous_low must not exceed simultaneous_high');
	// The drawing nests the pointwise interval INSIDE the simultaneous band, and
	// that nesting is the whole visual argument ("correcting for four arms widens
	// it"). A band that failed to contain its own pointwise interval would draw a
	// lie, so it gates instead.
	if (simLow > ciLow || simHigh < ciHigh) {
		fail(context, 'the simultaneous band must contain the pointwise interval');
	}
	// Both shipped flags are still read and still gated against their own endpoints
	// — an artifact whose flag disagrees with its bounds is corrupt and must gate.
	// Neither is bound to a name here: a local `excludesPointwise` is the same
	// sign-blind boolean one scope down, and it would be the obvious thing to pass
	// on next time this parser grows a field.
	if (
		boolean(side.excludes_zero_pointwise, `${context}.excludes_zero_pointwise`) !==
		(ciLow > 0 || ciHigh < 0)
	) {
		fail(`${context}.excludes_zero_pointwise`, 'must equal ci95_low > 0 || ci95_high < 0');
	}
	if (
		boolean(side.excludes_zero_simultaneous, `${context}.excludes_zero_simultaneous`) !==
		(simLow > 0 || simHigh < 0)
	) {
		fail(
			`${context}.excludes_zero_simultaneous`,
			'must equal simultaneous_low > 0 || simultaneous_high < 0'
		);
	}
	return {
		ap: unit(side.average_precision, `${context}.average_precision`),
		delta: number(side.delta, `${context}.delta`),
		ciLow,
		ciHigh,
		simLow,
		simHigh,
		pGreater: unit(side.p_greater_than_zero, `${context}.p_greater_than_zero`),
		se: number(side.bootstrap_se, `${context}.bootstrap_se`),
		pointwiseStanding: standingOfBounds(ciLow, ciHigh),
		simultaneousStanding: standingOfBounds(simLow, simHigh)
	};
}

/** Ticks every half AP point across the domain; zero is always among them. */
function ticksOf(minPts: number, maxPts: number): number[] {
	const out: number[] = [];
	// Integer halves throughout: float accumulation would drift the labels.
	for (let half = Math.ceil(minPts * 2); half <= Math.floor(maxPts * 2); half += 1) {
		out.push(half / 2);
	}
	if (!out.some((tick) => tick === 0)) fail('domain', 'the zero tick is outside the axis');
	return out;
}

/**
 * Build the drawable figure from a validated artifact. Throws on any drift — an
 * unknown arm id, an over-budget label, a domain that excludes zero — so the
 * caller gates to `unavailable` rather than rendering geometry that is silently
 * wrong.
 */
function buildFigure(raw: UnknownRecord): PaperRobustnessFigure {
	const kind = text(raw.artifact_kind, 'artifact_kind');
	if (kind !== PAPER_ROBUSTNESS_ARTIFACT_KIND) {
		fail('artifact_kind', `expected ${PAPER_ROBUSTNESS_ARTIFACT_KIND}, got ${kind}`);
	}

	const metricProse: ShippedProse = {
		shipped: text(raw.metric, 'metric'),
		plain: ROBUSTNESS_PLAIN.metric
	};
	const bootstrapDesignProse: ShippedProse = {
		shipped: text(raw.bootstrap_design, 'bootstrap_design'),
		plain: ROBUSTNESS_PLAIN.bootstrapDesign
	};

	const reference = record(raw.reference, 'reference');
	const referenceDescriptionProse: ShippedProse = {
		shipped: text(reference.description, 'reference.description'),
		plain: ROBUSTNESS_PLAIN.referenceDescription
	};
	const referenceId = text(reference.id, 'reference.id');
	const referenceDisplay = PAPER_ROBUSTNESS_DISPLAY[referenceId];
	if (referenceDisplay === undefined) {
		fail('reference.id', `no canonical display name for "${referenceId}"`);
	}

	const panels = record(raw.panels, 'panels');
	const primaryPanel = parsePanel(panels.primary, 'panels.primary', false);
	const sensitivityPanel = parsePanel(panels.sensitivity, 'panels.sensitivity', true);
	if (sensitivityPanel.nStatements >= primaryPanel.nStatements) {
		fail('panels.sensitivity', 'the sensitivity panel must be a strict subset of the primary');
	}
	if (sensitivityPanel.nPositive !== primaryPanel.nPositive) {
		fail('panels.sensitivity', 'the label-completeness check drops negatives only');
	}

	const completenessRaw = record(raw.label_completeness, 'label_completeness');
	const labelCompletenessNoteProse: ShippedProse = {
		shipped: text(completenessRaw.note, 'label_completeness.note'),
		plain: ROBUSTNESS_PLAIN.labelCompletenessNote
	};
	const labelCompleteness: PaperRobustnessLabelCompleteness = {
		field: text(completenessRaw.field, 'label_completeness.field'),
		nDropped: positiveInteger(completenessRaw.n_dropped, 'label_completeness.n_dropped'),
		droppedShareOfAllNegatives: unit(
			completenessRaw.dropped_share_of_all_negatives,
			'label_completeness.dropped_share_of_all_negatives'
		),
		negativeFractionBefore: unit(
			completenessRaw.negative_fraction_before,
			'label_completeness.negative_fraction_before'
		),
		negativeFractionAfter: unit(
			completenessRaw.negative_fraction_after,
			'label_completeness.negative_fraction_after'
		),
		// All three are asserted by the compute script; gating on them here means a
		// weakened artifact takes the figure down rather than quietly changing what
		// "adjudication-safe" is claimed to mean.
		allDroppedAreNegative: exactly(
			completenessRaw.all_dropped_are_negative,
			true,
			'label_completeness.all_dropped_are_negative'
		),
		allDroppedHaveIncompleteEvidenceReview: exactly(
			completenessRaw.all_dropped_have_incomplete_evidence_review,
			true,
			'label_completeness.all_dropped_have_incomplete_evidence_review'
		),
		noModelIsRefit: exactly(
			completenessRaw.no_model_is_refit,
			true,
			'label_completeness.no_model_is_refit'
		),
		note: labelCompletenessNoteProse.shipped,
		noteProse: labelCompletenessNoteProse
	};
	if (labelCompleteness.nDropped !== primaryPanel.nStatements - sensitivityPanel.nStatements) {
		fail('label_completeness.n_dropped', 'must equal the difference in panel sizes');
	}
	if (labelCompleteness.nDropped !== primaryPanel.nNegative - sensitivityPanel.nNegative) {
		fail('label_completeness.n_dropped', 'must equal the difference in negative counts');
	}

	const multiplicityRaw = record(raw.multiplicity, 'multiplicity');
	const multiplicityNoteProse: ShippedProse = {
		shipped: text(multiplicityRaw.note, 'multiplicity.note'),
		plain: ROBUSTNESS_PLAIN.multiplicityNote
	};
	const runPlan = record(multiplicityRaw.run_plan, 'multiplicity.run_plan');
	const family = multiplicityRaw.family;
	if (!Array.isArray(family)) fail('multiplicity.family', 'expected an array');
	const runPlanStages = runPlan.stages;
	if (!Array.isArray(runPlanStages) || runPlanStages.length === 0) {
		fail('multiplicity.run_plan.stages', 'expected a non-empty array');
	}
	const criticalValue = number(multiplicityRaw.critical_value, 'multiplicity.critical_value');
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
		fail('multiplicity.critical_value', 'a simultaneous band cannot be narrower than pointwise');
	}
	if (!(criticalValue <= bonferroniZ)) {
		fail('multiplicity.critical_value', 'max-t cannot exceed the Bonferroni critical value');
	}
	const multiplicityMethodProse: ShippedProse = {
		shipped: text(multiplicityRaw.method, 'multiplicity.method'),
		plain: ROBUSTNESS_PLAIN.multiplicityMethod
	};
	const multiplicity: PaperRobustnessMultiplicity = {
		familySize: positiveInteger(multiplicityRaw.family_size, 'multiplicity.family_size'),
		method: multiplicityMethodProse.shipped,
		methodProse: multiplicityMethodProse,
		familyAlpha: unit(multiplicityRaw.family_alpha, 'multiplicity.family_alpha'),
		criticalValue,
		pointwiseNormalCriticalValue: pointwiseZ,
		bonferroniCriticalValue: bonferroniZ,
		scoreSpearmanMin: number(
			multiplicityRaw.score_spearman_min,
			'multiplicity.score_spearman_min'
		),
		scoreSpearmanMax: number(
			multiplicityRaw.score_spearman_max,
			'multiplicity.score_spearman_max'
		),
		noDesignatedPrimaryArm: exactly(
			multiplicityRaw.no_designated_primary_arm,
			true,
			'multiplicity.no_designated_primary_arm'
		),
		runPlanPath: text(runPlan.path, 'multiplicity.run_plan.path'),
		runPlanSha256: text(runPlan.sha256, 'multiplicity.run_plan.sha256'),
		runPlanFrozenAt: text(runPlan.frozen_at, 'multiplicity.run_plan.frozen_at'),
		runPlanStages: runPlanStages.map((stage, index) =>
			text(stage, `multiplicity.run_plan.stages[${index}]`)
		),
		note: multiplicityNoteProse.shipped,
		noteProse: multiplicityNoteProse
	};

	const armsRaw = raw.arms;
	if (!Array.isArray(armsRaw) || armsRaw.length === 0) fail('arms', 'expected a non-empty array');
	if (armsRaw.length !== multiplicity.familySize) {
		fail('arms', 'the drawn arms must be exactly the multiplicity family');
	}
	if (armsRaw.length !== family.length) {
		fail('multiplicity.family', 'must name exactly the drawn arms');
	}

	const specByArmId = new Map(PAPER_LITERAL_ARM_SPECS.map((spec) => [spec.id, spec]));
	const geometry = PAPER_ROBUSTNESS_GEOMETRY;
	let y = geometry.topPad;
	const lanes: PaperRobustnessLane[] = [];
	const extents: number[] = [];

	armsRaw.forEach((entry, index) => {
		const context = `arms[${index}]`;
		const arm = record(entry, context);
		const id = text(arm.id, `${context}.id`);
		const label = text(arm.label, `${context}.label`);
		const display = PAPER_ROBUSTNESS_DISPLAY[id];
		if (display === undefined) fail(`${context}.id`, `no canonical display name for "${id}"`);
		// Join-key parity: the artifact's frozen label must still be the label the
		// rest of the page joins this arm on. A rename on either side gates.
		const spec = specByArmId.get(id);
		if (!spec) fail(`${context}.id`, `"${id}" is not a canonical paper-literal arm`);
		if (spec.label !== label) {
			fail(`${context}.label`, `frozen join key drifted: "${label}" vs "${spec.label}"`);
		}

		const primary = parseSide(arm.primary, `${context}.primary`);
		const sensitivity = parseSide(arm.sensitivity, `${context}.sensitivity`);

		const laneY = y + geometry.laneHeight / 2;
		y += geometry.laneHeight;

		const readoutPrimary = budget(
			`${fmtPts(pts(primary.delta))} pts  P(>0) ${fmt3(primary.pGreater)}`,
			PAPER_ROBUSTNESS_READOUT_BUDGET_CHARS,
			`${context}.readoutPrimary`
		);
		const readoutSensitivity = budget(
			`${sensitivityPanel.nStatements} · ${fmtPts(pts(sensitivity.delta))} pts`,
			PAPER_ROBUSTNESS_READOUT_BUDGET_CHARS,
			`${context}.readoutSensitivity`
		);

		const simultaneousInterval: PaperRobustnessInterval = {
			deltaPts: pts(primary.delta),
			lowPts: pts(primary.simLow),
			highPts: pts(primary.simHigh),
			standing: primary.simultaneousStanding
		};

		lanes.push({
			id,
			label,
			display: budget(display, PAPER_ROBUSTNESS_LABEL_BUDGET_CHARS, `${context}.display`),
			y: laneY,
			primaryY: laneY + geometry.primaryOffset,
			sensitivityY: laneY + geometry.sensitivityOffset,
			pointwise: {
				deltaPts: pts(primary.delta),
				lowPts: pts(primary.ciLow),
				highPts: pts(primary.ciHigh),
				standing: primary.pointwiseStanding
			},
			simultaneous: simultaneousInterval,
			sensitivity: {
				deltaPts: pts(sensitivity.delta),
				lowPts: pts(sensitivity.ciLow),
				highPts: pts(sensitivity.ciHigh),
				standing: sensitivity.pointwiseStanding
			},
			simultaneousAdverse: adverseOf(simultaneousInterval),
			pGreaterThanZero: primary.pGreater,
			sensitivityPGreaterThanZero: sensitivity.pGreater,
			primaryAp: primary.ap,
			sensitivityAp: sensitivity.ap,
			primaryDelta: primary.delta,
			sensitivityDelta: sensitivity.delta,
			bootstrapSe: primary.se,
			readoutPrimary,
			readoutSensitivity,
			title:
				`${display} — ${fmtPts(pts(primary.delta))} AP points vs ${referenceDisplay} on ` +
				`${primaryPanel.nStatements} statements with the labels published in 2023; pointwise 95% ` +
				`[${fmtPts(pts(primary.ciLow))}, ${fmtPts(pts(primary.ciHigh))}], and widened to cover all ` +
				`${multiplicity.familySize} reading models at once [${fmtPts(pts(primary.simLow))}, ${fmtPts(pts(primary.simHigh))}]; ` +
				`on our ${sensitivityPanel.nStatements}-statement label revision ${fmtPts(pts(sensitivity.delta))}`
		});

		extents.push(
			pts(primary.simLow),
			pts(primary.simHigh),
			pts(primary.ciLow),
			pts(primary.ciHigh),
			pts(sensitivity.ciLow),
			pts(sensitivity.ciHigh),
			0
		);
	});

	// Presentation order is a CONTRACT, not a convenience: the page's headline
	// sentence names the first lane as the best arm, so a re-emitted artifact that
	// arrived in another order would put a losing arm in that sentence. Gate on it.
	for (let i = 1; i < lanes.length; i += 1) {
		if (lanes[i].primaryDelta > lanes[i - 1].primaryDelta) {
			fail('arms', 'must be ordered by primary delta, descending');
		}
	}

	const doseRaw = record(raw.dose_response, 'dose_response');
	const doseResponseBasisProse: ShippedProse = {
		shipped: text(doseRaw.basis, 'dose_response.basis'),
		plain: ROBUSTNESS_PLAIN.doseResponseBasis
	};
	const smallestArmId = text(doseRaw.smallest_arm_id, 'dose_response.smallest_arm_id');
	const smallestArmDisplay = PAPER_ROBUSTNESS_DISPLAY[smallestArmId];
	if (smallestArmDisplay === undefined) {
		fail('dose_response.smallest_arm_id', `no canonical display name for "${smallestArmId}"`);
	}
	const negativeLanes = lanes.filter((lane) => lane.primaryDelta < 0);
	const doseResponse: PaperRobustnessDoseResponse = {
		smallestArmId,
		smallestArmDisplay,
		nNegative: positiveInteger(
			doseRaw.n_arms_with_negative_delta,
			'dose_response.n_arms_with_negative_delta'
		),
		nPositive: positiveInteger(
			doseRaw.n_arms_with_positive_delta,
			'dose_response.n_arms_with_positive_delta'
		),
		onlyNegativeArmIsTheSmallest: exactly(
			doseRaw.only_negative_arm_is_the_smallest,
			true,
			'dose_response.only_negative_arm_is_the_smallest'
		),
		basis: doseResponseBasisProse.shipped,
		basisProse: doseResponseBasisProse
	};
	// The dose-response sentence is a claim about the drawn arms, so it is checked
	// against them rather than trusted: if some other arm goes negative, the claim
	// is false and the figure gates rather than printing it.
	if (negativeLanes.length !== doseResponse.nNegative) {
		fail('dose_response.n_arms_with_negative_delta', 'disagrees with the drawn arms');
	}
	if (negativeLanes.length !== 1 || negativeLanes[0].id !== smallestArmId) {
		fail('dose_response', 'the only arm with a negative delta is no longer the smallest arm');
	}
	if (doseResponse.nNegative + doseResponse.nPositive !== lanes.length) {
		fail('dose_response', 'the sign census must cover every drawn arm');
	}

	const shipped = record(raw.shipped_reconciliation, 'shipped_reconciliation');
	const shippedResidual = number(
		shipped.worst_residual_vs_shipped,
		'shipped_reconciliation.worst_residual_vs_shipped'
	);
	const shippedTolerance = number(shipped.tolerance, 'shipped_reconciliation.tolerance');
	// The pointwise half of this figure claims to BE the shipped head-to-head
	// result. If the compute script could not reproduce it, that claim is false.
	if (!(shippedResidual <= shippedTolerance)) {
		fail(
			'shipped_reconciliation',
			`the recomputed pointwise result does not reproduce the shipped one ` +
				`(${shippedResidual} > ${shippedTolerance})`
		);
	}

	const powerRaw = record(raw.power, 'power');
	const halfWidths = record(powerRaw.pointwise_ci_half_width, 'power.pointwise_ci_half_width');
	const worstHalfWidthPts = Math.max(
		...lanes.map((lane) =>
			pts(number(halfWidths[lane.id], `power.pointwise_ci_half_width[${lane.id}]`))
		)
	);
	const bestDeltaPts = Math.max(...lanes.map((lane) => lane.pointwise.deltaPts));

	// Domain snapped outward to the HALF-point grid over everything DRAWN — the
	// same spacing as the ticks, so the extreme band never lands flush on the axis
	// start where its end cap would read as a crop rather than as a bound.
	const lo = Math.min(...extents);
	const hi = Math.max(...extents);
	const domainMinPts = Math.floor(lo * 2) / 2;
	const domainMaxPts = Math.ceil(hi * 2) / 2;
	if (!(domainMinPts < 0 && domainMaxPts > 0)) {
		fail('domain', 'the zero rule must sit strictly inside the axis');
	}

	const zeroLabel = `0 — no difference from ${referenceDisplay}`;
	const zeroAt =
		geometry.plotLeft +
		((0 - domainMinPts) / (domainMaxPts - domainMinPts)) * (geometry.plotRight - geometry.plotLeft);

	return {
		lanes,
		domainMinPts,
		domainMaxPts,
		ticksPts: ticksOf(domainMinPts, domainMaxPts),
		height: y + geometry.axisPad,
		zeroLabel,
		zeroLabelFits:
			zeroAt +
				PAPER_ROBUSTNESS_ZERO_LABEL_PAD +
				zeroLabel.length * geometry.readoutUnitsPerChar <=
			geometry.width,
		referenceDisplay,
		referenceDescription: referenceDescriptionProse.shipped,
		primaryPanel,
		sensitivityPanel,
		labelCompleteness,
		multiplicity,
		doseResponse,
		worstHalfWidthPts,
		bestDeltaPts,
		metric: metricProse.shipped,
		seed: positiveInteger(raw.seed, 'seed'),
		nBootstrap: positiveInteger(raw.n_bootstrap, 'n_bootstrap'),
		bootstrapDesign: bootstrapDesignProse.shipped,
		prose: {
			metric: metricProse,
			bootstrapDesign: bootstrapDesignProse,
			referenceDescription: referenceDescriptionProse,
			multiplicityNote: multiplicityNoteProse,
			multiplicityMethod: multiplicityMethodProse,
			labelCompletenessNote: labelCompletenessNoteProse,
			doseResponseBasis: doseResponseBasisProse,
			primaryLabelProvenance: primaryPanel.labelProvenanceProse,
			sensitivityLabelProvenance: sensitivityPanel.labelProvenanceProse
		},
		shippedResidual,
		shippedTolerance
	};
}

/**
 * Pure, fail-closed validator for the parsed `paper_margin_robustness.json`.
 * Returns `status:'ok'` with a drawable figure, or `status:'unavailable'` with a
 * reason on any shape or invariant drift. Never throws.
 */
export function validatePaperRobustness(
	raw: unknown,
	context: PaperRobustnessContext = {}
): PaperRobustnessLoad {
	const artifactPath = context.artifactPath ?? '';
	const artifactSha256 = context.artifactSha256 ?? null;
	try {
		return {
			status: 'ok',
			figure: buildFigure(record(raw, 'paper_margin_robustness')),
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
