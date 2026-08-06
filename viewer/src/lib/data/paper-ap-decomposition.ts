/**
 * Typed data contract for the AP-decomposition figure.
 *
 * Source artifact: `data/results/indra_paper_literal_models_20260724/
 * ap_decomposition_by_paper_band.json`, emitted by
 * `scripts/compute_paper_ap_decomposition.py`. Average precision decomposes with
 * no residual — each true statement contributes (precision at the cut-point that
 * includes its tie group) / n_true — so differencing those contributions against
 * the paper's literal RF+promoter reference and accumulating them across bands
 * reproduces each arm's shipped ΔAP exactly at the last band.
 *
 * BANDING. The bands are a power-of-two ladder on the number of EVIDENCE ENTRIES
 * a statement carries — an integer corpus census, fixed before any model ran.
 * The first version of this figure banded by the reference arm's own out-of-fold
 * score, which conditions on the reference's own estimation noise and manufactures
 * a shape (readers lose where the reference was sure, win where it was not) that
 * reverses exactly when you band by the compared arm instead. The artifact ships
 * that reversal as `banding_sensitivity`, and this module carries the geometry for
 * drawing it, because volunteering the artifact is the evidence for the choice.
 *
 * Band membership is a pure function of the evidence count, so no statement is
 * assigned by a tie-break — a strictly stronger property than the "the split tie
 * group is entirely false" argument the decile version had to make.
 *
 * This module is import-safe on the client: typed shape, fixed figure geometry,
 * and a pure `validateApDecomposition()` that THROWS on any shape or arithmetic
 * drift. `validatePaperLiteral()` calls it inside its own try/catch so a missing
 * or drifted payload fails the whole paper load closed rather than silently
 * dropping the figure.
 *
 * ONE-WAY BY CONSTRUCTION, and it was not always. `paper-literal.ts` value-imports
 * `validateApDecomposition` from this module. This module used to value-import
 * `standingOfBounds` back, which made a genuine runtime cycle that svelte-check
 * and the bundler both tolerated. `standingOfBounds` now comes from the leaf
 * `paper-interval.ts`, and what remains from `paper-literal.ts` is `import type`,
 * which is erased. `scripts/test-paper-render-invariants.mjs` fails if the edge
 * comes back — the fan slot
 * table below repeats the five canonical arm labels deliberately, and the
 * contract test asserts they still match `PAPER_LITERAL_ARM_SPECS` label-for-
 * label and kind-for-kind.
 */

import { budget, fail, record, unit, text, nonNegativeInteger, positiveInteger } from './paper-validate.ts';
import { standingOfBounds, type Standing } from './paper-interval.ts';
// `import type` and NOT `import { type … }`: verbatimModuleSyntax is on
// (.svelte-kit/tsconfig.json:18), so the braced form still emits `import {} from
// './paper-literal.ts'` — a real runtime edge back to the module that imports
// THIS one. Only the `import type` form is erased outright.
import type { PaperArmKind, ShippedProse } from './paper-literal.ts';

/**
 * THE PLAIN HALF OF EVERY TWIN THIS MODULE EMITS.
 *
 * `ap_decomposition_by_paper_band.json` explains its own banding choice in three
 * sentences, and all three shipped raw: "an integer census of the corpus",
 * "carrying no arm's estimation noise", "the reader gate is purely subtractive".
 * The restatements below say the same things, at the same strength — the mirror
 * reversal, the ceiling argument and the saturating input all survive.
 */
const AP_DECOMP_PLAIN = {
	whyExogenous:
		'A whole-number count taken over the corpus, fixed before any model ran. It is not a ' +
		'score, it is not fitted to any label, and it carries none of a model’s own estimation ' +
		'noise — so there is no mirror image of it to flip under. It is also the input that makes ' +
		'the combination rule saturate: one minus the product, over every source, of that ' +
		'source’s systematic error rate plus its random error rate raised to the number of ' +
		'evidence entries that source supplied.',
	whyNotTheNoisyOr:
		'The unfitted combination-rule belief is the other candidate that is fixed in advance, ' +
		'and it is rejected: the reading step can only take belief away, so no reading model’s ' +
		'belief ever exceeds it (verified: 0 of 6,756). It is each reading model’s own ceiling, ' +
		'so cutting the bands on it would condition on part of the reading model’s own score. It ' +
		'is kept as a mirror check instead.',
	direction:
		'left = 1 evidence entry, where the combination rule is at its weakest; right = 33 or ' +
		'more, where it has saturated'
} as const;

/**
 * Fixed, un-broken, zero-anchored y-domain in AP POINTS (1 pt = 0.01 AP). Hard-
 * coded rather than data-fit so the zero rule never drifts between deploys; it
 * covers Gemma 4 E2B's ci95 low (−2.56) and Gemma 4 26B's ci95 high (+1.85) with
 * padding. `validateApDecomposition` FAILS CLOSED if any drawn value escapes it,
 * so a future artifact can never be silently clipped or quietly widen the axis.
 */
export const AP_DECOMP_Y_MIN = -2.8;
export const AP_DECOMP_Y_MAX = 2.1;

/** Y ticks in AP points. No gridlines — tick marks and the zero rule only. */
export const AP_DECOMP_Y_TICKS = [
	-2.5, -2.0, -1.5, -1.0, -0.5, 0, 0.5, 1.0, 1.5, 2.0
] as const;

/**
 * The evidence-count ladder the bands are cut on: [low, high] entries per
 * statement, `null` = open upper end. Geometric because the noisy-OR saturates
 * exponentially in the count (belief = 1 − ∏ₛ (systₛ + randₛ^nₛ)). FROZEN — the
 * validator requires the artifact's own edges to equal this list, so a re-banded
 * artifact gates the figure instead of being drawn against the wrong axis.
 */
export const AP_DECOMP_BAND_EDGES: readonly (readonly [number, number | null])[] = [
	[1, 1],
	[2, 2],
	[3, 4],
	[5, 8],
	[9, 16],
	[17, 32],
	[33, null]
] as const;

export const AP_DECOMP_BAND_COUNT = AP_DECOMP_BAND_EDGES.length;

/** Band nets must sum to the arm's point ΔAP this tightly (shipped parity gate). */
export const AP_DECOMP_PARITY_TOL = 1e-9;

/**
 * The five drawn arms, in FIXED fan order (descending point ΔAP, with the paper's
 * own alternate-feature variant sitting between the winners and the control so
 * the eye crosses the neutral arm on the way down). Not sortable, not
 * configurable: `validateApDecomposition` requires the artifact's arm sequence to
 * equal this label sequence exactly.
 *
 * `shortLines` is the two-line direct label stacked under the fan baseline;
 * `captionShort` is the compact name used in the prose caption.
 *
 * ENCODING: each arm carries its OWN `stroke` and `dash`, and the figure
 * direct-labels every trace at its right terminus. The palette is a luminance
 * ramp inside one hue family: same family reads as "these are the same kind of
 * thing", while luminance + dash give each trace an identity that survives
 * greyscale printing and colour-vision deficiency, which hue alone does not.
 * Every stroke clears 3:1 against --paper (WCAG 1.4.11 for graphical objects) and
 * adjacent ramp steps differ by 1.3-1.7x in luminance. The ramp is ordered by
 * total delta, so lightness carries rank rather than being arbitrary.
 */
export interface ApDecompFanSlot {
	label: string;
	kind: PaperArmKind;
	shortLines: readonly string[];
	captionShort: string;
	group: 'top-llm' | 'control' | 'paper-variant';
	strokeWidth: number;
	dashed: boolean;
	/**
	 * On-screen name. DECOUPLED from `label`, which is a FROZEN join key into
	 * paper_literal_vs_llms.point_metrics. Where the arm is one of the 2023
	 * paper's published methods this is their own Table 6 name for it; where it is
	 * one of ours it reads "… reading", the step where the model reads each piece
	 * of evidence and keeps or drops it. Identical to the name the same model
	 * carries on the belief-model ladder and in the review queue, so one model
	 * reads as one model across the page.
	 */
	display: string;
	/** Per-arm stroke; see the ENCODING note above. */
	stroke: string;
	/** Per-arm dash signature, or undefined for solid. Greyscale-robust identity. */
	dash?: string;
}

export const AP_DECOMP_FAN_SLOTS: readonly ApDecompFanSlot[] = [
	{
		label: 'Gemma 4 26B',
		kind: 'llm',
		shortLines: ['Gemma 4', '26B'],
		captionShort: '26B',
		group: 'top-llm',
		strokeWidth: 2.0,
		dashed: false,
		display: 'Gemma 4 26B reading',
		stroke: '#463809'
	},
	{
		label: 'GLM-5',
		kind: 'llm',
		shortLines: ['GLM-5'],
		captionShort: 'GLM-5',
		group: 'top-llm',
		strokeWidth: 1.75,
		dashed: true,
		display: 'GLM-5 reading',
		stroke: '#6f5a16',
		dash: '7 3'
	},
	{
		label: 'Gemma 4 31B',
		kind: 'llm',
		shortLines: ['Gemma 4', '31B'],
		captionShort: '31B',
		group: 'top-llm',
		strokeWidth: 1.75,
		dashed: true,
		display: 'Gemma 4 31B reading',
		stroke: '#947b2b',
		dash: '2.5 2.5'
	},
	{
		label: 'Paper literal RF+prom/avglen',
		kind: 'paper',
		shortLines: ['RF 2k-d13', '+prom/avglen'],
		captionShort: 'RF +prom/avglen',
		group: 'paper-variant',
		strokeWidth: 1.5,
		dashed: false,
		display: 'RF 2k-d13 + Type/#PMIDs/prom/avglen',
		stroke: 'var(--accent)'
	},
	{
		label: 'Gemma 4 E2B',
		kind: 'llm',
		shortLines: ['Gemma 4', 'E2B'],
		captionShort: 'E2B',
		group: 'control',
		strokeWidth: 1.5,
		dashed: true,
		display: 'Gemma 4 E2B reading',
		stroke: '#a89047',
		dash: '1 3'
	}
] as const;

/**
 * Geometry of the TERMINUS FAN, exported so its label fit is derived from the
 * same numbers the component draws with. Mirrors the layout constants in
 * `ApDecompositionByPaperRank.svelte`.
 */
export const AP_DECOMP_FAN_GEOMETRY = {
	width: 900,
	/** Left edge of the fan band; the five slots divide [gapRight, fanRight]. */
	gapRight: 596,
	fanRight: 836,
	nameFontPx: 7,
	/** Measured advance of the mono face at 7px: 5.4186 × 7/9. */
	nameUnitsPerChar: 4.2145,
	/** The band census note is left-anchored at `gapRight` and runs right. */
	noteFontPx: 7
} as const;

/**
 * FAN NAME FIT — and why a plain character budget is the WRONG check here.
 *
 * The fan direct-labels each trace under its own terminus, CENTRE-anchored in a
 * slot of (836 − 596) / 5 = 48 units. A naive per-slot budget is 48 ÷ 4.2145 =
 * 11.4 → 11 characters, and today's longest short line, "+prom/avglen", is 12 —
 * so a naive budget would fail a figure that is in fact fine, because a centred
 * label only ever collides with its NEIGHBOUR's centred label, and this one's
 * neighbours are short ("Gemma 4", 7 chars). The check that matters is therefore
 * pairwise: adjacent half-widths must sum to no more than the slot pitch, and
 * the outermost labels must stay inside the viewBox.
 *
 * Measured clearance on today's slots: the tightest adjacent pair is
 * "+prom/avglen" against "Gemma 4" at 40.05 of 48 units — 7.95 units, about 1.9
 * characters of headroom. That is the tightest gutter in this figure, and it is
 * now a test rather than an observation.
 */
export function apDecompFanNamesFit(
	slots: readonly ApDecompFanSlot[] = AP_DECOMP_FAN_SLOTS
): boolean {
	const g = AP_DECOMP_FAN_GEOMETRY;
	if (slots.length === 0) return false;
	const pitch = (g.fanRight - g.gapRight) / slots.length;
	const halfWidth = (slot: ApDecompFanSlot): number =>
		(Math.max(...slot.shortLines.map((line) => line.length)) * g.nameUnitsPerChar) / 2;
	const centre = (index: number): number => g.gapRight + pitch * (index + 0.5);
	for (let i = 0; i < slots.length; i += 1) {
		const half = halfWidth(slots[i]);
		if (centre(i) - half < 0 || centre(i) + half > g.width) return false;
		if (i + 1 < slots.length && half + halfWidth(slots[i + 1]) > pitch) return false;
	}
	return true;
}

/**
 * The band-census note is LEFT-anchored at the fan band's left edge and runs
 * toward the viewBox edge, so an overrun clips its TRAILING glyphs. Budget:
 * (900 − 596) ÷ 4.2145 = 72.1 → 72 characters. Today's note is 62 characters,
 * leaving 10 characters / 42.1 units.
 */
export const AP_DECOMP_COUNT_NOTE_BUDGET_CHARS = 72;

export function apDecompCountNoteFits(note: string): boolean {
	return note.length <= AP_DECOMP_COUNT_NOTE_BUDGET_CHARS;
}

/**
 * Painting order, back to front: the paper variant first, then the E2B control,
 * then 31B, GLM-5, 26B. Fixed, not sortable.
 */
export const AP_DECOMP_LINE_DRAW_ORDER: readonly string[] = [
	'Paper literal RF+prom/avglen',
	'Gemma 4 E2B',
	'Gemma 4 31B',
	'GLM-5',
	'Gemma 4 26B'
] as const;

/**
 * The four reader arms the mirror diagnostic compares, in the artifact's fixed
 * order. Derived from the fan slots so the two lists cannot drift; these strings
 * are FROZEN point_metrics join keys, never rendered.
 */
export const AP_DECOMP_READER_ARM_LABELS: readonly string[] = AP_DECOMP_FAN_SLOTS.filter(
	(slot) => slot.kind === 'llm'
).map((slot) => slot.label);

// ---------------------------------------------------------------------------
// Mirror strip: the banding-sensitivity evidence, drawn rather than footnoted.
// ---------------------------------------------------------------------------

/**
 * SVG geometry for the mirror strip, exported so the label budget below is
 * derived from it rather than eyeballed. The strip's own origin is its first
 * row's top edge; the component translates the whole group into place.
 *
 * Right-anchored SVG text that overruns its gutter loses its LEADING glyphs
 * silently — no layout error, no test failure — so both text gutters are
 * budgeted in characters and enforced in `buildApDecompMirror`.
 */
export const AP_DECOMP_MIRROR_GEOMETRY = {
	width: 900,
	/** Row labels are right-anchored here; usable gutter is 0 → 250 units. */
	labelAnchorX: 250,
	/** Shared zero rule for the head/tail bars. */
	zeroX: 470,
	/** Half-width of the fixed, symmetric bar domain, in user units. */
	halfWidth: 190,
	/** Readouts are left-anchored here; usable gutter is 676 → 900. */
	readoutX: 676,
	labelFontPx: 8,
	/**
	 * Measured advance of the mono face: 5.4186 user units per character at 9px,
	 * scaled to the 8px used here (5.4186 × 8/9).
	 */
	monoUnitsPerChar: 4.8165,
	rowHeight: 30,
	barHeight: 7,
	/** Bar top edges within a row: head above, tail below. */
	headBarY: 4,
	tailBarY: 15,
	/** Label baseline within a row, centred between the two bars. */
	labelBaselineY: 16,
	/**
	 * Rows are grouped by `kind` — whether the banding variable is one of the two
	 * scores being differenced. That distinction IS the finding, so it gets a
	 * header band rather than only a colour, which would not survive greyscale.
	 */
	groupHeaderHeight: 16,
	groupHeaderBaselineY: 11
} as const;

/**
 * LABEL BUDGET (this page has shipped a silent right-anchored clip three times).
 * Row labels: 250 units ÷ 4.8165 u/char at 8px = 51.9 → 51 characters. The
 * longest label this module produces is
 * "RF 2k-d13 + Type/#PMIDs/promoter’s own score" at 44 characters (211.9 units),
 * leaving 38.1 units of slack. `buildApDecompMirror` THROWS if any label exceeds
 * the budget, so a longer display name gates the figure to `unavailable` instead
 * of quietly eating its first glyphs.
 */
export const AP_DECOMP_MIRROR_LABEL_BUDGET_CHARS = 51;

/**
 * READOUT BUDGET: (900 − 676) units ÷ 4.8165 u/char at 8px = 46.5 → 46
 * characters. The longest readout this module produces is "head −0.52  tail
 * +0.95" at 22 characters. Left-anchored, so an overrun would clip the TRAILING
 * glyphs; budgeted and enforced all the same.
 */
export const AP_DECOMP_MIRROR_READOUT_BUDGET_CHARS = 46;

/**
 * Fixed, symmetric bar domain in AP points. Hard-coded rather than data-fit, on
 * the same rule as the main y-domain: `buildApDecompMirror` THROWS if a drawn
 * head or tail escapes it rather than clipping a bar to the axis.
 */
export const AP_DECOMP_MIRROR_DOMAIN_PTS = 1.25;

export type ApDecompMirrorKey =
	| 'reference_own_score'
	| 'drawn_arm_own_score'
	| 'unfitted_noisy_or'
	| 'evidence_count';

export interface ApDecompMirrorSpec {
	/** FROZEN artifact join key for this banding variable. */
	key: ApDecompMirrorKey;
	/**
	 * On-screen row label. DECOUPLED from the artifact's `banding_arm`, which for
	 * the two model rows is a FROZEN point_metrics join key
	 * ("Paper literal RF+promoter", "Gemma 4 26B") and must never be rendered.
	 */
	display: string;
	/**
	 * `endogenous` = the banding variable is one of the two scores being
	 * differenced, so the row is conditioning on its own noise.
	 */
	kind: 'endogenous' | 'exogenous';
	/** True for the variable the drawn waterfall above actually bands on. */
	drawn: boolean;
}

export const AP_DECOMP_MIRROR_SPECS: readonly ApDecompMirrorSpec[] = [
	{
		key: 'reference_own_score',
		display: 'RF 2k-d13 + Type/#PMIDs/promoter’s own score',
		kind: 'endogenous',
		drawn: false
	},
	{
		key: 'drawn_arm_own_score',
		display: 'Gemma 4 26B reading’s own score',
		kind: 'endogenous',
		drawn: false
	},
	{
		key: 'unfitted_noisy_or',
		display: 'noisy-OR SimpleScorer (direct) belief',
		kind: 'exogenous',
		drawn: false
	},
	{
		key: 'evidence_count',
		display: 'evidence entries per statement',
		kind: 'exogenous',
		drawn: true
	}
] as const;

export interface ApDecompositionBand {
	index: number;
	/**
	 * The band's identity string, e.g. "3–4" or "17+". Kept under the artifact's own
	 * field name because the shipped contract runner joins on it
	 * (`parsedDecomp.bands.map((band) => band.label)`); it is NEVER rendered.
	 */
	label: string;
	/**
	 * The same string as the x-axis TICK. Separate field on purpose: every other
	 * `label` on this page is a frozen join key, and a field called `label` in a
	 * render position is exactly how those keys have reached the screen. The render
	 * layer reads `display` and only `display`, with no exceptions to reason about.
	 */
	display: string;
	/** Evidence-entry range this band covers; `evidenceHigh: null` = open end. */
	evidenceLow: number;
	evidenceHigh: number | null;
	n: number;
	nTrue: number;
	nFalse: number;
	/** Share of the band that is false, in [0, 1]. */
	errorRate: number;
	/** Total evidence entries carried by the band's statements. */
	evidenceEntries: number;
	/** The reference's OWN average-precision mass here, AP points. */
	referenceContributionPts: number;
}

export interface ApDecompositionArm {
	label: string;
	kind: PaperArmKind;
	/** Pooled average precision of this arm on the shared 1689-statement panel. */
	averagePrecision: number;
	/** Observed point ΔAP vs the reference (NOT the bootstrap-mean `delta` field). */
	totalDeltaAp: number;
	/** The same observed point delta in AP points (×100) — the line's endpoint. */
	totalPts: number;
	/** Per-band net contribution difference, AP points, thinnest evidence first. */
	perBandNetPts: number[];
	/** Running cumulative of `perBandNetPts`; the last element IS `totalPts`. */
	cumulativePts: number[];
	/**
	 * How many bands move in the same direction as the total. `AP_DECOMP_BAND_COUNT`
	 * means the delta is diffuse — there is no band where the arm gives ground.
	 */
	nBandsAgreeingWithTotalSign: number;
	/** Largest single band's share of |total|, in [0, ∞). */
	largestBandShareOfTotal: number;
	/** 1-based index of that band. */
	largestBandIndex: number;
	/** Percentile 95% paired-bootstrap bounds, AP points. Totals only. */
	ci95LowPts: number;
	ci95HighPts: number;
	/** Share of resamples in which this arm beat the reference. */
	pArmGreater: number;
	/**
	 * WHERE THE INTERVAL SITS RELATIVE TO ZERO — three classes, never a boolean.
	 *
	 * This field was `clearsZero: boolean`, defined as `low > 0 || high < 0`. That
	 * is the identical sign-blind predicate as the `excludesZero` this project
	 * retired after it shipped SIX times, surviving here only under a synonym: it
	 * is TRUE for an interval lying entirely BELOW zero, so `clearsZero ? win : tie`
	 * reads a decisive LOSS as a win. `notClearing` in the component was one such
	 * branch away from occurrence #7.
	 *
	 * Retiring the name and keeping the boolean would have been a rename, not a fix.
	 * The three classes make the wrong sentence unwritable instead of merely
	 * discouraged: there is no boolean left to hang a two-way ternary on.
	 */
	standing: Standing;
}

/** One reader arm's head/tail summary under one candidate banding variable. */
export interface ApDecompMirrorArm {
	/** FROZEN point_metrics join key. Not rendered. */
	label: string;
	headPts: number;
	tailPts: number;
	/** headPts − tailPts: which end of the banding variable the arm appears to win at. */
	tiltPts: number;
	/** Worst movement of head/tail between the two extreme tie-break orderings. */
	maxTieBreakSpreadPts: number;
}

export interface ApDecompMirrorVariant {
	key: ApDecompMirrorKey;
	display: string;
	kind: 'endogenous' | 'exogenous';
	drawn: boolean;
	arms: ApDecompMirrorArm[];
}

export interface ApDecompBandingSensitivity {
	/** Deciles of the banding variable — the cut the head/tail summary uses. */
	nBands: number;
	headBands: number;
	tailBands: number;
	/** FROZEN join key of the arm the strip draws. Not rendered. */
	drawnArm: string;
	variants: ApDecompMirrorVariant[];
	/** FROZEN join keys of the arms whose tilt SIGN flips under mirroring. */
	armsReversingUnderMirroring: string[];
	nArmsCompared: number;
	maxAbsTiltEndogenousPts: number;
	maxAbsTiltExogenousPts: number;
	maxTieBreakSpreadPts: number;
	tieBreakSpreadTolerancePts: number;
}

export interface ApDecompositionBanding {
	/** Human-readable name of the banding variable. */
	variable: string;
	isExogenous: boolean;
	whyExogenous: string;
	/** `whyExogenous` with its plain restatement — `shipped` is byte-identical. */
	whyExogenousProse: ShippedProse;
	whyNotTheNoisyOr: string;
	/** `whyNotTheNoisyOr` with its plain restatement — `shipped` is byte-identical. */
	whyNotTheNoisyOrProse: ShippedProse;
	direction: string;
	/** `direction` with its plain restatement — `shipped` is byte-identical. */
	directionProse: ShippedProse;
	nDistinctEvidenceCounts: number;
	evidenceMin: number;
	evidenceMax: number;
	/** Statements on which the census agrees with the paper's own released counts. */
	nStatementsAgreeing: number;
	/** Unique (statement, evidence) pairs vs evidence entries — the scope note. */
	nUniquePairs: number;
	nEvidenceEntries: number;
	nStatementsChangingBandUnderUniquePairs: number;
}

export interface ApDecomposition {
	metric: string;
	unit: string;
	referenceArm: string;
	referenceAveragePrecision: number;
	nStatements: number;
	nTrue: number;
	nFalse: number;
	banding: ApDecompositionBanding;
	bands: ApDecompositionBand[];
	bandTrueCounts: number[];
	bandFalseCounts: number[];
	arms: ApDecompositionArm[];
	bandingSensitivity: ApDecompBandingSensitivity;
	/** Statements the banding assigned by a tie-break. Must be 0. */
	nAssignedByTieBreak: number;
	/** Reader beliefs exceeding the unfitted noisy-OR. Must be 0. */
	nReaderBeliefsExceedingNoisyOr: number;
	nReaderBeliefComparisons: number;
	nBootstrap: number;
	seed: number;
}

type UnknownRecord = Record<string, unknown>;



function finite(value: unknown, context: string): number {
	if (typeof value !== 'number' || !Number.isFinite(value)) {
		fail(context, 'expected a finite number');
	}
	return value;
}




function exactly(value: unknown, expected: number, context: string): number {
	const parsed = nonNegativeInteger(value, context);
	if (parsed !== expected) fail(context, `expected exactly ${expected}`);
	return parsed;
}

function boolean(value: unknown, context: string): boolean {
	if (typeof value !== 'boolean') fail(context, 'expected a boolean');
	return value;
}


function numberList(value: unknown, length: number, context: string): number[] {
	if (!Array.isArray(value) || value.length !== length) {
		fail(context, `expected an array of ${length} numbers`);
	}
	return value.map((entry, index) => finite(entry, `${context}[${index}]`));
}

function integerList(value: unknown, length: number, context: string): number[] {
	if (!Array.isArray(value) || value.length !== length) {
		fail(context, `expected an array of ${length} integers`);
	}
	return value.map((entry, index) => nonNegativeInteger(entry, `${context}[${index}]`));
}

function stringList(value: unknown, context: string): string[] {
	if (!Array.isArray(value)) fail(context, 'expected an array of strings');
	return value.map((entry, index) => text(entry, `${context}[${index}]`));
}

/** Every drawn coordinate must live inside the fixed, un-broken y-domain. */
function inDomain(value: number, context: string): number {
	if (value < AP_DECOMP_Y_MIN || value > AP_DECOMP_Y_MAX) {
		fail(context, `escapes the fixed y-domain [${AP_DECOMP_Y_MIN}, ${AP_DECOMP_Y_MAX}] AP points`);
	}
	return value;
}

/** The band label the ladder implies. En dash for ranges, matching the artifact. */
export function bandLabelFor(low: number, high: number | null): string {
	if (high === null) return `${low}+`;
	return low === high ? String(low) : `${low}–${high}`;
}

function parseBands(value: unknown): ApDecompositionBand[] {
	if (!Array.isArray(value) || value.length !== AP_DECOMP_BAND_COUNT) {
		fail('ap_decomposition.bands', `expected ${AP_DECOMP_BAND_COUNT} bands`);
	}
	return value.map((entry, index) => {
		const context = `ap_decomposition.bands[${index}]`;
		const band = record(entry, context);
		const [low, high] = AP_DECOMP_BAND_EDGES[index];
		const expectedLabel = bandLabelFor(low, high);
		if (band.index !== index + 1) fail(`${context}.index`, `expected ${index + 1}`);
		if (band.label !== expectedLabel) fail(`${context}.label`, `expected ${expectedLabel}`);
		// The figure's x-axis IS the ladder; an artifact banded on different edges
		// would draw the wrong statements under the wrong tick.
		if (band.evidence_low !== low) fail(`${context}.evidence_low`, `expected ${low}`);
		if ((band.evidence_high ?? null) !== high) {
			fail(`${context}.evidence_high`, `expected ${high === null ? 'null' : high}`);
		}
		const n = positiveInteger(band.n, `${context}.n`);
		const nTrue = nonNegativeInteger(band.n_true, `${context}.n_true`);
		const nFalse = nonNegativeInteger(band.n_false, `${context}.n_false`);
		if (nTrue + nFalse !== n) fail(context, 'n_true + n_false must equal n');
		const errorRate = unit(band.error_rate, `${context}.error_rate`);
		if (Math.abs(errorRate - nFalse / n) > 1e-9) {
			fail(`${context}.error_rate`, 'must equal n_false / n');
		}
		const evidenceMin = positiveInteger(band.evidence_min, `${context}.evidence_min`);
		const evidenceMax = positiveInteger(band.evidence_max, `${context}.evidence_max`);
		if (evidenceMin < low || (high !== null && evidenceMax > high)) {
			fail(context, 'the band holds statements outside its own evidence range');
		}
		return {
			index: index + 1,
			label: expectedLabel,
			display: expectedLabel,
			evidenceLow: low,
			evidenceHigh: high,
			n,
			nTrue,
			nFalse,
			errorRate,
			evidenceEntries: positiveInteger(band.evidence_entries, `${context}.evidence_entries`),
			referenceContributionPts: finite(
				band.reference_contribution_pts,
				`${context}.reference_contribution_pts`
			)
		};
	});
}

function parseArm(entry: unknown, index: number): ApDecompositionArm {
	const slot = AP_DECOMP_FAN_SLOTS[index];
	const context = `ap_decomposition.arms[${index}]`;
	const arm = record(entry, context);
	if (arm.name !== slot.label) {
		fail(`${context}.name`, `expected the fixed fan order — ${slot.label}`);
	}
	const totalPts = finite(arm.total_pts, `${context}.total_pts`);
	const totalDeltaAp = finite(arm.total_delta_ap, `${context}.total_delta_ap`);
	const perBandNetPts = numberList(
		arm.per_band_net_pts,
		AP_DECOMP_BAND_COUNT,
		`${context}.per_band_net_pts`
	);
	const cumulativePts = numberList(
		arm.cumulative_pts,
		AP_DECOMP_BAND_COUNT,
		`${context}.cumulative_pts`
	);
	const ci95LowPts = finite(arm.ci95_low_pts, `${context}.ci95_low_pts`);
	const ci95HighPts = finite(arm.ci95_high_pts, `${context}.ci95_high_pts`);
	if (ci95LowPts > ci95HighPts) fail(context, 'ci95_low_pts must not exceed ci95_high_pts');
	// The shipped flag is still CHECKED against the endpoints — dropping that would
	// lose a real cross-check on the artifact — but what leaves this loader is the
	// three-way class, so no caller can branch two ways on it.
	const clearsZero = boolean(arm.clears_zero, `${context}.clears_zero`);
	if (clearsZero !== (ci95LowPts > 0 || ci95HighPts < 0)) {
		fail(`${context}.clears_zero`, 'must equal ci95_low_pts > 0 || ci95_high_pts < 0');
	}

	// The parity that makes this figure honest: the band nets ARE the total, the
	// running cumulative IS their prefix sum, and the endpoint IS the shipped
	// point delta. Any drift fails the load rather than drawing a wrong endpoint.
	let running = 0;
	for (let i = 0; i < AP_DECOMP_BAND_COUNT; i += 1) {
		running += perBandNetPts[i];
		if (Math.abs(running - cumulativePts[i]) > AP_DECOMP_PARITY_TOL) {
			fail(`${context}.cumulative_pts[${i}]`, 'must equal the running sum of per_band_net_pts');
		}
		inDomain(cumulativePts[i], `${context}.cumulative_pts[${i}]`);
	}
	if (Math.abs(running - totalPts) > AP_DECOMP_PARITY_TOL) {
		fail(`${context}.total_pts`, 'must equal the sum of per_band_net_pts');
	}
	if (Math.abs(totalPts - totalDeltaAp * 100) > AP_DECOMP_PARITY_TOL) {
		fail(`${context}.total_pts`, 'must equal total_delta_ap in AP points (×100)');
	}
	inDomain(ci95LowPts, `${context}.ci95_low_pts`);
	inDomain(ci95HighPts, `${context}.ci95_high_pts`);

	// "Gains in every band" is a caption claim, so it is recomputed here rather
	// than trusted: the artifact's count has to match the drawn series.
	const agreeing = nonNegativeInteger(
		arm.n_bands_agreeing_with_total_sign,
		`${context}.n_bands_agreeing_with_total_sign`
	);
	const recomputed = perBandNetPts.filter((value) => Math.sign(value) === Math.sign(totalPts)).length;
	if (agreeing !== recomputed) {
		fail(
			`${context}.n_bands_agreeing_with_total_sign`,
			`disagrees with per_band_net_pts (${agreeing} vs ${recomputed})`
		);
	}
	if (agreeing > AP_DECOMP_BAND_COUNT) {
		fail(`${context}.n_bands_agreeing_with_total_sign`, `exceeds ${AP_DECOMP_BAND_COUNT}`);
	}

	return {
		label: slot.label,
		kind: slot.kind,
		averagePrecision: unit(arm.average_precision, `${context}.average_precision`),
		totalDeltaAp,
		totalPts,
		perBandNetPts,
		cumulativePts,
		nBandsAgreeingWithTotalSign: agreeing,
		largestBandShareOfTotal: finite(
			arm.largest_band_share_of_total,
			`${context}.largest_band_share_of_total`
		),
		largestBandIndex: positiveInteger(arm.largest_band_index, `${context}.largest_band_index`),
		ci95LowPts,
		ci95HighPts,
		pArmGreater: unit(arm.p_arm_greater, `${context}.p_arm_greater`),
		standing: standingOfBounds(ci95LowPts, ci95HighPts)
	};
}

function parseBandingSensitivity(
	value: unknown,
	referenceArm: string
): ApDecompBandingSensitivity {
	const context = 'ap_decomposition.banding_sensitivity';
	const obj = record(value, context);
	const drawnArm = text(obj.drawn_arm, `${context}.drawn_arm`);
	if (!AP_DECOMP_FAN_SLOTS.some((slot) => slot.label === drawnArm)) {
		fail(`${context}.drawn_arm`, `${drawnArm} is not one of the drawn arms`);
	}

	if (!Array.isArray(obj.variants) || obj.variants.length !== AP_DECOMP_MIRROR_SPECS.length) {
		fail(`${context}.variants`, `expected ${AP_DECOMP_MIRROR_SPECS.length} banding variants`);
	}
	const variants = obj.variants.map((entry, index) => {
		const spec = AP_DECOMP_MIRROR_SPECS[index];
		const rowContext = `${context}.variants[${index}]`;
		const row = record(entry, rowContext);
		if (row.key !== spec.key) fail(`${rowContext}.key`, `expected ${spec.key}`);
		if (row.kind !== spec.kind) fail(`${rowContext}.kind`, `expected ${spec.kind}`);
		// The display strings above are hand-written; pin them to the artifact's
		// own join key so a row can never be labelled as the wrong variable.
		const bandingArm = text(row.banding_arm, `${rowContext}.banding_arm`);
		if (spec.key === 'reference_own_score' && bandingArm !== referenceArm) {
			fail(`${rowContext}.banding_arm`, `expected the reference arm — ${referenceArm}`);
		}
		if (spec.key === 'drawn_arm_own_score' && bandingArm !== drawnArm) {
			fail(`${rowContext}.banding_arm`, `expected the drawn arm — ${drawnArm}`);
		}

		if (!Array.isArray(row.arms) || row.arms.length !== AP_DECOMP_READER_ARM_LABELS.length) {
			fail(`${rowContext}.arms`, `expected ${AP_DECOMP_READER_ARM_LABELS.length} reader arms`);
		}
		const arms = row.arms.map((armEntry, armIndex) => {
			const armContext = `${rowContext}.arms[${armIndex}]`;
			const arm = record(armEntry, armContext);
			const expected = AP_DECOMP_READER_ARM_LABELS[armIndex];
			if (arm.arm !== expected) fail(`${armContext}.arm`, `expected ${expected}`);
			const headPts = finite(arm.head_pts, `${armContext}.head_pts`);
			const tailPts = finite(arm.tail_pts, `${armContext}.tail_pts`);
			const tiltPts = finite(arm.tilt_pts, `${armContext}.tilt_pts`);
			if (Math.abs(tiltPts - (headPts - tailPts)) > AP_DECOMP_PARITY_TOL) {
				fail(`${armContext}.tilt_pts`, 'must equal head_pts − tail_pts');
			}
			return {
				label: expected,
				headPts,
				tailPts,
				tiltPts,
				maxTieBreakSpreadPts: finite(
					arm.max_tie_break_spread_pts,
					`${armContext}.max_tie_break_spread_pts`
				)
			};
		});
		return { key: spec.key, display: spec.display, kind: spec.kind, drawn: spec.drawn, arms };
	});

	const tolerance = finite(
		obj.tie_break_spread_tolerance_pts,
		`${context}.tie_break_spread_tolerance_pts`
	);
	const worst = finite(obj.max_tie_break_spread_pts, `${context}.max_tie_break_spread_pts`);
	// A decile edge inside a block of tied scores has to fall somewhere. If the
	// summary moves more than the tolerance between the two extreme orderings, the
	// strip would be reporting the tie-break rather than the banding variable.
	if (worst > tolerance) {
		fail(
			`${context}.max_tie_break_spread_pts`,
			`${worst} exceeds the artifact's own tolerance ${tolerance}`
		);
	}

	return {
		nBands: positiveInteger(obj.n_bands, `${context}.n_bands`),
		headBands: positiveInteger(obj.head_bands, `${context}.head_bands`),
		tailBands: positiveInteger(obj.tail_bands, `${context}.tail_bands`),
		drawnArm,
		variants,
		armsReversingUnderMirroring: stringList(
			obj.arms_whose_tilt_sign_reverses_under_mirroring,
			`${context}.arms_whose_tilt_sign_reverses_under_mirroring`
		),
		nArmsCompared: positiveInteger(obj.n_arms_compared, `${context}.n_arms_compared`),
		maxAbsTiltEndogenousPts: finite(
			obj.max_abs_tilt_endogenous_banding_pts,
			`${context}.max_abs_tilt_endogenous_banding_pts`
		),
		maxAbsTiltExogenousPts: finite(
			obj.max_abs_tilt_exogenous_banding_pts,
			`${context}.max_abs_tilt_exogenous_banding_pts`
		),
		maxTieBreakSpreadPts: worst,
		tieBreakSpreadTolerancePts: tolerance
	};
}

function parseBanding(value: unknown): ApDecompositionBanding {
	const context = 'ap_decomposition.banding';
	const obj = record(value, context);
	if (obj.kind !== 'power_of_two_ladder_on_evidence_count') {
		fail(`${context}.kind`, 'expected power_of_two_ladder_on_evidence_count');
	}
	if (!boolean(obj.variable_is_exogenous, `${context}.variable_is_exogenous`)) {
		// The whole premise of this figure. An artifact that declares its banding
		// variable endogenous must gate rather than be drawn under this caption.
		fail(`${context}.variable_is_exogenous`, 'the drawn banding variable must be exogenous');
	}
	if (!Array.isArray(obj.edges) || obj.edges.length !== AP_DECOMP_BAND_COUNT) {
		fail(`${context}.edges`, `expected ${AP_DECOMP_BAND_COUNT} band edges`);
	}
	obj.edges.forEach((entry, index) => {
		const [low, high] = AP_DECOMP_BAND_EDGES[index];
		if (!Array.isArray(entry) || entry.length !== 2) {
			fail(`${context}.edges[${index}]`, 'expected a [low, high] pair');
		}
		if (entry[0] !== low || (entry[1] ?? null) !== high) {
			fail(`${context}.edges[${index}]`, `expected [${low}, ${high === null ? 'null' : high}]`);
		}
	});

	const verified = record(obj.verified_against, `${context}.verified_against`);
	const scope = record(obj.unique_pair_scope, `${context}.unique_pair_scope`);
	// Parsed exactly as before; the plain restatement travels beside each, and the
	// flat string stays byte-identical to its twin's `shipped`.
	const whyExogenousProse: ShippedProse = {
		shipped: text(obj.why_exogenous, `${context}.why_exogenous`),
		plain: AP_DECOMP_PLAIN.whyExogenous
	};
	const whyNotTheNoisyOrProse: ShippedProse = {
		shipped: text(obj.why_not_the_noisy_or, `${context}.why_not_the_noisy_or`),
		plain: AP_DECOMP_PLAIN.whyNotTheNoisyOr
	};
	const directionProse: ShippedProse = {
		shipped: text(obj.direction, `${context}.direction`),
		plain: AP_DECOMP_PLAIN.direction
	};
	return {
		variable: text(obj.variable, `${context}.variable`),
		isExogenous: true,
		whyExogenous: whyExogenousProse.shipped,
		whyExogenousProse,
		whyNotTheNoisyOr: whyNotTheNoisyOrProse.shipped,
		whyNotTheNoisyOrProse,
		direction: directionProse.shipped,
		directionProse,
		nDistinctEvidenceCounts: positiveInteger(
			obj.n_distinct_evidence_counts,
			`${context}.n_distinct_evidence_counts`
		),
		evidenceMin: positiveInteger(obj.evidence_min, `${context}.evidence_min`),
		evidenceMax: positiveInteger(obj.evidence_max, `${context}.evidence_max`),
		nStatementsAgreeing: positiveInteger(
			verified.n_statements_agreeing,
			`${context}.verified_against.n_statements_agreeing`
		),
		nUniquePairs: positiveInteger(scope.n_unique_pairs, `${context}.unique_pair_scope.n_unique_pairs`),
		nEvidenceEntries: positiveInteger(
			scope.n_evidence_entries,
			`${context}.unique_pair_scope.n_evidence_entries`
		),
		nStatementsChangingBandUnderUniquePairs: nonNegativeInteger(
			scope.n_statements_changing_band_under_unique_pairs,
			`${context}.unique_pair_scope.n_statements_changing_band_under_unique_pairs`
		)
	};
}

/**
 * Pure, fail-closed parse of `ap_decomposition_by_paper_band.json`. THROWS on any
 * drift — shape, band arithmetic, arm order, y-domain escape, a banding variable
 * that is not the frozen evidence-count ladder, or a mirror summary that turns
 * out to be reporting its own tie-break. Callers that must not crash (i.e.
 * `validatePaperLiteral`) wrap it in their own try/catch and gate the whole load
 * to `unavailable`.
 */
export function validateApDecomposition(raw: unknown): ApDecomposition {
	const obj = record(raw, 'ap_decomposition');
	if (obj.artifact_kind !== 'paper_ap_decomposition_by_evidence_count') {
		fail('ap_decomposition.artifact_kind', 'expected paper_ap_decomposition_by_evidence_count');
	}
	if (obj.schema_version !== 2) fail('ap_decomposition.schema_version', 'expected 2');
	if (obj.metric !== 'pooled_average_precision') {
		fail('ap_decomposition.metric', 'expected pooled_average_precision');
	}

	const banding = parseBanding(obj.banding);
	const bands = parseBands(obj.bands);
	const bandTrueCounts = integerList(
		obj.band_true_counts,
		AP_DECOMP_BAND_COUNT,
		'ap_decomposition.band_true_counts'
	);
	const bandFalseCounts = integerList(
		obj.band_false_counts,
		AP_DECOMP_BAND_COUNT,
		'ap_decomposition.band_false_counts'
	);
	bands.forEach((band, index) => {
		if (band.nTrue !== bandTrueCounts[index] || band.nFalse !== bandFalseCounts[index]) {
			fail('ap_decomposition.band_true_counts', 'must agree with the per-band counts');
		}
	});

	const nStatements = positiveInteger(obj.n_statements, 'ap_decomposition.n_statements');
	const banded = bands.reduce((sum, band) => sum + band.n, 0);
	if (banded !== nStatements) {
		fail('ap_decomposition.bands', 'the bands must partition all n_statements');
	}
	if (banding.nStatementsAgreeing !== nStatements) {
		fail(
			'ap_decomposition.banding.verified_against.n_statements_agreeing',
			'the banding variable must agree with the paper’s own evidence census on every statement'
		);
	}

	if (!Array.isArray(obj.arms) || obj.arms.length !== AP_DECOMP_FAN_SLOTS.length) {
		fail('ap_decomposition.arms', `expected ${AP_DECOMP_FAN_SLOTS.length} arms in fan order`);
	}
	const arms = obj.arms.map((entry, index) => parseArm(entry, index));

	const checks = record(obj.checks, 'ap_decomposition.checks');
	// Band membership is a pure function of the evidence count, so this is 0 by
	// construction — and gating on it is how a re-banded artifact that quietly
	// went back to splitting tied statements gets caught.
	const nAssignedByTieBreak = exactly(
		checks.n_statements_assigned_by_a_tie_break,
		0,
		'ap_decomposition.checks.n_statements_assigned_by_a_tie_break'
	);
	const nReaderBeliefComparisons = positiveInteger(
		checks.n_reader_belief_comparisons,
		'ap_decomposition.checks.n_reader_belief_comparisons'
	);
	// The premise that disqualifies the unfitted noisy-OR as a banding variable:
	// it is every reader's own ceiling. If that stops holding, the caption's
	// justification is wrong and the figure must gate rather than argue it.
	const nReaderBeliefsExceedingNoisyOr = exactly(
		checks.n_reader_beliefs_exceeding_the_unfitted_noisy_or,
		0,
		'ap_decomposition.checks.n_reader_beliefs_exceeding_the_unfitted_noisy_or'
	);

	const referenceArm = text(obj.reference_arm, 'ap_decomposition.reference_arm');
	const bandingSensitivity = parseBandingSensitivity(obj.banding_sensitivity, referenceArm);

	const provenance = record(obj.provenance, 'ap_decomposition.provenance');
	const bootstrap = record(provenance.bootstrap, 'ap_decomposition.provenance.bootstrap');

	return {
		metric: 'pooled_average_precision',
		unit: text(obj.unit, 'ap_decomposition.unit'),
		referenceArm,
		referenceAveragePrecision: unit(
			obj.reference_average_precision,
			'ap_decomposition.reference_average_precision'
		),
		nStatements,
		nTrue: positiveInteger(obj.n_true, 'ap_decomposition.n_true'),
		nFalse: positiveInteger(obj.n_false, 'ap_decomposition.n_false'),
		banding,
		bands,
		bandTrueCounts,
		bandFalseCounts,
		arms,
		bandingSensitivity,
		nAssignedByTieBreak,
		nReaderBeliefsExceedingNoisyOr,
		nReaderBeliefComparisons,
		nBootstrap: positiveInteger(
			bootstrap.n_bootstrap,
			'ap_decomposition.provenance.bootstrap.n_bootstrap'
		),
		seed: positiveInteger(bootstrap.seed, 'ap_decomposition.provenance.bootstrap.seed')
	};
}

/**
 * Widest vertical separation, in AP points, between the three top-LLM cumulative
 * lines at any single band. The caption states it from the data instead of
 * quoting a constant that could go stale.
 */
export function topLlmBandSpreadPts(decomposition: ApDecomposition): number {
	const topLabels = new Set(
		AP_DECOMP_FAN_SLOTS.filter((slot) => slot.group === 'top-llm').map((slot) => slot.label)
	);
	const series = decomposition.arms
		.filter((arm) => topLabels.has(arm.label))
		.map((arm) => arm.cumulativePts);
	if (series.length === 0) return 0;
	let widest = 0;
	for (let band = 0; band < AP_DECOMP_BAND_COUNT; band += 1) {
		const values = series.map((line) => line[band]);
		widest = Math.max(widest, Math.max(...values) - Math.min(...values));
	}
	return widest;
}

// ---------------------------------------------------------------------------

export interface ApDecompMirrorBar {
	/** Rect origin and width in user units, on the strip's shared zero rule. */
	x: number;
	y: number;
	width: number;
	pts: number;
}

export interface ApDecompMirrorRow {
	key: ApDecompMirrorKey;
	/**
	 * Budget-checked, right-anchored row name. Never a frozen join key — the
	 * shipped contract runner asserts exactly that, and joins on this field name,
	 * so it keeps it.
	 */
	label: string;
	/**
	 * The same string, under the name the render layer reads. Separate field on
	 * purpose: the rule this page enforces is "nothing named `label` is rendered",
	 * and a rule with exceptions is a rule nobody can check at a glance.
	 */
	display: string;
	/** Budget-checked, left-anchored readout: "head −0.52  tail +0.95". */
	readout: string;
	kind: 'endogenous' | 'exogenous';
	drawn: boolean;
	/** Row top edge, relative to the strip group's own origin. */
	top: number;
	labelY: number;
	head: ApDecompMirrorBar;
	tail: ApDecompMirrorBar;
	tiltPts: number;
	/** Set on the first row of each `kind` group; null on the rest. */
	groupHeader: { display: string; y: number } | null;

}

/**
 * Group headers, keyed by `kind`. Right-anchored into the same 250-unit gutter as
 * the row labels and budget-checked with them.
 */
export const AP_DECOMP_MIRROR_GROUP_LABELS: Record<'endogenous' | 'exogenous', string> = {
	endogenous: 'banded by a score this figure is comparing',
	exogenous: 'banded by something outside both scores'
};

export interface ApDecompMirrorStrip {
	rows: ApDecompMirrorRow[];
	/** FROZEN join key of the arm drawn; use the fan slot's `display` to name it. */
	drawnArm: string;
	height: number;
	zeroX: number;
	/** Tick positions along the shared bar domain, in AP points. */
	ticks: number[];
	domainPts: number;
}

const MINUS = '−';

function signedPts(value: number): string {
	return `${value >= 0 ? '+' : MINUS}${Math.abs(value).toFixed(2)}`;
}


/**
 * Lay out the banding-sensitivity strip for one arm — by default the arm the
 * artifact nominates, which is the figure's headline arm.
 *
 * THROWS rather than clipping: an over-budget row label or a head/tail value
 * outside the fixed symmetric domain gates the figure. Callers wrap it the same
 * way they wrap `validateApDecomposition`.
 */
export function buildApDecompMirror(decomposition: ApDecomposition): ApDecompMirrorStrip {
	const geometry = AP_DECOMP_MIRROR_GEOMETRY;
	const sensitivity = decomposition.bandingSensitivity;
	const unitsPerPt = geometry.halfWidth / AP_DECOMP_MIRROR_DOMAIN_PTS;

	function bar(pts: number, top: number, context: string): ApDecompMirrorBar {
		if (Math.abs(pts) > AP_DECOMP_MIRROR_DOMAIN_PTS) {
			fail(
				context,
				`${pts} AP points escapes the fixed mirror domain ±${AP_DECOMP_MIRROR_DOMAIN_PTS}`
			);
		}
		const span = pts * unitsPerPt;
		return {
			x: span >= 0 ? geometry.zeroX : geometry.zeroX + span,
			y: top,
			width: Math.abs(span),
			pts
		};
	}

	let cursor = 0;
	let openGroup: string | null = null;
	const rows = sensitivity.variants.map((variant, index) => {
		const context = `ap_decomposition.banding_sensitivity.variants[${index}]`;
		const arm = variant.arms.find((entry) => entry.label === sensitivity.drawnArm);
		if (!arm) fail(context, `carries no row for the drawn arm ${sensitivity.drawnArm}`);

		let groupHeader: { display: string; y: number } | null = null;
		if (variant.kind !== openGroup) {
			groupHeader = {
				display: budget(
					AP_DECOMP_MIRROR_GROUP_LABELS[variant.kind],
					AP_DECOMP_MIRROR_LABEL_BUDGET_CHARS,
					`${context}.groupHeader`
				),
				y: cursor + geometry.groupHeaderBaselineY
			};
			cursor += geometry.groupHeaderHeight;
			openGroup = variant.kind;
		}
		const top = cursor;
		cursor += geometry.rowHeight;

		return {
			key: variant.key,
			label: budget(variant.display, AP_DECOMP_MIRROR_LABEL_BUDGET_CHARS, `${context}.label`),
			display: budget(variant.display, AP_DECOMP_MIRROR_LABEL_BUDGET_CHARS, `${context}.display`),
			readout: budget(
				`head ${signedPts(arm.headPts)}  tail ${signedPts(arm.tailPts)}`,
				AP_DECOMP_MIRROR_READOUT_BUDGET_CHARS,
				`${context}.readout`
			),
			kind: variant.kind,
			drawn: variant.drawn,
			top,
			labelY: top + geometry.labelBaselineY,
			head: bar(arm.headPts, top + geometry.headBarY, `${context}.head_pts`),
			tail: bar(arm.tailPts, top + geometry.tailBarY, `${context}.tail_pts`),
			tiltPts: arm.tiltPts,
			groupHeader
		};
	});

	return {
		rows,
		drawnArm: sensitivity.drawnArm,
		height: cursor,
		zeroX: geometry.zeroX,
		ticks: [-1, 0, 1],
		domainPts: AP_DECOMP_MIRROR_DOMAIN_PTS
	};
}
