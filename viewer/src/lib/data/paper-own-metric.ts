/**
 * The 2023 INDRA paper's OWN metric, with our arms placed on the same axis.
 *
 * METRIC (frozen by the published artifact's `metric_contract`): per-fold
 * `sklearn.precision_recall_curve` -> `auc(recall, precision)` (trapezoidal),
 * summarised as the arithmetic mean over 10 stratified folds, reported ± the
 * POPULATION STANDARD DEVIATION over those folds. That ± is a DISPERSION
 * measure. It is not a confidence interval, it is not an error bar on average
 * precision, and it belongs to the trapezoidal estimator alone — so this module
 * carries `foldMean`/`foldSd` and NOTHING else off `point_metrics`. Average
 * precision, AUROC, distinct-score counts and tie inflation are deliberately
 * absent here: they belong to the figure that follows this one.
 *
 * SLICE DISCIPLINE. The paper reports each method under four INPUT
 * CONFIGURATIONS (`readers` / `all sources`, each with and without
 * `include_more_specific`). Our panel is `all sources, specific`, so only those
 * 15 published rows are directly comparable; the other 44 render as an
 * explicitly-marked context band and never enter a comparison. See
 * `PAPER_METHOD_SLICES` in `./paper-method-landscape.ts`.
 *
 * NO PAIRED DELTA IS COMPUTED HERE, and none can be: the paper published rounded
 * fold summaries, not statement-level predictions. Co-plotting on a shared axis
 * is the strongest honest claim available, and the reproduction anchor (our
 * re-run of their code landing on their published row) is what licenses it.
 */

import type {
	PaperLiteralArm,
	PaperLiteralReproduction,
	ShippedProse
} from './paper-literal.ts';
import {
	PAPER_COMPARABLE_SLICE,
	PAPER_METHOD_SLICES,
	rankPublished,
	type PaperMethodLandscape,
	type PaperMethodRow,
	type PaperMethodSlice
} from './paper-method-landscape.ts';

/**
 * THE PLAIN HALF OF EVERY TWIN THIS MODULE EMITS.
 *
 * These three come off `indra_paper_2023_published_method_metrics.json`, and
 * this figure renders all three in ONE sentence — "…, …, reported ± the …" —
 * which is where "population standard deviation over the 10 folds" reached the
 * screen. The restatement says what the ± is in words a curator can act on; the
 * shipped wording travels beside it as the audit trail.
 */
const OWN_METRIC_PLAIN = {
	perFoldMetric:
		'within each fold: scikit-learn’s precision_recall_curve, then the area under it by ' +
		'joining adjacent points with straight lines',
	foldSummary: 'the plain average of those 10 fold results',
	uncertaintyField: 'spread of the 10 fold results around their own average'
} as const;

/**
 * SVG geometry, exported so the label budget below is derived from it rather
 * than eyeballed. Right-anchored SVG text that overruns its gutter loses its
 * LEADING glyphs silently — no layout error, no test failure — so both text
 * gutters are budgeted in characters and enforced in the builder.
 */
export const PAPER_OWN_METRIC_GEOMETRY = {
	width: 920,
	plotLeft: 212,
	plotRight: 812,
	/** Lane labels are right-anchored here; usable gutter is 0 → 204 units. */
	labelAnchorX: 204,
	/** Value readouts are left-anchored here; usable gutter is 820 → 920. */
	readoutX: 820,
	labelFontPx: 9,
	readoutFontPx: 8,
	/** Measured advance of the mono face at 9px, in user units per character. */
	monoUnitsPerChar: 5.4186,
	/** The same face at 8px: 5.4186 × 8/9. Used by the reference-label fit test. */
	readoutUnitsPerChar: 4.8165,
	laneHeight: 17,
	bandHeaderHeight: 30,
	topPad: 16,
	axisPad: 46,
	/** Vertical fan between marks sharing an anchor lane, in user units. */
	fanStep: 4.5
} as const;

/**
 * LABEL BUDGET (this page has shipped a silent right-anchored clip three times).
 * Lane labels: 204 units ÷ 5.4186 u/char at 9px = 37.6 → 37 characters. The
 * longest label the shipped artifacts produce is the paper's own
 * "RF 2k-d13 + Type/#PMIDs/prom/avglen" at 35 characters (189.7 units), leaving
 * 14.3 units of slack. `buildPaperOwnMetric` FAILS if any label exceeds the
 * budget, so a longer method name or arm display name gates the figure to
 * `unavailable` instead of quietly eating its first glyphs.
 */
export const PAPER_OWN_METRIC_LABEL_BUDGET_CHARS = 37;

/**
 * READOUT BUDGET: (920 − 820) units ÷ 4.8165 u/char at 8px = 20.8 → 20
 * characters. Longest shipped readout is a context range, "n=15 · 0.885–0.937"
 * at 18 characters. Left-anchored, so an overrun would clip the TRAILING
 * glyphs; budgeted and enforced all the same.
 */
export const PAPER_OWN_METRIC_READOUT_BUDGET_CHARS = 20;

/**
 * Each rendered series carries its own (stroke token, dash) pair plus its own
 * mark shape, so the figure survives greyscale and colour-vision deficiency.
 * Every stroke token clears 3:1 against --paper #fdfcf8 (WCAG 1.4.11):
 * --accent #7d2a1a = 9.2:1, --blocked #6f5a16 = 6.5:1, --ink-muted #6a6a6a = 5.3:1.
 */
export const PAPER_OWN_METRIC_GROUPS = [
	'published-comparable',
	'ours-reproduction',
	'ours-scored'
] as const;

export type PaperOwnMetricGroup = (typeof PAPER_OWN_METRIC_GROUPS)[number];

export interface PaperOwnMetricGroupStyle {
	group: PaperOwnMetricGroup;
	/** CSS custom property name, never a raw hex. */
	strokeVar: string;
	/** SVG stroke-dasharray for this series' dispersion bar; '' = solid. */
	dash: string;
	strokeWidth: number;
	shape: 'circle' | 'open-circle' | 'open-square' | 'diamond';
	legend: string;
}

export const PAPER_OWN_METRIC_GROUP_STYLES: Record<
	PaperOwnMetricGroup,
	PaperOwnMetricGroupStyle
> = {
	'published-comparable': {
		group: 'published-comparable',
		strokeVar: 'var(--accent)',
		dash: '',
		strokeWidth: 1.4,
		shape: 'circle',
		legend: 'published row, directly comparable configuration'
	},
	'ours-reproduction': {
		group: 'ours-reproduction',
		strokeVar: 'var(--accent)',
		dash: '1.5 1.5',
		strokeWidth: 1.4,
		shape: 'open-square',
		legend: 're-run of the published code for that row'
	},
	'ours-scored': {
		group: 'ours-scored',
		strokeVar: 'var(--blocked)',
		dash: '5 2',
		strokeWidth: 1.6,
		shape: 'diamond',
		legend: 'newly scored model, on the same statements'
	}
};

/**
 * Which published row each of our reproduction arms re-runs. Mirrors
 * `OOF_KEY_BY_ID` in `$lib/server/paper-literal` (the same strings address the
 * same released out-of-fold vectors); the semantic port reproduces the promoter
 * row too. Every value is checked against the comparable published rows at build
 * time, so a rename on either side gates the figure instead of drawing a false
 * pairing. Keyed by arm `id`, never by `display`.
 */
export const PAPER_REPRODUCED_ROW_BY_ARM_ID: Readonly<Record<string, string>> = {
	'paper-rf-promoter': 'RF 2k-d13 + Type/#PMIDs/promoter - all sources, specific',
	'paper-rf-prom-avglen': 'RF 2k-d13 + Type/#PMIDs/prom/avglen - all sources, specific',
	'port-rf-promoter': 'RF 2k-d13 + Type/#PMIDs/promoter - all sources, specific'
};

export interface PaperOwnMetricMark {
	key: string;
	/** On-screen name — a published method name, or an arm's `display`. */
	display: string;
	group: PaperOwnMetricGroup;
	foldMean: number;
	foldSd: number;
	/** Published rows are printed rounded to three decimals; ours are not. */
	rounded: boolean;
	/** Tooltip text, already carrying the not-a-CI wording. */
	title: string;
}

export interface PaperOwnMetricLane {
	/** Frozen join key (published `method_id`, arm `id`, or slice name) — never rendered. */
	key: string;
	/**
	 * Right-anchored lane name, budget-checked against `labelAnchorX`. Named
	 * `display` because this page's rule is that nothing called `label` is ever
	 * rendered — the frozen `point_metrics` keys are the things called `label`.
	 */
	display: string;
	readout: string;
	y: number;
	/**
	 * True where a published row and our re-run of it share the lane. These are
	 * the figure's credibility anchor, not a comparison. Only anchor lanes fan
	 * their marks vertically and draw the pairing connector.
	 */
	anchor: boolean;
	/**
	 * True for a lane that carries a whole configuration as one strip: marks sit
	 * on a single baseline with a span line, and per-mark dispersion bars are
	 * suppressed (14 overlapping bars is mush, and this band is context anyway —
	 * every one of those SDs is printed in the inspect table).
	 */
	strip: boolean;
	marks: PaperOwnMetricMark[];
}

export interface PaperOwnMetricBand {
	id: 'published-comparable' | 'ours';
	title: string;
	subtitle: string;
	/** False = the band is context; nothing in it may be read as a comparison. */
	comparable: boolean;
	headerY: number;
	lanes: PaperOwnMetricLane[];
}

export interface PaperOwnMetricAnchor {
	/** The published row we reproduce, at the paper's own printed precision. */
	publishedLabel: string;
	publishedMean: number;
	publishedSd: number;
	/** Our re-run of that row's code. */
	ourLabel: string;
	ourMean: number;
	ourSd: number;
	/** Our independent semantic port of the same model, where present. */
	portLabel: string | null;
	portMean: number | null;
	portSd: number | null;
	/** Manifest-sourced worst fold-mean deviation across every reproduced row. */
	maxAbsDeltaVsPublishedTable6: number | null;
}

export interface PaperOwnMetricFigure {
	bands: PaperOwnMetricBand[];
	domainMin: number;
	domainMax: number;
	ticks: number[];
	height: number;
	/** Their best row inside the comparable configuration — the only fair mark. */
	comparableBest: PaperMethodRow;
	comparableSlice: PaperMethodSlice;
	comparableCount: number;
	contextCount: number;
	anchor: PaperOwnMetricAnchor;
	/** Statements on our panel; null when the run manifest drifts. */
	nStatements: number | null;
	/** Verbatim manifest CV protocol; null when the manifest drifts. */
	cvProtocol: string | null;
	paperCodeCommit: string | null;
	paperNotebookCommit: string;
	/** Copied off the published artifact so the disclaimer cannot drift from it. */
	uncertaintyField: string;
	/** `uncertaintyField` with its plain restatement — `shipped` is byte-identical. */
	uncertaintyFieldProse: ShippedProse;
	uncertaintyIsConfidenceInterval: false;
	metricIsPooledAveragePrecision: false;
	perFoldMetric: string;
	/** `perFoldMetric` with its plain restatement — `shipped` is byte-identical. */
	perFoldMetricProse: ShippedProse;
	foldSummary: string;
	/** `foldSummary` with its plain restatement — `shipped` is byte-identical. */
	foldSummaryProse: ShippedProse;
}

export interface PaperOwnMetricOk {
	status: 'ok';
	figure: PaperOwnMetricFigure;
	reason: null;
	artifact_path: string;
	artifact_sha256: string;
}

export interface PaperOwnMetricUnavailable {
	status: 'unavailable';
	figure: null;
	reason: string;
	artifact_path: string;
	artifact_sha256: string | null;
}

export type PaperOwnMetricLoad = PaperOwnMetricOk | PaperOwnMetricUnavailable;

/** Manifest-sourced provenance threaded in by the server loader. */
export interface PaperOwnMetricProvenance {
	nStatements: number | null;
	reproduction: PaperLiteralReproduction | null;
}

function fail(context: string, message: string): never {
	throw new Error(`${context}: ${message}`);
}

/** Three decimals, matching the precision the paper prints its table at. */
export function fmt3(value: number): string {
	return value.toFixed(3);
}

function readout(mean: number, sd: number): string {
	return `${fmt3(mean)} ±${fmt3(sd)}`;
}

function budget(text: string, chars: number, context: string): string {
	if (text.length > chars) {
		fail(context, `"${text}" is ${text.length} chars; the gutter budget is ${chars}`);
	}
	return text;
}

function publishedMark(row: PaperMethodRow, group: PaperOwnMetricGroup): PaperOwnMetricMark {
	return {
		key: row.method_id,
		display: row.method,
		group,
		foldMean: row.fold_mean_trapezoidal_pr_auc,
		foldSd: row.fold_population_sd,
		rounded: true,
		title:
			`${row.method} — published ${fmt3(row.fold_mean_trapezoidal_pr_auc)}. ` +
			`The statements are split into 10 folds — 10 groups — and this model is fitted ten times, each time scored on the group it did not train on; ` +
			`across those 10 folds the score varies by ±${fmt3(row.fold_population_sd)} (population standard deviation). ` +
			`That is spread between folds, not a confidence interval.`
	};
}

/**
 * The two halves of the axis are not scored the same way, and the tooltip says
 * which one it is looking at. A refit model needs the split; a model that reads
 * evidence is never fitted on these labels, so nothing has to be withheld from
 * it — it is scored once over every statement and then assigned the same 10
 * fold indices so that the identical estimator applies to both halves. Stated
 * on the mark itself because it is the first thing a reviewer asks about a
 * figure that puts fitted and unfitted scores on one axis.
 */
function armMark(arm: PaperLiteralArm, group: PaperOwnMetricGroup): PaperOwnMetricMark {
	const refitted = arm.kind === 'paper' || arm.kind === 'port';
	return {
		key: arm.id,
		display: arm.display,
		group,
		foldMean: arm.trapezoidal,
		foldSd: arm.foldPopulationSd,
		rounded: false,
		title:
			`${arm.display} — ${fmt3(arm.trapezoidal)}, measured with trapezoidal PR-AUC, the formula the published rows use. ` +
			(refitted
				? `The statements are split into 10 folds — 10 groups — and this model is fitted ten times, each time scored on the group it did not train on; `
				: `This model is not fitted on these labels, so nothing has to be withheld from it: it scores every statement once and is then assigned the same 10 folds — the same 10 groups — so the identical estimator applies; `) +
			`across those 10 folds the score varies by ±${fmt3(arm.foldPopulationSd)} (population standard deviation). ` +
			`That is spread between folds, not a confidence interval.`
	};
}

/**
 * Domain snapped outward to the 0.01 grid, over every value the figure actually
 * DRAWS — bar ends for dispersion lanes, bare means for strip lanes — so nothing
 * is cropped and no axis width is spent on a bar that is never rendered.
 */
function domainOf(extents: number[]): { min: number; max: number; ticks: number[] } {
	let lo = Number.POSITIVE_INFINITY;
	let hi = Number.NEGATIVE_INFINITY;
	for (const value of extents) {
		lo = Math.min(lo, value);
		hi = Math.max(hi, value);
	}
	if (!Number.isFinite(lo) || !Number.isFinite(hi)) fail('domain', 'no marks to scale');
	// Integer hundredths throughout: float accumulation would drift the ticks.
	const minH = Math.max(0, Math.floor(lo * 100));
	const maxH = Math.min(100, Math.ceil(hi * 100));
	if (maxH - minH < 2) fail('domain', 'degenerate axis range');
	const ticks: number[] = [];
	for (let h = Math.ceil(minH / 2) * 2; h <= maxH; h += 2) ticks.push(h / 100);
	return { min: minH / 100, max: maxH / 100, ticks };
}

/**
 * Join the published landscape to our arms on the paper's own estimator.
 * Throws on any drift — a missing published counterpart, a slice census change,
 * an over-budget label — so the caller can gate to `unavailable` rather than
 * render a figure whose pairing or geometry is silently wrong.
 */
export function buildPaperOwnMetric(
	landscape: PaperMethodLandscape,
	arms: readonly PaperLiteralArm[],
	provenance: PaperOwnMetricProvenance
): PaperOwnMetricFigure {
	if (arms.length === 0) fail('arms', 'no arms to place on the axis');

	const byMethodName = new Map(landscape.methods.map((row) => [row.method, row]));
	const comparableRows = landscape.methods
		.filter((row) => row.slice === PAPER_COMPARABLE_SLICE)
		.sort(rankPublished);
	if (comparableRows.length !== landscape.slice_counts[PAPER_COMPARABLE_SLICE]) {
		fail('landscape.comparable', 'comparable-configuration census disagrees with the artifact');
	}

	// Our reproduction arms overlay the published lane they re-ran; every other
	// arm gets its own lane. Placement is by arm id, so a display rename is free.
	const reproductionByRow = new Map<string, PaperLiteralArm[]>();
	const scoredArms: PaperLiteralArm[] = [];
	for (const arm of arms) {
		const rowName = PAPER_REPRODUCED_ROW_BY_ARM_ID[arm.id];
		if (rowName === undefined) {
			scoredArms.push(arm);
			continue;
		}
		const row = byMethodName.get(rowName);
		if (!row) fail('reproduction', `published row "${rowName}" for arm ${arm.id} is missing`);
		if (row.slice !== PAPER_COMPARABLE_SLICE) {
			fail('reproduction', `published row "${rowName}" is not in the comparable configuration`);
		}
		const bucket = reproductionByRow.get(rowName);
		if (bucket) bucket.push(arm);
		else reproductionByRow.set(rowName, [arm]);
	}
	scoredArms.sort((a, b) => b.trapezoidal - a.trapezoidal || a.id.localeCompare(b.id));

	const geometry = PAPER_OWN_METRIC_GEOMETRY;
	let y = geometry.topPad;
	const bands: PaperOwnMetricBand[] = [];
	const allMarks: PaperOwnMetricMark[] = [];
	const extents: number[] = [];

	function pushBand(
		id: PaperOwnMetricBand['id'],
		title: string,
		subtitle: string,
		comparable: boolean,
		build: () => Omit<PaperOwnMetricLane, 'y'>[]
	): void {
		const headerY = y;
		y += geometry.bandHeaderHeight;
		const lanes = build().map((lane) => {
			const placed: PaperOwnMetricLane = { ...lane, y: y + geometry.laneHeight / 2 };
			y += geometry.laneHeight;
			budget(placed.display, PAPER_OWN_METRIC_LABEL_BUDGET_CHARS, `lane[${placed.key}].display`);
			budget(placed.readout, PAPER_OWN_METRIC_READOUT_BUDGET_CHARS, `lane[${placed.key}].readout`);
			allMarks.push(...placed.marks);
			for (const mark of placed.marks) {
				if (placed.strip) extents.push(mark.foldMean);
				else extents.push(mark.foldMean - mark.foldSd, mark.foldMean + mark.foldSd);
			}
			return placed;
		});
		bands.push({ id, title, subtitle, comparable, headerY, lanes });
	}

	const nRows = comparableRows.length;
	pushBand(
		'published-comparable',
		`published (${nRows})`,
		'open squares are the re-run of the published code',
		true,
		() =>
			comparableRows.map((row) => {
				const ours = reproductionByRow.get(row.method) ?? [];
				const published = readout(row.fold_mean_trapezoidal_pr_auc, row.fold_population_sd);
				return {
					key: row.method_id,
					display: row.base_method,
					// A shared lane carries more than one value, so its readout says
					// which one it prints; a solo lane cannot be misread.
					readout: ours.length > 0 ? `${published} pub` : published,
					anchor: ours.length > 0,
					strip: false,
					marks: [
						publishedMark(row, 'published-comparable'),
						...ours.map((arm) => armMark(arm, 'ours-reproduction'))
					]
				};
			})
	);

	const reproductionArmCount = arms.length - scoredArms.length;
	pushBand(
		'ours',
		`newly scored (${scoredArms.length})`,
		'same statements, same folds, same published labels',
		true,
		() =>
			scoredArms.map((arm) => ({
				key: arm.id,
				display: arm.display,
				readout: readout(arm.trapezoidal, arm.foldPopulationSd),
				anchor: false,
				strip: false,
				marks: [armMark(arm, 'ours-scored')]
			}))
	);

	// THE THIRD BAND IS GONE. It carried the 2023 paper's other 44 published rows —
	// the same methods fed different evidence (text-mining sources only, or without
	// evidence from more-specific claims), drawn as a range strip and never compared
	// to anything. A reader could not tell what it was for, and it was not load-bearing:
	// nothing on the page cites it and no claim rests on it. Removed on request rather
	// than relabelled again. The rows remain in
	// data/benchmark/indra_paper_2023_published_method_metrics.json and the landscape
	// loader still parses all 59, so nothing is lost from the record — only from the plate.
	// `contextCount` survives because the COUNT is still a true and useful fact: it is
	// what says the comparable 15 are a subset of 59, not the whole published table.
	const contextCount = PAPER_METHOD_SLICES.filter(
		(slice) => slice !== PAPER_COMPARABLE_SLICE
	).reduce((sum, slice) => sum + landscape.slice_counts[slice], 0);

	const placedArmKeys = new Set(
		allMarks.filter((mark) => mark.group !== 'published-comparable').map((mark) => mark.key)
	);
	if (placedArmKeys.size !== arms.length) {
		fail('arms', `placed ${placedArmKeys.size} of ${arms.length} arms exactly once`);
	}

	const anchorRow = landscape.comparable_best;
	const anchorArms = reproductionByRow.get(anchorRow.method) ?? [];
	const literalArm = anchorArms.find((arm) => arm.kind === 'paper');
	const portArm = anchorArms.find((arm) => arm.kind === 'port');
	if (!literalArm) {
		// Hard gate on purpose. Co-plotting our arms on the paper's axis is only
		// defensible because our re-run of their code lands on their best
		// comparable published row; with no such pairing the figure has lost its
		// premise, not just a caption.
		fail(
			'anchor',
			`no literal re-run is paired with the highest published comparable row "${anchorRow.method}" — the reproduction anchor that licenses this axis is gone`
		);
	}

	const scale = domainOf(extents);
	return {
		bands,
		domainMin: scale.min,
		domainMax: scale.max,
		ticks: scale.ticks,
		height: y + geometry.axisPad,
		comparableBest: anchorRow,
		comparableSlice: PAPER_COMPARABLE_SLICE,
		comparableCount: nRows,
		contextCount,
		anchor: {
			publishedLabel: anchorRow.method,
			publishedMean: anchorRow.fold_mean_trapezoidal_pr_auc,
			publishedSd: anchorRow.fold_population_sd,
			ourLabel: literalArm.display,
			ourMean: literalArm.trapezoidal,
			ourSd: literalArm.foldPopulationSd,
			portLabel: portArm?.display ?? null,
			portMean: portArm?.trapezoidal ?? null,
			portSd: portArm?.foldPopulationSd ?? null,
			maxAbsDeltaVsPublishedTable6:
				provenance.reproduction?.maxAbsDeltaVsPublishedTable6 ?? null
		},
		nStatements: provenance.nStatements,
		cvProtocol: provenance.reproduction?.cvProtocol ?? null,
		paperCodeCommit: provenance.reproduction?.paperCodeCommit ?? null,
		paperNotebookCommit: landscape.source.commit,
		uncertaintyField: landscape.metric_contract.uncertainty_field,
		uncertaintyFieldProse: {
			shipped: landscape.metric_contract.uncertainty_field,
			plain: OWN_METRIC_PLAIN.uncertaintyField
		},
		uncertaintyIsConfidenceInterval: landscape.metric_contract.uncertainty_is_confidence_interval,
		metricIsPooledAveragePrecision: landscape.metric_contract.metric_is_pooled_average_precision,
		perFoldMetric: landscape.metric_contract.per_fold_metric,
		perFoldMetricProse: {
			shipped: landscape.metric_contract.per_fold_metric,
			plain: OWN_METRIC_PLAIN.perFoldMetric
		},
		foldSummary: landscape.metric_contract.summary,
		foldSummaryProse: {
			shipped: landscape.metric_contract.summary,
			plain: OWN_METRIC_PLAIN.foldSummary
		}
	};
}
