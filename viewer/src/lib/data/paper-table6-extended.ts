/**
 * Typed data contract for the EXTENDED TABLE 6 — the "all sources, specific"
 * block published in the 2023 INDRA assembly paper, with the newly scored models
 * INTERLEAVED BY RANK.
 *
 * WHY THIS FIGURE EXISTS, given that `PaperOwnMetric` already plots the same
 * numbers. `PaperOwnMetric` bands the rows by provenance: published rows in one
 * band, newly scored rows in another, context in a third. That layout is the
 * right one for asking "does the re-run land on the published row", and it is the
 * wrong one for asking "where do the newly scored models sit in the list" —
 * banding is exactly what stops a reader seeing that ranks 1, 2 and 3 are newly
 * scored. This module produces one ordered list, rank 1..N, on trapezoidal PR-AUC
 * and on the reporting convention the 2023 tables use. Neither figure replaces
 * the other.
 *
 * THE ± IS DISPERSION, NOT AN INTERVAL. Each method is summarised as the
 * arithmetic mean of ten fold-wise `auc(recall, precision)` values ± the
 * POPULATION standard deviation over those folds. That is a spread statistic: it
 * does not shrink with the number of folds and it is not a test. The artifact
 * carries `uncertainty_is_dispersion_not_a_confidence_interval`, and this module
 * GATES on it being true — if the artifact ever stops asserting it, the figure
 * disappears rather than printing a ± a reader could take for a 95% interval.
 *
 * WHAT THE RANKING METRIC ADDS, RENDERED RATHER THAN HIDDEN. Trapezoidal PR-AUC
 * interpolates across tied score blocks, so it hands a coarse-scored model area
 * no cutoff can reach. Over these 1,689 statements that interpolation credit is
 * +0.0097..+0.0143 to the reading models and −0.0008..+0.0006 to the models
 * published in 2023. Every row therefore carries its stepped average precision
 * and its credit beside trapezoidal PR-AUC, and its rank on BOTH. The INDRA CoGEx
 * hybrid is the control that shows the effect tracks tie density rather than
 * authorship: it is newly scored, it emits 1,176 distinct scores, and it collects
 * +0.0006 — inside the range the 2023 rows sit in.
 *
 * HOW THE FOLDS WORK, and the asymmetry a reviewer will ask about.
 * `StratifiedKFold(10, shuffle=False)` over all 1,689 statements; nothing is held
 * out as a separate test set, and every statement is scored once by the fold that
 * excluded it. The FITTED models (random forest, logistic regression, KNN, SVC)
 * need that, and their numbers are out-of-fold. The READING models are never
 * trained and never see a label, so nothing has to be held out from them: each is
 * scored once over all 1,689 statements and then given the same fold indices,
 * purely so the identical estimator applies to every row. The one thing fitted on
 * everything is the score cutoff in the error-class F1 figure, which discloses it
 * there.
 *
 * SHIPPED FIELDS ARE READ, NEVER RECOMPUTED. Every mean, SD, average precision,
 * credit, rank, deviation and count comes off
 * `data/results/indra_paper_literal_models_20260724/paper_table6_extended.json`
 * verbatim. No metric is re-derived here. The only arithmetic is (a) rounding for
 * display and (b) CONSISTENCY GATES that difference or compare two already-shipped
 * scalars against a third — a credit against mean−AP, a deviation against
 * mean−published, an argmax against the label the artifact names, a shipped range
 * against the extremes of the rows its own labels name. Each of those fails the
 * figure closed rather than adjusting a number.
 *
 * `label` IS THE JOIN KEY AND IS NEVER RENDERED; `display` is the screen name.
 * The two are checked to differ. Newly scored rows additionally resolve, through
 * the frozen table below, to a canonical `paper-literal` arm id, so the models
 * listed here are provably the same models the rest of /paper measures.
 *
 * This module is import-safe on the client: typed shape plus a pure, fail-closed
 * validator. All filesystem work lives in `$lib/server/paper-table6-extended`.
 */

import { boolean, budget, fail, nonNegativeInteger, number, positiveInteger, record, text, unit } from './paper-validate.ts';
import { PAPER_LITERAL_ARM_SPECS, type ShippedProse } from './paper-literal.ts';

/**
 * THE PLAIN HALF OF EVERY TWIN THIS MODULE EMITS.
 *
 * `paper_table6_extended.json` is the artifact that ships the two coinages this
 * page most needs to stop saying: "gift", for the credit a coarse score
 * distribution collects from trapezoidal interpolation, and "tie-robust", for
 * the number that does not collect it. Both reach the screen at runtime off a
 * sha-pinned file. So does the measurement contract — "arithmetic mean over 10
 * cross-validation folds", "population standard deviation over the 10 folds" —
 * which is the single most important thing on the figure for a reader deciding
 * whether a ± here is a test. It is not, and the restatement says so first.
 *
 * FOLD, CROSS-VALIDATION and OUT-OF-FOLD are the field's own words and are kept:
 * the audience published a cross-validation paper, and swapping in a coinage
 * nobody uses is the same defect as "arm" or "tau" pointing the other way. Each
 * is glossed ONCE, at first use — `perFoldMetric` carries the gloss for this
 * module, which is where a reader meets the word in the methods paragraph.
 */
const TABLE6_PLAIN = {
	whatThisIs:
		'Table 6 of the 2023 INDRA assembly paper — its “all sources, specific” block — with ' +
		'newly scored models added as rows and the whole list ranked together on trapezoidal ' +
		'PR-AUC. The newly scored rows take places 1 to 3; the best fitted random forest in the ' +
		'published table is 4th.',
	rankingRule:
		'Ranked on trapezoidal PR-AUC, highest first; where two rows tie, the one whose result ' +
		'moves less between folds goes above; where they still tie, the stable row name breaks ' +
		'it. A total order, so a row’s place never depends on the order the rows arrived in — ' +
		'the same rule the table on screen uses.',
	perFoldMetric:
		'Within each fold — one of the 10 groups the statements are split into, each fitted ' +
		'model being scored on the group it did not train on — scikit-learn’s ' +
		'precision_recall_curve, then the area under it by joining adjacent points with ' +
		'straight lines.',
	summary: 'The plain average of those 10 fold results.',
	uncertaintyNote:
		'The ± is the spread of the 10 fold results around their own average. It describes how ' +
		'much the result MOVES from fold to fold. It is NOT a confidence interval, it does not ' +
		'shrink as you add folds, and no ± anywhere in this file may be read as a test.',
	trapezoidalNote:
		'Joining adjacent points with straight lines credits a model for ground it never ' +
		'actually covers, and it does so most when many statements share one score. That credit ' +
		'is worth up to +0.0143 to a model whose scores come in a few big blocks, and essentially ' +
		'nothing to one whose scores are finely spread — whoever built it.',
	estimatorSource:
		'sorgerlab/indra_assembly_paper@63abdf1274d2f5534ed822585775031712916c83 :: ' +
		'notebooks/Training Belief ML Models.ipynb — precision_recall_curve, then the area under ' +
		'it, once per fold, then the average and the spread across folds. The commit is quoted ' +
		'in full: an abbreviated one cannot be checked against the repository.',
	originPaperRerun:
		'the released code from the 2023 paper, re-run here on the published statements and ' +
		'labels, with every statement scored by a copy of the model that never saw it',
	originPaperPublishedOnly:
		'printed in the 2023 tables and never re-run here, so there are no per-statement scores ' +
		'for it and no tie-corrected number can exist',
	originOurs:
		'newly scored models, run on exactly the same statements and given exactly the same 10 ' +
		'fold indices — nothing here is fitted, so no fold is held out from them; the indices ' +
		'only make the estimator identical to the one the fitted rows are scored by',
	tieDefinition:
		'The interpolation credit is trapezoidal PR-AUC minus average precision computed over ' +
		'all 1,689 statements at once. Average precision does not interpolate, so the ' +
		'difference is what the straight-line joining was worth to that row.',
	tieReconciliation:
		'The credit looks like it tracks who wrote the model, until the INDRA CoGEx hybrid is ' +
		'read: it is one of the newly scored rows, it produces 1,176 distinct scores over 1,689 ' +
		'statements, and it collects only +0.0006 — inside the range the 2023 models sit in. ' +
		'Every model whose scores come in a few big blocks collects the credit and every model ' +
		'whose scores are finely spread does not, regardless of who built it.',
	tieSeparationNote:
		'No finely-scored model’s credit, in either direction, reaches any coarsely-scored ' +
		'model’s credit; the two groups do not overlap.',
	tieSpearmanMethod:
		'Spearman’s rho over MID-RANKS — tied values share their average place — computed with ' +
		'indra_belief.metrics._rankdata_avg over the 13 rows that have a score vector. The count ' +
		'of distinct scores itself contains ties here, so ranking by argsort-of-argsort would ' +
		'break them arbitrarily and would not be Spearman.',
	reproductionNote:
		'Published averages and spreads are printed to three decimals in the 2023 table, so how ' +
		'close a re-run row can possibly get is limited by that rounding; every re-run row is ' +
		'within 0.0016 of its published value.',
	conventionVerified:
		'The frozen extraction of the published 2023 tables carries exactly an average and a ' +
		'spread across folds for all 59 rows, and no field of any test-like kind.',
	headlineTieBreak:
		'two published rows tie at the best value; the one whose result moves less between folds ' +
		'wins, matching the ranking rule this file uses',
	headlineNote:
		'The improvement published in 2023 — the best fitted random forest over the unfitted ' +
		'Belief Orig baseline, in the same setting — is +0.0190, or 1.36 times the amount the ' +
		'result moves from fold to fold, and it was reported with no test at all. That is the ' +
		'scale against which every row in this table should be read.',
	/**
	 * `uncertainty_field` names the ± ON TWO SURFACES — this table and the
	 * own-metric figure — and both printed it raw. It is the shortest string on
	 * the page and the one a reader is most likely to mistake for an interval, so
	 * it gets a restatement of its own rather than leaning on `uncertaintyNote`.
	 */
	uncertaintyField: 'spread of the 10 fold results around their own average'
} as const;

/**
 * THE TIE-CORRECTED BEST-VS-BEST SENTENCE, restated.
 *
 * It is the only twin on this figure that is BUILT rather than written out, and
 * the reason is the substance of the sentence: the row that leads on the paper's
 * own estimator and the row that leads once the interpolation credit is removed
 * are DIFFERENT MODELS OF OURS. A restatement that says "our best model" twice
 * reads as one model and deletes the finding. A restatement that hard-codes two
 * names re-introduces the very thing the shipped sentence uses join keys for.
 * So the names arrive as resolved `display` fields, the numbers as parsed
 * values, and the verb as a three-way standing — never as the sign of a number
 * formatted at the call site.
 *
 * `renderedProse` scans the result for frozen join keys like every other
 * rendered string, so this half is renderable BECAUSE it is checked.
 */
const TABLE6_MARGIN_VERB: Readonly<Record<MarginStanding, string>> = {
	ahead: 'leads',
	behind: 'trails',
	level: 'is exactly level with'
};

/** What the tie correction did to the standing. One sentence per case. */
const TABLE6_TIE_BEST_OUTCOME: Readonly<Record<MarginStanding, string>> = {
	ahead: 'The lead survives the correction; most of what it started with does not.',
	behind:
		'The lead does NOT survive the correction: once the interpolation credit is removed the ' +
		'newly scored row is behind the published one.',
	level:
		'The lead does NOT survive the correction: once the interpolation credit is removed the ' +
		'two rows are exactly level.'
};

/**
 * The two margins the sentence carries, with the models each belongs to named
 * and the head-to-head figure it is reconciled against printed beside it.
 */
function table6TieBestPlain(input: {
	ourPaperMetricDisplay: string;
	theirPaperMetricDisplay: string;
	paperMetricMargin: number;
	paperMetricStanding: MarginStanding;
	ourApDisplay: string;
	theirApDisplay: string;
	apMargin: number;
	apStanding: MarginStanding;
	referenceDisplay: string;
	headToHeadApDelta: number;
	headToHeadTolerance: number;
}): string {
	return (
		`On trapezoidal PR-AUC the best-placed newly scored row is ${input.ourPaperMetricDisplay}, ` +
		`and it ${TABLE6_MARGIN_VERB[input.paperMetricStanding]} the best-placed published row, ` +
		`${input.theirPaperMetricDisplay}, by ${fmtSigned4(input.paperMetricMargin)}. Corrected for ` +
		`the interpolation credit, the leading newly scored row is a DIFFERENT model — ` +
		`${input.ourApDisplay}, best by average precision — and it ` +
		`${TABLE6_MARGIN_VERB[input.apStanding]} the best published row on that measure, ` +
		`${input.theirApDisplay}, by ${fmtSigned4(input.apMargin)}: within ` +
		`${fmt4(input.headToHeadTolerance)} of the average-precision margin the head-to-head ` +
		`reports for ${input.ourApDisplay} against ${input.referenceDisplay} ` +
		`(${fmtSigned4(input.headToHeadApDelta)}). ${TABLE6_TIE_BEST_OUTCOME[input.apStanding]}`
	);
}

/**
 * How close the tie-corrected best-vs-best margin must sit to the head-to-head's
 * own average-precision margin for the restatement's "within X" to be true. The
 * shipped sentence states 0.001; it is GATED here rather than asserted, so an
 * artifact whose two numbers drift apart takes the figure down instead of
 * printing a reconciliation that no longer reconciles.
 */
export const PAPER_TABLE6_TIE_BEST_HEAD_TO_HEAD_TOL = 0.001;

/** The artifact kind this module will accept, and nothing else. */
export const PAPER_TABLE6_EXTENDED_ARTIFACT_KIND = 'paper_table6_extended_all_sources_specific';

/** The one schema this module knows how to read. */
export const PAPER_TABLE6_EXTENDED_SCHEMA_VERSION = 1;

/**
 * The agreement bound that LICENSES putting our rows in their table: every row we
 * re-ran from the paper's own released code must land within this of the value
 * the paper printed. Independent of the artifact's own `tolerance` field, which
 * is checked against this rather than trusted — an artifact that relaxed its own
 * bound would otherwise widen the licence silently.
 */
export const PAPER_TABLE6_AGREEMENT_BOUND = 0.0016;

/**
 * The paper prints its tables to three decimals, so a published-only row's true
 * value lies within ±0.0005 of what we can read. Two adjacent ranks closer than
 * this are NOT separable at the paper's printed precision when either of them is
 * published-only, and the figure says so instead of presenting the ordering as
 * resolved. (On today's artifact this is exactly one adjacency: their unfitted
 * Belief Orig at 0.923 against their re-run Log LR at 0.9232.)
 */
export const PAPER_TABLE6_PUBLISHED_ROUNDING_HALF_WIDTH = 0.0005;

/**
 * How far a re-run row's fold SD may sit from the SD the paper PRINTED beside the
 * same row. Derived, not chosen: the printed figure is a three-decimal rounding
 * (±0.0005 by itself), and on top of that sits whatever the re-run moves, which
 * is bounded by the same agreement bound the means are held to. The worst row on
 * today's artifact is 0.00057 (RF 2k-d13 + avglen), well inside it. The SD is a
 * dispersion statistic and never a margin, so this bound licenses nothing — it
 * only stops a mutated `published_fold_population_sd` from going unnoticed.
 */
export const PAPER_TABLE6_PUBLISHED_SD_BOUND =
	PAPER_TABLE6_PUBLISHED_ROUNDING_HALF_WIDTH + PAPER_TABLE6_AGREEMENT_BOUND;

/** Cross-field consistency gates compare doubles at ~0.95; 1e-12 is decisive. */
export const PAPER_TABLE6_CONSISTENCY_TOLERANCE = 1e-12;

export const PAPER_TABLE6_ORIGINS = ['ours', 'paper_rerun', 'paper_published_only'] as const;

/**
 * The artifact's own family tag for our reader arms. It is what defines the set
 * `tie_disclosure.llm_reader_arms` must summarise, so the group's two headline
 * numbers are checked against the rows this tag selects rather than taken on
 * trust from the group's own `labels`. See `PAPER_TABLE6_FAMILIES` for why the
 * tag itself is enum-checked before any gate is allowed to read it.
 */
export const PAPER_TABLE6_LLM_READER_FAMILY: PaperTable6Family = 'llm_reader';

export type PaperTable6Origin = (typeof PAPER_TABLE6_ORIGINS)[number];

/**
 * The closed set of family tags. CLOSED ON PURPOSE, and validated before any gate
 * reads the field: `family` is what defines the comparison group the reader-arm
 * tie range is checked against, and what selects the two rows the paper's own
 * headline gain differences. A free-text tag lets a regenerated artifact define
 * its own group and then satisfy an aggregate against it — the gate would still
 * pass, having compared a claim to a class the same file invented. Adding a
 * family is therefore a deliberate edit here, not a string an artifact can coin.
 */
export const PAPER_TABLE6_FAMILIES = [
	'llm_reader',
	'paper_fitted_ml',
	'paper_unfitted_belief',
	'indra_hybrid'
] as const;

export type PaperTable6Family = (typeof PAPER_TABLE6_FAMILIES)[number];

/**
 * Which side of the table each family belongs to. The enum alone bounds what a
 * tag may say; this bounds what it may say about a row whose ORIGIN is already
 * gated — retagging one of the paper's rows as `llm_reader` to widen or narrow
 * our reader group is caught here rather than propagating into the tie range.
 */
export const PAPER_TABLE6_FAMILY_IS_OURS: Readonly<Record<PaperTable6Family, boolean>> = {
	llm_reader: true,
	indra_hybrid: true,
	paper_fitted_ml: false,
	paper_unfitted_belief: false
};

/** The paper's own FITTED models — the class its published headline gain tops. */
export const PAPER_TABLE6_FITTED_FAMILY: PaperTable6Family = 'paper_fitted_ml';

/** The paper's unfitted baseline — the other end of that same published gain. */
export const PAPER_TABLE6_UNFITTED_FAMILY: PaperTable6Family = 'paper_unfitted_belief';

/**
 * Which canonical `paper-literal` arm each of OUR rows is, keyed on the
 * artifact's frozen row `label`. This is the indirection that keeps a display
 * string from ever becoming a join key: the artifact's `display` ("GLM-5") is
 * shaped like the head-to-head's frozen `point_metrics` key, and reaching for it
 * as a key is the defect that has regressed seven times on this page. An `ours`
 * row with no entry here gates the figure — a new arm must be re-pinned
 * deliberately, not absorbed silently.
 */
export const PAPER_TABLE6_OUR_ARM_ID_BY_LABEL: Readonly<Record<string, string>> = {
	ours_glm_5: 'glm-5',
	ours_gemma_4_26b: 'gemma-4-26b',
	ours_gemma_4_31b: 'gemma-4-31b',
	ours_gemma_4_e2b: 'gemma-4-e2b',
	ours_indra_cogex_hybrid: 'indra-cogex-hybrid'
};

/**
 * SVG geometry, exported so the character budgets below are DERIVED from it and
 * so a contract runner can re-derive them. Right-anchored SVG text that overruns
 * its gutter loses its LEADING glyphs silently: no layout error, no test failure,
 * and the <desc> beside it still emits the full string, so screen readers and
 * a11y checks both report success. Every gutter here is measured and enforced in
 * `buildFigure`, which throws rather than clipping.
 */
export const PAPER_TABLE6_GEOMETRY = {
	width: 920,
	/** Rank numerals are right-anchored here; usable gutter is 0 → 26 units. */
	rankAnchorX: 26,
	/** Method names start no further left than this. */
	labelGutterLeft: 32,
	/** Method names are right-anchored here; usable gutter is 32 → 252 units. */
	labelAnchorX: 252,
	plotLeft: 264,
	plotRight: 662,
	/** Paper-metric readouts are left-anchored here; gutter is 672 → 764. */
	metricX: 672,
	/** Tie-robust readouts are left-anchored here; gutter is 764 → 920. */
	tieX: 764,
	labelFontPx: 9,
	readoutFontPx: 8,
	/** Measured advance of the mono face at 9px, in user units per character. */
	monoUnitsPerChar: 5.4186,
	/** The same face at 8px: 5.4186 × 8/9. */
	readoutUnitsPerChar: 4.8165,
	laneHeight: 17,
	topPad: 44,
	/** Baseline of the single column-header line. */
	headerY: 20,
	/** The rule under the column headers. */
	headerRuleY: 27,
	axisPad: 52,
	/** Half-height of a fold-SD bar's end cap. */
	sdCap: 3,
	/** Half-height of the hairline marking a published value. */
	publishedTick: 4,
	/** The tie-gift connector is drawn this far below its lane's baseline. */
	giftOffsetY: 5.5,
	markRadius: 3.4
} as const;

/**
 * METHOD-NAME BUDGET: (252 − 32) = 220 units ÷ 5.4186 u/char at 9px = 40.6 → 40
 * characters. The longest name either side can produce is the paper's own
 * "RF 2k-d13 + Type/#PMIDs/prom/avglen" at 35 characters (189.7 units), which
 * anchors at x = 62.3 — 30.3 units clear of the 32-unit gutter wall. Vertically,
 * a 9px line box is ~10.8 units inside a 17-unit lane, so consecutive baselines
 * (17 units apart) never collide. `buildFigure` FAILS on an over-budget name.
 */
export const PAPER_TABLE6_LABEL_BUDGET_CHARS = 40;

/**
 * RANK BUDGET: 26 units ÷ 4.8165 u/char at 8px = 5.4 → 5 characters. The widest
 * rank this artifact can print is "20" (9.6 units), anchoring at x = 16.4.
 */
export const PAPER_TABLE6_RANK_BUDGET_CHARS = 5;

/**
 * PAPER-METRIC READOUT BUDGET: (764 − 672) = 92 units ÷ 4.8165 u/char at 8px =
 * 19.1 → 19 characters. Longest shipped readout is a full-precision row,
 * "0.9649 ±0.0103" at 14 characters (67.4 units). Left-anchored, so an overrun
 * would clip TRAILING glyphs and collide with the tie column; budgeted anyway.
 */
export const PAPER_TABLE6_METRIC_BUDGET_CHARS = 19;

/**
 * TIE-ROBUST READOUT BUDGET: (920 − 764) = 156 units ÷ 4.8165 u/char at 8px =
 * 32.4 → 32 characters. Longest shipped readout is "0.9228  +0.0006  AP rank 13"
 * at 27 characters (130.0 units).
 */
export const PAPER_TABLE6_TIE_BUDGET_CHARS = 32;

export interface PaperTable6OriginStyle {
	origin: PaperTable6Origin;
	/** CSS custom property name, never a raw hex. */
	strokeVar: string;
	/** SVG stroke-dasharray for the fold-SD bar; '' = solid. */
	dash: string;
	strokeWidth: number;
	shape: 'diamond' | 'circle' | 'open-circle';
	legend: string;
	/**
	 * The origin as a SPOKEN phrase, for the `<desc>` row enumeration. Origin is
	 * this figure's primary visual variable — it is what colour, mark shape and
	 * dash all encode — so a row read aloud without it is missing the one fact the
	 * whole layout exists to carry. Short on purpose: it is repeated once per row.
	 * The artifact's own `origins.*` strings are a paragraph each and are printed
	 * in the table instead.
	 */
	spoken: string;
}

/**
 * Each origin carries its own (stroke token, dash, mark shape), so the list
 * survives greyscale and colour-vision deficiency. Tokens match the rest of
 * /paper — --accent marks a published row, --blocked a newly scored one — and
 * each clears 3:1 against
 * --paper #fdfcf8 (WCAG 1.4.11): --accent #7d2a1a = 9.2:1, --blocked #6f5a16 =
 * 6.5:1, --ink-muted #6a6a6a = 5.3:1.
 */
export const PAPER_TABLE6_ORIGIN_STYLES: Record<PaperTable6Origin, PaperTable6OriginStyle> = {
	ours: {
		origin: 'ours',
		strokeVar: 'var(--blocked)',
		dash: '',
		strokeWidth: 1.6,
		shape: 'diamond',
		legend:
			'newly scored — run on exactly the same statements, with the published labels, under the same 10 folds',
		spoken: 'newly scored'
	},
	paper_rerun: {
		origin: 'paper_rerun',
		strokeVar: 'var(--accent)',
		dash: '',
		strokeWidth: 1.4,
		shape: 'circle',
		legend: 'published, re-run — the released 2023 code, re-run here; the hairline marks the published value',
		spoken: 'published, re-run here'
	},
	paper_published_only: {
		origin: 'paper_published_only',
		strokeVar: 'var(--ink-muted)',
		dash: '3 3',
		strokeWidth: 1,
		shape: 'open-circle',
		legend:
			'published, printed only — never re-run here, so there are no per-statement scores and no stepped area',
		spoken: 'published, printed only'
	}
};

export interface PaperTable6Row {
	/** FROZEN artifact join key. Never rendered — render `display`. */
	label: string;
	/** On-screen name: the paper's own row name, or our arm's name. */
	display: string;
	origin: PaperTable6Origin;
	/** Enum-checked against `PAPER_TABLE6_FAMILIES` before any gate reads it. */
	family: PaperTable6Family;
	rank: number;
	/** SHIPPED `fold_mean_trapezoidal_pr_auc` — the paper's own estimator. */
	foldMean: number;
	/** SHIPPED `fold_population_sd` — DISPERSION over the folds, never an interval. */
	foldSd: number;
	foldCount: number;
	/** True where the only value that exists is the paper's printed 3dp figure. */
	rounded: boolean;
	/** The value the paper printed, where there is one. */
	publishedMean: number | null;
	/** SHIPPED |re-run − published|; null unless we re-ran the row. */
	absDevVsPublished: number | null;
	/** SHIPPED tie-robust `pooled_average_precision`; null with no score vector. */
	ap: number | null;
	/** SHIPPED rank on that tie-robust metric, among the rows that have one. */
	apRank: number | null;
	/** SHIPPED `tie_gift` = paper estimator − tie-robust AP. */
	tieGift: number | null;
	distinctScores: number | null;
	/** The arm every paired delta on this page is measured against. */
	isReference: boolean;
	/** How the head-to-head names this arm; null for the paper's own rows. */
	headToHeadDisplay: string | null;
	/**
	 * True when this row and its neighbour are closer than the paper's own 3dp
	 * printing can resolve, and one of them exists only at that precision.
	 */
	rankAmbiguousAtPrintedPrecision: boolean;
	y: number;
	rankReadout: string;
	metricReadout: string;
	tieReadout: string;
	title: string;
}

export interface PaperTable6MetricContract {
	perFoldMetric: string;
	summary: string;
	uncertaintyField: string;
	/** `uncertaintyField` with its plain restatement — `shipped` is byte-identical. */
	uncertaintyFieldProse: ShippedProse;
	uncertaintyNote: string;
	trapezoidalNote: string;
	estimatorSource: string;
	/** GATED false: this ± is not an interval. */
	uncertaintyIsConfidenceInterval: false;
	/** GATED true: the artifact must keep asserting the ± is dispersion. */
	uncertaintyIsDispersion: true;
	/** GATED false: the ranking metric is trapezoidal, not average precision. */
	metricIsPooledAveragePrecision: false;
}

export interface PaperTable6Reproduction {
	nRerunRows: number;
	nRerunRowsWithScores: number;
	maxAbsDev: number;
	/** Screen name of the worst-deviating row. */
	maxAbsDevDisplay: string;
	tolerance: number;
	publishedValuesRoundedTo3dp: true;
	note: string;
}

export interface PaperTable6TieGroup {
	min: number;
	max: number;
	count: number;
	distinctScoresMin: number;
	distinctScoresMax: number;
}

export interface PaperTable6TieControl {
	display: string;
	tieGift: number;
	distinctScores: number;
}

export interface PaperTable6TieSeparation {
	coarseMaxDistinctScores: number;
	minGiftAmongCoarse: number;
	maxAbsGiftAmongFine: number;
	nCoarse: number;
	nFine: number;
	note: string;
}

/** Which way a margin points. Never inferred at a render site. */
export type MarginStanding = 'ahead' | 'behind' | 'level';

/**
 * A margin's direction, decided once. `level` is reserved for an exact zero: this
 * pair carries no interval, so "level" here means the two rows tie, not that a
 * difference failed a significance test.
 */
function marginStanding(margin: number): MarginStanding {
	if (margin > 0) return 'ahead';
	if (margin < 0) return 'behind';
	return 'level';
}

export interface PaperTable6BestVsBest {
	/** Top row on the paper's own estimator, ours and theirs. */
	ourPaperMetricDisplay: string;
	theirPaperMetricDisplay: string;
	paperMetricMargin: number;
	/** Top row tie-corrected — a DIFFERENT arm of ours, which is the point. */
	ourApDisplay: string;
	theirApDisplay: string;
	ourAp: number;
	theirAp: number;
	apMargin: number;
	/**
	 * The head-to-head's own average-precision margin for the same pair, read off
	 * the artifact and gated to within `PAPER_TABLE6_TIE_BEST_HEAD_TO_HEAD_TOL` of
	 * `apMargin`. Carried so the reconciliation the sentence claims is a checked
	 * number rather than a figure typed into a caption.
	 */
	headToHeadApDelta: number;
	/**
	 * DIRECTION, DERIVED — sign-blindness occurrence #5 was here.
	 *
	 * Both margins were gated only against the two rows they name, never against
	 * zero, while the prose beside them hard-coded "leads" and "the lead survives
	 * the correction". On a fully self-consistent artifact with every gate green
	 * that rendered "leads theirs by −0.0010. The lead survives the correction."
	 * The component now selects its verb on these fields and never on the sign of
	 * a number it formats. Same fix as `standingFrom` in paper-error-f1.ts.
	 */
	paperMetricStanding: MarginStanding;
	apStanding: MarginStanding;
	referenceDisplay: string;
	/**
	 * The artifact's own reconciliation sentence. READ AND GATED, NEVER RENDERED:
	 * it names its arms by their frozen row labels ("ours_glm_5"), and a frozen
	 * join key shown to a reader is a key someone will reasonably ask us to
	 * rename. Every claim it makes is rendered from the resolved `display` fields
	 * above instead. `assertNoJoinKeysInProse` enforces the split.
	 */
	noteWithJoinKeys: string;
	/**
	 * The SAME sentence with its plain restatement, and the restatement names its
	 * models by DISPLAY. This used to be the one shipped sentence on the figure
	 * that could not be shown at all — the join keys made it unrenderable, so a
	 * real reconciliation was carried and never read. The plain half joins the
	 * join-key scan on the same footing as every other rendered string, so it is
	 * renderable BECAUSE it is checked, not in spite of being unchecked. Render
	 * `noteProse.plain`; `noteProse.shipped` stays audit-only like the flat field.
	 */
	noteProse: ShippedProse;
}

export interface PaperTable6Tie {
	definition: string;
	readers: PaperTable6TieGroup;
	paperRerun: PaperTable6TieGroup;
	control: PaperTable6TieControl;
	separation: PaperTable6TieSeparation;
	best: PaperTable6BestVsBest;
	reconciliation: string;
	spearman: number;
	spearmanMethod: string;
}

export interface PaperTable6HeadlineGain {
	bestFittedMean: number;
	bestFittedSd: number;
	beliefOrigMean: number;
	beliefOrigSd: number;
	gain: number;
	gainInFoldSd: number;
	tieBreak: string;
	note: string;
}

export interface PaperTable6Convention {
	nPublishedRows: number;
	reportedStatistics: string[];
	nPValues: number;
	nConfidenceIntervals: number;
	nMultiplicityCorrections: number;
	verified: string;
	repository: string;
	commit: string;
	notebookPath: string;
	notebookSha256: string;
	headline: PaperTable6HeadlineGain;
}

/**
 * One PAIR of rows — adjacent or not — the paper's own printed precision cannot
 * resolve. Non-adjacent pairs are real: three rows inside half a printed digit
 * of each other make three unresolvable pairs, and a scan that walked only
 * adjacencies would report two of them and present the third rank gap as settled.
 */
export interface PaperTable6AmbiguousPair {
	/**
	 * Stable list key, built from the two RANKS. Not a display string (two rows
	 * may legitimately share one) and never a frozen join key.
	 */
	id: string;
	higherDisplay: string;
	lowerDisplay: string;
	gap: number;
}

/**
 * One pair whose two READOUTS are the same string, though the underlying values
 * differ and the order is real. Reported so the figure can say which ranks its
 * own printing cannot separate, rather than leaving a reader to assume two
 * identical numbers were ordered arbitrarily. Also scanned over all pairs, not
 * just adjacencies: rows printed at different precisions sit between rows that
 * print identically.
 */
export interface PaperTable6PrintedTie {
	/** Stable list key, built from the two RANKS. See `PaperTable6AmbiguousPair`. */
	id: string;
	higherDisplay: string;
	lowerDisplay: string;
	readout: string;
	gap: number;
}

/**
 * Every shipped sentence this figure carries, each with the plain restatement a
 * reader is handed instead. One bag: the question a render site asks is "which
 * of these strings do I have a plain half for", and the answer should be a list
 * it cannot read past.
 *
 * `noteWithJoinKeys` is deliberately ABSENT. That sentence names its rows by
 * their frozen join keys and is already read-and-gated-never-rendered; giving it
 * a plain twin would invite a render site to print it.
 */
export interface PaperTable6Prose {
	whatThisIs: ShippedProse;
	rankingRule: ShippedProse;
	/** What is computed inside one fold — and, at first use, what a fold is. */
	perFoldMetric: ShippedProse;
	/** How the 10 fold results become one number. */
	summary: ShippedProse;
	/** That the ± is spread, not a test. The most load-bearing line here. */
	uncertaintyNote: ShippedProse;
	/** What the straight-line joining is worth, and to whom. */
	trapezoidalNote: ShippedProse;
	estimatorSource: ShippedProse;
	/** Keyed by the frozen origin id, exactly like `origins`. */
	origins: Record<PaperTable6Origin, ShippedProse>;
	tieDefinition: ShippedProse;
	tieReconciliation: ShippedProse;
	tieSeparationNote: ShippedProse;
	tieSpearmanMethod: ShippedProse;
	reproductionNote: ShippedProse;
	conventionVerified: ShippedProse;
	headlineTieBreak: ShippedProse;
	headlineNote: ShippedProse;
}

export interface PaperTable6Figure {
	rows: PaperTable6Row[];
	origins: Record<PaperTable6Origin, string>;
	config: string;
	whatThisIs: string;
	rankingRule: string;
	/**
	 * The plain half of every string this figure took off the artifact. Each flat
	 * field is byte-identical to its twin's `shipped`; the twin is the one to
	 * render, and the shipped half belongs behind the verification boundary.
	 */
	prose: PaperTable6Prose;
	nStatements: number;
	nPositive: number;
	nNegative: number;
	nFolds: number;
	metric: PaperTable6MetricContract;
	reproduction: PaperTable6Reproduction;
	tie: PaperTable6Tie;
	convention: PaperTable6Convention;
	ambiguousPairs: PaperTable6AmbiguousPair[];
	printedTies: PaperTable6PrintedTie[];
	/** How many leading ranks are ours — the fact the banded chart cannot show. */
	leadingOurRanks: number;
	/** Rank of the paper's best fitted row, i.e. where their table now starts. */
	theirBestRank: number;
	/** GATED true: the paper reports no p-value, interval or correction. */
	paperReportsNoTests: true;
	generatedBy: string;
	domainMin: number;
	domainMax: number;
	ticks: number[];
	height: number;
	plotBottom: number;
	/** Full text of the column headers, budget-checked like every other string. */
	rankHeader: string;
	labelHeader: string;
	metricHeader: string;
	tieHeader: string;
}

export interface PaperTable6ExtendedOk {
	status: 'ok';
	figure: PaperTable6Figure;
	reason: null;
	artifact_path: string;
	/**
	 * NULLABLE on the ok branch too. The server always supplies a digest; a pure
	 * client-side validate path (the contract runner) does not, and coalescing it
	 * to '' would print an empty provenance line that reads as a real, empty sha.
	 * Null is typed so the compiler finds every render site and each states it.
	 */
	artifact_sha256: string | null;
}

export interface PaperTable6ExtendedUnavailable {
	status: 'unavailable';
	figure: null;
	reason: string;
	artifact_path: string;
	artifact_sha256: string | null;
}

export type PaperTable6ExtendedLoad = PaperTable6ExtendedOk | PaperTable6ExtendedUnavailable;

export interface PaperTable6ExtendedContext {
	artifactPath?: string;
	artifactSha256?: string;
}

type UnknownRecord = Record<string, unknown>;





function nullableUnit(value: unknown, context: string): number | null {
	return value === null ? null : unit(value, context);
}


function nullablePositiveInteger(value: unknown, context: string): number | null {
	return value === null ? null : positiveInteger(value, context);
}




/**
 * Exactly `want`, or the figure gates. Used where a flipped flag would relabel a
 * dispersion statistic as an interval, or a printed-only row as a measured one.
 */
function exactly(value: unknown, want: boolean, context: string): boolean {
	const parsed = boolean(value, context);
	if (parsed !== want) fail(context, `expected ${want}`);
	return parsed;
}

function exactlyNull(value: unknown, context: string): null {
	if (value !== null) fail(context, 'expected null');
	return null;
}

function agrees(a: number, b: number): boolean {
	return Math.abs(a - b) <= PAPER_TABLE6_CONSISTENCY_TOLERANCE;
}

/** Three decimals — the precision the paper prints its own tables at. */
export function fmt3(value: number): string {
	return value.toFixed(3);
}

/** Four decimals — the precision at which this panel's ranks are separable. */
export function fmt4(value: number): string {
	return value.toFixed(4);
}

/**
 * Four decimals, rounded AWAY FROM ZERO. For numbers printed as a BOUND — "every
 * row lands within X" — where `fmt4` rounds to nearest and so can print a bound
 * the rows do not satisfy: a worst deviation of 0.00114 renders "0.0011" under
 * `fmt4`, and the sentence around it is then false of the very row it summarises.
 * The 1e-9 slack keeps a value that IS exact at four decimals from being pushed
 * up a digit by its own binary representation (0.0016 × 1e4 = 16.000000000000004).
 */
export function fmt4Ceil(value: number): string {
	return (Math.ceil(Math.abs(value) * 1e4 - 1e-9) / 1e4).toFixed(4);
}

/**
 * Spearman's ρ over MID-RANKS: Pearson over ranks where tied values share their
 * average rank. Mid-ranks are the whole point here — `distinct_scores` carries
 * ties on this table (two pairs of the paper's rows), and ranking by
 * argsort-of-argsort would break them by array position, which is not Spearman
 * and is the exact defect already fixed once on this page's fidelity panel.
 *
 * Returns null rather than 0 on a degenerate input (fewer than three rows, or a
 * column with no spread), so the caller GATES instead of comparing a shipped
 * correlation against a fabricated zero.
 *
 * Deliberately local: `$lib/server/belief-heuristic` holds the same computation,
 * but this module is import-safe on the client and may not reach into `$lib/server`.
 * The canonical mid-rank helper is `indra_belief.metrics._rankdata_avg`, which is
 * what produced the shipped value this re-derivation is checked against.
 */
function spearmanMidRank(xs: readonly number[], ys: readonly number[]): number | null {
	const n = xs.length;
	if (n !== ys.length || n < 3) return null;
	const midRanks = (values: readonly number[]): number[] => {
		const order = values.map((value, index) => [value, index] as const).sort((a, b) => a[0] - b[0]);
		const out = new Array<number>(values.length);
		for (let i = 0; i < order.length; ) {
			let j = i;
			while (j + 1 < order.length && order[j + 1][0] === order[i][0]) j += 1;
			const avg = (i + j) / 2 + 1;
			for (let k = i; k <= j; k += 1) out[order[k][1]] = avg;
			i = j + 1;
		}
		return out;
	};
	const rx = midRanks(xs);
	const ry = midRanks(ys);
	const mx = rx.reduce((a, b) => a + b, 0) / n;
	const my = ry.reduce((a, b) => a + b, 0) / n;
	let num = 0;
	let dx = 0;
	let dy = 0;
	for (let i = 0; i < n; i += 1) {
		num += (rx[i] - mx) * (ry[i] - my);
		dx += (rx[i] - mx) ** 2;
		dy += (ry[i] - my) ** 2;
	}
	if (!(dx > 0 && dy > 0)) return null;
	return num / Math.sqrt(dx * dy);
}

/**
 * Signed, four decimals, ASCII '+'/'-' on purpose: every gutter budget here is
 * measured in characters against a MEASURED mono advance, and a typographic minus
 * that fell back to another face would be a different width — silently breaking
 * the one guard that keeps this text from clipping.
 */
export function fmtSigned4(value: number): string {
	return `${value >= 0 ? '+' : '-'}${Math.abs(value).toFixed(4)}`;
}


interface ParsedRow {
	label: string;
	display: string;
	origin: PaperTable6Origin;
	family: PaperTable6Family;
	rank: number;
	foldMean: number;
	foldSd: number;
	foldCount: number;
	publishedMean: number | null;
	/** The SD the paper PRINTED beside this row; null where they printed none. */
	publishedSd: number | null;
	absDevVsPublished: number | null;
	ap: number | null;
	apRank: number | null;
	tieGift: number | null;
	distinctScores: number | null;
	hasScores: boolean;
	isReference: boolean;
}

function parseRow(
	entry: unknown,
	index: number,
	nFolds: number,
	nStatements: number
): ParsedRow {
	const context = `rows[${index}]`;
	const row = record(entry, context);

	const label = text(row.label, `${context}.label`);
	const display = text(row.display, `${context}.display`);
	// The rule this page keeps regressing on: the join key is not the screen name.
	// If they ever collapse into one string, the next rename silently repoints a
	// join — so they are required to differ.
	if (display === label) {
		fail(`${context}.display`, `must differ from the join key "${label}"`);
	}

	const originRaw = text(row.origin, `${context}.origin`);
	if (!(PAPER_TABLE6_ORIGINS as readonly string[]).includes(originRaw)) {
		fail(`${context}.origin`, `"${originRaw}" is outside the origin enum`);
	}
	const origin = originRaw as PaperTable6Origin;

	// Validated HERE, beside `origin`, and not at the end beside the fields that
	// only get rendered: `family` is a CLASS DEFINITION. Downstream gates use it to
	// select the group whose tie range, and whose published headline gain, they
	// then check — so a free-text tag would let the artifact draw its own
	// comparison group and satisfy an aggregate against it. Bounded twice: to the
	// enum, and to the side of the table the already-gated `origin` puts the row on.
	const familyRaw = text(row.family, `${context}.family`);
	if (!(PAPER_TABLE6_FAMILIES as readonly string[]).includes(familyRaw)) {
		fail(`${context}.family`, `"${familyRaw}" is outside the family enum`);
	}
	const family = familyRaw as PaperTable6Family;
	if (PAPER_TABLE6_FAMILY_IS_OURS[family] !== (origin === 'ours')) {
		fail(
			`${context}.family`,
			`"${family}" is a ${PAPER_TABLE6_FAMILY_IS_OURS[family] ? 'newly scored' : 'published'} family, but the row's origin is "${origin}"`
		);
	}

	const foldCount = positiveInteger(row.fold_count, `${context}.fold_count`);
	if (foldCount !== nFolds) fail(`${context}.fold_count`, `must equal n_folds (${nFolds})`);

	const hasScores = boolean(row.has_out_of_fold_scores, `${context}.has_out_of_fold_scores`);
	const ap = nullableUnit(row.pooled_average_precision, `${context}.pooled_average_precision`);
	const tieGift = row.tie_gift === null ? null : number(row.tie_gift, `${context}.tie_gift`);
	const distinctScores = nullablePositiveInteger(
		row.distinct_scores,
		`${context}.distinct_scores`
	);
	// A score vector has one entry per statement on this panel, so it cannot carry
	// more distinct values than there are statements. This count is the x-axis of
	// the whole tie argument — it is what "coarse" and "fine" mean here — so it is
	// bounded by the panel rather than left as any positive integer.
	if (distinctScores !== null && distinctScores > nStatements) {
		fail(
			`${context}.distinct_scores`,
			`${distinctScores} exceeds the ${nStatements} statements this table scores`
		);
	}
	const apRank = nullablePositiveInteger(row.pooled_ap_rank, `${context}.pooled_ap_rank`);

	// A tie-robust number exists exactly when a score vector does. Anything else
	// means an AP has been attached to a row that cannot have one.
	if (hasScores !== (ap !== null)) {
		fail(context, 'pooled_average_precision must be present exactly when scores are');
	}
	if ((ap !== null) !== (tieGift !== null)) {
		fail(context, 'tie_gift must be present exactly when pooled_average_precision is');
	}
	if ((ap !== null) !== (distinctScores !== null)) {
		fail(context, 'distinct_scores must be present exactly when pooled_average_precision is');
	}
	if ((ap !== null) !== (apRank !== null)) {
		fail(context, 'pooled_ap_rank must be present exactly when pooled_average_precision is');
	}

	const foldMean = unit(
		row.fold_mean_trapezoidal_pr_auc,
		`${context}.fold_mean_trapezoidal_pr_auc`
	);
	// The gift is READ, not derived — but it is a claim about two other shipped
	// numbers, so it is checked against them and gates if it has drifted.
	if (ap !== null && tieGift !== null && !agrees(tieGift, foldMean - ap)) {
		fail(`${context}.tie_gift`, 'must equal fold_mean_trapezoidal_pr_auc − pooled_average_precision');
	}

	const publishedMean = nullableUnit(row.published_mean, `${context}.published_mean`);
	const publishedSd = nullableUnit(
		row.published_fold_population_sd,
		`${context}.published_fold_population_sd`
	);
	const absDev =
		row.abs_dev_vs_published === null
			? null
			: unit(row.abs_dev_vs_published, `${context}.abs_dev_vs_published`);
	const foldSd = unit(row.fold_population_sd, `${context}.fold_population_sd`);

	if (origin === 'paper_published_only') {
		// The node's hardest guard: a row we never re-ran must not carry a
		// tie-robust number, because there is no score vector one could come from.
		exactlyNull(row.pooled_average_precision, `${context}.pooled_average_precision`);
		exactlyNull(row.tie_gift, `${context}.tie_gift`);
		exactlyNull(row.folds, `${context}.folds`);
		exactly(row.has_out_of_fold_scores, false, `${context}.has_out_of_fold_scores`);
		exactlyNull(row.abs_dev_vs_published, `${context}.abs_dev_vs_published`);
		if (publishedMean === null) fail(context, 'a published-only row must carry its printed value');
		if (publishedMean !== foldMean) {
			fail(context, 'a published-only row is its printed value; the two must be identical');
		}
		// Both halves of the readout are the paper's printed figure on these rows,
		// so the SD gets the same identity check as the mean. Without it the shipped
		// `published_fold_population_sd` is never read and could drift from the ±
		// this figure draws beside it.
		if (publishedSd === null) fail(context, 'a published-only row must carry its printed SD');
		if (publishedSd !== foldSd) {
			fail(context, 'a published-only row is its printed SD; the two must be identical');
		}
	} else if (origin === 'paper_rerun') {
		if (publishedMean === null || absDev === null) {
			fail(context, 're-run rows must carry the published value they are checked against');
		}
		if (!agrees(absDev, Math.abs(foldMean - publishedMean))) {
			fail(`${context}.abs_dev_vs_published`, 'must equal |fold mean − published mean|');
		}
		// The ± drawn on a re-run row is OUR SD, not theirs, so the two are checked
		// for agreement rather than identity — see PAPER_TABLE6_PUBLISHED_SD_BOUND
		// for where that bound comes from.
		if (publishedSd === null) {
			fail(context, 're-run rows must carry the published SD they are checked against');
		}
		if (Math.abs(foldSd - publishedSd) > PAPER_TABLE6_PUBLISHED_SD_BOUND) {
			fail(
				`${context}.published_fold_population_sd`,
				`${publishedSd} is more than ${PAPER_TABLE6_PUBLISHED_SD_BOUND} from the re-run SD ${foldSd}`
			);
		}
	} else {
		if (publishedMean !== null || absDev !== null || publishedSd !== null) {
			fail(context, 'a newly scored row was never published, so it cannot carry a published value');
		}
		if (!(label in PAPER_TABLE6_OUR_ARM_ID_BY_LABEL)) {
			fail(`${context}.label`, `"${label}" is not a pinned newly scored model — re-pin it deliberately`);
		}
	}

	return {
		label,
		display,
		origin,
		family,
		rank: positiveInteger(row.rank, `${context}.rank`),
		foldMean,
		foldSd,
		foldCount,
		publishedMean,
		publishedSd,
		absDevVsPublished: absDev,
		ap,
		apRank,
		tieGift,
		distinctScores,
		hasScores,
		isReference: boolean(row.is_reference_arm, `${context}.is_reference_arm`)
	};
}

/**
 * Parse one tie-gift group AND verify it against the rows it names.
 *
 * These two groups carry the headline numbers of the credit disclosure — the
 * +0.0097..+0.0143 handed to the reading models against the −0.0008..+0.0006
 * handed to the models published in 2023 — and the component prints both in a
 * sentence beside a table that contains every row's own credit. A group checked
 * only for `min <= max` is internally consistent with any pair of numbers, so an
 * inflated range would render as a disclosure the table below contradicted.
 * Every field is therefore differenced against the rows:
 *
 *   · every label must name a row that HAS a gift (no score vector, no gift);
 *   · the set of labels must be EXACTLY the class the group claims to summarise,
 *     so the range cannot be narrowed by dropping the row that widens it;
 *   · min/max must be the extremes of those rows' gifts, and the distinct-score
 *     bounds the extremes of their score-vector widths.
 */
function parseTieGroup(
	raw: unknown,
	context: string,
	byLabel: Map<string, ParsedRow>,
	expected: ParsedRow[],
	expectedWhat: string
): PaperTable6TieGroup {
	const group = record(raw, context);
	const labelsRaw = group.labels;
	if (!Array.isArray(labelsRaw) || labelsRaw.length === 0) {
		fail(`${context}.labels`, 'expected a non-empty array');
	}
	const labels = labelsRaw.map((value, index) => text(value, `${context}.labels[${index}]`));
	if (new Set(labels).size !== labels.length) {
		fail(`${context}.labels`, 'names the same row more than once');
	}

	const gifts: number[] = [];
	const widths: number[] = [];
	for (const label of labels) {
		const row = byLabel.get(label);
		if (!row) fail(`${context}.labels`, `"${label}" names no row in this table`);
		if (row.tieGift === null || row.distinctScores === null) {
			fail(`${context}.labels`, `"${label}" has no score vector, so it has no gift to summarise`);
		}
		gifts.push(row.tieGift);
		widths.push(row.distinctScores);
	}

	// Completeness, both ways: the group summarises this class and nothing else.
	const expectedLabels = new Set(expected.map((row) => row.label));
	for (const label of labels) {
		if (!expectedLabels.has(label)) {
			fail(`${context}.labels`, `"${label}" is not one of ${expectedWhat}`);
		}
	}
	if (labels.length !== expectedLabels.size) {
		fail(
			`${context}.labels`,
			`names ${labels.length} rows; ${expectedLabels.size} rows are ${expectedWhat}`
		);
	}

	const min = number(group.min, `${context}.min`);
	const max = number(group.max, `${context}.max`);
	if (min > max) fail(context, 'min must not exceed max');
	if (!agrees(min, Math.min(...gifts))) {
		fail(`${context}.min`, 'is not the smallest tie gift among the rows it names');
	}
	if (!agrees(max, Math.max(...gifts))) {
		fail(`${context}.max`, 'is not the largest tie gift among the rows it names');
	}
	const distinctMin = positiveInteger(group.distinct_scores_min, `${context}.distinct_scores_min`);
	const distinctMax = positiveInteger(group.distinct_scores_max, `${context}.distinct_scores_max`);
	if (distinctMin > distinctMax) fail(context, 'distinct_scores_min must not exceed its max');
	if (distinctMin !== Math.min(...widths)) {
		fail(`${context}.distinct_scores_min`, 'is not the narrowest score vector among those rows');
	}
	if (distinctMax !== Math.max(...widths)) {
		fail(`${context}.distinct_scores_max`, 'is not the widest score vector among those rows');
	}
	return {
		min,
		max,
		count: labels.length,
		distinctScoresMin: distinctMin,
		distinctScoresMax: distinctMax
	};
}

/** Ticks every 0.02 across the domain, in integer hundredths so they never drift. */
function ticksOf(min: number, max: number): number[] {
	const out: number[] = [];
	for (let h = Math.ceil(Math.round(min * 100) / 2) * 2; h <= Math.round(max * 100); h += 2) {
		out.push(h / 100);
	}
	if (out.length < 2) fail('domain', 'degenerate axis range');
	return out;
}

/**
 * Build the drawable figure from a parsed artifact. Throws on any drift — a rank
 * collision, a gap in the rank sequence, an origin outside the enum, a
 * published-only row carrying an average precision, a replication deviation past
 * the bound, an over-budget label — so the caller gates to `unavailable` rather
 * than drawing a ranked list whose ordering or geometry is silently wrong.
 */
function buildFigure(raw: UnknownRecord): PaperTable6Figure {
	const kind = text(raw.artifact_kind, 'artifact_kind');
	if (kind !== PAPER_TABLE6_EXTENDED_ARTIFACT_KIND) {
		fail('artifact_kind', `expected ${PAPER_TABLE6_EXTENDED_ARTIFACT_KIND}, got ${kind}`);
	}
	const schema = positiveInteger(raw.schema_version, 'schema_version');
	if (schema !== PAPER_TABLE6_EXTENDED_SCHEMA_VERSION) {
		fail('schema_version', `expected ${PAPER_TABLE6_EXTENDED_SCHEMA_VERSION}, got ${schema}`);
	}

	const nStatements = positiveInteger(raw.n_statements, 'n_statements');
	const nPositive = positiveInteger(raw.n_positive, 'n_positive');
	const nNegative = positiveInteger(raw.n_negative, 'n_negative');
	if (nPositive + nNegative !== nStatements) {
		fail('n_statements', 'n_positive + n_negative must equal n_statements');
	}
	const nFolds = positiveInteger(raw.n_folds, 'n_folds');

	const contractRaw = record(raw.metric_contract, 'metric_contract');
	const metric: PaperTable6MetricContract = {
		perFoldMetric: text(contractRaw.per_fold_metric, 'metric_contract.per_fold_metric'),
		summary: text(contractRaw.summary, 'metric_contract.summary'),
		uncertaintyField: text(contractRaw.uncertainty_field, 'metric_contract.uncertainty_field'),
		uncertaintyFieldProse: {
			shipped: text(contractRaw.uncertainty_field, 'metric_contract.uncertainty_field'),
			plain: TABLE6_PLAIN.uncertaintyField
		},
		uncertaintyNote: text(contractRaw.uncertainty_note, 'metric_contract.uncertainty_note'),
		trapezoidalNote: text(contractRaw.trapezoidal_note, 'metric_contract.trapezoidal_note'),
		estimatorSource: text(contractRaw.estimator_source, 'metric_contract.estimator_source'),
		// All three are load-bearing framing, so they are GATED rather than shown:
		// if the artifact ever stops calling this ± a dispersion statistic, the
		// figure disappears instead of printing something a reader reads as a test.
		uncertaintyIsConfidenceInterval: exactly(
			contractRaw.uncertainty_is_confidence_interval,
			false,
			'metric_contract.uncertainty_is_confidence_interval'
		) as false,
		uncertaintyIsDispersion: exactly(
			contractRaw.uncertainty_is_dispersion_not_a_confidence_interval,
			true,
			'metric_contract.uncertainty_is_dispersion_not_a_confidence_interval'
		) as true,
		metricIsPooledAveragePrecision: exactly(
			contractRaw.metric_is_pooled_average_precision,
			false,
			'metric_contract.metric_is_pooled_average_precision'
		) as false
	};

	const originsRaw = record(raw.origins, 'origins');
	const origins = {
		ours: text(originsRaw.ours, 'origins.ours'),
		paper_rerun: text(originsRaw.paper_rerun, 'origins.paper_rerun'),
		paper_published_only: text(originsRaw.paper_published_only, 'origins.paper_published_only')
	} satisfies Record<PaperTable6Origin, string>;

	const rowsRaw = raw.rows;
	if (!Array.isArray(rowsRaw) || rowsRaw.length === 0) fail('rows', 'expected a non-empty array');
	const nRows = positiveInteger(raw.n_rows, 'n_rows');
	if (rowsRaw.length !== nRows) fail('n_rows', `says ${nRows}; the file carries ${rowsRaw.length}`);

	const parsed = rowsRaw.map((entry, index) => parseRow(entry, index, nFolds, nStatements));

	// Ranks: unique, and a contiguous 1..N once sorted. A collision or a gap means
	// the ordering this whole figure asserts is not a total order.
	const seenRanks = new Set<number>();
	for (const row of parsed) {
		if (seenRanks.has(row.rank)) fail('rows', `rank ${row.rank} is used more than once`);
		seenRanks.add(row.rank);
	}
	const ordered = parsed.slice().sort((a, b) => a.rank - b.rank);
	ordered.forEach((row, index) => {
		if (row.rank !== index + 1) {
			fail('rows', `ranks must run 1..${nRows} without a gap; found ${row.rank} at position ${index + 1}`);
		}
	});

	// …and the order must be the order the artifact says it is: the paper metric
	// descending, ties broken by the tighter fold SD.
	for (let i = 1; i < ordered.length; i += 1) {
		const above = ordered[i - 1];
		const below = ordered[i];
		if (above.foldMean < below.foldMean) {
			fail('rows', `rank ${above.rank} scores below rank ${below.rank} on the ranking metric`);
		}
		if (above.foldMean === below.foldMean && above.foldSd > below.foldSd) {
			fail('rows', `rank ${above.rank} ties rank ${below.rank} but carries the wider fold SD`);
		}
	}

	// The tie-robust ranks are a second total order over the rows that have a
	// score vector, so they get the same treatment.
	const scored = ordered.filter((row) => row.ap !== null);
	const apOrdered = scored
		.slice()
		.sort((a, b) => (a.apRank as number) - (b.apRank as number));
	apOrdered.forEach((row, index) => {
		if (row.apRank !== index + 1) {
			fail('rows', `pooled_ap_rank must run 1..${scored.length} without a gap or collision`);
		}
		if (index > 0 && (apOrdered[index - 1].ap as number) < (row.ap as number)) {
			fail('rows', 'pooled_ap_rank must order the rows by stepped average precision');
		}
	});

	const byLabel = new Map(ordered.map((row) => [row.label, row]));
	const displayOf = (label: string, context: string): string => {
		const row = byLabel.get(label);
		if (!row) fail(context, `"${label}" names no row in this table`);
		return row.display;
	};

	const reproRaw = record(raw.reproduction_fidelity, 'reproduction_fidelity');
	const tolerance = number(reproRaw.tolerance, 'reproduction_fidelity.tolerance');
	// The artifact's own bound is checked against ours rather than trusted: a
	// relaxed tolerance would otherwise widen the licence for putting our rows in
	// their table without anyone noticing.
	if (!(tolerance > 0 && tolerance <= PAPER_TABLE6_AGREEMENT_BOUND)) {
		fail(
			'reproduction_fidelity.tolerance',
			`${tolerance} is outside (0, ${PAPER_TABLE6_AGREEMENT_BOUND}]`
		);
	}
	const rerunRows = ordered.filter((row) => row.origin === 'paper_rerun');
	if (rerunRows.length === 0) {
		fail('reproduction_fidelity', 'no re-run row is present, so nothing licenses this table');
	}
	const shippedMaxDev = number(
		reproRaw.max_abs_dev_vs_published,
		'reproduction_fidelity.max_abs_dev_vs_published'
	);
	let worst = rerunRows[0];
	for (const row of rerunRows) {
		const dev = row.absDevVsPublished as number;
		if (dev > tolerance) {
			fail(
				'reproduction_fidelity',
				`${row.label} deviates ${dev} from its published value, past the ${tolerance} bound`
			);
		}
		if (dev > (worst.absDevVsPublished as number)) worst = row;
	}
	if (!agrees(shippedMaxDev, worst.absDevVsPublished as number)) {
		fail('reproduction_fidelity.max_abs_dev_vs_published', 'disagrees with the drawn rows');
	}
	const maxDevLabel = text(reproRaw.max_abs_dev_row, 'reproduction_fidelity.max_abs_dev_row');
	if (maxDevLabel !== worst.label) {
		fail('reproduction_fidelity.max_abs_dev_row', `names ${maxDevLabel}, but ${worst.label} is worse`);
	}
	const reproduction: PaperTable6Reproduction = {
		nRerunRows: positiveInteger(reproRaw.n_rerun_rows, 'reproduction_fidelity.n_rerun_rows'),
		nRerunRowsWithScores: positiveInteger(
			reproRaw.n_rerun_rows_with_out_of_fold_scores,
			'reproduction_fidelity.n_rerun_rows_with_out_of_fold_scores'
		),
		maxAbsDev: shippedMaxDev,
		maxAbsDevDisplay: worst.display,
		tolerance,
		publishedValuesRoundedTo3dp: exactly(
			reproRaw.published_values_are_rounded_to_3dp,
			true,
			'reproduction_fidelity.published_values_are_rounded_to_3dp'
		) as true,
		note: text(reproRaw.note, 'reproduction_fidelity.note')
	};
	if (reproduction.nRerunRows !== rerunRows.length) {
		fail('reproduction_fidelity.n_rerun_rows', 'disagrees with the drawn rows');
	}
	if (reproduction.nRerunRowsWithScores !== rerunRows.filter((row) => row.hasScores).length) {
		fail('reproduction_fidelity.n_rerun_rows_with_out_of_fold_scores', 'disagrees with the drawn rows');
	}

	const tieRaw = record(raw.tie_disclosure, 'tie_disclosure');
	// Each group is bound to the rows it summarises before its numbers are read.
	// The reading models are the rows the artifact tags as such; the other group is
	// every re-run row that has a score vector, since a row without one has no gift.
	const readers = parseTieGroup(
		tieRaw.llm_reader_arms,
		'tie_disclosure.llm_reader_arms',
		byLabel,
		ordered.filter((row) => row.family === PAPER_TABLE6_LLM_READER_FAMILY),
		'the reading models in this table'
	);
	const paperRerunGroup = parseTieGroup(
		tieRaw.paper_rerun_rows,
		'tie_disclosure.paper_rerun_rows',
		byLabel,
		rerunRows.filter((row) => row.ap !== null),
		'the published rows re-run here with a score vector'
	);
	const controlRaw = record(tieRaw.indra_cogex_hybrid, 'tie_disclosure.indra_cogex_hybrid');
	const controlLabel = text(controlRaw.label, 'tie_disclosure.indra_cogex_hybrid.label');
	const controlRow = byLabel.get(controlLabel);
	if (!controlRow || controlRow.tieGift === null || controlRow.distinctScores === null) {
		fail('tie_disclosure.indra_cogex_hybrid', 'the control row is missing or carries no gift');
	}
	if (controlRow.origin !== 'ours') {
		// The control's whole force is that it is NEWLY SCORED and still collects
		// nothing, which is what separates tie density from authorship.
		fail('tie_disclosure.indra_cogex_hybrid', 'the control must be one of the newly scored rows');
	}
	if (!agrees(number(controlRaw.tie_gift, 'tie_disclosure.indra_cogex_hybrid.tie_gift'), controlRow.tieGift)) {
		fail('tie_disclosure.indra_cogex_hybrid.tie_gift', 'disagrees with the row it names');
	}

	// Every row with a gift, published or newly scored: a gift exists exactly where a score
	// vector does, so this is the population the coarse/fine split partitions and
	// the population the gift-vs-tie-density correlation runs over.
	const giftRows = ordered.filter((row) => row.tieGift !== null && row.distinctScores !== null);
	if (giftRows.length < 3) {
		fail('tie_disclosure', 'fewer than three rows carry a gift, so no tie-density claim is testable');
	}

	// SEPARATION — every field re-derived from those rows. Previously only the
	// no-overlap inequality was checked, which is satisfiable by any two numbers:
	// the counts and the cut were rendered on trust, so a group could be redrawn
	// around whichever rows made the claim look cleanest and the gate would agree
	// with itself. The split is now applied to the rows and the shipped scalars are
	// differenced against the result.
	const separationRaw = record(tieRaw.separation, 'tie_disclosure.separation');
	const separationContext = 'tie_disclosure.separation';
	const coarseMaxDistinct = positiveInteger(
		separationRaw.coarse_max_distinct_scores,
		`${separationContext}.coarse_max_distinct_scores`
	);
	const coarseRows = giftRows.filter((row) => (row.distinctScores as number) <= coarseMaxDistinct);
	const fineRows = giftRows.filter((row) => (row.distinctScores as number) > coarseMaxDistinct);
	if (coarseRows.length === 0 || fineRows.length === 0) {
		fail(`${separationContext}.coarse_max_distinct_scores`, 'puts every scored row on one side');
	}
	// The cut must be ATTAINED, not merely somewhere in the gap: an unattained
	// threshold is a free parameter, and sliding it moves rows between the two
	// groups without contradicting anything the artifact says about itself.
	if (Math.max(...coarseRows.map((row) => row.distinctScores as number)) !== coarseMaxDistinct) {
		fail(
			`${separationContext}.coarse_max_distinct_scores`,
			'is not the widest score vector among the rows it puts on the coarse side'
		);
	}
	const nCoarse = positiveInteger(separationRaw.n_coarse, `${separationContext}.n_coarse`);
	if (nCoarse !== coarseRows.length) {
		fail(`${separationContext}.n_coarse`, `says ${nCoarse}; ${coarseRows.length} drawn rows are coarse`);
	}
	const nFine = positiveInteger(separationRaw.n_fine, `${separationContext}.n_fine`);
	if (nFine !== fineRows.length) {
		fail(`${separationContext}.n_fine`, `says ${nFine}; ${fineRows.length} drawn rows are fine`);
	}
	const minCoarse = number(
		separationRaw.min_gift_among_coarse_scored_arms,
		`${separationContext}.min_gift_among_coarse_scored_arms`
	);
	if (!agrees(minCoarse, Math.min(...coarseRows.map((row) => row.tieGift as number)))) {
		fail(
			`${separationContext}.min_gift_among_coarse_scored_arms`,
			'is not the smallest gift among the coarse-scored rows this table draws'
		);
	}
	const maxFine = number(
		separationRaw.max_abs_gift_among_fine_scored_arms,
		`${separationContext}.max_abs_gift_among_fine_scored_arms`
	);
	if (!agrees(maxFine, Math.max(...fineRows.map((row) => Math.abs(row.tieGift as number))))) {
		fail(
			`${separationContext}.max_abs_gift_among_fine_scored_arms`,
			'is not the largest absolute gift among the fine-scored rows this table draws'
		);
	}
	// The claim the figure makes out loud is that the two groups do not overlap.
	// If they ever do, the claim is false and the figure gates rather than print it.
	if (!(minCoarse > maxFine)) {
		fail(separationContext, 'the coarse and fine gift ranges now overlap');
	}
	const separation: PaperTable6TieSeparation = {
		coarseMaxDistinctScores: coarseMaxDistinct,
		minGiftAmongCoarse: minCoarse,
		maxAbsGiftAmongFine: maxFine,
		nCoarse,
		nFine,
		note: text(separationRaw.note, `${separationContext}.note`)
	};

	const bestRaw = record(tieRaw.tie_corrected_best_vs_best, 'tie_disclosure.tie_corrected_best_vs_best');
	const bestContext = 'tie_disclosure.tie_corrected_best_vs_best';
	const ourApLabel = text(bestRaw.our_best_label, `${bestContext}.our_best_label`);
	const theirApLabel = text(bestRaw.their_best_label, `${bestContext}.their_best_label`);
	const ourMetricLabel = text(
		bestRaw.our_best_paper_metric_label,
		`${bestContext}.our_best_paper_metric_label`
	);
	const theirMetricLabel = text(
		bestRaw.their_best_paper_metric_label,
		`${bestContext}.their_best_paper_metric_label`
	);
	const referenceLabel = text(bestRaw.reference_label, `${bestContext}.reference_label`);

	// Each "best" is an argmax over rows already parsed, so it is checked against
	// them. A stale label here would put the wrong arm in the figure's headline.
	const argmax = (
		rows: ParsedRow[],
		pick: (row: ParsedRow) => number | null,
		context: string
	): ParsedRow => {
		let best: ParsedRow | null = null;
		for (const row of rows) {
			const value = pick(row);
			if (value === null) continue;
			const bestValue = best === null ? null : pick(best);
			if (bestValue === null || value > bestValue) best = row;
		}
		if (!best) fail(context, 'no row carries the metric this claim ranks on');
		return best;
	};
	const oursRows = ordered.filter((row) => row.origin === 'ours');
	const theirRows = ordered.filter((row) => row.origin !== 'ours');
	const checkedBest = (
		claimed: string,
		rows: ParsedRow[],
		pick: (row: ParsedRow) => number | null,
		context: string
	): ParsedRow => {
		const actual = argmax(rows, pick, context);
		if (actual.label !== claimed) {
			fail(context, `names ${claimed}, but ${actual.label} leads on that metric`);
		}
		return actual;
	};
	const ourApRow = checkedBest(ourApLabel, oursRows, (row) => row.ap, `${bestContext}.our_best_label`);
	const theirApRow = checkedBest(
		theirApLabel,
		theirRows,
		(row) => row.ap,
		`${bestContext}.their_best_label`
	);
	const ourMetricRow = checkedBest(
		ourMetricLabel,
		oursRows,
		(row) => row.foldMean,
		`${bestContext}.our_best_paper_metric_label`
	);
	const theirMetricRow = checkedBest(
		theirMetricLabel,
		theirRows,
		(row) => row.foldMean,
		`${bestContext}.their_best_paper_metric_label`
	);
	const apMargin = number(bestRaw.margin, `${bestContext}.margin`);
	const paperMetricMargin = number(bestRaw.paper_metric_margin, `${bestContext}.paper_metric_margin`);
	if (!agrees(apMargin, (ourApRow.ap as number) - (theirApRow.ap as number))) {
		fail(`${bestContext}.margin`, 'disagrees with the two rows it names');
	}
	if (!agrees(paperMetricMargin, ourMetricRow.foldMean - theirMetricRow.foldMean)) {
		fail(`${bestContext}.paper_metric_margin`, 'disagrees with the two rows it names');
	}
	// The head-to-head figure this sentence reconciles against. Read, not
	// hard-coded, and GATED: the shipped sentence claims the two land within
	// 0.001 of each other, so an artifact where they no longer do gates the
	// figure rather than printing a reconciliation that does not reconcile.
	const headToHeadApDelta = number(
		bestRaw.head_to_head_ap_delta_gemma_4_26b_vs_reference,
		`${bestContext}.head_to_head_ap_delta_gemma_4_26b_vs_reference`
	);
	if (Math.abs(apMargin - headToHeadApDelta) > PAPER_TABLE6_TIE_BEST_HEAD_TO_HEAD_TOL) {
		fail(
			`${bestContext}.head_to_head_ap_delta_gemma_4_26b_vs_reference`,
			`is ${headToHeadApDelta} against a tie-corrected margin of ${apMargin}; the sentence ` +
				`claims they agree to ${PAPER_TABLE6_TIE_BEST_HEAD_TO_HEAD_TOL}`
		);
	}

	// The correlation is the figure's general claim — "coarse scores collect the
	// gift, whoever wrote them" — and the two columns it runs over are both drawn
	// in the table below it. Re-derived over exactly those rows, so a shipped ρ
	// that no longer describes them takes the figure down.
	const spearman = number(
		tieRaw.spearman_gift_vs_distinct_scores,
		'tie_disclosure.spearman_gift_vs_distinct_scores'
	);
	const spearmanFromRows = spearmanMidRank(
		giftRows.map((row) => row.tieGift as number),
		giftRows.map((row) => row.distinctScores as number)
	);
	if (spearmanFromRows === null) {
		fail(
			'tie_disclosure.spearman_gift_vs_distinct_scores',
			'the drawn gift and distinct-score columns admit no rank correlation'
		);
	}
	if (!agrees(spearman, spearmanFromRows)) {
		fail(
			'tie_disclosure.spearman_gift_vs_distinct_scores',
			`is not the mid-rank Spearman of the ${giftRows.length} drawn rows' gift against their distinct-score count`
		);
	}

	const tie: PaperTable6Tie = {
		definition: text(tieRaw.definition, 'tie_disclosure.definition'),
		readers,
		paperRerun: paperRerunGroup,
		control: {
			display: controlRow.display,
			tieGift: controlRow.tieGift,
			distinctScores: controlRow.distinctScores
		},
		separation,
		best: (() => {
			const bestVsBest = {
				paperMetricStanding: marginStanding(paperMetricMargin),
				apStanding: marginStanding(apMargin),
				ourPaperMetricDisplay: ourMetricRow.display,
				theirPaperMetricDisplay: theirMetricRow.display,
				paperMetricMargin,
				ourApDisplay: ourApRow.display,
				theirApDisplay: theirApRow.display,
				ourAp: ourApRow.ap as number,
				theirAp: theirApRow.ap as number,
				apMargin,
				referenceDisplay: displayOf(referenceLabel, `${bestContext}.reference_label`),
				headToHeadApDelta,
				noteWithJoinKeys: text(bestRaw.note, `${bestContext}.note`)
			};
			return {
				...bestVsBest,
				// Audit-only above (it names frozen join keys, which is why it is not in
				// `renderedProse`); this is the half that may be read. Built from the
				// resolved displays and the parsed margins so both models are NAMED —
				// they are different models, which is the whole point of the sentence.
				noteProse: {
					shipped: bestVsBest.noteWithJoinKeys,
					plain: table6TieBestPlain({
						...bestVsBest,
						headToHeadTolerance: PAPER_TABLE6_TIE_BEST_HEAD_TO_HEAD_TOL
					})
				}
			};
		})(),
		reconciliation: text(tieRaw.reconciliation, 'tie_disclosure.reconciliation'),
		spearman,
		spearmanMethod: text(tieRaw.spearman_method, 'tie_disclosure.spearman_method')
	};
	exactly(
		tieRaw.tracks_tie_density_not_authorship,
		true,
		'tie_disclosure.tracks_tie_density_not_authorship'
	);

	const conventionRaw = record(raw.paper_reporting_convention, 'paper_reporting_convention');
	const statistics = conventionRaw.reported_statistics;
	if (!Array.isArray(statistics) || statistics.length === 0) {
		fail('paper_reporting_convention.reported_statistics', 'expected a non-empty array');
	}
	const sourceRaw = record(conventionRaw.pinned_source, 'paper_reporting_convention.pinned_source');
	const headlineRaw = record(
		conventionRaw.their_own_headline_gain,
		'paper_reporting_convention.their_own_headline_gain'
	);
	const headlineContext = 'paper_reporting_convention.their_own_headline_gain';
	const bestFittedMean = unit(headlineRaw.best_fitted_mean, `${headlineContext}.best_fitted_mean`);
	const bestFittedSd = unit(
		headlineRaw.best_fitted_fold_population_sd,
		`${headlineContext}.best_fitted_fold_population_sd`
	);
	const beliefOrigMean = unit(headlineRaw.belief_orig_mean, `${headlineContext}.belief_orig_mean`);
	const beliefOrigSd = unit(
		headlineRaw.belief_orig_fold_population_sd,
		`${headlineContext}.belief_orig_fold_population_sd`
	);
	const gain = number(headlineRaw.gain, `${headlineContext}.gain`);

	// BOTH ENDS OF THIS GAIN NAME ROWS IN THIS TABLE, so both are re-derived from
	// them. Differencing the gain against its own two means only ever proved the
	// subtraction; the means themselves were rendered on trust, and this sentence
	// is the SCALE the component tells a reader to judge every row above against.
	// The values are the paper's PRINTED three-decimal figures, which is what the
	// published headline was stated in — not our re-run means.
	const fittedRows = ordered.filter(
		(row) => row.family === PAPER_TABLE6_FITTED_FAMILY && row.publishedMean !== null
	);
	if (fittedRows.length === 0) {
		fail(headlineContext, 'this table carries no published fitted model to top the gain');
	}
	// The published ranking rule, the one this file already sorts by: best published
	// mean, ties broken by the tighter published SD. Two published rows DO tie at
	// the top here, which is why the tie-break is stated rather than assumed away.
	let bestFittedRow = fittedRows[0];
	for (const row of fittedRows) {
		const mean = row.publishedMean as number;
		const best = bestFittedRow.publishedMean as number;
		if (mean > best) {
			bestFittedRow = row;
		} else if (mean === best && row.publishedSd !== null && bestFittedRow.publishedSd !== null) {
			if (row.publishedSd < bestFittedRow.publishedSd) bestFittedRow = row;
		}
	}
	if (!agrees(bestFittedMean, bestFittedRow.publishedMean as number)) {
		fail(
			`${headlineContext}.best_fitted_mean`,
			'is not the best published mean among the fitted rows in this table'
		);
	}
	if (bestFittedRow.publishedSd === null || !agrees(bestFittedSd, bestFittedRow.publishedSd)) {
		fail(
			`${headlineContext}.best_fitted_fold_population_sd`,
			'is not the SD the paper printed beside that same row'
		);
	}
	const unfittedRows = ordered.filter((row) => row.family === PAPER_TABLE6_UNFITTED_FAMILY);
	if (unfittedRows.length !== 1) {
		fail(
			headlineContext,
			`the gain is stated over one unfitted baseline; this table carries ${unfittedRows.length}`
		);
	}
	const beliefRow = unfittedRows[0];
	if (beliefRow.publishedMean === null || !agrees(beliefOrigMean, beliefRow.publishedMean)) {
		fail(
			`${headlineContext}.belief_orig_mean`,
			'is not the published mean of the unfitted baseline row this table draws'
		);
	}
	if (beliefRow.publishedSd === null || !agrees(beliefOrigSd, beliefRow.publishedSd)) {
		fail(
			`${headlineContext}.belief_orig_fold_population_sd`,
			'is not the SD the paper printed beside that same row'
		);
	}
	// Their own published improvement is the SCALE every row here is read against,
	// so it is checked against the two values it differences.
	if (!agrees(gain, bestFittedMean - beliefOrigMean)) {
		fail(`${headlineContext}.gain`, 'disagrees with the two means it differences');
	}
	// …and the fold-SD restatement of it is checked against the SD it divides by,
	// which is now itself bound to a drawn row. Rendered to two decimals, so an
	// unbound value could have moved the sentence by a whole SD unnoticed.
	const gainInFoldSd = number(headlineRaw.gain_in_fold_sd, `${headlineContext}.gain_in_fold_sd`);
	if (!(bestFittedSd > 0)) {
		fail(`${headlineContext}.best_fitted_fold_population_sd`, 'is zero, so the gain has no SD scale');
	}
	if (!agrees(gainInFoldSd, gain / bestFittedSd)) {
		fail(
			`${headlineContext}.gain_in_fold_sd`,
			'is not the gain divided by the best fitted row’s published fold SD'
		);
	}
	const convention: PaperTable6Convention = {
		nPublishedRows: positiveInteger(
			conventionRaw.n_published_rows,
			'paper_reporting_convention.n_published_rows'
		),
		reportedStatistics: statistics.map((value, index) =>
			text(value, `paper_reporting_convention.reported_statistics[${index}]`)
		),
		nPValues: nonNegativeInteger(conventionRaw.n_p_values, 'paper_reporting_convention.n_p_values'),
		nConfidenceIntervals: nonNegativeInteger(
			conventionRaw.n_confidence_intervals,
			'paper_reporting_convention.n_confidence_intervals'
		),
		nMultiplicityCorrections: nonNegativeInteger(
			conventionRaw.n_multiplicity_corrections,
			'paper_reporting_convention.n_multiplicity_corrections'
		),
		verified: text(conventionRaw.verified, 'paper_reporting_convention.verified'),
		repository: text(sourceRaw.repository, 'paper_reporting_convention.pinned_source.repository'),
		commit: text(sourceRaw.commit, 'paper_reporting_convention.pinned_source.commit'),
		notebookPath: text(
			sourceRaw.notebook_path,
			'paper_reporting_convention.pinned_source.notebook_path'
		),
		notebookSha256: text(
			sourceRaw.notebook_sha256,
			'paper_reporting_convention.pinned_source.notebook_sha256'
		),
		headline: {
			bestFittedMean,
			bestFittedSd,
			beliefOrigMean,
			beliefOrigSd,
			gain,
			gainInFoldSd,
			tieBreak: text(headlineRaw.tie_break, `${headlineContext}.tie_break`),
			note: text(headlineRaw.note, `${headlineContext}.note`)
		}
	};
	const paperReportsNoTests = exactly(
		raw.paper_reports_no_tests,
		true,
		'paper_reports_no_tests'
	) as true;
	// "The paper reports no tests" is a sentence this figure prints. It is only
	// true if the census behind it is zero on every count, so it is checked.
	if (
		convention.nPValues !== 0 ||
		convention.nConfidenceIntervals !== 0 ||
		convention.nMultiplicityCorrections !== 0
	) {
		fail('paper_reporting_convention', 'the no-tests claim disagrees with its own census');
	}

	// The artifact pins the anchors this figure is about. Checking them here means
	// a re-generated file that moved a rank takes the figure down rather than
	// quietly re-ordering a list the prose describes.
	const checksRaw = record(raw.checks, 'checks');
	const expected = checksRaw.expected_ranks;
	if (!Array.isArray(expected) || expected.length === 0) {
		fail('checks.expected_ranks', 'expected a non-empty array');
	}
	expected.forEach((entry, index) => {
		const context = `checks.expected_ranks[${index}]`;
		const check = record(entry, context);
		const label = text(check.label, `${context}.label`);
		const row = byLabel.get(label);
		if (!row) fail(context, `"${label}" names no row in this table`);
		const rank = positiveInteger(check.rank, `${context}.rank`);
		if (row.rank !== rank) fail(context, `expects rank ${rank}; the row carries ${row.rank}`);
		const mean4 = number(check.fold_mean_4dp, `${context}.fold_mean_4dp`);
		if (Number(row.foldMean.toFixed(4)) !== mean4) {
			fail(context, `expects ${mean4} at four decimals; the row rounds to ${row.foldMean.toFixed(4)}`);
		}
		if (check.fold_population_sd_4dp !== null) {
			const sd4 = number(check.fold_population_sd_4dp, `${context}.fold_population_sd_4dp`);
			if (Number(row.foldSd.toFixed(4)) !== sd4) {
				fail(context, `expects fold SD ${sd4} at four decimals; the row rounds to ${row.foldSd.toFixed(4)}`);
			}
		}
	});
	if (positiveInteger(checksRaw.n_rows, 'checks.n_rows') !== nRows) {
		fail('checks.n_rows', 'disagrees with n_rows');
	}
	if (
		!agrees(
			number(checksRaw.max_abs_dev_vs_published_tolerance, 'checks.max_abs_dev_vs_published_tolerance'),
			tolerance
		)
	) {
		fail('checks.max_abs_dev_vs_published_tolerance', 'disagrees with reproduction_fidelity.tolerance');
	}
	exactly(
		checksRaw.published_only_rows_carry_no_tie_robust_number,
		true,
		'checks.published_only_rows_carry_no_tie_robust_number'
	);

	const provenanceRaw = record(raw.provenance, 'provenance');
	const generatedBy = text(provenanceRaw.generated_by, 'provenance.generated_by');
	const config = text(raw.config, 'config');
	const whatThisIs = text(raw.what_this_is, 'what_this_is');
	const rankingRule = text(raw.ranking_rule, 'ranking_rule');

	// Pairs the paper's own printed precision cannot separate. Published-only rows
	// exist to us only at three decimals, so any row within half of that last digit
	// is not ordered against them by any evidence we have — and the figure says so
	// rather than presenting the rank as settled.
	//
	// ALL PAIRS, not adjacencies. Three rows inside half a printed digit make three
	// unresolvable pairs; an adjacency walk reports two of them and leaves the
	// outer gap looking settled. It also misses the case that actually occurs here:
	// two of OUR fully-measured rows either side of a printed-only row, where
	// neither adjacency is flagged as ours-vs-theirs but the outer pair is real.
	// Rows are ordered by descending mean, so the inner loop stops as soon as the
	// gap opens past the half-width.
	const ambiguousPairs: PaperTable6AmbiguousPair[] = [];
	const ambiguousLabels = new Set<string>();
	for (let i = 0; i < ordered.length; i += 1) {
		for (let j = i + 1; j < ordered.length; j += 1) {
			const above = ordered[i];
			const below = ordered[j];
			const gap = above.foldMean - below.foldMean;
			if (gap > PAPER_TABLE6_PUBLISHED_ROUNDING_HALF_WIDTH) break;
			const roundedEither =
				above.origin === 'paper_published_only' || below.origin === 'paper_published_only';
			if (!roundedEither) continue;
			ambiguousPairs.push({
				id: `${above.rank}-${below.rank}`,
				higherDisplay: above.display,
				lowerDisplay: below.display,
				gap
			});
			ambiguousLabels.add(above.label);
			ambiguousLabels.add(below.label);
		}
	}

	let leadingOurRanks = 0;
	while (leadingOurRanks < ordered.length && ordered[leadingOurRanks].origin === 'ours') {
		leadingOurRanks += 1;
	}
	const theirBest = ordered.find((row) => row.origin !== 'ours');
	if (!theirBest) fail('rows', 'no published row is left in this table to rank against');

	const geometry = PAPER_TABLE6_GEOMETRY;
	const extents: number[] = [];
	const rows: PaperTable6Row[] = ordered.map((row, index) => {
		const context = `rows[${row.label}]`;
		const y = geometry.topPad + index * geometry.laneHeight + geometry.laneHeight / 2;
		const armId = PAPER_TABLE6_OUR_ARM_ID_BY_LABEL[row.label];
		let headToHeadDisplay: string | null = null;
		if (row.origin === 'ours') {
			const spec = PAPER_LITERAL_ARM_SPECS.find((candidate) => candidate.id === armId);
			if (!spec) {
				fail(`${context}.label`, `pins arm id "${armId}", which is not a paper-literal arm`);
			}
			headToHeadDisplay = spec.display;
		}

		// Mixed precision is deliberate and is the signal: a row that exists only as
		// a figure printed in 2023 is shown at the three decimals it was printed to,
		// and a row measured here carries the fourth. The legend says so.
		const metricReadout = budget(
			row.origin === 'paper_published_only'
				? `${fmt3(row.foldMean)} ±${fmt3(row.foldSd)}`
				: `${fmt4(row.foldMean)} ±${fmt4(row.foldSd)}`,
			PAPER_TABLE6_METRIC_BUDGET_CHARS,
			`${context}.metricReadout`
		);
		// Never a bare dash where a number is missing: a dash beside a column of
		// numbers reads as "zero" or "not applicable", and the true answer is that
		// this row has no score vector for a stated reason.
		const tieReadout = budget(
			row.ap !== null && row.tieGift !== null
				? `${fmt4(row.ap)}  ${fmtSigned4(row.tieGift)}  AP rank ${row.apRank}`
				: row.origin === 'paper_published_only'
					? 'published value only'
					: 'fold areas only',
			PAPER_TABLE6_TIE_BUDGET_CHARS,
			`${context}.tieReadout`
		);

		// The title quotes the row at the precision the row HAS: a published-only
		// row exists here only at the three decimals it was printed to, and printing
		// a fourth would invent one.
		const fmtOwn = row.origin === 'paper_published_only' ? fmt3 : fmt4;
		const dispersion =
			`population SD ${fmtOwn(row.foldSd)} over ${row.foldCount} folds ` +
			`(how far the score moves between folds, not a confidence interval)`;
		let provenanceClause: string;
		if (row.origin === 'paper_rerun') {
			provenanceClause =
				` A published row, re-run here from the released 2023 code: printed as ` +
				`${fmt3(row.publishedMean as number)}, re-run to ${fmt4(row.foldMean)}, deviation ` +
				`${fmt4(row.absDevVsPublished as number)}.`;
		} else if (row.origin === 'paper_published_only') {
			provenanceClause =
				` A published row, printed at three decimals in 2023 and never re-run here, so there ` +
				`are no per-statement scores for it and no stepped area can be computed.`;
		} else {
			provenanceClause =
				` Newly scored — ${headToHeadDisplay} in the head-to-head — run on the identical ` +
				`statements and labels, under the identical folds, and never trained, so no fold is ` +
				`held out from it. Stepped average precision ${fmt4(row.ap as number)}, so trapezoidal ` +
				`PR-AUC hands it ${fmtSigned4(row.tieGift as number)} of area no cutoff reaches, and it ` +
				`ranks ${row.apRank} of ${scored.length} once that is taken back.`;
		}
		const referenceClause = row.isReference
			? ' This is the reference every paired margin on this page is measured against.'
			: '';
		const ambiguityClause = ambiguousLabels.has(row.label)
			? ' Its rank against at least one other row here falls inside the three decimals the 2023 table was printed to, and is not resolvable.'
			: '';

		extents.push(row.foldMean - row.foldSd, row.foldMean + row.foldSd);
		if (row.ap !== null) extents.push(row.ap);
		if (row.publishedMean !== null) extents.push(row.publishedMean);

		return {
			label: row.label,
			display: budget(row.display, PAPER_TABLE6_LABEL_BUDGET_CHARS, `${context}.display`),
			origin: row.origin,
			family: row.family,
			rank: row.rank,
			foldMean: row.foldMean,
			foldSd: row.foldSd,
			foldCount: row.foldCount,
			rounded: row.origin === 'paper_published_only',
			publishedMean: row.publishedMean,
			absDevVsPublished: row.absDevVsPublished,
			ap: row.ap,
			apRank: row.apRank,
			tieGift: row.tieGift,
			distinctScores: row.distinctScores,
			isReference: row.isReference,
			headToHeadDisplay,
			rankAmbiguousAtPrintedPrecision: ambiguousLabels.has(row.label),
			y,
			rankReadout: budget(
				String(row.rank),
				PAPER_TABLE6_RANK_BUDGET_CHARS,
				`${context}.rankReadout`
			),
			metricReadout,
			tieReadout,
			title:
				`Rank ${row.rank} of ${nRows} — ${row.display}: ${fmtOwn(row.foldMean)} on trapezoidal ` +
				`PR-AUC averaged over the folds, ${dispersion}.${provenanceClause}${referenceClause}${ambiguityClause}`
		};
	});

	// Pairs our OWN printing cannot separate, which is a different failure from the
	// paper's rounding above: the values differ and the order is real, but two rows
	// render the same string. Named so the figure can say which pair that is
	// instead of leaving it to look arbitrary. All pairs again, and here the reason
	// is concrete: this table prints published-only rows at three decimals and
	// measured rows at four, so two identically-printed measured rows can sit
	// either side of a printed-only row whose readout differs from both.
	const printedTies: PaperTable6PrintedTie[] = [];
	for (let i = 0; i < rows.length; i += 1) {
		for (let j = i + 1; j < rows.length; j += 1) {
			const above = rows[i];
			const below = rows[j];
			if (above.metricReadout !== below.metricReadout) continue;
			printedTies.push({
				id: `${above.rank}-${below.rank}`,
				higherDisplay: above.display,
				lowerDisplay: below.display,
				readout: above.metricReadout,
				gap: above.foldMean - below.foldMean
			});
		}
	}

	// Domain snapped outward to the 0.01 grid over everything the figure DRAWS —
	// SD bar ends, tie-robust ghosts and published hairlines — so nothing is
	// cropped and no axis width is spent on a mark that is never rendered. A bound
	// that lands EXACTLY on an extreme is stepped one more hundredth: an end cap
	// flush against the axis start reads as a crop rather than as a bound. Same
	// device as `paper-robustness`, which snaps to its own half-point grid.
	const lo = Math.min(...extents);
	const hi = Math.max(...extents);
	let minH = Math.floor(lo * 100);
	let maxH = Math.ceil(hi * 100);
	if (minH >= lo * 100 - 1e-9) minH -= 1;
	if (maxH <= hi * 100 + 1e-9) maxH += 1;
	const domainMin = Math.max(0, minH / 100);
	const domainMax = Math.min(1, maxH / 100);
	if (!(domainMax > domainMin)) fail('domain', 'degenerate axis range');

	const height = geometry.topPad + rows.length * geometry.laneHeight + geometry.axisPad;

	// The defect class this page keeps re-shipping is a FROZEN JOIN KEY reaching a
	// reader — usually not through a field called `label`, but inside a shipped
	// prose string that happens to name its arms by key. Every free-text field the
	// component prints is therefore checked against every row label here, so the
	// rule holds for text the artifact wrote as well as for text we wrote.
	// `tie.best.noteWithJoinKeys` is the one SHIPPED string that fails it, which is
	// exactly why it is excluded from this list and never rendered — but its plain
	// restatement names its models by display and IS scanned, below.
	// THE PLAIN TWINS, assembled from the values already parsed and gated above.
	// Built from the flat field rather than re-read from `raw`, so `shipped` IS the
	// string this figure carries — the two cannot come apart by construction, and
	// there is no second parse to keep in step. Every plain half joins the join-key
	// scan below on the same footing as the shipped half: a restatement that names
	// a frozen row key is the same defect as a shipped sentence that does.
	const prose: PaperTable6Prose = {
		whatThisIs: { shipped: whatThisIs, plain: TABLE6_PLAIN.whatThisIs },
		rankingRule: { shipped: rankingRule, plain: TABLE6_PLAIN.rankingRule },
		perFoldMetric: { shipped: metric.perFoldMetric, plain: TABLE6_PLAIN.perFoldMetric },
		summary: { shipped: metric.summary, plain: TABLE6_PLAIN.summary },
		uncertaintyNote: { shipped: metric.uncertaintyNote, plain: TABLE6_PLAIN.uncertaintyNote },
		trapezoidalNote: { shipped: metric.trapezoidalNote, plain: TABLE6_PLAIN.trapezoidalNote },
		estimatorSource: { shipped: metric.estimatorSource, plain: TABLE6_PLAIN.estimatorSource },
		origins: {
			ours: { shipped: origins.ours, plain: TABLE6_PLAIN.originOurs },
			paper_rerun: { shipped: origins.paper_rerun, plain: TABLE6_PLAIN.originPaperRerun },
			paper_published_only: {
				shipped: origins.paper_published_only,
				plain: TABLE6_PLAIN.originPaperPublishedOnly
			}
		},
		tieDefinition: { shipped: tie.definition, plain: TABLE6_PLAIN.tieDefinition },
		tieReconciliation: { shipped: tie.reconciliation, plain: TABLE6_PLAIN.tieReconciliation },
		tieSeparationNote: { shipped: tie.separation.note, plain: TABLE6_PLAIN.tieSeparationNote },
		tieSpearmanMethod: { shipped: tie.spearmanMethod, plain: TABLE6_PLAIN.tieSpearmanMethod },
		reproductionNote: { shipped: reproduction.note, plain: TABLE6_PLAIN.reproductionNote },
		conventionVerified: { shipped: convention.verified, plain: TABLE6_PLAIN.conventionVerified },
		headlineTieBreak: {
			shipped: convention.headline.tieBreak,
			plain: TABLE6_PLAIN.headlineTieBreak
		},
		headlineNote: { shipped: convention.headline.note, plain: TABLE6_PLAIN.headlineNote }
	};

	const rowLabels = ordered.map((row) => row.label);
	const renderedProse = [
		prose.whatThisIs.plain,
		prose.rankingRule.plain,
		prose.perFoldMetric.plain,
		prose.summary.plain,
		prose.uncertaintyNote.plain,
		prose.trapezoidalNote.plain,
		prose.estimatorSource.plain,
		prose.origins.ours.plain,
		prose.origins.paper_rerun.plain,
		prose.origins.paper_published_only.plain,
		prose.tieDefinition.plain,
		prose.tieReconciliation.plain,
		prose.tieSeparationNote.plain,
		prose.tieSpearmanMethod.plain,
		prose.reproductionNote.plain,
		prose.conventionVerified.plain,
		prose.headlineTieBreak.plain,
		prose.headlineNote.plain,
		whatThisIs,
		rankingRule,
		metric.perFoldMetric,
		metric.summary,
		metric.uncertaintyField,
		metric.uncertaintyNote,
		metric.trapezoidalNote,
		metric.estimatorSource,
		origins.ours,
		origins.paper_rerun,
		origins.paper_published_only,
		reproduction.note,
		tie.definition,
		tie.reconciliation,
		tie.spearmanMethod,
		tie.separation.note,
		convention.verified,
		convention.headline.note,
		convention.headline.tieBreak,
		tie.best.noteProse.plain,
		...rows.map((row) => row.title)
	];
	for (const prose of renderedProse) {
		for (const label of rowLabels) {
			if (prose.includes(label)) {
				fail('prose', `a rendered string names the frozen join key "${label}": "${prose.slice(0, 90)}…"`);
			}
		}
	}

	return {
		rows,
		origins,
		config,
		whatThisIs,
		rankingRule,
		prose,
		nStatements,
		nPositive,
		nNegative,
		nFolds,
		metric,
		reproduction,
		tie,
		convention,
		ambiguousPairs,
		printedTies,
		leadingOurRanks,
		theirBestRank: theirBest.rank,
		paperReportsNoTests,
		generatedBy,
		domainMin,
		domainMax,
		ticks: ticksOf(domainMin, domainMax),
		height,
		plotBottom: height - geometry.axisPad,
		rankHeader: budget('#', PAPER_TABLE6_RANK_BUDGET_CHARS, 'rankHeader'),
		// Headers sit in the same gutters at 8px, where the 220-unit name gutter is
		// worth 45 characters; budgeted at the stricter 9px figure regardless.
		// The claim it keeps: a published row is listed under the name Table 6
		// printed for it, not a name coined here. What it dropped is the possessive
		// that also, wrongly, implied the five newly scored rows were named there.
		labelHeader: budget(
			'method — published names kept',
			PAPER_TABLE6_LABEL_BUDGET_CHARS,
			'labelHeader'
		),
		// THE TWO COLUMNS ARE ONE QUANTITY COMPUTED TWO WAYS, and the headers now say
		// so. They read "paper metric ±SD" and "tie-robust AP · gift · rank", which
		// named neither quantity: "paper metric" says whose it is rather than what it
		// is, and "tie-robust" / "gift" are this project's own coinages. Both columns
		// are the area under the precision-recall curve — the first drawing straight
		// lines between the points, the second stepping between them — and "credit" is
		// what the straight lines hand a row that the steps do not.
		metricHeader: budget('straight-line area', PAPER_TABLE6_METRIC_BUDGET_CHARS, 'metricHeader'),
		tieHeader: budget('stepped area · credit · rank', PAPER_TABLE6_TIE_BUDGET_CHARS, 'tieHeader')
	};
}

/**
 * Pure, fail-closed validator for the parsed `paper_table6_extended.json`.
 * Returns `status:'ok'` with a drawable ranked list, or `status:'unavailable'`
 * with a reason on any shape or invariant drift. Never throws.
 */
export function validatePaperTable6Extended(
	raw: unknown,
	context: PaperTable6ExtendedContext = {}
): PaperTable6ExtendedLoad {
	const artifactPath = context.artifactPath ?? '';
	const artifactSha256 = context.artifactSha256 ?? null;
	try {
		return {
			status: 'ok',
			figure: buildFigure(record(raw, 'paper_table6_extended')),
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
