/**
 * Typed data contract for the 2023 INDRA paper's LITERAL belief model placed
 * head-to-head with the LLM belief scorers on the identical 1689-statement
 * all-sources-specific panel.
 *
 * This module is import-safe on the client: it holds only the typed shape, the
 * pure ranking math the server calls (`aurocFromPairs`, `aurocOnRankedBlock`),
 * and a pure `validatePaperLiteral()` that fails closed on shape drift. All
 * filesystem work (reading the artifacts, joining LLM probs, computing per-arm
 * score distributions and PR curves) lives in `$lib/server/paper-literal`.
 *
 * DEGRADE CONTRACT. Every server-computed field is NULLABLE (or an empty array),
 * never a numeric placeholder. An arm whose prediction vector fails to join keeps
 * `ece: null`, not `ece: 0` — 0 is the IDEAL calibration error, so a placeholder
 * zero renders a failed join as flawless, better than every arm that joined. Same
 * for the four Brier terms and the ranked-block AUROC. If you add a
 * server-computed number here, make it nullable and let the compiler find the
 * render sites.
 *
 * DEFAULT VIEW AND QUOTED MARGIN ARE DELIBERATELY DIFFERENT. The panel OPENS in
 * the paper's own estimator — fold-mean trapezoidal PR-AUC with the population SD
 * over their folds, their Table 6 idiom — because this page is read by the paper's
 * authors and their measurement should be the first one they see. That estimator
 * flatters the reader arms (trapezoidal interpolation over-credits heavily-tied
 * scores), so the inflation is disclosed in the same view, and the number we QUOTE
 * wherever a single margin is stated stays tie-robust pooled average precision
 * (AP); AUROC is the secondary cross-arm number. Do not "reconcile" the two by
 * re-flipping the default — both halves are intended. This node is distinct from
 * `paper-method-landscape` (the 59-row unpaired published Table-6 summaries) —
 * the two are never merged.
 *
 * AP IS THE MOST CONSERVATIVE LENS, NOT THIS PAGE'S VERDICT. It is quoted because
 * it is the tie-robust ranking number — and it is conservative because this panel
 * is SATURATED. The panel is 73.2% positive and the RF+promoter reference already
 * scores AP 0.9412, so only 0.0588 of the scale is left to win; AP weights the top
 * of a ranking that was already nearly right. The gate's actual work is pushing
 * ERRORS DOWN, which is measured at statement grain by error-class F1: best margin
 * there is +0.1416 (GLM-5 gate) against the same reference, where the best AP
 * margin is +0.0098 (Gemma 4 26B gate). That figure is reported separately, and it
 * carries an ORACLE, quoted here too because a margin without its disclosure is an
 * over-claim wherever it is written: error-class F1 needs a threshold, and every
 * arm's threshold — the reference RF's included — is the panel-wide best-F1 cut,
 * FITTED AND EVALUATED on these same 1,689 statements. The oracle favours the RF,
 * which offers 1,546 candidate cuts against a reader gate's 475–498, so the margin
 * is measured under a rule that is generous to the model it beats. Read an AP
 * margin here as the floor of the comparison — not as the number that settles it,
 * and not as a number that is wrong.
 */

import { validateApDecomposition, type ApDecomposition } from './paper-ap-decomposition.ts';

/**
 * A SHIPPED STRING AND THE PLAIN SENTENCE THAT REPLACES IT ON SCREEN.
 *
 * The dialect that survived three "clean" sweeps of /paper does not live in any
 * template or in any static string in this repo. It arrives at RUNTIME, off
 * sha-pinned artifact JSON — "tau = the smallest of the arm's own distinct
 * scores…", "paired fold-stratified bootstrap over the paper's own out-of-fold
 * fold assignment". A scan of the source cannot see a string that does not exist
 * until a file is read, which is exactly why every sweep came back green while
 * the words were on the screen.
 *
 * The artifact bytes are sha-pinned and may not be edited, so the translation
 * happens HERE, at the loader, once per field, beside the parse that reads it:
 *
 *   · `shipped` is parsed exactly as before — same `text()` call, same
 *     fail-closed gate, byte-identical to the artifact. It is the AUDIT TRAIL and
 *     belongs behind the page's single verification boundary.
 *   · `plain` is authored in the loader that owns the field, as a STATIC string,
 *     which is the whole point: a static string is a string the dialect guard can
 *     finally see, so the restatement a reader is handed is under test where the
 *     artifact's own wording never could be. Each module keeps its restatements in
 *     one block near the top (`ERROR_F1_PLAIN`, `REVIEW_QUEUE_PLAIN`,
 *     `ROBUSTNESS_PLAIN`, `DEPLOYED_BASELINE_PLAIN`, `TABLE6_PLAIN`) so the whole
 *     of what a reader is told can be read in one place.
 *
 * Neither half is optional and neither is derived from the other. Dropping
 * `shipped` would delete the audit trail; dropping `plain` puts a sentence
 * written for a referee in front of a curator.
 */
export interface ShippedProse {
	/**
	 * The string EXACTLY as it ships — artifact bytes verbatim wherever the field
	 * reads them off a sha-pinned file. Audit only; never the sentence a reader is
	 * handed outside the verification boundary.
	 */
	shipped: string;
	/** The plain restatement, authored here. THIS is what a reader sees. */
	plain: string;
}

/**
 * One plain restatement, PINNED to the shipped sentence it restates.
 *
 * A twin for a NAMED field is safe by construction — `threshold_rule`'s plain
 * half is written under the key that reads `threshold_rule`, and the two cannot
 * come apart. A twin for an ARRAY is not: `caveats[4]` is bound to its
 * restatement by position alone, so a reissued artifact that inserts, drops or
 * reorders one caveat would silently print restatement 4 under caveat 5 — a
 * wrong sentence attributed to a sha-pinned file, which is the exact failure the
 * whole page is built to make impossible.
 *
 * So each positional twin carries a short verbatim fragment of the sentence it
 * was written for, and `pairShippedProse` gates on finding it. Drift stops the
 * figure instead of mislabelling it.
 */
export interface AnchoredProse {
	/**
	 * A distinctive fragment of the SHIPPED sentence, quoted verbatim. Artifact
	 * bytes: it is deliberately left in the artifact's own words — including its
	 * dialect — because its job is to match those bytes, not to be read.
	 */
	artifactAnchor: string;
	/** The plain restatement of the sentence that anchor identifies. */
	plain: string;
}

/**
 * Pair a shipped string array with its plain restatements, FAIL-CLOSED on drift.
 * Throws — every caller is already inside a validator's try/catch, so a drifted
 * artifact gates its figure exactly like any other shape failure.
 */
export function pairShippedProse(
	shipped: readonly string[],
	twins: readonly AnchoredProse[],
	context: string
): ShippedProse[] {
	if (shipped.length !== twins.length) {
		throw new Error(
			`${context}: expected ${twins.length} entries with plain restatements, got ${shipped.length}`
		);
	}
	return shipped.map((entry, index) => {
		const twin = twins[index];
		if (!entry.includes(twin.artifactAnchor)) {
			throw new Error(
				`${context}[${index}]: no longer the sentence its plain restatement was written for ` +
					`(expected to contain ${JSON.stringify(twin.artifactAnchor)})`
			);
		}
		return { shipped: entry, plain: twin.plain };
	});
}

/**
 * ONE shipped string with its plain restatement, PINNED the same way.
 *
 * The array form above covers `caveats[]`. This covers the other unsafe shape:
 * a field whose text varies ROW BY ROW — every `note` on a nine-model table,
 * every `origin` on an operating-point row, every `what_it_computes` on a form
 * of INDRA's belief. Those cannot be twinned under a key that names them,
 * because one key (`note`) addresses nine different sentences, so each one
 * carries a verbatim fragment of the sentence its restatement was written for
 * and a drifted row gates the figure instead of being mislabelled.
 *
 * Use the plain `{ shipped: text(obj.x, …), plain: MODULE_PLAIN.x }` form for a
 * field whose key names exactly one sentence — there, the twin is safe by
 * construction and an anchor would only be ceremony.
 */
export function anchoredShippedProse(
	shipped: string,
	twin: AnchoredProse,
	context: string
): ShippedProse {
	if (!shipped.includes(twin.artifactAnchor)) {
		throw new Error(
			`${context}: no longer the sentence its plain restatement was written for ` +
				`(expected to contain ${JSON.stringify(twin.artifactAnchor)})`
		);
	}
	return { shipped, plain: twin.plain };
}

/**
 * Look a row's twin up by its FROZEN key, then pin it to the text as above.
 * A key with no restatement gates: a new row is a new sentence to explain, and
 * shipping it unexplained is the failure the whole twin mechanism exists to stop.
 */
export function keyedShippedProse(
	key: string,
	shipped: string,
	twins: Readonly<Record<string, AnchoredProse>>,
	context: string
): ShippedProse {
	const twin = twins[key];
	if (twin === undefined) {
		throw new Error(`${context}: no plain restatement is authored for "${key}"`);
	}
	return anchoredShippedProse(shipped, twin, context);
}

export type PaperMetric = 'ap' | 'auroc' | 'trapezoidal';

export const PAPER_METRIC_LABELS: Record<PaperMetric, string> = {
	ap: 'average precision',
	auroc: 'AUROC',
	trapezoidal: 'trapezoidal PR-AUC'
};

/**
 * The lens the head-to-head OPENS in: the paper's own fold-mean trapezoidal
 * PR-AUC, shown with its tie inflation disclosed inline. This is the default
 * VIEW, not the quoted margin — the number we quote stays tie-robust average
 * precision (`ap`), the most conservative of the three lenses, one click away on
 * the same switch. Do not re-flip this to `ap`.
 */
export const PAPER_DEFAULT_METRIC: PaperMetric = 'trapezoidal';

export type PaperArmKind = 'paper' | 'port' | 'llm';

/** Fixed server-side histogram/curve geometry, shared by loader and consumers. */
export const PAPER_SCORE_BIN_COUNT = 40;
export const PAPER_SCORE_TOP_PILES = 4;
export const PAPER_PR_CURVE_MAX_POINTS = 120;
/** Reliability (calibration) geometry: equal-width [0,1] bins, server-filled. */
export const PAPER_RELIABILITY_BIN_COUNT = 10;
/**
 * A statement is in an arm's RANKED block when its score is strictly above this.
 * The reader gates emit exactly 0 for a statement whose evidence they rejected
 * outright, so 0 is not a low rank — it is the arm declining to rank at all, and
 * the whole zeroed block is one tie.
 */
export const PAPER_RANKED_BLOCK_MIN_SCORE = 0;

/**
 * The single arm→color source of truth shared by ScoreDistribution,
 * ApDecompositionByPaperRank, and PaperReliabilityStrip so the paper/port arms
 * and every LLM keep ONE hue across all three panels. Returns an existing layout
 * token — the paper's literal/port model in `--accent`, every LLM in `--blocked`
 * — never a new hardcoded color. Each of those panels shows every LLM in one
 * hue, so keying on `kind` fully satisfies "each LLM keeps one hue everywhere".
 */
export function paperArmColorVar(kind: PaperArmKind): string {
	return kind === 'llm' ? 'var(--blocked)' : 'var(--accent)';
}

/** The paired-delta reference arm: its delta is null (it is the baseline). */
export const PAPER_LITERAL_REFERENCE_ARM_ID = 'paper-rf-promoter';

export interface PaperLiteralArmSpec {
	id: string;
	/** Exact `point_metrics` display label in paper_literal_vs_llms.json. */
	label: string;
	kind: PaperArmKind;
	/**
	 * What the reader sees. DECOUPLED from `label` on purpose: `label` is a frozen
	 * join key into already-emitted artifacts and must never change, but "Paper
	 * literal" is our coinage, not the paper's, and this page goes to the people
	 * who named these models. Where an arm IS one of their published methods, this
	 * is their own Table name for it.
	 *
	 * The four LLM rows read "… reading", not "… gate". "Gate" was our dialect for
	 * the step where the model reads each piece of evidence and keeps or drops it;
	 * nobody outside this repo knows that, so the on-screen name says what the step
	 * IS. The shipped artifacts still say "gate" in their own frozen prose and in
	 * every join key — those bytes are sha-pinned and are quoted verbatim where
	 * they are printed, with the plain restatement beside them.
	 */
	display: string;
	/**
	 * The name the SHIPPED artifacts carry for this arm in their own `display`
	 * field — sha-pinned bytes, as frozen as `label` and never rendered.
	 *
	 * It exists because `display` above stopped being the artifact's string when
	 * "gate" became "reading", and `statement_error_f1.json` carries a `display`
	 * per arm that a loader checks for DRIFT. Comparing it to our on-screen name
	 * would have meant either editing sha-pinned bytes or deleting the check; this
	 * field keeps the check live against a frozen expectation, so a reissued
	 * artifact that renames an arm still gates the figure. Where the two are the
	 * same string they are still written out separately, because that is the only
	 * thing stopping a future rename from silently moving both.
	 */
	artifactDisplay: string;
}

/**
 * Canonical ordered arm set. `label` is the join key into `point_metrics` and
 * `paired_delta_vs_paper_literal` — FROZEN, it addresses shipped data. `display`
 * is the on-screen name. `kind` follows the paper|port|llm union (the INDRA CoGEx
 * hybrid model-bundle arm is carried under `llm`).
 */
export const PAPER_LITERAL_ARM_SPECS: readonly PaperLiteralArmSpec[] = [
	{
		id: 'paper-rf-promoter',
		label: 'Paper literal RF+promoter',
		display: 'RF 2k-d13 + Type/#PMIDs/promoter',
		artifactDisplay: 'RF 2k-d13 + Type/#PMIDs/promoter',
		kind: 'paper'
	},
	{
		id: 'paper-rf-prom-avglen',
		label: 'Paper literal RF+prom/avglen',
		display: 'RF 2k-d13 + Type/#PMIDs/prom/avglen',
		artifactDisplay: 'RF 2k-d13 + Type/#PMIDs/prom/avglen',
		kind: 'paper'
	},
	{
		id: 'port-rf-promoter',
		label: 'Paper semantic port RF+promoter',
		display: 'Port of RF + Type/#PMIDs/promoter',
		artifactDisplay: 'Our port of RF + Type/#PMIDs/promoter',
		kind: 'port'
	},
	{
		id: 'gemma-4-e2b',
		label: 'Gemma 4 E2B',
		display: 'Gemma 4 E2B reading',
		artifactDisplay: 'Gemma 4 E2B gate',
		kind: 'llm'
	},
	{
		id: 'gemma-4-26b',
		label: 'Gemma 4 26B',
		display: 'Gemma 4 26B reading',
		artifactDisplay: 'Gemma 4 26B gate',
		kind: 'llm'
	},
	{
		id: 'gemma-4-31b',
		label: 'Gemma 4 31B',
		display: 'Gemma 4 31B reading',
		artifactDisplay: 'Gemma 4 31B gate',
		kind: 'llm'
	},
	{
		id: 'glm-5',
		label: 'GLM-5',
		display: 'GLM-5 reading',
		artifactDisplay: 'GLM-5 gate',
		kind: 'llm'
	},
	{
		id: 'indra-cogex-hybrid',
		label: 'INDRA CoGEx hybrid',
		display: 'INDRA CoGEx hybrid',
		artifactDisplay: 'INDRA CoGEx hybrid',
		kind: 'llm'
	}
] as const;

// ---------------------------------------------------------------------------
// WHERE AN INTERVAL SITS RELATIVE TO ZERO — the page's ONE classifier
//
// This replaces a boolean named `excludesZero`, defined `ciLow > 0 || ciHigh < 0`
// and therefore TRUE for an interval lying entirely BELOW zero. Its name invited
// `x.excludesZero ? 'better' : 'not better'`, and that exact two-way branch
// shipped SIX times on /paper — the last time in brand-new code written in a wave
// whose own invariants warned about it, printing "Clears zero." over an interval
// of −0.0256 to −0.0061. A static guard existed and did not catch it: the guard
// only fires when a directional WORD sits in the same expression, and "Clears
// zero." contains none.
//
// Guards did not stop it, so the boolean is gone. Loaders export this three-way
// class instead, derived ONCE here from the ENDPOINTS. A two-way branch on a
// direction cannot be written, because there is no boolean to write it against;
// a render site that genuinely needs "significant either way" asks
// `standing !== 'not-significant'`, which reads as what it is.
//
// Render sites should key a TOTAL `Record<Standing, string>` (or a `switch` with
// no default) so the compiler demands a sentence for every case.
// ---------------------------------------------------------------------------

/** Where one interval sits relative to zero. There is no fourth class. */
export type Standing = 'ahead' | 'behind' | 'not-significant';

/**
 * The one classifier. Read off the two ENDPOINTS, never off a shipped flag and
 * never off the sign of the point estimate: `low > 0` and `high < 0` are the two
 * facts and they cannot both hold. Callers whose point estimate may sit outside
 * its own interval must gate on that themselves — a direction and a range that
 * disagree is not a case this function can classify honestly.
 */
export function standingOfBounds(low: number, high: number): Standing {
	if (low > 0) return 'ahead';
	if (high < 0) return 'behind';
	return 'not-significant';
}

export interface PaperLiteralDeltaEntry {
	delta: number;
	ciLow: number;
	ciHigh: number;
	/** ahead / behind / not-significant, from this entry's own CI endpoints. */
	standing: Standing;
}

/**
 * The head-to-head Δ column's triangle, one per class. Authored here rather than
 * in the component so the sweep that reads every string a `paper-*.ts` module
 * writes can see it; the component prints it as given.
 */
export const PAPER_DELTA_MARK: Readonly<Record<Standing, string>> = {
	ahead: '▲',
	behind: '▼',
	'not-significant': ''
};

/**
 * The sentence read aloud beside that triangle. TOTAL over the three classes, so
 * the compiler demands one for each: the fourth case with no words of its own is
 * how a case ends up wearing another case's sentence, which is what the deleted
 * two-way boolean did here for two significantly LOSING models.
 */
export const PAPER_DELTA_STANDING_SENTENCE: Readonly<Record<Standing, string>> = {
	ahead: 'the whole range sits on one side of zero — clearly ahead of the random forest',
	behind: 'the whole range sits on one side of zero — clearly behind the random forest',
	'not-significant': 'the range includes zero'
};

export type PaperLiteralDelta = Record<PaperMetric, PaperLiteralDeltaEntry>;

export interface PaperLiteralScorePile {
	value: number;
	count: number;
}

export interface PaperLiteralPrPoint {
	recall: number;
	precision: number;
}

export interface PaperLiteralReliabilityBin {
	/** Mean predicted probability of the statements that fell in this bin. */
	p_mean: number;
	/** Observed positive (released-correct) rate among them. */
	y_rate: number;
	/** Statements in the bin. */
	n: number;
}

/**
 * AUROC re-measured on the block an arm actually ORDERS, i.e. the statements it
 * scores strictly above `PAPER_RANKED_BLOCK_MIN_SCORE`.
 *
 * Why this rides beside the AUROC lens. The reader gates zero a statement whose
 * evidence they reject wholesale, so their score vector is one big tied block at
 * 0 plus a ranked remainder. AUROC over the whole panel credits that binary
 * split — every ranked statement beats every zeroed one — and on this panel the
 * split is where nearly all of the reader arms' AUROC margin comes from. Holding
 * the arm's own ranked subset fixed and scoring the paper reference arm on the
 * SAME statements separates "orders the panel better" from "splits it in two".
 * `armAuroc` and `referenceAuroc` are therefore always measured on one identical
 * statement set; comparing an arm's `armAuroc` to another arm's is NOT valid
 * (different subsets), which is why the render always pairs it with its own
 * reference number.
 */
export interface PaperLiteralAurocOnRanked {
	/** Statements this arm scores above zero — the ones it puts in an order. */
	nRanked: number;
	/** Statements it zeroes into one tied block; 0 for an arm that ranks everything. */
	nZeroed: number;
	/** This arm's AUROC over `nRanked` alone. */
	armAuroc: number;
	/** The paper reference arm's AUROC over those SAME `nRanked` statements. */
	referenceAuroc: number;
}

export interface PaperLiteralArm {
	id: string;
	label: string;
	/** On-screen name; see PaperLiteralArmSpec.display. */
	display: string;
	kind: PaperArmKind;
	/**
	 * Pooled average precision — the tie-robust cross-arm lens we quote, and the
	 * most CONSERVATIVE of the three: this panel is saturated (majority-positive,
	 * reference already at ~0.94), so an AP margin is the floor of the comparison,
	 * not its verdict. See the module note.
	 */
	ap: number;
	auroc: number;
	/**
	 * Fold-mean trapezoidal PR-AUC — the paper's own published estimator and the
	 * default VIEW. It inflates heavily-tied scores, so it is never the quoted
	 * margin; it also documents Table-6 reproduction faithfulness.
	 */
	trapezoidal: number;
	/**
	 * Population SD of that fold mean across the paper's own cross-validation
	 * folds — their Table 6 idiom. A dispersion measure, NOT a confidence
	 * interval, and it belongs to the trapezoidal estimator alone: never render it
	 * beside `ap` or `auroc`.
	 */
	foldPopulationSd: number;
	distinctScores: number;
	/** Paired delta vs the RF+promoter reference; null for the reference itself. */
	delta: PaperLiteralDelta | null;
	/** Server-computed: PAPER_SCORE_BIN_COUNT fixed histogram bins over [0,1]. */
	scoreBins: number[];
	/** Server-computed: the top exact-value score piles. */
	scoreTopPiles: PaperLiteralScorePile[];
	/** Server-computed: stepped PR curve on the paper released label. */
	prCurve: PaperLiteralPrPoint[];
	/** Server-computed: occupied equal-width [0,1] reliability (calibration) bins. */
	reliabilityBins: PaperLiteralReliabilityBin[];
	/**
	 * Server-computed expected calibration error over those bins — null when the
	 * arm's predictions did not join. NEVER 0 on that path: 0 is the ideal value,
	 * so a placeholder would print a broken join as perfect calibration. Consumers
	 * must render an explicit unavailable state, never a number.
	 */
	ece: number | null;
	/**
	 * Server-computed logistic-recalibration slope (label ~ 1 + logit(score));
	 * ideal 1, LLMs ~0.15–0.21 (over-dispersed). null where the MLE is not
	 * identifiable (single class / non-convergence), matching the Python NaN.
	 */
	calibrationSlope: number | null;
	/** Server-computed logistic-recalibration intercept; ideal 0. null as above. */
	calibrationIntercept: number | null;
	/**
	 * Server-computed Murphy Brier = uncertainty + reliability − resolution (raw
	 * mean sq. error). Null on the un-joined path, for the same reason `ece` is:
	 * 0 is the perfect score, so a placeholder would flatter a failed join.
	 */
	brier: number | null;
	/** Server-computed miscalibration penalty Σ(n_k/N)(p̄_k−ō_k)² over the 10 bins; null as above. */
	brierReliability: number | null;
	/** Server-computed discrimination credit Σ(n_k/N)(ō_k−ō)² over the 10 bins; null as above. */
	brierResolution: number | null;
	/** Server-computed irreducible floor ō(1−ō); null as above. */
	brierUncertainty: number | null;
	/**
	 * Server-computed AUROC on this arm's own ranked block, with the paper
	 * reference arm scored on the same statements. Null when the arm or the
	 * reference fails to join, or when the subset is single-class — the AUROC lens
	 * then has to say the check is unavailable rather than show the panel-wide gain
	 * unqualified.
	 */
	aurocOnRanked: PaperLiteralAurocOnRanked | null;
}

export interface PaperLiteralFaithfulness {
	pearsonR: number;
	spearmanR: number;
	meanAbsDiff: number;
	maxAbsDiff: number;
	foldMeanPrAucLiteral: number;
	foldMeanPrAucPort: number;
}

/**
 * Published-Table-6 reconciliation + paper provenance, threaded off the run
 * manifest. Carried on the load so the viewer never re-reads a data file:
 * `maxAbsDeltaVsPublishedTable6` is the largest fold-mean deviation from the
 * paper's Table 6, `headlineLiteral`/`headlinePublished` are the RF+prom/avglen
 * all-sources-specific fold-means (literal reproduction vs the published value),
 * and `paperCodeCommit` pins the exact sorgerlab/indra_assembly_paper commit run.
 * `cvProtocol` is the manifest's verbatim cross-validation description, so any
 * claim about the paper's folds (their count, their shuffle, their seed) is read
 * off the artifact instead of typed into a component.
 */
export interface PaperLiteralReproduction {
	maxAbsDeltaVsPublishedTable6: number;
	headlineLiteral: number;
	headlinePublished: number;
	paperCodeCommit: string;
	/** Verbatim `protocol.cv` from the run manifest — never paraphrased. */
	cvProtocol: string;
}

export interface PaperLiteralOk {
	status: 'ok';
	reason: null;
	artifact_path: string;
	/**
	 * NULLABLE on the ok branch too. The server always supplies a digest; the pure
	 * client-side validate path does not, and coalescing that to '' printed an
	 * empty provenance line that read as a real, empty sha. Null is typed so the
	 * compiler finds every render site and each one states the absence.
	 */
	artifact_sha256: string | null;
	arms: PaperLiteralArm[];
	faithfulness: PaperLiteralFaithfulness;
	/** Table-6 reconciliation + pinned paper commit; null when the manifest drifts. */
	reproduction: PaperLiteralReproduction | null;
	/**
	 * Band-by-band decomposition of the ΔAP column. NON-OPTIONAL: the figure
	 * explains the head-to-head delta, so a missing or drifted payload gates the
	 * whole paper load to `unavailable` rather than rendering the table with the
	 * explanation quietly dropped.
	 */
	apDecomposition: ApDecomposition;
	generatedNote: string;
	/**
	 * `generatedNote` with the plain restatement beside it.
	 *
	 * Unlike every other twin on /paper this one's `shipped` half is OURS, not
	 * artifact bytes — the note is assembled here. It is carried in the same shape
	 * anyway so a render site never has to ask which fields have a plain half, and
	 * `shipped` still says "the string as it ships" rather than "the string as the
	 * artifact wrote it". The server loader appends " Reproduced <date>." when the
	 * manifest carries one, and it appends to BOTH halves; `generatedNote` above is
	 * always exactly `generatedNoteProse.shipped`.
	 */
	generatedNoteProse: ShippedProse;
}

export interface PaperLiteralUnavailable {
	status: 'unavailable';
	reason: string;
	artifact_path: string;
	artifact_sha256: string | null;
	arms: [];
	faithfulness: null;
	reproduction: null;
	apDecomposition: null;
	generatedNote: string | null;
	/** Null on the gated branch, exactly like `generatedNote` itself. */
	generatedNoteProse: ShippedProse | null;
}

export type PaperLiteralLoad = PaperLiteralOk | PaperLiteralUnavailable;

/** Provenance + sibling payloads the server threads through the pure validator. */
export interface PaperLiteralContext {
	artifactPath?: string;
	artifactSha256?: string;
	/**
	 * Parsed `ap_decomposition_by_paper_band.json`. REQUIRED — omitting it (or
	 * passing a drifted payload) gates the load to `unavailable`. It is carried on
	 * the context rather than inside the head-to-head artifact because it is a
	 * separate file with its own sha in the run manifest.
	 */
	apDecomposition?: unknown;
}

/**
 * One aligned (score, label) row. `key` is the canonical `statement_id` both the
 * paper OOF vector and the LLM prediction bundles resolve to, so two arms can be
 * intersected on the same statements; it is null when a row could not be keyed,
 * and such rows are excluded from any cross-arm comparison.
 */
export interface PaperScoredPair {
	key: string | null;
	score: number;
	label: number;
}

/**
 * Tie-aware AUROC (Mann–Whitney U with mid-ranks), the identical estimator to
 * `sklearn.metrics.roc_auc_score`. VERIFIED against the shipped artifact: run
 * over the joined vectors it reproduces `point_metrics[*].auroc` for all eight
 * arms to 4 dp (0.8516 / 0.8527 / 0.8519 / 0.8400 / 0.9010 / 0.8979 / 0.9025 /
 * 0.8272). Mid-ranks matter here — the reader arms tie hundreds of statements at
 * a time, and the naive "count strictly-greater pairs" form would silently score
 * those ties as wins. Returns null when the vector is single-class (AUROC is then
 * undefined, not 0.5).
 */
export function aurocFromPairs(pairs: readonly { score: number; label: number }[]): number | null {
	const sorted = [...pairs].sort((a, b) => a.score - b.score);
	const ranks = new Array<number>(sorted.length);
	let index = 0;
	while (index < sorted.length) {
		let last = index;
		while (last + 1 < sorted.length && sorted[last + 1].score === sorted[index].score) last += 1;
		const midRank = (index + last) / 2 + 1;
		for (let at = index; at <= last; at += 1) ranks[at] = midRank;
		index = last + 1;
	}
	let rankSumPositive = 0;
	let positives = 0;
	for (let at = 0; at < sorted.length; at += 1) {
		if (sorted[at].label > 0) {
			rankSumPositive += ranks[at];
			positives += 1;
		}
	}
	const negatives = sorted.length - positives;
	if (positives === 0 || negatives === 0) return null;
	return (rankSumPositive - (positives * (positives + 1)) / 2) / (positives * negatives);
}

/**
 * Re-measure one arm, and the paper reference arm, on the statements that arm
 * actually orders — see `PaperLiteralAurocOnRanked` for why the AUROC lens needs
 * this. `nRanked` counts only rows that are above-zero AND resolvable in
 * `referenceScoreByKey`, because the two AUROCs must be over one identical
 * statement set for the comparison to mean anything. `nZeroed` counts every
 * zeroed row of the arm, keyed or not — it describes the arm, not the join.
 *
 * Returns null (never a stand-in number) when nothing survives the intersection
 * or either side is single-class on it.
 */
export function aurocOnRankedBlock(
	armPairs: readonly PaperScoredPair[],
	referenceScoreByKey: ReadonlyMap<string, number>
): PaperLiteralAurocOnRanked | null {
	const ranked: { score: number; label: number; referenceScore: number }[] = [];
	let nZeroed = 0;
	for (const pair of armPairs) {
		if (!(pair.score > PAPER_RANKED_BLOCK_MIN_SCORE)) {
			nZeroed += 1;
			continue;
		}
		if (pair.key === null) continue;
		const referenceScore = referenceScoreByKey.get(pair.key);
		if (referenceScore === undefined) continue;
		ranked.push({ score: pair.score, label: pair.label, referenceScore });
	}
	if (ranked.length === 0) return null;
	const armAuroc = aurocFromPairs(ranked);
	const referenceAuroc = aurocFromPairs(
		ranked.map((row) => ({ score: row.referenceScore, label: row.label }))
	);
	if (armAuroc === null || referenceAuroc === null) return null;
	return {
		nRanked: ranked.length,
		nZeroed,
		armAuroc: Math.round(armAuroc * 1e6) / 1e6,
		referenceAuroc: Math.round(referenceAuroc * 1e6) / 1e6
	};
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

/** Any finite number (deltas and CI bounds are signed). */
function number(value: unknown, context: string): number {
	if (typeof value !== 'number' || !Number.isFinite(value)) fail(context, 'expected a finite number');
	return value;
}

/** Finite number constrained to the [0,1] probability/metric range. */
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

function sameLabelSet(got: string[], want: readonly string[]): boolean {
	if (got.length !== want.length) return false;
	const a = [...got].sort();
	const b = [...want].sort();
	return a.every((value, index) => value === b[index]);
}

function parseDeltaEntry(value: unknown, context: string): PaperLiteralDeltaEntry {
	const obj = record(value, context);
	const delta = number(obj.delta, `${context}.delta`);
	const ciLow = number(obj.ci95_low, `${context}.ci95_low`);
	const ciHigh = number(obj.ci95_high, `${context}.ci95_high`);
	if (ciLow > ciHigh) fail(context, 'ci95_low must not exceed ci95_high');
	return { delta, ciLow, ciHigh, standing: standingOfBounds(ciLow, ciHigh) };
}

function parseDelta(value: unknown, context: string): PaperLiteralDelta {
	const obj = record(value, context);
	return {
		ap: parseDeltaEntry(obj.pooled_average_precision, `${context}.pooled_average_precision`),
		auroc: parseDeltaEntry(obj.auroc, `${context}.auroc`),
		trapezoidal: parseDeltaEntry(
			obj.fold_mean_trapezoidal_pr_auc,
			`${context}.fold_mean_trapezoidal_pr_auc`
		)
	};
}

function parseArm(
	spec: PaperLiteralArmSpec,
	pointMetrics: UnknownRecord,
	pairedDelta: UnknownRecord
): PaperLiteralArm {
	const context = `point_metrics[${spec.label}]`;
	const point = record(pointMetrics[spec.label], context);
	const rawDelta = pairedDelta[spec.label];
	return {
		id: spec.id,
		label: spec.label,
		display: spec.display,
		kind: spec.kind,
		ap: unit(point.pooled_average_precision, `${context}.pooled_average_precision`),
		auroc: unit(point.auroc, `${context}.auroc`),
		trapezoidal: unit(point.fold_mean_trapezoidal_pr_auc, `${context}.fold_mean_trapezoidal_pr_auc`),
		foldPopulationSd: unit(point.fold_population_sd, `${context}.fold_population_sd`),
		distinctScores: positiveInteger(point.distinct_scores, `${context}.distinct_scores`),
		delta:
			rawDelta === undefined
				? null
				: parseDelta(rawDelta, `paired_delta_vs_paper_literal[${spec.label}]`),
		// Score geometry is server-computed from the aligned prediction vectors.
		scoreBins: [],
		scoreTopPiles: [],
		prCurve: [],
		// Reliability geometry is server-computed from those same aligned vectors.
		reliabilityBins: [],
		// Every server-computed SCALAR defaults to null, never 0. `ece: 0` and
		// `brier: 0` are the IDEAL values on their scales, so a numeric placeholder
		// makes an arm that never joined outscore every arm that did.
		ece: null,
		calibrationSlope: null,
		calibrationIntercept: null,
		brier: null,
		brierReliability: null,
		brierResolution: null,
		brierUncertainty: null,
		aurocOnRanked: null
	};
}

function parseFaithfulness(value: unknown): PaperLiteralFaithfulness {
	const obj = record(value, 'faithfulness_literal_vs_port');
	return {
		pearsonR: unit(obj.pearson_r, 'faithfulness_literal_vs_port.pearson_r'),
		spearmanR: unit(obj.spearman_r, 'faithfulness_literal_vs_port.spearman_r'),
		meanAbsDiff: unit(obj.mean_abs_diff, 'faithfulness_literal_vs_port.mean_abs_diff'),
		maxAbsDiff: unit(obj.max_abs_diff, 'faithfulness_literal_vs_port.max_abs_diff'),
		foldMeanPrAucLiteral: unit(
			obj.fold_mean_pr_auc_literal,
			'faithfulness_literal_vs_port.fold_mean_pr_auc_literal'
		),
		foldMeanPrAucPort: unit(
			obj.fold_mean_pr_auc_port,
			'faithfulness_literal_vs_port.fold_mean_pr_auc_port'
		)
	};
}

/**
 * Pure, fail-closed parse of the run manifest's reproduction-fidelity block.
 * Reads `reproduction_fidelity.max_abs_delta_vs_published_table6`,
 * `reproduction_fidelity.headline_rf_prom_avglen_all_sources_specific.{literal,
 * published}`, `paper.code_commit`, and the verbatim `protocol.cv`. Returns the
 * five fields, or `null` on any shape drift (missing keys, wrong types,
 * out-of-[0,1] metrics, empty commit, empty CV protocol).
 * Never throws — reuses the module's `record`/`unit` guards and swallows their
 * throw into a null so a manifest change can never crash the load.
 */
export function parsePaperReproduction(manifestRaw: unknown): PaperLiteralReproduction | null {
	try {
		const manifest = record(manifestRaw, 'manifest');
		const fidelity = record(manifest.reproduction_fidelity, 'manifest.reproduction_fidelity');
		const headline = record(
			fidelity.headline_rf_prom_avglen_all_sources_specific,
			'manifest.reproduction_fidelity.headline_rf_prom_avglen_all_sources_specific'
		);
		const paper = record(manifest.paper, 'manifest.paper');
		if (typeof paper.code_commit !== 'string' || paper.code_commit.length === 0) {
			fail('manifest.paper.code_commit', 'expected a non-empty string');
		}
		const protocol = record(manifest.protocol, 'manifest.protocol');
		if (typeof protocol.cv !== 'string' || protocol.cv.length === 0) {
			fail('manifest.protocol.cv', 'expected a non-empty string');
		}
		return {
			maxAbsDeltaVsPublishedTable6: unit(
				fidelity.max_abs_delta_vs_published_table6,
				'manifest.reproduction_fidelity.max_abs_delta_vs_published_table6'
			),
			headlineLiteral: unit(
				headline.literal,
				'manifest.reproduction_fidelity.headline_rf_prom_avglen_all_sources_specific.literal'
			),
			headlinePublished: unit(
				headline.published,
				'manifest.reproduction_fidelity.headline_rf_prom_avglen_all_sources_specific.published'
			),
			paperCodeCommit: paper.code_commit,
			cvProtocol: protocol.cv
		};
	} catch {
		return null;
	}
}

/**
 * Pure, fail-closed validator for the parsed `paper_literal_vs_llms.json`.
 * Returns `status:'ok'` with the 8 canonical arms carrying their scalar metrics
 * and paired deltas (score geometry left empty for the server to fill), or
 * `status:'unavailable'` with a reason on any shape drift. Never throws.
 *
 * `context.apDecomposition` is REQUIRED: the AP-decomposition payload is
 * validated here and gates the whole load, so the figure that explains the ΔAP
 * column can never silently disappear while the table it explains still renders.
 */
export function validatePaperLiteral(
	raw: unknown,
	context: PaperLiteralContext = {}
): PaperLiteralLoad {
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

		const labels = PAPER_LITERAL_ARM_SPECS.map((spec) => spec.label);
		if (!sameLabelSet(Object.keys(pointMetrics), labels)) {
			fail('paper_literal_vs_llms.point_metrics', 'must contain exactly the canonical arm labels');
		}

		const referenceSpec = PAPER_LITERAL_ARM_SPECS.find(
			(spec) => spec.id === PAPER_LITERAL_REFERENCE_ARM_ID
		);
		if (!referenceSpec) fail('paper_literal_vs_llms', 'reference arm spec is missing');
		const nonReferenceLabels = labels.filter((label) => label !== referenceSpec.label);
		if (!sameLabelSet(Object.keys(pairedDelta), nonReferenceLabels)) {
			fail(
				'paper_literal_vs_llms.paired_delta_vs_paper_literal',
				'must contain every non-reference arm exactly once'
			);
		}

		const arms = PAPER_LITERAL_ARM_SPECS.map((spec) => parseArm(spec, pointMetrics, pairedDelta));
		const faithfulness = parseFaithfulness(obj.faithfulness_literal_vs_port);

		if (context.apDecomposition === undefined) {
			fail(
				'ap_decomposition_by_paper_band',
				'payload is missing — the ΔAP decomposition is required, not optional'
			);
		}
		const apDecomposition = validateApDecomposition(context.apDecomposition);

		const generatedNoteProse: ShippedProse = {
			shipped: `Paper literal belief model vs LLM scorers over ${nStatements} all-sources-specific statements; quoted margin = ${PAPER_METRIC_LABELS.ap}, the most conservative lens on this saturated benchmark.`,
			plain: `The belief models published in the 2023 INDRA assembly paper, re-run against four models that read the evidence, over the same ${nStatements} statements. The margin quoted is average precision — the strictest of the three ways to score an ordering here, and the one that leaves an already-good ordering the least room to gain.`
		};

		return {
			status: 'ok',
			reason: null,
			artifact_path: artifactPath,
			artifact_sha256: artifactSha256,
			arms,
			faithfulness,
			// Manifest-sourced; the server loader fills it via parsePaperReproduction.
			reproduction: null,
			apDecomposition,
			generatedNote: generatedNoteProse.shipped,
			generatedNoteProse
		};
	} catch (error) {
		return {
			status: 'unavailable',
			reason: error instanceof Error ? error.message : String(error),
			artifact_path: artifactPath,
			artifact_sha256: artifactSha256,
			arms: [],
			faithfulness: null,
			reproduction: null,
			apDecomposition: null,
			generatedNote: null,
			generatedNoteProse: null
		};
	}
}
