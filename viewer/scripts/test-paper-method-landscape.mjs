import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';

import {
	PAPER_COMPARABLE_SLICE,
	PAPER_METHOD_EXPECTED_SHA256,
	PAPER_METHOD_SLICE_COUNTS,
	PAPER_METHOD_SLICES,
	validatePaperMethodLandscape
} from '../src/lib/data/paper-method-landscape.ts';
import { PAPER_LITERAL_ARM_SPECS } from '../src/lib/data/paper-literal.ts';
import {
	PAPER_OWN_METRIC_GEOMETRY,
	PAPER_OWN_METRIC_GROUPS,
	PAPER_OWN_METRIC_GROUP_STYLES,
	PAPER_OWN_METRIC_LABEL_BUDGET_CHARS,
	PAPER_OWN_METRIC_READOUT_BUDGET_CHARS,
	PAPER_REPRODUCED_ROW_BY_ARM_ID,
	buildPaperOwnMetric
} from '../src/lib/data/paper-own-metric.ts';

const artifactUrl = new URL('../../data/benchmark/indra_paper_2023_published_method_metrics.json', import.meta.url);
const bytes = readFileSync(artifactUrl);
const digest = createHash('sha256').update(bytes).digest('hex');
const raw = JSON.parse(bytes.toString('utf8'));
let failures = 0;

function ok(value, label) {
	if (!value) {
		failures += 1;
		console.error(`FAIL ${label}`);
	}
}

function eq(got, want, label) {
	if (JSON.stringify(got) !== JSON.stringify(want)) {
		failures += 1;
		console.error(`FAIL ${label}: got ${JSON.stringify(got)}, want ${JSON.stringify(want)}`);
	}
}

function rejects(candidate, label, fragment, candidateDigest = digest) {
	try {
		validatePaperMethodLandscape(candidate, candidateDigest);
		failures += 1;
		console.error(`FAIL ${label}: accepted invalid artifact`);
	} catch (error) {
		ok(String(error).includes(fragment), `${label}: reports ${fragment}`);
	}
}

eq(digest, PAPER_METHOD_EXPECTED_SHA256, 'on-disk paper artifact retains frozen digest');
const landscape = validatePaperMethodLandscape(raw, digest);
eq(landscape.method_count, 59, 'all 59 published rows retained');
eq(
	landscape.family_counts,
	{ 'original belief': 3, 'random forest': 20, 'logistic regression': 20, SVC: 8, KNN: 8 },
	'method families reconcile'
);
eq(landscape.baseline.method, 'Belief Orig - readers', 'original belief anchor is explicit');
eq(landscape.baseline.fold_mean_trapezoidal_pr_auc, 0.917, 'original belief rounded anchor retained');
eq(landscape.best.method_id, 'paper_table_6:35', 'best tie resolves to lower fold SD');
eq(landscape.best.fold_mean_trapezoidal_pr_auc, 0.942, 'best published rounded anchor retained');
eq(landscape.metric_contract.uncertainty_is_confidence_interval, false, 'fold SD is not relabelled CI');
eq(landscape.metric_contract.metric_is_pooled_average_precision, false, 'paper metric is not pooled AP');
eq(landscape.metric_contract.directly_comparable_to_pair_error_f1, false, 'paper rows stay outside pair frontier');

const wrongDigest = '0'.repeat(64);
rejects(raw, 'wrong source digest', PAPER_METHOD_EXPECTED_SHA256, wrongDigest);

const falseDirect = structuredClone(raw);
falseDirect.metric_contract.directly_comparable_to_pair_error_f1 = true;
rejects(falseDirect, 'false direct comparison', 'must be false');

const falseCi = structuredClone(raw);
falseCi.metric_contract.uncertainty_is_confidence_interval = true;
rejects(falseCi, 'fold SD relabelled as CI', 'must be false');

const falsePooled = structuredClone(raw);
falsePooled.metric_contract.metric_is_pooled_average_precision = true;
rejects(falsePooled, 'fold trapezoid relabelled pooled AP', 'must be false');

const missingRow = structuredClone(raw);
missingRow.methods.pop();
rejects(missingRow, 'missing literature row', 'exactly 59 rows');

const mismatchedProjection = structuredClone(raw);
mismatchedProjection.tables[0].rows[0].fold_mean_trapezoidal_pr_auc = 0.5;
rejects(mismatchedProjection, 'table/method projection mismatch', 'exact ordered projection');

// ── dataset-slice discipline ────────────────────────────────────────────────
// The 59 rows span four INPUT CONFIGURATIONS. Comparing our all-sources-specific
// panel against a `readers` row is a category error, so the census is frozen and
// an unrecognised configuration must gate rather than silently bucket.
eq(landscape.slice_counts, PAPER_METHOD_SLICE_COUNTS, 'published rows partition into four configurations');
eq(
	Object.values(PAPER_METHOD_SLICE_COUNTS).reduce((a, b) => a + b, 0),
	59,
	'configuration census covers every published row'
);
eq(PAPER_METHOD_SLICES[0], PAPER_COMPARABLE_SLICE, 'the comparable configuration is listed first');
eq(landscape.comparable_best.method_id, 'paper_table_6:35', 'best COMPARABLE row resolves to lower fold SD');
eq(landscape.comparable_best.slice, PAPER_COMPARABLE_SLICE, 'best comparable row is in the comparable configuration');
eq(landscape.comparable_best.fold_mean_trapezoidal_pr_auc, 0.942, 'best comparable rounded anchor retained');
eq(landscape.comparable_best.fold_population_sd, 0.014, 'best comparable fold SD retained');
eq(
	landscape.comparable_best.base_method,
	'RF 2k-d13 + Type/#PMIDs/promoter',
	'configuration suffix is stripped off the rendered label'
);
ok(
	landscape.methods.every((method) => method.method === `${method.base_method} - ${method.slice}`),
	'every row round-trips base method + configuration'
);
// `best` spans configurations and `comparable_best` does not: they may coincide
// on today's artifact, but they are different questions and must stay separate.
ok('comparable_best' in landscape && 'best' in landscape, 'global best and comparable best are distinct fields');

const strangeSlice = structuredClone(raw);
strangeSlice.methods[0].method = 'RF 2k-d13 - readers, tentative';
strangeSlice.tables[0].rows[0].method = 'RF 2k-d13 - readers, tentative';
rejects(strangeSlice, 'unknown input configuration', 'unknown input configuration');

const noSlice = structuredClone(raw);
noSlice.methods[0].method = 'RF 2k-d13';
noSlice.tables[0].rows[0].method = 'RF 2k-d13';
rejects(noSlice, 'configuration segment removed', 'no " - <configuration>" segment');

const movedSlice = structuredClone(raw);
movedSlice.methods[30].method = 'Belief Orig - readers';
movedSlice.tables[0].rows[30].method = 'Belief Orig - readers';
rejects(movedSlice, 'configuration census drift', 'rows in this input configuration');

// ── the paper's own metric, with our arms on the same axis ──────────────────
const vsLlms = JSON.parse(
	readFileSync(new URL('../../data/results/indra_paper_literal_models_20260724/paper_literal_vs_llms.json', import.meta.url), 'utf8')
);
// Only the two fields this figure is allowed to read; anything else would be a
// different metric wearing the paper's ± .
const arms = PAPER_LITERAL_ARM_SPECS.map((spec) => ({
	id: spec.id,
	label: spec.label,
	display: spec.display,
	kind: spec.kind,
	trapezoidal: vsLlms.point_metrics[spec.label].fold_mean_trapezoidal_pr_auc,
	foldPopulationSd: vsLlms.point_metrics[spec.label].fold_population_sd
}));
const provenance = {
	nStatements: 1689,
	reproduction: {
		maxAbsDeltaVsPublishedTable6: 0.002,
		headlineLiteral: 0.942,
		headlinePublished: 0.942,
		paperCodeCommit: '63abdf1274d2f5534ed822585775031712916c83',
		cvProtocol: 'StratifiedKFold(10, shuffle=False)'
	}
};
const figure = buildPaperOwnMetric(landscape, arms, provenance);

eq(
	figure.bands.map((band) => band.id),
	['published-comparable', 'ours'],
	'two bands: the comparable published rows, then the models added here'
);
eq(
	figure.bands.map((band) => band.comparable),
	[true, true],
	'every drawn band is comparable — the non-comparable one is no longer drawn'
);
eq(
	figure.bands.map((band) => band.lanes.length),
	[15, 5],
	'15 comparable published rows and 5 scored models'
);
eq(figure.comparableCount, 15, 'comparable count matches the artifact census');
// THE THIRD BAND WAS REMOVED, THE COUNT WAS NOT. Those 44 rows used different
// evidence and were drawn as an uncompared range strip nobody could read; the
// count still matters, because it is what tells a reader the comparable 15 are a
// subset of 59 rather than the whole published table.
eq(figure.contextCount, 44, 'the count of published rows run on different evidence survives the band');
ok(
	figure.bands.every((band) => band.id !== 'published-context'),
	'no band draws the rows run on different evidence'
);
eq(figure.comparableBest.method_id, 'paper_table_6:35', 'the reference mark is their best COMPARABLE row');
ok(
	figure.bands.length === 2,
	'no third band'
);
ok(
	figure.bands[0].lanes.every((lane) => !lane.strip),
	'comparable lanes carry per-row dispersion, not a strip'
);

// Every arm lands exactly once: the three re-runs overlay their published row,
// the rest get their own lane. A dropped arm must never be a silent omission.
const armKeysOnAxis = figure.bands
	.flatMap((band) => band.lanes)
	.flatMap((lane) => lane.marks)
	.filter((mark) => mark.group === 'ours-reproduction' || mark.group === 'ours-scored')
	.map((mark) => mark.key);
eq([...armKeysOnAxis].sort(), arms.map((arm) => arm.id).sort(), 'every arm is placed exactly once');
eq(
	figure.bands[0].lanes.filter((lane) => lane.anchor).length,
	2,
	'both reproduced published rows are anchor lanes'
);
eq(Object.keys(PAPER_REPRODUCED_ROW_BY_ARM_ID).length, 3, 'three arms re-run a published row');
ok(
	Object.values(PAPER_REPRODUCED_ROW_BY_ARM_ID).every((name) =>
		landscape.methods.some((method) => method.method === name && method.slice === PAPER_COMPARABLE_SLICE)
	),
	'every reproduced row exists in the comparable configuration'
);

// The reproduction anchor is what licenses the axis; it must be legible and it
// must actually agree with the published row at the paper's printed precision.
eq(figure.anchor.publishedMean, 0.942, 'anchor prints their published fold mean');
eq(figure.anchor.publishedSd, 0.014, 'anchor prints their published fold SD');
eq(figure.anchor.ourMean.toFixed(3), '0.941', 'our re-run reproduces their row to three decimals');
eq(figure.anchor.portMean.toFixed(3), '0.942', 'our independent port reproduces it too');
ok(Math.abs(figure.anchor.ourMean - figure.anchor.publishedMean) <= 0.002, 're-run is within the manifest fidelity bound');

// Label geometry: right-anchored SVG text clips its LEADING glyphs silently.
const lanes = figure.bands.flatMap((band) => band.lanes);
ok(
	lanes.every((lane) => lane.display.length <= PAPER_OWN_METRIC_LABEL_BUDGET_CHARS),
	'every lane name fits the right-anchored gutter budget'
);
// The lane's on-screen name is `display`, never `label`: on this page `label` is
// always a frozen point_metrics join key, and the rule that keeps those off the
// screen is only checkable if nothing rendered is ever called `label`.
ok(
	lanes.every((lane) => lane.label === undefined),
	'no lane carries a field named `label` for a render site to reach for'
);
ok(
	lanes.every((lane) => lane.readout.length <= PAPER_OWN_METRIC_READOUT_BUDGET_CHARS),
	'every readout fits the right-hand gutter budget'
);
eq(
	Math.floor(PAPER_OWN_METRIC_GEOMETRY.labelAnchorX / PAPER_OWN_METRIC_GEOMETRY.monoUnitsPerChar),
	PAPER_OWN_METRIC_LABEL_BUDGET_CHARS,
	'label budget is derived from the gutter, not guessed'
);
const readoutUnitsPerChar =
	(PAPER_OWN_METRIC_GEOMETRY.monoUnitsPerChar * PAPER_OWN_METRIC_GEOMETRY.readoutFontPx) /
	PAPER_OWN_METRIC_GEOMETRY.labelFontPx;
eq(
	Math.floor((PAPER_OWN_METRIC_GEOMETRY.width - PAPER_OWN_METRIC_GEOMETRY.readoutX) / readoutUnitsPerChar),
	PAPER_OWN_METRIC_READOUT_BUDGET_CHARS,
	'readout budget is derived from its gutter too'
);

// Axis must contain every value the figure DRAWS: bar ends where dispersion is
// drawn, bare means on the strips.
for (const band of figure.bands) {
	for (const lane of band.lanes) {
		for (const mark of lane.marks) {
			const lo = lane.strip ? mark.foldMean : mark.foldMean - mark.foldSd;
			const hi = lane.strip ? mark.foldMean : mark.foldMean + mark.foldSd;
			ok(lo >= figure.domainMin && hi <= figure.domainMax, `axis contains ${mark.key}`);
		}
	}
}
ok(figure.ticks.length >= 3, 'axis carries readable ticks');
ok(figure.ticks[0] >= figure.domainMin && figure.ticks.at(-1) <= figure.domainMax, 'ticks stay inside the domain');

// Redundant encoding: one hue for two series is a defect, so every series needs
// its own (stroke, dash) pair AND its own mark shape.
const styles = PAPER_OWN_METRIC_GROUPS.map((group) => PAPER_OWN_METRIC_GROUP_STYLES[group]);
eq(new Set(styles.map((style) => `${style.strokeVar}|${style.dash}`)).size, styles.length, 'every series has its own stroke+dash pair');
eq(new Set(styles.map((style) => style.shape)).size, styles.length, 'every series has its own mark shape');
ok(styles.every((style) => style.strokeVar.startsWith('var(--')), 'series colours are layout tokens, never raw hex');

// THIS figure is on the paper's metric only. Average precision, AUROC, tie
// inflation and distinct-score counts belong to the figure BELOW it; leaking one
// in here would put a number next to a ± that does not belong to it.
const payload = JSON.stringify(figure);
for (const forbidden of ['pooled_average_precision', 'average precision', 'auroc', 'AUROC', 'distinct', 'inflation']) {
	eq(payload.includes(forbidden), false, `payload carries no ${forbidden}`);
}
eq(figure.uncertaintyIsConfidenceInterval, false, 'the ± is never promoted to a confidence interval');
eq(figure.metricIsPooledAveragePrecision, false, 'the metric is never relabelled pooled AP');
ok(figure.uncertaintyField.includes('population standard deviation'), 'the ± is named as a population SD');

// Fail-closed: a broken pairing or an over-budget label must throw, not draw.
function throwsWith(fn, label, fragment) {
	try {
		fn();
		failures += 1;
		console.error(`FAIL ${label}: accepted invalid input`);
	} catch (error) {
		ok(String(error).includes(fragment), `${label}: reports ${fragment}`);
	}
}
throwsWith(
	() => {
		const renamed = structuredClone(raw);
		renamed.methods[34].method = 'RF 2k-d13 + Type/#PMIDs/booster - all sources, specific';
		renamed.tables[0].rows[34].method = 'RF 2k-d13 + Type/#PMIDs/booster - all sources, specific';
		buildPaperOwnMetric(validatePaperMethodLandscape(renamed, digest), arms, provenance);
	},
	'reproduced row renamed out from under the pairing',
	'is missing'
);
throwsWith(
	() =>
		buildPaperOwnMetric(
			landscape,
			arms.map((arm) =>
				arm.id === 'glm-5' ? { ...arm, display: 'G'.repeat(PAPER_OWN_METRIC_LABEL_BUDGET_CHARS + 1) } : arm
			),
			provenance
		),
	'over-budget arm display name',
	'gutter budget is'
);
throwsWith(() => buildPaperOwnMetric(landscape, [], provenance), 'no arms', 'no arms to place');

// ── what these two figures must SAY, tested as claims and not as sentences ──
// These assertions used to be byte pins on exact sentences. Two ways that goes
// wrong, both of which this page has already paid for: a pinned sentence turns a
// legitimate re-wording into a red build, and `includes()` over raw source is
// satisfied by a CODE COMMENT that no reader will ever see. So each component is
// first reduced to the words that reach the screen, and the claim is then
// asserted in whatever wording carries it. NEGATIVE checks stay on the raw
// source, where they are the stricter of the two.
function readerText(source) {
	let body = source
		.replace(/<!--[\s\S]*?-->/g, ' ')
		.replace(/<script\b[^>]*>[\s\S]*?<\/script\s*>/gi, ' ')
		.replace(/<style\b[^>]*>[\s\S]*?<\/style\s*>/gi, ' ');
	// `{…}` is a value or a control-flow keyword, never a word on the page, and
	// dropping it FIRST stops a `<` inside an expression from eating the markup
	// that follows it.
	for (;;) {
		const next = body.replace(/\{[^{}]*\}/g, ' ');
		if (next === body) break;
		body = next;
	}
	return body.replace(/<[^>]+>/g, ' ').replace(/\s+/g, ' ');
}

/** A denial of the confidence-interval reading, in any wording. */
const NOT_A_CI = /\bnot\s+(a\s+)?confidence intervals?\b/i;
/** Any name for "how far the number moves" — the paper's own term or plain words. */
const SPREAD =
	/\b(sd|standard deviations?|deviations?|dispersion|spreads?|scatter|moves?|moved|moving|movement|varies|variation|swing|range)\b/i;
/** "over the 10 folds" / "across the ten slices": the count, and the thing counted. */
const OVER_TEN_SLICES = /\b(10|ten)\b[^.]{0,40}\b(slices?|folds?)\b/i;

/**
 * Every mention of a confidence interval has to be a DENIAL of one. Positional
 * rather than counted, so a second, affirming mention cannot hide behind the
 * first one's denial. Run over raw source: a comment that muses about intervals
 * is a future edit waiting to happen.
 */
function everyCiMentionIsADenial(source, label) {
	for (const match of source.matchAll(/confidence intervals?/gi)) {
		const before = source.slice(Math.max(0, match.index - 16), match.index);
		ok(/\bnot\s+(a\s+)?$/i.test(before), `${label}: "…${before}${match[0]}" reads as an affirmation`);
	}
}

/**
 * The ± must be characterised WHERE IT IS DENIED: within a window either side of
 * a "confidence interval", the figure has to say what the bar measures and over
 * how many re-scorings. Windowed rather than global so an unrelated caption
 * elsewhere on the figure cannot stand in for the explanation.
 */
function characterisedBesideTheDenial(text) {
	const WINDOW = 240;
	return [...text.matchAll(/confidence intervals?/gi)].some((match) => {
		const around = text.slice(Math.max(0, match.index - WINDOW), match.index + WINDOW);
		return SPREAD.test(around) && OVER_TEN_SLICES.test(around);
	});
}

const ownMetricSource = readFileSync(
	new URL('../src/lib/components/PaperOwnMetric.svelte', import.meta.url),
	'utf8'
);
const ownMetricText = readerText(ownMetricSource);

ok(NOT_A_CI.test(ownMetricText), 'UI states on screen that the ± is not a confidence interval');
everyCiMentionIsADenial(ownMetricSource, 'own-metric figure');
// WHICH spread, over HOW MANY re-scorings. The artifact's own name for the ±
// ("population standard deviation over the 10 folds") is sha-pinned, so it is
// checked structurally — the figure interpolates the field instead of retyping
// it — and the plain restatement beside it is checked as a claim: something
// moves, and there are ten of them.
ok(
	/\{figure\.uncertaintyField\}/.test(ownMetricSource),
	"the artifact's own name for the ± is rendered from the artifact, not retyped"
);
ok(
	characterisedBesideTheDenial(ownMetricText),
	'UI says what the ± measures, and over how many slices, beside the denial'
);
// The denial has to be INSIDE the drawing too: the figure is read, screenshotted
// and pasted on its own, without the prose that surrounds it here.
const ownMetricSvgs = [...ownMetricSource.matchAll(/<svg\b[\s\S]*?<\/svg\s*>/gi)].map((match) =>
	readerText(match[0])
);
ok(ownMetricSvgs.length > 0, 'the own-metric figure draws an SVG');
ok(
	ownMetricSvgs.some((svg) => NOT_A_CI.test(svg)),
	'the not-a-CI warning is inside the figure, not only in prose around it'
);
// The dataset-slice hazard, as the four facts it has to carry rather than as the
// sentence that carried them: the paper has FOUR input configurations, exactly
// ONE of them can be set beside our statements, that configuration is named from
// the artifact, and the two row counts that follow are read off the figure
// instead of typed. Any wording that carries all four passes.
ok(
	/\b(four|4)\b/.test(ownMetricText) && /\b(configurations?|ways?|inputs?|setups?|versions?)\b/i.test(ownMetricText),
	'UI says how many input configurations the paper has'
);
ok(
	/\b(only|just|exactly)\s+(one|1)\b/i.test(ownMetricText) ||
		/\bone\s+of\s+(the\s+|those\s+|these\s+|its\s+)?(four|4)\b/i.test(ownMetricText) ||
		/\ba\s+single\s+(one|configuration|way|input|setup|version)\b/i.test(ownMetricText),
	'UI says exactly one of them is comparable'
);
ok(
	/\{figure\.comparableSlice\}/.test(ownMetricSource),
	'UI names WHICH configuration ours is, from the artifact'
);
ok(
	/\{figure\.comparableCount\}/.test(ownMetricSource) && /\{figure\.contextCount\}/.test(ownMetricSource),
	'both row counts are read off the figure rather than typed into the prose'
);
ok(
	/\bnot\s+(shown|drawn)\b/i.test(ownMetricText),
	'UI says the rows run on different evidence are not shown'
);
// RE-ANCHORED 2026-07-29. This quoted one sentence verbatim, which made the
// guard a lock on WORDING rather than on the claim — and the wording it locked
// ("Their code, their data, our run — and their published row comes back.") was
// the us-vs-them phrasing this page was rewritten to drop: the 2023 paper is
// this lab's own prior work, so "theirs" named nothing and conceded something
// that was never in dispute. What has to stay legible is the CLAIM, which is
// unchanged: re-running the published code returns the published row. Any
// wording carrying the re-run, the published row and their agreement passes.
const REPRODUCTION_ANCHOR =
	/\bre-?run\b[^.]{0,120}\bpublished\s+(?:row|number)\b[^.]{0,40}\bcomes?\s+back\b/i;
ok(
	REPRODUCTION_ANCHOR.test(ownMetricText),
	'UI makes the reproduction anchor legible: the re-run returns the published row'
);
ok(
	ownMetricSource.includes('Do not read the top of this axis as a verdict'),
	'UI declines the victory claim and points at the tie figure below'
);
ok(!/\b(wins?|beats?|best in class|state of the art)\b/i.test(ownMetricSource), 'UI makes no victory claim');

const componentSource = readFileSync(
	new URL('../src/lib/components/PaperMethodLandscape.svelte', import.meta.url),
	'utf8'
);
ok(componentSource.includes('not another frontier'), 'UI names the literature layer as separate');
ok(
	componentSource.includes('Do not subtract these rounded values from the direct scores above.'),
	'UI blocks intuitive but invalid subtraction'
);
const landscapeText = readerText(componentSource);
ok(NOT_A_CI.test(landscapeText), 'UI states on screen that the bars are not confidence intervals');
everyCiMentionIsADenial(componentSource, 'literature landscape');
ok(
	characterisedBesideTheDenial(landscapeText),
	'UI says what the bars measure, and over how many slices, beside the denial'
);
ok(componentSource.includes('cannot enter paired deltas, parity claims, or the cost Pareto frontier'), 'UI excludes all invalid direct uses');
ok(componentSource.includes('Inspect all 59 published rows'), 'UI makes every source row inspectable');

if (failures) {
	console.error(`\n${failures} paper-method landscape assertion(s) failed`);
	process.exit(1);
}
console.log('paper-method landscape contract assertions passed');
