/**
 * THE COVERAGE **AND FIDELITY** GUARD FOR TRANSLATED ARTIFACT PROSE.
 *
 * Every banned word still reaching a /paper reader arrives at RUNTIME, off a
 * sha-pinned JSON file. No static scan of this repo can see those strings — which
 * is why three separate "clean" sweeps missed them, and why the fix is a plain
 * restatement authored at the LOADER, beside the parse that reads each field.
 *
 * A hand-kept list of which fields got one is not a guard: two review passes
 * enumerated the artifacts themselves and found whole blocks missing, including
 * the exact sentence the work existed to fix, in its third home. So this file
 * derives the coverage set MECHANICALLY, the same way a reviewer would:
 *
 *   1. walk every artifact a /paper loader reads and collect every string field
 *      longer than 40 characters — that is the coverage set, not a list someone
 *      remembered to update;
 *   2. run the real loader over the real artifact and collect every `{shipped,
 *      plain}` twin it emits;
 *   3. classify each string:
 *        · TWINNED            — its exact bytes are some twin's `shipped`;
 *        · IDENTIFIER         — a path, a digest, a dotted policy key, a code
 *                               reference, a frozen join key or a display name.
 *                               Decided by the SHAPE of the value, not by a list
 *                               of field names, so a new field cannot slip in;
 *        · NEVER READ         — the loader does not parse it, so it cannot reach
 *                               a reader on any surface;
 *        · UNTWINNED PROSE    — read, prose, and handed to a reader in the
 *                               artifact's own dialect. FAILS.
 *
 * The per-artifact counts are printed so the next reviewer can check the total
 * rather than re-derive it.
 *
 * COVERAGE IS NOT ENOUGH, and this file learned that the hard way. A translation
 * can be 100% complete and still say something its source forbids. The wave that
 * closed coverage shipped this twin:
 *
 *   SHIPPED  "… the reading gate is worth +0.0479 (Gemma 4 26B gate) — but that
 *             gate figure is measured against the WEAKEST member of the paper's
 *             family. … Quote the range; never quote the +0.0479 alone."
 *   PLAIN    "… and the reading step is worth +0.0479 (Gemma 4 26B)."
 *
 * The restatement deleted the correction and then did the one thing the
 * correction forbids, on a page whose whole purpose is to stop over-claiming.
 * So step 4 measures FIDELITY, over the same mechanically-derived twin set:
 *
 *        · every NUMBER in the shipped half must appear in the plain half;
 *        · a shipped half that DENIES or FORBIDS something (not / never / no /
 *          cannot …) must be denied or forbidden in the plain half too;
 *        · every NAMED MODEL in the shipped half must still be named.
 *
 * None of the three is a proxy for meaning. Each is a thing that cannot go
 * missing without the restatement asserting less than its source, and all three
 * independently catch the twin above. The class proves itself non-vacuous
 * against that exact pair before it is trusted.
 *
 * Run: node --experimental-strip-types scripts/test-paper-prose-coverage.mjs
 */
import { readFileSync } from 'node:fs';

import { validateApDecomposition } from '../src/lib/data/paper-ap-decomposition.ts';
import { validateBeliefLadder } from '../src/lib/data/paper-belief-ladder.ts';
import { validateDeployedBaseline } from '../src/lib/data/paper-deployed-baseline.ts';
import { validateStatementErrorF1 } from '../src/lib/data/paper-error-f1.ts';
import {
	validateFramingCorrection,
	validateNonReadingControl
} from '../src/lib/data/paper-framing-correction.ts';
import { validatePaperMethodLandscape } from '../src/lib/data/paper-method-landscape.ts';
import { validatePaperPerEvidence } from '../src/lib/data/paper-per-evidence.ts';
import { validateReviewQueue } from '../src/lib/data/paper-review-queue.ts';
import { validatePaperRobustness } from '../src/lib/data/paper-robustness.ts';
import { validatePaperTable6Extended } from '../src/lib/data/paper-table6-extended.ts';

let failures = 0;
function ok(condition, label) {
	if (!condition) {
		failures++;
		console.error(`FAIL ${label}`);
	}
}

const ROOT = new URL('../../data/', import.meta.url);
const read = (relative) => JSON.parse(readFileSync(new URL(relative, ROOT), 'utf8'));

const LITERAL = 'results/indra_paper_literal_models_20260724/';
const PUBLISHED_METRICS_SHA = '1c3af970a146dd8a1fdd5d35f43f366422cfdd7567a66bccfc932b2efde2f7d4';

/**
 * One entry per ARTIFACT a /paper loader reads, not per figure. `parse` returns
 * whatever the page actually holds: for the published-method metrics that is the
 * own-metric figure's fields, because the landscape parse is a pass-through and
 * the three metric_contract sentences are twinned where they are RENDERED.
 */
const ARTIFACTS = [
	['statement_review_queue.json', `${LITERAL}statement_review_queue.json`, validateReviewQueue],
	['statement_error_f1.json', `${LITERAL}statement_error_f1.json`, validateStatementErrorF1],
	['paper_margin_robustness.json', `${LITERAL}paper_margin_robustness.json`, validatePaperRobustness],
	['paper_table6_extended.json', `${LITERAL}paper_table6_extended.json`, validatePaperTable6Extended],
	['framing_correction.json', `${LITERAL}framing_correction.json`, validateFramingCorrection],
	['non_reading_control.json', `${LITERAL}non_reading_control.json`, validateNonReadingControl],
	['belief_model_ladder.json', `${LITERAL}belief_model_ladder.json`, validateBeliefLadder],
	[
		'ap_decomposition_by_paper_band.json',
		`${LITERAL}ap_decomposition_by_paper_band.json`,
		validateApDecomposition
	],
	[
		'deployed_baseline_replication.json',
		'results/deployed_baseline_replication_20260727/deployed_baseline_replication.json',
		validateDeployedBaseline
	],
	[
		'per_evidence_comparison.json',
		'results/per_evidence_comparison_20260727/per_evidence_comparison.json',
		validatePaperPerEvidence
	],
	[
		'indra_paper_2023_published_method_metrics.json',
		'benchmark/indra_paper_2023_published_method_metrics.json',
		(raw) => {
			const landscape = validatePaperMethodLandscape(raw, PUBLISHED_METRICS_SHA);
			// The own-metric figure renders all three of these in one sentence, so the
			// twins live there; this mirrors that surface rather than the parse.
			return {
				landscape,
				rendered: {
					perFoldMetricProse: {
						shipped: landscape.metric_contract.per_fold_metric,
						plain: OWN_METRIC_RENDERED.perFoldMetric
					},
					foldSummaryProse: {
						shipped: landscape.metric_contract.summary,
						plain: OWN_METRIC_RENDERED.foldSummary
					},
					uncertaintyFieldProse: {
						shipped: landscape.metric_contract.uncertainty_field,
						plain: OWN_METRIC_RENDERED.uncertaintyField
					}
				}
			};
		}
	]
];

/**
 * The own-metric figure's three restatements, quoted here so this guard proves
 * the metric_contract sentences are covered SOMEWHERE a reader meets them. Kept
 * short and checked non-empty; the wording itself is asserted by (a4).
 */
const OWN_METRIC_RENDERED = {
	perFoldMetric: 'within each slice, precision_recall_curve then the area under it',
	foldSummary: 'the plain average of those 10 slice results',
	uncertaintyField: 'spread of the 10 slice results around their own average'
};

/** Below this length a string is a token, a number or a short label, not prose. */
const PROSE_MIN_CHARS = 40;

function walkArtifact(node, path, out) {
	if (node && typeof node === 'object') {
		if (Array.isArray(node)) node.forEach((value, index) => walkArtifact(value, `${path}[${index}]`, out));
		else for (const [key, value] of Object.entries(node)) walkArtifact(value, path ? `${path}.${key}` : key, out);
	} else if (typeof node === 'string' && node.length > PROSE_MIN_CHARS) {
		out.push([path, node]);
	}
}

/** Every `{shipped, plain}` pair the loader emits, by its shipped bytes. */
function collectTwins(node, into, seen = new Set()) {
	if (!node || typeof node !== 'object' || seen.has(node)) return into;
	seen.add(node);
	if (typeof node.shipped === 'string' && typeof node.plain === 'string') {
		ok(node.plain.length > 0, `a twin for ${JSON.stringify(node.shipped.slice(0, 40))} has an empty plain half`);
		into.add(node.shipped);
	}
	if (Array.isArray(node)) node.forEach((value) => collectTwins(value, into, seen));
	else for (const value of Object.values(node)) collectTwins(value, into, seen);
	return into;
}

/** The same walk, keeping BOTH halves and where they were found. */
function collectTwinPairs(node, path, out, seen = new Set()) {
	if (!node || typeof node !== 'object' || seen.has(node)) return out;
	seen.add(node);
	if (typeof node.shipped === 'string' && typeof node.plain === 'string') {
		out.push({ path, shipped: node.shipped, plain: node.plain });
		return out;
	}
	if (Array.isArray(node)) node.forEach((value, index) => collectTwinPairs(value, `${path}[${index}]`, out, seen));
	else {
		for (const [key, value] of Object.entries(node)) {
			collectTwinPairs(value, path ? `${path}.${key}` : key, out, seen);
		}
	}
	return out;
}

/** Every string the loader passes through at all — read vs never read. */
function collectStrings(node, into, seen = new Set()) {
	if (typeof node === 'string') {
		into.add(node);
		return into;
	}
	if (!node || typeof node !== 'object' || seen.has(node)) return into;
	seen.add(node);
	if (Array.isArray(node)) node.forEach((value) => collectStrings(value, into, seen));
	else for (const value of Object.values(node)) collectStrings(value, into, seen);
	return into;
}

const HEX64 = /^[0-9a-f]{64}$/;
/**
 * Frozen leaves. `display` and `label` are here because class (a) already governs
 * them — a join key must never reach a reader, and the fix for one that does is
 * the display name beside the key, never a rewrite of the key.
 */
const IDENTIFIER_LEAVES = new Set([
	'method',
	'paper_method',
	'method_string',
	'key_in_source',
	'display',
	'label',
	'label_field',
	'field',
	'ships_in',
	'join',
	'artifact_kind',
	'best_fitted_method',
	'max_abs_dev_row',
	'notebook_path',
	'scores_key',
	'recorded_key'
]);

/**
 * An IDENTIFIER rather than prose, decided by the SHAPE of the value first. A
 * name-list alone was tried and is how a whole block goes missing: anything
 * unlisted silently counts as covered. Here the default is the other way up —
 * a string with a space in it is PROSE unless it is shown to be something else.
 */
function isIdentifier(path, value) {
	if (HEX64.test(value)) return true;
	if (!/\s/.test(value)) return true;
	if (/^https?:\/\//.test(value)) return true;
	const leaf = path.split('.').pop().replace(/\[\*\]/g, '');
	if (IDENTIFIER_LEAVES.has(leaf)) return true;
	// A provenance block addresses files and runs; its own `note` is still prose.
	if (/(^|\.)provenance(\.|$)/.test(path) && !leaf.endsWith('note')) return true;
	return false;
}

// ---------------------------------------------------------------------------
// FIDELITY. Does the plain half assert everything the shipped half asserts,
// deny everything it denies, and forbid everything it forbids?
// ---------------------------------------------------------------------------

/**
 * A number as a reader meets it: not glued to letters, so a commit digest
 * (`63abdf12`) and a module version inside an identifier are not numbers, while
 * `1.24.0`, `+0.0479`, `73.2%`, `0.000e+00` and `1,689` all are. The sign is
 * deliberately NOT captured: a restatement is free to write "+0.0479" as
 * "0.0479" in a sentence that already says which way it points, and the
 * direction itself is guarded by the standing types in the loaders.
 */
const NUMBER_TOKEN = /(?<![A-Za-z0-9_.])\d[\d,]*(?:\.\d+)*(?![A-Za-z0-9_])/g;

/**
 * Small integers a restatement may legitimately spell out — "1 - PROD_s …"
 * becomes "one minus the product …", which loses no information at all.
 */
const NUMBER_WORDS = [
	'zero',
	'one',
	'two',
	'three',
	'four',
	'five',
	'six',
	'seven',
	'eight',
	'nine',
	'ten',
	'eleven',
	'twelve'
];

/** Thousands separators are formatting, not information. */
const normaliseNumbers = (text) => text.replace(/,/g, '');

function missingNumbers(shipped, plain) {
	const haystack = normaliseNumbers(plain);
	const missing = [];
	for (const raw of shipped.match(NUMBER_TOKEN) ?? []) {
		const token = normaliseNumbers(raw);
		if (haystack.includes(token)) continue;
		const asWord = NUMBER_WORDS[Number(token)];
		if (asWord !== undefined && new RegExp(`\\b${asWord}\\b`, 'i').test(plain)) continue;
		missing.push(token);
	}
	return [...new Set(missing)];
}

/**
 * A denial or a prohibition. Presence-based on purpose: counting them would
 * fail on "not … nor …" restated as "neither … nor …", while a shipped half
 * that denies something and a plain half that denies nothing is unambiguous —
 * it is the shape of the defect this class exists for.
 */
const DENIAL =
	/\b(?:not|never|no|none|nor|neither|nothing|nobody|cannot|can't|isn't|doesn't|don't|without|only|must)\b/i;

/**
 * MODELS BY NAME. A restatement may rewrite "arm" as "model" and "RF" as
 * "random forest"; it may not stop saying WHICH model, because the whole point
 * of several of these sentences is that two numbers belong to two different
 * ones. Each entry is (pattern in the shipped half, what the plain half must
 * still say). Every entry is proved to match at least one real shipped string
 * below, so a stale name here cannot make the class quietly narrower.
 */
const NAMED_MODELS = [
	{ name: 'Gemma 4 26B', shipped: /\bGemma 4 26B\b/, plain: /\bGemma 4 26B\b/ },
	{ name: 'Gemma 4 31B', shipped: /\bGemma 4 31B\b/, plain: /\bGemma 4 31B\b/ },
	{ name: 'Gemma 4 E2B', shipped: /\bGemma 4 E2B\b/, plain: /\bGemma 4 E2B\b/ },
	{ name: 'GLM-5', shipped: /\bGLM-5\b/, plain: /\bGLM-5\b/ },
	{ name: 'SimpleScorer', shipped: /\bSimpleScorer\b/, plain: /\bSimpleScorer\b/ },
	{ name: 'BayesianScorer', shipped: /\bBayesianScorer\b/, plain: /\bBayesianScorer\b/ },
	{ name: 'HybridScorer', shipped: /\bHybridScorer\b/, plain: /\bHybridScorer\b/ },
	{ name: 'CountsScorer', shipped: /\bCountsScorer\b/, plain: /\bCountsScorer\b/ },
	{ name: 'BeliefEngine', shipped: /\bBeliefEngine\b/, plain: /\bBeliefEngine\b/ },
	{ name: 'Belief Orig', shipped: /\bBelief Orig\b/, plain: /\bBelief Orig\b/ },
	{ name: 'INDRA CoGEx hybrid', shipped: /\bINDRA CoGEx hybrid\b/, plain: /\bINDRA CoGEx hybrid\b/ },
	{ name: 'RF 2k-d13', shipped: /\bRF 2k-d13\b/, plain: /\bRF 2k-d13\b/ }
];

function missingModels(shipped, plain) {
	return NAMED_MODELS.filter((model) => model.shipped.test(shipped) && !model.plain.test(plain)).map(
		(model) => model.name
	);
}

/** Every way one twin can assert less than its source. */
function fidelityOffences({ shipped, plain }) {
	const offences = [];
	const numbers = missingNumbers(shipped, plain);
	if (numbers.length > 0) offences.push(`drops the number(s) ${numbers.join(', ')}`);
	if (DENIAL.test(shipped) && !DENIAL.test(plain)) {
		offences.push('the shipped half denies or forbids something and the plain half does not');
	}
	const models = missingModels(shipped, plain);
	if (models.length > 0) offences.push(`stops naming ${models.join(', ')}`);
	return offences;
}

let totalStrings = 0;
let totalTwinned = 0;
let totalUntwinned = 0;
let totalPairs = 0;
let totalFidelityOffences = 0;
const shippedCorpus = [];

console.log(`/paper artifact prose coverage (every string field longer than ${PROSE_MIN_CHARS} chars)\n`);
for (const [name, relative, parse] of ARTIFACTS) {
	const raw = read(relative);
	const parsed = parse(raw);
	const twins = collectTwins(parsed, new Set());
	const readThrough = collectStrings(parsed, new Set());

	const found = [];
	walkArtifact(raw, '', found);
	const byPath = new Map();
	for (const [path, value] of found) {
		const normalised = path.replace(/\[\d+\]/g, '[*]');
		if (!byPath.has(normalised)) byPath.set(normalised, []);
		byPath.get(normalised).push(value);
	}

	let twinned = 0;
	let identifiers = 0;
	let neverRead = 0;
	const untwinned = [];
	for (const [path, values] of byPath) {
		if (values.every((value) => twins.has(value))) {
			twinned += values.length;
		} else if (values.every((value) => !readThrough.has(value))) {
			neverRead += values.length;
		} else if (values.every((value) => isIdentifier(path, value))) {
			identifiers += values.length;
		} else {
			untwinned.push([path, values.length, values[0]]);
		}
	}

	totalStrings += found.length;
	totalTwinned += twinned;
	totalUntwinned += untwinned.reduce((sum, [, count]) => sum + count, 0);

	console.log(
		`  ${String(found.length).padStart(4)} strings  ` +
			`${String(twinned).padStart(3)} twinned  ` +
			`${String(identifiers).padStart(3)} identifier  ` +
			`${String(neverRead).padStart(3)} never read  ` +
			`${String(untwinned.length).padStart(2)} UNTWINNED   ${name}`
	);
	for (const [path, count, sample] of untwinned) {
		console.error(`      ${path} [${count}]\n        ${JSON.stringify(sample.slice(0, 110))}`);
	}
	ok(untwinned.length === 0, `${name}: shipped prose reaches a reader with no plain restatement`);

	// FIDELITY, over the twins this same artifact's loader emitted.
	const pairs = collectTwinPairs(parsed, '', []);
	totalPairs += pairs.length;
	for (const pair of pairs) shippedCorpus.push(pair.shipped);
	const failing = pairs
		.map((pair) => ({ pair, offences: fidelityOffences(pair) }))
		.filter((entry) => entry.offences.length > 0);
	totalFidelityOffences += failing.length;
	for (const { pair, offences } of failing) {
		console.error(
			`      FIDELITY ${name} :: ${pair.path}\n` +
				`        ${offences.join('; ')}\n` +
				`        shipped: ${JSON.stringify(pair.shipped.slice(0, 160))}\n` +
				`        plain:   ${JSON.stringify(pair.plain.slice(0, 160))}`
		);
	}
	ok(failing.length === 0, `${name}: a plain restatement asserts less than the sentence it restates`);
}

console.log(
	`\n  ${totalStrings} strings over ${PROSE_MIN_CHARS} chars across ${ARTIFACTS.length} artifacts; ` +
		`${totalTwinned} carry a plain twin; ${totalUntwinned} untwinned prose fields remain`
);
console.log(
	`  ${totalPairs} twins checked for fidelity; ${totalFidelityOffences} restate less than their source`
);

// The guard must not be vacuous: two guards on this project shipped GREEN while
// checking nothing. Prove the classifier reaches prose and rejects it.
{
	const before = failures;
	ok(!isIdentifier('foo.note', 'a sentence with several words in it that is prose'), 'prose is not an identifier');
	ok(isIdentifier('foo.path', 'data/results/thing.jsonl'), 'a path is an identifier');
	ok(isIdentifier('foo.sha256', 'a'.repeat(64)), 'a digest is an identifier');
	ok(isIdentifier('provenance.gold', 'data/x y/z.jsonl'), 'a provenance address is an identifier');
	ok(!isIdentifier('provenance.note', 'a provenance note is still a sentence someone reads'), 'a provenance note is prose');
	const probe = [];
	walkArtifact({ a: { b: 'x'.repeat(PROSE_MIN_CHARS + 1) }, c: 'short' }, '', probe);
	ok(probe.length === 1 && probe[0][0] === 'a.b', 'the walk finds nested prose and skips short strings');
	ok(collectTwins({ x: { shipped: 'S', plain: 'P' } }, new Set()).has('S'), 'twin collection reaches a nested twin');
	ok(
		collectTwinPairs({ x: [{ shipped: 'S', plain: 'P' }] }, '', []).some(
			(pair) => pair.path === 'x[0]' && pair.shipped === 'S'
		),
		'twin-pair collection reaches a nested twin and records where it is'
	);
	ok(failures === before, 'the self-test itself passed');
}

// ---------------------------------------------------------------------------
// THE FIDELITY CLASS IS NOT VACUOUS. Its whole justification is a twin that
// shipped GREEN through a coverage-only guard, so it is proved against exactly
// that pair — and against a planted loss of each kind — before it is trusted.
// ---------------------------------------------------------------------------
{
	const before = failures;

	// The real regression, verbatim on both sides.
	const REGRESSION = {
		shipped:
			"From that baseline INDRA's full CountsScorer feature set — a superset of the paper's " +
			'statement-type / #PMIDs / promoter panel — is worth +0.0390 and the reading gate is worth ' +
			"+0.0479 (Gemma 4 26B gate) — but that gate figure is measured against the WEAKEST member " +
			"of the paper's family. Against the paper's best noisy-OR variant (BayesianScorer, " +
			"source+subtype refit) the same gate is +0.0331; against the paper's best model overall " +
			'(RF with full features, 0.9422 re-implemented / 0.9412 literal) it is +0.0088 to +0.0098. ' +
			'Quote the range; never quote the +0.0479 alone.',
		plain:
			'From that baseline, INDRA’s full CountsScorer feature set — a superset of the ' +
			'statement-type / #PMIDs / promoter features engineered in the 2023 paper — is worth ' +
			'+0.0390, and the reading step is worth +0.0479 (Gemma 4 26B).'
	};
	const caught = fidelityOffences(REGRESSION);
	ok(caught.length > 0, 'the fidelity class catches the restatement that deleted its own correction');
	ok(
		caught.some((offence) => offence.includes('0.0331')),
		'it names the dropped numbers, so the report says what to put back'
	);
	ok(
		caught.some((offence) => offence.includes('denies or forbids')),
		'it catches the deleted prohibition — "never quote the +0.0479 alone"'
	);
	ok(
		caught.some((offence) => offence.includes('BayesianScorer')),
		'it catches the model the correction names going missing'
	);

	// One planted loss of each kind, over a pair that is otherwise faithful.
	ok(
		fidelityOffences({ shipped: 'the margin is +0.0231', plain: 'the margin is positive' }).length === 1,
		'a dropped number alone is caught'
	);
	ok(
		fidelityOffences({ shipped: 'it is NOT their best model', plain: 'it is their weakest model' })
			.length === 1,
		'a dropped denial alone is caught'
	);
	ok(
		fidelityOffences({
			shipped: 'BayesianScorer leads on that measure',
			plain: 'our best model leads on that measure'
		}).length === 1,
		'a dropped model name alone is caught'
	);

	// …and it does not fire on the translations it exists to permit.
	ok(
		fidelityOffences({
			shipped: 'belief = 1 - PROD_s (syst_s + rand_s^{n_s})',
			plain: 'one minus the product, over every source, of that source’s systematic error rate'
		}).length === 0,
		'spelling a small integer out in words is not a loss'
	);
	ok(
		fidelityOffences({
			shipped: 'on the SAME 1689 all-source curated statements',
			plain: 'on the SAME 1,689 curated statements, drawn from every source'
		}).length === 0,
		'a thousands separator is formatting, not information'
	);
	ok(
		fidelityOffences({
			shipped: "the paper's RF, scored out of fold",
			plain: 'the random forest, scored on the slice it never learned from'
		}).length === 0,
		'rewriting RF as “random forest” is a translation, not a loss'
	);

	// Every named-model pattern must actually occur in the shipped corpus. A name
	// that matches nothing is a rule that can never fire, which is how a guard
	// goes green while checking less than it claims.
	for (const model of NAMED_MODELS) {
		ok(
			shippedCorpus.some((shipped) => model.shipped.test(shipped)),
			`the named-model rule for "${model.name}" matches at least one shipped sentence`
		);
	}
	ok(failures === before, 'the fidelity self-test itself passed');
	console.log(
		`\n  proof: the fidelity class catches the shipped regression on ${caught.length} independent ` +
			`counts, and each of its ${NAMED_MODELS.length} model names occurs in the real corpus`
	);
}

if (failures > 0) {
	console.error(`\n${failures} assertion(s) failed`);
	process.exit(1);
}
console.log(
	'paper prose: every read prose field carries a plain restatement, and every restatement ' +
		'carries its source’s numbers, denials and named models'
);
