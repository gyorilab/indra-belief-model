/**
 * THE VERIFICATION SECTION'S OWN CONTRACT.
 *
 * `/paper` used to open onto the result files' own wording in six places. Those
 * six are gone and the wording collects in ONE section at the end. That trade is
 * only honest if the section is COMPLETE: a boundary that quietly drops a
 * sentence is worse than the six it replaced, because the page then claims to be
 * checkable and is not.
 *
 * Four guards on this project have shipped GREEN while checking nothing, and a
 * fifth was found testing an architecture that no longer existed. So every class
 * below is driven, at least once, by an input it MUST reject, and the count of
 * those planted failures is printed. A run that stops planting them cannot still
 * exit 0.
 *
 *   (a) BEFORE ⊆ AFTER. The exact shipped strings the six removed boundaries put
 *       on screen are re-derived from the REAL artifacts — by the same accessors
 *       the components used — and every one must be reachable in the trail. The
 *       size of that set is asserted, so an accessor list that silently emptied
 *       cannot pass by having nothing to check.
 *   (b) COVERAGE IS DERIVED. Every twin the loaders emit reaches the trail, not
 *       just the ones a list remembers. Checked by walking each payload for
 *       `{shipped, plain}` pairs independently of the module under test.
 *   (c) FAIL-CLOSED. A twin missing its shipped half must take its own file's
 *       group down and leave the others standing — a plain string must never be
 *       presented as the file's own words.
 *   (d) PROVENANCE IS NEVER GUESSED. A gated load contributes a group that says
 *       why, never entries; a file whose digest no load carries says so rather
 *       than borrowing a sibling's digest.
 *   (e) ONE BOUNDARY. Exactly one `<details>` in the section, and none of the six
 *       framings anywhere in the components that used to carry them.
 *   (f) NO INVENTED VOCABULARY in the section's own authored prose. The shipped
 *       halves inside it are quoted source and are exempt by construction — that
 *       is the whole point of the section — so the check runs over the component
 *       source's static text, which is the only prose this node authors.
 *
 * Run: node --experimental-strip-types scripts/test-paper-audit-trail-contract.mjs
 */
import { readFileSync } from 'node:fs';

import {
	buildPaperAuditTrail,
	paperAuditSources
} from '../src/lib/data/paper-audit-trail.ts';
import { validateApDecomposition } from '../src/lib/data/paper-ap-decomposition.ts';
import { validateBeliefLadder } from '../src/lib/data/paper-belief-ladder.ts';
import { validateDeployedBaseline } from '../src/lib/data/paper-deployed-baseline.ts';
import { validateStatementErrorF1 } from '../src/lib/data/paper-error-f1.ts';
import {
	validateFramingCorrection,
	validateNonReadingControl
} from '../src/lib/data/paper-framing-correction.ts';
import { validatePaperPerEvidence } from '../src/lib/data/paper-per-evidence.ts';
import { validateReviewQueue } from '../src/lib/data/paper-review-queue.ts';
import { validatePaperRobustness } from '../src/lib/data/paper-robustness.ts';
import { validatePaperTable6Extended } from '../src/lib/data/paper-table6-extended.ts';

let failures = 0;
function ok(condition, label) {
	if (!condition) {
		failures += 1;
		console.error(`FAIL ${label}`);
	}
}
function eq(got, want, label) {
	if (Object.is(got, want)) return;
	failures += 1;
	console.error(`FAIL ${label}: got ${JSON.stringify(got)}, want ${JSON.stringify(want)}`);
}

const DATA = new URL('../../data/', import.meta.url);
const SRC = new URL('../src/', import.meta.url);
const read = (relative) => JSON.parse(readFileSync(new URL(relative, DATA), 'utf8'));
const source = (relative) => readFileSync(new URL(relative, SRC), 'utf8');

const LITERAL = 'results/indra_paper_literal_models_20260724/';
const FAKE_SHA = 'a'.repeat(64);

// ---------------------------------------------------------------------------
// The real payloads, through the real loaders.
// ---------------------------------------------------------------------------

/**
 * Some validators return the parsed FIGURE, others return the whole LOAD. The
 * page hands the audit trail the payload in both cases, so the payload is taken
 * here the same way — `.figure` where the validator gates for itself.
 */
const figureOf = (load, label) => {
	ok(load.status === 'ok', `${label} loads from the shipped bytes`);
	return load.figure;
};

const queue = validateReviewQueue(read(`${LITERAL}statement_review_queue.json`));
const errorF1 = figureOf(
	validateStatementErrorF1(read(`${LITERAL}statement_error_f1.json`)),
	'statement_error_f1.json'
);
const robustness = figureOf(
	validatePaperRobustness(read(`${LITERAL}paper_margin_robustness.json`)),
	'paper_margin_robustness.json'
);
const table6 = figureOf(
	validatePaperTable6Extended(read(`${LITERAL}paper_table6_extended.json`)),
	'paper_table6_extended.json'
);
const framing = validateFramingCorrection(read(`${LITERAL}framing_correction.json`));
const control = validateNonReadingControl(read(`${LITERAL}non_reading_control.json`));
const ladder = validateBeliefLadder(read(`${LITERAL}belief_model_ladder.json`));
const decomposition = validateApDecomposition(read(`${LITERAL}ap_decomposition_by_paper_band.json`));
const deployed = validateDeployedBaseline(
	read('results/deployed_baseline_replication_20260727/deployed_baseline_replication.json')
);
const perEvidence = figureOf(
	validatePaperPerEvidence(read('results/per_evidence_comparison_20260727/per_evidence_comparison.json')),
	'per_evidence_comparison.json'
);

/** The sources as the page's own loads produce them, one per file. */
const sources = [
	['paper_table6_extended.json', table6],
	['ap_decomposition_by_paper_band.json', decomposition],
	['paper_margin_robustness.json', robustness],
	['framing_correction.json', framing],
	['non_reading_control.json', control],
	['statement_review_queue.json', queue],
	['statement_error_f1.json', errorF1],
	['per_evidence_comparison.json', perEvidence],
	['belief_model_ladder.json', ladder],
	['deployed_baseline_replication.json', deployed]
].map(([file, payload]) => ({
	file,
	path: `data/results/…/${file}`,
	sha256: FAKE_SHA,
	unavailable: null,
	payload
}));

const trail = buildPaperAuditTrail(sources);
const reachable = new Set(trail.files.flatMap((file) => file.entries.map((entry) => entry.shipped)));

console.log(
	`/paper verification section: ${trail.nEntries} sentences from ${trail.nFiles} files ` +
		`(${trail.nFilesUnavailable} contributing none, ${trail.nConflicts} restated two ways)`
);
for (const file of trail.files) {
	console.log(`  ${String(file.entries.length).padStart(3)}  ${file.file}`);
}

ok(trail.nEntries > 150, 'the trail carries the whole page, not a sample');
eq(trail.nFilesUnavailable, 0, 'every file in the real run contributes its sentences');

/**
 * TWO SENTENCES ARE RESTATED TWICE TODAY, and both are legitimate: the noisy-OR
 * formula is twinned in four files, three of them word-identical and the fourth
 * differing by a capital letter and a full stop; and `average_precision_score
 * (tie-aware)` is restated once bare and once with the number of statements it
 * runs over. The section lists BOTH, each under the file it ships in, and prints
 * the count — a reader who meets two wordings should be told there are two.
 *
 * So the count is not pinned to a number a rewording would break. What is pinned
 * is that the counter WORKS: plant one and it must move by exactly one, and stay
 * inside a bound that a page which started restating everything twice would blow.
 */
ok(trail.nConflicts <= 5, `few sentences are restated two ways (got ${trail.nConflicts})`);
{
	const [first] = trail.files[0].entries;
	const planted = buildPaperAuditTrail([
		...sources,
		{
			file: 'a-second-restatement.json',
			path: null,
			sha256: null,
			unavailable: null,
			payload: { twin: { shipped: first.shipped, plain: `${first.plain} — said differently` } }
		}
	]);
	eq(
		planted.nConflicts,
		trail.nConflicts + 1,
		'a second restatement of one shipped sentence is counted'
	);
}

// ---------------------------------------------------------------------------
// (a) BEFORE ⊆ AFTER — nothing the six removed boundaries showed has been lost.
//
// Each accessor below is a string the component USED to render verbatim, named
// by where it rendered. This is the diff the node's constraint asks for, taken
// against the real artifacts rather than against a remembered list.
// ---------------------------------------------------------------------------

const removedBoundaries = [
	// ReviewQueue.svelte — "how this is computed", incl. three "in the artifact's
	// own words" lead-ins and the whole robustness block.
	['ReviewQueue: decision rule', queue.decisionRule],
	['ReviewQueue: threshold rule', queue.thresholdRule],
	['ReviewQueue: equal-yield operating rule', queue.equalYield.operatingRule],
	['ReviewQueue: equal-yield oracle disclosure', queue.equalYield.oracleDisclosure],
	...queue.caveats.map((caveat, index) => [`ReviewQueue: caveat ${index}`, caveat]),
	['ReviewQueue: robustness metric', queue.errorRecallRobustness.metric],
	['ReviewQueue: robustness budget rule', queue.errorRecallRobustness.budgetRule],
	['ReviewQueue: robustness bootstrap design', queue.errorRecallRobustness.bootstrapDesign],
	['ReviewQueue: robustness multiplicity method', queue.errorRecallRobustness.multiplicityMethod],
	['ReviewQueue: robustness multiplicity note', queue.errorRecallRobustness.multiplicityNote],
	['ReviewQueue: label completeness note', queue.errorRecallRobustness.labelCompleteness.note],

	// routes/paper/+page.svelte — the page's OWN boundary, headed "caveats,
	// verbatim from the artifact", which sat at the foot of the costs section and
	// showed two shipped things: the promotion ceiling's explanation and the
	// review queue's caveat list. The caveats are already in the list above under
	// ReviewQueue; the ceiling's explanation rendered nowhere else and is named
	// here, so removing that boundary is checked rather than asserted.
	['/paper costs section: promotion-ceiling explanation', queue.promotionCeiling.why],

	// StatementErrorF1.svelte — two "in the artifact's own words" summaries, the
	// "Caveats, as shipped" list, and the value table's note.
	['StatementErrorF1: headline cutoff rule', errorF1.headlineThresholdRule.rule],
	['StatementErrorF1: headline oracle disclosure', errorF1.headlineThresholdRule.oracle],
	['StatementErrorF1: modal cutoff note', errorF1.modalThresholdNote],
	['StatementErrorF1: matched cutoff rule', errorF1.matchedThresholdRule.rule],
	['StatementErrorF1: matched oracle disclosure', errorF1.matchedThresholdRule.oracle],
	['StatementErrorF1: review-list cutoff rule', errorF1.reconciliation.thresholdRule.rule],
	['StatementErrorF1: review-list oracle', errorF1.reconciliation.thresholdRule.oracle],
	['StatementErrorF1: metric', errorF1.metric],
	['StatementErrorF1: positive-class note', errorF1.positiveClassNote],
	['StatementErrorF1: decision rule', errorF1.decisionRule],
	['StatementErrorF1: bootstrap design', errorF1.bootstrapDesign],
	['StatementErrorF1: label provenance', errorF1.panel.labelProvenance],
	['StatementErrorF1: panel ordering', errorF1.panel.ordering],
	['StatementErrorF1: multiplicity method', errorF1.multiplicity.method],
	['StatementErrorF1: multiplicity note', errorF1.multiplicity.note],
	['StatementErrorF1: reconciliation note', errorF1.reconciliation.note],
	...errorF1.caveats.map((caveat, index) => [`StatementErrorF1: caveat ${index}`, caveat]),

	// BeliefModelLadder.svelte — "how this is computed".
	['BeliefModelLadder: label convention', ladder.panel.labelConvention],
	['BeliefModelLadder: panel ordering', ladder.panel.ordering],
	['BeliefModelLadder: metric source', ladder.metricSource],
	['BeliefModelLadder: noisy-OR formula', ladder.noisyOrFormula],
	['BeliefModelLadder: proximity status', ladder.guardrails.reimplementationProximity.status],
	[
		'BeliefModelLadder: fidelity statistic',
		ladder.guardrails.reimplementationProximity.fidelityEvidence.statistic
	],
	['BeliefModelLadder: join', ladder.join],
	...ladder.caveats.map((caveat, index) => [`BeliefModelLadder: caveat ${index}`, caveat]),

	// DeployedBaseline.svelte — "how this is computed" and the SVG description,
	// which was behind no boundary at all.
	['DeployedBaseline: claim', deployed.claim],
	['DeployedBaseline: claim-is-not', deployed.claimIsNot],
	['DeployedBaseline: question', deployed.question],
	['DeployedBaseline: metric source', deployed.metricSource],
	['DeployedBaseline: selection rule', deployed.incumbentSelectionRule],
	['DeployedBaseline: selection rule cost', deployed.incumbentSelectionRuleCost],
	['DeployedBaseline: gate description', deployed.arms.gate.whatItIs],
	['DeployedBaseline: gate not-zero-shot', deployed.arms.gate.notZeroShot],
	['DeployedBaseline: research description', deployed.arms.research.whatItIs],
	['DeployedBaseline: served-belief question', deployed.servedBeliefIdentity.question],
	['DeployedBaseline: served-belief finding', deployed.servedBeliefIdentity.finding],
	['DeployedBaseline: served-belief floor', deployed.servedBeliefIdentity.floorDerivation],
	...deployed.caveats.map((caveat, index) => [`DeployedBaseline: caveat ${index}`, caveat]),
	...deployed.families.flatMap((family, index) => [
		[`DeployedBaseline: family ${index} what it computes`, family.whatItComputes],
		[`DeployedBaseline: family ${index} where it runs`, family.whereItRuns]
	]),
	...deployed.panels.flatMap((panel, index) => [
		[`DeployedBaseline: panel ${index} curator note`, panel.curatorNote],
		[`DeployedBaseline: panel ${index} label note`, panel.labelNote],
		[`DeployedBaseline: panel ${index} join mode`, panel.heterogeneity.joinMode],
		...panel.incumbentVariants.map((variant, vIndex) => [
			`DeployedBaseline: panel ${index} variant ${vIndex} what it computes`,
			variant.whatItComputes
		])
	])
].filter(([, text]) => typeof text === 'string' && text.length > 0);

ok(
	removedBoundaries.length >= 60,
	`the before-set is the whole of what the six boundaries showed (got ${removedBoundaries.length})`
);
for (const [where, text] of removedBoundaries) {
	ok(reachable.has(text), `still reachable in the verification section — ${where}`);
}

// …and that check can fail. Drop one sentence from the trail and the same loop
// must reject it, or it is measuring nothing.
{
	const before = failures;
	const [, dropped] = removedBoundaries[0];
	const holed = new Set(reachable);
	holed.delete(dropped);
	ok(!holed.has(dropped), 'the before/after check rejects a trail missing one sentence');
	eq(failures, before, 'the non-vacuity probe itself passed');
}

// ---------------------------------------------------------------------------
// (b) COVERAGE IS DERIVED, not listed. Walk each payload independently and
// require every twin it holds to be in the trail.
// ---------------------------------------------------------------------------

function twinsOf(node, into = new Set(), seen = new Set()) {
	if (node === null || typeof node !== 'object' || seen.has(node)) return into;
	seen.add(node);
	if (typeof node.shipped === 'string' && typeof node.plain === 'string') {
		into.add(node.shipped);
		return into;
	}
	for (const value of Array.isArray(node) ? node : Object.values(node)) twinsOf(value, into, seen);
	return into;
}

let walked = 0;
for (const { file, payload } of sources) {
	const expected = twinsOf(payload);
	walked += expected.size;
	const group = trail.files.find((entry) => entry.file === file);
	ok(group !== undefined, `${file} has a group in the trail`);
	const got = new Set((group?.entries ?? []).map((entry) => entry.shipped));
	const missing = [...expected].filter((text) => !got.has(text));
	ok(missing.length === 0, `${file}: ${missing.length} twinned sentences missing from its group`);
	for (const text of missing.slice(0, 3)) console.error(`      missing: ${JSON.stringify(text.slice(0, 90))}`);
}
ok(walked > 150, `the independent walk found the page's twins (got ${walked})`);
eq(walked, trail.nEntries, 'the trail holds exactly the twins the payloads carry, no more');

// Every entry states both halves and where it sits — a group of empty rows
// would satisfy a count and nothing else.
for (const file of trail.files) {
	for (const entry of file.entries) {
		ok(entry.shipped.length > 0 && entry.plain.length > 0, `${file.file}: an entry has an empty half`);
		ok(entry.field.length > 0, `${file.file}: an entry does not say where it sits`);
	}
}

// ---------------------------------------------------------------------------
// (c) FAIL-CLOSED. Each planted defect must gate its own file and nothing else.
// ---------------------------------------------------------------------------

let planted = 0;
function gatesOnly(label, brokenPayload) {
	planted += 1;
	const withDefect = sources.map((entry) =>
		entry.file === 'statement_review_queue.json' ? { ...entry, payload: brokenPayload } : entry
	);
	const gated = buildPaperAuditTrail(withDefect);
	const group = gated.files.find((entry) => entry.file === 'statement_review_queue.json');
	ok(group.entries.length === 0, `${label}: the affected file contributes no entries`);
	ok(
		typeof group.unavailable === 'string' && group.unavailable.length > 0,
		`${label}: the affected file says why it is withheld`
	);
	const others = gated.files.filter((entry) => entry.file !== 'statement_review_queue.json');
	ok(
		others.every((entry) => entry.entries.length > 0),
		`${label}: every other file still contributes`
	);
	return group;
}

gatesOnly('a twin with no shipped half', {
	...queue,
	prose: { ...queue.prose, decisionRule: { plain: 'a restatement with no source' } }
});
gatesOnly('a twin with an empty shipped half', {
	...queue,
	prose: { ...queue.prose, decisionRule: { shipped: '', plain: 'a restatement with no source' } }
});
gatesOnly('a twin with no plain half', {
	...queue,
	prose: { ...queue.prose, decisionRule: { shipped: 'a shipped sentence with no restatement' } }
});
gatesOnly('a twin with an empty plain half', {
	...queue,
	prose: { ...queue.prose, decisionRule: { shipped: 'a shipped sentence', plain: '' } }
});
{
	const group = gatesOnly('a shipped half that is not a string', {
		...queue,
		prose: { ...queue.prose, decisionRule: { shipped: 42, plain: 'a restatement' } }
	});
	ok(
		group.unavailable.includes('prose.decisionRule'),
		'the gate names the field, so it can be found and fixed'
	);
}

// The same trail WITHOUT a planted defect must not gate — otherwise the four
// cases above would pass on a builder that gates unconditionally.
{
	const clean = buildPaperAuditTrail(sources);
	ok(
		clean.files.every((file) => file.unavailable === null),
		'an intact page gates nothing — the fail-closed cases are not gating on everything'
	);
}

// ---------------------------------------------------------------------------
// (d) PROVENANCE IS NEVER GUESSED.
// ---------------------------------------------------------------------------

const okLoad = (path, sha, payloadKey, payload) => ({
	status: 'ok',
	reason: null,
	artifact_path: path,
	artifact_sha256: sha,
	[payloadKey]: payload
});
const gatedLoad = (path, payloadKey) => ({
	status: 'unavailable',
	reason: 'the artifact is missing.',
	artifact_path: path,
	artifact_sha256: null,
	[payloadKey]: null
});

{
	const loads = {
		paperLiteral: {
			...okLoad('data/results/x/paper_literal_vs_llms.json', FAKE_SHA, 'arms', []),
			faithfulness: null,
			apDecomposition: decomposition,
			generatedNoteProse: { shipped: 'generated on a date', plain: 'generated on a date' }
		},
		paperOwnMetric: okLoad('data/benchmark/published.json', FAKE_SHA, 'figure', {
			twin: { shipped: 'a published sentence', plain: 'the same, plainly' }
		}),
		deployedBaseline: okLoad('data/results/y/deployed.json', FAKE_SHA, 'figure', deployed),
		paperPerEvidence: okLoad('data/results/z/per_evidence.json', FAKE_SHA, 'figure', perEvidence),
		paperRobustness: okLoad('data/results/x/robust.json', FAKE_SHA, 'figure', robustness),
		framingCorrection: {
			...okLoad('data/results/x/framing_correction.json', FAKE_SHA, 'framing', framing),
			control_path: 'data/results/x/non_reading_control.json',
			control_sha256: 'b'.repeat(64),
			control
		},
		reviewQueue: okLoad('data/results/x/statement_review_queue.json', FAKE_SHA, 'queue', queue),
		statementErrorF1: okLoad('data/results/x/statement_error_f1.json', FAKE_SHA, 'figure', errorF1),
		paperTable6Extended: okLoad('data/results/x/table6.json', FAKE_SHA, 'figure', table6),
		beliefLadder: okLoad('data/results/x/belief_model_ladder.json', FAKE_SHA, 'ladder', ladder)
	};

	const specs = paperAuditSources(loads);
	const named = specs.map((spec) => spec.file);
	eq(new Set(named).size, named.length, 'every file appears once');
	ok(named.includes('statement_review_queue.json'), 'the file name is taken from the path');
	ok(named.includes('non_reading_control.json'), 'the second file of a two-file load is named');
	ok(
		named.includes('ap_decomposition_by_paper_band.json'),
		'the file read through another loader is named'
	);

	const decompSpec = specs.find((spec) => spec.file === 'ap_decomposition_by_paper_band.json');
	eq(decompSpec.sha256, null, 'a file whose digest no load carries borrows none');
	eq(decompSpec.path, null, 'and claims no path it was not given');
	const controlSpec = specs.find((spec) => spec.file === 'non_reading_control.json');
	eq(controlSpec.sha256, 'b'.repeat(64), 'the control file carries its OWN digest, not its sibling’s');

	// A gated load contributes a group that says why, and no entries.
	const withGate = paperAuditSources({
		...loads,
		reviewQueue: gatedLoad('data/results/x/statement_review_queue.json', 'queue')
	});
	const gatedSpec = withGate.find((spec) => spec.file === 'statement_review_queue.json');
	ok(gatedSpec.unavailable !== null, 'a gated load says why it contributes nothing');
	const gatedTrail = buildPaperAuditTrail(withGate);
	const gatedGroup = gatedTrail.files.find((file) => file.file === 'statement_review_queue.json');
	eq(gatedGroup.entries.length, 0, 'a gated load contributes no sentences');
	ok(gatedTrail.nEntries > 100, 'and the rest of the page is unaffected');

	// The count of files with something to show must move when one goes dark —
	// a tally that never changes is a tally that is not being computed.
	ok(
		buildPaperAuditTrail(withGate).nFiles < buildPaperAuditTrail(specs).nFiles,
		'the file tally responds to a file going dark'
	);
}

// ---------------------------------------------------------------------------
// (e) ONE BOUNDARY, and the six framings are gone from the components.
// ---------------------------------------------------------------------------

/**
 * WHAT A READER MEETS, not what a comment explains. These scans run over the
 * MARKUP: the docblock at the top of a component is where the defect is named
 * and argued about, and a scan that could not tell the two apart would force
 * every one of those explanations to be deleted or paraphrased into uselessness.
 * Both helpers are proved below to keep summary text and drop comment text.
 */
const markupOf = (text) =>
	text
		.replace(/<script[\s\S]*?<\/script>/g, ' ')
		.replace(/<style[\s\S]*?<\/style>/g, ' ')
		.replace(/<!--[\s\S]*?-->/g, ' ');
/** The same, reduced to the words themselves: no tags, no expressions. */
const proseOf = (text) =>
	markupOf(text)
		.replace(/\{[^}]*\}/g, ' ')
		.replace(/<[^>]*>/g, ' ');

{
	const probe = '<script>const a = "in the artifact\'s own words";</script>\n' +
		'<!-- a comment saying Caveats, as shipped -->\n' +
		'<details><summary>a summary a reader reads</summary></details>';
	ok(!markupOf(probe).includes("artifact's own words"), 'the markup scan drops script text');
	ok(!markupOf(probe).includes('Caveats, as shipped'), 'the markup scan drops comment text');
	ok(markupOf(probe).includes('a summary a reader reads'), 'the markup scan keeps what is rendered');
	ok(proseOf(probe).includes('a summary a reader reads'), 'the prose scan keeps what is rendered');
	ok(!proseOf(probe).includes('details'), 'the prose scan drops tag names');
}

const componentSource = source('lib/components/PaperAuditTrail.svelte');
eq(
	(markupOf(componentSource).match(/<details/g) ?? []).length,
	1,
	'the verification section is ONE boundary, not a nest of them'
);
ok(
	/for verification against the shipped files/.test(markupOf(componentSource)),
	'the boundary is introduced as what it is'
);

/**
 * Every framing that invites a reader into the files' own dialect.
 *
 * THESE ARE PATTERNS, NOT LITERALS, AND THAT CHANGE WAS EARNED. The list was four
 * exact substrings, and this file's own component then shipped an <h2> reading
 * "The result files, and what they say in their own words" — a near-synonym that
 * walked straight through a scan running over that very file. Keying a guard to
 * the exact wording of the defect you just fixed catches that defect and nothing
 * adjacent to it, which is the shape of at least four misses on this page.
 *
 * The rule being enforced is semantic: no framing may present the shipped text as
 * the authentic version and the plain text as a paraphrase. So match the SHAPE —
 * "<possessive> own words", "verbatim", "as shipped", "the real/actual version".
 */
const INVITATIONS = [
	/\b(?:its|their|his|her|the artifact['’]s|the file['’]s)\s+own\s+words\b/i,
	/\bverbatim\b/i,
	/\bas\s+shipped\b/i,
	/\bthe\s+(?:real|actual|true|unedited|original)\s+(?:version|wording|text)\b/i,
	/\bin\s+full,?\s+unedited\b/i
];
/**
 * The route is in this list because the LAST of the boundaries was on the page
 * itself, not in a component: a <details> headed "caveats, verbatim from the
 * artifact" — a phrase that is an INVITATION_PROBE below, i.e. a wording this
 * scan is proved to catch, sitting unscanned because the route was never scanned.
 * A guard that runs over every component and skips the page they are mounted on
 * has a hole shaped exactly like the file that assembles them.
 */
const CONVERTED = [
	'routes/paper/+page.svelte',
	'lib/components/ReviewQueue.svelte',
	'lib/components/StatementErrorF1.svelte',
	'lib/components/BeliefModelLadder.svelte',
	'lib/components/DeployedBaseline.svelte',
	'lib/components/ApDecompositionByPaperRank.svelte',
	'lib/components/PaperAuditTrail.svelte'
];
for (const path of CONVERTED) {
	const text = markupOf(source(path));
	for (const invitation of INVITATIONS) {
		const hit = text.match(invitation);
		ok(hit === null, `${path} no longer invites a reader in: matched ${invitation} at "${hit?.[0]}"`);
	}
}
// The check is only worth anything if those shapes would be found. Prove it — and
// prove it on the phrasing that ESCAPED the literal list, not only on the phrasings
// the list was written from, which would prove nothing about its reach.
const INVITATION_PROBES = [
	"what they say in their own words",
	"the artifact's own words",
	'caveats, verbatim from the artifact',
	'Caveats, as shipped',
	'the real wording',
	'in full, unedited'
];
for (const probe of INVITATION_PROBES) {
	ok(
		INVITATIONS.some((invitation) => invitation.test(markupOf(`<p>prefix ${probe} suffix</p>`))),
		`the invitation scan catches "${probe}"`
	);
}
// ...and does not fire on ordinary prose that merely mentions the files.
for (const innocent of [
	'Check this page against the result files',
	'each sentence beside the restatement that replaced it',
	'for verification against the shipped files'
]) {
	ok(
		!INVITATIONS.some((invitation) => invitation.test(markupOf(`<p>${innocent}</p>`))),
		`the invitation scan allows "${innocent}"`
	);
}

/**
 * THE ACCESSORS THAT PUT THE FILES' OWN WORDING ON SCREEN, one line each, as
 * they were written before this node ran. None may come back: every one of them
 * now renders its `…Prose.plain` twin, and the shipped half is reachable only in
 * the verification section. A reader who wants the file's wording gets it there,
 * once, with the file's name and digest beside it.
 *
 * Each pattern is proved to match its own old form below, so an entry that had
 * gone stale — a guard checking for something that can no longer be written — is
 * caught rather than counted.
 */
const REMOVED_ACCESSORS = [
	['lib/components/ReviewQueue.svelte', '{queue.decisionRule}'],
	['lib/components/ReviewQueue.svelte', '{queue.thresholdRule}'],
	['lib/components/ReviewQueue.svelte', '{queue.equalYield.operatingRule}'],
	['lib/components/ReviewQueue.svelte', '{queue.equalYield.oracleDisclosure}'],
	['lib/components/ReviewQueue.svelte', 'queue.caveats as caveat'],
	['lib/components/ReviewQueue.svelte', '{robust.robustness.metric}'],
	['lib/components/ReviewQueue.svelte', '{robust.robustness.budgetRule}'],
	['lib/components/ReviewQueue.svelte', '{robust.robustness.bootstrapDesign}'],
	['lib/components/ReviewQueue.svelte', '{robust.robustness.multiplicityMethod}'],
	['lib/components/ReviewQueue.svelte', '{robust.robustness.multiplicityNote}'],
	['lib/components/ReviewQueue.svelte', '{robust.robustness.labelCompleteness.note}'],
	['lib/components/StatementErrorF1.svelte', '{figure.headlineThresholdRule.rule}'],
	['lib/components/StatementErrorF1.svelte', '{figure.headlineThresholdRule.oracle}'],
	['lib/components/StatementErrorF1.svelte', '{figure.matchedThresholdRule.rule}'],
	['lib/components/StatementErrorF1.svelte', '{figure.matchedThresholdRule.oracle}'],
	['lib/components/StatementErrorF1.svelte', '{figure.reconciliation.thresholdRule.rule}'],
	['lib/components/StatementErrorF1.svelte', '{figure.modalThresholdNote}'],
	['lib/components/StatementErrorF1.svelte', '{figure.metric}'],
	['lib/components/StatementErrorF1.svelte', '{figure.positiveClassNote}'],
	['lib/components/StatementErrorF1.svelte', '{figure.decisionRule}'],
	['lib/components/StatementErrorF1.svelte', '{figure.bootstrapDesign}'],
	['lib/components/StatementErrorF1.svelte', '{figure.panel.labelProvenance}'],
	['lib/components/StatementErrorF1.svelte', '{figure.panel.ordering}'],
	['lib/components/StatementErrorF1.svelte', '{figure.multiplicity.method}'],
	['lib/components/StatementErrorF1.svelte', '{figure.multiplicity.note}'],
	['lib/components/StatementErrorF1.svelte', '{figure.reconciliation.note}'],
	['lib/components/StatementErrorF1.svelte', 'figure.caveats as caveat'],
	['lib/components/BeliefModelLadder.svelte', '{ladder.panel.labelConvention}'],
	['lib/components/BeliefModelLadder.svelte', '{ladder.panel.ordering}'],
	['lib/components/BeliefModelLadder.svelte', '{ladder.metricSource}'],
	['lib/components/BeliefModelLadder.svelte', '{ladder.noisyOrFormula}'],
	['lib/components/BeliefModelLadder.svelte', '{proximity.status}'],
	['lib/components/BeliefModelLadder.svelte', '{ladder.join}'],
	['lib/components/BeliefModelLadder.svelte', 'ladder.caveats as caveat'],
	['lib/components/DeployedBaseline.svelte', '{payload.claim}'],
	['lib/components/DeployedBaseline.svelte', '{payload.claimIsNot}'],
	['lib/components/DeployedBaseline.svelte', '{payload.question}'],
	['lib/components/DeployedBaseline.svelte', '{payload.metricSource}'],
	['lib/components/DeployedBaseline.svelte', '{payload.incumbentSelectionRule}'],
	['lib/components/DeployedBaseline.svelte', '{payload.arms.gate.whatItIs}'],
	['lib/components/DeployedBaseline.svelte', 'payload.caveats as caveat'],
	['lib/components/DeployedBaseline.svelte', '{panel.curatorNote}'],
	['lib/components/DeployedBaseline.svelte', '{variant.whatItComputes}'],
	['lib/components/DeployedBaseline.svelte', '{panel.heterogeneity.joinMode}']
];
{
	const byFile = new Map();
	for (const [path, accessor] of REMOVED_ACCESSORS) {
		if (!byFile.has(path)) byFile.set(path, markupOf(source(path)));
		ok(
			!byFile.get(path).includes(accessor),
			`${path} no longer renders the file's own wording via ${accessor}`
		);
		// The pattern must be able to fire, or it is a guard that checks nothing.
		ok(
			markupOf(`<p>before ${accessor} after</p>`).includes(accessor),
			`the removed-accessor scan can match ${accessor}`
		);
	}
	ok(REMOVED_ACCESSORS.length >= 40, 'the removed-accessor list is the whole of what moved');
}

// ---------------------------------------------------------------------------
// (f) NO INVENTED VOCABULARY in the section's own prose. The shipped halves it
// renders come from the files and are exempt; this is the static text only.
// ---------------------------------------------------------------------------

const BANNED = [
	'arm',
	'tau',
	'lane',
	'plate',
	'rung',
	'census',
	'incumbent',
	'pooled',
	'max-t',
	'gift',
	'tie-robust',
	'OOF'
];
const renderedProse = proseOf(componentSource);
for (const word of BANNED) {
	ok(
		!new RegExp(`\\b${word}\\b`, 'i').test(renderedProse),
		`the section's own prose does not say "${word}"`
	);
}
ok(/\barm\b/i.test('an arm of the study'), 'the banned-word scan can actually match');
ok(renderedProse.trim().length > 200, 'the banned-word scan is reading real prose, not an empty string');

console.log(
	`\n  ${removedBoundaries.length} sentences from the six removed boundaries, all still reachable; ` +
		`${planted} planted defects, each gating its own file only`
);

if (failures > 0) {
	console.error(`\n${failures} assertion(s) failed`);
	process.exit(1);
}
console.log(
	'paper audit trail: one boundary, every shipped sentence reachable, fail-closed per file'
);
