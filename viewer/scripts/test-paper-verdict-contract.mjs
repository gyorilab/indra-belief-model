/**
 * Contract runner for the ranked verdict — the block above every figure on /paper.
 *
 * WHY THIS FILE EXISTS. The verdict is the one surface where being WRONG and
 * being CONFIDENT arrive together: it states three claims in strength order, and
 * a reader who stops after it has read the whole page. So the three things that
 * could make it lie are pinned here against the REAL shipped artifacts:
 *
 *   1. THE RANKING IS THE EVIDENCE'S, not the author's. The tiers must come out
 *      sorted by strength, with `rank` stamped after the sort.
 *   2. EVERY NUMBER IS READ. Each figure the block prints is checked against the
 *      shipped artifact it came from, so a typed number cannot survive here.
 *   3. A MISSING LOAD GATES ITS TIER, with the loader's own reason, rather than
 *      printing a claim with no evidence behind it.
 *
 * Plus the page-wide rule the verdict is held to hardest, because it is the
 * first prose a reader meets: no invented vocabulary in any string it renders.
 * Class (a4) of test-paper-render-invariants.mjs scans the module statically;
 * this runner scans the strings AS BUILT from the shipped bytes, which is the
 * half a static scan cannot see.
 */
import { readFileSync, readdirSync } from 'node:fs';

import { validateStatementErrorF1 } from '../src/lib/data/paper-error-f1.ts';
import { validateReviewQueue } from '../src/lib/data/paper-review-queue.ts';
import { validatePaperRobustness } from '../src/lib/data/paper-robustness.ts';
import { buildPaperVerdict } from '../src/lib/data/paper-verdict.ts';

let failures = 0;
function ok(condition, label) {
	if (!condition) {
		console.error(`FAIL ${label}`);
		failures += 1;
	}
}
function eq(actual, expected, label) {
	ok(
		Object.is(actual, expected),
		`${label}: got ${JSON.stringify(actual)}, want ${JSON.stringify(expected)}`
	);
}

const MODEL_DIR = '../../data/results/indra_paper_literal_models_20260724/';
const read = (name) => JSON.parse(readFileSync(new URL(MODEL_DIR + name, import.meta.url), 'utf8'));

const rawErrorF1 = read('statement_error_f1.json');
const rawQueue = read('statement_review_queue.json');
const rawRobustness = read('paper_margin_robustness.json');

const errorF1Load = validateStatementErrorF1(rawErrorF1, { artifactPath: 'fixture' });
const robustnessLoad = validatePaperRobustness(rawRobustness, { artifactPath: 'fixture' });
const queueLoad = {
	status: 'ok',
	reason: null,
	artifact_path: 'fixture',
	artifact_sha256: 'deadbeef',
	queue: validateReviewQueue(rawQueue)
};
eq(errorF1Load.status, 'ok', 'the shipped error-finding artifact validates');
eq(robustnessLoad.status, 'ok', 'the shipped ordering artifact validates');

const dark = (reason) => ({
	status: 'unavailable',
	reason,
	artifact_path: 'fixture',
	artifact_sha256: null,
	figure: null,
	queue: null
});

const verdict = buildPaperVerdict({
	statementErrorF1: errorF1Load,
	reviewQueue: queueLoad,
	paperRobustness: robustnessLoad
});

console.log('\n(1) three tiers, ranked by the evidence');
eq(verdict.status, 'ok', 'the block renders');
eq(verdict.tiers.length, 3, 'three questions, always');
const ids = verdict.tiers.map((tier) => tier.id);
ok(new Set(ids).size === 3, `each question appears once: ${ids.join(', ')}`);
verdict.tiers.forEach((tier, index) => {
	eq(tier.rank, index + 1, `${tier.id}: rank is its position`);
	ok(tier.question.length > 0, `${tier.id}: states its question`);
});

// The ordering is a fact about the evidence: strengths must be non-decreasing in
// weakness down the block. Written against the same order the module declares so
// a new strength cannot be slipped in without landing somewhere in this list.
const STRENGTH_ORDER = ['solid', 'real-but-small', 'not-shown', 'cannot-be-settled'];
const weights = verdict.tiers.map((tier) =>
	tier.status === 'ok' ? STRENGTH_ORDER.indexOf(tier.strength) : STRENGTH_ORDER.length
);
ok(
	weights.every((weight) => weight >= 0),
	`every strength is a known one: ${JSON.stringify(
		verdict.tiers.map((tier) => tier.strength ?? 'unavailable')
	)}`
);
ok(
	weights.every((weight, index) => index === 0 || weights[index - 1] <= weight),
	`tiers descend in strength: ${JSON.stringify(weights)}`
);

console.log('\n(2) the shipped result, as the block ranks it today');
const byId = new Map(verdict.tiers.map((tier) => [tier.id, tier]));
const finding = byId.get('finds-wrong-statements');
const ordering = byId.get('orders-statements');
const generality = byId.get('how-much-better');
eq(finding.status, 'ok', 'the error-finding claim renders');
eq(finding.strength, 'solid', 'finding wrong statements is the solid claim');
eq(finding.rank, 1, 'and it leads the block');
eq(ordering.status, 'ok', 'the ordering claim renders');
eq(ordering.strength, 'real-but-small', 'ordering is real but small');
eq(generality.status, 'ok', 'the generality claim renders');
eq(generality.strength, 'cannot-be-settled', 'how much better cannot be settled here');
ok(
	generality.rank === 3,
	`the unsettleable question ranks last: got ${generality.rank}`
);

console.log('\n(3) every number is read off the artifact, not typed');
// The three winners of the error-finding comparison, straight from the bytes.
const winners = rawErrorF1.arms.filter((arm) => arm.excludes_zero_simultaneous === true);
eq(winners.length, 3, 'three reading models clear zero once the interval is widened');
const leader = winners.reduce((best, arm) =>
	arm.operating_point.error_precision > best.operating_point.error_precision ? arm : best
);
const findingText = JSON.stringify(finding);
const pct = (value) => `${Math.round(value * 100)}%`;
ok(
	findingText.includes(pct(leader.operating_point.error_precision)),
	`the flag-precision figure is the shipped one (${pct(leader.operating_point.error_precision)})`
);
ok(
	findingText.includes(pct(rawErrorF1.reference.operating_point.error_precision)),
	`the random forest's own figure is shipped too (${pct(rawErrorF1.reference.operating_point.error_precision)})`
);
const signed = (value) => `${value >= 0 ? '+' : '-'}${Math.abs(value).toFixed(4)}`;
const margins = winners.map((arm) => arm.delta_error_f1);
ok(
	findingText.includes(signed(Math.min(...margins))) &&
		findingText.includes(signed(Math.max(...margins))),
	`the margin range is the shipped one (${signed(Math.min(...margins))} to ${signed(Math.max(...margins))})`
);
ok(
	findingText.includes(rawErrorF1.multiplicity.max_t_critical_value.toFixed(4)),
	'the value the margin has to clear is read, not typed'
);
// The review budget: the block quotes it only against the model the 2023 paper
// published, so the number and the name are checked together.
const reference = rawQueue.equal_yield.references.find(
	(entry) => entry.arm === rawQueue.equal_yield.reference_arm
);
const comparator = reference.comparators.find(
	(entry) => entry.arm === rawQueue.equal_yield.budget_sweep.comparator_arm
);
ok(findingText.includes(String(reference.budget)), `the budget is shipped (${reference.budget})`);
ok(
	findingText.includes(String(reference.true_errors_caught)),
	`the catch is shipped (${reference.true_errors_caught})`
);
ok(findingText.includes(String(rawQueue.panel.n_errors)), 'the error count is shipped');
ok(
	findingText.includes(String(comparator.budget_for_equal_yield)),
	`the random forest's budget is shipped (${comparator.budget_for_equal_yield})`
);

const apLeader = rawRobustness.arms.reduce((best, arm) =>
	arm.primary.delta > best.primary.delta ? arm : best
);
const orderingText = JSON.stringify(ordering);
ok(orderingText.includes(signed(apLeader.primary.delta)), 'the ordering margin is shipped');
ok(
	orderingText.includes(signed(apLeader.primary.ci95_low)) &&
		orderingText.includes(signed(apLeader.primary.ci95_high)),
	'its own interval is shipped'
);
ok(
	orderingText.includes(signed(apLeader.primary.simultaneous_low)) &&
		orderingText.includes(signed(apLeader.primary.simultaneous_high)),
	'the corrected interval is shipped'
);
// The corrected interval is the fact the tier turns on: it must cross zero for
// "real but small" to be the honest word, and the artifact must say so itself.
eq(
	apLeader.primary.excludes_zero_simultaneous,
	false,
	'the corrected interval crosses zero in the shipped bytes'
);
eq(apLeader.primary.excludes_zero_pointwise, true, 'its own interval does not');
const generalityText = JSON.stringify(generality);
ok(
	generalityText.includes(rawRobustness.panels.primary.n_statements.toLocaleString()),
	'the benchmark size is shipped'
);

console.log('\n(4) a dark artifact gates its own tier, with the loader’s reason');
const noFinding = buildPaperVerdict({
	statementErrorF1: dark('statement_error_f1.json is missing.'),
	reviewQueue: queueLoad,
	paperRobustness: robustnessLoad
});
eq(noFinding.status, 'ok', 'the rest of the block still stands');
const gatedFinding = noFinding.tiers.find((tier) => tier.id === 'finds-wrong-statements');
eq(gatedFinding.status, 'unavailable', 'the error-finding tier gates');
eq(gatedFinding.reason, 'statement_error_f1.json is missing.', 'and states the loader’s reason');
eq(gatedFinding.rank, 3, 'a tier that makes no claim cannot outrank one that does');

ok(
	gatedFinding.plainReason.length > 0 && gatedFinding.plainReason !== gatedFinding.reason,
	'and hands the reader a sentence of its own rather than the loader’s'
);

const noQueue = buildPaperVerdict({
	statementErrorF1: errorF1Load,
	reviewQueue: dark('statement_review_queue.json is missing.'),
	paperRobustness: robustnessLoad
});
const findingWithoutQueue = noQueue.tiers.find((tier) => tier.id === 'finds-wrong-statements');
eq(findingWithoutQueue.status, 'ok', 'the claim survives losing a supporting figure');
if (findingWithoutQueue.status === 'ok' && finding.status === 'ok') {
	eq(
		findingWithoutQueue.numbers.length,
		finding.numbers.length - 1,
		'and loses exactly the row that artifact carried'
	);
	ok(
		!JSON.stringify(findingWithoutQueue).includes(String(comparator.budget_for_equal_yield)),
		'the budget number is gone rather than stale'
	);
}

const allDark = buildPaperVerdict({
	statementErrorF1: dark('statement_error_f1.json is missing.'),
	reviewQueue: dark('statement_review_queue.json is missing.'),
	paperRobustness: dark('paper_margin_robustness.json is missing.')
});
eq(allDark.status, 'unavailable', 'with nothing loaded the block gates');
eq(allDark.tiers.length, 3, 'and still names all three questions');
ok(
	allDark.tiers.every((tier) => tier.status === 'unavailable' && tier.reason.length > 0),
	'each with its own reason'
);
ok(
	allDark.tiers.every((tier) => tier.plainReason.length > 0),
	'and its own sentence for the reader'
);

console.log('\n(5) DIRECTION — the sign of a margin decides the words, three ways');
// THE DEFECT THIS SECTION EXISTS FOR. `excludesZero` is `low > 0 || high < 0`,
// TRUE for an interval lying entirely BELOW zero, and the block used to print
// `excludesZero ? 'Clears zero.' : 'Crosses zero.'`. Driven with a losing model it
// printed "its own 95% interval = -0.0256 to -0.0061 | Clears zero." beneath a
// claim that no model ordered better. That was the SIXTH sign-blindness regression
// on this page, so every reachable sign case is driven here, from the real
// validated figure, with only the numbers under test changed.
const pts = (value) => value * 100;

/**
 * The shipped figure with every model's ordering margin replaced. Every other
 * field stays exactly as the validator built it, so this is the real object shape
 * and not a hand-built stand-in.
 *
 * The interval's own `standing` is set to whatever the caller asks. That is the
 * point of two of the drives below: this block classifies the ENDPOINTS itself
 * and must reach the right words even when the object it is handed carries a
 * class that disagrees with its own bounds. (Its predecessor took a lying
 * `excludesZero` boolean the same way; that boolean no longer exists anywhere,
 * and the class that replaced it is held to the same standard.)
 */
function standingOf(low, high) {
	if (low > 0) return 'ahead';
	if (high < 0) return 'behind';
	return 'not-significant';
}
function orderingDriven({ delta, low, high, simLow, simHigh, standing, standingSim }) {
	const load = structuredClone(robustnessLoad);
	for (const lane of load.figure.lanes) {
		lane.primaryDelta = delta;
		lane.pointwise = {
			deltaPts: pts(delta),
			lowPts: pts(low),
			highPts: pts(high),
			standing: standing ?? standingOf(low, high)
		};
		lane.simultaneous = {
			deltaPts: pts(delta),
			lowPts: pts(simLow),
			highPts: pts(simHigh),
			standing: standingSim ?? standingOf(simLow, simHigh)
		};
	}
	return load;
}
const drive = (patch) =>
	buildPaperVerdict({
		statementErrorF1: errorF1Load,
		reviewQueue: queueLoad,
		paperRobustness: orderingDriven(patch)
	});
const tierOf = (block, id) => block.tiers.find((tier) => tier.id === id);
const textOf = (tier) => JSON.stringify(tier);

// (5a) A LOSS. The real gemma-4-e2b numbers, which is what the reviewer drove.
const behind = drive({
	delta: -0.01601,
	low: -0.02564,
	high: -0.00615,
	simLow: -0.0275,
	simHigh: -0.00452
});
const behindOrdering = tierOf(behind, 'orders-statements');
eq(behindOrdering.status, 'ok', 'a losing model still renders a tier');
eq(behindOrdering.strength, 'not-shown', 'a margin below zero is not a claim we showed');
ok(
	!/clears zero/i.test(textOf(behindOrdering)),
	`nothing in a losing tier says it clears zero: ${textOf(behindOrdering).slice(0, 160)}`
);
ok(
	/below zero/i.test(behindOrdering.numbers[1].note),
	`the interval note names the side it lies on: ${JSON.stringify(behindOrdering.numbers[1].note)}`
);
ok(
	/below/i.test(behindOrdering.claim),
	`and the claim says the reading model is the one behind: ${JSON.stringify(behindOrdering.claim)}`
);

// (5b) A LOSS PINNED TIGHTLY — the generality tier's own sign bug. Magnitude alone
// made this "real but small" under a claim reading "how much better … the range is
// narrower than the effect it brackets".
const behindTight = drive({
	delta: -0.01601,
	low: -0.0164,
	high: -0.0156,
	simLow: -0.017,
	simHigh: -0.015
});
const behindGenerality = tierOf(behindTight, 'how-much-better');
eq(behindGenerality.status, 'ok', 'the generality tier renders on a pinned loss');
eq(behindGenerality.strength, 'not-shown', 'a pinned LOSS is never "real but small"');
ok(
	!/how much better language-model reading is can be put within a range/i.test(
		behindGenerality.claim
	),
	`and never claims a range on how much BETTER: ${JSON.stringify(behindGenerality.claim)}`
);
ok(/below/i.test(behindGenerality.claim), 'it states the direction it actually measured');
// The same tightness with the sign flipped is the case that sentence was written
// for, and it must still reach it.
const aheadTight = drive({
	delta: 0.01601,
	low: 0.0156,
	high: 0.0164,
	simLow: 0.015,
	simHigh: 0.017
});
eq(
	tierOf(aheadTight, 'how-much-better').strength,
	'real-but-small',
	'a pinned WIN is the case that claim was written for'
);

// (5c) AHEAD, own interval crossing zero — the fourth reachable case, which had
// three branches to fall into and no sentence of its own.
const grazing = drive({
	delta: 0.00771,
	low: -0.00125,
	high: 0.0167,
	simLow: -0.00287,
	simHigh: 0.01828
});
const grazingOrdering = tierOf(grazing, 'orders-statements');
eq(grazingOrdering.strength, 'not-shown', 'a margin that clears nothing shows nothing');
ok(
	!/still lies entirely/i.test(textOf(grazingOrdering)),
	'no interval note claims a side it does not lie on'
);

// (5d) A WIN THAT SURVIVES THE CORRECTION.
const solid = drive({
	delta: 0.00979,
	low: 0.00121,
	high: 0.01853,
	simLow: 0.0005,
	simHigh: 0.02
});
const solidOrdering = tierOf(solid, 'orders-statements');
eq(solidOrdering.strength, 'solid', 'a margin clearing zero after the correction is solid');
ok(
	/entirely above zero/i.test(solidOrdering.numbers[2].note),
	`and the corrected note says so: ${JSON.stringify(solidOrdering.numbers[2].note)}`
);

// All four cases are reachable and none of them shares another's sentences: the
// fall-through that shipped was a case with no words of its own.
const cases = [behindOrdering, grazingOrdering, solidOrdering, ordering];
const claims = new Set(cases.map((tier) => tier.claim));
const doubts = new Set(cases.map((tier) => tier.doubt));
eq(claims.size, 4, 'each ordering case states its own claim');
eq(doubts.size, 4, 'and names its own reason to doubt it');

// (5e) THE SHIPPED CLASS IS NOT READ. A crossing interval handed to this block
// labelled 'ahead' — exactly what a downstream consumer would trip over — must
// change nothing, because the block re-derives the class from the endpoints.
const lyingClass = drive({
	delta: 0.00771,
	low: -0.00125,
	high: 0.0167,
	simLow: -0.00287,
	simHigh: 0.01828,
	standing: 'ahead',
	standingSim: 'ahead'
});
eq(
	JSON.stringify(tierOf(lyingClass, 'orders-statements')),
	JSON.stringify(grazingOrdering),
	'a lying standing changes nothing: the block reads the endpoints'
);
// AND THE OTHER DIRECTION, which the boolean version could not express: a
// strictly-negative interval mislabelled 'ahead' must still print the losing
// words. Without this, (5e) would pass on a block that had simply stopped
// reading the field at all while still being wrong about the sign.
const lyingAboutALoss = drive({
	delta: -0.01601,
	low: -0.02564,
	high: -0.00615,
	simLow: -0.0275,
	simHigh: -0.00452,
	standing: 'ahead',
	standingSim: 'ahead'
});
eq(
	JSON.stringify(tierOf(lyingAboutALoss, 'orders-statements')),
	JSON.stringify(behindOrdering),
	'an interval below zero labelled ahead still reads as a loss'
);

// (5f) A MARGIN OUTSIDE ITS OWN INTERVAL gates the tier rather than printing a
// direction and a range that disagree.
const contradictory = drive({
	delta: 0.02,
	low: -0.00125,
	high: 0.0167,
	simLow: -0.00287,
	simHigh: 0.01828
});
for (const id of ['orders-statements', 'how-much-better']) {
	const tier = tierOf(contradictory, id);
	eq(tier.status, 'unavailable', `${id}: contradictory numbers gate the tier`);
	ok(tier.reason.length > 0 && tier.plainReason.length > 0, `${id}: with both reasons`);
}

// (5f2) THE SAME CLASS ONE TIER UP. When nothing clears the widened interval the
// error-finding tier used to print ONE doubt for both sub-cases: "widened to cover
// every reading model we ran, the interval crosses zero." Driven with four models
// that all sit entirely BELOW zero — a measured shortfall, the opposite of an open
// question — that sentence is simply false.
function errorF1AllBehind() {
	const load = structuredClone(errorF1Load);
	for (const lane of load.figure.lanes) {
		if (!lane.delta || !lane.inMaxTFamily) continue;
		lane.delta.delta = -0.13;
		lane.delta.ciLow = -0.16;
		lane.delta.ciHigh = -0.1;
		lane.delta.simLow = -0.17;
		lane.delta.simHigh = -0.09;
		lane.delta.simultaneousStanding = 'behind';
		lane.delta.winsPointwise = false;
		lane.delta.winsSimultaneously = false;
		lane.delta.standing = 'behind';
		lane.delta.pointwiseStanding = 'behind';
	}
	return load;
}
const allBehind = buildPaperVerdict({
	statementErrorF1: errorF1AllBehind(),
	reviewQueue: queueLoad,
	paperRobustness: robustnessLoad
});
const allBehindFinding = tierOf(allBehind, 'finds-wrong-statements');
eq(allBehindFinding.status, 'ok', 'the error-finding tier renders with every model behind');
eq(allBehindFinding.strength, 'not-shown', 'and shows nothing');
ok(
	!/the interval crosses zero/i.test(allBehindFinding.doubt),
	`the doubt does not call a measured shortfall an open question: ${JSON.stringify(
		allBehindFinding.doubt
	)}`
);
ok(
	!/clears zero/i.test(JSON.stringify(allBehindFinding.numbers)),
	`and no caption counts "margins clearing zero" when four of them cleared it downward: ${JSON.stringify(
		allBehindFinding.numbers
	)}`
);

// (5g) STRUCTURAL, AND NOT ONLY FOR THIS MODULE. No CODE anywhere under /paper
// names a sign-blind zero-exclusion boolean — the prose may, since naming the
// defect is how it stays fixed.
//
// This began as a one-module scan, written the day the sixth occurrence was found
// HERE. That was the wrong scope: the boolean lived on seven loader interfaces and
// nineteen component branches, and a rule that cleaned one module while the field
// still existed would have caught none of the previous five. The field is now gone
// from every loader, so the scan is over every loader and every component — the
// only form in which "the two-way branch cannot be written" is a fact rather than
// a hope.
//
// Comments are stripped, not the whole file: half of these files carry a docblock
// explaining why the boolean is gone, and a rule that forbade SAYING it would take
// the explanation with it.
const stripComments = (source) =>
	source.replace(/\/\*[\s\S]*?\*\//g, '').replace(/^\s*\/\/.*$/gm, '').replace(/<!--[\s\S]*?-->/g, '');
//
// WHAT IS AND IS NOT FLAGGED, and the line is the same one class (a4) draws:
// snake_case is the ARTIFACTS' spelling and camelCase is OURS. A loader MUST read
// `excludes_zero_pointwise` off the bytes to gate it against its own endpoints —
// forbidding that would delete the fail-closed check. What is forbidden is OUR
// side of the boundary carrying the boolean onward under a camelCase name.
const ZERO_BOOLEAN = /\bexcludesZero\w*\b/;

// NOT VACUOUS, PROVED BOTH WAYS ON FIXTURES BEFORE THE REAL FILES ARE READ. Three
// guards on this project have run green while checking nothing, and a fourth was
// found testing an architecture that no longer existed.
for (const [what, snippet] of [
	['a camelCase read', 'const a = x.excludesZero ? 1 : 2;'],
	['a suffixed read', 'if (d.excludesZeroSimultaneous) draw();'],
	['a destructure', 'const { excludesZero } = interval;'],
	['a field definition', '\texcludesZeroPointwise: boolean;'],
	['a read on a line that also holds a comment', 'const c = d.excludesZero; // note']
]) {
	ok(ZERO_BOOLEAN.test(stripComments(snippet)), `(5g) the detector catches ${what}`);
}
for (const [what, snippet] of [
	['a line comment naming it', '// `excludesZero` is sign-blind; deleted'],
	['a block comment naming it', '/** replaced excludesZeroPointwise */'],
	['an HTML comment naming it', '<!-- no excludesZero here -->'],
	['the artifact’s own snake_case field, which must stay readable', 'boolean(row.excludes_zero_pointwise, ctx);'],
	['a three-way class', 'const s = standingOfBounds(low, high);']
]) {
	ok(!ZERO_BOOLEAN.test(stripComments(snippet)), `(5g) the detector allows ${what}`);
}

const PAPER_SOURCES = [
	...readdirSync(new URL('../src/lib/data/', import.meta.url))
		.filter((name) => name.startsWith('paper-') && name.endsWith('.ts'))
		.map((name) => [`src/lib/data/${name}`, new URL(`../src/lib/data/${name}`, import.meta.url)]),
	...readdirSync(new URL('../src/lib/components/', import.meta.url))
		.filter((name) => name.endsWith('.svelte'))
		.map((name) => [
			`src/lib/components/${name}`,
			new URL(`../src/lib/components/${name}`, import.meta.url)
		])
];
ok(PAPER_SOURCES.length > 10, `(5g) the scan reaches the real files: ${PAPER_SOURCES.length} scanned`);
let namedIt = 0;
for (const [label, url] of PAPER_SOURCES) {
	const source = readFileSync(url, 'utf8');
	if (ZERO_BOOLEAN.test(source)) namedIt += 1;
	const offending = stripComments(source)
		.split('\n')
		.map((line, index) => [index + 1, line])
		.filter(([, line]) => ZERO_BOOLEAN.test(line));
	ok(
		offending.length === 0,
		`${label} reads a sign-blind zero-exclusion boolean: ` +
			offending.map(([n, line]) => `${n}: ${line.trim().slice(0, 70)}`).join(' | ')
	);
}
ok(namedIt > 0, 'the scan is not vacuous — some of these files still discuss the deleted boolean');

// (5h) STRUCTURAL, the other half. A gated tier's `reason` is audit text written
// for whoever is holding a broken artifact — it names that artifact's own fields.
// The component printed it verbatim beside two working tiers, which is a runtime
// string no static dialect sweep can see. So the template is checked here: it
// prints `plainReason`, and it does not print `reason`.
const componentSource = readFileSync(
	new URL('../src/lib/components/PaperVerdict.svelte', import.meta.url),
	'utf8'
);
const template = componentSource.slice(componentSource.indexOf('</script>'));
const templateCode = template.replace(/<!--[\s\S]*?-->/g, '');
ok(
	/\{tier\.plainReason\}/.test(templateCode),
	'the component prints the gated tier’s plain sentence'
);
ok(
	!/\{tier\.reason\}/.test(templateCode),
	'and never prints the loader’s own reason, which is audit text'
);

console.log('\n(6) no invented vocabulary in a string the reader meets');
// The operator's banned list, verbatim from GOAL. `panel` and `fold` are on it
// too: this block defines neither, so it may use neither.
const BANNED = [
	[/\barms?\b/i, 'model, or the model’s own name'],
	[/\btau\b/i, 'score cutoff'],
	[/\blanes?\b/i, 'row'],
	[/\bplates?\b/i, 'figure'],
	[/\brungs?\b/i, 'step'],
	[/\bcensus\b/i, 'count'],
	[/\bincumbents?\b/i, 'the scorer INDRA ships today'],
	[/\bpooled\b/i, 'over all statements at once'],
	[/\bmax[-\s]?t\b/i, '“corrected for having run four models”'],
	[/\bOOF\b/, 'scored on the slice it never learned from'],
	[/\bgift\b/i, 'interpolation credit'],
	[/\btie[-\s]?robust\b/i, 'stepped'],
	[/\bpanels?\b/i, 'benchmark, or “the 1,689 statements”'],
	[/\bfolds?\b/i, 'slice'],
	[/\b(?:the\s+)?paper[’'`]s(?:\s+own)?\s+(?:models?|RFs?|metrics?|estimators?)\b/i, 'name it']
];
/**
 * Every string the component prints, in the order it prints them.
 *
 * A GATED tier contributes `plainReason` and NOT `reason`: the loader's reason is
 * audit text, free to name an artifact's own fields, and the component does not
 * print it. Scanning it here would have flagged a string no reader sees; NOT
 * scanning `plainReason` would repeat the defect this list exists to catch.
 */
function readerStrings(block) {
	const out = [];
	for (const tier of block.tiers) {
		out.push(tier.question);
		if (tier.status === 'ok') {
			out.push(tier.strengthWord, tier.claim, tier.doubt);
			for (const number of tier.numbers) {
				out.push(number.caption, number.value);
				if (number.note) out.push(number.note);
			}
		} else {
			out.push(tier.plainReason);
		}
	}
	return out;
}
// Every block built in this file, not just the shipped one: the sentences a
// losing model reaches, and the sentences a gated tier reaches, are prose a reader
// meets too, and they are exactly the ones no earlier sweep had ever rendered.
const strings = [
	verdict,
	behind,
	behindTight,
	aheadTight,
	grazing,
	solid,
	contradictory,
	allBehind,
	noFinding,
	allDark
].flatMap(readerStrings);
ok(strings.length >= 20, `the block renders prose to scan: ${strings.length} strings`);
for (const text of strings) {
	for (const [pattern, plain] of BANNED) {
		const hit = text.match(pattern);
		ok(hit === null, `“${hit?.[0]}” → ${plain}\n        ${JSON.stringify(text.slice(0, 110))}`);
	}
}
// The scan is not vacuous: it must catch the words if they were there.
const planted = ['this arm wins', 'below tau', 'on the panel', 'max-t corrected', 'the paper’s model'];
for (const sentence of planted) {
	const caught = BANNED.some(([pattern]) => pattern.test(sentence));
	ok(caught, `the scan catches ${JSON.stringify(sentence)}`);
}
console.log(`  ${strings.length} reader strings scanned, ${BANNED.length} terms enforced`);

if (failures > 0) {
	console.error(`\n${failures} contract failure(s)`);
	process.exit(1);
}
console.log('\npaper-verdict contract OK');
