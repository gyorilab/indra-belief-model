/**
 * /paper RENDER INVARIANTS — the cross-cutting sweep.
 *
 * The per-figure contract runners check each figure against its own artifact.
 * This one checks the three defect CLASSES that have each shipped more than
 * once across the whole /paper surface, and it checks them at the source level,
 * because every one of them is invisible to a rendering test:
 *
 *   (a) A FROZEN JOIN KEY RENDERED AS A DISPLAY NAME. Strings like
 *       `Paper literal RF+promoter` address shipped `point_metrics` bytes. They
 *       must never change, which is exactly why they must never be shown to a
 *       reader who would reasonably ask us to change them. This has leaked six
 *       times. The rule enforced here has no exceptions to remember: nothing
 *       named `.label` (or `.labels` / `.armLabel` / `.gateLabel`) may appear in
 *       a RENDER position. Key expressions inside `{#each … (key)}` are not
 *       render positions and are not flagged.
 *
 *   (b) AN UNENFORCED SVG LABEL BUDGET. SVG text does not wrap and does not
 *       warn: a right-anchored string that overruns its gutter loses its LEADING
 *       glyphs, and the <desc> beside it still emits the full string, so screen
 *       readers and a11y checks both report success. Five of these have shipped.
 *       Every gutter on this page is now measured, budgeted, and ENFORCED in a
 *       builder that throws or a predicate that gates — this runner re-derives
 *       each budget from the geometry it claims to come from, and exercises the
 *       enforcement on a string that should fail.
 *
 *   (c) A PLACEHOLDER THAT RENDERS AS A FACT. `ece: 0` shipped: a failed join
 *       printed "ECE 0.000", the IDEAL value, ranking a broken arm above every
 *       arm that joined. Its siblings are any `?? 0` / `?? ''` / `|| 0` sitting
 *       in a render position, and any measurement field typed non-nullable on a
 *       path that can degrade.
 *
 * Runs with `node --experimental-strip-types`. Pure: reads source text and
 * imports the pure data modules. Touches no artifact and no network.
 */

import { readdirSync, readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

import {
	AP_DECOMP_COUNT_NOTE_BUDGET_CHARS,
	AP_DECOMP_FAN_GEOMETRY,
	AP_DECOMP_FAN_SLOTS,
	AP_DECOMP_MIRROR_GEOMETRY,
	AP_DECOMP_MIRROR_LABEL_BUDGET_CHARS,
	AP_DECOMP_MIRROR_READOUT_BUDGET_CHARS,
	apDecompCountNoteFits,
	apDecompFanNamesFit
} from '../src/lib/data/paper-ap-decomposition.ts';
import {
	BELIEF_LADDER_BASELINE_TAG_WORST_CASE_CHARS,
	BELIEF_LADDER_DISPLAY_BUDGET_CHARS,
	BELIEF_LADDER_ENTRY_SPECS,
	BELIEF_LADDER_GEOMETRY,
	beliefLadderBaselineTag,
	beliefLadderDisplay
} from '../src/lib/data/paper-belief-ladder.ts';
import {
	STATEMENT_ERROR_F1_DRAWN_ARM_IDS,
	STATEMENT_ERROR_F1_GEOMETRY,
	STATEMENT_ERROR_F1_LABEL_BUDGET_CHARS,
	STATEMENT_ERROR_F1_LONGEST_DRAWN_LABEL,
	STATEMENT_ERROR_F1_READOUT_A_BUDGET_CHARS,
	STATEMENT_ERROR_F1_READOUT_B_BUDGET_CHARS,
	STATEMENT_ERROR_F1_SUB_BUDGET_CHARS,
	validateStatementErrorF1
} from '../src/lib/data/paper-error-f1.ts';
import {
	FRAMING_CONTROL_GEOMETRY,
	FRAMING_CONTROL_LABEL_BUDGET_CHARS
} from '../src/lib/data/paper-framing-correction.ts';
import { PAPER_LITERAL_ARM_SPECS } from '../src/lib/data/paper-literal.ts';
import {
	PAPER_TABLE6_GEOMETRY,
	PAPER_TABLE6_LABEL_BUDGET_CHARS,
	PAPER_TABLE6_METRIC_BUDGET_CHARS,
	PAPER_TABLE6_RANK_BUDGET_CHARS,
	PAPER_TABLE6_TIE_BUDGET_CHARS,
	validatePaperTable6Extended
} from '../src/lib/data/paper-table6-extended.ts';
import {
	PAPER_OWN_METRIC_GEOMETRY,
	PAPER_OWN_METRIC_LABEL_BUDGET_CHARS,
	PAPER_OWN_METRIC_READOUT_BUDGET_CHARS
} from '../src/lib/data/paper-own-metric.ts';
import {
	PAPER_ROBUSTNESS_GEOMETRY,
	PAPER_ROBUSTNESS_LABEL_BUDGET_CHARS,
	PAPER_ROBUSTNESS_READOUT_BUDGET_CHARS
} from '../src/lib/data/paper-robustness.ts';
import {
	TIE_SCATTER_GEOMETRY,
	tieScatterLabelFits
} from '../src/lib/data/paper-tie-inflation.ts';

let failures = 0;
function ok(condition, label) {
	if (condition) return;
	failures += 1;
	console.error(`FAIL ${label}`);
}
function eq(got, want, label) {
	if (Object.is(got, want)) return;
	failures += 1;
	console.error(`FAIL ${label}: got ${JSON.stringify(got)}, want ${JSON.stringify(want)}`);
}
function read(relative) {
	return readFileSync(fileURLToPath(new URL(relative, import.meta.url)), 'utf8');
}
/** The shipped artifact behind a /paper figure, parsed. One reader, one path. */
function readArtifact(name) {
	return JSON.parse(read(`../../data/results/indra_paper_literal_models_20260724/${name}`));
}
/** A deep copy to mutate, so an enforcement can be exercised on real bytes. */
function mutable(artifact) {
	return JSON.parse(JSON.stringify(artifact));
}

// ---------------------------------------------------------------------------
// The surface. Every /paper component and data module in this sweep's scope.
// `ReviewQueue` / `DeployedBaseline` and their data modules are checked by their
// own runners; they are listed here too because these three defect classes are
// page-wide and a per-figure runner is exactly what missed them before.
//
// THE LIST IS A FLOOR, NOT THE SURFACE. A hand-kept list of files to scan has
// exactly one failure mode and it is silent: a component lands on the page and
// nobody adds it, so every class here runs green over a file it never opened.
// That already happened — `PerEvidenceGrain.svelte` has been mounted on /paper
// since the per-evidence figure shipped and appeared in NEITHER list below, so
// classes (a), (a2), (a4) and (c) had never read it. The surface is therefore
// DERIVED from the one place that decides what is on the page — the route's own
// import graph — and unioned with the list, which can then only add files the
// route does not import YET (`PaperVerdict.svelte` is exactly that today: built,
// enforced from the day it lands, mounted by a later node).
// ---------------------------------------------------------------------------
const COMPONENT_DIR = '../src/lib/components/';
const PAPER_ROUTE = '../src/routes/paper/+page.svelte';

/** Every `$lib/components/X.svelte` a source imports. */
function componentImportsIn(source) {
	return [...source.matchAll(/['"]\$lib\/components\/([A-Za-z0-9_$]+\.svelte)['"]/g)].map(
		(match) => match[1]
	);
}
/**
 * Every component the /paper route can reach, followed TRANSITIVELY: no /paper
 * component imports another one today, and the day one does, the child is on the
 * page just as much as its parent.
 */
function discoverPaperComponents() {
	const seen = new Set();
	const pending = componentImportsIn(read(PAPER_ROUTE));
	while (pending.length > 0) {
		const name = pending.shift();
		if (seen.has(name)) continue;
		seen.add(name);
		for (const child of componentImportsIn(read(`${COMPONENT_DIR}${name}`))) pending.push(child);
	}
	return [...seen].sort();
}

/**
 * Built and enforced BEFORE the route imports them. Everything the route already
 * imports arrives through discovery instead, so this list holds only the files a
 * later node will mount — never a file that is already on the page.
 */
const COMPONENTS_UNMOUNTED = [
	// EMPTY, and that is the point rather than an oversight. `PaperVerdict.svelte`
	// sat here from the day it landed — enforced before it was mounted, because it
	// is the FIRST prose on the page and may be all a reader reads. The route now
	// imports it, and the assertion below refuses to let a mounted file stay named
	// here, so discovery is the only thing keeping it in scope. `PaperAuditTrail
	// .svelte` never needed an entry: it came into scope through the same import
	// graph on the same day. The list stays because the next component built
	// ahead of its mount needs somewhere to be enforced from.
];

const COMPONENTS_MOUNTED = discoverPaperComponents();

/**
 * Scanned and REPORTED, not gated — and the distinction is stated rather than
 * quietly encoded. Both files render a field literally named `label`, but in both
 * cases the string is a budget-checked DISPLAY name (`arm.display` /
 * `panel.display` / the released label FIELD name), not a frozen `point_metrics`
 * join key: no key reaches a reader today. What is missing is the naming
 * convention that makes the rule checkable at a glance, and these two files are
 * owned elsewhere in this pass. The fix is a rename at each construction site
 * (`label:` -> `display:`) plus its render site; when it lands, move these two
 * names into COMPONENTS above and the rule becomes exception-free.
 */
const COMPONENTS_DEFERRED = ['DeployedBaseline.svelte', 'ReviewQueue.svelte'];

/** Mounted ∪ built-but-not-yet-mounted, less the two reported separately. */
const COMPONENTS = [...new Set([...COMPONENTS_MOUNTED, ...COMPONENTS_UNMOUNTED])]
	.filter((name) => !COMPONENTS_DEFERRED.includes(name))
	.sort();
/** Every /paper component, however it got into scope. */
const COMPONENTS_ALL = [...COMPONENTS, ...COMPONENTS_DEFERRED].sort();

// The discovery is the surface, so it is exercised before anything trusts it: a
// parser that silently returns [] would empty every class in this file at once,
// which is the exact way two guards on this project shipped green over nothing.
console.log('\n(surface) the files this sweep scans are derived from the route');
eq(
	componentImportsIn("import Zed from '$lib/components/Zed.svelte';").join(),
	'Zed.svelte',
	'the import parser reads a component import'
);
eq(
	componentImportsIn("import { fmtDelta } from '$lib/data/paper-error-f1';").length,
	0,
	'a data-module import is not a component'
);
ok(
	COMPONENTS_MOUNTED.length >= 16,
	`the /paper route imports ${COMPONENTS_MOUNTED.length} components — the parser found too few`
);
{
	const onDisk = new Set(
		readdirSync(fileURLToPath(new URL(COMPONENT_DIR, import.meta.url))).filter((name) =>
			name.endsWith('.svelte')
		)
	);
	for (const name of [...COMPONENTS_UNMOUNTED, ...COMPONENTS_DEFERRED]) {
		ok(onDisk.has(name), `${name} is named in this sweep but is no longer on disk`);
	}
	// A file named here AND imported by the route is a list that has stopped
	// meaning anything: the union would hide the duplication, so it is said out loud.
	for (const name of COMPONENTS_UNMOUNTED) {
		ok(
			!COMPONENTS_MOUNTED.includes(name),
			`${name} is mounted on /paper now — drop it from COMPONENTS_UNMOUNTED, discovery has it`
		);
	}
}
console.log(
	`  ${COMPONENTS_MOUNTED.length} imported by the route` +
		` + ${COMPONENTS_UNMOUNTED.length} built but not yet mounted` +
		` = ${COMPONENTS_ALL.length} components in scope` +
		` (${COMPONENTS_DEFERRED.length} of them reported, not gated, by class (a))`
);

/**
 * Strip <script>, <style> and HTML comments: only the template can render.
 *
 * BLANKED, not deleted — every stripped byte becomes a space and every newline is
 * kept. The offsets of what remains are therefore the offsets of the real file,
 * which is what makes `line N` in a report the line a person can open. Collapsing
 * each region to a single character (the obvious implementation, and the one this
 * had) shifted every reported line number in every template by however much sat
 * above it, so the class printed a line number that pointed at nothing.
 */
function templateOf(source) {
	return source.replace(
		/<!--[\s\S]*?-->|<script\b[^>]*>[\s\S]*?<\/script\s*>|<style\b[^>]*>[\s\S]*?<\/style\s*>/gi,
		(match) => match.replace(/[^\n]/g, ' ')
	);
}

/**
 * Every RENDER-position mustache in a Svelte template, brace-matched so nested
 * `{}` inside an expression stay with their expression.
 *
 * Control-flow tags (`{#…}`, `{/…}`, `{:…}`, `{@…}`) are excluded, and that is
 * what makes `{#each rows as row (row.entry.label)}` legal: a key expression
 * identifies a row, it never reaches the screen. Nothing else in a control-flow
 * tag renders either — an `{#if x.label === 'y'}` condition draws no text.
 * Attribute mustaches (`title={arm.label}`) ARE included: a tooltip is a render.
 */
function renderExpressions(template) {
	const out = [];
	for (let i = 0; i < template.length; i += 1) {
		if (template[i] !== '{') continue;
		let depth = 0;
		let j = i;
		for (; j < template.length; j += 1) {
			if (template[j] === '{') depth += 1;
			else if (template[j] === '}') {
				depth -= 1;
				if (depth === 0) break;
			}
		}
		if (depth !== 0) break;
		const body = template.slice(i + 1, j);
		i = j;
		if (/^[#/:@]/.test(body.trimStart())) continue;
		out.push(body);
	}
	return out;
}

// ---------------------------------------------------------------------------
// (a) no frozen join key in a render position
// ---------------------------------------------------------------------------
const JOIN_KEY_ACCESS = /\.\s*(label|labels|armLabel|gateLabel)\b/;

console.log('\n(a) frozen join keys never reach a render position');
for (const name of COMPONENTS) {
	const template = templateOf(read(`../src/lib/components/${name}`));
	const offenders = renderExpressions(template).filter((expression) =>
		JOIN_KEY_ACCESS.test(expression)
	);
	ok(
		offenders.length === 0,
		`${name} renders a join-key field: ${offenders.map((o) => o.trim().slice(0, 60)).join(' | ')}`
	);
}
console.log(`  ${COMPONENTS.length} components enforced`);
for (const name of COMPONENTS_DEFERRED) {
	const template = templateOf(read(`../src/lib/components/${name}`));
	const offenders = renderExpressions(template).filter((expression) =>
		JOIN_KEY_ACCESS.test(expression)
	);
	if (offenders.length === 0) continue;
	console.log(
		`  DEFERRED ${name}: ${offenders.length} render(s) of a field named \`label\`` +
			' — a display string today, but rename it and move this file into COMPONENTS'
	);
}

// The indirection that makes the rule enforceable rather than aspirational: a
// join key can only become a name by going through a lookup that FAILS on an
// unknown key. A `?? label` fallback would quietly reinstate the defect.
eq(
	beliefLadderDisplay('Hierarchy propagation'),
	'noisy-OR + hierarchy propagation',
	'the ladder resolves a join key to its own display name'
);
let threw = false;
try {
	beliefLadderDisplay('a rung that does not exist');
} catch {
	threw = true;
}
ok(threw, 'beliefLadderDisplay throws on an unknown key instead of echoing it');

// Every fan slot's on-screen name is decoupled from the point_metrics key it
// joins on. The paper-side slot is the one that has actually leaked.
for (const slot of AP_DECOMP_FAN_SLOTS) {
	if (!slot.label.startsWith('Paper literal')) continue;
	ok(slot.display !== slot.label, `${slot.label}: display must not be the frozen join key`);
}

// ---------------------------------------------------------------------------
// (a2) no render site decides the DIRECTION of a margin for itself
//
// The class this sweep was created for and did not actually cover. `excludesZero`
// is `ciLow > 0 || ciHigh < 0` — SIGN-BLIND by construction — so a two-way branch
// on it prints the same words for a significant win and a significant loss. It
// has recurred FIVE times: the ● counter in PaperLiteralComparison, A2's join
// keys, StatementErrorF1's outer partition, its inner pointwise clause, and
// PaperTable6Extended's "the lead survives the correction", which shipped into a
// brand-new component AFTER this sweep was extended to cover that component —
// because the sweep had no sign class at all. It does now.
//
// THE RULE, AND ITS LIMIT, BOTH STATED. Flagged: a `excludesZero*` test whose
// branches differ in DIRECTION. Not flagged, deliberately, because both are
// correct and a rule that cries wolf gets switched off:
//   · `excludesZero ? 'yes' : 'no'` — PaperRobustness renders exactly this, and
//     it is TRUE either way: the question asked is zero-exclusion, not direction.
//   · `delta > 0 ? '▲' : '▼'` — sign-AWARE, reading the same number it draws.
//
// WHAT THIS CANNOT CATCH, said plainly rather than left to be discovered: the
// occurrence-#5 defect was a directional verb in UNCONDITIONAL prose beside an
// ungated margin ("leads theirs by −0.0010"). No template regex reaches that; it
// is caught at the loader, by requiring such a margin to carry a derived
// standing. This class is a backstop, not the guarantee.
// ---------------------------------------------------------------------------
const DIRECTION_WORD = /\b(?:ahead|behind|leads?|trails?|beats?|wins?|loses?|above|below|better|worse)\b/i;
const ZERO_TEST = /\bexcludesZero\w*\b/;

console.log('\n(a2) a sign-blind predicate never selects a directional word');
for (const name of COMPONENTS) {
	const template = templateOf(read(`../src/lib/components/${name}`));
	const offenders = renderExpressions(template).filter(
		(expression) => ZERO_TEST.test(expression) && DIRECTION_WORD.test(expression)
	);
	ok(
		offenders.length === 0,
		`${name} picks a directional word from a sign-blind test: ` +
			`${offenders.map((o) => o.trim().slice(0, 70)).join(' | ')}`
	);
}
console.log(`  ${COMPONENTS.length} components enforced`);

// ---------------------------------------------------------------------------
// (a4) NO PRIVATE DIALECT IN A STRING A READER SEES
//
// /paper was written in a house dialect. A rewrite pass took the COMPONENT prose
// from 383 jargon instances to 51 — and the dialect survived anyway, in the one
// place a template scan cannot see: the RENDERING FIELDS of the data modules.
// `display`, `legend`, `name`, `title`, `subLabel` are strings the components
// print verbatim, and they carried `arm`, `tau`, `arms [`, `the paper's RF` and
// `incumbent` straight to the screen after the prose above them had been fixed.
// Twice now the same words have been rewritten and grown back. This class is the
// third rewrite made permanent.
//
// WHAT IS SCANNED — both surfaces, because a class that read only templates would
// have caught NONE of the fifteen strings the last pass had to fix by hand:
//   · every /paper component template and the /paper route: text nodes (which is
//     where <title> and <desc> live too), the reader-facing attributes
//     (title / aria-label / alt / placeholder), and string literals sitting
//     inside a render mustache;
//   · every `paper-*.ts` data module, by WALKING the source rather than grepping
//     it, so `arm.display`, `figure.lanes` and `STATEMENT_ERROR_F1_…` — names in
//     code, seen by no reader — are never mistaken for prose.
//
// THE DATA-MODULE RULE IS THE OTHER WAY UP from the one you would guess, and this
// is the single most important line in the class. It does NOT scan a list of
// field names; it treats EVERY string a module writes as prose unless the string
// can be shown to be something else. Scanning a field list was tried first and it
// was quietly, badly wrong: half the dialect that actually shipped is assembled in
// a local `const` — `const dispersion = \`population SD … over … folds\`` — and
// interpolated into a `title` later, so the string never sits under a field name
// at all. That first cut ran GREEN over `one of our … arms`, `incumbent sees all
// of it`, `fold areas only`, `scored on the identical panel, labels and folds`
// and `0 — level with the paper's model`, every one of which was on the screen.
//
// WHAT IS NOT SCANNED. Each exclusion is a claim about who can see the string,
// and each one is exercised below rather than asserted:
//   · VALIDATION-FAILURE STRINGS — `fail(ctx, 'names no arm on this figure')`,
//     `throw new Error(…)`. These render only when a figure GATES, i.e. instead
//     of the figure; no reader on a working page ever sees one. They are written
//     for whoever is holding a broken artifact, and they must be free to name the
//     artifact's own fields. Scanning them would flood this class with dozens of
//     false positives and get it switched off, which is how the last two guards
//     died. Said out loud so nobody re-adds them as "coverage".
//   · ADDRESSES — `${context}.display`, `row[${key}].subLabel`,
//     `paired_delta_vs_reference[${id}].pooled.auroc`, and the one-word root a
//     validator opens with (`record(raw.panel, 'panel')`). No whitespace once the
//     `${…}` holes are closed; a path character in it, or a trailing argument
//     position. A word with no path character (`OOF`, `GLM-5`) is still read.
//   · ARTIFACT BYTES, which we may not rewrite even when they carry the dialect:
//     anything under a snake_case name (`uncertainty_field`, `per_fold_metric` —
//     snake_case is the ARTIFACTS' spelling, ours is camelCase), anything under an
//     `expected` / `canonical` / `contract` / `artifact…` name, and frozen-ness is
//     INHERITED by everything nested inside one. `artifactDisplay` holds the
//     sha-pinned string the artifact ships; this class must never be the reason
//     someone edits artifact bytes. Where such a string renders, the shipped
//     pattern is verbatim + a plain-English restatement beside it.
//   · FROZEN KEYS — `label` / `key` / `id` / `series` and their kin. Class (a)
//     already forbids these reaching a render position; this class must not
//     double back and demand they be RENAMED. If (a4) ever flags a string that is
//     also a join key, STOP and report it: the fix is the display name beside the
//     key, never the key.
//   · COMMENTS — a docblock is for us. This very file says `arm` forty times.
//
// `name` is on the scanned list because rule names render — the error-F1 figure
// draws its three cutoff rules by `name`. Where a module uses `name` as a LOOKUP
// instead (REVIEW_QUEUE_ARM_SPECS.name addresses the artifact), the values are
// model names carrying no dialect, so the two uses do not collide today. If one
// ever does, it is the STOP case above: rename the display, never the key.
//
// THE VOCABULARY, and the one correction this class has had to take. Replace only
// the words that are OURS: arm(s) → model(s) or the model's own name · panel →
// benchmark, or plainly "the 1,689 statements" · gate → reading · tau → score
// cutoff · pooled → over all statements at once · lane → row · plate → figure ·
// rung → step · census → count · incumbent → the method in use today.
//
// `fold`, `out-of-fold` and `cross-validation` are NOT ours. They are the field's
// standard terms, and the audience for this page is the lab that published a
// cross-validation paper. An earlier pass replaced `fold` with the coinage
// `slice` throughout, which was the same defect as `arm` and `tau` pointing the
// other way: it charged the reader a word nobody uses in exchange for one they
// already own. REVERSED 2026-07-29. `slice` is now BANNED as a synonym for
// `fold`, so the reversal cannot silently undo itself — the next sweep that
// reaches for the coinage goes red. Also kept, unglossed and unbanned: average
// precision, precision, recall, F1, random forest, noisy-OR. `max-t` stays banned
// unless it carries its gloss, because that one is an abbreviation, not a term.
//
// `fold` and `panel` are CONDITIONAL — not banned, and not free either. The
// operator's rule is "use the field's standard term, and DEFINE IT ONCE at first
// use", and a define-once rule that nothing checks is a rule that rots the first
// time a sentence is trimmed. So a file that GLOSSES the word in its own prose
// may use it; a file that only uses it may not. That is the whole difference
// between this and the ban above: the ban asks for a different word, the
// conditional asks for one sentence of English beside the same word.
//
// The gloss the page settled on, for reference: "the 10 folds — the statements
// are split into 10 groups, and each fitted model is scored on the group it did
// not train on". The design it describes was verified against the pinned source:
// `StratifiedKFold(10, shuffle=False)`, nothing held out as a separate test set,
// every statement scored once by the fold that excluded it. Only the FITTED
// models need it; the reading models are never trained, are scored once over all
// 1,689 statements, and are then assigned the same fold indices so the identical
// estimator applies to every row.
// ---------------------------------------------------------------------------
console.log('\n(a4) no private dialect in a string a reader sees');

/** Every /paper surface that prints prose. Route included: it is the page. */
const JARGON_TEMPLATES = [
	...COMPONENTS_ALL.map((name) => `${COMPONENT_DIR}${name}`),
	PAPER_ROUTE
];
/** Discovered, not listed: a new /paper data module is scanned the day it lands. */
const JARGON_MODULES = readdirSync(fileURLToPath(new URL('../src/lib/data/', import.meta.url)))
	.filter((name) => name.startsWith('paper-') && name.endsWith('.ts'))
	.sort();

/**
 * Property names whose string values are printed — the operator's five, plus the
 * other names this page actually draws through (`legendText`, `figureTitle`,
 * `chip`, `gloss`, `headline`). Matched on the WHOLE name.
 */
const RENDER_FIELDS = new Set([
	'display',
	'legend',
	'name',
	'title',
	'sublabel',
	'note',
	'text',
	'gloss',
	'chip',
	'headline',
	'subheadline',
	'caption',
	'blurb',
	'eyebrow'
]);
/**
 * The same, matched on a camelCase WORD. `label` and `labels` are here and NOT in
 * the whole-name set above: a bare `label` is a join key (class (a)), but
 * `zeroRuleLabel`, `comparatorLabel` and `descLabels` are annotations this page
 * draws — treating every compound containing `label` as frozen is what hid
 * `0 — level with the paper's model`, a drawn rule annotation, from the first cut
 * of this class.
 */
const RENDER_WORDS = new Set([...RENDER_FIELDS, 'label', 'labels']);
/**
 * Property names whose string values are ADDRESSES, not prose — class (a)'s list,
 * plus the shapes that only ever hold a lookup. Matched on the WHOLE name.
 */
const FROZEN_FIELDS = new Set([
	'key',
	'keys',
	'id',
	'ids',
	'series',
	'label',
	'labels',
	'armlabel',
	'gatelabel',
	'context',
	'path',
	'slug',
	'field',
	'status'
]);
/** The frozen list, matched on a camelCase WORD — deliberately narrower. */
const FROZEN_WORDS = new Set([
	'key',
	'keys',
	'id',
	'ids',
	'artifact',
	'context',
	'path',
	'slug',
	// An object called `expected` / `canonical` / a `contract` is a statement about
	// bytes we did not write. Everything inside one is quoted, not authored.
	'expected',
	'canonical',
	'contract'
]);
/** Calls whose arguments are written for a broken artifact, not for a reader. */
const VALIDATION_CALLEES = new Set(['fail', 'Error', 'TypeError', 'RangeError', 'assert', 'close']);
/**
 * The same, by shape rather than by name. A `parseX` / `validateX` / `expectX`
 * takes the artifact apart and its string arguments exist to describe the shape
 * it wanted when it does not get it — `parseTieGroup(raw, ctx, byLabel, rows,
 * 'one of our llm_reader arms')` is the text of a gate, and its own contract
 * runner pins it. Prose is passed to builders, never to parsers.
 */
const VALIDATION_CALLEE_SHAPE = /^(?:parse|validate|expect|assert|require|reject)[A-Z]/;

function identifierWords(identifier) {
	return identifier
		.replace(/([a-z0-9])([A-Z])/g, '$1 $2')
		.toLowerCase()
		.split(/[^a-z0-9]+/)
		.filter(Boolean);
}
/** 'render' (a reader sees it) · 'frozen' (an artifact or a lookup owns it) · '' */
function classifyField(identifier) {
	const whole = identifier.toLowerCase();
	if (RENDER_FIELDS.has(whole)) return 'render';
	if (FROZEN_FIELDS.has(whole)) return 'frozen';
	// snake_case is the ARTIFACTS' spelling — `uncertainty_field`,
	// `per_fold_metric`, `positive_class`. Our own render fields are camelCase.
	// A snake_case name therefore addresses shipped bytes, and the strings under
	// it are the artifact's own words, quoted here so a drifted artifact gates.
	// They must render verbatim (with a gloss beside them), never be rewritten.
	if (/^[a-z0-9]+(?:_[a-z0-9]+)+$/.test(identifier)) return 'frozen';
	const words = identifierWords(identifier);
	if (words.some((word) => FROZEN_WORDS.has(word))) return 'frozen';
	if (words.some((word) => RENDER_WORDS.has(word))) return 'render';
	return '';
}

function lineOf(source, index) {
	return source.slice(0, index).split('\n').length;
}

/**
 * Every string a TS module WRITES for a reader — that is, every string literal in
 * it except the ones shown to be an address, a key, artifact bytes, a comment or
 * a gate's own failure text.
 *
 * A hand-rolled walk rather than a parse: it tracks the enclosing property name,
 * the enclosing call and its argument index, whether the enclosing call is a
 * gate, and whether a colon after a string is a property colon or a TERNARY's.
 * That is enough to tell `legend: 'strongest form of INDRA's own belief'` (prose)
 * from `series: 'incumbent'` (a lookup) from `budget(x, N, `${ctx}.display`)` (a
 * failure path) from `x ? ' This is the reference arm …' : ''` (prose again, and
 * the case a naive key test loses) — which is what decides whether this class is
 * useful or noise.
 */
function harvestRenderStrings(source) {
	// A bare token this module also uses as a property NAME is a key, not prose:
	// `['incumbent', 'gate', …] as SeriesKey[]` is a key list, and no reader meets
	// it. Narrow on purpose — only single lowercase tokens qualify.
	const keyTokens = new Set(
		[...source.matchAll(/(?:^|[\s{,([])['"]?([A-Za-z_$][\w$]*)['"]?\s*:/gm)].map((m) => m[1])
	);
	const out = [];
	const stack = [{ governing: '', property: '', isCall: false, callee: '', argIndex: 0, dead: false }];
	const top = () => stack[stack.length - 1];
	/**
	 * The innermost property name that DECIDES — a render field or a frozen one.
	 * A name that decides nothing (`legendText: { research: '…' }` — `research` is
	 * a series, not a verdict on who reads the string) is walked past, so the
	 * rendering field that owns the map still governs its values. A name that does
	 * decide stops the walk, which is what keeps `legend: [{ series: 'incumbent' }]`
	 * out: `series` is frozen, so the string under it is an address.
	 */
	const governing = () => {
		for (let i = stack.length - 1; i >= 0; i -= 1) {
			const frame = stack[i];
			if (frame.property && classifyField(frame.property)) return frame.property;
			if (frame.governing && classifyField(frame.governing)) return frame.governing;
		}
		return '';
	};
	/**
	 * Frozen-ness is INHERITED. `metric_contract: { summary: 'arithmetic mean over
	 * 10 cross-validation folds' }` — `summary` says nothing, but everything under
	 * an artifact-shaped block is the artifact's own wording, and this class must
	 * never ask for sha-pinned bytes to be reworded.
	 */
	const insideFrozen = () =>
		stack.some(
			(frame) =>
				(frame.property && classifyField(frame.property) === 'frozen') ||
				(frame.governing && classifyField(frame.governing) === 'frozen')
		);
	let pending = '';
	// The last significant character, so a TERNARY's colon is not mistaken for a
	// property colon: `x ? 'this is prose' : ''` ends in `:` exactly like a key
	// does, and reading it as a key hid a rendered clause — ' This is the
	// reference arm …' — from the first cut of this class.
	let lastSignificant = '';
	// `const legendText: Record<SeriesKey, string> = { … }` — the comma inside the
	// TYPE would otherwise close the property before the value opens, and the map
	// of legend prose would be scanned as if it belonged to nobody. A declaration
	// governs its initialiser, so the name is carried to the `=`.
	let declaring = false;
	let declared = '';
	let i = 0;
	while (i < source.length) {
		const c = source[i];
		if (c === '/' && source[i + 1] === '/') {
			while (i < source.length && source[i] !== '\n') i += 1;
			continue;
		}
		if (c === '/' && source[i + 1] === '*') {
			const end = source.indexOf('*/', i + 2);
			i = end === -1 ? source.length : end + 2;
			continue;
		}
		if (c === '"' || c === "'" || c === '`') {
			let j = i + 1;
			while (j < source.length) {
				if (source[j] === '\\') {
					j += 2;
					continue;
				}
				if (source[j] === c) break;
				j += 1;
			}
			const literal = source.slice(i + 1, j);
			let k = j + 1;
			while (k < source.length && /\s/.test(source[k])) k += 1;
			const isKeyPosition =
				source[k] === ':' && ['', '{', ',', ';', '(', '['].includes(lastSignificant);
			const field = governing();
			// An ADDRESS, not a sentence: no whitespace once its `${…}` holes are
			// closed, and a path character in it. `${context}.arms["${name}"]`,
			// `paired_delta_vs_reference[${id}].pooled.auroc`, `row[${key}].subLabel`.
			// A word with no path character (`OOF`, `GLM-5`) is still scanned.
			const skeleton = stripInterpolations(literal, '');
			const address = !/\s/.test(skeleton) && /[.[\]/_]/.test(skeleton);
			// The same thing without a path character: the ROOT of a validator's
			// context, `record(raw.panel, 'panel')`. A trailing argument with no
			// space in it is an address by position even when it is one word.
			const contextRoot = top().isCall && top().argIndex > 0 && !/\s/.test(skeleton);
			const reusedKey = keyTokens.has(literal.trim());
			if (
				!isKeyPosition &&
				!top().dead &&
				!address &&
				!contextRoot &&
				!reusedKey &&
				!insideFrozen() &&
				classifyField(field) !== 'frozen'
			) {
				// `shippedHalf` is carried for class (a6), which may not ask a
				// sha-pinned quotation to be reworded: a string written under
				// `shipped:` is the artifact's own wording, and (a6) scans the
				// `plain:` half beside it instead. (a4) ignores this field.
				out.push({
					field,
					literal,
					line: lineOf(source, i),
					shippedHalf: stack.some((frame) => frame.property === 'shipped')
				});
			}
			i = j + 1;
			pending = '';
			lastSignificant = c;
			continue;
		}
		if (/[A-Za-z_$]/.test(c)) {
			let j = i;
			while (j < source.length && /[\w$]/.test(source[j])) j += 1;
			pending = source.slice(i, j);
			if (declaring) {
				declared = pending;
				declaring = false;
			} else if (pending === 'const' || pending === 'let' || pending === 'var') {
				declaring = true;
			}
			i = j;
			lastSignificant = 'a';
			continue;
		}
		// A real assignment, not `==` / `=>` / `!=` / `<=` / `>=`.
		const assignment =
			c === '=' &&
			source[i + 1] !== '=' &&
			source[i + 1] !== '>' &&
			!['=', '!', '<', '>'].includes(source[i - 1]);
		if (assignment) {
			if (declared) top().property = declared;
			declared = '';
			pending = '';
			lastSignificant = c;
			i += 1;
			continue;
		}
		if (c === '{' || c === '[' || c === '(') {
			const callee = c === '(' ? pending : '';
			stack.push({
				governing: governing(),
				property: '',
				isCall: c === '(',
				callee,
				argIndex: 0,
				dead:
					top().dead || VALIDATION_CALLEES.has(callee) || VALIDATION_CALLEE_SHAPE.test(callee)
			});
			pending = '';
			lastSignificant = c;
			i += 1;
			continue;
		}
		if (c === '}' || c === ']' || c === ')') {
			if (stack.length > 1) stack.pop();
			pending = '';
			lastSignificant = c;
			i += 1;
			continue;
		}
		if (c === ':') {
			if (pending) top().property = pending;
			pending = '';
			lastSignificant = c;
			i += 1;
			continue;
		}
		if (c === ',') {
			if (top().isCall) top().argIndex += 1;
			else top().property = '';
			pending = '';
			lastSignificant = c;
			i += 1;
			continue;
		}
		if (c === ';') {
			top().property = '';
			pending = '';
			lastSignificant = c;
			i += 1;
			continue;
		}
		pending = '';
		if (!/\s/.test(c)) lastSignificant = c;
		i += 1;
	}
	return out;
}

const READER_ATTRIBUTES = /\b(?:title|aria-label|aria-description|alt|placeholder)\s*=\s*(?:"([^"]*)"|'([^']*)')/g;

/** Everything a Svelte template puts in front of a reader, and nothing else. */
function harvestTemplateStrings(source) {
	const template = templateOf(source);
	const out = [];
	let i = 0;
	while (i < template.length) {
		const c = template[i];
		if (c === '<') {
			// Consume the tag, honouring quoted values and attribute mustaches so a
			// `>` inside either does not end it early.
			let j = i + 1;
			let tag = '';
			while (j < template.length) {
				const d = template[j];
				if (d === '"' || d === "'") {
					let k = j + 1;
					while (k < template.length && template[k] !== d) k += 1;
					tag += template.slice(j, k + 1);
					j = k + 1;
					continue;
				}
				if (d === '{') {
					let depth = 0;
					let k = j;
					for (; k < template.length; k += 1) {
						if (template[k] === '{') depth += 1;
						else if (template[k] === '}') {
							depth -= 1;
							if (depth === 0) break;
						}
					}
					tag += template.slice(j, k + 1);
					j = k + 1;
					continue;
				}
				if (d === '>') break;
				tag += d;
				j += 1;
			}
			for (const match of tag.matchAll(READER_ATTRIBUTES)) {
				out.push({ field: 'attribute', literal: match[1] ?? match[2], line: lineOf(template, i) });
			}
			for (const expression of renderExpressions(tag)) {
				for (const literal of stringLiteralsIn(expression)) {
					out.push({ field: 'attribute', literal, line: lineOf(template, i) });
				}
			}
			i = j + 1;
			continue;
		}
		if (c === '{') {
			let depth = 0;
			let j = i;
			for (; j < template.length; j += 1) {
				if (template[j] === '{') depth += 1;
				else if (template[j] === '}') {
					depth -= 1;
					if (depth === 0) break;
				}
			}
			const body = template.slice(i + 1, j);
			if (!/^[#/:@]/.test(body.trimStart())) {
				for (const literal of stringLiteralsIn(body)) {
					out.push({ field: 'mustache', literal, line: lineOf(template, i) });
				}
			}
			i = j + 1;
			continue;
		}
		let j = i;
		while (j < template.length && template[j] !== '<' && template[j] !== '{') j += 1;
		const text = template.slice(i, j);
		if (text.trim()) out.push({ field: 'text', literal: text, line: lineOf(template, i) });
		i = j;
	}
	return out;
}

function stringLiteralsIn(expression) {
	const out = [];
	for (let i = 0; i < expression.length; i += 1) {
		const c = expression[i];
		if (c !== '"' && c !== "'" && c !== '`') continue;
		let j = i + 1;
		while (j < expression.length) {
			if (expression[j] === '\\') {
				j += 2;
				continue;
			}
			if (expression[j] === c) break;
			j += 1;
		}
		out.push(expression.slice(i + 1, j));
		i = j;
	}
	return out;
}

/**
 * `${…}` and `{…}` are CODE. `on ${panel.display}` says "panel" to nobody, and a
 * class that reads it as prose would demand a variable be renamed to fix a
 * sentence. Brace-matched, so a nested `${fmt(a[0])}` goes out whole.
 */
function stripInterpolations(text, filler = ' ') {
	let out = '';
	for (let i = 0; i < text.length; i += 1) {
		if (text[i] === '{' || (text[i] === '$' && text[i + 1] === '{')) {
			const start = text[i] === '$' ? i + 1 : i;
			let depth = 0;
			let j = start;
			for (; j < text.length; j += 1) {
				if (text[j] === '{') depth += 1;
				else if (text[j] === '}') {
					depth -= 1;
					if (depth === 0) break;
				}
			}
			out += filler;
			i = j;
			continue;
		}
		out += text[i];
	}
	return out;
}

/** Names the 2023 paper published. Cited, not coined — and glossed where drawn. */
const CITED_PHRASES = [
	/fold-mean trapezoidal PR-AUC/gi,
	/\b2023 INDRA assembly paper\b/gi,
	/\bINDRA assembly paper\b/gi
];

const BANNED_TERMS = [
	{ term: 'arm', pattern: /\barms?\b/i, plain: 'model / models, or the model’s own name' },
	{ term: 'tau', pattern: /\btau\b/i, plain: 'score cutoff' },
	{ term: 'lane', pattern: /\blanes?\b/i, plain: 'row' },
	{ term: 'plate', pattern: /\bplates?\b/i, plain: 'figure' },
	{ term: 'rung', pattern: /\brungs?\b/i, plain: 'step, or the scorer’s own name' },
	{ term: 'census', pattern: /\bcensus\b/i, plain: 'count' },
	{ term: 'incumbent', pattern: /\bincumbents?\b/i, plain: 'the scorer INDRA ships today' },
	{ term: 'pooled', pattern: /\bpooled\b/i, plain: 'over all statements at once' },
	{ term: 'max-t', pattern: /\bmax[-\s]?t\b/i, plain: '“corrected for having run four models”' },
	// Case-sensitive: the BARE acronym. "out-of-fold", spelled out, is fold-family
	// and answers to the conditional rule below instead — it is standard, and it is
	// kept. Only the three letters are banned.
	{
		term: 'OOF',
		pattern: /\bOOF\b/,
		plain: 'spell it out — “out of fold”, glossed once: scored by a copy fitted on the other nine folds'
	},
	{
		// THE COINAGE THIS PAGE RETIRED. "slice" was introduced by a plain-language
		// pass as a replacement for "fold" and reverted on 2026-07-29; this entry is
		// what stops the replacement growing back, which is the failure mode every
		// other line in this table exists for. It is a ban and not a conditional
		// because there is nothing to gloss: the word the reader wants is `fold`.
		//
		// It goes red TODAY on two components four sweeps never opened —
		// FidelityPanel and PaperLiteralComparison, neither of which appeared in any
		// shard's file list — which is the only kind of evidence worth anything that
		// a guard is not decoration.
		//
		// The OTHER sense of the word is untouched, and the distinction is why this
		// bans the WORD IN PROSE and not the identifier: `PAPER_COMPARABLE_SLICE`,
		// `slice_counts` and `PaperMethodSlice` are the four INPUT CONFIGURATIONS the
		// 2023 paper scored, not cross-validation folds, and renaming those to `fold`
		// would be a factual error. Their rendered text already says "input
		// configuration"; only the identifiers say slice, and this class does not
		// read identifiers.
		term: 'slice',
		pattern: /\bslices?\b/i,
		plain: 'fold — the field’s own word, glossed once at first use'
	},
	{
		// The article is optional because the defect is the POSSESSIVE, not the
		// phrase: `paper’s RF — research model, never shipped` names nothing to a
		// reader who has not read the paper, exactly as `the paper's RF` does not.
		term: "the paper's model / RF / metric / estimator",
		pattern: /\b(?:the\s+)?paper[’'`]s(?:\s+own)?\s+(?:models?|RFs?|metrics?|estimators?)\b/i,
		plain: 'name it: the random forest · average precision · trapezoidal PR-AUC'
	}
];

/**
 * Allowed where the same file DEFINES the word, banned where it is merely used.
 * The definition test is a real one: the term and its plain-English equivalent in
 * ONE sentence (`[^.]` cannot cross a full stop), joined by a definitional
 * connective. "Our panel is the set of statements this page scores" passes;
 * "of the panel's 1,689 errors … correct statements queued" does not.
 */
const CONDITIONAL_TERMS = [
	{
		// REWRITTEN 2026-07-29, because the old rule tested the WRONG THING. It
		// demanded the word "slice" in the definition, which made a coinage nobody
		// uses the price of saying "fold" — and "fold", "out-of-fold" and
		// "cross-validation" are the field's own terms, owned by the audience this
		// page is written for. What has to hold is that the file GLOSSES the word
		// where it uses it, in whatever plain noun carries the split.
		//
		// `slices?` was in the gloss list for one afternoon, as a courtesy to files
		// mid-conversion. It is gone: `slice` is BANNED above, and a conditional that
		// accepts a banned word as its own definition is a rule arguing with itself —
		// the file would satisfy (a4) and fail (a4) on the same sentence.
		term: 'fold',
		pattern: /\bfolds?\b/i,
		definition:
			/\bfolds?\b[^.]{0,24}(?:—|–|-{1,2}|:|\bthat is\b|\bis\b|\bare\b)[^.]{0,160}\b(?:groups?|parts?|subsets?|splits?)\b/i,
		plain:
			'gloss it once in place — “the 10 folds — the statements split into 10 groups, ' +
			'each model scored on the group it did not train on”'
	},
	{
		term: 'panel',
		pattern: /\bpanels?\b/i,
		definition:
			/\bpanels?\b[^.]{0,24}(?:—|–|-{1,2}|:|\bthat is\b|\bis\b|\bare\b)[^.]{0,160}\b(?:statements|benchmark)\b/i,
		plain: 'benchmark, or plainly “the 1,689 statements”'
	}
];

/**
 * The class itself. `strings` are the reader strings of ONE file; the file is the
 * unit because that is the unit the allow-rule is written in.
 */
function dialectOffences(strings) {
	const readable = strings.map((entry) => {
		let text = stripInterpolations(entry.literal);
		for (const phrase of CITED_PHRASES) text = text.replace(phrase, ' ');
		return { ...entry, text };
	});
	// Joined with a FULL STOP, not a space: the definition test is written in
	// sentences (`[^.]` cannot cross one), and two unrelated strings laid end to
	// end would otherwise read as a sentence that defines the word by accident.
	// That exact accident let `our panel matches` pass in paper-own-metric.ts.
	const fileText = readable.map((entry) => entry.text).join(' . ');
	const offences = [];
	for (const entry of readable) {
		for (const banned of BANNED_TERMS) {
			const hit = entry.text.match(banned.pattern);
			if (hit) offences.push({ ...entry, hit: hit[0], term: banned.term, plain: banned.plain });
		}
		for (const conditional of CONDITIONAL_TERMS) {
			const hit = entry.text.match(conditional.pattern);
			if (!hit) continue;
			if (conditional.definition.test(fileText)) continue;
			offences.push({
				...entry,
				hit: hit[0],
				term: `${conditional.term} (undefined in this file)`,
				plain: conditional.plain
			});
		}
	}
	return offences;
}

function reportOffences(where, offences) {
	ok(
		offences.length === 0,
		`${where} prints the private dialect:\n` +
			offences
				.map(
					(o) =>
						`      line ${o.line} ${o.field}: “${o.hit}” → ${o.plain}\n` +
						`        ${JSON.stringify(o.text.trim().replace(/\s+/g, ' ').slice(0, 96))}`
				)
				.join('\n')
	);
}

// ---------------------------------------------------------------------------
// (a4) IS NOT VACUOUS. Two guards on this project shipped GREEN while checking
// nothing, so the class is exercised before it is trusted, in three ways:
//   1. every banned term, in a sentence, must be CAUGHT;
//   2. the legitimate vocabulary — precision, recall, F1, random forest,
//      noisy-OR, the paper's own published names, and a `fold`/`panel` the file
//      defines — must be ALLOWED, or the class gets switched off inside a week;
//   3. the real shipped sources, MUTATED in memory, must be caught: this is what
//      proves the two HARVESTS reach the strings they claim to, which is where a
//      template-only class would have been silently empty.
// ---------------------------------------------------------------------------
{
	const caught = (text) => dialectOffences([{ field: 'fixture', literal: text, line: 0 }]);

	const MUST_CATCH = [
		['arm', 'the arm’s own full-panel best-error-F1 cut'],
		['arms', 'a simultaneous band over all four reader arms'],
		['tau', 'the review queue’s own rule, under belief ≤ tau'],
		['lane', 'each lane is one method'],
		['lanes', 'the four lanes below'],
		['plate', 'the plate above sets the two side by side'],
		['rung', 'the top rung of the ladder'],
		['census', 'the census note beside the fan'],
		['incumbent', 'the strongest incumbent we could source'],
		['pooled', 'pooled average precision over the statements'],
		['max-t', 'a simultaneous max-t band over the reader family'],
		['max t', 'the max t family correction'],
		['OOF', 'INDRA Bayes source (OOF)'],
		['slice', 'the score moves from slice to slice'],
		['slices', 'averaged over the 10 slices'],
		["the paper's model", 'measured against the paper’s model'],
		["the paper's RF", 'the paper’s RF, re-run here'],
		["the paper's metric", 'scored on the paper’s metric'],
		["the paper's estimator", 'the paper’s own estimator'],
		["paper's RF, no article", 'paper’s RF — research model, never shipped'],
		['fold, undefined in the file', 'the population standard deviation over the paper’s 10 folds'],
		['panel, undefined in the file', 'our model, scored on the same panel']
	];
	for (const [term, sentence] of MUST_CATCH) {
		ok(caught(sentence).length > 0, `(a4) catches “${term}”: ${JSON.stringify(sentence)}`);
	}

	// The regression itself, verbatim: strings that WERE shipped in these very
	// rendering fields and had to be rewritten by hand. If this class cannot catch
	// the text that caused it to be written, it is decoration.
	const PRIOR_REGRESSIONS = [
		'error recall — of the panel’s known errors, the share this arm catches',
		'the same margin under a simultaneous max-t band over the whole reader family',
		'the arm’s own full-panel best-error-F1 cut',
		'a second cut: the cheapest cut catching ≥60% of the panel’s errors',
		'a third cut: the review queue’s own target-recall rule, under belief ≤ tau',
		'our arm, scored on the same panel',
		'same panel, simultaneous max-t band over all four reader arms',
		'OURS — scored on the paper’s identical panel, labels and folds',
		'population SD over 10 folds (dispersion, not a confidence interval)',
		'on the paper’s own estimator, population SD over 10 folds',
		// THE COINAGE, quoted off the two components that were still rendering it on
		// 2026-07-29 — after four shards swept /paper for exactly this word and after
		// this class ran green over both files. Neither appeared in any shard's file
		// list, which is how a page-wide word survives a page-wide sweep.
		'trapezoidal PR-AUC · averaged over the 10 slices',
		'The statements are split into 10 equal slices, and every model here is scored only on the ' +
			'slice it never learned from',
		'the largest gap, on any method we reproduced, between its slice-averaged score',
		'over the 10 slices the statements are split into',
		'averaged over 10 slices ± movement between slices',
		'How the slices were cut'
	];
	for (const sentence of PRIOR_REGRESSIONS) {
		ok(
			caught(sentence).length > 0,
			`(a4) catches a string the last rewrite fixed by hand: ${JSON.stringify(sentence.slice(0, 64))}`
		);
	}

	const MUST_ALLOW = [
		'error precision — of the statements this model flags, the share that really are errors',
		'error recall — of the errors we already know about in these 1,689 statements',
		'error-class F1 — their harmonic mean, so it always lands between the two marks',
		'a random forest on evidence-count features, re-measured',
		'noisy-OR over per-source reliabilities, nothing fitted',
		'scored exactly the way the 2023 INDRA assembly paper scored',
		'fold-mean trapezoidal PR-AUC, the measure the paper reports',
		'the paper’s own released labels, re-run here',
		'Gemma 4 26B reading',
		'the score cutoff that gives each model its own best error-F1',
		'RF 2k-d13 + Type/#PMIDs/promoter',
		// The conditional pair, DEFINED — the exact shape PaperOwnMetric uses. The
		// old fixture quoted the sentence this page shipped before 2026-07-29,
		// which both coined "slices" for the standard word and named the folds by
		// whose they were ("the paper's 10 folds"). Re-quoted from what ships now.
		'the population standard deviation over the 10 cross-validation folds — the statements ' +
			'are split into 10 groups, each fitted model is scored on the group it did not train on',
		'Our panel is the set of statements this page scores'
	];
	for (const sentence of MUST_ALLOW) {
		const offences = caught(sentence);
		ok(
			offences.length === 0,
			`(a4) allows ${JSON.stringify(sentence.slice(0, 60))}: flagged ${offences
				.map((o) => o.hit)
				.join(', ')}`
		);
	}

	// A definition earns the word for the FILE, not for the sentence: the same
	// undefined use passes once the file defines it, and only then.
	const undefinedUse = { field: 'fixture', literal: 'the spread across the 10 folds', line: 0 };
	const definition = {
		field: 'fixture',
		// Glossed in a plain noun, NOT in the retired coinage: this fixture used to
		// read "each of the 10 slices", which taught the banned word from inside the
		// test that is supposed to forbid it.
		literal: 'each fold — that is, each of the 10 groups the statements are split into',
		line: 0
	};
	ok(dialectOffences([undefinedUse]).length === 1, '(a4) “folds” alone is flagged');
	ok(
		dialectOffences([undefinedUse, definition]).length === 0,
		'(a4) the same use is allowed once the file defines the word'
	);

	// The harvests, exercised on the REAL sources. A class that scans nothing
	// passes everything, and both harvests have a hundred ways to return [].
	const mutations = [
		{
			file: '../src/lib/data/paper-error-f1.ts',
			harvest: harvestRenderStrings,
			from: 'error recall — of the errors we already know about',
			to: 'error recall — of the arm’s known errors'
		},
		{
			file: '../src/lib/data/paper-review-queue.ts',
			harvest: harvestRenderStrings,
			from: 'BayesianScorer, source+subtype refit',
			to: 'BayesianScorer arm, source+subtype refit'
		},
		// The two shapes a field-list scan cannot see, pinned so nobody rewrites
		// this class back into one: prose assembled in a LOCAL CONST and only later
		// interpolated into a `title`, and prose passed as a positional ARGUMENT.
		{
			file: '../src/lib/data/paper-table6-extended.ts',
			harvest: harvestRenderStrings,
			// Re-quoted 2026-07-29 with the same-day revert of "slices" to "folds":
			// the anchor has to quote what the module ships, and the invariant it
			// proves (prose assembled in a local const is still scanned) is unchanged.
			from: 'population SD ${fmtOwn(row.foldSd)} over ${row.foldCount} folds',
			to: 'pooled SD ${fmtOwn(row.foldSd)} over ${row.foldCount} folds'
		},
		{
			file: '../src/lib/data/paper-own-metric.ts',
			harvest: harvestRenderStrings,
			// RE-ANCHORED 2026-07-29. The old anchor quoted a band subtitle verbatim,
			// so trimming that subtitle broke the harness rather than the invariant —
			// the same brittleness this project has hit four times. Anchored now on the
			// shortest durable clause in the band, and the plant introduces the dialect
			// word rather than depending on the sentence around it. Re-quoted the same
			// day when "slices" went back to "folds", the field's own word.
			from: 'same statements, same folds, same published labels',
			to: 'same statements, same folds, same published labels per arm'
		},
		{
			file: '../src/lib/components/StatementErrorF1.svelte',
			harvest: harvestTemplateStrings,
			from: 'That score is error-class F1',
			to: 'That score is pooled error-class F1'
		},
		{
			file: '../src/routes/paper/+page.svelte',
			harvest: harvestTemplateStrings,
			from: 'a random forest on evidence-count features, re-measured',
			to: 'the paper’s model, re-measured'
		}
	];
	for (const mutation of mutations) {
		const source = read(mutation.file);
		ok(source.includes(mutation.from), `(a4) fixture text is still in ${mutation.file}`);
		const clean = dialectOffences(mutation.harvest(source));
		ok(clean.length === 0, `(a4) ${mutation.file} is clean before mutation`);
		const dirty = dialectOffences(
			mutation.harvest(source.replace(mutation.from, mutation.to))
		);
		ok(
			dirty.length === 1,
			`(a4) a dialect word planted in ${mutation.file} is caught (got ${dirty.length})`
		);
	}
	// The same planting, into a COMMENT of the same component, must NOT be caught —
	// the one exclusion big enough to hide the whole page if it were mispositioned.
	{
		const source = read('../src/lib/components/StatementErrorF1.svelte');
		const inComment = source.replace(
			'The margin at statement grain, named: error-class F1',
			'The margin at statement grain, named: pooled error-class F1'
		);
		ok(inComment !== source, '(a4) the comment fixture text is still there');
		ok(
			dialectOffences(harvestTemplateStrings(inComment)).length === 0,
			'(a4) a dialect word planted in a docblock is NOT reported — comments are ours'
		);
	}

	// And the exclusions are exercised too, because an exclusion that swallows the
	// whole file is how a green guard checks nothing. Each of these WOULD be a hit
	// if it were reader text, and each must be skipped for its own stated reason.
	const excluded = [
		["fail()'s message", `fail(ctx, 'names no arm on this figure: ${'${key}'}');`],
		['a context path', "budget(panel.display, N, `row[${panel.key}].display`)"],
		// Both carry a BANNED word on purpose: an exclusion fixture whose string is
		// clean proves nothing about the exclusion.
		['a frozen key', "const specs = [{ label: 'Gemma 4 26B arm', display: 'Gemma 4 26B reading' }];"],
		['an artifact expectation', "const spec = { artifactDisplay: 'Gemma 4 26B arm' };"],
		['a comment', "// the arm’s own tau, pooled over the panel\nconst x = 1;"],
		['an identifier', 'const arms = figure.lanes.map((lane) => lane.display);']
	];
	for (const [what, snippet] of excluded) {
		const offences = dialectOffences(harvestRenderStrings(snippet));
		ok(offences.length === 0, `(a4) does not read ${what} as reader text: ${offences[0]?.hit}`);
	}
	// …and the same harvest, on the same snippet with the string moved into a
	// rendering field, DOES flag it. Without this pair the exclusions above could
	// be hiding everything.
	ok(
		dialectOffences(harvestRenderStrings("const spec = { display: 'Gemma 4 26B arm' };")).length === 1,
		'(a4) the excluded snippets are excluded for their position, not because the harvest is blind'
	);
	console.log(
		`  proof: ${MUST_CATCH.length} banned terms + ${PRIOR_REGRESSIONS.length} of the last pass’s` +
			` own strings caught, ${MUST_ALLOW.length} legitimate strings` +
			` allowed, ${mutations.length + 1} shipped sources mutated, ${excluded.length} exclusions exercised`
	);
}

/**
 * The modules that carry prose TODAY. Their harvest may not come back empty: a
 * harvest that stops finding strings is the failure mode this whole class dies
 * of, and it dies silently and green.
 */
const MODULES_WITH_PROSE = [
	'paper-ap-decomposition.ts',
	'paper-belief-ladder.ts',
	'paper-deployed-baseline.ts',
	'paper-error-f1.ts',
	'paper-literal.ts',
	'paper-own-metric.ts',
	'paper-per-evidence.ts',
	'paper-review-queue.ts',
	'paper-robustness.ts',
	'paper-table6-extended.ts'
];

let scannedStrings = 0;
for (const relative of JARGON_TEMPLATES) {
	const source = read(relative);
	const strings = harvestTemplateStrings(source);
	scannedStrings += strings.length;
	// NON-VACUITY, per file: a template that yields no prose is a broken harvest,
	// not a clean file. Every one of these has a paragraph in it.
	ok(strings.length > 0, `${relative}: the template harvest found no reader text at all`);
	reportOffences(relative.replace(/^.*\//, ''), dialectOffences(strings));
}
let scannedModuleStrings = 0;
for (const name of JARGON_MODULES) {
	const source = read(`../src/lib/data/${name}`);
	const strings = harvestRenderStrings(source);
	scannedStrings += strings.length;
	scannedModuleStrings += strings.length;
	if (MODULES_WITH_PROSE.includes(name)) {
		ok(strings.length > 0, `${name}: the rendering-field harvest came back empty`);
	}
	reportOffences(name, dialectOffences(strings));
}
console.log(
	`  ${JARGON_TEMPLATES.length} templates + ${JARGON_MODULES.length} data modules,` +
		` ${scannedStrings} reader strings scanned` +
		` (${scannedModuleStrings} of them written in the data modules, where the dialect survived)`
);

// ---------------------------------------------------------------------------
// (a5) THE SHIPPED HALF OF A PROSE TWIN NEVER LEAVES THE AUDIT BOUNDARY
//
// Class (a4) above can only see a string that EXISTS in the source. Every banned
// word that survived three "clean" sweeps arrived at RUNTIME, off sha-pinned
// artifact JSON — "tau = the smallest of the arm's own distinct scores…",
// "paired fold-stratified bootstrap over the paper's own out-of-fold fold
// assignment". No static scan will ever read those, and pretending otherwise is
// how the last three sweeps came back green with the dialect on the screen.
//
// So this class does not scan the text. It scans the CHANNEL. Every loader field
// carrying artifact prose is now a `ShippedProse` twin — `{ shipped, plain }`,
// `shipped` byte-identical to the artifact, `plain` authored in the loader and
// therefore visible to (a4). The FIELD ACCESS that chooses between them is in the
// source even though the string is not, so the rule is checkable:
//
//   a /paper component may name `shipped` only if it IS the audit boundary.
//
// WHAT COUNTS AS CODE, and why the rule needs no exceptions. Only two regions of
// a `.svelte` file can read a field: a `<script>` body and a `{…}` expression.
// Control-flow tags are INCLUDED here where class (a) excludes them, because the
// defect is different: `{#each rows as { shipped }}` destructures the audit half
// into every row of a loop, and `{@html row.shipped}` draws it. Everything else
// in the file is text a reader sees, and text is allowed to contain the English
// word — "Caveats, as shipped" is a heading on this page today, and a rule that
// flagged it would be switched off within the week. Comments are stripped first:
// a docblock explaining which half to render is not a render.
//
// WHAT THIS DOES NOT PROVE, stated here rather than left to be discovered. It
// guards the channel a component can reach, NOT the whole path from artifact to
// reader. The loaders also publish FLAT aliases — `decisionRule: decisionRuleProse
// .shipped` — which are the same artifact bytes under a name with no `shipped` in
// it, and the render sites still read those. Those aliases are counted and named
// below, without gating, because emptying them is the render-site work of another
// node, not this class's; when the last one goes, the count prints 0 and the
// channel guard becomes the whole guarantee. A number that must reach zero and is
// printed every run is worth more than a rule nobody can turn green.
// ---------------------------------------------------------------------------
console.log('\n(a5) the shipped half of a prose twin stays behind the audit boundary');

/**
 * The ONE file allowed to name it: the page's single verification boundary,
 * "for verification against the shipped files". It may not exist yet — this
 * class runs before it lands, and an allowlist entry for a missing file gates
 * nothing, so its absence is reported rather than assumed away.
 */
const AUDIT_TRAIL_COMPONENTS = ['PaperAuditTrail.svelte'];
/**
 * The module that ASSEMBLES the boundary's contents. Its reads are the boundary
 * working, not a leak, so they are counted apart from the flat aliases below —
 * otherwise the number that "must reach 0" never could, and a target nobody can
 * hit is a target everybody stops reading.
 */
const AUDIT_TRAIL_MODULES = ['paper-audit-trail.ts'];
/** Every /paper surface the rule GATES: everything that is not the boundary. */
const GATED_COMPONENTS = COMPONENTS_ALL.filter(
	(name) => !AUDIT_TRAIL_COMPONENTS.includes(name)
);

/**
 * Blank `//` and `/* *\/` comments, preserving length and lines, STRING-AWARE in
 * both directions — and both directions have a real failure behind them:
 *   · a `//` inside `'https://…'` must not blank the rest of the line, or a read
 *     sitting after it disappears and the file reads as clean;
 *   · a `${…}` hole inside a template literal IS code, so skipping the literal
 *     whole — the obvious implementation — loses `` `${prose.shipped}` `` and,
 *     again, the file reads as clean. Holes are therefore re-entered, nested ones
 *     included.
 */
function blankJsComments(code) {
	const out = code.split('');
	const blank = (from, to) => {
		for (let i = from; i < to && i < out.length; i += 1) if (out[i] !== '\n') out[i] = ' ';
	};
	const scan = (start, end) => {
		let i = start;
		while (i < end) {
			const c = code[i];
			if (c === '"' || c === "'") {
				let j = i + 1;
				while (j < end) {
					if (code[j] === '\\') {
						j += 2;
						continue;
					}
					if (code[j] === c) break;
					j += 1;
				}
				i = j + 1;
				continue;
			}
			if (c === '`') {
				let j = i + 1;
				while (j < end) {
					if (code[j] === '\\') {
						j += 2;
						continue;
					}
					if (code[j] === '`') break;
					if (code[j] === '$' && code[j + 1] === '{') {
						let depth = 0;
						let k = j + 1;
						for (; k < end; k += 1) {
							if (code[k] === '{') depth += 1;
							else if (code[k] === '}') {
								depth -= 1;
								if (depth === 0) break;
							}
						}
						scan(j + 2, k);
						j = k + 1;
						continue;
					}
					j += 1;
				}
				i = j + 1;
				continue;
			}
			if (c === '/' && code[i + 1] === '/') {
				let j = i;
				while (j < end && code[j] !== '\n') j += 1;
				blank(i, j);
				i = j;
				continue;
			}
			if (c === '/' && code[i + 1] === '*') {
				const close = code.indexOf('*/', i + 2);
				const j = close === -1 || close > end ? end : close + 2;
				blank(i, j);
				i = j;
				continue;
			}
			i += 1;
		}
	};
	scan(0, code.length);
	return out.join('');
}

/** The same for a `.svelte` file: HTML comments too, then the script bodies. */
function commentBlanked(source) {
	const out = source.split('');
	for (const match of source.matchAll(/<!--[\s\S]*?-->/g)) {
		for (let i = match.index; i < match.index + match[0].length; i += 1) {
			if (out[i] !== '\n') out[i] = ' ';
		}
	}
	let text = out.join('');
	for (const match of [...text.matchAll(/<script\b[^>]*>([\s\S]*?)<\/script\s*>/gi)]) {
		const start = match.index + match[0].indexOf(match[1]);
		text = text.slice(0, start) + blankJsComments(match[1]) + text.slice(start + match[1].length);
	}
	return text;
}

/**
 * The byte ranges of a `.svelte` file that are CODE: every `<script>` body and
 * every brace-matched `{…}`, control-flow tags included. `<style>` is skipped
 * whole so its CSS braces are not mistaken for expressions.
 */
function codeRangesOf(source) {
	const ranges = [];
	let i = 0;
	while (i < source.length) {
		if (source[i] === '<') {
			const opener = /^<(script|style)\b[^>]*>/i.exec(source.slice(i, i + 400));
			if (opener) {
				const bodyStart = i + opener[0].length;
				const closer = new RegExp(`</${opener[1]}\\s*>`, 'i').exec(source.slice(bodyStart));
				const bodyEnd = closer ? bodyStart + closer.index : source.length;
				if (opener[1].toLowerCase() === 'script') ranges.push([bodyStart, bodyEnd]);
				i = closer ? bodyEnd + closer[0].length : source.length;
				continue;
			}
		}
		if (source[i] === '{') {
			let depth = 0;
			let j = i;
			for (; j < source.length; j += 1) {
				if (source[j] === '{') depth += 1;
				else if (source[j] === '}') {
					depth -= 1;
					if (depth === 0) break;
				}
			}
			ranges.push([i + 1, j]);
			i = j + 1;
			continue;
		}
		i += 1;
	}
	return ranges;
}

const SHIPPED_TOKEN = /(?<![\w$])shipped(?![\w$])/g;

/**
 * The KEY group of an `{#each … as row (key)}`, if this range is one.
 *
 * The one carve-out, and it is not a new one: class (a) already exempts key
 * expressions, for the reason that decides this class too — a key identifies a
 * row for the diffing algorithm and NEVER reaches the screen. `{#each caveats as
 * caveat (caveat.shipped)}` keys on stable artifact bytes and draws
 * `{caveat.plain}`; there is no channel to a reader in it, and a rule that
 * flagged it would be crying wolf on the shape the fix itself produces.
 *
 * The carve-out is the key GROUP and nothing else: `{#each list.shipped as row}`
 * iterates the audit half, `{#each rows as { shipped }}` destructures it into the
 * loop body, and both stay gated.
 */
function eachKeyRangeOf(source, from, to) {
	const body = source.slice(from, to);
	if (!/^\s*#each\b/.test(body)) return null;
	const end = body.replace(/\s+$/, '').length;
	if (body[end - 1] !== ')') return null;
	let depth = 0;
	for (let i = end - 1; i >= 0; i -= 1) {
		if (body[i] === ')') depth += 1;
		else if (body[i] === '(') {
			depth -= 1;
			if (depth === 0) return [from + i, from + end];
		}
	}
	return null;
}

/**
 * Every place a `.svelte` file's CODE names `shipped`. `figure.shippedResidual`
 * is not one of them — the token is bounded, and a compound is a different field.
 * Each hit says whether it sits in an `{#each}` key, which is exempt from the
 * gate and reported instead, so the exemption is visible rather than silent.
 */
function shippedReadsInComponent(source) {
	const blanked = commentBlanked(source);
	const ranges = codeRangesOf(blanked);
	const out = [];
	for (const match of blanked.matchAll(SHIPPED_TOKEN)) {
		const range = ranges.find(([from, to]) => match.index >= from && match.index < to);
		if (!range) continue;
		const key = eachKeyRangeOf(blanked, range[0], range[1]);
		// The SOURCE LINE, not the enclosing range: a `<script>` body is one range
		// and quoting it whole tells a reader nothing about where the read is.
		const lineStart = source.lastIndexOf('\n', match.index) + 1;
		const lineEnd = source.indexOf('\n', match.index);
		out.push({
			line: lineOf(blanked, match.index),
			keyPosition: key !== null && match.index >= key[0] && match.index < key[1],
			expression: source
				.slice(lineStart, lineEnd === -1 ? source.length : lineEnd)
				.trim()
				.slice(0, 90)
		});
	}
	return out;
}

/** The hits the gate acts on: everything a reader could actually meet. */
function shippedRendersInComponent(source) {
	return shippedReadsInComponent(source).filter((hit) => !hit.keyPosition);
}

/**
 * Every place a `.ts` module READS the shipped half — `x.shipped`, as opposed to
 * WRITING one, `shipped: text(…)`. A read in a loader is the laundering channel:
 * it copies artifact bytes into a field whose name no longer says so.
 */
function shippedReadsInModule(source) {
	const blanked = blankJsComments(source);
	const out = [];
	for (const match of blanked.matchAll(SHIPPED_TOKEN)) {
		const before = blanked.slice(0, match.index);
		if (!/[\w$)\]]\s*\??\.\s*$/.test(before)) continue;
		const lineStart = before.lastIndexOf('\n') + 1;
		const lineEnd = blanked.indexOf('\n', match.index);
		out.push({
			line: lineOf(blanked, match.index),
			expression: source
				.slice(lineStart, lineEnd === -1 ? source.length : lineEnd)
				.trim()
				.slice(0, 90)
		});
	}
	return out;
}

// --- (a5) IS NOT VACUOUS ----------------------------------------------------
// A guard whose detector cannot fire is worse than no guard, and three on this
// project have shipped exactly that way. So the detector is driven, in both
// directions, on fixtures AND on the real shipped sources.
{
	const reads = (svelte) => shippedRendersInComponent(svelte);

	const MUST_CATCH = [
		['a property read in the script', '<script>const s = figure.prose.claim.shipped;</script>'],
		['a property read in a mustache', '<p>{figure.prose.claim.shipped}</p>'],
		['an optional read', '<p>{figure.prose?.claim?.shipped}</p>'],
		['a destructure in a control-flow tag', '{#each rows as { shipped } (shipped)}<p>x</p>{/each}'],
		['an @html tag', '{@html row.shipped}'],
		['a renamed destructure', '<script>const { shipped: verbatim } = prose;</script>'],
		['a reader-facing attribute', '<p title={row.prose.shipped}>x</p>'],
		['an {#if} that branches on it', '{#if prose.shipped}<p>x</p>{/if}'],
		// The comment stripper is STRING-AWARE, and this is the fixture that proves
		// it: a naive `//` strip eats the rest of the line from inside a URL, and a
		// read after one on the same line would vanish — a false negative that reads
		// exactly like a clean file.
		['a read after a URL literal', "<script>const u = 'https://x/y'; const s = p.shipped;</script>"],
		['a read inside a template-literal hole', '<script>const s = `x ${p.shipped} y`;</script>'],
		[
			'a read after comment-looking text in a template literal',
			'<script>const s = `a // b`; const t = q.shipped;</script>'
		],
		// The carve-out is the KEY GROUP and nothing else. Each of these three sits
		// in an `{#each}` tag and each is a real channel to a reader.
		['iterating the audit half', '{#each list.shipped as row (row.id)}<p>x</p>{/each}'],
		['destructuring it in the binding', '{#each rows as { shipped } (id)}<p>x</p>{/each}'],
		[
			'a body render under an exempt key',
			'{#each rows as row (row.shipped)}<p>{row.shipped}</p>{/each}'
		]
	];
	for (const [what, snippet] of MUST_CATCH) {
		ok(reads(snippet).length > 0, `(a5) catches ${what}: ${JSON.stringify(snippet)}`);
	}

	// The other direction, and it matters as much: every one of these is on the
	// page today, and flagging any of them would get the class switched off.
	const MUST_ALLOW = [
		['the English word in a heading', '<h4>Caveats, as shipped</h4>'],
		['the word in a table header', '<th>interpolation credit (as shipped)</th>'],
		['the word in a quoted attribute', '<p title="results, as shipped">x</p>'],
		['a different field that starts the same way', '<p>{figure.shippedResidual}</p>'],
		['an identifier that starts the same way', '<script>const shippedRows = rows[0];</script>'],
		['an HTML comment', '<!-- render .plain, never prose.shipped -->'],
		['a line comment in the script', '<script>// prose.shipped is the audit half\n</script>'],
		['a block comment in the script', '<script>/** `whyProse.shipped === why`. */\n</script>'],
		['CSS that happens to brace', '<style>.shipped { color: red; }</style>'],
		// The carve-out itself, and it is on the page today: the fix keys the loop on
		// the stable artifact bytes and draws the restatement.
		['an {#each} keyed on the audit half', '{#each c as x (x.shipped)}<p>{x.plain}</p>{/each}'],
		[
			'the same with an index and a nested call',
			'{#each c as x, i (hash(x.shipped, i))}<p>{x.plain}</p>{/each}'
		]
	];
	for (const [what, snippet] of MUST_ALLOW) {
		const hits = reads(snippet);
		ok(hits.length === 0, `(a5) allows ${what}: flagged ${JSON.stringify(hits[0]?.expression)}`);
	}

	/**
	 * EVERY real component, MUTATED. A fixture proves the regex; only the real
	 * files prove the harvest reaches the template of each one — a `<style>` block
	 * that swallowed the rest of a file, or a brace-match that gave up early, would
	 * leave the class scanning a prefix and reporting success.
	 *
	 * The plant is derived from STRUCTURE, never from a sentence. Pinning it to a
	 * quoted render site was the first cut and it was wrong in a way this project
	 * keeps rediscovering: this rebuild exists to MOVE those sites, so the proof
	 * would have gone red the day the defect was fixed, and the fix would have been
	 * blamed. A guard must not punish its own remedy. So: open a new mustache at
	 * the first tag boundary after the script, which every `.svelte` file has.
	 */
	const plantShippedRead = (source) => {
		const afterScript = source.lastIndexOf('</script>');
		const boundary = source.indexOf('>', afterScript + '</script>'.length);
		if (afterScript === -1 || boundary === -1) return source;
		return `${source.slice(0, boundary + 1)}\n{row.prose.note.shipped}\n${source.slice(boundary + 1)}`;
	};
	let mutated = 0;
	for (const name of GATED_COMPONENTS) {
		const source = read(`${COMPONENT_DIR}${name}`);
		const planted = plantShippedRead(source);
		ok(planted !== source, `(a5) a read can be planted in ${name} at all`);
		const before = shippedRendersInComponent(source).length;
		eq(before, 0, `(a5) ${name} is clean before mutation`);
		eq(
			shippedRendersInComponent(planted).length,
			before + 1,
			`(a5) a shipped read planted in ${name}'s real template is caught`
		);
		mutated += 1;
	}
	// The allowlist is a rule with a live consequence, not a comment: the SAME
	// planted source is reported under any other name and exempt under this one.
	ok(
		AUDIT_TRAIL_COMPONENTS.length === 1,
		'(a5) exactly one file is allowed to name the shipped half'
	);
	console.log(
		`  proof: ${MUST_CATCH.length} shapes of read caught, ${MUST_ALLOW.length} legitimate uses` +
			` allowed, ${mutated} real component templates mutated`
	);
}

// --- (a5) THE GATE ----------------------------------------------------------
const keyExempt = [];
for (const relative of [
	...GATED_COMPONENTS.map((name) => `${COMPONENT_DIR}${name}`),
	PAPER_ROUTE
]) {
	const name = relative.replace(/^.*\//, '') === '+page.svelte' ? 'the /paper route' : relative.replace(/^.*\//, '');
	const source = read(relative);
	const hits = shippedReadsInComponent(source);
	for (const hit of hits.filter((entry) => entry.keyPosition)) {
		keyExempt.push(`${name} line ${hit.line}`);
	}
	const rendered = hits.filter((hit) => !hit.keyPosition);
	ok(
		rendered.length === 0,
		`${name} reads the shipped half of a prose twin — it is not the audit boundary:\n` +
			rendered.map((hit) => `      line ${hit.line}: ${hit.expression}`).join('\n')
	);
}
if (keyExempt.length > 0) {
	console.log(
		`  ${keyExempt.length} {#each} key(s) name the audit half and are EXEMPT — a key identifies` +
			' a row and is never drawn, the same carve-out class (a) makes for join keys:' +
			`\n      ${keyExempt.join(' · ')}`
	);
}
{
	const onDisk = new Set(readdirSync(fileURLToPath(new URL(COMPONENT_DIR, import.meta.url))));
	const present = AUDIT_TRAIL_COMPONENTS.filter((name) => onDisk.has(name));
	// AN ALLOWLIST IS A HOLE UNLESS ITS ENTRY IS CHECKED. The one exempt file is
	// exempt for a reason — it IS the audit trail — so once it is on disk it must
	// actually read the shipped half. A boundary that shows no verbatim text is
	// not a boundary, it is a hole in this class shaped exactly like one.
	let reading = 0;
	for (const name of present) {
		const reads = shippedRendersInComponent(read(`${COMPONENT_DIR}${name}`)).length;
		if (reads > 0) reading += 1;
		ok(
			reads > 0,
			`${name} is the one file allowed to read the shipped half and reads none — ` +
				'either it is not the audit boundary, or the exemption is unearned'
		);
	}
	console.log(
		`  ${GATED_COMPONENTS.length + 1} /paper surfaces gated,` +
			` ${present.length} of ${AUDIT_TRAIL_COMPONENTS.length} audit boundaries on disk,` +
			` ${reading} of them reading verbatim text` +
			`${present.length === AUDIT_TRAIL_COMPONENTS.length ? '' : ' (the rest allowlisted for the node building them)'}`
	);
}

// --- (a5) REPORTED, NOT GATED: the flat aliases the render sites still read ---
{
	let aliases = 0;
	let boundaryReads = 0;
	const perModule = [];
	for (const name of JARGON_MODULES) {
		const hits = shippedReadsInModule(read(`../src/lib/data/${name}`));
		if (hits.length === 0) continue;
		if (AUDIT_TRAIL_MODULES.includes(name)) {
			boundaryReads += hits.length;
			continue;
		}
		aliases += hits.length;
		perModule.push(`${name.replace(/^paper-|\.ts$/g, '')} ${hits.length}`);
	}
	// The detector, exercised — otherwise "0 aliases" and "the scan is broken" print
	// the same line, which is precisely how a green guard checks nothing.
	eq(
		shippedReadsInModule('const x = { decisionRule: p.shipped };').length,
		1,
		'(a5) the alias scan sees a read'
	);
	eq(
		shippedReadsInModule("const x = { shipped: text(o.rule, 'ctx') };").length,
		0,
		'(a5) the alias scan does not read a twin being BUILT as a leak'
	);
	eq(
		shippedReadsInModule('/** `noteProse.shipped === note`. */\nconst x = 1;').length,
		0,
		'(a5) the alias scan does not read a docblock as a leak'
	);
	console.log(
		`  REPORTED, not gated: ${aliases} flat alias(es) of a shipped half still published by the` +
			` loaders — artifact bytes under a name with no \`shipped\` in it, which the render sites` +
			` read today. This number must reach 0; when it does, (a5) is the whole guarantee.` +
			`\n      ${perModule.join(' · ')}` +
			`\n      (${boundaryReads} further read(s) in ${AUDIT_TRAIL_MODULES.join(', ')} are the` +
			' boundary assembling its own verbatim text, and are not aliases)'
	);
}

// ---------------------------------------------------------------------------
// (a6) NOTHING A READER SEES NAMES A METHOD OR A GROUP BY WHOSE IT IS
//
// The 2023 INDRA assembly paper is OUR OWN LAB'S. /paper was nevertheless written
// as a contest between two camps — "their methods and ours on one axis", "their
// measure, not ours", "that agreement is the only thing that earns our models a
// place on their axis", "ranks 1, 2, 3 are ours" — which frames a lab measuring
// its own prior work as a supplicant asking permission. The same habit named
// every method RELATIVELY: "the paper's model", "the paper's own metric", "the
// paper's folds", "paper metric". A reader who has not read that paper learns
// nothing from any of those, which is the whole defect: a possessive is not a
// name.
//
// THE RULE. Name the thing; use the citation only where provenance is the point.
//   the random forest on evidence-count features    not  "the paper's model"
//   trapezoidal PR-AUC / average precision          not  "the paper's metric"
//   noisy-OR over per-source reliabilities          not  "their estimator"
//   the 1,689 statements / the benchmark            not  "the paper's panel"
//   the 10 folds                                    not  "the paper's folds"
//   published · newly scored                        not  THEIRS · OURS
// "Published in 2023" is ALLOWED, and deliberately: the FACT of publication is
// what makes a row a fixed reference point, and saying so carries information.
// "The paper's X" as a NAME carries none — which is why the band headers lost
// nothing when they went from "THEIRS — same evidence as ours (15)" to
// "published (15)" and got shorter.
//
// WHY IT IS A SEPARATE CLASS FROM (a4). (a4) bans a private VOCABULARY — words a
// reader has never been taught (arm, tau, slice, max-t). This one bans a private
// STANCE, and the stance survives a vocabulary sweep untouched: "our models" and
// "the published rows" use nothing but ordinary English. Four sweeps of /paper
// left 159 instances of it standing.
//
// WHAT IS SCANNED — the same surface (a4) reads, discovered the same way
// (readdirSync over the data modules, the route's import graph over the
// components), PLUS one surface (a4) does not read and should:
//   · COMPONENT <script> BODIES. (a4) strips them, on the reasoning that only a
//     template renders. That is false for this page: `KIND_BADGE = { paper: "the
//     paper's code", port: 'our rewrite' }` in PaperLiteralComparison's script is
//     a badge under every mark in the figure, and `KIND_TITLE` beside it is its
//     tooltip. Both were invisible to every sweep, which is exactly how the
//     dialect survived in the data modules three times running.
//
// WHAT IS NOT SCANNED, each exclusion exercised below rather than asserted:
//   · THE SHIPPED HALF OF A TWIN. `{ shipped, plain }` — `shipped` is byte-
//     identical to a sha-pinned artifact and MUST stay verbatim; class (a5)
//     keeps it behind the audit boundary. This class reads the `plain` half,
//     which is the one a reader gets. Asking for `shipped` to be reworded would
//     be asking someone to edit artifact bytes to satisfy a linter.
//   · JOIN KEYS. `name: 'Paper RF 2k-d13 + Type/#PMIDs/promoter'` is the string
//     that ADDRESSES the shipped row (`REVIEW_QUEUE_ARM_PROSE` is keyed by it);
//     the `display` beside it, 'RF 2k-d13 + Type/#PMIDs/promoter', is what draws.
//     A string this module also uses as a quoted object key is a key, full stop.
//     This class must never be the reason a join key moves — that has regressed
//     seven times. Same doctrine as (a): the fix is the display beside the key.
//   · INTERNAL TOKENS. `pushBand('ours', …)`, `'ours-reproduction'`, the union
//     member `'ours'` — a bare lowercase token with no whitespace, outside a
//     rendering field, is an id. The page draws `newly scored (5)`; nobody draws
//     `ours`. The moment such a token is UPPERCASE, or sits under a rendering
//     field, it is a name again and it is flagged.
//   · everything (a4) already excludes: comments, `fail()` text, addresses,
//     artifact-shaped names, and the frozen fields.
//
// WHAT IS DELIBERATELY NOT AN OFFENCE, said out loud so nobody "tightens" it into
// noise and gets the class switched off:
//   · FIRST PERSON ABOUT OUR OWN PROCEDURE. "We read it; we do not re-derive it",
//     "we lower each model's cutoff", "leaving them out is our label revision".
//     These disclose who did something — which is information — rather than
//     naming a method by its owner. `revision` is therefore not in the noun list
//     and `we`/`our` alone are not patterns.
//   · ORDINARY ANAPHORA. "error-class F1 — their harmonic mean", "the 10 fold
//     results around their own average", "statements matched by their sorted
//     ids". `their` is normal English; only `their` + a noun that NAMES a method,
//     measure, artefact or group is the defect.
//   · THE CITATION. "the 2023 INDRA assembly paper", "Table 6 of the 2023
//     paper", "published in 2023" — provenance, not a name.
//
// WHAT THIS CANNOT CATCH, in the register the rest of this file uses:
//   · PROSE THAT ARRIVES AT RUNTIME off the artifact. A shipped sentence still
//     says "our arms" and "the paper's own estimator" in the JSON; the guarantee
//     that a reader never meets it is class (a5)'s channel rule, not this one.
//     This class is the static half of the same job.
//   · THE SERVER LOADERS. `src/lib/server/paper-*.ts` write the `reason` string a
//     figure prints when it GATES, and that string is on the page — it is not
//     validation text nobody sees. They sit outside the surface this sweep
//     scans (components, the route, `$lib/data/paper-*.ts`), so the two that
//     said "Our arms are unavailable" were fixed by hand rather than by a rule.
//     Widening to `$lib/server` is a coherent next node; it needs the gate-text
//     exclusions thought through first, since those files also throw.
// ---------------------------------------------------------------------------
console.log('\n(a6) no reader-facing string names a method or a group by whose it is');

/**
 * Nouns that NAME a method, a measure, an artefact or a group. A possessive in
 * front of one of these is the defect; a possessive in front of anything else is
 * English. Deliberately excludes `score`, `number`, `best`, `evidence` and
 * `revision`: each of those is anaphoric or describes an ACT, and flagging them
 * would flood the class with sentences that are already correct.
 */
const OWNED_THING =
	'(?:' +
	// Multi-word names first, so "our grounding rules" is caught as the NAME it is
	// rather than needing `grounding` in the modifier list below.
	'grounding\\s+rules|combin(?:ing|ation)\\s+rules?|reading\\s+steps?|random\\s+forests?' +
	'|score\\s+cutoffs?|belief\\s+models?' +
	'|estimators?|models?|metrics?|measures?|panels?|benchmarks?|folds?|slices?' +
	'|tables?|code|codebase|methods?|rows?|labels|data|statements|axis|axes' +
	'|forests?|RFs?|rankings?|scorers?|readers?|arms?|ports?|rewrites?' +
	'|re-?runs?|re-?implementations?|implementations?|versions?|features?' +
	'|curators?|authors?|rules|prompts?|leads?|margins?)';
/** Straight, curly and backtick apostrophes — all three ship on this page. */
const APOSTROPHE = "[’'`´]";
/**
 * What may sit BETWEEN the possessive and the noun. This is where the defect
 * hides, not where it stops: the first cut of this class read "our models" and
 * walked straight past "our reading models", the headline of a whole figure.
 *
 * WIDENED 2026-07-29, after probing the class with phrasings it was NOT written
 * from — the discipline this project arrived at the hard way, four guards having
 * shipped green while checking nothing. One provenance word was not enough. Every
 * one of these ran clean through the previous version:
 *     "our best model beats the published row"
 *     "our four reading models all clear it"
 *     "our whole average-precision lead"          ← was LIVE in TieInflation
 *     "their strongest row is the promoter forest"
 *     "their 10 folds, assigned once"
 *     "their own 10 folds"
 * A guard keyed on the exact strings its author had in hand catches its author
 * and nobody else. So the list now also takes quantity and rank words, and
 * repeats up to three times — "their own 10 published rows" is one phrase.
 *
 * STILL A CLOSED LIST, never `\w+`, and the difference is two false positives
 * that were live for one run of this class: "the share of their evidence curators
 * marked correct" and "Their best cutoff is the one they would have used
 * untuned". Both are ordinary anaphora — `their` pointing at a plural already
 * named. `evidence` is therefore not here, and `cutoff` is not in OWNED_THING, so
 * both stay clean with `best` on the list. A class that flags those is a class
 * somebody switches off, which is the failure this whole file is written against.
 */
const MODIFIER_WORD =
	'(?:own|whole|entire|full|only|other|remaining' +
	// quantity and rank: "our four reading models", "their 10 folds", "our best model"
	'|\\d+|one|two|three|four|five|six|ten|first|second|third|last' +
	'|best|worst|strongest|weakest|top|leading|lowest|highest' +
	// THE MEASURE NAMES THIS PAGE USES, which is how "our whole average-precision
	// lead" got past a modifier list that already had `whole` on it: the noun was
	// `lead` and the word in front of it was a measure, not a provenance word. A
	// closed list of the page's own measures, because the alternative — one
	// arbitrary `\\w+` — is exactly what re-flags "their evidence curators".
	'|average[-\\s]precision|precision|recall|error[-\\s]?class|error[-\\s]?F1|F1' +
	'|AUROC|PR[-\\s]?AUC|AP|belief|trapezoidal|stepped|noisy[-\\s]?OR|calibration' +
	// provenance: where the thing came from
	'|released|published|printed|reported|fitted|unfitted|re-?run' +
	'|re-?implemented|reading|scored|shipped|literal|semantic|original|new|newly)';
const MODIFIER = `(?:\\s+${MODIFIER_WORD}){0,3}`;

const OWNERSHIP_TERMS = [
	{
		term: "the paper's <anything>",
		pattern: new RegExp(`\\bpapers?${APOSTROPHE}s?\\b(?!\\s*$)`, 'i'),
		plain:
			'name it: the random forest on evidence-count features · trapezoidal PR-AUC · ' +
			'the 1,689 statements — or say “published in 2023”, which is a fact, not a name'
	},
	{
		term: 'paper <method>',
		pattern:
			/\bpaper[-\s](?:RFs?|models?|metrics?|panels?|methods?|rows?|code|labels?|estimators?|forests?|tables?|measures?|axis|folds?|slices?|statements?|benchmarks?|scorers?)\b/i,
		plain: 'the published <thing>, or the thing’s own name'
	},
	{
		term: 'their <method>',
		pattern: new RegExp(`\\btheir${MODIFIER}\\s+${OWNED_THING}\\b`, 'i'),
		plain: 'the published <thing> — or name it'
	},
	{
		term: 'our <method>',
		pattern: new RegExp(`\\bour${MODIFIER}\\s+${OWNED_THING}\\b`, 'i'),
		plain: 'the newly scored <thing>, or the thing’s own name'
	},
	{
		term: 'ours / theirs',
		pattern: /\b(?:ours|theirs)\b/i,
		plain: '“newly scored” / “published”'
	},
	{
		// TWO SIDES, WRITTEN SEPARATELY, and the asymmetry is the point. "us" is
		// only ever the camp, so it takes the wider verb list. "them" is ordinary
		// English for a plural already named — "give them all the same job", "the
		// worry behind them" — and both of those are on the page and correct, so
		// `them` takes only the verbs that CONTRAST two parties. A rule that cried
		// wolf on the anaphor would be switched off inside a week.
		//
		// `between|puts|places|ranks|lifts|pushes|separates` added 2026-07-29 off the
		// same probe that widened MODIFIER: "the straight line puts us ahead of the
		// published row" and "the margin between us and the published rows" both ran
		// clean through the first verb list, which had been written from the
		// sentences its author was looking at. `us` is still never banned bare —
		// "the figure tells us three things" is English, not a camp.
		term: 'us / them',
		pattern:
			/\b(?:than|beats?|flatters?|favou?rs?|rewards?|against|versus|vs\.?|earns?|costs?|gives?|hands?|between|puts?|places?|ranks?|lifts?|pushes?|separates?|ahead\s+of|behind)\s+(?:to\s+)?us\b|\b(?:than|beats?|flatters?|favou?rs?|against|versus|vs\.?)\s+them\b|\b(?:we|they)\s+(?:beat|beats|won|win|lost|lose)\b/i,
		plain: 'say which method and which measure, not which camp'
	}
];

/**
 * Quoted object KEYS in a module — multi-word ones included, which is the whole
 * point: `'Paper RF 2k-d13 + Type/#PMIDs/promoter':` addresses shipped bytes.
 * Anchored on `{`, `,`, `;` or a line start so a TERNARY's colon
 * (`x ? "the paper's code" : y`) is never mistaken for a key and cannot launder
 * prose out of this class.
 */
function quotedKeysIn(source) {
	return new Set(
		[...source.matchAll(/(?:^|[{,;])\s*['"]([^'"\n]{2,})['"]\s*:/gm)].map((match) => match[1])
	);
}

/** A bare lowercase token outside a rendering field: an id, not a name. */
function isInternalToken(entry) {
	const bare = stripInterpolations(entry.literal, '').trim();
	if (bare === '' || /\s/.test(bare)) return false;
	if (bare !== bare.toLowerCase()) return false;
	return entry.field !== 'text' && classifyField(entry.field) !== 'render';
}

/** The class. `strings` are the reader strings of ONE file, as (a4) harvests them. */
function ownershipOffences(strings, joinKeys = new Set()) {
	const offences = [];
	for (const entry of strings) {
		if (entry.shippedHalf) continue;
		if (joinKeys.has(entry.literal.trim())) continue;
		if (isInternalToken(entry)) continue;
		const text = stripInterpolations(entry.literal);
		for (const owned of OWNERSHIP_TERMS) {
			const hit = text.match(owned.pattern);
			if (hit) offences.push({ ...entry, text, hit: hit[0], term: owned.term, plain: owned.plain });
		}
	}
	return offences;
}

function reportOwnership(where, offences) {
	ok(
		offences.length === 0,
		`${where} names something by whose it is:\n` +
			offences
				.map(
					(o) =>
						`      line ${o.line} ${o.field || 'string'}: “${o.hit.replace(/\s+/g, ' ')}” → ${o.plain}\n` +
						`        ${JSON.stringify(windowAround(o.text, o.hit))}`
				)
				.join('\n')
	);
}

/**
 * The hit IN ITS SENTENCE. Printing the first 96 characters of the string instead
 * — the obvious thing — hides the offence whenever it sits in a long paragraph,
 * and a report you cannot act on gets the class switched off.
 */
function windowAround(text, hit) {
	const flat = text.replace(/\s+/g, ' ').trim();
	const at = flat.indexOf(hit.replace(/\s+/g, ' '));
	if (at === -1) return flat.slice(0, 96);
	const from = Math.max(0, at - 44);
	const to = Math.min(flat.length, at + hit.length + 44);
	return `${from > 0 ? '…' : ''}${flat.slice(from, to)}${to < flat.length ? '…' : ''}`;
}

/**
 * Every `<script>` body of a `.svelte` file, with the line it starts on — prose
 * is authored in them, and a report has to be able to say where.
 */
function scriptBodiesOf(source) {
	return [...source.matchAll(/<script\b[^>]*>([\s\S]*?)<\/script\s*>/gi)].map((match) => ({
		body: match[1],
		firstLine: lineOf(source, match.index + match[0].indexOf(match[1]))
	}));
}
/** Template text + script strings: everything a component can put on a screen. */
function harvestComponentStrings(source) {
	const out = harvestTemplateStrings(source);
	for (const { body, firstLine } of scriptBodiesOf(source)) {
		for (const entry of harvestRenderStrings(body)) {
			out.push({ ...entry, line: entry.line + firstLine - 1 });
		}
	}
	return out;
}

// --- (a6) IS NOT VACUOUS ----------------------------------------------------
// Four guards on this project have passed green while checking nothing, so this
// class is probed with THE PHRASINGS THAT ESCAPED — the strings four sweeps left
// on the screen, and the ones a literal-keyed guard would have missed — before
// anything trusts it.
{
	const caught = (text, keys) =>
		ownershipOffences([{ field: 'fixture', literal: text, line: 0 }], keys);

	// Every string here was RENDERING on /paper after three passes at the dialect,
	// or is the shape that got past a guard keyed on literal text.
	const ESCAPED = [
		['OURS band header', 'OURS — scored on the paper’s identical panel, labels and folds'],
		['THEIRS band header', 'THEIRS — same evidence as ours (15)'],
		['their measure', 'their measure, not ours'],
		['the supplicant frame', 'that agreement is the only thing that earns our models a place on their axis'],
		['ranks are ours', 'ranks 1, 2, 3 are ours'],
		['no leading rank', 'no leading rank is ours'],
		['their code, their data', 'Their code, their data, our run — and their published row comes back.'],
		['the paper’s own estimator', 'on the paper’s own estimator, population SD over 10 folds'],
		['our arms', 'our arms were never published, so they cannot carry a published value'],
		['our re-run of their code', 'open squares are our re-run of their code'],
		['it is ours, not the paper’s', 'it is ours, not the paper’s: INDRA’s count features include more'],
		['our port of their method', 'our own rewrite of the paper’s method, checked against their code'],
		['the paper’s measure flatters us', 'The paper’s own measure flatters us.'],
		['the paper’s N methods', 'Where the paper’s 59 reported methods sat'],
		['paper methods, no possessive', 'one of the 14 paper methods we re-ran'],
		['paper rows, no possessive', 'The paper rows span different eligible statement sets'],
		['a change to their labels', 'a change we made to their labels, not a cleaner reading of their data'],
		['their folds', 'the 10 folds they assigned, so every redraw keeps their fold make-up'],
		['our panel', 'our model, scored on the same panel as ours'],
		['the paper’s folds', 'the population standard deviation over the paper’s 10 folds'],
		['paper metric, the operator’s own example', 'ranked on the paper metric'],
		['the straight apostrophe', "the paper's own released code"],
		['the backtick apostrophe', 'the paper`s own released code'],
		['their estimator', 'so their estimator hands it the interpolation credit'],
		['our models', 'our models give many statements the same score'],
		['our re-run', 'our re-run of the released code'],
		['flatters us', 'this scoring flatters us'],
		['we beat', 'It also favours the model we beat'],
		['favours them', 'the trapezoidal estimator favours them'],
		['ahead of us', 'no published row is ahead of us'],
		// AND THE PHRASINGS THAT ESCAPED THIS CLASS'S OWN FIRST CUT. Every one of
		// these was live on /paper while an earlier version of the patterns above
		// ran green over it, which is the only evidence worth anything about a
		// guard: not that it catches what it was written from, but that it catches
		// what got past the last person who thought they were done.
		['a modifier between the possessive and the noun', 'Interpolation hands our reading models 0.02 they did not earn'],
		['their + provenance word', 'the four belief models re-running their released code'],
		['a multi-word method name', 'That is a property of OUR grounding rules, not of any reading model'],
		['a multi-word method name, second', 'the 96 statements our reading step scored 0.65'],
		['a verb the first verb list missed', 'the straight line then rewards us for an ordering we never produced'],
		['the possessive in an aria-label', "the 2023 paper's belief formula: what it promises against what it delivers"],
		['a provenance footer', 'Generated by paper_literal_models.py · paper code'],
		['prose authored in a component <script>', "run here with the 2023 paper's own released code"],
		// AND THE PHRASINGS THAT ESCAPED THE SECOND CUT. Probed 2026-07-29 by
		// writing sentences this class was NOT written from — the only test of a
		// guard that means anything, since a guard keyed on its author's own strings
		// catches its author and nobody else. All eight ran GREEN through the
		// version above them, and the third was live in TieInflation.svelte until a
		// shard fixed it BY HAND, which is what a working guard is supposed to spare
		// you. They are the reason MODIFIER now takes quantity and rank words and
		// the us/them verb list takes contrast verbs.
		['our + a superlative', 'our best model beats the published row'],
		['our + a count', 'our four reading models all clear it'],
		['our whole X — was live on the page', 'our whole average-precision lead'],
		['their + a superlative', 'their strongest row is the promoter forest'],
		['their + a count', 'their 10 folds, assigned once'],
		['their + own + a count', 'their own 10 folds'],
		['us after a verb the first list missed', 'the straight line puts us ahead of the published row'],
		['us in a between-contrast', 'the margin between us and the published rows']
	];
	for (const [what, sentence] of ESCAPED) {
		ok(caught(sentence).length > 0, `(a6) catches ${what}: ${JSON.stringify(sentence.slice(0, 72))}`);
	}

	// The other half of a useful class: the legitimate register. If this list ever
	// goes red the class is crying wolf and will be switched off within the week.
	const MUST_ALLOW = [
		'published (15)',
		'newly scored (5)',
		'same statements, same folds, same published labels',
		'open squares are the re-run of the published code',
		'a random forest on evidence-count features — 2,000 trees, depth 13',
		'trapezoidal PR-AUC, averaged over 10 folds — the published measure',
		'average precision, the same area computed step-wise',
		'noisy-OR over per-source reliabilities, nothing fitted',
		'the 2023 INDRA assembly paper · scored exactly as published',
		'Table 6 of the 2023 paper, extended',
		'The headline result published in 2023 — the best fitted random forest',
		'two authors of the 2023 paper labelled 5,379 statements',
		'the labels released in 2023, unmodified',
		'the random forest released with the 2023 paper, never deployed',
		// Ordinary anaphora, all four shipping on /paper today.
		'error-class F1 — their harmonic mean, so it always lands between the two marks',
		'the spread of the 10 fold results around their own average',
		'the statements matched by their sorted statement ids',
		'these statements are labelled WRONG and their evidence review was never finished',
		'the reading models can only be operated at a few discrete points. Their lowest score is 0.65',
		// The two anaphors that one loosening of the modifier rule DID flag, kept
		// here so the loosening cannot come back without being noticed.
		'the share of their evidence curators marked correct here runs from 61% to 74%',
		'Their best cutoff is the one they would have used untuned',
		// First person about our own procedure — a disclosure, not a name.
		'We read it; we do not re-derive it',
		'we lower each model’s cutoff until the same share of errors is caught',
		'leaving them out is our label revision — a change made here, not a cleaner reading',
		'The reading model we actually ship'
	];
	for (const sentence of MUST_ALLOW) {
		const offences = caught(sentence);
		ok(
			offences.length === 0,
			`(a6) allows ${JSON.stringify(sentence.slice(0, 56))}: flagged ${offences
				.map((o) => `“${o.hit}”`)
				.join(', ')}`
		);
	}

	// The three exclusions, each exercised on the shape it exists for — and each
	// fixture carries a REAL offence, because an exclusion fixture whose string is
	// clean proves nothing about the exclusion.
	eq(
		ownershipOffences(harvestRenderStrings("const s = { shipped: 'our arms, on the paper’s panel', plain: 'the newly scored models, on the same statements' };")).length,
		0,
		'(a6) does not ask a sha-pinned shipped half to be reworded'
	);
	eq(
		ownershipOffences(harvestRenderStrings("const s = { plain: 'our arms, on the paper’s panel' };")).length,
		2,
		'(a6) reads the plain half of the same twin — the exclusion is positional, not blind'
	);
	{
		const module = "const S = [{ name: 'Paper RF 2k-d13', display: 'RF 2k-d13' }];\nconst P = {\n\t'Paper RF 2k-d13': { plain: 'the published random forest' }\n};";
		eq(
			ownershipOffences(harvestRenderStrings(module), quotedKeysIn(module)).length,
			0,
			'(a6) does not demand a join key be renamed'
		);
		eq(
			ownershipOffences(harvestRenderStrings(module)).length,
			1,
			'(a6) would flag that same string as prose — it is spared for being a key, not for being invisible'
		);
	}
	eq(
		ownershipOffences(harvestRenderStrings("const bands = ['ours', 'published-comparable'];")).length,
		0,
		'(a6) does not read a bare lowercase id as a name'
	);
	eq(
		ownershipOffences(harvestRenderStrings("const band = { display: 'ours' };")).length,
		1,
		'(a6) reads the same token as a name once it sits under a rendering field'
	);
	eq(ownershipOffences([{ field: 'text', literal: 'OURS', line: 0 }]).length, 1, '(a6) reads a drawn OURS');

	// THE HARVESTS, ON THE REAL SOURCES. A class that scans nothing passes
	// everything; these plant the defect in shipped files and require it to be
	// caught, including in the two positions no earlier sweep could see: a
	// component <script> const, and the plain half of a twin.
	const plants = [
		{
			file: '../src/routes/paper/+page.svelte',
			harvest: harvestComponentStrings,
			from: 'more wrong statements than the random forest does',
			to: 'more wrong statements than their model does'
		},
		{
			file: '../src/lib/components/PaperOwnMetric.svelte',
			harvest: harvestComponentStrings,
			from: 'published measure, unchanged',
			to: 'their measure, not ours'
		},
		{
			// The surface (a4) cannot see: prose authored in a component's script.
			file: '../src/lib/components/PaperLiteralComparison.svelte',
			harvest: harvestComponentStrings,
			from: "llm: 'language model'",
			to: "llm: 'our model'"
		},
		{
			file: '../src/lib/data/paper-own-metric.ts',
			harvest: harvestRenderStrings,
			from: 'newly scored model, on the same statements',
			to: 'our model, on the same statements'
		},
		{
			// A PLAIN half of a twin, which is where the register actually lives.
			file: '../src/lib/data/paper-robustness.ts',
			harvest: harvestRenderStrings,
			from: 'the labels released in 2023',
			to: 'the paper’s own labels'
		}
	];
	for (const plant of plants) {
		const source = read(plant.file);
		ok(source.includes(plant.from), `(a6) plant anchor is still in ${plant.file}`);
		const keys = quotedKeysIn(source);
		eq(
			ownershipOffences(plant.harvest(source), keys).length,
			0,
			`(a6) ${plant.file} is clean before the plant`
		);
		ok(
			ownershipOffences(plant.harvest(source.replace(plant.from, plant.to)), keys).length > 0,
			`(a6) a possessive planted in ${plant.file} is caught`
		);
	}
	// …and the same word planted in a COMMENT of the same file is NOT caught. This
	// file's own docblocks say "the paper's model" a dozen times; a class that read
	// comments would be unusable, and one that read nothing looks identical.
	{
		const source = read('../src/lib/components/PaperTable6Extended.svelte');
		const inComment = source.replace(
			"draw the artifact's own bytes",
			"draw the paper's own bytes"
		);
		ok(inComment !== source, '(a6) the comment fixture text is still there');
		eq(
			ownershipOffences(harvestComponentStrings(inComment), quotedKeysIn(source)).length,
			0,
			'(a6) a possessive planted in a docblock is not reported — comments are ours'
		);
	}
	console.log(
		`  proof: ${ESCAPED.length} phrasings that escaped earlier sweeps caught,` +
			` ${MUST_ALLOW.length} legitimate strings allowed,` +
			` ${plants.length} shipped sources planted (script consts and plain twins included),` +
			' 3 exclusions exercised in both directions'
	);
}

// --- (a6) THE GATE ----------------------------------------------------------
{
	let scanned = 0;
	let scriptStrings = 0;
	for (const relative of JARGON_TEMPLATES) {
		const source = read(relative);
		const strings = harvestComponentStrings(source);
		for (const { body } of scriptBodiesOf(source)) scriptStrings += harvestRenderStrings(body).length;
		scanned += strings.length;
		ok(strings.length > 0, `${relative}: the component harvest found no reader text at all`);
		reportOwnership(relative.replace(/^.*\//, ''), ownershipOffences(strings, quotedKeysIn(source)));
	}
	for (const name of JARGON_MODULES) {
		const source = read(`../src/lib/data/${name}`);
		const strings = harvestRenderStrings(source);
		scanned += strings.length;
		if (MODULES_WITH_PROSE.includes(name)) {
			ok(strings.length > 0, `${name}: the rendering-field harvest came back empty`);
		}
		reportOwnership(name, ownershipOffences(strings, quotedKeysIn(source)));
	}
	// The new surface must actually BE a surface: if the script harvest returns
	// nothing, this class has quietly shrunk back to the one (a4) already covers.
	ok(
		scriptStrings > 0,
		'the component <script> harvest found no strings — (a6) has lost the surface it was widened for'
	);
	console.log(
		`  ${JARGON_TEMPLATES.length} components + ${JARGON_MODULES.length} data modules,` +
			` ${scanned} reader strings scanned` +
			` (${scriptStrings} of them authored in a component <script>, which class (a4) does not read)`
	);
}

// ---------------------------------------------------------------------------
// (b) every SVG gutter is measured, budgeted from its own geometry, and enforced
// ---------------------------------------------------------------------------
console.log('\n(b) SVG label budgets, re-derived from the geometry they claim');

/** floor(gutter / advance): the budget any right/left-anchored run of text has. */
function budgetOf(units, unitsPerChar) {
	return Math.floor(units / unitsPerChar);
}
function report(figure, gutterUnits, unitsPerChar, budget, longest) {
	const used = longest.length * unitsPerChar;
	console.log(
		`  ${figure.padEnd(34)} budget ${String(budget).padStart(3)} ch` +
			`  longest ${String(longest.length).padStart(3)} ch` +
			`  headroom ${String(budget - longest.length).padStart(3)} ch` +
			` / ${(gutterUnits - used).toFixed(1).padStart(6)} u`
	);
}

// --- belief-model ladder: right-anchored rung names -------------------------
{
	const g = BELIEF_LADDER_GEOMETRY;
	eq(
		BELIEF_LADDER_DISPLAY_BUDGET_CHARS,
		budgetOf(g.labelAnchorX - g.leftGutter, g.monoUnitsPerChar),
		'ladder display budget is (gutter − left margin) / advance'
	);
	const longest = BELIEF_LADDER_ENTRY_SPECS.map((spec) => spec.display).reduce((a, b) =>
		b.length > a.length ? b : a
	);
	ok(
		BELIEF_LADDER_ENTRY_SPECS.every(
			(spec) => spec.display.length <= BELIEF_LADDER_DISPLAY_BUDGET_CHARS
		),
		'every ladder rung name is inside the axis gutter'
	);
	report(
		'ladder rung names',
		g.labelAnchorX - g.leftGutter,
		g.monoUnitsPerChar,
		BELIEF_LADDER_DISPLAY_BUDGET_CHARS,
		longest
	);

	// The origin tag is left-anchored at a data-dependent x, so it is measured
	// against the origin it will really be drawn at, not against a fixed budget.
	const todayOrigin = 272; // where the two negative rungs put it on the shipped data
	ok(
		beliefLadderBaselineTag('noisy-OR SimpleScorer (direct)', '0.9031', todayOrigin) !== null,
		'the shipped origin tag fits beside the shipped origin'
	);
	ok(
		beliefLadderBaselineTag('noisy-OR SimpleScorer (direct)', '0.9031', g.plotRight) === null,
		'an origin at the far right of the axis gates the tag rather than clipping it'
	);
	eq(
		BELIEF_LADDER_BASELINE_TAG_WORST_CASE_CHARS,
		budgetOf(g.width - g.plotRight - g.baselineTagOffsetX, g.tagUnitsPerChar),
		'the documented worst-case tag budget is (width − plotRight − offset) / advance'
	);
}

// --- framing correction: right-anchored control-strip rows ------------------
{
	const g = FRAMING_CONTROL_GEOMETRY;
	eq(
		FRAMING_CONTROL_LABEL_BUDGET_CHARS,
		budgetOf(g.labelAnchorX, g.monoUnitsPerChar),
		'control-strip budget is gutter / advance'
	);
	// The enforcement, exercised: an over-budget row name must gate the panel.
	const control = await import('../src/lib/data/paper-framing-correction.ts');
	const artifact = readArtifact('non_reading_control.json');
	// The validator THROWS and the server loader gates on the throw, so the
	// enforcement is exercised the way the loader meets it.
	const clean = control.validateNonReadingControl(artifact);
	const longest = clean.rows
		.map((row) => row.display)
		.concat(clean.contrast.display)
		.reduce((a, b) => (b.length > a.length ? b : a));
	report(
		'control-strip row names',
		g.labelAnchorX,
		g.monoUnitsPerChar,
		FRAMING_CONTROL_LABEL_BUDGET_CHARS,
		longest
	);
	const over = mutable(artifact);
	over.rows[0].label = 'x'.repeat(FRAMING_CONTROL_LABEL_BUDGET_CHARS + 1);
	let gated = false;
	try {
		control.validateNonReadingControl(over);
	} catch {
		gated = true;
	}
	ok(gated, 'an over-budget control-strip row name gates the panel');
}

// --- AP decomposition: the terminus fan and the census note -----------------
{
	const g = AP_DECOMP_FAN_GEOMETRY;
	ok(apDecompFanNamesFit(), 'the shipped fan names clear each other');
	// The pairwise clearance, printed, because it is this figure's tightest gutter.
	const pitch = (g.fanRight - g.gapRight) / AP_DECOMP_FAN_SLOTS.length;
	const halfWidth = (slot) =>
		(Math.max(...slot.shortLines.map((line) => line.length)) * g.nameUnitsPerChar) / 2;
	let tightest = Number.POSITIVE_INFINITY;
	for (let i = 0; i + 1 < AP_DECOMP_FAN_SLOTS.length; i += 1) {
		tightest = Math.min(
			tightest,
			pitch - (halfWidth(AP_DECOMP_FAN_SLOTS[i]) + halfWidth(AP_DECOMP_FAN_SLOTS[i + 1]))
		);
	}
	console.log(
		`  ${'fan terminus names'.padEnd(34)} pitch ${pitch} u` +
			`  tightest adjacent clearance ${tightest.toFixed(2)} u` +
			` / ${(tightest / g.nameUnitsPerChar).toFixed(2)} ch`
	);
	ok(tightest > 0, 'no two adjacent fan names overlap');
	// The enforcement, exercised: widen one short line and the fan must fail.
	const widened = AP_DECOMP_FAN_SLOTS.map((slot, index) =>
		index === 0 ? { ...slot, shortLines: ['W'.repeat(24)] } : slot
	);
	ok(!apDecompFanNamesFit(widened), 'an over-wide fan name is rejected, not clipped');

	eq(
		AP_DECOMP_COUNT_NOTE_BUDGET_CHARS,
		budgetOf(g.width - g.gapRight, g.nameUnitsPerChar),
		'the census-note budget is (width − fan left edge) / advance'
	);
	ok(
		apDecompCountNoteFits('x'.repeat(AP_DECOMP_COUNT_NOTE_BUDGET_CHARS)),
		'a note at exactly the budget fits'
	);
	ok(
		!apDecompCountNoteFits('x'.repeat(AP_DECOMP_COUNT_NOTE_BUDGET_CHARS + 1)),
		'a note one character over the budget is rejected'
	);

	const m = AP_DECOMP_MIRROR_GEOMETRY;
	eq(
		AP_DECOMP_MIRROR_LABEL_BUDGET_CHARS,
		budgetOf(m.labelAnchorX, m.monoUnitsPerChar),
		'the mirror label budget is gutter / advance'
	);
	eq(
		AP_DECOMP_MIRROR_READOUT_BUDGET_CHARS,
		budgetOf(m.width - m.readoutX, m.monoUnitsPerChar),
		'the mirror readout budget is (width − readout x) / advance'
	);
}

// --- tie-inflation scatter: direct labels, both anchors ---------------------
{
	const g = TIE_SCATTER_GEOMETRY;
	// The real worst case the shipped arms produce: the longest display name,
	// end-anchored at the rightmost mark this figure can place. DERIVED from the
	// specs rather than pinned to a string — a pinned name goes stale the moment a
	// model is renamed, and then this assertion measures a label nobody draws.
	const longest = PAPER_LITERAL_ARM_SPECS.map((spec) => spec.display).reduce((a, b) =>
		b.length > a.length ? b : a
	);
	ok(
		tieScatterLabelFits(longest, g.right - g.labelOffsetX, 'end'),
		'the longest arm name fits end-anchored at the right edge of the plot'
	);
	ok(
		!tieScatterLabelFits('x'.repeat(200), g.right - g.labelOffsetX, 'end'),
		'an over-long end-anchored label is rejected'
	);
	ok(
		!tieScatterLabelFits('x'.repeat(200), g.width - 10, 'start'),
		'an over-long start-anchored label is rejected'
	);
	// Not a fixed budget: the gutter is whatever the mark's own x leaves, so what
	// is reported is the WORST case the plot can produce — the longest name,
	// end-anchored at the rightmost position a mark can take.
	const gutter = g.right - g.labelOffsetX - g.left;
	report(
		'tie scatter (worst-case mark)',
		gutter,
		g.monoUnitsPerChar,
		budgetOf(gutter, g.monoUnitsPerChar),
		longest
	);
}

// --- statement-grain error-F1: four gutters, and a budget nothing can reach ---
{
	const g = STATEMENT_ERROR_F1_GEOMETRY;
	eq(
		STATEMENT_ERROR_F1_LABEL_BUDGET_CHARS,
		budgetOf(g.labelAnchorX, g.monoUnitsPerChar),
		'the error-F1 lane budget is gutter / advance'
	);
	eq(
		STATEMENT_ERROR_F1_SUB_BUDGET_CHARS,
		budgetOf(g.labelAnchorX, g.subUnitsPerChar),
		'the tau sub-label budget is the same gutter at the 7.5px advance'
	);
	eq(
		STATEMENT_ERROR_F1_READOUT_A_BUDGET_CHARS,
		budgetOf(g.panelAReadoutRight - g.panelAReadoutX, g.readoutUnitsPerChar),
		'the panel-A readout budget is (readout right − readout x) / advance'
	);
	eq(
		STATEMENT_ERROR_F1_READOUT_B_BUDGET_CHARS,
		budgetOf(g.width - g.panelBReadoutX, g.readoutUnitsPerChar),
		'the panel-B readout budget is (width − readout x) / advance'
	);

	// Measured through the loader on the shipped bytes, not off the docblock: the
	// four budgeted strings are what buildFigure() actually produced.
	const artifact = readArtifact('statement_error_f1.json');
	const load = validateStatementErrorF1(artifact);
	ok(load.status === 'ok', `the shipped error-F1 artifact loads: ${load.reason ?? ''}`);
	if (load.status === 'ok') {
		const lanes = load.figure.lanes;
		ok(
			lanes.every((lane) => lane.display.length <= STATEMENT_ERROR_F1_LABEL_BUDGET_CHARS),
			'every drawn error-F1 lane name is inside the label gutter'
		);
		ok(
			lanes.every((lane) => lane.subLabel.length <= STATEMENT_ERROR_F1_SUB_BUDGET_CHARS),
			'every drawn tau sub-label is inside the label gutter'
		);
		ok(
			lanes.every((lane) => lane.readoutA.length <= STATEMENT_ERROR_F1_READOUT_A_BUDGET_CHARS),
			'every panel-A readout is inside its gutter'
		);
		ok(
			lanes.every((lane) => lane.readoutB.length <= STATEMENT_ERROR_F1_READOUT_B_BUDGET_CHARS),
			'every panel-B readout is inside its gutter'
		);
		// Both rule annotations flip side rather than clip; on the shipped data both
		// fit, and the component reads the flag rather than measuring for itself.
		ok(
			load.figure.referenceRuleLabelFits === true &&
				load.figure.zeroRuleLabelFits === true,
			'the shipped rule annotations fit beside their own rules'
		);
		const longest = lanes
			.map((lane) => lane.display)
			.reduce((a, b) => (b.length > a.length ? b : a));
		eq(
			longest,
			STATEMENT_ERROR_F1_LONGEST_DRAWN_LABEL,
			'the longest drawn lane name is the one the module derives from the frozen set'
		);
		report(
			'error-F1 lane names',
			g.labelAnchorX,
			g.monoUnitsPerChar,
			STATEMENT_ERROR_F1_LABEL_BUDGET_CHARS,
			longest
		);
	}

	// NON-VACUITY, then the gate that makes the budget a bound. Unlike the control
	// strip, NO artifact-controlled string on this figure can reach `budget()`: the
	// lane name comes from the frozen spec table, and the tau sub-label and both
	// readouts are formatted from numbers that a range or identity gate has already
	// bounded (tau outside [0,1] and a delta that is not the drawn difference both
	// gate first). What the artifact can do is NAME a different arm — so that is
	// what is exercised, on the two canonical displays that do not fit.
	const overBudget = PAPER_LITERAL_ARM_SPECS.filter(
		(spec) => spec.display.length > STATEMENT_ERROR_F1_LABEL_BUDGET_CHARS
	);
	ok(
		overBudget.length > 0,
		'the lane budget is a live bound: some canonical display exceeds it'
	);
	ok(
		overBudget.every((spec) => !STATEMENT_ERROR_F1_DRAWN_ARM_IDS.includes(spec.id)),
		`an over-budget canonical display is in the drawn set: ${overBudget
			.map((spec) => spec.display)
			.join(' | ')}`
	);
	for (const spec of overBudget) {
		const swapped = mutable(artifact);
		swapped.arms[0].label = spec.label;
		ok(
			validateStatementErrorF1(swapped).status === 'unavailable',
			`naming "${spec.display}" (${spec.display.length} ch) as a drawn arm gates the figure`
		);
	}
}

// --- extended Table 6: four columns, all four budgeted ----------------------
{
	const g = PAPER_TABLE6_GEOMETRY;
	eq(
		PAPER_TABLE6_LABEL_BUDGET_CHARS,
		budgetOf(g.labelAnchorX - g.labelGutterLeft, g.monoUnitsPerChar),
		'the Table 6 method-name budget is (anchor − gutter wall) / advance'
	);
	eq(
		PAPER_TABLE6_RANK_BUDGET_CHARS,
		budgetOf(g.rankAnchorX, g.readoutUnitsPerChar),
		'the rank budget is gutter / advance'
	);
	eq(
		PAPER_TABLE6_METRIC_BUDGET_CHARS,
		budgetOf(g.tieX - g.metricX, g.readoutUnitsPerChar),
		'the paper-metric readout budget is (tie x − metric x) / advance'
	);
	eq(
		PAPER_TABLE6_TIE_BUDGET_CHARS,
		budgetOf(g.width - g.tieX, g.readoutUnitsPerChar),
		'the tie-robust readout budget is (width − tie x) / advance'
	);

	const artifact = readArtifact('paper_table6_extended.json');
	const load = validatePaperTable6Extended(artifact);
	ok(load.status === 'ok', `the shipped Table 6 artifact loads: ${load.reason ?? ''}`);
	if (load.status === 'ok') {
		const rows = load.figure.rows;
		ok(
			rows.every((row) => row.display.length <= PAPER_TABLE6_LABEL_BUDGET_CHARS),
			'every ranked method name is inside the name gutter'
		);
		ok(
			rows.every((row) => row.rankReadout.length <= PAPER_TABLE6_RANK_BUDGET_CHARS),
			'every rank numeral is inside its gutter'
		);
		ok(
			rows.every((row) => row.metricReadout.length <= PAPER_TABLE6_METRIC_BUDGET_CHARS),
			'every paper-metric readout is inside its gutter'
		);
		ok(
			rows.every((row) => row.tieReadout.length <= PAPER_TABLE6_TIE_BUDGET_CHARS),
			'every tie-robust readout is inside its gutter'
		);
		// The column headers sit in the same four gutters as the rows beneath them,
		// and are the one string on this figure that is not data — budgeted anyway.
		ok(
			load.figure.rankHeader.length <= PAPER_TABLE6_RANK_BUDGET_CHARS &&
				load.figure.labelHeader.length <= PAPER_TABLE6_LABEL_BUDGET_CHARS &&
				load.figure.metricHeader.length <= PAPER_TABLE6_METRIC_BUDGET_CHARS &&
				load.figure.tieHeader.length <= PAPER_TABLE6_TIE_BUDGET_CHARS,
			'every column header is inside the gutter of the column it heads'
		);
		const longest = rows
			.map((row) => row.display)
			.concat(load.figure.labelHeader)
			.reduce((a, b) => (b.length > a.length ? b : a));
		report(
			'Table 6 method names',
			g.labelAnchorX - g.labelGutterLeft,
			g.monoUnitsPerChar,
			PAPER_TABLE6_LABEL_BUDGET_CHARS,
			longest
		);
	}

	// The enforcement, exercised where the artifact really owns the string: this
	// figure's names are READ from the artifact (the paper's own row names), so an
	// over-budget one must take the ranked list down rather than clip a row.
	const over = mutable(artifact);
	over.rows[0].display = 'x'.repeat(PAPER_TABLE6_LABEL_BUDGET_CHARS + 1);
	ok(
		validatePaperTable6Extended(over).status === 'unavailable',
		'an over-budget method name gates the ranked list'
	);
}

// --- the two figures that already enforced: assert the arithmetic still holds
{
	const g = PAPER_OWN_METRIC_GEOMETRY;
	eq(
		PAPER_OWN_METRIC_LABEL_BUDGET_CHARS,
		budgetOf(g.labelAnchorX, g.monoUnitsPerChar),
		"the paper's-own-metric lane budget is gutter / advance"
	);
	eq(
		PAPER_OWN_METRIC_READOUT_BUDGET_CHARS,
		budgetOf(g.width - g.readoutX, 4.8165),
		"the paper's-own-metric readout budget is (width − readout x) / advance"
	);
	const r = PAPER_ROBUSTNESS_GEOMETRY;
	eq(
		PAPER_ROBUSTNESS_LABEL_BUDGET_CHARS,
		budgetOf(r.labelAnchorX, r.monoUnitsPerChar),
		'the robustness lane budget is gutter / advance'
	);
	eq(
		PAPER_ROBUSTNESS_READOUT_BUDGET_CHARS,
		budgetOf(r.width - r.readoutX, r.readoutUnitsPerChar),
		'the robustness readout budget is (width − readout x) / advance'
	);
}

// ---------------------------------------------------------------------------
// (c) no placeholder can render as a measurement
// ---------------------------------------------------------------------------
console.log('\n(c) placeholders never render as measurements');

const PLACEHOLDER = /\?\?\s*(0\b|''|""|`\`)|\|\|\s*0\b/;
for (const name of COMPONENTS) {
	const template = templateOf(read(`../src/lib/components/${name}`));
	const offenders = renderExpressions(template).filter((expression) =>
		PLACEHOLDER.test(expression)
	);
	ok(
		offenders.length === 0,
		`${name} coalesces to a placeholder in a render position: ${offenders
			.map((o) => o.trim().slice(0, 60))
			.join(' | ')}`
	);
}

/**
 * The degrade contract that `ece: 0` broke. Every server-computed calibration
 * and ranked-block scalar must be typed nullable, so the compiler — not a
 * reviewer — finds the render sites when one of them fails to join.
 */
{
	const source = read('../src/lib/data/paper-literal.ts');
	for (const field of [
		'ece',
		'brier',
		'brierReliability',
		'brierResolution',
		'brierUncertainty',
		'calibrationSlope',
		'calibrationIntercept',
		'aurocOnRanked'
	]) {
		ok(
			new RegExp(`\\n\\t${field}:\\s*[^;\\n]*\\|\\s*null;`).test(source),
			`PaperLiteralArm.${field} is nullable, so a failed join cannot print an ideal value`
		);
	}
	// The ok branch coalesced a missing digest to '' and printed it as a real,
	// empty sha. Nullable on BOTH branches, in every module that does this.
	for (const module of [
		'paper-literal.ts',
		'paper-tie-inflation.ts',
		'paper-robustness.ts'
	]) {
		const text = read(`../src/lib/data/${module}`);
		ok(
			!/artifact_sha256:\s*artifactSha256\s*\?\?\s*''/.test(text),
			`${module} does not coalesce a missing artifact digest to an empty string`
		);
	}
}

/**
 * The shared-prior claim on the belief-heuristic panel names ONE belief value and
 * says every drawn source was assigned it. The loader used to keep whichever
 * source came last (`let singleBelief = 0`), which would have absorbed a source
 * with a different prior into a claim of sameness — and, with no qualifying
 * source at all, printed 0 as "0%".
 */
{
	const source = read('../src/lib/server/belief-heuristic.ts');
	ok(
		!/let singleBelief = 0;/.test(source),
		'the single-evidence belief is not seeded with a renderable placeholder'
	);
	ok(
		/singleBeliefs\.size !== 1/.test(source),
		'the shared-prior claim is checked against the data rather than assumed'
	);
	// The math lives in a dependency-free module precisely so it can be exercised
	// here: `$lib/server/belief-heuristic` cannot be loaded outside SvelteKit, and
	// a statistic reachable only through a page is a statistic nobody re-checks.
	const { homogeneityChiSquare } = await import('../src/lib/data/homogeneity.ts');
	ok(
		/homogeneityChiSquare\(singleRows\)/.test(source),
		'the loader ships the homogeneity test with the single-evidence rung'
	);
	// Golden: the shipped single-evidence rung — five sources, 315 statements, all
	// assigned the identical belief. Cross-checked against scipy.stats.chi2.sf.
	const shipped = homogeneityChiSquare([
		{ count: 54, correct: 35 },
		{ count: 42, correct: 24 },
		{ count: 73, correct: 40 },
		{ count: 104, correct: 50 },
		{ count: 42, correct: 5 }
	]);
	ok(shipped !== null, 'the homogeneity test returns a result for the shipped rung');
	if (shipped) {
		ok(Math.abs(shipped.chi2 - 30.66) < 0.01, `chi-square is 30.66, got ${shipped.chi2}`);
		eq(shipped.df, 4, 'four degrees of freedom for five sources');
		ok(
			Math.abs(shipped.p / 3.5854e-6 - 1) < 1e-3,
			`p is 3.59e-6 (scipy golden), got ${shipped.p}`
		);
		console.log(
			`  single-evidence homogeneity: chi2 ${shipped.chi2.toFixed(2)} df ${shipped.df} p ${shipped.p.toExponential(2)}`
		);
	}
	eq(
		homogeneityChiSquare([{ count: 10, correct: 5 }]),
		null,
		'one source is not a homogeneity test'
	);
	eq(
		homogeneityChiSquare([
			{ count: 10, correct: 10 },
			{ count: 10, correct: 10 }
		]),
		null,
		'a degenerate table returns null rather than a chi-square of zero'
	);
}

if (failures) {
	console.error(`\n${failures} /paper render-invariant assertion(s) failed`);
	process.exit(1);
}
console.log('\n/paper render invariants passed');
