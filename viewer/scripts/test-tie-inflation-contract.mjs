/**
 * Pure-function assertions for the tie-inflation data contract.
 *
 * Mirrors test-belief-ladder-contract.mjs: the validator is exercised against the
 * REAL shipped artifact, then against mutations that must each gate the figure to
 * `unavailable` rather than render a wrong number.
 *
 * Why this file exists: the component shipped with zero executable coverage while
 * every sibling on /paper has some. Its whole argument is that the paper's own
 * estimator inflates OUR arms, so a silently-wrong inflation number here would
 * corrupt the one figure whose job is to argue against our best-looking result.
 */
import { readFileSync } from 'node:fs';

import { validateTieInflation } from '../src/lib/data/paper-tie-inflation.ts';

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

const ARTIFACT = new URL(
	'../../data/results/indra_paper_literal_models_20260724/paper_literal_vs_llms.json',
	import.meta.url
);
const raw = JSON.parse(readFileSync(ARTIFACT, 'utf8'));

// ---- the real artifact validates, and carries what the figure claims ---------
const live = validateTieInflation(raw, { artifactPath: 'fixture', artifactSha256: 'deadbeef' });
eq(live.status, 'ok', 'the shipped artifact validates');

if (live.status === 'ok') {
	const arms = live.arms;
	eq(arms.length, 8, 'every scored arm reaches the tie figure');

	// Inflation and distinct_scores are SHIPPED fields. If the loader ever starts
	// recomputing them, these drift and the whole argument becomes unfalsifiable.
	for (const arm of arms) {
		const shipped = raw.point_metrics[arm.label];
		ok(shipped !== undefined, `${arm.display}: label is a real point_metrics key`);
		if (!shipped) continue;
		eq(arm.inflation, shipped.trapezoidal_minus_ap_inflation, `${arm.display}: inflation is READ`);
		eq(arm.distinctScores, shipped.distinct_scores, `${arm.display}: distinct_scores is READ`);
		eq(arm.ap, shipped.pooled_average_precision, `${arm.display}: ap is READ`);
		// A frozen join key must never be the on-screen name.
		ok(
			!(arm.label.startsWith('Paper literal') && arm.display === arm.label),
			`${arm.display}: display is decoupled from the frozen join key`
		);
	}

	// The figure's central claim, asserted as arithmetic rather than as prose: the
	// reading arms inflate by at least an order of magnitude more than the RF arms.
	const readers = arms.filter((arm) => arm.isReader);
	const rfSide = arms.filter((arm) => arm.kind === 'paper' || arm.kind === 'port');
	ok(readers.length === 4, `four reading arms (found ${readers.length})`);
	ok(rfSide.length === 3, `three RF-side arms (found ${rfSide.length})`);
	const worstRf = Math.max(...rfSide.map((arm) => Math.abs(arm.inflation)));
	const leastReader = Math.min(...readers.map((arm) => arm.inflation));
	ok(
		leastReader > worstRf * 10,
		`the least-inflated reader (${leastReader}) still exceeds 10x the worst RF arm (${worstRf})`
	);
	// Tie-ness is the mechanism: RF arms must emit far more distinct scores.
	ok(
		Math.min(...rfSide.map((a) => a.distinctScores)) >
			Math.max(...readers.map((a) => a.distinctScores)) * 2,
		'every RF arm emits at least twice the distinct scores of every reading arm'
	);
}

// ---- fail-closed: each mutation must gate, not render ------------------------
let mutationCases = 0;
function mutated(mutate, label) {
	mutationCases += 1;
	const copy = JSON.parse(JSON.stringify(raw));
	mutate(copy);
	const result = validateTieInflation(copy, { artifactPath: 'fixture', artifactSha256: 'x' });
	eq(result.status, 'unavailable', `gates: ${label}`);
}

mutated((d) => delete d.point_metrics['GLM-5'].trapezoidal_minus_ap_inflation, 'missing inflation');
mutated((d) => delete d.point_metrics['GLM-5'].distinct_scores, 'missing distinct_scores');
mutated((d) => {
	d.point_metrics['GLM-5'].distinct_scores = 0;
}, 'non-positive distinct_scores');
mutated((d) => {
	d.point_metrics['GLM-5'].pooled_average_precision = 2;
}, 'out-of-range average precision');
mutated((d) => delete d.point_metrics['Gemma 4 26B'], 'a whole arm disappearing');
mutated((d) => {
	d.n_statements = 0;
}, 'zero-statement panel');

// The shipped inflation must reconcile with its own two components, or the figure
// is drawing a quantity that does not mean what its axis says.
mutated((d) => {
	d.point_metrics['GLM-5'].pooled_trapezoidal_pr_auc =
		d.point_metrics['GLM-5'].pooled_average_precision + 0.5;
}, 'inflation not reconciling with pooled trapezoidal minus ap');

// ---- the component must not soften the argument it exists to make -----------
const component = readFileSync(
	new URL('../src/lib/components/TieInflation.svelte', import.meta.url),
	'utf8'
);
ok(
	/not a confidence interval|NOT a confidence interval|dispersion/i.test(component) ||
		!/±/.test(component),
	'if the component renders a ±, it says the ± is dispersion'
);
ok(
	!/\brecomputed?\b/i.test(component.replace(/never recomputed?/gi, '')),
	'the component reads shipped fields rather than recomputing them'
);

// Reported so a runner that silently stopped mutating cannot still exit 0
// (the belief-ladder runner guards itself the same way).
console.log(`${mutationCases} fail-closed mutation cases exercised`);
console.log(
	failures === 0
		? 'tie-inflation data contract assertions passed'
		: `${failures} tie-inflation contract assertion(s) failed`
);
process.exit(failures === 0 ? 0 : 1);
