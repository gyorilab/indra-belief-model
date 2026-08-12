/**
 * Contract for the INDRA-belief-vs-reading comparison.
 *
 * Two halves:
 *   1. FIXTURE assertions on the pure math — binning, ECE, the identical-belief
 *      cohort, the disagreement band — with hand-checkable inputs.
 *   2. A LIVE assertion against the committed artifacts, pinning the headline
 *      numbers the /belief page states in prose. These were independently
 *      derived in Python before the TypeScript existed; if the port drifts from
 *      them, this fails.
 *
 * Run: npm run test:belief-vs-indra
 */
import assert from 'node:assert/strict';
import { existsSync, readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

import {
	BIN_COUNT,
	LOW_BELIEF,
	MIN_SOURCE_N,
	beliefSeries,
	disagreementBand,
	identicalBeliefCohort,
	probeSeries
} from '../src/lib/data/belief-vs-indra.ts';

const HERE = dirname(fileURLToPath(import.meta.url));
const DATA = join(HERE, '..', '..', 'data');

/* ------------------------------------------------------------------ fixtures */

{
	// Four statements, two per bin, with known outcomes.
	const pred = new Map([
		['a', 0.05],
		['b', 0.05],
		['c', 0.95],
		['d', 0.95]
	]);
	const gold = new Map([
		['a', 0],
		['b', 0],
		['c', 1],
		['d', 0]
	]);
	const s = beliefSeries('t', 'test', pred, gold);
	assert.equal(s.n, 4);
	assert.equal(s.bins.length, 2, 'two occupied bins');
	assert.deepEqual(s.bins[0], { p_mean: 0.05, y_rate: 0, n: 2 });
	assert.deepEqual(s.bins[1], { p_mean: 0.95, y_rate: 0.5, n: 2 });
	// ECE = .5*|0-.05| + .5*|.5-.95| = .025 + .225
	assert.equal(s.ece, 0.25);
	assert.equal(s.below_half, 2);
	assert.equal(s.min, 0.05);
	assert.equal(s.max, 0.95);
	assert.equal(s.distinct_values, 2);
}

{
	// A statement absent from gold must not be scored.
	const s = beliefSeries('t', 'test', new Map([['a', 0.5], ['ghost', 0.9]]), new Map([['a', 1]]));
	assert.equal(s.n, 1, 'ungolded predictions are dropped, not counted');
}

{
	// Values at the bin edges land in the expected bin, and 1.0 does not overflow.
	const pred = new Map([['a', 0], ['b', 0.1], ['c', 1]]);
	const gold = new Map([['a', 1], ['b', 1], ['c', 1]]);
	const s = beliefSeries('t', 'test', pred, gold);
	assert.equal(s.bins.length, 3);
	assert.equal(s.bins.at(-1).p_mean, 1, '1.0 clamps into the last bin, not a 11th');
	assert.equal(BIN_COUNT, 10);
}

{
	// Cohort: the most common value wins; sources below MIN_SOURCE_N are not drawn.
	const pred = new Map();
	const gold = new Map();
	const profiles = new Map();
	for (let i = 0; i < 20; i += 1) {
		const id = `big${i}`;
		pred.set(id, 0.65);
		gold.set(id, i < 5 ? 1 : 0); // 25% correct
		profiles.set(id, { evidence_count: 1, sources: ['reach'] });
	}
	for (let i = 0; i < 3; i += 1) {
		const id = `small${i}`;
		pred.set(id, 0.65);
		gold.set(id, 1);
		profiles.set(id, { evidence_count: 1, sources: ['tiny'] });
	}
	pred.set('other', 0.9);
	gold.set('other', 1);
	profiles.set('other', { evidence_count: 2, sources: ['reach'] });

	const c = identicalBeliefCohort(pred, gold, profiles);
	assert.equal(c.belief, 0.65);
	assert.equal(c.n, 23);
	assert.equal(c.evidence_count, 1, 'whole cohort agrees on one evidence');
	const drawn = c.by_source.map((s) => s.source);
	assert.ok(drawn.includes('reach'), 'reach has 20 >= MIN_SOURCE_N');
	assert.ok(!drawn.includes('tiny'), `tiny has 3 < ${MIN_SOURCE_N} and must not be drawn`);
}

{
	// A cohort whose members disagree on evidence count must not claim one.
	const pred = new Map([['a', 0.65], ['b', 0.65]]);
	const gold = new Map([['a', 1], ['b', 0]]);
	const profiles = new Map([
		['a', { evidence_count: 1, sources: [] }],
		['b', { evidence_count: 3, sources: [] }]
	]);
	assert.equal(identicalBeliefCohort(pred, gold, profiles).evidence_count, null);
}

{
	// Disagreement band: only reader-low statements, median over the counting model.
	const reader = new Map([['a', 0.01], ['b', 0.02], ['c', 0.99]]);
	const counting = new Map([['a', 0.7], ['b', 0.95], ['c', 0.99]]);
	const gold = new Map([['a', 0], ['b', 1], ['c', 1]]);
	const d = disagreementBand(reader, counting, gold);
	assert.equal(d.n, 2, 'only the two below the threshold');
	assert.equal(d.threshold, LOW_BELIEF);
	assert.equal(d.observed_rate, 0.5);
	assert.equal(d.indra_min, 0.7);
	assert.equal(d.indra_median, 0.825, 'even count averages the middle pair');
	assert.equal(d.indra_at_least_90, 1);
}

/* ---------------------------------------------------------------- live pins */

const GOLD = join(DATA, 'results', 'indra_paper_statement_gold_20260717', 'paper_statement_gold.jsonl');
const INDRA = join(
	DATA,
	'results',
	'current_indra_simple_paper_20260717',
	'current_indra_simple_default_predictions.jsonl'
);
const READER = join(DATA, 'comparison', 'models', 'gemma_4_26b', 'all_source_predictions.jsonl');

if (![GOLD, INDRA, READER].every(existsSync)) {
	console.log('belief-vs-indra: fixtures OK; live artifacts absent, live pins skipped');
} else {
	const gold = new Map();
	for (const line of readFileSync(GOLD, 'utf8').split('\n')) {
		if (!line.trim()) continue;
		const row = JSON.parse(line);
		const sid = row?.canonical_corpus?.statement_id;
		const label = row?.paper_replication_policy?.released_paper_correct;
		if (typeof sid === 'string' && typeof label === 'number') gold.set(sid, label);
	}
	const preds = (path) => {
		const out = new Map();
		for (const line of readFileSync(path, 'utf8').split('\n')) {
			if (!line.trim()) continue;
			const row = JSON.parse(line);
			if (typeof row?.statement_id === 'string') out.set(row.statement_id, row.probability_correct);
		}
		return out;
	};
	const indra = preds(INDRA);
	const reader = preds(READER);

	assert.equal(gold.size, 1689, 'the released gold set is 1,689 statements');

	const si = beliefSeries('i', 'indra', indra, gold);
	const sr = beliefSeries('r', 'reader', reader, gold);
	assert.equal(si.n, 1689);
	assert.equal(sr.n, 1689);

	// THE FINDING: the noisy-OR cannot express doubt on this corpus.
	assert.equal(si.min, 0.65, "INDRA's floor on this corpus is 0.65");
	assert.equal(si.below_half, 0, 'INDRA assigns no statement a belief below 0.5');
	assert.equal(si.bins.length, 3, 'INDRA occupies only 3 of 10 bins');
	assert.ok(sr.below_half > 400, `reading occupies the low band (${sr.below_half} below 0.5)`);
	assert.ok(sr.bins.length > si.bins.length, 'reading spans more of the scale');

	// Calibration error, to three places, as the page prints it.
	assert.equal(si.ece.toFixed(3), '0.179');
	assert.equal(sr.ece.toFixed(3), '0.102');
	assert.ok(sr.ece < si.ece, 'reading is closer to honest odds');

	const c = identicalBeliefCohort(indra, gold, null);
	assert.equal(c.belief, 0.65);
	assert.equal(c.n, 328, '328 statements share the identical belief 0.65');
	assert.equal(c.observed_rate.toFixed(3), '0.473', 'and are correct 47.3% of the time');

	const d = disagreementBand(reader, indra, gold);
	assert.equal(d.n, 462);
	assert.equal(d.observed_rate.toFixed(3), '0.234');
	assert.equal(d.indra_min, 0.65);
	assert.equal(d.indra_at_least_90, 206, 'INDRA calls 206 of them at least 90% likely');
	// The mechanism: both series run INDRA's own aggregation, and the only way a
	// belief of exactly 0 arises is an empty product.
	const zero = [...reader.entries()].filter(([sid, v]) => gold.has(sid) && v === 0).length;
	assert.equal(zero, 462, 'the 462 at exactly 0.0 are the all-evidence-rejected statements');
	assert.equal(zero, d.n, 'the empty-product set IS the disagreement band');
	const agg = JSON.parse(readFileSync(join(DATA, 'comparison', 'aggregation.json'), 'utf8'));
	assert.equal(
		agg.aggregation,
		'indra_default_hard_gate',
		'the reading series uses INDRA aggregation, NOT the fitted log-odds — if this changes, the ' +
			'/belief mechanism panel is describing the wrong model'
	);

	// The probe: measured on its OWN holdout with its OWN bin edges. Its ECEs must
	// reproduce the frozen decision artifact exactly, or the page is quoting a
	// number the artifact does not support.
	const probeScores = join(DATA, 'probe_battery', 'holdout_scores_C_incumbent_plus_battery.jsonl');
	const probeDecision = join(DATA, 'probe_battery', 'decision_C_incumbent_plus_battery.json');
	if (existsSync(probeScores) && existsSync(probeDecision)) {
		const rows = readFileSync(probeScores, 'utf8')
			.split('\n')
			.filter((l) => l.trim())
			.map((l) => JSON.parse(l));
		const dec = JSON.parse(readFileSync(probeDecision, 'utf8'));
		const inc = probeSeries('i', 'incumbent', rows.map((r) => ({ score: r.incumbent_score, correct: r.gold_correct })));
		const cand = probeSeries('c', 'candidate', rows.map((r) => ({ score: r.candidate_score, correct: r.gold_correct })));
		assert.equal(inc.n, 500);
		assert.equal(inc.distinct_values, 3, 'the deployed verdict grid emits three distinct scores');
		assert.equal(cand.distinct_values, 30, 'the probe emits thirty');
		// probeSeries rounds to six places, so the tolerance is half of the last
		// retained digit. A tighter bound would fail on the rounding, not the math.
		const TOL = 5e-7;
		assert.ok(
			Math.abs(inc.ece - dec.incumbent.ece) < TOL,
			`incumbent ECE reproduces the artifact (${inc.ece} vs ${dec.incumbent.ece})`
		);
		assert.ok(
			Math.abs(cand.ece - dec.candidate.ece) < TOL,
			`candidate ECE reproduces the artifact (${cand.ece} vs ${dec.candidate.ece})`
		);
		assert.equal(dec.gate?.verdict ?? dec.verdict, 'GO');
		assert.ok(dec.paired_bootstrap.ci95_low > 0, 'the delta-AUROC interval excludes zero');
		console.log('belief-vs-indra: fixtures + live pins + probe pins OK');
	} else {
		console.log('belief-vs-indra: fixtures + live pins OK; probe artifacts absent');
	}
}
