/**
 * What the 2023 paper's belief heuristic PROMISES vs what it DELIVERS.
 *
 * The paper's `indra.belief.SimpleScorer` is a noisy-OR over per-source priors:
 *
 *     belief = 1 − Π_s (syst_s + rand_s^{n_s})
 *
 * It is therefore a pure function of the evidence-count profile — the (source,
 * count) tuple — and cannot see what any sentence actually says. Two statements
 * with the same profile receive the identical belief whether the reading is right
 * or wrong. This loader makes that concrete by walking the committed provenance
 * (which persists `source_counts` per statement) and joining the released paper
 * gold, so each rung of the ladder carries BOTH the belief the heuristic assigned
 * and the rate at which those statements were actually correct.
 *
 * Mirrors the `paper-literal.ts` conventions: DATA_DIR join, size cap, and an
 * `unavailable()` payload rather than a throw. Computes only counts and rates
 * from committed artifacts — it fits nothing and calls no scorer.
 */
import { existsSync, readFileSync, statSync } from 'node:fs';
import { join } from 'node:path';

import { homogeneityChiSquare, type HomogeneityResult } from '$lib/data/homogeneity';
import { DATA_DIR } from '$lib/data/runs';

const PROVENANCE_PATH = join(
	DATA_DIR,
	'results',
	'current_indra_simple_paper_20260717',
	'current_indra_simple_default_prediction_provenance.jsonl'
);
const GOLD_PATH = join(
	DATA_DIR,
	'results',
	'indra_paper_statement_gold_20260717',
	'paper_statement_gold.jsonl'
);

/** The gold file is ~4 MB; keep the cap above it but still bounded. */
const MAX_ARTIFACT_BYTES = 16 * 1024 * 1024;

/** A rung is one evidence count for one single-source profile. */
export interface HeuristicRung {
	/** Number of evidences (all from the one source). */
	n: number;
	/** Belief the heuristic assigned — identical for every statement on this rung. */
	belief: number;
	/** Statements on this rung. */
	count: number;
	/** Fraction of them the paper's own gold marked correct. */
	correctRate: number;
}

export interface HeuristicSource {
	source: string;
	/** Total single-source statements for this source. */
	total: number;
	/** Rungs with at least `filters.minRung` statements — the drawable ones. */
	rungs: HeuristicRung[];
	/**
	 * Spearman rank correlation between evidence count and correctness over ALL
	 * of this source's single-source statements — undrawn rungs included. This is
	 * the trend statement; no individual rung is powered to make one.
	 */
	rho: number;
	/** Statements on rungs too small to draw (total − drawn). */
	nUndrawn: number;
}

/** One source's slice of the single-evidence rung, where belief is identical. */
export interface SingleEvidenceRow {
	source: string;
	count: number;
	correct: number;
	correctRate: number;
}

export interface HeuristicLoad {
	status: 'ok';
	/** Single-source ladders, richest source first. */
	sources: HeuristicSource[];
	/** Panel-wide: statements, distinct belief values emitted (exact, unrounded). */
	nStatements: number;
	distinctBeliefs: number;
	/** The single most-populated belief value, and how it actually performed. */
	modal: { belief: number; count: number; correctRate: number };
	/**
	 * The n=1 rung: every statement here is assigned the SAME belief, because the
	 * default priors give these readers the identical (rand, syst) pair. Best-powered
	 * statement on the page, and the reason `trips` must not be dropped.
	 */
	singleEvidence: {
		belief: number;
		total: number;
		rows: SingleEvidenceRow[];
		/**
		 * Pearson chi-square that these rows share one correct rate — the test the
		 * "one shared prior, very different readers" claim rests on. Null when the
		 * table is degenerate (one row, or no variation at all); the panel then
		 * states the spread without calling it more than a spread.
		 */
		homogeneity: HomogeneityResult | null;
	};
	/**
	 * Where the formula spends its granularity. It emits many distinct values, but
	 * nearly all of them land in the saturated region above `cut`, where every
	 * statement is already effectively certain. Below `cut` — where the review
	 * decisions actually happen — it has very few answers. This is the honest
	 * reconciliation of "690 distinct values" with "the grain is uninformative".
	 */
	saturation: {
		cut: number;
		nAbove: number;
		distinctAbove: number;
		nBelow: number;
		distinctBelow: number;
	};
	/** Every filter applied above, so the page can disclose them. */
	filters: { minRung: number; minSource: number; minSingleEvidence: number };
}

export type BeliefHeuristicLoad = HeuristicLoad | { status: 'unavailable'; reason: string };

function unavailable(reason: string): BeliefHeuristicLoad {
	return { status: 'unavailable', reason };
}

function readGuarded(path: string, label: string): { ok: true; text: string } | { ok: false; reason: string } {
	if (!existsSync(path)) return { ok: false, reason: `${label} is missing.` };
	try {
		if (statSync(path).size > MAX_ARTIFACT_BYTES) {
			return { ok: false, reason: `${label} exceeds ${MAX_ARTIFACT_BYTES} bytes.` };
		}
		return { ok: true, text: readFileSync(path, 'utf-8') };
	} catch (error) {
		return { ok: false, reason: `${label} could not be read: ${(error as Error).message}` };
	}
}

/**
 * Minimum statements on a rung before it is worth drawing (rate noise otherwise).
 * DISCLOSED on the page: it is a display filter, and reading a trend off the
 * surviving endpoints is exactly the mistake it invites. Trends come from `rho`,
 * which is computed over every statement, drawn or not.
 */
const MIN_RUNG = 5;
/** Minimum single-source statements before a source gets its own ladder. */
const MIN_SOURCE = 40;
/** Minimum single-EVIDENCE statements before a source joins the n=1 comparison. */
const MIN_SINGLE_EVIDENCE = 20;
/** Above this the noisy-OR has saturated: statements here are already ~certain. */
const SATURATION_CUT = 0.99;

/** Spearman rank correlation, average ranks for ties. */
function spearman(xs: number[], ys: number[]): number {
	const rank = (v: number[]): number[] => {
		const order = v.map((value, i) => [value, i] as const).sort((a, b) => a[0] - b[0]);
		const out = new Array<number>(v.length);
		for (let i = 0; i < order.length; ) {
			let j = i;
			while (j + 1 < order.length && order[j + 1][0] === order[i][0]) j += 1;
			const avg = (i + j) / 2 + 1;
			for (let k = i; k <= j; k += 1) out[order[k][1]] = avg;
			i = j + 1;
		}
		return out;
	};
	const rx = rank(xs);
	const ry = rank(ys);
	const n = rx.length;
	if (n < 3) return 0;
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
	return dx > 0 && dy > 0 ? num / Math.sqrt(dx * dy) : 0;
}

export function loadBeliefHeuristic(): BeliefHeuristicLoad {
	const prov = readGuarded(PROVENANCE_PATH, 'simple-scorer prediction provenance');
	if (!prov.ok) return unavailable(prov.reason);
	const goldRead = readGuarded(GOLD_PATH, 'paper statement gold');
	if (!goldRead.ok) return unavailable(goldRead.reason);

	// hash -> released paper label, the same join key paper-literal.ts uses.
	const gold = new Map<string, number>();
	for (const line of goldRead.text.split('\n')) {
		if (!line.trim()) continue;
		try {
			const row = JSON.parse(line) as Record<string, any>;
			const released = row?.paper_replication_policy?.released_paper_correct;
			const hash = row?.paper_statement_hash;
			if (released === null || released === undefined || hash === null || hash === undefined) continue;
			gold.set(String(hash), Number(released));
		} catch {
			continue;
		}
	}
	if (!gold.size) return unavailable('paper statement gold carried no released labels.');

	interface Acc {
		count: number;
		correct: number;
		belief: number;
	}
	// source -> n -> accumulator, for statements whose evidence is ALL one source.
	const ladders = new Map<string, Map<number, Acc>>();
	// belief value -> accumulator, across every statement (the degeneracy view).
	const byBelief = new Map<number, { count: number; correct: number }>();
	let nStatements = 0;

	for (const line of prov.text.split('\n')) {
		if (!line.trim()) continue;
		let row: Record<string, any>;
		try {
			row = JSON.parse(line) as Record<string, any>;
		} catch {
			continue;
		}
		const hash = row?.matches_hash;
		const belief = row?.probability_correct;
		const counts = row?.source_counts;
		if (hash === undefined || typeof belief !== 'number' || !counts) continue;
		const label = gold.get(String(hash));
		if (label === undefined) continue;
		nStatements += 1;

		// Key on the EXACT value. Rounding to 4 dp merges genuinely-distinct beliefs
		// (the widest 4 dp cluster spans 9.8e-05, ~11 orders above float noise) and
		// collapses 690 values into 86 — which would understate the heuristic's
		// granularity ~8x and invite a "coarse" claim that is simply false. The
		// heuristic is FINE-grained; the objection is that the grain is uninformative.
		// Verified: the modal value is unchanged either way (0.65, n=328, 47.3%).
		const key = belief;
		const seen = byBelief.get(key) ?? { count: 0, correct: 0 };
		seen.count += 1;
		seen.correct += label;
		byBelief.set(key, seen);

		const sources = Object.keys(counts);
		if (sources.length !== 1) continue; // a clean ladder needs one source
		const source = sources[0];
		const n = Number(counts[source]);
		if (!Number.isFinite(n) || n <= 0) continue;
		const ladder = ladders.get(source) ?? new Map<number, Acc>();
		const acc = ladder.get(n) ?? { count: 0, correct: 0, belief };
		acc.count += 1;
		acc.correct += label;
		acc.belief = belief; // identical for every statement on the rung, by construction
		ladder.set(n, acc);
		ladders.set(source, ladder);
	}

	if (!nStatements) return unavailable('no provenance rows joined the paper gold.');

	const sources: HeuristicSource[] = [];
	const singleRows: SingleEvidenceRow[] = [];
	/**
	 * The single-evidence belief, collected per source rather than overwritten.
	 * The panel's LEAD claim is that this number is IDENTICAL for every source
	 * drawn — "one shared prior, five very different readers" — and the previous
	 * `let singleBelief = 0` kept whichever source happened to come last, so a
	 * source with a different prior would have been silently absorbed into a claim
	 * of sameness (and, with no qualifying source, 0 would have printed as "0%").
	 */
	const singleBeliefs = new Set<number>();
	for (const [source, ladder] of ladders) {
		const rungs: HeuristicRung[] = [];
		const xs: number[] = [];
		const ys: number[] = [];
		let total = 0;
		let drawn = 0;
		for (const [n, acc] of [...ladder.entries()].sort((a, b) => a[0] - b[0])) {
			total += acc.count;
			for (let i = 0; i < acc.count; i += 1) {
				xs.push(n);
				ys.push(i < acc.correct ? 1 : 0);
			}
			if (acc.count < MIN_RUNG) continue;
			drawn += acc.count;
			rungs.push({ n, belief: acc.belief, count: acc.count, correctRate: acc.correct / acc.count });
		}
		const one = ladder.get(1);
		if (one && one.count >= MIN_SINGLE_EVIDENCE) {
			singleBeliefs.add(one.belief);
			singleRows.push({
				source,
				count: one.count,
				correct: one.correct,
				correctRate: one.correct / one.count
			});
		}
		if (total >= MIN_SOURCE && rungs.length >= 2) {
			sources.push({ source, total, rungs, rho: spearman(xs, ys), nUndrawn: total - drawn });
		}
	}
	sources.sort((a, b) => b.total - a.total);
	singleRows.sort((a, b) => b.correctRate - a.correctRate);
	if (!sources.length) return unavailable('no single-source ladder had enough statements to draw.');
	if (!singleRows.length) return unavailable('no source had enough single-evidence statements.');
	// The claim is sameness, so sameness is checked, not assumed. Exact equality is
	// the right test: at n=1 the noisy-OR is 1 − (syst + rand), so sources sharing a
	// prior pair land on a bit-identical double.
	if (singleBeliefs.size !== 1) {
		return unavailable(
			`the drawn sources do not share one single-evidence belief (${singleBeliefs.size} distinct values), so the shared-prior reading does not hold.`
		);
	}
	const [singleBelief] = singleBeliefs;

	// The modal belief value — the heuristic's most frequent single answer.
	let modal = { belief: 0, count: 0, correctRate: 0 };
	for (const [belief, acc] of byBelief) {
		if (acc.count > modal.count) {
			modal = { belief, count: acc.count, correctRate: acc.correct / acc.count };
		}
	}

	return {
		status: 'ok',
		sources,
		nStatements,
		distinctBeliefs: byBelief.size,
		modal,
		saturation: (() => {
			let nAbove = 0;
			let nBelow = 0;
			const above = new Set<number>();
			const below = new Set<number>();
			for (const [belief, acc] of byBelief) {
				if (belief > SATURATION_CUT) {
					nAbove += acc.count;
					above.add(belief);
				} else {
					nBelow += acc.count;
					below.add(belief);
				}
			}
			return {
				cut: SATURATION_CUT,
				nAbove,
				distinctAbove: above.size,
				nBelow,
				distinctBelow: below.size
			};
		})(),
		singleEvidence: {
			belief: singleBelief,
			total: singleRows.reduce((a, r) => a + r.count, 0),
			rows: singleRows,
			homogeneity: homogeneityChiSquare(singleRows)
		},
		filters: {
			minRung: MIN_RUNG,
			minSource: MIN_SOURCE,
			minSingleEvidence: MIN_SINGLE_EVIDENCE
		}
	};
}
