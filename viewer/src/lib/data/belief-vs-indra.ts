/**
 * Pure comparison math for INDRA's belief against a reading model — no imports,
 * no `$lib`, no filesystem, so it can be exercised directly by
 * `node --experimental-strip-types` against fixtures (see
 * `viewer/scripts/test-belief-vs-indra-contract.mjs`). The server loader
 * (`$lib/server/belief-vs-indra`) does the file reading and calls into here.
 *
 * INDRA's `SimpleScorer` is a noisy-OR over per-source priors:
 *
 *     belief = 1 − Π_s (syst_s + rand_s^{n_s})
 *
 * `rand_s` is the source's random error rate, which repetition can wear down —
 * a second independent reading of the same relation is unlikely to fail the
 * same way. `syst_s` is its systematic error rate, which repetition cannot
 * touch, because the reader makes that mistake every time. Belief is therefore
 * a function of the evidence-count profile alone, and the sentence is never
 * consulted.
 */

/** Ten equal-width bins over [0,1] — the frozen edges the comparison artifact declares. */
export const BIN_COUNT = 10;
/** A source needs at least this many statements in a cohort before its rate is drawn. */
export const MIN_SOURCE_N = 15;
/** "The reader judged this unsupported" — the band whose disagreement with INDRA is the point. */
export const LOW_BELIEF = 0.1;

export interface ReliabilityBin {
	/** Mean belief assigned inside the bin. */
	p_mean: number;
	/** Fraction of those statements the curators marked correct. */
	y_rate: number;
	n: number;
}

export interface BeliefSeries {
	id: string;
	label: string;
	bins: ReliabilityBin[];
	/** Count-weighted mean |observed − assigned| over occupied bins. */
	ece: number;
	n: number;
	min: number;
	max: number;
	/** Statements assigned a belief below 0.5 — whether the model can express doubt at all. */
	below_half: number;
	distinct_values: number;
}

export interface SourceRate {
	source: string;
	n: number;
	rate: number;
}

export interface IdenticalBeliefCohort {
	belief: number;
	n: number;
	share: number;
	observed_rate: number;
	/** Non-null only when every statement in the cohort carries the same count. */
	evidence_count: number | null;
	by_source: SourceRate[];
}

export interface DisagreementBand {
	threshold: number;
	n: number;
	observed_rate: number;
	indra_min: number;
	indra_median: number;
	indra_at_least_90: number;
}

/** One statement's source profile, as persisted in the prediction provenance. */
export interface StatementProfile {
	evidence_count: number | null;
	sources: string[];
}

/**
 * The probe battery's own bin partition, ported from `indra_belief.metrics.BINS_8`.
 * The frozen decision artifact reports ECE over THESE edges, not ten equal ones,
 * so reproducing its number requires reproducing its partition. Verified: both
 * arms' ECE match the artifact to ten decimal places.
 */
export const BINS_8: ReadonlyArray<readonly [number, number]> = [
	[0.0, 0.05],
	[0.05, 0.2],
	[0.2, 0.35],
	[0.35, 0.5],
	[0.5, 0.65],
	[0.65, 0.8],
	[0.8, 0.95],
	[0.95, 1.001]
];

export function round6(value: number): number {
	return Math.round(value * 1e6) / 1e6;
}

/** One holdout record: a score under some arm, and whether it was actually correct. */
export interface ScoredRecord {
	score: number;
	correct: boolean;
}

/**
 * Reliability over the probe battery's own eight bins. Separate from
 * `beliefSeries` on purpose: that one bins ten-equal over the paper corpus,
 * this one must reproduce a frozen artifact computed over BINS_8, and silently
 * sharing a partition between two different evaluations would make the two
 * ECEs look comparable when they are not.
 */
export function probeSeries(id: string, label: string, records: ScoredRecord[]): BeliefSeries {
	const nb = BINS_8.length;
	const sum = new Array<number>(nb).fill(0);
	const positives = new Array<number>(nb).fill(0);
	const counts = new Array<number>(nb).fill(0);
	const distinct = new Set<number>();
	let min = Number.POSITIVE_INFINITY;
	let max = Number.NEGATIVE_INFINITY;
	let belowHalf = 0;

	for (const { score, correct } of records) {
		const p = clamp01(score);
		let index = nb - 1;
		for (let k = 0; k < nb; k += 1) {
			if (p >= BINS_8[k][0] && p < BINS_8[k][1]) {
				index = k;
				break;
			}
		}
		sum[index] += p;
		if (correct) positives[index] += 1;
		counts[index] += 1;
		distinct.add(round6(p));
		if (p < min) min = p;
		if (p > max) max = p;
		if (p < 0.5) belowHalf += 1;
	}

	const n = records.length;
	const bins: ReliabilityBin[] = [];
	let ece = 0;
	for (let i = 0; i < nb; i += 1) {
		if (counts[i] === 0) continue;
		const pMean = sum[i] / counts[i];
		const yRate = positives[i] / counts[i];
		bins.push({ p_mean: round6(pMean), y_rate: round6(yRate), n: counts[i] });
		ece += (counts[i] / n) * Math.abs(yRate - pMean);
	}

	return {
		id,
		label,
		bins,
		ece: round6(ece),
		n,
		min: n ? round6(min) : 0,
		max: n ? round6(max) : 0,
		below_half: belowHalf,
		distinct_values: distinct.size
	};
}

function clamp01(value: number): number {
	return value < 0 ? 0 : value > 1 ? 1 : value;
}

/**
 * Reliability over BIN_COUNT equal-width [0,1] bins from an aligned
 * (belief, curated-label) vector. Empty bins are dropped; ECE is the
 * count-weighted mean |observed − assigned| across occupied bins.
 */
export function beliefSeries(
	id: string,
	label: string,
	predictions: Map<string, number>,
	gold: Map<string, number>
): BeliefSeries {
	const sum = new Array<number>(BIN_COUNT).fill(0);
	const positives = new Array<number>(BIN_COUNT).fill(0);
	const counts = new Array<number>(BIN_COUNT).fill(0);
	let n = 0;
	let min = Number.POSITIVE_INFINITY;
	let max = Number.NEGATIVE_INFINITY;
	let belowHalf = 0;
	const distinct = new Set<number>();

	for (const [sid, raw] of predictions) {
		const observed = gold.get(sid);
		if (observed === undefined) continue;
		const p = clamp01(raw);
		let index = Math.floor(p * BIN_COUNT);
		if (index >= BIN_COUNT) index = BIN_COUNT - 1;
		if (index < 0) index = 0;
		sum[index] += p;
		if (observed > 0) positives[index] += 1;
		counts[index] += 1;
		n += 1;
		if (p < min) min = p;
		if (p > max) max = p;
		if (p < 0.5) belowHalf += 1;
		distinct.add(round6(p));
	}

	const bins: ReliabilityBin[] = [];
	let ece = 0;
	for (let i = 0; i < BIN_COUNT; i += 1) {
		if (counts[i] === 0) continue;
		const pMean = sum[i] / counts[i];
		const yRate = positives[i] / counts[i];
		bins.push({ p_mean: round6(pMean), y_rate: round6(yRate), n: counts[i] });
		ece += (counts[i] / n) * Math.abs(yRate - pMean);
	}

	return {
		id,
		label,
		bins,
		ece: round6(ece),
		n,
		min: n ? round6(min) : 0,
		max: n ? round6(max) : 0,
		below_half: belowHalf,
		distinct_values: distinct.size
	};
}

/**
 * The single most common belief the counting model assigns, and how those
 * statements actually turned out, broken down by the source that reported them.
 * Every statement in the cohort carries the identical number by construction;
 * the spread of curated outcomes across sources is what that number cannot say.
 */
export function identicalBeliefCohort(
	predictions: Map<string, number>,
	gold: Map<string, number>,
	profiles: Map<string, StatementProfile> | null
): IdenticalBeliefCohort | null {
	const tally = new Map<number, string[]>();
	for (const [sid, p] of predictions) {
		if (!gold.has(sid)) continue;
		const key = round6(p);
		const bucket = tally.get(key);
		if (bucket) bucket.push(sid);
		else tally.set(key, [sid]);
	}

	let belief = 0;
	let members: string[] = [];
	for (const [value, sids] of tally) {
		// Ties break toward the lower belief: deterministic, and the lower value is
		// the one nearer the model's floor, which is the phenomenon under study.
		if (sids.length > members.length || (sids.length === members.length && value < belief)) {
			members = sids;
			belief = value;
		}
	}
	if (members.length === 0) return null;

	const correct = members.reduce((acc, sid) => acc + (gold.get(sid) ?? 0), 0);
	const bySource: SourceRate[] = [];
	let evidenceCount: number | null = null;

	if (profiles) {
		const sourceMembers = new Map<string, string[]>();
		const evidenceCounts = new Set<number>();
		for (const sid of members) {
			const profile = profiles.get(sid);
			if (!profile) continue;
			if (profile.evidence_count !== null) evidenceCounts.add(profile.evidence_count);
			for (const source of profile.sources) {
				const bucket = sourceMembers.get(source);
				if (bucket) bucket.push(sid);
				else sourceMembers.set(source, [sid]);
			}
		}
		// Only claim a single evidence count when the whole cohort agrees on it.
		evidenceCount = evidenceCounts.size === 1 ? [...evidenceCounts][0] : null;
		for (const [source, sids] of sourceMembers) {
			if (sids.length < MIN_SOURCE_N) continue;
			const hits = sids.reduce((acc, sid) => acc + (gold.get(sid) ?? 0), 0);
			bySource.push({ source, n: sids.length, rate: round6(hits / sids.length) });
		}
		bySource.sort((a, b) => a.rate - b.rate || a.source.localeCompare(b.source));
	}

	return {
		belief,
		n: members.length,
		share: round6(members.length / gold.size),
		observed_rate: round6(correct / members.length),
		evidence_count: evidenceCount,
		by_source: bySource
	};
}

/**
 * Statements the reading model puts near zero, and what the counting model says
 * about those same statements. The curated outcome says which is closer.
 */
export function disagreementBand(
	reader: Map<string, number>,
	counting: Map<string, number>,
	gold: Map<string, number>,
	threshold: number = LOW_BELIEF
): DisagreementBand | null {
	const members: string[] = [];
	for (const [sid, p] of reader) {
		if (gold.has(sid) && p < threshold) members.push(sid);
	}
	if (members.length === 0) return null;

	const countingValues = members
		.map((sid) => counting.get(sid))
		.filter((v): v is number => v !== undefined)
		.sort((a, b) => a - b);
	if (countingValues.length === 0) return null;

	const correct = members.reduce((acc, sid) => acc + (gold.get(sid) ?? 0), 0);
	const mid = Math.floor(countingValues.length / 2);
	const median =
		countingValues.length % 2 === 0
			? (countingValues[mid - 1] + countingValues[mid]) / 2
			: countingValues[mid];

	return {
		threshold,
		n: members.length,
		observed_rate: round6(correct / members.length),
		indra_min: round6(countingValues[0]),
		indra_median: round6(median),
		indra_at_least_90: countingValues.filter((v) => v >= 0.9).length
	};
}
