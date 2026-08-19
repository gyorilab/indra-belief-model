/**
 * INDRA's belief and ours, on the same statements, against the same curations.
 *
 * Thin I/O around `$lib/data/belief-vs-indra`, which holds the math and is
 * exercised directly by `scripts/test-belief-vs-indra-contract.mjs`. This module
 * reads three committed artifacts and joins them on the canonical
 * `statement_id`:
 *
 *   · INDRA 1.24.0 SimpleScorer predictions at the priors the library ships
 *   · the reading model's predictions over the identical statement set
 *   · the released paper gold, `paper_replication_policy.released_paper_correct`
 *
 * It fits nothing, calls no scorer, and hard-codes no rate — every number in the
 * payload is derived from those files or the payload reports itself unavailable.
 *
 * Conventions mirror `paper-literal.ts` and `belief-heuristic.ts`: DATA_DIR
 * joins, a size cap, and an `unavailable()` payload rather than a throw.
 */
import { existsSync, readFileSync, statSync } from 'node:fs';
import { join } from 'node:path';

import {
	beliefSeries,
	disagreementBand,
	identicalBeliefCohort,
	probeSeries,
	round6,
	type BeliefSeries,
	type DisagreementBand,
	type IdenticalBeliefCohort,
	type ScoredRecord,
	type StatementProfile
} from '$lib/data/belief-vs-indra';
import { DATA_DIR } from '$lib/data/runs';

const GOLD_PATH = join(
	DATA_DIR,
	'results',
	'indra_paper_statement_gold_20260717',
	'paper_statement_gold.jsonl'
);
const INDRA_PATH = join(
	DATA_DIR,
	'results',
	'current_indra_simple_paper_20260717',
	'current_indra_simple_default_predictions.jsonl'
);
const PROVENANCE_PATH = join(
	DATA_DIR,
	'results',
	'current_indra_simple_paper_20260717',
	'current_indra_simple_default_prediction_provenance.jsonl'
);
const READER_PATH = join(
	DATA_DIR,
	'comparison',
	'models',
	'gemma_4_26b',
	'all_source_predictions.jsonl'
);

const AGGREGATION_PATH = join(DATA_DIR, 'comparison', 'aggregation.json');
const READER_MANIFEST_PATH = join(DATA_DIR, 'comparison', 'models', 'gemma_4_26b', 'manifest.json');
const PROBE_SCORES_PATH = join(
	DATA_DIR,
	'probe_battery',
	'holdout_scores_C_incumbent_plus_battery.jsonl'
);
const PROBE_DECISION_PATH = join(
	DATA_DIR,
	'probe_battery',
	'decision_C_incumbent_plus_battery.json'
);

/** The gold file is ~4 MB; keep the cap above it but bounded. */
const MAX_ARTIFACT_BYTES = 16 * 1024 * 1024;

/**
 * The single-probe ablation from research/probe_battery_findings.md §1b. Quoted
 * rather than recomputed: the per-probe score vectors it was derived from are
 * not committed, so recomputing here is impossible and inventing it is worse.
 * These are the DEFENSIBLE headline; the 16-probe arm's delta is larger.
 */
const SINGLE_PROBE_DELTA = 0.045872;
const SINGLE_PROBE_CI: [number, number] = [0.020968, 0.069776];

/**
 * How the reading series was actually aggregated — read from the frozen
 * artifacts, never asserted. This matters more than it looks: the reading arm
 * here does NOT use the fitted log-odds calibration. It runs INDRA's own
 * aggregation over a filtered evidence set, so the page must describe that
 * mechanism and not the fitted one.
 */
export interface Mechanism {
	/** e.g. "indra_default_hard_gate" — from data/comparison/aggregation.json. */
	aggregation: string;
	/** e.g. "indra_belief.statement_belief:statement_belief" — from the model manifest. */
	implementation: string | null;
	/** Statements where every source was removed, leaving an empty product. */
	all_evidence_rejected: number;
}

/**
 * A frozen token-probability experiment that predates the W1 scoring path.
 *
 * A separate call with reasoning disabled, reading the model's probability at a
 * single forced verdict token. It is scored on its OWN holdout — different
 * statements, different curations, a different bin partition — so it must never
 * share axes with the corpus comparison above. Its interest here is historical:
 * it records why the verdict grid was rejected, but its grid-plus-probe candidate
 * is not the calibrated single-score contract now used by serving.
 */
export interface ProbePanel {
	n: number;
	/** The frozen decision artifact's own verdict, e.g. "GO". */
	verdict: string | null;
	historical: true;
	/** Boundary between this old experiment and the current sentence scorer. */
	historical_note: string;
	incumbent: BeliefSeries;
	candidate: BeliefSeries;
	delta_auroc: number | null;
	ci95: [number, number] | null;
	incumbent_seconds: number | null;
	candidate_seconds: number | null;
	/** The ablation that moved the claim: one probe carries essentially the whole gain. */
	single_probe_delta: number | null;
	single_probe_ci95: [number, number] | null;
}

export interface BeliefVsIndra {
	available: true;
	n: number;
	base_rate: number;
	indra: BeliefSeries;
	reader: BeliefSeries;
	mechanism: Mechanism;
	cohort: IdenticalBeliefCohort | null;
	disagreement: DisagreementBand | null;
	probe: ProbePanel | null;
}

export interface BeliefVsIndraUnavailable {
	available: false;
	reason: string;
}

export type BeliefVsIndraLoad = BeliefVsIndra | BeliefVsIndraUnavailable;

function unavailable(reason: string): BeliefVsIndraUnavailable {
	return { available: false, reason };
}

function readCapped(path: string): string | null {
	if (!existsSync(path)) return null;
	if (statSync(path).size > MAX_ARTIFACT_BYTES) return null;
	return readFileSync(path, 'utf8');
}

function asRecord(value: unknown): Record<string, unknown> | null {
	return typeof value === 'object' && value !== null && !Array.isArray(value)
		? (value as Record<string, unknown>)
		: null;
}

function finite(value: unknown): number | null {
	return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

/** statement_id -> released_paper_correct. The RELEASED label, matching every other arm. */
function loadGold(text: string): Map<string, number> {
	const out = new Map<string, number>();
	for (const line of text.split('\n')) {
		if (!line.trim()) continue;
		let row: Record<string, unknown> | null;
		try {
			row = asRecord(JSON.parse(line));
		} catch {
			continue;
		}
		if (!row) continue;
		const sid = asRecord(row.canonical_corpus)?.statement_id;
		const label = finite(asRecord(row.paper_replication_policy)?.released_paper_correct);
		if (typeof sid === 'string' && label !== null) out.set(sid, label);
	}
	return out;
}

/** statement_id -> probability_correct. */
function loadPredictions(text: string): Map<string, number> {
	const out = new Map<string, number>();
	for (const line of text.split('\n')) {
		if (!line.trim()) continue;
		let row: Record<string, unknown> | null;
		try {
			row = asRecord(JSON.parse(line));
		} catch {
			continue;
		}
		if (!row) continue;
		const sid = row.statement_id;
		const p = finite(row.probability_correct);
		if (typeof sid === 'string' && p !== null) out.set(sid, p);
	}
	return out;
}

/** statement_id -> its evidence count and the sources that reported it. */
function loadProfiles(text: string | null): Map<string, StatementProfile> | null {
	if (!text) return null;
	const out = new Map<string, StatementProfile>();
	for (const line of text.split('\n')) {
		if (!line.trim()) continue;
		let row: Record<string, unknown> | null;
		try {
			row = asRecord(JSON.parse(line));
		} catch {
			continue;
		}
		if (!row) continue;
		const sid = row.statement_id;
		if (typeof sid !== 'string') continue;
		const counts = asRecord(row.source_counts);
		out.set(sid, {
			evidence_count: finite(row.evidence_count),
			sources: counts ? Object.keys(counts) : []
		});
	}
	return out.size ? out : null;
}

/** The aggregation that actually produced the reading series, read from the artifacts. */
function loadMechanism(zeroCount: number): Mechanism {
	let aggregation = 'unknown';
	let implementation: string | null = null;
	try {
		const agg = asRecord(JSON.parse(readFileSync(AGGREGATION_PATH, 'utf8')));
		if (typeof agg?.aggregation === 'string') aggregation = agg.aggregation;
	} catch {
		/* leave as unknown — the page says so rather than guessing */
	}
	try {
		const man = asRecord(JSON.parse(readFileSync(READER_MANIFEST_PATH, 'utf8')));
		const impl = asRecord(man?.implementation)?.implementation;
		if (typeof impl === 'string') implementation = impl;
	} catch {
		/* same */
	}
	return { aggregation, implementation, all_evidence_rejected: zeroCount };
}

function loadProbePanel(): ProbePanel | null {
	const scoresText = readCapped(PROBE_SCORES_PATH);
	const decisionText = readCapped(PROBE_DECISION_PATH);
	if (!scoresText || !decisionText) return null;

	const incumbent: ScoredRecord[] = [];
	const candidate: ScoredRecord[] = [];
	for (const line of scoresText.split('\n')) {
		if (!line.trim()) continue;
		let row: Record<string, unknown> | null;
		try {
			row = asRecord(JSON.parse(line));
		} catch {
			continue;
		}
		if (!row) continue;
		const correct = row.gold_correct === true;
		const inc = finite(row.incumbent_score);
		const cand = finite(row.candidate_score);
		if (inc !== null) incumbent.push({ score: inc, correct });
		if (cand !== null) candidate.push({ score: cand, correct });
	}
	if (!incumbent.length || !candidate.length) return null;

	let verdict: string | null = null;
	let delta: number | null = null;
	let ci: [number, number] | null = null;
	let incSec: number | null = null;
	let candSec: number | null = null;
	try {
		const d = asRecord(JSON.parse(decisionText));
		if (typeof d?.verdict === 'string') verdict = d.verdict;
		const gate = asRecord(d?.gate);
		if (verdict === null && typeof gate?.verdict === 'string') verdict = gate.verdict;
		const pb = asRecord(d?.paired_bootstrap);
		delta = finite(pb?.delta_auroc);
		const lo = finite(pb?.ci95_low);
		const hi = finite(pb?.ci95_high);
		if (lo !== null && hi !== null) ci = [lo, hi];
		const cost = asRecord(d?.cost);
		incSec = finite(cost?.incumbent_s_per_record);
		candSec = finite(cost?.candidate_s_per_record);
	} catch {
		/* metrics stay null; the panel renders without them rather than inventing */
	}

	return {
		n: candidate.length,
		verdict,
		historical: true,
		historical_note:
			'this frozen grid-plus-probe candidate predates W1 and is not the calibrated ' +
			'single-score contract now emitted by the sentence scorer',
		incumbent: probeSeries('probe_incumbent', 'Historical verdict grid', incumbent),
		candidate: probeSeries('probe_candidate', 'Historical grid plus the probe', candidate),
		delta_auroc: delta,
		ci95: ci,
		incumbent_seconds: incSec,
		candidate_seconds: candSec,
		// From the final integrated review's ablation, recorded in
		// research/probe_battery_findings.md §1b. A single probe carries
		// essentially the whole gain; the other fifteen add a CI spanning zero.
		single_probe_delta: SINGLE_PROBE_DELTA,
		single_probe_ci95: SINGLE_PROBE_CI
	};
}

export function loadBeliefVsIndra(): BeliefVsIndraLoad {
	const goldText = readCapped(GOLD_PATH);
	if (!goldText) return unavailable('released paper gold is not available');
	const indraText = readCapped(INDRA_PATH);
	if (!indraText) return unavailable('INDRA SimpleScorer predictions are not available');
	const readerText = readCapped(READER_PATH);
	if (!readerText) return unavailable('reading-model predictions are not available');

	const gold = loadGold(goldText);
	if (gold.size === 0) return unavailable('gold file parsed to zero labelled statements');
	const indra = loadPredictions(indraText);
	const reader = loadPredictions(readerText);

	const indraSeries = beliefSeries(
		'indra_simple_default',
		'INDRA SimpleScorer, shipped priors',
		indra,
		gold
	);
	const readerSeries = beliefSeries('reader_gemma_4_26b', 'Reading the evidence', reader, gold);

	if (indraSeries.n === 0 || readerSeries.n === 0) {
		return unavailable('predictions did not join to the gold statement set');
	}
	if (indraSeries.n !== readerSeries.n) {
		return unavailable(
			`the two models cover different statement sets (${indraSeries.n} vs ${readerSeries.n}); ` +
				'a calibration comparison across different sets would not be a comparison'
		);
	}

	let correct = 0;
	for (const label of gold.values()) correct += label > 0 ? 1 : 0;

	// Every source removed leaves an empty product, and 1 − 1 = 0. This is the
	// only way a belief of exactly zero can arise, so counting it counts the
	// statements where the reader rejected every piece of evidence.
	let allRejected = 0;
	for (const [sid, p] of reader) if (gold.has(sid) && p === 0) allRejected += 1;

	return {
		available: true,
		n: indraSeries.n,
		base_rate: round6(correct / gold.size),
		indra: indraSeries,
		reader: readerSeries,
		mechanism: loadMechanism(allRejected),
		cohort: identicalBeliefCohort(indra, gold, loadProfiles(readCapped(PROVENANCE_PATH))),
		disagreement: disagreementBand(reader, indra, gold),
		probe: loadProbePanel()
	};
}
