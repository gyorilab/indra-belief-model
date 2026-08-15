/**
 * The query API — every payload the viewer needs, derived from the monolithic
 * JSONL exports.
 *
 * Everything here is pure projection/aggregation over the in-memory store; no
 * database, no SQL, no native addon. Fields the monolithic export cannot
 * provide (probe traces, gold F1, calibration-by-type strata) are surfaced as
 * explicit `unavailable` markers, not faked.
 */
import { listRuns, latestRun, resolveRun } from './runs';
import {
	getRunData,
	getEvidenceIndex,
	evidenceForStatement,
	getCurationIndex,
	getRunMetrics,
	goldForRow
} from './store';
import type {
	RunMeta,
	StatementRollup,
	EvidenceRow,
	GoldVerdict,
	ReasoningTrace,
	RunMetrics
} from './types';
import {
	calibrationArtifactConsistency,
	calibrationCompatibility,
	classifyCalibrationEvaluation,
	metricArm,
	selectCalibrationPredecessor,
	type CalibrationArtifactConsistency,
	type CalibrationCompatibility,
	type CalibrationEvaluation,
	type CalibrationTier
} from './calibration';

// ── Shared shapes ───────────────────────────────────────────────────────────

export interface RunSummary {
	run_id: string;
	model: string;
	status: string | null;
	generated_date: string | null;
	export_schema_version: number | null;
	source_run: string | null;
	provenance: RunMeta['provenance'];
	n_statements: number;
	n_evidences: number;
	bucket_counts: Record<string, number>;
	/** Run-level observed cost (null on legacy exports / no exporter cost). */
	cost: RunMeta['cost'];
	soft_calibration: RunMeta['soft_calibration'];
}

function runSummary(m: RunMeta): RunSummary {
	return {
		run_id: m.run_id,
		model: m.model,
		status: m.status,
		generated_date: m.generated_date,
		export_schema_version: m.export_schema_version,
		source_run: m.source_run,
		provenance: m.provenance ?? null,
		n_statements: m.counts.statements ?? 0,
		n_evidences: m.counts.unique_evidence_rows ?? 0,
		bucket_counts: m.bucket_counts,
		cost: m.cost ?? null,
		soft_calibration: m.soft_calibration
	};
}

export function getRuns(): RunSummary[] {
	return listRuns().map(runSummary);
}

// ── Overview ────────────────────────────────────────────────────────────────

export interface Overview {
	statementCount: number;
	evidenceCount: number;
	runs: RunSummary[];
	latest: RunSummary | null;
}

export function getOverview(): Overview {
	const runs = listRuns();
	const latest = runs[0] ?? null;
	return {
		statementCount: latest?.counts.statements ?? 0,
		evidenceCount: latest?.counts.unique_evidence_rows ?? 0,
		runs: runs.map(runSummary),
		latest: latest ? runSummary(latest) : null
	};
}

// ── Calibration / validity ──────────────────────────────────────────────────

const SCORE_OK = (r: EvidenceRow): boolean =>
	r.our_score != null && r.rasmachine_belief != null && r.verdict != null;

function residual(r: EvidenceRow): number {
	return (r.our_score as number) - (r.rasmachine_belief as number);
}

interface SliceStat {
	value: string;
	n: number;
	mae: number;
	bias: number;
}

function sliceStats(rows: EvidenceRow[], key: (r: EvidenceRow) => string): SliceStat[] {
	const groups = new Map<string, EvidenceRow[]>();
	for (const r of rows) {
		const k = key(r) || '—';
		let arr = groups.get(k);
		if (!arr) groups.set(k, (arr = []));
		arr.push(r);
	}
	const out: SliceStat[] = [];
	for (const [value, rs] of groups) {
		let mae = 0;
		let bias = 0;
		for (const r of rs) {
			const d = residual(r);
			mae += Math.abs(d);
			bias += d;
		}
		out.push({ value, n: rs.length, mae: mae / rs.length, bias: bias / rs.length });
	}
	return out.sort((a, b) => b.mae - a.mae);
}

export interface Validity {
	run_id: string;
	model: string;
	verdicts: Array<{ verdict: string; n: number }>;
	calibration: { mae: number | null; bias: number | null; n: number };
	byIndraType: SliceStat[];
	bySourceApi: SliceStat[];
	confidenceCalibration: Array<{ confidence: string; n: number; mae: number | null; bias: number | null }>;
	buckets: Array<{ bucket: string; n: number }>;
	/** Strata the monolithic export cannot provide. */
	unavailable: string[];
}

const CONF_ORDER: Record<string, number> = { high: 0, medium: 1, low: 2 };

export function getValidity(runId?: string): Validity | null {
	const meta = resolveRun(runId);
	if (!meta) return null;
	const rows = getEvidenceIndex(meta).all;
	const scored = rows.filter(SCORE_OK);

	// verdict distribution
	const vCounts = new Map<string, number>();
	for (const r of rows) {
		const v = r.verdict ?? 'unscored';
		vCounts.set(v, (vCounts.get(v) ?? 0) + 1);
	}

	// global calibration
	let mae = 0;
	let bias = 0;
	for (const r of scored) {
		const d = residual(r);
		mae += Math.abs(d);
		bias += d;
	}
	const n = scored.length;

	// confidence calibration (all-family buckets)
	const byConf = sliceStats(
		scored.filter((r) => r.confidence),
		(r) => r.confidence as string
	)
		.map((s) => ({ confidence: s.value, n: s.n, mae: s.mae, bias: s.bias }))
		.sort((a, b) => (CONF_ORDER[a.confidence] ?? 9) - (CONF_ORDER[b.confidence] ?? 9));

	// bucket distribution (report taxonomy)
	const bCounts = new Map<string, number>();
	for (const r of rows) {
		const b = r.bucket ?? 'unclassified';
		bCounts.set(b, (bCounts.get(b) ?? 0) + 1);
	}

	return {
		run_id: meta.run_id,
		model: meta.model,
		verdicts: [...vCounts].map(([verdict, n]) => ({ verdict, n })),
		calibration: { mae: n ? mae / n : null, bias: n ? bias / n : null, n },
		byIndraType: sliceStats(scored, (r) => r.stmt_type),
		bySourceApi: sliceStats(scored, (r) => r.source_api ?? '—'),
		confidenceCalibration: byConf,
		buckets: [...bCounts].map(([bucket, n]) => ({ bucket, n })).sort((a, b) => b.n - a.n),
		unavailable: [
			'calibration-by-type strata vs gold (no gold labels in monolithic export)',
			'inter-evidence consistency, supports-graph plausibility'
		]
	};
}

// ── Residual distribution ───────────────────────────────────────────────────

export interface ResidualDistribution {
	run_id: string;
	bins: number[];
	n_total: number;
	mean_residual: number | null;
}

export function getResidualDistribution(runId?: string): ResidualDistribution | null {
	const meta = resolveRun(runId);
	if (!meta) return null;
	const scored = getEvidenceIndex(meta).all.filter(SCORE_OK);
	const N = 11;
	const bins = new Array(N).fill(0);
	let sum = 0;
	for (const r of scored) {
		const d = residual(r); // [-1, 1]
		sum += d;
		let idx = Math.floor(((d + 1) / 2) * N);
		if (idx < 0) idx = 0;
		if (idx >= N) idx = N - 1;
		bins[idx]++;
	}
	return {
		run_id: meta.run_id,
		bins,
		n_total: scored.length,
		mean_residual: scored.length ? sum / scored.length : null
	};
}

// ── Calibration surface (C4) — served byte-exact from metrics.json (G4) ──────
//
// The viewer NEVER recomputes a calibration number: every ECE/Brier/reliability
// bin/confusion cell here is passed through from the per-run `metrics.json`
// (results.build_run_metrics). The only thing this query DERIVES is the P5
// delta-vs-prev — and even that is the difference of two byte-exact served ECEs,
// not a recomputed metric. The per-stratum strip is the residual stratification
// (a different measure than gold-ECE; see getValidity.byIndraType), surfaced
// here to retire the `unavailable` apology.

export type Tier = CalibrationTier;

/** P5 delta: this run's headline ECE minus the previous run's (same tier+arm),
 *  both read byte-exact from metrics.json. null when there's no comparable prev. */
export interface EceDelta {
	prev_run_id: string;
	prev_model: string;
	arm: string;
	prev_ece: number;
	delta: number; // this_ece − prev_ece (negative = improved)
}

export interface RunCalibration {
	run_id: string;
	model: string;
	/** Whole metrics.json, served verbatim. null ⇒ export predates the cal arc. */
	metrics: RunMetrics | null;
	/** Whether a metrics.json was present at all (drives the named-empty copy). */
	present: boolean;
	/** End-to-end export/metrics/profile contract validation. */
	consistency: CalibrationArtifactConsistency;
	/** Whether these numbers are fit diagnostics or independent validation. */
	evaluation: CalibrationEvaluation;
	/** P5: ECE delta vs the previous run, per tier, for a common canonical arm
	 *  on the same model/substrate/gold/schema/contract. Tier-2 promotes the v3+
	 *  hybrid only when both runs carry it; v2 remains on its legacy hard arm.
	 *  null when no unique, genuinely earlier comparable run exists. */
	delta: { ev: EceDelta | null; stmt: EceDelta | null };
}

function armEce(m: RunMetrics | null, tier: Tier, arm: string): number | null {
	return metricArm(m, tier, arm)?.ece ?? null;
}

export function getRunCalibration(runId?: string): RunCalibration | null {
	const meta = resolveRun(runId);
	if (!meta) return null;
	const metrics = getRunMetrics(meta);

	// A predecessor is selected independently per tier. It must be the same
	// model on the same substrate/gold/metrics contract, have a
	// common canonical arm, and be uniquely earlier by run timestamp. Directory
	// or same-day registry order is never used as chronology.
	const runs = listRuns();

	const deltaFor = (tier: Tier): EceDelta | null => {
		const predecessor = selectCalibrationPredecessor(
			meta,
			runs,
			metrics,
			tier,
			getRunMetrics
		);
		if (!predecessor) return null;
		const { run: prev, metrics: prevMetrics, arm } = predecessor;
		const here = armEce(metrics, tier, arm);
		const there = armEce(prevMetrics, tier, arm);
		if (here == null || there == null) return null;
		return {
			prev_run_id: prev.run_id,
			prev_model: prev.model,
			arm,
			prev_ece: there,
			delta: here - there
		};
	};

	return {
		run_id: meta.run_id,
		model: meta.model,
		metrics,
		present: metrics != null,
		consistency: calibrationArtifactConsistency(meta, metrics),
		evaluation: classifyCalibrationEvaluation(meta, metrics),
		delta: { ev: deltaFor('ev'), stmt: deltaFor('stmt') }
	};
}

// ── Run-comparison calibration (C5) — ?mode=calib on /compare ────────────────
//
// Mirrors getRunCalibration but for an A/B PAIR. Both runs' metrics.json are
// served VERBATIM (getRunMetrics); the only derived quantities are ΔECE/ΔBrier
// = the difference of the two SERVED values per tier+arm (gate G5v: viewer
// deltas == the C1/C2 held-out script outputs). Nothing is recomputed.

/** ΔECE/ΔBrier for one tier+arm: B minus A over the two byte-exact served arms.
 *  delta_* convention matches C4 (negative = B better, since lower ECE/Brier
 *  is better). null when either side lacks the arm. */
export interface CalibDelta {
	arm: string;
	a_ece: number | null;
	b_ece: number | null;
	delta_ece: number | null; // b − a (negative ⇒ B better calibrated)
	a_brier: number | null;
	b_brier: number | null;
	delta_brier: number | null; // b − a (negative ⇒ B better)
}

export interface CompareCalibration {
	run_a: { run_id: string; model: string };
	run_b: { run_id: string; model: string };
	/** Whole metrics.json per side, served verbatim. null ⇒ predates cal arc. */
	a_metrics: RunMetrics | null;
	b_metrics: RunMetrics | null;
	a_present: boolean;
	b_present: boolean;
	a_evaluation: CalibrationEvaluation;
	b_evaluation: CalibrationEvaluation;
	/** Provenance/contract gate. Deltas and overlays render only when the base
	 *  contract and the selected tier's common canonical arm are compatible. */
	compatibility: CalibrationCompatibility;
	/** Headline ΔECE/ΔBrier per tier. Tier-2 compares calibrated arms when
	 *  both v3+ products carry one, otherwise their common hard fallback. */
	delta: { ev: CalibDelta | null; stmt: CalibDelta | null };
}

export function compareCalibration(runIdA: string, runIdB: string): CompareCalibration | null {
	const a = resolveRun(runIdA);
	const b = resolveRun(runIdB);
	if (!a || !b) return null;
	const am = getRunMetrics(a);
	const bm = getRunMetrics(b);
	const compatibility = calibrationCompatibility(a, b, am, bm);

	const deltaFor = (tier: Tier): CalibDelta | null => {
		const tierCompatibility = compatibility.tiers[tier];
		const arm = tierCompatibility.arm;
		if (!compatibility.compatible || !tierCompatibility.compatible || !arm) return null;
		const aa = metricArm(am, tier, arm);
		const ba = metricArm(bm, tier, arm);
		if (!aa || !ba) return null;
		const aE = aa.ece;
		const bE = ba.ece;
		const aB = aa.brier;
		const bB = ba.brier;
		return {
			arm,
			a_ece: aE,
			b_ece: bE,
			delta_ece: bE - aE,
			a_brier: aB,
			b_brier: bB,
			delta_brier: bB - aB
		};
	};

	return {
		run_a: { run_id: a.run_id, model: a.model },
		run_b: { run_id: b.run_id, model: b.model },
		a_metrics: am,
		b_metrics: bm,
		a_present: am != null,
		b_present: bm != null,
		a_evaluation: classifyCalibrationEvaluation(a, am),
		b_evaluation: classifyCalibrationEvaluation(b, bm),
		compatibility,
		delta: { ev: deltaFor('ev'), stmt: deltaFor('stmt') }
	};
}

// ── Statement matrix ────────────────────────────────────────────────────────

export interface MatrixRow {
	stmt_hash: string;
	subject: string;
	stmt_type: string;
	object: string;
	indra_belief: number | null;
	our_score: number | null;
	delta: number | null;
	n_evidence: number;
	n_incorrect: number;
	dominant_bucket: string | null;
	sources: string[];
}

function toMatrixRow(s: StatementRollup): MatrixRow {
	const our = s.our_mean_score;
	const indra = s.rasmachine_belief;
	return {
		stmt_hash: s.stmt_hash,
		subject: s.subject,
		stmt_type: s.stmt_type,
		object: s.object,
		indra_belief: indra,
		our_score: our,
		delta: our != null && indra != null ? our - indra : null,
		n_evidence: s.n_evidence,
		n_incorrect: s.n_incorrect,
		dominant_bucket: s.dominant_bucket,
		sources: s.sources
	};
}

export function getStatementMatrix(runId?: string): { run_id: string; rows: MatrixRow[] } | null {
	const meta = resolveRun(runId);
	if (!meta) return null;
	return { run_id: meta.run_id, rows: getRunData(meta).perStatement.map(toMatrixRow) };
}

// ── Focus statement (biggest disagreement) ──────────────────────────────────

export interface FocusEvidence {
	evidence_hash: string;
	source_api: string | null;
	text: string | null;
	our_score: number | null;
	verdict: string | null;
	confidence: string | null;
	bucket: string | null;
	reasoning: string | null;
	tier: string | null;
	grounding_status: string | null;
}

export interface Focus {
	run_id: string;
	stmt: { stmt_hash: string; indra_type: string; subject: string; object: string };
	our_score: number | null;
	indra_score: number | null;
	n_evidences: number;
	evidences: FocusEvidence[];
	why_this_one: string;
}

function toFocusEvidence(r: EvidenceRow): FocusEvidence {
	return {
		evidence_hash: r.evidence_hash,
		source_api: r.source_api,
		text: r.evidence_text,
		our_score: r.our_score,
		verdict: r.verdict,
		confidence: r.confidence,
		bucket: r.bucket,
		reasoning: r.reasoning,
		tier: r.tier,
		grounding_status: r.grounding_status
	};
}

export function getFocus(runId?: string, focusHash?: string): Focus | null {
	const meta = resolveRun(runId);
	if (!meta) return null;
	const run = getRunData(meta);
	let s: StatementRollup | undefined;
	let why = '';
	if (focusHash) {
		s = run.byHash.get(focusHash);
	}
	if (!s) {
		// biggest |our_mean - rasmachine_belief|
		let best: StatementRollup | null = null;
		let bestD = -1;
		for (const r of run.perStatement) {
			if (r.our_mean_score == null || r.rasmachine_belief == null) continue;
			const d = Math.abs(r.our_mean_score - r.rasmachine_belief);
			if (d > bestD) {
				bestD = d;
				best = r;
			}
		}
		s = best ?? undefined;
		why = s ? `the largest disagreement with INDRA in this run · ${s.n_evidence} evidence` : '';
	}
	if (!s) return null;
	const evidences = evidenceForStatement(meta, s.stmt_hash)
		.map(toFocusEvidence)
		.sort((a, b) => (a.our_score ?? 1) - (b.our_score ?? 1));
	return {
		run_id: meta.run_id,
		stmt: { stmt_hash: s.stmt_hash, indra_type: s.stmt_type, subject: s.subject, object: s.object },
		our_score: s.our_mean_score,
		indra_score: s.rasmachine_belief,
		n_evidences: s.n_evidence,
		evidences,
		why_this_one: why
	};
}

// ── Statement detail ────────────────────────────────────────────────────────

export interface StatementDetail {
	run_id: string;
	rollup: StatementRollup;
	evidences: FocusEvidence[];
}

export function getStatementDetail(stmtHash: string, runId?: string): StatementDetail | null {
	const meta = resolveRun(runId);
	if (!meta) return null;
	const rollup = getRunData(meta).byHash.get(stmtHash);
	if (!rollup) return null;
	return {
		run_id: meta.run_id,
		rollup,
		evidences: evidenceForStatement(meta, stmtHash).map(toFocusEvidence)
	};
}

// ── Findings (degraded: biggest-disagreement lane only) ─────────────────────

export interface Finding {
	stmt_hash: string;
	subject: string;
	stmt_type: string;
	object: string;
	our_score: number | null;
	indra_belief: number | null;
	delta: number;
}

export function getFindings(runId?: string, limit = 8): { biggest_disagreement: Finding[] } {
	const meta = resolveRun(runId);
	if (!meta) return { biggest_disagreement: [] };
	const rows = getRunData(meta)
		.perStatement.filter((s) => s.our_mean_score != null && s.rasmachine_belief != null)
		.map((s) => ({
			stmt_hash: s.stmt_hash,
			subject: s.subject,
			stmt_type: s.stmt_type,
			object: s.object,
			our_score: s.our_mean_score,
			indra_belief: s.rasmachine_belief,
			delta: (s.our_mean_score as number) - (s.rasmachine_belief as number)
		}))
		.sort((a, b) => Math.abs(b.delta) - Math.abs(a.delta))
		.slice(0, limit);
	return { biggest_disagreement: rows };
}

// ── Two-run comparison (gemma vs medpsy) ────────────────────────────────────

export interface CompareRow {
	stmt_hash: string;
	subject: string;
	stmt_type: string;
	object: string;
	indra_belief: number | null;
	a: { score: number | null; verdict_mix: string };
	b: { score: number | null; verdict_mix: string };
	score_delta: number | null;
}

function verdictMix(s: StatementRollup): string {
	return `${s.n_correct}✓ ${s.n_incorrect}✗`;
}

export interface Comparison {
	run_a: RunSummary;
	run_b: RunSummary;
	n_shared: number;
	rows: CompareRow[];
}

export function compareRuns(runIdA: string, runIdB: string, limit = 200): Comparison | null {
	const a = resolveRun(runIdA);
	const b = resolveRun(runIdB);
	if (!a || !b) return null;
	const da = getRunData(a);
	const db = getRunData(b);
	const rows: CompareRow[] = [];
	for (const sa of da.perStatement) {
		const sb = db.byHash.get(sa.stmt_hash);
		if (!sb) continue;
		const delta =
			sa.our_mean_score != null && sb.our_mean_score != null
				? sa.our_mean_score - sb.our_mean_score
				: null;
		rows.push({
			stmt_hash: sa.stmt_hash,
			subject: sa.subject,
			stmt_type: sa.stmt_type,
			object: sa.object,
			indra_belief: sa.rasmachine_belief,
			a: { score: sa.our_mean_score, verdict_mix: verdictMix(sa) },
			b: { score: sb.our_mean_score, verdict_mix: verdictMix(sb) },
			score_delta: delta
		});
	}
	const n_shared = rows.length;
	rows.sort((x, y) => Math.abs(y.score_delta ?? 0) - Math.abs(x.score_delta ?? 0));
	return { run_a: runSummary(a), run_b: runSummary(b), n_shared, rows: rows.slice(0, limit) };
}

// ── Evidence-level comparison (the progressive A-vs-B anatomy) ───────────────
//
// `compareRuns` above joins at the STATEMENT level and sorts by |score delta| —
// good for "which statements swing most", wrong for "how do the two models
// disagree". A systematic leniency skew (model A says correct where B says
// incorrect, pervasively) is an evidence-level, directional fact that statement
// rollups smear away. These queries join per-evidence on the content-addressed
// key (stmt_hash, evidence_hash) — NOT (stmt_i, evidence_i), which differs
// between exports — and expose the verdict confusion matrix as the entry point.

/** A verdict normalized to the three states that matter for the matrix. */
type V = 'correct' | 'incorrect' | 'none';
function normV(v: string | null): V {
	if (v === 'correct') return 'correct';
	if (v === 'incorrect') return 'incorrect';
	return 'none';
}

/** One evidence present in both runs, with each run's verdict + classification. */
export interface JoinedEvidence {
	stmt_hash: string;
	evidence_hash: string;
	subject: string;
	stmt_type: string;
	object: string;
	source_api: string | null;
	pmid: string | null;
	evidence_text: string | null;
	rasmachine_belief: number | null;
	a_verdict: V;
	b_verdict: V;
	/** Calibrated per-sentence probabilities carried by each export. */
	a_score: number | null;
	b_score: number | null;
	a_confidence: string | null;
	b_confidence: string | null;
	/** Bucket from each run (the artifact-vs-semantic taxonomy). */
	a_bucket: string | null;
	b_bucket: string | null;
	a_group: string | null;
	b_group: string | null;
	grounding_status: string | null;
	/** INDRA human curation gold for this evidence, or null when uncurated.
	 *  gold.verdict is 'correct'|'incorrect' (any-incorrect-wins over curators). */
	gold: GoldVerdict | null;
}

/** Build the joined-evidence set for two runs (cached per (a,b) by run mtime is
 *  unnecessary — the evidence index itself is mtime-cached in the store).
 *  Each row is also annotated with INDRA curation gold (when present) by
 *  coercing the export's string indra_matches_hash + int source_hash into the
 *  curation index's int|int key. */
function joinEvidence(runIdA: string, runIdB: string): { a: RunMeta; b: RunMeta; joined: JoinedEvidence[] } | null {
	const a = resolveRun(runIdA);
	const b = resolveRun(runIdB);
	if (!a || !b) return null;
	const ia = getEvidenceIndex(a);
	const ib = getEvidenceIndex(b);
	// Index BOTH runs by content key, deduped. The exports contain ~1.3k
	// duplicate (stmt_hash, evidence_hash) rows each — the same statement
	// citing the same evidence sentence twice (rasmachine re-ingesting a
	// paper's repeated mentions). Iterating raw rows would double-count and
	// mis-pair across runs; collapse to one row per content key per side
	// (first occurrence wins) before joining.
	const aByKey = new Map<string, EvidenceRow>();
	for (const r of ia.all) {
		const k = `${r.stmt_hash} ${r.evidence_hash}`;
		if (!aByKey.has(k)) aByKey.set(k, r);
	}
	const bByKey = new Map<string, EvidenceRow>();
	for (const r of ib.all) {
		const k = `${r.stmt_hash} ${r.evidence_hash}`;
		if (!bByKey.has(k)) bByKey.set(k, r);
	}
	const cur = getCurationIndex();
	const joined: JoinedEvidence[] = [];
	for (const [k, ra] of aByKey) {
		const rb = bByKey.get(k);
		if (!rb) continue;
		// Gold lookup — the single canonical helper (coerce + key + index get).
		const gold: GoldVerdict | null = goldForRow(cur, ra);
		joined.push({
			stmt_hash: ra.stmt_hash,
			evidence_hash: ra.evidence_hash,
			subject: ra.subject,
			stmt_type: ra.stmt_type,
			object: ra.object,
			source_api: ra.source_api,
			pmid: ra.pmid,
			evidence_text: ra.evidence_text,
			rasmachine_belief: ra.rasmachine_belief,
			a_verdict: normV(ra.verdict),
			b_verdict: normV(rb.verdict),
			a_score: ra.our_score,
			b_score: rb.our_score,
			a_confidence: ra.confidence,
			b_confidence: rb.confidence,
			a_bucket: ra.bucket,
			b_bucket: rb.bucket,
			a_group: ra.bucket_group,
			b_group: rb.bucket_group,
			grounding_status: ra.grounding_status,
			gold
		});
	}
	return { a, b, joined };
}

/** The four cells of the verdict confusion matrix (excludes none/none and any
 *  cell where either side is `none` from the 2×2 — those are tracked separately). */
export interface ConfusionCell {
	a: V;
	b: V;
	n: number;
}

export interface CompareAnatomy {
	run_a: RunSummary;
	run_b: RunSummary;
	/** Evidences present in BOTH runs (content-joined). */
	n_joined: number;
	/** Of those, rows where both runs returned a parseable correct/incorrect. */
	n_both_scored: number;
	/** Agreement over both-scored rows. */
	agree: number;
	agree_pct: number;
	/** The 2×2 over both-scored rows: AcBc, AcBi, AiBc, AiBi. */
	matrix: { a_correct_b_correct: number; a_correct_b_incorrect: number; a_incorrect_b_correct: number; a_incorrect_b_incorrect: number };
	/** Rows where exactly one side failed to produce a verdict. */
	a_only_none: number;
	b_only_none: number;
	both_none: number;
	/** Same matrix restricted to rows where BOTH runs land in a semantic bucket
	 *  (strips artifact disagreements: reader_hallucination, no_evidence, …). */
	semantic: { a_correct_b_correct: number; a_correct_b_incorrect: number; a_incorrect_b_correct: number; a_incorrect_b_incorrect: number; n: number; agree: number; agree_pct: number };
	/** Human-curation gold, when any joined rows are curated. Per-cell tallies
	 *  answer "when they land in this cell, who was right per the human?" —
	 *  raw counts only (coverage is sparse; never a rate). `present=false` when
	 *  no curations are pulled at all. */
	gold: GoldCoverage;
}

/** Per-cell gold tally: of the rows in this confusion cell that ALSO have a
 *  human curation, how many did each model get right. */
export interface GoldCellTally {
	n_covered: number;
	a_right: number;
	b_right: number;
	both_right: number;
	neither_right: number;
}

export interface GoldCoverage {
	present: boolean;
	/** evaluable = joined rows (both scored) that also have gold. */
	n_evaluable: number;
	/** of evaluable, gold-correct vs gold-incorrect. */
	n_gold_correct: number;
	n_gold_incorrect: number;
	/** denominator for the coverage caveat. */
	n_both_scored: number;
	/** per-model accuracy on the evaluable subset. */
	a_accuracy: { right: number; n: number };
	b_accuracy: { right: number; n: number };
	/** per-model precision/recall/F1, BOTH positive classes. The 'supported'
	 *  view is flattered by the ~92%-positive gold; the 'error' view (positive =
	 *  incorrect) is the decision-relevant one — how well the model catches the
	 *  extractions a curator flagged wrong. */
	a_prf: GoldPRF;
	b_prf: GoldPRF;
	/** per confusion cell. */
	cells: {
		acbc: GoldCellTally;
		acbi: GoldCellTally;
		aibc: GoldCellTally;
		aibi: GoldCellTally;
	};
}

/** Precision/recall/F1 for a model vs gold, for each positive class. */
export interface GoldPRF {
	/** positive class = 'correct' (supported-detection). */
	supported: { precision: number; recall: number; f1: number; tp: number; fp: number; fn: number; tn: number };
	/** positive class = 'incorrect' (error-detection — the hard, useful task). */
	error: { precision: number; recall: number; f1: number; tp: number; fp: number; fn: number; tn: number };
}

/** Confusion P/R/F1 over (goldPositive, predPositive) booleans — the TS twin of
 *  indra_belief.metrics.confusion_metrics, kept tiny + local to the gold query. */
function prf(pairs: Array<[boolean, boolean]>): { precision: number; recall: number; f1: number; tp: number; fp: number; fn: number; tn: number } {
	let tp = 0, fp = 0, fn = 0, tn = 0;
	for (const [gold, pred] of pairs) {
		if (pred && gold) tp++;
		else if (pred) fp++;
		else if (gold) fn++;
		else tn++;
	}
	const p = tp + fp ? tp / (tp + fp) : 0;
	const r = tp + fn ? tp / (tp + fn) : 0;
	const f1 = p + r ? (2 * p * r) / (p + r) : 0;
	return { precision: p, recall: r, f1, tp, fp, fn, tn };
}

function emptyTally(): GoldCellTally {
	return { n_covered: 0, a_right: 0, b_right: 0, both_right: 0, neither_right: 0 };
}

export function compareAnatomy(runIdA: string, runIdB: string): CompareAnatomy | null {
	const j = joinEvidence(runIdA, runIdB);
	if (!j) return null;
	const { a, b, joined } = j;
	let acbc = 0, acbi = 0, aibc = 0, aibi = 0;
	let aNone = 0, bNone = 0, bothNone = 0;
	let sAcbc = 0, sAcbi = 0, sAibc = 0, sAibi = 0;
	// gold accumulators
	const gCells = { acbc: emptyTally(), acbi: emptyTally(), aibc: emptyTally(), aibi: emptyTally() };
	let gPresent = false, gEval = 0, gCorrect = 0, gIncorrect = 0, gARight = 0, gBRight = 0;
	// (goldPositive, predPositive) pairs per model, for both positive classes
	const aSup: Array<[boolean, boolean]> = [], bSup: Array<[boolean, boolean]> = [];
	const aErr: Array<[boolean, boolean]> = [], bErr: Array<[boolean, boolean]> = [];
	for (const e of joined) {
		const an = e.a_verdict === 'none';
		const bn = e.b_verdict === 'none';
		if (an && bn) { bothNone++; continue; }
		if (an) { aNone++; continue; }
		if (bn) { bNone++; continue; }
		// both scored — pick the cell key
		let cellKey: 'acbc' | 'acbi' | 'aibc' | 'aibi';
		if (e.a_verdict === 'correct' && e.b_verdict === 'correct') { acbc++; cellKey = 'acbc'; }
		else if (e.a_verdict === 'correct' && e.b_verdict === 'incorrect') { acbi++; cellKey = 'acbi'; }
		else if (e.a_verdict === 'incorrect' && e.b_verdict === 'correct') { aibc++; cellKey = 'aibc'; }
		else { aibi++; cellKey = 'aibi'; }
		// semantic-only: both buckets in the 'semantic' group
		if (e.a_group === 'semantic' && e.b_group === 'semantic') {
			if (e.a_verdict === 'correct' && e.b_verdict === 'correct') sAcbc++;
			else if (e.a_verdict === 'correct' && e.b_verdict === 'incorrect') sAcbi++;
			else if (e.a_verdict === 'incorrect' && e.b_verdict === 'correct') sAibc++;
			else sAibi++;
		}
		// gold tally (only when this row is curated)
		if (e.gold) {
			gPresent = true;
			gEval++;
			const g = e.gold.verdict;
			if (g === 'correct') gCorrect++; else gIncorrect++;
			const aRight = e.a_verdict === g;
			const bRight = e.b_verdict === g;
			if (aRight) gARight++;
			if (bRight) gBRight++;
			// supported-detection: positive = 'correct'
			aSup.push([g === 'correct', e.a_verdict === 'correct']);
			bSup.push([g === 'correct', e.b_verdict === 'correct']);
			// error-detection: positive = 'incorrect'
			aErr.push([g === 'incorrect', e.a_verdict === 'incorrect']);
			bErr.push([g === 'incorrect', e.b_verdict === 'incorrect']);
			const t = gCells[cellKey];
			t.n_covered++;
			if (aRight) t.a_right++;
			if (bRight) t.b_right++;
			if (aRight && bRight) t.both_right++;
			if (!aRight && !bRight) t.neither_right++;
		}
	}
	const bothScored = acbc + acbi + aibc + aibi;
	const agree = acbc + aibi;
	const sN = sAcbc + sAcbi + sAibc + sAibi;
	const sAgree = sAcbc + sAibi;
	const gold: GoldCoverage = {
		present: gPresent,
		n_evaluable: gEval,
		n_gold_correct: gCorrect,
		n_gold_incorrect: gIncorrect,
		n_both_scored: bothScored,
		a_accuracy: { right: gARight, n: gEval },
		b_accuracy: { right: gBRight, n: gEval },
		a_prf: { supported: prf(aSup), error: prf(aErr) },
		b_prf: { supported: prf(bSup), error: prf(bErr) },
		cells: gCells
	};
	return {
		run_a: runSummary(a),
		run_b: runSummary(b),
		n_joined: joined.length,
		n_both_scored: bothScored,
		agree,
		agree_pct: bothScored ? agree / bothScored : 0,
		matrix: {
			a_correct_b_correct: acbc,
			a_correct_b_incorrect: acbi,
			a_incorrect_b_correct: aibc,
			a_incorrect_b_incorrect: aibi
		},
		a_only_none: aNone,
		b_only_none: bNone,
		both_none: bothNone,
		semantic: {
			a_correct_b_correct: sAcbc,
			a_correct_b_incorrect: sAcbi,
			a_incorrect_b_correct: sAibc,
			a_incorrect_b_incorrect: sAibi,
			n: sN,
			agree: sAgree,
			agree_pct: sN ? sAgree / sN : 0
		},
		gold
	};
}

// ── Gold performance: the per-error partition + CI-bearing P/R/F1 ────────────
//
// The flat P/R/F1 table hid the two facts a skeptic needs: (1) the sample is
// TINY on the class that matters (gold-incorrect), and (2) the model gap is
// inside the noise. This query returns the structure to SHOW that: each error
// as a discrete unit with per-model caught flags + identity, and recall with
// Wilson 95% CIs so the overlap is perceptible, not asserted.

export type GoldGranularity = 'evidence' | 'statement';

/** One gold-labelled item (evidence or rolled-up statement) with each model's
 *  verdict, for the discrete error/positive units the viz renders. */
export interface GoldItem {
	stmt_hash: string;
	evidence_hash: string | null; // null for statement-granularity rows
	subject: string;
	stmt_type: string;
	object: string;
	gold: 'correct' | 'incorrect';
	a_verdict: 'correct' | 'incorrect';
	b_verdict: 'correct' | 'incorrect';
	tags: string[]; // curation tags (the "why" on errors)
	n_ev: number; // evidences backing a statement row (1 for evidence granularity)
}

interface WilsonCI { p: number; lo: number; hi: number; k: number; n: number }

function wilsonCI(k: number, n: number, z = 1.96): WilsonCI {
	if (n === 0) return { p: 0, lo: 0, hi: 0, k, n };
	const p = k / n;
	const denom = 1 + (z * z) / n;
	const centre = (p + (z * z) / (2 * n)) / denom;
	const margin = (z / denom) * Math.sqrt((p * (1 - p)) / n + (z * z) / (4 * n * n));
	return { p, lo: Math.max(0, centre - margin), hi: Math.min(1, centre + margin), k, n };
}

/** A model's 2×2 vs gold. Rows = gold (correct/incorrect), cols = model verdict.
 *  gcpc = gold-correct & pred-correct (TP for supported), etc. */
export interface ConfMatrix {
	gc_pc: number; // gold✓ pred✓
	gc_pi: number; // gold✓ pred✗ (model over-rejected a supported claim)
	gi_pc: number; // gold✗ pred✓ (model over-accepted an error — the dangerous one)
	gi_pi: number; // gold✗ pred✗ (caught the error)
	n: number;
}

export interface GoldPerformance {
	run_a: RunSummary;
	run_b: RunSummary;
	granularity: GoldGranularity;
	present: boolean;
	n: number; // total gold items
	n_supported: number; // gold-correct
	n_error: number; // gold-incorrect (the decisive, small class)
	a_conf: ConfMatrix;
	b_conf: ConfMatrix;
	/** the gold-incorrect items, ordered: both-caught, A-only, B-only, missed-by-both. */
	errors: GoldItem[];
	error_partition: { both: number; a_only: number; b_only: number; neither: number };
	/** the gold-correct items (for the supported strip; usually large). */
	supported_total: number;
	supported_a_right: number;
	supported_b_right: number;
	/** recall on each class, per model, WITH Wilson CIs (the honesty layer). */
	a_error_recall: WilsonCI;
	b_error_recall: WilsonCI;
	a_supported_recall: WilsonCI;
	b_supported_recall: WilsonCI;
}

function _v2(v: V): 'correct' | 'incorrect' | null {
	return v === 'correct' || v === 'incorrect' ? v : null;
}

export function goldPerformance(
	runIdA: string,
	runIdB: string,
	granularity: GoldGranularity = 'evidence'
): GoldPerformance | null {
	const j = joinEvidence(runIdA, runIdB);
	if (!j) return null;
	const { a, b, joined } = j;

	// units: each evidence, OR each statement (majority-vote rollup of its evidences)
	type Unit = { stmt_hash: string; evidence_hash: string | null; subject: string; stmt_type: string;
		object: string; gold: 'correct' | 'incorrect'; av: 'correct' | 'incorrect'; bv: 'correct' | 'incorrect';
		tags: string[]; n_ev: number };
	const units: Unit[] = [];

	if (granularity === 'evidence') {
		for (const e of joined) {
			if (!e.gold) continue;
			const av = _v2(e.a_verdict), bv = _v2(e.b_verdict);
			if (!av || !bv) continue; // both must be scored
			units.push({ stmt_hash: e.stmt_hash, evidence_hash: e.evidence_hash, subject: e.subject,
				stmt_type: e.stmt_type, object: e.object, gold: e.gold.verdict, av, bv, tags: e.gold.tags, n_ev: 1 });
		}
	} else {
		// roll up evidences per statement; majority-vote each of gold/A/B
		const byStmt = new Map<string, JoinedEvidence[]>();
		for (const e of joined) {
			if (!e.gold) continue;
			if (!_v2(e.a_verdict) || !_v2(e.b_verdict)) continue;
			let arr = byStmt.get(e.stmt_hash);
			if (!arr) byStmt.set(e.stmt_hash, (arr = []));
			arr.push(e);
		}
		const maj = (es: JoinedEvidence[], pick: (e: JoinedEvidence) => string): 'correct' | 'incorrect' => {
			const c = es.filter((e) => pick(e) === 'correct').length;
			return c > es.length / 2 ? 'correct' : 'incorrect';
		};
		for (const [sh, es] of byStmt) {
			const e0 = es[0];
			units.push({ stmt_hash: sh, evidence_hash: null, subject: e0.subject, stmt_type: e0.stmt_type,
				object: e0.object,
				gold: maj(es, (e) => e.gold!.verdict),
				av: maj(es, (e) => e.a_verdict), bv: maj(es, (e) => e.b_verdict),
				tags: [...new Set(es.flatMap((e) => e.gold!.tags))], n_ev: es.length });
		}
	}

	const toItem = (u: Unit): GoldItem => ({ stmt_hash: u.stmt_hash, evidence_hash: u.evidence_hash,
		subject: u.subject, stmt_type: u.stmt_type, object: u.object, gold: u.gold,
		a_verdict: u.av, b_verdict: u.bv, tags: u.tags, n_ev: u.n_ev });

	const errs = units.filter((u) => u.gold === 'incorrect');
	const sups = units.filter((u) => u.gold === 'correct');
	// catching an error = saying 'incorrect'
	const aErr = errs.filter((u) => u.av === 'incorrect').length;
	const bErr = errs.filter((u) => u.bv === 'incorrect').length;
	let both = 0, aOnly = 0, bOnly = 0, neither = 0;
	for (const u of errs) {
		const ac = u.av === 'incorrect', bc = u.bv === 'incorrect';
		if (ac && bc) both++;
		else if (ac) aOnly++;
		else if (bc) bOnly++;
		else neither++;
	}
	// order errors: both-caught, A-only, B-only, missed-by-both (reads left→worst-right)
	const rank = (u: Unit) => {
		const ac = u.av === 'incorrect', bc = u.bv === 'incorrect';
		if (ac && bc) return 0;
		if (ac) return 1;
		if (bc) return 2;
		return 3;
	};
	const orderedErrors = [...errs].sort((x, y) => rank(x) - rank(y)).map(toItem);

	const aSupRight = sups.filter((u) => u.av === 'correct').length;
	const bSupRight = sups.filter((u) => u.bv === 'correct').length;

	const conf = (pick: (u: Unit) => 'correct' | 'incorrect'): ConfMatrix => {
		let gc_pc = 0, gc_pi = 0, gi_pc = 0, gi_pi = 0;
		for (const u of units) {
			const pv = pick(u);
			if (u.gold === 'correct' && pv === 'correct') gc_pc++;
			else if (u.gold === 'correct') gc_pi++;
			else if (pv === 'correct') gi_pc++;
			else gi_pi++;
		}
		return { gc_pc, gc_pi, gi_pc, gi_pi, n: units.length };
	};

	return {
		run_a: runSummary(a),
		run_b: runSummary(b),
		granularity,
		present: units.length > 0,
		n: units.length,
		n_supported: sups.length,
		n_error: errs.length,
		a_conf: conf((u) => u.av),
		b_conf: conf((u) => u.bv),
		errors: orderedErrors,
		error_partition: { both, a_only: aOnly, b_only: bOnly, neither },
		supported_total: sups.length,
		supported_a_right: aSupRight,
		supported_b_right: bSupRight,
		a_error_recall: wilsonCI(aErr, errs.length),
		b_error_recall: wilsonCI(bErr, errs.length),
		a_supported_recall: wilsonCI(aSupRight, sups.length),
		b_supported_recall: wilsonCI(bSupRight, sups.length)
	};
}

// ── L1: stratify one confusion cell across an axis ──────────────────────────

export type CompareCell = 'acbc' | 'acbi' | 'aibc' | 'aibi';
export type StratAxis = 'source_api' | 'stmt_type' | 'grounding_status' | 'bucket_a' | 'bucket_b';
/** Gold-relationship filter for the cohort. 'match' = both models agree with
 *  gold; 'fp' = a model says correct but gold is incorrect (over-trust);
 *  'fn' = a model says incorrect but gold is correct; 'disagree' = at least one
 *  model contradicts gold; 'any' = any curated row. */
export type GoldFilter = 'any' | 'match' | 'fp' | 'fn' | 'disagree';

function goldFilterMatch(e: JoinedEvidence, gf: GoldFilter): boolean {
	if (!e.gold) return false; // every gold filter requires a curation
	if (gf === 'any') return true;
	const g = e.gold.verdict;
	const aR = e.a_verdict === g;
	const bR = e.b_verdict === g;
	switch (gf) {
		case 'match': return aR && bR;
		case 'disagree': return !aR || !bR;
		// fp: a model said 'correct' while gold is 'incorrect' (over-acceptance)
		case 'fp': return g === 'incorrect' && (e.a_verdict === 'correct' || e.b_verdict === 'correct');
		// fn: a model said 'incorrect' while gold is 'correct' (over-rejection)
		case 'fn': return g === 'correct' && (e.a_verdict === 'incorrect' || e.b_verdict === 'incorrect');
	}
}

function cellMatch(e: JoinedEvidence, cell: CompareCell): boolean {
	switch (cell) {
		case 'acbc': return e.a_verdict === 'correct' && e.b_verdict === 'correct';
		case 'acbi': return e.a_verdict === 'correct' && e.b_verdict === 'incorrect';
		case 'aibc': return e.a_verdict === 'incorrect' && e.b_verdict === 'correct';
		case 'aibi': return e.a_verdict === 'incorrect' && e.b_verdict === 'incorrect';
	}
}

function axisValue(e: JoinedEvidence, axis: StratAxis): string {
	switch (axis) {
		case 'source_api': return e.source_api || '—';
		case 'stmt_type': return e.stmt_type || '—';
		case 'grounding_status': return e.grounding_status || '—';
		case 'bucket_a': return e.a_bucket || '—';
		case 'bucket_b': return e.b_bucket || '—';
	}
}

export interface StratRow {
	value: string;
	n: number;
	pct: number;
	/** of the n rows in this stratum, how many are curated + how many of those
	 *  each model got right (raw counts; coverage is sparse). */
	gold_n: number;
	a_right: number;
	b_right: number;
}

export interface CellStratification {
	run_a: RunSummary;
	run_b: RunSummary;
	cell: CompareCell;
	semantic_only: boolean;
	total: number;
	axis: StratAxis;
	rows: StratRow[];
}

export function stratifyCell(
	runIdA: string,
	runIdB: string,
	cell: CompareCell,
	axis: StratAxis,
	semanticOnly: boolean
): CellStratification | null {
	const j = joinEvidence(runIdA, runIdB);
	if (!j) return null;
	const { a, b, joined } = j;
	const counts = new Map<string, { n: number; gold_n: number; a_right: number; b_right: number }>();
	let total = 0;
	for (const e of joined) {
		if (e.a_verdict === 'none' || e.b_verdict === 'none') continue;
		if (semanticOnly && !(e.a_group === 'semantic' && e.b_group === 'semantic')) continue;
		if (!cellMatch(e, cell)) continue;
		total++;
		const v = axisValue(e, axis);
		let agg = counts.get(v);
		if (!agg) counts.set(v, (agg = { n: 0, gold_n: 0, a_right: 0, b_right: 0 }));
		agg.n++;
		if (e.gold) {
			agg.gold_n++;
			if (e.a_verdict === e.gold.verdict) agg.a_right++;
			if (e.b_verdict === e.gold.verdict) agg.b_right++;
		}
	}
	const rows: StratRow[] = [...counts.entries()]
		.map(([value, c]) => ({ value, n: c.n, pct: total ? c.n / total : 0, gold_n: c.gold_n, a_right: c.a_right, b_right: c.b_right }))
		.sort((x, y) => y.n - x.n);
	return { run_a: runSummary(a), run_b: runSummary(b), cell, semantic_only: semanticOnly, total, axis, rows };
}

// ── L2: cohort table for a cell (+ optional axis filter), salience-sorted ────

export interface CohortRow {
	stmt_hash: string;
	evidence_hash: string;
	subject: string;
	stmt_type: string;
	object: string;
	source_api: string | null;
	pmid: string | null;
	evidence_text: string | null;
	rasmachine_belief: number | null;
	a_verdict: V;
	b_verdict: V;
	a_score: number | null;
	b_score: number | null;
	a_confidence: string | null;
	b_confidence: string | null;
	a_bucket: string | null;
	b_bucket: string | null;
	/** human-curation gold for this row (null when uncurated). */
	gold_verdict: 'correct' | 'incorrect' | null;
	gold_n: number;
	gold_tags: string[];
	/** Salience: both runs maximally confident in OPPOSITE verdicts ranks first. */
	salience: number;
}

const CONF_RANK: Record<string, number> = { high: 3, medium: 2, low: 1 };

function salience(e: JoinedEvidence): number {
	// Sharpest contradiction = both confident, opposite verdicts. For agreement
	// cells salience still orders by joint confidence (most-certain agreement).
	const ca = CONF_RANK[e.a_confidence ?? ''] ?? 0;
	const cb = CONF_RANK[e.b_confidence ?? ''] ?? 0;
	return ca + cb;
}

export interface Cohort {
	run_a: RunSummary;
	run_b: RunSummary;
	cell: CompareCell;
	axis: StratAxis | null;
	axis_value: string | null;
	semantic_only: boolean;
	gold_filter: GoldFilter | null;
	total: number;
	/** how many of `total` are curated — for the gold caveat. */
	gold_covered: number;
	rows: CohortRow[];
}

export function cohortForCell(
	runIdA: string,
	runIdB: string,
	cell: CompareCell,
	opts: { axis?: StratAxis | null; axisValue?: string | null; semanticOnly?: boolean; goldFilter?: GoldFilter | null; limit?: number } = {}
): Cohort | null {
	const j = joinEvidence(runIdA, runIdB);
	if (!j) return null;
	const { a, b, joined } = j;
	const { axis = null, axisValue: av = null, semanticOnly = false, goldFilter = null, limit = 200 } = opts;
	const matched: JoinedEvidence[] = [];
	let goldCovered = 0;
	for (const e of joined) {
		if (e.a_verdict === 'none' || e.b_verdict === 'none') continue;
		if (semanticOnly && !(e.a_group === 'semantic' && e.b_group === 'semantic')) continue;
		if (!cellMatch(e, cell)) continue;
		if (axis && av != null && axisValue(e, axis) !== av) continue;
		if (goldFilter && !goldFilterMatch(e, goldFilter)) continue;
		matched.push(e);
		if (e.gold) goldCovered++;
	}
	// When a gold filter is active, sort gold-relevant rows first; otherwise the
	// existing joint-confidence salience. Within gold, curated rows lead.
	matched.sort((x, y) => {
		if (goldFilter) {
			const gx = x.gold ? 1 : 0, gy = y.gold ? 1 : 0;
			if (gx !== gy) return gy - gx;
		}
		return salience(y) - salience(x);
	});
	const rows: CohortRow[] = matched.slice(0, limit).map((e) => ({
		stmt_hash: e.stmt_hash,
		evidence_hash: e.evidence_hash,
		subject: e.subject,
		stmt_type: e.stmt_type,
		object: e.object,
		source_api: e.source_api,
		pmid: e.pmid,
		evidence_text: e.evidence_text,
		rasmachine_belief: e.rasmachine_belief,
		a_verdict: e.a_verdict,
		b_verdict: e.b_verdict,
		a_score: e.a_score,
		b_score: e.b_score,
		a_confidence: e.a_confidence,
		b_confidence: e.b_confidence,
		a_bucket: e.a_bucket,
		b_bucket: e.b_bucket,
		gold_verdict: e.gold?.verdict ?? null,
		gold_n: e.gold?.n ?? 0,
		gold_tags: e.gold?.tags ?? [],
		salience: salience(e)
	}));
	return {
		run_a: runSummary(a),
		run_b: runSummary(b),
		cell,
		axis,
		axis_value: av,
		semantic_only: semanticOnly,
		gold_filter: goldFilter,
		total: matched.length,
		gold_covered: goldCovered,
		rows
	};
}

// ── L3: one evidence, both runs' full reasoning side by side ─────────────────

export interface SideBySideEvidence {
	run_a: RunSummary;
	run_b: RunSummary;
	stmt_hash: string;
	evidence_hash: string;
	subject: string;
	stmt_type: string;
	object: string;
	source_api: string | null;
	pmid: string | null;
	evidence_text: string | null;
	rasmachine_belief: number | null;
	a: { verdict: V; confidence: string | null; bucket: string | null; reasoning: string | null; tier: string | null; score: number | null; reasoning_trace: ReasoningTrace | null };
	b: { verdict: V; confidence: string | null; bucket: string | null; reasoning: string | null; tier: string | null; score: number | null; reasoning_trace: ReasoningTrace | null };
	/** Human-curation gold trace — the third judge. null when this evidence has
	 *  no INDRA curation. Curators don't write reasoning; they assign tags + an
	 *  optional note, so this carries the tag taxonomy + curators + notes. */
	gold: { verdict: 'correct' | 'incorrect'; n: number; tags: string[]; curators: string[]; notes: string[] } | null;
}

export function evidenceSideBySide(
	runIdA: string,
	runIdB: string,
	stmtHash: string,
	evidenceHash: string
): SideBySideEvidence | null {
	const a = resolveRun(runIdA);
	const b = resolveRun(runIdB);
	if (!a || !b) return null;
	const ra = getEvidenceIndex(a).all.find((r) => r.stmt_hash === stmtHash && r.evidence_hash === evidenceHash);
	const rb = getEvidenceIndex(b).all.find((r) => r.stmt_hash === stmtHash && r.evidence_hash === evidenceHash);
	if (!ra || !rb) return null;
	// Gold lookup — the single canonical helper.
	const gv = goldForRow(getCurationIndex(), ra);
	const gold: SideBySideEvidence['gold'] = gv
		? { verdict: gv.verdict, n: gv.n, tags: gv.tags, curators: gv.curators, notes: gv.notes }
		: null;
	return {
		run_a: runSummary(a),
		run_b: runSummary(b),
		stmt_hash: stmtHash,
		evidence_hash: evidenceHash,
		subject: ra.subject,
		stmt_type: ra.stmt_type,
		object: ra.object,
		source_api: ra.source_api,
		pmid: ra.pmid,
		evidence_text: ra.evidence_text,
		rasmachine_belief: ra.rasmachine_belief,
		a: { verdict: normV(ra.verdict), confidence: ra.confidence, bucket: ra.bucket, reasoning: ra.reasoning, tier: ra.tier, score: ra.our_score, reasoning_trace: ra.reasoning_trace ?? null },
		b: { verdict: normV(rb.verdict), confidence: rb.confidence, bucket: rb.bucket, reasoning: rb.reasoning, tier: rb.tier, score: rb.our_score, reasoning_trace: rb.reasoning_trace ?? null },
		gold
	};
}

// ── Frontier: cost × error-detection-F1 across N runs of one substrate ───────
//
// The N-run entry surface. For each run scored on the SAME substrate (corpus —
// the only runs that share a gold set and a content-hash join), we read the
// baked cost block and derive error-detection F1 on the run's gold (the lead
// metric — never accuracy). Cost and F1 together define a Pareto frontier: a run
// is dominated iff another run is no dearer AND no worse. We do NOT pretend the
// frontier is exact — every F1 carries a bootstrap 95% CI so the n=60-class
// substrates read as "differences within noise," not a false ranking.

export interface FrontierRun {
	run_id: string;
	model: string;
	generated_date: string | null;
	cost: RunMeta['cost'];
	/** USD per 1k priced evidences — the size-normalized x-coordinate. null when
	 *  cost is unavailable (NOT plotted on the cost axis, never faked as 0). */
	usd_per_1k: number | null;
	/** known-zero local model (genuine $0, ZERO_COST set) — pinned in the gutter. */
	is_free: boolean;
	cost_known: boolean;
	/** cost is a Bedrock-grounded estimate (self-hosted model), not observed spend. */
	cost_estimated: boolean;
	n_gold: number;
	/** error-detection (positive class = 'incorrect') P/R/F1 on this run's gold. */
	err_f1: number;
	err_precision: number;
	err_recall: number;
	/** bootstrap 95% CI on err_f1 — the honesty band the y-axis whisker draws. */
	err_f1_lo: number;
	err_f1_hi: number;
	/** plain accuracy vs gold (right / n) — secondary, shown but never led with. */
	accuracy: number;
	// ── model-size axis (ground truth, baked; closed models → unknown) ──
	/** total parameters (billions) — the plotted size; null when undisclosed. */
	params_total_b: number | null;
	/** active params per forward pass (MoE); null for dense or unknown. */
	params_active_b: number | null;
	size_known: boolean;
	/** size is an estimate/inference, not a published spec — renders hollow on the
	 *  size axis, the mirror of cost_estimated on the cost axis. */
	size_estimated: boolean;
	/** open-weight (size is ground truth) vs closed (size undisclosed) vs unknown. */
	is_open: boolean | null;
	// ── Pareto status, computed per axis (cost↓×F1↑ and size↓×F1↑) ──
	on_frontier_cost: boolean;
	dominated_by_cost: string | null;
	on_frontier_size: boolean;
	dominated_by_size: string | null;
	/** Repeat runs of the SAME model on this substrate are one model measured N
	 *  times, not N models — they fold into ONE point. n_reps>1 ⇒ err_f1 etc. are
	 *  the across-rep mean and the CI band spans the reps' spread. */
	n_reps: number;
	/** run_ids of every rep (sorted); `run_id` above is the representative rep
	 *  (closest to the model mean) that carries the /runs and /compare drills. */
	rep_run_ids: string[];
}

export interface FrontierSubstrate {
	key: string; // the corpus basename — the join boundary
	label: string;
	n_runs: number;
	n_cost_known: number;
	gold_n: number; // gold coverage of a representative run (same corpus ⇒ same gold)
}

export interface Frontier {
	substrates: FrontierSubstrate[];
	selected: string | null;
	runs: FrontierRun[]; // sorted err_f1 desc
	n_gold: number;
	cost_span: { min: number | null; max: number | null }; // over priced runs only
	size_span: { min: number | null; max: number | null }; // over size-known runs
	f1_span: { min: number; max: number };
	/** small-sample honesty caveat, or null when n is large enough to rank. */
	note: string | null;
}

/** error-detection (gold, pred) pairs for one run's evidence — positive class is
 *  'incorrect'. Gold is read from the baked per-row field, falling back to the
 *  global curation index for legacy exports. */
function errorPairs(meta: RunMeta): Array<[boolean, boolean]> {
	const cur = getCurationIndex();
	const pairs: Array<[boolean, boolean]> = [];
	for (const r of getEvidenceIndex(meta).all) {
		const gv = r.gold ?? goldForRow(cur, r);
		if (!gv) continue;
		pairs.push([gv.verdict === 'incorrect', r.verdict === 'incorrect']);
	}
	return pairs;
}

/** Deterministic bootstrap 95% CI on F1 over (gold, pred) pairs. Seeded by the
 *  run so the band never flickers between loads (no Math.random). */
function bootstrapF1CI(pairs: Array<[boolean, boolean]>, seed: number, B = 1000): { lo: number; hi: number } {
	const n = pairs.length;
	if (n === 0) return { lo: 0, hi: 0 };
	let s = (seed ^ 0x9e3779b9) >>> 0;
	const rnd = () => ((s = (Math.imul(s, 1664525) + 1013904223) >>> 0) / 4294967296);
	const f1s: number[] = [];
	for (let bi = 0; bi < B; bi++) {
		let tp = 0, fp = 0, fn = 0;
		for (let i = 0; i < n; i++) {
			const [g, p] = pairs[(rnd() * n) | 0];
			if (p && g) tp++;
			else if (p) fp++;
			else if (g) fn++;
		}
		const pr = tp + fp ? tp / (tp + fp) : 0;
		const rc = tp + fn ? tp / (tp + fn) : 0;
		f1s.push(pr + rc ? (2 * pr * rc) / (pr + rc) : 0);
	}
	f1s.sort((x, y) => x - y);
	return { lo: f1s[Math.floor(0.025 * B)], hi: f1s[Math.min(B - 1, Math.ceil(0.975 * B) - 1)] };
}

function seedFromRunId(runId: string): number {
	let h = 0;
	for (let i = 0; i < runId.length; i++) h = (Math.imul(h, 31) + runId.charCodeAt(i)) >>> 0;
	return h;
}

function substrateLabel(key: string): string {
	return key.replace(/_statements\.json$/, '').replace(/\.json$/, '');
}

/** Build the cost × error-F1 frontier for one substrate (the corpus the runs
 *  share). With no argument, defaults to the substrate richest in cost-known,
 *  gold-bearing runs — the one where the tradeoff is actually visible. */
export function frontier(substrateKey?: string | null): Frontier {
	const all = listRuns();

	// Group runs by their join boundary; a run with no recorded substrate is
	// keyed by its run_id so it never silently merges with another corpus.
	const bySub = new Map<string, RunMeta[]>();
	for (const m of all) {
		const k = m.substrate ?? `run:${m.run_id}`;
		(bySub.get(k) ?? bySub.set(k, []).get(k)!).push(m);
	}

	const substrates: FrontierSubstrate[] = [];
	for (const [key, runs] of bySub) {
		substrates.push({
			key,
			label: substrateLabel(key),
			n_runs: runs.length,
			n_cost_known: runs.filter((r) => r.cost && r.cost.status !== 'unavailable').length,
			gold_n: Math.max(0, ...runs.map((r) => r.gold_coverage?.covered ?? 0))
		});
	}
	// Rank substrates so the default is the one where cost AND gold both exist for
	// the most runs (the frontier is only meaningful there).
	substrates.sort((a, b) => b.n_cost_known - a.n_cost_known || b.n_runs - a.n_runs);

	const selected =
		(substrateKey && bySub.has(substrateKey) ? substrateKey : null) ?? substrates[0]?.key ?? null;
	const runsMeta = selected ? bySub.get(selected) ?? [] : [];

	const perRun: FrontierRun[] = [];
	for (const meta of runsMeta) {
		const pairs = errorPairs(meta);
		const nGold = pairs.length;
		const stats = prf(pairs);
		// accuracy = fraction of gold rows the run's verdict matched
		let right = 0;
		for (const [g, p] of pairs) if (g === p) right++;
		const ci = bootstrapF1CI(pairs, seedFromRunId(meta.run_id));
		const c = meta.cost;
		const known = !!c && c.status !== 'unavailable';
		const perK = known ? c!.usd_per_1k_evidence : null;
		const mm = meta.model_meta;
		perRun.push({
			run_id: meta.run_id,
			model: meta.model,
			generated_date: meta.generated_date,
			cost: c ?? null,
			usd_per_1k: perK,
			// genuine zero ONLY (the ZERO_COST local set). A known run with a null
			// per-1k (e.g. every row no_llm → nothing costed) is NOT free — it has no
			// cost datum, falls through to fmtCostFull → em-dash, and is not plotted.
			is_free: known && perK === 0,
			cost_known: known,
			cost_estimated: !!c && c.status === 'estimated',
			n_gold: nGold,
			err_f1: stats.f1,
			err_precision: stats.precision,
			err_recall: stats.recall,
			err_f1_lo: ci.lo,
			err_f1_hi: ci.hi,
			accuracy: nGold ? right / nGold : 0,
			params_total_b: mm && mm.status === 'known' ? mm.total_b : null,
			params_active_b: mm && mm.status === 'known' ? mm.active_b : null,
			size_known: !!mm && mm.status === 'known' && mm.total_b != null,
			size_estimated: !!mm && mm.status === 'known' && !!mm.estimated,
			is_open: mm ? mm.is_open : null,
			on_frontier_cost: false,
			dominated_by_cost: null,
			on_frontier_size: false,
			dominated_by_size: null,
			n_reps: 1,
			rep_run_ids: [meta.run_id]
		});
	}

	// Fold repeat runs of the SAME model (same substrate) into one per-model
	// point. Reps are repeated MEASUREMENTS of one model, not separate models;
	// plotting each as its own point put a model on the frontier several times
	// (duplicate labels, phantom staircase steps) and let a lucky rep define the
	// front. Central tendency = the across-rep mean; the rep spread WIDENS the CI
	// band (honest variance), never narrows it. Single-run models pass through
	// unchanged. Group by the EXACT model string so cross-host deployments
	// (bedrock- vs remote-) stay distinct points.
	const mean = (xs: number[]) => xs.reduce((a, b) => a + b, 0) / xs.length;
	const byModel = new Map<string, FrontierRun[]>();
	for (const r of perRun) (byModel.get(r.model) ?? byModel.set(r.model, []).get(r.model)!).push(r);
	const runs: FrontierRun[] = [];
	for (const reps of byModel.values()) {
		if (reps.length === 1) {
			runs.push(reps[0]);
			continue;
		}
		const f1 = mean(reps.map((r) => r.err_f1));
		// representative = the rep whose F1 is closest to the model mean (the
		// "typical" run); it carries run_id for the /runs + /compare drill-throughs.
		const rep = reps
			.slice()
			.sort((a, b) => Math.abs(a.err_f1 - f1) - Math.abs(b.err_f1 - f1) || a.run_id.localeCompare(b.run_id))[0];
		const priced = reps.filter((r) => r.cost_known && r.usd_per_1k != null);
		const perK = priced.length ? mean(priced.map((r) => r.usd_per_1k!)) : null;
		const totals = priced.map((r) => r.cost?.total_usd).filter((v): v is number => v != null);
		// Cost STATUS tracks the PRICED subset, not the F1-representative rep: a
		// folded point must never carry a mean cost (usd_per_1k != null) while
		// reading cost_known=false, which would silently drop it from the cost
		// frontier (applyDominance keys off cost_known && usd_per_1k != null). Base
		// the cost block on a priced rep when one exists; mark estimated only if NO
		// rep had observed spend.
		const costRep = priced[0] ?? rep;
		// band spans BOTH the widest within-run CI and the across-rep spread of the
		// point estimates, so it never reads tighter than the reps actually disagree.
		const lo = Math.min(...reps.map((r) => Math.min(r.err_f1_lo, r.err_f1)));
		const hi = Math.max(...reps.map((r) => Math.max(r.err_f1_hi, r.err_f1)));
		const dates = reps.map((r) => r.generated_date ?? '').filter(Boolean).sort();
		runs.push({
			...rep,
			err_f1: f1,
			err_precision: mean(reps.map((r) => r.err_precision)),
			err_recall: mean(reps.map((r) => r.err_recall)),
			accuracy: mean(reps.map((r) => r.accuracy)),
			err_f1_lo: lo,
			err_f1_hi: hi,
			cost_known: priced.length > 0,
			cost_estimated: priced.length > 0 && priced.every((r) => r.cost_estimated),
			is_free: priced.length > 0 && perK === 0,
			usd_per_1k: perK,
			cost: costRep.cost
				? { ...costRep.cost, total_usd: totals.length ? mean(totals) : costRep.cost.total_usd, usd_per_1k_evidence: perK }
				: costRep.cost,
			generated_date: dates.length ? dates[dates.length - 1] : rep.generated_date,
			n_gold: Math.max(...reps.map((r) => r.n_gold)),
			n_reps: reps.length,
			rep_run_ids: reps.map((r) => r.run_id).slice().sort()
		});
	}

	// Pareto dominance, computed independently for each axis (cost↓×F1↑ and
	// size↓×F1↑): R is dominated by S iff S is no worse on x AND no worse on F1,
	// strict on at least one. Only runs with an x-value on that axis participate;
	// the cheapest/smallest dominator wins (the most damning "why pay/scale more").
	const applyDominance = (
		xOf: (r: FrontierRun) => number | null,
		setOn: (r: FrontierRun, v: boolean) => void,
		setDom: (r: FrontierRun, v: string | null) => void
	) => {
		const pts = runs.filter((r) => xOf(r) != null);
		for (const r of pts) {
			const rx = xOf(r)!;
			let dom: FrontierRun | null = null;
			for (const s of pts) {
				if (s === r) continue;
				const sx = xOf(s)!;
				if (sx <= rx && s.err_f1 >= r.err_f1 && (sx < rx || s.err_f1 > r.err_f1)) {
					if (!dom || sx < xOf(dom)!) dom = s;
				}
			}
			setDom(r, dom ? dom.model : null);
			setOn(r, dom === null);
		}
	};
	applyDominance(
		(r) => (r.cost_known && r.usd_per_1k != null ? r.usd_per_1k : null),
		(r, v) => (r.on_frontier_cost = v),
		(r, v) => (r.dominated_by_cost = v)
	);
	applyDominance(
		(r) => (r.size_known ? r.params_total_b : null),
		(r, v) => (r.on_frontier_size = v),
		(r, v) => (r.dominated_by_size = v)
	);

	runs.sort((a, b) => b.err_f1 - a.err_f1 || (a.usd_per_1k ?? Infinity) - (b.usd_per_1k ?? Infinity));

	const pricedVals = runs
		.filter((r) => r.cost_known && r.usd_per_1k != null && r.usd_per_1k > 0)
		.map((r) => r.usd_per_1k!);
	const sizeVals = runs.filter((r) => r.size_known).map((r) => r.params_total_b!);
	const nGold = runs.length ? Math.max(...runs.map((r) => r.n_gold)) : 0;
	const f1s = runs.map((r) => r.err_f1);
	const note =
		nGold > 0 && nGold < 300
			? `gold n=${nGold} per run — error-F1 bands overlap; read rank as indicative, not decisive.`
			: null;

	// backfill the representative gold_n now that we've computed it
	const selSub = substrates.find((s) => s.key === selected);
	if (selSub) selSub.gold_n = nGold;

	return {
		substrates,
		selected,
		runs,
		n_gold: nGold,
		cost_span: {
			min: pricedVals.length ? Math.min(...pricedVals) : null,
			max: pricedVals.length ? Math.max(...pricedVals) : null
		},
		size_span: {
			min: sizeVals.length ? Math.min(...sizeVals) : null,
			max: sizeVals.length ? Math.max(...sizeVals) : null
		},
		f1_span: { min: f1s.length ? Math.min(...f1s) : 0, max: f1s.length ? Math.max(...f1s) : 0 },
		note
	};
}

// ── Generalization across gold benchmarks — the dataset-size dimension ───────
// frontier() reads ONE substrate. generalization() reads the SAME models ACROSS
// substrates: each model's error-F1 (+CI) at every gold it was scored on, ordered
// by gold size. This is how the small-gold mirage shows: a model measured on a
// small, narrow gold (rasmachine_v1, n=60) vs a large, de-biased one
// (external-578) MOVES, and its CI band visibly NARROWS as n grows — generalization
// and precision in one read. HONEST CAVEAT (carried into the UI): the substrates
// differ in COMPOSITION, not only size, so a shift conflates "more data" with
// "different (less curator-captured) data" — it's "across benchmarks", x-ordered
// by n, not a within-dataset learning curve (which would be flat — predetermined).

export interface GenPoint {
	sub_key: string;
	sub_label: string;
	gold_n: number;
	f1: number;
	lo: number;
	hi: number;
	n_reps: number;
}
export interface GenModel {
	model: string;
	is_open: boolean | null;
	points: GenPoint[]; // sorted ascending by gold_n
	delta: number; // f1(largest gold) − f1(smallest gold): negative = small gold overrated it
}
export interface Generalization {
	models: GenModel[];
	substrates: Array<{ key: string; label: string; gold_n: number }>;
	gold_min: number;
	gold_max: number;
	f1_min: number;
	f1_max: number;
}

export function generalization(): Generalization {
	const all = listRuns();
	type Cell = { f1: number; lo: number; hi: number; n: number };
	const byKey = new Map<string, Cell[]>(); // `${model}${substrate}` → reps
	const subN = new Map<string, number>(); // substrate → max gold_n observed
	const openOf = new Map<string, boolean | null>();
	for (const m of all) {
		const sub = m.substrate ?? `run:${m.run_id}`;
		const pairs = errorPairs(m);
		if (!pairs.length) continue;
		const n = pairs.length;
		const stats = prf(pairs);
		const ci = bootstrapF1CI(pairs, seedFromRunId(m.run_id));
		const k = `${m.model}${sub}`;
		(byKey.get(k) ?? byKey.set(k, []).get(k)!).push({ f1: stats.f1, lo: ci.lo, hi: ci.hi, n });
		subN.set(sub, Math.max(subN.get(sub) ?? 0, n));
		if (!openOf.has(m.model)) openOf.set(m.model, m.model_meta ? m.model_meta.is_open : null);
	}

	// Fold (model, substrate) repeats: mean F1; band widened to the rep union
	// (min lo, max hi) so repeated measurement never narrows uncertainty.
	const byModel = new Map<string, GenPoint[]>();
	for (const [k, cells] of byKey) {
		const sep = k.indexOf('');
		const model = k.slice(0, sep);
		const sub = k.slice(sep + 1);
		const f1 = cells.reduce((a, c) => a + c.f1, 0) / cells.length;
		(byModel.get(model) ?? byModel.set(model, []).get(model)!).push({
			sub_key: sub,
			sub_label: substrateLabel(sub),
			gold_n: Math.max(...cells.map((c) => c.n)),
			f1,
			lo: Math.min(...cells.map((c) => c.lo)),
			hi: Math.max(...cells.map((c) => c.hi)),
			n_reps: cells.length
		});
	}

	const models: GenModel[] = [];
	for (const [model, pts] of byModel) {
		if (pts.length < 2) continue; // need ≥2 substrates to draw a line
		pts.sort((a, b) => a.gold_n - b.gold_n);
		models.push({
			model,
			is_open: openOf.get(model) ?? null,
			points: pts,
			delta: pts[pts.length - 1].f1 - pts[0].f1
		});
	}
	models.sort((a, b) => a.delta - b.delta); // steepest droppers first

	const pts = models.flatMap((m) => m.points);
	const substrates = [...subN.entries()]
		.filter(([k]) => pts.some((p) => p.sub_key === k))
		.map(([key, n]) => ({ key, label: substrateLabel(key), gold_n: n }))
		.sort((a, b) => a.gold_n - b.gold_n);

	return {
		models,
		substrates,
		gold_min: pts.length ? Math.min(...pts.map((p) => p.gold_n)) : 0,
		gold_max: pts.length ? Math.max(...pts.map((p) => p.gold_n)) : 1,
		f1_min: pts.length ? Math.min(...pts.map((p) => p.lo)) : 0,
		f1_max: pts.length ? Math.max(...pts.map((p) => p.hi)) : 1
	};
}
