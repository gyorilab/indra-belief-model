/**
 * The query API — every payload the viewer needs, derived from the monolithic
 * JSONL exports. This replaces viewer/src/lib/db.ts (the DuckDB query layer).
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
	goldForRow
} from './store';
import type { RunMeta, StatementRollup, EvidenceRow, GoldVerdict } from './types';

// ── Shared shapes ───────────────────────────────────────────────────────────

export interface RunSummary {
	run_id: string;
	model: string;
	status: string | null;
	generated_date: string | null;
	n_statements: number;
	n_evidences: number;
	bucket_counts: Record<string, number>;
}

function runSummary(m: RunMeta): RunSummary {
	return {
		run_id: m.run_id,
		model: m.model,
		status: m.status,
		generated_date: m.generated_date,
		n_statements: m.counts.statements ?? 0,
		n_evidences: m.counts.unique_evidence_rows ?? 0,
		bucket_counts: m.bucket_counts
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
	/** per confusion cell. */
	cells: {
		acbc: GoldCellTally;
		acbi: GoldCellTally;
		aibc: GoldCellTally;
		aibi: GoldCellTally;
	};
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
	a: { verdict: V; confidence: string | null; bucket: string | null; reasoning: string | null; tier: string | null; score: number | null };
	b: { verdict: V; confidence: string | null; bucket: string | null; reasoning: string | null; tier: string | null; score: number | null };
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
		a: { verdict: normV(ra.verdict), confidence: ra.confidence, bucket: ra.bucket, reasoning: ra.reasoning, tier: ra.tier, score: ra.our_score },
		b: { verdict: normV(rb.verdict), confidence: rb.confidence, bucket: rb.bucket, reasoning: rb.reasoning, tier: rb.tier, score: rb.our_score },
		gold
	};
}
