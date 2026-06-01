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
import { getRunData, getEvidenceIndex, evidenceForStatement } from './store';
import type { RunMeta, StatementRollup, EvidenceRow } from './types';

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
