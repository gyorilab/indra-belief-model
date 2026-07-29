/**
 * Server-only loader for the tie-inflation explainer.
 *
 * Mirrors `loadPaperLiteral`: DATA_DIR join, size cap, `unavailable()` status
 * object (never a throw), sha256 via node:crypto, `displayPath`.
 *
 * Beyond the shipped scalars (validated purely in `$lib/data/paper-tie-inflation`)
 * it rebuilds the REAL stepped precision-recall geometry for exactly two arms —
 * the featured reader and the paired-delta reference — from their prediction
 * vectors joined to the PAPER RELEASED label
 * (`paper_replication_policy.released_paper_correct`, the same label the reported
 * AP used). Nothing here is a schematic: the figure this feeds is drawn from the
 * same vectors the shipped metrics were computed from.
 *
 * FAIL-CLOSED RECONCILIATION. For each drawn arm the loader recomputes pooled
 * average precision (`Σ ΔR·P_new`), the pooled trapezoid (`Σ ΔR·(P_old+P_new)/2`)
 * and the triangle sum (`Σ ΔR·ΔP/2`), and requires all three to reproduce the
 * SHIPPED `pooled_average_precision`, `pooled_trapezoidal_pr_auc`, and their
 * difference within TIE_RECONCILIATION_TOLERANCE. A drifted join — a changed gold
 * file, a re-emitted prediction bundle — gates the whole panel to `unavailable`
 * rather than drawing triangles that no longer add up to the number they claim to
 * explain.
 *
 * Strictly read-only: no data file is ever written or mutated.
 */

import { createHash } from 'node:crypto';
import { existsSync, readFileSync, statSync } from 'node:fs';
import { join, relative, resolve } from 'node:path';

import { DATA_DIR } from '$lib/data/runs';
import {
	TIE_RECONCILIATION_TOLERANCE,
	featuredReaderArm,
	tieReferenceArm,
	validateTieInflation,
	type TieBlock,
	type TieFeaturedArm,
	type TieInflationArm,
	type TieInflationLoad,
	type TiePoint
} from '$lib/data/paper-tie-inflation';

const MODEL_DIR = join(DATA_DIR, 'results', 'indra_paper_literal_models_20260724');
const VS_LLMS_PATH = join(MODEL_DIR, 'paper_literal_vs_llms.json');
const OOF_PATH = join(MODEL_DIR, 'paper_literal_table6_and_oof.json');
const GOLD_PATH = join(
	DATA_DIR,
	'results',
	'indra_paper_statement_gold_20260717',
	'paper_statement_gold.jsonl'
);

/** Same cap as the sibling paper loader: the 4.14 MB gold JSONL must fit. */
const MAX_ARTIFACT_BYTES = 16 * 1024 * 1024;

/** Paper arms read their aligned scores from the released out-of-fold vector. */
const OOF_KEY_BY_ID: Record<string, string> = {
	'paper-rf-promoter': 'RF 2k-d13 + Type/#PMIDs/promoter - all sources, specific',
	'paper-rf-prom-avglen': 'RF 2k-d13 + Type/#PMIDs/prom/avglen - all sources, specific'
};

/** LLM arms read `probability_correct` from their comparison prediction bundle. */
const LLM_DIR_BY_ID: Record<string, string> = {
	'gemma-4-e2b': 'gemma_4_e2b',
	'gemma-4-26b': 'gemma_4_26b',
	'gemma-4-31b': 'gemma_4_31b',
	'glm-5': 'glm_5',
	'indra-cogex-hybrid': 'indra_cogex_hybrid'
};

const PORT_PATH = join(
	DATA_DIR,
	'results',
	'indra_paper_reproduction_20260717',
	'rf_promoter_all_sources_specific_predictions.jsonl'
);

interface ScoredPair {
	score: number;
	label: number;
}

/** One achievable operating point plus the tied block that produced it. */
interface Vertex {
	recall: number;
	precision: number;
	/** The tied score admitted at this vertex; null for the (0,1) origin. */
	score: number | null;
	blockTrue: number;
	blockFalse: number;
}

type GuardedRead = { ok: true; bytes: Buffer } | { ok: false; reason: string };

function displayPath(path: string): string {
	const repoRoot = resolve(DATA_DIR, '..');
	const rel = relative(repoRoot, path).replaceAll('\\', '/');
	return rel && !rel.startsWith('../') ? rel : path;
}

function unavailable(reason: string, digest: string | null = null): TieInflationLoad {
	return {
		status: 'unavailable',
		reason,
		artifact_path: displayPath(VS_LLMS_PATH),
		artifact_sha256: digest,
		nStatements: null,
		nPositives: null,
		arms: [],
		featured: null,
		margin: null,
		generatedNote: null
	};
}

function readGuarded(path: string): GuardedRead {
	if (!existsSync(path)) return { ok: false, reason: `${displayPath(path)} is missing.` };
	try {
		if (statSync(path).size > MAX_ARTIFACT_BYTES) {
			return { ok: false, reason: `${displayPath(path)} exceeds ${MAX_ARTIFACT_BYTES} bytes.` };
		}
		return { ok: true, bytes: readFileSync(path) };
	} catch (error) {
		return { ok: false, reason: `${displayPath(path)} could not be read: ${String(error)}` };
	}
}

function asRecord(value: unknown): Record<string, unknown> | null {
	return value !== null && typeof value === 'object' && !Array.isArray(value)
		? (value as Record<string, unknown>)
		: null;
}

function finiteNumber(value: unknown): number | null {
	return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

/**
 * The OOF `stmt_hash` values are bare 64-bit integer literals and most exceed
 * 2^53, so a plain JSON.parse silently rounds them and the string join to the
 * gold `paper_statement_hash` collapses. Quote every `stmt_hash` literal so it
 * parses as an exact string. Same guard the sibling paper loader applies; it is
 * duplicated rather than imported so this node owns its own read path and cannot
 * be broken by an unrelated edit to that loader.
 */
function parseOofPreservingHashes(text: string): unknown {
	return JSON.parse(text.replace(/("stmt_hash"\s*:\s*)(-?\d+)/g, '$1"$2"'));
}

function buildReleasedLabelMaps(goldBytes: Buffer): {
	byHash: Map<string, number>;
	bySid: Map<string, number>;
} {
	const byHash = new Map<string, number>();
	const bySid = new Map<string, number>();
	for (const line of goldBytes.toString('utf8').split('\n')) {
		if (!line.trim()) continue;
		let parsed: unknown;
		try {
			parsed = JSON.parse(line);
		} catch {
			continue;
		}
		const row = asRecord(parsed);
		if (!row) continue;
		const policy = asRecord(row.paper_replication_policy);
		const released = policy ? finiteNumber(policy.released_paper_correct) : null;
		if (released === null) continue;
		const hash = row.paper_statement_hash;
		if (typeof hash === 'string') byHash.set(hash, released);
		else if (typeof hash === 'number') byHash.set(String(hash), released);
		const corpus = asRecord(row.canonical_corpus);
		const sid = corpus?.statement_id;
		if (typeof sid === 'string') bySid.set(sid, released);
	}
	return { byHash, bySid };
}

function collectOofPairs(records: unknown, byHash: Map<string, number>): ScoredPair[] | null {
	if (!Array.isArray(records)) return null;
	const pairs: ScoredPair[] = [];
	for (const entry of records) {
		const row = asRecord(entry);
		if (!row) continue;
		const score = finiteNumber(row.prob_correct);
		if (score === null) continue;
		const hash = row.stmt_hash;
		const key = typeof hash === 'number' ? String(hash) : typeof hash === 'string' ? hash : null;
		if (key === null) continue;
		const label = byHash.get(key);
		if (label === undefined) continue;
		pairs.push({ score, label });
	}
	return pairs.length ? pairs : null;
}

function collectPredictionPairs(path: string, bySid: Map<string, number>): ScoredPair[] | null {
	const read = readGuarded(path);
	if (!read.ok) return null;
	const pairs: ScoredPair[] = [];
	for (const line of read.bytes.toString('utf8').split('\n')) {
		if (!line.trim()) continue;
		let parsed: unknown;
		try {
			parsed = JSON.parse(line);
		} catch {
			continue;
		}
		const row = asRecord(parsed);
		if (!row) continue;
		const score = finiteNumber(row.probability_correct);
		const sid = row.statement_id;
		if (score === null || typeof sid !== 'string') continue;
		const label = bySid.get(sid);
		if (label === undefined) continue;
		pairs.push({ score, label });
	}
	return pairs.length ? pairs : null;
}

function armPairs(
	arm: TieInflationArm,
	oofPredictions: Record<string, unknown> | null,
	byHash: Map<string, number>,
	bySid: Map<string, number>
): ScoredPair[] | null {
	const oofKey = OOF_KEY_BY_ID[arm.id];
	if (oofKey) return oofPredictions ? collectOofPairs(oofPredictions[oofKey], byHash) : null;
	if (arm.id === 'port-rf-promoter') return collectPredictionPairs(PORT_PATH, bySid);
	const dir = LLM_DIR_BY_ID[arm.id];
	if (!dir) return null;
	return collectPredictionPairs(
		join(DATA_DIR, 'comparison', 'models', dir, 'all_source_predictions.jsonl'),
		bySid
	);
}

/**
 * The achievable PR operating points, one vertex per DISTINCT score. Sorting by
 * score descending and admitting every tied statement in one go is precisely why
 * the curve steps: no threshold exists between two tied statements, so no
 * operating point exists between two adjacent vertices.
 */
function verticesOf(pairs: ScoredPair[]): Vertex[] | null {
	const positives = pairs.reduce((sum, pair) => sum + (pair.label > 0 ? 1 : 0), 0);
	if (positives === 0) return null;
	const sorted = [...pairs].sort((a, b) => b.score - a.score);
	const out: Vertex[] = [{ recall: 0, precision: 1, score: null, blockTrue: 0, blockFalse: 0 }];
	let tp = 0;
	let fp = 0;
	let index = 0;
	while (index < sorted.length) {
		const threshold = sorted[index].score;
		let blockTrue = 0;
		let blockFalse = 0;
		while (index < sorted.length && sorted[index].score === threshold) {
			if (sorted[index].label > 0) blockTrue += 1;
			else blockFalse += 1;
			index += 1;
		}
		tp += blockTrue;
		fp += blockFalse;
		out.push({
			recall: tp / positives,
			precision: tp / (tp + fp),
			score: threshold,
			blockTrue,
			blockFalse
		});
	}
	return out;
}

interface CurveTotals {
	ap: number;
	trapezoid: number;
	triangles: number;
}

/**
 * The whole argument in six lines. Over one segment the trapezoid credits the
 * MEAN of the two precisions across ΔR; average precision credits only the new
 * (achievable) precision. Their difference is the triangle ΔR·ΔP/2, and summing
 * the three quantities over the curve gives AP, the trapezoid, and exactly the
 * gap between them.
 */
function curveTotals(vertices: Vertex[]): CurveTotals {
	let ap = 0;
	let trapezoid = 0;
	let triangles = 0;
	for (let i = 1; i < vertices.length; i += 1) {
		const previous = vertices[i - 1];
		const current = vertices[i];
		const dR = current.recall - previous.recall;
		ap += dR * current.precision;
		trapezoid += (dR * (previous.precision + current.precision)) / 2;
		triangles += (dR * (previous.precision - current.precision)) / 2;
	}
	return { ap, trapezoid, triangles };
}

/** Transport precision: 9 dp keeps the drawn path indistinguishable from exact. */
function round9(value: number): number {
	return Math.round(value * 1e9) / 1e9;
}

function toPoint(vertex: Vertex): TiePoint {
	return { recall: round9(vertex.recall), precision: round9(vertex.precision) };
}

/** Achievable (stepped) precision at `recall`: the first vertex that reaches it. */
function stepPrecisionAt(vertices: Vertex[], recall: number): number {
	for (const vertex of vertices) {
		if (vertex.recall >= recall) return vertex.precision;
	}
	return vertices[vertices.length - 1].precision;
}

/** Precision the trapezoid's chord credits at `recall` (linear between vertices). */
function chordPrecisionAt(vertices: Vertex[], recall: number): number {
	for (let i = 1; i < vertices.length; i += 1) {
		const a = vertices[i - 1];
		const b = vertices[i];
		if (recall <= b.recall) {
			const span = b.recall - a.recall;
			if (span <= 0) return b.precision;
			const t = (recall - a.recall) / span;
			return a.precision + t * (b.precision - a.precision);
		}
	}
	return vertices[vertices.length - 1].precision;
}

interface ArmGeometry {
	featured: TieFeaturedArm;
	vertices: Vertex[];
	totals: CurveTotals;
}

/**
 * Build one arm's drawable geometry over the featured recall window and check it
 * against the shipped scalars. `vertices` carries every in-window operating point
 * plus the last one before the window, so the path ENTERS the frame rather than
 * starting inside it (the drawing clips to the axes). `windowInflation` sums only
 * the segments whose two endpoints both lie in the window, so it never counts a
 * partially visible triangle.
 */
function buildGeometry(
	arm: TieInflationArm,
	vertices: Vertex[],
	recallFrom: number,
	midRecall: number
): ArmGeometry | { error: string } {
	const totals = curveTotals(vertices);
	const shippedGap = arm.pooledTrapezoidal - arm.ap;
	const residuals = [
		Math.abs(totals.ap - arm.ap),
		Math.abs(totals.trapezoid - arm.pooledTrapezoidal),
		Math.abs(totals.triangles - shippedGap)
	];
	const worst = Math.max(...residuals);
	if (worst > TIE_RECONCILIATION_TOLERANCE) {
		return {
			error:
				`${arm.display}: rebuilt precision-recall geometry does not reproduce the shipped ` +
				`metrics (worst residual ${worst.toExponential(2)} > ${TIE_RECONCILIATION_TOLERANCE.toExponential(0)}).`
		};
	}

	const firstInWindow = vertices.findIndex((vertex) => vertex.recall >= recallFrom);
	const start = firstInWindow <= 0 ? 0 : firstInWindow - 1;
	const drawn = vertices.slice(start);

	let windowInflation = 0;
	for (let i = 1; i < vertices.length; i += 1) {
		const previous = vertices[i - 1];
		const current = vertices[i];
		if (previous.recall < recallFrom) continue;
		windowInflation += ((current.recall - previous.recall) * (previous.precision - current.precision)) / 2;
	}

	return {
		vertices,
		totals,
		featured: {
			id: arm.id,
			display: arm.display,
			kind: arm.kind,
			vertices: drawn.map(toPoint),
			windowInflation,
			totalInflation: totals.triangles,
			reconciliationResidual: worst,
			midStepPrecision: stepPrecisionAt(vertices, midRecall),
			midChordPrecision: chordPrecisionAt(vertices, midRecall)
		}
	};
}

/**
 * The segment contributing the most interpolated area — the dominant tie block.
 * Returns the UNROUNDED endpoint vertices alongside the transportable block, so
 * the window is filtered on exact recalls: rounding the window bound to 9 dp can
 * push it a whole ulp above its own endpoint and silently drop the one segment
 * the figure exists to draw.
 */
function dominantBlock(
	vertices: Vertex[],
	armInflation: number
): { block: TieBlock; from: Vertex; to: Vertex } | null {
	let best: { area: number; index: number } | null = null;
	for (let i = 1; i < vertices.length; i += 1) {
		const previous = vertices[i - 1];
		const current = vertices[i];
		const area = ((current.recall - previous.recall) * (previous.precision - current.precision)) / 2;
		if (!best || area > best.area) best = { area, index: i };
	}
	if (!best) return null;
	const previous = vertices[best.index - 1];
	const current = vertices[best.index];
	if (current.score === null) return null;
	return {
		from: previous,
		to: current,
		block: {
			score: current.score,
			size: current.blockTrue + current.blockFalse,
			nTrue: current.blockTrue,
			nFalse: current.blockFalse,
			from: toPoint(previous),
			to: toPoint(current),
			area: best.area,
			shareOfArmInflation: armInflation === 0 ? 0 : best.area / armInflation
		}
	};
}

/**
 * Load the tie-inflation panel: shipped inflation/tie-ness scalars for all eight
 * arms, plus real stepped PR geometry for the featured reader and the paper's own
 * reference RF over the reader's dominant tied block.
 */
export function loadPaperTieInflation(): TieInflationLoad {
	const vs = readGuarded(VS_LLMS_PATH);
	if (!vs.ok) return unavailable(vs.reason);
	const digest = createHash('sha256').update(vs.bytes).digest('hex');

	let vsRaw: unknown;
	try {
		vsRaw = JSON.parse(vs.bytes.toString('utf8'));
	} catch {
		return unavailable(`${displayPath(VS_LLMS_PATH)} is not valid JSON.`, digest);
	}

	const base = validateTieInflation(vsRaw, {
		artifactPath: displayPath(VS_LLMS_PATH),
		artifactSha256: digest
	});
	if (base.status !== 'ok') return base;

	const reader = featuredReaderArm(base.arms);
	const reference = tieReferenceArm(base.arms);
	if (!reader || !reference) {
		return unavailable('Featured reader or reference arm is missing from the artifact.', digest);
	}

	const gold = readGuarded(GOLD_PATH);
	if (!gold.ok) return unavailable(gold.reason, digest);
	const oof = readGuarded(OOF_PATH);
	if (!oof.ok) return unavailable(oof.reason, digest);
	let oofRaw: unknown;
	try {
		oofRaw = parseOofPreservingHashes(oof.bytes.toString('utf8'));
	} catch {
		return unavailable(`${displayPath(OOF_PATH)} is not valid JSON.`, digest);
	}

	const { byHash, bySid } = buildReleasedLabelMaps(gold.bytes);
	const oofRoot = asRecord(oofRaw);
	const oofPredictions = oofRoot ? asRecord(oofRoot.oof_predictions) : null;

	const readerPairs = armPairs(reader, oofPredictions, byHash, bySid);
	const referencePairs = armPairs(reference, oofPredictions, byHash, bySid);
	if (!readerPairs || !referencePairs) {
		return unavailable('Aligned prediction vectors for the featured arms are unavailable.', digest);
	}
	const readerVertices = verticesOf(readerPairs);
	const referenceVertices = verticesOf(referencePairs);
	if (!readerVertices || !referenceVertices) {
		return unavailable('The featured arms carry no positive-labelled statements.', digest);
	}

	// The window is the dominant tied block's own recall span, so the figure is
	// framed on the mechanism rather than on an arbitrary zoom. Filtering uses the
	// EXACT endpoint recalls, never their 9 dp transport rounding.
	const dominant = dominantBlock(readerVertices, reader.pooledTrapezoidal - reader.ap);
	if (!dominant) {
		return unavailable('No tied block could be identified for the featured arm.', digest);
	}
	const { block } = dominant;
	const recallFrom = dominant.from.recall;
	const recallTo = dominant.to.recall;
	const midRecall = (recallFrom + recallTo) / 2;

	const readerGeometry = buildGeometry(reader, readerVertices, recallFrom, midRecall);
	if ('error' in readerGeometry) return unavailable(readerGeometry.error, digest);
	const referenceGeometry = buildGeometry(reference, referenceVertices, recallFrom, midRecall);
	if ('error' in referenceGeometry) return unavailable(referenceGeometry.error, digest);

	// Shared precision axis: the extent of the two arms' IN-window operating
	// points (the entry vertex sitting outside the window is drawn but clipped,
	// so it must not stretch the scale).
	const inWindow = [...readerVertices, ...referenceVertices].filter(
		(vertex) => vertex.recall >= recallFrom && vertex.recall <= recallTo
	);
	if (inWindow.length === 0) {
		return unavailable('The featured window contains no operating points.', digest);
	}
	const precisions = inWindow.map((vertex) => vertex.precision);

	const nPositives = readerPairs.reduce((sum, pair) => sum + (pair.label > 0 ? 1 : 0), 0);

	return {
		...base,
		nPositives,
		featured: {
			reader: readerGeometry.featured,
			reference: referenceGeometry.featured,
			block,
			recallFrom: round9(recallFrom),
			recallTo: round9(recallTo),
			precisionMin: Math.min(...precisions),
			precisionMax: Math.max(...precisions),
			midRecall
		}
	};
}
