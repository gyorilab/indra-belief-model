/**
 * Server-only loader for the paper-literal vs LLM head-to-head.
 *
 * Mirrors `loadPaperMethodLandscape`: DATA_DIR join, size cap, `unavailable()`
 * status object, sha256 via node:crypto, and `displayPath`. Beyond the scalar
 * contract (validated purely in `$lib/data/paper-literal`), it reads the OOF
 * score vector, the semantic-port predictions, and the LLM prediction bundles,
 * joins each to the PAPER RELEASED label (paper_replication_policy
 * .released_paper_correct == OOF y_true, the exact label the reported AP used),
 * and computes per-arm score histograms, top exact-value piles, and stepped PR
 * curves. Strictly read-only: no data file is ever written or mutated.
 *
 * It also reads the sibling ap_decomposition_by_paper_band.json (the band-by-band
 * breakdown of the ΔAP column) and threads it through the same fail-closed
 * validator, so the figure that explains the delta cannot vanish silently.
 *
 * Because the arms must be COMPARED and not only described, the joins are keyed:
 * every pair carries the canonical `statement_id` (the paper's OOF `stmt_hash` is
 * resolved to it through the gold file), which lets each arm's AUROC be re-taken
 * on the block it actually orders with the paper reference arm scored on those
 * same statements — the qualification the AUROC lens carries.
 *
 * DEGRADE PATH: an arm whose vector fails to join keeps the validator's nulls.
 * It must never be handed a 0 for ece/Brier/AUROC — those are ideal values on
 * their scales, and a placeholder would render a broken join as a perfect arm.
 *
 * The 1 MB cap used by the sibling loader would reject the 1.64 MB OOF JSON and
 * the 4.14 MB gold JSONL, so the cap is raised to 16 MB here.
 */

import { createHash } from 'node:crypto';
import { existsSync, readFileSync, statSync } from 'node:fs';
import { join, relative, resolve } from 'node:path';

import { calibrationInterceptSlope } from '$lib/data/paper-calibration';
import { DATA_DIR } from '$lib/data/runs';
import {
	PAPER_LITERAL_ARM_SPECS,
	PAPER_LITERAL_REFERENCE_ARM_ID,
	PAPER_PR_CURVE_MAX_POINTS,
	PAPER_RELIABILITY_BIN_COUNT,
	PAPER_SCORE_BIN_COUNT,
	PAPER_SCORE_TOP_PILES,
	aurocOnRankedBlock,
	parsePaperReproduction,
	validatePaperLiteral,
	type PaperLiteralArm,
	type PaperLiteralLoad,
	type PaperLiteralPrPoint,
	type PaperLiteralReliabilityBin,
	type PaperLiteralScorePile,
	type PaperScoredPair
} from '$lib/data/paper-literal';

const MODEL_DIR = join(DATA_DIR, 'results', 'indra_paper_literal_models_20260724');
const VS_LLMS_PATH = join(MODEL_DIR, 'paper_literal_vs_llms.json');
const OOF_PATH = join(MODEL_DIR, 'paper_literal_table6_and_oof.json');
const MANIFEST_PATH = join(MODEL_DIR, 'manifest.json');
/** Band-by-band decomposition of the ΔAP column (~12 KB, same run directory). */
const AP_DECOMPOSITION_PATH = join(MODEL_DIR, 'ap_decomposition_by_paper_band.json');
const GOLD_PATH = join(
	DATA_DIR,
	'results',
	'indra_paper_statement_gold_20260717',
	'paper_statement_gold.jsonl'
);
const PORT_PATH = join(
	DATA_DIR,
	'results',
	'indra_paper_reproduction_20260717',
	'rf_promoter_all_sources_specific_predictions.jsonl'
);

/** OOF JSON (1.64 MB) and gold JSONL (4.14 MB) both fit comfortably under 16 MB. */
const MAX_ARTIFACT_BYTES = 16 * 1024 * 1024;

/** Paper arms read their aligned scores straight from the released OOF vector. */
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

/**
 * The aligned row shape lives in `$lib/data/paper-literal` so the pure ranking
 * math and this loader agree on one definition of "keyed pair".
 */
type ScoredPair = PaperScoredPair;

type GuardedRead =
	| { ok: true; bytes: Buffer }
	| { ok: false; reason: string };

function displayPath(path: string): string {
	const repoRoot = resolve(DATA_DIR, '..');
	const rel = relative(repoRoot, path).replaceAll('\\', '/');
	return rel && !rel.startsWith('../') ? rel : path;
}

function unavailable(reason: string, digest: string | null = null): PaperLiteralLoad {
	return {
		status: 'unavailable',
		reason,
		artifact_path: displayPath(VS_LLMS_PATH),
		artifact_sha256: digest,
		arms: [],
		faithfulness: null,
		reproduction: null,
		apDecomposition: null,
		generatedNote: null,
		generatedNoteProse: null
	};
}

/** existsSync + statSync size cap + readFileSync, mirroring the sibling loader. */
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

/**
 * The OOF `stmt_hash` values are bare 64-bit integer literals; 1429/1689 exceed
 * 2^53, so a plain JSON.parse silently rounds them and the string join to the
 * gold `paper_statement_hash` collapses (only ~688/1689 survive). Quote every
 * `stmt_hash` literal so it parses as an exact string before matching. Only that
 * key is touched; float scores and small ints keep their numeric type.
 */
function parseOofPreservingHashes(text: string): unknown {
	return JSON.parse(text.replace(/("stmt_hash"\s*:\s*)(-?\d+)/g, '$1"$2"'));
}

function finiteNumber(value: unknown): number | null {
	return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

/**
 * paper_statement_hash(String) -> released_paper_correct AND
 * statement_id -> released_paper_correct. Uses the PAPER released label, never
 * the adjudicated label (which diverges on 111/1689 and would desync the server
 * PR curve from the reported AP).
 *
 * Also returns paper_statement_hash -> statement_id: the paper arms are keyed by
 * hash and the LLM arms by statement_id, so this map is what makes the two
 * families intersectable on one identical statement set.
 */
function buildReleasedLabelMaps(goldBytes: Buffer): {
	byHash: Map<string, number>;
	bySid: Map<string, number>;
	sidByHash: Map<string, string>;
} {
	const byHash = new Map<string, number>();
	const bySid = new Map<string, number>();
	const sidByHash = new Map<string, string>();
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
		const hashKey =
			typeof hash === 'string' ? hash : typeof hash === 'number' ? String(hash) : null;
		if (hashKey !== null) byHash.set(hashKey, released);
		const corpus = asRecord(row.canonical_corpus);
		const sid = corpus?.statement_id;
		if (typeof sid === 'string') {
			bySid.set(sid, released);
			if (hashKey !== null) sidByHash.set(hashKey, sid);
		}
	}
	return { byHash, bySid, sidByHash };
}

function collectOofPairs(
	records: unknown,
	byHash: Map<string, number>,
	sidByHash: Map<string, string>
): ScoredPair[] | null {
	if (!Array.isArray(records)) return null;
	const pairs: ScoredPair[] = [];
	for (const entry of records) {
		const row = asRecord(entry);
		if (!row) continue;
		const score = finiteNumber(row.prob_correct);
		if (score === null) continue;
		const hash = row.stmt_hash;
		const hashKey =
			typeof hash === 'number' ? String(hash) : typeof hash === 'string' ? hash : null;
		if (hashKey === null) continue;
		const label = byHash.get(hashKey);
		if (label === undefined) continue;
		// The cross-arm key is the canonical statement_id; a hash with no id still
		// scores and plots, it just cannot take part in a paired subset.
		pairs.push({ key: sidByHash.get(hashKey) ?? null, score, label });
	}
	return pairs.length ? pairs : null;
}

function collectPredictionPairs(
	path: string,
	bySid: Map<string, number>
): ScoredPair[] | null {
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
		pairs.push({ key: sid, score, label });
	}
	return pairs.length ? pairs : null;
}

function scoreBins(pairs: ScoredPair[]): number[] {
	const bins = new Array<number>(PAPER_SCORE_BIN_COUNT).fill(0);
	for (const { score } of pairs) {
		const clamped = score < 0 ? 0 : score > 1 ? 1 : score;
		let index = Math.floor(clamped * PAPER_SCORE_BIN_COUNT);
		if (index >= PAPER_SCORE_BIN_COUNT) index = PAPER_SCORE_BIN_COUNT - 1;
		if (index < 0) index = 0;
		bins[index] += 1;
	}
	return bins;
}

function scoreTopPiles(pairs: ScoredPair[]): PaperLiteralScorePile[] {
	const counts = new Map<number, number>();
	for (const { score } of pairs) counts.set(score, (counts.get(score) ?? 0) + 1);
	return [...counts.entries()]
		.map(([value, count]) => ({ value, count }))
		.sort((a, b) => b.count - a.count || a.value - b.value)
		.slice(0, PAPER_SCORE_TOP_PILES);
}

function round6(value: number): number {
	return Math.round(value * 1e6) / 1e6;
}

/**
 * Murphy Brier decomposition over PAPER_RELIABILITY_BIN_COUNT (10) equal-width
 * [0,1] bins on the same aligned (score,label) vector: uncertainty = ō(1−ō);
 * reliability = Σ(n_k/N)(p̄_k−ō_k)²; resolution = Σ(n_k/N)(ō_k−ō)²; brier =
 * mean((score−label)²) (the canonical metrics.py `brier`). The classical
 * identity brier ≈ uncertainty + reliability − resolution holds up to the
 * within-bin forecast-variance residual (≤~1e-3 here) — brier itself is exact.
 */
function brierDecomposition(pairs: ScoredPair[]): {
	brier: number;
	brierReliability: number;
	brierResolution: number;
	brierUncertainty: number;
} {
	const total = pairs.length;
	const sum = new Array<number>(PAPER_RELIABILITY_BIN_COUNT).fill(0);
	const positives = new Array<number>(PAPER_RELIABILITY_BIN_COUNT).fill(0);
	const counts = new Array<number>(PAPER_RELIABILITY_BIN_COUNT).fill(0);
	let sumLabel = 0;
	let sqError = 0;
	for (const { score, label } of pairs) {
		const y = label > 0 ? 1 : 0;
		sumLabel += y;
		const clamped = score < 0 ? 0 : score > 1 ? 1 : score;
		sqError += (clamped - y) * (clamped - y);
		let index = Math.floor(clamped * PAPER_RELIABILITY_BIN_COUNT);
		if (index >= PAPER_RELIABILITY_BIN_COUNT) index = PAPER_RELIABILITY_BIN_COUNT - 1;
		if (index < 0) index = 0;
		sum[index] += clamped;
		if (y > 0) positives[index] += 1;
		counts[index] += 1;
	}
	const oBar = sumLabel / total;
	let reliability = 0;
	let resolution = 0;
	for (let i = 0; i < PAPER_RELIABILITY_BIN_COUNT; i += 1) {
		const n = counts[i];
		if (n === 0) continue;
		const pMean = sum[i] / n;
		const yRate = positives[i] / n;
		reliability += (n / total) * (pMean - yRate) * (pMean - yRate);
		resolution += (n / total) * (yRate - oBar) * (yRate - oBar);
	}
	return {
		brier: round6(sqError / total),
		brierReliability: round6(reliability),
		brierResolution: round6(resolution),
		brierUncertainty: round6(oBar * (1 - oBar))
	};
}

/**
 * Per-arm reliability (calibration) over PAPER_RELIABILITY_BIN_COUNT equal-width
 * [0,1] bins, from the SAME aligned (score,label) vector as scoreBins. Each
 * occupied bin carries mean predicted probability, observed positive
 * (released-correct) rate, and count; ECE is the count-weighted mean
 * |observed − predicted| across occupied bins. Empty bins are dropped. The
 * paper RF, near-continuous, hugs the diagonal (small ECE); the LLM arms, piled
 * on a few scores, deviate. `pairs` is always non-empty here (armPairs returns
 * null for a scoreless arm), so the ECE denominator is never zero.
 */
function reliability(pairs: ScoredPair[]): {
	reliabilityBins: PaperLiteralReliabilityBin[];
	ece: number;
} {
	const sum = new Array<number>(PAPER_RELIABILITY_BIN_COUNT).fill(0);
	const positives = new Array<number>(PAPER_RELIABILITY_BIN_COUNT).fill(0);
	const counts = new Array<number>(PAPER_RELIABILITY_BIN_COUNT).fill(0);
	for (const { score, label } of pairs) {
		const clamped = score < 0 ? 0 : score > 1 ? 1 : score;
		let index = Math.floor(clamped * PAPER_RELIABILITY_BIN_COUNT);
		if (index >= PAPER_RELIABILITY_BIN_COUNT) index = PAPER_RELIABILITY_BIN_COUNT - 1;
		if (index < 0) index = 0;
		sum[index] += clamped;
		if (label > 0) positives[index] += 1;
		counts[index] += 1;
	}
	const total = pairs.length;
	const bins: PaperLiteralReliabilityBin[] = [];
	let ece = 0;
	for (let i = 0; i < PAPER_RELIABILITY_BIN_COUNT; i += 1) {
		const n = counts[i];
		if (n === 0) continue;
		const pMean = sum[i] / n;
		const yRate = positives[i] / n;
		bins.push({ p_mean: round6(pMean), y_rate: round6(yRate), n });
		ece += (n / total) * Math.abs(yRate - pMean);
	}
	return { reliabilityBins: bins, ece: round6(ece) };
}

/**
 * sklearn-style stepped PR curve on the released label: sort by score desc,
 * accumulate TP/FP across each distinct-score threshold, emit one vertex per
 * threshold (plus the (recall=0, precision=1) origin), then uniformly cap the
 * vertex count for the client while preserving the endpoints.
 */
function prCurve(pairs: ScoredPair[]): PaperLiteralPrPoint[] {
	const positives = pairs.reduce((sum, pair) => sum + (pair.label > 0 ? 1 : 0), 0);
	if (positives === 0) return [];
	const sorted = [...pairs].sort((a, b) => b.score - a.score);
	const points: PaperLiteralPrPoint[] = [{ recall: 0, precision: 1 }];
	let tp = 0;
	let fp = 0;
	let index = 0;
	while (index < sorted.length) {
		const threshold = sorted[index].score;
		while (index < sorted.length && sorted[index].score === threshold) {
			if (sorted[index].label > 0) tp += 1;
			else fp += 1;
			index += 1;
		}
		const precision = tp + fp > 0 ? tp / (tp + fp) : 1;
		points.push({ recall: round6(tp / positives), precision: round6(precision) });
	}
	return capCurve(points, PAPER_PR_CURVE_MAX_POINTS);
}

function capCurve(points: PaperLiteralPrPoint[], max: number): PaperLiteralPrPoint[] {
	if (points.length <= max) return points;
	const out: PaperLiteralPrPoint[] = [];
	const stride = (points.length - 1) / (max - 1);
	let previous = -1;
	for (let step = 0; step < max; step += 1) {
		const at = step === max - 1 ? points.length - 1 : Math.round(step * stride);
		if (at !== previous) {
			out.push(points[at]);
			previous = at;
		}
	}
	return out;
}

function armPairs(
	arm: PaperLiteralArm,
	oofPredictions: Record<string, unknown> | null,
	byHash: Map<string, number>,
	bySid: Map<string, number>,
	sidByHash: Map<string, string>
): ScoredPair[] | null {
	const oofKey = OOF_KEY_BY_ID[arm.id];
	if (oofKey) {
		return oofPredictions ? collectOofPairs(oofPredictions[oofKey], byHash, sidByHash) : null;
	}
	if (arm.id === 'port-rf-promoter') {
		return collectPredictionPairs(PORT_PATH, bySid);
	}
	const dir = LLM_DIR_BY_ID[arm.id];
	if (!dir) return null;
	return collectPredictionPairs(
		join(DATA_DIR, 'comparison', 'models', dir, 'all_source_predictions.jsonl'),
		bySid
	);
}

/** Load the paper-literal comparison, computing per-arm score geometry. */
export function loadPaperLiteral(): PaperLiteralLoad {
	const vs = readGuarded(VS_LLMS_PATH);
	if (!vs.ok) return unavailable(vs.reason);
	const digest = createHash('sha256').update(vs.bytes).digest('hex');

	let vsRaw: unknown;
	try {
		vsRaw = JSON.parse(vs.bytes.toString('utf8'));
	} catch {
		return unavailable(`${displayPath(VS_LLMS_PATH)} is not valid JSON.`, digest);
	}

	const manifest = readGuarded(MANIFEST_PATH);
	if (!manifest.ok) return unavailable(manifest.reason, digest);
	let manifestRaw: unknown;
	try {
		manifestRaw = JSON.parse(manifest.bytes.toString('utf8'));
	} catch {
		return unavailable(`${displayPath(MANIFEST_PATH)} is not valid JSON.`, digest);
	}

	const oof = readGuarded(OOF_PATH);
	if (!oof.ok) return unavailable(oof.reason, digest);
	let oofRaw: unknown;
	try {
		oofRaw = parseOofPreservingHashes(oof.bytes.toString('utf8'));
	} catch {
		return unavailable(`${displayPath(OOF_PATH)} is not valid JSON.`, digest);
	}

	const gold = readGuarded(GOLD_PATH);
	if (!gold.ok) return unavailable(gold.reason, digest);

	// The decomposition explains the head-to-head delta column, so a missing or
	// malformed payload gates the whole load — never a silently missing figure.
	const decomposition = readGuarded(AP_DECOMPOSITION_PATH);
	if (!decomposition.ok) return unavailable(decomposition.reason, digest);
	let decompositionRaw: unknown;
	try {
		decompositionRaw = JSON.parse(decomposition.bytes.toString('utf8'));
	} catch {
		return unavailable(`${displayPath(AP_DECOMPOSITION_PATH)} is not valid JSON.`, digest);
	}

	const base = validatePaperLiteral(vsRaw, {
		artifactPath: displayPath(VS_LLMS_PATH),
		artifactSha256: digest,
		apDecomposition: decompositionRaw
	});
	if (base.status !== 'ok') return base;

	const { byHash, bySid, sidByHash } = buildReleasedLabelMaps(gold.bytes);
	const oofRoot = asRecord(oofRaw);
	const oofPredictions = oofRoot ? asRecord(oofRoot.oof_predictions) : null;

	// Join every arm FIRST: the ranked-block AUROC scores the paper reference arm
	// on another arm's subset, so the reference vector has to exist before any
	// arm's geometry is built.
	const pairsByArmId = new Map<string, ScoredPair[] | null>(
		base.arms.map((arm) => [arm.id, armPairs(arm, oofPredictions, byHash, bySid, sidByHash)])
	);
	const referencePairs = pairsByArmId.get(PAPER_LITERAL_REFERENCE_ARM_ID) ?? null;
	// statement_id -> reference score. Empty when the reference itself fails to
	// join, which leaves every arm's `aurocOnRanked` null — the AUROC lens then
	// reports the check as unavailable instead of showing the gain unqualified.
	const referenceScoreByKey = new Map<string, number>();
	for (const pair of referencePairs ?? []) {
		if (pair.key !== null) referenceScoreByKey.set(pair.key, pair.score);
	}

	const arms = base.arms.map((arm) => {
		const pairs = pairsByArmId.get(arm.id) ?? null;
		// Degrade a single arm rather than fail the load. It keeps the validator's
		// NULL scalars — handing it 0 would print a failed join as ideal.
		if (!pairs) return arm;
		return {
			...arm,
			scoreBins: scoreBins(pairs),
			scoreTopPiles: scoreTopPiles(pairs),
			prCurve: prCurve(pairs),
			...reliability(pairs),
			...calibrationInterceptSlope(pairs),
			...brierDecomposition(pairs),
			aurocOnRanked: aurocOnRankedBlock(pairs, referenceScoreByKey)
		};
	});

	const manifestRoot = asRecord(manifestRaw);
	const createdAt =
		manifestRoot && typeof manifestRoot.created_at === 'string' ? manifestRoot.created_at : null;
	// BOTH HALVES OF THE TWIN take the suffix. Appending to the flat string alone
	// would leave `generatedNoteProse.shipped` a prefix of the string this loader
	// actually ships, which is the one thing the twin promises it is not.
	const generatedNoteProse = createdAt
		? {
				shipped: `${base.generatedNoteProse.shipped} Reproduced ${createdAt}.`,
				plain: `${base.generatedNoteProse.plain} Reproduced ${createdAt}.`
			}
		: base.generatedNoteProse;
	const generatedNote = generatedNoteProse.shipped;

	// Fail-closed manifest parse; null degrades the reproduction block gracefully.
	const reproduction = parsePaperReproduction(manifestRaw);

	return { ...base, arms, generatedNote, generatedNoteProse, reproduction };
}
