/**
 * Server-only loader for the belief-model ladder (/paper beat 3).
 *
 * Mirrors `loadReviewQueue` exactly: DATA_DIR join, size cap, `unavailable()`
 * status object, sha256 via node:crypto, `displayPath`, and a fail-closed
 * validator. The ladder artifact is additionally pinned to the run manifest's
 * `output_sha256` — the emitting script writes both in one pass, so a mismatch
 * means the file was regenerated (or edited) without its manifest entry, and the
 * figure goes dark rather than presenting unsigned bytes as the shipped result.
 *
 * ONE ASYMMETRY, disclosed rather than hidden: the ladder's caption also needs the
 * paper's own fold-to-fold SDs and the paper's strongest LITERAL arm, both of
 * which live in the sibling `paper_literal_vs_llms.json`. That sibling carries NO
 * `output_sha256` entry in the run manifest (verified — only the four newer
 * outputs do), so it cannot be byte-pinned the way the ladder is. It is instead
 * shape-validated on read and CONTENT-cross-checked: `beliefLadderReferents`
 * throws unless the delta it derives from that file lands inside the range the
 * ladder itself shipped. Its digest is not claimed anywhere.
 *
 * Both the validator and the referent derivation sit inside one try/catch, so any
 * drift in either file returns `unavailable` rather than a degraded panel.
 *
 * Strictly read-only: no data file is ever written or mutated.
 */

import { createHash } from 'node:crypto';
import { existsSync, readFileSync, statSync } from 'node:fs';
import { join, relative, resolve } from 'node:path';

import { DATA_DIR } from '$lib/data/runs';
import {
	BELIEF_LADDER_VS_LLMS_BASENAME,
	beliefLadderReferents,
	validateBeliefLadder,
	type BeliefLadderLoad
} from '$lib/data/paper-belief-ladder';

const MODEL_DIR = join(DATA_DIR, 'results', 'indra_paper_literal_models_20260724');
const ARTIFACT_NAME = 'belief_model_ladder.json';
const ARTIFACT_PATH = join(MODEL_DIR, ARTIFACT_NAME);
const MANIFEST_PATH = join(MODEL_DIR, 'manifest.json');
const VS_LLMS_PATH = join(MODEL_DIR, BELIEF_LADDER_VS_LLMS_BASENAME);

/** The ladder is ~19 KB, the manifest ~7 KB, the sibling ~8 KB; 1 MB is headroom. */
const MAX_ARTIFACT_BYTES = 1024 * 1024;

type GuardedRead = { ok: true; bytes: Buffer } | { ok: false; reason: string };

function displayPath(path: string): string {
	const repoRoot = resolve(DATA_DIR, '..');
	const rel = relative(repoRoot, path).replaceAll('\\', '/');
	return rel && !rel.startsWith('../') ? rel : path;
}

function unavailable(reason: string, digest: string | null = null): BeliefLadderLoad {
	return {
		status: 'unavailable',
		reason,
		artifact_path: displayPath(ARTIFACT_PATH),
		artifact_sha256: digest,
		ladder: null,
		referents: null
	};
}

/** existsSync + statSync size cap + readFileSync, mirroring the sibling loaders. */
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

/** Load the belief-model ladder, gating to `unavailable` on any drift. Never throws. */
export function loadBeliefLadder(): BeliefLadderLoad {
	const artifact = readGuarded(ARTIFACT_PATH);
	if (!artifact.ok) return unavailable(artifact.reason);
	const digest = createHash('sha256').update(artifact.bytes).digest('hex');

	let raw: unknown;
	try {
		raw = JSON.parse(artifact.bytes.toString('utf8'));
	} catch {
		return unavailable(`${displayPath(ARTIFACT_PATH)} is not valid JSON.`, digest);
	}

	// Manifest parity: the bytes drawn must be the bytes the run signed.
	const manifest = readGuarded(MANIFEST_PATH);
	if (!manifest.ok) return unavailable(manifest.reason, digest);
	let manifestRaw: unknown;
	try {
		manifestRaw = JSON.parse(manifest.bytes.toString('utf8'));
	} catch {
		return unavailable(`${displayPath(MANIFEST_PATH)} is not valid JSON.`, digest);
	}
	const shaTable =
		manifestRaw !== null && typeof manifestRaw === 'object' && !Array.isArray(manifestRaw)
			? (manifestRaw as Record<string, unknown>).output_sha256
			: null;
	const recorded =
		shaTable !== null && typeof shaTable === 'object' && !Array.isArray(shaTable)
			? (shaTable as Record<string, unknown>)[ARTIFACT_NAME]
			: undefined;
	if (typeof recorded !== 'string' || recorded.length === 0) {
		return unavailable(
			`${displayPath(MANIFEST_PATH)} records no output_sha256 for ${ARTIFACT_NAME}.`,
			digest
		);
	}
	if (recorded !== digest) {
		return unavailable(
			`${displayPath(ARTIFACT_PATH)} does not match the sha256 the run manifest records.`,
			digest
		);
	}

	// The sibling head-to-head artifact: not byte-pinned (no manifest sha entry),
	// so it is shape-validated here and content-cross-checked below.
	const vsLlms = readGuarded(VS_LLMS_PATH);
	if (!vsLlms.ok) return unavailable(vsLlms.reason, digest);
	let vsLlmsRaw: unknown;
	try {
		vsLlmsRaw = JSON.parse(vsLlms.bytes.toString('utf8'));
	} catch {
		return unavailable(`${displayPath(VS_LLMS_PATH)} is not valid JSON.`, digest);
	}
	const pointMetrics =
		vsLlmsRaw !== null && typeof vsLlmsRaw === 'object' && !Array.isArray(vsLlmsRaw)
			? (vsLlmsRaw as Record<string, unknown>).point_metrics
			: undefined;

	try {
		const ladder = validateBeliefLadder(raw);
		return {
			status: 'ok',
			reason: null,
			artifact_path: displayPath(ARTIFACT_PATH),
			artifact_sha256: digest,
			ladder,
			referents: beliefLadderReferents(ladder, pointMetrics)
		};
	} catch (error) {
		return unavailable(
			`${displayPath(ARTIFACT_PATH)} failed its belief-ladder contract: ${String(error)}`,
			digest
		);
	}
}
