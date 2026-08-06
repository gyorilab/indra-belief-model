/**
 * Server-only loader for the REASONING ABLATION surface.
 *
 * Mirrors `loadPaperPerEvidence` / `loadPaperTieInflation`: DATA_DIR join, size
 * cap, sha256 via node:crypto, an `unavailable()` status object rather than a
 * throw, and `displayPath` for the on-screen artifact path.
 *
 * It reads ONE file, produced by `scripts/compute_reasoning_ablation.py`. That
 * script will not write the file at all unless the deliberating side, recomputed
 * from its own raw readings, reproduces the SHIPPED statement probabilities and
 * the SHIPPED average precision / AUROC / Brier / calibration error / confusion
 * counts exactly. `validatePaperReasoningAblation` re-gates on the flags that
 * record those checks, so a hand-edited or drifted artifact takes the figure down
 * instead of drawing a comparison between two runs that are no longer the same.
 *
 * Strictly read-only: no data file is ever written or mutated.
 */

import { createHash } from 'node:crypto';
import { existsSync, readFileSync, statSync } from 'node:fs';
import { join, relative, resolve } from 'node:path';

import { DATA_DIR } from '$lib/data/runs';
import {
	validatePaperReasoningAblation,
	type ReasoningAblationLoad
} from '$lib/data/paper-reasoning-ablation';

const ARTIFACT_DIR = join(DATA_DIR, 'results', 'reasoning_ablation_20260805');
const ARTIFACT_NAME = 'reasoning_ablation.json';
const ARTIFACT_PATH = join(ARTIFACT_DIR, ARTIFACT_NAME);

/** The artifact is ~30 KB; anything near this cap means the file changed kind,
 *  which is a reason to gate rather than parse. */
const MAX_ARTIFACT_BYTES = 8 * 1024 * 1024;

function displayPath(path: string): string {
	const repoRoot = resolve(DATA_DIR, '..');
	const rel = relative(repoRoot, path).replaceAll('\\', '/');
	return rel && !rel.startsWith('../') ? rel : path;
}

function unavailable(reason: string, digest: string | null = null): ReasoningAblationLoad {
	return {
		status: 'unavailable',
		reason,
		figure: null,
		artifact_path: displayPath(ARTIFACT_PATH),
		artifact_sha256: digest
	};
}

/** existsSync + statSync size cap + readFileSync, mirroring the sibling loaders. */
function readGuarded(path: string): { ok: true; bytes: Buffer } | { ok: false; reason: string } {
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

export function loadReasoningAblation(): ReasoningAblationLoad {
	const read = readGuarded(ARTIFACT_PATH);
	if (!read.ok) return unavailable(read.reason);

	const digest = createHash('sha256').update(read.bytes).digest('hex');
	let parsed: unknown;
	try {
		parsed = JSON.parse(read.bytes.toString('utf8'));
	} catch (error) {
		return unavailable(`${displayPath(ARTIFACT_PATH)} is not valid JSON: ${String(error)}`, digest);
	}

	return validatePaperReasoningAblation(parsed, {
		artifactPath: displayPath(ARTIFACT_PATH),
		artifactSha256: digest
	});
}
