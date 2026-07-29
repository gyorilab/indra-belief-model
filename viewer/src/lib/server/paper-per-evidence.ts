/**
 * Server-only loader for the PER-EVIDENCE grain surface.
 *
 * Mirrors `loadPaperRobustness` / `loadPaperTieInflation`: DATA_DIR join, size
 * cap, sha256 via node:crypto, an `unavailable()` status object rather than a
 * throw, and `displayPath` for the on-screen artifact path.
 *
 * It reads ONE file: the shipped per-evidence comparison artifact. Every number
 * the figure draws is READ off it — no AUROC, interval, census count, chi-square
 * or reconciliation residual is recomputed here. The artifact is produced by
 * `scripts/compute_per_evidence_comparison.py`, which asserts (a) its closed-form
 * AUROC/AP equal sklearn's on the full sample, and (b) the per-evidence verdicts
 * it recovered rebuild the SHIPPED statement probabilities exactly.
 * `validatePaperPerEvidence` gates on that second reconciliation, so a drifted
 * rerun takes the figure down instead of drawing a connector between two grains
 * that are no longer the same run.
 *
 * Strictly read-only: no data file is ever written or mutated.
 */

import { createHash } from 'node:crypto';
import { existsSync, readFileSync, statSync } from 'node:fs';
import { join, relative, resolve } from 'node:path';

import { DATA_DIR } from '$lib/data/runs';
import {
	validatePaperPerEvidence,
	type PaperPerEvidenceLoad
} from '$lib/data/paper-per-evidence';

const ARTIFACT_DIR = join(DATA_DIR, 'results', 'per_evidence_comparison_20260727');
const ARTIFACT_NAME = 'per_evidence_comparison.json';
const ARTIFACT_PATH = join(ARTIFACT_DIR, ARTIFACT_NAME);

/** The artifact is ~120 KB; anything near this cap means the file changed kind,
 *  which is a reason to gate rather than parse. */
const MAX_ARTIFACT_BYTES = 8 * 1024 * 1024;

function displayPath(path: string): string {
	const repoRoot = resolve(DATA_DIR, '..');
	const rel = relative(repoRoot, path).replaceAll('\\', '/');
	return rel && !rel.startsWith('../') ? rel : path;
}

function unavailable(reason: string, digest: string | null = null): PaperPerEvidenceLoad {
	return {
		status: 'unavailable',
		figure: null,
		reason,
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

export function loadPaperPerEvidence(): PaperPerEvidenceLoad {
	const read = readGuarded(ARTIFACT_PATH);
	if (!read.ok) return unavailable(read.reason);

	const digest = createHash('sha256').update(read.bytes).digest('hex');
	let parsed: unknown;
	try {
		parsed = JSON.parse(read.bytes.toString('utf8'));
	} catch (error) {
		return unavailable(`${displayPath(ARTIFACT_PATH)} is not valid JSON: ${String(error)}`, digest);
	}

	return validatePaperPerEvidence(parsed, {
		artifactPath: displayPath(ARTIFACT_PATH),
		artifactSha256: digest
	});
}
