/**
 * Server-only loader for the margin-robustness surface.
 *
 * Mirrors `loadFramingCorrection` / `loadPaperTieInflation`: DATA_DIR join, size
 * cap, sha256 via node:crypto, an `unavailable()` status object rather than a
 * throw, and `displayPath` for the on-screen artifact path.
 *
 * It reads TWO files and nothing else: the shipped robustness artifact, and the
 * run manifest it must be signed in. Every number the figure draws is READ off
 * the artifact — no delta, interval, band bound or census count is recomputed
 * here. The artifact is produced by `scripts/compute_paper_robustness.py`, which
 * asserts that its own pointwise numbers reproduce the shipped head-to-head
 * exactly; `validatePaperRobustness` gates on that reconciliation residual, so a
 * drifted rerun takes the figure down instead of drawing a band around a margin
 * the page reports differently elsewhere.
 *
 * Strictly read-only: no data file is ever written or mutated.
 */

import { createHash } from 'node:crypto';
import { existsSync, readFileSync, statSync } from 'node:fs';
import { join, relative, resolve } from 'node:path';

import { DATA_DIR } from '$lib/data/runs';
import {
	validatePaperRobustness,
	type PaperRobustnessLoad
} from '$lib/data/paper-robustness';

const MODEL_DIR = join(DATA_DIR, 'results', 'indra_paper_literal_models_20260724');
const ARTIFACT_NAME = 'paper_margin_robustness.json';
const ARTIFACT_PATH = join(MODEL_DIR, ARTIFACT_NAME);
const MANIFEST_PATH = join(MODEL_DIR, 'manifest.json');

/** The artifact is ~13 KB and the manifest a few KB; anything near this cap
 *  means the file changed kind, which is a reason to gate rather than parse. */
const MAX_ARTIFACT_BYTES = 4 * 1024 * 1024;

type GuardedRead = { ok: true; bytes: Buffer } | { ok: false; reason: string };

function displayPath(path: string): string {
	const repoRoot = resolve(DATA_DIR, '..');
	const rel = relative(repoRoot, path).replaceAll('\\', '/');
	return rel && !rel.startsWith('../') ? rel : path;
}

function unavailable(reason: string, digest: string | null = null): PaperRobustnessLoad {
	return {
		status: 'unavailable',
		figure: null,
		reason,
		artifact_path: displayPath(ARTIFACT_PATH),
		artifact_sha256: digest
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

/** The sha256 the run manifest signed for one output, or null with a reason. */
function recordedSha(
	manifestRaw: unknown,
	name: string
): { ok: true; sha: string } | { ok: false; reason: string } {
	const shaTable =
		manifestRaw !== null && typeof manifestRaw === 'object' && !Array.isArray(manifestRaw)
			? (manifestRaw as Record<string, unknown>).output_sha256
			: null;
	const recorded =
		shaTable !== null && typeof shaTable === 'object' && !Array.isArray(shaTable)
			? (shaTable as Record<string, unknown>)[name]
			: undefined;
	if (typeof recorded !== 'string' || recorded.length === 0) {
		return {
			ok: false,
			reason: `${displayPath(MANIFEST_PATH)} records no output_sha256 for ${name}.`
		};
	}
	return { ok: true, sha: recorded };
}

/**
 * Load the robustness payload, gating to `unavailable` on any drift: a missing
 * or oversized file, invalid JSON, a manifest that did not sign these exact
 * bytes, or any validator invariant. Never throws.
 */
export function loadPaperRobustness(): PaperRobustnessLoad {
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
	const signed = recordedSha(manifestRaw, ARTIFACT_NAME);
	if (!signed.ok) return unavailable(signed.reason, digest);
	if (signed.sha !== digest) {
		return unavailable(
			`${displayPath(ARTIFACT_PATH)} is not the artifact the run manifest signed ` +
				`(${digest.slice(0, 12)}… on disk, ${signed.sha.slice(0, 12)}… recorded).`,
			digest
		);
	}

	return validatePaperRobustness(raw, {
		artifactPath: displayPath(ARTIFACT_PATH),
		artifactSha256: digest
	});
}
