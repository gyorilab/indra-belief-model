/**
 * Server-only loader for the curator review-queue figure (/paper beat 2).
 *
 * Mirrors `loadPaperLiteral` / `loadPaperMethodLandscape`: DATA_DIR join, size
 * cap, `unavailable()` status object, sha256 via node:crypto, `displayPath`, and
 * a fail-closed validator. Unlike the head-to-head loader it reads NOTHING but
 * the one artifact and the run manifest — every count the figure draws is
 * already in `statement_review_queue.json`, so there is no join to re-do here
 * and no way for the figure to disagree with the file it cites.
 *
 * It additionally pins the artifact to the run manifest's `output_sha256`. The
 * emitting script writes both in one pass, so a mismatch means the file was
 * regenerated (or edited) without its manifest entry — the figure goes dark
 * rather than presenting unsigned bytes as the shipped result.
 *
 * Strictly read-only: no data file is ever written or mutated.
 */

import { createHash } from 'node:crypto';
import { existsSync, readFileSync, statSync } from 'node:fs';
import { join, relative, resolve } from 'node:path';

import { DATA_DIR } from '$lib/data/runs';
import { validateReviewQueue, type ReviewQueueLoad } from '$lib/data/paper-review-queue';

const MODEL_DIR = join(DATA_DIR, 'results', 'indra_paper_literal_models_20260724');
const ARTIFACT_NAME = 'statement_review_queue.json';
const ARTIFACT_PATH = join(MODEL_DIR, ARTIFACT_NAME);
const MANIFEST_PATH = join(MODEL_DIR, 'manifest.json');

/** The artifact is ~39 KB (the budget sweep is most of it); 1 MB is headroom. */
const MAX_ARTIFACT_BYTES = 1024 * 1024;

type GuardedRead = { ok: true; bytes: Buffer } | { ok: false; reason: string };

function displayPath(path: string): string {
	const repoRoot = resolve(DATA_DIR, '..');
	const rel = relative(repoRoot, path).replaceAll('\\', '/');
	return rel && !rel.startsWith('../') ? rel : path;
}

function unavailable(reason: string, digest: string | null = null): ReviewQueueLoad {
	return {
		status: 'unavailable',
		reason,
		artifact_path: displayPath(ARTIFACT_PATH),
		artifact_sha256: digest,
		queue: null
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

/** Load the review queue, gating to `unavailable` on any drift. Never throws. */
export function loadReviewQueue(): ReviewQueueLoad {
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

	try {
		return {
			status: 'ok',
			reason: null,
			artifact_path: displayPath(ARTIFACT_PATH),
			artifact_sha256: digest,
			queue: validateReviewQueue(raw)
		};
	} catch (error) {
		return unavailable(
			`${displayPath(ARTIFACT_PATH)} failed its review-queue contract: ${String(error)}`,
			digest
		);
	}
}
