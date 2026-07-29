/**
 * Server-only loader for the framing-correction panel (/paper beat 2).
 *
 * Mirrors `loadReviewQueue` exactly: DATA_DIR join, size cap, `unavailable()`
 * status object, sha256 via node:crypto, `displayPath`, and a fail-closed
 * validator. It differs in one way only — it reads TWO artifacts from the same
 * run directory:
 *
 *   · `framing_correction.json`  legs (a) declaration, (b) subtractive, (c) the
 *     reachable-value check and its permutation floor;
 *   · `non_reading_control.json` leg (d), drawn straight from the file that
 *     emitted it rather than restated.
 *
 * Both are pinned to the run manifest's `output_sha256`, both are validated, and
 * either failing gates the WHOLE payload to a single `unavailable`. The panel's
 * argument is one argument; half of it is not a weaker version of it, so half is
 * never drawn. The reason names the file that failed.
 *
 * Strictly read-only: no data file is ever written or mutated.
 */

import { createHash } from 'node:crypto';
import { existsSync, readFileSync, statSync } from 'node:fs';
import { join, relative, resolve } from 'node:path';

import { DATA_DIR } from '$lib/data/runs';
import {
	crossCheckFramingAndControl,
	validateFramingCorrection,
	validateNonReadingControl,
	type FramingCorrectionLoad
} from '$lib/data/paper-framing-correction';

const MODEL_DIR = join(DATA_DIR, 'results', 'indra_paper_literal_models_20260724');
const ARTIFACT_NAME = 'framing_correction.json';
const ARTIFACT_PATH = join(MODEL_DIR, ARTIFACT_NAME);
const CONTROL_NAME = 'non_reading_control.json';
const CONTROL_PATH = join(MODEL_DIR, CONTROL_NAME);
const MANIFEST_PATH = join(MODEL_DIR, 'manifest.json');

/** The artifacts are ~18 KB and ~8 KB; 1 MB is generous headroom. */
const MAX_ARTIFACT_BYTES = 1024 * 1024;

type GuardedRead = { ok: true; bytes: Buffer } | { ok: false; reason: string };

function displayPath(path: string): string {
	const repoRoot = resolve(DATA_DIR, '..');
	const rel = relative(repoRoot, path).replaceAll('\\', '/');
	return rel && !rel.startsWith('../') ? rel : path;
}

function unavailable(
	reason: string,
	digest: string | null = null,
	controlDigest: string | null = null
): FramingCorrectionLoad {
	return {
		status: 'unavailable',
		reason,
		artifact_path: displayPath(ARTIFACT_PATH),
		artifact_sha256: digest,
		control_path: displayPath(CONTROL_PATH),
		control_sha256: controlDigest,
		framing: null,
		control: null
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
		return { ok: false, reason: `${displayPath(MANIFEST_PATH)} records no output_sha256 for ${name}.` };
	}
	return { ok: true, sha: recorded };
}

/**
 * Load both artifacts behind the framing correction, gating to `unavailable` on
 * any drift in either. Never throws.
 */
export function loadFramingCorrection(): FramingCorrectionLoad {
	const artifact = readGuarded(ARTIFACT_PATH);
	if (!artifact.ok) return unavailable(artifact.reason);
	const digest = createHash('sha256').update(artifact.bytes).digest('hex');

	const controlBytes = readGuarded(CONTROL_PATH);
	if (!controlBytes.ok) return unavailable(controlBytes.reason, digest);
	const controlDigest = createHash('sha256').update(controlBytes.bytes).digest('hex');

	let raw: unknown;
	try {
		raw = JSON.parse(artifact.bytes.toString('utf8'));
	} catch {
		return unavailable(`${displayPath(ARTIFACT_PATH)} is not valid JSON.`, digest, controlDigest);
	}
	let controlRaw: unknown;
	try {
		controlRaw = JSON.parse(controlBytes.bytes.toString('utf8'));
	} catch {
		return unavailable(`${displayPath(CONTROL_PATH)} is not valid JSON.`, digest, controlDigest);
	}

	// Manifest parity: the bytes drawn must be the bytes the run signed, for both.
	const manifest = readGuarded(MANIFEST_PATH);
	if (!manifest.ok) return unavailable(manifest.reason, digest, controlDigest);
	let manifestRaw: unknown;
	try {
		manifestRaw = JSON.parse(manifest.bytes.toString('utf8'));
	} catch {
		return unavailable(`${displayPath(MANIFEST_PATH)} is not valid JSON.`, digest, controlDigest);
	}
	for (const [name, path, actual] of [
		[ARTIFACT_NAME, ARTIFACT_PATH, digest],
		[CONTROL_NAME, CONTROL_PATH, controlDigest]
	] as const) {
		const signed = recordedSha(manifestRaw, name);
		if (!signed.ok) return unavailable(signed.reason, digest, controlDigest);
		if (signed.sha !== actual) {
			return unavailable(
				`${displayPath(path)} does not match the sha256 the run manifest records.`,
				digest,
				controlDigest
			);
		}
	}

	try {
		const framing = validateFramingCorrection(raw);
		const control = validateNonReadingControl(controlRaw);
		crossCheckFramingAndControl(framing, control);
		return {
			status: 'ok',
			reason: null,
			artifact_path: displayPath(ARTIFACT_PATH),
			artifact_sha256: digest,
			control_path: displayPath(CONTROL_PATH),
			control_sha256: controlDigest,
			framing,
			control
		};
	} catch (error) {
		return unavailable(
			`${displayPath(ARTIFACT_PATH)} and ${displayPath(CONTROL_PATH)} failed their framing-correction contract: ${String(error)}`,
			digest,
			controlDigest
		);
	}
}
