import { createHash } from 'node:crypto';
import { existsSync, readFileSync, statSync } from 'node:fs';
import { join, relative, resolve } from 'node:path';

import { DATA_DIR } from '$lib/data/runs';
import {
	validatePaperMethodLandscape,
	type PaperMethodLandscapeLoad
} from '$lib/data/paper-method-landscape';

const ARTIFACT_PATH = join(DATA_DIR, 'benchmark', 'indra_paper_2023_published_method_metrics.json');
const MAX_ARTIFACT_BYTES = 1024 * 1024;

function displayPath(path: string): string {
	const repoRoot = resolve(DATA_DIR, '..');
	const rel = relative(repoRoot, path).replaceAll('\\', '/');
	return rel && !rel.startsWith('../') ? rel : path;
}

function unavailable(path: string, reason: string, digest: string | null = null): PaperMethodLandscapeLoad {
	return {
		status: 'unavailable',
		landscape: null,
		reason,
		artifact_path: displayPath(path),
		artifact_sha256: digest
	};
}

/** Load the immutable published-method reference as a separate display layer. */
export function loadPaperMethodLandscape(): PaperMethodLandscapeLoad {
	if (!existsSync(ARTIFACT_PATH)) return unavailable(ARTIFACT_PATH, 'Pinned 2023 paper-method artifact is missing.');
	let bytes: Buffer;
	try {
		if (statSync(ARTIFACT_PATH).size > MAX_ARTIFACT_BYTES) {
			return unavailable(ARTIFACT_PATH, `Paper-method artifact exceeds ${MAX_ARTIFACT_BYTES} bytes.`);
		}
		bytes = readFileSync(ARTIFACT_PATH);
	} catch (error) {
		return unavailable(ARTIFACT_PATH, `Paper-method artifact could not be read: ${String(error)}`);
	}
	const digest = createHash('sha256').update(bytes).digest('hex');
	let raw: unknown;
	try {
		raw = JSON.parse(bytes.toString('utf8'));
	} catch {
		return unavailable(ARTIFACT_PATH, 'Paper-method artifact is not valid JSON.', digest);
	}
	try {
		return {
			status: 'available',
			landscape: validatePaperMethodLandscape(raw, digest),
			reason: null,
			artifact_path: displayPath(ARTIFACT_PATH),
			artifact_sha256: digest
		};
	} catch (error) {
		return unavailable(
			ARTIFACT_PATH,
			`Paper-method artifact failed its unpaired-reference contract: ${String(error)}`,
			digest
		);
	}
}
