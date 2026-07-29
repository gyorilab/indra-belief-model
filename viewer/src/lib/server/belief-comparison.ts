import { createHash } from 'node:crypto';
import { existsSync, readFileSync, statSync } from 'node:fs';
import { isAbsolute, join, relative, resolve } from 'node:path';

import { DATA_DIR } from '$lib/data/runs';
import {
	validateBeliefComparisonArtifact,
	type BeliefArtifactValidation
} from '$lib/data/belief-comparison';

const DEFAULT_FILENAME = 'indra_belief_comparison_metrics.json';
const MAX_ARTIFACT_BYTES = 16 * 1024 * 1024;

export type BeliefComparisonLoad = BeliefArtifactValidation & {
	artifact_path: string;
	artifact_sha256: string | null;
};

function configuredPath(): string {
	const configured = process.env.INDRA_BELIEF_COMPARISON_METRICS;
	if (!configured) return join(DATA_DIR, 'results', DEFAULT_FILENAME);
	return isAbsolute(configured) ? configured : resolve(process.cwd(), configured);
}

function displayPath(path: string): string {
	const repoRoot = resolve(DATA_DIR, '..');
	const rel = relative(repoRoot, path).replaceAll('\\', '/');
	return rel && !rel.startsWith('../') ? rel : path;
}

function unavailable(path: string, reason: string, digest: string | null = null): BeliefComparisonLoad {
	return {
		status: 'unavailable',
		frozen_at: null,
		provenance: null,
		panels: [],
		reasons: [reason],
		artifact_path: displayPath(path),
		artifact_sha256: digest
	};
}

/** Load the optional comparison artifact. Missing, oversized, malformed, or
 * contract-incompatible data gates the complete two-panel result. */
export function loadBeliefComparison(): BeliefComparisonLoad {
	const path = configuredPath();
	if (!existsSync(path)) {
		return unavailable(path, 'Statement-belief metrics artifact is not present.');
	}
	let bytes: Buffer;
	try {
		const size = statSync(path).size;
		if (size > MAX_ARTIFACT_BYTES) {
			return unavailable(path, `Metrics artifact exceeds the ${MAX_ARTIFACT_BYTES}-byte safety limit.`);
		}
		bytes = readFileSync(path);
	} catch (error) {
		return unavailable(path, `Metrics artifact could not be read: ${String(error)}`);
	}
	const digest = createHash('sha256').update(bytes).digest('hex');
	let raw: unknown;
	try {
		raw = JSON.parse(bytes.toString('utf8'));
	} catch {
		return unavailable(path, 'Metrics artifact is not valid JSON.', digest);
	}
	const validation = validateBeliefComparisonArtifact(raw);
	return {
		...validation,
		artifact_path: displayPath(path),
		artifact_sha256: digest
	};
}
