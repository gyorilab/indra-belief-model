/**
 * Server-only loader for "the paper's own metric, with our arms on it".
 *
 * Mirrors `loadPaperMethodLandscape` / `loadPaperLiteral`: DATA_DIR join, size
 * cap, `unavailable()` status object rather than a throw, and `displayPath` for
 * the on-screen artifact path.
 *
 * It reads NO prediction files. The published side comes from the already
 * checksum-pinned `loadPaperMethodLandscape()`; our side is threaded in from the
 * `PaperLiteralLoad` the page has already built, so the 6 MB of prediction
 * bundles behind it are read exactly once per request. The only extra file it
 * touches is the small run manifest, for the panel size — read fail-soft, in the
 * same shape as `parsePaperReproduction`, so a manifest change drops one caption
 * instead of taking the figure down.
 */

import { existsSync, readFileSync, statSync } from 'node:fs';
import { join, relative, resolve } from 'node:path';

import { DATA_DIR } from '$lib/data/runs';
import type { PaperLiteralLoad } from '$lib/data/paper-literal';
import {
	buildPaperOwnMetric,
	type PaperOwnMetricLoad
} from '$lib/data/paper-own-metric';
import { loadPaperMethodLandscape } from '$lib/server/paper-method-landscape';

const MANIFEST_PATH = join(
	DATA_DIR,
	'results',
	'indra_paper_literal_models_20260724',
	'manifest.json'
);
/** The manifest is a few KB; anything near this cap means the file changed kind. */
const MAX_MANIFEST_BYTES = 256 * 1024;

function displayPath(path: string): string {
	const repoRoot = resolve(DATA_DIR, '..');
	const rel = relative(repoRoot, path).replaceAll('\\', '/');
	return rel && !rel.startsWith('../') ? rel : path;
}

function unavailable(
	path: string,
	reason: string,
	digest: string | null = null
): PaperOwnMetricLoad {
	return { status: 'unavailable', figure: null, reason, artifact_path: path, artifact_sha256: digest };
}

/**
 * Panel size off the run manifest (`inputs.extended_curation_dataset_pickle
 * .n_rows`). Never throws: a drifted manifest yields null and the figure renders
 * without the statement count, matching `parsePaperReproduction`'s fail-soft
 * contract for the same file.
 */
function readPanelSize(): number | null {
	try {
		if (!existsSync(MANIFEST_PATH)) return null;
		if (statSync(MANIFEST_PATH).size > MAX_MANIFEST_BYTES) return null;
		const raw: unknown = JSON.parse(readFileSync(MANIFEST_PATH, 'utf8'));
		if (raw === null || typeof raw !== 'object') return null;
		const inputs = (raw as Record<string, unknown>).inputs;
		if (inputs === null || typeof inputs !== 'object') return null;
		const dataset = (inputs as Record<string, unknown>).extended_curation_dataset_pickle;
		if (dataset === null || typeof dataset !== 'object') return null;
		const rows = (dataset as Record<string, unknown>).n_rows;
		if (typeof rows !== 'number' || !Number.isInteger(rows) || rows < 1) return null;
		return rows;
	} catch {
		return null;
	}
}

/**
 * Build the co-plot payload. Takes the page's existing `PaperLiteralLoad` so the
 * head-to-head artifacts are not read twice; pass the very same value the page
 * puts on `data.paperLiteral`.
 */
export function loadPaperOwnMetric(literal: PaperLiteralLoad): PaperOwnMetricLoad {
	const reference = loadPaperMethodLandscape();
	if (reference.status === 'unavailable') {
		return unavailable(
			reference.artifact_path,
			`Published 2023 method metrics are unavailable: ${reference.reason}`,
			reference.artifact_sha256
		);
	}
	if (literal.status === 'unavailable') {
		return unavailable(
			literal.artifact_path || displayPath(MANIFEST_PATH),
			`The newly scored models are unavailable: ${literal.reason}`,
			literal.artifact_sha256
		);
	}
	try {
		return {
			status: 'ok',
			figure: buildPaperOwnMetric(reference.landscape, literal.arms, {
				nStatements: readPanelSize(),
				reproduction: literal.reproduction
			}),
			reason: null,
			artifact_path: reference.artifact_path,
			artifact_sha256: reference.artifact_sha256
		};
	} catch (error) {
		return unavailable(
			reference.artifact_path,
			`Published rows and newly scored models could not be placed on one axis: ${String(error)}`,
			reference.artifact_sha256
		);
	}
}
