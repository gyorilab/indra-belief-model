/**
 * The curatable statement datasets — the universes /curate can draw a random
 * (statement, evidence) sample from. Server-only ($lib/server).
 *
 * Two KINDS of universe, sampled differently:
 *  - 'cogex-pool'        a JSONL of {matches_hash, source_hash, …} lines; a uniform
 *                        random line is materialized UNBIASED by matches_hash via
 *                        /statements/from_hashes. This is the representative draw.
 *  - 'inline-statements' a PREPROCESSED JSONL (one INDRA statement per line, evidence
 *                        inline) with every load-bearing hash quoted as a STRING; pick
 *                        a random statement × random evidence — no network, no fetch.
 *                        Built by scripts/build_curate_pool.py (Python reads the raw
 *                        64-bit source_hash ints exactly; the viewer's JSON.parse would
 *                        round a bare int and curate the wrong evidence).
 *
 * `character` + `sizeN`/`sizeLabel` drive the UI's distinction between an
 * evidence-row reservoir representing a large source population and a bounded
 * named corpus. Add a dataset = one entry here; the selector scales to N.
 */
import { existsSync } from 'node:fs';
import { resolve } from 'node:path';

export type DatasetKind = 'cogex-pool' | 'inline-statements';

export interface Dataset {
	id: string;
	label: string;
	kind: DatasetKind;
	/** one-line character blurb shown under the name in the selector. */
	blurb: string;
	/** human size (e.g. '44.9M', '47.4k') and its magnitude (drives the scale bar). */
	sizeLabel: string;
	sizeN: number;
	/** 'representative' = evidence-row reservoir; 'bounded' = a fixed named set. */
	character: 'representative' | 'bounded';
	/** path under the viewer's DATA_DIR (../data). */
	file: string;
	/** Operator-facing instruction when a generated local frame is absent. */
	provisioning?: string;
}

export const DATASETS: Dataset[] = [
	{
		id: 'representative',
		label: 'representative',
		kind: 'cogex-pool',
		blurb: '5k evidence-row reservoir from 44.94M grounded/assembled CoGEx rows',
		sizeLabel: '44.94M',
		sizeN: 44_944_056,
		character: 'representative',
		// This tracked exact-key manifest is the runtime frame. The ignored 4.4 MB
		// materialization contains extra evidence text but is not required because
		// resolverFor fetches the statement/evidence from INDRA by exact hashes.
		file: 'benchmark/cogex_representative_pool_manifest.jsonl'
	},
	{
		id: 'rasmachine',
		label: 'rasmachine',
		kind: 'inline-statements',
		blurb: 'a fixed benchmark corpus · 8,724 statements',
		sizeLabel: '47.4k',
		sizeN: 47_434,
		character: 'bounded',
		file: 'corpora/rasmachine_curate_pool.jsonl',
		provisioning: 'run PYTHONPATH=src python scripts/build_curate_pool.py from the repository root'
	}
];

/** Resolve a dataset by id, defaulting to the representative draw. */
export function getDataset(id: string | null | undefined): Dataset {
	return DATASETS.find((d) => d.id === id) ?? DATASETS[0];
}

/** The selector needs only the safe, non-path fields client-side. */
export function datasetsForClient() {
	return DATASETS.map(({ id, label, blurb, sizeLabel, sizeN, character, file, provisioning }) => {
		const available = existsSync(resolve(process.cwd(), '..', 'data', file));
		return {
			id,
			label,
			blurb,
			sizeLabel,
			sizeN,
			character,
			available,
			unavailableReason: available ? null : (provisioning ?? 'dataset frame is not provisioned')
		};
	});
}
