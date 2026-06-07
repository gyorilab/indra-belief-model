/**
 * INDRA curations as gold — the viewer's canonical curation-domain module, and
 * the deliberate cross-language twin of `src/indra_belief/curation.py`. The gold
 * rule, the (matches_hash, source_hash) join + str->int coercion, the index
 * reduction, and the single evidence->gold lookup all live here ONCE. store.ts
 * owns caching/IO; queries.ts and adjudicate.ts consume `goldForRow`. Parity
 * with the Python module is enforced by tests/test_curation_parity.
 *
 * The gold rule: a curation tag is gold-correct iff it is exactly "correct";
 * every other tag means the reader's extraction is wrong. Multiple curations on
 * one evidence aggregate with any-incorrect-wins (one objection flips it).
 */
import type { CurationRow, GoldVerdict, EvidenceRow } from './types';

/** The one tag denoting a correct extraction. Everything else is "incorrect". */
export const CORRECT_TAG = 'correct';

/** The gold atom: a single curation tag is correct iff it is exactly "correct". */
export function isGoldCorrect(tag: string | null | undefined): boolean {
	return tag === CORRECT_TAG;
}

/** Aggregate one evidence's curation tags into a gold verdict, any-incorrect-wins.
 *  Returns null for an empty set (uncurated). */
export function aggregateGold(tags: string[]): 'correct' | 'incorrect' | null {
	if (tags.length === 0) return null;
	return tags.every(isGoldCorrect) ? 'correct' : 'incorrect';
}

/** Content-addressed join key. Curation hashes are ints; run-export
 *  `indra_matches_hash` is a string — coerce here so no call site repeats it.
 *  Returns null when either hash is missing or unparseable. */
export function curationKey(
	matchesHash: number | string | null | undefined,
	sourceHash: number | string | null | undefined
): string | null {
	if (matchesHash == null || sourceHash == null) return null;
	const mh = typeof matchesHash === 'string' ? parseInt(matchesHash, 10) : Number(matchesHash);
	const sh = typeof sourceHash === 'string' ? parseInt(sourceHash, 10) : Number(sourceHash);
	if (!Number.isFinite(mh) || !Number.isFinite(sh)) return null;
	return `${mh}|${sh}`;
}

/** The in-memory curation index: raw groups + derived gold verdicts, keyed by
 *  `${matches_hash}|${source_hash}`. Mirrors curation.py's CurationIndex. */
export interface CurationIndex {
	byKey: Map<string, CurationRow[]>;
	goldByKey: Map<string, GoldVerdict>;
	nStatements: number;
	nEvidences: number;
	/** true when the curations file was present at all (vs. not pulled). */
	present: boolean;
}

export const EMPTY_CURATION_INDEX: CurationIndex = {
	byKey: new Map(),
	goldByKey: new Map(),
	nStatements: 0,
	nEvidences: 0,
	present: false
};

/** Reduce parsed curation rows into a CurationIndex (raw groups + gold). The
 *  pure core — store.ts wraps it with file IO + mtime caching. */
export function buildCurationIndex(rows: Iterable<CurationRow>): CurationIndex {
	const byKey = new Map<string, CurationRow[]>();
	const statements = new Set<number>();
	for (const c of rows) {
		if (c._matches_hash == null || c.source_hash == null) continue;
		const key = curationKey(c._matches_hash, c.source_hash);
		if (key == null) continue;
		let arr = byKey.get(key);
		if (!arr) byKey.set(key, (arr = []));
		arr.push(c);
		statements.add(Number(c._matches_hash));
	}

	const goldByKey = new Map<string, GoldVerdict>();
	for (const [key, curs] of byKey) {
		const tags = curs.map((c) => c.tag);
		const verdict = aggregateGold(tags) ?? 'incorrect';
		goldByKey.set(key, {
			verdict,
			n: curs.length,
			tags,
			curators: [...new Set(curs.map((c) => c.curator).filter(Boolean))],
			notes: curs.map((c) => c.text).filter((t) => t && t.trim())
		});
	}

	return { byKey, goldByKey, nStatements: statements.size, nEvidences: byKey.size, present: true };
}

/** Look up the gold verdict for an evidence row by coercing its (string)
 *  indra_matches_hash + source_hash to the index key. The single canonical
 *  replacement for the coerce-and-lookup that was written three times across
 *  queries.ts (joinEvidence, evidenceSideBySide) and adjudicate.ts (goldFor). */
export function goldForRow(index: CurationIndex, row: EvidenceRow | null | undefined): GoldVerdict | null {
	if (!row) return null;
	// Per-run baked gold travels with the run and switches when you switch runs.
	// A baked run sets `gold` on every row (object = curated, null = uncurated);
	// only a legacy run leaves it `undefined`, and only then do we consult the
	// global index. This is what makes gold follow the selected run.
	if (row.gold !== undefined) return row.gold;
	const key = curationKey(row.indra_matches_hash, row.source_hash);
	if (key == null) return null;
	return index.goldByKey.get(key) ?? null;
}
