/** Lossless curation-history helpers for the server-side /curate sampler.
 *
 * INDRA statement/evidence hashes are signed 64-bit integers.  The list-all API
 * currently serializes them as bare JSON numbers, so ordinary JSON.parse would
 * silently round them in JavaScript.  We quote only bare pa_hash/source_hash
 * values before parsing and keep exact-pair identities as digit strings.
 */

const HASH_RE = /^-?\d+$/;

export interface CurationHistoryRow {
	curator: string;
	tag: string;
	paHash: string | null;
	sourceHash: string | null;
}

export interface PoolPair {
	stmtHash: string;
	sourceHash: string;
}

function hashString(value: unknown): string | null {
	if (typeof value !== 'string') return null;
	return HASH_RE.test(value) ? value : null;
}

/** Parse keyed /curation/list JSON without losing signed 64-bit hash digits.
 *
 * The negative lookbehind avoids touching an escaped `\"source_hash\"` inside a
 * curator note.  Quoting nested real hash fields is harmless; only the top-level
 * pa_hash/source_hash values are read below.
 */
export function parseCurationHistory(text: string): CurationHistoryRow[] {
	const lossless = text.replace(
		/(?<!\\)("(?:pa_hash|source_hash)"\s*:\s*)(-?\d+)(?=\s*[,}])/g,
		'$1"$2"'
	);
	const payload: unknown = JSON.parse(lossless);
	if (!Array.isArray(payload)) throw new Error('unexpected curation counts payload');
	return payload
		.filter((row): row is Record<string, unknown> => row != null && typeof row === 'object')
		.map((row) => ({
			curator: typeof row.curator === 'string' ? row.curator : '',
			tag: typeof row.tag === 'string' ? row.tag : '',
			paHash: hashString(row.pa_hash),
			sourceHash: hashString(row.source_hash)
		}));
}

/** Exact pair key shared by API history and dataset pool rows. */
export function exactPairKey(
	stmtHash: string | null | undefined,
	sourceHash: string | null | undefined
): string | null {
	if (!stmtHash || !sourceHash || !HASH_RE.test(stmtHash) || !HASH_RE.test(sourceHash)) return null;
	return `${stmtHash}:${sourceHash}`;
}

/** The exact pairs previously curated by one authenticated account. */
export function curatorPairKeys(rows: CurationHistoryRow[], email: string): Set<string> {
	const wanted = email.trim().toLowerCase();
	const keys = new Set<string>();
	for (const row of rows) {
		if (row.curator.trim().toLowerCase() !== wanted) continue;
		const key = exactPairKey(row.paHash, row.sourceHash);
		if (key) keys.add(key);
	}
	return keys;
}

/** Extract a representative-pool line's exact pair without JSON number parsing. */
export function poolPairOf(line: string): PoolPair | null {
	const exactField = (name: string): string | null => {
		const match = line.match(
			new RegExp(`"${name}"\\s*:\\s*(?:"(-?\\d+)"|(-?\\d+))(?=\\s*[,}])`)
		);
		return match?.[1] ?? match?.[2] ?? null;
	};
	// The ignored materialization calls this `stmt_hash`; the tracked clean-
	// checkout manifest calls it `matches_hash`. Both identify the same INDRA
	// statement and are parsed losslessly here.
	const sm = exactField('stmt_hash') ?? exactField('matches_hash');
	const sh = exactField('source_hash');
	if (!sm || !sh) return null;
	const stmtHash = sm;
	const sourceHash = sh;
	if (!HASH_RE.test(stmtHash) || !HASH_RE.test(sourceHash)) return null;
	return { stmtHash, sourceHash };
}

/** A reservoir row retained for provenance but blocked from new curation. */
export function poolLineExcluded(line: string): boolean {
	return /"excluded_from_curation"\s*:\s*true(?=\s*[,}])/.test(line);
}

/** Valid, unique representative-pool rows this curator has not already done. */
export function unseenPoolLines(lines: string[], curatedKeys: ReadonlySet<string>): string[] {
	const emitted = new Set<string>();
	const unseen: string[] = [];
	for (const line of lines) {
		if (poolLineExcluded(line)) continue;
		const pair = poolPairOf(line);
		if (!pair) continue;
		const key = exactPairKey(pair.stmtHash, pair.sourceHash);
		if (!key || curatedKeys.has(key) || emitted.has(key)) continue;
		emitted.add(key);
		unseen.push(line);
	}
	return unseen;
}
