/**
 * The in-memory store: loads + indexes a run's export files, cached by file
 * mtime.
 *
 * per_statement.json (~6 MB) is parsed eagerly; per_evidence.jsonl (~130 MB,
 * 47k rows) is indexed lazily on first detail/cohort request and cached.
 */
import { readFileSync, statSync, existsSync } from 'node:fs';
import { join } from 'node:path';
import { DATA_DIR } from './runs';
import {
	buildCurationIndex,
	EMPTY_CURATION_INDEX,
	curationKey,
	goldForRow,
	type CurationIndex
} from './curation';
import type { RunMeta, StatementRollup, EvidenceRow, CurationRow } from './types';

// Re-export the curation domain surface so existing call sites can keep importing
// from './store' (the IO entry point) without reaching into ./curation directly.
export { curationKey, goldForRow, type CurationIndex };

export interface RunData {
	meta: RunMeta;
	perStatement: StatementRollup[];
	byHash: Map<string, StatementRollup>;
}

interface EvidenceIndex {
	byStmt: Map<string, EvidenceRow[]>;
	all: EvidenceRow[];
}

const _runCache = new Map<string, { mtime: number; data: RunData }>();
const _evCache = new Map<string, { mtime: number; index: EvidenceIndex }>();

function statementsPath(meta: RunMeta): string {
	return join(meta.export_dir, 'per_statement.json');
}
function evidencePath(meta: RunMeta): string {
	return join(meta.export_dir, 'per_evidence.jsonl');
}

/** Load (and cache) a run's per-statement rollups + hash index. */
export function getRunData(meta: RunMeta): RunData {
	const path = statementsPath(meta);
	const mtime = statSync(path).mtimeMs;
	const cached = _runCache.get(meta.run_id);
	if (cached && cached.mtime === mtime) return cached.data;

	const perStatement = JSON.parse(readFileSync(path, 'utf8')) as StatementRollup[];
	const byHash = new Map(perStatement.map((s) => [s.stmt_hash, s]));
	const data: RunData = { meta, perStatement, byHash };
	_runCache.set(meta.run_id, { mtime, data });
	return data;
}

/** Load (and cache) the per-evidence index for a run. ~1–2s on first call. */
export function getEvidenceIndex(meta: RunMeta): EvidenceIndex {
	const path = evidencePath(meta);
	if (!existsSync(path)) return { byStmt: new Map(), all: [] };
	const mtime = statSync(path).mtimeMs;
	const cached = _evCache.get(meta.run_id);
	if (cached && cached.mtime === mtime) return cached.index;

	const all: EvidenceRow[] = [];
	const byStmt = new Map<string, EvidenceRow[]>();
	const text = readFileSync(path, 'utf8');
	let nl = 0;
	while (nl < text.length) {
		let end = text.indexOf('\n', nl);
		if (end === -1) end = text.length;
		if (end > nl) {
			try {
				const row = JSON.parse(text.slice(nl, end)) as EvidenceRow;
				all.push(row);
				let arr = byStmt.get(row.stmt_hash);
				if (!arr) {
					arr = [];
					byStmt.set(row.stmt_hash, arr);
				}
				arr.push(row);
			} catch {
				// skip malformed line
			}
		}
		nl = end + 1;
	}
	const index: EvidenceIndex = { byStmt, all };
	_evCache.set(meta.run_id, { mtime, index });
	return index;
}

/** Evidence rows for one statement (ordered as in the export). */
export function evidenceForStatement(meta: RunMeta, stmtHash: string): EvidenceRow[] {
	return getEvidenceIndex(meta).byStmt.get(stmtHash) ?? [];
}

// ── Curation gold index (corpus-scoped, not per-run) ────────────────────────
//
// INDRA curations are human correctness labels keyed on INDRA (matches_hash,
// source_hash) ints — the SAME for every scoring run (gold belongs to the
// corpus, not a pass). store.ts owns ONLY the IO + mtime cache here; the domain
// (gold rule, join key, index reduction, evidence->gold lookup) lives in
// ./curation, the cross-language twin of src/indra_belief/curation.py.

const CURATIONS_PATH = join(DATA_DIR, 'benchmark', 'rasmachine_curations.jsonl');

let _curCache: { mtime: number; index: CurationIndex } | null = null;

/** Load (and cache) the global curation index. Returns the empty index
 *  (present=false) when the file has not been pulled — never throws, so gold
 *  surfaces can render an honest "no curations pulled" state. Pure domain
 *  (parse->index, gold rule) is delegated to buildCurationIndex. */
export function getCurationIndex(): CurationIndex {
	if (!existsSync(CURATIONS_PATH)) {
		_curCache = null;
		return EMPTY_CURATION_INDEX;
	}
	const mtime = statSync(CURATIONS_PATH).mtimeMs;
	if (_curCache && _curCache.mtime === mtime) return _curCache.index;

	const rows: CurationRow[] = [];
	for (const line of readFileSync(CURATIONS_PATH, 'utf8').split('\n')) {
		if (!line.trim()) continue;
		try {
			rows.push(JSON.parse(line) as CurationRow);
		} catch {
			/* skip malformed line */
		}
	}
	const index = buildCurationIndex(rows);
	_curCache = { mtime, index };
	return index;
}
