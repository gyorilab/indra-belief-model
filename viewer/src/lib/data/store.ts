/**
 * The in-memory store: loads + indexes a run's export files, cached by file
 * mtime (mirrors db.ts's old stat-based invalidation). This is the entire
 * replacement for the DuckDB connection manager.
 *
 * per_statement.json (~6 MB) is parsed eagerly; per_evidence.jsonl (~130 MB,
 * 47k rows) is indexed lazily on first detail/cohort request and cached.
 */
import { readFileSync, statSync, existsSync } from 'node:fs';
import { join } from 'node:path';
import type { RunMeta, StatementRollup, EvidenceRow } from './types';

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
