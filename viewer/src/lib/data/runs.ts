/**
 * Run discovery + registry. Replaces the DuckDB `score_run` table: a run is an
 * export directory under `data/exports/` containing an `export_meta.json`.
 */
import { readdirSync, readFileSync, existsSync, statSync } from 'node:fs';
import { join, resolve } from 'node:path';
import type { RunMeta } from './types';

/** Repo `data/` dir, resolved relative to the viewer's cwd (viewer/ → ../data). */
export const DATA_DIR = resolve(process.cwd(), '..', 'data');
const EXPORTS_DIR = join(DATA_DIR, 'exports');

/** Read the raw run's `.meta.json` (status + timing) when it sits next to the
 * source JSONL. Best-effort: returns nulls if absent. */
function readSourceMeta(sourceRun: string | null): Pick<RunMeta, 'status' | 'started_at' | 'finished_at'> {
	const empty = { status: null, started_at: null, finished_at: null };
	if (!sourceRun) return empty;
	const metaPath = resolve(DATA_DIR, '..', sourceRun).replace(/\.jsonl$/, '.meta.json');
	if (!existsSync(metaPath)) return empty;
	try {
		const m = JSON.parse(readFileSync(metaPath, 'utf8'));
		return {
			status: m.status ?? null,
			started_at: m.started_or_resumed_at ?? m.started_at ?? null,
			finished_at: m.completed_at ?? m.stopped_at ?? null
		};
	} catch {
		return empty;
	}
}

function toRunMeta(dir: string, m: Record<string, unknown>): RunMeta {
	const generatedFrom = (m.generated_from ?? {}) as { run?: string };
	const sourceRun = generatedFrom.run ?? null;
	return {
		run_id: String(m.run_id ?? ''),
		export_dir: dir,
		model: String(m.model ?? 'unknown'),
		generated_date: (m.generated_date as string) ?? null,
		counts: (m.counts as RunMeta['counts']) ?? {},
		bucket_counts: (m.bucket_counts as Record<string, number>) ?? {},
		source_run: sourceRun,
		...readSourceMeta(sourceRun)
	};
}

let _cache: { mtimeKey: string; runs: RunMeta[] } | null = null;

/** All runs, newest first. Cached until an export_meta.json mtime changes. */
export function listRuns(): RunMeta[] {
	if (!existsSync(EXPORTS_DIR)) return [];
	const dirs = readdirSync(EXPORTS_DIR, { withFileTypes: true })
		.filter((d) => d.isDirectory())
		.map((d) => join(EXPORTS_DIR, d.name))
		.filter((dir) => existsSync(join(dir, 'export_meta.json')));

	// Invalidate the registry when any export_meta.json changes.
	const mtimeKey = dirs
		.map((dir) => `${dir}:${statSync(join(dir, 'export_meta.json')).mtimeMs}`)
		.join('|');
	if (_cache && _cache.mtimeKey === mtimeKey) return _cache.runs;

	const runs: RunMeta[] = [];
	for (const dir of dirs) {
		try {
			const m = JSON.parse(readFileSync(join(dir, 'export_meta.json'), 'utf8'));
			if (m.run_id) runs.push(toRunMeta(dir, m));
		} catch {
			// skip malformed exports
		}
	}
	runs.sort((a, b) => (b.generated_date ?? '').localeCompare(a.generated_date ?? ''));
	_cache = { mtimeKey, runs };
	return runs;
}

/** The default/latest run, or null if no exports exist. */
export function latestRun(): RunMeta | null {
	return listRuns()[0] ?? null;
}

/** Resolve a run by id (full or 8-char prefix), falling back to latest. */
export function resolveRun(runId?: string | null): RunMeta | null {
	const runs = listRuns();
	if (!runId) return runs[0] ?? null;
	return runs.find((r) => r.run_id === runId || r.run_id.startsWith(runId)) ?? null;
}
