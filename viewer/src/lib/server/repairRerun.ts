import { randomUUID } from 'node:crypto';
import { existsSync, mkdirSync, readFileSync, realpathSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { DuckDBInstance, type DuckDBConnection } from '@duckdb/node-api';
import { closeInstance, connect, dbExists, dbPath } from '$lib/db';
import {
	REPAIR_RERUN_LINEAGE_DDL,
	REPAIR_RERUN_TERMINAL_STATUS_SQL,
	repairCandidateActiveRerunIntentExistsSql,
	repairCandidateAvailablePredicateSql,
	repairCandidateConsumedExistsSql,
	repairRerunChildRunIdSql,
	repairRerunParentCorrectionIdNumberSql,
	repairRerunParentCorrectionIdStringSql,
	repairRerunSourceDumpIdSql,
	type RepairRerunLineageSqlOptions
} from '$lib/repairRerunSql';
import {
	activePairedWorkflowStates,
	activeWriterLock,
	acquireWriterLock,
	clearWriterLockToken,
	writerLockConflictText
} from '$lib/server/pairedState';
import { OBSERVED_COST_SQL } from '$lib/server/scoreRunLifecycle';
import { sqlQuote } from '../cohorts/sqlEscape';


export type RepairRerunArchitecture = 'decomposed' | 'monolithic';
export type RepairRerunScoringMode = 'aggregate' | 'probe_only';
const MAX_RERUN_CORRECTION_IDS = 500;
// Phase 4 hard caps: one visible statement can fan out into many repair
// candidates (each evidence + the statement-scope candidate itself), so
// the per-statement candidate count and the total evidence fan-out both
// need explicit ceilings before write. These are *write-side* caps; the
// preflight surface MAX_RERUN_CORRECTION_IDS is the read-side limit.
const MAX_REPAIR_RERUN_EVIDENCES = 5_000;
const MAX_REPAIR_RERUN_STATEMENTS = 1_000;
const SOURCE_DUMP_RE = /^[a-z][a-z0-9_-]{1,63}$/i;
const TYPED_REPAIR_RERUN_LINEAGE_SQL: RepairRerunLineageSqlOptions = { typedLineage: true };
const REPAIR_RERUN_TERMINAL_STATUSES = new Set(['succeeded', 'failed', 'canceled', 'cancelled', 'aborted']);
const REPAIR_RERUN_TOMBSTONE_REASON = 'stale_repair_child_released_from_repair_ui';
const PROBE_REPAIR_SLOT_ORDER = [
	'substrate_route',
	'subject_role_probe',
	'object_role_probe',
	'relation_axis_probe',
	'scope_probe'
] as const;

export interface ExportRepairRerunInput {
	run_id: string;
	correction_ids?: number[] | null;
}

export interface DroppedCorrectionId {
	correction_id: number;
	reason: 'not_found' | 'wrong_run' | 'not_repair_candidate' | 'not_open' | 'already_rerun' | 'rerun_in_progress' | 'cap_exceeded';
}

export interface ExportRepairRerunResult {
	parent_run_id: string;
	architecture: RepairRerunArchitecture;
	path: string;
	source_dump_id: string;
	n_candidates: number;
	n_statements: number;
	n_evidences: number;
	n_raw_json_evidences: number;
	n_table_evidences: number;
	evidence_count_validated: boolean;
	correction_ids: number[];
	requested_correction_ids: number[];
	dropped_correction_ids: DroppedCorrectionId[];
	n_selected_evidence_candidates: number;
	n_statement_scope_candidates: number;
	n_scope_expansion_evidences: number;
	n_collateral_evidences: number;
	n_probe_slot_reviewed_candidates: number;
	probe_slot_reviews: ExportedProbeSlotReview[];
	probe_slot_counts: Record<string, number>;
	scoring_mode: RepairRerunScoringMode;
	probe_step_filter: string[];
	max_correction_ids: number;
	truncated: boolean;
}

export interface ExportedProbeSlotReview {
	correction_id: number;
	selected_slots: string[];
	note: string | null;
	reviewed_at: string | null;
	review_count: number;
}

export interface RecordRepairRerunInput {
	parent_run_id: string;
	child_run_id: string;
	architecture: RepairRerunArchitecture;
	source_dump_id: string;
	correction_ids: number[];
	path?: string;
	scoring_mode?: RepairRerunScoringMode;
	probe_step_filter?: string[];
}

export interface RecordRepairRerunResult {
	parent_run_id: string;
	child_run_id: string;
	recorded: number;
	skipped_existing: number;
	collateral_recorded: number;
	collateral_skipped_existing: number;
}

export interface RecordRepairRerunIntentResult {
	parent_run_id: string;
	child_run_id: string;
	recorded: number;
	skipped_existing: number;
}

export interface RecordRepairRerunUncoveredResult {
	parent_run_id: string;
	child_run_id: string;
	recorded: number;
	skipped_existing: number;
}

export interface TombstoneRepairRerunChildResult {
	parent_run_id: string;
	child_run_id: string;
	status: string;
	canceled: boolean;
	correction_ids: number[];
}


function dataRoot(): string {
	return resolve(dbPath(), '..');
}

function repairRerunRoot(): string {
	return resolve(dataRoot(), 'repair_reruns');
}

function normalizedCorrectionIds(ids: number[] | null | undefined): number[] {
	if (!ids) return [];
	return Array.from(new Set(
		ids
			.map((id) => Number(id))
			.filter((id) => Number.isInteger(id) && id > 0)
	));
}

function parseProbeSlotList(raw: unknown): string[] {
	if (raw == null) return [];
	const allowed = new Set<string>(PROBE_REPAIR_SLOT_ORDER);
	const seen = new Set<string>();
	const out: string[] = [];
	for (const slot of String(raw).split(',').map((s) => s.trim()).filter(Boolean)) {
		if (!allowed.has(slot) || seen.has(slot)) continue;
		seen.add(slot);
		out.push(slot);
	}
	return out;
}

function countProbeSlots(reviews: ExportedProbeSlotReview[]): Record<string, number> {
	const counts: Record<string, number> = {};
	for (const review of reviews) {
		for (const slot of review.selected_slots) {
			counts[slot] = (counts[slot] ?? 0) + 1;
		}
	}
	return counts;
}

function rawJsonEvidenceCount(stmtHash: string, rawJson: string): number {
	let parsed: unknown;
	try {
		parsed = JSON.parse(rawJson);
	} catch (err) {
		throw new Error(
			`repair rerun raw_json for statement ${stmtHash} is not parseable JSON: ${
				err instanceof Error ? err.message : String(err)
			}`
		);
	}
	if (parsed == null || typeof parsed !== 'object' || Array.isArray(parsed)) {
		throw new Error(`repair rerun raw_json for statement ${stmtHash} is not a statement object`);
	}
	const evidence = (parsed as { evidence?: unknown }).evidence;
	if (!Array.isArray(evidence)) {
		throw new Error(`repair rerun raw_json for statement ${stmtHash} has no evidence array`);
	}
	return evidence.length;
}

function probeStepFilterFromCounts(counts: Record<string, number>): string[] {
	const allowed = new Set<string>(PROBE_REPAIR_SLOT_ORDER);
	return Object.entries(counts)
		.filter(([slot, n]) => allowed.has(slot) && slot !== 'substrate_route' && Number(n) > 0)
		.map(([slot]) => slot)
		.sort((a, b) => a.localeCompare(b));
}

function normalizeProbeStepFilter(raw: unknown): string[] {
	const allowed = new Set<string>(PROBE_REPAIR_SLOT_ORDER);
	const seen = new Set<string>();
	const values = Array.isArray(raw)
		? raw
		: typeof raw === 'string'
			? raw.split(',')
			: [];
	const out: string[] = [];
	for (const value of values) {
		const step = String(value ?? '').trim();
		if (!step || step === 'substrate_route' || !allowed.has(step) || seen.has(step)) continue;
		seen.add(step);
		out.push(step);
	}
	return out.sort((a, b) => a.localeCompare(b));
}

async function latestProbeSlotReviewsForExport(
	con: DuckDBConnection,
	runIdSql: string,
	selectedIds: number[],
	lineageSqlOptions: RepairRerunLineageSqlOptions
): Promise<ExportedProbeSlotReview[]> {
	if (selectedIds.length === 0) return [];
	const parentCorrectionIdSql = repairRerunParentCorrectionIdNumberSql('r', lineageSqlOptions);
	const reader = await con.runAndReadAll(
		`WITH ranked AS (
			SELECT
			   ${parentCorrectionIdSql} AS parent_correction_id,
			   json_extract_string(r.value_json, '$.selected_probe_slots') AS selected_probe_slots,
			   COALESCE(json_extract_string(r.value_json, '$.reviewer_note'), r.note) AS reviewer_note,
			   r.created_at::VARCHAR AS reviewed_at,
			   COUNT(*) OVER (PARTITION BY ${parentCorrectionIdSql}) AS review_count,
			   ROW_NUMBER() OVER (
			     PARTITION BY ${parentCorrectionIdSql}
			     ORDER BY r.created_at DESC, r.correction_id DESC
			   ) AS rn
			 FROM scorer_step_correction r
			WHERE r.run_id='${runIdSql}'
			  AND r.correction_kind='probe_slot_review'
			  AND ${parentCorrectionIdSql} IN (${selectedIds.join(',')})
		)
		SELECT parent_correction_id, selected_probe_slots, reviewer_note, reviewed_at, review_count
		  FROM ranked
		 WHERE rn=1
		 ORDER BY parent_correction_id`
	);
	return reader.getRowObjects().map((row) => ({
		correction_id: Number(row.parent_correction_id),
		selected_slots: parseProbeSlotList(row.selected_probe_slots),
		note: row.reviewer_note == null ? null : String(row.reviewer_note),
		reviewed_at: row.reviewed_at == null ? null : String(row.reviewed_at),
		review_count: Number(row.review_count ?? 0)
	}));
}

async function repairRerunLineageSqlOptions(
	con: DuckDBConnection
): Promise<RepairRerunLineageSqlOptions> {
	const reader = await con.runAndReadAll(
		`SELECT COUNT(*) AS n
		   FROM information_schema.columns
		  WHERE table_name='scorer_step_correction'
		    AND column_name IN ('parent_correction_id', 'child_run_id', 'repair_source_dump_id')`
	);
	return { typedLineage: Number(reader.getRowObjects()[0]?.n ?? 0) === 3 };
}

async function ensureRepairRerunLineageColumns(con: DuckDBConnection): Promise<void> {
	await con.run(REPAIR_RERUN_LINEAGE_DDL);
}

async function tableColumns(con: DuckDBConnection, table: string): Promise<Set<string>> {
	const reader = await con.runAndReadAll(
		`SELECT column_name
		   FROM information_schema.columns
		  WHERE table_name=?`,
		[table]
	);
	return new Set(reader.getRowObjects().map((row) => String(row.column_name)));
}

function assertSafeSourceDumpId(source_dump_id: string): void {
	if (!SOURCE_DUMP_RE.test(source_dump_id)) {
		throw new Error('source_dump_id must be a safe repair rerun token');
	}
}

function repairRerunMetaPath(source_dump_id: string): string {
	assertSafeSourceDumpId(source_dump_id);
	return resolve(repairRerunRoot(), `${source_dump_id}.meta.json`);
}

function readRepairRerunMeta(source_dump_id: string): Record<string, unknown> | null {
	const metaPath = repairRerunMetaPath(source_dump_id);
	if (!existsSync(metaPath)) return null;
	return JSON.parse(readFileSync(metaPath, 'utf-8')) as Record<string, unknown>;
}

function repairRerunScoringMode(input: RecordRepairRerunInput): RepairRerunScoringMode {
	if (input.scoring_mode === 'probe_only') return 'probe_only';
	const meta = readRepairRerunMeta(input.source_dump_id);
	return meta?.scoring_mode === 'probe_only' ? 'probe_only' : 'aggregate';
}

function repairRerunProbeStepFilter(input: RecordRepairRerunInput): string[] {
	const explicit = normalizeProbeStepFilter(input.probe_step_filter);
	if (explicit.length > 0) return explicit;
	const meta = readRepairRerunMeta(input.source_dump_id);
	return normalizeProbeStepFilter(meta?.probe_step_filter);
}

function canonicalComparablePath(path: string): string {
	try {
		return realpathSync(path);
	} catch {
		return resolve(path);
	}
}

function assertRepairRerunMeta(
	input: RecordRepairRerunInput,
	ids: number[],
	options: { allowSubset?: boolean } = {}
): void {
	assertSafeSourceDumpId(input.source_dump_id);
	const meta = readRepairRerunMeta(input.source_dump_id);
	if (!meta) {
		throw new Error(`repair rerun meta not found for source_dump_id ${input.source_dump_id}`);
	}
	if (meta.parent_run_id !== input.parent_run_id) {
		throw new Error('repair rerun source_dump_id does not match parent_run_id');
	}
	if (meta.architecture !== input.architecture) {
		throw new Error('repair rerun source_dump_id does not match architecture');
	}
	if (meta.source_dump_id !== input.source_dump_id) {
		throw new Error('repair rerun meta source_dump_id mismatch');
	}
	if (
		input.path &&
		meta.path &&
		canonicalComparablePath(String(meta.path)) !== canonicalComparablePath(input.path)
	) {
		throw new Error('repair rerun source_dump_id does not match corpus path');
	}
	const metaIds = normalizedCorrectionIds(Array.isArray(meta.correction_ids) ? meta.correction_ids.map(Number) : [])
		.sort((a, b) => a - b);
	const inputIds = [...ids].sort((a, b) => a - b);
	if (options.allowSubset) {
		const metaIdSet = new Set(metaIds);
		if (!inputIds.every((id) => metaIdSet.has(id))) {
			throw new Error('repair rerun source_dump_id does not include requested correction_ids');
		}
	} else if (
		metaIds.length !== inputIds.length ||
		metaIds.some((id, i) => id !== inputIds[i])
	) {
		throw new Error('repair rerun source_dump_id does not match correction_ids');
	}
}

async function assertPersistedRepairRerunIntent(
	con: DuckDBConnection,
	input: RecordRepairRerunInput,
	ids: number[],
	lineageSqlOptions: RepairRerunLineageSqlOptions
): Promise<void> {
	assertSafeSourceDumpId(input.source_dump_id);
	const idPredicate = `parent_correction_id IN (${ids.join(',')})`;
	const parentCorrectionIdSql = repairRerunParentCorrectionIdNumberSql('intent', lineageSqlOptions);
	const childRunIdSql = repairRerunChildRunIdSql('intent', lineageSqlOptions);
	const sourceDumpIdSql = repairRerunSourceDumpIdSql('intent', lineageSqlOptions);
	const intentReader = await con.runAndReadAll(
		`WITH intent AS (
			SELECT DISTINCT
			       ${parentCorrectionIdSql} AS parent_correction_id
			  FROM scorer_step_correction intent
			 WHERE intent.run_id=?
			   AND intent.correction_kind='rerun_intent'
			   AND COALESCE(json_extract_string(intent.value_json, '$.parent_run_id'), intent.run_id)=?
			   AND ${childRunIdSql}=?
			   AND ${sourceDumpIdSql}=?
			   AND COALESCE(json_extract_string(intent.value_json, '$.architecture'), ?)=?
		)
		SELECT COUNT(*) AS n
		  FROM intent
		 WHERE ${idPredicate}`,
		[
			input.parent_run_id,
			input.parent_run_id,
			input.child_run_id,
			input.source_dump_id,
			input.architecture,
			input.architecture
		]
	);
	const matched = Number(intentReader.getRowObjects()[0]?.n ?? 0);
	if (matched !== ids.length) {
		throw new Error('repair rerun source_dump_id does not have matching persisted rerun_intent rows');
	}
}

async function assertRepairRerunSource(
	con: DuckDBConnection,
	input: RecordRepairRerunInput,
	ids: number[],
	lineageSqlOptions: RepairRerunLineageSqlOptions
): Promise<void> {
	const metaPath = repairRerunMetaPath(input.source_dump_id);
	if (existsSync(metaPath)) {
		assertRepairRerunMeta(input, ids, { allowSubset: true });
	}
	await assertPersistedRepairRerunIntent(con, input, ids, lineageSqlOptions);
}

function sleep(ms: number): Promise<void> {
	return new Promise((resolve) => setTimeout(resolve, ms));
}

function isDuckDBLockError(err: unknown): boolean {
	const message = err instanceof Error ? err.message : String(err);
	return message.includes('Could not set lock') || message.includes('Conflicting lock');
}

async function withRepairWriteConnection<T>(
	fn: (con: DuckDBConnection) => Promise<T>
): Promise<T> {
	let lastErr: unknown = null;
	for (let attempt = 0; attempt < 10; attempt += 1) {
		let instance: DuckDBInstance | null = null;
		let con: DuckDBConnection | null = null;
		try {
			await closeInstance();
			instance = await DuckDBInstance.create(dbPath());
			con = await instance.connect();
			return await fn(con);
		} catch (err) {
			lastErr = err;
			if (!isDuckDBLockError(err) || attempt === 9) break;
			await sleep(150 * (attempt + 1));
		} finally {
			try {
				con?.disconnectSync?.();
			} finally {
				instance?.closeSync();
			}
		}
	}
	throw lastErr instanceof Error ? lastErr : new Error(String(lastErr));
}

export async function exportRepairRerunCorpus(
	input: ExportRepairRerunInput
): Promise<ExportRepairRerunResult> {
	if (!dbExists()) throw new Error('corpus DuckDB does not exist');
	const con = await connect();
	try {
		const runId = sqlQuote(input.run_id);
		const hasCorrectionTableReader = await con.runAndReadAll(
			`SELECT COUNT(*) AS n
			   FROM information_schema.tables
			  WHERE table_name='scorer_step_correction'`
		);
		const hasCorrectionTable = Number(hasCorrectionTableReader.getRowObjects()[0]?.n ?? 0) > 0;
			if (!hasCorrectionTable) {
				throw new Error('repair backlog table does not exist');
			}
			const lineageSqlOptions = await repairRerunLineageSqlOptions(con);
			const runReader = await con.runAndReadAll(
				`SELECT architecture
				   FROM score_run
				  WHERE run_id='${runId}'`
			);
			const runRows = runReader.getRowObjects();
			if (runRows.length === 0) throw new Error('parent run not found');
			const parentArchitecture = String(runRows[0].architecture ?? 'unknown');
			if (parentArchitecture !== 'decomposed' && parentArchitecture !== 'monolithic') {
				throw new Error(`parent run architecture ${parentArchitecture} is not rerunnable`);
			}

			const ids = normalizedCorrectionIds(input.correction_ids);
			if (ids.length > MAX_RERUN_CORRECTION_IDS) {
				throw new Error(`at most ${MAX_RERUN_CORRECTION_IDS} repair candidates can be rerun at once`);
			}
			const idPredicate = ids.length > 0
				? `AND correction_id IN (${ids.join(',')})`
				: '';
			const selectedReader = await con.runAndReadAll(
				`SELECT correction_id, stmt_hash, evidence_hash
				   FROM scorer_step_correction c
				  WHERE c.run_id='${runId}'
				    AND c.status='open'
				    AND c.correction_kind='repair_candidate'
				    ${idPredicate}
				    AND ${repairCandidateAvailablePredicateSql('c', lineageSqlOptions)}
				  ORDER BY correction_id
				  LIMIT ${MAX_RERUN_CORRECTION_IDS}`
			);
			const selectedRows = selectedReader.getRowObjects();
			const selectedIds = selectedRows.map((row) => Number(row.correction_id));
			const selectedIdSet = new Set(selectedIds);
				const selectedEvidenceHashesByStatement = new Map<string, Set<string>>();
				const statementScopeStatementHashes = new Set<string>();
				for (const row of selectedRows) {
					const stmtHash = String(row.stmt_hash);
					if (row.evidence_hash == null) {
						statementScopeStatementHashes.add(stmtHash);
						continue;
					}
					const evidenceHashes = selectedEvidenceHashesByStatement.get(stmtHash) ?? new Set<string>();
					evidenceHashes.add(String(row.evidence_hash));
					selectedEvidenceHashesByStatement.set(stmtHash, evidenceHashes);
				}
				const nSelectedEvidenceCandidates = Array.from(selectedEvidenceHashesByStatement.values())
					.reduce((sum, evidenceHashes) => sum + evidenceHashes.size, 0);
				const nStatementScopeCandidates = selectedRows.filter((row) => row.evidence_hash == null).length;
				const probeSlotReviews = await latestProbeSlotReviewsForExport(
					con,
					runId,
					selectedIds,
					lineageSqlOptions
				);
				const probeSlotCounts = countProbeSlots(probeSlotReviews);
				const probeStepFilter = probeStepFilterFromCounts(probeSlotCounts);
			const dropped: DroppedCorrectionId[] = [];
			if (ids.length > 0) {
				const requestedReader = await con.runAndReadAll(
					`SELECT c.correction_id,
						        c.run_id,
						        c.status,
						        c.correction_kind,
						        ${repairCandidateConsumedExistsSql('c', lineageSqlOptions)} AS consumed,
						        ${repairCandidateActiveRerunIntentExistsSql('c', lineageSqlOptions)} AS active_rerun
					   FROM scorer_step_correction c
					  WHERE c.correction_id IN (${ids.join(',')})`
				);
				const requestedRows = new Map(
					requestedReader.getRowObjects().map((row) => [Number(row.correction_id), row])
				);
				for (const id of ids) {
					if (selectedIdSet.has(id)) continue;
					const row = requestedRows.get(id);
					if (!row) {
						dropped.push({ correction_id: id, reason: 'not_found' });
					} else if (String(row.run_id) !== input.run_id) {
						dropped.push({ correction_id: id, reason: 'wrong_run' });
					} else if (String(row.correction_kind) !== 'repair_candidate') {
						dropped.push({ correction_id: id, reason: 'not_repair_candidate' });
					} else if (String(row.status) !== 'open') {
						dropped.push({ correction_id: id, reason: 'not_open' });
					} else if (Boolean(row.consumed)) {
						dropped.push({ correction_id: id, reason: 'already_rerun' });
					} else if (Boolean(row.active_rerun)) {
						dropped.push({ correction_id: id, reason: 'rerun_in_progress' });
					} else {
						dropped.push({ correction_id: id, reason: 'cap_exceeded' });
					}
				}
			}
			if (selectedRows.length === 0) {
				const detail = dropped.length > 0
					? `: ${dropped.map((d) => `#${d.correction_id} ${d.reason.replace(/_/g, ' ')}`).join(', ')}`
					: '';
				throw new Error(`no open repair candidates match this rerun request${detail}`);
			}
			const evidenceCountSql = `(SELECT COUNT(*) FROM statement_evidence se WHERE se.stmt_hash=selected_stmt.stmt_hash)`;

			const stmtReader = await con.runAndReadAll(
				`WITH selected AS (
					SELECT correction_id, stmt_hash
					  FROM scorer_step_correction
					 WHERE correction_id IN (${selectedIds.join(',')})
				),
				selected_stmt AS (
					SELECT stmt_hash,
					       COUNT(DISTINCT correction_id) AS n_candidates
					  FROM selected
					 GROUP BY stmt_hash
				)
				SELECT
				   selected_stmt.stmt_hash,
				   selected_stmt.n_candidates,
				   CAST(s.raw_json AS VARCHAR) AS statement_raw_json,
				   ${evidenceCountSql} AS n_evidences
				 FROM selected_stmt
				 JOIN statement s ON s.stmt_hash=selected_stmt.stmt_hash
				 ORDER BY selected_stmt.stmt_hash`
			);
			const stmtRows = stmtReader.getRowObjects();

			const statementJson: string[] = [];
			let nEvidences = 0;
			let nRawJsonEvidences = 0;
			let nCandidates = 0;
			let nScopeExpansionEvidences = 0;
			let nEvidenceBeyondSelectedEvidenceCandidates = 0;
			for (const row of stmtRows) {
				const stmtHash = String(row.stmt_hash);
				const rowEvidences = Number(row.n_evidences ?? 0);
				const rawStatementJson = String(row.statement_raw_json);
				const rowRawJsonEvidences = rawJsonEvidenceCount(stmtHash, rawStatementJson);
				if (rowRawJsonEvidences !== rowEvidences) {
							throw new Error(
								`repair rerun evidence denominator mismatch for statement ${stmtHash}: ` +
								`raw_json has ${rowRawJsonEvidences} evidence row${rowRawJsonEvidences === 1 ? '' : 's'}, ` +
								`normalized evidence rows have ${rowEvidences}`
							);
				}
				const selectedEvidenceCount = selectedEvidenceHashesByStatement.get(stmtHash)?.size ?? 0;
				statementJson.push(rawStatementJson);
				nEvidences += rowEvidences;
				nRawJsonEvidences += rowRawJsonEvidences;
				nCandidates += Number(row.n_candidates ?? 0);
				nEvidenceBeyondSelectedEvidenceCandidates += Math.max(0, rowEvidences - selectedEvidenceCount);
				if (!statementScopeStatementHashes.has(stmtHash)) {
					nScopeExpansionEvidences += Math.max(0, rowEvidences - selectedEvidenceCount);
				}
			}
			if (statementJson.length === 0 || nEvidences === 0) {
				throw new Error('repair rerun export produced no scorable evidence');
			}
			if (statementJson.length > MAX_REPAIR_RERUN_STATEMENTS) {
				throw new Error(
					`repair rerun export exceeded statement cap: ${statementJson.length} > ` +
					`${MAX_REPAIR_RERUN_STATEMENTS}. Reduce the candidate selection or raise ` +
					`MAX_REPAIR_RERUN_STATEMENTS after confirming token spend is acceptable.`
				);
			}
			if (nEvidences > MAX_REPAIR_RERUN_EVIDENCES) {
				throw new Error(
					`repair rerun export exceeded evidence fan-out cap: ${nEvidences} > ` +
					`${MAX_REPAIR_RERUN_EVIDENCES}. Reduce the candidate selection or raise ` +
					`MAX_REPAIR_RERUN_EVIDENCES after confirming token spend is acceptable.`
				);
			}
			const scoringMode: RepairRerunScoringMode =
				parentArchitecture === 'decomposed' &&
				probeSlotReviews.length === nCandidates &&
				probeStepFilter.length > 0 &&
				nStatementScopeCandidates === 0
					? 'probe_only'
					: 'aggregate';

			const source_dump_id = `repair_${input.run_id.slice(0, 8)}_${randomUUID().replace(/-/g, '').slice(0, 12)}`;
			const root = repairRerunRoot();
			mkdirSync(root, { recursive: true });
			const path = resolve(root, `${source_dump_id}.json`);
			writeFileSync(path, `[\n${statementJson.join(',\n')}\n]\n`);
			writeFileSync(
				resolve(root, `${source_dump_id}.meta.json`),
				JSON.stringify({
					parent_run_id: input.run_id,
					architecture: parentArchitecture,
					source_dump_id,
					path,
					export_scope: 'candidate_statements',
					n_candidates: nCandidates,
					n_statements: statementJson.length,
					n_evidences: nEvidences,
					n_raw_json_evidences: nRawJsonEvidences,
					n_table_evidences: nEvidences,
					evidence_count_validated: true,
					correction_ids: selectedIds,
					requested_correction_ids: ids.length > 0 ? ids : selectedIds,
					dropped_correction_ids: dropped,
					n_selected_evidence_candidates: nSelectedEvidenceCandidates,
					n_statement_scope_candidates: nStatementScopeCandidates,
					n_scope_expansion_evidences: nScopeExpansionEvidences,
					n_collateral_evidences: nEvidenceBeyondSelectedEvidenceCandidates,
					n_probe_slot_reviewed_candidates: probeSlotReviews.length,
					probe_slot_reviews: probeSlotReviews,
					probe_slot_counts: probeSlotCounts,
					scoring_mode: scoringMode,
					probe_step_filter: probeStepFilter,
					max_correction_ids: MAX_RERUN_CORRECTION_IDS,
					truncated: ids.length === 0 && selectedIds.length === MAX_RERUN_CORRECTION_IDS
				}, null, 2)
			);

			return {
				parent_run_id: input.run_id,
				architecture: parentArchitecture,
				path,
				source_dump_id,
				n_candidates: nCandidates,
				n_statements: statementJson.length,
				n_evidences: nEvidences,
				n_raw_json_evidences: nRawJsonEvidences,
				n_table_evidences: nEvidences,
				evidence_count_validated: true,
				correction_ids: selectedIds,
				requested_correction_ids: ids.length > 0 ? ids : selectedIds,
				dropped_correction_ids: dropped,
				n_selected_evidence_candidates: nSelectedEvidenceCandidates,
				n_statement_scope_candidates: nStatementScopeCandidates,
				n_scope_expansion_evidences: nScopeExpansionEvidences,
				n_collateral_evidences: nEvidenceBeyondSelectedEvidenceCandidates,
				n_probe_slot_reviewed_candidates: probeSlotReviews.length,
				probe_slot_reviews: probeSlotReviews,
				probe_slot_counts: probeSlotCounts,
				scoring_mode: scoringMode,
				probe_step_filter: probeStepFilter,
				max_correction_ids: MAX_RERUN_CORRECTION_IDS,
				truncated: ids.length === 0 && selectedIds.length === MAX_RERUN_CORRECTION_IDS
			};
		} finally {
			con.disconnectSync?.();
			await closeInstance();
	}
}

async function recordRepairRerunChildOnConnection(
	con: DuckDBConnection,
	input: RecordRepairRerunInput,
	ids: number[]
): Promise<RecordRepairRerunResult> {
	await ensureRepairRerunLineageColumns(con);
	await assertRepairRerunSource(con, input, ids, TYPED_REPAIR_RERUN_LINEAGE_SQL);
	const childReader = await con.runAndReadAll(
		`SELECT COALESCE(architecture, 'unknown') AS architecture,
		        COALESCE(status, 'unknown') AS status,
		        parent_run_id
		   FROM score_run
		  WHERE run_id=?
		    AND parent_run_id=?`,
		[input.child_run_id, input.parent_run_id]
	);
	const childRows = childReader.getRowObjects();
	if (childRows.length === 0) {
		throw new Error('child repair run not found for parent_run_id');
	}
	const child = childRows[0];
	if (String(child.status) !== 'succeeded') {
		throw new Error(`child repair run ${input.child_run_id.slice(0, 8)} is ${String(child.status)}; rerun markers require status=succeeded`);
	}
	if (String(child.architecture) !== input.architecture) {
		throw new Error(`child repair run architecture ${String(child.architecture)} does not match ${input.architecture}`);
	}

	const runId = sqlQuote(input.parent_run_id);
	const idPredicate = `correction_id IN (${ids.join(',')})`;
	const reader = await con.runAndReadAll(
		`SELECT correction_id, step_hash, run_id, architecture, stmt_hash, evidence_hash,
		        source_route, source_filters_json
		   FROM scorer_step_correction
		  WHERE run_id='${runId}'
		    AND correction_kind='repair_candidate'
		    AND ${idPredicate}
		  ORDER BY correction_id`
	);
	const candidates = reader.getRowObjects();
	if (candidates.length !== ids.length) {
		throw new Error('repair rerun completion correction_ids do not all match parent repair candidates');
	}
	const scoringMode = repairRerunScoringMode(input);
	const probeStepFilter = repairRerunProbeStepFilter(input);
	if (scoringMode === 'probe_only' && probeStepFilter.length === 0) {
		throw new Error('probe-only repair rerun completion requires probe_step_filter metadata');
	}
	const probeCoverageSql = (stepKind: string) => `
		(POSITION('${sqlQuote(stepKind)}' IN ?) = 0 OR EXISTS (
			SELECT 1 FROM scorer_step child
			 WHERE child.run_id=?
			   AND child.step_kind='${sqlQuote(stepKind)}'
			   AND child.stmt_hash=c.stmt_hash
			   AND child.evidence_hash=c.evidence_hash
		))`;
	const coverageSql = scoringMode === 'probe_only'
		? `WITH candidate AS (
			SELECT correction_id, stmt_hash, evidence_hash
			  FROM scorer_step_correction
			 WHERE run_id='${runId}'
			   AND correction_kind='repair_candidate'
			   AND ${idPredicate}
		)
		SELECT
		   SUM(CASE
		         WHEN c.evidence_hash IS NULL THEN 1
		         WHEN ${probeCoverageSql('subject_role_probe')}
		          AND ${probeCoverageSql('object_role_probe')}
		          AND ${probeCoverageSql('relation_axis_probe')}
		          AND ${probeCoverageSql('scope_probe')}
		         THEN 0 ELSE 1
		       END) AS missing_evidence_candidates,
		   SUM(CASE WHEN c.evidence_hash IS NULL THEN 1 ELSE 0 END) AS missing_statement_candidates
		  FROM candidate c`
		: `WITH candidate AS (
			SELECT correction_id, stmt_hash, evidence_hash
			  FROM scorer_step_correction
			 WHERE run_id='${runId}'
			   AND correction_kind='repair_candidate'
			   AND ${idPredicate}
		)
		SELECT
		   SUM(CASE WHEN c.evidence_hash IS NOT NULL AND cs.step_hash IS NULL THEN 1 ELSE 0 END) AS missing_evidence_candidates,
		   SUM(CASE
		         WHEN c.evidence_hash IS NULL
		          AND (
		            NOT EXISTS (
		              SELECT 1 FROM statement_evidence pe
		               WHERE pe.stmt_hash=c.stmt_hash
		            )
		            OR EXISTS (
		              SELECT 1
		                FROM statement_evidence pe
		               WHERE pe.stmt_hash=c.stmt_hash
		                 AND NOT EXISTS (
		                   SELECT 1
		                     FROM scorer_step child
		                    WHERE child.run_id=?
		                      AND child.step_kind='aggregate'
		                      AND child.stmt_hash=pe.stmt_hash
		                      AND child.evidence_hash=pe.evidence_hash
		                 )
		            )
		          )
		         THEN 1 ELSE 0
		       END) AS missing_statement_candidates
		  FROM candidate c
		  LEFT JOIN scorer_step cs
		    ON cs.run_id=?
		   AND cs.step_kind='aggregate'
		   AND cs.stmt_hash=c.stmt_hash
		   AND cs.evidence_hash=c.evidence_hash`;
	const probeFilterCsv = probeStepFilter.join(',');
	const coverageParams = scoringMode === 'probe_only'
		? [
			probeFilterCsv, input.child_run_id,
			probeFilterCsv, input.child_run_id,
			probeFilterCsv, input.child_run_id,
			probeFilterCsv, input.child_run_id
		]
		: [input.child_run_id, input.child_run_id];
	const coverageReader = await con.runAndReadAll(coverageSql, coverageParams);
	const coverage = coverageReader.getRowObjects()[0] ?? {};
	const missingEvidenceCandidates = Number(coverage.missing_evidence_candidates ?? 0);
	const missingStatementCandidates = Number(coverage.missing_statement_candidates ?? 0);
	if (missingEvidenceCandidates > 0 || missingStatementCandidates > 0) {
		throw new Error(
			`child repair run does not cover selected repair candidates: ` +
			`${missingEvidenceCandidates} evidence candidate(s), ${missingStatementCandidates} statement candidate(s) missing`
		);
	}
	let recorded = 0;
	let skipped_existing = 0;
	let collateral_recorded = 0;
	let collateral_skipped_existing = 0;
	await con.run('BEGIN TRANSACTION');
	try {
		for (const c of candidates) {
			const correctionId = Number(c.correction_id);
			const existingReader = await con.runAndReadAll(
				`SELECT COUNT(*) AS n
				   FROM scorer_step_correction
				  WHERE correction_kind='rerun_child'
				    AND ${repairRerunParentCorrectionIdStringSql('scorer_step_correction', TYPED_REPAIR_RERUN_LINEAGE_SQL)}=?`,
				[String(correctionId)]
			);
			const existing = Number(existingReader.getRowObjects()[0]?.n ?? 0);
			if (existing > 0) {
				skipped_existing += 1;
				continue;
			}
			const valueJson = JSON.stringify({
				kind: 'repair_rerun_child',
				parent_correction_id: correctionId,
				parent_run_id: input.parent_run_id,
				child_run_id: input.child_run_id,
				source_dump_id: input.source_dump_id,
				architecture: input.architecture,
				scoring_mode: scoringMode,
				probe_step_filter: probeStepFilter,
				probe_step_filter_csv: probeStepFilter.join(',')
			});
			await con.run(
				`INSERT INTO scorer_step_correction
				 (step_hash, run_id, architecture, stmt_hash, evidence_hash,
				  correction_kind, status, reviewer, note, value_json,
				  parent_correction_id, child_run_id, repair_source_dump_id,
				  source_route, source_filters_json)
				 VALUES (?, ?, ?, ?, ?, 'rerun_child', 'resolved', 'viewer',
				         ?, ?::JSON, ?, ?, ?, ?, ?::JSON)`,
				[
					String(c.step_hash),
					String(c.run_id),
					String(c.architecture ?? input.architecture),
					String(c.stmt_hash),
					c.evidence_hash == null ? null : String(c.evidence_hash),
					`rerun child ${input.child_run_id.slice(0, 8)}`,
					valueJson,
					correctionId,
					input.child_run_id,
					input.source_dump_id,
					`/runs/${input.child_run_id}`,
					c.source_filters_json == null ? null : String(c.source_filters_json)
				]
			);
			recorded += 1;
		}
		const candidateIdsByStatement = new Map<string, number[]>();
		for (const c of candidates) {
			const stmt = String(c.stmt_hash);
			const idsForStatement = candidateIdsByStatement.get(stmt) ?? [];
			idsForStatement.push(Number(c.correction_id));
			candidateIdsByStatement.set(stmt, idsForStatement);
		}
		if (scoringMode === 'probe_only') {
			await con.run('COMMIT');
			return {
				parent_run_id: input.parent_run_id,
				child_run_id: input.child_run_id,
				recorded,
				skipped_existing,
				collateral_recorded,
				collateral_skipped_existing
			};
		}
		const collateralReader = await con.runAndReadAll(
			`WITH candidate AS (
				SELECT correction_id, stmt_hash, evidence_hash
				  FROM scorer_step_correction
				 WHERE run_id='${runId}'
				   AND correction_kind='repair_candidate'
				   AND ${idPredicate}
			),
			candidate_stmt AS (
				SELECT stmt_hash,
				       MAX(CASE WHEN evidence_hash IS NULL THEN 1 ELSE 0 END) AS has_statement_scope
				  FROM candidate
				 GROUP BY stmt_hash
			),
			candidate_evidence AS (
				SELECT DISTINCT evidence_hash
				  FROM candidate
				 WHERE evidence_hash IS NOT NULL
			)
			SELECT
			   COALESCE(ps.step_hash, cs.step_hash) AS lineage_step_hash,
			   cs.step_hash AS child_step_hash,
			   cs.stmt_hash,
			   cs.evidence_hash
			  FROM scorer_step cs
			  JOIN candidate_stmt st ON st.stmt_hash=cs.stmt_hash
			  LEFT JOIN scorer_step ps
			    ON ps.run_id=?
			   AND ps.step_kind='aggregate'
			   AND ps.stmt_hash=cs.stmt_hash
			   AND ps.evidence_hash=cs.evidence_hash
			 WHERE cs.run_id=?
			   AND cs.step_kind='aggregate'
			   AND cs.evidence_hash IS NOT NULL
			   AND st.has_statement_scope=0
			   AND NOT EXISTS (
			     SELECT 1 FROM candidate_evidence ce
			      WHERE ce.evidence_hash=cs.evidence_hash
			   )
			 ORDER BY cs.stmt_hash, cs.evidence_hash`,
			[input.parent_run_id, input.child_run_id]
		);
		for (const row of collateralReader.getRowObjects()) {
			const evidenceHash = String(row.evidence_hash);
			const existingReader = await con.runAndReadAll(
				`SELECT COUNT(*) AS n
				   FROM scorer_step_correction
				  WHERE correction_kind='rerun_collateral'
				    AND evidence_hash=?
				    AND ${repairRerunChildRunIdSql('scorer_step_correction', TYPED_REPAIR_RERUN_LINEAGE_SQL)}=?`,
				[evidenceHash, input.child_run_id]
			);
			const existing = Number(existingReader.getRowObjects()[0]?.n ?? 0);
			if (existing > 0) {
				collateral_skipped_existing += 1;
				continue;
			}
			const stmtHash = String(row.stmt_hash);
			const valueJson = JSON.stringify({
				kind: 'repair_rerun_collateral',
				parent_run_id: input.parent_run_id,
				child_run_id: input.child_run_id,
				source_dump_id: input.source_dump_id,
				architecture: input.architecture,
				child_step_hash: String(row.child_step_hash),
				selected_parent_correction_ids: candidateIdsByStatement.get(stmtHash) ?? []
			});
			await con.run(
				`INSERT INTO scorer_step_correction
				 (step_hash, run_id, architecture, stmt_hash, evidence_hash,
				  correction_kind, status, reviewer, note, value_json,
				  parent_correction_id, child_run_id, repair_source_dump_id,
				  source_route, source_filters_json)
				 VALUES (?, ?, ?, ?, ?, 'rerun_collateral', 'resolved', 'viewer',
				         ?, ?::JSON, NULL, ?, ?, ?, NULL)`,
				[
					String(row.lineage_step_hash),
					input.parent_run_id,
					input.architecture,
					stmtHash,
					evidenceHash,
					`collateral rerun by child ${input.child_run_id.slice(0, 8)}`,
					valueJson,
					input.child_run_id,
					input.source_dump_id,
					`/runs/${input.child_run_id}`
				]
			);
			collateral_recorded += 1;
		}
		await con.run('COMMIT');
	} catch (e) {
		await con.run('ROLLBACK');
		throw e;
	}
	return {
		parent_run_id: input.parent_run_id,
		child_run_id: input.child_run_id,
		recorded,
		skipped_existing,
		collateral_recorded,
		collateral_skipped_existing
	};
}

async function recordRepairRerunUncoveredOnConnection(
	con: DuckDBConnection,
	input: RecordRepairRerunInput,
	ids: number[]
): Promise<RecordRepairRerunUncoveredResult> {
	await ensureRepairRerunLineageColumns(con);
	await assertRepairRerunSource(con, input, ids, TYPED_REPAIR_RERUN_LINEAGE_SQL);
	const childReader = await con.runAndReadAll(
		`SELECT COALESCE(architecture, 'unknown') AS architecture,
		        COALESCE(status, 'unknown') AS status,
		        parent_run_id
		   FROM score_run
		  WHERE run_id=?
		    AND parent_run_id=?`,
		[input.child_run_id, input.parent_run_id]
	);
	const childRows = childReader.getRowObjects();
	if (childRows.length === 0) {
		throw new Error('child repair run not found for parent_run_id');
	}
	const child = childRows[0];
	if (String(child.status) !== 'succeeded') {
		throw new Error(`child repair run ${input.child_run_id.slice(0, 8)} is ${String(child.status)}; uncovered release requires status=succeeded`);
	}
	if (String(child.architecture) !== input.architecture) {
		throw new Error(`child repair run architecture ${String(child.architecture)} does not match ${input.architecture}`);
	}

	const runId = sqlQuote(input.parent_run_id);
	const idPredicate = `correction_id IN (${ids.join(',')})`;
	const reader = await con.runAndReadAll(
		`SELECT correction_id, step_hash, run_id, architecture, stmt_hash, evidence_hash,
		        source_route, source_filters_json
		   FROM scorer_step_correction
		  WHERE run_id='${runId}'
		    AND correction_kind='repair_candidate'
		    AND ${idPredicate}
		  ORDER BY correction_id`
	);
	const candidates = reader.getRowObjects();
	if (candidates.length !== ids.length) {
		throw new Error('repair rerun uncovered correction_ids do not all match parent repair candidates');
	}
	const coverageReader = await con.runAndReadAll(
		`WITH candidate AS (
			SELECT correction_id, stmt_hash, evidence_hash
			  FROM scorer_step_correction
			 WHERE run_id='${runId}'
			   AND correction_kind='repair_candidate'
			   AND ${idPredicate}
		)
		SELECT c.correction_id,
		       CASE
		         WHEN c.evidence_hash IS NOT NULL AND cs.step_hash IS NOT NULL THEN 1
		         WHEN c.evidence_hash IS NULL
		          AND EXISTS (
		            SELECT 1 FROM statement_evidence pe
		             WHERE pe.stmt_hash=c.stmt_hash
		          )
		          AND NOT EXISTS (
		            SELECT 1
		              FROM statement_evidence pe
		             WHERE pe.stmt_hash=c.stmt_hash
		               AND NOT EXISTS (
		                 SELECT 1
		                   FROM scorer_step child
		                  WHERE child.run_id=?
		                    AND child.step_kind='aggregate'
		                    AND child.stmt_hash=pe.stmt_hash
		                    AND child.evidence_hash=pe.evidence_hash
		               )
		          ) THEN 1
		         ELSE 0
		       END AS child_covers
		  FROM candidate c
		  LEFT JOIN scorer_step cs
		    ON cs.run_id=?
		   AND cs.step_kind='aggregate'
		   AND cs.stmt_hash=c.stmt_hash
		   AND cs.evidence_hash=c.evidence_hash
		  `,
		[input.child_run_id, input.child_run_id]
	);
	const coveredIds = coverageReader.getRowObjects()
		.filter((row) => Number(row.child_covers ?? 0) > 0)
		.map((row) => Number(row.correction_id));
	if (coveredIds.length > 0) {
		throw new Error(`covered repair candidates require completion markers before uncovered release: ${coveredIds.join(',')}`);
	}

	let recorded = 0;
	let skipped_existing = 0;
	await con.run('BEGIN TRANSACTION');
	try {
		for (const c of candidates) {
			const correctionId = Number(c.correction_id);
			const existingReader = await con.runAndReadAll(
				`SELECT COUNT(*) AS n
				   FROM scorer_step_correction
				  WHERE correction_kind='rerun_uncovered'
				    AND ${repairRerunParentCorrectionIdStringSql('scorer_step_correction', TYPED_REPAIR_RERUN_LINEAGE_SQL)}=?
				    AND ${repairRerunChildRunIdSql('scorer_step_correction', TYPED_REPAIR_RERUN_LINEAGE_SQL)}=?`,
				[String(correctionId), input.child_run_id]
			);
			const existing = Number(existingReader.getRowObjects()[0]?.n ?? 0);
			if (existing > 0) {
				skipped_existing += 1;
				continue;
			}
			const valueJson = JSON.stringify({
				kind: 'repair_rerun_uncovered',
				parent_correction_id: correctionId,
				parent_run_id: input.parent_run_id,
				child_run_id: input.child_run_id,
				source_dump_id: input.source_dump_id,
				architecture: input.architecture,
				reason: 'child_run_missing_aggregate'
			});
			await con.run(
				`INSERT INTO scorer_step_correction
				 (step_hash, run_id, architecture, stmt_hash, evidence_hash,
				  correction_kind, status, reviewer, note, value_json,
				  parent_correction_id, child_run_id, repair_source_dump_id,
				  source_route, source_filters_json)
				 VALUES (?, ?, ?, ?, ?, 'rerun_uncovered', 'resolved', 'viewer',
				         ?, ?::JSON, ?, ?, ?, ?, ?::JSON)`,
				[
					String(c.step_hash),
					String(c.run_id),
					String(c.architecture ?? input.architecture),
					String(c.stmt_hash),
					c.evidence_hash == null ? null : String(c.evidence_hash),
					`rerun child ${input.child_run_id.slice(0, 8)} missing aggregate`,
					valueJson,
					correctionId,
					input.child_run_id,
					input.source_dump_id,
					`/runs/${input.child_run_id}`,
					c.source_filters_json == null ? null : String(c.source_filters_json)
				]
			);
			recorded += 1;
		}
		await con.run('COMMIT');
	} catch (e) {
		await con.run('ROLLBACK');
		throw e;
	}
	return {
		parent_run_id: input.parent_run_id,
		child_run_id: input.child_run_id,
		recorded,
		skipped_existing
	};
}

async function recordRepairRerunIntentOnConnection(
	con: DuckDBConnection,
	input: RecordRepairRerunInput,
	ids: number[]
): Promise<RecordRepairRerunIntentResult> {
	await ensureRepairRerunLineageColumns(con);
	assertRepairRerunMeta(input, ids);
	const runId = sqlQuote(input.parent_run_id);
	const idPredicate = `correction_id IN (${ids.join(',')})`;
	const reader = await con.runAndReadAll(
		`SELECT correction_id, step_hash, run_id, architecture, stmt_hash, evidence_hash,
		        source_route, source_filters_json
		   FROM scorer_step_correction c
		  WHERE c.run_id='${runId}'
		    AND c.status='open'
		    AND c.correction_kind='repair_candidate'
		    AND ${idPredicate}
		    AND ${repairCandidateAvailablePredicateSql('c', TYPED_REPAIR_RERUN_LINEAGE_SQL)}
		  ORDER BY correction_id`
	);
	const candidates = reader.getRowObjects();
	if (candidates.length !== ids.length) {
		throw new Error('repair rerun intent correction_ids are stale, closed, already consumed, or already have an active child run; estimate this repair rerun again');
	}
	let recorded = 0;
	let skipped_existing = 0;
	await con.run('BEGIN TRANSACTION');
	try {
		for (const c of candidates) {
			const correctionId = Number(c.correction_id);
			const existingReader = await con.runAndReadAll(
				`SELECT COUNT(*) AS n
				   FROM scorer_step_correction
				  WHERE correction_kind='rerun_intent'
				    AND ${repairRerunParentCorrectionIdStringSql('scorer_step_correction', TYPED_REPAIR_RERUN_LINEAGE_SQL)}=?
				    AND ${repairRerunChildRunIdSql('scorer_step_correction', TYPED_REPAIR_RERUN_LINEAGE_SQL)}=?`,
				[String(correctionId), input.child_run_id]
			);
			const existing = Number(existingReader.getRowObjects()[0]?.n ?? 0);
			if (existing > 0) {
				skipped_existing += 1;
				continue;
			}
			const valueJson = JSON.stringify({
				kind: 'repair_rerun_intent',
				parent_correction_id: correctionId,
				parent_run_id: input.parent_run_id,
				child_run_id: input.child_run_id,
				source_dump_id: input.source_dump_id,
				architecture: input.architecture,
				scoring_mode: input.scoring_mode ?? 'aggregate',
				probe_step_filter: input.probe_step_filter ?? [],
				probe_step_filter_csv: normalizeProbeStepFilter(input.probe_step_filter).join(',')
			});
			await con.run(
				`INSERT INTO scorer_step_correction
				 (step_hash, run_id, architecture, stmt_hash, evidence_hash,
				  correction_kind, status, reviewer, note, value_json,
				  parent_correction_id, child_run_id, repair_source_dump_id,
				  source_route, source_filters_json)
				 VALUES (?, ?, ?, ?, ?, 'rerun_intent', 'open', 'viewer',
				         ?, ?::JSON, ?, ?, ?, ?, ?::JSON)`,
				[
					String(c.step_hash),
					String(c.run_id),
					String(c.architecture ?? input.architecture),
					String(c.stmt_hash),
					c.evidence_hash == null ? null : String(c.evidence_hash),
					`rerun intent ${input.child_run_id.slice(0, 8)}`,
					valueJson,
					correctionId,
					input.child_run_id,
					input.source_dump_id,
					`/runs/${input.child_run_id}`,
					c.source_filters_json == null ? null : String(c.source_filters_json)
				]
			);
			recorded += 1;
		}
		await con.run('COMMIT');
	} catch (e) {
		await con.run('ROLLBACK');
		throw e;
	}
	return {
		parent_run_id: input.parent_run_id,
		child_run_id: input.child_run_id,
		recorded,
		skipped_existing
	};
}

async function tombstoneRepairRerunChildOnConnection(
	con: DuckDBConnection,
	input: RecordRepairRerunInput,
	ids: number[]
): Promise<TombstoneRepairRerunChildResult> {
	await ensureRepairRerunLineageColumns(con);
	await assertRepairRerunSource(con, input, ids, TYPED_REPAIR_RERUN_LINEAGE_SQL);
	const childReader = await con.runAndReadAll(
		`SELECT COALESCE(architecture, 'unknown') AS architecture,
		        COALESCE(status, 'unknown') AS status,
		        parent_run_id
		   FROM score_run
		  WHERE run_id=?
		    AND parent_run_id=?`,
		[input.child_run_id, input.parent_run_id]
	);
	const child = childReader.getRowObjects()[0] ?? null;
	if (!child) {
		throw new Error('child repair run row not found; queued repair intents expire automatically if the worker never starts');
	}
	if (String(child.architecture) !== input.architecture) {
		throw new Error(`child repair run architecture ${String(child.architecture)} does not match ${input.architecture}`);
	}
	const status = String(child.status ?? 'unknown');
	if (REPAIR_RERUN_TERMINAL_STATUSES.has(status)) {
		return {
			parent_run_id: input.parent_run_id,
			child_run_id: input.child_run_id,
			status,
			canceled: false,
			correction_ids: ids
		};
	}
	const activeIntentParentIdSql = repairRerunParentCorrectionIdNumberSql('intent', TYPED_REPAIR_RERUN_LINEAGE_SQL);
	const activeIntentChildRunIdSql = repairRerunChildRunIdSql('intent', TYPED_REPAIR_RERUN_LINEAGE_SQL);
	const activeIntentSourceDumpIdSql = repairRerunSourceDumpIdSql('intent', TYPED_REPAIR_RERUN_LINEAGE_SQL);
	const activeIntentReader = await con.runAndReadAll(
		`SELECT DISTINCT
		        ${activeIntentParentIdSql} AS parent_correction_id,
		        ${activeIntentSourceDumpIdSql} AS source_dump_id
		   FROM scorer_step_correction intent
		   JOIN scorer_step_correction c
		     ON c.correction_id=${activeIntentParentIdSql}
		    AND c.run_id=?
		    AND c.correction_kind='repair_candidate'
		    AND c.status='open'
		  WHERE intent.run_id=?
		    AND intent.correction_kind='rerun_intent'
		    AND COALESCE(json_extract_string(intent.value_json, '$.parent_run_id'), intent.run_id)=?
		    AND ${activeIntentChildRunIdSql}=?
		    AND ${activeIntentParentIdSql} IS NOT NULL`,
		[input.parent_run_id, input.parent_run_id, input.parent_run_id, input.child_run_id]
	);
	const activeIntentRows = activeIntentReader.getRowObjects();
	if (activeIntentRows.some((row) => row.source_dump_id == null || String(row.source_dump_id) !== input.source_dump_id)) {
		throw new Error('child repair run has active intents with a missing or different source_dump_id; inspect the child run before tombstoning');
	}
	const activeIntentIds = Array.from(new Set(
		activeIntentRows.map((row) => Number(row.parent_correction_id))
			.filter((id) => Number.isInteger(id) && id > 0)
	)).sort((a, b) => a - b);
	const inputIds = [...ids].sort((a, b) => a - b);
	if (
		activeIntentIds.length !== inputIds.length ||
		activeIntentIds.some((id, i) => id !== inputIds[i])
	) {
		throw new Error('tombstone correction_ids must match all active repair intents for this child run; reload the repair page');
	}

	const columns = await tableColumns(con, 'score_run');
	const setClauses = [`status='canceled'`];
	const params: string[] = [];
	if (columns.has('finished_at')) {
		setClauses.push('finished_at=COALESCE(finished_at, CURRENT_TIMESTAMP)');
	}
	if (columns.has('terminated_by')) {
		setClauses.push("terminated_by=COALESCE(terminated_by, 'user')");
	}
	if (columns.has('termination_reason')) {
		setClauses.push('termination_reason=COALESCE(termination_reason, ?)');
		params.push(REPAIR_RERUN_TOMBSTONE_REASON);
	}
	if (columns.has('cost_actual_usd')) {
		setClauses.push(`cost_actual_usd=COALESCE(cost_actual_usd, (${OBSERVED_COST_SQL}))`);
		params.push(input.child_run_id);
	}

	await con.run('BEGIN TRANSACTION');
	try {
		await con.run(
			`UPDATE score_run
			    SET ${setClauses.join(', ')}
			  WHERE run_id=?
			    AND parent_run_id=?
			    AND COALESCE(status, 'running') NOT IN (${REPAIR_RERUN_TERMINAL_STATUS_SQL})`,
			[...params, input.child_run_id, input.parent_run_id]
		);
		const statusReader = await con.runAndReadAll(
			`SELECT COALESCE(status, 'unknown') AS status
			   FROM score_run
			  WHERE run_id=?
			    AND parent_run_id=?`,
			[input.child_run_id, input.parent_run_id]
		);
		const nextStatus = String(statusReader.getRowObjects()[0]?.status ?? 'unknown');
		await con.run('COMMIT');
		return {
			parent_run_id: input.parent_run_id,
			child_run_id: input.child_run_id,
			status: nextStatus,
			canceled: nextStatus === 'canceled' && status !== 'canceled',
			correction_ids: ids
		};
	} catch (e) {
		await con.run('ROLLBACK');
		throw e;
	}
}

export async function recordRepairRerunIntent(
	input: RecordRepairRerunInput
): Promise<RecordRepairRerunIntentResult> {
	if (!dbExists()) throw new Error('corpus DuckDB does not exist');
	const ids = normalizedCorrectionIds(input.correction_ids);
	if (ids.length === 0) throw new Error('correction_ids required');
	return await withRepairWriteConnection((con) => recordRepairRerunIntentOnConnection(con, input, ids));
}

export async function recordRepairRerunChildAfterScore(
	input: RecordRepairRerunInput
): Promise<RecordRepairRerunResult> {
	if (!dbExists()) throw new Error('corpus DuckDB does not exist');
	const ids = normalizedCorrectionIds(input.correction_ids);
	if (ids.length === 0) throw new Error('correction_ids required');
	return await withRepairWriteConnection((con) => recordRepairRerunChildOnConnection(con, input, ids));
}

export async function recordRepairRerunChild(
	input: RecordRepairRerunInput
): Promise<RecordRepairRerunResult> {
	if (!dbExists()) throw new Error('corpus DuckDB does not exist');
	const ids = normalizedCorrectionIds(input.correction_ids);
	if (ids.length === 0) throw new Error('correction_ids required');
	const activePair = activePairedWorkflowStates()[0] ?? null;
	if (activePair) {
		throw new Error(`paired workflow ${activePair.pair_id} is already ${activePair.status}; wait, cancel it, or inspect ${activePair.href}`);
	}
	const activeLock = activeWriterLock();
	if (activeLock) throw new Error(writerLockConflictText(activeLock));
	const writerLock = acquireWriterLock({
		kind: 'repair',
		label: 'repair rerun completion',
		source_dump_id: input.source_dump_id,
		dataset_path: dbPath(),
		architecture: input.architecture,
		pid: process.pid
	});
	if (!writerLock) {
		const lock = activeWriterLock();
		throw new Error(lock ? writerLockConflictText(lock) : 'DuckDB writer lock is busy; retry after the active worker finishes');
	}

	try {
		return await recordRepairRerunChildAfterScore(input);
	} finally {
		clearWriterLockToken(writerLock.token);
	}
}

export async function tombstoneStaleRepairRerunChild(
	input: RecordRepairRerunInput
): Promise<TombstoneRepairRerunChildResult> {
	if (!dbExists()) throw new Error('corpus DuckDB does not exist');
	const ids = normalizedCorrectionIds(input.correction_ids);
	if (ids.length === 0) throw new Error('correction_ids required');
	if (ids.length > MAX_RERUN_CORRECTION_IDS) {
		throw new Error(`at most ${MAX_RERUN_CORRECTION_IDS} repair candidates can be tombstoned at once`);
	}
	const activePair = activePairedWorkflowStates()[0] ?? null;
	if (activePair) {
		throw new Error(`paired workflow ${activePair.pair_id} is already ${activePair.status}; wait, cancel it, or inspect ${activePair.href}`);
	}
	const activeLock = activeWriterLock();
	if (activeLock) throw new Error(writerLockConflictText(activeLock));
	const writerLock = acquireWriterLock({
		kind: 'repair',
		label: 'stale repair child tombstone',
		source_dump_id: input.source_dump_id,
		dataset_path: dbPath(),
		architecture: input.architecture,
		pid: process.pid
	});
	if (!writerLock) {
		const lock = activeWriterLock();
		throw new Error(lock ? writerLockConflictText(lock) : 'DuckDB writer lock is busy; retry after the active worker finishes');
	}

	try {
		return await withRepairWriteConnection((con) => tombstoneRepairRerunChildOnConnection(con, input, ids));
	} finally {
		clearWriterLockToken(writerLock.token);
	}
}

export async function recordRepairRerunUncovered(
	input: RecordRepairRerunInput
): Promise<RecordRepairRerunUncoveredResult> {
	if (!dbExists()) throw new Error('corpus DuckDB does not exist');
	const ids = normalizedCorrectionIds(input.correction_ids);
	if (ids.length === 0) throw new Error('correction_ids required');
	const activePair = activePairedWorkflowStates()[0] ?? null;
	if (activePair) {
		throw new Error(`paired workflow ${activePair.pair_id} is already ${activePair.status}; wait, cancel it, or inspect ${activePair.href}`);
	}
	const activeLock = activeWriterLock();
	if (activeLock) throw new Error(writerLockConflictText(activeLock));
	const writerLock = acquireWriterLock({
		kind: 'repair',
		label: 'repair rerun uncovered release',
		source_dump_id: input.source_dump_id,
		dataset_path: dbPath(),
		architecture: input.architecture,
		pid: process.pid
	});
	if (!writerLock) {
		const lock = activeWriterLock();
		throw new Error(lock ? writerLockConflictText(lock) : 'DuckDB writer lock is busy; retry after the active worker finishes');
	}

	try {
		return await withRepairWriteConnection((con) => recordRepairRerunUncoveredOnConnection(con, input, ids));
	} finally {
		clearWriterLockToken(writerLock.token);
	}
}
