import { DuckDBInstance, type DuckDBConnection } from '@duckdb/node-api';
import { closeInstance, dbExists, dbPath } from '$lib/db';
import { repairRerunParentCorrectionIdNumberSql } from '$lib/repairRerunSql';
import {
	activePairedWorkflowStates,
	activeWriterLock,
	acquireWriterLock,
	clearWriterLockToken,
	writerLockConflictText
} from '$lib/server/pairedState';
import { CORRECTION_DDL } from '$lib/server/repairBacklog';

export const PROBE_REPAIR_SLOTS = [
	'substrate_route',
	'subject_role_probe',
	'object_role_probe',
	'relation_axis_probe',
	'scope_probe'
] as const;

export type ProbeRepairSlot = typeof PROBE_REPAIR_SLOTS[number];

export interface RecordProbeSlotReviewInput {
	run_id: string;
	correction_id: number;
	selected_slots: string[];
	note?: string | null;
	reviewer?: string | null;
}

export interface RecordProbeSlotReviewResult {
	run_id: string;
	correction_id: number;
	recorded: number;
	selected_slots: ProbeRepairSlot[];
	note: string | null;
}

function parseSlotList(raw: string | null | undefined): ProbeRepairSlot[] {
	if (!raw) return [];
	const allowed = new Set<string>(PROBE_REPAIR_SLOTS);
	const out: ProbeRepairSlot[] = [];
	for (const slot of raw.split(',').map((s) => s.trim()).filter(Boolean)) {
		if (allowed.has(slot) && !out.includes(slot as ProbeRepairSlot)) out.push(slot as ProbeRepairSlot);
	}
	return out;
}

function normalizeSelectedSlots(slots: string[]): ProbeRepairSlot[] {
	const selected = new Set(slots.map((slot) => String(slot).trim()).filter(Boolean));
	return PROBE_REPAIR_SLOTS.filter((slot) => selected.has(slot));
}

async function openWriteConnection<T>(fn: (con: DuckDBConnection) => Promise<T>): Promise<T> {
	let instance: DuckDBInstance | null = null;
	let con: DuckDBConnection | null = null;
	try {
		closeInstance();
		instance = await DuckDBInstance.create(dbPath());
		con = await instance.connect();
		return await fn(con);
	} finally {
		try {
			con?.disconnectSync?.();
		} finally {
			instance?.closeSync();
		}
	}
}

export async function recordProbeSlotReview(
	input: RecordProbeSlotReviewInput
): Promise<RecordProbeSlotReviewResult> {
	if (!dbExists()) throw new Error('corpus DuckDB does not exist');
	const correctionId = Number(input.correction_id);
	if (!Number.isInteger(correctionId) || correctionId <= 0) {
		throw new Error('correction_id must be a positive integer');
	}
	const selectedSlots = normalizeSelectedSlots(input.selected_slots);
	if (selectedSlots.length === 0) {
		throw new Error('select at least one missing probe slot');
	}
	const activePair = activePairedWorkflowStates()[0] ?? null;
	if (activePair) {
		throw new Error(`paired workflow ${activePair.pair_id} is already ${activePair.status}; wait, cancel it, or inspect ${activePair.href}`);
	}
	const activeLock = activeWriterLock();
	if (activeLock) throw new Error(writerLockConflictText(activeLock));
	const writerLock = acquireWriterLock({
		kind: 'repair',
		label: 'probe slot review',
		source_dump_id: null,
		dataset_path: dbPath(),
		pid: process.pid
	});
	if (!writerLock) {
		const lock = activeWriterLock();
		throw new Error(lock ? writerLockConflictText(lock) : 'DuckDB writer lock is busy; retry after the active worker finishes');
	}

	try {
		return await openWriteConnection(async (con) => {
			await con.run(CORRECTION_DDL);
			const reader = await con.runAndReadAll(
				`SELECT
				   c.correction_id,
				   c.step_hash,
				   c.run_id,
				   COALESCE(c.architecture, sr.architecture, 'unknown') AS architecture,
				   c.stmt_hash,
				   c.evidence_hash,
				   c.correction_kind,
				   c.status,
				   c.source_route,
				   CAST(c.source_filters_json AS VARCHAR) AS source_filters_json,
				   json_extract_string(c.value_json, '$.observed.probe_coverage') AS probe_coverage,
				   json_extract_string(c.value_json, '$.observed.missing_probe_slots') AS missing_probe_slots
				 FROM scorer_step_correction c
				 LEFT JOIN score_run sr ON sr.run_id=c.run_id
				 WHERE c.run_id=?
				   AND c.correction_id=?
				 LIMIT 1`,
				[input.run_id, correctionId]
			);
			const candidate = reader.getRowObjects()[0];
			if (!candidate) throw new Error('repair candidate not found');
			if (String(candidate.correction_kind) !== 'repair_candidate') {
				throw new Error('correction_id is not a repair candidate');
			}
			if (String(candidate.status) !== 'open') {
				throw new Error('probe slot review requires an open repair candidate');
			}
			if (String(candidate.probe_coverage ?? '') !== 'present') {
				throw new Error('probe slot review requires a probe coverage repair candidate');
			}
			const missingSlots = parseSlotList(candidate.missing_probe_slots == null ? null : String(candidate.missing_probe_slots));
			const missingSet = new Set(missingSlots);
			const invalid = selectedSlots.filter((slot) => !missingSet.has(slot));
			if (invalid.length > 0) {
				throw new Error(`selected slots are not missing for this repair candidate: ${invalid.join(', ')}`);
			}
			const note = input.note ? input.note.trim().slice(0, 1000) : null;
			const reviewer = input.reviewer ? input.reviewer.trim().slice(0, 120) : null;
			const valueJson = JSON.stringify({
				kind: 'probe_slot_review',
				parent_correction_id: correctionId,
				selected_probe_slots: selectedSlots.join(','),
				selected_probe_slot_list: selectedSlots,
				missing_probe_slots: missingSlots.join(','),
				reviewer_note: note
			});
			await con.run(
				`INSERT INTO scorer_step_correction
				 (step_hash, run_id, architecture, stmt_hash, evidence_hash,
				  correction_kind, status, reviewer, note, value_json,
				  parent_correction_id, source_route, source_filters_json)
				 VALUES (?, ?, ?, ?, ?, 'probe_slot_review', 'recorded', ?, ?, ?::JSON, ?, ?, ?::JSON)`,
				[
					String(candidate.step_hash),
					String(candidate.run_id),
					String(candidate.architecture),
					String(candidate.stmt_hash),
					candidate.evidence_hash == null ? null : String(candidate.evidence_hash),
					reviewer,
					note,
					valueJson,
					correctionId,
					candidate.source_route == null ? null : String(candidate.source_route),
					candidate.source_filters_json == null ? null : String(candidate.source_filters_json)
				]
			);
			const countReader = await con.runAndReadAll(
				`SELECT COUNT(*) AS n
				   FROM scorer_step_correction r
				  WHERE r.run_id=?
				    AND r.correction_kind='probe_slot_review'
				    AND ${repairRerunParentCorrectionIdNumberSql('r', { typedLineage: true })}=?`,
				[input.run_id, correctionId]
			);
			return {
				run_id: input.run_id,
				correction_id: correctionId,
				recorded: Number(countReader.getRowObjects()[0]?.n ?? 0),
				selected_slots: selectedSlots,
				note
			};
		});
	} finally {
		clearWriterLockToken(writerLock.token);
	}
}
