import { existsSync, mkdirSync, readdirSync, readFileSync, renameSync, statSync, unlinkSync, writeFileSync } from 'node:fs';
import { randomUUID } from 'node:crypto';
import { resolve } from 'node:path';
import { dbPath } from '$lib/db';

export type PairedArchitecture = 'monolithic' | 'decomposed';
export type PairedWorkflowStatus = 'queued' | 'running' | 'succeeded' | 'failed' | 'canceled';
const WRITER_LOCK_KINDS = ['paired_score', 'single_score', 'ingest', 'truth_set', 'repair'] as const;
export type WriterLockKind = typeof WRITER_LOCK_KINDS[number] | 'malformed';
export type PairedArchWorkflowStatus =
	| 'queued'
	| 'loading'
	| 'running'
	| 'succeeded'
	| 'failed'
	| 'canceled'
	| 'blocked';

export interface PairedArchWorkflowState {
	architecture: PairedArchitecture;
	status: PairedArchWorkflowStatus;
	pid: number | null;
	run_id: string | null;
	cost_threshold_usd: number;
	cost_so_far_usd: number | null;
	n_evidences_done: number;
	n_evidences_total: number | null;
	latest_stmt_hash: string | null;
	started_at: string | null;
	finished_at: string | null;
	duration_s: number | null;
	error: string | null;
	updated_at: string;
}

export interface PairedWorkflowState {
	pair_id: string;
	status: PairedWorkflowStatus;
	source_dump_id: string;
	dataset_path: string;
	model: string;
	scorer_version: string;
	total_cost_threshold_usd: number;
	href: string;
	created_at: string;
	updated_at: string;
	started_at: string | null;
	finished_at: string | null;
	termination_reason: string | null;
	architectures: Record<PairedArchitecture, PairedArchWorkflowState>;
}

export interface WriterLockState {
	kind: WriterLockKind;
	token: string;
	pid: number | null;
	label: string;
	source_dump_id: string | null;
	dataset_path: string | null;
	pair_id: string | null;
	architecture: PairedArchitecture | null;
	model: string | null;
	started_at: string;
	updated_at: string;
	malformed_reason: string | null;
}

export interface PublicWriterLockState {
	kind: WriterLockKind;
	pid: number | null;
	label: string;
	pair_id: string | null;
	architecture: PairedArchitecture | null;
	model: string | null;
	started_at: string;
	updated_at: string;
	malformed_reason: string | null;
}

export interface CreateWriterLockInput {
	kind: WriterLockKind;
	label: string;
	source_dump_id?: string | null;
	dataset_path?: string | null;
	pair_id?: string | null;
	architecture?: PairedArchitecture | null;
	model?: string | null;
	pid?: number | null;
	started_at?: string;
}

export interface CreatePairedWorkflowInput {
	pair_id: string;
	source_dump_id: string;
	dataset_path: string;
	model: string;
	scorer_version: string;
	total_cost_threshold_usd: number;
	caps: Record<PairedArchitecture, number>;
}

export type WriterLockConflictCode = 'writer_lock_busy' | 'writer_lock_malformed';

const PENDING_WRITER_LOCK_STALE_MS = 10 * 60 * 1000;
const PAIR_SIDECAR_STALE_MS = 10 * 60 * 1000;

function dataRoot(): string {
	return resolve(dbPath(), '..');
}

function stateRoot(): string {
	return resolve(dataRoot(), 'viewer_state', 'paired');
}

function writerLockPath(): string {
	return resolve(dataRoot(), 'viewer_state', 'writer_lock.json');
}

function statePath(pair_id: string): string {
	return resolve(stateRoot(), `${pair_id}.json`);
}

export function pairedWorkflowStateFileExists(pair_id: string): boolean {
	return existsSync(statePath(pair_id));
}

function nowIso(): string {
	return new Date().toISOString();
}

function ensureStateRoot(): void {
	mkdirSync(stateRoot(), { recursive: true });
	mkdirSync(resolve(dataRoot(), 'viewer_state'), { recursive: true });
}

function writeJsonAtomic(path: string, value: unknown): void {
	ensureStateRoot();
	const tmp = `${path}.tmp-${process.pid}-${Date.now()}`;
	writeFileSync(tmp, JSON.stringify(value, null, 2));
	renameSync(tmp, path);
}

function writeJsonExclusive(path: string, value: unknown): boolean {
	ensureStateRoot();
	try {
		writeFileSync(path, JSON.stringify(value, null, 2), { flag: 'wx' });
		return true;
	} catch (e) {
		const code = (e as NodeJS.ErrnoException).code;
		if (code === 'EEXIST') return false;
		throw e;
	}
}

function staleAgeMs(state: PairedWorkflowState): number | null {
	const updated = Date.parse(state.updated_at);
	if (!Number.isFinite(updated)) return null;
	return Date.now() - updated;
}

function processIsAlive(pid: number): boolean {
	if (!Number.isFinite(pid) || pid <= 0) return false;
	try {
		process.kill(pid, 0);
		return true;
	} catch (e) {
		return (e as NodeJS.ErrnoException).code === 'EPERM';
	}
}

function lockLooksStale(lock: WriterLockState): boolean {
	if (lock.kind === 'malformed') return false;
	if (lock.pid != null) return !processIsAlive(lock.pid);
	const updated = Date.parse(lock.updated_at);
	if (!Number.isFinite(updated)) return true;
	return Date.now() - updated > PENDING_WRITER_LOCK_STALE_MS;
}

function writerLockFileTime(path: string): string {
	try {
		return statSync(path).mtime.toISOString();
	} catch {
		return nowIso();
	}
}

function malformedWriterLock(path: string, reason: string): WriterLockState {
	const at = writerLockFileTime(path);
	return {
		kind: 'malformed',
		token: '__malformed_writer_lock__',
		pid: null,
		label: 'malformed writer lock',
		source_dump_id: null,
		dataset_path: null,
		pair_id: null,
		architecture: null,
		model: null,
		started_at: at,
		updated_at: at,
		malformed_reason: reason
	};
}

function validWriterLockKind(value: unknown): value is Exclude<WriterLockKind, 'malformed'> {
	return typeof value === 'string' && WRITER_LOCK_KINDS.includes(value as Exclude<WriterLockKind, 'malformed'>);
}

function validLockTimestamp(value: unknown): value is string {
	return typeof value === 'string' && value.length > 0 && Number.isFinite(Date.parse(value));
}

function validOptionalString(value: unknown): value is string | null | undefined {
	return value == null || typeof value === 'string';
}

function validOptionalPid(value: unknown): value is number | null | undefined {
	return value == null || (typeof value === 'number' && Number.isInteger(value) && value > 0);
}

function readWriterLockRaw(): WriterLockState | null {
	const path = writerLockPath();
	if (!existsSync(path)) return null;
	try {
		const raw = JSON.parse(readFileSync(path, 'utf-8')) as unknown;
		if (!raw || typeof raw !== 'object' || Array.isArray(raw)) {
			return malformedWriterLock(path, 'writer_lock.json must contain an object');
		}
		const parsed = raw as Partial<WriterLockState>;
		if (!validWriterLockKind(parsed.kind)) return malformedWriterLock(path, 'writer_lock.json has missing or invalid kind');
		if (typeof parsed.token !== 'string' || !parsed.token) return malformedWriterLock(path, 'writer_lock.json has missing token');
		if (!validLockTimestamp(parsed.started_at))
			return malformedWriterLock(path, 'writer_lock.json has missing or invalid started_at');
		if (!validLockTimestamp(parsed.updated_at))
			return malformedWriterLock(path, 'writer_lock.json has missing or invalid updated_at');
		if (!validOptionalPid(parsed.pid)) return malformedWriterLock(path, 'writer_lock.json has invalid pid');
		if (parsed.architecture != null && parsed.architecture !== 'monolithic' && parsed.architecture !== 'decomposed')
			return malformedWriterLock(path, 'writer_lock.json has invalid architecture');
		if (!validOptionalString(parsed.label)) return malformedWriterLock(path, 'writer_lock.json has invalid label');
		if (!validOptionalString(parsed.source_dump_id)) return malformedWriterLock(path, 'writer_lock.json has invalid source_dump_id');
		if (!validOptionalString(parsed.dataset_path)) return malformedWriterLock(path, 'writer_lock.json has invalid dataset_path');
		if (!validOptionalString(parsed.pair_id)) return malformedWriterLock(path, 'writer_lock.json has invalid pair_id');
		if (!validOptionalString(parsed.model)) return malformedWriterLock(path, 'writer_lock.json has invalid model');
		return {
			kind: parsed.kind,
			token: parsed.token,
			pid: typeof parsed.pid === 'number' ? parsed.pid : null,
			label: parsed.label ?? parsed.kind,
			source_dump_id: parsed.source_dump_id ?? null,
			dataset_path: parsed.dataset_path ?? null,
			pair_id: parsed.pair_id ?? null,
			architecture: parsed.architecture === 'monolithic' || parsed.architecture === 'decomposed' ? parsed.architecture : null,
			model: parsed.model ?? null,
			started_at: parsed.started_at,
			updated_at: parsed.updated_at,
			malformed_reason: null
		};
	} catch {
		return malformedWriterLock(path, 'writer_lock.json is unreadable JSON');
	}
}

function emptyArch(architecture: PairedArchitecture, cap: number, at: string): PairedArchWorkflowState {
	return {
		architecture,
		status: 'queued',
		pid: null,
		run_id: null,
		cost_threshold_usd: cap,
		cost_so_far_usd: null,
		n_evidences_done: 0,
		n_evidences_total: null,
		latest_stmt_hash: null,
		started_at: null,
		finished_at: null,
		duration_s: null,
		error: null,
		updated_at: at
	};
}

export function createPairedWorkflowState(input: CreatePairedWorkflowInput): PairedWorkflowState {
	ensureStateRoot();
	const path = statePath(input.pair_id);
	const at = nowIso();
	const state: PairedWorkflowState = {
		pair_id: input.pair_id,
		status: 'queued',
		source_dump_id: input.source_dump_id,
		dataset_path: input.dataset_path,
		model: input.model,
		scorer_version: input.scorer_version,
		total_cost_threshold_usd: input.total_cost_threshold_usd,
		href: `/pairs/${input.pair_id}`,
		created_at: at,
		updated_at: at,
		started_at: null,
		finished_at: null,
		termination_reason: null,
		architectures: {
			monolithic: emptyArch('monolithic', input.caps.monolithic, at),
			decomposed: emptyArch('decomposed', input.caps.decomposed, at)
		}
	};
	if (!writeJsonExclusive(path, state)) {
		throw new Error('paired workflow state already exists');
	}
	return state;
}

export function readPairedWorkflowState(pair_id: string): PairedWorkflowState | null {
	const path = statePath(pair_id);
	if (!existsSync(path)) return null;
	try {
		return JSON.parse(readFileSync(path, 'utf-8')) as PairedWorkflowState;
	} catch {
		return null;
	}
}

export function updatePairedWorkflowState(
	pair_id: string,
	update: (state: PairedWorkflowState) => PairedWorkflowState
): PairedWorkflowState | null {
	const current = readPairedWorkflowState(pair_id);
	if (!current) return null;
	const next = update({ ...current, architectures: { ...current.architectures } });
	next.updated_at = nowIso();
	writeJsonAtomic(statePath(pair_id), next);
	return next;
}

export function listPairedWorkflowStates(limit = 16): PairedWorkflowState[] {
	const root = stateRoot();
	if (!existsSync(root)) return [];
	const out: PairedWorkflowState[] = [];
	for (const name of readdirSync(root)) {
		if (!name.endsWith('.json')) continue;
		const id = name.slice(0, -'.json'.length);
		const state = readPairedWorkflowState(id);
		if (state) out.push(state);
	}
	out.sort((a, b) => b.updated_at.localeCompare(a.updated_at));
	return out.slice(0, limit);
}

function pairedWorkflowOpenState(state: PairedWorkflowState): boolean {
	if (state.status === 'succeeded' || state.status === 'failed' || state.status === 'canceled') {
		return false;
	}
	const archStates = Object.values(state.architectures);
	return (
		state.status === 'queued' ||
		state.status === 'running' ||
		archStates.some((a) => a.status === 'queued' || a.status === 'loading' || a.status === 'running')
	);
}

export function pairedWorkflowStaleReason(state: PairedWorkflowState): string | null {
	if (!pairedWorkflowOpenState(state)) return null;
	const archStates = Object.values(state.architectures);
	for (const arch of archStates) {
		if ((arch.status === 'loading' || arch.status === 'running') && arch.pid != null && processIsAlive(arch.pid)) {
			return null;
		}
	}
	const age = staleAgeMs(state);
	if (age == null) return 'paired workflow sidecar has invalid updated_at';
	if (age <= PAIR_SIDECAR_STALE_MS) return null;
	const ageMin = Math.floor(age / 60000);
	return `paired workflow sidecar stale for ${ageMin}m with no live worker pid`;
}

export function reconcileStalePairedWorkflowState(pair_id: string): PairedWorkflowState | null {
	const staleState = readPairedWorkflowState(pair_id);
	if (!staleState) return null;
	const reason = pairedWorkflowStaleReason(staleState);
	if (!reason) return staleState;
	const finished = nowIso();
	return updatePairedWorkflowState(pair_id, (state) => {
		const architectures = { ...state.architectures };
		for (const architecture of Object.keys(architectures) as PairedArchitecture[]) {
			const current = architectures[architecture];
			if (current.status === 'succeeded' || current.status === 'failed' || current.status === 'canceled' || current.status === 'blocked') {
				continue;
			}
			const wasRunning = current.status === 'loading' || current.status === 'running';
			architectures[architecture] = {
				...current,
				status: wasRunning ? 'failed' : 'blocked',
				pid: null,
				finished_at: current.finished_at ?? finished,
				error: wasRunning
					? `${reason}; marked failed without mutating any scorer output`
					: `${reason}; queued architecture never started`,
				updated_at: finished
			};
		}
		return {
			...state,
			status: 'failed',
			finished_at: state.finished_at ?? finished,
			termination_reason: reason,
			architectures
		};
	});
}

export function reconcileStalePairedWorkflowStates(): PairedWorkflowState[] {
	return listPairedWorkflowStates(Infinity)
		.map((state) => reconcileStalePairedWorkflowState(state.pair_id) ?? state);
}

export function pairedWorkflowIsActive(state: PairedWorkflowState): boolean {
	if (!pairedWorkflowOpenState(state)) return false;
	const archStates = Object.values(state.architectures);
	for (const arch of archStates) {
		if ((arch.status === 'loading' || arch.status === 'running') && arch.pid != null && processIsAlive(arch.pid)) {
			return true;
		}
	}
	const age = staleAgeMs(state);
	return age != null && age <= PAIR_SIDECAR_STALE_MS;
}

export function activePairedWorkflowStates(): PairedWorkflowState[] {
	return reconcileStalePairedWorkflowStates().filter((s) => pairedWorkflowIsActive(s));
}

export function readWriterLock(): WriterLockState | null {
	return readWriterLockRaw();
}

export function activeWriterLock(): WriterLockState | null {
	const lock = readWriterLockRaw();
	if (!lock || lockLooksStale(lock)) return null;
	return lock;
}

export function publicWriterLockSnapshot(lock: WriterLockState | null): PublicWriterLockState | null {
	if (!lock) return null;
	return {
		kind: lock.kind,
		pid: lock.pid,
		label: lock.label,
		pair_id: lock.pair_id,
		architecture: lock.architecture,
		model: lock.model,
		started_at: lock.started_at,
		updated_at: lock.updated_at,
		malformed_reason: lock.malformed_reason
	};
}

export function activePublicWriterLock(): PublicWriterLockState | null {
	return publicWriterLockSnapshot(activeWriterLock());
}

export function gcWriterLock(): void {
	const lock = readWriterLockRaw();
	if (!lock) return;
	if (lockLooksStale(lock)) {
		try { unlinkSync(writerLockPath()); } catch { /* ignore */ }
	}
}

export function writerLockConflictText(lock: WriterLockState): string {
	if (lock.kind === 'malformed') {
		const detail = lock.malformed_reason ? ` (${lock.malformed_reason})` : '';
		return `writer_lock_malformed: DuckDB writer lock state is malformed${detail}; reads and writes pause until viewer_state/writer_lock.json is repaired or removed only after confirming no Python or Node writer is active`;
	}
	const arch = lock.architecture ? ` ${lock.architecture}` : '';
	const pair = lock.pair_id ? ` pair ${lock.pair_id}` : '';
	const pid = lock.pid != null ? ` pid ${lock.pid}` : '';
	return `DuckDB writer is busy with ${lock.kind}${arch}${pair}${pid}; wait for it to finish or cancel it from the dashboard`;
}

export function writerLockConflictCode(lock: WriterLockState | null): WriterLockConflictCode {
	return lock?.kind === 'malformed' || lock?.malformed_reason ? 'writer_lock_malformed' : 'writer_lock_busy';
}

export function writerLockConflictPayload(lock: WriterLockState | null): { code: WriterLockConflictCode; message: string } {
	return {
		code: writerLockConflictCode(lock),
		message: lock
			? writerLockConflictText(lock)
			: 'DuckDB writer lock is busy; retry after the active worker finishes'
	};
}

export function acquireWriterLock(input: CreateWriterLockInput): WriterLockState | null {
	gcWriterLock();
	const existing = activeWriterLock();
	if (existing) return null;
	const at = input.started_at ?? nowIso();
	const lock: WriterLockState = {
		kind: input.kind,
		token: randomUUID(),
		pid: input.pid ?? null,
		label: input.label,
		source_dump_id: input.source_dump_id ?? null,
		dataset_path: input.dataset_path ?? null,
		pair_id: input.pair_id ?? null,
		architecture: input.architecture ?? null,
		model: input.model ?? null,
		started_at: at,
		updated_at: at,
		malformed_reason: null
	};
	return writeJsonExclusive(writerLockPath(), lock) ? lock : null;
}

export function updateWriterLock(token: string, patch: Partial<Omit<WriterLockState, 'token'>>): WriterLockState | null {
	const current = readWriterLockRaw();
	if (!current || current.kind === 'malformed' || current.token !== token) return null;
	const next: WriterLockState = {
		...current,
		...patch,
		token,
		updated_at: nowIso()
	};
	writeJsonAtomic(writerLockPath(), next);
	return next;
}

export function clearWriterLockToken(token: string): void {
	const path = writerLockPath();
	if (!existsSync(path)) return;
	try {
		const existing = readWriterLockRaw();
		if (existing && existing.kind !== 'malformed' && existing.token === token) unlinkSync(path);
	} catch {
		// Ambiguous lock state must fail closed; do not delete it here.
	}
}

export function clearWriterLock(pair_id: string, architecture: PairedArchitecture, pid: number | null): void {
	const path = writerLockPath();
	if (!existsSync(path)) return;
	try {
		const existing = readWriterLockRaw();
		if (
			existing?.kind === 'paired_score' &&
			existing.pair_id === pair_id &&
			existing.architecture === architecture &&
			(pid == null || existing.pid === pid)
		) {
			unlinkSync(path);
		}
	} catch {
		// Ambiguous lock state must fail closed; do not delete it here.
	}
}
