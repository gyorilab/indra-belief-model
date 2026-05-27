<script lang="ts">
	import { goto, invalidateAll } from '$app/navigation';
	import { onMount } from 'svelte';
	import type { PageData } from './$types';
	import { fmtBelief, fmtDelta, shortHash } from '$lib/format';

	let { data }: { data: PageData } = $props();
	const b = $derived(data.backlog);
	type ScoreArchitecture = 'decomposed' | 'monolithic';
	type CostEstimate = {
		model_id: string;
		architecture: ScoreArchitecture;
		cost_usd: number;
		n_stmts: number;
		n_evidences_est: number;
		n_llm_calls_est: number;
		scoring_mode?: 'aggregate' | 'probe_only';
		probe_step_filter?: string[];
	};
			type RepairCorpus = {
			parent_run_id: string;
			architecture: ScoreArchitecture;
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
			dropped_correction_ids: Array<{ correction_id: number; reason: string }>;
				n_selected_evidence_candidates: number;
				n_statement_scope_candidates: number;
				n_scope_expansion_evidences: number;
				n_collateral_evidences: number;
				n_probe_slot_reviewed_candidates: number;
				probe_slot_reviews: Array<{
					correction_id: number;
					selected_slots: string[];
					note: string | null;
					reviewed_at: string | null;
					review_count: number;
				}>;
				probe_slot_counts: Record<string, number>;
				scoring_mode: 'aggregate' | 'probe_only';
				probe_step_filter: string[];
				max_correction_ids: number;
				truncated: boolean;
			};
			type RepairCompletionRecovery = {
				parent_run_id: string;
				child_run_id: string;
				architecture: ScoreArchitecture;
				source_dump_id: string;
				correction_ids: number[];
			};
			type RecoverableRerunIntent = {
				child_run_id: string;
				architecture: string;
				status: string;
				source_dump_id: string;
				correction_ids: number[];
				uncovered_correction_ids: number[];
				n_candidates: number;
				n_missing_markers: number;
				n_uncovered_candidates: number;
				first_intent_at: string;
				last_intent_at: string;
			};
			type ActiveRerunIntent = {
				child_run_id: string;
				architecture: string;
				status: string;
				source_dump_id: string;
				correction_ids: number[];
				n_candidates: number;
				has_child_run: boolean;
				first_intent_at: string;
				last_intent_at: string;
			};
			type RecoveredCompletionState =
				| { phase: 'working' }
				| { phase: 'error'; message: string };
			type ActiveTombstoneState =
				| { phase: 'working' }
				| { phase: 'error'; message: string };
			type ProbeSlotReviewState =
				| { phase: 'working' }
				| { phase: 'error'; message: string };
			type RepairWriterLock = NonNullable<PageData['writerLock']>;
				type RerunState =
					| { phase: 'idle' }
					| { phase: 'estimating' }
					| { phase: 'estimated'; architecture: ScoreArchitecture; corpus: RepairCorpus; estimates: CostEstimate[]; scorer_version: string; selection_key: string }
					| { phase: 'scoring'; architecture: ScoreArchitecture; corpus: RepairCorpus; model: string; run_id: string | null; scorer_version: string; cost_cap_usd: number; cost_so_far_usd: number | null; n_evidences_done: number; n_evidences_total: number | null; latest_stmt: string | null }
					| { phase: 'done'; architecture: ScoreArchitecture; corpus: RepairCorpus; run_id: string; model: string; n_evidences: number; duration_s: number | null; completion_recorded: number | null; completion_error: string | null; collateral_recorded: number | null; recovery: RepairCompletionRecovery | null; completion_retrying?: boolean }
					| { phase: 'error'; message: string; run_id?: string | null };

		const selectedRerunArchitecture = $derived(b.architecture === 'monolithic' ? 'monolithic' : 'decomposed');
		let capMultiplier = $state(1.25);
		let repairNote = $state('');
		let rerunState = $state<RerunState>({ phase: 'idle' });
		let selectedRepairOverride = $state<number[] | null>(null);
		let recoveredCompletionState = $state<Record<string, RecoveredCompletionState>>({});
		let activeTombstoneState = $state<Record<string, ActiveTombstoneState>>({});
		let probeSlotReviewState = $state<Record<number, ProbeSlotReviewState>>({});
		let liveWriterLock = $state<RepairWriterLock | null>(null);
		let clientReady = $state(false);
		let writerLockPollReady = $state(false);
		let writerLockRefreshError = $state<string | null>(null);
		let estimateRequestSerial = 0;
		let rerunAbortController: AbortController | null = null;
		const effectiveWriterLock = $derived(
			writerLockPollReady ? liveWriterLock : (data.writerLock ?? liveWriterLock)
		);
		const writerLockBusy = $derived(effectiveWriterLock != null || writerLockRefreshError != null);
		const visibleRepairIds = $derived(b.rows.map((r) => r.correction_id));
		const selectedRepairIds = $derived(selectedRepairOverride ?? visibleRepairIds);
		const selectedRepairCount = $derived(selectedRepairIds.length);
		const visibleRepairCount = $derived(visibleRepairIds.length);
		const allVisibleSelected = $derived(
			visibleRepairCount > 0 && selectedRepairIds.length === visibleRepairCount
		);
		const preflightScopeLocked = $derived(
			rerunState.phase === 'estimating' || rerunState.phase === 'estimated' || rerunState.phase === 'scoring'
		);
		const recoveryPageStart = $derived(
			b.recoverableRerunIntentCount === 0 ? 0 : Math.min(b.recoverableRerunIntentOffset + 1, b.recoverableRerunIntentCount)
		);
		const recoveryPageEnd = $derived(
			Math.min(b.recoverableRerunIntentOffset + b.recoverableRerunIntents.length, b.recoverableRerunIntentCount)
		);
		const previousRecoveryOffset = $derived(
			Math.max(0, b.recoverableRerunIntentOffset - b.recoverableRerunIntentLimit)
		);
		const nextRecoveryOffset = $derived(
			b.recoverableRerunIntentOffset + b.recoverableRerunIntentLimit
		);
		const normalizedCapMultiplier = $derived(clampCapMultiplier(capMultiplier));
		const repairLineageToken = $derived(
			repairScorerVersion(b.run_id, selectedRerunArchitecture, selectedRepairIds, repairNote)
		);

		$effect(() => {
			if (selectedRepairOverride == null || rerunState.phase === 'scoring') return;
			const visible = new Set(visibleRepairIds);
			const next = selectedRepairOverride.filter((id) => visible.has(id));
			if (next.length !== selectedRepairOverride.length || next.some((id, i) => id !== selectedRepairOverride?.[i])) {
				selectedRepairOverride = next;
			}
		});

		$effect(() => {
			liveWriterLock = data.writerLock ?? null;
			writerLockPollReady = false;
			writerLockRefreshError = null;
		});

		onMount(() => {
			clientReady = true;
			let stopped = false;
			let inFlight = false;
			let refreshController: AbortController | null = null;
			async function refreshWriterLock() {
				if (stopped || inFlight || b.activeRerunIntents.length === 0) return;
				inFlight = true;
				const ctrl = new AbortController();
				refreshController = ctrl;
				const timeout = setTimeout(() => ctrl.abort(), 2000);
				try {
					const res = await fetch('/api/writer-lock', {
						headers: { accept: 'application/json' },
						cache: 'no-store',
						signal: ctrl.signal
					});
					if (!res.ok) {
						throw new Error(`writer lock snapshot failed: ${res.status}`);
					}
					const body = await res.json() as { writerLock?: RepairWriterLock | null };
					if (stopped) return;
					liveWriterLock = body.writerLock ?? null;
					writerLockPollReady = true;
					writerLockRefreshError = null;
				} catch (e) {
					if (!stopped) writerLockRefreshError = String(e).slice(0, 120);
				} finally {
					clearTimeout(timeout);
					if (refreshController === ctrl) refreshController = null;
					inFlight = false;
				}
			}
			void refreshWriterLock();
			const h = setInterval(() => { void refreshWriterLock(); }, 3000);
			return () => {
				stopped = true;
				refreshController?.abort();
				clearInterval(h);
			};
		});

	function archMark(architecture: string | null | undefined): string {
		if (architecture === 'monolithic') return '[M]';
		if (architecture === 'decomposed') return '[D]';
		return '[?]';
	}

	function isRepairArchitecture(architecture: string | null | undefined): architecture is ScoreArchitecture {
		return architecture === 'decomposed' || architecture === 'monolithic';
	}

	function writerLockKindText(kind: string): string {
		if (kind === 'single_score') return 'single score';
		if (kind === 'paired_score') return 'paired score';
		if (kind === 'truth_set') return 'truth set';
		return kind;
	}

	function writerLockText(lock: RepairWriterLock): string {
		const bits = [
			writerLockKindText(lock.kind),
			lock.label && lock.label !== lock.kind ? lock.label : null,
			lock.architecture ? archMark(lock.architecture) : null,
			lock.model,
			lock.pair_id ? `pair ${lock.pair_id}` : null,
			lock.pid != null ? `pid ${lock.pid}` : null
		].filter(Boolean);
		return bits.join(' · ');
	}

	function writerLockActionBlockText(action: string): string {
		if (effectiveWriterLock) {
			return `writer is active: ${writerLockText(effectiveWriterLock)}; wait before ${action}.`;
		}
		return `writer lock status is unavailable; retrying before ${action}.`;
	}

	function fmtCost(c: number | null): string {
		if (c == null) return '—';
		if (c < 0.01) return '<$0.01';
		if (c < 100) return '$' + c.toFixed(2);
		return '$' + c.toFixed(0);
	}

	function fmtMetric(n: number | null): string {
		return n == null || Number.isNaN(n) ? '—' : n.toFixed(3);
	}

	function fmtTimestamp(value: string | null | undefined): string {
		return value ? value.replace(/\.\d+$/, '') : 'unknown start';
	}

	function clampCapMultiplier(value: number | string): number {
		return Math.min(3, Math.max(1, Number(value) || 1));
	}

	function updateCapMultiplier(ev: Event) {
		capMultiplier = clampCapMultiplier((ev.currentTarget as HTMLInputElement).value);
	}

	function selectionKey(ids: number[]): string {
		return [...ids].sort((a, b) => a - b).join(',');
	}

	function selectionFingerprint(ids: number[]): string {
		let h = 2166136261;
		for (const id of [...ids].sort((a, b) => a - b)) {
			for (const ch of String(id)) {
				h ^= ch.charCodeAt(0);
				h = Math.imul(h, 16777619);
			}
			h ^= 44;
			h = Math.imul(h, 16777619);
		}
		return (h >>> 0).toString(36).slice(0, 6);
	}

	function noteSlug(note: string): string {
		return note
			.trim()
			.toLowerCase()
			.replace(/[^a-z0-9]+/g, '-')
			.replace(/^-+|-+$/g, '')
			.slice(0, 24);
	}

	function repairScorerVersion(
		parentRunId: string,
		architecture: ScoreArchitecture,
		correctionIds: number[],
		note: string
	): string {
		const base = `repair-${parentRunId.slice(0, 8)}-${architecture.slice(0, 4)}-${correctionIds.length}c-${selectionFingerprint(correctionIds)}`;
		const suffix = noteSlug(note);
		return suffix ? `${base}-${suffix}` : base;
	}

	function capForEstimate(cost: number): number {
		return Math.max(0.01, cost * normalizedCapMultiplier);
	}

	function fmtSigned(n: number | null): string {
		if (n == null || Number.isNaN(n)) return '—';
		const sign = n >= 0 ? '+' : '−';
		return `${sign}${Math.abs(n).toFixed(3)}`;
	}

	function movementLabel(movement: string): string {
		if (movement === 'new_child_aggregate') return 'new child aggregate';
		if (movement === 'verdict_to_correct') return 'verdict to correct';
		if (movement === 'verdict_to_incorrect') return 'verdict to incorrect';
		if (movement === 'score_improved') return 'score improved';
		if (movement === 'score_regressed') return 'score regressed';
		return 'unchanged';
	}

	function safeRoute(route: string | null): string | null {
		if (!route || !route.startsWith('/')) return null;
		return route;
	}

	function filterPairs(raw: string | null): Array<[string, string]> {
		if (!raw) return [];
		try {
			const parsed = JSON.parse(raw) as Record<string, unknown>;
			return Object.entries(parsed)
				.filter(([, v]) => v !== null && v !== undefined && v !== false && v !== '')
				.map(([k, v]) => [k, String(v)]);
		} catch {
			return [['filters', raw]];
		}
	}

	function missingProbeSlots(raw: string | null | undefined): string[] {
		if (!raw) return [];
		return raw.split(',').map((slot) => slot.trim()).filter(Boolean);
	}

	function reviewedProbeSlots(r: PageData['backlog']['rows'][number]): string[] {
		return missingProbeSlots(r.probe_slot_review_slots);
	}

	function probeSlotCountPairs(corpus: RepairCorpus): Array<[string, number]> {
		return Object.entries(corpus.probe_slot_counts ?? {})
			.filter(([, n]) => Number(n) > 0)
			.sort(([a], [b]) => a.localeCompare(b));
	}

	function probeStepFilterForCorpus(corpus: RepairCorpus): string[] {
		return corpus.probe_step_filter?.length ? corpus.probe_step_filter : probeSlotCountPairs(corpus).map(([slot]) => slot);
	}

	function probeOnlyForCorpus(corpus: RepairCorpus): boolean {
		return corpus.scoring_mode === 'probe_only' && probeStepFilterForCorpus(corpus).length > 0;
	}

	function defaultProbeSlotChecked(r: PageData['backlog']['rows'][number], slot: string): boolean {
		const reviewed = reviewedProbeSlots(r);
		return reviewed.length > 0 ? reviewed.includes(slot) : missingProbeSlots(r.missing_probe_slots).includes(slot);
	}

	function probeCountPairs(r: PageData['backlog']['rows'][number]): Array<[string, number | null]> {
		return [
			['substrate_route', r.n_substrate_route],
			['subject_role_probe', r.n_subject_role_probe],
			['object_role_probe', r.n_object_role_probe],
			['relation_axis_probe', r.n_relation_axis_probe],
			['scope_probe', r.n_scope_probe]
		];
	}

	function setProbeSlotReviewState(id: number, state: ProbeSlotReviewState) {
		probeSlotReviewState = { ...probeSlotReviewState, [id]: state };
	}

	function clearProbeSlotReviewState(id: number) {
		const next = { ...probeSlotReviewState };
		delete next[id];
		probeSlotReviewState = next;
	}

	function probeSlotReviewError(id: number): string | null {
		const state = probeSlotReviewState[id];
		return state?.phase === 'error' ? state.message : null;
	}

	async function saveProbeSlotReview(ev: SubmitEvent, r: PageData['backlog']['rows'][number]) {
		ev.preventDefault();
		if (writerLockBusy) {
			setProbeSlotReviewState(r.correction_id, { phase: 'error', message: writerLockActionBlockText('recording probe slot review') });
			return;
		}
		const form = ev.currentTarget as HTMLFormElement;
		const data = new FormData(form);
		const selectedSlots = data.getAll('probe_slot').map((slot) => String(slot));
		const note = String(data.get('probe_slot_note') ?? '').slice(0, 1000);
		setProbeSlotReviewState(r.correction_id, { phase: 'working' });
		try {
			const res = await fetch('/api/repairs/probe-slots', {
				method: 'POST',
				headers: { 'content-type': 'application/json' },
				body: JSON.stringify({
					run_id: b.run_id,
					correction_id: r.correction_id,
					selected_slots: selectedSlots,
					note,
					reviewer: 'viewer'
				})
			});
			const body = await res.json();
			if (res.ok) {
				await invalidateAll();
				clearProbeSlotReviewState(r.correction_id);
				return;
			}
			setProbeSlotReviewState(r.correction_id, {
				phase: 'error',
				message: body?.message ?? 'probe slot review failed'
			});
		} catch (e) {
			setProbeSlotReviewState(r.correction_id, {
				phase: 'error',
				message: String(e).slice(0, 160)
			});
		}
	}

	function toggleRepairId(id: number) {
		estimateRequestSerial += 1;
		if (selectedRepairIds.includes(id)) {
			selectedRepairOverride = selectedRepairIds.filter((x) => x !== id);
		} else {
			selectedRepairOverride = [...selectedRepairIds, id];
		}
		rerunState = { phase: 'idle' };
	}

	function selectAllVisible() {
		estimateRequestSerial += 1;
		selectedRepairOverride = visibleRepairIds;
		rerunState = { phase: 'idle' };
	}

	function clearSelection() {
		estimateRequestSerial += 1;
		selectedRepairOverride = [];
		rerunState = { phase: 'idle' };
	}

	function reviseRepairPreflight() {
		estimateRequestSerial += 1;
		rerunState = { phase: 'idle' };
	}

	function recoveryIntentKey(intent: Pick<RecoverableRerunIntent, 'child_run_id' | 'source_dump_id'>): string {
		return `${intent.child_run_id}:${intent.source_dump_id}`;
	}

	function activeIntentKey(intent: Pick<ActiveRerunIntent, 'child_run_id' | 'source_dump_id'>): string {
		return `${intent.child_run_id}:${intent.source_dump_id}`;
	}

	function recoveryPageHref(offset: number): string {
		return offset > 0
			? `/runs/${b.run_id}/repairs?recovery_offset=${offset}`
			: `/runs/${b.run_id}/repairs`;
	}

	async function returnToRecoveryQueueFront() {
		if (b.recoverableRerunIntentOffset > 0) {
			await goto(recoveryPageHref(0), { invalidateAll: true });
			return;
		}
		await invalidateAll();
	}

	function setRecoveredCompletionState(key: string, state: RecoveredCompletionState) {
		recoveredCompletionState = { ...recoveredCompletionState, [key]: state };
	}

	function setActiveTombstoneState(key: string, state: ActiveTombstoneState) {
		activeTombstoneState = { ...activeTombstoneState, [key]: state };
	}

	function delay(ms: number): Promise<void> {
		return new Promise((resolve) => setTimeout(resolve, ms));
	}

	async function waitForWriterLockRelease(timeoutMs = 5000): Promise<boolean> {
		const deadline = Date.now() + timeoutMs;
		while (Date.now() < deadline) {
			try {
				const res = await fetch('/api/writer-lock', {
					headers: { accept: 'application/json' },
					cache: 'no-store'
				});
				if (res.ok) {
					const body = await res.json() as { writerLock?: RepairWriterLock | null };
					liveWriterLock = body.writerLock ?? null;
					writerLockPollReady = true;
					writerLockRefreshError = null;
					if (!body.writerLock) return true;
				}
			} catch (e) {
				writerLockRefreshError = String(e).slice(0, 120);
			}
			await delay(150);
		}
		return false;
	}

	async function tombstoneActiveRerun(intent: ActiveRerunIntent) {
		const key = activeIntentKey(intent);
		if (!isRepairArchitecture(intent.architecture)) {
			setActiveTombstoneState(key, { phase: 'error', message: 'child architecture is not repairable' });
			return;
		}
		setActiveTombstoneState(key, { phase: 'working' });
		try {
			const tombstoneRes = await fetch('/api/repairs/rerun/tombstone', {
				method: 'POST',
				headers: { 'content-type': 'application/json' },
				body: JSON.stringify({
					parent_run_id: b.run_id,
					child_run_id: intent.child_run_id,
					architecture: intent.architecture,
					source_dump_id: intent.source_dump_id,
					correction_ids: intent.correction_ids
				})
			});
			const tombstoneBody = await tombstoneRes.json();
			if (tombstoneRes.ok) {
				await invalidateAll();
				return;
			}
			setActiveTombstoneState(key, {
				phase: 'error',
				message: tombstoneBody?.message ?? 'child cancel marker failed'
			});
		} catch (e) {
			setActiveTombstoneState(key, {
				phase: 'error',
				message: String(e).slice(0, 160)
			});
		}
	}

		async function cancelRepairRerun() {
			const scoring = rerunState.phase === 'scoring' ? rerunState : null;
			if (rerunAbortController) {
				rerunAbortController.abort();
				rerunAbortController = null;
			}
			if (scoring) {
				rerunState = {
					phase: 'error',
					message: `cancel requested after ${scoring.n_evidences_done}/${scoring.n_evidences_total ?? '—'} evidences; the child run will be marked canceled when the worker exits cleanly, and no repair candidates were marked rerun-complete`,
					run_id: scoring.run_id
				};
				if (await waitForWriterLockRelease()) {
					await delay(250);
					await invalidateAll();
					await delay(250);
					await invalidateAll();
				}
			}
		}

		async function estimateRepairRerun() {
			if (selectedRepairIds.length === 0) {
				rerunState = { phase: 'error', message: 'select at least one repair candidate before previewing rerun cost' };
				return;
			}
			if (writerLockBusy) {
				rerunState = { phase: 'error', message: writerLockActionBlockText('previewing repair rerun cost') };
				return;
			}
			const requestedSelectionKey = selectionKey(selectedRepairIds);
			const requestedLineageToken = repairLineageToken;
			const requestSerial = ++estimateRequestSerial;
			rerunState = { phase: 'estimating' };
			try {
				const res = await fetch('/api/repairs/rerun/estimate', {
				method: 'POST',
				headers: { 'content-type': 'application/json' },
				body: JSON.stringify({
					run_id: b.run_id,
					arch: selectedRerunArchitecture,
					correction_ids: selectedRepairIds
				})
			});
			const body = await res.json();
				if (!res.ok || body?.ok === false) {
					rerunState = { phase: 'error', message: body?.message ?? body?.stderr ?? 'repair rerun estimate failed' };
					return;
				}
				if (
					requestSerial !== estimateRequestSerial ||
					requestedSelectionKey !== selectionKey(selectedRepairIds) ||
					requestedLineageToken !== repairLineageToken
				) {
					return;
				}
				rerunState = {
					phase: 'estimated',
					architecture: selectedRerunArchitecture,
					corpus: body.corpus as RepairCorpus,
					estimates: ((body.summary?.estimates ?? []) as CostEstimate[]),
					scorer_version: requestedLineageToken,
					selection_key: requestedSelectionKey
				};
		} catch (e) {
			rerunState = { phase: 'error', message: String(e).slice(0, 240) };
			}
		}

		async function completeRecoveredRerun(intent: RecoverableRerunIntent) {
			const key = recoveryIntentKey(intent);
			if (writerLockBusy) {
				setRecoveredCompletionState(key, { phase: 'error', message: writerLockActionBlockText('finalizing repair markers') });
				return;
			}
			setRecoveredCompletionState(key, { phase: 'working' });
			try {
				const completeRes = await fetch('/api/repairs/rerun/complete', {
					method: 'POST',
					headers: { 'content-type': 'application/json' },
					body: JSON.stringify({
						parent_run_id: b.run_id,
						child_run_id: intent.child_run_id,
						architecture: intent.architecture,
						source_dump_id: intent.source_dump_id,
						correction_ids: intent.correction_ids
					})
				});
				const completeBody = await completeRes.json();
				if (completeRes.ok) {
					await returnToRecoveryQueueFront();
					return;
				}
				setRecoveredCompletionState(key, {
					phase: 'error',
					message: completeBody?.message ?? 'completion marker failed'
				});
			} catch (e) {
				setRecoveredCompletionState(key, {
					phase: 'error',
					message: String(e).slice(0, 160)
				});
			}
		}

		async function releaseRecoveredUncovered(intent: RecoverableRerunIntent) {
			const key = recoveryIntentKey(intent);
			if (writerLockBusy) {
				setRecoveredCompletionState(key, { phase: 'error', message: writerLockActionBlockText('releasing uncovered repair candidates') });
				return;
			}
			setRecoveredCompletionState(key, { phase: 'working' });
			try {
				const releaseRes = await fetch('/api/repairs/rerun/release', {
					method: 'POST',
					headers: { 'content-type': 'application/json' },
					body: JSON.stringify({
						parent_run_id: b.run_id,
						child_run_id: intent.child_run_id,
						architecture: intent.architecture,
						source_dump_id: intent.source_dump_id,
						correction_ids: intent.uncovered_correction_ids
					})
				});
				const releaseBody = await releaseRes.json();
				if (releaseRes.ok) {
					await returnToRecoveryQueueFront();
					return;
				}
				setRecoveredCompletionState(key, {
					phase: 'error',
					message: releaseBody?.message ?? 'uncovered release failed'
				});
			} catch (e) {
				setRecoveredCompletionState(key, {
					phase: 'error',
					message: String(e).slice(0, 160)
				});
			}
		}

		async function retryRepairCompletion(recovery: RepairCompletionRecovery) {
			if (rerunState.phase !== 'done') return;
			if (writerLockBusy) {
				rerunState = {
					...rerunState,
					completion_error: writerLockActionBlockText('retrying repair marker recovery'),
					completion_retrying: false
				};
				return;
			}
			rerunState = { ...rerunState, completion_retrying: true };
			try {
				const completeRes = await fetch('/api/repairs/rerun/complete', {
					method: 'POST',
					headers: { 'content-type': 'application/json' },
					body: JSON.stringify(recovery)
				});
				const completeBody = await completeRes.json();
				if (completeRes.ok) {
					rerunState = {
						...rerunState,
						completion_recorded: Number(completeBody.recorded ?? 0),
						collateral_recorded: Number(completeBody.collateral_recorded ?? 0),
						completion_error: null,
						recovery: null,
						completion_retrying: false
					};
					await invalidateAll();
				} else {
					rerunState = {
						...rerunState,
						completion_error: completeBody?.message ?? 'completion marker failed',
						completion_retrying: false
					};
				}
			} catch (e) {
				rerunState = {
					...rerunState,
					completion_error: String(e).slice(0, 160),
					completion_retrying: false
				};
			}
		}

		async function scoreRepairRerun(estimate: CostEstimate) {
		if (rerunState.phase !== 'estimated') return;
		if (writerLockBusy) {
			rerunState = { phase: 'error', message: writerLockActionBlockText('starting a repair child run') };
			return;
		}
			const corpus = rerunState.corpus;
			const architecture = rerunState.architecture;
				const cost_cap_usd = capForEstimate(estimate.cost_usd);
			const ctrl = new AbortController();
			rerunAbortController = ctrl;
			rerunState = {
				phase: 'scoring',
				architecture,
					corpus,
					model: estimate.model_id,
					run_id: null,
					scorer_version: rerunState.scorer_version,
					cost_cap_usd,
					cost_so_far_usd: null,
				n_evidences_done: 0,
			n_evidences_total: corpus.n_evidences,
			latest_stmt: null
		};
		try {
			const res = await fetch('/api/runs/score', {
				method: 'POST',
				headers: { 'content-type': 'application/json' },
				signal: ctrl.signal,
				body: JSON.stringify({
						path: corpus.path,
						source_dump_id: corpus.source_dump_id,
						model: estimate.model_id,
						scorer_version: rerunState.scorer_version,
							arch: architecture,
							parent_run_id: b.run_id,
							skip_ingest: true,
							cost_threshold_usd: cost_cap_usd,
							repair_rerun: {
								correction_ids: corpus.correction_ids,
								probe_step_filter: probeStepFilterForCorpus(corpus),
								probe_only: probeOnlyForCorpus(corpus)
							}
						})
					});
			if (!res.ok || !res.body) {
				rerunAbortController = null;
				const text = await res.text();
				rerunState = { phase: 'error', message: text.slice(0, 240) || 'repair rerun failed to start' };
				return;
			}
			const reader = res.body.getReader();
			const decoder = new TextDecoder();
			let buffer = '';
				let completedRunId: string | null = null;
				let completedEvidences = 0;
				let completedDuration: number | null = null;
				let completionRecorded: number | null = null;
				let completionError: string | null = null;
				let collateralRecorded: number | null = null;
				while (true) {
				const { value, done } = await reader.read();
				if (done) break;
				buffer += decoder.decode(value, { stream: true });
				let sep: number;
				while ((sep = buffer.indexOf('\n\n')) >= 0) {
					const frame = buffer.slice(0, sep);
					buffer = buffer.slice(sep + 2);
						for (const line of frame.split('\n')) {
							if (!line.startsWith('data:')) continue;
							const ev = JSON.parse(line.slice(5).trim());
							if (ev.event === 'started' && rerunState.phase === 'scoring') {
								rerunState = {
									...rerunState,
									run_id: typeof ev.run_id === 'string' ? ev.run_id : rerunState.run_id
								};
							} else if (ev.event === 'loaded' && rerunState.phase === 'scoring') {
								rerunState = {
									...rerunState,
									n_evidences_total: typeof ev.n_evidences === 'number' ? ev.n_evidences : rerunState.n_evidences_total
							};
						} else if (ev.event === 'progress' && rerunState.phase === 'scoring') {
							rerunState = {
								...rerunState,
								n_evidences_done: typeof ev.n_evidences_done === 'number' ? ev.n_evidences_done : rerunState.n_evidences_done,
								latest_stmt: typeof ev.latest_stmt_hash === 'string' ? ev.latest_stmt_hash : rerunState.latest_stmt,
								cost_so_far_usd: typeof ev.cost_so_far_usd === 'number' ? ev.cost_so_far_usd : rerunState.cost_so_far_usd
							};
							} else if (ev.event === 'done') {
								completedRunId = String(ev.run_id);
								completedEvidences = Number(ev.n_evidences_done ?? 0);
								completedDuration = typeof ev.duration_s === 'number' ? ev.duration_s : null;
								} else if (ev.event === 'repair_rerun_completed') {
									completionRecorded = Number(ev.recorded ?? 0);
									collateralRecorded = Number(ev.collateral_recorded ?? 0);
								} else if (ev.event === 'repair_rerun_completion_failed') {
									completionError = String(ev.error ?? 'completion marker failed').slice(0, 240);
								} else if (ev.event === 'error' || ev.event === 'spawn_error') {
									rerunState = { phase: 'error', message: ev.stderr ?? ev.error ?? 'repair rerun failed' };
							} else if (ev.event === 'cancel_tombstone_failed') {
								rerunState = {
									phase: 'error',
									message: `repair rerun canceled, but child tombstone write failed: ${String(ev.error ?? 'unknown error').slice(0, 180)}`,
									run_id: typeof ev.run_id === 'string' ? ev.run_id : null
								};
							}
					}
				}
				}
				if (completedRunId) {
						if (completionRecorded == null && completionError == null) {
							completionError = 'scoring finished but the server did not report repair completion markers';
						}
					rerunState = {
						phase: 'done',
					architecture,
					corpus,
					run_id: completedRunId,
						model: estimate.model_id,
						n_evidences: completedEvidences,
							duration_s: completedDuration,
							completion_recorded: completionRecorded,
							completion_error: completionError,
							collateral_recorded: collateralRecorded,
							recovery: completionError
								? {
									parent_run_id: b.run_id,
									child_run_id: completedRunId,
									architecture,
									source_dump_id: corpus.source_dump_id,
									correction_ids: corpus.correction_ids
								}
								: null
						};
						if (completionError) selectedRepairOverride = [];
					rerunAbortController = null;
					await invalidateAll();
				}
			} catch (e) {
				rerunAbortController = null;
				if (ctrl.signal.aborted) {
					if (rerunState.phase !== 'error') {
						rerunState = { phase: 'error', message: 'repair rerun cancel requested; child run will be marked canceled when the worker exits cleanly, and no completion markers were written' };
					}
				} else {
					rerunState = { phase: 'error', message: String(e).slice(0, 240) };
				}
		}
	}
</script>

<svelte:head><title>{b.run_id.slice(0, 8)} repairs · INDRA Belief</title></svelte:head>

<header>
	<div class="crumb">
		<a href="/">corpus</a><span class="sep"> / </span><a href={`/runs/${b.run_id}`}>run {b.run_id.slice(0, 8)}</a><span class="sep"> / </span><strong>repairs</strong>
	</div>
	<div class="meta">
		<span class={`run-arch run-arch-${b.architecture}`}>{archMark(b.architecture)}</span>
		<span>{b.architecture}</span>
	</div>
</header>

<main id="main">
	<section class="repair-head">
		<h1>repair backlog</h1>
		<p>
			{b.openCount.toLocaleString()} open repair candidate{b.openCount === 1 ? '' : 's'} for run <code>{b.run_id.slice(0, 8)}</code>.
			{#if b.rows.length >= b.limit}<span class="muted"> Showing first {b.limit.toLocaleString()} rows.</span>{/if}
		</p>
		<p class="repair-note">
			These rows preserve review intent append-only. They do not rewrite prior scorer output; rerun-from-cohort and before/after comparison are separate workflow states.
		</p>
	</section>

	<section class="rerun-panel" aria-label="rerun repair cohort">
			<h2>rerun open candidates</h2>
			<div class="rerun-controls">
				<div class="locked-field">
					<span>architecture</span>
					<strong>{archMark(selectedRerunArchitecture)} {selectedRerunArchitecture}</strong>
					<small>locked to parent run</small>
				</div>
				<div class="generated-field">
					<span>child lineage</span>
					<code>{selectedRepairCount > 0 ? repairLineageToken : 'no selected candidates'}</code>
					<small>{selectedRepairCount > 0 ? 'generated from parent run and selected candidates' : 'select candidates to generate child lineage'}</small>
				</div>
				<label>
					optional note
					<input bind:value={repairNote} disabled={preflightScopeLocked} placeholder="brief tag" maxlength="32" />
				</label>
				<label class="cap-control">
					cap multiplier
					<input type="number" value={normalizedCapMultiplier} min="1" max="3" step="0.05" disabled={rerunState.phase === 'scoring'} oninput={updateCapMultiplier} />
				</label>
					<button type="button" aria-label="preview repair rerun cost" disabled={writerLockBusy || selectedRepairCount === 0 || rerunState.phase === 'estimating' || rerunState.phase === 'scoring'} onclick={estimateRepairRerun}>
						preview rerun cost
					</button>
				</div>
				{#if writerLockBusy}
					<p class="repair-warning">{writerLockActionBlockText('starting repair writes')}</p>
				{/if}
			<div class="selection-bar" aria-label="repair rerun selection">
			<span>{selectedRepairCount.toLocaleString()} of {visibleRepairCount.toLocaleString()} visible candidates selected</span>
					<button type="button" disabled={visibleRepairCount === 0 || allVisibleSelected || preflightScopeLocked} onclick={selectAllVisible}>select visible</button>
					<button type="button" disabled={selectedRepairCount === 0 || preflightScopeLocked} onclick={clearSelection}>clear</button>
				{#if b.openCount > b.rows.length}
					<span class="muted">Only visible rows can be selected here; open backlog {b.openCount.toLocaleString()}, visible {b.rows.length.toLocaleString()}, rerun cap {b.limit.toLocaleString()}.</span>
				{/if}
				{#if b.activeRerunIntentCount > 0}
					<span class="muted">{b.activeRerunIntentCount.toLocaleString()} active child rerun group{b.activeRerunIntentCount === 1 ? '' : 's'} temporarily lock candidates below.</span>
				{/if}
			</div>
		{#if rerunState.phase === 'idle'}
			<p class="repair-note">The preview exports the current open repair candidate statements to a local subset file and estimates spend before any scorer API call.</p>
			{#if b.architecture === 'decomposed' || b.architecture === 'monolithic'}
				<p class="repair-note">Cross-architecture reruns are intentionally gated here; use the paired workbench when the question is [M] vs [D].</p>
			{/if}
		{:else if rerunState.phase === 'estimating'}
			<p class="repair-note">exporting repair subset and estimating spend…</p>
		{:else if rerunState.phase === 'estimated'}
				<div class="cost-panel">
					<p>
						{rerunState.corpus.n_candidates} candidates · {rerunState.corpus.n_statements} candidate statements · {rerunState.corpus.n_evidences} evidences.
						Scoring will write a child run with <code>parent_run_id={b.run_id.slice(0, 8)}…</code>, scorer version <code>{rerunState.scorer_version}</code>, and will skip re-ingest.
						<button type="button" class="inline-reset" onclick={reviseRepairPreflight}>revise selection or note</button>
					</p>
					{#if rerunState.corpus.evidence_count_validated}
						<p class="repair-note">
							Evidence denominator validated: raw statement JSON {rerunState.corpus.n_raw_json_evidences.toLocaleString()} = normalized evidence rows {rerunState.corpus.n_table_evidences.toLocaleString()}.
						</p>
					{/if}
					{#if rerunState.corpus.dropped_correction_ids.length > 0}
						<p class="repair-warning">
							{rerunState.corpus.dropped_correction_ids.length} requested candidate{rerunState.corpus.dropped_correction_ids.length === 1 ? '' : 's'} excluded:
							{#each rerunState.corpus.dropped_correction_ids as d, i}
								{#if i > 0}, {/if}<code>#{d.correction_id}</code> {d.reason.replace(/_/g, ' ')}
							{/each}
						</p>
					{/if}
							{#if rerunState.corpus.n_statement_scope_candidates > 0}
								<p class="repair-warning">
									{rerunState.corpus.n_statement_scope_candidates} statement-scope candidate{rerunState.corpus.n_statement_scope_candidates === 1 ? '' : 's'} will intentionally score whole selected statements; this preflight covers {rerunState.corpus.n_evidences} evidence row{rerunState.corpus.n_evidences === 1 ? '' : 's'}.
								</p>
							{/if}
								{#if rerunState.corpus.n_scope_expansion_evidences > 0}
									<p class="repair-warning">
										{rerunState.corpus.n_scope_expansion_evidences} evidence-grain sibling row{rerunState.corpus.n_scope_expansion_evidences === 1 ? '' : 's'} will also be scored beyond the {rerunState.corpus.n_selected_evidence_candidates} individually selected evidence candidate{rerunState.corpus.n_selected_evidence_candidates === 1 ? '' : 's'}; append-only collateral markers are written by the scoring stream when the child run succeeds.
									</p>
								{/if}
							{#if rerunState.corpus.n_probe_slot_reviewed_candidates > 0}
								<p class="repair-warning">
									{rerunState.corpus.n_probe_slot_reviewed_candidates} probe-slot reviewed candidate{rerunState.corpus.n_probe_slot_reviewed_candidates === 1 ? '' : 's'} {rerunState.corpus.n_probe_slot_reviewed_candidates === 1 ? 'carries' : 'carry'} rerun metadata
									{#if probeSlotCountPairs(rerunState.corpus).length > 0}
										:
										{#each probeSlotCountPairs(rerunState.corpus) as [slot, n], i}
											{#if i > 0} · {/if}<code>{slot}</code> {n}
										{/each}
									{/if}
									{#if probeOnlyForCorpus(rerunState.corpus)}
										. This launch will run only the reviewed probes, then merge them with the parent probe trace for deterministic aggregate re-adjudication.
									{:else}
										. The worker will materialize native probe rows for these reviewed slots; aggregate scoring still reruns the selected statement evidence.
									{/if}
								</p>
							{/if}
							{#if probeOnlyForCorpus(rerunState.corpus)}
								<p class="repair-note">Probe-only repair runs lower the spend boundary to selected slots; aggregate rows are deterministic merge outputs, not fresh aggregate LLM calls.</p>
							{/if}
						<p class="repair-note">
							Spend cap uses the selected {normalizedCapMultiplier.toFixed(2)}x multiplier before launch; observed spend is checked after each evidence and the worker stops before the next evidence once the cap is crossed.
						</p>
						<div class="cost-table-wrap">
							<table>
								<thead><tr><th>model</th><th class="num">est cost</th><th class="num">cap</th><th class="num">calls</th><th></th></tr></thead>
								<tbody>
									{#each rerunState.estimates as e}
										{@const cap = capForEstimate(e.cost_usd)}
										<tr>
											<td><code>{e.model_id}</code></td>
											<td class="num">{fmtCost(e.cost_usd)}</td>
											<td class="num">{fmtCost(cap)}</td>
											<td class="num">{e.n_llm_calls_est.toLocaleString()}</td>
											<td>
													<button type="button" class="spend" disabled={writerLockBusy} onclick={() => scoreRepairRerun(e)}>
														run {e.model_id} · cap {fmtCost(cap)}
													</button>
											</td>
										</tr>
									{/each}
								</tbody>
							</table>
						</div>
			</div>
		{:else if rerunState.phase === 'scoring'}
				<p class="repair-note">
					{archMark(rerunState.architecture)} {probeOnlyForCorpus(rerunState.corpus) ? 'probe-only rescoring' : 'scoring'} {rerunState.n_evidences_done}/{rerunState.n_evidences_total ?? '—'} evidences on {rerunState.model}
					· spent {fmtCost(rerunState.cost_so_far_usd ?? 0)} / {fmtCost(rerunState.cost_cap_usd)}
					{#if rerunState.run_id}<span> · child <a href={`/runs/${rerunState.run_id}`}><code>{shortHash(rerunState.run_id)}</code></a></span>{/if}
					{#if rerunState.latest_stmt}<span> · latest <code>{shortHash(rerunState.latest_stmt)}</code></span>{/if}
					<button type="button" class="inline-cancel" onclick={cancelRepairRerun}>cancel</button>
				</p>
		{:else if rerunState.phase === 'done'}
			<p class="repair-note">
						child run <a href={`/runs/${rerunState.run_id}`}><code>{shortHash(rerunState.run_id)}</code></a> {probeOnlyForCorpus(rerunState.corpus) ? 'rescored selected probes for' : 'scored'} {rerunState.n_evidences} evidences on {rerunState.model}{#if rerunState.duration_s != null} in {rerunState.duration_s.toFixed(1)}s{/if}.
						{#if rerunState.completion_recorded != null}<span> {rerunState.completion_recorded} repair candidates now have append-only rerun markers.</span>{/if}
						{#if rerunState.collateral_recorded}<span> {rerunState.collateral_recorded} collateral evidence markers recorded.</span>{/if}
						{#if rerunState.completion_error}<span class="repair-error"> completion marker failed: {rerunState.completion_error}. Selection was cleared to avoid accidental re-spend; inspect the child run before retrying.</span>{/if}
						{#if rerunState.recovery}
							{@const recovery = rerunState.recovery}
								<button type="button" class="inline-reset" disabled={writerLockBusy || rerunState.completion_retrying} onclick={() => retryRepairCompletion(recovery)}>
									{rerunState.completion_retrying ? 'retrying marker insert…' : 'retry marker insert'}
								</button>
						{/if}
					</p>
			{:else if rerunState.phase === 'error'}
				<p class="repair-error">
					{rerunState.message}
					{#if rerunState.run_id}<span> child <a href={`/runs/${rerunState.run_id}`}><code>{shortHash(rerunState.run_id)}</code></a></span>{/if}
				</p>
			{/if}
		</section>

	{#if b.activeRerunIntents.length > 0}
		<section class="recovery-panel" aria-label="active repair reruns locking candidates">
			<h2>active reruns locking candidates</h2>
			<p class="repair-note">
				These child runs are not terminal yet, so their repair candidates are hidden from the spendable queue. They become recoverable after success, or selectable again after failure/cancel.
				{#if b.activeRerunIntentCount > b.activeRerunIntents.length}
					<span> Showing first {b.activeRerunIntents.length.toLocaleString()} of {b.activeRerunIntentCount.toLocaleString()} active child run groups.</span>
				{/if}
			</p>
			<div class="recovery-list">
				{#each b.activeRerunIntents as intent}
					{@const key = activeIntentKey(intent)}
					{@const tombstone = activeTombstoneState[key]}
					<article class="recovery-row" aria-labelledby={`active-${activeIntentKey(intent)}`}>
						<div>
							<div class="rerun-card-head" id={`active-${activeIntentKey(intent)}`}>
								{#if intent.has_child_run}
									<a href={`/runs/${intent.child_run_id}`}><code>{shortHash(intent.child_run_id)}</code></a>
								{:else}
									<code>{shortHash(intent.child_run_id)}</code>
								{/if}
								<span class={`run-arch run-arch-${intent.architecture}`}>{archMark(intent.architecture)}</span>
								<span>{intent.status}</span>
							</div>
							<p class="muted">
								{intent.n_candidates.toLocaleString()} locked candidate{intent.n_candidates === 1 ? '' : 's'} · source <code>{intent.source_dump_id}</code> · intent {intent.last_intent_at.replace(/\.\d+$/, '')}
							</p>
							{#if !intent.has_child_run}
								<p class="repair-warning">
									waiting for child run row; this queued lock expires automatically if the worker never starts.
								</p>
							{:else if writerLockBusy}
								<p class="repair-warning">
									{#if effectiveWriterLock}
										writer is active: {writerLockText(effectiveWriterLock)}; use the live run controls or wait before marking this child canceled.
									{:else}
										writer lock status is unavailable; retrying before this child can be marked canceled.
									{/if}
								</p>
							{/if}
							{#if tombstone?.phase === 'error'}
								<p class="repair-error">{tombstone.message}</p>
							{/if}
						</div>
						{#if intent.has_child_run}
							<div class="recovery-actions">
									<button
										type="button"
										class="inline-reset"
										aria-label={`mark repair child ${shortHash(intent.child_run_id)} canceled`}
										disabled={writerLockBusy || !isRepairArchitecture(intent.architecture) || intent.correction_ids.length === 0 || tombstone?.phase === 'working' || rerunState.phase === 'scoring'}
										onclick={() => tombstoneActiveRerun(intent)}
									>
									{tombstone?.phase === 'working' ? 'marking canceled…' : 'mark child canceled'}
								</button>
							</div>
						{/if}
					</article>
				{/each}
			</div>
		</section>
	{/if}

	{#if b.recoverableRerunIntents.length > 0}
		<section class="recovery-panel" aria-label="completed repair reruns awaiting markers">
			<h2>completed reruns awaiting markers</h2>
			<p class="repair-note">
				These child runs already succeeded and are blocking re-spend through persisted intent rows. Finalizing writes only append-only completion markers; it does not call the scorer.
				{#if b.recoverableRerunIntentCount > b.recoverableRerunIntents.length}
					<span> Showing {recoveryPageStart.toLocaleString()}-{recoveryPageEnd.toLocaleString()} of {b.recoverableRerunIntentCount.toLocaleString()} recoverable child run groups.</span>
				{/if}
			</p>
			{#if b.recoverableRerunIntentCount > b.recoverableRerunIntentLimit}
				<nav class="recovery-pager" aria-label="completed rerun marker recovery pages">
					{#if b.recoverableRerunIntentOffset > 0}
						<a href={recoveryPageHref(previousRecoveryOffset)}>previous</a>
					{:else}
						<span class="muted">previous</span>
					{/if}
					<span>{recoveryPageStart.toLocaleString()}-{recoveryPageEnd.toLocaleString()}</span>
					{#if nextRecoveryOffset < b.recoverableRerunIntentCount}
						<a href={recoveryPageHref(nextRecoveryOffset)}>next</a>
					{:else}
						<span class="muted">next</span>
					{/if}
				</nav>
			{/if}
			<div class="recovery-list">
				{#each b.recoverableRerunIntents as intent}
					{@const key = recoveryIntentKey(intent)}
					{@const completion = recoveredCompletionState[key]}
					<article class="recovery-row">
						<div>
							<div class="rerun-card-head">
								<a href={`/runs/${intent.child_run_id}`}><code>{shortHash(intent.child_run_id)}</code></a>
								<span class={`run-arch run-arch-${intent.architecture}`}>{archMark(intent.architecture)}</span>
								<span>{intent.status}</span>
							</div>
							<p class="muted">
								{intent.n_candidates.toLocaleString()} finalizable marker{intent.n_candidates === 1 ? '' : 's'} · {intent.n_missing_markers.toLocaleString()} missing total · source <code>{intent.source_dump_id}</code> · intent {intent.last_intent_at.replace(/\.\d+$/, '')}
							</p>
							{#if intent.n_uncovered_candidates > 0}
								<p class="repair-warning">
									{intent.n_uncovered_candidates.toLocaleString()} intended candidate{intent.n_uncovered_candidates === 1 ? '' : 's'} lack child aggregate rows; release them to make them selectable for a new repair run.
								</p>
							{/if}
							{#if !isRepairArchitecture(intent.architecture)}
								<p class="repair-warning">
									child architecture is unknown; inspect the child run before marker recovery.
								</p>
							{/if}
							{#if completion?.phase === 'error'}
								<p class="repair-error">{completion.message}</p>
							{/if}
						</div>
						<div class="recovery-actions">
								<button type="button" class="inline-reset" disabled={writerLockBusy || !isRepairArchitecture(intent.architecture) || intent.correction_ids.length === 0 || completion?.phase === 'working' || rerunState.phase === 'scoring'} onclick={() => completeRecoveredRerun(intent)}>
									{completion?.phase === 'working' ? 'working…' : 'finalize markers'}
								</button>
								<button type="button" class="inline-reset" disabled={writerLockBusy || !isRepairArchitecture(intent.architecture) || intent.uncovered_correction_ids.length === 0 || completion?.phase === 'working' || rerunState.phase === 'scoring'} onclick={() => releaseRecoveredUncovered(intent)}>
									{completion?.phase === 'working' ? 'working…' : 'release uncovered'}
								</button>
						</div>
					</article>
				{/each}
			</div>
		</section>
	{/if}

	<section class="rerun-history" aria-label="repair rerun comparisons">
		<h2>repair reruns</h2>
		{#if b.reruns.length === 0}
			<p class="repair-note">no child repair runs yet.</p>
		{:else}
			<div class="rerun-cards">
				{#each b.reruns as r}
					<article class="rerun-card">
						<div class="rerun-card-head">
							<a href={`/runs/${r.run_id}`}><code>{shortHash(r.run_id)}</code></a>
							<span class={`run-arch run-arch-${r.architecture}`}>{archMark(r.architecture)}</span>
							<span>{r.status}</span>
						</div>
							<p class="muted">{fmtTimestamp(r.started_at)} · {r.model_id_default ?? 'unknown model'} · actual {fmtCost(r.cost_actual_usd)}</p>
							<p class="rerun-links"><a href={`/runs/${b.run_id}/repairs/${r.run_id}`}>full repair comparison</a></p>
						{#if r.not_defined_reason}
							<p class="repair-note">not defined: {r.not_defined_reason}</p>
						{:else}
							<dl>
								<div><dt>overlap</dt><dd>{r.n_overlap_evidences.toLocaleString()} ev</dd></div>
								<div><dt>score rows</dt><dd>{r.n_score_evidences.toLocaleString()} ev</dd></div>
								<div><dt>MAE</dt><dd>{fmtMetric(r.parent_mae)} → {fmtMetric(r.child_mae)}</dd></div>
								<div><dt>bias</dt><dd>{fmtSigned(r.parent_bias)} → {fmtSigned(r.child_bias)}</dd></div>
									<div><dt>verdict rows</dt><dd>{r.n_verdict_evidences.toLocaleString()} ev</dd></div>
									<div><dt>verdicts</dt><dd>{r.verdicts_moved_total} moved · {r.verdicts_moved_to_correct} to correct · {r.verdicts_moved_to_incorrect} to incorrect</dd></div>
									<div><dt>repair candidates</dt><dd>{r.n_child_covered_candidates.toLocaleString()} / {r.n_candidate_evidences.toLocaleString()} covered{#if r.n_new_child_aggregate_candidates > 0} · {r.n_new_child_aggregate_candidates.toLocaleString()} new aggregate{/if}</dd></div>
								</dl>
								{#if r.n_score_evidences !== r.n_overlap_evidences}
									<p class="repair-note">MAE/bias use the {r.n_overlap_evidences.toLocaleString()}-evidence overlap; {Math.abs(r.n_score_evidences - r.n_overlap_evidences).toLocaleString()} score row{Math.abs(r.n_score_evidences - r.n_overlap_evidences) === 1 ? '' : 's'} exist outside that overlap.</p>
								{/if}
								{#if r.candidate_lanes.length > 0}
									<div class="candidate-lanes" aria-label={`candidate before after repair lanes for ${shortHash(r.run_id)}`}>
										{#each r.candidate_lanes as lane}
											<div class="candidate-lane">
												<div>
													<span class={`movement movement-${lane.movement}`}>{movementLabel(lane.movement)}</span>
													<code>#{lane.correction_id}</code>
													{#if lane.suspected_step_kind}<span class="muted"> · {lane.suspected_step_kind}</span>{/if}
												</div>
												<div>
													<a href={`/statements/${lane.stmt_hash}?run_id=${b.run_id}`}><code>{shortHash(lane.stmt_hash)}</code></a>
													{#if lane.evidence_hash}<span class="muted"> / <code>{shortHash(lane.evidence_hash)}</code></span>{/if}
													{#if lane.source_api}<span class="muted"> · {lane.source_api}</span>{/if}
												</div>
												<div class="before-after">
													<span>parent {lane.parent_verdict ?? 'no aggregate'} · {fmtBelief(lane.parent_score)}</span>
													<span>child {lane.child_verdict ?? 'no aggregate'} · {fmtBelief(lane.child_score)}</span>
													<span>Δ error {fmtSigned(lane.abs_error_delta)}</span>
												</div>
											</div>
										{/each}
									</div>
								{/if}
							{/if}
						</article>
				{/each}
			</div>
		{/if}
	</section>

	{#if b.rows.length === 0}
		<p class="hint">
			{#if b.activeRerunIntentCount > 0}
				no open repair candidates are available; active child repair runs are still locking candidates above.
			{:else if b.recoverableRerunIntentCount > 0}
				no open repair candidates are available; completed repair reruns still need marker finalization above.
			{:else}
				no repair candidates exist for this run. Create them from a cohort route.
			{/if}
		</p>
	{:else}
		<div class="table-wrap">
			<table>
				<thead>
					<tr>
						<th class="select-col">rerun</th>
						<th>candidate</th>
						<th>claim</th>
						<th>observed</th>
						<th>statement</th>
						<th>source cohort</th>
						<th>review</th>
					</tr>
				</thead>
				<tbody>
					{#each b.rows as r}
						<tr>
							<td class="select-col">
								<input
										type="checkbox"
										checked={selectedRepairIds.includes(r.correction_id)}
										disabled={preflightScopeLocked}
									aria-label={`select repair candidate ${r.correction_id}`}
									onchange={() => toggleRepairId(r.correction_id)}
								/>
							</td>
							<td>
								<div class="candidate-id">
									<span class={`status status-${r.status}`}>{r.status}</span>
									<code>#{r.correction_id}</code>
								</div>
								<div class={`arch run-arch-${r.architecture}`}>{archMark(r.architecture)} {r.architecture}</div>
								<div class="muted">{r.correction_kind}</div>
							</td>
							<td>
								<div><strong>{r.suspected_step_kind ?? r.step_kind ?? 'aggregate'}</strong></div>
								<div class="muted">{r.severity ?? 'untriaged'}</div>
								{#if r.probe_coverage === 'present'}
									<div class="probe-repair" aria-label={`probe repair facts for correction ${r.correction_id}`}>
										<span>probe coverage</span>
										<strong>persisted</strong>
										<div class="probe-slots">
											{#each missingProbeSlots(r.missing_probe_slots) as slot}
												<code>{slot}</code>
											{/each}
											{#if missingProbeSlots(r.missing_probe_slots).length === 0}
												<span class="muted">no missing probe slots recorded</span>
											{/if}
										</div>
									</div>
									<form class="probe-slot-editor" aria-label={`probe slot review for correction ${r.correction_id}`} onsubmit={(ev) => saveProbeSlotReview(ev, r)}>
										<fieldset disabled={!clientReady || writerLockBusy || probeSlotReviewState[r.correction_id]?.phase === 'working'}>
											<legend>probe slot review</legend>
											<div class="probe-slot-options">
												{#each missingProbeSlots(r.missing_probe_slots) as slot}
													<label>
														<input type="checkbox" name="probe_slot" value={slot} checked={defaultProbeSlotChecked(r, slot)} />
														<code>{slot}</code>
													</label>
												{/each}
											</div>
											{#if missingProbeSlots(r.missing_probe_slots).length === 0}
												<p class="muted">no missing probe slots to target.</p>
											{/if}
											<textarea name="probe_slot_note" rows="2">{r.probe_slot_review_note ?? ''}</textarea>
											<div class="probe-slot-editor-actions">
												<button type="submit" class="inline-reset" disabled={!clientReady || missingProbeSlots(r.missing_probe_slots).length === 0}>
													{probeSlotReviewState[r.correction_id]?.phase === 'working' ? 'recording…' : 'record slot review'}
												</button>
												{#if r.probe_slot_review_count > 0}
													<span class="muted">
														latest {r.probe_slot_reviewed_at?.replace(/\.\d+$/, '') ?? 'recorded'}
														{#if reviewedProbeSlots(r).length > 0} · {reviewedProbeSlots(r).join(', ')}{/if}
													</span>
												{/if}
											</div>
											{#if probeSlotReviewError(r.correction_id)}
												<p class="repair-error">{probeSlotReviewError(r.correction_id)}</p>
											{/if}
										</fieldset>
									</form>
								{/if}
								{#if r.reviewer_hypothesis}
									<p>{r.reviewer_hypothesis}</p>
								{:else if r.note}
									<p>{r.note}</p>
								{:else}
									<p class="muted">no reviewer hypothesis captured</p>
								{/if}
							</td>
							<td>
								<div>{r.verdict ?? 'unknown'} <span class="muted">· {r.confidence ?? '—'}</span></div>
								<div class="num">score {fmtBelief(r.score)} · Δ {fmtDelta(r.residual)}</div>
								{#if r.probe_coverage === 'present'}
									<div class="probe-counts" aria-label={`probe counts for correction ${r.correction_id}`}>
										{#each probeCountPairs(r) as [slot, n]}
											<span><code>{slot}</code> {n ?? 0}</span>
										{/each}
									</div>
								{/if}
								{#if r.source_api}<div><code>{r.source_api}</code>{#if r.pmid}<span class="muted"> · {r.pmid}</span>{/if}</div>{/if}
							</td>
							<td>
								<a href={`/statements/${r.stmt_hash}?run_id=${b.run_id}`}><code>{shortHash(r.stmt_hash)}</code></a>
								{#if r.evidence_hash}<span class="muted"> / <code>{shortHash(r.evidence_hash)}</code></span>{/if}
								<div class="type">{r.indra_type ?? 'unknown type'}</div>
								<div class="agents" title={r.agent_names}>{r.agent_names}</div>
								<p>{r.text ?? 'no evidence text'}</p>
							</td>
							<td>
								{#if safeRoute(r.source_route)}
									<a href={safeRoute(r.source_route)}>{r.source_route}</a>
								{:else}
									<span class="muted">{r.source_route ?? 'unknown source'}</span>
								{/if}
								<div class="filters">
									{#each filterPairs(r.source_filters_json) as [k, v]}
										<span class="filter-chip"><span>{k}</span><strong>{v}</strong></span>
									{/each}
								</div>
							</td>
							<td>
								<div>{r.reviewer ?? 'anonymous'}</div>
								<div class="muted">{r.created_at.replace(/\.\d+$/, '')}</div>
							</td>
						</tr>
					{/each}
				</tbody>
			</table>
		</div>
	{/if}
</main>

<style>
	:global(:root) {
		--ink: #1a1a1a;
		--ink-muted: #6a6a6a;
		--ink-faint: #727272;
		--paper: #fdfcf8;
		--rule: #e6e2d6;
		--accent: #7d2a1a;
		--ok-green: #2a6f2a;
		--mono: ui-monospace, 'SF Mono', 'JetBrains Mono', Menlo, monospace;
		--serif: 'Iowan Old Style', 'Source Serif Pro', Georgia, serif;
	}
	:global(html, body) {
		background: var(--paper);
		color: var(--ink);
		font-family: var(--serif);
		font-size: 16px;
		line-height: 1.45;
		margin: 0;
	}
	header {
		display: flex;
		justify-content: space-between;
		align-items: baseline;
		padding: 0.6rem 1.5rem;
		border-bottom: 1px solid var(--rule);
		font-family: var(--mono);
		font-size: 0.78rem;
		color: var(--ink-muted);
	}
	.crumb a,
	.rerun-card a,
	td a {
		color: var(--accent);
		text-decoration: none;
	}
	.crumb a:hover,
	.rerun-card a:hover,
	td a:hover {
		text-decoration: underline;
	}
	.crumb strong,
	.run-arch {
		color: var(--ink);
		font-weight: 500;
	}
	.sep,
	.muted {
		color: var(--ink-faint);
	}
	.meta {
		display: flex;
		gap: 0.5rem;
		align-items: baseline;
	}
	.run-arch-monolithic { color: var(--accent); }
	.run-arch-decomposed { color: var(--ok-green); }
	main {
		max-width: 1400px;
		margin: 0 auto;
		padding: 1.5rem 1.5rem 4rem;
	}
	h1 {
		font-family: var(--serif);
		font-size: 1.5rem;
		font-weight: 400;
		margin: 0 0 0.4rem;
	}
	.repair-head p {
		margin: 0 0 0.5rem;
	}
	.repair-note,
	.hint {
		color: var(--ink-muted);
		font-size: 0.9rem;
	}
	.rerun-panel {
		border-top: 1px solid var(--rule);
		border-bottom: 1px solid var(--rule);
		margin: 1.2rem 0;
		padding: 0.9rem 0;
	}
	.rerun-panel h2 {
		font-family: var(--serif);
		font-size: 1.05rem;
		font-weight: 400;
		margin: 0 0 0.55rem;
	}
	.rerun-controls {
		display: flex;
		flex-wrap: wrap;
		gap: 0.65rem;
		align-items: end;
		font-family: var(--mono);
		font-size: 0.76rem;
	}
	.rerun-controls label {
		display: flex;
		flex-direction: column;
		gap: 0.2rem;
		color: var(--ink-muted);
	}
		.locked-field {
			display: flex;
			flex-direction: column;
			gap: 0.12rem;
			color: var(--ink-muted);
			min-width: 9rem;
		}
		.generated-field {
			display: flex;
			flex-direction: column;
			gap: 0.12rem;
			color: var(--ink-muted);
			min-width: min(100%, 24rem);
		}
		.locked-field strong,
		.generated-field code {
			color: var(--ink);
			font-weight: 500;
		}
		.generated-field code {
			border: 1px solid var(--rule);
			background: rgba(255, 255, 255, 0.35);
			padding: 0.2rem 0.4rem;
			overflow-wrap: anywhere;
		}
		.locked-field small,
		.generated-field small {
			color: var(--ink-faint);
			font-size: 0.68rem;
		}
	.rerun-controls input {
		font-family: var(--mono);
		font-size: 0.78rem;
		border: 1px solid var(--rule);
		background: var(--paper);
		color: var(--ink);
		padding: 0.22rem 0.4rem;
	}
		.rerun-controls input {
			width: 11rem;
		}
		.rerun-controls .cap-control input {
			width: 5rem;
		}
	.rerun-controls button,
	.spend {
		font-family: var(--mono);
		font-size: 0.76rem;
		border: 1px solid var(--accent);
		background: transparent;
		color: var(--accent);
		cursor: pointer;
		padding: 0.24rem 0.55rem;
	}
	.rerun-controls button:disabled {
		border-color: var(--rule);
		color: var(--ink-faint);
		cursor: default;
	}
	.selection-bar {
		display: flex;
		flex-wrap: wrap;
		gap: 0.5rem;
		align-items: baseline;
		margin-top: 0.55rem;
		font-family: var(--mono);
		font-size: 0.76rem;
		color: var(--ink-muted);
	}
		.selection-bar button,
		.inline-cancel,
		.inline-reset {
			font-family: var(--mono);
			font-size: 0.74rem;
			border: 1px solid var(--rule);
		background: transparent;
		color: var(--accent);
		cursor: pointer;
		padding: 0.12rem 0.45rem;
	}
	.selection-bar button:disabled {
		color: var(--ink-faint);
		cursor: default;
	}
		.inline-cancel {
			margin-left: 0.45rem;
			border-color: var(--accent);
		}
		.inline-reset {
			margin-left: 0.45rem;
		}
		.inline-reset:hover {
			border-color: var(--accent);
		}
		.inline-reset:disabled {
			border-color: var(--rule);
			color: var(--ink-faint);
			cursor: default;
		}
		.cost-panel {
			margin-top: 0.65rem;
		}
		.cost-table-wrap {
			overflow-x: auto;
		}
		.cost-panel table {
			min-width: 620px;
		}
	.cost-panel p {
		margin: 0 0 0.45rem;
		color: var(--ink-muted);
	}
	.repair-error {
		color: var(--accent);
		font-family: var(--mono);
		font-size: 0.82rem;
	}
	.repair-warning {
		border-left: 2px solid var(--accent);
		color: var(--ink);
		font-family: var(--mono);
		font-size: 0.8rem;
		margin: 0.45rem 0;
		padding-left: 0.55rem;
	}
	.probe-repair {
		border-left: 2px solid var(--ok-green);
		font-family: var(--mono);
		font-size: 0.74rem;
		margin: 0.35rem 0;
		padding-left: 0.5rem;
	}
	.probe-repair > span {
		color: var(--ink-muted);
		margin-right: 0.35rem;
	}
	.probe-slots,
	.probe-counts {
		display: flex;
		flex-wrap: wrap;
		gap: 0.25rem 0.4rem;
		margin-top: 0.25rem;
	}
	.probe-slots code,
	.probe-counts span {
		border: 1px solid var(--rule);
		padding: 0.08rem 0.25rem;
	}
	.probe-slot-editor {
		border-left: 2px solid var(--rule);
		font-family: var(--mono);
		font-size: 0.74rem;
		margin: 0.45rem 0;
		padding-left: 0.5rem;
	}
	.probe-slot-editor fieldset {
		border: 0;
		margin: 0;
		padding: 0;
	}
	.probe-slot-editor legend {
		color: var(--ink);
		font-weight: 500;
		margin-bottom: 0.25rem;
		padding: 0;
	}
	.probe-slot-options {
		display: flex;
		flex-wrap: wrap;
		gap: 0.25rem 0.45rem;
		margin-bottom: 0.35rem;
	}
	.probe-slot-options label {
		align-items: center;
		display: inline-flex;
		gap: 0.2rem;
	}
	.probe-slot-options input {
		margin: 0;
	}
	.probe-slot-editor textarea {
		background: #fffefb;
		border: 1px solid var(--rule);
		box-sizing: border-box;
		color: var(--ink);
		font-family: var(--mono);
		font-size: 0.74rem;
		max-width: 100%;
		min-height: 2.4rem;
		resize: vertical;
		width: 100%;
	}
	.probe-slot-editor-actions {
		align-items: baseline;
		display: flex;
		flex-wrap: wrap;
		gap: 0.35rem 0.5rem;
		margin-top: 0.3rem;
	}
	.probe-counts {
		font-family: var(--mono);
		font-size: 0.72rem;
		color: var(--ink-muted);
	}
	.rerun-history {
		margin: 1.2rem 0;
	}
	.recovery-panel {
		border-bottom: 1px solid var(--rule);
		margin: 1.2rem 0;
		padding-bottom: 1rem;
	}
	.recovery-panel h2,
	.rerun-history h2 {
		font-family: var(--serif);
		font-size: 1.05rem;
		font-weight: 400;
		margin: 0 0 0.55rem;
	}
	.recovery-list {
		display: grid;
		gap: 0.6rem;
		margin-top: 0.65rem;
	}
	.recovery-pager {
		display: flex;
		gap: 0.7rem;
		align-items: baseline;
		font-family: var(--mono);
		font-size: 0.76rem;
		margin: 0.55rem 0 0;
	}
	.recovery-pager a {
		color: var(--accent);
		text-decoration: none;
	}
	.recovery-pager a:hover {
		text-decoration: underline;
	}
	.recovery-row {
		border-left: 2px solid var(--accent);
		display: flex;
		flex-wrap: wrap;
		gap: 0.5rem 1rem;
		justify-content: space-between;
		padding-left: 0.75rem;
	}
	.recovery-row p {
		margin: 0.2rem 0 0;
	}
	.recovery-actions {
		display: flex;
		flex-wrap: wrap;
		gap: 0.35rem;
		align-items: start;
	}
	.recovery-actions .inline-reset {
		margin-left: 0;
	}
	.rerun-cards {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
		gap: 0.75rem;
	}
	.rerun-card {
		border-left: 2px solid var(--rule);
		padding-left: 0.75rem;
	}
	.rerun-card-head {
		display: flex;
		flex-wrap: wrap;
		gap: 0.45rem;
		align-items: baseline;
		font-family: var(--mono);
		font-size: 0.82rem;
	}
	.rerun-card p {
		margin: 0.25rem 0;
	}
	.rerun-links {
		font-family: var(--mono);
		font-size: 0.76rem;
	}
	.rerun-card dl {
		display: grid;
		grid-template-columns: repeat(2, minmax(0, 1fr));
		gap: 0.4rem 0.8rem;
		margin: 0.45rem 0 0;
		font-family: var(--mono);
		font-size: 0.76rem;
	}
	.candidate-lanes {
		display: grid;
		gap: 0.45rem;
		margin-top: 0.65rem;
	}
	.candidate-lane {
		border-left: 2px solid var(--rule);
		font-family: var(--mono);
		font-size: 0.74rem;
		padding-left: 0.5rem;
	}
	.candidate-lane a {
		color: var(--accent);
		text-decoration: none;
	}
	.candidate-lane a:hover {
		text-decoration: underline;
	}
	.movement {
		color: var(--ink);
		font-weight: 500;
	}
	.movement-new_child_aggregate,
	.movement-verdict_to_correct,
	.movement-score_improved {
		color: var(--ok-green);
	}
	.movement-verdict_to_incorrect,
	.movement-score_regressed {
		color: var(--accent);
	}
	.before-after {
		display: flex;
		flex-wrap: wrap;
		gap: 0.3rem 0.55rem;
		color: var(--ink-muted);
	}
	.rerun-card dt {
		color: var(--ink-muted);
		margin: 0;
	}
	.rerun-card dd {
		margin: 0.05rem 0 0;
	}
	.table-wrap {
		overflow-x: auto;
		border-top: 1px solid var(--ink);
		margin-top: 1.2rem;
	}
	table {
		width: 100%;
		border-collapse: collapse;
		font-family: var(--mono);
		font-size: 0.78rem;
		font-variant-numeric: tabular-nums;
	}
	th,
	td {
		text-align: left;
		vertical-align: top;
		padding: 0.55rem 0.7rem 0.55rem 0;
		border-bottom: 1px dotted var(--rule);
	}
	th {
		color: var(--ink-muted);
		font-weight: 500;
	}
	.select-col {
		width: 3.6rem;
	}
	.select-col input {
		inline-size: 1rem;
		block-size: 1rem;
		accent-color: var(--accent);
	}
	td p {
		font-family: var(--serif);
		font-size: 0.82rem;
		max-width: 34rem;
		margin: 0.25rem 0 0;
		color: var(--ink-muted);
	}
	code {
		font-family: var(--mono);
	}
	.candidate-id {
		display: flex;
		gap: 0.35rem;
		align-items: baseline;
	}
	.status {
		text-transform: lowercase;
	}
	.status-open {
		color: var(--accent);
	}
	.status-resolved,
	.status-superseded {
		color: var(--ok-green);
	}
	.arch,
	.type {
		margin-top: 0.2rem;
	}
	.agents {
		max-width: 20rem;
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
		color: var(--ink-muted);
	}
	.num {
		font-variant-numeric: tabular-nums;
	}
	.filters {
		display: flex;
		flex-wrap: wrap;
		gap: 0.35rem 0.5rem;
		margin-top: 0.35rem;
	}
	.filter-chip {
		border-left: 2px solid var(--rule);
		padding-left: 0.4rem;
	}
	.filter-chip span {
		color: var(--ink-muted);
		margin-right: 0.25rem;
	}
	@media (max-width: 760px) {
		header {
			align-items: flex-start;
			flex-direction: column;
			gap: 0.25rem;
		}
		table {
			min-width: 980px;
		}
	}
</style>
