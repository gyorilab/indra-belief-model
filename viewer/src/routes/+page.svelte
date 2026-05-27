<script lang="ts">
	import type { PageData } from './$types';
	import { invalidateAll } from '$app/navigation';
	import { page } from '$app/state';
	import { onMount, onDestroy } from 'svelte';
	import BeliefPrimitive from '$lib/components/BeliefPrimitive.svelte';
	import HeuristicCoverage from '$lib/components/HeuristicCoverage.svelte';
	import Validity from '$lib/components/Validity.svelte';
	import { isPairedRunBlockedCode } from '$lib/pairedRunConflicts';
	import { isWriterActionBlockedCode } from '$lib/writerActionConflicts';

	let { data }: { data: PageData } = $props();
	const o = $derived(data.overview);
	const focus = $derived(data.focus);
	const findings = $derived(data.findings);
	const residuals = $derived(data.residuals);
	const narratives = $derived(data.narratives);
	const coverage = $derived(data.coverage);
	const datasets = $derived(data.datasets);
	const pairedWorkflows = $derived(data.pairedWorkflows ?? []);
	const writerLock = $derived(data.writerLock ?? null);
	const durablePairActive = $derived(
		pairedWorkflows.some((p) => p.is_active)
	);
	const globalWriterActive = $derived(durablePairActive || writerLock != null);

	function fmtBytes(n: number): string {
		if (n < 1024) return `${n} B`;
		if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
		if (n < 1024 * 1024 * 1024) return `${(n / (1024 * 1024)).toFixed(1)} MB`;
		return `${(n / (1024 * 1024 * 1024)).toFixed(2)} GB`;
	}

	/** Slugify a filename into a valid truth_set_id / source_dump_id. */
	function slugFromPath(path: string): string {
		const base = path.split('/').pop() ?? path;
		let slug = base
			.replace(/\.(jsonl|json|json\.gz|gz)$/i, '')
			.replace(/[^A-Za-z0-9_]+/g, '_')
			.replace(/^_+|_+$/g, '')
			.toLowerCase();
		if (!slug) slug = 'dataset';
		if (!/^[a-z]/i.test(slug)) slug = `dataset_${slug}`;
		if (slug.length < 3) slug = `${slug}_set`;
		return slug.slice(0, 64);
	}

	// Per-card action state. `n_done`/`n_total` carry live progress for ingest;
	// `t_started` powers the elapsed-time readout. All optional — register and
	// other lightweight verbs use only phase + message.
	type ActionPhase = 'idle' | 'confirming' | 'running' | 'done' | 'blocked' | 'error';
	type ActionState = {
		phase: ActionPhase;
		message?: string;
		code?: string | null;
		n_done?: number;
		n_total?: number | null;
		t_started?: number;
	};
	let actionStates: Record<string, ActionState> = $state({});
	// AbortControllers for in-flight ingest streams, indexed by dataset path.
	const ingestControllers = new Map<string, AbortController>();
	const truthSetControllers = new Map<string, AbortController>();

	// Wall-clock tick for elapsed-time displays. Only advances while at
	// least one ingest or paired score run is live so the dashboard isn't doing 1Hz work
	// when nothing's in flight.
	let tickNow = $state(Date.now());
	$effect(() => {
		const anyRunning = Object.values(actionStates).some((s) => s.phase === 'running');
		if (!anyRunning) return;
		const h = setInterval(() => { tickNow = Date.now(); }, 1000);
		return () => clearInterval(h);
	});

	function setAction(path: string, state: ActionState) {
		actionStates = { ...actionStates, [path]: state };
	}

	function actionState(path: string): ActionState {
		return actionStates[path] ?? { phase: 'idle' };
	}
	function actionErrorText(state: ActionState): string {
		return state.message ?? 'action failed';
	}
	function actionBlockedText(state: ActionState): string {
		return state.message ?? 'writer action blocked';
	}

	type LocalCancelOrigin = 'user' | 'page_exit';

	function localCancelPhrase(origin: LocalCancelOrigin): string {
		return origin === 'page_exit' ? 'local stream closed because the page closed or navigated away' : 'canceled by user';
	}

	function cancelIngest(path: string, origin: LocalCancelOrigin = 'user') {
		const ctrl = ingestControllers.get(path);
		if (ctrl) {
			ctrl.abort();
			ingestControllers.delete(path);
		}
		const cur = actionState(path);
		if (cur.phase === 'running') {
			const done = cur.n_done ?? 0;
			const total = cur.n_total;
			// Honest about partial state: the worker writes statements one
			// at a time; SIGTERM/SIGKILL leaves whatever it had committed
			// in the corpus. Re-ingest is idempotent on stmt_hash so a
			// retry resumes cleanly — but the user needs to know the rows
			// aren't gone just because they hit cancel.
			const tail = done > 0
				? ` — those ${done.toLocaleString()} statements are committed in corpus.duckdb (idempotent re-ingest will resume)`
				: '';
			setAction(path, {
				phase: 'error',
				message: `${localCancelPhrase(origin)} at ${done.toLocaleString()}${total ? '/' + total.toLocaleString() : ''}${tail}`
			});
		}
	}

	function cancelTruthSet(path: string, origin: LocalCancelOrigin = 'user') {
		const ctrl = truthSetControllers.get(path);
		if (ctrl) {
			ctrl.abort();
			truthSetControllers.delete(path);
		}
		const cur = actionState(path);
		if (cur.phase === 'running') {
			setAction(path, {
				phase: 'error',
				message: `${localCancelPhrase(origin)} during truth_set registration${cur.t_started ? ` after ${Math.round((Date.now() - cur.t_started) / 1000)}s` : ''}`
			});
		}
	}

	async function registerAsTruthSet(d: { path: string; filename: string }) {
		const truth_set_id = slugFromPath(d.path);
		const truth_set_name = d.filename;
		const ctrl = new AbortController();
		truthSetControllers.set(d.path, ctrl);
		setAction(d.path, { phase: 'running', message: 'registering truth_set', t_started: Date.now() });
		try {
			const res = await fetch('/api/truth-sets', {
				method: 'POST',
				headers: { 'content-type': 'application/json' },
				body: JSON.stringify({
					path: d.path,
					truth_set_id,
					truth_set_name,
					target_kind: 'evidence',
					field: 'tag'
				}),
				signal: ctrl.signal
			});
			if (!res.ok) {
				const failure = await responseErrorPayload(res);
				setAction(d.path, {
					phase: isWriterActionBlockedCode(failure.code) ? 'blocked' : 'error',
					code: failure.code,
					message: failure.message.slice(0, 200)
				});
				return;
			}
			const body = await res.json();
			const sum = body?.summary as { n_loaded?: number; n_unique_targets?: number; n_unique_labels?: number; n_replaced_labels?: number; n_missing_target?: number; n_missing_relation_target?: number; duration_s?: number } | null;
			const labelCount = sum?.n_unique_labels ?? sum?.n_unique_targets ?? sum?.n_loaded;
			const collapsed = sum?.n_loaded != null && labelCount != null && labelCount < sum.n_loaded
				? ` (${sum.n_loaded - labelCount} duplicate label rows collapsed)`
				: '';
			const replaced = sum?.n_replaced_labels ? ` · replaced ${sum.n_replaced_labels} prior labels atomically` : '';
			const contextMiss = sum?.n_missing_relation_target ? ` · ${sum.n_missing_relation_target} missing statement context` : '';
			setAction(d.path, {
				phase: 'done',
				message: sum
					? `registered tag truth_set_id=${truth_set_id} · ${labelCount ?? '?'} active labels${collapsed}${replaced} · tag=correct is the validity positive class when scored evidence overlaps · ${sum.n_missing_target ?? 0} missing target${contextMiss} · ${sum.duration_s ?? '?'}s`
					: 'registered'
			});
			await invalidateAll();
		} catch (err) {
			const cur = actionState(d.path);
			if ((err as Error).name === 'AbortError' && cur.phase === 'error') {
				// cancelTruthSet already made the local cancellation visible.
			} else {
				setAction(d.path, { phase: 'error', message: String(err).slice(0, 200) });
			}
		} finally {
			truthSetControllers.delete(d.path);
		}
	}

	// Cost preflight + score state per dataset path
	type ScoreArchitecture = 'decomposed' | 'monolithic';
	const ARCH_LANES: ScoreArchitecture[] = ['monolithic', 'decomposed'];
	type CostEstimate = { model_id: string; architecture: ScoreArchitecture; cost_usd: number; n_stmts: number; n_evidences_est: number; n_llm_calls_est: number };
	type PairedEstimate = {
		model_id: string;
		monolithic: CostEstimate;
		decomposed: CostEstimate;
		total_cost_usd: number;
		total_cap_usd: number;
	};
	type PairedArchState = {
		state: 'queued' | 'running' | 'done' | 'canceled' | 'crashed' | 'blocked';
		cost_cap_usd: number;
		cost_so_far_usd: number | null;
		n_evidences_done: number;
		n_evidences_total: number | null;
		latest_stmt: string | null;
		run_id: string | null;
		duration_s: number | null;
		message: string | null;
		running_started_at: number | null;
		last_event_at: number | null;
	};
	type PairedState =
		| { phase: 'idle' }
		| { phase: 'estimating' }
		| { phase: 'estimated'; estimates: PairedEstimate[] }
		| { phase: 'scoring'; pair_id: string; model: string; total_cap_usd: number; t_started: number; architectures: Record<ScoreArchitecture, PairedArchState> }
		| { phase: 'done'; pair_id: string; model: string; href: string; monolithic_run_id: string | null; decomposed_run_id: string | null; duration_s: number }
		| { phase: 'canceled'; pair_id: string | null; message: string }
		| { phase: 'blocked'; message: string; code: string | null; pair_id?: string | null; href?: string | null }
		| { phase: 'error'; message: string; pair_id?: string | null; href?: string | null };
	type DurablePairedWorkflow = PageData['pairedWorkflows'][number];
	type DurablePairedArch = DurablePairedWorkflow['architectures'][ScoreArchitecture];
	type WriterLock = NonNullable<PageData['writerLock']>;
	type PreflightState =
		| { phase: 'idle' }
		| { phase: 'estimating' }
		| { phase: 'estimated'; architecture: ScoreArchitecture; estimates: CostEstimate[] }
		| { phase: 'scoring'; architecture: ScoreArchitecture; model: string; cost_cap_usd: number; cost_so_far_usd: number | null; n_evidences_done: number; n_evidences_total: number | null; latest_stmt: string | null; t_started: number }
		| { phase: 'scored'; architecture: ScoreArchitecture; run_id: string; model: string; cost_cap_usd: number; n_evidences: number; duration_s: number }
		| { phase: 'blocked'; architecture: ScoreArchitecture; code: string | null; message: string }
		| { phase: 'error'; message: string };
	let preflightStates: Record<string, PreflightState> = $state({});
	let preflightArchitectures: Record<string, ScoreArchitecture> = $state({});
	let pairedStates: Record<string, PairedState> = $state({});
	let durablePairCancelErrors: Record<string, string> = $state({});
	let clonePairSourcePath = $state<string | null>(null);
	let clonePairModel = $state<string | null>(null);
	let clonePairScorer = $state<string | null>(null);
	// AbortControllers for in-flight score runs, indexed by dataset path.
	// Kept outside $state because AbortController isn't a plain JSON value.
	const scoreControllers = new Map<string, AbortController>();
	const pairedControllers = new Map<string, AbortController>();
	$effect(() => {
		const anyPairedRunning = durablePairActive || Object.values(pairedStates).some((s) => s.phase === 'scoring');
		if (!anyPairedRunning) return;
		const h = setInterval(() => { tickNow = Date.now(); }, 1000);
		return () => clearInterval(h);
	});
	function setPre(path: string, st: PreflightState) {
		preflightStates = { ...preflightStates, [path]: st };
	}
	function setPair(path: string, st: PairedState) {
		pairedStates = { ...pairedStates, [path]: st };
	}
	function preState(path: string): PreflightState {
		return preflightStates[path] ?? { phase: 'idle' };
	}
	function pairState(path: string): PairedState {
		return pairedStates[path] ?? { phase: 'idle' };
	}
	function pairBusy(st: PairedState): boolean {
		return st.phase === 'estimating' || st.phase === 'scoring';
	}
	function pairedRequestBlocked(code: string | null): boolean {
		return isPairedRunBlockedCode(code);
	}
	async function responseErrorPayload(res: Response): Promise<{ code: string | null; message: string }> {
		const text = await res.text();
		try {
			const parsed = JSON.parse(text) as { code?: unknown; message?: unknown };
			if (typeof parsed.message === 'string') {
				return {
					code: typeof parsed.code === 'string' ? parsed.code : null,
					message: parsed.message
				};
			}
		} catch {
			// Plain-text SvelteKit errors and non-JSON proxies still land here.
		}
		return { code: null, message: text || `HTTP ${res.status}` };
	}
	function preflightArchitecture(path: string, pre?: PreflightState): ScoreArchitecture {
		if (pre && 'architecture' in pre) return pre.architecture;
		return preflightArchitectures[path] ?? 'decomposed';
	}
	function setPreflightArchitecture(path: string, architecture: ScoreArchitecture) {
		const cur = preState(path);
		if (cur.phase === 'scoring' || cur.phase === 'estimating') return;
		preflightArchitectures = { ...preflightArchitectures, [path]: architecture };
		if (cur.phase !== 'idle') {
			setPre(path, { phase: 'idle' });
		}
	}
	function cancelScore(path: string, origin: LocalCancelOrigin = 'user') {
		const ctrl = scoreControllers.get(path);
		if (ctrl) {
			ctrl.abort();
			scoreControllers.delete(path);
		}
		const cur = preState(path);
		if (cur.phase === 'scoring') {
			setPre(path, {
				phase: 'error',
				message: `${localCancelPhrase(origin)} after ${cur.n_evidences_done} evidences (~${Math.round((Date.now() - cur.t_started) / 1000)}s)`
			});
		}
	}
	type PairedCancelStep = {
		architecture?: string;
		pid?: number;
		run_id?: string | null;
		ok?: boolean;
		error?: string;
	};
	type PairedCancelBody = {
		ok?: boolean;
		killed?: unknown;
		tombstones?: unknown;
	};
	function pairedCancelSteps(value: unknown): PairedCancelStep[] {
		if (!Array.isArray(value)) return [];
		return value.filter((v): v is PairedCancelStep => typeof v === 'object' && v != null);
	}
	function pairedCancelFailureSummary(body: PairedCancelBody): string | null {
		const failures = [
			...pairedCancelSteps(body.killed)
				.filter((s) => s.ok === false)
				.map((s) => `${s.architecture ?? 'worker'} kill failed${s.pid != null ? ` pid ${s.pid}` : ''}${s.error ? `: ${s.error}` : ''}`),
			...pairedCancelSteps(body.tombstones)
				.filter((s) => s.ok === false)
				.map((s) => `${s.architecture ?? 'worker'} tombstone failed${s.run_id ? ` run ${s.run_id.slice(0, 8)}` : ''}${s.error ? `: ${s.error}` : ''}`)
		];
		if (body.ok === false) failures.unshift('cancel endpoint returned ok=false');
		return failures.length ? failures.slice(0, 2).join('; ') : null;
	}
	async function pairedCancelFailureFromResponse(res: Response): Promise<string | null> {
		const text = await res.text();
		let body: PairedCancelBody | null = null;
		if (text) {
			try {
				const parsed = JSON.parse(text) as unknown;
				if (typeof parsed === 'object' && parsed != null) body = parsed as PairedCancelBody;
			} catch {
				if (!res.ok) return text.slice(0, 200);
			}
		}
		if (!res.ok) {
			return text.trim() ? text.trim().slice(0, 200) : `cancel failed with HTTP ${res.status}`;
		}
		return body ? pairedCancelFailureSummary(body) : null;
	}
	function cancelPairedScore(path: string, origin: LocalCancelOrigin = 'user') {
		const ctrl = pairedControllers.get(path);
		const cur = pairState(path);
		if (cur.phase === 'scoring') {
			// pagehide and onDestroy can both run during navigation; only the
			// scoring phase is allowed to emit the durable cancel POST.
			fetch(`/api/runs/score-paired/${cur.pair_id}/cancel`, {
				method: 'POST',
				keepalive: origin === 'page_exit'
			}).then(async (res) => {
				const failure = await pairedCancelFailureFromResponse(res);
				if (failure && origin === 'user') {
					setPair(path, {
						phase: 'error',
						message: `paired cancel incomplete: ${failure}`,
						pair_id: cur.pair_id,
						href: `/pairs/${cur.pair_id}`
					});
				}
			}).catch((err) => {
				if (origin === 'user') {
					setPair(path, {
						phase: 'error',
						message: `paired cancel request failed: ${String(err).slice(0, 180)}`,
						pair_id: cur.pair_id,
						href: `/pairs/${cur.pair_id}`
					});
				}
				// Abort below is the fallback for dev-server disconnect behavior.
			});
			if (ctrl) {
				ctrl.abort();
				pairedControllers.delete(path);
			}
			setPair(path, {
				phase: 'canceled',
				pair_id: cur.pair_id,
				message: `${localCancelPhrase(origin)} after ${Math.round((Date.now() - cur.t_started) / 1000)}s`
			});
			// Keep the terminal state visible via the canceled message; server-side
			// SIGTERM prevents further spend, and any completed run remains linked
			// through the pair workbench once the dashboard refreshes.
		} else if (ctrl) {
			ctrl.abort();
			pairedControllers.delete(path);
		}
	}
	function cancelAllLocalStreams(origin: LocalCancelOrigin = 'page_exit') {
		// Single score and ingest stop through fetch abort/request-disconnect
		// propagation. Paired score also has a durable cancel endpoint because it
		// owns queued architecture state outside this component.
		for (const path of [...ingestControllers.keys()]) cancelIngest(path, origin);
		for (const path of [...truthSetControllers.keys()]) cancelTruthSet(path, origin);
		for (const path of [...scoreControllers.keys()]) cancelScore(path, origin);
		for (const path of [...pairedControllers.keys()]) cancelPairedScore(path, origin);
	}
	async function cancelDurablePair(pair_id: string) {
		durablePairCancelErrors = { ...durablePairCancelErrors, [pair_id]: '' };
		try {
			const res = await fetch(`/api/runs/score-paired/${pair_id}/cancel`, { method: 'POST' });
			const failure = await pairedCancelFailureFromResponse(res);
			if (failure) throw new Error(failure);
			await invalidateAll();
		} catch (err) {
			durablePairCancelErrors = { ...durablePairCancelErrors, [pair_id]: String(err).slice(0, 200) };
		}
	}

	function fmtCost(c: number): string {
		if (c < 0.01) return '<$0.01';
		if (c < 1) return '$' + c.toFixed(2);
		if (c < 100) return '$' + c.toFixed(2);
		return '$' + c.toFixed(0);
	}
	function fmtEta(ms: number | null): string {
		if (ms == null || !Number.isFinite(ms) || ms < 0) return 'eta —';
		const s = Math.round(ms / 1000);
		if (s < 60) return `eta ${s}s`;
		return `eta ${Math.floor(s / 60)}m ${s % 60}s`;
	}
	function parseTimeMs(value: string | null | undefined): number | null {
		if (!value) return null;
		const ms = Date.parse(value);
		return Number.isFinite(ms) ? ms : null;
	}
	function fmtAge(ms: number | null): string {
		if (ms == null || !Number.isFinite(ms) || ms < 0) return 'time —';
		const s = Math.max(0, Math.round(ms / 1000));
		if (s < 60) return `${s}s`;
		if (s < 3600) return `${Math.floor(s / 60)}m ${s % 60}s`;
		const h = Math.floor(s / 3600);
		const m = Math.floor((s % 3600) / 60);
		return `${h}h ${m}m`;
	}
	function since(value: string | null | undefined): number | null {
		const ms = parseTimeMs(value);
		return ms == null ? null : tickNow - ms;
	}
	function durablePairStateKind(p: DurablePairedWorkflow): 'active' | 'stale' | 'terminal' {
		if (p.is_active) return 'active';
		return p.status === 'queued' || p.status === 'running' ? 'stale' : 'terminal';
	}
	function durablePairStateLabel(p: DurablePairedWorkflow): string {
		const kind = durablePairStateKind(p);
		if (kind === 'active') return 'active';
		if (kind === 'stale') return 'stale sidecar';
		return 'terminal';
	}
	function durablePairClockText(p: DurablePairedWorkflow): string {
		const basis = p.finished_at ?? p.started_at ?? p.created_at;
		const verb = p.finished_at ? 'finished' : p.started_at ? 'started' : 'created';
		const parts = [`${verb} ${fmtAge(since(basis))} ago`, `updated ${fmtAge(since(p.updated_at))} ago`];
		if (p.termination_reason) parts.push(p.termination_reason.slice(0, 120));
		return parts.join(' · ');
	}
	function durableArchEtaText(a: DurablePairedArch): string {
		if (a.status !== 'running') return '';
		const started = parseTimeMs(a.started_at);
		const total = a.n_evidences_total;
		const done = a.n_evidences_done;
		if (started == null || total == null || total <= 0 || done <= 0) return 'eta pending';
		return fmtEta(((tickNow - started) / Math.max(done, 1)) * (total - done));
	}
	function durableArchStallText(a: DurablePairedArch): string {
		if (a.status !== 'running') return '';
		const elapsed = since(a.updated_at);
		return elapsed != null && elapsed > 30000 ? ' · no progress >30s' : '';
	}
	function pairIdForLaunch(): string {
		const stamp = new Date().toISOString().replace(/\D/g, '').slice(0, 14);
		const rand = Math.random().toString(16).slice(2, 10);
		return `pair_${stamp}_${rand}`;
	}
	function isClonePairTarget(path: string): boolean {
		return clonePairSourcePath === path;
	}
	function clonePairTargetVisible(): boolean {
		return Boolean(clonePairSourcePath && datasets.some((d) => isClonePairTarget(d.path)));
	}
	function freshArchState(cost_cap_usd: number): PairedArchState {
		return {
			state: 'queued',
			cost_cap_usd,
			cost_so_far_usd: null,
			n_evidences_done: 0,
			n_evidences_total: null,
			latest_stmt: null,
			run_id: null,
			duration_s: null,
			message: null,
			running_started_at: null,
			last_event_at: null
		};
	}
	function pairProgressText(a: PairedArchState): string {
		const total = a.n_evidences_total;
		const done = a.n_evidences_done;
		const progress = total != null ? `${done}/${total} ev` : `${done} ev`;
		if (a.state === 'queued') return 'queued for DuckDB writer lock';
		if (a.state === 'running') {
			const elapsed = a.last_event_at ? tickNow - a.last_event_at : 0;
			const eta = total && done > 0
				? fmtEta(((tickNow - (a.running_started_at ?? tickNow)) / Math.max(done, 1)) * (total - done))
				: 'eta pending';
			const stall = elapsed > 30000 ? ' · no progress >30s' : '';
			const spend = a.cost_so_far_usd != null ? ` · spent ${fmtCost(a.cost_so_far_usd)}/${fmtCost(a.cost_cap_usd)}` : '';
			return `${progress}${spend} · ${eta}${a.latest_stmt ? ` · latest ${a.latest_stmt.slice(0, 8)}` : ''}${stall}`;
		}
		if (a.state === 'done') return `done${a.run_id ? ` · run ${a.run_id.slice(0, 8)}` : ''}${a.duration_s != null ? ` · ${a.duration_s.toFixed(1)}s` : ''}`;
		if (a.state === 'blocked') return a.message ? `blocked · ${a.message}` : 'blocked';
		return a.message ?? a.state;
	}
	function durablePairActiveState(p: DurablePairedWorkflow): boolean {
		return p.is_active;
	}
	function durableArchText(a: DurablePairedArch): string {
		const total = a.n_evidences_total;
		const done = a.n_evidences_done;
		const progress = total != null ? `${done}/${total} ev` : `${done} ev`;
		if (a.status === 'queued') return 'queued';
		if (a.status === 'loading') return `loading corpus · pid ${a.pid ?? '-'}`;
		if (a.status === 'running') {
			const spend = a.cost_so_far_usd != null ? ` · spent ${fmtCost(a.cost_so_far_usd)}/${fmtCost(a.cost_threshold_usd)}` : '';
			const eta = durableArchEtaText(a);
			return `${progress}${spend}${eta ? ` · ${eta}` : ''}${a.latest_stmt_hash ? ` · latest ${a.latest_stmt_hash.slice(0, 8)}` : ''}${durableArchStallText(a)}${a.pid ? ` · pid ${a.pid}` : ''}`;
		}
		if (a.status === 'succeeded') return `done${a.run_id ? ` · run ${a.run_id.slice(0, 8)}` : ''}`;
		if (a.status === 'canceled') return a.error && a.error !== 'canceled_by_user' ? `canceled · ${a.error.slice(0, 80)}` : 'canceled';
		if (a.status === 'blocked') return a.error ? `blocked · ${a.error.slice(0, 80)}` : 'blocked';
		return a.error ? `failed · ${a.error.slice(0, 80)}` : 'failed';
	}
	function writerLockKindText(kind: string): string {
		if (kind === 'malformed') return 'writer lock needs repair';
		if (kind === 'single_score') return 'single score';
		if (kind === 'paired_score') return 'paired score';
		if (kind === 'truth_set') return 'truth set';
		if (kind === 'ingest') return 'ingest';
		return kind;
	}
	function writerLockText(lock: WriterLock): string {
		const bits = [
			writerLockKindText(lock.kind),
			lock.label && lock.label !== lock.kind ? lock.label : null,
			lock.architecture ? archMark(lock.architecture) : null,
			lock.model,
			lock.pair_id ? `pair ${lock.pair_id}` : null,
			lock.pid != null ? `pid ${lock.pid}` : null,
			lock.malformed_reason
		].filter(Boolean);
		return bits.join(' · ');
	}

	async function estimateCost(d: { path: string }) {
		const architecture = preflightArchitecture(d.path);
		setPre(d.path, { phase: 'estimating' });
		try {
			const res = await fetch('/api/runs/estimate-cost', {
				method: 'POST',
				headers: { 'content-type': 'application/json' },
				body: JSON.stringify({ path: d.path, arch: architecture })
			});
			const body = await res.json();
			if (!res.ok) {
				setPre(d.path, { phase: 'error', message: body?.stderr?.slice?.(0, 200) ?? 'estimate failed' });
				return;
			}
			const estimates = (body?.summary?.estimates as CostEstimate[]) ?? [];
			setPre(d.path, { phase: 'estimated', architecture, estimates });
		} catch (err) {
			setPre(d.path, { phase: 'error', message: String(err).slice(0, 200) });
		}
	}

	async function estimatePairedCost(d: { path: string }) {
		setPair(d.path, { phase: 'estimating' });
		try {
			const estimateFor = async (arch: ScoreArchitecture): Promise<CostEstimate[]> => {
				const res = await fetch('/api/runs/estimate-cost', {
					method: 'POST',
					headers: { 'content-type': 'application/json' },
					body: JSON.stringify({ path: d.path, arch })
				});
				const body = await res.json();
				if (!res.ok) {
					throw new Error(body?.stderr?.slice?.(0, 200) ?? `estimate failed for ${arch}`);
				}
				return (body?.summary?.estimates as CostEstimate[]) ?? [];
			};
			const [monolithic, decomposed] = await Promise.all([
				estimateFor('monolithic'),
				estimateFor('decomposed')
			]);
			const monoByModel = new Map(monolithic.map((e) => [e.model_id, e]));
			const estimates = decomposed
				.filter((dEst) => monoByModel.has(dEst.model_id))
				.map((dEst): PairedEstimate => {
					const mEst = monoByModel.get(dEst.model_id)!;
					const total = mEst.cost_usd + dEst.cost_usd;
					const monoCap = Math.max(0.01, mEst.cost_usd * 1.25);
					const decompCap = Math.max(0.01, dEst.cost_usd * 1.25);
					return {
						model_id: dEst.model_id,
						monolithic: mEst,
						decomposed: dEst,
						total_cost_usd: total,
						total_cap_usd: monoCap + decompCap
					};
				});
			if (estimates.length === 0) {
				setPair(d.path, { phase: 'error', message: 'no model appears in both architecture cost estimates' });
				return;
			}
			setPair(d.path, { phase: 'estimated', estimates });
		} catch (err) {
			setPair(d.path, { phase: 'error', message: String(err).slice(0, 200) });
		}
	}

	function updatePairArch(path: string, architecture: ScoreArchitecture, patch: Partial<PairedArchState>) {
		const cur = pairState(path);
		if (cur.phase !== 'scoring') return;
		const current = cur.architectures[architecture];
		const runningStartedAt =
			patch.running_started_at ??
			(patch.state === 'running' && current.running_started_at == null ? Date.now() : current.running_started_at);
		setPair(path, {
			...cur,
			architectures: {
				...cur.architectures,
				[architecture]: {
					...current,
					...patch,
					running_started_at: runningStartedAt
				}
			}
		});
	}

	function eventArchitecture(ev: Record<string, unknown>): ScoreArchitecture | null {
		return ev.architecture === 'monolithic' || ev.architecture === 'decomposed'
			? ev.architecture
			: null;
	}

	async function scorePairedCorpus(d: { path: string; filename: string }, estimate: PairedEstimate) {
		const source_dump_id = slugFromPath(d.path);
		const scorer_version = 'prod-v1';
		const pair_id = pairIdForLaunch();
		const total_cap_usd = estimate.total_cap_usd;
		setPair(d.path, {
			phase: 'scoring',
			pair_id,
			model: estimate.model_id,
			total_cap_usd,
			t_started: Date.now(),
			architectures: {
				monolithic: freshArchState(Math.max(0.01, estimate.monolithic.cost_usd * 1.25)),
				decomposed: freshArchState(Math.max(0.01, estimate.decomposed.cost_usd * 1.25))
			}
		});
		const ctrl = new AbortController();
		pairedControllers.set(d.path, ctrl);
		try {
			const monoCap = Math.max(0.01, estimate.monolithic.cost_usd * 1.25);
			const decompCap = Math.max(0.01, estimate.decomposed.cost_usd * 1.25);
			const res = await fetch('/api/runs/score-paired', {
				method: 'POST',
				headers: { 'content-type': 'application/json' },
				body: JSON.stringify({
					path: d.path,
					source_dump_id,
					model: estimate.model_id,
					scorer_version,
					paired_run_group_id: pair_id,
					monolithic_cost_threshold_usd: monoCap,
					decomposed_cost_threshold_usd: decompCap,
					total_cost_threshold_usd: total_cap_usd
				}),
				signal: ctrl.signal
			});
			if (!res.ok || !res.body) {
				const failure = await responseErrorPayload(res);
				if (pairedRequestBlocked(failure.code)) {
					setPair(d.path, { phase: 'blocked', code: failure.code, message: failure.message.slice(0, 300) });
				} else {
					setPair(d.path, { phase: 'error', message: failure.message.slice(0, 300) });
				}
				return;
			}
			const reader = res.body.getReader();
			const decoder = new TextDecoder();
			let buf = '';
			while (true) {
				const { value, done } = await reader.read();
				if (done) break;
				buf += decoder.decode(value, { stream: true });
				let nl: number;
				while ((nl = buf.indexOf('\n\n')) >= 0) {
					const block = buf.slice(0, nl);
					buf = buf.slice(nl + 2);
					const dataLine = block.split('\n').find((l) => l.startsWith('data: '));
					if (!dataLine) continue;
					let ev: Record<string, unknown>;
					try {
						ev = JSON.parse(dataLine.slice(6));
					} catch {
						continue;
					}
					const t = ev.event as string;
					const arch = eventArchitecture(ev);
					if (t === 'arch_state' && arch) {
						const state = String(ev.state ?? 'running') as PairedArchState['state'];
						const curPair = pairState(d.path);
						const prevRunId = curPair.phase === 'scoring' ? curPair.architectures[arch].run_id : null;
						updatePairArch(d.path, arch, {
							state,
							message: typeof ev.stderr === 'string' ? ev.stderr.slice(0, 200) : typeof ev.reason === 'string' ? ev.reason : null,
							run_id: typeof ev.run_id === 'string' ? ev.run_id : prevRunId,
							last_event_at: Date.now()
						});
					} else if (t === 'arch_loaded' && arch) {
						updatePairArch(d.path, arch, {
							state: 'running',
							n_evidences_total: typeof ev.n_evidences === 'number' ? ev.n_evidences : null,
							last_event_at: Date.now()
						});
					} else if (t === 'arch_progress' && arch) {
						updatePairArch(d.path, arch, {
							state: 'running',
							n_evidences_done: Number(ev.n_evidences_done ?? 0),
							cost_so_far_usd: typeof ev.cost_so_far_usd === 'number' ? ev.cost_so_far_usd : undefined,
							cost_cap_usd: typeof ev.cost_cap_usd === 'number' ? ev.cost_cap_usd : undefined,
							latest_stmt: (ev.latest_stmt_hash as string) ?? null,
							last_event_at: Date.now()
						});
					} else if (t === 'arch_done' && arch) {
						updatePairArch(d.path, arch, {
							state: 'done',
							run_id: String(ev.run_id),
							n_evidences_done: Number(ev.n_evidences_done ?? 0),
							duration_s: Number(ev.duration_s ?? 0),
							last_event_at: Date.now()
						});
					} else if (t === 'paired_done') {
						const href = String(ev.href ?? `/pairs/${pair_id}`);
						const curPair = pairState(d.path);
						const tStarted = curPair.phase === 'scoring' ? curPair.t_started : Date.now();
						setPair(d.path, {
							phase: 'done',
							pair_id,
							model: estimate.model_id,
							href,
							monolithic_run_id: typeof ev.monolithic_run_id === 'string' ? ev.monolithic_run_id : null,
							decomposed_run_id: typeof ev.decomposed_run_id === 'string' ? ev.decomposed_run_id : null,
							duration_s: Math.round((Date.now() - tStarted) / 1000)
						});
						await invalidateAll();
						window.location.href = href;
					} else if (t === 'paired_failed') {
						const code = typeof ev.code === 'string' ? ev.code : null;
						const message = String(ev.error ?? ev.failed_architecture ?? 'paired run failed').slice(0, 300);
						const href = typeof ev.href === 'string' ? ev.href : `/pairs/${pair_id}`;
						if (pairedRequestBlocked(code)) {
							setPair(d.path, { phase: 'blocked', code, pair_id, href, message });
						} else {
							setPair(d.path, {
								phase: 'error',
								pair_id,
								href,
								message
							});
						}
					} else if (t === 'paired_canceled') {
						setPair(d.path, { phase: 'canceled', pair_id, message: String(ev.reason ?? 'canceled') });
					}
				}
			}
		} catch (err) {
			const cur = pairState(d.path);
			if ((err as Error).name === 'AbortError' && cur.phase === 'canceled') {
				// cancelPairedScore already made the local tombstone visible.
			} else {
				setPair(d.path, { phase: 'error', message: String(err).slice(0, 200) });
			}
		} finally {
			pairedControllers.delete(d.path);
		}
	}

	async function scoreCorpus(d: { path: string; filename: string }, model: string, estimatedCostUsd: number, architecture: ScoreArchitecture) {
		const source_dump_id = slugFromPath(d.path);
		const scorer_version = 'prod-v1';
		// Pass the projected cost as a hard cap so the worker aborts if its own
		// upfront estimate (run against the same file) lands materially higher
		// — a check against the file/estimator drifting since preview.
		const cost_threshold_usd = Math.max(0.01, estimatedCostUsd * 1.25);
		const cost_cap_usd = cost_threshold_usd;
		setPre(d.path, {
			phase: 'scoring',
			architecture,
			model,
			cost_cap_usd,
			cost_so_far_usd: null,
			n_evidences_done: 0,
			n_evidences_total: null,
			latest_stmt: null,
			t_started: Date.now()
		});
		const ctrl = new AbortController();
		scoreControllers.set(d.path, ctrl);
		try {
			const res = await fetch('/api/runs/score', {
				method: 'POST',
				headers: { 'content-type': 'application/json' },
				body: JSON.stringify({
					path: d.path,
					source_dump_id,
					model,
					scorer_version,
					arch: architecture,
					cost_threshold_usd
				}),
				signal: ctrl.signal
			});
			if (!res.ok || !res.body) {
				const failure = await responseErrorPayload(res);
				if (isWriterActionBlockedCode(failure.code)) {
					setPre(d.path, {
						phase: 'blocked',
						architecture,
						code: failure.code,
						message: failure.message.slice(0, 300)
					});
				} else {
					setPre(d.path, { phase: 'error', message: failure.message.slice(0, 300) });
				}
				return;
			}
			// SSE-ish stream — server emits `data: <json>\n\n` per event
			const reader = res.body.getReader();
			const decoder = new TextDecoder();
			let buf = '';
			while (true) {
				const { value, done } = await reader.read();
				if (done) break;
				buf += decoder.decode(value, { stream: true });
				let nl: number;
				while ((nl = buf.indexOf('\n\n')) >= 0) {
					const block = buf.slice(0, nl);
					buf = buf.slice(nl + 2);
					const dataLine = block.split('\n').find((l) => l.startsWith('data: '));
					if (!dataLine) continue;
					let ev: Record<string, unknown>;
					try {
						ev = JSON.parse(dataLine.slice(6));
					} catch {
						continue;
					}
					const t = ev.event as string;
					if (t === 'loaded') {
						const cur = preState(d.path);
						if (cur.phase === 'scoring') {
							// Honest denominator from the worker; null if absent
							const n_total = typeof ev.n_evidences === 'number' ? ev.n_evidences : null;
							setPre(d.path, { ...cur, n_evidences_total: n_total });
						}
					} else if (t === 'progress') {
						const cur = preState(d.path);
						if (cur.phase === 'scoring') {
							setPre(d.path, {
								...cur,
								n_evidences_done: Number(ev.n_evidences_done),
								cost_so_far_usd: typeof ev.cost_so_far_usd === 'number' ? ev.cost_so_far_usd : cur.cost_so_far_usd,
								cost_cap_usd: typeof ev.cost_cap_usd === 'number' ? ev.cost_cap_usd : cur.cost_cap_usd,
								latest_stmt: (ev.latest_stmt_hash as string) ?? null
							});
						}
					} else if (t === 'done') {
						setPre(d.path, {
							phase: 'scored',
							architecture,
							run_id: String(ev.run_id),
							model,
							cost_cap_usd,
							n_evidences: Number(ev.n_evidences_done),
							duration_s: Number(ev.duration_s)
						});
						await invalidateAll();
					} else if (t === 'error') {
						setPre(d.path, {
							phase: 'error',
							message: String(ev.stderr ?? ev.error ?? 'score failed').slice(0, 400)
						});
					}
				}
			}
		} catch (err) {
			// AbortError is the user-canceled path; cancelScore() set state
			const cur = preState(d.path);
			if ((err as Error).name === 'AbortError' && cur.phase === 'error') {
				// already handled in cancelScore()
			} else {
				setPre(d.path, { phase: 'error', message: String(err).slice(0, 200) });
			}
		} finally {
			scoreControllers.delete(d.path);
		}
	}

	async function ingestCorpus(d: { path: string; filename: string }) {
		const source_dump_id = slugFromPath(d.path);
		setAction(d.path, {
			phase: 'running',
			n_done: 0,
			n_total: null,
			t_started: Date.now()
		});
		const ctrl = new AbortController();
		ingestControllers.set(d.path, ctrl);
		try {
			const res = await fetch('/api/datasets/ingest', {
				method: 'POST',
				headers: { 'content-type': 'application/json' },
				body: JSON.stringify({ path: d.path, source_dump_id }),
				signal: ctrl.signal
			});
			if (!res.ok || !res.body) {
				const failure = await responseErrorPayload(res);
				setAction(d.path, {
					phase: isWriterActionBlockedCode(failure.code) ? 'blocked' : 'error',
					code: failure.code,
					message: failure.message.slice(0, 300)
				});
				return;
			}
			// SSE: server emits `data: <json>\n\n` per event.
			const reader = res.body.getReader();
			const decoder = new TextDecoder();
			let buf = '';
			while (true) {
				const { value, done } = await reader.read();
				if (done) break;
				buf += decoder.decode(value, { stream: true });
				let nl: number;
				while ((nl = buf.indexOf('\n\n')) >= 0) {
					const block = buf.slice(0, nl);
					buf = buf.slice(nl + 2);
					const dataLine = block.split('\n').find((l) => l.startsWith('data: '));
					if (!dataLine) continue;
					let ev: Record<string, unknown>;
					try {
						ev = JSON.parse(dataLine.slice(6));
					} catch {
						continue;
					}
					const t = ev.event as string;
					const cur = actionState(d.path);
					if (t === 'loaded') {
						if (cur.phase === 'running') {
							setAction(d.path, {
								...cur,
								n_total: typeof ev.n_statements === 'number' ? ev.n_statements : null
							});
						}
					} else if (t === 'progress') {
						if (cur.phase === 'running') {
							setAction(d.path, {
								...cur,
								n_done: Number(ev.n_statements_done),
								n_total: typeof ev.n_statements_total === 'number' ? ev.n_statements_total : cur.n_total
							});
						}
					} else if (t === 'done') {
						const n = typeof ev.n_statements === 'number' ? ev.n_statements : null;
						const dur = typeof ev.duration_s === 'number' ? ev.duration_s : null;
						setAction(d.path, {
							phase: 'done',
							message: n != null
								? `ingested ${n.toLocaleString()} statements as source_dump_id=${source_dump_id}${dur != null ? ` · ${dur.toFixed(1)}s` : ''}`
								: 'ingested'
						});
						await invalidateAll();
					} else if (t === 'error') {
						setAction(d.path, {
							phase: 'error',
							message: String(ev.stderr ?? ev.error ?? 'ingest failed').slice(0, 400)
						});
					}
				}
			}
		} catch (err) {
			const cur = actionState(d.path);
			if ((err as Error).name === 'AbortError' && cur.phase === 'error') {
				// cancelIngest() already set the message
			} else {
				setAction(d.path, { phase: 'error', message: String(err).slice(0, 200) });
			}
		} finally {
			ingestControllers.delete(d.path);
		}
	}

	function statusGlyph(status: string): string {
		if (status === 'succeeded') return '✓';
		if (status === 'running') return '↻';
		if (status === 'failed') return '✗';
		if (status === 'canceled') return '!';
		return '?';
	}

	function archMark(architecture: string | null | undefined): string {
		if (architecture === 'monolithic') return '[M]';
		if (architecture === 'decomposed') return '[D]';
		return '[?]';
	}
	function archLabel(architecture: string | null | undefined): string {
		if (architecture === 'monolithic') return 'monolithic';
		if (architecture === 'decomposed') return 'decomposed';
		return 'unknown';
	}

	function fmtCostSummary(cost: number): string {
		return cost < 0.01 ? '<$0.01' : cost < 1 ? '$' + cost.toFixed(2) : '$' + cost.toFixed(0);
	}

	const FINDING_LANES: Array<{ key: keyof NonNullable<typeof findings>; title: string; emptyMsg: string }> = [
		{ key: 'biggest_disagreement', title: 'we disagree most with INDRA on these', emptyMsg: 'no scored statements' },
		{ key: 'probe_split', title: 'the four probes disagreed among themselves', emptyMsg: 'no multi-probe statements' },
		{ key: 'low_confidence_high_stakes', title: 'mid-range belief, multi-evidence — worth a closer look', emptyMsg: 'no mid-range statements with multi-evidence' },
		{ key: 'verdict_regression', title: 'verdict moved correct → incorrect since prev run', emptyMsg: 'no prev run or no regressions' },
		{ key: 'verdict_recovery', title: 'verdict moved incorrect → correct since prev run', emptyMsg: 'no prev run or no recoveries' }
	];

	function fmt(n: number): string {
		return n.toLocaleString('en-US');
	}

	// Cost projection for a "score the loaded corpus" run. Source of truth
	// for prices: src/indra_belief/corpus/cost.py::MODEL_PRICES_PER_M_TOKENS.
	// Mirrored client-side for the panel; keep in sync if rates change.
	const COST_MODELS: Array<[string, string, number, number]> = [
		// [display_name, model_id, in_per_m_usd, out_per_m_usd]
		['Flash', 'gemini-2.5-flash', 0.075, 0.30],
		['Haiku', 'claude-haiku-4-5', 0.80, 4.00],
		['Sonnet', 'claude-sonnet-4-6', 3.00, 15.00],
		['Opus', 'claude-opus-4-7', 15.00, 75.00]
	];
	const TOKENS_PER_LLM_CALL_IN = 330;
	const TOKENS_PER_LLM_CALL_OUT = 70;
	const LLM_CALLS_PER_EVIDENCE = 5;
	const MONOLITHIC_LLM_CALLS_PER_EVIDENCE = 1;

	function llmCallsPerEvidence(architecture: ScoreArchitecture): number {
		return architecture === 'monolithic' ? MONOLITHIC_LLM_CALLS_PER_EVIDENCE : LLM_CALLS_PER_EVIDENCE;
	}

	function projectCost(ev_count: number, in_per_m: number, out_per_m: number, architecture: ScoreArchitecture = 'decomposed'): number {
		const calls = ev_count * llmCallsPerEvidence(architecture);
		return calls * (TOKENS_PER_LLM_CALL_IN * in_per_m + TOKENS_PER_LLM_CALL_OUT * out_per_m) / 1_000_000;
	}

	// Phase 5d minimum-viable live-tail: poll the load function on a
	// fixed cadence (3s) so the dashboard refreshes counts / latest run /
	// validity without manual reload. Lighter than SSE; richer than a
	// static page. Bret Victor: the page is a live document, not a snapshot.
	//
	// Iter-3 brutalist BLOCKER #1: the live dot used to pulse unconditionally
	// (decoration, no signal). It now flashes only when data changed since
	// last poll — flash duration <200ms per D10 motion budget.
	let pollHandle: ReturnType<typeof setInterval> | null = null;
	let freshTimer: ReturnType<typeof setTimeout> | null = null;
	let lastSignature = $state<string>('');
	let dotFresh = $state(false);

	const currentSignature = $derived(
		`${o.statementCount}|${o.evidenceCount}|${o.scorerRuns.length}|${o.scorerRuns[0]?.run_id ?? ''}|${o.latestValidity?.run_id ?? ''}`
	);

	// Flash the dot for 800ms when data signature changes, then revert.
	// Previously this used `Date.now() - dataChangedAt < 800` inside $derived,
	// but Date.now() isn't reactive — once the inequality went true, nothing
	// triggered re-evaluation, so the dot stayed "fresh" forever until the
	// next change. Explicit setTimeout makes the 800ms revert real.
	$effect(() => {
		if (lastSignature && lastSignature !== currentSignature) {
			dotFresh = true;
			if (freshTimer) clearTimeout(freshTimer);
			freshTimer = setTimeout(() => { dotFresh = false; }, 800);
		}
		lastSignature = currentSignature;
	});

	// Empty-state pipeline snippet — held in a const so f-string `${...}`
	// interpolation doesn't conflict with Svelte's `{...}` template syntax.
	const PIPELINE_SNIPPET = `import duckdb
from indra.statements import stmts_from_json_file
from indra_belief.model_client import ModelClient
from indra_belief.corpus import (
    apply_schema, ingest_statements,
    score_corpus, export_beliefs, model_card,
)

# 1. Ingest INDRA Statements (lossless)
con = duckdb.connect("data/corpus.duckdb")
apply_schema(con)
stmts = stmts_from_json_file("data/corpora/latest_statements_rasmachine.json")
ingest_statements(con, stmts, source_dump_id="rasmachine_emmaa")

# 2. Score through the four-probe pipeline (auto-runs compute_validity)
client = ModelClient("claude-sonnet-4-6")  # or any ModelClient backend
run_id = score_corpus(con, stmts, client=client,
                      scorer_version="prod-v1", decompose=True)

# 3. Export INDRA-native JSON with our beliefs + model card
export_beliefs(con, run_id, f"data/exports/{run_id}_indra.json")
model_card(con, run_id, out_path=f"data/exports/{run_id}_card.json")
con.close()`;

	function localWriterActive(): boolean {
		return (
			Object.values(actionStates).some((s) => s.phase === 'running') ||
			Object.values(preflightStates).some((s) => s.phase === 'scoring') ||
			Object.values(pairedStates).some((s) => s.phase === 'scoring')
		);
	}

	onMount(() => {
		const cancelForPageExit = (event: PageTransitionEvent) => {
			if (event.persisted) return;
			cancelAllLocalStreams('page_exit');
		};
		clonePairSourcePath = page.url.searchParams.get('pair_source');
		clonePairModel = page.url.searchParams.get('pair_model');
		clonePairScorer = page.url.searchParams.get('pair_scorer');
		lastSignature = currentSignature;
		pollHandle = setInterval(() => {
			if (localWriterActive()) return;
			invalidateAll();
		}, 3000);
		window.addEventListener('pagehide', cancelForPageExit);
		return () => {
			window.removeEventListener('pagehide', cancelForPageExit);
		};
	});

	onDestroy(() => {
		cancelAllLocalStreams('page_exit');
		if (pollHandle) clearInterval(pollHandle);
		if (freshTimer) clearTimeout(freshTimer);
	});
</script>

<svelte:head>
	<title>INDRA Belief — Corpus</title>
</svelte:head>

<header>
	<div class="crumb">corpus<span class="sep"> / </span><strong>overview</strong></div>
	<div class="meta">
		<span class="live-indicator" title="dashboard polls every 3s; dot flashes when data changes">
			<span class="live-dot" class:live-dot-flash={dotFresh}></span>
			{dotFresh ? 'fresh' : 'live'}
		</span>
		<a href="/statements" class="nav-link">browse statements →</a>
		<span class="db-path" title={o.dbPath}>{o.dbPath.replace(/.*\//, '')}</span>
	</div>
</header>

<main id="main">
	<h1 class="visually-hidden">Corpus dashboard</h1>
	{#if !o.dbExists}
		<section class="empty">
			<h1>no corpus loaded</h1>
			<p class="lede">
				The viewer is wired to <code>{o.dbPath}</code>, but no DuckDB file exists there yet.
			</p>
			<p>
				Run the full pipeline from a Python REPL:
			</p>
			<pre>{PIPELINE_SNIPPET}</pre>
			<p class="hint">
				Or set <code>VIEWER_DUCKDB_PATH</code> to point at an existing <code>.duckdb</code> file.
			</p>
		</section>
	{:else}
		<p class="dashboard-subtitle">
			{#if o.scorerRuns.length === 0}
				INDRA Statement belief rescorer. No runs yet — ingest a dataset and click [score] on its card below to produce the first.
			{:else}
				INDRA Statement belief rescorer. Below: the statement that disagreed most with INDRA's prior in the latest run, what changed, and where we are weakest.
			{/if}
		</p>

		<section class="focus" aria-label="focus statement — biggest disagreement with INDRA in the latest run">
			{#if focus}
				<BeliefPrimitive
					stmt={focus.stmt}
					our_score={focus.our_score}
					indra_score={focus.indra_score}
					probes={focus.probes}
					evidences={focus.evidences}
					why_this_one={focus.why_this_one}
					mode="full"
					level="h2"
				/>
				<p class="focus-deeplink">
					<a href={`/statements/${focus.stmt.stmt_hash}?run_id=${focus.run_id}`}>open deep-dive →</a>
				</p>
			{:else}
				<div class="focus-empty">
					<p class="hint">no belief in focus yet · ingest a dataset and click [score] below — or jump to <a href="#datasets">datasets on disk</a></p>
				</div>
			{/if}
		</section>

		{#if findings}
			{@const focusHash = focus?.stmt.stmt_hash ?? null}
			<section class="findings" aria-label="other notable statements from this run">
				<h2 class="visually-hidden">findings — other notable statements from this run</h2>
				{#each FINDING_LANES as lane}
					{@const allRows = (findings[lane.key] as import('$lib/db').FindingRow[]) ?? []}
					{@const rows = allRows.filter((r) => r.stmt_hash !== focusHash)}
					{#if rows.length > 0}
						<div class="lane">
							<h3 class="lane-h">{lane.title} <span class="lane-n">({rows.length})</span></h3>
							<div class="lane-body">
								{#each rows as r}
									<BeliefPrimitive
										mode="compact"
										stmt={{ stmt_hash: r.stmt_hash, indra_type: r.indra_type, agents: r.agents }}
										our_score={r.our_score}
										indra_score={r.indra_score}
										evidences={Array(r.n_evidences).fill({ evidence_hash: '', source_api: null, text: null })}
										why_this_one={r.why_text}
										href={`/?focus=${r.stmt_hash}`}
									/>
								{/each}
							</div>
						</div>
					{/if}
				{/each}
			</section>
		{/if}

		{#if o.latestValidity}
			<Validity v={o.latestValidity} residuals={residuals} />
		{/if}

		{#if coverage}
			<HeuristicCoverage {coverage} />
		{/if}

		{#if writerLock}
			<section class="writer-lock" aria-label="DuckDB writer lock">
				<div>
					<h2 class="wl-h">DuckDB writer occupied</h2>
					<p>{writerLockText(writerLock)}</p>
				</div>
				<p class="wl-note">writer actions are disabled until this worker exits; refresh after completion if this tab was not the one that started it</p>
			</section>
		{/if}

		{#if pairedWorkflows.length > 0}
			<section class="pair-workflows" aria-label="paired workflow state ledger">
				<h2 class="pw-h">paired workflow state</h2>
				<div class="pw-list">
					{#each pairedWorkflows as pw}
						<article class={`pw-row pw-row-${pw.status} pw-row-${durablePairStateKind(pw)}`}>
							<div class="pw-main">
								<a href={pw.href}><code>{pw.pair_id}</code></a>
								<span class="pw-status">{pw.status}</span>
								<span class={`pw-state-kind pw-state-kind-${durablePairStateKind(pw)}`}>{durablePairStateLabel(pw)}</span>
								<span>{pw.model}</span>
								<span>cap {fmtCost(pw.total_cost_threshold_usd)}</span>
								<span class="muted">{pw.source_dump_id}</span>
							</div>
							<div class="pw-state-rail">
								<span>{durablePairClockText(pw)}</span>
							</div>
							<div class="pw-arches">
								{#each ARCH_LANES as arch}
									{@const a = pw.architectures[arch]}
									<div class={`pw-arch pw-arch-${a.status}`}>
										<span>{archMark(arch)} {arch}</span>
										<strong>{a.status}</strong>
										<em>{durableArchText(a)}</em>
									</div>
								{/each}
							</div>
							<div class="pw-actions">
								<a href={pw.href}>open pair</a>
								{#if durablePairActiveState(pw)}
									<button type="button" onclick={() => cancelDurablePair(pw.pair_id)}>cancel</button>
								{/if}
							</div>
							{#if durablePairCancelErrors[pw.pair_id]}
								<p class="pw-error" role="alert">cancel failed: {durablePairCancelErrors[pw.pair_id]}</p>
							{/if}
						</article>
					{/each}
				</div>
			</section>
		{/if}

		<section class="grid">
			<article class="run-feed-article">
				<h2>runs</h2>
				{#if o.scorerRuns.length === 0}
					<p class="hint">no runs yet · ingest a dataset below and click its [score] action to produce the first</p>
				{:else}
					<ul class="run-feed">
						{#each o.scorerRuns as r}
							{@const n = narratives[r.run_id]}
							<li class="run-row" class:run-row-failed={r.status === 'failed'} class:run-row-running={r.status === 'running'} class:run-row-canceled={r.status === 'canceled'}>
								<span class="run-glyph" class:status-failed={r.status === 'failed'} class:status-running={r.status === 'running'} class:status-canceled={r.status === 'canceled'} title={r.status}>{statusGlyph(r.status)}</span>
								<a class="run-hash" href={`/runs/${r.run_id}`} title={r.run_id}><code>{r.run_id.slice(0, 8)}</code></a>
								<span class={`run-arch run-arch-${r.architecture}`} title={`architecture: ${r.architecture}`}>{archMark(r.architecture)}</span>
								<span class="run-version" title={r.scorer_version}>{r.scorer_version.length > 22 ? r.scorer_version.slice(0, 21) + '…' : r.scorer_version}</span>
								<span class="run-when" title={r.started_at}>{r.started_at.replace(/\.\d+$/, '').replace(/^(\d{4}-\d{2}-\d{2}) /, '$1·')}</span>
								<span class="run-narrative">
									{#if r.status !== 'succeeded'}
										<span class="muted">{r.status}{#if r.termination_reason} · {r.termination_reason}{/if}</span>
									{:else}
										<span class="run-n">{r.n_stmts ?? '—'} stmts</span>
										{#if r.cost_estimate_usd != null}
											<span class="muted">·</span>
											<span class="run-cost">{r.cost_estimate_usd < 0.01 ? '<$0.01' : '$' + r.cost_estimate_usd.toFixed(2)}</span>
										{/if}
										{#if n?.summary_sentence}
											<span class="muted">·</span>
											<span class="run-summary">{n.summary_sentence}</span>
										{:else if r.mae != null}
											<span class="muted">·</span>
											<span class="run-summary">MAE {r.mae.toFixed(3)}{#if r.bias != null} · bias {r.bias >= 0 ? '+' : '−'}{Math.abs(r.bias).toFixed(3)}{/if}</span>
										{/if}
									{/if}
								</span>
								<span class="run-exports">
									{#if r.paired_run_group_id}<a href={`/pairs/${r.paired_run_group_id}`} class="dl-link" title="paired architecture workbench">pair</a>{/if}
									{#if r.hasIndraExport}<a href={`/export/${r.run_id}/indra`} class="dl-link" title="INDRA beliefs JSON">↓ beliefs</a>{:else}<span class="dl-link dl-missing" title="Run `export_beliefs(con, run_id, ...)` first">↓ beliefs</span>{/if}
									{#if r.hasCardExport}<a href={`/export/${r.run_id}/card`} class="dl-link" title="model card JSON">↓ card</a>{:else}<span class="dl-link dl-missing" title="Run `model_card(con, run_id, out_path=...)` first">↓ card</span>{/if}
								</span>
							</li>
						{/each}
					</ul>
				{/if}
			</article>

			<details class="cost-expander">
				<summary>next-run cost projection {#if o.evidenceCount > 0}<span class="muted">— [D] Flash ≈ {fmtCostSummary(projectCost(o.evidenceCount, COST_MODELS[0][2], COST_MODELS[0][3], 'decomposed'))} · [M] Flash ≈ {fmtCostSummary(projectCost(o.evidenceCount, COST_MODELS[0][2], COST_MODELS[0][3], 'monolithic'))}</span>{/if}</summary>
				{#if o.evidenceCount === 0}
					<p class="hint">no evidences loaded · run <code>ingest_statements</code> first</p>
				{:else}
					<table>
						<thead>
							<tr><th>model</th><th class="num">[D] cost</th><th class="num">[M] cost</th><th class="num">[D] calls</th></tr>
						</thead>
						<tbody>
							{#each COST_MODELS as [name, id, in_p, out_p]}
								{@const cost = projectCost(o.evidenceCount, in_p, out_p, 'decomposed')}
								{@const monoCost = projectCost(o.evidenceCount, in_p, out_p, 'monolithic')}
								{@const calls = o.evidenceCount * LLM_CALLS_PER_EVIDENCE}
								<tr>
									<td>{name} <span class="muted">{id}</span></td>
									<td class="num">${cost < 1 ? cost.toFixed(2) : cost.toFixed(0)}</td>
									<td class="num">${monoCost < 1 ? monoCost.toFixed(2) : monoCost.toFixed(0)}</td>
									<td class="num">{fmt(calls)}</td>
								</tr>
							{/each}
						</tbody>
					</table>
					<p class="hint">
						[D] assumes ~{LLM_CALLS_PER_EVIDENCE} LLM calls/evidence; [M] assumes {MONOLITHIC_LLM_CALLS_PER_EVIDENCE} aggregate call/evidence · {TOKENS_PER_LLM_CALL_IN}+{TOKENS_PER_LLM_CALL_OUT} tokens/call.
						Pass <code>cost_threshold_usd</code> to <code>score_corpus</code> to gate before spend.
					</p>
				{/if}
			</details>

			<!-- validity moved out of grid; rendered by Validity component above -->
		</section>

		{#if datasets}
			{@const corpora = datasets.filter((d) => d.kind === 'corpus')}
			{@const benchmarks = datasets.filter((d) => d.kind === 'benchmark')}
			<section class="datasets" id="datasets">
				<h2 class="ds-h">datasets on disk</h2>
				<p class="ds-intro">JSON / JSONL files in <code>data/corpora/</code> and <code>data/benchmark/</code>. Buttons trigger Python workers that mutate <code>corpus.duckdb</code> — see the warnings on each commit button before clicking.</p>
				{#if clonePairSourcePath && !clonePairTargetVisible()}
					<p class="ds-clone-note ds-clone-note-empty">
						clone preflight from pair page
						{#if clonePairModel}<span> · previous model <code>{clonePairModel}</code></span>{/if}
						{#if clonePairScorer}<span> · scorer <code>{clonePairScorer}</code></span>{/if}
						<span> · source file is not visible in this dashboard DB root; choose a corpus below or add <code>{clonePairSourcePath}</code> under this repo's <code>data/</code> tree, then preview cost again before spend</span>
					</p>
				{/if}

				<div class="ds-group">
					<h3 class="ds-group-h">data/corpora/ <span class="muted">({corpora.length})</span></h3>
					{#if corpora.length === 0}
						<p class="ds-empty">no corpora yet — drop a JSON of INDRA statements in <code>data/corpora/</code> and refresh.</p>
					{:else}
						<ul class="ds-list">
							{#each corpora as d}
								{@const st = actionState(d.path)}
								{@const pre = preState(d.path)}
								{@const pair = pairState(d.path)}
								{@const selectedArchitecture = preflightArchitecture(d.path, pre)}
								{@const canIngest = d.shape.kind_detail === 'indra_json' && (d.ingest?.n_in_file ?? 0) > 0 && (d.ingest?.n_already_ingested ?? 0) < (d.ingest?.n_in_file ?? 0)}
								{@const canIngestGz = d.shape.kind_detail === 'json_gz'}
									<li class="ds-row" class:ds-row-clone-target={isClonePairTarget(d.path)}>
									<div class="ds-row-head">
										<code class="ds-name">{d.filename}</code>
										<span class="ds-meta">
											<span>{fmtBytes(d.size_bytes)}</span>
											<span class="muted">·</span>
											<span>{d.shape.n_records ?? '—'} {d.shape.kind_detail === 'jsonl_records' ? 'records' : d.shape.kind_detail === 'indra_json' ? 'statements' : d.shape.kind_detail === 'json_gz' ? 'compressed' : ''}</span>
											{#if d.shape.source_apis.length > 0}
												<span class="muted">·</span>
												<span>sources: {d.shape.source_apis.join(', ')}</span>
											{/if}
										</span>
										{#if d.duplicate_of && d.duplicate_of.length > 0}
											<span class="ds-badge ds-badge-dup" title={`byte-identical to ${d.duplicate_of.join(', ')} (same size + same first 16KB)`}>duplicate of {d.duplicate_of.join(', ')}</span>
										{/if}
										{#if d.ingest}
											{@const ing = d.ingest}
											{#if ing.n_in_file === 0}
												<span class="ds-badge ds-badge-unknown" title={ing.notes.join(' · ')}>ingest status unknown</span>
											{:else if ing.n_already_ingested === 0}
												<span class="ds-badge ds-badge-fresh">not ingested</span>
											{:else if ing.n_already_ingested >= ing.n_in_file}
												<span class="ds-badge ds-badge-done">{ing.sampled ? `${ing.n_already_ingested}+ ingested` : 'fully ingested'}</span>
											{:else}
												<span class="ds-badge ds-badge-partial">partial · {ing.n_already_ingested}/{ing.n_in_file} ingested</span>
											{/if}
										{/if}
										{#if canIngest && st.phase === 'idle'}
											<button class="ds-action" disabled={globalWriterActive || localWriterActive()} onclick={() => ingestCorpus(d)}>ingest into corpus.duckdb →</button>
										{:else if canIngestGz && st.phase === 'idle'}
											<button class="ds-action ds-action-heavy" disabled={globalWriterActive || localWriterActive()} onclick={() => ingestCorpus(d)} title="streams gunzip, json.loads in-memory, writes statements to corpus.duckdb. May take minutes and use 5-10x the compressed size in RAM.">ingest from .gz (decompresses in-memory) →</button>
										{:else if st.phase === 'running'}
											<span class="ds-action ds-action-running" role="status" aria-live="polite">
												{#if st.n_total != null && st.n_total > 0 && st.n_done != null}
													ingested {st.n_done.toLocaleString()} / {st.n_total.toLocaleString()} stmts · {Math.round((tickNow - (st.t_started ?? tickNow)) / 1000)}s
												{:else if st.n_done != null && st.n_done > 0}
													ingested {st.n_done.toLocaleString()} stmts · {Math.round((tickNow - (st.t_started ?? tickNow)) / 1000)}s
												{:else}
													parsing {fmtBytes(d.size_bytes)}{d.ext === 'json.gz' ? ' (.gz · 5–10× in RAM)' : ''}… {st.t_started ? Math.round((tickNow - st.t_started) / 1000) + 's' : ''}
												{/if}
											</span>
											<button class="ds-action ds-action-cancel" onclick={() => cancelIngest(d.path)}>cancel →</button>
										{:else if st.phase === 'done'}
											<span class="ds-action ds-action-done">✓ {st.message}</span>
										{:else if st.phase === 'blocked'}
											<span class="ds-action ds-action-blocked" title={actionBlockedText(st)}>blocked: {actionBlockedText(st).slice(0, 120)}</span>
											<button class="ds-action ds-action-cancel" onclick={() => setAction(d.path, { phase: 'idle' })}>reset</button>
										{:else if st.phase === 'error'}
											<span class="ds-action ds-action-error" title={actionErrorText(st)}>✗ {actionErrorText(st).slice(0, 100)}</span>
										{/if}
									</div>
									{#if d.shape.sample_lines.length > 0}
										<ul class="ds-samples">
											{#each d.shape.sample_lines as s}
												<li><span class="ds-sample">{s}</span></li>
											{/each}
										</ul>
									{/if}
									{#if d.shape.notes.length > 0}
										<p class="ds-notes">{d.shape.notes.join(' · ')}</p>
									{/if}
									{#if d.shape.kind_detail === 'indra_json'}
										<div class="ds-preflight">
											<div class="ds-arch-toggle" role="group" aria-label="scoring architecture">
												<button
													type="button"
													class="ds-arch-btn"
													class:ds-arch-active={selectedArchitecture === 'decomposed'}
													disabled={pre.phase === 'scoring' || pre.phase === 'estimating' || pairBusy(pair)}
													onclick={() => setPreflightArchitecture(d.path, 'decomposed')}
													title="four-probe decomposed scorer with native probe trace rows"
												>[D] decomposed</button>
												<button
													type="button"
													class="ds-arch-btn"
													class:ds-arch-active={selectedArchitecture === 'monolithic'}
													disabled={pre.phase === 'scoring' || pre.phase === 'estimating' || pairBusy(pair)}
													onclick={() => setPreflightArchitecture(d.path, 'monolithic')}
													title="single aggregate scorer; no decomposed probe pillbars"
												>[M] monolithic</button>
											</div>
											{#if pre.phase === 'idle'}
												<button class="ds-action" onclick={() => estimateCost(d)}>preview {archLabel(selectedArchitecture)} scoring cost →</button>
											{:else if pre.phase === 'estimating'}
												<span class="ds-action ds-action-running" role="status" aria-live="polite">estimating {archLabel(selectedArchitecture)}…</span>
											{:else if pre.phase === 'estimated'}
												<div class="ds-cost-panel">
													<p class="ds-cost-intro">
														{#if selectedArchitecture === 'decomposed'}
															Projected [D] cost <em>before</em> substrate short-circuits (which typically reduce actual spend 30–60%). Each row is a separate spend commitment to that model's API:
														{:else}
															Projected [M] cost for one aggregate scorer call per evidence. Monolithic runs write aggregate trace rows and do not emit four-probe native trace rows:
														{/if}
													</p>
													<table class="ds-cost-table">
														<thead>
															<tr><th>model</th><th class="num">est cost</th><th class="num">est range</th><th class="num">calls</th><th></th></tr>
														</thead>
														<tbody>
															{#each pre.estimates as e}
																{@const lowEnd = e.cost_usd * 0.4}
																{@const highEnd = e.cost_usd}
																<tr>
																	<td><code>{e.model_id}</code></td>
																	<td class="num">{fmtCost(e.cost_usd)}</td>
																	<td class="num muted">{fmtCost(lowEnd)}–{fmtCost(highEnd)}</td>
																	<td class="num">{e.n_llm_calls_est.toLocaleString()}</td>
																	<td>
																		<button
																			class="ds-action-spend"
																			disabled={pairBusy(pair) || globalWriterActive || localWriterActive()}
																			onclick={() => scoreCorpus(d, e.model_id, e.cost_usd, selectedArchitecture)}
																			title={`will spend up to ${fmtCost(e.cost_usd * 1.25)} via ${e.model_id}'s API and write ${archLabel(selectedArchitecture)} scorer_step + score_run rows · cancel via the scoring panel; page close relies on browser disconnect reaching the endpoint`}
																		>
																			spend up to {fmtCost(e.cost_usd * 1.25)} on {archMark(selectedArchitecture)} {e.model_id} →
																		</button>
																	</td>
																</tr>
															{/each}
														</tbody>
													</table>
													<p class="ds-cost-warn">
														Clicking a spend button charges your provider's API key and writes rows into <code>corpus.duckdb</code>. The worker accepts a hard cap (1.25× est); spend beyond that aborts. The cancel button is the direct stop control; closing the page only asks the browser to drop the stream so the endpoint can observe disconnect and terminate the worker.
													</p>
													<button class="ds-action ds-action-cancel" onclick={() => setPre(d.path, { phase: 'idle' })}>nevermind</button>
												</div>
											{:else if pre.phase === 'scoring'}
												<div class="ds-cost-panel ds-scoring" role="status" aria-live="polite">
													<p class="ds-scoring-line">
														<strong>{archMark(pre.architecture)} spending on {pre.model}</strong>
														{#if pre.n_evidences_total != null}· {pre.n_evidences_done} / {pre.n_evidences_total} evidences{:else}· {pre.n_evidences_done} evidences scored{/if}
														· spent {pre.cost_so_far_usd != null ? fmtCost(pre.cost_so_far_usd) : 'pending'} / {fmtCost(pre.cost_cap_usd)}
														· elapsed {Math.round((Date.now() - pre.t_started) / 1000)}s
														{#if pre.latest_stmt}<span class="muted">· latest stmt {pre.latest_stmt.slice(0, 8)}</span>{/if}
													</p>
													<p class="ds-cost-warn">
														Stream connected · actual token spend is checked after each evidence and aborts before the next evidence if it exceeds cap. Closing the page aborts the stream and depends on the endpoint observing disconnect before it kills the worker; an already accepted provider request may still bill.
													</p>
													<button class="ds-action-cancel-prominent" onclick={() => cancelScore(d.path)}>cancel this run →</button>
												</div>
											{:else if pre.phase === 'scored'}
												<span class="ds-action ds-action-done">✓ {archMark(pre.architecture)} scored as run {pre.run_id.slice(0, 8)} · {pre.n_evidences} evidences · cap {fmtCost(pre.cost_cap_usd)} · {pre.duration_s.toFixed(1)}s</span>
											{:else if pre.phase === 'blocked'}
												<span class="ds-action ds-action-blocked" title={pre.message}>{archMark(pre.architecture)} scoring blocked: {pre.message.slice(0, 140)}</span>
												<button class="ds-action ds-action-cancel" onclick={() => setPre(d.path, { phase: 'idle' })}>reset</button>
											{:else if pre.phase === 'error'}
												<span class="ds-action ds-action-error" title={pre.message}>✗ {pre.message.slice(0, 100)}</span>
												<button class="ds-action ds-action-cancel" onclick={() => setPre(d.path, { phase: 'idle' })}>reset</button>
											{/if}
										</div>
											<div class="ds-preflight ds-paired-preflight">
												<div class="ds-pair-head">
													<span class="ds-pair-title">paired architecture run</span>
													<span class="muted">serial writer · one pair id · workbench landing</span>
												</div>
												{#if isClonePairTarget(d.path)}
													<p class="ds-clone-note">
														clone preflight from pair page
														{#if clonePairModel}<span> · previous model <code>{clonePairModel}</code></span>{/if}
														{#if clonePairScorer}<span> · scorer <code>{clonePairScorer}</code></span>{/if}
														<span> · preview cost again before spend</span>
													</p>
												{/if}
												{#if pair.phase === 'idle'}
													<button class="ds-action" disabled={pre.phase === 'scoring' || pre.phase === 'estimating' || globalWriterActive} onclick={() => estimatePairedCost(d)}>{isClonePairTarget(d.path) ? 'preview cloned paired cost' : 'preview paired [M]+[D] cost'} →</button>
											{:else if pair.phase === 'estimating'}
												<span class="ds-action ds-action-running" role="status" aria-live="polite">estimating [M] and [D] cost…</span>
											{:else if pair.phase === 'estimated'}
												<div class="ds-cost-panel ds-pair-panel">
													<p class="ds-cost-intro">Paired preflight shows both architecture costs before spend. The cap is an estimate gate checked before each worker starts; the server then runs [M] and [D] serially under one <code>paired_run_group_id</code>.</p>
													<table class="ds-cost-table ds-pair-table">
														<thead>
															<tr><th>model</th><th class="num">[M] cost</th><th class="num">[D] cost</th><th class="num">total</th><th class="num">cap</th><th></th></tr>
														</thead>
														<tbody>
															{#each pair.estimates as e}
																<tr>
																	<td><code>{e.model_id}</code></td>
																	<td class="num">{fmtCost(e.monolithic.cost_usd)}</td>
																	<td class="num">{fmtCost(e.decomposed.cost_usd)}</td>
																	<td class="num">{fmtCost(e.total_cost_usd)}</td>
																	<td class="num">{fmtCost(e.total_cap_usd)}</td>
																	<td>
																		<button
																			class="ds-action-spend"
																			disabled={globalWriterActive || localWriterActive()}
																			onclick={() => scorePairedCorpus(d, e)}
																			title={`will run monolithic then decomposed serially if their estimates fit inside ${fmtCost(e.total_cap_usd)} total cap, then land on the paired workbench`}
																		>
																			start pair with {fmtCost(e.total_cap_usd)} estimate cap →
																		</button>
																	</td>
																</tr>
															{/each}
														</tbody>
													</table>
													<p class="ds-cost-warn">
														Cancel asks the server to kill the current worker; queued architecture work never starts. Completed partial runs remain append-only and can be inspected from the pair page once present.
													</p>
													<button class="ds-action ds-action-cancel" onclick={() => setPair(d.path, { phase: 'idle' })}>nevermind</button>
												</div>
											{:else if pair.phase === 'scoring'}
												<div class="ds-cost-panel ds-pair-panel ds-pair-scoring" role="status" aria-live="polite">
													<p class="ds-scoring-line">
														<strong>pair <code>{pair.pair_id}</code></strong>
														· model {pair.model}
														· cap {fmtCost(pair.total_cap_usd)}
														· elapsed {Math.round((tickNow - pair.t_started) / 1000)}s
													</p>
													<div class="ds-pair-lanes">
														{#each ARCH_LANES as arch}
															{@const a = pair.architectures[arch]}
															<div class={`ds-pair-lane ds-pair-lane-${a.state}`}>
																<span class="ds-pair-arch">{archMark(arch)} {arch}</span>
																<span class="ds-pair-state">{a.state}</span>
																<span class="ds-pair-progress">{pairProgressText(a)}</span>
																<span class="ds-pair-cap">cap {fmtCost(a.cost_cap_usd)}</span>
																{#if a.run_id}<a href={`/runs/${a.run_id}`}>run {a.run_id.slice(0, 8)}</a>{/if}
															</div>
														{/each}
													</div>
													<p class="ds-cost-warn">
														DuckDB writer lock is serial: one architecture runs while the other is queued. Actual token spend is checked after each evidence against each architecture cap. Page close sends a best-effort paired cancel request and aborts the stream.
													</p>
													<button class="ds-action-cancel-prominent" onclick={() => cancelPairedScore(d.path)}>cancel paired run →</button>
												</div>
											{:else if pair.phase === 'done'}
												<span class="ds-action ds-action-done">✓ pair {pair.pair_id} done · <a href={pair.href}>open workbench</a></span>
											{:else if pair.phase === 'canceled'}
												<span class="ds-action ds-action-error" title={pair.message}>cancelled pair {pair.pair_id ?? ''} · {pair.message}</span>
												<button class="ds-action ds-action-cancel" onclick={() => setPair(d.path, { phase: 'idle' })}>reset</button>
												{:else if pair.phase === 'blocked'}
													<span class="ds-action ds-action-blocked" title={pair.message}>paired run blocked: {pair.message.slice(0, 180)}</span>
													{#if pair.href}<a class="ds-action" href={pair.href}>open partial pair</a>{/if}
													<button class="ds-action ds-action-cancel" onclick={() => setPair(d.path, { phase: 'idle' })}>reset</button>
											{:else if pair.phase === 'error'}
												<span class="ds-action ds-action-error" title={pair.message}>✗ paired run failed: {pair.message.slice(0, 100)}</span>
												{#if pair.href}<a class="ds-action" href={pair.href}>open partial pair</a>{/if}
												<button class="ds-action ds-action-cancel" onclick={() => setPair(d.path, { phase: 'idle' })}>reset</button>
											{/if}
										</div>
									{/if}
								</li>
							{/each}
						</ul>
					{/if}
				</div>

				<div class="ds-group">
					<h3 class="ds-group-h">data/benchmark/ <span class="muted">({benchmarks.length})</span></h3>
					{#if benchmarks.length === 0}
						<p class="ds-empty">no benchmark files found.</p>
					{:else}
						<ul class="ds-list">
							{#each benchmarks as d}
								{@const st = actionState(d.path)}
								{@const canRegister = d.shape.kind_detail === 'jsonl_records' && (d.shape.n_records ?? 0) > 0}
								{@const canIngestGzB = d.shape.kind_detail === 'json_gz'}
								<li class="ds-row">
									<div class="ds-row-head">
										<code class="ds-name">{d.filename}</code>
										<span class="ds-meta">
											<span>{fmtBytes(d.size_bytes)}</span>
											<span class="muted">·</span>
											<span>{d.shape.n_records ?? '—'} {d.shape.kind_detail === 'jsonl_records' ? 'records' : d.shape.kind_detail === 'indra_json' ? 'statements' : d.shape.kind_detail === 'json_gz' ? 'compressed' : 'unparsed'}</span>
											{#if d.shape.source_apis.length > 0}
												<span class="muted">·</span>
												<span>sources: {d.shape.source_apis.join(', ')}</span>
											{/if}
										</span>
										{#if d.duplicate_of && d.duplicate_of.length > 0}
											<span class="ds-badge ds-badge-dup" title={`byte-identical to ${d.duplicate_of.join(', ')} (same size + same first 16KB)`}>duplicate of {d.duplicate_of.join(', ')}</span>
										{/if}
										{#if d.ingest && d.ingest.n_in_file > 0}
											{@const ing = d.ingest}
											{#if ing.n_already_ingested === 0}
												<span class="ds-badge ds-badge-fresh">not yet ingested</span>
											{:else if ing.n_already_ingested >= ing.n_in_file}
												<span class="ds-badge ds-badge-done">{ing.sampled ? `${ing.n_already_ingested}+ ingested` : 'fully ingested'}</span>
											{:else}
												<span class="ds-badge ds-badge-partial">partial · {ing.n_already_ingested}/{ing.n_in_file} ingested</span>
											{/if}
										{/if}
										{#if canRegister && st.phase === 'idle'}
											<button class="ds-action" disabled={globalWriterActive || localWriterActive()} onclick={() => registerAsTruthSet(d)} title="reads `tag` field on each record; registers as evidence-level truth_set; truth-present validity treats tag=correct as positive and every other tag as not-correct when scored evidence overlaps">register `tag` as truth_set →</button>
										{:else if canIngestGzB && st.phase === 'idle'}
											<button class="ds-action ds-action-heavy" disabled={globalWriterActive || localWriterActive()} onclick={() => ingestCorpus(d)} title="streams gunzip, json.loads in-memory, writes statements to corpus.duckdb. May take minutes and use 5-10x the compressed size in RAM.">ingest from .gz (decompresses in-memory) →</button>
										{:else if st.phase === 'running' && canIngestGzB}
											<span class="ds-action ds-action-running" role="status" aria-live="polite">
												{#if st.n_total != null && st.n_total > 0 && st.n_done != null}
													ingested {st.n_done.toLocaleString()} / {st.n_total.toLocaleString()} stmts · {Math.round((tickNow - (st.t_started ?? tickNow)) / 1000)}s
												{:else if st.n_done != null && st.n_done > 0}
													ingested {st.n_done.toLocaleString()} stmts · {Math.round((tickNow - (st.t_started ?? tickNow)) / 1000)}s
												{:else}
													parsing {fmtBytes(d.size_bytes)}{d.ext === 'json.gz' ? ' (.gz · 5–10× in RAM)' : ''}… {st.t_started ? Math.round((tickNow - st.t_started) / 1000) + 's' : ''}
												{/if}
											</span>
											<button class="ds-action ds-action-cancel" onclick={() => cancelIngest(d.path)}>cancel →</button>
										{:else if st.phase === 'running'}
											<span class="ds-action ds-action-running" role="status" aria-live="polite">registering truth_set… {st.t_started ? Math.round((tickNow - st.t_started) / 1000) + 's' : ''}</span>
											<button class="ds-action ds-action-cancel" onclick={() => cancelTruthSet(d.path)}>cancel →</button>
											<span class="ds-action-note">cancel signals the worker; registration commits atomically, but a committed replacement remains active</span>
										{:else if st.phase === 'done'}
											<span class="ds-action ds-action-done">✓ {st.message}</span>
										{:else if st.phase === 'blocked'}
											<span class="ds-action ds-action-blocked" title={actionBlockedText(st)}>blocked: {actionBlockedText(st).slice(0, 120)}</span>
											<button class="ds-action ds-action-cancel" onclick={() => setAction(d.path, { phase: 'idle' })}>reset</button>
										{:else if st.phase === 'error'}
											<span class="ds-action ds-action-error" title={actionErrorText(st)}>✗ {actionErrorText(st).slice(0, 100)}</span>
										{/if}
									</div>
									{#if d.shape.sample_lines.length > 0}
										<ul class="ds-samples">
											{#each d.shape.sample_lines as s}
												<li><span class="ds-sample">{s}</span></li>
											{/each}
										</ul>
									{/if}
									{#if d.shape.notes.length > 0}
										<p class="ds-notes">{d.shape.notes.join(' · ')}</p>
									{/if}
								</li>
							{/each}
						</ul>
					{/if}
				</div>
			</section>
		{/if}

		<footer class="data-footer">
			<div class="df-line">
				<span class="df-count">{fmt(o.statementCount)} stmts</span>
				<span class="df-sep">·</span>
				<span class="df-count">{fmt(o.evidenceCount)} ev</span>
				<span class="df-sep">·</span>
				<span class="df-count">{fmt(o.agentCount)} agents</span>
				<span class="df-sep">·</span>
				<span class="df-count">{fmt(o.supportsEdgeCount)} supports</span>
				<span class="df-sep">·</span>
				<span class="df-count">{fmt(o.truthLabelCount)} truth labels</span>
			</div>
			{#if o.truthSets.length > 0}
				<div class="df-line df-line-sub">
					<span class="df-label">truth_sets</span>
					{#each o.truthSets as t, i}{#if i > 0}<span class="df-sep">·</span>{/if}<span class="df-item"><code>{t.id}</code> {fmt(t.rowCount)}</span>{/each}
				</div>
			{/if}
			{#if o.sourceDumps.length > 0}
				<div class="df-line df-line-sub">
					<span class="df-label">source_dump_id</span>
					{#each o.sourceDumps.slice(0, 6) as s, i}{#if i > 0}<span class="df-sep">·</span>{/if}<span class="df-item"><code>{s.source_dump_id ?? '<null>'}</code> {fmt(s.n)}</span>{/each}{#if o.sourceDumps.length > 6}<span class="df-sep">·</span><span class="df-item df-more">+{o.sourceDumps.length - 6} more</span>{/if}
				</div>
			{/if}
			{#if o.indraTypes.length > 0}
				<details class="df-details">
					<summary>indra_type breakdown ({o.indraTypes.length})</summary>
					<div class="df-line df-line-sub">
						{#each o.indraTypes as t, i}{#if i > 0}<span class="df-sep">·</span>{/if}<span class="df-item">{t.indra_type} {fmt(t.n)}</span>{/each}
					</div>
				</details>
			{/if}
		</footer>
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
		--accent-wash: rgba(125, 42, 26, 0.04);
		--blocked: #6f5a16;
		--ok-green: #2a6f2a;
		--mono: ui-monospace, 'SF Mono', 'JetBrains Mono', Menlo, monospace;
		--serif: 'Iowan Old Style', 'Source Serif Pro', Georgia, serif;
		--sans: -apple-system, BlinkMacSystemFont, 'Inter', sans-serif;
	}

	:global(html, body) {
		background: var(--paper);
		color: var(--ink);
		font-family: var(--serif);
		font-size: 16px;
		line-height: 1.5;
		margin: 0;
		padding: 0;
	}

	header {
		display: flex;
		justify-content: space-between;
		align-items: baseline;
		gap: 0.8rem 1.2rem;
		padding: 0.6rem 1.5rem;
		border-bottom: 1px solid var(--rule);
		font-family: var(--mono);
		font-size: 0.78rem;
		color: var(--ink-muted);
	}

	.crumb strong {
		color: var(--ink);
		font-weight: 500;
	}

	.crumb .sep {
		color: var(--ink-faint);
	}

	.db-path {
		color: var(--ink-faint);
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}

	.meta {
		display: flex;
		gap: 1.2rem;
		align-items: baseline;
		min-width: 0;
	}

	.nav-link {
		color: var(--accent);
		text-decoration: none;
	}

	.nav-link:hover {
		text-decoration: underline;
	}

	main {
		max-width: 1200px;
		margin: 0 auto;
		padding: 2rem 1.5rem 4rem;
	}

	.empty {
		max-width: 60ch;
		margin: 4rem auto;
	}

	.empty h1 {
		font-family: var(--serif);
		font-weight: 400;
		font-size: 1.6rem;
		color: var(--ink);
		margin: 0 0 0.5rem;
	}

	.empty .lede {
		color: var(--ink-muted);
		margin-bottom: 1.2rem;
	}

	pre {
		background: transparent;
		border-left: 2px solid var(--accent);
		padding: 0.4rem 0 0.4rem 0.8rem;
		font-family: var(--mono);
		font-size: 0.82rem;
		color: var(--ink);
		overflow-x: auto;
	}

	code {
		font-family: var(--mono);
		font-size: 0.88em;
	}

	.hint {
		color: var(--ink-muted);
		font-style: italic;
		font-size: 0.92em;
	}


	.muted { color: var(--ink-faint); }

	.status-failed { color: var(--accent); font-weight: 500; }
	.status-running { color: var(--ink); font-style: italic; }
	.status-canceled { color: var(--accent); font-style: italic; }

	.run-feed-article {
		grid-column: 1 / -1;
	}
	.run-feed {
		list-style: none;
		padding: 0;
		margin: 0;
		font-family: var(--mono);
		font-size: 0.78rem;
	}
	.run-row {
		display: grid;
		grid-template-columns: 1.4ch 9ch 4ch minmax(0, max-content) minmax(0, max-content) minmax(0, 1fr) auto;
		gap: 0.6rem;
		align-items: baseline;
		padding: 0.25rem 0;
		border-bottom: 1px dotted var(--rule);
	}
	.run-version, .run-when {
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}
	.run-row:last-child {
		border-bottom: none;
	}
	.run-row-failed { color: var(--accent); }
	.run-row-running { color: var(--ink); }
	.run-row-canceled { color: var(--accent); }
	.run-glyph {
		font-variant-numeric: tabular-nums;
		text-align: center;
	}
	.run-hash {
		color: var(--ink);
		text-decoration: none;
	}
	.run-hash:hover {
		color: var(--accent);
		text-decoration: underline;
	}
	.run-arch {
		color: var(--ink);
		font-variant-numeric: tabular-nums;
	}
	.run-arch-monolithic {
		color: var(--accent);
	}
	.run-arch-unknown {
		color: var(--ink-faint);
	}
	.run-version {
		color: var(--ink-muted);
	}
	.run-when {
		color: var(--ink-faint);
	}
	.run-narrative {
		color: var(--ink);
		font-variant-numeric: tabular-nums;
		min-width: 0;
	}
	.run-n, .run-cost {
		color: var(--ink);
	}
	.run-summary {
		color: var(--ink);
	}
	.run-exports {
		font-family: var(--mono);
		font-size: 0.72rem;
	}

	.cost-expander {
		grid-column: 1 / -1;
		padding: 0.4rem 0.8rem;
		margin-top: 0.4rem;
		border: 1px solid var(--rule);
		font-family: var(--mono);
		font-size: 0.78rem;
		color: var(--ink-muted);
	}
	.cost-expander summary {
		cursor: pointer;
		font-size: 0.74rem;
		color: var(--ink);
		text-transform: lowercase;
		letter-spacing: 0.02em;
	}
	.cost-expander[open] summary {
		margin-bottom: 0.6rem;
	}
	.cost-expander table {
		width: 100%;
		font-family: var(--mono);
		font-size: 0.78rem;
	}
	.cost-expander td.num {
		font-variant-numeric: tabular-nums;
	}

	.live-indicator {
		display: inline-flex;
		align-items: center;
		gap: 0.3rem;
		font-family: var(--mono);
		font-size: 0.7rem;
		color: var(--ink-faint);
		text-transform: lowercase;
		letter-spacing: 0.04em;
	}

	.live-dot {
		display: inline-block;
		width: 6px;
		height: 6px;
		border-radius: 50%;
		background: var(--ink-faint);
		transition: background 200ms ease;
	}

	.live-dot.live-dot-flash {
		background: var(--accent);
		transform: scale(1.4);
	}

	.dl-link {
		color: var(--accent);
		text-decoration: none;
		margin-right: 0.6rem;
	}

	.dl-link:hover {
		text-decoration: underline;
	}

	.dl-missing {
		color: var(--ink-faint);
		cursor: help;
	}

	.dashboard-subtitle {
		font-family: var(--serif);
		font-size: 1rem;
		color: var(--ink-muted);
		margin: 0.3rem 0 1.6rem;
		line-height: 1.5;
		max-width: 60ch;
	}

	.focus {
		margin-top: 0.5rem;
		margin-bottom: 2.5rem;
	}

	.findings {
		margin: 0 0 2.5rem;
	}

	.writer-lock {
		margin: 0 0 1.4rem;
		padding: 0.7rem 0.8rem;
		border-left: 3px solid var(--accent);
		background: var(--accent-wash);
		display: flex;
		justify-content: space-between;
		gap: 1rem;
		align-items: baseline;
		font-family: var(--mono);
		font-size: 0.76rem;
	}
	.wl-h {
		margin: 0 0 0.25rem;
		font-size: 0.76rem;
		font-weight: 600;
		text-transform: lowercase;
		letter-spacing: 0.02em;
	}
	.writer-lock p {
		margin: 0;
	}
	.wl-note {
		color: var(--ink-muted);
		max-width: 44ch;
	}

	.pair-workflows {
		margin: 0 0 2.5rem;
		border-top: 1px solid var(--ink);
		padding-top: 0.8rem;
	}
	.pw-h {
		font-family: var(--mono);
		font-size: 0.72rem;
		color: var(--ink);
		text-transform: lowercase;
		letter-spacing: 0.02em;
		font-weight: 500;
		margin: 0 0 0.55rem;
	}
	.pw-list {
		display: grid;
		gap: 0.65rem;
	}
	.pw-row {
		border-left: 2px solid var(--rule);
		padding-left: 0.65rem;
		font-family: var(--mono);
		font-size: 0.76rem;
	}
	.pw-row-running,
	.pw-row-queued {
		border-left-color: var(--ink);
	}
	.pw-row-active {
		border-left-color: var(--ok-green);
	}
	.pw-row-stale {
		border-left-color: #8a6a00;
	}
	.pw-row-failed,
	.pw-row-canceled {
		border-left-color: var(--accent);
	}
	.pw-main,
	.pw-actions {
		display: flex;
		flex-wrap: wrap;
		gap: 0.55rem;
		align-items: baseline;
	}
	.pw-main a,
	.pw-actions a {
		color: var(--accent);
		text-decoration: none;
	}
	.pw-main a:hover,
	.pw-actions a:hover {
		text-decoration: underline;
	}
	.pw-status {
		border: 1px solid currentColor;
		padding: 0 0.25rem;
		text-transform: lowercase;
	}
	.pw-state-kind {
		padding: 0 0.25rem;
		text-transform: lowercase;
		border-left: 2px solid currentColor;
		color: var(--ink-muted);
	}
	.pw-state-kind-active {
		color: var(--ok-green);
	}
	.pw-state-kind-stale {
		color: #8a6a00;
	}
	.pw-state-kind-terminal {
		color: var(--ink-faint);
	}
	.pw-state-rail {
		margin: 0.28rem 0 0;
		color: var(--ink-muted);
		line-height: 1.35;
	}
	.pw-arches {
		display: grid;
		grid-template-columns: repeat(2, minmax(0, 1fr));
		gap: 0.45rem;
		margin: 0.45rem 0;
	}
	.pw-arch {
		display: grid;
		grid-template-columns: auto auto minmax(0, 1fr);
		gap: 0.35rem;
		min-width: 0;
		color: var(--ink-muted);
	}
	.pw-arch strong {
		color: var(--ink);
		font-weight: 500;
	}
	.pw-arch em {
		font-style: normal;
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}
	.pw-actions button {
		font-family: var(--mono);
		font-size: 0.72rem;
		color: var(--accent);
		background: transparent;
		border: 1px solid var(--accent);
		cursor: pointer;
	}
	.pw-actions button:hover {
		background: var(--accent-wash);
	}
	.pw-error {
		margin: 0.35rem 0 0;
		color: var(--accent);
		font-family: var(--mono);
		font-size: 0.72rem;
	}

	.lane {
		margin-bottom: 1.2rem;
	}

	.lane-h {
		font-family: var(--mono);
		font-size: 0.74rem;
		color: var(--ink);
		text-transform: lowercase;
		letter-spacing: 0.02em;
		font-weight: 400;
		margin: 0 0 0.2rem;
		border-bottom: 1px dotted var(--rule);
		padding-bottom: 0.2rem;
	}

	.lane-n {
		color: var(--ink-faint);
		font-weight: 400;
	}

	.lane-body {
		display: flex;
		flex-direction: column;
	}

	.focus-deeplink {
		font-family: var(--mono);
		font-size: 0.78rem;
		text-align: right;
		margin: 0.4rem 0 0;
	}
	.focus-deeplink a {
		color: var(--accent);
		text-decoration: none;
	}
	.focus-deeplink a:hover {
		text-decoration: underline;
	}
	.focus-empty {
		padding: 1.6rem;
		border-left: 3px solid var(--rule);
	}

	.datasets {
		margin: 0 0 2.5rem;
	}
	.ds-h {
		font-family: var(--serif);
		font-size: 1.15rem;
		font-weight: 400;
		color: var(--ink);
		margin: 0 0 0.4rem;
	}
	.ds-intro {
		font-family: var(--serif);
		font-style: italic;
		font-size: 0.88rem;
		color: var(--ink-muted);
		margin: 0 0 1rem;
		line-height: 1.5;
	}
	.ds-group {
		margin-bottom: 1.4rem;
	}
	.ds-group-h {
		font-family: var(--mono);
		font-size: 0.74rem;
		color: var(--ink-muted);
		text-transform: lowercase;
		letter-spacing: 0.04em;
		font-weight: 500;
		margin: 0 0 0.4rem;
		border-bottom: 1px dotted var(--rule);
		padding-bottom: 0.2rem;
	}
	.ds-empty {
		font-family: var(--serif);
		font-style: italic;
		font-size: 0.86rem;
		color: var(--ink-faint);
		margin: 0;
	}
	.ds-list {
		list-style: none;
		padding: 0;
		margin: 0;
	}
	.ds-row {
		padding: 0.5rem 0;
		border-bottom: 1px dotted var(--rule);
	}
		.ds-row:last-child {
			border-bottom: none;
		}
		.ds-row-clone-target {
			border-left: 2px solid var(--ink);
			padding-left: 0.6rem;
		}
	.ds-row-head {
		display: flex;
		gap: 0.8rem;
		align-items: baseline;
		flex-wrap: wrap;
		font-family: var(--mono);
		font-size: 0.82rem;
	}
	.ds-name {
		color: var(--ink);
		font-weight: 500;
	}
	.ds-meta {
		font-size: 0.74rem;
		color: var(--ink-muted);
		display: inline-flex;
		gap: 0.4rem;
		align-items: baseline;
		flex-wrap: wrap;
	}
	.ds-samples {
		list-style: none;
		padding: 0;
		margin: 0.3rem 0 0 1rem;
		border-left: 2px solid var(--rule);
	}
	.ds-samples li {
		padding: 0.15rem 0.6rem;
	}
	.ds-sample {
		font-family: var(--serif);
		font-style: italic;
		font-size: 0.88rem;
		color: var(--ink);
		line-height: 1.4;
	}
	.ds-badge {
		font-family: var(--mono);
		font-size: 0.7rem;
		padding: 0.05rem 0.4rem;
		border: 1px solid currentColor;
		text-transform: lowercase;
		letter-spacing: 0.04em;
	}
	.ds-badge-fresh { color: var(--ink-muted); }
	.ds-badge-partial { color: var(--accent); }
	.ds-badge-done { color: var(--ok-green); }
	.ds-badge-unknown { color: var(--ink-faint); }
	.ds-badge-dup {
		color: var(--ink-muted);
		font-style: italic;
		text-transform: none;
		letter-spacing: 0;
		font-family: var(--serif);
	}

	.ds-action {
		font-family: var(--mono);
		font-size: 0.74rem;
		padding: 0.15rem 0.5rem;
		border: 1px solid var(--accent);
		background: transparent;
		color: var(--accent);
		cursor: pointer;
		text-transform: lowercase;
		letter-spacing: 0.02em;
	}
	.ds-action:hover {
		background: var(--accent-wash);
	}
	.ds-action:disabled {
		color: var(--ink-faint);
		border-color: var(--rule);
		cursor: default;
		background: transparent;
	}
	.ds-action-running {
		border: 1px dashed var(--ink-muted);
		color: var(--ink-muted);
		cursor: progress;
	}
	.ds-action-done {
		border: 1px solid var(--ok-green);
		color: var(--ok-green);
		cursor: default;
	}
		.ds-action-error {
			border: 1px solid var(--accent);
			color: var(--accent);
			cursor: help;
		}
		.ds-action-blocked {
			border: 1px dashed var(--blocked);
			color: var(--blocked);
			cursor: help;
		}
	.ds-action-note {
		font-family: var(--mono);
		font-size: 0.72rem;
		color: var(--ink-muted);
		max-width: 56ch;
	}
	/* Heavy / slow ops — distinct from cheap ingests. Same accent color
	   so it reads as the same family of action, but a double-rule border
	   warns the reader: this one takes minutes and burns RAM. */
	.ds-action-heavy {
		border: 1px double var(--accent);
		padding: 0.15rem 0.6rem;
	}
	.ds-action-cancel {
		font-family: var(--mono);
		font-size: 0.74rem;
		padding: 0.15rem 0.5rem;
		border: 1px solid var(--ink-faint);
		background: transparent;
		color: var(--ink-muted);
		cursor: pointer;
		text-transform: lowercase;
		letter-spacing: 0.02em;
	}
	.ds-action-cancel:hover {
		color: var(--ink);
		border-color: var(--ink);
	}

	/* Distinct visual species for spend-commit buttons: filled accent,
	   not the bordered "navigation" species. P11 reversibility legibility —
	   the irreversible action must NOT look like the safe ones around it. */
	.ds-action-spend {
		font-family: var(--mono);
		font-size: 0.78rem;
		padding: 0.3rem 0.7rem;
		border: 1px solid var(--accent);
		background: var(--accent);
		color: var(--paper);
		cursor: pointer;
		font-weight: 500;
		text-transform: lowercase;
		letter-spacing: 0.02em;
	}
	.ds-action-spend:hover {
		filter: brightness(1.1);
	}
	.ds-action-spend:disabled {
		background: var(--rule);
		border-color: var(--rule);
		color: var(--ink-muted);
		cursor: default;
		filter: none;
	}

	/* Cancel-in-flight button: visible, distinct species, but quieter than
	   spend. Outlined accent — signals "destructive of in-flight work but
	   not of API budget". */
	.ds-action-cancel-prominent {
		font-family: var(--mono);
		font-size: 0.86rem;
		padding: 0.35rem 0.8rem;
		border: 1.5px solid var(--accent);
		background: transparent;
		color: var(--accent);
		cursor: pointer;
		font-weight: 500;
		text-transform: lowercase;
		letter-spacing: 0.02em;
		margin-top: 0.4rem;
	}
	.ds-action-cancel-prominent:hover {
		background: var(--accent-wash);
	}

	.ds-preflight {
		margin-top: 0.6rem;
	}
	.ds-arch-toggle {
		display: flex;
		flex-wrap: wrap;
		gap: 0.4rem;
		margin: 0 0 0.5rem;
	}
	.ds-arch-btn {
		font-family: var(--mono);
		font-size: 0.72rem;
		color: var(--ink-muted);
		background: var(--paper);
		border: 1px solid var(--rule);
		padding: 0.22rem 0.45rem;
		cursor: pointer;
	}
	.ds-arch-btn:hover:not(:disabled) {
		border-color: var(--accent);
		color: var(--accent);
	}
	.ds-arch-btn:disabled {
		cursor: default;
		opacity: 0.7;
	}
	.ds-arch-active {
		color: var(--accent);
		border-color: var(--accent);
		background: var(--accent-wash);
	}
	.ds-cost-panel {
		margin-top: 0.4rem;
		padding: 0.8rem 1rem;
		border-left: 3px solid var(--accent);
		background: var(--accent-wash);
	}
	.ds-cost-intro {
		font-family: var(--serif);
		font-size: 0.86rem;
		color: var(--ink);
		margin: 0 0 0.6rem;
		line-height: 1.5;
	}
	.ds-cost-table {
		width: 100%;
		max-width: 520px;
		border-collapse: collapse;
		font-family: var(--mono);
		font-size: 0.78rem;
		margin: 0.3rem 0;
	}
	.ds-cost-table th {
		text-align: left;
		font-weight: 500;
		color: var(--ink-muted);
		font-size: 0.7rem;
		padding: 0.2rem 0.6rem 0.2rem 0;
		border-bottom: 1px dotted var(--rule);
	}
	.ds-cost-table td {
		padding: 0.3rem 0.6rem 0.3rem 0;
		vertical-align: baseline;
	}
	.ds-cost-table td.num {
		text-align: right;
		font-variant-numeric: tabular-nums;
	}
	.ds-cost-warn {
		font-family: var(--serif);
		font-style: italic;
		font-size: 0.78rem;
		color: var(--accent);
		margin: 0.4rem 0 0.6rem;
		line-height: 1.45;
	}

	.ds-scoring {
		border-left-color: var(--ok-green);
	}
	.ds-scoring-line {
		font-family: var(--serif);
		font-size: 0.95rem;
		color: var(--ink);
		margin: 0 0 0.3rem;
	}
	.ds-paired-preflight {
		border-top: 1px dotted var(--rule);
		padding-top: 0.6rem;
	}
	.ds-pair-head {
		display: flex;
		flex-wrap: wrap;
		gap: 0.5rem;
		align-items: baseline;
		margin-bottom: 0.35rem;
		font-family: var(--mono);
		font-size: 0.72rem;
	}
		.ds-pair-title {
			color: var(--ink);
			text-transform: lowercase;
			letter-spacing: 0.02em;
		}
		.ds-clone-note {
			margin: 0.2rem 0 0.5rem;
			color: var(--ink);
			font-family: var(--mono);
			font-size: 0.74rem;
		}
		.ds-clone-note span {
			color: var(--ink-muted);
		}
	.ds-pair-panel {
		border-left-color: var(--ink);
		background: transparent;
	}
	.ds-pair-table {
		max-width: 760px;
	}
	.ds-pair-scoring {
		background: var(--accent-wash);
	}
	.ds-pair-lanes {
		display: grid;
		grid-template-columns: repeat(2, minmax(0, 1fr));
		gap: 0.55rem;
		margin-top: 0.6rem;
	}
	.ds-pair-lane {
		display: grid;
		grid-template-columns: auto auto minmax(0, 1fr);
		gap: 0.35rem 0.55rem;
		align-items: baseline;
		border-left: 2px solid var(--rule);
		padding-left: 0.55rem;
		font-family: var(--mono);
		font-size: 0.74rem;
		min-width: 0;
	}
	.ds-pair-lane-running {
		border-left-color: var(--ink);
	}
	.ds-pair-lane-done {
		border-left-color: var(--ok-green);
	}
	.ds-pair-lane-canceled,
	.ds-pair-lane-crashed,
	.ds-pair-lane-blocked {
		border-left-color: var(--accent);
	}
	.ds-pair-arch,
	.ds-pair-state {
		color: var(--ink);
	}
	.ds-pair-state {
		text-transform: lowercase;
		border: 1px solid currentColor;
		padding: 0 0.25rem;
	}
	.ds-pair-progress {
		min-width: 0;
		color: var(--ink-muted);
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}
	.ds-pair-cap {
		grid-column: 1 / -1;
		color: var(--ink-faint);
	}

	.ds-notes {
		font-family: var(--mono);
		font-size: 0.7rem;
		color: var(--ink-faint);
		margin: 0.2rem 0 0;
		font-style: italic;
	}

	.data-footer {
		margin-top: 4rem;
		padding-top: 1rem;
		border-top: 1px solid var(--rule);
		font-family: var(--mono);
		font-size: 0.72rem;
		color: var(--ink-faint);
	}
	.df-line {
		display: flex;
		flex-wrap: wrap;
		gap: 0.4rem;
		align-items: baseline;
		line-height: 1.7;
	}
	.df-line-sub {
		color: var(--ink-muted);
		margin-top: 0.2rem;
	}
	.df-count {
		color: var(--ink);
	}
	.df-sep {
		color: var(--ink-faint);
	}
	.df-label {
		text-transform: lowercase;
		letter-spacing: 0.04em;
		color: var(--ink-faint);
		margin-right: 0.4rem;
	}
	.df-item code {
		color: inherit;
	}
	.df-more {
		font-style: italic;
	}
	.df-details {
		margin-top: 0.4rem;
	}
	.df-details summary {
		cursor: pointer;
		color: var(--ink-muted);
		font-family: var(--mono);
	}

	.grid {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(min(420px, 100%), 1fr));
		gap: 2rem 3rem;
		margin-top: 2.5rem;
	}

	article h2 {
		font-family: var(--mono);
		font-size: 0.72rem;
		color: var(--ink-muted);
		text-transform: lowercase;
		letter-spacing: 0.02em;
		margin: 0 0 0.5rem;
		font-weight: 500;
		border-bottom: 1px solid var(--rule);
		padding-bottom: 0.2rem;
	}

	table {
		width: 100%;
		border-collapse: collapse;
		font-family: var(--mono);
		font-size: 0.82rem;
		font-variant-numeric: tabular-nums;
	}

	th, td {
		padding: 0.25rem 0.6rem 0.25rem 0;
		text-align: left;
		vertical-align: baseline;
	}

	th {
		font-weight: 500;
		color: var(--ink-muted);
		font-size: 0.72rem;
		text-transform: lowercase;
	}

	tbody tr {
		border-top: 1px dotted var(--rule);
	}

	td {
		color: var(--ink);
	}

	.num {
		text-align: right;
		font-variant-numeric: tabular-nums;
	}

	@media (max-width: 700px) {
		header {
			flex-wrap: wrap;
			align-items: flex-start;
		}
		.meta {
			flex: 1 1 100%;
			flex-wrap: wrap;
			gap: 0.5rem 0.8rem;
		}
		main {
			padding-inline: 1.5rem;
		}
		.run-row {
			grid-template-columns: 1.4ch 8ch 4ch minmax(0, 1fr);
			gap: 0.25rem 0.5rem;
			padding: 0.5rem 0;
		}
		.run-when,
		.run-narrative,
		.run-exports {
			grid-column: 2 / -1;
		}
		.run-when,
		.run-narrative {
			white-space: normal;
			overflow-wrap: anywhere;
		}
		.run-exports {
			display: flex;
			flex-wrap: wrap;
			gap: 0.4rem;
		}
		.writer-lock {
			display: block;
		}
		.wl-note {
			margin-top: 0.35rem;
			max-width: none;
		}
		.ds-pair-lanes {
			grid-template-columns: 1fr;
		}
		.ds-pair-lane {
			grid-template-columns: auto auto;
		}
		.pw-arches {
			grid-template-columns: 1fr;
		}
		.pw-arch {
			grid-template-columns: auto auto;
		}
		.pw-arch em {
			grid-column: 1 / -1;
			white-space: normal;
		}
		.ds-pair-progress,
		.ds-pair-cap {
			grid-column: 1 / -1;
			white-space: normal;
		}
	}
</style>
