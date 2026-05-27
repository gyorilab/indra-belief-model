<script lang="ts">
	import type { PageData } from './$types';
	import { fmtBelief, fmtDelta, shortHash } from '$lib/format';
	import { TRACE_STATE_LABELS, TRACE_STATE_NOTES, type TraceFidelityState } from '$lib/traceState';

	let { data }: { data: PageData } = $props();
	const c = $derived(data.cohort);
	let repairState = $state<
		| { phase: 'idle' }
		| { phase: 'estimating' }
		| {
				phase: 'estimated';
				inspected: number;
				unique_selected: number;
				would_create: number;
				skipped_existing: number;
				skipped_duplicate_selection: number;
				estimate_token: string;
				estimate_expires_at: string | null;
				estimate_key: string;
		  }
		| { phase: 'running' }
		| {
				phase: 'done';
				created: number;
				skipped_existing: number;
				skipped_duplicate_selection: number;
				inspected: number;
		  }
		| { phase: 'stale'; code: string; message: string }
		| { phase: 'blocked'; code: string; message: string }
		| { phase: 'error'; code: string; message: string }
	>({ phase: 'idle' });

	function archMark(architecture: string | null | undefined): string {
		if (architecture === 'monolithic') return '[M]';
		if (architecture === 'decomposed') return '[D]';
		return '[?]';
	}

	const activeFilters = $derived.by(() => {
		const out: Array<[string, string]> = [];
		for (const [k, v] of Object.entries(c.filters)) {
			if (v === null || v === undefined || v === false || v === '') continue;
			out.push([k, String(v)]);
		}
		return out;
		});
		const partialRun = $derived(c.status !== 'succeeded');
		const traceStateCohort = $derived(Boolean(c.filters.trace_state));
		const probeCoverageCohort = $derived(c.filters.probe_coverage === 'present');
		const tracePlaneCohort = $derived(traceStateCohort || probeCoverageCohort);
		const traceStateLabel = $derived(
		c.filters.trace_state
			? TRACE_STATE_LABELS[c.filters.trace_state as TraceFidelityState]
			: null
	);
	const traceStateMeaning = $derived.by(() => {
		const state = c.filters.trace_state as TraceFidelityState | null | undefined;
		if (!state) return null;
		if (state === 'aggregate_only' && c.architecture === 'decomposed') {
			return 'These rows have aggregate verdicts but no decomposed native/probe rows in the trace plane. That makes probe coverage unavailable here, not zero; this route does not prove whether the cause is legacy import, worker failure, migration skew, or another persistence gap.';
		}
		if (state === 'aggregate_only' && c.architecture === 'monolithic') {
			return 'These rows have aggregate verdicts, but monolithic native diagnostic fields are absent or collapsed. This is not a decomposed probe-coverage cohort.';
		}
		return TRACE_STATE_NOTES[state];
	});
		const cohortUnit = $derived(
			c.grain === 'statement'
				? 'statement'
				: tracePlaneCohort
					? 'trace evidence'
					: 'aggregate evidence'
	);
	const unpinnedCohortHref = $derived.by(() => {
		const params = new URLSearchParams();
		for (const [key, value] of Object.entries(c.filters)) {
			if (key === 'trace_snapshot') continue;
			if (value === null || value === undefined || value === false || value === '') continue;
			params.set(key, String(value));
		}
		const query = params.toString();
		return query ? `/runs/${c.run_id}/cohort?${query}` : `/runs/${c.run_id}/cohort`;
	});
	const currentCohortHref = $derived.by(() => {
		const params = new URLSearchParams();
		for (const [key, value] of Object.entries(c.filters)) {
			if (value === null || value === undefined || value === false || value === '') continue;
			params.set(key, String(value));
		}
		const query = params.toString();
		return query ? `/runs/${c.run_id}/cohort?${query}` : `/runs/${c.run_id}/cohort`;
	});
	const snapshotActionLabel = $derived(c.status === 'running' ? 'latest cohort' : 'remove pin');
	const statementEvidenceFiltersActive = $derived(
		c.grain === 'statement' && Boolean(c.filters.source || c.filters.verdict || c.filters.confidence || c.filters.truth_set || c.filters.trace_fidelity)
	);
		const multiEvidenceNote = $derived.by(() => {
			if (!c.filters.multi_evidence) return null;
			if (tracePlaneCohort) {
				return 'multi_evidence selects statements with more than one trace evidence row across the trace evidence plane.';
			}
		if (c.grain === 'statement') {
			return 'multi_evidence selects statements with more than one scored aggregate evidence row.';
		}
		return 'multi_evidence selects aggregate evidence rows from statements with more than one aggregate scorer row.';
	});
	const repairEstimateReady = $derived(
		repairState.phase === 'estimated' &&
		repairState.would_create > 0 &&
		repairState.estimate_key === repairEstimateKey()
	);
		const nonTracePartialRepairBlocked = $derived(partialRun && !traceStateCohort && !probeCoverageCohort);
		const repairEstimateDisabled = $derived(
			repairState.phase === 'estimating' ||
			repairState.phase === 'running' ||
			c.totalRows === 0 ||
			nonTracePartialRepairBlocked
		);
	const repairEstimateStale = $derived(
		repairState.phase === 'estimated' && repairState.estimate_key !== repairEstimateKey()
	);

	function repairFailureState(body: { code?: unknown; message?: unknown }):
		| { phase: 'stale'; code: string; message: string }
		| { phase: 'blocked'; code: string; message: string }
		| { phase: 'error'; code: string; message: string } {
		const code = typeof body?.code === 'string' ? body.code : 'repair_cohort_failed';
		const message = typeof body?.message === 'string' ? body.message : 'repair backlog failed';
			if (code === 'run_status_changed') {
				return { phase: 'stale', code, message };
			}
			if (code === 'trace_snapshot_required' || code === 'repair_estimate_stale' || code === 'repair_estimate_expired' || code === 'expected_run_status_required') {
				return code === 'trace_snapshot_required' || code === 'repair_estimate_stale' || code === 'repair_estimate_expired'
					? { phase: 'stale', code, message }
					: { phase: 'error', code, message };
			}
			if (code === 'writer_lock_busy' || code === 'writer_lock_malformed' || code === 'paired_workflow_active' || code === 'corpus_db_missing' || code === 'repair_requires_succeeded_run') {
				return { phase: 'blocked', code, message };
			}
			return { phase: 'error', code, message };
		}

		function repairRequestBody(estimateToken?: string) {
			return {
				run_id: c.run_id,
				filters: c.filters,
				source_route: `${window.location.pathname}${window.location.search}`,
				expected_run_status: c.status,
				note: `repair candidates from ${c.grain} cohort`,
				...(estimateToken ? { estimate_token: estimateToken } : {})
			};
		}

		function repairEstimateKey() {
			const body = repairRequestBody();
			return JSON.stringify({
				run_id: body.run_id,
				filters: body.filters,
				source_route: body.source_route,
				expected_run_status: body.expected_run_status
			});
		}

		async function estimateRepairBacklog() {
			repairState = { phase: 'estimating' };
			try {
				const res = await fetch('/api/repairs/cohort/estimate', {
					method: 'POST',
					headers: { 'content-type': 'application/json' },
					body: JSON.stringify(repairRequestBody())
				});
				const body = await res.json();
				if (!res.ok) {
					repairState = repairFailureState(body);
					return;
				}
				repairState = {
					phase: 'estimated',
					inspected: Number(body.inspected ?? 0),
					unique_selected: Number(body.unique_selected ?? 0),
					would_create: Number(body.would_create ?? 0),
					skipped_existing: Number(body.skipped_existing ?? 0),
					skipped_duplicate_selection: Number(body.skipped_duplicate_selection ?? 0),
					estimate_token: typeof body.estimate_token === 'string' ? body.estimate_token : '',
					estimate_expires_at: typeof body.estimate_expires_at === 'string' ? body.estimate_expires_at : null,
					estimate_key: repairEstimateKey()
				};
			} catch (e) {
				repairState = { phase: 'error', code: 'repair_estimate_failed', message: String(e).slice(0, 200) };
			}
		}

		async function createRepairBacklog() {
			if (!repairEstimateReady || repairState.phase !== 'estimated') return;
			const estimateToken = repairState.estimate_token;
			repairState = { phase: 'running' };
			try {
				const res = await fetch('/api/repairs/cohort', {
					method: 'POST',
					headers: { 'content-type': 'application/json' },
					body: JSON.stringify(repairRequestBody(estimateToken))
				});
			const body = await res.json();
			if (!res.ok) {
				repairState = repairFailureState(body);
				return;
			}
			repairState = {
				phase: 'done',
				created: Number(body.created ?? 0),
				skipped_existing: Number(body.skipped_existing ?? 0),
				skipped_duplicate_selection: Number(body.skipped_duplicate_selection ?? 0),
				inspected: Number(body.inspected ?? 0)
			};
		} catch (e) {
			repairState = { phase: 'error', code: 'repair_request_failed', message: String(e).slice(0, 200) };
		}
	}
</script>

<svelte:head><title>{c.run_id.slice(0, 8)} cohort · INDRA Belief</title></svelte:head>

<header>
	<div class="crumb">
		<a href="/">corpus</a><span class="sep"> / </span><a href={`/runs/${c.run_id}`}>run {c.run_id.slice(0, 8)}</a><span class="sep"> / </span><strong>cohort</strong>
	</div>
	<div class="meta">
		<span class={`run-arch run-arch-${c.architecture}`}>{archMark(c.architecture)}</span>
		<span>{c.architecture}</span>
	</div>
</header>

<main id="main">
	<section class="cohort-head">
		<h1>{c.grain} cohort</h1>
		{#if partialRun}
			<div class="run-warning" role="note">
				<span>{c.status === 'running' ? 'live run sample' : 'terminated run sample'}</span>
				<p>
					This cohort is from a <strong>{c.status}</strong> run. Rows are the persisted worker-order subset, not a completed run distribution; repair and comparison decisions should use a completed rerun.
				</p>
			</div>
		{/if}
		{#if c.filters.trace_snapshot}
			<div class="snapshot-pin" role="note">
				<span>snapshot pinned</span>
				<strong>{c.filters.trace_snapshot}</strong>
				<a href={unpinnedCohortHref}>{snapshotActionLabel}</a>
			</div>
		{/if}
		<p class="cohort-summary">
			{c.totalRows.toLocaleString()} {cohortUnit} row{c.totalRows === 1 ? '' : 's'} in run <code>{c.run_id.slice(0, 8)}</code>.
			{#if c.totalRows > c.limit}<span class="muted"> Showing first {c.limit.toLocaleString()} rows.</span>{/if}
		</p>
		{#if activeFilters.length > 0}
			<div class="filters" aria-label="active cohort filters">
				{#each activeFilters as [k, v]}
					<span class="filter-chip"><span>{k}</span><strong>{v}</strong></span>
				{/each}
			</div>
		{:else}
			<p class="muted">no filters: this is the run's full {cohortUnit} cohort.</p>
		{/if}
			{#if c.grain === 'statement' && (statementEvidenceFiltersActive || c.filters.source_stratum)}
				<p class="filter-note">
					Statement rows rank the full statement mean. Evidence-level filters must match the same scored evidence row; the evidence/source cells show one matching row, and repair backlogs fan out to matching scored evidence rows. <code>source_stratum</code> selects the separate summary stratum.
				</p>
			{/if}
			{#if multiEvidenceNote}
				<p class="filter-note">{multiEvidenceNote}</p>
			{/if}
			<div class="repair-action">
				<a class="repair-link" href={`/runs/${c.run_id}/repairs`}>open repair backlog</a>
				<button type="button" disabled={repairEstimateDisabled} onclick={estimateRepairBacklog}>
					estimate repair write
				</button>
				<button type="button" disabled={!repairEstimateReady || repairState.phase === 'running'} onclick={createRepairBacklog}>
					create repair backlog
				</button>
				{#if repairState.phase === 'estimating'}
					<span>estimating repair candidates…</span>
				{:else if repairState.phase === 'estimated' && repairEstimateStale}
					<span class="repair-stale">estimate belongs to a different cohort · estimate again</span>
				{:else if repairState.phase === 'estimated'}
					<span>
						{repairState.would_create} new from {repairState.inspected} inspected{#if repairState.skipped_existing > 0} · {repairState.skipped_existing} already open for this route{/if}{#if repairState.skipped_duplicate_selection > 0} · {repairState.skipped_duplicate_selection} duplicate selected row{repairState.skipped_duplicate_selection === 1 ? '' : 's'} collapsed{/if}
					</span>
				{:else if repairState.phase === 'running'}
					<span>writing repair candidates…</span>
			{:else if repairState.phase === 'done'}
				<span>
					{repairState.created} new · {repairState.skipped_existing} already open for this route{#if repairState.skipped_duplicate_selection > 0} · {repairState.skipped_duplicate_selection} duplicate selected row{repairState.skipped_duplicate_selection === 1 ? '' : 's'} collapsed{/if} · {repairState.inspected} inspected · <a href={`/runs/${c.run_id}/repairs`}>review</a>
				</span>
			{:else if repairState.phase === 'stale'}
				<span class="repair-stale" title={repairState.code}>{repairState.message} · <a href={currentCohortHref}>refresh</a></span>
			{:else if repairState.phase === 'blocked'}
				<span class="repair-blocked" title={repairState.code}>{repairState.message}</span>
				{:else if repairState.phase === 'error'}
					<span class="repair-error" title={repairState.code}>{repairState.message}</span>
				{:else if nonTracePartialRepairBlocked}
					<span class="repair-blocked" title="repair_requires_succeeded_run">aggregate and statement repair backlogs require a completed run</span>
				{:else if probeCoverageCohort}
					<span>repair writes one candidate per probe-covered evidence row, with observed probe counts and missing probe slots preserved</span>
				{/if}
			</div>
		{#if c.filters.trace_fidelity === 'aggregate_only'}
			<p class="repair-note">
				Aggregate-only cohorts usually need a rerun-from-cohort rather than row-level correction; this backlog records the repair intent until rerun comparison lands.
			</p>
		{/if}
			{#if traceStateMeaning && traceStateLabel}
				<div class="trace-state-note" role="note" aria-label="trace-state meaning">
					<span>trace state: {traceStateLabel}</span>
					<p>{traceStateMeaning}</p>
				</div>
			{/if}
			{#if probeCoverageCohort}
				<div class="trace-state-note" role="note" aria-label="probe coverage cohort meaning">
					<span>probe coverage evidence</span>
					<p>These rows have persisted decomposed substrate/probe slots. This is the exact denominator used by the four-probe coverage panel; aggregate verdict rows may be absent, so verdict, cost, and latency panels use a different denominator. Repair candidates attach to the representative persisted probe step and preserve missing-slot facts for rerun review.</p>
				</div>
			{/if}
			{#if c.filters.trace_state}
			<p class="repair-note">
				Trace-state cohorts are grouped from persisted scorer steps, not just aggregate verdict rows. Repair candidates attach to the representative persisted step for each evidence and keep the trace state in the correction payload.
				{#if c.filters.trace_snapshot} This cohort is pinned through <code>{c.filters.trace_snapshot}</code>.{/if}
			</p>
		{/if}
	</section>

	<div class="table-wrap">
		<table>
			<thead>
				<tr>
					<th>statement</th>
					<th>type</th>
					<th>agents</th>
					<th>evidence</th>
						{#if tracePlaneCohort}<th>trace</th>{/if}
					<th>verdict</th>
					<th class="num">score</th>
					<th class="num">Δ INDRA</th>
					<th>source</th>
					<th>graph</th>
					<th class="num">tok</th>
				</tr>
			</thead>
			<tbody>
				{#each c.rows as r}
					<tr>
						<td><a href={`/statements/${r.stmt_hash}?run_id=${c.run_id}`}><code>{shortHash(r.stmt_hash)}</code></a></td>
						<td class="type">{r.indra_type}</td>
						<td class="agents" title={r.agent_names}>{r.agent_names}</td>
						<td class="ev">
							{#if r.evidence_hash}
								<code>{shortHash(r.evidence_hash)}</code>
							{:else}
								<span>{r.n_evidences_for_stmt} scored evidence row{r.n_evidences_for_stmt === 1 ? '' : 's'}</span>
								{#if r.representative_evidence_hash}<span class="muted"> · example <code>{shortHash(r.representative_evidence_hash)}</code></span>{/if}
							{/if}
							{#if r.evidence_hash && r.n_evidences_for_stmt > 1}<span class="muted"> · {r.n_evidences_for_stmt} ev stmt</span>{/if}
							<p>{r.text ? `${c.grain === 'statement' ? 'example scored evidence: ' : ''}${r.text}` : 'no evidence text'}</p>
						</td>
							{#if tracePlaneCohort}
								<td><span class="trace-state">{r.trace_state ?? c.filters.trace_state}</span><span class="muted"> · {r.step_kind ?? 'step'}</span></td>
							{/if}
						<td>
							{#if c.grain === 'statement'}
								<span class="muted">statement mean</span>
								{:else if tracePlaneCohort && !r.verdict && r.trace_state === 'missing_aggregate'}
									<span class="trace-verdict">no aggregate row</span>
									<span class="muted"> · no verdict</span>
								{:else if tracePlaneCohort && !r.verdict && r.trace_state === 'terminated_inflight'}
									<span class="trace-verdict">not finalized</span>
									<span class="muted"> · no verdict</span>
								{:else if tracePlaneCohort && !r.verdict && r.trace_state === 'step_error'}
								<span class="trace-verdict">step error</span>
								<span class="muted"> · no aggregate verdict</span>
							{:else}
								<span class="verdict verdict-{r.verdict ?? 'unknown'}">{r.verdict ?? 'unknown'}</span>
								<span class="muted"> · {r.confidence ?? '—'}</span>
							{/if}
						</td>
						<td class="num">{fmtBelief(r.score)}</td>
						<td class="num" class:delta-pos={r.residual != null && r.residual > 0.05} class:delta-neg={r.residual != null && r.residual < -0.05}>{fmtDelta(r.residual)}</td>
						<td>
							<code>{r.source_api ?? '—'}</code>
							{#if c.grain === 'statement' && r.source_stratum && r.source_stratum !== r.source_api}<span class="muted"> · stratum <code>{r.source_stratum}</code></span>{/if}
							{#if r.pmid}<span class="muted"> · {r.pmid}</span>{/if}
						</td>
						<td>{r.has_supports ? 'supports-rich' : 'no supports'}</td>
						<td class="num">{r.prompt_tokens ?? '—'}→{r.out_tokens ?? '—'}</td>
					</tr>
				{/each}
			</tbody>
		</table>
		{#if c.rows.length === 0}
			<div class="empty-state" role="note" aria-label="empty cohort diagnostics">
				<p class="hint">no {cohortUnit} rows match this cohort filter.</p>
				{#if c.emptyDiagnostics.length > 0}
					<ul>
						{#each c.emptyDiagnostics as d}
							<li class={`empty-${d.kind}`}>
								{#if d.filter}<code>{d.filter}</code>{#if d.value}<span>=</span><code>{d.value}</code>{/if}<span> · </span>{/if}{d.message}
							</li>
						{/each}
					</ul>
				{/if}
			</div>
		{/if}
	</div>
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
	.crumb a {
		color: var(--ink-muted);
		text-decoration: none;
	}
	.crumb a:hover {
		color: var(--ink);
	}
	.crumb strong,
	.run-arch {
		color: var(--ink);
		font-weight: 500;
	}
	.crumb .sep,
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
	.cohort-head {
		margin-bottom: 1.2rem;
	}
	.run-warning {
		max-width: 920px;
		border-left: 3px solid #8a6a00;
		background: rgba(138, 106, 0, 0.04);
		padding: 0.55rem 0 0.55rem 0.85rem;
		margin: 0 0 0.9rem;
	}
	.run-warning span {
		display: block;
		font-family: var(--mono);
		font-size: 0.72rem;
		color: #8a6a00;
		text-transform: lowercase;
		letter-spacing: 0.02em;
	}
	.run-warning p {
		margin: 0.2rem 0 0;
		color: var(--ink-muted);
		font-size: 0.9rem;
		line-height: 1.45;
	}
	.snapshot-pin {
		display: flex;
		flex-wrap: wrap;
		align-items: baseline;
		gap: 0.45rem 0.75rem;
		max-width: 920px;
		border-left: 3px solid var(--accent);
		background: color-mix(in srgb, var(--accent) 5%, transparent);
		padding: 0.5rem 0 0.5rem 0.85rem;
		margin: 0 0 0.9rem;
		font-family: var(--mono);
		font-size: 0.78rem;
	}
	.snapshot-pin span {
		color: var(--accent);
		text-transform: lowercase;
	}
	.snapshot-pin strong {
		color: var(--ink);
		font-weight: 500;
	}
	.snapshot-pin a {
		color: var(--accent);
		text-decoration: none;
	}
	.snapshot-pin a:hover {
		text-decoration: underline;
	}
	h1 {
		font-family: var(--serif);
		font-size: 1.5rem;
		font-weight: 400;
		margin: 0 0 0.4rem;
	}
	.cohort-summary {
		margin: 0 0 0.6rem;
		color: var(--ink);
	}
	.filters {
		display: flex;
		flex-wrap: wrap;
		gap: 0.4rem;
		font-family: var(--mono);
	}
	.repair-action {
		display: flex;
		flex-wrap: wrap;
		gap: 0.55rem;
		align-items: baseline;
		margin-top: 0.75rem;
		font-family: var(--mono);
		font-size: 0.76rem;
	}
	.repair-action button {
		font-family: var(--mono);
		font-size: 0.76rem;
		padding: 0.2rem 0.55rem;
		border: 1px solid var(--accent);
		background: transparent;
		color: var(--accent);
		cursor: pointer;
		text-transform: lowercase;
	}
	.repair-action button:hover:not(:disabled) {
		background: color-mix(in srgb, var(--accent) 8%, transparent);
	}
	.repair-action button:disabled {
		color: var(--ink-faint);
		border-color: var(--rule);
		cursor: default;
	}
	.repair-action span {
		color: var(--ink-muted);
	}
	.repair-link,
	.repair-action span a {
		color: var(--accent);
		text-decoration: none;
	}
	.repair-link:hover,
	.repair-action span a:hover {
		text-decoration: underline;
	}
	.repair-action .repair-error {
		color: var(--accent);
	}
	.repair-action .repair-stale {
		color: #8a6a00;
	}
	.repair-action .repair-blocked {
		color: var(--ink-muted);
	}
	.repair-note {
		font-family: var(--mono);
		font-size: 0.74rem;
		color: var(--ink-muted);
		margin: 0.45rem 0 0;
	}
	.trace-state-note {
		max-width: 920px;
		border-left: 3px solid var(--accent);
		background: color-mix(in srgb, var(--accent) 4%, transparent);
		padding: 0.55rem 0 0.55rem 0.85rem;
		margin: 0.75rem 0 0;
	}
	.trace-state-note span {
		display: block;
		font-family: var(--mono);
		font-size: 0.72rem;
		color: var(--accent);
		text-transform: lowercase;
		letter-spacing: 0.02em;
	}
	.trace-state-note p {
		margin: 0.25rem 0 0;
		color: var(--ink-muted);
		font-size: 0.9rem;
		line-height: 1.45;
		max-width: 82ch;
	}
	.filter-note {
		font-family: var(--mono);
		font-size: 0.74rem;
		color: var(--ink-muted);
		margin: 0.45rem 0 0;
		max-width: 78ch;
	}
	.filter-chip {
		border-left: 2px solid var(--rule);
		padding-left: 0.45rem;
		font-size: 0.78rem;
	}
	.filter-chip span {
		color: var(--ink-muted);
		margin-right: 0.3rem;
	}
	.table-wrap {
		overflow-x: auto;
		border-top: 1px solid var(--ink);
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
		padding: 0.45rem 0.7rem 0.45rem 0;
		border-bottom: 1px dotted var(--rule);
	}
	th {
		color: var(--ink-muted);
		font-size: 0.7rem;
		font-weight: 500;
		text-transform: lowercase;
		letter-spacing: 0.02em;
	}
	.num {
		text-align: right;
	}
	a {
		color: var(--accent);
		text-decoration: none;
	}
	a:hover {
		text-decoration: underline;
	}
	.type,
	.verdict-correct {
		color: var(--ok-green);
	}
	.trace-state {
		color: var(--accent);
		white-space: nowrap;
	}
	.trace-verdict {
		color: var(--accent);
	}
	.verdict-incorrect,
	.delta-neg {
		color: var(--accent);
	}
	.verdict-abstain {
		color: var(--ink-muted);
	}
	.delta-pos {
		color: var(--ok-green);
	}
	.agents {
		max-width: 16rem;
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}
	.ev {
		max-width: 28rem;
	}
	.ev p {
		font-family: var(--serif);
		font-size: 0.86rem;
		color: var(--ink);
		margin: 0.2rem 0 0;
		line-height: 1.35;
	}
	.hint {
		font-style: italic;
		color: var(--ink-muted);
	}
	.empty-state {
		max-width: 760px;
		padding: 0.85rem 0;
	}
	.empty-state ul {
		list-style: none;
		margin: 0.45rem 0 0;
		padding: 0;
		font-family: var(--mono);
		font-size: 0.76rem;
		color: var(--ink-muted);
	}
	.empty-state li {
		border-left: 2px solid var(--rule);
		padding: 0.22rem 0 0.22rem 0.55rem;
		margin: 0.2rem 0;
	}
	.empty-state li.empty-absent_in_run {
		border-left-color: #8a6a00;
		color: #745900;
	}
	.empty-state li.empty-not_applicable {
		border-left-color: var(--accent);
		color: var(--accent);
	}
	.empty-state li.empty-no_intersection {
		border-left-color: var(--ok-green);
		color: var(--ink);
	}
	.empty-state li.empty-empty_run {
		border-left-color: var(--ink);
		color: var(--ink);
	}
	.empty-state code {
		color: var(--ink);
	}
</style>
