<script lang="ts">
	import type { TraceFidelitySummary } from '$lib/db';
	import {
		TRACE_FIDELITY_STATES,
		TRACE_STATE_LABELS,
		TRACE_STATE_NOTES,
		type TraceFidelityState
	} from '$lib/traceState';

	let {
		fidelity,
		runStatus = 'unknown',
		compareToParam = null
	}: { fidelity: TraceFidelitySummary; runStatus?: string; compareToParam?: string | null } = $props();

	const states = TRACE_FIDELITY_STATES;
	const partialSample = $derived(runStatus !== 'succeeded' && fidelity.n_evidences > 0);
	const liveSample = $derived(runStatus === 'running');
	const sampleLabel = $derived(liveSample
		? 'running run snapshot'
		: runStatus === 'canceled'
			? 'canceled run sample'
			: runStatus === 'failed'
				? 'failed run sample'
				: 'terminated run sample');
	const sampleCopy = $derived(liveSample
		? 'Trace rows may continue accumulating while the worker is active. Counts and detail pages are pinned to the readable persisted-step snapshot named here.'
		: 'Trace rows are the worker-order subset persisted before termination, not a random sample or completed run distribution.');

	function labelFor(state: TraceFidelityState): string {
		if (state === 'terminated_inflight' && liveSample) return 'pending';
		return TRACE_STATE_LABELS[state];
	}

	function noteFor(state: TraceFidelityState): string {
		if (state === 'terminated_inflight' && liveSample) {
			return 'pre-aggregate steps are persisted and may still finalize into an aggregate verdict';
		}
		return TRACE_STATE_NOTES[state];
	}

	function pct(n: number): number {
		return fidelity.n_evidences > 0 ? (n / fidelity.n_evidences) * 100 : 0;
	}

	function countFor(state: TraceFidelityState): number {
		return fidelity.counts[state] ?? 0;
	}

	function cohortHrefFor(state: TraceFidelityState): string | null {
		if (countFor(state) === 0 || state === 'not_applicable') return null;
		const sp = new URLSearchParams({ trace_state: state });
		if (fidelity.detail_snapshot_started_at) sp.set('trace_snapshot', fidelity.detail_snapshot_started_at);
		return `/runs/${fidelity.run_id}/cohort?${sp.toString()}`;
	}

	function pctStr(n: number): string {
		const p = pct(n);
		if (p === 0) return '0%';
		if (p < 1) return '<1%';
		if (p > 99 && p < 100) return '>99%';
		return `${Math.round(p)}%`;
	}

	const railLabel = $derived.by(() => {
		const active = states.filter((s) => countFor(s) > 0);
		if (active.length === 0) return 'trace fidelity: no evidence rows';
		return `trace fidelity: ${active.map((s) => `${labelFor(s)} ${countFor(s)}`).join(', ')}`;
	});
	const detailPage = $derived(Math.floor(fidelity.detail_offset / fidelity.detail_limit) + 1);
	const detailPageCount = $derived(Math.max(1, Math.ceil(fidelity.n_evidences / fidelity.detail_limit)));
	const detailStart = $derived(fidelity.n_evidences === 0 ? 0 : fidelity.detail_offset + 1);
	const detailEnd = $derived(Math.min(fidelity.detail_offset + fidelity.rows.length, fidelity.n_evidences));
	const detailWindowLabel = $derived(
		fidelity.n_evidences === 0
			? '0 evidence trace states'
			: `evidence trace states ${detailStart}-${detailEnd} of ${fidelity.n_evidences}`
	);

	function detailHref(page: number): string {
		const sp = new URLSearchParams();
		if (compareToParam) sp.set('compare_to', compareToParam);
		if (page > 1) sp.set('trace_page', String(page));
		if (fidelity.detail_snapshot_started_at) sp.set('trace_snapshot', fidelity.detail_snapshot_started_at);
		const qs = sp.toString();
		return `/runs/${fidelity.run_id}${qs ? `?${qs}` : ''}#trace-fidelity-details`;
	}

	function latestSnapshotHref(): string {
		const sp = new URLSearchParams();
		if (compareToParam) sp.set('compare_to', compareToParam);
		const qs = sp.toString();
		return `/runs/${fidelity.run_id}${qs ? `?${qs}` : ''}#trace-fidelity-details`;
	}
</script>

<section class="tf">
	<h2 class="tf-h">
		trace fidelity
		<span class="tf-run-id" title="run_id">{fidelity.run_id.slice(0, 8)}</span>
		<span class="tf-arch" title="architecture">{fidelity.architecture}</span>
	</h2>

	{#if partialSample}
		<div class="tf-interrupted" role="note">
			<span>{sampleLabel}</span>
			<p>
				This run is <strong>{runStatus}</strong>; {sampleCopy}
				Trace fidelity counts any persisted step; adjacent panels may use aggregate verdict rows
				or decomposed substrate/probe rows as their own denominators.
				{#if liveSample && fidelity.detail_snapshot_started_at}
					Snapshot through <code>{fidelity.detail_snapshot_started_at}</code>.
					<a href={latestSnapshotHref()}>latest snapshot</a>
				{/if}
			</p>
		</div>
	{/if}

	<div class="tf-grammar" aria-label="native trace grammar">
		{#each fidelity.native_grammar as part, i}
			<span class="tf-grammar-part">{part}</span>
			{#if i < fidelity.native_grammar.length - 1}<span class="tf-grammar-arrow">-&gt;</span>{/if}
		{/each}
	</div>

	<div class="tf-rail" class:tf-rail-partial={partialSample} role="img" aria-label={railLabel}>
		{#each states as state}
			{@const n = countFor(state)}
			{#if n > 0}
				<span class="tf-seg tf-seg-{state}" style:width="{pct(n)}%" title="{labelFor(state)} {n}/{fidelity.n_evidences}"></span>
			{/if}
		{/each}
	</div>

	<table class="tf-counts" aria-label="trace fidelity counts">
		<thead>
			<tr>
				<th>state</th>
				<th class="num">count</th>
				<th class="num">share</th>
				<th>meaning</th>
			</tr>
		</thead>
		<tbody>
			{#each states as state}
				{@const n = countFor(state)}
				<tr class:tf-zero={n === 0}>
					<td>
						{#if cohortHrefFor(state)}
							<a class="tf-state-link" href={cohortHrefFor(state) ?? ''}><span class="tf-state tf-state-{state}">{labelFor(state)}</span></a>
						{:else}
							<span class="tf-state tf-state-{state}">{labelFor(state)}</span>
						{/if}
					</td>
					<td class="num">{n}</td>
					<td class="num">{pctStr(n)}</td>
					<td>{noteFor(state)}</td>
				</tr>
			{/each}
		</tbody>
	</table>

	{#if fidelity.limitations.length > 0}
		<ul class="tf-limitations">
			{#each fidelity.limitations as limitation}
				<li>{limitation}</li>
			{/each}
		</ul>
	{/if}

	{#if fidelity.detail_snapshot_started_at && !liveSample}
		<p class="tf-snapshot">trace snapshot through <code>{fidelity.detail_snapshot_started_at}</code></p>
	{/if}

	<details class="tf-details" id="trace-fidelity-details" open={detailPage > 1}>
		<summary>{detailWindowLabel}{#if partialSample}; partial{/if}</summary>
		{#if detailPageCount > 1}
			<nav class="tf-pager" aria-label="trace evidence pages">
				{#if detailPage > 1}
					<a href={detailHref(detailPage - 1)}>prev</a>
				{:else}
					<span aria-disabled="true">prev</span>
				{/if}
				<span>page {detailPage} of {detailPageCount}</span>
				{#if detailPage < detailPageCount}
					<a href={detailHref(detailPage + 1)}>next</a>
				{:else}
					<span aria-disabled="true">next</span>
				{/if}
			</nav>
		{/if}
		<table class="tf-evidence" aria-label="per-evidence trace fidelity">
			<thead>
				<tr>
					<th>evidence</th>
					<th>state</th>
					<th>captured</th>
					<th>missing native fields</th>
				</tr>
			</thead>
			<tbody>
				{#each fidelity.rows as row}
					<tr>
						<td>
							{#if row.stmt_hash}
								<a href={`/statements/${row.stmt_hash}?run_id=${fidelity.run_id}`}><code>{row.evidence_hash.slice(0, 8)}</code></a>
							{:else}
								<code>{row.evidence_hash.slice(0, 8)}</code>
							{/if}
						</td>
						<td><span class="tf-state tf-state-{row.state}">{labelFor(row.state)}</span></td>
						<td>{row.captured_steps.join(', ') || '-'}</td>
						<td title={row.note}>{row.missing_native_steps.join(', ') || '-'}</td>
					</tr>
				{/each}
			</tbody>
		</table>
		{#if detailPageCount > 1}
			<p class="tf-more">showing {detailStart}-{detailEnd} of {fidelity.n_evidences} evidence rows.</p>
		{/if}
	</details>

	<h3 class="tf-subh">panel applicability</h3>
	<table class="tf-app" aria-label="panel applicability by architecture">
		<thead>
			<tr>
				<th>panel</th>
				<th>kind</th>
				<th>why</th>
			</tr>
		</thead>
		<tbody>
			{#each fidelity.panel_applicability as row}
				<tr>
					<td>
						{#if row.cohort_href}
							<a href={row.cohort_href}>{row.panel}</a>
						{:else}
							{row.panel}
						{/if}
					</td>
					<td><span class="tf-app-kind tf-app-{row.applicability}">{row.applicability.replace('_', '-')}</span></td>
					<td>{row.reason}</td>
				</tr>
			{/each}
		</tbody>
	</table>
</section>

<style>
	.tf {
		margin: 0 0 2.5rem;
		--tf-partial: #8a6a00;
	}
	.tf-h {
		font-family: var(--serif);
		font-size: 1.15rem;
		font-weight: 400;
		color: var(--ink);
		margin: 0 0 0.6rem;
		display: flex;
		align-items: baseline;
		gap: 0.6rem;
	}
	.tf-run-id,
	.tf-arch {
		font-family: var(--mono);
		font-size: 0.72rem;
		color: var(--ink-faint);
	}
	.tf-grammar {
		display: flex;
		flex-wrap: wrap;
		gap: 0.35rem;
		align-items: center;
		max-width: 840px;
		margin: 0 0 0.8rem;
		font-family: var(--mono);
		font-size: 0.76rem;
		color: var(--ink-muted);
	}
	.tf-grammar-part {
		color: var(--ink);
	}
	.tf-grammar-arrow {
		color: var(--ink-faint);
	}
	.tf-interrupted {
		max-width: 840px;
		border-left: 3px solid var(--tf-partial);
		padding: 0.55rem 0 0.55rem 0.85rem;
		margin: 0 0 0.8rem;
		background: rgba(138, 106, 0, 0.04);
	}
	.tf-interrupted span {
		display: block;
		font-family: var(--mono);
		font-size: 0.72rem;
		color: var(--tf-partial);
		text-transform: lowercase;
		letter-spacing: 0.02em;
	}
	.tf-interrupted p {
		margin: 0.2rem 0 0;
		color: var(--ink-muted);
		font-size: 0.9rem;
		line-height: 1.45;
	}
	.tf-rail {
		display: flex;
		width: 100%;
		max-width: 840px;
		height: 18px;
		border: 1px solid var(--rule);
		overflow: hidden;
		margin: 0 0 0.7rem;
		position: relative;
	}
	.tf-rail-partial::after {
		content: '';
		position: absolute;
		inset: 0;
		pointer-events: none;
		background: repeating-linear-gradient(
			135deg,
			rgba(138, 106, 0, 0.28),
			rgba(138, 106, 0, 0.28) 2px,
			transparent 2px,
			transparent 7px
		);
	}
	.tf-seg {
		display: block;
		min-width: 2px;
	}
	.tf-seg-full { background: var(--ok-green); }
	.tf-seg-partial { background: var(--tf-partial); }
	.tf-seg-aggregate_only { background: var(--ink-muted); }
	.tf-seg-terminated_inflight {
		background: repeating-linear-gradient(
			135deg,
			var(--ink-faint),
			var(--ink-faint) 2px,
			var(--paper) 2px,
			var(--paper) 6px
		);
	}
	.tf-seg-not_applicable {
		background: repeating-linear-gradient(
			135deg,
			var(--ink-faint),
			var(--ink-faint) 2px,
			var(--paper) 2px,
			var(--paper) 5px
		);
	}
	.tf-seg-missing_aggregate {
		background: repeating-linear-gradient(
			135deg,
			var(--accent),
			var(--accent) 2px,
			var(--paper) 2px,
			var(--paper) 6px
		);
	}
	.tf-seg-step_error { background: var(--accent); }
	.tf-counts,
	.tf-evidence,
	.tf-app {
		width: 100%;
		max-width: 920px;
		border-collapse: collapse;
		font-family: var(--mono);
		font-size: 0.78rem;
		font-variant-numeric: tabular-nums;
	}
	.tf-counts th,
	.tf-counts td,
	.tf-evidence th,
	.tf-evidence td,
	.tf-app th,
	.tf-app td {
		text-align: left;
		padding: 0.35rem 0.6rem 0.35rem 0;
		border-bottom: 1px dotted var(--rule);
		vertical-align: top;
		overflow-wrap: anywhere;
	}
	.tf-counts th,
	.tf-evidence th,
	.tf-app th {
		font-weight: 500;
		color: var(--ink-muted);
		font-size: 0.7rem;
		text-transform: lowercase;
		letter-spacing: 0.02em;
	}
	.tf-counts .num {
		text-align: right;
	}
	.tf-state-link,
	.tf-evidence a {
		text-decoration: none;
	}
	.tf-state-link:hover,
	.tf-evidence a:hover {
		text-decoration: underline;
		text-decoration-thickness: 1px;
	}
	.tf-zero {
		color: var(--ink-faint);
	}
	.tf-state,
	.tf-app-kind {
		white-space: nowrap;
	}
	.tf-state-full { color: var(--ok-green); }
	.tf-state-partial { color: var(--tf-partial); }
	.tf-state-aggregate_only,
	.tf-state-not_applicable,
	.tf-state-terminated_inflight { color: var(--ink-muted); }
	.tf-state-missing_aggregate,
	.tf-state-step_error { color: var(--accent); }
	.tf-zero .tf-state-missing_aggregate,
	.tf-zero .tf-state-step_error {
		color: var(--ink-faint);
	}
	.tf-limitations {
		margin: 0.8rem 0 0;
		padding: 0 0 0 1rem;
		font-family: var(--serif);
		font-size: 0.9rem;
		color: var(--ink-muted);
	}
	.tf-snapshot {
		margin: 0.7rem 0 0;
		font-family: var(--mono);
		font-size: 0.76rem;
		color: var(--ink-muted);
	}
	.tf-details {
		margin: 1rem 0 1.3rem;
	}
	.tf-pager {
		display: flex;
		gap: 0.7rem;
		align-items: center;
		margin: 0.45rem 0 0.55rem;
		font-family: var(--mono);
		font-size: 0.76rem;
		color: var(--ink-muted);
	}
	.tf-pager span[aria-disabled='true'] {
		color: var(--ink-faint);
	}
	.tf-details summary,
	.tf-subh {
		font-family: var(--mono);
		font-size: 0.78rem;
		color: var(--ink-muted);
		cursor: pointer;
		font-weight: 500;
		text-transform: lowercase;
		letter-spacing: 0.02em;
	}
	.tf-subh {
		cursor: default;
		margin: 1rem 0 0.35rem;
	}
	.tf-more {
		font-family: var(--mono);
		font-size: 0.76rem;
		color: var(--ink-faint);
	}
	.tf-app-arch_blind { color: var(--ok-green); }
	.tf-app-arch_conditioned { color: var(--ink); }
	.tf-app-paired_only { color: #8a6a00; }
	.tf-app-not_defined { color: var(--ink-muted); }
	a {
		color: var(--accent);
		text-decoration: none;
	}
	a:hover {
		text-decoration: underline;
	}
	@media (max-width: 640px) {
		.tf-counts,
		.tf-evidence,
		.tf-app {
			display: block;
			max-width: 100%;
			overflow-x: auto;
		}
	}
</style>
