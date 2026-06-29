<script lang="ts">
	import FrontierStrip from '$lib/components/FrontierStrip.svelte';
	import { fmtCost, fmtCostFull } from '$lib/format';
	import { page } from '$app/state';
	import type { FrontierRun } from '$lib/data/queries';

	let { data } = $props();
	const f = $derived(data.frontier);

	// the x-axis lens: economic frontier (cost) or scaling frontier (model size).
	// Client state so switching MORPHS the plot rather than reloading; seeded from
	// (and synced to) ?axis= so the lens is shareable/bookmarkable.
	let axis = $state<'cost' | 'size'>(page.url.searchParams.get('axis') === 'size' ? 'size' : 'cost');
	function setAxis(a: 'cost' | 'size') {
		axis = a;
		const u = new URL(page.url);
		u.searchParams.set('axis', a);
		history.replaceState(history.state, '', u); // update URL, no reload (keep the morph)
	}
	const onFrontier = (r: FrontierRun) => (axis === 'cost' ? r.on_frontier_cost : r.on_frontier_size);
	const domBy = (r: FrontierRun) => (axis === 'cost' ? r.dominated_by_cost : r.dominated_by_size);
	const plottable = (r: FrontierRun) =>
		axis === 'cost' ? r.cost_known && r.usd_per_1k != null : r.size_known;
	const fmtParams = (b: number | null) => (b == null ? '—' : b >= 1000 ? `${b / 1000}T` : `${b}B`);
	const onFrontierN = $derived(f.runs.filter(onFrontier).length);
	const spanLabel = $derived(
		axis === 'cost'
			? f.cost_span.min == null
				? '$0 only'
				: `${fmtCostFull(f.cost_span.min)}–${fmtCostFull(f.cost_span.max)}`
			: f.size_span.min == null
				? 'n/a'
				: `${fmtParams(f.size_span.min)}–${fmtParams(f.size_span.max)}`
	);

	// pair selection (client-only): pick two runs → drill into /compare
	let selected = $state<string[]>([]);
	function toggle(id: string) {
		if (selected.includes(id)) selected = selected.filter((s) => s !== id);
		else if (selected.length < 2) selected = [...selected, id];
		else selected = [selected[1], id]; // roll the window: keep most-recent two
	}
	const pairHref = $derived(
		selected.length === 2 ? `/compare?a=${selected[0]}&b=${selected[1]}&mode=gold` : null
	);
	function modelOf(id: string): string {
		return f.runs.find((r) => r.run_id === id)?.model ?? id.slice(0, 8);
	}

	// f.runs is now one row PER MODEL (repeat runs folded in); the raw run total
	// is the sum of reps, shown alongside so "models" vs "runs" stays honest.
	const totalRuns = $derived(f.runs.reduce((a, r) => a + r.n_reps, 0));

	function rank(i: number): string {
		return String(i + 1);
	}
	function ci(r: FrontierRun): string {
		return `[${r.err_f1_lo.toFixed(2)}–${r.err_f1_hi.toFixed(2)}]`;
	}
</script>

<svelte:head><title>frontier · cost × error-F1</title></svelte:head>

<main id="main" class="wrap">
	<header class="head">
		<h1>the frontier</h1>
		<p class="lede">Every run scored on one substrate, read as cost against error-detection F1.</p>
	</header>

	<!-- substrate selector: the join boundary -->
	<nav class="subs" aria-label="substrate">
		{#each f.substrates as s}
			<a
				href="/frontier?substrate={encodeURIComponent(s.key)}"
				class:active={s.key === f.selected}
				aria-current={s.key === f.selected ? 'page' : undefined}
			>
				{s.label}
				<span class="sub-meta">{s.n_runs} runs · gold {s.gold_n} · {s.n_cost_known} priced</span>
			</a>
		{/each}
	</nav>

	{#if f.runs.length === 0}
		<p class="empty">No runs on this substrate.</p>
	{:else}
		<!-- axis lens: the switch that re-measures the same runs -->
		<div class="axis-switch" role="group" aria-label="x-axis">
			<span class="ax-label">F1 over</span>
			<button type="button" class:on={axis === 'cost'} aria-pressed={axis === 'cost'} onclick={() => setAxis('cost')}>cost</button>
			<button type="button" class:on={axis === 'size'} aria-pressed={axis === 'size'} onclick={() => setAxis('size')}>model size</button>
		</div>

		<!-- headline scalars -->
		<dl class="scalars">
			<div><dt>models</dt><dd>{f.runs.length}</dd></div>
			{#if totalRuns > f.runs.length}
				<div><dt>runs</dt><dd>{totalRuns}</dd></div>
			{/if}
			<div><dt>on frontier</dt><dd>{onFrontierN}</dd></div>
			<div><dt>gold n</dt><dd>{f.n_gold}</dd></div>
			<div>
				<dt>{axis === 'cost' ? 'cost span /1k' : 'size span'}</dt>
				<dd>{spanLabel}</dd>
			</div>
			<div>
				<dt>F1 span</dt>
				<dd>{f.f1_span.min.toFixed(2)}–{f.f1_span.max.toFixed(2)}</dd>
			</div>
		</dl>

		<FrontierStrip runs={f.runs} {axis} {selected} onpick={toggle} />

		{#if f.note}<p class="note">⚠ {f.note}</p>{/if}

		<!-- pair-drill action bar -->
		<div class="pair-bar" aria-live="polite">
			{#if selected.length === 0}
				<span class="hint">pick two runs (dot or row) to drill the pair →</span>
			{:else if selected.length === 1}
				<span class="hint">{modelOf(selected[0])} selected · pick one more →</span>
			{:else}
				<a class="drill" href={pairHref}>compare {modelOf(selected[0])} vs {modelOf(selected[1])} →</a>
				<button class="clear" onclick={() => (selected = [])}>clear</button>
			{/if}
		</div>

		<!-- the ledger: scannable numbers behind the plot; columns follow the axis -->
		<table class="ledger">
			<thead>
				<tr>
					<th class="r-rank">#</th>
					<th class="r-model">model</th>
					<th class="r-num">err-F1</th>
					<th class="r-ci">95% CI</th>
					<th class="r-num">acc</th>
					{#if axis === 'cost'}
						<th class="r-num">$/1k</th>
						<th class="r-num">total</th>
					{:else}
						<th class="r-num">params</th>
						<th class="r-num">active</th>
					{/if}
					<th class="r-front">frontier</th>
				</tr>
			</thead>
			<tbody>
				{#each f.runs as r, i (r.run_id)}
					<tr
						class:sel={selected.includes(r.run_id)}
						class:dominated={!onFrontier(r)}
						onclick={() => toggle(r.run_id)}
					>
						<td class="r-rank">
							<button
								type="button"
								class="pickbtn"
								class:lead={i === 0}
								aria-pressed={selected.includes(r.run_id)}
								aria-label="select {r.model} for the pairwise compare"
								onclick={(e) => {
									e.stopPropagation();
									toggle(r.run_id);
								}}>{rank(i)}</button>
						</td>
						<td class="r-model">
							<a href="/runs/{r.run_id}" onclick={(e) => e.stopPropagation()}>{r.model}</a>
							{#if r.n_reps > 1}
								<span
									class="reps"
									title="mean of {r.n_reps} repeat runs; err-F1 and CI fold their spread. Drill opens the typical run."
									>×{r.n_reps}</span
								>
							{/if}
							<span class="date">{r.generated_date ?? ''}</span>
						</td>
						<td class="r-num strong">{r.err_f1.toFixed(2)}</td>
						<td class="r-ci">{ci(r)}</td>
						<td class="r-num">{r.accuracy.toFixed(2)}</td>
						{#if axis === 'cost'}
							<td class="r-num">{r.cost_estimated ? '~' : ''}{fmtCostFull(r.usd_per_1k)}</td>
							<td class="r-num">{fmtCost(r.cost)}</td>
						{:else}
							<td class="r-num">{fmtParams(r.params_total_b)}</td>
							<td class="r-num">{r.params_active_b != null ? fmtParams(r.params_active_b) : r.size_known ? 'dense' : '—'}</td>
						{/if}
						<td class="r-front">
							{#if !plottable(r)}<span class="na">{axis === 'cost' ? 'unpriced' : 'size n/a'}</span>
							{:else if onFrontier(r)}<span class="on">▲ frontier</span>
							{:else}<span class="dag">† {domBy(r)}</span>{/if}
						</td>
					</tr>
				{/each}
			</tbody>
		</table>
		<p class="foot">
			Click a row or dot to add it to the pair; the rank is by error-F1 on n={f.n_gold} gold —
			treat it as indicative where the CIs overlap. † marks a run another beats on both
			{axis === 'cost' ? 'cost' : 'model size'} and F1.
			Repeat runs of one model fold into a single point (×N): err-F1 and cost are the across-run
			mean and the CI band spans their spread — so a model is never plotted, labelled, or ranked
			more than once.
			{axis === 'cost'
				? 'Cost is read verbatim from each run’s baked block; the viewer holds no price table. No model is free to run: self-hosted models have no observed spend, so they carry a Bedrock-grounded estimate — marked ~ and drawn as a hollow dot.'
				: 'Model size is baked per run from a ground-truth registry. A hollow dot marks an estimated/inferred size (exact count undisclosed, or inferred from the base version); closed-weight models have no disclosed size and are not plotted.'}
			A hollow dot always means the x-value is an estimate, not an observation — on either axis.
		</p>
	{/if}
</main>

<style>
	.wrap {
		max-width: 1100px;
		margin: 0 auto;
		padding: 1.6rem 1.5rem 4rem;
	}
	.head h1 {
		font-family: var(--serif);
		font-weight: 400;
		font-size: 1.7rem;
		margin: 0 0 0.3rem;
	}
	.lede {
		font-family: var(--serif);
		color: var(--ink-muted);
		max-width: 62ch;
		line-height: 1.5;
		margin: 0 0 1.3rem;
	}
	.subs {
		display: flex;
		flex-wrap: wrap;
		gap: 0.5rem 1.4rem;
		border-bottom: 1px solid var(--rule);
		padding-bottom: 0.6rem;
		margin-bottom: 1.2rem;
		font-family: var(--mono);
		font-size: 0.82rem;
	}
	.subs a {
		color: var(--ink-muted);
		text-decoration: none;
		border-bottom: 2px solid transparent;
		padding-bottom: 0.3rem;
	}
	.subs a.active {
		color: var(--ink);
		border-bottom-color: var(--accent);
	}
	.subs a:hover {
		color: var(--ink);
	}
	.sub-meta {
		display: block;
		font-size: 0.68rem;
		color: var(--ink-faint);
	}
	.axis-switch {
		display: flex;
		align-items: baseline;
		gap: 0.9rem;
		margin: 0 0 1rem;
		font-family: var(--mono);
	}
	.ax-label {
		color: var(--ink-faint);
		font-size: 0.7rem;
		text-transform: lowercase;
		letter-spacing: 0.04em;
	}
	.axis-switch button {
		background: none;
		border: none;
		padding: 0 0 0.25rem;
		font-family: var(--mono);
		font-size: 0.86rem;
		color: var(--ink-muted);
		cursor: pointer;
		border-bottom: 2px solid transparent;
	}
	.axis-switch button:hover {
		color: var(--ink);
	}
	.axis-switch button.on {
		color: var(--ink);
		border-bottom-color: var(--accent);
	}
	.axis-switch button:focus-visible {
		outline: 1.5px solid var(--accent);
		outline-offset: 2px;
	}
	.scalars {
		display: flex;
		flex-wrap: wrap;
		gap: 1.8rem;
		margin: 0 0 0.9rem;
	}
	.scalars div {
		display: flex;
		flex-direction: column;
	}
	.scalars dt {
		font-family: var(--mono);
		font-size: 0.66rem;
		text-transform: lowercase;
		letter-spacing: 0.04em;
		color: var(--ink-faint);
	}
	.scalars dd {
		font-family: var(--mono);
		font-size: 1.05rem;
		margin: 0.1rem 0 0;
		font-variant-numeric: tabular-nums;
		color: var(--ink);
	}
	.note {
		font-family: var(--mono);
		font-size: 0.74rem;
		color: var(--blocked);
		margin: 0.5rem 0 0;
	}
	.pair-bar {
		min-height: 1.8rem;
		margin: 1rem 0 0.4rem;
		font-family: var(--mono);
		font-size: 0.82rem;
		display: flex;
		align-items: center;
		gap: 1rem;
	}
	.pair-bar .hint {
		color: var(--ink-faint);
	}
	.drill {
		color: var(--accent);
		text-decoration: none;
		border-bottom: 1px solid var(--accent);
	}
	.clear {
		background: none;
		border: none;
		color: var(--ink-faint);
		font-family: var(--mono);
		font-size: 0.78rem;
		cursor: pointer;
		text-decoration: underline;
		text-underline-offset: 2px;
	}
	.ledger {
		width: 100%;
		border-collapse: collapse;
		font-family: var(--mono);
		font-size: 0.82rem;
		margin-top: 0.6rem;
	}
	.ledger th {
		text-align: left;
		font-weight: 400;
		font-size: 0.66rem;
		text-transform: lowercase;
		letter-spacing: 0.04em;
		color: var(--ink-faint);
		border-bottom: 1px solid var(--rule);
		padding: 0 0.6rem 0.35rem 0;
	}
	.ledger td {
		padding: 0.4rem 0.6rem 0.4rem 0;
		border-bottom: 1px dotted var(--rule);
		vertical-align: baseline;
	}
	.ledger tbody tr {
		cursor: pointer;
	}
	.ledger tbody tr:hover {
		background: var(--accent-wash);
	}
	.ledger tr.sel {
		background: var(--accent-wash);
	}
	.ledger tr.sel td {
		border-bottom-color: var(--accent);
	}
	.ledger tr.dominated .r-model a,
	.ledger tr.dominated .r-num {
		color: var(--ink-muted);
	}
	.r-num {
		text-align: right;
		font-variant-numeric: tabular-nums;
	}
	.r-num.strong {
		color: var(--ink);
		font-weight: 500;
	}
	.r-ci {
		color: var(--ink-faint);
		font-variant-numeric: tabular-nums;
	}
	.r-rank {
		width: 1.6rem;
	}
	.pickbtn {
		background: none;
		border: none;
		padding: 0.1rem 0.25rem;
		margin: -0.1rem -0.25rem;
		font-family: var(--mono);
		font-size: 0.82rem;
		color: var(--ink-faint);
		cursor: pointer;
		border-radius: 0;
	}
	.pickbtn:hover {
		color: var(--ink);
	}
	.pickbtn[aria-pressed='true'] {
		color: var(--accent);
	}
	.pickbtn:focus-visible {
		outline: 1.5px solid var(--accent);
		outline-offset: 1px;
	}
	.pickbtn.lead {
		font-family: var(--serif);
		font-size: 1rem;
		color: var(--ink);
	}
	.r-model a {
		color: var(--ink);
		text-decoration: none;
		border-bottom: 1px solid var(--rule);
	}
	.r-model a:hover {
		border-bottom-color: var(--accent);
	}
	.r-model .reps {
		color: var(--accent);
		font-size: 0.66rem;
		margin-left: 0.4rem;
		cursor: help;
		font-variant-numeric: tabular-nums;
	}
	.r-model .date {
		color: var(--ink-faint);
		font-size: 0.7rem;
		margin-left: 0.5rem;
	}
	.on {
		color: var(--ink);
	}
	.dag {
		color: var(--accent);
	}
	.na {
		color: var(--ink-faint);
	}
	.foot {
		font-family: var(--serif);
		font-size: 0.8rem;
		color: var(--ink-muted);
		max-width: 70ch;
		line-height: 1.45;
		margin-top: 1.1rem;
	}
	.empty {
		font-family: var(--mono);
		color: var(--ink-faint);
	}
</style>
