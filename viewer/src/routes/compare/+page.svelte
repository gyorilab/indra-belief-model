<script lang="ts">
	import { goto } from '$app/navigation';
	import { page } from '$app/stores';
	import type { PageData } from './$types';

	let { data }: { data: PageData } = $props();
	const runs = $derived(data.runs);
	const comparison = $derived(data.comparison);
	const selectedA = $derived(data.selectedA);
	const selectedB = $derived(data.selectedB);

	function shortRun(id: string): string {
		return id.slice(0, 8);
	}

	function fmtScore(s: number | null): string {
		return s == null ? '—' : s.toFixed(3);
	}

	function fmtDelta(d: number | null): string {
		if (d == null) return '—';
		const sign = d > 0 ? '+' : d < 0 ? '−' : '±';
		return `${sign}${Math.abs(d).toFixed(3)}`;
	}

	// A statement is a "doubt direction" disagreement when the two runs land on
	// opposite sides of the 0.5 belief midline (one trusts the evidence, the
	// other doubts it), or when their verdict mix differs notably.
	function sideOf(s: number | null): number | null {
		if (s == null) return null;
		if (s > 0.5) return 1;
		if (s < 0.5) return -1;
		return 0;
	}

	function disagrees(row: PageData['comparison'] extends null ? never : NonNullable<PageData['comparison']>['rows'][number]): boolean {
		const sa = sideOf(row.a.score);
		const sb = sideOf(row.b.score);
		if (sa != null && sb != null && sa !== sb && sa !== 0 && sb !== 0) return true;
		// verdict-mix divergence: different correct/incorrect signature
		return row.a.verdict_mix !== row.b.verdict_mix;
	}

	// Navigate to ?a=&b= when a picker changes, preserving the other side.
	function setRun(which: 'a' | 'b', run_id: string) {
		const params = new URLSearchParams($page.url.searchParams);
		const other = which === 'a' ? (selectedB ?? '') : (selectedA ?? '');
		params.set(which, run_id);
		if (other) params.set(which === 'a' ? 'b' : 'a', other);
		goto(`?${params.toString()}`, { keepFocus: true, noScroll: true });
	}

	const haveTwo = $derived(runs.length >= 2);
	const sameRun = $derived(selectedA != null && selectedA === selectedB);
</script>

<svelte:head>
	<title>INDRA Belief — Compare runs</title>
</svelte:head>

<main id="main">
	<h1>compare two runs</h1>
	<p class="lede">
		Two monolithic scoring passes over the shared corpus, statement by statement.
		Each evidence is one LLM call; the belief here is the mean of those calls.
		Rows are sorted by score divergence — where the two models most disagree about
		whether the literature supports the claim.
	</p>

	{#if !haveTwo}
		<section class="empty">
			<p class="empty-h">need two exported runs to compare</p>
			<p class="hint">
				{runs.length === 0
					? 'no runs are exported yet · score a corpus and export it to populate this view'
					: 'only one run is exported · export a second run to compare them side by side'}
			</p>
			{#if runs.length === 1}
				<p class="hint mono">have: {runs[0].model} · {shortRun(runs[0].run_id)}</p>
			{/if}
		</section>
	{:else}
		<section class="pickers" aria-label="run selectors">
			<div class="picker">
				<label for="run-a">run A</label>
				<select
					id="run-a"
					value={selectedA ?? ''}
					onchange={(e) => setRun('a', (e.currentTarget as HTMLSelectElement).value)}
				>
					{#each runs as r}
						<option value={r.run_id}>
							{r.model} · {shortRun(r.run_id)} · {r.n_statements} stmts
						</option>
					{/each}
				</select>
			</div>
			<span class="vs">vs</span>
			<div class="picker">
				<label for="run-b">run B</label>
				<select
					id="run-b"
					value={selectedB ?? ''}
					onchange={(e) => setRun('b', (e.currentTarget as HTMLSelectElement).value)}
				>
					{#each runs as r}
						<option value={r.run_id}>
							{r.model} · {shortRun(r.run_id)} · {r.n_statements} stmts
						</option>
					{/each}
				</select>
			</div>
		</section>

		{#if sameRun}
			<section class="empty">
				<p class="empty-h">pick two different runs</p>
				<p class="hint">run A and run B are the same export — choose distinct runs to compare</p>
			</section>
		{:else if comparison}
			{@const c = comparison}
			<header class="cmp-head">
				<div class="cmp-vs">
					<span class="cmp-model cmp-a">{c.run_a.model}</span>
					<span class="cmp-id mono">{shortRun(c.run_a.run_id)}</span>
					<span class="cmp-sep">vs</span>
					<span class="cmp-model cmp-b">{c.run_b.model}</span>
					<span class="cmp-id mono">{shortRun(c.run_b.run_id)}</span>
				</div>
				<div class="cmp-shared mono">{c.n_shared} shared statements</div>
			</header>

			{#if c.rows.length === 0}
				<section class="empty">
					<p class="empty-h">no shared statements</p>
					<p class="hint">these two runs scored disjoint statement sets — nothing to compare</p>
				</section>
			{:else}
				<table class="cmp">
					<thead>
						<tr>
							<th class="col-stmt">statement</th>
							<th class="col-num">indra</th>
							<th class="col-score">A score</th>
							<th class="col-score">B score</th>
							<th class="col-num col-delta">Δ A−B</th>
						</tr>
					</thead>
					<tbody>
						{#each c.rows as row}
							<tr class:disagree={disagrees(row)}>
								<td class="col-stmt">
									<a href={`/statements/${row.stmt_hash}?run_id=${c.run_a.run_id}`}>
										<span class="agent">{row.subject}</span>
										<span class="rel">[{row.stmt_type}]</span>
										<span class="agent">{row.object}</span>
									</a>
								</td>
								<td class="col-num mono">{fmtScore(row.indra_belief)}</td>
								<td class="col-score">
									<span class="mono score">{fmtScore(row.a.score)}</span>
									<span class="vmix mono">{row.a.verdict_mix}</span>
								</td>
								<td class="col-score">
									<span class="mono score">{fmtScore(row.b.score)}</span>
									<span class="vmix mono">{row.b.verdict_mix}</span>
								</td>
								<td
									class="col-num col-delta mono"
									class:doubt={(row.score_delta ?? 0) < 0}
									class:trust={(row.score_delta ?? 0) > 0}
								>
									{fmtDelta(row.score_delta)}
								</td>
							</tr>
						{/each}
					</tbody>
				</table>
				<p class="legend mono">
					<span class="legend-swatch disagree-swatch"></span> highlighted rows: the two runs land
					on opposite sides of the 0.5 belief midline, or their verdict mix (✓ correct · ✗ incorrect)
					differs. <span class="legend-accent">Δ A−B</span> in accent = A doubts more than B.
				</p>
			{/if}
		{/if}
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

	main {
		max-width: 1200px;
		margin: 0 auto;
		padding: 2rem 1.5rem 4rem;
		font-family: var(--serif);
		color: var(--ink);
	}

	h1 {
		font-family: var(--serif);
		font-weight: 400;
		font-size: 1.6rem;
		margin: 0 0 0.4rem;
	}

	.lede {
		color: var(--ink-muted);
		max-width: 64ch;
		margin: 0 0 1.8rem;
		line-height: 1.5;
	}

	.mono {
		font-family: var(--mono);
		font-variant-numeric: tabular-nums;
	}

	.hint {
		color: var(--ink-muted);
		font-style: italic;
		font-size: 0.92em;
	}

	.empty {
		margin: 3rem auto;
		max-width: 60ch;
		border-left: 3px solid var(--rule);
		padding: 1rem 1.2rem;
	}

	.empty-h {
		font-family: var(--serif);
		font-size: 1.15rem;
		margin: 0 0 0.4rem;
		color: var(--ink);
	}

	/* ── pickers ─────────────────────────────────────────────── */
	.pickers {
		display: flex;
		flex-wrap: wrap;
		align-items: flex-end;
		gap: 0.8rem 1.2rem;
		margin: 0 0 1.6rem;
		padding-bottom: 1.2rem;
		border-bottom: 1px solid var(--rule);
	}

	.picker {
		display: flex;
		flex-direction: column;
		gap: 0.25rem;
	}

	.picker label {
		font-family: var(--mono);
		font-size: 0.7rem;
		text-transform: lowercase;
		letter-spacing: 0.06em;
		color: var(--ink-faint);
	}

	.picker select {
		font-family: var(--mono);
		font-size: 0.82rem;
		color: var(--ink);
		background: var(--paper);
		border: 1px solid var(--rule);
		padding: 0.4rem 0.5rem;
		min-width: 22rem;
		max-width: 100%;
	}

	.picker select:focus {
		outline: 2px solid var(--accent);
		outline-offset: 1px;
	}

	.vs {
		font-family: var(--mono);
		font-size: 0.78rem;
		color: var(--ink-faint);
		padding-bottom: 0.5rem;
	}

	/* ── comparison header ───────────────────────────────────── */
	.cmp-head {
		display: flex;
		flex-wrap: wrap;
		justify-content: space-between;
		align-items: baseline;
		gap: 0.6rem 1.2rem;
		margin: 0 0 1rem;
	}

	.cmp-vs {
		display: flex;
		flex-wrap: wrap;
		align-items: baseline;
		gap: 0.4rem 0.6rem;
		font-family: var(--serif);
	}

	.cmp-model {
		font-size: 1.2rem;
		color: var(--ink);
	}

	.cmp-id {
		font-size: 0.72rem;
		color: var(--ink-faint);
	}

	.cmp-sep {
		color: var(--ink-faint);
		font-size: 0.95rem;
		font-style: italic;
	}

	.cmp-shared {
		font-size: 0.78rem;
		color: var(--ink-muted);
	}

	/* ── table ───────────────────────────────────────────────── */
	table.cmp {
		width: 100%;
		border-collapse: collapse;
		font-size: 0.84rem;
	}

	table.cmp thead th {
		font-family: var(--mono);
		font-size: 0.68rem;
		font-weight: 500;
		text-transform: lowercase;
		letter-spacing: 0.04em;
		color: var(--ink-muted);
		text-align: left;
		padding: 0.4rem 0.6rem;
		border-bottom: 1px solid var(--ink);
		white-space: nowrap;
	}

	table.cmp tbody td {
		padding: 0.5rem 0.6rem;
		border-bottom: 1px dotted var(--rule);
		vertical-align: baseline;
	}

	.col-num {
		text-align: right;
		white-space: nowrap;
	}

	.col-score {
		white-space: nowrap;
	}

	.col-delta {
		font-weight: 500;
	}

	tr.disagree {
		background: var(--accent-wash);
	}

	tr.disagree td.col-stmt {
		box-shadow: inset 2px 0 0 var(--accent);
	}

	.col-stmt a {
		color: var(--ink);
		text-decoration: none;
		line-height: 1.4;
	}

	.col-stmt a:hover {
		color: var(--accent);
		text-decoration: underline;
	}

	.agent {
		font-weight: 500;
	}

	.rel {
		font-family: var(--mono);
		font-size: 0.74rem;
		color: var(--ink-muted);
	}

	.score {
		color: var(--ink);
	}

	.vmix {
		display: inline-block;
		margin-left: 0.4rem;
		font-size: 0.72rem;
		color: var(--ink-faint);
	}

	/* doubt direction in accent: A scored lower than B (negative Δ = more doubt) */
	td.doubt {
		color: var(--accent);
	}

	td.trust {
		color: var(--ink);
	}

	/* ── legend ──────────────────────────────────────────────── */
	.legend {
		margin: 1rem 0 0;
		font-size: 0.72rem;
		color: var(--ink-muted);
		line-height: 1.6;
		max-width: 70ch;
	}

	.legend-swatch {
		display: inline-block;
		width: 0.8rem;
		height: 0.8rem;
		vertical-align: -1px;
		border-left: 2px solid var(--accent);
		background: var(--accent-wash);
	}

	.legend-accent {
		color: var(--accent);
	}

	@media (max-width: 700px) {
		.picker select {
			min-width: 0;
			width: 100%;
		}
		table.cmp {
			font-size: 0.78rem;
		}
		.vmix {
			display: block;
			margin-left: 0;
		}
	}
</style>
