<script lang="ts">
	import type { PageData } from './$types';

	let { data }: { data: PageData } = $props();
	const runs = $derived(data.runs);

	function statusGlyph(status: string | null): string {
		if (status === 'completed' || status === 'succeeded') return '✓';
		if (status === 'running') return '↻';
		if (status === 'failed') return '✗';
		if (status === 'canceled' || status === 'stopped') return '!';
		return '·';
	}

	function fmtCount(n: number): string {
		return n.toLocaleString('en-US');
	}
</script>

<svelte:head>
	<title>runs · INDRA belief viewer</title>
</svelte:head>

<header>
	<div class="crumb">
		<strong>runs</strong>
	</div>
	<div class="meta">
		<span class="count">{runs.length}</span>
	</div>
</header>

<main id="main">
	<h1 class="visually-hidden">All scoring runs</h1>

	{#if runs.length === 0}
		<section class="empty">
			<h1>no runs yet</h1>
			<p>
				No export directories with an <code>export_meta.json</code> were found under
				<code>data/exports/</code>. Visit <a href="/">overview</a> for the export pipeline.
			</p>
		</section>
	{:else}
		<ul class="run-feed">
			{#each runs as r}
				<li
					class="run-row"
					class:run-row-failed={r.status === 'failed'}
					class:run-row-running={r.status === 'running'}
				>
					<span class="run-glyph" title={r.status ?? 'unknown'}>{statusGlyph(r.status)}</span>
					<a class="run-hash" href={`/runs/${r.run_id}`} title={r.run_id}>
						<code>{r.run_id.slice(0, 8)}</code>
					</a>
					<span class="run-model" title={r.model}>{r.model}</span>
					<span class="run-when" title={r.generated_date ?? ''}>{r.generated_date ?? '—'}</span>
					<span class="run-narrative">
						<span class="run-n">{fmtCount(r.n_statements)} stmts</span>
						<span class="muted">·</span>
						<span class="run-n">{fmtCount(r.n_evidences)} evidence</span>
						{#if r.status && r.status !== 'completed' && r.status !== 'succeeded'}
							<span class="muted">·</span>
							<span class="run-status muted">{r.status}</span>
						{/if}
					</span>
				</li>
			{/each}
		</ul>
	{/if}
</main>

<style>
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
	.meta {
		display: flex;
		gap: 1.2rem;
		align-items: baseline;
	}
	.count {
		color: var(--ink-faint);
		font-variant-numeric: tabular-nums;
	}

	main {
		max-width: 1100px;
		margin: 0 auto;
		padding: 1rem 1.5rem 3rem;
	}

	.visually-hidden {
		position: absolute;
		width: 1px;
		height: 1px;
		padding: 0;
		margin: -1px;
		overflow: hidden;
		clip: rect(0, 0, 0, 0);
		white-space: nowrap;
		border: 0;
	}

	.empty {
		padding: 2rem 0;
		color: var(--ink-muted);
	}
	.empty h1 {
		font-family: var(--serif);
		font-size: 1.4rem;
		font-weight: 500;
		margin: 0 0 0.5rem;
		color: var(--ink);
	}
	.empty code {
		font-family: var(--mono);
		font-size: 0.86em;
		color: var(--ink-muted);
	}
	.empty a {
		color: var(--accent);
	}

	.muted {
		color: var(--ink-faint);
	}

	.run-feed {
		list-style: none;
		padding: 0;
		margin: 0;
		font-family: var(--mono);
		font-size: 0.8rem;
	}
	.run-row {
		display: grid;
		grid-template-columns: 1.4ch 9ch minmax(0, max-content) minmax(0, max-content) minmax(0, 1fr);
		gap: 0.6rem;
		align-items: baseline;
		padding: 0.3rem 0;
		border-bottom: 1px dotted var(--rule);
	}
	.run-row:last-child {
		border-bottom: none;
	}
	.run-row-failed {
		color: var(--accent);
	}
	.run-row-running {
		color: var(--ink);
	}
	.run-model,
	.run-when {
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}
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
	.run-model {
		color: var(--accent);
	}
	.run-when {
		color: var(--ink-faint);
	}
	.run-narrative {
		color: var(--ink);
		font-variant-numeric: tabular-nums;
		min-width: 0;
	}
	.run-n {
		color: var(--ink);
	}

	@media (max-width: 720px) {
		.run-row {
			grid-template-columns: 1.4ch 9ch 1fr;
		}
		.run-model,
		.run-when {
			display: none;
		}
	}
</style>
