<script lang="ts">
	import type { PageData } from './$types';
	import { invalidateAll } from '$app/navigation';
	import { onMount, onDestroy } from 'svelte';
	import BeliefPrimitive from '$lib/components/BeliefPrimitive.svelte';
	import Validity from '$lib/components/Validity.svelte';

	let { data }: { data: PageData } = $props();
	const o = $derived(data.overview);
	const focus = $derived(data.focus);
	const findings = $derived(data.findings);
	const residuals = $derived(data.residuals);
	const validity = $derived(data.validity);

	function statusGlyph(status: string | null): string {
		if (status === 'succeeded') return '✓';
		if (status === 'running') return '↻';
		if (status === 'failed') return '✗';
		if (status === 'canceled') return '!';
		return '·';
	}

	function fmt(n: number): string {
		return n.toLocaleString('en-US');
	}

	function fmtWhen(d: string | null): string {
		if (!d) return '—';
		return d.replace(/\.\d+$/, '').replace(/^(\d{4}-\d{2}-\d{2})[ T]/, '$1·');
	}

	// Phase 5d minimum-viable live-tail: poll the load function on a fixed cadence
	// (3s) so the dashboard refreshes counts / latest run / validity without manual
	// reload. The live dot flashes only when data changed since the last poll.
	let pollHandle: ReturnType<typeof setInterval> | null = null;
	let freshTimer: ReturnType<typeof setTimeout> | null = null;
	let lastSignature = $state<string>('');
	let dotFresh = $state(false);

	const currentSignature = $derived(
		`${o.statementCount}|${o.evidenceCount}|${o.runs.length}|${o.runs[0]?.run_id ?? ''}|${validity?.run_id ?? ''}`
	);

	// Flash the dot for 800ms when the data signature changes, then revert.
	$effect(() => {
		if (lastSignature && lastSignature !== currentSignature) {
			dotFresh = true;
			if (freshTimer) clearTimeout(freshTimer);
			freshTimer = setTimeout(() => {
				dotFresh = false;
			}, 800);
		}
		lastSignature = currentSignature;
	});

	// Empty-state pipeline snippet — held in a const so f-string `${...}`
	// interpolation doesn't conflict with Svelte's `{...}` template syntax.
	const PIPELINE_SNIPPET = `# Score a corpus through the monolithic pipeline, then export JSONL:
python scripts/run_rasmachine_monolithic.py \\
    --model gemma --out data/exports/<run_id>

# Each export dir is self-contained:
#   per_statement.json   — one rollup per statement
#   per_evidence.jsonl   — one row per (statement, evidence): score + reasoning
#   export_meta.json     — run metadata + bucket counts
# The viewer reads these directly — no database.`;

	onMount(() => {
		lastSignature = currentSignature;
		// Refresh the dashboard every 3s so CLI-produced runs show up live.
		pollHandle = setInterval(() => {
			invalidateAll();
		}, 3000);
	});

	onDestroy(() => {
		if (pollHandle) clearInterval(pollHandle);
		if (freshTimer) clearTimeout(freshTimer);
	});
</script>

<svelte:head>
	<title>INDRA Belief — Corpus</title>
</svelte:head>

<header>
	<div class="crumb"></div>
	<div class="meta">
		<span class="live-indicator" title="dashboard polls every 3s; dot flashes when data changes">
			<span class="live-dot" class:live-dot-flash={dotFresh}></span>
			{dotFresh ? 'fresh' : 'live'}
		</span>
	</div>
</header>

<main id="main">
	<h1 class="visually-hidden">Corpus dashboard</h1>
	{#if o.runs.length === 0}
		<section class="empty">
			<h1>no runs loaded</h1>
			<p class="lede">
				The viewer reads monolithic JSONL exports from <code>data/exports/</code>, but none
				were found there yet.
			</p>
			<p>Score a corpus and export it:</p>
			<pre>{PIPELINE_SNIPPET}</pre>
		</section>
	{:else}
		<p class="dashboard-subtitle">
			INDRA Statement belief rescorer. Below: the statement that disagreed most with
			INDRA's prior in the latest run, what changed, and where we are weakest.
		</p>

		<section class="focus" aria-label="focus statement — biggest disagreement with INDRA in the latest run">
			{#if focus}
				<BeliefPrimitive
					stmt={focus.stmt}
					our_score={focus.our_score}
					indra_score={focus.indra_score}
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
					<p class="hint">no belief in focus yet · score a corpus to populate this view</p>
				</div>
			{/if}
		</section>

		{#if findings && findings.biggest_disagreement.length > 0}
			{@const focusHash = focus?.stmt.stmt_hash ?? null}
			{@const rows = findings.biggest_disagreement.filter((r) => r.stmt_hash !== focusHash)}
			{#if rows.length > 0}
				<section class="findings" aria-label="other notable statements from this run">
					<h2 class="visually-hidden">findings — other notable statements from this run</h2>
					<div class="lane">
						<h3 class="lane-h">we disagree most with INDRA on these <span class="lane-n">({rows.length})</span></h3>
						<div class="lane-body">
							{#each rows as r}
								<BeliefPrimitive
									mode="compact"
									stmt={{ stmt_hash: r.stmt_hash, indra_type: r.stmt_type, subject: r.subject, object: r.object }}
									our_score={r.our_score}
									indra_score={r.indra_belief}
									href={`/?focus=${r.stmt_hash}`}
								/>
							{/each}
						</div>
					</div>
				</section>
			{/if}
		{/if}

		{#if validity}
			<Validity v={validity} {residuals} />
		{/if}

		<section class="grid">
			<article class="run-feed-article">
				<h2>runs</h2>
				{#if o.runs.length === 0}
					<p class="hint">no runs yet</p>
				{:else}
					<ul class="run-feed">
						{#each o.runs as r}
							<li class="run-row" class:run-row-failed={r.status === 'failed'} class:run-row-running={r.status === 'running'} class:run-row-canceled={r.status === 'canceled'}>
								<span class="run-glyph" class:status-failed={r.status === 'failed'} class:status-running={r.status === 'running'} class:status-canceled={r.status === 'canceled'} title={r.status ?? 'unknown'}>{statusGlyph(r.status)}</span>
								<a class="run-hash" href={`/runs/${r.run_id}`} title={r.run_id}><code>{r.run_id.slice(0, 8)}</code></a>
								<span class="run-model" title={r.model}>{r.model}</span>
								<span class="run-when" title={r.generated_date ?? ''}>{fmtWhen(r.generated_date)}</span>
								<span class="run-detail">
									{#if r.status && r.status !== 'succeeded'}
										<span class="run-status-tag">{r.status}</span><span class="run-sep">·</span>
									{/if}
									<span class="run-n">{fmt(r.n_statements)} stmts</span>
									<span class="run-sep">·</span>
									<span class="run-n">{fmt(r.n_evidences)} ev</span>
								</span>
							</li>
						{/each}
					</ul>
				{/if}
			</article>
		</section>

		<footer class="data-footer">
			<div class="df-line">
				<span class="df-count">{fmt(o.statementCount)} stmts</span>
				<span class="df-sep">·</span>
				<span class="df-count">{fmt(o.evidenceCount)} ev</span>
				{#if o.latest}
					<span class="df-sep">·</span>
					<span class="df-label">latest</span>
					<span class="df-item"><code>{o.latest.run_id.slice(0, 8)}</code> {o.latest.model}</span>
				{/if}
				<span class="df-sep">·</span>
				<span class="df-count">{fmt(o.runs.length)} run{o.runs.length === 1 ? '' : 's'}</span>
			</div>
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

	.meta {
		display: flex;
		gap: 1.2rem;
		align-items: baseline;
		min-width: 0;
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
		grid-template-columns: 1.4ch 9ch minmax(6rem, 14rem) auto 1fr;
		column-gap: 0.65rem;
		row-gap: 0.12rem;
		align-items: baseline;
		padding: 0.45rem 0;
		border-bottom: 1px dotted var(--rule);
	}
	.run-glyph { grid-column: 1; grid-row: 1; font-variant-numeric: tabular-nums; text-align: center; }
	.run-hash { grid-column: 2; grid-row: 1; }
	.run-model { grid-column: 3; grid-row: 1; }
	.run-when { grid-column: 4; grid-row: 1; }
	/* Detail line drops to row 2, indented under the hash, full width. */
	.run-detail {
		grid-column: 2 / -1;
		grid-row: 2;
		color: var(--ink-muted);
		font-size: 0.74rem;
		font-variant-numeric: tabular-nums;
		line-height: 1.4;
		min-width: 0;
	}
	.run-model, .run-when {
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
	.run-hash {
		color: var(--ink);
		text-decoration: none;
	}
	.run-hash:hover {
		color: var(--accent);
		text-decoration: underline;
	}
	.run-model {
		color: var(--ink-muted);
	}
	.run-when {
		color: var(--ink-faint);
	}
	.run-sep {
		color: var(--ink-faint);
		margin: 0 0.35rem;
	}
	.run-n {
		color: var(--ink-muted);
	}
	.run-status-tag {
		color: var(--accent);
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
			display: flex;
			flex-wrap: wrap;
			align-items: baseline;
			gap: 0.2rem 0.5rem;
			padding: 0.5rem 0;
		}
		.run-glyph,
		.run-hash,
		.run-model,
		.run-when,
		.run-detail {
			grid-column: auto;
			grid-row: auto;
		}
		.run-model,
		.run-when {
			white-space: normal;
			overflow-wrap: anywhere;
		}
		.run-detail {
			flex-basis: 100%;
		}
	}
</style>
