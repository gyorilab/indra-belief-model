<script lang="ts">
	import type { PageData } from './$types';
	import { shortHash } from '$lib/format';

	let { data }: { data: PageData } = $props();
	const d = $derived(data.detail);

	function archMark(architecture: string | null | undefined): string {
		if (architecture === 'monolithic') return '[M]';
		if (architecture === 'decomposed') return '[D]';
		return '[?]';
	}

	function statusLabel(status: string): string {
		if (status === 'no_aggregate_verdict') return 'no verdict';
		if (status === 'context_mismatch') return 'context mismatch';
		if (status === 'not_scored_in_run') return 'not scored';
		if (status === 'not_in_corpus') return 'not in corpus';
		return status;
	}

	function statusReason(status: string): string {
		if (status === 'compared') return 'aggregate verdict row matched this label target and context';
		if (status === 'no_aggregate_verdict') return 'aggregate row exists, but it has no verdict to compare';
		if (status === 'context_mismatch') return 'same evidence was scored under another statement context';
		if (status === 'not_scored_in_run') return 'label target exists in the corpus but was not scored in this run';
		if (status === 'not_in_corpus') return 'label target is not present in the local evidence table';
		return 'unclassified overlap state';
	}

	function notDefinedReason(): string {
		if (d.metric_step_kind !== 'aggregate' && d.n_metric_compared_rows > 0) return 'cohort drilldown is only defined for aggregate evidence rows';
		if (d.n_scored_evidences === 0) return `no scored ${d.metric_step_kind} evidence rows with verdicts in this run`;
		if (d.n_metric_compared_rows === 0) return `no scored ${d.metric_step_kind} evidence rows overlap this truth_set`;
		return '';
	}

	const statusRows = $derived([
		{ key: 'compared', label: 'compared', n: d.n_compared_labels },
		{ key: 'no_aggregate_verdict', label: 'no verdict', n: d.n_no_aggregate_verdict },
		{ key: 'context_mismatch', label: 'context mismatch', n: d.n_context_mismatch },
		{ key: 'not_scored_in_run', label: 'not scored', n: d.n_not_scored_in_run },
		{ key: 'not_in_corpus', label: 'not in corpus', n: d.n_not_in_corpus }
	].filter((row) => row.n > 0 || row.key === 'compared'));
</script>

<svelte:head><title>{d.truth_set_id} overlap - INDRA Belief</title></svelte:head>

<header>
	<div class="crumb">
		<a href="/">corpus</a><span class="sep"> / </span><a href={`/runs/${d.run_id}`}>run {shortHash(d.run_id)}</a><span class="sep"> / </span><strong>{d.truth_set_id}</strong>
	</div>
	<div class="meta">
		<span class={`run-arch run-arch-${d.architecture}`}>{archMark(d.architecture)}</span>
		<span>{d.architecture}</span>
	</div>
</header>

<main id="main">
	<section class="overlap-head">
		<h1>truth-set overlap</h1>
		<p class="lead">
			<strong>{d.truth_set_name}</strong> against run <code>{shortHash(d.run_id)}</code>, using {d.metric_step_kind} verdicts.
		</p>
		<dl class="facts">
			<div><dt>gold labels</dt><dd>{d.n_gold_labels.toLocaleString()}</dd></div>
			<div><dt>applicable labels</dt><dd>{d.n_applicable_gold_labels.toLocaleString()}</dd></div>
			<div><dt>metric rows</dt><dd>{d.n_metric_compared_rows.toLocaleString()}</dd></div>
			<div><dt>scored evidence</dt><dd>{d.n_scored_evidences.toLocaleString()}</dd></div>
			<div><dt>label fields</dt><dd>{d.gold_fields.length > 0 ? d.gold_fields.join('+') : 'none'}</dd></div>
			<div><dt>positive class</dt><dd>{d.positive_gold_label}</dd></div>
		</dl>
		{#if d.cohort_href}
			<p class="action-line"><a href={d.cohort_href}>open compared cohort</a></p>
		{:else}
			<p class="not-defined">not defined for this run because {notDefinedReason()}.</p>
		{/if}
	</section>

	<section class="status-rail" aria-label="truth-set overlap states">
		{#each statusRows as row}
			<div class="status-cell status-{row.key}">
				<span>{row.label}</span>
				<strong>{row.n.toLocaleString()}</strong>
			</div>
		{/each}
	</section>

	<section class="legend">
		{#each statusRows as row}
			<p><strong>{row.label}</strong>: {statusReason(row.key)}</p>
		{/each}
		<p><strong>metric rows</strong>: scorer verdict rows after contextual gold selection; this can differ from label count when generic evidence labels match multiple scored statement contexts.</p>
	</section>

	<section class="labels">
		<div class="labels-head">
			<h2>gold labels</h2>
			{#if d.n_gold_labels > d.row_limit}
				<span>showing up to {d.status_row_limit.toLocaleString()} per state, {d.row_limit.toLocaleString()} max</span>
			{:else if d.n_gold_labels > d.rows.length}
				<span>showing up to {d.status_row_limit.toLocaleString()} per state</span>
			{/if}
		</div>
		{#if d.rows.length === 0}
			<p class="empty">this truth_set has no evidence-level verdict/tag labels.</p>
		{:else}
			<table>
				<thead>
					<tr>
						<th>state</th>
						<th>gold</th>
						<th>target</th>
						<th>context</th>
						<th>scorer</th>
						<th>evidence</th>
					</tr>
				</thead>
				<tbody>
					{#each d.rows as row}
						<tr>
							<td><span class="state state-{row.status}">{statusLabel(row.status)}</span></td>
							<td>
								<code>{row.field}</code>
								<span>{row.value_text ?? '-'}</span>
								{#if row.provenance}<small>{row.provenance}</small>{/if}
							</td>
							<td><code>{shortHash(row.target_id)}</code></td>
							<td>
								{#if row.relation_target_id}
									<code>{shortHash(row.relation_target_id)}</code>
								{:else}
									<span class="muted">any statement</span>
								{/if}
							</td>
							<td>
								{#if row.statement_href}
									<a href={row.statement_href}>{row.sample_matched_stmt_hash ? shortHash(row.sample_matched_stmt_hash) : shortHash(row.corpus_stmt_hash ?? '')}</a>
								{:else if row.sample_scored_stmt_hash}
									<code>{shortHash(row.sample_scored_stmt_hash)}</code>
								{:else}
									<span class="muted">none</span>
								{/if}
								<small>
									{row.sample_our_verdict ?? 'no verdict'}; matched {row.matched_verdict_rows}/{row.same_evidence_scored_rows}
								</small>
							</td>
							<td>
								<span class="source">{row.source_api ?? '-'}</span>
								<p>{row.evidence_text ?? '(label target not in evidence table)'}</p>
							</td>
						</tr>
					{/each}
				</tbody>
			</table>
		{/if}
	</section>
</main>

<style>
	:global(:root) {
		--ink: #1a1a1a;
		--ink-muted: #6a6a6a;
		--ink-faint: #727272;
		--paper: #fdfcf8;
		--rule: #e6e2d6;
		--accent: #7d2a1a;
		--accent-wash: rgba(125, 42, 26, 0.05);
		--ok-green: #2a6f2a;
		--warn: #8a6a00;
		--mono: ui-monospace, 'SF Mono', 'JetBrains Mono', Menlo, monospace;
		--serif: 'Iowan Old Style', 'Source Serif Pro', Georgia, serif;
	}
	:global(html, body) {
		background: var(--paper);
		color: var(--ink);
		font-family: var(--serif);
		font-size: 16px;
		line-height: 1.5;
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
	.action-line a,
	td a {
		color: var(--accent);
		text-decoration: none;
	}
	.crumb a:hover,
	.action-line a:hover,
	td a:hover {
		text-decoration: underline;
	}
	.meta {
		display: flex;
		gap: 0.45rem;
		align-items: baseline;
	}
	main {
		max-width: 1180px;
		margin: 0 auto;
		padding: 1.5rem;
	}
	.overlap-head {
		border-bottom: 1px solid var(--rule);
		padding-bottom: 1rem;
	}
	h1,
	h2 {
		font-family: var(--serif);
		font-weight: 400;
		margin: 0;
	}
	h1 {
		font-size: 1.8rem;
	}
	h2 {
		font-size: 1.1rem;
	}
	.lead {
		margin: 0.4rem 0 1rem;
		color: var(--ink-muted);
	}
	code {
		font-family: var(--mono);
		font-size: 0.86em;
	}
	.facts {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
		gap: 0.65rem 1rem;
		font-family: var(--mono);
		margin: 0;
	}
	.facts div {
		border-top: 1px solid var(--rule);
		padding-top: 0.35rem;
	}
	dt {
		color: var(--ink-muted);
		font-size: 0.72rem;
		text-transform: lowercase;
	}
	dd {
		margin: 0;
		font-size: 0.95rem;
	}
	.action-line,
	.not-defined {
		margin: 1rem 0 0;
		font-family: var(--mono);
		font-size: 0.82rem;
	}
	.not-defined {
		color: var(--warn);
	}
	.status-rail {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
		gap: 1px;
		margin: 1rem 0;
		background: var(--rule);
		border: 1px solid var(--rule);
		font-family: var(--mono);
	}
	.status-cell {
		background: var(--paper);
		padding: 0.55rem 0.7rem;
		display: flex;
		justify-content: space-between;
		gap: 0.75rem;
	}
	.status-cell span {
		color: var(--ink-muted);
	}
	.status-compared strong { color: var(--ok-green); }
	.status-no_aggregate_verdict strong,
	.status-context_mismatch strong { color: var(--warn); }
	.status-not_scored_in_run strong,
	.status-not_in_corpus strong { color: var(--accent); }
	.legend {
		margin: 0 0 1.2rem;
		font-family: var(--mono);
		font-size: 0.78rem;
		color: var(--ink-muted);
	}
	.legend p {
		margin: 0.15rem 0;
	}
	.labels {
		border-top: 1px solid var(--rule);
		padding-top: 1rem;
	}
	.labels-head {
		display: flex;
		justify-content: space-between;
		gap: 1rem;
		align-items: baseline;
		margin-bottom: 0.7rem;
	}
	.labels-head span,
	.muted {
		color: var(--ink-muted);
	}
	table {
		width: 100%;
		border-collapse: collapse;
		font-family: var(--mono);
		font-size: 0.78rem;
	}
	th {
		text-align: left;
		color: var(--ink-muted);
		font-weight: 400;
		border-bottom: 1px solid var(--rule);
		padding: 0.4rem 0.5rem;
	}
	td {
		vertical-align: top;
		border-bottom: 1px dotted var(--rule);
		padding: 0.55rem 0.5rem;
	}
	td p {
		margin: 0.2rem 0 0;
		font-family: var(--serif);
		font-size: 0.88rem;
		line-height: 1.35;
		max-width: 34rem;
	}
	td small {
		display: block;
		color: var(--ink-muted);
		font-size: 0.7rem;
		margin-top: 0.15rem;
	}
	.state {
		white-space: nowrap;
	}
	.state-compared { color: var(--ok-green); }
	.state-no_aggregate_verdict,
	.state-context_mismatch { color: var(--warn); }
	.state-not_scored_in_run,
	.state-not_in_corpus { color: var(--accent); }
	.source {
		color: var(--ink-muted);
	}
	.empty {
		color: var(--ink-muted);
		font-family: var(--mono);
		font-size: 0.82rem;
	}
	.run-arch {
		font-weight: 600;
	}
	@media (max-width: 760px) {
		header {
			align-items: flex-start;
			flex-direction: column;
			gap: 0.35rem;
		}
		main {
			padding: 1rem;
		}
		table {
			display: block;
			overflow-x: auto;
			white-space: nowrap;
		}
		td p {
			white-space: normal;
			min-width: 18rem;
		}
	}
</style>
