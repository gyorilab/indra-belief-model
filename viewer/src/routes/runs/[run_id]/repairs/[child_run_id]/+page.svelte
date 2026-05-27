<script lang="ts">
	import type { PageData } from './$types';
	import { fmtBelief, shortHash } from '$lib/format';

	let { data }: { data: PageData } = $props();
	const d = $derived(data.detail);
	const pageStart = $derived(
		d.candidate_lane_total === 0
			? 0
			: Math.min(d.candidate_lane_offset + 1, d.candidate_lane_total)
	);
	const pageEnd = $derived(
		Math.min(d.candidate_lane_offset + d.candidate_lanes.length, d.candidate_lane_total)
	);
	const previousOffset = $derived(Math.max(0, d.candidate_lane_offset - d.candidate_lane_limit));
	const nextOffset = $derived(d.candidate_lane_offset + d.candidate_lanes.length);
	const hasPrevious = $derived(d.candidate_lane_offset > 0);
	const hasNext = $derived(nextOffset < d.candidate_lane_total);

	function pageHref(offset: number): string {
		const params = new URLSearchParams();
		if (offset > 0) params.set('offset', String(offset));
		if (d.candidate_lane_limit !== 100) params.set('limit', String(d.candidate_lane_limit));
		const query = params.toString();
		return query
			? `/runs/${d.parent_run_id}/repairs/${d.child_run_id}?${query}`
			: `/runs/${d.parent_run_id}/repairs/${d.child_run_id}`;
	}

	function fmtCost(c: number | null): string {
		if (c == null) return '-';
		if (c < 0.01) return '<$0.01';
		if (c < 100) return '$' + c.toFixed(2);
		return '$' + c.toFixed(0);
	}

	function fmtMetric(n: number | null): string {
		return n == null || Number.isNaN(n) ? '-' : n.toFixed(3);
	}

	function fmtTimestamp(value: string | null | undefined): string {
		return value ? value.replace(/\.\d+$/, '') : 'unknown start';
	}

	function fmtSigned(n: number | null): string {
		if (n == null || Number.isNaN(n)) return '-';
		const sign = n >= 0 ? '+' : '-';
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
</script>

<svelte:head>
	<title>repair comparison {shortHash(d.parent_run_id)} / {shortHash(d.child_run_id)}</title>
</svelte:head>

<header>
	<div class="crumb">
		<a href="/">runs</a>
		<span>/</span>
		<a href={`/runs/${d.parent_run_id}`}>{shortHash(d.parent_run_id)}</a>
		<span>/</span>
		<a href={`/runs/${d.parent_run_id}/repairs`}>repairs</a>
		<span>/</span>
		<strong>{shortHash(d.child_run_id)}</strong>
	</div>
	<div class="status-line">
		<span class={`run-arch run-arch-${d.architecture}`}>{d.architecture}</span>
		<span>{d.status}</span>
	</div>
</header>

<main>
	<section class="summary" aria-label="full repair comparison summary">
		<div>
			<p class="eyebrow">repair child run</p>
			<h1>full repair comparison</h1>
			<p class="muted">
				parent <a href={`/runs/${d.parent_run_id}`}><code>{shortHash(d.parent_run_id)}</code></a>
				to child <a href={`/runs/${d.child_run_id}`}><code>{shortHash(d.child_run_id)}</code></a>
			</p>
			<p class="muted">{fmtTimestamp(d.started_at)} - {d.model_id_default ?? 'unknown model'} - actual {fmtCost(d.cost_actual_usd)}</p>
		</div>
		{#if d.not_defined_reason}
			<p class="repair-note">not defined: {d.not_defined_reason}</p>
		{/if}
		<dl>
			<div><dt>architectures</dt><dd>{d.parent_architecture} to {d.architecture}</dd></div>
			<div><dt>overlap</dt><dd>{d.n_overlap_evidences.toLocaleString()} ev</dd></div>
			<div><dt>score rows</dt><dd>{d.n_score_evidences.toLocaleString()} ev</dd></div>
			<div><dt>MAE</dt><dd>{fmtMetric(d.parent_mae)} to {fmtMetric(d.child_mae)}</dd></div>
			<div><dt>bias</dt><dd>{fmtSigned(d.parent_bias)} to {fmtSigned(d.child_bias)}</dd></div>
			<div><dt>verdict rows</dt><dd>{d.n_verdict_evidences.toLocaleString()} ev</dd></div>
			<div><dt>verdict movement</dt><dd>{d.verdicts_moved_total.toLocaleString()} moved - {d.verdicts_moved_to_correct.toLocaleString()} to correct - {d.verdicts_moved_to_incorrect.toLocaleString()} to incorrect</dd></div>
			<div><dt>repair candidates</dt><dd>{d.n_child_covered_candidates.toLocaleString()} / {d.n_candidate_evidences.toLocaleString()} covered{#if d.n_new_child_aggregate_candidates > 0} - {d.n_new_child_aggregate_candidates.toLocaleString()} new aggregate{/if}</dd></div>
		</dl>
	</section>

	<section class="lanes" aria-label="full repair candidate before after lanes">
		<div class="section-head">
			<h2>candidate lanes</h2>
			<p class="muted">
				showing {pageStart.toLocaleString()}-{pageEnd.toLocaleString()} of {d.candidate_lane_total.toLocaleString()} repair candidates
			</p>
		</div>

		{#if d.candidate_lanes.length === 0}
			<p class="hint">no repair candidate lanes are visible for this page.</p>
		{:else}
			<div class="table-wrap">
				<table>
					<thead>
						<tr>
							<th>movement</th>
							<th>candidate</th>
							<th>statement</th>
							<th>parent aggregate</th>
							<th>child aggregate</th>
							<th>error delta</th>
						</tr>
					</thead>
					<tbody>
						{#each d.candidate_lanes as lane}
							<tr>
								<td><span class={`movement movement-${lane.movement}`}>{movementLabel(lane.movement)}</span></td>
								<td>
									<code>#{lane.correction_id}</code>
									{#if lane.suspected_step_kind}<div class="muted">{lane.suspected_step_kind}</div>{/if}
								</td>
								<td>
									<a href={`/statements/${lane.stmt_hash}?run_id=${d.parent_run_id}`}><code>{shortHash(lane.stmt_hash)}</code></a>
									{#if lane.evidence_hash}<span class="muted"> / <code>{shortHash(lane.evidence_hash)}</code></span>{/if}
									{#if lane.indra_type}<div>{lane.indra_type}</div>{/if}
									{#if lane.source_api}<div class="muted">{lane.source_api}</div>{/if}
								</td>
								<td>
									<div>{lane.parent_verdict ?? 'no aggregate'}</div>
									<div class="muted">score {fmtBelief(lane.parent_score)} - abs {fmtMetric(lane.parent_abs_error)}</div>
								</td>
								<td>
									<div>{lane.child_verdict ?? 'no aggregate'}</div>
									<div class="muted">score {fmtBelief(lane.child_score)} - abs {fmtMetric(lane.child_abs_error)}</div>
								</td>
								<td>
									<div>{fmtSigned(lane.abs_error_delta)}</div>
									<div class="muted">truth {fmtBelief(lane.indra_belief)}</div>
								</td>
							</tr>
						{/each}
					</tbody>
				</table>
			</div>
		{/if}

		<nav class="pager" aria-label="repair comparison pagination">
			{#if hasPrevious}
				<a href={pageHref(previousOffset)}>previous</a>
			{:else}
				<span>previous</span>
			{/if}
			{#if hasNext}
				<a href={pageHref(nextOffset)}>next</a>
			{:else}
				<span>next</span>
			{/if}
		</nav>
	</section>
</main>

<style>
	:global(:root) {
		--ink: #1a1a1a;
		--ink-muted: #6a6a6a;
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
		align-items: baseline;
		border-bottom: 1px solid var(--rule);
		color: var(--ink-muted);
		display: flex;
		font-family: var(--mono);
		font-size: 0.78rem;
		justify-content: space-between;
		gap: 1rem;
		padding: 0.6rem 1.5rem;
	}
	a {
		color: var(--accent);
		text-decoration: none;
	}
	a:hover {
		text-decoration: underline;
	}
	.crumb,
	.status-line {
		display: flex;
		flex-wrap: wrap;
		gap: 0.4rem;
	}
	.crumb strong,
	.run-arch {
		color: var(--ink);
		font-weight: 500;
	}
	main {
		margin: 0 auto;
		max-width: 1180px;
		padding: 1.5rem;
	}
	.summary {
		border-bottom: 1px solid var(--rule);
		padding-bottom: 1rem;
	}
	.eyebrow,
	.muted,
	.repair-note,
	.hint {
		color: var(--ink-muted);
	}
	.eyebrow {
		font-family: var(--mono);
		font-size: 0.72rem;
		letter-spacing: 0;
		margin: 0 0 0.2rem;
		text-transform: uppercase;
	}
	h1 {
		font-size: 2rem;
		font-weight: 500;
		line-height: 1.1;
		margin: 0 0 0.3rem;
	}
	h2 {
		font-size: 1rem;
		font-weight: 600;
		margin: 0;
	}
	p {
		margin: 0.25rem 0;
	}
	dl {
		display: grid;
		font-family: var(--mono);
		font-size: 0.78rem;
		gap: 0.55rem 1rem;
		grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
		margin: 1rem 0 0;
	}
	dt {
		color: var(--ink-muted);
	}
	dd {
		margin: 0.05rem 0 0;
	}
	.section-head {
		align-items: baseline;
		display: flex;
		flex-wrap: wrap;
		gap: 0.8rem;
		justify-content: space-between;
		margin-top: 1.2rem;
	}
	.table-wrap {
		border-top: 1px solid var(--ink);
		margin-top: 0.8rem;
		overflow-x: auto;
	}
	table {
		border-collapse: collapse;
		font-family: var(--mono);
		font-size: 0.78rem;
		width: 100%;
	}
	th {
		border-bottom: 1px solid var(--rule);
		color: var(--ink-muted);
		font-weight: 500;
		text-align: left;
	}
	th,
	td {
		padding: 0.5rem 0.55rem;
		vertical-align: top;
	}
	td {
		border-bottom: 1px solid var(--rule);
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
	.pager {
		display: flex;
		font-family: var(--mono);
		font-size: 0.78rem;
		gap: 0.8rem;
		justify-content: flex-end;
		margin-top: 0.8rem;
	}
	.pager span {
		color: var(--ink-muted);
	}
	@media (max-width: 720px) {
		header,
		.section-head {
			align-items: flex-start;
			flex-direction: column;
		}
		h1 {
			font-size: 1.55rem;
		}
		main {
			padding: 1rem;
		}
	}
</style>
