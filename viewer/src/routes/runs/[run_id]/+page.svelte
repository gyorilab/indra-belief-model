<script lang="ts">
	import type { PageData } from './$types';
	import HeuristicCoverage from '$lib/components/HeuristicCoverage.svelte';
	import TraceFidelity from '$lib/components/TraceFidelity.svelte';

	let { data }: { data: PageData } = $props();
	const m = $derived(data.meta);
	const n = $derived(data.narrative);
	const cov = $derived(data.coverage);
	const fidelity = $derived(data.fidelity);
	const allRuns = $derived(data.allRuns);

		type DenominatorRow = {
			label: string;
			n: number | null;
			panel: string;
			source: string;
			meaning: string;
			href?: string | null;
		};

	function fmtCost(c: number | null): string {
		if (c == null) return '—';
		if (c < 0.01) return '<$0.01';
		if (c < 1) return '$' + c.toFixed(2);
		if (c < 100) return '$' + c.toFixed(2);
		return '$' + c.toFixed(0);
	}

	function actualCostLabel(status: string, c: number | null): string {
		if (status === 'succeeded') return fmtCost(c);
		if (c == null) return 'not finalized';
		return `${fmtCost(c)} partial`;
	}

	function statementCountLabel(status: string, n: number | null): string {
		if (n == null) return '—';
		return status === 'succeeded' ? String(n) : `${n} target`;
	}

	function evidenceCountLabel(status: string, n: number | null): string {
		if (n == null) return '—';
		return status === 'succeeded' ? String(n) : `${n} persisted`;
	}

	function fmtCount(n: number | null): string {
		return n == null ? '—' : n.toLocaleString('en-US');
	}

	function fmtSigned(n: number, digits = 3): string {
		const sign = n >= 0 ? '+' : '−';
		return `${sign}${Math.abs(n).toFixed(digits)}`;
	}

	function archMark(architecture: string | null | undefined): string {
		if (architecture === 'monolithic') return '[M]';
		if (architecture === 'decomposed') return '[D]';
		return '[?]';
	}

	function comparisonBlockedIsArchitecture(reason: string | null): boolean {
		return Boolean(reason?.includes('across architectures'));
	}

	const compareOptions = $derived(
		allRuns.filter((r) => r.run_id !== m.run_id && r.architecture === m.architecture)
	);
	const succeededWithoutScorerRows = $derived(
		m.status === 'succeeded' &&
		(m.n_stmts ?? 0) > 0 &&
		(m.n_scorer_steps ?? 0) === 0
	);
	const denominatorRows = $derived.by((): DenominatorRow[] => {
		const rows: DenominatorRow[] = [
			{
				label: 'target statements',
				n: m.n_stmts,
				panel: 'run scope and cost preflight',
				source: 'score_run.n_stmts',
				meaning: 'requested statement input; not proof that scoring output exists'
			},
			{
				label: 'persisted scorer steps',
				n: m.n_scorer_steps,
				panel: 'storage plane',
				source: 'scorer_step rows',
				meaning: 'append-only rows written by the worker for this run'
			}
		];
		if (fidelity) {
			rows.push({
				label: 'trace evidence states',
				n: fidelity.n_evidences,
				panel: 'trace fidelity rail and detail table',
				source: 'trace-state CTE',
				meaning: 'one evidence-level trace state for each evidence with persisted step evidence'
			});
		}
		if (cov) {
			rows.push(
				{
					label: 'aggregate verdict evidence',
					n: cov.n_evidences,
					panel: 'verdict, belief, cost, latency',
					source: "scorer_step where step_kind='aggregate'",
					meaning: 'completed aggregate rows; architecture-blind metric denominator',
					href: `/runs/${m.run_id}/cohort`
				},
				{
					label: 'probe coverage evidence',
					n: cov.n_probe_evidences,
					panel: 'four-probe substrate/LLM coverage',
					source: 'substrate_route/probe rows',
					meaning: 'decomposed rows with persisted probe slots; coverage rates use this denominator',
					href: cov.n_probe_evidences > 0 ? `/runs/${m.run_id}/cohort?probe_coverage=present` : null
				}
			);
		}
		return rows;
	});
	const denominatorDiverges = $derived.by(() => {
		const counts = new Set(denominatorRows.map((row) => row.n).filter((n): n is number => n != null));
		return counts.size > 1;
	});
	</script>

<svelte:head><title>{m.run_id.slice(0, 8)} · run · INDRA Belief</title></svelte:head>

<header>
	<div class="crumb">
		<a href="/">corpus</a><span class="sep"> / </span><strong>run {m.run_id.slice(0, 8)}</strong>
	</div>
	<div class="meta">
		<a href={`/runs/${m.run_id}/repairs`} class="nav-link">repair backlog</a>
		<a href="/" class="nav-link">← dashboard</a>
	</div>
</header>

<main id="main">
	<section class="run-meta">
		<h1 class="run-h">
			<span class="muted">run</span>
			<code>{m.run_id.slice(0, 8)}</code>
			<span class={`run-arch run-arch-${m.architecture}`} title={`architecture: ${m.architecture}`}>{archMark(m.architecture)}</span>
			<span class="muted">·</span>
			<code class="run-version">{m.scorer_version}</code>
			<span class="run-status run-status-{m.status}">{m.status}</span>
		</h1>
		<dl class="run-fields">
			<div><dt>started</dt><dd>{m.started_at}</dd></div>
			<div><dt>architecture</dt><dd>{m.architecture}</dd></div>
			{#if m.paired_run_group_id}
				<div><dt>paired group</dt><dd><a href={`/pairs/${m.paired_run_group_id}`}>{m.paired_run_group_id}</a></dd></div>
			{/if}
			<div><dt>model</dt><dd>{m.model_id_default ?? '—'}</dd></div>
			<div><dt>indra</dt><dd>{m.indra_version ?? '—'}</dd></div>
			<div><dt>statements</dt><dd>{statementCountLabel(m.status, m.n_stmts)}</dd></div>
			<div><dt>evidences</dt><dd>{evidenceCountLabel(m.status, m.n_evidences)}</dd></div>
			<div><dt>est cost</dt><dd>{fmtCost(m.cost_estimate_usd)}</dd></div>
			<div><dt>actual cost</dt><dd>{actualCostLabel(m.status, m.cost_actual_usd)}</dd></div>
			{#if m.terminated_by || m.termination_reason}
				<div><dt>termination</dt><dd>{m.terminated_by ?? 'unknown'}{#if m.termination_reason} · {m.termination_reason}{/if}</dd></div>
			{/if}
		</dl>
		{#if succeededWithoutScorerRows}
			<div class="run-integrity-warning" role="note" aria-label="run output integrity warning">
				<span>run output integrity</span>
				<p>
					This run is marked <strong>succeeded</strong> with {m.n_stmts} target statement{m.n_stmts === 1 ? '' : 's'}, but no persisted <code>scorer_step</code> rows. Trace, repair, and comparison panels are not evidence of completed scoring until the run is rerun or the database is repaired.
				</p>
			</div>
		{/if}
		</section>

		<section class="run-narrative">
		<h2 class="rn-h">how this run compares</h2>
		{#if m.status !== 'succeeded'}
			<p class="rn-sentence">
				comparison not defined: this run is <strong>{m.status}</strong>, so before/after and predecessor deltas would mix incomplete evidence with completed runs.
			</p>
			{#if m.termination_reason}
				<p class="hint">termination: {m.terminated_by ?? 'unknown'} · {m.termination_reason}</p>
			{/if}
		{:else if !n}
			<p class="hint">no validity computed for this run · invoke <code>compute_validity(con, '{m.run_id.slice(0, 8)}…')</code> in Python</p>
		{:else if n.comparison_blocked_reason}
			<p class="rn-sentence">{n.summary_sentence}</p>
			{#if comparisonBlockedIsArchitecture(n.comparison_blocked_reason)}
				<p class="hint">
					cross-architecture deltas require the paired workbench so overlap and non-comparable fields are visible first.
					{#if m.paired_run_group_id}<a href={`/pairs/${m.paired_run_group_id}`}>open paired workbench</a>{/if}
				</p>
			{:else}
				<p class="hint">trace fidelity must have finalized aggregate verdict rows before run deltas are meaningful.</p>
			{/if}
		{:else if !n.prev_run_id}
			<p class="rn-sentence">{n.summary_sentence}</p>
			<p class="hint">no prior run to compare against. After a second <code>score_corpus</code>, this section will surface deltas.</p>
		{:else}
			<p class="rn-sentence"><strong>vs <code>{n.prev_run_id.slice(0, 8)}</code>:</strong> {n.summary_sentence}</p>
			<dl class="rn-deltas">
				{#if n.mae_delta != null}
					<div class="rn-delta">
						<dt>MAE Δ</dt>
						<dd class={n.mae_delta < 0 ? 'rn-good' : n.mae_delta > 0 ? 'rn-bad' : ''}>{fmtSigned(n.mae_delta)}</dd>
					</div>
				{/if}
				{#if n.bias_delta != null}
					<div class="rn-delta">
						<dt>bias Δ</dt>
						<dd>{fmtSigned(n.bias_delta)}</dd>
					</div>
				{/if}
				<div class="rn-delta">
					<dt>verdicts moved</dt>
					<dd>{n.verdicts_moved_total} <span class="muted">({n.verdicts_moved_to_correct} → correct, {n.verdicts_moved_to_incorrect} → incorrect)</span></dd>
				</div>
			</dl>
			{#if n.verdict_crossings.length > 0}
				<details class="rn-crossings">
					<summary>verdict-crossing statements ({n.verdict_crossings.length})</summary>
					<ul class="rn-crossings-list">
						{#each n.verdict_crossings.slice(0, 30) as c}
							<li>
								<a href={`/statements/${c.stmt_hash}?run_id=${m.run_id}`}><code>{c.stmt_hash.slice(0, 8)}</code></a>
								<span class="muted">{c.prev_verdict} → </span>
								<span class={`rn-cross-${c.curr_verdict}`}>{c.curr_verdict}</span>
							</li>
						{/each}
					</ul>
					{#if n.verdict_crossings.length > 30}
						<p class="muted">…and {n.verdict_crossings.length - 30} more.</p>
					{/if}
				</details>
			{/if}
		{/if}
		{#if m.status === 'succeeded' && compareOptions.length > 0}
			<form class="rn-compare-form" method="get">
				<label class="rn-compare-label">
					compare against:
					<select name="compare_to" onchange={(ev) => {
						const v = (ev.target as HTMLSelectElement).value;
						const url = v ? `?compare_to=${v}` : '';
						window.location.href = window.location.pathname + url;
					}}>
						<option value="">(auto — most-recent earlier run)</option>
						{#each compareOptions as r}
							<option value={r.run_id} selected={data.compareToParam === r.run_id}>
								{archMark(r.architecture)} {r.run_id.slice(0, 8)} · {r.scorer_version} · {r.started_at.replace(/\.\d+$/, '')}
							</option>
						{/each}
					</select>
				</label>
			</form>
			{/if}
		</section>

		<section class="run-denominators" aria-label="run denominator boundaries">
			<h2 class="rd-h">denominator boundaries</h2>
			<p class="rd-summary">
				{#if denominatorDiverges}
					These panels intentionally do not share one denominator; compare rows only inside the panel whose source matches the question.
				{:else}
					These panels currently share the same count, but each count still comes from a different evidence source.
				{/if}
			</p>
			<div class="rd-table-wrap">
				<table class="rd-table">
					<thead>
						<tr>
							<th>population</th>
							<th class="num">n</th>
							<th>panel</th>
							<th>source</th>
							<th>meaning</th>
						</tr>
					</thead>
					<tbody>
						{#each denominatorRows as row}
							<tr>
									<td>{#if row.href}<a href={row.href}>{row.label}</a>{:else}{row.label}{/if}</td>
								<td class="num">{fmtCount(row.n)}</td>
								<td>{row.panel}</td>
								<td><code>{row.source}</code></td>
								<td>{row.meaning}</td>
							</tr>
						{/each}
					</tbody>
				</table>
			</div>
		</section>

		{#if fidelity}
			<TraceFidelity {fidelity} runStatus={m.status} compareToParam={data.compareToParam} />
	{/if}

	{#if cov}
		<HeuristicCoverage coverage={cov} runStatus={m.status} />
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
		--ok-green: #2a6f2a;
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
	.crumb a { color: var(--ink-muted); text-decoration: none; }
	.crumb a:hover { color: var(--ink); }
	.crumb strong { color: var(--ink); font-weight: 500; }
	.crumb .sep { color: var(--ink-faint); }
	.meta {
		display: flex;
		gap: 0.8rem;
		align-items: baseline;
	}
	.nav-link {
		color: var(--accent);
		text-decoration: none;
	}
	.nav-link:hover { text-decoration: underline; }
	.muted { color: var(--ink-faint); }
	main {
		max-width: 1200px;
		margin: 0 auto;
		padding: 2rem 1.5rem 4rem;
	}

	.run-h {
		font-family: var(--serif);
		font-size: 1.4rem;
		font-weight: 400;
		margin: 0 0 0.6rem;
		display: flex;
		gap: 0.5rem;
		align-items: baseline;
		flex-wrap: wrap;
	}
	.run-version { font-family: var(--mono); font-size: 0.86rem; color: var(--ink); }
	.run-arch {
		font-family: var(--mono);
		font-size: 0.86rem;
		color: var(--ink);
	}
	.run-arch-monolithic { color: var(--accent); }
	.run-arch-decomposed { color: var(--ok-green); }
	.run-arch-unknown { color: var(--ink-faint); }
	.run-status {
		font-family: var(--mono);
		font-size: 0.74rem;
		text-transform: lowercase;
		letter-spacing: 0.04em;
		padding: 0 0.4rem;
	}
	.run-status-succeeded { color: var(--ok-green); }
	.run-status-running { color: var(--ink); font-style: italic; }
	.run-status-failed { color: var(--accent); font-weight: 500; }
	.run-status-canceled { color: var(--accent); font-style: italic; }

	.run-fields {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
		gap: 0;
		margin: 0 0 2.5rem;
	}
	.run-fields > div {
		padding: 0.4rem 1rem 0.4rem 0;
		border-right: 1px solid var(--rule);
	}
	.run-fields > div:last-child {
		border-right: none;
	}
	.run-fields dt {
		font-family: var(--mono);
		font-size: 0.7rem;
		color: var(--ink-muted);
		text-transform: lowercase;
		letter-spacing: 0.02em;
		margin: 0;
	}
	.run-fields dd {
		font-family: var(--mono);
		font-size: 0.86rem;
		color: var(--ink);
		margin: 0.1rem 0 0;
		font-variant-numeric: tabular-nums;
		overflow: hidden;
		text-overflow: ellipsis;
	}
	.run-integrity-warning {
		margin: -1.2rem 0 2.5rem;
		border-left: 3px solid #9d4a1a;
		background: rgba(157, 74, 26, 0.08);
		padding: 0.75rem 0.9rem;
		max-width: 760px;
	}
	.run-integrity-warning span {
		display: block;
		font-family: var(--mono);
		font-size: 0.72rem;
		text-transform: uppercase;
		letter-spacing: 0.04em;
		color: #7d2a1a;
		margin-bottom: 0.25rem;
	}
	.run-integrity-warning p {
		margin: 0;
		color: var(--ink);
	}

	.run-narrative {
		margin-bottom: 2.5rem;
	}
	.rn-h {
		font-family: var(--serif);
		font-size: 1.15rem;
		font-weight: 400;
		margin: 0 0 0.5rem;
	}
	.rn-sentence {
		font-family: var(--serif);
		font-size: 1rem;
		color: var(--ink);
		margin: 0 0 0.8rem;
		line-height: 1.5;
		font-variant-numeric: tabular-nums;
	}
	.hint {
		font-family: var(--serif);
		font-style: italic;
		font-size: 0.9rem;
		color: var(--ink-muted);
		margin: 0 0 0.5rem;
	}

	.rn-deltas {
		display: flex;
		flex-wrap: wrap;
		gap: 1.6rem;
		margin: 0.5rem 0 1rem;
	}
	.rn-delta dt {
		font-family: var(--mono);
		font-size: 0.7rem;
		color: var(--ink-muted);
		text-transform: lowercase;
		letter-spacing: 0.02em;
		margin: 0;
	}
	.rn-delta dd {
		font-family: var(--mono);
		font-size: 1.1rem;
		font-variant-numeric: tabular-nums;
		margin: 0.1rem 0 0;
	}
	.rn-good { color: var(--ok-green); }
	.rn-bad { color: var(--accent); }

	.rn-crossings {
		margin-top: 0.6rem;
	}
	.rn-crossings summary {
		font-family: var(--mono);
		font-size: 0.78rem;
		color: var(--ink-muted);
		cursor: pointer;
	}
	.rn-crossings-list {
		list-style: none;
		padding: 0.5rem 0 0 1rem;
		margin: 0;
		font-family: var(--mono);
		font-size: 0.8rem;
	}
	.rn-crossings-list li {
		padding: 0.15rem 0;
	}
	.rn-cross-correct { color: var(--ok-green); }
	.rn-cross-incorrect { color: var(--accent); }
	.rn-cross-abstain { color: var(--ink-muted); font-style: italic; }

	.rn-compare-form {
		margin-top: 1.2rem;
		padding-top: 0.8rem;
		border-top: 1px dotted var(--rule);
	}
	.rn-compare-label {
		font-family: var(--mono);
		font-size: 0.78rem;
		color: var(--ink-muted);
		display: flex;
		gap: 0.5rem;
		align-items: baseline;
		flex-wrap: wrap;
	}
		.rn-compare-label select {
			font-family: var(--mono);
			font-size: 0.78rem;
			padding: 0.2rem 0.4rem;
			background: var(--paper);
			color: var(--ink);
			border: 1px solid var(--rule);
		}

		.run-denominators {
			margin: 0 0 2.5rem;
		}
		.rd-h {
			font-family: var(--serif);
			font-size: 1.15rem;
			font-weight: 400;
			margin: 0 0 0.4rem;
		}
		.rd-summary {
			max-width: 760px;
			margin: 0 0 0.65rem;
			color: var(--ink-muted);
			font-size: 0.92rem;
			line-height: 1.45;
		}
		.rd-table {
			width: 100%;
			border-collapse: collapse;
			font-family: var(--mono);
			font-size: 0.78rem;
			font-variant-numeric: tabular-nums;
		}
		.rd-table th {
			text-align: left;
			font-weight: 500;
			color: var(--ink-muted);
			font-size: 0.7rem;
			text-transform: lowercase;
			letter-spacing: 0.02em;
			padding: 0.3rem 0.65rem 0.3rem 0;
			border-bottom: 1px dotted var(--rule);
		}
		.rd-table td {
			vertical-align: top;
			padding: 0.42rem 0.65rem 0.42rem 0;
			border-bottom: 1px dotted var(--rule);
		}
		.rd-table tr:last-child td {
			border-bottom: none;
		}
		.rd-table .num {
			text-align: right;
			white-space: nowrap;
			color: var(--ink);
		}
		.rd-table code {
			font-size: 0.74rem;
			white-space: nowrap;
		}
		.rd-table-wrap {
			overflow-x: auto;
		}
	</style>
