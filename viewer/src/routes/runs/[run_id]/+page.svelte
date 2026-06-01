<script lang="ts">
	import type { PageData } from './$types';

	let { data }: { data: PageData } = $props();
	const run = $derived(data.run);
	const v = $derived(data.validity);
	const residual = $derived(data.residual);

	function fmtCount(n: number | null | undefined): string {
		return n == null ? '—' : n.toLocaleString('en-US');
	}

	function fmt3(n: number | null | undefined): string {
		return n == null ? '—' : n.toFixed(3);
	}

	function fmtSigned(n: number | null | undefined, digits = 3): string {
		if (n == null) return '—';
		const sign = n >= 0 ? '+' : '−';
		return `${sign}${Math.abs(n).toFixed(digits)}`;
	}

	// Verdict mix, ordered correct → incorrect → unscored → rest.
	const VERDICT_ORDER: Record<string, number> = { correct: 0, incorrect: 1, unscored: 9 };
	const verdicts = $derived(
		[...v.verdicts].sort(
			(a, b) => (VERDICT_ORDER[a.verdict] ?? 5) - (VERDICT_ORDER[b.verdict] ?? 5) || b.n - a.n
		)
	);
	const verdictTotal = $derived(verdicts.reduce((s, x) => s + x.n, 0));

	const bucketTotal = $derived(v.buckets.reduce((s, x) => s + x.n, 0));

	// Residual histogram: 11 bins spanning [-1, +1]; tallest bar = 100% height.
	const residualMax = $derived(residual ? Math.max(1, ...residual.bins) : 1);
	function binLabel(i: number, n: number): string {
		const lo = (-1 + (2 * i) / n).toFixed(2);
		const hi = (-1 + (2 * (i + 1)) / n).toFixed(2);
		return `${lo} … ${hi}`;
	}

	function verdictClass(verdict: string): string {
		if (verdict === 'correct') return 'v-correct';
		if (verdict === 'incorrect') return 'v-incorrect';
		return 'v-other';
	}
</script>

<svelte:head><title>{run.run_id.slice(0, 8)} · run · INDRA Belief</title></svelte:head>

<header>
	<div class="crumb">
		<a href="/runs">runs</a><span class="sep"> / </span><strong>{run.run_id.slice(0, 8)}</strong>
	</div>
	<div class="meta">
		<a class="nav-link" href="/statements">statements</a>
	</div>
</header>

<main id="main">
	<section class="run-meta">
		<h1 class="run-h">
			<span class="muted">run</span>
			<code>{run.run_id.slice(0, 8)}</code>
			<span class="muted">·</span>
			<span class="run-model">{run.model}</span>
			{#if run.status}
				<span class="run-status">{run.status}</span>
			{/if}
		</h1>
		<dl class="run-fields">
			<div><dt>model</dt><dd>{run.model}</dd></div>
			<div><dt>generated</dt><dd>{run.generated_date ?? '—'}</dd></div>
			<div><dt>statements</dt><dd>{fmtCount(run.n_statements)}</dd></div>
			<div><dt>evidences</dt><dd>{fmtCount(run.n_evidences)}</dd></div>
			<div><dt>scored evidence</dt><dd>{fmtCount(v.calibration.n)}</dd></div>
		</dl>
	</section>

	<section class="calibration">
		<h2 class="sec-h">calibration vs INDRA belief</h2>
		<p class="sec-sub">
			Residual is our per-evidence score minus the published RasMachine belief, over the
			{fmtCount(v.calibration.n)} evidence rows with a score, a belief, and a verdict.
		</p>
		<dl class="stat-row">
			<div class="stat">
				<dt>MAE</dt>
				<dd>{fmt3(v.calibration.mae)}</dd>
			</div>
			<div class="stat">
				<dt>bias</dt>
				<dd>{fmtSigned(v.calibration.bias)}</dd>
			</div>
			<div class="stat">
				<dt>mean residual</dt>
				<dd>{fmtSigned(residual?.mean_residual)}</dd>
			</div>
		</dl>

		{#if residual && residual.n_total > 0}
			<div class="hist" role="img" aria-label="residual distribution histogram, 11 bins from −1 to +1">
				{#each residual.bins as count, i}
					<div class="hist-col" title={`${binLabel(i, residual.bins.length)} · ${fmtCount(count)}`}>
						<div class="hist-bar" style={`height:${(count / residualMax) * 100}%`}></div>
					</div>
				{/each}
			</div>
			<div class="hist-axis">
				<span>−1.0</span>
				<span class="muted">residual (our − INDRA)</span>
				<span>+1.0</span>
			</div>
		{/if}
	</section>

	<section class="verdicts">
		<h2 class="sec-h">verdict distribution</h2>
		<ul class="dist">
			{#each verdicts as row}
				<li class={`dist-row ${verdictClass(row.verdict)}`}>
					<span class="dist-label">{row.verdict}</span>
					<span class="dist-bar-wrap">
						<span
							class="dist-bar"
							style={`width:${verdictTotal ? (row.n / verdictTotal) * 100 : 0}%`}
						></span>
					</span>
					<span class="dist-n">{fmtCount(row.n)}</span>
				</li>
			{/each}
		</ul>
	</section>

	<section class="buckets">
		<h2 class="sec-h">bucket taxonomy</h2>
		<p class="sec-sub">Report-taxonomy classification of every evidence row in this run.</p>
		<ul class="dist">
			{#each v.buckets as row}
				<li class="dist-row">
					<span class="dist-label">{row.bucket}</span>
					<span class="dist-bar-wrap">
						<span
							class="dist-bar dist-bar-neutral"
							style={`width:${bucketTotal ? (row.n / bucketTotal) * 100 : 0}%`}
						></span>
					</span>
					<span class="dist-n">{fmtCount(row.n)}</span>
				</li>
			{/each}
		</ul>
	</section>

	{#if v.confidenceCalibration.length > 0}
		<section class="slice">
			<h2 class="sec-h">calibration by stated confidence</h2>
			<table class="slice-table">
				<thead>
					<tr>
						<th>confidence</th>
						<th class="num">n</th>
						<th class="num">MAE</th>
						<th class="num">bias</th>
					</tr>
				</thead>
				<tbody>
					{#each v.confidenceCalibration as c}
						<tr>
							<td>{c.confidence}</td>
							<td class="num">{fmtCount(c.n)}</td>
							<td class="num">{fmt3(c.mae)}</td>
							<td class="num">{fmtSigned(c.bias)}</td>
						</tr>
					{/each}
				</tbody>
			</table>
		</section>
	{/if}

	{#if v.byIndraType.length > 0}
		<section class="slice">
			<h2 class="sec-h">calibration by INDRA statement type</h2>
			<p class="sec-sub">Sorted by mean absolute residual — worst-calibrated types first.</p>
			<table class="slice-table">
				<thead>
					<tr>
						<th>statement type</th>
						<th class="num">n</th>
						<th class="num">MAE</th>
						<th class="num">bias</th>
					</tr>
				</thead>
				<tbody>
					{#each v.byIndraType as s}
						<tr>
							<td>{s.value}</td>
							<td class="num">{fmtCount(s.n)}</td>
							<td class="num">{fmt3(s.mae)}</td>
							<td class="num">{fmtSigned(s.bias)}</td>
						</tr>
					{/each}
				</tbody>
			</table>
		</section>
	{/if}

	{#if v.bySourceApi.length > 0}
		<section class="slice">
			<h2 class="sec-h">calibration by source API</h2>
			<table class="slice-table">
				<thead>
					<tr>
						<th>source</th>
						<th class="num">n</th>
						<th class="num">MAE</th>
						<th class="num">bias</th>
					</tr>
				</thead>
				<tbody>
					{#each v.bySourceApi as s}
						<tr>
							<td>{s.value}</td>
							<td class="num">{fmtCount(s.n)}</td>
							<td class="num">{fmt3(s.mae)}</td>
							<td class="num">{fmtSigned(s.bias)}</td>
						</tr>
					{/each}
				</tbody>
			</table>
		</section>
	{/if}

	{#if v.unavailable.length > 0}
		<section class="unavailable" aria-label="unavailable strata">
			<h2 class="sec-h">not available from monolithic exports</h2>
			<ul class="unavailable-list">
				{#each v.unavailable as u}
					<li>{u}</li>
				{/each}
			</ul>
		</section>
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
	.crumb a {
		color: var(--ink-muted);
		text-decoration: none;
	}
	.crumb a:hover {
		color: var(--ink);
	}
	.crumb strong {
		color: var(--ink);
		font-weight: 500;
	}
	.crumb .sep {
		color: var(--ink-faint);
	}
	.meta {
		display: flex;
		gap: 0.8rem;
		align-items: baseline;
	}
	.nav-link {
		color: var(--accent);
		text-decoration: none;
	}
	.nav-link:hover {
		text-decoration: underline;
	}
	.muted {
		color: var(--ink-faint);
	}
	main {
		max-width: 1100px;
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
	.run-model {
		font-family: var(--mono);
		font-size: 0.9rem;
		color: var(--accent);
	}
	.run-status {
		font-family: var(--mono);
		font-size: 0.74rem;
		text-transform: lowercase;
		letter-spacing: 0.04em;
		color: var(--ok-green);
		padding: 0 0.4rem;
	}

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

	section {
		margin: 0 0 2.5rem;
	}
	.sec-h {
		font-family: var(--serif);
		font-size: 1.15rem;
		font-weight: 400;
		margin: 0 0 0.4rem;
	}
	.sec-sub {
		max-width: 760px;
		margin: 0 0 0.8rem;
		color: var(--ink-muted);
		font-size: 0.92rem;
		line-height: 1.45;
	}

	.stat-row {
		display: flex;
		flex-wrap: wrap;
		gap: 2rem;
		margin: 0 0 1.2rem;
	}
	.stat dt {
		font-family: var(--mono);
		font-size: 0.7rem;
		color: var(--ink-muted);
		text-transform: lowercase;
		letter-spacing: 0.02em;
		margin: 0;
	}
	.stat dd {
		font-family: var(--mono);
		font-size: 1.3rem;
		font-variant-numeric: tabular-nums;
		margin: 0.1rem 0 0;
		color: var(--ink);
	}

	.hist {
		display: flex;
		align-items: flex-end;
		gap: 2px;
		height: 110px;
		border-bottom: 1px solid var(--rule);
		margin-top: 0.5rem;
	}
	.hist-col {
		flex: 1 1 0;
		display: flex;
		align-items: flex-end;
		height: 100%;
	}
	.hist-bar {
		width: 100%;
		background: var(--accent);
		opacity: 0.78;
		min-height: 1px;
	}
	.hist-axis {
		display: flex;
		justify-content: space-between;
		font-family: var(--mono);
		font-size: 0.7rem;
		color: var(--ink-faint);
		margin-top: 0.3rem;
	}

	.dist {
		list-style: none;
		padding: 0;
		margin: 0;
		font-family: var(--mono);
		font-size: 0.8rem;
	}
	.dist-row {
		display: grid;
		grid-template-columns: 12rem 1fr 6ch;
		gap: 0.8rem;
		align-items: center;
		padding: 0.2rem 0;
	}
	.dist-label {
		color: var(--ink);
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}
	.dist-bar-wrap {
		background: var(--accent-wash);
		height: 0.85rem;
	}
	.dist-bar {
		display: block;
		height: 100%;
		background: var(--ink-muted);
	}
	.dist-bar-neutral {
		background: var(--ink-muted);
	}
	.v-correct .dist-bar {
		background: var(--ok-green);
	}
	.v-incorrect .dist-bar {
		background: var(--accent);
	}
	.dist-n {
		text-align: right;
		font-variant-numeric: tabular-nums;
		color: var(--ink);
	}

	.slice-table {
		width: 100%;
		border-collapse: collapse;
		font-family: var(--mono);
		font-size: 0.78rem;
		font-variant-numeric: tabular-nums;
	}
	.slice-table th {
		text-align: left;
		font-weight: 500;
		color: var(--ink-muted);
		font-size: 0.7rem;
		text-transform: lowercase;
		letter-spacing: 0.02em;
		padding: 0.3rem 0.65rem 0.3rem 0;
		border-bottom: 1px dotted var(--rule);
	}
	.slice-table td {
		padding: 0.32rem 0.65rem 0.32rem 0;
		border-bottom: 1px dotted var(--rule);
		color: var(--ink);
	}
	.slice-table tr:last-child td {
		border-bottom: none;
	}
	.slice-table .num {
		text-align: right;
		white-space: nowrap;
	}

	.unavailable {
		border-left: 3px solid var(--rule);
		padding-left: 0.9rem;
	}
	.unavailable-list {
		margin: 0;
		padding-left: 1.1rem;
		color: var(--ink-muted);
		font-size: 0.92rem;
	}
	.unavailable-list li {
		padding: 0.12rem 0;
	}

	@media (max-width: 720px) {
		.dist-row {
			grid-template-columns: 8rem 1fr 6ch;
		}
	}
</style>
