<!--
  DriverMosaic (E11) — where the tiered verdict's errors live, BY WHAT DROVE the
  decision. Extends the ConfusionMosaic honest-geometry register to a third axis:
  one column per reject-driver, column width ∝ how many statements that driver
  decided, each column a vertical stack of outcomes (caught / over-rejected /
  missed / agreed), height ∝ count. Same single-palette semantics as
  ConfusionMosaic — forest-green caught, dark-green agreed, amber over-rejected,
  rust the dangerous miss. The shape IS the finding: deterministic is a thin
  flawless sliver, the LLM-review column carries the over-rejection amber, and the
  unflagged column is where the rust misses hide.

  Counts come from metrics.json tiers.stmt.stratified.by_driver verbatim (G4).
-->
<script lang="ts">
	import type { StratumBlock } from '$lib/data/types';

	let { drivers }: { drivers: Record<string, StratumBlock> } = $props();

	// Order best→worst by what the driver MEANS, not alphabetically: the flawless
	// deterministic hard-flag, then the noisier LLM review, then the silent pass.
	const ORDER = ['deterministic', 'llm', 'none'];
	const LABEL: Record<string, string> = {
		deterministic: 'deterministic',
		llm: 'LLM review',
		none: 'unflagged'
	};
	const GLOSS: Record<string, string> = {
		deterministic: 'grounding hard-flag',
		llm: 'credible LLM incorrect',
		none: 'nothing flagged it'
	};

	type Lane = {
		key: string;
		n: number;
		tp: number;
		fp: number;
		fn: number;
		tn: number;
		flagged: boolean;
		stat: string;
	};

	const lanes = $derived<Lane[]>(
		ORDER.filter((k) => drivers[k]).map((k) => {
			const c = drivers[k].verdict_err;
			const flagged = c.tp + c.fp >= c.fn + c.tn; // pred-error tier vs pred-correct tier
			const prec = c.tp + c.fp > 0 ? c.tp / (c.tp + c.fp) : null;
			const stat = flagged
				? prec != null
					? `precision ${prec.toFixed(2)}`
					: '—'
				: `${c.fn} slipped through`;
			return { key: k, n: c.n, tp: c.tp, fp: c.fp, fn: c.fn, tn: c.tn, flagged, stat };
		})
	);
	const total = $derived(lanes.reduce((s, l) => s + l.n, 0) || 1);
	// Column widths ∝ n (floored so the thin deterministic lane stays labelable).
	// A grid (not flex columns) keeps every head / stack / footer on a shared
	// baseline regardless of how each gloss wraps.
	const cols = $derived(lanes.map((l) => `minmax(6.5rem, ${l.n}fr)`).join(' '));
</script>

<figure class="dm">
	<div class="grid" style="grid-template-columns:{cols}">
		{#each lanes as l, i (l.key)}
			<div class="col-head" style="grid-column:{i + 1};grid-row:1">
				<span class="dl">{LABEL[l.key] ?? l.key}</span>
				<span class="dn">{l.n.toLocaleString('en-US')} · {((l.n / total) * 100).toFixed(0)}%</span>
				<span class="dg">{GLOSS[l.key] ?? ''}</span>
			</div>
		{/each}
		{#each lanes as l, i (l.key)}
			<div class="stack" style="grid-column:{i + 1};grid-row:2">
				{#if l.tp}
					<div class="seg catch" style="flex:{l.tp}" title={`caught error — ${l.tp}`}>
						<span>{l.tp}</span>
					</div>
				{/if}
				{#if l.fp}
					<div class="seg over" style="flex:{l.fp}" title={`over-rejected a correct statement — ${l.fp}`}>
						<span>{l.fp}</span>
					</div>
				{/if}
				{#if l.fn}
					<div class="seg miss" style="flex:{l.fn}" title={`missed a real error (silent pass) — ${l.fn}`}>
						<span>{l.fn}</span>
					</div>
				{/if}
				{#if l.tn}
					<div class="seg agree" style="flex:{l.tn}" title={`agreed correct — ${l.tn}`}>
						<span>{l.tn}</span>
					</div>
				{/if}
			</div>
		{/each}
		{#each lanes as l, i (l.key)}
			<div class="col-foot" style="grid-column:{i + 1};grid-row:3">{l.stat}</div>
		{/each}
	</div>
	<div class="legend">
		<span><i class="sw catch"></i> error caught</span>
		<span><i class="sw over"></i> over-rejected</span>
		<span><i class="sw miss"></i> missed (silent)</span>
		<span><i class="sw agree"></i> agreed ✓</span>
	</div>
</figure>

<style>
	.dm {
		margin: 0;
	}
	.grid {
		display: grid;
		/* rows: header band (equal height) · the 11rem stack · footer band */
		grid-template-rows: auto 11rem auto;
		column-gap: 3px;
		row-gap: 0.35rem;
		align-items: stretch;
	}
	.col-head {
		display: flex;
		flex-direction: column;
		gap: 0.05rem;
		align-self: end; /* labels sit just above their stack */
		min-width: 0;
	}
	.dl {
		font-family: var(--mono);
		font-size: 0.74rem;
		color: var(--ink);
	}
	.dn {
		font-family: var(--mono);
		font-size: 0.92rem;
		font-variant-numeric: tabular-nums;
		color: var(--ink);
	}
	.dg {
		font-family: var(--mono);
		font-size: 0.62rem;
		color: var(--ink-faint);
		letter-spacing: 0.01em;
	}
	.stack {
		display: flex;
		flex-direction: column;
		gap: 2px;
		height: 100%; /* fills the grid's 11rem stack row */
		min-height: 0;
		min-width: 0;
	}
	.seg {
		display: flex;
		align-items: center;
		justify-content: center;
		min-height: 0;
		overflow: hidden;
	}
	.seg span {
		font-family: var(--mono);
		font-size: 0.72rem;
		color: var(--paper);
		font-variant-numeric: tabular-nums;
		opacity: 0.95;
	}
	/* Same single-palette semantics as ConfusionMosaic: forest-green caught,
	   dark-green agreed, amber over-rejected, rust the dangerous miss. */
	.seg.catch {
		background: var(--ok-green);
	}
	.seg.agree {
		background: color-mix(in srgb, var(--ok-green) 78%, black);
	}
	.seg.over {
		background: #6f5a16;
	}
	.seg.miss {
		background: var(--accent);
	}
	.col-foot {
		font-family: var(--mono);
		font-size: 0.64rem;
		color: var(--ink-muted);
		align-self: start;
		font-variant-numeric: tabular-nums;
	}
	.legend {
		display: flex;
		flex-wrap: wrap;
		gap: 0.3rem 1rem;
		margin-top: 0.6rem;
		font-family: var(--mono);
		font-size: 0.64rem;
		color: var(--ink-faint);
	}
	.sw {
		display: inline-block;
		width: 0.66rem;
		height: 0.66rem;
		vertical-align: -1px;
		margin-right: 0.2rem;
	}
	.sw.catch {
		background: var(--ok-green);
	}
	.sw.agree {
		background: color-mix(in srgb, var(--ok-green) 78%, black);
	}
	.sw.over {
		background: #6f5a16;
	}
	.sw.miss {
		background: var(--accent);
	}
</style>
