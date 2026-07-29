<!--
  ReliabilityDiagram (E5) — the instrument's confession.

  x = mean predicted belief, y = empirical correct-rate, diagonal = perfect
  calibration. A mark BELOW the diagonal = over-confidence; ABOVE = under-
  confidence (the documented C0 finding: belief ~0.46 but ~65-72% correct shows
  as a mark sitting high above the line). Marks are sized by bin-n so a sparse
  bin cannot lie about a confident curve. Reads metrics.json bins VERBATIM —
  recomputes nothing.

  OVERLAY (C5): pass `bins2`/`hue2`/`label2` to render a SECOND series (run B)
  against the SAME shared diagonal — the run-comparison strong center. The eye
  reads divergence between A and B AND from the diagonal at once. Both series'
  marks share one bin-n scale so their relative weights stay honest. When the
  overlay is active the per-mark gap stems are dropped (two stem fans would
  cross-hatch into noise); divergence from the diagonal is carried by the line
  paths instead.
-->
<script lang="ts">
	import type { ReliabilityBin } from '$lib/data/types';
	import { rOf, pathOf, px, py } from '$lib/reliability-geometry';

	let {
		bins,
		n,
		hue = 'var(--accent)',
		label = '',
		bins2 = null,
		n2 = null,
		hue2 = 'var(--b-hue)',
		label2 = ''
	}: {
		bins: ReliabilityBin[];
		n: number;
		hue?: string;
		label?: string;
		/** Optional second series (run B). When present the diagram is an A/B overlay. */
		bins2?: ReliabilityBin[] | null;
		n2?: number | null;
		hue2?: string;
		label2?: string;
	} = $props();

	const overlay = $derived(bins2 != null);

	function toPts(bs: ReliabilityBin[]) {
		return bs
			.filter((b) => b.n > 0 && b.mean_pred != null && b.empirical != null)
			.map((b) => ({
				x: b.mean_pred as number,
				y: b.empirical as number,
				n: b.n,
				gap: (b.empirical as number) - (b.mean_pred as number)
			}));
	}
	// occupied bins only carry a point; the others still anchor the x-axis grid.
	const pts = $derived(toPts(bins));
	const pts2 = $derived(bins2 ? toPts(bins2) : []);
	// shared bin-n scale across BOTH series — relative weights stay comparable.
	const maxBinN = $derived(Math.max(1, ...pts.map((p) => p.n), ...pts2.map((p) => p.n)));

	// SVG viewbox in calibration space [0,1]×[0,1], y inverted (1 at top).
	// px/py/rOf/pathOf live in $lib/reliability-geometry (shared with the paper strip).
	function fmtPct(v: number): string {
		return `${(v * 100).toFixed(0)}%`;
	}
</script>

<figure class="rel">
	{#if label}<figcaption>{label}</figcaption>{/if}
	<svg viewBox="0 0 100 100" role="img" aria-label="reliability diagram, predicted vs empirical">
		<!-- perfect-calibration diagonal -->
		<line class="diag" x1={px(0)} y1={py(0)} x2={px(1)} y2={py(1)} />
		<!-- gridlines at 0.5 -->
		<line class="grid" x1={px(0.5)} y1={py(0)} x2={px(0.5)} y2={py(1)} />
		<line class="grid" x1={px(0)} y1={py(0.5)} x2={px(1)} y2={py(0.5)} />
		{#if overlay}
			<!-- A/B overlay: a connecting curve per series carries each model's
			     trajectory off the shared diagonal (no per-bin stems — two fans
			     would cross-hatch). Marks still area ∝ n on the shared scale. -->
			{#if pathOf(pts)}<path class="curve" d={pathOf(pts)} style="--hue:{hue}" />{/if}
			{#if pathOf(pts2)}<path class="curve" d={pathOf(pts2)} style="--hue:{hue2}" />{/if}
			{#each pts as p}
				<circle class="mark" cx={px(p.x)} cy={py(p.y)} r={rOf(p.n, maxBinN)} style="--hue:{hue}">
					<title>{`${label || 'A'} · predicted ${fmtPct(p.x)} · empirical ${fmtPct(p.y)} · n=${p.n} · ${p.gap >= 0 ? 'under' : 'over'}-confident by ${fmtPct(Math.abs(p.gap))}`}</title>
				</circle>
			{/each}
			{#each pts2 as p}
				<circle class="mark mark-b" cx={px(p.x)} cy={py(p.y)} r={rOf(p.n, maxBinN)} style="--hue:{hue2}">
					<title>{`${label2 || 'B'} · predicted ${fmtPct(p.x)} · empirical ${fmtPct(p.y)} · n=${p.n} · ${p.gap >= 0 ? 'under' : 'over'}-confident by ${fmtPct(Math.abs(p.gap))}`}</title>
				</circle>
			{/each}
		{:else}
			<!-- gap stems: how far each bin sits off the diagonal (the lie, drawn) -->
			{#each pts as p}
				<line class="stem" x1={px(p.x)} y1={py(p.x)} x2={px(p.x)} y2={py(p.y)} />
			{/each}
			<!-- bin marks, area ∝ n -->
			{#each pts as p}
				<circle
					class="mark"
					cx={px(p.x)}
					cy={py(p.y)}
					r={rOf(p.n, maxBinN)}
					style="--hue:{hue}"
				>
					<title>{`predicted ${fmtPct(p.x)} · empirical ${fmtPct(p.y)} · n=${p.n} · ${p.gap >= 0 ? 'under' : 'over'}-confident by ${fmtPct(Math.abs(p.gap))}`}</title>
				</circle>
			{/each}
		{/if}
	</svg>
	<div class="axes">
		<span class="ax-y">empirical correct-rate →</span>
		<span class="ax-x">mean predicted belief →</span>
	</div>
	{#if overlay}
		<p class="caption">
			<span class="key-a" style="--hue:{hue}">● {label || 'A'}</span>
			<span class="key-b" style="--hue:{hue2}">● {label2 || 'B'}</span>
			<span class="diag-key">— perfect</span> · marks sized by bin-n · above the line =
			under-confident, below = over-confident · n={n.toLocaleString('en-US')} / {(n2 ?? 0).toLocaleString('en-US')}
		</p>
	{:else}
		<p class="caption">
			<span class="diag-key">— perfect</span> · marks sized by bin-n · above the line =
			under-confident, below = over-confident · n={n.toLocaleString('en-US')}
		</p>
	{/if}
</figure>

<style>
	.rel {
		margin: 0;
	}
	figcaption {
		font-family: var(--mono);
		font-size: 0.74rem;
		color: var(--ink-muted);
		margin-bottom: 0.4rem;
		letter-spacing: 0.02em;
	}
	svg {
		display: block;
		width: 100%;
		max-width: 22rem;
		aspect-ratio: 1 / 1;
		overflow: visible;
	}
	.diag {
		stroke: var(--ink-faint);
		stroke-width: 0.5;
		stroke-dasharray: 2 2;
	}
	.grid {
		stroke: var(--rule);
		stroke-width: 0.4;
	}
	.stem {
		stroke: var(--ink-faint);
		stroke-width: 0.6;
		opacity: 0.5;
	}
	.mark {
		fill: var(--hue);
		fill-opacity: 0.7;
		stroke: var(--paper);
		stroke-width: 0.6;
	}
	/* B series sits hollow so an A mark behind it stays readable when they cross */
	.mark-b {
		fill: var(--paper);
		fill-opacity: 0.92;
		stroke: var(--hue);
		stroke-width: 1.1;
	}
	.curve {
		fill: none;
		stroke: var(--hue);
		stroke-width: 1.1;
		stroke-opacity: 0.85;
		stroke-linejoin: round;
		stroke-linecap: round;
	}
	.axes {
		display: flex;
		justify-content: space-between;
		font-family: var(--mono);
		font-size: 0.66rem;
		color: var(--ink-faint);
		margin-top: 0.2rem;
		max-width: 22rem;
	}
	.ax-y {
		writing-mode: initial;
	}
	.caption {
		font-family: var(--mono);
		font-size: 0.66rem;
		color: var(--ink-faint);
		margin: 0.4rem 0 0;
		max-width: 22rem;
		line-height: 1.4;
	}
	.diag-key {
		color: var(--ink-muted);
	}
	.key-a,
	.key-b {
		color: var(--hue);
		font-weight: 600;
		margin-right: 0.5rem;
	}
</style>
