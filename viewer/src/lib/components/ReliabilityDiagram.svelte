<!--
  ReliabilityDiagram (E5) — the instrument's confession.

  x = mean predicted belief, y = empirical correct-rate, diagonal = perfect
  calibration. A mark BELOW the diagonal = over-confidence; ABOVE = under-
  confidence (the documented C0 finding: belief ~0.46 but ~65-72% correct shows
  as a mark sitting high above the line). Marks are sized by bin-n so a sparse
  bin cannot lie about a confident curve. Reads metrics.json bins VERBATIM —
  recomputes nothing.
-->
<script lang="ts">
	import type { ReliabilityBin } from '$lib/data/types';

	let {
		bins,
		n,
		hue = 'var(--accent)',
		label = ''
	}: { bins: ReliabilityBin[]; n: number; hue?: string; label?: string } = $props();

	// occupied bins only carry a point; the others still anchor the x-axis grid.
	const pts = $derived(
		bins
			.filter((b) => b.n > 0 && b.mean_pred != null && b.empirical != null)
			.map((b) => ({
				x: b.mean_pred as number,
				y: b.empirical as number,
				n: b.n,
				gap: (b.empirical as number) - (b.mean_pred as number)
			}))
	);
	const maxBinN = $derived(Math.max(1, ...pts.map((p) => p.n)));

	// SVG viewbox in calibration space [0,1]×[0,1], y inverted (1 at top).
	const PAD = 6; // % padding inside the 0..100 box for stroke breathing room
	function px(v: number): number {
		return PAD + v * (100 - 2 * PAD);
	}
	function py(v: number): number {
		return PAD + (1 - v) * (100 - 2 * PAD);
	}
	// radius 1.6%..5% of the box, by bin share — sparse bins read as small.
	function rOf(bn: number): number {
		return 1.6 + 3.4 * Math.sqrt(bn / maxBinN);
	}
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
				r={rOf(p.n)}
				style="--hue:{hue}"
			>
				<title>{`predicted ${fmtPct(p.x)} · empirical ${fmtPct(p.y)} · n=${p.n} · ${p.gap >= 0 ? 'under' : 'over'}-confident by ${fmtPct(Math.abs(p.gap))}`}</title>
			</circle>
		{/each}
	</svg>
	<div class="axes">
		<span class="ax-y">empirical correct-rate →</span>
		<span class="ax-x">mean predicted belief →</span>
	</div>
	<p class="caption">
		<span class="diag-key">— perfect</span> · marks sized by bin-n · above the line =
		under-confident, below = over-confident · n={n.toLocaleString('en-US')}
	</p>
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
</style>
