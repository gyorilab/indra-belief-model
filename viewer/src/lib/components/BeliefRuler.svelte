<script lang="ts">
	// Shared 0–1 belief ruler for model-vs-model comparison — the same geometry
	// the app teaches everywhere (BeliefPrimitive's spectrum axis): position is
	// the encoding. Two model markers ride ABOVE the axis (A in --a-hue, B in
	// --b-hue) with the gap between them drawn as a literal length — that length
	// IS the disagreement. The INDRA prior is a hollow mark ON the axis; the human
	// gold verdict, when present, is a --gold-hue tick at the end it sided with.
	//
	// When the two beliefs nearly coincide the markers fan vertically (A up, B
	// down at the same x) so neither model is occluded and "they agree" reads as a
	// tight vertical pair rather than a vanished dot.
	//
	// `compact` renders a fluid inline micro-ruler (one per cohort row) and is
	// DECORATIVE — it reinforces the row's own A/B/gold cells, so it carries no
	// aria. Full mode is a labelled continuous scale with ends and a legend.

	let {
		a = null,
		b = null,
		prior = null,
		gold = null,
		aLabel = 'A',
		bLabel = 'B',
		compact = false
	}: {
		a?: number | null;
		b?: number | null;
		prior?: number | null;
		gold?: 'correct' | 'incorrect' | null;
		aLabel?: string;
		bLabel?: string;
		compact?: boolean;
	} = $props();

	const W = $derived(compact ? 116 : 320);
	const M = $derived(compact ? 6 : 14);
	const axisY = $derived(compact ? 13 : 30);
	const dotLane = $derived(compact ? 6 : 16);
	const H = $derived(compact ? 22 : 50);
	const dotR = $derived(compact ? 3 : 4.5);
	const fan = $derived(compact ? 2.6 : 3.2);

	const track = $derived(W - 2 * M);
	function tickX(s: number): number {
		return M + Math.max(0, Math.min(1, s)) * track;
	}

	const ax = $derived(a == null ? null : tickX(a));
	const bx = $derived(b == null ? null : tickX(b));
	// "agree" = markers within a dot-width; fan them so neither is painted over.
	const close = $derived(ax != null && bx != null && Math.abs(ax - bx) < 2 * dotR);
	const aY = $derived(close ? dotLane - fan : dotLane);
	const bY = $derived(close ? dotLane + fan : dotLane);
	const goldX = $derived(gold == null ? null : gold === 'correct' ? 0.95 : 0.05);

	function clause(label: string, x: number | null): string {
		return x == null ? `${label} not scored` : `${label} ${x.toFixed(2)}`;
	}
	const ariaLabel = $derived(
		`belief 0 to 1: ${clause(aLabel, a)}, ${clause(bLabel, b)}` +
			(a != null && b != null ? `, gap ${Math.abs(a - b).toFixed(2)}` : '') +
			(prior != null ? `, INDRA prior ${prior.toFixed(2)}` : '') +
			(gold != null ? `, human gold ${gold}` : '')
	);
</script>

<span
	class="bruler"
	class:compact
	role={compact ? undefined : 'img'}
	aria-label={compact ? undefined : ariaLabel}
	aria-hidden={compact ? 'true' : undefined}
>
	<svg viewBox="0 0 {W} {H}" preserveAspectRatio="xMidYMid meet">
		<!-- axis + endpoint ticks -->
		<line x1={M} y1={axisY} x2={W - M} y2={axisY} stroke="var(--ink)" stroke-width="1" />
		<line x1={M} y1={axisY - 3} x2={M} y2={axisY + 3} stroke="var(--ink-faint)" />
		<line x1={W - M} y1={axisY - 3} x2={W - M} y2={axisY + 3} stroke="var(--ink-faint)" />
		<!-- the A↔B gap, drawn as a length when the two beliefs differ -->
		{#if ax != null && bx != null && !close}
			<line x1={ax} y1={dotLane} x2={bx} y2={dotLane} stroke="var(--ink-faint)" stroke-width="1" opacity="0.5" />
		{/if}

		<!-- INDRA prior: hollow mark on the axis -->
		{#if prior != null}
			<circle cx={tickX(prior)} cy={axisY} r={dotR} fill="var(--paper)" stroke="var(--ink)" stroke-width="1.3" />
		{/if}

		<!-- gold: a tick at the end the human sided with -->
		{#if goldX != null}
			<line x1={tickX(goldX)} y1={axisY + 1} x2={tickX(goldX)} y2={axisY + (compact ? 5 : 8)} stroke="var(--gold-hue)" stroke-width="1.8" />
		{/if}

		<!-- model markers: filled, above the axis (position = belief) -->
		{#if ax != null}
			<line x1={ax} y1={aY} x2={ax} y2={axisY - 1} stroke="var(--a-hue)" stroke-width="0.7" opacity="0.4" />
			<circle cx={ax} cy={aY} r={dotR} fill="var(--a-hue)" />
		{/if}
		{#if bx != null}
			<line x1={bx} y1={bY} x2={bx} y2={axisY - 1} stroke="var(--b-hue)" stroke-width="0.7" opacity="0.4" />
			<circle cx={bx} cy={bY} r={dotR} fill="var(--b-hue)" />
		{/if}
	</svg>
	{#if !compact}
		<span class="ends">
			<span>doubt</span>
			<span class="lg">
				<span class="sw a"></span>{clause(aLabel, a)}
				<span class="sw b"></span>{clause(bLabel, b)}
				{#if prior != null}<span class="sw indra"></span>INDRA {prior.toFixed(2)}{/if}
				{#if gold != null}<span class="sw gold"></span>gold {gold}{/if}
			</span>
			<span>trust</span>
		</span>
	{/if}
</span>

<style>
	.bruler {
		display: block;
	}
	.bruler svg {
		display: block;
		width: 100%;
		height: auto;
	}
	.bruler.compact {
		min-width: 104px;
	}
	.ends {
		display: flex;
		justify-content: space-between;
		align-items: center;
		gap: 0.6rem;
		margin-top: 0.15rem;
		font-size: 0.62rem;
		font-variant: small-caps;
		letter-spacing: 0.04em;
		color: var(--ink-faint);
	}
	.lg {
		display: inline-flex;
		flex-wrap: wrap;
		align-items: center;
		justify-content: center;
		gap: 0.25rem;
		min-width: 0;
		font-variant: normal;
		letter-spacing: 0;
		color: var(--ink-muted);
	}
	.lg .sw {
		display: inline-block;
		width: 0.5rem;
		height: 0.5rem;
		border-radius: 50%;
		margin-left: 0.5rem;
	}
	.lg .sw:first-child {
		margin-left: 0;
	}
	.lg .sw.a {
		background: var(--a-hue);
	}
	.lg .sw.b {
		background: var(--b-hue);
	}
	.lg .sw.indra {
		background: var(--paper);
		border: 1.3px solid var(--ink);
	}
	.lg .sw.gold {
		background: var(--gold-hue);
		border-radius: 1px;
	}
</style>
