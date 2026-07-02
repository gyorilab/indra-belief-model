<script lang="ts">
	// The dataset-size dimension: the SAME models read ACROSS gold benchmarks,
	// x-ordered by gold size (log n). Each model is a line through its (n, error-F1)
	// points; the bootstrap 95% CI is a whisker at each point that visibly NARROWS as
	// n grows (precision), while the point itself MOVES (generalization). The headline
	// it makes concrete: a model rated high on a small, narrow gold (rasmachine_v1,
	// n=60) settles lower on a large, de-biased one (external-578) — the small-gold
	// mirage. HONEST: substrates differ in composition too, so a shift is "more AND
	// different data", not a within-set learning curve (which would be flat).
	import type { Generalization, GenModel } from '$lib/data/queries';

	let { data }: { data: Generalization } = $props();

	const W = 760,
		H = 384;
	const padL = 46,
		padR = 150,
		padT = 20,
		padB = 52;
	const plotL = padL,
		plotR = W - padR,
		plotT = padT,
		plotB = H - padB;
	const plotH = plotB - plotT;

	let hovered = $state<string | null>(null);

	const stripHost = (m: string) => m.replace(/^(bedrock|remote|local|google)-/, '');

	// Headline lines carry a persistent label + whiskers without hover: the steepest
	// movers (|delta| largest — most overrated/underrated by the small gold) plus the
	// production model (gemma-4-26b), so the mirage is legible at a glance.
	const HEADLINE = $derived(
		new Set(
			[
				...data.models.slice(0, 3).map((m) => m.model), // steepest droppers (sorted by delta asc)
				...data.models.slice(-1).map((m) => m.model), // steepest riser (if any)
				...data.models.filter((m) => /gemma-4-26b$/.test(m.model)).map((m) => m.model)
			].filter(Boolean)
		)
	);

	const lmin = $derived(Math.log10(Math.max(1, data.gold_min)));
	const lmax = $derived(Math.log10(Math.max(data.gold_min + 1, data.gold_max)));
	const ylo = $derived(Math.max(0, data.f1_min - 0.03));
	const yhi = $derived(Math.min(1, data.f1_max + 0.03));

	function xOf(n: number): number {
		if (lmax === lmin) return (plotL + plotR) / 2;
		return plotL + ((Math.log10(Math.max(1, n)) - lmin) / (lmax - lmin)) * (plotR - plotL);
	}
	function yOf(f1: number): number {
		return plotB - ((f1 - ylo) / (yhi - ylo || 1)) * plotH;
	}

	const isHot = (m: GenModel) => HEADLINE.has(m.model) || hovered === m.model;
	const linePath = (m: GenModel) =>
		m.points.map((p, i) => `${i ? 'L' : 'M'} ${xOf(p.gold_n).toFixed(1)} ${yOf(p.f1).toFixed(1)}`).join(' ');

	const yTicks = $derived.by(() => {
		const out: Array<{ y: number; label: string }> = [];
		for (let t = Math.ceil(ylo * 10) / 10; t <= yhi + 1e-9; t += 0.1)
			out.push({ y: yOf(t), label: t.toFixed(1) });
		return out;
	});

	// Right-edge labels (at the largest-n point of each hot line), de-collided by a
	// simple greedy vertical nudge so adjacent models don't overprint.
	const labels = $derived.by(() => {
		const hot = data.models.filter(isHot);
		const raw = hot
			.map((m) => {
				const last = m.points[m.points.length - 1];
				return { model: m.model, name: stripHost(m.model), x: xOf(last.gold_n), y: yOf(last.f1), f1: last.f1 };
			})
			.sort((a, b) => a.y - b.y);
		const MINGAP = 11;
		for (let i = 1; i < raw.length; i++)
			if (raw[i].y - raw[i - 1].y < MINGAP) raw[i].y = raw[i - 1].y + MINGAP;
		return raw;
	});

	const ariaLabel = $derived(
		`Error-detection F1 across ${data.substrates.length} gold benchmarks (n=${data.gold_min} to ${data.gold_max}) for ${data.models.length} models; ` +
			`each line is one model, CI whiskers narrow as n grows. Lines generally fall from the small gold to the large de-biased one.`
	);
</script>

<figure class="gen">
	<svg viewBox="0 0 {W} {H}" role="img" aria-label={ariaLabel}>
		{#each yTicks as t}
			<line x1={plotL} y1={t.y} x2={plotR} y2={t.y} stroke="var(--rule)" stroke-width="0.6" />
			<text x={plotL - 6} y={t.y + 3} text-anchor="end" class="tick">{t.label}</text>
		{/each}
		<text x={12} y={plotT + plotH / 2} class="axis-title" transform="rotate(-90 12 {plotT + plotH / 2})"
			>error-detection F1</text
		>

		<!-- substrate columns: a guide rule + n / name at the foot of each gold -->
		{#each data.substrates as s}
			{@const x = xOf(s.gold_n)}
			<line x1={x} y1={plotT} x2={x} y2={plotB} stroke="var(--ink-faint)" stroke-width="0.5" stroke-dasharray="2 3" opacity="0.6" />
			<text {x} y={plotB + 15} text-anchor="middle" class="tick col-n">n={s.gold_n}</text>
			<text {x} y={plotB + 26} text-anchor="middle" class="tick col-name">{s.label}</text>
		{/each}
		<text x={(plotL + plotR) / 2} y={plotB + 42} text-anchor="middle" class="axis-title"
			>gold benchmark · size n (log) — larger &amp; less curator-captured →</text
		>

		<!-- dominated/quiet lines first (thin, muted), hot lines on top -->
		{#each data.models as m (m.model)}
			{#if !isHot(m)}
				<path d={linePath(m)} fill="none" stroke="var(--ink-faint)" stroke-width="1" opacity="0.32" />
			{/if}
		{/each}

		<!-- wide invisible hit-areas so thin lines are hoverable -->
		{#each data.models as m (m.model)}
			<path
				d={linePath(m)}
				fill="none"
				stroke="transparent"
				stroke-width="12"
				role="button"
				tabindex="0"
				aria-label="{m.model}: error-F1 {m.points[0].f1.toFixed(2)} at n={m.points[0]
					.gold_n} → {m.points[m.points.length - 1].f1.toFixed(2)} at n={m.points[m.points.length - 1].gold_n}"
				onmouseenter={() => (hovered = m.model)}
				onmouseleave={() => (hovered = null)}
				onfocus={() => (hovered = m.model)}
				onblur={() => (hovered = null)}
			/>
		{/each}

		{#each data.models as m (m.model)}
			{#if isHot(m)}
				<!-- CI whiskers at each point: wide at small n, tight at large n -->
				{#each m.points as p}
					<line
						x1={xOf(p.gold_n)}
						y1={yOf(p.hi)}
						x2={xOf(p.gold_n)}
						y2={yOf(p.lo)}
						stroke="var(--ink-faint)"
						stroke-width="5"
						opacity="0.22"
					/>
				{/each}
				<path
					d={linePath(m)}
					fill="none"
					stroke={hovered === m.model ? 'var(--accent)' : 'var(--ink)'}
					stroke-width={hovered === m.model ? 2 : 1.4}
					opacity="0.95"
				/>
				{#each m.points as p}
					<circle cx={xOf(p.gold_n)} cy={yOf(p.f1)} r="3" fill={hovered === m.model ? 'var(--accent)' : 'var(--ink)'} />
				{/each}
			{/if}
		{/each}

		{#each labels as l (l.model)}
			<text x={plotR + 6} y={l.y + 3} text-anchor="start" class="lbl" class:hot={hovered === l.model}>{l.name}</text>
		{/each}
	</svg>
	<p class="caveat">
		Lines connect one model's error-F1 across gold benchmarks, ordered by size. They generally
		<strong>fall</strong> from the small, single-curator gold (n=60) to the large, de-biased one
		(n=578) — the small-gold mirage — while the CI whisker <strong>narrows</strong> with n
		(precision). The x-axis mixes size with composition, so a drop is “more <em>and</em> less-captured
		data”, not a within-set learning curve.
	</p>
</figure>

<style>
	.gen {
		margin: 0;
	}
	.gen svg {
		display: block;
		width: 100%;
		height: auto;
		overflow: visible;
	}
	.tick {
		font-family: var(--mono);
		font-size: 8.5px;
		fill: var(--ink-faint);
	}
	.col-n {
		fill: var(--ink-muted);
		font-size: 9px;
	}
	.col-name {
		fill: var(--ink-faint);
		font-size: 7.5px;
	}
	.axis-title {
		font-family: var(--mono);
		font-size: 8.5px;
		fill: var(--ink-muted);
		letter-spacing: 0.03em;
	}
	.lbl {
		font-family: var(--mono);
		font-size: 9px;
		fill: var(--ink-muted);
		paint-order: stroke;
		stroke: var(--paper);
		stroke-width: 2.5px;
		stroke-linejoin: round;
	}
	.lbl.hot {
		fill: var(--accent);
	}
	path[role='button'] {
		cursor: pointer;
		outline: none;
	}
	path[role='button']:focus-visible {
		stroke: var(--accent);
		stroke-opacity: 0.25;
	}
	.caveat {
		font-family: var(--mono);
		font-size: 0.72rem;
		line-height: 1.5;
		color: var(--ink-faint);
		margin: 0.6rem 0 0;
		max-width: 64ch;
	}
</style>
