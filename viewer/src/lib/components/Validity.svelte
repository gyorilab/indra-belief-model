<script lang="ts">
	import type { Validity, ResidualDistribution } from '$lib/data/queries';
	import { bucketLabel } from '$lib/components/BeliefPrimitive.svelte';
	import { residualBraille } from '$lib/residuals';

	let {
		v,
		residuals
	}: {
		v: Validity;
		residuals: ResidualDistribution | null;
	} = $props();

	const verdictTotal = $derived(v.verdicts.reduce((s, vd) => s + vd.n, 0));
	const verdictPct = (n: number) => (verdictTotal === 0 ? 0 : (n / verdictTotal) * 100);

	function verdictDisplayName(name: string): string {
		return name === 'correct'
			? 'supported'
			: name === 'incorrect'
				? 'contradicted'
				: name === 'unscored'
					? 'unscored'
					: name;
	}

	/** "How do we compare to INDRA's priors?" — the calibration headline. */
	const indraLine = $derived.by(() => {
		const c = v.calibration;
		if (c.mae == null) {
			return { glyph: '—', direction: 'na' as const, headline: 'no INDRA-comparable beliefs in this run', detail: '' };
		}
		const bias = c.bias ?? 0;
		const oneSided = c.mae > 0 && Math.abs(Math.abs(bias) - c.mae) < 1e-3;
		const n = c.n;
		if (Math.abs(bias) < 0.05) {
			return {
				glyph: '≈',
				direction: 'eq' as const,
				headline: 'closely matched',
				detail: `MAE ${c.mae.toFixed(2)} · n=${n}`
			};
		}
		const dir = bias > 0 ? 'over' : 'under';
		const arrow = bias > 0 ? '▲' : '▼';
		const oneSidedNote =
			oneSided && c.mae > 0.001
				? `every score landed ${bias < 0 ? 'below' : 'above'} INDRA`
				: `MAE ${c.mae.toFixed(2)}`;
		return {
			glyph: arrow,
			direction: bias > 0 ? ('up' as const) : ('down' as const),
			headline: `${dir}-confident by ${Math.abs(bias).toFixed(2)}`,
			detail: `${oneSidedNote} · n=${n}`
		};
	});

	/** Show the histogram only when there are enough points for the shape to mean
	 *  something. Below ~30 the bars are misleading — the prose carries the bias. */
	const SHOW_HISTOGRAM_THRESHOLD = 30;
	const showHistogram = $derived(residuals != null && residuals.n_total >= SHOW_HISTOGRAM_THRESHOLD);

	// ── Perceptual encodings ────────────────────────────────────────────────────
	// Disagreement magnitude → bar length; bias direction → bar colour.
	// brick = we scored below INDRA (more skeptical), green = above, neutral = ≈.
	function biasTone(bias: number | null): 'under' | 'over' | 'eq' {
		if (bias == null || Math.abs(bias) < 0.05) return 'eq';
		return bias < 0 ? 'under' : 'over';
	}
	function maePct(mae: number | null): number {
		if (mae == null) return 0;
		return Math.max(0, Math.min(100, mae * 100));
	}

	// Weakest slices as ranked bars (worst MAE first); queries already sorts these.
	const weakType = $derived(v.byIndraType.slice(0, 6));
	const weakSource = $derived(v.bySourceApi.slice(0, 6));

	// Confidence calibration buckets (high / medium / low) as bars.
	const conf = $derived(v.confidenceCalibration.filter((r) => r.n > 0));

	// Bucket distribution (report taxonomy) — the categorical "what the model saw".
	const bucketTotal = $derived(v.buckets.reduce((s, b) => s + b.n, 0));
	const topBuckets = $derived(v.buckets.slice(0, 8));

	// Residual histogram bins, each coloured by which side of zero it sits on.
	// our − INDRA: left of 0 = we scored below INDRA, right = above.
	const residualBars = $derived.by(() => {
		if (!residuals) return [];
		const bins = residuals.bins;
		const n = bins.length;
		const max = bins.reduce((m, c) => Math.max(m, c), 0) || 1;
		const W = 320;
		const step = W / n;
		const mid = (n - 1) / 2;
		return bins.map((count, i) => ({
			x: i * step + 0.6,
			w: step - 1.2,
			h: count / max,
			count,
			tone: i < mid ? 'under' : i > mid ? 'over' : 'eq'
		}));
	});
</script>

<section class="validity">
	<h2 class="v-h">
		how is the system doing in this run?
		<span class="v-run-id" title="run_id">{v.run_id.slice(0, 8)}</span>
		<span class="v-run-id" title="model">{v.model}</span>
	</h2>

	<!-- One MAE bar: length = disagreement vs INDRA (0–1), colour = direction. -->
	{#snippet maeBar(name: string, mae: number | null, bias: number | null, n: number)}
		<div class="bar-row">
			<span class="bar-name" title={name}>{name}</span>
			<span class="bar-track" aria-hidden="true"
				><span class="bar-fill tone-{biasTone(bias)}" style:width="{maePct(mae)}%"></span></span
			>
			<span class="bar-val">{mae == null ? '—' : mae.toFixed(2)}</span>
			<span class="bar-n">n={n}</span>
		</div>
	{/snippet}

	<!-- Calibration vs INDRA: the headline verdict + the shape of disagreement -->
	<div class="cal">
		<div class="v-line cal-headline">
			<span class="v-line-label">vs INDRA's priors</span>
			<span class="v-glyph v-glyph-{indraLine.direction}">{indraLine.glyph}</span>
			<span class="v-headline">{indraLine.headline}</span>
			{#if indraLine.detail}<span class="muted v-detail">· {indraLine.detail}</span>{/if}
		</div>
		{#if showHistogram}
			<figure class="res-hist">
				<span class="v-braille sr-only">{residualBraille(residuals!.bins)}</span>
				<svg viewBox="0 0 320 66" class="res-svg" role="img" aria-label="residual histogram, our minus INDRA, n={residuals!.n_total}">
					<line x1="160" y1="0" x2="160" y2="56" stroke="var(--ink-faint)" stroke-width="0.6" stroke-dasharray="2 2" />
					{#each residualBars as b}
						{#if b.h > 0}
							<rect class="res-bar res-{b.tone}" x={b.x} y={56 - b.h * 54} width={b.w} height={b.h * 54}>
								<title>{b.count}</title>
							</rect>
						{/if}
					{/each}
					<line x1="0" y1="56" x2="320" y2="56" stroke="var(--rule)" stroke-width="1" />
				</svg>
				<figcaption class="res-axis">
					<span>↞ we scored below INDRA</span>
					<span class="res-zero">0</span>
					<span>above ↠</span>
				</figcaption>
			</figure>
		{/if}
	</div>

	{#if verdictTotal > 0}
		<div class="v-line v-line-block">
			<span class="v-line-label" title="our scorer's classification of each evidence">per-evidence verdicts</span>
			<div class="v-pillbar-wrap">
				<div class="v-pillbar" role="img" aria-label="verdict distribution">
					{#each v.verdicts as vd}
						{@const pct = verdictPct(vd.n)}
						{#if pct > 0}
							<span class="v-pill v-pill-{vd.verdict}" style:width="{pct}%" title="{verdictDisplayName(vd.verdict)}: {vd.n} of {verdictTotal} ({pct.toFixed(1)}%)"></span>
						{/if}
					{/each}
				</div>
				<div class="v-pill-caption">
					{#each v.verdicts as vd, i}{#if i > 0}<span class="muted"> · </span>{/if}<span class="v-pill-tag v-pill-tag-{vd.verdict}">{verdictDisplayName(vd.verdict)}</span> <span class="v-pill-num">{vd.n}</span>{/each}
					<span class="muted"> · n={verdictTotal}</span>
				</div>
				<p class="v-explain">each evidence sentence is judged separately — a statement with multiple evidences contributes multiple counts</p>
			</div>
		</div>
	{/if}

	{#if topBuckets.length > 0}
		<div class="v-line v-line-block">
			<span class="v-line-label" title="report-taxonomy bucket for each scored evidence">evidence buckets</span>
			<div class="v-pillbar-wrap">
				<ul class="bucket-list">
					{#each topBuckets as b (b.bucket)}
						{@const pct = bucketTotal === 0 ? 0 : (b.n / bucketTotal) * 100}
						<li class="bucket-row">
							<span class="bucket-name" title={b.bucket}>{bucketLabel(b.bucket)}</span>
							<span class="bucket-track" aria-hidden="true"><span class="bucket-fill" style:width="{pct}%"></span></span>
							<span class="bucket-n">{b.n}</span>
						</li>
					{/each}
				</ul>
				<p class="v-explain">the report taxonomy — what each evidence sentence actually said about the claim</p>
			</div>
		</div>
	{/if}

	{#if weakType.length > 0 || weakSource.length > 0}
		<div class="v-line v-line-block">
			<span class="v-line-label">weakest by slice</span>
			<p class="v-explain">bar = mean disagreement vs INDRA (0–1); brick = we scored below, green = above</p>
			<div class="bars-cols">
				{#if weakType.length > 0}
					<div class="bars">
						<span class="bars-h">by indra_type</span>
						{#each weakType as s (s.value)}
							{@render maeBar(s.value, s.mae, s.bias, s.n)}
						{/each}
					</div>
				{/if}
				{#if weakSource.length > 0}
					<div class="bars">
						<span class="bars-h">by source_api</span>
						{#each weakSource as s (s.value)}
							{@render maeBar(s.value, s.mae, s.bias, s.n)}
						{/each}
					</div>
				{/if}
			</div>
		</div>
	{/if}

	{#if conf.length > 0}
		<div class="v-line v-line-block">
			<span class="v-line-label">confidence calibration</span>
			<div class="bars bars-conf">
				{#each conf as row (row.confidence)}
					{@render maeBar(row.confidence, row.mae, row.bias, row.n)}
				{/each}
			</div>
			<p class="v-explain">disagreement vs INDRA's published belief, split by the model's own stated confidence</p>
		</div>
	{/if}

	{#if v.unavailable.length > 0}
		<p class="v-unavailable">not available in monolithic export: {v.unavailable.join('; ')}</p>
	{/if}
</section>

<style>
	.validity {
		margin: 0 0 2.5rem;
	}
	.v-h {
		font-family: var(--serif);
		font-size: 1.15rem;
		font-weight: 400;
		color: var(--ink);
		margin: 0 0 1rem;
		display: flex;
		align-items: baseline;
		gap: 0.6rem;
	}
	.v-run-id {
		font-family: var(--mono);
		font-size: 0.72rem;
		color: var(--ink-faint);
	}
	.v-line {
		display: flex;
		flex-wrap: wrap;
		gap: 0.6rem;
		align-items: baseline;
		padding: 0.4rem 0;
		border-bottom: 1px dotted var(--rule);
		font-family: var(--mono);
		font-size: 0.86rem;
	}
	.v-line-block {
		flex-direction: column;
		align-items: flex-start;
		gap: 0.3rem;
	}
	.v-line:last-child {
		border-bottom: none;
	}

	.v-line-label {
		flex-basis: 14rem;
		flex-shrink: 0;
		color: var(--ink-muted);
		text-transform: lowercase;
		letter-spacing: 0.02em;
		font-size: 0.78rem;
	}
	/* In block (column-flex) lines, flex-basis: 14rem becomes 14rem of *height*
	   rather than width — which produced a giant vertical gap. Reset to auto. */
	.v-line-block .v-line-label {
		flex-basis: auto;
	}

	.v-glyph {
		font-family: var(--mono);
		font-weight: 500;
		min-width: 1.2rem;
		text-align: center;
	}
	.v-glyph-up { color: var(--ok-green); }
	.v-glyph-down { color: var(--accent); }
	.v-glyph-eq { color: var(--ink-muted); }
	.v-glyph-na { color: var(--ink-faint); }

	.v-headline {
		color: var(--ink);
		font-variant-numeric: tabular-nums;
	}
	.v-detail {
		font-size: 0.78rem;
	}

	/* Verdict pillbar */
	.v-pillbar-wrap {
		display: flex;
		flex-direction: column;
		gap: 0.2rem;
		width: 100%;
		max-width: 480px;
	}
	.v-pillbar {
		display: flex;
		width: 100%;
		height: 16px;
		border: 1px solid var(--rule);
		overflow: hidden;
	}
	.v-pill {
		display: block;
		min-width: 2px;
	}
	.v-pill-correct { background: var(--ok-green); }
	.v-pill-incorrect { background: var(--accent); }
	.v-pill-abstain { background: var(--ink-faint); }
	.v-pill-unscored { background: var(--rule); }
	.v-pill-caption {
		font-family: var(--mono);
		font-size: 0.78rem;
		font-variant-numeric: tabular-nums;
	}
	.v-pill-tag-correct { color: var(--ok-green); }
	.v-pill-tag-incorrect { color: var(--accent); }
	.v-pill-tag-abstain { color: var(--ink-muted); }
	.v-pill-tag-unscored { color: var(--ink-faint); }
	.v-pill-num { color: var(--ink); font-weight: 500; }
	.v-explain {
		font-family: var(--serif);
		font-style: italic;
		font-size: 0.78rem;
		color: var(--ink-faint);
		margin: 0.2rem 0 0;
	}

	/* Bucket distribution */
	.bucket-list {
		list-style: none;
		margin: 0;
		padding: 0;
		display: flex;
		flex-direction: column;
		gap: 0.18rem;
		width: 100%;
	}
	.bucket-row {
		display: grid;
		grid-template-columns: minmax(8rem, 12rem) 1fr 2.6rem;
		align-items: center;
		gap: 0.5rem;
		font-family: var(--mono);
		font-size: 0.76rem;
		color: var(--ink);
	}
	.bucket-name {
		color: var(--ink);
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}
	.bucket-track {
		position: relative;
		height: 0.62rem;
		background: var(--rule);
		border-radius: 1px;
		overflow: hidden;
	}
	.bucket-fill {
		position: absolute;
		left: 0;
		top: 0;
		bottom: 0;
		min-width: 1px;
		border-radius: 1px;
		background: var(--ink-muted);
		transition: width 200ms ease-out;
	}
	.bucket-n {
		color: var(--ink);
		font-weight: 600;
		font-variant-numeric: tabular-nums;
		text-align: right;
	}

	/* Calibration hero: headline + residual shape */
	.cal {
		display: flex;
		flex-direction: column;
		gap: 0.4rem;
		padding: 0.4rem 0 0.7rem;
		border-bottom: 1px dotted var(--rule);
	}
	.cal-headline {
		border-bottom: none;
		padding-bottom: 0;
	}
	.res-hist {
		margin: 0;
		max-width: 22rem;
	}
	.res-svg {
		width: 100%;
		height: auto;
		display: block;
	}
	.res-bar {
		transition: opacity 120ms ease-out;
	}
	.res-under {
		fill: var(--accent);
	}
	.res-over {
		fill: var(--ok-green);
	}
	.res-eq {
		fill: var(--ink-faint);
	}
	.res-axis {
		display: flex;
		justify-content: space-between;
		font-family: var(--mono);
		font-size: 0.64rem;
		color: var(--ink-faint);
		margin-top: 0.1rem;
	}
	.res-zero {
		color: var(--ink-muted);
	}

	/* MAE bars (weakest slices + confidence calibration) */
	.bars-cols {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(15rem, 1fr));
		gap: 0.5rem 1.6rem;
		width: 100%;
	}
	.bars {
		display: flex;
		flex-direction: column;
		gap: 0.18rem;
		min-width: 0;
	}
	.bars-h {
		font-family: var(--mono);
		font-size: 0.7rem;
		color: var(--ink-muted);
		margin-bottom: 0.15rem;
		text-transform: lowercase;
		letter-spacing: 0.02em;
	}
	.bars-conf {
		max-width: 24rem;
	}
	.bar-row {
		display: grid;
		grid-template-columns: minmax(5rem, 9rem) 1fr 2.6rem auto;
		align-items: center;
		gap: 0.5rem;
		font-family: var(--mono);
		font-size: 0.76rem;
		color: var(--ink);
		padding: 0.08rem 0.12rem;
		border-radius: 2px;
	}
	.bar-name {
		color: var(--ink);
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}
	.bar-track {
		position: relative;
		height: 0.62rem;
		background: var(--rule);
		border-radius: 1px;
		overflow: hidden;
	}
	.bar-fill {
		position: absolute;
		left: 0;
		top: 0;
		bottom: 0;
		min-width: 1px;
		border-radius: 1px;
		transition: width 200ms ease-out;
	}
	.bar-fill.tone-under {
		background: var(--accent);
	}
	.bar-fill.tone-over {
		background: var(--ok-green);
	}
	.bar-fill.tone-eq {
		background: var(--ink-muted);
	}
	.bar-val {
		color: var(--ink);
		font-weight: 600;
		font-variant-numeric: tabular-nums;
		text-align: right;
	}
	.bar-n {
		color: var(--ink-faint);
		font-variant-numeric: tabular-nums;
		white-space: nowrap;
	}

	.v-unavailable {
		margin: 0.9rem 0 0;
		font-family: var(--mono);
		font-size: 0.72rem;
		color: var(--ink-faint);
		line-height: 1.5;
	}

	.sr-only {
		position: absolute;
		width: 1px;
		height: 1px;
		padding: 0;
		margin: -1px;
		overflow: hidden;
		clip: rect(0, 0, 0, 0);
		white-space: nowrap;
		border: 0;
	}

	.muted { color: var(--ink-faint); }
</style>
