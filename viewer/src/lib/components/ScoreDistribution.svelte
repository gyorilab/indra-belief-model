<script lang="ts">
	import {
		PAPER_SCORE_TOP_PILES,
		paperArmColorVar,
		type PaperLiteralLoad,
		type PaperLiteralArm
	} from '$lib/data/paper-literal';
	import { residualPath, residualBraille } from '$lib/residuals';

	let { data }: { data: PaperLiteralLoad } = $props();

	// The fixed all-sources-specific panel size — the spec's `/1689` denominator.
	// Deliberately a constant, NOT sum(scoreBins): LLM joins drop unmatched rows so
	// per-arm bin totals differ, and distinctScores is measured against this panel.
	const PANEL_N = 1689;

	// Small-multiple geometry (single viewBox reused per arm). The ridge is drawn
	// by the shared residuals bin→path helper; everything is laid out in a plot
	// group translated by (PAD_X, PAD_TOP) so score x ∈ [0,1] maps to [0, PLOT_W]
	// and the [0,1] baseline sits at y = PLOT_H.
	const VB_W = 220;
	const VB_H = 72;
	const PAD_X = 10;
	const PAD_TOP = 6;
	const PLOT_W = VB_W - PAD_X * 2; // 200
	const PLOT_H = 40;
	const TICK_BOT = PLOT_H + 4;
	const ORIENT_TICK_BOT = PLOT_H + 2;
	const VALUE_Y = PLOT_H + 11;
	const COUNT_Y = PLOT_H + 20;
	const ORIENT_TICKS = [0, 0.5, 1] as const;

	type Anchor = 'start' | 'middle' | 'end';

	// The two share-of-panel thresholds that class a cell's shape. The per-cell
	// description AND the summary below read this one function off the same
	// measurement, so the summary can no longer describe buckets that leave a cell
	// out: a two-bucket "RF arms / reader arms" legend covered 7 of 8 and silently
	// dropped `INDRA CoGEx hybrid`, which the cells themselves draw as a third shape.
	const NEAR_CONTINUOUS_SHARE = 0.5;
	const COARSE_SHARE = 0.1;
	type ShapeWord = 'near-continuous' | 'coarse-grained' | 'piled at a few exact scores';
	/** Summary order: finest grain first, so the rows read continuous → piled. */
	const SHAPE_ORDER: readonly ShapeWord[] = [
		'near-continuous',
		'coarse-grained',
		'piled at a few exact scores'
	];

	function shapeWordFor(distinctScores: number): ShapeWord {
		const share = distinctScores / PANEL_N;
		if (share >= NEAR_CONTINUOUS_SHARE) return 'near-continuous';
		if (share >= COARSE_SHARE) return 'coarse-grained';
		return 'piled at a few exact scores';
	}

	interface PileView {
		x: number;
		value: number;
		count: number;
		anchor: Anchor;
		valueLabel: string;
	}

	interface ArmView {
		arm: PaperLiteralArm;
		/** '' when scoreBins is empty (degraded arm) — residualPath returns '' there. */
		path: string;
		hasShape: boolean;
		piles: PileView[];
		braille: string;
		/** Shared arm→color token — paper/port in --accent, every LLM in --blocked. */
		color: string;
		desc: string;
		shape: ShapeWord;
		/** This arm's OWN scored rows (sum of its bins) — the pile-share denominator. */
		rows: number;
		/** Rows sitting on this arm's biggest exact values. */
		topPileRows: number;
	}

	/** One shape row of the summary: which cells are in it, and their spread. */
	interface ShapeTier {
		word: ShapeWord;
		/**
		 * The member cells' ON-SCREEN names. Called `displays`, not `labels`, because
		 * every arm here also carries a frozen `point_metrics` join key called
		 * `label`, and a field named `labels` in a render position is exactly how
		 * that key has reached the screen before.
		 */
		displays: string[];
		minDistinct: number;
		maxDistinct: number;
		/**
		 * Share of each member's own rows carried by its biggest exact values — null
		 * unless every member reported a full pile set, so a degraded arm suppresses
		 * the claim rather than quoting a share measured over fewer piles.
		 */
		minTopShare: number | null;
		maxTopShare: number | null;
	}

	function anchorFor(value: number): Anchor {
		if (value <= 0.05) return 'start';
		if (value >= 0.95) return 'end';
		return 'middle';
	}

	function buildView(arm: PaperLiteralArm): ArmView {
		// Consume server geometry as-is: residualPath is bin-index based, so the
		// 40-bin [0,1] scoreBins map directly across [0, PLOT_W]. No client re-binning.
		const path = residualPath(arm.scoreBins, PLOT_W, PLOT_H);
		const piles: PileView[] = arm.scoreTopPiles.map((p) => ({
			x: p.value * PLOT_W,
			value: p.value,
			count: p.count,
			anchor: anchorFor(p.value),
			valueLabel: p.value.toFixed(2)
		}));
		const pileDesc =
			piles.length > 0
				? piles.map((p) => `${p.count} at ${p.valueLabel}`).join(', ')
				: 'none recorded';
		// Derive the shape from the MEASUREMENT, never from arm.kind. Keying this on
		// the category label made the CoGEx cell contradict the very count printed
		// beside it (1,176 distinct scores described as "a few exact scores").
		const shapeWord = shapeWordFor(arm.distinctScores);
		// The arm's own scored rows, NOT PANEL_N: an LLM join that dropped rows must
		// not inflate the share of its panel that the biggest exact values carry.
		const rows = arm.scoreBins.reduce((total, count) => total + count, 0);
		return {
			arm,
			path,
			hasShape: path.length > 0,
			piles,
			braille: residualBraille(arm.scoreBins),
			color: paperArmColorVar(arm.kind),
			desc: `${arm.display}: ${shapeWord}. It gives ${arm.distinctScores.toLocaleString()} different scores across the ${PANEL_N.toLocaleString()} statements, on a scale from 0 to 1. The exact values shared by the most statements: ${pileDesc}.`,
			shape: shapeWord,
			rows,
			topPileRows: piles.reduce((total, pile) => total + pile.count, 0)
		};
	}

	const arms = $derived<ArmView[]>(data.status === 'ok' ? data.arms.map(buildView) : []);

	// Every cell lands in exactly one row, so the summary's coverage is structural
	// rather than a claim someone has to re-check against the artifact.
	const tiers = $derived<ShapeTier[]>(
		SHAPE_ORDER.flatMap((word) => {
			const members = arms.filter((view) => view.shape === word);
			if (members.length === 0) return [];
			const distinct = members.map((view) => view.arm.distinctScores);
			const measured = members.every(
				(view) => view.rows > 0 && view.piles.length === PAPER_SCORE_TOP_PILES
			);
			const shares = members.map((view) => view.topPileRows / view.rows);
			return [
				{
					word,
					displays: members.map((view) => view.arm.display),
					minDistinct: Math.min(...distinct),
					maxDistinct: Math.max(...distinct),
					minTopShare: measured ? Math.min(...shares) : null,
					maxTopShare: measured ? Math.max(...shares) : null
				}
			];
		})
	);

	/** The one cell in the LLM column that is not a reader arm; see paper-literal.ts. */
	const NON_READER_LLM_ARM_ID = 'indra-cogex-hybrid';
	const nonReader = $derived(arms.find((view) => view.arm.id === NON_READER_LLM_ARM_ID) ?? null);
	/**
	 * The share of this arm's rows carried by its biggest exact values, DERIVED.
	 * The sentence used to assert "no dominant pile" as a literal, which is true
	 * on today's data but would contradict its own shape word if the distribution
	 * ever moved into the piled tier.
	 */
	const nonReaderTopShare = $derived(
		nonReader && nonReader.rows > 0 && nonReader.piles.length > 0
			? { pct: (nonReader.topPileRows / nonReader.rows) * 100, n: nonReader.piles.length }
			: null
	);

	/** `a` when the tier holds one value, `a–b` otherwise — never `1,176–1,176`. */
	function countRange(min: number, max: number): string {
		return min === max ? min.toLocaleString() : `${min.toLocaleString()}–${max.toLocaleString()}`;
	}
	/** A share that rounds to 0% still holds rows, so it prints `<1`, never `0`. */
	function pctLabel(share: number): string {
		const pct = share * 100;
		return pct > 0 && pct < 1 ? '<1' : String(Math.round(pct));
	}
	function shareRange(min: number, max: number): string {
		const low = pctLabel(min);
		const high = pctLabel(max);
		return low === high ? `${low}%` : `${low}–${high}%`;
	}
</script>

{#if data.status !== 'ok'}
	<section class="sd sd-unavailable">
		<p class="sd-note">score distributions unavailable — {data.reason}</p>
	</section>
{:else}
	<section class="sd" aria-label="the raw spread of scores each model produces, one chart per model">
		<div class="sd-grid">
			{#each arms as view (view.arm.id)}
				<figure class="sd-cell" style="color:{view.color}">
					<svg
						class="sd-svg"
						viewBox="0 0 {VB_W} {VB_H}"
						role="img"
						aria-labelledby="sd-t-{view.arm.id} sd-d-{view.arm.id}"
						preserveAspectRatio="xMidYMid meet"
					>
						<title id="sd-t-{view.arm.id}"
							>{view.arm.display} — how its belief scores are spread out</title
						>
						<desc id="sd-d-{view.arm.id}">{view.desc}</desc>
						<g transform="translate({PAD_X} {PAD_TOP})">
							<!-- [0,1] baseline + orientation ticks at 0 / 0.5 / 1 -->
							<line class="sd-base" x1="0" y1={PLOT_H} x2={PLOT_W} y2={PLOT_H} />
							{#each ORIENT_TICKS as t}
								<line
									class="sd-base"
									x1={t * PLOT_W}
									y1={PLOT_H}
									x2={t * PLOT_W}
									y2={ORIENT_TICK_BOT}
								/>
							{/each}

							<!-- the distribution AS SHAPE: filled ridge from server scoreBins -->
							{#if view.hasShape}
								<path class="sd-ridge" d={view.path} />
							{:else}
								<text class="sd-empty" x={PLOT_W / 2} y={PLOT_H / 2} text-anchor="middle"
									>no scores</text
								>
							{/if}

							<!-- exact-value piles: tick + value + count at each pile's x -->
							{#each view.piles as pile (pile.value)}
								<line class="sd-pile" x1={pile.x} y1={PLOT_H} x2={pile.x} y2={TICK_BOT} />
								<text class="sd-pile-val" x={pile.x} y={VALUE_Y} text-anchor={pile.anchor}
									>{pile.valueLabel}</text
								>
								<text class="sd-pile-count" x={pile.x} y={COUNT_Y} text-anchor={pile.anchor}
									>{pile.count.toLocaleString()}</text
								>
							{/each}
						</g>
					</svg>
					<figcaption class="sd-cap">
						<span class="sd-label" title={view.arm.display}>{view.arm.display}</span>
						<span class="sd-braille" aria-hidden="true">{view.braille}</span>
						<span class="sd-distinct"
							>{view.arm.distinctScores.toLocaleString()} different scores across {PANEL_N.toLocaleString()}
							statements</span
						>
					</figcaption>
				</figure>
			{/each}
		</div>

		{#if tiers.length > 0}
			<!-- The shape rows PARTITION the cells above: each one is classed by the
			     count it prints, so this summary describes all of them or none. -->
			<p class="sd-legend">
				Every one of the {arms.length} charts above falls into exactly one row below, sorted by the
				number printed beneath it rather than by what kind of model it is.
			</p>
			<ul class="sd-tiers">
				{#each tiers as tier (tier.word)}
					<li>
						<span class="sd-tier-word">{tier.word}</span>
						<span class="sd-tier-figures"
							>{countRange(tier.minDistinct, tier.maxDistinct)} different scores across {PANEL_N.toLocaleString()}
							statements{#if tier.minTopShare !== null && tier.maxTopShare !== null}
								· the {PAPER_SCORE_TOP_PILES} commonest exact values account for {shareRange(
									tier.minTopShare,
									tier.maxTopShare
								)} of the statements{/if}</span
						>
						<span class="sd-tier-arms">{tier.displays.join(' · ')}</span>
					</li>
				{/each}
			</ul>
			<p class="sd-legend">
				Where a chart repeats the same handful of numbers, the language model did not invent them.
				Those are the values INDRA's own <em>noisy-OR</em> can produce — its SimpleScorer, running on
				the default reliability of each source with nothing fitted to this data — applied to whatever
				evidence the reading step chose to keep.{#if nonReader}
					{nonReader.arm.display} never went through that reading step: it is INDRA's own hybrid
					belief model, scored here through its fitted-counts route, and it lands in the
					{nonReader.shape} row with {nonReader.arm.distinctScores.toLocaleString()} different values{#if nonReaderTopShare},
						its {nonReaderTopShare.n} commonest exact values accounting for only
						{nonReaderTopShare.pct.toFixed(1)}% of the statements it scored{/if}.{/if}
			</p>
		{/if}
		<!-- The rest is provenance for that claim: collapsed, not cut. -->
		<details class="sd-method">
			<summary>why the reading models' piles are INDRA's own values, not invented ones</summary>
			<p>
				Once the reading step has thrown some evidence away, whatever survives is scored by the same
				formula as before, so it lands back on the same fixed ladder of reachable values. That is why
				those charts repeat the same handful of numbers. The ladder is INDRA's SimpleScorer over its
				default per-source reliabilities — the noisy-OR the reading step feeds, with nothing fitted
				to this data. The one value the reading step adds, and INDRA never emits, is <em>0</em>: the formula
				bottoms out at 0.65 over these statements and never returns 0, but a read statement does, when
				the reading step rejects every piece of evidence and no source is left standing. Piling scores
				like this still has a cost — it makes the stated probabilities less trustworthy (see the calibration
				charts) and it flatters the straight-line precision-recall area, which is why average
				precision is the number we quote.
			</p>
		</details>
	</section>
{/if}

<style>
	.sd {
		margin: 0 0 2rem;
	}
	.sd-grid {
		display: grid;
		grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
		gap: 1rem 1.2rem;
		width: 100%;
	}
	.sd-cell {
		margin: 0;
		min-width: 0;
		display: flex;
		flex-direction: column;
		gap: 0.25rem;
		/* `color` (currentColor for the ridge/piles) is set inline from the shared
		   paperArmColorVar helper — paper/port in --accent, every LLM in --blocked —
		   so this panel and the reliability strip resolve each arm to one hue. */
	}
	.sd-svg {
		display: block;
		width: 100%;
		height: auto;
		max-width: 100%;
	}

	.sd-base {
		stroke: var(--rule);
		stroke-width: 0.7;
	}
	.sd-ridge {
		fill: currentColor;
		fill-opacity: 0.14;
		stroke: currentColor;
		stroke-width: 0.8;
		stroke-linejoin: round;
	}
	.sd-empty {
		fill: var(--ink-faint);
		font-family: var(--serif);
		font-style: italic;
		font-size: 7px;
	}
	.sd-pile {
		stroke: currentColor;
		stroke-width: 1;
	}
	.sd-pile-val {
		fill: var(--ink-muted);
		font-family: var(--mono);
		font-size: 6px;
		font-variant-numeric: tabular-nums;
	}
	.sd-pile-count {
		fill: currentColor;
		font-family: var(--mono);
		font-size: 6.5px;
		font-weight: 600;
		font-variant-numeric: tabular-nums;
	}

	.sd-cap {
		display: flex;
		flex-direction: column;
		gap: 0.05rem;
		font-family: var(--mono);
		min-width: 0;
	}
	.sd-label {
		color: var(--ink);
		font-size: 0.76rem;
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}
	.sd-braille {
		color: var(--ink-faint);
		font-size: 0.66rem;
		line-height: 1.1;
		letter-spacing: -0.04em;
		overflow: hidden;
		text-overflow: clip;
		white-space: nowrap;
	}
	.sd-distinct {
		color: var(--ink-muted);
		font-size: 0.68rem;
		font-variant-numeric: tabular-nums;
	}

	.sd-legend {
		margin: 1rem 0 0;
		max-width: 46rem;
		font-family: var(--serif);
		font-style: italic;
		font-size: 0.82rem;
		line-height: 1.5;
		color: var(--ink-muted);
	}
	.sd-legend em {
		color: var(--ink);
		font-style: italic;
	}
	/* One row per shape, wrapping rather than truncating: the arm list is the
	   coverage proof, so it must never be clipped out of view. */
	.sd-tiers {
		margin: 0.5rem 0 0.75rem;
		padding: 0;
		max-width: 46rem;
		list-style: none;
		font-family: var(--mono);
		font-size: 0.7rem;
	}
	.sd-tiers li {
		display: flex;
		flex-wrap: wrap;
		gap: 0.1rem 0.75rem;
		padding: 0.3rem 0;
		border-bottom: 1px dotted var(--rule);
	}
	.sd-tiers li:last-child {
		border-bottom: 0;
	}
	.sd-tier-word {
		color: var(--ink);
	}
	.sd-tier-figures {
		color: var(--ink-muted);
		font-variant-numeric: tabular-nums;
	}
	.sd-tier-arms {
		flex: 1 0 100%;
		color: var(--ink-faint);
	}

	/* Collapsed provenance for the legend claim; the `.method` details pattern. */
	.sd-method {
		margin: 0.35rem 0 0;
		max-width: 46rem;
	}
	.sd-method summary {
		font-family: var(--mono);
		font-size: 0.66rem;
		color: var(--ink-faint);
		text-transform: uppercase;
		letter-spacing: 0.05em;
		cursor: pointer;
	}
	.sd-method summary:hover {
		color: var(--ink-muted);
	}
	.sd-method[open] summary {
		margin-bottom: 0.5rem;
	}
	.sd-method p {
		margin: 0;
		font-family: var(--serif);
		font-size: 0.78rem;
		line-height: 1.5;
		color: var(--ink-faint);
	}
	.sd-method em {
		color: var(--ink-muted);
		font-style: italic;
	}

	.sd-unavailable {
		padding: 0.4rem 0;
	}
	.sd-note {
		margin: 0;
		font-family: var(--mono);
		font-size: 0.76rem;
		color: var(--ink-faint);
	}
</style>
