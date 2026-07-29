<script lang="ts">
	/**
	 * The belief-model ladder — every belief model on this panel, on one axis.
	 *
	 * The 2023 INDRA assembly paper's own lineage (seven re-implemented rungs), the
	 * random forest on evidence-count features re-run from that paper's released
	 * code, and the four evidence-gated reading arms, each placed by its change in
	 * pooled average precision from the unfitted noisy-OR over per-source
	 * reliabilities. The noisy-OR is the origin because it is what BOTH the
	 * engineered features and evidence-gated reading modify — the artifact says so
	 * in `baseline.why`, and that sentence is printed verbatim under the axis
	 * rather than paraphrased here.
	 *
	 * Every number is read from `belief_model_ladder.json` (plus the fold SDs the
	 * loader folds in from the sibling head-to-head artifact) through the
	 * fail-closed validator. No average precision, delta, count or caveat is
	 * hard-coded in this file; only SVG layout constants are.
	 *
	 * Deliberate constraints (do not "improve" these):
	 *   · ONE shared, ZERO-ANCHORED axis in RAW average-precision units. The
	 *     baseline mark IS the origin, so the two negative rungs draw left. Never a
	 *     truncated axis over the occupied band, never a rescale into hundredths.
	 *   · Rows ordered by average precision ascending; each prints its own absolute
	 *     average precision beside the bar, so the axis stays a change axis without
	 *     hiding the level.
	 *   · Colour comes ONLY from the page-wide `paperArmColorVar` convention via the
	 *     kind mapping. The re-run row is set apart by STROKE treatment and by its
	 *     own label — never by a new colour token.
	 *   · Evidence-gated reading's delta from the baseline never travels alone: the
	 *     against-their-strongest range rides on the same caption line, every
	 *     referent named, conservative end first.
	 *   · The proximity between our re-implemented random forest and the same
	 *     forest re-run from the 2023 released code is a CONSISTENCY CHECK across
	 *     different corpora. The word
	 *     "fidelity" belongs only to the Pearson-r pointer in the method note,
	 *     never to that gap.
	 *   · Rung names on screen are ALWAYS `.display`. `.label` is a frozen join key
	 *     into `provenance.scores` / `provenance.recorded_values` and appears here
	 *     only inside `{#each}` key expressions, which render nothing.
	 */
	import {
		BELIEF_LADDER_VS_LLMS_BASENAME,
		beliefLadderBaselineTag,
		beliefLadderColorVar,
		beliefLadderDisplayOrder,
		type BeliefLadderEntry,
		type BeliefLadderLoad
	} from '$lib/data/paper-belief-ladder';

	let { data }: { data: BeliefLadderLoad } = $props();

	// ---- fixed layout (900x518 user units) -----------------------------------
	/**
	 * Screen readers were being handed our internal taxonomy slugs verbatim
	 * ("paper-family"), which also mislabels seven rungs the 2023 paper never
	 * published — it has no Bayesian, subtype, hierarchy or HybridScorer arm.
	 */
	function kindWords(kind: string): string {
		if (kind === 'paper-literal') return 're-run from the 2023 released code';
		if (kind === 'reader-gate') return 'evidence-gated reading';
		return 're-implemented on current INDRA';
	}

	const LABEL_RIGHT = 230;
	const BRACKET_X = 240;
	const PLOT_LEFT = 250;
	const PLOT_RIGHT = 742;
	const PLOT_SPAN = PLOT_RIGHT - PLOT_LEFT;
	const AP_X = 806;
	const DELTA_X = 888;
	const LEGEND_Y = 56;
	const BASELINE_TAG_Y = 70;
	const ROW_TOP = 80;
	const ROW_PITCH = 30;
	const BAR_H = 15;
	const AXIS_Y = 444;
	/** Fraction of the drawn span held as breathing room at each end. */
	const AXIS_PAD = 0.04;
	/** Roughly this many ticks; the step is snapped to a 1/2/5 x 10^k grid. */
	const TICK_TARGET = 6;
	/** Every metric value is printed to this many decimals. */
	const DIGITS = 4;

	/** Two-decimal SVG user units — coordinate rounding only, never a metric. */
	function r2(value: number): number {
		return Number(value.toFixed(2));
	}
	/** Absolute average precision, raw units. */
	function ap(value: number): string {
		return value.toFixed(DIGITS);
	}
	/** A change in average precision, raw units, always signed. */
	function signed(value: number): string {
		return `${value < 0 ? '-' : '+'}${Math.abs(value).toFixed(DIGITS)}`;
	}
	/** Coarsest 1/2/5 x 10^k step that yields about TICK_TARGET ticks over `span`. */
	function niceStep(span: number): number {
		const raw = span / TICK_TARGET;
		const magnitude = Math.pow(10, Math.floor(Math.log10(raw)));
		const normalized = raw / magnitude;
		const factor = normalized <= 1 ? 1 : normalized <= 2 ? 2 : normalized <= 5 ? 5 : 10;
		return factor * magnitude;
	}

	// ---- data ---------------------------------------------------------------
	const ladder = $derived(data.status === 'ok' ? data.ladder : null);
	const referents = $derived(data.status === 'ok' ? data.referents : null);
	const ordered = $derived<BeliefLadderEntry[]>(ladder ? beliefLadderDisplayOrder(ladder) : []);

	/** Zero-anchored domain: 0 is always inside it, so the origin is drawable. */
	const domain = $derived.by(() => {
		const deltas = ordered.map((entry) => entry.deltaVsNoisyOrBaseline);
		const low = Math.min(0, ...deltas);
		const high = Math.max(0, ...deltas);
		const pad = (high - low) * AXIS_PAD;
		return { low: low - pad, high: high + pad };
	});
	const perUnit = $derived(
		domain.high > domain.low ? PLOT_SPAN / (domain.high - domain.low) : 0
	);
	function x(value: number): number {
		return PLOT_LEFT + (value - domain.low) * perUnit;
	}
	const zeroX = $derived(x(0));

	const tickStep = $derived(niceStep(domain.high - domain.low));
	const tickDigits = $derived(Math.max(0, Math.ceil(-Math.log10(tickStep))));
	const ticks = $derived.by(() => {
		const out: number[] = [];
		if (tickStep <= 0) return out;
		const first = Math.ceil(domain.low / tickStep) * tickStep;
		for (let value = first; value <= domain.high; value += tickStep) out.push(value);
		return out;
	});

	const rows = $derived(
		ordered.map((entry, index) => {
			const top = ROW_TOP + ROW_PITCH * index;
			const end = x(entry.deltaVsNoisyOrBaseline);
			return {
				entry,
				top,
				midY: top + BAR_H / 2,
				barX: Math.min(zeroX, end),
				barW: Math.abs(end - zeroX),
				end,
				color: beliefLadderColorVar(entry)
			};
		})
	);

	/**
	 * The two rows that are ONE fitted model reported twice. They tie on average
	 * precision, so the display sort leaves them adjacent; the bracket is drawn
	 * only when they actually are, and the artifact's own note labels it.
	 */
	const samePair = $derived.by(() => {
		if (!ladder) return null;
		const [first, second] = ladder.checks.sameFittedModelPair;
		const a = rows.findIndex((row) => row.entry.label === first);
		const b = rows.findIndex((row) => row.entry.label === second);
		if (a < 0 || b < 0 || Math.abs(a - b) !== 1) return null;
		// No `?? ''` fallback: the bracket exists only to carry the artifact's own
		// explanation of why two rows are one model, so an empty note means there is
		// nothing to explain and the bracket is not drawn at all.
		const note = ladder.entries.find((entry) => entry.label === second)?.note;
		if (!note) return null;
		return { top: rows[Math.min(a, b)].top, bottom: rows[Math.max(a, b)].top + BAR_H, note };
	});

	// ---- caption values, all derived ----------------------------------------
	const gate = $derived(ladder ? ladder.guardrails.readingGate : null);
	const features = $derived(ladder ? ladder.guardrails.engineeredFeatures : null);
	/** The gate's referents, conservative end first (sorted by the data module). */
	const against = $derived(referents ? referents.referents : []);
	/** Flat rungs, weakest first, each printed with its own signed value. */
	const flat = $derived(
		ladder ? [...ladder.guardrails.flatAgainstBaseline].sort((a, b) => a.delta - b.delta) : []
	);
	const proximity = $derived(ladder ? ladder.guardrails.reimplementationProximity : null);
	/**
	 * The origin tag, measured against the origin it will actually be drawn at.
	 * `beliefLadderBaselineTag` returns null when the string would run past the
	 * viewBox's right edge; that gates the figure (see the `{#if}` below) rather
	 * than drawing a rule labelled with a truncated model name. SVG does not wrap
	 * and does not warn, so nothing else would catch it.
	 */
	const baselineTag = $derived(
		ladder ? beliefLadderBaselineTag(ladder.baseline.display, ap(ladder.baseline.averagePrecision), zeroX) : null
	);
	/** Where the fold SDs come from: the first rung that cites the sibling artifact. */
	const foldSdSource = $derived(
		ladder
			? (Object.values(ladder.recordedValues).find((recorded) =>
					recorded.path.endsWith(BELIEF_LADDER_VS_LLMS_BASENAME)
				)?.path ?? null)
			: null
	);
</script>

<section class="belief-ladder" aria-labelledby="belief-ladder-title">
	{#if data.status !== 'ok' || ladder === null || referents === null || baselineTag === null}
		<div class="gate" role="status">
			<p class="eyebrow">belief-model ladder</p>
			<h2 id="belief-ladder-title">Belief-model ladder unavailable</h2>
			<p>
				{data.status !== 'ok'
					? data.reason
					: ladder === null || referents === null
						? 'the belief-ladder payload is missing.'
						: 'the name of the starting model does not fit beside the line that marks it, so the axis cannot be labelled.'}
			</p>
			<code>{data.artifact_path}</code>
		</div>
	{:else}
		<h2 id="belief-ladder-title" class="visually-hidden">Where every belief model lands</h2>
		<figure>
			<div class="scroller">
				<svg
					viewBox="0 0 900 518"
					preserveAspectRatio="xMidYMid meet"
					role="img"
					aria-labelledby="belief-ladder-chart-title belief-ladder-chart-desc"
				>
					<title id="belief-ladder-chart-title"
						>Every belief model on one axis, by change in average precision from the noisy-OR
						baseline</title
					>
					<desc id="belief-ladder-chart-desc"
						>{ordered.length} horizontal bars on one shared axis anchored at zero, in raw
						average-precision units. Zero is {ladder.baseline.display} at {ap(
							ladder.baseline.averagePrecision
						)}; bars to the right of it do better, bars to the left do worse.
						{#each rows as row (row.entry.label)}{row.entry.display} ({kindWords(row.entry.kind)}): average precision
							{ap(row.entry.averagePrecision)}, change {signed(row.entry.deltaVsNoisyOrBaseline)}.
						{/each}</desc
					>

					<!-- title block (y = 0..44) -->
					<text class="fig-title" x="12" y="20">Where every belief model lands</text>
					<text class="fig-subtitle" x="12" y="37"
						>{ladder.entries.length} models, the same {ladder.panel.n} statements, one measure</text
					>

					<!--
						Legend: three kinds, two hues, one stroke treatment. Each entry names the
						thing rather than pointing at a paper the reader has not read, so the x
						positions are re-spaced for the longer names. Measured at 7px mono with
						0.06em letter-spacing = 4.62 units per character: 31 ch from x=24 ends at
						167 (next key 177), 30 ch from x=189 ends at 328 (next key 338), 22 ch from
						x=350 ends at 452 — clear of the right-anchored "average precision" column
						head, which starts at ~727.
					-->
					<rect class="key" x="12" y={LEGEND_Y - 7} width="8" height="8" fill="var(--accent)" />
					<text class="legend" x="24" y={LEGEND_Y}>re-implemented on current INDRA</text>
					<rect
						class="key literal"
						x="177"
						y={LEGEND_Y - 7}
						width="8"
						height="8"
						fill="var(--accent)"
						stroke="var(--accent)"
					/>
					<text class="legend" x="189" y={LEGEND_Y}>re-run from 2023 released code</text>
					<rect class="key" x="338" y={LEGEND_Y - 7} width="8" height="8" fill="var(--blocked)" />
					<text class="legend" x="350" y={LEGEND_Y}>evidence-gated reading</text>
					<text class="col-head" x={AP_X} y={LEGEND_Y}>average precision</text>
					<text class="col-head" x={DELTA_X} y={LEGEND_Y}>change</text>

					<!-- the baseline: a labelled rule THROUGH the origin -->
					<line class="baseline-rule" x1={r2(zeroX)} y1={BASELINE_TAG_Y + 4} x2={r2(zeroX)} y2={AXIS_Y} />
					<!-- Left-anchored and origin-relative, so its budget is measured against
					     the origin the data puts it at; see beliefLadderBaselineTag. -->
					<text class="baseline-tag" x={r2(zeroX) + 5} y={BASELINE_TAG_Y}>{baselineTag}</text>

					{#if samePair}
						<path
							class="bracket"
							d="M{BRACKET_X},{samePair.top} L{BRACKET_X - 6},{samePair.top} L{BRACKET_X -
								6},{samePair.bottom} L{BRACKET_X},{samePair.bottom}"
						/>
					{/if}

					{#each rows as row (row.entry.label)}
						<text class="arm" x={LABEL_RIGHT - 2} y={r2(row.midY) + 3} fill={row.color}
							>{row.entry.display}</text
						>
						<rect
							class="rung"
							class:literal={row.entry.kind === 'paper-literal'}
							x={r2(row.barX)}
							y={row.top}
							width={r2(row.barW)}
							height={BAR_H}
							fill={row.color}
							stroke={row.color}
						/>
						<circle class="rung-end" cx={r2(row.end)} cy={r2(row.midY)} r="2.2" fill={row.color} />
						<text class="value" x={AP_X} y={r2(row.midY) + 3}>{ap(row.entry.averagePrecision)}</text>
						<text class="value change" x={DELTA_X} y={r2(row.midY) + 3}
							>{signed(row.entry.deltaVsNoisyOrBaseline)}</text
						>
					{/each}

					<!-- shared zero-anchored axis: raw average-precision units, no gridlines -->
					<line class="axis" x1={PLOT_LEFT} y1={AXIS_Y} x2={PLOT_RIGHT} y2={AXIS_Y} />
					{#each ticks as tick (tick)}
						<line class="tick-mark" x1={r2(x(tick))} y1={AXIS_Y} x2={r2(x(tick))} y2={AXIS_Y + 4} />
						<text class="tick" x={r2(x(tick))} y={AXIS_Y + 14}
							>{tick === 0 ? '0' : `${tick < 0 ? '-' : '+'}${Math.abs(tick).toFixed(tickDigits)}`}</text
						>
					{/each}
					<text class="axis-title" x={PLOT_LEFT} y={AXIS_Y + 30}
						>change in average precision from the starting model &#8594;</text
					>
					<text class="fig-note" x="12" y={AXIS_Y + 48}
						>Zero is {ladder.baseline.display}: {ladder.baseline.whyProse.plain}.</text
					>
					{#if samePair}
						<text class="fig-note" x="12" y={AXIS_Y + 64}>Bracketed: {samePair.note}</text>
					{/if}
				</svg>
			</div>

			<figcaption>
				<p>
					{ladder.entries.length} ways of computing belief, each placed by how much it changes average
					precision from the same starting point: a noisy-OR over per-source reliabilities with
					nothing fitted to data. That is the model both the hand-built features and evidence-gated
					reading are modifications of.
				</p>
				{#if gate && features && against.length > 2}
					<p class="key-line">
						Set against the three strongest non-reading models, evidence-gated reading
						({gate.display}) adds {signed(against[0].delta)} over the re-implementation of the
						random forest with its full feature set ({ap(against[0].armAp)}),
						{signed(against[1].delta)} over {against[1].armDisplay} ({ap(against[1].armAp)}), the
						strongest model published in 2023, and {signed(against[2].delta)} over
						{against[2].armDisplay} ({ap(against[2].armAp)}), the one every paired interval on this
						page is measured against. Measured from the unfitted starting model instead, reading is
						worth {signed(gate.deltaVsNoisyOrBaseline)} and the hand-built features
						{signed(features.deltaVsNoisyOrBaseline)}. That feature-based forest is the same model as
						the first of the three above, and not the published one: INDRA's own count features
						include the statement type, paper count and promoter flag the 2023 forest was given, and
						more. Reading adds a further
						{signed(gate.deltaVsBestNoisyOrVariant.delta)} over the best noisy-OR here, which is also
						fitted for the first time here — no version resolving statement subtypes was
						published in 2023.
					</p>
				{/if}
				{#if flat.length > 1}
					<p>
						Two of these land exactly where the unfitted starting model lands: {flat[0].display}
						({signed(flat[0].delta)}) and {flat[1].display} ({signed(flat[1].delta)}). Fitting a
						better weighting of the evidence counts buys nothing; what buys something is the
						features that are not counts.
					</p>
				{/if}
				<p>
					The statements are split into 10 folds — 10 groups, each fitted model scored on the group
					it did not train on. How much a model's score wobbles from fold to fold runs {ap(referents.foldSd.min)}&#8211;{ap(
						referents.foldSd.max
					)} across the {referents.foldSd.nArms} models that report it, measured as trapezoidal
					PR-AUC, the estimator published in 2023 — precision-recall area with a straight line drawn
					between points. Both ends
					of that range are evidence-gated reading models, not the random forest. That wobble
					describes spread between folds; it is not an error bar on average precision, and which
					model beats which is settled by the paired resampling elsewhere on this page, not here.
				</p>

				<details class="method">
					<summary>how this is computed</summary>
					<p>
						Every model is scored on the same {ladder.panel.n} assembled statements, the ones
						carrying <code>{ladder.panel.labelField}</code> ({ladder.prose.labelConvention.plain}):
						{ladder.panel.nCorrect} correct and {ladder.panel.nErrors} wrong. Of the wrong ones,
						{ladder.panel.adjudicationSafeNegatives} had their evidence review finished and
						{ladder.panel.flaggedNotAdjudicationSafe} did not, and are flagged
						<code>label_is_adjudication_safe: false</code>. Ordering: {ladder.prose.panelOrdering
							.plain}. Measure: {ladder.metric} via {ladder.prose.metricSource.plain}. The starting
						model combines its per-source reliabilities as {ladder.prose.noisyOrFormula.plain}.
					</p>
					{#if proximity}
						<p>
							{ap(proximity.reimplementedRfFullFeatures)} vs {ap(proximity.paperLiteralRfPromoter)},
							{ap(proximity.absoluteGap)} apart &#8212; {ladder.prose.proximityStatus.plain}. The
							separate evidence: {ladder.prose.fidelityStatistic.plain} = {ap(
								proximity.fidelityEvidence.value
							)}, from
							<code>{proximity.fidelityEvidence.source}</code>.
						</p>
					{/if}
					{#each ladder.prose.caveats as caveat (caveat.shipped)}
						<p>{caveat.plain}</p>
					{/each}
					<p>
						Only the fitted models need that split: they are scored on the group they did not
						train on. The reading models are never trained and never see a label, so nothing has
						to be held back from them — they are scored once over all {ladder.panel.n} statements
						and then given the same fold indices, so that every row is summarised by the identical
						estimator.
					</p>
					<p>
						The fold-to-fold wobble figures are the ones published in 2023, read from
						<code>{foldSdSource}</code>
						for the {referents.foldSd.nArms} models that report one. That file has no
						<code>output_sha256</code> entry in the run manifest, so its shape is validated and its
						range cross-checked against what this figure draws, rather than being pinned byte for
						byte. Gold: <code>{ladder.gold}</code>. Join: {ladder.prose.join.plain}. Generated by
						<code>{ladder.generatedBy}</code>.
					</p>
					<ul class="grain">
						{#each ordered as entry (entry.label)}
							<li>
								<b>{entry.display}</b> &#8212; {ap(entry.averagePrecision)} ({signed(
									entry.deltaVsNoisyOrBaseline
								)}), {entry.distinctScores} distinct scores &#183;
								<code>{entry.scoresPath}</code>
							</li>
						{/each}
					</ul>
				</details>
			</figcaption>
		</figure>
		<footer>
			<code>{data.artifact_path}</code> · sha256 <code>{data.artifact_sha256.slice(0, 12)}</code>
		</footer>
	{/if}
</section>

<style>
	.belief-ladder {
		margin-top: 1.5rem;
		padding-top: 1.25rem;
		border-top: 2px solid var(--ink);
	}
	figure {
		margin: 0;
	}
	.scroller {
		overflow-x: auto;
	}
	svg {
		display: block;
		width: 100%;
		min-width: 680px;
		height: auto;
	}
	.fig-title {
		fill: var(--ink);
		font-family: var(--serif);
		font-size: 15px;
	}
	.fig-subtitle {
		fill: var(--ink-muted);
		font-family: var(--serif);
		font-size: 9.5px;
		font-style: italic;
	}
	.fig-note {
		fill: var(--ink-muted);
		font-family: var(--serif);
		font-size: 9px;
		font-style: italic;
	}
	.legend,
	.col-head {
		fill: var(--ink-faint);
		font-family: var(--mono);
		font-size: 7px;
		letter-spacing: 0.06em;
		text-transform: uppercase;
	}
	.col-head {
		text-anchor: end;
	}
	/* The literal arm keeps the paper hue and is set apart by stroke only. */
	.key.literal {
		fill-opacity: 0.3;
		stroke-width: 1;
		stroke-dasharray: 2 1.5;
	}
	.arm {
		font-family: var(--mono);
		font-size: 9px;
		text-anchor: end;
	}
	.rung {
		shape-rendering: crispEdges;
		stroke-width: 0;
	}
	.rung.literal {
		fill-opacity: 0.3;
		stroke-width: 1;
		stroke-dasharray: 3 2;
	}
	.value {
		fill: var(--ink-muted);
		font-family: var(--mono);
		font-size: 9px;
		text-anchor: end;
		font-variant-numeric: tabular-nums;
	}
	.value.change {
		fill: var(--ink);
	}
	.baseline-rule {
		stroke: var(--ink);
		stroke-width: 1.2;
	}
	.baseline-tag {
		fill: var(--ink);
		font-family: var(--mono);
		font-size: 8px;
	}
	.bracket {
		fill: none;
		stroke: var(--ink-faint);
		stroke-width: 1;
	}
	.axis,
	.tick-mark {
		stroke: var(--rule);
		stroke-width: 1;
	}
	.tick {
		fill: var(--ink-faint);
		font-family: var(--mono);
		font-size: 8px;
		text-anchor: middle;
		font-variant-numeric: tabular-nums;
	}
	.axis-title {
		fill: var(--ink-muted);
		font-family: var(--mono);
		font-size: 8.5px;
	}
	figcaption {
		margin-top: 0.7rem;
		max-width: 74ch;
	}
	figcaption p {
		margin: 0.2rem 0 0;
		font-family: var(--serif);
		font-size: 0.86rem;
		line-height: 1.55;
		color: var(--ink-muted);
	}
	figcaption .key-line {
		color: var(--ink);
	}
	.grain {
		margin: 0.2rem 0 0;
		padding-left: 1.1rem;
	}
	.grain li {
		font-family: var(--mono);
		font-size: 0.7rem;
		color: var(--ink-muted);
		line-height: 1.5;
	}
	.method p {
		margin: 0 0 0.55rem;
		font-family: var(--serif);
		font-size: 0.8rem;
		line-height: 1.5;
		color: var(--ink-muted);
	}
	/* The full method note: one click, nothing hidden, nothing lost. */
	.method {
		margin-top: 0.45rem;
		max-width: 74ch;
	}
	.method summary {
		font-family: var(--mono);
		font-size: 0.66rem;
		color: var(--ink-faint);
		text-transform: uppercase;
		letter-spacing: 0.05em;
		cursor: pointer;
	}
	.method summary:hover {
		color: var(--ink-muted);
	}
	.method[open] summary {
		margin-bottom: 0.6rem;
	}
	footer {
		margin-top: 0.3rem;
		font-family: var(--mono);
		font-size: 0.62rem;
		color: var(--ink-faint);
	}
	code {
		font-family: var(--mono);
		font-size: 0.62rem;
		color: var(--ink-faint);
		/* Long artifact paths wrap instead of widening the page on narrow screens. */
		overflow-wrap: anywhere;
	}
	.gate {
		border: 1px solid var(--rule);
		border-left: 3px solid var(--blocked);
		padding: 1rem;
	}
	.gate h2 {
		margin: 0.18rem 0 0;
		font-family: var(--serif);
		font-size: 1.35rem;
		font-weight: 400;
	}
	.eyebrow {
		margin: 0;
		font-family: var(--mono);
		font-size: 0.68rem;
		color: var(--ink-faint);
		text-transform: uppercase;
		letter-spacing: 0.05em;
	}
	.gate p:not(.eyebrow) {
		font-family: var(--serif);
		color: var(--ink-muted);
	}
</style>
