<script lang="ts">
	/**
	 * The framing correction — evidence-gated reading IS INDRA's own unfitted
	 * noisy-OR over per-source reliabilities.
	 *
	 * This panel runs BEFORE any head-to-head, because an author who reads the
	 * comparison as "the 2023 belief model vs an LLM scorer" reads it wrong. There
	 * is one belief model here. The reader emits a keep/reject verdict per
	 * (statement, evidence) pair; the surviving counts go through
	 * `belief = 1 - PROD_s (syst_s + rand_s^{n_s})` unchanged. Removing evidence
	 * removes factors from a product of numbers below 1, so the arm can only ever
	 * LOWER belief.
	 *
	 * Order, and it is deliberate:
	 *   (b) the subtractive proof, as the headline — zero of the panel x four
	 *       reader beliefs sits above the noisy-OR on its own statement;
	 *   (c) the supporting table — every non-zero score is a value that same
	 *       formula emits, beside the permutation chance floor that makes the
	 *       number worth stating;
	 *   (a) one compact declaration line — what the four bundle manifests say,
	 *       checked against bytes by the emitting script, not transcribed;
	 *   (d) the no-LLM control — the three non-reading subtractions with no model
	 *       verdict anywhere, landing BELOW the ungated baseline.
	 *
	 * Every number is read from `framing_correction.json` and
	 * `non_reading_control.json` through the fail-closed loader. No count, share,
	 * threshold or caveat is hard-coded here; only layout constants are. If either
	 * artifact fails, the whole panel renders the named empty state — the argument
	 * is one argument, and half of it is not a weaker version of it.
	 *
	 * Deliberate constraints (do not "improve" these):
	 *   · The confirmed count is printed with any budget-exhausted remainder named
	 *     beside it. A 100% the search did not earn must never appear.
	 *   · The arm is described as INDRA's unfitted noisy-OR over per-source
	 *     reliabilities, run on kept evidence. Never as a rival scorer, a competing
	 *     model, or "our model".
	 *   · Prose outside the <details> stays under 260 words, counted the way
	 *     `scripts/test-paper-literal-contract.mjs` counts it (247 today); the
	 *     provenance, the formula and the label convention live inside it and may
	 *     be relocated but never deleted.
	 *
	 * The old ceiling read "under 140 words" and was never tied to a counter, so it
	 * could not be checked; it is now 260 against the page's own word counter, and
	 * this file measures 247. It rose in the plain-language pass, and
	 * the reason is the same one that raised the page's own budget: the headline
	 * had to say what a noisy-OR IS ("every source carries a fixed reliability,
	 * nothing is fitted to this data, and a statement scores higher the more
	 * surviving evidence it has") before it could claim anything about it. That
	 * sentence is ~35 words and it is the first thing on the page that explains the
	 * formula every later beat measures against. Rendered prose here says "model",
	 * never "arm"; identifiers below still say `arm`, and that is deliberate — the
	 * artifact field names are frozen.
	 */
	import {
		FRAMING_CONTROL_GEOMETRY,
		framingArmColorVar,
		framingLargestPriorGroup,
		framingPaperColorVar,
		framingUnresolvedTotal,
		type FramingCorrectionLoad
	} from '$lib/data/paper-framing-correction';

	let { data }: { data: FramingCorrectionLoad } = $props();

	// ---- fixed layout (control strip: 900x150 user units) --------------------
	/**
	 * Taken from the data module rather than repeated here, because the row-label
	 * character budget is DERIVED from these numbers and ENFORCED there: the
	 * validator throws above `FRAMING_CONTROL_LABEL_BUDGET_CHARS` (66 chars into a
	 * 340-unit gutter at 8.5px mono), which gates the panel instead of clipping a
	 * label's leading glyphs against x = 0. Two of the artifact's own labels were
	 * doing exactly that at the previous 300-unit gutter. Change the geometry in
	 * one place and the budget moves with it.
	 */
	const STRIP_LEFT = FRAMING_CONTROL_GEOMETRY.stripLeft;
	const STRIP_RIGHT = FRAMING_CONTROL_GEOMETRY.stripRight;
	const STRIP_TOP = 40;
	const STRIP_PITCH = 20;
	const STRIP_AXIS_PAD = 22;
	const DOT_R = 3.6;
	/** Axis padding as a share of the drawn AP span, so no dot sits on the edge. */
	const AXIS_MARGIN = 0.08;

	function r2(value: number): number {
		return Math.round(value * 100) / 100;
	}
	function pct(value: number): string {
		return `${(value * 100).toFixed(1)}%`;
	}
	function ap(value: number): string {
		return value.toFixed(4);
	}
	function group(value: number): string {
		return value.toLocaleString('en-US');
	}

	// ---- data -----------------------------------------------------------------
	const framing = $derived(data.status === 'ok' ? data.framing : null);
	const control = $derived(data.status === 'ok' ? data.control : null);
	// The 0 / {lo:0,hi:1} fallbacks below are UNREACHABLE, not defaults: the
	// template gates the whole panel when either artifact is null. They exist
	// because a module-scope `$derived` cannot narrow on that gate.
	const armCount = $derived(framing ? framing.subtractive.arms.length : 0);
	const unresolved = $derived(framing ? framingUnresolvedTotal(framing) : 0);
	const priorGroup = $derived(framing ? framingLargestPriorGroup(framing) : null);

	/** One row per arm: leg (c) beside leg (b)'s zero block, in artifact order. */
	const armRows = $derived(
		framing
			? framing.reachable.arms.map((arm, index) => ({
					arm,
					color: framingArmColorVar(index)
				}))
			: []
	);

	// ---- control strip geometry, all derived ---------------------------------
	const stripRows = $derived(control ? control.rows : []);
	const stripBaseline = $derived(
		control ? (stripRows.find((row) => row.key === control.baselineRow) ?? null) : null
	);
	const stripDomain = $derived.by(() => {
		if (!control) return { lo: 0, hi: 1 };
		const values = [...stripRows.map((row) => row.averagePrecision), control.contrast.averagePrecision];
		const lo = Math.min(...values);
		const hi = Math.max(...values);
		const pad = (hi - lo) * AXIS_MARGIN || 0.01;
		return { lo: lo - pad, hi: hi + pad };
	});
	function stripX(value: number): number {
		const { lo, hi } = stripDomain;
		return STRIP_LEFT + ((value - lo) / (hi - lo)) * (STRIP_RIGHT - STRIP_LEFT);
	}
	const stripPlot = $derived(
		stripRows.map((row, index) => ({
			row,
			y: STRIP_TOP + STRIP_PITCH * index,
			x: stripX(row.averagePrecision),
			isBaseline: row.key === control?.baselineRow
		}))
	);
	const contrastPlot = $derived(
		control
			? {
					contrast: control.contrast,
					y: STRIP_TOP + STRIP_PITCH * stripRows.length,
					x: stripX(control.contrast.averagePrecision)
				}
			: null
	);
	const stripHeight = $derived(STRIP_TOP + STRIP_PITCH * (stripRows.length + 1) + STRIP_AXIS_PAD);
</script>

<section class="framing" aria-labelledby="framing-title">
	{#if data.status !== 'ok' || framing === null || control === null}
		<div class="gate" role="status">
			<p class="eyebrow">framing correction</p>
			<h2 id="framing-title">Framing correction unavailable</h2>
			<p>{data.status === 'ok' ? 'the framing-correction payload is missing.' : data.reason}</p>
			<code>{data.artifact_path}</code>
			<code>{data.control_path}</code>
		</div>
	{:else}
		<h2 id="framing-title">
			Evidence-gated reading: the language model removes evidence, INDRA's own formula still does the
			scoring
		</h2>

		<!-- (b) the subtractive proof, as the headline -->
		<p class="headline">
			That formula is INDRA's noisy-OR: every source carries a fixed reliability, nothing is fitted to
			this data, and a statement scores higher the more surviving evidence it has. The language model
			only decides which evidence survives. Across {group(framing.panel.n)} statements and {armCount} reading
			models — {group(framing.subtractive.nComparisons)} comparisons —
			<b>{framing.subtractive.nExceedingNoisyOr}</b> of the models' scores came out above that formula's
			own score for the same statement. Taking evidence away removes factors from a product of numbers
			below 1, so reading can only subtract.
		</p>

		<!-- (c) every non-zero score is a value that same formula emits -->
		<figure>
			<div class="scroller">
				<table>
					<caption>Where each model's scores land</caption>
					<thead>
						<tr>
							<th scope="col">model</th>
							<th scope="col" class="num">non-zero</th>
							<th scope="col" class="num">on a value the formula can emit</th>
							<th scope="col" class="num">same, after shuffling</th>
							<th scope="col" class="num">exactly 0.0</th>
						</tr>
					</thead>
					<tbody>
						{#each armRows as row (row.arm.key)}
							<tr>
								<th scope="row" style:color={row.color}>{row.arm.display}</th>
								<td class="num">{group(row.arm.nNonzero)}</td>
								<td class="num strong"
									>{group(row.arm.nConfirmedReachable)}
									<span class="share">{pct(row.arm.shareConfirmed)}</span>
									{#if row.arm.nBudgetExhausted > 0}<span class="remainder"
											>+{group(row.arm.nBudgetExhausted)} unsettled</span
										>{/if}</td
								>
								<td class="num floor"
									>{pct(row.arm.permutedRateMean)}
									<span class="share"
										>{pct(row.arm.permutedRateMin)}–{pct(row.arm.permutedRateMax)}</span
									></td
								>
								<td class="num">{group(row.arm.nAtExactlyZero)}</td>
							</tr>
						{/each}
					</tbody>
				</table>
			</div>
			<figcaption>
				<p>
					The middle column counts the scores that land exactly on a value that formula could have
					produced from some subset of the evidence. “Same, after shuffling” takes those same scores and
					reassigns them to other statements at random, {framing.reachable.nullBaseline.nPermutations} times
					over: such values are common, so a match on its own proves little — this is the rate to beat.{#if unresolved > 0}
						{group(unresolved)} scores did not settle inside the search budget.{/if}
				</p>
			</figcaption>
		</figure>

		<!-- (a) the declaration, checked against bytes by the emitting script -->
		<p class="declaration">
			All {framing.declaration.arms.length} run manifests declare the same scoring,
			<code>{framing.declaration.requiredAggregation}</code>, with no reader profile
			(<code>reader_profile: null</code>) and one shared table of source reliabilities —
			{#if priorGroup}{priorGroup.sources.join(', ')} all at random-error rate {priorGroup.rand} and
				systematic-error rate {priorGroup.syst}{/if}.
		</p>

		<!-- (d) the no-LLM control, from the artifact that emitted it -->
		<figure>
			<div class="scroller">
				<svg
					viewBox="0 0 900 {stripHeight}"
					preserveAspectRatio="xMidYMid meet"
					role="img"
					aria-labelledby="framing-strip-title framing-strip-desc"
				>
					<title id="framing-strip-title">Average precision when no language model is involved</title
					>
					<desc id="framing-strip-desc"
						>{control.finding}
						{#each stripRows as row (row.key)}{row.display}: {ap(row.averagePrecision)}.
						{/each}{control.contrast.display}: {ap(control.contrast.averagePrecision)}.</desc
					>

					<text class="fig-title" x="12" y="18">With no reading at all</text>
					<text class="fig-subtitle" x="12" y="32">{control.metric}, {control.metricSource}</text>

					{#if stripBaseline}
						<line
							class="baseline"
							x1={r2(stripX(stripBaseline.averagePrecision))}
							y1={STRIP_TOP - 12}
							x2={r2(stripX(stripBaseline.averagePrecision))}
							y2={STRIP_TOP + STRIP_PITCH * stripRows.length + 4}
						/>
						<!--
							The tick says what the baseline IS rather than naming it: no verdict
							removed anything, every evidence entry counted. Centred, and safe to
							centre — AXIS_MARGIN keeps any plotted value inside 6.9%–93.1% of the
							drawn span, so the centre lands between x = 382.4 and x = 787.6.
							Measured in-browser against the real font stack rather than estimated:
							4.82 units per character. RE-MEASURED for the plain-language pass, which
							replaced "no gate — every evidence entry" (37 chars incl. the value)
							with the wording below: 42 chars incl. the 6-char value = 202.4 units,
							half-width 101.2, so the worst case is drawn 281.2→888.8 inside the
							900-unit viewBox. The budget is min(382.4, 900 − 787.6) = 112.4 units of
							half-width, i.e. 46 characters; this label uses 42. It cannot clip at
							either end, whatever the numbers are.
						-->
						<text
							class="baseline-label"
							x={r2(stripX(stripBaseline.averagePrecision))}
							y={STRIP_TOP - 16}
							text-anchor="middle"
							>nothing dropped — all evidence kept {ap(stripBaseline.averagePrecision)}</text
						>
					{/if}

					{#each stripPlot as point (point.row.key)}
						<text class="row-label" x={STRIP_LEFT - 10} y={point.y + 3}>{point.row.display}</text>
						<circle
							class="dot"
							class:is-baseline={point.isBaseline}
							cx={r2(point.x)}
							cy={point.y}
							r={DOT_R}
							fill={framingPaperColorVar()}
						/>
						<text class="value" x={r2(point.x) + 9} y={point.y + 3}
							>{ap(point.row.averagePrecision)}</text
						>
					{/each}

					{#if contrastPlot}
						<text class="row-label reading" x={STRIP_LEFT - 10} y={contrastPlot.y + 3}
							>{contrastPlot.contrast.display}</text
						>
						<circle
							class="dot"
							cx={r2(contrastPlot.x)}
							cy={contrastPlot.y}
							r={DOT_R}
							fill={framingArmColorVar(0)}
						/>
						<text class="value" x={r2(contrastPlot.x) + 9} y={contrastPlot.y + 3}
							>{ap(contrastPlot.contrast.averagePrecision)}</text
						>
					{/if}

					<line
						class="axis"
						x1={STRIP_LEFT}
						y1={stripHeight - STRIP_AXIS_PAD + 4}
						x2={STRIP_RIGHT}
						y2={stripHeight - STRIP_AXIS_PAD + 4}
					/>
					<text class="axis-title" x={STRIP_LEFT} y={stripHeight - 4}
						>average precision &#8594;</text
					>
				</svg>
			</div>
			<figcaption>
				<p>
					No language model touched those rows: evidence was removed by fixed rules alone. They land
					below the line, so what the reading buys is not simply having less evidence.
				</p>

				<details class="method">
					<summary>formula, provenance, and what “error” means here</summary>
					<p>
						<b>{control.question}</b>
						{control.finding}
						{framing.reachable.noisyOrFloorNote} Across these {group(framing.panel.n)} statements that
						floor is {framing.reachable.noisyOrFloor}.
					</p>
					<p>
						The scoring formula is <code>{framing.noisyOrFormula}</code> over
						<code>sorted(sources)</code>, with <code>n_s</code> the surviving evidence count for source
						<code>s</code>. A source whose evidence is entirely rejected drops out of the product
						altogether; the language model never changes a source's reliability, only which sources
						are still in the product.
						{framing.declaration.dispatch}
					</p>
					<!--
						The panel's ONE disclosure of the label convention. It used to be
						stated twice — here, and again in a caveat the artifact carried
						twenty lines below — with the raw flag name left bare in both. The
						duplicate was removed at the emitting script; this paragraph is now
						the only place it is said, so it says what the flag means.
					-->
					<p>
						<b>The label convention.</b> “Error” is the published curation label released with the
						2023 INDRA assembly paper,
						<code>{framing.panel.labelField}</code>: {group(framing.panel.negativeBreakdown.nErrors)} of {group(
							framing.panel.n
						)} statements. For {group(framing.panel.negativeBreakdown.adjudicationSafeNegatives)} of
						them every piece of the statement's evidence has been reviewed and all of it came back
						negative. The other {group(framing.panel.negativeBreakdown.flaggedNotAdjudicationSafe)} still
						have evidence nobody has reviewed, so the negative rests on an incomplete review — the gold
						file marks those <code>label_is_adjudication_safe: false</code>. Both groups stay in,
						labelled exactly as the published curations have them, and the same labels score every
						model. Nothing here re-labels those curations.
					</p>
					<p>
						<b>How the table above is checked.</b>
						{framing.reachable.definition}. Matched to within {framing.reachable.tolerance}, with a
						tighter bit-exact tier: {#each armRows as row, index (row.arm.key)}{index > 0
								? ', '
								: ''}{row.arm.display} {group(row.arm.nBitExact)}/{group(row.arm.nNonzero)}{/each}.
						The search is depth-first over
						<code>sorted(sources)</code>, capped at {group(framing.reachable.nodeBudgetPerStatement)} nodes
						per statement; the worst statement here used {group(framing.reachable.maxNodesUsed)}.
						{framing.reachable.nullBaseline.method} Seed {framing.reachable.nullBaseline.seed}, over the
						{group(framing.reachable.nullBaseline.nStatementsEnumerable)} of {group(
							framing.reachable.nullBaseline.nStatementsOnPanel
						)} statements whose reachable set can be enumerated outright.
					</p>
					{#each framing.caveats as caveat (caveat)}
						<p>{caveat}</p>
					{/each}
					<p>
						<b>The rows in the figure above, in full.</b>
						{#each control.rows as row (row.key)}{row.display} — {ap(row.averagePrecision)} ({group(
								row.nEvidenceScored
							)} evidence). {/each}Contrast: {control.contrast.display} — {ap(
							control.contrast.averagePrecision
						)}. Generated by <code>{control.generatedBy}</code>.
					</p>
					<p>
						<b>Provenance.</b>
						{#each Object.entries(framing.provenance) as [key, value] (key)}<code>{key}</code>:
							<code>{value}</code>.
						{/each}
					</p>
					<ul class="declared">
						{#each framing.declaration.arms as arm (arm.key)}
							<li>
								<b>{arm.display}</b> — <code>{arm.manifestPath}</code>, <code>{arm.implementation}</code>,
								<code>{arm.aggregation}</code>, dedup {arm.dedup}
							</li>
						{/each}
						<li>
							<code>{framing.declaration.aggregationConfig.path}</code> sha256
							<code>{framing.declaration.aggregationConfig.sha256.slice(0, 12)}</code>,
							<code>{framing.declaration.noiseModelSource.path}</code> sha256
							<code>{framing.declaration.noiseModelSource.sha256.slice(0, 12)}</code>,
							<code>{framing.declaration.statementBeliefSource.path}</code> sha256
							<code>{framing.declaration.statementBeliefSource.sha256.slice(0, 12)}</code>
						</li>
					</ul>
				</details>
			</figcaption>
		</figure>

		<footer>
			<code>{data.artifact_path}</code> · sha256
			<code>{data.artifact_sha256.slice(0, 12)}</code> ·
			<code>{data.control_path}</code> · sha256
			<code>{data.control_sha256.slice(0, 12)}</code>
		</footer>
	{/if}
</section>

<style>
	.framing {
		margin-top: 1.5rem;
		padding-top: 1.25rem;
		border-top: 2px solid var(--ink);
	}
	h2 {
		margin: 0 0 0.5rem;
		font-family: var(--serif);
		font-size: 1.35rem;
		font-weight: 400;
	}
	.headline {
		margin: 0 0 0.9rem;
		font-family: var(--serif);
		font-size: 0.95rem;
		line-height: 1.55;
		color: var(--ink);
		max-width: 74ch;
	}
	.headline b {
		font-weight: 600;
		font-variant-numeric: tabular-nums;
	}
	figure {
		margin: 0 0 0.9rem;
	}
	.scroller {
		overflow-x: auto;
	}
	table {
		border-collapse: collapse;
		width: 100%;
		min-width: 560px;
		font-family: var(--mono);
		font-size: 0.72rem;
	}
	caption {
		font-family: var(--serif);
		font-size: 0.85rem;
		color: var(--ink);
		text-align: left;
		padding-bottom: 0.35rem;
	}
	thead th {
		font-weight: 400;
		font-size: 0.62rem;
		text-transform: uppercase;
		letter-spacing: 0.05em;
		color: var(--ink-faint);
		border-bottom: 1px solid var(--rule);
		padding: 0.25rem 0.5rem 0.3rem;
		text-align: left;
	}
	tbody th {
		font-weight: 500;
		text-align: left;
		padding: 0.32rem 0.5rem;
		border-bottom: 1px solid var(--rule);
	}
	tbody td {
		padding: 0.32rem 0.5rem;
		border-bottom: 1px solid var(--rule);
		color: var(--ink-muted);
		font-variant-numeric: tabular-nums;
	}
	.num {
		text-align: right;
	}
	td.strong {
		color: var(--ink);
		font-weight: 600;
	}
	td.floor {
		color: var(--ink-faint);
	}
	.share {
		display: block;
		font-size: 0.62rem;
		font-weight: 400;
		color: var(--ink-faint);
	}
	.remainder {
		display: block;
		font-size: 0.62rem;
		font-weight: 400;
		color: var(--blocked);
	}
	.declaration {
		margin: 0 0 1rem;
		padding: 0.55rem 0.7rem;
		border: 1px solid var(--rule);
		border-left: 3px solid var(--accent);
		background: color-mix(in srgb, var(--ink) 3%, transparent);
		font-family: var(--serif);
		font-size: 0.82rem;
		line-height: 1.5;
		color: var(--ink-muted);
		max-width: 74ch;
	}
	svg {
		display: block;
		width: 100%;
		min-width: 620px;
		height: auto;
	}
	.fig-title {
		fill: var(--ink);
		font-family: var(--serif);
		font-size: 13px;
	}
	.fig-subtitle {
		fill: var(--ink-muted);
		font-family: var(--serif);
		font-size: 9px;
		font-style: italic;
	}
	.baseline {
		stroke: var(--ink);
		stroke-width: 1;
		stroke-dasharray: 3 2;
	}
	.baseline-label {
		fill: var(--ink);
		font-family: var(--mono);
		font-size: 7.5px;
		letter-spacing: 0.04em;
		text-transform: uppercase;
	}
	.row-label {
		fill: var(--ink-muted);
		font-family: var(--mono);
		font-size: 8.5px;
		text-anchor: end;
	}
	.row-label.reading {
		fill: var(--ink);
	}
	.dot {
		stroke: var(--paper);
		stroke-width: 0.8;
	}
	.dot.is-baseline {
		stroke: var(--ink);
		stroke-width: 1;
	}
	.value {
		fill: var(--ink-muted);
		font-family: var(--mono);
		font-size: 8.5px;
		font-variant-numeric: tabular-nums;
	}
	.axis {
		stroke: var(--rule);
		stroke-width: 1;
	}
	.axis-title {
		fill: var(--ink-muted);
		font-family: var(--mono);
		font-size: 8px;
	}
	figcaption {
		margin-top: 0.5rem;
		max-width: 74ch;
	}
	figcaption p {
		margin: 0.2rem 0 0;
		font-family: var(--serif);
		font-size: 0.82rem;
		line-height: 1.5;
		color: var(--ink-muted);
	}
	.method {
		margin-top: 0.4rem;
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
	.method p {
		margin: 0 0 0.55rem;
		font-family: var(--serif);
		font-size: 0.8rem;
		line-height: 1.5;
		color: var(--ink-muted);
	}
	.declared {
		margin: 0.2rem 0 0;
		padding-left: 1.1rem;
	}
	.declared li {
		font-family: var(--mono);
		font-size: 0.66rem;
		color: var(--ink-muted);
		line-height: 1.55;
		/* Long manifest paths wrap instead of widening the page. */
		overflow-wrap: anywhere;
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
	.gate code {
		display: block;
	}
</style>
