<script lang="ts">
	/**
	 * The 2023 paper's own metric, with our arms on the same axis.
	 *
	 * Sits directly ABOVE the tie-inflation figure and deliberately sets it up:
	 * this panel makes no verdict claim, prints no average precision, and shows
	 * no distinct-score counts. Everything it renders is the paper's estimator —
	 * fold-mean trapezoidal PR-AUC ± the POPULATION SD over their 10 folds, which
	 * is a dispersion measure and is labelled as one everywhere it appears.
	 *
	 * Built on the checksum-pinned published-method stack
	 * (`$lib/data/paper-method-landscape` + its loader); the co-plot payload,
	 * geometry, label budgets and per-series encodings all live in
	 * `$lib/data/paper-own-metric` so this file only draws.
	 *
	 * Distinct from `PaperMethodLandscape.svelte`, which is bound to /frontier
	 * with a different job (blocking misuse of these rows in a cost frontier).
	 * Same data module, different figure — neither duplicates the other's parse.
	 */
	import {
		PAPER_OWN_METRIC_GEOMETRY as G,
		PAPER_OWN_METRIC_GROUPS,
		PAPER_OWN_METRIC_GROUP_STYLES,
		fmt3,
		type PaperOwnMetricLoad,
		type PaperOwnMetricMark
	} from '$lib/data/paper-own-metric';

	let { data }: { data: PaperOwnMetricLoad } = $props();

	const figure = $derived(data.status === 'ok' ? data.figure : null);
	const plotBottom = $derived(figure ? figure.height - G.axisPad : 0);

	/**
	 * The reference line: their best row INSIDE the comparable configuration —
	 * never `landscape.best`, which spans configurations. Its label is the one
	 * free-floating string in the plot, so its fit is measured (8px mono =
	 * 4.8165 u/char) and it flips to the left of the line rather than clip.
	 */
	const reference = $derived.by(() => {
		if (!figure) return null;
		const value = figure.comparableBest.fold_mean_trapezoidal_pr_auc;
		const at = x(value, figure.domainMin, figure.domainMax);
		const display = `highest published row on these inputs · ${fmt3(value)}`;
		return { at, display, fits: at + 5 + display.length * G.readoutUnitsPerChar <= G.width };
	});

	function x(value: number, domainMin: number, domainMax: number): number {
		return G.plotLeft + ((value - domainMin) / (domainMax - domainMin)) * (G.plotRight - G.plotLeft);
	}

	/**
	 * Marks in an ANCHOR lane fan vertically so a pairing reads as a pairing.
	 * Every other lane keeps its marks on one baseline — a strip lane carries a
	 * whole configuration (up to 15 rows) and fanning those would spill it into
	 * its neighbours.
	 */
	function markOffset(index: number, total: number, fan: boolean): number {
		return fan && total > 1 ? (index - (total - 1) / 2) * G.fanStep : 0;
	}

	function span(marks: PaperOwnMetricMark[]): { lo: number; hi: number } {
		const values = marks.map((mark) => mark.foldMean);
		return { lo: Math.min(...values), hi: Math.max(...values) };
	}

	function diamond(cx: number, cy: number): string {
		return `M ${cx} ${cy - 4.2} L ${cx + 4.2} ${cy} L ${cx} ${cy + 4.2} L ${cx - 4.2} ${cy} Z`;
	}

	/** Every rendered value, flattened for the inspect table. */
	const rows = $derived(
		(figure?.bands ?? []).flatMap((band) =>
			band.lanes.flatMap((lane) =>
				lane.marks.map((mark) => ({ band, lane, mark }))
			)
		)
	);

	function shortSha(value: string): string {
		return `${value.slice(0, 10)}…`;
	}

	function markValue(mark: PaperOwnMetricMark): string {
		return mark.rounded ? fmt3(mark.foldMean) : mark.foldMean.toFixed(4);
	}
</script>

<section class="own-metric" aria-labelledby="own-metric-title">
	{#if data.status === 'unavailable' || figure === null}
		<div class="gate" role="status">
			<p class="eyebrow">scored on the published measure</p>
			<h2 id="own-metric-title">This comparison could not be loaded</h2>
			<p>{data.reason}</p>
			<code>{data.artifact_path}</code>
		</div>
	{:else}
		<header>
			<div>
				<p class="eyebrow">2023 INDRA assembly paper · scored exactly as published</p>
				<h2 id="own-metric-title">Trapezoidal PR-AUC: published and newly scored methods, one axis</h2>
			</div>
			<strong>published measure, unchanged</strong>
		</header>

		<p class="metric-line">
			{figure.perFoldMetric}, {figure.foldSummary}, reported ± the {figure.uncertaintyField}.
		</p>

		<div class="guard" role="note">
			<strong>That ± is a dispersion measure, not a confidence interval.</strong>
			It is the population standard deviation over the 10 cross-validation folds — the statements
			are split into 10 groups, each fitted model is scored on the group it did not train on, and
			the ± is how far its score moves between groups. The reading models are never fitted on these
			labels, so nothing is withheld from them: each scores every statement once, then takes the same
			10 groups, so one estimator covers both. It belongs to this measure alone, and this figure
			never subtracts one number from another.
		</div>

		<div class="guard slice" role="note">
			<strong>Only one of the four input configurations is comparable to the statements this page
				scores{#if figure.nStatements}, the {figure.nStatements.toLocaleString()} shown above{/if}.</strong>
			Every published method was run four ways: on evidence from <code>readers</code> alone or from
			<code>all sources</code>, each of those with and without <code>include_more_specific</code>.
			These statements are <code>{figure.comparableSlice}</code>, so only those
			{figure.comparableCount} published rows can be set beside the models here. The other
			{figure.contextCount} were run on different evidence and are not shown.
		</div>

		<dl class="anchors">
			<div>
				<dt>the published number</dt>
				<dd>{fmt3(figure.anchor.publishedMean)} <small>± {fmt3(figure.anchor.publishedSd)} between folds · {figure.anchor.publishedLabel}</small></dd>
			</div>
			<div>
				<dt>re-running the published code</dt>
				<dd>{fmt3(figure.anchor.ourMean)} <small>± {fmt3(figure.anchor.ourSd)} between folds · {figure.anchor.ourLabel}</small></dd>
			</div>
			{#if figure.anchor.portMean !== null && figure.anchor.portSd !== null}
				<div>
					<dt>an independent rewrite of that method</dt>
					<dd>{fmt3(figure.anchor.portMean)} <small>± {fmt3(figure.anchor.portSd)} between folds · {figure.anchor.portLabel}</small></dd>
				</div>
			{/if}
		</dl>

		<p class="anchor-note">
			Same code, same data, re-run here — and the published row comes back.{#if figure.anchor.maxAbsDeltaVsPublishedTable6 !== null}{' '}Every
				method reproduced here lands within {figure.anchor.maxAbsDeltaVsPublishedTable6.toFixed(3)} of
				its average in Table 6 of the 2023 paper.{/if} That agreement is what puts both kinds of row
			on one axis.
		</p>

		<figure>
			<!-- The caption names the MEASURE and nothing else. It used to append
			     a second clause naming the two kinds of row, which restated the <h2>
			     six lines up ("published and newly scored methods, one axis") and
			     then restated itself. -->
			<figcaption>Trapezoidal PR-AUC, averaged over 10 folds</figcaption>
			<svg
				viewBox="0 0 {G.width} {figure.height}"
				style:min-height="{figure.height}px"
				role="img"
				aria-labelledby="own-metric-chart-title own-metric-chart-desc"
			>
				<title id="own-metric-chart-title"
					>The methods published in the 2023 INDRA assembly paper and the models scored here, all on
					trapezoidal PR-AUC averaged over 10 folds</title
				>
				<desc id="own-metric-chart-desc"
					>Two bands. The top band is the {figure.comparableCount} published rows measured on the
					{figure.comparableSlice} inputs, the ones these statements match, with the re-runs of the
					published code laid over the two rows reproduced here. The lower band is the models added
					here, on the same statements and the same measure. The other {figure.contextCount}
					published rows were run on different evidence and are not drawn. Horizontal bars are the
					population standard deviation over the 10 folds — how far a score moves between folds,
					not confidence intervals.</desc
				>

				{#each figure.ticks as tick (tick)}
					<line
						class="grid"
						x1={x(tick, figure.domainMin, figure.domainMax)}
						y1={G.topPad}
						x2={x(tick, figure.domainMin, figure.domainMax)}
						y2={plotBottom}
					/>
					<text class="tick" x={x(tick, figure.domainMin, figure.domainMax)} y={plotBottom + 14}
						>{tick.toFixed(2)}</text
					>
				{/each}

				<!-- Their best row INSIDE the comparable configuration. Deliberately
				     stops before the context band: the reference must never appear to
				     reach across configurations. -->
				{#if reference}
					<line
						class="reference"
						x1={reference.at}
						y1={G.topPad}
						x2={reference.at}
						y2={figure.bands[figure.bands.length - 1].headerY}
					/>
					<text
						class="reference-label"
						x={reference.fits ? reference.at + 5 : reference.at - 5}
						y={G.topPad - 5}
						text-anchor={reference.fits ? 'start' : 'end'}>{reference.display}</text
					>
				{/if}

				{#each figure.bands as band (band.id)}
					{#if !band.comparable}
						<rect
							class="context-wash"
							x="0"
							y={band.headerY + 5}
							width={G.width}
							height={band.lanes.length * G.laneHeight + G.bandHeaderHeight - 5}
						/>
					{/if}
					<line class="band-rule" x1="0" y1={band.headerY + 4} x2={G.width} y2={band.headerY + 4} />
					<text class="band-title" class:context={!band.comparable} x="2" y={band.headerY + 17}
						>{band.title}</text
					>
					<text class="band-sub" x="2" y={band.headerY + 27}>{band.subtitle}</text>

					{#each band.lanes as lane (lane.key)}
						{#if lane.anchor}
							<rect
								class="anchor-lane"
								x={G.plotLeft}
								y={lane.y - G.laneHeight / 2}
								width={G.plotRight - G.plotLeft}
								height={G.laneHeight}
							/>
						{/if}
						<!-- 9px mono = 5.4186 u/char; 204-unit gutter = 37 chars, enforced
						     in buildPaperOwnMetric(). Longest shipped label is 35. -->
						<text class="lane-label" x={G.labelAnchorX} y={lane.y + 3}>{lane.display}</text>
						<text class="readout" x={G.readoutX} y={lane.y + 3}>{lane.readout}</text>

						{#if lane.anchor && lane.marks.length > 1}
							<!-- Connector through an anchor lane's marks: vertical when the
							     published row and our re-run agree, visibly skewed if not. -->
							<polyline
								class="pairing"
								points={lane.marks
									.map(
										(mark, index) =>
											`${x(mark.foldMean, figure.domainMin, figure.domainMax)},${lane.y + markOffset(index, lane.marks.length, true)}`
									)
									.join(' ')}
							/>
						{/if}
						{#if lane.strip && lane.marks.length > 1}
							<!-- A strip lane shows the SPAN of its configuration's fold means,
							     not per-row dispersion; those SDs are in the table below. -->
							{@const range = span(lane.marks)}
							{@const strip = PAPER_OWN_METRIC_GROUP_STYLES[lane.marks[0].group]}
							<line
								class="sd"
								x1={x(range.lo, figure.domainMin, figure.domainMax)}
								x2={x(range.hi, figure.domainMin, figure.domainMax)}
								y1={lane.y}
								y2={lane.y}
								stroke={strip.strokeVar}
								stroke-width={strip.strokeWidth}
								stroke-dasharray={strip.dash || undefined}
							/>
						{/if}

						{#each lane.marks as mark, index (mark.key)}
							{@const cy = lane.y + markOffset(index, lane.marks.length, lane.anchor)}
							{@const cx = x(mark.foldMean, figure.domainMin, figure.domainMax)}
							{@const style = PAPER_OWN_METRIC_GROUP_STYLES[mark.group]}
							{#if !lane.strip}
								<line
									class="sd"
									x1={x(mark.foldMean - mark.foldSd, figure.domainMin, figure.domainMax)}
									x2={x(mark.foldMean + mark.foldSd, figure.domainMin, figure.domainMax)}
									y1={cy}
									y2={cy}
									stroke={style.strokeVar}
									stroke-width={style.strokeWidth}
									stroke-dasharray={style.dash || undefined}
								/>
							{/if}
							{#if style.shape === 'diamond'}
								<path class="mark" d={diamond(cx, cy)} fill={style.strokeVar}>
									<title>{mark.title}</title>
								</path>
							{:else if style.shape === 'open-square'}
								<rect
									class="mark"
									x={cx - 3.4}
									y={cy - 3.4}
									width="6.8"
									height="6.8"
									fill="var(--paper)"
									stroke={style.strokeVar}
									stroke-width="1.5"
								>
									<title>{mark.title}</title>
								</rect>
							{:else if style.shape === 'open-circle'}
								<circle class="mark" {cx} {cy} r="2.7" fill="var(--paper)" stroke={style.strokeVar} stroke-width="1.1">
									<title>{mark.title}</title>
								</circle>
							{:else}
								<circle class="mark" {cx} {cy} r="3.3" fill={style.strokeVar} stroke="var(--paper)" stroke-width="0.8">
									<title>{mark.title}</title>
								</circle>
							{/if}
						{/each}
					{/each}
				{/each}

				<line class="axis" x1={G.plotLeft} y1={plotBottom} x2={G.plotRight} y2={plotBottom} />
				<text class="axis-label" x={(G.plotLeft + G.plotRight) / 2} y={plotBottom + 30}
					>trapezoidal PR-AUC, averaged over 10 folds — the published measure →</text
				>
				<!-- 8px mono = 4.8165 u/char; 68 chars = 327.5 units, centred on the
				     600-unit plot, so it never reaches either gutter. The trailing clause
				     ("context line = the range of those averages") went with the band it
				     described: no lane sets `strip`, so no span line is ever drawn and the
				     note was labelling a mark this figure cannot produce. -->
				<text class="axis-note" x={(G.plotLeft + G.plotRight) / 2} y={plotBottom + 42}
					>bars = how far a score moves between folds, NOT a confidence interval</text
				>
			</svg>
		</figure>

		<ul class="legend">
			{#each PAPER_OWN_METRIC_GROUPS as group (group)}
				{@const style = PAPER_OWN_METRIC_GROUP_STYLES[group]}
				<li>
					<svg viewBox="0 0 44 12" aria-hidden="true" class="swatch">
						<line
							x1="2"
							y1="6"
							x2="42"
							y2="6"
							stroke={style.strokeVar}
							stroke-width={style.strokeWidth}
							stroke-dasharray={style.dash || undefined}
						/>
						{#if style.shape === 'diamond'}
							<path d={diamond(22, 6)} fill={style.strokeVar} />
						{:else if style.shape === 'open-square'}
							<rect x="18.6" y="2.6" width="6.8" height="6.8" fill="var(--paper)" stroke={style.strokeVar} stroke-width="1.5" />
						{:else if style.shape === 'open-circle'}
							<circle cx="22" cy="6" r="2.7" fill="var(--paper)" stroke={style.strokeVar} stroke-width="1.1" />
						{:else}
							<circle cx="22" cy="6" r="3.3" fill={style.strokeVar} stroke="var(--paper)" stroke-width="0.8" />
						{/if}
					</svg>
					<span>{style.legend}</span>
				</li>
			{/each}
		</ul>

		<p class="forward">
			Do not read the top of this axis as a verdict: this measure draws straight lines across blocks
			of tied scores, and the reading models tie far more often than the random forests — the figure
			below measures how much of the gap those lines buy.
		</p>

		<details>
			<summary>Inspect every value on this axis</summary>
			<div class="table-scroll">
				<table>
					<thead>
						<tr>
							<th>row</th>
							<th>where it came from</th>
							<th>input setting</th>
							<th>average over the 10 folds</th>
							<th>movement between folds</th>
							<th>comparable to these statements</th>
						</tr>
					</thead>
					<tbody>
						{#each rows as row (`${row.band.id}:${row.lane.key}:${row.mark.key}`)}
							<tr class:context={!row.band.comparable}>
								<td>{row.mark.display}</td>
								<td>{row.mark.rounded ? 'published 2023' : 'scored in this run'}</td>
								<td>{row.band.comparable ? figure.comparableSlice : row.lane.display}</td>
								<td>{markValue(row.mark)}</td>
								<td>{fmt3(row.mark.foldSd)} <small>a spread</small></td>
								<td>{row.band.comparable ? 'yes' : 'no — context only'}</td>
							</tr>
						{/each}
					</tbody>
				</table>
			</div>
			<p class="table-note">
				Published averages are reproduced at the precision they were printed at (three decimals);
				the rows scored here are shown to four so nothing is hidden by rounding. Every spread in this table is a
				population standard deviation over 10 folds — how far the score moves as each fold is swapped
				in.
			</p>
		</details>

		<footer>
			Published rows: pinned notebook commit <code>{shortSha(figure.paperNotebookCommit)}</code> ·
			<code>{data.artifact_path}</code> · artifact
			<span title={data.artifact_sha256}>{shortSha(data.artifact_sha256)}</span>
			{#if figure.paperCodeCommit}<br />Re-run: published analysis code
				<code>{shortSha(figure.paperCodeCommit)}</code>{/if}
			{#if figure.cvProtocol}<br />How the folds were cut: {figure.cvProtocol}{/if}
		</footer>
	{/if}
</section>

<style>
	.own-metric {
		margin-top: 1.5rem;
		padding-top: 1.25rem;
		border-top: 2px solid var(--ink);
	}
	.own-metric > header {
		display: flex;
		justify-content: space-between;
		align-items: start;
		gap: 1rem;
	}
	.own-metric h2,
	.gate h2 {
		margin: 0.18rem 0 0;
		font-family: var(--serif);
		font-size: 1.35rem;
		font-weight: 400;
	}
	.own-metric > header > strong {
		flex: 0 0 auto;
		padding: 0.2rem 0.38rem;
		border: 1px solid var(--accent);
		color: var(--accent);
		font-family: var(--mono);
		font-size: 0.62rem;
		font-weight: 500;
		text-transform: uppercase;
		letter-spacing: 0.04em;
	}
	.eyebrow {
		margin: 0;
		font-family: var(--mono);
		font-size: 0.68rem;
		color: var(--ink-faint);
		text-transform: uppercase;
		letter-spacing: 0.05em;
	}
	.metric-line {
		margin: 0.7rem 0 0;
		font-family: var(--mono);
		font-size: 0.68rem;
		color: var(--ink-muted);
		line-height: 1.5;
	}
	.guard {
		margin-top: 0.7rem;
		padding: 0.75rem 0.9rem;
		border: 1px solid var(--blocked);
		border-left-width: 3px;
		background: color-mix(in srgb, var(--blocked) 3%, transparent);
		font-family: var(--serif);
		font-size: 0.82rem;
		line-height: 1.5;
		color: var(--ink-muted);
	}
	.guard strong {
		color: var(--ink);
	}
	.guard code {
		font-size: 0.74rem;
		color: var(--ink);
	}
	.anchors {
		display: grid;
		grid-template-columns: repeat(3, minmax(0, 1fr));
		gap: 0.8rem;
		margin: 1rem 0 0.5rem;
	}
	.anchors div {
		padding-top: 0.45rem;
		border-top: 1px dotted var(--rule);
	}
	.anchors dt {
		font-family: var(--mono);
		font-size: 0.63rem;
		color: var(--ink-faint);
	}
	.anchors dd {
		margin: 0.2rem 0 0;
		font-family: var(--mono);
		font-size: 0.9rem;
	}
	.anchors small {
		display: block;
		margin-top: 0.12rem;
		font-size: 0.61rem;
		color: var(--ink-faint);
		line-height: 1.35;
	}
	.anchor-note,
	.forward {
		font-family: var(--serif);
		font-size: 0.85rem;
		line-height: 1.5;
		color: var(--ink-muted);
		max-width: 68ch;
	}
	.anchor-note {
		margin: 0 0 0.4rem;
	}
	.forward {
		margin: 0.9rem 0 0;
		padding-left: 0.7rem;
		border-left: 2px solid var(--accent);
	}
	figure {
		margin: 0.9rem 0 0;
	}
	figcaption {
		font-family: var(--mono);
		font-size: 0.68rem;
		color: var(--ink-muted);
	}
	svg {
		display: block;
		width: 100%;
		overflow: visible;
	}
	.grid {
		stroke: var(--rule);
		stroke-width: 1;
	}
	.axis {
		stroke: var(--ink);
		stroke-width: 1;
	}
	.band-rule {
		stroke: var(--ink);
		stroke-width: 0.8;
	}
	.context-wash {
		fill: color-mix(in srgb, var(--ink-muted) 5%, transparent);
	}
	.anchor-lane {
		fill: var(--accent-wash);
	}
	.reference {
		stroke: var(--accent);
		stroke-width: 1;
		stroke-dasharray: 4 3;
		opacity: 0.65;
	}
	.pairing {
		fill: none;
		stroke: var(--accent);
		stroke-width: 1;
		opacity: 0.55;
	}
	.tick,
	.lane-label,
	.readout,
	.axis-label,
	.axis-note,
	.band-title,
	.band-sub,
	.reference-label {
		font-family: var(--mono);
	}
	.tick {
		font-size: 10px;
		fill: var(--ink-faint);
		text-anchor: middle;
	}
	.axis-label {
		font-size: 10px;
		fill: var(--ink-muted);
		text-anchor: middle;
	}
	.axis-note {
		font-size: 8px;
		fill: var(--blocked);
		text-anchor: middle;
	}
	/* Right-anchored: budgeted at 37 chars in buildPaperOwnMetric(). */
	.lane-label {
		font-size: 9px;
		fill: var(--ink-muted);
		text-anchor: end;
	}
	.readout {
		font-size: 8px;
		fill: var(--ink-faint);
		text-anchor: start;
		font-variant-numeric: tabular-nums;
	}
	.band-title {
		font-size: 9.5px;
		fill: var(--ink);
		letter-spacing: 0.03em;
	}
	.band-title.context {
		fill: var(--ink-muted);
	}
	.band-sub {
		font-size: 8px;
		fill: var(--ink-faint);
	}
	.reference-label {
		font-size: 8px;
		fill: var(--accent);
	}
	.legend {
		display: flex;
		flex-wrap: wrap;
		gap: 0.35rem 1.1rem;
		margin: 0.5rem 0 0;
		padding: 0;
		list-style: none;
	}
	.legend li {
		display: flex;
		align-items: center;
		gap: 0.35rem;
		font-family: var(--mono);
		font-size: 0.64rem;
		color: var(--ink-muted);
	}
	.swatch {
		flex: 0 0 auto;
		width: 44px;
		height: 12px;
		overflow: visible;
	}
	details {
		margin-top: 0.8rem;
		border-top: 1px solid var(--rule);
		padding-top: 0.65rem;
	}
	summary {
		cursor: pointer;
		font-family: var(--mono);
		font-size: 0.68rem;
		color: var(--ink-muted);
	}
	.table-scroll {
		overflow-x: auto;
		margin-top: 0.7rem;
	}
	table {
		width: 100%;
		border-collapse: collapse;
		font-family: var(--mono);
		font-size: 0.66rem;
	}
	th {
		color: var(--ink-faint);
		font-weight: 500;
		text-align: left;
	}
	th,
	td {
		padding: 0.38rem 0.45rem;
		border-bottom: 1px dotted var(--rule);
		vertical-align: top;
		white-space: nowrap;
	}
	td:nth-child(n + 4) {
		font-variant-numeric: tabular-nums;
	}
	tr.context td {
		color: var(--ink-faint);
	}
	td small {
		color: var(--blocked);
	}
	.table-note {
		margin: 0.6rem 0 0;
		font-family: var(--mono);
		font-size: 0.62rem;
		line-height: 1.5;
		color: var(--ink-faint);
		max-width: 80ch;
	}
	code,
	footer {
		font-family: var(--mono);
		font-size: 0.62rem;
		color: var(--ink-faint);
	}
	footer {
		margin-top: 0.8rem;
		line-height: 1.6;
	}
	.gate {
		border: 1px solid var(--rule);
		border-left: 3px solid var(--blocked);
		padding: 1rem;
	}
	.gate p:not(.eyebrow) {
		font-family: var(--serif);
		color: var(--ink-muted);
	}
	@media (max-width: 720px) {
		.anchors {
			grid-template-columns: 1fr;
		}
		.own-metric > header {
			display: block;
		}
		.own-metric > header > strong {
			display: inline-block;
			margin-top: 0.6rem;
		}
	}
</style>
