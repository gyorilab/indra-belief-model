<script lang="ts">
	/**
	 * The Table 6 "all sources, specific" block published in the 2023 INDRA
	 * assembly paper, with the newly scored models INTERLEAVED BY RANK rather than
	 * banded beside it.
	 *
	 * WHY IT IS NOT `PaperOwnMetric`. That figure bands the rows by provenance,
	 * which is the right layout for asking whether a re-run lands on the row that
	 * was published, and the wrong one for asking where the newly scored models
	 * sit in the list — banding is precisely what hides that ranks 1, 2 and 3 are
	 * newly scored. Both figures stay; this one answers the ranking question and
	 * defers every reproduction-anchor detail to the one that follows it.
	 *
	 * WHAT THE RANKING METRIC ADDS. The list is ranked on TRAPEZOIDAL PR-AUC — the
	 * precision-recall area taken with straight-line interpolation between points,
	 * averaged over 10 folds — so it interpolates across tied score blocks. Every
	 * row therefore carries its AVERAGE PRECISION (the same area computed
	 * step-wise, so tied scores earn no interpolated credit), its interpolation
	 * credit and its rank on BOTH metrics, side by side — and the INDRA CoGEx
	 * hybrid, itself newly scored, collects what the 2023 rows collect rather than
	 * what the reading models collect, which is the control showing the credit
	 * tracks tie density and not authorship.
	 *
	 * FOLDS, and the asymmetry between the two kinds of row. `StratifiedKFold(10,
	 * shuffle=False)` over all 1,689 statements: nothing is held out as a separate
	 * test set, and each fitted model is scored on the fold it did not train on.
	 * The reading models are never trained and never see a label, so nothing has to
	 * be held out from them — each is scored once over every statement and then
	 * given the same fold indices, so the identical estimator applies to every row.
	 * Both facts are stated on the surface, not just here.
	 *
	 * Every number here is READ off
	 * `data/results/indra_paper_literal_models_20260724/paper_table6_extended.json`.
	 * Geometry, gutter budgets, origin encodings and every fail-closed gate live in
	 * `$lib/data/paper-table6-extended`; this file only draws.
	 */
	import {
		PAPER_TABLE6_GEOMETRY as G,
		PAPER_TABLE6_ORIGINS,
		PAPER_TABLE6_ORIGIN_STYLES,
		fmt3,
		fmt4,
		fmt4Ceil,
		fmtSigned4,
		type MarginStanding,
		type PaperTable6ExtendedLoad,
		type PaperTable6Row
	} from '$lib/data/paper-table6-extended';

	let { data }: { data: PaperTable6ExtendedLoad } = $props();

	/**
	 * The verb for a margin, chosen from the loader's derived direction and never
	 * from the sign of a number this file formats. Sign-blindness has recurred
	 * five times in this project; a render site that decides direction for itself
	 * is how each one shipped.
	 */
	function standingVerb(standing: MarginStanding): string {
		if (standing === 'ahead') return 'leads';
		if (standing === 'behind') return 'TRAILS';
		return 'ties';
	}

	const figure = $derived(data.status === 'ok' ? data.figure : null);
	const rows = $derived(figure?.rows ?? []);

	/**
	 * Census counted off the drawn rows rather than written into the prose, so a
	 * re-generated artifact that changed the answer changes the sentence.
	 */
	const ourRanks = $derived(rows.filter((row) => row.origin === 'ours').map((row) => row.rank));
	const countByOrigin = $derived(
		Object.fromEntries(
			PAPER_TABLE6_ORIGINS.map((origin) => [
				origin,
				rows.filter((row) => row.origin === origin).length
			])
		) as Record<(typeof PAPER_TABLE6_ORIGINS)[number], number>
	);
	/**
	 * The interpolation-credit stub is SIGNED and both signs are drawn: on most of
	 * the paper's own rows the trapezoid scores BELOW average precision, so the stub
	 * runs the other way. Counted off the drawn rows rather than asserted, so the
	 * legend cannot describe a direction the figure has stopped drawing.
	 */
	const giftAdds = $derived(rows.filter((row) => row.tieGift !== null && row.tieGift > 0).length);
	const giftTakes = $derived(rows.filter((row) => row.tieGift !== null && row.tieGift < 0).length);

	/** The boundary between our leading block and the first row of theirs. */
	const boundaryY = $derived(
		figure ? G.topPad + figure.leadingOurRanks * G.laneHeight : G.topPad
	);

	function x(value: number): number {
		if (!figure) return G.plotLeft;
		const span = figure.domainMax - figure.domainMin;
		return G.plotLeft + ((value - figure.domainMin) / span) * (G.plotRight - G.plotLeft);
	}

	function diamond(cx: number, cy: number, r: number): string {
		return `M ${cx} ${cy - r} L ${cx + r} ${cy} L ${cx} ${cy + r} L ${cx - r} ${cy} Z`;
	}

	/** The row's value at the precision the row actually has. */
	function own(row: PaperTable6Row, value: number): string {
		return row.rounded ? fmt3(value) : fmt4(value);
	}

	function shortSha(value: string): string {
		return `${value.slice(0, 10)}…`;
	}
</script>

<section class="table6" aria-labelledby="table6-title">
	{#if data.status === 'unavailable' || figure === null}
		<div class="gate" role="status">
			<p class="eyebrow">trapezoidal PR-AUC, one ranked list</p>
			<h2 id="table6-title">This ranking could not be loaded</h2>
			<p>{data.reason}</p>
			<code>{data.artifact_path}</code>
		</div>
	{:else}
		<header>
			<div>
				<p
					class="eyebrow"
					title="Trapezoidal PR-AUC: the area under the precision-recall curve, taken with straight lines drawn between neighbouring points, averaged over the 10 cross-validation folds. The folds and the curation labels are the ones released with the 2023 INDRA assembly paper."
				>
					trapezoidal PR-AUC · the same 10 folds · the published labels
				</p>
				<h2 id="table6-title">Where the newly scored models land in Table 6</h2>
			</div>
			<!-- The badge states the count it found. With no leading newly scored rank
			     the empty join would render "ranks  newly scored", so that case says so
			     instead. -->
			{#if figure.leadingOurRanks > 0}
				<strong>ranks {ourRanks.slice(0, figure.leadingOurRanks).join(', ')} newly scored</strong>
			{:else}
				<strong>no leading rank is newly scored</strong>
			{/if}
		</header>

		<div class="lede" role="note">
			<p>
				This is Table 6 of the 2023 paper — the <em>{figure.config}</em> block — re-sorted with the
				newly scored models added to it. Everything is ranked on trapezoidal PR-AUC: the area under
				the precision-recall curve, taken with straight lines drawn between neighbouring points and
				averaged over 10 cross-validation folds — the statements are split into 10 groups, each
				fitted model scored on the group it did not train on.
				{#if figure.leadingOurRanks > 0}
					On that measure the newly scored models take the first {figure.leadingOurRanks} places, and
					the best fitted random forest comes {figure.theirBestRank}th.
				{:else}
					On that measure no newly scored model leads: the best fitted random forest comes
					{figure.theirBestRank}th.
				{/if}
				Each of the {figure.reproduction.nRerunRows} rows re-run here from the released 2023 code comes
				back within {fmt4Ceil(figure.reproduction.maxAbsDev)} of the published number, and that
				agreement is the only thing that licenses the shared list. The other
				{countByOrigin.paper_published_only} rows were never re-run and show the figures published in 2023.
			</p>
		</div>

		<figure>
			<figcaption>
				Trapezoidal PR-AUC averaged over {figure.nFolds} folds, ± how far it moves between folds ·
				{figure.nStatements.toLocaleString()} statements.
			</figcaption>
			<svg
				viewBox="0 0 {G.width} {figure.height}"
				style:min-height="{figure.height}px"
				role="img"
				aria-labelledby="table6-chart-title table6-chart-desc"
			>
				<title id="table6-chart-title"
					>Table 6 of the 2023 INDRA assembly paper, the {figure.config} block, with the newly
					scored models sorted into it by rank, showing for each row its trapezoidal PR-AUC — the
					area under the precision-recall curve, taken with straight lines drawn between neighbouring
					points and averaged over the folds — how far that number moves between folds, and, where
					the row's own scores are held here, its average precision, the same area measured in steps
					so that tied scores earn nothing extra, and the credit the straight lines add between the
					two</title
				>
				<!-- The row enumeration inside this <desc> names each row's ORIGIN, which the
				     drawing encodes three ways (colour, mark shape, dash) and the text
				     encoded none of. It is separated by a LEADING space expression per row,
				     never a trailing one: a trailing space before `{/each}` is trimmed by
				     the compiler, which ran the sentences together as "…AP rank 2.rank 2,
				     Gemma 4 26B, …" for the one reader who has nothing but this text. -->
				<desc id="table6-chart-desc"
					>One row per method, ordered 1 to {figure.rows.length} by trapezoidal PR-AUC: the area under
					the precision-recall curve, taken with straight lines drawn between neighbouring points and
					averaged over {figure.nFolds} cross-validation folds — the statements are split into
					{figure.nFolds} groups, and each fitted model is scored on the group it did not train on. The
					reading models are never trained and never see a label, so nothing has to be held out from
					them: each is scored once over every statement and then given the same fold indices, so the
					identical measure applies to every row. Each row shows a mark at that
					average, with a horizontal bar reaching one standard deviation either side; the bar is a
					dispersion measure over {figure.nFolds} folds — how far the number moves from fold to fold —
					and is not a confidence interval. The {figure.reproduction.nRerunRows} rows re-run here from the
					released 2023 code also carry a vertical hairline at the published number, so a
					reproduction shows as a tight pair; the {countByOrigin.paper_published_only} printed-only rows
					carry no hairline, because on those the mark is itself the printed number. Rows whose own
					scores are held here carry a second, lower mark at their average precision — the same area measured
					in steps, so that tied scores earn nothing extra — joined to the first mark by a dashed
					segment whose length is the credit the straight lines add, area awarded across tied scores
					that no cutoff can actually reach. Each row below names where it came from, because that is
					what the colour, the mark shape and the dash all encode. In full, by rank:{#each figure.rows as row (row.rank)}{' '}rank
						{row.rank}, {row.display}, {PAPER_TABLE6_ORIGIN_STYLES[row.origin].spoken},
						{row.metricReadout}, {row.tieReadout}.{/each}</desc
				>

				{#each figure.ticks as tick (tick)}
					<line class="grid" x1={x(tick)} y1={G.headerRuleY + 5} x2={x(tick)} y2={figure.plotBottom} />
					<text class="tick" x={x(tick)} y={figure.plotBottom + 15}>{tick.toFixed(2)}</text>
				{/each}

				<!-- Column headers. All four are budget-checked in buildFigure(); the two
				     right-anchored ones grow LEFT and are measured against their gutter
				     walls (26 and 32 units) rather than against the viewBox. -->
				<text class="col-head" x={G.rankAnchorX} y={G.headerY} text-anchor="end">{figure.rankHeader}</text>
				<text class="col-head" x={G.labelAnchorX} y={G.headerY} text-anchor="end">{figure.labelHeader}</text>
				<text class="col-head" x={G.metricX} y={G.headerY} text-anchor="start">{figure.metricHeader}</text>
				<text class="col-head" x={G.tieX} y={G.headerY} text-anchor="start">{figure.tieHeader}</text>
				<line class="head-rule" x1="0" y1={G.headerRuleY} x2={G.width} y2={G.headerRuleY} />

				{#each figure.rows as row (row.rank)}
					{@const style = PAPER_TABLE6_ORIGIN_STYLES[row.origin]}
					{#if row.isReference}
						<!-- The arm every paired margin on this page is measured against gets a
						     wash rather than another glyph: it is an annotation, not a series. -->
						<rect
							class="reference-wash"
							x="0"
							y={row.y - G.laneHeight / 2}
							width={G.width}
							height={G.laneHeight}
						/>
					{/if}

					<!-- 8px mono = 4.8165 u/char; 26-unit gutter = 5 chars, enforced. -->
					<text class="rank" x={G.rankAnchorX} y={row.y + 3}>{row.rankReadout}</text>
					<!-- 9px mono = 5.4186 u/char; 220-unit gutter = 40 chars, enforced.
					     Longest shipped name is 35 chars = 189.7 units, anchoring at x=62.3. -->
					<text class="name" x={G.labelAnchorX} y={row.y + 3} style:fill={style.strokeVar}
						>{row.display}</text
					>

					<line
						class="sd-bar"
						x1={x(row.foldMean - row.foldSd)}
						x2={x(row.foldMean + row.foldSd)}
						y1={row.y}
						y2={row.y}
						stroke={style.strokeVar}
						stroke-width={style.strokeWidth}
						stroke-dasharray={style.dash || undefined}
					/>
					{#each [row.foldMean - row.foldSd, row.foldMean + row.foldSd] as end, cap (cap)}
						<line
							class="sd-bar"
							x1={x(end)}
							x2={x(end)}
							y1={row.y - G.sdCap}
							y2={row.y + G.sdCap}
							stroke={style.strokeVar}
							stroke-width={style.strokeWidth}
						/>
					{/each}

					{#if row.origin === 'paper_rerun' && row.publishedMean !== null}
						<!-- The reproduction, drawn to scale: the paper's printed value against
						     our re-run of their code. At this axis the worst deviation on the
						     whole table is ~4 units, so agreement reads as a tight pair.
						     RE-RUN ROWS ONLY. A published-only row's mean IS its printed value
						     (the loader gates them identical), so a hairline there would land
						     exactly on the mark and read as a reproduction that never happened —
						     and the <desc> would then describe a different figure from the one
						     drawn, which a screen-reader user cannot check. -->
						<line
							class="published-tick"
							x1={x(row.publishedMean)}
							x2={x(row.publishedMean)}
							y1={row.y - G.publishedTick}
							y2={row.y + G.publishedTick}
						/>
					{/if}

					{#if row.ap !== null}
						<!-- The interpolation credit, drawn as the distance between trapezoidal
						     PR-AUC and average precision. For their models it is invisible; for
						     our coarse-scored readers it is the whole argument against ourselves. -->
						<line
							class="gift"
							x1={x(row.ap)}
							x2={x(row.foldMean)}
							y1={row.y + G.giftOffsetY}
							y2={row.y + G.giftOffsetY}
						/>
						<circle class="ap-ghost" cx={x(row.ap)} cy={row.y + G.giftOffsetY} r="2" />
					{/if}

					{#if style.shape === 'diamond'}
						<path d={diamond(x(row.foldMean), row.y, G.markRadius)} fill={style.strokeVar} />
					{:else if style.shape === 'circle'}
						<circle cx={x(row.foldMean)} cy={row.y} r="3" fill={style.strokeVar} />
					{:else}
						<circle
							cx={x(row.foldMean)}
							cy={row.y}
							r="3"
							fill="var(--paper)"
							stroke={style.strokeVar}
							stroke-width="1.3"
						/>
					{/if}

					<!-- 8px mono; 92-unit gutter = 19 chars and 156-unit gutter = 32 chars,
					     both enforced in buildFigure(). -->
					<text class="readout" x={G.metricX} y={row.y + 3}>{row.metricReadout}</text>
					<text
						class="readout tie"
						class:absent={row.ap === null}
						x={G.tieX}
						y={row.y + 3}>{row.tieReadout}</text
					>

					<!-- One hit target per row, so the full sentence is reachable anywhere
					     along it rather than only on the 3-unit mark. -->
					<rect
						class="hit"
						x="0"
						y={row.y - G.laneHeight / 2}
						width={G.width}
						height={G.laneHeight}
					>
						<title>{row.title}</title>
					</rect>
				{/each}

				{#if figure.leadingOurRanks > 0 && figure.leadingOurRanks < figure.rows.length}
					<line class="boundary" x1="0" y1={boundaryY} x2={G.width} y2={boundaryY} />
					<!-- 8px mono, left-anchored at the plot edge: 41 chars × 4.8165 u/char =
					     197.5 units into a 398-unit plot, so it cannot reach the right gutter.
					     Naming the model rather than its owner cost a character, not a gutter. -->
					<text class="boundary-label" x={G.plotLeft} y={boundaryY - 3}
						>the best fitted random forest starts here</text
					>
				{/if}

				<line class="axis" x1={G.plotLeft} y1={figure.plotBottom} x2={G.plotRight} y2={figure.plotBottom} />
				<!-- 10px mono = 6.0186 u/char, middle-anchored at 463 (the plot centre).
				     69 chars = 415.3 u → [255.4, 670.6], inside the 920-unit viewBox. -->
				<text class="axis-label" x={(G.plotLeft + G.plotRight) / 2} y={figure.plotBottom + 30}
					>trapezoidal PR-AUC — straight lines drawn between neighbouring points</text
				>
				<!-- 8px mono = 4.8165 u/char, middle-anchored at 463. 87 chars = 419.0 u
				     → [253.5, 672.5], inside the viewBox. -->
				<text class="axis-note" x={(G.plotLeft + G.plotRight) / 2} y={figure.plotBottom + 43}
					>bar = how far the score moves across the {figure.nFolds} folds — a spread, not a confidence interval</text
				>
			</svg>
		</figure>

		<ul class="legend">
			{#each PAPER_TABLE6_ORIGINS as origin (origin)}
				{@const style = PAPER_TABLE6_ORIGIN_STYLES[origin]}
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
							<path d={diamond(22, 6, 3.4)} fill={style.strokeVar} />
						{:else if style.shape === 'circle'}
							<circle cx="22" cy="6" r="3" fill={style.strokeVar} />
						{:else}
							<circle cx="22" cy="6" r="3" fill="var(--paper)" stroke={style.strokeVar} stroke-width="1.3" />
						{/if}
					</svg>
					<span>{style.legend} · {countByOrigin[origin]} rows</span>
				</li>
			{/each}
			<li>
				<svg viewBox="0 0 44 12" aria-hidden="true" class="swatch">
					<line class="gift" x1="8" y1="6" x2="36" y2="6" />
					<circle class="ap-ghost" cx="8" cy="6" r="2" />
				</svg>
				<!-- The stub is a SIGNED gap, not an addition. It runs right of the ghost
				     where the trapezoid interpolates area no threshold reaches, and LEFT of
				     it where the trapezoid scores below average precision — which is what
				     most of the fine-scored published rows do. Both counts are read
				     off the drawn rows, so the legend cannot outlive the drawing. -->
				<span
					title="Average precision is the same precision-recall area as trapezoidal PR-AUC, measured in steps rather than with straight lines drawn between points, so tied scores earn nothing extra."
					>average precision; the dashed stub is the gap between it and trapezoidal PR-AUC — it runs
					right of the hollow mark on the {giftAdds} rows where the straight lines add credit, and left
					of it on the {giftTakes} rows where they take some away</span
				>
			</li>
			<li>
				<svg viewBox="0 0 44 12" aria-hidden="true" class="swatch">
					<line class="published-tick" x1="22" y1="1" x2="22" y2="11" />
				</svg>
				<span
					>the published number, on the rows that were re-run. Three decimals means the figure is
					the one printed in 2023; four means it was measured here</span
				>
			</li>
		</ul>

		<div class="checks">
			<article>
				<h3>what the straight lines add</h3>
				<p>
					Those straight lines are drawn straight across tied scores, and the reading models pile
					hundreds of statements onto a handful of values. The credit comes to
					{fmtSigned4(figure.tie.readers.min)} to {fmtSigned4(figure.tie.readers.max)} for the
					{figure.tie.readers.count} reading models, and
					{fmtSigned4(figure.tie.paperRerun.min)} to {fmtSigned4(figure.tie.paperRerun.max)} for the
					{figure.tie.paperRerun.count} published rows whose scores are held here.
					<!-- BOTH MODELS NAMED IN THE OPEN. The two margins belong to DIFFERENT
					     newly scored models — that is the substance, not a footnote — and the
					     disambiguation used to live only inside the collapsed <details>.
					     Unnamed, a reader looks up the printed AP, lands on the
					     tie-corrected model's row, and finds a ranking-column margin that is
					     not the one this sentence quotes. -->
					Counted without that credit, the strongest model by average precision
					({figure.tie.best.ourApDisplay}) scores {fmt4(figure.tie.best.ourAp)} against the best
					published row's {fmt4(figure.tie.best.theirAp)} — a margin of {fmtSigned4(figure.tie.best.apMargin)}, not the
					{fmtSigned4(figure.tie.best.paperMetricMargin)} the ranking column shows for
					{figure.tie.best.ourPaperMetricDisplay}.
				</p>
				<p>
					The control: the newly scored {figure.tie.control.display} gives
					{figure.tie.control.distinctScores.toLocaleString()} different scores and collects
					{fmtSigned4(figure.tie.control.tieGift)} — inside the published range, not the reading
					models'. Scores that come in a few coarse steps collect the credit; finely spread ones do
					not, whoever built them.
				</p>
			</article>

			<article>
				<h3>how to read the ± spread</h3>
				<p>
					<!-- The plain half, not the shipped one: the shipped bytes are sha-pinned
					     and belong behind the audit boundary, and the restatement already
					     carries the fold count, so the sentence around it must not state a
					     second one. -->
					The ± is the {figure.metric.uncertaintyFieldProse.plain}. It says how far the number moves from one
					fold to the next; it does not shrink as more folds are added, and no row here is a
					statistical test. None was run in 2023: {figure.convention.nPublishedRows} rows,
					{figure.convention.nPValues} p-values, {figure.convention.nConfidenceIntervals} intervals,
					{figure.convention.nMultiplicityCorrections} corrections for testing many things at once.
				</p>
				<p>
					The headline result published in 2023 — the best fitted random forest over the unfitted
					starting point (<span title="Noisy-OR over per-source reliabilities: one fixed reliability per source, combined as 1 − Π(1 − r); nothing is fitted to labels."
						>noisy-OR: one fixed reliability per source, nothing fitted to the labels</span
					>), on the same inputs — is {fmtSigned4(figure.convention.headline.gain)}, or
					{figure.convention.headline.gainInFoldSd.toFixed(2)} times that fold-to-fold movement, and
					it too was published with no statistical test. That is the scale to read every row above
					against.
				</p>
			</article>
		</div>

		{#if figure.ambiguousPairs.length > 0 || figure.printedTies.length > 0}
			<div class="caveats">
				<ul>
					<!-- Keyed on the pair's rank-derived id: with an all-pairs scan one row
					     can appear in several pairs, so a display name is not a unique key. -->
					{#each figure.ambiguousPairs as pair (pair.id)}
						<li>
							{pair.higherDisplay} and {pair.lowerDisplay} are {fmt4(pair.gap)} apart — closer together
							than the three decimals the paper printed, so which of them comes first cannot be settled
							from what was published.
						</li>
					{/each}
					{#each figure.printedTies as tie (tie.id)}
						<li>
							{tie.higherDisplay} and {tie.lowerDisplay} both print {tie.readout}; they differ by
							{tie.gap.toExponential(1)} and are ordered on the full-precision values below.
						</li>
					{/each}
				</ul>
			</div>
		{/if}

		<details>
			<summary>Inspect every row, and the reasoning behind the ranking</summary>
			<div class="table-scroll">
				<table>
					<thead>
						<tr>
							<th>rank</th>
							<th>method</th>
							<th>where it came from</th>
							<th>trapezoidal PR-AUC</th>
							<th>movement between folds</th>
							<th>published</th>
							<th>gap from the published number</th>
							<th>average precision</th>
							<th>credit from the straight lines</th>
							<th>different scores it gives</th>
							<th>rank on average precision</th>
						</tr>
					</thead>
					<tbody>
						{#each figure.rows as row (row.rank)}
							<tr class:reference={row.isReference}>
								<td>{row.rank}</td>
								<td
									>{row.display}{row.headToHeadDisplay === null
										? ''
										: ` — ${row.headToHeadDisplay} elsewhere on this page`}{row.isReference
										? ' (what every difference on this page is measured against)'
										: ''}</td
								>
								<td>{figure.prose.origins[row.origin].plain}</td>
								<td>{own(row, row.foldMean)}</td>
								<td>{own(row, row.foldSd)}</td>
								<td>{row.publishedMean === null ? 'never published' : fmt3(row.publishedMean)}</td>
								<td
									>{row.absDevVsPublished === null
										? 'not re-run'
										: fmt4(row.absDevVsPublished)}</td
								>
								<td>{row.ap === null ? 'we do not hold its scores' : fmt4(row.ap)}</td>
								<td>{row.tieGift === null ? 'we do not hold its scores' : fmtSigned4(row.tieGift)}</td>
								<td
									>{row.distinctScores === null
										? 'we do not hold its scores'
										: row.distinctScores.toLocaleString()}</td
								>
								<td>{row.apRank === null ? 'not ranked' : row.apRank}</td>
							</tr>
						{/each}
					</tbody>
				</table>
			</div>
			<p class="table-note">
				Trapezoidal PR-AUC is the area under the precision-recall curve, taken with straight lines
				drawn between neighbouring points, averaged over the {figure.nFolds} folds the statements were
				split into for the 2023 INDRA assembly paper. Average precision is the same area measured in
				steps, so tied scores earn nothing extra; the credit is the difference between the two — area
				awarded across tied scores that no score cutoff can actually reach.
			</p>
			<!-- THE PLAIN HALF OF EVERY TWIN, not the shipped half. These notes used to
			     draw the artifact's own bytes, which is how a whole vocabulary of
			     possessives stayed on the screen through three rewrites: no static scan
			     reads a string that arrives at runtime. The plain halves are authored in
			     `$lib/data/paper-table6-extended`, are checked against their source for
			     numbers, denials and named models by `npm run test:paper-prose`, and name
			     each method rather than whose it is. The shipped bytes stay in the
			     artifact, which the footer pins by sha. -->
			<p class="table-note">
				How each number was computed. {figure.prose.perFoldMetric.plain}
				{figure.prose.summary.plain}
				{figure.prose.uncertaintyNote.plain}
			</p>
			<p class="table-note">{figure.prose.trapezoidalNote.plain}</p>
			<p class="table-note">How the rows are ordered: {figure.prose.rankingRule.plain}</p>
			<p class="table-note">
				Reproduction: {figure.reproduction.nRerunRows} rows re-run from the released 2023 code,
				{figure.reproduction.nRerunRowsWithScores} of them with the scores released alongside it — each
				statement scored out of fold, that is by a copy of the model fitted on the other nine folds, so
				no fitted model is ever scored on a statement it learned from. Nothing is held back as a
				separate test set. The reading models are the asymmetric case: they are never trained and never
				see a label, so there is nothing to hold out from them — each reads every statement once and is
				then given the same fold indices, only so that the identical measure applies to every row. The
				worst gap from a published value is {fmt4(figure.reproduction.maxAbsDev)} on
				{figure.reproduction.maxAbsDevDisplay}, against a {fmt4(figure.reproduction.tolerance)} bound.
				{figure.prose.reproductionNote.plain}
			</p>
			<p class="table-note">
				{figure.prose.tieDefinition.plain}
				{figure.prose.tieReconciliation.plain}
			</p>
			<p class="table-note">
				{figure.prose.tieSeparationNote.plain} Methods whose scores come in coarse steps give at most
				{figure.tie.separation.coarseMaxDistinctScores.toLocaleString()} different scores; the
				{figure.tie.separation.nCoarse} of them collect at least
				{fmtSigned4(figure.tie.separation.minGiftAmongCoarse)}, while the
				{figure.tie.separation.nFine} rows with finely spread scores never move more than
				{fmt4(figure.tie.separation.maxAbsGiftAmongFine)} either way. The rank correlation (Spearman)
				between that credit and how many different scores a method gives is
				{figure.tie.spearman.toFixed(3)}. {figure.prose.tieSpearmanMethod.plain}
			</p>
			<!-- The artifact's own reconciliation sentence names its models by frozen
			     row label, so it is restated here from the resolved display names rather
			     than printed: a join key shown to a reader is a key someone will
			     reasonably ask us to rename. -->
			<p class="table-note">
				<!-- VERB FROM THE LOADER, NOT FROM A FORMATTED NUMBER. This sentence
				     hard-coded "leads" and "the lead survives the correction" while the
				     loader gated both margins only against the two rows they name, never
				     against zero — so a self-consistent artifact with every gate green
				     rendered "leads the published row by −0.0010. The lead survives the
				     correction." That was sign-blindness occurrence #5. -->
				On trapezoidal PR-AUC the highest-ranked newly scored model
				({figure.tie.best.ourPaperMetricDisplay})
				{standingVerb(figure.tie.best.paperMetricStanding)} the highest-ranked published row
				({figure.tie.best.theirPaperMetricDisplay}) by
				{fmtSigned4(figure.tie.best.paperMetricMargin)}. On average precision, the strongest newly
				scored model ({figure.tie.best.ourApDisplay})
				{standingVerb(figure.tie.best.apStanding)} the best published row on that measure
				({figure.tie.best.theirApDisplay}) by
				{fmtSigned4(figure.tie.best.apMargin)}.
				{#if figure.tie.best.paperMetricStanding === 'ahead' && figure.tie.best.apStanding === 'ahead'}
					The lead survives once the straight-line credit is taken away; most of its apparent size
					does not.
				{:else if figure.tie.best.paperMetricStanding === 'ahead'}
					The lead on trapezoidal PR-AUC does NOT survive once the straight-line credit is taken
					away.
				{/if} Every difference elsewhere on this page is measured against
				{figure.tie.best.referenceDisplay}.
			</p>
			<p class="table-note">
				{figure.prose.headlineNote.plain} On the tie at the top: {figure.prose.headlineTieBreak.plain}.
				{figure.prose.conventionVerified.plain}
			</p>
			<p class="table-note">
				Where the measure itself comes from: {figure.prose.estimatorSource.plain} The benchmark is the
				{figure.nStatements.toLocaleString()} statements, {figure.nPositive.toLocaleString()} of them
				marked correct and {figure.nNegative.toLocaleString()} marked incorrect by the curators whose
				labels were released in 2023, scored under the same {figure.nFolds} folds.
				{figure.prose.whatThisIs.plain}
			</p>
		</details>

		<footer>
			<!-- No '—' placeholder for a missing digest: an unpinned artifact says so,
			     because a dash beside a path reads as "pinned, just abbreviated". -->
			<code>{data.artifact_path}</code> · artifact
			{#if data.artifact_sha256}<span title={data.artifact_sha256}>{shortSha(data.artifact_sha256)}</span
				>{:else}<span class="sha-missing">not sha-pinned</span>{/if}<br />
			Generated by <code>{figure.generatedBy}</code> · published code
			<code>{figure.convention.repository}</code>
			<span title={figure.convention.commit}>{shortSha(figure.convention.commit)}</span> ·
			<code>{figure.convention.notebookPath}</code>
			<span title={figure.convention.notebookSha256}>{shortSha(figure.convention.notebookSha256)}</span>
		</footer>
	{/if}
</section>

<style>
	.table6 {
		margin-top: 1.5rem;
		padding-top: 1.25rem;
		border-top: 2px solid var(--ink);
	}
	.table6 > header {
		display: flex;
		justify-content: space-between;
		align-items: start;
		gap: 1rem;
	}
	.table6 h2,
	.gate h2 {
		margin: 0.18rem 0 0;
		font-family: var(--serif);
		font-size: 1.35rem;
		font-weight: 400;
	}
	.table6 > header > strong {
		flex: 0 0 auto;
		padding: 0.2rem 0.38rem;
		border: 1px solid var(--blocked);
		color: var(--blocked);
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
	.lede {
		margin-top: 0.9rem;
		padding: 0.75rem 0.9rem;
		border: 1px solid var(--blocked);
		border-left-width: 3px;
		background: color-mix(in srgb, var(--blocked) 3%, transparent);
		font-family: var(--serif);
		font-size: 0.85rem;
		line-height: 1.55;
		color: var(--ink-muted);
	}
	.lede p {
		margin: 0;
		max-width: 74ch;
	}
	figure {
		margin: 1.1rem 0 0;
	}
	figcaption {
		font-family: var(--mono);
		font-size: 0.68rem;
		color: var(--ink-muted);
		line-height: 1.5;
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
	.head-rule {
		stroke: var(--ink);
		stroke-width: 1;
	}
	.boundary {
		stroke: var(--ink);
		stroke-width: 1.2;
		stroke-dasharray: 6 3;
	}
	.reference-wash {
		fill: color-mix(in srgb, var(--accent) 5%, transparent);
	}
	.sd-bar {
		stroke-linecap: butt;
	}
	.published-tick {
		stroke: var(--accent);
		stroke-width: 1.6;
	}
	.gift {
		stroke: var(--ink-faint);
		stroke-width: 1;
		stroke-dasharray: 2 1.5;
	}
	.ap-ghost {
		fill: var(--paper);
		stroke: var(--ink-faint);
		stroke-width: 1.1;
	}
	.hit {
		fill: transparent;
	}
	.tick,
	.rank,
	.name,
	.readout,
	.col-head,
	.axis-label,
	.axis-note,
	.boundary-label {
		font-family: var(--mono);
	}
	.tick {
		font-size: 9px;
		fill: var(--ink-faint);
		text-anchor: middle;
	}
	/* Right-anchored: budgeted at 5 chars in buildFigure(). */
	.rank {
		font-size: 8px;
		fill: var(--ink-faint);
		text-anchor: end;
		font-variant-numeric: tabular-nums;
	}
	/* Right-anchored: budgeted at 40 chars in buildFigure(). */
	.name {
		font-size: 9px;
		text-anchor: end;
	}
	/* Left-anchored: budgeted at 19 and 32 chars in buildFigure(). */
	.readout {
		font-size: 8px;
		fill: var(--ink-muted);
		text-anchor: start;
		font-variant-numeric: tabular-nums;
	}
	.readout.tie {
		fill: var(--ink-faint);
	}
	/* An absent number states its reason; --blocked is this page's caveat token,
	   so it never reads as a measured value in the same column. */
	.readout.absent {
		fill: var(--blocked);
	}
	.col-head {
		font-size: 8px;
		fill: var(--ink);
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
	.boundary-label {
		font-size: 8px;
		fill: var(--ink);
		text-anchor: start;
	}
	.legend {
		display: flex;
		flex-direction: column;
		gap: 0.3rem;
		margin: 0.6rem 0 0;
		padding: 0;
		list-style: none;
	}
	.legend li {
		display: flex;
		align-items: center;
		gap: 0.45rem;
		font-family: var(--mono);
		font-size: 0.64rem;
		line-height: 1.4;
		color: var(--ink-muted);
	}
	.swatch {
		flex: 0 0 auto;
		width: 44px;
		height: 12px;
		overflow: visible;
	}
	.checks {
		display: grid;
		grid-template-columns: repeat(2, minmax(0, 1fr));
		gap: 1.1rem;
		margin-top: 1.2rem;
	}
	.checks article {
		padding-top: 0.5rem;
		border-top: 1px solid var(--rule);
	}
	.checks h3 {
		margin: 0 0 0.4rem;
		font-family: var(--mono);
		font-size: 0.7rem;
		font-weight: 500;
		letter-spacing: 0.03em;
		color: var(--ink);
		text-transform: uppercase;
	}
	.checks p {
		margin: 0 0 0.5rem;
		font-family: var(--serif);
		font-size: 0.82rem;
		line-height: 1.55;
		color: var(--ink-muted);
	}
	.caveats {
		margin-top: 1.1rem;
		padding: 0.8rem 0.95rem;
		border: 1px solid var(--blocked);
		border-left-width: 3px;
		background: color-mix(in srgb, var(--blocked) 3%, transparent);
	}
	.caveats ul {
		margin: 0;
		padding-left: 1.1rem;
	}
	.caveats li {
		font-family: var(--serif);
		font-size: 0.82rem;
		line-height: 1.55;
		color: var(--ink-muted);
		max-width: 78ch;
	}
	.caveats li + li {
		margin-top: 0.45rem;
	}
	details {
		margin-top: 1rem;
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
		padding: 0.34rem 0.45rem;
		border-bottom: 1px dotted var(--rule);
		vertical-align: top;
		white-space: nowrap;
	}
	td:nth-child(n + 4) {
		font-variant-numeric: tabular-nums;
	}
	tr.reference td {
		background: color-mix(in srgb, var(--accent) 5%, transparent);
	}
	.table-note {
		margin: 0.6rem 0 0;
		font-family: var(--mono);
		font-size: 0.62rem;
		line-height: 1.55;
		color: var(--ink-faint);
		max-width: 88ch;
	}
	code,
	footer {
		font-family: var(--mono);
		font-size: 0.62rem;
		color: var(--ink-faint);
	}
	footer {
		margin-top: 0.9rem;
		line-height: 1.6;
	}
	.sha-missing {
		color: var(--blocked);
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
		.checks {
			grid-template-columns: 1fr;
		}
		.table6 > header {
			display: block;
		}
		.table6 > header > strong {
			display: inline-block;
			margin-top: 0.6rem;
		}
	}
</style>
