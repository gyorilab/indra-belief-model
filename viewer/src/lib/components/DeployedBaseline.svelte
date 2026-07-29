<script lang="ts">
	/**
	 * AGAINST INDRA'S OWN BELIEF — four panels, four named comparators.
	 *
	 * PROP CONTRACT. `data` is the `DeployedBaselineLoad` returned by
	 * `loadDeployedBaseline()` in `$lib/server/paper-deployed-baseline`. Pass it
	 * straight through from the page's server load; this component reads no file,
	 * fetches nothing, and holds no state.
	 *
	 * WHAT THE FIGURE HAS TO LAND WITHOUT A CAPTION. Each row is one gold panel.
	 * The open square is the STRONGEST form of INDRA's own belief that panel can
	 * source. The filled dot is the same INDRA scorer over the evidence a reader
	 * kept. The accent bar between them is what reading bought. Four rows, four
	 * independently-sourced golds, every bar pointing the same way.
	 *
	 * AND THE THING THE OLD FIGURE GOT WRONG. It read as "the same comparison,
	 * four times", and it was not: the comparator differs by panel, and so does
	 * the evidence per statement. Two lines fix that, both drawn, neither in a
	 * method note. Under every row label sits `vs <the form of INDRA belief this
	 * row was measured against>`. Under every bar sits that panel's composition —
	 * evidence read per statement, what the incumbent was scored over, curators,
	 * class balance, in/out-of-sample, join. The faint ticks to the LEFT of each
	 * bar are the other forms of INDRA's belief on that panel; the argmax declined
	 * them because they are weaker, the dashed segment is the margin that refusal
	 * forfeits, and the readout column prints its price.
	 *
	 * THE PAPER'S RF IS DRAWN ON ITS OWN PANEL ONLY, as a diamond ON the bar, so
	 * the bar is visibly cut into two segments whose values are both printed: the
	 * gain over the strongest INDRA belief, and the smaller gain over their fitted
	 * RESEARCH model. Both are true, and neither can be read without the other.
	 * The diamond cannot appear elsewhere — the data contract requires the paper's
	 * released out-of-fold predictions, which exist on that panel alone.
	 *
	 * PROSE DISCIPLINE. This page's word budget is a design constraint. Every
	 * on-screen string here comes from the validated artifact through the builder,
	 * where it is character-budgeted against its own gutter; nothing reader-facing
	 * is typed in this file. Everything explanatory lives in `<desc>` (the
	 * screen-reader equivalent of the figure) or behind `<details>` — neither is
	 * counted, and both carry the full method note.
	 *
	 * NO NUMBER IS TYPED IN THIS FILE. Every AUROC, delta, interval, count, census
	 * and panel name comes from the validated artifact through the fail-closed
	 * loader; the only constants here are SVG layout offsets read from the shared
	 * geometry in `$lib/data/paper-deployed-baseline`.
	 */
	import {
		DEPLOYED_BASELINE_GEOMETRY as G,
		fmt1,
		fmt3,
		pct0,
		signed3,
		type DeployedBaselineLoad,
		type DeployedBaselineRow
	} from '$lib/data/paper-deployed-baseline';

	let { data }: { data: DeployedBaselineLoad } = $props();

	const figure = $derived(data.status === 'ok' ? data.figure : null);
	const payload = $derived(figure?.data ?? null);

	const MARK_HALF = G.markRadius;
	const SEGMENT_DY = 12;

	const axisY = $derived(figure ? G.rowsTop + figure.rows.length * G.rowHeight : 0);

	/** Two-decimal SVG user units — coordinate rounding only, never a metric. */
	function r2(value: number): number {
		return Number(value.toFixed(2));
	}

	/** The diamond's x on the paper row, or null where no research model exists. */
	function researchX(row: DeployedBaselineRow): number | null {
		return row.marks.find((mark) => mark.series === 'research')?.x ?? null;
	}
</script>

<section class="deployed-baseline" aria-labelledby="deployed-baseline-title">
	{#if data.status !== 'ok' || figure === null || payload === null}
		<div class="gate" role="status">
			<p class="eyebrow">the belief model INDRA ships today</p>
			<h2 id="deployed-baseline-title">This comparison is unavailable</h2>
			<p>{data.reason ?? data.artifact_path}</p>
			<code>{data.artifact_path}</code>
		</div>
	{:else}
		<h2 id="deployed-baseline-title" class="sr">
			Evidence-gated reading against the strongest form of INDRA's own belief, on four independently
			curated sets of statements
		</h2>
		<figure>
			<div class="scroller">
				<svg
					viewBox="0 0 {G.width} {figure.height}"
					preserveAspectRatio="xMidYMid meet"
					role="img"
					aria-labelledby="deployed-baseline-chart-title deployed-baseline-chart-desc"
				>
					<title id="deployed-baseline-chart-title"
						>{payload.prose.claim.plain}</title
					>
					<desc id="deployed-baseline-chart-desc"
						>{figure.rows.length} horizontal rows on one shared axis of AUROC — the chance that a
						correct statement picked at random scores above a wrong one picked at random — running
						from {fmt3(figure.domainMin)} (pure chance) to {fmt3(figure.domainMax)}. Each row is one
						independently curated set of statements, and each names the model it is measured against.
						{figure.regimeLine}. {payload.prose.claimIsNot.plain}
						There are two families of INDRA belief here and both are INDRA's own.
						{#each payload.families as family (family.key)}{family.display}: {family
								.whatItComputesProse.plain} It is {family.fitted ? 'fitted' : 'unfitted'}, and it is
							{family.whereItRunsProse.plain}. It ships in
							{family.shipsIn}.
						{/each}
						{payload.servedBeliefIdentity.questionProse.plain}
						{payload.servedBeliefIdentity.findingProse.plain}
						{payload.servedBeliefIdentity.floorDerivationProse.plain}
						The open square on each row is the strongest form of INDRA's own belief this set of
						statements can support; {payload.prose.incumbentSelectionRule.plain}
						{payload.prose.incumbentSelectionRuleCost.plain}
						The filled dot is {payload.arms.gate.display} — {payload.arms.gate.whatItIsProse.plain}.
						The accent bar between them is what reading bought. Faint ticks left of the bar are the
						forms the rule declined. On the set published in 2023 a diamond marks
						{payload.arms.research.display}: {payload.arms.research.whatItIsProse.plain}
						{#each figure.rows as row (row.key)}Set {row.display}: {row.panel
								.nStatements} statements, {row.panel.nCorrect} correct and {row.panel.nErrors}
							wrong, labelled by {row.panel.labelNoteProse.plain}, curated by {row.panel
								.curatorNoteProse.plain}. The
							reading model saw {fmt1(
								row.panel.heterogeneity.evidenceReadsPerStatement.mean
							)} pieces of evidence per statement, median {fmt1(
								row.panel.heterogeneity.evidenceReadsPerStatement.median
							)}, at most {row.panel.heterogeneity.evidenceReadsPerStatement.max}, and {row.panel
								.heterogeneity.evidenceReadsPerStatement.nSingle} of {row.panel.heterogeneity
								.evidenceReadsPerStatement.nStatements} statements ({pct0(
								row.panel.heterogeneity.evidenceReadsPerStatement.shareSingle
							)}) rest on exactly one piece, where reading has nothing to combine and must keep or
							drop on a single sentence.
							{#if row.panel.heterogeneity.readerSawFullEvidence}That is the statement's whole
								evidence, so the shipped model and the reading model saw the same material here.{:else if row
									.panel.heterogeneity.corpusEvidencePerStatement !== null}The statement carries
								{fmt1(
									row.panel.heterogeneity.corpusEvidencePerStatement.mean
								)} pieces of evidence in the corpus, so the reading model saw {pct0(
									row.panel.heterogeneity.readerEvidenceShareOfCorpus ?? 0
								)} of what the shipped model was scored over.{:else if row.panel.heterogeneity
									.corpusEvidenceAbsentBecauseProse}{row.panel.heterogeneity
									.corpusEvidenceAbsentBecauseProse.plain}{/if}
							The join was: {row.panel.heterogeneity.joinModeProse.plain}.
							{#if row.panel.heterogeneity.nUndefinedExcluded > 0}{row.panel.heterogeneity
									.nUndefinedExcluded} statements had no defined belief and were excluded rather than
								imputed.{/if}
							Measured against: {row.panel.incumbent.display}, AUROC {fmt3(
								row.panel.incumbent.auroc
							)}, which is {row.panel.incumbent.fitted ? 'fitted to data' : 'not fitted to data'}.
							Evidence-gated reading scores AUROC {fmt3(row.panel.gate.auroc)}; the difference is {signed3(
								row.panel.deltaAuroc
							)} with a paired 95 percent interval of {signed3(row.panel.bootstrap.ci95Low)} to {signed3(
								row.panel.bootstrap.ci95High
							)}.
							{#each row.panel.incumbentVariants as variant (variant.key)}{variant.display} scores {fmt3(
									variant.auroc
								)} here; reading beats it by {signed3(variant.deltaAuroc)} [{signed3(
									variant.bootstrap.ci95Low
								)}, {signed3(variant.bootstrap.ci95High)}]. {variant.whatItComputesProse.plain}
								{#if variant.provenanceCaveatProse}Where it came from: {variant.provenanceCaveatProse
										.plain}.{/if}
								{#if variant.crossCheck}{variant.crossCheck.whyItDiffersProse.plain}{/if}
							{/each}
							{#if row.panel.incumbentVariants.length > 1}Choosing the strongest of those gives away {fmt3(
									row.panel.selectionCostAuroc
								)} AUROC of margin we would have had against the weakest.{/if}
							{#if row.panel.evidenceMatchedControl}{row.panel.evidenceMatchedControl.display} — the
								same INDRA scorer restricted to the evidence the reading model actually saw — reaches
								only {fmt3(row.panel.evidenceMatchedControl.auroc)} here, {row.panel
									.evidenceMatchedControl.atOrBelowChance
									? 'at or below chance'
									: 'above chance'}. {row.panel.evidenceMatchedControl.noteProse
									.plain}{:else if row.panel.evidenceMatchedControlAbsentBecauseProse}{row.panel
									.evidenceMatchedControlAbsentBecauseProse.plain}{/if}
							{#if row.panel.researchModel}The random forest fitted in 2023 reaches AUROC {fmt3(
									row.panel.researchModel.auroc
								)} on the same statements, which is {signed3(
									row.panel.researchModel.deltaResearchMinusIncumbent
								)} above the model it is measured against and {signed3(
									row.panel.researchModel.deltaGateMinusResearch
								)} below evidence-gated reading.{/if}
							{#if row.panel.inSampleNoteProse}{row.panel.inSampleNoteProse.plain}{/if}
						{/each}</desc
					>

					<!-- header block, left gutter; the legend has its own band at legendMarkX -->
					<text class="fig-title" x={G.titleX} y="22">{figure.title}</text>
					<!--
						The replication counts ride HERE rather than in a figcaption: a
						caption restating what four right-pointing bars already show is
						prose the page cannot afford, and the per-row intervals are printed
						beside every bar anyway. Both lines are builder-computed counts.
					-->
					<text class="fig-subtitle" x={G.headerX} y="40">{figure.headline}</text>
					<text class="fig-subtitle" x={G.headerX} y="52">{figure.subheadline}</text>
					<!--
						The evidence-regime span. Four wins across four copies of one
						regime is a much weaker result than four wins across a 16-fold
						spread, so the spread is drawn beside the count, not left to the
						method note. Builder-computed, budget-checked, y from geometry.
					-->
					<text class="fig-subtitle regime" x={G.headerX} y={G.regimeLineY}>{figure.regimeLine}</text>

					<!-- legend: four marks, four lines, all of them data -->
					<g class="legend">
						{#each figure.legend as entry (entry.series)}
							{#if entry.series === 'incumbent'}
								<rect
									x={G.legendMarkX - MARK_HALF}
									y={entry.y - MARK_HALF - 3}
									width={MARK_HALF * 2}
									height={MARK_HALF * 2}
									class="mark-incumbent"
								/>
							{:else if entry.series === 'gate'}
								<circle cx={G.legendMarkX} cy={entry.y - 3} r={MARK_HALF} class="mark-gate" />
							{:else if entry.series === 'research'}
								<path
									d="M {G.legendMarkX} {entry.y - 3 - MARK_HALF} L {G.legendMarkX +
										MARK_HALF} {entry.y - 3} L {G.legendMarkX} {entry.y - 3 + MARK_HALF} L {G.legendMarkX -
										MARK_HALF} {entry.y - 3} Z"
									class="mark-research"
								/>
							{:else}
								<line
									x1={G.legendMarkX}
									y1={entry.y - 3 - G.declinedHalfHeight}
									x2={G.legendMarkX}
									y2={entry.y - 3 + G.declinedHalfHeight}
									class="mark-declined"
								/>
							{/if}
							<text class="legend-text" x={G.legendTextX} y={entry.y}>{entry.text}</text>
						{/each}
					</g>

					<!-- rows -->
					{#each figure.rows as row (row.key)}
						{@const trackY = row.y + G.trackDy}
						{@const rx = researchX(row)}
						<g class="row" class:largest={row.largest}>
							<text class="row-label" x={G.labelAnchorX} y={row.y + G.labelDy}>{row.display}</text>
							<text class="row-sub" x={G.labelAnchorX} y={row.y + G.subLabelDy}>{row.subLabel}</text>
							<text class="row-comparator" x={G.labelAnchorX} y={row.y + G.comparatorDy}
								>{row.comparatorLabel}</text
							>
							{#if row.chip !== ''}
								<text class="row-chip" x={G.labelAnchorX} y={row.y + G.chipDy}>{row.chip}</text>
							{/if}

							<!-- full-width context track -->
							<line
								x1={G.plotLeft}
								y1={trackY}
								x2={G.plotRight}
								y2={trackY}
								class="track"
								stroke-linecap="round"
							/>
							<!-- the margin the strongest-incumbent rule gives away -->
							{#if row.hasForfeit}
								<line
									x1={r2(row.forfeitFrom)}
									y1={trackY}
									x2={r2(row.trackFrom)}
									y2={trackY}
									class="forfeit"
								/>
							{/if}
							<!-- the advance the reading bought -->
							<line
								x1={r2(row.trackFrom)}
								y1={trackY}
								x2={r2(row.trackTo)}
								y2={trackY}
								class="advance"
								stroke-linecap="round"
							/>

							{#if rx !== null}
								<!-- the two segment values, printed so neither can be read alone -->
								<text class="segment" x={r2((row.trackFrom + rx) / 2)} y={trackY - SEGMENT_DY}
									>{signed3(row.panel.researchModel?.deltaResearchMinusIncumbent ?? 0)}</text
								>
								<text class="segment" x={r2((rx + row.trackTo) / 2)} y={trackY - SEGMENT_DY}
									>{signed3(row.panel.researchModel?.deltaGateMinusResearch ?? 0)}</text
								>
							{/if}

							{#each row.marks as mark (mark.id)}
								{#if mark.series === 'declined'}
									<line
										x1={r2(mark.x)}
										y1={trackY - G.declinedHalfHeight}
										x2={r2(mark.x)}
										y2={trackY + G.declinedHalfHeight}
										class="mark-declined"
									>
										<title>{mark.title}</title>
									</line>
								{:else if mark.series === 'incumbent'}
									<rect
										x={r2(mark.x - MARK_HALF)}
										y={trackY - MARK_HALF}
										width={MARK_HALF * 2}
										height={MARK_HALF * 2}
										class="mark-incumbent"
									>
										<title>{mark.title}</title>
									</rect>
								{:else if mark.series === 'gate'}
									<circle cx={r2(mark.x)} cy={trackY} r={MARK_HALF} class="mark-gate">
										<title>{mark.title}</title>
									</circle>
								{:else}
									<path
										d="M {r2(mark.x)} {trackY - MARK_HALF} L {r2(mark.x + MARK_HALF)} {trackY} L {r2(
											mark.x
										)} {trackY + MARK_HALF} L {r2(mark.x - MARK_HALF)} {trackY} Z"
										class="mark-research"
									>
										<title>{mark.title}</title>
									</path>
								{/if}
							{/each}

							<!-- the panel's own composition, so "not identical" is visible -->
							<text class="row-hetero" x={G.plotLeft} y={row.y + G.heteroDy}>{row.heteroLabel}</text>

							<text class="readout" x={G.readoutX} y={row.y + G.deltaDy}>{row.deltaReadout}</text>
							<text class="readout ci" x={G.readoutX} y={row.y + G.ciDy}>{row.ciReadout}</text>
							<text class="readout rule" x={G.readoutX} y={row.y + G.ruleCostDy}
								>{row.ruleCostReadout}</text
							>

							<line
								x1="12"
								y1={row.y + G.separatorDy}
								x2={G.width - 12}
								y2={row.y + G.separatorDy}
								class="separator"
							/>
						</g>
					{/each}

					<!-- axis -->
					<line x1={G.plotLeft} y1={axisY} x2={G.plotRight} y2={axisY} class="axis" />
					{#each figure.ticks as tick (tick)}
						{@const tx =
							G.plotLeft +
							((tick - figure.domainMin) / (figure.domainMax - figure.domainMin)) *
								(G.plotRight - G.plotLeft)}
						<line x1={r2(tx)} y1={axisY} x2={r2(tx)} y2={axisY + 4} class="axis" />
						<text class="tick" x={r2(tx)} y={axisY + 14}>{fmt3(tick)}</text>
					{/each}
					<text class="tick chance" x={G.plotLeft} y={axisY + 25}>chance</text>
					<text class="axis-title metric-label" x={G.axisTitleX} y={axisY + 14}
						>{figure.axisTitle}</text
					>
					<text class="axis-title" x={G.readoutX} y={axisY + 14}>{figure.readoutTitle[0]}</text>
					<text class="axis-title" x={G.readoutX} y={axisY + 25}>{figure.readoutTitle[1]}</text>
				</svg>
			</div>

			<figcaption>
				<details class="method">
					<summary>how this is computed</summary>
					<p>{payload.prose.question.plain}</p>
					<p>{payload.prose.claim.plain}</p>
					<p>{payload.prose.claimIsNot.plain}</p>
					<p>
						There are two families of INDRA belief here, and both are INDRA's own.
						{#each payload.families as family (family.key)}
							<strong>{family.display}</strong> is {family.whatItComputesProse.plain} It is
							{family.fitted ? 'fitted to data' : 'not fitted to data'}, it is
							{family.whereItRunsProse.plain}, and it ships in <code>{family.shipsIn}</code>.
						{/each}
					</p>
					<p>
						<strong>{payload.servedBeliefIdentity.questionProse.plain}</strong>
						{payload.servedBeliefIdentity.findingProse.plain}
						{payload.servedBeliefIdentity.floorDerivationProse.plain} Floor
						{fmt3(payload.servedBeliefIdentity.simpleScorerFloor)}, read from
						<code>{payload.servedBeliefIdentity.floorSource}</code> (sha256
						{payload.servedBeliefIdentity.floorSourceSha256.slice(0, 12)}).
						{#each payload.servedBeliefIdentity.perPanel as row (row.panelKey)}
							{row.panelDisplay}: {row.nBelowFloor} of {row.nServed} ({pct0(
								row.fractionBelowFloor
							)}) below the floor.
						{/each}
					</p>
					<p>
						The reading model is {payload.arms.gate.whatItIsProse.plain}; it has been shown worked
						examples rather than none ({payload.arms.gate.notZeroShotProse.plain}). The random forest
						fitted in 2023 is <strong>{payload.arms.research.display}</strong> —
						{payload.arms.research.whatItIsProse.plain}
					</p>
					<p>
						{payload.prose.incumbentSelectionRule.plain}
						{payload.prose.incumbentSelectionRuleCost.plain}
					</p>
					<p>
						Measure: {payload.metric} via {payload.prose.metricSource.plain}; the class being detected
						is
						{payload.positiveClass}. Each interval comes from resampling the statements, with both
						models scored on the same resample every time, so it is an interval on the difference
						between them rather than on either one alone. Every form of INDRA's belief gets its own
						interval, not only the one drawn:
						{payload.replication.nIncumbentVariantsCiExcludesZero} of
						{payload.replication.nIncumbentVariantsTotal} stay clear of zero.
					</p>
					{#each payload.prose.caveats as caveat (caveat.shipped)}
						<p>{caveat.plain}</p>
					{/each}
					<ul class="grain">
						{#each payload.panels as panel (panel.key)}
							<li>
								<strong>{panel.display}</strong> &#183; n={panel.nStatements} ({panel.nCorrect} correct /
								{panel.nErrors} wrong, share correct {fmt3(panel.baseRateCorrect)}{panel.balancedByConstruction
									? ', balanced 50/50 by design'
									: ''}) &#183; {panel.curatorNoteProse.plain} &#183; reading model
								<code>{panel.provenance.readerModel}</code>
								&#183; evidence read per statement: mean
								{fmt1(panel.heterogeneity.evidenceReadsPerStatement.mean)}, median
								{fmt1(panel.heterogeneity.evidenceReadsPerStatement.median)}, max
								{panel.heterogeneity.evidenceReadsPerStatement.max}, total
								{panel.heterogeneity.evidenceReadsPerStatement.total};
								{panel.heterogeneity.evidenceReadsPerStatement.nSingle} of
								{panel.heterogeneity.evidenceReadsPerStatement.nStatements} statements ({pct0(
									panel.heterogeneity.evidenceReadsPerStatement.shareSingle
								)}) rest on exactly one piece of evidence, so on that share reading combines nothing
								and must keep or drop on a single sentence.
								{#if panel.heterogeneity.corpusEvidencePerStatement !== null}Evidence in the corpus
									per statement: mean {fmt1(panel.heterogeneity.corpusEvidencePerStatement.mean)}, max
									{panel.heterogeneity.corpusEvidencePerStatement.max}; the reading model saw
									{pct0(panel.heterogeneity.readerEvidenceShareOfCorpus ?? 0)} of it.{:else if panel
										.heterogeneity.corpusEvidenceAbsentBecauseProse}{panel.heterogeneity
										.corpusEvidenceAbsentBecauseProse.plain}{/if}
								Join: {panel.heterogeneity.joinModeProse.plain}; {panel.heterogeneity
										.nUndefinedExcluded} statements
								excluded because no belief was defined for them. &#183; reading {fmt3(
									panel.gate.auroc
								)} vs {panel.incumbent.display}
								{fmt3(panel.incumbent.auroc)} &#183; {signed3(panel.deltaAuroc)} [{signed3(
									panel.bootstrap.ci95Low
								)}, {signed3(panel.bootstrap.ci95High)}], {panel.bootstrap.nValidResamples} of
								{panel.bootstrap.nBootstrap} resamples, seed {panel.bootstrap.seed}. Every form of
								INDRA's belief that could be sourced here, scored:
								{#each panel.incumbentVariants as variant, index (variant.key)}{index > 0
										? '; '
										: ''}{variant.display} ({variant.family}, {variant.fitted
										? 'fitted to data'
										: 'not fitted to data'}, scored over {variant.evidenceScopeNoteProse.plain})
									{fmt3(variant.auroc)}, reading {signed3(variant.deltaAuroc)} [{signed3(
										variant.bootstrap.ci95Low
									)}, {signed3(variant.bootstrap.ci95High)}] — {variant.whatItComputesProse.plain}
									{#if variant.provenanceCaveatProse}Where it came from: {variant
											.provenanceCaveatProse.plain}.{/if}
									{#if variant.crossCheck}Cross-check: <code>{variant.crossCheck.sibling}</code>
										scores the same model on the same {variant.crossCheck.siblingN} statements at
										{fmt3(variant.crossCheck.siblingAuroc)} ({variant.crossCheck.siblingKey}), which is
										{signed3(-variant.crossCheck.deltaVsSibling)} against the value drawn here;
										{variant.crossCheck.nStatementsScoredDifferently} statements score differently.
										{variant.crossCheck.whyItDiffersProse.plain}{/if}
									Source <code>{variant.source}</code> (sha256 {variant.sourceSha256.slice(0, 12)}){/each}.
								Choosing the strongest of those gives away {fmt3(panel.selectionCostAuroc)} AUROC here
								by declining {panel.weakestVariantDisplay} at {fmt3(panel.weakestVariantAuroc)}.
								Average precision (NOT comparable between these sets, because they differ in how many
								statements are wrong): reading {fmt3(panel.gate.averagePrecision)}, the model it is
								measured against {fmt3(panel.incumbent.averagePrecision)}.
								{#if panel.evidenceMatchedControl}{panel.evidenceMatchedControl.display} reaches
									{fmt3(panel.evidenceMatchedControl.auroc)} ({signed3(
										-panel.evidenceMatchedControl.deltaIncumbentMinusControl
									)} against the model it is measured against){panel.evidenceMatchedControl
										.atOrBelowChance
										? ', at or below chance'
										: ''}. {panel.evidenceMatchedControl.noteProse
											.plain}{:else if panel.evidenceMatchedControlAbsentBecauseProse}{panel
											.evidenceMatchedControlAbsentBecauseProse.plain}{/if}
								{#if panel.gateSensitivity}The reading model we actually ship, which also replaces
									INDRA's default per-source reliabilities with recalibrated ones, reaches
									{fmt3(panel.gateSensitivity.auroc)} here ({signed3(
										panel.gateSensitivity.deltaAuroc
									)}); it is recorded, not drawn.{/if}
								{#if panel.inSampleNoteProse}{panel.inSampleNoteProse.plain}{/if} Gold
								<code>{panel.provenance.gold}</code> (sha256 {panel.provenance.goldSha256.slice(0, 12)}),
								run <code>{panel.provenance.run}</code>
								(sha256 {panel.provenance.runSha256.slice(0, 12)}).
							</li>
						{/each}
					</ul>
					<p>Generated by <code>{payload.generatedBy}</code>.</p>
				</details>

				<footer>
					<code>{data.artifact_path}</code> &#183; sha256 {data.artifact_sha256.slice(0, 12)}
				</footer>
			</figcaption>
		</figure>
	{/if}
</section>

<style>
	.deployed-baseline {
		margin: 0;
	}
	.scroller {
		overflow-x: auto;
	}
	.scroller svg {
		display: block;
		width: 100%;
		min-width: 760px;
		height: auto;
		overflow: visible;
	}
	figure {
		margin: 0;
	}

	.fig-title {
		fill: var(--ink);
		font-family: var(--serif);
		font-size: 15px;
	}
	.fig-subtitle {
		fill: var(--ink-faint);
		font-family: var(--mono);
		font-size: 8.5px;
		letter-spacing: 0.04em;
	}
	/* The regime line carries half the claim, so it reads at the same weight as
	   the counts above it rather than as a third tier of fine print. */
	.fig-subtitle.regime {
		fill: var(--ink-muted);
	}
	.legend-text {
		fill: var(--ink-muted);
		font-family: var(--mono);
		font-size: 8.5px;
	}

	.row-label {
		fill: var(--ink);
		font-family: var(--mono);
		font-size: 9px;
		text-anchor: end;
	}
	.row-sub {
		fill: var(--ink-faint);
		font-family: var(--mono);
		font-size: 8px;
		text-anchor: end;
	}
	/* The comparator is the row's identity, not a footnote: same weight as the
	   sub-label but in the incumbent's own ink, so it reads with the square. */
	.row-comparator {
		fill: var(--ink-muted);
		font-family: var(--mono);
		font-size: 8px;
		text-anchor: end;
	}
	.row-chip {
		fill: var(--ink-muted);
		font-family: var(--mono);
		font-size: 7.5px;
		text-anchor: end;
		letter-spacing: 0.08em;
		text-transform: uppercase;
	}
	.row-hetero {
		fill: var(--ink-faint);
		font-family: var(--mono);
		font-size: 7.5px;
		text-anchor: start;
	}
	.row.largest .row-label {
		font-weight: 600;
	}

	.track {
		stroke: var(--rule);
		stroke-width: 1;
	}
	/* The advance is the figure's only filled length: what reading bought. */
	.advance {
		stroke: var(--accent);
		stroke-width: 5;
		opacity: 0.28;
	}
	/* The margin the strongest-incumbent rule gives away. Dashed, thin, faint:
	   present and readable, never competing with the claim. */
	.forfeit {
		stroke: var(--ink-faint);
		stroke-width: 1.4;
		stroke-dasharray: 3 2;
	}

	/* Shapes carry the series, so the figure survives greyscale and CVD. */
	.mark-incumbent {
		fill: var(--paper);
		stroke: var(--ink-muted);
		stroke-width: 1.4;
	}
	.mark-gate {
		fill: var(--accent);
		stroke: var(--accent);
		stroke-width: 1;
	}
	.mark-research {
		fill: var(--paper);
		stroke: var(--blocked);
		stroke-width: 1.4;
	}
	.mark-declined {
		stroke: var(--ink-faint);
		stroke-width: 1.2;
	}

	.segment {
		fill: var(--ink-muted);
		font-family: var(--mono);
		font-size: 8px;
		text-anchor: middle;
		font-variant-numeric: tabular-nums;
	}
	.readout {
		fill: var(--ink);
		font-family: var(--mono);
		font-size: 9px;
		font-variant-numeric: tabular-nums;
	}
	.readout.ci {
		fill: var(--ink-faint);
		font-size: 7.5px;
	}
	.readout.rule {
		fill: var(--ink-faint);
		font-size: 7.5px;
	}
	.separator {
		stroke: var(--rule);
		stroke-width: 0.5;
		opacity: 0.6;
	}

	.axis {
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
	.tick.chance {
		font-size: 7.5px;
	}
	.axis-title {
		fill: var(--ink-muted);
		font-family: var(--mono);
		font-size: 8.5px;
	}
	/* The one right-anchored axis title; explicit, never a positional selector. */
	.axis-title.metric-label {
		text-anchor: end;
	}

	figcaption {
		margin-top: 0.7rem;
		max-width: 74ch;
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
	.method {
		margin-top: 0.45rem;
		max-width: 74ch;
	}
	.method p {
		margin: 0 0 0.55rem;
		font-family: var(--serif);
		font-size: 0.8rem;
		line-height: 1.5;
		color: var(--ink-muted);
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

	.sr {
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
</style>
