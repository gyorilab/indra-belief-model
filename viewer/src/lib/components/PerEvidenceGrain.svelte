<!--
  PerEvidenceGrain — the level the reader actually works at.

  Every other figure on /paper scores an assembled STATEMENT. The reader does not
  produce statements. It produces one correct/incorrect verdict per (statement,
  evidence) pair; statement belief is that verdict set pushed through INDRA's
  noisy-OR. This figure puts both levels of the SAME model on one page so the
  reading can be separated from the aggregation.

  PLAIN-LANGUAGE RULE (2026-07-27). Written for a working biologist who has never
  read the 2023 paper and does not work in machine-learning evaluation. In
  RENDERED PROSE — never in identifiers or these comments — "arm" is a model,
  "plate" is a figure, "lane" is a row, "grain" is the level a score is taken at
  (one piece of evidence, or one assembled statement), "census" is a count, and
  "prior" is what INDRA assumes before reading anything. AUROC is defined in the
  caption that first uses it.

  ── TWO REGISTERS, TWO AXES, ONE FIGURE ─────────────────────────────────────
  Register A is a DISCRIMINATION axis (AUROC). Each model gets one row carrying
  two marks: a filled mark at its per-evidence AUROC over 5,379 reviewed pairs
  with its shipped bootstrap interval, and an open mark at the same model's
  statement AUROC over 1,689 statements. A hairline connects them. The connector
  is NOT a paired increment and the component never calls it one — the two marks
  are measurements on two different item populations at two different base rates,
  and `figure.twoGrainNote` (shipped) carries that wording.

  Register B is a PROBABILITY axis (P(correct) at one evidence) and it is drawn
  separately for exactly that reason: putting a probability on a discrimination
  axis would be a category error. It shows INDRA's bundled prior as a single rule
  and each source's observed correct fraction as a dot, so the shared-prior defect
  is visible as five dots strung out along one vertical line.

  NOTHING HERE IS RECOMPUTED. Every AUROC, interval bound, census count,
  chi-square and reconciliation residual is READ off the shipped artifact. The
  builder gates the figure to `unavailable` if the recovered per-evidence verdicts
  do not rebuild the shipped statement probabilities exactly, because without that
  the connector joins two things that are not the same run.

  PROSE BUDGET. /paper counts visible words. Method detail lives in <desc>,
  <title> and <details>, which are not counted; the visible chrome is a heading,
  one lead line, the figure's own labels, and a terse legend.

  ── SVG LABEL BUDGET (measured; right-anchored text clips leading glyphs
  silently, and a <desc> does not save it) ───────────────────────────────────
  At font-size 9px the mono advance is 5.4186 user units per character; at 8px it
  is 4.8165. viewBox 920 wide, plot x ∈ [196, 782].

    lane names   right-anchored at x=184 → gutter 184 u = 33 chars at 9px.
                 Longest shipped is "INDRA source prior (bundled)" at 28 chars
                 = 151.7 u, leaving 32.3 u. ENFORCED in buildPaperPerEvidence().
    readouts     left-anchored at x=790 → gutter (920−790) = 130 u = 26 chars at
                 8px. Longest shipped is "0.850 ev · 0.898 stmt" at 21 chars
                 = 101.1 u. ENFORCED in buildPaperPerEvidence().
    band heads   start-anchored at x=0 → budget 920 u = 169 chars at 8px; the
                 longest built here is 62 chars = 298.6 u.
    axis ticks   "0.95" 4 ch = 19.3 u at 8px, middle-anchored on x ∈ [196, 782]
                 → [186.4, 791.7] ⊂ [0, 920].
    chance rule  label fit is MEASURED by chanceLabelFits() in the data module and
                 the label flips side rather than clip.
    source rows  names right-anchored at x=184, same 33-char budget; the longest
                 INDRA source name on this panel is "sparser" at 7 chars.
    axis titles  middle-anchored on the 586 u plot; the longer is 44 chars
                 = 212 u at 10px-equivalent measurement, well inside.
  Arm display names are never abbreviated to fit — the budget is sized to them.
-->
<script lang="ts">
	import {
		PAPER_PER_EVIDENCE_CHANCE,
		PAPER_PER_EVIDENCE_GEOMETRY,
		PAPER_PER_EVIDENCE_SERIES,
		PAPER_PER_EVIDENCE_SOURCE_DOMAIN,
		PAPER_PER_EVIDENCE_SOURCE_TICKS,
		fmt3,
		type PaperPerEvidenceLane,
		type PaperPerEvidenceLoad
	} from '$lib/data/paper-per-evidence';

	let { data }: { data: PaperPerEvidenceLoad } = $props();

	const G = PAPER_PER_EVIDENCE_GEOMETRY;
	const figure = $derived(data.status === 'ok' ? data.figure : null);

	/** Discrimination axis. */
	function x(value: number): number {
		if (figure === null) return G.plotLeft;
		const span = figure.domainMax - figure.domainMin;
		return G.plotLeft + ((value - figure.domainMin) / span) * (G.plotRight - G.plotLeft);
	}

	/** Probability axis for the per-source register. A SECOND scale on purpose. */
	function px(value: number): number {
		const span = PAPER_PER_EVIDENCE_SOURCE_DOMAIN.max - PAPER_PER_EVIDENCE_SOURCE_DOMAIN.min;
		return (
			G.plotLeft +
			((value - PAPER_PER_EVIDENCE_SOURCE_DOMAIN.min) / span) * (G.plotRight - G.plotLeft)
		);
	}

	function diamond(cx: number, cy: number, r = 4.2): string {
		return `M ${cx} ${cy - r} L ${cx + r} ${cy} L ${cx} ${cy + r} L ${cx - r} ${cy} Z`;
	}

	/** An open bracket: the baselines' mark, distinct in shape as well as hue. */
	function bracket(cx: number, cy: number, r = 4.2): string {
		return `M ${cx - r} ${cy - r} L ${cx} ${cy - r} L ${cx} ${cy + r} L ${cx - r} ${cy + r}`;
	}

	function pct(value: number): string {
		return `${(value * 100).toFixed(1)}%`;
	}

	function signed(value: number): string {
		return `${value >= 0 ? '+' : '-'}${Math.abs(value).toFixed(3)}`;
	}

	function shortSha(value: string): string {
		return `${value.slice(0, 10)}…`;
	}

	function seriesFor(lane: PaperPerEvidenceLane) {
		return lane.kind === 'reader'
			? PAPER_PER_EVIDENCE_SERIES['evidence-reader']
			: PAPER_PER_EVIDENCE_SERIES['evidence-baseline'];
	}

	const statementSeries = PAPER_PER_EVIDENCE_SERIES.statement;
	const chanceX = $derived(x(PAPER_PER_EVIDENCE_CHANCE));
	const readers = $derived(figure === null ? [] : figure.lanes.filter((l) => l.kind === 'reader'));
	const baselines = $derived(
		figure === null ? [] : figure.lanes.filter((l) => l.kind === 'baseline')
	);
	/** 3dp with an explicit sign, for the paired-delta readout. */
	function signed3(v: number): string {
		return `${v >= 0 ? '+' : '\u2212'}${Math.abs(v).toFixed(3)}`;
	}

	/** The strongest reader and the reference baseline, for the two readouts. */
	const bestReader = $derived(readers.length > 0 ? readers[0] : null);
	const reference = $derived(
		figure === null ? null : (figure.lanes.find((l) => l.id === figure.referenceId) ?? null)
	);
</script>

<section class="grain" aria-labelledby="per-evidence-title">
	{#if data.status === 'unavailable' || figure === null}
		<div class="gate" role="status">
			<p class="eyebrow">one piece of evidence at a time</p>
			<h2 id="per-evidence-title">The per-evidence comparison is unavailable</h2>
			<p>{data.reason}</p>
			<code>{data.artifact_path}</code>
		</div>
	{:else}
		<header>
			<div>
				<p class="eyebrow">one verdict per piece of evidence</p>
				<h2 id="per-evidence-title">One piece of evidence at a time</h2>
			</div>
			<strong>{figure.nReviewedPairs.toLocaleString()} pieces of evidence</strong>
		</header>

		<figure>
			<figcaption>
				AUROC — the chance a model scores a randomly picked correct item above a randomly picked
				incorrect one; 0.5 is a coin flip — over {figure.nReviewedPairs.toLocaleString()} human-reviewed
				pieces of evidence, and over the {figure.nStatements.toLocaleString()} assembled statements those
				pieces of evidence support
			</figcaption>
			<svg
				viewBox="0 0 {G.width} {figure.height}"
				style:min-height="{figure.height}px"
				role="img"
				aria-labelledby="per-evidence-chart-title per-evidence-chart-desc"
			>
				<title id="per-evidence-chart-title"
					>How well each model tells correct evidence from incorrect evidence, measured once per
					piece of evidence and once per assembled statement, with the single reliability INDRA
					assumes for several sources shown against what those sources actually deliver</title
				>
				<desc id="per-evidence-chart-desc"
					>Two charts stacked on one figure. The upper one measures how well each model separates
					correct from incorrect: one row per model, the reading models first, then INDRA's
					source-reliability scores, each group ordered by its per-evidence AUROC. In every row the
					upper mark is the result over {figure.nReviewedPairs} human-reviewed pieces of evidence — a
					filled diamond for a reading model, an open bracket for a source-reliability score — with a
					bar for its 95% range from {figure.nBootstrap}
					{figure.bootstrapDesign}. The lower open circle in the same row is the SAME model scored one
					assembled statement at a time, over {figure.nStatements} statements, joined to the upper mark
					by a hairline. That hairline is not a before-and-after: {figure.twoGrainNote} The heavy
					vertical rule is chance, AUROC 0.5 — a score that is the same constant for every piece of
					evidence from a given source cannot beat that rule, by construction. The lower chart is a
					separate probability axis carrying the chance that one piece of evidence is correct, because
					a probability may not be drawn on the scale above. It has one row per source: a filled dot at
					the fraction of that source's reviewed evidence the curators marked correct, and a tick at
					the belief INDRA assigns that source from one piece of evidence before reading anything.
					Where those ticks line up on a single vertical while the dots spread out is the defect: one
					assumed reliability standing in for sources that behave differently.</desc
				>

				<!-- ── register A: discrimination ─────────────────────────────────────
				     Its grid STOPS at its own axis rule. The register below carries a
				     probability, not a discrimination, so an AUROC grid line running
				     through it would invite reading a dot against the wrong scale. -->
				{#each figure.ticks as tick (tick)}
					<line class="grid" x1={x(tick)} y1={G.topPad} x2={x(tick)} y2={figure.discriminationAxisY} />
					<text class="tick" x={x(tick)} y={figure.discriminationAxisY + 14}>{tick.toFixed(2)}</text>
				{/each}

				<rect
					class="chance-wash"
					x={chanceX - 3}
					y={G.topPad - 10}
					width="6"
					height={figure.discriminationAxisY - (G.topPad - 10)}
				/>
				<line
					class="chance"
					x1={chanceX}
					y1={G.topPad - 10}
					x2={chanceX}
					y2={figure.discriminationAxisY}
				/>
				<text
					class="chance-label"
					x={figure.chanceLabelFits ? chanceX + 6 : chanceX - 6}
					y={G.topPad - 14}
					text-anchor={figure.chanceLabelFits ? 'start' : 'end'}>{figure.chanceLabel}</text
				>

				<text class="band-head" x="0" y={figure.readerBandY + 12}
					>PER PIECE OF EVIDENCE · reading models ({readers.length})</text
				>
				<text class="band-head" x="0" y={figure.baselineBandY + 12}
					>PER PIECE OF EVIDENCE · INDRA source-reliability scores ({baselines.length})</text
				>

				{#each figure.lanes as lane (lane.id)}
					{@const series = seriesFor(lane)}

					<!-- 9px mono = 5.4186 u/char; 184-unit gutter = 33 chars, enforced in
					     buildPaperPerEvidence(). Longest shipped display is 28. -->
					<text class="lane-name" x={G.labelAnchorX} y={lane.y + 3}>{lane.display}</text>

					<!-- The connector between the two grains. Drawn first and thin, so it
					     reads as a relation between the marks, never as an interval. -->
					<line
						class="connector"
						x1={x(lane.evidence.value)}
						y1={lane.evidenceY}
						x2={x(lane.statementAuroc)}
						y2={lane.statementY}
					/>

					<line
						class="bar"
						x1={x(lane.evidence.low)}
						x2={x(lane.evidence.high)}
						y1={lane.evidenceY}
						y2={lane.evidenceY}
						stroke={series.strokeVar}
						stroke-width={series.strokeWidth}
						stroke-dasharray={series.dash}
					/>
					{#each [lane.evidence.low, lane.evidence.high] as end, cap (cap)}
						<line
							class="bar"
							x1={x(end)}
							x2={x(end)}
							y1={lane.evidenceY - G.intervalCap}
							y2={lane.evidenceY + G.intervalCap}
							stroke={series.strokeVar}
							stroke-width={series.strokeWidth}
						/>
					{/each}

					{#if lane.kind === 'reader'}
						<path
							class="mark"
							d={diamond(x(lane.evidence.value), lane.evidenceY)}
							fill={series.strokeVar}
						>
							<title>{lane.title}</title>
						</path>
					{:else}
						<path
							class="mark"
							d={bracket(x(lane.evidence.value), lane.evidenceY)}
							fill="none"
							stroke={series.strokeVar}
							stroke-width="1.8"
						>
							<title>{lane.title}</title>
						</path>
					{/if}

					<circle
						class="mark"
						cx={x(lane.statementAuroc)}
						cy={lane.statementY}
						r="3.2"
						fill="var(--paper)"
						stroke={statementSeries.strokeVar}
						stroke-width="1.3"
					>
						<title
							>{lane.display}, scored one statement at a time: AUROC {fmt3(lane.statementAuroc)} over
							{figure.nStatements} assembled statements. {figure.twoGrainNote}</title
						>
					</circle>

					<!-- 8px mono = 4.8165 u/char; 130-unit gutter = 26 chars, enforced in
					     buildPaperPerEvidence(). Longest shipped readout is 21. -->
					<text class="readout" x={G.readoutX} y={lane.y + 3}>{lane.readout}</text>
					<!-- The paired delta vs the reference baseline. This is the figure's
					     inferential claim: the point estimates above are separable, and
					     by how much. Shipped by the artifact's own bootstrap and read,
					     never recomputed. Budget: 26 chars at readoutX; longest form
					     "+0.137 [+0.122,+0.152]" is 22. -->
					{#if lane.pairedDelta}
						<!-- The emphasis marks "this margin is separable from zero", which is a
						     question about width and not about direction — the signed number is
						     printed right there. Asked as `!== 'not-significant'` so it cannot
						     be read as, or copied as, a claim about which side. -->
						<text
							class="paired"
							class:excludes={lane.pairedDelta.standing !== 'not-significant'}
							x={G.readoutX}
							y={lane.y + 13}
							>{signed3(lane.pairedDelta.delta)} [{signed3(lane.pairedDelta.ciLow)},{signed3(
								lane.pairedDelta.ciHigh
							)}]</text
						>
					{/if}
				{/each}

				<line
					class="axis"
					x1={G.plotLeft}
					y1={figure.discriminationAxisY}
					x2={G.plotRight}
					y2={figure.discriminationAxisY}
				/>

				<!-- ── register B: probability — its OWN axis, its own ticks ──────── -->
				<!-- 8px mono = 4.8165 u/char, start-anchored at x=0 in a 920-unit viewBox:
				     the budget is 169 characters and this line is 108. -->
				<text class="band-head" x="0" y={figure.sourceBandY + 12}
					>BY SOURCE · chance one piece of evidence is correct · dot = what curators found, tick =
					what INDRA assumes</text
				>
				{#each PAPER_PER_EVIDENCE_SOURCE_TICKS as tick (tick)}
					<line
						class="grid"
						x1={px(tick)}
						y1={figure.sourceBandY + 18}
						x2={px(tick)}
						y2={figure.probabilityAxisY}
					/>
					<text class="tick faint" x={px(tick)} y={figure.probabilityAxisY + 14}
						>{tick.toFixed(2)}</text
					>
				{/each}
				{#if figure.sharedPrior}
					<line
						class="prior-rule"
						x1={px(figure.sharedPrior.sharedPrior)}
						y1={figure.sourceBandY + 18}
						x2={px(figure.sharedPrior.sharedPrior)}
						y2={figure.probabilityAxisY}
					>
						<title
							>Out of the box, INDRA gives {figure.sharedPrior.sources.join(', ')} the identical belief
							{fmt3(figure.sharedPrior.sharedPrior)} from one piece of evidence, while the share of
							their evidence curators marked correct here runs from {fmt3(
								figure.sharedPrior.observedMin
							)} to {fmt3(figure.sharedPrior.observedMax)} (chi-square {figure.sharedPrior.chi2.toFixed(
								1
							)} on {figure.sharedPrior.dof} degrees of freedom)</title
						>
					</line>
				{/if}
				{#each figure.sourceRows as row (row.source)}
					<text class="lane-name small" x={G.labelAnchorX} y={row.y + 3}>{row.source}</text>
					<line
						class="leader"
						x1={px(row.bundledPriorAtOneEvidence)}
						y1={row.y}
						x2={px(row.observedCorrectFraction)}
						y2={row.y}
					/>
					<line
						class="prior-tick"
						x1={px(row.bundledPriorAtOneEvidence)}
						y1={row.y - 4}
						x2={px(row.bundledPriorAtOneEvidence)}
						y2={row.y + 4}
					/>
					<circle
						class="mark"
						cx={px(row.observedCorrectFraction)}
						cy={row.y}
						r={row.metricRow ? 3.4 : 2}
						fill={row.metricRow ? 'var(--accent)' : 'var(--ink-faint)'}
					>
						<title
							>{row.source}: {row.positivePairs} of {row.reviewedPairs} reviewed pieces of evidence are
							correct ({pct(row.observedCorrectFraction)}); out of the box INDRA gives this source
							{fmt3(row.bundledPriorAtOneEvidence)} from one piece of evidence{row.metricRow
								? ''
								: ' — too few reviewed pieces to score, counted here but not measured'}</title
						>
					</circle>
					<text class="source-readout" x={G.readoutX} y={row.y + 3}
						>{pct(row.observedCorrectFraction)} · n={row.reviewedPairs}</text
					>
				{/each}

				<line
					class="axis"
					x1={G.plotLeft}
					y1={figure.probabilityAxisY}
					x2={G.plotRight}
					y2={figure.probabilityAxisY}
				/>
				<!-- 9px mono = 5.4186 u/char, middle-anchored on the 586-unit plot: 62
				     characters = 336 u, so it spans 321..657 inside the 920 viewBox. -->
				<text
					class="axis-label"
					x={(G.plotLeft + G.plotRight) / 2}
					y={figure.probabilityAxisY + 30}
					>top: AUROC · bottom: chance one piece of evidence is correct</text
				>
			</svg>

			<ul class="legend">
				{#each [PAPER_PER_EVIDENCE_SERIES['evidence-reader'], PAPER_PER_EVIDENCE_SERIES['evidence-baseline'], PAPER_PER_EVIDENCE_SERIES.statement] as series (series.id)}
					<li>
						<svg class="swatch" viewBox="0 0 44 12" aria-hidden="true">
							<line
								x1="2"
								y1="6"
								x2="42"
								y2="6"
								stroke={series.strokeVar}
								stroke-width={series.strokeWidth}
								stroke-dasharray={series.dash}
							/>
							{#if series.shape === 'diamond'}
								<path d={diamond(22, 6, 4)} fill={series.strokeVar} />
							{:else if series.shape === 'bracket'}
								<path
									d={bracket(24, 6, 4)}
									fill="none"
									stroke={series.strokeVar}
									stroke-width="1.8"
								/>
							{:else}
								<circle
									cx="22"
									cy="6"
									r="3.2"
									fill="var(--paper)"
									stroke={series.strokeVar}
									stroke-width="1.3"
								/>
							{/if}
						</svg>
						<span>{series.legend}</span>
					</li>
				{/each}
			</ul>
		</figure>

		<details>
			<summary>how the two levels are tied together</summary>
			<p>
				The reader gives one verdict per piece of evidence. A statement's belief is those verdicts
				pushed through INDRA's <code>{figure.reaggregation.aggregation}</code> noisy-OR, which raises
				belief as independent sources pile up. Those are not two models: the per-evidence verdicts
				recovered for this figure were pushed back through that formula and reproduced the shipped
				statement scores for
				<strong
					>{figure.reaggregation.nExact.toLocaleString()} of {figure.reaggregation.nStatements.toLocaleString()}</strong
				>
				statements, the largest difference anywhere being {figure.reaggregation.maxAbsDiff}. The figure
				refuses to draw itself if that check is anything other than exact, because without it the
				hairline in each row joins two things that are not the same run.
			</p>
			<p>
				{figure.twoGrainNote} The evidence side is {figure.nReviewedPairs.toLocaleString()} pieces of
				evidence, {pct(1 - figure.negativeFraction)} of them correct; the statement side is
				{figure.nStatements.toLocaleString()} statements, {pct(figure.statementPositiveRate)} of them
				correct. The evidence side carries {figure.powerRatio.toFixed(1)}× as many items.
				{figure.powerNote}
			</p>
		</details>

		<details>
			<summary>what each model is, and what it is not</summary>
			<ul class="attributions">
				{#each figure.lanes as lane (lane.id)}
					<li>
						<strong>{lane.display}</strong>
						<span>{lane.attribution}</span>
					</li>
				{/each}
			</ul>
			<p>
				{#each figure.coverage.excludedBaselines as excluded (excluded.family)}
					The {excluded.family} published in the 2023 paper cannot appear in this figure at all:
					{excluded.reason}.
				{/each}
			</p>
			<p class="basis">
				How the numbers are computed: AUROC and average precision handle tied scores correctly and
				match scikit-learn's on the full sample. Trapezoidal PR-AUC — precision-recall area with
				straight lines drawn between the points — is {figure.estimatorNote}. Ranges are 95% percentile
				bootstrap, {figure.nBootstrap.toLocaleString()} resamples, seed {figure.seed},
				{figure.bootstrapDesign}. Error-class F1 uses a {figure.decisionThreshold} cutoff, the reader's
				own decision boundary.
			</p>
		</details>

		<details>
			<summary>coverage, curation and contamination</summary>
			<p>
				{figure.coverage.executedUniquePairs.toLocaleString()} pieces of evidence were run;
				{figure.coverage.reviewedPairs.toLocaleString()} carry a human review and
				{figure.coverage.unreviewedPairs.toLocaleString()} do not. Every reviewed one carries a
				reader verdict —
				<strong>{figure.coverage.unscoredPairs}</strong> reviewed pieces are unscored — so no model is
				measured on a different set from any other. Reviews come from
				{#each figure.curators as curator, index (curator.curator)}{index > 0
						? ', '
						: ''}{curator.curator} ({curator.reviewedPairs.toLocaleString()}){/each}.
			</p>
			<div class="table-scroll">
				<table>
					<caption>how the reader reached each reviewed piece of evidence</caption>
					<thead>
						<tr><th scope="col">route</th><th scope="col">pieces of evidence reviewed</th></tr>
					</thead>
					<tbody>
						{#each figure.coverage.tierCensus as tier (tier.tier)}
							<tr><td><code>{tier.tier}</code></td><td>{tier.pairs.toLocaleString()}</td></tr>
						{/each}
					</tbody>
				</table>
			</div>
			<p class="warn">
				At this level the worked examples put in front of the model are NOT separate from what it is
				scored on. {figure.contamination.demonstrationSentences} distinct example sentences were checked
				against
				{figure.contamination.reviewedPairsScanned.toLocaleString()} reviewed pieces of evidence:
				<strong>{figure.contamination.overlappingPairs}</strong> carry one of those sentences word for
				word, and <strong>{figure.contamination.overlappingPairsSameClaim}</strong> of those also make
				the same claim, down to the molecules named and the kind of relation — the model had been shown
				that exact judgement.
				{#if figure.contamination.pairsKept !== null && figure.contamination.maxAurocAbsShift !== null}
					Dropping them leaves {figure.contamination.pairsKept.toLocaleString()} pieces of evidence and
					moves no model's AUROC by more than {figure.contamination.maxAurocAbsShift.toFixed(4)}; the
					main figure keeps every reviewed piece, because throwing items out after seeing them is its
					own bias.
				{/if}
				Whether these sentences were in the language model's original training data is a different
				question and out of scope: the benchmark corpus and the repository released with the 2023 paper
				are both public.
			</p>
		</details>

		<details>
			<summary>every number in this figure</summary>
			<div class="table-scroll">
				<table>
					<caption>results per piece of evidence and per statement, as shipped</caption>
					<thead>
						<tr>
							<th scope="col">model</th>
							<th scope="col">AUROC (per piece of evidence)</th>
							<th scope="col">95% range</th>
							<th scope="col">AUROC (per statement)</th>
							<th scope="col">change</th>
							<th scope="col">average precision, error class</th>
							<th scope="col">error-class F1</th>
							<th scope="col">distinct scores</th>
						</tr>
					</thead>
					<tbody>
						{#each figure.lanes as lane (lane.id)}
							<tr class:reference={lane.id === figure.referenceId}>
								<th scope="row">{lane.display}</th>
								<td>{fmt3(lane.evidence.value)}</td>
								<td>[{fmt3(lane.evidence.low)}, {fmt3(lane.evidence.high)}]</td>
								<td>{fmt3(lane.statementAuroc)}</td>
								<td>{signed(lane.grainShift)}</td>
								<td>{fmt3(lane.averagePrecisionIncorrect)}</td>
								<td>{fmt3(lane.errorF1)}</td>
								<td>{lane.distinctScores.toLocaleString()}</td>
							</tr>
						{/each}
					</tbody>
				</table>
			</div>
			<div class="table-scroll">
				<table>
					<caption>AUROC WITHIN each source, one piece of evidence at a time</caption>
					<thead>
						<tr>
							<th scope="col">model</th>
							{#each figure.lanes[0].perSource as tick (tick.source)}
								<th scope="col">{tick.source}</th>
							{/each}
						</tr>
					</thead>
					<tbody>
						{#each figure.lanes as lane (lane.id)}
							<tr>
								<th scope="row">{lane.display}</th>
								{#each lane.perSource as tick (tick.source)}
									<td>{fmt3(tick.auroc)}</td>
								{/each}
							</tr>
						{/each}
					</tbody>
				</table>
			</div>
			<p class="basis">
				A score that never varies inside a source scores 0.5 there, by construction, and the scores
				refitted on data they never learned from sit below it. The statements are split into 10
				folds — 10 groups, each fitted model scored on the group it did not train on — so the only
				thing that moves those scores inside a source is which fold a statement fell into, and that
				runs mechanically against the labels of the held-out fold.
				{#if reference}Every paired margin in this figure is against {reference.display}, the
					strongest source-reliability score per piece of evidence.{/if}
				{#if bestReader && reference}
					Best reading model {bestReader.display}: {fmt3(bestReader.evidence.value)} vs {fmt3(
						reference.evidence.value
					)}.
				{/if}
			</p>
			{#if figure.sharedPrior}
				<p class="basis">
					One assumed reliability, {fmt3(figure.sharedPrior.sharedPrior)}, covers
					{figure.sharedPrior.sources.join(', ')}; the share of their evidence curators actually marked
					correct runs from {fmt3(figure.sharedPrior.observedMin)} to {fmt3(
						figure.sharedPrior.observedMax
					)} (chi-square {figure.sharedPrior.chi2.toFixed(1)}, {figure.sharedPrior.dof} degrees of
					freedom, p = {figure.sharedPrior.pValue.toExponential(2)}).
				</p>
			{/if}
			<p class="basis">
				{data.artifact_path}{#if data.artifact_sha256}
					· sha256 {shortSha(data.artifact_sha256)}{/if}
			</p>
		</details>
	{/if}
</section>

<style>
	.grain {
		margin: 2rem 0;
	}
	.gate {
		border-left: 3px solid var(--blocked);
		padding: 0.7rem 0.95rem;
	}
	.gate p,
	.gate code {
		font-family: var(--mono);
		font-size: 0.72rem;
		color: var(--ink-muted);
	}
	header {
		display: flex;
		align-items: baseline;
		justify-content: space-between;
		gap: 1rem;
		border-bottom: 1px solid var(--rule);
		padding-bottom: 0.5rem;
	}
	header strong {
		font-family: var(--mono);
		font-size: 0.7rem;
		font-weight: 500;
		color: var(--accent);
		white-space: nowrap;
	}
	.eyebrow {
		margin: 0;
		font-family: var(--mono);
		font-size: 0.64rem;
		letter-spacing: 0.08em;
		text-transform: uppercase;
		color: var(--ink-faint);
	}
	h2 {
		margin: 0.15rem 0 0;
		font-family: var(--serif);
		font-size: 1.15rem;
		font-weight: 500;
		color: var(--ink);
	}
	figure {
		margin: 1rem 0 0;
	}
	figcaption {
		font-family: var(--mono);
		font-size: 0.66rem;
		color: var(--ink-muted);
		margin-bottom: 0.35rem;
	}
	svg {
		display: block;
		width: 100%;
		height: auto;
		font-family: var(--mono);
		overflow: visible;
	}
	.grid {
		stroke: var(--rule);
		stroke-width: 0.6;
	}
	.axis {
		stroke: var(--ink);
		stroke-width: 0.9;
	}
	.chance {
		stroke: var(--ink);
		stroke-width: 1.1;
	}
	.chance-wash {
		fill: var(--accent-wash);
	}
	.chance-label {
		font-size: 8px;
		fill: var(--ink);
	}
	.connector {
		stroke: var(--ink-faint);
		stroke-width: 0.7;
	}
	.bar {
		stroke-linecap: butt;
	}
	.mark {
		vector-effect: non-scaling-stroke;
	}
	.prior-rule {
		stroke: var(--blocked);
		stroke-width: 1.4;
		stroke-dasharray: 4 2;
	}
	.prior-tick {
		stroke: var(--blocked);
		stroke-width: 1.4;
	}
	.leader {
		stroke: var(--rule);
		stroke-width: 1.6;
	}
	.band-head {
		font-size: 8px;
		fill: var(--ink-faint);
		letter-spacing: 0.05em;
	}
	.lane-name {
		font-size: 9px;
		fill: var(--ink);
		text-anchor: end;
	}
	.lane-name.small {
		font-size: 8px;
		fill: var(--ink-muted);
	}
	.tick {
		font-size: 8px;
		fill: var(--ink-muted);
		text-anchor: middle;
	}
	.tick.faint {
		fill: var(--ink-faint);
	}
	.paired {
		font-family: var(--mono);
		font-size: 7.5px;
		fill: var(--ink-muted);
	}
	.paired.excludes {
		fill: var(--ink);
		font-weight: 600;
	}
	.readout {
		font-size: 8px;
		fill: var(--ink-muted);
		font-variant-numeric: tabular-nums;
	}
	.source-readout {
		font-size: 8px;
		fill: var(--ink-faint);
		font-variant-numeric: tabular-nums;
	}
	.axis-label {
		font-size: 9px;
		fill: var(--ink-muted);
		text-anchor: middle;
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
		font-size: 0.62rem;
		line-height: 1.4;
		color: var(--ink-muted);
	}
	.swatch {
		flex: 0 0 auto;
		width: 44px;
		height: 12px;
		overflow: visible;
	}
	details {
		margin-top: 0.9rem;
		border-top: 1px solid var(--rule);
		padding-top: 0.6rem;
	}
	summary {
		cursor: pointer;
		font-family: var(--mono);
		font-size: 0.68rem;
		color: var(--ink-muted);
	}
	details p {
		font-family: var(--serif);
		font-size: 0.82rem;
		line-height: 1.55;
		color: var(--ink-muted);
		max-width: 78ch;
	}
	details p.basis {
		font-family: var(--mono);
		font-size: 0.66rem;
		color: var(--ink-faint);
	}
	details p.warn {
		border-left: 3px solid var(--blocked);
		padding-left: 0.7rem;
		background: color-mix(in srgb, var(--blocked) 3%, transparent);
	}
	.attributions {
		margin: 0.4rem 0 0;
		padding-left: 1.1rem;
	}
	.attributions li {
		font-family: var(--serif);
		font-size: 0.8rem;
		line-height: 1.5;
		color: var(--ink-muted);
		max-width: 78ch;
	}
	.attributions li + li {
		margin-top: 0.35rem;
	}
	.attributions strong {
		font-family: var(--mono);
		font-size: 0.7rem;
		color: var(--ink);
	}
	.table-scroll {
		overflow-x: auto;
		margin-top: 0.6rem;
	}
	table {
		width: 100%;
		border-collapse: collapse;
		font-family: var(--mono);
		font-size: 0.66rem;
		font-variant-numeric: tabular-nums;
	}
	caption {
		text-align: left;
		font-size: 0.64rem;
		color: var(--ink-faint);
		padding-bottom: 0.25rem;
	}
	th,
	td {
		border-bottom: 1px solid var(--rule);
		padding: 0.25rem 0.5rem 0.25rem 0;
		text-align: right;
		white-space: nowrap;
	}
	th[scope='row'],
	thead th:first-child {
		text-align: left;
	}
	thead th {
		color: var(--ink-faint);
		font-weight: 500;
	}
	tr.reference th[scope='row'] {
		color: var(--accent);
	}
</style>
