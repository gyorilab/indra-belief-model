<!--
  ReasoningAblation — what the thinking step was actually buying.

  Every reading model on this page was run TWICE over the identical 33,361
  readings: once with the provider's private deliberation on and a prompt telling
  the model to argue with itself before answering, once with both removed so it
  emits a verdict and nothing else. Same evidence, same statements, same
  reliability numbers, same rollup, same score cutoff. The difference is the
  thinking.

  PLAIN-LANGUAGE RULE. Written for a working biologist. In RENDERED PROSE — never
  in identifiers or these comments — "arm" is a model, "plate" is a figure, "lane"
  is a row, and the two runs are "thinking first" and "verdict only". AUROC is
  defined in the caption that uses it.

  ── TWO REGISTERS, TWO GRAINS, ONE FINDING ──────────────────────────────────
  Register A scores the assembled statement — the level the benchmark is built
  at — on ERROR-CATCHING F1, which is what /paper leads on and what a reading
  model is actually for. One row per model, marks on a shared axis, joined by a
  connector.

  THIS REGISTER WAS FIRST DRAWN ON AUROC, AND IT SAID THE OPPOSITE. Five
  statements in seven on this benchmark are already correct, so a ranking measure
  spends its range on the undisputed majority: on that axis three of the four
  models barely moved. On the error class those three are the decided ones. The
  ranking numbers are still here, in the table, because they are true — they are
  just not the claim.

  THAT CONNECTOR IS A PAIRED BEFORE-AND-AFTER, and this figure is allowed to say
  so where PerEvidenceGrain is not. Its connector joins two different item
  populations at two different base rates, so it carries a disclaimer. This one
  joins the SAME 1,689 statements scored twice, and the range beside it is a
  paired resample over exactly those statements. Direction is never decided here:
  every signed sentence is looked up in REASONING_ABLATION_STANDING_SENTENCE,
  which is keyed by the three interval classes, so there is no two-way test to be
  sign-blind with.

  Register B is a COUNT axis and it is drawn separately for that reason. It shows
  where the readings went: how many the model rejected while thinking and accepted
  without, against how many went the other way. Putting counts on the AUROC axis
  above would be a category error, and the two registers are the whole point —
  register B is large and one-sided on every model, register A mostly is not.

  NOTHING HERE IS RECOMPUTED. Every measurement, range bound, transition count
  and dollar bound is READ off the shipped artifact. The producer will not write
  that artifact unless the thinking-first side, rebuilt from its own raw readings,
  reproduces the statement scores and the measurements this page already publishes
  EXACTLY, and the validator re-gates on the flags recording those checks.

  THE ONE THING THIS SURFACE MAY NOT IMPLY. The verdict-only run has no packaging
  digest behind it: the packaging step refuses it, for a reason the artifact
  carries verbatim. `verdictOnlyBundled` is typed `false` and the refusal is
  printed, in the artifact's own words and in plain English beside them.

  ── SVG LABEL BUDGET (measured; right-anchored text clips leading glyphs
  silently, and a <desc> does not save it) ───────────────────────────────────
  At 9px the mono advance is 5.4186 user units per character; at 8px, 4.8165.
  viewBox 920 wide, plot x ∈ [196, 782].

    model names  right-anchored at x=184 → 184 u = 33 chars at 9px. Longest
                 shipped is "Gemma 4 E2B" at 11 chars = 59.6 u. ENFORCED by
                 budget() in parseModel().
    readouts     left-anchored at x=790 → (920−790) = 130 u = 26 chars at 8px.
                 Longest built here is the range line, "−0.0221 [−0.0375,−0.0069]"
                 at 25 chars = 120.4 u. CHECKED by reasoningAblationReadoutFits()
                 and the row gates rather than clips.
    band heads   start-anchored at x=0 → 920 u = 190 chars at 8px.
    axis ticks   4 chars = 19.3 u, middle-anchored inside the plot.
-->
<script lang="ts">
	import {
		REASONING_ABLATION_GEOMETRY,
		REASONING_ABLATION_SIDE_DISPLAY,
		REASONING_ABLATION_SIMULTANEOUS_SENTENCE,
		REASONING_ABLATION_STANDING_SENTENCE,
		fmt3,
		fmtSignedDelta,
		reasoningAblationExtent,
		reasoningAblationReadoutFits,
		type ReasoningAblationBenchmark,
		type ReasoningAblationBenchmarkId,
		type ReasoningAblationLoad,
		type ReasoningAblationModel
	} from '$lib/data/paper-reasoning-ablation';

	let { data }: { data: ReasoningAblationLoad } = $props();

	const G = REASONING_ABLATION_GEOMETRY;
	const figure = $derived(data.status === 'ok' ? data.figure : null);

	/** The benchmark drawn in full. The other is reported in the detail table. */
	const DRAWN: ReasoningAblationBenchmarkId = 'paper_all_source';

	const models = $derived(figure === null ? [] : figure.models);

	function benchmarkOf(model: ReasoningAblationModel): ReasoningAblationBenchmark | null {
		return model.benchmarks.find((entry) => entry.id === DRAWN) ?? null;
	}

	/**
	 * ERROR-CATCHING axis, padded off the observed extent so no mark sits on the
	 * frame. Null-safe: an empty figure never divides by a zero span.
	 *
	 * The extent must cover every mark the figure draws, and the figure draws
	 * THREE per model — both sides at their own cut, plus the verdict-only side at
	 * the cutoff left where it was. Sizing on two of the three would push the
	 * third off the frame, silently, which for Gemma 4 E2B it does.
	 */
	const extent = $derived.by(() => {
		const values: number[] = [];
		for (const model of models) {
			const benchmark = model.benchmarks.find((entry) => entry.id === DRAWN);
			if (!benchmark) continue;
			values.push(
				benchmark.errorClass.ownCut.reasoning.errorF1,
				benchmark.errorClass.ownCut.verdictOnly.errorF1,
				benchmark.errorClass.deployedCut.verdictOnly.errorF1
			);
		}
		if (values.length === 0) return null;
		return { min: Math.min(...values), max: Math.max(...values) };
	});
	const domain = $derived.by(() => {
		if (extent === null) return { min: 0.45, max: 0.8 };
		const pad = Math.max((extent.max - extent.min) * 0.18, 0.006);
		return { min: extent.min - pad, max: extent.max + pad };
	});
	const ticks = $derived.by(() => {
		const step = domain.max - domain.min > 0.2 ? 0.05 : 0.02;
		const out: number[] = [];
		const start = Math.ceil(domain.min / step) * step;
		for (let value = start; value <= domain.max + 1e-9; value += step) {
			out.push(Math.round(value * 1000) / 1000);
		}
		return out;
	});

	function x(value: number): number {
		const span = domain.max - domain.min;
		if (span <= 0) return G.plotLeft;
		return G.plotLeft + ((value - domain.min) / span) * (G.plotRight - G.plotLeft);
	}

	/** Count axis for register B. Symmetric about its own centre so the two
	 *  directions are comparable by length rather than by reading two scales. */
	const flipMax = $derived(
		models.length === 0
			? 1
			: Math.max(...models.map((m) => Math.max(m.evidence.toCorrect, m.evidence.toIncorrect)))
	);
	const flipCentre = (G.plotLeft + G.plotRight) / 2;
	function flipWidth(count: number): number {
		if (flipMax <= 0) return 0;
		return (count / flipMax) * ((G.plotRight - G.plotLeft) / 2);
	}

	/**
	 * Whether any model's two threshold rules chose DIFFERENT cutoffs. When none
	 * do, the third mark never draws and the legend must not promise it — a key
	 * naming a mark that is not on the chart is the same defect as a label that
	 * clips, just spelled differently.
	 */
	const anyCutDiffers = $derived(
		models.some((model) => {
			const benchmark = model.benchmarks.find((entry) => entry.id === DRAWN);
			return benchmark !== undefined && !benchmark.errorClass.cutsAgree;
		})
	);

	const rowY = (index: number) => G.topPad + index * G.rowHeight;
	const registerBTop = $derived(G.topPad + models.length * G.rowHeight + 74);
	const height = $derived(registerBTop + models.length * G.rowHeight + 44);
	const axisAY = $derived(G.topPad + models.length * G.rowHeight - 14);
	const axisBY = $derived(registerBTop + models.length * G.rowHeight - 14);

	/**
	 * Tick counts for register B, symmetric about the centre. Without them the
	 * count axis is a bare rule and the bars read as a proportion of nothing —
	 * the reader cannot tell 1,500 changed readings from 7,000.
	 */
	const flipTicks = $derived.by(() => {
		if (flipMax <= 0) return [] as number[];
		// A 1/2/5 ladder sized so at most four ticks land on each side. A finer
		// step overlaps its own labels, which is the same silent-clipping defect
		// the gutter budgets exist to prevent, just in the middle of the plot.
		const target = flipMax / 4;
		const magnitude = Math.pow(10, Math.floor(Math.log10(target)));
		const step = (([1, 2, 5, 10].find((m) => magnitude * m >= target) ?? 10) * magnitude);
		const out: number[] = [];
		for (let value = step; value <= flipMax; value += step) out.push(value);
		return out;
	});

	function diamond(cx: number, cy: number, r = 4.2): string {
		return `M ${cx} ${cy - r} L ${cx + r} ${cy} L ${cx} ${cy + r} L ${cx - r} ${cy} Z`;
	}

	function pct(value: number): string {
		return `${(value * 100).toFixed(1)}%`;
	}

	function shortSha(value: string): string {
		return `${value.slice(0, 10)}…`;
	}

	function usd(value: number): string {
		return `$${value.toFixed(2)}`;
	}

	/** "−0.0689 [−0.1010,−0.0378]" — the row's inferential claim, budget-checked. */
	function rangeReadout(benchmark: ReasoningAblationBenchmark): string {
		const delta = benchmark.errorF1OwnCutDelta;
		return `${fmtSignedDelta(delta.value)} [${fmtSignedDelta(delta.low)},${fmtSignedDelta(delta.high)}]`;
	}

	function valueReadout(benchmark: ReasoningAblationBenchmark): string {
		return `${fmt3(benchmark.errorClass.ownCut.reasoning.errorF1)} → ${fmt3(
			benchmark.errorClass.ownCut.verdictOnly.errorF1
		)}`;
	}

	/**
	 * A row draws only when BOTH of its readouts fit their gutter. A clipped
	 * readout is the defect this budget exists for, and gating one row is cheaper
	 * than printing a number missing its leading digits.
	 */
	function rowFits(benchmark: ReasoningAblationBenchmark): boolean {
		return (
			reasoningAblationReadoutFits(rangeReadout(benchmark)) &&
			reasoningAblationReadoutFits(valueReadout(benchmark))
		);
	}
</script>

<section class="ablation" aria-labelledby="reasoning-ablation-title">
	{#if data.status === 'unavailable' || figure === null}
		<div class="gate" role="status">
			<p class="eyebrow">thinking on, thinking off</p>
			<h2 id="reasoning-ablation-title">The thinking comparison is unavailable</h2>
			<p>{data.reason}</p>
			<code>{data.artifact_path}</code>
		</div>
	{:else}
		<header>
			<div>
				<p class="eyebrow">the same readings, twice</p>
				<h2 id="reasoning-ablation-title">What the thinking step was buying</h2>
			</div>
			<strong>{figure.census.nExecutions.toLocaleString()} readings, each run twice</strong>
		</header>

		<!-- No lead paragraph: the page's own `.framing` lead-in above this figure
		     does that job, and AUROC is defined in the figure immediately before
		     this one, so the caption below does not spend the page's word budget
		     defining it a second time. -->
		<figure>
			<figcaption>
				Above: finding the {figure.census.errors[DRAWN]} wrong statements among {figure.census.statements[
					DRAWN
				].toLocaleString()}, scored twice, at cutoffs picked on these same statements. Below: single
				readings that changed answer.
			</figcaption>
			<svg
				viewBox="0 0 {G.width} {height}"
				style:min-height="{height}px"
				role="img"
				aria-labelledby="reasoning-ablation-chart-title reasoning-ablation-chart-desc"
			>
				<title id="reasoning-ablation-chart-title"
					>How each reading model scores with its thinking step and without it, and how many
					individual readings changed answer when that step was removed</title
				>
				<desc id="reasoning-ablation-chart-desc"
					>Two charts stacked on one figure. The upper one has a row per model on a shared AUROC
					axis. The filled diamond is the model with its thinking step, the open circle is the same
					model without it, and the hairline between them is a genuine before-and-after: the same
					statements, scored twice, so the range printed beside each row is a paired resample over
					exactly those statements ({figure.resamples.toLocaleString()} draws). A row is drawn in the
					emphasis colour only when that range does not cover zero. The lower chart is a separate
					count axis, because a count of readings may not be drawn on the scale above. Each model has
					one row split at the centre: the bar growing right counts readings the model rejected while
					thinking and accepted without it, the bar growing left counts readings that went the other
					way. On every model the right bar is the longer one, which is the finding — the thinking
					step was mostly producing rejections.</desc
				>

				<!-- ── register A: discrimination ───────────────────────────────────
				     Its grid STOPS at its own axis rule. The register below carries a
				     count, not a discrimination, so a shared grid line would invite
				     reading a bar against the wrong scale. -->
				<text class="band-head" x="0" y={G.topPad - 26}
					>PER ASSEMBLED STATEMENT · filled = thinking first, open = verdict only{#if anyCutDiffers}, tick
						= cutoff left alone{/if}</text
				>
				{#each ticks as tick (tick)}
					<line class="grid" x1={x(tick)} y1={G.topPad - 16} x2={x(tick)} y2={axisAY} />
					<text class="tick" x={x(tick)} y={axisAY + 14}>{tick.toFixed(2)}</text>
				{/each}

				{#each models as model, index (model.id)}
					{@const benchmark = benchmarkOf(model)}
					{#if benchmark && rowFits(benchmark)}
						{@const y = rowY(index)}
						<!-- Emphasis keys off the CORRECTED band, not the pointwise one. Two
						     of these four models clear zero pointwise by less than 0.001, and
						     drawing them as settled would be the page's own sixth regression
						     wearing different clothes. -->
						{@const decided =
							benchmark.errorF1OwnCutDelta.simultaneousStanding !== 'not-significant'}
						{@const own = benchmark.errorClass.ownCut}
						{@const kept = benchmark.errorClass.deployedCut.verdictOnly}
						<text class="model-name" x={G.labelAnchorX} y={y + 3}>{model.display}</text>

						<line
							class="connector"
							class:decided
							x1={x(own.reasoning.errorF1)}
							y1={y}
							x2={x(own.verdictOnly.errorF1)}
							y2={y}
						/>
						<path class="mark" d={diamond(x(own.reasoning.errorF1), y)} fill="var(--accent)">
							<title
								>{model.display} with its thinking step: error-catching F1 {fmt3(
									own.reasoning.errorF1
								)}, catching {own.reasoning.tp} of {benchmark.nNegative} wrong statements at a score cutoff
								of {fmt3(own.reasoning.tau)}</title
							>
						</path>
						<!-- The THIRD mark, and the reason it exists: the open circle is the
						     verdict-only side allowed to re-pick its cutoff, which is an
						     oracle. This tick is the same side with the cutoff left where it
						     was. Where they separate, the gap IS what re-tuning buys back. -->
						{#if !benchmark.errorClass.cutsAgree}
							<line
								class="kept-cut"
								x1={x(kept.errorF1)}
								x2={x(own.verdictOnly.errorF1)}
								y1={y}
								y2={y}
							/>
							<line
								class="kept-tick"
								x1={x(kept.errorF1)}
								x2={x(kept.errorF1)}
								y1={y - 5}
								y2={y + 5}
							>
								<title
									>{model.display} with the thinking step removed and the score cutoff left where it
									was: error-catching F1 {fmt3(kept.errorF1)}, catching {kept.tp} of {benchmark.nNegative}
									wrong statements. The distance from here to the open circle is what re-picking the cutoff
									on these same statements buys back.</title
								>
							</line>
						{/if}
						<circle
							class="mark"
							cx={x(own.verdictOnly.errorF1)}
							cy={y}
							r="3.6"
							fill="var(--paper)"
							stroke="var(--ink)"
							stroke-width="1.3"
						>
							<title
								>{model.display} with the thinking step removed, at its own best cutoff {fmt3(
									own.verdictOnly.tau
								)}: error-catching F1 {fmt3(own.verdictOnly.errorF1)}, catching {own.verdictOnly.tp} of
								{benchmark.nNegative} wrong statements. {REASONING_ABLATION_STANDING_SENTENCE[
									benchmark.errorF1OwnCutDelta.standing
								]} {REASONING_ABLATION_SIMULTANEOUS_SENTENCE[
									benchmark.errorF1OwnCutDelta.simultaneousStanding
								]}</title
							>
						</circle>

						<text class="readout" x={G.readoutX} y={y + 3}>{valueReadout(benchmark)}</text>
						<!-- The emphasis marks "this change is separable from zero", a question
						     about the range's width and not about its direction — the signed
						     number is printed right there, and the sentence that names a
						     direction is looked up by interval class, never branched on. -->
						<text class="range" class:decided x={G.readoutX} y={y + 13}
							>{rangeReadout(benchmark)}</text
						>
					{/if}
				{/each}
				<line class="axis" x1={G.plotLeft} y1={axisAY} x2={G.plotRight} y2={axisAY} />
				<text class="axis-label" x={(G.plotLeft + G.plotRight) / 2} y={axisAY + 30}
					>error-catching F1 — precision and recall on the wrong statements</text
				>

				<!-- ── register B: counts — its OWN axis, its own centre ───────────── -->
				<text class="band-head" x="0" y={registerBTop - 26}
					>PER SINGLE READING · left = accepted while thinking, then rejected · right = rejected
					while thinking, then accepted</text
				>
				<line class="centre" x1={flipCentre} y1={registerBTop - 16} x2={flipCentre} y2={axisBY} />
				{#each models as model, index (model.id)}
					{@const y = registerBTop + index * G.rowHeight}
					<text class="model-name" x={G.labelAnchorX} y={y + 3}>{model.display}</text>
					<rect
						class="flip to-incorrect"
						x={flipCentre - flipWidth(model.evidence.toIncorrect)}
						y={y - 7}
						width={flipWidth(model.evidence.toIncorrect)}
						height="14"
					>
						<title
							>{model.display}: {model.evidence.toIncorrect.toLocaleString()} readings it accepted while
							thinking and rejected without</title
						>
					</rect>
					<rect
						class="flip to-correct"
						x={flipCentre}
						y={y - 7}
						width={flipWidth(model.evidence.toCorrect)}
						height="14"
					>
						<title
							>{model.display}: {model.evidence.toCorrect.toLocaleString()} readings it rejected while
							thinking and accepted without</title
						>
					</rect>
					<text class="readout" x={G.readoutX} y={y + 3}
						>{pct(model.evidence.agreement)} unchanged</text
					>
				{/each}
				<line class="axis" x1={G.plotLeft} y1={axisBY} x2={G.plotRight} y2={axisBY} />
				{#each flipTicks as tick (tick)}
					{#each [-1, 1] as side (side)}
						<line
							class="grid"
							x1={flipCentre + side * flipWidth(tick)}
							y1={registerBTop - 16}
							x2={flipCentre + side * flipWidth(tick)}
							y2={axisBY}
						/>
						<text class="tick" x={flipCentre + side * flipWidth(tick)} y={axisBY + 14}
							>{tick.toLocaleString()}</text
						>
					{/each}
				{/each}
				<text class="tick" x={flipCentre} y={axisBY + 14}>0</text>
				<text class="axis-label" x={flipCentre} y={axisBY + 30}
					>readings that changed answer, out of {figure.census.nModelRead.toLocaleString()} the model
					was asked for</text
				>
			</svg>
		</figure>

		<details>
			<summary>Both benchmarks, the score cutoff, and what the run cost</summary>

			<div class="table-scroll">
				<table>
					<caption
						>Each model on both benchmarks, thinking first then verdict only. The first four columns
						are the wrong statements; the last three are the ranking measures, which move far less
						because five statements in every seven are already correct.</caption
					>
					<thead>
						<tr>
							<th scope="col">model</th>
							<th scope="col">error-catching F1</th>
							<th scope="col">cutoff left alone</th>
							<th scope="col">errors found</th>
							<th scope="col">of those flagged, wrong</th>
							<th scope="col">average precision</th>
							<th scope="col">AUROC</th>
							<th scope="col">calibration error</th>
						</tr>
					</thead>
					<tbody>
						{#each models as model (model.id)}
							{#each model.benchmarks as benchmark (benchmark.id)}
								{@const own = benchmark.errorClass.ownCut}
								{@const kept = benchmark.errorClass.deployedCut}
								<tr>
									<th scope="row">{model.display} · {benchmark.display}</th>
									<td>{fmt3(own.reasoning.errorF1)} → {fmt3(own.verdictOnly.errorF1)}</td>
									<td>{fmt3(kept.reasoning.errorF1)} → {fmt3(kept.verdictOnly.errorF1)}</td>
									<td
										>{own.reasoning.tp} → {own.verdictOnly.tp} of {benchmark.nNegative}</td
									>
									<td
										>{fmt3(own.reasoning.errorPrecision)} → {fmt3(own.verdictOnly.errorPrecision)}</td
									>
									<td
										>{fmt3(benchmark.reasoning.averagePrecision)} → {fmt3(
											benchmark.verdictOnly.averagePrecision
										)}</td
									>
									<td>{fmt3(benchmark.reasoning.auroc)} → {fmt3(benchmark.verdictOnly.auroc)}</td>
									<td>{fmt3(benchmark.reasoning.ece)} → {fmt3(benchmark.verdictOnly.ece)}</td>
								</tr>
							{/each}
						{/each}
					</tbody>
				</table>
			</div>

			<ul class="costs">
				{#each models as model (model.id)}
					<li>
						<strong>{model.display}</strong>
						{usd(model.reasoningCost.lower)}–{usd(model.reasoningCost.upper)} thinking first{#if model.verdictOnlyCost}, {usd(
								model.verdictOnlyCost.lower
							)}–{usd(model.verdictOnlyCost.upper)} verdict only{:else}; the verdict-only spend was
							not summed for this run{/if}
					</li>
				{/each}
			</ul>

			<!-- A threshold-fitted number without its threshold rule is an over-claim
			     wherever it is written. These three sentences are the rule, and the
			     oracle among them is not optional: every cutoff in the drawn figure
			     was chosen on the statements it is then scored on. -->
			<p class="basis rule">
				{figure.errorRules.decision.plain}
				{figure.errorRules.threshold.plain}
				<strong>{figure.errorRules.oracle.plain}</strong>
				That is why the figure also draws the cutoff left where it was: under that rule the
				verdict-only side gets no such advantage.
			</p>

			<p class="basis">
				{REASONING_ABLATION_SIDE_DISPLAY.reasoning} and {REASONING_ABLATION_SIDE_DISPLAY.verdict_only}
				were scored by the same code on the same statements. Ranges are {figure.resamples.toLocaleString()}
				paired resamples, drawn within each of the 10 folds — the statements are split into 10 groups,
				and each fitted model is scored on the group it did not train on — and keeping each group's size,
				with one shared draw across every model and both runs. A row is emphasised only when its range
				still misses zero after being widened to cover having run four models, not one.
			</p>

			<!-- The artifact's own wording of this refusal is carried in the file and
			     is checked against these restatements by pairShippedProse(), but it
			     is NOT printed here: the audit half of a prose twin stays behind the
			     page's single verification boundary. -->
			<p class="basis blocked">
				<strong>No packaging digest stands behind the verdict-only run.</strong>
				{figure.bundler.error.plain}
				{figure.bundler.cause.plain}
				{figure.bundler.consequence.plain}
			</p>

			<p class="basis">
				{data.artifact_path}{#if data.artifact_sha256}
					· sha256 {shortSha(data.artifact_sha256)}{/if} · frozen {figure.frozenAt}
			</p>
		</details>
	{/if}
</section>

<style>
	.ablation {
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
		margin-bottom: 0.4rem;
		max-width: 68ch;
		font-size: 0.78rem;
		line-height: 1.45;
		color: var(--ink-muted);
	}
	svg {
		width: 100%;
		height: auto;
	}
	.grid {
		stroke: var(--rule);
		stroke-width: 0.5;
	}
	.axis {
		stroke: var(--ink-faint);
		stroke-width: 0.8;
	}
	.centre {
		stroke: var(--ink-faint);
		stroke-width: 0.8;
	}
	.connector {
		stroke: var(--ink-faint);
		stroke-width: 0.9;
	}
	.connector.decided {
		stroke: var(--accent);
		stroke-width: 1.6;
	}
	.kept-cut {
		stroke: var(--ink-faint);
		stroke-width: 0.8;
		stroke-dasharray: 2 2;
	}
	.kept-tick {
		stroke: var(--ink);
		stroke-width: 1.3;
	}
	.mark {
		vector-effect: non-scaling-stroke;
	}
	.flip {
		fill: var(--ink-faint);
	}
	.flip.to-correct {
		fill: var(--accent);
	}
	.flip.to-incorrect {
		fill: var(--accent-wash);
		stroke: var(--accent);
		stroke-width: 0.6;
	}
	.band-head {
		font-family: var(--mono);
		font-size: 8px;
		letter-spacing: 0.06em;
		fill: var(--ink-faint);
	}
	.model-name {
		font-family: var(--mono);
		font-size: 9px;
		text-anchor: end;
		fill: var(--ink);
	}
	.tick {
		font-family: var(--mono);
		font-size: 8px;
		text-anchor: middle;
		fill: var(--ink-faint);
	}
	.readout {
		font-family: var(--mono);
		font-size: 8px;
		fill: var(--ink-muted);
	}
	.range {
		font-family: var(--mono);
		font-size: 8px;
		fill: var(--ink-faint);
	}
	.range.decided {
		fill: var(--accent);
	}
	.axis-label {
		font-family: var(--mono);
		font-size: 8px;
		text-anchor: middle;
		fill: var(--ink-faint);
	}
	details {
		margin-top: 1rem;
		font-size: 0.8rem;
		color: var(--ink-muted);
	}
	summary {
		cursor: pointer;
		font-family: var(--mono);
		font-size: 0.7rem;
		color: var(--ink-muted);
	}
	.table-scroll {
		overflow-x: auto;
		margin-top: 0.6rem;
	}
	table {
		border-collapse: collapse;
		font-family: var(--mono);
		font-size: 0.68rem;
		white-space: nowrap;
	}
	caption {
		text-align: left;
		padding-bottom: 0.4rem;
		font-family: var(--serif);
		font-size: 0.78rem;
		color: var(--ink-muted);
	}
	th,
	td {
		border-bottom: 1px solid var(--rule);
		padding: 0.28rem 0.6rem 0.28rem 0;
		text-align: left;
		font-weight: 400;
	}
	thead th {
		color: var(--ink-faint);
	}
	.costs {
		margin: 0.7rem 0 0;
		padding: 0;
		list-style: none;
		font-family: var(--mono);
		font-size: 0.68rem;
	}
	.costs li + li {
		margin-top: 0.2rem;
	}
	.costs strong {
		font-weight: 500;
		color: var(--ink);
	}
	.basis {
		margin: 0.6rem 0 0;
		max-width: 72ch;
		font-family: var(--mono);
		font-size: 0.66rem;
		line-height: 1.5;
		color: var(--ink-faint);
	}
	.basis.rule strong {
		font-weight: 500;
		color: var(--ink-muted);
	}
	.basis.blocked {
		border-left: 3px solid var(--blocked);
		padding-left: 0.6rem;
		color: var(--ink-muted);
	}
</style>
