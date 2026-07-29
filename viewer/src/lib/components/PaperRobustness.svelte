<script lang="ts">
	/**
	 * How the margin behaves under two standard robustness checks.
	 *
	 * The PRIMARY result — the 1689-statement panel with the labels released in
	 * 2023, unmodified — is stated first, restated on the figure, and is the
	 * only result this component ever calls the result. Neither check replaces it:
	 * the simultaneous band re-measures the SAME margin under a family-wise
	 * correction, and the 1578-statement panel is explicitly OUR revision of those
	 * published labels, shown because dropping label-incomplete negatives is a fair
	 * ask and hiding its answer would be the dishonest option. The possessive that
	 * stays is the one that discloses who changed something; the possessive that
	 * goes is the one used as a name for a published method.
	 *
	 * Every number rendered here is READ off
	 * `data/results/indra_paper_literal_models_20260724/paper_margin_robustness.json`,
	 * whose generator asserts that its pointwise half reproduces the shipped
	 * head-to-head exactly. Geometry, label budgets and per-series encodings live
	 * in `$lib/data/paper-robustness`; this file only draws.
	 *
	 * EVERY SENTENCE OFF THE ARTIFACT DRAWS ITS `plain` HALF. Four did not: the
	 * metric line, the bootstrap design, the reference description and the
	 * dose-response basis all rendered the flat alias, which IS the shipped half —
	 * so the artifact's own idiom reached the screen ("the paper's own out-of-fold
	 * fold assignment", "deltas are arm minus the paper's re-run RF", "exactly one
	 * arm is negative"). The restatements existed the whole time and nothing drew
	 * them. The shipped bytes stay verbatim in the artifact and behind the audit
	 * boundary, where class (a5) keeps them; the reader gets the twin.
	 */
	import {
		PAPER_ROBUSTNESS_GEOMETRY as G,
		PAPER_ROBUSTNESS_SERIES,
		PAPER_ROBUSTNESS_SERIES_IDS,
		fmt3,
		type PaperRobustnessInterval,
		type PaperRobustnessLoad
	} from '$lib/data/paper-robustness';
	import type { Standing } from '$lib/data/paper-literal';

	/**
	 * The inspect table's own column is "clear of zero?", which is a question about
	 * WIDTH and not about direction — an interval entirely below zero is clear of
	 * zero, and the signed margin sits in the column beside it. Written as a TOTAL
	 * record over the three classes so the compiler demands an answer for each; the
	 * two-way `excludesZero ? 'yes' : 'no'` it replaces gave the right answer here
	 * and the wrong one everywhere it was copied to.
	 */
	const CLEAR_OF_ZERO: Record<Standing, string> = {
		ahead: 'yes',
		behind: 'yes',
		'not-significant': 'no'
	};

	let { data }: { data: PaperRobustnessLoad } = $props();

	const figure = $derived(data.status === 'ok' ? data.figure : null);
	const plotBottom = $derived(figure ? figure.height - G.axisPad : 0);

	/**
	 * Census of what the figure shows, counted off the drawn lanes rather than
	 * written into the prose. Every sentence below that states "n of m arms" reads
	 * one of these, so a rerun that changes the answer changes the sentence.
	 */
	const lanes = $derived(figure?.lanes ?? []);
	// `standing === 'ahead'` IS "the whole interval lies above zero", so these
	// counts are of intervals that clear zero UPWARD and cannot pick up a loss.
	// Their predecessor was `excludesZero && deltaPts > 0`, correct only because
	// someone remembered the second conjunct.
	const clearsPointwise = $derived(
		lanes.filter((lane) => lane.pointwise.standing === 'ahead').length
	);
	const clearsSimultaneous = $derived(
		lanes.filter((lane) => lane.simultaneous.standing === 'ahead').length
	);
	const clearsSensitivity = $derived(
		lanes.filter((lane) => lane.sensitivity.standing === 'ahead').length
	);
	const movedTowardZero = $derived(
		lanes.filter(
			(lane) => Math.abs(lane.sensitivity.deltaPts) < Math.abs(lane.pointwise.deltaPts)
		).length
	);
	/** The one arm below zero, by the artifact's own gated dose-response census. */
	const negativeLane = $derived(lanes.find((lane) => lane.pointwise.deltaPts < 0) ?? null);
	// The question here really is "significant either way, on all three views" —
	// the lane is already known to be below zero — so it is asked in the form that
	// says so and cannot be misread as a direction.
	const decisive = (value: PaperRobustnessInterval): boolean =>
		value.standing !== 'not-significant';
	const negativeIsDecisiveEverywhere = $derived(
		negativeLane !== null &&
			decisive(negativeLane.pointwise) &&
			decisive(negativeLane.simultaneous) &&
			decisive(negativeLane.sensitivity)
	);

	function x(valuePts: number): number {
		if (!figure) return G.plotLeft;
		const span = figure.domainMaxPts - figure.domainMinPts;
		return G.plotLeft + ((valuePts - figure.domainMinPts) / span) * (G.plotRight - G.plotLeft);
	}

	const zeroX = $derived(x(0));

	function diamond(cx: number, cy: number): string {
		return `M ${cx} ${cy - 4} L ${cx + 4} ${cy} L ${cx} ${cy + 4} L ${cx - 4} ${cy} Z`;
	}

	/** Signed AP points, two decimals, ASCII sign — matching the readout format. */
	function pts(value: number): string {
		return `${value >= 0 ? '+' : '-'}${Math.abs(value).toFixed(2)}`;
	}

	function interval(value: PaperRobustnessInterval): string {
		return `[${pts(value.lowPts)}, ${pts(value.highPts)}]`;
	}

	function pct(value: number): string {
		return `${(value * 100).toFixed(1)}%`;
	}

	function shortSha(value: string): string {
		return `${value.slice(0, 10)}…`;
	}
</script>

<section class="robustness" aria-labelledby="robustness-title">
	{#if data.status === 'unavailable' || figure === null}
		<div class="gate" role="status">
			<p class="eyebrow">robustness of the margin</p>
			<h2 id="robustness-title">The robustness checks are unavailable</h2>
			<p>{data.reason}</p>
			<code>{data.artifact_path}</code>
		</div>
	{:else}
		<header>
			<div>
				<p class="eyebrow">the same margin, under two standard checks</p>
				<h2 id="robustness-title">How far does the margin travel?</h2>
			</div>
			<strong>primary result unchanged</strong>
		</header>

		<div class="primary" role="note">
			<p>
				<strong>The primary result stands.</strong>
				Scored on the {figure.primaryPanel.nStatements.toLocaleString()} statements published in 2023,
				with the correct/incorrect labels released beside them ({figure.primaryPanel.labelField}) left
				exactly as released, {figure.lanes[0].display} beats the random forest we re-ran from that
				release by {pts(figure.lanes[0].pointwise.deltaPts)} points of average precision
				(one point = 0.01). Judged on its own, that model's 95% interval runs
				{interval(figure.lanes[0].pointwise)}, and the margin stayed above zero in
				{fmt3(figure.lanes[0].pGreaterThanZero)} of the resampled re-runs. Everything below
				re-measures <em>that</em> lead. Nothing below replaces it.
			</p>
		</div>

		<figure>
			<figcaption>
				How much better or worse than {figure.referenceDisplay}, in points of average precision — one
				number for how well a model's scores push the correct statements to the top of a ranked list
				(one point = 0.01)
			</figcaption>
			<svg
				viewBox="0 0 {G.width} {figure.height}"
				style:min-height="{figure.height}px"
				role="img"
				aria-labelledby="robustness-chart-title robustness-chart-desc"
			>
				<title id="robustness-chart-title"
					>Each reading model's margin over the random forest re-run from the code released with the
					2023 paper, shown three ways: the 95% interval for that model judged on its own, the wider
					interval that has to hold for all {figure.multiplicity.familySize} reading models at once,
					and the same margin after the incorrect statements whose review was never finished are
					dropped</title
				>
				<desc id="robustness-chart-desc"
					>One row per reading model, ordered by margin. In each row the upper line is the headline
					result on {figure.primaryPanel.nStatements} statements with the labels published in 2023,
					unmodified: a diamond at the measured margin, a solid bar for that model's own 95% interval, and
					a dashed bar around it for the wider interval that has to hold for all {figure.multiplicity
						.familySize} reading models at once. Where a dashed bar reaches past the zero line, that
					overhang is drawn as a filled block, so a bar that just grazes zero is visibly different from
					one that misses by a wide margin. The lower line is the same margin measured over only
					{figure.sensitivityPanel.nStatements} statements, after dropping the incorrect ones whose
					evidence review was never finished: an open circle and a finely dashed interval. The heavy
					vertical rule is zero — no difference from the random forest.</desc
				>

				{#each figure.ticksPts as tick (tick)}
					{#if tick !== 0}
						<line class="grid" x1={x(tick)} y1={G.topPad} x2={x(tick)} y2={plotBottom} />
					{/if}
					<text class="tick" x={x(tick)} y={plotBottom + 15}>{tick.toFixed(1)}</text>
				{/each}

				<!-- The zero rule. Every question on this panel is which side of this
				     line a bar ends on, so it gets a wash channel behind the data, a
				     heavy rule, and (after the lanes) a thin overlay that keeps it
				     continuous where a bar crosses it without hiding the bar. -->
				<rect
					class="zero-wash"
					x={zeroX - 3}
					y={G.topPad - 12}
					width="6"
					height={plotBottom + 6 - (G.topPad - 12)}
				/>
				<line class="zero" x1={zeroX} y1={G.topPad - 12} x2={zeroX} y2={plotBottom + 6} />
				<!-- 8px mono = 4.8165 u/char; the fit is measured in the builder and the
				     label flips side rather than clipping at the viewBox edge. -->
				<text
					class="zero-label"
					x={figure.zeroLabelFits ? zeroX + 6 : zeroX - 6}
					y={G.topPad - 16}
					text-anchor={figure.zeroLabelFits ? 'start' : 'end'}>{figure.zeroLabel}</text
				>

				{#each figure.lanes as lane, index (lane.id)}
					{#if index > 0}
						<line
							class="lane-rule"
							x1="0"
							y1={lane.y - G.laneHeight / 2}
							x2={G.width}
							y2={lane.y - G.laneHeight / 2}
						/>
					{/if}

					<!-- 9px mono = 5.4186 u/char; 150-unit gutter = 27 chars, enforced in
					     buildFigure(). Longest shipped display is 16. -->
					<text class="lane-label" x={G.labelAnchorX} y={lane.primaryY + 3}>{lane.display}</text>
					<text class="lane-sub" x={G.labelAnchorX} y={lane.sensitivityY + 3}
						>{figure.sensitivityPanel.nStatements}</text
					>

					{@const sim = PAPER_ROBUSTNESS_SERIES.simultaneous}
					{@const point = PAPER_ROBUSTNESS_SERIES.pointwise}
					{@const sens = PAPER_ROBUSTNESS_SERIES.sensitivity}

					<!-- Simultaneous band first, so the pointwise interval nests visibly
					     inside it: the widening IS the correction. -->
					<line
						class="bar"
						x1={x(lane.simultaneous.lowPts)}
						x2={x(lane.simultaneous.highPts)}
						y1={lane.primaryY}
						y2={lane.primaryY}
						stroke={sim.strokeVar}
						stroke-width={sim.strokeWidth}
						stroke-dasharray={sim.dash}
					>
						<title>{lane.title}</title>
					</line>
					{#each [lane.simultaneous.lowPts, lane.simultaneous.highPts] as end, cap (cap)}
						<line
							class="bar"
							x1={x(end)}
							x2={x(end)}
							y1={lane.primaryY - G.simultaneousCap}
							y2={lane.primaryY + G.simultaneousCap}
							stroke={sim.strokeVar}
							stroke-width={sim.strokeWidth}
						/>
					{/each}
					{#if lane.simultaneousAdverse}
						<!-- The overhang past zero, drawn to scale. Same series, same hue —
						     it is a part of that band, not another one. -->
						<rect
							class="adverse"
							x={x(lane.simultaneousAdverse.fromPts)}
							y={lane.primaryY - 3.5}
							width={Math.max(x(lane.simultaneousAdverse.toPts) - x(lane.simultaneousAdverse.fromPts), 0.6)}
							height="7"
							fill={sim.strokeVar}
						>
							<title
								>{lane.display}: the wider all-at-once interval reaches {pts(
									lane.simultaneousAdverse.fromPts === 0
										? lane.simultaneousAdverse.toPts
										: lane.simultaneousAdverse.fromPts
								)} points past zero, so it does not clear zero</title
							>
						</rect>
					{/if}

					<line
						class="bar"
						x1={x(lane.pointwise.lowPts)}
						x2={x(lane.pointwise.highPts)}
						y1={lane.primaryY}
						y2={lane.primaryY}
						stroke={point.strokeVar}
						stroke-width={point.strokeWidth}
					/>
					{#each [lane.pointwise.lowPts, lane.pointwise.highPts] as end, cap (cap)}
						<line
							class="bar"
							x1={x(end)}
							x2={x(end)}
							y1={lane.primaryY - G.pointwiseCap}
							y2={lane.primaryY + G.pointwiseCap}
							stroke={point.strokeVar}
							stroke-width={point.strokeWidth}
						/>
					{/each}
					<path class="mark" d={diamond(x(lane.pointwise.deltaPts), lane.primaryY)} fill={point.strokeVar}>
						<title>{lane.title}</title>
					</path>

					<line
						class="bar"
						x1={x(lane.sensitivity.lowPts)}
						x2={x(lane.sensitivity.highPts)}
						y1={lane.sensitivityY}
						y2={lane.sensitivityY}
						stroke={sens.strokeVar}
						stroke-width={sens.strokeWidth}
						stroke-dasharray={sens.dash}
					/>
					{#each [lane.sensitivity.lowPts, lane.sensitivity.highPts] as end, cap (cap)}
						<line
							class="bar"
							x1={x(end)}
							x2={x(end)}
							y1={lane.sensitivityY - G.sensitivityCap}
							y2={lane.sensitivityY + G.sensitivityCap}
							stroke={sens.strokeVar}
							stroke-width={sens.strokeWidth}
						/>
					{/each}
					<circle
						class="mark"
						cx={x(lane.sensitivity.deltaPts)}
						cy={lane.sensitivityY}
						r="3"
						fill="var(--paper)"
						stroke={sens.strokeVar}
						stroke-width="1.3"
					>
						<title
							>{lane.display} over only the {figure.sensitivityPanel.nStatements} statements whose
							review was finished: {pts(lane.sensitivity.deltaPts)} points, 95% interval for this
							model on its own {interval(lane.sensitivity)}</title
						>
					</circle>

					<!-- 8px mono = 4.8165 u/char; 112-unit gutter = 23 chars, enforced in
					     buildFigure(). Longest shipped readout is 22. -->
					<text class="readout" x={G.readoutX} y={lane.primaryY + 3}>{lane.readoutPrimary}</text>
					<text class="readout faint" x={G.readoutX} y={lane.sensitivityY + 3}
						>{lane.readoutSensitivity}</text
					>
				{/each}

				<!-- Drawn last so the zero rule stays one unbroken line through every
				     bar that crosses it; thin and semi-transparent so it never hides
				     the interval underneath. -->
				<line class="zero-over" x1={zeroX} y1={G.topPad - 12} x2={zeroX} y2={plotBottom + 6} />

				<line class="axis" x1={G.plotLeft} y1={plotBottom} x2={G.plotRight} y2={plotBottom} />
				<text class="axis-end" x={G.plotLeft} y={plotBottom + 30} text-anchor="start"
					>← the random forest scores better</text
				>
				<text class="axis-end" x={G.plotRight} y={plotBottom + 30} text-anchor="end"
					>evidence-gated reading scores better →</text
				>
				<text class="axis-label" x={(G.plotLeft + G.plotRight) / 2} y={plotBottom + 30}
					>average-precision difference, in points</text
				>
				<!-- 8px mono = 4.8165 u/char; centred on the 638-unit plot. -->
				<text class="axis-note" x={(G.plotLeft + G.plotRight) / 2} y={plotBottom + 43}
					>solid bar = this model's own 95% interval · dashed bar = the wider interval that holds for
					all {figure.multiplicity.familySize} at once · filled block = how far it reaches past zero</text
				>
			</svg>
		</figure>

		<ul class="legend">
			{#each PAPER_ROBUSTNESS_SERIES_IDS as id (id)}
				{@const style = PAPER_ROBUSTNESS_SERIES[id]}
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
						{:else if style.shape === 'open-circle'}
							<circle cx="22" cy="6" r="3" fill="var(--paper)" stroke={style.strokeVar} stroke-width="1.3" />
						{:else}
							<line x1="22" y1="1" x2="22" y2="11" stroke={style.strokeVar} stroke-width={style.strokeWidth} />
						{/if}
					</svg>
					<span>{style.legend}</span>
				</li>
			{/each}
		</ul>

		<div class="checks">
			<article>
				<h3>
					A · one interval wide enough for all {figure.multiplicity.familySize} reading models at once
				</h3>
				<p>
					The run plan <code>{figure.multiplicity.runPlanPath}</code> was frozen on
					<time>{figure.multiplicity.runPlanFrozenAt}</time>, before any of the comparisons it
					governs were run. It lists all {figure.multiplicity.runPlanStages.length} models
					({figure.multiplicity.runPlanStages.join(', ')}) and singles none of them out as the one we
					expected to win. With no favourite picked in advance, the interval has to be wide enough
					that all {figure.multiplicity.familySize} are right at the same time:
					with {figure.multiplicity.familySize} chances, one model can look like a winner on luck alone.
				</p>
				<p>
					Widening costs little here, because the {figure.multiplicity.familySize} models put the
					statements in nearly the same order (Spearman rank correlation
					{figure.multiplicity.scoreSpearmanMin.toFixed(2)}–{figure.multiplicity.scoreSpearmanMax.toFixed(
						2
					)}). Reusing the same resampled re-runs, correcting across all
					{figure.multiplicity.familySize} at once asks for
					{figure.multiplicity.criticalValue.toFixed(3)} standard errors, against
					{figure.multiplicity.pointwiseNormalCriticalValue.toFixed(3)} for one model judged alone and
					{figure.multiplicity.bonferroniCriticalValue.toFixed(3)} if they were treated as
					unrelated (the Bonferroni rule).
				</p>
			</article>

			<article>
				<h3>
					B · dropping the {figure.labelCompleteness.nDropped} incorrect statements whose review was never
					finished
				</h3>
				<p>
					{figure.labelCompleteness.nDropped} of the {figure.primaryPanel.nNegative} statements the 2023
					release marks incorrect carry <code>{figure.labelCompleteness.field} == false</code>, meaning
					nobody finished reviewing their evidence. The published labels still count them as
					<em>incorrect</em>, so leaving them out is our label revision — a change made here, not a
					cleaner reading of the data behind those labels. It is a check on how fragile the result
					is, never the headline.
				</p>
				<p>
					Leaving them out is not free either. It removes
					{pct(figure.labelCompleteness.droppedShareOfAllNegatives)} of the incorrect statements and
					takes the mix from {pct(figure.labelCompleteness.negativeFractionBefore)} incorrect to
					{pct(figure.labelCompleteness.negativeFractionAfter)}, so it changes what is being measured,
					not just how well it is labelled: on the easier mix the random forest's average
					precision rises from {figure.primaryPanel.referenceAp.toFixed(4)} to
					{figure.sensitivityPanel.referenceAp.toFixed(4)}. Nothing is retrained — the same scores are
					simply read over fewer statements.
				</p>
			</article>
		</div>

		<div class="reading">
			<h3>What the figure shows</h3>
			<ul>
				<li>
					On the labels published in 2023, {clearsPointwise} of the {figure.lanes.length} reading models
					are clearly ahead of the random forest when each is judged on its own — their whole 95%
					interval sits above zero. {clearsSimultaneous} stay clearly ahead once the interval is
					widened to be right about all {figure.lanes.length} at the same time. The filled block
					against the zero line shows how far past zero a widened interval reaches when it fails to
					clear it.
				</li>
				{#if negativeLane}
					<li>
						{negativeLane.display} is the only model that comes out <em>behind</em> the random forest{negativeIsDecisiveEverywhere
							? ', and the only one whose result is clear-cut every way we measure it: its whole interval stays below zero judged alone, judged against all of them at once, and on the reduced set of statements'
							: ''}. It is also the smallest model here and the other
						{figure.doseResponse.nPositive} are bigger, so the wins and the loss line up with model
						size rather than scattering.
					</li>
					<li class="basis">{figure.doseResponse.basisProse.plain}</li>
				{/if}
				<li>
					Over only the {figure.sensitivityPanel.nStatements} statements whose review was finished,
					every margin shrinks toward zero ({movedTowardZero} of {figure.lanes.length}), and
					{clearsSensitivity} of the models that were ahead stay clearly ahead.
				</li>
				<li>
					The widest interval reaches {figure.worstHalfWidthPts.toFixed(2)} points either side of the
					margin it surrounds, and the largest margin on the figure is only
					{figure.bestDeltaPts.toFixed(2)} points: the uncertainty is the same size as the effect. The
					lead is real on the published data and it is borderline at this many statements. That is too
					few statements to settle the question, not an absent effect, and it is a statement about
					these {figure.primaryPanel.nStatements.toLocaleString()} statements rather than a verdict on
					the models.
				</li>
			</ul>
		</div>

		<details>
			<summary>Inspect every value on this figure</summary>
			<div class="table-scroll">
				<table>
					<thead>
						<tr>
							<th>model</th>
							<th>statements scored</th>
							<th>average precision</th>
							<th>margin vs the random forest</th>
							<th>95% interval, this model alone</th>
							<th>95% interval, all models at once</th>
							<th>share of re-runs ahead</th>
							<th>clear of zero?</th>
						</tr>
					</thead>
					<tbody>
						{#each figure.lanes as lane (lane.id)}
							<tr>
								<td>{lane.display}</td>
								<td>{figure.primaryPanel.nStatements} · the published labels</td>
								<td>{lane.primaryAp.toFixed(4)}</td>
								<td>{pts(lane.pointwise.deltaPts)}</td>
								<td>{interval(lane.pointwise)}</td>
								<td>{interval(lane.simultaneous)}</td>
								<td>{fmt3(lane.pGreaterThanZero)}</td>
								<td
									>alone: {CLEAR_OF_ZERO[lane.pointwise.standing]} · all at once:
									{CLEAR_OF_ZERO[lane.simultaneous.standing]}</td
								>
							</tr>
							<tr class="sensitivity-row">
								<td></td>
								<td>{figure.sensitivityPanel.nStatements} · unreviewed errors dropped</td>
								<td>{lane.sensitivityAp.toFixed(4)}</td>
								<td>{pts(lane.sensitivity.deltaPts)}</td>
								<td>{interval(lane.sensitivity)}</td>
								<td>—</td>
								<td>{fmt3(lane.sensitivityPGreaterThanZero)}</td>
								<td>alone: {CLEAR_OF_ZERO[lane.sensitivity.standing]}</td>
							</tr>
						{/each}
					</tbody>
				</table>
			</div>
			<p class="table-note">
				Every margin is measured against {figure.referenceDisplay} — {figure.prose.referenceDescription
					.plain}
				Metric: {figure.prose.metric.plain}
				{figure.prose.bootstrapDesign.plain}
				{figure.nBootstrap.toLocaleString()} resamples, seed {figure.seed}. The
				{figure.primaryPanel.nStatements}-statement rows reproduce the head-to-head table above to
				within {figure.shippedResidual.toExponential(0)}; the all-models-at-once interval comes from
				those same resampled re-runs, not from a second round of resampling.
			</p>
		</details>

		<footer>
			<!-- No '—' placeholder for a missing digest: an unpinned artifact says so,
			     because a dash beside a path reads as "pinned, just abbreviated". -->
			<code>{data.artifact_path}</code> · artifact
			{#if data.artifact_sha256}<span title={data.artifact_sha256}
					>{shortSha(data.artifact_sha256)}</span
				>{:else}<span class="sha-missing">not sha-pinned</span>{/if}<br />
			Run plan <code>{figure.multiplicity.runPlanPath}</code>
			<span title={figure.multiplicity.runPlanSha256}>{shortSha(figure.multiplicity.runPlanSha256)}</span>
			· frozen {figure.multiplicity.runPlanFrozenAt}
		</footer>
	{/if}
</section>

<style>
	.robustness {
		margin-top: 1.5rem;
		padding-top: 1.25rem;
		border-top: 2px solid var(--ink);
	}
	.robustness > header {
		display: flex;
		justify-content: space-between;
		align-items: start;
		gap: 1rem;
	}
	.robustness h2,
	.gate h2 {
		margin: 0.18rem 0 0;
		font-family: var(--serif);
		font-size: 1.35rem;
		font-weight: 400;
	}
	.robustness > header > strong {
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
	.primary {
		margin-top: 0.9rem;
		padding: 0.75rem 0.9rem;
		border: 1px solid var(--accent);
		border-left-width: 3px;
		background: var(--accent-wash);
		font-family: var(--serif);
		font-size: 0.85rem;
		line-height: 1.55;
		color: var(--ink-muted);
	}
	.primary p {
		margin: 0;
		max-width: 74ch;
	}
	.primary strong {
		color: var(--ink);
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
	.zero {
		stroke: var(--ink);
		stroke-width: 2.4;
	}
	.zero-wash {
		fill: color-mix(in srgb, var(--ink) 8%, transparent);
	}
	.zero-over {
		stroke: var(--ink);
		stroke-width: 0.9;
		opacity: 0.55;
	}
	.lane-rule {
		stroke: var(--rule);
		stroke-width: 1;
		stroke-dasharray: 2 4;
	}
	.bar {
		stroke-linecap: butt;
	}
	.adverse {
		opacity: 0.9;
	}
	.tick,
	.lane-label,
	.lane-sub,
	.readout,
	.axis-label,
	.axis-note,
	.axis-end,
	.zero-label {
		font-family: var(--mono);
	}
	.tick {
		font-size: 9px;
		fill: var(--ink-faint);
		text-anchor: middle;
	}
	/* Right-anchored: budgeted at 27 chars in buildFigure(). */
	.lane-label {
		font-size: 9px;
		fill: var(--ink);
		text-anchor: end;
	}
	.lane-sub {
		font-size: 7.5px;
		fill: var(--ink-faint);
		text-anchor: end;
	}
	/* Left-anchored: budgeted at 23 chars in buildFigure(). */
	.readout {
		font-size: 8px;
		fill: var(--ink-muted);
		text-anchor: start;
		font-variant-numeric: tabular-nums;
	}
	.readout.faint {
		fill: var(--ink-faint);
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
	.axis-end {
		font-size: 8px;
		fill: var(--ink-faint);
	}
	.zero-label {
		font-size: 8px;
		fill: var(--ink);
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
	.checks h3,
	.reading h3 {
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
	.reading {
		margin-top: 1.1rem;
		padding: 0.8rem 0.95rem;
		border: 1px solid var(--blocked);
		border-left-width: 3px;
		background: color-mix(in srgb, var(--blocked) 3%, transparent);
	}
	.reading ul {
		margin: 0;
		padding-left: 1.1rem;
	}
	.reading li {
		font-family: var(--serif);
		font-size: 0.82rem;
		line-height: 1.55;
		color: var(--ink-muted);
		max-width: 78ch;
	}
	.reading li + li {
		margin-top: 0.45rem;
	}
	.reading li.basis {
		font-family: var(--mono);
		font-size: 0.68rem;
		color: var(--ink-faint);
		list-style: none;
		margin-left: -1.1rem;
		padding-left: 0.6rem;
		border-left: 1px solid var(--rule);
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
	td:nth-child(n + 3) {
		font-variant-numeric: tabular-nums;
	}
	tr.sensitivity-row td {
		color: var(--ink-faint);
		border-bottom: 1px solid var(--rule);
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
	/* An absent digest is stated, never dashed: --blocked is this page's token for
	   "this is a caveat", and it must not read as a shortened hash. */
	.sha-missing {
		color: var(--blocked);
	}
	time {
		font-family: var(--mono);
		font-size: 0.72rem;
		color: var(--ink);
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
		.robustness > header {
			display: block;
		}
		.robustness > header > strong {
			display: inline-block;
			margin-top: 0.6rem;
		}
	}
</style>
