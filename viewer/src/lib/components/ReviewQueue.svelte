<script lang="ts">
	/**
	 * The review queue — what each belief model would actually put in front of a
	 * human curator, at one matched operating point.
	 *
	 * Every model's score cutoff is lowered until it has flagged the SAME share of
	 * the benchmark's known errors (flag as wrong iff belief <= tau; tau is the
	 * smallest of that model's own distinct scores that reaches the target).
	 * Matched recall is what makes the bars comparable, so the only remaining
	 * variable is how much correct work a curator wades through to get there: one
	 * horizontal bar per model, split into errors actually caught and correct
	 * statements flagged anyway.
	 *
	 * PLAIN-LANGUAGE RULE (2026-07-27). Written for a working biologist who has
	 * never read the 2023 paper and does not work in machine-learning evaluation.
	 * In RENDERED PROSE — never in identifiers, artifact field names or these
	 * comments — "arm" is a model, "panel" is the benchmark or plainly the 1,689
	 * statements, "tau" is a score cutoff, "gate" is evidence-gated reading,
	 * "oracle" is a cutoff tuned with the answers already in hand, "max-t" is the
	 * correction for testing all four reading models at once, and "pts" is
	 * percentage points. A term that needs prior knowledge is defined in the
	 * sentence that first uses it or replaced. The decision rule, the cutoff rule,
	 * the disclosures and the caveats all render their PLAIN half — `prose.x.plain`,
	 * never the flat field beside it. The method note used to introduce three of
	 * them with "in the artifact's own words" and print the shipped bytes; that
	 * wording now lives, with every other shipped sentence on the page, in the one
	 * verification section at the end (`PaperAuditTrail.svelte`), each beside the
	 * restatement that replaced it and under the file name and digest it came from.
	 *
	 * Every number is read from `statement_review_queue.json` through the
	 * fail-closed validator — no count, threshold, share, interval or caveat is
	 * hard-coded here; only layout constants are. The method caveats render in the
	 * `<details>` beneath the caption, following the ApDecompositionByPaperRank
	 * precedent, and their COUNT is asserted by the data module so one cannot
	 * quietly go missing. Nothing there may be deleted — only relocated.
	 *
	 * Beneath it, the same benchmark with the YIELD fixed instead of the cutoff —
	 * the operational form of the result, and the one an author reads off without
	 * a metric. Evidence-gated reading's operating point is one nobody chose (the
	 * statements whose evidence the reader rejected outright); the sweep shows what
	 * each model finds at every review budget, and the two brackets read straight
	 * off it: how many MORE statements the strongest formula-based belief model
	 * must have read for the same catch, and how many FEWER errors it finds for the
	 * same amount of reading.
	 *
	 * THE ASYMMETRY IS THE CLAIM. The comparator's cutoff is chosen on these very
	 * statements with these labels already in hand — nobody could have picked it
	 * before the curation existed. Evidence-gated reading has NO cutoff at all: its
	 * point is the block of statements whose evidence it rejected outright, which
	 * nobody tuned and which is the same block on any benchmark. The side handed
	 * the advantage still needs ~200 more reviews. Both halves of that must be
	 * legible from the figure WITHOUT opening the method note, which is why "no
	 * cutoff" and "cutoff tuned on the answers" are drawn on the chart itself
	 * rather than only written in it.
	 *
	 * NAMING RULE. Name the method; cite the paper as provenance, never as
	 * identity. The comparator is a random forest on evidence-count features —
	 * 2,000 trees, depth 13, over per-source evidence counts plus statement type,
	 * PMID count and a promoter flag — released with the 2023 INDRA assembly paper
	 * and re-run here so that each statement is scored by a copy of the model that
	 * never learned from it. It is the one model on this benchmark that paper
	 * actually published. Our CountsScorer port and our BayesianScorer refit are
	 * carried too, and none of the three may be called "the paper's RF": see each
	 * model's `provenance`.
	 *
	 * Deliberate constraints (do not "improve" these):
	 *   · ONE shared count scale across all nine bars, so length is directly
	 *     comparable. No per-bar normalisation, no percentage bars.
	 *   · Bars ordered by queue size ascending — least work at the top.
	 *   · No interaction, no toggle, no target-recall selector, and no legend
	 *     beyond the two inline swatch labels above the first bar.
	 *   · The sweep draws the WHOLE budget range, including the budgets where
	 *     evidence-gated reading LOSES. The advantage is a property of how much
	 *     gets read, and an author who suspects a cherry-picked operating point
	 *     has to be able to see that.
	 *   · The word "the paper reported" must never appear: the 2023 paper published
	 *     no decision or threshold metric. This is our derivation on their model.
	 *
	 * PROSE BUDGET. /paper caps reader-facing words and this figure carries the
	 * page's strongest result, so every visible sentence here is paid for. Text in
	 * <details>, <desc>, <title> and .sr is NOT counted — method detail belongs
	 * there, and the hidden heading is `.sr` for exactly that reason. If you add a
	 * visible word, take one out.
	 */
	import {
		buildReviewQueueSweep,
		reviewQueueColorVar,
		reviewQueueDisplayOrder,
		reviewQueueEqualYieldPair,
		reviewQueueRobustnessHeadline,
		REVIEW_QUEUE_PROVENANCE_GLOSS,
		REVIEW_QUEUE_WIDENED_BAND_CLAUSE as WIDENED_BAND_CLAUSE,
		type ReviewQueueArm,
		type ReviewQueueLoad
	} from '$lib/data/paper-review-queue';

	let { data }: { data: ReviewQueueLoad } = $props();

	// ---- layout (900 units wide; HEIGHT DERIVED from the row count) ----------
	// Widened from 150 when the arms took their canonical names: the longest,
	// 'BayesianScorer, source+subtype refit' (36 chars), needs 36 x 5.4186 = 195u
	// at 9px mono. Right-anchored at GUTTER_RIGHT - 4, a 150 gutter put its left
	// edge at x = -49, outside the viewBox, and the SVG viewport would have eaten
	// nine glyphs while <desc> still read the label in full. BAR_SPAN rescales the
	// bars automatically, so moving BAR_LEFT with it costs only bar width.
	//
	// The VERTICAL extent used to be two constants (AXIS_Y = 300, viewBox 340)
	// sized by hand for seven arms. Adding two arms silently pushed the last two
	// rows past the viewBox edge and drew the axis straight through the seventh —
	// no error, no test failure, just two missing models. Both are now derived
	// from `rows.length`, so the panel grows with its data instead of clipping it.
	const GUTTER_RIGHT = 210;
	const BAR_LEFT = 218;
	const BAR_RIGHT_MAX = 690;
	const LEGEND_Y = 62;
	const ROW_TOP = 74;
	const ROW_PITCH = 38;
	const BAR_H = 20;
	/** Clear air between the last bar's baseline and the shared count axis. */
	const AXIS_GAP = 24;
	/** Room under the axis for its tick labels (+14) and its title (+30). */
	const AXIS_FOOT = 40;
	const BAR_SPAN = BAR_RIGHT_MAX - BAR_LEFT;
	/** Candidate axis steps, coarsest tick set that still gives >= 3 gridless ticks. */
	const TICK_STEPS = [25, 50, 100, 200, 250, 500, 1000];

	function r2(value: number): number {
		return Math.round(value * 100) / 100;
	}
	function pct(value: number): string {
		return `${(value * 100).toFixed(1)}%`;
	}
	function whole(value: number): string {
		return `${Math.round(value * 100)}%`;
	}
	/**
	 * A recall DIFFERENCE, in percentage points, always signed. Never "%": the
	 * quantity is a gap between two shares, and printing it as a percentage is the
	 * standard way that kind of number gets misread as a relative change. The unit
	 * is carried once per phrase, not once per endpoint — "[+10.7 pts, +22.0 pts]"
	 * is the same interval read twice as slowly.
	 */
	function signed(value: number): string {
		return `${value >= 0 ? '+' : '−'}${Math.abs(value * 100).toFixed(1)}`;
	}
	function pts(value: number): string {
		return `${signed(value)} pts`;
	}
	function band(low: number, high: number): string {
		return `[${signed(low)}, ${signed(high)}] pts`;
	}
	/**
	 * A budget that cuts a block of tied scores catches its errors pro rata, so a
	 * swept count can be fractional. Print the fraction where there is one rather
	 * than rounding a 276.5 to a 276 the artifact never claimed.
	 */
	function count(value: number): string {
		return Number.isInteger(value) ? String(value) : value.toFixed(1);
	}

	// ---- data ---------------------------------------------------------------
	const queue = $derived(data.status === 'ok' ? data.queue : null);
	const ordered = $derived<ReviewQueueArm[]>(queue ? reviewQueueDisplayOrder(queue) : []);
	/** The shared scale: every bar measured against the longest queue on the panel. */
	const maxQueue = $derived(
		ordered.reduce((widest, arm) => Math.max(widest, arm.operatingPoint.queue), 0)
	);
	const perStatement = $derived(maxQueue > 0 ? BAR_SPAN / maxQueue : 0);
	const tickStep = $derived(
		TICK_STEPS.find((step) => maxQueue / step <= 4) ?? TICK_STEPS[TICK_STEPS.length - 1]
	);
	const ticks = $derived.by(() => {
		const out: number[] = [];
		for (let value = 0; value <= maxQueue; value += tickStep) out.push(value);
		return out;
	});

	const rows = $derived(
		ordered.map((arm, index) => {
			const point = arm.operatingPoint;
			const top = ROW_TOP + ROW_PITCH * index;
			const caughtW = point.trueErrorsCaught * perStatement;
			const totalW = point.queue * perStatement;
			return {
				arm,
				top,
				midY: top + BAR_H / 2,
				caughtW,
				falseW: totalW - caughtW,
				falseX: BAR_LEFT + caughtW,
				labelX: BAR_LEFT + totalW + 8,
				color: reviewQueueColorVar(arm)
			};
		})
	);

	/**
	 * The axis sits below the LAST bar, and the viewBox below the axis — both
	 * measured from the rows actually drawn. A panel that gains an arm gets
	 * taller; it never loses one off the bottom edge.
	 */
	const axisY = $derived(ROW_TOP + ROW_PITCH * Math.max(rows.length - 1, 0) + BAR_H + AXIS_GAP);
	const chartHeight = $derived(axisY + AXIS_FOOT);

	// ---- callout + caption numbers, all derived ------------------------------
	const shortest = $derived(rows[0]?.arm ?? null);
	const longest = $derived(rows[rows.length - 1]?.arm ?? null);

	// ---- the budget sweep ----------------------------------------------------
	// The pair the callout and the two brackets name. Derived from the artifact
	// (shortest whole-flag-set zero pile, strongest paper comparator), never
	// hard-coded, so a change of arms moves the claim with the data.
	const yieldPair = $derived(queue ? reviewQueueEqualYieldPair(queue) : null);
	const sweepMeta = $derived(queue?.equalYield.budgetSweep ?? null);
	/**
	 * `buildReviewQueueSweep` THROWS on drift the validator cannot see — an arm
	 * missing from the sweep, a degenerate axis, or a display name that would
	 * overrun its rail. Gate the panel on it rather than draw a clipped label.
	 */
	const sweep = $derived.by(() => {
		if (!queue) return null;
		try {
			return { figure: buildReviewQueueSweep(queue), reason: null };
		} catch (error) {
			return { figure: null, reason: String(error instanceof Error ? error.message : error) };
		}
	});
	const fig = $derived(sweep?.figure ?? null);
	/**
	 * Annotation placement, in regions the geometry guarantees are empty. Both
	 * curves rise monotonically, so everything BELOW the lower curve to the right
	 * of a point is clear: the equal-yield label sits under the reference-catch
	 * level just past where the comparator crosses it (the comparator is above
	 * that level from there on), and the shortfall label sits under the
	 * comparator's own value at the reference budget. Neither is right-anchored,
	 * so neither can lose leading glyphs; both are haloed against the axis rules
	 * and the dashed budget marker they pass over.
	 */
	const yieldLabel = $derived(
		fig ? { x: r2(fig.equalYield.x + 6), y: r2(fig.equalYield.y + 13) } : null
	);
	const shortfallLabel = $derived(
		fig ? { x: r2(fig.marker.x + 6), y: r2(fig.marker.yComparator + 14) } : null
	);
	/** Frozen artifact join key -> the on-screen name, for the method note. */
	function displayOf(name: string): string {
		return queue?.arms.find((arm) => arm.name === name)?.display ?? name;
	}
	function provenanceOf(name: string): string {
		const arm = queue?.arms.find((entry) => entry.name === name);
		return arm ? REVIEW_QUEUE_PROVENANCE_GLOSS[arm.provenance] : '';
	}

	/**
	 * Is the operational gap real? Read whole from the artifact — the delta, the
	 * band corrected simultaneously across all four reading arms, and the same number on the
	 * adjudication-safe panel. Nothing here is computed in the component and
	 * nothing is hard-coded; a missing block gates the line rather than printing a
	 * plausible interval.
	 */
	const robust = $derived(queue ? reviewQueueRobustnessHeadline(queue) : null);

	// Which evidence-gated readings actually beat the strongest belief model, and which do
	// not. The bar is the SHORTEST belief-model queue on the panel — the hardest
	// one to beat, derived rather than named, so nothing here depends on which arm
	// happens to be strongest today. The caption states a dominance claim, and an
	// arm that loses must be able to falsify it from the artifact instead of being
	// quietly left out of the figure.
	const hardestModel = $derived(
		(queue?.arms ?? [])
			.filter((a) => a.kind === 'paper-model')
			.reduce<ReviewQueueArm | null>(
				(best, arm) =>
					best === null || arm.operatingPoint.queue < best.operatingPoint.queue ? arm : best,
				null
			)
	);
	const gateLosers = $derived(
		hardestModel
			? (queue?.arms ?? []).filter(
					(a) => a.kind === 'llm-gate' && a.operatingPoint.queue > hardestModel.operatingPoint.queue
				)
			: []
	);
	const gateWinners = $derived(
		hardestModel
			? (queue?.arms ?? []).filter(
					(a) =>
						a.kind === 'llm-gate' && a.operatingPoint.queue <= hardestModel.operatingPoint.queue
				)
			: []
	);
</script>

<section class="review-queue" aria-labelledby="review-queue-title">
	{#if data.status !== 'ok' || queue === null}
		<div class="gate" role="status">
			<p class="eyebrow">what a curator would have to read</p>
			<h2 id="review-queue-title">The list of statements to review is unavailable</h2>
			<p>{data.status === 'ok' ? 'the review-queue payload is missing.' : data.reason}</p>
			<code>{data.artifact_path}</code>
		</div>
	{:else}
		<!--
			`.sr`, not `.visually-hidden`: this heading is a verbatim duplicate of the
			figure's own drawn title, present only so the landmark has an accessible
			name. Counting it against the page's visible-prose budget would charge the
			page twice for one line and buy the reader nothing.
		-->
		<h2 id="review-queue-title" class="sr">What a curator would have to read</h2>
		<figure>
			<div class="scroller">
				<svg
					viewBox="0 0 900 {chartHeight}"
					preserveAspectRatio="xMidYMid meet"
					role="img"
					aria-labelledby="review-queue-chart-title review-queue-chart-desc"
				>
					<title id="review-queue-chart-title"
						>How many statements each belief model sends a curator, when every model has to find the
						same share of the known errors</title
					>
					<desc id="review-queue-chart-desc"
						>{rows.length} horizontal bars, one per model, on one shared count scale. Each bar is the
						number of the {queue.panel.n} statements that model would flag as wrong once its score
						cutoff is lowered far enough to catch at least {whole(queue.headlineTargetRecall)} of the
						{queue.panel.nErrors} errors we already know about, split into the errors actually caught
						and the correct statements flagged alongside them.
						{#each rows as row (row.arm.name)}{row.arm.display}: {row.arm.operatingPoint.queue} statements
							to read, {row.arm.operatingPoint.trueErrorsCaught} of them real errors and {row.arm
								.operatingPoint.falseAlarms} false alarms, so {pct(row.arm.operatingPoint.precision)} of
							what it flags is a real error.
						{/each}
						{#if shortest && longest}The shortest list belongs to {shortest.display} and the longest to
							{longest.display}; the shorter bars also carry the longer caught segments.{/if}</desc
					>

					<defs>
						<!-- False alarms read as texture, not as a second solid colour. -->
						<pattern
							id="review-queue-hatch"
							width="5"
							height="5"
							patternUnits="userSpaceOnUse"
							patternTransform="rotate(45)"
						>
							<rect width="5" height="5" fill="var(--paper)" />
							<line x1="0" y1="0" x2="0" y2="5" stroke="var(--ink-faint)" stroke-width="1.8" />
						</pattern>
					</defs>

					<!-- title block (y = 0..48) -->
					<text class="fig-title" x="12" y="20">What a curator would have to read</text>
					<!-- 9.5px serif, start-anchored at x=12 inside a 900-unit viewBox: the
					     longest form of this line is ~72 characters, well under the rail. -->
					<text class="fig-subtitle" x="12" y="37"
						>each model's score cutoff lowered until it flags at least
						{whole(queue.headlineTargetRecall)} of the {queue.panel.nErrors} known errors</text
					>

					<!-- the only legend: two inline swatches above the first bar -->
					<rect class="swatch" x={BAR_LEFT} y={LEGEND_Y - 7} width="8" height="8" fill="var(--ink)" />
					<text class="legend" x={BAR_LEFT + 12} y={LEGEND_Y}>errors caught</text>
					<rect
						class="swatch outline"
						x={BAR_LEFT + 128}
						y={LEGEND_Y - 7}
						width="8"
						height="8"
						fill="url(#review-queue-hatch)"
					/>
					<text class="legend" x={BAR_LEFT + 140} y={LEGEND_Y}>false alarms</text>

					{#each rows as row (row.arm.name)}
						<text class="arm" x={GUTTER_RIGHT - 4} y={r2(row.midY) + 3} fill={row.color}
							>{row.arm.display}</text
						>
						<rect
							class="caught"
							x={BAR_LEFT}
							y={row.top}
							width={r2(row.caughtW)}
							height={BAR_H}
							fill="var(--ink)"
						/>
						<rect
							class="wasted"
							x={r2(row.falseX)}
							y={row.top}
							width={r2(row.falseW)}
							height={BAR_H}
							fill="url(#review-queue-hatch)"
						/>
						<text class="counts" x={r2(row.labelX)} y={r2(row.midY) + 3}
							><tspan class="n-caught">{row.arm.operatingPoint.trueErrorsCaught}</tspan> caught ·
							<tspan class="n-wasted">{row.arm.operatingPoint.falseAlarms}</tspan> false</text
						>
					{/each}

					<!-- shared count axis: one scale, no gridlines -->
					<line class="axis" x1={BAR_LEFT} y1={axisY} x2={BAR_RIGHT_MAX} y2={axisY} />
					{#each ticks as tick (tick)}
						<line
							class="tick-mark"
							x1={r2(BAR_LEFT + tick * perStatement)}
							y1={axisY}
							x2={r2(BAR_LEFT + tick * perStatement)}
							y2={axisY + 4}
						/>
						<text class="tick" x={r2(BAR_LEFT + tick * perStatement)} y={axisY + 14}>{tick}</text>
					{/each}
					<text class="axis-title" x={BAR_LEFT} y={axisY + 30}
						>statements sent to a human &#8594;</text
					>
				</svg>
			</div>

			{#if fig && yieldPair && sweepMeta && yieldLabel && shortfallLabel}
				<!--
					The claim, in the order it has to be read: the gate's point is untuned,
					the comparator's is fitted here, and the fitted one still loses. The
					two words carrying the asymmetry ("no threshold", "fitted on this very
					panel") are the ones that must survive any future edit.
				-->
				<p class="callout">
					<b>{yieldPair.referenceArm.display}</b> set no cutoff at all: the
					{yieldPair.reference.budget}
					statements whose evidence it rejected outright already hold
					{yieldPair.reference.trueErrorsCaught} of the {queue.panel.nErrors} errors.
					<b>{yieldPair.comparatorArm.display}</b> has to be read down to
					{yieldPair.comparator.budgetForEqualYield} statements to find the same
					{yieldPair.reference.trueErrorsCaught} — {Math.abs(yieldPair.comparator.extraReviews)}
					{yieldPair.comparator.extraReviews >= 0 ? 'more' : 'fewer'} — and its cutoff was chosen on
					these very statements with the answers already in hand. The advantage was handed to the
					other side.
				</p>
				{#if robust}
					<!--
						The operational result's own uncertainty, at a budget neither arm
						chose (a fixed share of the panel), simultaneous over all four
						gates, and repeated on the adjudication-safe panel — where it moves
						the OPPOSITE way to the ranking margin. Every number is read from
						the artifact; a missing block hides the line.
					-->
					<p class="robust">
						<b>{pts(robust.primary.delta)}</b> more of the errors found when both are given the same
						{whole(robust.robustness.budgetShare)} of the statements to read ·
						{whole(1 - robust.robustness.familyAlpha)} range, widened to cover all
						{robust.robustness.family.length} reading models
						{band(robust.primary.simultaneousLow, robust.primary.simultaneousHigh)} ·
						{pts(robust.sensitivity.delta)} on the stricter label set, which drops the statements
						whose review was incomplete
					</p>
				{/if}

				<div class="scroller">
					<!--
						Same panel, yield fixed instead of threshold. Everything drawn here is
						placed by `buildReviewQueueSweep`, which throws rather than clip; the
						two brackets are the two numbers the callout above states.
					-->
					<svg
						viewBox="0 0 {fig.geometry.width} {fig.geometry.height}"
						preserveAspectRatio="xMidYMid meet"
						role="img"
						aria-labelledby="review-queue-sweep-title review-queue-sweep-desc"
					>
						<title id="review-queue-sweep-title"
							>Errors found, against how many statements are read: evidence-gated reading — a language
							model drops the evidence it judges unsupported, then INDRA's usual belief formula, a
							noisy-OR over how reliable each source is, scores what remains — against the strongest
							formula-based belief model</title
						>
						<desc id="review-queue-sweep-desc"
							>Two rising curves on one pair of axes. The horizontal axis is the number of statements
							a curator reads, from none to all {queue.panel.n}; the vertical axis is how many of the
							{queue.panel.nErrors} known errors that reading finds, taking each model's lowest-scoring
							statements first. {fig.reference.label} rises faster than {fig.comparator.label} over most
							of the range, and the shaded region between them is the gain. After
							{fig.marker.budget} statements — evidence-gated reading's own operating point, the ones
							whose evidence it rejected outright — it has found {fig.marker.caught} errors and {fig
								.comparator.label} has found {count(fig.marker.comparatorCaught)}, a gap of {count(
								fig.marker.advantage
							)}. {fig.comparator.label} does not reach {fig.marker.caught} errors until
							{fig.equalYield.budget} statements have been read, {fig.equalYield.extraReviews} more than
							evidence-gated reading needed. That point is found by choosing the comparator's cutoff on
							these very statements with the answers already in hand — help it would not have before
							the curation existed, and help that favours it. Evidence-gated reading is given no cutoff
							at all. The gain is not uniform: it is negative below {sweepMeta.firstPositiveBudget}
							statements read, peaks at {sweepMeta.peakBudget}, and decays back to nothing once
							everything has been read.{#if robust}
								When both are given the same {robust.robustness.primary.budget} statements to read — a
								{whole(robust.robustness.budgetShare)} share of the benchmark, chosen by neither side —
								evidence-gated reading finds {pts(robust.primary.delta)} more of the errors than the
								comparator, with a {whole(1 - robust.robustness.familyAlpha)} range, widened to cover
								all {robust.robustness.family.length} reading models at once, of {band(
									robust.primary.simultaneousLow,
									robust.primary.simultaneousHigh
								)} — a range that stays above zero. On the stricter label set, which drops the
								statements whose evidence review was incomplete, the same gap is {pts(
									robust.sensitivity.delta
								)}.{/if}</desc
						>

						<text class="fig-title" x="12" y="20">Errors found, against how much is read</text>

						<!-- Both signs of the gap are drawn. The budgets where the gate LOSES
						     are the whole reason this figure sweeps rather than points. -->
						{#each fig.deficitBands as band, index (index)}
							<path class="band" d={band} fill={fig.comparator.color} />
						{/each}
						{#each fig.leadBands as band, index (index)}
							<path class="band" d={band} fill={fig.reference.color} />
						{/each}

						<line
							class="axis"
							x1={fig.geometry.plotLeft}
							y1={fig.geometry.plotBottom}
							x2={fig.geometry.plotRight}
							y2={fig.geometry.plotBottom}
						/>
						{#each fig.xTicks as tick (tick.value)}
							<line
								class="tick-mark"
								x1={tick.x}
								y1={fig.geometry.plotBottom}
								x2={tick.x}
								y2={fig.geometry.plotBottom + 4}
							/>
							<text class="tick" x={tick.x} y={fig.geometry.plotBottom + 14}>{tick.value}</text>
						{/each}
						{#each fig.yTicks as tick (tick.value)}
							<text class="tick anchor-end" x={fig.geometry.plotLeft - 6} y={tick.y + 3}
								>{tick.value}</text
							>
						{/each}

						<!--
							Evidence-gated reading's own operating point, marked on the axis it
							belongs to, and named for what it is. "no cutoff" is two words of
							visible prose and they are the cheapest way to make the asymmetry
							legible to someone who never opens the method note.
						-->
						<line
							class="marker"
							x1={fig.marker.x}
							y1={fig.geometry.plotBottom}
							x2={fig.marker.x}
							y2={fig.marker.yReference}
						/>
						<text class="tick strong" x={fig.marker.x} y={fig.geometry.plotBottom + 14}
							>{fig.marker.budget}</text
						>
						<text class="tick strong" x={fig.marker.x} y={fig.geometry.plotBottom + 24}
							>no cutoff</text
						>

						<polyline
							class="series dashed"
							points={fig.comparator.polyline}
							stroke={fig.comparator.color}
						/>
						<polyline class="series" points={fig.reference.polyline} stroke={fig.reference.color} />
						<text
							class="series-label"
							x={fig.geometry.labelX}
							y={fig.reference.labelY}
							fill={fig.reference.color}>{fig.reference.label}</text
						>
						<text
							class="series-label"
							x={fig.geometry.labelX}
							y={fig.comparator.labelY}
							fill={fig.comparator.color}>{fig.comparator.label}</text
						>

						<!-- same catch, further right -->
						<line
							class="bracket"
							x1={fig.marker.x}
							y1={fig.equalYield.y}
							x2={fig.equalYield.x}
							y2={fig.equalYield.y}
						/>
						<line
							class="bracket"
							x1={fig.equalYield.x}
							y1={fig.equalYield.y - 4}
							x2={fig.equalYield.x}
							y2={fig.equalYield.y + 4}
						/>
						<!-- Signed both ways: the artifact permits a comparator that WINS here,
						     and this label must never read "+-40 reviews" if one ever does.
						     "cutoff tuned on the answers" is the other half of the asymmetry the
						     marker names — this point exists only because the comparator's cutoff
						     was chosen on these statements, with these labels already in hand.
						     9px mono = 5.4186 u/char, left-anchored at x≈308: the longest form
						     of this label is 55 ch = 298 u, ending at ~606, inside plotRight 700. -->
						<text class="note" x={yieldLabel.x} y={yieldLabel.y}
							>cutoff tuned on the answers: {fig.equalYield.extraReviews >= 0 ? '+' : '−'}{Math.abs(
								fig.equalYield.extraReviews
							)} more for the same {fig.marker.caught}</text
						>

						<!-- same budget, lower down -->
						<line
							class="bracket"
							x1={fig.marker.x}
							y1={fig.marker.yComparator}
							x2={fig.marker.x}
							y2={fig.marker.yReference}
						/>
						<text class="note" x={shortfallLabel.x} y={shortfallLabel.y}
							>{count(Math.abs(fig.marker.advantage))}
							{fig.marker.advantage >= 0 ? 'fewer' : 'more'} errors for the same reading</text
						>

						<text class="axis-title" x={fig.geometry.plotLeft} y={fig.geometry.plotBottom + 30}
							>statements reviewed &#8594;</text
						>
						<text
							class="axis-title"
							transform="rotate(-90 12 {fig.geometry.plotTop + 46})"
							x="12"
							y={fig.geometry.plotTop + 46}>errors found</text
						>
					</svg>
				</div>
			{:else}
				<!-- Fail-closed: no fallback number, no partial figure, just the reason. -->
				<p class="gate-line">
					the comparison at a fixed amount of reading is unavailable — <code
						>{sweep?.reason ?? 'the artifact carries no equal-yield block.'}</code
					>
				</p>
			{/if}

			<figcaption>
				<!--
					The two sentences this caption used to open with ("solid is caught, hatched
					is flagged anyway" and "less to read and more found, on both axes at once")
					are now carried by the legend swatches and by the sweep above, so they were
					cut rather than restated. The counter-example stays on the page: a
					concession that only appears behind a <details> is not a concession.
				-->
				<p>
					{gateWinners.length} of the {gateWinners.length + gateLosers.length} reading models draw a
					shorter bar <em>and</em> a longer solid segment than every model that scores from evidence
					counts alone — less for a curator to read, and more of it real errors.{#if gateLosers.length && hardestModel}{' '}
						{gateLosers.map((a) => a.display).join(', ')}
						{gateLosers.length === 1 ? 'is the exception' : 'are the exceptions'}:
						{gateLosers.length === 1 ? 'it sends' : 'they send'} a curator <em>more</em> to read than
						{hardestModel.display} ({gateLosers
							.map((a) => a.operatingPoint.queue.toLocaleString())
							.join(', ')} vs {hardestModel.operatingPoint.queue.toLocaleString()} statements), and a
						smaller share of it is real errors.{/if}
				</p>
				{#if sweepMeta}
					<p>
						How much is read decides this: the advantage is negative below {sweepMeta.firstPositiveBudget}
						statements, widest at {sweepMeta.peakBudget}{#if sweepMeta.halfPeakDecayBudget !== null},
							and halved by {sweepMeta.halfPeakDecayBudget}{/if}.
					</p>
				{/if}

				<details class="method">
					<summary>how this is computed</summary>
					<p>
						Cutoff: we lower each model's score cutoff until it has flagged
						{whole(queue.headlineTargetRecall)} of the errors we already know about, and report what
						that costs. Scores come in steps, so a model that cannot land exactly on that target
						overshoots it — which is why the models differ in how many errors they catch as well as in
						how much reading they ask for. The 2023 paper published no cutoff and no decision measure
						at all; these operating points are chosen here, on the same statements they are scored
						on.
					</p>
					<p>
						What counts as flagging a statement: {queue.prose.decisionRule.plain} How the cutoff is
						reached: {queue.prose.thresholdRule.plain}
					</p>
					<p>
						The statements are the {queue.panel.n} assembled statements carrying
						<code>{queue.panel.label}</code>, of which {queue.panel.nErrors} ({pct(
							queue.panel.errorBaseRate
						)}) are errors. Generated by <code>{queue.generatedBy}</code>.
					</p>
					{#each queue.prose.caveats as caveat (caveat.shipped)}
						<p>{caveat.plain}</p>
					{/each}
					{#if yieldPair}
						<p>
							Reading order: a curator with time for B statements reads the B lowest-scoring ones.
							Where B lands inside a block of statements that all share one score, the model cannot
							say which of them to read first, so that block contributes its errors in proportion and
							the count is the number a curator would find on average.
						</p>
						<p>{queue.prose.operatingRule.plain}</p>
						<p>
							Each of the other models is then given the cutoff that lands it on exactly the same
							error count as {yieldPair.referenceArm.display} — a cutoff chosen on these very
							statements with the answers in hand, which is help
							{yieldPair.referenceArm.display} is not given.
						</p>
						<p>{queue.prose.oracleDisclosure.plain}</p>
						<p>
							Every belief model against {yieldPair.referenceArm.display} at its own untuned point ({yieldPair
								.reference.budget} statements read, {yieldPair.reference.trueErrorsCaught} errors found,
							{pct(yieldPair.reference.precision)} of what it flagged being real errors) — how much each
							must have read for the same catch, then what each finds having read that same amount:
						</p>
						<ul class="grain">
							{#each yieldPair.reference.comparators as comparator (comparator.arm)}
								<li>
									<b>{displayOf(comparator.arm)}</b> — {comparator.budgetForEqualYield} statements read ({comparator.extraReviews >
									0
										? '+'
										: ''}{comparator.extraReviews}) for the same
									{yieldPair.reference.trueErrorsCaught}, with {pct(comparator.precisionAtEqualYield)} of
									what it flagged being real errors;
									{count(comparator.errorsCaughtAtReferenceBudget)} found in the first {yieldPair
										.reference.budget}
								</li>
							{/each}
						</ul>
					{/if}
					{#if robust}
						<p>
							{robust.robustness.prose.metric.plain}
							{robust.robustness.prose.budgetRule.plain} The comparator is
							{robust.reference.display}, the one model here the 2023 paper actually published.
							{robust.robustness.prose.bootstrapDesign.plain}
						</p>
						<p>
							Because four reading models are tested at once, one of them can look good by chance;
							the ranges below are widened so that they are all right together
							{whole(1 - robust.robustness.familyAlpha)} of the time, not each separately.
							{robust.robustness.prose.multiplicityMethod.plain} Combined error rate
							{robust.robustness.familyAlpha} across the {robust.robustness.family.length} reading models.
							{robust.robustness.prose.multiplicityNote.plain}
						</p>
						<p>
							The amount read here ({robust.robustness.primary.budget} statements,
							{whole(robust.robustness.budgetShare)} of them all) is NOT evidence-gated reading's own
							operating point of {yieldPair?.reference.budget ?? '—'}: a fixed share is an amount
							neither side chose, and evidence-gated reading's own point is exactly where its
							advantage peaks. The range is quoted at the neutral amount, and the drawn curve passes
							through it.
						</p>
						{#each [robust.robustness.primary, robust.robustness.sensitivity] as side (side.id)}
							<p>
								<b>{side.role === 'primary' ? 'Main' : 'Second look, stricter labels'}</b> —
								<code>{side.id}</code>,
								{side.nStatements} statements, {side.nErrors} errors, {side.budget} of them read,
								{displayOf(robust.robustness.referenceArm)} finds {pct(side.referenceErrorRecall)} of
								them; covering all four at once costs {side.maxTCriticalValue.toFixed(3)} standard
								errors, against {robust.robustness.pointwiseNormalCriticalValue.toFixed(3)} with no
								correction at all and {side.bonferroniCriticalValue.toFixed(3)} for the blunt
								Bonferroni version.
							</p>
							<ul class="grain">
								{#each robust.robustness.family as name (name)}
									<li>
										<b>{displayOf(name)}</b> — {pct(side.arms[name].errorRecall)} of the errors,
										{pts(side.arms[name].delta)}; on its own
										{band(side.arms[name].ci95Low, side.arms[name].ci95High)}, widened to cover all four
										{band(side.arms[name].simultaneousLow, side.arms[name].simultaneousHigh)}
										{WIDENED_BAND_CLAUSE[side.arms[name].simultaneousStanding]};
										{side.arms[name].nValidResamples.toLocaleString()} resamples
									</li>
								{/each}
							</ul>
						{/each}
						<p>{robust.robustness.prose.labelCompletenessNote.plain}</p>
					{/if}
					<p>
						How many genuinely different lists each model can produce across the {queue.targetRecalls
							.length} targets — a model that can only produce one has a single operating point, so its
						results at the lower targets are the same statements counted again — and whose model each
						one is:
					</p>
					<ul class="grain">
						{#each ordered as arm (arm.name)}
							<li>
								<b>{arm.display}</b> — {arm.distinctQueueSizesAcrossTargets};
								{provenanceOf(arm.name)}
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
	.review-queue {
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
		min-width: 640px;
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
	.legend {
		fill: var(--ink-faint);
		font-family: var(--mono);
		font-size: 7px;
		letter-spacing: 0.06em;
		text-transform: uppercase;
	}
	.swatch.outline {
		stroke: var(--ink-faint);
		stroke-width: 0.6;
	}
	.arm {
		font-family: var(--mono);
		font-size: 9px;
		text-anchor: end;
	}
	.caught,
	.wasted {
		shape-rendering: crispEdges;
	}
	.wasted {
		stroke: var(--ink-faint);
		stroke-width: 0.6;
	}
	.counts {
		fill: var(--ink-muted);
		font-family: var(--mono);
		font-size: 9px;
		font-variant-numeric: tabular-nums;
	}
	.counts .n-caught {
		fill: var(--ink);
		font-weight: 600;
	}
	.counts .n-wasted {
		fill: var(--ink-muted);
	}
	.axis {
		stroke: var(--rule);
		stroke-width: 1;
	}
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
	/* ---- budget sweep ---- */
	/* The gain, drawn rather than asserted: it opens, peaks and closes on its own. */
	/*
	 * The gap, tinted with the hue of whichever arm is ahead — fill comes from the
	 * series colour in the markup, so the two bands follow the page's arm hues
	 * rather than restating them. Same opacity both ways: the budgets where the
	 * gate LOSES are drawn at full strength, not de-emphasised.
	 */
	.band {
		fill-opacity: 0.14;
		stroke: none;
	}
	.series {
		fill: none;
		stroke-width: 1.6;
		stroke-linejoin: round;
	}
	/* Dashed, so the pair survives greyscale and colour-vision deficiency. */
	.series.dashed {
		stroke-dasharray: 5 3;
	}
	.series-label {
		font-family: var(--mono);
		font-size: 9px;
	}
	.tick.anchor-end {
		text-anchor: end;
	}
	.tick.strong {
		fill: var(--ink);
		font-weight: 600;
	}
	.marker {
		stroke: var(--ink-faint);
		stroke-width: 1;
		stroke-dasharray: 2 3;
	}
	/* Neutral on purpose: a bracket measures the gap, it does not take a side. */
	.bracket {
		stroke: var(--ink);
		stroke-width: 1.2;
	}
	/* Haloed against the rules and the dashed budget marker the notes cross. */
	.note {
		fill: var(--ink);
		font-family: var(--mono);
		font-size: 9px;
		font-variant-numeric: tabular-nums;
		paint-order: stroke;
		stroke: var(--paper);
		stroke-width: 3;
		stroke-linejoin: round;
	}
	.gate-line {
		margin: 0.85rem 0 0;
		font-family: var(--mono);
		font-size: 0.7rem;
		color: var(--ink-faint);
	}
	.callout {
		margin: 0.85rem 0 0;
		padding: 0.6rem 0.75rem;
		border: 1px solid var(--rule);
		border-left: 3px solid var(--accent);
		background: color-mix(in srgb, var(--ink) 3%, transparent);
		font-family: var(--serif);
		font-size: 0.85rem;
		line-height: 1.5;
		color: var(--ink-muted);
		max-width: 74ch;
	}
	.callout b {
		color: var(--ink);
		font-weight: 600;
	}
	/*
	 * The interval, in the mono voice the rest of the page uses for measured
	 * quantities — it reads as a readout, not as another sentence, so the callout
	 * above keeps the argument and this keeps the evidence.
	 */
	.robust {
		margin: 0.4rem 0 0;
		padding-left: 0.75rem;
		border-left: 3px solid transparent;
		font-family: var(--mono);
		font-size: 0.7rem;
		line-height: 1.6;
		color: var(--ink-muted);
		max-width: 74ch;
	}
	.robust b {
		color: var(--ink);
		font-weight: 600;
		font-variant-numeric: tabular-nums;
	}
	/*
	 * Screen-reader-only. Same rule DeployedBaseline uses, and the same reason:
	 * a heading that duplicates the drawn title verbatim.
	 */
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
	figcaption {
		margin-top: 0.7rem;
		max-width: 74ch;
	}
	figcaption p,
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
	/* `.method` nests inside `figcaption`, so this covers the collapsed note too. */
	figcaption em {
		font-style: italic;
		color: var(--ink);
	}
	/* The full method note: one click, nothing hidden, nothing lost. */
	.method {
		margin-top: 0.35rem;
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
