<!--
  PaperReliabilityStrip — the limits panel: ranking better is not being right
  about the odds.

  Each arm cell LEADS with the logistic-recalibration diagnostic
  `slope S · intercept I` against the ideal 1.00 / 0.00 — the sharp
  miscalibration signal. The caption's slope range and the "how many times too
  extreme" multiple are DERIVED from those same server-computed slopes, never
  typed in. Beneath it, the calibration small-multiple: predicted probability (x)
  against observed released-correct rate (y) over the y=x diagonal, each occupied
  reliability bin a mark sized by count and connected left→right, so a model's
  trajectory off the diagonal reads as one line. ECE stays as a demoted secondary
  number, and a compact per-arm BrierBar (reused unmodified) splits realized error
  into floor / miscalibration penalty / discrimination credit. Reuses
  ReliabilityDiagram's px/py/rOf/pathOf idiom and the shared paperArmColorVar hue.
  Consumes the server-computed slope/intercept/reliabilityBins/ece/Brier verbatim —
  recomputes nothing. Calibration is a LIMITATION axis; AP stays the ranking lens.

  UNMEASURED IS NOT PERFECT. Every calibration scalar arrives nullable, and an arm
  whose predictions failed to join has them all null. It is rendered as an explicit
  unavailable cell — no ECE, no Brier bar. Printing a placeholder here would print
  "ECE 0.000", the IDEAL value, beside an empty diagram: a broken join would read
  as the best-calibrated arm on the page. Never coalesce these to a number.
-->
<script lang="ts">
	import BrierBar from '$lib/components/BrierBar.svelte';
	import { rOf, pathOf, px, py } from '$lib/reliability-geometry';
	import {
		PAPER_LITERAL_REFERENCE_ARM_ID,
		paperArmColorVar,
		type PaperLiteralArm,
		type PaperLiteralLoad,
		type PaperLiteralReliabilityBin
	} from '$lib/data/paper-literal';

	let { data }: { data: PaperLiteralLoad } = $props();

	// Calibration space [0,1]×[0,1], y inverted (1 at top): px/py/rOf/pathOf live in
	// $lib/reliability-geometry (the ReliabilityDiagram idiom, shared).

	// A mark counts as "on the diagonal" within this predicted-vs-observed slack;
	// the n-weighted mean gap decides the (color-free) deviation wording in the desc.
	const HUG_SLACK = 0.02;

	/**
	 * SVG COPY BUDGET for the in-square empty state. The square is a 100-unit
	 * viewBox and the text is centred at x=50, so the budget is the full width:
	 * the empty label renders at font-size 7px in the serif face, whose advance
	 * averages ≈0.48em → ≈3.4 user units per character, giving 100 / 3.4 = 29.4 →
	 * 29 characters before the copy reaches the square's edges. Both strings below
	 * are held under it, and the contract runner re-measures them from this source
	 * so a longer message cannot ship silently.
	 */
	const EMPTY_COPY_BUDGET_CHARS = 29;
	/** 19 chars — the arm joined but produced no occupied bin. */
	const EMPTY_NO_BINS = 'no calibration data';
	/** 24 chars — the arm's prediction vector never joined; nothing was measured. */
	const EMPTY_NO_JOIN = 'predictions did not join';
	/**
	 * Enforcement, not decoration: over-budget copy falls back to the shortest
	 * honest word rather than running past the square's edge (SVG text does not
	 * wrap and `overflow: visible` lets it escape the cell instead of clipping
	 * visibly). The contract runner re-measures the two strings above against the
	 * budget, so this fallback is a backstop that should never fire.
	 */
	function budgeted(copy: string): string {
		return copy.length <= EMPTY_COPY_BUDGET_CHARS ? copy : 'unmeasured';
	}

	interface Point {
		x: number;
		y: number;
		n: number;
		gap: number;
	}

	/** The four Murphy terms, present together or not at all. */
	interface BrierTerms {
		brier: number;
		reliability: number;
		resolution: number;
		uncertainty: number;
	}

	interface ArmView {
		arm: PaperLiteralArm;
		/** Shared arm→color token — paper/port in --accent, every LLM in --blocked. */
		color: string;
		points: Point[];
		path: string;
		hasShape: boolean;
		/**
		 * True when calibration was actually MEASURED on this arm — i.e. the server
		 * joined its predictions and returned a real ECE. False leaves every number
		 * off the cell; it never falls back to a value.
		 */
		measured: boolean;
		/** True when the logistic recalibration was identifiable (slope/intercept non-null). */
		identifiable: boolean;
		slopeLabel: string;
		interceptLabel: string;
		/** null when unmeasured — deliberately not a string like '0.000'. */
		eceLabel: string | null;
		/** null when unmeasured; the BrierBar is not drawn at all in that case. */
		brier: BrierTerms | null;
		emptyCopy: string;
		deviation: string;
		desc: string;
	}

	function toPoints(bins: PaperLiteralReliabilityBin[]): Point[] {
		return bins.map((b) => ({ x: b.p_mean, y: b.y_rate, n: b.n, gap: b.y_rate - b.p_mean }));
	}

	// Shared bin-n scale across every arm so a sparse bin can't read as a confident
	// mark — the same honesty rule ReliabilityDiagram applies across its series.
	const maxBinN = $derived(
		data.status === 'ok'
			? Math.max(1, ...data.arms.flatMap((arm) => arm.reliabilityBins.map((b) => b.n)))
			: 1
	);

	// The count-weighted mean signed gap (observed − predicted) → the deviation word.
	function deviationWord(points: Point[]): string {
		if (points.length === 0) return 'no calibration data';
		const total = points.reduce((sum, p) => sum + p.n, 0);
		const meanGap = points.reduce((sum, p) => sum + p.gap * p.n, 0) / total;
		if (Math.abs(meanGap) < HUG_SLACK) return 'tracks the perfect-calibration diagonal closely';
		return meanGap > 0
			? 'sits above the diagonal — it claims less confidence than it earns'
			: 'sits below the diagonal — it claims more confidence than it earns';
	}

	function fmtPct(v: number): string {
		return `${(v * 100).toFixed(0)}%`;
	}

	/** The four Murphy terms, or null unless every one of them was measured. */
	function brierTermsOf(arm: PaperLiteralArm): BrierTerms | null {
		const { brier, brierReliability, brierResolution, brierUncertainty } = arm;
		if (
			brier === null ||
			brierReliability === null ||
			brierResolution === null ||
			brierUncertainty === null
		) {
			return null;
		}
		return {
			brier,
			reliability: brierReliability,
			resolution: brierResolution,
			uncertainty: brierUncertainty
		};
	}

	function buildView(arm: PaperLiteralArm): ArmView {
		const points = toPoints(arm.reliabilityBins);
		const deviation = deviationWord(points);
		// The null ECE is the "this arm never joined" signal; it stays null all the
		// way to the template, which prints an unavailable state instead of a number.
		const eceLabel = arm.ece === null ? null : arm.ece.toFixed(3);
		const measured = eceLabel !== null && points.length > 0;
		const identifiable = arm.calibrationSlope !== null && arm.calibrationIntercept !== null;
		const slopeLabel = arm.calibrationSlope === null ? '—' : arm.calibrationSlope.toFixed(3);
		const interceptLabel =
			arm.calibrationIntercept === null ? '—' : arm.calibrationIntercept.toFixed(3);
		const calDesc = identifiable
			? `slope ${slopeLabel} and intercept ${interceptLabel} when its scores are refitted to the outcomes, against the ideal 1.000 and 0.000`
			: 'the slope and intercept could not be estimated';
		const desc = measured
			? `${arm.display}: ${calDesc}; average gap between the probability claimed and the share that turned out correct, ${eceLabel}; claimed against observed ${deviation} across ${points.length} band${points.length === 1 ? '' : 's'} of statements.`
			: `${arm.display}: nothing was measured here — this model's predictions did not match up with the statements, so it has no calibration gap and no breakdown of its squared error. It is not a well-calibrated model; it is an unmeasured one.`;
		return {
			arm,
			color: paperArmColorVar(arm.kind),
			points,
			path: pathOf(points),
			hasShape: points.length > 0,
			measured,
			identifiable,
			slopeLabel,
			interceptLabel,
			eceLabel,
			brier: brierTermsOf(arm),
			emptyCopy: budgeted(eceLabel === null ? EMPTY_NO_JOIN : EMPTY_NO_BINS),
			deviation,
			desc
		};
	}

	const arms = $derived<ArmView[]>(data.status === 'ok' ? data.arms.map(buildView) : []);

	/**
	 * Caption numbers, all derived: the LLM-reader slope range (the CoGEx hybrid is
	 * carried under kind 'llm' but is not a reader, and it misses the OTHER way, so
	 * it is named separately) and how many times too extreme those logits are.
	 */
	const readerSlopes = $derived(
		arms
			.filter((view) => view.arm.kind === 'llm' && view.arm.id !== 'indra-cogex-hybrid')
			.map((view) => view.arm.calibrationSlope)
			.filter((slope): slope is number => slope !== null && slope > 0)
			.sort((a, b) => a - b)
	);
	const cogexSlope = $derived(
		arms.find((view) => view.arm.id === 'indra-cogex-hybrid')?.arm.calibrationSlope ?? null
	);
	/**
	 * The paper's own RF, named from the arm set rather than typed into the caption.
	 * It was hard-coded here as a literal — the one place on this page where a
	 * method name was written by hand instead of read from `display`, which is how
	 * a caption drifts away from the arm it describes.
	 */
	const referenceDisplay = $derived(
		arms.find((view) => view.arm.id === PAPER_LITERAL_REFERENCE_ARM_ID)?.arm.display ?? null
	);
	function fmt3(value: number): string {
		return value.toFixed(3);
	}
</script>

{#if data.status !== 'ok'}
	<section class="rs rs-unavailable">
		<p class="rs-note">calibration charts unavailable — {data.reason}</p>
	</section>
{:else}
	<section
		class="rs"
		aria-label="how well each model's stated probabilities match reality, one small chart per model"
	>
		<div class="rs-grid">
			{#each arms as view (view.arm.id)}
				<figure class="rs-cell" style="--hue:{view.color}">
					<figcaption class="rs-cap">
						<span class="rs-label" title={view.arm.display}>{view.arm.display}</span>
						<!-- Unmeasured arms print NO calibration number anywhere in this cell:
						     an ECE of 0.000 is the ideal, so a placeholder would rank a failed
						     join above every arm that joined. -->
						{#if !view.measured}
							<span
								class="rs-cal rs-cal-null"
								title="This model's predictions could not be matched to the statements, so nothing about its calibration was computed. It is unmeasured, not well calibrated."
								>not measured — predictions did not match up</span
							>
							<span class="rs-ideal">no average gap, no error breakdown</span>
						{:else if view.identifiable}
							<span class="rs-cal"
								>slope <strong>{view.slopeLabel}</strong> · intercept
								<strong>{view.interceptLabel}</strong></span
							>
							<span class="rs-ideal">ideal slope 1.000 · intercept 0.000</span>
							<span class="rs-ece">average gap, ECE {view.eceLabel}</span>
						{:else}
							<span class="rs-cal rs-cal-null">slope and intercept could not be estimated</span>
							<span class="rs-ideal">ideal slope 1.000 · intercept 0.000</span>
							<span class="rs-ece">average gap, ECE {view.eceLabel}</span>
						{/if}
					</figcaption>

					<svg
						class="rs-svg"
						viewBox="0 0 100 100"
						role="img"
						aria-labelledby="rs-t-{view.arm.id} rs-d-{view.arm.id}"
						preserveAspectRatio="xMidYMid meet"
					>
						<title id="rs-t-{view.arm.id}"
							>{view.arm.display} — probability claimed against share actually correct</title
						>
						<desc id="rs-d-{view.arm.id}">{view.desc}</desc>

						<!-- perfect-calibration diagonal + 0.5 gridlines -->
						<line class="rs-diag" x1={px(0)} y1={py(0)} x2={px(1)} y2={py(1)} />
						<line class="rs-gridline" x1={px(0.5)} y1={py(0)} x2={px(0.5)} y2={py(1)} />
						<line class="rs-gridline" x1={px(0)} y1={py(0.5)} x2={px(1)} y2={py(0.5)} />

						{#if view.hasShape}
							<!-- the arm's trajectory off the diagonal, then a mark per bin (area ∝ n) -->
							{#if view.path}
								<path class="rs-curve" d={view.path} />
							{/if}
							{#each view.points as p (p.x + ':' + p.y)}
								<circle class="rs-mark" cx={px(p.x)} cy={py(p.y)} r={rOf(p.n, maxBinN)}>
									<title
										>claimed {fmtPct(p.x)} · actually correct {fmtPct(p.y)} · {p.n.toLocaleString(
											'en-US'
										)} statements · {p.gap >= 0 ? 'under' : 'over'}-confident by {fmtPct(
											Math.abs(p.gap)
										)}</title
									>
								</circle>
							{/each}
						{:else}
							<text class="rs-empty" x="50" y="50" text-anchor="middle">{view.emptyCopy}</text>
						{/if}
					</svg>

					<!-- The Brier bar draws only when all four terms were measured; there is
					     no zero-filled bar, because a zero bar reads as no error at all. -->
					{#if view.brier}
						<div class="rs-brier">
							<BrierBar
								reliability={view.brier.reliability}
								resolution={view.brier.resolution}
								uncertainty={view.brier.uncertainty}
								brier={view.brier.brier}
							/>
						</div>
					{/if}
				</figure>
			{/each}
		</div>

		<p class="rs-legend">
			Ranking statements well is not the same as being right about the odds, and this is where the
			two come apart. Each square plots the probability a model claimed against the share of those
			statements that really were correct; a model that means what it says lands on the diagonal.
			Above each square is what happens when you refit that model's own scores to the outcomes: a
			<em>slope</em> of 1.000 and an <em>intercept</em> of 0.000 would mean its probabilities need no
			correction at all.
			{#if readerSlopes.length > 0}
				The language-model readers come out at <em
					>{fmt3(readerSlopes[0])}–{fmt3(readerSlopes[readerSlopes.length - 1])}</em
				>, which means their confidence is stretched
				{(1 / readerSlopes[readerSlopes.length - 1]).toFixed(0)}–{(1 / readerSlopes[0]).toFixed(0)}
				times too far from the middle: their numbers order statements usefully but cannot be read as
				probabilities.
			{/if}
			{#if referenceDisplay}{referenceDisplay} sits close to the ideal{/if}{#if cogexSlope !== null}, and the INDRA CoGEx hybrid misses
				the other way (slope {fmt3(cogexSlope)}) — it hedges when it could commit{/if}. The average
			gap, ECE, is the same story in one number: how far the claimed probability sits from the
			observed rate, averaged over the statements. The bar underneath splits each model's squared
			error into the part no model could avoid, the part it pays for stating the wrong odds, and the
			credit it earns for telling correct and wrong statements apart. All of this is a separate
			question from ranking; average precision remains the view we rank by.
		</p>
	</section>
{/if}

<style>
	.rs {
		margin: 0 0 2rem;
	}
	.rs-grid {
		display: grid;
		grid-template-columns: repeat(auto-fill, minmax(min(100%, 300px), 1fr));
		gap: 1.4rem 1.4rem;
		width: 100%;
	}
	.rs-cell {
		margin: 0;
		min-width: 0;
		display: flex;
		flex-direction: column;
		gap: 0.45rem;
	}
	.rs-svg {
		display: block;
		width: 100%;
		height: auto;
		max-width: 220px;
		margin: 0 auto;
		aspect-ratio: 1 / 1;
		overflow: visible;
	}

	.rs-diag {
		stroke: var(--ink-faint);
		stroke-width: 0.5;
		stroke-dasharray: 2 2;
	}
	.rs-gridline {
		/* the 0.5 calibration gridlines, distinct from the .rs-grid layout grid */
		stroke: var(--rule);
		stroke-width: 0.4;
	}
	.rs-curve {
		fill: none;
		stroke: var(--hue);
		stroke-width: 1.1;
		stroke-opacity: 0.85;
		stroke-linejoin: round;
		stroke-linecap: round;
	}
	.rs-mark {
		fill: var(--hue);
		fill-opacity: 0.7;
		stroke: var(--paper);
		stroke-width: 0.6;
	}
	.rs-empty {
		fill: var(--ink-faint);
		font-family: var(--serif);
		font-style: italic;
		font-size: 7px;
	}

	.rs-cap {
		display: flex;
		flex-direction: column;
		gap: 0.1rem;
		font-family: var(--mono);
		min-width: 0;
	}
	.rs-label {
		color: var(--ink);
		font-size: 0.76rem;
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}
	/* The lead calibration number: slope · intercept vs the ideal reference. */
	.rs-cal {
		color: var(--ink);
		font-size: 0.82rem;
		font-variant-numeric: tabular-nums;
		line-height: 1.25;
	}
	.rs-cal strong {
		font-weight: 700;
	}
	.rs-cal-null {
		color: var(--ink-muted);
		font-style: italic;
	}
	.rs-ideal {
		color: var(--ink-faint);
		font-size: 0.66rem;
		font-variant-numeric: tabular-nums;
	}
	/* ECE demoted to a secondary line beneath the slope/intercept lead. */
	.rs-ece {
		color: var(--ink-muted);
		font-size: 0.7rem;
		font-variant-numeric: tabular-nums;
	}

	/* Compact per-arm Brier bar; scrolls inside its own box on very narrow cells. */
	.rs-brier {
		overflow-x: auto;
		margin-top: 0.1rem;
	}

	.rs-legend {
		margin: 1rem 0 0;
		max-width: 46rem;
		font-family: var(--serif);
		font-style: italic;
		font-size: 0.82rem;
		line-height: 1.5;
		color: var(--ink-muted);
	}
	.rs-legend em {
		color: var(--ink);
		font-style: italic;
	}

	.rs-unavailable {
		padding: 0.4rem 0;
	}
	.rs-note {
		margin: 0;
		font-family: var(--mono);
		font-size: 0.76rem;
		color: var(--ink-faint);
	}
</style>
