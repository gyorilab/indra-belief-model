<!--
  TieInflation — why the paper's own metric flatters OUR arms.

  Sits directly beneath the paper's-own-metric comparison and argues against the
  best-looking number on this page. Two figures:

  1. THE MECHANISM, from real data. Zoomed on the featured reader's dominant tied
     block, over the same recall window for both arms: the achievable curve STEPS
     (no threshold separates tied statements, so the whole block is admitted at
     once), while the trapezoid draws a straight chord. The shaded triangle
     between them is the interpolated area — literally the quantity the shipped
     `trapezoidal_minus_ap_inflation` measures. The paper's own RF is drawn in the
     same window: at 1,546 distinct scores it takes hundreds of tiny steps and its
     chord lies on its own staircase.
  2. THE PER-ARM RELATIONSHIP. Shipped `trapezoidal_minus_ap_inflation` against
     shipped `distinct_scores`, with our whole average-precision margin drawn as a
     reference line so the reader can see that the inflation is the size of the
     margin.

  NOTHING HERE IS A SCHEMATIC. Every vertex is a real operating point from the
  same aligned score vectors the shipped metrics were computed from, and the
  server loader refuses to emit geometry whose triangles do not reconstruct the
  shipped trapezoid-minus-AP gap (TIE_RECONCILIATION_TOLERANCE).

  SHIPPED FIELDS ARE READ, NEVER RECOMPUTED: `trapezoidal_minus_ap_inflation`,
  `distinct_scores`, and both paired deltas come off the artifact verbatim.

  ── SVG LABEL BUDGET (measured; right-anchored text clips leading glyphs
  silently, and a <desc> does not save it) ────────────────────────────────────
  At font-size 9px the mono advance is 5.4186 user units per character.

  Figure 1 (viewBox 760×300, plot x∈[56,736], y∈[22,232]):
    y ticks    "0.95"                              4 ch = 21.7 u; right-anchored
               at x=48 → gutter 48 u. 2.2× headroom.
    x ticks    "0.878"                             5 ch = 27.1 u; middle-anchored,
               ±13.5 u around x∈[81,711] → [67.5, 724.5] ⊂ [0,760].
    x title    62 ch = 336 u, middle-anchored at 396 → [228, 564] ⊂ [0,760].
    y title    "precision" 9 ch = 48.8 u, rotated → budget is the 210 u plot height.
    callouts   start-anchored at x=412 → budget (736−412)/5.4186 = 59 ch;
               measured longest on live data is 57 ch = 308.9 u → ends 720.9.
    probes     longest 34 ch = 184.2 u, start-anchored at x=402 → ends 586.2 ≤ 736.

  Figure 2 (viewBox 760×320, plot x∈[62,726], y∈[26,250]):
    y ticks    "0.015"                             5 ch = 27.1 u; right-anchored
               at x=54 → gutter 54 u. 2.0× headroom.
    x ticks    "1600"                              4 ch = 21.7 u; middle-anchored.
    x title    71 ch = 384.7 u, middle-anchored at 394 → [201.6, 586.4] ⊂ [0,760].
    y title    29 ch = 157.1 u, rotated → budget is the 224 u plot height.
    point labels are anchored AWAY from the nearer edge — start-anchored right of
    marks in the left half, end-anchored left of marks in the right half. Worst
    case is the longest display name on the rightmost mark:
      "Our port of RF + Type/#PMIDs/promoter" 37 ch = 200.5 u, end-anchored at
      x(1546)−9 = 630.8 → starts 430.3 ≥ 62. 6.0× headroom on the left gutter.
      "RF 2k-d13 + Type/#PMIDs/prom/avglen"   35 ch = 189.7 u, end-anchored at
      x(1681)−9 = 697.7 → starts 508.0 ≥ 62.
    the margin note is end-anchored at x=722; 43 ch = 233.0 u → starts 489.0 ≥ 62.
    the zero note is start-anchored at x=68; 16 ch = 86.7 u → ends 154.7 ≤ 726.
    Both paint after the marks: an opaque mark on a reference rule occluded this
    label's leading glyphs in review.
  Arm display names are never abbreviated to fit — the budget is sized to them.
-->
<script lang="ts">
	import {
		TIE_SCATTER_GEOMETRY,
		featuredReaderArm,
		tieScatterLabelFits,
		tieRange,
		type TieInflationArm,
		type TieInflationLoad,
		type TiePoint
	} from '$lib/data/paper-tie-inflation';

	let { data }: { data: TieInflationLoad } = $props();

	// ── figure 1 geometry ────────────────────────────────────────────────────
	const F1 = { left: 56, right: 736, top: 22, bottom: 232 } as const;
	/** Recall padding as a fraction of the window, so the block's edges are visible. */
	const F1_PAD_X = 0.04;
	const F1_PAD_Y = 0.06;

	// ── figure 2 geometry ────────────────────────────────────────────────────
	// left/right come from the data module, because the point-label fit test lives
	// there and a geometry change must move the budget with it.
	const F2 = {
		left: TIE_SCATTER_GEOMETRY.left,
		right: TIE_SCATTER_GEOMETRY.right,
		top: 26,
		bottom: 250
	} as const;
	const F2_X_MIN = 380;
	const F2_X_MAX = 1720;
	/**
	 * The zero line is deliberately lifted off the axis: four near-zero arms stack
	 * there, and their decluttered labels need ~36 user units of clear space below
	 * the marks or they land on the x tick row.
	 */
	const F2_Y_MIN = -0.004;
	const F2_Y_MAX = 0.016;
	const F2_X_TICKS = [400, 800, 1200, 1600] as const;
	const F2_Y_TICKS = [0, 0.005, 0.01, 0.015] as const;
	/** Minimum vertical separation between two point labels, in user units. */
	const F2_LABEL_GAP = 12;

	const ok = $derived(data.status === 'ok');
	const arms = $derived<TieInflationArm[]>(data.status === 'ok' ? data.arms : []);
	const featured = $derived(data.status === 'ok' ? data.featured : null);
	const margin = $derived(data.status === 'ok' ? data.margin : null);

	/**
	 * Panel size and provenance. NOT defaulted. `?? 0` here would have printed
	 * "the 0-statement panel" in three places — a fabricated measurement in the
	 * denominator of every share this figure quotes — and an empty sha would have
	 * printed as a real, empty digest. Both stay null and gate their own render.
	 */
	const nStatements = $derived<number | null>(data.nStatements);
	const artifactSha = $derived<string | null>(data.artifact_sha256);

	const readers = $derived(arms.filter((arm) => arm.isReader));
	const paperSide = $derived(arms.filter((arm) => arm.kind === 'paper' || arm.kind === 'port'));
	const paperDistinctMin = $derived(
		paperSide.length === 0 ? 0 : Math.min(...paperSide.map((arm) => arm.distinctScores))
	);
	const paperDistinctMax = $derived(
		paperSide.length === 0 ? 0 : Math.max(...paperSide.map((arm) => arm.distinctScores))
	);
	const readerRange = $derived(tieRange(readers, (arm) => arm.inflation));
	/** Largest |inflation| among the paper's own RF and our reproduction of it. */
	const paperWorst = $derived(
		paperSide.length === 0 ? null : Math.max(...paperSide.map((arm) => Math.abs(arm.inflation)))
	);
	const featuredArm = $derived(featuredReaderArm(arms));

	/**
	 * How far the like-for-like (pooled-vs-pooled) inflation moves each reading arm
	 * away from the shipped headline, and how many of those moves go UP. Derived so
	 * the estimator-mismatch note states a measured bound instead of asserting one —
	 * and so it stays self-critical if the correction ever starts favouring us.
	 */
	const estimatorShift = $derived.by(() => {
		if (readers.length === 0) return null;
		return {
			worst: Math.max(...readers.map((arm) => Math.abs(arm.sameEstimatorInflation - arm.inflation))),
			upward: readers.filter((arm) => arm.sameEstimatorInflation > arm.inflation).length,
			total: readers.length
		};
	});

	/**
	 * Share of the paper's-metric margin that is interpolation rather than
	 * ranking. Derived from the two shipped paired deltas; null when the
	 * trapezoidal margin is not positive (the statement would be meaningless).
	 */
	const interpolatedShare = $derived(
		margin && margin.trapezoidal > 0 ? (margin.trapezoidal - margin.ap) / margin.trapezoidal : null
	);

	/**
	 * True only when the trapezoid's chord credits our featured arm MORE than the
	 * paper's RF actually achieves at the probe recall, while our achievable
	 * precision there is LESS than theirs. Gated rather than asserted: if the data
	 * ever stops supporting the sentence, the sentence disappears.
	 */
	const chordOvertakes = $derived(
		featured !== null &&
			featured.reader.midChordPrecision > featured.reference.midStepPrecision &&
			featured.reader.midStepPrecision < featured.reference.midStepPrecision
	);

	function signed(value: number, digits: number): string {
		return `${value >= 0 ? '+' : '−'}${Math.abs(value).toFixed(digits)}`;
	}

	function pct(value: number): string {
		return `${(value * 100).toFixed(0)}%`;
	}

	function count(value: number): string {
		return value.toLocaleString('en-US');
	}

	// ── figure 1 scales ──────────────────────────────────────────────────────
	const f1Domain = $derived.by(() => {
		if (!featured) return null;
		const rSpan = Math.max(featured.recallTo - featured.recallFrom, 1e-9);
		const pSpan = Math.max(featured.precisionMax - featured.precisionMin, 1e-9);
		return {
			xMin: featured.recallFrom - rSpan * F1_PAD_X,
			xMax: featured.recallTo + rSpan * F1_PAD_X,
			yMin: featured.precisionMin - pSpan * F1_PAD_Y,
			yMax: featured.precisionMax + pSpan * F1_PAD_Y
		};
	});

	function fx(recall: number): number {
		const d = f1Domain;
		if (!d) return F1.left;
		return F1.left + ((recall - d.xMin) / (d.xMax - d.xMin)) * (F1.right - F1.left);
	}

	function fy(precision: number): number {
		const d = f1Domain;
		if (!d) return F1.bottom;
		return F1.bottom - ((precision - d.yMin) / (d.yMax - d.yMin)) * (F1.bottom - F1.top);
	}

	/** Achievable path: drop to the new precision, then run out to the new recall. */
	function stepPath(points: TiePoint[]): string {
		if (points.length === 0) return '';
		let d = `M ${fx(points[0].recall).toFixed(2)} ${fy(points[0].precision).toFixed(2)}`;
		for (let i = 1; i < points.length; i += 1) {
			d += ` L ${fx(points[i - 1].recall).toFixed(2)} ${fy(points[i].precision).toFixed(2)}`;
			d += ` L ${fx(points[i].recall).toFixed(2)} ${fy(points[i].precision).toFixed(2)}`;
		}
		return d;
	}

	/** What the trapezoid integrates: a straight chord between operating points. */
	function chordPath(points: TiePoint[]): string {
		if (points.length === 0) return '';
		return points
			.map(
				(point, index) =>
					`${index === 0 ? 'M' : 'L'} ${fx(point.recall).toFixed(2)} ${fy(point.precision).toFixed(2)}`
			)
			.join(' ');
	}

	/** The interpolated area itself: chord, back along the step, close. */
	const trianglePath = $derived.by(() => {
		if (!featured) return '';
		const { from, to } = featured.block;
		return (
			`M ${fx(from.recall).toFixed(2)} ${fy(from.precision).toFixed(2)}` +
			` L ${fx(to.recall).toFixed(2)} ${fy(to.precision).toFixed(2)}` +
			` L ${fx(from.recall).toFixed(2)} ${fy(to.precision).toFixed(2)} Z`
		);
	});

	const f1XTicks = $derived(
		featured ? [featured.recallFrom, featured.midRecall, featured.recallTo] : []
	);

	/** Round precision ticks inside the shared axis, at 0.05 spacing. */
	const f1YTicks = $derived.by(() => {
		const d = f1Domain;
		if (!d) return [];
		const out: number[] = [];
		for (let v = Math.ceil(d.yMin * 20) / 20; v <= d.yMax + 1e-9; v += 0.05) {
			out.push(Math.round(v * 100) / 100);
		}
		return out;
	});

	// ── figure 2 scales + label declutter ────────────────────────────────────
	function gx(distinct: number): number {
		return F2.left + ((distinct - F2_X_MIN) / (F2_X_MAX - F2_X_MIN)) * (F2.right - F2.left);
	}

	function gy(inflation: number): number {
		return F2.bottom - ((inflation - F2_Y_MIN) / (F2_Y_MAX - F2_Y_MIN)) * (F2.bottom - F2.top);
	}

	type MarkShape = 'circle' | 'square' | 'diamond' | 'triangle';

	interface Mark {
		arm: TieInflationArm;
		x: number;
		y: number;
		labelX: number;
		labelY: number;
		anchor: 'start' | 'end';
		leader: boolean;
		shape: MarkShape;
		/** Series class: its own (stroke, dash) pair, redundant with the shape. */
		series: 'paper' | 'port' | 'reader' | 'bundle';
	}

	function shapeOf(arm: TieInflationArm): { shape: MarkShape; series: Mark['series'] } {
		if (arm.kind === 'paper') return { shape: 'square', series: 'paper' };
		if (arm.kind === 'port') return { shape: 'diamond', series: 'port' };
		if (arm.isReader) return { shape: 'circle', series: 'reader' };
		return { shape: 'triangle', series: 'bundle' };
	}

	/**
	 * Place each point label on the side away from the nearer plot edge, then push
	 * overlapping labels apart within each side (top-down, minimum F2_LABEL_GAP)
	 * and draw a leader wherever a label had to move off its mark.
	 */
	const marks = $derived.by<Mark[]>(() => {
		const midX = (F2.left + F2.right) / 2;
		const built = arms.map((arm) => {
			const x = gx(arm.distinctScores);
			const y = gy(arm.inflation);
			const anchor: 'start' | 'end' = x > midX ? 'end' : 'start';
			return {
				arm,
				x,
				y,
				labelX: anchor === 'start' ? x + TIE_SCATTER_GEOMETRY.labelOffsetX : x - TIE_SCATTER_GEOMETRY.labelOffsetX,
				labelY: y,
				anchor,
				leader: false,
				...shapeOf(arm)
			};
		});
		for (const side of ['start', 'end'] as const) {
			const group = built
				.filter((mark) => mark.anchor === side)
				.sort((a, b) => a.y - b.y || a.arm.id.localeCompare(b.arm.id));
			let lastY = Number.NEGATIVE_INFINITY;
			for (const mark of group) {
				mark.labelY = Math.max(mark.y, lastY + F2_LABEL_GAP);
				lastY = mark.labelY;
				mark.leader = Math.abs(mark.labelY - mark.y) > 2.5;
			}
		}
		return built.map((mark) => ({ ...mark, labelY: mark.labelY + 3 }));
	});

	/**
	 * Every point label measured against the gutter it is anchored into. Direct
	 * labelling is the whole legend of figure 2, so a label that would run off the
	 * frame gates the figure rather than shipping a clipped model name — SVG
	 * clips silently and the <desc> hides the damage from every other check.
	 */
	const scatterLabelsFit = $derived(
		marks.every((mark) => tieScatterLabelFits(mark.arm.display, mark.labelX, mark.anchor))
	);

	const marginY = $derived(margin ? gy(margin.ap) : null);

	function shortSha(value: string): string {
		return `${value.slice(0, 10)}…`;
	}
</script>

<section class="tie" aria-labelledby="tie-title">
	{#if !ok || !featured || !margin || !featuredArm || nStatements === null}
		<div class="gate" role="status">
			<p class="eyebrow">how the score is computed</p>
			<h2 id="tie-title">Interpolation credit unavailable</h2>
			<p>
				{data.status === 'unavailable'
					? data.reason
					: 'The rebuilt precision-recall curves are missing, so the extra credit cannot be shown.'}
			</p>
			<code>{data.artifact_path}</code>
		</div>
	{:else}
		<header>
			<div>
				<p class="eyebrow">the same score, turned against this result</p>
				<h2 id="tie-title">Why that way of scoring flatters the reading models and not the random forests</h2>
			</div>
			<strong>argues against the margin above</strong>
		</header>

		

		<!-- ── figure 1 · the mechanism, real vertices ───────────────────────── -->
		<figure>
			<figcaption>
				The biggest group of statements {featured.reader.display} gave one and the same score, with
				the random forest drawn over the same stretch for comparison
			</figcaption>
			<svg viewBox="0 0 760 300" role="img" aria-labelledby="tie-f1-title tie-f1-desc">
				<title id="tie-f1-title">
					What a score cutoff can really reach, against the straight line drawn between two points
				</title>
				<desc id="tie-f1-desc">
					Across this stretch of recall, {featured.recallFrom.toFixed(3)} to
					{featured.recallTo.toFixed(3)}, {featured.reader.display} offers only one place to put a
					cutoff: {count(featured.block.size)} statements carry the identical score
					{featured.block.score.toFixed(2)}, so they are all admitted together and precision drops
					from {featured.block.from.precision.toFixed(3)} to {featured.block.to.precision.toFixed(3)}
					in a single step. Straight-line interpolation instead runs a diagonal across that group; the
					shaded triangle between the diagonal and the step is {signed(featured.block.area, 4)} of
					area awarded for an ordering that does not exist. The random forest,
					{featured.reference.display}, crosses the same stretch in
					{count(featured.reference.vertices.length - 1)} much smaller steps, so the diagonal drawn
					through its points sits on its own staircase.
				</desc>
				<defs>
					<pattern
						id="tie-hatch"
						width="6"
						height="6"
						patternUnits="userSpaceOnUse"
						patternTransform="rotate(45)"
					>
						<line class="hatch" x1="0" y1="0" x2="0" y2="6" />
					</pattern>
					<clipPath id="tie-f1-clip">
						<rect
							x={F1.left}
							y={F1.top}
							width={F1.right - F1.left}
							height={F1.bottom - F1.top}
						/>
					</clipPath>
				</defs>

				{#each f1YTicks as tick (tick)}
					<line class="grid" x1={F1.left} y1={fy(tick)} x2={F1.right} y2={fy(tick)} />
					<text class="tick tick-y" x={F1.left - 8} y={fy(tick) + 3}>{tick.toFixed(2)}</text>
				{/each}
				{#each f1XTicks as tick (tick)}
					<line class="grid" x1={fx(tick)} y1={F1.top} x2={fx(tick)} y2={F1.bottom} />
					<text class="tick tick-x" x={fx(tick)} y={F1.bottom + 15}>{tick.toFixed(3)}</text>
				{/each}

				<g clip-path="url(#tie-f1-clip)">
					<path class="triangle" d={trianglePath} />
					<path class="ref-step" d={stepPath(featured.reference.vertices)} />
					<path class="ref-chord" d={chordPath(featured.reference.vertices)} />
					<path class="reader-step" d={stepPath(featured.reader.vertices)} />
					<path class="reader-chord" d={chordPath(featured.reader.vertices)} />

					<!-- mid-window probe: what is credited, what is achievable, what the RF gets -->
					<line
						class="probe"
						x1={fx(featured.midRecall)}
						y1={fy(featured.reader.midChordPrecision)}
						x2={fx(featured.midRecall)}
						y2={fy(featured.reader.midStepPrecision)}
					/>
					<circle class="probe-dot credited" cx={fx(featured.midRecall)} cy={fy(featured.reader.midChordPrecision)} r="3" />
					<circle class="probe-dot" cx={fx(featured.midRecall)} cy={fy(featured.reference.midStepPrecision)} r="3" />
					<circle class="probe-dot" cx={fx(featured.midRecall)} cy={fy(featured.reader.midStepPrecision)} r="3" />
				</g>

				<!-- callouts sit in the empty wedge above the chord (the chord passes
				     y≈132 at x=412, these sit at y 52/66). Start-anchored at x=412 →
				     budget (736−412)/5.4186 = 59 characters; longest is 57. -->
				<text class="callout" x="412" y="52">
					{count(featured.block.size)} statements tied at {featured.block.score.toFixed(2)} — {count(
						featured.block.nTrue
					)} of them true
				</text>
				<text class="callout muted" x="412" y="66">
					{signed(featured.block.area, 4)} of unearned area, {pct(
						featured.block.shareOfArmInflation
					)} of this model’s total
				</text>

				<!-- probe labels; budget 184.2 u from x=402, ends 586.2 ≤ 736 -->
				<text class="probe-label" x={fx(featured.midRecall) + 6} y={fy(featured.reader.midChordPrecision) - 5}>
					{featured.reader.midChordPrecision.toFixed(3)} credited by the straight line
				</text>
				<text class="probe-label" x={fx(featured.midRecall) + 6} y={fy(featured.reference.midStepPrecision) - 5}>
					{featured.reference.midStepPrecision.toFixed(3)} the random forest really gets
				</text>
				<text class="probe-label" x={fx(featured.midRecall) + 6} y={fy(featured.reader.midStepPrecision) - 5}>
					{featured.reader.midStepPrecision.toFixed(3)} the best any score cutoff gets
				</text>

				<line class="axis" x1={F1.left} y1={F1.bottom} x2={F1.right} y2={F1.bottom} />
				<line class="axis" x1={F1.left} y1={F1.top} x2={F1.left} y2={F1.bottom} />
				<text class="axis-label" x={(F1.left + F1.right) / 2} y={F1.bottom + 34}>
					recall — share of the correct statements among these {count(nStatements)} that have been found
					→
				</text>
				<text
					class="axis-label"
					transform={`rotate(-90 16 ${(F1.top + F1.bottom) / 2})`}
					x="16"
					y={(F1.top + F1.bottom) / 2}>precision</text
				>
			</svg>
			<ul class="legend">
				<li>
					<svg class="swatch" viewBox="0 0 26 10" aria-hidden="true"
						><line class="reader-step" x1="1" y1="8" x2="9" y2="8" /><line
							class="reader-step"
							x1="9"
							y1="8"
							x2="9"
							y2="2"
						/><line class="reader-step" x1="9" y1="2" x2="25" y2="2" /></svg
					>
					<span><b>{featured.reader.display}</b>, what a score cutoff can actually reach</span>
				</li>
				<li>
					<svg class="swatch" viewBox="0 0 26 10" aria-hidden="true"
						><line class="reader-chord" x1="1" y1="8" x2="25" y2="2" /></svg
					>
					<span><b>{featured.reader.display}</b>, what the straight line between points scores</span>
				</li>
				<li>
					<svg class="swatch" viewBox="0 0 26 10" aria-hidden="true"
						><line class="ref-step" x1="1" y1="8" x2="25" y2="3" /></svg
					>
					<span><b>{featured.reference.display}</b>, the random forest, what its cutoffs reach</span>
				</li>
				<li>
					<svg class="swatch" viewBox="0 0 26 10" aria-hidden="true"
						><line class="ref-chord" x1="1" y1="8" x2="25" y2="3" /></svg
					>
					<span
						><b>{featured.reference.display}</b>, what the straight line scores for it — no different
						from its own steps, worth {signed(featured.reference.windowInflation, 6)} of extra area
						across this whole stretch</span
					>
				</li>
				<li>
					<svg class="swatch" viewBox="0 0 26 10" aria-hidden="true"
						><defs
							><pattern
								id="tie-hatch-legend"
								width="6"
								height="6"
								patternUnits="userSpaceOnUse"
								patternTransform="rotate(45)"
							>
								<line class="hatch" x1="0" y1="0" x2="0" y2="6" />
							</pattern></defs
						><rect class="swatch-fill" x="1" y="1" width="24" height="8" /></svg
					>
					<span>interpolation credit — area the straight line awards that no cutoff can reach</span>
				</li>
			</ul>
		</figure>

		{#if chordOvertakes}
			<p class="reading">
				Read the vertical probe on the figure. At recall {featured.midRecall.toFixed(3)} the straight
				line credits {featured.reader.display} with precision
				{featured.reader.midChordPrecision.toFixed(3)} — precision being the share of the statements
				accepted at that point that really are correct. No cutoff reaches it: the best any cutoff
				achieves there is {featured.reader.midStepPrecision.toFixed(3)}, while the random forest
				genuinely achieves {featured.reference.midStepPrecision.toFixed(3)}. So at that point the
				interpolation puts the reading model above the random forest when in truth it is below.
			</p>
		{/if}

		<!-- ── figure 2 · inflation against tie-ness, all eight arms ─────────── -->
		<figure>
			<figcaption>
				Interpolation credit against how many statements each model gives the same score
			</figcaption>
			{#if !scatterLabelsFit}
				<!-- The reason rides in `title`, the page-wide home for method detail
				     (same convention as the ranked-block and unmeasured-cell tooltips).
				     Visible copy stays at the statement itself. -->
				<p
					class="fig-gate"
					role="status"
					title="This figure names every model at its own mark. One of those names no longer fits the space it is anchored into, and SVG cuts it off with no error and no failing accessibility check, because the description still emits the full string. The figure is withheld rather than drawn with a chopped model name. Every value it would plot is in the table below, unchanged."
				>
					Figure withheld: a model name no longer fits its space. Every value it would plot is in
					the table below.
				</p>
			{:else}
			<svg viewBox="0 0 760 320" role="img" aria-labelledby="tie-f2-title tie-f2-desc">
				<title id="tie-f2-title"
					>Interpolation credit against how many different scores each model produces</title
				>
				<desc id="tie-f2-desc">
					Each model is one mark: horizontally, how many different scores it gives across the
					{count(nStatements)} statements; vertically, how much the straight-line score exceeds the
					step-wise one. The random forests re-run here from the released 2023 code give
					{count(paperDistinctMin)} to {count(paperDistinctMax)} different scores and sit on the zero
					line. The reading models give a few hundred and sit far above it.
				</desc>

				{#each F2_Y_TICKS as tick (tick)}
					<line class="grid" x1={F2.left} y1={gy(tick)} x2={F2.right} y2={gy(tick)} />
					<text class="tick tick-y" x={F2.left - 8} y={gy(tick) + 3}>{tick.toFixed(3)}</text>
				{/each}
				{#each F2_X_TICKS as tick (tick)}
					<text class="tick tick-x" x={gx(tick)} y={F2.bottom + 15}>{count(tick)}</text>
				{/each}

				<line class="zero" x1={F2.left} y1={gy(0)} x2={F2.right} y2={gy(0)} />
				{#if marginY !== null}
					<line class="margin-line" x1={F2.left} y1={marginY} x2={F2.right} y2={marginY} />
				{/if}

				{#each marks as mark (mark.arm.id)}
					{#if mark.leader}
						<line
							class="leader"
							x1={mark.x + (mark.anchor === 'start' ? 5 : -5)}
							y1={mark.y}
							x2={mark.labelX + (mark.anchor === 'start' ? -2 : 2)}
							y2={mark.labelY - 3}
						/>
					{/if}
					{#if mark.shape === 'circle'}
						<circle class="mark {mark.series}" cx={mark.x} cy={mark.y} r="4.2" />
					{:else if mark.shape === 'square'}
						<rect class="mark {mark.series}" x={mark.x - 3.8} y={mark.y - 3.8} width="7.6" height="7.6" />
					{:else if mark.shape === 'diamond'}
						<!-- The port reproduces the paper RF's tie structure exactly, so the two
						     marks genuinely coincide. Neither position is nudged; the diamond is
						     drawn large enough that the square reads as nested inside it. -->
						<path
							class="mark {mark.series}"
							d={`M ${mark.x} ${mark.y - 8} L ${mark.x + 8} ${mark.y} L ${mark.x} ${mark.y + 8} L ${mark.x - 8} ${mark.y} Z`}
						/>
					{:else}
						<path
							class="mark {mark.series}"
							d={`M ${mark.x} ${mark.y - 5} L ${mark.x + 4.6} ${mark.y + 3.4} L ${mark.x - 4.6} ${mark.y + 3.4} Z`}
						/>
					{/if}
					<text class="point-label" x={mark.labelX} y={mark.labelY} text-anchor={mark.anchor}>
						{mark.arm.display}
					</text>
				{/each}

				<!-- Reference-line labels paint AFTER the marks: an opaque mark sitting on a
				     rule silently ate the leading glyphs of this label once already. The
				     zero note is start-anchored on the LEFT, where the near-zero cluster
				     never reaches — 16 ch = 86.7 u from x=68 ends at 154.7, and the
				     leftmost mark on that rule sits at x≈456. -->
				<text class="ref-note ref-note-left" x={F2.left + 6} y={gy(0) - 5}>nothing interpolated</text>
				{#if marginY !== null}
					<text class="ref-note margin-note" x={F2.right - 4} y={marginY - 5}>
						the whole average-precision lead, {signed(margin.ap, 4)}
					</text>
				{/if}

				<line class="axis" x1={F2.left} y1={F2.bottom} x2={F2.right} y2={F2.bottom} />
				<line class="axis" x1={F2.left} y1={F2.top} x2={F2.left} y2={F2.bottom} />
				<text class="axis-label" x={(F2.left + F2.right) / 2} y={F2.bottom + 34}>
					different scores given to the {count(nStatements)} statements (fewer = more statements
					share a score) →
				</text>
				<text
					class="axis-label"
					transform={`rotate(-90 16 ${(F2.top + F2.bottom) / 2})`}
					x="16"
					y={(F2.top + F2.bottom) / 2}>credit for interpolation</text
				>
			</svg>
			{/if}
		</figure>

		<!-- ── the punchline ─────────────────────────────────────────────────── -->
		<div class="punchline">
			{#if readerRange && paperWorst !== null}
				<p>
					Interpolation hands the reading models between {signed(readerRange.min, 4)} and
					{signed(readerRange.max, 4)} they did not earn. The random forest — released code, re-run
					here — and the re-implementation of it gain at most {paperWorst.toFixed(4)}, because they give
					almost every statement its own score, leaving almost nothing to interpolate across.
				</p>
			{/if}
			{#if interpolatedShare !== null}
				<p class="hard">
					Scored the way the 2023 paper scored it — precision-recall area with a straight line drawn
					between points — {margin.armDisplay} beats {margin.referenceDisplay} by
					{signed(margin.trapezoidal, 4)}. Scored step-wise instead, so tied statements earn no
					interpolated credit, the same head-to-head over the same {count(nStatements)} statements
					comes to {signed(margin.ap, 4)}. {pct(interpolatedShare)} of that lead was interpolation
					across tied groups in the reading models’ own scores, not better ranking.
				</p>
			{/if}
			<p>
				One group of statements does most of it. The {count(featured.block.size)} statements the
				reading step scored {featured.block.score.toFixed(2)} — {count(featured.block.nTrue)} of them
				correct — contribute {signed(featured.block.area, 4)} on their own,
				{pct(featured.block.shareOfArmInflation)} of {featured.reader.display}’s total{#if featured
					.block.area > margin.ap}, and more than the entire average-precision lead above{/if}. That
				group cannot be put in order, by construction: reading can only throw evidence away, so every
				statement left with nothing lands on the same score, and the straight line then awards credit
				for an ordering no model here produced.
			</p>
		</div>

		<details>
			<summary>The numbers per model, and the one apples-to-oranges worry behind them</summary>
			<p class="lede">
			The 2023 paper scores each method as <code>auc(recall, precision)</code>: it plots the method's
			precision-recall points and takes the area under a straight line drawn between each pair of
			neighbouring points. No score cutoff actually delivers any point on that line — a group of
			statements carrying the <em>same</em> score is accepted all at once, so the curve you can really
			reach is a staircase. Average precision takes the area under the staircase. The difference
			between the two numbers is pure interpolation, and it grows the more statements a model gives
			identical scores. The published random forest gives almost every statement its own score. The
			reading models do not.
		</p>
			<p class="note">
				One fair objection: the published number is an average over ten folds — the statements split
				into ten groups, each fitted model scored on the group it did not train on — while average
				precision here is computed over all statements at once, so part of the difference could be
				that mismatch rather than interpolation. The like-for-like column settles it — both
				numbers computed over all statements at once, from the same file.{#if estimatorShift}
					Doing that moves the reading models by at most {estimatorShift.worst.toFixed(5)}, and for
					{estimatorShift.upward} of {estimatorShift.total} it moves the number <em>up</em>. The
					mismatch is not what inflates the reading models.{/if}
			</p>
			<div class="table-scroll">
				<table>
					<thead>
						<tr>
							<th>model</th>
							<th>different scores given</th>
							<th>interpolation credit (as shipped)</th>
							<th>like-for-like</th>
							<th>average precision</th>
							<th>straight-line area, averaged over 10 folds</th>
						</tr>
					</thead>
					<tbody>
						{#each arms as arm (arm.id)}
							<tr class:reader={arm.isReader}>
								<td>{arm.display}</td>
								<td>{count(arm.distinctScores)}</td>
								<td>{signed(arm.inflation, 5)}</td>
								<td>{signed(arm.sameEstimatorInflation, 5)}</td>
								<td>{arm.ap.toFixed(4)}</td>
								<td>{arm.foldMeanTrapezoidal.toFixed(4)}</td>
							</tr>
						{/each}
					</tbody>
				</table>
			</div>
			<p class="note">
				The 2023 paper reports, beside each score, how much that score varies across the ten
				folds. That is a measure of wobble between folds, not a confidence interval, and
				it belongs to the straight-line score alone — which is why it is deliberately absent from the
				average-precision columns here.
			</p>
		</details>

		<footer>
			Curves rebuilt from the aligned score vectors and the labels published in 2023; the drawn
			triangles reproduce each model’s shipped straight-line-minus-step-wise gap to within
			{Math.max(
				featured.reader.reconciliationResidual,
				featured.reference.reconciliationResidual
			).toExponential(1)}. <code>{data.artifact_path}</code> · artifact
			<!-- A missing digest is stated, not dashed: '—' beside a path reads as
			     "pinned, abbreviated", which is the opposite of what it means. -->
			{#if artifactSha}<span title={artifactSha}>{shortSha(artifactSha)}</span>{:else}<span
					class="sha-missing">not sha-pinned</span
				>{/if}
		</footer>
	{/if}
</section>

<style>
	.tie {
		margin-top: 1.5rem;
		padding-top: 1.25rem;
		border-top: 2px solid var(--ink);
	}
	/* A withheld figure, styled like the section gate: it is a stated absence, not
	   an aside. Same token as .gate so the two read as one convention. */
	.fig-gate {
		margin: 0;
		border: 1px solid var(--rule);
		border-left: 3px solid var(--blocked);
		padding: 0.8rem 0.9rem;
		font-family: var(--serif);
		font-size: 0.82rem;
		line-height: 1.5;
		color: var(--ink-muted);
	}
	/* An absent digest is stated, never dashed. */
	.sha-missing {
		color: var(--blocked);
	}
	.tie > header {
		display: flex;
		justify-content: space-between;
		align-items: start;
		gap: 1rem;
	}
	.tie h2,
	.gate h2 {
		margin: 0.18rem 0 0;
		font-family: var(--serif);
		font-size: 1.35rem;
		font-weight: 400;
	}
	.tie > header > strong {
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
	.lede,
	.reading {
		font-family: var(--serif);
		font-size: 0.88rem;
		line-height: 1.55;
		color: var(--ink-muted);
		max-width: 72ch;
		margin: 0.9rem 0 0;
	}
	.reading {
		padding-left: 0.7rem;
		border-left: 2px solid var(--rule);
	}
	.lede code {
		font-size: 0.8rem;
	}

	figure {
		margin: 1.3rem 0 0;
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
	.tick,
	.axis-label,
	.callout,
	.probe-label,
	.point-label,
	.ref-note {
		font-family: var(--mono);
		font-size: 9px;
		fill: var(--ink-faint);
	}
	.tick-y {
		text-anchor: end;
	}
	.tick-x,
	.axis-label {
		text-anchor: middle;
	}
	/* Halo: these labels sit over the hatched area, the reference staircase and the
	   margin rule. Painting the stroke first puts a --paper outline behind each
	   glyph, so a crossing line never eats a digit. */
	.callout,
	.probe-label,
	.point-label,
	.ref-note {
		paint-order: stroke fill;
		stroke: var(--paper);
		stroke-width: 3px;
		stroke-linejoin: round;
	}
	.callout,
	.probe-label {
		fill: var(--ink-muted);
	}
	.callout.muted {
		fill: var(--ink-faint);
	}
	.ref-note {
		text-anchor: end;
	}
	.ref-note-left {
		text-anchor: start;
	}
	.margin-note {
		fill: var(--ink);
	}

	/* Each series carries its OWN (stroke, dash) pair; the reader/reference pair is
	   further separated by width, and by shape (one leap versus a staircase). All
	   four strokes clear 3:1 against --paper: --ink ≈ 16:1, --accent ≈ 9.2:1,
	   --blocked ≈ 6.5:1. */
	.reader-step {
		fill: none;
		stroke: var(--ink);
		stroke-width: 1.8;
	}
	.reader-chord {
		fill: none;
		stroke: var(--blocked);
		stroke-width: 1.8;
		stroke-dasharray: 7 4;
	}
	.ref-step {
		fill: none;
		stroke: var(--accent);
		stroke-width: 1.1;
	}
	.ref-chord {
		fill: none;
		stroke: var(--accent);
		stroke-width: 1.1;
		stroke-dasharray: 2 3;
	}
	.triangle {
		fill: url(#tie-hatch);
		stroke: none;
	}
	.hatch {
		stroke: var(--blocked);
		stroke-width: 1;
		opacity: 0.32;
	}
	.probe {
		stroke: var(--ink);
		stroke-width: 1;
		stroke-dasharray: 1 2;
	}
	.probe-dot {
		fill: var(--paper);
		stroke: var(--ink);
		stroke-width: 1.2;
	}
	.probe-dot.credited {
		fill: var(--blocked);
		stroke: var(--paper);
	}

	.legend {
		list-style: none;
		margin: 0.55rem 0 0;
		padding: 0;
		display: grid;
		gap: 0.28rem;
	}
	.legend li {
		display: flex;
		align-items: baseline;
		gap: 0.5rem;
		font-family: var(--mono);
		font-size: 0.66rem;
		line-height: 1.4;
		color: var(--ink-faint);
	}
	.legend b {
		color: var(--ink-muted);
		font-weight: 500;
	}
	.swatch {
		flex: 0 0 auto;
		width: 26px;
		height: 10px;
		overflow: visible;
	}
	/* The legend swatch carries its own pattern def — a paint server referenced
	   across two separate inline <svg> roots is not reliably resolved. */
	.swatch-fill {
		fill: url(#tie-hatch-legend);
		stroke: var(--rule);
		stroke-width: 1;
	}

	.zero {
		stroke: var(--ink-faint);
		stroke-width: 1;
		stroke-dasharray: 1 3;
	}
	.margin-line {
		stroke: var(--ink);
		stroke-width: 1;
		stroke-dasharray: 5 3;
	}
	.leader {
		stroke: var(--ink-faint);
		stroke-width: 0.8;
	}
	.point-label {
		fill: var(--ink-muted);
	}
	/* Four series, four (stroke, dash) pairs, four marker shapes. The open marks are
	   fill:none, not paper-filled: the port reproduces the paper RF's tie structure
	   exactly, so those two marks coincide, and an opaque fill on the one drawn
	   second erased the one drawn first. */
	.mark.paper {
		fill: none;
		stroke: var(--accent);
		stroke-width: 1.6;
	}
	.mark.port {
		fill: none;
		stroke: var(--accent);
		stroke-width: 1.6;
		stroke-dasharray: 3 2;
	}
	.mark.reader {
		fill: var(--blocked);
		stroke: var(--paper);
		stroke-width: 1;
	}
	.mark.bundle {
		fill: none;
		stroke: var(--blocked);
		stroke-width: 1.6;
		stroke-dasharray: 2 2;
	}

	.punchline {
		margin-top: 1.4rem;
		padding-top: 0.9rem;
		border-top: 1px dotted var(--rule);
	}
	.punchline p {
		font-family: var(--serif);
		font-size: 0.88rem;
		line-height: 1.55;
		color: var(--ink-muted);
		max-width: 72ch;
		margin: 0 0 0.7rem;
	}
	.punchline p:last-child {
		margin-bottom: 0;
	}
	.punchline .hard {
		color: var(--ink);
		border-left: 3px solid var(--accent);
		padding-left: 0.7rem;
	}

	details {
		margin-top: 1.1rem;
		border-top: 1px solid var(--rule);
		padding-top: 0.65rem;
	}
	summary {
		cursor: pointer;
		font-family: var(--mono);
		font-size: 0.68rem;
		color: var(--ink-muted);
	}
	.note {
		font-family: var(--serif);
		font-size: 0.78rem;
		line-height: 1.5;
		color: var(--ink-faint);
		max-width: 74ch;
		margin: 0.7rem 0;
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
	td:nth-child(n + 2) {
		font-variant-numeric: tabular-nums;
	}
	tr.reader td {
		color: var(--ink);
	}

	code,
	footer {
		font-family: var(--mono);
		font-size: 0.62rem;
		color: var(--ink-faint);
	}
	footer {
		margin-top: 1rem;
		line-height: 1.5;
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
		.tie > header {
			display: block;
		}
		.tie > header > strong {
			display: inline-block;
			margin-top: 0.6rem;
		}
	}
</style>
