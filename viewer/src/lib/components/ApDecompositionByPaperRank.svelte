<script lang="ts">
	/**
	 * Where the point comes from — the ΔAP column of the head-to-head table above,
	 * walked up an EXOGENOUS variable: how much evidence each statement carries.
	 *
	 * Average precision decomposes with no residual, so each arm's shipped ΔAP is
	 * reached, not asserted: per-band contribution differences accumulated left to
	 * right, endpoint = the observed point delta. Every number is read from
	 * `ap_decomposition_by_paper_band.json` through the fail-closed validator — no
	 * series, count, or interval is hard-coded here; only layout constants are.
	 *
	 * WHY THE BANDS ARE EVIDENCE COUNTS. The first version of this figure banded by
	 * the reference arm's own out-of-fold score and read a story off the shape.
	 * That shape is regression to the mean on the reference's own noise: band by
	 * the compared arm instead and the sign flips exactly. The strip beneath the
	 * waterfall SHOWS that reversal rather than burying it — volunteering the
	 * artifact is the evidence for banding on a corpus census instead.
	 *
	 * Deliberate constraints (do not "improve" these):
	 *   · No band/ribbon/envelope around any line. Intermediate cumulative points
	 *     have NO interval; the 95% whiskers exist only at the terminus fan.
	 *   · The fan is not a second chart: it shares this figure's y-scale and its
	 *     zero rule, and never gets its own axis or zero line.
	 *   · y-domain is fixed to [-2.8, +2.1] AP points and never broken or widened;
	 *     the validator rejects any payload that would escape it. The mirror strip
	 *     has its own fixed ±1.25 pt domain and the builder rejects escapes there.
	 *   · Every arm has its own stroke AND dash; identity resolves at the fan.
	 *     No metric toggle, no interaction, no "significant"/"wins"/check-mark
	 *     copy, and no band is ever labelled "fixed"/"rescued"/"recovered".
	 *   · Arm names in prose are ALWAYS `slot.display`. `slot.label` is a frozen
	 *     point_metrics join key and is never rendered.
	 */
	import {
		AP_DECOMP_BAND_COUNT,
		AP_DECOMP_FAN_GEOMETRY,
		AP_DECOMP_FAN_SLOTS,
		AP_DECOMP_LINE_DRAW_ORDER,
		AP_DECOMP_MIRROR_GEOMETRY,
		AP_DECOMP_Y_MAX,
		AP_DECOMP_Y_MIN,
		AP_DECOMP_Y_TICKS,
		apDecompCountNoteFits,
		apDecompFanNamesFit,
		buildApDecompMirror,
		topLlmBandSpreadPts,
		type ApDecompFanSlot,
		type ApDecompMirrorStrip,
		type ApDecompositionArm
	} from '$lib/data/paper-ap-decomposition';
	import type { PaperLiteralLoad, Standing } from '$lib/data/paper-literal';

	let { data }: { data: PaperLiteralLoad } = $props();

	/**
	 * What the readout says about where an interval sits. A TOTAL record, so adding
	 * a fourth class would fail to compile rather than falling through to a default
	 * that quietly reads as one of the other three. The old line was
	 * `arm.clearsZero ? '' : ' · crosses 0'`, which printed the blank — the reading
	 * a viewer takes as "clears zero, this one won" — for an interval lying entirely
	 * BELOW zero.
	 */
	const ZERO_NOTE: Record<Standing, string> = {
		ahead: '',
		behind: ' · entirely below 0',
		'not-significant': ' · crosses 0'
	};

	// ---- fixed layout (900x644 user units) ----------------------------------
	const BODY_LEFT = 56;
	const BODY_RIGHT = 560;
	// From the data module, where the fan's label-fit check is derived from them.
	const GAP_RIGHT = AP_DECOMP_FAN_GEOMETRY.gapRight;
	const FAN_RIGHT = AP_DECOMP_FAN_GEOMETRY.fanRight;
	const SEPARATOR_X = 578;
	const PLOT_TOP = 48;
	const PLOT_BOTTOM = 330;
	const BAND_W = (BODY_RIGHT - BODY_LEFT) / AP_DECOMP_BAND_COUNT;
	const FAN_SLOT_W = (FAN_RIGHT - GAP_RIGHT) / AP_DECOMP_FAN_SLOTS.length;
	const PTS_PER_UNIT = (PLOT_BOTTOM - PLOT_TOP) / (AP_DECOMP_Y_MAX - AP_DECOMP_Y_MIN);
	/** Top of the banding-sensitivity strip's own coordinate system. */
	const MIRROR_TOP = 466;
	const MIRROR_UNITS_PER_PT = AP_DECOMP_MIRROR_GEOMETRY.halfWidth / 1.25;

	function bandCentre(index: number): number {
		return BODY_LEFT + BAND_W * (index + 0.5);
	}
	function fanCentre(index: number): number {
		return GAP_RIGHT + FAN_SLOT_W * (index + 0.5);
	}
	/** AP points -> user units. Shared by the plot body AND the terminus fan. */
	function sy(points: number): number {
		return PLOT_TOP + (AP_DECOMP_Y_MAX - points) * PTS_PER_UNIT;
	}
	function r2(value: number): number {
		return Math.round(value * 100) / 100;
	}

	const zeroY = sy(0);

	// ---- formatting ---------------------------------------------------------
	const MINUS = '−';
	function signed(value: number, digits: number): string {
		return `${value >= 0 ? '+' : MINUS}${Math.abs(value).toFixed(digits)}`;
	}
	/** Ticks read in average precision itself; the internal scale is AP x100. */
	function tickLabel(value: number): string {
		return value === 0 ? '0' : signed(value / 100, 3);
	}
	function share(p: number): string {
		return `${(p * 100).toFixed(1)}%`;
	}
	/** Locale-independent thousands separator (SSR and client must agree). */
	function grouped(value: number): string {
		return String(value).replace(/\B(?=(\d{3})+(?!\d))/g, ',');
	}

	// ---- data ---------------------------------------------------------------
	const decomposition = $derived(data.status === 'ok' ? data.apDecomposition : null);
	const armByLabel = $derived(
		new Map((decomposition?.arms ?? []).map((arm) => [arm.label, arm] as const))
	);
	const slotArms = $derived(
		AP_DECOMP_FAN_SLOTS.map((slot) => ({ slot, arm: armByLabel.get(slot.label) })).filter(
			(entry): entry is { slot: ApDecompFanSlot; arm: ApDecompositionArm } =>
				entry.arm !== undefined
		)
	);
	/** Painting order, back to front: paper variant, E2B, 31B, GLM-5, 26B. */
	const drawOrder = $derived(
		AP_DECOMP_LINE_DRAW_ORDER.map((label) =>
			slotArms.find((entry) => entry.slot.label === label)
		).filter((entry): entry is { slot: ApDecompFanSlot; arm: ApDecompositionArm } => Boolean(entry))
	);
	/**
	 * The banding-sensitivity strip. `buildApDecompMirror` THROWS on an over-budget
	 * label or a bar outside its fixed domain, and the reversal it draws is the
	 * justification for the bands above — so a failure gates the whole figure
	 * rather than quietly dropping the evidence.
	 */
	const mirror = $derived.by<ApDecompMirrorStrip | null>(() => {
		if (!decomposition) return null;
		try {
			return buildApDecompMirror(decomposition);
		} catch {
			return null;
		}
	});
	/**
	 * The terminus fan IS this figure's legend — every trace is identified only by
	 * the name under its own endpoint — so a collision between two of those names
	 * takes the figure's identity encoding with it. Checked pairwise (see
	 * `apDecompFanNamesFit`), and a failure gates the whole figure.
	 */
	const fanNamesFit = apDecompFanNamesFit();
	const displayByLabel = $derived(
		new Map(AP_DECOMP_FAN_SLOTS.map((slot) => [slot.label, slot.display] as const))
	);
	const shortByLabel = $derived(
		new Map(AP_DECOMP_FAN_SLOTS.map((slot) => [slot.label, slot.captionShort] as const))
	);
	/** Reader arms in the mirror table's column order, named by `captionShort`. */
	const mirrorColumns = $derived(
		(decomposition?.bandingSensitivity.variants[0]?.arms ?? []).map((arm) => ({
			label: arm.label,
			short: shortByLabel.get(arm.label) ?? arm.label,
			display: displayByLabel.get(arm.label) ?? arm.label
		}))
	);

	function polyline(arm: ApDecompositionArm): string {
		return arm.cumulativePts
			.map((value, index) => `${r2(bandCentre(index))},${r2(sy(value))}`)
			.join(' ');
	}

	/**
	 * Fan value labels sit beside their dot, but Gemma 4 26B and GLM-5 end 0.03 pts
	 * apart (~2 user units) and their labels would fuse. Walk the slots left to
	 * right and push each label down to clear the previous one — derived from the
	 * data, so no per-arm offset is hard-coded.
	 */
	const LABEL_LIFT = 6;
	const LABEL_MIN_GAP = 12;
	const fanRows = $derived.by(() => {
		let previous = Number.NEGATIVE_INFINITY;
		return slotArms.map(({ slot, arm }, index) => {
			const dotY = sy(arm.totalPts);
			const labelY = Math.max(dotY - LABEL_LIFT, previous + LABEL_MIN_GAP);
			previous = labelY;
			return {
				slot,
				arm,
				x: fanCentre(index),
				dotY,
				labelY,
				highY: sy(arm.ci95HighPts),
				lowY: sy(arm.ci95LowPts),
				color: slot.stroke,
				value: `${signed(arm.totalDeltaAp, 4)} · ${share(arm.pArmGreater)}`,
				// Rendered as a separate muted tspan on exactly the hollow slots, so the
				// caveat reads as a caveat rather than as part of the value.
				crosses: ZERO_NOTE[arm.standing]
			};
		});
	});

	// ---- caption numbers, all derived ---------------------------------------
	/**
	 * The `?? 0` / `: 0` fallbacks in this block are UNREACHABLE, not defaults: the
	 * template's own `{#if}` gates the whole figure when `decomposition === null`,
	 * so no zero here can reach the screen as a measurement. They exist because a
	 * `$derived` at module scope cannot narrow on that gate. If the gate is ever
	 * loosened, type these null and let the compiler find the render sites — a
	 * printed 0 in this figure is an average precision of zero, not a missing one.
	 */
	const referenceAp = $derived(decomposition?.referenceAveragePrecision ?? 0);
	const headroomPts = $derived((1 - referenceAp) * 100);
	const spreadPts = $derived(decomposition ? topLlmBandSpreadPts(decomposition) : 0);
	const errorRatePct = $derived(
		decomposition ? (decomposition.nFalse / decomposition.nStatements) * 100 : 0
	);
	const deltaRoll = $derived(
		slotArms
			.map(({ slot, arm }) => `${slot.captionShort} ${signed(arm.totalDeltaAp, 4)}`)
			.join(', ')
	);
	// THREE CLASSES, NOT A BOOLEAN. `clearsZero` was `low > 0 || high < 0` — true for
	// an interval entirely BELOW zero — so `clearing` would have counted a decisive
	// LOSS as an arm that cleared, and `notClearing` would have called it a tie. That
	// is the same sign-blind predicate this project retired after six occurrences,
	// surviving here under another name. Both now select on the loader's class.
	const clearing = $derived(
		slotArms.filter(({ arm }) => arm.standing === 'ahead' && arm.totalPts > 0)
	);
	const notClearing = $derived(
		slotArms.filter(({ slot, arm }) => slot.group === 'top-llm' && arm.standing === 'not-significant')
	);
	const behind = $derived(
		slotArms.filter(({ slot, arm }) => slot.group === 'top-llm' && arm.standing === 'behind')
	);
	const winners = $derived(slotArms.filter(({ slot }) => slot.group === 'top-llm'));
	/** True when no winning arm gives ground in any band — the diffuse-gain claim. */
	const winnersAgreeEverywhere = $derived(
		winners.length > 0 &&
			winners.every(({ arm }) => arm.nBandsAgreeingWithTotalSign === AP_DECOMP_BAND_COUNT)
	);
	/**
	 * Grammatical either way, so the caption states the diffuse-gain claim from the
	 * data instead of asserting it: "in every one of the 7 bands" today, a range if
	 * a future artifact stops agreeing everywhere.
	 */
	const winnersBandPhrase = $derived.by(() => {
		if (winners.length === 0) return `in none of the ${AP_DECOMP_BAND_COUNT} bands`;
		const counts = winners.map(({ arm }) => arm.nBandsAgreeingWithTotalSign);
		const low = Math.min(...counts);
		const high = Math.max(...counts);
		if (low === AP_DECOMP_BAND_COUNT) return `in every one of the ${AP_DECOMP_BAND_COUNT} bands`;
		const range = low === high ? `${low}` : `${low}–${high}`;
		return `in ${range} of the ${AP_DECOMP_BAND_COUNT} bands`;
	});
	const variantRow = $derived(slotArms.find(({ slot }) => slot.group === 'paper-variant'));
	const controlRow = $derived(slotArms.find(({ slot }) => slot.group === 'control'));
	/** Share of the control's whole loss that lands in the thinnest-evidence band. */
	const controlFirstBandShare = $derived(
		controlRow && controlRow.arm.totalPts !== 0
			? Math.abs(controlRow.arm.perBandNetPts[0] / controlRow.arm.totalPts)
			: 0
	);
	const firstBand = $derived(decomposition?.bands[0] ?? null);
	const lastBand = $derived(decomposition?.bands[AP_DECOMP_BAND_COUNT - 1] ?? null);
	/**
	 * The band-census note, built then MEASURED. Null when it would run past the
	 * viewBox edge — the note is a gloss on the count strip beside it, so dropping
	 * it costs a reader nothing the strip does not already show, whereas a note
	 * missing its last three characters is a number without its unit.
	 */
	const countNote = $derived.by<string | null>(() => {
		if (!firstBand || !lastBand) return null;
		const note = `${share(firstBand.errorRate)} of the single-evidence statements are wrong, ${share(lastBand.errorRate)} at ${lastBand.evidenceLow}+`;
		return apDecompCountNoteFits(note) ? note : null;
	});
	const topRow = $derived(slotArms[0]);
	/** How many times wider the top arm's interval is than its point estimate. */
	const topIntervalRatio = $derived(
		topRow ? (topRow.arm.ci95HighPts - topRow.arm.ci95LowPts) / topRow.arm.totalPts : 0
	);
	/** How many times narrower the paper variant's interval is than the top arm's. */
	const variantNarrowness = $derived(
		topRow && variantRow
			? (topRow.arm.ci95HighPts - topRow.arm.ci95LowPts) /
					(variantRow.arm.ci95HighPts - variantRow.arm.ci95LowPts)
			: 0
	);
	const glmRow = $derived(slotArms.find(({ slot }) => slot.label === 'GLM-5'));
	const thirdRow = $derived(slotArms.find(({ slot }) => slot.label === 'Gemma 4 31B'));
	const nearestGap = $derived(
		glmRow && thirdRow ? Math.abs(glmRow.arm.totalDeltaAp - thirdRow.arm.totalDeltaAp) : 0
	);
	const sensitivity = $derived(decomposition?.bandingSensitivity ?? null);
	/** The drawn arm's tilt under the reference's own banding, and under its own. */
	const drawnTilt = $derived.by(() => {
		if (!sensitivity) return null;
		const pick = (key: string) =>
			sensitivity.variants
				.find((variant) => variant.key === key)
				?.arms.find((arm) => arm.label === sensitivity.drawnArm) ?? null;
		return {
			reference: pick('reference_own_score'),
			mirror: pick('drawn_arm_own_score'),
			display: displayByLabel.get(sensitivity.drawnArm) ?? sensitivity.drawnArm
		};
	});
	/**
	 * "reverses to" is a claim about the two tilts, so it is read off them. If a
	 * future artifact stops reversing, the caption says "moves to" instead of
	 * asserting something the strip beside it contradicts.
	 */
	const mirrorPhrase = $derived.by(() => {
		if (!drawnTilt?.reference || !drawnTilt?.mirror) return '';
		const verb =
			Math.sign(drawnTilt.reference.tiltPts) !== Math.sign(drawnTilt.mirror.tiltPts)
				? 'reverses to'
				: 'moves to';
		return `${verb} ${signed(drawnTilt.mirror.tiltPts / 100, 4)}`;
	});
	const cogexDelta = $derived(
		data.status === 'ok'
			? (data.arms.find((arm) => arm.id === 'indra-cogex-hybrid')?.delta?.ap.delta ?? null)
			: null
	);
</script>

<section class="ap-decomp" aria-labelledby="ap-decomp-title">
	{#if data.status !== 'ok' || decomposition === null || mirror === null || !fanNamesFit}
		<div class="gate" role="status">
			<p class="eyebrow">where the difference comes from</p>
			<h2 id="ap-decomp-title">This breakdown is unavailable</h2>
			<p>
				{data.status !== 'ok'
					? data.reason
					: !fanNamesFit
						? 'two model names at the right-hand edge would overlap, and those names are this figure’s only key.'
						: 'the breakdown is missing, or the strip that justifies its grouping does not fit its own fixed geometry.'}
			</p>
			<code>{data.artifact_path}</code>
		</div>
	{:else}
		<h2 id="ap-decomp-title" class="visually-hidden">Where the point comes from</h2>
		<figure>
			<div class="scroller">
				<svg
					viewBox="0 0 900 644"
					preserveAspectRatio="xMidYMid meet"
					role="img"
					aria-labelledby="ap-decomp-chart-title ap-decomp-chart-desc"
				>
					<title id="ap-decomp-chart-title"
						>Running difference from RF 2k-d13 + Type/#PMIDs/promoter across seven groups of
						statements sorted by how much evidence they carry, with a strip showing how the same
						breakdown behaves when the statements are grouped four other ways</title
					>
					<desc id="ap-decomp-chart-desc"
						>Five lines, one per model, each with its own shade and dash pattern, run left to right
						across {AP_DECOMP_BAND_COUNT} groups of statements sorted by how many pieces of evidence
						each one carries — from the {decomposition.bands[0].n} statements carrying exactly one
						piece to the {decomposition.bands[AP_DECOMP_BAND_COUNT - 1].n} carrying
						{decomposition.bands[AP_DECOMP_BAND_COUNT - 1].evidenceLow} or more. Each line is the
						running total of how much that model adds to or subtracts from average precision within
						each group, compared with the random forest the 2023 paper published and whose released
						code we re-ran. The three strongest reading models rise steadily and end at
						{signed(slotArms[0].arm.totalDeltaAp, 4)}, {signed(slotArms[1].arm.totalDeltaAp, 4)} and
						{signed(slotArms[2].arm.totalDeltaAp, 4)} average precision; the same forest with one
						feature block swapped stays flat near zero; and the smallest reading model, Gemma 4 E2B,
						falls fastest where evidence is thinnest and ends at {signed(
							slotArms[AP_DECOMP_FAN_SLOTS.length - 1].arm.totalDeltaAp,
							4
						)}. The five end points are repeated at the right on the same scale with their 95%
						intervals from paired resampling; those intervals apply to the totals only. Beneath, a
						strip of four rows shows the same breakdown of {drawnTilt?.display}, summarised as one
						bar for the first groups and one for the last, under four different ways of grouping the
						statements: group them by either of the two scores being compared and the bars point
						opposite ways, group them by anything outside those two scores and they nearly vanish.</desc
					>

					<!-- title block (y = 0..48) -->
					<text class="fig-title" x={BODY_LEFT} y="20">Where the point comes from</text>
					<text class="fig-subtitle" x={BODY_LEFT} y="37"
						>Statements are grouped by how much evidence they carry — a count fixed before any model
						ran.</text
					>

					<!-- y axis: fixed, un-broken, zero-anchored, shared with the fan -->
					<text class="y-title" x="16" y="200" transform="rotate(-90 16 200)"
						>average precision</text
					>
					<text class="y-dir" x="16" y="92" transform="rotate(-90 16 92)">better ↑</text>
					<text class="y-dir" x="16" y="308" transform="rotate(-90 16 308)">↓ worse</text>
					{#each AP_DECOMP_Y_TICKS as tick (tick)}
						<line class="tick-mark" x1={BODY_LEFT - 4} y1={r2(sy(tick))} x2={BODY_LEFT} y2={r2(sy(tick))} />
						<text class="tick" x={BODY_LEFT - 8} y={r2(sy(tick)) + 3}>{tickLabel(tick)}</text>
					{/each}

					<!-- the gap hairline: the fan is dodged endpoints, not a second panel -->
					<line
						class="separator"
						x1={SEPARATOR_X}
						y1={PLOT_TOP + 6}
						x2={SEPARATOR_X}
						y2={PLOT_BOTTOM}
					/>

					<!-- zero rule: one continuous line through body, gap and fan -->
					<line class="zero" x1={BODY_LEFT} y1={r2(zeroY)} x2={FAN_RIGHT} y2={r2(zeroY)} />
					<!-- BELOW the rule: every winning arm now runs just above zero from the
					     first band on, so the old position sat underneath three traces. -->
					<text class="zero-label" x={BODY_LEFT + 2} y={r2(zeroY) + 11}
						>RF 2k-d13 + Type/#PMIDs/promoter</text
					>
					<text class="zero-label" x={BODY_LEFT + 2} y={r2(zeroY) + 19}
						>everything is compared with this · average precision {referenceAp.toFixed(4)}</text
					>

					<!-- five cumulative lines; no markers, no ribbon, no shading -->
					{#each drawOrder as { slot, arm } (slot.label)}
						<polyline
							class="series"
							points={polyline(arm)}
							stroke={slot.stroke}
							stroke-width={slot.strokeWidth}
							stroke-dasharray={slot.dash}
						/>
					{/each}

					<!-- two annotations, both leadered, both muted, both derived -->
					{#if winnersAgreeEverywhere}
						<text class="annotation annotation-end" x="556" y="80"
							>no group where a reading model gives ground</text
						>
						<line class="leader" x1={r2(bandCentre(5))} y1="84" x2={r2(bandCentre(5))} y2="117" />
					{/if}
					{#if controlRow && firstBand}
						<text class="annotation annotation-end" x="548" y="292"
							>{controlRow.slot.display} loses {(controlFirstBandShare * 100).toFixed(0)}% of its
							total</text
						>
						<text class="annotation annotation-end" x="548" y="302"
							>on the {firstBand.n} single-evidence statements</text
						>
						<!-- starts clear of the right-anchored text block (which begins near
						     x=356 at 8px mono) and stops just under the trace it names -->
						<line
							class="leader"
							x1="348"
							y1="294"
							x2={r2(bandCentre(1))}
							y2={r2(sy(controlRow.arm.cumulativePts[1]) + 3)}
						/>
					{/if}

					<!-- band ticks + count strip (y = 330..382) -->
					{#each decomposition.bands as band, index (band.display)}
						<text class="band-tick" x={r2(bandCentre(index))} y="325">{band.display}</text>
						<text class="count" x={r2(bandCentre(index))} y="353">{band.nTrue}</text>
						<text class="count" x={r2(bandCentre(index))} y="379">{band.nFalse}</text>
					{/each}
					<text class="count-label" x={BODY_LEFT} y="341">correct in group</text>
					<text class="count-label" x={BODY_LEFT} y="367">wrong in group</text>
					<!-- Left-anchored beside the viewBox edge, so an overrun clips its
					     trailing glyphs; the string is built and measured first. -->
					{#if countNote}
						<text class="count-note" x={GAP_RIGHT} y="379">{countNote}</text>
					{/if}
					<text class="axis-title" x={(BODY_LEFT + BODY_RIGHT) / 2} y="402"
						>evidence entries per statement &#8594;</text
					>

					<!-- terminus fan: the five lines' own endpoints, dodged -->
					<text class="fan-header" x={(GAP_RIGHT + FAN_RIGHT) / 2} y="57"
						>total, 95% interval</text
					>
					{#each fanRows as row (row.slot.label)}
						<!-- the leader carries the trace's own dash, so following a line to
						     its name needs no colour memory and no legend -->
						<line
							class="fan-leader"
							x1={r2(bandCentre(AP_DECOMP_BAND_COUNT - 1))}
							y1={r2(row.dotY)}
							x2={r2(row.x)}
							y2={r2(row.dotY)}
							stroke={row.color}
							stroke-dasharray={row.slot.dash}
						/>
						<line
							class="whisker"
							x1={r2(row.x)}
							y1={r2(row.highY)}
							x2={r2(row.x)}
							y2={r2(row.lowY)}
							stroke={row.color}
						/>
						<line
							class="whisker-cap"
							x1={r2(row.x) - 3.5}
							y1={r2(row.highY)}
							x2={r2(row.x) + 3.5}
							y2={r2(row.highY)}
							stroke={row.color}
						/>
						<line
							class="whisker-cap"
							x1={r2(row.x) - 3.5}
							y1={r2(row.lowY)}
							x2={r2(row.x) + 3.5}
							y2={r2(row.lowY)}
							stroke={row.color}
						/>
						<circle
							class="dot"
							class:hollow={row.arm.standing !== 'ahead'}
							cx={r2(row.x)}
							cy={r2(row.dotY)}
							r="2.5"
							fill={row.arm.standing === 'ahead' ? row.color : 'var(--paper)'}
							stroke={row.color}
						/>
						{#each row.slot.shortLines as line, index (line)}
							<text class="fan-name" x={r2(row.x)} y={341 + index * 9}>{line}</text>
						{/each}
					{/each}
					<!-- Value labels last, so their halo clears the NEXT slot's whisker. -->
					{#each fanRows as row (row.slot.label)}
						<text class="fan-value" x={r2(row.x) + 8} y={r2(row.labelY)}
							>{row.value}{#if row.crosses}<tspan class="fan-crosses">{row.crosses}</tspan>{/if}</text
						>
					{/each}

					<!-- ==== banding-sensitivity strip: the mirror, shown not footnoted ==== -->
					<line class="panel-rule" x1={BODY_LEFT} y1="424" x2={FAN_RIGHT} y2="424" />
					<text class="fig-title panel-title" x={BODY_LEFT} y="442"
						>Why the groups are a count of evidence, not a score</text
					>
					<text class="fig-subtitle" x={BODY_LEFT} y="456"
						>Same model ({drawnTilt?.display}); head = the first {sensitivity?.headBands} tenths, tail
						= the last {sensitivity?.tailBands}.</text
					>
					<rect class="legend-swatch head" x="676" y="449" width="10" height="7" />
					<text class="legend-text" x="690" y="456">head</text>
					<rect class="legend-swatch tail" x="726" y="449" width="10" height="7" />
					<text class="legend-text" x="740" y="456">tail</text>

					<g transform="translate(0 {MIRROR_TOP})">
						<line
							class="mirror-zero"
							x1={mirror.zeroX}
							y1="4"
							x2={mirror.zeroX}
							y2={mirror.height - 2}
						/>
						{#each mirror.rows as row (row.key)}
							{#if row.groupHeader}
								<text
									class="mirror-group"
									x={AP_DECOMP_MIRROR_GEOMETRY.labelAnchorX}
									y={r2(row.groupHeader.y)}>{row.groupHeader.display}</text
								>
							{/if}
							<text
								class="mirror-label"
								class:drawn={row.drawn}
								x={AP_DECOMP_MIRROR_GEOMETRY.labelAnchorX}
								y={r2(row.labelY)}>{row.display}</text
							>
							{#if row.drawn}
								<!-- marks the one row whose banding the waterfall above uses -->
								<text class="mirror-caret" x={AP_DECOMP_MIRROR_GEOMETRY.labelAnchorX + 8} y={r2(row.labelY)}
									>&#9656;</text
								>
							{/if}
							<rect
								class="mirror-bar head"
								class:endogenous={row.kind === 'endogenous'}
								x={r2(row.head.x)}
								y={r2(row.head.y)}
								width={r2(row.head.width)}
								height={AP_DECOMP_MIRROR_GEOMETRY.barHeight}
							/>
							<rect
								class="mirror-bar tail"
								class:endogenous={row.kind === 'endogenous'}
								x={r2(row.tail.x)}
								y={r2(row.tail.y)}
								width={r2(row.tail.width)}
								height={AP_DECOMP_MIRROR_GEOMETRY.barHeight}
							/>
							<text
								class="mirror-readout"
								x={AP_DECOMP_MIRROR_GEOMETRY.readoutX}
								y={r2(row.labelY)}>{row.readout}</text
							>
						{/each}
						{#each mirror.ticks as tick (tick)}
							<text
								class="mirror-tick"
								x={r2(mirror.zeroX + tick * MIRROR_UNITS_PER_PT)}
								y={r2(mirror.height + 12)}>{tickLabel(tick)}</text
							>
						{/each}
						<text class="mirror-axis" x={AP_DECOMP_MIRROR_GEOMETRY.labelAnchorX} y={r2(mirror.height + 12)}
							>net change in average precision</text
						>
					</g>
				</svg>
			</div>

			<figcaption>
				<p>
					Where each line ends <em>is</em> that model's whole margin in average precision
					({deltaRoll}) — the groups add up to it exactly, so the total is reached rather than
					asserted. Grouped by amount of evidence,
					{#each winners as row, index (row.slot.label)}{index > 0 ? ', ' : ''}{row.slot
							.display}{/each}
					gain {winnersBandPhrase}. The whiskers on the right are 95% intervals on the totals alone:
					{#each clearing as row, index (row.slot.label)}{index > 0 ? ' and ' : ''}{row.slot
							.display}{/each}
					stay clear of zero,
					{#each notClearing as row (row.slot.label)}{row.slot.display}{/each} does not.
				</p>

				{#if drawnTilt?.reference && drawnTilt?.mirror}
					<p>
						The strip beneath is why the groups are a count of evidence and not a score. Group the
						same breakdown by the <em>random forest's</em> own score and it leans
						{signed(drawnTilt.reference.tiltPts / 100, 4)} toward one end; group it by the reading
						model's <em>own</em> score and the lean {mirrorPhrase}. That is regression to the mean on
						whichever score the groups were cut on — an artefact of the grouping, not a finding.
					</p>
				{/if}

				<details class="method">
					<summary>how this is computed</summary>
					<p>
						Average precision splits up cleanly, with nothing left over. Each correct statement
						contributes the precision at the cut-point that admits it and everything tied with it,
						divided by {decomposition.nTrue}, and those {decomposition.nTrue} contributions add up to
						average precision exactly. Here each model's contributions are subtracted from those of
						RF 2k-d13 + Type/#PMIDs/promoter — the random forest run from the code
						released with the 2023 paper — and accumulated left to right across the {AP_DECOMP_BAND_COUNT} groups,
						so where each line ends <em>is</em> that model's whole margin: reached, not asserted.
						Checked against scikit-learn: {referenceAp.toFixed(10)} either way, and every model's
						group-by-group sum reproduces the shipped file ({deltaRoll}).
					</p>
					<p>
						The statements are grouped by {decomposition.banding.variable}: a plain count of the
						corpus, fixed before any model ran. It is checked statement by statement against the
						shared gold's own evidence count and against the sum of the counts the 2023 paper
						released — all three agree on {decomposition.banding.nStatementsAgreeing} of {decomposition.nStatements}.
						The group boundaries double (1, 2, 3–4, … , {lastBand?.evidenceLow}+) because belief
						under a noisy-OR climbs fast and then flattens as evidence accumulates, and because
						group membership is then decided by the count alone: every statement carrying the same
						amount of evidence lands in the same group, so {decomposition.nAssignedByTieBreak}
						statements are placed by a tie-break. Equal-sized groups could not say that —
						{firstBand?.n} statements carry exactly one piece of evidence, so a boundary would have to
						cut that block arbitrarily. One scope note: the reading models were run over
						{grouped(decomposition.banding.nUniquePairs)} unique statement-evidence pairs while this
						count includes all {grouped(decomposition.banding.nEvidenceEntries)} entries, which is
						what the released counts do; {decomposition.banding
							.nStatementsChangingBandUnderUniquePairs} statements would move group under the other choice.
					</p>
					{#if sensitivity}
						<p>
							The strip below is drawn, not asserted. Lean is the first tenths minus the last tenths
							— one signed number in average precision for which end of the grouping variable a model
							appears to win at — computed from equal-sized tenths of each candidate variable.
						</p>
						<div class="table-scroll">
							<table class="tilt">
								<caption
									>Lean (first tenths − last tenths) in average precision, by grouping variable and
									reading model</caption
								>
								<thead>
									<tr>
										<th scope="col">grouped by</th>
										{#each mirrorColumns as column (column.label)}
											<th scope="col" title={column.display}>{column.short}</th>
										{/each}
									</tr>
								</thead>
								<tbody>
									{#each sensitivity.variants as variant (variant.key)}
										<tr class:endogenous={variant.kind === 'endogenous'}>
											<th scope="row"
												><!-- &nbsp; because Svelte trims the span's leading space -->
												{variant.display}{#if variant.drawn}<span class="drawn-tag"
														>&nbsp;· drawn above</span
													>{/if}</th
											>
											{#each variant.arms as arm (arm.label)}
												<td>{signed(arm.tiltPts / 100, 4)}</td>
											{/each}
										</tr>
									{/each}
								</tbody>
							</table>
						</div>
						<p>
							The first two rows group by one of the very scores this figure is comparing, and their
							leans run as far as {(sensitivity.maxAbsTiltEndogenousPts / 100).toFixed(4)} average
							precision, always in whichever direction the grouping score points. The last two group
							by something outside both scores, and every lean collapses to at most {(
								sensitivity.maxAbsTiltExogenousPts / 100
							).toFixed(4)}. A boundary falling inside a block of tied scores has to land somewhere,
							so every cell was recomputed with that block ordered both extreme ways; the worst
							movement is {(sensitivity.maxTieBreakSpreadPts / 100).toFixed(4)} average precision, an
							order of magnitude smaller than the reversal it might have explained. The unfitted
							noisy-OR appears as a diagnostic row rather than as the grouping actually used, for a
							reason this file checks: reading can only take evidence away, so
							{decomposition.nReaderBeliefsExceedingNoisyOr} of
							{grouped(decomposition.nReaderBeliefComparisons)} reading-model beliefs come out above
							it. It is each reading model's own ceiling, and grouping by it would mean grouping by
							part of the reading model's own score.
						</p>
					{/if}
					<p>
						The measure here is average precision over all statements at once, computed step-wise so
						that tied scores earn no interpolated credit. It is not AUROC, and it is not the
						straight-line precision-recall area the 2023 paper used, which flatters exactly these
						heavily tied language-model scores. The end dots are the measured margins; the whiskers
						are 95% intervals from resampling the statements
						({grouped(decomposition.nBootstrap)} resamples, seed {decomposition.seed}, every model
						scored on the same resample each time, over the identical
						{decomposition.nStatements} statements with the published labels). The whiskers
						apply to the totals alone — no point along a line has its own interval.
					</p>
					<p>
						{#each clearing as row, index (row.slot.label)}{index > 0 ? ' and ' : ''}{row.slot
							.display}
						({signed(row.arm.totalDeltaAp, 4)}){/each} stay clear of zero;
						{#each notClearing as row (row.slot.label)}{row.slot.display} ({signed(
						row.arm.totalDeltaAp,
						4
						)}){/each} does not — though it still lands above the random forest in {share(
						(notClearing[0] ?? clearing[0]).arm.pArmGreater
						)} of the resampled re-runs. 95% is a convention, not a cliff, and the two differ by only {nearestGap.toFixed(
						3
						)} average precision. For scale: a margin of 0.01 average precision on top of {referenceAp.toFixed(
						4
						)} is about a sixth of the {(headroomPts / 100).toFixed(
						4
						)} that is left to gain at all, and the leading model's interval is {topIntervalRatio.toFixed(
						1
						)} times as wide as the margin it surrounds. These intervals are not widened to account for
						comparing seven models against one; widen them any of the usual ways and neither of the
						two that only just clear zero survives.
					</p>
					{#if variantRow}
						<p>
						<!-- display, never .label: that string is a frozen point_metrics join key. -->
							The flat line hugging zero is the same random forest with one feature block swapped
							({variantRow.slot.display}, {signed(
							variantRow.arm.totalDeltaAp,
							4
							)} [{signed(variantRow.arm.ci95LowPts / 100, 4)}, {signed(
							variantRow.arm.ci95HighPts / 100,
							4
							)}]). Because it is the same forest with one change, it scores almost identically to the
							model it is compared against, and its interval is about {variantNarrowness.toFixed(1)}
							times narrower than the language models' for that reason alone. Read it as a ruler for
							how small an internal change looks, not as a floor on the noise. It is also why it could
							never have caught the grouping problem: tied to the comparison model, it draws flat
							however the statements are grouped. The three best language models share one colour
							family because they stay within {(spreadPts / 100).toFixed(4)} average precision of each
							other in every group; each is told apart by its dash pattern and by the name at the end
							of its line.
						</p>
					{/if}
					<p>
						The groups say where the difference lands, not what caused it: a group's contribution can
						move because a correct statement in it rose <em>or</em> because a wrong statement fell
						past it, so no group is a count of statements a model fixed. And a group's figure is a
						sum, not a rate: the groups hold different numbers of correct statements, and correct
						statements are what carry average precision, which is why the counts under the chart are
						part of the figure rather than decoration.
						{#if controlRow}
							{controlRow.slot.display} is here as a control, not as a straw man: it is the same
							pipeline at a smaller model size, and the fact that the results line up with size
							across all four reading models is what makes this look like a dose-response rather
							than four unrelated draws.
						{/if}
					</p>
					<p>
						{#if cogexDelta !== null}
							INDRA CoGEx hybrid (margin {signed(cogexDelta, 4)}, the resampling-average figure the
							table above quotes rather than the single measured margin drawn here) is left out for
							legibility and appears in that table.
						{/if}
						One risk this figure cannot show: every model is scored against the same 2023 labels on
						the same {decomposition.nStatements} statements, so any noise or quirk in those labels
						moves all five lines together, and a one-point margin sits comfortably inside what a
						quirky gold set could manufacture on its own. {decomposition.nStatements} statements with
						{errorRatePct.toFixed(1)}% of them wrong is one collection; nothing here says what would
						happen on a different corpus.
					</p>
				</details>
			</figcaption>
		</figure>
		<footer>
			<code>{decomposition.referenceArm}</code> is what everything here is compared with ·
			<code>{data.artifact_path}</code>
		</footer>
	{/if}
</section>

<style>
	.ap-decomp {
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
	.panel-title {
		font-size: 12.5px;
	}
	.fig-subtitle {
		fill: var(--ink-muted);
		font-family: var(--serif);
		font-size: 9.5px;
		font-style: italic;
	}
	/* Which way is good. The zero rule already says what zero IS. */
	.y-dir {
		fill: var(--ink-faint);
		font-family: var(--mono);
		font-size: 7px;
		text-anchor: middle;
	}
	.y-title {
		fill: var(--ink-muted);
		font-family: var(--mono);
		font-size: 8.5px;
		text-anchor: middle;
	}
	.tick {
		fill: var(--ink-faint);
		font-family: var(--mono);
		font-size: 8px;
		text-anchor: end;
		font-variant-numeric: tabular-nums;
	}
	.tick-mark {
		stroke: var(--rule);
		stroke-width: 1;
	}
	.separator {
		stroke: var(--ink-muted);
		stroke-width: 1;
		opacity: 0.3;
	}
	.panel-rule {
		stroke: var(--rule);
		stroke-width: 1;
	}
	.zero {
		stroke: var(--ink);
		stroke-width: 2;
	}
	.zero-label {
		fill: var(--ink-muted);
		font-family: var(--mono);
		font-size: 7.5px;
		letter-spacing: 0.04em;
	}
	.series {
		fill: none;
		stroke-linejoin: round;
		stroke-linecap: round;
	}
	.annotation {
		fill: var(--ink-muted);
		font-family: var(--mono);
		font-size: 8px;
	}
	.annotation-end {
		text-anchor: end;
	}
	.leader {
		stroke: var(--ink-muted);
		stroke-width: 0.75;
	}
	.band-tick {
		fill: var(--ink-faint);
		font-family: var(--mono);
		font-size: 8.5px;
		text-anchor: middle;
	}
	.count {
		fill: var(--ink-muted);
		font-family: var(--mono);
		font-size: 9px;
		text-anchor: middle;
		font-variant-numeric: tabular-nums;
	}
	.count-label,
	.count-note,
	.fan-header {
		fill: var(--ink-faint);
		font-family: var(--mono);
		font-size: 7px;
		letter-spacing: 0.06em;
		text-transform: uppercase;
	}
	.count-note {
		text-transform: none;
		letter-spacing: 0.02em;
	}
	.fan-header {
		text-anchor: middle;
	}
	.axis-title {
		fill: var(--ink-muted);
		font-family: var(--mono);
		font-size: 8.5px;
		text-anchor: middle;
	}
	.fan-leader {
		stroke-width: 0.75;
	}
	.whisker {
		stroke-width: 1.5;
	}
	.whisker-cap {
		stroke-width: 1.5;
	}
	.dot {
		stroke-width: 0;
	}
	.dot.hollow {
		stroke-width: 2;
	}
	.fan-crosses {
		fill: var(--ink-muted);
	}
	.fan-value {
		fill: var(--ink);
		font-family: var(--mono);
		font-size: 8px;
		font-variant-numeric: tabular-nums;
		paint-order: stroke;
		stroke: var(--paper);
		stroke-width: 2.5;
		stroke-linejoin: round;
	}
	/* 7px keeps "paper's own" inside its 48-unit slot without touching its neighbours. */
	.fan-name {
		fill: var(--ink-muted);
		font-family: var(--mono);
		font-size: 7px;
		text-anchor: middle;
	}
	/* ---- banding-sensitivity strip ---- */
	.legend-swatch.head {
		fill: var(--ink);
	}
	.legend-swatch.tail {
		fill: var(--paper);
		stroke: var(--ink);
		stroke-width: 1;
	}
	.legend-text,
	.mirror-tick,
	.mirror-axis {
		fill: var(--ink-faint);
		font-family: var(--mono);
		font-size: 7px;
	}
	.mirror-tick {
		text-anchor: middle;
		font-variant-numeric: tabular-nums;
	}
	.mirror-axis {
		text-anchor: end;
		letter-spacing: 0.02em;
	}
	.mirror-group {
		fill: var(--ink-faint);
		font-family: var(--mono);
		font-size: 7px;
		text-anchor: end;
		letter-spacing: 0.06em;
		text-transform: uppercase;
	}
	.mirror-label {
		fill: var(--ink-muted);
		font-family: var(--mono);
		font-size: 8px;
		text-anchor: end;
	}
	.mirror-label.drawn {
		fill: var(--ink);
	}
	.mirror-caret {
		fill: var(--ink);
		font-size: 8px;
	}
	.mirror-zero {
		stroke: var(--ink);
		stroke-width: 1.5;
	}
	/* Fill carries head vs tail (solid vs open), so the pair survives greyscale;
	   the colour token carries endogenous vs exogenous, which the group headers
	   also state in words. */
	.mirror-bar.head {
		fill: var(--ink);
	}
	.mirror-bar.tail {
		fill: var(--paper);
		stroke: var(--ink);
		stroke-width: 1;
	}
	.mirror-bar.head.endogenous {
		fill: var(--blocked);
	}
	.mirror-bar.tail.endogenous {
		stroke: var(--blocked);
	}
	.mirror-readout {
		fill: var(--ink-muted);
		font-family: var(--mono);
		font-size: 8px;
		font-variant-numeric: tabular-nums;
	}
	figcaption {
		margin-top: 0.7rem;
		max-width: 74ch;
	}
	figcaption p,
	.method p {
		margin: 0 0 0.55rem;
		font-family: var(--serif);
		font-size: 0.8rem;
		line-height: 1.5;
		color: var(--ink-muted);
	}
	figcaption em,
	.method em {
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
	/* Wide content scrolls inside its own container; the page never scrolls. */
	.table-scroll {
		overflow-x: auto;
		margin: 0 0 0.6rem;
	}
	table.tilt {
		border-collapse: collapse;
		font-family: var(--mono);
		font-size: 0.62rem;
		color: var(--ink-muted);
	}
	table.tilt caption {
		caption-side: top;
		text-align: left;
		padding-bottom: 0.3rem;
		font-size: 0.6rem;
		color: var(--ink-faint);
		text-transform: uppercase;
		letter-spacing: 0.05em;
	}
	table.tilt th,
	table.tilt td {
		padding: 0.2rem 0.5rem;
		border-bottom: 1px solid var(--rule);
		white-space: nowrap;
	}
	table.tilt thead th {
		border-bottom: 1px solid var(--ink);
		font-weight: 400;
		color: var(--ink-faint);
		text-align: right;
	}
	table.tilt thead th:first-child,
	table.tilt tbody th {
		text-align: left;
		font-weight: 400;
	}
	table.tilt td {
		text-align: right;
		font-variant-numeric: tabular-nums;
	}
	table.tilt tr.endogenous th,
	table.tilt tr.endogenous td {
		color: var(--blocked);
	}
	.drawn-tag {
		color: var(--ink-faint);
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
