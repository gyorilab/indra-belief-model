<script lang="ts">
	import {
		PAPER_METRIC_LABELS,
		PAPER_DEFAULT_METRIC,
		PAPER_LITERAL_REFERENCE_ARM_ID,
		type PaperLiteralArm,
		type PaperLiteralAurocOnRanked,
		type PaperLiteralDelta,
		PAPER_DELTA_MARK as DELTA_MARK,
		PAPER_DELTA_STANDING_SENTENCE as DELTA_SR,
		type PaperLiteralLoad,
		type PaperMetric,
		type Standing
	} from '$lib/data/paper-literal';
	import { page } from '$app/state';

	let { data }: { data: PaperLiteralLoad } = $props();

	// The three cross-arm lenses, in the order shown. The paper's own trapezoidal
	// estimator leads because this panel is read by the paper's authors — but the
	// default VIEW is not the quoted margin: the number we quote stays tie-robust
	// average precision, the most CONSERVATIVE of the three (this panel is
	// saturated — majority-positive, reference already at ~0.94 AP), and the
	// concession below the lens says so in every view. None of the three settles
	// the comparison on its own; the statement-grain error-class margin is the
	// larger result and is reported in its own figure — under a disclosed oracle
	// (thresholds fitted AND evaluated on these same 1,689 statements, a rule that
	// favours the RF: 1,546 candidate cuts against a reader gate's 475–498), which
	// is why the size of that margin is quoted there and not here.
	const METRICS = ['trapezoidal', 'ap', 'auroc'] as const;

	function isMetric(value: string | null): value is PaperMetric {
		return value === 'ap' || value === 'auroc' || value === 'trapezoidal';
	}

	// Metric lens: client state so switching RE-MEASURES the same arms rather than
	// reloading; seeded from (and synced to) ?metric= so the lens is shareable.
	// Mirrors the frontier axis-lens (routes/frontier/+page.svelte).
	const requestedMetric = page.url.searchParams.get('metric');
	let metric = $state<PaperMetric>(isMetric(requestedMetric) ? requestedMetric : PAPER_DEFAULT_METRIC);
	function setMetric(m: PaperMetric) {
		metric = m;
		const u = new URL(page.url);
		u.searchParams.set('metric', m);
		history.replaceState(history.state, '', u); // update URL, no reload (keep the re-measure)
	}

	/** Signed fixed-decimal string: '+' prefix for non-negative, native '-' otherwise. */
	function signed(value: number, digits = 3): string {
		return `${value >= 0 ? '+' : ''}${value.toFixed(digits)}`;
	}

	// The AP margins are thousandths, so 3dp rounds the quoted number to a
	// different value than the concession note (and the memo) print. The Δ column
	// therefore carries FOUR decimals under the average-precision lens and three
	// under the others, which are on a coarser scale. One number, one rendering.
	const AP_DELTA_DIGITS = 4;
	const deltaDigits = $derived(metric === 'ap' ? AP_DELTA_DIGITS : 3);

	/** An arm that carries a paired delta (i.e. not the reference arm itself). */
	type DeltaArm = PaperLiteralArm & { delta: PaperLiteralDelta };

	const arms = $derived<PaperLiteralArm[]>(data.status === 'ok' ? data.arms : []);

	// The named arm in the concession below. Same rule as routes/paper: the LLM arm
	// with the largest ΔAP — the margin we quote, not the lens currently shown. It
	// must be named, because a different arm leads under trapezoidal and an
	// unscoped "our margin" claim would then be wrong.
	// `INDRA CoGEx hybrid` rides in the arm set under kind 'llm', but it is INDRA's
	// deployed non-reader baseline, not a fifth reader — it emits far more distinct
	// scores, so including it would misstate the reader tie range that explains the
	// trapezoidal/AP gap. It stays in the table, and is excluded from the reader
	// tie range AND from the `best` reader headline below.
	const NON_READER_LLM_ARM_ID = 'indra-cogex-hybrid';

	const best = $derived<DeltaArm | null>(
		arms
			// NON_READER_LLM_ARM_ID is carried under kind 'llm' for layout only (see
			// paper-literal.ts); it is INDRA's own hybrid, not a reader, so it must
			// not be eligible for the reader headline. Today it has the worst delta,
			// which masked this — that is data luck, not a guarantee.
			.filter(
				(arm): arm is DeltaArm =>
					arm.kind === 'llm' && arm.id !== NON_READER_LLM_ARM_ID && arm.delta !== null
			)
			.slice()
			.sort((a, b) => b.delta.ap.delta - a.delta.ap.delta)[0] ?? null
	);

	/** The paired-delta reference: the paper's literal model, and the tie contrast. */
	const reference = $derived(arms.find((arm) => arm.id === PAPER_LITERAL_REFERENCE_ARM_ID) ?? null);

	/** How many arms the ● marks are drawn across — the multiplicity the CIs ignore. */
	const deltaArmCount = $derived(arms.filter((arm) => arm.delta !== null).length);
	// The ● is LENS-DEPENDENT: an arm can clear zero on the paper's trapezoidal
	// estimator and not on the tie-robust average-precision lens (31B does exactly
	// that). Both counts are stated so the default view cannot imply a margin the
	// conservative lens does not support.
	// This count used to be one number over a sign-blind `excludesZero`, which
	// lumped significant LOSSES in with significant wins. Two arms here are
	// significantly BEHIND the reference on average precision — Gemma 4 E2B at
	// −0.0159 and the INDRA CoGEx hybrid at −0.0183 — and both were being marked,
	// bolded, stripe-accented and counted exactly like the arms that beat it, under
	// a footer reading "4 arms clear zero". That boolean is gone: the loader now
	// carries a three-way `standing` per lens, so direction is a class here and not
	// a conjunction someone has to remember to write.
	const at = (arm: (typeof arms)[number], lens: PaperMetric, standing: Standing) =>
		arm.delta !== null && arm.delta[lens].standing === standing;
	const nAheadHere = $derived(arms.filter((arm) => at(arm, metric, 'ahead')).length);
	const nBehindHere = $derived(arms.filter((arm) => at(arm, metric, 'behind')).length);
	const nAheadAp = $derived(arms.filter((arm) => at(arm, 'ap', 'ahead')).length);
	const nBehindAp = $derived(arms.filter((arm) => at(arm, 'ap', 'behind')).length);
	// The paired deltas are all measured against ONE reference, which is not
	// necessarily the strongest paper-side arm. If a stronger one exists, the
	// headline margin is quoted against a weaker baseline than it could be, and
	// the page must say so rather than let "the paper's model" stand for the best.
	const strongestPaper = $derived(
		arms
			.filter((arm) => arm.kind === 'paper' || arm.kind === 'port')
			.reduce((top, arm) => (top === null || arm.ap > top.ap ? arm : top), null as (typeof arms)[number] | null)
	);
	const marginVsStrongestPaper = $derived(
		best && strongestPaper ? best.ap - strongestPaper.ap : null
	);


	/**
	 * What the badge SAYS, which is not the same thing as what `kind` IS. `kind` is
	 * a layout axis — it drives one hue per family in paperArmColorVar — and the
	 * CoGEx arm rides under 'llm' for that reason, as paper-literal.ts documents.
	 * The badge, though, is read as a claim about the arm, and printing `LLM` beside
	 * INDRA's own hybrid belief model asserts something false: that arm reads no
	 * text. Only the printed word changes here; `kind` itself is left alone.
	 */
	const KIND_BADGE: Record<PaperLiteralArm['kind'], string> = {
		paper: 'published code',
		port: 'independent rewrite',
		llm: 'language model'
	};
	const KIND_TITLE: Record<PaperLiteralArm['kind'], string> = {
		paper: 'run here with the code released with the 2023 paper',
		port: 'an independent rewrite of that method, checked against the released code',
		llm: 'a language model reading the evidence sentences'
	};
	function kindBadge(arm: PaperLiteralArm): string {
		return arm.id === NON_READER_LLM_ARM_ID ? 'reads no text' : KIND_BADGE[arm.kind];
	}
	function kindTitle(arm: PaperLiteralArm): string {
		return arm.id === NON_READER_LLM_ARM_ID
			? "INDRA's own belief model — drawn with the language models so the colours stay in families, but it reads no text"
			: KIND_TITLE[arm.kind];
	}

	const readerDistinctScores = $derived(
		arms
			.filter((arm) => arm.kind === 'llm' && arm.id !== NON_READER_LLM_ARM_ID)
			.map((arm) => arm.distinctScores)
	);

	/**
	 * THE AUROC LENS IS NOT A RANKING RESULT ON ITS OWN, and it is the largest
	 * margin on this page, so it cannot be shown bare.
	 *
	 * A reader gate emits exactly 0 for a statement whose evidence it rejected
	 * outright — one tied block, not a low rank. AUROC over the whole panel pays
	 * for that binary split: every ranked statement outranks every zeroed one. On
	 * this panel the split is nearly the entire reader margin. Held to the
	 * statements an arm actually orders, with the paper's RF scored on those SAME
	 * statements, three of the four reader arms rank WORSE than the RF and the
	 * fourth leads by a fraction of its panel-wide gain. Every number below is
	 * server-computed from the shipped prediction joins (`aurocOnRanked`) — the
	 * component asserts nothing.
	 */
	type RankedArm = PaperLiteralArm & { aurocOnRanked: PaperLiteralAurocOnRanked };
	type RankedDeltaArm = RankedArm & { delta: PaperLiteralDelta };

	const rankedReaders = $derived<RankedArm[]>(
		arms.filter(
			(arm): arm is RankedArm =>
				arm.kind === 'llm' && arm.id !== NON_READER_LLM_ARM_ID && arm.aurocOnRanked !== null
		)
	);
	/** Reader arms that rank worse than the paper's RF on their own ranked block. */
	const readersBehindOnRanked = $derived(
		rankedReaders.filter((arm) => arm.aurocOnRanked.armAuroc < arm.aurocOnRanked.referenceAuroc)
			.length
	);
	/**
	 * The arm the qualification names: the reader that leads THIS lens, not the
	 * ΔAP leader. Naming the AP leader here would qualify a number the lens is not
	 * showing — and on this panel the AUROC leader is also the one arm that does
	 * hold a ranked-block edge, so quoting it is the least flattering choice
	 * available, which is the right one.
	 */
	const aurocLeader = $derived<RankedDeltaArm | null>(
		rankedReaders
			.filter((arm): arm is RankedDeltaArm => arm.delta !== null)
			.slice()
			.sort((a, b) => b.delta.auroc.delta - a.delta.auroc.delta)[0] ?? null
	);

	function f3(value: number): string {
		return value.toFixed(3);
	}
	function count(value: number): string {
		return value.toLocaleString('en-US');
	}

	/**
	 * Tooltip for the per-row ranked-block figure. It carries the one thing the
	 * two-number cell cannot: each arm zeroes a DIFFERENT block, so these figures
	 * are comparable to the reference number beside them and to nothing else.
	 */
	function rankedTitle(arm: PaperLiteralArm): string {
		const ranked = arm.aurocOnRanked;
		if (ranked === null) return 'this check did not load for this model';
		const referenceName = reference?.display ?? 'the random forest we compare against';
		if (ranked.nZeroed === 0) {
			return `this model gives every statement a score above zero — all ${count(ranked.nRanked)} of them are in its ranking`;
		}
		return `AUROC over only the ${count(ranked.nRanked)} statements this model scores above zero (${f3(ranked.armAuroc)}), beside ${referenceName} scored on those same statements (${f3(ranked.referenceAuroc)}). Each model throws out a different set of statements, so this number can be compared with the one next to it and with no other model's.`;
	}
</script>

<section class="paper-literal" aria-labelledby="paper-literal-title">
	{#if data.status !== 'ok'}
		<div class="gate" role="status">
			<p class="eyebrow">the code released with the 2023 paper · every method on the same statements</p>
			<h2 id="paper-literal-title">This comparison could not be loaded</h2>
			<p>{data.reason}</p>
			<code>{data.artifact_path}</code>
		</div>
	{:else}
		<header>
			<div>
				<p class="eyebrow">the code released with the 2023 paper vs language models · the same statements for both</p>
				<h2 id="paper-literal-title">Every method, scored one measure at a time</h2>
			</div>
			<!-- the badge names the measure we quote AS one of three; it is the least
			     flattering of them, not the number that settles the page.
			     FIT, measured rather than assumed (headless Chromium, this header's own
			     CSS): the measured string was 37 characters and rendered 249.8px on one
			     line; this one is 36, so it is narrower still. The badge is
			     `flex: 0 0 auto` — content-sized, NOT a fixed gutter — so it is never
			     asked to fit a smaller box. scrollWidth == clientWidth (248px) at every
			     width from 1440 down to 320; at 721px it sits inside a 673px header with
			     the left column shrinking, and at 720px and below the media query below
			     stacks it on its own line inside a 272px header. No ancestor sets
			     overflow:hidden, white-space:nowrap or text-overflow, so no glyph can be
			     dropped silently. If this string grows past ~40 characters, re-measure. -->
			<strong>least flattering · {PAPER_METRIC_LABELS.ap}</strong>
		</header>

		<!-- metric lens: the switch that re-measures the same arms -->
		<div class="metric-switch" role="group" aria-label="choose which measure to score every model by">
			<span class="mx-label">measure by</span>
			{#each METRICS as m}
				<button
					type="button"
					class:on={metric === m}
					aria-pressed={metric === m}
					onclick={() => setMetric(m)}>{PAPER_METRIC_LABELS[m]}</button
				>
			{/each}
		</div>

		<!-- the concession, in EVERY lens: trapezoidal PR-AUC flatters the reading
		     models, and we say by how much -->
		{#if best && reference && readerDistinctScores.length > 0}
			<p class="note" role="note">
				<strong>The published measure flatters the reading models.</strong> Trapezoidal PR-AUC — the area under
				the precision-recall curve, with straight lines drawn between neighbouring points, averaged
				over the 10 folds the statements are split into — puts {best.display}
				{signed(best.delta.trapezoidal.delta)} ahead of the random forest the paper fitted. Average
				precision measures that same area in steps instead, so scores that are tied earn no credit
				for the ground between them; it puts the same model {signed(best.delta.ap.delta, 4)} ahead,
				and that smaller number is the one we quote. Ties are the whole difference: across these
				statements the random forest gives {reference.distinctScores.toLocaleString()} different
				scores, the reading models only {Math.min(...readerDistinctScores)}–{Math.max(
					...readerDistinctScores
				)}, so the straight lines have far more tied ground to fill in. Average precision is the
				least flattering of the three for a second reason as well: the random forest already scores
				{f3(reference.ap)} out of a possible 1, leaving very little room to improve on it. The
				reading models’ margin on catching wrong statements, reported in its own figure, is much
				larger.
			</p>
		{/if}

		<!-- the AUROC lens never ships bare: most of its margin is the zero block -->
		{#if metric === 'auroc'}
			{#if aurocLeader}
				<p class="note" role="note">
					<strong>Most of this gap is a yes/no split, not a ranking.</strong>
					{aurocLeader.display} leads on this measure by {signed(aurocLeader.delta.auroc.delta)}. But
					it gives a score of exactly zero to the {count(aurocLeader.aurocOnRanked.nZeroed)}
					statements whose evidence it threw out — one undivided block, in no order at all — and
					AUROC rewards that split as though it were ranking. On the
					{count(aurocLeader.aurocOnRanked.nRanked)} statements it does put in order it scores
					{f3(aurocLeader.aurocOnRanked.armAuroc)}, against
					{f3(aurocLeader.aurocOnRanked.referenceAuroc)} for the random forest on those same
					statements. Set the zeroed block aside and {readersBehindOnRanked} of the
					{rankedReaders.length} reading models rank worse than the random forest does.
				</p>
			{:else}
				<p class="note" role="note">
					<strong>Nothing here separates ranking from throwing evidence out:</strong> that check did
					not load, so we cannot say how much of this gap is which.
				</p>
			{/if}
		{/if}

		<p class="reference-note">
			The comparison is against one model, not the whole published table: the random forest is one of
			the 14 published methods re-run here.
		</p>

		<div class="table-scroll">
			<table>
				<caption class="sr">
					{#if metric === 'trapezoidal'}
						Each method's {PAPER_METRIC_LABELS.trapezoidal}. The score column gives the average
						over the 10 folds the statements are split into, followed by how far that
						score moves from fold to fold, which is a spread and not a confidence interval.
					{:else}
						Each method's {PAPER_METRIC_LABELS[metric]}, taken over all the statements at once,
						with no spread.
						{#if metric === 'auroc'}
							Each score is followed by the same measure taken only on the statements that method
							scores above zero, and then by the random forest scored on those same statements.
							Each method throws out a different set of statements, so that second figure can be
							compared with the number printed beside it and with no other method's.
						{/if}
					{/if}
					The last column is a different quantity: the difference from the random forest on the same
					statements, and in brackets the range within which that difference is 95% likely to lie.
				</caption>
				<thead>
					<tr>
						<th scope="col">method</th>
						<th scope="col" class="num">
							{#if metric === 'trapezoidal'}
								{PAPER_METRIC_LABELS.trapezoidal}, averaged over 10 folds ± movement between
								folds
							{:else}
								{PAPER_METRIC_LABELS[metric]}
							{/if}
						</th>
						<!-- A COLUMN HEADER NAMES THE COLUMN. This one carried a whole sentence
						     ("— one of the 14 published methods re-run here"), which wrapped to two
						     lines above a numeric column and still had to be read on every row. The
						     qualifier is real and is the only place it was stated, so it moved to
						     the note under the table rather than being dropped. -->
						<th scope="col" class="num">difference vs random forest · 95% interval</th>
					</tr>
				</thead>
				<tbody>
					{#each data.arms as arm (arm.id)}
						<tr
							class:reference={arm.delta === null}
							class:sig={at(arm, metric, 'ahead')}
							class:behind={at(arm, metric, 'behind')}
						>
							<td class="arm">
								<span class="arm-label">{arm.display}</span>
								<span class="kind" title={kindTitle(arm)}>{kindBadge(arm)}</span>
							</td>
							<!-- the ± rides ONLY with the fold-mean level; AP and AUROC show no spread.
							     Under AUROC the level carries its own qualification: the same
							     measure taken on the block this arm actually orders, beside the
							     paper's RF on those same statements. -->
							<td class="num value"
								>{arm[metric].toFixed(3)}{#if metric === 'trapezoidal'}&nbsp;±&nbsp;{arm.foldPopulationSd.toFixed(
										3
									)}&nbsp;<small>between folds</small>{:else if metric === 'auroc'}<small
										class="ranked"
										title={rankedTitle(arm)}
										>{#if arm.aurocOnRanked === null}ranked block unavailable{:else if arm.aurocOnRanked.nZeroed === 0}throws
											nothing out{:else}on the {count(arm.aurocOnRanked.nRanked)} it ranks:
											{f3(arm.aurocOnRanked.armAuroc)} vs forest
											{f3(arm.aurocOnRanked.referenceAuroc)}{/if}</small
									>{/if}</td
							>
							<td class="num delta">
								{#if arm.delta === null}
									<span
										class="baseline"
										title={`every difference in this column is measured against this row (${PAPER_LITERAL_REFERENCE_ARM_ID})`}
										>— what the others are compared with</span
									>
								{:else}
									{@const d = arm.delta[metric]}
									<span
										class="delta-figure"
										class:excludes={d.standing === 'ahead'}
										class:excludes-behind={d.standing === 'behind'}
									>
										{signed(d.delta, deltaDigits)}
										<span class="ci"
											>[{signed(d.ciLow, deltaDigits)}, {signed(d.ciHigh, deltaDigits)}]</span
										>
										{#if d.standing !== 'not-significant'}
											<span class="mark" class:behind={d.standing === 'behind'} aria-hidden="true"
												>{DELTA_MARK[d.standing]}</span
											>
											<span class="sr">{DELTA_SR[d.standing]}</span>
										{/if}
									</span>
								{/if}
							</td>
						</tr>
					{/each}
				</tbody>
			</table>
		</div>

		<!-- Provenance collapsed: it is method, not argument, and the page runs to a
		     prose budget. Nothing is deleted — the summary opens it in place. -->
		<details class="method">
			<summary>what the ± and the bracket mean, and which random forest this is</summary>
			<p>
				Under {PAPER_METRIC_LABELS.trapezoidal} the score is an average over the 10 folds; the ± is
				how far it moves from fold to fold, a spread and not a confidence interval. The last column
				is a different quantity whichever measure is shown: the difference from
<em>{reference?.display ?? 'the random forest we compare against'}</em>, obtained by
				re-drawing the same statements at random ten thousand times, with the middle 95% of those
				re-draws in brackets. The forest itself is the published
				<code class="cv">RF 2k-d13 + Type/#PMIDs/promoter</code>: 2,000 decision trees grown to depth
				13 over how many pieces of evidence each source supplied, plus the kind of statement, how many
				papers mention it, and a flag for whether it is about a promoter.
				<!-- the fold count is quoted off the run manifest; with no manifest, no claim -->
				{#if data.status === 'ok' && data.reproduction}How the folds were cut:
					<code class="cv">{data.reproduction.cvProtocol}</code>{/if}
			</p>
		</details>

		<footer>
			<!-- The two concessions ride WITH the mark that draws them, not in a caveats
			     block further down: a reader who sees ● must see what it does not cover. -->
			<span class="legend">
				<!-- The legend key is what the table actually draws: two direction-carrying
				     marks, not one sign-blind ●. The glyphs are aria-hidden because each
				     row already states its direction in words to a screen reader. -->
				<span class="mark" aria-hidden="true">▲</span> the whole 95% range sits above zero — clearly
				ahead of the random forest; <span class="mark behind" aria-hidden="true">▼</span> the whole
				range sits below zero — clearly behind it, <em>on the measure now shown</em>
				({PAPER_METRIC_LABELS[metric]}). {deltaArmCount} methods are each compared with that one
				forest, and the ranges are not widened to account for making that many comparisons at
				once.{#if best && strongestPaper && reference && strongestPaper.id !== reference.id && marginVsStrongestPaper !== null}
					<!-- `best` joins the guard so the second figure needs no `?? 0`. It had one:
					     a 0 there would print "+0.0000" as the headline margin — the single
					     number on this page a reader is most likely to carry away. -->
					That forest is not the highest-scoring method the 2023 paper reports:
					<em>{strongestPaper.display}</em> scores higher, so measured against
					<em>it</em> the lead is {signed(marginVsStrongestPaper, 4)} rather than
					{signed(best.delta.ap.delta, 4)}.{/if} The marks change with the
				measure: here {nAheadHere} clearly ahead of the forest and {nBehindHere} clearly behind
				it{#if metric !== 'ap'}; under {PAPER_METRIC_LABELS.ap}, the least flattering of the three,
					{nAheadAp} ahead and {nBehindAp} behind — so a ▲ in this view does not settle anything on
					its own{/if}. The forest is scored on data it never learned from: the statements are split
				into 10 equal folds and each statement is scored by a forest fitted on the other nine. The
				reading models were never fitted on these statements at all — they were prompted with a few
				fixed hand-written examples, and never shown these labels — which is a difference in the
				forest's favour.
			</span>
			{#if metric === 'trapezoidal'}
				<span class="legend"
					>± is how far the score moves between folds — not a confidence interval</span
				>
			{/if}
			<code>{data.artifact_path}</code>
		</footer>
	{/if}
</section>

<style>
	.paper-literal {
		margin-top: 1.5rem;
		padding-top: 1.25rem;
		border-top: 2px solid var(--ink);
	}
	.paper-literal > header {
		display: flex;
		justify-content: space-between;
		align-items: start;
		gap: 1rem;
	}
	.paper-literal h2,
	.gate h2 {
		margin: 0.18rem 0 0;
		font-family: var(--serif);
		font-size: 1.35rem;
		font-weight: 400;
	}
	.paper-literal > header > strong {
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

	/* metric lens — mirrors the frontier .axis-switch */
	.metric-switch {
		display: flex;
		align-items: baseline;
		flex-wrap: wrap;
		gap: 0.9rem;
		margin: 1rem 0;
		font-family: var(--mono);
	}
	.mx-label {
		color: var(--ink-faint);
		font-size: 0.7rem;
		text-transform: lowercase;
		letter-spacing: 0.04em;
	}
	.metric-switch button {
		background: none;
		border: none;
		padding: 0 0 0.25rem;
		font-family: var(--mono);
		font-size: 0.86rem;
		color: var(--ink-muted);
		cursor: pointer;
		border-bottom: 2px solid transparent;
	}
	.metric-switch button:hover {
		color: var(--ink);
	}
	.metric-switch button.on {
		color: var(--ink);
		border-bottom-color: var(--accent);
	}
	.metric-switch button:focus-visible {
		outline: 1.5px solid var(--accent);
		outline-offset: 2px;
	}

	/* A plain statement of fact that sits with the table, NOT a caveat. `.note` is
	   the warning treatment on this figure — bordered, tinted, --blocked — and
	   reusing it here would have dressed a neutral scoping sentence as an alert. */
	.reference-note {
		margin: 0 0 0.5rem;
		font-family: var(--serif);
		font-size: 0.8rem;
		line-height: 1.5;
		color: var(--ink-muted);
		max-width: 70ch;
	}
	.note {
		margin: 0 0 1rem;
		padding: 0.75rem 0.9rem;
		border: 1px solid var(--blocked);
		border-left-width: 3px;
		background: color-mix(in srgb, var(--blocked) 3%, transparent);
		font-family: var(--serif);
		font-size: 0.82rem;
		line-height: 1.5;
		color: var(--ink-muted);
	}
	.note strong {
		color: var(--ink);
	}

	/* method note: what the ± is, what the bracket is, and the folds verbatim.
	   Collapsed by default — one click, nothing hidden, nothing lost. Mirrors the
	   `.method` details pattern in ReviewQueue / BeliefModelLadder. */
	.method {
		margin: 0.9rem 0 0;
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
		margin-bottom: 0.5rem;
	}
	.method p {
		margin: 0;
		font-family: var(--serif);
		font-size: 0.74rem;
		line-height: 1.55;
		color: var(--ink-faint);
	}
	.method em {
		font-style: normal;
		color: var(--ink-muted);
	}
	.method .cv {
		word-break: break-word;
	}

	.table-scroll {
		overflow-x: auto;
	}
	table {
		width: 100%;
		border-collapse: collapse;
		font-family: var(--mono);
		font-size: 0.74rem;
	}
	th {
		color: var(--ink-faint);
		font-weight: 500;
		text-align: left;
	}
	th,
	td {
		padding: 0.42rem 0.55rem;
		border-bottom: 1px dotted var(--rule);
		vertical-align: baseline;
	}
	.num {
		text-align: right;
		font-variant-numeric: tabular-nums;
	}
	.arm-label {
		color: var(--ink);
	}
	.kind {
		margin-left: 0.5rem;
		padding: 0.02rem 0.28rem;
		border: 1px solid var(--rule);
		border-radius: 2px;
		font-size: 0.58rem;
		text-transform: uppercase;
		letter-spacing: 0.05em;
		color: var(--ink-faint);
	}
	.value {
		color: var(--ink);
	}
	/* the SD marker rides with the fold-mean level, never beside AP or AUROC */
	.value small {
		font-size: 0.86em;
		color: var(--ink-faint);
	}
	/* the ranked-block figure sits UNDER the level as its own line, so the two
	   AUROCs are never read as one compound number */
	.value small.ranked {
		display: block;
		margin-top: 0.12rem;
		font-size: 0.82em;
		white-space: nowrap;
	}
	tr.reference {
		background: var(--accent-wash);
	}
	tr.reference .arm-label {
		font-weight: 600;
	}
	.baseline {
		color: var(--ink-faint);
	}
	.delta-figure {
		color: var(--ink-muted);
	}
	.delta-figure.excludes,
	.delta-figure.excludes-behind {
		color: var(--ink);
		font-weight: 700;
	}
	.ci {
		color: var(--ink-faint);
	}
	.delta-figure.excludes .ci,
	.delta-figure.excludes-behind .ci {
		color: var(--ink-muted);
	}
	.mark {
		margin-left: 0.15rem;
		color: var(--accent);
	}
	/* A significant LOSS must never wear the same accent as a significant win. */
	.mark.behind {
		color: var(--ink-faint);
	}
	tr.sig {
		box-shadow: inset 2px 0 0 var(--accent);
	}
	tr.behind {
		box-shadow: inset 2px 0 0 var(--ink-faint);
	}

	footer {
		display: flex;
		justify-content: space-between;
		flex-wrap: wrap;
		gap: 0.6rem 1rem;
		margin-top: 0.9rem;
		font-family: var(--mono);
		font-size: 0.62rem;
		color: var(--ink-faint);
	}
	.legend .mark {
		margin-left: 0;
	}
	/* The ● line carries two concessions, so it needs room to wrap as a sentence. */
	footer .legend:first-child {
		flex: 1 1 40ch;
		max-width: 88ch;
		line-height: 1.5;
	}
	code {
		font-family: var(--mono);
		font-size: 0.62rem;
		color: var(--ink-faint);
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

	/* visually-hidden but screen-reader available */
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

	@media (max-width: 720px) {
		.paper-literal > header {
			display: block;
		}
		.paper-literal > header > strong {
			display: inline-block;
			margin-top: 0.6rem;
		}
	}
</style>
