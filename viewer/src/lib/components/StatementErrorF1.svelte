<script lang="ts">
	/**
	 * The margin at statement grain, named: error-class F1 — precision and recall
	 * for identifying WRONG statements — at each model's own best-error-class-F1
	 * cut, against our re-run of the random forest on evidence-count features
	 * released with the 2023 INDRA assembly paper (2,000 trees, depth 13, over
	 * per-source evidence counts plus statement type, PMID count and a promoter
	 * flag), scored over that paper's 1,689-statement benchmark, its published
	 * curation labels and the same 10 cross-validation folds.
	 *
	 * NAMING RULE. Every model and every metric on this figure is named by what it
	 * IS; the paper is cited as provenance, never as identity. "the paper's model",
	 * "the paper's RF", "err-F1" and "max-t" are insider shorthand and must not
	 * come back — a label here has to stand alone for a reader who has never
	 * opened the 2023 paper.
	 *
	 * PLAIN-LANGUAGE RULE (2026-07-27). The reader this page is written for is a
	 * working biologist who has never read the 2023 paper and does not work in
	 * machine-learning evaluation. In RENDERED PROSE — not in identifiers, not in
	 * these comments — "arm" is a model, "panel" is the benchmark or plainly the
	 * 1,689 statements, "tau" is a score cutoff, "delta" is a margin, "lane" is a
	 * row, "census" is a count, "residual" is a difference, "pooled" is over all
	 * statements at once, and "max-t" is the correction for testing all four
	 * reading models at once. A term that needs prior knowledge is either defined
	 * in the sentence that first uses it or replaced. "Fold", "out-of-fold" and
	 * "cross-validation" are NOT replaced — they are the field's standard terms
	 * and the audience published a cross-validation paper — so the lede defines
	 * "fold" in the clause that first uses it and every later use is bare.
	 *
	 * WHY THIS FIGURE EXISTS. The head-to-head above quotes average precision —
	 * the most conservative view on a benchmark that is 73.2% positive with the
	 * reference already at 0.9412. The work evidence-gated reading actually does —
	 * a language model drops evidence it judges unsupported, then the same
	 * noisy-OR over per-source reliabilities scores what remains — is pushing
	 * ERRORS DOWN, and that is a decision at a cutoff, so it is measured here
	 * and reported with its cutoff rule attached.
	 *
	 * PRECISION AND RECALL ARE DRAWN, NOT JUST F1. F1 is their harmonic mean, so it
	 * always lands between them; the span between the two marks is exactly what an
	 * F1 column hides. The mechanism of this margin is precision — the random
	 * forest is right about a minority of the statements it flags, the winning
	 * evidence-gated readings about three-quarters — and a reader who cannot see
	 * that cannot check it.
	 *
	 * THREE CUTOFF RULES, THREE DISCLOSURES. Chart A and its readouts are the
	 * best-F1 cut and carry `headlineThresholdRule`. The matched-recall block is a
	 * SECOND benchmark-fitted cut by a different rule and carries
	 * `matchedThresholdRule`. The review-queue cross-check is a THIRD rule and
	 * carries `reconciliation.thresholdRule`. No number appears without the rule
	 * that produced its cutoff AND the disclosure that governs it. Every one of
	 * them renders its PLAIN half: `ruleProse.plain`, never `rule`. The shipped
	 * `rule` / `oracle` strings are written in the idiom of whoever generated the
	 * file, and this component used to put them on screen under two summaries
	 * offering "the artifact's own words". They are not deleted — they collect,
	 * with every other shipped sentence on the page, in the one verification
	 * section at the end (`PaperAuditTrail.svelte`), each beside the restatement
	 * that replaced it. What is on screen HERE is the readable form, which is the
	 * whole point of a disclosure.
	 *
	 * THE MATCHED-RECALL BLOCK NAMES BOTH DELTAS AND LEADS WITH THE MATCHED ONE.
	 * `deltaAtMatchedRecall` re-cuts the reference at each row's ACHIEVED recall.
	 * `deltaEachSideAtItsOwnTargetCut` does not, and flips Gemma 4 E2B's sign; it is
	 * kept, labelled for what it is, and never presented as recall-matched.
	 * `signsDisagree` is derived so the flip is pointed at rather than left for a
	 * reader to spot. How the two columns compare is COUNTED, not asserted: the
	 * prose here read "larger for every arm" while INDRA CoGEx hybrid's two values
	 * are equal to the last bit, so the sentence now selects on
	 * `unmatchedLarger/Equal/SmallerDisplays`.
	 *
	 * Every number here is READ off
	 * `data/results/indra_paper_literal_models_20260724/statement_error_f1.json`.
	 * Geometry, label budgets, per-series encodings and every fail-closed gate live
	 * in `$lib/data/paper-error-f1`; this file only draws.
	 */
	import {
		STATEMENT_ERROR_F1_GEOMETRY as G,
		STATEMENT_ERROR_F1_SERIES,
		STATEMENT_ERROR_F1_SERIES_IDS,
		fmt4,
		fmtDelta,
		fmtPct,
		type ErrorF1Axis,
		type ErrorF1Delta,
		type ErrorF1Lane,
		type Standing,
		type StatementErrorF1Load
	} from '$lib/data/paper-error-f1';

	let { data }: { data: StatementErrorF1Load } = $props();

	const figure = $derived(data.status === 'ok' ? data.figure : null);
	const lanes = $derived(figure?.lanes ?? []);
	/** Every lane that carries a margin — i.e. everything but the reference. */
	const contenders = $derived(lanes.filter((lane) => !lane.isReference));
	/**
	 * The same set, NARROWED: the reference is the only lane the loader gives a
	 * null `delta`, so every margin sentence below reads `lane.delta.x` outright.
	 * The predicate is what buys that — the clauses used to write
	 * `lane.delta?.ciLow ?? 0`, which would have printed +0.0000 as a measured
	 * interval bound if the narrowing assumption ever became false, and hid the
	 * assumption from the compiler in the meantime.
	 */
	type DecidedLane = ErrorF1Lane & { delta: ErrorF1Delta };
	const decided = $derived(
		lanes.filter((lane): lane is DecidedLane => lane.delta !== null)
	);
	const plotBottom = $derived(figure ? figure.height - G.axisPad : 0);
	const ruleTop = G.topPad - 10;

	/**
	 * Census of what the figure shows, counted off the drawn lanes rather than
	 * written into the prose, so a rerun that changes the answer changes the
	 * sentence. `winsSimultaneously` is SIGN-AWARE in the loader — a bare
	 * "excludes zero" once marked two significant LOSSES as wins on this page.
	 */
	const winners = $derived(decided.filter((lane) => lane.delta.winsSimultaneously));
	/**
	 * THREE WAYS, NOT TWO — AT EVERY DEPTH. `!winsSimultaneously` merges "not
	 * significant" with "significantly worse", and the single sentence that served
	 * both asserted the n.s. reading for both: a sign-flipped GLM-5 rendered
	 * "−0.1416, interval [−0.1744, −0.1090], which spans zero — level with the
	 * the random forest". Splitting the OUTER partition three ways left the clause
	 * INSIDE `familyLevel` two-way on a sign-blind pointwise boolean, which printed the
	 * identical words for a pointwise [−0.0381, −0.0020] and a [+0.1090, +0.1744]
	 * and named neither sign. Both now select on a loader class: `standing` for the
	 * deciding interval, `pointwiseStanding` for the pointwise one that is printed
	 * beside it. That boolean no longer exists in any loader on /paper.
	 */
	const familyLevel = $derived(
		decided.filter((lane) => lane.inMaxTFamily && lane.delta.standing === 'not-significant')
	);
	const familyBehind = $derived(
		decided.filter((lane) => lane.inMaxTFamily && lane.delta.standing === 'behind')
	);
	/**
	 * Contenders the simultaneous correction does not cover. They carry no band,
	 * so they are neither claimed as wins nor silently dropped: their pointwise
	 * standing is stated in the reading list, in the same three classes.
	 */
	const outsideFamily = $derived(decided.filter((lane) => !lane.inMaxTFamily));
	/** Where an interval sits relative to zero, phrased from the loader's class. */
	const zeroClause = (standing: Standing): string =>
		standing === 'ahead'
			? 'entirely above zero'
			: standing === 'behind'
				? 'entirely below zero'
				: 'spanning zero';
	/**
	 * The range a standing was decided on, named so the printed interval matches
	 * it — as a noun phrase that reads after "the range that decides it, the …".
	 */
	const basisName = (lane: DecidedLane): string =>
		lane.delta.standingBasis === 'simultaneous'
			? `95% range widened to cover all ${figure?.multiplicity.familySize ?? 0} reading models at once`
			: '95% range for this model alone';
	const deltaOf = (lane: DecidedLane): number => lane.delta.delta;
	const best = $derived(
		winners.length ? winners.reduce((a, b) => (deltaOf(b) > deltaOf(a) ? b : a)) : null
	);
	const narrowestWin = $derived(
		winners.length ? winners.reduce((a, b) => (deltaOf(b) < deltaOf(a) ? b : a)) : null
	);
	/** The oracle's size on the winning side: candidate cuts they had to pick from. */
	const winnerCuts = $derived(winners.map((lane) => lane.distinctScores));

	/**
	 * The models whose two matched-recall margins disagree in sign, so the
	 * sentence naming the flip reads its magnitudes off the same rows the table
	 * draws instead of restating them.
	 */
	const flipped = $derived(contenders.filter((lane) => lane.matched.signsDisagree));

	/**
	 * How the UNMATCHED column compares to the matched one, assembled from the
	 * loader's three lists. This clause used to read "larger for every arm", which
	 * is false: INDRA CoGEx hybrid's two margins are equal to the last bit.
	 */
	const unmatchedComparison = $derived.by(() => {
		if (!figure) return '';
		const clauses = [
			`larger for ${figure.unmatchedLargerDisplays.length} of the ${contenders.length} models`
		];
		if (figure.unmatchedEqualDisplays.length > 0) {
			clauses.push(
				`identical for ${figure.unmatchedEqualDisplays.join(', ')} — where the random forest ` +
					`re-cut at that model's own recall scores exactly what it scores at its own cutoff`
			);
		}
		if (figure.unmatchedSmallerDisplays.length > 0) {
			clauses.push(`smaller for ${figure.unmatchedSmallerDisplays.join(', ')}`);
		}
		return clauses.join(', ');
	});

	function scale(axis: ErrorF1Axis, value: number): number {
		return axis.left + ((value - axis.min) / (axis.max - axis.min)) * (axis.right - axis.left);
	}

	/** Panel B tick text: signed, two decimals, ASCII sign; bare '0' at the rule. */
	function tickB(value: number): string {
		if (value === 0) return '0';
		return `${value > 0 ? '+' : '-'}${Math.abs(value).toFixed(2)}`;
	}

	function diamond(cx: number, cy: number): string {
		return `M ${cx} ${cy - 4} L ${cx + 4} ${cy} L ${cx} ${cy + 4} L ${cx - 4} ${cy} Z`;
	}

	function shortSha(value: string): string {
		return `${value.slice(0, 10)}…`;
	}
</script>

<section class="errorf1" aria-labelledby="errorf1-title">
	{#if data.status === 'unavailable' || figure === null}
		<div class="gate" role="status">
			<p class="eyebrow">one score per statement</p>
			<h2 id="errorf1-title">The error-catching comparison is unavailable</h2>
			<p>{data.reason}</p>
			<code>{data.artifact_path}</code>
		</div>
	{:else}
		<header>
			<div>
				<!-- The result file's own name for this metric is written in its idiom
				     ("statement-grain error-class F1 at each arm's own best-F1 cut"). It
				     is kept, unchanged, in the verification section at the end of the
				     page; the eyebrow and the value table's note both say the same thing
				     in words a reader does not have to be taught. -->
				<p class="eyebrow">how well each model finds the wrong statements</p>
				<h2 id="errorf1-title">Catching the errors</h2>
			</div>
			<strong>cutoffs tuned on the answers</strong>
		</header>

		<div class="primary" role="note">
			<p>
				<!-- The lede has a STANDALONE form. With no simultaneous winner the whole
				     `{#if}` used to be skipped and the paragraph opened "GLM-5 gate does
				     not: …" with nothing for "does not" to refer to. The else branch
				     supplies the antecedent from the same counted fields. What the two
				     branches SHARE — what the score is, what it is measured against, and
				     on which statements — is written once, outside them: it is true
				     either way, and stating it twice cost the reader the same sentence
				     twice over. -->
				{#if best && narrowestWin}
					<strong
						>{best.display}
						{fmt4(best.operating.errorF1)} against {figure.reference.display}
						{fmt4(figure.reference.operating.errorF1)}.</strong
					>
				{:else}
					<strong
						>No reading model beats {figure.reference.display}
						{fmt4(figure.reference.operating.errorF1)} on these
						{figure.panel.n.toLocaleString()} statements.</strong
					>
				{/if}
				That score is error-class F1: of the statements a model flags as wrong, the share that really
				are wrong (precision), and of the errors we already know about, the share it flags (recall),
				combined into one number. {figure.reference.display} is the random forest released with the
				2023 INDRA assembly paper — 2,000 trees, depth 13, given only how much evidence each source
				contributed to a statement plus its type, its number of papers and a promoter flag. Every
				model here is scored on the same {figure.panel.n.toLocaleString()} statements, on the
				curation labels released with them ({figure.panel.labelField}), unchanged, and split into the
				same {figure.panel.nFolds} folds: equal groups, each fitted model scored on the group it did
				not train on. The challengers are reading models: a language model drops the evidence it judges
				unsupported, then INDRA's usual belief formula scores what survives.
				{#if best && narrowestWin}
					{figure.nWinsSimultaneously} of the {figure.nInFamily} of them beat it, by {fmtDelta(
						deltaOf(narrowestWin)
					)} to
					{fmtDelta(deltaOf(best))}, and every one of those margins stays above zero even after the
					range around it is widened to cover all {figure.nInFamily} at once.
				{:else}
					None of the {figure.nInFamily} of them keeps a margin above zero once the range around it
					is widened to cover all of them at once.
				{/if}
				{#each familyLevel as lane (lane.id)}
					{lane.display} does not: {fmtDelta(deltaOf(lane))}, and the range that decides it, the
					{basisName(lane)}, runs [{fmtDelta(lane.delta.standingLow)}, {fmtDelta(
						lane.delta.standingHigh
					)}], which crosses zero. Level with the random forest, not ahead of it. Taken on its own,
					its 95% range is [{fmtDelta(lane.delta.ciLow)}, {fmtDelta(lane.delta.ciHigh)}], {zeroClause(
						lane.delta.pointwiseStanding
					)},
					<!-- THREE WAYS. Two-way on the deleted pointwise boolean, this clause printed
					     the same "does not, so the family-wide correction is what makes this
					     a tie" for an interval entirely BELOW zero as for one entirely
					     ABOVE it, and named neither sign. -->
					{#if lane.delta.pointwiseStanding === 'ahead'}
						so on its own it would read as a win; testing all four reading models at once is what
						makes it a tie.
					{:else if lane.delta.pointwiseStanding === 'behind'}
						so on its own it would read as a LOSS; testing all four reading models at once is what
						makes it a tie.
					{:else}
						which agrees.
					{/if}
				{/each}
				{#each familyBehind as lane (lane.id)}
					{lane.display} is BEHIND it: {fmtDelta(deltaOf(lane))}, and the range that decides it, the
					{basisName(lane)}, runs [{fmtDelta(lane.delta.standingLow)}, {fmtDelta(
						lane.delta.standingHigh
					)}], entirely below zero; on its own the 95% range is [{fmtDelta(lane.delta.ciLow)},
					{fmtDelta(lane.delta.ciHigh)}], {zeroClause(lane.delta.pointwiseStanding)}. That is a loss
					on these statements, not a tie.
				{/each}
			</p>
			{#if figure.winnerPrecisionMin !== null && figure.winnerPrecisionMax !== null}
				<p>
					<strong>Precision is what moved.</strong>
					When the random forest flags a statement as wrong it is right
					{fmtPct(figure.reference.operating.errorPrecision)} of the time; the reading models that
					win are right {fmtPct(figure.winnerPrecisionMin)}–{fmtPct(figure.winnerPrecisionMax)} of
					the time. Chart A draws precision and recall as two separate marks because one F1 number
					hides which of the two moved.
				</p>
			{/if}
		</div>

		<div class="disclosure" role="note">
			<h3>Every cutoff here was picked with the answers already in hand</h3>
			<p>
				Cutoff: a model flags a statement as wrong when its belief falls below a cutoff, and we slide
				each model's cutoff down until that model's own error-class F1 is the highest it can be on
				these same {figure.panel.n.toLocaleString()} statements. Nobody could choose that cutoff
				before the curation existed — it is fitted and scored on one and the same set of answers.
				It also favours the reference: the random forest has
				{figure.reference.distinctScores.toLocaleString()} different scores to place a cutoff between,
				far more places to try than any reading model has.
			</p>
			<!-- WHAT THE CUTOFF VALUE MEANS. The scores are not arbitrary decimals:
			     every one of them is reachable only by some exact amount of surviving
			     evidence, so a cutoff is really a rule about evidence. Both cutoffs the
			     models chose land EXACTLY on such a value (0.6500 and 0.8775 are
			     1-(0.05+0.3) and 1-(0.05+0.3)^2 at INDRA's published numbers for the
			     text-reading programs), which is worth saying because a reader who sees
			     "0.8775" has no way to know it is one program short of a sentence. -->
			<p>
				A score is fixed by how much evidence survived, so a cutoff is really a rule about evidence.
				One program finding a statement in one sentence scores 0.65; two programs finding it once
				each scores 0.8775; one program finding it three times scores 0.9230. So a cutoff at 0.65
				means <em>check anything held up by no more than a single sentence from a single program</em>.
				A cutoff at 0.8775 means <em>check anything held up by no more than one sentence from each of
				two programs</em> — a much wider sweep, which is why the model that chose it sends more for
				checking and is right about them less often.
			</p>
			<!-- The disclosure stays IN THE OPEN, beside the numbers it governs; what
			     sits under the summary is the same three sentences in full, in
			     ordinary words. The result file's own wording of all three is in the
			     page's verification section at the end, beside these restatements. -->
			<details class="rule-detail">
				<summary>the cutoff rule in full, and what choosing it here costs</summary>
				<p class="rule">{figure.headlineThresholdRule.ruleProse.plain}</p>
				<p class="rule oracle">{figure.headlineThresholdRule.oracleProse.plain}</p>
				{#if figure.modalCutValue !== null}
					<p class="rule oracle">{figure.prose.modalThresholdNote.plain}</p>
				{/if}
			</details>
		</div>

		<figure>
			<figcaption>
				A · each model at its own best cutoff. B · margin over {figure.reference.display}.
			</figcaption>
			<svg
				viewBox="0 0 {G.width} {figure.height}"
				style:min-height="{figure.height}px"
				role="img"
				aria-labelledby="errorf1-chart-title errorf1-chart-desc"
			>
				<title id="errorf1-chart-title"
					>Error-class precision, recall and F1 — precision and recall for identifying WRONG
					statements — one score per statement, for each belief model, over the {figure.panel.n}
					statements published with the 2023 INDRA assembly paper, beside each model's margin in
					error-class F1 over the re-run of the random forest released with it: 2,000 trees, depth
					13, given how much evidence each source contributed plus statement type, paper count and a
					promoter flag</title
				>
				<desc id="errorf1-chart-desc"
					>One row per model, ordered by error-class F1 and ranked against each other rather than
					grouped, so the random forest sits among the reading models at its own score. The challengers are reading
					models: a language model drops the evidence it judges unsupported, then INDRA's usual
					belief formula — a noisy-OR over how reliable each source is — scores what remains. Each
					row is named at the left, with the score cutoff that produced it beneath the name; in the
					order drawn those read: {figure.descLabels}. In chart A a light span joins that model's
					error precision (filled square) to its error recall (open circle), and a heavy vertical
					tick marks their harmonic mean, the error-class F1; the vertical rule crossing every row is
					the random forest's own error-class F1, labelled {figure.referenceRuleLabel}. In chart B a
					diamond marks the margin in error-class F1 over that random forest, a solid bar gives the
					95% range for that model alone, and a dashed bar the range widened to cover all
					{figure.multiplicity.familySize} reading models at once — one widening for all of them, so a
					win claimed here is a win claimed for every reading model at once; the heavy vertical rule
					is zero, level with the random forest. The random forest's own row carries no mark in chart
					B because every margin is measured against it.</desc
				>

				<text class="panel-head" x={G.panelALeft} y="20"
					>A — precision and recall at each model's own cutoff</text
				>
				<!-- 9px mono = 5.4186 u/char; panelBLeft 646 → viewBox 960 leaves 314 u =
				     57 chars. "B — margin over the random forest" is 33 (178.8 u). -->
				<text class="panel-head" x={G.panelBLeft} y="20">B — margin over the random forest</text>

				{#each figure.panelA.ticks as tick (tick)}
					<line
						class="grid"
						x1={scale(figure.panelA, tick)}
						y1={G.topPad}
						x2={scale(figure.panelA, tick)}
						y2={plotBottom}
					/>
					<text class="tick" x={scale(figure.panelA, tick)} y={plotBottom + 15}
						>{tick.toFixed(2)}</text
					>
				{/each}
				{#each figure.panelB.ticks as tick (tick)}
					{#if tick !== 0}
						<line
							class="grid"
							x1={scale(figure.panelB, tick)}
							y1={G.topPad}
							x2={scale(figure.panelB, tick)}
							y2={plotBottom}
						/>
					{/if}
					<text class="tick" x={scale(figure.panelB, tick)} y={plotBottom + 15}>{tickB(tick)}</text>
				{/each}

				<!-- The two rules everything on this figure is read against: the random
				     forest's own error-class F1 in panel A, and zero in panel B. Each gets a wash
				     channel behind the data, a heavy rule, and (after the lanes) a thin
				     overlay so it stays continuous where a mark crosses it. -->
				<!-- `display`, not `label`: nothing named `label` may reach a render
				     position on this page, and a local property that happens to be a
				     display string is exactly how that rule stops being checkable. -->
				{#each [{ x: scale(figure.panelA, figure.referenceRuleValue), display: figure.referenceRuleLabel, fits: figure.referenceRuleLabelFits }, { x: scale(figure.panelB, 0), display: figure.zeroRuleLabel, fits: figure.zeroRuleLabelFits }] as rule (rule.display)}
					<rect
						class="rule-wash"
						x={rule.x - 3}
						y={ruleTop}
						width="6"
						height={plotBottom + 6 - ruleTop}
					/>
					<line class="rule-line" x1={rule.x} y1={ruleTop} x2={rule.x} y2={plotBottom + 6} />
					<!-- 8px mono = 4.8165 u/char; the fit is measured in the builder against
					     this rule's own free space, and the label flips side rather than
					     clipping at the viewBox edge. -->
					<text
						class="rule-label"
						x={rule.fits ? rule.x + G.rulePad : rule.x - G.rulePad}
						y={G.topPad - 14}
						text-anchor={rule.fits ? 'start' : 'end'}>{rule.display}</text
					>
				{/each}

				{#each lanes as lane, index (lane.id)}
					{@const prec = STATEMENT_ERROR_F1_SERIES.precision}
					{@const rec = STATEMENT_ERROR_F1_SERIES.recall}
					{@const f1 = STATEMENT_ERROR_F1_SERIES.f1}
					{@const point = STATEMENT_ERROR_F1_SERIES.pointwise}
					{@const sim = STATEMENT_ERROR_F1_SERIES.simultaneous}

					{#if index > 0}
						<line
							class="lane-rule"
							x1="0"
							y1={lane.y - G.laneHeight / 2}
							x2={G.width}
							y2={lane.y - G.laneHeight / 2}
						/>
					{/if}

					<!-- Right-anchored into a 186-unit gutter: 9px mono = 5.4186 u/char =
					     34 chars, enforced in buildFigure(). Longest producible name is the
					     reference's own at 32 chars (173.40 units). The 7.5px tau sub-line
					     sits 11 units below the name inside a 34-unit lane. Both strings
					     are emitted in full in <desc>. -->
					<text
						class="lane-label"
						class:is-reference={lane.isReference}
						x={G.labelAnchorX}
						y={lane.nameY}>{lane.display}</text
					>
					<text class="lane-sub" x={G.labelAnchorX} y={lane.subY}>{lane.subLabel}</text>

					<!-- Panel A. The span is the whole point: F1 is the harmonic mean of
					     the two marks and always lands between them. -->
					<line
						class="span"
						x1={scale(figure.panelA, Math.min(lane.operating.errorPrecision, lane.operating.errorRecall))}
						x2={scale(figure.panelA, Math.max(lane.operating.errorPrecision, lane.operating.errorRecall))}
						y1={lane.y}
						y2={lane.y}
					/>
					<rect
						class="mark"
						x={scale(figure.panelA, lane.operating.errorPrecision) - 3.2}
						y={lane.y - 3.2}
						width="6.4"
						height="6.4"
						fill={prec.strokeVar}
					>
						<title>{lane.titleA}</title>
					</rect>
					<circle
						class="mark"
						cx={scale(figure.panelA, lane.operating.errorRecall)}
						cy={lane.y}
						r="3.4"
						fill="var(--paper)"
						stroke={rec.strokeVar}
						stroke-width={rec.strokeWidth}
					>
						<title>{lane.titleA}</title>
					</circle>
					<line
						class="mark"
						x1={scale(figure.panelA, lane.operating.errorF1)}
						x2={scale(figure.panelA, lane.operating.errorF1)}
						y1={lane.y - G.f1TickHalf}
						y2={lane.y + G.f1TickHalf}
						stroke={f1.strokeVar}
						stroke-width={f1.strokeWidth}
					>
						<title>{lane.titleA}</title>
					</line>
					<!-- 8px mono = 4.8165 u/char; 92-unit gutter = 19 chars, enforced in
					     buildFigure(). Longest shipped readout is "F1 0.7855" at 9. -->
					<text class="readout" x={G.panelAReadoutX} y={lane.y + 3}>{lane.readoutA}</text>

					<!-- Panel B. Simultaneous band first, so the pointwise interval nests
					     visibly inside it: the widening IS the family-wise correction. -->
					{#if lane.delta}
						{@const margin = lane.delta}
						{#if margin.simLow !== null && margin.simHigh !== null}
							{@const band = [margin.simLow, margin.simHigh]}
							<line
								class="bar"
								x1={scale(figure.panelB, band[0])}
								x2={scale(figure.panelB, band[1])}
								y1={lane.y}
								y2={lane.y}
								stroke={sim.strokeVar}
								stroke-width={sim.strokeWidth}
								stroke-dasharray={sim.dash}
							>
								<title>{lane.titleB}</title>
							</line>
							{#each band as end, cap (cap)}
								<line
									class="bar"
									x1={scale(figure.panelB, end)}
									x2={scale(figure.panelB, end)}
									y1={lane.y - G.simultaneousCap}
									y2={lane.y + G.simultaneousCap}
									stroke={sim.strokeVar}
									stroke-width={sim.strokeWidth}
								/>
							{/each}
						{/if}
						<line
							class="bar"
							x1={scale(figure.panelB, margin.ciLow)}
							x2={scale(figure.panelB, margin.ciHigh)}
							y1={lane.y}
							y2={lane.y}
							stroke={point.strokeVar}
							stroke-width={point.strokeWidth}
						/>
						{#each [margin.ciLow, margin.ciHigh] as end, cap (cap)}
							<line
								class="bar"
								x1={scale(figure.panelB, end)}
								x2={scale(figure.panelB, end)}
								y1={lane.y - G.pointwiseCap}
								y2={lane.y + G.pointwiseCap}
								stroke={point.strokeVar}
								stroke-width={point.strokeWidth}
							/>
						{/each}
						<path
							class="mark"
							d={diamond(scale(figure.panelB, margin.delta), lane.y)}
							fill={point.strokeVar}
						>
							<title>{lane.titleB}</title>
						</path>
					{/if}
					<!-- 8px mono = 4.8165 u/char; 86-unit gutter = 17 chars, enforced in
					     buildFigure(). Longest shipped readout is "reference" at 9. -->
					<text class="readout" class:faint={lane.isReference} x={G.panelBReadoutX} y={lane.y + 3}
						>{lane.readoutB}</text
					>
				{/each}

				<!-- Drawn last so each rule stays one unbroken line through every mark
				     that crosses it; thin and semi-transparent so it never hides one. -->
				{#each [scale(figure.panelA, figure.referenceRuleValue), scale(figure.panelB, 0)] as ruleX (ruleX)}
					<line class="rule-over" x1={ruleX} y1={ruleTop} x2={ruleX} y2={plotBottom + 6} />
				{/each}

				<line class="axis" x1={G.panelALeft} y1={plotBottom} x2={G.panelARight} y2={plotBottom} />
				<line class="axis" x1={G.panelBLeft} y1={plotBottom} x2={G.panelBRight} y2={plotBottom} />
				<text class="axis-label" x={(G.panelALeft + G.panelARight) / 2} y={plotBottom + 32}
					>precision and recall · axis starts at {figure.panelA.min.toFixed(2)}</text
				>
				<text class="axis-label" x={(G.panelBLeft + G.panelBRight) / 2} y={plotBottom + 32}
					>margin in error-class F1</text
				>
				<!-- 8px mono = 4.8165 u/char, both notes middle-anchored on their own
				     panel. A: 62 ch = 299 u, centred on 357 → 208..506. B: 66 ch = 318 u,
				     centred on 756 → 597..915, inside the 960 viewBox and clear of A. -->
				<text class="axis-note" x={(G.panelALeft + G.panelARight) / 2} y={plotBottom + 45}
					>square = precision · circle = recall · tick = F1, the two combined</text
				>
				<text class="axis-note" x={(G.panelBLeft + G.panelBRight) / 2} y={plotBottom + 45}
					>solid = 95% range · dashed = widened to cover all {figure.multiplicity.familySize} models</text
				>
			</svg>
		</figure>

		<ul class="legend">
			{#each STATEMENT_ERROR_F1_SERIES_IDS as id (id)}
				{@const style = STATEMENT_ERROR_F1_SERIES[id]}
				<li>
					<svg viewBox="0 0 44 12" aria-hidden="true" class="swatch">
						{#if style.shape === 'square'}
							<rect x="18" y="2" width="8" height="8" fill={style.strokeVar} />
						{:else if style.shape === 'open-circle'}
							<circle
								cx="22"
								cy="6"
								r="3.4"
								fill="var(--paper)"
								stroke={style.strokeVar}
								stroke-width={style.strokeWidth}
							/>
						{:else if style.shape === 'tick'}
							<line
								x1="22"
								y1="1"
								x2="22"
								y2="11"
								stroke={style.strokeVar}
								stroke-width={style.strokeWidth}
							/>
						{:else}
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
							{:else}
								<line
									x1="22"
									y1="1"
									x2="22"
									y2="11"
									stroke={style.strokeVar}
									stroke-width={style.strokeWidth}
								/>
							{/if}
						{/if}
					</svg>
					<span>{style.panel} · {style.legend}</span>
				</li>
			{/each}
		</ul>

		<div class="matched">
			<h3>A second cutoff, chosen a different way</h3>
			<p>
				Instead of letting each model pick the cutoff that flatters it, give them all the same job:
				lower each model's score cutoff until it has flagged
				{fmtPct(figure.reference.matched.targetErrorRecall)} of the
				{figure.panel.nErrors} errors we already know about, and report what that costs. Scores come
				in steps, so a model usually overshoots that target rather than landing on it, and each row is
				therefore compared against the random forest re-cut to the recall that row actually reached —
				the <strong>margin at matched recall</strong>. The last column does something different: it
				subtracts the random forest at its own cutoff ({fmt4(figure.reference.matched.point.tau)},
				recall
				{fmt4(figure.reference.matched.point.errorRecall)}, F1
				{fmt4(figure.reference.matched.point.errorF1)}) from every row wherever that row landed, so
				the two sides are not matched on recall. It is {unmatchedComparison}. Kept because deleting a
				real quantity loses information.
			</p>
			{#each flipped as lane (lane.id)}
				<p class="flip">
					<strong>{lane.display} is where that matters.</strong>
					Matched it is {fmtDelta(lane.matched.deltaAtMatchedRecall)}; unmatched it reads
					{fmtDelta(lane.matched.deltaEachSideAtItsOwnTargetCut)}. Same statements, opposite sign.
					The matched one is the number to use.
				</p>
			{/each}
			<div class="table-scroll">
				<table>
					<thead>
						<tr>
							<th>model</th>
							<th>its own {fmtPct(figure.reference.matched.targetErrorRecall)} cutoff</th>
							<th>error recall reached</th>
							<th>random forest re-cut there</th>
							<th>margin at matched recall</th>
							<th class="faint"
								>margin with each side at its own cutoff — <em>not</em> recall-matched</th
							>
						</tr>
					</thead>
					<tbody>
						{#each contenders as lane (lane.id)}
							<tr>
								<td>{lane.display}</td>
								<td>{fmt4(lane.matched.point.tau)}</td>
								<td>{fmt4(lane.matched.point.errorRecall)}</td>
								<td
									>{fmt4(lane.matched.referenceAtThisRowsRecall.tau)} · F1
									{fmt4(lane.matched.referenceErrorF1AtThisRow)}</td
								>
								<td class="lead">{fmtDelta(lane.matched.deltaAtMatchedRecall)}</td>
								<td class="faint">{fmtDelta(lane.matched.deltaEachSideAtItsOwnTargetCut)}</td>
							</tr>
						{/each}
					</tbody>
				</table>
			</div>
			<p>
				This second cutoff was picked with the answers in hand too, by a different rule: chosen on
				the same statements it is scored on, so it is no more a held-out result than the first.
			</p>
			<!-- This cutoff is benchmark-fitted too, by a DIFFERENT rule, so it carries
			     its own disclosure rather than borrowing the headline's placement above.
			     Every cutoff-based number on this page renders beside the disclosure
			     that governs its cutoff — in the open, in plain words; the file's own
			     wording of both is in the verification section at the end. -->
			<details class="rule-detail">
				<summary>the second cutoff rule in full, and what choosing it here costs</summary>
				<p class="rule">{figure.matchedThresholdRule.ruleProse.plain}</p>
				<p class="rule oracle">{figure.matchedThresholdRule.oracleProse.plain}</p>
			</details>
			<p class="note">
				Single numbers only: no ranges here, and no correction for testing several models at once, so
				the claim being tested stays the one in chart B.
			</p>
		</div>

		<div class="reading">
			<h3>What the figure shows</h3>
			<ul>
				<li>
					{figure.nWinsSimultaneously} of {figure.nInFamily} reading models beat the random forest even
					after the range around each margin is widened to cover all of them at once — the count checks
					the direction too, so a model whose range clears zero from BELOW is not counted as a win. That
					widening costs {figure.multiplicity.criticalValue.toFixed(3)} standard errors where the
					blunt version of the same correction (Bonferroni) would cost
					{figure.multiplicity.bonferroniCriticalValue.toFixed(3)}: the reading models rise and fall
					together, so covering all four at once is cheap.
				</li>
				{#if winnerCuts.length > 0}
					<li>
						The tuning favours the other side. {figure.reference.display} gets
						{figure.reference.distinctScores.toLocaleString()} different scores to place its cutoff between;
						the reading models that win get {Math.min(...winnerCuts)}–{Math.max(...winnerCuts)}. The
						far finer search still loses.
					</li>
				{/if}
				{#each outsideFamily as lane (lane.id)}
					<li>
						{lane.display} is not one of the reading models that widening covers, so it gets no widened
						range and is quoted on its own: {fmtDelta(deltaOf(lane))}, [{fmtDelta(lane.delta.ciLow)},
						{fmtDelta(lane.delta.ciHigh)}], {zeroClause(lane.delta.standing)}. It is drawn, and it
						counts toward neither the four nor the count above.
					</li>
				{/each}
				{#if figure.modalCutValue !== null}
					<li>
						{figure.modalCutDisplays.join(', ')} all land on the same cutoff, {fmt4(
							figure.modalCutValue
						)}: at that cutoff each one flags exactly the statements whose evidence it had already
						rejected outright. Their best cutoff is the one they would have used untuned, so the
						tuning bought them nothing.
					</li>
				{/if}
				<li>
					This is one cutoff and one decision. It answers what list of statements a curator is handed
					to check; it replaces and withdraws nothing above it.
				</li>
			</ul>
		</div>

		<details>
			<summary>Every value, the cross-check against the review list, and the caveats</summary>

			<div class="table-scroll">
				<table>
					<thead>
						<tr>
							<th>model</th>
							<th>score cutoff</th>
							<th>flagged</th>
							<th>errors caught</th>
							<th>false alarms</th>
							<th>error precision</th>
							<th>error recall</th>
							<th>error F1</th>
							<th>correct F1</th>
							<th>accuracy</th>
							<th>margin in error-class F1</th>
							<th>95% range, this model alone</th>
							<th>95% range covering all four at once</th>
							<th>t</th>
							<th>P(margin&gt;0)</th>
						</tr>
					</thead>
					<tbody>
						{#each lanes as lane (lane.id)}
							<tr>
								<td>{lane.display}</td>
								<td>{fmt4(lane.operating.tau)}</td>
								<td>{lane.operating.flagged}</td>
								<td>{lane.operating.tp}</td>
								<td>{lane.operating.fp}</td>
								<td>{fmt4(lane.operating.errorPrecision)}</td>
								<td>{fmt4(lane.operating.errorRecall)}</td>
								<td>{fmt4(lane.operating.errorF1)}</td>
								<td>{fmt4(lane.operating.correctF1)}</td>
								<td>{fmt4(lane.operating.accuracy)}</td>
								{#if lane.delta}
									<td>{fmtDelta(lane.delta.delta)}</td>
									<td>[{fmtDelta(lane.delta.ciLow)}, {fmtDelta(lane.delta.ciHigh)}]</td>
									<td
										>{lane.delta.simLow !== null && lane.delta.simHigh !== null
											? `[${fmtDelta(lane.delta.simLow)}, ${fmtDelta(lane.delta.simHigh)}]`
											: 'not one of the four — its own range only'}</td
									>
									<td>{lane.delta.tStatistic.toFixed(2)}</td>
									<td>{lane.delta.pGreaterThanZero.toFixed(4)}</td>
								{:else}
									<td colspan="5" class="faint">reference — every margin is measured against it</td>
								{/if}
							</tr>
						{/each}
					</tbody>
				</table>
			</div>
			<p class="table-note">
				What is measured: {figure.prose.metric.plain} The class being detected: {figure.positiveClass}.
				{figure.prose.positiveClassNote.plain}
				{figure.prose.decisionRule.plain} Metric implementation
				<code>{figure.metricImplementation}</code>, generated by
				<code>{figure.generatedBy}</code>.
			</p>
			<p class="table-note">
				{figure.prose.bootstrapDesign.plain}
				{figure.nBootstrap.toLocaleString()} resamples, seed {figure.seed}. Benchmark:
				{figure.panel.id}, {figure.panel.nErrors} errors and {figure.panel.nCorrect} correct,
				{figure.prose.labelProvenance.plain}; ordering {figure.prose.panelOrdering.plain}.
			</p>
			<p class="table-note">
				Only the random forest needs that split: it is fitted, so it is scored on the group it did
				not train on. The reading models are never trained and never see a label, so nothing has to
				be held back from them — they are scored once over all {figure.panel.n.toLocaleString()}
				statements and then given the same fold indices, so that every row here is resampled and
				summarised by the identical estimator.
			</p>
			<p class="table-note">
				Correction for testing all four reading models at once:
				{figure.prose.multiplicityMethod.plain} Combined error rate
				{figure.multiplicity.familyAlpha} across them.
				{figure.prose.multiplicityNote.plain}
			</p>

			<h4>A third cutoff rule: the review list's own</h4>
			<p>
				The review-list figure on this page reaches its cutoff by a third rule — lower the cutoff
				until a target share of the known errors is flagged — and this table checks that the two
				agree statement for statement where they should.
			</p>
			<p class="rule">{figure.reconciliation.thresholdRule.ruleProse.plain}</p>
			<p class="rule oracle">{figure.reconciliation.thresholdRule.oracleProse.plain}</p>
			<div class="table-scroll">
				<table>
					<thead>
						<tr>
							<th>model</th>
							<th>its key in the review-list file</th>
							<th>review-list cutoff</th>
							<th>review-list error F1</th>
							<th>this figure's cutoff</th>
							<th>this figure's F1</th>
							<th>difference</th>
							<th>flags the same statements</th>
						</tr>
					</thead>
					<tbody>
						{#each figure.reconciliation.rows as row (row.id)}
							<tr>
								<td>{row.display}</td>
								<td><code>{row.reviewQueueModelKey}</code></td>
								<td>{fmt4(row.reviewQueueTau)}</td>
								<td>{fmt4(row.reviewQueueErrorF1)}</td>
								<td>{fmt4(row.thisArtifactTau)}</td>
								<td>{fmt4(row.thisArtifactErrorF1)}</td>
								<td>{row.residual.toFixed(6)}</td>
								<td>{row.sameFlagSet ? 'yes' : 'no'}</td>
							</tr>
						{/each}
					</tbody>
				</table>
			</div>
			<p class="table-note">
				Source <code>{figure.reconciliation.source}</code>
				<span title={figure.reconciliation.sha256}>{shortSha(figure.reconciliation.sha256)}</span>,
				target recall {fmt4(figure.reconciliation.sourceTargetRecall)}, largest difference
				{figure.reconciliation.worstResidual.toFixed(6)} against a tolerance of
				{figure.reconciliation.tolerance}, on the same {figure.reconciliation.panelN} statements and
				{figure.reconciliation.panelNErrors} errors. {figure.prose.reconciliationNote.plain}
			</p>

			<h4>Caveats</h4>
			<ul class="caveats">
				{#each figure.prose.caveats as caveat, index (index)}
					<li>{caveat.plain}</li>
				{/each}
			</ul>
		</details>

		<footer>
			<!-- No '—' placeholder for a missing digest: an unpinned artifact says so,
			     because a dash beside a path reads as "pinned, just abbreviated". -->
			<code>{data.artifact_path}</code> · artifact
			{#if data.artifact_sha256}<span title={data.artifact_sha256}
					>{shortSha(data.artifact_sha256)}</span
				>{:else}<span class="sha-missing">not sha-pinned</span>{/if}<br />
			Run plan <code>{figure.multiplicity.runPlanPath}</code>
			<span title={figure.multiplicity.runPlanSha256}
				>{shortSha(figure.multiplicity.runPlanSha256)}</span
			>
			· stages {figure.multiplicity.runPlanStages.join(', ')}
		</footer>
	{/if}
</section>

<style>
	.errorf1 {
		margin-top: 1.5rem;
		padding-top: 1.25rem;
		border-top: 2px solid var(--ink);
	}
	.errorf1 > header {
		display: flex;
		justify-content: space-between;
		align-items: start;
		gap: 1rem;
	}
	.errorf1 h2,
	.gate h2 {
		margin: 0.18rem 0 0;
		font-family: var(--serif);
		font-size: 1.35rem;
		font-weight: 400;
	}
	.errorf1 > header > strong {
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
		max-width: 76ch;
	}
	.primary p + p {
		margin-top: 0.5rem;
	}
	.primary strong {
		color: var(--ink);
	}
	/* The oracle disclosure is a load-bearing qualification, so it sits in the
	   open beside the claim it qualifies — never behind a <details>. */
	.disclosure {
		margin-top: 0.8rem;
		padding: 0.7rem 0.9rem;
		border: 1px solid var(--blocked);
		border-left-width: 3px;
		background: color-mix(in srgb, var(--blocked) 3%, transparent);
	}
	.disclosure h3,
	.matched h3,
	.reading h3 {
		margin: 0 0 0.4rem;
		font-family: var(--mono);
		font-size: 0.7rem;
		font-weight: 500;
		letter-spacing: 0.03em;
		color: var(--ink);
		text-transform: uppercase;
	}
	.disclosure p,
	.matched p {
		margin: 0 0 0.5rem;
		font-family: var(--serif);
		font-size: 0.8rem;
		line-height: 1.55;
		color: var(--ink-muted);
		max-width: 82ch;
	}
	.disclosure p:last-child,
	.matched p:last-child {
		margin-bottom: 0;
	}
	/* Shipped rule text: the threshold rule itself, quoted rather than paraphrased. */
	.disclosure .rule,
	.matched .rule,
	details .rule {
		font-family: var(--mono);
		font-size: 0.66rem;
		color: var(--ink-faint);
		border-left: 1px solid var(--rule);
		padding-left: 0.6rem;
	}
	/* An oracle disclosure is a caveat, not a footnote: --blocked is this page's
	   token for "read this before you use the number", and it marks the shipped
	   oracle text wherever a second or third threshold rule quotes it. */
	.matched .rule.oracle,
	details .rule.oracle {
		border-left-color: var(--blocked);
		color: var(--ink-muted);
	}
	/* A cutoff rule and its disclosure in full, one click away, in ordinary words.
	   The SHORT FORM of the disclosure is always in the open beside the numbers it
	   governs; this is the same thing at length, for a reader who wants all of it.
	   The result file's own wording lives in the verification section at the end
	   of the page, and never here. It must not read as a section of its own. */
	.disclosure .rule-detail,
	.matched .rule-detail {
		margin: 0 0 0.5rem;
		border-top: 0;
		padding-top: 0;
	}
	.disclosure .rule-detail summary,
	.matched .rule-detail summary {
		font-family: var(--mono);
		font-size: 0.64rem;
		color: var(--ink-faint);
		cursor: pointer;
	}
	.disclosure .rule-detail[open] summary,
	.matched .rule-detail[open] summary {
		margin-bottom: 0.4rem;
	}
	.disclosure .rule-detail p,
	.matched .rule-detail p {
		margin-bottom: 0.4rem;
	}
	.matched {
		margin-top: 1.1rem;
		padding-top: 0.6rem;
		border-top: 1px solid var(--rule);
	}
	.matched .flip {
		border-left: 3px solid var(--accent);
		padding-left: 0.6rem;
		color: var(--ink);
	}
	.matched .note {
		font-family: var(--mono);
		font-size: 0.66rem;
		color: var(--ink-faint);
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
	.rule-line {
		stroke: var(--ink);
		stroke-width: 2.4;
	}
	.rule-wash {
		fill: color-mix(in srgb, var(--ink) 8%, transparent);
	}
	.rule-over {
		stroke: var(--ink);
		stroke-width: 0.9;
		opacity: 0.55;
	}
	.lane-rule {
		stroke: var(--rule);
		stroke-width: 1;
		stroke-dasharray: 2 4;
	}
	.span {
		stroke: var(--ink-faint);
		stroke-width: 1;
		opacity: 0.55;
	}
	.bar {
		stroke-linecap: butt;
	}
	.tick,
	.lane-label,
	.lane-sub,
	.readout,
	.panel-head,
	.axis-label,
	.axis-note,
	.rule-label {
		font-family: var(--mono);
	}
	.tick {
		font-size: 8px;
		fill: var(--ink-faint);
		text-anchor: middle;
	}
	/* Right-anchored: budgeted at 34 chars in buildFigure(). */
	.lane-label {
		font-size: 9px;
		fill: var(--ink);
		text-anchor: end;
	}
	.lane-label.is-reference {
		fill: var(--accent);
	}
	.lane-sub {
		font-size: 7.5px;
		fill: var(--ink-faint);
		text-anchor: end;
	}
	/* Left-anchored: budgeted at 19 and 17 chars in buildFigure(). */
	.readout {
		font-size: 8px;
		fill: var(--ink-muted);
		text-anchor: start;
		font-variant-numeric: tabular-nums;
	}
	.readout.faint {
		fill: var(--ink-faint);
	}
	.panel-head {
		font-size: 9px;
		fill: var(--ink);
		text-anchor: start;
	}
	.axis-label {
		font-size: 9px;
		fill: var(--ink-muted);
		text-anchor: middle;
	}
	.axis-note {
		font-size: 8px;
		fill: var(--blocked);
		text-anchor: middle;
	}
	.rule-label {
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
		max-width: 80ch;
	}
	.reading li + li {
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
	details h4 {
		margin: 1rem 0 0.35rem;
		font-family: var(--mono);
		font-size: 0.68rem;
		font-weight: 500;
		color: var(--ink);
		text-transform: uppercase;
		letter-spacing: 0.03em;
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
		vertical-align: bottom;
	}
	th,
	td {
		padding: 0.34rem 0.45rem;
		border-bottom: 1px dotted var(--rule);
		vertical-align: top;
		white-space: nowrap;
	}
	td:nth-child(n + 2) {
		font-variant-numeric: tabular-nums;
	}
	td.lead {
		color: var(--ink);
		font-weight: 500;
	}
	td.faint,
	th.faint {
		color: var(--ink-faint);
	}
	.caveats {
		margin: 0.4rem 0 0;
		padding-left: 1.1rem;
	}
	.caveats li {
		font-family: var(--serif);
		font-size: 0.78rem;
		line-height: 1.5;
		color: var(--ink-muted);
		max-width: 88ch;
	}
	.caveats li + li {
		margin-top: 0.35rem;
	}
	.table-note {
		margin: 0.6rem 0 0;
		font-family: var(--mono);
		font-size: 0.62rem;
		line-height: 1.55;
		color: var(--ink-faint);
		max-width: 92ch;
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
		.errorf1 > header {
			display: block;
		}
		.errorf1 > header > strong {
			display: inline-block;
			margin-top: 0.6rem;
		}
	}
</style>
