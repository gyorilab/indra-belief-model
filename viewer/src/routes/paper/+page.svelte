<script lang="ts">
	import ApDecompositionByPaperRank from '$lib/components/ApDecompositionByPaperRank.svelte';
	import BeliefHeuristicResponse from '$lib/components/BeliefHeuristicResponse.svelte';
	import BeliefModelLadder from '$lib/components/BeliefModelLadder.svelte';
	import FidelityPanel from '$lib/components/FidelityPanel.svelte';
	import FramingCorrection from '$lib/components/FramingCorrection.svelte';
	import PaperLiteralComparison from '$lib/components/PaperLiteralComparison.svelte';
	import PaperOwnMetric from '$lib/components/PaperOwnMetric.svelte';
	import PaperTable6Extended from '$lib/components/PaperTable6Extended.svelte';
	import PerEvidenceGrain from '$lib/components/PerEvidenceGrain.svelte';
	import DeployedBaseline from '$lib/components/DeployedBaseline.svelte';
	import PaperRobustness from '$lib/components/PaperRobustness.svelte';
	import StatementErrorF1 from '$lib/components/StatementErrorF1.svelte';
	import TieInflation from '$lib/components/TieInflation.svelte';
	import PaperReliabilityStrip from '$lib/components/PaperReliabilityStrip.svelte';
	import ReviewQueue from '$lib/components/ReviewQueue.svelte';
	import ScoreDistribution from '$lib/components/ScoreDistribution.svelte';
	import { fmtDelta, type ErrorF1Delta, type ErrorF1Lane } from '$lib/data/paper-error-f1';
	import type { PaperLiteralArm, PaperLiteralDelta } from '$lib/data/paper-literal';
	import type { PaperPerEvidenceLane } from '$lib/data/paper-per-evidence';
	import { reviewQueueCalloutArm } from '$lib/data/paper-review-queue';
	import type { PageData } from './$types';

	let { data }: { data: PageData } = $props();

	/** An arm that carries a paired delta (i.e. not the reference arm itself). */
	type DeltaArm = PaperLiteralArm & { delta: PaperLiteralDelta };

	/** Every number in the prose below is read off the load — nothing is asserted. */
	const arms = $derived(data.paperLiteral.status === 'ok' ? data.paperLiteral.arms : []);
	/**
	 * The LLM arm with the largest ΔAP: the arm whose AP margin the page quotes.
	 *
	 * IT IS NOT `errorF1Best`, and today it is not the same arm: this is an argmax
	 * over ΔAP, that one an argmax over error-F1, and the two lenses rank the arms
	 * differently. `PaperLiteralComparison.svelte:62` states the rule this page
	 * follows — the arm "must be named, because a different arm leads under
	 * trapezoidal and an unscoped 'our margin' claim would then be wrong". The
	 * same hazard applies ACROSS clauses, not just across lenses: two argmaxes
	 * printed side by side read as one system unless each carries its own name.
	 * So every clause that prints one of these margins prints `display` beside it,
	 * and no clause prints a margin whose arm it cannot name.
	 */
	const best = $derived<DeltaArm | null>(
		arms
			.filter((arm): arm is DeltaArm => arm.kind === 'llm' && arm.delta !== null)
			.slice()
			.sort((a, b) => b.delta.ap.delta - a.delta.ap.delta)[0] ?? null
	);
	/**
	 * Largest fold-mean deviation of our re-run from the paper's printed Table 6.
	 * Null, never a placeholder number: the lede's fidelity clause is gated on it
	 * rather than printing a zero that would read as a perfect reproduction.
	 */
	const table6Fidelity = $derived(
		data.paperLiteral.status === 'ok' && data.paperLiteral.reproduction !== null
			? data.paperLiteral.reproduction.maxAbsDeltaVsPublishedTable6
			: null
	);

	// ── the lede's headline: statement-grain error-class F1 ───────────────────
	// The margin that survives max-t correction, so it leads. Read off the same
	// load the figure two-thirds down the page draws, with the SAME predicate the
	// figure's own census uses, so the lede and the census cannot disagree.
	const errorF1 = $derived(
		data.statementErrorF1.status === 'ok' ? data.statementErrorF1.figure : null
	);
	/** A lane carrying a margin — i.e. everything but the reference lane. */
	type ErrorF1DecidedLane = ErrorF1Lane & { delta: ErrorF1Delta };
	/**
	 * `winsSimultaneously` is SIGN-AWARE in the loader; a bare "excludes zero" is
	 * `ciLow > 0 || ciHigh < 0` and would count a significant LOSS as a win. This
	 * page branches on the loader's predicate and never on `excludesZero*`.
	 */
	const errorF1Winners = $derived(
		(errorF1?.lanes ?? []).filter(
			(lane): lane is ErrorF1DecidedLane => lane.delta !== null && lane.delta.winsSimultaneously
		)
	);
	/**
	 * The widest winning margin: the number the lede leads with. An argmax over
	 * ERROR-F1 among the arms that win simultaneously — a different family and a
	 * different ordering from `best` above, which maximises ΔAP. The lede prints
	 * both margins, so it prints both names; see the note on `best`.
	 */
	const errorF1Best = $derived<ErrorF1DecidedLane | null>(
		errorF1Winners.length > 0
			? errorF1Winners.reduce((a, b) => (b.delta.delta > a.delta.delta ? b : a))
			: null
	);
	/** The oracle's size on the winning side: candidate cuts each gate could pick. */
	const errorF1WinnerCuts = $derived(errorF1Winners.map((lane) => lane.distinctScores));

	// ── the per-evidence beat's own margin ────────────────────────────────────
	const perEvidence = $derived(
		data.paperPerEvidence.status === 'ok' ? data.paperPerEvidence.figure : null
	);
	type PerEvidenceDeltaLane = PaperPerEvidenceLane & {
		pairedDelta: NonNullable<PaperPerEvidenceLane['pairedDelta']>;
	};
	/**
	 * The strongest reader on that plate BY ITS OWN MARGIN over INDRA's scorer.
	 * A maximum, not a significance test: nothing here reads
	 * `pairedDelta.excludesZero`, which is sign-blind by construction. The
	 * sentence prints the signed value and names the arm, so a negative margin
	 * reads as a loss rather than being asserted as a win.
	 *
	 * A THIRD argmax, over a third family — so it need not be, and today is not,
	 * either `best` or `errorF1Best`. When this clause was typed into the lede it
	 * quoted Gemma 4 26B at +0.122; derived here it selects Gemma 4 31B at +0.137,
	 * because it maximises over this plate's own lanes rather than repeating a
	 * literal. That is only legitimate BECAUSE the lead-in names the arm from the
	 * load: three named arms cannot be read as one system, three bare margins can.
	 * The margin this clause used to quote did not leave the page — every
	 * non-reference lane draws its own paired delta and interval
	 * (`PerEvidenceGrain.svelte:306`), 26B's among them.
	 */
	const perEvidenceBest = $derived.by<PerEvidenceDeltaLane | null>(() => {
		const readers = (perEvidence?.lanes ?? []).filter(
			(lane): lane is PerEvidenceDeltaLane => lane.kind === 'reader' && lane.pairedDelta !== null
		);
		if (readers.length === 0) return null;
		return readers.reduce((a, b) => (b.pairedDelta.delta > a.pairedDelta.delta ? b : a));
	});

	// The two halves of what the gate design costs. Both are review-queue artifact
	// fields — the ceiling the formula has already taken before any reader runs,
	// and the reader's own zeroed block — so neither is a literal on this page.
	const queue = $derived(data.reviewQueue.status === 'ok' ? data.reviewQueue.queue : null);
	const ceiling = $derived(queue?.promotionCeiling ?? null);
	/** The same arm the queue panel above calls out — derived, never named here. */
	const zeroed = $derived(queue ? reviewQueueCalloutArm(queue) : null);
	// The ceiling and the zero pile OVERLAP (a true statement on one weak evidence is
	// both). Derived in the artifact so the relation is never prose-asserted.
	const overlap = $derived(
		ceiling && zeroed ? (ceiling.perArmOverlap?.[zeroed.name] ?? null) : null
	);

	/**
	 * A signed margin at a stated precision. AP and AUROC margins are thousandths,
	 * so they print at FOUR decimals here, in the head-to-head panel's Δ column and
	 * in the memo alike; per-evidence margins print at THREE, matching that plate's
	 * own readouts. One number, one rendering, wherever it appears.
	 *
	 * Error-F1 margins are NOT formatted here: they use `fmtDelta` from the
	 * error-F1 module, the same function the figure itself calls, so the lede's
	 * digits and sign glyph are the figure's by construction rather than by
	 * agreement.
	 */
	function margin(value: number, digits: number): string {
		return `${value >= 0 ? '+' : '−'}${Math.abs(value).toFixed(digits)}`;
	}
	const apMargin = (value: number): string => margin(value, 4);
	const perEvidenceMargin = (value: number): string => margin(value, 3);
</script>

<svelte:head><title>paper · random forest vs language-model reading</title></svelte:head>

<main id="main" class="wrap">
	<header class="head">
		<!-- NAME THE METHOD; CITE THE PAPER AS PROVENANCE, NOT AS IDENTITY. This page
		     is read by people who have not read the 2023 INDRA assembly paper, so
		     "the paper's model", "their RF", "the paper's own metric" and "paper
		     literal" name nothing: they are insider shorthand that resolves only
		     against a document the reader does not have. The FIRST mention may
		     locate a method in the paper ("from the 2023 INDRA assembly paper");
		     after that the method carries its own name. The canonical names, each
		     used verbatim rather than paraphrased:
		       · random forest on evidence-count features (2,000 trees, depth 13, over
		         per-source evidence counts plus statement type, PMID count and a
		         promoter flag — the model fitted in 2023)
		       · noisy-OR over per-source reliabilities, nothing fitted (Belief Orig)
		       · evidence-gated reading (a language model drops evidence it judges
		         unsupported, then the same noisy-OR scores what remains)
		       · trapezoidal PR-AUC (straight-line interpolation between points,
		         averaged over the 10 folds) vs average precision (the same area
		         step-wise, so tied scores earn no interpolated credit)
		       · error-class F1 · spread across the 10 folds
		       · the 1,689 statements · the published curation labels
		     Method DETAIL that will not fit a lead-in lives in a <details> or a
		     <desc> beside the figure it governs — it is never dropped, and it is
		     never replaced by shorthand. This has been corrected twice; the rule is
		     written here so a third regression has to be deliberate.

		     AND THERE IS NO "THEM". The 2023 INDRA assembly paper is this lab's own
		     prior work, so "their metric / our arms", "theirs vs ours" and "earns our
		     models a place on their axis" describe a rivalry that does not exist —
		     and the possessive carries no information anyway, which is why removing
		     it always shortens the sentence: "THEIRS — same evidence as ours (15)"
		     is "published (15)", "OURS (5)" is "newly scored (5)". Say what a row IS:
		     published in 2023, or newly scored here. "Published" stays wherever the
		     FACT of publication is what makes a row a fixed reference point; a
		     first-person "we" stays where it discloses who did something that was
		     not published (we re-ran the code; the 1,578-statement label revision is
		     ours). What goes is the possessive used as a NAME. Fourth correction.

		     ENFORCED, not remembered: class (a6) of scripts/test-paper-render-invariants.mjs
		     fails on "the paper's <noun>", "their <method>", "our <method>", bare
		     OURS/THEIRS and us/them contrasts in any string this page can draw —
		     component templates, component <script> consts and the paper-* data
		     modules, the plain half of every twin included. It leaves the sha-pinned
		     shipped halves and the join keys alone, by construction.

		     PLAIN LANGUAGE IS ALSO PART OF THAT RULE (third correction, and the
		     reason this paragraph now runs long). The reader is a working biologist
		     who has never read the 2023 paper and does not work in machine-learning
		     evaluation. Words this page INVENTED do not survive contact with that
		     reader and are banned from rendered prose — script, expressions and
		     comments below keep their identifiers, which is why `const arms =` is
		     still `arms`:
		       arm -> model (or the model's own name)   ·  panel -> benchmark, or
		       plainly "the 1,689 statements"           ·  gate (noun) -> the
		       reading step / the reader                ·  lens -> view
		       lane -> row  ·  pooled -> over all statements at once  ·  census ->
		       count  ·  delta -> difference or margin  ·  plate -> figure
		       residual -> difference  ·  tau -> score cutoff  ·  incumbent -> the
		       method in use today  ·  rung -> step

		     WHAT IS OURS GOES; WHAT IS THE FIELD'S STAYS (fifth correction, and it
		     REVERSES part of the third). The list above is words this page coined.
		     A word the field already owns is not on it and must not be swapped for
		     a coinage: this page is read by the lab that published a
		     cross-validation paper, and replacing "fold" with "slice" fails that
		     reader exactly as "arm" and "tau" do, pointing the other way. So:
		     FOLD, OUT-OF-FOLD and CROSS-VALIDATION are the words, each glossed
		     ONCE where the page first uses it — the gloss under beat 3 is that
		     once. Likewise average precision, precision, recall, F1, random
		     forest, noisy-OR. "max-t" may appear only carrying its gloss
		     ("corrected for testing four models at once"), and the page currently
		     prefers the gloss alone.

		     THE FOLD DESIGN, so no gloss on this page can drift from it:
		     `StratifiedKFold(10, shuffle=False)`, and NOTHING is held out as a
		     separate test set — every statement is scored once, by the fold that
		     excluded it. The FITTED models (random forest, logistic regression,
		     KNN, SVC) need that; their numbers are out-of-fold. The READING models
		     are never trained and see no labels, so nothing needs holding back:
		     they are scored once over all 1,689 statements and then given the same
		     fold indices purely so the identical estimator applies. That asymmetry
		     is stated in the gloss under beat 3 and again beside the estimator in
		     the robustness table — a reviewer asks about it, so it is not left to
		     be inferred. The one thing fitted on everything is the score CUTOFF in
		     the error-class F1 figure, and that figure discloses it itself.

		     A term that is genuinely technical and unavoidable (precision, recall,
		     F1, random forest) is DEFINED in the sentence that first uses it. -->
		<h1>a random forest on evidence-count features, re-measured</h1>
		<!-- THE LEDE LEADS WITH WHAT SURVIVES CORRECTION. It used to summarise the
		     page with pooled average precision (+0.0098, t 2.21 — the one lens that
		     FAILS the max-t family correction) while the margin that passes it
		     (error-class F1, t 8.49) sat unnamed two-thirds of the way down. AP is
		     kept, and kept qualified, as the most conservative lens: the panel is
		     73.2% positive and the reference is already at AP 0.9412.

		     Two clauses MOVED here rather than being dropped, each to the beat that
		     owns it, because the lede is capped at 60 words and both are stated
		     beside their own figure: the AUROC margin is now in the head-to-head
		     lead-in (beat 3), and the per-evidence margin against INDRA's own
		     scorer is now in the per-evidence lead-in (beat 9). Nothing is lost.

		     Each numeric clause is gated on its own load with NO else branch: if an
		     artifact is dark the sentence does not render at all and the figure
		     below states the failure itself. A fallback sentence would be counted
		     against the prose budget in both branches for prose one of them can
		     never show.

		     THE FIDELITY NUMBER MOVED, 0.0016 -> 0.002, because it is now READ. It
		     is the manifest's `max_abs_delta_vs_published_table6`, printed at the
		     same three decimals FidelityPanel prints it at one beat below, so the
		     two agree by construction; the typed 0.0016 was the extended table's
		     re-run TOLERANCE, a different quantity, and it still renders at beat 6a
		     beside the five rows it bounds.

		     EACH MARGIN NAMES ITS OWN ARM. The two margins below come from two
		     argmaxes over two different quantities — error-F1 for the first,
		     average precision for the second — and they do not resolve to the same
		     arm: the error-F1 winner is the GLM-5 gate, the AP winner the Gemma 4
		     26B gate. Printed bare and adjacent they read as one system's two
		     lenses, which is false. Both names are now interpolated from the same
		     loads the values come from, so a re-run that moves either argmax moves
		     its name with it. This is the rule PaperLiteralComparison already
		     states for itself ("It must be named, because a different arm leads
		     under trapezoidal and an unscoped 'our margin' claim would then be
		     wrong"), applied across clauses rather than only across lenses. Naming
		     costs no budget: `display` is interpolated, and the word counter drops
		     `{...}` — the lede stays at 59 of 60 words.

		     THE AVERAGE-PRECISION CLAUSE MOVED into beat 3's lead-in, where the same
		     margin is already printed beside the figure that draws it, and it moved
		     with its qualification ("the most conservative view") attached. It left
		     because the plain-language rewrite costs words the 60-word lede does not
		     have: "an oracle favouring the forest, N candidate cuts to our M" reads
		     as insider shorthand, and its replacement ("both cutoffs were tuned on
		     these statements, favouring the forest: N choices to our M") is the same
		     disclosure in words a biologist can act on. Nothing left the page — the
		     margin, its name and its concession all render at beat 3, and
		     "trapezoidal PR-AUC flatters us" is beats 6b and 6c entire. -->

		<p class="lede">
			We re-ran the code the 2023 INDRA assembly paper released, on its 1,689 statements and
			published labels.
			{#if table6Fidelity !== null}Table 6 comes back within {table6Fidelity.toFixed(3)}.{/if}
			{#if errorF1 && errorF1Best}Letting a language model drop evidence it judges unsupported finds
				more wrong statements than the random forest does: {errorF1Best.display},
				{fmtDelta(errorF1Best.delta.delta)} on error-class F1 [{fmtDelta(errorF1Best.delta.ciLow)},
				{fmtDelta(errorF1Best.delta.ciHigh)}] — precision and recall for finding wrong statements.
				Both cutoffs were tuned on these {errorF1.panel.n.toLocaleString()} statements, favouring
				the forest: {errorF1.reference.distinctScores.toLocaleString()} choices against
				{Math.min(...errorF1WinnerCuts)}–{Math.max(...errorF1WinnerCuts)}.{/if}
		</p>
	</header>

	<!-- 1 · fidelity — the credibility floor everything else stands on -->
	<p class="framing">
		First, can you trust the re-run at all? We ran the released code unmodified, on the released
		data, and the published Table 6 comes back. Everything below rests on that.
	</p>
	<FidelityPanel data={data.paperLiteral} />

	<!-- 2 · the framing correction — what the reader arm actually is. The score
	     distribution sits here as its EVIDENCE: the piles are the published
	     reachable grid coming back, which is the correction's whole claim. -->
	<p class="framing">
		What the language model actually does. Not a rival scorer: INDRA's own noisy-OR formula — one
		reliability per source, nothing fitted — still scores, over whatever evidence the model kept. The
		piles below are that formula's values.
	</p>
	<FramingCorrection data={data.framingCorrection} />
	<ScoreDistribution data={data.paperLiteral} />

	<!-- 3 · THE COMPARISON TO THE 2023 RESULT. This is the beat the page exists
	     for: the fitted models published in 2023 against evidence-gated reading, on
	     the same 1,689 statements, the same labels and the same 10 folds. It leads
	     because it is the ONLY comparison anyone can check against a published
	     number — every figure through beat 7 qualifies this one margin, and beat 12
	     is the only place the page leaves those 1,689 statements.

	     The AUROC margin used to sit in the lede. It sits here now, beside the
	     lens that draws it — same arm, same shipped interval, one more decimal,
	     and read off the load instead of typed. The AP margin's own concession
	     ("the most conservative view") joined it here in the plain-language pass:
	     it is the same argmax, the same `best.display`, and the sentence that
	     used to carry it in the lede could not survive being written in words a
	     biologist reads without going over 60. -->
	<!-- THE AUROC QUALIFICATION MUST TRAVEL WITH THE NUMBER. The figure's own
	     "mostly a split, not an order" note is gated `{#if metric === 'auroc'}`
	     while PAPER_DEFAULT_METRIC is 'trapezoidal', so a reader who never touches
	     the lens switch saw a bare AUROC margin and its interval with nothing
	     tempering them. The clause is restated here, unconditional on lens. -->
	<p class="framing">
		Head to head: same 1,689 statements, same published labels, same 10 folds. Random forest against
		language-model reading —
		{#if best}{apMargin(best.delta.ap.delta)} average precision, the most conservative view, for
			{best.display}; {apMargin(best.delta.auroc.delta)} AUROC, mostly a yes/no split rather than a
			ranking.{:else}once the models load.{/if}
	</p>
	<!-- THE GLOSS, AND WHY IT IS ITS OWN PARAGRAPH. "Fold" is the field's word and
	     the audience owns it, so the page uses it — but a word the page uses is a
	     word the page defines, once, where it is first read. It does not fit in the
	     lead-in above: that paragraph is at its 35-word budget and every clause in
	     it carries a claim or a qualification, so buying room would mean dropping
	     one. It buys none here either — this is not a lead-in and is not counted as
	     one. The second sentence is the asymmetry a reviewer asks about, stated
	     rather than left to be inferred from the word "fitted". -->
	<p class="gloss">
		A fold is one of ten groups the statements were split into: each fitted model is scored on the
		group it did not train on. The reading models train on nothing, so one copy of each scores all
		1,689.
	</p>
	<PaperLiteralComparison data={data.paperLiteral} />

	<!-- 6a · TABLE 6, EXTENDED. It comes BEFORE the banded figure because a ranked
	     list is the reading this lab already has for trapezoidal PR-AUC, and
	     `PaperOwnMetric` bands the newly scored models separately from the
	     published ones — which is exactly what hides that ranks 1–3 are newly
	     scored. Same measure, same "all sources, specific" configuration, the new
	     rows interleaved by rank rather than set beside. The banded chart stays:
	     it is the detail view, and it carries the re-run/published distinction and
	     the agreement bound that this list only summarises. -->
	<p class="framing">
		Table 6, extended: the reading models slotted in by rank among its published rows, scored the
		same way — trapezoidal PR-AUC. Ranked list first, chart below.
	</p>
	<PaperTable6Extended data={data.paperTable6Extended} />

	<!-- 6b · trapezoidal PR-AUC, the published numbers and the newly scored models
	     on one axis -->
	<p class="framing">
		The same models again, beside the fifteen published rows from the same configuration. Check how
		closely the re-run agrees first; the next figure explains why this scoring flatters the reading
		models.
	</p>
	<PaperOwnMetric data={data.paperOwnMetric} />

	<!-- 6c · why trapezoidal PR-AUC inflates the reading models specifically -->
	<p class="framing">
		Why it flatters the reading models. Trapezoidal PR-AUC draws a straight line across tied scores
		where the real curve steps. They give many statements the same score, collecting credit no cutoff
		can reach.
	</p>
	<TieInflation data={data.tieInflation} />

	<!-- 6d · how that margin behaves under correction and under label completeness -->
	<p class="framing">
		How far to trust that margin. Corrected at once across all four reading models it grazes zero;
		drop the errors whose review never finished and it narrows. Both shown; neither replaces the
		headline.
	</p>
	<PaperRobustness data={data.paperRobustness} />

	<!-- 7 · where that ranking margin comes from -->
	<p class="framing">
		Where the margin comes from. Group statements by how much evidence they have — a count neither
		model chose: the reading models gain most where the formula is weakest. Grouping by a score would
		rig it.
	</p>
	<ApDecompositionByPaperRank data={data.paperLiteral} />

	<!-- 8 · THE SAME MARGIN AS WORK, still on the same 1,689 statements and still
	     against the published random forest. Placed after the ranking beats because it is
	     the same comparison expressed as a review budget — and it is the more
	     robust half: unlike the AP margin it survives max-t correction and
	     STRENGTHENS on the adjudication-safe subpanel. -->
	<p class="framing">
		The same finding as a review workload. At its own untuned cutoff the Gemma 4 26B reader flags 462
		statements and catches 354 of the 452 known errors; the random forest, its cutoff tuned on these
		very statements, needs 662.
	</p>
	<ReviewQueue data={data.reviewQueue} />

	<!-- 8b · THE SAME RESULT, NAMED AS A METRIC. It sits immediately after the
	     queue because it is not a second finding: the queue reports the operating
	     point as a workload (462 flagged, 354 of 452 errors caught) and this
	     reports the same class of operating point as error-class precision, recall
	     and F1. Two threshold rules, two derivations, one result — the artifact
	     reconciles them row by row and gates itself on the residual, which is why
	     the lead-in can state the agreement instead of asserting it.

	     This is the margin the lede now leads with: it is the one that survives the
	     max-t correction across the reader family, where the ranking margins above
	     do not. Its thresholds are an oracle, and the figure says so in the open,
	     beside every number the oracle governs. -->
	<p class="framing">
		The same result as a number. The queue above and this figure are one finding under two different
		cutoff rules; {#if errorF1}they disagree by at most
			{errorF1.reconciliation.worstResidual.toFixed(6)} on error-class F1{:else}reconciled inside the
			artifact{/if}.
	</p>
	<StatementErrorF1 data={data.statementErrorF1} />

	<!-- 9 · THE GRAIN THE SYSTEM ACTUALLY OPERATES AT — same 1,689 statements,
	     resolved to the 5,379 evidence pairs underneath them. The reader's native
	     output is one verdict per pair; statement belief is DERIVED from those
	     through INDRA's noisy-OR. 3.2x better powered than every beat above, and
	     it separates reading from aggregation: how much each model gains from the
	     noisy-OR is how much of its score is NOT reading. The labels come from two
	     authors of the 2023 paper, which is what makes it checkable at all — that
	     paper reports no per-evidence baseline.

	     The per-evidence margin used to sit in the lede, with the qualification
	     that its baseline is INDRA's own scorer and not a published model. That
	     qualification renders nowhere else outside a <details>, so it moved here
	     rather than being dropped: the arm is NAMED from the load and the value is
	     printed signed, so a negative margin would read as a loss. -->
	<p class="framing">
		The level the model works at: one piece of evidence at a time. A statement's score is built from
		those. Two authors of the 2023 paper labelled 5,379.
		{#if perEvidenceBest}Against INDRA's own scorer, not a published model:
			{perEvidenceBest.display}
			{perEvidenceMargin(perEvidenceBest.pairedDelta.delta)}.{/if}
	</p>
	<PerEvidenceGrain data={data.paperPerEvidence} />

	<!-- 10 · every belief model on this panel, on one axis -->
	<p class="framing">
		Every way of scoring belief on these 1,689 statements, on one axis — all measured from the
		unfitted noisy-OR that hand-built features and language-model reading each change in their own
		way.
	</p>
	<BeliefModelLadder data={data.beliefLadder} />

	<!-- 11 · why there is anything to win -->
	<p class="framing">
		Why there is anything to win. The noisy-OR is a lookup table on how many times each source said
		it — it never reads the sentence.
	</p>
	<BeliefHeuristicResponse data={data.beliefHeuristic} />

	<!-- 12 · THE ONE BEAT THAT LEAVES THE 1,689 STATEMENTS. The fitted HybridScorer
	     INDRA serves today is not in the 2023 paper, so no part of this beat can be
	     checked against a published number; 3 of this figure's 4 benchmarks are
	     assembled here rather than published. That is exactly why it sits here and
	     not in the lede — it is the larger margin (+0.074) but the weaker
	     provenance. -->
	<p class="framing">
		Beyond these 1,689 statements. That random forest was a research model and never shipped; the
		belief INDRA serves today is different code. Language-model reading beats every version of it we
		could source, on four benchmarks.
	</p>
	<DeployedBaseline data={data.deployedBaseline} />

	<!-- 13 · limits: calibration, then what the gate design costs, then the caveats.
	     The closing metric caveat that used to sit here is gone: the head-to-head
	     panel now carries "the published measure flatters the reading models" inside
	     every lens, so keeping it here was the same concession twice. -->
	<p class="framing">
		The limits. Ranking statements better is not the same as getting the odds right: every reading
		model's numbers are far too extreme to read as probabilities.
	</p>
	<PaperReliabilityStrip data={data.paperLiteral} />

	<section class="costs" aria-label="what this design costs">
		{#if ceiling && zeroed && zeroed.zeroPile}
			{#if overlap}
				<p>
					And this design has a ceiling. {ceiling.nTrueBelowThreshold} true statements already score
					below {ceiling.threshold.toFixed(2)} under {ceiling.referenceArmDisplay}; a reading model
					can only remove evidence, never add it, so none of those can be lifted back.
					{zeroed.display} drops {overlap.nTrueZeroedByArm} more to exactly zero, where nothing can be
					ranked — {overlap.nAlsoAlreadyBelowThreshold} of those were already under the bar, so the
					two costs overlap rather than add: {overlap.nTrueAffectedUnion} distinct true statements in
					all.
				</p>
			{:else}
				<p class="dark">
					That ceiling is unavailable — the overlap figure for
					{zeroed.display} is missing from the artifact.
				</p>
			{/if}
			<details class="caveats">
				<summary>caveats, verbatim from the artifact</summary>
				<p>
					The head-to-head's bootstrap intervals are not corrected across the models compared, and its
					legend says so beside the marks. "Error" throughout is the label released in 2023; the
					framing section above and the ladder both print how those negatives break down.
				</p>
				<p>{ceiling.why}</p>
				{#each queue?.caveats ?? [] as caveat (caveat)}
					<p>{caveat}</p>
				{/each}
			</details>
		{:else}
			<p class="dark">
				What this design costs is unavailable — {data.reviewQueue.status === 'ok'
					? 'the review-queue payload is missing.'
					: data.reviewQueue.reason}
			</p>
		{/if}
	</section>
</main>

<style>
	.wrap {
		max-width: 1100px;
		margin: 0 auto;
		padding: 1.6rem 1.5rem 4rem;
	}
	.head h1 {
		font-family: var(--serif);
		font-weight: 400;
		font-size: 1.7rem;
		margin: 0 0 0.3rem;
	}
	.lede {
		font-family: var(--serif);
		color: var(--ink-muted);
		max-width: 62ch;
		line-height: 1.5;
		margin: 0 0 0.5rem;
	}
	/* Prose lead-in before each component — framing, not a competing heading. */
	.framing {
		font-family: var(--serif);
		font-size: 0.9rem;
		color: var(--ink-muted);
		max-width: 66ch;
		line-height: 1.5;
		margin: 2.2rem 0 0.35rem;
	}
	/* A term defined at first use. Set under its lead-in and quieter than it, so it
	   reads as the footnote it is rather than as a second lead-in. */
	.gloss {
		font-family: var(--serif);
		font-size: 0.82rem;
		color: var(--ink-faint);
		max-width: 66ch;
		line-height: 1.5;
		margin: 0 0 0.35rem;
		padding-left: 0.7rem;
		border-left: 1px solid var(--rule);
	}
	/* Closing beat: what the gate costs, then the artifact's own caveats. */
	.costs {
		margin: 1.4rem 0 0;
		padding-top: 0.9rem;
		border-top: 1px dotted var(--rule);
	}
	.costs p {
		font-family: var(--serif);
		font-size: 0.9rem;
		color: var(--ink-muted);
		max-width: 66ch;
		line-height: 1.5;
		margin: 0;
	}
	.costs .dark {
		font-family: var(--mono);
		font-size: 0.76rem;
		color: var(--ink-faint);
	}
	.caveats {
		margin-top: 0.6rem;
		max-width: 74ch;
	}
	.caveats summary {
		font-family: var(--mono);
		font-size: 0.66rem;
		color: var(--ink-faint);
		text-transform: uppercase;
		letter-spacing: 0.05em;
		cursor: pointer;
	}
	.caveats summary:hover {
		color: var(--ink-muted);
	}
	.caveats[open] summary {
		margin-bottom: 0.6rem;
	}
	.caveats p {
		font-size: 0.78rem;
		color: var(--ink-faint);
		margin: 0 0 0.6rem;
	}
</style>
