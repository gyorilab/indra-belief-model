<!--
  BeliefVsIndra — INDRA's belief and ours on one calibration picture.

  THE PERCEPTUAL POINT, and the reason for the shaded band: SimpleScorer AT THE
  PRIORS THE LIBRARY SHIPS cannot produce a belief below 0.65. Its worst default
  source is reach at (rand 0.30, syst 0.05), so one piece of evidence gives
  1 − (0.05 + 0.30) = 0.65, and the noisy-OR is monotone increasing in evidence —
  more readings only push upward. The band makes that reachability limit VISIBLE
  rather than something to infer from where the marks happen to sit.

  SCOPE, corrected 2026-08-13 after this shipped wrong: the band is NOT a claim
  that "INDRA cannot assign a lower belief". Stored INDRA beliefs in our own
  data/benchmark/belief_benchmark.jsonl run down to 0.3195, with 941 of 9,342
  statements (10.1%) below 0.65 — those come from source priors weaker than any
  entry in the shipped table. The floor belongs to THIS scorer at THESE priors.
  The page must say that, because the paper corpus alone cannot distinguish a
  model limit from a corpus that was selected at 0.65.

  Everything is read from the server payload. No rate, count or belief value is
  written in this file; only SVG layout constants are.

  Deliberate constraints (do not "improve" these):
    · Both series share ONE unit square with the y = x diagonal. Never a
      truncated axis — the empty lower-left IS the finding.
    · Marks are area-scaled by statement count, so a thin bin cannot draw a
      confident-looking curve.
    · The gap stem drops from each mark to the diagonal. Below the line is
      over-confidence; the eye should read stem LENGTH as "how wrong the odds
      were" without consulting a number.
    · Vocabulary is INDRA's: statement, evidence, source, belief, curation,
      random vs systematic error. No coinages.
-->
<script lang="ts">
	import type { BeliefVsIndra } from '$lib/server/belief-vs-indra';

	let { data }: { data: BeliefVsIndra } = $props();

	// Unit square in user units; the viewBox does the rest.
	const S = 100;
	const PAD = 12;
	const px = (p: number) => PAD + p * S;
	const py = (p: number) => PAD + (1 - p) * S;
	/** Area-proportional radius, so a 20-statement bin cannot look like a 900-statement one. */
	const rOf = (n: number, max: number) => 1.6 + 4.4 * Math.sqrt(n / Math.max(max, 1));

	const maxN = $derived(
		Math.max(
			...data.indra.bins.map((b) => b.n),
			...data.reader.bins.map((b) => b.n),
			1
		)
	);
	const pathOf = (bins: { p_mean: number; y_rate: number }[]) =>
		bins.map((b, i) => `${i ? 'L' : 'M'}${px(b.p_mean).toFixed(2)},${py(b.y_rate).toFixed(2)}`).join(' ');

	const probeMaxN = $derived(
		data.probe
			? Math.max(
					...data.probe.incumbent.bins.map((b) => b.n),
					...data.probe.candidate.bins.map((b) => b.n),
					1
				)
			: 1
	);

	/** The unreachable region: everything below INDRA's observed floor. */
	const floor = $derived(data.indra.min);
	const pct = (v: number) => `${(v * 100).toFixed(0)}%`;
	const pct1 = (v: number) => `${(v * 100).toFixed(1)}%`;
</script>

<section class="wrap">
	<header>
		<h2>What the belief number can and cannot say</h2>
		<p class="dek">
			Both models score the same {data.n.toLocaleString()} assembled statements against the same
			curations. {pct(data.base_rate)} of those statements are correct.
		</p>
	</header>

	<div class="grid">
		<figure class="plot">
			<svg viewBox="0 0 {S + PAD * 2} {S + PAD * 2}" role="img"
				aria-label="Calibration: assigned belief against observed correct rate, for INDRA's SimpleScorer and for reading the evidence.">
				<!-- the region INDRA's noisy-OR cannot reach -->
				<rect x={px(0)} y={py(1)} width={px(floor) - px(0)} height={S} class="unreachable" />
				<text x={px(0) + 2} y={py(1) + 6} class="band-label">
					below SimpleScorer's floor at the shipped priors
				</text>

				<!-- perfect calibration -->
				<line x1={px(0)} y1={py(0)} x2={px(1)} y2={py(1)} class="diagonal" />
				<text x={px(1) - 1} y={py(1) + 7} class="diag-label" text-anchor="end">
					assigned = observed
				</text>

				<!-- gap stems: distance from the diagonal is the miscalibration -->
				{#each [...data.indra.bins.map((b) => ({ ...b, k: 'a' })), ...data.reader.bins.map((b) => ({ ...b, k: 'b' }))] as b}
					<line class="stem {b.k}" x1={px(b.p_mean)} y1={py(b.y_rate)} x2={px(b.p_mean)} y2={py(b.p_mean)} />
				{/each}

				<path d={pathOf(data.indra.bins)} class="line a" />
				<path d={pathOf(data.reader.bins)} class="line b" />

				{#each data.indra.bins as b}
					<circle cx={px(b.p_mean)} cy={py(b.y_rate)} r={rOf(b.n, maxN)} class="mark a" />
				{/each}
				{#each data.reader.bins as b}
					<circle cx={px(b.p_mean)} cy={py(b.y_rate)} r={rOf(b.n, maxN)} class="mark b" />
				{/each}

				<line x1={px(0)} y1={py(0)} x2={px(1)} y2={py(0)} class="axis" />
				<line x1={px(0)} y1={py(0)} x2={px(0)} y2={py(1)} class="axis" />
			</svg>
			<figcaption>
				<span class="ax-x">belief assigned to the statement →</span>
				<span class="ax-y">↑ share actually correct</span>
			</figcaption>
		</figure>

		<div class="read">
			<p class="lede">
				A mark sitting <strong>below</strong> the diagonal means the belief was too high: the
				statements in that bin were correct less often than the number promised. The stem is how
				far off the odds were.
			</p>

			<div class="legend">
				<div class="ser a">
					<span class="swatch"></span>
					<div>
						<p class="name">{data.indra.label}</p>
						<p class="meta">
							belief spans {data.indra.min.toFixed(2)}–{data.indra.max.toFixed(2)} ·
							{data.indra.bins.length} of 10 bins occupied ·
							<strong>{data.indra.below_half.toLocaleString()}</strong> statements below 0.5
						</p>
					</div>
				</div>
				<div class="ser b">
					<span class="swatch"></span>
					<div>
						<p class="name">{data.reader.label}</p>
						<p class="meta">
							belief spans {data.reader.min.toFixed(2)}–{data.reader.max.toFixed(2)} ·
							{data.reader.bins.length} of 10 bins occupied ·
							<strong>{data.reader.below_half.toLocaleString()}</strong> statements below 0.5
						</p>
					</div>
				</div>
			</div>

			<p>
				INDRA's belief comes from the noisy-OR over per-source priors,
				<code>1 − Π<sub>s</sub> (syst<sub>s</sub> + rand<sub>s</sub><sup>n<sub>s</sub></sup>)</code>.
				Every input to it is a <em>count</em>: which sources reported the statement, and how many
				times each. The sentence is never consulted. At the priors the library ships, the worst
				source is <code>reach</code> at <code>(0.30, 0.05)</code>, so a single piece of evidence
				gives 0.65 and more evidence only raises it — on this corpus there is nothing below that
				to say <em>this reading looks wrong</em>.
			</p>
			<p class="scope">
				That floor is this scorer's, at these priors — not INDRA's in general. Stored INDRA
				beliefs elsewhere in our benchmark data reach 0.3195, with about a tenth of statements
				below 0.65, from sources weaker than any in the shipped table. This corpus on its own
				cannot separate a reachability limit from a corpus selected at 0.65; the arithmetic can,
				and it is the arithmetic being shown.
			</p>

			<dl class="ece">
				<div><dt>calibration error, INDRA</dt><dd>{data.indra.ece.toFixed(3)}</dd></div>
				<div><dt>calibration error, reading</dt><dd>{data.reader.ece.toFixed(3)}</dd></div>
			</dl>
			<p class="foot">
				Mean gap between the belief assigned and the share actually correct, weighted by how many
				statements sit in each bin. Lower is closer to honest odds.
			</p>
		</div>
	</div>

	<div class="panel mech">
		<h3>Both numbers come out of the same formula</h3>
		<p>
			This is the part that is easy to miss. The reading series is <em>not</em> a different belief
			model — it is INDRA's noisy-OR, at INDRA's shipped priors, run by
			{#if data.mechanism.implementation}<code>{data.mechanism.implementation}</code>{:else}the same
				aggregation{/if} under
			<code>{data.mechanism.aggregation}</code>. The arithmetic is identical. Only the input differs.
		</p>
		<div class="tworow">
			<div class="side">
				<p class="side-h">INDRA today</p>
				<p class="side-b">
					Every piece of evidence enters the product. A source that reported the statement
					contributes <code>syst<sub>s</sub> + rand<sub>s</sub><sup>n<sub>s</sub></sup></code>,
					whatever its sentences say.
				</p>
			</div>
			<div class="side accent">
				<p class="side-h">Reading the evidence</p>
				<p class="side-b">
					A reader judges each sentence first. Evidence it rejects is dropped, and a source whose
					evidence is <em>all</em> rejected is removed from the product entirely — as if it never
					reported the statement.
				</p>
			</div>
		</div>
		<p>
			That single change is what lets the second curve reach the bottom of the scale. Remove every
			source and the product is empty, so belief is
			<code>1 − 1 = 0</code>. Exactly
			<strong>{data.mechanism.all_evidence_rejected.toLocaleString()}</strong> statements land there —
			the same statements as the disagreement band below. Nothing else in the formula moved.
		</p>
		<p class="foot">
			A separately fitted calibration for the reader also exists in the codebase, and it is
			<em>not</em> what produced these numbers; this page reports the aggregation the frozen
			artifact declares.
		</p>
	</div>

	{#if data.cohort}
		<div class="panel">
			<h3>One belief, five readers</h3>
			<p>
				{data.cohort.n.toLocaleString()} statements — {pct(data.cohort.share)} of the corpus —
				receive <strong>exactly {data.cohort.belief.toFixed(2)}</strong>{#if data.cohort.evidence_count === 1}, every one of
					them backed by a single piece of evidence{/if}. The shipped priors give those readers the
				same random and systematic error rates, so the noisy-OR returns the same number for all of
				them. Their curated outcomes do not agree:
			</p>
			<ul class="bars">
				{#each data.cohort.by_source as s}
					<li>
						<span class="src">{s.source}</span>
						<span class="bar"><span class="fill" style="width:{s.rate * 100}%"></span></span>
						<span class="val">{pct1(s.rate)}</span>
						<span class="n">n={s.n}</span>
					</li>
				{/each}
			</ul>
			<p class="foot">
				All of these carry the identical belief of {data.cohort.belief.toFixed(2)}; together they are
				correct {pct1(data.cohort.observed_rate)} of the time. Sources with fewer than 15 statements
				in the group are not drawn.
			</p>
		</div>
	{/if}

	{#if data.disagreement}
		<div class="panel">
			<h3>Where the two models disagree hardest</h3>
			<p>
				Reading the evidence puts <strong>{data.disagreement.n.toLocaleString()}</strong> statements
				below {data.disagreement.threshold.toFixed(2)} — it judges the sentence not to support the
				claim. Those statements are correct {pct1(data.disagreement.observed_rate)} of the time, so
				the low belief is largely earned.
			</p>
			<p>
				INDRA scores the same statements at a median of
				<strong>{data.disagreement.indra_median.toFixed(2)}</strong>, never below
				{data.disagreement.indra_min.toFixed(2)}, and puts
				<strong>{data.disagreement.indra_at_least_90.toLocaleString()}</strong> of them at 0.90 or
				higher. The counting model has no way to register the objection, because nothing about the
				count changed — only what the sentence says, which it never reads.
			</p>
		</div>
	{/if}

	{#if data.probe}
		<div class="panel">
			<div class="probe-head">
				<h3>Asking the model for a probability instead of a word</h3>
				<span class="tag">measured · not deployed</span>
			</div>
			<p>
				Everything above turns on a single word from the reader: <em>correct</em> or
				<em>incorrect</em>. Three words in, three numbers out — the deployed scorer emits
				<strong>{data.probe.incumbent.distinct_values}</strong> distinct scores across
				{data.probe.n.toLocaleString()} statements, so it can rank almost nothing within a verdict.
			</p>
			<p>
				A second call, with reasoning switched off, instead reads the model's probability at the
				one token where the verdict is about to appear. That is a number, not a word, and it gives
				<strong>{data.probe.candidate.distinct_values}</strong> distinct scores over the same
				statements — at
				{#if data.probe.candidate_seconds && data.probe.incumbent_seconds}
					{data.probe.candidate_seconds.toFixed(1)}&thinsp;s per statement against
					{data.probe.incumbent_seconds.toFixed(1)}&thinsp;s for the reading it supplements{/if}.
			</p>

			<div class="probe-grid">
				<figure class="plot small">
					<svg viewBox="0 0 {S + PAD * 2} {S + PAD * 2}" role="img"
						aria-label="Calibration of the deployed verdict grid against the same grid plus the token-probability probe.">
						<line x1={px(0)} y1={py(0)} x2={px(1)} y2={py(1)} class="diagonal" />
						{#each data.probe.incumbent.bins as b}
							<circle cx={px(b.p_mean)} cy={py(b.y_rate)} r={rOf(b.n, probeMaxN)} class="mark a" />
						{/each}
						{#each data.probe.candidate.bins as b}
							<circle cx={px(b.p_mean)} cy={py(b.y_rate)} r={rOf(b.n, probeMaxN)} class="mark b" />
						{/each}
						<path d={pathOf(data.probe.incumbent.bins)} class="line a" />
						<path d={pathOf(data.probe.candidate.bins)} class="line b" />
						<line x1={px(0)} y1={py(0)} x2={px(1)} y2={py(0)} class="axis" />
						<line x1={px(0)} y1={py(0)} x2={px(0)} y2={py(1)} class="axis" />
					</svg>
					<figcaption>
						<span class="ax-x">score →</span><span class="ax-y">↑ share correct</span>
					</figcaption>
				</figure>

				<div class="probe-read">
					<dl class="ece">
						<div>
							<dt>{data.probe.incumbent.label}</dt>
							<dd>{data.probe.incumbent.ece.toFixed(3)}</dd>
						</div>
						<div>
							<dt>{data.probe.candidate.label}</dt>
							<dd>{data.probe.candidate.ece.toFixed(3)}</dd>
						</div>
					</dl>
					{#if data.probe.single_probe_delta !== null && data.probe.single_probe_ci95}
						<p>
							Adding the probe to the reading improves ranking by
							<strong>{data.probe.single_probe_delta.toFixed(3)}</strong> AUROC
							(95% CI {data.probe.single_probe_ci95[0].toFixed(3)}
							to {data.probe.single_probe_ci95[1].toFixed(3)}), for roughly 1.5% more compute.
							The interval excludes zero.
						</p>
					{/if}
					<p class="foot">
						Sixteen such probes were tried. The ablation found <em>one</em> carries essentially the
						whole gain and the other fifteen add an amount whose interval spans zero. Replacing the
						reading with the probes outright was a no-go: ranking did not improve.
					</p>
				</div>
			</div>

			<p class="warn">
				This is a different evaluation — {data.probe.n.toLocaleString()} statements from a separate
				holdout with its own curations and its own bin edges. Its numbers are not comparable to the
				corpus figures above and the two are deliberately not drawn on shared axes.
				<strong>It is not deployed:</strong> {data.probe.deployment_note}.
			</p>
		</div>
	{/if}
</section>

<style>
	.wrap { display: flex; flex-direction: column; gap: 1.6rem; }
	header h2 { font-size: 1.3rem; margin: 0 0 .25rem; font-weight: 600; }
	.dek { color: var(--ink-muted); margin: 0; font-size: .95rem; }
	.grid { display: grid; grid-template-columns: minmax(280px, 1fr) minmax(280px, 1fr); gap: 1.6rem; align-items: start; }
	@media (max-width: 780px) { .grid { grid-template-columns: 1fr; } }

	.plot { margin: 0; }
	.plot svg { width: 100%; height: auto; overflow: visible; }
	.unreachable { fill: var(--accent); opacity: .06; }
	.band-label { font-family: var(--mono, monospace); font-size: 3.1px; fill: var(--accent); opacity: .8; }
	.diagonal { stroke: var(--ink-faint); stroke-width: .4; stroke-dasharray: 2 1.6; }
	.diag-label { font-family: var(--mono, monospace); font-size: 3.1px; fill: var(--ink-faint); }
	.axis { stroke: var(--rule); stroke-width: .5; }
	.line { fill: none; stroke-width: .9; opacity: .75; }
	.line.a, .mark.a, .stem.a { stroke: var(--a-hue, #1d4e6f); }
	.line.b, .mark.b, .stem.b { stroke: var(--accent); }
	.mark { stroke-width: .7; }
	.mark.a { fill: var(--a-hue, #1d4e6f); fill-opacity: .28; }
	.mark.b { fill: var(--accent); fill-opacity: .28; }
	.stem { stroke-width: .5; opacity: .45; }
	figcaption { display: flex; justify-content: space-between; font-family: var(--mono, monospace); font-size: .66rem; color: var(--ink-faint); margin-top: .3rem; }

	.read { display: flex; flex-direction: column; gap: .8rem; font-size: .93rem; }
	.read p { margin: 0; }
	.lede { font-size: 1rem; }
	.legend { display: flex; flex-direction: column; gap: .55rem; border-block: 1px solid var(--rule); padding: .7rem 0; }
	.ser { display: flex; gap: .55rem; align-items: flex-start; }
	.swatch { width: .7rem; height: .7rem; border-radius: 50%; flex: none; margin-top: .22rem; }
	.ser.a .swatch { background: var(--a-hue, #1d4e6f); }
	.ser.b .swatch { background: var(--accent); }
	.name { font-weight: 600; }
	.meta { font-family: var(--mono, monospace); font-size: .68rem; color: var(--ink-muted); }
	code { font-family: var(--mono, monospace); font-size: .82em; background: var(--rule); padding: .12em .35em; border-radius: 2px; }

	.ece { display: flex; gap: 1.6rem; margin: 0; }
	.ece div { display: flex; flex-direction: column; }
	dt { font-family: var(--mono, monospace); font-size: .64rem; letter-spacing: .07em; text-transform: uppercase; color: var(--ink-muted); }
	dd { margin: 0; font-size: 1.35rem; font-variant-numeric: tabular-nums; }
	.foot { font-size: .76rem; color: var(--ink-faint); }
	.scope { font-size: .82rem; color: var(--ink-muted); border-left: 2px solid var(--blocked, #6f5a16); padding-left: .7rem; }

	.panel { border-top: 1px solid var(--rule); padding-top: 1.1rem; display: flex; flex-direction: column; gap: .6rem; }
	.panel h3 { font-size: 1.02rem; margin: 0; font-weight: 600; }
	.panel p { margin: 0; font-size: .93rem; }
	.bars { list-style: none; margin: .2rem 0 0; padding: 0; display: flex; flex-direction: column; gap: .3rem; }
	.bars li { display: grid; grid-template-columns: 5.5rem 1fr 3rem 3.5rem; gap: .5rem; align-items: center; font-family: var(--mono, monospace); font-size: .72rem; }
	.src { color: var(--ink); }
	.bar { background: var(--rule); height: .62rem; position: relative; }
	.fill { position: absolute; inset: 0 auto 0 0; background: var(--accent); opacity: .55; }
	.val { text-align: right; font-variant-numeric: tabular-nums; }
	.n { color: var(--ink-faint); text-align: right; font-variant-numeric: tabular-nums; }

	.tworow { display: grid; grid-template-columns: 1fr 1fr; gap: .9rem; margin: .3rem 0; }
	@media (max-width: 640px) { .tworow { grid-template-columns: 1fr; } }
	.side { border-left: 2px solid var(--rule-firm, var(--rule)); padding-left: .7rem; }
	.side.accent { border-left-color: var(--accent); }
	.side-h { font-weight: 600; font-size: .84rem; margin: 0 0 .2rem; }
	.side-b { font-size: .86rem; color: var(--ink-muted); margin: 0; }

	.probe-head { display: flex; align-items: baseline; gap: .7rem; flex-wrap: wrap; }
	.tag { font-family: var(--mono, monospace); font-size: .6rem; letter-spacing: .09em; text-transform: uppercase; padding: .16em .5em; background: var(--warn-soft, #f2ebd6); color: var(--blocked, #6f5a16); border-radius: 2px; white-space: nowrap; }
	.probe-grid { display: grid; grid-template-columns: minmax(170px, .8fr) 1.2fr; gap: 1.2rem; align-items: start; margin-top: .4rem; }
	@media (max-width: 640px) { .probe-grid { grid-template-columns: 1fr; } }
	.plot.small { max-width: 15rem; }
	.probe-read { display: flex; flex-direction: column; gap: .55rem; }
	.probe-read p { margin: 0; font-size: .88rem; }
	.warn { font-size: .78rem; color: var(--ink-muted); border-left: 2px solid var(--blocked, #6f5a16); padding-left: .7rem; }
</style>
