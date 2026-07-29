<!--
  BeliefHeuristicResponse — what the paper's belief heuristic PROMISES vs what it
  DELIVERS, per source, as evidence accumulates.

  The 2023 model is a noisy-OR over per-source priors,
  belief = 1 − Π_s (syst_s + rand_s^{n_s}),
  so belief is a pure function of the evidence-count profile.

  LEAD (best powered): the single-evidence rung. Every statement there is assigned
  the IDENTICAL belief, because the default priors give reach/sparser/trips/medscan/
  rlimsp the same (rand, syst) pair — yet measured accuracy across those sources
  runs about five-fold. One shared prior, five very different readers.

  SUPPORT: each ladder is one source. The SOLID line is the belief assigned at each
  evidence count (identical for every statement on that rung, by construction); the
  DASHED line is the rate those same statements were actually correct in the paper's
  own gold. Belief climbs and measured accuracy does not follow — but that is a
  FLAT reading, not an inverted one: the rank correlation (rho, over every statement
  including undrawn rungs) is near zero. Rungs below the disclosed minimum are not
  drawn, so no trend may be read off the drawn endpoints.
-->
<script lang="ts">
	import type { BeliefHeuristicLoad, HeuristicSource } from '$lib/server/belief-heuristic';

	let { data }: { data: BeliefHeuristicLoad } = $props();

	// Plot box in a 0..100 viewBox; x = evidence count, y = probability.
	const PAD = 11;
	const px = (n: number, maxN: number) => PAD + ((n - 1) / Math.max(1, maxN - 1)) * (100 - 2 * PAD);
	const py = (p: number) => 100 - PAD - p * (100 - 2 * PAD);
	const rOf = (count: number, maxCount: number) => 1.5 + 3.2 * Math.sqrt(count / Math.max(1, maxCount));

	function path(source: HeuristicSource, maxN: number, pick: (r: HeuristicSource['rungs'][0]) => number) {
		return source.rungs
			.map((r, i) => `${i === 0 ? 'M' : 'L'}${px(r.n, maxN).toFixed(2)},${py(pick(r)).toFixed(2)}`)
			.join(' ');
	}

	const pct = (v: number) => `${(v * 100).toFixed(0)}%`;
	/** One decimal, so a 64.8% accuracy is never mistaken for the 65% belief. */
	const pct1 = (v: number) => `${(v * 100).toFixed(1)}%`;
	const rho = (v: number) => `${v >= 0 ? '+' : '−'}${Math.abs(v).toFixed(2)}`;

	/** Bars are scaled to the best measured rate, not to 100%. */
	const maxRate = (rows: { correctRate: number }[]) =>
		Math.max(...rows.map((r) => r.correctRate)) || 1;

	/** Accuracy range across a source's DRAWN rungs — deliberately not endpoints. */
	function rateRange(source: HeuristicSource): string {
		const rates = source.rungs.map((r) => r.correctRate);
		const lo = Math.min(...rates);
		const hi = Math.max(...rates);
		return lo === hi ? pct(lo) : `${pct(lo)}–${pct(hi)}`;
	}
</script>

{#if data.status !== 'ok'}
	<section class="bh"><p class="note">belief charts unavailable — {data.reason}</p></section>
{:else}
	<section
		class="bh"
		aria-label="the belief formula published in 2023: what it promises against what it delivers"
	>
		<div
			class="single"
			aria-label="statements resting on a single piece of evidence: one belief value, several different accuracies"
		>
			<p class="single-head">
				One sentence, one source, <strong>{pct(data.singleEvidence.belief)}</strong> — the same
				number for every one of these {data.singleEvidence.total} statements, because out of the box
				these readers are all assigned the identical pair of reliability settings. What they actually
				delivered:
			</p>
			<ul class="bars">
				{#each data.singleEvidence.rows as row (row.source)}
					<li>
						<span class="bl">{row.source}</span>
						<span class="bt">
							<span
								class="bf"
								style="width:{(row.correctRate / maxRate(data.singleEvidence.rows)) * 100}%"
							></span>
						</span>
						<span class="bv"
							><strong>{pct1(row.correctRate)}</strong>
							<span class="frac">{row.correct}/{row.count}</span></span
						>
					</li>
				{/each}
			</ul>
			<!-- Which sources make the list, and whether the spread beside it is more
			     than noise. Inside <details> because the bars carry the finding and
			     this is its warrant — not because it is optional. -->
			<details class="disclose">
				<summary>which sources are drawn, and is the spread real</summary>
				<p>
					Every source with at least {data.filters.minSingleEvidence} statements resting on one piece
					of evidence, including sources with too few multi-evidence statements to draw a chart of
					their own below.
				</p>
				{#if data.singleEvidence.homogeneity}
					<p>
						Those {data.singleEvidence.rows.length} accuracies are not one accuracy measured
						{data.singleEvidence.rows.length} times. Testing them against the hypothesis that all the
						sources share a single correct rate (Pearson chi-square) gives
						{data.singleEvidence.homogeneity.chi2.toFixed(1)} on
						{data.singleEvidence.homogeneity.df} degrees of freedom, p = {data.singleEvidence.homogeneity.p.toExponential(
							1
						)}, over all {data.singleEvidence.total} statements — far too large to be chance. Every one
						of those statements was assigned {pct(data.singleEvidence.belief)}.
					</p>
				{/if}
			</details>
		</div>

		<div class="grid">
			{#each data.sources as source (source.source)}
				{@const maxN = Math.max(...source.rungs.map((r) => r.n))}
				{@const maxCount = Math.max(...source.rungs.map((r) => r.count))}
				<figure>
					<figcaption>
						<span class="src">{source.source}</span>
						<span class="n">{source.total} single-source statements</span>
					</figcaption>
					<svg viewBox="0 0 100 100" role="img" aria-labelledby="bh-t-{source.source} bh-d-{source.source}">
						<title id="bh-t-{source.source}">{source.source} — belief assigned vs correct rate</title>
						<desc id="bh-d-{source.source}"
							>As the amount of evidence rises from {source.rungs[0].n} piece to {maxN}, the belief
							the formula assigns goes from {pct(source.rungs[0].belief)} to {pct(
								source.rungs[source.rungs.length - 1].belief
							)}, while the share of those statements that are actually correct stays within
							{rateRange(source)} across the steps drawn here. Ranking the statements by amount of
							evidence and by correctness gives a rank correlation of {rho(source.rho)} over all
							{source.total} statements.</desc
						>

						<line class="axis" x1={PAD} y1={py(0)} x2={100 - PAD} y2={py(0)} />
						<line class="axis" x1={PAD} y1={py(0)} x2={PAD} y2={py(1)} />
						<line class="half" x1={PAD} y1={py(0.5)} x2={100 - PAD} y2={py(0.5)} />

						<!-- the gap the heuristic cannot support -->
						<path
							class="gap"
							d="{path(source, maxN, (r) => r.belief)} L{px(
								source.rungs[source.rungs.length - 1].n,
								maxN
							).toFixed(2)},{py(source.rungs[source.rungs.length - 1].correctRate).toFixed(2)} {source.rungs
								.slice()
								.reverse()
								.map((r) => `L${px(r.n, maxN).toFixed(2)},${py(r.correctRate).toFixed(2)}`)
								.join(' ')} Z"
						/>

						<path class="promised" d={path(source, maxN, (r) => r.belief)} />
						<path class="delivered" d={path(source, maxN, (r) => r.correctRate)} />

						{#each source.rungs as r (r.n)}
							<circle class="m-promised" cx={px(r.n, maxN)} cy={py(r.belief)} r={rOf(r.count, maxCount)}>
								<title
									>{r.n} pieces of evidence · the formula says {pct(r.belief)} · {r.count} statements</title
								>
							</circle>
							<circle class="m-delivered" cx={px(r.n, maxN)} cy={py(r.correctRate)} r={rOf(r.count, maxCount)}>
								<title
									>{r.n} pieces of evidence · actually correct {pct(r.correctRate)} · {r.count} statements</title
								>
							</circle>
						{/each}
					</svg>
					<p class="rungnote">
						{source.rungs[0].n}→{maxN} pieces of evidence · claims
						<strong>{pct(source.rungs[0].belief)}→{pct(source.rungs[source.rungs.length - 1].belief)}</strong>
						· delivers <strong class="delivered-t">{rateRange(source)}</strong>, with no trend (rank
						correlation {rho(source.rho)} over all {source.total} statements)
					</p>
				</figure>
			{/each}
		</div>

		<p class="legend">
			<span class="key promised-k">— belief the formula assigns</span>
			<span class="key delivered-k">--- share of those statements that were actually correct</span>
			<span class="key">· mark size shows how many statements</span>
			<span class="key"
				>· steps holding fewer than {data.filters.minRung} statements are not drawn (the rank
				correlation still counts them)</span
			>
		</p>

		<p class="reading">
			The belief published in 2023 is a <em>noisy-OR over per-source reliabilities</em>: it is computed
			entirely from how many pieces of evidence arrived from which source, and the sentences
			themselves are never read. These charts say it again — belief climbs as evidence accumulates
			while the share actually correct stays flat. Its commonest answer,
			<strong>{pct(data.modal.belief)}</strong>
			({data.modal.count} statements), is right <strong>{pct(data.modal.correctRate)}</strong> of the
			time.
		</p>
		<!-- The same fact told through the single-evidence rung and the value grid;
		     collapsed for the page budget, not dropped. -->
		<details class="reading-more">
			<summary>the same fact, two more ways</summary>
			<p>
				Where a statement rests on a single piece of evidence, every source is assigned
				<strong>{pct(data.singleEvidence.belief)}</strong>, and measured accuracy runs from
				<strong>{pct1(data.singleEvidence.rows[data.singleEvidence.rows.length - 1].correctRate)}</strong
				>
				({data.singleEvidence.rows[data.singleEvidence.rows.length - 1].source}) to
				<strong>{pct1(data.singleEvidence.rows[0].correctRate)}</strong>
				({data.singleEvidence.rows[0].source}) — one shared setting, {data.singleEvidence.rows.length}
				very different readers. The objection is not that the formula is coarse — it emits
				<strong>{data.distinctBeliefs} distinct values</strong> across {data.nStatements}
				statements, finer than any of the reading models above. It is that the fineness is spent
				where nothing is decided: <strong>{data.saturation.distinctAbove}</strong> of those
				values sit above {data.saturation.cut}, covering {data.saturation.nAbove} statements
				that are already effectively certain. Below {data.saturation.cut} — where the review
				decisions actually happen — the formula has only
				<strong>{data.saturation.distinctBelow} distinct answers</strong> for
				{data.saturation.nBelow} statements, and its single most-used one,
				<strong>{pct(data.modal.belief)}</strong> ({data.modal.count} statements,
				{pct1(data.modal.count / data.nStatements)} of all of them), is right
				<strong>{pct1(data.modal.correctRate)}</strong> of the time.
			</p>
		</details>
	</section>
{/if}

<style>
	.bh {
		margin: 0 0 2rem;
	}
	.grid {
		display: grid;
		grid-template-columns: repeat(auto-fill, minmax(min(100%, 250px), 1fr));
		gap: 1.3rem;
	}

	.single {
		margin: 0 0 1.5rem;
		padding: 0 0 1.1rem;
		border-bottom: 1px solid var(--rule);
	}
	.single-head {
		margin: 0 0 0.7rem;
		max-width: 66ch;
		font-family: var(--serif);
		font-size: 0.88rem;
		line-height: 1.55;
		color: var(--ink-muted);
	}
	.single-head strong {
		color: var(--blocked);
		font-weight: 600;
	}
	.bars {
		list-style: none;
		margin: 0;
		padding: 0;
		display: flex;
		flex-direction: column;
		gap: 0.3rem;
		max-width: 34rem;
	}
	.bars li {
		display: grid;
		grid-template-columns: 4.6rem minmax(0, 1fr) 6.4rem;
		align-items: center;
		gap: 0.5rem;
		font-family: var(--mono);
		font-size: 0.7rem;
	}
	.bl {
		color: var(--ink);
	}
	.bt {
		display: block;
		height: 0.62rem;
		background: var(--rule);
	}
	.bf {
		display: block;
		height: 100%;
		background: var(--accent);
		opacity: 0.75;
	}
	.bv {
		color: var(--ink-muted);
		text-align: right;
		white-space: nowrap;
	}
	.bv strong {
		color: var(--ink);
		font-weight: 500;
	}
	.frac {
		color: var(--ink-faint);
	}
	.disclose {
		margin: 0.55rem 0 0;
		max-width: 60ch;
	}
	.disclose summary,
	.reading-more summary {
		font-family: var(--mono);
		font-size: 0.66rem;
		color: var(--ink-faint);
		text-transform: uppercase;
		letter-spacing: 0.05em;
		cursor: pointer;
	}
	.disclose summary:hover,
	.reading-more summary:hover {
		color: var(--ink-muted);
	}
	.disclose[open] summary,
	.reading-more[open] summary {
		margin-bottom: 0.4rem;
	}
	.disclose p {
		margin: 0;
		font-family: var(--mono);
		font-size: 0.66rem;
		color: var(--ink-faint);
		line-height: 1.4;
	}
	figure {
		margin: 0;
		min-width: 0;
	}
	figcaption {
		display: flex;
		flex-direction: column;
		gap: 0.05rem;
		font-family: var(--mono);
		margin-bottom: 0.25rem;
	}
	.src {
		color: var(--ink);
		font-size: 0.8rem;
	}
	.n {
		color: var(--ink-faint);
		font-size: 0.66rem;
	}
	svg {
		display: block;
		width: 100%;
		height: auto;
		max-width: 240px;
		aspect-ratio: 1 / 1;
		overflow: visible;
	}

	.axis {
		stroke: var(--rule);
		stroke-width: 0.5;
	}
	.half {
		stroke: var(--ink-faint);
		stroke-width: 0.35;
		stroke-dasharray: 1.5 2;
		opacity: 0.5;
	}
	.gap {
		fill: var(--blocked);
		fill-opacity: 0.08;
		stroke: none;
	}
	.promised {
		fill: none;
		stroke: var(--blocked);
		stroke-width: 1.3;
	}
	.delivered {
		fill: none;
		stroke: var(--accent);
		stroke-width: 1.3;
		stroke-dasharray: 3 2;
	}
	.m-promised {
		fill: var(--blocked);
		fill-opacity: 0.75;
		stroke: var(--paper);
		stroke-width: 0.5;
	}
	.m-delivered {
		fill: var(--accent);
		fill-opacity: 0.75;
		stroke: var(--paper);
		stroke-width: 0.5;
	}

	.rungnote {
		margin: 0.3rem 0 0;
		font-family: var(--mono);
		font-size: 0.66rem;
		color: var(--ink-muted);
		line-height: 1.4;
	}
	.rungnote strong {
		color: var(--blocked);
		font-weight: 500;
	}
	.rungnote .delivered-t {
		color: var(--accent);
	}

	.legend {
		margin: 0.9rem 0 0;
		font-family: var(--mono);
		font-size: 0.68rem;
		color: var(--ink-faint);
		display: flex;
		gap: 1.1rem;
		flex-wrap: wrap;
	}
	.promised-k {
		color: var(--blocked);
	}
	.delivered-k {
		color: var(--accent);
	}

	.reading {
		margin: 0.9rem 0 0;
		max-width: 66ch;
		font-family: var(--serif);
		font-size: 0.88rem;
		line-height: 1.55;
		color: var(--ink-muted);
	}
	.reading em {
		color: var(--ink);
		font-style: italic;
	}
	.reading strong {
		color: var(--ink);
		font-weight: 600;
	}
	.reading-more {
		margin: 0.4rem 0 0;
		max-width: 66ch;
	}
	.reading-more p {
		margin: 0;
		font-family: var(--serif);
		font-size: 0.82rem;
		line-height: 1.55;
		color: var(--ink-faint);
	}
	.reading-more strong {
		color: var(--ink-muted);
		font-weight: 600;
	}
</style>
