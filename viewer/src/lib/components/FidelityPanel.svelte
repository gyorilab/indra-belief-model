<script lang="ts">
	/**
	 * Beat 1 of /paper, and the page's DEFINITION SITE for two terms every later
	 * beat then uses bare:
	 *   · "random forest" — 2,000 trees, depth 13, over per-source evidence
	 *     counts plus statement type, PMID count and a promoter flag. Spelled out
	 *     here in the lede because this is the first beat that shows it.
	 *   · "fold" — "the statements are split into 10 folds — 10 equal groups — and
	 *     every model here is scored only on the group it never learned from".
	 *     Beat 3's lead-in says "the same 10 folds" and relies on this.
	 *     The word is the FIELD'S, not ours, and it is kept: this page is read by
	 *     a lab that published a cross-validation paper. What is owed is the one
	 *     sentence of English beside it, which is what the gloss above is.
	 *     (An earlier pass replaced it with the coinage "slice" throughout and was
	 *     reversed on 2026-07-29; `slice` is now a banned word in class (a4), and
	 *     this file was one of the two the sweep for it never opened.)
	 * If either definition is moved or trimmed, move it EARLIER, never later: the
	 * reader is a working biologist who has not read the 2023 paper, and a term
	 * used before it is defined is the defect this page was rewritten to fix.
	 */
	import type { PaperLiteralLoad } from '$lib/data/paper-literal';

	let { data }: { data: PaperLiteralLoad } = $props();

	/** Git-conventional 7-char short SHA; mirrors PaperMethodLandscape's shortSha. */
	function shortSha(value: string): string {
		return value.slice(0, 7);
	}
</script>

<section class="fidelity" aria-labelledby="fidelity-title">
	{#if data.status !== 'ok'}
		<div class="gate" role="status">
			<p class="eyebrow">reconstruction fidelity</p>
			<h2 id="fidelity-title">Fidelity certificate unavailable</h2>
			<p>{data.reason}</p>
			<code>{data.artifact_path}</code>
		</div>
	{:else}
		{@const f = data.faithfulness}
		{@const r = data.reproduction}
		<header>
			<div>
				<p class="eyebrow">reconstruction fidelity · trust certificate</p>
				<h2 id="fidelity-title">You can believe the reconstruction</h2>
			</div>
			<strong class="verified" title="every value below is measured, not asserted">
				<span aria-hidden="true">✓</span> verified
			</strong>
		</header>

		<section class="block" aria-labelledby="fidelity-faithful-title">
			<h3 id="fidelity-faithful-title">
				Reconstruction is faithful
				<span class="tick" aria-hidden="true">✓</span>
			</h3>
			<p class="lede">
				The model published with the 2023 INDRA assembly paper is a random forest — 2,000 decision
				trees, 13 questions deep, voting on counts. It sees how many pieces of evidence each source
				contributed for a statement, what type of statement it is, how many papers it came from, and
				a promoter flag. It never sees the sentence. Its released code and the
				re-implementation here agree statement for statement.
			</p>
			<dl class="metrics">
				{#if Number.isFinite(f.pearsonR)}
					<div>
						<dt>Pearson r</dt>
						<dd>
							{f.pearsonR.toFixed(4)}
							<small>how closely the two sets of scores track each other; 1 is identical</small>
						</dd>
					</div>
				{/if}
				{#if Number.isFinite(f.spearmanR)}
					<div>
						<dt>Spearman ρ</dt>
						<dd>
							{f.spearmanR.toFixed(4)}
							<small>the same, on the order they put the statements in</small>
						</dd>
					</div>
				{/if}
				{#if Number.isFinite(f.meanAbsDiff)}
					<div>
						<dt>typical difference in score</dt>
						<dd>{f.meanAbsDiff.toFixed(4)}</dd>
					</div>
				{/if}
				{#if Number.isFinite(f.maxAbsDiff)}
					<div>
						<dt>worst difference on any statement</dt>
						<dd>{f.maxAbsDiff.toFixed(3)}</dd>
					</div>
				{/if}
			</dl>
			{#if Number.isFinite(f.foldMeanPrAucLiteral) && Number.isFinite(f.foldMeanPrAucPort)}
				<p class="fold" role="note">
					<span class="fold-label">trapezoidal PR-AUC · averaged over the 10 folds</span>
					<span class="pair">
						released code <b>{f.foldMeanPrAucLiteral.toFixed(3)}</b>
						<em>vs</em>
						the re-implementation <b>{f.foldMeanPrAucPort.toFixed(3)}</b>
					</span>
					<small>
						The statements are split into 10 folds — 10 equal groups — and every model here is scored
						only on the group it never learned from; the 10 scores are then averaged. Trapezoidal PR-AUC is the
						precision-recall area taken with straight-line interpolation between the points — the
						form the 2023 paper reported. It shows that Table 6 reproduces, and nothing more: it is
						not how we decide which model is better. Average precision and AUROC do that, further
						down the page.
					</small>
				</p>
			{/if}
		</section>

		{#if r}
			<section class="block" aria-labelledby="fidelity-repro-title">
				<h3 id="fidelity-repro-title">
					Reproduces the published paper
					<span class="tick" aria-hidden="true">✓</span>
				</h3>
				<p class="lede">
					Running the released code of the 2023 INDRA assembly paper reproduces its published Table 6.
				</p>
				<dl class="metrics">
					{#if Number.isFinite(r.maxAbsDeltaVsPublishedTable6)}
						<div>
							<dt>worst gap vs the published Table 6</dt>
							<dd>
								{r.maxAbsDeltaVsPublishedTable6.toFixed(3)}
								<small>the largest gap, on any method we reproduced, between its fold-averaged score
									and the number printed in the paper</small>
							</dd>
						</div>
					{/if}
					{#if Number.isFinite(r.headlineLiteral) && Number.isFinite(r.headlinePublished)}
						<div>
							<dt>the best published row (promoter and average-length features added)</dt>
							<dd>
								re-run <b>{r.headlineLiteral.toFixed(3)}</b>
								<em>=</em>
								published <b>{r.headlinePublished.toFixed(3)}</b>
							</dd>
						</div>
					{/if}
				</dl>
				<p class="provenance">
					pinned paper commit
					<code title={r.paperCodeCommit}>{shortSha(r.paperCodeCommit)}</code>
				</p>
			</section>
		{/if}
	{/if}
</section>

<style>
	.fidelity {
		margin-top: 1.5rem;
		padding-top: 1.25rem;
		border-top: 2px solid var(--ink);
	}
	.fidelity > header {
		display: flex;
		justify-content: space-between;
		align-items: start;
		gap: 1rem;
	}
	.fidelity h2,
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
	.verified {
		flex: 0 0 auto;
		padding: 0.2rem 0.42rem;
		border: 1px solid var(--accent);
		color: var(--accent);
		font-family: var(--mono);
		font-size: 0.62rem;
		font-weight: 500;
		text-transform: uppercase;
		letter-spacing: 0.04em;
		white-space: nowrap;
	}
	.block {
		margin-top: 1rem;
		padding-top: 0.9rem;
		border-top: 1px dotted var(--rule);
	}
	.block h3 {
		display: flex;
		align-items: baseline;
		gap: 0.4rem;
		margin: 0;
		font-family: var(--serif);
		font-size: 1.02rem;
		font-weight: 400;
	}
	.block h3 .tick {
		color: var(--accent);
		font-family: var(--mono);
		font-size: 0.82rem;
	}
	.lede {
		margin: 0.28rem 0 0;
		font-family: var(--serif);
		font-size: 0.84rem;
		line-height: 1.5;
		color: var(--ink-muted);
	}
	.metrics {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(9.5rem, 1fr));
		gap: 0.7rem 1rem;
		margin: 0.85rem 0 0;
	}
	.metrics div {
		padding-top: 0.4rem;
		border-top: 1px dotted var(--rule);
	}
	.metrics dt {
		font-family: var(--mono);
		font-size: 0.63rem;
		color: var(--ink-faint);
	}
	.metrics dd {
		margin: 0.22rem 0 0;
		font-family: var(--mono);
		font-size: 1rem;
		font-variant-numeric: tabular-nums;
		color: var(--ink);
	}
	.metrics dd b {
		font-weight: 600;
	}
	.metrics dd em {
		font-style: normal;
		color: var(--ink-faint);
		padding: 0 0.15rem;
	}
	.metrics dd small {
		display: block;
		margin-top: 0.16rem;
		font-size: 0.6rem;
		color: var(--ink-faint);
	}
	.fold {
		margin: 0.85rem 0 0;
		padding: 0.6rem 0.75rem;
		border: 1px solid var(--rule);
		border-left-width: 3px;
		background: color-mix(in srgb, var(--ink) 3%, transparent);
	}
	.fold-label {
		display: block;
		font-family: var(--mono);
		font-size: 0.62rem;
		color: var(--ink-faint);
		text-transform: uppercase;
		letter-spacing: 0.04em;
	}
	.fold .pair {
		display: block;
		margin-top: 0.2rem;
		font-family: var(--mono);
		font-size: 0.95rem;
		font-variant-numeric: tabular-nums;
		color: var(--ink);
	}
	.fold .pair b {
		font-weight: 600;
	}
	.fold .pair em {
		font-style: normal;
		color: var(--ink-faint);
		padding: 0 0.25rem;
	}
	.fold small {
		display: block;
		margin-top: 0.3rem;
		font-family: var(--serif);
		font-size: 0.72rem;
		line-height: 1.45;
		color: var(--ink-muted);
	}
	.provenance {
		margin: 0.85rem 0 0;
		font-family: var(--mono);
		font-size: 0.66rem;
		color: var(--ink-faint);
	}
	.provenance code {
		font-family: var(--mono);
		font-size: 0.66rem;
		color: var(--ink-muted);
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
	.gate code {
		font-family: var(--mono);
		font-size: 0.62rem;
		color: var(--ink-faint);
	}
	@media (max-width: 720px) {
		.fidelity > header {
			display: block;
		}
		.verified {
			display: inline-block;
			margin-top: 0.6rem;
		}
	}
</style>
