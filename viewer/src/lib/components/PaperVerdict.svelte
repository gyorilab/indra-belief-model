<script lang="ts">
	/**
	 * THE RANKED VERDICT — the first thing on /paper, above every figure.
	 *
	 * It prints; it does not decide. The strength of each claim, the sentence that
	 * states it, the sentence that doubts it and the order the three appear in are
	 * all settled in `$lib/data/paper-verdict.ts` from the shipped artifacts. This
	 * file chooses no wording by condition and formats no number, which is the
	 * point: five separate sign-blindness regressions on this page were all a
	 * template deciding a direction.
	 *
	 * NO BADGES. The reader who prompted this block called the existing figure
	 * badges "scorecards in an argument between authors and referees". A tier
	 * states its claim, shows the numbers, and names the one strongest reason to
	 * doubt it. The strength is a word, set in the same ink as the rest.
	 *
	 * A GATED TIER PRINTS `plainReason`, NEVER `reason`. A loader's `reason` is
	 * written for whoever is holding a broken artifact and names that artifact's
	 * own fields; this block used to print it verbatim, beside two tiers that were
	 * working, which put artifact dialect on an otherwise working page. `reason`
	 * still travels in the data for the page's verification boundary. It is not
	 * read here, and that is deliberate rather than an oversight.
	 */
	import type { PaperVerdictLoad } from '$lib/data/paper-verdict';

	let { data }: { data: PaperVerdictLoad } = $props();
</script>

<section class="verdict" aria-labelledby="verdict-title">
	<header>
		<p class="eyebrow">what we found</p>
		<h2 id="verdict-title">Strongest claim first</h2>
		<p class="intro">
			Three questions, ranked by how well this benchmark answers them. Everything below this block
			supports one of these three, or qualifies it.
		</p>
	</header>

	{#if data.status !== 'ok'}
		<p class="dark" role="status">The verdict is unavailable — {data.reason}</p>
	{/if}

	<ol class="tiers">
		{#each data.tiers as tier (tier.id)}
			<li>
				<p class="question">{tier.question}</p>
				{#if tier.status === 'ok'}
					<p class="strength">{tier.strengthWord}</p>
					<p class="claim">{tier.claim}</p>
					<dl class="numbers">
						{#each tier.numbers as number (number.caption)}
							<div>
								<dt>{number.caption}</dt>
								<dd>
									{number.value}
									{#if number.note}<small>{number.note}</small>{/if}
								</dd>
							</div>
						{/each}
					</dl>
					<p class="doubt">
						<span class="doubt-label">the one reason to doubt it</span>
						{tier.doubt}
					</p>
				{:else}
					<p class="strength dark">unanswered here</p>
					<p class="dark">{tier.plainReason}</p>
				{/if}
			</li>
		{/each}
	</ol>
</section>

<style>
	.verdict {
		margin-top: 1.2rem;
		padding-top: 1.1rem;
		border-top: 2px solid var(--ink);
	}
	.eyebrow {
		margin: 0;
		font-family: var(--mono);
		font-size: 0.68rem;
		color: var(--ink-faint);
		text-transform: uppercase;
		letter-spacing: 0.05em;
	}
	.verdict h2 {
		margin: 0.18rem 0 0;
		font-family: var(--serif);
		font-weight: 400;
		font-size: 1.35rem;
	}
	.intro {
		margin: 0.35rem 0 0;
		max-width: 62ch;
		font-family: var(--serif);
		font-size: 0.86rem;
		line-height: 1.5;
		color: var(--ink-muted);
	}
	.tiers {
		list-style: none;
		counter-reset: tier;
		margin: 0.9rem 0 0;
		padding: 0;
	}
	.tiers > li {
		counter-increment: tier;
		position: relative;
		margin-top: 1rem;
		padding: 0.85rem 0 0 2.1rem;
		border-top: 1px dotted var(--rule);
	}
	.tiers > li::before {
		content: counter(tier);
		position: absolute;
		left: 0;
		top: 0.8rem;
		font-family: var(--mono);
		font-size: 1.25rem;
		font-variant-numeric: tabular-nums;
		color: var(--ink-faint);
	}
	.question {
		margin: 0;
		font-family: var(--mono);
		font-size: 0.66rem;
		text-transform: uppercase;
		letter-spacing: 0.05em;
		color: var(--ink-faint);
	}
	.strength {
		margin: 0.28rem 0 0;
		font-family: var(--serif);
		font-size: 1.08rem;
		color: var(--ink);
	}
	.claim {
		margin: 0.2rem 0 0;
		max-width: 66ch;
		font-family: var(--serif);
		font-size: 0.94rem;
		line-height: 1.5;
		color: var(--ink);
	}
	.numbers {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(13rem, 1fr));
		gap: 0.7rem 1.4rem;
		margin: 0.8rem 0 0;
	}
	.numbers div {
		padding-top: 0.4rem;
		border-top: 1px dotted var(--rule);
	}
	.numbers dt {
		font-family: var(--mono);
		font-size: 0.63rem;
		line-height: 1.35;
		color: var(--ink-faint);
	}
	.numbers dd {
		margin: 0.24rem 0 0;
		font-family: var(--mono);
		font-size: 0.92rem;
		font-variant-numeric: tabular-nums;
		line-height: 1.3;
		color: var(--ink);
	}
	.numbers dd small {
		display: block;
		margin-top: 0.22rem;
		font-family: var(--serif);
		font-size: 0.74rem;
		line-height: 1.45;
		color: var(--ink-muted);
	}
	.doubt {
		margin: 0.8rem 0 0;
		max-width: 66ch;
		padding-left: 0.7rem;
		border-left: 2px solid var(--rule);
		font-family: var(--serif);
		font-size: 0.84rem;
		line-height: 1.5;
		color: var(--ink-muted);
	}
	.doubt-label {
		display: block;
		font-family: var(--mono);
		font-size: 0.6rem;
		text-transform: uppercase;
		letter-spacing: 0.05em;
		color: var(--ink-faint);
	}
	.dark {
		margin: 0.28rem 0 0;
		font-family: var(--mono);
		font-size: 0.76rem;
		color: var(--ink-faint);
	}
	@media (max-width: 620px) {
		.tiers > li {
			padding-left: 1.6rem;
		}
	}
</style>
