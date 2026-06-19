<script lang="ts">
	// Materializes one model's reasoning_trace: a peripheral CoT-access sigil
	// (status + weight of deliberation), the model's committed support/objection
	// (the reliable, always-parseable justification — primary), and the free
	// chain-of-thought gated by status (readable / sealed / withheld / none).
	// Single source of truth used by /compare L3 and /adjudicate.
	import { reasoningBody, reasoningSigil, reasoningStatusTone } from '$lib/format';
	import type { ReasoningStatus, ReasoningTrace } from '$lib/data/types';

	let { trace, reasoning }: { trace: ReasoningTrace | null; reasoning: string | null } = $props();

	// 3 forms for the peripheral channel — present / sealed / gap; the label and
	// colour disambiguate withheld vs none.
	function rtGlyph(status: ReasoningStatus | null): string {
		const t = reasoningStatusTone(status);
		return t === 'open' ? '◉' : t === 'sealed' ? '◼' : '○';
	}
	const tone = $derived(trace ? reasoningStatusTone(trace.status) : 'gap');
	const cj = $derived(trace?.committed_justification ?? null);
</script>

{#if trace}
	<div class="rt-sigil rt-{tone}">
		<span class="rt-glyph" aria-hidden="true">{rtGlyph(trace.status)}</span>
		<span class="rt-label">{reasoningSigil(trace)}</span>
	</div>
	{#if cj && (cj.support || cj.objection)}
		<div class="rt-committed">
			{#if cj.support}
				<div class="rt-field"><span class="rt-k">⊢ support</span><span class="rt-v">{cj.support}</span></div>
			{/if}
			{#if cj.objection}
				<div class="rt-field"><span class="rt-k rt-obj">⊘ objection</span><span class="rt-v">{cj.objection}</span></div>
			{/if}
		</div>
	{/if}
	{#if tone === 'open'}
		<pre class="rt-cot">{reasoningBody(trace.free_cot || reasoning) || '(empty)'}</pre>
		{#if trace.free_cot && trace.free_cot_chars && trace.free_cot_chars > trace.free_cot.length}
			<p class="rt-foot">+{trace.free_cot_chars - trace.free_cot.length} more chars · clipped at export</p>
		{/if}
	{:else if trace.status === 'encrypted'}
		<p class="rt-foot rt-foot-seal">⊠ chain-of-thought sealed by provider — it reasoned, the text isn't returned</p>
	{:else if trace.status === 'not_returned'}
		<p class="rt-foot rt-foot-gap">⊠ reasoning requested but not returned by the endpoint</p>
	{:else}
		<p class="rt-foot rt-foot-gap">○ no separate chain-of-thought surfaced</p>
	{/if}
{:else}
	<pre class="rt-cot">{reasoningBody(reasoning) || '(no reasoning captured)'}</pre>
{/if}

<style>
	.rt-sigil {
		display: flex;
		align-items: baseline;
		gap: 0.35rem;
		margin: 0 0 0.4rem;
	}
	.rt-glyph {
		font-size: 0.7rem;
		line-height: 1;
	}
	.rt-label {
		font-family: var(--mono);
		font-size: 0.66rem;
		font-variant: small-caps;
		letter-spacing: 0.03em;
		color: var(--ink-faint);
	}
	/* open = readable CoT (calm neutral); sealed = reasoned but hidden (neutral);
	   gap = no usable CoT (accent — the epistemic-gap attention state). */
	.rt-open .rt-glyph {
		color: var(--ink-muted);
	}
	.rt-sealed .rt-glyph {
		color: var(--ink-faint);
	}
	.rt-gap .rt-glyph,
	.rt-gap .rt-label {
		color: var(--accent);
	}
	.rt-committed {
		display: flex;
		flex-direction: column;
		gap: 0.25rem;
		margin: 0 0 0.5rem;
		padding-bottom: 0.45rem;
		border-bottom: 1px dotted var(--rule);
	}
	.rt-field {
		display: grid;
		grid-template-columns: 5.5rem 1fr;
		gap: 0.4rem;
	}
	.rt-k {
		font-family: var(--mono);
		font-size: 0.64rem;
		font-variant: small-caps;
		letter-spacing: 0.03em;
		color: var(--ink-faint);
	}
	.rt-k.rt-obj {
		color: var(--accent);
	}
	.rt-v {
		font-size: 0.74rem;
		line-height: 1.45;
		color: var(--ink);
	}
	.rt-cot {
		font-family: var(--mono);
		font-size: 0.74rem;
		line-height: 1.5;
		white-space: pre-wrap;
		word-break: break-word;
		margin: 0;
		max-height: 32rem;
		overflow-y: auto;
		color: var(--ink);
	}
	.rt-foot {
		font-family: var(--mono);
		font-size: 0.66rem;
		color: var(--ink-faint);
		margin: 0.3rem 0 0;
	}
	.rt-foot-seal {
		color: var(--ink-muted);
	}
	.rt-foot-gap {
		color: var(--accent);
	}
</style>
