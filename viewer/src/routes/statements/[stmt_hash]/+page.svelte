<script lang="ts">
	import type { PageData } from './$types';
	import { fmtBelief, fmtDelta, pluralS, scoringMethod, shortHash, verdictDisplay } from '$lib/format';

	let { data }: { data: PageData } = $props();
	const d = $derived(data.detail);
	const r = $derived(d.rollup);
	const evidences = $derived(d.evidences);

	// our_mean − INDRA prior, when both exist.
	const delta = $derived(
		r.our_mean_score != null && r.rasmachine_belief != null
			? r.our_mean_score - r.rasmachine_belief
			: null
	);

	function beliefPhrase(b: number | null): string {
		if (b == null) return 'unscored';
		if (b >= 0.85) return 'near-certain';
		if (b >= 0.7) return 'confident';
		if (b >= 0.5) return 'moderate';
		if (b >= 0.3) return 'doubtful';
		return 'low';
	}

	const verdictLine = $derived.by(() => {
		const our = r.our_mean_score;
		const indra = r.rasmachine_belief;
		if (our == null && indra == null) return 'Not yet scored. No INDRA prior.';
		if (our == null) return `Not yet scored · INDRA prior ${fmtBelief(indra)} (${beliefPhrase(indra)}).`;
		const ourP = beliefPhrase(our);
		if (indra == null) return `We score this ${fmtBelief(our)} (${ourP}) · no INDRA prior to compare.`;
		const diff = our - indra;
		let comparison: string;
		if (Math.abs(diff) < 0.1) comparison = `matches INDRA's ${fmtBelief(indra)} (${beliefPhrase(indra)})`;
		else if (diff < -0.4) comparison = `we doubt it strongly · INDRA was ${fmtBelief(indra)} (${beliefPhrase(indra)})`;
		else if (diff < 0) comparison = `less confident than INDRA's ${fmtBelief(indra)} (${beliefPhrase(indra)})`;
		else if (diff > 0.4) comparison = `we believe it more strongly than INDRA's ${fmtBelief(indra)} (${beliefPhrase(indra)})`;
		else comparison = `more confident than INDRA's ${fmtBelief(indra)} (${beliefPhrase(indra)})`;
		return `We score this ${fmtBelief(our)} (${ourP}) · ${comparison}.`;
	});

	const scoredN = $derived(r.n_correct + r.n_incorrect);

	// Per-evidence collapsible reasoning.
	let expandedEvHash = $state<string | null>(null);
	function toggleExpand(h: string) {
		expandedEvHash = expandedEvHash === h ? null : h;
	}

	// Reasoning may carry a leading "[TIER 2 LLM]" marker; surface it as a chip
	// and show the remaining prose as the body.
	function reasoningParts(reasoning: string | null): { tag: string | null; body: string } {
		if (!reasoning) return { tag: null, body: '' };
		const m = reasoning.match(/^\s*(\[[^\]]+\])\s*/);
		if (m) return { tag: m[1], body: reasoning.slice(m[0].length) };
		return { tag: null, body: reasoning };
	}

	const REASONING_CLAMP = 280;
	function isLong(reasoning: string | null): boolean {
		return !!reasoning && reasoningParts(reasoning).body.length > REASONING_CLAMP;
	}

	function verdictTone(v: string | null): 'doubt' | 'trust' | 'abstain' {
		if (v === 'incorrect') return 'doubt';
		if (v === 'correct') return 'trust';
		return 'abstain';
	}
</script>

<svelte:head><title>{r.stmt_type} · {shortHash(r.stmt_hash)} · INDRA Belief</title></svelte:head>

<header>
	<div class="crumb">
		<a href="/statements">statements</a><span class="sep"> / </span><strong>{shortHash(r.stmt_hash)}</strong>
	</div>
	<div class="meta">
		<span class="muted">run {shortHash(d.run_id)}</span>
	</div>
</header>

<main id="main">
	<!-- Statement header (plain; the monolithic layer carries no probe trace,
	     so BeliefPrimitive's full trace mode does not apply here). -->
	<section class="stmt-header">
		<h1 class="b-sentence">{r.subject} <span class="verb">{r.stmt_type}</span> {r.object}</h1>
		<p class="b-verdict-line">{verdictLine}</p>
		<div class="b-meta">
			<span class="b-type">{r.stmt_type}</span>
			<span class="b-hash">{shortHash(r.stmt_hash)}</span>
		</div>

		<!-- Score row: ours (mean) vs INDRA prior, plus the alternative
		     aggregations the export ships. -->
		<dl class="scores">
			<div><dt>ours (mean)</dt><dd class="score-ours">{fmtBelief(r.our_mean_score)}</dd></div>
			<div><dt>INDRA prior</dt><dd>{fmtBelief(r.rasmachine_belief)}</dd></div>
			<div><dt>Δ</dt><dd class:delta-pos={delta != null && delta > 0.05} class:delta-neg={delta != null && delta < -0.05}>{fmtDelta(delta)}</dd></div>
			<div><dt>noisy-or</dt><dd>{fmtBelief(r.our_noisy_or)}</dd></div>
			<div><dt>min / max</dt><dd>{fmtBelief(r.our_min_score)} / {fmtBelief(r.our_max_score)}</dd></div>
		</dl>

		<div class="stmt-meta">
			{#if scoredN > 0}
				<span class="verdict-tally">
					{#if r.n_correct > 0}<span class="vt-correct">supported {r.n_correct}</span>{/if}
					{#if r.n_correct > 0 && r.n_incorrect > 0}<span class="dot">·</span>{/if}
					{#if r.n_incorrect > 0}<span class="vt-incorrect">contradicted {r.n_incorrect}</span>{/if}
					{#if r.n_unscored > 0}<span class="dot">·</span><span class="muted">unscored {r.n_unscored}</span>{/if}
					<span class="muted">of {r.n_evidence}</span>
				</span>
			{:else}
				<span class="hint">unscored · INDRA {fmtBelief(r.rasmachine_belief)}</span>
			{/if}
			{#if r.dominant_bucket}
				<span class="dot">·</span>
				<span>bucket {r.dominant_bucket}</span>
			{/if}
			{#if r.sources.length > 0}
				<span class="dot">·</span>
				<span class="source">{r.sources.join(', ')}</span>
			{/if}
			{#if r.pmids.length > 0}
				<span class="dot">·</span>
				<span class="muted">{r.pmids.length} pmid{pluralS(r.pmids.length)}</span>
			{/if}
		</div>
	</section>

	<section class="evidences-section">
		<h2>
			evidences
			<span class="counter">{evidences.length}</span>
		</h2>
		<p class="ev-section-note">
			Each evidence is one LLM call. Its verdict (supported / contradicted / abstained) and confidence map
			to a categorical score; the model's chain-of-thought is shown under <em>reasoning</em>.
		</p>

		{#if evidences.length === 0}
			<p class="hint">no evidence rows exported for this statement</p>
		{:else}
			{#each evidences as e (e.evidence_hash)}
				{@const parts = reasoningParts(e.reasoning)}
				{@const isExpanded = expandedEvHash === e.evidence_hash}
				{@const long = isLong(e.reasoning)}
				<article class="evidence" class:ev-expanded-state={isExpanded}>
					<div class="ev-verdict-line">
						{#if e.verdict}
							<span class="ev-verdict tone-{verdictTone(e.verdict)}">{verdictDisplay(e.verdict)}</span>
						{:else}
							<span class="ev-verdict tone-abstain">unscored</span>
						{/if}
						{#if e.confidence}<span class="ev-confidence">{e.confidence} confidence</span>{/if}
						<span class="ev-score">score <span class="ev-score-num">{fmtBelief(e.our_score)}</span></span>
					</div>
					<div class="ev-meta-secondary">
						<span class="ev-source">[{e.source_api ?? 'no source'}]</span>
						{#if e.bucket}<span class="ev-bucket">{e.bucket}</span>{/if}
						{#if e.tier}<span class="ev-tier">{scoringMethod(e.tier)}</span>{/if}
						{#if e.grounding_status}<span class="ev-grounding">grounding: {e.grounding_status}</span>{/if}
						<code class="ev-hash" title={e.evidence_hash}>{shortHash(e.evidence_hash)}</code>
					</div>
					<p class="ev-text">{e.text ?? '(no evidence text)'}</p>

					{#if parts.body}
						<div class="reasoning" class:reasoning-clamped={long && !isExpanded}>
							<div class="reasoning-head">
								<span class="reasoning-label">reasoning</span>
							</div>
							<p class="reasoning-body">{parts.body}</p>
						</div>
						{#if long}
							<button class="reasoning-toggle" aria-expanded={isExpanded} onclick={() => toggleExpand(e.evidence_hash)}>
								{isExpanded ? '▾ collapse reasoning' : '▸ expand reasoning'}
							</button>
						{/if}
					{/if}
				</article>
			{/each}
		{/if}
	</section>
</main>

<style>
	:global(:root) {
		--ink: #1a1a1a;
		--ink-muted: #6a6a6a;
		--ink-faint: #a8a8a8;
		--paper: #fdfcf8;
		--rule: #e6e2d6;
		--accent: #7d2a1a;
		--ok-green: #2a6f2a;
		--accent-wash: rgba(125, 42, 26, 0.06);
		--mono: ui-monospace, 'SF Mono', 'JetBrains Mono', Menlo, monospace;
		--serif: 'Iowan Old Style', 'Source Serif Pro', Georgia, serif;
	}

	:global(html, body) {
		background: var(--paper);
		color: var(--ink);
		font-family: var(--serif);
		font-size: 16px;
		line-height: 1.4;
		margin: 0;
	}

	header {
		display: flex;
		justify-content: space-between;
		align-items: baseline;
		padding: 0.6rem 1.5rem;
		border-bottom: 1px solid var(--rule);
		font-family: var(--mono);
		font-size: 0.78rem;
		color: var(--ink-muted);
		position: sticky;
		top: 0;
		background: var(--paper);
		z-index: 2;
	}

	.crumb a {
		color: var(--ink-muted);
		text-decoration: none;
	}
	.crumb a:hover { color: var(--ink); }
	.crumb strong { color: var(--ink); font-weight: 500; }
	.sep { color: var(--ink-faint); }

	.muted { color: var(--ink-faint); }

	main {
		max-width: 880px;
		margin: 0 auto;
		padding: 1.6rem 1.5rem 4rem;
	}

	.stmt-header {
		padding: 0 0 1.2rem;
		border-bottom: 1px solid var(--ink);
		margin: 0 0 1.5rem;
	}

	.b-sentence {
		font-family: var(--serif);
		font-style: italic;
		font-size: 1.35rem;
		font-weight: 400;
		margin: 0 0 0.3rem;
		line-height: 1.3;
		color: var(--ink);
	}
	.verb {
		font-style: normal;
		color: var(--ink-muted);
		font-size: 0.92rem;
		text-transform: lowercase;
		letter-spacing: 0.02em;
	}

	.b-verdict-line {
		font-family: var(--serif);
		font-size: 1rem;
		color: var(--ink);
		margin: 0.2rem 0 0.6rem;
		line-height: 1.45;
	}

	.b-meta {
		font-family: var(--mono);
		font-size: 0.72rem;
		color: var(--ink-faint);
		display: flex;
		gap: 0.8rem;
		margin-bottom: 1rem;
	}
	.b-type { text-transform: lowercase; letter-spacing: 0.04em; }
	.b-hash { color: var(--ink-faint); }

	.scores {
		display: flex;
		flex-wrap: wrap;
		gap: 0.4rem 1.6rem;
		margin: 0 0 1rem;
		font-family: var(--mono);
	}
	.scores div { display: flex; flex-direction: column; gap: 0.1rem; }
	.scores dt {
		font-size: 0.66rem;
		color: var(--ink-faint);
		text-transform: lowercase;
		letter-spacing: 0.03em;
	}
	.scores dd {
		margin: 0;
		font-size: 1rem;
		font-variant-numeric: tabular-nums;
		color: var(--ink);
	}
	.score-ours { color: var(--accent); font-weight: 600; }
	.delta-pos { color: var(--ok-green); }
	.delta-neg { color: var(--accent); }

	.stmt-meta {
		font-family: var(--mono);
		font-size: 0.78rem;
		color: var(--ink-muted);
		display: flex;
		flex-wrap: wrap;
		gap: 0.4rem;
		align-items: baseline;
	}
	.dot { color: var(--ink-faint); }
	.vt-correct { color: var(--ok-green); }
	.vt-incorrect { color: var(--accent); }
	.hint {
		font-family: var(--serif);
		font-style: italic;
		color: var(--ink-muted);
	}
	.source { color: var(--ink-muted); }

	.evidences-section h2 {
		font-family: var(--mono);
		font-size: 0.9rem;
		font-weight: 600;
		text-transform: lowercase;
		letter-spacing: 0.03em;
		color: var(--ink);
		border-bottom: 1px solid var(--ink);
		padding-bottom: 0.3rem;
		margin: 0 0 0.4rem;
		display: flex;
		align-items: baseline;
		gap: 0.6rem;
	}
	.counter {
		font-size: 0.74rem;
		color: var(--ink-faint);
		font-weight: 400;
	}
	.ev-section-note {
		font-family: var(--serif);
		font-style: italic;
		font-size: 0.86rem;
		color: var(--ink-faint);
		margin: 0 0 1.2rem;
		line-height: 1.45;
	}

	.evidence {
		border-bottom: 1px solid var(--rule);
		padding: 0.9rem 0;
	}
	.evidence:last-child { border-bottom: none; }

	.ev-verdict-line {
		display: flex;
		flex-wrap: wrap;
		gap: 0.8rem;
		align-items: baseline;
		font-family: var(--mono);
		font-size: 0.8rem;
		margin-bottom: 0.3rem;
	}
	.ev-verdict { font-weight: 600; }
	.ev-verdict.tone-trust { color: var(--ok-green); }
	.ev-verdict.tone-doubt { color: var(--accent); }
	.ev-verdict.tone-abstain { color: var(--ink-faint); }
	.ev-confidence { color: var(--ink-muted); }
	.ev-score { color: var(--ink-muted); }
	.ev-score-num { color: var(--ink); font-weight: 600; }

	.ev-meta-secondary {
		display: flex;
		flex-wrap: wrap;
		gap: 0.7rem;
		align-items: baseline;
		font-family: var(--mono);
		font-size: 0.68rem;
		color: var(--ink-faint);
		margin-bottom: 0.4rem;
	}
	.ev-source { color: var(--ink-muted); }
	.ev-bucket { color: var(--ink-muted); }
	.ev-tier { color: var(--ink-muted); }
	.ev-grounding { color: var(--ink-muted); }
	.ev-hash { color: var(--ink-faint); }

	.ev-text {
		font-family: var(--serif);
		font-size: 1rem;
		line-height: 1.55;
		color: var(--ink);
		margin: 0.2rem 0 0.6rem;
	}

	.reasoning {
		background: var(--accent-wash);
		border-left: 2px solid var(--accent);
		padding: 0.55rem 0.7rem;
		margin: 0.4rem 0 0;
	}
	.reasoning-clamped {
		max-height: 7.5rem;
		overflow: hidden;
		mask-image: linear-gradient(to bottom, #000 60%, transparent);
		-webkit-mask-image: linear-gradient(to bottom, #000 60%, transparent);
	}
	.reasoning-head {
		display: flex;
		gap: 0.6rem;
		align-items: baseline;
		margin-bottom: 0.3rem;
	}
	.reasoning-label {
		font-family: var(--mono);
		font-size: 0.66rem;
		text-transform: lowercase;
		letter-spacing: 0.04em;
		color: var(--ink-faint);
	}
	.reasoning-body {
		font-family: var(--serif);
		font-size: 0.92rem;
		line-height: 1.55;
		color: var(--ink-muted);
		margin: 0;
		white-space: pre-wrap;
	}
	.reasoning-toggle {
		all: unset;
		cursor: pointer;
		font-family: var(--mono);
		font-size: 0.7rem;
		color: var(--accent);
		margin-top: 0.4rem;
		display: inline-block;
	}
	.reasoning-toggle:hover { text-decoration: underline; }
</style>
