<script lang="ts">
	import type { PageData, ActionData } from './$types';
	import ReasoningPanel from '$lib/components/ReasoningPanel.svelte';
	let { data, form }: { data: PageData; form: ActionData } = $props();

	const VERDICT = [
		['correct', 'supported — the sentence supports this exact claim'],
		['incorrect', "incorrect — the sentence doesn't support / contradicts it"],
		['abstain', 'abstain — genuinely unclear']
	];
	const QUALITY = [
		['sound', 'sound'],
		['wrong', 'wrong'],
		['na', 'n/a']
	];

	function verdictWord(v: string | null): string {
		return v === 'correct' ? 'supported' : v === 'incorrect' ? 'contradicted' : (v ?? '—');
	}
	// phase 2 active once commit has revealed both models
	const rev = $derived(form && 'revealed' in form && form.revealed ? form : null);
	// which model did the blinded verdict side with? (models disagree, so ≤1 matches)
	const sided = $derived(
		!rev ? null : rev.human_verdict === 'abstain'
			? 'neither'
			: rev.human_verdict === rev.a.verdict
				? 'a'
				: rev.human_verdict === rev.b.verdict
					? 'b'
					: 'neither'
	);
	const sidedModel = $derived(sided === 'a' ? rev?.a.model : sided === 'b' ? rev?.b.model : null);

	// INDRA community curation — the independent third judge, revealed alongside
	// the models. Present only when this evidence has been curated upstream.
	const gold = $derived(rev && 'gold' in rev ? rev.gold : null);
	const humanVsGold = $derived(
		!gold || !rev ? null : rev.human_verdict === 'abstain' ? 'abstain' : rev.human_verdict === gold.verdict ? 'match' : 'miss'
	);
</script>

<svelte:head><title>adjudicate · disagreements</title></svelte:head>

<main class="aj">
	<header class="aj-head">
		<h1>disagreement adjudication</h1>
		<span class="aj-meta">
			<strong>{data.annotator}</strong>
			{#if data.models}· <span class="aj-vs">{data.models.a}</span> vs <span class="aj-vs">{data.models.b}</span>{/if}
			· <span class="aj-prog">{data.progress.done}/{data.progress.total} done</span>
		</span>
		{#if data.annotators.length > 1}
			<span class="aj-switch">
				annotator:
				{#each data.annotators as a}
					<a class:active={a === data.annotator} href={`/adjudicate?annotator=${a}`}>{a}</a>
				{/each}
			</span>
		{/if}
	</header>

	{#if form?.error}<p class="aj-err">{form.error}</p>{/if}

	{#if !data.ready}
		<div class="aj-done">
			<p>No disagreement queue found.</p>
			<p class="muted">
				Build one: <code
					>python scripts/build_disagreement_queue.py --run-a &lt;export-a&gt; --run-b &lt;export-b&gt;</code
				>
			</p>
		</div>
	{:else if !data.item}
		<div class="aj-done">
			<p>✓ queue complete for <strong>{data.annotator}</strong> ({data.progress.total} items).</p>
			<p class="muted">
				Each label records the blinded verdict + which model it sided with, reweightable to the full
				disagreement population.
			</p>
		</div>
	{:else}
		{@const it = data.item}
		<section class="aj-item">
			<div class="aj-claim">
				<span class="aj-subj">{it.subject}</span>
				<span class="aj-rel">{it.stmt_type}</span>
				<span class="aj-obj">{it.object}</span>
			</div>
			<p class="aj-evidence">{it.evidence_text || '(no evidence text)'}</p>
			<p class="aj-prov">[{it.source_api ?? 'no source'}]{#if it.pmid} · PMID {it.pmid}{/if}</p>

			{#if !rev}
				<!-- PHASE 1 · blinded — neither model's verdict is on this page yet -->
				<form method="POST" action={`?/commit&annotator=${encodeURIComponent(data.annotator)}`} class="aj-form">
					<input type="hidden" name="item_id" value={it.item_id} />
					<fieldset>
						<legend>your verdict — does the sentence support the claim?</legend>
						{#each VERDICT as [val, label]}
							<label><input type="radio" name="human_verdict" value={val} required /> {label}</label>
						{/each}
					</fieldset>
					<button class="aj-go" type="submit">commit &amp; reveal both models →</button>
					<p class="aj-hint muted">
						The two models gave opposite verdicts here. Commit your own call first; both reveal only
						after.
					</p>
				</form>
			{:else}
				<!-- PHASE 2 · both models revealed -->
				<div class="aj-banner-row">
					{#if sided === 'neither'}
						<span class="aj-banner neither">you sided with neither model</span>
					{:else}
						<span class="aj-banner sided">✓ you sided with <strong>{sidedModel}</strong></span>
					{/if}
				</div>

				<div class="aj-pair">
					{#each [rev.a, rev.b] as m, i}
						{@const isSided = (sided === 'a' && i === 0) || (sided === 'b' && i === 1)}
						<div class="aj-model" class:sided={isSided}>
							<div class="aj-model-head">
								<span class="aj-model-name">{m.model}</span>
								{#if isSided}<span class="aj-tick">agrees with you</span>{/if}
							</div>
							<div class="aj-verdict aj-{m.verdict}">{verdictWord(m.verdict)}</div>
							<div class="aj-scoreline">
								{m.score?.toFixed(2) ?? '—'} · <span class="muted">{m.bucket ?? '—'}</span>
							</div>
							<div class="aj-reasoning">
								<ReasoningPanel trace={m.reasoning_trace} reasoning={m.reasoning} />
							</div>
						</div>
					{/each}
				</div>

				{#if gold}
					<div class="aj-gold" class:gmiss={humanVsGold === 'miss'}>
						<div class="aj-gold-head">
							<span class="aj-gold-name">INDRA community curation</span>
							<span class="aj-gold-verdict aj-{gold.verdict}">{verdictWord(gold.verdict)}</span>
							<span class="muted">· {gold.n} curation{gold.n === 1 ? '' : 's'}, any-incorrect-wins</span>
						</div>
						<div class="aj-gold-tags">
							{#each gold.tags as t}<span class="aj-tag" class:ok={t === 'correct'}>{t}</span>{/each}
						</div>
						<div class="aj-gold-grade">
							{#if humanVsGold === 'match'}<span class="g-ok">✓ your call matches the curation</span>
							{:else if humanVsGold === 'miss'}<span class="g-bad">✗ your call differs from the curation</span>
							{:else}<span class="muted">you abstained</span>{/if}
							· <span class="muted">{rev.a.model}</span> {rev.a.verdict === gold.verdict ? '✓' : '✗'}
							· <span class="muted">{rev.b.model}</span> {rev.b.verdict === gold.verdict ? '✓' : '✗'}
						</div>
						{#if gold.curators.length}<div class="aj-gold-cur muted">curators: {gold.curators.join(', ')}</div>{/if}
						{#if gold.notes.length}<div class="aj-gold-note">{gold.notes.join(' · ')}</div>{/if}
					</div>
				{:else}
					<p class="aj-nogold muted">no INDRA curation for this evidence — your call stands as the reference here</p>
				{/if}

				<form method="POST" action={`?/submit&annotator=${encodeURIComponent(data.annotator)}`} class="aj-form">
					<input type="hidden" name="item_id" value={rev.item_id} />
					<input type="hidden" name="human_verdict" value={rev.human_verdict} />
					<div class="aj-quality">
						<div class="aj-qcol">
							<span class="aj-qlabel">{rev.a.model} reasoning</span>
							{#each QUALITY as [val, label]}
								<label><input type="radio" name="reasoning_a" value={val} /> {label}</label>
							{/each}
						</div>
						<div class="aj-qcol">
							<span class="aj-qlabel">{rev.b.model} reasoning</span>
							{#each QUALITY as [val, label]}
								<label><input type="radio" name="reasoning_b" value={val} /> {label}</label>
							{/each}
						</div>
					</div>
					<label class="aj-amb"
						><input type="checkbox" name="ambiguous" /> genuinely ambiguous — reasonable annotators could differ</label
					>
					<label class="aj-notes">notes <textarea name="notes" rows="2" placeholder="optional"></textarea></label>
					<button class="aj-go" type="submit">save &amp; next →</button>
				</form>
			{/if}
		</section>
	{/if}
</main>

<style>
	:root {
		--ink: #1a1a1a; --ink-muted: #6a6a6a; --ink-faint: #727272;
		--paper: #fdfcf8; --rule: #e6e2d6; --accent: #7d2a1a;
		--accent-wash: rgba(125, 42, 26, 0.04); --ok-green: #2a6f2a;
		--ok-wash: rgba(42, 111, 42, 0.05);
		--mono: ui-monospace, 'SF Mono', Menlo, monospace;
		--serif: 'Iowan Old Style', 'Source Serif Pro', Georgia, serif;
	}
	.aj { max-width: 48rem; margin: 0 auto; padding: 1.5rem; font-family: var(--serif); color: var(--ink); }
	.aj-head { border-bottom: 1px solid var(--rule); padding-bottom: 0.6rem; margin-bottom: 1.2rem; }
	.aj-head h1 { font-size: 1.1rem; font-weight: 400; margin: 0; }
	.aj-meta, .aj-switch { font-family: var(--mono); font-size: 0.74rem; color: var(--ink-muted); }
	.aj-vs { color: var(--ink); }
	.aj-switch { display: block; margin-top: 0.3rem; }
	.aj-switch a { color: var(--accent); text-decoration: none; margin-left: 0.4rem; }
	.aj-switch a.active { color: var(--ink); font-weight: 600; text-decoration: underline; }
	.aj-prog { color: var(--ink); }
	.aj-err { font-family: var(--mono); font-size: 0.8rem; color: var(--accent); }
	.aj-done { padding: 2rem 0; font-size: 1.05rem; }

	.aj-claim { font-size: 1.3rem; line-height: 1.3; margin-bottom: 0.7rem; }
	.aj-subj, .aj-obj { font-style: italic; }
	.aj-rel { font-family: var(--mono); font-size: 0.95rem; color: var(--ink-muted); padding: 0 0.35rem; }
	.aj-evidence {
		font-size: 1.05rem; line-height: 1.55; color: var(--ink);
		border-left: 3px solid var(--rule); padding: 0.4rem 0 0.4rem 1rem; margin: 0.4rem 0;
	}
	.aj-prov { font-family: var(--mono); font-size: 0.7rem; color: var(--ink-faint); margin: 0 0 1.3rem; }

	.aj-form fieldset { border: none; margin: 0 0 1.1rem; padding: 0; }
	.aj-form legend { font-family: var(--mono); font-size: 0.78rem; color: var(--ink-muted); margin-bottom: 0.4rem; }
	.aj-form label { display: block; padding: 0.22rem 0; font-size: 0.95rem; cursor: pointer; }
	.aj-form input[type='radio'] { margin-right: 0.5rem; }
	.aj-go {
		font-family: var(--mono); font-size: 0.82rem; color: var(--paper); background: var(--ink);
		border: none; padding: 0.5rem 1rem; cursor: pointer; border-radius: 2px; margin-top: 0.3rem;
	}
	.aj-go:hover { background: var(--accent); }
	.aj-hint { margin: 0.5rem 0 0; font-size: 0.74rem; font-style: italic; }

	.aj-banner-row { margin: 0.4rem 0 0.8rem; }
	.aj-banner { font-family: var(--mono); font-size: 0.8rem; font-weight: 600; }
	.aj-banner.sided { color: var(--ok-green); }
	.aj-banner.neither { color: var(--ink-muted); }

	.aj-pair { display: grid; grid-template-columns: 1fr 1fr; gap: 0.8rem; margin-bottom: 1.3rem; }
	.aj-model { border: 1px solid var(--rule); border-radius: 3px; padding: 0.7rem 0.85rem; }
	.aj-model.sided { border-left: 3px solid var(--ok-green); background: var(--ok-wash); }
	.aj-model-head { display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 0.4rem; }
	.aj-model-name { font-family: var(--mono); font-size: 0.78rem; font-weight: 600; color: var(--ink); }
	.aj-tick { font-family: var(--mono); font-size: 0.64rem; color: var(--ok-green); }
	.aj-verdict { font-family: var(--mono); font-size: 0.95rem; font-weight: 600; }
	.aj-verdict.aj-correct { color: var(--ok-green); }
	.aj-verdict.aj-incorrect { color: var(--accent); }
	.aj-scoreline { font-family: var(--mono); font-size: 0.72rem; color: var(--ink-muted); margin: 0.15rem 0 0.5rem; }
	/* spacing/divider wrapper only — ReasoningPanel owns all text rendering */
	.aj-reasoning {
		border-top: 1px solid var(--rule);
		padding-top: 0.5rem;
		margin: 0.3rem 0 0;
	}

	/* INDRA curation gold reveal (third judge) */
	.aj-gold {
		border: 1px solid var(--gold-hue, #5a4a86); border-left-width: 3px;
		background: var(--gold-wash, rgba(90, 74, 134, 0.08));
		border-radius: 3px; padding: 0.7rem 0.85rem; margin-bottom: 1.3rem;
	}
	.aj-gold.gmiss { border-color: var(--accent); border-left-color: var(--accent); }
	.aj-gold-head { display: flex; align-items: baseline; gap: 0.5rem; flex-wrap: wrap; margin-bottom: 0.4rem; }
	.aj-gold-name { font-family: var(--mono); font-size: 0.78rem; font-weight: 600; color: var(--gold-hue, #5a4a86); }
	.aj-gold-verdict { font-family: var(--mono); font-size: 0.9rem; font-weight: 600; }
	.aj-gold-tags { display: flex; flex-wrap: wrap; gap: 0.25rem; margin-bottom: 0.4rem; }
	.aj-tag { font-family: var(--mono); font-size: 0.64rem; padding: 0.05rem 0.35rem; border-radius: 2px; background: var(--accent-wash); color: var(--accent); }
	.aj-tag.ok { background: rgba(42, 111, 42, 0.1); color: var(--ok-green); }
	.aj-gold-grade { font-size: 0.82rem; margin-bottom: 0.3rem; }
	.g-ok { color: var(--ok-green); font-weight: 600; }
	.g-bad { color: var(--accent); font-weight: 600; }
	.aj-gold-cur { font-family: var(--mono); font-size: 0.68rem; }
	.aj-gold-note { font-size: 0.8rem; font-style: italic; color: var(--ink-muted); margin-top: 0.3rem; }
	.aj-nogold { font-size: 0.82rem; font-style: italic; margin-bottom: 1.3rem; }

	.aj-quality { display: grid; grid-template-columns: 1fr 1fr; gap: 0.8rem; margin-bottom: 0.9rem; }
	.aj-qcol { font-size: 0.9rem; }
	.aj-qlabel { display: block; font-family: var(--mono); font-size: 0.7rem; color: var(--ink-muted); margin-bottom: 0.3rem; }
	.aj-qcol label { display: inline-block; margin-right: 0.7rem; padding: 0.15rem 0; cursor: pointer; }
	.aj-amb { display: block; font-size: 0.88rem; margin-bottom: 0.7rem; cursor: pointer; }
	.aj-amb input { margin-right: 0.4rem; }
	.aj-notes { display: block; font-family: var(--mono); font-size: 0.72rem; color: var(--ink-muted); margin-bottom: 0.8rem; }
	.aj-notes textarea {
		display: block; width: 100%; margin-top: 0.2rem; font-family: var(--serif); font-size: 0.9rem;
		border: 1px solid var(--rule); border-radius: 2px; padding: 0.4rem; background: var(--paper);
	}
	.muted { color: var(--ink-faint); }
	code { font-family: var(--mono); font-size: 0.82em; }
	@media (max-width: 640px) {
		.aj-pair, .aj-quality { grid-template-columns: 1fr; }
	}
</style>
