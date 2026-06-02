<script lang="ts">
	import type { PageData, ActionData } from './$types';
	let { data, form }: { data: PageData; form: ActionData } = $props();

	const AXIS_A = [
		['faithful', 'faithful — the sentence expresses this exact relation'],
		['unfaithful', "unfaithful — the sentence doesn't express it (reader error)"],
		['no_text', 'no usable evidence text'],
		['cant_tell', "can't tell"]
	];
	const AXIS_B = [
		['correct', 'supports the claim'],
		['incorrect', "contradicts / doesn't support"],
		['abstain', 'abstain — genuinely unclear']
	];
	const AXIS_C = [
		['sound', 'sound'],
		['right_call_wrong_reason', 'right call, wrong reason'],
		['wrong', 'wrong'],
		['na', 'n/a']
	];
	const AXIS_D = [
		['genuine_semantic', 'genuine semantic mismatch'],
		['reader_artifact', 'reader hallucination / artifact'],
		['empty_evidence', 'empty / no evidence'],
		['direction_reversed', 'direction or sign reversed'],
		['hedge_mishandled', 'hedge mishandled'],
		['partial_claim', 'partial / incomplete claim'],
		['other', 'other']
	];

	function verdictWord(v: string | null): string {
		return v === 'correct' ? 'supported' : v === 'incorrect' ? 'contradicted' : (v ?? '—');
	}
	function reasoningBody(r: string | null): string {
		if (!r) return '';
		return r.replace(/^\s*\[[^\]]+\][ \t]*\n?/, '').trim();
	}
	// phase 2 is active once the commit action has revealed the model's call.
	// Capture the narrowed (non-null) reveal payload so the template type-checks.
	const rev = $derived(form && 'revealed' in form && form.revealed ? form : null);
	const agree = $derived(!!rev && rev.axis_b === rev.model?.verdict);
</script>

<svelte:head><title>review · {data.pass}</title></svelte:head>

<main class="rv">
	<header class="rv-head">
		<h1>blinded review</h1>
		<span class="rv-meta">
			{data.pass} · <strong>{data.annotator}</strong>
			{#if data.run}· {data.run.model}{/if}
			· <span class="rv-prog">{data.progress.done}/{data.progress.total} done</span>
		</span>
		{#if data.annotators.length > 1}
			<span class="rv-switch">
				annotator:
				{#each data.annotators as a}
					<a class:active={a === data.annotator} href={`/review?pass=${data.pass}&annotator=${a}`}>{a}</a>
				{/each}
			</span>
		{/if}
	</header>

	{#if form?.error}<p class="rv-err">{form.error}</p>{/if}

	{#if !data.item}
		<div class="rv-done">
			<p>✓ queue complete for <strong>{data.annotator}</strong> ({data.progress.total} items).</p>
			<p class="muted">Run <code>scripts/analyze_labels.py</code> to reweight + report (next).</p>
		</div>
	{:else}
		{@const it = data.item}
		<section class="rv-item">
			<div class="rv-claim">
				<span class="rv-subj">{it.subject}</span>
				<span class="rv-rel">{it.stmt_type}</span>
				<span class="rv-obj">{it.object}</span>
			</div>
			<p class="rv-evidence">{it.evidence_text || '(no evidence text)'}</p>
			<p class="rv-prov">[{it.source_api ?? 'no source'}]{#if it.pmid} · PMID {it.pmid}{/if}</p>

			{#if !rev}
				<!-- PHASE 1 · blinded — the model's call is NOT on this page yet -->
				<form method="POST" action={`?/commit&pass=${data.pass}&annotator=${encodeURIComponent(data.annotator)}`} class="rv-form">
					<input type="hidden" name="item_id" value={it.item_id} />
					<fieldset>
						<legend>A · does this <em>sentence</em> express the relation above?</legend>
						{#each AXIS_A as [val, label]}
							<label><input type="radio" name="axis_a_faithful" value={val} required /> {label}</label>
						{/each}
					</fieldset>
					<fieldset>
						<legend>B · <strong>your</strong> verdict on the extraction</legend>
						{#each AXIS_B as [val, label]}
							<label><input type="radio" name="axis_b_human_verdict" value={val} required /> {label}</label>
						{/each}
					</fieldset>
					<button class="rv-go" type="submit">commit &amp; reveal model →</button>
					<p class="rv-hint muted">Commit your own judgment first; the model's verdict + reasoning reveal only after.</p>
				</form>
			{:else}
				<!-- PHASE 2 · revealed -->
				<div class="rv-reveal" class:agree class:disagree={!agree}>
					<span class="rv-banner">{agree ? '✓ you agree with the model' : '✗ you disagree with the model'}</span>
					<div class="rv-model">
						model: <strong>{verdictWord(rev.model.verdict)}</strong>
						· {rev.model.score?.toFixed(2)} · {rev.model.confidence}
						· <span class="muted">{rev.model.bucket}</span>
					</div>
					{#if reasoningBody(rev.model.reasoning)}
						<details class="rv-reasoning"><summary>model reasoning</summary><p>{reasoningBody(rev.model.reasoning)}</p></details>
					{/if}
				</div>

				<form method="POST" action={`?/submit&pass=${data.pass}&annotator=${encodeURIComponent(data.annotator)}`} class="rv-form">
					<input type="hidden" name="item_id" value={rev.item_id} />
					<input type="hidden" name="axis_a_faithful" value={rev.axis_a} />
					<input type="hidden" name="axis_b_human_verdict" value={rev.axis_b} />
					<fieldset>
						<legend>C · model reasoning quality</legend>
						{#each AXIS_C as [val, label]}
							<label><input type="radio" name="axis_c_reasoning" value={val} /> {label}</label>
						{/each}
					</fieldset>
					{#if !agree}
						<fieldset>
							<legend>D · failure mode <span class="muted">(you disagreed)</span></legend>
							{#each AXIS_D as [val, label]}
								<label><input type="radio" name="axis_d_failure" value={val} required /> {label}</label>
							{/each}
						</fieldset>
					{/if}
					<label class="rv-notes">notes <textarea name="notes" rows="2" placeholder="optional"></textarea></label>
					<button class="rv-go" type="submit">save &amp; next →</button>
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
		--mono: ui-monospace, 'SF Mono', Menlo, monospace;
		--serif: 'Iowan Old Style', 'Source Serif Pro', Georgia, serif;
	}
	.rv { max-width: 44rem; margin: 0 auto; padding: 1.5rem; font-family: var(--serif); color: var(--ink); }
	.rv-head { border-bottom: 1px solid var(--rule); padding-bottom: 0.6rem; margin-bottom: 1.2rem; }
	.rv-head h1 { font-size: 1.1rem; font-weight: 400; margin: 0; }
	.rv-meta, .rv-switch { font-family: var(--mono); font-size: 0.74rem; color: var(--ink-muted); }
	.rv-switch { display: block; margin-top: 0.3rem; }
	.rv-switch a { color: var(--accent); text-decoration: none; margin-left: 0.4rem; }
	.rv-switch a.active { color: var(--ink); font-weight: 600; text-decoration: underline; }
	.rv-prog { color: var(--ink); }
	.rv-err { font-family: var(--mono); font-size: 0.8rem; color: var(--accent); }
	.rv-done { padding: 2rem 0; font-size: 1.05rem; }

	.rv-claim { font-size: 1.3rem; line-height: 1.3; margin-bottom: 0.7rem; }
	.rv-subj, .rv-obj { font-style: italic; }
	.rv-rel { font-family: var(--mono); font-size: 0.95rem; color: var(--ink-muted); padding: 0 0.35rem; }
	.rv-evidence {
		font-size: 1.05rem; line-height: 1.55; color: var(--ink);
		border-left: 3px solid var(--rule); padding: 0.4rem 0 0.4rem 1rem; margin: 0.4rem 0;
	}
	.rv-prov { font-family: var(--mono); font-size: 0.7rem; color: var(--ink-faint); margin: 0 0 1.3rem; }

	.rv-form fieldset { border: none; margin: 0 0 1.1rem; padding: 0; }
	.rv-form legend { font-family: var(--mono); font-size: 0.78rem; color: var(--ink-muted); margin-bottom: 0.4rem; }
	.rv-form label { display: block; padding: 0.22rem 0; font-size: 0.95rem; cursor: pointer; }
	.rv-form input[type='radio'] { margin-right: 0.5rem; }
	.rv-go {
		font-family: var(--mono); font-size: 0.82rem; color: var(--paper); background: var(--ink);
		border: none; padding: 0.5rem 1rem; cursor: pointer; border-radius: 2px; margin-top: 0.3rem;
	}
	.rv-go:hover { background: var(--accent); }
	.rv-hint { margin: 0.5rem 0 0; font-size: 0.74rem; font-style: italic; }
	.rv-notes { display: block; font-family: var(--mono); font-size: 0.72rem; color: var(--ink-muted); margin-bottom: 0.8rem; }
	.rv-notes textarea {
		display: block; width: 100%; margin-top: 0.2rem; font-family: var(--serif); font-size: 0.9rem;
		border: 1px solid var(--rule); border-radius: 2px; padding: 0.4rem; background: var(--paper);
	}

	.rv-reveal { border: 1px solid var(--rule); border-radius: 3px; padding: 0.7rem 0.9rem; margin: 0.5rem 0 1.3rem; }
	.rv-reveal.agree { border-left: 3px solid var(--ok-green); }
	.rv-reveal.disagree { border-left: 3px solid var(--accent); background: var(--accent-wash); }
	.rv-banner { font-family: var(--mono); font-size: 0.78rem; font-weight: 600; }
	.rv-reveal.agree .rv-banner { color: var(--ok-green); }
	.rv-reveal.disagree .rv-banner { color: var(--accent); }
	.rv-model { font-family: var(--mono); font-size: 0.82rem; margin-top: 0.4rem; }
	.rv-reasoning { margin-top: 0.5rem; }
	.rv-reasoning summary { font-family: var(--mono); font-size: 0.72rem; color: var(--ink-muted); cursor: pointer; }
	.rv-reasoning p { font-family: var(--mono); font-size: 0.78rem; line-height: 1.5; color: var(--ink-muted); white-space: pre-wrap; margin: 0.4rem 0 0; }
	.muted { color: var(--ink-faint); }
	code { font-family: var(--mono); font-size: 0.85em; }
</style>
