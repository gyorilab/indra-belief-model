<script lang="ts">
	import { enhance } from '$app/forms';
	import type { SubmitFunction } from '@sveltejs/kit';
	import type { PageData } from './$types';
	import type { EvidenceSample } from '$lib/data/types';
	import { CURATION_TAGS } from '$lib/data/curation';

	let { data }: { data: PageData } = $props();

	/** One-line help per tag — what each curation category means, so the curator
	 *  picks the right "why". Mirrors the tag glossary in curation.py. */
	const TAG_HELP: Record<string, string> = {
		correct: 'the sentence supports this exact extraction',
		no_relation: 'the sentence asserts no relation between these agents',
		wrong_relation: 'a relation exists but the type or direction is wrong',
		grounding: 'an agent is mapped to the wrong identifier / gene',
		polarity: 'up vs down is reversed (e.g. activation that should be inhibition)',
		act_vs_amt: 'confuses an activity change with an amount change',
		hypothesis: 'stated as a hypothesis or question, not an assertion',
		negative_result: 'the sentence reports the relation did NOT happen',
		entity_boundaries: 'the agent text span is wrong (too much / too little)',
		agent_conditions: 'wrong or missing mutation / modification / condition on an agent',
		mod_site: 'wrong residue or position on a modification',
		other: 'some other problem — describe it in the note'
	};

	// `current`/`sampleError` start from the SSR load, then we override them in
	// place via the sample/submit actions (we never call update(), so the load
	// value is just the initial seed — modelled as a derived override so no
	// reactive prop is captured into $state).
	let override = $state<EvidenceSample | null>(null);
	const current = $derived(override ?? data.sample);
	let tag = $state('');
	let busy = $state(false);
	let sampleErrorOverride = $state<string | null | undefined>(undefined);
	const sampleError = $derived(
		sampleErrorOverride !== undefined ? sampleErrorOverride : data.sampleError
	);
	let lastResult = $state<{ id: number | null; tag: string } | { error: string } | null>(null);
	let sampleFormEl: HTMLFormElement | null = $state(null);
	// Note is controlled state so it resets per sample — an uncontrolled textarea
	// keeps its DOM value across in-place sample swaps and would post a stale note
	// with the next curation.
	let note = $state('');
	// Lock a (statement, evidence) pair once submitted, so a re-click (e.g. after a
	// post-submit auto-advance fails and the card lingers) can't write a duplicate.
	let submittedKey = $state<string | null>(null);
	const currentKey = $derived(current ? `${current.matchesHash}:${current.sourceHash}` : null);
	const alreadySubmitted = $derived(currentKey != null && currentKey === submittedKey);
	type EvidenceSegment = { text: string; agentIndex: number | null };

	const evidenceSegments = $derived.by((): EvidenceSegment[] => {
		if (!current) return [];
		const text = current.text;
		const mentions = current.agents
			.map((a, i) => ({ needle: (a.rawText || a.name || '').trim(), i }))
			.filter((m) => m.needle.length >= 2)
			.sort((a, b) => b.needle.length - a.needle.length);
		const spans: Array<{ start: number; end: number; i: number }> = [];
		const occupied = new Array(text.length).fill(false);
		for (const m of mentions) {
			const lowerText = text.toLowerCase();
			const lowerNeedle = m.needle.toLowerCase();
			let pos = 0;
			while ((pos = lowerText.indexOf(lowerNeedle, pos)) !== -1) {
				const end = pos + m.needle.length;
				if (!occupied.slice(pos, end).some(Boolean)) {
					spans.push({ start: pos, end, i: m.i });
					for (let j = pos; j < end; j += 1) occupied[j] = true;
				}
				pos = end;
			}
		}
		spans.sort((a, b) => a.start - b.start);
		const out: EvidenceSegment[] = [];
		let cursor = 0;
		for (const s of spans) {
			if (s.start > cursor) out.push({ text: text.slice(cursor, s.start), agentIndex: null });
			out.push({ text: text.slice(s.start, s.end), agentIndex: s.i });
			cursor = s.end;
		}
		if (cursor < text.length) out.push({ text: text.slice(cursor), agentIndex: null });
		return out.length ? out : [{ text, agentIndex: null }];
	});

	function agentTitle(i: number): string {
		const a = current?.agents[i];
		if (!a) return '';
		const refs = Object.entries(a.dbRefs)
			.filter(([k]) => k !== 'TEXT')
			.map(([k, v]) => `${k}:${v}`)
			.join(' · ');
		const raw = a.rawText && a.rawText !== a.name ? `${a.rawText} → ` : '';
		return `${raw}${a.name}${refs ? ` · ${refs}` : ''}`;
	}

	// Draw a new sample, replacing the current card in place. Does NOT touch
	// lastResult, so a "✓ submitted" banner survives an auto-advance to the next
	// item; manual skips clear it via the button's onclick.
	const sampleEnhance: SubmitFunction = () => {
		busy = true;
		return async ({ result }) => {
			busy = false;
			if (result.type === 'success' && result.data?.sampled) {
				override = result.data.sampled as EvidenceSample;
				tag = '';
				note = '';
				submittedKey = null;
				sampleErrorOverride = null;
			} else if (result.type === 'failure') {
				sampleErrorOverride = (result.data?.sampleError as string) ?? 'sampling failed';
			} else if (result.type === 'error') {
				sampleErrorOverride = 'sampling failed (network error)';
			}
		};
	};

	const submitEnhance: SubmitFunction = ({ cancel }) => {
		if (!tag || alreadySubmitted) {
			cancel();
			return;
		}
		busy = true;
		return async ({ result }) => {
			busy = false;
			if (result.type === 'success' && result.data?.submitted) {
				const s = result.data.submitted as { id: number | null; tag: string };
				lastResult = { id: s.id, tag: s.tag };
				submittedKey = currentKey; // lock this exact pair against a duplicate write
				sampleFormEl?.requestSubmit(); // auto-advance to a fresh sample
			} else if (result.type === 'failure') {
				lastResult = { error: (result.data?.submitError as string) ?? 'submission failed' };
			} else if (result.type === 'error') {
				lastResult = { error: 'submission failed (network error)' };
			}
		};
	};
</script>

<svelte:head><title>curate · submit to INDRA</title></svelte:head>

<main class="cur">
	<header class="cur-head">
		<h1>curate</h1>
		<span class="cur-meta">
			sample evidence from the live INDRA DB, judge the extraction, submit a curation
			· <span class="cur-host">{data.dbHost}</span>
		</span>
		{#if data.user}
			<span class="cur-as"
				>curating as <strong>{data.user.email}</strong> — your INDRA account; each curation is
				submitted under and attributed to you</span
			>
		{/if}
	</header>

	{#if lastResult}
		{#if 'error' in lastResult}
			<div class="cur-banner bad">✗ {lastResult.error}</div>
		{:else}
			<div class="cur-banner ok">
				✓ submitted curation{#if lastResult.id != null} <strong>#{lastResult.id}</strong>{/if}
				· <span class="cur-tagchip" class:ok={lastResult.tag === 'correct'}>{lastResult.tag}</span> to INDRA
			</div>
		{/if}
	{/if}

	{#if current}
		<section class="cur-item">
			{#if sampleError}
				<div class="cur-banner bad">⚠ {sampleError} — try “sample another”.</div>
			{/if}
			<div class="cur-claim">
				<span class="cur-subj">{current.claim.subject}</span>
				<span class="cur-rel">{current.claim.relation}</span>
				<span class="cur-obj">{current.claim.object}</span>
			</div>
			<p class="cur-evidence">
				{#each evidenceSegments as seg}
					{#if seg.agentIndex == null}
						{seg.text}
					{:else}
						<span
							class="cur-entity"
							class:subj={current.agents[seg.agentIndex]?.role === 'subject'}
							class:obj={current.agents[seg.agentIndex]?.role === 'object'}
							class:member={current.agents[seg.agentIndex]?.role === 'member'}
							style={`--agent-i: ${seg.agentIndex}`}
							data-tip={agentTitle(seg.agentIndex)}>{seg.text}</span
						>
					{/if}
				{/each}
			</p>
			<p class="cur-prov">
				[{current.sourceApi ?? 'no source'}]
				{#if current.pmid}
					· <a href="https://pubmed.ncbi.nlm.nih.gov/{current.pmid}" target="_blank" rel="noreferrer"
						>PMID {current.pmid}</a
					>
				{/if}
				· {current.stmtType}
				{#if current.belief != null} · belief {current.belief.toFixed(2)}{/if}
				· {current.evCount} evidence · via {current.agentQuery}
			</p>
			<p class="cur-hash muted">stmt {current.matchesHash} · ev {current.sourceHash}</p>

			<form method="POST" action="?/submit" use:enhance={submitEnhance} class="cur-form">
				<input type="hidden" name="matches_hash" value={current.matchesHash} />
				<input type="hidden" name="source_hash" value={current.sourceHash} />
				<fieldset>
					<legend>does the sentence support this exact extraction? pick the curation tag</legend>
					{#each CURATION_TAGS as t}
						<label class="cur-opt" class:selected={tag === t} class:isok={t === 'correct'}>
							<input type="radio" name="tag" value={t} bind:group={tag} required />
							<span class="cur-opt-name">{t}</span>
							<span class="cur-opt-help">{TAG_HELP[t]}</span>
						</label>
					{/each}
				</fieldset>
				<label class="cur-notes">
					note {tag && tag !== 'correct' ? '(what is wrong)' : '(optional)'}
					<textarea
						name="text"
						rows="2"
						bind:value={note}
						placeholder={tag && tag !== 'correct' ? 'briefly, what is wrong' : 'optional'}
					></textarea>
				</label>
				<div class="cur-actions">
					<button class="cur-go" type="submit" disabled={busy || alreadySubmitted}>
						{busy ? 'working…' : alreadySubmitted ? 'submitted ✓' : 'submit curation →'}
					</button>
				</div>
			</form>

			<form
				method="POST"
				action="?/sample"
				use:enhance={sampleEnhance}
				bind:this={sampleFormEl}
				class="cur-skip"
			>
				<button type="submit" class="cur-skip-btn" disabled={busy} onclick={() => (lastResult = null)}>
					skip · sample another →
				</button>
			</form>
		</section>
	{:else}
		<div class="cur-empty">
			<p class="cur-err">{sampleError ?? 'No evidence sampled.'}</p>
			<form method="POST" action="?/sample" use:enhance={sampleEnhance} bind:this={sampleFormEl}>
				<button type="submit" class="cur-go" disabled={busy}>{busy ? 'working…' : 'try again →'}</button>
			</form>
		</div>
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
	.cur { max-width: 48rem; margin: 0 auto; padding: 1.5rem; font-family: var(--serif); color: var(--ink); }
	.cur-head { border-bottom: 1px solid var(--rule); padding-bottom: 0.6rem; margin-bottom: 1.2rem; }
	.cur-head h1 { font-size: 1.1rem; font-weight: 400; margin: 0; }
	.cur-meta { font-family: var(--mono); font-size: 0.74rem; color: var(--ink-muted); }
	.cur-host { color: var(--ink); }
	.cur-as {
		display: block; font-family: var(--mono); font-size: 0.72rem; color: var(--ink-faint); margin-top: 0.3rem;
	}
	.cur-as strong { color: var(--ink); font-weight: 600; }

	.cur-banner {
		font-family: var(--mono); font-size: 0.82rem; border-radius: 3px;
		padding: 0.5rem 0.75rem; margin-bottom: 1.1rem; border: 1px solid;
	}
	.cur-banner.ok { color: var(--ok-green); background: var(--ok-wash); border-color: var(--ok-green); }
	.cur-banner.bad { color: var(--accent); background: var(--accent-wash); border-color: var(--accent); }
	.cur-tagchip { padding: 0.05rem 0.35rem; border-radius: 2px; background: var(--accent-wash); color: var(--accent); }
	.cur-tagchip.ok { background: rgba(42, 111, 42, 0.1); color: var(--ok-green); }

	.cur-claim { font-size: 1.3rem; line-height: 1.3; margin-bottom: 0.7rem; }
	.cur-subj, .cur-obj { font-style: italic; }
	.cur-rel { font-family: var(--mono); font-size: 0.95rem; color: var(--ink-muted); padding: 0 0.35rem; }
	.cur-evidence {
		font-size: 1.05rem; line-height: 1.55; color: var(--ink);
		border-left: 3px solid var(--rule); padding: 0.4rem 0 0.4rem 1rem; margin: 0.4rem 0;
	}
	.cur-entity {
		position: relative; border-radius: 2px; padding: 0 0.12rem; cursor: help;
		background: color-mix(in srgb, #d8b13f 26%, transparent);
		box-shadow: inset 0 -2px 0 color-mix(in srgb, #d8b13f 55%, transparent);
	}
	.cur-entity.subj { background: color-mix(in srgb, #3a7f77 22%, transparent); box-shadow: inset 0 -2px 0 color-mix(in srgb, #3a7f77 55%, transparent); }
	.cur-entity.obj { background: color-mix(in srgb, #9c4f33 20%, transparent); box-shadow: inset 0 -2px 0 color-mix(in srgb, #9c4f33 52%, transparent); }
	.cur-entity.member:nth-of-type(2n) { background: color-mix(in srgb, #4f6fa8 20%, transparent); box-shadow: inset 0 -2px 0 color-mix(in srgb, #4f6fa8 52%, transparent); }
	.cur-entity[data-tip]:hover::after,
	.cur-entity[data-tip]:focus-visible::after {
		content: attr(data-tip); position: absolute; z-index: 10; left: 0; bottom: calc(100% + 0.35rem);
		width: max-content; max-width: min(28rem, 80vw); white-space: normal;
		font-family: var(--mono); font-size: 0.68rem; line-height: 1.35; color: var(--paper);
		background: var(--ink); border: 1px solid var(--ink); border-radius: 2px;
		padding: 0.35rem 0.45rem; box-shadow: 0 2px 10px rgba(0, 0, 0, 0.18);
	}
	.cur-prov { font-family: var(--mono); font-size: 0.7rem; color: var(--ink-faint); margin: 0 0 0.2rem; }
	.cur-prov a { color: var(--accent); text-decoration: none; }
	.cur-prov a:hover { text-decoration: underline; }
	.cur-hash { font-family: var(--mono); font-size: 0.62rem; margin: 0 0 1.3rem; word-break: break-all; }

	.cur-form fieldset { border: none; margin: 0 0 1.1rem; padding: 0; }
	.cur-form legend { font-family: var(--mono); font-size: 0.78rem; color: var(--ink-muted); margin-bottom: 0.5rem; }
	.cur-opt {
		display: grid; grid-template-columns: 1.2rem auto 1fr; align-items: baseline; gap: 0.4rem;
		padding: 0.28rem 0.4rem; border: 1px solid transparent; border-radius: 2px; cursor: pointer;
	}
	.cur-opt:hover { background: var(--accent-wash); }
	.cur-opt.selected { border-color: var(--rule); background: var(--accent-wash); }
	.cur-opt.isok.selected { background: var(--ok-wash); border-color: var(--ok-green); }
	.cur-opt input { margin: 0; }
	.cur-opt-name { font-family: var(--mono); font-size: 0.82rem; color: var(--ink); }
	.cur-opt.isok .cur-opt-name { color: var(--ok-green); }
	.cur-opt-help { font-size: 0.82rem; color: var(--ink-faint); }

	.cur-notes {
		display: block; font-family: var(--mono); font-size: 0.72rem; color: var(--ink-muted); margin-bottom: 0.8rem;
	}
	.cur-notes textarea {
		display: block; width: 100%; margin-top: 0.25rem; font-family: var(--serif); font-size: 0.92rem;
		border: 1px solid var(--rule); border-radius: 2px; padding: 0.4rem; background: var(--paper); color: var(--ink);
	}

	.cur-actions { margin-top: 0.3rem; }
	.cur-go {
		font-family: var(--mono); font-size: 0.82rem; color: var(--paper); background: var(--ink);
		border: none; padding: 0.5rem 1rem; cursor: pointer; border-radius: 2px;
	}
	.cur-go:hover:not(:disabled) { background: var(--accent); }
	.cur-go:disabled { opacity: 0.5; cursor: default; }

	.cur-skip { margin-top: 1rem; border-top: 1px solid var(--rule); padding-top: 0.8rem; }
	.cur-skip-btn {
		font-family: var(--mono); font-size: 0.76rem; color: var(--ink-muted); background: none;
		border: none; padding: 0; cursor: pointer;
	}
	.cur-skip-btn:hover:not(:disabled) { color: var(--ink); text-decoration: underline; text-underline-offset: 3px; }
	.cur-skip-btn:disabled { opacity: 0.5; cursor: default; }

	.cur-empty { padding: 2rem 0; }
	.cur-err { font-family: var(--mono); font-size: 0.82rem; color: var(--accent); margin-bottom: 1rem; }
	.muted { color: var(--ink-faint); }
</style>
