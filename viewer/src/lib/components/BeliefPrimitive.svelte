<script lang="ts" module>
	/** One scored (statement, evidence) row — the monolithic Focus.evidences shape. */
	export interface BeliefEvidence {
		evidence_hash: string;
		source_api: string | null;
		text: string | null;
		our_score?: number | null;
		verdict?: string | null;
		confidence?: string | null;
		bucket?: string | null;
		reasoning?: string | null;
		tier?: string | null;
		grounding_status?: string | null;
	}

	/**
	 * Map a report-taxonomy bucket to reader-facing language. The monolithic
	 * scorer emits one bucket per evidence (no probe reason-codes anymore).
	 */
	const BUCKET_LABEL: Record<string, string> = {
		semantic_incorrect: 'contradicts the claim',
		reader_hallucination: 'reader hallucination',
		no_evidence: 'no evidence in sentence',
		incomplete_claim: 'incomplete claim',
		hedged_evidence: 'hedged',
		semantic_correct: 'supports the claim',
		placeholder_text: 'placeholder',
		row_error: 'scoring error'
	};

	export function bucketLabel(bucket: string | null | undefined): string {
		if (!bucket) return 'unclassified';
		return BUCKET_LABEL[bucket] ?? bucket.replace(/_/g, ' ');
	}

	/** Verdict → which end of the doubt↔trust ruler the dot sits on. */
	export function verdictTone(v: string | null | undefined): 'doubt' | 'trust' | 'abstain' {
		if (v === 'incorrect') return 'doubt';
		if (v === 'correct') return 'trust';
		return 'abstain';
	}

	/**
	 * Reasoning may carry a leading "[TIER 2 LLM]" marker line; split it off so
	 * we can show the remaining chain-of-thought as the body.
	 */
	export function reasoningParts(reasoning: string | null | undefined): {
		tag: string | null;
		body: string;
	} {
		if (!reasoning) return { tag: null, body: '' };
		const m = reasoning.match(/^\s*(\[[^\]]+\])[ \t]*\n?/);
		if (m) return { tag: m[1], body: reasoning.slice(m[0].length).trimStart() };
		return { tag: null, body: reasoning };
	}
</script>

<script lang="ts">
	import {
		beliefSemantic,
		fmtBelief,
		fmtDelta,
		shortHash,
		verdictDisplay
	} from '$lib/format';

	export interface BeliefPrimitiveProps {
		stmt: {
			stmt_hash: string;
			indra_type: string;
			subject: string;
			object: string;
		};
		our_score: number | null;
		indra_score: number | null;
		/** The statement's per-evidence rows (monolithic single-call rows). */
		evidences?: BeliefEvidence[];
		why_this_one?: string;
		mode?: 'full' | 'compact';
		/** When true, the card is clickable and behaves as <a href={href}>. */
		href?: string;
		/**
		 * Heading level for the biology sentence in full mode. Defaults to
		 * 'h3' (legacy). Call sites should set this based on context:
		 * - 'h1' on /statements/[hash] (the page is *about* the statement)
		 * - 'h2' for a focus card on a dashboard that owns the page-level h1
		 * - 'h3' inside a nested section with its own h2 ancestor
		 * Compact mode never emits a heading (renders `<span>`).
		 */
		level?: 'h1' | 'h2' | 'h3';
	}

	let {
		stmt,
		our_score,
		indra_score,
		evidences = [],
		why_this_one,
		mode = 'full',
		href,
		level = 'h3'
	}: BeliefPrimitiveProps = $props();

	// Sub-heading tracks one level deeper than the main sentence so the document
	// doesn't skip levels (e.g. h1 → h4 on the deep-dive page).
	const subLevel = $derived(level === 'h1' ? 'h2' : level === 'h2' ? 'h3' : 'h4');

	// The biology sentence is rendered structurally as subject · type · object —
	// matching the monolithic export, which carries subject/object as strings.
	const delta = $derived(
		our_score != null && indra_score != null ? our_score - indra_score : null
	);
	const semantic = $derived(beliefSemantic(our_score));

	function beliefPhrase(b: number | null): string {
		if (b == null) return 'unscored';
		if (b >= 0.85) return 'near-certain';
		if (b >= 0.7) return 'confident';
		if (b >= 0.5) return 'moderate';
		if (b >= 0.3) return 'doubtful';
		return 'low';
	}

	const verdictLine = $derived.by(() => {
		if (our_score == null && indra_score == null) {
			return 'Not yet scored. No INDRA prior.';
		}
		if (our_score == null) {
			return `Not yet scored · INDRA prior ${indra_score!.toFixed(2)} (${beliefPhrase(indra_score)}).`;
		}
		const ourP = beliefPhrase(our_score);
		if (indra_score == null) {
			return `We score this ${our_score.toFixed(2)} (${ourP}) · no INDRA prior to compare.`;
		}
		const d = our_score - indra_score;
		let comparison: string;
		if (Math.abs(d) < 0.1) {
			comparison = `matches INDRA's ${indra_score.toFixed(2)} (${beliefPhrase(indra_score)})`;
		} else if (d < -0.4) {
			comparison = `we doubt it strongly · INDRA was ${indra_score.toFixed(2)} (${beliefPhrase(indra_score)})`;
		} else if (d < 0) {
			comparison = `less confident than INDRA's ${indra_score.toFixed(2)} (${beliefPhrase(indra_score)})`;
		} else if (d > 0.4) {
			comparison = `we believe it more strongly than INDRA's ${indra_score.toFixed(2)} (${beliefPhrase(indra_score)})`;
		} else {
			comparison = `more confident than INDRA's ${indra_score.toFixed(2)} (${beliefPhrase(indra_score)})`;
		}
		return `We score this ${our_score.toFixed(2)} (${ourP}) · ${comparison}.`;
	});

	/**
	 * The seven verdict-confidence buckets the scorer can emit. Showing them as
	 * ladder ticks on the score axis surfaces that the score is categorical.
	 */
	const SCORE_BUCKETS = [0.05, 0.2, 0.35, 0.5, 0.65, 0.8, 0.95];

	/** Score-axis geometry: viewBox 0..320 wide, ticks land at x = margin + score * track. */
	const AXIS_W = 320;
	const AXIS_MARGIN = 14;
	const AXIS_TRACK = AXIS_W - 2 * AXIS_MARGIN;
	function tickX(score: number): number {
		return AXIS_MARGIN + Math.max(0, Math.min(1, score)) * AXIS_TRACK;
	}

	/**
	 * Adaptive label anchor for the simple two-marker axis: hug the edge when a
	 * tick sits near the viewBox boundary so centered text doesn't clip.
	 */
	function labelAnchor(tx: number): { anchor: 'middle' | 'start' | 'end'; x: number } {
		const labelHalf = 26;
		if (tx + labelHalf > AXIS_W) return { anchor: 'end', x: tx };
		if (tx - labelHalf < 0) return { anchor: 'start', x: tx };
		return { anchor: 'middle', x: tx };
	}

	// ── Evidence-spectrum "why we doubt" (progressive disclosure) ───────────────
	// Level 1: the spectrum (all evidences as dots on the same doubt↔trust ruler).
	// Level 2: the reason tally (bucket distribution — the categorical "why").
	// Level 3: open an evidence → sentence + verdict/confidence + reasoning + tier.
	// (Monolithic has NO probes and NO per-probe LLM calls — the rationale IS the
	//  single `reasoning` field.)
	const hasSpectrum = $derived(
		mode === 'full' && evidences.some((e) => e.our_score != null)
	);

	let openEv = $state<string | null>(null);
	let showFullReasoning = $state<string | null>(null);

	function toggleEv(hash: string) {
		openEv = openEv === hash ? null : hash;
		showFullReasoning = null;
	}

	const LANE_GAP = 11;
	const DOT_BASE_Y = 14; // top lane baseline; lanes stack downward toward the axis
	/** Greedy vertical stacking: dots whose x is within 13px share a column, so
	 * near-identical scores fan out instead of overplotting. */
	const spectrum = $derived.by(() => {
		const scored = evidences.filter((e) => e.our_score != null);
		const sorted = [...scored].sort((a, b) => a.our_score! - b.our_score!);
		const dots: Array<{ e: BeliefEvidence; x: number; lane: number }> = [];
		for (const e of sorted) {
			const x = tickX(e.our_score!);
			let lane = 0;
			while (dots.some((d) => d.lane === lane && Math.abs(d.x - x) < 13)) lane++;
			dots.push({ e, x, lane });
		}
		const maxLane = dots.reduce((m, d) => Math.max(m, d.lane), 0);
		return { dots, maxLane };
	});
	const specAxisY = $derived(DOT_BASE_Y + spectrum.maxLane * LANE_GAP + 14);
	const specH = $derived(specAxisY + 18);

	/** L2 reason tally, built from each evidence's report-taxonomy bucket. */
	const reasonTally = $derived.by(() => {
		if (!hasSpectrum) return [] as Array<{ bucket: string; label: string; count: number }>;
		const counts = new Map<string, number>();
		for (const e of evidences) {
			const b = e.bucket ?? 'unclassified';
			counts.set(b, (counts.get(b) ?? 0) + 1);
		}
		return [...counts]
			.map(([bucket, count]) => ({ bucket, label: bucketLabel(bucket), count }))
			.sort((a, b) => b.count - a.count);
	});

	/** Clamp a long evidence paragraph to its head; "only text when needed". */
	function clampText(text: string | null, max = 320): string {
		if (!text) return '(no text)';
		return text.length > max ? text.slice(0, max).trimEnd() + '…' : text;
	}

	const REASONING_CLAMP = 280;
	function reasoningIsLong(reasoning: string | null | undefined): boolean {
		return reasoningParts(reasoning).body.length > REASONING_CLAMP;
	}
	function reasoningClamped(reasoning: string | null | undefined): string {
		const body = reasoningParts(reasoning).body;
		return body.length > REASONING_CLAMP ? body.slice(0, REASONING_CLAMP).trimEnd() + '…' : body;
	}
</script>

{#if mode === 'compact'}
	{#snippet compactBody()}
		<span class="b-sentence">{stmt.subject} <span class="b-verb">{stmt.indra_type}</span> {stmt.object}</span>
		<span class="b-num-pair" title={delta == null ? '' : `Δ ${fmtDelta(delta)} (we ${delta < 0 ? 'doubt' : delta > 0 ? 'support' : 'match'} more than INDRA)`}>
			<span class="b-num-pair-label">we</span>
			<span class="b-score-compact b-{semantic}">{fmtBelief(our_score)}</span>
			<span class="b-num-pair-label">indra</span>
			<span class="b-num-mid-compact">{fmtBelief(indra_score)}</span>
		</span>
		{#if evidences.length > 0}
			<span class="b-ev-count" title="number of evidences for this statement">{evidences.length} ev</span>
		{/if}
		<span class="b-hash">{shortHash(stmt.stmt_hash)}</span>
	{/snippet}

	{#if href}
		<a class="b-card b-card-compact" {href}>{@render compactBody()}</a>
	{:else}
		<div class="b-card b-card-compact">{@render compactBody()}</div>
	{/if}
{:else}
	<article class="b-card b-card-full">
		<svelte:element this={level} class="b-sentence b-sentence-full">{stmt.subject} <span class="b-verb">{stmt.indra_type}</span> {stmt.object}</svelte:element>
		<p class="b-verdict-line">{verdictLine}</p>
		<div class="b-meta">
			<span class="b-type">{stmt.indra_type}</span>
			<span class="b-hash">{shortHash(stmt.stmt_hash)}</span>
		</div>

		{#if hasSpectrum}
			<!-- Merged ruler: one 0→1 scale carrying the per-evidence distribution
			     (dots above), our mean (caret), INDRA's prior (hollow), and the 7
			     verdict×confidence buckets. The mean reads as the dots' centroid. -->
			<div
				class="ruler"
				role="img"
				aria-label={`${evidences.length} evidences on a 0-to-1 belief scale from doubt to trust; our mean ${fmtBelief(our_score)}, INDRA prior ${fmtBelief(indra_score)}`}
			>
				<svg viewBox="0 0 {AXIS_W} {specH}" class="ruler-svg" preserveAspectRatio="xMidYMid meet">
					<line x1={AXIS_MARGIN} y1={specAxisY} x2={AXIS_W - AXIS_MARGIN} y2={specAxisY} stroke="var(--ink)" stroke-width="1" />
					<line x1={AXIS_MARGIN} y1={specAxisY - 4} x2={AXIS_MARGIN} y2={specAxisY + 4} stroke="var(--ink-faint)" />
					<line x1={AXIS_W - AXIS_MARGIN} y1={specAxisY - 4} x2={AXIS_W - AXIS_MARGIN} y2={specAxisY + 4} stroke="var(--ink-faint)" />
					<!-- 7 verdict-bucket ticks (categorical rungs the prod scorer can hit) -->
					{#each SCORE_BUCKETS as b}
						<line x1={tickX(b)} y1={specAxisY - 3} x2={tickX(b)} y2={specAxisY + 3} stroke="var(--ink-faint)" stroke-width="0.6" opacity="0.65" />
					{/each}
					<!-- ours: dashed guide from the cluster down to a caret on the line -->
					{#if our_score != null}
						<line x1={tickX(our_score)} y1={DOT_BASE_Y - 6} x2={tickX(our_score)} y2={specAxisY} stroke="var(--accent)" stroke-width="0.7" stroke-dasharray="2 2" opacity="0.55" />
						<path d={`M ${tickX(our_score)} ${specAxisY + 2} l -4.5 7 l 9 0 z`} fill="var(--accent)" />
					{/if}
					<!-- INDRA prior: hollow circle on the line -->
					{#if indra_score != null}
						<circle cx={tickX(indra_score)} cy={specAxisY} r="4.5" fill="var(--paper)" stroke="var(--ink)" stroke-width="1.4" />
					{/if}
					<!-- evidence dots above the line, stacked when scores collide -->
					{#each spectrum.dots as d (d.e.evidence_hash)}
						{@const cy = specAxisY - 11 - d.lane * LANE_GAP}
						<line x1={d.x} y1={cy} x2={d.x} y2={specAxisY - 1} stroke="var(--rule)" stroke-width="0.8" />
						<circle
							class="dot tone-{verdictTone(d.e.verdict)}"
							class:dot-open={openEv === d.e.evidence_hash}
							cx={d.x}
							cy={cy}
							r="4"
							role="button"
							tabindex="0"
							aria-label={`evidence ${shortHash(d.e.evidence_hash)}, score ${d.e.our_score?.toFixed(2)}, ${verdictDisplay(d.e.verdict)} — open`}
							onclick={() => toggleEv(d.e.evidence_hash)}
							onkeydown={(e) => {
								if (e.key === 'Enter' || e.key === ' ') {
									e.preventDefault();
									toggleEv(d.e.evidence_hash);
								}
							}}
						/>
					{/each}
				</svg>
				<div class="ruler-ends">
					<span>doubt</span>
					<span class="ruler-legend">
						<span class="lg lg-ev">●</span> evidence
						<span class="lg lg-mean">▲</span> ours {fmtBelief(our_score)}
						<span class="lg lg-indra">○</span> indra {fmtBelief(indra_score)}
					</span>
					<span>trust</span>
				</div>
				<p class="ruler-note">
					each ● is one evidence on the same 0–1 scale; ours snaps to 7 verdict×confidence buckets, INDRA's prior is continuous
				</p>
			</div>
		{:else if our_score != null || indra_score != null}
			{@const anyNearZero = (our_score != null && our_score <= 0.05) || (indra_score != null && indra_score <= 0.05)}
			{@const anyNearOne = (our_score != null && our_score >= 0.95) || (indra_score != null && indra_score >= 0.95)}
			<div class="b-axis-wrap" role="img" aria-label="belief scale 0 to 1; ours above the axis, INDRA below">
				<svg viewBox="0 0 {AXIS_W} 60" class="b-axis-svg" preserveAspectRatio="xMidYMid meet">
					<!-- axis track -->
					<line x1={AXIS_MARGIN} y1="28" x2={AXIS_W - AXIS_MARGIN} y2="28" stroke="var(--ink)" stroke-width="1"/>
					<!-- endpoint tick marks -->
					<line x1={AXIS_MARGIN} y1="24" x2={AXIS_MARGIN} y2="32" stroke="var(--ink-faint)" stroke-width="1"/>
					<line x1={AXIS_W - AXIS_MARGIN} y1="24" x2={AXIS_W - AXIS_MARGIN} y2="32" stroke="var(--ink-faint)" stroke-width="1"/>
					<!-- 7 verdict-bucket ticks (categorical rungs the prod scorer can hit) -->
					{#each SCORE_BUCKETS as b}
						<line x1={tickX(b)} y1="26" x2={tickX(b)} y2="30" stroke="var(--ink-faint)" stroke-width="0.6" opacity="0.7"/>
					{/each}
					<!-- 0 / 1 scale anchors: hidden when a tick value already says the same thing -->
					{#if !anyNearZero}
						<text x={AXIS_MARGIN} y="56" text-anchor="middle" class="b-axis-endlabel">0</text>
					{/if}
					{#if !anyNearOne}
						<text x={AXIS_W - AXIS_MARGIN} y="56" text-anchor="middle" class="b-axis-endlabel">1</text>
					{/if}
					<!-- INDRA lives below the axis: outlined circle + label at y=44 -->
					{#if indra_score != null}
						{@const tx = tickX(indra_score)}
						{@const la = labelAnchor(tx)}
						<circle cx={tx} cy="28" r="4.5" fill="var(--paper)" stroke="var(--ink)" stroke-width="1.5"/>
						<text x={la.x} y="44" text-anchor={la.anchor} class="b-axis-label b-axis-label-indra">indra {indra_score.toFixed(2)}</text>
					{/if}
					<!-- Ours lives above the axis: filled circle + label at y=14 -->
					{#if our_score != null}
						{@const tx = tickX(our_score)}
						{@const la = labelAnchor(tx)}
						<circle cx={tx} cy="28" r="5" fill="var(--accent)"/>
						<text x={la.x} y="14" text-anchor={la.anchor} class="b-axis-label b-axis-label-ours">ours {our_score.toFixed(2)}</text>
					{/if}
				</svg>
				<p class="b-axis-footnote">
					ours snaps to one of 7 categorical buckets (verdict × confidence); INDRA's prior is continuous
				</p>
			</div>
		{/if}

		{#if hasSpectrum}
			<section class="why">
				<svelte:element this={subLevel} class="why-h">Why we doubt</svelte:element>

				<!-- L2 · reason tally: the categorical "why", sized by count -->
				<ul class="reasons" aria-label="reasons across evidences">
					{#each reasonTally as r (r.bucket)}
						<li class="reason-tok"><span class="reason-n">{r.count}</span> {r.label}</li>
					{/each}
				</ul>

				<!-- L3 · drill: evidence sentence → verdict/confidence → the model's reasoning -->
				<ul class="ev-list">
					{#each evidences as e (e.evidence_hash)}
						{@const open = openEv === e.evidence_hash}
						{@const rparts = reasoningParts(e.reasoning)}
						<li class="ev" class:ev-open={open}>
							<button class="ev-row" aria-expanded={open} onclick={() => toggleEv(e.evidence_hash)}>
								<span class="ev-swatch tone-{verdictTone(e.verdict)}"></span>
								<span class="ev-score">{e.our_score?.toFixed(2) ?? '—'}</span>
								<span class="ev-verdict">{verdictDisplay(e.verdict)}{e.confidence ? ` · ${e.confidence}` : ''}</span>
								<span class="ev-reason">{bucketLabel(e.bucket)}</span>
								<span class="ev-src">[{e.source_api ?? 'no source'}]</span>
								<span class="ev-caret" aria-hidden="true">{open ? '▾' : '▸'}</span>
							</button>
							{#if open}
								<div class="ev-body">
									<p class="ev-sentence">{clampText(e.text)}</p>

									<div class="ev-tags">
										<span class="ev-tag">verdict <b class="tone-fg-{verdictTone(e.verdict)}">{verdictDisplay(e.verdict)}</b></span>
										{#if e.confidence}<span class="ev-tag">confidence <b>{e.confidence}</b></span>{/if}
										{#if e.bucket}<span class="ev-tag">bucket <b>{bucketLabel(e.bucket)}</b></span>{/if}
										{#if e.tier}<span class="ev-tag">tier <b>{e.tier}</b></span>{/if}
										{#if e.grounding_status}<span class="ev-tag">grounding <b>{e.grounding_status}</b></span>{/if}
									</div>

									{#if rparts.body}
										<div class="reasoning">
											<div class="reasoning-meta">
												<span class="reasoning-lbl">reasoned</span>{#if rparts.tag}<span class="reasoning-tier">{rparts.tag}</span>{/if}
											</div>
											<p class="reasoning-body">{showFullReasoning === e.evidence_hash || !reasoningIsLong(e.reasoning) ? rparts.body : reasoningClamped(e.reasoning)}</p>
											{#if reasoningIsLong(e.reasoning)}
												<button
													class="reasoning-toggle"
													onclick={() => (showFullReasoning = showFullReasoning === e.evidence_hash ? null : e.evidence_hash)}
												>{showFullReasoning === e.evidence_hash ? '− less' : '+ full reasoning'}</button>
											{/if}
										</div>
									{:else}
										<p class="reasoning-empty">no reasoning recorded for this evidence</p>
									{/if}
								</div>
							{/if}
						</li>
					{/each}
				</ul>
			</section>
		{/if}

		{#if why_this_one}
			<div class="b-why">why this one: {why_this_one}</div>
		{/if}
	</article>
{/if}

<style>
	.b-card {
		font-family: var(--serif);
	}
	.b-card-full {
		padding: 1.2rem 1.4rem;
		border-left: 3px solid var(--accent);
		background: transparent;
		margin: 0 0 1.6rem;
	}
	.b-card-compact {
		display: grid;
		grid-template-columns: minmax(0, 1fr) auto auto auto;
		gap: 1rem;
		align-items: baseline;
		padding: 0.4rem 0.4rem;
		border-bottom: 1px dotted var(--rule);
		font-size: 0.92rem;
		text-decoration: none;
		color: var(--ink);
	}
	a.b-card-compact:hover {
		background: var(--accent-wash);
	}

	.b-sentence {
		font-family: var(--serif);
		font-style: italic;
		color: var(--ink);
	}
	.b-verb {
		font-style: normal;
		font-family: var(--mono);
		font-size: 0.82em;
		color: var(--ink-muted);
		text-transform: lowercase;
		letter-spacing: 0.02em;
	}
	.b-sentence-full {
		font-size: 1.25rem;
		font-weight: 400;
		margin: 0 0 0.2rem;
		line-height: 1.3;
	}
	.b-verdict-line {
		font-family: var(--serif);
		font-size: 1rem;
		color: var(--ink);
		margin: 0.1rem 0 0.6rem;
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
	.b-type {
		text-transform: lowercase;
		letter-spacing: 0.04em;
	}

	.b-high { color: #2a6f2a; }
	.b-low { color: var(--accent); }
	.b-mid { color: var(--ink); }
	.b-absent { color: var(--ink-faint); }

	/* Score axis — comparison as position */
	.b-axis-wrap {
		margin: 0.6rem 0 1.2rem;
		max-width: 360px;
	}
	.b-axis-svg {
		width: 100%;
		height: auto;
		display: block;
	}
	.b-axis-footnote {
		font-family: var(--serif);
		font-style: italic;
		font-size: 0.78rem;
		color: var(--ink-faint);
		margin: 0.2rem 0 0;
		line-height: 1.4;
	}
	:global(.b-axis-endlabel) {
		font-family: var(--mono);
		font-size: 7px;
		fill: var(--ink-faint);
	}
	:global(.b-axis-label) {
		font-family: var(--mono);
		font-size: 8px;
		font-variant-numeric: tabular-nums;
	}
	:global(.b-axis-label-ours) { fill: var(--accent); font-weight: 500; }
	:global(.b-axis-label-indra) { fill: var(--ink-muted); }

	.b-why {
		margin-top: 1rem;
		padding-top: 0.6rem;
		border-top: 1px dotted var(--rule);
		font-family: var(--mono);
		font-size: 0.72rem;
		color: var(--ink-faint);
	}

	/* Compact-mode cells */
	.b-card-compact .b-sentence {
		font-style: italic;
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
		font-size: 1rem;
	}
	.b-num-pair {
		font-family: var(--mono);
		font-size: 0.82rem;
		font-variant-numeric: tabular-nums;
		text-align: right;
		display: inline-flex;
		gap: 0.3rem;
		align-items: baseline;
	}
	.b-num-pair-label {
		color: var(--ink-faint);
		font-size: 0.7rem;
		text-transform: lowercase;
	}
	.b-score-compact { color: var(--ink); font-weight: 500; }
	.b-num-mid-compact { color: var(--ink-muted); }
	.b-ev-count {
		font-family: var(--mono);
		font-size: 0.72rem;
		color: var(--ink-faint);
	}
	.b-hash {
		font-family: var(--mono);
		font-size: 0.7rem;
		color: var(--ink-faint);
	}

	/* ── Evidence-spectrum "why we doubt" ─────────────────────────────────── */
	.why {
		margin-top: 1.4rem;
	}
	.why-h {
		font-family: var(--mono);
		font-size: 0.72rem;
		font-weight: 600;
		letter-spacing: 0.06em;
		text-transform: lowercase;
		color: var(--ink-muted);
		margin: 0 0 0.5rem;
	}

	/* Merged ruler: distribution + mean + prior + buckets on one scale */
	.ruler {
		max-width: 30rem;
		margin: 1rem 0 1.1rem;
	}
	.ruler-svg {
		display: block;
		width: 100%;
		height: auto;
		overflow: visible;
	}
	.dot {
		cursor: pointer;
		transition:
			r 120ms ease-out,
			fill 120ms ease-out;
	}
	.dot:hover {
		r: 5;
	}
	.dot:focus-visible {
		outline: 2px solid var(--accent);
		outline-offset: 1px;
	}
	.dot.tone-doubt {
		fill: var(--accent);
	}
	.dot.tone-trust {
		fill: var(--ink);
	}
	.dot.tone-abstain {
		fill: var(--paper);
		stroke: var(--ink-faint);
		stroke-width: 1.2;
	}
	.dot.dot-open {
		stroke: var(--ink);
		stroke-width: 1.6;
	}
	.ruler-ends {
		display: flex;
		justify-content: space-between;
		align-items: baseline;
		gap: 0.75rem;
		font-family: var(--mono);
		font-size: 0.64rem;
		letter-spacing: 0.04em;
		color: var(--ink-faint);
		margin-top: 0.15rem;
	}
	.ruler-legend {
		color: var(--ink-muted);
		display: inline-flex;
		gap: 0.55rem;
		flex-wrap: wrap;
		justify-content: center;
	}
	.lg {
		font-size: 0.72rem;
		line-height: 1;
	}
	.lg-ev {
		color: var(--accent);
	}
	.lg-mean {
		color: var(--accent);
	}
	.lg-indra {
		color: var(--ink);
	}
	.ruler-note {
		font-family: var(--serif);
		font-style: italic;
		font-size: 0.72rem;
		line-height: 1.4;
		color: var(--ink-faint);
		margin: 0.4rem 0 0;
		max-width: 30rem;
	}

	/* L2 · reason tally */
	.reasons {
		list-style: none;
		margin: 0 0 1rem;
		padding: 0;
		display: flex;
		flex-wrap: wrap;
		gap: 0.3rem 0.5rem;
	}
	.reason-tok {
		font-family: var(--serif);
		font-size: 0.82rem;
		color: var(--ink-muted);
		display: inline-flex;
		align-items: baseline;
		gap: 0.3rem;
	}
	.reason-tok + .reason-tok {
		padding-left: 0.5rem;
		border-left: 1px solid var(--rule);
	}
	.reason-n {
		font-family: var(--mono);
		font-size: 0.78rem;
		font-weight: 600;
		color: var(--ink);
	}

	/* L3 · evidence list */
	.ev-list {
		list-style: none;
		margin: 0;
		padding: 0;
		border-top: 1px solid var(--rule);
	}
	.ev {
		border-bottom: 1px solid var(--rule);
	}
	.ev-row {
		display: grid;
		grid-template-columns: auto 3ch minmax(8rem, auto) 1fr auto auto;
		align-items: baseline;
		gap: 0.6rem;
		width: 100%;
		padding: 0.42rem 0.1rem;
		background: none;
		border: none;
		text-align: left;
		cursor: pointer;
		font-family: var(--serif);
		color: var(--ink);
		transition: background 120ms ease-out;
	}
	.ev-row:hover {
		background: var(--accent-wash);
	}
	.ev-swatch {
		width: 0.6rem;
		height: 0.6rem;
		border-radius: 50%;
		align-self: center;
	}
	.ev-swatch.tone-doubt {
		background: var(--accent);
	}
	.ev-swatch.tone-trust {
		background: var(--ink);
	}
	.ev-swatch.tone-abstain {
		background: var(--paper);
		box-shadow: inset 0 0 0 1.2px var(--ink-faint);
	}
	.ev-score {
		font-family: var(--mono);
		font-size: 0.82rem;
		font-weight: 600;
		font-variant-numeric: tabular-nums;
	}
	.ev-verdict {
		font-size: 0.82rem;
		color: var(--ink-muted);
	}
	.ev-reason {
		font-size: 0.82rem;
		color: var(--ink);
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}
	.ev-src {
		font-family: var(--mono);
		font-size: 0.66rem;
		color: var(--ink-faint);
	}
	.ev-caret {
		font-size: 0.7rem;
		color: var(--ink-faint);
	}

	/* L3 · expanded body */
	.ev-body {
		padding: 0.2rem 0.1rem 0.9rem 1.2rem;
		border-left: 2px solid var(--rule);
		margin-left: 0.18rem;
	}
	.ev-sentence {
		font-family: var(--serif);
		font-style: italic;
		font-size: 0.92rem;
		line-height: 1.5;
		color: var(--ink-muted);
		margin: 0.2rem 0 0.7rem;
	}

	/* verdict / confidence / bucket / tier / grounding tags */
	.ev-tags {
		display: flex;
		flex-wrap: wrap;
		gap: 0.4rem 0.7rem;
		font-family: var(--mono);
		font-size: 0.7rem;
		color: var(--ink-faint);
		margin-bottom: 0.6rem;
	}
	.ev-tag b {
		color: var(--ink);
		font-weight: 600;
		margin-left: 0.2rem;
	}
	.tone-fg-doubt { color: var(--accent); }
	.tone-fg-trust { color: var(--ink); }
	.tone-fg-abstain { color: var(--ink-faint); }

	/* The single chain-of-thought reasoning field (replaces the probe call drill) */
	.reasoning {
		margin: 0.2rem 0 0.2rem;
		padding: 0.55rem 0.7rem;
		background: var(--accent-wash);
		border-left: 2px solid var(--accent);
		font-family: var(--mono);
	}
	.reasoning-meta {
		display: flex;
		gap: 0.5rem;
		align-items: baseline;
		font-size: 0.64rem;
		color: var(--ink-faint);
		margin-bottom: 0.35rem;
	}
	.reasoning-lbl {
		text-transform: lowercase;
		letter-spacing: 0.02em;
	}
	.reasoning-tier {
		color: var(--accent);
		font-weight: 600;
		letter-spacing: 0.02em;
	}
	.reasoning-body {
		font-size: 0.76rem;
		line-height: 1.55;
		color: var(--ink);
		margin: 0;
		white-space: pre-wrap;
		word-break: break-word;
	}
	.reasoning-toggle {
		margin-top: 0.4rem;
		padding: 0;
		background: none;
		border: none;
		font-family: var(--mono);
		font-size: 0.66rem;
		color: var(--ink-faint);
		cursor: pointer;
		text-decoration: underline;
		text-underline-offset: 2px;
	}
	.reasoning-empty {
		font-family: var(--serif);
		font-style: italic;
		font-size: 0.78rem;
		color: var(--ink-faint);
		margin: 0.2rem 0 0;
	}
</style>
