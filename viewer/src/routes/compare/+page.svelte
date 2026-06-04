<script lang="ts">
	import { goto } from '$app/navigation';
	import { page } from '$app/stores';
	import type { PageData } from './$types';

	let { data }: { data: PageData } = $props();
	const runs = $derived(data.runs);
	const anatomy = $derived(data.anatomy);
	const strat = $derived(data.stratification);
	const cohort = $derived(data.cohort);
	const sbs = $derived(data.sideBySide);

	const CELL_LABEL: Record<string, string> = {
		acbc: 'both correct',
		acbi: 'A correct · B incorrect',
		aibc: 'A incorrect · B correct',
		aibi: 'both incorrect'
	};
	const AXIS_LABEL: Record<string, string> = {
		source_api: 'reader / source',
		stmt_type: 'statement type',
		grounding_status: 'grounding',
		bucket_a: 'A bucket',
		bucket_b: 'B bucket'
	};

	function shortRun(id: string): string {
		return id.slice(0, 8);
	}
	function pct(x: number): string {
		return `${(x * 100).toFixed(1)}%`;
	}
	function vsym(v: string): string {
		return v === 'correct' ? '✓' : v === 'incorrect' ? '✗' : '·';
	}

	// ── URL state helpers ───────────────────────────────────────────
	function nav(updates: Record<string, string | null>) {
		const p = new URLSearchParams($page.url.searchParams);
		for (const [k, v] of Object.entries(updates)) {
			if (v == null) p.delete(k);
			else p.set(k, v);
		}
		goto(`?${p.toString()}`, { keepFocus: true, noScroll: true });
	}
	function setRun(which: 'a' | 'b', run_id: string) {
		// Changing a run resets the deeper drill state.
		nav({ [which]: run_id, cell: null, axis: null, val: null, ev: null });
	}
	function toggleSemantic() {
		nav({ sem: data.semanticOnly ? null : '1', ev: null });
	}
	function selectCell(cell: string) {
		nav({ cell, axis: null, val: null, ev: null });
	}
	function setAxis(axis: string) {
		nav({ axis, val: null });
	}
	function filterAxis(val: string) {
		nav({ val });
	}
	function clearAxisFilter() {
		nav({ val: null });
	}
	function openEvidence(stmt_hash: string, evidence_hash: string) {
		nav({ ev: `${stmt_hash}.${evidence_hash}` });
	}
	function closeEvidence() {
		nav({ ev: null });
	}
	function backToAnatomy() {
		nav({ cell: null, axis: null, val: null, ev: null });
	}
	function toggleGoldMode() {
		nav({ mode: data.goldMode ? null : 'gold' });
	}
	function setGoldFilter(g: string | null) {
		nav({ gold: data.goldFilter === g ? null : g });
	}

	const haveTwo = $derived(runs.length >= 2);
	const sameRun = $derived(data.selectedA != null && data.selectedA === data.selectedB);

	// ── gold (human curation) ────────────────────────────────────────
	const gold = $derived(anatomy?.gold ?? null);
	const goldOn = $derived(!!gold?.present);
	type GoldTally = { n_covered: number; a_right: number; b_right: number; both_right: number; neither_right: number };
	function goldCell(c: string): GoldTally | null {
		if (!gold?.present) return null;
		return (gold.cells as Record<string, GoldTally>)[c] ?? null;
	}
	const GOLD_FILTER_LABEL: Record<string, string> = {
		match: 'both match gold',
		fp: 'over-accept (FP)',
		fn: 'over-reject (FN)',
		disagree: 'contradict gold'
	};
	function goldTone(gv: string | null, mv: string): string {
		if (!gv) return 'none';
		return gv === mv ? 'match' : 'miss';
	}

	// Matrix cell accessor (which matrix: full or semantic)
	const matrix = $derived(
		anatomy ? (data.semanticOnly ? anatomy.semantic : anatomy.matrix) : null
	);
	const matrixN = $derived(
		anatomy ? (data.semanticOnly ? anatomy.semantic.n : anatomy.n_both_scored) : 0
	);
	const matrixAgreePct = $derived(
		anatomy ? (data.semanticOnly ? anatomy.semantic.agree_pct : anatomy.agree_pct) : 0
	);

	function cellVal(c: string): number {
		if (!matrix) return 0;
		const m = matrix as Record<string, number>;
		return m[
			c === 'acbc'
				? 'a_correct_b_correct'
				: c === 'acbi'
					? 'a_correct_b_incorrect'
					: c === 'aibc'
						? 'a_incorrect_b_correct'
						: 'a_incorrect_b_incorrect'
		];
	}
	// max disagreement cell drives the heat emphasis
	const maxCell = $derived(
		matrix ? Math.max(cellVal('acbi'), cellVal('aibc'), 1) : 1
	);
	function heat(c: string): number {
		if (c === 'acbc' || c === 'aibi') return 0; // agreement cells stay neutral
		return Math.min(1, cellVal(c) / maxCell);
	}

	const aModel = $derived(anatomy?.run_a.model ?? 'A');
	const bModel = $derived(anatomy?.run_b.model ?? 'B');
</script>

<svelte:head>
	<title>INDRA Belief — Compare runs</title>
</svelte:head>

<!-- gold tally shown under a matrix cell when gold mode is on + the cell is curated.
     Raw counts only (coverage is single-digit on disagreement cells — never a rate). -->
{#snippet goldSub(c: string)}
	{#if data.goldMode && goldOn}
		{@const t = goldCell(c)}
		{#if t && t.n_covered > 0}
			<span class="cell-gold mono">
				{#if c === 'acbc' || c === 'aibi'}
					gold: {t.both_right} both right{#if t.neither_right} · {t.neither_right} both wrong{/if} (n={t.n_covered})
				{:else}
					gold: A {t.a_right} / B {t.b_right} (n={t.n_covered})
				{/if}
			</span>
		{:else}
			<span class="cell-gold none">no gold</span>
		{/if}
	{/if}
{/snippet}

<main id="main">
	<h1>compare two runs</h1>
	<p class="lede">
		Two scoring passes over the shared corpus, joined per-evidence on content hash.
		The disagreement is read directionally: where does <span class="mA">A</span> trust
		evidence that <span class="mB">B</span> rejects? Start with the anatomy, then dig into
		any cell — strata, cohort, then both models' full reasoning side by side.
	</p>

	{#if !haveTwo}
		<section class="empty">
			<p class="empty-h">need two exported runs to compare</p>
			<p class="hint">score a corpus and export it to populate this view</p>
		</section>
	{:else}
		<!-- run pickers -->
		<section class="pickers" aria-label="run selectors">
			<div class="picker">
				<label for="run-a">run A</label>
				<select id="run-a" value={data.selectedA ?? ''} onchange={(e) => setRun('a', (e.currentTarget as HTMLSelectElement).value)}>
					{#each runs as r}
						<option value={r.run_id}>{r.model} · {shortRun(r.run_id)} · {r.n_statements} stmts</option>
					{/each}
				</select>
			</div>
			<span class="vs">vs</span>
			<div class="picker">
				<label for="run-b">run B</label>
				<select id="run-b" value={data.selectedB ?? ''} onchange={(e) => setRun('b', (e.currentTarget as HTMLSelectElement).value)}>
					{#each runs as r}
						<option value={r.run_id}>{r.model} · {shortRun(r.run_id)} · {r.n_statements} stmts</option>
					{/each}
				</select>
			</div>
		</section>

		{#if sameRun}
			<section class="empty"><p class="empty-h">pick two different runs</p></section>
		{:else if anatomy}
			{@const an = anatomy}

			<!-- breadcrumb -->
			<nav class="crumbs" aria-label="drill path">
				<button class="crumb" class:active={!data.cell} onclick={backToAnatomy}>anatomy</button>
				{#if data.cell}
					<span class="crumb-sep">›</span>
					<button class="crumb" class:active={data.cell && !sbs} onclick={() => selectCell(data.cell!)}>
						{CELL_LABEL[data.cell]}{#if data.axis && data.axisValue} · {data.axisValue}{/if}
					</button>
				{/if}
				{#if sbs}
					<span class="crumb-sep">›</span>
					<span class="crumb active">evidence</span>
				{/if}
			</nav>

			<!-- ════════════ L0: ANATOMY ════════════ -->
			{#if !data.cell && !sbs}
				<section class="anatomy">
					<header class="cmp-head">
						<div class="cmp-vs">
							<span class="cmp-model mA">{an.run_a.model}</span>
							<span class="cmp-sep">vs</span>
							<span class="cmp-model mB">{an.run_b.model}</span>
						</div>
						<div class="toggles">
							<label class="sem-toggle">
								<input type="checkbox" checked={data.semanticOnly} onchange={toggleSemantic} />
								semantic only <span class="hint">(strip reader-artifact buckets)</span>
							</label>
							{#if goldOn}
								<label class="sem-toggle gold-toggle">
									<input type="checkbox" checked={data.goldMode} onchange={toggleGoldMode} />
									show human gold <span class="hint">(INDRA curations)</span>
								</label>
							{/if}
						</div>
					</header>

					{#if goldOn && gold}
						<div class="gold-coverage" class:active={data.goldMode}>
							<span class="gc-key mono">human-curation gold:</span>
							<span class="gc-val mono">{gold.n_evaluable.toLocaleString()}</span>
							of {gold.n_both_scored.toLocaleString()} evidences curated
							<span class="gc-pct mono">({pct(gold.n_evaluable / Math.max(1, gold.n_both_scored))})</span>
							<span class="gc-caveat">— spot-check only, not corpus-wide</span>
						</div>
					{/if}

					<!-- headline scalars -->
					<div class="scalars">
						<div class="scalar">
							<div class="s-num mono">{matrixN.toLocaleString()}</div>
							<div class="s-lbl">evidences compared{#if data.semanticOnly} (semantic){/if}</div>
						</div>
						<div class="scalar">
							<div class="s-num mono">{pct(matrixAgreePct)}</div>
							<div class="s-lbl">agreement</div>
						</div>
						<div class="scalar">
							<div class="s-num mono" class:skew={cellVal('acbi') - cellVal('aibc') !== 0}>
								{cellVal('acbi') - cellVal('aibc') > 0 ? '+' : ''}{(cellVal('acbi') - cellVal('aibc')).toLocaleString()}
							</div>
							<div class="s-lbl">net leniency · A says ✓ where B says ✗</div>
						</div>
						{#if !data.semanticOnly}
							<div class="scalar">
								<div class="s-num mono">{an.a_only_none.toLocaleString()} / {an.b_only_none.toLocaleString()}</div>
								<div class="s-lbl">parse-fails · A / B</div>
							</div>
						{/if}
						{#if data.goldMode && gold?.present}
							<div class="scalar gold-scalar">
								<div class="s-num mono">
									<span class="mA">{pct(gold.a_accuracy.right / Math.max(1, gold.a_accuracy.n))}</span>
									<span class="sep">/</span>
									<span class="mB">{pct(gold.b_accuracy.right / Math.max(1, gold.b_accuracy.n))}</span>
								</div>
								<div class="s-lbl">accuracy vs gold · A / B <span class="hint">(n={gold.n_evaluable})</span></div>
							</div>
						{/if}
					</div>

					<!-- confusion matrix -->
					<div class="matrix-wrap">
						<table class="matrix">
							<thead>
								<tr>
									<th class="corner"></th>
									<th colspan="2" class="axis-b">{bModel} →</th>
								</tr>
								<tr>
									<th class="corner axis-a">{aModel} ↓</th>
									<th>correct</th>
									<th>incorrect</th>
								</tr>
							</thead>
							<tbody>
								<tr>
									<th>correct</th>
									<td><button class="cell agree" onclick={() => selectCell('acbc')}>
										<span class="cell-n mono">{cellVal('acbc').toLocaleString()}</span>
										<span class="cell-tag">both ✓</span>
										{@render goldSub('acbc')}
									</button></td>
									<td><button class="cell disagree" style="--heat:{heat('acbi')}" onclick={() => selectCell('acbi')}>
										<span class="cell-n mono">{cellVal('acbi').toLocaleString()}</span>
										<span class="cell-tag">A✓ B✗</span>
										{@render goldSub('acbi')}
									</button></td>
								</tr>
								<tr>
									<th>incorrect</th>
									<td><button class="cell disagree" style="--heat:{heat('aibc')}" onclick={() => selectCell('aibc')}>
										<span class="cell-n mono">{cellVal('aibc').toLocaleString()}</span>
										<span class="cell-tag">A✗ B✓</span>
										{@render goldSub('aibc')}
									</button></td>
									<td><button class="cell agree" onclick={() => selectCell('aibi')}>
										<span class="cell-n mono">{cellVal('aibi').toLocaleString()}</span>
										<span class="cell-tag">both ✗</span>
										{@render goldSub('aibi')}
									</button></td>
								</tr>
							</tbody>
						</table>
						<p class="matrix-hint hint">
							diagonal = agreement · off-diagonal = disagreement (shaded by size).
							click any cell to dig in.
							{#if data.semanticOnly}
								artifact rows (reader_hallucination, no_evidence, …) are excluded.
							{/if}
						</p>
					</div>
				</section>

			<!-- ════════════ L1 + L2: CELL DRILL ════════════ -->
			{:else if data.cell && !sbs}
				<section class="drill">
					<header class="drill-head">
						<h2>
							<span class="cell-badge {data.cell}">{CELL_LABEL[data.cell]}</span>
							<span class="mono drill-total">{(cohort?.total ?? 0).toLocaleString()} evidences</span>
							{#if data.semanticOnly}<span class="sem-pill">semantic only</span>{/if}
							{#if cohort && goldOn}<span class="sem-pill gold-pill">{cohort.gold_covered} curated</span>{/if}
						</h2>
						{#if goldOn}
							<div class="gold-filters" aria-label="gold filters">
								<span class="gf-label mono">vs gold:</span>
								{#each ['disagree', 'fp', 'fn', 'match'] as gf}
									<button class="gf-chip" class:active={data.goldFilter === gf} onclick={() => setGoldFilter(gf)}>{GOLD_FILTER_LABEL[gf]}</button>
								{/each}
								{#if data.goldFilter}<button class="link gf-clear" onclick={() => setGoldFilter(null)}>clear</button>{/if}
							</div>
						{/if}
					</header>

					<!-- L1: stratification -->
					{#if strat}
						<div class="strat">
							<div class="strat-axes" role="tablist" aria-label="stratify by">
								{#each ['source_api', 'stmt_type', 'grounding_status', 'bucket_a', 'bucket_b'] as ax}
									<button
										class="ax-tab"
										class:active={(data.axis ?? 'source_api') === ax}
										onclick={() => setAxis(ax)}
									>{AXIS_LABEL[ax]}</button>
								{/each}
							</div>
							<div class="strat-bars">
								{#each strat.rows as row}
									{@const isFilter = data.axisValue === row.value}
									<button
										class="strat-row"
										class:filtered={isFilter}
										onclick={() => (isFilter ? clearAxisFilter() : filterAxis(row.value))}
									>
										<span class="strat-val">{row.value}</span>
										<span class="strat-bar-track">
											<span class="strat-bar-fill" style="width:{row.pct * 100}%"></span>
										</span>
										<span class="strat-n mono">{row.n.toLocaleString()}</span>
										<span class="strat-pct mono">{pct(row.pct)}</span>
									</button>
								{/each}
							</div>
							{#if data.axisValue}
								<p class="hint filter-note">
									filtered to <strong>{data.axisValue}</strong> ·
									<button class="link" onclick={clearAxisFilter}>clear</button>
								</p>
							{/if}
						</div>
					{/if}

					<!-- L2: cohort table -->
					{#if cohort}
						<table class="cohort">
							<thead>
								<tr>
									<th class="c-stmt">statement</th>
									<th class="c-ev">evidence</th>
									<th class="c-src">src</th>
									<th class="c-v">A</th>
									<th class="c-v">B</th>
									{#if goldOn}<th class="c-v c-gold">gold</th>{/if}
									<th class="c-conf">conf</th>
								</tr>
							</thead>
							<tbody>
								{#each cohort.rows as r}
									<tr
										onclick={() => openEvidence(r.stmt_hash, r.evidence_hash)}
										class="cohort-row"
										class:gold-fp={r.gold_verdict === 'incorrect' && (r.a_verdict === 'correct' || r.b_verdict === 'correct')}
										class:gold-fn={r.gold_verdict === 'correct' && (r.a_verdict === 'incorrect' || r.b_verdict === 'incorrect')}
									>
										<td class="c-stmt">
											<span class="agent">{r.subject}</span>
											<span class="rel">[{r.stmt_type}]</span>
											<span class="agent">{r.object}</span>
										</td>
										<td class="c-ev">{r.evidence_text ?? '—'}</td>
										<td class="c-src mono">{r.source_api ?? '—'}</td>
										<td class="c-v"><span class="v {r.a_verdict}" class:gmiss={r.gold_verdict && r.a_verdict !== r.gold_verdict}>{vsym(r.a_verdict)}</span></td>
										<td class="c-v"><span class="v {r.b_verdict}" class:gmiss={r.gold_verdict && r.b_verdict !== r.gold_verdict}>{vsym(r.b_verdict)}</span></td>
										{#if goldOn}
											<td class="c-v c-gold">
												{#if r.gold_verdict}
													<span class="v gold {r.gold_verdict}" title={r.gold_tags.join(', ')}>{vsym(r.gold_verdict)}</span>
												{:else}<span class="gnone">·</span>{/if}
											</td>
										{/if}
										<td class="c-conf mono">{(r.a_confidence ?? '?').slice(0, 1)}/{(r.b_confidence ?? '?').slice(0, 1)}</td>
									</tr>
								{/each}
							</tbody>
						</table>
						{#if cohort.total > cohort.rows.length}
							<p class="hint">showing top {cohort.rows.length} of {cohort.total.toLocaleString()}{#if data.goldFilter} matching <strong>{GOLD_FILTER_LABEL[data.goldFilter]}</strong>{:else} by joint confidence (sharpest disagreements first){/if}</p>
						{/if}
					{/if}
				</section>

			<!-- ════════════ L3: SIDE-BY-SIDE REASONING ════════════ -->
			{:else if sbs}
				<section class="sbs">
					<header class="sbs-claim">
						<div class="claim-line">
							<span class="agent">{sbs.subject}</span>
							<span class="rel">[{sbs.stmt_type}]</span>
							<span class="agent">{sbs.object}</span>
						</div>
						<div class="claim-meta mono">
							{sbs.source_api ?? '—'}{#if sbs.pmid} · PMID {sbs.pmid}{/if}
							{#if sbs.rasmachine_belief != null} · rasmachine {sbs.rasmachine_belief.toFixed(3)}{/if}
						</div>
						<blockquote class="evidence-text">{sbs.evidence_text ?? '(no evidence text)'}</blockquote>
						<button class="link back" onclick={closeEvidence}>← back to cohort</button>
					</header>

					<div class="traces" class:with-gold={sbs.gold}>
						<article class="trace tA" class:gmiss={sbs.gold && sbs.a.verdict !== sbs.gold.verdict}>
							<header class="trace-head">
								<span class="trace-model mA">{sbs.run_a.model}</span>
								<span class="trace-verdict v {sbs.a.verdict}">{vsym(sbs.a.verdict)} {sbs.a.verdict}</span>
								<span class="trace-conf mono">{sbs.a.confidence ?? '?'}{#if sbs.a.bucket} · {sbs.a.bucket}{/if}</span>
								{#if sbs.gold}<span class="trace-gflag {sbs.a.verdict === sbs.gold.verdict ? 'ok' : 'bad'}">{sbs.a.verdict === sbs.gold.verdict ? 'matches gold' : 'vs gold'}</span>{/if}
							</header>
							<pre class="reasoning">{sbs.a.reasoning ?? '(no reasoning captured)'}</pre>
						</article>
						<article class="trace tB" class:gmiss={sbs.gold && sbs.b.verdict !== sbs.gold.verdict}>
							<header class="trace-head">
								<span class="trace-model mB">{sbs.run_b.model}</span>
								<span class="trace-verdict v {sbs.b.verdict}">{vsym(sbs.b.verdict)} {sbs.b.verdict}</span>
								<span class="trace-conf mono">{sbs.b.confidence ?? '?'}{#if sbs.b.bucket} · {sbs.b.bucket}{/if}</span>
								{#if sbs.gold}<span class="trace-gflag {sbs.b.verdict === sbs.gold.verdict ? 'ok' : 'bad'}">{sbs.b.verdict === sbs.gold.verdict ? 'matches gold' : 'vs gold'}</span>{/if}
							</header>
							<pre class="reasoning">{sbs.b.reasoning ?? '(no reasoning captured)'}</pre>
						</article>
						{#if sbs.gold}
							{@const g = sbs.gold}
							<article class="trace tGold">
								<header class="trace-head">
									<span class="trace-model gold-name">human curation</span>
									<span class="trace-verdict v gold {g.verdict}">{vsym(g.verdict)} {g.verdict}</span>
									<span class="trace-conf mono">INDRA · {g.n} curation{g.n === 1 ? '' : 's'}</span>
								</header>
								<div class="gold-body">
									<div class="gold-field">
										<span class="gf-k">tags</span>
										<span class="gf-chips">
											{#each g.tags as t}<span class="tag-chip" class:tag-correct={t === 'correct'}>{t}</span>{/each}
										</span>
									</div>
									<div class="gold-field">
										<span class="gf-k">curators</span>
										<span class="gf-v">{g.curators.join(', ') || '—'}</span>
									</div>
									{#if g.notes.length}
										<div class="gold-field">
											<span class="gf-k">notes</span>
											<span class="gf-v gold-notes">{g.notes.join(' · ')}</span>
										</div>
									{/if}
									<p class="gold-rule">any-incorrect-wins over {g.n} curator{g.n === 1 ? '' : 's'} — one objection flips to incorrect</p>
								</div>
							</article>
						{/if}
					</div>
					{#if !sbs.gold}
						<p class="hint no-gold-note">no human curation for this evidence — adjudication is model-vs-model only here</p>
					{/if}
				</section>
			{/if}
		{/if}
	{/if}
</main>

<style>
	:global(:root) {
		--ink: #1a1a1a;
		--ink-muted: #6a6a6a;
		--ink-faint: #8a8a8a;
		--paper: #fdfcf8;
		--rule: #e6e2d6;
		--accent: #7d2a1a;
		--accent-wash: rgba(125, 42, 26, 0.06);
		--a-hue: #1d4e6f;
		--a-wash: rgba(29, 78, 111, 0.07);
		--b-hue: #6b4a16;
		--b-wash: rgba(107, 74, 22, 0.07);
		--gold-hue: #5a4a86;
		--gold-wash: rgba(90, 74, 134, 0.08);
		--ok-green: #2a6f2a;
		--mono: ui-monospace, 'SF Mono', 'JetBrains Mono', Menlo, monospace;
		--serif: 'Iowan Old Style', 'Source Serif Pro', Georgia, serif;
		--sans: -apple-system, BlinkMacSystemFont, 'Inter', sans-serif;
	}
	main {
		max-width: 1100px;
		margin: 0 auto;
		padding: 2rem 1.5rem 5rem;
		font-family: var(--serif);
		color: var(--ink);
	}
	h1 {
		font-weight: 400;
		font-size: 1.6rem;
		margin: 0 0 0.4rem;
	}
	.lede {
		color: var(--ink-muted);
		max-width: 66ch;
		margin: 0 0 1.6rem;
		line-height: 1.5;
	}
	.mA {
		color: var(--a-hue);
		font-weight: 600;
	}
	.mB {
		color: var(--b-hue);
		font-weight: 600;
	}
	.mono {
		font-family: var(--mono);
		font-variant-numeric: tabular-nums;
	}
	.hint {
		color: var(--ink-faint);
		font-style: italic;
		font-size: 0.9em;
	}
	.link {
		background: none;
		border: none;
		color: var(--accent);
		cursor: pointer;
		font: inherit;
		text-decoration: underline;
		padding: 0;
	}
	.empty {
		margin: 3rem auto;
		max-width: 60ch;
		border-left: 3px solid var(--rule);
		padding: 1rem 1.2rem;
	}
	.empty-h {
		font-size: 1.15rem;
		margin: 0 0 0.3rem;
	}

	/* pickers */
	.pickers {
		display: flex;
		flex-wrap: wrap;
		align-items: flex-end;
		gap: 0.8rem 1.2rem;
		margin: 0 0 1.2rem;
		padding-bottom: 1rem;
		border-bottom: 1px solid var(--rule);
	}
	.picker {
		display: flex;
		flex-direction: column;
		gap: 0.25rem;
	}
	.picker label {
		font-family: var(--mono);
		font-size: 0.68rem;
		letter-spacing: 0.06em;
		color: var(--ink-faint);
	}
	.picker select {
		font-family: var(--mono);
		font-size: 0.8rem;
		background: var(--paper);
		border: 1px solid var(--rule);
		padding: 0.4rem 0.5rem;
		min-width: 21rem;
	}
	.vs {
		font-family: var(--mono);
		font-size: 0.78rem;
		color: var(--ink-faint);
		padding-bottom: 0.5rem;
	}

	/* breadcrumb */
	.crumbs {
		display: flex;
		align-items: center;
		gap: 0.5rem;
		margin: 0 0 1.2rem;
		font-size: 0.82rem;
	}
	.crumb {
		background: none;
		border: none;
		font: inherit;
		font-family: var(--mono);
		font-size: 0.76rem;
		color: var(--ink-muted);
		cursor: pointer;
		padding: 0.1rem 0;
	}
	.crumb.active {
		color: var(--ink);
		font-weight: 600;
	}
	.crumb:hover:not(.active) {
		color: var(--accent);
	}
	.crumb-sep {
		color: var(--ink-faint);
	}

	/* anatomy header */
	.cmp-head {
		display: flex;
		flex-wrap: wrap;
		justify-content: space-between;
		align-items: baseline;
		gap: 0.6rem 1.2rem;
		margin: 0 0 1.2rem;
	}
	.cmp-vs {
		display: flex;
		align-items: baseline;
		gap: 0.5rem;
	}
	.cmp-model {
		font-size: 1.2rem;
	}
	.cmp-sep {
		color: var(--ink-faint);
		font-style: italic;
	}
	.sem-toggle {
		font-size: 0.82rem;
		color: var(--ink-muted);
		display: flex;
		align-items: center;
		gap: 0.4rem;
		cursor: pointer;
	}

	/* scalars */
	.scalars {
		display: flex;
		flex-wrap: wrap;
		gap: 1.6rem 2.4rem;
		margin: 0 0 1.8rem;
		padding: 1rem 0;
		border-top: 1px solid var(--rule);
		border-bottom: 1px solid var(--rule);
	}
	.scalar {
		min-width: 7rem;
	}
	.s-num {
		font-size: 1.5rem;
		line-height: 1;
	}
	.s-num.skew {
		color: var(--accent);
	}
	.s-lbl {
		font-size: 0.72rem;
		color: var(--ink-muted);
		margin-top: 0.3rem;
		max-width: 16ch;
	}

	/* confusion matrix */
	.matrix-wrap {
		margin: 0 0 1rem;
	}
	table.matrix {
		border-collapse: collapse;
	}
	table.matrix th {
		font-family: var(--mono);
		font-size: 0.72rem;
		font-weight: 500;
		color: var(--ink-muted);
		padding: 0.4rem 0.7rem;
		text-align: center;
	}
	table.matrix th.axis-b {
		color: var(--b-hue);
		border-bottom: 1px solid var(--b-wash);
	}
	table.matrix th.axis-a {
		color: var(--a-hue);
		text-align: right;
		vertical-align: middle;
	}
	table.matrix td {
		padding: 0;
	}
	.cell {
		display: flex;
		flex-direction: column;
		align-items: center;
		justify-content: center;
		gap: 0.2rem;
		width: 9.5rem;
		height: 5rem;
		border: 1px solid var(--rule);
		background: var(--paper);
		cursor: pointer;
		font: inherit;
		transition: background 0.12s;
	}
	.cell:hover {
		border-color: var(--ink);
	}
	.cell.agree {
		background: color-mix(in srgb, var(--ok-green) 6%, var(--paper));
	}
	.cell.disagree {
		background: color-mix(in srgb, var(--accent) calc(var(--heat, 0) * 38%), var(--paper));
	}
	.cell-n {
		font-size: 1.35rem;
	}
	.cell-tag {
		font-family: var(--mono);
		font-size: 0.66rem;
		color: var(--ink-muted);
	}
	.cell.disagree .cell-tag {
		color: var(--accent);
	}
	.matrix-hint {
		margin: 0.7rem 0 0;
		max-width: 60ch;
	}

	/* drill header */
	.drill-head h2 {
		display: flex;
		align-items: center;
		gap: 0.7rem;
		font-weight: 400;
		font-size: 1.15rem;
		margin: 0 0 1.2rem;
	}
	.cell-badge {
		font-family: var(--mono);
		font-size: 0.78rem;
		padding: 0.2rem 0.5rem;
		border: 1px solid var(--rule);
	}
	.cell-badge.acbi,
	.cell-badge.aibc {
		background: var(--accent-wash);
		color: var(--accent);
		border-color: var(--accent);
	}
	.drill-total {
		color: var(--ink-muted);
		font-size: 0.9rem;
	}
	.sem-pill {
		font-family: var(--mono);
		font-size: 0.66rem;
		color: var(--ink-faint);
		border: 1px solid var(--rule);
		padding: 0.1rem 0.4rem;
	}

	/* L1 stratification */
	.strat {
		margin: 0 0 2rem;
	}
	.strat-axes {
		display: flex;
		flex-wrap: wrap;
		gap: 0.3rem;
		margin: 0 0 0.8rem;
	}
	.ax-tab {
		background: none;
		border: 1px solid var(--rule);
		font-family: var(--mono);
		font-size: 0.72rem;
		color: var(--ink-muted);
		padding: 0.25rem 0.6rem;
		cursor: pointer;
	}
	.ax-tab.active {
		background: var(--ink);
		color: var(--paper);
		border-color: var(--ink);
	}
	.strat-bars {
		display: flex;
		flex-direction: column;
		gap: 0.15rem;
	}
	.strat-row {
		display: grid;
		grid-template-columns: 11rem 1fr 4rem 3.5rem;
		align-items: center;
		gap: 0.6rem;
		background: none;
		border: none;
		border-bottom: 1px dotted var(--rule);
		font: inherit;
		font-size: 0.82rem;
		padding: 0.3rem 0.2rem;
		cursor: pointer;
		text-align: left;
	}
	.strat-row:hover {
		background: var(--accent-wash);
	}
	.strat-row.filtered {
		background: var(--accent-wash);
		box-shadow: inset 2px 0 0 var(--accent);
	}
	.strat-val {
		font-family: var(--mono);
		font-size: 0.76rem;
		white-space: nowrap;
		overflow: hidden;
		text-overflow: ellipsis;
	}
	.strat-bar-track {
		height: 0.7rem;
		background: var(--rule);
	}
	.strat-bar-fill {
		display: block;
		height: 100%;
		background: var(--accent);
		opacity: 0.55;
	}
	.strat-n {
		text-align: right;
		font-size: 0.78rem;
	}
	.strat-pct {
		text-align: right;
		font-size: 0.74rem;
		color: var(--ink-muted);
	}
	.filter-note {
		margin: 0.6rem 0 0;
	}

	/* L2 cohort */
	table.cohort {
		width: 100%;
		border-collapse: collapse;
		font-size: 0.82rem;
	}
	table.cohort thead th {
		font-family: var(--mono);
		font-size: 0.66rem;
		font-weight: 500;
		text-align: left;
		color: var(--ink-muted);
		padding: 0.4rem 0.5rem;
		border-bottom: 1px solid var(--ink);
	}
	table.cohort td {
		padding: 0.45rem 0.5rem;
		border-bottom: 1px dotted var(--rule);
		vertical-align: top;
	}
	.cohort-row {
		cursor: pointer;
	}
	.cohort-row:hover {
		background: var(--accent-wash);
	}
	.c-ev {
		color: var(--ink-muted);
		max-width: 30rem;
		line-height: 1.35;
	}
	.c-v {
		text-align: center;
	}
	.c-conf {
		font-size: 0.72rem;
		color: var(--ink-faint);
	}
	.agent {
		font-weight: 500;
	}
	.rel {
		font-family: var(--mono);
		font-size: 0.72rem;
		color: var(--ink-muted);
	}
	.v.correct {
		color: var(--ok-green);
	}
	.v.incorrect {
		color: var(--accent);
	}
	.v.none {
		color: var(--ink-faint);
	}

	/* L3 side-by-side */
	.sbs-claim {
		margin: 0 0 1.4rem;
		padding: 0 0 1rem;
		border-bottom: 1px solid var(--rule);
	}
	.claim-line {
		font-size: 1.15rem;
		margin-bottom: 0.3rem;
	}
	.claim-meta {
		font-size: 0.74rem;
		color: var(--ink-muted);
		margin-bottom: 0.7rem;
	}
	.evidence-text {
		margin: 0 0 0.7rem;
		padding: 0.7rem 1rem;
		background: var(--paper);
		border-left: 3px solid var(--ink);
		font-size: 0.95rem;
		line-height: 1.5;
		max-width: 75ch;
	}
	.back {
		font-size: 0.8rem;
	}
	.traces {
		display: grid;
		grid-template-columns: 1fr 1fr;
		gap: 1rem;
	}
	.trace {
		border: 1px solid var(--rule);
		padding: 0.8rem;
	}
	.trace.tA {
		background: var(--a-wash);
	}
	.trace.tB {
		background: var(--b-wash);
	}
	.trace-head {
		display: flex;
		flex-wrap: wrap;
		align-items: baseline;
		gap: 0.5rem;
		margin: 0 0 0.6rem;
		padding-bottom: 0.5rem;
		border-bottom: 1px solid var(--rule);
	}
	.trace-model {
		font-size: 1rem;
	}
	.trace-verdict {
		font-family: var(--mono);
		font-size: 0.78rem;
		font-weight: 600;
	}
	.trace-conf {
		font-size: 0.7rem;
		color: var(--ink-faint);
	}
	.reasoning {
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

	/* ── gold (human curation) ───────────────────────────────── */
	.toggles {
		display: flex;
		flex-direction: column;
		gap: 0.3rem;
		align-items: flex-end;
	}
	.gold-toggle {
		color: var(--gold-hue);
	}
	.gold-coverage {
		margin: 0 0 1.2rem;
		padding: 0.5rem 0.7rem;
		font-size: 0.82rem;
		color: var(--ink-muted);
		background: var(--gold-wash);
		border-left: 3px solid var(--gold-hue);
	}
	.gold-coverage.active {
		font-weight: 500;
	}
	.gc-key {
		color: var(--gold-hue);
		font-size: 0.72rem;
	}
	.gc-val {
		font-weight: 600;
		color: var(--ink);
	}
	.gc-pct {
		color: var(--ink-faint);
	}
	.gc-caveat {
		font-style: italic;
		color: var(--accent);
		font-size: 0.9em;
	}
	.gold-scalar .sep {
		color: var(--ink-faint);
		margin: 0 0.2rem;
	}
	.cell-gold {
		display: block;
		margin-top: 0.25rem;
		font-size: 0.62rem;
		color: var(--gold-hue);
		line-height: 1.2;
	}
	.cell-gold.none {
		color: var(--ink-faint);
		font-style: italic;
		opacity: 0.6;
	}
	.gold-pill {
		color: var(--gold-hue);
		border-color: var(--gold-hue);
	}
	.gold-filters {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		gap: 0.3rem;
		margin-top: 0.5rem;
	}
	.gf-label {
		font-size: 0.7rem;
		color: var(--gold-hue);
	}
	.gf-chip {
		background: none;
		border: 1px solid var(--rule);
		font-family: var(--mono);
		font-size: 0.68rem;
		color: var(--ink-muted);
		padding: 0.18rem 0.5rem;
		cursor: pointer;
	}
	.gf-chip.active {
		background: var(--gold-hue);
		color: var(--paper);
		border-color: var(--gold-hue);
	}
	.gf-clear {
		font-size: 0.72rem;
	}
	/* cohort gold column + row tints */
	.c-gold {
		text-align: center;
	}
	.v.gold {
		color: var(--gold-hue);
	}
	.v.gmiss {
		outline: 1px solid var(--accent);
		border-radius: 2px;
		padding: 0 2px;
	}
	.gnone {
		color: var(--ink-faint);
	}
	tr.gold-fp {
		background: rgba(125, 42, 26, 0.05);
	}
	tr.gold-fn {
		background: rgba(180, 120, 20, 0.06);
	}
	/* L3 third trace */
	.traces.with-gold {
		grid-template-columns: 1fr 1fr 1fr;
	}
	.trace.tGold {
		background: var(--gold-wash);
	}
	.trace.gmiss {
		box-shadow: inset 0 2px 0 var(--accent);
	}
	.gold-name {
		color: var(--gold-hue);
		font-size: 1rem;
	}
	.trace-gflag {
		font-family: var(--mono);
		font-size: 0.62rem;
		padding: 0.05rem 0.35rem;
		border-radius: 2px;
	}
	.trace-gflag.ok {
		color: var(--ok-green);
	}
	.trace-gflag.bad {
		color: var(--accent);
		background: var(--accent-wash);
	}
	.gold-body {
		font-size: 0.78rem;
		line-height: 1.5;
	}
	.gold-field {
		display: flex;
		gap: 0.5rem;
		margin-bottom: 0.4rem;
	}
	.gf-k {
		font-family: var(--mono);
		font-size: 0.64rem;
		color: var(--ink-faint);
		min-width: 4rem;
		text-transform: uppercase;
		letter-spacing: 0.04em;
	}
	.gf-chips {
		display: flex;
		flex-wrap: wrap;
		gap: 0.25rem;
	}
	.tag-chip {
		font-family: var(--mono);
		font-size: 0.66rem;
		padding: 0.05rem 0.35rem;
		background: var(--accent-wash);
		color: var(--accent);
		border-radius: 2px;
	}
	.tag-chip.tag-correct {
		background: rgba(42, 111, 42, 0.1);
		color: var(--ok-green);
	}
	.gf-v {
		color: var(--ink);
		word-break: break-word;
	}
	.gold-notes {
		font-style: italic;
		color: var(--ink-muted);
	}
	.gold-rule {
		margin: 0.6rem 0 0;
		font-size: 0.66rem;
		font-style: italic;
		color: var(--ink-faint);
	}
	.no-gold-note {
		margin-top: 0.8rem;
	}

	@media (max-width: 760px) {
		.traces {
			grid-template-columns: 1fr;
		}
		.cell {
			width: 7rem;
			height: 4.2rem;
		}
		.strat-row {
			grid-template-columns: 7rem 1fr 3rem 3rem;
		}
		.picker select {
			min-width: 0;
			width: 100%;
		}
	}
</style>
