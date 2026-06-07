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
	function setGran(g: 'evidence' | 'statement') {
		nav({ gran: g === 'statement' ? 'statement' : null });
	}

	// gold-performance payload (error partition + CI recalls), gold mode only
	const gp = $derived(data.goldPerf);

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
							{#if gold.n_evaluable / Math.max(1, gold.n_both_scored) < 0.9}
								<span class="gc-caveat">— spot-check only, not corpus-wide</span>
							{:else}
								<span class="gc-caveat">— gold on (nearly) every compared evidence</span>
							{/if}
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

						{#if data.goldMode && gp?.present}
							{@const er = gp.error_partition}
							{@const aR = gp.a_error_recall}
							{@const bR = gp.b_error_recall}
							<section class="goldperf">
								<header class="gp-head">
									<h3>vs human gold</h3>
									<span class="gp-unit mono">{gp.n} {data.granularity === 'statement' ? 'statements' : 'evidences'} · {gp.n_supported} supported · {gp.n_error} errors</span>
									<span class="gp-gran" role="group" aria-label="granularity">
										<button class:active={data.granularity !== 'statement'} onclick={() => setGran('evidence')}>evidence</button>
										<button class:active={data.granularity === 'statement'} onclick={() => setGran('statement')}>statement</button>
									</span>
								</header>

								<!-- THE CLIFF: both classes on ONE recall axis. the gap = positive-vs-negative -->
								<div class="gp-cliff">
									<div class="gp-cliff-axis">
										<span class="gp-axend left">catching errors <span class="mono">n={gp.n_error}</span></span>
										<span class="gp-axmid mono">recall →</span>
										<span class="gp-axend right">confirming supported <span class="mono">n={gp.n_supported}</span></span>
									</div>
									<div class="gp-cliff-plot">
										<span class="gp-tick" style="left:0%"></span>
										<span class="gp-tick" style="left:25%"></span>
										<span class="gp-tick mid" style="left:50%"></span>
										<span class="gp-tick" style="left:75%"></span>
										<span class="gp-tick" style="left:100%"></span>
										{#each [{m:aModel,e:gp.a_error_recall,su:gp.a_supported_recall,c:'mA'},{m:bModel,e:gp.b_error_recall,su:gp.b_supported_recall,c:'mB'}] as row}
											<div class="gp-slope">
												<span class="gp-slope-label {row.c}">{row.m.split(' ')[0]}</span>
												<span class="gp-slope-track">
													<!-- the span between error-recall and supported-recall = the cliff -->
													<span class="gp-gap {row.c}g" style="left:{row.e.p*100}%; right:{(1-row.su.p)*100}%"></span>
													<!-- error CI whisker + dot (the uncertain, decisive end) -->
													<span class="gp-eband {row.c}b" style="left:{row.e.lo*100}%; right:{(1-row.e.hi)*100}%"></span>
													<span class="gp-pt err {row.c}p" style="left:{row.e.p*100}%" title="error recall {(row.e.p*100).toFixed(0)}% [{(row.e.lo*100).toFixed(0)}-{(row.e.hi*100).toFixed(0)}], n={row.e.n}"></span>
													<span class="gp-pt sup {row.c}p" style="left:{row.su.p*100}%" title="supported recall {(row.su.p*100).toFixed(0)}%, n={row.su.n}"></span>
													<span class="gp-lab err" style="left:{row.e.p*100}%">{(row.e.p*100).toFixed(0)}%</span>
													<span class="gp-lab sup" style="left:{row.su.p*100}%">{(row.su.p*100).toFixed(0)}%</span>
												</span>
											</div>
										{/each}
									</div>
									<p class="gp-cliff-note hint">
										each line is one model's drop from the easy task (supported, right) to the hard one (errors, left).
										both fall off a cliff: ~95% → {Math.round(Math.min(gp.a_error_recall.p,gp.b_error_recall.p)*100)}–{Math.round(Math.max(gp.a_error_recall.p,gp.b_error_recall.p)*100)}%.
										the error-end whiskers (95% CI) overlap, so the {aModel.split(' ')[0]}/{bModel.split(' ')[0]} difference there is inside the noise.
									</p>
								</div>

								<div class="gp-task gp-errors">
									<div class="gp-task-label">
										<strong>catching errors</strong> <span class="gp-n mono">n={gp.n_error}</span>
										<span class="hint">— the curator-flagged wrong extractions. each mark is one.</span>
									</div>
									<div class="gp-matrix">
										<div class="gp-row">
											<span class="gp-rowlabel mA">{aModel.split(' ')[0]}</span>
											<span class="gp-dots">
												{#each gp.errors as e}
													<button class="gp-dot {e.a_verdict === 'incorrect' ? 'caught' : 'missed'}"
														title={`${e.subject} [${e.stmt_type}] ${e.object} — ${aModel.split(' ')[0]} ${e.a_verdict === 'incorrect' ? 'CAUGHT' : 'missed'} · gold tags: ${e.tags.join(', ')}`}
														onclick={() => e.evidence_hash && openEvidence(e.stmt_hash, e.evidence_hash)}
														aria-label="error case"></button>
												{/each}
											</span>
											<span class="gp-recall mono">{aR.k}/{aR.n}</span>
										</div>
										<div class="gp-row">
											<span class="gp-rowlabel mB">{bModel.split(' ')[0]}</span>
											<span class="gp-dots">
												{#each gp.errors as e}
													<button class="gp-dot {e.b_verdict === 'incorrect' ? 'caught' : 'missed'}"
														title={`${e.subject} [${e.stmt_type}] ${e.object} — ${bModel.split(' ')[0]} ${e.b_verdict === 'incorrect' ? 'CAUGHT' : 'missed'} · gold tags: ${e.tags.join(', ')}`}
														onclick={() => e.evidence_hash && openEvidence(e.stmt_hash, e.evidence_hash)}
														aria-label="error case"></button>
												{/each}
											</span>
											<span class="gp-recall mono">{bR.k}/{bR.n}</span>
										</div>
										<div class="gp-partition mono">
											<span class="gp-seg both" style="flex:{er.both}" title="both models caught">{er.both ? `${er.both} both` : ''}</span>
											<span class="gp-seg aonly" style="flex:{er.a_only}" title="A only">{er.a_only || ''}</span>
											<span class="gp-seg bonly" style="flex:{er.b_only}" title="B only">{er.b_only || ''}</span>
											<span class="gp-seg neither" style="flex:{er.neither}" title="MISSED by both">{er.neither ? `${er.neither} missed by both` : ''}</span>
										</div>
									</div>

								</div>

								<!-- raw 2x2 per model: area = count, so the imbalance is GEOMETRY not a footnote -->
								<div class="gp-conf">
									<div class="gp-conf-title hint">full 2×2 — cell area ∝ count (gold ↓ · model →)</div>
									<div class="gp-conf-pair">
										{#each [{m:aModel,c:gp.a_conf,k:'mA'},{m:bModel,c:gp.b_conf,k:'mB'}] as cm}
											{@const tot = cm.c.n || 1}
											{@const rc = (cm.c.gc_pc + cm.c.gc_pi)}
											{@const ri = (cm.c.gi_pc + cm.c.gi_pi)}
											<figure class="gp-mosaic">
												<figcaption class="{cm.k}">{cm.m.split(' ')[0]}</figcaption>
												<div class="mosaic" style="--rc:{rc}; --ri:{ri}">
													<div class="mrow" style="flex:{rc}">
														<div class="mcell agree" style="flex:{cm.c.gc_pc}" title="gold supported, model supported — {cm.c.gc_pc}"><span>{cm.c.gc_pc}</span></div>
														<div class="mcell miss-fn" style="flex:{cm.c.gc_pi}" title="gold supported, model said error (over-reject) — {cm.c.gc_pi}"><span>{cm.c.gc_pi}</span></div>
													</div>
													<div class="mrow" style="flex:{ri}">
														<div class="mcell miss-fp" style="flex:{cm.c.gi_pc}" title="gold ERROR, model said supported (over-accept, the dangerous miss) — {cm.c.gi_pc}"><span>{cm.c.gi_pc}</span></div>
														<div class="mcell catch" style="flex:{cm.c.gi_pi}" title="gold error, model caught it — {cm.c.gi_pi}"><span>{cm.c.gi_pi}</span></div>
													</div>
												</div>
											</figure>
										{/each}
									</div>
									<div class="gp-conf-legend hint">
										<span><span class="sw agree"></span> agree ✓</span>
										<span><span class="sw catch"></span> error caught</span>
										<span><span class="sw miss-fp"></span> error missed (said ✓)</span>
										<span><span class="sw miss-fn"></span> supported rejected</span>
									</div>
								</div>

							</section>
						{/if}

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
	.goldperf {
		margin: 0 0 1.8rem;
		padding: 0.2rem 0 0;
	}
	.gp-head {
		display: flex;
		align-items: baseline;
		gap: 0.7rem;
		flex-wrap: wrap;
		margin-bottom: 0.9rem;
	}
	.gp-head h3 {
		font-family: var(--serif);
		font-weight: 400;
		font-size: 1.05rem;
		margin: 0;
		color: var(--gold-hue);
	}
	.gp-unit {
		font-size: 0.72rem;
		color: var(--ink-muted);
	}
	.gp-gran {
		margin-left: auto;
		display: inline-flex;
		border: 1px solid var(--rule);
		border-radius: 3px;
		overflow: hidden;
	}
	.gp-gran button {
		font-family: var(--mono);
		font-size: 0.68rem;
		padding: 0.2rem 0.6rem;
		background: var(--paper);
		border: none;
		color: var(--ink-muted);
		cursor: pointer;
	}
	.gp-gran button.active {
		background: var(--gold-hue);
		color: var(--paper);
	}
	.gp-task {
		margin-bottom: 1.1rem;
	}
	.gp-task-label {
		font-family: var(--serif);
		font-size: 0.92rem;
		margin-bottom: 0.5rem;
	}
	.gp-n {
		font-size: 0.74rem;
		color: var(--ink-muted);
		margin-left: 0.2rem;
	}
	/* THE CLIFF — both classes on one recall axis; the slope IS the contrast */
	.gp-cliff {
		margin: 0.2rem 0 1.4rem;
	}
	.gp-cliff-axis {
		display: flex;
		justify-content: space-between;
		align-items: baseline;
		font-size: 0.76rem;
		color: var(--ink-muted);
		margin-bottom: 0.5rem;
	}
	.gp-cliff-axis .gp-axend.left {
		color: var(--accent);
		font-weight: 600;
	}
	.gp-cliff-axis .gp-axend.right {
		color: var(--ok-green);
	}
	.gp-axmid {
		font-size: 0.66rem;
		color: var(--ink-faint);
	}
	.gp-cliff-plot {
		position: relative;
		padding: 0.6rem 0 0.3rem;
	}
	.gp-tick {
		position: absolute;
		top: 0;
		bottom: 0;
		width: 1px;
		background: var(--rule);
	}
	.gp-tick.mid {
		background: color-mix(in srgb, var(--ink-faint) 40%, transparent);
	}
	.gp-slope {
		position: relative;
		display: flex;
		align-items: center;
		height: 2.4rem;
		gap: 0.6rem;
	}
	.gp-slope-label {
		font-family: var(--mono);
		font-size: 0.74rem;
		width: 5.5rem;
		text-align: right;
		flex: none;
		z-index: 2;
	}
	.gp-slope-track {
		position: relative;
		flex: 1;
		height: 100%;
	}
	/* the gap line: spans error-recall → supported-recall. its LENGTH = the cliff */
	.gp-gap {
		position: absolute;
		top: 50%;
		height: 3px;
		transform: translateY(-50%);
		border-radius: 2px;
		opacity: 0.45;
	}
	.gp-gap.mAg { background: var(--a-hue); }
	.gp-gap.mBg { background: var(--b-hue); }
	/* error-end CI whisker (the uncertain, decisive end) */
	.gp-eband {
		position: absolute;
		top: 50%;
		height: 0.55rem;
		transform: translateY(-50%);
		border-radius: 3px;
		opacity: 0.28;
	}
	.gp-eband.mAb { background: var(--a-hue); }
	.gp-eband.mBb { background: var(--b-hue); }
	.gp-pt {
		position: absolute;
		top: 50%;
		transform: translate(-50%, -50%);
		border-radius: 50%;
		z-index: 2;
	}
	.gp-pt.sup {
		width: 11px;
		height: 11px;
	}
	.gp-pt.err {
		width: 11px;
		height: 11px;
		background: var(--paper) !important;
		border: 2px solid;
	}
	.gp-pt.mAp { background: var(--a-hue); border-color: var(--a-hue); }
	.gp-pt.mBp { background: var(--b-hue); border-color: var(--b-hue); }
	.gp-lab {
		position: absolute;
		top: calc(50% + 0.7rem);
		transform: translateX(-50%);
		font-family: var(--mono);
		font-size: 0.64rem;
		color: var(--ink-muted);
		white-space: nowrap;
	}
	.gp-lab.err {
		color: var(--accent);
		font-weight: 600;
	}
	.gp-cliff-note {
		margin: 0.5rem 0 0;
		max-width: 76ch;
		line-height: 1.45;
	}
	/* errors detail — the dot matrix (drill-down under the cliff) */
	.gp-matrix {
		display: flex;
		flex-direction: column;
		gap: 0.3rem;
		margin: 0.2rem 0 0.9rem;
	}
	.gp-row {
		display: flex;
		align-items: center;
		gap: 0.6rem;
	}
	.gp-rowlabel {
		font-family: var(--mono);
		font-size: 0.72rem;
		width: 5.5rem;
		text-align: right;
		flex: none;
	}
	.gp-dots {
		display: flex;
		gap: 3px;
		flex-wrap: wrap;
		flex: 1;
	}
	.gp-dot {
		width: 15px;
		height: 15px;
		border-radius: 3px;
		border: 1px solid var(--rule);
		padding: 0;
		cursor: pointer;
		background: var(--paper);
		transition: transform 0.1s;
	}
	.gp-dot.caught {
		background: var(--ok-green);
		border-color: var(--ok-green);
	}
	.gp-dot.missed {
		background: var(--paper);
		border: 1px solid var(--accent);
		box-shadow: inset 0 0 0 2px var(--paper), inset 0 0 0 3px color-mix(in srgb, var(--accent) 25%, var(--paper));
	}
	.gp-dot:hover {
		transform: scale(1.25);
		z-index: 1;
	}
	.gp-recall {
		font-size: 0.78rem;
		width: 3rem;
		flex: none;
		color: var(--ink);
	}
	/* partition bar under the dots: both | A-only | B-only | MISSED */
	.gp-partition {
		display: flex;
		gap: 2px;
		margin-left: 6.1rem;
		margin-right: 3.6rem;
		font-size: 0.6rem;
		height: 1.1rem;
		margin-top: 0.15rem;
	}
	.gp-seg {
		display: flex;
		align-items: center;
		justify-content: center;
		color: var(--paper);
		white-space: nowrap;
		overflow: hidden;
		min-width: 0;
		border-radius: 2px;
	}
	.gp-seg.both { background: var(--ok-green); }
	.gp-seg.aonly { background: var(--a-hue); }
	.gp-seg.bonly { background: var(--b-hue); }
	.gp-seg.neither {
		background: var(--accent);
		font-weight: 600;
	}
	.gp-seg:empty { background: transparent; }
	/* CI strip: overlap = the gap is noise */
	.gp-ci {
		margin-top: 0.6rem;
	}
	.gp-cirow {
		display: flex;
		align-items: center;
		gap: 0.6rem;
		margin-bottom: 0.25rem;
	}
	.gp-cilabel {
		font-family: var(--mono);
		font-size: 0.72rem;
		width: 5.5rem;
		text-align: right;
		flex: none;
	}
	.gp-citrack {
		position: relative;
		flex: 1;
		height: 0.9rem;
		background: linear-gradient(var(--rule), var(--rule)) center / 100% 1px no-repeat;
	}
	.gp-ciband {
		position: absolute;
		top: 50%;
		transform: translateY(-50%);
		height: 0.5rem;
		border-radius: 3px;
		opacity: 0.3;
	}
	.gp-ciband.mAb { background: var(--a-hue); }
	.gp-ciband.mBb { background: var(--b-hue); }
	.gp-cipoint {
		position: absolute;
		top: 50%;
		width: 8px;
		height: 8px;
		border-radius: 50%;
		transform: translate(-50%, -50%);
	}
	.gp-cipoint.mAp { background: var(--a-hue); }
	.gp-cipoint.mBp { background: var(--b-hue); }
	.gp-cival {
		font-size: 0.74rem;
		width: 3rem;
		flex: none;
	}
	.gp-ci-note {
		margin: 0.4rem 0 0;
		max-width: 74ch;
		line-height: 1.45;
	}
	/* mosaic confusion: cell area ∝ count — the imbalance is geometry */
	.gp-conf {
		margin-top: 1.2rem;
		padding-top: 1rem;
		border-top: 1px solid var(--rule);
	}
	.gp-conf-title {
		margin-bottom: 0.6rem;
		font-size: 0.72rem;
	}
	.gp-conf-pair {
		display: flex;
		gap: 1.6rem;
		flex-wrap: wrap;
	}
	.gp-mosaic {
		margin: 0;
		flex: 1;
		min-width: 14rem;
	}
	.gp-mosaic figcaption {
		font-family: var(--mono);
		font-size: 0.74rem;
		margin-bottom: 0.3rem;
	}
	.gp-mosaic figcaption.mA { color: var(--a-hue); }
	.gp-mosaic figcaption.mB { color: var(--b-hue); }
	.mosaic {
		display: flex;
		flex-direction: column;
		gap: 2px;
		height: 8rem;
	}
	.mrow {
		display: flex;
		gap: 2px;
		min-height: 0;
	}
	.mcell {
		display: flex;
		align-items: center;
		justify-content: center;
		min-width: 0;
		border-radius: 2px;
		overflow: hidden;
	}
	.mcell span {
		font-family: var(--mono);
		font-size: 0.74rem;
		color: var(--paper);
		font-variant-numeric: tabular-nums;
	}
	/* small cells: drop the number rather than overflow */
	.mcell[style*='flex:1'] span,
	.mcell[style*='flex:2'] span,
	.mcell[style*='flex:3'] span {
		font-size: 0.6rem;
	}
	.mcell.agree { background: color-mix(in srgb, var(--ok-green) 78%, black); }
	.mcell.catch { background: var(--ok-green); }
	.mcell.miss-fp { background: var(--accent); }
	.mcell.miss-fn { background: var(--blocked, #6f5a16); }
	.mcell span { opacity: 0.95; }
	.gp-conf-legend {
		display: flex;
		flex-wrap: wrap;
		gap: 0.4rem 1rem;
		margin-top: 0.6rem;
		font-size: 0.68rem;
	}
	.gp-conf-legend .sw {
		display: inline-block;
		width: 0.7rem;
		height: 0.7rem;
		border-radius: 2px;
		vertical-align: -1px;
		margin-right: 0.25rem;
	}
	.gp-conf-legend .sw.agree { background: color-mix(in srgb, var(--ok-green) 78%, black); }
	.gp-conf-legend .sw.catch { background: var(--ok-green); }
	.gp-conf-legend .sw.miss-fp { background: var(--accent); }
	.gp-conf-legend .sw.miss-fn { background: var(--blocked, #6f5a16); }
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
