<script lang="ts">
	import type { LatestValidity, ResidualDistribution } from '$lib/db';
	import { residualBraille, residualPath } from '$lib/residuals';

	let {
		v,
		residuals
	}: {
		v: LatestValidity;
		residuals: ResidualDistribution | null;
	} = $props();

	const verdictTotal = $derived(v.verdicts.reduce((s, vd) => s + vd.n, 0));
	const verdictPct = (n: number) => (verdictTotal === 0 ? 0 : (n / verdictTotal) * 100);

	function verdictDisplayName(name: string): string {
		return name === 'correct' ? 'supported' : name === 'incorrect' ? 'contradicted' : name;
	}

	/** "Did our system match the hand-labeled gold?" One row per truth_set. */
	const goldLines = $derived(
		v.truthPresent.map((t) => {
			const tp = t.tp;
			const fp = t.fp;
			const fn = t.fn;
			const tn = t.tn;
			const matched = tp + tn;
			const total = t.n_compared;
			const perfect = fp === 0 && fn === 0;
			const setLabel = t.truth_set_id.replace(/^indra_/, '').replace(/^source_db_/, '').replace(/_/g, ' ');
			const fields = t.gold_fields.length > 0 ? t.gold_fields.join('+') : 'verdict';
			const goldLabelCount = t.n_gold_labels ?? total;
			const applicableLabelCount = t.n_applicable_gold_labels ?? total;
			const coverage = `overlap rows=${total} · applicable labels=${applicableLabelCount}/${goldLabelCount}`;
			let headline: string;
			let detail: string;
			let glyph = perfect ? '✓' : '✗';
			let glyphKind = perfect ? 'ok' : 'fail';
			if (t.unavailable_reason) {
				glyph = '—';
				glyphKind = 'na';
				headline = 'not defined for this run';
				detail = `${t.unavailable_reason} · applicable labels=${applicableLabelCount}/${goldLabelCount} · scored evidence=${t.n_scored_evidences ?? 0} · labels=${fields}`;
			} else if (perfect) {
				headline = `matched ${matched}/${total}`;
				detail = `${tp} true-positive · ${tn} true-negative · ${coverage} · labels=${fields} · positive=${t.positive_gold_label}`;
			} else {
				headline = `matched ${matched}/${total} · F1 ${(t.f1 ?? 0).toFixed(2)} (P ${(t.precision ?? 0).toFixed(2)} R ${(t.recall ?? 0).toFixed(2)})`;
				detail = `${fp} false-positive · ${fn} false-negative · ${tn} true-negative · ${coverage} · labels=${fields} · ${t.negative_gold_rule}`;
			}
			return {
				set_label: setLabel,
				step_kind: t.step_kind,
				n_compared: total,
				n_gold_labels: goldLabelCount,
				n_applicable_gold_labels: applicableLabelCount,
				glyph,
				glyphKind,
				headline,
				detail,
				cohort_href: t.cohort_href,
				truth_href: t.truth_href,
				unavailable_reason: t.unavailable_reason,
				applicability: t.applicability
			};
		})
	);

	/** "How do we compare to INDRA's priors?" */
	const indraLine = $derived.by(() => {
		const c = v.calibration;
		if (c.mae == null) {
			return { glyph: '—', direction: 'na' as const, headline: 'no INDRA-comparable beliefs in this run', detail: '', cohort_href: c.cohort_href };
		}
		const bias = c.bias ?? 0;
		const oneSided = c.mae > 0 && Math.abs(Math.abs(bias) - c.mae) < 1e-3;
		const n = c.n_stmts ?? 0;
		if (Math.abs(bias) < 0.05) {
			return {
				glyph: '≈',
				direction: 'eq' as const,
				headline: `closely matched`,
				detail: `MAE ${c.mae.toFixed(2)} · n=${n}`,
				cohort_href: c.cohort_href
			};
		}
		const dir = bias > 0 ? 'over' : 'under';
		const arrow = bias > 0 ? '▲' : '▼';
		const oneSidedNote =
			oneSided && c.mae > 0.001
				? `every score landed ${bias < 0 ? 'below' : 'above'} INDRA`
				: `MAE ${c.mae.toFixed(2)}`;
		return {
			glyph: arrow,
			direction: bias > 0 ? ('up' as const) : ('down' as const),
			headline: `${dir}-confident by ${Math.abs(bias).toFixed(2)}`,
			detail: `${oneSidedNote} · n=${n}`,
			cohort_href: c.cohort_href
		};
	});

	/** Stratification: only worth surfacing when the slice has > 1 stratum,
	 *  otherwise "top 1" is the whole thing and tells you nothing. */
	const byType = $derived(v.byIndraType.length > 1 ? v.byIndraType.slice(0, 5) : []);
	const bySource = $derived(v.bySourceApi.length > 1 ? v.bySourceApi.slice(0, 5) : []);
	const lonelyType = $derived(v.byIndraType.length === 1 ? v.byIndraType[0] : null);
	const lonelySource = $derived(v.bySourceApi.length === 1 ? v.bySourceApi[0] : null);

	const consistencySentence = $derived.by(() => {
		const c = v.inter_evidence_consistency;
		if (c.mean_stdev == null) return null;
		return {
			headline: `per-evidence scores agreed within stdev ${c.mean_stdev.toFixed(2)}`,
			detail: `${c.n_multi_ev ?? 0} statement${c.n_multi_ev === 1 ? '' : 's'} had multiple evidences`
		};
	});

	/** Show the histogram only when there are enough points for the shape to mean
	 *  something. Below ~30 the bars are misleading — the prose already carries
	 *  the bias direction. */
	const SHOW_HISTOGRAM_THRESHOLD = 30;
	const showHistogram = $derived(residuals != null && residuals.n_total >= SHOW_HISTOGRAM_THRESHOLD);
	const confidenceRows = $derived(v.confidenceCalibration.filter((r) => r.n > 0).slice(0, 8));

	function taxonomyHref(panel: string): string | null {
		return v.metricTaxonomy.find((row) => row.panel === panel)?.cohort_href ?? null;
	}

	type DenominatorStatus = 'defined' | 'not_defined' | 'applicable';

	function cellGlyph(status: DenominatorStatus): string {
		if (status === 'defined') return '✓';
		if (status === 'applicable') return '•';
		return '—';
	}

	function cellStatusLabel(status: DenominatorStatus): string {
		if (status === 'not_defined') return 'not defined';
		return status;
	}

	type DenominatorCell = {
		label: string;
		status: DenominatorStatus;
		unit: string;
		n: number | null;
		href: string | null;
		detail: string;
	};
	type DenominatorGroup = {
		key: string;
		label: string;
		scope: string;
		cells: DenominatorCell[];
	};
	const denominatorGroups = $derived.by((): DenominatorGroup[] => {
		const verdictHref = taxonomyHref('verdict distribution');
		const probeTax = v.metricTaxonomy.find((row) => row.panel === 'decomposed probe health');
		const pairTax = v.metricTaxonomy.find((row) => row.panel === 'paired deltas');
		const truthCells: DenominatorCell[] = goldLines.length > 0
			? goldLines.map((g): DenominatorCell => ({
				label: `${g.set_label}${g.step_kind === 'aggregate' ? '' : ` / ${g.step_kind}`}`,
				status: g.unavailable_reason ? 'not_defined' : 'defined',
				unit: 'truth overlap row',
				n: g.n_compared,
				href: g.unavailable_reason ? g.truth_href : g.cohort_href,
				detail: g.unavailable_reason
					? `${g.unavailable_reason}; opens overlap diagnostic; applicable labels ${g.n_applicable_gold_labels}/${g.n_gold_labels}`
					: `compared rows ${g.n_compared}; applicable labels ${g.n_applicable_gold_labels}/${g.n_gold_labels}`
			}))
			: [{
				label: 'gold-pool comparison',
				status: 'not_defined',
				unit: 'truth overlap row',
				n: null,
				href: null,
				detail: 'no truth_set rows are registered for this run'
			}];
		const confidenceCells: DenominatorCell[] = confidenceRows.length > 0
			? confidenceRows.map((row): DenominatorCell => ({
				label: row.family === 'all'
					? `all ${row.confidence}`
					: `${row.family.replace(/^indra_type:/, 'type ')} ${row.confidence}`,
				status: 'defined',
				unit: row.family === 'all' ? 'aggregate evidence bucket' : 'overlapping type slice',
				n: row.n,
				href: row.cohort_href,
				detail: row.family === 'all'
					? `all aggregate rows in this confidence bucket with score and INDRA belief; MAE ${row.mae == null ? '—' : row.mae.toFixed(2)}`
					: `type subset of the INDRA-anchor aggregate row population, not additive with all-family buckets; MAE ${row.mae == null ? '—' : row.mae.toFixed(2)}`
			}))
			: [{
				label: 'confidence buckets',
				status: 'not_defined',
				unit: 'INDRA-anchor slice',
				n: null,
				href: null,
				detail: 'no confidence bucket has enough INDRA-anchored rows to report'
			}];
		const groups: DenominatorGroup[] = [
			{
				key: 'aggregate-evidence',
				label: 'aggregate verdict evidence',
				scope: 'arch-blind run rows',
				cells: [
					{
						label: 'verdict distribution',
						status: verdictTotal > 0 ? 'defined' : 'not_defined',
						unit: 'aggregate evidence',
						n: verdictTotal,
						href: verdictHref,
						detail: verdictTotal > 0
							? 'one count per aggregate evidence verdict in this run'
							: 'no aggregate verdict rows exist yet'
					}
				]
			},
			{
				key: 'truth-overlap',
				label: 'truth-set overlap',
				scope: 'gold labels matched to this run',
				cells: truthCells
			},
			{
				key: 'indra-anchors',
				label: 'INDRA belief anchors',
				scope: 'whole-run statement denominator',
				cells: [
					{
						label: 'INDRA prior calibration',
						status: v.calibration.mae == null ? 'not_defined' : 'defined',
						unit: 'statement',
						n: v.calibration.n_stmts,
						href: v.calibration.cohort_href,
						detail: v.calibration.mae == null
							? 'no INDRA-comparable beliefs in this run'
							: `statements with INDRA belief anchors; MAE ${v.calibration.mae.toFixed(2)}; bias ${v.calibration.bias == null ? '—' : v.calibration.bias.toFixed(2)}`
					}
				]
			},
			{
				key: 'confidence-buckets',
				label: 'confidence calibration buckets',
				scope: 'overlapping INDRA-anchor aggregate evidence buckets; rows are not additive',
				cells: confidenceCells
			},
			{
				key: 'multi-evidence',
				label: 'statement structure',
				scope: 'statement subset denominator',
				cells: [
					{
						label: 'multi-evidence agreement',
						status: v.inter_evidence_consistency.cohort_href ? 'defined' : 'not_defined',
						unit: 'statement',
						n: v.inter_evidence_consistency.n_multi_ev,
						href: v.inter_evidence_consistency.cohort_href,
						detail: v.inter_evidence_consistency.cohort_href
							? `mean stdev ${v.inter_evidence_consistency.mean_stdev?.toFixed(2) ?? '—'}`
							: v.inter_evidence_consistency.not_defined_reason ?? 'needs statements with more than one scored evidence'
					}
				]
			},
			{
				key: 'architecture-native',
				label: 'native / paired applicability',
				scope: 'routes are diagnostics, not shared denominators',
				cells: [
					{
						label: 'decomposed probe health',
						status: probeTax?.applicability === 'arch_conditioned' ? 'applicable' : 'not_defined',
						unit: 'native panel',
						n: null,
						href: probeTax?.cohort_href ?? null,
						detail: probeTax?.reason
							? `${probeTax.reason}; route opens aggregate evidence with native decomposed trace rows`
							: 'native probe health is defined only for decomposed runs'
					},
					{
						label: 'paired deltas',
						status: pairTax?.applicability === 'paired_only' ? 'applicable' : 'not_defined',
						unit: 'paired workbench',
						n: null,
						href: pairTax?.cohort_href ?? null,
						detail: pairTax?.reason ?? 'architecture deltas require an overlap-first paired workbench'
					}
				]
			}
		];
		return groups;
	});
</script>

<section class="validity">
	<h2 class="v-h">
		how is the system doing in this run?
		<span class="v-run-id" title="run_id">{v.run_id.slice(0, 8)}</span>
		<span class="v-run-id" title="architecture">{v.architecture}</span>
	</h2>

	<details class="v-denominator-map" aria-label="summary metric denominator contract">
		<summary>
			<span class="v-denominator-summary-row">
				<span>denominator contract</span>
				<span class="muted">metric populations, route availability, and not-defined states</span>
			</span>
		</summary>
		<div class="v-denominator-grid">
			{#each denominatorGroups as group}
				<section class="v-denom-group" aria-label={group.label}>
					<h3>{group.label} <span>{group.scope}</span></h3>
					<ul>
						{#each group.cells as cell}
							<li class="v-denom-row v-denom-{cell.status}">
								<span class="v-denom-state" aria-label={cellStatusLabel(cell.status)}>
									<span class="v-denom-glyph">{cellGlyph(cell.status)}</span>
									<span>{cellStatusLabel(cell.status)}</span>
								</span>
								{#if cell.href}
									<a class="v-denom-cell v-link" href={cell.href}>{cell.label}</a>
								{:else}
									<span class="v-denom-cell">{cell.label}</span>
								{/if}
								<span class="v-denom-n">{cell.n == null ? 'n —' : `n=${cell.n}`}</span>
								<span class="v-denom-unit">{cell.unit}</span>
								<span class="muted">{cell.detail}</span>
							</li>
						{/each}
					</ul>
				</section>
			{/each}
		</div>
	</details>

	{#each goldLines as g}
		<div class="v-line">
			<a class="v-line-label v-link" href={g.truth_href}>vs gold <span class="muted">({g.set_label}{g.step_kind === 'aggregate' ? '' : ` / ${g.step_kind}`})</span></a>
			<span class="v-glyph v-glyph-{g.glyphKind}">{g.glyph}</span>
			<span class="v-headline">{g.headline}</span>
			<span class="muted v-detail">· {g.detail}</span>
			{#if g.unavailable_reason}
				<a class="v-inline-link" href={g.truth_href}>overlap details</a>
			{:else if g.cohort_href}
				<a class="v-inline-link" href={g.cohort_href}>cohort</a>
			{/if}
		</div>
	{/each}

	<div class="v-line">
		<a class="v-line-label v-link" href={indraLine.cohort_href}>vs INDRA's priors</a>
		<span class="v-glyph v-glyph-{indraLine.direction}">{indraLine.glyph}</span>
		<span class="v-headline">{indraLine.headline}</span>
		{#if indraLine.detail}<span class="muted v-detail">· {indraLine.detail}</span>{/if}
	</div>

	{#if verdictTotal > 0}
		<div class="v-line v-line-block">
			<span class="v-line-label" title="our scorer's classification of each evidence">per-evidence verdicts</span>
			<div class="v-pillbar-wrap">
				<div class="v-pillbar" role="img" aria-label="verdict distribution">
					{#each v.verdicts as vd}
						{@const pct = verdictPct(vd.n)}
						{#if pct > 0}
							<a class="v-pill v-pill-{vd.verdict}" href={vd.cohort_href} style:width="{pct}%" title="{verdictDisplayName(vd.verdict)}: {vd.n} of {verdictTotal} ({pct.toFixed(1)}%)"></a>
						{/if}
					{/each}
				</div>
				<div class="v-pill-caption">
					{#each v.verdicts as vd, i}{#if i > 0}<span class="muted"> · </span>{/if}<a href={vd.cohort_href} class="v-pill-tag v-pill-tag-{vd.verdict}">{verdictDisplayName(vd.verdict)}</a> <span class="v-pill-num">{vd.n}</span>{/each}
					<span class="muted"> · n={verdictTotal}</span>
				</div>
				<p class="v-explain">each evidence sentence is judged separately — a statement with multiple evidences contributes multiple counts</p>
			</div>
		</div>
	{/if}

	{#if byType.length > 0 || bySource.length > 0 || lonelyType || lonelySource}
		<div class="v-line v-line-block">
			<span class="v-line-label">weakest by slice</span>
			<div class="v-weakest-list">
				{#if byType.length > 0}
					<div class="v-weak-group">
						<span class="muted">by indra_type:</span>
						{#each byType as s, i}
							{#if i > 0}<span class="muted">·</span>{/if}
							<a class="v-weak-row v-link" href={s.cohort_href}>
								<span class="v-weak-name">{s.value}</span>
								<span class="v-weak-mae">MAE {s.mae.toFixed(2)}</span>
								<span class="v-weak-bias" class:b-pos={s.bias >= 0.05} class:b-neg={s.bias <= -0.05}>{s.bias >= 0 ? '▲' : '▼'}{Math.abs(s.bias).toFixed(2)}</span>
								<span class="muted">n={s.n}</span>
							</a>
						{/each}
					</div>
				{:else if lonelyType}
					<div class="v-weak-group">
						<span class="muted">by indra_type: only one type in this run —</span>
						<a class="v-weak-row v-link" href={lonelyType.cohort_href}>
							<span class="v-weak-name">{lonelyType.value}</span>
							<span class="v-weak-mae">MAE {lonelyType.mae.toFixed(2)}</span>
							<span class="muted">n={lonelyType.n}</span>
						</a>
					</div>
				{/if}
				{#if bySource.length > 0}
					<div class="v-weak-group">
						<span class="muted">by source_api:</span>
						{#each bySource as s, i}
							{#if i > 0}<span class="muted">·</span>{/if}
							<a class="v-weak-row v-link" href={s.cohort_href}>
								<span class="v-weak-name">{s.value}</span>
								<span class="v-weak-mae">MAE {s.mae.toFixed(2)}</span>
								<span class="v-weak-bias" class:b-pos={s.bias >= 0.05} class:b-neg={s.bias <= -0.05}>{s.bias >= 0 ? '▲' : '▼'}{Math.abs(s.bias).toFixed(2)}</span>
								<span class="muted">n={s.n}</span>
							</a>
						{/each}
					</div>
				{:else if lonelySource}
					<div class="v-weak-group">
						<span class="muted">by source_api: only one source in this run —</span>
						<a class="v-weak-row v-link" href={lonelySource.cohort_href}>
							<span class="v-weak-name">{lonelySource.value}</span>
							<span class="v-weak-mae">MAE {lonelySource.mae.toFixed(2)}</span>
							<span class="muted">n={lonelySource.n}</span>
						</a>
					</div>
				{/if}
			</div>
		</div>
	{/if}

	{#if confidenceRows.length > 0}
		<div class="v-line v-line-block">
			<span class="v-line-label">confidence calibration</span>
			<div class="v-conf-list">
				{#each confidenceRows as row}
					<a class="v-conf-row v-link" href={row.cohort_href}>
						<span class="v-conf-family">{row.family}</span>
						<span class="v-conf-bucket">{row.confidence}</span>
						<span>MAE {row.mae == null ? '—' : row.mae.toFixed(2)}</span>
						<span class:b-pos={row.bias != null && row.bias >= 0.05} class:b-neg={row.bias != null && row.bias <= -0.05}>{row.bias == null ? '—' : `${row.bias >= 0 ? '▲' : '▼'}${Math.abs(row.bias).toFixed(2)}`}</span>
						<span class="muted">n={row.n}</span>
					</a>
				{/each}
			</div>
			<p class="v-explain">measured inside {v.architecture} using INDRA published belief as the calibration anchor</p>
		</div>
	{/if}

	{#if consistencySentence}
		<div class="v-line">
			{#if v.inter_evidence_consistency.cohort_href}
				<a class="v-line-label v-link" href={v.inter_evidence_consistency.cohort_href} title="when a statement has multiple evidences, do their per-evidence scores agree?">multi-evidence agreement</a>
			{:else}
				<span class="v-line-label" title="when a statement has multiple evidences, do their per-evidence scores agree?">multi-evidence agreement</span>
			{/if}
			<span class="v-headline">{consistencySentence.headline}</span>
			<span class="muted v-detail">· {consistencySentence.detail}</span>
		</div>
	{:else if v.inter_evidence_consistency.applicability === 'not_defined'}
		<div class="v-line">
			<span class="v-line-label">multi-evidence agreement</span>
			<span class="v-headline">not defined for this run</span>
			<span class="muted v-detail">· {v.inter_evidence_consistency.not_defined_reason}</span>
		</div>
	{/if}

	{#if v.supports_graph_delta != null}
		<div class="v-line">
			{#if v.supports_graph_href}
				<a class="v-line-label v-link" href={v.supports_graph_href} title="how much our scores stratify across the supports_edge graph">supports-graph plausibility</a>
			{:else}
				<span class="v-line-label" title="how much our scores stratify across the supports_edge graph">supports-graph plausibility</span>
			{/if}
			<span class="v-headline">Δ {v.supports_graph_delta >= 0 ? '+' : ''}{v.supports_graph_delta.toFixed(2)}</span>
		</div>
	{:else if v.supports_graph_not_defined_reason}
		<div class="v-line">
			<span class="v-line-label">supports-graph plausibility</span>
			<span class="v-headline">not defined for this run</span>
			<span class="muted v-detail">· {v.supports_graph_not_defined_reason}</span>
		</div>
	{/if}

	{#if showHistogram}
		<div class="v-residuals">
			<p class="v-residuals-label">residual distribution (our − INDRA), n={residuals!.n_total}</p>
			<svg viewBox="0 0 240 40" preserveAspectRatio="none" class="v-res-svg" aria-label="residual histogram (our − INDRA), n={residuals!.n_total}">
				<line x1="120" y1="0" x2="120" y2="40" stroke="var(--ink-faint)" stroke-width="0.5" stroke-dasharray="2,2"/>
				<path d={residualPath(residuals!.bins, 240, 40)} fill="var(--ink)" opacity="0.85"/>
			</svg>
			<div class="v-res-axis">
				<span>−1</span><span>0</span><span>+1</span>
			</div>
			<span class="v-braille muted" title="braille fallback">{residualBraille(residuals!.bins)}</span>
		</div>
	{/if}
</section>

<style>
	.validity {
		margin: 0 0 2.5rem;
	}
	.v-h {
		font-family: var(--serif);
		font-size: 1.15rem;
		font-weight: 400;
		color: var(--ink);
		margin: 0 0 1rem;
		display: flex;
		align-items: baseline;
		gap: 0.6rem;
	}
	.v-run-id {
		font-family: var(--mono);
		font-size: 0.72rem;
		color: var(--ink-faint);
	}
	.v-line {
		display: flex;
		flex-wrap: wrap;
		gap: 0.6rem;
		align-items: baseline;
		padding: 0.4rem 0;
		border-bottom: 1px dotted var(--rule);
		font-family: var(--mono);
		font-size: 0.86rem;
	}
	.v-line-block {
		flex-direction: column;
		align-items: flex-start;
		gap: 0.3rem;
	}
	.v-line:last-child {
		border-bottom: none;
	}

	.v-line-label {
		flex-basis: 14rem;
		flex-shrink: 0;
		color: var(--ink-muted);
		text-transform: lowercase;
		letter-spacing: 0.02em;
		font-size: 0.78rem;
	}
	.v-link {
		color: var(--accent);
		text-decoration: none;
	}
	.v-inline-link {
		color: var(--accent);
		font-size: 0.78rem;
		text-decoration: none;
	}

	.v-denominator-map {
		margin: 0 0 1rem;
		font-family: var(--mono);
		font-size: 0.74rem;
		border-top: 1px solid var(--rule);
		border-bottom: 1px dotted var(--rule);
		padding: 0.42rem 0;
	}
	.v-denominator-map summary {
		cursor: pointer;
		color: var(--ink);
		text-transform: lowercase;
	}
	.v-denominator-map summary::marker {
		color: var(--ink-muted);
	}
	.v-denominator-summary-row {
		display: inline-flex;
		flex-wrap: wrap;
		gap: 0.5rem;
		align-items: baseline;
	}
	.v-denominator-grid {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(17rem, 1fr));
		gap: 0.75rem;
		margin-top: 0.65rem;
	}
	.v-denom-group {
		border-top: 1px dotted var(--rule);
		padding-top: 0.45rem;
	}
	.v-denom-group h3 {
		margin: 0 0 0.35rem;
		font-size: 0.76rem;
		font-weight: 500;
		color: var(--ink);
		text-transform: lowercase;
		letter-spacing: 0.02em;
	}
	.v-denom-group h3 span {
		display: block;
		margin-top: 0.1rem;
		color: var(--ink-muted);
		font-weight: 400;
		text-transform: none;
		letter-spacing: 0;
	}
	.v-denom-group ul {
		list-style: none;
		margin: 0;
		padding: 0;
		display: flex;
		flex-direction: column;
		gap: 0.28rem;
	}
	.v-denom-row {
		display: grid;
		grid-template-columns: 6.2rem minmax(8rem, 1fr) auto;
		column-gap: 0.45rem;
		row-gap: 0.1rem;
		align-items: baseline;
	}
	.v-denom-state {
		color: var(--ok-green);
		display: inline-flex;
		align-items: baseline;
		gap: 0.3rem;
		font-weight: 500;
	}
	.v-denom-not_defined .v-denom-state {
		color: var(--ink-faint);
	}
	.v-denom-applicable .v-denom-state {
		color: var(--ink-muted);
	}
	.v-denom-glyph {
		display: inline-block;
		min-width: 0.7rem;
		text-align: center;
	}
	.v-denom-cell {
		color: var(--ink);
	}
	.v-denom-n {
		color: var(--ink-muted);
		font-variant-numeric: tabular-nums;
		white-space: nowrap;
	}
	.v-denom-unit {
		grid-column: 2 / -1;
		color: var(--ink-muted);
		font-size: 0.7rem;
	}
	.v-denom-group .muted {
		grid-column: 2 / -1;
		font-size: 0.7rem;
		line-height: 1.35;
	}

	.v-link:hover,
	.v-denom-group a:hover,
	.v-inline-link:hover,
	.v-pill-tag:hover {
		text-decoration: underline;
	}
	/* In block (column-flex) lines, flex-basis: 14rem becomes 14rem of *height*
	   rather than width — which produced a giant vertical gap. Reset to auto. */
	.v-line-block .v-line-label {
		flex-basis: auto;
	}

	.v-glyph {
		font-family: var(--mono);
		font-weight: 500;
		min-width: 1.2rem;
		text-align: center;
	}
	.v-glyph-ok { color: var(--ok-green); }
	.v-glyph-fail { color: var(--accent); }
	.v-glyph-up { color: var(--ok-green); }
	.v-glyph-down { color: var(--accent); }
	.v-glyph-eq { color: var(--ink-muted); }
	.v-glyph-na { color: var(--ink-faint); }

	.v-headline {
		color: var(--ink);
		font-variant-numeric: tabular-nums;
	}
	.v-detail {
		font-size: 0.78rem;
	}

	/* Verdict pillbar */
	.v-pillbar-wrap {
		display: flex;
		flex-direction: column;
		gap: 0.2rem;
		width: 100%;
		max-width: 480px;
	}
	.v-pillbar {
		display: flex;
		width: 100%;
		height: 16px;
		border: 1px solid var(--rule);
		overflow: hidden;
	}
	.v-pill {
		display: block;
		min-width: 2px;
	}
	.v-pill-correct { background: var(--ok-green); }
	.v-pill-incorrect { background: var(--accent); }
	.v-pill-abstain { background: var(--ink-faint); }
	.v-pill-caption {
		font-family: var(--mono);
		font-size: 0.78rem;
		font-variant-numeric: tabular-nums;
	}
	.v-pill-tag-correct { color: var(--ok-green); }
	.v-pill-tag-incorrect { color: var(--accent); }
	.v-pill-tag-abstain { color: var(--ink-muted); }
	.v-pill-num { color: var(--ink); font-weight: 500; }
	.v-explain {
		font-family: var(--serif);
		font-style: italic;
		font-size: 0.78rem;
		color: var(--ink-faint);
		margin: 0.2rem 0 0;
	}

	/* Weakest by slice */
	.v-weakest-list {
		display: flex;
		flex-direction: column;
		gap: 0.3rem;
		font-family: var(--mono);
		font-size: 0.8rem;
	}
	.v-weak-group {
		display: flex;
		flex-wrap: wrap;
		gap: 0.5rem;
		align-items: baseline;
	}
	.v-weak-row {
		display: inline-flex;
		gap: 0.4rem;
		align-items: baseline;
	}
	.v-weak-name { color: var(--ink); }
	.v-weak-mae { color: var(--ink); font-variant-numeric: tabular-nums; }
	.v-weak-bias { color: var(--ink-muted); font-variant-numeric: tabular-nums; }
	.v-weak-bias.b-pos { color: var(--ok-green); }
	.v-weak-bias.b-neg { color: var(--accent); }
	.v-conf-list {
		display: flex;
		flex-direction: column;
		gap: 0.25rem;
		font-family: var(--mono);
		font-size: 0.8rem;
	}
	.v-conf-row {
		display: flex;
		flex-wrap: wrap;
		gap: 0.55rem;
		align-items: baseline;
	}
	.v-conf-family {
		color: var(--ink);
		min-width: 9rem;
	}
	.v-conf-bucket {
		color: var(--ink-muted);
		min-width: 4rem;
	}
	.b-pos { color: var(--ok-green); }
	.b-neg { color: var(--accent); }

	/* Residual histogram (only at n ≥ 30) */
	.v-residuals {
		margin-top: 1rem;
		padding-top: 0.6rem;
		border-top: 1px dotted var(--rule);
		max-width: 280px;
	}
	.v-residuals-label {
		font-family: var(--mono);
		font-size: 0.74rem;
		color: var(--ink-muted);
		margin: 0 0 0.2rem;
	}
	.v-res-svg {
		width: 100%;
		height: 40px;
		display: block;
	}
	.v-res-axis {
		display: flex;
		justify-content: space-between;
		font-family: var(--mono);
		font-size: 0.66rem;
		color: var(--ink-faint);
	}
	.v-braille {
		font-family: var(--mono);
		letter-spacing: -0.05em;
		display: block;
		margin-top: 0.1rem;
	}

	.muted { color: var(--ink-faint); }
</style>
