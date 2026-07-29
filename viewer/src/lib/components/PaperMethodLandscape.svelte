<script lang="ts">
	import {
		PAPER_METHOD_FAMILIES,
		type PaperMethodLandscapeLoad,
		type PaperMethodRow
	} from '$lib/data/paper-method-landscape';

	let { reference }: { reference: PaperMethodLandscapeLoad } = $props();

	const plotLeft = 170;
	const plotRight = 735;
	const domainMin = 0.85;
	const domainMax = 0.97;
	const ticks = [0.86, 0.88, 0.9, 0.92, 0.94, 0.96] as const;

	function x(value: number): number {
		return plotLeft + ((value - domainMin) / (domainMax - domainMin)) * (plotRight - plotLeft);
	}

	function familyY(family: PaperMethodRow['family']): number {
		return 42 + PAPER_METHOD_FAMILIES.indexOf(family) * 50;
	}

	function methodY(method: PaperMethodRow, methods: PaperMethodRow[]): number {
		const familyMethods = methods.filter((candidate) => candidate.family === method.family);
		const index = familyMethods.findIndex((candidate) => candidate.method_id === method.method_id);
		if (familyMethods.length === 1) return familyY(method.family);
		return familyY(method.family) - 13 + (index / (familyMethods.length - 1)) * 26;
	}

	function shortSha(value: string): string {
		return `${value.slice(0, 10)}…`;
	}
</script>

<section class="paper-landscape" aria-labelledby="paper-landscape-title">
	{#if reference.status === 'unavailable'}
		<div class="gate" role="status">
			<p class="eyebrow">2023 paper method context</p>
			<h2 id="paper-landscape-title">Published landscape unavailable</h2>
			<p>{reference.reason}</p>
			<code>{reference.artifact_path}</code>
		</div>
	{:else}
		{@const landscape = reference.landscape}
		<header>
			<div>
				<p class="eyebrow">published 2023 reference · unpaired context</p>
				<h2 id="paper-landscape-title">Where the paper’s 59 reported methods sat</h2>
			</div>
			<strong>not another frontier</strong>
		</header>
		<div class="guard" role="note">
			<strong>Do not subtract these rounded values from the direct scores above.</strong>
			The paper rows span different eligible statement sets and reconstructed folds. Their bars are
			population SD across 10 folds—not confidence intervals—and no statement predictions or comparable
			costs were published. They cannot enter paired deltas, parity claims, or the cost Pareto frontier.
		</div>

		<dl class="anchors">
			<div>
				<dt>original belief · readers</dt>
				<dd>{landscape.baseline.fold_mean_trapezoidal_pr_auc.toFixed(3)} <small>± {landscape.baseline.fold_population_sd.toFixed(3)} fold SD</small></dd>
			</div>
			<div>
				<dt>best published row</dt>
				<dd>{landscape.best.fold_mean_trapezoidal_pr_auc.toFixed(3)} <small>± {landscape.best.fold_population_sd.toFixed(3)} fold SD</small></dd>
			</div>
			<div>
				<dt>direct bridge</dt>
				<dd>2 RF arms <small>reproduced in the all-source tab above on the shared 1,689-statement panel</small></dd>
			</div>
		</dl>

		<figure>
			<figcaption>Published fold-mean trapezoidal PR area by method family</figcaption>
			<svg viewBox="0 0 760 300" role="img" aria-labelledby="paper-chart-title paper-chart-desc">
				<title id="paper-chart-title">Fifty-nine published 2023 INDRA method summaries</title>
				<desc id="paper-chart-desc">Each point is a rounded fold-mean trapezoidal precision-recall area. Horizontal bars are population standard deviations over ten folds, not confidence intervals. The chart is contextual and unpaired.</desc>
				{#each ticks as tick}
					<line class="grid" x1={x(tick)} y1="20" x2={x(tick)} y2="256" />
					<text class="tick" x={x(tick)} y="278">{tick.toFixed(2)}</text>
				{/each}
				{#each PAPER_METHOD_FAMILIES as family}
					<line class="family-rule" x1={plotLeft} y1={familyY(family)} x2={plotRight} y2={familyY(family)} />
					<text class="family-label" x="158" y={familyY(family) + 4}>{family}</text>
					<text class="family-count" x="744" y={familyY(family) + 4}>n={landscape.family_counts[family]}</text>
				{/each}
				{#each landscape.methods as method (method.method_id)}
					{@const y = methodY(method, landscape.methods)}
					<line
						class="sd"
						x1={x(method.fold_mean_trapezoidal_pr_auc - method.fold_population_sd)}
						x2={x(method.fold_mean_trapezoidal_pr_auc + method.fold_population_sd)}
						y1={y}
						y2={y}
					/>
					<circle
						class:secondary={method.table_id === 'paper_not_table_6'}
						class:anchor={method.method_id === landscape.baseline.method_id || method.method_id === landscape.best.method_id}
						cx={x(method.fold_mean_trapezoidal_pr_auc)}
						cy={y}
						r={method.method_id === landscape.baseline.method_id || method.method_id === landscape.best.method_id ? 4.5 : 3}
					>
						<title>{method.method}: {method.fold_mean_trapezoidal_pr_auc.toFixed(3)} ± {method.fold_population_sd.toFixed(3)} fold SD</title>
					</circle>
				{/each}
				<line class="axis" x1={plotLeft} y1="256" x2={plotRight} y2="256" />
				<text class="axis-label" x={(plotLeft + plotRight) / 2} y="296">published fold-mean trapezoidal PR area →</text>
			</svg>
		</figure>

		<details>
			<summary>Inspect all 59 published rows</summary>
			<div class="table-scroll">
				<table>
					<thead><tr><th>method</th><th>family</th><th>fold mean</th><th>fold SD</th><th>source row</th></tr></thead>
					<tbody>
						{#each landscape.methods as method (method.method_id)}
							<tr>
								<td>{method.method}</td>
								<td>{method.family}</td>
								<td>{method.fold_mean_trapezoidal_pr_auc.toFixed(3)}</td>
								<td>{method.fold_population_sd.toFixed(3)} <small>not CI</small></td>
								<td><code>{method.method_id}</code></td>
							</tr>
						{/each}
					</tbody>
				</table>
			</div>
		</details>

		<footer>
			Pinned paper commit <code>{landscape.source.commit.slice(0, 10)}…</code> ·
			<code>{reference.artifact_path}</code> · artifact
			<span title={reference.artifact_sha256}>{shortSha(reference.artifact_sha256)}</span>
		</footer>
	{/if}
</section>

<style>
	.paper-landscape { margin-top: 1.5rem; padding-top: 1.25rem; border-top: 2px solid var(--ink); }
	.paper-landscape > header { display: flex; justify-content: space-between; align-items: start; gap: 1rem; }
	.paper-landscape h2, .gate h2 { margin: 0.18rem 0 0; font-family: var(--serif); font-size: 1.35rem; font-weight: 400; }
	.paper-landscape > header > strong { flex: 0 0 auto; padding: 0.2rem 0.38rem; border: 1px solid var(--blocked); color: var(--blocked); font-family: var(--mono); font-size: 0.62rem; font-weight: 500; text-transform: uppercase; letter-spacing: 0.04em; }
	.eyebrow { margin: 0; font-family: var(--mono); font-size: 0.68rem; color: var(--ink-faint); text-transform: uppercase; letter-spacing: 0.05em; }
	.guard { margin-top: 0.8rem; padding: 0.75rem 0.9rem; border: 1px solid var(--blocked); border-left-width: 3px; background: color-mix(in srgb, var(--blocked) 3%, transparent); font-family: var(--serif); font-size: 0.82rem; line-height: 1.5; color: var(--ink-muted); }
	.guard strong { color: var(--ink); }
	.anchors { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 0.8rem; margin: 1rem 0; }
	.anchors div { padding-top: 0.45rem; border-top: 1px dotted var(--rule); }
	.anchors dt { font-family: var(--mono); font-size: 0.63rem; color: var(--ink-faint); }
	.anchors dd { margin: 0.2rem 0 0; font-family: var(--mono); font-size: 0.9rem; }
	.anchors small { display: block; margin-top: 0.12rem; font-size: 0.61rem; color: var(--ink-faint); }
	figure { margin: 0.8rem 0 0; }
	figcaption { font-family: var(--mono); font-size: 0.68rem; color: var(--ink-muted); }
	svg { display: block; width: 100%; min-height: 300px; overflow: visible; }
	.grid { stroke: var(--rule); stroke-width: 1; }
	.family-rule { stroke: color-mix(in srgb, var(--rule) 75%, transparent); stroke-width: 1; stroke-dasharray: 2 3; }
	.axis { stroke: var(--ink); stroke-width: 1; }
	.tick, .family-label, .family-count, .axis-label { fill: var(--ink-faint); font-family: var(--mono); font-size: 10px; }
	.tick, .axis-label { text-anchor: middle; }
	.family-label { text-anchor: end; fill: var(--ink-muted); }
	.family-count { font-size: 9px; }
	.sd { stroke: color-mix(in srgb, var(--ink-muted) 28%, transparent); stroke-width: 1; }
	circle { fill: var(--accent); stroke: var(--paper); stroke-width: 0.8; opacity: 0.78; }
	circle.secondary { fill: var(--paper); stroke: var(--accent); stroke-width: 1.2; }
	circle.anchor { fill: var(--blocked); stroke: var(--paper); stroke-width: 1.4; opacity: 1; }
	details { margin-top: 0.6rem; border-top: 1px solid var(--rule); padding-top: 0.65rem; }
	summary { cursor: pointer; font-family: var(--mono); font-size: 0.68rem; color: var(--ink-muted); }
	.table-scroll { overflow-x: auto; margin-top: 0.7rem; }
	table { width: 100%; border-collapse: collapse; font-family: var(--mono); font-size: 0.66rem; }
	th { color: var(--ink-faint); font-weight: 500; text-align: left; }
	th, td { padding: 0.38rem 0.45rem; border-bottom: 1px dotted var(--rule); vertical-align: top; }
	td:nth-child(n + 3) { font-variant-numeric: tabular-nums; }
	td small { color: var(--blocked); }
	code, footer { font-family: var(--mono); font-size: 0.62rem; color: var(--ink-faint); }
	footer { margin-top: 0.8rem; }
	.gate { border: 1px solid var(--rule); border-left: 3px solid var(--blocked); padding: 1rem; }
	.gate p:not(.eyebrow) { font-family: var(--serif); color: var(--ink-muted); }
	@media (max-width: 720px) {
		.anchors { grid-template-columns: 1fr; }
		.paper-landscape > header { display: block; }
		.paper-landscape > header > strong { display: inline-block; margin-top: 0.6rem; }
	}
</style>
