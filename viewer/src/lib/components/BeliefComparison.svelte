<script lang="ts">
	import { page } from '$app/state';
	import {
		BELIEF_PRIMARY_METRICS,
		BELIEF_THRESHOLD_METRICS,
		type BeliefArtifactValidation,
		type BeliefArm,
		type BeliefEstimate,
		type BeliefMetricKey,
		type BeliefPairedComparison
	} from '$lib/data/belief-comparison';

	type ComparisonPayload = BeliefArtifactValidation & {
		artifact_path: string;
		artifact_sha256: string | null;
	};

	let { comparison }: { comparison: ComparisonPayload } = $props();

	const metricLabels: Record<BeliefMetricKey, string> = {
		fold_mean_trapezoidal_pr_auc: 'fold-mean trapezoidal PR area',
		pooled_average_precision: 'pooled average precision',
		auroc: 'AUROC',
		brier: 'Brier',
		log_loss: 'log loss',
		calibration_ece: 'calibration ECE',
		calibration_intercept_abs_error: 'calibration intercept distance |intercept|',
		calibration_slope_abs_error: 'calibration slope distance |slope − 1|',
		threshold_accuracy: 'threshold accuracy',
		threshold_precision: 'threshold precision',
		threshold_recall: 'threshold recall',
		threshold_f1: 'threshold F1'
	};

	let selectedPanel = $state('');
	let selectedMetric = $state<BeliefMetricKey>('fold_mean_trapezoidal_pr_auc');
	$effect(() => {
		const requestedPanel = page.url.searchParams.get('belief_substrate');
		const requestedExists = comparison.panels.some(
			(candidate) => candidate.substrate_id === requestedPanel
		);
		const currentExists = comparison.panels.some(
			(candidate) => candidate.substrate_id === selectedPanel
		);
		const nextPanel = requestedExists
			? (requestedPanel ?? '')
			: currentExists
				? selectedPanel
				: comparison.panels[0]
					? comparison.panels[0].substrate_id
					: '';
		if (nextPanel !== selectedPanel) selectedPanel = nextPanel;
	});
	const panel = $derived(
		comparison.panels.find((candidate) => candidate.substrate_id === selectedPanel) ??
			comparison.panels[0] ??
			null
	);
	const familyRank = { paper: 0, current: 1, llm: 2 } as const;
	const arms = $derived(
		panel
			? [...panel.arms].sort(
					(a, b) => familyRank[a.family] - familyRank[b.family] || a.label.localeCompare(b.label)
				)
			: []
	);
	const strictArms = $derived(
		panel
			? [...panel.strict_e0_resolved_sensitivity.arms].sort(
					(a, b) => familyRank[a.family] - familyRank[b.family] || a.label.localeCompare(b.label)
				)
			: []
	);
	const comparisonMetrics = $derived(
		panel
			? [...new Set(panel.comparisons.map((row) => row.metric))]
			: []
	);
	const visibleComparisons = $derived(
		panel
			? panel.comparisons.filter((row) => row.metric === selectedMetric)
			: []
	);
	type HeadlinePoint = {
		arm: BeliefArm;
		costLower: number;
		cost: number;
		performance: number;
		ci95: [number, number];
		index: number;
		pointPareto: boolean;
		uncertaintyPareto: boolean;
	};
	const headlineCostView = $derived.by(() => {
		if (!panel) return null;
		const candidates = panel.pareto.views
			.filter((view) => view.eligible_arm_ids.length >= 2)
			.sort(
				(a, b) =>
					b.eligible_arm_ids.length - a.eligible_arm_ids.length ||
					a.view_id.localeCompare(b.view_id)
			);
		return candidates[0] ?? null;
	});
	const headlinePoints = $derived.by((): HeadlinePoint[] => {
		if (!headlineCostView) return [];
		const ranking = [...arms]
			.sort(
				(a, b) =>
					b.metrics.fold_mean_trapezoidal_pr_auc.estimate -
					a.metrics.fold_mean_trapezoidal_pr_auc.estimate
			)
			.map((arm) => arm.arm_id);
		return arms.flatMap((arm) =>
			arm.cost.status === 'available' &&
			arm.cost.view_id === headlineCostView.view_id &&
			headlineCostView.eligible_arm_ids.includes(arm.arm_id)
				? [
						{
							arm,
							costLower: arm.cost.usd_per_1k_statements_lower,
							cost: arm.cost.usd_per_1k_statements_upper,
							performance: arm.metrics.fold_mean_trapezoidal_pr_auc.estimate,
							ci95: arm.metrics.fold_mean_trapezoidal_pr_auc.ci95,
							index: ranking.indexOf(arm.arm_id) + 1,
							pointPareto: headlineCostView.point_frontier_arm_ids.includes(arm.arm_id),
							uncertaintyPareto: headlineCostView.uncertainty_frontier_arm_ids.includes(
								arm.arm_id
							)
						}
					]
				: []
		);
	});
	const headlineFrontierPoints = $derived(
		[...headlinePoints].filter((point) => point.pointPareto).sort((a, b) => a.cost - b.cost)
	);
	const headlineCostOmissions = $derived(
		headlineCostView
			? arms.filter((arm) => !headlineCostView.eligible_arm_ids.includes(arm.arm_id))
			: arms
	);
	const rankedArms = $derived(
		[...arms].sort(
			(a, b) =>
				b.metrics.fold_mean_trapezoidal_pr_auc.estimate -
				a.metrics.fold_mean_trapezoidal_pr_auc.estimate
		)
	);
	type OrientedDelta = {
		estimate: number;
		ci95: [number, number];
		comparison: BeliefPairedComparison;
	};
	type ReaderBridge = {
		original: BeliefArm;
		production: BeliefArm;
		bestObservedLlm: BeliefArm | null;
		productionMinusOriginal: OrientedDelta;
		llmMinusOriginal: OrientedDelta | null;
	};
	function orientedDelta(
		fromArmId: string,
		toArmId: string,
		metric: BeliefMetricKey
	): OrientedDelta | null {
		if (!panel) return null;
		const comparison = panel.comparisons.find(
			(row) =>
				row.metric === metric &&
				((row.a_arm_id === fromArmId && row.b_arm_id === toArmId) ||
					(row.a_arm_id === toArmId && row.b_arm_id === fromArmId))
		);
		if (!comparison) return null;
		if (comparison.a_arm_id === fromArmId) {
			return {
				estimate: comparison.delta.estimate,
				ci95: comparison.delta.ci95,
				comparison
			};
		}
		return {
			estimate: -comparison.delta.estimate,
			ci95: [-comparison.delta.ci95[1], -comparison.delta.ci95[0]],
			comparison
		};
	}
	const readerBridge = $derived.by((): ReaderBridge | null => {
		if (!panel) return null;
		const original = arms.find((arm) => arm.arm_id === 'orig_belief_readers');
		const production = arms.find(
			(arm) => arm.arm_id === 'indra_cogex_hybrid_readers'
		);
		if (!original || !production) return null;
		const productionMinusOriginal = orientedDelta(
			original.arm_id,
			production.arm_id,
			'fold_mean_trapezoidal_pr_auc'
		);
		if (!productionMinusOriginal) return null;
		const bestObservedLlm =
			[...arms]
				.filter((arm) => arm.family === 'llm')
				.sort(
					(a, b) =>
						b.metrics.fold_mean_trapezoidal_pr_auc.estimate -
						a.metrics.fold_mean_trapezoidal_pr_auc.estimate
				)[0] ?? null;
		return {
			original,
			production,
			bestObservedLlm,
			productionMinusOriginal,
			llmMinusOriginal: bestObservedLlm
				? orientedDelta(
						original.arm_id,
						bestObservedLlm.arm_id,
						'fold_mean_trapezoidal_pr_auc'
					)
				: null
		};
	});
	const headlineCostMax = $derived(
		niceUpper(Math.max(0, ...headlinePoints.map((point) => point.cost)))
	);
	const chartTicks = [0, 0.25, 0.5, 0.75, 1] as const;

	$effect(() => {
		const requestedMetric = page.url.searchParams.get('belief_metric') as BeliefMetricKey | null;
		if (requestedMetric && comparisonMetrics.includes(requestedMetric)) {
			if (requestedMetric !== selectedMetric) selectedMetric = requestedMetric;
		} else if (comparisonMetrics.length && !comparisonMetrics.includes(selectedMetric)) {
			selectedMetric = comparisonMetrics[0];
		}
	});

	function selectPanel(id: string) {
		selectedPanel = id;
		const url = new URL(page.url);
		url.searchParams.set('view', 'belief');
		url.searchParams.set('belief_substrate', id);
		history.replaceState(history.state, '', url);
	}

	function selectMetric(metric: BeliefMetricKey) {
		selectedMetric = metric;
		const url = new URL(page.url);
		url.searchParams.set('view', 'belief');
		url.searchParams.set('belief_metric', metric);
		history.replaceState(history.state, '', url);
	}

	function shortSha(value: string | null): string {
		return value ? `${value.slice(0, 10)}…` : '—';
	}

	function estimate(value: BeliefEstimate, digits = 3): string {
		return value.estimate.toFixed(digits);
	}

	function interval(value: BeliefEstimate, digits = 3): string {
		return `[${value.ci95[0].toFixed(digits)}, ${value.ci95[1].toFixed(digits)}]`;
	}

	function signed(value: number): string {
		return `${value >= 0 ? '+' : '−'}${Math.abs(value).toFixed(3)}`;
	}

	function money(value: number): string {
		if (value === 0) return '$0';
		if (value < 0.01) return `$${value.toFixed(4)}`;
		return `$${value.toFixed(2)}`;
	}

	function niceUpper(value: number): number {
		if (value <= 0) return 1;
		const magnitude = 10 ** Math.floor(Math.log10(value));
		const normalized = value / magnitude;
		const ceiling = normalized <= 1 ? 1 : normalized <= 2 ? 2 : normalized <= 5 ? 5 : 10;
		return ceiling * magnitude;
	}

	function chartX(cost: number): number {
		return 68 + (cost / headlineCostMax) * 652;
	}

	function chartY(performance: number): number {
		return 18 + (1 - performance) * 262;
	}

	function armLabel(id: string): string {
		return arms.find((arm) => arm.arm_id === id)?.label ?? id;
	}

	function armLabels(ids: string[]): string {
		return ids.length ? ids.map(armLabel).join(' · ') : 'none';
	}

	function coverage(arm: BeliefArm): string {
		return `${arm.coverage.predicted}/${arm.coverage.eligible}`;
	}

	function calibrationX(value: number): number {
		return 10 + value * 100;
	}

	function calibrationY(value: number): number {
		return 110 - value * 100;
	}

	function pointRadius(n: number, total: number): number {
		return 2.2 + 3.8 * Math.sqrt(n / Math.max(total, 1));
	}
</script>

<section class="belief" aria-labelledby="belief-title">
	{#if comparison.status === 'unavailable'}
		<div class="gate" role="status">
			<p class="eyebrow">statement-belief comparison</p>
			<h2 id="belief-title">No head-to-head result yet</h2>
			<p class="gate-intro">The chart stays hidden until these inputs are validated together.</p>
			<ul class="readiness" aria-label="head-to-head readiness">
				<li><span>same statement set + gold</span><strong>not ready</strong></li>
				<li><span>paper + current INDRA + LLM scores</span><strong>not ready</strong></li>
				<li><span>retry-inclusive USD / 1k costs</span><strong>not ready</strong></li>
			</ul>
			<details class="gate-details">
				<summary>Current blocker</summary>
				<p>{comparison.reasons[0] ?? 'The frozen comparison artifact is not ready.'}</p>
				<code>{comparison.artifact_path}</code>
			</details>
		</div>
	{:else}
		<nav class="panels" aria-label="statement-belief substrate">
			{#each comparison.panels as candidate (candidate.substrate_id)}
				<button
					type="button"
					class:on={candidate.substrate_id === panel?.substrate_id}
					aria-pressed={candidate.substrate_id === panel?.substrate_id}
					onclick={() => selectPanel(candidate.substrate_id)}
				>
					<span>{candidate.label}</span>
					<small>paper · ready</small>
				</button>
			{/each}
		</nav>

		{#if panel}
			{@const substrate = panel}
			{#if (comparison.provenance?.bootstrap_resamples ?? 0) < 10_000}
				<aside class="diagnostic-banner" role="status">
					<strong>Diagnostic only · {comparison.provenance?.bootstrap_resamples.toLocaleString()} paired resamples</strong>
					<span>Publication requires 10,000. Do not interpret this build as a released parity, superiority, or deployment result.</span>
				</aside>
			{/if}
			<header class="panel-head">
				<div>
					<p class="eyebrow">paper substrate · released binary target</p>
					<h2 id="belief-title">{substrate.label}</h2>
				</div>
				<dl>
					<div><dt>eligible</dt><dd>{substrate.n_evaluable}</dd></div>
					<div><dt>released target</dt><dd>{substrate.released_label_audit.released.statements} binary labels</dd></div>
					<div><dt>class</dt><dd>{substrate.n_positive}+ / {substrate.n_negative}−</dd></div>
					<div><dt>positive</dt><dd>correct statement</dd></div>
					<div><dt>strict E0</dt><dd>{substrate.released_label_audit.strict_e0.resolved} resolved · {substrate.released_label_audit.strict_e0.unresolved} unresolved</dd></div>
				</dl>
			</header>

			<section class="descriptive-card" aria-labelledby="descriptive-title">
				<p class="eyebrow">comparison status</p>
				<h3 id="descriptive-title">Direct shared-gold comparison</h3>
				<p>
					Primary metrics preserve the paper’s released binary target. That target is directly comparable
					to the published benchmark, but it is not presented as complete evidence-level truth.
				</p>
			</section>

			<section class="gold-audit" aria-labelledby="gold-audit-title">
				<div>
					<p class="eyebrow">released-label audit</p>
					<h3 id="gold-audit-title">The paper-compatible target contains a documented negative assumption</h3>
					<p>{substrate.released_label_audit.released_label_rule}.</p>
					<p><strong>Strict E0:</strong> {substrate.released_label_audit.strict_e0_rule}.</p>
				</div>
				<dl>
					<div><dt>strictly resolved</dt><dd>{substrate.released_label_audit.strict_e0.resolved.toLocaleString()}</dd></div>
					<div><dt>unresolved</dt><dd>{substrate.released_label_audit.strict_e0.unresolved.toLocaleString()}</dd></div>
					<div><dt>share of released negatives</dt><dd>{(substrate.released_label_audit.released_negative_assumption.share_of_released_negatives * 100).toFixed(1)}%</dd></div>
				</dl>
			</section>

			<section class="strict-sensitivity" aria-labelledby="strict-sensitivity-title">
				<header>
					<div>
						<p class="eyebrow">fixed sensitivity · no duplicated cost or Pareto</p>
						<h3 id="strict-sensitivity-title">Strict E0 resolved-only results</h3>
					</div>
					<strong>{substrate.strict_e0_resolved_sensitivity.n_evaluable.toLocaleString()} resolved · {substrate.strict_e0_resolved_sensitivity.excluded_unresolved} excluded unresolved</strong>
				</header>
				<p>{substrate.strict_e0_resolved_sensitivity.selection_rule}. This checks label sensitivity; it does not replace the released-target parity panel.</p>
				<div class="table-scroll">
					<table>
						<thead><tr><th>configuration</th><th>fold-mean PR area</th><th>pooled AP</th><th>Brier</th><th>log loss</th><th>ECE</th></tr></thead>
						<tbody>
							{#each strictArms as arm (arm.arm_id)}
								<tr>
									<td><span class="family {arm.family}">{arm.family}</span>{arm.label}</td>
									<td>{estimate(arm.metrics.fold_mean_trapezoidal_pr_auc)} <small>{interval(arm.metrics.fold_mean_trapezoidal_pr_auc)}</small></td>
									<td>{estimate(arm.metrics.pooled_average_precision)} <small>{interval(arm.metrics.pooled_average_precision)}</small></td>
									<td>{estimate(arm.metrics.brier)} <small>{interval(arm.metrics.brier)}</small></td>
									<td>{estimate(arm.metrics.log_loss)} <small>{interval(arm.metrics.log_loss)}</small></td>
									<td>{estimate(arm.metrics.calibration.ece)} <small>{interval(arm.metrics.calibration.ece)}</small></td>
								</tr>
							{/each}
						</tbody>
					</table>
				</div>
				<details>
					<summary>Paired strict-E0 mean-fold PR deltas</summary>
					<div class="table-scroll">
						<table>
							<thead><tr><th>first → second</th><th>Δ</th><th>95% CI</th><th>valid / requested</th></tr></thead>
							<tbody>
								{#each substrate.strict_e0_resolved_sensitivity.comparisons.filter((row) => row.metric === 'fold_mean_trapezoidal_pr_auc') as row}
									<tr>
										<td>{armLabel(row.a_arm_id)} → {armLabel(row.b_arm_id)}</td>
										<td>{signed(row.delta.estimate)}</td>
										<td>[{signed(row.delta.ci95[0])}, {signed(row.delta.ci95[1])}]</td>
										<td>{row.delta.valid_resamples.toLocaleString()} / {row.delta.resamples.toLocaleString()}</td>
									</tr>
								{/each}
							</tbody>
						</table>
					</div>
				</details>
			</section>

			{#if readerBridge}
				<section class="reader-bridge" aria-labelledby="reader-bridge-title">
					<header>
						<div>
							<p class="eyebrow">direct readers-only bridge</p>
							<h3 id="reader-bridge-title">Production INDRA and the LLM, against the paper baseline</h3>
						</div>
						<strong>same {substrate.n_evaluable.toLocaleString()} statements</strong>
					</header>
					<p class="bridge-contract">
						These are direct shared-gold scores: the same reader-only statements, released labels,
						frozen folds, and paper-compatible fold-mean trapezoidal PR area. The deltas use paired
						statement resamples, so no subtraction from rounded paper-table summaries is needed.
					</p>
					<div class="bridge-grid">
						<article>
							<span>paper baseline</span>
							<strong>{readerBridge.original.metrics.fold_mean_trapezoidal_pr_auc.estimate.toFixed(3)}</strong>
							<small>{readerBridge.original.label} · 95% CI {interval(readerBridge.original.metrics.fold_mean_trapezoidal_pr_auc)}</small>
						</article>
						<article>
							<span>production INDRA Hybrid</span>
							<strong>{readerBridge.production.metrics.fold_mean_trapezoidal_pr_auc.estimate.toFixed(3)}</strong>
							<small>{readerBridge.production.label} · 95% CI {interval(readerBridge.production.metrics.fold_mean_trapezoidal_pr_auc)}</small>
							<em>Δ vs paper {signed(readerBridge.productionMinusOriginal.estimate)} · 95% CI [{signed(readerBridge.productionMinusOriginal.ci95[0])}, {signed(readerBridge.productionMinusOriginal.ci95[1])}]</em>
						</article>
						{#if readerBridge.bestObservedLlm && readerBridge.llmMinusOriginal}
							<article>
								<span>best observed LLM <i>descriptive</i></span>
								<strong>{readerBridge.bestObservedLlm.metrics.fold_mean_trapezoidal_pr_auc.estimate.toFixed(3)}</strong>
								<small>{readerBridge.bestObservedLlm.label} · 95% CI {interval(readerBridge.bestObservedLlm.metrics.fold_mean_trapezoidal_pr_auc)}</small>
								<em>Δ vs paper {signed(readerBridge.llmMinusOriginal.estimate)} · 95% CI [{signed(readerBridge.llmMinusOriginal.ci95[0])}, {signed(readerBridge.llmMinusOriginal.ci95[1])}]</em>
							</article>
						{/if}
					</div>
					<p class="bridge-caveat">
						<strong>Interpret descriptively.</strong> Training overlap between the deployed production Hybrid
						artifact and this released paper corpus is unknown; the best-observed LLM is selected after
						seeing these results. Neither comparison is promoted to a parity, superiority, or clean
						out-of-sample claim.
					</p>
				</section>
			{/if}

			<section class="headline-comparison" aria-labelledby="landscape-title">
				<header class="headline-copy">
					<p class="contract-pass"><span aria-hidden="true">✓</span> same statements · same gold · same score</p>
					<h3 id="landscape-title">Performance first, then cost</h3>
					<p>Every configuration and the cost frontier use the paper-compatible fold-mean trapezoidal PR area. The largest mutually comparable retry-inclusive ledger view supplies the cost axis.</p>
				</header>
				<p class="chart-note"><strong>Interval scope:</strong> paired bootstrap draws resample statements independently within each frozen fold, preserving each fold's size and sharing one weight vector across configurations. CIs are conditional on frozen fits, folds, and predictions; systems were not refit or rerun.</p>
				<p class="chart-note"><strong>Shared-run cost:</strong> all-source and reader LLM totals are two views of the same paid run. Never add them.</p>
				<p class="performance-label">correct-statement ranking (fold-mean PR area; higher is better)</p>
				<div class="performance-scale" aria-hidden="true"><span>0</span><span>paper-compatible fold-mean trapezoidal PR area · 95% CI</span><span>1</span></div>
				<ol class="performance-strip" aria-label="ranked headline performance with 95 percent confidence intervals">
					{#each rankedArms as arm, rank (arm.arm_id)}
						{@const headline = arm.metrics.fold_mean_trapezoidal_pr_auc}
						<li>
							<span class="point-key {arm.family}">{rank + 1}</span>
							<div class="performance-arm">
								<strong>{arm.label}</strong>
								<small><span class="family {arm.family}">{arm.family}</span>rank {rank + 1}</small>
							</div>
							<div class="performance-track" title={`${arm.label}: headline estimate ${headline.estimate.toFixed(3)}, 95% CI ${interval(headline)}`}>
								<span class="performance-ci" style={`left: ${headline.ci95[0] * 100}%; width: ${(headline.ci95[1] - headline.ci95[0]) * 100}%`}></span>
								<span class="performance-point {arm.family}" style={`left: ${headline.estimate * 100}%`}></span>
							</div>
							<strong class="performance-value">{headline.estimate.toFixed(3)}<small>{interval(headline)}</small></strong>
						</li>
					{/each}
				</ol>

			{#if headlineCostView && headlinePoints.length === headlineCostView.eligible_arm_ids.length}
				<h4>Cost × performance landscape</h4>
				<p class="chart-contract">Exact axes: mean-fold trapezoidal PR area × USD per 1,000 statements.</p>
				<div class="chart-meta">
					<span>{headlineCostView.basis} costs · {headlineCostView.view_id}</span>
					<span>{headlinePoints.length}/{arms.length} configurations cost-comparable</span>
					<span>{substrate.n_evaluable} statements per configuration</span>
					<span>retries + relation subcalls included</span>
					</div>
					<div class="landscape">
						<div class="chart-wrap">
							<p class="human-axis y">paper-compatible mean-fold PR area (higher is better)</p>
							<svg viewBox="0 0 760 330" role="img" aria-labelledby="landscape-chart-title landscape-chart-desc">
								<title id="landscape-chart-title">Mean-fold trapezoidal PR area versus cost per 1,000 statements</title>
							<desc id="landscape-chart-desc">Arms with mutually comparable cost under the named ledger view. Performance uses the same statements and gold labels. Higher and farther left is better.</desc>
								{#each chartTicks as tick}
									<line class="grid" x1="68" y1={chartY(tick)} x2="720" y2={chartY(tick)} />
									<text class="tick y" x="58" y={chartY(tick) + 4}>{tick.toFixed(2)}</text>
									<line class="grid" x1={chartX(tick * headlineCostMax)} y1="18" x2={chartX(tick * headlineCostMax)} y2="280" />
									<text class="tick x" x={chartX(tick * headlineCostMax)} y="299">{money(tick * headlineCostMax)}</text>
								{/each}
								<line class="axis" x1="68" y1="280" x2="720" y2="280" />
								<line class="axis" x1="68" y1="18" x2="68" y2="280" />
								{#if headlineFrontierPoints.length > 1}
									<polyline
										class="frontier-line"
										points={headlineFrontierPoints.map((point) => `${chartX(point.cost)},${chartY(point.performance)}`).join(' ')}
									/>
								{/if}
								{#each headlinePoints as point (point.arm.arm_id)}
									<line class="cost-whisker" x1={chartX(point.costLower)} y1={chartY(point.performance)} x2={chartX(point.cost)} y2={chartY(point.performance)} />
									<line class="ci-whisker" x1={chartX(point.cost)} y1={chartY(point.ci95[1])} x2={chartX(point.cost)} y2={chartY(point.ci95[0])} />
									<line class="ci-cap" x1={chartX(point.cost) - 5} y1={chartY(point.ci95[1])} x2={chartX(point.cost) + 5} y2={chartY(point.ci95[1])} />
									<line class="ci-cap" x1={chartX(point.cost) - 5} y1={chartY(point.ci95[0])} x2={chartX(point.cost) + 5} y2={chartY(point.ci95[0])} />
									<g class="chart-point {point.arm.family}" transform={`translate(${chartX(point.cost)} ${chartY(point.performance)})`}>
										<circle r="12" class:frontier={point.pointPareto}>
											<title>{point.arm.label}: mean-fold PR area {point.performance.toFixed(3)}, {money(point.costLower)}–{money(point.cost)} per 1,000 statements</title>
										</circle>
										<text y="4">{point.index}</text>
									</g>
								{/each}
								<text class="axis-label x" x="394" y="324">retry-inclusive accounted cost interval / 1,000 →</text>
								<text class="axis-label y" transform="translate(15 149) rotate(-90)">mean-fold PR area →</text>
							</svg>
						</div>
					</div>
				{#if headlineCostOmissions.length}
					<p class="chart-note cost-omissions">
						<strong>Cost view excludes {headlineCostOmissions.map((arm) => arm.label).join(' · ')}.</strong>
						Their performance remains in the complete ranking above; absent or differently based cost is not treated as zero. The line connects only this view's point-estimate frontier.
					</p>
				{:else}
					<p class="chart-note">Every plotted configuration shares this cost view and comparison contract. The line uses conservative upper endpoints; uncertainty dominance also requires a challenger upper endpoint no greater than the candidate lower endpoint.</p>
				{/if}
			{:else}
				<div class="cost-gate" role="status">
					<strong>Performance is ready; no multi-configuration cost view is.</strong>
					<span>The ranking above is the head-to-head result. Cost × performance needs at least two configurations with retry-inclusive costs under one comparable view.</span>
				</div>
				{/if}
			</section>

			{#if substrate.excluded_arms.length}
				<section class="exclusions" aria-labelledby="exclusions-title">
					<h3 id="exclusions-title">Relevant configurations not plotted</h3>
					<p>These configurations remain in scope. They are shown as exclusions instead of being silently omitted or assigned invented predictions.</p>
					{#each substrate.excluded_arms as excluded (excluded.arm_id)}
						<article>
							<header><span class="family {excluded.family}">{excluded.family}</span><strong>{excluded.label}</strong></header>
							<p>{excluded.reason}</p>
							<small>required: {excluded.required_artifact} · provenance: {excluded.provenance}</small>
						</article>
					{/each}
				</section>
			{/if}

			<details class="analysis-drilldown">
				<summary>
					<span>Detailed analysis</span>
					<small>metrics · calibration · thresholds · paired deltas · provenance</small>
				</summary>
				<div class="drilldown-content">
			<div class="digest-strip" aria-label="frozen contract identities">
				<span title={substrate.contract.substrate_sha256}>substrate {shortSha(substrate.contract.substrate_sha256)}</span>
				<span title={substrate.contract.gold_sha256}>gold {shortSha(substrate.contract.gold_sha256)}</span>
				<span title={substrate.contract.evaluation_set_sha256}>evaluation {shortSha(substrate.contract.evaluation_set_sha256)}</span>
				<span>{substrate.pr_summary_contract.fold_count} frozen folds</span>
			</div>

			<div class="table-scroll">
				<table class="metrics">
					<thead>
						<tr>
							<th>configuration</th>
							<th>PR area <span>fold-mean trapezoid · 95% CI</span></th>
							<th>avg precision <span>pooled · correct + · 95% CI</span></th>
							<th>AUROC <span>95% CI</span></th>
							<th>Brier <span>95% CI</span></th>
							<th>log loss <span>95% CI</span></th>
							<th>calibration <span>ECE · intercept · slope · target distances</span></th>
							<th>coverage</th>
							<th>cost <span>measured + reserve · interval · /1k · view</span></th>
						</tr>
					</thead>
					<tbody>
						{#each arms as arm (arm.arm_id)}
							<tr>
								<td class="arm">
									<span class="family {arm.family}">{arm.family}</span>{arm.label}
									{#if arm.pareto.status === 'available'}
										<small class="frontier-flags">
											{arm.pareto.point_pareto ? 'point frontier' : 'point dominated'} ·
											{arm.pareto.uncertainty_pareto ? 'uncertainty frontier' : 'robustly dominated'}
										</small>
									{/if}
								</td>
								<td title={arm.metrics.fold_mean_trapezoidal_pr_auc.method}>
									<strong>{estimate(arm.metrics.fold_mean_trapezoidal_pr_auc)}</strong>
									<small>fold SD {arm.metrics.fold_mean_trapezoidal_pr_auc.fold_population_sd.toFixed(3)} · CI {interval(arm.metrics.fold_mean_trapezoidal_pr_auc)}</small>
								</td>
								<td title={arm.metrics.pooled_average_precision.method}><strong>{estimate(arm.metrics.pooled_average_precision)}</strong><small>{interval(arm.metrics.pooled_average_precision)}</small></td>
								<td title={arm.metrics.auroc.method}><strong>{estimate(arm.metrics.auroc)}</strong><small>{interval(arm.metrics.auroc)}</small></td>
								<td title={arm.metrics.brier.method}><strong>{estimate(arm.metrics.brier)}</strong><small>{interval(arm.metrics.brier)}</small></td>
								<td title={arm.metrics.log_loss.method}><strong>{estimate(arm.metrics.log_loss)}</strong><small>{interval(arm.metrics.log_loss)}</small></td>
								<td class="cal">
									<strong>{estimate(arm.metrics.calibration.ece)}</strong>
									<small>i {estimate(arm.metrics.calibration.intercept)} · s {estimate(arm.metrics.calibration.slope)}</small>
									<small>|i| {estimate(arm.metrics.calibration.intercept_abs_error)} · |s−1| {estimate(arm.metrics.calibration.slope_abs_error)}</small>
								</td>
								<td><strong>{coverage(arm)}</strong><small>invalid {arm.coverage.invalid}</small></td>
								<td>
									{#if arm.cost.status === 'available'}
										<strong title={`exact lower ${arm.cost.inference_usd_lower_exact}; exact upper ${arm.cost.inference_usd_upper_exact}`}>{money(arm.cost.inference_usd_lower)}–{money(arm.cost.inference_usd_upper)}</strong>
										{#if arm.cost.provider_measured_usd_total !== null && arm.cost.conservative_reserved_usd_total !== null}<small>{money(arm.cost.provider_measured_usd_total)} measured + {money(arm.cost.conservative_reserved_usd_total)} reserved</small>{/if}
										<small>{money(arm.cost.usd_per_1k_statements_lower)}–{money(arm.cost.usd_per_1k_statements_upper)} /1k · {arm.cost.basis} / {arm.cost.view_id}</small>
										<small>{arm.cost.execution_count} executions · {arm.cost.attempt_count} attempts · {arm.cost.retry_attempt_count} retries</small>
										{#if arm.cost.provider_measured_call_count !== null && arm.cost.conservative_call_count !== null}<small>{arm.cost.provider_measured_call_count} measured calls · {arm.cost.conservative_call_count} conservative</small>{/if}
										<small>{arm.cost.pricing.provider} · {arm.cost.pricing.provider_model_id} · {arm.cost.pricing.region}</small>
										<small>${arm.cost.pricing.tariff.input_usd_per_million} input / ${arm.cost.pricing.tariff.output_usd_per_million} output per 1M tokens · on-demand Standard/default</small>
										<small>{arm.cost.projection === 'all_executions' ? 'all observed executions' : 'observed reader subset'} · shared run; panel costs are non-additive</small>
									{:else}
										<strong>unavailable</strong><small title={arm.cost.reason}>{arm.cost.reason}</small>
									{/if}
								</td>
							</tr>
						{/each}
					</tbody>
				</table>
			</div>

			<section class="calibration-curves" aria-labelledby="calibration-title">
				<header>
					<h3 id="calibration-title">calibration reliability</h3>
					<p>When it says 80%, is it right about 80%? Frozen bins answer that directly; the diagonal is perfect calibration.</p>
				</header>
				<div class="curve-grid">
					{#each arms as arm (arm.arm_id)}
						<figure>
							<figcaption><span class="family {arm.family}">{arm.family}</span>{arm.label}</figcaption>
							<svg viewBox="0 0 120 120" role="img" aria-label="Reliability curve for {arm.label}">
								<line class="axis" x1="10" y1="110" x2="110" y2="110" />
								<line class="axis" x1="10" y1="110" x2="10" y2="10" />
								<line class="ideal" x1="10" y1="110" x2="110" y2="10" />
								{#each arm.metrics.calibration.reliability_bins.filter((bin) => bin.n > 0) as bin}
									<circle
										cx={calibrationX(bin.mean_prediction ?? 0)}
										cy={calibrationY(bin.observed_fraction ?? 0)}
										r={pointRadius(bin.n, substrate.n_evaluable)}
									>
										<title>{bin.n} statements · predicted {(bin.mean_prediction ?? 0).toFixed(3)} · observed {(bin.observed_fraction ?? 0).toFixed(3)}</title>
									</circle>
								{/each}
							</svg>
							<small>ECE {estimate(arm.metrics.calibration.ece)}</small>
						</figure>
					{/each}
				</div>
			</section>

			<section class="thresholds" aria-labelledby="threshold-title">
				<h3 id="threshold-title">pre-frozen operating points</h3>
				<p>Secondary metrics appear only where the arm’s threshold source was frozen outside this evaluation gold.</p>
				<div class="table-scroll">
					<table>
						<thead><tr><th>arm</th><th>threshold</th><th>accuracy</th><th>precision</th><th>recall</th><th>F1</th><th>released-negative assumption errors</th><th>source</th></tr></thead>
						<tbody>
							{#each arms as arm (arm.arm_id)}
								<tr>
									<td>{arm.label}</td>
									{#if arm.metrics.threshold.status === 'available'}
										<td>≥ {arm.metrics.threshold.value.toFixed(3)}</td>
										<td>{estimate(arm.metrics.threshold.metrics.accuracy)}</td>
										<td>{estimate(arm.metrics.threshold.metrics.precision)}</td>
										<td>{estimate(arm.metrics.threshold.metrics.recall)}</td>
										<td>{estimate(arm.metrics.threshold.metrics.f1)}</td>
										<td>
											{#if arm.released_label_error_strata}
												{arm.released_label_error_strata.released_negative_assumption.fp} FP / {arm.released_label_error_strata.released_negative_assumption.statements}
												<small>{arm.released_label_error_strata.released_negative_assumption.errors} of {arm.released_label_error_strata.strict_e0_resolved.errors + arm.released_label_error_strata.released_negative_assumption.errors} released-label errors</small>
											{:else}—{/if}
										</td>
										<td title={arm.metrics.threshold.source_sha256}>{arm.metrics.threshold.source_path}</td>
									{:else}
										<td colspan="7">unavailable · {arm.metrics.threshold.reason}</td>
									{/if}
								</tr>
							{/each}
						</tbody>
					</table>
				</div>
			</section>

			<section class="deltas" aria-labelledby="delta-title">
				<header class="section-head">
					<div>
						<h3 id="delta-title">paired deltas</h3>
						<p>Second arm minus first arm, using identical statement-bootstrap draws.</p>
					</div>
					<label>
						metric
						<select value={selectedMetric} onchange={(event) => selectMetric(event.currentTarget.value as BeliefMetricKey)}>
							{#each comparisonMetrics as metric}<option value={metric}>{metricLabels[metric]}</option>{/each}
						</select>
					</label>
				</header>
				<table>
					<thead><tr><th>comparison</th><th>metric</th><th>Δ</th><th>95% CI</th><th>interpretation</th><th>valid / requested</th></tr></thead>
					<tbody>
						{#each visibleComparisons as row}
							<tr>
								<td>{armLabel(row.a_arm_id)} → {armLabel(row.b_arm_id)}</td>
								<td>{metricLabels[row.metric]}</td>
								<td>{signed(row.delta.estimate)}</td>
								<td>[{signed(row.delta.ci95[0])}, {signed(row.delta.ci95[1])}]</td>
								<td>{row.better_when.replaceAll('_', ' ')}</td>
								<td>{row.delta.valid_resamples.toLocaleString()} / {row.delta.resamples.toLocaleString()}</td>
							</tr>
						{/each}
					</tbody>
				</table>
			</section>

			<section class="pareto" aria-labelledby="pareto-title">
				<h3 id="pareto-title">cost × paper-compatible mean-fold PR frontiers</h3>
				<p>The point frontier uses conservative upper cost endpoints. The uncertainty frontier removes a configuration only when the challenger upper endpoint is no greater than the candidate lower endpoint and paired performance-CI dominance also holds.</p>
				{#if substrate.pareto.views.length === 0}
					<p class="empty">No configuration has a publication-ready cost ledger.</p>
				{:else}
					{#each substrate.pareto.views as view (view.view_id)}
						<article>
							<header><strong>{view.view_id}</strong><span>{view.basis} costs · {view.eligible_arm_ids.length} eligible</span></header>
							<dl>
								<div><dt>point frontier</dt><dd>{armLabels(view.point_frontier_arm_ids)}</dd></div>
								<div><dt>uncertainty frontier</dt><dd>{armLabels(view.uncertainty_frontier_arm_ids)}</dd></div>
							</dl>
							{#if view.audit.length}
								<details>
									<summary>ordered-pair dominance audit ({view.audit.length})</summary>
									<div class="table-scroll">
										<table>
											<thead><tr><th>challenger → candidate</th><th>challenger cost interval</th><th>candidate cost interval</th><th>Δ upper cost /1k</th><th>interval no worse?</th><th>Δ mean-fold PR</th><th>paired 95% CI</th><th>point</th><th>uncertainty</th></tr></thead>
											<tbody>
												{#each view.audit as audit}
													<tr>
													<td>{armLabel(audit.challenger_arm_id)} → {armLabel(audit.candidate_arm_id)}</td>
													<td>{money(audit.challenger_cost_per_1k_interval[0])}–{money(audit.challenger_cost_per_1k_interval[1])}</td>
													<td>{money(audit.candidate_cost_per_1k_interval[0])}–{money(audit.candidate_cost_per_1k_interval[1])}</td>
													<td>{signed(audit.challenger_minus_candidate_cost_per_1k)}</td>
													<td>{audit.cost_interval_definitely_not_worse ? 'yes' : 'no'}</td>
														<td>{signed(audit.challenger_minus_candidate_performance)}</td>
														<td>[{signed(audit.performance_delta_ci95[0])}, {signed(audit.performance_delta_ci95[1])}]</td>
														<td>{audit.point_dominates ? 'dominates' : 'no'}</td>
														<td>{audit.uncertainty_dominates ? 'dominates' : 'no'}</td>
													</tr>
												{/each}
											</tbody>
										</table>
									</div>
								</details>
							{/if}
						</article>
					{/each}
				{/if}
			</section>

			<section class="provenance" aria-labelledby="provenance-title">
				<h3 id="provenance-title">prediction and cost provenance</h3>
				{#each arms as arm (arm.arm_id)}
					<details>
						<summary><span class="family {arm.family}">{arm.family}</span>{arm.label}</summary>
						<dl>
							<div><dt>implementation</dt><dd>{arm.provenance.implementation}</dd></div>
							<div><dt>implementation digest</dt><dd><code>{arm.provenance.implementation_digest}</code></dd></div>
							<div><dt>predictions</dt><dd><code>{arm.provenance.predictions_path} · {arm.provenance.predictions_sha256}</code></dd></div>
							<div><dt>training data digest</dt><dd><code>{arm.provenance.training_data_sha256 ?? 'unknown or not applicable'}</code></dd></div>
							<div><dt>environment</dt><dd>{arm.provenance.environment}</dd></div>
							{#if arm.cost.status === 'available'}
								<div><dt>cost ledger</dt><dd><code>{arm.cost.ledger_path} · {arm.cost.ledger_sha256}</code></dd></div>
								<div><dt>price source</dt><dd><a href={arm.cost.price_source}>{arm.cost.pricing.provider} pricing</a> · retrieved {arm.cost.price_date}</dd></div>
								<div><dt>provider tariff</dt><dd>{arm.cost.pricing.provider_model_id} · ${arm.cost.pricing.tariff.input_usd_per_million} input / ${arm.cost.pricing.tariff.output_usd_per_million} output per 1M tokens · {arm.cost.pricing.region} · on-demand Standard/default</dd></div>
								<div><dt>cost projection</dt><dd>{arm.cost.projection === 'all_executions' ? 'all observed executions' : 'observed execution subset'} · shared run <code>{arm.cost.shared_run_id}</code> · not additive across panels</dd></div>
								<div><dt>cost comparability</dt><dd><code>{arm.cost.cost_comparability_id}</code></dd></div>
								<div><dt>cost denominator</dt><dd>{arm.cost.denominator.statements} assembled statements · {arm.cost.denominator.evidence_executions} evidence executions</dd></div>
								<div><dt>cost scope</dt><dd>includes {arm.cost.scope.included_cost_categories.join(', ')}; excludes {arm.cost.scope.excluded_cost_categories.join(', ')}</dd></div>
							{/if}
							{#if arm.provenance.notes}<div><dt>notes</dt><dd>{arm.provenance.notes}</dd></div>{/if}
						</dl>
					</details>
				{/each}
			</section>

			<footer class="artifact">
				Frozen {comparison.frozen_at ?? '—'} · <code>{comparison.artifact_path}</code> · artifact
				<span title={comparison.artifact_sha256 ?? ''}>{shortSha(comparison.artifact_sha256)}</span> ·
				metrics code <span title={comparison.provenance?.metrics_code_sha256 ?? ''}>{shortSha(comparison.provenance?.metrics_code_sha256 ?? null)}</span> ·
				source manifest <span title={comparison.provenance?.source_manifest_sha256 ?? ''}>{shortSha(comparison.provenance?.source_manifest_sha256 ?? null)}</span> ·
				scorer registry <span title={comparison.provenance?.scorer_registry.sha256 ?? ''}>{shortSha(comparison.provenance?.scorer_registry.sha256 ?? null)}</span> ·
				{comparison.provenance?.bootstrap_resamples.toLocaleString() ?? '—'} paired resamples
			</footer>
				</div>
			</details>
		{/if}
	{/if}
</section>

<style>
	.belief { margin-top: 0.2rem; }
	.gate { border: 1px solid var(--rule); border-left: 3px solid var(--blocked); padding: 1.2rem 1.35rem; max-width: 72ch; background: color-mix(in srgb, var(--blocked) 4%, transparent); }
	.gate h2, .panel-head h2 { font-family: var(--serif); font-size: 1.35rem; font-weight: 400; margin: 0.15rem 0 0.6rem; }
	.gate p, .gate li { font-family: var(--serif); font-size: 0.9rem; line-height: 1.5; color: var(--ink-muted); }
	.gate-intro { margin-bottom: 0.75rem; }
	.readiness { list-style: none; padding: 0 !important; margin: 0; border-top: 1px solid var(--rule); }
	.readiness li { display: flex; justify-content: space-between; gap: 1rem; padding: 0.45rem 0; border-bottom: 1px dotted var(--rule); font-family: var(--mono) !important; font-size: 0.7rem !important; }
	.readiness strong { color: var(--blocked); font-weight: 500; text-align: right; }
	.gate-details { margin-top: 0.8rem; font-family: var(--mono); font-size: 0.7rem; color: var(--ink-muted); }
	.gate-details summary { cursor: pointer; color: var(--ink); }
	.diagnostic-banner { display: grid; gap: 0.2rem; margin: 0 0 1rem; padding: 0.75rem 0.9rem; border: 2px solid var(--blocked); background: color-mix(in srgb, var(--blocked) 7%, var(--paper)); font-family: var(--mono); font-size: 0.68rem; color: var(--ink-muted); }
	.diagnostic-banner strong { color: var(--blocked); text-transform: uppercase; letter-spacing: 0.035em; }
	.descriptive-card { margin-top: 1rem; padding: 0.9rem 1rem; border: 1px solid var(--rule); border-left: 3px solid var(--blocked); background: color-mix(in srgb, var(--blocked) 3%, transparent); }
	.descriptive-card h3 { margin: 0.18rem 0 0; font-family: var(--serif); font-size: 1.2rem; font-weight: 400; }
	.descriptive-card > p:last-child { margin: 0.65rem 0 0; max-width: 78ch; font-family: var(--serif); font-size: 0.8rem; line-height: 1.45; color: var(--ink-muted); }
	.gold-audit, .strict-sensitivity { margin-top: 1rem; padding: 0.9rem 1rem; border: 1px solid var(--rule); }
	.gold-audit { display: grid; grid-template-columns: minmax(0, 1fr) auto; gap: 1.2rem; border-left: 3px solid var(--blocked); }
	.gold-audit h3, .strict-sensitivity h3 { margin: 0.18rem 0 0.5rem; font-family: var(--serif); font-size: 1.1rem; font-weight: 400; }
	.gold-audit p, .strict-sensitivity > p { margin: 0.25rem 0; max-width: 86ch; font-family: var(--serif); font-size: 0.78rem; line-height: 1.45; color: var(--ink-muted); }
	.gold-audit dl { display: grid; gap: 0.45rem; min-width: 12rem; margin: 0; }
	.gold-audit dl div { display: grid; grid-template-columns: 1fr auto; gap: 1rem; border-bottom: 1px dotted var(--rule); }
	.gold-audit dt, .gold-audit dd { font-family: var(--mono); font-size: 0.66rem; }
	.gold-audit dt { color: var(--ink-faint); }
	.gold-audit dd { margin: 0; color: var(--ink); }
	.strict-sensitivity { border-left: 3px solid var(--accent); }
	.strict-sensitivity > header { display: flex; align-items: start; justify-content: space-between; gap: 1rem; }
	.strict-sensitivity > header > strong { font-family: var(--mono); font-size: 0.64rem; font-weight: 500; color: var(--accent); }
	.strict-sensitivity .table-scroll { margin-top: 0.7rem; }
	.strict-sensitivity details { margin-top: 0.75rem; border-top: 1px dotted var(--rule); padding-top: 0.55rem; }
	.strict-sensitivity summary { cursor: pointer; font-family: var(--mono); font-size: 0.68rem; color: var(--ink-muted); }
	.reader-bridge { margin-top: 1rem; padding: 0.9rem 1rem; border: 1px solid var(--rule); border-left: 3px solid var(--accent); background: var(--accent-wash); }
	.reader-bridge > header { display: flex; justify-content: space-between; align-items: start; gap: 1rem; }
	.reader-bridge h3 { margin: 0.18rem 0 0; font-family: var(--serif); font-size: 1.2rem; font-weight: 400; }
	.reader-bridge > header > strong { flex: 0 0 auto; font-family: var(--mono); font-size: 0.62rem; font-weight: 500; color: var(--accent); text-transform: uppercase; letter-spacing: 0.035em; }
	.bridge-contract, .bridge-caveat { margin: 0.65rem 0; max-width: 86ch; font-family: var(--serif); font-size: 0.8rem; line-height: 1.45; color: var(--ink-muted); }
	.bridge-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(13rem, 1fr)); gap: 0.7rem; margin: 0.8rem 0; }
	.bridge-grid article { min-width: 0; padding: 0.65rem 0.7rem; border: 1px solid var(--rule); background: color-mix(in srgb, var(--paper) 92%, transparent); }
	.bridge-grid span, .bridge-grid small, .bridge-grid em { display: block; font-family: var(--mono); }
	.bridge-grid span { font-size: 0.62rem; color: var(--ink-faint); text-transform: uppercase; letter-spacing: 0.035em; }
	.bridge-grid span i { font-style: normal; color: var(--blocked); }
	.bridge-grid article > strong { display: block; margin-top: 0.24rem; font-family: var(--mono); font-size: 1.2rem; font-weight: 500; font-variant-numeric: tabular-nums; }
	.bridge-grid small { margin-top: 0.15rem; font-size: 0.61rem; line-height: 1.4; color: var(--ink-faint); }
	.bridge-grid em { margin-top: 0.5rem; padding-top: 0.4rem; border-top: 1px dotted var(--rule); font-size: 0.65rem; font-style: normal; color: var(--ink); font-variant-numeric: tabular-nums; }
	.bridge-caveat { margin-bottom: 0; }
	.bridge-caveat strong { color: var(--ink); }
	.gate code, .artifact code { font-family: var(--mono); font-size: 0.72rem; color: var(--ink-faint); }
	.eyebrow { font-family: var(--mono) !important; font-size: 0.68rem !important; text-transform: uppercase; letter-spacing: 0.05em; color: var(--ink-faint) !important; margin: 0 !important; }
	.panels { display: flex; flex-wrap: wrap; gap: 0.65rem; margin-bottom: 1.25rem; }
	.panels button { background: none; border: 1px solid var(--rule); padding: 0.5rem 0.7rem; font-family: var(--mono); color: var(--ink-muted); text-align: left; cursor: pointer; }
	.panels button small { display: block; font-size: 0.64rem; color: var(--ink-faint); margin-top: 0.15rem; }
	.panels button.on { border-color: var(--accent); color: var(--ink); background: var(--accent-wash); }
	.panel-head { display: flex; justify-content: space-between; align-items: end; gap: 1.5rem; border-bottom: 1px solid var(--rule); padding-bottom: 0.8rem; }
	.panel-head dl { display: flex; flex-wrap: wrap; gap: 1.2rem; margin: 0; }
	.panel-head dl div { display: flex; flex-direction: column; }
	.panel-head dt, .provenance dt, .pareto dt { font-family: var(--mono); font-size: 0.62rem; color: var(--ink-faint); text-transform: lowercase; }
	.panel-head dd { font-family: var(--mono); font-size: 0.78rem; margin: 0.15rem 0 0; }
	.headline-comparison { margin-top: 1rem; border: 1px solid var(--rule); padding: 1rem; }
	.headline-copy { display: flex; align-items: baseline; flex-wrap: wrap; gap: 0.3rem 1.1rem; }
	.headline-copy h3 { order: 2; margin: 0; font-family: var(--serif); font-size: 1.25rem; font-weight: 400; }
	.headline-copy > p:last-child { order: 3; flex-basis: 100%; margin: 0.15rem 0 0; font-family: var(--serif); font-size: 0.82rem; color: var(--ink-muted); }
	.contract-pass { order: 1; flex-basis: 100%; margin: 0; font-family: var(--mono); font-size: 0.7rem; color: var(--ok-green); letter-spacing: 0.02em; }
	.contract-pass span { font-weight: 700; }
	.performance-label { margin: 0.9rem 0 0.35rem; font-family: var(--mono); font-size: 0.7rem; color: var(--ink); }
	.performance-scale { display: grid; grid-template-columns: 1fr auto 1fr; align-items: center; margin-left: min(19rem, 36%); font-family: var(--mono); font-size: 0.58rem; color: var(--ink-faint); }
	.performance-scale span:last-child { text-align: right; }
	.performance-strip { list-style: none; padding: 0; margin: 0.2rem 0 0; border-top: 1px solid var(--rule); }
	.performance-strip li { display: grid; grid-template-columns: 1.7rem minmax(10rem, 16rem) minmax(12rem, 1fr) 7.5rem; gap: 0.65rem; align-items: center; min-height: 2.8rem; border-bottom: 1px dotted var(--rule); }
	.performance-arm strong { display: block; font-family: var(--mono); font-size: 0.72rem; font-weight: 500; }
	.performance-arm small { display: block; margin-top: 0.18rem; font-family: var(--mono); font-size: 0.6rem; color: var(--ink-faint); }
	.performance-track { position: relative; height: 1.2rem; background: linear-gradient(to right, transparent calc(50% - 0.5px), var(--rule) 50%, transparent calc(50% + 0.5px)); border-left: 1px solid var(--rule); border-right: 1px solid var(--rule); }
	.performance-ci { position: absolute; top: calc(50% - 1.5px); height: 3px; min-width: 2px; background: var(--ink-muted); }
	.performance-ci::before, .performance-ci::after { content: ''; position: absolute; top: -3px; width: 1px; height: 9px; background: var(--ink-muted); }
	.performance-ci::before { left: 0; }
	.performance-ci::after { right: 0; }
	.performance-point { position: absolute; top: calc(50% - 5px); width: 10px; height: 10px; margin-left: -5px; border: 2px solid var(--ink-muted); border-radius: 50%; background: var(--paper); }
	.performance-point.current, .performance-point.llm { border-color: var(--accent); }
	.performance-point.llm { background: var(--accent-wash); }
	.performance-value { font-family: var(--mono); font-size: 0.78rem; font-weight: 500; text-align: right; }
	.performance-value small { display: block; margin-top: 0.12rem; font-size: 0.6rem; font-weight: 400; color: var(--ink-faint); }
	.headline-comparison h4 { margin: 1rem 0 0; padding-top: 0.75rem; border-top: 1px solid var(--rule); font-family: var(--serif); font-size: 1rem; font-weight: 400; }
	.chart-contract { margin: 0.15rem 0 0; font-family: var(--mono); font-size: 0.62rem; color: var(--ink-faint); }
	.chart-meta { display: flex; flex-wrap: wrap; gap: 0.45rem 1rem; margin: 0.75rem 0 0.35rem; font-family: var(--mono); font-size: 0.64rem; color: var(--ink-faint); }
	.chart-meta span + span::before { content: '·'; margin-right: 1rem; }
	.landscape { min-width: 0; }
	.chart-wrap { min-width: 0; }
	.human-axis { margin: 0 0 -0.25rem 3rem; font-family: var(--mono); font-size: 0.62rem; color: var(--ink-muted); }
	.chart-wrap svg { display: block; width: 100%; min-height: 285px; overflow: visible; }
	.chart-wrap .grid { stroke: var(--rule); stroke-width: 0.7; }
	.chart-wrap .axis { stroke: var(--ink-muted); stroke-width: 1; }
	.chart-wrap .tick, .chart-wrap .axis-label { fill: var(--ink-faint); font-family: var(--mono); font-size: 10px; }
	.chart-wrap .tick.x { text-anchor: middle; }
	.chart-wrap .tick.y { text-anchor: end; }
	.chart-wrap .axis-label { text-anchor: middle; font-size: 11px; }
	.frontier-line { fill: none; stroke: var(--accent); stroke-width: 1.5; stroke-dasharray: 4 4; opacity: 0.65; }
	.ci-whisker, .ci-cap { stroke: var(--ink-muted); stroke-width: 1.2; }
	.cost-whisker { stroke: var(--accent); stroke-width: 3; opacity: 0.55; }
	.chart-point circle { fill: var(--paper); stroke: var(--ink-muted); stroke-width: 1.5; }
	.chart-point.current circle { stroke: var(--accent); }
	.chart-point.llm circle { fill: var(--accent-wash); stroke: var(--accent); }
	.chart-point circle.frontier { stroke-width: 3; }
	.chart-point text { fill: var(--ink); font-family: var(--mono); font-size: 9px; text-anchor: middle; pointer-events: none; }
	.point-key { display: grid; place-items: center; width: 1.45rem; height: 1.45rem; border: 2px solid var(--ink-muted); border-radius: 50%; font-family: var(--mono); font-size: 0.65rem; }
	.point-key.current, .point-key.llm { border-color: var(--accent); }
	.point-key.llm { background: var(--accent-wash); }
	.chart-note { margin: 0.5rem 0 0; padding-top: 0.5rem; border-top: 1px solid var(--rule); font-family: var(--serif); font-size: 0.74rem; line-height: 1.45; color: var(--ink-muted); }
	.chart-note.cost-omissions { border-top-color: var(--blocked); }
	.chart-note strong { color: var(--ink); font-weight: 500; }
	.cost-gate { display: grid; gap: 0.25rem; margin-top: 0.8rem; padding: 0.75rem; border-left: 3px solid var(--blocked); background: color-mix(in srgb, var(--blocked) 4%, transparent); font-family: var(--mono); font-size: 0.7rem; color: var(--ink-muted); }
	.cost-gate strong { color: var(--ink); font-weight: 500; }
	.exclusions { margin-top: 1rem; padding: 0.85rem 1rem; border: 1px dashed var(--rule); }
	.exclusions > h3 { margin: 0; font-family: var(--serif); font-size: 1rem; font-weight: 400; }
	.exclusions > p { margin: 0.2rem 0 0.7rem; font-family: var(--serif); font-size: 0.78rem; line-height: 1.45; color: var(--ink-muted); }
	.exclusions article { padding: 0.55rem 0; border-top: 1px dotted var(--rule); }
	.exclusions article header { display: flex; align-items: baseline; gap: 0.1rem; font-family: var(--mono); font-size: 0.72rem; }
	.exclusions article header strong { font-weight: 500; }
	.exclusions article p { margin: 0.25rem 0; font-family: var(--serif); font-size: 0.76rem; line-height: 1.4; color: var(--ink-muted); }
	.exclusions article small { display: block; overflow-wrap: anywhere; font-family: var(--mono); font-size: 0.61rem; color: var(--ink-faint); }
	.analysis-drilldown { margin-top: 1rem; border-top: 1px solid var(--rule); border-bottom: 1px solid var(--rule); }
	.analysis-drilldown > summary { display: flex; justify-content: space-between; gap: 1rem; padding: 0.7rem 0; cursor: pointer; font-family: var(--mono); font-size: 0.75rem; color: var(--ink); }
	.analysis-drilldown > summary small { color: var(--ink-faint); font-size: 0.65rem; }
	.drilldown-content { padding-bottom: 0.8rem; }
	.digest-strip { display: flex; flex-wrap: wrap; gap: 1rem; font-family: var(--mono); font-size: 0.67rem; color: var(--ink-faint); padding: 0.55rem 0 0.85rem; }
	.table-scroll { overflow-x: auto; }
	table { width: 100%; border-collapse: collapse; font-family: var(--mono); font-size: 0.75rem; }
	th { text-align: left; font-weight: 400; font-size: 0.62rem; color: var(--ink-faint); border-bottom: 1px solid var(--rule); padding: 0 0.8rem 0.4rem 0; white-space: nowrap; }
	th span { display: block; font-size: 0.9em; }
	td { padding: 0.55rem 0.8rem 0.55rem 0; border-bottom: 1px dotted var(--rule); vertical-align: top; font-variant-numeric: tabular-nums; white-space: nowrap; }
	td strong, td small { display: block; }
	td strong { font-weight: 500; }
	td small { font-size: 0.66rem; color: var(--ink-faint); max-width: 26ch; overflow: hidden; text-overflow: ellipsis; }
	.arm { white-space: normal; min-width: 13rem; }
	.frontier-flags { margin-top: 0.3rem; }
	.family { display: inline-block; font-family: var(--mono); font-size: 0.58rem; letter-spacing: 0.03em; text-transform: uppercase; padding: 0.08rem 0.25rem; margin-right: 0.45rem; border: 1px solid var(--rule); color: var(--ink-muted); }
	.family.current { border-color: var(--accent); }
	.family.llm { background: var(--accent-wash); }
	.calibration-curves, .thresholds, .deltas, .pareto, .provenance { margin-top: 1.8rem; }
	.calibration-curves h3, .thresholds h3, .deltas h3, .pareto h3, .provenance h3 { font-family: var(--serif); font-size: 1rem; font-weight: 400; margin: 0 0 0.2rem; }
	.calibration-curves header p, .thresholds > p, .deltas p, .pareto > p { font-family: var(--serif); font-size: 0.78rem; color: var(--ink-muted); margin: 0 0 0.7rem; }
	.curve-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(145px, 1fr)); gap: 0.8rem; }
	.curve-grid figure { margin: 0; border: 1px solid var(--rule); padding: 0.6rem; }
	.curve-grid figcaption { font-family: var(--mono); font-size: 0.68rem; min-height: 2rem; }
	.curve-grid svg { width: 100%; max-height: 155px; overflow: visible; }
	.curve-grid .axis { stroke: var(--rule); stroke-width: 1; }
	.curve-grid .ideal { stroke: var(--ink-faint); stroke-width: 0.8; stroke-dasharray: 3 3; }
	.curve-grid circle { fill: var(--accent); fill-opacity: 0.65; stroke: var(--ink); stroke-width: 0.6; }
	.curve-grid figure > small { display: block; text-align: center; font-family: var(--mono); color: var(--ink-faint); font-size: 0.64rem; }
	.section-head { display: flex; justify-content: space-between; align-items: end; gap: 1rem; }
	.section-head label { display: grid; gap: 0.2rem; font-family: var(--mono); font-size: 0.62rem; color: var(--ink-faint); }
	.section-head select { font: inherit; color: var(--ink); background: var(--paper); border: 1px solid var(--rule); padding: 0.35rem; }
	.pareto article { border: 1px solid var(--rule); padding: 0.7rem 0.85rem; margin-top: 0.65rem; }
	.pareto article > header { display: flex; justify-content: space-between; gap: 1rem; font-family: var(--mono); font-size: 0.72rem; }
	.pareto article > header span { color: var(--ink-faint); }
	.pareto dl { display: grid; gap: 0.35rem; margin: 0.65rem 0; }
	.pareto dl div { display: grid; grid-template-columns: 9rem 1fr; gap: 0.7rem; }
	.pareto dd { margin: 0; font-family: var(--mono); font-size: 0.72rem; }
	.pareto details, .provenance details { border-top: 1px dotted var(--rule); padding: 0.45rem 0; font-family: var(--mono); font-size: 0.75rem; }
	.pareto summary, .provenance summary { cursor: pointer; }
	.provenance dl { margin: 0.7rem 0 0.3rem 1.2rem; display: grid; gap: 0.45rem; }
	.provenance dl div { display: grid; grid-template-columns: 9rem minmax(0, 1fr); gap: 0.8rem; }
	.provenance dd { margin: 0; min-width: 0; overflow-wrap: anywhere; }
	.provenance code { font-size: 0.68rem; }
	.artifact { font-family: var(--mono); font-size: 0.66rem; color: var(--ink-faint); border-top: 1px solid var(--rule); margin-top: 1.6rem; padding-top: 0.55rem; }
	.empty { font-family: var(--serif); color: var(--ink-muted); }

	@media (max-width: 760px) {
		.panel-head, .section-head, .gold-audit, .strict-sensitivity > header { display: block; }
		.panel-head dl, .section-head label { margin-top: 0.8rem; }
		.gold-audit dl, .strict-sensitivity > header > strong { margin-top: 0.8rem; }
		.performance-scale { margin-left: 0; }
		.performance-strip li { grid-template-columns: 1.7rem minmax(8rem, 1fr) 6.2rem; padding: 0.45rem 0; }
		.performance-track { grid-column: 2 / -1; grid-row: 2; }
		.chart-wrap svg { min-height: 230px; }
		.analysis-drilldown > summary { display: block; }
		.analysis-drilldown > summary small { display: block; margin-top: 0.2rem; }
		.provenance dl div, .pareto dl div { grid-template-columns: 1fr; gap: 0.1rem; }
	}

	@media (max-width: 480px) {
		.chart-meta span + span::before { content: ''; margin: 0; }
	}
</style>
