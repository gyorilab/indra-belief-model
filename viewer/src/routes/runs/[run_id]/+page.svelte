<script lang="ts">
	import type { PageData } from './$types';
	import { fmtCost, fmtCostFull } from '$lib/format';
	import { goto } from '$app/navigation';
	import { page } from '$app/stores';
	import ReliabilityDiagram from '$lib/components/ReliabilityDiagram.svelte';
	import BrierBar from '$lib/components/BrierBar.svelte';
	import ConfusionMosaic from '$lib/components/ConfusionMosaic.svelte';
	import DriverMosaic from '$lib/components/DriverMosaic.svelte';
	import { armAvailable, type MetricArm, type TierBlock } from '$lib/data/types';
	import {
		calibrationArmLabel,
		calibrationContractLabel,
		canonicalCalibrationArm,
		HYBRID_METRICS_SCHEMA
	} from '$lib/data/calibration';
	import type { Tier } from '$lib/data/queries';

	let { data }: { data: PageData } = $props();
	const run = $derived(data.run);
	const v = $derived(data.validity);
	const residual = $derived(data.residual);
	const cal = $derived(data.calibration);
	const tier = $derived(data.tier);
	const profileState = $derived(run?.soft_calibration ?? null);

	// ── Calibration surface (C4) — every number served byte-exact from
	// metrics.json; the page derives nothing but display strings + the toggle. ──
	const metrics = $derived(cal?.metrics ?? null);

	// The Tier toggle is the ONE reactive lever. ?tier=ev|stmt.
	function setTier(t: Tier) {
		const p = new URLSearchParams($page.url.searchParams);
		if (t === 'ev') p.delete('tier');
		else p.set('tier', t);
		const qs = p.toString();
		goto(qs ? `?${qs}` : $page.url.pathname, { replaceState: false, keepFocus: true, noScroll: true });
	}

	// One tier block (the active one) + the OTHER tier, so both render stacked.
	const TIER_LABEL: Record<Tier, string> = {
		ev: 'Tier 1 — per evidence',
		stmt: 'Tier 2 — per statement'
	};
	const TIER_SUB: Record<Tier, string> = {
		ev: 'the realized per-evidence score vs human gold',
		stmt: 'rolled-up statement belief vs statement gold (any-incorrect-wins)'
	};

	function tierBlock(t: Tier): TierBlock | null {
		return metrics?.tiers?.[t] ?? null;
	}

	// The arms to show per tier, headline first. Tier-1 has one realized arm
	// (score); Tier-2 preserves all historical slots but labels/selects them by
	// metrics schema so v2 survival weights never masquerade as the v3 hybrid.
	const TIER_ARMS: Record<Tier, string[]> = { ev: ['score'], stmt: ['hard', 'parametric', 'soft'] };
	function armLabel(arm: string): string {
		return calibrationArmLabel(metrics, arm);
	}
	function armOf(block: TierBlock | null, arm: string): MetricArm | null {
		if (!block || block.status !== 'available') return null;
		const a = block.arms[arm];
		return armAvailable(a) ? a : null;
	}
	function armReason(block: TierBlock | null, arm: string): string | null {
		if (!block || block.status !== 'available') return null;
		const a = block.arms[arm];
		return a && 'status' in a ? a.reason : null;
	}
	function headlineArm(t: Tier): string {
		return canonicalCalibrationArm(metrics, t) ?? (t === 'ev' ? 'score' : 'hard');
	}

	// P5 delta string for the headline arm of a tier.
	function deltaStr(t: Tier): string | null {
		const d = cal?.delta?.[t];
		if (!d) return null;
		const sign = d.delta <= 0 ? '▼' : '▲'; // ECE: lower is better
		const arm = t === 'stmt' ? ` ${armLabel(d.arm)}` : '';
		return `${sign}${Math.abs(d.delta).toFixed(3)}${arm} vs ${d.prev_model} (${d.prev_run_id.slice(0, 8)})`;
	}
	function deltaImproved(t: Tier): boolean {
		const d = cal?.delta?.[t];
		return !!d && d.delta <= 0;
	}

	function f3c(n: number | null | undefined): string {
		return n == null ? '—' : n.toFixed(3);
	}

	// Tiers to render, active one first, both always shown stacked + labeled.
	const orderedTiers = $derived<Tier[]>(tier === 'stmt' ? ['stmt', 'ev'] : ['ev', 'stmt']);

	function fmtCount(n: number | null | undefined): string {
		return n == null ? '—' : n.toLocaleString('en-US');
	}

	function fmt3(n: number | null | undefined): string {
		return n == null ? '—' : n.toFixed(3);
	}

	function fmtSigned(n: number | null | undefined, digits = 3): string {
		if (n == null) return '—';
		const sign = n >= 0 ? '+' : '−';
		return `${sign}${Math.abs(n).toFixed(digits)}`;
	}

	// Verdict mix, ordered correct → incorrect → unscored → rest.
	const VERDICT_ORDER: Record<string, number> = { correct: 0, incorrect: 1, unscored: 9 };
	const verdicts = $derived(
		[...v.verdicts].sort(
			(a, b) => (VERDICT_ORDER[a.verdict] ?? 5) - (VERDICT_ORDER[b.verdict] ?? 5) || b.n - a.n
		)
	);
	const verdictTotal = $derived(verdicts.reduce((s, x) => s + x.n, 0));

	const bucketTotal = $derived(v.buckets.reduce((s, x) => s + x.n, 0));

	// Residual histogram: 11 bins spanning [-1, +1]; tallest bar = 100% height.
	const residualMax = $derived(residual ? Math.max(1, ...residual.bins) : 1);
	function binLabel(i: number, n: number): string {
		const lo = (-1 + (2 * i) / n).toFixed(2);
		const hi = (-1 + (2 * (i + 1)) / n).toFixed(2);
		return `${lo} … ${hi}`;
	}

	function verdictClass(verdict: string): string {
		if (verdict === 'correct') return 'v-correct';
		if (verdict === 'incorrect') return 'v-incorrect';
		return 'v-other';
	}

	// ── E11: statement verdict error-detection (schema v2 metrics) ──────────────
	// A different question than calibration: does the TIERED decision (correct /
	// review / incorrect) catch the statements gold says are wrong? verdict_err +
	// stratified residual ride on tiers.stmt from schema v2; legacy runs omit them.
	const stmtTier = $derived(metrics?.tiers?.stmt ?? null);
	const stmtVerdict = $derived(
		stmtTier && stmtTier.status === 'available' ? (stmtTier.verdict_err ?? null) : null
	);
	const stratified = $derived(
		stmtTier && stmtTier.status === 'available' ? (stmtTier.stratified ?? null) : null
	);

	const STRAT_AXES = [
		{ key: 'by_stmt_type', label: 'statement type' },
		{ key: 'by_n_evidence', label: 'evidence depth' },
		{ key: 'by_n_sources', label: 'source breadth' },
		{ key: 'by_dominant_bucket', label: 'bucket' }
	] as const;
	type StratAxis = (typeof STRAT_AXES)[number]['key'];
	let stratAxis = $state<StratAxis>('by_stmt_type');

	// Worst error-F1 first — the eye should land on where the verdict fails most.
	const sortedStrata = $derived.by(() => {
		const layer = stratified?.[stratAxis];
		if (!layer) return [];
		return Object.entries(layer)
			.map(([value, b]) => ({
				value,
				n: b.n,
				f1: b.verdict_err.f1,
				fp: b.verdict_err.fp,
				fn: b.verdict_err.fn
			}))
			.sort((a, b) => a.f1 - b.f1 || b.n - a.n);
	});

	function f1Class(f1: number): string {
		if (f1 >= 0.85) return 'f1-good';
		if (f1 >= 0.72) return 'f1-mid';
		return 'f1-weak';
	}
</script>

<svelte:head><title>{run.run_id.slice(0, 8)} · run · INDRA Belief</title></svelte:head>

<header>
	<div class="crumb">
		<a href="/runs">runs</a><span class="sep"> / </span><strong>{run.run_id.slice(0, 8)}</strong>
	</div>
	<div class="meta">
		<a class="nav-link" href="/statements">statements</a>
	</div>
</header>

<main id="main">
	<section class="run-meta">
		<h1 class="run-h">
			<span class="muted">run</span>
			<code>{run.run_id.slice(0, 8)}</code>
			<span class="muted">·</span>
			<span class="run-model">{run.model}</span>
			{#if run.status}
				<span class="run-status">{run.status}</span>
			{/if}
		</h1>
		<dl class="run-fields">
			<div><dt>model</dt><dd>{run.model}</dd></div>
			<div><dt>generated</dt><dd>{run.generated_date ?? '—'}</dd></div>
			<div><dt>statements</dt><dd>{fmtCount(run.n_statements)}</dd></div>
			<div><dt>evidences</dt><dd>{fmtCount(run.n_evidences)}</dd></div>
			<div><dt>scored evidence</dt><dd>{fmtCount(v.calibration.n)}</dd></div>
		</dl>
	</section>

	{#if run.cost && run.cost.status !== 'unavailable'}
		<section class="cost">
			<h2 class="sec-h">observed LLM cost</h2>
			<p class="sec-sub">
				Computed from observed token usage at export time. Local / self-hosted models
				are billed at $0.00; rows with an unverified price are excluded{#if run.cost.status === 'partial'}
					({fmtCount(run.cost.n_evidence_unavailable)} rows unavailable){/if}.
			</p>
			<dl class="stat-row">
				<div class="stat"><dt>total</dt><dd>{fmtCost(run.cost)}</dd></div>
				<div class="stat"><dt>per 1k LLM-scored evidence</dt><dd>{fmtCostFull(run.cost.usd_per_1k_evidence)}</dd></div>
				<div class="stat"><dt>input tokens</dt><dd>{fmtCount(run.cost.input_tokens)}</dd></div>
				<div class="stat"><dt>output tokens</dt><dd>{fmtCount(run.cost.output_tokens)}</dd></div>
			</dl>
			<p class="sec-sub muted">models: {run.cost.models.join(', ') || 'none (no LLM calls billed)'}</p>
		</section>
	{:else}
		<section class="cost unavailable">
			<h2 class="sec-h">observed LLM cost</h2>
			<p class="sec-sub">
				Cost unavailable — {run.cost
					? `${run.cost.models.join(', ') || 'this run'}'s per-token price is not verified`
					: 'this export predates cost capture'}. Token usage is recorded but no verified USD
				rate exists, so no dollar figure is shown (never $0 for an unpriced model).
			</p>
		</section>
	{/if}

	<!-- ── C4: run-on-its-own calibration surface (above the residual histogram) ── -->
	<section class="cal-surface" aria-label="calibration vs human gold">
		<h2 class="sec-h">calibration vs human gold</h2>
		{#if !cal || !cal.present}
			<div class="named-empty">
				<p class="ne-head">no calibration products for this run</p>
				<p class="ne-why">
					This export predates the calibration arc (no <code>metrics.json</code>). Re-export with
					baked gold to populate the reliability diagram, Brier decomposition, and confusion
					mosaic. The residual-vs-INDRA view below is unaffected.
				</p>
			</div>
		{:else if !cal.consistency.valid}
			<div class="named-empty">
				<p class="ne-head">calibration artifact contract is inconsistent</p>
				<p class="ne-why">
					No headline, hybrid promotion, or temporal delta is shown because the export and
					<code>metrics.json</code> do not establish one trusted contract:
					{cal.consistency.reasons.join(' · ')}.
				</p>
			</div>
		{:else if !metrics?.gold}
			<div class="named-empty">
				<p class="ne-head">no gold baked for this run</p>
				<p class="ne-why">
					{(metrics?.tiers?.ev && 'reason' in metrics.tiers.ev && metrics.tiers.ev.reason) ||
						'This run was scored without a human-curation gold set, so there is nothing to calibrate against.'}
					Cost, verdict mix, and residual-vs-INDRA below are still shown.
				</p>
			</div>
		{:else}
			{@const headBlock = tierBlock(tier)}
			{@const headArmKey = headlineArm(tier)}
			{@const headArm = armOf(headBlock, headArmKey)}
			<p class="sec-sub">
					The reliability diagram shows where stated belief and observed correct-rate differ on
					this evaluation set. The selected {tier === 'stmt' ? 'statement' : 'evidence'} tier plots
					{(headBlock && headBlock.status === 'available' ? headBlock.n : 0).toLocaleString('en-US')}
					evaluated units; the baked gold covers {metrics.gold.covered.toLocaleString('en-US')}
					evidence rows (<code>{metrics.gold.source?.split('/').pop()}</code>). Every number is served from
				<code>metrics.json</code> — the viewer computes none of it. Contract:
					<strong>{calibrationContractLabel(metrics)}</strong>.
				</p>
				<p class="contract-note" class:legacy={cal.evaluation.kind !== 'independent-validation-pass'}>
					Displayed evaluation: <strong>{cal.evaluation.label}</strong>.
					{#if cal.evaluation.kind === 'in-sample-fit'}
						These are descriptive fit diagnostics, not independent validation evidence.
					{:else if cal.evaluation.kind === 'independent-validation-pass'}
						This exact run and gold source match the profile's recorded held-out validation.
					{:else if cal.evaluation.kind === 'out-of-sample'}
						The profile was frozen elsewhere, but this run is not its recorded validation gate.
					{:else if cal.evaluation.kind === 'unprofiled'}
						No fitted-profile provenance is available; do not read this surface as validation.
					{:else}
						The recorded independent gate did not pass.
					{/if}
				</p>
			{#if metrics.schema_version < HYBRID_METRICS_SCHEMA}
				<p class="contract-note legacy">
					Legacy schema v{metrics.schema_version}: its <code>soft</code> slot is the historical
					survival-weight score. It is shown for audit only; the canonical statement headline
					remains the hard gate and is never treated as the current hybrid.
				</p>
			{/if}
			{#if profileState}
					<p class="contract-note" class:legacy={profileState.status !== 'available'}>
						{#if profileState.status === 'available' && profileState.soft_weights && 'profile_id' in profileState.soft_weights}
							Profile <strong>{profileState.soft_weights.profile_id}</strong> · exact configuration
							<code>{profileState.reader_configuration?.id ?? profileState.soft_weights.reader_configuration}</code>.<br />
							Fit: <code>{profileState.soft_weights.fit_run}</code> on
							<code>{profileState.soft_weights.fit_gold}</code>
							(n={profileState.soft_weights.fit_unique_pairs.toLocaleString('en-US')}, SHA
							<code>{profileState.soft_weights.fit_gold_sha256?.slice(0, 12) ?? '—'}</code>).
							Recorded validation:
							<strong>{profileState.soft_weights.validation?.result?.toUpperCase() ?? 'UNRECORDED'}</strong>
							{profileState.soft_weights.validation?.gate ?? ''} on
							<code>{profileState.soft_weights.validation?.run ?? '—'}</code> +
							<code>{profileState.soft_weights.validation?.gold ?? '—'}</code> (SHA
							<code>{profileState.soft_weights.validation?.gold_sha256?.slice(0, 12) ?? '—'}</code>).
					{:else}
						Hybrid profile unavailable for
						<code>{profileState.reader_configuration?.id ?? run?.model ?? 'unknown configuration'}</code>:
						{profileState.reason ?? 'no ship-approved exact configuration profile'}.
					{/if}
				</p>
			{/if}

			<!-- P1: ECE as the HEADLINE number, with P5 delta -->
			<div class="headline">
				<div class="ece">
					<span class="ece-lab">ECE</span>
					<span class="ece-val">{f3c(headArm?.ece)}</span>
					<span class="ece-tier">{TIER_LABEL[tier]} · {armLabel(headArmKey)}</span>
				</div>
				{#if deltaStr(tier)}
					<span class="ece-delta" class:improved={deltaImproved(tier)} class:worse={!deltaImproved(tier)}>
						{deltaStr(tier)}
					</span>
				{:else}
					<span class="ece-delta none">no compatible earlier run to compare</span>
				{/if}
			</div>

			<!-- The ONE reactive lever: Tier toggle -->
			<div class="tier-toggle" role="group" aria-label="tier toggle">
				<button class:active={tier === 'ev'} onclick={() => setTier('ev')}>Tier 1 · evidence</button>
				<button class:active={tier === 'stmt'} onclick={() => setTier('stmt')}>Tier 2 · statement</button>
			</div>

			<!-- Both tiers rendered stacked + labeled, active one first; NEVER merged -->
			{#each orderedTiers as t (t)}
				{@const block = tierBlock(t)}
				<section class="tier-panel" class:dimmed={t !== tier}>
					<header class="tier-head">
						<h3 class="tier-h">{TIER_LABEL[t]}</h3>
						<span class="tier-sub">{TIER_SUB[t]}</span>
						{#if block && block.status === 'available'}
							<span class="tier-n"
								>n={block.n.toLocaleString('en-US')} · base-rate correct
								{(block.base_rate_correct * 100).toFixed(0)}%</span
							>
						{/if}
					</header>

					{#if !block || block.status !== 'available'}
						<div class="named-empty inline">
							<p class="ne-why">
								{block && 'reason' in block ? block.reason : 'this tier is unavailable for this run'}
							</p>
						</div>
					{:else}
						{@const headlineKey = headlineArm(t)}
						{@const headline = armOf(block, headlineKey)}
						<!-- reliability diagram (strong center) + confusion mosaic side by side -->
						{#if headline}
							<div class="diag-row">
								<div class="diag-cell">
									<ReliabilityDiagram
										bins={headline.bins}
										n={headline.n}
										label="reliability — {armLabel(headlineKey)}"
									/>
								</div>
								<div class="diag-cell">
									<p class="mini-lab">confusion vs gold</p>
									<ConfusionMosaic
										tp={headline.confusion.tp}
										fp={headline.confusion.fp}
										fn={headline.confusion.fn}
										tn={headline.confusion.tn}
										axis={t === 'ev' ? 'correct' : 'error'}
									/>
								</div>
							</div>
						{/if}

						<!-- per-arm row: ECE + Brier decomposition for each arm -->
						<div class="arm-grid">
							{#each TIER_ARMS[t] as armKey}
								{@const arm = armOf(block, armKey)}
								<div class="arm-card" class:head={armKey === headlineKey}>
									<div class="arm-head">
										<span class="arm-name">{armLabel(armKey)}</span>
										{#if arm}
											<span class="arm-metrics"
												>ECE {f3c(arm.ece)} · AUROC {f3c(arm.auroc)} · AUPRC {f3c(arm.auprc)}</span
											>
										{/if}
									</div>
									{#if arm}
										<BrierBar
											reliability={arm.reliability}
											resolution={arm.resolution}
											uncertainty={arm.uncertainty}
											brier={arm.brier}
										/>
									{:else}
										<p class="ne-why inline">{armReason(block, armKey) ?? 'arm unavailable'}</p>
									{/if}
								</div>
							{/each}
						</div>
					{/if}
				</section>
			{/each}

			<!-- P5/stratum: the residual stratification retires the unavailable apology -->
			{#if v.byIndraType.length > 0}
				<div class="stratum-strip">
					<p class="strip-lab">
						residual dispersion by statement type — worst-calibrated types first (this is the
						belief-vs-INDRA residual, a different axis than the gold-ECE above; it is what we
						previously filed as "unavailable")
					</p>
					<div class="strip-rows">
						{#each v.byIndraType.slice(0, 8) as s}
							<div class="strip-row">
								<span class="strip-type">{s.value}</span>
								<span class="strip-n">n={fmtCount(s.n)}</span>
								<span class="strip-mae">MAE {fmt3(s.mae)}</span>
								<span class="strip-bias">{fmtSigned(s.bias)}</span>
							</div>
						{/each}
					</div>
				</div>
			{/if}
		{/if}
	</section>

	<!-- ── E11: statement verdict error-detection (schema v2) ── -->
	{#if stmtVerdict}
		<section class="verdict-surface" aria-label="statement verdict error detection">
			<h2 class="sec-h">statement verdict — error detection</h2>
			<p class="sec-sub">
				A different question than calibration. Not "is the belief scalar well-calibrated" but:
				does the tiered <em>decision</em> — correct / review / incorrect — catch the statements
				human gold says are wrong? Positive = error; <strong>review and incorrect both count as
				flagged</strong>. Every number served from <code>metrics.json</code>.
			</p>

			<!-- strong center: error-F1 headline -->
			<div class="headline">
				<div class="ece">
					<span class="ece-lab">error F1</span>
					<span class="ece-val">{f3c(stmtVerdict.f1)}</span>
					<span class="ece-tier"
						>precision {f3c(stmtVerdict.precision)} · recall {f3c(stmtVerdict.recall)} · n={stmtVerdict.n.toLocaleString(
							'en-US'
						)}</span
					>
				</div>
			</div>

			<div class="diag-row">
				<div class="diag-cell">
					<p class="mini-lab">verdict vs statement gold</p>
					<ConfusionMosaic
						tp={stmtVerdict.tp}
						fp={stmtVerdict.fp}
						fn={stmtVerdict.fn}
						tn={stmtVerdict.tn}
						axis="error"
					/>
				</div>
				{#if stratified}
					<div class="diag-cell driver">
						<p class="mini-lab">where the errors live — by what drove the decision</p>
						<DriverMosaic drivers={stratified.by_driver} />
					</div>
				{/if}
			</div>

			<!-- progressive disclosure: residual map by stratum, worst error-F1 first -->
			{#if stratified}
				<div class="stratum-strip">
					<div class="strat-tabs" role="group" aria-label="stratification axis">
						{#each STRAT_AXES as ax (ax.key)}
							<button class:active={stratAxis === ax.key} onclick={() => (stratAxis = ax.key)}>
								{ax.label}
							</button>
						{/each}
					</div>
					<p class="strip-lab">
						error detection by {STRAT_AXES.find((a) => a.key === stratAxis)?.label} — worst F1 first.
						<span class="weak-key">faint = under-powered (n&lt;50)</span>; fp = a correct statement
						over-rejected, fn = a real error that slipped through.
					</p>
					<div class="strip-rows">
						{#each sortedStrata as s (s.value)}
							<div class="vstrip-row" class:weak={s.n < 50}>
								<span class="strip-type">{s.value}</span>
								<span class="strip-n">n={fmtCount(s.n)}</span>
								<span class="f1-track" title={`error F1 ${f3c(s.f1)}`}>
									<span class="f1-fill {f1Class(s.f1)}" style={`width:${s.f1 * 100}%`}></span>
								</span>
								<span class="strip-f1 {f1Class(s.f1)}">{f3c(s.f1)}</span>
								<span class="strip-fr">fp {s.fp} · fn {s.fn}</span>
							</div>
						{/each}
					</div>
				</div>
			{/if}
		</section>
	{/if}

	<section class="calibration">
		<h2 class="sec-h">residual vs INDRA belief</h2>
		<p class="sec-sub">
			Residual is our per-evidence score minus the published RasMachine belief, over the
			{fmtCount(v.calibration.n)} evidence rows with a score, a belief, and a verdict.
			This is a different question than the gold calibration above — it measures agreement with
			INDRA's prior, not correctness.
		</p>
		<dl class="stat-row">
			<div class="stat">
				<dt>MAE</dt>
				<dd>{fmt3(v.calibration.mae)}</dd>
			</div>
			<div class="stat">
				<dt>bias</dt>
				<dd>{fmtSigned(v.calibration.bias)}</dd>
			</div>
			<div class="stat">
				<dt>mean residual</dt>
				<dd>{fmtSigned(residual?.mean_residual)}</dd>
			</div>
		</dl>

		{#if residual && residual.n_total > 0}
			<div class="hist" role="img" aria-label="residual distribution histogram, 11 bins from −1 to +1">
				{#each residual.bins as count, i}
					<div class="hist-col" title={`${binLabel(i, residual.bins.length)} · ${fmtCount(count)}`}>
						<div class="hist-bar" style={`height:${(count / residualMax) * 100}%`}></div>
					</div>
				{/each}
			</div>
			<div class="hist-axis">
				<span>−1.0</span>
				<span class="muted">residual (our − INDRA)</span>
				<span>+1.0</span>
			</div>
		{/if}
	</section>

	<section class="verdicts">
		<h2 class="sec-h">verdict distribution</h2>
		<ul class="dist">
			{#each verdicts as row}
				<li class={`dist-row ${verdictClass(row.verdict)}`}>
					<span class="dist-label">{row.verdict}</span>
					<span class="dist-bar-wrap">
						<span
							class="dist-bar"
							style={`width:${verdictTotal ? (row.n / verdictTotal) * 100 : 0}%`}
						></span>
					</span>
					<span class="dist-n">{fmtCount(row.n)}</span>
				</li>
			{/each}
		</ul>
	</section>

	<section class="buckets">
		<h2 class="sec-h">bucket taxonomy</h2>
		<p class="sec-sub">Report-taxonomy classification of every evidence row in this run.</p>
		<ul class="dist">
			{#each v.buckets as row}
				<li class="dist-row">
					<span class="dist-label">{row.bucket}</span>
					<span class="dist-bar-wrap">
						<span
							class="dist-bar dist-bar-neutral"
							style={`width:${bucketTotal ? (row.n / bucketTotal) * 100 : 0}%`}
						></span>
					</span>
					<span class="dist-n">{fmtCount(row.n)}</span>
				</li>
			{/each}
		</ul>
	</section>

	{#if v.confidenceCalibration.length > 0}
		<section class="slice">
			<h2 class="sec-h">calibration by stated confidence</h2>
			<table class="slice-table">
				<thead>
					<tr>
						<th>confidence</th>
						<th class="num">n</th>
						<th class="num">MAE</th>
						<th class="num">bias</th>
					</tr>
				</thead>
				<tbody>
					{#each v.confidenceCalibration as c}
						<tr>
							<td>{c.confidence}</td>
							<td class="num">{fmtCount(c.n)}</td>
							<td class="num">{fmt3(c.mae)}</td>
							<td class="num">{fmtSigned(c.bias)}</td>
						</tr>
					{/each}
				</tbody>
			</table>
		</section>
	{/if}

	{#if v.byIndraType.length > 0}
		<section class="slice">
			<h2 class="sec-h">calibration by INDRA statement type</h2>
			<p class="sec-sub">Sorted by mean absolute residual — worst-calibrated types first.</p>
			<table class="slice-table">
				<thead>
					<tr>
						<th>statement type</th>
						<th class="num">n</th>
						<th class="num">MAE</th>
						<th class="num">bias</th>
					</tr>
				</thead>
				<tbody>
					{#each v.byIndraType as s}
						<tr>
							<td>{s.value}</td>
							<td class="num">{fmtCount(s.n)}</td>
							<td class="num">{fmt3(s.mae)}</td>
							<td class="num">{fmtSigned(s.bias)}</td>
						</tr>
					{/each}
				</tbody>
			</table>
		</section>
	{/if}

	{#if v.bySourceApi.length > 0}
		<section class="slice">
			<h2 class="sec-h">calibration by source API</h2>
			<table class="slice-table">
				<thead>
					<tr>
						<th>source</th>
						<th class="num">n</th>
						<th class="num">MAE</th>
						<th class="num">bias</th>
					</tr>
				</thead>
				<tbody>
					{#each v.bySourceApi as s}
						<tr>
							<td>{s.value}</td>
							<td class="num">{fmtCount(s.n)}</td>
							<td class="num">{fmt3(s.mae)}</td>
							<td class="num">{fmtSigned(s.bias)}</td>
						</tr>
					{/each}
				</tbody>
			</table>
		</section>
	{/if}

	{#if v.unavailable.length > 0}
		<section class="unavailable" aria-label="unavailable strata">
			<h2 class="sec-h">not available from monolithic exports</h2>
			<ul class="unavailable-list">
				{#each v.unavailable as u}
					<li>{u}</li>
				{/each}
			</ul>
		</section>
	{/if}
</main>

<style>
	:global(:root) {
		--ink: #1a1a1a;
		--ink-muted: #6a6a6a;
		--ink-faint: #727272;
		--paper: #fdfcf8;
		--rule: #e6e2d6;
		--accent: #7d2a1a;
		--accent-wash: rgba(125, 42, 26, 0.04);
		--ok-green: #2a6f2a;
		--mono: ui-monospace, 'SF Mono', 'JetBrains Mono', Menlo, monospace;
		--serif: 'Iowan Old Style', 'Source Serif Pro', Georgia, serif;
	}
	:global(html, body) {
		background: var(--paper);
		color: var(--ink);
		font-family: var(--serif);
		font-size: 16px;
		line-height: 1.5;
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
	}
	.crumb a {
		color: var(--ink-muted);
		text-decoration: none;
	}
	.crumb a:hover {
		color: var(--ink);
	}
	.crumb strong {
		color: var(--ink);
		font-weight: 500;
	}
	.crumb .sep {
		color: var(--ink-faint);
	}
	.meta {
		display: flex;
		gap: 0.8rem;
		align-items: baseline;
	}
	.nav-link {
		color: var(--accent);
		text-decoration: none;
	}
	.nav-link:hover {
		text-decoration: underline;
	}
	.muted {
		color: var(--ink-faint);
	}
	main {
		max-width: 1100px;
		margin: 0 auto;
		padding: 2rem 1.5rem 4rem;
	}

	.run-h {
		font-family: var(--serif);
		font-size: 1.4rem;
		font-weight: 400;
		margin: 0 0 0.6rem;
		display: flex;
		gap: 0.5rem;
		align-items: baseline;
		flex-wrap: wrap;
	}
	.run-model {
		font-family: var(--mono);
		font-size: 0.9rem;
		color: var(--accent);
	}
	.run-status {
		font-family: var(--mono);
		font-size: 0.74rem;
		text-transform: lowercase;
		letter-spacing: 0.04em;
		color: var(--ok-green);
		padding: 0 0.4rem;
	}

	.run-fields {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
		gap: 0;
		margin: 0 0 2.5rem;
	}
	.run-fields > div {
		padding: 0.4rem 1rem 0.4rem 0;
		border-right: 1px solid var(--rule);
	}
	.run-fields > div:last-child {
		border-right: none;
	}
	.run-fields dt {
		font-family: var(--mono);
		font-size: 0.7rem;
		color: var(--ink-muted);
		text-transform: lowercase;
		letter-spacing: 0.02em;
		margin: 0;
	}
	.run-fields dd {
		font-family: var(--mono);
		font-size: 0.86rem;
		color: var(--ink);
		margin: 0.1rem 0 0;
		font-variant-numeric: tabular-nums;
		overflow: hidden;
		text-overflow: ellipsis;
	}

	section {
		margin: 0 0 2.5rem;
	}
	.sec-h {
		font-family: var(--serif);
		font-size: 1.15rem;
		font-weight: 400;
		margin: 0 0 0.4rem;
	}
	.sec-sub {
		max-width: 760px;
		margin: 0 0 0.8rem;
		color: var(--ink-muted);
		font-size: 0.92rem;
		line-height: 1.45;
	}

	.stat-row {
		display: flex;
		flex-wrap: wrap;
		gap: 2rem;
		margin: 0 0 1.2rem;
	}
	.stat dt {
		font-family: var(--mono);
		font-size: 0.7rem;
		color: var(--ink-muted);
		text-transform: lowercase;
		letter-spacing: 0.02em;
		margin: 0;
	}
	.stat dd {
		font-family: var(--mono);
		font-size: 1.3rem;
		font-variant-numeric: tabular-nums;
		margin: 0.1rem 0 0;
		color: var(--ink);
	}

	/* ── C4 calibration surface ── */
	.cal-surface {
		border-top: 2px solid var(--ink);
		padding-top: 1.2rem;
	}
	.contract-note {
		max-width: 760px;
		margin: -0.15rem 0 0.9rem;
		padding: 0.55rem 0.75rem;
		border-left: 3px solid var(--rule);
		color: var(--ink-muted);
		font-size: 0.82rem;
		line-height: 1.45;
	}
	.contract-note.legacy {
		border-left-color: var(--ink-muted);
	}
	.headline {
		display: flex;
		align-items: baseline;
		flex-wrap: wrap;
		gap: 0.5rem 1.4rem;
		margin: 0.6rem 0 1rem;
	}
	.ece {
		display: flex;
		align-items: baseline;
		gap: 0.5rem;
		flex-wrap: wrap;
	}
	.ece-lab {
		font-family: var(--mono);
		font-size: 0.74rem;
		color: var(--ink-muted);
		text-transform: uppercase;
		letter-spacing: 0.08em;
	}
	.ece-val {
		font-family: var(--mono);
		font-size: 2.6rem;
		line-height: 1;
		font-variant-numeric: tabular-nums;
		color: var(--ink);
	}
	.ece-tier {
		font-family: var(--mono);
		font-size: 0.72rem;
		color: var(--ink-muted);
	}
	.ece-delta {
		font-family: var(--mono);
		font-size: 0.8rem;
		font-variant-numeric: tabular-nums;
	}
	.ece-delta.improved {
		color: var(--ok-green);
	}
	.ece-delta.worse {
		color: var(--accent);
	}
	.ece-delta.none {
		color: var(--ink-faint);
	}

	.tier-toggle {
		display: inline-flex;
		border: 1px solid var(--rule);
		margin: 0 0 1.4rem;
	}
	.tier-toggle button {
		font-family: var(--mono);
		font-size: 0.76rem;
		background: transparent;
		border: none;
		color: var(--ink-muted);
		padding: 0.4rem 0.9rem;
		cursor: pointer;
	}
	.tier-toggle button + button {
		border-left: 1px solid var(--rule);
	}
	.tier-toggle button.active {
		background: var(--ink);
		color: var(--paper);
	}
	.tier-toggle button:hover:not(.active) {
		color: var(--ink);
	}

	.tier-panel {
		margin: 0 0 2rem;
		padding-left: 0.9rem;
		border-left: 3px solid var(--accent);
	}
	.tier-panel.dimmed {
		border-left-color: var(--rule);
		opacity: 0.82;
	}
	.tier-head {
		display: flex;
		align-items: baseline;
		flex-wrap: wrap;
		gap: 0.3rem 0.8rem;
		margin-bottom: 0.9rem;
	}
	.tier-h {
		font-family: var(--serif);
		font-size: 1.02rem;
		font-weight: 400;
		margin: 0;
	}
	.tier-sub {
		font-size: 0.86rem;
		color: var(--ink-muted);
	}
	.tier-n {
		font-family: var(--mono);
		font-size: 0.7rem;
		color: var(--ink-faint);
		margin-left: auto;
	}
	.diag-row {
		display: flex;
		flex-wrap: wrap;
		gap: 2rem;
		margin-bottom: 1.2rem;
	}
	.diag-cell {
		flex: 1 1 18rem;
		min-width: 16rem;
	}
	.mini-lab {
		font-family: var(--mono);
		font-size: 0.74rem;
		color: var(--ink-muted);
		margin: 0 0 0.4rem;
		letter-spacing: 0.02em;
	}
	.arm-grid {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(15rem, 1fr));
		gap: 1.4rem 1.6rem;
	}
	.arm-card {
		min-width: 0;
	}
	.arm-head {
		display: flex;
		flex-direction: column;
		gap: 0.1rem;
		margin-bottom: 0.5rem;
	}
	.arm-name {
		font-family: var(--mono);
		font-size: 0.78rem;
		color: var(--ink);
	}
	.arm-metrics {
		font-family: var(--mono);
		font-size: 0.68rem;
		color: var(--ink-muted);
		font-variant-numeric: tabular-nums;
	}

	.named-empty {
		border-left: 3px solid var(--rule);
		padding: 0.2rem 0 0.2rem 0.9rem;
		margin: 0.4rem 0 1rem;
	}
	.named-empty.inline {
		margin: 0.4rem 0;
	}
	.ne-head {
		font-family: var(--mono);
		font-size: 0.82rem;
		color: var(--ink);
		margin: 0 0 0.3rem;
	}
	.ne-why {
		font-size: 0.88rem;
		color: var(--ink-muted);
		margin: 0;
		max-width: 720px;
		line-height: 1.45;
	}
	.ne-why.inline {
		font-family: var(--mono);
		font-size: 0.72rem;
	}

	.stratum-strip {
		margin-top: 1.4rem;
		padding-top: 1rem;
		border-top: 1px dotted var(--rule);
	}
	.strip-lab {
		font-size: 0.82rem;
		color: var(--ink-muted);
		max-width: 760px;
		margin: 0 0 0.6rem;
		line-height: 1.4;
	}
	.strip-rows {
		display: flex;
		flex-direction: column;
		gap: 0.1rem;
	}
	.strip-row {
		display: grid;
		grid-template-columns: 12rem 7ch 9ch 7ch;
		gap: 0.8rem;
		font-family: var(--mono);
		font-size: 0.74rem;
		font-variant-numeric: tabular-nums;
		padding: 0.18rem 0;
		border-bottom: 1px dotted var(--rule);
	}
	.strip-type {
		color: var(--ink);
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}
	.strip-n,
	.strip-mae {
		color: var(--ink-muted);
		text-align: right;
	}
	.strip-bias {
		color: var(--ink);
		text-align: right;
	}

	/* ── E11 verdict-surface — error detection on the tiered decision ── */
	.verdict-surface {
		border-top: 2px solid var(--ink);
		padding-top: 1.2rem;
	}
	.diag-cell.driver {
		flex: 2 1 26rem;
		min-width: 20rem;
	}
	.strat-tabs {
		display: inline-flex;
		border: 1px solid var(--rule);
		margin: 0 0 0.7rem;
		flex-wrap: wrap;
	}
	.strat-tabs button {
		font-family: var(--mono);
		font-size: 0.72rem;
		background: transparent;
		border: none;
		color: var(--ink-muted);
		padding: 0.35rem 0.8rem;
		cursor: pointer;
	}
	.strat-tabs button + button {
		border-left: 1px solid var(--rule);
	}
	.strat-tabs button.active {
		background: var(--ink);
		color: var(--paper);
	}
	.strat-tabs button:hover:not(.active) {
		color: var(--ink);
	}
	.weak-key {
		color: var(--ink-faint);
	}
	.vstrip-row {
		display: grid;
		grid-template-columns: minmax(7rem, 12rem) 7ch 1fr 6ch 13ch;
		gap: 0.8rem;
		align-items: center;
		font-family: var(--mono);
		font-size: 0.74rem;
		font-variant-numeric: tabular-nums;
		padding: 0.22rem 0;
		border-bottom: 1px dotted var(--rule);
	}
	.vstrip-row.weak {
		opacity: 0.5;
	}
	.f1-track {
		height: 0.5rem;
		background: var(--accent-wash);
		overflow: hidden;
	}
	.f1-fill {
		display: block;
		height: 100%;
		transition: width 200ms ease-out;
	}
	.f1-fill.f1-good {
		background: var(--ok-green);
	}
	.f1-fill.f1-mid {
		background: #6f5a16;
	}
	.f1-fill.f1-weak {
		background: var(--accent);
	}
	.strip-f1 {
		text-align: right;
	}
	.strip-f1.f1-good {
		color: var(--ok-green);
	}
	.strip-f1.f1-mid {
		color: #6f5a16;
	}
	.strip-f1.f1-weak {
		color: var(--accent);
	}
	.strip-fr {
		color: var(--ink-muted);
		text-align: right;
	}

	.hist {
		display: flex;
		align-items: flex-end;
		gap: 2px;
		height: 110px;
		border-bottom: 1px solid var(--rule);
		margin-top: 0.5rem;
	}
	.hist-col {
		flex: 1 1 0;
		display: flex;
		align-items: flex-end;
		height: 100%;
	}
	.hist-bar {
		width: 100%;
		background: var(--accent);
		opacity: 0.78;
		min-height: 1px;
	}
	.hist-axis {
		display: flex;
		justify-content: space-between;
		font-family: var(--mono);
		font-size: 0.7rem;
		color: var(--ink-faint);
		margin-top: 0.3rem;
	}

	.dist {
		list-style: none;
		padding: 0;
		margin: 0;
		font-family: var(--mono);
		font-size: 0.8rem;
	}
	.dist-row {
		display: grid;
		grid-template-columns: 12rem 1fr 6ch;
		gap: 0.8rem;
		align-items: center;
		padding: 0.2rem 0;
	}
	.dist-label {
		color: var(--ink);
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}
	.dist-bar-wrap {
		background: var(--accent-wash);
		height: 0.85rem;
	}
	.dist-bar {
		display: block;
		height: 100%;
		background: var(--ink-muted);
	}
	.dist-bar-neutral {
		background: var(--ink-muted);
	}
	.v-correct .dist-bar {
		background: var(--ok-green);
	}
	.v-incorrect .dist-bar {
		background: var(--accent);
	}
	.dist-n {
		text-align: right;
		font-variant-numeric: tabular-nums;
		color: var(--ink);
	}

	.slice-table {
		width: 100%;
		border-collapse: collapse;
		font-family: var(--mono);
		font-size: 0.78rem;
		font-variant-numeric: tabular-nums;
	}
	.slice-table th {
		text-align: left;
		font-weight: 500;
		color: var(--ink-muted);
		font-size: 0.7rem;
		text-transform: lowercase;
		letter-spacing: 0.02em;
		padding: 0.3rem 0.65rem 0.3rem 0;
		border-bottom: 1px dotted var(--rule);
	}
	.slice-table td {
		padding: 0.32rem 0.65rem 0.32rem 0;
		border-bottom: 1px dotted var(--rule);
		color: var(--ink);
	}
	.slice-table tr:last-child td {
		border-bottom: none;
	}
	.slice-table .num {
		text-align: right;
		white-space: nowrap;
	}

	.unavailable {
		border-left: 3px solid var(--rule);
		padding-left: 0.9rem;
	}
	.unavailable-list {
		margin: 0;
		padding-left: 1.1rem;
		color: var(--ink-muted);
		font-size: 0.92rem;
	}
	.unavailable-list li {
		padding: 0.12rem 0;
	}

	@media (max-width: 720px) {
		.dist-row {
			grid-template-columns: 8rem 1fr 6ch;
		}
	}
</style>
