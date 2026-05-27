<script lang="ts">
	import type { HeuristicCoverage, ProbeCoverageRow } from '$lib/db';

	let { coverage, runStatus = 'unknown' }: { coverage: HeuristicCoverage; runStatus?: string } = $props();

	const totalProbeSlots = $derived(coverage.per_probe.reduce((s, p) => s + p.total, 0));
	const totalSubstrate = $derived(coverage.per_probe.reduce((s, p) => s + p.substrate_n, 0));
	const totalLlm = $derived(coverage.per_probe.reduce((s, p) => s + p.llm_n, 0));
	const totalAbstain = $derived(coverage.per_probe.reduce((s, p) => s + p.abstain_n, 0));
	const totalNotrun = $derived(coverage.per_probe.reduce((s, p) => s + p.notrun_n, 0));
	const probeCoverageAvailable = $derived(coverage.n_probe_evidences > 0 && totalProbeSlots > 0);
	const probeRecordsAbsent = $derived(
		coverage.applicability !== 'not_defined' &&
		coverage.n_evidences > 0 &&
		!probeCoverageAvailable
	);

	function pctOf(n: number, total: number): number {
		return total > 0 ? (n / total) * 100 : 0;
	}

	function pctStr(p: number): string {
		if (p === 0) return '0%';
		if (p < 1) return '<1%';
		if (p > 99 && p < 100) return '>99%';
		return `${Math.round(p)}%`;
	}

	function probeLabel(p: string): string {
		return (
			{
				subject_role: 'subj-role',
				object_role: 'obj-role',
				relation_axis: 'relation-axis',
				scope: 'scope'
			} as Record<string, string>
		)[p] ?? p;
	}

	const summary = $derived.by(() => {
		if (coverage.applicability === 'not_defined') {
			return coverage.not_defined_reason ?? 'this panel is not defined for the selected architecture';
		}
		if (probeRecordsAbsent) {
			const noun = coverage.n_evidences === 1 ? 'row' : 'rows';
			const verb = coverage.n_evidences === 1 ? 'exists' : 'exist';
			return `${coverage.n_evidences} aggregate evidence ${noun} ${verb}, but no decomposed substrate_route/probe rows are persisted for this run. Probe coverage is unavailable, not zero.`;
		}
			if (!probeCoverageAvailable) return 'no probes recorded for this run';
			const mean = coverage.mean_invoked_probes;
			const firedSlots = totalSubstrate + totalLlm + totalAbstain;
			const sPct = pctStr(pctOf(totalSubstrate, firedSlots));
			const lPct = pctStr(pctOf(totalLlm, firedSlots));
			const aPct = pctStr(pctOf(totalAbstain, firedSlots));
			const denominatorBoundary = coverage.n_evidences === coverage.n_probe_evidences
				? ''
				: ` Coverage uses ${coverage.n_probe_evidences} probe evidence row${coverage.n_probe_evidences === 1 ? '' : 's'}; aggregate verdict panels use ${coverage.n_evidences} aggregate evidence row${coverage.n_evidences === 1 ? '' : 's'}.`;
			// Lead with the sharpest finding — the average number of probes that
			// actually fired. "the LLM did not fire" was a passive observation;
			// "the system invoked X of 4 probes per evidence" is the instrument state.
			const leadFact = `On average the system invoked ${mean.toFixed(2)} of 4 probes per probe-covered evidence — the remaining ${(4 - mean).toFixed(2)} short-circuited (a preceding probe's finding made them unnecessary).`;
			const substrateFact = totalSubstrate > 0
				? ` Of the probes that fired, substrate (regex / Gilda / catalog) resolved ${sPct} without calling an LLM`
				: ` No probes resolved via substrate.`;
		const llmFact = totalLlm > 0
			? `; ${lPct} required an LLM call.`
			: `; no LLM calls were made this run.`;
			const abstainFact = totalAbstain > 0
				? ` ${aPct} of fired probes abstained.`
				: '';
			return leadFact + substrateFact + llmFact + abstainFact + denominatorBoundary;
	});

	function probeOrder(p: ProbeCoverageRow): number {
		return (
			{ subject_role: 0, object_role: 1, relation_axis: 2, scope: 3 } as Record<string, number>
		)[p.probe] ?? 99;
	}
	const orderedProbes = $derived([...coverage.per_probe].sort((a, b) => probeOrder(a) - probeOrder(b)));
	const partialCoverageSample = $derived(
			runStatus !== 'succeeded' &&
			coverage.applicability !== 'not_defined' &&
			(coverage.n_evidences > 0 || coverage.n_probe_evidences > 0)
	);
	const liveSample = $derived(runStatus === 'running');
	const sampleLabel = $derived(liveSample
		? 'live run sample'
		: runStatus === 'canceled'
			? 'canceled run sample'
			: runStatus === 'failed'
				? 'failed run sample'
				: 'terminated run sample');
	const sampleCopy = $derived(liveSample
		? 'Coverage counts are still accumulating and reflect only decomposed substrate/probe rows persisted at this page load.'
		: 'Coverage is measured over the worker-order subset with persisted decomposed substrate/probe rows before termination, not a random sample. LLM-routed evidence can be underrepresented because it takes longer to finish.');
</script>

<section class="coverage">
	<h2 class="cov-h">
		what the system is doing in this run
		<span class="cov-run-id" title="run_id">{coverage.run_id.slice(0, 8)}</span>
		<span class="cov-run-id" title="architecture">{coverage.architecture}</span>
	</h2>

	{#if partialCoverageSample}
		<div class="cov-interrupted" role="note">
			<span>{sampleLabel}</span>
			<p>
				This run is <strong>{runStatus}</strong>; {sampleCopy}
			</p>
		</div>
	{/if}

	<p class="cov-summary">{summary}</p>

	{#if coverage.applicability === 'not_defined'}
		<div class="cov-na" role="note">
			<span class="cov-na-kicker">not defined for this architecture</span>
			<p>
				This is not missing data: the four-probe pillbar belongs to decomposed scoring only.
				The native trace-fidelity panel above shows this run's architecture-specific trace.
			</p>
			{#if coverage.n_evidences > 0}
				<p class="cov-na-foot">{coverage.n_evidences} aggregate evidence row{coverage.n_evidences === 1 ? '' : 's'} can still support architecture-blind verdict, cost, and latency metrics.</p>
			{/if}
		</div>
	{:else if probeRecordsAbsent}
		<div class="cov-missing-probes" role="note">
			<span class="cov-missing-kicker">probe records absent for this run state</span>
			<p>
				This is not a zero-LLM or zero-probe result. The aggregate rows can still support architecture-blind summary metrics, but the four-probe health panel needs decomposed substrate/probe rows before substrate, LLM, abstain, and short-circuit rates are meaningful.
			</p>
			<div class="cov-trace-diagnostic" aria-label="persisted trace observation">
				<span class="cov-trace-kicker">persisted trace observation</span>
				<p>{coverage.trace_diagnostic.message}</p>
				<span class="cov-trace-kicker">run lifecycle boundary</span>
				<p>{coverage.trace_diagnostic.lifecycle_message}</p>
				<dl>
					<div>
						<dt>aggregate evidences</dt>
						<dd>{coverage.trace_diagnostic.n_aggregate_evidences}</dd>
					</div>
					<div>
						<dt>native steps</dt>
						<dd>{coverage.trace_diagnostic.n_nonaggregate_steps}</dd>
					</div>
					<div>
						<dt>route/probe rows</dt>
						<dd>{coverage.trace_diagnostic.n_substrate_route_steps + coverage.trace_diagnostic.n_probe_steps}</dd>
					</div>
					<div>
						<dt>run status</dt>
						<dd>{coverage.trace_diagnostic.run_status}</dd>
					</div>
					<div>
						<dt>scorer version</dt>
						<dd>{coverage.trace_diagnostic.scorer_version ?? 'not captured'}</dd>
					</div>
					<div>
						<dt>termination</dt>
						<dd>
							{#if coverage.trace_diagnostic.terminated_by || coverage.trace_diagnostic.termination_reason}
								{coverage.trace_diagnostic.terminated_by ?? 'unknown'}{#if coverage.trace_diagnostic.termination_reason} · {coverage.trace_diagnostic.termination_reason}{/if}
							{:else}
								none recorded
							{/if}
						</dd>
					</div>
				</dl>
			</div>
			<a class="cov-diagnostic-link" href={`/runs/${coverage.run_id}/cohort?trace_state=aggregate_only`}>open aggregate-only trace cohort</a>
		</div>
			{:else if probeCoverageAvailable}
				<p class="cov-denominator" aria-label="probe coverage denominator">
					probe coverage denominator: <a href={`/runs/${coverage.run_id}/cohort?probe_coverage=present`}>{coverage.n_probe_evidences} evidence row{coverage.n_probe_evidences === 1 ? '' : 's'} with persisted substrate/probe slots</a> · aggregate verdict denominator: {coverage.n_evidences} evidence row{coverage.n_evidences === 1 ? '' : 's'}
				</p>
			<p class="cov-legend" aria-hidden="true">
				pillbar colors: <span class="cov-leg cov-leg-substrate">■</span> substrate · <span class="cov-leg cov-leg-llm">■</span> llm · <span class="cov-leg cov-leg-abstain">■</span> abstain · <span class="cov-leg cov-leg-notrun">▦</span> probe didn't run (short-circuited by an earlier finding)
			</p>
		<table class="cov-table" aria-label="per-probe substrate / LLM / abstain / not-run breakdown">
			<thead>
				<tr>
					<th>probe</th>
					<th class="cov-bar-head">coverage</th>
					<th class="num">substrate</th>
					<th class="num">llm</th>
					<th class="num">abstain</th>
					<th class="num">not run</th>
				</tr>
			</thead>
			<tbody>
				{#each orderedProbes as p}
					{@const sP = pctOf(p.substrate_n, p.total)}
					{@const lP = pctOf(p.llm_n, p.total)}
					{@const aP = pctOf(p.abstain_n, p.total)}
					{@const nP = pctOf(p.notrun_n, p.total)}
					<tr>
						<td class="cov-name">{probeLabel(p.probe)}</td>
						<td class="cov-bar-cell">
							<div class="cov-bar" class:cov-bar-partial={partialCoverageSample} role="img" aria-label="{probeLabel(p.probe)} coverage: substrate {sP.toFixed(0)}%, llm {lP.toFixed(0)}%, abstain {aP.toFixed(0)}%, not run {nP.toFixed(0)}%">
								{#if sP > 0}<span class="cov-seg cov-seg-substrate" style:width="{sP}%" title="substrate {p.substrate_n}/{p.total}"></span>{/if}
								{#if lP > 0}<span class="cov-seg cov-seg-llm" style:width="{lP}%" title="llm {p.llm_n}/{p.total}"></span>{/if}
								{#if aP > 0}<span class="cov-seg cov-seg-abstain" style:width="{aP}%" title="abstain {p.abstain_n}/{p.total}"></span>{/if}
								{#if nP > 0}<span class="cov-seg cov-seg-notrun" style:width="{nP}%" title="not run {p.notrun_n}/{p.total}"></span>{/if}
							</div>
						</td>
						<td class="num cov-c-substrate">{pctStr(sP)}</td>
						<td class="num cov-c-llm">{pctStr(lP)}</td>
						<td class="num cov-c-abstain">{pctStr(aP)}</td>
						<td class="num cov-c-notrun">{pctStr(nP)}</td>
					</tr>
				{/each}
			</tbody>
		</table>

		<div class="cov-aux">
			<span class="cov-aux-row">
				<span class="cov-aux-label">mean probes invoked / evidence</span>
				<span class="cov-aux-val">{coverage.mean_invoked_probes.toFixed(2)} of 4</span>
				<span class="muted">— how many of the four probes actually fired on the average evidence</span>
			</span>
				<span class="cov-aux-row">
					<span class="cov-aux-label">evidences with ≥1 short-circuit</span>
					<span class="cov-aux-val">{pctStr(coverage.short_circuited_rate * 100)} of {coverage.n_probe_evidences}</span>
					<span class="muted">— a preceding probe's finding made later probes unnecessary</span>
				</span>
				<span class="cov-aux-row">
					<span class="cov-aux-label">evidences with zero LLM calls</span>
					<span class="cov-aux-val">{pctStr(coverage.all_substrate_rate * 100)} of {coverage.n_probe_evidences}</span>
					<span class="muted">— every invoked probe was answered by substrate (regex / Gilda / catalog)</span>
				</span>
		</div>
	{:else}
		<p class="cov-empty">no probe records persisted for this run yet</p>
	{/if}
</section>

<style>
	.coverage {
		margin: 0 0 2.5rem;
	}
	.cov-h {
		font-family: var(--serif);
		font-size: 1.15rem;
		font-weight: 400;
		color: var(--ink);
		margin: 0 0 0.6rem;
		display: flex;
		align-items: baseline;
		gap: 0.6rem;
	}
	.cov-run-id {
		font-family: var(--mono);
		font-size: 0.72rem;
		color: var(--ink-faint);
	}
	.cov-summary {
		font-family: var(--serif);
		font-size: 0.98rem;
		color: var(--ink);
		margin: 0 0 1rem;
		line-height: 1.5;
		font-variant-numeric: tabular-nums;
	}
	.cov-interrupted {
		max-width: 720px;
		border-left: 3px solid #8a6a00;
		padding: 0.55rem 0 0.55rem 0.85rem;
		margin: 0 0 0.9rem;
		background: rgba(138, 106, 0, 0.04);
	}
	.cov-interrupted span {
		display: block;
		font-family: var(--mono);
		font-size: 0.72rem;
		color: #8a6a00;
		text-transform: lowercase;
		letter-spacing: 0.02em;
	}
	.cov-interrupted p {
		margin: 0.2rem 0 0;
		color: var(--ink-muted);
		font-size: 0.9rem;
		line-height: 1.45;
	}
	.cov-na {
		max-width: 720px;
		border-left: 3px solid var(--ink);
		padding: 0.65rem 0 0.65rem 0.9rem;
		margin: 0 0 1rem;
	}
	.cov-na-kicker {
		font-family: var(--mono);
		font-size: 0.72rem;
		color: var(--ink);
		text-transform: lowercase;
		letter-spacing: 0.02em;
	}
	.cov-na p {
		margin: 0.25rem 0 0;
		color: var(--ink-muted);
		font-size: 0.9rem;
		line-height: 1.45;
	}
	.cov-na-foot {
		font-family: var(--mono);
		font-size: 0.76rem;
	}
	.cov-missing-probes {
		max-width: 720px;
		border-left: 3px solid var(--ink-muted);
		padding: 0.65rem 0 0.65rem 0.9rem;
		margin: 0 0 1rem;
		background: rgba(0, 0, 0, 0.025);
	}
	.cov-missing-kicker {
		font-family: var(--mono);
		font-size: 0.72rem;
		color: var(--ink);
		text-transform: lowercase;
		letter-spacing: 0.02em;
	}
	.cov-missing-probes p {
		margin: 0.25rem 0 0;
		color: var(--ink-muted);
		font-size: 0.9rem;
		line-height: 1.45;
	}
	.cov-trace-diagnostic {
		margin-top: 0.65rem;
		padding-top: 0.55rem;
		border-top: 1px dotted var(--rule);
	}
	.cov-trace-kicker {
		display: block;
		font-family: var(--mono);
		font-size: 0.7rem;
		color: var(--ink-muted);
		text-transform: lowercase;
		letter-spacing: 0.02em;
	}
	.cov-trace-diagnostic p {
		margin: 0.2rem 0 0.45rem;
	}
	.cov-trace-diagnostic dl {
		display: flex;
		flex-wrap: wrap;
		gap: 0.45rem 0.85rem;
		margin: 0;
		font-family: var(--mono);
		font-size: 0.74rem;
	}
	.cov-trace-diagnostic div {
		display: flex;
		gap: 0.3rem;
		align-items: baseline;
	}
	.cov-trace-diagnostic dt {
		color: var(--ink-muted);
	}
	.cov-trace-diagnostic dd {
		margin: 0;
		color: var(--ink);
		font-variant-numeric: tabular-nums;
	}
	.cov-diagnostic-link {
		display: inline-block;
		margin-top: 0.45rem;
		font-family: var(--mono);
		font-size: 0.76rem;
		color: var(--accent);
		text-decoration: none;
	}
		.cov-diagnostic-link:hover {
			text-decoration: underline;
		}
		.cov-denominator {
			max-width: 720px;
			margin: 0 0 0.35rem;
			font-family: var(--mono);
			font-size: 0.74rem;
			color: var(--ink-muted);
			font-variant-numeric: tabular-nums;
		}
		.cov-denominator a {
			color: var(--accent);
			text-decoration: none;
		}
		.cov-denominator a:hover {
			text-decoration: underline;
		}
		.cov-legend {
			font-family: var(--mono);
			font-size: 0.74rem;
		color: var(--ink-muted);
		margin: 0 0 0.4rem;
	}
	.cov-leg-substrate { color: var(--ok-green); }
	.cov-leg-llm { color: var(--accent); }
	.cov-leg-abstain { color: var(--ink-muted); }
	.cov-leg-notrun { color: var(--ink-faint); }
	.cov-table {
		width: 100%;
		max-width: 720px;
		border-collapse: collapse;
		font-family: var(--mono);
		font-size: 0.82rem;
		font-variant-numeric: tabular-nums;
	}
	.cov-table th {
		text-align: left;
		font-weight: 500;
		color: var(--ink-muted);
		font-size: 0.72rem;
		text-transform: lowercase;
		letter-spacing: 0.02em;
		padding: 0.3rem 0.6rem 0.3rem 0;
		border-bottom: 1px dotted var(--rule);
	}
	.cov-table td {
		padding: 0.4rem 0.6rem 0.4rem 0;
		vertical-align: middle;
		border-bottom: 1px dotted var(--rule);
	}
	.cov-table tr:last-child td {
		border-bottom: none;
	}
	.cov-table th.num,
	.cov-table td.num {
		text-align: right;
	}
	.cov-name {
		color: var(--ink);
	}
	.cov-bar-head {
		min-width: 16rem;
	}
	.cov-bar-cell {
		min-width: 12rem;
		width: 100%;
	}
	.cov-bar {
		display: flex;
		width: 100%;
		min-width: 12rem;
		max-width: 30rem;
		height: 14px;
		border: 1px solid var(--rule);
		overflow: hidden;
		position: relative;
	}
	.cov-bar-partial::after {
		content: '';
		position: absolute;
		inset: 0;
		pointer-events: none;
		background: repeating-linear-gradient(
			135deg,
			rgba(138, 106, 0, 0.28),
			rgba(138, 106, 0, 0.28) 2px,
			transparent 2px,
			transparent 7px
		);
	}
	.cov-seg {
		display: block;
		min-width: 1px;
	}
	.cov-seg-substrate { background: var(--ok-green); }
	.cov-seg-llm { background: var(--accent); }
	.cov-seg-abstain { background: var(--ink-muted); }
	.cov-seg-notrun {
		background: repeating-linear-gradient(
			135deg,
			var(--ink-faint),
			var(--ink-faint) 2px,
			var(--paper) 2px,
			var(--paper) 4px
		);
	}
	.cov-c-substrate { color: var(--ok-green); }
	.cov-c-llm { color: var(--accent); }
	.cov-c-abstain { color: var(--ink-muted); }
	.cov-c-notrun { color: var(--ink-faint); }

	.cov-aux {
		display: flex;
		flex-direction: column;
		gap: 0.3rem;
		margin-top: 1rem;
		font-family: var(--mono);
		font-size: 0.82rem;
	}
	.cov-aux-row {
		display: flex;
		gap: 0.6rem;
		align-items: baseline;
	}
	.cov-aux-label {
		color: var(--ink-muted);
		font-size: 0.74rem;
		min-width: 16rem;
	}
	.cov-aux-val {
		color: var(--ink);
		font-weight: 500;
		font-variant-numeric: tabular-nums;
	}
	.cov-empty {
		font-family: var(--serif);
		font-style: italic;
		color: var(--ink-muted);
		margin: 0;
	}
	.muted { color: var(--ink-faint); }
</style>
