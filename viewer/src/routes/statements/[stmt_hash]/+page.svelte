<script lang="ts">
	import { onMount } from 'svelte';
	import type { PageData } from './$types';
	import type { ScorerStepRow, StatementDetail, TruthLabelRow } from '$lib/db';
	import BeliefPrimitive from '$lib/components/BeliefPrimitive.svelte';
	import {
		evidenceParts,
		extractProbeCue,
		fmtBelief,
		pluralS,
		shortHash,
		verdictDisplay
	} from '$lib/format';

	let { data }: { data: PageData } = $props();
	const d = $derived(data.detail);
	const compareDetail = $derived(data.compare_detail);
	const probes = $derived(data.probes);
	const selectedRunIsPartial = $derived(
		d.selected_run_id != null &&
		d.selected_run_status != null &&
		d.selected_run_status !== 'succeeded'
	);

	function parseDbRefs(json: string): Array<[string, string]> {
		try {
			const o = JSON.parse(json);
			return Object.entries(o).map(([k, v]) => [k, String(v)]);
		} catch {
			return [];
		}
	}

	function epistemicsRow(label: string, val: boolean | null): string {
		if (val == null) return '';
		return val ? '✓' : '✗';
	}

	// Group truth labels by truth_set_id and target_kind
	function truthByTarget(
		labels: TruthLabelRow[],
		kind: string,
		targetId: string
	): TruthLabelRow[] {
		return labels.filter((l) => l.target_kind === kind && l.target_id === targetId);
	}

	const stmtTruthLabels = $derived(
		truthByTarget(d.truth_labels, 'stmt', d.stmt_hash)
	);

	const presentSets = $derived(
		Array.from(new Set(d.truth_labels.map((l) => l.truth_set_id))).sort()
	);
	const absentSets = $derived(
		d.registered_truth_sets.filter((tid) => !presentSets.includes(tid))
	);

	function shortTruthLabel(tid: string): string {
		return tid
			.replace(/^indra_/, '')
			.replace(/^source_db_/, 'src:')
			.replace(/^gold_pool_/, 'gold:')
			.replace(/^u2_per_probe_/, 'u2:');
	}

	// Aggregate our score across this statement's scored evidences (mean —
	// matches Phase 6 default aggregator + Phase 4 calibration target)
	const scoredEvidences = $derived.by(() => latestAggregateStepsFor(d));
	const persistedAggregateRowCount = $derived.by(() => aggregateStepsFor(d).length);
	const supersededAggregateRowCount = $derived.by(() =>
		Math.max(0, persistedAggregateRowCount - scoredEvidences.length)
	);
	const ourBelief = $derived.by(() => {
		const scores: number[] = [];
		for (const step of scoredEvidences) {
			try {
				const v = JSON.parse(step.output_json)?.score;
				if (typeof v === 'number') scores.push(v);
			} catch {}
		}
		return scores.length ? scores.reduce((a, b) => a + b, 0) / scores.length : null;
	});
	const verdictTally = $derived.by((): Record<string, number> => {
		// Permissive: count any string verdict, not just the known three.
		// If Python ever emits a new verdict, the tally stays honest about
		// totals (correct+incorrect+abstain+others = total scored evidences).
		const out: Record<string, number> = { correct: 0, incorrect: 0, abstain: 0 };
		for (const step of scoredEvidences) {
			try {
				const v = JSON.parse(step.output_json)?.verdict;
				if (typeof v === 'string') out[v] = (out[v] ?? 0) + 1;
			} catch {}
		}
		return out;
	});

	let hoveredEvidenceHash: string | null = $state(null);
	let expandedEvHash: string | null = $state(null);
	let expandedCompareEvidenceHash: string | null = $state(null);
	let copiedHash: string | null = $state(null);
	let expandedCall: string | null = $state(null);

	function toggleExpand(h: string) {
		expandedEvHash = expandedEvHash === h ? null : h;
	}

	const STEP_KINDS: Array<[string, string]> = [
		['parse_claim', 'parse_claim'],
		['build_context', 'build_context'],
		['substrate_route', 'route'],
		['subject_role_probe', 'subj_role'],
		['object_role_probe', 'obj_role'],
		['relation_axis_probe', 'axis'],
		['scope_probe', 'scope'],
		['grounding', 'grounding'],
		['adjudicate', 'adjudicate']
	];

	function archMark(architecture: string | null | undefined): string {
		if (architecture === 'monolithic') return '[M]';
		if (architecture === 'decomposed') return '[D]';
		return '[?]';
	}

	function archOrder(architecture: string | null | undefined): number {
		if (architecture === 'monolithic') return 0;
		if (architecture === 'decomposed') return 1;
		return 2;
	}

	function partialRunSampleLabel(status: string | null | undefined): string {
		if (status === 'running') return 'live run sample';
		if (status === 'canceled') return 'canceled run sample';
		if (status === 'failed') return 'failed run sample';
		return 'terminated run sample';
	}

	function fmtDelta(d: number): string {
		const sign = d >= 0 ? '+' : '−';
		return `${sign}${Math.abs(d).toFixed(2)}`;
	}

	function safeParseJSON(s: string | null): Record<string, unknown> | null {
		if (!s) return null;
		try {
			return JSON.parse(s);
		} catch {
			return null;
		}
	}

	function aggregateStepsFor(detail: StatementDetail): ScorerStepRow[] {
		return detail.scorer_steps.filter((s) => s.step_kind === 'aggregate');
	}

	function compareStepRecency(a: ScorerStepRow, b: ScorerStepRow): number {
		const at = a.started_at ? Date.parse(a.started_at) : 0;
		const bt = b.started_at ? Date.parse(b.started_at) : 0;
		if (bt !== at) return bt - at;
		return b.step_hash.localeCompare(a.step_hash);
	}

	function aggregateStepsForEvidence(detail: StatementDetail, evidenceHash: string): ScorerStepRow[] {
		return aggregateStepsFor(detail)
			.filter((s) => s.evidence_hash === evidenceHash)
			.sort(compareStepRecency);
	}

	function latestAggregateStepsFor(detail: StatementDetail): ScorerStepRow[] {
		const byEvidence = new Map<string, ScorerStepRow>();
		for (const step of aggregateStepsFor(detail).sort(compareStepRecency)) {
			if (!step.evidence_hash || byEvidence.has(step.evidence_hash)) continue;
			byEvidence.set(step.evidence_hash, step);
		}
		return Array.from(byEvidence.values());
	}

	function aggregateStepForEvidence(detail: StatementDetail, evidenceHash: string): ScorerStepRow | null {
		return aggregateStepsForEvidence(detail, evidenceHash)[0] ?? null;
	}

	function aggregateOutputForEvidence(detail: StatementDetail, evidenceHash: string): Record<string, unknown> | null {
		const step = aggregateStepForEvidence(detail, evidenceHash);
		return step ? safeParseJSON(step.output_json) : null;
	}

	function nativeStepsForEvidence(detail: StatementDetail, evidenceHash: string): ScorerStepRow[] {
		return detail.scorer_steps.filter((s) => s.evidence_hash === evidenceHash && s.step_kind !== 'aggregate');
	}

	function evidenceAnchor(evidenceHash: string): string {
		return `evidence-${evidenceHash}`;
	}

	function evidenceRunHref(detail: StatementDetail, evidenceHash?: string): string {
		const href = detail.selected_run_id
			? `/statements/${detail.stmt_hash}?run_id=${detail.selected_run_id}`
			: `/statements/${detail.stmt_hash}`;
		return evidenceHash ? `${href}#${evidenceAnchor(evidenceHash)}` : href;
	}

	const compareDetails = $derived.by(() => {
		if (!compareDetail) return [];
		return [d, compareDetail].sort(
			(a, b) => archOrder(a.selected_architecture) - archOrder(b.selected_architecture)
		);
	});

	const compareIsArchitecturePair = $derived.by(() => {
		const archs = new Set(compareDetails.map((detail) => detail.selected_architecture));
		return archs.has('monolithic') && archs.has('decomposed');
	});
	const compareModeActive = $derived(Boolean(compareDetail && compareDetails.length === 2));

	function pairedStatementHref(primary: StatementDetail): string {
		const params = new URLSearchParams();
		if (primary.selected_run_id) params.set('run_id', primary.selected_run_id);
		const secondary = compareDetails.find((detail) => detail.selected_run_id !== primary.selected_run_id);
		if (secondary?.selected_run_id) params.set('compare_run_id', secondary.selected_run_id);
		const query = params.toString();
		return `/statements/${d.stmt_hash}${query ? `?${query}` : ''}#paired-trace-compare`;
	}

	function pairedLaneState(detail: StatementDetail): string {
		return detail.selected_run_id === d.selected_run_id ? 'primary trace below' : 'make primary';
	}

	type CompareEvidenceRecord = StatementDetail['evidences'][number] & { score_only: boolean };

	const compareEvidenceRows = $derived.by((): CompareEvidenceRecord[] => {
		const byHash = new Map<string, CompareEvidenceRecord>();
		for (const evidence of d.evidences) {
			byHash.set(evidence.evidence_hash, { ...evidence, score_only: false });
		}
		for (const detail of compareDetails) {
			for (const step of latestAggregateStepsFor(detail)) {
				if (!step.evidence_hash || byHash.has(step.evidence_hash)) continue;
				byHash.set(step.evidence_hash, {
					evidence_hash: step.evidence_hash,
					source_api: null,
					source_id: null,
					pmid: null,
					text: null,
					is_direct: null,
					is_negated: null,
					is_curated: null,
					epistemics_json: null,
					score_only: true
				});
			}
		}
		return Array.from(byHash.values());
	});

	function runScoreSummary(detail: StatementDetail): string {
		const steps = latestAggregateStepsFor(detail);
		const persistedN = aggregateStepsFor(detail).length;
		if (steps.length === 0) return 'no aggregate rows for this statement';
		const verdicts: Record<string, number> = {};
		let scoreSum = 0;
		let scoreN = 0;
		for (const step of steps) {
			const out = safeParseJSON(step.output_json);
			const verdict = typeof out?.verdict === 'string' ? out.verdict : 'missing verdict';
			verdicts[verdict] = (verdicts[verdict] ?? 0) + 1;
			if (typeof out?.score === 'number') {
				scoreSum += out.score;
				scoreN += 1;
			}
		}
		const verdictText = Object.entries(verdicts)
			.map(([verdict, n]) => `${verdictDisplay(verdict)} ${n}`)
			.join(' · ');
		const scoreText = scoreN > 0 ? `mean ${fmtBelief(scoreSum / scoreN)}` : 'mean not captured';
		const supersededN = Math.max(0, persistedN - steps.length);
		const persistedText = supersededN > 0
			? ` · ${supersededN} superseded aggregate row${supersededN === 1 ? '' : 's'} persisted`
			: '';
		return `${steps.length} active evidence row${steps.length === 1 ? '' : 's'} · ${scoreText} · ${verdictText}${persistedText}`;
	}

	function nativeGrammarLine(detail: StatementDetail, evidenceHash: string): string {
		const out = aggregateOutputForEvidence(detail, evidenceHash);
		if (!out) return `not scored in run ${detail.selected_run_id?.slice(0, 8) ?? 'unknown'}`;
		const variants = aggregateStepsForEvidence(detail, evidenceHash);
		const variantText = variants.length > 1
			? ` · ${variants.length - 1} superseded aggregate row${variants.length - 1 === 1 ? '' : 's'}`
			: '';
		if (detail.selected_architecture === 'monolithic') {
			const tier = typeof out.tier === 'string' ? out.tier : 'tier not captured';
			const callLog = Array.isArray(out.call_log) ? out.call_log.length : 0;
			const selectedExamples = 'selected_example_ids' in out || 'selected_examples' in out
				? 'selected examples captured'
				: 'selected examples not persisted';
			return `${tier} · model calls ${callLog} · ${selectedExamples}${variantText}`;
		}
		if (detail.selected_architecture === 'decomposed') {
			const nativeSteps = nativeStepsForEvidence(detail, evidenceHash);
			const probeN = nativeSteps.filter((step) => step.step_kind.endsWith('_probe')).length;
			const stepText = nativeSteps.length > 0
				? nativeSteps.map((step) => step.step_kind.replace(/_probe$/, '')).join(', ')
				: 'aggregate only';
			return `${nativeSteps.length} native step${nativeSteps.length === 1 ? '' : 's'} · probes ${probeN} · ${stepText}${variantText}`;
		}
		return 'architecture not captured for this run';
	}

	function aggregateStepOutcomeLine(step: ScorerStepRow): string {
		const out = safeParseJSON(step.output_json);
		if (!out) return 'unparseable aggregate output';
		const score = typeof out.score === 'number' ? fmtBelief(out.score) : 'score not captured';
		const verdict = typeof out.verdict === 'string' ? verdictDisplay(out.verdict) : 'verdict not captured';
		const confidence = typeof out.confidence === 'string' ? `${out.confidence} confidence` : 'confidence not captured';
		return `${score} · ${verdict} · ${confidence}`;
	}

	function selectedExamplesForOutput(out: Record<string, unknown> | null): Array<Record<string, unknown>> {
		if (!out || !Array.isArray(out.selected_examples)) return [];
		return out.selected_examples.filter((ex): ex is Record<string, unknown> => Boolean(ex) && typeof ex === 'object' && !Array.isArray(ex));
	}

	function selectedExampleIdsForOutput(out: Record<string, unknown> | null): string[] {
		if (!out || !Array.isArray(out.selected_example_ids)) return [];
		return out.selected_example_ids.filter((id): id is string => typeof id === 'string' && id.length > 0);
	}

	function evidenceOutcomeLine(detail: StatementDetail, evidenceHash: string): string {
		const out = aggregateOutputForEvidence(detail, evidenceHash);
		if (!out) return 'not scored';
		const score = typeof out.score === 'number' ? fmtBelief(out.score) : 'score not captured';
		const verdict = typeof out.verdict === 'string' ? verdictDisplay(out.verdict) : 'verdict not captured';
		const confidence = typeof out.confidence === 'string' ? `${out.confidence} confidence` : 'confidence not captured';
		return `${score} · ${verdict} · ${confidence}`;
	}

	function compareExpansionText(evidenceHash: string): string {
		return expandedCompareEvidenceHash === evidenceHash ? 'collapse native rows' : 'compare native rows';
	}

	onMount(() => {
		const syncEvidenceHash = () => {
			const hash = decodeURIComponent(window.location.hash.slice(1));
			if (hash.startsWith('evidence-')) {
				expandedEvHash = hash.slice('evidence-'.length);
			} else if (hash.startsWith('compare-evidence-')) {
				expandedCompareEvidenceHash = hash.slice('compare-evidence-'.length);
			}
		};
		syncEvidenceHash();
		window.addEventListener('hashchange', syncEvidenceHash);
		return () => window.removeEventListener('hashchange', syncEvidenceHash);
	});

	function firstStepOutput(kind: string, evidenceHash?: string | null): Record<string, unknown> | null {
		for (const s of d.scorer_steps) {
			if (s.step_kind !== kind) continue;
			if (evidenceHash !== undefined && s.evidence_hash !== evidenceHash) continue;
			const o = safeParseJSON(s.output_json);
			if (o) return o;
		}
		return null;
	}

	function probeOutputsForEvidence(evidenceHash: string): Array<{ kind: string; out: Record<string, unknown>; step: ScorerStepRow }> {
		const probeKinds = ['subject_role_probe', 'object_role_probe', 'relation_axis_probe', 'scope_probe'];
		const rows = d.scorer_steps.filter((s) => probeKinds.includes(s.step_kind) && s.evidence_hash === evidenceHash);
		return rows
			.map((step) => {
				const out = safeParseJSON(step.output_json);
				return out ? { kind: step.step_kind, out, step } : null;
			})
			.filter((x): x is { kind: string; out: Record<string, unknown>; step: ScorerStepRow } => x !== null);
	}

	function cueForEvidence(evidenceHash: string): string | null {
		const probes = probeOutputsForEvidence(evidenceHash);
		for (const { out } of probes) {
			const c = extractProbeCue((out.rationale as string | null) ?? null);
			if (c) return c;
		}
		return null;
	}

	const probeStepLabels: Record<string, string> = {
		subject_role_probe: 'subj-role',
		object_role_probe: 'obj-role',
		relation_axis_probe: 'relation-axis',
		scope_probe: 'scope'
	};

	/**
	 * Trace narrative for the whole statement. Returns an ordered list of
	 * sentence lines explaining the journey the pipeline took. Unrun steps
	 * collapse into a single explanatory line instead of N missing dots.
	 */
	const traceLines = $derived.by(() => {
		const lines: Array<{ key: string; prose: string; muted?: boolean }> = [];

		if (d.selected_architecture === 'monolithic') {
			const aggregateOutputs = scoredEvidences
				.map((s) => safeParseJSON(s.output_json))
				.filter((o): o is Record<string, unknown> => o !== null);
			if (aggregateOutputs.length > 0) {
				lines.push({
					key: 'monolithic_native',
					prose: `Monolithic native trace: ${aggregateOutputs.length} aggregate evidence row${aggregateOutputs.length === 1 ? '' : 's'}; decomposed probe slots are not expected.`
				});
				const tierCounts: Record<string, number> = {};
				let rowsWithCalls = 0;
				let rowsWithSelectedExamples = 0;
				let parseFailures = 0;
				for (const out of aggregateOutputs) {
					const tier = typeof out.tier === 'string' ? out.tier : 'tier-not-captured';
					tierCounts[tier] = (tierCounts[tier] ?? 0) + 1;
					if (Array.isArray(out.call_log) && out.call_log.length > 0) rowsWithCalls += 1;
					if (
						(Array.isArray(out.selected_example_ids) && out.selected_example_ids.length > 0) ||
						(Array.isArray(out.selected_examples) && out.selected_examples.length > 0)
					) rowsWithSelectedExamples += 1;
					if (out.verdict == null) parseFailures += 1;
				}
				const tierText = Object.entries(tierCounts)
					.map(([tier, n]) => `${tier} ${n}`)
					.join(', ');
				lines.push({
					key: 'monolithic_tiers',
					prose: `Tier path: ${tierText}.`
				});
				lines.push({
					key: 'monolithic_calls',
					muted: rowsWithCalls === 0,
					prose: rowsWithCalls > 0
						? `Model call logs are captured for ${rowsWithCalls} evidence row${rowsWithCalls === 1 ? '' : 's'}; selected example IDs are captured for ${rowsWithSelectedExamples}.`
						: 'No model call log is captured; this may be a deterministic auto-reject or older instrumentation.'
				});
				if (parseFailures > 0) {
					lines.push({
						key: 'monolithic_parse_failures',
						muted: true,
						prose: `${parseFailures} aggregate row${parseFailures === 1 ? '' : 's'} did not persist a parsed verdict.`
					});
				}
			}
			return lines;
		}

		const pc = firstStepOutput('parse_claim');
		if (pc) {
			const stype = pc.stmt_type ?? d.indra_type;
			const agentList = d.agents.map((a) => a.name).join(' / ');
			lines.push({
				key: 'parse_claim',
				prose: `Parsed the claim as ${stype}${agentList ? ` (${agentList})` : ''}.`
			});
		}

		const bc = firstStepOutput('build_context');
		if (bc) {
			const aliasN = (bc.n_aliases as number | undefined) ?? 0;
			const relN = (bc.n_detected_relations as number | undefined) ?? 0;
			lines.push({
				key: 'build_context',
				prose: `Built context: ${aliasN} alias${aliasN === 1 ? '' : 'es'}, ${relN} relation${relN === 1 ? '' : 's'} detected.`
			});
		}

		const sr = d.scorer_steps.find((s) => s.step_kind === 'substrate_route');
		if (sr) {
			lines.push({
				key: 'substrate_route',
				prose: 'Routed to the deterministic substrate layer (regex / Gilda) before any LLM probe.'
			});
		}

		const probeOrder = ['subject_role_probe', 'object_role_probe', 'relation_axis_probe', 'scope_probe'];
		const ranProbeKinds: string[] = [];
		const skippedProbeKinds: string[] = [];
		for (const kind of probeOrder) {
			const out = firstStepOutput(kind);
			if (out && (out.answer ?? null) !== null && out.answer !== 'abstain') {
				ranProbeKinds.push(kind);
				const source = out.source ?? '?';
				const conf = out.confidence ?? '?';
				const cue = extractProbeCue((out.rationale as string | null) ?? null);
				const cueText = cue ? ` (cue: “${cue}”)` : '';
				lines.push({
					key: kind,
					prose: `${probeStepLabels[kind]}: ${out.answer} — ${source}, ${conf} confidence${cueText}.`
				});
			} else {
				skippedProbeKinds.push(probeStepLabels[kind]);
			}
		}
		if (skippedProbeKinds.length > 0 && ranProbeKinds.length > 0) {
			lines.push({
				key: 'probes_skipped',
				muted: true,
				prose: `${skippedProbeKinds.join(', ')} did not fire — substrate's earlier finding short-circuited the chain.`
			});
		} else if (skippedProbeKinds.length > 0 && ranProbeKinds.length === 0) {
			lines.push({
				key: 'probes_all_skipped',
				muted: true,
				prose: `No probes fired in this run.`
			});
		}

		const gr = firstStepOutput('grounding');
		if (gr) {
			lines.push({
				key: 'grounding',
				prose: `Checked entity grounding.`
			});
		}

		const agg = firstStepOutput('aggregate');
		if (agg) {
			const verdict = (agg.verdict as string | null) ?? '?';
			const conf = (agg.confidence as string | null) ?? '?';
			const score = agg.score as number | null;
			const scoreText = typeof score === 'number' ? score.toFixed(2) : '—';
			lines.push({
				key: 'aggregate',
				prose: `Aggregated to verdict: ${verdictDisplay(verdict)} (${conf} confidence, score ${scoreText}).`
			});
		}

		return lines;
	});

	/** epistemics flag → reader-facing phrase */
	function epistemicsLine(e: { is_direct: boolean | null; is_negated: boolean | null; is_curated: boolean | null }): string {
		const parts: string[] = [];
		if (e.is_direct === false) parts.push('indirect citation');
		else if (e.is_direct === true) parts.push('direct citation');
		if (e.is_negated === true) parts.push('explicitly negated');
		else if (e.is_negated === false) parts.push('not negated');
		if (e.is_curated === true) parts.push('human-curated');
		else if (e.is_curated === false) parts.push('not human-curated');
		return parts.length === 0 ? 'no epistemic flags set' : parts.join(' · ');
	}

	/** Reader-friendly description for the truth-set IDs we ship. */
	function truthSetDescription(tid: string): string {
		const known: Record<string, string> = {
			indra_published_belief: "INDRA's published belief score for the statement (prior)",
			indra_grounding: 'entity grounding mappings from INDRA (db_refs)',
			indra_epistemics: "INDRA's epistemic flags on the evidence (direct / negated / curated)",
			demo_gold: 'hand-labeled gold examples used for P/R/F1'
		};
		return known[tid] ?? tid;
	}

	let showDebug: boolean = $state(false);
</script>

<svelte:head><title>{d.indra_type} · {shortHash(d.stmt_hash)} · INDRA Belief</title></svelte:head>

	<header>
		<div class="crumb">
			<a href="/">corpus</a><span class="sep"> / </span><a href="/statements">statements</a><span class="sep"> / </span><strong>{shortHash(d.stmt_hash)}</strong>
		</div>
		<div class="meta">
			{#if compareModeActive}
				<span class="compare-mode-label">paired trace compare</span>
				<nav class="paired-run-switcher" aria-label="paired statement trace lanes">
					{#each compareDetails as detail}
						{@const isPrimaryLane = detail.selected_run_id === d.selected_run_id}
						{#if isPrimaryLane}
							<span
								class={`paired-run-token paired-run-token-${detail.selected_architecture ?? 'unknown'} paired-run-token-active`}
								aria-current="page"
							>
								<strong>{archMark(detail.selected_architecture)} run {detail.selected_run_id?.slice(0, 8) ?? 'unscored'}</strong>
								<span>{pairedLaneState(detail)}</span>
							</span>
						{:else}
							<a
								class={`paired-run-token paired-run-token-${detail.selected_architecture ?? 'unknown'}`}
								href={pairedStatementHref(detail)}
							>
								<strong>{archMark(detail.selected_architecture)} run {detail.selected_run_id?.slice(0, 8) ?? 'unscored'}</strong>
								<span>{pairedLaneState(detail)}</span>
							</a>
						{/if}
					{/each}
				</nav>
				{#if d.selected_run_id}
					<a class="single-run-exit" href={`/statements/${d.stmt_hash}?run_id=${d.selected_run_id}#main`}>single-run view</a>
				{/if}
			{:else}
				{#if d.selected_run_id}
					<span>{archMark(d.selected_architecture)} <a href={`/runs/${d.selected_run_id}`}>run {d.selected_run_id.slice(0, 8)}</a> · {d.selected_scorer_version ?? '—'} · {d.selected_run_status ?? 'unknown'}</span>
				{:else}
					<span class="muted">unscored</span>
				{/if}
				{#if d.available_runs.length > 1}
					<select aria-label="statement run" onchange={(ev) => {
						const v = (ev.target as HTMLSelectElement).value;
						window.location.href = v ? `/statements/${d.stmt_hash}?run_id=${v}` : `/statements/${d.stmt_hash}`;
					}}>
						{#each d.available_runs as r}
							<option value={r.run_id} selected={r.run_id === d.selected_run_id}>
								{archMark(r.architecture)} {r.run_id.slice(0, 8)} · {r.scorer_version} · {r.started_at.replace(/\.\d+$/, '')}
							</option>
						{/each}
					</select>
				{/if}
			{/if}
		</div>
	</header>

<main id="main">
	{#if selectedRunIsPartial}
		<section class="run-warning" role="note">
			<span>{partialRunSampleLabel(d.selected_run_status)}</span>
			<p>
				This statement is showing scorer rows from a <strong>{d.selected_run_status}</strong> run.
				They are persisted rows, not a completed run context; aggregate verdicts and traces may change or be biased toward earlier worker-order evidence.
			</p>
		</section>
	{/if}

	<section class="stmt-header">
		<BeliefPrimitive
			stmt={{ stmt_hash: d.stmt_hash, indra_type: d.indra_type, agents: d.agents }}
			our_score={ourBelief}
			indra_score={d.indra_belief}
			probes={probes}
			evidences={[]}
			mode="full"
			level="h1"
		/>
		<div class="stmt-meta">
			{#if ourBelief != null}
				<span class="verdict-tally">
					{#each Object.entries(verdictTally).filter(([_, n]) => n > 0) as [v, n], i}
						{#if i > 0}<span class="dot">·</span>{/if}
						<span class="vt-{v}">{v} {n}</span>
					{/each}
					<span class="muted">of {scoredEvidences.length}</span>
				</span>
				<span class="dot">·</span>
			{:else}
				<span class="hint">unscored · INDRA {fmtBelief(d.indra_belief)}</span>
				<span class="dot">·</span>
			{/if}
			<span>supports {d.supports_count}/{d.supported_by_count}</span>
			<span class="dot">·</span>
			<span class="source">{d.source_dump_id ?? '<no source_dump>'}</span>
		</div>
	</section>

	{#if compareDetail && compareDetails.length === 2}
		<section id="paired-trace-compare" class="paired-trace-compare" aria-label="paired statement trace comparison">
			<div class="paired-trace-head">
				<div>
					<h2>paired statement trace compare</h2>
					<p>
						One statement, one evidence list, two native grammars. This panel compares row presence and trace capture;
						paired metrics still belong to the pair workbench overlap denominator.
					</p>
				</div>
				{#if d.selected_run_id}
					<a href={`/statements/${d.stmt_hash}?run_id=${d.selected_run_id}`}>single-run view</a>
				{/if}
			</div>
			{#if !compareIsArchitecturePair}
				<p class="compare-warning">
					This is not a monolithic/decomposed pair. Architecture hypotheses need one [M] run and one [D] run; same-architecture comparisons stay diagnostic only.
				</p>
			{/if}
			<div class="compare-run-grid">
				{#each compareDetails as detail}
					<article class="compare-run-card compare-run-{detail.selected_architecture ?? 'unknown'}">
						<h3>{archMark(detail.selected_architecture)} run {detail.selected_run_id?.slice(0, 8) ?? 'unscored'}</h3>
						<p>{detail.selected_scorer_version ?? 'scorer unknown'} · {detail.selected_run_status ?? 'status unknown'}</p>
						<strong>{runScoreSummary(detail)}</strong>
						<a href={evidenceRunHref(detail)}>open this native trace</a>
					</article>
				{/each}
			</div>
			<div class="compare-evidence-list">
				{#each compareEvidenceRows as e}
					{@const isCompareOpen = expandedCompareEvidenceHash === e.evidence_hash}
					<article id={`compare-evidence-${e.evidence_hash}`} class="compare-evidence" class:compare-score-only={e.score_only}>
						<div class="compare-evidence-head">
							<div>
								<code>{shortHash(e.evidence_hash)}</code>
								<span>{e.source_api ?? (e.score_only ? 'score row without evidence join' : 'no source')}</span>
								{#if e.pmid}<span>pmid:{e.pmid}</span>{/if}
							</div>
							<button
								type="button"
								aria-expanded={isCompareOpen}
								onclick={() => {
									expandedCompareEvidenceHash = isCompareOpen ? null : e.evidence_hash;
								}}
							>{compareExpansionText(e.evidence_hash)}</button>
						</div>
						<p class="compare-evidence-text">{e.text ?? 'No evidence text row joined for this scored evidence hash; scorer rows remain inspectable below.'}</p>
						<div class="compare-native-grid">
								{#each compareDetails as detail}
									{@const out = aggregateOutputForEvidence(detail, e.evidence_hash)}
									{@const nativeHref = e.score_only ? evidenceRunHref(detail) : evidenceRunHref(detail, e.evidence_hash)}
									<section class={`compare-side compare-side-${detail.selected_architecture ?? 'unknown'}`} class:compare-side-missing={!out}>
										<h4>{archMark(detail.selected_architecture)} {detail.selected_architecture ?? 'unknown'}</h4>
										<p class="compare-outcome">{evidenceOutcomeLine(detail, e.evidence_hash)}</p>
										<p class="compare-native-line">{nativeGrammarLine(detail, e.evidence_hash)}</p>
										<a href={nativeHref} class:compare-missing-link={!out}>
											{out
												? `inspect ${archMark(detail.selected_architecture)} ${e.score_only ? 'statement trace' : 'evidence trace'}`
												: `open ${archMark(detail.selected_architecture)} evidence row`}
										</a>
									</section>
								{/each}
						</div>
						{#if isCompareOpen}
							<div class="compare-native-detail">
								{#each compareDetails as detail}
									{@const out = aggregateOutputForEvidence(detail, e.evidence_hash)}
									{@const nativeSteps = nativeStepsForEvidence(detail, e.evidence_hash)}
									<section class={`compare-side compare-side-${detail.selected_architecture ?? 'unknown'}`}>
										<h4>{archMark(detail.selected_architecture)} native rows</h4>
										{#if !out}
											<p class="exp-not-captured">No aggregate row for this evidence in run {detail.selected_run_id?.slice(0, 8) ?? 'unknown'}.</p>
										{:else if detail.selected_architecture === 'monolithic'}
											{@const aggregateRows = aggregateStepsForEvidence(detail, e.evidence_hash)}
											{#if aggregateRows.length > 1}
												<div class="compare-aggregate-rows">
													<h5>aggregate rows for this evidence</h5>
													{#each aggregateRows as step, index}
														<p><span>{index === 0 ? 'active latest' : 'superseded'}</span> {aggregateStepOutcomeLine(step)} <code>{shortHash(step.step_hash)}</code></p>
													{/each}
												</div>
											{/if}
											<dl class="mono-fields">
												<div><dt>tier</dt><dd>{out.tier ?? 'not captured'}</dd></div>
												<div><dt>grounding</dt><dd>{out.grounding_status ?? 'not captured'}</dd></div>
												<div><dt>provenance</dt><dd>{out.provenance_triggered === true ? 'triggered' : out.provenance_triggered === false ? 'not triggered' : 'not captured'}</dd></div>
												<div><dt>parse</dt><dd>{out.verdict == null ? 'verdict not parsed' : `${out.verdict} / ${out.confidence ?? '—'}`}</dd></div>
											</dl>
											<p class="exp-not-captured">Selected example IDs are not persisted unless the aggregate row explicitly captured them.</p>
										{:else if nativeSteps.length > 0}
											{@const aggregateRows = aggregateStepsForEvidence(detail, e.evidence_hash)}
											{#if aggregateRows.length > 1}
												<div class="compare-aggregate-rows">
													<h5>aggregate rows for this evidence</h5>
													{#each aggregateRows as step, index}
														<p><span>{index === 0 ? 'active latest' : 'superseded'}</span> {aggregateStepOutcomeLine(step)} <code>{shortHash(step.step_hash)}</code></p>
													{/each}
												</div>
											{/if}
											<table class="compare-step-table">
												<thead>
													<tr><th>step</th><th>source</th><th>output</th></tr>
												</thead>
												<tbody>
													{#each nativeSteps as step}
														{@const stepOut = safeParseJSON(step.output_json)}
														<tr>
															<td>{step.step_kind}</td>
															<td>{step.is_substrate_answered === true ? 'substrate' : step.is_substrate_answered === false ? 'LLM' : '—'}</td>
															<td>
																{#if stepOut?.answer}<span>{stepOut.answer}</span>{/if}
																{#if stepOut?.confidence}<span class="muted"> · {stepOut.confidence}</span>{/if}
																{#if stepOut?.rationale}<span class="muted"> · {stepOut.rationale}</span>{/if}
																{#if stepOut?.stmt_type}<span>{stepOut.stmt_type}</span>{/if}
															</td>
														</tr>
													{/each}
												</tbody>
											</table>
										{:else}
											<p class="exp-not-captured">Only the aggregate row is persisted for this evidence; native decomposed substeps are not captured.</p>
										{/if}
									</section>
								{/each}
							</div>
						{/if}
					</article>
				{/each}
			</div>
		</section>
	{/if}

	<div class="cols">
		<!-- Center column: scorer trace + evidences -->
		<div class="trace">
			<h2>how the pipeline scored this</h2>
			{#if traceLines.length === 0}
				<p class="hint">
					Not yet scored — invoke <code>score_corpus(con, [stmt], decompose=True)</code> to see the pipeline trace.
				</p>
			{:else}
				<ol class="trace-narrative">
					{#each traceLines as line, i}
						<li class="trace-line" class:trace-muted={line.muted}>
							<span class="trace-num">{i + 1}</span>
							<span class="trace-prose">{line.prose}</span>
						</li>
					{/each}
				</ol>
			{/if}

			<h2>
				evidences
				<span class="counter">
					{#if d.selected_run_id}{scoredEvidences.length} of {/if}{d.evidences.length}
				</span>
			</h2>
			{#if supersededAggregateRowCount > 0}
				<p class="ev-section-note">
					{persistedAggregateRowCount.toLocaleString()} aggregate scorer rows persist for this statement; this view uses the latest
					row per evidence and discloses superseded rows inside the evidence expansion.
				</p>
			{/if}
			<p class="ev-section-note">
				Each evidence carries a verdict ∈ {'{'}supported, contradicted, abstained{'}'} and a confidence ∈ {'{'}high, medium, low{'}'}. The displayed score is a lookup from that (verdict, confidence) pair — only 7 distinct values are possible:
				<span class="ev-bucket-table" title="src/indra_belief/scorers/commitments.py::_VERDICT_SCORE">
					correct/high <code>0.95</code> · correct/medium <code>0.80</code> · correct/low <code>0.65</code> · abstain/* <code>0.50</code> · incorrect/low <code>0.35</code> · incorrect/medium <code>0.20</code> · incorrect/high <code>0.05</code>
				</span>
			</p>
			{#each d.evidences as e}
				{@const evTruth = truthByTarget(d.truth_labels, 'evidence', e.evidence_hash)}
					{@const score = aggregateStepForEvidence(d, e.evidence_hash)}
					{@const out = score ? JSON.parse(score.output_json) : null}
				{@const cue = cueForEvidence(e.evidence_hash)}
				{@const parts = evidenceParts(e.text, cue)}
				<!-- D10 Bret Victor lever: hovering an evidence card highlights truth-set rows
				     that judge any of its labels (epistemics on this evidence) -->
				{@const isExpanded = expandedEvHash === e.evidence_hash}
				<!-- Whole-article click target per iter-3 brutalist BLOCKER #15:
				     body sentence used to be dead space. Now the entire <article> is
				     a button. Inner .ev-expanded panel has pointer-events:none so
				     clicking inside the expansion does NOT collapse the parent. -->
				<!-- svelte-ignore a11y_no_noninteractive_tabindex -->
					<article id={evidenceAnchor(e.evidence_hash)}
						class="evidence" class:ev-expanded-state={isExpanded}
					class:ev-clickable={!!score}
					data-evidence-hash={e.evidence_hash}
					role={score ? 'button' : undefined}
					tabindex={score ? 0 : undefined}
					aria-expanded={score ? isExpanded : undefined}
					onmouseenter={() => { hoveredEvidenceHash = e.evidence_hash; }}
					onmouseleave={() => { hoveredEvidenceHash = null; }}
					onclick={() => { if (score) toggleExpand(e.evidence_hash); }}
					onkeydown={(ev) => {
						if (score && (ev.key === 'Enter' || ev.key === ' ')) {
							ev.preventDefault();
							toggleExpand(e.evidence_hash);
						}
					}}>
					{#if out}
						<div class="ev-verdict-line">
							<span class="ev-verdict ev-verdict-{out.verdict}">verdict: {verdictDisplay(out.verdict)}</span>
							<span class="ev-confidence">{out.confidence ?? '—'} confidence</span>
							<span class="ev-score">score <span class="ev-score-num">{out.score?.toFixed?.(2) ?? '—'}</span></span>
							{#if score}
								<span class="ev-chevron" aria-hidden="true">{isExpanded ? '▾ collapse' : '▸ expand'}</span>
							{/if}
						</div>
					{:else if d.selected_run_id}
						<div class="ev-verdict-line ev-verdict-line-missing">
							<span class="ev-not-scored">not scored in run {d.selected_run_id.slice(0, 8)}</span>
						</div>
					{/if}
					<div class="ev-meta-secondary">
						<span class="ev-source">[{e.source_api ?? 'no source'}]</span>
						{#if e.pmid}
							<span class="ev-pmid">pmid:{e.pmid}</span>
						{/if}
						<code class="ev-hash" title={e.evidence_hash}>{shortHash(e.evidence_hash)}</code>
						{#if score?.latency_ms != null && score.latency_ms > 0}
							<span class="latency">{score.latency_ms}ms</span>
						{/if}
					</div>
					<p class="ev-text">{#each parts as part}{#if part.highlight}<mark class="ev-text-cue">{part.text}</mark>{:else}{part.text}{/if}{/each}</p>
					<div class="ev-flags">
						<span class="ev-flags-prose">{epistemicsLine(e)}</span>
						{#if evTruth.length > 0}
							<span class="truth-stamp">· {evTruth.length} truth label{pluralS(evTruth.length)}</span>
						{/if}
					</div>
					{#if out && out.reasons && out.reasons.length > 0}
						<div class="reasons">
							<span class="reasons-label">reasons</span>
							{#each out.reasons as r}
								<span class="reason-token">· {r}</span>
							{/each}
						</div>
					{/if}
					{#if isExpanded && out}
						{@const callLog = (out.call_log as Array<Record<string, unknown>> | undefined) ?? []}
						{@const evSteps = d.scorer_steps.filter((s) => s.evidence_hash === e.evidence_hash && s.step_kind !== 'aggregate')}
						<!-- svelte-ignore a11y_click_events_have_key_events -->
						<!-- svelte-ignore a11y_no_static_element_interactions -->
						<div class="ev-expanded" onclick={(ev) => ev.stopPropagation()}>

							<!-- Aggregate rationale (the scorer's prose summary for this evidence) -->
							{#if out.rationale}
								<div class="exp-block">
									<h4 class="exp-h">rationale</h4>
									<p class="exp-rationale">{out.rationale}</p>
								</div>
							{/if}

							<!-- LLM calls: each row is one model invocation; click to expand. -->
							<div class="exp-block">
								<h4 class="exp-h">LLM calls <span class="exp-h-count">({callLog.length})</span>
									<span class="exp-h-note">one row per call sent to the model during this aggregate step · click a row to see the prompt and the model's response</span>
								</h4>
								{#if callLog.length === 0}
									<p class="exp-empty">
										{#if d.selected_architecture === 'monolithic'}
											no model calls captured — this can be a tier-1 deterministic reject or older aggregate-only instrumentation
										{:else}
											no LLM calls — every probe was answered by the substrate layer (regex / Gilda)
										{/if}
									</p>
								{:else}
									<div class="call-list">
										{#each callLog as call, ci}
											{@const callKey = `${e.evidence_hash}-${ci}`}
											{@const isCallOpen = expandedCall === callKey}
											{@const hasContent = typeof call.content === 'string' && call.content.length > 0}
											{@const hasReasoning = typeof call.reasoning === 'string' && call.reasoning.length > 0}
											{@const hasMessages = Array.isArray(call.messages)}
											{@const hasError = typeof call.error === 'string'}
											<!-- svelte-ignore a11y_no_noninteractive_tabindex -->
											<!-- svelte-ignore a11y_click_events_have_key_events -->
											<div class="call-row" class:call-row-open={isCallOpen} class:call-row-error={hasError}
												role="button"
												tabindex="0"
												onclick={(ev) => { ev.stopPropagation(); expandedCall = isCallOpen ? null : callKey; }}>
												<span class="call-chev" aria-hidden="true">{isCallOpen ? '▾' : '▸'}</span>
												<span class="call-kind">{(call.kind as string | undefined) ?? '—'}</span>
												<span class="call-model muted">{(call.model_id as string | undefined) ?? ''}</span>
												<span class="call-duration">{((call.duration_s as number | undefined) ?? 0).toFixed(2)}s</span>
												<span class="call-tokens">{(call.prompt_tokens as number | null | undefined) ?? '—'}→{(call.out_tokens as number | null | undefined) ?? '—'}</span>
												<span class="call-finish muted">{(call.finish_reason as string | null | undefined) ?? '—'}</span>
												<span class="call-flags muted">
													{#if hasReasoning}reasoning {(call.reasoning as string).length}c · {/if}{#if hasContent}content {(call.content as string).length}c{:else}<span class="exp-not-captured-inline">no content captured</span>{/if}
												</span>
											</div>
											{#if isCallOpen}
												<div class="call-detail">
													{#if hasError}
														<div class="call-detail-block">
															<h5 class="call-detail-h">error</h5>
															<pre class="call-detail-pre call-detail-error">{call.error}</pre>
														</div>
													{/if}
													{#if (call.system as string | undefined)}
														<div class="call-detail-block">
															<h5 class="call-detail-h">system prompt</h5>
															<pre class="call-detail-pre">{call.system}</pre>
														</div>
													{/if}
													{#if hasMessages}
														<div class="call-detail-block">
															<h5 class="call-detail-h">messages ({(call.messages as unknown[]).length})</h5>
															{#each (call.messages as Array<{role: string; content: string}>) as msg}
																<div class="call-msg">
																	<span class="call-msg-role">{msg.role}</span>
																	<pre class="call-detail-pre call-detail-msg">{msg.content}</pre>
																</div>
															{/each}
														</div>
													{:else if !hasError}
														<div class="call-detail-block">
															<h5 class="call-detail-h">prompt</h5>
															<p class="exp-not-captured">not captured — pre-Layer-B runs don't persist input messages. Re-score to capture.</p>
														</div>
													{/if}
													{#if hasReasoning}
														<div class="call-detail-block">
															<h5 class="call-detail-h">reasoning <span class="muted">({(call.reasoning as string).length} chars)</span></h5>
															<pre class="call-detail-pre call-detail-reasoning">{call.reasoning}</pre>
														</div>
													{/if}
													{#if hasContent}
														<div class="call-detail-block">
															<h5 class="call-detail-h">response</h5>
															<pre class="call-detail-pre call-detail-content">{call.content}</pre>
														</div>
													{:else if !hasError}
														<div class="call-detail-block">
															<h5 class="call-detail-h">response</h5>
															<p class="exp-not-captured">not captured — pre-Layer-B runs don't persist response text. Re-score to capture.</p>
														</div>
													{/if}
												</div>
											{/if}
										{/each}
									</div>
								{/if}
							</div>

							<!-- Per-step records: one row per scorer_step DB row for this evidence -->
							{#if d.selected_architecture === 'monolithic'}
								{@const selectedExamples = selectedExamplesForOutput(out)}
								{@const selectedExampleIds = selectedExampleIdsForOutput(out)}
								<div class="exp-block">
									<h4 class="exp-h">monolithic native fields
										<span class="exp-h-note">aggregate-row fields specific to the monolithic grammar</span>
									</h4>
									<dl class="mono-fields">
										<div><dt>tier</dt><dd>{out.tier ?? 'not captured'}</dd></div>
										<div><dt>grounding</dt><dd>{out.grounding_status ?? 'not captured'}</dd></div>
										<div><dt>provenance</dt><dd>{out.provenance_triggered === true ? 'triggered' : out.provenance_triggered === false ? 'not triggered' : 'not captured'}</dd></div>
										<div><dt>parse</dt><dd>{out.verdict == null ? 'verdict not parsed' : `${out.verdict} / ${out.confidence ?? '—'}`}</dd></div>
									</dl>
									{#if selectedExamples.length > 0}
										<div class="selected-examples" aria-label="selected monolithic contrastive examples">
											<h5>selected examples <span class="muted">({selectedExamples.length})</span></h5>
											<ol>
												{#each selectedExamples as ex}
													<li>
														<code>{String(ex.id ?? 'example')}</code>
														<span>{String(ex.claim ?? 'claim not captured')}</span>
														<small>{String(ex.verdict ?? 'verdict ?')}{#if ex.confidence} · {String(ex.confidence)}{/if}</small>
													</li>
												{/each}
											</ol>
										</div>
									{:else if selectedExampleIds.length > 0}
										<p class="exp-not-captured">selected example IDs captured: {selectedExampleIds.join(', ')}. Re-score with current instrumentation to capture compact example metadata.</p>
									{:else}
										<p class="exp-not-captured">selected example IDs are not persisted yet, so exemplar attribution is unavailable for this row.</p>
									{/if}
								</div>
							{:else if evSteps.length > 0}
								<div class="exp-block">
									<h4 class="exp-h">pipeline steps <span class="exp-h-count">({evSteps.length})</span>
										<span class="exp-h-note">one row per <code>scorer_step</code> persisted for this evidence (parse, probes, grounding, ...)</span>
									</h4>
									<table class="trace-table">
										<thead>
											<tr>
												<th>step</th>
												<th>source</th>
												<th>output</th>
												<th class="num">in→out tok</th>
												<th class="num">latency</th>
											</tr>
										</thead>
										<tbody>
											{#each evSteps as step}
												{@const stepOut = safeParseJSON(step.output_json)}
												{@const hasInputPayload = step.input_payload_json != null && step.input_payload_json !== 'null'}
												<tr>
													<td>{step.step_kind}</td>
													<td>
														{#if step.is_substrate_answered === true}<span class="step-source-tag step-source-substrate">substrate</span>
														{:else if step.is_substrate_answered === false}<span class="step-source-tag step-source-llm">LLM</span>
														{:else}<span class="muted">—</span>{/if}
													</td>
													<td class="step-output">
														{#if !stepOut}
															<span class="muted">—</span>
														{:else}
															{#if stepOut.answer != null}<span class="step-answer">{stepOut.answer}</span>{/if}
															{#if stepOut.confidence != null}<span class="muted">· {stepOut.confidence}</span>{/if}
															{#if stepOut.rationale}<span class="step-rationale">— {stepOut.rationale}</span>{/if}
															{#if stepOut.stmt_type != null}<span>{stepOut.stmt_type}</span>{/if}
															{#if stepOut.n_aliases != null}<span class="muted">{stepOut.n_aliases} aliases, {stepOut.n_detected_relations ?? 0} relations</span>{/if}
															{#if stepOut.span}<span class="step-span">“{stepOut.span}”</span>{/if}
														{/if}
													</td>
													<td class="num">
														{#if step.prompt_tokens != null || step.out_tokens != null}{step.prompt_tokens ?? '—'}→{step.out_tokens ?? '—'}
														{:else}<span class="muted">—</span>{/if}
													</td>
													<td class="num">
														{#if step.latency_ms != null && step.latency_ms > 0}{step.latency_ms}ms
														{:else}<span class="muted">—</span>{/if}
													</td>
												</tr>
												{#if showDebug}
													<tr class="exp-debug-row">
														<td colspan="5" class="exp-debug-cell">
															<span class="exp-debug-label">step_hash</span>
															<code>{step.step_hash}</code>
															<span class="exp-debug-label">input_payload</span>
															<span>{hasInputPayload ? 'present' : 'not captured'}</span>
															<span class="exp-debug-label">model</span>
															<span>{step.model_id ?? '—'}</span>
															{#if step.error}<span class="exp-debug-label">error</span><span class="exp-error-inline">{step.error}</span>{/if}
														</td>
													</tr>
												{/if}
											{/each}
										</tbody>
									</table>
								</div>
							{/if}

							{#if out.error}
								<div class="exp-block exp-error-block">
									<h4 class="exp-h">aggregate error</h4>
									<p class="exp-error">{out.error}</p>
								</div>
							{/if}

							<!-- Developer detail: hashes, model id, raw payload pointers -->
							<div class="exp-block">
								<label class="exp-debug-toggle">
									<input type="checkbox" bind:checked={showDebug} onclick={(ev) => ev.stopPropagation()}/>
									<span>show developer detail (step_hash, model_id, payload state)</span>
								</label>
								{#if showDebug}
									<div class="exp-debug-block">
										<span class="exp-debug-label">aggregate step_hash</span>
										<button type="button" class="hash-copy"
											onclick={(ev) => {
												ev.stopPropagation();
												navigator.clipboard?.writeText(score?.step_hash ?? '').then(() => {
													copiedHash = score?.step_hash ?? null;
													setTimeout(() => { copiedHash = null; }, 1200);
												});
											}}
											title="Copy step_hash to clipboard">
											<code>{score?.step_hash}</code>
											<span class="copy-mark">{copiedHash === score?.step_hash ? '✓ copied' : '⎘'}</span>
										</button>
									</div>
								{/if}
							</div>
						</div>
					{/if}
				</article>
			{/each}
		</div>

		<!-- Right column: truth panel + agents + supports -->
		<aside class="truth-panel">
			<h2>what other systems say about this</h2>
			<ul class="truth-list">
				{#each presentSets as tsetId}
					{@const labels = d.truth_labels.filter((l) => l.truth_set_id === tsetId)}
					{@const isActive = hoveredEvidenceHash != null && labels.some((l) =>
						(l.target_kind === 'evidence' && l.target_id === hoveredEvidenceHash) ||
						l.target_kind === 'stmt' ||
						l.target_kind === 'agent'
					)}
					{@const isPassive = hoveredEvidenceHash != null && !isActive}
					<li class="truth-row" class:truth-active={isActive} class:truth-passive={isPassive}>
						<div class="truth-row-head">
							<span class="truth-name"><code>{tsetId}</code></span>
							<span class="truth-count">{labels.length} label{labels.length === 1 ? '' : 's'}</span>
						</div>
						<p class="truth-desc">{truthSetDescription(tsetId)}</p>
					</li>
				{/each}
				{#if absentSets.length > 0}
					<li class="truth-row truth-row-absent" title={absentSets.join(', ')}>
						<div class="truth-row-head">
							<span class="truth-name muted">no labels yet from</span>
							<span class="truth-count muted">{absentSets.length} set{absentSets.length === 1 ? '' : 's'}</span>
						</div>
						<p class="truth-desc muted">{absentSets.map(shortTruthLabel).join(' · ')}</p>
					</li>
				{/if}
			</ul>

			<h2>agents <span class="counter">{d.agents.length}</span></h2>
			{#each d.agents as a}
				{@const agTruth = truthByTarget(d.truth_labels, 'agent', a.agent_hash)}
				<div class="agent">
					<div class="agent-line">
						<span class="agent-role">{a.role}</span>
						<span class="agent-name">{a.name}</span>
					</div>
					<div class="agent-refs">
						{#each parseDbRefs(a.db_refs_json) as [ns, id]}
							<span class="ref-chip"><span class="ref-ns">{ns}</span>:{id}</span>
						{/each}
					</div>
					{#if a.location}
						<div class="agent-loc">loc: {a.location}</div>
					{/if}
					{#if agTruth.length > 0}
						<div class="agent-truth">{agTruth.length} truth label{pluralS(agTruth.length)}</div>
					{/if}
				</div>
			{/each}

			{#if d.supports_edges.length > 0}
				<h2>supports edges <span class="counter">{d.supports_edges.length}</span></h2>
				<ul class="edges">
					{#each d.supports_edges as edge}
						<li>
							<span class="edge-kind">{edge.kind}</span>
							<code>{shortHash(edge.to_stmt_hash)}</code>
						</li>
					{/each}
				</ul>
			{/if}
		</aside>
	</div>
</main>

<style>
	:global(:root) {
		--ink: #1a1a1a;
		--ink-muted: #6a6a6a;
		--ink-faint: #727272;
		--paper: #fdfcf8;
		--rule: #e6e2d6;
			--accent: #7d2a1a;
			--mono: ui-monospace, 'SF Mono', 'JetBrains Mono', Menlo, monospace;
			--serif: 'Iowan Old Style', 'Source Serif Pro', Georgia, serif;
			--sticky-target-offset: 5.5rem;
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
		position: sticky;
		top: 0;
		background: var(--paper);
		z-index: 2;
	}
	.crumb a { color: var(--ink-muted); text-decoration: none; }
	.crumb a:hover { color: var(--ink); }
	.crumb strong { color: var(--ink); font-weight: 500; }
	.crumb .sep, .muted, .dot { color: var(--ink-faint); }
	.dot { margin: 0 0.4rem; }
	.meta {
		display: flex;
		gap: 0.6rem;
		align-items: baseline;
		flex-wrap: wrap;
		justify-content: flex-end;
	}
	.meta a {
		color: var(--accent);
		text-decoration: none;
	}
		.meta a:hover {
			text-decoration: underline;
		}
		.compare-mode-label {
			color: var(--ink);
			font-weight: 600;
			border-bottom: 1px solid var(--ink);
			padding-bottom: 0.15rem;
			white-space: nowrap;
		}
		.paired-run-switcher {
			display: flex;
			flex-wrap: wrap;
			gap: 0.35rem;
			justify-content: flex-end;
			align-items: stretch;
		}
		.meta .paired-run-token {
			display: grid;
			grid-template-columns: 1fr;
			gap: 0.05rem;
			min-width: 8.5rem;
			border: 1px solid var(--rule);
			border-left: 3px solid var(--ink);
			color: var(--ink-muted);
			padding: 0.2rem 0.45rem;
			text-decoration: none;
			background: rgba(255, 255, 255, 0.34);
		}
		.meta .paired-run-token-monolithic {
			border-left-color: var(--ink);
		}
		.meta .paired-run-token-decomposed {
			border-left-color: var(--accent);
		}
		.meta .paired-run-token-active {
			color: var(--ink);
			border-color: var(--ink);
			background: rgba(26, 26, 26, 0.055);
			box-shadow: inset 0 -3px 0 var(--ink);
		}
		.meta a.paired-run-token:hover {
			background: rgba(26, 26, 26, 0.035);
			border-color: var(--ink-muted);
			text-decoration: none;
		}
		.meta a.paired-run-token:focus-visible,
		.meta a.single-run-exit:focus-visible {
			outline: 2px solid var(--accent);
			outline-offset: 2px;
			text-decoration: none;
		}
		.paired-run-token strong,
		.paired-run-token span {
			display: block;
			line-height: 1.15;
		}
		.paired-run-token strong {
			font-size: 0.82rem;
			font-weight: 600;
		}
		.paired-run-token span {
			color: var(--ink-faint);
			font-size: 0.7rem;
		}
		.meta a.single-run-exit {
			border: 1px solid var(--rule);
			color: var(--accent);
			font-size: 0.76rem;
			padding: 0.25rem 0.45rem;
			text-decoration: none;
			white-space: nowrap;
		}
		.meta a.single-run-exit:hover {
			border-color: var(--accent);
			text-decoration: none;
		}
		.meta select {
			font-family: var(--mono);
			font-size: 0.74rem;
		padding: 0.15rem 0.3rem;
		background: var(--paper);
		color: var(--ink);
		border: 1px solid var(--rule);
		max-width: 22rem;
	}

	main {
		max-width: 1200px;
		margin: 0 auto;
		padding: 1.5rem 1.5rem 4rem;
	}

	.run-warning {
		max-width: 920px;
		border-left: 3px solid #8a6a00;
		background: rgba(138, 106, 0, 0.04);
		padding: 0.55rem 0 0.55rem 0.85rem;
		margin: 0 0 1rem;
	}
	.run-warning span {
		display: block;
		font-family: var(--mono);
		font-size: 0.72rem;
		color: #8a6a00;
		text-transform: lowercase;
		letter-spacing: 0.02em;
	}
	.run-warning p {
		margin: 0.2rem 0 0;
		color: var(--ink-muted);
		font-size: 0.9rem;
		line-height: 1.45;
	}

	.stmt-header {
		padding-bottom: 1.2rem;
		border-bottom: 1px solid var(--ink);
		margin-bottom: 1.5rem;
	}

		.paired-trace-compare {
			border-bottom: 1px solid var(--ink);
			margin: -0.4rem 0 1.5rem;
			padding: 0 0 1.25rem;
			scroll-margin-top: var(--sticky-target-offset);
		}

	.paired-trace-head {
		display: flex;
		align-items: flex-start;
		justify-content: space-between;
		gap: 1rem;
		margin-bottom: 0.75rem;
	}

	.paired-trace-head h2 {
		margin: 0;
		font-family: var(--mono);
		font-size: 0.86rem;
		font-weight: 600;
		letter-spacing: 0;
		text-transform: lowercase;
	}

	.paired-trace-head p,
	.compare-warning {
		margin: 0.25rem 0 0;
		max-width: 58rem;
		color: var(--ink-muted);
		font-size: 0.88rem;
		line-height: 1.45;
	}

		.paired-trace-head a,
		.compare-run-card a,
		.compare-native-grid a {
			color: var(--accent);
			font-family: var(--mono);
			font-size: 0.72rem;
			text-decoration: none;
			white-space: nowrap;
		}

		.compare-native-grid a.compare-missing-link {
			color: var(--ink-muted);
			white-space: normal;
		}

	.paired-trace-head a:hover,
	.compare-run-card a:hover,
	.compare-native-grid a:hover {
		text-decoration: underline;
	}

	.compare-warning {
		border-left: 2px solid var(--accent);
		padding-left: 0.6rem;
	}

	.compare-run-grid,
	.compare-native-grid,
	.compare-native-detail {
		display: grid;
		grid-template-columns: repeat(2, minmax(0, 1fr));
		gap: 0.75rem;
	}

	.compare-run-grid {
		margin-bottom: 0.75rem;
	}

	.compare-run-card,
	.compare-native-grid section,
	.compare-native-detail section {
		border: 1px solid var(--rule);
		border-left-width: 3px;
		padding: 0.55rem 0.65rem;
	}

	.compare-run-monolithic,
	.compare-side-monolithic {
		border-left-color: var(--ink);
	}

	.compare-run-decomposed,
	.compare-side-decomposed {
		border-left-color: var(--accent);
	}

	.compare-run-card h3,
	.compare-native-grid h4,
	.compare-native-detail h4 {
		margin: 0 0 0.25rem;
		font-family: var(--mono);
		font-size: 0.75rem;
		font-weight: 600;
	}

	.compare-run-card p,
	.compare-run-card strong,
	.compare-outcome,
	.compare-native-line {
		display: block;
		margin: 0.18rem 0;
	}

	.compare-run-card p,
	.compare-native-line,
	.compare-side-missing {
		color: var(--ink-muted);
	}

	.compare-run-card strong,
	.compare-outcome {
		font-weight: 500;
	}

	.compare-evidence-list {
		display: grid;
		gap: 0.55rem;
	}

		.compare-evidence {
			border-top: 1px solid var(--rule);
			padding-top: 0.55rem;
			scroll-margin-top: var(--sticky-target-offset);
		}

		.compare-evidence:target {
			border-top-color: var(--accent);
			box-shadow: inset 4px 0 0 var(--accent), 0 0 0 2px rgba(125, 42, 26, 0.08);
			background: rgba(125, 42, 26, 0.045);
		}

	.compare-score-only {
		border-left: 2px solid var(--accent);
		padding-left: 0.65rem;
	}

	.compare-evidence-head {
		display: flex;
		align-items: center;
		justify-content: space-between;
		gap: 0.75rem;
		font-family: var(--mono);
		font-size: 0.72rem;
	}

	.compare-evidence-head > div {
		display: flex;
		flex-wrap: wrap;
		gap: 0.4rem;
		min-width: 0;
		color: var(--ink-muted);
	}

	.compare-evidence-head code {
		color: var(--ink);
	}

	.compare-evidence-head button {
		border: 1px solid var(--rule);
		background: transparent;
		color: var(--ink);
		cursor: pointer;
		font-family: var(--mono);
		font-size: 0.72rem;
		padding: 0.16rem 0.4rem;
		white-space: nowrap;
	}

	.compare-evidence-head button:hover,
	.compare-evidence-head button:focus-visible {
		background: var(--ink);
		color: var(--paper);
		outline: none;
	}

	.compare-evidence-text {
		margin: 0.28rem 0 0.5rem;
	}

	.compare-native-detail {
		margin-top: 0.6rem;
	}

	.compare-aggregate-rows {
		border-bottom: 1px solid var(--rule);
		margin: 0 0 0.45rem;
		padding-bottom: 0.35rem;
	}

	.compare-aggregate-rows h5 {
		margin: 0 0 0.25rem;
		color: var(--ink-muted);
		font-family: var(--mono);
		font-size: 0.66rem;
		font-weight: 500;
		text-transform: lowercase;
	}

	.compare-aggregate-rows p {
		margin: 0.12rem 0;
		font-size: 0.78rem;
	}

	.compare-aggregate-rows span {
		display: inline-block;
		min-width: 5.9rem;
		color: var(--ink-muted);
		font-family: var(--mono);
		font-size: 0.68rem;
	}

	.compare-step-table {
		width: 100%;
		border-collapse: collapse;
		font-size: 0.8rem;
	}

	.compare-step-table th,
	.compare-step-table td {
		border-top: 1px solid var(--rule);
		padding: 0.22rem 0.28rem;
		text-align: left;
		vertical-align: top;
	}

	.compare-step-table th {
		color: var(--ink-muted);
		font-family: var(--mono);
		font-size: 0.68rem;
		font-weight: 500;
	}

	.stmt-glyph {
		font-family: var(--serif);
		font-size: 1.5rem;
		font-weight: 400;
		color: var(--ink);
		margin: 0 0 0.4rem;
		line-height: 1.3;
	}

	.indra-type {
		color: var(--accent);
	}

	.paren {
		color: var(--ink-faint);
	}

	.stmt-meta {
		font-family: var(--mono);
		font-size: 0.78rem;
		color: var(--ink-muted);
		display: flex;
		flex-wrap: wrap;
		align-items: baseline;
	}

	.belief-line {
		font-variant-numeric: tabular-nums;
	}

	.belief-num {
		color: var(--ink);
		font-weight: 500;
		font-variant-numeric: tabular-nums;
	}

	.belief-indra {
		color: var(--ink-muted);
		font-variant-numeric: tabular-nums;
	}

	.delta {
		font-family: var(--mono);
		font-variant-numeric: tabular-nums;
	}

	.delta.pos { color: #2a6f2a; }
	.delta.neg { color: var(--accent); }

	.verdict-tally {
		font-family: var(--mono);
		font-variant-numeric: tabular-nums;
	}

	.vt-correct { color: #2a6f2a; }
	.vt-incorrect { color: var(--accent); }
	.vt-abstain { color: var(--ink-muted); font-style: italic; }


	.hint {
		color: var(--ink-faint);
		font-style: italic;
		font-size: 0.92em;
	}

	.cols {
		display: grid;
		grid-template-columns: 1fr 320px;
		gap: 2rem;
	}

		@media (max-width: 880px) {
			:global(:root) {
				--sticky-target-offset: 10.5rem;
			}
			header {
				align-items: flex-start;
				flex-wrap: wrap;
				gap: 0.45rem;
			}
			.meta {
				justify-content: flex-start;
				width: 100%;
			}
			.paired-run-switcher {
				justify-content: flex-start;
				width: 100%;
			}
			.meta .paired-run-token {
				flex: 1 1 8.5rem;
				min-width: min(8.5rem, 100%);
			}
			.cols,
			.compare-run-grid,
			.compare-native-grid,
		.compare-native-detail {
			grid-template-columns: 1fr;
		}
		.paired-trace-head {
			display: block;
		}
		.paired-trace-head a {
			display: inline-block;
			margin-top: 0.4rem;
		}
		.compare-evidence-head {
			align-items: flex-start;
			flex-direction: column;
		}
		.compare-evidence-head button {
			min-height: 2.4rem;
		}
	}

	h2 {
		font-family: var(--mono);
		font-size: 0.72rem;
		color: var(--ink-muted);
		text-transform: lowercase;
		letter-spacing: 0.02em;
		margin: 1.5rem 0 0.6rem;
		font-weight: 500;
		border-bottom: 1px solid var(--rule);
		padding-bottom: 0.2rem;
	}

	h2:first-child {
		margin-top: 0;
	}

	.counter {
		color: var(--ink-faint);
		font-weight: 400;
	}


		.evidence {
			padding: 0.7rem 0;
			border-bottom: 1px dotted var(--rule);
			scroll-margin-top: var(--sticky-target-offset);
		}

		.evidence:target {
			background: rgba(125, 42, 26, 0.055);
			box-shadow: inset 4px 0 0 var(--accent), 0 0 0 2px rgba(125, 42, 26, 0.08);
		}

	.evidence:last-child {
		border-bottom: none;
	}

	/* Trace narrative — sentence-flow replacement of the old 9-dot rail */
	.trace-narrative {
		list-style: none;
		counter-reset: trace;
		padding: 0;
		margin: 0 0 1.6rem;
	}
	.trace-line {
		display: grid;
		grid-template-columns: 1.6rem 1fr;
		gap: 0.5rem;
		align-items: baseline;
		padding: 0.3rem 0;
		font-family: var(--serif);
		font-size: 0.98rem;
		line-height: 1.45;
	}
	.trace-num {
		font-family: var(--mono);
		font-size: 0.74rem;
		color: var(--ink-faint);
		text-align: right;
	}
	.trace-prose {
		color: var(--ink);
	}
	.trace-line.trace-muted .trace-prose {
		color: var(--ink-faint);
		font-style: italic;
	}

	/* Verdict line — top of each evidence card */
	.ev-verdict-line {
		display: flex;
		gap: 0.8rem;
		align-items: baseline;
		flex-wrap: wrap;
		font-family: var(--mono);
		font-size: 0.92rem;
		margin: 0.4rem 0 0.2rem;
	}
	.ev-verdict {
		font-weight: 500;
	}
	.ev-verdict-line-missing {
		color: var(--ink-muted);
		font-size: 0.78rem;
	}
	.ev-not-scored {
		border-left: 2px solid var(--rule);
		padding-left: 0.5rem;
	}
	.ev-verdict-correct { color: var(--ok-green); }
	.ev-verdict-incorrect { color: var(--accent); }
	.ev-verdict-abstain { color: var(--ink-muted); font-style: italic; }
	.ev-confidence { color: var(--ink); }
	.ev-score { color: var(--ink); }
	.ev-score-num {
		font-variant-numeric: tabular-nums;
		font-weight: 500;
	}
	.ev-chevron {
		margin-left: auto;
		color: var(--ink-faint);
		font-family: var(--mono);
		font-size: 0.72rem;
	}
	.ev-clickable:hover .ev-chevron {
		color: var(--accent);
	}
	.evidence.ev-expanded-state .ev-chevron {
		color: var(--accent);
	}

	.ev-meta-secondary {
		font-family: var(--mono);
		font-size: 0.7rem;
		color: var(--ink-faint);
		display: flex;
		gap: 0.8rem;
		margin-bottom: 0.5rem;
	}
	.ev-source { color: var(--accent); }
	.ev-pmid { color: var(--ink-muted); }
	.ev-hash { color: var(--ink-faint); }
	.latency { color: var(--ink-faint); }

	.ev-text {
		font-family: var(--serif);
		font-size: 1.04rem;
		line-height: 1.5;
		margin: 0.2rem 0 0.4rem;
		color: var(--ink);
	}
	.ev-text-cue {
		background: var(--accent-wash);
		color: var(--accent);
		padding: 0 0.15em;
		font-style: italic;
		font-weight: 500;
		border-bottom: 1px solid var(--accent);
	}

	.ev-flags {
		font-family: var(--serif);
		font-style: italic;
		font-size: 0.86rem;
		color: var(--ink-muted);
		margin: 0.2rem 0;
	}
	.ev-flags-prose { color: var(--ink-muted); }

	.ev-section-note {
		font-family: var(--serif);
		font-style: italic;
		font-size: 0.84rem;
		color: var(--ink-muted);
		margin: 0 0 1rem;
		line-height: 1.5;
		max-width: 65ch;
	}
	.ev-bucket-table {
		display: block;
		font-family: var(--mono);
		font-style: normal;
		font-size: 0.74rem;
		color: var(--ink-faint);
		margin-top: 0.3rem;
		line-height: 1.6;
	}
	.ev-bucket-table code {
		color: var(--ink);
		font-variant-numeric: tabular-nums;
	}
	.truth-stamp {
		color: var(--accent);
		font-family: var(--mono);
		font-style: normal;
		font-size: 0.74rem;
		margin-left: 0.4rem;
	}

	.reasons {
		font-family: var(--mono);
		font-size: 0.74rem;
		margin-top: 0.35rem;
		color: var(--ink);
	}

	.reasons-label {
		color: var(--ink-muted);
		text-transform: lowercase;
		letter-spacing: 0.04em;
		margin-right: 0.4rem;
	}

	.reason-token {
		margin-right: 0.4rem;
	}

	/* Truth panel */
	.truth-panel {
		font-family: var(--mono);
		font-size: 0.82rem;
	}

	.truth-list {
		margin: 0;
	}

	.truth-row {
		display: flex;
		justify-content: space-between;
		padding: 0.18rem 0;
		border-bottom: 1px dotted var(--rule);
	}

	.truth-row {
		display: block;
		padding: 0.5rem 0;
		border-bottom: 1px dotted var(--rule);
	}
	.truth-row:last-child {
		border-bottom: none;
	}
	.truth-row-head {
		display: flex;
		justify-content: space-between;
		align-items: baseline;
		gap: 0.6rem;
		font-family: var(--mono);
		font-size: 0.82rem;
	}
	.truth-name {
		color: var(--ink);
	}
	.truth-count {
		color: var(--ink);
		font-variant-numeric: tabular-nums;
		font-size: 0.76rem;
	}
	.truth-desc {
		font-family: var(--serif);
		font-style: italic;
		font-size: 0.86rem;
		color: var(--ink-muted);
		margin: 0.2rem 0 0;
		line-height: 1.4;
	}
	.truth-row-absent .truth-name,
	.truth-row-absent .truth-count {
		color: var(--ink-faint);
	}
	.truth-row.truth-active .truth-name {
		color: var(--accent);
		font-weight: 500;
	}
	.truth-row.truth-active .truth-count {
		color: var(--accent);
		font-weight: 500;
	}
	.truth-row.truth-passive {
		opacity: 0.45;
	}

	.evidence:hover {
		background: rgba(125, 42, 26, 0.025);
	}

	.ev-clickable {
		cursor: pointer;
	}

	.ev-clickable:hover .ev-chevron {
		color: var(--accent);
	}

	.ev-clickable:focus-visible {
		outline: 2px solid var(--accent);
		outline-offset: 2px;
	}

	/* Inner expansion content shouldn't trigger collapse on click */
	.ev-expanded {
		pointer-events: auto;  /* keeps copy button functional */
	}

	.ev-chevron {
		margin-left: auto;
		font-size: 0.7rem;
		color: var(--ink-faint);
		font-family: var(--mono);
	}

	.evidence.ev-expanded-state .ev-chevron {
		color: var(--accent);
	}

	.ev-expanded {
		margin-top: 0.6rem;
		padding: 0.6rem 0.8rem;
		background: rgba(0, 0, 0, 0.02);
		border-left: 2px solid var(--rule);
		font-family: var(--mono);
		font-size: 0.78rem;
	}

	/* Each expanded section: rationale, LLM calls, pipeline steps, debug */
	.exp-block {
		margin-bottom: 0.9rem;
	}
	.exp-block:last-child {
		margin-bottom: 0;
	}
	.exp-h {
		font-family: var(--mono);
		font-size: 0.74rem;
		color: var(--ink-muted);
		text-transform: lowercase;
		letter-spacing: 0.04em;
		margin: 0 0 0.4rem;
		font-weight: 500;
	}
	.exp-h-count {
		color: var(--ink-faint);
		font-weight: 400;
		margin-left: 0.2rem;
	}
	.exp-h-note {
		display: block;
		font-family: var(--serif);
		font-style: italic;
		font-size: 0.78rem;
		color: var(--ink-faint);
		text-transform: none;
		letter-spacing: 0;
		margin-top: 0.2rem;
	}
	.exp-rationale {
		font-family: var(--serif);
		font-size: 0.92rem;
		color: var(--ink);
		margin: 0;
		line-height: 1.45;
	}
	.exp-empty {
		font-family: var(--serif);
		font-style: italic;
		font-size: 0.86rem;
		color: var(--ink-muted);
		margin: 0;
	}
	.exp-not-captured {
		font-family: var(--serif);
		font-style: italic;
		font-size: 0.78rem;
		color: var(--ink-faint);
		margin: 0.4rem 0 0;
	}
	.exp-error,
	.exp-error-block .exp-error {
		color: var(--accent);
		font-family: var(--serif);
		font-size: 0.9rem;
		margin: 0;
	}
	.exp-error-inline {
		color: var(--accent);
	}
	.mono-fields {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
		gap: 0.35rem 1rem;
		margin: 0;
	}
	.mono-fields div {
		min-width: 0;
	}
	.mono-fields dt {
		font-size: 0.68rem;
		color: var(--ink-muted);
		text-transform: lowercase;
		letter-spacing: 0.02em;
		margin: 0;
	}
	.mono-fields dd {
		margin: 0.1rem 0 0;
		color: var(--ink);
		overflow-wrap: anywhere;
	}
	.selected-examples {
		margin-top: 0.7rem;
		border-top: 1px dotted var(--rule);
		padding-top: 0.55rem;
	}
	.selected-examples h5 {
		font-family: var(--mono);
		font-size: 0.72rem;
		font-weight: 600;
		color: var(--ink-muted);
		margin: 0 0 0.35rem;
		text-transform: lowercase;
		letter-spacing: 0.02em;
	}
	.selected-examples ol {
		list-style: decimal;
		margin: 0;
		padding-left: 1.2rem;
		display: grid;
		gap: 0.35rem;
	}
	.selected-examples li {
		font-size: 0.78rem;
		line-height: 1.35;
		color: var(--ink);
	}
	.selected-examples code {
		font-family: var(--mono);
		font-size: 0.72rem;
		margin-right: 0.35rem;
		color: var(--ink-muted);
	}
	.selected-examples small {
		display: block;
		font-family: var(--mono);
		font-size: 0.68rem;
		color: var(--ink-faint);
		margin-top: 0.1rem;
	}

	/* Tables inside the expansion (LLM calls + pipeline steps) */
	.trace-table {
		width: 100%;
		border-collapse: collapse;
		font-family: var(--mono);
		font-size: 0.78rem;
	}
	.trace-table th {
		text-align: left;
		font-weight: 500;
		color: var(--ink-muted);
		font-size: 0.7rem;
		padding: 0.2rem 0.6rem 0.2rem 0;
		border-bottom: 1px dotted var(--rule);
	}
	.trace-table td {
		padding: 0.25rem 0.6rem 0.25rem 0;
		vertical-align: baseline;
		border-bottom: 1px dotted var(--rule);
	}
	.trace-table .num {
		text-align: right;
		font-variant-numeric: tabular-nums;
	}

	.step-output {
		font-family: var(--mono);
		font-size: 0.78rem;
		display: flex;
		flex-wrap: wrap;
		gap: 0.35rem;
		align-items: baseline;
	}
	.step-answer { color: var(--ink); font-weight: 500; }
	.step-rationale { color: var(--ink-muted); font-family: var(--serif); font-style: italic; font-size: 0.86rem; flex-basis: 100%; }
	.step-span { color: var(--ink-muted); font-style: italic; }
	.step-source-tag {
		font-family: var(--mono);
		font-size: 0.66rem;
		text-transform: uppercase;
		letter-spacing: 0.04em;
		padding: 0 0.3rem;
	}
	.step-source-substrate { color: var(--accent); }
	.step-source-llm { color: var(--ok-green); }

	/* LLM call expandable rows */
	.call-list {
		display: flex;
		flex-direction: column;
		gap: 0;
		font-family: var(--mono);
		font-size: 0.78rem;
	}
	.call-row {
		display: grid;
		grid-template-columns: 1.5rem minmax(0, 1fr) minmax(0, 1.2fr) 4ch 7ch 6ch minmax(0, 1.4fr);
		gap: 0.6rem;
		align-items: baseline;
		padding: 0.3rem 0.4rem;
		border-bottom: 1px dotted var(--rule);
		cursor: pointer;
		font-variant-numeric: tabular-nums;
	}
	.call-row:hover {
		background: var(--accent-wash);
	}
	.call-row-open {
		background: var(--accent-wash);
	}
	.call-row-error {
		color: var(--accent);
	}
	.call-chev {
		color: var(--ink-faint);
	}
	.call-kind {
		color: var(--ink);
		font-weight: 500;
		white-space: nowrap;
		overflow: hidden;
		text-overflow: ellipsis;
	}
	.call-model {
		font-size: 0.7rem;
		white-space: nowrap;
		overflow: hidden;
		text-overflow: ellipsis;
	}
	.call-duration, .call-tokens, .call-finish {
		color: var(--ink);
		font-size: 0.74rem;
	}
	.call-flags {
		font-size: 0.7rem;
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}
	.exp-not-captured-inline {
		font-style: italic;
		color: var(--ink-faint);
	}

	.call-detail {
		padding: 0.6rem 0.8rem;
		margin-bottom: 0.4rem;
		border-left: 2px solid var(--accent);
		background: rgba(0, 0, 0, 0.015);
	}
	.call-detail-block {
		margin-bottom: 0.8rem;
	}
	.call-detail-block:last-child {
		margin-bottom: 0;
	}
	.call-detail-h {
		font-family: var(--mono);
		font-size: 0.7rem;
		color: var(--ink-muted);
		text-transform: lowercase;
		letter-spacing: 0.04em;
		margin: 0 0 0.3rem;
		font-weight: 500;
	}
	.call-detail-pre {
		font-family: var(--mono);
		font-size: 0.78rem;
		background: var(--paper);
		border: 1px solid var(--rule);
		padding: 0.5rem 0.7rem;
		margin: 0;
		white-space: pre-wrap;
		word-break: break-word;
		max-height: 320px;
		overflow-y: auto;
		line-height: 1.45;
		color: var(--ink);
	}
	.call-detail-error {
		color: var(--accent);
	}
	.call-detail-reasoning {
		font-style: italic;
		color: var(--ink-muted);
	}
	.call-detail-content {
		color: var(--ink);
	}
	.call-msg {
		margin-bottom: 0.4rem;
	}
	.call-msg-role {
		display: inline-block;
		font-family: var(--mono);
		font-size: 0.66rem;
		text-transform: uppercase;
		letter-spacing: 0.04em;
		color: var(--ink-muted);
		margin-bottom: 0.15rem;
	}
	.call-detail-msg {
		margin-top: 0.1rem;
	}

	/* Developer detail block — collapsed by default behind a checkbox */
	.exp-debug-toggle {
		display: inline-flex;
		gap: 0.4rem;
		font-family: var(--mono);
		font-size: 0.7rem;
		color: var(--ink-muted);
		cursor: pointer;
	}
	.exp-debug-block,
	.exp-debug-row {
		font-family: var(--mono);
		font-size: 0.7rem;
		color: var(--ink-faint);
	}
	.exp-debug-block {
		padding: 0.4rem 0 0;
		display: flex;
		gap: 0.6rem;
		flex-wrap: wrap;
		align-items: baseline;
	}
	.exp-debug-cell {
		padding-left: 0;
	}
	.exp-debug-cell .exp-debug-label {
		margin-left: 1rem;
	}
	.exp-debug-cell .exp-debug-label:first-child {
		margin-left: 0;
	}
	.exp-debug-label {
		color: var(--ink-faint);
		text-transform: lowercase;
	}

	.hash-copy {
		all: unset;
		cursor: pointer;
		display: inline-flex;
		align-items: baseline;
		gap: 0.3rem;
	}

	.hash-copy:hover .copy-mark {
		color: var(--accent);
	}

	.copy-mark {
		font-family: var(--mono);
		font-size: 0.74rem;
		color: var(--ink-faint);
	}


	/* Agent cards */
	.agent {
		padding: 0.5rem 0;
		border-bottom: 1px dotted var(--rule);
	}

	.agent-line {
		display: flex;
		gap: 0.6rem;
		font-size: 0.92rem;
	}

	.agent-role {
		font-family: var(--mono);
		font-size: 0.72rem;
		color: var(--ink-muted);
		text-transform: uppercase;
		min-width: 4ch;
	}

	.agent-name {
		font-family: var(--serif);
		font-weight: 500;
	}

	.agent-refs {
		display: flex;
		flex-wrap: wrap;
		gap: 0.3rem;
		margin-top: 0.25rem;
	}

	.ref-chip {
		font-family: var(--mono);
		font-size: 0.72rem;
		padding: 0.05rem 0.35rem;
		background: rgba(125, 42, 26, 0.06);
		color: var(--ink);
	}

	.ref-ns {
		color: var(--accent);
	}

	.agent-loc {
		font-family: var(--mono);
		font-size: 0.72rem;
		color: var(--ink-muted);
		margin-top: 0.25rem;
	}

	.agent-truth {
		font-family: var(--mono);
		font-size: 0.72rem;
		color: var(--accent);
		margin-top: 0.25rem;
	}

	.edges {
		list-style: none;
		padding: 0;
		margin: 0;
	}

	.edges li {
		padding: 0.15rem 0;
		font-family: var(--mono);
		font-size: 0.78rem;
	}

	.edge-kind {
		color: var(--ink-muted);
		display: inline-block;
		min-width: 12ch;
	}

	code {
		font-family: var(--mono);
		font-size: 0.88em;
	}
</style>
