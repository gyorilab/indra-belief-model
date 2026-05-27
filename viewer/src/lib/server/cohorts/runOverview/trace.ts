// Split from $lib/server/cohorts/runOverview.ts on 2026-05-27
// in response to brutalist HIGH finding (one file, multiple surfaces).
// Original db.ts callers continue to import via $lib/db (re-export
// chain: db.ts -> runOverview/index.ts -> this file).
import { stepOrder } from './heuristic';

// Auto-extracted from $lib/db.ts on 2026-05-27 to satisfy B4 of the
// active goal (cohort orchestration extraction).
//
// All callers continue to import these symbols from $lib/db; db.ts
// re-exports each function from the corresponding $lib/server/cohorts/*
// module. Shared internal helpers (sqlQuote, rows, readRows, scalar,
// tableExists, normalizeDuckValue, timestampMs, durationSeconds,
// safeRate, repairRerunLineageSqlOptions, etc.) and connection
// management (connect, closeInstance, dbPath, dbExists) still live in
// $lib/db; this file imports them from there.

import { FINDING_K, TRACE_FIDELITY_DETAIL_LIMIT, closeInstance, connect, dbExists, dbPath, durationSeconds, normalizeDuckValue, readRows, repairRerunLineageSqlOptions, resolveTraceSnapshotStartedAt, rows, safeRate, scalar, sqlQuote, tableExists, timestampMs } from '$lib/db';
import type { Findings, HeuristicCoverage, ResidualDistribution, RunNarrative } from '$lib/db';
import type { ProbeKind } from '$lib/probeAttribution';
import { isTraceFidelityState, zeroTraceCounts } from '$lib/traceState';
import { traceEvidenceCteForRun } from '$lib/traceStateSql';
import type { DuckDBConnection } from '@duckdb/node-api';
import { error } from '@sveltejs/kit';

import type { HeuristicCoverageTraceDiagnostic, HeuristicCoverageTraceLifecycleKind, ProbeCoverageRow, TraceFidelityOptions, TraceFidelitySummary, TraceFidelityRow, PanelApplicabilityRow, FindingKind, FindingRow } from '$lib/db';


export async function getTraceFidelity(
	run_id: string,
	options: TraceFidelityOptions = {}
): Promise<TraceFidelitySummary | null> {
	if (!dbExists()) return null;
	const con = await connect();
	try {
		const qr = run_id.replace(/'/g, "''");
		const runRows = await rows<{ architecture: string | null; paired_run_group_id: string | null; status: string | null }>(
			con,
			`SELECT architecture, paired_run_group_id, status
			 FROM score_run
			 WHERE run_id='${qr}'`
		);
		if (runRows.length === 0) return null;
		const architecture = runRows[0].architecture ?? 'unknown';
		const pairedGroup = runRows[0].paired_run_group_id ?? null;
		const runStatus = runRows[0].status ?? 'unknown';
		const requestedDetailLimit = Number.isFinite(options.detailLimit)
			? Math.floor(options.detailLimit as number)
			: TRACE_FIDELITY_DETAIL_LIMIT;
		const detailLimit = Math.min(100, Math.max(1, requestedDetailLimit));
		const detailSnapshotStartedAt = await resolveTraceSnapshotStartedAt(
			con,
			run_id,
			options.detailSnapshotStartedAt,
			runStatus === 'running'
		);
		const summaryRows = await rows<{ trace_state: string; n: number | bigint }>(
			con,
			`WITH ${traceEvidenceCteForRun(run_id, architecture, runStatus, detailSnapshotStartedAt)}
			 SELECT trace_state, COUNT(*) AS n
			 FROM trace_rows
			 GROUP BY trace_state`
		);

		const counts = zeroTraceCounts();
		for (const row of summaryRows) {
			if (!isTraceFidelityState(row.trace_state)) {
				throw new Error(`trace CTE emitted invalid trace_state: ${row.trace_state}`);
			}
			counts[row.trace_state] = Number(row.n ?? 0);
		}
		const nEvidences = Object.values(counts).reduce((sum, n) => sum + n, 0);
		const requestedOffset = Math.max(0, Math.floor(options.detailOffset ?? 0));
		const requestedPageOffset = Math.floor(requestedOffset / detailLimit) * detailLimit;
		const maxOffset = nEvidences > 0 ? Math.floor((nEvidences - 1) / detailLimit) * detailLimit : 0;
		const detailOffset = Math.min(requestedPageOffset, maxOffset);
		const traceRows = await rows<{
			stmt_hash: string | null;
			evidence_hash: string;
			trace_state: string;
			captured_step_kinds: string | null;
			n_parse_claim: number | bigint | null;
			n_build_context: number | bigint | null;
			n_substrate_route: number | bigint | null;
			n_subject_role_probe: number | bigint | null;
			n_object_role_probe: number | bigint | null;
			n_relation_axis_probe: number | bigint | null;
			n_scope_probe: number | bigint | null;
			n_grounding: number | bigint | null;
			n_adjudicate: number | bigint | null;
			has_tier: number | bigint | null;
			has_grounding_status: number | bigint | null;
			has_parse_state: number | bigint | null;
			has_call_log: number | bigint | null;
			has_raw_text: number | bigint | null;
			has_selected_examples: number | bigint | null;
		}>(
			con,
			`WITH ${traceEvidenceCteForRun(run_id, architecture, runStatus, detailSnapshotStartedAt)}
			 SELECT stmt_hash,
			        evidence_hash,
			        trace_state,
			        captured_step_kinds,
			        n_parse_claim,
			        n_build_context,
			        n_substrate_route,
			        n_subject_role_probe,
			        n_object_role_probe,
			        n_relation_axis_probe,
			        n_scope_probe,
			        n_grounding,
			        n_adjudicate,
			        has_tier,
			        has_grounding_status,
			        has_parse_state,
			        has_call_log,
			        has_raw_text,
			        has_selected_examples
			 FROM trace_rows
			 ORDER BY evidence_hash
			 LIMIT ${detailLimit} OFFSET ${detailOffset}`
		);

		const rowsOut: TraceFidelityRow[] = [];
		const present = (v: number | bigint | null | undefined) => Number(v ?? 0) > 0;
		for (const traceRow of traceRows) {
			const evidence_hash = traceRow.evidence_hash;
			const captured_steps = (traceRow.captured_step_kinds ?? '')
				.split(',')
				.filter(Boolean)
				.sort((a, b) => stepOrder(a) - stepOrder(b));
			if (!isTraceFidelityState(traceRow.trace_state)) {
				throw new Error(`trace CTE emitted invalid trace_state: ${traceRow.trace_state}`);
			}
			const state = traceRow.trace_state;
			let missing_native_steps: string[] = [];
			let note = '';

			if (state === 'step_error') {
				note = 'one or more persisted scorer steps carries an error';
			} else if (state === 'missing_aggregate') {
				note = 'no aggregate row was persisted for this evidence in a succeeded run';
				missing_native_steps.push('aggregate');
			} else if (state === 'terminated_inflight') {
				note = runStatus === 'running'
					? 'this evidence has persisted pre-aggregate steps and may still reach aggregate'
					: 'this evidence was in flight when the run terminated; no aggregate verdict was finalized';
				missing_native_steps.push('aggregate');
			} else if (architecture === 'decomposed') {
				const core = ['parse_claim', 'build_context', 'substrate_route'];
				const hasCore = {
					parse_claim: present(traceRow.n_parse_claim),
					build_context: present(traceRow.n_build_context),
					substrate_route: present(traceRow.n_substrate_route)
				};
				const coreMissing = core.filter((k) => !hasCore[k as keyof typeof hasCore]);
				const probeSteps = [
					'subject_role_probe',
					'object_role_probe',
					'relation_axis_probe',
					'scope_probe'
				];
				const hasProbe = {
					subject_role_probe: present(traceRow.n_subject_role_probe),
					object_role_probe: present(traceRow.n_object_role_probe),
					relation_axis_probe: present(traceRow.n_relation_axis_probe),
					scope_probe: present(traceRow.n_scope_probe)
				};
				const probeMissing = probeSteps.filter((k) => !hasProbe[k as keyof typeof hasProbe]);
				const fullTail = ['grounding', 'adjudicate'];
				const hasTail = {
					grounding: present(traceRow.n_grounding),
					adjudicate: present(traceRow.n_adjudicate)
				};
				const tailMissing = fullTail.filter((k) => !hasTail[k as keyof typeof hasTail]);
				if (state === 'aggregate_only') {
					missing_native_steps = [...core, ...probeSteps, ...fullTail];
					note = 'only the final aggregate dict is present; decomposed substeps were not persisted for this run';
				} else if (state === 'full') {
					note = 'native decomposed trace rows include all named probe events through adjudication';
				} else {
					missing_native_steps = [...coreMissing, ...probeMissing, ...tailMissing];
					note = probeMissing.length > 0
						? 'decomposed trace rows are visible, but one or more named probe event rows is absent; route slots alone do not prove full probe-event persistence'
						: tailMissing.length > 0
							? 'deterministic and probe event rows are visible, while later grounding/adjudication remains collapsed into the aggregate row'
							: 'some native decomposed substeps are visible, but the trace is not complete';
				}
			} else if (architecture === 'monolithic') {
				const hasTier = present(traceRow.has_tier);
				const hasGroundingStatus = present(traceRow.has_grounding_status);
				const hasParseState = present(traceRow.has_parse_state);
				const hasCallLog = present(traceRow.has_call_log);
				const hasRawText = present(traceRow.has_raw_text);
				const hasSelectedExamples = present(traceRow.has_selected_examples);
				if (state === 'full') {
					note = 'monolithic tier, grounding, call, examples, and parse fields are captured';
				} else if (state === 'partial') {
					missing_native_steps = [
						...(hasTier ? [] : ['tier']),
						...(hasGroundingStatus ? [] : ['grounding_status']),
						...(hasParseState ? [] : ['verdict_or_score']),
						...(hasCallLog ? [] : ['call_log']),
						...(hasRawText ? [] : ['raw_text']),
						...(hasSelectedExamples ? [] : ['selected_examples'])
					];
					note = 'monolithic native aggregate fields are present, but not every diagnostic field is persisted';
				} else {
					missing_native_steps = ['tier', 'grounding_status', 'verdict_or_score', 'call_log', 'raw_text', 'selected_examples'];
					note = 'the final dict exists, but monolithic native diagnostic fields are absent';
				}
			} else {
				missing_native_steps = ['architecture'];
				note = 'run architecture is unknown, so only aggregate presence can be interpreted';
			}

			rowsOut.push({
				stmt_hash: traceRow.stmt_hash ?? '',
				evidence_hash,
				state,
				captured_steps,
				missing_native_steps,
				note
			});
		}

		const native_grammar = architecture === 'monolithic'
			? ['tier-1 grounding gate', 'tier-2 comprehension/tool path', 'model call log', 'response parse', 'verdict bucket']
			: architecture === 'decomposed'
				? ['parse claim', 'build context', 'substrate route', 'subject-role probe', 'object-role probe', 'relation-axis probe', 'scope probe', 'grounding/adjudicate', 'aggregate verdict']
				: ['aggregate verdict'];
		const limitations = architecture === 'monolithic'
			? ['selected example IDs are expected on monolithic LLM-tier aggregate rows; deterministic tier-1 rows may have no example prompt', 'decomposed probe slots are not part of this architecture']
			: architecture === 'decomposed'
				? ['full fidelity requires the four named probe event rows; substrate-route slots alone are partial trace evidence', 'LLM-escalated probe details and adjudication may still be collapsed into aggregate call_log rows']
				: ['architecture is unknown; trace fidelity is limited to aggregate row presence'];

		const panel_applicability: PanelApplicabilityRow[] = [
			{
				panel: 'verdict, belief, cost, latency',
				applicability: 'arch_blind',
				reason: 'these fields are read from aggregate scorer_step rows and apply to both architectures',
				cohort_href: `/runs/${run_id}`
			},
			{
				panel: 'four-probe substrate/LLM coverage',
				applicability: architecture === 'decomposed' ? 'arch_conditioned' : 'not_defined',
				reason: architecture === 'decomposed'
					? 'this run uses the decomposed four-probe grammar'
					: 'monolithic runs do not emit subject/object/relation/scope probe slots',
				cohort_href: architecture === 'decomposed' ? `/runs/${run_id}` : null
			},
			{
				panel: 'native monolithic tier path',
				applicability: architecture === 'monolithic' ? 'arch_conditioned' : 'not_defined',
				reason: architecture === 'monolithic'
					? 'this run emits tier and grounding-path fields on aggregate rows'
					: 'decomposed runs route through deterministic/probe/adjudication stages instead',
				cohort_href: architecture === 'monolithic' ? `/runs/${run_id}` : null
			},
			{
				panel: 'paired architecture deltas',
				applicability: pairedGroup ? 'paired_only' : 'not_defined',
				reason: pairedGroup
					? `requires overlap inside paired group ${pairedGroup}`
					: 'requires a shared paired_run_group_id before deltas are meaningful',
				cohort_href: pairedGroup ? `/pairs/${pairedGroup}` : null
			}
		];

		return {
			run_id,
			architecture,
			n_evidences: nEvidences,
			counts,
			native_grammar,
			limitations,
			rows: rowsOut,
			detail_offset: detailOffset,
			detail_limit: detailLimit,
			detail_snapshot_started_at: detailSnapshotStartedAt,
			panel_applicability
		};
	} finally {
		con.disconnectSync?.();
	}
}

