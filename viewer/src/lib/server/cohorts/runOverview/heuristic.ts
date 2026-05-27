// Split from $lib/server/cohorts/runOverview.ts on 2026-05-27
// in response to brutalist HIGH finding (one file, multiple surfaces).
// Original db.ts callers continue to import via $lib/db (re-export
// chain: db.ts -> runOverview/index.ts -> this file).

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


export function stepOrder(kind: string): number {
	return (
		{
			parse_claim: 0,
			build_context: 1,
			substrate_route: 2,
			subject_role_probe: 3,
			object_role_probe: 4,
			relation_axis_probe: 5,
			scope_probe: 6,
			grounding: 7,
			adjudicate: 8,
			aggregate: 9
		} as Record<string, number>
	)[kind] ?? 99;
}



export function heuristicLifecycleMessage(
	architecture: string,
	counts: Omit<HeuristicCoverageTraceDiagnostic, 'kind' | 'message' | 'lifecycle_kind' | 'lifecycle_message'>
): { lifecycle_kind: HeuristicCoverageTraceLifecycleKind; lifecycle_message: string } {
	const status = counts.run_status || 'unknown';
	const termination = [counts.terminated_by, counts.termination_reason].filter(Boolean).join(' · ');
	if (architecture !== 'decomposed') {
		return {
			lifecycle_kind: 'not_applicable',
			lifecycle_message: 'run lifecycle does not define four-probe trace cause for this architecture'
		};
	}
	if (counts.n_substrate_route_steps > 0 || counts.n_probe_steps > 0) {
		return {
			lifecycle_kind: 'probe_records_present',
			lifecycle_message: 'probe rows are present, so substrate/LLM coverage is measured from persisted trace slots'
		};
	}
	if (status === 'running') {
		return {
			lifecycle_kind: 'running_snapshot',
			lifecycle_message:
				'run lifecycle is running; this page is a persisted snapshot and probe absence can still change as the worker appends rows'
		};
	}
	if (status === 'failed' || status === 'canceled' || status === 'cancelled' || status === 'aborted') {
		return {
			lifecycle_kind: 'interrupted_run',
			lifecycle_message:
				`run lifecycle is ${status}${termination ? ` (${termination})` : ''}; probe absence is in an interrupted-run context, so coverage is not a completed trace-health measurement`
		};
	}
	if (status === 'succeeded') {
		return {
			lifecycle_kind: 'succeeded_without_cause_provenance',
			lifecycle_message:
				'run lifecycle is succeeded, so the stored run row does not support user cancellation as the explanation; importer, worker-log, and migration provenance are still absent, so root cause remains unclassified'
		};
	}
	return {
		lifecycle_kind: 'unknown_lifecycle',
		lifecycle_message:
			`run lifecycle is ${status}; without a recognized terminal/running state, the UI cannot classify why probe records are absent`
	};
}



export function heuristicTraceDiagnostic(
	architecture: string,
	counts: Omit<HeuristicCoverageTraceDiagnostic, 'kind' | 'message' | 'lifecycle_kind' | 'lifecycle_message'>
): HeuristicCoverageTraceDiagnostic {
	const lifecycle = heuristicLifecycleMessage(architecture, counts);
	if (architecture !== 'decomposed') {
		return {
			...counts,
			...lifecycle,
			kind: 'not_applicable',
			message: 'four-probe trace diagnostics are not defined for this architecture'
		};
	}
		if (counts.n_substrate_route_steps > 0 || counts.n_probe_steps > 0) {
			return {
				...counts,
				...lifecycle,
				kind: 'probe_rows_present',
				message: counts.n_aggregate_evidences === 0
					? 'persisted decomposed substrate/probe rows exist before aggregate verdict rows; the coverage table is measured from trace records while the aggregate verdict denominator remains 0'
					: 'persisted decomposed substrate/probe rows exist, so the coverage table is measured from trace records'
			};
		}
		if (counts.n_aggregate_evidences === 0) {
			return {
				...counts,
				...lifecycle,
				kind: 'no_aggregate_evidence',
				message: 'no aggregate evidence rows are finalized for this run yet'
			};
		}
	if (counts.n_nonaggregate_steps > 0) {
		return {
			...counts,
			...lifecycle,
			kind: 'native_steps_without_probe_slots',
			message:
				`persisted scorer_step rows include ${counts.n_nonaggregate_steps} decomposed native step row${counts.n_nonaggregate_steps === 1 ? '' : 's'}, but 0 substrate_route/probe rows; probe health is unavailable until those slot records exist`
		};
	}
	return {
		...counts,
		...lifecycle,
		kind: 'aggregate_only_trace',
		message:
			`persisted scorer_step rows show ${counts.n_aggregate_evidences} aggregate evidence row${counts.n_aggregate_evidences === 1 ? '' : 's'} and 0 decomposed native step rows; the DB cannot distinguish legacy/imported aggregate-only data from failed trace persistence`
	};
}



/**
 * Per-probe substrate/LLM/abstain coverage for a single run.
 *
 * Reads two sources and unions:
 *   1. substrate_route rows expose per-probe slot results.
 *   2. Individual {probe}_probe rows are written when substrate answers
 *      OR when LLM fires its own probe step.
 *
 * Coverage = the "final" source per (evidence, probe), with substrate-route's
 * slot taking precedence (it's authoritative for what substrate decided).
 * Missing data on both sides means "probe did not run for this evidence" —
 * counted in `notrun_n`, not silently dropped.
 */
export async function getHeuristicCoverage(run_id: string): Promise<HeuristicCoverage | null> {
	if (!dbExists()) return null;
	const con = await connect();
	try {
		const qr = run_id.replace(/'/g, "''");
		const runRows = await rows<{
			architecture: string | null;
			scorer_version: string | null;
			started_at: string | null;
			status: string | null;
			terminated_by: string | null;
			termination_reason: string | null;
			n_evidences: number;
			n_nonaggregate_steps: number;
			n_substrate_route_steps: number;
			n_probe_steps: number;
			n_grounding_steps: number;
			n_adjudicate_steps: number;
		}>(
			con,
			`SELECT
			   architecture,
			   scorer_version,
			   started_at::VARCHAR AS started_at,
			   status,
			   terminated_by,
			   termination_reason,
			   (SELECT COUNT(DISTINCT evidence_hash)
			    FROM scorer_step
			    WHERE run_id='${qr}' AND step_kind='aggregate') AS n_evidences,
			   (SELECT COUNT(*)
			    FROM scorer_step
			    WHERE run_id='${qr}'
			      AND evidence_hash IS NOT NULL
			      AND step_kind<>'aggregate') AS n_nonaggregate_steps,
			   (SELECT COUNT(*)
			    FROM scorer_step
			    WHERE run_id='${qr}'
			      AND evidence_hash IS NOT NULL
			      AND step_kind='substrate_route') AS n_substrate_route_steps,
			   (SELECT COUNT(*)
			    FROM scorer_step
			    WHERE run_id='${qr}'
			      AND evidence_hash IS NOT NULL
			      AND step_kind IN ('subject_role_probe','object_role_probe','relation_axis_probe','scope_probe')) AS n_probe_steps,
			   (SELECT COUNT(*)
			    FROM scorer_step
			    WHERE run_id='${qr}'
			      AND evidence_hash IS NOT NULL
			      AND step_kind='grounding') AS n_grounding_steps,
			   (SELECT COUNT(*)
			    FROM scorer_step
			    WHERE run_id='${qr}'
			      AND evidence_hash IS NOT NULL
			      AND step_kind='adjudicate') AS n_adjudicate_steps
			 FROM score_run
			 WHERE run_id='${qr}'`
		);
		if (runRows.length === 0) return null;
		const architecture = runRows[0].architecture ?? 'unknown';
		const trace_diagnostic = heuristicTraceDiagnostic(architecture, {
			n_aggregate_evidences: Number(runRows[0].n_evidences ?? 0),
			n_nonaggregate_steps: Number(runRows[0].n_nonaggregate_steps ?? 0),
			n_substrate_route_steps: Number(runRows[0].n_substrate_route_steps ?? 0),
			n_probe_steps: Number(runRows[0].n_probe_steps ?? 0),
			n_grounding_steps: Number(runRows[0].n_grounding_steps ?? 0),
			n_adjudicate_steps: Number(runRows[0].n_adjudicate_steps ?? 0),
			run_status: runRows[0].status ?? 'unknown',
			scorer_version: runRows[0].scorer_version ?? null,
			started_at: runRows[0].started_at ?? null,
			terminated_by: runRows[0].terminated_by ?? null,
			termination_reason: runRows[0].termination_reason ?? null
		});
			if (architecture !== 'decomposed') {
				return {
					run_id,
					architecture,
					applicability: 'not_defined',
					not_defined_reason:
						'four-probe substrate/LLM coverage is defined only for decomposed runs; monolithic runs use a single aggregate tier grammar',
					n_evidences: runRows[0].n_evidences ?? 0,
					n_probe_evidences: 0,
					per_probe: [],
					trace_diagnostic,
					all_substrate_rate: 0,
				short_circuited_rate: 0,
				mean_invoked_probes: 0
			};
			}
			const probes: ProbeKind[] = ['subject_role', 'object_role', 'relation_axis', 'scope'];
			const probeNames = probes.map((p) => `SELECT '${p}' AS probe`).join(' UNION ALL ');
			const slotUnions = probes
				.map(
					(p) => `
					SELECT evidence_hash, '${p}' AS probe,
				       json_extract_string(output_json, '$.${p}.source') AS substrate_source,
				       json_extract_string(output_json, '$.${p}.answer') AS substrate_answer
					FROM scorer_step
					WHERE run_id='${qr}' AND step_kind='substrate_route'`
				)
				.join(' UNION ALL ');
			const probeCoverageCtes = `
				probe_names AS (${probeNames}),
				substrate_slots AS (${slotUnions}),
				individual_probes AS (
					SELECT evidence_hash,
					       replace(step_kind, '_probe', '') AS probe,
					       json_extract_string(output_json, '$.source') AS llm_source,
					       json_extract_string(output_json, '$.answer') AS llm_answer
					FROM scorer_step
					WHERE run_id='${qr}'
					  AND step_kind IN ('subject_role_probe','object_role_probe','relation_axis_probe','scope_probe')
				),
				probe_evidence AS (
					SELECT DISTINCT evidence_hash FROM substrate_slots
					UNION
					SELECT DISTINCT evidence_hash FROM individual_probes
				),
				probe_keys AS (
					SELECT pe.evidence_hash, pn.probe
					FROM probe_evidence pe
					CROSS JOIN probe_names pn
				),
				joined AS (
					SELECT pk.evidence_hash, pk.probe,
					       CASE
					         WHEN ss.substrate_source = 'substrate' THEN 'substrate'
					         WHEN ip.llm_source = 'llm' THEN 'llm'
					         WHEN ip.llm_source = 'abstain' OR ip.llm_answer = 'abstain' THEN 'abstain'
					         WHEN ip.llm_source IS NOT NULL THEN ip.llm_source
					         ELSE NULL
					       END AS final_source
					FROM probe_keys pk
					LEFT JOIN substrate_slots ss
					  ON ss.evidence_hash = pk.evidence_hash AND ss.probe = pk.probe
					LEFT JOIN individual_probes ip
					  ON ip.evidence_hash = pk.evidence_hash AND ip.probe = pk.probe
				)`;
			const sql = `
				WITH ${probeCoverageCtes},
				per_evidence AS (
					SELECT evidence_hash,
					       SUM(CASE WHEN final_source = 'substrate' THEN 1 ELSE 0 END) AS substrate_count,
				       SUM(CASE WHEN final_source = 'llm' THEN 1 ELSE 0 END) AS llm_count,
				       SUM(CASE WHEN final_source = 'abstain' THEN 1 ELSE 0 END) AS abstain_count,
				       SUM(CASE WHEN final_source IS NULL THEN 1 ELSE 0 END) AS notrun_count
				FROM joined
				GROUP BY evidence_hash
			)
			SELECT
				probe,
				COUNT(*) AS total,
				SUM(CASE WHEN final_source='substrate' THEN 1 ELSE 0 END) AS substrate_n,
				SUM(CASE WHEN final_source='llm' THEN 1 ELSE 0 END) AS llm_n,
				SUM(CASE WHEN final_source='abstain' THEN 1 ELSE 0 END) AS abstain_n,
				SUM(CASE WHEN final_source IS NULL THEN 1 ELSE 0 END) AS notrun_n
			FROM joined
			GROUP BY probe
			ORDER BY probe`;
		const perProbeRaw = await rows<{
			probe: string;
			total: number;
			substrate_n: number;
			llm_n: number;
			abstain_n: number;
			notrun_n: number;
			}>(con, sql);
			const per_probe: ProbeCoverageRow[] = perProbeRaw.map((r) => ({
				probe: r.probe as ProbeKind,
				total: Number(r.total ?? 0),
				substrate_n: Number(r.substrate_n ?? 0),
				llm_n: Number(r.llm_n ?? 0),
				abstain_n: Number(r.abstain_n ?? 0),
				notrun_n: Number(r.notrun_n ?? 0)
			}));

		const aggRows = await rows<{
			n_evidences: number;
			all_substrate: number;
			short_circuited: number;
			mean_invoked: number;
		}>(
				con,
				`WITH ${probeCoverageCtes},
				per_evidence AS (
					SELECT evidence_hash,
					       SUM(CASE WHEN final_source='substrate' THEN 1 ELSE 0 END) AS substrate_count,
				       SUM(CASE WHEN final_source='llm' THEN 1 ELSE 0 END) AS llm_count,
				       SUM(CASE WHEN final_source='abstain' THEN 1 ELSE 0 END) AS abstain_count,
				       SUM(CASE WHEN final_source IS NULL THEN 1 ELSE 0 END) AS notrun_count
				FROM joined
				GROUP BY evidence_hash
			)
			SELECT
				COUNT(*) AS n_evidences,
				SUM(CASE WHEN llm_count=0 AND abstain_count=0 THEN 1 ELSE 0 END) AS all_substrate,
				SUM(CASE WHEN notrun_count > 0 THEN 1 ELSE 0 END) AS short_circuited,
				AVG(substrate_count + llm_count + abstain_count) AS mean_invoked
			FROM per_evidence`
		);

			const agg = aggRows[0] ?? { n_evidences: 0, all_substrate: 0, short_circuited: 0, mean_invoked: 0 };
			const n_evidences = Number(runRows[0].n_evidences ?? 0);
			const n_probe_evidences = Number(agg.n_evidences ?? 0);
			return {
				run_id,
				architecture,
				applicability: 'arch_conditioned',
				not_defined_reason: null,
				n_evidences,
				n_probe_evidences,
				per_probe,
				trace_diagnostic,
				all_substrate_rate: n_probe_evidences > 0 ? Number(agg.all_substrate ?? 0) / n_probe_evidences : 0,
				short_circuited_rate: n_probe_evidences > 0 ? Number(agg.short_circuited ?? 0) / n_probe_evidences : 0,
				mean_invoked_probes: Number(agg.mean_invoked ?? 0)
			};
	} finally {
		con.disconnectSync?.();
	}
}

