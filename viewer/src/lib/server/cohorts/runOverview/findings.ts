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


/**
 * Rank the latest run's statements along five "interesting" axes. Each lane
 * returns up to K rows with the same shape; the UI renders them as compact
 * BeliefPrimitive cards. Empty lanes are normal — e.g. no prev_run means no
 * regressions can be computed.
 */
export async function getFindings(): Promise<Findings | null> {
	if (!dbExists()) return null;
	const con = await connect();
	try {
		const latestRows = await rows<{ run_id: string; architecture: string }>(
			con,
			`SELECT run_id, architecture FROM score_run
			 WHERE status='succeeded'
			 ORDER BY started_at DESC LIMIT 1`
		);
		if (latestRows.length === 0) return null;
		const run_id = latestRows[0].run_id;
		const qa = (latestRows[0].architecture ?? 'unknown').replace(/'/g, "''");
		const prevRows = await rows<{ run_id: string }>(
			con,
			`SELECT run_id FROM score_run
			 WHERE status='succeeded'
			   AND architecture='${qa}'
			   AND started_at < (SELECT started_at FROM score_run WHERE run_id='${run_id.replace(/'/g, "''")}')
			 ORDER BY started_at DESC LIMIT 1`
		);
		const prev_run_id = prevRows[0]?.run_id ?? null;
		const qr = run_id.replace(/'/g, "''");
		const qp = prev_run_id ? prev_run_id.replace(/'/g, "''") : null;

		const baseSelect = (whereExtra: string, orderBy: string) => `
				WITH ours AS (
					SELECT stmt_hash,
					       AVG(CAST(json_extract(output_json, '$.score') AS DOUBLE)) AS our,
					       CASE
					         WHEN SUM(CASE WHEN json_extract_string(output_json, '$.verdict')='incorrect' THEN 1 ELSE 0 END) > 0 THEN 'incorrect'
					         WHEN SUM(CASE WHEN json_extract_string(output_json, '$.verdict')='abstain' THEN 1 ELSE 0 END) > 0 THEN 'abstain'
					         WHEN SUM(CASE WHEN json_extract_string(output_json, '$.verdict')='correct' THEN 1 ELSE 0 END) = COUNT(*) THEN 'correct'
					         WHEN SUM(CASE WHEN json_extract_string(output_json, '$.verdict')='correct' THEN 1 ELSE 0 END) > 0 THEN 'mixed'
					         ELSE 'unknown'
					       END AS verdict
					FROM scorer_step
				WHERE run_id = '${qr}'
				  AND step_kind = 'aggregate'
				GROUP BY stmt_hash
			),
			ev_n AS (
				SELECT stmt_hash, COUNT(*) AS n_evidences FROM statement_evidence GROUP BY stmt_hash
			),
			ags AS (
				SELECT stmt_hash,
				       list({role: role, name: name}
				            ORDER BY CASE role
				              WHEN 'subj' THEN 0 WHEN 'enz' THEN 0
				              WHEN 'obj' THEN 1 WHEN 'sub' THEN 1
				              WHEN 'member' THEN 2 ELSE 3 END,
				            role_index) AS agents
				FROM agent GROUP BY stmt_hash
			)
			SELECT s.stmt_hash, s.indra_type, s.indra_belief AS indra_score,
			       ours.our AS our_score, ours.verdict AS verdict,
			       COALESCE(ev_n.n_evidences, 0) AS n_evidences,
			       ags.agents AS agents
			FROM statement s
			JOIN ours ON ours.stmt_hash = s.stmt_hash
			LEFT JOIN ev_n ON ev_n.stmt_hash = s.stmt_hash
			LEFT JOIN ags ON ags.stmt_hash = s.stmt_hash
			${whereExtra}
			ORDER BY ${orderBy}
			LIMIT ${FINDING_K}`;

		type Row = {
			stmt_hash: string;
			indra_type: string;
			indra_score: number | null;
			our_score: number | null;
			verdict: string | null;
			n_evidences: number;
			agents: Array<{ role: string; name: string }> | null;
		};

		const [disagreeRows, probeSplitRows, lowConfRows, prevAggRows] = await Promise.all([
			rows<Row>(
				con,
				baseSelect(
					`WHERE s.indra_belief IS NOT NULL AND ours.our IS NOT NULL`,
					`ABS(ours.our - s.indra_belief) DESC, s.stmt_hash`
				)
			),
			rows<Row & { probe_stdev: number }>(
				con,
				`WITH probe_votes AS (
					SELECT stmt_hash,
					       CASE
					         WHEN step_kind = 'subject_role_probe' AND json_extract_string(output_json, '$.answer') = 'present_as_subject' THEN 1.0
					         WHEN step_kind = 'object_role_probe'  AND json_extract_string(output_json, '$.answer') = 'present_as_object'  THEN 1.0
					         WHEN step_kind = 'relation_axis_probe' AND json_extract_string(output_json, '$.answer') = 'direct_sign_match' THEN 1.0
					         WHEN step_kind = 'scope_probe' AND json_extract_string(output_json, '$.answer') = 'asserted' THEN 1.0
					         WHEN step_kind = 'scope_probe' AND json_extract_string(output_json, '$.answer') = 'negated'  THEN -1.0
					         WHEN json_extract_string(output_json, '$.answer') IN ('absent','present_as_decoy','direct_sign_mismatch','direct_axis_mismatch','no_relation') THEN -1.0
					         ELSE 0
					       END AS vote
					FROM scorer_step
					WHERE run_id = '${qr}'
					  AND step_kind IN ('subject_role_probe','object_role_probe','relation_axis_probe','scope_probe')
				),
				stdevs AS (
					SELECT stmt_hash, STDDEV_POP(vote) AS probe_stdev
					FROM probe_votes GROUP BY stmt_hash
				)
				SELECT s.stmt_hash, s.indra_type, s.indra_belief AS indra_score,
					       (SELECT AVG(CAST(json_extract(output_json, '$.score') AS DOUBLE))
					        FROM scorer_step WHERE run_id='${qr}'
					          AND step_kind='aggregate' AND stmt_hash=s.stmt_hash) AS our_score,
					       (SELECT CASE
					          WHEN SUM(CASE WHEN json_extract_string(output_json, '$.verdict')='incorrect' THEN 1 ELSE 0 END) > 0 THEN 'incorrect'
					          WHEN SUM(CASE WHEN json_extract_string(output_json, '$.verdict')='abstain' THEN 1 ELSE 0 END) > 0 THEN 'abstain'
					          WHEN SUM(CASE WHEN json_extract_string(output_json, '$.verdict')='correct' THEN 1 ELSE 0 END) = COUNT(*) THEN 'correct'
					          WHEN SUM(CASE WHEN json_extract_string(output_json, '$.verdict')='correct' THEN 1 ELSE 0 END) > 0 THEN 'mixed'
					          ELSE 'unknown'
					        END
					        FROM scorer_step WHERE run_id='${qr}'
					          AND step_kind='aggregate' AND stmt_hash=s.stmt_hash) AS verdict,
				       (SELECT COUNT(*) FROM statement_evidence WHERE stmt_hash=s.stmt_hash) AS n_evidences,
				       (SELECT list({role: role, name: name}
				                    ORDER BY CASE role
				                      WHEN 'subj' THEN 0 WHEN 'enz' THEN 0
				                      WHEN 'obj' THEN 1 WHEN 'sub' THEN 1
				                      WHEN 'member' THEN 2 ELSE 3 END,
				                    role_index)
				        FROM agent WHERE stmt_hash=s.stmt_hash) AS agents,
				       stdevs.probe_stdev AS probe_stdev
				FROM statement s
				JOIN stdevs ON stdevs.stmt_hash = s.stmt_hash
				WHERE stdevs.probe_stdev > 0
				ORDER BY stdevs.probe_stdev DESC, s.stmt_hash
				LIMIT ${FINDING_K}`
			),
			rows<Row>(
				con,
				baseSelect(
					`WHERE ours.our BETWEEN 0.4 AND 0.6
					  AND COALESCE(ev_n.n_evidences, 0) >= 3`,
					`ev_n.n_evidences DESC, ABS(ours.our - 0.5) ASC, s.stmt_hash`
				)
			),
			prev_run_id
				? rows<{ stmt_hash: string; verdict: string }>(
						con,
							`SELECT stmt_hash,
							        CASE
							          WHEN SUM(CASE WHEN json_extract_string(output_json, '$.verdict')='incorrect' THEN 1 ELSE 0 END) > 0 THEN 'incorrect'
							          WHEN SUM(CASE WHEN json_extract_string(output_json, '$.verdict')='abstain' THEN 1 ELSE 0 END) > 0 THEN 'abstain'
							          WHEN SUM(CASE WHEN json_extract_string(output_json, '$.verdict')='correct' THEN 1 ELSE 0 END) = COUNT(*) THEN 'correct'
							          WHEN SUM(CASE WHEN json_extract_string(output_json, '$.verdict')='correct' THEN 1 ELSE 0 END) > 0 THEN 'mixed'
							          ELSE 'unknown'
							        END AS verdict
							 FROM scorer_step WHERE run_id='${qp}' AND step_kind='aggregate'
							 GROUP BY stmt_hash`
					)
				: Promise.resolve([])
		]);

		const prevVerdict = new Map(prevAggRows.map((r) => [r.stmt_hash, r.verdict]));
		const allCurrent = await rows<Row>(
			con,
			baseSelect('WHERE 1=1', 's.stmt_hash')
		);
		const verdictMoved = (curr: string | null, prev: string | undefined, dir: 'down' | 'up') => {
			if (!curr || !prev) return false;
			if (dir === 'down') return prev === 'correct' && curr === 'incorrect';
			return prev === 'incorrect' && curr === 'correct';
		};
		const regressions: Row[] = [];
		const recoveries: Row[] = [];
		for (const r of allCurrent) {
			const pv = prevVerdict.get(r.stmt_hash);
			if (verdictMoved(r.verdict, pv, 'down')) regressions.push(r);
			if (verdictMoved(r.verdict, pv, 'up')) recoveries.push(r);
		}

		const toFinding = (
			kind: FindingKind,
			r: Row,
			rank: number,
			why: string
		): FindingRow => ({
			kind,
			stmt_hash: r.stmt_hash,
			indra_type: r.indra_type,
			agents: r.agents ?? [],
			our_score: r.our_score,
			indra_score: r.indra_score,
			n_evidences: r.n_evidences,
			rank_value: rank,
			why_text: why
		});

		const fmtSigned = (n: number) => `${n >= 0 ? '+' : '−'}${Math.abs(n).toFixed(2)}`;

		return {
			run_id,
			prev_run_id,
			biggest_disagreement: disagreeRows.map((r) => {
				const d = (r.our_score ?? 0) - (r.indra_score ?? 0);
				return toFinding(
					'biggest_disagreement',
					r,
					Math.abs(d),
					`Δ ${fmtSigned(d)} vs INDRA · n_ev=${r.n_evidences}`
				);
			}),
			probe_split: probeSplitRows.map((r) =>
				toFinding(
					'probe_split',
					r,
					r.probe_stdev,
					`probe stdev ${r.probe_stdev.toFixed(2)} · disagreement across the four probes`
				)
			),
			verdict_regression: regressions
				.slice(0, FINDING_K)
				.map((r) =>
					toFinding(
						'verdict_regression',
						r,
						1,
						`verdict moved correct → incorrect since prev run`
					)
				),
			verdict_recovery: recoveries
				.slice(0, FINDING_K)
				.map((r) =>
					toFinding(
						'verdict_recovery',
						r,
						1,
						`verdict moved incorrect → correct since prev run`
					)
				),
			low_confidence_high_stakes: lowConfRows.map((r) =>
				toFinding(
					'low_confidence_high_stakes',
					r,
					r.n_evidences,
					`belief ${(r.our_score ?? 0).toFixed(2)} (mid-range) · n_ev=${r.n_evidences} ≥ 3`
				)
			)
		};
	} finally {
		con.disconnectSync?.();
	}
}

