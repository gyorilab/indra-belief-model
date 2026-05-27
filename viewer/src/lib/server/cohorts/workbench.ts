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

import { closeInstance, connect, dbExists, dbPath, durationSeconds, normalizeDuckValue, readRows, repairRerunLineageSqlOptions, rows, safeRate, scalar, sqlQuote, tableExists, timestampMs } from '$lib/db';
import type { PairRunSummary, PairedArchProbeRow, PairedArchTierRow, PairedComparableMetrics, PairedDenominatorRow, PairedExampleRow, PairedOverlapStats, PairedResourceFrontier, PairedResourceFrontierArch, PairedVerdictPairRow, PairedWorkbench } from '$lib/db';
import { error } from '@sveltejs/kit';

export async function getPairedWorkbench(pair_id: string): Promise<PairedWorkbench | null> {
	if (!dbExists()) return null;
	const con = await connect();
	try {
		const qp = sqlQuote(pair_id);
		const runsRaw = await rows<PairRunSummary>(
			con,
			`SELECT
			   sr.run_id,
			   COALESCE(sr.architecture, 'unknown') AS architecture,
				   sr.scorer_version,
			   sr.model_id_default,
			   sr.started_at::VARCHAR AS started_at,
			   sr.finished_at::VARCHAR AS finished_at,
			   sr.status,
			   sr.n_stmts,
			   (SELECT COUNT(DISTINCT evidence_hash)
			    FROM scorer_step
			    WHERE run_id=sr.run_id AND step_kind='aggregate' AND evidence_hash IS NOT NULL) AS n_evidences,
			   sr.cost_estimate_usd,
			   sr.cost_actual_usd
			 FROM score_run sr
			 WHERE sr.paired_run_group_id='${qp}'
			 ORDER BY CASE sr.status WHEN 'succeeded' THEN 0 ELSE 1 END,
			          sr.started_at DESC`
		);
		if (runsRaw.length === 0) return null;
		const runs = runsRaw.map((r) => ({
			...r,
			n_stmts: r.n_stmts == null ? null : Number(r.n_stmts),
			n_evidences: Number(r.n_evidences ?? 0),
			finished_at: r.finished_at ?? null,
			duration_s: durationSeconds(r.started_at, r.finished_at),
			cost_estimate_usd: r.cost_estimate_usd == null ? null : Number(r.cost_estimate_usd),
			cost_actual_usd: r.cost_actual_usd == null ? null : Number(r.cost_actual_usd)
		}));
		const pick = (arch: string) =>
			runs.find((r) => r.architecture === arch && r.status === 'succeeded')
			?? runs.find((r) => r.architecture === arch)
			?? null;
		const monolithic = pick('monolithic');
		const decomposed = pick('decomposed');

		const empty: PairedWorkbench = {
			pair_id,
			runs,
			monolithic,
			decomposed,
			overlap: null,
			comparable: null,
			resource_frontier: null,
			denominator_ledger: [],
			exemplars: {
				monolithic_wins: [],
				decomposed_wins: [],
				verdict_disagreements: [],
				mutual_failures: [],
				monolithic_only: [],
				decomposed_only: [],
				excluded_by_integrity: []
			},
			arch_conditioned: {
				monolithic_tiers: [],
				decomposed_probes: []
			},
			not_defined_reason: null
		};
		if (!monolithic || !decomposed) {
			return {
				...empty,
				not_defined_reason: 'paired workbench needs one monolithic run and one decomposed run in this paired_run_group_id'
			};
		}
		if (monolithic.status !== 'succeeded' || decomposed.status !== 'succeeded') {
			return {
				...empty,
				not_defined_reason: 'paired workbench compares only succeeded runs; one side is still running, cancelled, or failed'
			};
		}

		const qm = sqlQuote(monolithic.run_id);
		const qd = sqlQuote(decomposed.run_id);
		const comparableSideCte = (alias: string, runId: string) => `
			${alias}_integrity AS (
				SELECT stmt_hash,
				       evidence_hash,
				       SUM(CASE WHEN step_kind='aggregate' THEN 1 ELSE 0 END) AS n_aggregate,
				       SUM(CASE WHEN step_kind='aggregate'
				                  AND json_extract_string(output_json, '$.verdict') IS NOT NULL THEN 1 ELSE 0 END) AS n_verdict_aggregate,
				       SUM(CASE
				         WHEN step_kind='aggregate'
				              AND (
				                NULLIF(TRIM(COALESCE(error, '')), '') IS NOT NULL
				                OR json_extract(output_json, '$.error') IS NOT NULL
				              ) THEN 1
				         ELSE 0
				       END) AS n_aggregate_step_errors,
				       SUM(CASE
				         WHEN step_kind<>'aggregate'
				              AND (
				                NULLIF(TRIM(COALESCE(error, '')), '') IS NOT NULL
				                OR json_extract(output_json, '$.error') IS NOT NULL
				              ) THEN 1
				         ELSE 0
				       END) AS n_nonaggregate_step_errors
				FROM scorer_step
				WHERE run_id='${runId}' AND evidence_hash IS NOT NULL
				GROUP BY stmt_hash, evidence_hash
			),
			${alias}_aggregate_candidates AS (
				SELECT ss.stmt_hash, ss.evidence_hash,
				       CAST(json_extract(ss.output_json, '$.score') AS DOUBLE) AS score,
				       json_extract_string(ss.output_json, '$.verdict') AS verdict,
				       json_extract_string(ss.output_json, '$.confidence') AS confidence,
				       ss.latency_ms,
				       ss.prompt_tokens,
				       ss.out_tokens,
				       ROW_NUMBER() OVER (
				         PARTITION BY ss.stmt_hash, ss.evidence_hash
				         ORDER BY ss.started_at DESC, ss.step_hash DESC
				       ) AS rn
				FROM scorer_step ss
				JOIN ${alias}_integrity i USING (stmt_hash, evidence_hash)
				WHERE ss.run_id='${runId}'
				  AND ss.step_kind='aggregate'
				  AND ss.evidence_hash IS NOT NULL
				  AND i.n_aggregate_step_errors = 0
				  AND i.n_verdict_aggregate > 0
				  AND NULLIF(TRIM(COALESCE(ss.error, '')), '') IS NULL
				  AND json_extract(ss.output_json, '$.error') IS NULL
				  AND json_extract_string(ss.output_json, '$.verdict') IS NOT NULL
			),
			${alias} AS (
				SELECT stmt_hash, evidence_hash, score, verdict, confidence, latency_ms, prompt_tokens, out_tokens
				FROM ${alias}_aggregate_candidates
				WHERE rn = 1
			)`;
		const pairCte = `WITH
			${comparableSideCte('m', qm)},
			${comparableSideCte('d', qd)},
			j AS (
				SELECT
				   m.stmt_hash,
				   m.evidence_hash,
				   m.score AS monolithic_score,
				   d.score AS decomposed_score,
				   m.verdict AS monolithic_verdict,
				   d.verdict AS decomposed_verdict,
				   m.confidence AS monolithic_confidence,
				   d.confidence AS decomposed_confidence,
				   m.latency_ms AS monolithic_latency_ms,
				   d.latency_ms AS decomposed_latency_ms,
				   m.prompt_tokens AS monolithic_prompt_tokens,
				   d.prompt_tokens AS decomposed_prompt_tokens,
				   m.out_tokens AS monolithic_out_tokens,
				   d.out_tokens AS decomposed_out_tokens
				FROM m JOIN d USING (stmt_hash, evidence_hash)
			)`;
		const rawAggregateCte = (alias: string, runId: string) => `
			${alias}_raw_candidates AS (
				SELECT stmt_hash,
				       evidence_hash,
				       CAST(json_extract(output_json, '$.score') AS DOUBLE) AS score,
				       json_extract_string(output_json, '$.verdict') AS verdict,
				       ROW_NUMBER() OVER (
				         PARTITION BY stmt_hash, evidence_hash
				         ORDER BY started_at DESC, step_hash DESC
				       ) AS rn
				FROM scorer_step
				WHERE run_id='${runId}'
				  AND step_kind='aggregate'
				  AND evidence_hash IS NOT NULL
			),
			${alias}_raw AS (
				SELECT stmt_hash, evidence_hash, score, verdict
				FROM ${alias}_raw_candidates
				WHERE rn = 1
			)`;

		const overlapRows = await rows<PairedOverlapStats>(
			con,
			`WITH
			 ${comparableSideCte('m', qm)},
			 ${comparableSideCte('d', qd)},
			 o AS (
				SELECT m.stmt_hash, m.evidence_hash FROM m JOIN d USING (stmt_hash, evidence_hash)
			 )
			 SELECT
			   (SELECT COUNT(*) FROM m_integrity WHERE n_aggregate > 0) AS monolithic_evidences,
			   (SELECT COUNT(*) FROM d_integrity WHERE n_aggregate > 0) AS decomposed_evidences,
			   (SELECT COUNT(*) FROM m) AS monolithic_comparable_evidences,
			   (SELECT COUNT(*) FROM d) AS decomposed_comparable_evidences,
			   (SELECT COUNT(*) FROM o) AS overlap_evidences,
			   (SELECT COUNT(DISTINCT stmt_hash) FROM o) AS overlap_statements,
			   (SELECT COUNT(*) FROM m LEFT JOIN d USING (stmt_hash, evidence_hash) WHERE d.evidence_hash IS NULL) AS monolithic_only_evidences,
			   (SELECT COUNT(*) FROM d LEFT JOIN m USING (stmt_hash, evidence_hash) WHERE m.evidence_hash IS NULL) AS decomposed_only_evidences,
			   (SELECT COUNT(*) FROM m LEFT JOIN d_integrity di USING (stmt_hash, evidence_hash) WHERE di.evidence_hash IS NULL) AS monolithic_true_nonoverlap_evidences,
			   (SELECT COUNT(*) FROM d LEFT JOIN m_integrity mi USING (stmt_hash, evidence_hash) WHERE mi.evidence_hash IS NULL) AS decomposed_true_nonoverlap_evidences,
			   CASE WHEN (SELECT COUNT(*) FROM m) > 0
			        THEN (SELECT COUNT(*) FROM o)::DOUBLE / (SELECT COUNT(*) FROM m)
			        ELSE NULL END AS monolithic_overlap_pct,
			   CASE WHEN (SELECT COUNT(*) FROM d) > 0
			        THEN (SELECT COUNT(*) FROM o)::DOUBLE / (SELECT COUNT(*) FROM d)
			        ELSE NULL END AS decomposed_overlap_pct,
			   (SELECT COUNT(*) FROM m_integrity WHERE n_aggregate_step_errors > 0) AS monolithic_step_error_evidences,
			   (SELECT COUNT(*) FROM d_integrity WHERE n_aggregate_step_errors > 0) AS decomposed_step_error_evidences,
			   (SELECT COUNT(*) FROM m_integrity WHERE n_nonaggregate_step_errors > 0) AS monolithic_nonaggregate_step_error_evidences,
			   (SELECT COUNT(*) FROM d_integrity WHERE n_nonaggregate_step_errors > 0) AS decomposed_nonaggregate_step_error_evidences,
			   (SELECT COUNT(*) FROM m_integrity WHERE n_aggregate = 0) AS monolithic_missing_aggregate_evidences,
			   (SELECT COUNT(*) FROM d_integrity WHERE n_aggregate = 0) AS decomposed_missing_aggregate_evidences,
			   (SELECT COUNT(*) FROM m_integrity WHERE n_aggregate > 0 AND n_aggregate_step_errors = 0 AND n_verdict_aggregate = 0) AS monolithic_nonverdict_aggregate_evidences,
			   (SELECT COUNT(*) FROM d_integrity WHERE n_aggregate > 0 AND n_aggregate_step_errors = 0 AND n_verdict_aggregate = 0) AS decomposed_nonverdict_aggregate_evidences`
		);
		const toNumberOrNull = (v: unknown): number | null => v == null ? null : Number(v);
		const toNumber = (v: unknown): number => Number(v ?? 0);
		const overlapRaw = overlapRows[0] ?? null;
		const overlap: PairedOverlapStats | null = overlapRaw
			? {
				monolithic_evidences: toNumber(overlapRaw.monolithic_evidences),
				decomposed_evidences: toNumber(overlapRaw.decomposed_evidences),
				monolithic_comparable_evidences: toNumber(overlapRaw.monolithic_comparable_evidences),
				decomposed_comparable_evidences: toNumber(overlapRaw.decomposed_comparable_evidences),
				overlap_evidences: toNumber(overlapRaw.overlap_evidences),
				overlap_statements: toNumber(overlapRaw.overlap_statements),
				monolithic_only_evidences: toNumber(overlapRaw.monolithic_only_evidences),
				decomposed_only_evidences: toNumber(overlapRaw.decomposed_only_evidences),
				monolithic_true_nonoverlap_evidences: toNumber(overlapRaw.monolithic_true_nonoverlap_evidences),
				decomposed_true_nonoverlap_evidences: toNumber(overlapRaw.decomposed_true_nonoverlap_evidences),
				monolithic_overlap_pct: toNumberOrNull(overlapRaw.monolithic_overlap_pct),
				decomposed_overlap_pct: toNumberOrNull(overlapRaw.decomposed_overlap_pct),
				monolithic_step_error_evidences: toNumber(overlapRaw.monolithic_step_error_evidences),
				decomposed_step_error_evidences: toNumber(overlapRaw.decomposed_step_error_evidences),
				monolithic_nonaggregate_step_error_evidences: toNumber(overlapRaw.monolithic_nonaggregate_step_error_evidences),
				decomposed_nonaggregate_step_error_evidences: toNumber(overlapRaw.decomposed_nonaggregate_step_error_evidences),
				monolithic_missing_aggregate_evidences: toNumber(overlapRaw.monolithic_missing_aggregate_evidences),
				decomposed_missing_aggregate_evidences: toNumber(overlapRaw.decomposed_missing_aggregate_evidences),
				monolithic_nonverdict_aggregate_evidences: toNumber(overlapRaw.monolithic_nonverdict_aggregate_evidences),
				decomposed_nonverdict_aggregate_evidences: toNumber(overlapRaw.decomposed_nonverdict_aggregate_evidences)
			}
			: null;

		const comparableRows = await rows<PairedComparableMetrics>(
			con,
			`${pairCte}
			 SELECT
			   COUNT(*) AS n_overlap,
			   SUM(CASE WHEN s.indra_belief IS NOT NULL THEN 1 ELSE 0 END) AS n_truth_overlap,
			   SUM(CASE WHEN monolithic_verdict = decomposed_verdict THEN 1 ELSE 0 END) AS verdict_agreement_n,
			   CASE WHEN COUNT(*) > 0
			        THEN SUM(CASE WHEN monolithic_verdict = decomposed_verdict THEN 1 ELSE 0 END)::DOUBLE / COUNT(*)
			        ELSE NULL END AS verdict_agreement_rate,
			   SUM(CASE WHEN monolithic_verdict='correct' AND decomposed_verdict='correct' THEN 1 ELSE 0 END) AS both_correct_n,
			   SUM(CASE WHEN monolithic_verdict='correct' AND decomposed_verdict <> 'correct' THEN 1 ELSE 0 END) AS monolithic_only_correct_n,
			   SUM(CASE WHEN monolithic_verdict <> 'correct' AND decomposed_verdict='correct' THEN 1 ELSE 0 END) AS decomposed_only_correct_n,
			   SUM(CASE WHEN monolithic_verdict <> 'correct' AND decomposed_verdict <> 'correct' THEN 1 ELSE 0 END) AS both_incorrect_n,
			   AVG(monolithic_score) AS monolithic_score_mean,
			   AVG(decomposed_score) AS decomposed_score_mean,
			   AVG(CASE WHEN s.indra_belief IS NOT NULL AND monolithic_score IS NOT NULL
			            THEN ABS(monolithic_score - s.indra_belief) ELSE NULL END) AS monolithic_mae,
			   AVG(CASE WHEN s.indra_belief IS NOT NULL AND decomposed_score IS NOT NULL
			            THEN ABS(decomposed_score - s.indra_belief) ELSE NULL END) AS decomposed_mae,
			   AVG(CASE WHEN s.indra_belief IS NOT NULL AND monolithic_score IS NOT NULL
			            THEN monolithic_score - s.indra_belief ELSE NULL END) AS monolithic_bias,
			   AVG(CASE WHEN s.indra_belief IS NOT NULL AND decomposed_score IS NOT NULL
			            THEN decomposed_score - s.indra_belief ELSE NULL END) AS decomposed_bias,
			   AVG(CASE WHEN monolithic_score IS NOT NULL AND decomposed_score IS NOT NULL
			            THEN decomposed_score - monolithic_score ELSE NULL END) AS mean_score_delta,
			   AVG(monolithic_latency_ms) AS monolithic_latency_mean_ms,
			   AVG(decomposed_latency_ms) AS decomposed_latency_mean_ms,
			   SUM(CASE WHEN monolithic_latency_ms IS NOT NULL THEN 1 ELSE 0 END) AS monolithic_latency_observed_n,
			   SUM(CASE WHEN decomposed_latency_ms IS NOT NULL THEN 1 ELSE 0 END) AS decomposed_latency_observed_n,
			   SUM(COALESCE(monolithic_prompt_tokens, 0) + COALESCE(monolithic_out_tokens, 0)) AS monolithic_tokens_total,
			   SUM(COALESCE(decomposed_prompt_tokens, 0) + COALESCE(decomposed_out_tokens, 0)) AS decomposed_tokens_total,
			   SUM(CASE WHEN monolithic_prompt_tokens IS NOT NULL OR monolithic_out_tokens IS NOT NULL THEN 1 ELSE 0 END) AS monolithic_tokens_observed_n,
			   SUM(CASE WHEN decomposed_prompt_tokens IS NOT NULL OR decomposed_out_tokens IS NOT NULL THEN 1 ELSE 0 END) AS decomposed_tokens_observed_n
			 FROM j
			 LEFT JOIN statement s ON s.stmt_hash=j.stmt_hash`
		);
		const comparableRaw = comparableRows[0] ?? null;
		const comparableOverlap = toNumber(comparableRaw?.n_overlap);
		const verdictPairRows = comparableOverlap > 0
			? (await rows<PairedVerdictPairRow>(
				con,
				`${pairCte}
				 SELECT
				   monolithic_verdict,
				   decomposed_verdict,
				   CASE
				     WHEN monolithic_verdict='correct' AND decomposed_verdict='correct' THEN 'both_supported'
				     WHEN monolithic_verdict='correct' AND decomposed_verdict <> 'correct' THEN 'monolithic_only'
				     WHEN monolithic_verdict <> 'correct' AND decomposed_verdict='correct' THEN 'decomposed_only'
				     ELSE 'neither_supported'
				   END AS support_cell,
				   COUNT(*) AS n
				 FROM j
				 GROUP BY monolithic_verdict, decomposed_verdict, support_cell
				 ORDER BY
				   CASE
				     WHEN monolithic_verdict='correct' AND decomposed_verdict='correct' THEN 0
				     WHEN monolithic_verdict='correct' AND decomposed_verdict <> 'correct' THEN 1
				     WHEN monolithic_verdict <> 'correct' AND decomposed_verdict='correct' THEN 2
				     ELSE 3
				   END,
				   n DESC,
				   monolithic_verdict,
				   decomposed_verdict`
			)).map((r) => ({
				...r,
				n: toNumber(r.n)
			}))
			: [];
		const comparable: PairedComparableMetrics | null = comparableRaw && comparableOverlap > 0
			? {
				n_overlap: comparableOverlap,
				n_truth_overlap: toNumber(comparableRaw.n_truth_overlap),
				verdict_agreement_n: toNumber(comparableRaw.verdict_agreement_n),
				verdict_agreement_rate: toNumberOrNull(comparableRaw.verdict_agreement_rate),
				both_correct_n: toNumber(comparableRaw.both_correct_n),
				monolithic_only_correct_n: toNumber(comparableRaw.monolithic_only_correct_n),
				decomposed_only_correct_n: toNumber(comparableRaw.decomposed_only_correct_n),
				both_incorrect_n: toNumber(comparableRaw.both_incorrect_n),
				verdict_label_pairs: verdictPairRows,
				monolithic_score_mean: toNumberOrNull(comparableRaw.monolithic_score_mean),
				decomposed_score_mean: toNumberOrNull(comparableRaw.decomposed_score_mean),
				monolithic_mae: toNumberOrNull(comparableRaw.monolithic_mae),
				decomposed_mae: toNumberOrNull(comparableRaw.decomposed_mae),
				monolithic_bias: toNumberOrNull(comparableRaw.monolithic_bias),
				decomposed_bias: toNumberOrNull(comparableRaw.decomposed_bias),
				mean_score_delta: toNumberOrNull(comparableRaw.mean_score_delta),
				monolithic_latency_mean_ms: toNumberOrNull(comparableRaw.monolithic_latency_mean_ms),
				decomposed_latency_mean_ms: toNumberOrNull(comparableRaw.decomposed_latency_mean_ms),
				monolithic_latency_observed_n: toNumber(comparableRaw.monolithic_latency_observed_n),
				decomposed_latency_observed_n: toNumber(comparableRaw.decomposed_latency_observed_n),
				monolithic_tokens_total: toNumber(comparableRaw.monolithic_tokens_total),
				decomposed_tokens_total: toNumber(comparableRaw.decomposed_tokens_total),
				monolithic_tokens_observed_n: toNumber(comparableRaw.monolithic_tokens_observed_n),
				decomposed_tokens_observed_n: toNumber(comparableRaw.decomposed_tokens_observed_n)
			}
			: null;
		const resourceFrontierArch = (
			architecture: 'monolithic' | 'decomposed',
			run: PairRunSummary,
			latencyMeanMs: number | null | undefined,
			latencyObservedN: number | null | undefined,
			tokensTotal: number | null | undefined,
			tokensObservedN: number | null | undefined,
			mae: number | null | undefined
		): PairedResourceFrontierArch => {
			const actualCost = run.cost_actual_usd == null ? null : Number(run.cost_actual_usd);
			const estimateCost = run.cost_estimate_usd == null ? null : Number(run.cost_estimate_usd);
			const runCost = actualCost ?? estimateCost;
			return {
				architecture,
				run_id: run.run_id,
				run_cost_usd: runCost,
				run_cost_basis: actualCost != null ? 'actual' : estimateCost != null ? 'estimate' : 'missing',
				n_evidences: run.n_evidences,
				cost_per_evidence_usd: safeRate(runCost, run.n_evidences),
				duration_s: run.duration_s,
				wall_seconds_per_evidence: safeRate(run.duration_s, run.n_evidences),
				clean_overlap_n: comparable?.n_overlap ?? 0,
				clean_overlap_latency_mean_ms: latencyMeanMs == null ? null : Number(latencyMeanMs),
				clean_overlap_latency_observed_n: Number(latencyObservedN ?? 0),
				clean_overlap_tokens_total: Number(tokensTotal ?? 0),
				clean_overlap_tokens_observed_n: Number(tokensObservedN ?? 0),
				clean_overlap_tokens_per_observed_evidence: safeRate(Number(tokensTotal ?? 0), Number(tokensObservedN ?? 0)),
				truth_overlap_n: comparable?.n_truth_overlap ?? 0,
				mae: mae == null ? null : Number(mae)
			};
		};
		const resource_frontier: PairedResourceFrontier | null = comparable
			? {
				monolithic: resourceFrontierArch(
					'monolithic',
					monolithic,
					comparable.monolithic_latency_mean_ms,
					comparable.monolithic_latency_observed_n,
					comparable.monolithic_tokens_total,
					comparable.monolithic_tokens_observed_n,
					comparable.monolithic_mae
				),
				decomposed: resourceFrontierArch(
					'decomposed',
					decomposed,
					comparable.decomposed_latency_mean_ms,
					comparable.decomposed_latency_observed_n,
					comparable.decomposed_tokens_total,
					comparable.decomposed_tokens_observed_n,
					comparable.decomposed_mae
				),
				spend_scope: 'whole-run spend; denominator is each side aggregate evidence count, not clean overlap',
				latency_scope: 'clean shared aggregate verdict evidence with per-row telemetry counts shown',
				quality_scope: comparable.n_truth_overlap > 0
					? 'MAE over truth-anchored clean overlap'
					: 'quality not defined: clean overlap has no INDRA belief anchors',
				not_defined_reason: null
			}
			: null;

		const exampleSelect = (whereExtra: string, orderBy: string) => `${pairCte},
			ags AS (
				SELECT stmt_hash,
				       string_agg(name, ', ' ORDER BY
				         CASE role
				           WHEN 'subj' THEN 0 WHEN 'enz' THEN 0
				           WHEN 'obj' THEN 1 WHEN 'sub' THEN 1
				           WHEN 'member' THEN 2 ELSE 3 END,
				         role_index) AS agent_names
				FROM agent
				GROUP BY stmt_hash
			),
			base AS (
				SELECT
				   j.stmt_hash,
				   j.evidence_hash,
				   s.indra_type,
				   COALESCE(ags.agent_names, '') AS agent_names,
				   e.text,
				   e.source_api,
				   e.pmid,
				   s.indra_belief,
				   j.monolithic_score,
				   j.decomposed_score,
				   j.monolithic_verdict,
				   j.decomposed_verdict,
				   CASE WHEN s.indra_belief IS NOT NULL AND j.monolithic_score IS NOT NULL
				        THEN ABS(j.monolithic_score - s.indra_belief) ELSE NULL END AS monolithic_error,
				   CASE WHEN s.indra_belief IS NOT NULL AND j.decomposed_score IS NOT NULL
				        THEN ABS(j.decomposed_score - s.indra_belief) ELSE NULL END AS decomposed_error,
				   CASE WHEN s.indra_belief IS NOT NULL AND j.monolithic_score IS NOT NULL AND j.decomposed_score IS NOT NULL
				        THEN ABS(j.decomposed_score - s.indra_belief) - ABS(j.monolithic_score - s.indra_belief)
				        ELSE NULL END AS abs_error_delta,
				   '/statements/' || j.stmt_hash || '?run_id=${qm}' AS monolithic_href,
				   '/statements/' || j.stmt_hash || '?run_id=${qd}' AS decomposed_href,
				   NULL::VARCHAR AS excluded_side,
				   NULL::VARCHAR AS excluded_reason
				 FROM j
				 JOIN statement s ON s.stmt_hash=j.stmt_hash
				 LEFT JOIN evidence e ON e.evidence_hash=j.evidence_hash
				 LEFT JOIN ags ON ags.stmt_hash=j.stmt_hash
			)
			SELECT * FROM base
			 WHERE ${whereExtra}
			 ORDER BY ${orderBy}, stmt_hash
			 LIMIT 8`;

		const nonOverlapSelect = (side: 'monolithic' | 'decomposed') => {
			const currentRun = side === 'monolithic' ? qm : qd;
			const otherRun = side === 'monolithic' ? qd : qm;
			const scoreColumns = side === 'monolithic'
				? `br.score AS monolithic_score,
				   NULL::DOUBLE AS decomposed_score,
				   br.verdict AS monolithic_verdict,
				   NULL::VARCHAR AS decomposed_verdict,
				   CASE WHEN s.indra_belief IS NOT NULL AND br.score IS NOT NULL
				        THEN ABS(br.score - s.indra_belief) ELSE NULL END AS monolithic_error,
				   NULL::DOUBLE AS decomposed_error`
				: `NULL::DOUBLE AS monolithic_score,
				   br.score AS decomposed_score,
				   NULL::VARCHAR AS monolithic_verdict,
				   br.verdict AS decomposed_verdict,
				   NULL::DOUBLE AS monolithic_error,
				   CASE WHEN s.indra_belief IS NOT NULL AND br.score IS NOT NULL
				        THEN ABS(br.score - s.indra_belief) ELSE NULL END AS decomposed_error`;
			return `WITH
				${comparableSideCte('br', currentRun)},
				${comparableSideCte('other', otherRun)},
				ags AS (
					SELECT stmt_hash,
					       string_agg(name, ', ' ORDER BY
					         CASE role
					           WHEN 'subj' THEN 0 WHEN 'enz' THEN 0
					           WHEN 'obj' THEN 1 WHEN 'sub' THEN 1
					           WHEN 'member' THEN 2 ELSE 3 END,
					         role_index) AS agent_names
					FROM agent
					GROUP BY stmt_hash
				),
				base AS (
					SELECT
					   br.stmt_hash,
					   br.evidence_hash,
					   s.indra_type,
					   COALESCE(ags.agent_names, '') AS agent_names,
					   e.text,
					   e.source_api,
					   e.pmid,
					   s.indra_belief,
					   ${scoreColumns},
					   NULL::DOUBLE AS abs_error_delta,
					   '/statements/' || br.stmt_hash || '?run_id=${qm}' AS monolithic_href,
					   '/statements/' || br.stmt_hash || '?run_id=${qd}' AS decomposed_href,
					   NULL::VARCHAR AS excluded_side,
					   NULL::VARCHAR AS excluded_reason
					FROM br
					JOIN statement s ON s.stmt_hash=br.stmt_hash
					LEFT JOIN evidence e ON e.evidence_hash=br.evidence_hash
					LEFT JOIN ags ON ags.stmt_hash=br.stmt_hash
					LEFT JOIN other_integrity oi
					  ON oi.stmt_hash=br.stmt_hash
					 AND oi.evidence_hash=br.evidence_hash
					WHERE oi.evidence_hash IS NULL
				)
				SELECT * FROM base
				ORDER BY COALESCE(monolithic_error, decomposed_error, 0) DESC, stmt_hash
				LIMIT 8`;
		};
		const excludedByIntegritySelect = `WITH
			${comparableSideCte('m', qm)},
			${comparableSideCte('d', qd)},
			${rawAggregateCte('mr', qm)},
			${rawAggregateCte('dr', qd)},
			keys AS (
				SELECT stmt_hash, evidence_hash FROM m_integrity
				UNION
				SELECT stmt_hash, evidence_hash FROM d_integrity
			),
			ags AS (
				SELECT stmt_hash,
				       string_agg(name, ', ' ORDER BY
				         CASE role
				           WHEN 'subj' THEN 0 WHEN 'enz' THEN 0
				           WHEN 'obj' THEN 1 WHEN 'sub' THEN 1
				           WHEN 'member' THEN 2 ELSE 3 END,
				         role_index) AS agent_names
				FROM agent
				GROUP BY stmt_hash
			),
			classified AS (
				SELECT
				   k.stmt_hash,
				   k.evidence_hash,
				   COALESCE(s.indra_type, 'unknown') AS indra_type,
				   COALESCE(ags.agent_names, '') AS agent_names,
				   e.text,
				   e.source_api,
				   e.pmid,
				   s.indra_belief,
				   mr.score AS monolithic_score,
				   dr.score AS decomposed_score,
				   mr.verdict AS monolithic_verdict,
				   dr.verdict AS decomposed_verdict,
				   CASE WHEN s.indra_belief IS NOT NULL AND mr.score IS NOT NULL
				        THEN ABS(mr.score - s.indra_belief) ELSE NULL END AS monolithic_error,
				   CASE WHEN s.indra_belief IS NOT NULL AND dr.score IS NOT NULL
				        THEN ABS(dr.score - s.indra_belief) ELSE NULL END AS decomposed_error,
				   CASE WHEN s.indra_belief IS NOT NULL AND mr.score IS NOT NULL AND dr.score IS NOT NULL
				        THEN ABS(dr.score - s.indra_belief) - ABS(mr.score - s.indra_belief)
				        ELSE NULL END AS abs_error_delta,
				   '/statements/' || k.stmt_hash || '?run_id=${qm}' AS monolithic_href,
				   '/statements/' || k.stmt_hash || '?run_id=${qd}' AS decomposed_href,
				   CASE
				     WHEN mi.n_aggregate_step_errors > 0 THEN '[M] aggregate step error'
				     WHEN mi.evidence_hash IS NOT NULL AND mi.n_aggregate = 0 THEN '[M] missing aggregate'
				     WHEN mi.n_aggregate > 0 AND mi.n_aggregate_step_errors = 0 AND mi.n_verdict_aggregate = 0 THEN '[M] aggregate lacks verdict'
				     ELSE NULL
				   END AS monolithic_reason,
				   CASE
				     WHEN di.n_aggregate_step_errors > 0 THEN '[D] aggregate step error'
				     WHEN di.evidence_hash IS NOT NULL AND di.n_aggregate = 0 THEN '[D] missing aggregate'
				     WHEN di.n_aggregate > 0 AND di.n_aggregate_step_errors = 0 AND di.n_verdict_aggregate = 0 THEN '[D] aggregate lacks verdict'
				     ELSE NULL
				   END AS decomposed_reason
				FROM keys k
				LEFT JOIN m_integrity mi USING (stmt_hash, evidence_hash)
				LEFT JOIN d_integrity di USING (stmt_hash, evidence_hash)
				LEFT JOIN mr_raw mr USING (stmt_hash, evidence_hash)
				LEFT JOIN dr_raw dr USING (stmt_hash, evidence_hash)
				LEFT JOIN statement s ON s.stmt_hash=k.stmt_hash
				LEFT JOIN evidence e ON e.evidence_hash=k.evidence_hash
				LEFT JOIN ags ON ags.stmt_hash=k.stmt_hash
			)
			SELECT
			   stmt_hash,
			   evidence_hash,
			   indra_type,
			   agent_names,
			   text,
			   source_api,
			   pmid,
			   indra_belief,
			   monolithic_score,
			   decomposed_score,
			   monolithic_verdict,
			   decomposed_verdict,
			   monolithic_error,
			   decomposed_error,
			   abs_error_delta,
			   monolithic_href,
			   decomposed_href,
			   CASE
			     WHEN monolithic_reason IS NOT NULL AND decomposed_reason IS NOT NULL THEN '[M]/[D]'
			     WHEN monolithic_reason IS NOT NULL THEN '[M]'
			     ELSE '[D]'
			   END AS excluded_side,
			   CASE
			     WHEN monolithic_reason IS NOT NULL AND decomposed_reason IS NOT NULL THEN monolithic_reason || '; ' || decomposed_reason
			     ELSE COALESCE(monolithic_reason, decomposed_reason)
			   END AS excluded_reason
			FROM classified
			WHERE monolithic_reason IS NOT NULL OR decomposed_reason IS NOT NULL
			ORDER BY
			  CASE
			    WHEN COALESCE(monolithic_reason, '') LIKE '%step error%' OR COALESCE(decomposed_reason, '') LIKE '%step error%' THEN 0
			    WHEN COALESCE(monolithic_reason, '') LIKE '%missing aggregate%' OR COALESCE(decomposed_reason, '') LIKE '%missing aggregate%' THEN 1
			    ELSE 2
			  END,
			  stmt_hash
			LIMIT 8`;

		const [
			monolithicWins,
			decomposedWins,
			verdictDisagreements,
			mutualFailures,
			monolithicOnly,
			decomposedOnly,
			excludedByIntegrity,
			monoTiers,
			decomposedProbes
		] = await Promise.all([
			rows<PairedExampleRow>(
				con,
				exampleSelect(
					`abs_error_delta IS NOT NULL AND abs_error_delta > 0`,
					`abs_error_delta DESC`
				)
			),
			rows<PairedExampleRow>(
				con,
				exampleSelect(
					`abs_error_delta IS NOT NULL AND abs_error_delta < 0`,
					`abs_error_delta ASC`
				)
			),
			rows<PairedExampleRow>(
				con,
				exampleSelect(
					`COALESCE(monolithic_verdict, '') <> COALESCE(decomposed_verdict, '')`,
					`ABS(COALESCE(abs_error_delta, 0)) DESC`
				)
			),
			rows<PairedExampleRow>(
				con,
				exampleSelect(
					`monolithic_error >= 0.25 AND decomposed_error >= 0.25`,
					`(monolithic_error + decomposed_error) DESC`
				)
			),
			rows<PairedExampleRow>(con, nonOverlapSelect('monolithic')),
			rows<PairedExampleRow>(con, nonOverlapSelect('decomposed')),
			rows<PairedExampleRow>(con, excludedByIntegritySelect),
			rows<PairedArchTierRow>(
				con,
				`SELECT
				   COALESCE(json_extract_string(output_json, '$.tier'), 'unknown') AS tier,
				   COUNT(*) AS n,
				   AVG(CAST(json_extract(output_json, '$.score') AS DOUBLE)) AS mean_score
				 FROM scorer_step
				 WHERE run_id='${qm}' AND step_kind='aggregate'
				 GROUP BY tier
				 ORDER BY n DESC, tier`
			),
			rows<PairedArchProbeRow>(
				con,
				`SELECT
				   step_kind AS name,
				   COUNT(*) AS n,
				   SUM(CASE WHEN is_substrate_answered THEN 1 ELSE 0 END) AS substrate_n,
				   SUM(CASE WHEN error IS NOT NULL THEN 1 ELSE 0 END) AS error_n
				 FROM scorer_step
				 WHERE run_id='${qd}'
				   AND step_kind IN ('subject_role_probe','object_role_probe','relation_axis_probe','scope_probe','substrate_route')
				 GROUP BY step_kind
				 ORDER BY CASE step_kind
				   WHEN 'substrate_route' THEN 0
				   WHEN 'subject_role_probe' THEN 1
				   WHEN 'object_role_probe' THEN 2
				   WHEN 'relation_axis_probe' THEN 3
				   WHEN 'scope_probe' THEN 4
				   ELSE 5 END`
			)
		]);

		const withNumericExamples = (rs: PairedExampleRow[]) => rs.map((r) => ({
			...r,
			indra_belief: r.indra_belief == null ? null : Number(r.indra_belief),
			monolithic_score: r.monolithic_score == null ? null : Number(r.monolithic_score),
			decomposed_score: r.decomposed_score == null ? null : Number(r.decomposed_score),
			monolithic_error: r.monolithic_error == null ? null : Number(r.monolithic_error),
			decomposed_error: r.decomposed_error == null ? null : Number(r.decomposed_error),
			abs_error_delta: r.abs_error_delta == null ? null : Number(r.abs_error_delta)
		}));
		const numericTiers = monoTiers.map((r) => ({
			...r,
			n: toNumber(r.n),
			mean_score: toNumberOrNull(r.mean_score)
		}));
		const numericProbes = decomposedProbes.map((r) => ({
			...r,
			n: toNumber(r.n),
			substrate_n: toNumber(r.substrate_n),
			error_n: toNumber(r.error_n)
		}));
		const notDefinedReason = !overlap
			? 'paired workbench needs overlap accounting before comparable metrics are defined'
			: overlap.overlap_evidences === 0
				? 'comparison not defined: monolithic and decomposed runs share zero non-error aggregate verdict evidence rows'
				: comparable
					? null
					: 'comparison not defined: paired aggregate rows exist, but no comparable metric denominator could be formed';
		const monolithicIntegrityExclusions = overlap
			? overlap.monolithic_step_error_evidences +
				overlap.monolithic_missing_aggregate_evidences +
				overlap.monolithic_nonverdict_aggregate_evidences
			: 0;
		const decomposedIntegrityExclusions = overlap
			? overlap.decomposed_step_error_evidences +
				overlap.decomposed_missing_aggregate_evidences +
				overlap.decomposed_nonverdict_aggregate_evidences
			: 0;
		const integrityExclusions = monolithicIntegrityExclusions + decomposedIntegrityExclusions;
		const traceWarnings = overlap
			? overlap.monolithic_nonaggregate_step_error_evidences +
				overlap.decomposed_nonaggregate_step_error_evidences
			: 0;
		const probeRows = numericProbes.reduce((sum, r) => sum + r.n, 0);
		const tierRows = numericTiers.reduce((sum, r) => sum + r.n, 0);
		const denominatorLedger: PairedDenominatorRow[] = overlap
			? [
					{
						key: 'side_aggregate_evidence_rows',
						panel: 'overlap accounting',
						applicability: 'paired_only',
						metric_kind: 'denominator_base',
						ledger_role: 'root',
						parent_key: null,
						unit: 'side aggregate evidence rows',
						denominator_n: null,
						monolithic_n: overlap.monolithic_evidences,
						decomposed_n: overlap.decomposed_evidences,
						overlap_n: overlap.overlap_evidences,
						excluded_n: integrityExclusions,
						reason: 'Starts from persisted aggregate rows on each side before any paired metric is allowed to share a denominator.'
					},
					{
						key: 'clean_shared_aggregate_verdict_evidence',
						panel: 'clean shared paired base',
						applicability: comparable ? 'paired_only' : 'not_defined',
						metric_kind: 'denominator_base',
						ledger_role: 'base',
						parent_key: 'side_aggregate_evidence_rows',
						unit: 'clean shared aggregate verdict evidence',
						denominator_n: comparable?.n_overlap ?? null,
						monolithic_n: comparable?.n_overlap ?? null,
						decomposed_n: comparable?.n_overlap ?? null,
						overlap_n: overlap.overlap_evidences,
						excluded_n: integrityExclusions,
						reason: comparable
							? 'Only evidence with non-error aggregate verdict rows on both architectures enters any clean shared paired metric.'
							: (notDefinedReason ?? 'comparison not defined for the paired clean-overlap denominator')
					},
					{
						key: 'exact_label_agreement_metric',
						panel: '[M] vs [D] exact verdict label match',
						applicability: comparable ? 'paired_only' : 'not_defined',
						metric_kind: 'arch_arch_exact_label',
						ledger_role: 'metric',
						parent_key: 'clean_shared_aggregate_verdict_evidence',
						unit: 'clean shared aggregate verdict evidence',
						denominator_n: comparable?.n_overlap ?? null,
						monolithic_n: comparable?.n_overlap ?? null,
						decomposed_n: comparable?.n_overlap ?? null,
						overlap_n: overlap.overlap_evidences,
						excluded_n: integrityExclusions,
						reason: comparable
							? 'Compares exact aggregate verdict labels between architectures. It is inter-architecture agreement, not accuracy.'
							: (notDefinedReason ?? 'exact label agreement needs clean shared aggregate verdict rows')
					},
					{
						key: 'support_state_recode_metric',
						panel: 'support-state matrix and exact-label taxonomy',
						applicability: comparable ? 'paired_only' : 'not_defined',
						metric_kind: 'arch_arch_support_recode',
						ledger_role: 'metric',
						parent_key: 'clean_shared_aggregate_verdict_evidence',
						unit: 'clean shared aggregate verdict evidence',
						denominator_n: comparable?.n_overlap ?? null,
						monolithic_n: comparable?.n_overlap ?? null,
						decomposed_n: comparable?.n_overlap ?? null,
						overlap_n: overlap.overlap_evidences,
						excluded_n: integrityExclusions,
						reason: comparable
							? 'Recodes exact labels into supported/not-supported buckets, then shows the exact labels inside each bucket.'
							: (notDefinedReason ?? 'support-state recode needs clean shared aggregate verdict rows')
					},
					{
						key: 'score_posture_metric',
						panel: 'mean score movement',
						applicability: comparable ? 'paired_only' : 'not_defined',
						metric_kind: 'arch_arch_score_posture',
						ledger_role: 'metric',
						parent_key: 'clean_shared_aggregate_verdict_evidence',
						unit: 'scores over clean shared aggregate verdict evidence',
						denominator_n: comparable?.n_overlap ?? null,
						monolithic_n: comparable?.n_overlap ?? null,
						decomposed_n: comparable?.n_overlap ?? null,
						overlap_n: overlap.overlap_evidences,
						excluded_n: integrityExclusions,
						reason: comparable
							? 'Compares mean score posture between architectures. It is not a winner and not an INDRA residual.'
							: (notDefinedReason ?? 'score posture needs clean shared aggregate verdict rows')
					},
					{
						key: 'resource_counter_metric',
						panel: 'latency and token counters',
						applicability: comparable ? 'paired_only' : 'not_defined',
						metric_kind: 'arch_arch_resource',
						ledger_role: 'metric',
						parent_key: 'clean_shared_aggregate_verdict_evidence',
						unit: 'clean shared aggregate verdict evidence',
						denominator_n: comparable?.n_overlap ?? null,
						monolithic_n: comparable?.n_overlap ?? null,
						decomposed_n: comparable?.n_overlap ?? null,
						overlap_n: overlap.overlap_evidences,
						excluded_n: integrityExclusions,
						reason: comparable
							? 'Compares resource counters over the clean shared denominator; resource direction is not residual quality.'
							: (notDefinedReason ?? 'resource counters need clean shared aggregate verdict rows')
					},
					{
						key: 'truth_anchored_overlap_evidence',
						panel: 'INDRA MAE and bias',
						applicability: comparable && comparable.n_truth_overlap > 0 ? 'paired_only' : 'not_defined',
						metric_kind: 'arch_indra_residual',
						ledger_role: 'metric',
						parent_key: 'clean_shared_aggregate_verdict_evidence',
						unit: 'truth-anchored overlap evidence',
						denominator_n: comparable?.n_truth_overlap ?? null,
						monolithic_n: comparable?.n_truth_overlap ?? null,
						decomposed_n: comparable?.n_truth_overlap ?? null,
						overlap_n: comparable?.n_overlap ?? overlap.overlap_evidences,
						excluded_n: comparable ? comparable.n_overlap - comparable.n_truth_overlap : null,
						reason: comparable && comparable.n_truth_overlap > 0
							? 'Residual metrics compare each architecture to INDRA belief, not to the other architecture.'
							: 'Residual metrics need clean overlap rows with INDRA belief anchors.'
					},
					{
						key: 'side_evidence_integrity_exclusions',
						panel: 'integrity exclusions',
						applicability: 'paired_only',
						metric_kind: 'integrity_gate',
						ledger_role: 'gate',
						parent_key: 'side_aggregate_evidence_rows',
						unit: 'side-evidence exclusions',
						denominator_n: integrityExclusions,
						monolithic_n: monolithicIntegrityExclusions,
						decomposed_n: decomposedIntegrityExclusions,
						overlap_n: null,
						excluded_n: integrityExclusions,
						reason: 'Aggregate step errors, missing aggregates, and aggregate rows without verdicts are withheld from comparable metrics but remain inspectable.'
					},
					{
						key: 'clean_true_nonoverlap_evidence',
						panel: 'true non-overlap exemplars',
						applicability: 'paired_only',
						metric_kind: 'paired_nonoverlap',
						ledger_role: 'gate',
						parent_key: 'side_aggregate_evidence_rows',
						unit: 'clean evidence rows',
						denominator_n: overlap.monolithic_true_nonoverlap_evidences + overlap.decomposed_true_nonoverlap_evidences,
						monolithic_n: overlap.monolithic_true_nonoverlap_evidences,
						decomposed_n: overlap.decomposed_true_nonoverlap_evidences,
						overlap_n: 0,
						excluded_n: overlap.monolithic_true_nonoverlap_evidences + overlap.decomposed_true_nonoverlap_evidences,
						reason: 'These lanes only include clean rows where the other architecture has no persisted rows for the evidence, so they stay outside paired deltas.'
					},
					{
						key: 'monolithic_aggregate_rows',
						panel: 'native monolithic tier path',
						applicability: 'arch_conditioned',
						metric_kind: 'architecture_native',
						ledger_role: 'native',
						parent_key: null,
						unit: 'monolithic aggregate rows',
						denominator_n: tierRows,
						monolithic_n: tierRows,
						decomposed_n: null,
						overlap_n: null,
						excluded_n: monolithicIntegrityExclusions,
						reason: 'Tier diagnostics are native to monolithic aggregate rows and are not converted into decomposed probe grammar.'
					},
					{
						key: 'decomposed_native_step_rows',
						panel: 'native decomposed probe health',
						applicability: 'arch_conditioned',
						metric_kind: 'architecture_native',
						ledger_role: 'native',
						parent_key: null,
						unit: 'decomposed native step rows',
						denominator_n: probeRows,
						monolithic_n: null,
						decomposed_n: probeRows,
						overlap_n: null,
						excluded_n: overlap.decomposed_nonaggregate_step_error_evidences,
						reason: 'Probe health is measured over decomposed native step rows, not over the paired clean-overlap denominator.'
					},
					{
						key: 'nonaggregate_step_error_evidence',
						panel: 'non-aggregate trace warnings',
						applicability: traceWarnings > 0 ? 'arch_conditioned' : 'not_defined',
						metric_kind: 'architecture_native_trace_health',
						ledger_role: 'native',
						parent_key: null,
						unit: 'evidence with non-aggregate step errors',
						denominator_n: traceWarnings,
						monolithic_n: overlap.monolithic_nonaggregate_step_error_evidences,
						decomposed_n: overlap.decomposed_nonaggregate_step_error_evidences,
						overlap_n: null,
						excluded_n: 0,
						reason: traceWarnings > 0
							? 'These warnings describe native trace health outside the aggregate comparable gate.'
							: 'No non-aggregate step errors are persisted for either architecture in this pair.'
				}
			]
			: [];

		return {
			pair_id,
			runs,
			monolithic,
			decomposed,
			overlap,
			comparable,
			resource_frontier,
			denominator_ledger: denominatorLedger,
			exemplars: {
				monolithic_wins: withNumericExamples(monolithicWins),
				decomposed_wins: withNumericExamples(decomposedWins),
				verdict_disagreements: withNumericExamples(verdictDisagreements),
				mutual_failures: withNumericExamples(mutualFailures),
				monolithic_only: withNumericExamples(monolithicOnly),
				decomposed_only: withNumericExamples(decomposedOnly),
				excluded_by_integrity: withNumericExamples(excludedByIntegrity)
			},
			arch_conditioned: {
				monolithic_tiers: numericTiers,
				decomposed_probes: numericProbes
			},
			not_defined_reason: notDefinedReason
		};
	} finally {
		con.disconnectSync?.();
	}
}
