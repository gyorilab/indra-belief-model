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

import { closeInstance, cohortHref, connect, dbExists, dbPath, durationSeconds, normalizeDuckValue, readRows, repairRerunLineageSqlOptions, rows, safeRate, scalar, sqlQuote, tableExists, timestampMs } from '$lib/db';
import type { ConfidenceCalibrationRow, LatestValidity, StratumRow, TruthPresentRow } from '$lib/db';
import type { PanelApplicabilityKind } from '$lib/pairedMetricKinds';
import type { DuckDBConnection } from '@duckdb/node-api';

export async function getLatestValidity(con: DuckDBConnection): Promise<LatestValidity | null> {
	const runs = await readRows<{
		run_id: string;
		scorer_version: string;
		architecture: string | null;
		paired_run_group_id: string | null;
	}>(
		con,
		`SELECT run_id, scorer_version, architecture, paired_run_group_id FROM score_run
		 WHERE status = 'succeeded'
		 ORDER BY started_at DESC LIMIT 1`
	);
	if (runs.length === 0) return null;
	const { run_id, scorer_version } = runs[0];
	const architecture = runs[0].architecture ?? 'unknown';
	const pairedGroup = runs[0].paired_run_group_id ?? null;

	type TruthPresentRaw = Omit<TruthPresentRow, 'cohort_href' | 'truth_href' | 'applicability' | 'gold_fields'> & {
		gold_fields_json: string | null;
	};
	const latestValidityReads = [
		readRows<{ verdict: string; n: number }>(
			con,
			`SELECT replace(replace(metric_name, 'verdict_share.', ''), '"', '') AS verdict,
			        CAST(json_extract(slice_json, '$.n') AS BIGINT) AS n
			 FROM metric
			 WHERE run_id = '${run_id.replace(/'/g, "''")}' AND metric_name LIKE 'verdict_share.%'
			 ORDER BY n DESC`
		),
		readRows<{ value: number }>(
			con,
			`SELECT value FROM metric
			 WHERE run_id = '${run_id.replace(/'/g, "''")}'
			   AND metric_name = 'indra_belief_calibration.mae' AND truth_set_id = 'indra_published_belief' LIMIT 1`
		),
		readRows<{ value: number }>(
			con,
			`SELECT value FROM metric
			 WHERE run_id = '${run_id.replace(/'/g, "''")}'
			   AND metric_name = 'indra_belief_calibration.rmse' AND truth_set_id = 'indra_published_belief' LIMIT 1`
		),
		readRows<{ value: number }>(
			con,
			`SELECT value FROM metric
			 WHERE run_id = '${run_id.replace(/'/g, "''")}'
			   AND metric_name = 'indra_belief_calibration.bias' AND truth_set_id = 'indra_published_belief' LIMIT 1`
		),
		readRows<{ n: number }>(
			con,
			`SELECT CAST(json_extract(slice_json, '$.n_stmts') AS BIGINT) AS n
			 FROM metric
			 WHERE run_id = '${run_id.replace(/'/g, "''")}'
			   AND metric_name = 'indra_belief_calibration.mae' AND truth_set_id = 'indra_published_belief' LIMIT 1`
		),
		readRows<{ value: number; n: number }>(
			con,
			`SELECT value, CAST(json_extract(slice_json, '$.n_multi_evidence_stmts') AS BIGINT) AS n
			 FROM metric
			 WHERE run_id = '${run_id.replace(/'/g, "''")}'
			   AND metric_name = 'inter_evidence_consistency.mean_stdev' LIMIT 1`
		),
		readRows<{ value: number; unavailable_reason: string | null }>(
			con,
			`SELECT value, json_extract_string(slice_json, '$.unavailable_reason') AS unavailable_reason
			 FROM metric
			 WHERE run_id = '${run_id.replace(/'/g, "''")}'
			   AND metric_name = 'supports_graph_plausibility.delta' LIMIT 1`
		),
		readRows<Omit<StratumRow, 'cohort_href' | 'applicability'>>(
			con,
			`WITH mae_rows AS (
				SELECT json_extract(slice_json, '$.value')::VARCHAR AS value,
				       CAST(json_extract(slice_json, '$.n') AS BIGINT) AS n,
				       value AS mae
				FROM metric
				WHERE run_id = '${run_id.replace(/'/g, "''")}'
				  AND metric_name = 'indra_belief_calibration_by_type.mae' AND truth_set_id = 'indra_published_belief'
			),
			bias_rows AS (
				SELECT json_extract(slice_json, '$.value')::VARCHAR AS value,
				       value AS bias
				FROM metric
				WHERE run_id = '${run_id.replace(/'/g, "''")}'
				  AND metric_name = 'indra_belief_calibration_by_type.bias' AND truth_set_id = 'indra_published_belief'
			)
			SELECT replace(m.value, '"', '') AS value, m.n, m.mae, COALESCE(b.bias, 0) AS bias
			FROM mae_rows m LEFT JOIN bias_rows b USING(value)
			ORDER BY m.mae DESC`
		),
		readRows<Omit<StratumRow, 'cohort_href' | 'applicability'>>(
			con,
			`WITH mae_rows AS (
				SELECT json_extract(slice_json, '$.value')::VARCHAR AS value,
				       CAST(json_extract(slice_json, '$.n') AS BIGINT) AS n,
				       value AS mae
				FROM metric
				WHERE run_id = '${run_id.replace(/'/g, "''")}'
				  AND metric_name = 'indra_belief_calibration_by_source.mae' AND truth_set_id = 'indra_published_belief'
			),
			bias_rows AS (
				SELECT json_extract(slice_json, '$.value')::VARCHAR AS value,
				       value AS bias
				FROM metric
				WHERE run_id = '${run_id.replace(/'/g, "''")}'
				  AND metric_name = 'indra_belief_calibration_by_source.bias' AND truth_set_id = 'indra_published_belief'
			)
			SELECT replace(m.value, '"', '') AS value, m.n, m.mae, COALESCE(b.bias, 0) AS bias
			FROM mae_rows m LEFT JOIN bias_rows b USING(value)
			ORDER BY m.mae DESC`
		),
		readRows<TruthPresentRaw>(
			con,
			`WITH p AS (
				SELECT truth_set_id,
				       replace(json_extract(slice_json, '$.step_kind')::VARCHAR, '"', '') AS step_kind,
				       value AS precision,
				       CAST(json_extract(slice_json, '$.n_compared') AS BIGINT) AS n_compared,
				       CAST(json_extract(slice_json, '$.tp') AS BIGINT) AS tp,
				       CAST(json_extract(slice_json, '$.fp') AS BIGINT) AS fp,
				       CAST(json_extract(slice_json, '$.fn') AS BIGINT) AS fn,
				       COALESCE(
				         CAST(json_extract(slice_json, '$.tn') AS BIGINT),
				         CAST(json_extract(slice_json, '$.n_compared') AS BIGINT)
				           - CAST(json_extract(slice_json, '$.tp') AS BIGINT)
				           - CAST(json_extract(slice_json, '$.fp') AS BIGINT)
				           - CAST(json_extract(slice_json, '$.fn') AS BIGINT)
				       ) AS tn,
				       CAST(json_extract(slice_json, '$.n_gold_labels') AS BIGINT) AS n_gold_labels,
				       CAST(json_extract(slice_json, '$.n_applicable_gold_labels') AS BIGINT) AS n_applicable_gold_labels,
				       CAST(json_extract(slice_json, '$.n_scored_evidences') AS BIGINT) AS n_scored_evidences,
				       json_extract(slice_json, '$.gold_fields')::VARCHAR AS gold_fields_json,
				       COALESCE(json_extract_string(slice_json, '$.positive_gold_label'), 'correct') AS positive_gold_label,
				       COALESCE(json_extract_string(slice_json, '$.negative_gold_rule'), 'any value != ''correct''') AS negative_gold_rule,
				       json_extract_string(slice_json, '$.unavailable_reason') AS unavailable_reason
				FROM metric
				WHERE run_id = '${run_id.replace(/'/g, "''")}'
				  AND metric_name LIKE 'truth_present.%.precision'
			),
			r AS (
				SELECT truth_set_id,
				       replace(json_extract(slice_json, '$.step_kind')::VARCHAR, '"', '') AS step_kind,
				       value AS recall
				FROM metric
				WHERE run_id = '${run_id.replace(/'/g, "''")}'
				  AND metric_name LIKE 'truth_present.%.recall'
			),
			f AS (
				SELECT truth_set_id,
				       replace(json_extract(slice_json, '$.step_kind')::VARCHAR, '"', '') AS step_kind,
				       value AS f1
				FROM metric
				WHERE run_id = '${run_id.replace(/'/g, "''")}'
				  AND metric_name LIKE 'truth_present.%.f1'
			),
			measured AS (
				SELECT p.truth_set_id, p.step_kind,
				       p.precision, COALESCE(r.recall, 0) AS recall, COALESCE(f.f1, 0) AS f1,
				       p.n_compared, p.tp, p.fp, p.fn, p.tn,
				       p.n_gold_labels,
				       p.n_applicable_gold_labels,
				       p.n_scored_evidences,
				       p.gold_fields_json, p.positive_gold_label, p.negative_gold_rule,
				       p.unavailable_reason
				FROM p
				LEFT JOIN r USING (truth_set_id, step_kind)
				LEFT JOIN f USING (truth_set_id, step_kind)
			)
			SELECT * FROM measured
			ORDER BY truth_set_id, step_kind`
		),
		readRows<Omit<ConfidenceCalibrationRow, 'cohort_href' | 'applicability'>>(
			con,
			`WITH aggregate_rows AS (
				SELECT
				   ss.stmt_hash,
				   s.indra_type,
				   CAST(json_extract(ss.output_json, '$.score') AS DOUBLE) AS score,
				   s.indra_belief AS indra_belief,
				   COALESCE(json_extract_string(ss.output_json, '$.confidence'), 'unknown') AS confidence
				FROM scorer_step ss
				JOIN statement s ON s.stmt_hash = ss.stmt_hash
				WHERE ss.run_id = '${run_id.replace(/'/g, "''")}'
				  AND ss.step_kind = 'aggregate'
				  AND json_extract(ss.output_json, '$.score') IS NOT NULL
				  AND s.indra_belief IS NOT NULL
			),
			grouped AS (
				SELECT
				   'all' AS family,
				   confidence,
				   COUNT(*) AS n,
				   AVG(score) AS mean_score,
				   AVG(indra_belief) AS mean_indra_belief,
				   AVG(ABS(score - indra_belief)) AS mae,
				   AVG(score - indra_belief) AS bias
				FROM aggregate_rows
				GROUP BY confidence
				UNION ALL
				SELECT
				   'indra_type:' || indra_type AS family,
				   confidence,
				   COUNT(*) AS n,
				   AVG(score) AS mean_score,
				   AVG(indra_belief) AS mean_indra_belief,
				   AVG(ABS(score - indra_belief)) AS mae,
				   AVG(score - indra_belief) AS bias
				FROM aggregate_rows
				GROUP BY indra_type, confidence
				HAVING COUNT(*) >= 2
			)
			SELECT family, confidence, n, mean_score, mean_indra_belief, mae, bias
			FROM grouped
			ORDER BY CASE family WHEN 'all' THEN 0 ELSE 1 END, family, confidence`
		)
	] as const;
	const latestSettled = await Promise.allSettled(latestValidityReads);
	const latestRejected = latestSettled.find((r): r is PromiseRejectedResult => r.status === 'rejected');
	if (latestRejected) throw latestRejected.reason;
	const [verdictsRaw, mae, rmse, bias, calN, consistency, supports, byIndraTypeRaw, bySourceApiRaw, truthPresentRaw, confidenceCalibrationRaw] =
		latestSettled.map((r) => (r as PromiseFulfilledResult<unknown>).value) as [
			Array<{ verdict: string; n: number }>,
			Array<{ value: number }>,
			Array<{ value: number }>,
			Array<{ value: number }>,
			Array<{ n: number }>,
			Array<{ value: number; n: number }>,
			Array<{ value: number; unavailable_reason: string | null }>,
			Array<Omit<StratumRow, 'cohort_href' | 'applicability'>>,
			Array<Omit<StratumRow, 'cohort_href' | 'applicability'>>,
			TruthPresentRaw[],
			Array<Omit<ConfidenceCalibrationRow, 'cohort_href' | 'applicability'>>
		];

	const cleanVal = (rs: { value: number }[]) => {
		if (rs.length === 0) return null;
		const v = rs[0].value;
		return typeof v === 'number' && !Number.isNaN(v) ? v : null;
	};
	const jsonStringArray = (raw: string | null | undefined): string[] => {
		if (!raw) return [];
		try {
			const parsed = JSON.parse(raw) as unknown;
			return Array.isArray(parsed)
				? parsed.filter((v): v is string => typeof v === 'string')
				: [];
		} catch {
			return [];
		}
	};
	const verdicts = verdictsRaw.map((r) => ({
		...r,
		cohort_href: cohortHref(run_id, { verdict: r.verdict })
	}));
	const byIndraType = byIndraTypeRaw.map((r) => ({
		...r,
		cohort_href: cohortHref(run_id, { grain: 'statement', type: r.value, indra_belief_present: true }),
		applicability: 'arch_blind' as PanelApplicabilityKind
	}));
	const bySourceApi = bySourceApiRaw.map((r) => ({
		...r,
		cohort_href: cohortHref(run_id, { grain: 'statement', source_stratum: r.value, indra_belief_present: true }),
		applicability: 'arch_blind' as PanelApplicabilityKind
	}));
	const truthPresent = truthPresentRaw.map(({ gold_fields_json, ...r }) => ({
		...r,
		gold_fields: jsonStringArray(gold_fields_json),
		cohort_href: cohortHref(run_id, {
			truth_set: r.truth_set_id,
			step_kind: r.step_kind,
			verdict_present: true
		}),
		truth_href: `/runs/${run_id}/truth-sets/${encodeURIComponent(r.truth_set_id)}?step_kind=${encodeURIComponent(r.step_kind)}`,
		applicability: 'arch_blind' as PanelApplicabilityKind
	}));
	const confidenceCalibration = confidenceCalibrationRaw.map((r) => {
		const typeFilter = r.family.startsWith('indra_type:')
			? r.family.slice('indra_type:'.length)
			: null;
		return {
			...r,
			cohort_href: cohortHref(run_id, {
				confidence: r.confidence,
				type: typeFilter,
				score_present: true,
				indra_belief_present: true
			}),
			applicability: 'arch_blind' as PanelApplicabilityKind
		};
	});
	const consistencyRow = consistency[0] ?? null;
	const supportsValue = cleanVal(supports);
	const supportsReason = supports[0]?.unavailable_reason ?? null;

	return {
		run_id,
		scorer_version,
		architecture,
		verdicts,
		calibration: {
			mae: cleanVal(mae),
			rmse: cleanVal(rmse),
			bias: cleanVal(bias),
			n_stmts: calN[0]?.n ?? null,
			cohort_href: cohortHref(run_id, { grain: 'statement', indra_belief_present: true }),
			applicability: 'arch_blind'
		},
		inter_evidence_consistency: {
			mean_stdev: consistencyRow && !Number.isNaN(consistencyRow.value)
				? consistencyRow.value
				: null,
			n_multi_ev: consistencyRow?.n ?? null,
			cohort_href: consistencyRow && !Number.isNaN(consistencyRow.value)
				? cohortHref(run_id, { grain: 'statement', multi_evidence: true })
				: null,
			applicability: consistencyRow && !Number.isNaN(consistencyRow.value)
				? 'arch_blind'
				: 'not_defined',
			not_defined_reason: consistencyRow && !Number.isNaN(consistencyRow.value)
				? null
				: 'needs statements with more than one scored evidence'
		},
		supports_graph_delta: supportsValue,
		supports_graph_href: supportsValue == null ? null : cohortHref(run_id, { grain: 'statement', supports_compare: true }),
		supports_graph_not_defined_reason: supportsValue == null
			? (supportsReason ?? 'needs both supports-rich and non-supports evidence buckets')
			: null,
		byIndraType,
		bySourceApi,
		truthPresent,
		confidenceCalibration,
		metricTaxonomy: [
			{
				panel: 'verdict distribution',
				applicability: 'arch_blind',
				reason: 'aggregate verdict fields exist for both scorer architectures; cohorts stay scoped to one run architecture',
				cohort_href: cohortHref(run_id, { verdict_present: true })
			},
			{
				panel: 'gold-pool comparison',
				applicability: 'arch_blind',
				reason: 'evidence-level verdict/tag gold applies to aggregate rows from either architecture; values stay scoped to one run',
				cohort_href: cohortHref(run_id)
			},
			{
				panel: 'INDRA prior calibration',
				applicability: 'arch_blind',
				reason: 'mean score residuals can be computed for any run with INDRA belief anchors',
				cohort_href: cohortHref(run_id, { grain: 'statement', indra_belief_present: true })
			},
			{
				panel: 'confidence calibration by family',
				applicability: 'arch_blind',
				reason: 'confidence buckets are aggregate fields on both architectures and are measured separately inside the selected architecture',
				cohort_href: cohortHref(run_id, { score_present: true, indra_belief_present: true })
			},
			{
				panel: 'decomposed probe health',
				applicability: architecture === 'decomposed' ? 'arch_conditioned' : 'not_defined',
				reason: architecture === 'decomposed'
					? 'probe rows are native to decomposed scoring'
					: 'monolithic scoring does not emit decomposed probe rows',
				cohort_href: architecture === 'decomposed' ? cohortHref(run_id, { trace_fidelity: 'native_decomposed' }) : null
			},
			{
				panel: 'paired deltas',
				applicability: pairedGroup ? 'paired_only' : 'not_defined',
				reason: pairedGroup
					? `architecture deltas require overlap accounting inside paired group ${pairedGroup}`
					: 'architecture deltas require an overlap-first paired workbench',
				cohort_href: pairedGroup ? `/pairs/${pairedGroup}` : null
			}
		]
	};
}
