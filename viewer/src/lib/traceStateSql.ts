function sqlQuote(s: string): string {
	return s.replace(/'/g, "''");
}

function sqlString(s: string): string {
	return `'${sqlQuote(s)}'`;
}

function traceEvidenceCteSql(
	runId: string,
	architectureSql: string,
	statusSql: string,
	snapshotSql: string | null = null
): string {
	const qr = sqlQuote(runId);
	const err = `(NULLIF(ss.error, '') IS NOT NULL OR json_extract_string(ss.output_json, '$.error') IS NOT NULL)`;
	const snapshotPredicate = snapshotSql ? `AND ss.started_at <= ${snapshotSql}` : '';
	return `ranked_steps AS (
			SELECT
			   ss.*,
			   ROW_NUMBER() OVER (
			     PARTITION BY ss.run_id, ss.stmt_hash, ss.evidence_hash
			     ORDER BY
			       CASE WHEN ${err} THEN 0 WHEN ss.step_kind='aggregate' THEN 1 ELSE 2 END,
			       ss.started_at DESC,
			       ss.step_hash DESC
			   ) AS representative_rank
			 FROM scorer_step ss
			 WHERE ss.run_id='${qr}'
			   AND ss.evidence_hash IS NOT NULL
			   ${snapshotPredicate}
		),
		trace_evidence AS (
			SELECT
			   ss.run_id,
			   COALESCE(MIN(ss.architecture), ${architectureSql}) AS architecture,
			   ss.stmt_hash,
			   ss.evidence_hash,
			   MAX(ss.started_at) AS latest_started_at,
			   MAX(CASE WHEN ss.representative_rank=1 THEN ss.step_hash ELSE NULL END) AS step_hash,
			   MAX(CASE WHEN ss.representative_rank=1 THEN ss.step_kind ELSE NULL END) AS step_kind,
			   SUM(CASE WHEN ss.step_kind='aggregate' THEN 1 ELSE 0 END) AS n_aggregate,
			   SUM(CASE WHEN ss.step_kind<>'aggregate' THEN 1 ELSE 0 END) AS n_nonaggregate,
			   SUM(CASE WHEN ${err} THEN 1 ELSE 0 END) AS n_errors,
			   SUM(CASE WHEN ss.step_kind='parse_claim' THEN 1 ELSE 0 END) AS n_parse_claim,
			   SUM(CASE WHEN ss.step_kind='build_context' THEN 1 ELSE 0 END) AS n_build_context,
			   SUM(CASE WHEN ss.step_kind='substrate_route' THEN 1 ELSE 0 END) AS n_substrate_route,
			   SUM(CASE WHEN ss.step_kind='subject_role_probe' THEN 1 ELSE 0 END) AS n_subject_role_probe,
			   SUM(CASE WHEN ss.step_kind='object_role_probe' THEN 1 ELSE 0 END) AS n_object_role_probe,
			   SUM(CASE WHEN ss.step_kind='relation_axis_probe' THEN 1 ELSE 0 END) AS n_relation_axis_probe,
			   SUM(CASE WHEN ss.step_kind='scope_probe' THEN 1 ELSE 0 END) AS n_scope_probe,
			   SUM(CASE WHEN ss.step_kind='grounding' THEN 1 ELSE 0 END) AS n_grounding,
			   SUM(CASE WHEN ss.step_kind='adjudicate' THEN 1 ELSE 0 END) AS n_adjudicate,
			   STRING_AGG(DISTINCT ss.step_kind, ',') AS captured_step_kinds,
			   MAX(CASE WHEN ss.step_kind='aggregate' AND json_extract_string(ss.output_json, '$.tier') IS NOT NULL THEN 1 ELSE 0 END) AS has_tier,
			   MAX(CASE WHEN ss.step_kind='aggregate' AND json_extract_string(ss.output_json, '$.grounding_status') IS NOT NULL THEN 1 ELSE 0 END) AS has_grounding_status,
			   MAX(CASE WHEN ss.step_kind='aggregate' AND (json_extract_string(ss.output_json, '$.verdict') IS NOT NULL OR json_extract(ss.output_json, '$.score') IS NOT NULL) THEN 1 ELSE 0 END) AS has_parse_state,
			   MAX(CASE WHEN ss.step_kind='aggregate' AND json_type(ss.output_json, '$.call_log') = 'ARRAY' THEN 1 ELSE 0 END) AS has_call_log,
			   MAX(CASE WHEN ss.step_kind='aggregate' AND json_extract_string(ss.output_json, '$.raw_text') IS NOT NULL THEN 1 ELSE 0 END) AS has_raw_text,
			   MAX(CASE WHEN ss.step_kind='aggregate' AND (json_type(ss.output_json, '$.selected_example_ids') = 'ARRAY' OR json_type(ss.output_json, '$.selected_examples') = 'ARRAY') THEN 1 ELSE 0 END) AS has_selected_examples,
			   MAX(CASE WHEN ss.step_kind='aggregate' THEN CAST(json_extract(ss.output_json, '$.score') AS DOUBLE) ELSE NULL END) AS score,
			   MAX(CASE WHEN ss.step_kind='aggregate' THEN json_extract_string(ss.output_json, '$.verdict') ELSE NULL END) AS verdict,
			   MAX(CASE WHEN ss.step_kind='aggregate' THEN json_extract_string(ss.output_json, '$.confidence') ELSE NULL END) AS confidence,
			   CASE WHEN SUM(CASE WHEN ss.latency_ms IS NOT NULL THEN 1 ELSE 0 END) > 0 THEN SUM(COALESCE(ss.latency_ms, 0)) ELSE NULL END AS latency_ms,
			   CASE WHEN SUM(CASE WHEN ss.prompt_tokens IS NOT NULL THEN 1 ELSE 0 END) > 0 THEN SUM(COALESCE(ss.prompt_tokens, 0)) ELSE NULL END AS prompt_tokens,
			   CASE WHEN SUM(CASE WHEN ss.out_tokens IS NOT NULL THEN 1 ELSE 0 END) > 0 THEN SUM(COALESCE(ss.out_tokens, 0)) ELSE NULL END AS out_tokens
			 FROM ranked_steps ss
			 GROUP BY ss.run_id, ss.stmt_hash, ss.evidence_hash
		),
		trace_rows AS (
			SELECT
			   *,
			   CASE
			     WHEN n_errors > 0 THEN 'step_error'
			     WHEN n_aggregate = 0 AND ${statusSql} = 'succeeded' THEN 'missing_aggregate'
			     WHEN n_aggregate = 0 THEN 'terminated_inflight'
			     WHEN ${architectureSql} = 'decomposed' THEN
			       CASE
			         WHEN n_nonaggregate = 0 THEN 'aggregate_only'
			         WHEN n_parse_claim > 0 AND n_build_context > 0 AND n_substrate_route > 0
			          AND n_subject_role_probe > 0 AND n_object_role_probe > 0
			          AND n_relation_axis_probe > 0 AND n_scope_probe > 0
			          AND n_grounding > 0 AND n_adjudicate > 0 THEN 'full'
			         ELSE 'partial'
			       END
			     WHEN ${architectureSql} = 'monolithic' THEN
			       CASE
			         WHEN has_tier > 0 AND has_grounding_status > 0 AND has_parse_state > 0
			          AND has_call_log > 0 AND has_selected_examples > 0 THEN 'full'
			         WHEN has_tier > 0 OR has_grounding_status > 0 OR has_call_log > 0 OR has_raw_text > 0 THEN 'partial'
			         ELSE 'aggregate_only'
			       END
			     ELSE 'not_applicable'
			   END AS trace_state
			FROM trace_evidence
		)`;
}

export function traceEvidenceCteForRun(
	runId: string,
	architecture: string,
	status: string,
	snapshotStartedAt?: string | null
): string {
	const snapshotSql = snapshotStartedAt ? `TIMESTAMP '${sqlQuote(snapshotStartedAt)}'` : null;
	return traceEvidenceCteSql(runId, sqlString(architecture), sqlString(status), snapshotSql);
}

export function traceEvidenceCtesWithRunMeta(runId: string, snapshotStartedAt?: string | null): string {
	const qr = sqlQuote(runId);
	const snapshotSql = snapshotStartedAt ? `TIMESTAMP '${sqlQuote(snapshotStartedAt)}'` : null;
	return `run_meta AS (
			SELECT COALESCE(architecture, 'unknown') AS architecture,
			       COALESCE(status, 'unknown') AS status
			FROM score_run
			WHERE run_id='${qr}'
		),
		${traceEvidenceCteSql(
			runId,
			'(SELECT architecture FROM run_meta)',
			'(SELECT status FROM run_meta)',
			snapshotSql
		)}`;
}
