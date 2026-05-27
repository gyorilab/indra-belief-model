// Renamed from $lib/runCohortQueries.ts on 2026-05-27 (B4 of deferred hypergraph).
// No code changes vs the prior location; see git mv for history.

import type { RunCohortFilters } from './types';
import { statementEvidenceSourceStratumSql } from './evidenceMembership';
import { cohortEvidenceLevelPredicates } from './sql';
import { traceEvidenceCteForRun } from '../traceStateSql';
import { sqlQuote } from './sqlEscape';



export function runCohortAgentNamesCte(): string {
	return `ags AS (
				SELECT stmt_hash,
				       string_agg(name, ', ' ORDER BY
				         CASE role
				           WHEN 'subj' THEN 0 WHEN 'enz' THEN 0
				           WHEN 'obj' THEN 1 WHEN 'sub' THEN 1
				           WHEN 'member' THEN 2 ELSE 3 END,
				         role_index) AS agent_names
				FROM agent
				GROUP BY stmt_hash
			)`;
}

export function statementCohortCte(
	run_id: string,
	architecture: string,
	filters: RunCohortFilters
): string {
	const qr = sqlQuote(run_id);
	// DuckDB is brittle about truth_label EXISTS predicates inside window
	// ORDER BY expressions. The WHERE layer still enforces truth_set
	// membership; representative ranking only uses directly-scored facets.
	const representativeFilters = { ...filters, truth_set: undefined };
	const representativeEvidencePredicates = cohortEvidenceLevelPredicates(representativeFilters, {
		step: 'ss',
		evidence: 'e',
		truthLabel: 'tl_rep'
	});
	const representativePreferenceSql = representativeEvidencePredicates.length
		? `CASE WHEN ${representativeEvidencePredicates.join(' AND ')} THEN 0 ELSE 1 END,`
		: '';

	return `WITH ${runCohortAgentNamesCte()},
			scored_statement_evidence AS (
				SELECT
				   ss.run_id,
				   COALESCE(ss.architecture, '${sqlQuote(architecture)}') AS architecture,
				   ss.stmt_hash,
				   ss.evidence_hash,
				   e.source_api,
				   e.pmid,
				   e.text,
				   ss.latency_ms,
				   ss.prompt_tokens,
				   ss.out_tokens,
				   CAST(json_extract(ss.output_json, '$.score') AS DOUBLE) AS evidence_score,
				   ROW_NUMBER() OVER (
				     PARTITION BY ss.run_id, ss.stmt_hash
				     ORDER BY ${representativePreferenceSql}
				              ABS(COALESCE(CAST(json_extract(ss.output_json, '$.score') AS DOUBLE) - s.indra_belief, 0)) DESC,
				              ss.evidence_hash
				   ) AS representative_rank
				FROM scorer_step ss
				JOIN statement s ON s.stmt_hash = ss.stmt_hash
				LEFT JOIN evidence e ON e.evidence_hash = ss.evidence_hash
				WHERE ss.run_id='${qr}'
				  AND ss.step_kind='aggregate'
				  AND json_extract(ss.output_json, '$.score') IS NOT NULL
			),
			stmt_scores AS (
				SELECT
				   se.run_id,
				   COALESCE(MIN(se.architecture), '${sqlQuote(architecture)}') AS architecture,
				   s.stmt_hash,
				   NULL::VARCHAR AS evidence_hash,
				   MAX(CASE WHEN se.representative_rank=1 THEN se.evidence_hash ELSE NULL END) AS representative_evidence_hash,
				   s.indra_type,
				   COALESCE(ags.agent_names, '') AS agent_names,
				   MAX(CASE WHEN se.representative_rank=1 THEN se.source_api ELSE NULL END) AS source_api,
				   ${statementEvidenceSourceStratumSql('s.stmt_hash')} AS source_stratum,
				   MAX(CASE WHEN se.representative_rank=1 THEN se.pmid ELSE NULL END) AS pmid,
				   MAX(CASE WHEN se.representative_rank=1 THEN se.text ELSE NULL END) AS text,
				   s.indra_belief,
				   AVG(se.evidence_score) AS score,
				   CASE
				     WHEN s.indra_belief IS NOT NULL AND AVG(se.evidence_score) IS NOT NULL
				     THEN AVG(se.evidence_score) - s.indra_belief
				     ELSE NULL
				   END AS residual,
				   NULL::VARCHAR AS verdict,
				   NULL::VARCHAR AS confidence,
				   NULL::VARCHAR AS trace_state,
				   NULL::VARCHAR AS step_kind,
				   SUM(se.latency_ms) AS latency_ms,
				   SUM(se.prompt_tokens) AS prompt_tokens,
				   SUM(se.out_tokens) AS out_tokens,
				   COUNT(*) AS n_evidences_for_stmt,
				   (s.supports_count > 0 OR s.supported_by_count > 0) AS has_supports
				 FROM scored_statement_evidence se
				 JOIN statement s ON s.stmt_hash = se.stmt_hash
				 LEFT JOIN ags ON ags.stmt_hash = se.stmt_hash
				 GROUP BY se.run_id, s.stmt_hash, s.indra_type, s.indra_belief,
				          s.supports_count, s.supported_by_count, ags.agent_names
			)`;
}

export function statementCohortRowsSql(stmtCte: string, whereSql: string, limit: number): string {
	return `${stmtCte}
				 SELECT * FROM stmt_scores
				 WHERE ${whereSql}
				 ORDER BY ABS(COALESCE(residual, 0)) DESC, stmt_hash
				 LIMIT ${limit}`;
}

export function traceCohortCte(
	run_id: string,
	architecture: string,
	status: string,
	snapshotStartedAt: string | null
): string {
	return `WITH ${runCohortAgentNamesCte()}, ${traceEvidenceCteForRun(run_id, architecture, status, snapshotStartedAt)}`;
}

export function traceCohortCountSql(traceCte: string, whereSql: string): string {
	return `${traceCte}
				 SELECT COUNT(*)
				 FROM trace_rows ts
				 LEFT JOIN statement s ON s.stmt_hash = ts.stmt_hash
				 LEFT JOIN evidence e ON e.evidence_hash = ts.evidence_hash
				 WHERE ${whereSql}`;
}

export function traceCohortRowsSql(
	traceCte: string,
	architecture: string,
	whereSql: string,
	limit: number
): string {
	return `${traceCte}
				SELECT
				   ts.run_id,
				   COALESCE(ts.architecture, '${sqlQuote(architecture)}') AS architecture,
				   ts.stmt_hash,
				   ts.evidence_hash,
				   COALESCE(s.indra_type, 'unknown') AS indra_type,
				   COALESCE(ags.agent_names, '') AS agent_names,
				   e.source_api,
				   e.pmid,
				   e.text,
				   s.indra_belief,
				   ts.score,
				   CASE
				     WHEN s.indra_belief IS NOT NULL AND ts.score IS NOT NULL
				     THEN ts.score - s.indra_belief
				     ELSE NULL
				   END AS residual,
				   ts.verdict,
				   ts.confidence,
				   ts.trace_state,
				   ts.step_kind,
				   ts.latency_ms,
				   ts.prompt_tokens,
				   ts.out_tokens,
				   (SELECT COUNT(*) FROM trace_rows ts2
				    WHERE ts2.run_id=ts.run_id AND ts2.stmt_hash=ts.stmt_hash) AS n_evidences_for_stmt,
				   (COALESCE(s.supports_count, 0) > 0 OR COALESCE(s.supported_by_count, 0) > 0) AS has_supports
				 FROM trace_rows ts
				 LEFT JOIN statement s ON s.stmt_hash = ts.stmt_hash
				 LEFT JOIN evidence e ON e.evidence_hash = ts.evidence_hash
				 LEFT JOIN ags ON ags.stmt_hash = ts.stmt_hash
				 WHERE ${whereSql}
				 ORDER BY ABS(COALESCE(ts.score - s.indra_belief, 0)) DESC,
				      ts.stmt_hash,
				      ts.evidence_hash
				 LIMIT ${limit}`;
}

export function traceCohortBaseSql(traceCte: string): string {
	return `${traceCte}, cohort_base AS (
						SELECT
						   ts.run_id,
						   ts.stmt_hash,
						   ts.evidence_hash,
							   ts.trace_state,
							   ts.n_substrate_route,
							   ts.n_subject_role_probe,
							   ts.n_object_role_probe,
							   ts.n_relation_axis_probe,
							   ts.n_scope_probe,
							   ts.verdict,
						   ts.confidence,
						   s.indra_belief,
						   ts.score,
						   COALESCE(s.indra_type, '') AS indra_type,
						   COALESCE(e.source_api, '') AS source_api,
						   COALESCE(${statementEvidenceSourceStratumSql('ts.stmt_hash')}, '') AS source_stratum,
						   (SELECT COUNT(*) FROM trace_rows ts2 WHERE ts2.run_id=ts.run_id AND ts2.stmt_hash=ts.stmt_hash) AS n_trace_rows_for_stmt,
						   (COALESCE(s.supports_count, 0) > 0 OR COALESCE(s.supported_by_count, 0) > 0) AS has_supports
						FROM trace_rows ts
						LEFT JOIN statement s ON s.stmt_hash = ts.stmt_hash
						LEFT JOIN evidence e ON e.evidence_hash = ts.evidence_hash
					)`;
}

export function aggregateEvidenceCohortCountSql(whereSql: string): string {
	return `SELECT COUNT(*)
			 FROM scorer_step ss
			 JOIN statement s ON s.stmt_hash = ss.stmt_hash
			 LEFT JOIN evidence e ON e.evidence_hash = ss.evidence_hash
			 WHERE ${whereSql}`;
}

export function aggregateEvidenceCohortRowsSql(
	architecture: string,
	whereSql: string,
	limit: number
): string {
	return `WITH ${runCohortAgentNamesCte()}
			SELECT
			   ss.run_id,
			   COALESCE(ss.architecture, '${sqlQuote(architecture)}') AS architecture,
			   ss.stmt_hash,
			   ss.evidence_hash,
			   s.indra_type,
			   COALESCE(ags.agent_names, '') AS agent_names,
			   e.source_api,
			   e.pmid,
			   e.text,
			   s.indra_belief,
			   CAST(json_extract(ss.output_json, '$.score') AS DOUBLE) AS score,
			   CASE
			     WHEN s.indra_belief IS NOT NULL AND json_extract(ss.output_json, '$.score') IS NOT NULL
			     THEN CAST(json_extract(ss.output_json, '$.score') AS DOUBLE) - s.indra_belief
			     ELSE NULL
			   END AS residual,
			   json_extract_string(ss.output_json, '$.verdict') AS verdict,
			   json_extract_string(ss.output_json, '$.confidence') AS confidence,
			   NULL::VARCHAR AS trace_state,
			   ss.step_kind AS step_kind,
			   ss.latency_ms,
			   ss.prompt_tokens,
			   ss.out_tokens,
			   (SELECT COUNT(*) FROM scorer_step ss2
			    WHERE ss2.run_id=ss.run_id AND ss2.step_kind='aggregate' AND ss2.stmt_hash=ss.stmt_hash) AS n_evidences_for_stmt,
			   (s.supports_count > 0 OR s.supported_by_count > 0) AS has_supports
			 FROM scorer_step ss
			 JOIN statement s ON s.stmt_hash = ss.stmt_hash
			 LEFT JOIN evidence e ON e.evidence_hash = ss.evidence_hash
			 LEFT JOIN ags ON ags.stmt_hash = ss.stmt_hash
			 WHERE ${whereSql}
			 ORDER BY ABS(COALESCE(
			        CAST(json_extract(ss.output_json, '$.score') AS DOUBLE) - s.indra_belief,
			        0
			      )) DESC,
			      ss.stmt_hash,
			      ss.evidence_hash
			 LIMIT ${limit}`;
}

export function aggregateEvidenceCohortBaseSql(run_id: string): string {
	const qr = sqlQuote(run_id);
	return `WITH cohort_base AS (
					SELECT
					   ss.run_id,
					   ss.stmt_hash,
					   ss.evidence_hash,
					   ss.step_kind,
					   s.indra_type,
					   s.indra_belief,
					   json_extract(ss.output_json, '$.score') AS score,
					   COALESCE(e.source_api, '') AS source_api,
					   COALESCE(${statementEvidenceSourceStratumSql('s.stmt_hash')}, '') AS source_stratum,
					   json_extract_string(ss.output_json, '$.verdict') AS verdict,
					   json_extract_string(ss.output_json, '$.confidence') AS confidence,
					   (SELECT COUNT(*) FROM scorer_step ss2
					    WHERE ss2.run_id=ss.run_id AND ss2.step_kind='aggregate' AND ss2.stmt_hash=ss.stmt_hash) AS n_evidences_for_stmt,
					   (s.supports_count > 0 OR s.supported_by_count > 0) AS has_supports
					FROM scorer_step ss
					JOIN statement s ON s.stmt_hash = ss.stmt_hash
					LEFT JOIN evidence e ON e.evidence_hash = ss.evidence_hash
					WHERE ss.run_id='${qr}'
					  AND ss.step_kind='aggregate'
				)`;
}
