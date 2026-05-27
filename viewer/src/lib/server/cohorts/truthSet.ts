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

import { TRUTH_SET_OVERLAP_ROWS_PER_STATUS, TRUTH_SET_OVERLAP_ROW_LIMIT, closeInstance, cohortHref, connect, dbExists, dbPath, durationSeconds, normalizeDuckValue, readRows, repairRerunLineageSqlOptions, rows, safeRate, scalar, sqlQuote, tableExists, timestampMs, truthSetOverlapCte } from '$lib/db';
import type { TruthSetOverlapDetail, TruthSetOverlapRow, TruthSetOverlapStatus } from '$lib/db';

export async function getTruthSetOverlapDetail(
	run_id: string,
	truth_set_id: string,
	step_kind = 'aggregate'
): Promise<TruthSetOverlapDetail | null> {
	if (!dbExists()) return null;
	const con = await connect();
	try {
		const qr = sqlQuote(run_id);
		const qt = sqlQuote(truth_set_id);
		const runRows = await readRows<{
			run_id: string;
			architecture: string | null;
			status: string;
		}>(
			con,
			`SELECT run_id, architecture, status
			 FROM score_run
			 WHERE run_id='${qr}'`
		);
		if (runRows.length === 0) return null;
		const truthRows = await readRows<{ id: string; name: string | null }>(
			con,
			`SELECT id, name FROM truth_set WHERE id='${qt}'`
		);
		if (truthRows.length === 0) return null;

		const cte = truthSetOverlapCte(run_id, truth_set_id, step_kind);
		const summaryRows = await readRows<{
			n_gold_labels: number;
			n_applicable_gold_labels: number;
			n_metric_compared_rows: number;
			n_scored_evidences: number;
			n_compared_labels: number;
			n_no_aggregate_verdict: number;
			n_context_mismatch: number;
			n_not_scored_in_run: number;
			n_not_in_corpus: number;
			gold_fields_json: string | null;
		}>(
			con,
			`${cte}
			SELECT
			  COALESCE((SELECT COUNT(*) FROM classified_labels), 0) AS n_gold_labels,
			  COALESCE((SELECT SUM(CASE WHEN matched_verdict_rows > 0 THEN 1 ELSE 0 END) FROM classified_labels), 0) AS n_applicable_gold_labels,
			  COALESCE((SELECT COUNT(*) FROM selected_gold), 0) AS n_metric_compared_rows,
			  COALESCE((SELECT COUNT(*) FROM scorer_verdicts), 0) AS n_scored_evidences,
			  COALESCE((SELECT SUM(CASE WHEN status = 'compared' THEN 1 ELSE 0 END) FROM classified_labels), 0) AS n_compared_labels,
			  COALESCE((SELECT SUM(CASE WHEN status = 'no_aggregate_verdict' THEN 1 ELSE 0 END) FROM classified_labels), 0) AS n_no_aggregate_verdict,
			  COALESCE((SELECT SUM(CASE WHEN status = 'context_mismatch' THEN 1 ELSE 0 END) FROM classified_labels), 0) AS n_context_mismatch,
			  COALESCE((SELECT SUM(CASE WHEN status = 'not_scored_in_run' THEN 1 ELSE 0 END) FROM classified_labels), 0) AS n_not_scored_in_run,
			  COALESCE((SELECT SUM(CASE WHEN status = 'not_in_corpus' THEN 1 ELSE 0 END) FROM classified_labels), 0) AS n_not_in_corpus,
			  COALESCE(
			    (SELECT to_json(list(DISTINCT gold_field ORDER BY gold_field))::VARCHAR FROM selected_gold),
			    (SELECT COALESCE(to_json(list(DISTINCT field ORDER BY field))::VARCHAR, '[]') FROM classified_labels),
			    '[]'
			  ) AS gold_fields_json`
		);
		const detailRows = await readRows<Omit<TruthSetOverlapRow, 'statement_href'> & { status: TruthSetOverlapStatus }>(
			con,
			`${cte},
			ranked_labels AS (
				SELECT *,
				       row_number() OVER (
				         PARTITION BY status
				         ORDER BY field, target_id, relation_target_id NULLS LAST
				       ) AS status_rank
				FROM classified_labels
			)
			SELECT label_id,
			       target_id,
			       relation_target_id,
			       field,
			       value_text,
			       provenance,
			       corpus_stmt_hash,
			       source_api,
			       evidence_text,
			       same_evidence_scored_rows,
			       matched_step_rows,
			       matched_verdict_rows,
			       matched_no_verdict_rows,
			       sample_matched_stmt_hash,
			       sample_our_verdict,
			       sample_scored_stmt_hash,
			       status
			FROM ranked_labels
			WHERE status_rank <= ${TRUTH_SET_OVERLAP_ROWS_PER_STATUS}
			ORDER BY CASE status
			           WHEN 'compared' THEN 0
			           WHEN 'no_aggregate_verdict' THEN 1
			           WHEN 'context_mismatch' THEN 2
			           WHEN 'not_scored_in_run' THEN 3
			           ELSE 4
			         END,
			         field,
			         target_id,
			         relation_target_id NULLS LAST
			LIMIT ${TRUTH_SET_OVERLAP_ROW_LIMIT}`
		);
		const s = summaryRows[0] ?? {
			n_gold_labels: 0,
			n_applicable_gold_labels: 0,
			n_metric_compared_rows: 0,
			n_scored_evidences: 0,
			n_compared_labels: 0,
			n_no_aggregate_verdict: 0,
			n_context_mismatch: 0,
			n_not_scored_in_run: 0,
			n_not_in_corpus: 0,
			gold_fields_json: '[]'
		};
		let goldFields: string[] = [];
		try {
			const parsed = JSON.parse(s.gold_fields_json ?? '[]') as unknown;
			goldFields = Array.isArray(parsed)
				? parsed.filter((v): v is string => typeof v === 'string')
				: [];
		} catch {
			goldFields = [];
		}
		return {
			run_id,
			truth_set_id,
			truth_set_name: truthRows[0].name ?? truth_set_id,
			architecture: runRows[0].architecture ?? 'unknown',
			status: runRows[0].status,
			metric_step_kind: step_kind,
			n_gold_labels: Number(s.n_gold_labels ?? 0),
			n_applicable_gold_labels: Number(s.n_applicable_gold_labels ?? 0),
			n_metric_compared_rows: Number(s.n_metric_compared_rows ?? 0),
			n_scored_evidences: Number(s.n_scored_evidences ?? 0),
			n_compared_labels: Number(s.n_compared_labels ?? 0),
			n_no_aggregate_verdict: Number(s.n_no_aggregate_verdict ?? 0),
			n_context_mismatch: Number(s.n_context_mismatch ?? 0),
			n_not_scored_in_run: Number(s.n_not_scored_in_run ?? 0),
			n_not_in_corpus: Number(s.n_not_in_corpus ?? 0),
			positive_gold_label: 'correct',
			negative_gold_rule: "any value != 'correct'",
			gold_fields: goldFields,
			cohort_href: step_kind === 'aggregate' && Number(s.n_metric_compared_rows ?? 0) > 0
				? cohortHref(run_id, { truth_set: truth_set_id, step_kind, verdict_present: true })
				: null,
			rows: detailRows.map((r) => ({
				...r,
				label_id: Number(r.label_id),
				same_evidence_scored_rows: Number(r.same_evidence_scored_rows ?? 0),
				matched_step_rows: Number(r.matched_step_rows ?? 0),
				matched_verdict_rows: Number(r.matched_verdict_rows ?? 0),
				matched_no_verdict_rows: Number(r.matched_no_verdict_rows ?? 0),
				statement_href: r.sample_matched_stmt_hash
					? `/statements/${r.sample_matched_stmt_hash}?run_id=${run_id}`
					: r.corpus_stmt_hash
						? `/statements/${r.corpus_stmt_hash}?run_id=${run_id}`
						: null
			})),
			row_limit: TRUTH_SET_OVERLAP_ROW_LIMIT,
			status_row_limit: TRUTH_SET_OVERLAP_ROWS_PER_STATUS
		};
	} finally {
		con.disconnectSync?.();
	}
}
