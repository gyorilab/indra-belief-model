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

import { REPAIR_RECOVERY_PAGE_LIMIT, closeInstance, connect, dbExists, dbPath, durationSeconds, normalizeDuckValue, readRows, repairRerunLineageSqlOptions, rows, safeRate, scalar, sqlQuote, tableExists, timestampMs } from '$lib/db';
import type { RepairRerunActiveIntent, RepairRerunCandidateLane, RepairRerunComparison, RepairRerunDetail, RepairRerunRecoveryIntent, RunRepairBacklog, RunRepairBacklogOptions, RunRepairCandidateRow } from '$lib/db';
import type { RepairRerunLineageSqlOptions } from '$lib/repairRerunSql';
import { REPAIR_RERUN_QUEUED_INTENT_LOCK_MINUTES, REPAIR_RERUN_TERMINAL_STATUS_SQL, repairCandidateAvailablePredicateSql, repairRerunChildRunIdSql, repairRerunParentCorrectionIdNumberSql, repairRerunParentCorrectionIdStringSql, repairRerunSourceDumpIdSql } from '$lib/repairRerunSql';
import { RUN_COHORT_LIMIT } from '$lib/cohorts/sql';
import type { DuckDBConnection } from '@duckdb/node-api';

export function boundedNonNegativeInteger(value: number | null | undefined, fallback: number): number {
	const n = Number(value);
	return Number.isInteger(n) && n >= 0 ? n : fallback;
}


export async function getRunRepairBacklog(
	run_id: string,
	options: RunRepairBacklogOptions = {}
): Promise<RunRepairBacklog | null> {
	if (!dbExists()) return null;
	const recoveryOffset = boundedNonNegativeInteger(options.recoveryOffset, 0);
	const recoveryLimit = Math.min(
		100,
		Math.max(1, boundedNonNegativeInteger(options.recoveryLimit, REPAIR_RECOVERY_PAGE_LIMIT))
	);
	const con = await connect();
	try {
		const qr = sqlQuote(run_id);
		const runRows = await rows<{ architecture: string | null }>(
			con,
			`SELECT architecture FROM score_run WHERE run_id='${qr}'`
		);
		if (runRows.length === 0) return null;
		const architecture = runRows[0].architecture ?? 'unknown';
		const reruns = await getRepairRerunComparisons(con, run_id, architecture);
		const hasCorrectionTable = await scalar(
			con,
			`SELECT COUNT(*)
			   FROM information_schema.tables
			  WHERE table_name='scorer_step_correction'`
		);
		if (hasCorrectionTable === 0) {
			return {
				run_id,
				architecture,
				openCount: 0,
				rows: [],
				reruns,
				activeRerunIntents: [],
				activeRerunIntentCount: 0,
				recoverableRerunIntents: [],
				recoverableRerunIntentCount: 0,
				recoverableRerunIntentOffset: recoveryOffset,
				recoverableRerunIntentLimit: recoveryLimit,
				limit: RUN_COHORT_LIMIT
			};
		}
		const hasArchitectureColumn = await scalar(
			con,
			`SELECT COUNT(*)
			   FROM information_schema.columns
			  WHERE table_name='scorer_step_correction'
			    AND column_name='architecture'`
		);
		const lineageSqlOptions = await repairRerunLineageSqlOptions(con);
		const correctionArchitectureSql = hasArchitectureColumn > 0 ? 'c.architecture' : 'NULL::VARCHAR';
		const intentArchitectureSql = hasArchitectureColumn > 0 ? 'i.architecture' : `'${sqlQuote(architecture)}'`;
		const recoverableRerunIntentsResult = await getRecoverableRepairRerunIntents(
			con,
			run_id,
			architecture,
			intentArchitectureSql,
			lineageSqlOptions,
			recoveryOffset,
			recoveryLimit
		);
		const activeRerunIntents = await getActiveRepairRerunIntents(
			con,
			run_id,
			architecture,
			intentArchitectureSql,
			lineageSqlOptions
		);

		const availablePredicate = repairCandidateAvailablePredicateSql('c', lineageSqlOptions);
		const slotReviewParentCorrectionIdSql = repairRerunParentCorrectionIdNumberSql('r', lineageSqlOptions);
		const openCount = await scalar(
			con,
			`SELECT COUNT(*)
			   FROM scorer_step_correction c
			  WHERE c.run_id='${qr}'
			    AND c.status='open'
			    AND c.correction_kind='repair_candidate'
			    AND ${availablePredicate}`
		);
		const limit = RUN_COHORT_LIMIT;
		const repairRows = await rows<RunRepairCandidateRow>(
			con,
			`WITH ags AS (
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
			latest_probe_slot_review AS (
				SELECT
				   parent_correction_id,
				   probe_slot_review_slots,
				   probe_slot_review_note,
				   probe_slot_review_reviewer,
				   probe_slot_reviewed_at,
				   probe_slot_review_count
				FROM (
					SELECT
					   ${slotReviewParentCorrectionIdSql} AS parent_correction_id,
					   json_extract_string(r.value_json, '$.selected_probe_slots') AS probe_slot_review_slots,
					   COALESCE(json_extract_string(r.value_json, '$.reviewer_note'), r.note) AS probe_slot_review_note,
					   r.reviewer AS probe_slot_review_reviewer,
					   r.created_at::VARCHAR AS probe_slot_reviewed_at,
					   COUNT(*) OVER (PARTITION BY ${slotReviewParentCorrectionIdSql}) AS probe_slot_review_count,
					   ROW_NUMBER() OVER (
					     PARTITION BY ${slotReviewParentCorrectionIdSql}
					     ORDER BY r.created_at DESC, r.correction_id DESC
					   ) AS rn
					 FROM scorer_step_correction r
					 WHERE r.run_id='${qr}'
					   AND r.correction_kind='probe_slot_review'
				)
				WHERE rn=1
			)
			SELECT
			   c.correction_id,
			   c.step_hash,
			   c.run_id,
			   COALESCE(${correctionArchitectureSql}, ss.architecture, '${sqlQuote(architecture)}') AS architecture,
			   c.stmt_hash,
			   c.evidence_hash,
			   c.correction_kind,
			   c.status,
			   c.reviewer,
			   c.note,
			   CAST(c.value_json AS VARCHAR) AS value_json,
			   c.source_route,
			   CAST(c.source_filters_json AS VARCHAR) AS source_filters_json,
			   c.created_at::VARCHAR AS created_at,
			   ss.step_kind,
			   json_extract_string(c.value_json, '$.suspected_step_kind') AS suspected_step_kind,
			   json_extract_string(c.value_json, '$.observed.probe_coverage') AS probe_coverage,
			   json_extract_string(c.value_json, '$.observed.missing_probe_slots') AS missing_probe_slots,
			   CAST(json_extract(c.value_json, '$.observed.probe_counts.substrate_route') AS BIGINT) AS n_substrate_route,
			   CAST(json_extract(c.value_json, '$.observed.probe_counts.subject_role_probe') AS BIGINT) AS n_subject_role_probe,
			   CAST(json_extract(c.value_json, '$.observed.probe_counts.object_role_probe') AS BIGINT) AS n_object_role_probe,
			   CAST(json_extract(c.value_json, '$.observed.probe_counts.relation_axis_probe') AS BIGINT) AS n_relation_axis_probe,
			   CAST(json_extract(c.value_json, '$.observed.probe_counts.scope_probe') AS BIGINT) AS n_scope_probe,
			   pr.probe_slot_review_slots,
			   pr.probe_slot_review_note,
			   pr.probe_slot_review_reviewer,
			   pr.probe_slot_reviewed_at,
			   COALESCE(pr.probe_slot_review_count, 0) AS probe_slot_review_count,
			   json_extract_string(c.value_json, '$.severity') AS severity,
			   json_extract_string(c.value_json, '$.reviewer_hypothesis') AS reviewer_hypothesis,
			   s.indra_type,
			   COALESCE(ags.agent_names, '') AS agent_names,
			   e.source_api,
			   e.pmid,
			   e.text,
			   COALESCE(json_extract_string(c.value_json, '$.observed.verdict'), json_extract_string(ss.output_json, '$.verdict')) AS verdict,
			   COALESCE(json_extract_string(c.value_json, '$.observed.confidence'), json_extract_string(ss.output_json, '$.confidence')) AS confidence,
			   COALESCE(CAST(json_extract(c.value_json, '$.observed.score') AS DOUBLE), CAST(json_extract(ss.output_json, '$.score') AS DOUBLE)) AS score,
			   COALESCE(
			     CAST(json_extract(c.value_json, '$.observed.residual') AS DOUBLE),
			     CASE
			       WHEN s.indra_belief IS NOT NULL AND json_extract(ss.output_json, '$.score') IS NOT NULL
			       THEN CAST(json_extract(ss.output_json, '$.score') AS DOUBLE) - s.indra_belief
			       ELSE NULL
			     END
			   ) AS residual
			 FROM scorer_step_correction c
			 LEFT JOIN scorer_step ss ON ss.step_hash = c.step_hash
			 LEFT JOIN statement s ON s.stmt_hash = c.stmt_hash
			 LEFT JOIN evidence e ON e.evidence_hash = c.evidence_hash
			 LEFT JOIN ags ON ags.stmt_hash = c.stmt_hash
			 LEFT JOIN latest_probe_slot_review pr ON pr.parent_correction_id=c.correction_id
			 WHERE c.run_id='${qr}'
			   AND c.status='open'
			   AND c.correction_kind='repair_candidate'
			   AND ${availablePredicate}
			 ORDER BY CASE c.status WHEN 'open' THEN 0 ELSE 1 END,
			          c.created_at DESC,
			          c.correction_id DESC
			 LIMIT ${limit}`
		);
		return {
			run_id,
			architecture,
			openCount,
			rows: repairRows,
			reruns,
			activeRerunIntents: activeRerunIntents.rows,
			activeRerunIntentCount: activeRerunIntents.totalCount,
			recoverableRerunIntents: recoverableRerunIntentsResult.rows,
			recoverableRerunIntentCount: recoverableRerunIntentsResult.totalCount,
			recoverableRerunIntentOffset: recoverableRerunIntentsResult.offset,
			recoverableRerunIntentLimit: recoveryLimit,
			limit
		};
	} finally {
		con.disconnectSync?.();
	}
}


export async function getActiveRepairRerunIntents(
	con: DuckDBConnection,
	parent_run_id: string,
	parent_architecture: string,
	intentArchitectureSql: string,
	lineageSqlOptions: RepairRerunLineageSqlOptions
): Promise<{ rows: RepairRerunActiveIntent[]; totalCount: number }> {
	const qp = sqlQuote(parent_run_id);
	const qa = sqlQuote(parent_architecture);
	const intentChildRunIdSql = repairRerunChildRunIdSql('i', lineageSqlOptions);
	const intentSourceDumpIdSql = repairRerunSourceDumpIdSql('i', lineageSqlOptions);
	const intentParentCorrectionIdSql = repairRerunParentCorrectionIdNumberSql('i', lineageSqlOptions);
	const toNumberArray = (value: unknown): number[] => Array.isArray(value)
		? value.map((id) => Number(id)).filter((id) => Number.isInteger(id) && id > 0)
		: [];
	const rawRows = await rows<{
		child_run_id: string | null;
		architecture: string | null;
		status: string | null;
		source_dump_id: string | null;
		correction_ids: unknown;
		n_candidates: number;
		has_child_run: number;
		total_groups: number;
		first_intent_at: string;
		last_intent_at: string;
	}>(
		con,
		`WITH active_intent AS (
			 SELECT
			   ${intentChildRunIdSql} AS child_run_id,
			   ${intentSourceDumpIdSql} AS source_dump_id,
			   COALESCE(
			     json_extract_string(i.value_json, '$.architecture'),
			     NULLIF(${intentArchitectureSql}, 'unknown'),
			     sr.architecture,
			     '${qa}'
			   ) AS architecture,
			   ${intentParentCorrectionIdSql} AS parent_correction_id,
			   COALESCE(sr.status, 'queued') AS status,
			   CASE WHEN sr.run_id IS NULL THEN 0 ELSE 1 END AS has_child_run,
			   i.created_at
			  FROM scorer_step_correction i
			  LEFT JOIN score_run sr
			    ON sr.run_id=${intentChildRunIdSql}
			   AND sr.parent_run_id='${qp}'
			 WHERE i.run_id='${qp}'
			   AND i.correction_kind='rerun_intent'
			   AND COALESCE(json_extract_string(i.value_json, '$.parent_run_id'), i.run_id)='${qp}'
			   AND (
			        (
			          sr.run_id IS NOT NULL
			          AND COALESCE(sr.status, 'running') NOT IN (${REPAIR_RERUN_TERMINAL_STATUS_SQL})
			        )
			        OR (
			          sr.run_id IS NULL
			          AND i.created_at >= CURRENT_TIMESTAMP - INTERVAL '${REPAIR_RERUN_QUEUED_INTENT_LOCK_MINUTES} minutes'
			        )
			   )
		),
		active_open AS (
			SELECT ai.*
			  FROM active_intent ai
			  JOIN scorer_step_correction c
			    ON c.correction_id=ai.parent_correction_id
			   AND c.run_id='${qp}'
			   AND c.correction_kind='repair_candidate'
			   AND c.status='open'
			 WHERE ai.child_run_id IS NOT NULL
			   AND ai.source_dump_id IS NOT NULL
			   AND ai.parent_correction_id IS NOT NULL
		)
		,
		grouped AS (
			 SELECT child_run_id,
			        architecture,
			        status,
			        source_dump_id,
			        MAX(has_child_run) AS has_child_run,
			        LIST(parent_correction_id ORDER BY parent_correction_id) AS correction_ids,
			        COUNT(*) AS n_candidates,
			        MIN(created_at)::VARCHAR AS first_intent_at,
			        MAX(created_at)::VARCHAR AS last_intent_at
			   FROM active_open
			  GROUP BY child_run_id, architecture, status, source_dump_id
		)
		 SELECT *,
		        COUNT(*) OVER () AS total_groups
		   FROM grouped
		  ORDER BY last_intent_at DESC, child_run_id, source_dump_id
		  LIMIT 20`
	);
	return {
		rows: rawRows.map((row) => ({
			child_run_id: String(row.child_run_id ?? ''),
			architecture: String(row.architecture ?? parent_architecture),
			status: String(row.status ?? 'running'),
			source_dump_id: String(row.source_dump_id ?? ''),
			correction_ids: toNumberArray(row.correction_ids),
			n_candidates: Number(row.n_candidates ?? 0),
			has_child_run: Number(row.has_child_run ?? 0) > 0,
			first_intent_at: row.first_intent_at,
			last_intent_at: row.last_intent_at
		})),
		totalCount: Number(rawRows[0]?.total_groups ?? 0)
	};
}


export async function getRecoverableRepairRerunIntents(
	con: DuckDBConnection,
	parent_run_id: string,
	parent_architecture: string,
	intentArchitectureSql: string,
	lineageSqlOptions: RepairRerunLineageSqlOptions,
	offset: number,
	limit: number
): Promise<{ rows: RepairRerunRecoveryIntent[]; totalCount: number; offset: number }> {
	const qp = sqlQuote(parent_run_id);
	const qa = sqlQuote(parent_architecture);
	const intentChildRunIdSql = repairRerunChildRunIdSql('i', lineageSqlOptions);
	const intentSourceDumpIdSql = repairRerunSourceDumpIdSql('i', lineageSqlOptions);
	const intentParentCorrectionIdSql = repairRerunParentCorrectionIdNumberSql('i', lineageSqlOptions);
	const markerParentCorrectionIdSql = repairRerunParentCorrectionIdStringSql('rc', lineageSqlOptions);
	const markerChildRunIdSql = repairRerunChildRunIdSql('rc', lineageSqlOptions);
	const toNumberArray = (value: unknown): number[] => Array.isArray(value)
		? value.map((id) => Number(id)).filter((id) => Number.isInteger(id) && id > 0)
		: [];
	const baseCte = `WITH intent_raw AS (
		 SELECT
		   ${intentChildRunIdSql} AS child_run_id,
		   ${intentSourceDumpIdSql} AS source_dump_id,
		   COALESCE(
		     json_extract_string(i.value_json, '$.architecture'),
		     NULLIF(${intentArchitectureSql}, 'unknown'),
		     sr.architecture,
		     '${qa}'
		   ) AS architecture,
		   COALESCE(json_extract_string(i.value_json, '$.scoring_mode'), 'aggregate') AS scoring_mode,
		   COALESCE(json_extract_string(i.value_json, '$.probe_step_filter_csv'), '') AS probe_step_filter_csv,
		   ${intentParentCorrectionIdSql} AS parent_correction_id,
		   sr.status,
		   i.created_at
		  FROM scorer_step_correction i
		  JOIN score_run sr
		    ON sr.run_id=${intentChildRunIdSql}
		   AND sr.parent_run_id='${qp}'
		   AND sr.status='succeeded'
		 WHERE i.run_id='${qp}'
		   AND i.correction_kind='rerun_intent'
		   AND COALESCE(json_extract_string(i.value_json, '$.parent_run_id'), i.run_id)='${qp}'
	),
	intent AS (
		SELECT child_run_id,
		       source_dump_id,
		       architecture,
		       scoring_mode,
		       probe_step_filter_csv,
		       parent_correction_id,
		       status,
		       MIN(created_at) AS first_intent_at,
		       MAX(created_at) AS last_intent_at
		  FROM intent_raw
		 WHERE child_run_id IS NOT NULL
		   AND source_dump_id IS NOT NULL
		   AND parent_correction_id IS NOT NULL
		 GROUP BY child_run_id, source_dump_id, architecture, scoring_mode, probe_step_filter_csv, parent_correction_id, status
	),
	missing_marker AS (
		SELECT *
		  FROM intent i
		 WHERE NOT EXISTS (
		     SELECT 1
		       FROM scorer_step_correction rc
		      WHERE (
		             rc.correction_kind='rerun_child'
		          OR (
		             rc.correction_kind='rerun_uncovered'
		             AND ${markerChildRunIdSql}=i.child_run_id
		          )
		        )
		        AND ${markerParentCorrectionIdSql}=CAST(i.parent_correction_id AS VARCHAR)
		   )
	),
	coverage AS (
		SELECT i.*,
		       CASE
		         WHEN i.scoring_mode='probe_only'
		          AND c.evidence_hash IS NOT NULL
		          AND i.probe_step_filter_csv <> ''
		          AND (POSITION('subject_role_probe' IN i.probe_step_filter_csv)=0 OR EXISTS (
		                 SELECT 1 FROM scorer_step child
		                  WHERE child.run_id=i.child_run_id
		                    AND child.step_kind='subject_role_probe'
		                    AND child.stmt_hash=c.stmt_hash
		                    AND child.evidence_hash=c.evidence_hash
		              ))
		          AND (POSITION('object_role_probe' IN i.probe_step_filter_csv)=0 OR EXISTS (
		                 SELECT 1 FROM scorer_step child
		                  WHERE child.run_id=i.child_run_id
		                    AND child.step_kind='object_role_probe'
		                    AND child.stmt_hash=c.stmt_hash
		                    AND child.evidence_hash=c.evidence_hash
		              ))
		          AND (POSITION('relation_axis_probe' IN i.probe_step_filter_csv)=0 OR EXISTS (
		                 SELECT 1 FROM scorer_step child
		                  WHERE child.run_id=i.child_run_id
		                    AND child.step_kind='relation_axis_probe'
		                    AND child.stmt_hash=c.stmt_hash
		                    AND child.evidence_hash=c.evidence_hash
		              ))
		          AND (POSITION('scope_probe' IN i.probe_step_filter_csv)=0 OR EXISTS (
		                 SELECT 1 FROM scorer_step child
		                  WHERE child.run_id=i.child_run_id
		                    AND child.step_kind='scope_probe'
		                    AND child.stmt_hash=c.stmt_hash
		                    AND child.evidence_hash=c.evidence_hash
		              )) THEN 1
		         WHEN i.scoring_mode<>'probe_only' AND c.evidence_hash IS NOT NULL AND cs.step_hash IS NOT NULL THEN 1
		         WHEN c.evidence_hash IS NULL
		          AND i.scoring_mode<>'probe_only'
		          AND EXISTS (
		            SELECT 1 FROM statement_evidence pe
		             WHERE pe.stmt_hash=c.stmt_hash
		          )
		          AND NOT EXISTS (
		            SELECT 1
		              FROM statement_evidence pe
		             WHERE pe.stmt_hash=c.stmt_hash
		               AND NOT EXISTS (
		                 SELECT 1
		                   FROM scorer_step child
		                  WHERE child.run_id=i.child_run_id
		                    AND child.step_kind='aggregate'
		                    AND child.stmt_hash=pe.stmt_hash
		                    AND child.evidence_hash=pe.evidence_hash
		               )
		          ) THEN 1
		         ELSE 0
		       END AS child_covers
		  FROM missing_marker i
		  JOIN scorer_step_correction c
		    ON c.correction_id=i.parent_correction_id
		   AND c.run_id='${qp}'
		   AND c.correction_kind='repair_candidate'
		  LEFT JOIN scorer_step cs
		    ON cs.run_id=i.child_run_id
		   AND cs.step_kind='aggregate'
		   AND cs.stmt_hash=c.stmt_hash
		   AND cs.evidence_hash=c.evidence_hash
	),
	grouped AS (
		SELECT child_run_id,
		       architecture,
		       status,
		       source_dump_id,
		       LIST(parent_correction_id ORDER BY parent_correction_id) FILTER (WHERE child_covers=1) AS correction_ids,
		       LIST(parent_correction_id ORDER BY parent_correction_id) FILTER (WHERE child_covers=0) AS uncovered_correction_ids,
		       SUM(child_covers) AS n_candidates,
		       COUNT(*) AS n_missing_markers,
		       SUM(CASE WHEN child_covers=0 THEN 1 ELSE 0 END) AS n_uncovered_candidates,
		       MIN(first_intent_at)::VARCHAR AS first_intent_at,
		       MAX(last_intent_at)::VARCHAR AS last_intent_at
		  FROM coverage
		 GROUP BY child_run_id, architecture, status, source_dump_id
	)`;
	const rawRows = await rows<{
		child_run_id: string | null;
		architecture: string | null;
		status: string | null;
		source_dump_id: string | null;
		correction_ids: unknown;
		uncovered_correction_ids: unknown;
		n_candidates: number;
		n_missing_markers: number;
		n_uncovered_candidates: number;
		total_groups: number;
		page_offset: number;
		first_intent_at: string;
		last_intent_at: string;
	}>(
		con,
		`${baseCte},
		numbered AS (
			SELECT *,
			       COUNT(*) OVER () AS total_groups,
			       ROW_NUMBER() OVER (
			         ORDER BY last_intent_at DESC, child_run_id, source_dump_id
			       ) AS row_number
			  FROM grouped
		),
		windowed AS (
			SELECT *,
			       CASE
			         WHEN ${offset} >= total_groups
			         THEN CAST(FLOOR((CAST(total_groups AS DOUBLE) - 1) / ${limit}) AS BIGINT) * ${limit}
			         ELSE CAST(FLOOR(CAST(${offset} AS DOUBLE) / ${limit}) AS BIGINT) * ${limit}
			       END AS page_offset
			  FROM numbered
		)
		 SELECT child_run_id,
		       architecture,
		       status,
		       source_dump_id,
		       correction_ids,
		       uncovered_correction_ids,
		       n_candidates,
		       n_missing_markers,
		       n_uncovered_candidates,
		       total_groups,
		       page_offset,
		       first_intent_at,
		       last_intent_at
		  FROM windowed
		 WHERE row_number > page_offset
		   AND row_number <= page_offset + ${limit}
		 ORDER BY row_number`
	);
	const mapped = rawRows
		.map((row) => ({
			child_run_id: String(row.child_run_id ?? ''),
			architecture: String(row.architecture ?? parent_architecture),
			status: String(row.status ?? 'succeeded'),
			source_dump_id: String(row.source_dump_id ?? ''),
			correction_ids: toNumberArray(row.correction_ids),
			uncovered_correction_ids: toNumberArray(row.uncovered_correction_ids),
			n_candidates: Number(row.n_candidates ?? 0),
			n_missing_markers: Number(row.n_missing_markers ?? 0),
			n_uncovered_candidates: Number(row.n_uncovered_candidates ?? 0),
			first_intent_at: row.first_intent_at,
			last_intent_at: row.last_intent_at
		}));
	return {
		rows: mapped,
		totalCount: Number(rawRows[0]?.total_groups ?? 0),
		offset: Number(rawRows[0]?.page_offset ?? 0)
	};
}


export async function getRepairRerunComparisons(
	con: DuckDBConnection,
	parent_run_id: string,
	parent_architecture: string
): Promise<RepairRerunComparison[]> {
	const qp = sqlQuote(parent_run_id);
	const hasCorrectionTable = await scalar(
		con,
		`SELECT COUNT(*)
		   FROM information_schema.tables
		  WHERE table_name='scorer_step_correction'`
	);
	const lineageSqlOptions = hasCorrectionTable > 0
		? await repairRerunLineageSqlOptions(con)
		: { typedLineage: false };
	const markerParentCorrectionIdSql = repairRerunParentCorrectionIdNumberSql('m', lineageSqlOptions);
	const markerChildRunIdSql = repairRerunChildRunIdSql('m', lineageSqlOptions);
	const childRows = await rows<{
		run_id: string;
		architecture: string | null;
		status: string;
		started_at: string | null;
		model_id_default: string | null;
		cost_estimate_usd: number | null;
		cost_actual_usd: number | null;
	}>(
		con,
		`SELECT run_id,
		        COALESCE(architecture, 'unknown') AS architecture,
		        status,
		        started_at::VARCHAR AS started_at,
		        model_id_default,
		        cost_estimate_usd,
		        cost_actual_usd
		   FROM score_run
		  WHERE parent_run_id='${qp}'
		  ORDER BY started_at DESC NULLS LAST, run_id
		  LIMIT 20`
	);
	const out: RepairRerunComparison[] = [];
	for (const child of childRows) {
		const childArch = child.architecture ?? 'unknown';
		const base = {
			run_id: child.run_id,
			architecture: childArch,
			status: child.status,
			started_at: child.started_at,
			model_id_default: child.model_id_default,
			cost_estimate_usd: child.cost_estimate_usd == null ? null : Number(child.cost_estimate_usd),
			cost_actual_usd: child.cost_actual_usd == null ? null : Number(child.cost_actual_usd)
		};
		if (childArch !== parent_architecture) {
			out.push({
				...base,
				n_overlap_evidences: 0,
				n_score_evidences: 0,
				n_verdict_evidences: 0,
				parent_mae: null,
				child_mae: null,
				parent_bias: null,
				child_bias: null,
				verdicts_moved_total: 0,
				verdicts_moved_to_correct: 0,
				verdicts_moved_to_incorrect: 0,
				n_candidate_evidences: 0,
				n_parent_aggregate_candidates: 0,
				n_child_covered_candidates: 0,
				n_new_child_aggregate_candidates: 0,
				candidate_lanes: [],
				not_defined_reason: `before/after repair deltas are not defined across architectures (${parent_architecture} vs ${childArch}); use a paired workbench for architecture comparison`
			});
			continue;
		}
		const qc = sqlQuote(child.run_id);
		let candidateSummary = {
			n_candidate_evidences: 0,
			n_parent_aggregate_candidates: 0,
			n_child_covered_candidates: 0,
			n_new_child_aggregate_candidates: 0
		};
		let candidateLanes: RepairRerunCandidateLane[] = [];
		if (hasCorrectionTable > 0) {
			const candidateSummaryRows = await rows<typeof candidateSummary>(
				con,
				`WITH candidate AS (
					SELECT
					   c.correction_id,
					   c.stmt_hash,
					   c.evidence_hash,
					   p.step_hash AS parent_step_hash,
					   child.step_hash AS child_step_hash
					 FROM scorer_step_correction m
					 JOIN scorer_step_correction c
					   ON c.correction_id=${markerParentCorrectionIdSql}
					  AND c.correction_kind='repair_candidate'
					  AND c.run_id='${qp}'
					 LEFT JOIN scorer_step p
					   ON p.run_id='${qp}'
					  AND p.step_kind='aggregate'
					  AND p.stmt_hash=c.stmt_hash
					  AND p.evidence_hash=c.evidence_hash
					 LEFT JOIN scorer_step child
					   ON child.run_id='${qc}'
					  AND child.step_kind='aggregate'
					  AND child.stmt_hash=c.stmt_hash
					  AND child.evidence_hash=c.evidence_hash
					 WHERE m.correction_kind='rerun_child'
					   AND m.run_id='${qp}'
					   AND ${markerChildRunIdSql}='${qc}'
				)
				SELECT
				   COUNT(*) AS n_candidate_evidences,
				   SUM(CASE WHEN parent_step_hash IS NOT NULL THEN 1 ELSE 0 END) AS n_parent_aggregate_candidates,
				   SUM(CASE WHEN child_step_hash IS NOT NULL THEN 1 ELSE 0 END) AS n_child_covered_candidates,
				   SUM(CASE WHEN parent_step_hash IS NULL AND child_step_hash IS NOT NULL THEN 1 ELSE 0 END) AS n_new_child_aggregate_candidates
				 FROM candidate`
			);
			candidateSummary = {
				n_candidate_evidences: Number(candidateSummaryRows[0]?.n_candidate_evidences ?? 0),
				n_parent_aggregate_candidates: Number(candidateSummaryRows[0]?.n_parent_aggregate_candidates ?? 0),
				n_child_covered_candidates: Number(candidateSummaryRows[0]?.n_child_covered_candidates ?? 0),
				n_new_child_aggregate_candidates: Number(candidateSummaryRows[0]?.n_new_child_aggregate_candidates ?? 0)
			};
			const candidateRows = await rows<{
				correction_id: number;
				stmt_hash: string;
				evidence_hash: string | null;
				indra_type: string | null;
				source_api: string | null;
				suspected_step_kind: string | null;
				parent_verdict: string | null;
				child_verdict: string | null;
				parent_score: number | null;
				child_score: number | null;
				indra_belief: number | null;
				parent_abs_error: number | null;
				child_abs_error: number | null;
				abs_error_delta: number | null;
				movement_rank: number;
			}>(
				con,
				`WITH candidate AS (
					SELECT
					   c.correction_id,
					   c.stmt_hash,
					   c.evidence_hash,
					   json_extract_string(c.value_json, '$.suspected_step_kind') AS suspected_step_kind,
					   s.indra_type,
					   s.indra_belief,
					   e.source_api,
					   CAST(json_extract(p.output_json, '$.score') AS DOUBLE) AS parent_score,
					   CAST(json_extract(child.output_json, '$.score') AS DOUBLE) AS child_score,
					   json_extract_string(p.output_json, '$.verdict') AS parent_verdict,
					   json_extract_string(child.output_json, '$.verdict') AS child_verdict
					 FROM scorer_step_correction m
					 JOIN scorer_step_correction c
					   ON c.correction_id=${markerParentCorrectionIdSql}
					  AND c.correction_kind='repair_candidate'
					  AND c.run_id='${qp}'
					 LEFT JOIN scorer_step p
					   ON p.run_id='${qp}'
					  AND p.step_kind='aggregate'
					  AND p.stmt_hash=c.stmt_hash
					  AND p.evidence_hash=c.evidence_hash
					 LEFT JOIN scorer_step child
					   ON child.run_id='${qc}'
					  AND child.step_kind='aggregate'
					  AND child.stmt_hash=c.stmt_hash
					  AND child.evidence_hash=c.evidence_hash
					 LEFT JOIN statement s ON s.stmt_hash=c.stmt_hash
					 LEFT JOIN evidence e ON e.evidence_hash=c.evidence_hash
					 WHERE m.correction_kind='rerun_child'
					   AND m.run_id='${qp}'
					   AND ${markerChildRunIdSql}='${qc}'
				),
				scored AS (
					SELECT *,
					   CASE WHEN indra_belief IS NOT NULL AND parent_score IS NOT NULL THEN ABS(parent_score - indra_belief) ELSE NULL END AS parent_abs_error,
					   CASE WHEN indra_belief IS NOT NULL AND child_score IS NOT NULL THEN ABS(child_score - indra_belief) ELSE NULL END AS child_abs_error,
					   CASE
					     WHEN indra_belief IS NOT NULL AND parent_score IS NOT NULL AND child_score IS NOT NULL
					     THEN ABS(child_score - indra_belief) - ABS(parent_score - indra_belief)
					     ELSE NULL
					   END AS abs_error_delta,
					   CASE
					     WHEN parent_score IS NULL AND child_score IS NOT NULL THEN 0
					     WHEN parent_verdict IS NOT NULL AND child_verdict IS NOT NULL AND parent_verdict <> 'correct' AND child_verdict='correct' THEN 1
					     WHEN parent_verdict IS NOT NULL AND child_verdict IS NOT NULL AND parent_verdict <> 'incorrect' AND child_verdict='incorrect' THEN 2
					     WHEN indra_belief IS NOT NULL AND parent_score IS NOT NULL AND child_score IS NOT NULL AND ABS(child_score - indra_belief) < ABS(parent_score - indra_belief) THEN 3
					     WHEN indra_belief IS NOT NULL AND parent_score IS NOT NULL AND child_score IS NOT NULL AND ABS(child_score - indra_belief) > ABS(parent_score - indra_belief) THEN 4
					     ELSE 9
					   END AS movement_rank
					FROM candidate
				)
				SELECT *
				FROM scored
				ORDER BY movement_rank,
				         ABS(COALESCE(abs_error_delta, 0)) DESC,
				         correction_id
				LIMIT 8`
			);
			candidateLanes = candidateRows.map((row) => {
				const parentScore = row.parent_score == null ? null : Number(row.parent_score);
				const childScore = row.child_score == null ? null : Number(row.child_score);
				const parentAbs = row.parent_abs_error == null ? null : Number(row.parent_abs_error);
				const childAbs = row.child_abs_error == null ? null : Number(row.child_abs_error);
				const absDelta = row.abs_error_delta == null ? null : Number(row.abs_error_delta);
				let movement = 'unchanged';
				if (parentScore == null && childScore != null) movement = 'new_child_aggregate';
				else if (row.parent_verdict && row.child_verdict && row.parent_verdict !== 'correct' && row.child_verdict === 'correct') movement = 'verdict_to_correct';
				else if (row.parent_verdict && row.child_verdict && row.parent_verdict !== 'incorrect' && row.child_verdict === 'incorrect') movement = 'verdict_to_incorrect';
				else if (absDelta != null && absDelta < 0) movement = 'score_improved';
				else if (absDelta != null && absDelta > 0) movement = 'score_regressed';
				return {
					correction_id: Number(row.correction_id),
					stmt_hash: String(row.stmt_hash),
					evidence_hash: row.evidence_hash == null ? null : String(row.evidence_hash),
					indra_type: row.indra_type == null ? null : String(row.indra_type),
					source_api: row.source_api == null ? null : String(row.source_api),
					suspected_step_kind: row.suspected_step_kind == null ? null : String(row.suspected_step_kind),
					parent_verdict: row.parent_verdict == null ? null : String(row.parent_verdict),
					child_verdict: row.child_verdict == null ? null : String(row.child_verdict),
					parent_score: parentScore,
					child_score: childScore,
					indra_belief: row.indra_belief == null ? null : Number(row.indra_belief),
					parent_abs_error: parentAbs,
					child_abs_error: childAbs,
					abs_error_delta: absDelta,
					movement
				};
			});
		}
		const metricRows = await rows<{
			n_overlap_evidences: number;
			n_score_evidences: number;
			n_verdict_evidences: number;
			parent_mae: number | null;
			child_mae: number | null;
			parent_bias: number | null;
			child_bias: number | null;
			verdicts_moved_total: number;
			verdicts_moved_to_correct: number;
			verdicts_moved_to_incorrect: number;
		}>(
			con,
			`WITH overlap AS (
				SELECT
				   p.stmt_hash,
				   p.evidence_hash,
				   CAST(json_extract(p.output_json, '$.score') AS DOUBLE) AS parent_score,
				   CAST(json_extract(c.output_json, '$.score') AS DOUBLE) AS child_score,
				   json_extract_string(p.output_json, '$.verdict') AS parent_verdict,
				   json_extract_string(c.output_json, '$.verdict') AS child_verdict,
				   s.indra_belief
				 FROM scorer_step p
				 JOIN scorer_step c
				   ON c.stmt_hash=p.stmt_hash
				  AND c.evidence_hash=p.evidence_hash
				  AND c.step_kind='aggregate'
				 JOIN statement s ON s.stmt_hash=p.stmt_hash
				 WHERE p.run_id='${qp}'
				   AND c.run_id='${qc}'
				   AND p.step_kind='aggregate'
				   AND p.evidence_hash IS NOT NULL
			)
			SELECT
			   COUNT(*) AS n_overlap_evidences,
			   SUM(CASE WHEN indra_belief IS NOT NULL AND parent_score IS NOT NULL AND child_score IS NOT NULL THEN 1 ELSE 0 END) AS n_score_evidences,
			   SUM(CASE WHEN parent_verdict IS NOT NULL AND child_verdict IS NOT NULL THEN 1 ELSE 0 END) AS n_verdict_evidences,
			   AVG(CASE WHEN indra_belief IS NOT NULL AND parent_score IS NOT NULL AND child_score IS NOT NULL THEN ABS(parent_score - indra_belief) ELSE NULL END) AS parent_mae,
			   AVG(CASE WHEN indra_belief IS NOT NULL AND parent_score IS NOT NULL AND child_score IS NOT NULL THEN ABS(child_score - indra_belief) ELSE NULL END) AS child_mae,
			   AVG(CASE WHEN indra_belief IS NOT NULL AND parent_score IS NOT NULL AND child_score IS NOT NULL THEN parent_score - indra_belief ELSE NULL END) AS parent_bias,
			   AVG(CASE WHEN indra_belief IS NOT NULL AND parent_score IS NOT NULL AND child_score IS NOT NULL THEN child_score - indra_belief ELSE NULL END) AS child_bias,
			   SUM(CASE WHEN parent_verdict IS NOT NULL AND child_verdict IS NOT NULL AND parent_verdict <> child_verdict THEN 1 ELSE 0 END) AS verdicts_moved_total,
			   SUM(CASE WHEN parent_verdict IS NOT NULL AND child_verdict IS NOT NULL AND parent_verdict <> 'correct' AND child_verdict='correct' THEN 1 ELSE 0 END) AS verdicts_moved_to_correct,
			   SUM(CASE WHEN parent_verdict IS NOT NULL AND child_verdict IS NOT NULL AND parent_verdict <> 'incorrect' AND child_verdict='incorrect' THEN 1 ELSE 0 END) AS verdicts_moved_to_incorrect
			 FROM overlap`
		);
		const m = metricRows[0] ?? null;
		const nOverlap = Number(m?.n_overlap_evidences ?? 0);
		out.push({
			...base,
			n_overlap_evidences: nOverlap,
			n_score_evidences: Number(m?.n_score_evidences ?? 0),
			n_verdict_evidences: Number(m?.n_verdict_evidences ?? 0),
			parent_mae: m?.parent_mae == null ? null : Number(m.parent_mae),
			child_mae: m?.child_mae == null ? null : Number(m.child_mae),
			parent_bias: m?.parent_bias == null ? null : Number(m.parent_bias),
			child_bias: m?.child_bias == null ? null : Number(m.child_bias),
			verdicts_moved_total: Number(m?.verdicts_moved_total ?? 0),
			verdicts_moved_to_correct: Number(m?.verdicts_moved_to_correct ?? 0),
			verdicts_moved_to_incorrect: Number(m?.verdicts_moved_to_incorrect ?? 0),
			...candidateSummary,
			candidate_lanes: candidateLanes,
			not_defined_reason: nOverlap > 0 || candidateSummary.n_child_covered_candidates > 0
				? null
				: 'before/after repair deltas need overlapping aggregate evidence rows between parent and child run'
		});
	}
	return out;
}


export async function getRepairRerunDetail(
	parent_run_id: string,
	child_run_id: string,
	options: { offset?: number | null; limit?: number | null } = {}
): Promise<RepairRerunDetail | null> {
	if (!dbExists()) return null;
	const offset = boundedNonNegativeInteger(options.offset, 0);
	const limit = Math.min(200, Math.max(1, boundedNonNegativeInteger(options.limit, 100)));
	const con = await connect();
	try {
		const qp = sqlQuote(parent_run_id);
		const qc = sqlQuote(child_run_id);
		const runRows = await rows<{
			parent_architecture: string | null;
			run_id: string;
			architecture: string | null;
			status: string;
			started_at: string | null;
			model_id_default: string | null;
			cost_estimate_usd: number | null;
			cost_actual_usd: number | null;
		}>(
			con,
			`SELECT
			   COALESCE(p.architecture, 'unknown') AS parent_architecture,
			   c.run_id,
			   COALESCE(c.architecture, 'unknown') AS architecture,
			   c.status,
			   c.started_at::VARCHAR AS started_at,
			   c.model_id_default,
			   c.cost_estimate_usd,
			   c.cost_actual_usd
			 FROM score_run p
			 JOIN score_run c
			   ON c.parent_run_id=p.run_id
			  AND c.run_id='${qc}'
			 WHERE p.run_id='${qp}'
			 LIMIT 1`
		);
		const child = runRows[0];
		if (!child) return null;
		const parentArchitecture = child.parent_architecture ?? 'unknown';
		const childArch = child.architecture ?? 'unknown';
		const base = {
			run_id: child.run_id,
			parent_run_id,
			child_run_id: child.run_id,
			parent_architecture: parentArchitecture,
			architecture: childArch,
			status: child.status,
			started_at: child.started_at,
			model_id_default: child.model_id_default,
			cost_estimate_usd: child.cost_estimate_usd == null ? null : Number(child.cost_estimate_usd),
			cost_actual_usd: child.cost_actual_usd == null ? null : Number(child.cost_actual_usd),
			candidate_lane_offset: offset,
			candidate_lane_limit: limit
		};
		if (childArch !== parentArchitecture) {
			return {
				...base,
				n_overlap_evidences: 0,
				n_score_evidences: 0,
				n_verdict_evidences: 0,
				parent_mae: null,
				child_mae: null,
				parent_bias: null,
				child_bias: null,
				verdicts_moved_total: 0,
				verdicts_moved_to_correct: 0,
				verdicts_moved_to_incorrect: 0,
				n_candidate_evidences: 0,
				n_parent_aggregate_candidates: 0,
				n_child_covered_candidates: 0,
				n_new_child_aggregate_candidates: 0,
				candidate_lanes: [],
				candidate_lane_total: 0,
				not_defined_reason: `before/after repair deltas are not defined across architectures (${parentArchitecture} vs ${childArch}); use a paired workbench for architecture comparison`
			};
		}

		let candidateSummary = {
			n_candidate_evidences: 0,
			n_parent_aggregate_candidates: 0,
			n_child_covered_candidates: 0,
			n_new_child_aggregate_candidates: 0
		};
		let candidateLanes: RepairRerunCandidateLane[] = [];
		const hasCorrectionTable = await scalar(
			con,
			`SELECT COUNT(*)
			   FROM information_schema.tables
			  WHERE table_name='scorer_step_correction'`
		);
		if (hasCorrectionTable > 0) {
			const lineageSqlOptions = await repairRerunLineageSqlOptions(con);
			const markerParentCorrectionIdSql = repairRerunParentCorrectionIdNumberSql('m', lineageSqlOptions);
			const markerChildRunIdSql = repairRerunChildRunIdSql('m', lineageSqlOptions);
			const candidateSummaryRows = await rows<typeof candidateSummary>(
				con,
				`WITH candidate AS (
					SELECT
					   c.correction_id,
					   c.stmt_hash,
					   c.evidence_hash,
					   p.step_hash AS parent_step_hash,
					   child.step_hash AS child_step_hash
					 FROM scorer_step_correction m
					 JOIN scorer_step_correction c
					   ON c.correction_id=${markerParentCorrectionIdSql}
					  AND c.correction_kind='repair_candidate'
					  AND c.run_id='${qp}'
					 LEFT JOIN scorer_step p
					   ON p.run_id='${qp}'
					  AND p.step_kind='aggregate'
					  AND p.stmt_hash=c.stmt_hash
					  AND p.evidence_hash=c.evidence_hash
					 LEFT JOIN scorer_step child
					   ON child.run_id='${qc}'
					  AND child.step_kind='aggregate'
					  AND child.stmt_hash=c.stmt_hash
					  AND child.evidence_hash=c.evidence_hash
					 WHERE m.correction_kind='rerun_child'
					   AND m.run_id='${qp}'
					   AND ${markerChildRunIdSql}='${qc}'
				)
				SELECT
				   COUNT(*) AS n_candidate_evidences,
				   SUM(CASE WHEN parent_step_hash IS NOT NULL THEN 1 ELSE 0 END) AS n_parent_aggregate_candidates,
				   SUM(CASE WHEN child_step_hash IS NOT NULL THEN 1 ELSE 0 END) AS n_child_covered_candidates,
				   SUM(CASE WHEN parent_step_hash IS NULL AND child_step_hash IS NOT NULL THEN 1 ELSE 0 END) AS n_new_child_aggregate_candidates
				 FROM candidate`
			);
			candidateSummary = {
				n_candidate_evidences: Number(candidateSummaryRows[0]?.n_candidate_evidences ?? 0),
				n_parent_aggregate_candidates: Number(candidateSummaryRows[0]?.n_parent_aggregate_candidates ?? 0),
				n_child_covered_candidates: Number(candidateSummaryRows[0]?.n_child_covered_candidates ?? 0),
				n_new_child_aggregate_candidates: Number(candidateSummaryRows[0]?.n_new_child_aggregate_candidates ?? 0)
			};
			const candidateRows = await rows<{
				correction_id: number;
				stmt_hash: string;
				evidence_hash: string | null;
				indra_type: string | null;
				source_api: string | null;
				suspected_step_kind: string | null;
				parent_verdict: string | null;
				child_verdict: string | null;
				parent_score: number | null;
				child_score: number | null;
				indra_belief: number | null;
				parent_abs_error: number | null;
				child_abs_error: number | null;
				abs_error_delta: number | null;
				movement_rank: number;
			}>(
				con,
				`WITH candidate AS (
					SELECT
					   c.correction_id,
					   c.stmt_hash,
					   c.evidence_hash,
					   json_extract_string(c.value_json, '$.suspected_step_kind') AS suspected_step_kind,
					   s.indra_type,
					   s.indra_belief,
					   e.source_api,
					   CAST(json_extract(p.output_json, '$.score') AS DOUBLE) AS parent_score,
					   CAST(json_extract(child.output_json, '$.score') AS DOUBLE) AS child_score,
					   json_extract_string(p.output_json, '$.verdict') AS parent_verdict,
					   json_extract_string(child.output_json, '$.verdict') AS child_verdict
					 FROM scorer_step_correction m
					 JOIN scorer_step_correction c
					   ON c.correction_id=${markerParentCorrectionIdSql}
					  AND c.correction_kind='repair_candidate'
					  AND c.run_id='${qp}'
					 LEFT JOIN scorer_step p
					   ON p.run_id='${qp}'
					  AND p.step_kind='aggregate'
					  AND p.stmt_hash=c.stmt_hash
					  AND p.evidence_hash=c.evidence_hash
					 LEFT JOIN scorer_step child
					   ON child.run_id='${qc}'
					  AND child.step_kind='aggregate'
					  AND child.stmt_hash=c.stmt_hash
					  AND child.evidence_hash=c.evidence_hash
					 LEFT JOIN statement s ON s.stmt_hash=c.stmt_hash
					 LEFT JOIN evidence e ON e.evidence_hash=c.evidence_hash
					 WHERE m.correction_kind='rerun_child'
					   AND m.run_id='${qp}'
					   AND ${markerChildRunIdSql}='${qc}'
				),
				scored AS (
					SELECT *,
					   CASE WHEN indra_belief IS NOT NULL AND parent_score IS NOT NULL THEN ABS(parent_score - indra_belief) ELSE NULL END AS parent_abs_error,
					   CASE WHEN indra_belief IS NOT NULL AND child_score IS NOT NULL THEN ABS(child_score - indra_belief) ELSE NULL END AS child_abs_error,
					   CASE
					     WHEN indra_belief IS NOT NULL AND parent_score IS NOT NULL AND child_score IS NOT NULL
					     THEN ABS(child_score - indra_belief) - ABS(parent_score - indra_belief)
					     ELSE NULL
					   END AS abs_error_delta,
					   CASE
					     WHEN parent_score IS NULL AND child_score IS NOT NULL THEN 0
					     WHEN parent_verdict IS NOT NULL AND child_verdict IS NOT NULL AND parent_verdict <> 'correct' AND child_verdict='correct' THEN 1
					     WHEN parent_verdict IS NOT NULL AND child_verdict IS NOT NULL AND parent_verdict <> 'incorrect' AND child_verdict='incorrect' THEN 2
					     WHEN indra_belief IS NOT NULL AND parent_score IS NOT NULL AND child_score IS NOT NULL AND ABS(child_score - indra_belief) < ABS(parent_score - indra_belief) THEN 3
					     WHEN indra_belief IS NOT NULL AND parent_score IS NOT NULL AND child_score IS NOT NULL AND ABS(child_score - indra_belief) > ABS(parent_score - indra_belief) THEN 4
					     ELSE 9
					   END AS movement_rank
					FROM candidate
				)
				SELECT *
				FROM scored
				ORDER BY movement_rank,
				         ABS(COALESCE(abs_error_delta, 0)) DESC,
				         correction_id
				LIMIT ${limit}
				OFFSET ${offset}`
			);
			candidateLanes = candidateRows.map((row) => {
				const parentScore = row.parent_score == null ? null : Number(row.parent_score);
				const childScore = row.child_score == null ? null : Number(row.child_score);
				const parentAbs = row.parent_abs_error == null ? null : Number(row.parent_abs_error);
				const childAbs = row.child_abs_error == null ? null : Number(row.child_abs_error);
				const absDelta = row.abs_error_delta == null ? null : Number(row.abs_error_delta);
				let movement = 'unchanged';
				if (parentScore == null && childScore != null) movement = 'new_child_aggregate';
				else if (row.parent_verdict && row.child_verdict && row.parent_verdict !== 'correct' && row.child_verdict === 'correct') movement = 'verdict_to_correct';
				else if (row.parent_verdict && row.child_verdict && row.parent_verdict !== 'incorrect' && row.child_verdict === 'incorrect') movement = 'verdict_to_incorrect';
				else if (absDelta != null && absDelta < 0) movement = 'score_improved';
				else if (absDelta != null && absDelta > 0) movement = 'score_regressed';
				return {
					correction_id: Number(row.correction_id),
					stmt_hash: String(row.stmt_hash),
					evidence_hash: row.evidence_hash == null ? null : String(row.evidence_hash),
					indra_type: row.indra_type == null ? null : String(row.indra_type),
					source_api: row.source_api == null ? null : String(row.source_api),
					suspected_step_kind: row.suspected_step_kind == null ? null : String(row.suspected_step_kind),
					parent_verdict: row.parent_verdict == null ? null : String(row.parent_verdict),
					child_verdict: row.child_verdict == null ? null : String(row.child_verdict),
					parent_score: parentScore,
					child_score: childScore,
					indra_belief: row.indra_belief == null ? null : Number(row.indra_belief),
					parent_abs_error: parentAbs,
					child_abs_error: childAbs,
					abs_error_delta: absDelta,
					movement
				};
			});
		}
		const metricRows = await rows<{
			n_overlap_evidences: number;
			n_score_evidences: number;
			n_verdict_evidences: number;
			parent_mae: number | null;
			child_mae: number | null;
			parent_bias: number | null;
			child_bias: number | null;
			verdicts_moved_total: number;
			verdicts_moved_to_correct: number;
			verdicts_moved_to_incorrect: number;
		}>(
			con,
			`WITH overlap AS (
				SELECT
				   p.stmt_hash,
				   p.evidence_hash,
				   CAST(json_extract(p.output_json, '$.score') AS DOUBLE) AS parent_score,
				   CAST(json_extract(c.output_json, '$.score') AS DOUBLE) AS child_score,
				   json_extract_string(p.output_json, '$.verdict') AS parent_verdict,
				   json_extract_string(c.output_json, '$.verdict') AS child_verdict,
				   s.indra_belief
				 FROM scorer_step p
				 JOIN scorer_step c
				   ON c.stmt_hash=p.stmt_hash
				  AND c.evidence_hash=p.evidence_hash
				  AND c.step_kind='aggregate'
				 JOIN statement s ON s.stmt_hash=p.stmt_hash
				 WHERE p.run_id='${qp}'
				   AND c.run_id='${qc}'
				   AND p.step_kind='aggregate'
				   AND p.evidence_hash IS NOT NULL
			)
			SELECT
			   COUNT(*) AS n_overlap_evidences,
			   SUM(CASE WHEN indra_belief IS NOT NULL AND parent_score IS NOT NULL AND child_score IS NOT NULL THEN 1 ELSE 0 END) AS n_score_evidences,
			   SUM(CASE WHEN parent_verdict IS NOT NULL AND child_verdict IS NOT NULL THEN 1 ELSE 0 END) AS n_verdict_evidences,
			   AVG(CASE WHEN indra_belief IS NOT NULL AND parent_score IS NOT NULL AND child_score IS NOT NULL THEN ABS(parent_score - indra_belief) ELSE NULL END) AS parent_mae,
			   AVG(CASE WHEN indra_belief IS NOT NULL AND parent_score IS NOT NULL AND child_score IS NOT NULL THEN ABS(child_score - indra_belief) ELSE NULL END) AS child_mae,
			   AVG(CASE WHEN indra_belief IS NOT NULL AND parent_score IS NOT NULL AND child_score IS NOT NULL THEN parent_score - indra_belief ELSE NULL END) AS parent_bias,
			   AVG(CASE WHEN indra_belief IS NOT NULL AND parent_score IS NOT NULL AND child_score IS NOT NULL THEN child_score - indra_belief ELSE NULL END) AS child_bias,
			   SUM(CASE WHEN parent_verdict IS NOT NULL AND child_verdict IS NOT NULL AND parent_verdict <> child_verdict THEN 1 ELSE 0 END) AS verdicts_moved_total,
			   SUM(CASE WHEN parent_verdict IS NOT NULL AND child_verdict IS NOT NULL AND parent_verdict <> 'correct' AND child_verdict='correct' THEN 1 ELSE 0 END) AS verdicts_moved_to_correct,
			   SUM(CASE WHEN parent_verdict IS NOT NULL AND child_verdict IS NOT NULL AND parent_verdict <> 'incorrect' AND child_verdict='incorrect' THEN 1 ELSE 0 END) AS verdicts_moved_to_incorrect
			 FROM overlap`
		);
		const m = metricRows[0] ?? null;
		const nOverlap = Number(m?.n_overlap_evidences ?? 0);
		return {
			...base,
			n_overlap_evidences: nOverlap,
			n_score_evidences: Number(m?.n_score_evidences ?? 0),
			n_verdict_evidences: Number(m?.n_verdict_evidences ?? 0),
			parent_mae: m?.parent_mae == null ? null : Number(m.parent_mae),
			child_mae: m?.child_mae == null ? null : Number(m.child_mae),
			parent_bias: m?.parent_bias == null ? null : Number(m.parent_bias),
			child_bias: m?.child_bias == null ? null : Number(m.child_bias),
			verdicts_moved_total: Number(m?.verdicts_moved_total ?? 0),
			verdicts_moved_to_correct: Number(m?.verdicts_moved_to_correct ?? 0),
			verdicts_moved_to_incorrect: Number(m?.verdicts_moved_to_incorrect ?? 0),
			...candidateSummary,
			candidate_lanes: candidateLanes,
			candidate_lane_total: candidateSummary.n_candidate_evidences,
			not_defined_reason: nOverlap > 0 || candidateSummary.n_child_covered_candidates > 0
				? null
				: 'before/after repair deltas need overlapping aggregate evidence rows between parent and child run'
		};
	} finally {
		con.disconnectSync?.();
	}
}
