const SQL_ALIAS_RE = /^[A-Za-z_][A-Za-z0-9_]*$/;
const INTERNAL_ALIASES = new Set(['repair_consumed', 'repair_child_run', 'repair_uncovered']);
export const REPAIR_RERUN_QUEUED_INTENT_LOCK_MINUTES = 15;
export const REPAIR_RERUN_TERMINAL_STATUS_SQL = "'succeeded', 'failed', 'canceled', 'cancelled', 'aborted'";

export interface RepairRerunLineageSqlOptions {
	typedLineage?: boolean;
}

export const REPAIR_RERUN_LINEAGE_DDL = `
ALTER TABLE scorer_step_correction ADD COLUMN IF NOT EXISTS parent_correction_id BIGINT;
ALTER TABLE scorer_step_correction ADD COLUMN IF NOT EXISTS child_run_id VARCHAR;
ALTER TABLE scorer_step_correction ADD COLUMN IF NOT EXISTS repair_source_dump_id VARCHAR;
CREATE INDEX IF NOT EXISTS idx_scorer_step_correction_parent_correction ON scorer_step_correction(parent_correction_id);
CREATE INDEX IF NOT EXISTS idx_scorer_step_correction_child_run ON scorer_step_correction(child_run_id);
CREATE INDEX IF NOT EXISTS idx_scorer_step_correction_repair_source_dump ON scorer_step_correction(repair_source_dump_id);
`;

function assertSqlAlias(alias: string, options: { allowInternal?: boolean } = {}): void {
	if (!SQL_ALIAS_RE.test(alias)) {
		throw new Error(`unsafe SQL alias: ${alias}`);
	}
	if (!options.allowInternal && INTERNAL_ALIASES.has(alias)) {
		throw new Error(`SQL alias collides with repair-rerun predicate internals: ${alias}`);
	}
}

function parentCorrectionIdStringSql(
	alias: string,
	options: RepairRerunLineageSqlOptions = {},
	aliasOptions: { allowInternal?: boolean } = {}
): string {
	assertSqlAlias(alias, aliasOptions);
	if (options.typedLineage) {
		return `COALESCE(CAST(${alias}.parent_correction_id AS VARCHAR), json_extract_string(${alias}.value_json, '$.parent_correction_id'))`;
	}
	return `json_extract_string(${alias}.value_json, '$.parent_correction_id')`;
}

export function repairRerunParentCorrectionIdStringSql(
	alias: string,
	options: RepairRerunLineageSqlOptions = {}
): string {
	return parentCorrectionIdStringSql(alias, options);
}

export function repairRerunParentCorrectionIdNumberSql(
	alias: string,
	options: RepairRerunLineageSqlOptions = {}
): string {
	assertSqlAlias(alias);
	if (options.typedLineage) {
		return `COALESCE(${alias}.parent_correction_id, TRY_CAST(json_extract_string(${alias}.value_json, '$.parent_correction_id') AS BIGINT))`;
	}
	return `TRY_CAST(json_extract_string(${alias}.value_json, '$.parent_correction_id') AS BIGINT)`;
}

function childRunIdSql(
	alias: string,
	options: RepairRerunLineageSqlOptions = {},
	aliasOptions: { allowInternal?: boolean } = {}
): string {
	assertSqlAlias(alias, aliasOptions);
	if (options.typedLineage) {
		return `COALESCE(NULLIF(${alias}.child_run_id, ''), json_extract_string(${alias}.value_json, '$.child_run_id'))`;
	}
	return `json_extract_string(${alias}.value_json, '$.child_run_id')`;
}

export function repairRerunChildRunIdSql(
	alias: string,
	options: RepairRerunLineageSqlOptions = {}
): string {
	return childRunIdSql(alias, options);
}

export function repairRerunSourceDumpIdSql(
	alias: string,
	options: RepairRerunLineageSqlOptions = {}
): string {
	assertSqlAlias(alias);
	if (options.typedLineage) {
		return `COALESCE(NULLIF(${alias}.repair_source_dump_id, ''), json_extract_string(${alias}.value_json, '$.source_dump_id'))`;
	}
	return `json_extract_string(${alias}.value_json, '$.source_dump_id')`;
}

export function repairCandidateConsumedExistsSql(
	alias: string,
	options: RepairRerunLineageSqlOptions = {}
): string {
	assertSqlAlias(alias);
	const consumedParentCorrectionIdSql = parentCorrectionIdStringSql('repair_consumed', options, {
		allowInternal: true
	});
	const consumedChildRunIdSql = childRunIdSql('repair_consumed', options, { allowInternal: true });
	const uncoveredParentCorrectionIdSql = parentCorrectionIdStringSql('repair_uncovered', options, {
		allowInternal: true
	});
	const uncoveredChildRunIdSql = childRunIdSql('repair_uncovered', options, { allowInternal: true });
	return `EXISTS (
			SELECT 1 FROM scorer_step_correction repair_consumed
			 WHERE (
			        repair_consumed.correction_kind='rerun_child'
			    OR (
			        repair_consumed.correction_kind='rerun_intent'
			        AND EXISTS (
			            SELECT 1 FROM score_run repair_child_run
			             WHERE repair_child_run.run_id=${consumedChildRunIdSql}
			               AND repair_child_run.parent_run_id=${alias}.run_id
			               AND repair_child_run.status='succeeded'
			        )
			        AND NOT EXISTS (
			            SELECT 1 FROM scorer_step_correction repair_uncovered
			             WHERE repair_uncovered.correction_kind='rerun_uncovered'
			               AND ${uncoveredParentCorrectionIdSql} = CAST(${alias}.correction_id AS VARCHAR)
			               AND ${uncoveredChildRunIdSql} = ${consumedChildRunIdSql}
			        )
			    )
			   )
			   AND ${consumedParentCorrectionIdSql} = CAST(${alias}.correction_id AS VARCHAR)
		)`;
}

export function repairCandidateActiveRerunIntentExistsSql(
	alias: string,
	options: RepairRerunLineageSqlOptions = {}
): string {
	assertSqlAlias(alias);
	const activeParentCorrectionIdSql = parentCorrectionIdStringSql('repair_consumed', options, {
		allowInternal: true
	});
	const activeChildRunIdSql = childRunIdSql('repair_consumed', options, { allowInternal: true });
	return `EXISTS (
			SELECT 1 FROM scorer_step_correction repair_consumed
			WHERE repair_consumed.correction_kind='rerun_intent'
			  AND ${activeParentCorrectionIdSql} = CAST(${alias}.correction_id AS VARCHAR)
			  AND (
			       EXISTS (
			           SELECT 1 FROM score_run repair_child_run
			            WHERE repair_child_run.run_id=${activeChildRunIdSql}
			              AND repair_child_run.parent_run_id=${alias}.run_id
			              AND COALESCE(repair_child_run.status, 'running') NOT IN (${REPAIR_RERUN_TERMINAL_STATUS_SQL})
			       )
			       OR (
			           NOT EXISTS (
			               SELECT 1 FROM score_run repair_child_run
			                WHERE repair_child_run.run_id=${activeChildRunIdSql}
			                  AND repair_child_run.parent_run_id=${alias}.run_id
			           )
			           AND repair_consumed.created_at >= CURRENT_TIMESTAMP - INTERVAL '${REPAIR_RERUN_QUEUED_INTENT_LOCK_MINUTES} minutes'
			       )
			  )
		)`;
}

export function repairCandidateUnavailableExistsSql(
	alias: string,
	options: RepairRerunLineageSqlOptions = {}
): string {
	return `(${repairCandidateConsumedExistsSql(alias, options)} OR ${repairCandidateActiveRerunIntentExistsSql(alias, options)})`;
}

export function repairCandidateUnconsumedPredicateSql(
	alias: string,
	options: RepairRerunLineageSqlOptions = {}
): string {
	return `NOT ${repairCandidateConsumedExistsSql(alias, options)}`;
}

export function repairCandidateAvailablePredicateSql(
	alias: string,
	options: RepairRerunLineageSqlOptions = {}
): string {
	return `NOT ${repairCandidateUnavailableExistsSql(alias, options)}`;
}
