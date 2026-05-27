import { error } from '@sveltejs/kit';
import type { DuckDBConnection, DuckDBValue } from '@duckdb/node-api';
import type { RunCohortFilters } from './runCohortTypes';
import {
	aggregateEvidenceTraceFidelityPredicate,
	cohortEvidenceLevelFilterCount,
	statementEvidenceTraceFidelityPredicate
} from './runCohortSql';

export interface RunCohortEmptyDiagnostic {
	kind: 'absent_in_run' | 'not_applicable' | 'no_intersection' | 'empty_run';
	filter: string | null;
	value: string | null;
	message: string;
}

export interface EmptyCohortPredicate {
	filter: string;
	value: string;
	predicateSql: string;
	params?: DuckDBValue[];
	message?: string;
}

export interface RunScorerStepCounts {
	total: number;
	aggregate: number;
	aggregateWithScore: number;
	nonaggregate: number;
}

function filterValue(value: unknown): string | null {
	if (value === null || value === undefined || value === false || value === '') return null;
	return String(value);
}

function countPhrase(n: number, noun: string): string {
	return `${n.toLocaleString()} ${noun}${n === 1 ? '' : 's'}`;
}

export async function runScorerStepCounts(
	con: DuckDBConnection,
	run_id: string
): Promise<RunScorerStepCounts> {
	const reader = await con.runAndReadAll(
		`SELECT
		   COUNT(*) AS total,
		   COUNT(*) FILTER (WHERE step_kind='aggregate') AS aggregate,
		   COUNT(*) FILTER (
		     WHERE step_kind='aggregate'
		       AND json_extract(output_json, '$.score') IS NOT NULL
		   ) AS aggregate_with_score,
		   COUNT(*) FILTER (WHERE step_kind <> 'aggregate') AS nonaggregate
		 FROM scorer_step
		 WHERE run_id=?`,
		[run_id]
	);
	const row = reader.getRowObjects()[0] ?? {};
	const asNumber = (v: unknown): number => typeof v === 'bigint' ? Number(v) : Number(v ?? 0);
	return {
		total: asNumber(row.total),
		aggregate: asNumber(row.aggregate),
		aggregateWithScore: asNumber(row.aggregate_with_score),
		nonaggregate: asNumber(row.nonaggregate)
	};
}

function emptyRunMessage(
	unit: string,
	scope: string,
	status: string,
	counts: RunScorerStepCounts | null = null
): string {
	const livePrefix = status === 'running'
		? scope.includes('snapshot')
			? `${scope} comes from a running run and `
			: `${scope} is still running and `
		: `${scope} `;
	if (counts && counts.total === 0) {
		const rowState = status === 'running' ? 'has no scorer_step rows yet' : 'has no scorer_step rows';
		return `${livePrefix}${rowState}; this cohort needs persisted scorer output before it can contain rows`;
	}
	if (unit === 'statement') {
		if (counts && counts.aggregate === 0 && counts.nonaggregate > 0) {
			return `${livePrefix}has ${countPhrase(counts.nonaggregate, 'persisted substep row')} but no aggregate scorer rows; statement cohorts summarize aggregate rollups with numeric scores`;
		}
		if (counts && counts.aggregate > 0 && counts.aggregateWithScore === 0) {
			return `${livePrefix}has ${countPhrase(counts.aggregate, 'aggregate scorer row')}, but none have numeric scores; statement cohorts need numeric aggregate scores`;
		}
		return `${livePrefix}has no statement cohort rows because scored aggregate rows do not join to statements`;
	}
	if (unit === 'aggregate evidence') {
		if (counts && counts.aggregate === 0 && counts.nonaggregate > 0) {
			return `${livePrefix}has ${countPhrase(counts.nonaggregate, 'persisted substep row')} but no aggregate evidence scorer rows; aggregate cohorts appear after aggregate rollup output is persisted`;
		}
		if (counts && counts.aggregate > 0) {
			return `${livePrefix}has ${countPhrase(counts.aggregate, 'aggregate scorer row')}, but none join to statement rows for this cohort grain`;
		}
		return `${livePrefix}has no aggregate evidence scorer rows; this cohort grain needs persisted aggregate scorer output`;
	}
	if (unit === 'trace evidence') {
		if (counts && counts.total > 0) {
			return `${livePrefix}has ${countPhrase(counts.total, 'scorer_step row')}, but no trace evidence rows in this trace-state plane`;
		}
		return `${livePrefix}has no trace evidence rows in this trace-state plane`;
	}
	return `${scope} has no ${unit} rows for this cohort grain`;
}

function absentFilterMessage(unit: string, filter: string, value: string, scope: string): string {
	if (filter === 'trace_state') {
		return `${scope} has no ${unit} rows with trace_state=${value}`;
	}
	if (filter === 'probe_coverage') {
		return `${scope} has no ${unit} rows with persisted substrate/probe slots`;
	}
	if (filter === 'step_kind') {
		return `${scope} has no ${unit} rows with step_kind=${value}`;
	}
	if (filter === 'verdict_present') {
		return `${scope} has no ${unit} rows with a persisted aggregate verdict`;
	}
	if (filter === 'score_present') {
		return `${scope} has no ${unit} rows with a persisted score`;
	}
	if (filter === 'indra_belief_present') {
		return `${scope} has no ${unit} rows with an INDRA belief anchor`;
	}
	if (filter === 'truth_set') {
		return `truth_set=${value} exists, but ${scope} has no ${unit} rows carrying that label`;
	}
	if (filter === 'multi_evidence') {
		if (unit === 'statement') {
			return `${scope} has no statement rows belonging to statements with more than one scored aggregate evidence row`;
		}
		if (unit === 'aggregate evidence') {
			return `${scope} has no aggregate evidence rows belonging to statements with more than one aggregate scorer row`;
		}
		if (unit === 'trace evidence') {
			return `${scope} has no trace evidence rows belonging to statements with more than one trace evidence row across the trace evidence plane`;
		}
		return `${scope} has no ${unit} rows belonging to statements with more than one evidence row`;
	}
	if (filter === 'supports' || filter === 'supports_compare') {
		return `${scope} has no ${unit} rows on statements with support graph edges`;
	}
	if (filter === 'trace_fidelity') {
		return `${scope} has no ${unit} rows with trace_fidelity=${value}`;
	}
	return `${scope} has no ${unit} rows with ${filter}=${value}`;
}

function noIntersectionMessage(unit: string, filters: RunCohortFilters): string {
	if (unit === 'statement' && cohortEvidenceLevelFilterCount(filters) >= 2) {
		return 'each active evidence-level filter appears in this run, but no statement row has one scored evidence row satisfying them together';
	}
	return `each active filter appears in this run, but no ${unit} rows satisfy the whole filter set together`;
}

function hasDiagnosticFilter(filters: RunCohortFilters): boolean {
	return Object.entries(filters).some(([key, value]) => {
			if (key === 'grain' || key === 'trace_snapshot') return false;
			return filterValue(value) !== null;
		});
	}

export async function emptyCohortDiagnostics(
	con: DuckDBConnection,
	baseSql: string,
	unit: string,
	filters: RunCohortFilters,
	predicates: EmptyCohortPredicate[],
	diagnostics: RunCohortEmptyDiagnostic[] = [],
	scope = 'this run',
	status = 'unknown',
	stepCounts: RunScorerStepCounts | null = null
): Promise<RunCohortEmptyDiagnostic[]> {
	const countCols = predicates.map((p, i) => (
		`SUM(CASE WHEN (${p.predicateSql}) THEN 1 ELSE 0 END) AS p_${i}`
	));
	const sql = `${baseSql}
		SELECT COUNT(*) AS base_n${countCols.length ? `, ${countCols.join(', ')}` : ''}
		FROM cohort_base cb`;
	const params = predicates.flatMap((p) => p.params ?? []);
	let countRow: Record<string, unknown> = {};
	try {
		const reader = await con.runAndReadAll(sql, params);
		countRow = reader.getRowObjects()[0] ?? {};
	} catch (e) {
		console.error('cohort empty-diagnostic rollup failed', e);
		throw error(500, 'cohort empty-diagnostic rollup failed');
	}
	const asNumber = (v: unknown): number => typeof v === 'bigint' ? Number(v) : Number(v ?? 0);
	const baseCount = asNumber(countRow.base_n);
	if (baseCount === 0) {
		return [
			...diagnostics,
			{
				kind: 'empty_run',
				filter: null,
				value: null,
				message: emptyRunMessage(unit, scope, status, stepCounts)
			}
		];
	}
	for (const [i, p] of predicates.entries()) {
		if (asNumber(countRow[`p_${i}`]) === 0) {
			diagnostics.push({
				kind: 'absent_in_run',
				filter: p.filter,
				value: p.value,
				message: p.message ?? absentFilterMessage(unit, p.filter, p.value, scope)
			});
		}
	}
	if (diagnostics.length === 0 && hasDiagnosticFilter(filters)) {
		diagnostics.push({
			kind: 'no_intersection',
			filter: null,
			value: null,
			message: noIntersectionMessage(unit, filters)
		});
	}
	return diagnostics;
}

export function statementEmptyPredicates(filters: RunCohortFilters): {
	predicates: EmptyCohortPredicate[];
	diagnostics: RunCohortEmptyDiagnostic[];
} {
	const predicates: EmptyCohortPredicate[] = [];
	const diagnostics: RunCohortEmptyDiagnostic[] = [];
	const add = (
		filter: string,
		value: unknown,
		predicateSql: string,
		params: DuckDBValue[] = [],
		message?: string
	) => {
		const v = filterValue(value);
		if (v) predicates.push({ filter, value: v, predicateSql, params, message });
	};
	add('verdict', filters.verdict, `EXISTS (
		SELECT 1 FROM scorer_step ss2
		WHERE ss2.run_id=cb.run_id
		  AND ss2.step_kind='aggregate'
		  AND ss2.stmt_hash=cb.stmt_hash
		  AND json_extract(ss2.output_json, '$.score') IS NOT NULL
		  AND COALESCE(json_extract_string(ss2.output_json, '$.verdict'), '')=?
)`, [filters.verdict ?? '']);
	if (filters.verdict_present) {
		add('verdict_present', true, `EXISTS (
			SELECT 1 FROM scorer_step ss2
			WHERE ss2.run_id=cb.run_id
			  AND ss2.step_kind='aggregate'
			  AND ss2.stmt_hash=cb.stmt_hash
			  AND json_extract(ss2.output_json, '$.score') IS NOT NULL
			  AND json_extract(ss2.output_json, '$.verdict') IS NOT NULL
	)`);
	}
	if (filters.score_present) add('score_present', true, 'cb.score IS NOT NULL');
	if (filters.indra_belief_present) add('indra_belief_present', true, 'cb.indra_belief IS NOT NULL');
	add('confidence', filters.confidence, `EXISTS (
		SELECT 1 FROM scorer_step ss2
		WHERE ss2.run_id=cb.run_id
		  AND ss2.step_kind='aggregate'
		  AND ss2.stmt_hash=cb.stmt_hash
		  AND json_extract(ss2.output_json, '$.score') IS NOT NULL
		  AND COALESCE(json_extract_string(ss2.output_json, '$.confidence'), '')=?
)`, [filters.confidence ?? '']);
	add('type', filters.type, 'cb.indra_type=?', [filters.type ?? '']);
	add('source', filters.source, `EXISTS (
		SELECT 1 FROM scorer_step ss2
		LEFT JOIN evidence e2 ON e2.evidence_hash=ss2.evidence_hash
		WHERE ss2.run_id=cb.run_id
		  AND ss2.step_kind='aggregate'
		  AND ss2.stmt_hash=cb.stmt_hash
		  AND json_extract(ss2.output_json, '$.score') IS NOT NULL
		  AND COALESCE(e2.source_api, '')=?
)`, [filters.source ?? ''], filters.source
		? `source=${filters.source} exists, but this run has no statement rows with scored aggregate evidence from that source; statement means still summarize the full statement, so use evidence grain for source-specific evidence rows`
		: undefined);
	add('source_stratum', filters.source_stratum, "COALESCE(cb.source_stratum, '')=?", [filters.source_stratum ?? ''], filters.source_stratum
		? `source_stratum=${filters.source_stratum} exists, but this run has no statement rows in that statement-level stratum; source_stratum is the statement-level minimum evidence source`
		: undefined);
	add('truth_set', filters.truth_set, `EXISTS (
		SELECT 1 FROM scorer_step ss2
		JOIN truth_label tl ON tl.target_id=ss2.evidence_hash
		WHERE ss2.run_id=cb.run_id
		  AND ss2.step_kind='aggregate'
		  AND ss2.stmt_hash=cb.stmt_hash
		  AND json_extract(ss2.output_json, '$.score') IS NOT NULL
		  AND tl.truth_set_id=?
		  AND tl.target_kind='evidence'
)`, [filters.truth_set ?? '']);
	if (filters.multi_evidence) add('multi_evidence', true, 'cb.n_evidences_for_stmt > 1');
	if (filters.supports || filters.supports_compare) {
		add(filters.supports_compare ? 'supports_compare' : 'supports', true, 'cb.has_supports');
	}
	if (filters.step_kind && filters.step_kind !== 'aggregate') {
		diagnostics.push({
			kind: 'not_applicable',
			filter: 'step_kind',
			value: filters.step_kind,
			message: `step_kind=${filters.step_kind} is a persisted substep, but statement cohorts summarize aggregate evidence rows; use a trace-state cohort to inspect substep membership`
		});
	} else {
		add('step_kind', filters.step_kind, "EXISTS (SELECT 1 FROM scorer_step ss2 WHERE ss2.run_id=cb.run_id AND ss2.step_kind='aggregate' AND ss2.stmt_hash=cb.stmt_hash)");
	}
	const traceFidelityPredicate = statementEvidenceTraceFidelityPredicate(filters.trace_fidelity, 'cb');
	if (traceFidelityPredicate) add('trace_fidelity', filters.trace_fidelity, traceFidelityPredicate);
	return { predicates, diagnostics };
}

export function aggregateEvidenceEmptyPredicates(filters: RunCohortFilters): {
	predicates: EmptyCohortPredicate[];
	diagnostics: RunCohortEmptyDiagnostic[];
} {
	const predicates: EmptyCohortPredicate[] = [];
	const diagnostics: RunCohortEmptyDiagnostic[] = [];
	const add = (
		filter: string,
		value: unknown,
		predicateSql: string,
		params: DuckDBValue[] = [],
		message?: string
	) => {
		const v = filterValue(value);
		if (v) predicates.push({ filter, value: v, predicateSql, params, message });
	};
	add('verdict', filters.verdict, "COALESCE(cb.verdict, '')=?", [filters.verdict ?? '']);
	if (filters.verdict_present) add('verdict_present', true, 'cb.verdict IS NOT NULL');
	if (filters.score_present) add('score_present', true, 'cb.score IS NOT NULL');
	if (filters.indra_belief_present) add('indra_belief_present', true, 'cb.indra_belief IS NOT NULL');
	add('confidence', filters.confidence, "COALESCE(cb.confidence, '')=?", [filters.confidence ?? '']);
	add('type', filters.type, 'cb.indra_type=?', [filters.type ?? '']);
	add('source', filters.source, "COALESCE(cb.source_api, '')=?", [filters.source ?? '']);
	add('source_stratum', filters.source_stratum, "COALESCE(cb.source_stratum, '')=?", [filters.source_stratum ?? '']);
	add('truth_set', filters.truth_set, `EXISTS (
		SELECT 1 FROM truth_label tl
		WHERE tl.truth_set_id=?
		  AND tl.target_kind='evidence'
		  AND tl.target_id=cb.evidence_hash
)`, [filters.truth_set ?? '']);
	if (filters.multi_evidence) add('multi_evidence', true, 'cb.n_evidences_for_stmt > 1');
	if (filters.supports || filters.supports_compare) {
		add(filters.supports_compare ? 'supports_compare' : 'supports', true, 'cb.has_supports');
	}
	if (filters.step_kind && filters.step_kind !== 'aggregate') {
		diagnostics.push({
			kind: 'not_applicable',
			filter: 'step_kind',
			value: filters.step_kind,
			message: `step_kind=${filters.step_kind} is a persisted substep, but aggregate evidence cohorts only display aggregate scorer rows; use a trace-state cohort to inspect substep membership`
		});
	} else {
		add('step_kind', filters.step_kind, "cb.step_kind='aggregate'");
	}
	const traceFidelityPredicate = aggregateEvidenceTraceFidelityPredicate(filters.trace_fidelity, 'cb');
	if (traceFidelityPredicate) add('trace_fidelity', filters.trace_fidelity, traceFidelityPredicate);
	return { predicates, diagnostics };
}

export function traceEvidenceEmptyPredicates(
	filters: RunCohortFilters,
	snapshotStartedAt: string | null
): {
	predicates: EmptyCohortPredicate[];
	diagnostics: RunCohortEmptyDiagnostic[];
} {
	const predicates: EmptyCohortPredicate[] = [];
	const add = (
		filter: string,
		value: unknown,
		predicateSql: string,
		params: DuckDBValue[] = [],
		message?: string
	) => {
		const v = filterValue(value);
		if (v) predicates.push({ filter, value: v, predicateSql, params, message });
	};
	add('trace_state', filters.trace_state, 'cb.trace_state=?', [filters.trace_state ?? '']);
	if (filters.probe_coverage === 'present') {
		add('probe_coverage', filters.probe_coverage, `(
			COALESCE(cb.n_substrate_route, 0) > 0
			OR COALESCE(cb.n_subject_role_probe, 0) > 0
			OR COALESCE(cb.n_object_role_probe, 0) > 0
			OR COALESCE(cb.n_relation_axis_probe, 0) > 0
			OR COALESCE(cb.n_scope_probe, 0) > 0
		)`);
	}
	add('verdict', filters.verdict, "COALESCE(cb.verdict, '')=?", [filters.verdict ?? '']);
	if (filters.verdict_present) add('verdict_present', true, 'cb.verdict IS NOT NULL');
	if (filters.score_present) add('score_present', true, 'cb.score IS NOT NULL');
	if (filters.indra_belief_present) add('indra_belief_present', true, 'cb.indra_belief IS NOT NULL');
	add('confidence', filters.confidence, "COALESCE(cb.confidence, '')=?", [filters.confidence ?? '']);
	add('type', filters.type, "COALESCE(cb.indra_type, '')=?", [filters.type ?? '']);
	add('source', filters.source, "COALESCE(cb.source_api, '')=?", [filters.source ?? '']);
	add('source_stratum', filters.source_stratum, "COALESCE(cb.source_stratum, '')=?", [filters.source_stratum ?? '']);
	add('truth_set', filters.truth_set, `EXISTS (
		SELECT 1 FROM truth_label tl
		WHERE tl.truth_set_id=?
		  AND tl.target_kind='evidence'
		  AND tl.target_id=cb.evidence_hash
)`, [filters.truth_set ?? '']);
	if (filters.multi_evidence) add('multi_evidence', true, 'cb.n_trace_rows_for_stmt > 1');
	if (filters.supports || filters.supports_compare) {
		add(filters.supports_compare ? 'supports_compare' : 'supports', true, 'cb.has_supports');
	}
	if (filters.step_kind) {
		const snapshotClause = snapshotStartedAt ? 'AND ss_step.started_at <= CAST(? AS TIMESTAMP)' : '';
		add('step_kind', filters.step_kind, `EXISTS (
			SELECT 1 FROM scorer_step ss_step
			WHERE ss_step.run_id=cb.run_id
			  AND ss_step.evidence_hash=cb.evidence_hash
			  AND ss_step.step_kind=?
			  ${snapshotClause}
		)`, snapshotStartedAt ? [filters.step_kind, snapshotStartedAt] : [filters.step_kind]);
	}
	return { predicates, diagnostics: [] };
}
