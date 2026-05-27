import type { RunCohortFilters } from './runCohortTypes';
import { statementEvidenceSourceStratumSql } from './evidenceMembershipSql';
import { cleanTraceStateFilter, INVALID_TRACE_STATE } from './traceState';

export const RUN_COHORT_LIMIT = 500;
export const COHORT_EVIDENCE_LEVEL_FILTER_KEYS = [
	'source',
	'verdict',
	'verdict_present',
	'score_present',
	'confidence',
	'truth_set',
	'trace_fidelity'
] as const;

export type CohortEvidenceLevelFilterKey = typeof COHORT_EVIDENCE_LEVEL_FILTER_KEYS[number];
type AggregateEvidenceTraceFidelityAlias = 'ss' | 'ss_stmt' | 'ss_agg' | 'cb';
type StatementTraceFidelityAlias = 'stmt_scores' | 'cb';

interface CohortEvidenceLevelAliases {
	step: AggregateEvidenceTraceFidelityAlias;
	evidence: string;
	truthLabel?: string;
}

function sqlQuote(s: string): string {
	return s.replace(/'/g, "''");
}

function boolish(v: unknown): boolean {
	return v === true || v === 'true' || v === '1' || v === 'yes';
}

function filterValue(v: unknown): string | null {
	if (v === null || v === undefined || v === false || v === '') return null;
	return String(v);
}

export function activeCohortEvidenceLevelFilterKeys(
	filters: RunCohortFilters
): CohortEvidenceLevelFilterKey[] {
	return COHORT_EVIDENCE_LEVEL_FILTER_KEYS.filter((key) => filterValue(filters[key]) !== null);
}

export function cohortEvidenceLevelFilterCount(filters: RunCohortFilters): number {
	return activeCohortEvidenceLevelFilterKeys(filters).length;
}

export function cohortEvidenceLevelPredicates(
	filters: RunCohortFilters,
	aliases: CohortEvidenceLevelAliases
): string[] {
	const ss = aliases.step;
	const e = aliases.evidence;
	const tl = aliases.truthLabel ?? 'tl_ev_filter';
	const predicates: string[] = [];
	if (filters.source) {
		predicates.push(`COALESCE(${e}.source_api, '')='${sqlQuote(filters.source)}'`);
	}
	if (filters.verdict) {
		predicates.push(`COALESCE(json_extract_string(${ss}.output_json, '$.verdict'), '')='${sqlQuote(filters.verdict)}'`);
	}
	if (filters.verdict_present) {
		predicates.push(`json_extract(${ss}.output_json, '$.verdict') IS NOT NULL`);
	}
	if (filters.score_present) {
		predicates.push(`json_extract(${ss}.output_json, '$.score') IS NOT NULL`);
	}
	if (filters.confidence) {
		predicates.push(`COALESCE(json_extract_string(${ss}.output_json, '$.confidence'), '')='${sqlQuote(filters.confidence)}'`);
	}
	if (filters.truth_set) {
		predicates.push(`EXISTS (
			SELECT 1 FROM truth_label ${tl}
			WHERE ${tl}.truth_set_id='${sqlQuote(filters.truth_set)}'
			  AND ${tl}.target_kind='evidence'
			  AND ${tl}.target_id=${ss}.evidence_hash
			  AND ${tl}."field" IN ('verdict', 'tag')
			  AND (${tl}.relation_target_id IS NULL OR ${tl}.relation_target_id=${ss}.stmt_hash)
		)`);
	}
	const traceFidelityPredicate = aggregateEvidenceTraceFidelityPredicate(filters.trace_fidelity, ss);
	if (traceFidelityPredicate) predicates.push(traceFidelityPredicate);
	return predicates;
}

function traceFidelityPredicateFromExists(
	traceFidelity: RunCohortFilters['trace_fidelity'],
	existsSql: string
): string | null {
	if (!traceFidelity) return null;
	if (traceFidelity === 'aggregate_only') return `NOT ${existsSql}`;
	if (traceFidelity === 'native_decomposed') return existsSql;
	return 'FALSE';
}

export function aggregateEvidenceTraceFidelityPredicate(
	traceFidelity: RunCohortFilters['trace_fidelity'],
	stepAlias: AggregateEvidenceTraceFidelityAlias
): string | null {
	return traceFidelityPredicateFromExists(
		traceFidelity,
		`EXISTS (
			SELECT 1 FROM scorer_step ss_trace
			WHERE ss_trace.run_id=${stepAlias}.run_id
			  AND ss_trace.evidence_hash=${stepAlias}.evidence_hash
			  AND ss_trace.step_kind <> 'aggregate'
		)`
	);
}

export function statementEvidenceTraceFidelityPredicate(
	traceFidelity: RunCohortFilters['trace_fidelity'],
	statementAlias: StatementTraceFidelityAlias
): string | null {
	const evidencePredicate = aggregateEvidenceTraceFidelityPredicate(traceFidelity, 'ss_agg');
	if (!evidencePredicate) return null;
	return `EXISTS (
			SELECT 1 FROM scorer_step ss_agg
			WHERE ss_agg.run_id=${statementAlias}.run_id
			  AND ss_agg.stmt_hash=${statementAlias}.stmt_hash
			  AND ss_agg.step_kind='aggregate'
			  AND json_extract(ss_agg.output_json, '$.score') IS NOT NULL
			  AND ${evidencePredicate}
		)`;
}

export function traceStateCohortWhereClauses(
	filters: RunCohortFilters,
	snapshotStartedAt?: string | null
): string[] {
	const traceState = cleanTraceStateFilter(filters.trace_state);
	if (traceState === INVALID_TRACE_STATE) {
		throw new Error(`invalid trace_state: ${filters.trace_state}`);
	}
	if (!traceState) {
		if (filters.probe_coverage !== 'present') {
			throw new Error('trace_state or probe_coverage is required for trace-plane cohort selection');
		}
	}
	const where: string[] = [];
	if (traceState) where.push(`ts.trace_state='${sqlQuote(traceState)}'`);
	if (filters.probe_coverage === 'present') {
		where.push(`(
			COALESCE(ts.n_substrate_route, 0) > 0
			OR COALESCE(ts.n_subject_role_probe, 0) > 0
			OR COALESCE(ts.n_object_role_probe, 0) > 0
			OR COALESCE(ts.n_relation_axis_probe, 0) > 0
			OR COALESCE(ts.n_scope_probe, 0) > 0
		)`);
	}
	if (filters.verdict) where.push(`COALESCE(ts.verdict, '')='${sqlQuote(filters.verdict)}'`);
	if (filters.verdict_present) where.push('ts.verdict IS NOT NULL');
	if (filters.score_present) where.push('ts.score IS NOT NULL');
	if (filters.indra_belief_present) where.push('s.indra_belief IS NOT NULL');
	if (filters.confidence) where.push(`COALESCE(ts.confidence, '')='${sqlQuote(filters.confidence)}'`);
	if (filters.type) where.push(`COALESCE(s.indra_type, '')='${sqlQuote(filters.type)}'`);
	if (filters.source) where.push(`COALESCE(e.source_api, '')='${sqlQuote(filters.source)}'`);
	if (filters.source_stratum) {
		where.push(`COALESCE(${statementEvidenceSourceStratumSql('ts.stmt_hash')}, '')='${sqlQuote(filters.source_stratum)}'`);
	}
	if (boolish(filters.multi_evidence)) {
		where.push(`(SELECT COUNT(*) FROM trace_rows ts2 WHERE ts2.run_id=ts.run_id AND ts2.stmt_hash=ts.stmt_hash) > 1`);
	}
	if (boolish(filters.supports) || boolish(filters.supports_compare)) {
		where.push(`(COALESCE(s.supports_count, 0) > 0 OR COALESCE(s.supported_by_count, 0) > 0)`);
	}
	if (filters.step_kind) {
		where.push(`EXISTS (
			SELECT 1 FROM scorer_step ss_step
			WHERE ss_step.run_id=ts.run_id
			  AND ss_step.evidence_hash=ts.evidence_hash
			  AND ss_step.step_kind='${sqlQuote(filters.step_kind)}'
			  ${snapshotStartedAt ? `AND ss_step.started_at <= TIMESTAMP '${sqlQuote(snapshotStartedAt)}'` : ''}
		)`);
	}
	if (filters.truth_set) {
		where.push(`EXISTS (
			SELECT 1 FROM truth_label tl
			WHERE tl.truth_set_id='${sqlQuote(filters.truth_set)}'
			  AND tl.target_kind='evidence'
			  AND tl.target_id=ts.evidence_hash
			  AND tl."field" IN ('verdict', 'tag')
			  AND (tl.relation_target_id IS NULL OR tl.relation_target_id=ts.stmt_hash)
		)`);
	}
	return where;
}

export function nonTraceEvidenceCohortWhereClauses(
	run_id: string,
	filters: RunCohortFilters
): string[] {
	const qr = sqlQuote(run_id);
	const where = [`ss.run_id='${qr}'`, `ss.step_kind='aggregate'`];
	where.push(...cohortEvidenceLevelPredicates(filters, { step: 'ss', evidence: 'e', truthLabel: 'tl' }));
	if (filters.indra_belief_present) {
		where.push('s.indra_belief IS NOT NULL');
	}
	if (filters.type) {
		where.push(`s.indra_type='${sqlQuote(filters.type)}'`);
	}
	if (filters.source_stratum) {
		where.push(`COALESCE(${statementEvidenceSourceStratumSql('s.stmt_hash')}, '')='${sqlQuote(filters.source_stratum)}'`);
	}
	if (boolish(filters.multi_evidence)) {
		where.push(`(SELECT COUNT(*) FROM scorer_step ss2 WHERE ss2.run_id=ss.run_id AND ss2.step_kind='aggregate' AND ss2.stmt_hash=ss.stmt_hash) > 1`);
	}
	if (boolish(filters.supports) || boolish(filters.supports_compare)) {
		where.push(`(s.supports_count > 0 OR s.supported_by_count > 0)`);
	}
	if (filters.step_kind && filters.step_kind !== 'aggregate') {
		where.push('FALSE');
	}
	return where;
}

export function nonTraceStatementCohortWhereClauses(
	filters: RunCohortFilters
): string[] {
	const where: string[] = ['1=1'];
	if (filters.type) where.push(`indra_type='${sqlQuote(filters.type)}'`);
	if (filters.indra_belief_present) where.push('indra_belief IS NOT NULL');
	if (filters.score_present) where.push('score IS NOT NULL');
	const evidencePredicates = cohortEvidenceLevelPredicates(filters, {
		step: 'ss_stmt',
		evidence: 'e_stmt',
		truthLabel: 'tl_stmt'
	});
	if (evidencePredicates.length > 0) {
		where.push(`EXISTS (
			SELECT 1 FROM scorer_step ss_stmt
			LEFT JOIN evidence e_stmt ON e_stmt.evidence_hash=ss_stmt.evidence_hash
			WHERE ss_stmt.run_id=stmt_scores.run_id
			  AND ss_stmt.step_kind='aggregate'
			  AND ss_stmt.stmt_hash=stmt_scores.stmt_hash
			  AND json_extract(ss_stmt.output_json, '$.score') IS NOT NULL
			  AND ${evidencePredicates.join('\n\t\t\t  AND ')}
		)`);
	}
	if (filters.source_stratum) {
		where.push(`COALESCE(source_stratum, '')='${sqlQuote(filters.source_stratum)}'`);
	}
	if (boolish(filters.multi_evidence)) where.push('n_evidences_for_stmt > 1');
	if (boolish(filters.supports) || boolish(filters.supports_compare)) where.push('has_supports');
	if (filters.step_kind && filters.step_kind !== 'aggregate') where.push('FALSE');
	return where;
}
