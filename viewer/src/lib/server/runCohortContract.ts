import { error } from '@sveltejs/kit';
import type { DuckDBConnection, DuckDBValue } from '@duckdb/node-api';
import type { RunCohortFilters } from '$lib/cohorts/types';
import {
	cleanTraceSnapshot,
	INVALID_TRACE_SNAPSHOT,
	isTraceFidelityFilter,
	isTraceFidelityState,
	type TraceFidelityFilter,
	type TraceFidelityState
} from '$lib/traceState';

export const RUN_ID_RE = /^[a-f0-9]{32}$/i;

export const RUN_STATUSES = new Set([
	'queued',
	'loading',
	'running',
	'succeeded',
	'failed',
	'canceled',
	'cancelled',
	'crashed',
	'blocked',
	'unknown'
]);

const GRAINS = new Set(['evidence', 'statement']);
const VERDICTS = new Set(['correct', 'incorrect', 'abstain']);
const CONFIDENCES = new Set(['high', 'medium', 'low']);
const PROBE_COVERAGE = new Set(['present']);
const TRUE_TOKENS = new Set(['true', '1', 'yes']);
const FALSE_TOKENS = new Set(['false', '0', 'no']);
const MAX_FILTER_VALUE_LENGTH = 256;

const FILTER_KEYS = new Set([
	'grain',
	'verdict',
	'verdict_present',
	'score_present',
	'indra_belief_present',
	'confidence',
	'type',
	'source',
	'source_stratum',
	'truth_set',
	'step_kind',
		'trace_fidelity',
		'trace_state',
		'trace_snapshot',
		'probe_coverage',
		'multi_evidence',
	'supports',
	'supports_compare'
]);

function contractError(code: string, message: string): never {
	throw error(400, { message, code });
}

function optionalStringField(name: string, value: unknown): string | null {
	if (value === null || value === undefined || value === '') return null;
	if (typeof value !== 'string') {
		contractError(`invalid_${name}`, `${name} must be a string`);
	}
	if (value.length > MAX_FILTER_VALUE_LENGTH) {
		contractError(`invalid_${name}`, `${name} must be ${MAX_FILTER_VALUE_LENGTH} characters or fewer`);
	}
	return value;
}

function enumField(name: string, value: string | null, allowed: Set<string>): string | null {
	if (!value) return null;
	if (!allowed.has(value)) {
		contractError(`invalid_${name}`, `invalid ${name}: ${value}; expected one of ${[...allowed].join(', ')}`);
	}
	return value;
}

function boolField(name: string, v: unknown): boolean {
	if (v === null || v === undefined || v === '') return false;
	if (v === true || v === false) return v;
	if (typeof v !== 'string') {
		contractError(`invalid_${name}`, `${name} must be a boolean token`);
	}
	const token = v.toLowerCase();
	if (TRUE_TOKENS.has(token)) return true;
	if (FALSE_TOKENS.has(token)) return false;
	contractError(`invalid_${name}`, `invalid ${name}: ${v}`);
}

function validateTraceFilters(
	rawTraceState: string | null,
	rawTraceFidelity: string | null,
	rawTraceSnapshot: string | null,
	rawProbeCoverage: string | null
): Pick<RunCohortFilters, 'grain' | 'trace_state' | 'trace_fidelity' | 'trace_snapshot' | 'probe_coverage'> {
	let traceState: TraceFidelityState | null = null;
	if (rawTraceState) {
		if (!isTraceFidelityState(rawTraceState)) {
			contractError('invalid_trace_state', `invalid trace_state: ${rawTraceState}`);
		}
		traceState = rawTraceState;
	}
	let traceFidelity: TraceFidelityFilter | null = null;
	if (rawTraceFidelity) {
		if (!isTraceFidelityFilter(rawTraceFidelity)) {
			contractError('invalid_trace_fidelity', `invalid trace_fidelity: ${rawTraceFidelity}`);
		}
		traceFidelity = rawTraceFidelity;
	}
	const probeCoverage = enumField('probe_coverage', rawProbeCoverage, PROBE_COVERAGE) as 'present' | null;
	if (traceState && traceFidelity) {
		contractError('incompatible_trace_filters', 'trace_state and trace_fidelity cannot be combined');
	}
	const traceSnapshot = cleanTraceSnapshot(rawTraceSnapshot);
	if (traceSnapshot === INVALID_TRACE_SNAPSHOT) {
		contractError('invalid_trace_snapshot', 'trace_snapshot must be a scorer_step timestamp');
	}
	if (traceSnapshot && !traceState && !probeCoverage) {
		contractError('trace_plane_required', 'trace_snapshot requires trace_state or probe_coverage');
	}
	return {
		grain: traceState || probeCoverage ? 'evidence' : null,
		trace_state: traceState,
		trace_fidelity: traceFidelity,
		trace_snapshot: traceSnapshot,
		probe_coverage: probeCoverage
	};
}

function validateNonTraceFields(
	rawGrain: string | null,
	rawVerdict: string | null,
	rawConfidence: string | null,
	tracePlane: boolean
): Pick<RunCohortFilters, 'grain' | 'verdict' | 'confidence'> {
	const grain = enumField('grain', rawGrain, GRAINS);
	if (tracePlane && grain && grain !== 'evidence') {
		contractError('invalid_grain', 'trace-plane cohorts require grain=evidence');
	}
	return {
		grain: tracePlane ? 'evidence' : grain,
		verdict: enumField('verdict', rawVerdict, VERDICTS),
		confidence: enumField('confidence', rawConfidence, CONFIDENCES)
	};
}

export function cohortFiltersFromSearchParams(searchParams: URLSearchParams): RunCohortFilters {
	for (const key of searchParams.keys()) {
		if (!FILTER_KEYS.has(key)) {
			contractError('unknown_cohort_filter', `unknown cohort filter: ${key}`);
		}
	}
	const traces = validateTraceFilters(
		searchParams.get('trace_state'),
		searchParams.get('trace_fidelity'),
		searchParams.get('trace_snapshot'),
		searchParams.get('probe_coverage')
	);
	const fields = validateNonTraceFields(
			searchParams.get('grain'),
			searchParams.get('verdict'),
			searchParams.get('confidence'),
			Boolean(traces.trace_state || traces.probe_coverage)
		);
	return {
		grain: fields.grain,
		verdict: fields.verdict,
		verdict_present: boolField('verdict_present', searchParams.get('verdict_present')),
		score_present: boolField('score_present', searchParams.get('score_present')),
		indra_belief_present: boolField('indra_belief_present', searchParams.get('indra_belief_present')),
		confidence: fields.confidence,
		type: searchParams.get('type'),
		source: searchParams.get('source'),
		source_stratum: searchParams.get('source_stratum'),
		truth_set: searchParams.get('truth_set'),
		step_kind: searchParams.get('step_kind'),
			trace_fidelity: traces.trace_fidelity,
			trace_state: traces.trace_state,
			trace_snapshot: traces.trace_snapshot,
			probe_coverage: traces.probe_coverage,
			multi_evidence: boolField('multi_evidence', searchParams.get('multi_evidence')),
		supports: boolField('supports', searchParams.get('supports')),
		supports_compare: boolField('supports_compare', searchParams.get('supports_compare'))
	};
}

export function cohortFiltersFromRecord(raw: unknown): RunCohortFilters {
	const f = (raw && typeof raw === 'object' ? raw : {}) as Record<string, unknown>;
	for (const key of Object.keys(f)) {
		if (!FILTER_KEYS.has(key)) {
			contractError('unknown_cohort_filter', `unknown cohort filter: ${key}`);
		}
	}
	const traces = validateTraceFilters(
		optionalStringField('trace_state', f.trace_state),
		optionalStringField('trace_fidelity', f.trace_fidelity),
		optionalStringField('trace_snapshot', f.trace_snapshot),
		optionalStringField('probe_coverage', f.probe_coverage)
	);
	const fields = validateNonTraceFields(
			optionalStringField('grain', f.grain),
			optionalStringField('verdict', f.verdict),
			optionalStringField('confidence', f.confidence),
			Boolean(traces.trace_state || traces.probe_coverage)
		);
	return {
		grain: fields.grain,
		verdict: fields.verdict,
		verdict_present: boolField('verdict_present', f.verdict_present),
		score_present: boolField('score_present', f.score_present),
		indra_belief_present: boolField('indra_belief_present', f.indra_belief_present),
		confidence: fields.confidence,
		type: optionalStringField('type', f.type),
		source: optionalStringField('source', f.source),
		source_stratum: optionalStringField('source_stratum', f.source_stratum),
		truth_set: optionalStringField('truth_set', f.truth_set),
		step_kind: optionalStringField('step_kind', f.step_kind),
			trace_fidelity: traces.trace_fidelity,
			trace_state: traces.trace_state,
			trace_snapshot: traces.trace_snapshot,
			probe_coverage: traces.probe_coverage,
			multi_evidence: boolField('multi_evidence', f.multi_evidence),
		supports: boolField('supports', f.supports),
		supports_compare: boolField('supports_compare', f.supports_compare)
	};
}

export function expectedRunStatus(raw: unknown): string | null {
	const status = typeof raw === 'string' ? raw : null;
	if (status && !RUN_STATUSES.has(status)) {
		contractError('invalid_expected_run_status', `expected_run_status must be one of ${[...RUN_STATUSES].join(', ')}`);
	}
	return status;
}

async function hasRow(
	con: DuckDBConnection,
	sql: string,
	params: DuckDBValue[]
): Promise<boolean> {
	const reader = await con.runAndReadAll(sql, params);
	return reader.getRowObjects().length > 0;
}

export async function validateRunCohortFilterValues(
	con: DuckDBConnection,
	run_id: string,
	filters: RunCohortFilters
): Promise<void> {
	if (filters.trace_state && filters.trace_fidelity) {
		contractError('incompatible_trace_filters', 'trace_state and trace_fidelity cannot be combined');
	}
	if (filters.probe_coverage) {
		enumField('probe_coverage', filters.probe_coverage, PROBE_COVERAGE);
	}
	if (filters.trace_snapshot && !filters.trace_state && !filters.probe_coverage) {
		contractError('trace_plane_required', 'trace_snapshot requires trace_state or probe_coverage');
	}
	validateNonTraceFields(
			filters.grain ?? null,
			filters.verdict ?? null,
			filters.confidence ?? null,
			Boolean(filters.trace_state || filters.probe_coverage)
		);
	const checks: Array<Promise<void>> = [];
	const requireKnown = (label: string, value: string | null | undefined, sql: string, params: DuckDBValue[]) => {
		if (!value) return;
		checks.push(hasRow(con, sql, params).then((ok) => {
			if (ok) return;
			if (label === 'step_kind') {
				contractError('unknown_step_kind_filter', `unknown step_kind filter: ${value}`);
			}
			contractError(`unknown_${label}_filter`, `unknown ${label} filter: ${value}`);
		}));
	};
	requireKnown('type', filters.type, 'SELECT 1 FROM statement WHERE indra_type=? LIMIT 1', [filters.type ?? '']);
	requireKnown('source', filters.source, 'SELECT 1 FROM evidence WHERE source_api=? LIMIT 1', [filters.source ?? '']);
	requireKnown('source_stratum', filters.source_stratum, 'SELECT 1 FROM evidence WHERE source_api=? LIMIT 1', [filters.source_stratum ?? '']);
	requireKnown('truth_set', filters.truth_set, 'SELECT 1 FROM truth_set WHERE id=? LIMIT 1', [filters.truth_set ?? '']);
	requireKnown(
		'step_kind',
		filters.step_kind,
		'SELECT 1 FROM scorer_step WHERE step_kind=? LIMIT 1',
		[filters.step_kind ?? '']
	);
	await Promise.all(checks);
}
