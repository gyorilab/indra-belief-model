export const TRACE_FIDELITY_STATES = [
	'full',
	'partial',
	'aggregate_only',
	'not_applicable',
	'terminated_inflight',
	'missing_aggregate',
	'step_error'
] as const;

export type TraceFidelityState = (typeof TRACE_FIDELITY_STATES)[number];

export const TRACE_FIDELITY_FILTERS = ['aggregate_only', 'native_decomposed'] as const;
export type TraceFidelityFilter = (typeof TRACE_FIDELITY_FILTERS)[number];

export const INVALID_TRACE_STATE = '__invalid__';
export const INVALID_TRACE_SNAPSHOT = '__invalid_trace_snapshot__';
const TRACE_SNAPSHOT_RE = /^\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?$/;

export const TRACE_STATE_LABELS: Record<TraceFidelityState, string> = {
	full: 'full',
	partial: 'partial',
	aggregate_only: 'aggregate only',
	not_applicable: 'not applicable',
	terminated_inflight: 'unfinalized',
	missing_aggregate: 'missing aggregate',
	step_error: 'step error'
};

export const TRACE_STATE_NOTES: Record<TraceFidelityState, string> = {
	full: 'all expected native trace fields are captured',
	partial: 'native grammar is visible, but some diagnostic fields are collapsed or absent',
	aggregate_only: 'only the final scorer dict is available',
	not_applicable: 'the requested trace grammar does not apply',
	terminated_inflight: 'pre-aggregate steps were persisted but no final aggregate verdict was finalized',
	missing_aggregate: 'a succeeded run lacks the final aggregate row for this evidence',
	step_error: 'one or more persisted steps carries an explicit error'
};

const TRACE_STATE_SET = new Set<string>(TRACE_FIDELITY_STATES);
const TRACE_FIDELITY_FILTER_SET = new Set<string>(TRACE_FIDELITY_FILTERS);

export function isTraceFidelityState(value: unknown): value is TraceFidelityState {
	return typeof value === 'string' && TRACE_STATE_SET.has(value);
}

export function isTraceFidelityFilter(value: unknown): value is TraceFidelityFilter {
	return typeof value === 'string' && TRACE_FIDELITY_FILTER_SET.has(value);
}

export function cleanTraceStateFilter(
	state: string | null | undefined
): TraceFidelityState | typeof INVALID_TRACE_STATE | null {
	if (!state) return null;
	return isTraceFidelityState(state) ? state : INVALID_TRACE_STATE;
}

export function cleanTraceSnapshot(
	value: string | null | undefined
): string | typeof INVALID_TRACE_SNAPSHOT | null {
	if (!value) return null;
	const normalized = value.replace('T', ' ');
	return TRACE_SNAPSHOT_RE.test(normalized) ? normalized : INVALID_TRACE_SNAPSHOT;
}

export function zeroTraceCounts(): Record<TraceFidelityState, number> {
	return {
		full: 0,
		partial: 0,
		aggregate_only: 0,
		not_applicable: 0,
		terminated_inflight: 0,
		missing_aggregate: 0,
		step_error: 0
	};
}
