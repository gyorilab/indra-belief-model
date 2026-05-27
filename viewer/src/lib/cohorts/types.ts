// Renamed from $lib/runCohortTypes.ts on 2026-05-27 (B4 of deferred hypergraph).
// No code changes vs the prior location; see git mv for history.

import type { TraceFidelityFilter, TraceFidelityState } from '../traceState';

export interface RunCohortFilters {
	grain?: string | null;
	verdict?: string | null;
	verdict_present?: boolean;
	score_present?: boolean;
	indra_belief_present?: boolean;
	confidence?: string | null;
	type?: string | null;
	source?: string | null;
	source_stratum?: string | null;
	truth_set?: string | null;
	step_kind?: string | null;
	multi_evidence?: boolean;
	supports?: boolean;
	supports_compare?: boolean;
	trace_fidelity?: TraceFidelityFilter | null;
	trace_state?: TraceFidelityState | null;
	trace_snapshot?: string | null;
	probe_coverage?: 'present' | null;
}
