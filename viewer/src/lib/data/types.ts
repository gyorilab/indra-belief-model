/**
 * The JSONL-derived data model — a read-only projection of the monolithic
 * pipeline's exports (`data/exports/<run>/{per_statement.json, per_evidence.jsonl,
 * export_meta.json}`).
 *
 * A "run" is one monolithic scoring pass (e.g. gemma vs medpsy). Each export dir
 * is self-contained: per_evidence carries evidence_text + reasoning, so the only
 * thing joined from the source corpus is agent db_refs + the supports graph.
 */

/** One scoring run, discovered from `data/exports/<dir>/export_meta.json`. */
export interface RunMeta {
	run_id: string;
	/** Absolute path to the export directory. */
	export_dir: string;
	model: string;
	generated_date: string | null;
	counts: {
		unique_evidence_rows?: number;
		statements?: number;
		statements_scored?: number;
		run_lines?: number;
	};
	bucket_counts: Record<string, number>;
	/** Raw run JSONL this export was generated from. */
	source_run: string | null;
	/** The corpus this run was scored against (`generated_from.corpus`, basename).
	 *  The JOIN BOUNDARY for cross-run comparison: runs are only comparable —
	 *  content-hash-joinable, frontier-plottable — within one substrate. `null` on
	 *  legacy exports that did not record it. */
	substrate: string | null;
	/** Run-level gold coverage from `export_meta.gold` (how many evidences carry a
	 *  human curation). Read at discovery, no evidence load. `null` on runs with no
	 *  baked gold. */
	gold_coverage: { covered: number; total: number } | null;
	/** Ground-truth model size, baked from `export_meta.model_meta` (travels with
	 *  the run; the viewer holds no size table). `status: 'unknown'` (total_b null)
	 *  for closed-weight models — never a guessed number. `undefined` ⇒ legacy
	 *  export (pre model_meta). */
	model_meta?: {
		status: 'known' | 'unknown';
		total_b: number | null;
		active_b: number | null;
		is_open: boolean | null;
		/** size is inferred/undisclosed for this exact model (not a published spec)
		 *  — the size-axis analogue of an estimated cost; renders as a hollow dot. */
		estimated: boolean;
		source: string;
	} | null;
	/** From the raw run's .meta.json, when present. */
	status: string | null;
	started_at: string | null;
	finished_at: string | null;
	/** Run-level observed LLM cost (baked at export from per-evidence call_logs).
	 *  Numbers only — the viewer holds no price table. `null` ⇒ legacy export
	 *  (pre-cost field); render as "unavailable", never $0. */
	cost?: {
		status: 'known' | 'estimated' | 'partial' | 'unavailable';
		total_usd: number | null;
		input_tokens: number;
		output_tokens: number;
		n_evidence_costed: number;
		n_evidence_no_llm: number;
		n_evidence_unavailable: number;
		models: string[];
		usd_per_1k_evidence: number | null;
	} | null;
	/** Per-run soft-weight calibration (E5): the fitted triple that applies to
	 *  this reader, baked at export so it travels with the run. `status:
	 *  'unavailable'` (with a reason) when the reader has no fit; `undefined` ⇒
	 *  legacy export (schema < 4). The soft path is default-off — this records
	 *  which calibration *applies*, not that it was used. Named `soft_calibration`
	 *  (not `calibration`) to stay distinct from `Validity.calibration`, the
	 *  separate belief-vs-INDRA residual measure. */
	soft_calibration?: {
		status: 'available' | 'unavailable';
		model: string | null;
		soft_weights: { w_correct: number; w_incorrect: number; variant: string } | null;
		reason?: string;
	};
}

/** One statement rollup, from `per_statement.json`. */
export interface StatementRollup {
	stmt_hash: string;
	indra_matches_hash?: string | null;
	indra_id?: string | null;
	stmt_i?: number;
	subject: string;
	stmt_type: string;
	object: string;
	rasmachine_belief: number | null;
	our_mean_score: number | null;
	our_noisy_or: number | null;
	our_max_score: number | null;
	our_min_score: number | null;
	n_evidence: number;
	n_correct: number;
	n_incorrect: number;
	n_unscored: number;
	dominant_bucket: string | null;
	bucket_counts: Record<string, number>;
	pmids: string[];
	sources: string[];
}

/** Epistemic-access status of a model's chain-of-thought on one call.
 *  plaintext/inline → readable CoT present; encrypted → it reasoned but the text
 *  is sealed by the provider (reasoning_tokens > 0, no text); not_returned →
 *  reasoning requested but nothing came back; none → no reasoning. */
export type ReasoningStatus =
	| 'plaintext'
	| 'inline'
	| 'encrypted'
	| 'not_returned'
	| 'none';

/** Uniform per-call reasoning capture (model_client `reasoning_trace`, projected
 *  by results.compact_reasoning_trace). Separates "did it reason" (reasoning_tokens)
 *  from "can we read it" (free_cot + status), and carries the model's committed
 *  support/objection — the reliable, always-parseable justification. */
export interface ReasoningTrace {
	status: ReasoningStatus | null;
	/** Provider-reported reasoning-token count; -1 when not reported. */
	reasoning_tokens: number | null;
	/** Free chain-of-thought, clipped to free_cot_chars at export. */
	free_cot: string | null;
	/** Full (pre-clip) CoT length, so the UI can flag truncation. */
	free_cot_chars: number | null;
	provider_source: string | null;
	backend: string | null;
	model_id: string | null;
	finish_reason: string | null;
	committed_justification: {
		support: string | null;
		objection: string | null;
		source: string | null;
	};
}

/** One (statement, evidence) row, from `per_evidence.jsonl`. */
export interface EvidenceRow {
	stmt_hash: string;
	evidence_hash: string;
	source_hash?: string | number | null;
	indra_matches_hash?: string | null;
	indra_id?: string | null;
	stmt_i?: number;
	evidence_i?: number;
	subject: string;
	stmt_type: string;
	object: string;
	source_api: string | null;
	pmid: string | null;
	evidence_text: string | null;
	text_len: number;
	rasmachine_belief: number | null;
	our_score: number | null;
	verdict: string | null;
	confidence: string | null;
	/** The model's chain-of-thought / rationale (the monolithic "trace"). */
	reasoning: string | null;
	reasoning_truncated?: boolean;
	/** Uniform CoT-access + committed-justification capture. Null on legacy
	 *  exports (pre schema 5); render falls back to `reasoning` then. */
	reasoning_trace?: ReasoningTrace | null;
	grounding_status: string | null;
	tier: string | null;
	provenance_triggered?: boolean;
	/** Report taxonomy bucket (semantic_correct, reader_hallucination, …). */
	bucket: string | null;
	bucket_group: string | null;
	error: string | null;
	latency_s?: number | null;
	tokens?: number | null;
	/** Observed LLM cost for this evidence (computed at export from call_log).
	 *  cost_usd null + cost_status 'unavailable' ⇒ a model with no verified price.
	 *  Absent entirely ⇒ legacy export (pre-cost); treat as unavailable. */
	cost_usd?: number | null;
	cost_status?: 'known' | 'estimated' | 'unavailable';
	input_tokens?: number;
	output_tokens?: number;
	n_calls?: number;
	/** Per-run gold baked in at export time (the run's OWN curation source).
	 *  Present (verdict object OR null=uncurated) on baked runs; ABSENT on legacy
	 *  runs, which fall back to the global curation index. See goldForRow. */
	gold?: GoldVerdict | null;
}

// ── Calibration products (E5 metrics.json, schema_version 2) ────────────────
//
// Written alongside per_evidence.jsonl by results.build_run_metrics. The viewer
// READS these byte-exact and never recomputes (gate G4). Two tiers (ev/stmt),
// each either named-empty (status 'unavailable' + reason) or a block of arms.
// v2 adds tiers.stmt.verdict_err (statement error-detection confusion on the
// tiered verdict) + tiers.stmt.stratified (per-type/source/evidence/bucket/driver
// residual). Untyped here until the viewer reads them (E11); structural typing
// tolerates the extra keys meanwhile.

/** One reliability bin (BINS_8). Unoccupied bins carry n:0 + null pred/empirical
 *  so the x-axis is stable across runs (mirrors the n≥30 validity guard). */
export interface ReliabilityBin {
	lo: number;
	hi: number;
	n: number;
	mean_pred: number | null;
	empirical: number | null;
}

/** One arm's full calibration unit — the stable shape C4/C5 render. */
export interface MetricArm {
	n: number;
	ece: number;
	auroc: number | null;
	auprc: number | null;
	brier: number;
	/** Murphy decomposition: brier = reliability − resolution + uncertainty. */
	reliability: number;
	resolution: number;
	uncertainty: number;
	confusion: { tp: number; fp: number; fn: number; tn: number };
	bins: ReliabilityBin[];
}

/** An arm slot is either a realized MetricArm or a named-empty reason. */
export type ArmSlot = MetricArm | { status: 'unavailable'; reason: string };

export function armAvailable(a: ArmSlot | undefined): a is MetricArm {
	return !!a && !('status' in a);
}

/** One tier (ev = Tier-1 per-evidence, stmt = Tier-2 per-statement). */
export type TierBlock =
	| { status: 'unavailable'; reason: string }
	| {
			status: 'available';
			n: number;
			base_rate_correct: number;
			arms: Record<string, ArmSlot>;
	  };

/** The per-run metrics.json contract (E5). */
export interface RunMetrics {
	schema_version: number;
	run_id: string | null;
	model: string | null;
	generated_date: string;
	metrics_basis: Record<string, unknown>;
	gold: { source: string | null; covered: number; total: number } | null;
	tiers: { ev: TierBlock; stmt: TierBlock };
}

/**
 * One INDRA curation, from data/benchmark/rasmachine_curations.jsonl (pulled
 * from db.indra.bio/curation/list/<matches_hash>). A human correctness label
 * on a specific (statement, evidence). NOTE the native-int hashes: these are
 * the INDRA keys, distinct from the viewer's internal hex stmt_hash/evidence_hash.
 */
export interface CurationRow {
	/** INDRA statement matches_hash (== pa_hash). int. */
	_matches_hash: number;
	pa_hash: number;
	/** INDRA evidence source_hash. int. Joins to EvidenceRow.source_hash. */
	source_hash: number;
	/** correct | no_relation | wrong_relation | grounding | polarity |
	 *  act_vs_amt | hypothesis | negative_result | entity_boundaries |
	 *  mod_site | other */
	tag: string;
	curator: string;
	date: string;
	/** free-text curator note (often empty). */
	text: string;
}

/** The derived gold verdict for one (matches_hash, source_hash), aggregated
 *  over possibly-multiple curations with the any-incorrect-wins rule. */
export interface GoldVerdict {
	/** 'correct' iff every curation tag == 'correct'; else 'incorrect'. */
	verdict: 'correct' | 'incorrect';
	/** number of curations backing this verdict. */
	n: number;
	/** every tag seen (correct, wrong_relation, …) — the "why". */
	tags: string[];
	/** distinct curator identities. */
	curators: string[];
	/** non-empty free-text notes. */
	notes: string[];
}
