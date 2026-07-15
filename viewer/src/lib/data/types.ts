/**
 * The JSONL-derived data model — a read-only projection of the monolithic
 * pipeline's exports (`data/exports/<run>/{per_statement.json, per_evidence.jsonl,
 * export_meta.json}`).
 *
 * A "run" is one monolithic scoring pass (e.g. gemma vs medpsy). Each export dir
 * is self-contained: per_evidence carries evidence_text + reasoning, so the only
 * thing joined from the source corpus is agent db_refs + the supports graph.
 */

/** Gold-measured reader profile used by the fitted-reader belief model.
 *  Profile fields are derived from the verdict×gold confusion matrix; the final
 *  hybrid score additionally uses the separately fitted source-reliability bank. */
export interface ReaderCalibrationProfile {
	profile_id: string;
	reader_configuration: string;
	reader_model?: string;
	prompt_sha256?: string;
	deployment_status?: 'enabled' | 'disabled';
	validation?: {
		result: 'pass' | 'fail';
		gold: string;
		gold_sha256?: string;
		run: string;
		gate: string;
		note?: string;
	};
	fit_run: string;
	fit_gold: string;
	fit_gold_sha256?: string;
	fit_unique_pairs: number;
	gold_rule: string;
	confusion: {
		/** confirmed and gold-correct */
		cc: number;
		/** confirmed and gold-incorrect */
		ci: number;
		/** rejected and gold-correct */
		ic: number;
		/** rejected and gold-incorrect */
		ii: number;
	};
	sensitivity: number;
	false_positive_rate: number;
	specificity: number;
	miss_rate: number;
	/** log(P(confirm|correct) / P(confirm|incorrect)) */
	log_lr_confirm: number;
	/** log(P(reject|correct) / P(reject|incorrect)) */
	log_lr_reject: number;
	/** Correct evidence-pair prevalence in the profile fit; used as score anchor. */
	prior_correct: number;
	prior_logodds: number;
}

/** Content-addressed identity of the inputs and exact Tier-1 + Tier-2 evaluation sets.
 * These digests are baked when an export is created; the viewer must never
 * re-hash a mutable source path and pretend it describes an older artifact. */
export interface CalibrationProvenance {
	corpus_sha256: string | null;
	gold_sha256: string | null;
	evaluation_set_sha256: string | null;
}

/** Pre-v8 survival-weight payload retained so old schema-v7 exports still load. */
export interface LegacySoftWeights {
	w_correct: number;
	w_incorrect: number;
	variant?: string;
}

/** One scoring run, discovered from `data/exports/<dir>/export_meta.json`. */
export interface RunMeta {
	run_id: string;
	/** Absolute path to the export directory. */
	export_dir: string;
	model: string;
	generated_date: string | null;
	/** export_meta.json contract version; null on legacy exports. */
	export_schema_version: number | null;
	counts: {
		unique_evidence_rows?: number;
		statements?: number;
		statements_scored?: number;
		run_lines?: number;
	};
	bucket_counts: Record<string, number>;
	/** Raw run JSONL this export was generated from. */
	source_run: string | null;
	/** Content digests mirrored from metrics.json for end-to-end inspection.
	 * `undefined` on exports created before schema v8. */
	provenance?: CalibrationProvenance | null;
	/** Display name of the corpus this run was scored against
	 * (`generated_from.corpus`, basename). Cross-run calibration compatibility uses
	 * baked provenance digests, never this mutable/path-derived label. */
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
	/** Per-run reader measurement calibration (E5), baked at export so it travels
	 *  with the run. `status:
	 *  'unavailable'` (with a reason) when no ship-approved exact model+prompt
	 *  profile applies; `undefined` ⇒ a legacy export without this field. This records which confusion-derived profile
	 *  applies, not that it was used. Named `soft_calibration`
	 *  (not `calibration`) to stay distinct from `Validity.calibration`, the
	 *  separate belief-vs-INDRA residual measure. `soft_weights` is a legacy JSON
	 *  key: schema v8 carries a likelihood-ratio profile; schema v7 carried two
	 *  survival weights. */
	soft_calibration?: {
		status: 'available' | 'unavailable';
		model: string | null;
		reader_configuration?: {
			status: 'identified' | 'mixed' | 'mismatch' | 'missing_prompt';
			id: string | null;
			model: string | null;
			prompt_sha256: string | null;
			prompt_fingerprint_source?: 'call_log' | 'run_metadata' | null;
			declared_prompt_sha256?: string | null;
			prompt_fingerprints?: Record<string, number>;
		};
		soft_weights: ReaderCalibrationProfile | LegacySoftWeights | null;
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
	// ── E5/E11 three-way belief + statement gold (schema v7; optional: legacy
	//    exports pre-v7 lack them, so the viewer narrows on presence) ───────────
	/** Canonical production belief (schema v8): configuration-specific hybrid
	 *  log-odds score for fitted readers, with hard-gate fallback for unfitted. */
	belief?: number | null;
	/** Hard-gated parametric noisy-OR: comparison arm and unfitted-reader fallback. */
	belief_hard?: number | null;
	/** Ungated parametric noisy-OR — all evidence counted. */
	belief_parametric?: number | null;
	/** Configuration-specific hybrid log-odds score; null = reader has no fit. */
	belief_soft?: number | null;
	/** Tier-driven decision: deterministic reject hard-flags incorrect; credible
	 *  LLM incorrect routes to review; else correct. */
	belief_verdict_statement?: 'correct' | 'review' | 'incorrect';
	/** Statement-grain gold (any-incorrect-wins over the statement's evidence
	 *  gold). null = no curated evidence. Narrower than GoldVerdict (no curators/
	 *  notes — it is a rollup, not a single curation group). */
	gold_statement?: StatementGold | null;
	/** The multi-evidence depth behind the belief (POST-dedup tallies). */
	coherence_summary?: CoherenceSummary;
}

/** Statement-grain gold rollup baked into per_statement.json (schema v7). */
export interface StatementGold {
	verdict: 'correct' | 'incorrect';
	n: number;
	tags: string[];
}

/** Post-dedup multi-evidence tallies behind a statement's belief (schema v7). */
export interface CoherenceSummary {
	n_dedup_groups: number;
	n_distinct_sources: number;
	n_correct: number;
	n_incorrect: number;
	n_no_text: number;
	n_parse_fail: number;
	n_null_source: number;
	n_credible_incorrect_det: number;
	n_credible_incorrect_llm: number;
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

// ── Calibration products (E5 metrics.json, schema_version 2+) ───────────────
//
// Current contract is schema v3. Statement-decision/stratification fields were
// introduced in v2; v3 aligns Tier-2 gold, statement grain, de-dup/no-text
// handling, and calibration with the production export and configuration-
// specific hybrid score. The JSON arm key remains `soft` across both contracts,
// but its semantics do NOT: v2 = historical survival-weight score, v3+ = hybrid
// log-odds. Consumers must gate canonical selection on schema_version.
// Written alongside per_evidence.jsonl by results.build_run_metrics. The viewer
// READS these byte-exact and never recomputes (gate G4). Two tiers (ev/stmt),
// each either named-empty (status 'unavailable' + reason) or a block of arms.
// v2 adds tiers.stmt.verdict_err (statement error-detection confusion on the
// tiered verdict) + tiers.stmt.stratified (per-type/source/evidence/bucket/driver
// residual).

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

/** Error-detection confusion on the tiered verdict_statement (schema v2).
 *  positive = ERROR; pred_error = verdict != 'correct' (review AND incorrect
 *  flag); gold_error = statement gold incorrect. Distinct from a MetricArm: this
 *  is a decision, not a calibrated scalar — no bins/ece/brier. */
export interface ConfusionMetrics {
	n: number;
	tp: number;
	fp: number;
	fn: number;
	tn: number;
	accuracy: number;
	precision: number;
	recall: number;
	f1: number;
}

/** One stratum of the statement residual map (schema v2). `hard` is the belief
 *  calibration over the same statements; `verdict_err` the decision quality. */
export interface StratumBlock {
	n: number;
	base_rate_correct: number;
	verdict_err: ConfusionMetrics;
	hard: MetricArm;
}

/** The statement residual map (schema v2): each dim → {stratum value → block}.
 *  Strata are sparse (a None key — e.g. unknown stmt_type — is dropped). */
export interface StratificationLayer {
	by_stmt_type: Record<string, StratumBlock>;
	by_n_sources: Record<string, StratumBlock>;
	by_n_evidence: Record<string, StratumBlock>;
	by_dominant_bucket: Record<string, StratumBlock>;
	by_driver: Record<string, StratumBlock>;
}

/** One tier (ev = Tier-1 per-evidence, stmt = Tier-2 per-statement).
 *  verdict_err + stratified are Tier-2-only and present from schema v2; legacy
 *  exports omit them, so consumers narrow on presence. */
export type TierBlock =
	| { status: 'unavailable'; reason: string }
	| {
			status: 'available';
			n: number;
			base_rate_correct: number;
			arms: Record<string, ArmSlot>;
			verdict_err?: ConfusionMetrics;
			stratified?: StratificationLayer;
	  };

/** The per-run metrics.json contract (E5). `metrics_basis` is part of the
 * comparison identity; only its reader-specific `soft_calibration` profile may
 * differ between otherwise compatible reader runs. */
export interface RunMetrics {
	schema_version: number;
	run_id: string | null;
	model: string | null;
	generated_date: string;
	/** Immutable artifact/evaluation identity. Required by the v3 hybrid contract. */
	provenance?: CalibrationProvenance;
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

/** A statement rendered to a human-readable subject/relation/object triple for
 *  display on the curate page. `full` is the flat fallback string. */
export interface Claim {
	subject: string;
	relation: string;
	object: string;
	full: string;
}

export interface EvidenceAgent {
	role: 'subject' | 'object' | 'member' | 'agent';
	name: string;
	rawText: string | null;
	dbRefs: Record<string, string>;
}

/** One (statement, evidence) pair sampled live from the INDRA DB for curation.
 *
 *  CRITICAL: matchesHash and sourceHash are kept as STRINGS, not numbers. INDRA
 *  hashes are 64-bit ints that exceed Number.MAX_SAFE_INTEGER (e.g.
 *  -2318097519188363613), so coercing them to a JS number silently corrupts the
 *  low digits — which would curate the WRONG evidence. The INDRA REST API returns
 *  them already quoted as strings; we preserve that all the way to submission,
 *  where sourceHash is injected into the POST body as a bare integer literal.
 *  (CurationRow above uses number because that read path only ever joins
 *  same-lossy-to-same-lossy and never submits.) */
export interface EvidenceSample {
	/** statement pa_hash / matches_hash — exact digits as a string. */
	matchesHash: string;
	/** evidence source_hash — exact digits as a string. */
	sourceHash: string;
	/** the evidence sentence the curator judges the extraction against. */
	text: string;
	pmid: string | null;
	pmcid: string | null;
	sourceApi: string | null;
	stmtType: string;
	belief: number | null;
	claim: Claim;
	/** Statement agents plus evidence surface strings, when INDRA provides them. */
	agents: EvidenceAgent[];
	/** id of the dataset (universe) this pair was drawn from — the active universe
	 *  threaded to the persistent frame header on /curate (see datasets.ts). */
	dataset: string;
	/** total evidence supporting this statement (context for the curator). */
	evCount: number;
}
