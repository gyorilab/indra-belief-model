/**
 * The JSONL-derived data model — a read-only projection of the monolithic
 * pipeline's exports (`data/exports/<run>/{per_statement.json, per_evidence.jsonl,
 * export_meta.json}`). This replaces the DuckDB corpus schema entirely.
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
	/** From the raw run's .meta.json, when present. */
	status: string | null;
	started_at: string | null;
	finished_at: string | null;
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
	grounding_status: string | null;
	tier: string | null;
	provenance_triggered?: boolean;
	/** Report taxonomy bucket (semantic_correct, reader_hallucination, …). */
	bucket: string | null;
	bucket_group: string | null;
	error: string | null;
	latency_s?: number | null;
	tokens?: number | null;
}
