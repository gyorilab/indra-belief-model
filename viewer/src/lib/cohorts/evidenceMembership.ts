// Renamed from $lib/evidenceMembershipSql.ts on 2026-05-27 (B4 of deferred hypergraph).
// No code changes vs the prior location; see git mv for history.

// Membership SQL helpers. After Phase 1 of the active hypergraph goal,
// `statement_evidence` is the sole contextual membership surface; the
// legacy `evidence.stmt_hash` column is dropped.
export const STATEMENT_EVIDENCE_RELATION = 'statement_evidence';

export function statementEvidenceCountSql(stmtExpr: string): string {
	return `(SELECT COUNT(*) FROM statement_evidence se_count
	         WHERE se_count.stmt_hash=${stmtExpr})`;
}

export function statementEvidenceSourceStratumSql(stmtExpr: string): string {
	return `(SELECT MIN(e_src.source_api)
	           FROM statement_evidence se_src
	           JOIN evidence e_src ON e_src.evidence_hash=se_src.evidence_hash
	          WHERE se_src.stmt_hash=${stmtExpr})`;
}
