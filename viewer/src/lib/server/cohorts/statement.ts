// Auto-extracted from $lib/db.ts on 2026-05-27 to satisfy B4 of the
// active goal (cohort orchestration extraction).
//
// All callers continue to import these symbols from $lib/db; db.ts
// re-exports each function from the corresponding $lib/server/cohorts/*
// module. Shared internal helpers (sqlQuote, rows, readRows, scalar,
// tableExists, normalizeDuckValue, timestampMs, durationSeconds,
// safeRate, repairRerunLineageSqlOptions, etc.) and connection
// management (connect, closeInstance, dbPath, dbExists) still live in
// $lib/db; this file imports them from there.

import { PROBE_KINDS, closeInstance, connect, dbExists, dbPath, durationSeconds, normalizeDuckValue, readRows, repairRerunLineageSqlOptions, rows, safeRate, scalar, sqlQuote, tableExists, timestampMs } from '$lib/db';
import type { AgentRow, EvidenceRow, FocusStatement, StatementDetail, SupportsEdgeRow, TruthLabelRow } from '$lib/db';
import type { ProbeAttribution, ProbeConfidence, ProbeKind, ProbeOutput, ProbeSource } from '$lib/probeAttribution';
import { computeAttributions, summarizeAcrossEvidences } from '$lib/probeAttribution';
import { error } from '@sveltejs/kit';

import type { StatementRunOption, ScorerStepRow } from '$lib/db';
/**
 * Pick a focus statement to lead the dashboard with. Defaults to the
 * highest-|Δ vs INDRA| in the latest succeeded run; can be deep-linked to a
 * specific stmt_hash. Returns null if there's no scoring data yet.
 */
export async function getFocusStatement(
	focus_hash?: string,
	run_id?: string
): Promise<FocusStatement | null> {
	if (!dbExists()) return null;
	const con = await connect();
	try {
		let resolvedRun = run_id ?? null;
		if (!resolvedRun) {
			const r = await rows<{ run_id: string }>(
				con,
				`SELECT run_id FROM score_run WHERE status='succeeded' ORDER BY started_at DESC LIMIT 1`
			);
			resolvedRun = r[0]?.run_id ?? null;
		}
		if (!resolvedRun) return null;

		let resolvedHash = focus_hash ?? null;
		let whyKind: 'biggest_delta' | 'requested' = 'biggest_delta';
		let biggestDelta: number | null = null;
		if (!resolvedHash) {
			const r = await rows<{ stmt_hash: string; delta: number }>(
				con,
				`WITH ours AS (
					SELECT stmt_hash,
					       AVG(CAST(json_extract(output_json, '$.score') AS DOUBLE)) AS our
					FROM scorer_step
					WHERE run_id = '${resolvedRun.replace(/'/g, "''")}'
					  AND step_kind = 'aggregate'
					  AND json_extract(output_json, '$.score') IS NOT NULL
					GROUP BY stmt_hash
				)
				SELECT s.stmt_hash AS stmt_hash,
				       ours.our - s.indra_belief AS delta
				FROM statement s
				JOIN ours ON ours.stmt_hash = s.stmt_hash
				WHERE s.indra_belief IS NOT NULL
				ORDER BY ABS(ours.our - s.indra_belief) DESC, s.stmt_hash
				LIMIT 1`
			);
			if (r.length === 0) return null;
			resolvedHash = r[0].stmt_hash;
			biggestDelta = r[0].delta;
		} else {
			whyKind = 'requested';
		}

		const stmtRows = await rows<{
			stmt_hash: string;
			indra_type: string;
			indra_belief: number | null;
			our_score: number | null;
			n_evidences: number;
		}>(
			con,
			`WITH ours AS (
				SELECT stmt_hash,
				       AVG(CAST(json_extract(output_json, '$.score') AS DOUBLE)) AS our
				FROM scorer_step
				WHERE run_id = '${resolvedRun.replace(/'/g, "''")}'
				  AND step_kind = 'aggregate'
				  AND json_extract(output_json, '$.score') IS NOT NULL
				GROUP BY stmt_hash
			)
			SELECT s.stmt_hash, s.indra_type, s.indra_belief,
			       ours.our AS our_score,
			       (SELECT COUNT(*) FROM statement_evidence WHERE stmt_hash = s.stmt_hash) AS n_evidences
			FROM statement s
			LEFT JOIN ours ON ours.stmt_hash = s.stmt_hash
			WHERE s.stmt_hash = '${resolvedHash.replace(/'/g, "''")}'`
		);
		if (stmtRows.length === 0) return null;
		const s = stmtRows[0];

		const [agents, evidences, probes] = await Promise.all([
			rows<{ role: string; name: string }>(
				con,
				`SELECT role, name FROM agent
				 WHERE stmt_hash = '${resolvedHash.replace(/'/g, "''")}'
				 ORDER BY CASE role
				   WHEN 'subj' THEN 0 WHEN 'enz' THEN 0
				   WHEN 'obj' THEN 1 WHEN 'sub' THEN 1
				   WHEN 'member' THEN 2 ELSE 3 END, role_index`
			),
			rows<{ evidence_hash: string; source_api: string | null; text: string | null }>(
				con,
				`SELECT e.evidence_hash, e.source_api, e.text
				 FROM statement_evidence se
				 JOIN evidence e ON e.evidence_hash = se.evidence_hash
				 WHERE se.stmt_hash = '${resolvedHash.replace(/'/g, "''")}'
				 ORDER BY (e.text IS NULL), length(e.text) DESC LIMIT 3`
			),
			getProbeAttribution(resolvedRun, resolvedHash)
		]);

		const evPlural = s.n_evidences === 1 ? '' : 's';
		const evText = `${s.n_evidences} evidence${evPlural}`;
		// `why_this_one` only renders when the system chose the focus editorially
		// (largest |Δ|). When the user deep-linked to a stmt_hash, the URL
		// already explains why they're here — silencing avoids self-evident
		// bookkeeping ("opened via deep-link").
		const whyText = whyKind === 'biggest_delta'
			? `the largest disagreement with INDRA in this run · ${evText}`
			: '';

		return {
			run_id: resolvedRun,
			stmt: { stmt_hash: s.stmt_hash, indra_type: s.indra_type, agents },
			our_score: s.our_score,
			indra_score: s.indra_belief,
			probes,
			evidences,
			n_evidences: s.n_evidences,
			why_this_one: whyText
		};
	} finally {
		con.disconnectSync?.();
	}
}


/**
 * For a (run_id, stmt_hash), return the four probes' contributions to the
 * final score. When `evidence_hash` is supplied, returns the per-evidence
 * view; otherwise picks the highest-confidence answer per probe across all
 * evidences for that statement. Pure logic lives in probeAttribution.ts.
 */
export async function getProbeAttribution(
	run_id: string,
	stmt_hash: string,
	evidence_hash?: string
): Promise<ProbeAttribution[]> {
	if (!dbExists()) return [];
	const con = await connect();
	try {
		const evClause = evidence_hash
			? `AND evidence_hash = '${evidence_hash.replace(/'/g, "''")}'`
			: '';
		const [stepRows, substrateRows] = await Promise.all([
			rows<{
				step_kind: string;
				evidence_hash: string | null;
				answer: string | null;
				confidence: string | null;
				source: string | null;
				rationale: string | null;
			}>(
				con,
				`SELECT
				   step_kind,
				   evidence_hash,
				   json_extract_string(output_json, '$.answer') AS answer,
				   json_extract_string(output_json, '$.confidence') AS confidence,
				   json_extract_string(output_json, '$.source') AS source,
				   json_extract_string(output_json, '$.rationale') AS rationale
				 FROM scorer_step
				 WHERE run_id = '${run_id.replace(/'/g, "''")}'
				   AND stmt_hash = '${stmt_hash.replace(/'/g, "''")}'
				   AND step_kind IN ('subject_role_probe', 'object_role_probe', 'relation_axis_probe', 'scope_probe')
				   ${evClause}`
			),
			rows<{ evidence_hash: string | null; output_json: string }>(
				con,
				`SELECT evidence_hash, output_json::VARCHAR AS output_json
				 FROM scorer_step
				 WHERE run_id = '${run_id.replace(/'/g, "''")}'
				   AND stmt_hash = '${stmt_hash.replace(/'/g, "''")}'
				   AND step_kind = 'substrate_route'
				   ${evClause}`
			)
		]);

		const probeOutputs: ProbeOutput[] = [];
		const seen = new Set<string>();
		for (const r of stepRows) {
			const probe = PROBE_KINDS[r.step_kind];
			if (!probe) continue;
			probeOutputs.push({
				probe,
				evidence_hash: r.evidence_hash,
				answer: r.answer,
				confidence: (r.confidence as ProbeConfidence) ?? null,
				source: (r.source as ProbeSource) ?? null,
				rationale: r.rationale
			});
			seen.add(`${r.evidence_hash ?? ''}|${probe}`);
		}

		for (const sr of substrateRows) {
			let parsed: Record<string, { source?: string; answer?: string; confidence?: string }> | null = null;
			try {
				parsed = JSON.parse(sr.output_json);
			} catch {
				continue;
			}
			if (!parsed) continue;
			const probesInRow: ProbeKind[] = ['subject_role', 'object_role', 'relation_axis', 'scope'];
			for (const probe of probesInRow) {
				if (seen.has(`${sr.evidence_hash ?? ''}|${probe}`)) continue;
				const slot = parsed[probe];
				if (!slot || !slot.answer) continue;
				probeOutputs.push({
					probe,
					evidence_hash: sr.evidence_hash,
					answer: slot.answer ?? null,
					confidence: (slot.confidence as ProbeConfidence) ?? null,
					source: (slot.source as ProbeSource) ?? 'substrate',
					rationale: `substrate-resolved (no LLM call)`
				});
				seen.add(`${sr.evidence_hash ?? ''}|${probe}`);
			}
		}

		if (evidence_hash) {
			return computeAttributions(probeOutputs);
		}
		return computeAttributions(summarizeAcrossEvidences(probeOutputs));
	} finally {
		con.disconnectSync?.();
	}
}


export async function getStatementDetail(stmt_hash: string, run_id?: string | null): Promise<StatementDetail | null> {
	if (!dbExists()) return null;
	const con = await connect();
	try {
		const qStmt = stmt_hash.replace(/'/g, "''");
		const stmtRows = await rows<{
			stmt_hash: string;
			indra_type: string;
			indra_belief: number | null;
			supports_count: number;
			supported_by_count: number;
			source_dump_id: string | null;
			raw_json: string;
		}>(
			con,
			`SELECT stmt_hash, indra_type, indra_belief, supports_count,
			        supported_by_count, source_dump_id, raw_json::VARCHAR AS raw_json
			 FROM statement WHERE stmt_hash = '${qStmt}'`
		);
		if (stmtRows.length === 0) return null;
		const s = stmtRows[0];

		const available_runs = await rows<StatementRunOption>(
			con,
			`SELECT DISTINCT
			   sr.run_id,
			   sr.architecture,
			   sr.scorer_version,
			   sr.model_id_default,
			   sr.started_at::VARCHAR AS started_at,
			   sr.status
			 FROM score_run sr
			 JOIN scorer_step ss ON ss.run_id = sr.run_id
			 WHERE ss.stmt_hash = '${qStmt}'
			 ORDER BY sr.started_at DESC`
		);
		const requestedRun = run_id ? run_id.replace(/'/g, "''") : null;
		const selectedRun =
			(requestedRun
				? available_runs.find((r) => r.run_id === requestedRun)
				: available_runs.find((r) => r.status === 'succeeded') ?? available_runs[0]) ?? null;
		const scorerStepRunClause = selectedRun
			? `AND run_id = '${selectedRun.run_id.replace(/'/g, "''")}'`
			: 'AND 1=0';

		const [agents, evidences, truth_labels, registered_truth_sets_rows, supports_edges, scorer_steps] = await Promise.all([
			rows<AgentRow>(
				con,
				`SELECT agent_hash, role, role_index, name,
				        db_refs_json::VARCHAR AS db_refs_json,
				        mods_json::VARCHAR AS mods_json,
				        location
				 FROM agent
				 WHERE stmt_hash = '${qStmt}'
				 ORDER BY
				   CASE role
				     WHEN 'subj' THEN 0
				     WHEN 'enz'  THEN 0
				     WHEN 'obj'  THEN 1
				     WHEN 'sub'  THEN 1
				     WHEN 'member' THEN 2
				     ELSE 3
				   END,
				   role_index`
			),
			rows<EvidenceRow>(
				con,
				`SELECT e.evidence_hash, e.source_api, e.source_id, e.pmid, e.text,
				        e.is_direct, e.is_negated, e.is_curated,
				        e.epistemics_json::VARCHAR AS epistemics_json
				 FROM statement_evidence se
				 JOIN evidence e ON e.evidence_hash = se.evidence_hash
				 WHERE se.stmt_hash = '${qStmt}'
				 ORDER BY e.source_api, e.evidence_hash`
			),
			rows<TruthLabelRow>(
				con,
				`SELECT truth_set_id, target_kind, target_id, field, value_text,
				        value_json::VARCHAR AS value_json, provenance
				 FROM truth_label
				 WHERE (target_kind = 'stmt' AND target_id = '${qStmt}')
				    OR (target_kind = 'evidence' AND target_id IN (
				          SELECT evidence_hash FROM statement_evidence WHERE stmt_hash = '${qStmt}'))
				    OR (target_kind = 'agent' AND target_id IN (
				          SELECT agent_hash FROM agent WHERE stmt_hash = '${qStmt}'))
				 ORDER BY truth_set_id, field`
			),
			rows<{ id: string }>(
				con,
				'SELECT id FROM truth_set ORDER BY id'
			),
			rows<SupportsEdgeRow>(
				con,
				`SELECT from_stmt_hash, to_stmt_hash, kind
				 FROM supports_edge
				 WHERE from_stmt_hash = '${qStmt}'
				 ORDER BY kind, to_stmt_hash`
			),
			rows<ScorerStepRow>(
				con,
				`SELECT step_hash, run_id, evidence_hash, scorer_version,
				        architecture, model_id, step_kind, is_substrate_answered,
				        input_payload_json::VARCHAR AS input_payload_json,
				        output_json::VARCHAR AS output_json,
				        latency_ms, prompt_tokens, out_tokens, finish_reason, error,
				        started_at::VARCHAR AS started_at
				 FROM scorer_step
				 WHERE stmt_hash = '${qStmt}' ${scorerStepRunClause}
				 ORDER BY evidence_hash,
				   CASE step_kind
				     WHEN 'parse_claim' THEN 0
				     WHEN 'build_context' THEN 1
				     WHEN 'substrate_route' THEN 2
				     WHEN 'subject_role_probe' THEN 3
				     WHEN 'object_role_probe' THEN 4
				     WHEN 'relation_axis_probe' THEN 5
				     WHEN 'scope_probe' THEN 6
				     WHEN 'grounding' THEN 7
				     WHEN 'adjudicate' THEN 8
				     WHEN 'aggregate' THEN 9
				     ELSE 99
				   END,
				   started_at DESC,
				   step_hash DESC`
			)
		]);

		return {
			...s,
			selected_run_id: selectedRun?.run_id ?? null,
			selected_architecture: selectedRun?.architecture ?? null,
			selected_scorer_version: selectedRun?.scorer_version ?? null,
			selected_run_status: selectedRun?.status ?? null,
			available_runs,
			agents,
			evidences,
			truth_labels,
			registered_truth_sets: registered_truth_sets_rows.map((r) => r.id),
			supports_edges,
			scorer_steps
		};
	} finally {
		con.disconnectSync?.();
	}
}
