import { DuckDBInstance } from '@duckdb/node-api';
import { closeInstance, dbPath } from '$lib/db';

export const OBSERVED_COST_SQL = `
	SELECT SUM(
		CASE model_id
			WHEN 'claude-haiku-4-5' THEN COALESCE(prompt_tokens, 0) * 0.80 / 1000000.0 + COALESCE(out_tokens, 0) * 4.00 / 1000000.0
			WHEN 'claude-sonnet-4-6' THEN COALESCE(prompt_tokens, 0) * 3.00 / 1000000.0 + COALESCE(out_tokens, 0) * 15.00 / 1000000.0
			WHEN 'claude-opus-4-7' THEN COALESCE(prompt_tokens, 0) * 15.00 / 1000000.0 + COALESCE(out_tokens, 0) * 75.00 / 1000000.0
			WHEN 'gemini-2.5-flash' THEN COALESCE(prompt_tokens, 0) * 0.075 / 1000000.0 + COALESCE(out_tokens, 0) * 0.30 / 1000000.0
			WHEN 'gemini-2.5-pro' THEN COALESCE(prompt_tokens, 0) * 1.25 / 1000000.0 + COALESCE(out_tokens, 0) * 5.00 / 1000000.0
			WHEN 'gpt-4o' THEN COALESCE(prompt_tokens, 0) * 2.50 / 1000000.0 + COALESCE(out_tokens, 0) * 10.00 / 1000000.0
			WHEN 'gpt-4o-mini' THEN COALESCE(prompt_tokens, 0) * 0.15 / 1000000.0 + COALESCE(out_tokens, 0) * 0.60 / 1000000.0
			WHEN 'mock' THEN 0.0
			WHEN 'mock-model' THEN 0.0
			WHEN 'smoke-local' THEN 0.0
			WHEN 'unknown' THEN 0.0
			ELSE NULL
		END
	)
	  FROM scorer_step
	 WHERE run_id=?
	   AND step_kind='aggregate'
`;

export interface MarkScoreRunCanceledInput {
	run_id: string;
	scorer_version: string;
	architecture: string;
	model: string;
	paired_run_group_id?: string;
	parent_run_id?: string;
	reason: string;
}

function sleep(ms: number): Promise<void> {
	return new Promise((resolve) => setTimeout(resolve, ms));
}

function isDuckDBLockError(err: unknown): boolean {
	const message = err instanceof Error ? err.message : String(err);
	return message.includes('Could not set lock') || message.includes('Conflicting lock');
}

async function writeScoreRunCanceled(input: MarkScoreRunCanceledInput): Promise<void> {
	await closeInstance();
	const instance = await DuckDBInstance.create(dbPath());
	const con = await instance.connect();
	try {
		await con.run(
			`UPDATE score_run
			    SET status='canceled',
			        finished_at=COALESCE(finished_at, CURRENT_TIMESTAMP),
			        terminated_by=COALESCE(terminated_by, 'user'),
			        termination_reason=COALESCE(termination_reason, ?),
			        cost_actual_usd=COALESCE(cost_actual_usd, (${OBSERVED_COST_SQL}))
			  WHERE run_id=?
			    AND status IN ('running', 'failed')`,
			[input.reason, input.run_id, input.run_id]
		);
		await con.run(
			`INSERT INTO score_run
			   (run_id, scorer_version, indra_version, architecture,
			    paired_run_group_id, parent_run_id, model_id_default,
			    started_at, finished_at, n_stmts, status,
			    cost_actual_usd, terminated_by, termination_reason)
			 SELECT ?, ?, 'unknown', ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP,
			        0, 'canceled', NULL, 'user', ?
			  WHERE NOT EXISTS (SELECT 1 FROM score_run WHERE run_id=?)`,
			[
				input.run_id,
				input.scorer_version,
				input.architecture,
				input.paired_run_group_id ?? null,
				input.parent_run_id ?? null,
				input.model,
				input.reason,
				input.run_id
			]
		);
	} finally {
		con.disconnectSync?.();
		instance.closeSync();
	}
}

export async function markScoreRunCanceled(input: MarkScoreRunCanceledInput): Promise<void> {
	let lastErr: unknown = null;
	for (let attempt = 0; attempt < 10; attempt += 1) {
		try {
			await writeScoreRunCanceled(input);
			return;
		} catch (err) {
			lastErr = err;
			if (!isDuckDBLockError(err) || attempt === 9) break;
			await sleep(150 * (attempt + 1));
		}
	}
	throw lastErr instanceof Error ? lastErr : new Error(String(lastErr));
}

export interface MarkScoreRunPreStartedCancelledInput {
	run_id: string;
	scorer_version: string;
	architecture: string;
	model: string;
	paired_run_group_id?: string;
	parent_run_id?: string;
	reason: string;
}

async function writeScoreRunPreStartedCancelled(
	input: MarkScoreRunPreStartedCancelledInput
): Promise<void> {
	await closeInstance();
	const instance = await DuckDBInstance.create(dbPath());
	const con = await instance.connect();
	try {
		await con.run(
			`INSERT INTO score_run
			   (run_id, scorer_version, indra_version, architecture,
			    paired_run_group_id, parent_run_id, model_id_default,
			    started_at, finished_at, n_stmts, status,
			    cost_actual_usd, terminated_by, termination_reason)
			 SELECT ?, ?, 'unknown', ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP,
			        0, 'pre_started_cancelled', 0, 'user', ?
			  WHERE NOT EXISTS (SELECT 1 FROM score_run WHERE run_id=?)`,
			[
				input.run_id,
				input.scorer_version,
				input.architecture,
				input.paired_run_group_id ?? null,
				input.parent_run_id ?? null,
				input.model,
				input.reason,
				input.run_id
			]
		);
	} finally {
		con.disconnectSync?.();
		instance.closeSync();
	}
}

export async function markScoreRunPreStartedCancelled(
	input: MarkScoreRunPreStartedCancelledInput
): Promise<void> {
	let lastErr: unknown = null;
	for (let attempt = 0; attempt < 10; attempt += 1) {
		try {
			await writeScoreRunPreStartedCancelled(input);
			return;
		} catch (err) {
			lastErr = err;
			if (!isDuckDBLockError(err) || attempt === 9) break;
			await sleep(150 * (attempt + 1));
		}
	}
	throw lastErr instanceof Error ? lastErr : new Error(String(lastErr));
}
