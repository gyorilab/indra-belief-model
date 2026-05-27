export const PAIRED_RUN_CONFLICT = {
	pairedGroupIdTaken: 'paired_group_id_taken',
	pairedWorkflowStateExists: 'paired_workflow_state_exists',
	pairedWorkflowActive: 'paired_workflow_active',
	writerInProgress: 'writer_in_progress',
	writerLockMalformed: 'writer_lock_malformed'
} as const;

export const PAIRED_RUN_BLOCKED_CODES = [
	PAIRED_RUN_CONFLICT.pairedGroupIdTaken,
	PAIRED_RUN_CONFLICT.pairedWorkflowStateExists,
	PAIRED_RUN_CONFLICT.pairedWorkflowActive,
	PAIRED_RUN_CONFLICT.writerInProgress,
	PAIRED_RUN_CONFLICT.writerLockMalformed
] as const;

export type PairedRunBlockedCode = typeof PAIRED_RUN_BLOCKED_CODES[number];

export function isPairedRunBlockedCode(code: string | null | undefined): code is PairedRunBlockedCode {
	return PAIRED_RUN_BLOCKED_CODES.includes(code as PairedRunBlockedCode);
}
