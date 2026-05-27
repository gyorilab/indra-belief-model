import { PAIRED_RUN_CONFLICT } from './pairedRunConflicts';

export const WRITER_ACTION_CONFLICT = {
	pairedWorkflowActive: PAIRED_RUN_CONFLICT.pairedWorkflowActive,
	writerLockBusy: 'writer_lock_busy',
	writerLockMalformed: PAIRED_RUN_CONFLICT.writerLockMalformed
} as const;

export const WRITER_ACTION_BLOCKED_CODES = [
	WRITER_ACTION_CONFLICT.pairedWorkflowActive,
	WRITER_ACTION_CONFLICT.writerLockBusy,
	WRITER_ACTION_CONFLICT.writerLockMalformed
] as const;

export type WriterActionBlockedCode = typeof WRITER_ACTION_BLOCKED_CODES[number];

export function isWriterActionBlockedCode(code: string | null | undefined): code is WriterActionBlockedCode {
	return WRITER_ACTION_BLOCKED_CODES.includes(code as WriterActionBlockedCode);
}
