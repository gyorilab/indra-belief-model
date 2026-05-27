import { error } from '@sveltejs/kit';
import { activePublicWriterLock, type PublicWriterLockState } from './pairedState';

function writerInProgressMessage(lock: PublicWriterLockState): string {
	const arch = lock.architecture ? ` ${lock.architecture}` : '';
	const pair = lock.pair_id ? ` pair ${lock.pair_id}` : '';
	return `writer_in_progress: DuckDB writer is active with ${lock.kind}${arch}${pair}; page reads pause until it finishes.`;
}

function malformedWriterLockMessage(lock: PublicWriterLockState): string {
	const detail = lock.malformed_reason ? ` (${lock.malformed_reason})` : '';
	return `writer_lock_malformed: DuckDB writer lock state is malformed${detail}; page reads pause until viewer_state/writer_lock.json is repaired or removed after confirming no writer is active.`;
}

export function assertNoActiveWriter(): void {
	const writerLock = activePublicWriterLock();
	if (!writerLock) return;
	if (process.env.INDRA_VIEWER_SUPPRESS_WRITER_GUARD_LOG !== '1') {
		console.warn(
			writerLock.malformed_reason
				? 'writer_lock_malformed: page read blocked'
				: 'writer_in_progress: page read blocked',
			{
				kind: writerLock.kind,
				pair_id: writerLock.pair_id,
				architecture: writerLock.architecture,
				pid: writerLock.pid,
				malformed_reason: writerLock.malformed_reason
			}
		);
	}
	if (writerLock.malformed_reason) {
		throw error(503, {
			code: 'writer_lock_malformed',
			message: malformedWriterLockMessage(writerLock)
		});
	}
	throw error(503, {
		code: 'writer_in_progress',
		message: writerInProgressMessage(writerLock)
	});
}
