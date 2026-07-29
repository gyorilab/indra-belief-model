/** Persistent, exact-pair reservations for the curation sampler.
 *
 * A completed-curation history alone is not enough for sampling without
 * replacement: a skipped card, refresh, or concurrent tab could draw the same
 * pair again before it was submitted.  Each card is therefore reserved in an
 * on-disk ledger before it is returned to the browser.  Exclusive file creation
 * is the atomic boundary, so concurrent viewer processes sharing the same data
 * directory cannot reserve the same curator/dataset/pair.
 *
 * The default ledger lives under ignored `data/results/`; deployments may point
 * CURATION_DRAW_LEDGER_DIR at a persistent shared filesystem.  If that storage
 * is unavailable, sampling fails closed instead of silently reverting to draws
 * with replacement.
 */
import { createHash, randomUUID, timingSafeEqual } from 'node:crypto';
import {
	existsSync,
	mkdirSync,
	readFileSync,
	readdirSync,
	renameSync,
	rmSync,
	writeFileSync
} from 'node:fs';
import { dirname, resolve } from 'node:path';

const PAIR_RE = /^-?\d+:-?\d+$/;

interface ReservationRecord {
	schemaVersion: 1;
	dataset: string;
	pairKey: string;
	token: string;
	drawnAt: string;
	committedAt?: string;
}

function digest(value: string): string {
	return createHash('sha256').update(value).digest('hex');
}

function ledgerRoot(): string {
	const configured = process.env.CURATION_DRAW_LEDGER_DIR?.trim();
	if (
		process.env.NODE_ENV === 'production' &&
		(!configured || process.env.CURATION_DRAW_LEDGER_SHARED !== '1')
	) {
		throw new Error(
			'production curation sampling requires CURATION_DRAW_LEDGER_DIR on persistent shared storage ' +
				'and CURATION_DRAW_LEDGER_SHARED=1'
		);
	}
	return configured
		? resolve(configured)
		: resolve(process.cwd(), '..', 'data', 'results', 'curation_draw_ledger');
}

function validateDataset(dataset: string): void {
	if (!/^[a-z0-9_-]+$/i.test(dataset)) throw new Error('invalid curation dataset identity');
}

function curatorDir(email: string): string {
	if (!email.trim()) throw new Error('cannot reserve a draw without a curator identity');
	return resolve(ledgerRoot(), digest(email.trim().toLowerCase()));
}

function reservationPath(email: string, dataset: string, pairKey: string): string {
	validateDataset(dataset);
	if (!PAIR_RE.test(pairKey)) throw new Error('invalid exact pair reservation key');
	// Pair identity is curator-global, not dataset-scoped. The originating dataset
	// remains inside the record as provenance, while one atomic path prevents a
	// cross-lane duplicate when two datasets contain the same exact pair.
	return resolve(curatorDir(email), '_pairs', `${digest(pairKey)}.json`);
}

function legacyReservationPath(email: string, dataset: string, pairKey: string): string {
	return resolve(curatorDir(email), dataset, `${digest(pairKey)}.json`);
}

function findReservation(
	email: string,
	dataset: string,
	pairKey: string
): { path: string; row: ReservationRecord } | null {
	for (const path of [
		reservationPath(email, dataset, pairKey),
		legacyReservationPath(email, dataset, pairKey)
	]) {
		try {
			const row = JSON.parse(readFileSync(path, 'utf8')) as ReservationRecord;
			if (row.schemaVersion !== 1 || !PAIR_RE.test(row.pairKey) || row.pairKey !== pairKey) {
				throw new Error('curation draw ledger record does not match its path');
			}
			// A token may only submit through the dataset that produced the card.
			if (row.dataset !== dataset) return null;
			return { path, row };
		} catch (error) {
			if ((error as NodeJS.ErrnoException).code === 'ENOENT') continue;
			throw error;
		}
	}
	return null;
}

function claimPath(email: string, pairKey: string): string {
	return resolve(curatorDir(email), '_claims', `${digest(pairKey)}.claim`);
}

function sameToken(a: string, b: string): boolean {
	const aa = Buffer.from(a);
	const bb = Buffer.from(b);
	return aa.length === bb.length && timingSafeEqual(aa, bb);
}

/** Every pair already drawn for this curator in any dataset, submitted or not. */
export function reservedPairKeys(email: string, dataset: string): Set<string> {
	validateDataset(dataset);
	const dir = curatorDir(email);
	let names: string[];
	try {
		names = (readdirSync(dir, { recursive: true }) as string[]).filter((name) => name.endsWith('.json'));
	} catch (error) {
		if ((error as NodeJS.ErrnoException).code === 'ENOENT') return new Set();
		throw error;
	}
	const keys = new Set<string>();
	for (const name of names) {
		const row = JSON.parse(readFileSync(resolve(dir, name), 'utf8')) as ReservationRecord;
		if (
			row.schemaVersion !== 1 ||
			!/^[a-z0-9_-]+$/i.test(row.dataset) ||
			!PAIR_RE.test(row.pairKey)
		) {
			throw new Error(`invalid curation draw ledger record: ${name}`);
		}
		keys.add(row.pairKey);
	}
	return keys;
}

/** Atomically reserve a newly drawn pair. Null means another request got it. */
export function tryReserveDraw(email: string, dataset: string, pairKey: string): string | null {
	if (reservedPairKeys(email, dataset).has(pairKey)) return null;
	const path = reservationPath(email, dataset, pairKey);
	mkdirSync(dirname(path), { recursive: true });
	const token = randomUUID();
	const row: ReservationRecord = {
		schemaVersion: 1,
		dataset,
		pairKey,
		token,
		drawnAt: new Date().toISOString()
	};
	try {
		writeFileSync(path, `${JSON.stringify(row)}\n`, {
			flag: 'wx',
			mode: 0o600
		});
		return token;
	} catch (error) {
		if ((error as NodeJS.ErrnoException).code === 'EEXIST') return null;
		throw error;
	}
}

/** Claim a reservation exactly once while its curation POST is in flight. */
export function claimDrawReservation(
	email: string,
	dataset: string,
	pairKey: string,
	token: string
): boolean {
	const reservation = findReservation(email, dataset, pairKey);
	if (!reservation || reservation.row.committedAt || !sameToken(reservation.row.token, token)) return false;
	const path = claimPath(email, pairKey);
	// Respect a claim created by the immediately preceding dataset-scoped ledger
	// layout during a quiescent upgrade. All new claims use the curator-global
	// pair path below, including submissions backed by a legacy reservation.
	if (existsSync(`${reservation.path}.claim`)) return false;
	mkdirSync(dirname(path), { recursive: true });
	try {
		writeFileSync(path, `${new Date().toISOString()}\n`, {
			flag: 'wx',
			mode: 0o600
		});
		return true;
	} catch (error) {
		// Never reclaim a claim by age: INDRA has no idempotency key or fencing
		// token, so a paused request could resume after a TTL and race the new
		// owner. A crash therefore fails this pair closed until an operator checks
		// shared INDRA history and manually removes the orphaned `.claim` file.
		if ((error as NodeJS.ErrnoException).code === 'EEXIST') return false;
		throw error;
	}
}

/** Release an in-flight claim after a rejected/failed remote submission. */
export function releaseDrawClaim(email: string, dataset: string, pairKey: string): void {
	const reservation = findReservation(email, dataset, pairKey);
	if (reservation) {
		rmSync(claimPath(email, pairKey), { force: true });
		rmSync(`${reservation.path}.claim`, { force: true });
	}
}

/** Whether an upstream failure proves that no remote write was accepted.
 *
 * Network errors, 5xx, Request Timeout, Too Early, and nginx-style 499 outcomes
 * are ambiguous and must retain the claim. Ordinary client rejections are safe
 * to retry after repairing the request.
 */
export function submissionFailureIsDefinitive(status: number | undefined): boolean {
	return status != null && status >= 400 && status < 500 && ![408, 425, 499].includes(status);
}

/** Mark a successfully submitted reservation immutable and consumed. */
export function commitDrawReservation(
	email: string,
	dataset: string,
	pairKey: string,
	token: string
): void {
	const reservation = findReservation(email, dataset, pairKey);
	if (!reservation || !sameToken(reservation.row.token, token)) {
		throw new Error('curation draw reservation token mismatch');
	}
	const { path, row } = reservation;
	const tmp = `${path}.${process.pid}.${randomUUID()}.tmp`;
	writeFileSync(tmp, `${JSON.stringify({ ...row, committedAt: new Date().toISOString() })}\n`, {
		flag: 'wx',
		mode: 0o600
	});
	renameSync(tmp, path);
	rmSync(claimPath(email, pairKey), { force: true });
	rmSync(`${path}.claim`, { force: true });
}
