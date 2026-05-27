import { createServer } from 'vite';
import { createHmac } from 'node:crypto';
import assert from 'node:assert/strict';

// Phase 4 gate: signed/stateless repair estimate tokens. The implementation
// in viewer/src/lib/server/repairBacklog.ts replaces a process-local Map
// with HMAC-SHA256-signed tokens of the form `v1.<fp>.<expiresAt>.<sig>`.
// Any process holding INDRA_VIEWER_REPAIR_TOKEN_SECRET can verify a token
// issued by any other process; this test simulates the two-process round-
// trip by re-deriving the expected HMAC outside the module and confirming
// it matches what the issuer produced.

const SHARED_SECRET = 'phase-4-shared-test-secret-must-be-32-chars-long-12345';
assert.ok(SHARED_SECRET.length >= 32, 'shared test secret too short');
process.env.INDRA_VIEWER_REPAIR_TOKEN_SECRET = SHARED_SECRET;

const server = await createServer({
	configFile: './vite.config.ts',
	server: { middlewareMode: true, hmr: { port: 30_000 + (process.pid % 10_000) } },
	logLevel: 'error'
});

try {
	const mod = await server.ssrLoadModule('/src/lib/server/repairBacklog.ts');
	assert.ok(typeof mod.issueRepairEstimateToken === 'function', 'issueRepairEstimateToken not exported');
	assert.ok(typeof mod.consumeRepairEstimateToken === 'function', 'consumeRepairEstimateToken not exported');
	assert.ok(typeof mod.repairEstimateFingerprint === 'function', 'repairEstimateFingerprint not exported');

	const baseInput = { run_id: 'r1', source_route: '/x', expected_run_status: 'succeeded', reviewer: 'tester' };
	const baseFilters = { type: 'Activation' };
	const fingerprint = mod.repairEstimateFingerprint(baseInput, baseFilters);

	// 1. Issued token is well-formed
	const { token, expiresAt } = mod.issueRepairEstimateToken(fingerprint);
	const parts = token.split('.');
	assert.equal(parts.length, 4);
	assert.equal(parts[0], 'v1');
	assert.ok(expiresAt > Date.now());

	// 2. Independent HMAC over the same secret reproduces the signature —
	//    this is the cross-process verification property: another node
	//    process with the shared secret derives the same signature.
	const payload = `${parts[0]}.${parts[1]}.${parts[2]}`;
	const externalSig = createHmac('sha256', Buffer.from(SHARED_SECRET, 'utf-8'))
		.update(payload).digest('base64url');
	assert.equal(externalSig, parts[3], 'external HMAC must match the token signature');

	// 3. Round-trip with the matching fingerprint succeeds
	mod.consumeRepairEstimateToken(token, fingerprint);

	// 4. Wrong fingerprint is rejected
	const differentFingerprint = mod.repairEstimateFingerprint(
		{ ...baseInput, run_id: 'r2' }, baseFilters
	);
	assert.throws(
		() => mod.consumeRepairEstimateToken(token, differentFingerprint),
		(err) => err.code === 'repair_estimate_stale'
	);

	// 5. Tampered signature is rejected (preserves length to test timing-safe path)
	const flipped = parts.slice(0, 3).concat([
		parts[3].slice(0, -2) + (parts[3].endsWith('AA') ? 'BB' : 'AA')
	]).join('.');
	assert.throws(
		() => mod.consumeRepairEstimateToken(flipped, fingerprint),
		(err) => err.code === 'repair_estimate_expired'
	);

	// 6. Expired token is rejected (synthesize an old expiry, re-sign)
	const oldExpiry = String(Date.now() - 1000);
	const oldPayload = `v1.${parts[1]}.${oldExpiry}`;
	const oldSig = createHmac('sha256', Buffer.from(SHARED_SECRET, 'utf-8'))
		.update(oldPayload).digest('base64url');
	const expiredToken = `${oldPayload}.${oldSig}`;
	assert.throws(
		() => mod.consumeRepairEstimateToken(expiredToken, fingerprint),
		(err) => err.code === 'repair_estimate_expired'
	);

	// 7. Missing token is rejected with the right code
	assert.throws(
		() => mod.consumeRepairEstimateToken(null, fingerprint),
		(err) => err.code === 'repair_estimate_required'
	);

	console.log('repair estimate token round-trip tests passed');
} finally {
	await server.close();
}
