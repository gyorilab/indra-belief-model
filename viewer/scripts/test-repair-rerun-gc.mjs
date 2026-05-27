import { createServer } from 'vite';
import { mkdtempSync, mkdirSync, writeFileSync, existsSync, utimesSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { DuckDBInstance } from '@duckdb/node-api';
import assert from 'node:assert/strict';

// A2 of deferred hypergraph: repair-rerun JSON export GC + dedup.
// Verifies that gcRepairRerunExports retains exports under the retention
// window, retains referenced exports regardless of age, and deletes the
// rest.

const tmp = mkdtempSync(join(tmpdir(), 'idb-rrgc-'));
const dataDir = join(tmp, 'data');
const repairRerunsDir = join(dataDir, 'repair_reruns');
mkdirSync(repairRerunsDir, { recursive: true });
const dbFile = join(dataDir, 'corpus.duckdb');

{
	const seed = await DuckDBInstance.create(dbFile);
	const con = await seed.connect();
	await con.run(
		'CREATE TABLE scorer_step_correction (correction_id BIGINT, status VARCHAR, value_json JSON)'
	);
	await con.run(`INSERT INTO scorer_step_correction VALUES (
		1, 'open',
		json_object('source_dump_id', 'repair_referenced_old_export')
	)`);
	con.disconnectSync?.();
	seed.closeSync();
}

const now = Date.now();
const oldSec = (now - 30 * 24 * 60 * 60 * 1000) / 1000;
const recentSec = (now - 1 * 24 * 60 * 60 * 1000) / 1000;
for (const [id, mtimeSec] of [
	['repair_recent_export', recentSec],
	['repair_referenced_old_export', oldSec],
	['repair_unreferenced_old_export', oldSec]
]) {
	writeFileSync(join(repairRerunsDir, `${id}.json`), '[]\n');
	writeFileSync(join(repairRerunsDir, `${id}.meta.json`), JSON.stringify({ source_dump_id: id }));
	utimesSync(join(repairRerunsDir, `${id}.json`), mtimeSec, mtimeSec);
	utimesSync(join(repairRerunsDir, `${id}.meta.json`), mtimeSec, mtimeSec);
}

process.env.VIEWER_DUCKDB_PATH = dbFile;
const server = await createServer({
	configFile: './vite.config.ts',
	server: { middlewareMode: true, hmr: { port: 30_000 + (process.pid % 10_000) } },
	logLevel: 'error'
});

try {
	const mod = await server.ssrLoadModule('/src/lib/server/repairRerun.ts');
	const db = await server.ssrLoadModule('/src/lib/db.ts');
	const con = await db.connect();
	try {
		const result = await mod.gcRepairRerunExports(con, { retainDays: 14 });
		assert.deepEqual(result.deleted, ['repair_unreferenced_old_export']);
		assert.equal(result.kept_recent, 1);
		assert.equal(result.kept_referenced, 1);
		assert.ok(existsSync(join(repairRerunsDir, 'repair_recent_export.json')));
		assert.ok(existsSync(join(repairRerunsDir, 'repair_referenced_old_export.json')));
		assert.ok(!existsSync(join(repairRerunsDir, 'repair_unreferenced_old_export.json')));
		assert.ok(!existsSync(join(repairRerunsDir, 'repair_unreferenced_old_export.meta.json')));
	} finally {
		con.disconnectSync?.();
		await db.closeInstance();
	}

	const con2 = await db.connect();
	try {
		const result2 = await mod.gcRepairRerunExports(con2, { retainDays: 14 });
		assert.deepEqual(result2.deleted, []);
		assert.equal(result2.kept_recent, 1);
		assert.equal(result2.kept_referenced, 1);
	} finally {
		con2.disconnectSync?.();
		await db.closeInstance();
	}

	console.log('repair rerun GC tests passed');
} finally {
	await server.close();
	try {
		rmSync(tmp, { recursive: true, force: true });
	} catch {
		// best-effort
	}
}
