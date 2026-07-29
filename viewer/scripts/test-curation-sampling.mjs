/** Behavioral guard for lossless, without-replacement /curate sampling. */
import {
	curatorPairKeys,
	exactPairKey,
	parseCurationHistory,
	poolLineExcluded,
	poolPairOf,
	unseenPoolLines
} from '../src/lib/server/curation-history.ts';
import {
	claimDrawReservation,
	commitDrawReservation,
	releaseDrawClaim,
	reservedPairKeys,
	submissionFailureIsDefinitive,
	tryReserveDraw
} from '../src/lib/server/curation-draw-ledger.ts';
import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawn } from 'node:child_process';
import { DATASETS, datasetsForClient } from '../src/lib/server/datasets.ts';

let failures = 0;

function eq(got, want, label) {
	if (got !== want) {
		failures++;
		console.error(`FAIL ${label}: got ${JSON.stringify(got)}, want ${JSON.stringify(want)}`);
	}
}

function ok(got, label) {
	if (!got) {
		failures++;
		console.error(`FAIL ${label}`);
	}
}

const bigPositive = '29620476712050361';
const bigNegative = '-8960526783683006255';
const history = parseCurationHistory(
	`[{"curator":"Mock7ee@gmail.com","tag":"correct","pa_hash":${bigPositive},` +
		`"pa_json":{"source_hash":999999999999999999},"source_hash":${bigNegative},` +
		`"text":"escaped \\"source_hash\\": 123 stays text"},` +
		`{"curator":"other@example.org","tag":"grounding","pa_hash":"7","source_hash":"8"}]`
);

eq(history[0].paHash, bigPositive, 'pa_hash keeps every 64-bit digit');
eq(history[0].sourceHash, bigNegative, 'source_hash keeps every signed 64-bit digit');
eq(history[1].paHash, '7', 'already-quoted hashes remain valid');

const mine = curatorPairKeys(history, 'mock7ee@gmail.com');
eq(mine.size, 1, 'curator match is case-insensitive and scoped');
ok(mine.has(`${bigPositive}:${bigNegative}`), 'history key is exact');
eq(exactPairKey('1.5', '2'), null, 'non-integer hash rejected');

const lineA = `{"stmt_hash":${bigPositive},"source_hash":${bigNegative}}`;
const lineB = '{"stmt_hash":11,"source_hash":22}';
const lineBDuplicate = '{"stmt_hash":"11","source_hash":"22","extra":true}';
const lineC = '{"stmt_hash":33,"source_hash":44}';
const manifestLine = '{"matches_hash":55,"source_hash":66}';
const excludedLine =
	'{"matches_hash":77,"source_hash":88,"excluded_from_curation":true,"exclusion_reason":"fixture"}';
eq(poolPairOf(lineA)?.stmtHash, bigPositive, 'pool parser keeps exact statement hash');
eq(poolPairOf(lineA)?.sourceHash, bigNegative, 'pool parser keeps exact evidence hash');
eq(poolPairOf(manifestLine)?.stmtHash, '55', 'tracked manifest matches_hash is accepted');
eq(poolPairOf('{"stmt_hash":"11x","source_hash":22}'), null, 'malformed quoted hash rejected');
eq(poolPairOf('{"stmt_hash":11.5,"source_hash":22}'), null, 'non-integer numeric hash rejected');
ok(poolLineExcluded(excludedLine), 'explicit benchmark-overlap exclusion is recognized');

const unseen = unseenPoolLines([lineA, lineB, lineBDuplicate, 'invalid', lineC, excludedLine], mine);
eq(unseen.length, 2, 'seen, invalid, and duplicate pool pairs are removed');
eq(poolPairOf(unseen[0])?.stmtHash, '11', 'first unseen pair remains');
eq(poolPairOf(unseen[1])?.stmtHash, '33', 'second unseen pair remains');

const ledgerDir = mkdtempSync(join(tmpdir(), 'indra-curation-ledger-'));
process.env.CURATION_DRAW_LEDGER_DIR = ledgerDir;
const ledgerEmail = 'curator@example.org';
const ledgerDataset = 'representative';
const ledgerPair = '55:66';
const token = tryReserveDraw(ledgerEmail, ledgerDataset, ledgerPair);
ok(token, 'first exact-pair draw is atomically reserved');
eq(tryReserveDraw(ledgerEmail, ledgerDataset, ledgerPair), null, 'same pair cannot be drawn twice');
ok(reservedPairKeys(ledgerEmail, ledgerDataset).has(ledgerPair), 'draw survives a fresh ledger read');
ok(reservedPairKeys(ledgerEmail, 'rasmachine').has(ledgerPair), 'draw is blocked across dataset lanes');
eq(tryReserveDraw(ledgerEmail, 'rasmachine', ledgerPair), null, 'cross-lane replacement is rejected');
eq(claimDrawReservation(ledgerEmail, 'rasmachine', ledgerPair, token), false, 'token preserves originating dataset');
eq(claimDrawReservation(ledgerEmail, ledgerDataset, ledgerPair, 'wrong-token'), false, 'wrong token cannot submit');
eq(claimDrawReservation(ledgerEmail, ledgerDataset, ledgerPair, token), true, 'reservation can be claimed once');
eq(claimDrawReservation(ledgerEmail, ledgerDataset, ledgerPair, token), false, 'concurrent claim is rejected');
releaseDrawClaim(ledgerEmail, ledgerDataset, ledgerPair);
eq(claimDrawReservation(ledgerEmail, ledgerDataset, ledgerPair, token), true, 'failed submission can release its claim');
commitDrawReservation(ledgerEmail, ledgerDataset, ledgerPair, token);
eq(claimDrawReservation(ledgerEmail, ledgerDataset, ledgerPair, token), false, 'committed draw cannot be claimed again');

const legacyPair = '909:808';
const hex = (value) => createHash('sha256').update(value).digest('hex');
const legacyCuratorDir = join(ledgerDir, hex(ledgerEmail.toLowerCase()));
for (const [dataset, legacyToken] of [
	['representative', 'legacy-representative-token'],
	['rasmachine', 'legacy-rasmachine-token']
]) {
	const dir = join(legacyCuratorDir, dataset);
	mkdirSync(dir, { recursive: true });
	writeFileSync(
		join(dir, `${hex(legacyPair)}.json`),
		`${JSON.stringify({ schemaVersion: 1, dataset, pairKey: legacyPair, token: legacyToken, drawnAt: new Date(0).toISOString() })}\n`
	);
}
eq(
	claimDrawReservation(ledgerEmail, 'representative', legacyPair, 'legacy-representative-token'),
	true,
	'first legacy dataset reservation gets curator-global claim'
);
eq(
	claimDrawReservation(ledgerEmail, 'rasmachine', legacyPair, 'legacy-rasmachine-token'),
	false,
	'legacy reservations for same pair cannot claim concurrently'
);
releaseDrawClaim(ledgerEmail, 'representative', legacyPair);
eq(submissionFailureIsDefinitive(400), true, 'ordinary 4xx rejection can release claim');
eq(submissionFailureIsDefinitive(429), true, 'rate-limit rejection can release claim');
eq(submissionFailureIsDefinitive(undefined), false, 'network failure keeps claim fail-closed');
eq(submissionFailureIsDefinitive(408), false, 'request timeout is ambiguous');
eq(submissionFailureIsDefinitive(500), false, 'server error is ambiguous');
rmSync(ledgerDir, { recursive: true, force: true });

const oldNodeEnv = process.env.NODE_ENV;
const oldLedgerDir = process.env.CURATION_DRAW_LEDGER_DIR;
const oldLedgerShared = process.env.CURATION_DRAW_LEDGER_SHARED;
process.env.NODE_ENV = 'production';
delete process.env.CURATION_DRAW_LEDGER_DIR;
delete process.env.CURATION_DRAW_LEDGER_SHARED;
let productionFailedClosed = false;
try {
	reservedPairKeys(ledgerEmail, ledgerDataset);
} catch (error) {
	productionFailedClosed = String(error).includes('persistent shared storage');
}
ok(productionFailedClosed, 'production refuses an unshared or ephemeral draw ledger');
if (oldNodeEnv === undefined) delete process.env.NODE_ENV;
else process.env.NODE_ENV = oldNodeEnv;
if (oldLedgerDir === undefined) delete process.env.CURATION_DRAW_LEDGER_DIR;
else process.env.CURATION_DRAW_LEDGER_DIR = oldLedgerDir;
if (oldLedgerShared === undefined) delete process.env.CURATION_DRAW_LEDGER_SHARED;
else process.env.CURATION_DRAW_LEDGER_SHARED = oldLedgerShared;

const raceLedgerDir = mkdtempSync(join(tmpdir(), 'indra-curation-ledger-race-'));
const ledgerModuleUrl = new URL('../src/lib/server/curation-draw-ledger.ts', import.meta.url).href;
const workerCode = `
  import { tryReserveDraw } from ${JSON.stringify(ledgerModuleUrl)};
  const token = tryReserveDraw('race@example.org', 'representative', '101:202');
  process.stdout.write(token ? 'won' : 'lost');
`;
function raceWorker() {
	return new Promise((done) => {
		const child = spawn(
			process.execPath,
			['--experimental-strip-types', '--input-type=module', '--eval', workerCode],
			{
				env: {
					...process.env,
					NODE_ENV: 'test',
					CURATION_DRAW_LEDGER_DIR: raceLedgerDir
				},
				stdio: ['ignore', 'pipe', 'pipe']
			}
		);
		let stdout = '';
		let stderr = '';
		child.stdout.on('data', (chunk) => (stdout += String(chunk)));
		child.stderr.on('data', (chunk) => (stderr += String(chunk)));
		child.on('close', (code) => done({ code, stdout, stderr }));
	});
}
const raceResults = await Promise.all([raceWorker(), raceWorker()]);
eq(raceResults.filter((result) => result.code === 0 && result.stdout === 'won').length, 1, 'one process wins reservation race');
eq(raceResults.filter((result) => result.code === 0 && result.stdout === 'lost').length, 1, 'other process loses reservation race');
ok(raceResults.every((result) => result.stderr === ''), 'reservation race workers have no errors');
rmSync(raceLedgerDir, { recursive: true, force: true });

const representative = DATASETS.find((dataset) => dataset.id === 'representative');
eq(
	representative?.file,
	'benchmark/cogex_representative_pool_manifest.jsonl',
	'clean-checkout runtime uses tracked representative manifest'
);
const repoRoot = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
const runtimeLines = readFileSync(
	resolve(repoRoot, 'data/benchmark/cogex_representative_pool_manifest.jsonl'),
	'utf8'
)
	.split('\n')
	.filter((line) => line.trim());
eq(runtimeLines.length, 5_000, 'runtime representative frame has 5,000 reservoir rows');
eq(runtimeLines.filter(poolLineExcluded).length, 2, 'runtime frame blocks both prior-benchmark overlaps');
eq(unseenPoolLines(runtimeLines, new Set()).length, 4_998, 'effective clean curation frame is 4,998 pairs');
eq(
	datasetsForClient().find((dataset) => dataset.id === 'representative')?.available,
	true,
	'tracked representative frame is available to the server'
);
for (const dataset of DATASETS) {
	const client = datasetsForClient().find((candidate) => candidate.id === dataset.id);
	eq(client?.available, existsSync(resolve(repoRoot, 'data', dataset.file)), `${dataset.id} availability is truthful`);
	if (!client?.available && dataset.provisioning) {
		ok(client.unavailableReason?.includes('build_curate_pool.py'), `${dataset.id} gives its provisioning command`);
	}
}

if (failures > 0) {
	console.error(`\n${failures} curation-sampling assertion(s) failed`);
	process.exit(1);
}
console.log('curation-sampling: all assertions passed');
