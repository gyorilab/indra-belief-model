import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const source = readFileSync(new URL('../src/routes/+page.svelte', import.meta.url), 'utf-8');

function functionBody(name) {
	const start = source.indexOf(`function ${name}`);
	assert.notEqual(start, -1, `${name} exists`);
	const paramsOpen = source.indexOf('(', start);
	assert.notEqual(paramsOpen, -1, `${name} has parameters`);
	let paramsDepth = 0;
	let paramsClose = -1;
	for (let i = paramsOpen; i < source.length; i += 1) {
		const ch = source[i];
		if (ch === '(') paramsDepth += 1;
		if (ch === ')') paramsDepth -= 1;
		if (paramsDepth === 0) {
			paramsClose = i;
			break;
		}
	}
	assert.notEqual(paramsClose, -1, `${name} parameter list closes`);
	const open = source.indexOf('{', paramsClose);
	assert.notEqual(open, -1, `${name} has a body`);
	let depth = 0;
	for (let i = open; i < source.length; i += 1) {
		const ch = source[i];
		if (ch === '{') depth += 1;
		if (ch === '}') depth -= 1;
		if (depth === 0) return source.slice(open + 1, i);
	}
	throw new Error(`${name} body did not close`);
}

const cancelAllBody = functionBody('cancelAllLocalStreams');
assert.match(cancelAllBody, /ingestControllers\.keys\(\)[\s\S]*cancelIngest\(path, origin\)/);
assert.match(cancelAllBody, /truthSetControllers\.keys\(\)[\s\S]*cancelTruthSet\(path, origin\)/);
assert.match(cancelAllBody, /scoreControllers\.keys\(\)[\s\S]*cancelScore\(path, origin\)/);
assert.match(cancelAllBody, /pairedControllers\.keys\(\)[\s\S]*cancelPairedScore\(path, origin\)/);
assert.match(cancelAllBody, /request-disconnect/);

const cancelPairedBody = functionBody('cancelPairedScore');
assert.match(cancelPairedBody, /cur\.phase === 'scoring'/);
assert.match(cancelPairedBody, /durable cancel POST/);
assert.match(cancelPairedBody, /keepalive:\s*origin === 'page_exit'/);
assert.match(cancelPairedBody, /pairedCancelFailureFromResponse/);
assert.match(cancelPairedBody, /ctrl\.abort\(\)/);

const cancelScoreBody = functionBody('cancelScore');
assert.match(cancelScoreBody, /ctrl\.abort\(\)/);
assert.match(cancelScoreBody, /localCancelPhrase\(origin\)/);

const cancelIngestBody = functionBody('cancelIngest');
assert.match(cancelIngestBody, /ctrl\.abort\(\)/);
assert.match(cancelIngestBody, /localCancelPhrase\(origin\)/);

const cancelTruthSetBody = functionBody('cancelTruthSet');
assert.match(cancelTruthSetBody, /truthSetControllers\.get\(path\)/);
assert.match(cancelTruthSetBody, /ctrl\.abort\(\)/);
assert.match(cancelTruthSetBody, /truth_set registration/);
assert.match(source, /fetch\('\/api\/truth-sets'[\s\S]*signal:\s*ctrl\.signal/);
assert.match(source, /disabled=\{globalWriterActive \|\| localWriterActive\(\)\}[\s\S]*register `tag` as truth_set/);
assert.match(source, /if \(kind === 'truth_set'\) return 'truth set'/);
assert.match(source, /registering truth_set/);
assert.match(source, /registration commits atomically/);
assert.match(source, /tag=correct is the validity positive class/);
assert.match(source, /type ActionPhase = 'idle' \| 'confirming' \| 'running' \| 'done' \| 'blocked' \| 'error'/);
assert.match(source, /isWriterActionBlockedCode/);
assert.match(source, /blocked: \{actionBlockedText\(st\)\.slice\(0, 120\)\}/);
assert.match(source, /--blocked:\s*#[0-9a-f]{6}/i);
assert.match(source, /\.ds-action-blocked[\s\S]*border:\s*1px dashed var\(--blocked\)[\s\S]*color:\s*var\(--blocked\)/);

const cancelDurablePairBody = functionBody('cancelDurablePair');
assert.match(cancelDurablePairBody, /pairedCancelFailureFromResponse/);
assert.match(cancelDurablePairBody, /durablePairCancelErrors/);
assert.match(source, /role="alert">cancel failed:/);
assert.match(source, /tombstone failed/);
assert.match(source, /a\.status === 'canceled'[\s\S]*cancel_tombstone_failed|a\.error && a\.error !== 'canceled_by_user'/);
assert.match(source, /phase: 'blocked'/);
assert.match(source, /function pairedRequestBlocked/);
assert.match(source, /isPairedRunBlockedCode\(code\)/);
assert.match(source, /paired run blocked:/);
assert.match(source, /paired_failed[\s\S]*pairedRequestBlocked\(code\)[\s\S]*phase: 'blocked'/);
assert.match(source, /ds-action-blocked/);
assert.doesNotMatch(source, /blocked because the other architecture failed/);
assert.match(readFileSync(new URL('../src/lib/pairedRunConflicts.ts', import.meta.url), 'utf-8'), /writerLockMalformed:\s*'writer_lock_malformed'/);
const writerActionConflicts = readFileSync(new URL('../src/lib/writerActionConflicts.ts', import.meta.url), 'utf-8');
assert.match(writerActionConflicts, /writerLockBusy:\s*'writer_lock_busy'/);
assert.match(writerActionConflicts, /pairedWorkflowActive:\s*PAIRED_RUN_CONFLICT\.pairedWorkflowActive/);
assert.match(writerActionConflicts, /isWriterActionBlockedCode/);
assert.match(readFileSync(new URL('../src/routes/runs/[run_id]/cohort/+page.svelte', import.meta.url), 'utf-8'), /writer_lock_malformed[\s\S]*phase: 'blocked'/);

const scoreCorpusBody = functionBody('scoreCorpus');
assert.match(scoreCorpusBody, /responseErrorPayload\(res\)/);
assert.match(scoreCorpusBody, /isWriterActionBlockedCode\(failure\.code\)[\s\S]*phase: 'blocked'/);
assert.match(scoreCorpusBody, /architecture,[\s\S]*code: failure\.code/);

const ingestCorpusBody = functionBody('ingestCorpus');
assert.match(ingestCorpusBody, /responseErrorPayload\(res\)/);
assert.match(ingestCorpusBody, /phase: isWriterActionBlockedCode\(failure\.code\) \? 'blocked' : 'error'/);

const registerTruthSetBody = functionBody('registerAsTruthSet');
assert.match(registerTruthSetBody, /responseErrorPayload\(res\)/);
assert.match(registerTruthSetBody, /phase: isWriterActionBlockedCode\(failure\.code\) \? 'blocked' : 'error'/);

assert.match(source, /const cancelForPageExit = \(event: PageTransitionEvent\)/);
assert.match(source, /if \(event\.persisted\) return/);
assert.match(source, /window\.addEventListener\('pagehide', cancelForPageExit\)/);
assert.match(source, /window\.removeEventListener\('pagehide', cancelForPageExit\)/);
assert.match(source, /onDestroy\(\(\) => \{[\s\S]*cancelAllLocalStreams\('page_exit'\)/);

console.log('dashboard cancel source-shape tests passed');
