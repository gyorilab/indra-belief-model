import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const source = readFileSync(new URL('../src/routes/+page.svelte', import.meta.url), 'utf-8');
const pageServer = readFileSync(new URL('../src/routes/+page.server.ts', import.meta.url), 'utf-8');
const pairPage = readFileSync(new URL('../src/routes/pairs/[pair_id]/+page.svelte', import.meta.url), 'utf-8');
const pairServer = readFileSync(new URL('../src/routes/pairs/[pair_id]/+page.server.ts', import.meta.url), 'utf-8');
const pairedState = readFileSync(new URL('../src/lib/server/pairedState.ts', import.meta.url), 'utf-8');

assert.match(source, /durablePairActive \|\| Object\.values\(pairedStates\)\.some/);
assert.match(source, /function durablePairStateKind\(p: DurablePairedWorkflow\): 'active' \| 'stale' \| 'terminal'/);
assert.match(source, /function durablePairStateLabel\(p: DurablePairedWorkflow\): string/);
assert.match(source, /function durablePairClockText\(p: DurablePairedWorkflow\): string/);
assert.match(source, /function durableArchEtaText\(a: DurablePairedArch\): string/);
assert.match(source, /function durableArchStallText\(a: DurablePairedArch\): string/);
assert.match(source, /elapsed != null && elapsed > 30000 \? ' · no progress >30s' : ''/);
assert.match(source, /aria-label="paired workflow state ledger"/);
assert.match(source, /pw-row-\$\{durablePairStateKind\(pw\)\}/);
assert.match(source, /class=\{`pw-state-kind pw-state-kind-\$\{durablePairStateKind\(pw\)\}`\}/);
assert.match(source, /\{durablePairClockText\(pw\)\}/);
assert.match(source, /\.pw-row-active/);
assert.match(source, /\.pw-row-stale/);
assert.match(source, /\.pw-state-rail/);
assert.match(source, /\.pw-state-kind-active/);

assert.doesNotMatch(source, /aria-label="paired workflow tombstones"/);

assert.match(pageServer, /reconcileStalePairedWorkflowStates\(\)/);
assert.match(pairServer, /import \{ reconcileStalePairedWorkflowState \} from '\$lib\/server\/pairedState'/);
assert.match(pairServer, /const workflow = reconcileStalePairedWorkflowState\(params\.pair_id\)/);
assert.match(pairedState, /export function pairedWorkflowStaleReason/);
assert.match(pairedState, /export function reconcileStalePairedWorkflowState/);
assert.match(pairedState, /status: wasRunning \? 'failed' : 'blocked'/);
assert.match(pairedState, /marked failed without mutating any scorer output/);
assert.match(pairedState, /queued architecture never started/);
assert.match(pairedState, /return reconcileStalePairedWorkflowStates\(\)\.filter/);

assert.match(pairPage, /const WORKFLOW_STALL_MS = 30_000/);
assert.match(pairPage, /let tickNow = \$state\(Date\.now\(\)\)/);
assert.match(pairPage, /function workflowClockText\(\): string/);
assert.match(pairPage, /function workflowArchEtaText\(a: WorkflowArchState\): string/);
assert.match(pairPage, /function workflowArchStallText\(a: WorkflowArchState\): string/);
assert.match(pairPage, /elapsed != null && elapsed > WORKFLOW_STALL_MS \? 'no progress >30s' : ''/);
assert.match(pairPage, /class="workflow-clock">\{workflowClockText\(\)\}<\/p>/);
assert.match(pairPage, /class="workflow-progress-rail"/);
assert.match(pairPage, /role="progressbar"/);
assert.match(pairPage, /aria-label=\{`\$\{arch\} workflow progress \$\{workflowArchProgressText\(archState\)\}`\}/);
assert.match(pairPage, /<dl class="workflow-arch-facts">/);
assert.match(pairPage, /<div><dt>progress<\/dt><dd>\{workflowArchProgressText\(archState\)\}<\/dd><\/div>/);
assert.match(pairPage, /<div><dt>spend<\/dt><dd>\{workflowArchSpendText\(archState\)\}<\/dd><\/div>/);
assert.match(pairPage, /<div><dt>eta<\/dt><dd>\{workflowArchEtaText\(archState\)\}<\/dd><\/div>/);
assert.match(pairPage, /<div><dt>updated<\/dt><dd>\{workflowArchUpdatedText\(archState\)\}<\/dd><\/div>/);
assert.match(pairPage, /\.workflow-arch-stalled \.workflow-progress-rail/);

console.log('paired workflow state source-shape tests passed');
