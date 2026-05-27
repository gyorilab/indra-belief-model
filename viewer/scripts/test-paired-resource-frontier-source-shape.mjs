import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const db = readFileSync(new URL('../src/lib/db.ts', import.meta.url), 'utf-8');
const pairPage = readFileSync(new URL('../src/routes/pairs/[pair_id]/+page.svelte', import.meta.url), 'utf-8');
const pairServer = readFileSync(new URL('../src/routes/pairs/[pair_id]/+page.server.ts', import.meta.url), 'utf-8');

assert.match(db, /export interface PairedResourceFrontierArch/);
assert.match(db, /export interface PairedResourceFrontier/);
assert.match(db, /resource_frontier: PairedResourceFrontier \| null/);
assert.match(db, /sr\.finished_at::VARCHAR AS finished_at/);
assert.match(db, /duration_s: durationSeconds\(r\.started_at, r\.finished_at\)/);
assert.match(db, /cost_per_evidence_usd: safeRate\(runCost, run\.n_evidences\)/);
assert.match(db, /monolithic_latency_observed_n/);
assert.match(db, /monolithic_tokens_observed_n/);
assert.match(db, /clean_overlap_latency_observed_n: Number\(latencyObservedN \?\? 0\)/);
assert.match(db, /clean_overlap_tokens_observed_n: Number\(tokensObservedN \?\? 0\)/);
assert.match(db, /clean_overlap_tokens_per_observed_evidence: safeRate\(Number\(tokensTotal \?\? 0\), Number\(tokensObservedN \?\? 0\)\)/);
assert.match(db, /spend_scope: 'whole-run spend; denominator is each side aggregate evidence count, not clean overlap'/);
assert.match(db, /latency_scope: 'clean shared aggregate verdict evidence with per-row telemetry counts shown'/);

assert.match(pairServer, /resource_frontier: null/);

assert.match(pairPage, /import type \{[^}]*PairedResourceFrontierArch/s);
assert.match(pairPage, /id="resource-frontier"/);
assert.match(pairPage, /aria-label="cost latency frontier"/);
assert.match(pairPage, /whole-run spend beside clean-overlap counters/);
assert.match(pairPage, /Spend and wall clock use each side's full run/);
assert.match(pairPage, /frontier\.spend_scope/);
assert.match(pairPage, /frontier\.latency_scope/);
assert.match(pairPage, /frontier\.quality_scope/);
assert.match(pairPage, /rows report latency/);
assert.match(pairPage, /rows report tokens/);
assert.match(pairPage, /function metricRate/);
assert.match(pairPage, /function telemetryCoverageText/);
assert.match(pairPage, /function telemetryComplete/);
assert.match(pairPage, /function resourceTelemetryWinner/);
assert.match(pairPage, /if \(!telemetryComplete\(monolithicObserved, total\) \|\| !telemetryComplete\(decomposedObserved, total\)\) return 'none'/);
assert.match(pairPage, /aggregate scorer-step latency; missing telemetry is not counted as zero-latency evidence/);
assert.match(pairPage, /tokens\/reported evidence/);
assert.match(pairPage, /missing telemetry is not counted as zero-token evidence/);
assert.match(pairPage, /resourceTelemetryWinner\(\s*monolithicTokensPerObserved,\s*decomposedTokensPerObserved/s);
assert.doesNotMatch(pairPage, /<h3>tokens <span>n=\{fmtCount\(w\.comparable\.n_overlap\)\}/);
assert.match(pairPage, /href=\{ledgerHref\('resource_counter_metric'\)\}/);
assert.match(pairPage, /href=\{ledgerHref\('truth_anchored_overlap_evidence'\)\}/);
assert.match(pairPage, /lower resource use is not a quality signal; frontier rows show tradeoff posture, not a scalar winner/);
assert.match(pairPage, /function fmtUnitCost/);
assert.match(pairPage, /function frontierRows\(\): PairedResourceFrontierArch\[\]/);

console.log('paired resource frontier source-shape tests passed');
