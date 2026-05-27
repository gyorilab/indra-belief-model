import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const pairPage = readFileSync(new URL('../src/routes/pairs/[pair_id]/+page.svelte', import.meta.url), 'utf-8');

assert.match(pairPage, /function architectureLabel\(/);
assert.match(pairPage, /function runStatusClass\(/);
assert.match(pairPage, /function runPaneLabel\(/);
assert.doesNotMatch(pairPage, /function runTitle\(/);

assert.match(pairPage, /class="run-pane-head"/);
assert.match(pairPage, /<h2><span>\{archMark\(w\.monolithic\?\.architecture \?\? 'monolithic'\)\}<\/span> \{architectureLabel\(w\.monolithic\?\.architecture, 'monolithic'\)\}<\/h2>/);
assert.match(pairPage, /<h2><span>\{archMark\(w\.decomposed\?\.architecture \?\? 'decomposed'\)\}<\/span> \{architectureLabel\(w\.decomposed\?\.architecture, 'decomposed'\)\}<\/h2>/);
assert.match(pairPage, /class=\{`run-status run-status-\$\{runStatusClass\(w\.monolithic\.status\)\}`\}/);
assert.match(pairPage, /class=\{`run-status run-status-\$\{runStatusClass\(w\.decomposed\.status\)\}`\}/);

assert.match(pairPage, /class="run-link" title=\{w\.monolithic\.run_id\}><span>run<\/span><code>\{shortHash\(w\.monolithic\.run_id\)\}<\/code><\/a>/);
assert.match(pairPage, /class="run-link" title=\{w\.decomposed\.run_id\}><span>run<\/span><code>\{shortHash\(w\.decomposed\.run_id\)\}<\/code><\/a>/);
assert.doesNotMatch(pairPage, /<code>\{w\.monolithic\.run_id\}<\/code>/);
assert.doesNotMatch(pairPage, /<code>\{w\.decomposed\.run_id\}<\/code>/);

assert.match(pairPage, /<dl class="run-facts">/);
assert.match(pairPage, /<div><dt>duration<\/dt><dd>\{fmtSeconds\(w\.monolithic\.duration_s\)\}<\/dd><\/div>/);
assert.match(pairPage, /<div><dt>duration<\/dt><dd>\{fmtSeconds\(w\.decomposed\.duration_s\)\}<\/dd><\/div>/);
assert.doesNotMatch(pairPage, /<div><dt>status<\/dt><dd>\{w\.(monolithic|decomposed)\.status\}<\/dd><\/div>/);

assert.match(pairPage, /class="axis-denominator-link" href=\{ledgerHref\('clean_shared_aggregate_verdict_evidence'\)\}>base denominator<\/a>/);
assert.match(pairPage, /<dl class="massbar-key" aria-label="overlap partition counts">/);
assert.match(pairPage, /<div class="key-m"><dt>\[M\]<\/dt><dd>\{fmtCount\(w\.overlap\.monolithic_only_evidences\)\} only<\/dd><\/div>/);
assert.match(pairPage, /<div class="key-o"><dt>both<\/dt><dd>\{fmtCount\(w\.overlap\.overlap_evidences\)\} shared<\/dd><\/div>/);
assert.match(pairPage, /<div class="key-d"><dt>\[D\]<\/dt><dd>\{fmtCount\(w\.overlap\.decomposed_only_evidences\)\} only<\/dd><\/div>/);
assert.match(pairPage, /<div class="key-x"><dt>out<\/dt><dd>\{fmtCount\(excludedMass\(\)\)\} outside<\/dd><\/div>/);

assert.match(pairPage, /\.run-pane-head \{/);
assert.match(pairPage, /\.run-status-succeeded/);
assert.match(pairPage, /\.axis-denominator-link/);
assert.match(pairPage, /\.run-facts,/);

console.log('paired run axis source-shape tests passed');
