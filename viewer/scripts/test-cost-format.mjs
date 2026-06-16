/**
 * Pure-function gate for the run-cost formatters in src/lib/format.ts.
 *
 * The cost-rendering surface (fmtCost in the run feed, fmtCost + fmtCostFull on
 * the run-detail page) is otherwise typecheck-only — `npm run check` proves the
 * types line up but never that a known-$0 run reads "$0.00" instead of the
 * "price unverified" message. This locks the status-vs-total semantics so the
 * known/partial/unavailable/legacy branches can't silently regress.
 *
 * Runs via Node's native type-stripping:  node --experimental-strip-types
 * Exits non-zero on the first failed assertion (so `npm run test:cost` and the
 * pytest wrapper both gate on it).
 */
import { fmtCost, fmtCostFull } from '../src/lib/format.ts';

let failures = 0;
function eq(got, want, label) {
	if (got !== want) {
		failures++;
		console.error(`FAIL ${label}: got ${JSON.stringify(got)}, want ${JSON.stringify(want)}`);
	}
}

// ── fmtCost (run-feed compact label) ─────────────────────────────────────────
// Legacy export (no cost block) → never $0 for an unknown price.
eq(fmtCost(null), 'cost n/a', 'fmtCost(null) legacy');
eq(fmtCost(undefined), 'cost n/a', 'fmtCost(undefined) legacy');
// Genuinely unverified price → withheld, never $0.
eq(fmtCost({ status: 'unavailable', total_usd: null }), 'cost n/a', 'fmtCost unavailable');
// KNOWN-$0 run: status verified, no priced spend (all rows were no-LLM) →
// total_usd null but the run cost $0. THIS is the bug the fix closes: it must
// read $0.00, NOT "cost n/a".
eq(fmtCost({ status: 'known', total_usd: null }), '$0.00', 'fmtCost known + null total → $0.00');
// Free-local run that did call the LLM: total_usd is a real 0.0.
eq(fmtCost({ status: 'known', total_usd: 0 }), '$0.00', 'fmtCost known + 0 → $0.00');
// Partial run with no priced spend yet still verified-zero on counted rows.
eq(fmtCost({ status: 'partial', total_usd: null }), '$0.00', 'fmtCost partial + null total → $0.00');
// Sub-cent and ordinary priced totals.
eq(fmtCost({ status: 'known', total_usd: 0.004 }), '<$0.01', 'fmtCost sub-cent');
eq(fmtCost({ status: 'known', total_usd: 12.345 }), '$12.35', 'fmtCost rounds to cents');
eq(fmtCost({ status: 'partial', total_usd: 3.21 }), '$3.21', 'fmtCost partial priced');

// ── fmtCostFull (detail-page per-1k figure) ──────────────────────────────────
// null → em-dash (genuinely "no datum": no LLM-scored rows to average over).
eq(fmtCostFull(null), '—', 'fmtCostFull(null) → em-dash');
eq(fmtCostFull(undefined), '—', 'fmtCostFull(undefined) → em-dash');
eq(fmtCostFull(0), '$0.00', 'fmtCostFull(0)');
eq(fmtCostFull(0.0042), '$0.0042', 'fmtCostFull sub-cent → 4 decimals');
eq(fmtCostFull(10.5), '$10.50', 'fmtCostFull ≥ $0.01 → 2 decimals');

if (failures > 0) {
	console.error(`\n${failures} cost-format assertion(s) failed`);
	process.exit(1);
}
console.log('cost-format: all assertions passed');
