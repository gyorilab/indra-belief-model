/**
 * WHERE AN INTERVAL SITS RELATIVE TO ZERO — the page's ONE classifier.
 *
 * A LEAF MODULE ON PURPOSE. It imports nothing from this package, so anything may
 * depend on it without creating an edge back.
 *
 * That is not tidiness. This classifier used to live in `paper-literal.ts`, which
 * value-imports `validateApDecomposition` from `paper-ap-decomposition.ts` — and
 * `paper-ap-decomposition.ts` value-imported `standingOfBounds` back, so the two
 * modules formed a RUNTIME import cycle. It was type-only and harmless until the
 * 2026-07-29 pass that retired the sign-blind boolean made it a value import.
 *
 * Nothing caught it: svelte-check passes, bundlers tolerate cycles, and the
 * comments in paper-ap-decomposition.ts still declared the dependency one-way.
 * What a cycle costs is initialisation order — whichever module evaluates second
 * sees a partially initialised first. `standingOfBounds` is a hoisted function so
 * it was safe on the day; one module-level `const` that called it would have been
 * a TDZ error at runtime with no compile-time warning.
 *
 * Keep this module leaf. `scripts/test-paper-render-invariants.mjs` fails on any
 * value-import cycle among `paper-*.ts`.
 */

/** Where one interval sits relative to zero. There is no fourth class. */
export type Standing = 'ahead' | 'behind' | 'not-significant';

/**
 * The one classifier. Read off the two ENDPOINTS, never off a shipped flag and
 * never off the sign of the point estimate: `low > 0` and `high < 0` are the two
 * facts and they cannot both hold. Callers whose point estimate may sit outside
 * its own interval must gate on that themselves — a direction and a range that
 * disagree is not a case this function can classify honestly.
 */
export function standingOfBounds(low: number, high: number): Standing {
	if (low > 0) return 'ahead';
	if (high < 0) return 'behind';
	return 'not-significant';
}
