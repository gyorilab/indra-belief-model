/**
 * Pearson chi-square homogeneity for a k x 2 table, and its p-value.
 *
 * Lives in its own dependency-free module for two reasons. It is imported by a
 * SERVER loader (`$lib/server/belief-heuristic`) that cannot be loaded outside
 * SvelteKit, and the contract runner has to be able to exercise the arithmetic
 * directly — a statistic that gates a claim on the page but can only be reached
 * through a page is a statistic nobody re-checks.
 *
 * WHAT IT IS FOR. The belief-heuristic panel's lead claim is that the paper's
 * noisy-OR assigns ONE belief to every single-evidence statement while the
 * sources behind those statements deliver very different correct rates. Without a
 * test, "very different" is an eyeball claim about five bars. With one, it is
 * stated at the strength the counts actually support.
 *
 * No dependency: one p-value does not justify pulling in a stats library, and the
 * value is pinned against `scipy.stats.chi2.sf` in the contract runner.
 */

/** Lanczos log-gamma; accurate far past the precision this p-value is read at. */
function logGamma(z: number): number {
	const g = [
		676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059,
		12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7
	];
	if (z < 0.5) return Math.log(Math.PI / Math.sin(Math.PI * z)) - logGamma(1 - z);
	const zz = z - 1;
	let x = 0.99999999999980993;
	for (let i = 0; i < g.length; i += 1) x += g[i] / (zz + i + 1);
	const t = zz + g.length - 0.5;
	return 0.5 * Math.log(2 * Math.PI) + (zz + 0.5) * Math.log(t) - t + Math.log(x);
}

/**
 * Regularized upper incomplete gamma Q(s, x) by the Lentz continued fraction —
 * the chi-square survival function's only hard part. The fraction converges for
 * x > s + 1, which every chi-square this module produces satisfies comfortably.
 */
function gammaQ(s: number, x: number): number {
	const TINY = 1e-300;
	let b = x + 1 - s;
	let c = 1 / TINY;
	let d = 1 / b;
	let f = d;
	for (let i = 1; i < 300; i += 1) {
		const an = -i * (i - s);
		b += 2;
		d = an * d + b;
		if (Math.abs(d) < TINY) d = TINY;
		c = b + an / c;
		if (Math.abs(c) < TINY) c = TINY;
		d = 1 / d;
		const delta = d * c;
		f *= delta;
		if (Math.abs(delta - 1) < 1e-15) break;
	}
	return Math.exp(-x + s * Math.log(x) - logGamma(s)) * f;
}

export interface HomogeneityResult {
	chi2: number;
	df: number;
	p: number;
}

/**
 * Pearson chi-square that every row shares one success rate.
 *
 * Returns null rather than a number for any table the test does not apply to —
 * fewer than two rows, or no variation to explain at all. A chi-square of 0 on a
 * degenerate table would print as "no evidence of a difference", which is a
 * finding; "not applicable" is the truth.
 */
export function homogeneityChiSquare(
	rows: readonly { count: number; correct: number }[]
): HomogeneityResult | null {
	if (rows.length < 2) return null;
	const n = rows.reduce((total, row) => total + row.count, 0);
	const correct = rows.reduce((total, row) => total + row.correct, 0);
	if (n === 0 || correct === 0 || correct === n) return null;
	const rate = correct / n;
	let chi2 = 0;
	for (const row of rows) {
		const expectedCorrect = row.count * rate;
		const expectedWrong = row.count * (1 - rate);
		if (expectedCorrect <= 0 || expectedWrong <= 0) return null;
		chi2 += (row.correct - expectedCorrect) ** 2 / expectedCorrect;
		chi2 += (row.count - row.correct - expectedWrong) ** 2 / expectedWrong;
	}
	const df = rows.length - 1;
	return { chi2, df, p: gammaQ(df / 2, chi2 / 2) };
}
