/**
 * Pure port of the canonical calibration MLE — no imports, no `$lib`, so it can
 * be exercised directly by `node --experimental-strip-types` against a numeric
 * golden fixture (see `viewer/scripts/test-paper-literal-contract.mjs`). The
 * server loader (`$lib/server/paper-literal`) imports `calibrationInterceptSlope`
 * from here; the math is byte-identical to the former inline `calibration()`.
 */

/**
 * Clip epsilon for the calibration logit — the canonical `log_loss_epsilon` from
 * data/results/indra_belief_comparison_metrics.json (provenance), so the TS port
 * clips scores to [1e-6, 1-1e-6] identically to the Python.
 */
const CALIBRATION_EPSILON = 1e-6;

function round6(value: number): number {
	return Math.round(value * 1e6) / 1e6;
}

/** Numerically stable logistic, mirroring metrics.py `_expit`. */
function expit(x: number): number {
	return x >= 0 ? 1 / (1 + Math.exp(-x)) : Math.exp(x) / (1 + Math.exp(x));
}

/** np.logaddexp(0, eta) = log(1 + exp(eta)), stable in both tails. */
function logaddexp0(eta: number): number {
	return eta > 0 ? eta + Math.log1p(Math.exp(-eta)) : Math.log1p(Math.exp(eta));
}

/**
 * Faithful TypeScript port of `comparison/metrics.py::_calibration_intercept_slope`
 * (cited by function name — line numbers drift): the unpenalized logistic recalibration
 * `label ~ 1 + logit(clip(score, ε, 1-ε))` fit by Newton-Raphson MLE with
 * step-halving. All weights are 1 (the point-estimate case). Returns the
 * `{intercept, slope}` MLE, or `null` on the same guards the Python bails on:
 *  - single class (no positives or no negatives),
 *  - non-finite or condition-number > 1e14 information matrix,
 *  - no step accepted by the halving line-search, |beta| > 1e6, or 100 iters
 *    without convergence.
 *
 * PARITY SCOPE (differential-tested): on WELL-CONDITIONED inputs — the regime
 * all real /paper arm calibration data occupies (paper RF + LLM arms, slopes
 * ~0.2–3, n≈1689) — this returns the same null/finite decision as the Python
 * and the same intercept/slope to ~1e-6. It is NOT identical in the pathological
 * near-constant-score / near-separable regime (two scores within ~1e-6 in logit
 * space, |beta| → ~1e5): there the null-vs-finite decision and the blown-up
 * magnitudes can disagree with the Python in either direction, because this port
 * estimates the 2×2 condition number in closed form (trace/det, subject to
 * cancellation) + solves by Cramér's rule, whereas the Python uses SVD-based
 * np.linalg.cond + LU np.linalg.solve. Such inputs do not arise in real arm data
 * (the rendered reliability strip is unaffected); do not rely on this port for
 * adversarial/degenerate calibration inputs.
 * Ideal calibration is intercept 0, slope 1; the LLM arms sit near slope ~0.2.
 */
export function calibrationInterceptSlope(pairs: { score: number; label: number }[]): {
	calibrationSlope: number | null;
	calibrationIntercept: number | null;
} {
	const nan = { calibrationSlope: null, calibrationIntercept: null };
	const n = pairs.length;
	const logits = new Array<number>(n);
	const y = new Array<number>(n);
	let positives = 0;
	let negatives = 0;
	for (let i = 0; i < n; i += 1) {
		const label = pairs[i].label > 0 ? 1 : 0;
		y[i] = label;
		if (label > 0) positives += 1;
		else negatives += 1;
		let s = pairs[i].score;
		if (s < CALIBRATION_EPSILON) s = CALIBRATION_EPSILON;
		else if (s > 1 - CALIBRATION_EPSILON) s = 1 - CALIBRATION_EPSILON;
		logits[i] = Math.log(s / (1 - s));
	}
	// Single-class guard: Python returns NaN when either class carries no weight.
	if (positives <= 0 || negatives <= 0) return nan;

	const logLikelihood = (b0: number, b1: number): number => {
		let acc = 0;
		for (let i = 0; i < n; i += 1) {
			const eta = b0 + b1 * logits[i];
			acc += y[i] * eta - logaddexp0(eta);
		}
		return acc;
	};

	let beta0 = 0;
	let beta1 = 1;
	let likelihood = logLikelihood(beta0, beta1);
	for (let iter = 0; iter < 100; iter += 1) {
		let i00 = 0;
		let i01 = 0;
		let i11 = 0;
		let g0 = 0;
		let g1 = 0;
		for (let i = 0; i < n; i += 1) {
			const lg = logits[i];
			const p = expit(beta0 + beta1 * lg);
			const w = p * (1 - p);
			i00 += w;
			i01 += w * lg;
			i11 += w * lg * lg;
			const resid = y[i] - p;
			g0 += resid;
			g1 += resid * lg;
		}
		if (!Number.isFinite(i00) || !Number.isFinite(i01) || !Number.isFinite(i11)) return nan;
		// 2×2 symmetric-PSD condition number = λmax/λmin (np.linalg.cond, 2-norm).
		const trace = i00 + i11;
		const det = i00 * i11 - i01 * i01;
		const disc = Math.sqrt(Math.max(0, trace * trace - 4 * det));
		const lambdaMax = (trace + disc) / 2;
		const lambdaMin = (trace - disc) / 2;
		const cond = lambdaMin > 0 ? lambdaMax / lambdaMin : Infinity;
		if (!Number.isFinite(cond) || cond > 1e14 || det === 0) return nan;
		// Closed-form 2×2 solve of information · step = gradient (no linalg dep).
		const step0 = (i11 * g0 - i01 * g1) / det;
		const step1 = (i00 * g1 - i01 * g0) / det;
		// Step-halving line search: accept the first scale keeping the log-likelihood
		// non-decreasing (within 1e-12), scale from 1.0 down to 2^-20.
		let scale = 1;
		let accepted = false;
		let candidate0 = beta0;
		let candidate1 = beta1;
		let candidateLikelihood = likelihood;
		while (scale >= 2 ** -20) {
			candidate0 = beta0 + scale * step0;
			candidate1 = beta1 + scale * step1;
			candidateLikelihood = logLikelihood(candidate0, candidate1);
			if (Number.isFinite(candidateLikelihood) && candidateLikelihood >= likelihood - 1e-12) {
				accepted = true;
				break;
			}
			scale *= 0.5;
		}
		if (!accepted) return nan;
		beta0 = candidate0;
		beta1 = candidate1;
		likelihood = candidateLikelihood;
		if (Math.max(Math.abs(scale * step0), Math.abs(scale * step1)) < 1e-10) {
			if (Math.max(Math.abs(beta0), Math.abs(beta1)) > 1e6) return nan;
			return { calibrationSlope: round6(beta1), calibrationIntercept: round6(beta0) };
		}
	}
	return nan;
}
