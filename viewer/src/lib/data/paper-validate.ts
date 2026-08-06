/**
 * THE FAIL-CLOSED VALIDATOR KIT — one definition, not thirteen.
 *
 * A LEAF MODULE ON PURPOSE: it imports nothing from this package, so any loader
 * may depend on it without creating an edge back (see paper-interval.ts for what
 * a cycle here costs).
 *
 * Before this file, seven primitives were defined independently in up to thirteen
 * modules — 69 definitions in all. They had DRIFTED, and the drift was invisible
 * because each copy looked obviously correct on its own:
 *
 *   · `text()` rejected whitespace-only strings in EXACTLY ONE of the nine modules
 *     that defined it — paper-error-f1.ts. The other eight accepted them:
 *     ap-decomposition, belief-ladder, deployed-baseline, framing-correction,
 *     per-evidence, review-queue, robustness, table6-extended. (An earlier draft
 *     of this note said "two"; it was checked against `git show HEAD:` and was
 *     wrong. The minority rule is the one that survived.) An artifact shipping
 *     "   " in a prose field gated one figure and rendered a blank line in eight.
 *     The trimming form is canonical here: a field structurally present but
 *     visually empty is the "placeholder renders as a measurement" defect this
 *     page has already shipped once. Verified unreachable on today's data — all
 *     shipped artifacts scanned, zero whitespace-only strings — so the stricter
 *     rule gates nothing that renders.
 *   · `positiveInteger()` had two forms: a direct check, and one composed over
 *     `nonNegativeInteger`. They ACCEPT AND REJECT IDENTICALLY but do not fail
 *     identically — the composed form emitted "expected a non-negative integer"
 *     for every rejecting input except exactly 0, where the direct form always
 *     said "expected a positive integer". That is diagnostic drift across 73 call
 *     sites in four modules, and by the rule stated on `unit()` below a gate's
 *     message is part of its contract. No runner covers those 73, which is why
 *     the change is invisible rather than caught; the direct form is canonical.
 *   · `fail()` and `record()` were byte-identical everywhere.
 *
 * WHAT MUST NOT CHANGE. Every call site passes its own `context` string, and those
 * strings are the diagnostics a reader sees on a gated figure
 * ("arms[3].delta_error_f1: is not this arm's drawn error-F1 minus the
 * reference's"). Several are asserted by name in the contract runners. This kit
 * takes `context` and message verbatim and does nothing clever with them — do not
 * "improve" it into a template that generalises the wording away.
 */

/** Throw with the call site's own context. The one place a gate says no. */
export function fail(context: string, message: string): never {
	throw new Error(`${context}: ${message}`);
}

/** A JSON object, or a gate. */
export function record(value: unknown, context: string): Record<string, unknown> {
	if (typeof value !== 'object' || value === null || Array.isArray(value)) {
		fail(context, 'expected an object');
	}
	return value as Record<string, unknown>;
}

/** A finite number, or a gate. */
export function number(value: unknown, context: string): number {
	if (typeof value !== 'number' || !Number.isFinite(value)) {
		fail(context, 'expected a finite number');
	}
	return value;
}

/** A finite number in [0, 1] — a probability, a share, a metric on the unit scale. */
export function unit(value: unknown, context: string): number {
	const parsed = number(value, context);
	// WORDING IS THE CONTRACT. Both pre-extraction variants said "a number";
	// the extraction briefly said "a value" and test-statement-error-f1-contract
	// caught it, because a gate's message is asserted by name. Do not reword.
	if (parsed < 0 || parsed > 1) fail(context, 'expected a number in [0, 1]');
	return parsed;
}

/**
 * A non-empty string, WHITESPACE-TRIMMED before the emptiness test. The stricter
 * of the two forms that were in the tree; see the docblock above for why.
 */
export function text(value: unknown, context: string): string {
	if (typeof value !== 'string' || value.trim().length === 0) {
		fail(context, 'expected a non-empty string');
	}
	return value;
}

/** An integer >= 0, or a gate. */
export function nonNegativeInteger(value: unknown, context: string): number {
	if (typeof value !== 'number' || !Number.isInteger(value) || value < 0) {
		fail(context, 'expected a non-negative integer');
	}
	return value;
}

/** An integer >= 1, or a gate. */
export function positiveInteger(value: unknown, context: string): number {
	if (typeof value !== 'number' || !Number.isInteger(value) || value < 1) {
		fail(context, 'expected a positive integer');
	}
	return value;
}

/**
 * A LABEL THAT MUST FIT ITS GUTTER, or a gate.
 *
 * THE MECHANISM IS SHARED; THE VALUES ARE NOT. Seven modules defined this and six
 * were behaviourally identical (`value.length > chars`, differing only in whether
 * the parameter was named `value` or `text`). The seventh, paper-per-evidence.ts,
 * took a `fits` predicate instead — a real generalisation, because a
 * proportional-width label is not measured in characters. That form subsumes the
 * others: the default predicate is the character comparison.
 *
 * The BUDGETS and the geometry they come from stay in each figure's own module.
 * A waterfall and a ranked table share no geometry, and centralising the numbers
 * would invent coupling. What is shared is only the rule: measure, and GATE when
 * it does not fit.
 *
 * Why gating matters here rather than truncating: SVG text does not wrap and does
 * not warn. A right-anchored string that overruns its gutter loses its LEADING
 * glyphs, and the <desc> beside it still emits the full string — so screen readers
 * and automated a11y checks both report success while a sighted reader sees a
 * clipped word. Five of those shipped before the budgets were enforced.
 */
export function budget(
	value: string,
	chars: number,
	context: string,
	fits: (candidate: string) => boolean = (candidate) => candidate.length <= chars
): string {
	if (!fits(value)) {
		fail(context, `"${value}" is ${value.length} chars; the gutter budget is ${chars}`);
	}
	return value;
}

/**
 * A boolean, or a gate. Nine modules defined this identically.
 *
 * It is here for the same reason the rest of the kit is: a primitive that exists
 * nine times will eventually exist nine slightly different ways, and nobody will
 * notice until the difference decides whether a figure renders.
 */
export function boolean(value: unknown, context: string): boolean {
	if (typeof value !== 'boolean') fail(context, 'expected a boolean');
	return value;
}
