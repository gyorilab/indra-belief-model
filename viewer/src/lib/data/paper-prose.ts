/**
 * THE {shipped, plain} TWIN KIT — pairing, anchoring, gating. One mechanism.
 *
 * A TRUE LEAF: it imports NOTHING. An earlier version pulled in `fail`, `record`
 * and `text` from the validator kit and used none of them — the only two `text(`
 * occurrences in this file are inside comments — so its own "imports only the
 * validator kit" line described an edge that existed for no reason. `tsconfig`
 * sets no `noUnusedLocals`, so nothing would ever have said so.
 *
 * WHY THIS IS ITS OWN FILE. These helpers lived in `paper-literal.ts`, which is
 * also the arm registry and was, until today, the interval classifier as well —
 * three unrelated interfaces behind one name that 13 of 16 modules imported. A
 * loader needing only `pairShippedProse` had to name the module that owns the
 * frozen arm labels. Splitting by interface is the whole point: nothing here
 * knows what an arm is.
 *
 * WHAT THE MECHANISM IS FOR, and it is not tidiness. Artifact prose is sha-pinned
 * and ships in the file's own idiom; the page shows a plain restatement instead
 * and keeps the original for verification. Restatements are POSITIONAL against
 * arrays like `caveats[]`, so an artifact that reorders one caveat would silently
 * print restatement 4 under caveat 5 — a wrong sentence attributed to a pinned
 * file. Each twin therefore carries a verbatim fragment of the sentence it was
 * written for, and pairing GATES on finding it. Drift stops the figure rather
 * than mislabelling it.
 *
 * The restatements themselves stay in the modules that own the fields they
 * restate. Centralising 242 authored sentences would produce one string table
 * nobody can review, and would break the property that a twin sits beside the
 * field it explains. `scripts/test-paper-prose-coverage.mjs` checks all 242 for
 * fidelity — a plain half that drops a number, a negation or a named model its
 * shipped half carries is a failure.
 */
/**
 * A SHIPPED STRING AND THE PLAIN SENTENCE THAT REPLACES IT ON SCREEN.
 *
 * The dialect that survived three "clean" sweeps of /paper does not live in any
 * template or in any static string in this repo. It arrives at RUNTIME, off
 * sha-pinned artifact JSON — "tau = the smallest of the arm's own distinct
 * scores…", "paired fold-stratified bootstrap over the paper's own out-of-fold
 * fold assignment". A scan of the source cannot see a string that does not exist
 * until a file is read, which is exactly why every sweep came back green while
 * the words were on the screen.
 *
 * The artifact bytes are sha-pinned and may not be edited, so the translation
 * happens HERE, at the loader, once per field, beside the parse that reads it:
 *
 *   · `shipped` is parsed exactly as before — same `text()` call, same
 *     fail-closed gate, byte-identical to the artifact. It is the AUDIT TRAIL and
 *     belongs behind the page's single verification boundary.
 *   · `plain` is authored in the loader that owns the field, as a STATIC string,
 *     which is the whole point: a static string is a string the dialect guard can
 *     finally see, so the restatement a reader is handed is under test where the
 *     artifact's own wording never could be. Each module keeps its restatements in
 *     one block near the top (`ERROR_F1_PLAIN`, `REVIEW_QUEUE_PLAIN`,
 *     `ROBUSTNESS_PLAIN`, `DEPLOYED_BASELINE_PLAIN`, `TABLE6_PLAIN`) so the whole
 *     of what a reader is told can be read in one place.
 *
 * Neither half is optional and neither is derived from the other. Dropping
 * `shipped` would delete the audit trail; dropping `plain` puts a sentence
 * written for a referee in front of a curator.
 */
export interface ShippedProse {
	/**
	 * The string EXACTLY as it ships — artifact bytes verbatim wherever the field
	 * reads them off a sha-pinned file. Audit only; never the sentence a reader is
	 * handed outside the verification boundary.
	 */
	shipped: string;
	/** The plain restatement, authored here. THIS is what a reader sees. */
	plain: string;
}

/**
 * One plain restatement, PINNED to the shipped sentence it restates.
 *
 * A twin for a NAMED field is safe by construction — `threshold_rule`'s plain
 * half is written under the key that reads `threshold_rule`, and the two cannot
 * come apart. A twin for an ARRAY is not: `caveats[4]` is bound to its
 * restatement by position alone, so a reissued artifact that inserts, drops or
 * reorders one caveat would silently print restatement 4 under caveat 5 — a
 * wrong sentence attributed to a sha-pinned file, which is the exact failure the
 * whole page is built to make impossible.
 *
 * So each positional twin carries a short verbatim fragment of the sentence it
 * was written for, and `pairShippedProse` gates on finding it. Drift stops the
 * figure instead of mislabelling it.
 */
export interface AnchoredProse {
	/**
	 * A distinctive fragment of the SHIPPED sentence, quoted verbatim. Artifact
	 * bytes: it is deliberately left in the artifact's own words — including its
	 * dialect — because its job is to match those bytes, not to be read.
	 */
	artifactAnchor: string;
	/** The plain restatement of the sentence that anchor identifies. */
	plain: string;
}

/**
 * Pair a shipped string array with its plain restatements, FAIL-CLOSED on drift.
 * Throws — every caller is already inside a validator's try/catch, so a drifted
 * artifact gates its figure exactly like any other shape failure.
 */
export function pairShippedProse(
	shipped: readonly string[],
	twins: readonly AnchoredProse[],
	context: string
): ShippedProse[] {
	if (shipped.length !== twins.length) {
		throw new Error(
			`${context}: expected ${twins.length} entries with plain restatements, got ${shipped.length}`
		);
	}
	return shipped.map((entry, index) => {
		const twin = twins[index];
		if (!entry.includes(twin.artifactAnchor)) {
			throw new Error(
				`${context}[${index}]: no longer the sentence its plain restatement was written for ` +
					`(expected to contain ${JSON.stringify(twin.artifactAnchor)})`
			);
		}
		return { shipped: entry, plain: twin.plain };
	});
}

/**
 * ONE shipped string with its plain restatement, PINNED the same way.
 *
 * The array form above covers `caveats[]`. This covers the other unsafe shape:
 * a field whose text varies ROW BY ROW — every `note` on a nine-model table,
 * every `origin` on an operating-point row, every `what_it_computes` on a form
 * of INDRA's belief. Those cannot be twinned under a key that names them,
 * because one key (`note`) addresses nine different sentences, so each one
 * carries a verbatim fragment of the sentence its restatement was written for
 * and a drifted row gates the figure instead of being mislabelled.
 *
 * Use the plain `{ shipped: text(obj.x, …), plain: MODULE_PLAIN.x }` form for a
 * field whose key names exactly one sentence — there, the twin is safe by
 * construction and an anchor would only be ceremony.
 */
export function anchoredShippedProse(
	shipped: string,
	twin: AnchoredProse,
	context: string
): ShippedProse {
	if (!shipped.includes(twin.artifactAnchor)) {
		throw new Error(
			`${context}: no longer the sentence its plain restatement was written for ` +
				`(expected to contain ${JSON.stringify(twin.artifactAnchor)})`
		);
	}
	return { shipped, plain: twin.plain };
}

/**
 * Look a row's twin up by its FROZEN key, then pin it to the text as above.
 * A key with no restatement gates: a new row is a new sentence to explain, and
 * shipping it unexplained is the failure the whole twin mechanism exists to stop.
 */
export function keyedShippedProse(
	key: string,
	shipped: string,
	twins: Readonly<Record<string, AnchoredProse>>,
	context: string
): ShippedProse {
	const twin = twins[key];
	if (twin === undefined) {
		throw new Error(`${context}: no plain restatement is authored for "${key}"`);
	}
	return anchoredShippedProse(shipped, twin, context);
}
