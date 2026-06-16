/**
 * Shared formatting primitives for the viewer.
 *
 * Phase 5b primitives extraction: when a function is duplicated across 2+
 * routes/components, hoist it here. Tufte-pure, no styling — just data
 * shape. Components live in `$lib/components/`.
 */

/** First 8 chars of a hash; canonical short form for the UI. */
export function shortHash(h: string): string {
	return h.slice(0, 8);
}

/** Belief value to 2 decimals; em-dash for null. */
export function fmtBelief(b: number | null | undefined): string {
	return b == null ? '—' : b.toFixed(2);
}

/** Signed delta with proper minus glyph; em-dash for null. */
export function fmtDelta(d: number | null | undefined): string {
	if (d == null) return '—';
	const sign = d >= 0 ? '+' : '−';
	return `${sign}${Math.abs(d).toFixed(2)}`;
}

/** The baked run-level cost block (numbers only; the viewer holds no price
 *  table). Mirrors `RunMeta['cost']`; `null` ⇒ legacy export (pre-cost field). */
export interface RunCost {
	status: 'known' | 'partial' | 'unavailable';
	total_usd: number | null;
}

/**
 * Run-feed cost label (compact). Status is authoritative, NOT `total_usd`:
 *  - no cost block / `unavailable` → `"cost n/a"` (legacy export or no verified
 *    price; never $0 for an unpriced model).
 *  - `known`/`partial` with `total_usd == null` → `"$0.00"`. This is a genuine
 *    KNOWN-zero run: every row had 0 LLM calls (all no_text / auto-reject), so
 *    there was no priced spend to sum. The price IS verified — the run cost $0.
 *  - a real number → `"$0.00"` / `"<$0.01"` / `"$X.XX"`.
 */
export function fmtCost(c: RunCost | null | undefined): string {
	if (!c || c.status === 'unavailable') return 'cost n/a';
	if (c.total_usd == null || c.total_usd === 0) return '$0.00';
	return c.total_usd < 0.01 ? '<$0.01' : `$${c.total_usd.toFixed(2)}`;
}

/**
 * Full-precision cost for the detail page. A real $0 stays `$0.00`; sub-cent
 * shows 4 decimals; `null` → em-dash (the viewer never invents a price). Used
 * for the per-1k figure, where `null` (no LLM-scored rows) is genuinely "no
 * datum" and the em-dash is correct.
 */
export function fmtCostFull(n: number | null | undefined): string {
	return n == null ? '—' : n === 0 ? '$0.00' : '$' + n.toFixed(n < 0.01 ? 4 : 2);
}

/** Plural-aware label suffix. `n=1 → ''`, else `'s'`. */
export function pluralS(n: number): string {
	return n === 1 ? '' : 's';
}

export interface SentenceAgent {
	role: string;
	name: string;
}

/**
 * Render an INDRA statement as a readable English sentence. Falls back to
 * `Type(name1, name2)` for indra_types we don't have a verb form for.
 * Order of preference: subj/obj, then enz/sub, then members.
 */
export function sentenceFromStatement(
	indra_type: string,
	agents: SentenceAgent[]
): string {
	if (!agents || agents.length === 0) return indra_type;

	const by_role = (roles: string[]): SentenceAgent | undefined =>
		agents.find((a) => roles.includes(a.role));
	const members = agents.filter((a) => a.role === 'member');

	const verbs: Record<string, [string[], string[], string]> = {
		Activation: [['subj'], ['obj'], 'activates'],
		Inhibition: [['subj'], ['obj'], 'inhibits'],
		Phosphorylation: [['enz'], ['sub'], 'phosphorylates'],
		Dephosphorylation: [['enz'], ['sub'], 'dephosphorylates'],
		Ubiquitination: [['enz'], ['sub'], 'ubiquitinates'],
		Deubiquitination: [['enz'], ['sub'], 'deubiquitinates'],
		Methylation: [['enz'], ['sub'], 'methylates'],
		Demethylation: [['enz'], ['sub'], 'demethylates'],
		Acetylation: [['enz'], ['sub'], 'acetylates'],
		Deacetylation: [['enz'], ['sub'], 'deacetylates'],
		Sumoylation: [['enz'], ['sub'], 'sumoylates'],
		Desumoylation: [['enz'], ['sub'], 'desumoylates'],
		Hydroxylation: [['enz'], ['sub'], 'hydroxylates'],
		Dehydroxylation: [['enz'], ['sub'], 'dehydroxylates'],
		IncreaseAmount: [['subj'], ['obj'], 'increases the amount of'],
		DecreaseAmount: [['subj'], ['obj'], 'decreases the amount of'],
		Gef: [['gef'], ['ras'], 'activates'],
		Gap: [['gap'], ['ras'], 'inactivates'],
		Conversion: [['subj'], ['obj_from', 'obj'], 'converts'],
		RegulateActivity: [['subj'], ['obj'], 'regulates'],
		RegulateAmount: [['subj'], ['obj'], 'regulates the amount of']
	};

	const v = verbs[indra_type];
	if (v) {
		const [subjRoles, objRoles, verb] = v;
		const s = by_role(subjRoles);
		const o = by_role(objRoles);
		if (s && o) return `${s.name} ${verb} ${o.name}`;
	}

	if (indra_type === 'Complex' && members.length >= 2) {
		return `${members.map((m) => m.name).join(' · ')} (complex)`;
	}
	if (indra_type === 'Translocation' && agents.length > 0) {
		return `${agents[0].name} translocates`;
	}
	if (indra_type === 'Autophosphorylation' && agents.length > 0) {
		return `${agents[0].name} autophosphorylates`;
	}

	return `${indra_type}(${agents.map((a) => a.name).join(', ')})`;
}

/**
 * Map a belief score to a 3-class semantic role.
 * Used for color application on the score itself.
 */
export function beliefSemantic(b: number | null | undefined): 'high' | 'low' | 'mid' | 'absent' {
	if (b == null || Number.isNaN(b)) return 'absent';
	if (b >= 0.7) return 'high';
	if (b <= 0.3) return 'low';
	return 'mid';
}

/**
 * Pull a quoted cue from a probe rationale, e.g. `... 'was not' ...` → "was not".
 * Used to highlight the cue inside the corresponding evidence sentence so that
 * cause and effect sit visibly together.
 */
export function extractProbeCue(rationale: string | null | undefined): string | null {
	if (!rationale) return null;
	const m = rationale.match(/['"]([^'"\n]{2,40})['"]/);
	return m ? m[1] : null;
}

/**
 * Split an evidence sentence into pre/cue/post parts for inline highlighting.
 * When no cue or no match, returns one plain part.
 */
export function evidenceParts(
	text: string | null,
	cue: string | null
): Array<{ text: string; highlight: boolean }> {
	if (!text) return [{ text: '(no text)', highlight: false }];
	if (!cue) return [{ text, highlight: false }];
	const lower = text.toLowerCase();
	const i = lower.indexOf(cue.toLowerCase());
	if (i < 0) return [{ text, highlight: false }];
	return [
		{ text: text.slice(0, i), highlight: false },
		{ text: text.slice(i, i + cue.length), highlight: true },
		{ text: text.slice(i + cue.length), highlight: false }
	];
}

/**
 * Translate verdict enum to reader-facing language.
 * The stored values name the scorer's *internal* state ("correct" / "incorrect");
 * what the reader needs is the *claim's* status — supported, contradicted, or abstained.
 */
export function verdictDisplay(v: string | null | undefined): string {
	if (v === 'correct') return 'supported';
	if (v === 'incorrect') return 'contradicted';
	if (v === 'abstain') return 'abstained';
	return v ?? '—';
}

/** Plain-language name for HOW an evidence was scored — named for what the
 *  scorer actually did, not the internal pipeline-stage code it stores. */
export function scoringMethod(method: string | null | undefined): string | null {
	switch (method) {
		case 'llm_comprehension':
			return 'read the text';
		case 'llm_tool_use':
			return 'read + grounding check';
		case 'deterministic_mismatch':
			return 'grounding mismatch';
		case 'deterministic_pseudogene':
			return 'pseudogene';
		case 'no_text':
			return 'no sentence';
		case 'decomposed':
		case 'decomposed_probe_only':
			return 'decomposed probes';
		case 'panel':
			return 'objection panel';
		case 'row_error':
			return 'error';
		default:
			return method ?? null;
	}
}

/** Strip a leading bracket marker (e.g. an old "[TIER 2 LLM]" provenance tag
 *  baked into an already-exported trace) so the reasoning body renders clean.
 *  New runs carry no marker; this keeps historical runs tidy too. */
export function reasoningBody(reasoning: string | null | undefined): string {
	if (!reasoning) return '';
	return reasoning.replace(/^\s*\[[^\]]+\][ \t]*\n?/, '').trimStart();
}
