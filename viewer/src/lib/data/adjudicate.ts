/**
 * Inter-model disagreement adjudication. The queue (data/truth/queue_disagree.jsonl)
 * is the (statement, evidence) rows where two models gave OPPOSITE verdicts. The
 * human makes a blinded independent verdict first; only then are BOTH models'
 * calls + reasoning revealed, and we derive which model the human sided with.
 *
 * Blinding: `nextBlinded()` returns only the claim + sentence — neither model's
 * verdict/score/reasoning reaches the client until `revealPair()` is called by
 * the commit action, after the human's own verdict is recorded.
 */
import { readFileSync, appendFileSync, existsSync, mkdirSync } from 'node:fs';
import { join } from 'node:path';
import { DATA_DIR, resolveRun } from './runs';
import { evidenceForStatement, getCurationIndex, goldForRow } from './store';
import type { EvidenceRow, ReasoningTrace } from './types';

const TRUTH_DIR = join(DATA_DIR, 'truth');
const QUEUE = join(TRUTH_DIR, 'queue_disagree.jsonl');
const QUEUE_META = join(TRUTH_DIR, 'queue_disagree.meta.json');

export interface DisagreeItem {
	item_id: string;
	stmt_hash: string;
	evidence_hash: string;
	run_a: string;
	run_b: string;
	model_a: string;
	model_b: string;
	direction: string;
	stratum_weight: number | null;
	bucket_a: string | null;
	stmt_type: string | null;
	source_api: string | null;
	verdict_a: string;
	verdict_b: string;
	annotators: string[];
	double: boolean;
}

export interface DisagreeMeta {
	model_a: string;
	model_b: string;
	run_a: string;
	run_b: string;
	totals: { items: number };
}

export interface BlindedItem {
	item_id: string;
	subject: string;
	stmt_type: string;
	object: string;
	evidence_text: string | null;
	source_api: string | null;
	pmid: string | null;
}

export interface ModelCall {
	model: string;
	verdict: string | null;
	score: number | null;
	reasoning: string | null;
	bucket: string | null;
	reasoning_trace: ReasoningTrace | null;
}

function labelsPath(annotator: string): string {
	return join(TRUTH_DIR, `adjudications_${annotator}.jsonl`);
}

export function hasQueue(): boolean {
	return existsSync(QUEUE);
}

export function loadMeta(): DisagreeMeta | null {
	if (!existsSync(QUEUE_META)) return null;
	try {
		return JSON.parse(readFileSync(QUEUE_META, 'utf8')) as DisagreeMeta;
	} catch {
		return null;
	}
}

function loadQueue(): DisagreeItem[] {
	if (!existsSync(QUEUE)) return [];
	const out: DisagreeItem[] = [];
	for (const line of readFileSync(QUEUE, 'utf8').split('\n')) {
		if (line.trim()) {
			try {
				out.push(JSON.parse(line) as DisagreeItem);
			} catch {
				/* skip */
			}
		}
	}
	return out;
}

function labeledIds(annotator: string): Set<string> {
	const ids = new Set<string>();
	const p = labelsPath(annotator);
	if (!existsSync(p)) return ids;
	for (const line of readFileSync(p, 'utf8').split('\n')) {
		if (!line.trim()) continue;
		try {
			ids.add((JSON.parse(line) as { item_id: string }).item_id);
		} catch {
			/* skip */
		}
	}
	return ids;
}

function itemsFor(annotator: string): DisagreeItem[] {
	return loadQueue().filter((it) => it.annotators.includes(annotator));
}

export function progress(annotator: string): { total: number; done: number } {
	const items = itemsFor(annotator);
	const done = labeledIds(annotator);
	return { total: items.length, done: items.filter((it) => done.has(it.item_id)).length };
}

function findItem(itemId: string): DisagreeItem | null {
	return loadQueue().find((it) => it.item_id === itemId) ?? null;
}

function rowIn(runId: string, item: DisagreeItem): EvidenceRow | null {
	const run = resolveRun(runId);
	if (!run) return null;
	return evidenceForStatement(run, item.stmt_hash).find((r) => r.evidence_hash === item.evidence_hash) ?? null;
}

/** Next unlabeled disagreement for this annotator, blinded (no model calls). */
export function nextBlinded(annotator: string): { item: DisagreeItem; blinded: BlindedItem } | null {
	const done = labeledIds(annotator);
	const item = itemsFor(annotator).find((it) => !done.has(it.item_id));
	if (!item) return null;
	const row = rowIn(item.run_a, item) ?? rowIn(item.run_b, item);
	return {
		item,
		blinded: {
			item_id: item.item_id,
			subject: row?.subject ?? '?',
			stmt_type: row?.stmt_type ?? item.stmt_type ?? '?',
			object: row?.object ?? '?',
			evidence_text: row?.evidence_text ?? null,
			source_api: row?.source_api ?? item.source_api ?? null,
			pmid: row?.pmid ?? null
		}
	};
}

/** INDRA human-curation gold for an evidence row, via the int|int key bridge
 *  (the row carries both our hex hashes AND the INDRA int hashes). null when
 *  uncurated. This is a THIRD independent human judge — the INDRA community's
 *  curation — distinct from this tool's own blinded annotator. */
export interface GoldCall {
	verdict: 'correct' | 'incorrect';
	n: number;
	tags: string[];
	curators: string[];
	notes: string[];
}

function goldFor(row: EvidenceRow | null): GoldCall | null {
	const gv = goldForRow(getCurationIndex(), row);
	if (!gv) return null;
	return { verdict: gv.verdict, n: gv.n, tags: gv.tags, curators: gv.curators, notes: gv.notes };
}

/** Both models' calls + any INDRA curation gold — fetched ONLY after the human
 *  commits their own blinded verdict. */
export function revealPair(itemId: string): { a: ModelCall; b: ModelCall; gold: GoldCall | null } | null {
	const item = findItem(itemId);
	if (!item) return null;
	const ra = rowIn(item.run_a, item);
	const rb = rowIn(item.run_b, item);
	const call = (model: string, r: EvidenceRow | null, fallbackVerdict: string): ModelCall => ({
		model,
		verdict: r?.verdict ?? fallbackVerdict,
		score: r?.our_score ?? null,
		reasoning: r?.reasoning ?? null,
		bucket: r?.bucket ?? null,
		reasoning_trace: r?.reasoning_trace ?? null
	});
	return {
		a: call(item.model_a, ra, item.verdict_a),
		b: call(item.model_b, rb, item.verdict_b),
		gold: goldFor(ra ?? rb)
	};
}

export interface AdjInput {
	item_id: string;
	human_verdict: string; // correct | incorrect | abstain
	reasoning_a: string | null; // sound | wrong | na
	reasoning_b: string | null;
	ambiguous: boolean;
	notes?: string;
}

export function appendLabel(annotator: string, input: AdjInput): boolean {
	const item = findItem(input.item_id);
	if (!item) return false;
	const hv = input.human_verdict;
	// which model did the blinded human side with? (they disagree, so <=1 matches)
	const sided =
		hv === 'abstain' ? 'neither' : hv === item.verdict_a ? 'model_a' : hv === item.verdict_b ? 'model_b' : 'neither';
	// INDRA gold for this evidence (if any) — record how the human + each model
	// fared against the community curation, the independent third judge.
	const gold = goldFor(rowIn(item.run_a, item) ?? rowIn(item.run_b, item));
	const record = {
		item_id: input.item_id,
		stmt_hash: item.stmt_hash,
		evidence_hash: item.evidence_hash,
		annotator,
		run_a: item.run_a,
		run_b: item.run_b,
		model_a: item.model_a,
		model_b: item.model_b,
		direction: item.direction,
		stratum_weight: item.stratum_weight,
		bucket_a: item.bucket_a,
		stmt_type: item.stmt_type,
		// the blinded independent call + derived winner
		human_verdict: hv,
		model_a_verdict: item.verdict_a,
		model_b_verdict: item.verdict_b,
		sided_with: sided,
		// INDRA community gold (null when uncurated) + agreement flags
		gold_verdict: gold?.verdict ?? null,
		gold_tags: gold?.tags ?? null,
		gold_n: gold?.n ?? 0,
		human_matches_gold: gold ? (hv !== 'abstain' && hv === gold.verdict) : null,
		model_a_matches_gold: gold ? item.verdict_a === gold.verdict : null,
		model_b_matches_gold: gold ? item.verdict_b === gold.verdict : null,
		reasoning_quality_a: input.reasoning_a,
		reasoning_quality_b: input.reasoning_b,
		ambiguous: input.ambiguous,
		notes: input.notes ?? '',
		labeled_at: new Date().toISOString()
	};
	mkdirSync(TRUTH_DIR, { recursive: true });
	appendFileSync(labelsPath(annotator), JSON.stringify(record) + '\n');
	return true;
}

export function annotatorsFromMeta(): string[] {
	if (!existsSync(QUEUE_META)) return ['ann1'];
	try {
		const m = JSON.parse(readFileSync(QUEUE_META, 'utf8'));
		const p = m.params ?? {};
		return [...new Set([...(p.annotators ?? ['ann1']), p.double_annotator].filter(Boolean))] as string[];
	} catch {
		return ['ann1'];
	}
}
