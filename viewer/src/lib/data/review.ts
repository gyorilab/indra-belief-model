/**
 * Human-in-the-loop review: serve queue items BLINDED (no model verdict/score/
 * reasoning/bucket reaches the client until the human commits their own verdict),
 * then reveal, then capture an append-only multi-axis label.
 *
 * The blinding is the load-bearing design decision — `blindedItem()` returns only
 * what the annotator may see; `reveal()` is called by the commit action *after*
 * the human's verdict is recorded, so the model's persuasive reasoning never
 * anchors the first judgment.
 */
import { readFileSync, appendFileSync, existsSync, mkdirSync } from 'node:fs';
import { join } from 'node:path';
import { DATA_DIR, resolveRun } from './runs';
import { evidenceForStatement, getRunData } from './store';
import type { EvidenceRow, RunMeta } from './types';

const TRUTH_DIR = join(DATA_DIR, 'truth');

export interface QueueItem {
	item_id: string;
	stmt_hash: string;
	evidence_hash: string;
	stratum: string;
	bucket: string | null;
	stmt_type: string | null;
	source_api: string | null;
	stratum_weight: number | null;
	annotators: string[];
	double: boolean;
}

export interface QueueMeta {
	pass: string;
	run_id: string;
	model: string;
	export: string;
	totals: { items: number };
}

/** What the annotator may see before committing — NO model fields. */
export interface BlindedItem {
	item_id: string;
	subject: string;
	stmt_type: string;
	object: string;
	evidence_text: string | null;
	source_api: string | null;
	pmid: string | null;
}

/** Revealed only after the human commits Axis A + B. */
export interface RevealedModel {
	verdict: string | null;
	score: number | null;
	confidence: string | null;
	reasoning: string | null;
	bucket: string | null;
	tier: string | null;
	grounding_status: string | null;
}

function queuePath(pass: string): string {
	return join(TRUTH_DIR, `queue_${pass}.jsonl`);
}
function labelsPath(pass: string, annotator: string): string {
	return join(TRUTH_DIR, `labels_${pass}_${annotator}.jsonl`);
}

export function loadQueueMeta(pass: string): QueueMeta | null {
	const p = join(TRUTH_DIR, `queue_${pass}.meta.json`);
	if (!existsSync(p)) return null;
	try {
		return JSON.parse(readFileSync(p, 'utf8')) as QueueMeta;
	} catch {
		return null;
	}
}

export function loadQueue(pass: string): QueueItem[] {
	const p = queuePath(pass);
	if (!existsSync(p)) return [];
	const out: QueueItem[] = [];
	for (const line of readFileSync(p, 'utf8').split('\n')) {
		if (line.trim()) {
			try {
				out.push(JSON.parse(line) as QueueItem);
			} catch {
				/* skip */
			}
		}
	}
	return out;
}

/** item_ids this annotator has already labeled (append-only file). */
export function labeledIds(pass: string, annotator: string): Set<string> {
	const p = labelsPath(pass, annotator);
	const ids = new Set<string>();
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

/** This annotator's items, in queue order. */
function itemsForAnnotator(pass: string, annotator: string): QueueItem[] {
	return loadQueue(pass).filter((it) => it.annotators.includes(annotator));
}

export interface ReviewProgress {
	total: number;
	done: number;
	remaining: number;
}

export function progress(pass: string, annotator: string): ReviewProgress {
	const total = itemsForAnnotator(pass, annotator).length;
	const done = [...labeledIds(pass, annotator)].filter((id) =>
		itemsForAnnotator(pass, annotator).some((it) => it.item_id === id)
	).length;
	return { total, done, remaining: total - done };
}

function findQueueItem(pass: string, itemId: string): QueueItem | null {
	return loadQueue(pass).find((it) => it.item_id === itemId) ?? null;
}

function runForQueue(pass: string): RunMeta | null {
	const meta = loadQueueMeta(pass);
	return resolveRun(meta?.run_id);
}

function rowFor(pass: string, item: QueueItem): EvidenceRow | null {
	const run = runForQueue(pass);
	if (!run) return null;
	return evidenceForStatement(run, item.stmt_hash).find((r) => r.evidence_hash === item.evidence_hash) ?? null;
}

/** Next unlabeled item for this annotator, blinded. null when the queue is done. */
export function nextBlinded(pass: string, annotator: string): { item: QueueItem; blinded: BlindedItem } | null {
	const done = labeledIds(pass, annotator);
	const item = itemsForAnnotator(pass, annotator).find((it) => !done.has(it.item_id));
	if (!item) return null;
	const row = rowFor(pass, item);
	const blinded: BlindedItem = {
		item_id: item.item_id,
		subject: row?.subject ?? '?',
		stmt_type: row?.stmt_type ?? item.stmt_type ?? '?',
		object: row?.object ?? '?',
		evidence_text: row?.evidence_text ?? null,
		source_api: row?.source_api ?? item.source_api ?? null,
		pmid: row?.pmid ?? null
	};
	return { item, blinded };
}

/** The model's call — fetched ONLY by the commit action, after the human verdict. */
export function reveal(pass: string, itemId: string): RevealedModel | null {
	const item = findQueueItem(pass, itemId);
	if (!item) return null;
	const row = rowFor(pass, item);
	if (!row) return null;
	return {
		verdict: row.verdict,
		score: row.our_score,
		confidence: row.confidence,
		reasoning: row.reasoning,
		bucket: row.bucket,
		tier: row.tier,
		grounding_status: row.grounding_status
	};
}

export interface LabelInput {
	item_id: string;
	axis_a_faithful: string; // faithful | unfaithful | no_text | cant_tell
	axis_b_human_verdict: string; // correct | incorrect | abstain
	axis_c_reasoning: string | null; // sound | right_call_wrong_reason | wrong | na
	axis_d_failure: string | null; // reader_artifact | empty_evidence | direction_reversed | ...
	notes?: string;
}

/** Append the full label (human axes + frozen model snapshot) to the annotator's file. */
export function appendLabel(pass: string, annotator: string, input: LabelInput): boolean {
	const item = findQueueItem(pass, input.item_id);
	if (!item) return false;
	const row = rowFor(pass, item);
	const qmeta = loadQueueMeta(pass);
	const modelVerdict = row?.verdict ?? null;
	const record = {
		item_id: input.item_id,
		stmt_hash: item.stmt_hash,
		evidence_hash: item.evidence_hash,
		pass,
		annotator,
		stratum: item.stratum,
		bucket: item.bucket,
		stratum_weight: item.stratum_weight,
		// human axes
		axis_a_faithful: input.axis_a_faithful,
		axis_b_human_verdict: input.axis_b_human_verdict,
		axis_c_reasoning: input.axis_c_reasoning,
		axis_d_failure: input.axis_d_failure,
		notes: input.notes ?? '',
		// derived agreement (correct/incorrect only; abstain is neither)
		verdict_agree:
			input.axis_b_human_verdict === 'abstain' || modelVerdict == null
				? null
				: input.axis_b_human_verdict === modelVerdict,
		// frozen model snapshot — so a future rerun re-scores the SAME items
		model_verdict: modelVerdict,
		model_score: row?.our_score ?? null,
		model_confidence: row?.confidence ?? null,
		model_bucket: row?.bucket ?? null,
		model_snapshot: { run_id: qmeta?.run_id ?? null, model: qmeta?.model ?? null, export: qmeta?.export ?? null },
		labeled_at: new Date().toISOString()
	};
	mkdirSync(TRUTH_DIR, { recursive: true });
	appendFileSync(labelsPath(pass, annotator), JSON.stringify(record) + '\n');
	return true;
}

export function listPasses(): string[] {
	// passes that have a queue on disk
	const out: string[] = [];
	for (const pass of ['rough', 'robust']) {
		if (existsSync(queuePath(pass))) out.push(pass);
	}
	return out;
}
