/**
 * THE ONE VERIFICATION BOUNDARY FOR /paper.
 *
 * Six places on this page used to open onto the result files' own wording, each
 * introduced as "in the artifact's own words", "as shipped", "verbatim from the
 * artifact" or "how this is computed". A curator who opened one met a sentence
 * written for a referee — "tau = the smallest of the arm's own distinct scores",
 * "paired fold-stratified bootstrap over the paper's own out-of-fold fold
 * assignment" — and that was ALSO the pipe those words reached the screen
 * through after three static sweeps came back clean, because none of those
 * strings exists in this repository until a file is read.
 *
 * So the file's own wording collects HERE, once, at the end of the page, and is
 * introduced as what it is: the text you need to check this page against the
 * files it was built from. It is not a fuller version of anything above. It is
 * the same claim in the words the file uses, which is exactly why it is worth
 * having — a claim you can check is a claim stated twice, in two vocabularies,
 * by two people who cannot both be wrong in the same direction.
 *
 * WHAT THIS MODULE DOES, AND WHAT IT REFUSES TO DO
 *
 * It walks each loaded figure and collects every `{shipped, plain}` twin on it.
 * The set is DERIVED, never listed: a hand-kept list of which fields to show is
 * how the third home of one sentence went missing while its other two homes were
 * fixed. Anything a loader twins is in this section by construction, the moment
 * it is twinned, with no second edit anywhere.
 *
 * It refuses to guess provenance. Every group is named by the file the load
 * itself addresses, and carries the digest that load pinned. A file whose digest
 * the load does not carry says so, in place of a digest, because a section whose
 * entire job is verification must never print a number that cannot be checked.
 *
 * FAIL-CLOSED, PER FILE. A twin missing its shipped half means the page holds a
 * restatement whose source it cannot show. Rendering the plain half there would
 * present OUR sentence as the file's — the one thing this section exists to make
 * impossible — so that file's group gates and says so, and the rest of the trail
 * stands. Per file, not per section, for the same reason each figure gates on
 * its own artifact: one broken file must not take 240 checkable sentences with
 * it.
 */

import type { BeliefLadderLoad } from './paper-belief-ladder.ts';
import type { DeployedBaselineLoad } from './paper-deployed-baseline.ts';
import type { StatementErrorF1Load } from './paper-error-f1.ts';
import type { FramingCorrectionLoad } from './paper-framing-correction.ts';
import type { PaperLiteralLoad } from './paper-literal.ts';
import type { PaperOwnMetricLoad } from './paper-own-metric.ts';
import type { PaperPerEvidenceLoad } from './paper-per-evidence.ts';
import type { ReviewQueueLoad } from './paper-review-queue.ts';
import type { PaperRobustnessLoad } from './paper-robustness.ts';
import type { PaperTable6ExtendedLoad } from './paper-table6-extended.ts';

/** One shipped sentence beside the restatement the page uses in its place. */
export interface PaperAuditEntry {
	/**
	 * Where the twin sits on the loaded figure — `queue.prose.decisionRule`, not
	 * the file's own JSON path. An aid for whoever is holding the code; the
	 * sentence itself is what identifies the line in the file.
	 */
	field: string;
	/** The bytes as the file ships them. Quoted source: never edited, never trimmed. */
	shipped: string;
	/** The restatement every other part of this page renders in its place. */
	plain: string;
}

/** One result file, its digest, and every sentence this page draws from it. */
export interface PaperAuditFile {
	/** File name, taken from the path the load addresses. */
	file: string;
	/** Repo-relative path as the load carries it; null when it carries none. */
	path: string | null;
	/** sha256 of the bytes drawn; null when the load does not pin one. */
	sha256: string | null;
	/**
	 * Why this file contributes nothing, when it does not: either its own loader
	 * gated, or a twin on it was missing a half and this group gated in turn.
	 * Null when `entries` is what the page actually drew.
	 */
	unavailable: string | null;
	entries: PaperAuditEntry[];
}

export interface PaperAuditTrail {
	files: PaperAuditFile[];
	/** Files contributing at least one sentence. */
	nFiles: number;
	/** Sentences in the whole trail. */
	nEntries: number;
	/** Files carrying no sentence — gated loader, gated twin, or nothing twinned. */
	nFilesUnavailable: number;
	/**
	 * Shipped sentences restated two different ways somewhere on the page. Zero
	 * today. Not a gate — two contexts may legitimately need two restatements —
	 * but counted, because the two would otherwise be invisible to everyone
	 * except a reader who happened to scroll past both.
	 */
	nConflicts: number;
}

/** A load's provenance half: every /paper loader carries exactly this much. */
interface AuditedLoad {
	status: 'ok' | 'unavailable';
	reason: string | null;
	artifact_path: string;
	artifact_sha256: string | null;
}

/** One file to audit: where it came from, and the payload whose twins it owns. */
export interface PaperAuditSource {
	file: string;
	path: string | null;
	sha256: string | null;
	unavailable: string | null;
	/** The parsed payload. Walked for twins; never inspected field by field. */
	payload: unknown;
}

/**
 * `data/results/…/statement_error_f1.json` → `statement_error_f1.json`.
 * Derived from the path the load already carries rather than restated here: a
 * second copy of a file name is a second thing to keep true.
 */
function fileNameOf(path: string): string {
	const parts = path.split(/[\\/]/);
	return parts[parts.length - 1] || path;
}

/**
 * The provenance of one artifact-backed load, as a source. `payload` is passed
 * separately because the payload's key differs per loader (`queue`, `figure`,
 * `ladder`, …) and this needs none of that.
 */
function sourceOf(load: AuditedLoad, payload: unknown): PaperAuditSource {
	return {
		file: fileNameOf(load.artifact_path),
		path: load.artifact_path,
		sha256: load.artifact_sha256,
		unavailable: load.status === 'ok' ? null : (load.reason ?? 'this file is not on the page.'),
		payload: load.status === 'ok' ? payload : null
	};
}

/**
 * The band-by-band decomposition is read THROUGH the head-to-head loader, off a
 * sibling file in the same run directory, and that loader carries only the
 * head-to-head file's own path and digest. Its sentences are therefore named by
 * the file they came from and carry no digest, rather than borrowing the digest
 * of a different file — a wrong digest in a verification section is worse than
 * no digest, because it would check out against the wrong bytes.
 */
const AP_DECOMPOSITION_FILE = 'ap_decomposition_by_paper_band.json';

/** Every /paper load that reads an artifact carrying twinned prose. */
export interface PaperAuditPageLoads {
	paperLiteral: PaperLiteralLoad;
	paperOwnMetric: PaperOwnMetricLoad;
	deployedBaseline: DeployedBaselineLoad;
	paperPerEvidence: PaperPerEvidenceLoad;
	paperRobustness: PaperRobustnessLoad;
	framingCorrection: FramingCorrectionLoad;
	reviewQueue: ReviewQueueLoad;
	statementErrorF1: StatementErrorF1Load;
	paperTable6Extended: PaperTable6ExtendedLoad;
	beliefLadder: BeliefLadderLoad;
}

/**
 * The page's loads as audit sources, in the order the page reads them.
 *
 * Two loads contribute two files each. The framing correction reads its own
 * artifact and the non-reading control, and pins both; the head-to-head reads
 * the band-by-band decomposition through itself, and pins only its own.
 */
export function paperAuditSources(data: PaperAuditPageLoads): PaperAuditSource[] {
	return [
		/**
		 * THE HEAD-TO-HEAD FILE IS NOT A GROUP, and that is deliberate. The only
		 * twin its load carries is `generatedNoteProse`, and that twin's shipped
		 * half is ASSEMBLED HERE — the loader builds the sentence and appends the
		 * reproduction date from the run manifest — rather than read out of the
		 * file, as its own docblock in `paper-literal.ts` records. Listing it under
		 * that file's name and digest would put a sentence in this section that is
		 * not in those bytes, and the first person to check would find the one
		 * discrepancy in the one place built to surface discrepancies. The file's
		 * numbers are checked by `test-paper-literal-contract.mjs`; its sibling
		 * decomposition, whose sentences ARE artifact bytes, is the next entry.
		 */
		sourceOf(data.paperOwnMetric, data.paperOwnMetric.figure),
		sourceOf(data.paperTable6Extended, data.paperTable6Extended.figure),
		{
			file: AP_DECOMPOSITION_FILE,
			path: null,
			sha256: null,
			unavailable:
				data.paperLiteral.status === 'ok'
					? null
					: (data.paperLiteral.reason ?? 'this file is not on the page.'),
			payload: data.paperLiteral.apDecomposition
		},
		sourceOf(data.paperRobustness, data.paperRobustness.figure),
		sourceOf(data.framingCorrection, data.framingCorrection.framing),
		{
			file: fileNameOf(data.framingCorrection.control_path),
			path: data.framingCorrection.control_path,
			sha256: data.framingCorrection.control_sha256,
			unavailable:
				data.framingCorrection.status === 'ok'
					? null
					: (data.framingCorrection.reason ?? 'this file is not on the page.'),
			payload: data.framingCorrection.control
		},
		sourceOf(data.reviewQueue, data.reviewQueue.queue),
		sourceOf(data.statementErrorF1, data.statementErrorF1.figure),
		sourceOf(data.paperPerEvidence, data.paperPerEvidence.figure),
		sourceOf(data.beliefLadder, data.beliefLadder.ladder),
		sourceOf(data.deployedBaseline, data.deployedBaseline.figure)
	];
}

/**
 * Thrown by the walk when a twin is missing a half. Caught per file, so the
 * group gates with the reason and the rest of the trail is unaffected.
 */
class HalfTwinError extends Error {}

/**
 * Every twin under `payload`, in walk order, deduplicated by its two halves.
 *
 * A twin is any object carrying `shipped` or `plain`. Detection on EITHER key,
 * not both: an object with a plain half and no shipped half is precisely the
 * failure this must catch, and a rule that required both keys would silently
 * walk past it and render nothing at all.
 */
function collectEntries(payload: unknown): PaperAuditEntry[] {
	const byHalves = new Map<string, PaperAuditEntry>();
	const seen = new Set<object>();

	const walk = (node: unknown, path: string): void => {
		if (node === null || typeof node !== 'object') return;
		if (seen.has(node)) return;
		seen.add(node);

		if (!Array.isArray(node) && ('shipped' in node || 'plain' in node)) {
			const twin = node as { shipped?: unknown; plain?: unknown };
			if (typeof twin.shipped !== 'string' || twin.shipped.length === 0) {
				throw new HalfTwinError(
					`${path || 'the payload'} carries a restatement with no shipped sentence beside it`
				);
			}
			if (typeof twin.plain !== 'string' || twin.plain.length === 0) {
				throw new HalfTwinError(
					`${path || 'the payload'} carries a shipped sentence with no restatement beside it`
				);
			}
			const key = `${twin.shipped} ${twin.plain}`;
			if (!byHalves.has(key)) {
				byHalves.set(key, { field: path, shipped: twin.shipped, plain: twin.plain });
			}
			return;
		}

		if (Array.isArray(node)) {
			node.forEach((value, index) => walk(value, `${path}[${index}]`));
			return;
		}
		for (const [key, value] of Object.entries(node)) {
			walk(value, path ? `${path}.${key}` : key);
		}
	};

	walk(payload, '');
	return [...byHalves.values()];
}

/**
 * Assemble the trail. Pure: every sentence in it came off a load, and nothing
 * here rewrites, shortens or reorders a shipped string.
 */
export function buildPaperAuditTrail(sources: readonly PaperAuditSource[]): PaperAuditTrail {
	const files: PaperAuditFile[] = sources.map((source) => {
		if (source.unavailable !== null) {
			return {
				file: source.file,
				path: source.path,
				sha256: source.sha256,
				unavailable: source.unavailable,
				entries: []
			};
		}
		try {
			return {
				file: source.file,
				path: source.path,
				sha256: source.sha256,
				unavailable: null,
				entries: collectEntries(source.payload)
			};
		} catch (error) {
			if (!(error instanceof HalfTwinError)) throw error;
			return {
				file: source.file,
				path: source.path,
				sha256: source.sha256,
				unavailable: `this file's sentences are withheld — ${error.message}.`,
				entries: []
			};
		}
	});

	const plainsByShipped = new Map<string, Set<string>>();
	for (const file of files) {
		for (const entry of file.entries) {
			const plains = plainsByShipped.get(entry.shipped) ?? new Set<string>();
			plains.add(entry.plain);
			plainsByShipped.set(entry.shipped, plains);
		}
	}

	return {
		files,
		nFiles: files.filter((file) => file.entries.length > 0).length,
		nEntries: files.reduce((sum, file) => sum + file.entries.length, 0),
		nFilesUnavailable: files.filter((file) => file.entries.length === 0).length,
		nConflicts: [...plainsByShipped.values()].filter((plains) => plains.size > 1).length
	};
}

/** The whole trail from the page's loads, in one call. */
export function paperAuditTrail(data: PaperAuditPageLoads): PaperAuditTrail {
	return buildPaperAuditTrail(paperAuditSources(data));
}
