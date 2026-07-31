/**
 * THE RANKED VERDICT — the block that sits ABOVE every figure on /paper.
 *
 * A working biologist read this page and reported: "The page gives me six
 * answers and then argues with each of them. By the end I do not know whether I
 * am being told 'this is a real improvement', 'this is borderline', or 'the
 * benchmark is too small to say'. It never ranks its own claims." Sixteen
 * figures, every one of them scrupulously hedged, and no statement anywhere of
 * WHICH hedge matters. This module is the missing ranking, and the hedges below
 * it become support for a stated position instead of a seventh answer.
 *
 * WHAT IT IS. Three tiers. Each answers ONE question a curator actually asks,
 * carries the numbers that answer it, and carries the single strongest reason to
 * doubt the answer. Not a hedge per clause — one hedge, named, per claim.
 *
 * THREE RULES, each of which this page has already broken at least once:
 *
 *   1. NOTHING IS TYPED. Every number the block prints is read off a shipped
 *      artifact through its own loader. A tier whose load is dark renders as
 *      unavailable, carrying the loader's own reason for the verification
 *      boundary and an authored plain sentence for the reader; it never prints a
 *      tier with no evidence behind it, and it never substitutes a placeholder.
 *
 *   2. DIRECTION IS DECIDED HERE, NEVER AT A RENDER SITE, AND NEVER FROM A
 *      TWO-WAY TEST. Sign-blindness has shipped on this page six times, the sixth
 *      one HERE: `leader.pointwise.excludesZero ? 'Clears zero.' : 'Crosses
 *      zero.'` printed "Clears zero." under an interval of −0.0256 to −0.0061,
 *      because `excludesZero` is `low > 0 || high < 0` and is TRUE for an
 *      interval lying entirely below it. Reading the warning did not prevent it;
 *      the shape of the boolean did it. That boolean has since been deleted from
 *      every loader on /paper, so there is no longer one to read here or anywhere
 *      else. Every signed decision goes through `standingOfBounds`, which returns
 *      one of three classes, and every sentence that depends on a class is a total
 *      `Record` keyed by it — the compiler demands a sentence for each, and a
 *      two-way branch has nothing to branch on.
 *
 *   3. THE ORDER IS DERIVED. The tiers come out sorted by strength, and `rank`
 *      is assigned after the sort. If a rerun ever made the ordering claim
 *      stronger than the error-finding one, the block would reorder itself
 *      rather than keep printing a stale ranking.
 *
 * NO INVENTED VOCABULARY. This is the first thing a reader meets and may be all
 * they read, so it is held to the page's plain-language rule harder than
 * anything below it: models are named or called models, cutoffs are score
 * cutoffs, and "corrected for having run four reading models" is spelled out
 * every time rather than named after its statistic. Class (a4) of
 * `test-paper-render-invariants.mjs` reads every string in this file, because a
 * `paper-*.ts` module is scanned the day it lands.
 *
 * NO BADGES. The reader called the existing figure badges "scorecards in an
 * argument between authors and referees". A tier states its claim and states its
 * doubt; it does not perform modesty with a chip.
 *
 * Pure: no fs, no crypto, no artifact reading of its own. It consumes the loads
 * the page already performs, so it adds no read and cannot disagree with the
 * figures below it.
 */

import {
	fmtDelta,
	fmtPct,
	type ErrorF1Delta,
	type ErrorF1Figure,
	type ErrorF1Lane,
	type StatementErrorF1Load
} from './paper-error-f1.ts';
import { PAPER_LITERAL_ARM_SPECS, standingOfBounds, type Standing } from './paper-literal.ts';
import { reviewQueueEqualYieldPair, type ReviewQueueLoad } from './paper-review-queue.ts';
import type {
	PaperRobustnessFigure,
	PaperRobustnessInterval,
	PaperRobustnessLane,
	PaperRobustnessLoad
} from './paper-robustness.ts';

/**
 * How strong a claim is, in the only ordering this block uses.
 *
 *   · `solid`             — the margin clears zero even once the interval is
 *                           widened to cover every reading model we ran.
 *   · `real-but-small`    — it clears zero on its own and not once widened.
 *                           Unresolved, and reported as unresolved.
 *   · `not-shown`         — no margin clears zero at all. A claim we could not
 *                           demonstrate on this evidence.
 *   · `cannot-be-settled` — the benchmark itself is too small for the question:
 *                           the interval is at least as wide as the effect it
 *                           brackets, so a bigger or fresher set of labelled
 *                           statements is the only thing that would move it.
 *
 * `not-shown` ranks ABOVE `cannot-be-settled` deliberately: a claim this
 * evidence failed to demonstrate is still a question this evidence could answer,
 * and one the next run might. A question the benchmark cannot answer at any
 * outcome is weaker than that, and it is the honest floor of the page.
 */
export type VerdictStrength = 'solid' | 'real-but-small' | 'not-shown' | 'cannot-be-settled';

const VERDICT_STRENGTH_ORDER: Record<VerdictStrength, number> = {
	solid: 0,
	'real-but-small': 1,
	'not-shown': 2,
	'cannot-be-settled': 3
};

/** The strength word the block prints. Authored here; the component prints it. */
const VERDICT_STRENGTH_WORD: Record<VerdictStrength, string> = {
	solid: 'Solid',
	'real-but-small': 'Real but small',
	'not-shown': 'Not shown here',
	'cannot-be-settled': 'Cannot be settled here'
};

/**
 * The three questions, frozen. A tier's identity is its QUESTION; its strength
 * is read off the evidence. Keeping those apart is what lets the block reorder
 * itself without renaming anything.
 */
export type VerdictQuestionId = 'finds-wrong-statements' | 'orders-statements' | 'how-much-better';

/** One measured quantity behind a claim, formatted here and printed as given. */
export interface VerdictNumber {
	/** What the number is, in the reader's words. NOT `label` — that name is frozen. */
	caption: string;
	/** Already formatted. The component prints this; it never formats. */
	value: string;
	/** One clause of context, or null when the value stands alone. */
	note: string | null;
}

export interface VerdictTierOk {
	id: VerdictQuestionId;
	status: 'ok';
	/** 1-based, assigned AFTER the sort. 1 is the strongest claim on the page. */
	rank: number;
	strength: VerdictStrength;
	strengthWord: string;
	/** The question this tier answers, as the reader would ask it. */
	question: string;
	/** ONE sentence a curator could repeat. Chosen by strength, never assembled. */
	claim: string;
	/** The numbers that answer the question. Every one of them read off a load. */
	numbers: VerdictNumber[];
	/** The single strongest reason to doubt the claim. Exactly one, and named. */
	doubt: string;
}

export interface VerdictTierUnavailable {
	id: VerdictQuestionId;
	status: 'unavailable';
	rank: number;
	question: string;
	/**
	 * The loader's own reason, VERBATIM — never a paraphrase, never a placeholder.
	 *
	 * AUDIT TEXT, and not a sentence for a reader. A loader's reason is written for
	 * whoever is holding a broken artifact and is free to name that artifact's own
	 * fields: `arms[3].delta_error_f1 is not this arm's drawn error-F1` is a
	 * perfectly good one. It reached the screen anyway, because a gated tier used
	 * to print it straight, in a block whose other two tiers were working — so the
	 * page's own dialect rule was being enforced on every string EXCEPT the ones
	 * that appear when something breaks. It travels here for the page's single
	 * verification boundary; the component prints `plainReason` instead.
	 */
	reason: string;
	/**
	 * What the reader is told, authored here at the site that gates. It says which
	 * question is unanswered and why, in the block's own plain language, and it
	 * never quotes an artifact.
	 */
	plainReason: string;
}

export type VerdictTier = VerdictTierOk | VerdictTierUnavailable;

export interface PaperVerdictOk {
	status: 'ok';
	tiers: VerdictTier[];
	reason: null;
}

export interface PaperVerdictUnavailable {
	status: 'unavailable';
	/** Still the three questions, each carrying the reason its own evidence is dark. */
	tiers: VerdictTier[];
	reason: string;
}

export type PaperVerdictLoad = PaperVerdictOk | PaperVerdictUnavailable;

/** The loads this block reads. Exactly the ones /paper already performs. */
export interface PaperVerdictInput {
	statementErrorF1: StatementErrorF1Load;
	reviewQueue: ReviewQueueLoad;
	paperRobustness: PaperRobustnessLoad;
}

const QUESTIONS: Record<VerdictQuestionId, string> = {
	'finds-wrong-statements': 'Does it find wrong statements better?',
	'orders-statements': 'Does it put the whole list in a better order?',
	'how-much-better': 'How much better is it, in general?'
};

/**
 * What a reader is told when the artifact a question is answered from did not
 * load. One authored sentence per question, naming the evidence that is dark —
 * printed in place of the loader's own reason, which is audit text.
 */
const DARK_PLAIN_REASON: Record<VerdictQuestionId, string> = {
	'finds-wrong-statements':
		'The file the error-finding numbers are read from did not load, so this question is left ' +
		'unanswered rather than answered from memory.',
	'orders-statements':
		'The file the ordering numbers are read from did not load, so this question is left ' +
		'unanswered rather than answered from memory.',
	'how-much-better':
		'The file the ordering numbers are read from did not load, so this question is left ' +
		'unanswered rather than answered from memory.'
};

/** A row that carries a margin — i.e. everything but the reference row. */
type DecidedRow = ErrorF1Lane & { delta: ErrorF1Delta };

/** Robustness values are shipped in hundredths; the page argues in the raw unit. */
function fromPts(value: number): number {
	return value / 100;
}

/** A t statistic at one decimal — the precision the margin is argued at. */
function fmtT(value: number): string {
	return value.toFixed(1);
}

/**
 * A signed range, collapsing to a single value when the ends agree. Two models
 * with the same margin must not print "+0.1260 to +0.1260" as if it were a
 * spread, and one model must not print a range at all.
 */
function fmtRange(low: number, high: number, format: (value: number) => string): string {
	const a = format(low);
	const b = format(high);
	return a === b ? a : `${a} to ${b}`;
}

/**
 * How the reading models that did NOT win stand against the random forest, as a
 * verb. Selected from the shipped three-way standing rather than from a bare
 * "excludes zero", which is sign-blind: a model whose interval sits entirely
 * BELOW zero is behind, not level, and this page has printed the wrong one of
 * those before.
 */
function standingVerb(standings: Standing[]): string {
	if (standings.every((standing) => standing === 'behind')) return 'scored below';
	if (standings.every((standing) => standing === 'not-significant')) return 'came out level with';
	return 'did not beat';
}

/**
 * The doubt for the case where NOTHING clears the widened interval, keyed on
 * where the models we ran actually sit.
 *
 * The same defect as the ordering tier's, one tier up: a single unconditional
 * sentence — "widened to cover every reading model we ran, the interval crosses
 * zero" — was printed for BOTH sub-cases. It is provably true of a model that is
 * ahead pointwise and no further, and it is FALSE of a model whose widened
 * interval lies entirely below zero, which is a result rather than an absence of
 * one. Three cases, three sentences, chosen from the three-way standing the
 * error-finding loader already ships.
 */
type NoWinnerCase = 'all-behind' | 'none-behind' | 'mixed';

function noWinnerCaseOf(standings: Standing[]): NoWinnerCase {
	if (standings.length > 0 && standings.every((standing) => standing === 'behind')) {
		return 'all-behind';
	}
	if (standings.every((standing) => standing !== 'behind')) return 'none-behind';
	return 'mixed';
}

const NO_WINNER_DOUBT: Record<NoWinnerCase, string> = {
	'all-behind':
		'Every reading model we ran scored BELOW the random forest here, and the widened interval ' +
		'keeps them there — this is a measured shortfall, not an open question. What it cannot tell ' +
		'you is where a model we did not run would land.',
	'none-behind':
		'Widened to cover every reading model we ran, the interval crosses zero. Read it as ' +
		'unresolved rather than as a result.',
	mixed:
		'Some of the reading models we ran scored BELOW the random forest here by a margin that ' +
		'survives the widened interval, and none scored above it by one. Read the figure model by ' +
		'model rather than as a single answer.'
};

/** English list: "A", "A and B", "A, B and C". */
function joinNames(names: string[]): string {
	if (names.length <= 1) return names[0] ?? '';
	return `${names.slice(0, -1).join(', ')} and ${names[names.length - 1]}`;
}

const CANONICAL_DISPLAY_BY_ID = new Map(
	PAPER_LITERAL_ARM_SPECS.map((spec) => [spec.id, spec.display])
);

/**
 * ONE name per model across the whole block.
 *
 * The three figures this block reads do not agree with each other today: the
 * error-finding and review-budget surfaces draw the canonical "Gemma 4 26B
 * reading", the ordering surface still draws "Gemma 4 26B gate". A reader who
 * meets both in the same six sentences has no way to know they are one model, and
 * "gate" is the noun the page's plain-language rule replaced with "reading" in
 * the first place. So every name in this block resolves through the canonical
 * spec, keyed by the id the shipped artifacts share.
 *
 * The fallback is the figure's own display name, never the join key. It is
 * unreachable in practice — the ordering figure's own validator already gates
 * each id against the same spec table — but a name is what a reader needs, and
 * dropping the tier because a spec row went missing would cost more than it saves.
 */
function canonicalDisplay(id: string, shipped: string): string {
	return CANONICAL_DISPLAY_BY_ID.get(id) ?? shipped;
}

// ---------------------------------------------------------------------------
// WHERE AN INTERVAL SITS RELATIVE TO ZERO — the one classifier this block uses
//
// The ordering figure used to ship a bare `excludesZero` per interval — the
// boolean `low > 0 || high < 0`, TRUE for an interval lying entirely BELOW zero.
// Every sign-blindness regression this page has had — six, the sixth in this very
// module — was someone writing `excludesZero ? A : B` and getting the losing case
// printed as the winning one. That boolean no longer exists on /paper: it was
// deleted from every loader in favour of `standingOfBounds` (paper-literal.ts),
// which returns one of three classes. A two-way branch on direction is now
// unwritable, because there is no boolean to write it against.
//
// This block still classifies the ordering intervals ITSELF, from the endpoints,
// rather than reading the class the figure now carries. That is deliberate and it
// is contract-tested: a shipped class handed to this block is data like any other,
// and the block's job is to be right about the direction even when the object it
// is handed is wrong about it. It calls the same shared helper, so there is no
// second convention — only a second, independent application of the one rule.
// ---------------------------------------------------------------------------

/** The two intervals the ordering tier prints, each already classified. */
interface OrderingStanding {
	/** The model's own 95% interval. */
	pointwise: Standing;
	/** The same interval widened to cover every reading model we ran. */
	simultaneous: Standing;
}

/**
 * Classify the ordering leader's two intervals, or return null when the shipped
 * numbers contradict each other and no classification would be honest.
 *
 * The gate is not ceremony. A class read off the endpoints and a sign read off
 * the point estimate agree only when the point estimate lies inside the interval
 * drawn around it, and the ordering figure's own validator does not check that
 * (paper-robustness.ts `parseSide` checks endpoint order, band containment and
 * each shipped `excludes_zero_*` flag against its own endpoints — not this). A
 * margin outside its own interval would make the tier print a direction and a
 * range that disagree, so the tier gates instead.
 */
function orderingStandingOf(leader: PaperRobustnessLane): OrderingStanding | null {
	const consistent = (interval: PaperRobustnessInterval) =>
		interval.deltaPts >= interval.lowPts && interval.deltaPts <= interval.highPts;
	if (!consistent(leader.pointwise) || !consistent(leader.simultaneous)) return null;
	return {
		pointwise: standingOfBounds(leader.pointwise.lowPts, leader.pointwise.highPts),
		simultaneous: standingOfBounds(leader.simultaneous.lowPts, leader.simultaneous.highPts)
	};
}

/**
 * The note printed beside a model's OWN interval. One authored sentence per
 * class, so the sentence cannot be chosen by a two-way test.
 */
const INTERVAL_NOTE: Record<Standing, string> = {
	ahead: 'Lies entirely above zero.',
	behind: 'Lies entirely below zero — the random forest is the one ahead here.',
	'not-significant': 'Crosses zero.'
};

/**
 * The note printed beside the same interval widened to cover every reading model.
 * "Still" is accurate in both signed classes and only there: the widened band is
 * gated to CONTAIN the model's own interval (paper-robustness.ts `parseSide`), so
 * a widened band that clears zero has an own-interval that cleared it too.
 */
const CORRECTED_INTERVAL_NOTE: Record<Standing, string> = {
	ahead: 'Still lies entirely above zero.',
	behind: 'Still lies entirely below zero — the random forest is the one ahead here.',
	'not-significant': 'Crosses zero.'
};

// ---------------------------------------------------------------------------
// TIER 1 — does it find wrong statements better?
// ---------------------------------------------------------------------------

/**
 * The reading model whose flags are right most often. An argmax over ERROR
 * PRECISION, which is exactly the quantity the sentence beneath it prints —
 * "when it flags a statement as wrong it is right N% of the time". Maximising
 * one quantity and printing another is how three argmaxes further down this page
 * came to read as one system; here the two are the same by construction.
 */
function precisionLeaderOf(winners: DecidedRow[]): DecidedRow | null {
	if (winners.length === 0) return null;
	return winners.reduce((best, row) =>
		row.operating.errorPrecision > best.operating.errorPrecision ? row : best
	);
}

function errorFindingNumbers(
	figure: ErrorF1Figure,
	winners: DecidedRow[],
	leader: DecidedRow,
	reviewQueue: ReviewQueueLoad
): VerdictNumber[] {
	const margins = winners.map((row) => row.delta.delta);
	const tStats = winners.map((row) => row.delta.tStatistic);
	/** The oracle's size on the winning side — see the note on the first number. */
	const winnerCuts = winners.map((row) => row.distinctScores);
	/**
	 * ARE THE WINNERS TELLABLE APART? Derived, never asserted. A reader who meets
	 * five named models across six adjacent beats asks which one to actually run,
	 * and the page had no answer: each figure names its own best and none says the
	 * leaders are a tie. They are — every winner's margin lies inside every other
	 * winner's interval — so the honest guidance is that the choice is not a
	 * quality choice. If a future run separates them this goes false and the
	 * sentence stops rendering, which is the point of computing it.
	 */
	const winnersAreATie =
		winners.length > 1 &&
		winners.every((row) =>
			winners.every(
				(other) => row.delta.delta >= other.delta.ciLow && row.delta.delta <= other.delta.ciHigh
			)
		);
	const numbers: VerdictNumber[] = [
		{
			caption: 'when it flags a statement as wrong, it is right',
			value: `${fmtPct(leader.operating.errorPrecision)} of the time`,
			// BOTH SIDES ARE TUNED, AND THE TUNING FAVOURS THE FOREST. This note used
			// to attach "at the best cutoff" to the random forest alone, so the
			// strongest claim on the page read as "our untuned number beats their
			// tuned number" — the opposite of what the artifact says. Every cutoff
			// here was chosen with the labels in hand, on the same statements it is
			// then scored on. The advantage of that is the FOREST's: it had far more
			// candidate cutoffs to search, and it still loses.
			note:
				`${canonicalDisplay(leader.id, leader.display)} against the random forest's ` +
				`${fmtPct(figure.reference.operating.errorPrecision)}. Both cutoffs were chosen with the ` +
				`answers already in hand, on these same ${figure.panel.n.toLocaleString()} statements — ` +
				`and the search favoured the forest, which had ` +
				`${figure.reference.distinctScores.toLocaleString()} candidate cutoffs to choose from ` +
				`against ${fmtRange(Math.min(...winnerCuts), Math.max(...winnerCuts), (n) => n.toLocaleString())} ` +
				`for the reading models.`
		},
		{
			caption:
				'margin on error-class F1 — precision and recall taken together, for finding wrong statements',
			value: fmtRange(Math.min(...margins), Math.max(...margins), fmtDelta),
			note:
				`${joinNames(winners.map((row) => canonicalDisplay(row.id, row.display)))}, against the ` +
				`random forest. ` +
				`Ratio of margin to its own spread ${fmtRange(Math.min(...tStats), Math.max(...tStats), fmtT)}, ` +
				`where ${figure.multiplicity.criticalValue.toFixed(4)} is what it has to clear once the ` +
				`interval is widened to cover all ${figure.multiplicity.familySize} reading models we ran. ` +
				`Measured at the same chosen cutoffs as the row above, not on held-back statements.` +
				(winnersAreATie
					? ` These ${winners.length} cannot be told apart here — each one's margin sits inside ` +
						`the others' intervals — so the choice between them is cost, not quality.`
					: '')
		}
	];

	// The same finding as a curator's afternoon. Included only when the model it
	// is measured against is the one the 2023 paper actually published: the
	// comparator is derived upstream as whichever scorer needs fewest extra
	// reviews, so the sentence's "the random forest" has to be checked, not
	// assumed. Its absence drops a supporting row, never the claim.
	if (reviewQueue.status !== 'ok') return numbers;
	const queue = reviewQueue.queue;
	const pair = reviewQueueEqualYieldPair(queue);
	if (pair && pair.comparatorArm.provenance === 'paper-published') {
		numbers.push({
			caption: 'at a fixed checking budget',
			value:
				`${pair.reference.budget} statements read catches ` +
				`${pair.reference.trueErrorsCaught} of ${queue.panel.nErrors} known errors`,
			note:
				`${pair.referenceArm.display}, at a cutoff nobody chose — the statements whose evidence it ` +
				`threw out entirely. The random forest needs ${pair.comparator.budgetForEqualYield} reads to ` +
				`find the same ${pair.reference.trueErrorsCaught}, and its cutoff was picked with the answers ` +
				`already in hand.`
		});
	}
	return numbers;
}

function errorFindingDoubt(figure: ErrorF1Figure, winners: DecidedRow[]): string {
	const losers = figure.lanes.filter(
		(row): row is DecidedRow => row.inMaxTFamily && row.delta !== null && !row.delta.winsSimultaneously
	);
	if (losers.length === 0) {
		return (
			'Every statement counted wrong here is one the 2023 INDRA assembly paper released as wrong. ' +
			'A statement it released as correct that is in fact wrong counts against every model scored ' +
			'here, and nothing on this page can find those.'
		);
	}
	const verb = standingVerb(losers.map((row) => row.delta.standing));
	return (
		`It is not a property of reading as such. Of the ${figure.multiplicity.familySize} reading models ` +
		`we ran, ${joinNames(losers.map((row) => canonicalDisplay(row.id, row.display)))} ` +
		`${verb} the random forest ` +
		`(${joinNames(losers.map((row) => fmtDelta(row.delta.delta)))}), so this is a claim about the ` +
		`${winners.length === 1 ? 'model' : `${winners.length} models`} named above and not about ` +
		'language models in general.'
	);
}

function buildErrorFindingTier(
	statementErrorF1: StatementErrorF1Load,
	reviewQueue: ReviewQueueLoad
): Omit<VerdictTierOk, 'rank'> | Omit<VerdictTierUnavailable, 'rank'> {
	const id: VerdictQuestionId = 'finds-wrong-statements';
	const question = QUESTIONS[id];
	if (statementErrorF1.status !== 'ok') {
		return {
			id,
			status: 'unavailable',
			question,
			reason: statementErrorF1.reason,
			plainReason: DARK_PLAIN_REASON[id]
		};
	}
	const figure = statementErrorF1.figure;
	const decided = figure.lanes.filter((row): row is DecidedRow => row.delta !== null);
	// `winsSimultaneously` is null outside the family we ran, so a simultaneous
	// winner is already a reading model. The pointwise fallback is not, and the
	// sentences it feeds say "reading model" — the descriptive row this figure
	// also carries is not one, and must not be quoted as though it were.
	const winners = decided.filter((row) => row.delta.winsSimultaneously);
	const pointwiseOnly = decided.filter((row) => row.inMaxTFamily && row.delta.winsPointwise);

	if (winners.length === 0) {
		// Nothing clears zero once the interval covers every reading model we ran.
		// The tier still renders — with the claim the evidence supports, which is a
		// weaker one, and never the sentence written for the stronger case.
		const strength: VerdictStrength = pointwiseOnly.length > 0 ? 'real-but-small' : 'not-shown';
		const leader = precisionLeaderOf(pointwiseOnly);
		const family = decided.filter((row) => row.inMaxTFamily);
		return {
			id,
			status: 'ok',
			strength,
			strengthWord: VERDICT_STRENGTH_WORD[strength],
			question,
			claim:
				leader === null
					? 'On these statements no reading model finds wrong statements better than the random ' +
						'forest the 2023 INDRA assembly paper published.'
					: 'A reading model finds more wrong statements than the random forest the 2023 INDRA ' +
						'assembly paper published, but not by a margin that holds once we account for having ' +
						'tried several models.',
			numbers:
				leader === null
					? [
							{
								// AHEAD OF, not "clears zero". The count is `winsPointwise`, which is
								// sign-aware — `delta > 0` and the interval excluding zero — so a
								// caption reading "whose margin clears zero" would have described a
								// figure of four significant LOSSES as none of them clearing zero,
								// when in fact all four cleared it on the other side.
								caption: 'reading models ahead of the random forest on their own interval',
								value: `0 of ${figure.multiplicity.familySize}`,
								note: `Measured on ${figure.panel.n.toLocaleString()} statements against the random forest.`
							}
						]
					: errorFindingNumbers(figure, pointwiseOnly, leader, reviewQueue),
			doubt:
				leader === null
					? NO_WINNER_DOUBT[noWinnerCaseOf(family.map((row) => row.delta.standing))]
					: 'Widened to cover every reading model we ran, the interval crosses zero. Read it as ' +
						'unresolved rather than as a result.'
		};
	}

	const leader = precisionLeaderOf(winners);
	if (leader === null) {
		return {
			id,
			status: 'unavailable',
			question,
			reason: 'no reading model on the error-finding figure carries a margin to rank.',
			plainReason:
				'The figure behind this question loaded but names no reading model, so there is nothing ' +
				'to rank.'
		};
	}
	return {
		id,
		status: 'ok',
		strength: 'solid',
		strengthWord: VERDICT_STRENGTH_WORD.solid,
		question,
		claim:
			'Letting a language model throw out the evidence it judges unsupported finds more wrong ' +
			'statements than the random forest the 2023 INDRA assembly paper published, which counts how ' +
			'often each source reported a statement and never reads the sentence.',
		numbers: errorFindingNumbers(figure, winners, leader, reviewQueue),
		doubt: errorFindingDoubt(figure, winners)
	};
}

// ---------------------------------------------------------------------------
// TIER 2 — does it put the whole list in a better order?
// ---------------------------------------------------------------------------

/** The widest ordering margin. An argmax over the quantity the tier prints. */
function orderingLeaderOf(figure: PaperRobustnessFigure): PaperRobustnessLane | null {
	if (figure.lanes.length === 0) return null;
	return figure.lanes.reduce((best, row) => (row.primaryDelta > best.primaryDelta ? row : best));
}

function orderingNumbers(
	figure: PaperRobustnessFigure,
	leader: PaperRobustnessLane,
	standing: OrderingStanding
): VerdictNumber[] {
	return [
		{
			caption: 'margin on average precision, over all statements at once',
			value: fmtDelta(leader.primaryDelta),
			note:
				`${canonicalDisplay(leader.id, leader.display)} against the random forest, on ` +
				`${figure.primaryPanel.nStatements.toLocaleString()} statements.`
		},
		{
			caption: 'its own 95% interval',
			value: `${fmtDelta(fromPts(leader.pointwise.lowPts))} to ${fmtDelta(fromPts(leader.pointwise.highPts))}`,
			note: INTERVAL_NOTE[standing.pointwise]
		},
		{
			caption: `the same interval, corrected for having run ${figure.multiplicity.familySize} reading models`,
			value: `${fmtDelta(fromPts(leader.simultaneous.lowPts))} to ${fmtDelta(fromPts(leader.simultaneous.highPts))}`,
			note: CORRECTED_INTERVAL_NOTE[standing.simultaneous]
		}
	];
}

/**
 * What this benchmark says about the ordering question. FOUR cases, because four
 * are reachable — the previous revision wrote three branches for them and let the
 * fourth (ahead on the point estimate, own interval crossing zero) fall through
 * into a sentence written for a different case.
 *
 * Each case is a KEY, and every sentence the tier prints is a total `Record` over
 * that key. A new case cannot be added without the compiler demanding its claim,
 * its doubt and its strength, which is the structural half of the fix: the words
 * cannot fall out of step with the evidence by omission.
 */
type OrderingCase = 'clears-corrected' | 'clears-alone' | 'crosses-zero' | 'behind';

/**
 * Pick the case from the two signed standings. Sign FIRST, size second, and never
 * from a bare zero-exclusion test.
 *
 * `behind` is checked before `crosses-zero` and after both `ahead` classes: the
 * widened band contains the model's own interval, so `ahead` on the widened band
 * implies `ahead` on the model's own and the first two tests can never fire on a
 * model that is behind.
 */
function orderingCaseOf(standing: OrderingStanding): OrderingCase {
	if (standing.simultaneous === 'ahead') return 'clears-corrected';
	if (standing.pointwise === 'ahead') return 'clears-alone';
	if (standing.pointwise === 'behind' || standing.simultaneous === 'behind') return 'behind';
	return 'crosses-zero';
}

const ORDERING_STRENGTH: Record<OrderingCase, VerdictStrength> = {
	'clears-corrected': 'solid',
	'clears-alone': 'real-but-small',
	'crosses-zero': 'not-shown',
	behind: 'not-shown'
};

/**
 * The claim per case, with the margin PASSED IN rather than described.
 *
 * The sentences these replace said "by about a hundredth of a point of average
 * precision" — a number typed into prose, correct for the margin shipped today and
 * silently wrong the first time a rerun moves it. The size a claim states is the
 * size the row beneath it prints, because both come from the same value.
 */
const ORDERING_CLAIM: Record<OrderingCase, (margin: string) => string> = {
	'clears-corrected': (margin) =>
		`Language-model reading also puts the whole list of statements in a better order than that ` +
		`random forest, by ${margin} of average precision.`,
	'clears-alone': (margin) =>
		`Language-model reading also puts the whole list of statements in a slightly better order ` +
		`than that random forest, by ${margin} of average precision — but this one is unresolved, ` +
		`not won.`,
	'crosses-zero': (margin) =>
		`Whether language-model reading puts the whole list of statements in a better order than the ` +
		`random forest the 2023 INDRA assembly paper published is not resolved by this benchmark: ` +
		`its best margin, ${margin} of average precision, does not clear zero even before correcting ` +
		`for having run several reading models.`,
	behind: (margin) =>
		`No reading model puts the whole list of statements in a better order than the random forest ` +
		`the 2023 INDRA assembly paper published — the closest one scores BELOW it, by ${margin} of ` +
		`average precision, on an interval that stays below zero.`
};

const ORDERING_DOUBT: Record<OrderingCase, string> = {
	'clears-corrected':
		'It is still the most conservative view on this page: the random forest already scores near ' +
		'the top of the scale, so the room a better order can win in is small.',
	'clears-alone':
		'Corrected for having run several reading models rather than one, the interval crosses zero. ' +
		'Report it as unresolved, not as a win — it is not a number to lean on.',
	'crosses-zero':
		'The margin fails to clear zero on the model’s own interval, before any correction for having ' +
		'run several reading models, so nothing here separates it from no difference at all.',
	behind:
		'It is measured on the same statements the random forest was built from, which is the ground ' +
		'most favourable to it; a set drawn after the fact could put the difference somewhere else.'
};

function buildOrderingTier(
	paperRobustness: PaperRobustnessLoad
): Omit<VerdictTierOk, 'rank'> | Omit<VerdictTierUnavailable, 'rank'> {
	const id: VerdictQuestionId = 'orders-statements';
	const question = QUESTIONS[id];
	if (paperRobustness.status !== 'ok') {
		return {
			id,
			status: 'unavailable',
			question,
			reason: paperRobustness.reason,
			plainReason: DARK_PLAIN_REASON[id]
		};
	}
	const figure = paperRobustness.figure;
	const leader = orderingLeaderOf(figure);
	if (leader === null) {
		return {
			id,
			status: 'unavailable',
			question,
			reason: 'the ordering figure carries no model to rank.',
			plainReason:
				'The figure behind this question loaded but names no reading model, so there is nothing ' +
				'to rank.'
		};
	}
	const standing = orderingStandingOf(leader);
	if (standing === null) {
		return {
			id,
			status: 'unavailable',
			question,
			reason: 'the ordering margin lies outside the interval shipped around it.',
			plainReason:
				'The margin and the interval printed around it disagree in the shipped file, so this ' +
				'question is left unanswered rather than answered from numbers that contradict each other.'
		};
	}

	// Sign-aware, and three-way rather than two: a margin whose interval sits
	// entirely below zero is a LOSS, and "excludes zero" alone reports it as a win.
	// The case is chosen from the signed standings above; the claim, the doubt and
	// the strength are then looked up, never assembled.
	const orderingCase = orderingCaseOf(standing);
	const strength = ORDERING_STRENGTH[orderingCase];

	return {
		id,
		status: 'ok',
		strength,
		strengthWord: VERDICT_STRENGTH_WORD[strength],
		question,
		claim: ORDERING_CLAIM[orderingCase](fmtDelta(leader.primaryDelta)),
		numbers: orderingNumbers(figure, leader, standing),
		doubt: ORDERING_DOUBT[orderingCase]
	};
}

// ---------------------------------------------------------------------------
// TIER 3 — how much better, in general?
// ---------------------------------------------------------------------------

/**
 * What can be said about the SIZE of the ordering difference.
 *
 * The previous revision compared `Math.abs(delta)` against the interval's width
 * and nothing else, so a LOSS with a tight interval — say −0.0160 from −0.0164 to
 * −0.0156 — came out as "real but small" under a claim that reads "how much
 * better … the range is narrower than the effect it brackets". Magnitude alone
 * cannot answer a question with the word "better" in it.
 */
type GeneralityCase = 'uncertainty-exceeds-effect' | 'size-pinned-ahead' | 'size-pinned-behind';

/**
 * Sign FIRST, width second.
 *
 * The `not-significant` case needs no width test and gets none: an interval that
 * contains zero also contains the margin drawn inside it (gated in
 * `orderingStandingOf`), so its width is at least the size of that margin by
 * arithmetic. Deciding it on sign keeps the branch true even if that arithmetic
 * were ever reached with a margin of exactly zero, where a width test alone read
 * "the range is narrower than the effect" off an effect of nothing.
 */
function generalityCaseOf(
	standing: Standing,
	effect: number,
	width: number
): GeneralityCase {
	if (standing === 'not-significant') return 'uncertainty-exceeds-effect';
	if (width >= effect) return 'uncertainty-exceeds-effect';
	return standing === 'ahead' ? 'size-pinned-ahead' : 'size-pinned-behind';
}

const GENERALITY_CLAIM: Record<GeneralityCase, (nStatements: string) => string> = {
	'uncertainty-exceeds-effect': (n) =>
		`How much better language-model reading is, in general, is not something ${n} statements can ` +
		`settle: on the ordering question the uncertainty is the size of the effect.`,
	'size-pinned-ahead': (n) =>
		`How much better language-model reading is can be put within a range on these ${n} ` +
		`statements, and the range is narrower than the margin it brackets.`,
	'size-pinned-behind': (n) =>
		`On these ${n} statements the question is not how much better: on the ordering question ` +
		`language-model reading scores BELOW the random forest, by more than the width of the range ` +
		`around that shortfall.`
};

const GENERALITY_STRENGTH: Record<GeneralityCase, VerdictStrength> = {
	'uncertainty-exceeds-effect': 'cannot-be-settled',
	'size-pinned-ahead': 'real-but-small',
	// A shortfall the benchmark HAS pinned still fails to show the claim this
	// question asks about, so it takes the strength that says so.
	'size-pinned-behind': 'not-shown'
};

const GENERALITY_DOUBT: Record<GeneralityCase, string> = {
	'uncertainty-exceeds-effect':
		'Nothing on this page narrows it. More labelled statements would, and so would a benchmark ' +
		'drawn after the fact rather than the one this random forest was built from — which is the ' +
		'work, not a caveat.',
	'size-pinned-ahead':
		'The range is pinned on these statements only, and they are the ones the random forest was ' +
		'built from. Nothing here says the same size holds on statements drawn after the fact — which ' +
		'is the work, not a caveat.',
	'size-pinned-behind':
		'The range is pinned on these statements only, and they are the ones the random forest was ' +
		'built from. Nothing here says the same size holds on statements drawn after the fact — which ' +
		'is the work, not a caveat.'
};

function buildGeneralityTier(
	paperRobustness: PaperRobustnessLoad
): Omit<VerdictTierOk, 'rank'> | Omit<VerdictTierUnavailable, 'rank'> {
	const id: VerdictQuestionId = 'how-much-better';
	const question = QUESTIONS[id];
	if (paperRobustness.status !== 'ok') {
		return {
			id,
			status: 'unavailable',
			question,
			reason: paperRobustness.reason,
			plainReason: DARK_PLAIN_REASON[id]
		};
	}
	const figure = paperRobustness.figure;
	const leader = orderingLeaderOf(figure);
	if (leader === null) {
		return {
			id,
			status: 'unavailable',
			question,
			reason: 'the ordering figure carries no model to rank.',
			plainReason:
				'The figure behind this question loaded but names no reading model, so there is nothing ' +
				'to size.'
		};
	}
	const standing = orderingStandingOf(leader);
	if (standing === null) {
		return {
			id,
			status: 'unavailable',
			question,
			reason: 'the ordering margin lies outside the interval shipped around it.',
			plainReason:
				'The margin and the interval printed around it disagree in the shipped file, so this ' +
				'question is left unanswered rather than answered from numbers that contradict each other.'
		};
	}

	const width = fromPts(leader.pointwise.highPts - leader.pointwise.lowPts);
	const effect = Math.abs(leader.primaryDelta);
	// Two facts decide this tier, in this order. The SIGN says whether "how much
	// better" is even the question — a margin lying below zero is not a small
	// improvement, it is a shortfall, and no width test can tell the two apart.
	// The WIDTH then says whether the size is established at all: the boundary is
	// not a chosen threshold but the point where the interval stops being wider
	// than the thing it brackets.
	const generalityCase = generalityCaseOf(standing.pointwise, effect, width);
	const strength = GENERALITY_STRENGTH[generalityCase];

	return {
		id,
		status: 'ok',
		strength,
		strengthWord: VERDICT_STRENGTH_WORD[strength],
		question,
		claim: GENERALITY_CLAIM[generalityCase](figure.primaryPanel.nStatements.toLocaleString()),
		numbers: [
			{
				caption: 'statements in this benchmark',
				value: figure.primaryPanel.nStatements.toLocaleString(),
				note:
					`The set the 2023 INDRA assembly paper released, with its own labels — the only one its ` +
					`authors can check us against. ${figure.primaryPanel.nPositive.toLocaleString()} of them ` +
					`are labelled correct.`
			},
			{
				caption: 'the ordering margin, with its interval',
				value: `${fmtDelta(leader.primaryDelta)}, from ${fmtDelta(fromPts(leader.pointwise.lowPts))} to ${fmtDelta(fromPts(leader.pointwise.highPts))}`,
				// A ratio of two WIDTHS, and the only sign in the row is the one
				// `fmtDelta` prints on the numbers themselves. It reads the same for a
				// margin of +0.0098 and one of −0.0098 because it is the same fact
				// about both; the direction is carried by the claim above it.
				note:
					effect > 0
						? `The interval is ${(width / effect).toFixed(1)} times as wide as the margin it brackets.`
						: 'The margin is zero, so there is no width to compare it against.'
			},
			{
				caption: 'the same margin on our own revision of those labels',
				value: fmtDelta(leader.sensitivityDelta),
				note: `Recheck, never the headline: ${figure.sensitivityPanel.nStatements.toLocaleString()} statements, after dropping the ones whose review never finished.`
			}
		],
		doubt: GENERALITY_DOUBT[generalityCase]
	};
}

// ---------------------------------------------------------------------------
// the block
// ---------------------------------------------------------------------------

function strengthRankOf(tier: Omit<VerdictTierOk, 'rank'> | Omit<VerdictTierUnavailable, 'rank'>) {
	// An unavailable tier sorts last: it makes no claim, so it cannot outrank one.
	return tier.status === 'ok' ? VERDICT_STRENGTH_ORDER[tier.strength] : Number.MAX_SAFE_INTEGER;
}

/**
 * Build the ranked verdict from the loads /paper already performs.
 *
 * The three tiers are built independently and then SORTED by strength, so the
 * order on screen is a fact about the evidence rather than the order they happen
 * to be written in here. `rank` is stamped afterwards, which is why no tier
 * carries one until this point.
 */
export function buildPaperVerdict(input: PaperVerdictInput): PaperVerdictLoad {
	const built = [
		buildErrorFindingTier(input.statementErrorF1, input.reviewQueue),
		buildOrderingTier(input.paperRobustness),
		buildGeneralityTier(input.paperRobustness)
	];
	const ordered = built
		.map((tier, index) => ({ tier, index }))
		.sort((a, b) => strengthRankOf(a.tier) - strengthRankOf(b.tier) || a.index - b.index)
		.map(({ tier }, position): VerdictTier => ({ ...tier, rank: position + 1 }));

	const claiming = ordered.filter((tier) => tier.status === 'ok');
	if (claiming.length === 0) {
		return {
			status: 'unavailable',
			tiers: ordered,
			// Authored here and plain, like every other string a reader meets in this
			// block: it is printed on a page where nothing else worked either, but a
			// reader is still the one reading it.
			reason: 'none of the files behind these three questions loaded; each one says so below.'
		};
	}
	return { status: 'ok', tiers: ordered, reason: null };
}
