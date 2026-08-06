import { budget, fail, record, unit, text, nonNegativeInteger, positiveInteger } from './paper-validate.ts';
/**
 * Typed data contract for AGAINST INDRA'S OWN BELIEF — the replication figure.
 *
 * Source artifact: `data/results/deployed_baseline_replication_20260727/
 * deployed_baseline_replication.json` (schema 2), emitted by
 * `scripts/compute_deployed_baseline_replication.py`.
 *
 * THE CLAIM THE FIGURE MAKES, and the one it does NOT. It does not make the same
 * comparison four times — the comparator differs by panel, and so do the evidence
 * per statement, the curators, the class balance and the join. It makes this
 * claim: on four independently-sourced panels, against the STRONGEST form of
 * INDRA's own belief each panel can source, the reader gate wins every time. The
 * artifact carries both sentences (`claim`, `claim_is_not`) and this module
 * renders the comparator ON every row and the composition UNDER every bar, so a
 * reader can see what differs without being told.
 *
 * TWO INCUMBENT FAMILIES, BOTH INDRA'S. `indra_library_default` is
 * `indra.belief.SimpleScorer` at the priors `indra` bundles — unfitted.
 * `indra_production_served` is the belief INDRA's own export pipeline wrote on
 * the statement — fitted, and demonstrably NOT the library default: it falls
 * below SimpleScorer's hard floor on hundreds of statements. That proof ships as
 * `served_belief_identity` and this module FAILS CLOSED if it ever stops holding,
 * because a family split the data no longer supports must not be drawn.
 *
 * WHAT THIS MODULE MAY NOT DO. No AUROC, delta, interval, count, panel name,
 * comparator name or caveat is hard-coded here. The only constants are the frozen
 * panel and family join keys in their artifact order, the arm keys, the SVG
 * geometry, the character budgets derived from that geometry, and one arithmetic
 * parity tolerance.
 *
 * DISPLAY IS DECOUPLED FROM THE JOIN. `key` fields are frozen artifact join keys
 * and are what everything here matches on; `display` is what the figure renders.
 * Renaming an on-screen string must never move a join, and a renamed join key
 * must gate the figure rather than silently redraw it.
 *
 * FAIL-CLOSED. `validateDeployedBaseline` THROWS on any shape or arithmetic
 * drift, and `buildDeployedBaselineFigure` THROWS on any geometry or label-budget
 * violation. The server loader wraps both in one try/catch and gates the whole
 * panel to `unavailable`. A missing or drifted field never falls back to a
 * default — a figure that draws a default is a figure that lies quietly.
 */

import {
	anchoredShippedProse,
	keyedShippedProse,
	pairShippedProse,
	type AnchoredProse,
	type ShippedProse
} from './paper-literal.ts';

/**
 * Arithmetic parity tolerance for the identities the geometry rests on
 * (delta == gate − incumbent, base rate == n_correct / n). The artifact computes
 * both in float64 from the same values, so this is a float-noise tolerance, not
 * a licence for disagreement.
 */
export const DEPLOYED_BASELINE_PARITY_TOL = 1e-9;

export const DEPLOYED_BASELINE_ARTIFACT_KIND = 'deployed_baseline_replication';
/**
 * 3: every per-statement census carries `n_statements` / `n_single` /
 * `share_single`, and `replication` carries the evidence-regime fold span. The
 * figure REQUIRES those — a mean of 1.25 evidence per statement and "84% of
 * these statements hold exactly one evidence" are different disclosures, and the
 * second is the one that says what the gate was asked to do — so schema 2 no
 * longer validates.
 */
export const DEPLOYED_BASELINE_SCHEMA_VERSION = 3;

/** The one cross-panel metric. AUROC, because the base rates differ. */
export const DEPLOYED_BASELINE_METRIC = 'auroc';
export const DEPLOYED_BASELINE_POSITIVE_CLASS = 'gold-correct';

/**
 * The two families of INDRA belief, by frozen artifact key, in artifact order.
 * The library default is what `indra` computes; the served belief is what
 * INDRA's own pipeline stored. Both are INDRA's, only one is fitted, and the
 * figure must be able to say which one each row was measured against.
 */
export const DEPLOYED_BASELINE_FAMILY_KEYS = {
	library: 'indra_library_default',
	served: 'indra_production_served'
} as const;

/** The two arms that are not incumbents: our gate, and the paper's RF. */
export const DEPLOYED_BASELINE_ARM_KEYS = {
	gate: 'gemma_4_26b_gate',
	research: 'paper_rf_promoter'
} as const;

/**
 * The four panels in the artifact's FIXED order. `validateDeployedBaseline`
 * requires the artifact's panel sequence to equal this key sequence exactly, so
 * a dropped, added or reordered panel is a visible failure rather than a
 * different figure.
 */
export const DEPLOYED_BASELINE_PANEL_KEYS: readonly string[] = [
	'indra_paper_2023',
	'eval_curation_v1',
	'external_curator_v1',
	'holdout_cc'
] as const;

export const DEPLOYED_BASELINE_PAPER_PANEL_KEY = 'indra_paper_2023';

/** The artifact's caveat list travels with the figure; nothing may be dropped. */
export const DEPLOYED_BASELINE_CAVEAT_COUNT = 8;

/**
 * THE PLAIN HALF OF EVERY TWIN THIS MODULE EMITS.
 *
 * `deployed_baseline_replication.json` is the artifact that says "incumbent" —
 * the one word on this page that names nothing at all to a reader who has not
 * been in the room. It says it in the claim, in the selection rule, in the cost
 * of that rule and in the first caveat, all at runtime, none of it visible to a
 * scan of this repo. The restatements below name the thing instead: the scorer
 * INDRA ships today, in the two forms it actually comes in.
 *
 * Nothing is softened. The evidence asymmetry that runs AGAINST us, the margin
 * this figure forfeits by always taking the strongest comparison, and the flat
 * statement that this is a detector and not a ranker all survive at full
 * strength.
 */
const DEPLOYED_BASELINE_PLAIN = {
	claim:
		'The reading model beats every version of INDRA’s own belief score we can obtain, on all ' +
		'4 independently sourced sets of statements — 7 versions in all, and 7 of 7 with a paired ' +
		'95% interval that stays clear of zero — across evidence supplies that differ by a factor ' +
		'of 16 (1.2 to 19.8 pieces of evidence read per statement), under a rule that always takes ' +
		'the ' +
		'STRONGEST version each set can supply and gives up as much as 0.0532 AUROC of margin to ' +
		'do it.',
	claimIsNot:
		'This is not the same comparison four times. What the reading model is measured against ' +
		'differs by set — 2 sets can supply both forms of INDRA’s own belief, 2 can supply only ' +
		'the belief INDRA stored — and so do the evidence per statement, the curators, the mix of ' +
		'right and wrong, and the join. Statements with a single piece of evidence run from 20% of ' +
		'the benchmark published in 2023 to 84% of the 32-curator set, so on some rows the reading ' +
		'model is combining several readings and on others it is a bare keep-or-drop on one ' +
		'sentence. Every row names what it was measured against and prints its own make-up.',
	incumbentSelectionRule:
		'Every form of INDRA’s own belief score that can be obtained is scored on each set of ' +
		'statements — the library’s default scorer and the belief INDRA actually stored, alike — ' +
		'and the one carried into the headline is whichever has the HIGHEST AUROC on that set. ' +
		'The reading model is therefore measured against the strongest form of the scorer INDRA ' +
		'ships that each set can produce, never the most convenient one.',
	incumbentSelectionRuleCost:
		'Always taking the strongest form gives up margin, and the figure prints how much on every ' +
		'row. The largest amount given up is 0.0532 AUROC.',
	/**
	 * THE SVG `<desc>` — the accessible description — is assembled from the fields
	 * below. Before these twins existed a screen-reader user got the shipped
	 * wording raw, behind no boundary at all, which made this the widest untranslated
	 * surface on the page rather than the narrowest.
	 */
	question:
		'Does a reading model beat INDRA’s own belief — both the unfitted SimpleScorer the library ' +
		'ships and the fitted score INDRA’s pipeline actually stores — and does the answer hold up ' +
		'on more than one set of statements?',
	metricSource:
		'AUROC computed from rank sums, with tied scores given their average rank (checked against ' +
		'scikit-learn’s roc_auc_score).',
	noisyOrFormula:
		'One minus the product, over every source, of that source’s systematic error rate plus its ' +
		'random error rate raised to the number of evidence entries that source supplied.',
	gateWhatItIs:
		'INDRA’s own library-default combination rule over only the evidence the reading model ' +
		'kept; it can only take belief away, no cutoff is fitted, and nothing is trained.',
	gateNotZeroShot: '14 hand-written example pairs go out with every call; nothing is calibrated.',
	researchWhatItIs:
		'The supervised random forest released with the 2023 paper, with every statement scored by ' +
		'a copy that never saw it. A research model: it is not in indra and has never been served.',
	incumbentSelectedBy:
		'whichever form of INDRA’s own belief has the HIGHEST AUROC on these statements',
	servedIdentityQuestion:
		'Is the belief INDRA stored on the statement the same thing as indra.belief.SimpleScorer at ' +
		'the bundled default reliabilities?',
	servedIdentityFinding:
		'No. SimpleScorer at those reliabilities cannot score below 0.60, and passing belief up the ' +
		'hierarchy only raises it, yet 359 stored beliefs across the curated sets fall below that ' +
		'floor.',
	floorDerivation:
		'Belief is one minus a product of per-source terms; every term is at most that source’s ' +
		'systematic plus random error rate, so belief can never fall below one minus the largest ' +
		'such sum. The worst source in the bundled file is gnbr.'
} as const;

/**
 * The two families of INDRA belief, keyed by their FROZEN artifact key. Two
 * fields each, and both of them are the bulk of what the legend says a reader.
 */
const DEPLOYED_BASELINE_FAMILY_TWINS: Readonly<
	Record<string, { whatItComputes: AnchoredProse; whereItRuns: AnchoredProse }>
> = {
	indra_library_default: {
		whatItComputes: {
			artifactAnchor: 'Nothing about it is fitted',
			plain:
				'indra.belief.SimpleScorer at the per-source reliabilities bundled in ' +
				'indra/resources/default_belief_probs.json: one minus the product, over every source, ' +
				'of that source’s systematic error rate plus its random error rate raised to the number ' +
				'of evidence entries that source supplied. Nothing about it is fitted.'
		},
		whereItRuns: {
			artifactAnchor: 'installs indra and calls BeliefEngine',
			plain:
				'what anyone who installs indra and calls BeliefEngine gets, and what the 2023 paper ' +
				'calls the unfitted noisy-OR.'
		}
	},
	indra_production_served: {
		whatItComputes: {
			artifactAnchor: 'we do not re-derive it',
			plain:
				'the number INDRA’s own export and assembly pipeline computed and wrote onto the ' +
				'statement — the fitted HybridScorer path the CoGEx export uses, NOT the library ' +
				'default. We read it; we do not re-derive it.'
		},
		whereItRuns: {
			artifactAnchor: 'reads off the statement today',
			plain: 'what a db.indra.bio or CoGEx user reads off the statement today.'
		}
	}
};

/**
 * Each sourceable form of INDRA's belief, keyed by its FROZEN variant key. One
 * key here addresses up to three rows shipping identical text, so the anchor is
 * what binds the restatement to the sentence rather than the key alone.
 */
const DEPLOYED_BASELINE_VARIANT_TWINS: Readonly<Record<string, AnchoredProse>> = {
	simple_scorer_direct: {
		artifactAnchor: 'over every direct evidence entry',
		plain:
			'indra.belief.SimpleScorer 1.24.0 at the bundled default reliabilities, run on the ' +
			'object graph released in 2023 over every direct evidence entry (34,035 across the ' +
			'1,689 statements).'
	},
	simple_scorer_hierarchy: {
		artifactAnchor: 'inherits evidence from its more specific descendants',
		plain:
			'The same scorer run through indra.belief.BeliefEngine.get_hierarchy_probs over ' +
			'indra.belief.build_refinements_graph, so a statement also inherits evidence from the ' +
			'more specific statements underneath it. 477 of the 1,689 change.'
	},
	cogex_fitted_hybrid: {
		artifactAnchor: 'not a live deployment capture',
		plain:
			'HybridScorer — a fitted CountsScorer file with INDRA SimpleScorer as the fallback — run ' +
			'through the recorded indra_db export-assembly adapter; both scored sets take the ' +
			'counts-only route. It is an offline replay under Python 3.12.10 and INDRA 1.24.0, with ' +
			'sklearn 1.3.2 prediction behaviour reproduced exactly under sklearn 1.4.1.post1, and NOT ' +
			'a live production capture. It is the SAME family as the belief stored on the curated ' +
			'statements, replayed on the 1,689 statements published in 2023.'
	},
	simple_scorer_recomputed: {
		artifactAnchor: 'not just the curated rows',
		plain:
			'The bundled-reliability combination rule recomputed from the statement’s own INDRA ' +
			'source counts — the statement’s FULL database evidence, not just the curated rows. ' +
			'Reliabilities read verbatim from indra/resources/default_belief_probs.json (sha256 ' +
			'6c26f48e0a9a).'
	},
	indra_served_belief: {
		artifactAnchor: 'read off the statement and not re-derived',
		plain:
			'The belief INDRA’s own pipeline wrote onto the statement, read off the statement and not ' +
			're-derived. It is NOT the library-default SimpleScorer — the identity check below shows ' +
			'why.'
	}
};

/**
 * Everything that varies BY SET OF STATEMENTS, keyed by the frozen panel key:
 * who curated it, what its labels mean, how it was joined, and where a count
 * could not be taken. `labelNote` and `joinMode` repeat verbatim across three
 * sets, and they are written out three times on purpose — a shared entry would
 * make a reissued artifact that changes one of them silently reuse the others.
 */
interface DeployedBaselinePanelTwins {
	curatorNote: AnchoredProse;
	labelNote: AnchoredProse;
	joinMode: AnchoredProse;
	corpusEvidenceAbsentBecause: AnchoredProse | null;
	inSampleNote: AnchoredProse | null;
}

const CURATED_LABEL_NOTE: AnchoredProse = {
	artifactAnchor: 'every curated evidence pair is correct',
	plain:
		'a statement counts as correct only when every curated piece of evidence for it is correct'
};
const CURATED_JOIN_MODE: AnchoredProse = {
	artifactAnchor: 'truth-safe source fallback',
	plain:
		'each row matched exactly on its (matches_hash, source_hash) pair where possible, falling ' +
		'back to a source-only match that cannot leak the answer'
};

const DEPLOYED_BASELINE_PANEL_TWINS: Readonly<Record<string, DeployedBaselinePanelTwins>> = {
	indra_paper_2023: {
		curatorNote: {
			artifactAnchor: 'own curators',
			plain: 'the curators who labelled them for the 2023 paper'
		},
		labelNote: {
			artifactAnchor: 'we produced none of them',
			plain: 'the curation labels released with the 2023 paper; none of them were produced here'
		},
		joinMode: {
			artifactAnchor: 'sorted(statement_id)',
			plain:
				'statements matched by their sorted statement ids over the released-label set, using ' +
				'the same loader as the review-list figure (compute_statement_review_queue.load_panel)'
		},
		corpusEvidenceAbsentBecause: null,
		inSampleNote: null
	},
	eval_curation_v1: {
		curatorNote: {
			artifactAnchor: 'own authors',
			plain: 'curated by two authors of the 2023 paper'
		},
		labelNote: CURATED_LABEL_NOTE,
		joinMode: CURATED_JOIN_MODE,
		corpusEvidenceAbsentBecause: null,
		inSampleNote: {
			artifactAnchor: 'SOFT calibration profile was fitted on',
			plain:
				'This corpus is what the SOFT calibration profile was fitted on. What is drawn here ' +
				'uses INDRA’s default reliabilities and no fitted parameter, so it is still outside the ' +
				'data any fitted model here learned from — but the flag travels with these statements ' +
				'anyway.'
		}
	},
	external_curator_v1: {
		curatorNote: {
			artifactAnchor: 'none of them ours',
			plain: '32 curators from the public INDRA curation database, none of them on this project'
		},
		labelNote: CURATED_LABEL_NOTE,
		joinMode: CURATED_JOIN_MODE,
		corpusEvidenceAbsentBecause: {
			artifactAnchor: 'carries no per-source evidence counts',
			plain:
				'The gold for these 32-curator statements carries no per-source evidence counts, so the ' +
				'statement’s total corpus evidence cannot be counted here. What is drawn on them is the ' +
				'belief INDRA stored, which was computed over that full evidence set whatever its size.'
		},
		inSampleNote: null
	},
	holdout_cc: {
		curatorNote: {
			artifactAnchor: 'calibration ship gate',
			plain: 'the out-of-distribution holdout the calibration ship check is tested against'
		},
		labelNote: CURATED_LABEL_NOTE,
		joinMode: CURATED_JOIN_MODE,
		corpusEvidenceAbsentBecause: {
			artifactAnchor: 'carries no per-source evidence counts',
			plain:
				'The gold for these holdout statements carries no per-source evidence counts, so the ' +
				'statement’s total corpus evidence cannot be counted here. What is drawn on them is the ' +
				'belief INDRA stored, which was computed over that full evidence set whatever its size.'
		},
		inSampleNote: null
	}
};

/**
 * THE ON-SCREEN NAME OF EACH SET OF STATEMENTS, keyed by its FROZEN panel key.
 *
 * The artifact ships its own `display` for each, and the loader used to pass it
 * straight through — which put "the 2023 paper's own panel" on the row label, in
 * the row title and inside the SVG `<desc>`, in the artifact's dialect, behind
 * no boundary. `key` is what everything joins on and must not move; the name is
 * decided HERE, exactly as `BELIEF_LADDER_ENTRY_SPECS` decides the ladder's.
 *
 * `displayProse.shipped` keeps the artifact's own wording for the audit trail,
 * so nothing is lost — it is simply not what a reader is handed.
 */
export const DEPLOYED_BASELINE_PANEL_DISPLAY: Readonly<Record<string, string>> = {
	indra_paper_2023: 'the benchmark published in 2023',
	eval_curation_v1: 'two authors of the 2023 paper',
	external_curator_v1: '32 external curators',
	holdout_cc: 'out-of-distribution holdout'
};

/** The one form whose source file qualifies its own admissibility. */
const DEPLOYED_BASELINE_PROVENANCE_CAVEAT_TWIN: AnchoredProse = {
	artifactAnchor: 'compatibility-recovered replay',
	plain: 'CoGEx fitted Hybrid file — recovered for compatibility and replayed offline'
};

/** Why an already-shipped sibling lands somewhere else on the same statements. */
const DEPLOYED_BASELINE_CROSS_CHECK_TWIN: AnchoredProse = {
	artifactAnchor: 'an 18-source transcription of INDRA',
	plain:
		'A sibling result scores these same statements with src/indra_belief/noise_model.py’s ' +
		'INDRA_PRIORS — an 18-source transcription of INDRA’s reliabilities with a (0.30, 0.10) ' +
		'fallback for everything else. That module is byte-frozen under the reading bundle’s ' +
		'implementation digest and must not change, so this file reads ' +
		'indra/resources/default_belief_probs.json itself instead: 20+ sources, and rlimsp, hprd, ' +
		'tas and drugbank differ. 199 of these statements score differently. The value drawn is the ' +
		'STRONGER of the two, which is the direction the strongest-comparison rule requires.'
};

/** INDRA's own scorer held to exactly the evidence the reading model saw. */
const DEPLOYED_BASELINE_CONTROL_NOTE_TWIN: AnchoredProse = {
	artifactAnchor: 'INDRA never serves this number',
	plain:
		'INDRA’s library-default combination rule over exactly the evidence the reading model was ' +
		'shown, with no reading step applied. It is not one of the comparisons: INDRA never serves ' +
		'this number. It prices the difference in scope — the comparison above is scored over the ' +
		'statement’s full database evidence, the reading model over the curated subset only.'
};

/** Why that control does not exist on the 2023 paper's own statements. */
const DEPLOYED_BASELINE_CONTROL_ABSENT_TWIN: AnchoredProse = {
	artifactAnchor: 'does not arise',
	plain:
		'On the benchmark published in 2023 the comparison and the reading model read the SAME ' +
		'corpus — the assembled evidence released with it, 19.75 unique pairs per statement — so ' +
		'the difference in scope this control exists to price does not arise. What the reading model ' +
		'additionally drops without reading (duplicates, evidence with no sentence, deterministic ' +
		'grounding rejects) is priced separately by ' +
		'data/results/indra_paper_literal_models_20260724/non_reading_control.json.'
};

/** What the reading step is, and what it is not. */
const DEPLOYED_BASELINE_GATE_WHAT_IT_IS_TWIN: AnchoredProse = {
	artifactAnchor: 'purely subtractive',
	plain: DEPLOYED_BASELINE_PLAIN.gateWhatItIs
};

/** What the 2023 paper's own fitted model is, and where it never ran. */
const DEPLOYED_BASELINE_RESEARCH_WHAT_IT_IS_TWIN: AnchoredProse = {
	artifactAnchor: 'has never been served',
	plain: DEPLOYED_BASELINE_PLAIN.researchWhatItIs
};

/** How the drawn comparison is chosen on each set of statements. */
const DEPLOYED_BASELINE_SELECTED_BY_TWIN: AnchoredProse = {
	artifactAnchor: 'argmax auroc',
	plain: DEPLOYED_BASELINE_PLAIN.incumbentSelectedBy
};

/** The production reading step, carried but never drawn. */
const DEPLOYED_BASELINE_GATE_SENSITIVITY_TWIN: AnchoredProse = {
	artifactAnchor: 'Drawn nowhere',
	plain:
		'The production reading step, which also swaps INDRA’s reliabilities for the recalibrated ' +
		'ones. Drawn nowhere; carried so the cost of swapping them can be priced.'
};

/**
 * The three evidence supplies a score can be computed over, keyed by the frozen
 * `evidence_scope` id the artifact stamps beside each note. Selected by that id
 * rather than by matching the note's text, so a reworded note still gates on an
 * id nobody recognises instead of silently losing its restatement.
 */
const DEPLOYED_BASELINE_SCOPE_PLAIN: Readonly<Record<string, string>> = {
	statement_full: 'every piece of evidence the source database holds for this statement',
	statement_full_plus_hierarchy:
		'every piece of evidence the source database holds for this statement, plus evidence ' +
		'inherited from the more specific statements underneath it',
	reader_evidence_only: 'only the evidence the reading model was actually shown'
};

/** `caveats[]` in shipped order, pinned to each sentence by a verbatim fragment. */
const DEPLOYED_BASELINE_CAVEAT_TWINS: readonly AnchoredProse[] = [
	{
		artifactAnchor: 'There are TWO incumbents',
		plain:
			'There are TWO scorers here and both are INDRA’s own. The library default is the ' +
			'unfitted noisy-OR SimpleScorer at the priors the indra package bundles. The stored ' +
			'belief is what INDRA’s export pipeline wrote onto the statement, and it is NOT that ' +
			'scorer: it falls below SimpleScorer’s hard floor on thousands of statements here, ' +
			'which is recomputed on every build.'
	},
	{
		artifactAnchor: 'compatibility-recovered offline replay',
		plain:
			'The CoGEx hybrid drawn on the benchmark published in 2023 is a compatibility-recovered ' +
			'offline replay of the file the CoGEx export path refers to — not a live production ' +
			'capture, and what it was trained on is not established. If it was fitted on curations ' +
			'that overlap these statements then it is flattered here, which makes it a HARDER thing ' +
			'to beat, not an easier one.'
	},
	{
		artifactAnchor: 'Belief Orig',
		plain:
			'The unfitted SimpleScorer here is NOT the “Belief Orig” row of the 2023 table. “Belief ' +
			'Orig” refits how reliable each source is fold by fold — the 10 groups the statements ' +
			'were split into — by MCMC; this one is the shipped default-prior scorer, unfitted ' +
			'anywhere. No Bayesian, subtype or hierarchy model was published in 2023 at all.'
	},
	{
		artifactAnchor: 'NOT zero-shot',
		plain:
			'The reading models are NOT zero-shot: each call carries 14 hand-written example pairs. ' +
			'They are also not calibrated — this is the hard reading step, which can only take ' +
			'evidence away.'
	},
	{
		artifactAnchor: 'AUROC is the cross-panel metric',
		plain:
			'AUROC is the measure used across sets because the sets differ in how often a statement ' +
			'is right — 0.45 on the out-of-distribution holdout, 0.50 on the 32 external curators ' +
			'(balanced by construction), 0.52 on two authors of the 2023 paper (balanced by ' +
			'construction), 0.73 on the benchmark published in 2023. Average precision moves with that ' +
			'rate, so it is reported per set and must not be compared across sets.'
	},
	{
		artifactAnchor: 'FULL database evidence',
		plain:
			'On the 3 curation sets, INDRA’s own scorer is scored over the statement’s FULL ' +
			'database evidence while the reading model sees only the curated subset — 1.21 to 1.76 ' +
			'pieces of evidence per statement, and 62% to 84% of those statements carry exactly ' +
			'one, against 19.75 and 20% on the benchmark published in 2023. Evidence density between ' +
			'the two extremes therefore differs by a factor of 16.4. INDRA’s own scorer restricted ' +
			'to the reading model’s own evidence lands at or below chance on 2 of those 3 sets, so ' +
			'that asymmetry runs AGAINST the reading model, not for it.'
	},
	{
		artifactAnchor: 'replication of a DIRECTION',
		plain:
			'Four sets of statements is replication of a DIRECTION, not a meta-analysis: they ' +
			'differ in curator, corpus, sampling and how the reading model was deployed, and each ' +
			'interval belongs to its own set — none of them are combined into one.'
	},
	{
		artifactAnchor: 'DETECTOR, not a ranker',
		plain:
			'The reading step is a DETECTOR, not a ranker. Among the statements it does not zero ' +
			'out, it orders them WORSE than the fitted random forest. Its advantage comes from the ' +
			'statements it zeroes, which is what the operational review-list result measures ' +
			'directly.'
	}
];

/**
 * The verbatim fragment each caveat must still contain, in shipped order.
 *
 * EXPORTED because it is part of the contract, not an implementation detail: a
 * caveat this figure cannot restate is a caveat it may not print, and the
 * contract runner builds artifacts that have to satisfy that. Deriving it from
 * the twins means the runner cannot drift from the loader.
 */
export const DEPLOYED_BASELINE_CAVEAT_ANCHORS: readonly string[] =
	DEPLOYED_BASELINE_CAVEAT_TWINS.map((twin) => twin.artifactAnchor);

/**
 * The SAME contract, for every other sentence this figure restates. DERIVED from
 * the twins above, so the runner that builds a synthetic artifact cannot drift
 * from the loader that reads the real one: a sentence this figure cannot restate
 * is a sentence it may not print, and both sides now read that from one place.
 *
 * There is no `plain` here on purpose. A restatement is authored beside the
 * parse that owns it; what the contract needs to know is only WHICH sentence
 * each one was written for.
 */
export const DEPLOYED_BASELINE_PROSE_ANCHORS = {
	family: Object.fromEntries(
		Object.entries(DEPLOYED_BASELINE_FAMILY_TWINS).map(([key, twins]) => [
			key,
			{
				what_it_computes: twins.whatItComputes.artifactAnchor,
				where_it_runs: twins.whereItRuns.artifactAnchor
			}
		])
	),
	variant: Object.fromEntries(
		Object.entries(DEPLOYED_BASELINE_VARIANT_TWINS).map(([key, twin]) => [
			key,
			twin.artifactAnchor
		])
	),
	panel: Object.fromEntries(
		Object.entries(DEPLOYED_BASELINE_PANEL_TWINS).map(([key, twins]) => [
			key,
			{
				curator_note: twins.curatorNote.artifactAnchor,
				label_note: twins.labelNote.artifactAnchor,
				join_mode: twins.joinMode.artifactAnchor,
				corpus_evidence_absent_because:
					twins.corpusEvidenceAbsentBecause?.artifactAnchor ?? null,
				in_sample_note: twins.inSampleNote?.artifactAnchor ?? null
			}
		])
	),
	gate_what_it_is: DEPLOYED_BASELINE_GATE_WHAT_IT_IS_TWIN.artifactAnchor,
	research_what_it_is: DEPLOYED_BASELINE_RESEARCH_WHAT_IT_IS_TWIN.artifactAnchor,
	selected_by: DEPLOYED_BASELINE_SELECTED_BY_TWIN.artifactAnchor,
	provenance_caveat: DEPLOYED_BASELINE_PROVENANCE_CAVEAT_TWIN.artifactAnchor,
	why_it_differs: DEPLOYED_BASELINE_CROSS_CHECK_TWIN.artifactAnchor,
	control_note: DEPLOYED_BASELINE_CONTROL_NOTE_TWIN.artifactAnchor,
	control_absent_because: DEPLOYED_BASELINE_CONTROL_ABSENT_TWIN.artifactAnchor,
	gate_sensitivity_note: DEPLOYED_BASELINE_GATE_SENSITIVITY_TWIN.artifactAnchor
} as const;

/**
 * SVG geometry, exported so the character budgets below are DERIVED from it
 * rather than eyeballed. Right-anchored SVG text that overruns its gutter loses
 * its LEADING glyphs silently — no layout error, no test failure, and `<desc>`
 * hides the damage from review — so every gutter is budgeted in characters and
 * enforced in the builder, exactly as `./paper-own-metric.ts` does.
 */
export const DEPLOYED_BASELINE_GEOMETRY = {
	width: 920,
	/** Row labels, sub-labels and comparators are right-anchored here: 0 → 238. */
	labelAnchorX: 238,
	/**
	 * The axis title is right-anchored HERE, not at `labelAnchorX`: the first
	 * tick label is CENTRED on `plotLeft`, so at 8px it reaches back to ~237 and
	 * an axis title ending at 238 collides with it. 214 leaves ~23 units of gap.
	 */
	axisTitleX: 214,
	plotLeft: 250,
	plotRight: 800,
	/** Readouts are left-anchored here; usable gutter is 812 → 920 = 108 units. */
	readoutX: 812,
	labelFontPx: 9,
	subLabelFontPx: 8,
	comparatorFontPx: 8,
	chipFontPx: 7.5,
	/** The composition strip under each bar, left-anchored at `plotLeft`. */
	heteroFontPx: 7.5,
	readoutFontPx: 9,
	/**
	 * Measured advance of the mono face at 9px, in user units per character —
	 * the same measurement `PAPER_OWN_METRIC_GEOMETRY.monoUnitsPerChar` carries.
	 * The others scale linearly: × 8/9 = 4.8165, × 7.5/9 = 4.5155.
	 */
	monoUnitsPerChar: 5.4186,
	subMonoUnitsPerChar: 4.8165,
	chipMonoUnitsPerChar: 4.5155,
	/** The 8.5px header/legend face: 5.4186 × 8.5/9. */
	headerMonoUnitsPerChar: 5.1176,
	/**
	 * LETTER-SPACED ADVANCES. Two classes in the component carry `letter-spacing`,
	 * and CSS adds that space after EVERY character, so their real advance is the
	 * face's advance plus the tracking. Measured in a browser against the rendered
	 * figure: `.fig-subtitle` came back at 5.456 u/char and `.row-chip` at 5.114,
	 * not the 5.1176 and 4.5155 their budgets originally assumed — a 6.6% and a
	 * 13.3% under-measurement respectively. Nothing shipped clipped, because the
	 * longest strings were well inside both gutters, but a full-budget header line
	 * would have run 24 units INTO the legend mark and a full-budget chip 28 units
	 * past the left edge of its gutter. The budgets below are derived from these.
	 *
	 * `.fig-subtitle` is 8.5px with letter-spacing 0.04em: 5.1176 + 0.04 × 8.5.
	 */
	headerLetterSpacedUnitsPerChar: 5.4576,
	/** `.row-chip` is 7.5px with letter-spacing 0.08em: 4.5155 + 0.08 × 7.5. */
	chipLetterSpacedUnitsPerChar: 5.1155,
	/**
	 * A CONSERVATIVE UPPER BOUND on the 15px serif title's average advance, not a
	 * measurement: 7.6 units/char is 0.507em, above the ~0.48em a text serif
	 * averages. The title budget is therefore safe rather than tight, which is the
	 * right side to err on for the one string that is not monospaced.
	 */
	titleSerifUnitsPerChar: 7.6,
	titleFontPx: 15,
	titleX: 12,
	headerX: 12,
	legendY: 18,
	legendRowHeight: 13,
	/**
	 * The legend sits in its OWN horizontal band, right of the header block. It
	 * used to start at `plotLeft` (250), where its third row overlapped the second
	 * header line at 8.5px — a silent collision no character budget could see
	 * because the two strings were budgeted against different gutters and neither
	 * knew about the other. Moving the legend to 430 gives the header 402 units
	 * and the legend 466, and both are budgeted below.
	 */
	legendMarkX: 430,
	legendTextX: 442,
	/**
	 * The header band's third line: the evidence-regime span. It sits BELOW the
	 * legend's last baseline (18 + 3 × 13 = 57) and above the first row, and it
	 * is left-anchored at `headerX` so it shares the header's budget.
	 */
	regimeLineY: 68,
	rowsTop: 80,
	rowHeight: 78,
	/** Row-local vertical offsets, all measured from the row's own top. */
	labelDy: 16,
	subLabelDy: 27,
	comparatorDy: 38,
	chipDy: 49,
	trackDy: 22,
	/**
	 * The composition strip. It used to sit at 40 — two units under the readout
	 * column's last line at `ruleCostDy` 38 — so the two shared a vertical band
	 * and the strip's gutter had to stop short of `readoutX`, which capped it at
	 * 121 characters and left no room for the single-evidence share. At 52 the
	 * readout column is finished, so the strip runs the full content width and
	 * the panel's composition fits without abbreviating itself into a code.
	 * Still 10 units clear of `separatorDy` 62.
	 */
	heteroDy: 52,
	deltaDy: 16,
	ciDy: 27,
	ruleCostDy: 38,
	separatorDy: 62,
	trackHeight: 9,
	markRadius: 4.4,
	/** Half-height of the tick that marks a variant the argmax declined. */
	declinedHalfHeight: 5.5,
	axisPad: 40
} as const;

/**
 * LABEL BUDGET. 238 units ÷ 5.4186 u/char at 9px = 43.9 → 43 characters for the
 * right-anchored row label. The longest panel display the shipped artifact
 * produces is "out-of-distribution holdout" at 27 characters (146.3 units).
 * `buildDeployedBaselineFigure` FAILS if any label exceeds the budget, so a
 * longer panel name gates the figure to `unavailable` instead of quietly eating
 * its first glyphs.
 */
export const DEPLOYED_BASELINE_LABEL_BUDGET_CHARS = 43;

/**
 * SUB-LABEL BUDGET. Same gutter, 8px: 238 ÷ 4.8165 = 49.4 → 49 characters. The
 * longest the shipped artifact produces is the paper panel's
 * "1689 statements · 1237/452 correct/error" at 40. Right-anchored, so an
 * overrun would eat its leading glyphs too.
 */
export const DEPLOYED_BASELINE_SUBLABEL_BUDGET_CHARS = 49;

/**
 * COMPARATOR BUDGET. The same 238-unit gutter at 8px, so the same 49 characters.
 * This line is the fix for the defect that made the old figure read as "the same
 * comparison four times": every row now NAMES the form of INDRA's belief it was
 * measured against, in the gutter, not in a method note.
 */
export const DEPLOYED_BASELINE_COMPARATOR_BUDGET_CHARS = 49;

/**
 * CHIP BUDGET. Same 238-unit gutter at 7.5px, but `.row-chip` carries
 * `letter-spacing: 0.08em`, so its real advance is 5.1155 u/char, not the face's
 * 4.5155: 238 ÷ 5.1155 = 46.5 → 46 characters. The earlier 52 was measured
 * against the untracked face and would have licensed a chip 28 units wider than
 * its gutter — right-anchored, so it would have eaten its own leading glyphs.
 */
export const DEPLOYED_BASELINE_CHIP_BUDGET_CHARS = 46;

/**
 * COMPOSITION-STRIP BUDGET. Left-anchored at `plotLeft` (250) on its own
 * baseline (`heteroDy` 52), where the readout column has already finished at
 * `ruleCostDy` 38. It therefore runs to the content right edge — the separator's
 * own end, `width` (920) less the 12-unit margin — rather than stopping at
 * `readoutX`: (920 − 12 − 250) = 658 units ÷ 4.5155 u/char at 7.5px = 145.7 →
 * 145 characters. Left-anchored, so an overrun clips the TRAILING glyphs — the
 * composition would lose its join mode first, which is exactly the field a
 * reviewer wants. Budgeted and enforced all the same.
 *
 * The 121-character version of this budget (strip at 40, sharing a band with the
 * readouts) is why the strip could not carry the single-evidence share without
 * abbreviating "read/statement" into something a reader has to decode. The
 * budget moved because the geometry did, not the other way round.
 */
export const DEPLOYED_BASELINE_HETERO_BUDGET_CHARS = 145;

/**
 * READOUT BUDGET. (920 − 812) = 108 units ÷ 5.4186 u/char at 9px = 19.9 → 19
 * characters. The longest readout is a signed interval, "[+0.054, +0.095]" at 16
 * characters. Left-anchored, so an overrun clips the TRAILING glyphs; budgeted
 * and enforced all the same.
 */
export const DEPLOYED_BASELINE_READOUT_BUDGET_CHARS = 19;

/**
 * AXIS-TITLE BUDGET. 214 units ÷ 5.4186 u/char at 9px = 39.4 → 39 characters,
 * right-anchored at `axisTitleX`.
 */
export const DEPLOYED_BASELINE_AXIS_TITLE_BUDGET_CHARS = 39;

/**
 * LEGEND BUDGET. Left-anchored at `legendTextX` (442) and it may run to the
 * right edge less a 12-unit margin: (920 − 12 − 442) = 466 units ÷ 5.1176 u/char
 * at 8.5px = 91.1 → 91 characters. The longest legend line ships at 58.
 */
export const DEPLOYED_BASELINE_LEGEND_BUDGET_CHARS = 91;

/**
 * HEADER BUDGET. The two count lines are left-anchored at `headerX` (12) and,
 * because they sit at y 40 and 52, they share a horizontal band with the
 * legend's lower rows — so they must stop before the legend's leftmost ink,
 * `legendMarkX` less the mark's own half-width: (430 − 4.4 − 12 − 12) = 401.6
 * units. `.fig-subtitle` is letter-spaced, so the advance is 5.4576 u/char, not
 * 5.1176: 401.6 ÷ 5.4576 = 73.6 → 73 characters. The longest ships at 71.
 */
export const DEPLOYED_BASELINE_HEADER_BUDGET_CHARS = 73;

/**
 * REGIME-LINE BUDGET. The third header line sits at `regimeLineY` (68), BELOW
 * the legend's last baseline (57) and above the first row (80), so it is the one
 * header line with no legend beside it and it runs to the content right edge —
 * the separator's own end, `width` less the 12-unit margin: (920 − 12 − 12) =
 * 896 units ÷ 5.4576 u/char = 164.1 → 164 characters. It gets its own budget
 * rather than the header's 73 because it carries five formatted numbers, and a
 * three-digit fold span or density must widen the line, not gate the figure.
 */
export const DEPLOYED_BASELINE_REGIME_BUDGET_CHARS = 164;

/**
 * TITLE BUDGET. The one non-monospaced string in the figure, left-anchored at
 * `titleX` (12) at 15px, sharing the header band with the legend: 401.6 units ÷
 * 7.6 u/char (the conservative serif upper bound above) = 52.8 → 52 characters.
 * The title ships at 26.
 */
export const DEPLOYED_BASELINE_TITLE_BUDGET_CHARS = 52;

/**
 * The readout column's own heading, in the SAME 108-unit gutter the readouts
 * use, so it gets the SAME budget. It ships as two lines because one line of it
 * does not fit: the old "gate − deployed, 95% CI" was 23 characters (125 units)
 * and was clipping its trailing "CI" silently — the exact failure this file's
 * budgets exist to catch, caught only because the figure was rendered and looked
 * at. Split, both lines clear the budget with room.
 */
export const DEPLOYED_BASELINE_READOUT_TITLE: readonly [string, string] = [
	'gate − INDRA',
	'95% CI · rule cost'
] as const;

/**
 * The axis floor. AUROC has a meaningful zero — 0.5 is chance — so the axis is
 * anchored there and never truncated to the occupied band. A reader must be able
 * to see how much of the incumbent's score is just the base rate.
 */
export const DEPLOYED_BASELINE_AXIS_MIN = 0.5;

/**
 * Mark shape per drawn series, so the figure survives greyscale and
 * colour-vision deficiency without relying on hue. Stroke tokens are CSS custom
 * properties, never raw hex, and every one of them clears 3:1 against
 * `--paper` (#fdfcf8): --ink-muted 5.1:1, --accent 10.9:1, --blocked 6.5:1,
 * --ink-faint 4.6:1.
 */
export const DEPLOYED_BASELINE_SERIES = {
	incumbent: { shape: 'open-square', strokeVar: 'var(--ink-muted)', dash: 'none' },
	gate: { shape: 'circle', strokeVar: 'var(--accent)', dash: 'none' },
	research: { shape: 'diamond', strokeVar: 'var(--blocked)', dash: 'none' },
	declined: { shape: 'tick', strokeVar: 'var(--ink-faint)', dash: '3 2' }
} as const;

export type DeployedBaselineSeriesKey = keyof typeof DEPLOYED_BASELINE_SERIES;

// ---------------------------------------------------------------------------
// shape
// ---------------------------------------------------------------------------

export interface DeployedBaselineFamily {
	key: string;
	display: string;
	deployed: boolean;
	fitted: boolean;
	whatItComputes: string;
	/** `whatItComputes` with its plain restatement — `shipped` is byte-identical. */
	whatItComputesProse: ShippedProse;
	whereItRuns: string;
	/** `whereItRuns` with its plain restatement — `shipped` is byte-identical. */
	whereItRunsProse: ShippedProse;
	shipsIn: string;
}

export interface DeployedBaselineArm {
	key: string;
	display: string;
	deployed: boolean;
	fitted: boolean;
	whatItIs: string;
	/** `whatItIs` with its plain restatement — `shipped` is byte-identical. */
	whatItIsProse: ShippedProse;
}

export interface DeployedBaselineArms {
	gate: DeployedBaselineArm & {
		notZeroShot: string;
		/** `notZeroShot` with its plain restatement — `shipped` is byte-identical. */
		notZeroShotProse: ShippedProse;
	};
	research: DeployedBaselineArm;
}

export interface DeployedBaselineBootstrap {
	ci95Low: number;
	ci95High: number;
	pGateGreater: number;
	nValidResamples: number;
	nBootstrap: number;
	seed: number;
}

export interface DeployedBaselineVariant {
	key: string;
	display: string;
	/** Which of the two INDRA belief families this form belongs to. */
	family: string;
	fitted: boolean;
	evidenceScope: string;
	evidenceScopeNote: string;
	/** `evidenceScopeNote` with its plain restatement — `shipped` is byte-identical. */
	evidenceScopeNoteProse: ShippedProse;
	source: string;
	sourceSha256: string;
	auroc: number;
	averagePrecision: number;
	/** gate AUROC − this variant's AUROC. */
	deltaAuroc: number;
	/** This variant's OWN paired interval, so "beats every form" is checkable. */
	bootstrap: DeployedBaselineBootstrap;
	whatItComputes: string;
	/** `whatItComputes` with its plain restatement — `shipped` is byte-identical. */
	whatItComputesProse: ShippedProse;
	/** Present only where the source artifact qualifies its own admissibility. */
	provenanceCaveat: string | null;
	/** Null exactly where `provenanceCaveat` is. */
	provenanceCaveatProse: ShippedProse | null;
	/**
	 * Present where an already-shipped sibling scores the same arm on the same
	 * statements and lands somewhere else. Both numbers are pinned and the reason
	 * travels, so the divergence is a recorded fact rather than a discrepancy
	 * someone finds later.
	 */
	crossCheck: {
		sibling: string;
		siblingKey: string;
		siblingAuroc: number;
		siblingN: number;
		thisAuroc: number;
		deltaVsSibling: number;
		nStatementsScoredDifferently: number;
		whyItDiffers: string;
		/** `whyItDiffers` with its plain restatement — `shipped` is byte-identical. */
		whyItDiffersProse: ShippedProse;
	} | null;
}

export interface DeployedBaselineResearch {
	key: string;
	display: string;
	auroc: number;
	averagePrecision: number;
	/** How far the gate sits ABOVE the paper's fitted research model. */
	deltaGateMinusResearch: number;
	/** How far that research model sits above the DRAWN incumbent. */
	deltaResearchMinusIncumbent: number;
	bootstrap: DeployedBaselineBootstrap;
}

export interface DeployedBaselineLevel {
	key: string;
	display: string;
	auroc: number;
	averagePrecision: number;
}

/**
 * A per-statement count census: what the panel actually contains.
 *
 * `nSingle` / `shareSingle` are here because a MEAN cannot carry the difference
 * the figure has to disclose. `external_curator_v1` reads 1.25 evidence per
 * statement on average, but 84% of its statements hold exactly ONE — so on that
 * panel the gate has nothing left to aggregate and its decision is a bare
 * keep-or-drop on one sentence. On the paper's panel the same share is 20%. Both
 * are drawn.
 */
export interface DeployedBaselineSpread {
	mean: number;
	median: number;
	max: number;
	min: number;
	total: number;
	nStatements: number;
	nSingle: number;
	shareSingle: number;
}

export interface DeployedBaselineHeterogeneity {
	evidenceReadsPerStatement: DeployedBaselineSpread;
	/** The statement's evidence in the SOURCE corpus, where it can be censused. */
	corpusEvidencePerStatement: DeployedBaselineSpread | null;
	corpusEvidenceAbsentBecause: string | null;
	/** Null exactly where `corpusEvidenceAbsentBecause` is. */
	corpusEvidenceAbsentBecauseProse: ShippedProse | null;
	readerSawFullEvidence: boolean;
	readerEvidenceShareOfCorpus: number | null;
	nCurators: number | null;
	curatorNote: string;
	/** `curatorNote` with its plain restatement — `shipped` is byte-identical. */
	curatorNoteProse: ShippedProse;
	baseRateCorrect: number;
	balancedByConstruction: boolean;
	outOfSample: boolean;
	inSampleNote: string | null;
	/** Null exactly where `inSampleNote` is. */
	inSampleNoteProse: ShippedProse | null;
	labelField: string;
	labelNote: string;
	/** `labelNote` with its plain restatement — `shipped` is byte-identical. */
	labelNoteProse: ShippedProse;
	joinMode: string;
	/** `joinMode` with its plain restatement — `shipped` is byte-identical. */
	joinModeProse: ShippedProse;
	/** What the join ACTUALLY did, computed from its own row counts. */
	joinSummary: string;
	nUndefinedExcluded: number;
	incumbentEvidenceScope: string;
	incumbentEvidenceScopeNote: string;
	/** `incumbentEvidenceScopeNote` with its plain restatement. */
	incumbentEvidenceScopeNoteProse: ShippedProse;
	gateEvidenceScope: string;
	gateEvidenceScopeNote: string;
	/** `gateEvidenceScopeNote` with its plain restatement. */
	gateEvidenceScopeNoteProse: ShippedProse;
}

/**
 * INDRA's own scorer restricted to exactly the evidence the reader saw. NOT an
 * incumbent — INDRA serves no such number — so it never enters the argmax. It
 * exists to price the scope asymmetry on the panels where the incumbent is
 * scored over more evidence than the gate ever saw.
 */
export interface DeployedBaselineControl {
	key: string;
	display: string;
	auroc: number;
	averagePrecision: number;
	deltaIncumbentMinusControl: number;
	atOrBelowChance: boolean;
	isAnIncumbent: boolean;
	note: string;
	/** `note` with its plain restatement — `noteProse.shipped === note`. */
	noteProse: ShippedProse;
}

export interface DeployedBaselineProvenance {
	gold: string;
	goldSha256: string;
	run: string;
	runSha256: string;
	readerModel: string;
	join: string;
	/**
	 * `join` with its plain restatement. Same sentence as the row's `joinMode`, so
	 * this is the SAME object, never a second restatement of one string.
	 */
	joinProse: ShippedProse;
}

export interface DeployedBaselinePanel {
	key: string;
	/**
	 * On-screen name, authored HERE against the frozen `key` — never the
	 * artifact's own `display`, which says "the 2023 paper's own panel" and used
	 * to reach the row label, the row title and the SVG `<desc>` unchanged.
	 */
	display: string;
	/**
	 * The artifact's own `display` with the name above as its plain half, so the
	 * wording it ships is kept for the audit trail rather than dropped.
	 */
	displayProse: ShippedProse;
	isPaperPanel: boolean;
	nStatements: number;
	nCorrect: number;
	nErrors: number;
	baseRateCorrect: number;
	balancedByConstruction: boolean;
	nCurators: number | null;
	curatorNote: string;
	/** `curatorNote` with its plain restatement — the row's own twin object. */
	curatorNoteProse: ShippedProse;
	outOfSample: boolean;
	/** Non-null only where the panel fitted something of ours; the flag travels. */
	inSampleNote: string | null;
	/** Null exactly where `inSampleNote` is. */
	inSampleNoteProse: ShippedProse | null;
	labelField: string;
	labelNote: string;
	/** `labelNote` with its plain restatement — the row's own twin object. */
	labelNoteProse: ShippedProse;
	heterogeneity: DeployedBaselineHeterogeneity;
	gate: DeployedBaselineLevel;
	/** The STRONGEST incumbent variant this panel can source, never the weakest. */
	incumbent: DeployedBaselineLevel & {
		family: string;
		fitted: boolean;
		selectedBy: string;
		/** `selectedBy` with its plain restatement — `shipped` is byte-identical. */
		selectedByProse: ShippedProse;
	};
	incumbentVariants: DeployedBaselineVariant[];
	/** What the argmax rule forfeits here: strongest − weakest sourceable form. */
	selectionCostAuroc: number;
	weakestVariantKey: string;
	/** Render THIS, never `weakestVariantKey` — the key is a join, not a name. */
	weakestVariantDisplay: string;
	weakestVariantAuroc: number;
	gateBeatsEveryVariant: boolean;
	nVariantsCiExcludesZero: number;
	deltaAuroc: number;
	deltaFavorsGate: boolean;
	bootstrap: DeployedBaselineBootstrap;
	/** The paper's fitted RF — present on the paper's own panel alone. */
	researchModel: DeployedBaselineResearch | null;
	/** Our production gate, which also swaps the priors. Carried, never drawn. */
	gateSensitivity: {
		key: string;
		display: string;
		auroc: number;
		deltaAuroc: number;
		note: string;
		/** `note` with its plain restatement — `noteProse.shipped === note`. */
		noteProse: ShippedProse;
	} | null;
	evidenceMatchedControl: DeployedBaselineControl | null;
	evidenceMatchedControlAbsentBecause: string | null;
	/** Null exactly where `evidenceMatchedControlAbsentBecause` is. */
	evidenceMatchedControlAbsentBecauseProse: ShippedProse | null;
	provenance: DeployedBaselineProvenance;
}

/** The recomputed proof that the stored belief is not the library default. */
export interface DeployedBaselineServedIdentity {
	question: string;
	/** `question` with its plain restatement — `shipped` is byte-identical. */
	questionProse: ShippedProse;
	finding: string;
	/** `finding` with its plain restatement — `shipped` is byte-identical. */
	findingProse: ShippedProse;
	simpleScorerFloor: number;
	floorDerivation: string;
	/** `floorDerivation` with its plain restatement — `shipped` is byte-identical. */
	floorDerivationProse: ShippedProse;
	floorSource: string;
	floorSourceSha256: string;
	perPanel: {
		/** The FROZEN join key. Never rendered — `panelDisplay` is. */
		panelKey: string;
		/** The panel's on-screen name, resolved from the panel the key names. */
		panelDisplay: string;
		nServed: number;
		nBelowFloor: number;
		fractionBelowFloor: number;
	}[];
	nServedBelowFloor: number;
	nPanelsWithServedBelowFloor: number;
}

export interface DeployedBaselineReplication {
	nPanels: number;
	nPanelsFavoringGate: number;
	nPanelsCiExcludesZero: number;
	nPanelsGateBeatsEveryVariant: number;
	nIncumbentVariantsTotal: number;
	nIncumbentVariantsCiExcludesZero: number;
	deltaMin: number;
	deltaMax: number;
	largestPanelKey: string;
	largestPanelDelta: number;
	largestPanelIsAtTopOfRange: boolean;
	selectionCostAurocMax: number;
	/**
	 * The evidence regime, as a span. The claim the figure draws is that the
	 * result survives BOTH ends of it — 19.8 evidence read per statement on the
	 * paper's panel, 1.2 on the thinnest curation panel — so the span is a
	 * shipped, checked quantity rather than an author writing "16x".
	 */
	readsPerStatementMeanMin: number;
	readsPerStatementMeanMax: number;
	evidenceRegimeFoldSpan: number;
	shareSingleEvidenceMin: number;
	shareSingleEvidenceMax: number;
}

export interface DeployedBaseline {
	metric: string;
	metricSource: string;
	positiveClass: string;
	noisyOrFormula: string;
	question: string;
	/** The short on-figure title. Short BECAUSE it is drawn, not because it is all we can say. */
	figureTitle: string;
	claim: string;
	claimIsNot: string;
	families: DeployedBaselineFamily[];
	arms: DeployedBaselineArms;
	incumbentSelectionRule: string;
	incumbentSelectionRuleCost: string;
	servedBeliefIdentity: DeployedBaselineServedIdentity;
	panels: DeployedBaselinePanel[];
	replication: DeployedBaselineReplication;
	caveats: string[];
	/**
	 * The plain half of every string this figure took off the artifact. Each flat
	 * field above is byte-identical to its twin's `shipped`; the twin is the one to
	 * render, and the shipped half belongs behind the verification boundary. The
	 * per-row evidence-supply notes have their own twins where they are parsed —
	 * they differ row by row, so they cannot live in one bag.
	 */
	prose: DeployedBaselineProse;
	generatedBy: string;
}

/** Every shipped sentence this figure carries, each with its plain restatement. */
export interface DeployedBaselineProse {
	claim: ShippedProse;
	claimIsNot: ShippedProse;
	/** How the comparison INDRA already ships is chosen on each set of statements. */
	incumbentSelectionRule: ShippedProse;
	/** What that rule costs us, in margin we give up. */
	incumbentSelectionRuleCost: ShippedProse;
	/** Index-aligned with `caveats`, pinned to it by a verbatim fragment. */
	caveats: ShippedProse[];
	/**
	 * The header block — the four sentences the SVG `<desc>` opens with. These
	 * reached a screen-reader user in the artifact's own dialect behind no
	 * boundary at all, which is why they are here rather than only on the rows.
	 */
	question: ShippedProse;
	metricSource: ShippedProse;
	noisyOrFormula: ShippedProse;
	/** What the reading step is, and what it is not. */
	gateWhatItIs: ShippedProse;
	/** That it carries worked examples, and calibrates nothing. */
	gateNotZeroShot: ShippedProse;
	/** What the 2023 paper's own fitted model is. */
	researchWhatItIs: ShippedProse;
	/** The recomputed proof that the stored belief is not the library default. */
	servedIdentityQuestion: ShippedProse;
	servedIdentityFinding: ShippedProse;
	servedIdentityFloorDerivation: ShippedProse;
	/**
	 * The SAME objects the rows carry, in artifact order, so a legend and a row can
	 * never show different restatements of one sentence.
	 */
	familyWhatItComputes: ShippedProse[];
	familyWhereItRuns: ShippedProse[];
}

// ---------------------------------------------------------------------------
// validation
// ---------------------------------------------------------------------------

type UnknownRecord = Record<string, unknown>;



function array(value: unknown, context: string): unknown[] {
	if (!Array.isArray(value)) fail(context, 'expected an array');
	return value;
}

/**
 * The shipped evidence-supply note with its plain restatement, selected by the
 * FROZEN scope id the artifact stamps beside it. An id with no restatement gates
 * the figure: a new evidence supply is a new thing to explain, and shipping it
 * with no explanation is the failure this whole twin mechanism exists to stop.
 */
function scopeProse(scopeId: string, note: string, context: string): ShippedProse {
	const plain = DEPLOYED_BASELINE_SCOPE_PLAIN[scopeId];
	if (plain === undefined) fail(context, `no plain restatement for evidence scope "${scopeId}"`);
	return { shipped: note, plain };
}

function finite(value: unknown, context: string): number {
	if (typeof value !== 'number' || !Number.isFinite(value)) {
		fail(context, 'expected a finite number');
	}
	return value;
}




function boolean(value: unknown, context: string): boolean {
	if (typeof value !== 'boolean') fail(context, 'expected a boolean');
	return value;
}


function optionalText(value: unknown, context: string): string | null {
	if (value === null || value === undefined) return null;
	return text(value, context);
}

function close(got: number, want: number, context: string, message: string): void {
	if (!(Math.abs(got - want) <= DEPLOYED_BASELINE_PARITY_TOL)) fail(context, message);
}

function parseSpread(value: unknown, context: string): DeployedBaselineSpread {
	const obj = record(value, context);
	const min = finite(obj.min, `${context}.min`);
	const max = finite(obj.max, `${context}.max`);
	const mean = finite(obj.mean, `${context}.mean`);
	const median = finite(obj.median, `${context}.median`);
	// A census that does not bracket its own mean is not a census.
	if (!(min <= mean && mean <= max)) fail(context, 'the mean must lie inside [min, max]');
	if (!(min <= median && median <= max)) fail(context, 'the median must lie inside [min, max]');
	const total = positiveInteger(obj.total, `${context}.total`);
	const nStatements = positiveInteger(obj.n_statements, `${context}.n_statements`);
	const nSingle = nonNegativeInteger(obj.n_single, `${context}.n_single`);
	const shareSingle = unit(obj.share_single, `${context}.share_single`);
	if (nSingle > nStatements) fail(`${context}.n_single`, 'cannot exceed the statements counted');
	close(shareSingle, nSingle / nStatements, `${context}.share_single`, 'must equal n_single / n_statements');
	// The mean is the total over the count, or one of the three is not this
	// census. This is the arithmetic that makes "1.25 mean, 84% single" checkable
	// rather than two numbers a reader has to take on faith.
	close(mean, total / nStatements, `${context}.mean`, 'must equal total / n_statements');
	// Every statement contributes at least `min`, so a total below n × min is not
	// reachable and the census is describing a different set than it counts.
	if (total < nStatements * min) fail(`${context}.total`, 'is below n_statements × min');
	// A single-evidence statement is one whose count is 1, so a panel that
	// reports any of them cannot have a minimum above 1.
	if (nSingle > 0 && min !== 1) fail(`${context}.n_single`, 'reports single-count statements but min is not 1');
	return { mean, median, max, min, total, nStatements, nSingle, shareSingle };
}

function parseBootstrap(value: unknown, context: string): DeployedBaselineBootstrap {
	const obj = record(value, context);
	const low = finite(obj.ci95_low, `${context}.ci95_low`);
	const high = finite(obj.ci95_high, `${context}.ci95_high`);
	if (!(low <= high)) fail(context, 'ci95_low must not exceed ci95_high');
	const nValid = positiveInteger(obj.n_valid_resamples, `${context}.n_valid_resamples`);
	const nBoot = positiveInteger(obj.n_bootstrap, `${context}.n_bootstrap`);
	if (nValid > nBoot) fail(context, 'n_valid_resamples cannot exceed n_bootstrap');
	return {
		ci95Low: low,
		ci95High: high,
		pGateGreater: unit(obj.p_gate_greater, `${context}.p_gate_greater`),
		nValidResamples: nValid,
		nBootstrap: nBoot,
		seed: nonNegativeInteger(obj.seed, `${context}.seed`)
	};
}

const FAMILY_KEY_SET: readonly string[] = [
	DEPLOYED_BASELINE_FAMILY_KEYS.library,
	DEPLOYED_BASELINE_FAMILY_KEYS.served
];

function parseVariant(
	value: unknown,
	context: string,
	gateAuroc: number
): DeployedBaselineVariant {
	const obj = record(value, context);
	const variantKey = text(obj.key, `${context}.key`);
	const auroc = unit(obj.auroc, `${context}.auroc`);
	const deltaAuroc = finite(obj.delta_auroc, `${context}.delta_auroc`);
	close(
		deltaAuroc,
		gateAuroc - auroc,
		`${context}.delta_auroc`,
		'must equal the gate AUROC minus this variant’s AUROC'
	);
	const family = text(obj.family, `${context}.family`);
	if (!FAMILY_KEY_SET.includes(family)) {
		fail(`${context}.family`, `must be one of the two frozen family keys, got ${family}`);
	}
	const sha = text(obj.source_sha256, `${context}.source_sha256`);
	if (sha.length !== 64) fail(`${context}.source_sha256`, 'expected a sha256 digest');
	const evidenceScope = text(obj.evidence_scope, `${context}.evidence_scope`);
	const evidenceScopeNoteProse = scopeProse(
		evidenceScope,
		text(obj.evidence_scope_note, `${context}.evidence_scope_note`),
		`${context}.evidence_scope`
	);

	let crossCheck: DeployedBaselineVariant['crossCheck'] = null;
	if (obj.cross_check !== null && obj.cross_check !== undefined) {
		const xc = `${context}.cross_check`;
		const x = record(obj.cross_check, xc);
		const siblingAuroc = unit(x.sibling_auroc, `${xc}.sibling_auroc`);
		const thisAuroc = unit(x.this_auroc, `${xc}.this_auroc`);
		const delta = finite(x.delta_vs_sibling, `${xc}.delta_vs_sibling`);
		close(thisAuroc, auroc, `${xc}.this_auroc`, 'must be the variant’s own AUROC');
		close(delta, thisAuroc - siblingAuroc, `${xc}.delta_vs_sibling`, 'must equal this minus the sibling');
		// Reading INDRA's own resource must never have produced an EASIER
		// comparator than the shipped sibling; that would be the argmax rule
		// running backwards.
		if (delta < -DEPLOYED_BASELINE_PARITY_TOL) {
			fail(`${xc}.delta_vs_sibling`, 'the drawn form must not be weaker than the shipped sibling');
		}
		const whyItDiffersProse = anchoredShippedProse(
			text(x.why_it_differs, `${xc}.why_it_differs`),
			DEPLOYED_BASELINE_CROSS_CHECK_TWIN,
			`${xc}.why_it_differs`
		);
		crossCheck = {
			sibling: text(x.sibling, `${xc}.sibling`),
			siblingKey: text(x.sibling_key, `${xc}.sibling_key`),
			siblingAuroc,
			siblingN: positiveInteger(x.sibling_n, `${xc}.sibling_n`),
			thisAuroc,
			deltaVsSibling: delta,
			nStatementsScoredDifferently: nonNegativeInteger(
				x.n_statements_scored_differently,
				`${xc}.n_statements_scored_differently`
			),
			whyItDiffers: whyItDiffersProse.shipped,
			whyItDiffersProse
		};
	}

	// Looked up by the FROZEN variant key, then pinned to its own text: a form with
	// no authored restatement gates the figure, and three rows ship the SAME
	// sentence, so the key alone cannot bind the right restatement to the right row.
	const whatItComputesProse = keyedShippedProse(
		variantKey,
		text(obj.what_it_computes, `${context}.what_it_computes`),
		DEPLOYED_BASELINE_VARIANT_TWINS,
		`${context}.what_it_computes`
	);
	const provenanceCaveat = optionalText(obj.provenance_caveat, `${context}.provenance_caveat`);

	return {
		key: variantKey,
		display: text(obj.display, `${context}.display`),
		family,
		fitted: boolean(obj.fitted, `${context}.fitted`),
		evidenceScope: evidenceScope,
		evidenceScopeNote: evidenceScopeNoteProse.shipped,
		evidenceScopeNoteProse,
		source: text(obj.source, `${context}.source`),
		sourceSha256: sha,
		auroc,
		averagePrecision: unit(obj.average_precision, `${context}.average_precision`),
		deltaAuroc,
		bootstrap: parseBootstrap(obj.bootstrap, `${context}.bootstrap`),
		whatItComputes: whatItComputesProse.shipped,
		whatItComputesProse,
		provenanceCaveat,
		provenanceCaveatProse:
			provenanceCaveat === null
				? null
				: anchoredShippedProse(
						provenanceCaveat,
						DEPLOYED_BASELINE_PROVENANCE_CAVEAT_TWIN,
						`${context}.provenance_caveat`
					),
		crossCheck
	};
}

function parseHeterogeneity(
	value: unknown,
	context: string,
	nStatements: number,
	twins: DeployedBaselinePanelTwins
): DeployedBaselineHeterogeneity {
	const obj = record(value, context);
	const reads = parseSpread(obj.evidence_reads_per_statement, `${context}.evidence_reads_per_statement`);
	const corpusRaw = obj.corpus_evidence_per_statement;
	const corpus =
		corpusRaw === null || corpusRaw === undefined
			? null
			: parseSpread(corpusRaw, `${context}.corpus_evidence_per_statement`);
	const corpusAbsent = optionalText(obj.corpus_evidence_absent_because, `${context}.corpus_evidence_absent_because`);
	// Fail-closed on a silently missing census: either the numbers are here or
	// the reason they are not is. A blank both ways would render as "no data"
	// with nothing saying why.
	if (corpus === null && corpusAbsent === null) {
		fail(`${context}.corpus_evidence_per_statement`, 'a missing census must say why it is missing');
	}
	if (corpus !== null && corpusAbsent !== null) {
		fail(`${context}.corpus_evidence_absent_because`, 'a present census cannot also be absent');
	}
	const readerSawFull = boolean(obj.reader_saw_full_evidence, `${context}.reader_saw_full_evidence`);
	const shareRaw = obj.reader_evidence_share_of_corpus;
	let share: number | null = null;
	if (shareRaw !== null && shareRaw !== undefined) {
		share = unit(shareRaw, `${context}.reader_evidence_share_of_corpus`);
		if (corpus !== null) {
			close(
				share,
				reads.total / corpus.total,
				`${context}.reader_evidence_share_of_corpus`,
				'must equal the reader’s evidence total over the corpus evidence total'
			);
		}
	}
	// The share and the flag have to agree: "the reader saw everything" is a
	// share of exactly 1, and anything less must not claim it did.
	if (readerSawFull && share !== 1) {
		fail(`${context}.reader_saw_full_evidence`, 'claims the full corpus but the share is not 1');
	}
	if (!readerSawFull && share === 1) {
		fail(`${context}.reader_saw_full_evidence`, 'the share is 1 but the flag denies it');
	}
	if (reads.total < nStatements) {
		fail(`${context}.evidence_reads_per_statement`, 'fewer evidence reads than statements');
	}
	// Both censuses must be censuses of THIS panel's statements. Without this a
	// spread computed over a different (larger, older, differently-joined) set
	// would render beside the panel's own n as if it described it — and the
	// single-evidence share is exactly the number a reader would divide by n.
	if (reads.nStatements !== nStatements) {
		fail(`${context}.evidence_reads_per_statement.n_statements`, 'does not census this panel’s statements');
	}
	if (corpus !== null && corpus.nStatements !== nStatements) {
		fail(`${context}.corpus_evidence_per_statement.n_statements`, 'does not census this panel’s statements');
	}
	const nCuratorsRaw = obj.n_curators;
	if (
		nCuratorsRaw !== null &&
		(typeof nCuratorsRaw !== 'number' || !Number.isInteger(nCuratorsRaw) || nCuratorsRaw < 1)
	) {
		fail(`${context}.n_curators`, 'expected a positive integer or null');
	}
	const incumbentEvidenceScope = text(
		obj.incumbent_evidence_scope,
		`${context}.incumbent_evidence_scope`
	);
	const incumbentEvidenceScopeNoteProse = scopeProse(
		incumbentEvidenceScope,
		text(obj.incumbent_evidence_scope_note, `${context}.incumbent_evidence_scope_note`),
		`${context}.incumbent_evidence_scope`
	);
	const gateEvidenceScope = text(obj.gate_evidence_scope, `${context}.gate_evidence_scope`);
	const gateEvidenceScopeNoteProse = scopeProse(
		gateEvidenceScope,
		text(obj.gate_evidence_scope_note, `${context}.gate_evidence_scope_note`),
		`${context}.gate_evidence_scope`
	);
	// Every sentence that varies BY SET OF STATEMENTS, each pinned to its own text
	// through the twins this set was handed. `panelProse` below reuses these very
	// objects, so a row and its heading can never restate one sentence two ways.
	const curatorNoteProse = anchoredShippedProse(
		text(obj.curator_note, `${context}.curator_note`),
		twins.curatorNote,
		`${context}.curator_note`
	);
	const labelNoteProse = anchoredShippedProse(
		text(obj.label_note, `${context}.label_note`),
		twins.labelNote,
		`${context}.label_note`
	);
	const joinModeProse = anchoredShippedProse(
		text(obj.join_mode, `${context}.join_mode`),
		twins.joinMode,
		`${context}.join_mode`
	);
	const inSampleNote = optionalText(obj.in_sample_note, `${context}.in_sample_note`);
	// A flag that travels with no restatement is the failure this whole mechanism
	// exists to stop, so a note whose twin was never authored gates the figure.
	if (inSampleNote !== null && twins.inSampleNote === null) {
		fail(`${context}.in_sample_note`, 'ships a note with no plain restatement authored for it');
	}
	if (corpusAbsent !== null && twins.corpusEvidenceAbsentBecause === null) {
		fail(
			`${context}.corpus_evidence_absent_because`,
			'ships a reason with no plain restatement authored for it'
		);
	}

	return {
		evidenceReadsPerStatement: reads,
		corpusEvidencePerStatement: corpus,
		corpusEvidenceAbsentBecause: corpusAbsent,
		corpusEvidenceAbsentBecauseProse:
			corpusAbsent === null || twins.corpusEvidenceAbsentBecause === null
				? null
				: anchoredShippedProse(
						corpusAbsent,
						twins.corpusEvidenceAbsentBecause,
						`${context}.corpus_evidence_absent_because`
					),
		readerSawFullEvidence: readerSawFull,
		readerEvidenceShareOfCorpus: share,
		nCurators: nCuratorsRaw as number | null,
		curatorNote: curatorNoteProse.shipped,
		curatorNoteProse,
		baseRateCorrect: unit(obj.base_rate_correct, `${context}.base_rate_correct`),
		balancedByConstruction: boolean(obj.balanced_by_construction, `${context}.balanced_by_construction`),
		outOfSample: boolean(obj.out_of_sample, `${context}.out_of_sample`),
		inSampleNote,
		inSampleNoteProse:
			inSampleNote === null || twins.inSampleNote === null
				? null
				: anchoredShippedProse(inSampleNote, twins.inSampleNote, `${context}.in_sample_note`),
		labelField: text(obj.label_field, `${context}.label_field`),
		labelNote: labelNoteProse.shipped,
		labelNoteProse,
		joinMode: joinModeProse.shipped,
		joinModeProse,
		joinSummary: text(obj.join_summary, `${context}.join_summary`),
		nUndefinedExcluded: nonNegativeInteger(obj.n_undefined_excluded, `${context}.n_undefined_excluded`),
		incumbentEvidenceScope,
		incumbentEvidenceScopeNote: incumbentEvidenceScopeNoteProse.shipped,
		incumbentEvidenceScopeNoteProse,
		gateEvidenceScope,
		gateEvidenceScopeNote: gateEvidenceScopeNoteProse.shipped,
		gateEvidenceScopeNoteProse
	};
}

function parsePanel(value: unknown, index: number): DeployedBaselinePanel {
	const context = `deployed_baseline_replication.panels[${index}]`;
	const obj = record(value, context);
	const expectedKey = DEPLOYED_BASELINE_PANEL_KEYS[index];
	if (obj.key !== expectedKey) {
		fail(`${context}.key`, `expected the fixed panel order — ${expectedKey}`);
	}
	// The restatements this set of statements owns. A set with none authored gates
	// rather than reaching a reader in the artifact's own wording.
	const twins = DEPLOYED_BASELINE_PANEL_TWINS[expectedKey];
	if (twins === undefined) {
		fail(`${context}.key`, `no plain restatements are authored for "${expectedKey}"`);
	}

	const nStatements = positiveInteger(obj.n_statements, `${context}.n_statements`);
	const nCorrect = positiveInteger(obj.n_correct, `${context}.n_correct`);
	const nErrors = positiveInteger(obj.n_errors, `${context}.n_errors`);
	if (nCorrect + nErrors !== nStatements) {
		fail(context, 'n_correct + n_errors must equal n_statements');
	}
	const baseRate = unit(obj.base_rate_correct, `${context}.base_rate_correct`);
	close(baseRate, nCorrect / nStatements, `${context}.base_rate_correct`, 'must equal n_correct / n_statements');

	if (obj.positive_class !== DEPLOYED_BASELINE_POSITIVE_CLASS) {
		fail(`${context}.positive_class`, `expected ${DEPLOYED_BASELINE_POSITIVE_CLASS}`);
	}

	const gateObj = record(obj.gate, `${context}.gate`);
	if (gateObj.key !== DEPLOYED_BASELINE_ARM_KEYS.gate) {
		fail(`${context}.gate.key`, `expected ${DEPLOYED_BASELINE_ARM_KEYS.gate}`);
	}
	const gateAuroc = unit(gateObj.auroc, `${context}.gate.auroc`);
	const gate: DeployedBaselineLevel = {
		key: gateObj.key,
		display: text(gateObj.display, `${context}.gate.display`),
		auroc: gateAuroc,
		averagePrecision: unit(gateObj.average_precision, `${context}.gate.average_precision`)
	};

	const heterogeneity = parseHeterogeneity(
		obj.heterogeneity,
		`${context}.heterogeneity`,
		nStatements,
		twins
	);
	// The heterogeneity block restates the panel census; if the two ever disagree
	// the figure would print one number and caption another.
	close(
		heterogeneity.baseRateCorrect,
		baseRate,
		`${context}.heterogeneity.base_rate_correct`,
		'must equal the panel’s own base rate'
	);

	const variantsRaw = array(obj.incumbent_variants, `${context}.incumbent_variants`);
	if (variantsRaw.length === 0) {
		fail(`${context}.incumbent_variants`, 'a panel with no sourceable incumbent must be dropped, not quoted');
	}
	const incumbentVariants = variantsRaw.map((entry, i) =>
		parseVariant(entry, `${context}.incumbent_variants[${i}]`, gateAuroc)
	);
	if (nonNegativeInteger(obj.n_incumbent_variants, `${context}.n_incumbent_variants`) !== incumbentVariants.length) {
		fail(`${context}.n_incumbent_variants`, 'must be the count it reports, not a claim');
	}

	const incumbentObj = record(obj.incumbent, `${context}.incumbent`);
	const incumbentKey = text(incumbentObj.key, `${context}.incumbent.key`);
	const incumbentAuroc = unit(incumbentObj.auroc, `${context}.incumbent.auroc`);
	// THE rule the whole comparison rests on: the drawn incumbent is the
	// STRONGEST form of INDRA's own belief this panel can source. If a stronger
	// variant exists and was not chosen, the figure is comparing against a
	// convenient baseline and must not render.
	const strongest = incumbentVariants.reduce((best, v) => (v.auroc > best.auroc ? v : best));
	const weakest = incumbentVariants.reduce((worst, v) => (v.auroc < worst.auroc ? v : worst));
	if (incumbentKey !== strongest.key) {
		fail(
			`${context}.incumbent.key`,
			`must be the strongest sourceable incumbent (${strongest.key}), not ${incumbentKey}`
		);
	}
	close(incumbentAuroc, strongest.auroc, `${context}.incumbent.auroc`, 'must equal the strongest variant’s AUROC');
	const incumbentFamily = text(incumbentObj.family, `${context}.incumbent.family`);
	if (incumbentFamily !== strongest.family) {
		fail(`${context}.incumbent.family`, 'must be the family of the variant it drew');
	}
	if (boolean(incumbentObj.fitted, `${context}.incumbent.fitted`) !== strongest.fitted) {
		fail(`${context}.incumbent.fitted`, 'must be the fitted flag of the variant it drew');
	}
	// The rule's price, and it must be the price the variants imply.
	const selectionCost = finite(obj.selection_cost_auroc, `${context}.selection_cost_auroc`);
	close(
		selectionCost,
		strongest.auroc - weakest.auroc,
		`${context}.selection_cost_auroc`,
		'must equal the strongest minus the weakest sourceable form'
	);
	if (obj.weakest_variant_key !== weakest.key) {
		fail(`${context}.weakest_variant_key`, 'must name the weakest sourceable form');
	}
	close(
		finite(obj.weakest_variant_auroc, `${context}.weakest_variant_auroc`),
		weakest.auroc,
		`${context}.weakest_variant_auroc`,
		'must equal the weakest variant’s AUROC'
	);

	const deltaAuroc = finite(obj.delta_auroc, `${context}.delta_auroc`);
	close(
		deltaAuroc,
		gateAuroc - incumbentAuroc,
		`${context}.delta_auroc`,
		'must equal the gate AUROC minus the incumbent AUROC'
	);
	const deltaFavorsGate = boolean(obj.delta_favors_gate, `${context}.delta_favors_gate`);
	if (deltaFavorsGate !== deltaAuroc > 0) {
		fail(`${context}.delta_favors_gate`, 'must be the comparison it reports, not a claim');
	}
	const beatsEvery = boolean(obj.gate_beats_every_variant, `${context}.gate_beats_every_variant`);
	if (beatsEvery !== incumbentVariants.every((v) => v.deltaAuroc > 0)) {
		fail(`${context}.gate_beats_every_variant`, 'must be the comparison it reports, not a claim');
	}
	const nVariantsExcl = nonNegativeInteger(obj.n_variants_ci_excludes_zero, `${context}.n_variants_ci_excludes_zero`);
	if (nVariantsExcl !== incumbentVariants.filter((v) => v.bootstrap.ci95Low > 0).length) {
		fail(`${context}.n_variants_ci_excludes_zero`, 'must be the count it reports, not a claim');
	}
	const bootstrap = parseBootstrap(obj.bootstrap, `${context}.bootstrap`);
	// The headline interval must be the DRAWN incumbent's own, never a different
	// pairing that happens to be narrower.
	close(bootstrap.ci95Low, strongest.bootstrap.ci95Low, `${context}.bootstrap`, 'must be the drawn incumbent’s own interval');
	close(bootstrap.ci95High, strongest.bootstrap.ci95High, `${context}.bootstrap`, 'must be the drawn incumbent’s own interval');

	let researchModel: DeployedBaselineResearch | null = null;
	if (obj.research_model !== null && obj.research_model !== undefined) {
		const rc = `${context}.research_model`;
		const r = record(obj.research_model, rc);
		if (r.key !== DEPLOYED_BASELINE_ARM_KEYS.research) {
			fail(`${rc}.key`, `expected ${DEPLOYED_BASELINE_ARM_KEYS.research}`);
		}
		const rAuroc = unit(r.auroc, `${rc}.auroc`);
		const gateMinus = finite(r.delta_auroc_gate_minus_research, `${rc}.delta_auroc_gate_minus_research`);
		const researchMinus = finite(
			r.delta_auroc_research_minus_incumbent,
			`${rc}.delta_auroc_research_minus_incumbent`
		);
		close(gateMinus, gateAuroc - rAuroc, `${rc}.delta_auroc_gate_minus_research`, 'must equal gate − research');
		close(
			researchMinus,
			rAuroc - incumbentAuroc,
			`${rc}.delta_auroc_research_minus_incumbent`,
			'must equal research − incumbent'
		);
		researchModel = {
			key: r.key,
			display: text(r.display, `${rc}.display`),
			auroc: rAuroc,
			averagePrecision: unit(r.average_precision, `${rc}.average_precision`),
			deltaGateMinusResearch: gateMinus,
			deltaResearchMinusIncumbent: researchMinus,
			bootstrap: parseBootstrap(r.bootstrap, `${rc}.bootstrap`)
		};
	}
	const isPaperPanel = boolean(obj.is_paper_panel, `${context}.is_paper_panel`);
	if (isPaperPanel !== (obj.key === DEPLOYED_BASELINE_PAPER_PANEL_KEY)) {
		fail(`${context}.is_paper_panel`, 'must mark exactly the paper’s own panel');
	}
	// The paper's RF is only drawable where its released out-of-fold predictions
	// exist, which is the paper's own panel. Anywhere else it would be invented.
	if ((researchModel !== null) !== isPaperPanel) {
		fail(`${context}.research_model`, 'the paper’s fitted RF belongs to the paper’s panel alone');
	}

	let gateSensitivity: DeployedBaselinePanel['gateSensitivity'] = null;
	if (obj.gate_sensitivity !== null && obj.gate_sensitivity !== undefined) {
		const sc = `${context}.gate_sensitivity`;
		const s = record(obj.gate_sensitivity, sc);
		const sAuroc = unit(s.auroc, `${sc}.auroc`);
		const sDelta = finite(s.delta_auroc, `${sc}.delta_auroc`);
		close(sDelta, sAuroc - incumbentAuroc, `${sc}.delta_auroc`, 'must equal this variant − incumbent');
		const sNoteProse = anchoredShippedProse(
			text(s.note, `${sc}.note`),
			DEPLOYED_BASELINE_GATE_SENSITIVITY_TWIN,
			`${sc}.note`
		);
		gateSensitivity = {
			key: text(s.key, `${sc}.key`),
			display: text(s.display, `${sc}.display`),
			auroc: sAuroc,
			deltaAuroc: sDelta,
			note: sNoteProse.shipped,
			noteProse: sNoteProse
		};
	}

	let control: DeployedBaselineControl | null = null;
	const controlAbsent = optionalText(
		obj.evidence_matched_control_absent_because,
		`${context}.evidence_matched_control_absent_because`
	);
	if (obj.evidence_matched_control !== null && obj.evidence_matched_control !== undefined) {
		const cc = `${context}.evidence_matched_control`;
		const c = record(obj.evidence_matched_control, cc);
		const cAuroc = unit(c.auroc, `${cc}.auroc`);
		const gap = finite(c.delta_auroc_incumbent_minus_control, `${cc}.delta_auroc_incumbent_minus_control`);
		close(gap, incumbentAuroc - cAuroc, `${cc}.delta_auroc_incumbent_minus_control`, 'must equal incumbent − control');
		if (boolean(c.at_or_below_chance, `${cc}.at_or_below_chance`) !== cAuroc <= DEPLOYED_BASELINE_AXIS_MIN) {
			fail(`${cc}.at_or_below_chance`, 'must be the comparison it reports, not a claim');
		}
		// The control is a diagnostic, never a comparator. If it ever declares
		// itself an incumbent it would be eligible for the argmax, and a
		// below-chance "incumbent" would flatter the gate enormously.
		if (boolean(c.is_an_incumbent, `${cc}.is_an_incumbent`)) {
			fail(`${cc}.is_an_incumbent`, 'the evidence-matched control is not an incumbent');
		}
		const cNoteProse = anchoredShippedProse(
			text(c.note, `${cc}.note`),
			DEPLOYED_BASELINE_CONTROL_NOTE_TWIN,
			`${cc}.note`
		);
		control = {
			key: text(c.key, `${cc}.key`),
			display: text(c.display, `${cc}.display`),
			auroc: cAuroc,
			averagePrecision: unit(c.average_precision, `${cc}.average_precision`),
			deltaIncumbentMinusControl: gap,
			atOrBelowChance: cAuroc <= DEPLOYED_BASELINE_AXIS_MIN,
			isAnIncumbent: false,
			note: cNoteProse.shipped,
			noteProse: cNoteProse
		};
	}
	// Exactly one of "here is the control" and "here is why there is none".
	if ((control === null) === (controlAbsent === null)) {
		fail(
			`${context}.evidence_matched_control`,
			'either the control or the stated reason it is absent, never both and never neither'
		);
	}

	const pc = `${context}.provenance`;
	const prov = record(obj.provenance, pc);

	// FOUR sentences are shipped TWICE — once on the panel, once inside its own
	// heterogeneity block — and this figure prints them from both places. Each is
	// parsed independently, so a drifted copy still gates, and then asserted
	// byte-identical to the copy that already carries the restatement. Binding the
	// same twin object to both is only honest once they are known to be one
	// sentence; otherwise a row and its heading could restate different text under
	// one `shipped` value, which is the exact drift the twins exist to prevent.
	const panelInSampleNote = optionalText(obj.in_sample_note, `${context}.in_sample_note`);
	if (panelInSampleNote !== heterogeneity.inSampleNote) {
		fail(`${context}.in_sample_note`, 'must be the note the heterogeneity block already carries');
	}
	const panelCuratorNote = text(obj.curator_note, `${context}.curator_note`);
	if (panelCuratorNote !== heterogeneity.curatorNote) {
		fail(`${context}.curator_note`, 'must be the note the heterogeneity block already carries');
	}
	const panelLabelNote = text(obj.label_note, `${context}.label_note`);
	if (panelLabelNote !== heterogeneity.labelNote) {
		fail(`${context}.label_note`, 'must be the note the heterogeneity block already carries');
	}
	const provJoin = text(prov.join, `${pc}.join`);
	if (provJoin !== heterogeneity.joinMode) {
		fail(`${pc}.join`, 'must be the join the heterogeneity block already describes');
	}
	// How the drawn comparison was chosen. Short, and a banned word in every byte
	// of it — which is precisely why it needs a restatement rather than a pass.
	const selectedByProse = anchoredShippedProse(
		text(incumbentObj.selected_by, `${context}.incumbent.selected_by`),
		DEPLOYED_BASELINE_SELECTED_BY_TWIN,
		`${context}.incumbent.selected_by`
	);
	const controlAbsentProse =
		controlAbsent === null
			? null
			: anchoredShippedProse(
					controlAbsent,
					DEPLOYED_BASELINE_CONTROL_ABSENT_TWIN,
					`${context}.evidence_matched_control_absent_because`
				);

	// The name on the screen is resolved from the FROZEN key, never passed through
	// from the artifact: its own `display` is written in the artifact's dialect. A
	// key with no authored name gates the figure rather than falling back to the
	// shipped string, which is the fallback that put the dialect on screen.
	const authoredDisplay = DEPLOYED_BASELINE_PANEL_DISPLAY[expectedKey];
	if (authoredDisplay === undefined) {
		fail(`${context}.display`, `no on-screen name is authored for "${expectedKey}"`);
	}
	const displayProse: ShippedProse = {
		shipped: text(obj.display, `${context}.display`),
		plain: authoredDisplay
	};

	return {
		key: obj.key,
		display: authoredDisplay,
		displayProse,
		isPaperPanel,
		nStatements,
		nCorrect,
		nErrors,
		baseRateCorrect: baseRate,
		balancedByConstruction: boolean(obj.balanced_by_construction, `${context}.balanced_by_construction`),
		nCurators: heterogeneity.nCurators,
		curatorNote: panelCuratorNote,
		curatorNoteProse: heterogeneity.curatorNoteProse,
		outOfSample: boolean(obj.out_of_sample, `${context}.out_of_sample`),
		inSampleNote: panelInSampleNote,
		inSampleNoteProse: heterogeneity.inSampleNoteProse,
		labelField: text(obj.label_field, `${context}.label_field`),
		labelNote: panelLabelNote,
		labelNoteProse: heterogeneity.labelNoteProse,
		heterogeneity,
		gate,
		incumbent: {
			key: incumbentKey,
			display: text(incumbentObj.display, `${context}.incumbent.display`),
			family: incumbentFamily,
			fitted: strongest.fitted,
			auroc: incumbentAuroc,
			averagePrecision: unit(incumbentObj.average_precision, `${context}.incumbent.average_precision`),
			selectedBy: selectedByProse.shipped,
			selectedByProse
		},
		incumbentVariants,
		selectionCostAuroc: selectionCost,
		weakestVariantKey: weakest.key,
		weakestVariantDisplay: weakest.display,
		weakestVariantAuroc: weakest.auroc,
		gateBeatsEveryVariant: beatsEvery,
		nVariantsCiExcludesZero: nVariantsExcl,
		deltaAuroc,
		deltaFavorsGate,
		bootstrap,
		researchModel,
		gateSensitivity,
		evidenceMatchedControl: control,
		evidenceMatchedControlAbsentBecause: controlAbsent,
		evidenceMatchedControlAbsentBecauseProse: controlAbsentProse,
		provenance: {
			gold: text(prov.gold, `${pc}.gold`),
			goldSha256: text(prov.gold_sha256, `${pc}.gold_sha256`),
			run: text(prov.run, `${pc}.run`),
			runSha256: text(prov.run_sha256, `${pc}.run_sha256`),
			readerModel: text(prov.reader_model, `${pc}.reader_model`),
			join: provJoin,
			joinProse: heterogeneity.joinModeProse
		}
	};
}

/**
 * @param whatItIsTwin the restatement authored for THIS arm. Passed in rather
 * than looked up, because there are exactly two arms and each one's sentence is
 * about a different thing entirely — the reading step, and the 2023 paper's own
 * fitted model.
 */
function parseArm(
	value: unknown,
	context: string,
	expectedKey: string,
	whatItIsTwin: AnchoredProse
): DeployedBaselineArm {
	const obj = record(value, context);
	if (obj.key !== expectedKey) fail(`${context}.key`, `expected ${expectedKey}`);
	const whatItIsProse = anchoredShippedProse(
		text(obj.what_it_is, `${context}.what_it_is`),
		whatItIsTwin,
		`${context}.what_it_is`
	);
	return {
		key: expectedKey,
		display: text(obj.display, `${context}.display`),
		deployed: boolean(obj.deployed, `${context}.deployed`),
		fitted: boolean(obj.fitted, `${context}.fitted`),
		whatItIs: whatItIsProse.shipped,
		whatItIsProse
	};
}

function parseFamily(value: unknown, index: number): DeployedBaselineFamily {
	const context = `deployed_baseline_replication.incumbent_families[${index}]`;
	const obj = record(value, context);
	const expected = FAMILY_KEY_SET[index];
	if (obj.key !== expected) fail(`${context}.key`, `expected the fixed family order — ${expected}`);
	const twins = DEPLOYED_BASELINE_FAMILY_TWINS[expected];
	if (twins === undefined) {
		fail(`${context}.key`, `no plain restatements are authored for "${expected}"`);
	}
	const whatItComputesProse = anchoredShippedProse(
		text(obj.what_it_computes, `${context}.what_it_computes`),
		twins.whatItComputes,
		`${context}.what_it_computes`
	);
	const whereItRunsProse = anchoredShippedProse(
		text(obj.where_it_runs, `${context}.where_it_runs`),
		twins.whereItRuns,
		`${context}.where_it_runs`
	);
	return {
		key: expected,
		display: text(obj.display, `${context}.display`),
		deployed: boolean(obj.deployed, `${context}.deployed`),
		fitted: boolean(obj.fitted, `${context}.fitted`),
		whatItComputes: whatItComputesProse.shipped,
		whatItComputesProse,
		whereItRuns: whereItRunsProse.shipped,
		whereItRunsProse,
		shipsIn: text(obj.ships_in, `${context}.ships_in`)
	};
}

/**
 * @param displayByKey the panels' frozen join keys mapped to their on-screen
 * names. The identity block travels by KEY, but a key must never reach a render
 * position, so its display name is resolved here — where an unknown key fails
 * the contract — rather than in the component, where it could only be printed.
 */
function parseServedIdentity(
	value: unknown,
	context: string,
	displayByKey: Map<string, string>
): DeployedBaselineServedIdentity {
	const obj = record(value, context);
	const floor = unit(obj.simple_scorer_floor, `${context}.simple_scorer_floor`);
	const sha = text(obj.floor_source_sha256, `${context}.floor_source_sha256`);
	if (sha.length !== 64) fail(`${context}.floor_source_sha256`, 'expected a sha256 digest');
	const perPanel = array(obj.per_panel, `${context}.per_panel`).map((entry, i) => {
		const ec = `${context}.per_panel[${i}]`;
		const row = record(entry, ec);
		const key = text(row.panel_key, `${ec}.panel_key`);
		const panelDisplay = displayByKey.get(key);
		if (panelDisplay === undefined) fail(`${ec}.panel_key`, 'names a panel the artifact does not carry');
		const nServed = positiveInteger(row.n_served, `${ec}.n_served`);
		const nBelow = nonNegativeInteger(row.n_below_floor, `${ec}.n_below_floor`);
		if (nBelow > nServed) fail(`${ec}.n_below_floor`, 'cannot exceed the number of served beliefs');
		const fraction = unit(row.fraction_below_floor, `${ec}.fraction_below_floor`);
		close(fraction, nBelow / nServed, `${ec}.fraction_below_floor`, 'must equal n_below_floor / n_served');
		return { panelKey: key, panelDisplay, nServed, nBelowFloor: nBelow, fractionBelowFloor: fraction };
	});
	const total = nonNegativeInteger(obj.n_served_below_floor, `${context}.n_served_below_floor`);
	if (total !== perPanel.reduce((sum, p) => sum + p.nBelowFloor, 0)) {
		fail(`${context}.n_served_below_floor`, 'must be the count it reports, not a claim');
	}
	const nPanels = nonNegativeInteger(obj.n_panels_with_served_below_floor, `${context}.n_panels_with_served_below_floor`);
	if (nPanels !== perPanel.filter((p) => p.nBelowFloor > 0).length) {
		fail(`${context}.n_panels_with_served_below_floor`, 'must be the count it reports, not a claim');
	}
	// The whole two-family split rests on this. If no served belief falls below
	// the floor, the served belief is inside SimpleScorer's reachable range and
	// the figure must NOT draw a distinction the data no longer supports.
	if (total === 0) {
		fail(`${context}.n_served_below_floor`, 'the two-family split is unsupported: nothing falls below the floor');
	}
	const questionProse: ShippedProse = {
		shipped: text(obj.question, `${context}.question`),
		plain: DEPLOYED_BASELINE_PLAIN.servedIdentityQuestion
	};
	const findingProse: ShippedProse = {
		shipped: text(obj.finding, `${context}.finding`),
		plain: DEPLOYED_BASELINE_PLAIN.servedIdentityFinding
	};
	const floorDerivationProse: ShippedProse = {
		shipped: text(obj.floor_derivation, `${context}.floor_derivation`),
		plain: DEPLOYED_BASELINE_PLAIN.floorDerivation
	};
	return {
		question: questionProse.shipped,
		questionProse,
		finding: findingProse.shipped,
		findingProse,
		simpleScorerFloor: floor,
		floorDerivation: floorDerivationProse.shipped,
		floorDerivationProse,
		floorSource: text(obj.floor_source, `${context}.floor_source`),
		floorSourceSha256: sha,
		perPanel,
		nServedBelowFloor: total,
		nPanelsWithServedBelowFloor: nPanels
	};
}

/** Validate the shipped artifact. THROWS on any drift; never returns partial. */
export function validateDeployedBaseline(raw: unknown): DeployedBaseline {
	const context = 'deployed_baseline_replication';
	const obj = record(raw, context);
	if (obj.artifact_kind !== DEPLOYED_BASELINE_ARTIFACT_KIND) {
		fail(`${context}.artifact_kind`, `expected ${DEPLOYED_BASELINE_ARTIFACT_KIND}`);
	}
	if (obj.schema_version !== DEPLOYED_BASELINE_SCHEMA_VERSION) {
		fail(`${context}.schema_version`, `expected ${DEPLOYED_BASELINE_SCHEMA_VERSION}`);
	}
	if (obj.metric !== DEPLOYED_BASELINE_METRIC) {
		fail(`${context}.metric`, `expected ${DEPLOYED_BASELINE_METRIC}`);
	}
	if (obj.positive_class !== DEPLOYED_BASELINE_POSITIVE_CLASS) {
		fail(`${context}.positive_class`, `expected ${DEPLOYED_BASELINE_POSITIVE_CLASS}`);
	}

	const familiesRaw = array(obj.incumbent_families, `${context}.incumbent_families`);
	if (familiesRaw.length !== FAMILY_KEY_SET.length) {
		fail(`${context}.incumbent_families`, `expected ${FAMILY_KEY_SET.length} families`);
	}
	const families = familiesRaw.map(parseFamily);
	// The premise of the figure. Both families must be INDRA's own and marked
	// deployed; the library default must stay unfitted and the served belief must
	// stay fitted, because those two facts are the whole reason they are separate
	// rows in the reader's head.
	for (const family of families) {
		if (!family.deployed) fail(`${context}.incumbent_families`, `${family.key} must be a DEPLOYED form of INDRA belief`);
	}
	const library = families.find((f) => f.key === DEPLOYED_BASELINE_FAMILY_KEYS.library);
	const served = families.find((f) => f.key === DEPLOYED_BASELINE_FAMILY_KEYS.served);
	if (!library || library.fitted) fail(`${context}.incumbent_families`, 'the library default must be unfitted');
	if (!served || !served.fitted) fail(`${context}.incumbent_families`, 'the stored production belief is fitted');

	const armsObj = record(obj.arms, `${context}.arms`);
	const gateArm = parseArm(
		armsObj.gate,
		`${context}.arms.gate`,
		DEPLOYED_BASELINE_ARM_KEYS.gate,
		DEPLOYED_BASELINE_GATE_WHAT_IT_IS_TWIN
	);
	const researchArm = parseArm(
		armsObj.research_model,
		`${context}.arms.research_model`,
		DEPLOYED_BASELINE_ARM_KEYS.research,
		DEPLOYED_BASELINE_RESEARCH_WHAT_IT_IS_TWIN
	);
	if (researchArm.deployed) fail(`${context}.arms.research_model.deployed`, 'the paper’s RF was never deployed');
	if (!researchArm.fitted) fail(`${context}.arms.research_model.fitted`, 'the paper’s RF is a fitted model');
	if (gateArm.fitted) fail(`${context}.arms.gate.fitted`, 'the hard gate fits nothing');

	const panelsRaw = array(obj.panels, `${context}.panels`);
	if (panelsRaw.length !== DEPLOYED_BASELINE_PANEL_KEYS.length) {
		fail(`${context}.panels`, `expected ${DEPLOYED_BASELINE_PANEL_KEYS.length} panels`);
	}
	const panels = panelsRaw.map(parsePanel);
	// Every family the figure declares has to be one some panel actually drew,
	// or the legend describes a thing the reader cannot find.
	const drawnFamilies = new Set(panels.flatMap((p) => p.incumbentVariants.map((v) => v.family)));
	for (const family of families) {
		if (!drawnFamilies.has(family.key)) {
			fail(`${context}.incumbent_families`, `${family.key} is declared but no panel sources it`);
		}
	}

	const servedBeliefIdentity = parseServedIdentity(
		obj.served_belief_identity,
		`${context}.served_belief_identity`,
		new Map(panels.map((p) => [p.key, p.display]))
	);

	const rc = `${context}.replication`;
	const rep = record(obj.replication, rc);
	const deltas = panels.map((p) => p.deltaAuroc);
	const deltaMin = finite(rep.delta_min, `${rc}.delta_min`);
	const deltaMax = finite(rep.delta_max, `${rc}.delta_max`);
	close(deltaMin, Math.min(...deltas), `${rc}.delta_min`, 'must be the minimum panel delta');
	close(deltaMax, Math.max(...deltas), `${rc}.delta_max`, 'must be the maximum panel delta');
	const nFavoring = nonNegativeInteger(rep.n_panels_favoring_gate, `${rc}.n_panels_favoring_gate`);
	if (nFavoring !== panels.filter((p) => p.deltaFavorsGate).length) {
		fail(`${rc}.n_panels_favoring_gate`, 'must be the count it reports, not a claim');
	}
	const nExcludes = nonNegativeInteger(rep.n_panels_ci_excludes_zero, `${rc}.n_panels_ci_excludes_zero`);
	if (nExcludes !== panels.filter((p) => p.bootstrap.ci95Low > 0).length) {
		fail(`${rc}.n_panels_ci_excludes_zero`, 'must be the count it reports, not a claim');
	}
	const nBeatsEvery = nonNegativeInteger(rep.n_panels_gate_beats_every_variant, `${rc}.n_panels_gate_beats_every_variant`);
	if (nBeatsEvery !== panels.filter((p) => p.gateBeatsEveryVariant).length) {
		fail(`${rc}.n_panels_gate_beats_every_variant`, 'must be the count it reports, not a claim');
	}
	const nVariants = positiveInteger(rep.n_incumbent_variants_total, `${rc}.n_incumbent_variants_total`);
	if (nVariants !== panels.reduce((sum, p) => sum + p.incumbentVariants.length, 0)) {
		fail(`${rc}.n_incumbent_variants_total`, 'must be the count it reports, not a claim');
	}
	const nVariantsExcl = nonNegativeInteger(
		rep.n_incumbent_variants_ci_excludes_zero,
		`${rc}.n_incumbent_variants_ci_excludes_zero`
	);
	if (nVariantsExcl !== panels.reduce((sum, p) => sum + p.nVariantsCiExcludesZero, 0)) {
		fail(`${rc}.n_incumbent_variants_ci_excludes_zero`, 'must be the count it reports, not a claim');
	}
	const largest = panels.reduce((best, p) => (p.nStatements > best.nStatements ? p : best));
	if (rep.largest_panel_key !== largest.key) {
		fail(`${rc}.largest_panel_key`, 'must name the panel with the most statements');
	}
	const largestAtTop = boolean(rep.largest_panel_is_at_top_of_range, `${rc}.largest_panel_is_at_top_of_range`);
	if (largestAtTop !== (largest.deltaAuroc === Math.max(...deltas))) {
		fail(`${rc}.largest_panel_is_at_top_of_range`, 'must be the comparison it reports, not a claim');
	}
	const costMax = finite(rep.selection_cost_auroc_max, `${rc}.selection_cost_auroc_max`);
	close(
		costMax,
		Math.max(...panels.map((p) => p.selectionCostAuroc)),
		`${rc}.selection_cost_auroc_max`,
		'must be the largest margin the argmax rule forfeits'
	);

	// The evidence-regime span. The figure prints "the result holds across a
	// 16-fold difference in evidence density", so the span must be the panels'
	// own extremes and the fold must be their ratio — otherwise the strongest
	// sentence on the page rests on a number nothing checks.
	const readMeans = panels.map((p) => p.heterogeneity.evidenceReadsPerStatement.mean);
	const singleShares = panels.map((p) => p.heterogeneity.evidenceReadsPerStatement.shareSingle);
	const readMin = finite(rep.reads_per_statement_mean_min, `${rc}.reads_per_statement_mean_min`);
	const readMax = finite(rep.reads_per_statement_mean_max, `${rc}.reads_per_statement_mean_max`);
	close(readMin, Math.min(...readMeans), `${rc}.reads_per_statement_mean_min`, 'must be the thinnest panel’s mean');
	close(readMax, Math.max(...readMeans), `${rc}.reads_per_statement_mean_max`, 'must be the densest panel’s mean');
	if (!(readMin > 0)) fail(`${rc}.reads_per_statement_mean_min`, 'a fold span needs a positive denominator');
	const foldSpan = finite(rep.evidence_regime_fold_span, `${rc}.evidence_regime_fold_span`);
	close(foldSpan, readMax / readMin, `${rc}.evidence_regime_fold_span`, 'must be the ratio of the two extremes');
	const singleMin = unit(rep.share_single_evidence_min, `${rc}.share_single_evidence_min`);
	const singleMax = unit(rep.share_single_evidence_max, `${rc}.share_single_evidence_max`);
	close(singleMin, Math.min(...singleShares), `${rc}.share_single_evidence_min`, 'must be the lowest panel share');
	close(singleMax, Math.max(...singleShares), `${rc}.share_single_evidence_max`, 'must be the highest panel share');

	const caveats = array(obj.caveats, `${context}.caveats`).map((entry, i) =>
		text(entry, `${context}.caveats[${i}]`)
	);
	if (caveats.length !== DEPLOYED_BASELINE_CAVEAT_COUNT) {
		fail(`${context}.caveats`, `expected ${DEPLOYED_BASELINE_CAVEAT_COUNT} caveats`);
	}
	// Positional twins: a reissued artifact that reorders or rewrites a caveat gates
	// the figure rather than printing restatement N under caveat N+1.
	const caveatProse = pairShippedProse(
		caveats,
		DEPLOYED_BASELINE_CAVEAT_TWINS,
		`${context}.caveats`
	);
	const claimProse: ShippedProse = {
		shipped: text(obj.claim, `${context}.claim`),
		plain: DEPLOYED_BASELINE_PLAIN.claim
	};
	const claimIsNotProse: ShippedProse = {
		shipped: text(obj.claim_is_not, `${context}.claim_is_not`),
		plain: DEPLOYED_BASELINE_PLAIN.claimIsNot
	};
	const selectionRuleProse: ShippedProse = {
		shipped: text(obj.incumbent_selection_rule, `${context}.incumbent_selection_rule`),
		plain: DEPLOYED_BASELINE_PLAIN.incumbentSelectionRule
	};
	const selectionRuleCostProse: ShippedProse = {
		shipped: text(obj.incumbent_selection_rule_cost, `${context}.incumbent_selection_rule_cost`),
		plain: DEPLOYED_BASELINE_PLAIN.incumbentSelectionRuleCost
	};

	const gateObj = record(armsObj.gate, `${context}.arms.gate`);
	// The four header sentences the SVG `<desc>` opens with. Every one of them was
	// reaching a screen-reader user in the artifact's own wording, behind nothing.
	const questionProse: ShippedProse = {
		shipped: text(obj.question, `${context}.question`),
		plain: DEPLOYED_BASELINE_PLAIN.question
	};
	const metricSourceProse: ShippedProse = {
		shipped: text(obj.metric_source, `${context}.metric_source`),
		plain: DEPLOYED_BASELINE_PLAIN.metricSource
	};
	const noisyOrFormulaProse: ShippedProse = {
		shipped: text(obj.noisy_or_formula, `${context}.noisy_or_formula`),
		plain: DEPLOYED_BASELINE_PLAIN.noisyOrFormula
	};
	const notZeroShotProse: ShippedProse = {
		shipped: text(gateObj.not_zero_shot, `${context}.arms.gate.not_zero_shot`),
		plain: DEPLOYED_BASELINE_PLAIN.gateNotZeroShot
	};

	return {
		metric: DEPLOYED_BASELINE_METRIC,
		metricSource: metricSourceProse.shipped,
		positiveClass: DEPLOYED_BASELINE_POSITIVE_CLASS,
		noisyOrFormula: noisyOrFormulaProse.shipped,
		question: questionProse.shipped,
		figureTitle: text(obj.figure_title, `${context}.figure_title`),
		claim: claimProse.shipped,
		claimIsNot: claimIsNotProse.shipped,
		families,
		arms: {
			gate: {
				...gateArm,
				notZeroShot: notZeroShotProse.shipped,
				notZeroShotProse
			},
			research: researchArm
		},
		incumbentSelectionRule: selectionRuleProse.shipped,
		incumbentSelectionRuleCost: selectionRuleCostProse.shipped,
		servedBeliefIdentity,
		panels,
		replication: {
			nPanels: positiveInteger(rep.n_panels, `${rc}.n_panels`),
			nPanelsFavoringGate: nFavoring,
			nPanelsCiExcludesZero: nExcludes,
			nPanelsGateBeatsEveryVariant: nBeatsEvery,
			nIncumbentVariantsTotal: nVariants,
			nIncumbentVariantsCiExcludesZero: nVariantsExcl,
			deltaMin,
			deltaMax,
			largestPanelKey: largest.key,
			largestPanelDelta: finite(rep.largest_panel_delta, `${rc}.largest_panel_delta`),
			largestPanelIsAtTopOfRange: largestAtTop,
			selectionCostAurocMax: costMax,
			readsPerStatementMeanMin: readMin,
			readsPerStatementMeanMax: readMax,
			evidenceRegimeFoldSpan: foldSpan,
			shareSingleEvidenceMin: singleMin,
			shareSingleEvidenceMax: singleMax
		},
		caveats,
		prose: {
			claim: claimProse,
			claimIsNot: claimIsNotProse,
			incumbentSelectionRule: selectionRuleProse,
			incumbentSelectionRuleCost: selectionRuleCostProse,
			caveats: caveatProse,
			question: questionProse,
			metricSource: metricSourceProse,
			noisyOrFormula: noisyOrFormulaProse,
			gateWhatItIs: gateArm.whatItIsProse,
			gateNotZeroShot: notZeroShotProse,
			researchWhatItIs: researchArm.whatItIsProse,
			servedIdentityQuestion: servedBeliefIdentity.questionProse,
			servedIdentityFinding: servedBeliefIdentity.findingProse,
			servedIdentityFloorDerivation: servedBeliefIdentity.floorDerivationProse,
			// The SAME objects the family rows carry, in artifact order.
			familyWhatItComputes: families.map((family) => family.whatItComputesProse),
			familyWhereItRuns: families.map((family) => family.whereItRunsProse)
		},
		generatedBy: text(obj.generated_by, `${context}.generated_by`)
	};
}

// ---------------------------------------------------------------------------
// figure
// ---------------------------------------------------------------------------

export interface DeployedBaselineMark {
	series: DeployedBaselineSeriesKey;
	/** Distinguishes the declined ticks from one another in the DOM key. */
	id: string;
	x: number;
	auroc: number;
	/** Alt text only — never counted against the page's prose budget. */
	title: string;
}

export interface DeployedBaselineLegendEntry {
	series: DeployedBaselineSeriesKey;
	text: string;
	y: number;
}

export interface DeployedBaselineRow {
	key: string;
	/**
	 * The panel's on-screen name. Called `display`, not `label`, because a field
	 * named `label` on this page is a FROZEN JOIN KEY and must never reach a
	 * render position — `scripts/test-paper-render-invariants.mjs` enforces that
	 * by name. The three gutter strings below keep their `…Label` names: they are
	 * built here from `display` fields, never from keys, and the rule's regex
	 * (`/\.\s*(label|labels|armLabel|gateLabel)\b/`) does not match them.
	 */
	display: string;
	subLabel: string;
	/** The comparator, named ON the row: "vs <the form of INDRA belief drawn>". */
	comparatorLabel: string;
	/** The panel's composition, printed under its own bar. */
	heteroLabel: string;
	/** True for the panel with the most statements. */
	largest: boolean;
	/** Empty unless the row carries a chip (largest panel / in-sample fit). */
	chip: string;
	/** True where the panel fitted something of ours; the row says so. */
	inSample: boolean;
	y: number;
	/** Left end of the drawn advance (the incumbent), in user units. */
	trackFrom: number;
	/** Right end of the drawn advance (the gate), in user units. */
	trackTo: number;
	/** Left end of the forfeited margin (the weakest sourceable form). */
	forfeitFrom: number;
	/** True when the argmax actually gave something up — i.e. > 1 variant. */
	hasForfeit: boolean;
	marks: DeployedBaselineMark[];
	deltaReadout: string;
	ciReadout: string;
	ruleCostReadout: string;
	panel: DeployedBaselinePanel;
}

export interface DeployedBaselineFigure {
	rows: DeployedBaselineRow[];
	domainMin: number;
	domainMax: number;
	ticks: number[];
	height: number;
	/** x of the chance rule (AUROC 0.5), so the axis floor is drawable. */
	chanceX: number;
	/** Budget-checked gutter text, so the component types no label of its own. */
	axisTitle: string;
	readoutTitle: readonly [string, string];
	legend: DeployedBaselineLegendEntry[];
	/** The budget-checked on-figure title. */
	title: string;
	/** The three header lines, all derived from the shipped counts. */
	headline: string;
	subheadline: string;
	/**
	 * The evidence-regime span. It is a header line and not a caption because it
	 * is half the claim: four wins mean much less if the four panels are four
	 * copies of one regime, and this line is the figure saying — in its own
	 * counted numbers — that they are not.
	 */
	regimeLine: string;
	data: DeployedBaseline;
}

export interface DeployedBaselineOk {
	status: 'ok';
	reason: null;
	artifact_path: string;
	artifact_sha256: string;
	figure: DeployedBaselineFigure;
}

export interface DeployedBaselineUnavailable {
	status: 'unavailable';
	reason: string;
	artifact_path: string;
	artifact_sha256: string | null;
	figure: null;
}

export type DeployedBaselineLoad = DeployedBaselineOk | DeployedBaselineUnavailable;

/** Three decimals: the resolution the figure's separations are legible at. */
export function fmt3(value: number): string {
	return value.toFixed(3);
}

/** A signed change in AUROC, three decimals, sign always present. */
export function signed3(value: number): string {
	return `${value < 0 ? '−' : '+'}${Math.abs(value).toFixed(3)}`;
}

/** A per-statement count, one decimal — the resolution 1.2 vs 19.8 needs. */
export function fmt1(value: number): string {
	return value.toFixed(1);
}

/** A share of the panel as whole percent. */
export function pct0(value: number): string {
	return `${Math.round(value * 100)}%`;
}


/**
 * The census chip under each row label, built from the panel's own fields.
 * Fail-closed: every component is required, so a dropped count gates the figure
 * rather than rendering a shorter, friendlier, wrong sub-label.
 */
function subLabelOf(panel: DeployedBaselinePanel): string {
	return `${panel.nStatements} statements · ${panel.nCorrect}/${panel.nErrors} correct/error`;
}

/**
 * THE FIX for "the same comparison four times". Every row names the form of
 * INDRA's own belief it was measured against, in the gutter, where it cannot be
 * missed and cannot be confused with its neighbour.
 */
function comparatorLabelOf(panel: DeployedBaselinePanel): string {
	return `vs ${panel.incumbent.display}`;
}

/**
 * The panel's composition, printed under its own bar: how much evidence the
 * reader got, how much the incumbent was scored over, who curated it, how the
 * classes fall, whether it is in sample, and how it joined. Every field is a
 * computed number from the artifact — none is an adjective.
 */
function heteroLabelOf(panel: DeployedBaselinePanel): string {
	const h = panel.heterogeneity;
	const parts = [`${fmt1(h.evidenceReadsPerStatement.mean)} read/statement`];
	// The share the mean hides, and the reason these four rows are not one
	// comparison: where it is high the gate has nothing to aggregate and its
	// decision is a keep-or-drop on a single sentence.
	parts.push(`${pct0(h.evidenceReadsPerStatement.shareSingle)} single-evidence`);
	if (h.readerSawFullEvidence) parts.push('same evidence as INDRA’s scorer');
	else if (h.corpusEvidencePerStatement !== null) {
		parts.push(`INDRA’s scorer sees ${fmt1(h.corpusEvidencePerStatement.mean)} evidence`);
	} else parts.push('INDRA’s scorer sees all of it');

	if (h.nCurators !== null) parts.push(`${h.nCurators} curators`);
	else if (panel.isPaperPanel) parts.push('2023 released labels');
	else parts.push('held out');

	parts.push(
		h.balancedByConstruction
			? `${pct0(h.baseRateCorrect)} correct, balanced`
			: `${pct0(h.baseRateCorrect)} correct`
	);
	parts.push(h.outOfSample ? 'out-of-sample' : 'in-sample fit');
	// The join SUMMARY, computed from the join's own row counts in the emitting
	// script. Never inferred from `joinMode`: that string names a strategy that
	// always mentions the source-hash fallback, so parsing it here would label
	// three exact joins as fallbacks.
	parts.push(h.joinSummary);
	return parts.join(' · ');
}

/**
 * What the strongest-incumbent rule cost this row, printed beside the delta so
 * the rule's price is legible rather than buried in a method note. A panel that
 * can only source ONE form of INDRA belief forfeits nothing, and says so.
 */
function ruleCostReadoutOf(panel: DeployedBaselinePanel): string {
	if (panel.incumbentVariants.length < 2) return 'sole form sourced';
	return `rule cost ${fmt3(panel.selectionCostAuroc)}`;
}

/**
 * Place the four panels on one shared AUROC axis.
 *
 * ROW ORDER is by statement count DESCENDING. That is a LAYOUT decision made
 * here, not a claim about the data: it puts the largest and cleanest panel — the
 * paper's own — first, which is where an author's eye starts. It is NOT ordered
 * by delta, because ordering by the thing being claimed would manufacture a
 * trend out of a layout.
 *
 * THROWS on a degenerate axis, an over-budget label, or any mark that would fall
 * outside the plot, so the caller gates the panel to `unavailable` rather than
 * drawing clipped text, a squashed scale, or a mark in the margin.
 */
export function buildDeployedBaselineFigure(data: DeployedBaseline): DeployedBaselineFigure {
	const g = DEPLOYED_BASELINE_GEOMETRY;
	const drawn: number[] = [];
	for (const panel of data.panels) {
		drawn.push(panel.gate.auroc, ...panel.incumbentVariants.map((v) => v.auroc));
		if (panel.researchModel) drawn.push(panel.researchModel.auroc);
	}
	const hi = Math.max(...drawn);
	if (!Number.isFinite(hi)) fail('domain', 'no marks to scale');
	// Integer hundredths throughout: float accumulation would drift the ticks.
	const maxH = Math.min(100, Math.ceil(hi * 100) + 1);
	const minH = Math.round(DEPLOYED_BASELINE_AXIS_MIN * 100);
	if (maxH - minH < 4) fail('domain', 'degenerate axis range');
	const domainMin = minH / 100;
	const domainMax = maxH / 100;
	const span = g.plotRight - g.plotLeft;
	const x = (value: number) => g.plotLeft + ((value - domainMin) / (domainMax - domainMin)) * span;

	const ticks: number[] = [];
	for (let h = minH; h <= maxH; h += 5) ticks.push(h / 100);

	const ordered = [...data.panels].sort(
		(a, b) => b.nStatements - a.nStatements || a.key.localeCompare(b.key)
	);
	const largestKey = data.replication.largestPanelKey;

	const rows = ordered.map((panel, index) => {
		const y = g.rowsTop + index * g.rowHeight;
		const marks: DeployedBaselineMark[] = [];
		// The forms the argmax DECLINED, drawn first so the selected mark sits on
		// top of them if they ever coincide.
		for (const variant of panel.incumbentVariants) {
			if (variant.key === panel.incumbent.key) continue;
			marks.push({
				series: 'declined',
				id: `declined:${variant.key}`,
				x: x(variant.auroc),
				auroc: variant.auroc,
				title:
					`${variant.display} — another form of INDRA's own belief on ${panel.display}, ` +
					`AUROC ${fmt3(variant.auroc)}. Not drawn as the comparator because the ` +
					`rule picks the strongest of those forms, ${panel.incumbent.display} at ${fmt3(panel.incumbent.auroc)}. ` +
					`The gate beats this form by ${signed3(variant.deltaAuroc)} ` +
					`[${signed3(variant.bootstrap.ci95Low)}, ${signed3(variant.bootstrap.ci95High)}].`
			});
		}
		marks.push({
			series: 'incumbent',
			id: 'incumbent',
			x: x(panel.incumbent.auroc),
			auroc: panel.incumbent.auroc,
			title:
				`${panel.incumbent.display} — the strongest form of INDRA's own belief ${panel.display} can ` +
				`source — AUROC ${fmt3(panel.incumbent.auroc)}. ${panel.incumbent.selectedBy}.`
		});
		marks.push({
			series: 'gate',
			id: 'gate',
			x: x(panel.gate.auroc),
			auroc: panel.gate.auroc,
			title:
				`${panel.gate.display} — INDRA's own scorer over reader-kept evidence — ` +
				`AUROC ${fmt3(panel.gate.auroc)} on ${panel.display}`
		});
		if (panel.researchModel) {
			marks.push({
				series: 'research',
				id: 'research',
				x: x(panel.researchModel.auroc),
				auroc: panel.researchModel.auroc,
				title:
					`${panel.researchModel.display} — the random forest fitted for the 2023 paper, never deployed — ` +
					`AUROC ${fmt3(panel.researchModel.auroc)}`
			});
		}
		// Nothing may be drawn outside the plot. A mark in the margin reads as a
		// mark at the edge of the scale, which is a different number.
		for (const mark of marks) {
			if (mark.x < g.plotLeft - 1e-6 || mark.x > g.plotRight + 1e-6) {
				fail(`row[${panel.key}].marks`, `${mark.id} at AUROC ${mark.auroc} falls outside the plot`);
			}
		}

		const chip = panel.key === largestKey ? 'largest set of statements' : panel.inSampleNote !== null ? 'in-sample fit' : '';

		return {
			key: panel.key,
			display: budget(panel.display, DEPLOYED_BASELINE_LABEL_BUDGET_CHARS, `row[${panel.key}].display`),
			subLabel: budget(
				subLabelOf(panel),
				DEPLOYED_BASELINE_SUBLABEL_BUDGET_CHARS,
				`row[${panel.key}].subLabel`
			),
			comparatorLabel: budget(
				comparatorLabelOf(panel),
				DEPLOYED_BASELINE_COMPARATOR_BUDGET_CHARS,
				`row[${panel.key}].comparatorLabel`
			),
			heteroLabel: budget(
				heteroLabelOf(panel),
				DEPLOYED_BASELINE_HETERO_BUDGET_CHARS,
				`row[${panel.key}].heteroLabel`
			),
			largest: panel.key === largestKey,
			chip: chip === '' ? '' : budget(chip, DEPLOYED_BASELINE_CHIP_BUDGET_CHARS, `row[${panel.key}].chip`),
			inSample: panel.inSampleNote !== null,
			y,
			trackFrom: x(panel.incumbent.auroc),
			trackTo: x(panel.gate.auroc),
			forfeitFrom: x(panel.weakestVariantAuroc),
			hasForfeit: panel.incumbentVariants.length > 1,
			marks,
			deltaReadout: budget(
				signed3(panel.deltaAuroc),
				DEPLOYED_BASELINE_READOUT_BUDGET_CHARS,
				`row[${panel.key}].deltaReadout`
			),
			ciReadout: budget(
				`[${signed3(panel.bootstrap.ci95Low)}, ${signed3(panel.bootstrap.ci95High)}]`,
				DEPLOYED_BASELINE_READOUT_BUDGET_CHARS,
				`row[${panel.key}].ciReadout`
			),
			ruleCostReadout: budget(
				ruleCostReadoutOf(panel),
				DEPLOYED_BASELINE_READOUT_BUDGET_CHARS,
				`row[${panel.key}].ruleCostReadout`
			),
			panel
		};
	});

	// The legend is DATA so it is budget-checked like every other gutter string.
	// A static legend in the component is a string no budget can see, which is
	// how "gate − deployed, 95% CI" shipped clipped.
	const legendText: Record<DeployedBaselineSeriesKey, string> = {
		incumbent: 'strongest form of INDRA’s own belief this set of statements can source',
		gate: 'the same INDRA scorer over reader-kept evidence',
		research: 'random forest from the 2023 paper — research model, never shipped',
		declined: 'other forms of INDRA’s belief, all weaker, all beaten'
	};
	const legend: DeployedBaselineLegendEntry[] = (
		['incumbent', 'gate', 'research', 'declined'] as DeployedBaselineSeriesKey[]
	).map((series, index) => ({
		series,
		text: budget(legendText[series], DEPLOYED_BASELINE_LEGEND_BUDGET_CHARS, `legend[${series}]`),
		y: g.legendY + index * g.legendRowHeight
	}));
	// The legend's last baseline must clear the first row, or it draws on top of
	// the panel it is describing.
	const legendLastY = g.legendY + (legend.length - 1) * g.legendRowHeight;
	if (legendLastY >= g.rowsTop) {
		fail('legend', 'the legend overruns the first row');
	}
	// The regime line shares the header band with the legend and must sit between
	// the legend's last baseline and the first row. Adding a legend entry pushes
	// `legendLastY` down; without this the new entry would silently land on the
	// regime line, which is the class of failure a character budget cannot see.
	if (!(g.regimeLineY > legendLastY && g.regimeLineY < g.rowsTop)) {
		fail('regimeLine', 'the regime line does not fit between the legend and the first row');
	}

	const rep = data.replication;
	return {
		rows,
		domainMin,
		domainMax,
		ticks,
		height: g.rowsTop + rows.length * g.rowHeight + g.axisPad,
		chanceX: x(DEPLOYED_BASELINE_AXIS_MIN),
		axisTitle: budget(
			data.metric.toUpperCase(),
			DEPLOYED_BASELINE_AXIS_TITLE_BUDGET_CHARS,
			'axisTitle'
		),
		readoutTitle: [
			budget(DEPLOYED_BASELINE_READOUT_TITLE[0], DEPLOYED_BASELINE_READOUT_BUDGET_CHARS, 'readoutTitle[0]'),
			budget(DEPLOYED_BASELINE_READOUT_TITLE[1], DEPLOYED_BASELINE_READOUT_BUDGET_CHARS, 'readoutTitle[1]')
		],
		legend,
		title: budget(data.figureTitle, DEPLOYED_BASELINE_TITLE_BUDGET_CHARS, 'title'),
		// Both header lines are counts, not adjectives: they change when the data
		// changes and they cannot overstate what the panels show.
		headline: budget(
			`${rep.nPanels} statement sets · ${rep.nIncumbentVariantsTotal} forms of INDRA’s own belief · gate wins ${rep.nIncumbentVariantsCiExcludesZero}/${rep.nIncumbentVariantsTotal}`,
			DEPLOYED_BASELINE_HEADER_BUDGET_CHARS,
			'headline'
		),
		subheadline: budget(
			`${rep.nPanelsCiExcludesZero}/${rep.nPanels} headline intervals exclude zero · argmax forfeits up to ${fmt3(rep.selectionCostAurocMax)} ${data.metric.toUpperCase()}`,
			DEPLOYED_BASELINE_HEADER_BUDGET_CHARS,
			'subheadline'
		),
		regimeLine: budget(
			`evidence density spans ${rep.evidenceRegimeFoldSpan.toFixed(1)}× (${fmt1(rep.readsPerStatementMeanMin)}–${fmt1(rep.readsPerStatementMeanMax)} read/statement · ${pct0(rep.shareSingleEvidenceMin)}–${pct0(rep.shareSingleEvidenceMax)} single-evidence)`,
			DEPLOYED_BASELINE_REGIME_BUDGET_CHARS,
			'regimeLine'
		),
		data
	};
}
