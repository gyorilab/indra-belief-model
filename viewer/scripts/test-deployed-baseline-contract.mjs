/**
 * Pure-function assertions for the deployed-baseline replication data contract.
 *
 * A NEW runner rather than another tail on `test-paper-literal-contract.mjs`:
 * that file is shared with concurrently-running nodes, and two nodes appending
 * to the same tail collide. Everything here follows the same idiom — eq/ok
 * counters, a fixture builder, mutate-and-expect-throw helpers, exit 1 on any
 * failure — so the runners read the same way.
 *
 * Covers:
 *   · a well-formed synthetic fixture validates and lays out;
 *   · the SHIPPED bytes validate, and their sha256 matches the run manifest's
 *     `output_sha256` entry;
 *   · the CLAIM the figure exists to make survives the shipped data — both
 *     incumbent families are marked deployed, the library default unfitted and
 *     the stored production belief fitted, the paper's RF fitted and NOT
 *     deployed, and every panel's delta favours the gate against EVERY form of
 *     INDRA belief it can source, not just the one drawn;
 *   · the two-family split is EARNED: `served_belief_identity` must find stored
 *     beliefs below the SimpleScorer floor, or the figure fails closed rather
 *     than drawing a distinction the data no longer supports;
 *   · the strongest-incumbent rule: an artifact that draws anything other than
 *     its own strongest sourceable incumbent fails CLOSED, and the price of the
 *     rule is an arithmetic identity, not a sentence;
 *   · the evidence-matched control is never an incumbent, and a panel must
 *     carry either the control or the stated reason it has none;
 *   · the panel heterogeneity is internally consistent — the share equals the
 *     ratio of the two censuses, and the "reader saw everything" flag cannot
 *     disagree with it;
 *   · the paper's RF may be drawn on the paper's panel and nowhere else;
 *   · row order is statement count descending and the largest panel is first;
 *   · every label, sub-label, comparator, composition strip, chip, legend line
 *     and readout the builder emits fits its measured character budget, an
 *     over-budget label gates the figure, and no mark lands outside the plot;
 *   · every arithmetic identity the geometry rests on fails CLOSED when broken.
 */
import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';

import {
	DEPLOYED_BASELINE_ARM_KEYS,
	DEPLOYED_BASELINE_ARTIFACT_KIND,
	DEPLOYED_BASELINE_AXIS_MIN,
	DEPLOYED_BASELINE_AXIS_TITLE_BUDGET_CHARS,
	DEPLOYED_BASELINE_CAVEAT_COUNT,
	DEPLOYED_BASELINE_CHIP_BUDGET_CHARS,
	DEPLOYED_BASELINE_COMPARATOR_BUDGET_CHARS,
	DEPLOYED_BASELINE_FAMILY_KEYS,
	DEPLOYED_BASELINE_GEOMETRY,
	DEPLOYED_BASELINE_HEADER_BUDGET_CHARS,
	DEPLOYED_BASELINE_HETERO_BUDGET_CHARS,
	DEPLOYED_BASELINE_LABEL_BUDGET_CHARS,
	DEPLOYED_BASELINE_PANEL_DISPLAY,
	DEPLOYED_BASELINE_LEGEND_BUDGET_CHARS,
	DEPLOYED_BASELINE_METRIC,
	DEPLOYED_BASELINE_PANEL_KEYS,
	DEPLOYED_BASELINE_PAPER_PANEL_KEY,
	DEPLOYED_BASELINE_POSITIVE_CLASS,
	DEPLOYED_BASELINE_READOUT_BUDGET_CHARS,
	DEPLOYED_BASELINE_REGIME_BUDGET_CHARS,
	DEPLOYED_BASELINE_CAVEAT_ANCHORS,
	DEPLOYED_BASELINE_PROSE_ANCHORS,
	DEPLOYED_BASELINE_SCHEMA_VERSION,
	DEPLOYED_BASELINE_SUBLABEL_BUDGET_CHARS,
	DEPLOYED_BASELINE_TITLE_BUDGET_CHARS,
	buildDeployedBaselineFigure,
	fmt1,
	pct0,
	signed3,
	validateDeployedBaseline
} from '../src/lib/data/paper-deployed-baseline.ts';

let failures = 0;

function eq(got, want, label) {
	if (got !== want) {
		failures++;
		console.error(`FAIL ${label}: got ${JSON.stringify(got)}, want ${JSON.stringify(want)}`);
	}
}

function ok(got, label) {
	if (!got) {
		failures++;
		console.error(`FAIL ${label}`);
	}
}

function close(got, want, label) {
	if (!(Math.abs(got - want) <= 1e-9)) {
		failures++;
		console.error(`FAIL ${label}: got ${got}, want ${want}`);
	}
}

/** Mutate a deep clone and require validation (or layout) to throw. */
function throws(build, label) {
	try {
		build();
	} catch {
		return;
	}
	failures++;
	console.error(`FAIL ${label}: expected a throw, got none`);
}

function clone(value) {
	return JSON.parse(JSON.stringify(value));
}

// ---------------------------------------------------------------------------
// FIXTURE. Synthetic AUROCs; every delta, count, census and range is COMPUTED
// from them, so the fixture can never encode an identity it does not satisfy.
// ---------------------------------------------------------------------------
const LIB = DEPLOYED_BASELINE_FAMILY_KEYS.library;
const SERVED = DEPLOYED_BASELINE_FAMILY_KEYS.served;

const FIXTURE_PANELS = [
	{
		key: 'indra_paper_2023',
		display: 'the paper’s own panel',
		n: 1689,
		correct: 1237,
		variants: [
			['simple_scorer_direct', 'INDRA SimpleScorer, direct', LIB, false, 0.774],
			['simple_scorer_hierarchy', 'INDRA SimpleScorer + hierarchy', LIB, false, 0.7823],
			['cogex_fitted_hybrid', 'INDRA’s stored belief (fitted Hybrid)', SERVED, true, 0.8272]
		],
		gate: 0.901,
		research: 0.8516,
		curators: null,
		reads: [7, 759, 1, 33361, 338],
		corpus: [7, 759, 1, 33361, 338],
		readerSawFull: true,
		control: null
	},
	{
		key: 'eval_curation_v1',
		display: 'two of the paper’s authors',
		n: 913,
		correct: 476,
		variants: [
			['simple_scorer_recomputed', 'INDRA SimpleScorer, full evidence', LIB, false, 0.7364],
			['indra_served_belief', 'INDRA’s stored belief', SERVED, true, 0.7102]
		],
		gate: 0.7792,
		research: null,
		curators: 2,
		reads: [1, 8, 1, 1606, 570],
		corpus: [12, 771, 1, 29537, 122],
		readerSawFull: false,
		control: 0.4343
	},
	{
		key: 'external_curator_v1',
		display: '32 external curators',
		n: 464,
		correct: 234,
		variants: [['indra_served_belief', 'INDRA’s stored belief', SERVED, true, 0.6882]],
		gate: 0.801,
		research: null,
		curators: 32,
		reads: [1, 6, 1, 580, 392],
		corpus: null,
		readerSawFull: false,
		control: 0.4791
	},
	{
		key: 'holdout_cc',
		display: 'out-of-distribution holdout',
		n: 414,
		correct: 186,
		variants: [['indra_served_belief', 'INDRA’s stored belief', SERVED, true, 0.61]],
		gate: 0.7202,
		research: null,
		curators: null,
		reads: [1, 4, 1, 500, 349],
		corpus: null,
		readerSawFull: false,
		control: 0.5139
	}
];

function bootstrapFixture(delta) {
	return {
		ci95_low: delta - 0.02,
		ci95_high: delta + 0.02,
		p_gate_greater: 1.0,
		n_valid_resamples: 10000,
		n_bootstrap: 10000,
		seed: 20260717
	};
}

/**
 * A per-statement census. The MEAN is derived here rather than supplied: the
 * validator requires `mean === total / n_statements`, so a fixture that carried
 * its own mean could disagree with its own total and still be "the fixture".
 * `nSingle` is the count of statements holding exactly one evidence — the
 * disclosure a mean cannot make.
 */
function spreadFixture([median, max, min, total, nSingle], nStatements) {
	return {
		mean: total / nStatements,
		median,
		max,
		min,
		total,
		n_statements: nStatements,
		n_single: nSingle,
		share_single: nSingle / nStatements
	};
}

/**
 * Every restatement this figure emits is PINNED to a verbatim fragment of the
 * sentence it was written for, so a fixture whose prose is pure placeholder text
 * would gate the loader for the right reason and leave this runner unable to
 * exercise anything past it. The anchors are exported from the module and spliced
 * in here, exactly as `DEPLOYED_BASELINE_CAVEAT_ANCHORS` already is: the fixture
 * stays synthetic, and the contract stays a single source of truth.
 */
const A = DEPLOYED_BASELINE_PROSE_ANCHORS;

function panelFixture(spec) {
	const anchors = A.panel[spec.key];
	const variants = spec.variants.map(([key, display, family, fitted, auroc]) => ({
		key,
		display,
		family,
		fitted,
		evidence_scope: 'statement_full',
		evidence_scope_note: 'the statement’s full evidence in the source corpus',
		source: `data/results/fixture/${key}.jsonl`,
		source_sha256: 'c'.repeat(64),
		auroc,
		average_precision: 0.9,
		delta_auroc: spec.gate - auroc,
		bootstrap: bootstrapFixture(spec.gate - auroc),
		what_it_computes: `fixture variant — ${A.variant[key]}.`,
		// Only the recomputed library-default arm has a shipped sibling that
		// scores it differently; the fixture carries that shape so the divergence
		// identities are exercised.
		cross_check:
			key === 'simple_scorer_recomputed'
				? {
						sibling: 'data/results/fixture_sibling.json',
						sibling_key: 'belief_discrimination.belief_indra.all.auroc',
						sibling_auroc: auroc - 0.0009,
						sibling_n: spec.n,
						this_auroc: auroc,
						delta_vs_sibling: 0.0009,
						n_statements_scored_differently: 199,
						why_it_differs: `fixture divergence — ${A.why_it_differs}.`
					}
				: null
	}));
	const strongest = variants.reduce((best, v) => (v.auroc > best.auroc ? v : best));
	const weakest = variants.reduce((worst, v) => (v.auroc < worst.auroc ? v : worst));
	const delta = spec.gate - strongest.auroc;
	const reads = spreadFixture(spec.reads, spec.n);
	const corpus = spec.corpus === null ? null : spreadFixture(spec.corpus, spec.n);
	return {
		key: spec.key,
		display: spec.display,
		is_paper_panel: spec.key === DEPLOYED_BASELINE_PAPER_PANEL_KEY,
		n_statements: spec.n,
		n_correct: spec.correct,
		n_errors: spec.n - spec.correct,
		positive_class: DEPLOYED_BASELINE_POSITIVE_CLASS,
		base_rate_correct: spec.correct / spec.n,
		balanced_by_construction: spec.curators !== null,
		n_curators: spec.curators,
		curator_note: `fixture curators — ${anchors.curator_note}`,
		out_of_sample: spec.key !== 'eval_curation_v1',
		in_sample_note:
			spec.key === 'eval_curation_v1'
				? `fixture in-sample — ${anchors.in_sample_note}.`
				: null,
		label_field: 'fixture label',
		label_note: `fixture label note — ${anchors.label_note}`,
		heterogeneity: {
			evidence_reads_per_statement: reads,
			corpus_evidence_per_statement: corpus,
			corpus_evidence_absent_because:
				corpus === null ? `fixture gold — ${anchors.corpus_evidence_absent_because}.` : null,
			reader_saw_full_evidence: spec.readerSawFull,
			reader_evidence_share_of_corpus: corpus === null ? null : reads.total / corpus.total,
			n_curators: spec.curators,
			curator_note: `fixture curators — ${anchors.curator_note}`,
			base_rate_correct: spec.correct / spec.n,
			balanced_by_construction: spec.curators !== null,
			out_of_sample: spec.key !== 'eval_curation_v1',
			in_sample_note:
				spec.key === 'eval_curation_v1'
					? `fixture in-sample — ${anchors.in_sample_note}.`
					: null,
			label_field: 'fixture label',
			label_note: `fixture label note — ${anchors.label_note}`,
			join_mode: `fixture join — ${anchors.join_mode}`,
			join_summary: spec.key === 'holdout_cc' ? 'source-hash join' : 'exact join',
			n_undefined_excluded: 0,
			incumbent_evidence_scope: strongest.evidence_scope,
			incumbent_evidence_scope_note: strongest.evidence_scope_note,
			gate_evidence_scope: 'reader_evidence_only',
			gate_evidence_scope_note: 'only the evidence the reader was shown'
		},
		gate: {
			key: DEPLOYED_BASELINE_ARM_KEYS.gate,
			display: 'Gemma 4 26B reader gate',
			auroc: spec.gate,
			average_precision: 0.94
		},
		incumbent: {
			key: strongest.key,
			display: strongest.display,
			family: strongest.family,
			fitted: strongest.fitted,
			auroc: strongest.auroc,
			average_precision: strongest.average_precision,
			selected_by: 'argmax auroc over incumbent_variants'
		},
		incumbent_variants: variants,
		n_incumbent_variants: variants.length,
		selection_cost_auroc: strongest.auroc - weakest.auroc,
		weakest_variant_key: weakest.key,
		weakest_variant_auroc: weakest.auroc,
		gate_beats_every_variant: variants.every((v) => v.delta_auroc > 0),
		n_variants_ci_excludes_zero: variants.filter((v) => v.bootstrap.ci95_low > 0).length,
		delta_auroc: delta,
		delta_favors_gate: delta > 0,
		bootstrap: bootstrapFixture(delta),
		research_model:
			spec.research === null
				? null
				: {
						key: DEPLOYED_BASELINE_ARM_KEYS.research,
						display: 'paper RF + Type/#PMIDs/promoter',
						source: 'data/results/fixture/literal.json',
						key_in_source: 'RF fixture',
						auroc: spec.research,
						average_precision: 0.941,
						delta_auroc_gate_minus_research: spec.gate - spec.research,
						delta_auroc_research_minus_incumbent: spec.research - strongest.auroc,
						bootstrap: bootstrapFixture(spec.gate - spec.research)
					},
		gate_sensitivity: null,
		evidence_matched_control:
			spec.control === null
				? null
				: {
						key: 'simple_scorer_reader_evidence',
						display: 'INDRA SimpleScorer, reader’s evidence only',
						evidence_scope: 'reader_evidence_only',
						evidence_scope_note: 'only the evidence the reader was shown',
						auroc: spec.control,
						average_precision: 0.5,
						delta_auroc_incumbent_minus_control: strongest.auroc - spec.control,
						at_or_below_chance: spec.control <= DEPLOYED_BASELINE_AXIS_MIN,
						is_an_incumbent: false,
						note: `fixture control — ${A.control_note}.`
					},
		evidence_matched_control_absent_because:
			spec.control === null
				? `the fixture paper panel is evidence-symmetric, so the asymmetry ${A.control_absent_because}.`
				: null,
		provenance: {
			gold: 'data/benchmark/fixture.jsonl',
			gold_sha256: 'a'.repeat(64),
			run: 'data/results/fixture_run.jsonl',
			run_sha256: 'b'.repeat(64),
			reader_model: 'fixture-gemma',
			join: `fixture join — ${anchors.join_mode}`
		}
	};
}

function artifactFixture() {
	const panels = FIXTURE_PANELS.map(panelFixture);
	const deltas = panels.map((p) => p.delta_auroc);
	const readMeans = panels.map((p) => p.heterogeneity.evidence_reads_per_statement.mean);
	const singleShares = panels.map((p) => p.heterogeneity.evidence_reads_per_statement.share_single);
	const largest = panels.reduce((best, p) => (p.n_statements > best.n_statements ? p : best));
	const servedPanels = panels.filter((p) => p.key !== DEPLOYED_BASELINE_PAPER_PANEL_KEY);
	const perPanel = servedPanels.map((p, index) => {
		const nBelow = [86, 226, 47][index];
		return {
			panel_key: p.key,
			n_served: p.n_statements,
			n_below_floor: nBelow,
			fraction_below_floor: nBelow / p.n_statements
		};
	});
	return {
		artifact_kind: DEPLOYED_BASELINE_ARTIFACT_KIND,
		schema_version: DEPLOYED_BASELINE_SCHEMA_VERSION,
		metric: DEPLOYED_BASELINE_METRIC,
		metric_source: 'fixture rank-sum AUROC',
		positive_class: DEPLOYED_BASELINE_POSITIVE_CLASS,
		noisy_or_formula: 'belief = 1 - PROD_s (syst_s + rand_s^{n_s})',
		question: 'fixture question?',
		figure_title: 'fixture title',
		claim: 'fixture claim. Second sentence.',
		claim_is_not: 'fixture anti-claim.',
		incumbent_families: [
			{
				key: LIB,
				display: 'INDRA’s library default',
				deployed: true,
				fitted: false,
				what_it_computes: `the unfitted noisy-OR. ${A.family.indra_library_default.what_it_computes}.`,
				where_it_runs: `anyone who ${A.family.indra_library_default.where_it_runs}`,
				ships_in: 'indra.belief.SimpleScorer'
			},
			{
				key: SERVED,
				display: 'the belief INDRA stored',
				deployed: true,
				fitted: true,
				what_it_computes: `the fitted export-pipeline belief; ${A.family.indra_production_served.what_it_computes}.`,
				where_it_runs: `db.indra.bio — what a CoGEx user ${A.family.indra_production_served.where_it_runs}`,
				ships_in: 'indra_db export_assembly.py'
			}
		],
		arms: {
			gate: {
				key: DEPLOYED_BASELINE_ARM_KEYS.gate,
				display: 'Gemma 4 26B reader gate',
				deployed: false,
				fitted: false,
				what_it_is: `the same scorer over reader-kept evidence; ${A.gate_what_it_is}.`,
				not_zero_shot: '14 demonstration pairs per call'
			},
			research_model: {
				key: DEPLOYED_BASELINE_ARM_KEYS.research,
				display: 'paper RF + Type/#PMIDs/promoter',
				deployed: false,
				fitted: true,
				what_it_is: `the paper’s fitted research model; it ${A.research_what_it_is}.`
			}
		},
		incumbent_selection_rule: 'fixture rule',
		incumbent_selection_rule_cost: 'fixture rule cost',
		served_belief_identity: {
			question: 'fixture identity question?',
			finding: 'fixture identity finding.',
			simple_scorer_floor: 0.6,
			floor_derivation: 'fixture derivation.',
			floor_source: 'indra/resources/default_belief_probs.json',
			floor_source_sha256: 'd'.repeat(64),
			per_panel: perPanel,
			n_served_below_floor: perPanel.reduce((sum, p) => sum + p.n_below_floor, 0),
			n_panels_with_served_below_floor: perPanel.filter((p) => p.n_below_floor > 0).length
		},
		evidence_scopes: { statement_full: 'fixture scope' },
		panels,
		replication: {
			n_panels: panels.length,
			n_panels_favoring_gate: panels.filter((p) => p.delta_favors_gate).length,
			n_panels_ci_excludes_zero: panels.filter((p) => p.bootstrap.ci95_low > 0).length,
			n_panels_gate_beats_every_variant: panels.filter((p) => p.gate_beats_every_variant).length,
			n_incumbent_variants_total: panels.reduce((sum, p) => sum + p.incumbent_variants.length, 0),
			n_incumbent_variants_ci_excludes_zero: panels.reduce(
				(sum, p) => sum + p.n_variants_ci_excludes_zero,
				0
			),
			delta_min: Math.min(...deltas),
			delta_max: Math.max(...deltas),
			largest_panel_key: largest.key,
			largest_panel_delta: largest.delta_auroc,
			largest_panel_is_at_top_of_range: largest.delta_auroc === Math.max(...deltas),
			selection_cost_auroc_max: Math.max(...panels.map((p) => p.selection_cost_auroc)),
			reads_per_statement_mean_min: Math.min(...readMeans),
			reads_per_statement_mean_max: Math.max(...readMeans),
			evidence_regime_fold_span: Math.max(...readMeans) / Math.min(...readMeans),
			share_single_evidence_min: Math.min(...singleShares),
			share_single_evidence_max: Math.max(...singleShares)
		},
		// Each fixture caveat carries the verbatim fragment the loader pins its plain
		// restatement to. A caveat the figure cannot restate is one it may not print,
		// so a fixture of bare placeholders is no longer a valid artifact.
		caveats: DEPLOYED_BASELINE_CAVEAT_ANCHORS.map(
			(anchor, i) => `fixture caveat ${i}: ${anchor}.`
		),
		generated_by: 'scripts/compute_deployed_baseline_replication.py'
	};
}

// ---------------------------------------------------------------------------
// 1. the fixture validates and lays out
// ---------------------------------------------------------------------------
const fixture = validateDeployedBaseline(artifactFixture());
eq(fixture.panels.length, DEPLOYED_BASELINE_PANEL_KEYS.length, 'fixture panel count');
eq(fixture.panels[0].key, DEPLOYED_BASELINE_PAPER_PANEL_KEY, 'fixture panel order is the frozen order');
close(fixture.panels[0].deltaAuroc, 0.901 - 0.8272, 'fixture paper delta is gate minus the STRONGEST incumbent');
eq(fixture.panels[0].incumbent.key, 'cogex_fitted_hybrid', 'fixture paper incumbent is the strongest variant');
eq(fixture.panels[0].incumbent.family, SERVED, 'the drawn incumbent carries the family it came from');
close(fixture.panels[0].selectionCostAuroc, 0.8272 - 0.774, 'the rule’s price is strongest minus weakest');
eq(fixture.families.length, 2, 'both families of INDRA belief are declared');

const laid = buildDeployedBaselineFigure(fixture);
eq(laid.rows.length, 4, 'four rows are laid out');
eq(laid.domainMin, DEPLOYED_BASELINE_AXIS_MIN, 'the axis is anchored at chance, never truncated');
ok(
	laid.domainMax > Math.max(...laid.rows.flatMap((r) => r.marks.map((m) => m.auroc))),
	'the domain clears every drawn mark'
);
eq(
	laid.rows.map((r) => r.panel.nStatements).join(','),
	[...fixture.panels].map((p) => p.nStatements).sort((a, b) => b - a).join(','),
	'rows are ordered by statement count descending'
);
eq(laid.rows[0].largest, true, 'the largest panel is the first row');
eq(laid.rows.filter((r) => r.largest).length, 1, 'exactly one row is marked largest');
eq(
	laid.rows.filter((r) => r.marks.some((m) => m.series === 'research')).length,
	1,
	'the paper’s RF is drawn on exactly one row'
);
eq(
	laid.rows.find((r) => r.marks.some((m) => m.series === 'research'))?.key,
	DEPLOYED_BASELINE_PAPER_PANEL_KEY,
	'and that row is the paper’s own panel'
);
eq(laid.rows.find((r) => r.key === 'eval_curation_v1')?.inSample, true, 'the in-sample flag travels with its row');
eq(laid.legend.length, 4, 'the legend names all four drawn series');

// THE FIX for "the same comparison four times": every row NAMES its comparator,
// and a row whose comparator differs from its neighbour must say so on its face.
for (const row of laid.rows) {
	ok(row.comparatorLabel.includes(row.panel.incumbent.display), `row names its comparator: ${row.key}`);
	ok(row.heteroLabel.length > 0, `row prints its composition: ${row.key}`);
	// Composition strip is fed by the CENSUS, not by adjectives.
	ok(
		row.heteroLabel.includes(fmt1(row.panel.heterogeneity.evidenceReadsPerStatement.mean)),
		`composition strip carries the evidence census: ${row.key}`
	);
	ok(
		row.heteroLabel.includes(pct0(row.panel.baseRateCorrect)),
		`composition strip carries the class balance: ${row.key}`
	);
}
ok(
	new Set(laid.rows.map((r) => r.comparatorLabel)).size > 1,
	'the comparators are NOT all the same — which is exactly why they are drawn'
);
// DISPLAY vs FROZEN JOIN KEY: no reader-facing string may be a join key. This
// invariant has leaked before, most recently through the weakest-variant name.
for (const row of laid.rows) {
	const keys = [
		row.panel.incumbent.key,
		row.panel.weakestVariantKey,
		...row.panel.incumbentVariants.map((v) => v.key)
	];
	for (const key of keys) {
		ok(!row.comparatorLabel.includes(key), `comparator renders display, not ${key}`);
		ok(!row.subLabel.includes(key), `sub-label renders display, not ${key}`);
		ok(!row.heteroLabel.includes(key), `composition strip renders display, not ${key}`);
	}
	ok(
		row.panel.weakestVariantDisplay ===
			row.panel.incumbentVariants.find((v) => v.key === row.panel.weakestVariantKey)?.display,
		`the weakest form carries its own display: ${row.key}`
	);
}
// The declined forms are visible, and only on the panels that have any.
for (const row of laid.rows) {
	const declined = row.marks.filter((m) => m.series === 'declined');
	eq(declined.length, row.panel.incumbentVariants.length - 1, `declined ticks: ${row.key}`);
	eq(row.hasForfeit, row.panel.incumbentVariants.length > 1, `forfeit segment presence: ${row.key}`);
	if (row.hasForfeit) ok(row.forfeitFrom < row.trackFrom, `the forfeit runs up to the drawn incumbent: ${row.key}`);
	else close(row.forfeitFrom, row.trackFrom, `no forfeit means no dashed run: ${row.key}`);
}

// Every gutter budget the builder enforces, measured against what it emitted.
for (const row of laid.rows) {
	ok(row.display.length <= DEPLOYED_BASELINE_LABEL_BUDGET_CHARS, `display budget: ${row.key}`);
	ok(row.subLabel.length <= DEPLOYED_BASELINE_SUBLABEL_BUDGET_CHARS, `sub-label budget: ${row.key}`);
	ok(row.comparatorLabel.length <= DEPLOYED_BASELINE_COMPARATOR_BUDGET_CHARS, `comparator budget: ${row.key}`);
	ok(row.heteroLabel.length <= DEPLOYED_BASELINE_HETERO_BUDGET_CHARS, `composition budget: ${row.key}`);
	ok(row.chip.length <= DEPLOYED_BASELINE_CHIP_BUDGET_CHARS, `chip budget: ${row.key}`);
	ok(row.deltaReadout.length <= DEPLOYED_BASELINE_READOUT_BUDGET_CHARS, `delta readout budget: ${row.key}`);
	ok(row.ciReadout.length <= DEPLOYED_BASELINE_READOUT_BUDGET_CHARS, `CI readout budget: ${row.key}`);
	ok(row.ruleCostReadout.length <= DEPLOYED_BASELINE_READOUT_BUDGET_CHARS, `rule-cost readout budget: ${row.key}`);
	// The advance the figure draws IS incumbent -> gate, in that order.
	const incumbentX = row.marks.find((m) => m.series === 'incumbent')?.x;
	const gateX = row.marks.find((m) => m.series === 'gate')?.x;
	close(row.trackFrom, incumbentX, `track starts at the incumbent mark: ${row.key}`);
	close(row.trackTo, gateX, `track ends at the gate mark: ${row.key}`);
}
for (const entry of laid.legend) {
	ok(entry.text.length <= DEPLOYED_BASELINE_LEGEND_BUDGET_CHARS, `legend budget: ${entry.series}`);
}
ok(laid.title.length <= DEPLOYED_BASELINE_TITLE_BUDGET_CHARS, 'the figure title fits its band');
ok(laid.headline.length <= DEPLOYED_BASELINE_HEADER_BUDGET_CHARS, 'the headline fits the header band');
ok(
	laid.subheadline.length <= DEPLOYED_BASELINE_HEADER_BUDGET_CHARS,
	'the subheadline fits the header band'
);
// The header band and the legend band must not overlap. They DID: the legend
// used to start at plotLeft, where its third row sat on top of the second header
// line, and no per-string budget could see it because the two were budgeted
// against different gutters.
// The header lines are LETTER-SPACED (`.fig-subtitle`, 0.04em), and CSS adds
// that tracking after every character, so the budget is measured against
// `headerLetterSpacedUnitsPerChar` and not the bare face. Measured in a browser
// against the rendered figure: 5.456 u/char, against the 5.1176 the budget used
// to assume. Nothing shipped clipped, but a full-budget line would have run into
// the legend mark.
ok(
	DEPLOYED_BASELINE_GEOMETRY.headerX +
		DEPLOYED_BASELINE_HEADER_BUDGET_CHARS *
			DEPLOYED_BASELINE_GEOMETRY.headerLetterSpacedUnitsPerChar <=
		DEPLOYED_BASELINE_GEOMETRY.legendMarkX - DEPLOYED_BASELINE_GEOMETRY.markRadius,
	'a full-budget header line still clears the legend’s leftmost ink'
);
// The regime line sits BELOW the legend, so it is budgeted against the content
// right edge instead — and its own vertical clearance is asserted, because that
// clearance is the only reason it may be wider.
ok(
	DEPLOYED_BASELINE_GEOMETRY.regimeLineY >
		DEPLOYED_BASELINE_GEOMETRY.legendY + 3 * DEPLOYED_BASELINE_GEOMETRY.legendRowHeight,
	'the regime line clears the legend’s last baseline'
);
ok(
	DEPLOYED_BASELINE_GEOMETRY.regimeLineY < DEPLOYED_BASELINE_GEOMETRY.rowsTop,
	'the regime line stays inside the header band'
);
ok(
	DEPLOYED_BASELINE_GEOMETRY.headerX +
		DEPLOYED_BASELINE_REGIME_BUDGET_CHARS *
			DEPLOYED_BASELINE_GEOMETRY.headerLetterSpacedUnitsPerChar <=
		DEPLOYED_BASELINE_GEOMETRY.width - 12,
	'a full-budget regime line still clears the right edge'
);
// The chip is letter-spaced too (`.row-chip`, 0.08em): 5.114 u/char measured,
// against the 4.5155 its budget used to assume. Right-anchored, so an overrun
// eats the chip's LEADING glyphs — silently.
ok(
	DEPLOYED_BASELINE_CHIP_BUDGET_CHARS * DEPLOYED_BASELINE_GEOMETRY.chipLetterSpacedUnitsPerChar <=
		DEPLOYED_BASELINE_GEOMETRY.labelAnchorX,
	'a full-budget chip still fits its right-anchored gutter'
);
// The letter-spaced advances must actually BE the face plus their tracking, or
// the two constants above are just numbers.
close(
	DEPLOYED_BASELINE_GEOMETRY.headerLetterSpacedUnitsPerChar,
	DEPLOYED_BASELINE_GEOMETRY.headerMonoUnitsPerChar + 0.04 * 8.5,
	'the header advance is the 8.5px face plus its 0.04em tracking'
);
close(
	DEPLOYED_BASELINE_GEOMETRY.chipLetterSpacedUnitsPerChar,
	DEPLOYED_BASELINE_GEOMETRY.chipMonoUnitsPerChar + 0.08 * 7.5,
	'the chip advance is the 7.5px face plus its 0.08em tracking'
);
ok(
	DEPLOYED_BASELINE_GEOMETRY.titleX +
		DEPLOYED_BASELINE_TITLE_BUDGET_CHARS * DEPLOYED_BASELINE_GEOMETRY.titleSerifUnitsPerChar <=
		DEPLOYED_BASELINE_GEOMETRY.legendMarkX - DEPLOYED_BASELINE_GEOMETRY.markRadius,
	'a full-budget title still clears the legend’s leftmost ink'
);
ok(
	DEPLOYED_BASELINE_GEOMETRY.legendTextX +
		DEPLOYED_BASELINE_LEGEND_BUDGET_CHARS * DEPLOYED_BASELINE_GEOMETRY.headerMonoUnitsPerChar <=
		DEPLOYED_BASELINE_GEOMETRY.width - 12,
	'a full-budget legend line still clears the right edge'
);
// And the row gutters must not overlap the plot or the readout column.
ok(
	DEPLOYED_BASELINE_GEOMETRY.labelAnchorX < DEPLOYED_BASELINE_GEOMETRY.plotLeft,
	'the right-anchored label gutter ends before the plot begins'
);
// The composition strip has its OWN baseline (`heteroDy` 52), below the readout
// column's last line (`ruleCostDy` 38), so its budget is measured against the
// content right edge rather than `readoutX`. Both facts are asserted: the
// vertical separation is what earns the wider gutter, and if the strip is ever
// moved back up beside the readouts this pair fails rather than silently
// licensing a 145-character string into a 550-unit gap.
ok(
	DEPLOYED_BASELINE_GEOMETRY.heteroDy >=
		DEPLOYED_BASELINE_GEOMETRY.ruleCostDy + DEPLOYED_BASELINE_GEOMETRY.heteroFontPx,
	'the composition strip sits on its own baseline, clear of the readout column'
);
ok(
	DEPLOYED_BASELINE_GEOMETRY.plotLeft +
		DEPLOYED_BASELINE_HETERO_BUDGET_CHARS * DEPLOYED_BASELINE_GEOMETRY.chipMonoUnitsPerChar <=
		DEPLOYED_BASELINE_GEOMETRY.width - 12,
	'a full-budget composition strip still clears the right edge'
);
ok(
	DEPLOYED_BASELINE_GEOMETRY.readoutX +
		DEPLOYED_BASELINE_READOUT_BUDGET_CHARS * DEPLOYED_BASELINE_GEOMETRY.monoUnitsPerChar <=
		DEPLOYED_BASELINE_GEOMETRY.width,
	'a full-budget readout still clears the right edge'
);
// Row-local baselines must not collide with one another.
{
	const g = DEPLOYED_BASELINE_GEOMETRY;
	const gutter = [
		[g.labelDy, g.labelFontPx],
		[g.subLabelDy, g.subLabelFontPx],
		[g.comparatorDy, g.comparatorFontPx],
		[g.chipDy, g.chipFontPx]
	];
	for (let i = 1; i < gutter.length; i++) {
		ok(gutter[i][0] - gutter[i - 1][0] >= gutter[i][1], `label gutter line ${i} clears line ${i - 1}`);
	}
	const readout = [
		[g.deltaDy, g.readoutFontPx],
		[g.ciDy, g.readoutFontPx],
		[g.ruleCostDy, g.readoutFontPx]
	];
	for (let i = 1; i < readout.length; i++) {
		ok(readout[i][0] - readout[i - 1][0] >= readout[i][1], `readout line ${i} clears line ${i - 1}`);
	}
	ok(g.chipDy < g.separatorDy && g.heteroDy < g.separatorDy, 'every row line sits above its separator');
	ok(g.heteroDy - (g.trackDy + g.markRadius) >= g.heteroFontPx, 'the composition strip clears the track');
	ok(g.legendY + 3 * g.legendRowHeight < g.rowsTop, 'the legend clears the first row');
}

// The gutter TITLES are budget-checked in the builder too. One of them shipped
// clipped ("gate − deployed, 95% CI" lost its trailing "CI") because it was a
// static string in the component that no budget could see; it is data now.
ok(
	laid.axisTitle.length <= DEPLOYED_BASELINE_AXIS_TITLE_BUDGET_CHARS,
	'the axis title fits its measured gutter'
);
eq(laid.readoutTitle.length, 2, 'the readout column heading ships as two lines');
for (const [index, line] of laid.readoutTitle.entries()) {
	ok(
		line.length <= DEPLOYED_BASELINE_READOUT_BUDGET_CHARS,
		`readout title line ${index} fits the readout gutter`
	);
}

// An over-budget panel display gates the FIGURE, not just a test.
// THIS CASE CHANGED SHAPE WITH THE ARCHITECTURE, and the old form silently
// stopped testing anything. It used to overrun `panels[0].display` and expect a
// gate. Row names are no longer passed through from the artifact at all: the
// artifact's own `display` is written in the shipped dialect ("the 2023 paper's
// own panel"), so the loader now resolves the on-screen name from the FROZEN key
// through DEPLOYED_BASELINE_PANEL_DISPLAY and keeps the artifact's wording only as
// audit text. An over-long artifact `display` therefore cannot reach a gutter, and
// the old mutation stopped throwing — correctly.
//
// The budget still exists and still gates; what it guards is now the AUTHORED name.
// So this asserts the two things that are actually reachable: every authored name
// fits, and a key with no authored name gates rather than falling back to the
// shipped string — that fallback is exactly what put the dialect on screen.
for (const [key, name] of Object.entries(DEPLOYED_BASELINE_PANEL_DISPLAY)) {
	ok(
		name.length <= DEPLOYED_BASELINE_LABEL_BUDGET_CHARS,
		`authored row name fits its gutter: ${key} is ${name.length} chars, budget ${DEPLOYED_BASELINE_LABEL_BUDGET_CHARS}`
	);
}
throws(() => {
	const bad = clone(artifactFixture());
	bad.panels[0].key = 'a_panel_key_nobody_authored_a_name_for';
	buildDeployedBaselineFigure(validateDeployedBaseline(bad));
}, 'a row whose key has no authored on-screen name gates the figure');
throws(() => {
	const bad = clone(artifactFixture());
	bad.panels[0].incumbent.display = 'y'.repeat(DEPLOYED_BASELINE_COMPARATOR_BUDGET_CHARS);
	bad.panels[0].incumbent_variants[2].display = bad.panels[0].incumbent.display;
	buildDeployedBaselineFigure(validateDeployedBaseline(bad));
}, 'an over-budget comparator name gates the figure');
// A mark below the axis floor would be drawn in the margin, reading as a
// different number. It must gate instead.
throws(() => {
	const bad = clone(artifactFixture());
	const panel = bad.panels[2];
	panel.incumbent_variants[0].auroc = 0.4;
	panel.incumbent_variants[0].delta_auroc = panel.gate.auroc - 0.4;
	panel.incumbent_variants[0].bootstrap = bootstrapFixture(panel.gate.auroc - 0.4);
	panel.incumbent.auroc = 0.4;
	panel.delta_auroc = panel.gate.auroc - 0.4;
	panel.bootstrap = bootstrapFixture(panel.delta_auroc);
	panel.weakest_variant_auroc = 0.4;
	panel.evidence_matched_control.delta_auroc_incumbent_minus_control =
		0.4 - panel.evidence_matched_control.auroc;
	bad.replication.delta_min = Math.min(...bad.panels.map((p) => p.delta_auroc));
	bad.replication.delta_max = Math.max(...bad.panels.map((p) => p.delta_auroc));
	bad.replication.largest_panel_is_at_top_of_range =
		bad.panels[0].delta_auroc === bad.replication.delta_max;
	buildDeployedBaselineFigure(validateDeployedBaseline(bad));
}, 'a mark below the axis floor gates the figure');

// ---------------------------------------------------------------------------
// 2. every identity fails CLOSED
// ---------------------------------------------------------------------------
const mutations = [
	['artifact kind', (a) => (a.artifact_kind = 'something_else')],
	['schema version', (a) => (a.schema_version = 99)],
	['metric', (a) => (a.metric = 'average_precision')],
	['positive class', (a) => (a.positive_class = 'gold-incorrect')],
	['panel order', (a) => (a.panels = [a.panels[1], a.panels[0], a.panels[2], a.panels[3]])],
	['panel census', (a) => a.panels.pop()],
	['caveat census', (a) => a.caveats.pop()],
	['a missing claim', (a) => delete a.claim],
	['a missing figure title', (a) => delete a.figure_title],
	['a missing anti-claim', (a) => delete a.claim_is_not],
	['n_correct + n_errors', (a) => (a.panels[0].n_errors += 1)],
	['base rate', (a) => (a.panels[0].base_rate_correct = 0.5)],
	[
		'the heterogeneity block disagrees with the panel census',
		(a) => (a.panels[0].heterogeneity.base_rate_correct = 0.5)
	],
	['delta identity', (a) => (a.panels[0].delta_auroc += 0.01)],
	['variant delta identity', (a) => (a.panels[0].incumbent_variants[0].delta_auroc += 0.01)],
	['a variant without its own interval', (a) => delete a.panels[0].incumbent_variants[0].bootstrap],
	['a variant in no known family', (a) => (a.panels[0].incumbent_variants[0].family = 'some_other_family')],
	['a variant without a source digest', (a) => delete a.panels[0].incumbent_variants[0].source_sha256],
	['delta_favors_gate is a claim', (a) => (a.panels[0].delta_favors_gate = false)],
	['gate_beats_every_variant is a claim', (a) => (a.panels[0].gate_beats_every_variant = false)],
	['the variant CI count is a claim', (a) => (a.panels[0].n_variants_ci_excludes_zero = 0)],
	['the variant census is a claim', (a) => (a.panels[0].n_incumbent_variants = 99)],
	[
		'a weaker incumbent is drawn',
		(a) => {
			const weak = a.panels[0].incumbent_variants[0];
			a.panels[0].incumbent.key = weak.key;
			a.panels[0].incumbent.auroc = weak.auroc;
			a.panels[0].incumbent.family = weak.family;
			a.panels[0].incumbent.fitted = weak.fitted;
			a.panels[0].delta_auroc = a.panels[0].gate.auroc - weak.auroc;
			a.panels[0].delta_favors_gate = a.panels[0].delta_auroc > 0;
			a.panels[0].bootstrap = bootstrapFixture(a.panels[0].delta_auroc);
		}
	],
	[
		'the incumbent claims a family it did not come from',
		(a) => (a.panels[0].incumbent.family = LIB)
	],
	['the rule’s price is a claim', (a) => (a.panels[0].selection_cost_auroc += 0.01)],
	['the weakest form is misnamed', (a) => (a.panels[0].weakest_variant_key = 'cogex_fitted_hybrid')],
	[
		'the headline interval is not the drawn incumbent’s',
		(a) => (a.panels[0].bootstrap = bootstrapFixture(0.5))
	],
	['no sourceable incumbent', (a) => (a.panels[2].incumbent_variants = [])],
	[
		'the paper’s RF on a panel that is not the paper’s',
		(a) => (a.panels[1].research_model = clone(a.panels[0].research_model))
	],
	['the paper’s panel without its RF', (a) => (a.panels[0].research_model = null)],
	['is_paper_panel mismarked', (a) => (a.panels[1].is_paper_panel = true)],
	[
		'research delta identity',
		(a) => (a.panels[0].research_model.delta_auroc_gate_minus_research += 0.01)
	],
	[
		'research-minus-incumbent identity',
		(a) => (a.panels[0].research_model.delta_auroc_research_minus_incumbent += 0.01)
	],
	['a family stops being INDRA’s deployed belief', (a) => (a.incumbent_families[0].deployed = false)],
	['the library default becomes fitted', (a) => (a.incumbent_families[0].fitted = true)],
	['the stored production belief stops being fitted', (a) => (a.incumbent_families[1].fitted = false)],
	['a family key is renamed', (a) => (a.incumbent_families[1].key = 'some_other_family')],
	[
		'a declared family no panel sources',
		(a) => {
			for (const panel of a.panels) {
				for (const variant of panel.incumbent_variants) variant.family = SERVED;
				panel.incumbent.family = SERVED;
			}
		}
	],
	['the paper’s RF is called deployed', (a) => (a.arms.research_model.deployed = true)],
	['the paper’s RF stops being fitted', (a) => (a.arms.research_model.fitted = false)],
	['the gate is called fitted', (a) => (a.arms.gate.fitted = true)],
	['an arm key is renamed', (a) => (a.arms.gate.key = 'some_other_gate')],
	[
		'the two-family split loses its evidence',
		(a) => {
			for (const row of a.served_belief_identity.per_panel) {
				row.n_below_floor = 0;
				row.fraction_below_floor = 0;
			}
			a.served_belief_identity.n_served_below_floor = 0;
			a.served_belief_identity.n_panels_with_served_below_floor = 0;
		}
	],
	[
		'the floor evidence is a claim',
		(a) => (a.served_belief_identity.n_served_below_floor += 1)
	],
	[
		'the floor fraction is a claim',
		(a) => (a.served_belief_identity.per_panel[0].fraction_below_floor = 0.99)
	],
	[
		'the floor evidence names a panel that is not here',
		(a) => (a.served_belief_identity.per_panel[0].panel_key = 'some_other_panel')
	],
	[
		'the evidence-matched control calls itself an incumbent',
		(a) => (a.panels[1].evidence_matched_control.is_an_incumbent = true)
	],
	[
		'the control’s gap is a claim',
		(a) => (a.panels[1].evidence_matched_control.delta_auroc_incumbent_minus_control += 0.01)
	],
	[
		'a control that is both present and excused',
		(a) => (a.panels[1].evidence_matched_control_absent_because = 'and also absent.')
	],
	[
		'a control that is neither present nor excused',
		(a) => {
			a.panels[1].evidence_matched_control = null;
			a.panels[1].evidence_matched_control_absent_because = null;
		}
	],
	[
		'the evidence share disagrees with the censuses',
		(a) => (a.panels[1].heterogeneity.reader_evidence_share_of_corpus = 0.9)
	],
	[
		'a panel claims the reader saw everything when it did not',
		(a) => (a.panels[1].heterogeneity.reader_saw_full_evidence = true)
	],
	[
		'a census with no numbers and no reason',
		(a) => {
			a.panels[2].heterogeneity.corpus_evidence_per_statement = null;
			a.panels[2].heterogeneity.corpus_evidence_absent_because = null;
		}
	],
	[
		'a census whose mean falls outside its own range',
		(a) => (a.panels[0].heterogeneity.evidence_reads_per_statement.mean = 900)
	],
	['favouring count is a claim', (a) => (a.replication.n_panels_favoring_gate = 0)],
	['CI-excludes-zero count is a claim', (a) => (a.replication.n_panels_ci_excludes_zero = 0)],
	['beats-every-variant count is a claim', (a) => (a.replication.n_panels_gate_beats_every_variant = 0)],
	['variant total is a claim', (a) => (a.replication.n_incumbent_variants_total = 99)],
	['the rule’s worst price is a claim', (a) => (a.replication.selection_cost_auroc_max += 0.1)],
	['delta range', (a) => (a.replication.delta_max += 0.1)],
	['largest panel', (a) => (a.replication.largest_panel_key = 'holdout_cc')],
	[
		'largest-at-top is a claim',
		(a) => (a.replication.largest_panel_is_at_top_of_range = !a.replication.largest_panel_is_at_top_of_range)
	],
	['inverted bootstrap interval', (a) => (a.panels[0].bootstrap.ci95_low = a.panels[0].bootstrap.ci95_high + 0.1)],
	['more valid resamples than draws', (a) => (a.panels[0].bootstrap.n_valid_resamples = 20001)],
	['a missing provenance digest', (a) => delete a.panels[0].provenance.gold_sha256],
	['a missing curator note', (a) => delete a.panels[0].curator_note],
	['a missing join mode', (a) => delete a.panels[0].heterogeneity.join_mode],
	['a missing join summary', (a) => delete a.panels[0].heterogeneity.join_summary],
	[
		'a cross-check that does not describe this variant',
		(a) => (a.panels[1].incumbent_variants[0].cross_check.this_auroc += 0.01)
	],
	[
		'a cross-check whose divergence is a claim',
		(a) => (a.panels[1].incumbent_variants[0].cross_check.delta_vs_sibling += 0.01)
	],
	[
		'a cross-check showing we took the WEAKER form',
		(a) => {
			const xc = a.panels[1].incumbent_variants[0].cross_check;
			xc.sibling_auroc = xc.this_auroc + 0.01;
			xc.delta_vs_sibling = -0.01;
		}
	],
	// The heterogeneity disclosure. A mean of 1.25 reads and "84% of these
	// statements hold exactly one evidence" are different disclosures, and the
	// second is the one that says what the gate was asked to do — so it must be
	// present, must be arithmetic, and must census THIS panel.
	[
		'a census with no single-evidence count',
		(a) => delete a.panels[2].heterogeneity.evidence_reads_per_statement.n_single
	],
	[
		'a census with no statement count',
		(a) => delete a.panels[2].heterogeneity.evidence_reads_per_statement.n_statements
	],
	[
		'a single-evidence share that is a claim',
		(a) => (a.panels[2].heterogeneity.evidence_reads_per_statement.share_single = 0.1)
	],
	[
		'more single-evidence statements than statements',
		(a) => {
			const s = a.panels[2].heterogeneity.evidence_reads_per_statement;
			s.n_single = s.n_statements + 1;
			s.share_single = s.n_single / s.n_statements;
		}
	],
	[
		'a census whose mean disagrees with its own total',
		(a) => (a.panels[2].heterogeneity.evidence_reads_per_statement.mean += 0.5)
	],
	[
		'a census of a different statement set',
		(a) => {
			const s = a.panels[2].heterogeneity.evidence_reads_per_statement;
			s.n_statements += 10;
			s.mean = s.total / s.n_statements;
			s.share_single = s.n_single / s.n_statements;
		}
	],
	[
		'single-evidence statements reported where the minimum is above one',
		(a) => (a.panels[2].heterogeneity.evidence_reads_per_statement.min = 2)
	],
	[
		'a corpus census of a different statement set',
		(a) => {
			const s = a.panels[1].heterogeneity.corpus_evidence_per_statement;
			s.n_statements += 7;
			s.mean = s.total / s.n_statements;
			s.share_single = s.n_single / s.n_statements;
		}
	],
	// The evidence-regime span carries half the claim, so every part of it is an
	// identity rather than an assertion.
	['the evidence-regime fold span is a claim', (a) => (a.replication.evidence_regime_fold_span *= 2)],
	['the thinnest panel’s density is a claim', (a) => (a.replication.reads_per_statement_mean_min *= 0.5)],
	['the densest panel’s density is a claim', (a) => (a.replication.reads_per_statement_mean_max *= 2)],
	['the single-evidence range is a claim', (a) => (a.replication.share_single_evidence_max = 0.99)],
	['a missing evidence-regime span', (a) => delete a.replication.evidence_regime_fold_span],
	// The served-identity block travels by KEY and renders by DISPLAY, so a key
	// naming no panel must fail rather than reach the page as its own label.
	[
		'a served-identity row naming a panel that is not here',
		(a) => (a.served_belief_identity.per_panel[0].panel_key = 'some_other_panel')
	]
];
for (const [label, mutate] of mutations) {
	throws(() => {
		const bad = clone(artifactFixture());
		mutate(bad);
		validateDeployedBaseline(bad);
	}, `fails closed on: ${label}`);
}

// signed3 always carries a sign, and a minus is the typographic one.
eq(signed3(0.1187411), '+0.119', 'signed3 formats a gain');
eq(signed3(-0.02), '−0.020', 'signed3 formats a loss with a real minus sign');
eq(fmt1(19.751924), '19.8', 'fmt1 rounds a per-statement census to one decimal');
eq(pct0(0.7323860272350503), '73%', 'pct0 rounds a share to whole percent');

// ---------------------------------------------------------------------------
// 3. THE SHIPPED BYTES
// ---------------------------------------------------------------------------
const MODEL_DIR = '../../data/results/deployed_baseline_replication_20260727/';
const ARTIFACT_NAME = 'deployed_baseline_replication.json';
const shippedBytes = readFileSync(new URL(MODEL_DIR + ARTIFACT_NAME, import.meta.url));
const shippedDigest = createHash('sha256').update(shippedBytes).digest('hex');
const manifest = JSON.parse(readFileSync(new URL(MODEL_DIR + 'manifest.json', import.meta.url), 'utf8'));
eq(manifest.output_sha256?.[ARTIFACT_NAME], shippedDigest, 'the shipped artifact matches its manifest sha256');

const shipped = validateDeployedBaseline(JSON.parse(shippedBytes.toString('utf8')));
const shippedFigure = buildDeployedBaselineFigure(shipped);

// The claim, asserted against the shipped data rather than described in prose.
for (const family of shipped.families) {
	ok(family.deployed, `shipped family is a DEPLOYED form of INDRA belief: ${family.key}`);
}
eq(shipped.families.find((f) => f.key === LIB)?.fitted, false, 'the shipped library default is unfitted');
eq(shipped.families.find((f) => f.key === SERVED)?.fitted, true, 'the shipped stored belief is fitted');
ok(
	shipped.families.find((f) => f.key === LIB)?.shipsIn.includes('SimpleScorer'),
	'the library default names SimpleScorer as the thing INDRA ships'
);
ok(!shipped.arms.research.deployed, 'the paper’s RF is marked never-deployed');
ok(shipped.arms.research.fitted, 'the paper’s RF is marked fitted');

// The two-family split is EARNED by the shipped bytes, not asserted.
ok(shipped.servedBeliefIdentity.nServedBelowFloor > 0, 'stored beliefs fall below the SimpleScorer floor');
ok(
	shipped.servedBeliefIdentity.nPanelsWithServedBelowFloor >= 1,
	'at least one shipped panel demonstrates the split'
);
ok(
	shipped.servedBeliefIdentity.simpleScorerFloor > DEPLOYED_BASELINE_AXIS_MIN,
	'the SimpleScorer floor is a real floor above chance'
);

eq(
	shipped.replication.nPanelsFavoringGate,
	shipped.replication.nPanels,
	'every shipped panel favours the reader gate'
);
eq(
	shipped.replication.nPanelsCiExcludesZero,
	shipped.replication.nPanels,
	'every shipped headline interval excludes zero'
);
// The strong form of the claim: the gate beats EVERY sourceable form of INDRA's
// belief on every panel, not just the one the argmax happened to draw.
eq(
	shipped.replication.nPanelsGateBeatsEveryVariant,
	shipped.replication.nPanels,
	'the gate beats every sourceable form of INDRA belief on every panel'
);
eq(
	shipped.replication.nIncumbentVariantsCiExcludesZero,
	shipped.replication.nIncumbentVariantsTotal,
	'every per-variant interval excludes zero too'
);
ok(shipped.replication.deltaMin > 0, 'the weakest shipped panel still favours the gate');
eq(shipped.replication.largestPanelKey, DEPLOYED_BASELINE_PAPER_PANEL_KEY, 'the largest shipped panel is the paper’s');
// Both families must actually appear among the shipped comparators, or the
// figure is drawing a distinction it never uses.
const shippedFamilies = new Set(shipped.panels.flatMap((p) => p.incumbentVariants.map((v) => v.family)));
ok(shippedFamilies.has(LIB), 'the shipped panels source the library default somewhere');
ok(shippedFamilies.has(SERVED), 'the shipped panels source the stored production belief somewhere');
// The paper's own panel must source BOTH, or the strongest-incumbent rule was
// never actually exercised where it matters most.
const shippedPaperPanel = shipped.panels.find((p) => p.key === DEPLOYED_BASELINE_PAPER_PANEL_KEY);
ok(
	new Set(shippedPaperPanel?.incumbentVariants.map((v) => v.family)).size === 2,
	'the paper’s own panel is compared against BOTH families of INDRA belief'
);

for (const panel of shipped.panels) {
	ok(panel.deltaAuroc > 0, `shipped panel favours the gate: ${panel.key}`);
	ok(
		panel.incumbent.auroc === Math.max(...panel.incumbentVariants.map((v) => v.auroc)),
		`shipped panel draws its strongest incumbent: ${panel.key}`
	);
	ok(panel.provenance.goldSha256.length === 64, `shipped panel pins its gold: ${panel.key}`);
	ok(panel.provenance.runSha256.length === 64, `shipped panel pins its run: ${panel.key}`);
	for (const variant of panel.incumbentVariants) {
		ok(variant.sourceSha256.length === 64, `shipped variant pins its source: ${panel.key}/${variant.key}`);
	}
	// Panel heterogeneity is a census, so it has to be present and self-consistent.
	ok(
		panel.heterogeneity.evidenceReadsPerStatement.total >= panel.nStatements,
		`shipped panel censuses its evidence reads: ${panel.key}`
	);
	// Either the control or the reason there is none — never silence.
	ok(
		(panel.evidenceMatchedControl === null) !== (panel.evidenceMatchedControlAbsentBecause === null),
		`shipped panel accounts for its evidence-matched control: ${panel.key}`
	);
	if (panel.evidenceMatchedControl) {
		ok(!panel.evidenceMatchedControl.isAnIncumbent, `the control is never an incumbent: ${panel.key}`);
		ok(
			panel.evidenceMatchedControl.auroc < panel.incumbent.auroc,
			`the control is weaker than the comparator, so the scope asymmetry is conservative: ${panel.key}`
		);
	}
}
const paperRow = shippedFigure.rows.find((r) => r.key === DEPLOYED_BASELINE_PAPER_PANEL_KEY);
ok(paperRow !== undefined, 'the shipped figure draws the paper’s panel');
const research = paperRow?.panel.researchModel ?? null;
ok(research !== null, 'the shipped paper panel carries the paper’s fitted RF');
if (research) {
	// Both gaps are positive, so the segment split is meaningful: the strongest
	// INDRA belief, then their research model, then the gate, in that order.
	ok(research.deltaResearchMinusIncumbent > 0, 'the paper’s RF beats the strongest INDRA belief drawn');
	ok(research.deltaGateMinusResearch > 0, 'the gate beats the paper’s RF');
	close(
		research.deltaResearchMinusIncumbent + research.deltaGateMinusResearch,
		paperRow.panel.deltaAuroc,
		'the two drawn segments sum to the row’s total advance'
	);
}
// Geometry the SVG depends on, on the shipped data rather than the fixture.
eq(shippedFigure.rows.length, DEPLOYED_BASELINE_PANEL_KEYS.length, 'the shipped figure draws every panel');
eq(
	shippedFigure.height,
	DEPLOYED_BASELINE_GEOMETRY.rowsTop +
		shippedFigure.rows.length * DEPLOYED_BASELINE_GEOMETRY.rowHeight +
		DEPLOYED_BASELINE_GEOMETRY.axisPad,
	'the shipped figure height is its own row arithmetic'
);
for (const row of shippedFigure.rows) {
	ok(row.display.length <= DEPLOYED_BASELINE_LABEL_BUDGET_CHARS, `shipped display budget: ${row.key}`);
	ok(row.subLabel.length <= DEPLOYED_BASELINE_SUBLABEL_BUDGET_CHARS, `shipped sub-label budget: ${row.key}`);
	ok(
		row.comparatorLabel.length <= DEPLOYED_BASELINE_COMPARATOR_BUDGET_CHARS,
		`shipped comparator budget: ${row.key}`
	);
	ok(row.heteroLabel.length <= DEPLOYED_BASELINE_HETERO_BUDGET_CHARS, `shipped composition budget: ${row.key}`);
	ok(row.chip.length <= DEPLOYED_BASELINE_CHIP_BUDGET_CHARS, `shipped chip budget: ${row.key}`);
	ok(row.ciReadout.length <= DEPLOYED_BASELINE_READOUT_BUDGET_CHARS, `shipped CI readout budget: ${row.key}`);
	ok(
		row.ruleCostReadout.length <= DEPLOYED_BASELINE_READOUT_BUDGET_CHARS,
		`shipped rule-cost readout budget: ${row.key}`
	);
	ok(row.trackTo > row.trackFrom, `the shipped advance points right: ${row.key}`);
	for (const mark of row.marks) {
		ok(
			mark.x >= DEPLOYED_BASELINE_GEOMETRY.plotLeft && mark.x <= DEPLOYED_BASELINE_GEOMETRY.plotRight,
			`shipped mark is inside the plot: ${row.key}/${mark.id}`
		);
	}
}
for (const entry of shippedFigure.legend) {
	ok(entry.text.length <= DEPLOYED_BASELINE_LEGEND_BUDGET_CHARS, `shipped legend budget: ${entry.series}`);
}
ok(shippedFigure.title.length <= DEPLOYED_BASELINE_TITLE_BUDGET_CHARS, 'shipped title budget');
ok(shippedFigure.headline.length <= DEPLOYED_BASELINE_HEADER_BUDGET_CHARS, 'shipped headline budget');
ok(
	shippedFigure.subheadline.length <= DEPLOYED_BASELINE_HEADER_BUDGET_CHARS,
	'shipped subheadline budget'
);
ok(
	shippedFigure.regimeLine.length <= DEPLOYED_BASELINE_REGIME_BUDGET_CHARS,
	'shipped regime-line budget'
);

// ---------------------------------------------------------------------------
// 4. THE FIGURE'S CLAIM AND ITS DISCLOSURES, on the shipped bytes
// ---------------------------------------------------------------------------
// Every row must NAME the form of INDRA belief it was measured against, and the
// name must be the comparator's own display — this is the defect that made the
// figure read as "the same comparison, four times".
for (const row of shippedFigure.rows) {
	ok(
		row.comparatorLabel === `vs ${row.panel.incumbent.display}`,
		`the row names its own comparator: ${row.key}`
	);
	ok(
		!row.comparatorLabel.includes(row.panel.incumbent.key),
		`the row's comparator renders a display name, not a join key: ${row.key}`
	);
	ok(!row.display.includes(row.key), `the row label is a display name, not a join key: ${row.key}`);
	// The composition strip must carry the two disclosures a reviewer needs to
	// see that these are not four copies of one comparison.
	ok(
		row.heteroLabel.includes('read/statement') && row.heteroLabel.includes('single-evidence'),
		`the composition strip discloses evidence density AND the single-evidence share: ${row.key}`
	);
	ok(
		row.heteroLabel.includes(pct0(row.panel.heterogeneity.evidenceReadsPerStatement.shareSingle)),
		`the strip prints this panel's own single-evidence share: ${row.key}`
	);
	// The argmax rule's price is legible per row, either as a number or as the
	// statement that this panel had nothing to give up.
	ok(
		row.panel.incumbentVariants.length > 1
			? row.ruleCostReadout.includes(row.panel.selectionCostAuroc.toFixed(3))
			: row.ruleCostReadout === 'sole form sourced',
		`the rule's price is printed on the row: ${row.key}`
	);
	ok(
		row.hasForfeit === row.panel.incumbentVariants.length > 1,
		`the forfeit segment is drawn exactly where the rule gave something up: ${row.key}`
	);
}
// At least one shipped row must actually exercise the rule, or "we always draw
// the strongest incumbent" is a rule with no observed cost.
ok(
	shipped.panels.some((p) => p.selectionCostAuroc > 0),
	'the strongest-incumbent rule costs the figure real margin on at least one panel'
);
// The rule's direction: on every panel with more than one form, the drawn
// comparator is the one that makes our delta SMALLEST.
for (const panel of shipped.panels) {
	const smallest = Math.min(...panel.incumbentVariants.map((v) => v.deltaAuroc));
	close(panel.deltaAuroc, smallest, `the drawn delta is the smallest available: ${panel.key}`);
}
// The heterogeneity is COMPUTED: every census brackets its own arithmetic and
// the panels genuinely span different evidence regimes.
for (const panel of shipped.panels) {
	const reads = panel.heterogeneity.evidenceReadsPerStatement;
	eq(reads.nStatements, panel.nStatements, `the read census counts this panel: ${panel.key}`);
	close(reads.mean, reads.total / reads.nStatements, `the read census mean is its own arithmetic: ${panel.key}`);
	close(
		reads.shareSingle,
		reads.nSingle / reads.nStatements,
		`the single-evidence share is its own arithmetic: ${panel.key}`
	);
}
ok(
	shipped.replication.evidenceRegimeFoldSpan > 2,
	'the shipped panels span materially different evidence regimes'
);
close(
	shipped.replication.evidenceRegimeFoldSpan,
	shipped.replication.readsPerStatementMeanMax / shipped.replication.readsPerStatementMeanMin,
	'the shipped fold span is the ratio of the shipped extremes'
);
// The served-identity block resolves every key to a panel display name, so no
// join key can reach a render position through it.
for (const row of shipped.servedBeliefIdentity.perPanel) {
	const named = shipped.panels.find((p) => p.key === row.panelKey);
	ok(named !== undefined, `the served-identity row names a shipped panel: ${row.panelKey}`);
	eq(row.panelDisplay, named?.display, `the served-identity row carries a display name: ${row.panelKey}`);
}
// The claim string must be the one the data supports: it has to carry the
// counts, not adjectives. If the sentence stops naming the fold span or the
// rule's price, it has drifted back to "the same comparison four times".
for (const fragment of [
	String(shipped.replication.nPanels),
	String(shipped.replication.nIncumbentVariantsTotal),
	shipped.replication.evidenceRegimeFoldSpan.toFixed(0),
	shipped.replication.selectionCostAurocMax.toFixed(4)
]) {
	ok(shipped.claim.includes(fragment), `the shipped claim carries its own count: ${fragment}`);
}
ok(
	shipped.claimIsNot.toLowerCase().includes('not the same comparison'),
	'the shipped anti-claim still refuses the "same comparison four times" reading'
);

console.log(
	`\ndeployed-baseline contract: ${shipped.panels.length} panels, ` +
		`${shipped.replication.nIncumbentVariantsTotal} forms of INDRA belief, deltas ` +
		`${signed3(shipped.replication.deltaMin)}..${signed3(shipped.replication.deltaMax)}, ` +
		`${shipped.replication.nPanelsCiExcludesZero}/${shipped.replication.nPanels} headline intervals exclude zero, ` +
		`${shipped.replication.nIncumbentVariantsCiExcludesZero}/${shipped.replication.nIncumbentVariantsTotal} per-form intervals exclude zero`
);
console.log(
	`  served belief is NOT SimpleScorer: ${shipped.servedBeliefIdentity.nServedBelowFloor} stored ` +
		`beliefs below the ${shipped.servedBeliefIdentity.simpleScorerFloor} floor on ` +
		`${shipped.servedBeliefIdentity.nPanelsWithServedBelowFloor} panels`
);
for (const panel of shipped.panels) {
	console.log(
		`  ${panel.key.padEnd(22)} n=${String(panel.nStatements).padStart(5)}  ` +
			`reads/stmt ${fmt1(panel.heterogeneity.evidenceReadsPerStatement.mean).padStart(5)}  ` +
			`single-ev ${pct0(panel.heterogeneity.evidenceReadsPerStatement.shareSingle).padStart(4)}  ` +
			`vs ${panel.incumbent.key.padEnd(26)} ${panel.incumbent.auroc.toFixed(4)}  ` +
			`gate ${panel.gate.auroc.toFixed(4)}  ${signed3(panel.deltaAuroc)} ` +
			`[${signed3(panel.bootstrap.ci95Low)}, ${signed3(panel.bootstrap.ci95High)}]  ` +
			`rule cost ${panel.selectionCostAuroc.toFixed(4)}`
	);
}

if (failures) {
	console.error(`\n${failures} failure(s)`);
	process.exit(1);
}
console.log('deployed-baseline contract: OK');
