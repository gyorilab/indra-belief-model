# Running the 2023 assembly paper's own belief code: reproduction, and what a reading gate adds to the belief INDRA serves

> **Provenance note (2026-08-25).** This is a historical record. The repository
> was later cut to prod infrastructure plus the future-LLM compare loop, so some
> paths cited below name code that no longer exists — the comparison harness, the
> spend ledger, the viewer, the paper-replication lane, the probe battery. The
> measurements and conclusions stand as recorded; git history holds the code they
> were measured against. No guard checks these citations any more.

> **Status:** complete (2026-07-24; restructured 2026-07-27 for the 2023 paper's
> authors). Runs the 2023 INDRA assembly paper's **own released code** — not our
> reimplementation — to reproduce Table 6, and places INDRA's *deployed* belief
> and the paper's *research* random forest alongside LLM-reader arms on identical
> statements.
>
> **Primary artifact: the interactive `/paper` section of the viewer**
> (`viewer/src/routes/paper/+page.svelte`); this memo is its prose companion.
> Both read the same JSON, and every number below can be checked against the page
> and against the artifact the page loads.
>
> **Note for anyone outside the repo.** The result artifacts live under
> `data/results/`, which is **gitignored**. They are *working-tree outputs*, not
> tracked files — cloning this repository does not give you the bytes, and no
> commit contains them. Every one is regenerable from the script named beside it
> (each script takes its own `--out-json` / `--manifest`), and each is
> sha256-pinned in its run `manifest.json`, so provenance is verifiable without
> receiving the bytes. The two *tracked* inputs a reader can check directly are
> `data/benchmark/indra_paper_2023.manifest.json` (the paper's own four critical
> files, byte-for-byte) and
> `data/benchmark/indra_paper_2023_published_method_metrics.json` (the paper's 59
> published method rows, transcribed).
>
> Most figures come from `data/results/indra_paper_literal_models_20260724/`;
> §3 reads `deployed_baseline_replication_20260727/`, and §7 reads
> `current_indra_simple_paper_20260717/` and
> `indra_paper_statement_gold_20260717/`. Where the page is terse for its own
> length budget, the memo carries the long provenance.
> Companion: `research/indra_belief_comparison.md`.
>
> <details>
> <summary>Revision log</summary>
>
> **2026-07-26** — revised after two independent reviews (senior-author
> simulation + adversarial fact-check). Both killed the same claim — a "belief
> rises, accuracy falls" inversion on the reach ladder — which is **retracted**
> and replaced by the better-powered single-evidence result (§7).
>
> **2026-07-26b — reconciled to `/paper`.** Every number the memo and the page
> both state was diffed against the artifact the page's loader reads. Two moved,
> both by re-derivation rather than by editing:
> (1) the **bit-exact** reachability tier in Result 3c went from a partial count
> with a small unsettled remainder on each arm to **1,227/1,227, 1,211/1,211,
> 1,296/1,296 and 1,148/1,148 — `share_bit_exact = 1.0` on all four arms**, with
> zero budget-exhausted and zero counterexamples; the earlier remainders were a
> search-budget artifact, so the claim got *stronger*;
> (2) the permutation chance floor is now quoted **per arm** rather than pooled,
> which *raises* the bar for Gemma 4 E2B to **0.514**.
>
> **2026-07-26c — arm names, and three misattributions.** Naming only; no number
> moved. Three arms were named after the 2023 paper when they are not the
> paper's: the unfitted default-prior `SimpleScorer`, the out-of-fold
> source+subtype refit, and the full-feature `CountsScorer` RF. All three are now
> named for what they are.
>
> **2026-07-27 — restructured; five numbers corrected; one conclusion retracted.**
> The memo previously led with the ranking margin against the paper's *research*
> RF. That is the smallest and least robust result we have, and it is not the
> comparison a deployment decision turns on. It is now §5, stated honestly and
> secondarily. The memo leads instead with the comparison against the belief
> INDRA actually **serves** (§3) and with the operational review-queue result
> (§4). Corrections, each recorded rather than quietly applied:
> - **The AUROC headline is +0.074, not +0.127 or +0.1187.** It moved twice, each
>   time as the comparator got more honest. +0.127 is the no-propagation variant;
>   +0.1187 the hierarchy-propagated one. Both assumed INDRA's *stored* belief was
>   `SimpleScorer`. That premise is **refuted**: `SimpleScorer` at the bundled
>   priors has a hard floor of 1 − max(syst+rand) = **0.60** (gnbr), and hierarchy
>   propagation only raises a belief — yet 272 of 596 stored beliefs on
>   `external_curator_v1` fall below it, the minimum being 0.360. The served
>   belief is the **fitted HybridScorer**, which is a stronger incumbent (0.8272
>   on the paper panel), so under the argmax rule it is the comparator and the
>   margin is **+0.074 [+0.053, +0.095]** (§3).
> - **The paper's fitted RF needs 662 reviews, not 665, for equal yield.** 665 is
>   a different arm — our full-feature `CountsScorer` RF on current INDRA, whose
>   frozen artifact key is misleadingly spelled `paper RF (full features)`. Both
>   numbers are correct for their own arm and both are now stated (§4).
> - **The AUROC gain is not a ranking result** and is no longer described as one.
>   It is almost entirely the binary zero/non-zero split (§5).
> - **The reader arms are not zero-shot** — 14 hand-authored demonstration pairs
>   per call. Any surviving "zero-shot" phrasing is removed.
> - **Result 5's "reading makes the paper's ranking worse at the top" is
>   RETRACTED.** It was regression to the mean on an *endogenous* banding
>   variable. Banded on the compared arm's own score, the identical decomposition
>   of the identical arm says the exact opposite. The old numbers survive, as the
>   evidence for that retraction (§5e).
> </details>

**A naming note, since this memo goes to the people who published the formula.**
Four misattribution risks, all of which this memo holds to:

1. `SimpleScorer` runs the same noisy-OR equation the paper's belief models use,
   but over INDRA's default `(rand, syst)` table, **unfitted**. It is *not* the
   paper's `Belief Orig`, which refits per-source `pr`/`ps` per fold by MCMC.
2. The full-feature RF below is **our** port over INDRA's whole `CountsScorer`
   feature set — a superset of the paper's engineered Type/#PMIDs/promoter
   panel — **not** the paper's RF. The paper's RF appears here only as the
   literal `RF 2k-d13 + Type/#PMIDs/promoter`.
3. `BayesianScorer, source+subtype refit` is **ours**. The paper publishes no
   subtype-resolved belief arm among its 59 methods, and no hierarchy arm.
4. The reader arms are **not zero-shot**: every call carries 14 hand-authored
   demonstration pairs (9 prefixes × 28 messages). They are also **not
   calibrated**.

Paper method strings are quoted **exactly** as published (U+002D hyphen) from
`data/benchmark/indra_paper_2023_published_method_metrics.json`. Where an
artifact's JSON key still spells an old or misleading name, the key is quoted
**as a key** — those strings address shipped data and are frozen — and the arm is
named beside it by what it is.

---

## What to take away

**The arm this program has been calling an "LLM scorer" is not a rival belief
model.** It is **INDRA's shipped default-prior noisy-OR (`SimpleScorer`),
evaluated on the subset of evidence an LLM reader kept**. Everything below is an
ablation *of* that aggregation, not a competitor to it — and it is purely
subtractive, which is both the source of its gain and its most important limit.

Three results, in the order they matter:

1. **Against what INDRA actually serves** (§3), the reading gate is worth
   **+0.074 AUROC** on the paper's own panel, measured against the *strongest*
   sourceable form of that incumbent, and the direction **replicates on four
   panels** (Δ +0.044 to +0.119, every interval excluding zero).
2. **Operationally** (§4), at the gate's own untuned cut it queues **462**
   statements and catches **354 of 452** errors (78.3%, precision 76.6%). The
   paper's own fitted RF, given an **oracle** threshold fitted on this very
   panel, needs **662** reviews for the same yield — **+200 reviews for identical
   yield**. This is the most robust form of the result: it survives max-t
   correction across the four arms and it **strengthens** on the
   adjudication-safe subpanel.
3. **Against the paper's fitted research RF** (§5), the ranking margin is small
   and fragile: **+0.0098 AP**, pointwise-significant but **grazing zero** under
   max-t, and **falling ~60%** on the adjudication-safe subpanel. Conditional on
   the statements it does not zero, the gate ranks **worse** than the paper's RF.
   It is a **detector, not a ranker**.

The durable finding needs no LLM at all (§7): the formula assigns **exactly
0.65** to one sentence from any of five readers whose measured accuracy on this
panel runs from **11.9% to 64.8%** (χ² = 30.7, p = 3.6e-6, n = 315).

---

## Why this exists

The main comparison already carried a **semantic reconstruction** of the paper's
method (our reimplementation of RF 2k-d13 + Type/#PMIDs/promoter) as a paired
arm, reproducing the published 0.942 PR-AUC. The fair objection is: *how do you
know the reimplementation is faithful?* The only way to close that is to run the
paper's **literal** released code on the paper's **released** corpus, then
(a) confirm it reproduces the published Table 6, (b) confirm our port matches it
statement-for-statement, and (c) score it against the reader arms on the
identical panel and metric.

## Provenance (everything is pinned)

- Paper code: `github.com/sorgerlab/indra_assembly_paper` @
  `63abdf1274d2f5534ed822585775031712916c83`. All four critical files
  (`Training Belief ML Models.ipynb`, `classifiers.py`, `belief_models.py`,
  `group_curations.py`) match `data/benchmark/indra_paper_2023.manifest.json`
  **byte-for-byte**.
- Assembled statements: `data/benchmark/indra_benchmark_corpus.pkl`
  (sha `ed64a240…`, 894,939 statements, verified). All 1,689 curated statement
  hashes are present.
- Curated labels + counts: the repo's own
  `data/curation/extended_curation_dataset.pkl` (1,689 rows).
- Evaluation engine: the released notebook's cells 15+17 (`TrainTestResult`,
  `eval_models_relation`, `random.seed(4)` → `StratifiedKFold(10, shuffle=False)`)
  copied **verbatim**. Featurizer is `indra.belief.skl.CountsScorer` under
  indra 1.24.0, whose keyword feature API is identical to the 2023 notebook.
  Classifiers are the paper's own `bioexp.curation.classifiers`.
- Published values: notebook cells 47+48, transcribed to
  `data/benchmark/indra_paper_2023_published_method_metrics.json` —
  **59 methods** across four input configurations. Our panel is
  `all sources, specific`, so **15** of those rows are directly comparable; the
  other 44 are context only and never enter a comparison.

Two documented deviations, neither material: one pandas modernization
(`.any(1)` → `.any(axis=1)`, identical row-wise semantics) and an explicit RF
`random_state=1` (the paper published no RF seed). The MCMC `Belief Orig` arms
were not re-run here (they are separately reproduced by
`scripts/reproduce_indra_paper_headlines.py`); the deterministic RF / Log LR
arms carry the headline.

---

# 1 · Fidelity — we ran your code, and it reproduces

## Result 1 — the literal reproduction matches the published Table 6

Every one of the 14 reproduced deterministic methods lands within **0.0016** of
its published fold-mean trapezoidal PR-AUC (the measured bound is
max |Δ| = **0.001552**; the run manifest declares the looser round-number
tolerance **±0.002** it was checked against), with fold SDs matching to the
published precision. The headline "all sources, specific" models:

| Method (all sources, specific) | Published | Reproduced | Δ |
| --- | --- | --- | --- |
| RF 2k-d13 + Type/#PMIDs/promoter | 0.942 ± 0.014 | 0.9413 ± 0.0140 | −0.0007 |
| RF 2k-d13 + Type/#PMIDs/prom/avglen | 0.942 ± 0.015 | 0.9418 ± 0.0149 | −0.0002 |
| RF 2k-d13 + Type/#PMIDs | 0.937 ± 0.015 | 0.9359 ± 0.0150 | −0.0011 |
| RF 2k-d13 | 0.928 ± 0.015 | 0.9273 ± 0.0151 | −0.0007 |
| Log LR + Type/#PMIDs/prom/avglen | 0.937 ± 0.013 | 0.9367 ± 0.0126 | −0.0003 |
| Log LR | 0.924 ± 0.014 | 0.9232 ± 0.0138 | −0.0008 |

Largest residual across all 14 arms (including the readers-only rows) is
`RF 2k-d13 - readers`, 0.893 published vs 0.8914 reproduced. The residuals are
consistent with **3-decimal rounding in the published table plus the unpublished
RF seed** — not with the seed alone: seven of the 14 arms are Log LR, which is
deterministic, and they carry residuals up to 0.0008 with no seed anywhere in
them.

*Artifacts:* `data/results/indra_paper_literal_models_20260724/paper_literal_table6_and_oof.json`,
published values in `data/benchmark/indra_paper_2023_published_method_metrics.json`.
Rendered by `FidelityPanel.svelte`.

## Result 2 — our semantic port is a near-bit-exact stand-in

Literal reproduction vs the semantic-port arm, per statement over all 1,689:
**Pearson r = 0.9994**, Spearman 0.9988, mean |Δprob| = 0.0055, max |Δprob| =
0.0398, fold-mean trapezoidal PR-AUC 0.9413 vs 0.9416. The reconstruction used
across the rest of the program is faithful; nothing downstream needs to change.
*Artifact:* `paper_literal_vs_llms.json` → `faithfulness_literal_vs_port`.

---

# 2 · What the reader arm actually is

## Result 3 — the framing correction: the reader arms *are* the unfitted noisy-OR

This is the part worth reading closely, because the label "LLM scorer" has been
misleading — including in our own earlier write-up of this comparison.

Each reader arm produces one verdict per `(statement, evidence)` pair. Those
verdicts are then rolled up by **INDRA's default-prior aggregation**: evidence
the reader rejected is dropped, a source with zero surviving evidence leaves the
product entirely, and the surviving counts go through
`belief = 1 − Π_s (syst_s + rand_s^{n_s})`. That is `SimpleScorer`. Four
independent checks:

**(a) By declaration.** All four reader manifests
(`data/comparison/models/{gemma_4_26b,gemma_4_31b,gemma_4_e2b,glm_5}/manifest.json`)
carry `implementation.notes.aggregation = "indra_default_hard_gate"` and
`implementation.notes.reader_profile = null`. In
`src/indra_belief/statement_belief.py::statement_belief`, `soft=None` (the
unfitted-reader path) dispatches to
`src/indra_belief/noise_model.py::compute_gated_belief` without soft weights —
the per-source noisy-OR above. The priors in `data/comparison/aggregation.json`
are INDRA's published defaults (reach/sparser/trips/medscan/rlimsp all
`(rand 0.30, syst 0.05)`), i.e. the same table `SimpleScorer` reads. The framing
artifact re-checks every manifest's aggregation-config sha and every component
digest against the tree.

**(b) By construction — it can only subtract.** Dropping evidence removes
factors from the product, so belief can only fall. We checked this rather than
assuming it: across all **1,689 statements × 4 reader arms = 6,756
comparisons**, there are **zero** statements where the reader's belief exceeds
the unfiltered `SimpleScorer` belief on the same statement, and the maximum
excess is exactly **0.0**. A true statement that the formula under-scores can
therefore **never be promoted** by a reader.

### Result 3c — by value: every non-zero reader score is a value the formula emits

Every non-zero reader score is a value `SimpleScorer` itself emits for some
sub-multiset of that statement's evidence. Taking each statement's
`source_counts` from
`current_indra_simple_default_prediction_provenance.jsonl`, enumerating the
reachable set `{1 − Π_s f_s : f_s ∈ {1} ∪ {syst_s + rand_s^k, 1≤k≤n_s}}`, and
matching. The chance-floor column is the pairing the page's own table carries —
the match is only worth reading beside it.

| Arm | non-zero scores | on a reachable value | chance floor (permuted) | exactly 0.0 |
| --- | --- | --- | --- | --- |
| Gemma 4 26B | 1,227 | 1,227 (100%) | 46.8% (45.4–47.7%) | 462 (27.4%) |
| Gemma 4 31B | 1,211 | 1,211 (100%) | 45.2% (44.2–46.5%) | 478 (28.3%) |
| Gemma 4 E2B | 1,296 | 1,296 (100%) | **51.4%** (50.3–52.0%) | 393 (23.3%) |
| GLM-5 | 1,148 | 1,148 (100%) | 45.5% (44.2–46.6%) | 541 (32.0%) |

Match is **100%**, to within **9e-15**. Re-enumerating the reachable set in the
same sorted-source association order that `compute_gated_belief` actually
multiplies in tightens that to **bit-exact float equality on every arm**:
**1,227/1,227** for Gemma 26B, **1,211/1,211** for 31B, **1,296/1,296** for E2B,
**1,148/1,148** for GLM-5 — `share_bit_exact = 1.0` four times over, with
**zero counterexamples** and **zero statements left unsettled by the search**
(`framing_correction.json` → `reachable_values.arms[*]`: `n_bit_exact`,
`n_counterexamples`, `n_budget_exhausted`). The search is depth-first over
`sorted(sources)`, with the feasible factor window at each depth binary-searched
from a suffix min/max product interval; the worst statement on this panel used
**1,155,323** nodes against a **5,000,000**-node per-statement budget
(`reachable_values.search`). An earlier draft reported a small unsettled
remainder on each arm and read it as reachable sets too large to exhaust. That
was a property of the older, coarser search, not of the data: with the window
binary-searched every one of them settles, and none of them was ever a failure
to match.

The null baseline matters, because reachable values are not rare. Permuting each
arm's own non-zero scores across statements and re-running the same membership
test against the *recipient* statement's set gives the chance-floor column above:
**0.468** for 26B, **0.452** for 31B, **0.455** for GLM-5 — and **0.514** for
**Gemma 4 E2B**, the arm with the most non-zero scores and the only one whose
floor sits above one in two. Pooled across the four arms the floor is **0.472**
(range 0.442–0.520), but the pooled figure understates E2B, so we quote it only
beside that name. Basis, as on the page: **10 permutations, seed 20260717**, over
the **1,635 of 1,689** statements whose reachable set is exhaustively enumerable;
the 54 left out have the *largest* reachable sets, so including them could only
raise the floor. So the claim is 100% against a chance floor of 0.452 to 0.514
depending on the arm.

The reachable-value check asks whether the formula **can** emit each score for
some surviving-count vector, not whether it emitted it for the vector the reader
actually produced; two different vectors can land on the same value. Leg (a) is
what pins the aggregation itself. The zero block is the reader rejecting *every*
evidence it read, leaving an empty product. `SimpleScorer` itself never emits 0
on this panel — its floor is **0.65**, one reach/sparser/trips/medscan/rlimsp
sentence.

### Result 3d — what rides along that is not reading

Three subtractions sit inside "the evidence the reader kept" and none of them is
an LLM judgement, so we state them before claiming any gain. The execution map
records **33,361** unique `(statement, evidence)` pairs across five routes:
`plain` 29,640, `tool` 2,670, `no_text` 560, `deterministic_mismatch` 488,
`deterministic_pseudogene` 3. Only `plain` and `tool` are readable.

- **De-duplication.** The manifests declare `dedup: true`. The reader panel is
  built from **unique** pairs (33,361) while the unfitted `SimpleScorer` counts
  all **34,035** evidences — **674 excess pairs across 327 statements**,
  reproducible from the execution map alone. Production `statement_belief`
  collapses roughly 40 further within-source text-normalized near-duplicates
  (714/339 total), which needs the production de-dup pass to reproduce rather
  than the execution map.
- **560 `no_text` rows** — evidence with no sentence to read — are skipped
  before the gate ever runs.
- **488 `deterministic_mismatch` + 3 `deterministic_pseudogene`** rejections come
  from our own grounding rules, not from the model.

The control is to run exactly those three subtractions with **no LLM verdicts at
all** (every readable pair accepted) and score the result. Re-derived from the
execution map and `data/comparison/aggregation.json`:

| Arm | evidence scored | pooled AP | Δ vs ungated |
| --- | --- | --- | --- |
| noisy-OR SimpleScorer (direct), every evidence entry | 34,035 | 0.9031 | — |
| de-dup only | 33,361 | 0.9027 | −0.0005 |
| de-dup + drop evidence with no sentence | 32,801 | 0.8982 | −0.0049 |
| **de-dup + no-sentence + deterministic rejects, no LLM (control)** | **32,310** | **0.9017** | **−0.0014** |
| Gemma 26B (same subtractions **plus** reading) | 32,310 | 0.9510 | +0.0479 |

(Rows and their wording are the artifact's own, `non_reading_control.json` →
`rows[]`. The keys `raw`, `dedup_only`, `dedup_plus_no_text`, `full_control` are
frozen joins and still carry the execution map's route name `no_text`; every
reader-facing label says "no sentence" in words instead.)

The non-reading subtractions do not produce the gain — together they land
**0.0014 below** the ungated noisy-OR. Against the control specifically the
reader is worth **+0.0493**. The table is the execution-map pass; our production
`statement_belief` de-dup pass gives 0.9025 / 0.8981 / 0.9015, a ~0.0002 scope
difference. The conclusion is identical either way: the control sits *below* the
noisy-OR baseline, not above it, so the gain is not an artifact of what we
removed before reading. One disclosure the artifact makes and we repeat: **35
statements lose ALL their evidence to the deterministic rejects** and score
belief 0 in the full control. That is a property of our grounding rules, not of
any reader.

**So:** what follows measures what *reading the sentence* adds to that
aggregation, holding the aggregation fixed. Any gain is attributable to evidence
selection, and any gain has a matching cost in the direction the gate cannot go.

---

# 3 · THE HEADLINE — against the belief INDRA actually serves

This is the comparison a deployment decision turns on, and it is the one the
memo previously buried.

**The 2023 paper's random forest was a research model and was never deployed.**
It is not in `indra`, it has never been served, and its released out-of-fold
predictions exist only on the paper's own panel. What INDRA *ships* and computes
for every statement today is the unfitted noisy-OR: `indra.belief.SimpleScorer`
over `indra/resources/default_belief_probs.json`. That is the incumbent, and
that is what a reading gate has to beat to be worth anything.

## The paper's own panel

On the paper's own 1,689 statements with the paper's own released labels:

| Arm | AUROC | AP | what it is |
| --- | --- | --- | --- |
| INDRA `SimpleScorer`, run directly | 0.7740 | 0.9031 | the shipped default-prior noisy-OR |
| **INDRA `SimpleScorer` + hierarchy propagation** | **0.7823** | **0.9030** | **the same scorer plus the supports-graph propagation INDRA also applies — the drawn incumbent** |
| paper RF 2k-d13 + Type/#PMIDs/promoter | 0.8516 | 0.9412 | the paper's fitted research model, out-of-fold |
| Gemma 4 26B reader gate | 0.9010 | 0.9510 | the same INDRA scorer at the same priors, over the evidence the reader kept |

**Headline: +0.074 AUROC** against the strongest served incumbent (the fitted
HybridScorer, not `SimpleScorer` — see the floor refutation above), 95% CI
**[+0.0992, +0.1390]**, 10,000 resamples, seed 20260727.

**Quote +0.074.** Both +0.127 and +0.1187 rest on a refuted premise about what
INDRA serves. The larger figures are deltas against the
*no-propagation* variant. Propagation makes the incumbent **stronger** (0.7823 vs
0.7740), so the propagated variant is the comparator. Quoting +0.127 would walk
around our own conservative-baseline rule to buy 0.008 AUROC.

For reference, the same panel's other two gaps: the paper's research RF beats the
served incumbent by **+0.0693**, and the gate beats the paper's research RF by
**+0.0494** [+0.0313, +0.0680].

## It replicates on four panels

`deployed_baseline_replication.json` runs the same contrast on four panels.
**Every panel favours the gate, and every interval excludes zero.**

| Panel | n | curators | drawn incumbent | incumbent AUROC | gate AUROC | Δ | 95% CI |
| --- | --- | --- | --- | --- | --- | --- | --- |
| the 2023 paper's own panel | 1,689 | the paper's own | CoGEx fitted `HybridScorer` (strongest form) | 0.8272 | 0.9010 | **+0.0738** | [+0.0530, +0.0950] |
| 32 external curators | 464 | 32 | belief INDRA served | 0.6882 | 0.8010 | **+0.1128** | [+0.0557, +0.1722] |
| out-of-distribution holdout | 414 | — | belief INDRA served | 0.6100 | 0.7202 | **+0.1102** | [+0.0503, +0.1698] |
| two of the paper's authors | 913 | 2 | `SimpleScorer`, recomputed on full evidence | 0.7364 | 0.7792 | **+0.0428** | [+0.0050, +0.0800] |

AUROC is the cross-panel metric because the base rates differ (0.73 on the
paper's panel; 0.50 balanced by construction on two of the others; 0.45 on the
holdout). **Average precision must not be compared across these panels** and is
reported per panel only.

### Read this as four panels, not as one comparison four times

Three honest qualifications, in descending order of how much they matter. An
author will find all three, so we state them rather than wait.

**(i) The comparator is not the same object on every row.** Three different
sourceable forms of INDRA's deployed belief are drawn, one per panel, and they
are genuinely different things:

- `SimpleScorer` **+ hierarchy propagation** (paper panel) — we recompute the
  noisy-OR from the statement's INDRA source counts and then apply the
  supports-graph propagation, from
  `current_indra_hierarchy_paper_20260717/`.
- `SimpleScorer`, **recomputed** (author panel) — the noisy-OR recomputed from
  the statement's own INDRA source counts, **without** propagation.
- **belief INDRA served** (external-curator and holdout panels) — not recomputed
  at all: the `stored_belief` value INDRA's own pipeline *wrote onto the
  statement*, read back as-is. Whether propagation is baked into that stored
  value is a property of the upstream pipeline that produced it, which we read
  rather than re-derive; the artifact's note asserts it, and that assertion is
  weaker than the two recomputed rows above. Treat the served-belief rows as
  "what the pipeline emitted", which is exactly what a deployment comparison
  wants, and not as a verified statement about propagation.

The selection rule is a deliberate **argmax**: every sourceable form is scored on
each panel and the one with the **highest** AUROC is drawn. That rule is
conservative — it costs us **0.025** AUROC on the author panel (0.7355 drawn
instead of the served-belief variant's 0.7102) and **0.008** on the paper panel —
and it must stay. But it means the incumbent *label* differs by row, and the
figure must not be read as "the same comparison, four times".

**(ii) The panels differ in evidence per statement**, which is the gate's whole
input. The paper panel carries **~19.8** unique evidence reads per statement
(33,361 unique pairs / 1,689). The curation panels carry far fewer: **1.76** on
the author panel (1,606 rows / 913), **1.25** external (587 / 469), **1.21**
holdout (500 / 414). A subtractive gate has much more to work with on the paper
panel, and the deltas are ordered accordingly.

**(iii) Four panels is replication of a DIRECTION, not a meta-analysis.** The
panels differ in curator, corpus, sampling and reader deployment; the intervals
are per-panel and are not pooled. Two further per-panel flags travel with the
data: the author panel is **in-sample** for our *soft* calibration profile (the
arm drawn uses INDRA's default priors and no fitted parameter, so what is
plotted is still out-of-sample, but the flag travels with the panel); and the
holdout panel joined entirely by **truth-safe source fallback** (0 of 500 rows
joined by exact `(matches_hash, source_hash)`). The external-curator panel
excludes 5 statements as undefined.

**A sensitivity row the figure deliberately does not draw.** Our *production*
hard gate also swaps INDRA's priors for the recalibrated ones. Carried so the
prior swap can be priced: 0.7989 on the author panel (Δ +0.0634), 0.7940
external (Δ +0.1058), 0.7340 holdout (Δ +0.1240). It is not drawn because it
changes two things at once.

*Artifact:* `data/results/deployed_baseline_replication_20260727/deployed_baseline_replication.json`,
generated by `scripts/compute_deployed_baseline_replication.py`;
rendered by `DeployedBaseline.svelte`.

---

# 4 · THE OPERATIONAL RESULT — 462 reviews against an oracle-tuned 662

The most intuitive form of the finding, and the most robust one.

The 2023 paper published no decision or threshold metric — only trapezoidal
PR-AUC as a 10-fold mean. **Everything in this section is our derivation applied
to INDRA's belief models, not something the paper reported.** Label convention,
stated once and identically to the page: we use the paper's OWN
`paper_replication_policy.released_paper_correct`, so "error" means "the paper's
released label says incorrect" — n = 1,689, **452 errors**, base rate 26.8%. Of
those 452, **341** are adjudication-safe negatives with complete evidence and
**111** carry `label_is_adjudication_safe: false` / `differs_from_adjudicated_gold:
true`. We keep the paper's labelling because this is a comparison *to the paper*,
and it is applied identically to every arm.

## 4a · Equal yield: how many reviews to catch the same errors

The gate has an operating point **nobody chose**: the block of statements whose
evidence the reader rejected outright. No threshold was fitted; the block is what
it is.

> **Gemma 4 26B gate — queue 462, catches 354 of 452 errors (78.3%), precision
> 76.6%, 108 true statements wrongly queued (8.7% of the panel's 1,237 true).**

Now ask each comparator how many reviews it needs for that same yield of 354.
Each comparator's threshold is chosen **on this panel, with these labels already
in hand**, to land exactly on 354. That is an **oracle** threshold: it is fitted
and evaluated on the same 1,689 statements, it would not be available before
curation, and it **favours the comparator**. The gate is given no such help. The
comparison is stated in this direction deliberately.

| Comparator (oracle-thresholded) | reviews for 354 errors | extra reviews | precision there | caught at the gate's own 462 budget |
| --- | --- | --- | --- | --- |
| **paper RF 2k-d13 + Type/#PMIDs/promoter** (the paper's fitted model) | **662** | **+200** | 53.5% | **274 (60.6%)** |
| our full-feature `CountsScorer` RF, current INDRA | 665 | +203 | 53.2% | 276 |
| our `BayesianScorer`, source+subtype refit | 754 | +292 | 46.9% | 253 |
| INDRA `SimpleScorer` + hierarchy propagation (served) | 760 | +298 | 46.6% | 234.0 |
| INDRA `SimpleScorer`, unfitted noisy-OR (served) | 768 | +306 | 46.1% | 219.0 |

**662 and 665 are different arms, and both are correct.** 662 is the **paper's
own** literal RF (`RF 2k-d13 + Type/#PMIDs/promoter - all sources, specific`,
verified directly against its released out-of-fold predictions: strict index 662,
tie-group-complete 662, tie-group size 1 at the boundary, expected catch exactly
354.0). 665 is **ours** — the full-feature `CountsScorer` RF on current INDRA, and
**not** the paper's model. Earlier drafts quoted 665 as the paper's number; that
was a misattribution the old frozen key (`paper RF (full features)`) invited. Both
arms now ship in `statement_review_queue.json` under names that say whose they
are, alongside an explicit per-arm `provenance` field, and the artifact's derived
comparator — the belief model needing the fewest extra reviews — is the paper's
own RF at 662.

**Tie rule.** Where a budget falls inside a block of tied scores the arm cannot
say which of the tied statements to read first, so the block contributes its
errors **pro rata** and the count is the *expected* number of errors found
reading a uniformly random prefix. No arbitrary tie order is ever imposed: the
same panel scored in a different row order gives the same curve. This matters
here because the gate's operating point *is* a 462-statement tie block.

The other three reader arms, same rule, against the same five comparators
(`equal_yield.references`). Quoting the paper's own RF: GLM-5 queues 541 for 390
errors where the RF needs 766; the 31B gate queues 478 for 358 where it needs
679; E2B — the capacity control — queues 393 for 266 where it needs only 447.
That last row is the one to keep in view: against a weak enough reader the
supervised RF nearly closes the gap on its own. (Against **our** full-feature
`CountsScorer` RF the same three rows read 795 / 676 / 441 — close enough that
the choice of comparator does not carry the argument, which is why the artifact
derives it rather than naming it.)

## 4b · The advantage is a property of the budget, and we show it

At **200** reviews the gate's advantage over the paper's RF is **−1.8 errors** —
nothing. It turns positive at **225** (+3.4), reaches **+67.5** at 400, **peaks
at +80.0 at exactly 462** (the gate's own operating point), and decays: **+44.2**
at 600, **+34.9** at 700, half-peak at 700, closing to zero once the budget covers
the panel. Errors caught at a 200 / 400 / 600 budget:

| Arm | @200 | @400 | @600 |
| --- | --- | --- | --- |
| Gemma 4 26B gate | 153.2 | 306.5 | 382.2 |
| Gemma 4 31B gate | 149.8 | 299.6 | 383.1 |
| GLM-5 gate | 144.2 | 288.4 | 398.4 |
| Gemma 4 E2B gate | 135.4 | 268.3 | 333.3 |
| **paper RF 2k-d13 + promoter (comparator)** | 155.0 | 239.0 | 338.0 |
| our full-feature `CountsScorer` RF | 155.0 | 247.0 | 332.0 |
| `BayesianScorer` source+subtype | 124.5 | 224.0 | 308.0 |
| `SimpleScorer` + hierarchy (served) | 109.0 | 204.7 | 307.4 |
| unfitted noisy-OR (served) | 105.5 | 208.5 | 290.1 |

Read honestly: at a small budget the gate has no advantage, because its
advantage *is* the zero block and the zero block is 462 statements wide.

## 4c · The detection margin survives correction; the ranking margin does not

Error recall at a **25% review budget** (422 of 1,689), same paired
fold-stratified bootstrap design as the AP margin (10,000 resamples, seed
20260717, one shared index vector per resample), studentized **max-t** across the
family of four reader arms. Reference = the paper's literal RF, whose recall at
that budget is **0.5520**.

| Arm | error recall @25% | Δ vs paper RF | simultaneous max-t band |
| --- | --- | --- | --- |
| **Gemma 4 26B** | **0.7154** | **+0.163** | **[+0.106, +0.221]** |
| Gemma 4 31B | 0.6992 | +0.147 | [+0.091, +0.204] |
| GLM-5 | 0.6730 | +0.121 | [+0.068, +0.174] |
| Gemma 4 E2B | 0.6094 | +0.057 | [+0.005, +0.110] |

**All four exclude zero simultaneously.** On the adjudication-safe subpanel
(n = 1,578, 111 incomplete-review negatives removed) the 26B margin **rises** to
**+0.188** [+0.123, +0.253].

*Tie-convention disclosure.* The table uses the same **pro-rata** rule as §4a —
the expected recall over orderings of the gate's tie block. A single arbitrary
ordering of that block realises a nearby value: in `sorted(statement_id)` order
the 26B margin is **+0.157**, simultaneous [+0.094, +0.221], and the subpanel
value is **+0.188** either way. Every qualitative claim holds under both; the
pro-rata figure is quoted because it is order-independent and matches the equal-
yield rule the shipped artifact enforces.

**Contrast this with the ranking margin (§5), on the identical bootstrap.** The
26B AP margin is +0.0098 with simultaneous band **[−0.0004, +0.0200]** — it does
**not** exclude zero — and it **falls 60%** to +0.0039 on the same
adjudication-safe subpanel. The detection margin and the ranking margin behave in
opposite directions under exactly the same two stress tests. That is the
strongest single reason to describe this system as a **detector**.

## 4d · The full review queue at a matched recall target

Rule: flag a statement as wrong iff belief ≤ τ, with τ the smallest of that arm's
own distinct scores whose flag set catches at least 70% of the 452 errors.

| Arm | queue | errors caught | false alarms | precision | recall achieved |
| --- | --- | --- | --- | --- | --- |
| noisy-OR `SimpleScorer` (direct) | 676 | 325 | 351 | 48.1% | 71.9% |
| `BayesianScorer`, source+subtype refit | 622 | 317 | 305 | 51.0% | 70.1% |
| `CountsScorer` RF, full features | 563 | 317 | 246 | 56.3% | 70.1% |
| Gemma 4 26B gate | 462 | 354 | 108 | 76.6% | 78.3% |
| Gemma 4 31B gate | 478 | 358 | 120 | 74.9% | 79.2% |
| GLM-5 gate | 541 | 390 | 151 | 72.1% | 86.3% |

The first three rows are **not** the paper's published arms (§ naming note). The
literal 2023 model is not scored at this grain at all, because the paper
published no decision rule.

**Read this as dominance on both axes, not as a matched-recall comparison.** The
arms do not achieve equal recall: a reader arm's lowest score is one indivisible
tie block, so it cannot land on 70% and overshoots. What is true is that the
three strongest reader arms each have a **shorter queue** *and* catch **more
errors** than every belief-model arm at this target. Gemma 4 E2B — the weakest
reader, carried as the capacity control — does **not**: it needs a longer queue
than `CountsScorer` RF full features (642 vs 563) at lower precision (54.0% vs
56.3%). Reading only pays if the reader is good enough, and the figure draws the
arm that proves it.

**And this is an operating-point property, not a universal one.** Raise the
target to 80% and the *queue-length* advantage disappears for the two Gemma
arms, while the *catch-rate* advantage persists:

| Arm @ 80% target | queue | errors caught | precision |
| --- | --- | --- | --- |
| `CountsScorer` RF, full features | 687 | 362 | 52.7% |
| Gemma 4 26B gate | 687 | 400 | 58.2% |
| Gemma 4 31B gate | 682 | 400 | 58.7% |
| GLM-5 gate | 541 | 390 | 72.1% |
| noisy-OR `SimpleScorer` (direct) | 785 | 362 | 46.1% |
| `BayesianScorer`, source+subtype refit | 782 | 362 | 46.3% |

Only the queue length ties (687 vs 687): reading the identical number of
statements, the 26B gate still catches **38 more errors** at **5.5 points higher
precision**. GLM-5 does not move at all — its 70%-target flag set already clears
80%. The reader arms are coarse instruments: Gemma 26B returns the identical
462 statements at the 50%, 60% and 70% targets, then jumps to 687 with nothing
in between.

*Artifact:* `data/results/indra_paper_literal_models_20260724/statement_review_queue.json`,
generated by `scripts/compute_statement_review_queue.py`; rendered by
`ReviewQueue.svelte`. Its `arms[].name` values are **join keys** (they also index
`provenance.scores`, `promotion_ceiling.reference_arm` and — for the four reader
gates only — `promotion_ceiling.per_arm_overlap`), and at `schema_version` 4 they
say whose model each one is: `INDRA SimpleScorer, default priors`,
`INDRA SimpleScorer + hierarchy`, `Paper RF 2k-d13 + Type/#PMIDs/promoter`, the
four gates, then `Our BayesianScorer, source+subtype refit` and
`Our CountsScorer RF, full features`. Each also carries an explicit `provenance`
(`indra-served` / `paper-published` / `our-reimplementation` / `reader-gate`), and
the paper-published key is asserted verbatim against the 59 method strings in
`data/benchmark/indra_paper_2023_published_method_metrics.json`. The bootstrap
behind §4c ships in the same file under `error_recall_robustness`; the viewer
reads it rather than recomputing it, and its budget (422) is on the sweep grid so
the interval is quoted at a point the drawn curve passes through.

*Version numbering.* `schema_version` skips 3 deliberately and permanently: the
shipped viewer contract runner uses the literal `schema_version = 3` as its
negative control for "an unknown version fails closed", so emitting a 3 would
turn that assertion into a silent pass. No 3 was ever written.

---

# 5 · Against the paper's fitted RF — the harder, smaller bar

Everything in this section qualifies **one** margin: the reading gate against the
paper's own fitted research model on ranking metrics. It is the hardest bar in
the memo and the one we do worst on. Its limits are clustered here deliberately;
they are not doubts about §3 or §4.

## 5a · The head-to-head, in the paper's metric first

Identical 1,689 "all sources, specific" statements, released paper labels,
joined `stmt_hash ↔ statement_id` through the frozen gold. Fold-mean
**trapezoidal** PR-AUC ± population SD over the paper's own 10 folds — the
published estimator:

| Arm | fold-mean trapezoidal PR-AUC |
| --- | --- |
| RF 2k-d13 + Type/#PMIDs/promoter | 0.941 ± 0.014 |
| RF 2k-d13 + Type/#PMIDs/prom/avglen | 0.942 ± 0.015 |
| Our port of RF + Type/#PMIDs/promoter | 0.942 ± 0.014 |
| Gemma 4 26B | 0.961 ± 0.011 |
| GLM-5 | 0.965 ± 0.010 |
| Gemma 4 31B | 0.960 ± 0.011 |
| Gemma 4 E2B | 0.935 ± 0.017 |
| INDRA CoGEx hybrid | 0.923 ± 0.023 |

That ± is a **dispersion** measure over folds. It is not a confidence interval.

"INDRA CoGEx hybrid" is INDRA's currently deployed `HybridScorer` — a belief
model with no LLM anywhere in it, included as a present-day non-reader baseline,
not as another reader arm.

*Artifact:* `paper_literal_vs_llms.json` → `point_metrics`. Its keys are **frozen
join strings** from the emitting run and are not arm names: `Paper literal
RF+promoter`, `Paper literal RF+prom/avglen` and `Paper semantic port
RF+promoter` address, in order, the three non-reader rows above. "Paper literal"
is our coinage for *the paper's own released code run as-is*; the first two rows
are the paper's published methods under their published names, the third is our
port of the first.

## 5b · That metric flatters the reader arms, and we do not use it as the verdict

Trapezoidal interpolation over-credits heavily-tied score distributions. The
literal `RF 2k-d13 + Type/#PMIDs/promoter` emits **1,546** distinct scores over
1,689 statements — effectively continuous, so trapezoidal ≈ average precision
(inflation +0.0001). The reader arms emit **420–498** distinct scores, and
trapezoidal inflates them by **+0.010 to +0.014**. Read as a cross-arm verdict,
the paper's estimator roughly **doubles** the apparent reader advantage: Gemma
26B looks like **+0.020** against the literal model in trapezoidal, and is
**+0.0098** in tie-robust average precision. **We quote +0.0098.**

## 5c · Standard metrics on the identical panel

Pooled average precision (tie-aware) and AUROC, with paired bootstrap ΔAP
against the literal model (10,000 resamples, seed 20260717):

| Arm | AP | AUROC | ΔAP vs literal [95% CI] |
| --- | --- | --- | --- |
| RF 2k-d13 + Type/#PMIDs/promoter | 0.9412 | 0.8516 | — (reference) |
| Our port of RF + Type/#PMIDs/promoter | 0.9412 | 0.8519 | +0.0000 [−0.0005, +0.0006] |
| Gemma 4 26B | 0.9510 | 0.9010 | **+0.0098 [+0.0012, +0.0185]** |
| GLM-5 | 0.9507 | 0.9025 | **+0.0095 [+0.0007, +0.0184]** |
| Gemma 4 31B | 0.9489 | 0.8979 | +0.0077 [−0.0012, +0.0167] |
| Gemma 4 E2B | 0.9252 | 0.8400 | **−0.0159 [−0.0256, −0.0061]** |
| INDRA CoGEx hybrid | 0.9227 | 0.8272 | **−0.0183 [−0.0304, −0.0076]** |

Bold = 95% CI excludes zero. **These CIs are uncorrected for the 7 comparisons
made against a single reference**; see §5d for the corrected version. Read them
as descriptive.

**The AUROC gain here must NOT be read as a ranking result.** It is almost
entirely the binary zero/non-zero split. Two facts pin that down:

- The bare bit "the reader kept ≥ 1 evidence" — thrown away as a single
  bit, with no score at all — scores AUROC **0.8479** / AP **0.9037**. That is
  nearly the paper RF's entire continuous model (0.8516 / 0.9412) and it is
  **above** INDRA's served noisy-OR (0.7823 / 0.9030). The discriminative power
  *is* the binary decision.
- Restricted to the statements each arm actually orders (score > 0), three of the
  four arms rank **worse** than the paper's model: 26B **0.7683 vs 0.7881**, 31B
  0.7681 vs 0.7820, E2B 0.7630 vs 0.8158; only GLM-5 is marginally better at
  0.7665 vs 0.7599.

The gate's value is in deciding what to drop, not in ordering what it keeps.

**One asymmetry runs the other way and we name it rather than bank it.** The
literal `RF 2k-d13 + Type/#PMIDs/promoter` is scored **out-of-fold on this very
panel** — it saw 90% of these statements in training at every fold — while the
reader arms are **never fitted on this panel**. The reader arms are not zero-shot
either: each call carries a fixed set of hand-authored worked examples — 9
prefixes × 28 messages = **14 demonstration pairs per call** — but they are
hand-written and no parameter is estimated from these labels — but they are **not
disjoint from this panel**, and an earlier version of this memo said they were.

**Correction (2026-07-27).** The disjointness claim was measured with a parser
that read only `members`, a field carried by Complex statements alone, so it saw
692 of 1,689 statements (41%) and *could not match* any of the 997 binary
statements. Its own vacuity guard passed because bare agent names intersect. With
the parser reading both statement shapes (1,680 keys), the real figure is that
**11 of the 45 demonstration claims share an (agent set, statement type) with a
panel statement** — ADAM17/TGFB1, AGER/MMP2, AKT/CASP3, Actin/CDK9, BRMS1/CUL3,
GSK3B/ILK, IFNA/NFkappaB, IL1/IL6, MMP14/SPRY4, MYB/PPID, RAD18/TP53BP1.

That is a WEAK form of exposure — the same entity pair and relation, not the same
sentence — and its measured effect is negligible: those claims touch **11 of 1,689
statements (0.7%)**, and excluding them moves the gate-minus-served-belief AUROC
margin from +0.1270 to +0.1259, a shift of **−0.0011** against a +0.126 effect.
The stronger form, a demonstration's evidence sentence appearing verbatim, is
separately measured at **12 of 5,379 reviewed evidence pairs (0.22%)**, max AUROC
shift **0.0008**. Both are disclosed; neither changes a conclusion. The guard
(`tests/test_paper_panel_fewshot_disjoint.py`) now pins the measured overlap and
fails if it grows, rather than asserting a zero that was never true. This panel
sits outside
`scripts/check_contamination.py`'s default eval set, so the check is specific to
it. Pretraining contamination is a separate matter and cannot be excluded: the
benchmark corpus and the paper's repo are public. Weak evidence against gross
memorisation is that E2B — same family, same prompts, presumably the same
crawl — loses. The asymmetry favours the paper model, and the reader margins are
measured across it.

## 5d · How far to trust that margin: multiplicity and label completeness

`data/comparison/run_plan.json` was frozen **2026-07-22**, before the comparison
bundles were generated. It stages **four** reader arms and designates **none** as
confirmatory, so we cannot claim a pre-registered single arm and a simultaneous
band over the family of four is a fair ask. The arms are strongly rank-correlated
(score Spearman 0.836–0.980), so studentized max-t over the shared bootstrap
draws costs **2.301** SEs where Bonferroni would cost 2.498 and a pointwise
normal interval costs 1.960.

| Arm | ΔAP primary (n=1,689) | pointwise 95% | simultaneous max-t | ΔAP adjudication-safe (n=1,578) | simultaneous max-t |
| --- | --- | --- | --- | --- | --- |
| Gemma 4 26B | +0.0098 | [+0.0012, +0.0185] | **[−0.0004, +0.0200]** | +0.0039 | [−0.0029, +0.0108] |
| GLM-5 | +0.0095 | [+0.0007, +0.0184] | [−0.0009, +0.0198] | +0.0009 | [−0.0061, +0.0079] |
| Gemma 4 31B | +0.0077 | [−0.0012, +0.0167] | [−0.0029, +0.0183] | +0.0031 | [−0.0038, +0.0100] |
| Gemma 4 E2B | −0.0160 | [−0.0256, −0.0061] | **[−0.0275, −0.0045]** | −0.0096 | **[−0.0170, −0.0022]** |

**No reader arm's positive AP margin excludes zero simultaneously.** The only
simultaneously-significant AP result on this panel is E2B's, and it is
**negative**. On the adjudication-safe subpanel the positive margins fall by
60% (26B), 60% (31B) and 90% (GLM-5); E2B's negative margin shrinks by 40% and
stays significant.

The sensitivity panel removes the **111** negatives carrying
`label_is_adjudication_safe == false` — every one a negative whose evidence
review is incomplete. They ARE negative in the paper's released labels, so
dropping them is **our** revision of **their** labels; it is a sensitivity check
and never the primary result. It is not free: it removes **24.6%** of all
negatives and moves the panel from 26.8% to 21.6% negative, so it changes what
the panel is a sample of, not only its label quality. No model is refit; the same
prediction vectors are scored on fewer statements.

**Power.** The pointwise CI half-width is 0.0087–0.0097 across the arms — the
same order as the effect it is measuring. This panel is **underpowered** for an
effect this size, rather than silent about it.

**Dose-response.** Exactly one arm has a negative delta and it is the smallest
(Gemma 4 E2B, the family's edge variant). No finer size ordering is claimed —
GLM-5's parameter count is not published — so the checkable part is the sign
pattern alone.

*Artifact:* `paper_margin_robustness.json`, generated by
`scripts/compute_paper_robustness.py`, which asserts that every pointwise number
it recomputes reproduces the shipped `paired_delta_vs_paper_literal` value to
1e-12 (worst residual: **0.0**).

## 5e · Where that margin comes from — and a retraction

**RETRACTED: "among the statements the paper's model already ranks best, reading
makes the ranking worse."** Earlier drafts of this memo, and an earlier version
of the `/paper` panel, cut the panel into deciles of the paper RF's **own
out-of-fold score** and reported that every reader arm was negative in the
paper's top five bands and took the whole gain back in the bottom two. Those
numbers are real. The conclusion drawn from them is not.

The banding variable was the reference arm's own score — one of the two
quantities being differenced. Banding on it conditions on that arm's estimation
noise, and the decomposition regresses to the mean. The decisive check is the
**mirror**: band the identical decomposition of the identical arm on the
*compared* arm's own score instead, and the story **reverses sign**.

| Banding variable | kind | 26B head (D1–D5) | 26B tail (D9+D10) | tilt |
| --- | --- | --- | --- | --- |
| the paper RF's own score *(the old banding)* | endogenous | **−0.52** | **+0.95** | **−1.47** |
| the 26B gate's own score *(the mirror)* | endogenous | **+1.01** | **−0.97** | **+1.98** |
| the unfitted noisy-OR belief | exogenous | +0.35 | +0.09 | +0.26 |
| evidence entries per statement *(the drawn banding)* | exogenous | +0.51 | +0.09 | +0.42 |

In AP points (1 pt = 0.01 AP). Three of the four arms — 26B, GLM-5, 31B — have
their tilt **sign reverse** under mirroring. Max |tilt| under endogenous banding
is **2.23** points; under exogenous banding it is **0.42**. The tilt was an
artifact of the axis.

The full endogenous-banding table survives, because it is the evidence for the
retraction, not a casualty of it:

| Arm | D1–D5 (head) | D9+D10 (tail) | tilt |
| --- | --- | --- | --- |
| Gemma 4 26B | −0.5162 | +0.9518 | −1.4680 |
| GLM-5 | −0.4806 | +1.0111 | −1.4918 |
| Gemma 4 31B | −0.6810 | +0.9526 | −1.6335 |
| Gemma 4 E2B | −1.7961 | +0.4387 | −2.2348 |

### The decomposition as it now stands: banded on evidence count

The drawn banding is a **power-of-two ladder on evidence entries per
statement** — an integer census of the corpus, fixed before any model ran, not a
score, not fitted to any label, carrying no arm's estimation noise, and so having
**no mirror to reverse under**. It is also the noisy-OR's own saturating input.
(The unfitted noisy-OR belief is the other exogenous candidate and is rejected:
the gate is purely subtractive, so no reader belief ever exceeds it — verified,
0 of 6,756 — which makes it each arm's own ceiling. It is kept as a mirror
diagnostic instead.) The census agrees with the shared gold field and with the
paper's own released per-source counts on all **1,689** statements; band
membership is a pure function of the count, so **0** statements are assigned by a
tie-break.

| Evidence entries | n | true | false | error rate | 26B | GLM-5 | 31B | E2B | RF+prom/avglen |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 328 | 155 | 173 | 52.7% | +0.087 | +0.168 | +0.152 | −0.494 | −0.004 |
| 2 | 146 | 99 | 47 | 32.2% | +0.146 | +0.091 | +0.149 | −0.285 | −0.032 |
| 3–4 | 219 | 136 | 83 | 37.9% | +0.069 | +0.132 | +0.061 | −0.320 | +0.000 |
| 5–8 | 259 | 193 | 66 | 25.5% | +0.217 | +0.233 | +0.143 | −0.252 | +0.044 |
| 9–16 | 294 | 248 | 46 | 15.6% | +0.185 | +0.012 | +0.060 | −0.178 | +0.024 |
| 17–32 | 243 | 217 | 26 | 10.7% | +0.136 | +0.162 | +0.115 | −0.128 | +0.027 |
| 33+ | 200 | 189 | 11 | 5.5% | +0.139 | +0.147 | +0.091 | +0.056 | +0.021 |
| **total** | 1,689 | 1,237 | 452 | 26.8% | **+0.979** | **+0.946** | **+0.771** | **−1.601** | **+0.080** |

Banded this way the three strong reader arms are **positive in all 7 bands**
(`n_bands_agreeing_with_total_sign = 7`), with the largest single contribution in
the 5–8 band (22% of 26B's total). E2B is negative in 6 of 7. The last column is
a control from the paper's own published set — a second published RF variant,
also run literally — which moves by hundredths of a point in every band, so the
band structure is a property of the reader arms, not of the banding. The
distribution of statements across the evidence ladder is itself the story: the
panel's error rate falls monotonically from **52.7%** at one evidence entry to
**5.5%** at 33+, which is where the noisy-OR is strong and there is nothing left
to win.

**One scope note.** The reader panel is built from unique `(statement, evidence)`
pairs (33,361) while this census counts every entry (34,035), as the paper's own
released per-source counts do; **48** statements would change band under the
unique-pair scope.

*Artifact:* `data/results/indra_paper_literal_models_20260724/ap_decomposition_by_paper_band.json`
(`artifact_kind` = `paper_ap_decomposition_by_evidence_count`; the *filename*
is a frozen join key kept across the banding change), generated by
`scripts/compute_paper_ap_decomposition.py`; rendered by
`ApDecompositionByPaperRank.svelte`.

---

# 6 · The belief-model ladder

Every belief-model family INDRA ships plus the paper's literal model and the four
reader gates, all on the same 1,689 statements and the same gold, pooled
tie-aware AP. Row names are the ladder artifact's own `entries[].label`. Each
INDRA value is cross-checked against the value that run's own manifest recorded
(12 of 12 agree, tolerance 1e-12).

| Family | kind | pooled AP | Δ vs unfitted noisy-OR | distinct scores |
| --- | --- | --- | --- | --- |
| `CountsScorer` RF, source counts | INDRA | 0.9027 | −0.0004 | 1,258 |
| Hierarchy propagation | INDRA | 0.9030 | −0.0001 | 824 |
| **noisy-OR `SimpleScorer` (direct)** | **INDRA — baseline** | **0.9031** | — | 690 |
| `BayesianScorer`, source refit | INDRA | 0.9154 | +0.0123 | 1,271 |
| `BayesianScorer`, source+subtype refit | ours | 0.9178 | +0.0147 | 1,399 |
| RF 2k-d13 + Type/#PMIDs/promoter | **the paper's literal model** | 0.9412 | +0.0381 | 1,546 |
| `CountsScorer` RF, full features | ours | 0.9422 | +0.0390 | 1,683 |
| `HybridScorer`, full features | ours | 0.9422 | +0.0390 | 1,683 |
| Gemma 4 E2B gate | reader | 0.9252 | +0.0221 | 420 |
| Gemma 4 31B gate | reader | 0.9489 | +0.0458 | 498 |
| GLM-5 gate | reader | 0.9507 | +0.0475 | 475 |
| Gemma 4 26B gate | reader | 0.9510 | +0.0479 | 492 |

Manifests: `current_indra_counts_hybrid_paper_20260718/current_counts_hybrid_paper_manifest.json`
(both `CountsScorer` rows and `HybridScorer`),
`current_indra_hierarchy_paper_20260717/current_simple_hierarchy_paper_manifest.json`
(hierarchy), `current_indra_bayesian_paper_20260717/current_bayesian_paper_manifest.json`
(noisy-OR direct and both `BayesianScorer` rows).

Every row except `noisy-OR SimpleScorer (direct)` and `Hierarchy propagation` is
fitted and scored **out-of-fold**; those two are unfitted, so there is nothing to
hold out. Two rows are one model: `HybridScorer, full features` is the same
fitted RF served through the hybrid wrapper (absolute gap 0.0), which is why it
repeats 0.9422 — two INDRA scorer classes over one model, not two independent
results. And these are INDRA's classes as they stand today, not a
re-implementation of the paper's 59 published methods.

Two observations, offered as observations rather than verdicts:

1. **A consistency check, not a fidelity check.** `CountsScorer RF, full
   features` on current INDRA lands at **0.9422** AP; the literal 2023 model on
   the released corpus lands at **0.9412** — a gap of **0.00096**. That is not
   the same model twice: ours is a supervised RF over INDRA's whole
   `CountsScorer` feature set, a **superset** of the paper's engineered
   Type/#PMIDs/promoter panel. Two different codebases, two different feature
   sets, two different evidence snapshots — but two APs agreeing in one scalar on
   *different corpora* is agreement in one scalar, not evidence of implementation
   fidelity. The fidelity evidence is Result 2's per-statement r = 0.9994, which
   compares the literal model to its own port.
2. **Where the gains live.** From the same 0.9031 noisy-OR baseline, the
   supervised full-feature RF is worth **+0.0390** and the reading gate is worth
   **+0.0479** (best reader arm, 0.9510) — *but +0.0479 is measured against the
   weakest rung on the ladder*. Against the best noisy-OR variant
   (`BayesianScorer, source+subtype refit`, 0.9178 — ours, not a published paper
   arm) the same gate is **+0.0331**; against the strongest supervised rung
   (`CountsScorer RF, full features`, 0.9422) and against the paper's own
   published `RF 2k-d13 + Type/#PMIDs/promoter` run literally (0.9412) it is
   **+0.0088 to +0.0098**. Quote the range, never the +0.0479 alone. Meanwhile
   `Hierarchy propagation` (−0.0001) and `CountsScorer RF, source counts`
   (−0.0004) are flat against the unfitted `SimpleScorer` on this panel — the
   supervised lift comes from the non-count features, not from learning a better
   source-count function.

*Artifact:* `belief_model_ladder.json`; rendered by `BeliefModelLadder.svelte`.

---

# 7 · Why there is anything to win — one prior for five readers

This is the finding we think is most useful to the belief model itself, and it
does not depend on any LLM.

`SimpleScorer` is a function of the evidence-count profile alone. Two statements
with the same (source, count) tuple get the same belief no matter what the
sentences say. On this panel that degeneracy is concentrated exactly where the
decisions are:

- **86** distinct belief values at 4-decimal resolution (690 raw, but **656 of
  those raw values sit above 0.99** and cover 866 statements — the asymptote).
  **Below 0.99 the formula has only 34 distinct raw answers for 823 statements.**
- The single most common answer, **0.65**, covers **328 statements (19.4%)** —
  and those statements are correct **47.3%** of the time.

## The decisive measurement: one belief value, five accuracies

0.65 is not an arbitrary modal value. It is what the formula returns for *one*
sentence from *any* of the five readers, because INDRA's default priors hand
reach, sparser, trips, medscan and rlimsp the **identical `(rand 0.30, syst
0.05)`** pair. So the single-evidence rung is a controlled experiment the panel
runs for us: same assigned confidence, same evidence count, only the reader
differs.

| Source, single-evidence statements (belief = 0.65 for all) | statements | correct | measured accuracy |
| --- | --- | --- | --- |
| trips | 54 | 35 | **64.8%** |
| rlimsp | 42 | 24 | 57.1% |
| medscan | 73 | 40 | 54.8% |
| reach | 104 | 50 | 48.1% |
| sparser | 42 | 5 | **11.9%** |

χ² = **30.7** on 4 df, **p = 3.6e-6**, n = **315**. A five-fold spread in
measured accuracy at a single assigned confidence. (`isi`, too small for the
test at 13 single-evidence statements, sits at 7.7% on the same 0.65.)

The mechanism is a **shared prior**, not a broken independence assumption. The
constructive change is therefore not a within-source saturation term but
**per-source priors**: these five readers do not deserve the same
`(rand, syst)` on this panel, and separating them is a change inside the formula
that needs no LLM in the loop. Refitting them is worth **+0.0123** AP over
`SimpleScorer` on this panel (`BayesianScorer, source refit`, 0.9154 vs 0.9031,
§6), and the 2023 paper's own `Belief Orig` arms already refit per-source
`pr`/`ps` per fold by MCMC — the fit is available, it is just not what INDRA's
default prior table ships.

## The ladders: belief climbs, accuracy does not

The single-source ladders show the same money from the other side. Every source
climbs the *identical* claim curve 0.65 → ~0.95 as evidence accumulates. Measured
accuracy does not follow it up:

| Source (all single-source statements) | statements | belief range | accuracy at n=1 | accuracy at the top rung with ≥5 statements | Spearman(evidence count, correct) |
| --- | --- | --- | --- | --- | --- |
| reach | 219 | 0.65 → 0.9500 | 48.1% (104) | 30.8% (n=8, 13 stmts) | +0.022 (p = 0.75) |
| medscan | 120 | 0.65 → 0.9500 | 54.8% (73) | 75.0% (n=5, 8 stmts) | +0.110 (p = 0.23) |
| sparser | 113 | 0.65 → 0.9500 | 11.9% (42) | 20.0% (n=9, 5 stmts) | +0.165 (p = 0.08) |
| trips | 60 | 0.65 → 0.9498 | 64.8% (54) | — (only 6 stmts above n=1) | −0.104 (p = 0.43) |
| rlimsp | 54 | 0.65 → 0.9500 | 57.1% (42) | 80.0% (n=2, 5 stmts) | +0.233 (p = 0.09) |

**Belief rises by 30 points across every ladder and measured accuracy stays
flat.** That is the over-crediting argument, and it is well powered: it is a
statement about all **566** statements in these five single-source ladders, not
about any one rung. One presentation difference to note if you cross-check: the
`/paper` panel draws only **four** of these ladders (506 statements). Its loader
requires a source to have at least two rungs of ≥5 statements before it draws a
line, and `trips` has only one (54 of its 60 statements sit at n=1), so the panel
omits it. The table above is the more complete view; `trips` is also the source
whose rank correlation is the one negative of the five, so including it here is
the conservative choice.

What it is **not** is an inversion. An earlier draft of this memo, and the
`/paper` panel that shares its loader, read the reach ladder as "accuracy falls
as belief rises". That does not survive checking and we have removed it. The
rank correlation between evidence count and correctness is positive for four of
the five readers and significant for none of them (table above). The apparent
fall was an endpoint chosen by an undisclosed ≥5-statements-per-rung filter in
`viewer/src/lib/server/belief-heuristic.ts`, which skips reach at n=10 (75.0%,
4 statements) and sparser at n=10 (66.7%, 3 statements). The reach endpoint it
lands on is 4/13 statements, exact 95% CI **[9.1%, 61.4%]** — an interval that
comfortably contains the n=1 value it was being contrasted with. That filter is
no longer undisclosed: the `/paper` panel states the ≥5-statements-per-rung
drawing rule in its own legend and prints, beside every ladder, that source's
rank correlation over **all** its statements — undrawn rungs included — so the
page and this memo tell the same flat-not-inverted story from the same numbers.

*Artifacts:* `data/results/current_indra_simple_paper_20260717/current_indra_simple_default_prediction_provenance.jsonl`
joined to `data/results/indra_paper_statement_gold_20260717/paper_statement_gold.jsonl`;
the same computation is served by `viewer/src/lib/server/belief-heuristic.ts`
and rendered by `BeliefHeuristicResponse.svelte`.

---

# 8 · Limits

## 8a · Calibration — ranking better is not being right about the odds

None of the reader arms is calibrated, and we should not let an AUROC or AP
number imply otherwise. Logistic recalibration `label ~ 1 + logit(score)` (MLE,
clip 1e-6) and 10-bin expected calibration error, on the same 1,689 statements
(base rate 0.7324):

| Arm | calibration slope | intercept | ECE (10 bins) | mean score |
| --- | --- | --- | --- | --- |
| paper RF 2k-d13 + Type/#PMIDs/promoter | 1.297 | −0.169 | 0.028 | 0.728 |
| Gemma 4 31B gate | 0.201 | +1.635 | 0.107 | 0.660 |
| Gemma 4 26B gate | 0.207 | +1.612 | 0.102 | 0.665 |
| GLM-5 gate | 0.213 | +1.970 | 0.123 | 0.628 |
| Gemma 4 E2B gate | 0.153 | +1.227 | 0.122 | 0.696 |
| INDRA `SimpleScorer`, unfitted noisy-OR | 0.329 | −0.397 | **0.179** | 0.911 |

Ideal is slope 1, intercept 0. The paper's fitted RF is close (slope 1.30, ECE
0.028) — the only well-calibrated arm on the panel. The reader gates sit near
slope **0.2**: their probabilities are far too extreme in both directions and
should be read as a ranking, not as odds.

The row worth not skipping is the last one. **The unfitted noisy-OR INDRA serves
is the *most* miscalibrated arm here** — ECE 0.179, mean assigned belief 0.911
against a 0.732 base rate. Its slope is 0.33, closer to the reader gates than to
the paper's RF. Reader-arm miscalibration is a real limit on our arms; it is not
a way in which they are worse than the incumbent.

*Rendered by* `PaperReliabilityStrip.svelte`.

## 8b · What the current gate design costs

Because the reader can only remove evidence (Result 3b), the ceiling is fixed by
the formula:

- **249 true statements are scored below 0.90 by the unfitted `SimpleScorer` and
  can never be promoted** by any reader arm, however well it reads.
- **108 true statements are zeroed by Gemma 26B** into the unrankable zero
  block — losses, and part of why the reader arms' AUROC gain is larger than
  their AP gain. But read that 108 with the table below before reading it as 108
  losses *on top of* the 249.

Those two costs **overlap rather than add**. A true statement carried by a single
weak evidence is both already under the bar *and* zeroed the moment the reader
rejects that evidence, so it is counted in both bullets above. The artifact
derives and self-checks the split, so we quote it rather than doing the
arithmetic in prose — `statement_review_queue.json` →
`promotion_ceiling.per_arm_overlap`, against the panel's **1,237** true
statements:

| Arm | zeroed (true) | of those, already below 0.90 | newly lost | distinct true statements affected |
| --- | --- | --- | --- | --- |
| Gemma 4 26B gate | 108 | 65 | 43 | **292 (23.6%)** |
| Gemma 4 E2B gate | 127 | 75 | 52 | 301 (24.3%) |
| Gemma 4 31B gate | 120 | 62 | 58 | 307 (24.8%) |
| GLM-5 gate | 151 | 82 | 69 | 318 (25.7%) |

So for the 26B arm the reader newly loses **43** true statements, not 108, and
the union of the formula's ceiling and the reader's zero block is **292** true
statements — 23.6% of the true panel. That union, not either column alone, is
the number to carry. The 0.90 threshold is a stated illustrative promotion bar,
not a fitted one, and the count is monotone in it.

We should own this: subtractive-only gating buys triage precision at the cost of
recall on under-scored true statements. A reader that could also *supply* a
positive log-likelihood ratio — which is what the configuration-scoped hybrid
scalar in `src/indra_belief/statement_belief.py` does when a fitted reader
profile is supplied — is not what was run here. These arms are deliberately the
unfitted hard-gate fallback, so that the aggregation stays INDRA's shipped
default-prior noisy-OR and the only thing that varies is which evidence survives.

## 8c · Caveats, stated once and plainly

1. All arms outside the literal reproduction are scored on **current INDRA
   evidence** (indra 1.24.0). That is less of a gap than it sounds: the
   per-source evidence-count multisets are **identical** between the paper's
   frozen `historical_all_source_counts` and current INDRA for **all 1,689
   statements** — 0 mismatches, 34,035 entries each way — which is what makes
   the head-to-head like-for-like. What current INDRA can differ in is the
   *content* of an evidence row (text, grounding), not how many each source
   contributed.
2. Every threshold in §4 is chosen on **this same panel**. Nothing is held out;
   no operating point is validated out of sample. The equal-yield budgets are
   explicitly **oracle** thresholds, and that favours the comparator.
3. Bootstrap CIs in §5c are **uncorrected for 7 comparisons** against one
   reference; §5d gives the corrected version, under which no positive reader AP
   margin excludes zero. The literal `RF 2k-d13 + Type/#PMIDs/promoter` is also
   scored **out-of-fold on this panel** while the reader arms were never fitted
   on it — an asymmetry in the paper model's favour.
4. Reader-arm operating points are **coarse** — a few reachable thresholds, one
   large tie block at the bottom.
5. Reader arms are **uncalibrated** (§8a); AP/AUROC are ranking statements, not
   probability statements.
6. The 1,689-statement panel is the paper's own curated set, with its own
   sampling. It is not a corpus-representative sample of INDRA. The four panels
   in §3 differ in curator, corpus, sampling and reader deployment.
7. The reader arms are **not zero-shot** (14 hand-authored demonstration pairs
   per call), and three of the belief-model arms drawn here are **ours**, not the
   paper's (§ naming note).

---

## Bottom line

Run from its own pinned code on its own released corpus, the 2023 model
reproduces Table 6 to within 0.0016 and is a near-perfect twin of the semantic
port used across this program (r = 0.9994) — the reconstruction is validated.

The reader arms are not a competing belief model: they are INDRA's own unfitted
default-prior noisy-OR on a filtered evidence set. Against **what INDRA actually
serves**, that filter is worth **+0.074 AUROC** on the paper's panel against the
strongest served form (and +0.1187 against the unfitted `SimpleScorer`) and the
direction replicates on four panels. Operationally it queues **462** statements
to catch **354 of 452** errors where the paper's own fitted RF needs **662**
oracle-tuned reviews for the same yield — and that detection margin is the one
result here that both survives max-t correction and **strengthens** on the
adjudication-safe subpanel.

Against the paper's **fitted research RF** on ranking metrics the margin is
**+0.0098 AP**, it does not survive multiplicity correction, it falls 60% on the
adjudication-safe subpanel, and conditional on the statements the gate does not
zero it ranks **worse** than the paper's model. The AUROC gap is not a ranking
result: a single bit — "the reader kept ≥ 1 evidence" — already scores AUROC
0.8479. This is a **detector**, not a ranker, and the paper published ranking
metrics, so the metrics we inherited measure what it does worst.

The durable finding needs no LLM. At the single-evidence rung the formula assigns
**exactly 0.65** to a sentence from any of five readers whose measured accuracy
on this panel runs from **11.9% to 64.8%** (χ² = 30.7, p = 3.6e-6, n = 315). The
defect is one shared prior for five very different readers, and the fix —
per-source priors, worth +0.0123 AP here — is inside the formula.
