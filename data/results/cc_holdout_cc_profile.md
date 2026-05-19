# CC vs AA on holdout_cc (n=453)

**Holdout:** 500-record stratified, no eval_set_v4 contamination, hedged 0.40% + reader-error 1.60% (CC0 gates PASS, ceiling 98%).

**Phases compared:**
- **AA** (production, commit `f5ab653`): closed-set relation_axis with BINDING CRITERION + COUNTER-EXAMPLES.
- **CC** (this implementation): extract-then-bind-check — LLM extracts structured relation tuple, deterministic Python bind-check verdicts via alias resolution + axis taxonomy + sign reconciliation + binding-axis symmetry.

## Dim 1 — Binary metrics

| metric | AA | CC | Δ |
|---|---|---|---|
| **F1** | **0.657** | **0.607** | **-0.050** |
| accuracy | 0.634 | 0.583 | -0.051 |
| precision | 0.598 | 0.555 | -0.043 |
| recall | 0.729 | 0.670 | -0.060 |
| TP | 159 | 146 | -13 |
| FP | 107 | 117 | +10 |
| FN | 59 | 72 | +13 |
| TN | 128 | 118 | -10 |

## Dim 2 — McNemar's exact test

- Both correct (concordant):       227
- Both wrong (concordant):         129
- AA correct, CC wrong (b):        **60**
- AA wrong, CC correct (c):        **37**
- Discordant pairs (b+c):          97
- McNemar two-sided p-value:       **0.0250**
- Difference direction:            AA preferred
- Verdict:                         **SIGNIFICANT**

## Dim 3 — Per-gold-tag correct-call rate

| gold_tag | n | AA | CC |
|---|---|---|---|
| correct | 218 | 159/218 = 73% | 146/218 = 67% |
| grounding | 45 | 21/45 = 47% | 24/45 = 53% |
| no_relation | 40 | 29/40 = 72% | 20/40 = 50% |
| wrong_relation | 31 | 24/31 = 77% | 24/31 = 77% |
| entity_boundaries | 25 | 14/25 = 56% | 13/25 = 52% |
| act_vs_amt | 24 | 8/24 = 33% | 5/24 = 21% |
| polarity | 21 | 10/21 = 48% | 13/21 = 62% |
| other | 21 | 14/21 = 67% | 10/21 = 48% |
| negative_result | 10 | 5/10 = 50% | 6/10 = 60% |
| hypothesis | 10 | 1/10 = 10% | 1/10 = 10% |
| mod_site | 4 | 0/4 = 0% | 0/4 = 0% |
| agent_conditions | 4 | 2/4 = 50% | 2/4 = 50% |

## Dim 4 — Per-stmt_type accuracy

| stmt_type | n | AA | CC |
|---|---|---|---|
| Complex | 73 | 49/73 = 67% | 43/73 = 59% |
| Activation | 68 | 43/68 = 63% | 32/68 = 47% |
| Phosphorylation | 64 | 36/64 = 56% | 35/64 = 55% |
| Inhibition | 52 | 28/52 = 54% | 27/52 = 52% |
| IncreaseAmount | 39 | 29/39 = 74% | 27/39 = 69% |
| DecreaseAmount | 39 | 25/39 = 64% | 26/39 = 67% |
| Autophosphorylation | 22 | 12/22 = 55% | 15/22 = 68% |
| Dephosphorylation | 20 | 9/20 = 45% | 13/20 = 65% |
| Ubiquitination | 19 | 16/19 = 84% | 13/19 = 68% |
| Translocation | 17 | 12/17 = 71% | 7/17 = 41% |
| Acetylation | 16 | 10/16 = 62% | 9/16 = 56% |
| Deacetylation | 9 | 6/9 = 67% | 4/9 = 44% |
| ActiveForm | 5 | 4/5 = 80% | 5/5 = 100% |
| Methylation | 5 | 5/5 = 100% | 5/5 = 100% |
| Deubiquitination | 3 | 2/3 = 67% | 3/3 = 100% |
| Glycosylation | 1 | 1/1 = 100% | 0/1 = 0% |
| Sumoylation | 1 | 0/1 = 0% | 0/1 = 0% |

## Dim 5 — Per-source_api accuracy

| source_api | n | AA | CC |
|---|---|---|---|
| reach | 200 | 122/200 = 61% | 104/200 = 52% |
| sparser | 75 | 53/75 = 71% | 51/75 = 68% |
| trips | 72 | 47/72 = 65% | 45/72 = 62% |
| medscan | 58 | 33/58 = 57% | 33/58 = 57% |
| rlimsp | 44 | 28/44 = 64% | 27/44 = 61% |
| signor | 4 | 4/4 = 100% | 4/4 = 100% |

## Dim 6 — Calibration (ECE)

- AA ECE: **0.189**
- CC ECE: **0.215**

## Decision

**Do not ship CC.** Significant REGRESSION vs AA at α=0.05 (p=0.0250); architectural redesign under-performs on holdout_cc.
