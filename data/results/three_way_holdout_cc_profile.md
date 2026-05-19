# 3-way comparison on holdout_cc (n=453)

**Holdout:** 500-record stratified, no eval_set_v4 contamination, hedged 0.40% + reader-error 1.60% (CC0 gates PASS, ceiling 98%).

**Arms:**
- **AA (decomposed)**: four-probe (subject_role, object_role, relation_axis, scope) + adjudicator decision table. Production at commit `f5ab653`.
- **CC (decomposed + extract-bind)**: relation_axis replaced with LLM extraction + deterministic bind-check.
- **Monolithic**: single deterministic LLM call per (Stmt, Ev) (temp=0.1) with type-adaptive contrastive few-shots.

## Binary metrics

| metric | AA | CC | Monolithic |
|---|---|---|---|
| **F1** | **0.657** | **0.607** | **0.751** |
| accuracy | 0.634 | 0.583 | 0.744 |
| precision | 0.598 | 0.555 | 0.706 |
| recall | 0.729 | 0.670 | 0.803 |
| TP | 159 | 146 | 175 |
| FP | 107 | 117 | 73 |
| FN | 59 | 72 | 43 |
| TN | 128 | 118 | 162 |

## Pairwise McNemar's exact tests

| pair | both right | both wrong | a-only-right | b-only-right | p-value | preferred | alpha=0.05 |
|---|---|---|---|---|---|---|---|
| AA vs CC | 227 | 129 | 60 (AA) | 37 (CC) | 0.0250 | AA | SIG |
| AA vs Mono | 253 | 82 | 34 (AA) | 84 (Mono) | 0.0000 | Mono | SIG |
| CC vs Mono | 233 | 85 | 31 (CC) | 104 (Mono) | 0.0000 | Mono | SIG |

## Per-gold-tag correct-call rate

| gold_tag | n | AA | CC | Monolithic |
|---|---|---|---|---|
| correct | 218 | 159/218 = 73% | 146/218 = 67% | 175/218 = 80% |
| grounding | 45 | 21/45 = 47% | 24/45 = 53% | 25/45 = 56% |
| no_relation | 40 | 29/40 = 72% | 20/40 = 50% | 32/40 = 80% |
| wrong_relation | 31 | 24/31 = 77% | 24/31 = 77% | 25/31 = 81% |
| entity_boundaries | 25 | 14/25 = 56% | 13/25 = 52% | 15/25 = 60% |
| act_vs_amt | 24 | 8/24 = 33% | 5/24 = 21% | 16/24 = 67% |
| polarity | 21 | 10/21 = 48% | 13/21 = 62% | 17/21 = 81% |
| other | 21 | 14/21 = 67% | 10/21 = 48% | 13/21 = 62% |
| negative_result | 10 | 5/10 = 50% | 6/10 = 60% | 8/10 = 80% |
| hypothesis | 10 | 1/10 = 10% | 1/10 = 10% | 8/10 = 80% |
| mod_site | 4 | 0/4 = 0% | 0/4 = 0% | 2/4 = 50% |
| agent_conditions | 4 | 2/4 = 50% | 2/4 = 50% | 1/4 = 25% |

## Per-stmt_type accuracy

| stmt_type | n | AA | CC | Monolithic |
|---|---|---|---|---|
| Complex | 73 | 49/73 = 67% | 43/73 = 59% | 57/73 = 78% |
| Activation | 68 | 43/68 = 63% | 32/68 = 47% | 53/68 = 78% |
| Phosphorylation | 64 | 36/64 = 56% | 35/64 = 55% | 41/64 = 64% |
| Inhibition | 52 | 28/52 = 54% | 27/52 = 52% | 32/52 = 62% |
| IncreaseAmount | 39 | 29/39 = 74% | 27/39 = 69% | 27/39 = 69% |
| DecreaseAmount | 39 | 25/39 = 64% | 26/39 = 67% | 29/39 = 74% |
| Autophosphorylation | 22 | 12/22 = 55% | 15/22 = 68% | 19/22 = 86% |
| Dephosphorylation | 20 | 9/20 = 45% | 13/20 = 65% | 11/20 = 55% |
| Ubiquitination | 19 | 16/19 = 84% | 13/19 = 68% | 16/19 = 84% |
| Translocation | 17 | 12/17 = 71% | 7/17 = 41% | 16/17 = 94% |
| Acetylation | 16 | 10/16 = 62% | 9/16 = 56% | 15/16 = 94% |
| Deacetylation | 9 | 6/9 = 67% | 4/9 = 44% | 8/9 = 89% |
| ActiveForm | 5 | 4/5 = 80% | 5/5 = 100% | 4/5 = 80% |
| Methylation | 5 | 5/5 = 100% | 5/5 = 100% | 5/5 = 100% |
| Deubiquitination | 3 | 2/3 = 67% | 3/3 = 100% | 2/3 = 67% |
| Glycosylation | 1 | 1/1 = 100% | 0/1 = 0% | 1/1 = 100% |
| Sumoylation | 1 | 0/1 = 0% | 0/1 = 0% | 1/1 = 100% |

## Calibration (ECE)

- AA:         0.189
- CC:         0.215
- Monolithic: 0.206

## Headline

Best F1: **Mono = 0.751**

