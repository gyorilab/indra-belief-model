# Z-phase error profile — eval_set_v4 (n=98)

Comparison: **X-phase** (`73273b0`) vs **Y-phase** (Y1+Y2) vs **Z-phase** (Z1 binding-criterion few-shots + Z2 LOF augment).

## Dim 1 — Binary accuracy / P / R / F1 (3-way)

| metric | X | Y | Z | Δ(Z-Y) |
|---|---|---|---|---|
| accuracy | 0.643 | 0.684 | 0.755 | +0.071 |
| precision | 0.617 | 0.632 | 0.719 | +0.087 |
| recall | 0.755 | 0.878 | 0.837 | -0.041 |
| **F1** | **0.679** | **0.735** | **0.774** | **+0.039** |
| TP | 37 | 43 | 41 | -2 |
| FP | 23 | 25 | 16 | -9 |
| FN | 12 | 6 | 8 | +2 |
| TN | 26 | 24 | 33 | +9 |


## Class-A FP resolution (G_Z2 gate: ≥10 of 17 flipped to TN)

- Y-phase Class-A FPs: **7** records
- Of those, Z-phase flipped to TN (verdict=incorrect): **2** records
- Class-A FP resolution rate: **2/7 = 29%**

Per-record outcomes (top 20):

| source_hash | gold_tag | stmt_type | Y verdict | Z verdict | outcome |
|---|---|---|---|---|---|
| -82304368972 | entity_boundaries | Activation | correct | correct | — |
| 146897806955 | grounding | Activation | correct | incorrect | flipped |
| 235216649717 | grounding | Activation | correct | correct | — |
| 263687595086 | no_relation | Activation | correct | incorrect | flipped |
| 320735560855 | act_vs_amt | Inhibition | correct | correct | — |
| 822495377029 | grounding | Activation | correct | correct | — |
| 852029425782 | polarity | DecreaseAmount | correct | correct | — |


## Class-B FP resolution (Y2 LOF over-flip)

- Y-phase Class-B FPs (direct_sign_mismatch + LOF flip path): **4** records
- Z flipped to TN: **0** records


## G_Z3: Y-rescued recall preservation (threshold ≥ 0.85)

- Y-rescued TPs (X-FN → Y-TP, gold=correct): **6** records
- Z preserved: **5** records (83%)
- Gate G_Z3: FAIL


## Y→Z regressions

- Y-correct cases now Z-incorrect: **5**

| source_hash | gold_tag | stmt_type | Y verdict | Z verdict |
|---|---|---|---|---|
| -68292170109 | correct | Complex | correct | incorrect |
| -68172706912 | correct | Complex | correct | incorrect |
| -42168179535 | correct | Activation | correct | incorrect |
| -42107737579 | wrong_relation | Complex | incorrect | correct |
| 251628420438 | correct | Phosphorylation | correct | incorrect |


## Dim 2 — Per-gold-tag correct-call rate (3-way)

| gold_tag | n | X | Y | Z |
|---|---|---|---|---|
| correct | 49 | 37/49 = 76% | 43/49 = 88% | 41/49 = 84% |
| grounding | 18 | 10/18 = 56% | 9/18 = 50% | 12/18 = 67% |
| no_relation | 8 | 5/8 = 62% | 5/8 = 62% | 8/8 = 100% |
| wrong_relation | 7 | 5/7 = 71% | 5/7 = 71% | 5/7 = 71% |
| act_vs_amt | 6 | 1/6 = 17% | 0/6 = 0% | 3/6 = 50% |
| entity_boundaries | 5 | 3/5 = 60% | 3/5 = 60% | 3/5 = 60% |
| negative_result | 2 | 0/2 = 0% | 0/2 = 0% | 0/2 = 0% |
| polarity | 2 | 1/2 = 50% | 1/2 = 50% | 1/2 = 50% |
| other | 1 | 1/1 = 100% | 1/1 = 100% | 1/1 = 100% |


## Dim 5 — Expected Calibration Error

- X = 0.231
- Y = 0.206
- Z = 0.145


## Gate verdict

- **G_Z1** F1 ≥ 0.78: F1=0.774 → FAIL
- **G_Z2** Class-A FP resolution ≥ 10 of 17: 2/7 → FAIL
- **G_Z3** Y-rescued recall ≥ 0.85: 83% → FAIL
- **G_Z4** pytest: validated separately (289 passing)

### Overall: **PIVOT**

Falsifier triggered. Pivot options:
1. Probe redesign: subject-object extraction-then-bind-check
2. Adversarial calibration on Class-A residuals

