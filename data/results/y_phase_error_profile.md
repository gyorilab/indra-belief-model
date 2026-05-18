# Y-phase error profile — eval_set_v4 (n=100)

Comparison: **X-phase** (commit `73273b0`) vs **Y-phase** (Y1 LLM-confirm-negation + Y2 perturbation propagation).

## Dim 1 — Binary accuracy / P / R / F1

| metric | X-phase | Y-phase | Δ |
|---|---|---|---|
| accuracy | 0.630 | 0.680 | +0.050 |
| precision | 0.607 | 0.629 | +0.022 |
| recall | 0.740 | 0.880 | +0.140 |
| **F1** | **0.667** | **0.733** | **+0.067** |
| TP | 37 | 44 | +7 |
| FP | 24 | 26 | +2 |
| FN | 13 | 6 | -7 |
| TN | 26 | 24 | -2 |


## Dim 2 — Per-gold-tag breakdown

| gold_tag | n | X correct-call rate | Y correct-call rate | Δ |
|---|---|---|---|---|
| correct | 49 | 37/49 = 76% | 43/49 = 88% | +6 |
| grounding | 18 | 10/18 = 56% | 9/18 = 50% | -1 |
| no_relation | 8 | 5/8 = 62% | 5/8 = 62% | +0 |
| wrong_relation | 7 | 5/7 = 71% | 5/7 = 71% | +0 |
| act_vs_amt | 6 | 1/6 = 17% | 0/6 = 0% | -1 |
| entity_boundaries | 5 | 3/5 = 60% | 3/5 = 60% | +0 |
| polarity | 2 | 1/2 = 50% | 1/2 = 50% | +0 |
| negative_result | 2 | 0/2 = 0% | 0/2 = 0% | +0 |
| other | 1 | 1/1 = 100% | 1/1 = 100% | +0 |


## Dim 3 — Per-stmt_type recall (we caught gold=correct)

| stmt_type | n_total | n_gold=correct | X recall | Y recall | Δ |
|---|---|---|---|---|---|
| Complex | 37 | 18 | 16/18 = 89% | 17/18 = 94% | +1 |
| Activation | 25 | 7 | 4/7 = 57% | 5/7 = 71% | +1 |
| Phosphorylation | 17 | 15 | 11/15 = 73% | 13/15 = 87% | +2 |
| IncreaseAmount | 6 | 5 | 4/5 = 80% | 4/5 = 80% | +0 |
| Inhibition | 6 | 2 | 0/2 = 0% | 2/2 = 100% | +2 |
| DecreaseAmount | 3 | 0 | — | — | — |
| Sumoylation | 1 | 1 | 1/1 = 100% | 1/1 = 100% | +0 |
| Translocation | 1 | 1 | 1/1 = 100% | 1/1 = 100% | +0 |
| Dephosphorylation | 1 | 0 | — | — | — |
| Acetylation | 1 | 0 | — | — | — |


## Dim 4 — Per-source_api accuracy

| source_api | n | X accuracy | Y accuracy | Δ |
|---|---|---|---|---|
| reach | 39 | 22/39 = 56% | 23/39 = 59% | +1 |
| sparser | 29 | 24/29 = 83% | 25/29 = 86% | +1 |
| trips | 12 | 7/12 = 58% | 8/12 = 67% | +1 |
| medscan | 10 | 4/10 = 40% | 4/10 = 40% | +0 |
| rlimsp | 7 | 6/7 = 86% | 7/7 = 100% | +1 |
| isi | 1 | 0/1 = 0% | 0/1 = 0% | +0 |


## Dim 5 — Score calibration (reliability bins)

| score bin | X n | X gold=correct rate | Y n | Y gold=correct rate |
|---|---|---|---|---|
| <0.05 | 0 | — | 0 | — |
| 0.05-0.20 | 13 | 69% | 6 | 50% |
| 0.20-0.35 | 23 | 17% | 21 | 14% |
| 0.35-0.50 | 3 | 0% | 3 | 0% |
| 0.50-0.65 | 8 | 38% | 9 | 44% |
| 0.65-0.80 | 13 | 38% | 10 | 40% |
| 0.80-0.95 | 40 | 72% | 51 | 71% |
| 0.95-1.00 | 0 | — | 0 | — |

**Expected Calibration Error**: X = 0.241, Y = 0.206


## Dim 6 — Error-mode taxonomy (probe attribution for misclassifications)

| error mode | X count | Y count | Δ |
|---|---|---|---|
| axis_mismatch | 3 | 2 | -1 |
| other | 20 | 22 | 2 |
| scope=negated (LLM) | 5 | 3 | -2 |
| scope=negated (substrate) | 3 | 0 | -3 |
| sign_mismatch | 4 | 4 | 0 |


## Dim 8 — Y1 impact: substrate vs LLM scope=negated

| source | X scope=negated count | Y scope=negated count | Δ |
|---|---|---|---|
| substrate | 4 | 0 | -4 |
| llm | 7 | 4 | -3 |


Expected Y1 effect: substrate count → 0; LLM count rises
by however many cases the substrate previously short-circuited.


## Dim 10 — Confusion-direction asymmetry (vs gold)

X: FP=24 vs FN=13 — ratio 1.85x; too lenient

Y: FP=26 vs FN=6 — ratio 4.33x; too lenient

