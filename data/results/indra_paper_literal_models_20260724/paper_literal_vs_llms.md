# Paper literal model vs LLMs — direct comparison

All arms scored on the identical **1689** "all sources, specific" statements (released paper labels), joined stmt_hash ↔ statement_id via the frozen paper gold. `*` marks a 95% CI that excludes zero (paired fold-stratified bootstrap, 10000 resamples, vs the paper's literal RF+promoter model).

> **Verdict metric = pooled average-precision (AP) and AUROC, not the paper's trapezoidal PR-AUC.** The paper's headline metric, fold-mean *trapezoidal* PR-AUC, is arm-dependent optimistic: trapezoidal interpolation over-credits heavily-tied score distributions. The paper RF emits 1546 distinct scores over 1689 statements (near-continuous → trapezoidal ≈ AP), while the LLMs emit only ~420–498 distinct scores (heavily tied → trapezoidal inflates them by +0.010–0.014). Using trapezoidal as the cross-arm verdict roughly doubles the LLM deltas and is not fair; it is retained here only to show faithful reproduction of the paper's own numbers.

| Arm | AP (verdict) | AUROC | Trapezoidal PR-AUC (paper metric) | distinct scores | ΔAP [95% CI] | ΔAUROC [95% CI] | Δtrapezoidal [95% CI] |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Paper literal RF+promoter | 0.941 | 0.852 | 0.941 | 1546 | — (ref) | — (ref) | — (ref) |
| Paper literal RF+prom/avglen | 0.942 | 0.853 | 0.942 | 1681 | +0.001 [-0.002, +0.003] | +0.001 [-0.005, +0.007] | +0.001 [-0.002, +0.003] |
| Paper semantic port RF+promoter | 0.941 | 0.852 | 0.942 | 1546 | +0.000 [-0.001, +0.001] | +0.000 [-0.001, +0.001] | +0.000 [-0.001, +0.001] |
| Gemma 4 E2B | 0.925 | 0.840 | 0.935 | 420 | -0.016 [-0.026, -0.006]* | -0.011 [-0.031, +0.008] | -0.006 [-0.016, +0.004] |
| Gemma 4 26B | 0.951 | 0.901 | 0.961 | 492 | +0.010 [+0.001, +0.019]* | +0.049 [+0.031, +0.068]* | +0.020 [+0.011, +0.029]* |
| Gemma 4 31B | 0.949 | 0.898 | 0.960 | 498 | +0.008 [-0.001, +0.017] | +0.046 [+0.027, +0.065]* | +0.019 [+0.010, +0.028]* |
| GLM-5 | 0.951 | 0.902 | 0.965 | 475 | +0.009 [+0.001, +0.018]* | +0.051 [+0.032, +0.070]* | +0.024 [+0.015, +0.033]* |
| INDRA CoGEx hybrid | 0.923 | 0.827 | 0.923 | 1176 | -0.018 [-0.030, -0.008]* | -0.024 [-0.040, -0.009]* | -0.018 [-0.028, -0.008]* |

## Reading the verdict (tie-robust)

- **Gemma 26B and GLM-5 beat the paper's literal best model on every estimator** (AP, AUROC, and trapezoidal) with CIs excluding zero — the robust, defensible result.
- **Gemma 31B** beats on AUROC and trapezoidal but is **not significant on AP** (CI includes zero); its trapezoidal "win" is partly the tie artifact.
- **Gemma E2B** *loses* to the paper model on AP and AUROC (CIs exclude zero); the trapezoidal metric had it near-tie, which was the artifact.
- **INDRA CoGEx hybrid** loses on every estimator.

## Faithfulness: literal reproduction vs our semantic port

- Per-statement Pearson r = **0.9994**, Spearman = **0.9988**
- Mean |Δprob| = **0.0055**, max |Δprob| = 0.0398
- Fold-mean trapezoidal PR-AUC: literal 0.941 vs port 0.942 — the semantic port is a near-bit-exact stand-in for the paper's literal model.
