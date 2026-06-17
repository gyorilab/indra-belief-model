# Belief head-to-head — gemma-26b vs text-miner baseline

Gold `data/benchmark/holdout_cc.jsonl` · run `data/results/holdout_cc_gemma.jsonl`  
Coverage: 500/500 run rows joined to gold → **393 statements** (320 single, 73 multi; 183 gold-correct).

## Belief discrimination (statement grain, positive = correct)

| belief | subset | n | AUROC | ECE |
|---|---|--:|--:|--:|
| LLM gated belief (gemma-26b) | all | 393 | 0.750 | 0.160 |
| LLM gated belief (gemma-26b) | single_evidence | 320 | 0.749 | 0.183 |
| LLM gated belief (gemma-26b) | multi_evidence | 73 | 0.824 | 0.128 |
| LLM gated + calibrated soft (gemma-26b) | all | 393 | 0.759 | 0.133 |
| LLM gated + calibrated soft (gemma-26b) | single_evidence | 320 | 0.755 | 0.131 |
| LLM gated + calibrated soft (gemma-26b) | multi_evidence | 73 | 0.815 | 0.150 |
| text-miner belief · recalibrated priors | all | 393 | 0.624 | 0.417 |
| text-miner belief · recalibrated priors | single_evidence | 320 | 0.619 | 0.393 |
| text-miner belief · recalibrated priors | multi_evidence | 73 | 0.674 | 0.520 |
| text-miner belief · INDRA priors | all | 393 | 0.613 | 0.448 |
| text-miner belief · INDRA priors | single_evidence | 320 | 0.609 | 0.424 |
| text-miner belief · INDRA priors | multi_evidence | 73 | 0.657 | 0.551 |
| INDRA stored belief (w/ propagation) | all | 393 | 0.622 | 0.420 |
| INDRA stored belief (w/ propagation) | single_evidence | 320 | 0.625 | 0.391 |
| INDRA stored belief (w/ propagation) | multi_evidence | 73 | 0.654 | 0.549 |

## Statement-grain error detection (positive = gold-incorrect)

Flag rule: `verdict_statement != correct`. P=0.783 R=0.686 F1=0.731 (tp=144 fp=40 fn=66 tn=143).  
Deterministic hard-flag (`verdict_statement == incorrect`) precision: 0.750.
verdict_statement counts: {'correct': 209, 'review': 176, 'incorrect': 8}.
