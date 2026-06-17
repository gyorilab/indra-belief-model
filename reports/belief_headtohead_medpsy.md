# Belief head-to-head — medpsy-4b vs text-miner baseline

Gold `data/benchmark/eval_curation_v1.jsonl` · run `data/results/eval_curation_v1_medpsy.jsonl`  
Coverage: 1606/1606 run rows joined to gold → **913 statements** (570 single, 343 multi; 488 gold-correct).

## Belief discrimination (statement grain, positive = correct)

| belief | subset | n | AUROC | ECE |
|---|---|--:|--:|--:|
| LLM gated belief (medpsy-4b) | all | 913 | 0.778 | 0.142 |
| LLM gated belief (medpsy-4b) | single_evidence | 570 | 0.770 | 0.216 |
| LLM gated belief (medpsy-4b) | multi_evidence | 343 | 0.863 | 0.107 |
| LLM gated + calibrated soft (medpsy-4b) | all | 913 | 0.787 | 0.095 |
| LLM gated + calibrated soft (medpsy-4b) | single_evidence | 570 | 0.778 | 0.072 |
| LLM gated + calibrated soft (medpsy-4b) | multi_evidence | 343 | 0.859 | 0.193 |
| text-miner belief · recalibrated priors | all | 913 | 0.738 | 0.370 |
| text-miner belief · recalibrated priors | single_evidence | 570 | 0.744 | 0.297 |
| text-miner belief · recalibrated priors | multi_evidence | 343 | 0.785 | 0.492 |
| text-miner belief · INDRA priors | all | 913 | 0.741 | 0.401 |
| text-miner belief · INDRA priors | single_evidence | 570 | 0.744 | 0.338 |
| text-miner belief · INDRA priors | multi_evidence | 343 | 0.789 | 0.505 |
| INDRA stored belief (w/ propagation) | all | 913 | 0.714 | 0.372 |
| INDRA stored belief (w/ propagation) | single_evidence | 570 | 0.722 | 0.296 |
| INDRA stored belief (w/ propagation) | multi_evidence | 343 | 0.758 | 0.498 |

## Statement-grain error detection (positive = gold-incorrect)

Flag rule: `verdict_statement != correct`. P=0.808 R=0.713 F1=0.757 (tp=303 fp=72 fn=122 tn=416).  
Deterministic hard-flag (`verdict_statement == incorrect`) precision: 1.000.
verdict_statement counts: {'correct': 538, 'review': 360, 'incorrect': 15}.
