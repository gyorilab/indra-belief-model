# MedPsy-4B vs gemma-26B — human-curation eval (eval_curation_v1)

Balanced 1:1 human gold, fresh + de-contaminated. Gold pairs: 1606.

## Coverage
- MedPsy-4B: joined 1606  (parse-null 0, unmatched 0)
- gemma-26B: joined 1606  (parse-null 0, unmatched 0)
- paired (both parsed): 1604

## Headline — accuracy (verdict == gold)
- **MedPsy-4B: 80.4%**  (95% CI 78.4%–82.3%, n=1606)
- **gemma-26B: 84.1%**  (95% CI 82.3%–85.8%, n=1606)

## Error detection (positive class = curator-flagged INCORRECT)
| model | precision | recall | F1 | TP | FP | FN | TN |
|---|---|---|---|---|---|---|---|
| MedPsy-4B | 0.873 | 0.713 | **0.785** | 573 | 83 | 231 | 719 |
| gemma-26B | 0.869 | 0.803 | **0.835** | 646 | 97 | 158 | 705 |

_Recall = fraction of real errors caught; precision = of flagged, how many were truly wrong._

## Calibration (ECE, 8-bin)
- MedPsy-4B: **0.139**
- gemma-26B: **0.108**

## Per-gold-tag correct-call rate (where each model fails)
| gold tag | n | MedPsy-4B | gemma-26B |
|---|---|---|---|
| correct | 802 | 719/802 (90%) | 705/802 (88%) |
| no_relation | 225 | 167/225 (74%) | 197/225 (88%) |
| grounding | 167 | 109/167 (65%) | 118/167 (71%) |
| wrong_relation | 145 | 111/145 (77%) | 128/145 (88%) |
| other | 62 | 40/62 (65%) | 44/62 (71%) |
| act_vs_amt | 61 | 34/61 (56%) | 48/61 (79%) |
| entity_boundaries | 40 | 27/40 (68%) | 23/40 (57%) |
| polarity | 36 | 33/36 (92%) | 34/36 (94%) |
| hypothesis | 36 | 23/36 (64%) | 26/36 (72%) |
| negative_result | 27 | 26/27 (96%) | 25/27 (93%) |
| mod_site | 4 | 3/4 (75%) | 3/4 (75%) |
| agent_conditions | 1 | 0/1 (0%) | 0/1 (0%) |

## Per-stmt_type accuracy
| stmt_type | n | MedPsy-4B | gemma-26B |
|---|---|---|---|
| Complex | 936 | 83% | 88% |
| Activation | 370 | 78% | 82% |
| Phosphorylation | 120 | 72% | 76% |
| Inhibition | 90 | 71% | 73% |
| IncreaseAmount | 52 | 85% | 83% |
| DecreaseAmount | 30 | 93% | 77% |
| Dephosphorylation | 6 | 83% | 83% |
| Acetylation | 2 | 100% | 100% |

## Paired comparison (McNemar)
- both right: 1216   both wrong: 179
- MedPsy-4B right & gemma-26B wrong (b): **75**
- gemma-26B right & MedPsy-4B wrong (c): **134**
- McNemar two-sided exact p = **0.0001**  (significant at α=0.05)
- direction: gemma-26B

## Verdict disagreements (209 pairs)
- MedPsy-4B=correct, gemma-26B=incorrect: 148
- MedPsy-4B=incorrect, gemma-26B=correct: 61

### MedPsy-4B✓ / gemma-26B✗ (gold tag shown)
| subj | type | obj | gold | tag |
|---|---|---|---|---|
| CDK6 | Complex | EYA2 | correct | correct |
| CDK6 | Complex | EYA2 | correct | correct |
| ERK | Phosphorylation | STAT3 | correct | correct |
| VCL | Complex | SORBS1 | correct | correct |
| TARDBP | Inhibition | TARDBP | correct | correct |
| TARDBP | Inhibition | TARDBP | correct | correct |
| IFNB1 | Activation | IRF1 | correct | correct |
| TRIP10 | Complex | CDC42 | correct | correct |
| PI3K | Complex | PTK2 | correct | correct |
| TCR | Activation | MAPK | correct | correct |
| CTBP1 | Complex | HDGF | correct | correct |
| FOXO | Complex | SIRT1 | correct | correct |
| MIR126 | DecreaseAmount | CXCR4 | correct | correct |
| STK4 | Complex | AR | correct | correct |
| NGF | Phosphorylation | PKA | correct | correct |
| CDKN1A | Complex | ELAVL1 | correct | correct |
| RHOA | Activation | F_actin | correct | correct |
| TXN | Activation | AKT | correct | correct |
| p14_3_3 | Complex | TAZ | correct | correct |
| CBLC | Complex | PDLIM7 | correct | correct |
| RPS2 | Complex | FGF3 | correct | correct |
| MIR33A | DecreaseAmount | CTNNB1 | correct | correct |
| TBP | Complex | TP53 | correct | correct |
| NFkappaB | IncreaseAmount | LCN2 | correct | correct |
| NFkappaB | IncreaseAmount | LCN2 | correct | correct |
| MIR27A | DecreaseAmount | PHB | correct | correct |
| RASD1 | Complex | NONO | correct | correct |
| RNApo_II | Complex | RPAP2 | correct | correct |
| MACC1 | Activation | CTNNB1 | correct | correct |
| MIR10A | DecreaseAmount | NCOR2 | correct | correct |
| CCNB1 | Phosphorylation | MCL1 | correct | correct |
| MAPK7 | Phosphorylation | PTK2 | correct | correct |
| MIR410 | DecreaseAmount | AGTR1 | correct | correct |
| NR3C1 | Inhibition | ESR1 | correct | correct |
| CYP1B1 | Activation | CTNNB1 | correct | correct |
| UHRF1 | Inhibition | DNMT1 | correct | correct |
| UHRF1 | Inhibition | DNMT1 | correct | correct |
| CTBP1 | Complex | EDAR | correct | correct |
| CBL | Complex | CRKL | correct | correct |
| TNFSF10 | Activation | FADD | correct | correct |

### MedPsy-4B✗ / gemma-26B✓ (gold tag shown)
| subj | type | obj | gold | tag |
|---|---|---|---|---|
| VAV1 | Complex | DNM2 | correct | correct |
| DEFA1 | Activation | NFkappaB | correct | correct |
| EHMT2 | Complex | ERCC6 | correct | correct |
| CALM | Complex | CCND1 | correct | correct |
| TRIP10 | Complex | CDC42 | correct | correct |
| MRE11 | Complex | NBN | correct | correct |
| RANGAP1 | Activation | GTPase | correct | correct |
| NDC80 | Complex | PPP1CA | correct | correct |
| NDC80 | Complex | PPP1CA | correct | correct |
| NDC80 | Complex | PPP1CA | correct | correct |
| NDC80 | Complex | PPP1CA | correct | correct |
| NDC80 | Complex | PPP1CA | correct | correct |
| SWI_SNF | Complex | TP53 | correct | correct |
| SFN | Complex | HNRNPA1 | correct | correct |
| INS | Activation | NFE2L2 | correct | correct |
| TBL1X | Complex | CTBP1 | correct | correct |
| AGT | Phosphorylation | CREB | correct | correct |
| HIF1A | Activation | TP53 | correct | correct |
| RTN4 | Complex | FBLN5 | correct | correct |
| CDKN1B | Complex | HSPA8 | correct | correct |
| CDKN1B | Complex | HSPA8 | correct | correct |
| CDKN1B | Complex | HSPA8 | correct | correct |
| PYCARD | Activation | NFkappaB | correct | correct |
| RNH1 | Complex | ANG | correct | correct |
| TRIP11 | Complex | RAB2A | correct | correct |
| YAF2 | Activation | TP53 | correct | correct |
| SYNE2 | Complex | LMNA | correct | correct |
| VCL | Complex | PTK2 | incorrect | no_relation |
| DAB2 | Activation | Integrins | incorrect | entity_boundaries |
| DAB2 | Activation | Integrins | incorrect | entity_boundaries |
| p14_3_3 | Activation | CDK1 | incorrect | hypothesis |
| CYP1A1 | Complex | AHR | incorrect | other |
| CYP1A1 | Complex | AHR | incorrect | wrong_relation |
| SORT1 | Complex | PSRC1 | incorrect | no_relation |
| HNF1A | IncreaseAmount | Wnt | incorrect | negative_result |
| TAT | Activation | NECTIN1 | incorrect | other |
| DAPK3 | Complex | GRB14 | incorrect | grounding |
| DAPK3 | Complex | GRB14 | incorrect | grounding |
| FZD | Complex | DVL1P1 | incorrect | grounding |
| FZD | Complex | DVL1P1 | incorrect | grounding |
