# Per-evidence comparison — the 2023 INDRA paper panel

**5379 human-reviewed evidence pairs** (3520 correct / 1859 incorrect, 34.6% negative) over 1689 of 1689 statements. Join key `(paper_statement_hash, source_hash)`.

Reference arm for every Δ: **INDRA Bayes subtype (OOF)** (the strongest per-evidence baseline by AUROC).

| Arm | kind | AUROC | AP(correct) | AP(incorrect) | err-F1 @0.5 | err-F1 best | distinct |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Gemma 4 E2B | reader | 0.720 | 0.785 | 0.508 | 0.634 | 0.634 | 5 |
| Gemma 4 26B | reader | 0.835 | 0.869 | 0.648 | 0.772 | 0.772 | 4 |
| Gemma 4 31B | reader | 0.850 | 0.880 | 0.673 | 0.791 | 0.791 | 4 |
| GLM-5 | reader | 0.849 | 0.884 | 0.658 | 0.778 | 0.784 | 4 |
| INDRA source prior (bundled) | baseline | 0.502 | 0.656 | 0.347 | 0.000 | 0.515 | 3 |
| INDRA Bayes source (OOF) | baseline | 0.678 | 0.803 | 0.469 | 0.455 | 0.600 | 59 |
| INDRA Bayes subtype (OOF) | baseline | 0.713 | 0.819 | 0.516 | 0.511 | 0.614 | 241 |

## Paired Δ vs the reference (10,000 source-stratified resamples)

| Arm | ΔAUROC [95% CI] | ΔAP(incorrect) [95% CI] |
| --- | --- | --- |
| Gemma 4 E2B | +0.007 [-0.011, +0.025] | -0.009 [-0.029, +0.012] |
| Gemma 4 26B | +0.122 [+0.107, +0.138]* | +0.132 [+0.109, +0.153]* |
| Gemma 4 31B | +0.137 [+0.122, +0.152]* | +0.156 [+0.134, +0.178]* |
| GLM-5 | +0.136 [+0.121, +0.151]* | +0.141 [+0.120, +0.163]* |
| INDRA source prior (bundled) | -0.211 [-0.224, -0.197]* | -0.170 [-0.188, -0.153]* |
| INDRA Bayes source (OOF) | -0.035 [-0.045, -0.026]* | -0.047 [-0.062, -0.032]* |

## Per source

| Source | reviewed | correct frac | bundled prior @1 ev |
| --- | --- | --- | --- |
| reach | 1866 | 0.617 | 0.650 |
| sparser | 1290 | 0.456 | 0.650 |
| trips | 938 | 0.881 | 0.650 |
| medscan | 630 | 0.579 | 0.650 |
| rlimsp | 615 | 0.930 | 0.650 |
| isi | 23 | 0.087 | 0.650 |
| signor | 9 | 1.000 | 0.941 |
| hprd | 8 | 0.875 | 0.890 |

**Shared prior 0.650 at one evidence** covers sparser, medscan, reach, trips, rlimsp; observed correct fraction spans 0.456–0.930 (chi2=672.5, dof=4, p=3.13e-144).

## Grain bridge

Recovered per-evidence verdicts pushed back through `statement_belief` reproduce the shipped statement probabilities:
- Gemma 4 E2B: 1689/1689 exact, max |Δ| = 0
- Gemma 4 26B: 1689/1689 exact, max |Δ| = 0
- Gemma 4 31B: 1689/1689 exact, max |Δ| = 0
- GLM-5: 1689/1689 exact, max |Δ| = 0

## Two grains, one model

| Arm | per-evidence AUROC (n=5,379) | statement AUROC (n=1,689) | change |
| --- | --- | --- | --- |
| Gemma 4 E2B | 0.720 | 0.840 | +0.120 |
| Gemma 4 26B | 0.835 | 0.901 | +0.066 |
| Gemma 4 31B | 0.850 | 0.898 | +0.048 |
| GLM-5 | 0.849 | 0.902 | +0.053 |
| INDRA source prior (bundled) | 0.502 | 0.774 | +0.272 |
| INDRA Bayes source (OOF) | 0.678 | 0.797 | +0.119 |
| INDRA Bayes subtype (OOF) | 0.713 | 0.803 | +0.090 |

_A statement mark and an evidence mark are two measurements of ONE model on two different item populations (1,689 statements at a 73.2% positive rate vs 5,379 evidence pairs at 65.4%). The connector between them shows what changes when INDRA's noisy-OR turns per-evidence verdicts into a statement score; it is not a causal increment and the two AUROCs are not paired._

## Contamination at evidence-pair grain

50 distinct demonstration sentences vs 5371 reviewed pairs: **12 overlapping pairs**, of which **9** also match the demonstration claim at (agent set, statement type) grain.
Excluding them (5367 pairs kept) moves AUROC by at most 0.0008; the primary panel keeps every reviewed pair.
