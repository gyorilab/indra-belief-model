# Activity-vs-Amount prompt guidance — structured delta

**Date:** 2026-06-21 · **Model:** bedrock-gemma (gemma-4-26b) · **Variant:** disconfirm_relnature

## Change

A research-grounded pass at the one relation-logic weakness the error-profile probe
isolated for gemma-26B: **activity-vs-amount** (`act_vs_amt`) — the model reads
"represses/suppresses/activates" of a target's *transcription / expression / levels /
promoter / stability* as an activity Activation/Inhibition, when it is really an
Increase/DecreaseAmount.

Two edits (no new rule — IFScale: instruction *count* degrades, not length):
1. `_prompts_disconfirm.py` rule 3 — sharpened from an abstract statement into a
   **concrete trigger lexicon** (transcription, expression, mRNA/protein levels,
   abundance, promoter/luciferase/reporter activity, up-/down-regulation, stabilization,
   accumulation, proteasomal degradation) **+ the "what is repressed" nuance** (it is
   activity ONLY when the target's own function/enzymatic activity is gated).
2. `example_bank.json` `Inhibition_act_vs_amt` — replaced the muddy `AKT/Wnt` incorrect
   example (confounded by a hedging error) with a clean **transcriptional-repression**
   demonstration (`PRDM1 [Inhibition] CIITA`, authored / contamination-checked).

Research grounding (`~/clou/knowledge-base/research-foundations.md`): in-context learning
is implicit Bayesian situation-inference (a *demonstration* of the failing pattern beats
an abstract rule); concrete > abstract (content+format co-optimization); minimal
instruction count (IFScale threshold decay); no role theater. The brutalist has no KB.

## Probe: relation-logic discrimination (n=199, real gold)

Set: `data/benchmark/probe_relation_logic.jsonl` — 99 real-gold errors (60 axis
`act_vs_amt` + 36 sign `polarity` + 3 hand-verified subject↔object reversals) + 100
correct-directional controls, sampled from `eval_curation_v1`, contamination-checked
against the few-shots (1 leak removed). Baseline and guided runs use the **identical**
source-hash set (symmetric diff 0); only the guidance differs.

| Class | metric | baseline | guided | Δ |
|---|---|---|---|---|
| axis (`act_vs_amt`) | reject | 47/60 = 78% | **58/60 = 97%** | **+18pp** |
| sign (`polarity`) | reject | 34/36 = 94% | 34/36 = 94% | +0 |
| reversal | reject | 3/3 = 100% | 3/3 = 100% | +0 |
| control (correct) | accept | 81/100 = 81% | **84/100 = 84%** | **+3pp** |
| **all errors** | reject | 85% | **96%** | **+11pp** |

The targeted weakness moved **+18pp (11 cases, beyond gemma's ~2-3-case noise floor)
with no precision cost** — controls *improved* and sign/reversal held. Better
discrimination on both sides, not more rejection.

## Regression check: rasmachine n=61 production set

Identical set, prior (rendering-fix) vs guided:

| run | acc | errF1 | P | R |
|---|---|---|---|---|
| prior (v2) | 95.1% | 0.955 | 0.97 | 0.94 |
| guided | 93.4% | 0.943 | 0.92 | 0.97 |

3 gold-case flips, **none attributable to the act/amt rule**:
- `NFE2L2→KRAS` [IncreaseAmount] wrong_relation: correct→**right** (direction-reversal fix)
- `EXOC2–RALA` [Complex] correct→wrong — objection cites **"may" = hypothesis** (rule 4)
- `CDK2→RB1` [Inhibition] correct→wrong — objection cites **"doesn't explicitly state"**

The −0.012 errF1 is within gemma's run-to-run noise (2-3 flips/run all session); the two
regressions are hedging/explicitness wobble, not amount over-firing.

## Verdict & caveats

Robust win on the target weakness; no act/amt-attributable production regression. Ship-worthy.

Caveats: **single run each** (not repeated reps — the +18pp is beyond noise, the
rasmachine −0.012 is within it); n=61 cannot resolve a 2-case wobble. **Decisive
confirmation = n=1606 `eval_curation_v1`** across the roster.

## Reproduce

```
PYTHONPATH=src python scripts/eval_to_statements_json.py \
  --input data/benchmark/probe_relation_logic.jsonl \
  --output data/corpora/probe_relation_logic_statements.json
python scripts/run_rasmachine_monolithic.py --model bedrock-gemma \
  --input data/corpora/probe_relation_logic_statements.json \
  --output data/results/probe_relation_logic_gemma.jsonl --no-resume --workers 8
# per-class metric: join gemma output to probe gold on source_hash; reject-rate by _class
```
