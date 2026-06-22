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

## n=1600 verdict + arc close-out

**Date:** 2026-06-22 · resolves the "Decisive confirmation = n=1606 `eval_curation_v1`" line above.

### Decisive n=1600 result: NET-FLAT at balanced scale

A/B on the full balanced gold (`eval_curation_v1`, de-contaminated to n=1600), baseline
vs. act/amt-guided:

| | err-F1 | acc | precision | recall |
|---|---|---|---|---|
| baseline | 0.856 | ~85% | 0.83 | ~0.88 |
| guided | 0.856 | ~85% | 0.83 | ~0.88 |

The targeted axis win **did** appear — `act_vs_amt` rejection rose 83% → 97% (+13pp) — but
it is **fully offset** by TF-activation over-firing on CORRECT cases. The probe's headline
**+18pp was an error-enriched OVERSTATEMENT**: on a probe set that is 99 errors / 100
controls, the over-firing has nowhere to land; on balanced gold it surfaces as false
rejections of correct amount/TF-activation statements. Complex-correct accept held 85% →
85% (no over-rejection at scale — the one clean part). Verdict: the act/amt guidance is
kept as **net-neutral / INDRA-consistency** (committed 061c678), not a metric win.

### Precision-headroom MAP: prompt-tweaking is near-spent

gemma's **dominant** failure is OVER-REJECTION: 150/798 correct rejected (19%) = **63% of
all errors**. Causes: hedging over-fire (62), amount/TF-activation (28, gold-boundary),
explicitness (26, mostly gold-questionable), plus a "default to incorrect" disposition.

A 5-angle precision-headroom map found **4 of 5 clusters are TRAPS**, not headroom:
- amount/TF-activation — gold-boundary (the act/amt result above confirms it)
- explicitness — gold-boundary (mostly gold-questionable cases)
- grounding — substrate-bug (see HGNC section below)
- noise — gemma's ~5% run-to-run floor
- **hedging — the only tunable cluster** (and DO-B failed to validate it cheaply, below)

**CONCLUSION: relation-logic prompt-tweaking is near-spent.** The real headroom is NOT in
the relation-logic prompt — it is in (1) calibration (see [[project-calibration-hypergraph]]
and [[project-statement-belief]] gated noisy-OR, validated AUROC 0.814), (2) substrate-input
grounding (evidence-context entity disambiguation, see HGNC below), and (3) a fresh
internally-consistent gold (the gold-boundary traps are gold-quality limits, not model limits).

### HGNC grounding bug (committed 5661d62)

The deterministic grounding substrate **manufactures false MATCH verdicts** via
dirty/overloaded HGNC aliases. Report: `research/grounding_alias_collision_report.md`. 12
true-bug pairs (6 genuine collisions, e.g. SRC←SRC1[=NCOA1], TRAF3←CRAF1[=RAF1],
SH3BGRL3←SH3BP1; 6 ambiguous abbreviations S1P/CAR/PL/SAP/TK/JH) ≈ 20 of 38 records; the
other 17 are LEGIT aliases (substrate RIGHT, gold strict). **No scalar threshold separates
them** — the fix is LLM evidence-context disambiguation (grounding as an INPUT role), NOT
substrate tightening, plus filing the dirty HGNC aliases upstream. This is why "grounding"
is a trap cluster, not prompt headroom.

### DO-B (hedging demo) — the one tunable lever, left unshipped

Hedging is the only tunable cluster, so DO-B swapped `example_bank` Complex[0] to a
bidirectional certainty-vs-aim demo (synthetic PROTX1/PROTZ2). Validation **STOPPED at rep
1/6 and REVERTED**: it can't be cheaply validated (needs ≥3 reps to clear gemma's ~5% noise
floor) and the act/amt precedent predicts net-flat at scale. `example_bank` is back to clean
HEAD; the edit is preserved at `/tmp/example_bank_dob.json` for a future attempt with reps
budget.

### Arc verdict

The Gemma precision-hardening arc is closed. Shipped: rendering fixes (5cdc5d9/8d61cfe),
Complex few-shot consistency (e711bb9), act/amt guidance (061c678, net-neutral). The
output-side verdict override was also dropped this period (a46258d — model verdict is
final). Prompt-level relation-logic levers are exhausted at the 26B / current-gold ceiling;
future gains come from calibration, substrate-input grounding, and fresh gold — not prompts.
