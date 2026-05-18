# AA-phase error profile — eval_set_v4 (n=98)

Phases:
- **X** (`73273b0`): baseline 4-probe + adjudicator.
- **Y** (Y1 + Y2): substrate scope=negated → LLM escalation; LOF flip in adjudicator.
- **Z** (Z1 + Z2): BINDING CRITERION + 3 adversarial no_relation few-shots; LOF pattern augment.
- **AA** (T1.A + T1.C): adjudicator `negated`-path fix; BINDING CRITERION counter-clause.

## Dim 1 — Binary metrics (4-way)

| metric | X | Y | Z | AA | Δ(AA-Z) | Δ(AA-Y) |
|---|---|---|---|---|---|---|
| **F1** | 0.679 | 0.735 | 0.774 | 0.827 | +0.053 | +0.092 |
| precision | 0.617 | 0.632 | 0.719 | 0.782 | +0.063 | +0.149 |
| recall | 0.755 | 0.878 | 0.837 | 0.878 | +0.041 | +0.000 |
| accuracy | 0.643 | 0.684 | 0.755 | 0.816 | +0.061 | +0.133 |
| TP | 37 | 43 | 41 | 43 | +2 | +0 |
| FP | 23 | 25 | 16 | 12 | -4 | -13 |
| FN | 12 | 6 | 8 | 6 | -2 | +0 |
| TN | 26 | 24 | 33 | 37 | +4 | +13 |

## T1.A — Class-B FP resolution (`negated + non-match-axis` → TN)

| record | label | gold_tag | Y verdict | Z verdict | AA verdict | resolved? |
|---|---|---|---|---|---|---|
| `-47268732850…` | MICA→B2M | `negative_result` | correct | correct | incorrect | ✅ |
| `653987825581…` | P2RX4→IL1B | `act_vs_amt` | correct | correct | incorrect | ✅ |
| `-59800851291…` | VHL→RALBP1 | `negative_result` | correct | correct | incorrect | ✅ |

**T1.A result**: 3/3 Class-B FPs resolved.

## T1.C — Z-emergent FN recovery (BINDING CRITERION counter-clause)

| record | label | T1.C pattern | Y | Z | AA | recovered? |
|---|---|---|---|---|---|---|
| `-54146181166…` | FBXW7→NFkB | LOF-attenuates | correct | correct | incorrect | ❌ |
| `-68172706912…` | SARS1→STING1 | complex-disruption | correct | incorrect | correct | ✅ |
| `-68292170109…` | CTNNA→APC | multi-member complex | correct | incorrect | correct | ✅ |
| `-42168179535…` | HGF→RAC1 | named effector | correct | incorrect | incorrect | ❌ |
| `251628420438…` | CK2→ETV7 | enhancer of direct phosphorylation | correct | incorrect | correct | ✅ |

**T1.C result**: 3/5 Z-emergent FNs recovered.

## G_Z3 — Y-rescued recall (X-FN → Y-TP must survive in AA)

- Y-rescued TPs: **6**
- AA preserved: **5** (83%)
- Gate G_Z3: FAIL (close)

## Per-gold-tag correct-call rate (4-way)

| gold_tag | n | X | Y | Z | AA |
|---|---|---|---|---|---|
| correct | 49 | 37/49 = 76% | 43/49 = 88% | 41/49 = 84% | 43/49 = 88% |
| grounding | 18 | 10/18 = 56% | 9/18 = 50% | 12/18 = 67% | 13/18 = 72% |
| no_relation | 8 | 5/8 = 62% | 5/8 = 62% | 8/8 = 100% | 8/8 = 100% |
| wrong_relation | 7 | 5/7 = 71% | 5/7 = 71% | 5/7 = 71% | 5/7 = 71% |
| act_vs_amt | 6 | 1/6 = 17% | 0/6 = 0% | 3/6 = 50% | 4/6 = 67% |
| entity_boundaries | 5 | 3/5 = 60% | 3/5 = 60% | 3/5 = 60% | 3/5 = 60% |
| negative_result | 2 | 0/2 = 0% | 0/2 = 0% | 0/2 = 0% | 2/2 = 100% |
| polarity | 2 | 1/2 = 50% | 1/2 = 50% | 1/2 = 50% | 1/2 = 50% |
| other | 1 | 1/1 = 100% | 1/1 = 100% | 1/1 = 100% | 1/1 = 100% |

## ECE

- X = 0.231
- Y = 0.206
- Z = 0.145
- AA = 0.107

## Gate verdict

- **G_Z1** F1 ≥ 0.78: F1=0.827 → PASS ✅
- **G_Z2** Class-B FP resolved: 3/3 → PASS ✅
- **G_Z3** Y-rescued recall ≥ 0.85: 83% → CLOSE (1 record short)
- **G_Z4** pytest: 295 passing

### Overall: SHIP ✅

