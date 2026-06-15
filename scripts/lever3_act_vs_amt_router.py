import json, re, sys
sys.path.insert(0, "src")
from indra_belief.metrics import confusion_pr

MASK = (1 << 64) - 1
def um(x):
    try:
        return int(x) & MASK
    except Exception:
        return None

GOLD = {um(r["source_hash"]): r for r in (json.loads(l) for l in open("data/benchmark/eval_curation_v1.jsonl"))}

def load(p):
    return {um(r["source_hash"]): r for r in (json.loads(l) for l in open(p)) if r.get("verdict") is not None}

armA = load("data/results/eval_curation_v1_medpsy_disconfirm.jsonl")
gemma = load("data/results/eval_curation_v1_gemma.jsonl")

# amount-cue regex over gold evidence_text
CUE = re.compile(r"(downregulat|up-?regulat|repress|degrad|expression|transcription|promoter|luciferase|reporter|secretion|production|synthesis|mrna|protein level|abundance)", re.I)
AMT_TYPES = {"Activation", "Inhibition"}

def pr(pairs):
    return confusion_pr(pairs)

# universe: gold sh present in BOTH armA and gemma (so router can always resolve)
both = [sh for sh in GOLD if sh in armA and sh in gemma]
# also report armA-only universe (matches reference n)
armA_universe = [sh for sh in GOLD if sh in armA]

# Reference: Arm A alone, gemma alone (on armA_universe and gemma_universe respectively)
armA_pairs = [(GOLD[sh]["gold"] == "incorrect", armA[sh]["verdict"] == "incorrect") for sh in armA_universe]
gemma_universe = [sh for sh in GOLD if sh in gemma]
gemma_pairs = [(GOLD[sh]["gold"] == "incorrect", gemma[sh]["verdict"] == "incorrect") for sh in gemma_universe]

mA = pr(armA_pairs)
mG = pr(gemma_pairs)
print("=== reference (each on its own universe) ===")
print(f"n armA universe={len(armA_universe)}  n gemma universe={len(gemma_universe)}  n both={len(both)}")
print(f"Arm A : F1={mA['f1']:.4f} P={mA['p']:.4f} R={mA['r']:.4f} FP={mA['fp']} FN={mA['fn']} TP={mA['tp']} TN={mA['tn']}")
print(f"gemma : F1={mG['f1']:.4f} P={mG['p']:.4f} R={mG['r']:.4f} FP={mG['fp']} FN={mG['fn']} TP={mG['tp']} TN={mG['tn']}")

# ---- BLENDED router on the 'both' universe ----
def route_verdict(sh):
    g = GOLD[sh]
    cue_fires = bool(CUE.search(g.get("evidence_text") or ""))
    if g.get("stmt_type") in AMT_TYPES and cue_fires:
        return "gemma", gemma[sh]["verdict"]
    return "armA", armA[sh]["verdict"]

blended_pairs = []
escalated = []  # sh routed to gemma
for sh in both:
    src, v = route_verdict(sh)
    blended_pairs.append((GOLD[sh]["gold"] == "incorrect", v == "incorrect"))
    if src == "gemma":
        escalated.append(sh)

mB = pr(blended_pairs)

# Also: armA and gemma each measured on the SAME 'both' universe for a fair apples-to-apples delta
armA_both_pairs = [(GOLD[sh]["gold"] == "incorrect", armA[sh]["verdict"] == "incorrect") for sh in both]
gemma_both_pairs = [(GOLD[sh]["gold"] == "incorrect", gemma[sh]["verdict"] == "incorrect") for sh in both]
mA_both = pr(armA_both_pairs)
mG_both = pr(gemma_both_pairs)

print("\n=== on the shared 'both' universe (n={}) ===".format(len(both)))
print(f"Arm A (both): F1={mA_both['f1']:.4f} P={mA_both['p']:.4f} R={mA_both['r']:.4f}")
print(f"gemma (both): F1={mG_both['f1']:.4f} P={mG_both['p']:.4f} R={mG_both['r']:.4f}")
print(f"BLENDED     : F1={mB['f1']:.4f} P={mB['p']:.4f} R={mB['r']:.4f} FP={mB['fp']} FN={mB['fn']} TP={mB['tp']} TN={mB['tn']}")

# escalation stats
n_esc = len(escalated)
pct_esc = 100.0 * n_esc / len(both)
# selectivity: of escalated, fraction gold-tag act_vs_amt vs other
esc_tag_avt = sum(1 for sh in escalated if GOLD[sh].get("tag") == "act_vs_amt")
esc_tag_other = n_esc - esc_tag_avt
print(f"\nescalated to gemma: {n_esc} / {len(both)} = {pct_esc:.2f}%")
print(f"selectivity: act_vs_amt={esc_tag_avt} ({100.0*esc_tag_avt/n_esc:.1f}%)  other={esc_tag_other} ({100.0*esc_tag_other/n_esc:.1f}%)")

# how many act_vs_amt-tagged cases exist & how many were captured by the router (recall of the router on the tag)
total_avt = sum(1 for sh in both if GOLD[sh].get("tag") == "act_vs_amt")
print(f"total act_vs_amt-tagged in universe: {total_avt}; router captured {esc_tag_avt} of them")

# tag breakdown of escalated
from collections import Counter
esc_tags = Counter(GOLD[sh].get("tag") for sh in escalated)
print("escalated tag breakdown:", dict(esc_tags))

# ---- ORACLE upper bound: escalate ONLY act_vs_amt-tagged cases ----
oracle_pairs = []
oracle_esc = 0
for sh in both:
    if GOLD[sh].get("tag") == "act_vs_amt":
        v = gemma[sh]["verdict"]
        oracle_esc += 1
    else:
        v = armA[sh]["verdict"]
    oracle_pairs.append((GOLD[sh]["gold"] == "incorrect", v == "incorrect"))
mO = pr(oracle_pairs)
print(f"\n=== ORACLE (escalate ONLY act_vs_amt-tagged, n_esc={oracle_esc}) ===")
print(f"ORACLE: F1={mO['f1']:.4f} P={mO['p']:.4f} R={mO['r']:.4f} FP={mO['fp']} FN={mO['fn']}")

print("\n=== DELTAS vs reference bars ===")
print(f"blended F1 {mB['f1']:.4f}  vs Arm A 0.818 -> delta {mB['f1']-0.818:+.4f}")
print(f"blended F1 {mB['f1']:.4f}  vs gemma 0.836 -> delta {mB['f1']-0.836:+.4f}")
print(f"oracle  F1 {mO['f1']:.4f}  vs Arm A 0.818 -> delta {mO['f1']-0.818:+.4f}")
print(f"blended F1 {mB['f1']:.4f}  vs Arm A(both) {mA_both['f1']:.4f} -> delta {mB['f1']-mA_both['f1']:+.4f}")
