#!/usr/bin/env python3
"""Join INDRA curations (gold) to MedPsy + gemma scored evidence, compute real
accuracy on the curated subset, and break gold down across the 4 confusion cells.

Gold rule: an INDRA curation tag=='correct' => the reader's extraction is
supported (gold-correct); EVERY other tag (no_relation, wrong_relation,
grounding, polarity, act_vs_amt, hypothesis, negative_result, ...) => the
extraction is wrong (gold-incorrect). Same question our scorers answer.

The gold rule, hash-bridge join, and curation index all come from
indra_belief.curation — this script is just the analysis (accuracy + confusion)
over that gold.
"""
import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from indra_belief.curation import curation_key, load_index  # noqa: E402

CUR = str(ROOT / "data/benchmark/rasmachine_curations.jsonl")
MED = str(ROOT / "data/exports/6aeedd3b76c74f06817b44353c8e91a8/per_evidence.jsonl")
GEM = str(ROOT / "data/exports/rasmachine_belief/per_evidence.jsonl")

# --- gold: (matches_hash, source_hash) -> gold verdict (library-owned) ---
gold_index = load_index(CUR)
gold = {k: gv.verdict for k, gv in gold_index.gold_by_key.items()}

print(f"curated evidences (unique matches_hash+source_hash): {len(gold)}")
gc = collections.Counter(gold.values())
print(f"  gold: correct={gc['correct']} incorrect={gc['incorrect']}")

# --- index model verdicts by (matches_hash, source_hash) ---
def index(path):
    d = {}
    for line in open(path):
        r = json.loads(line)
        key = curation_key(r.get("indra_matches_hash"), r.get("source_hash"))
        if key is None:
            continue
        v = r.get("verdict")
        if v in ("correct", "incorrect"):
            d.setdefault(key, v)  # first wins (dedup)
    return d

med = index(MED)
gem = index(GEM)
print(f"MedPsy verdicts indexed: {len(med)}   gemma: {len(gem)}")

# --- evaluable set: gold present AND both models scored it ---
both = [k for k in gold if k in med and k in gem]
print(f"\nevaluable (gold ∩ MedPsy ∩ gemma): {len(both)}")

def acc(model, name):
    n = tp = tn = fp = fn = 0
    for k in both:
        g = gold[k]; v = model[k]
        n += 1
        if g == "correct" and v == "correct": tp += 1
        elif g == "incorrect" and v == "incorrect": tn += 1
        elif g == "incorrect" and v == "correct": fp += 1
        elif g == "correct" and v == "incorrect": fn += 1
    correct = tp + tn
    print(f"\n{name}: accuracy {correct}/{n} = {100*correct/n:.1f}%")
    # precision/recall treating 'correct' (supported) as positive
    prec = tp/(tp+fp) if tp+fp else float('nan')
    rec = tp/(tp+fn) if tp+fn else float('nan')
    f1 = 2*prec*rec/(prec+rec) if prec+rec else float('nan')
    print(f"  TP={tp} TN={tn} FP={fp} FN={fn}")
    print(f"  precision={prec:.3f} recall={rec:.3f} F1={f1:.3f}  (positive=supported)")
    return correct, n

mc, mn = acc(med, "MedPsy-4B")
gcc, gn = acc(gem, "gemma-26b")

# --- how does gold land in the 4 confusion-matrix cells? ---
print("\n=== gold distribution across the MedPsy×gemma cells (the matrix you asked about) ===")
cells = collections.defaultdict(lambda: collections.Counter())
for k in both:
    a, b = med[k], gem[k]
    cell = f"Med{'✓' if a=='correct' else '✗'} Gem{'✓' if b=='correct' else '✗'}"
    cells[cell][gold[k]] += 1
for cell in ["Med✓ Gem✓", "Med✓ Gem✗", "Med✗ Gem✓", "Med✗ Gem✗"]:
    c = cells[cell]
    tot = c['correct'] + c['incorrect']
    print(f"  {cell}: n={tot}  gold-correct={c['correct']} gold-incorrect={c['incorrect']}")

# the money cells: disagreements, who's right per gold
print("\n=== who is right on disagreements (per gold) ===")
acbi = cells["Med✓ Gem✗"]  # MedPsy says correct, gemma says incorrect
print(f"  Med✓/Gem✗ (n={acbi['correct']+acbi['incorrect']}): MedPsy right {acbi['correct']}, gemma right {acbi['incorrect']}")
aibc = cells["Med✗ Gem✓"]
print(f"  Med✗/Gem✓ (n={aibc['correct']+aibc['incorrect']}): gemma right {aibc['correct']}, MedPsy right {aibc['incorrect']}")
print("DONE")
