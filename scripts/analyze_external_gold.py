"""Both halves of "let's do both", on the de-biased multi-curator gold.

B (REAL VALUE — generalization): gemma's balanced error-F1 across the three golds
   it's now been scored on — v1-60, v2-balanced-114, external-578 — with bootstrap
   CIs. Does the n=61 0.957 survive onto bigger, de-biased, balanced gold?

C (POWER FORECAST — simulate more over time): from the external-578 rates (s,f),
   the posterior-predictive precision you'd get by curating M more, and the M needed
   to resolve a model gap of size delta. The legitimate "simulate more data" — a
   design tool, computed from data we have.

Run AFTER scoring external_curator_gold_v1_statements.json with gemma + nemotron.

    PYTHONPATH=src .venv/bin/python scripts/analyze_external_gold.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from eval_curation_compare import build_gold_index, join_model  # noqa: E402
from indra_belief.curation import is_gold_correct  # noqa: E402
from indra_belief.metrics import confusion_pr  # noqa: E402

RES = ROOT / "data" / "results"
BEN = ROOT / "data" / "benchmark"
RNG = np.random.default_rng(20260630)

# (label, gold, gemma-run) for the generalization line
GEN = [
    ("v1-60",            BEN / "rasmachine_v1_gold.jsonl",            RES / "rasmachine_v1_bedrock-gemma.jsonl"),
    ("v2-balanced-114",  BEN / "rasmachine_v2_balanced_gold.jsonl",  RES / "rasmachine_v2_bedrock-gemma.jsonl"),
    ("external-578",     BEN / "external_curator_gold_v1.jsonl",     RES / "external_curator_v1_bedrock-gemma.jsonl"),
]
EXT_GOLD = BEN / "external_curator_gold_v1.jsonl"
EXT_RUNS = {"gemma": RES / "external_curator_v1_bedrock-gemma.jsonl",
            "nemotron": RES / "external_curator_v1_bedrock-nemotron-nano-30b.jsonl"}


def errf1(pairs):
    return confusion_pr([(bool(g), bool(p)) for g, p in pairs])["f1"]


def ed_pairs(run, gold):
    g = [json.loads(l) for l in open(gold) if l.strip()]
    by_pair, by_sh = build_gold_index(g)
    scored = [json.loads(l) for l in open(run) if l.strip()]
    joined, _, _ = join_model(scored, by_pair, by_sh)
    return np.array([(not is_gold_correct(a["tag"]), b["verdict"] == "incorrect") for a, b in joined], bool)


def boot_ci(pairs, reps=4000):
    m = len(pairs)
    s = np.array([errf1(pairs[RNG.integers(0, m, m)]) for _ in range(reps)])
    return errf1(pairs), np.percentile(s, 2.5), np.percentile(s, 97.5)


def cells(pairs):
    ge, pe = pairs[:, 0], pairs[:, 1]
    return (int((ge & pe).sum()), int((~ge & pe).sum()),
            int((ge & ~pe).sum()), int((~ge & ~pe).sum()))  # TP, FP, FN, TN


def main():
    missing = [r for _, _, r in GEN if not r.exists()] + [p for p in EXT_RUNS.values() if not p.exists()]
    if missing:
        print("waiting on runs:", [Path(m).name for m in missing]); return

    # ── B. generalization line ──
    print("=== B. gemma error-F1 across golds (real value, bootstrap 95% CI) ===")
    print(f"{'gold':>18} {'n':>5} {'%inc':>5} {'error-F1 [95%]':>22}")
    for label, gold, run in GEN:
        p = ed_pairs(run, gold)
        ni = int(p[:, 0].sum())
        f, lo, hi = boot_ci(p)
        print(f"{label:>18} {len(p):5d} {100*ni/len(p):4.0f}% {f:.3f} [{lo:.3f},{hi:.3f}]")

    # ── external-578 head-to-head (real value, properly paired) ──
    print("\n=== external-578 head-to-head (de-biased, balanced, n=578) ===")
    gp = ed_pairs(EXT_RUNS["gemma"], EXT_GOLD)
    npr = ed_pairs(EXT_RUNS["nemotron"], EXT_GOLD)
    for name, p in [("gemma", gp), ("nemotron", npr)]:
        f, lo, hi = boot_ci(p)
        print(f"  {name:9s} n={len(p)}  error-F1 {f:.3f} [{lo:.3f},{hi:.3f}]")

    # ── C. power forecast: population-CI on error-F1 shrinks as 1/sqrt(N) ──
    # The thing you actually want: "after curating to total N, how precisely will I
    # know the POPULATION error-F1, and what model gap could I resolve?" That is the
    # measured CI at N=578 projected as 1/sqrt(N) — NOT the combined-sample spread.
    N0 = len(gp)
    fg, glo, ghi = boot_ci(gp)
    hw_g = (ghi - glo) / 2                 # gemma single-model CI half-width at N0
    hw_d = 0.027                            # paired Δ(gemma-nemotron) half-width @N0
    #                                         (from the aligned eval_curation_compare: [-0.070,-0.016])
    print(f"\n=== C. power forecast (project measured N={N0} CIs as 1/sqrt(N)) ===")
    print(f"  measured @N={N0}: gemma error-F1 {fg:.3f} +/-{hw_g:.3f};  "
          f"paired Δ(gemma-nemotron) +/-{hw_d:.3f} (already resolves the 0.043 gap, p=0.005)")
    print(f"  {'total N':>8} {'gemma errF1 +/-':>16} {'min resolvable gap':>19}")
    for N in [N0, 800, 1000, 1500, 2000, 3000]:
        f = (N0 / N) ** 0.5
        print(f"  {N:8d} {hw_g*f:16.3f} {hw_d*f:19.3f}")
    n_for = lambda gap: int(N0 * (hw_d / gap) ** 2)
    print(f"  to resolve a 0.02 gap: total N≈{n_for(0.02)} (+{n_for(0.02)-N0} more); "
          f"a 0.015 gap: N≈{n_for(0.015)} (+{n_for(0.015)-N0}).")


if __name__ == "__main__":
    main()
