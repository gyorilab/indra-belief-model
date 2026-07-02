"""Why subsampling is "predetermined" — and what "simulate more data over time"
can and cannot do. Demonstrated on gemma's existing v2 scored pairs (no new spend).

Panel A  SUBSAMPLE (no replacement, balanced) at increasing n:
         the mean error-F1 is FLAT (pinned to the full-pool value) and the band
         collapses to 0 at n=N. The curve is a deterministic function of the N
         scored points — it carries no information the single full-N estimate
         doesn't. THIS is "predetermined".

Panel B  SIMULATE collecting M MORE (Bayesian posterior-predictive): model the two
         rates error-F1 depends on — sensitivity s=P(pred-inc|gold-inc) and
         false-positive f=P(pred-inc|gold-cor) — with Beta posteriors, draw future
         balanced batches, recompute error-F1 at N+M. This is a REAL "more data over
         time" simulation. Watch two things:
           (1) the CENTER stays ~put (your best guess of unseen data IS your current
               estimate — simulation cannot move the estimate toward an unknown truth)
           (2) the SPREAD shrinks predictably — THIS is the only thing simulation buys:
               a calibrated answer to "how much precision will M more curations get me".

    PYTHONPATH=src .venv/bin/python scripts/demo_simulate_more_data.py
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

RUN = ROOT / "data" / "results" / "rasmachine_v2_bedrock-gemma.jsonl"
GOLD = ROOT / "data" / "benchmark" / "rasmachine_v2_gold.jsonl"
RNG = np.random.default_rng(20260630)


def f1_from_cells(tp, fp, fn):
    denom = 2 * tp + fp + fn
    return (2 * tp / denom) if denom else float("nan")


def load_pairs():
    gold = [json.loads(l) for l in open(GOLD) if l.strip()]
    by_pair, by_sh = build_gold_index(gold)
    scored = [json.loads(l) for l in open(RUN) if l.strip()]
    joined, _, _ = join_model(scored, by_pair, by_sh)
    ge = np.array([not is_gold_correct(g["tag"]) for g, _ in joined])      # gold error
    pe = np.array([s["verdict"] == "incorrect" for _, s in joined])         # pred error
    return ge, pe


def main():
    ge, pe = load_pairs()
    inc = np.where(ge)[0]      # gold-incorrect indices (positive class)
    cor = np.where(~ge)[0]     # gold-correct indices
    # current confusion cells
    TP = int((ge & pe).sum()); FN = int((ge & ~pe).sum())
    FP = int((~ge & pe).sum()); TN = int((~ge & ~pe).sum())
    f1_full = f1_from_cells(TP, FP, FN)
    print(f"gemma v2: incorrect={len(inc)} correct={len(cor)}  cells TP={TP} FN={FN} FP={FP} TN={TN}")
    print(f"full-pool error-F1 = {f1_full:.3f}\n")

    # ── Panel A: subsample WITHOUT replacement, balanced (n/2 each class) ──
    print("=== A. SUBSAMPLE (no replacement, balanced) — PREDETERMINED ===")
    print(f"{'n':>5} {'mean errF1':>11} {'±band(95%)':>11}  (mean is flat → pinned to pool)")
    per = min(len(inc), len(cor))
    for n in [20, 40, 60, 80, 100, 2 * per]:
        half = min(n // 2, per)
        stats = []
        for _ in range(3000):
            si = RNG.choice(inc, half, replace=False)
            sc = RNG.choice(cor, half, replace=False)
            g = ge[np.r_[si, sc]]; p = pe[np.r_[si, sc]]
            tp = int((g & p).sum()); fp = int((~g & p).sum()); fn = int((g & ~p).sum())
            stats.append(f1_from_cells(tp, fp, fn))
        stats = np.array(stats)
        band = (np.percentile(stats, 97.5) - np.percentile(stats, 2.5)) / 2
        tag = "  <- n=N: band→0, you have the whole pool" if half == per else ""
        print(f"{2*half:5d} {stats.mean():11.3f} {band:11.3f}{tag}")

    # ── Panel B: posterior-predictive "collect M more" ──
    # model the two rates error-F1 depends on, estimated from the CURRENT data:
    #   s = P(pred-inc | gold-inc)  (sensitivity);  f = P(pred-inc | gold-cor)  (FP rate)
    print("\n=== B. SIMULATE M MORE (Bayesian posterior-predictive) ===")
    print("  s=P(pred-inc|gold-inc)~Beta(50.5,8.5)  f=P(pred-inc|gold-cor)~Beta(25.5,221.5)")
    p_inc = len(inc) / (len(inc) + len(cor))
    reps = 6000

    def sim(M, frac_inc):
        m_inc = int(round(M * frac_inc)); m_cor = M - m_inc
        fut = np.empty(reps)
        for i in range(reps):
            s = RNG.beta(TP + 0.5, FN + 0.5)
            f = RNG.beta(FP + 0.5, TN + 0.5)
            dtp = RNG.binomial(m_inc, s); dfp = RNG.binomial(m_cor, f)
            fut[i] = f1_from_cells(TP + dtp, FP + dfp, FN + (m_inc - dtp))
        band = (np.percentile(fut, 97.5) - np.percentile(fut, 2.5)) / 2
        return fut.mean(), band

    print("\n  B1. future batches MATCH current composition (19% incorrect):")
    print(f"  {'+M':>6} {'mean future errF1':>18} {'±band(95%)':>11}   <- center FLAT, only spread shrinks")
    for M in [0, 100, 300, 1000, 3000]:
        m, b = sim(M, p_inc)
        print(f"  {M:6d} {m:18.3f} {b:11.3f}")

    print("\n  B2. future batches FORCED balanced (50% incorrect):")
    print(f"  {'+M':>6} {'mean future errF1':>18} {'±band(95%)':>11}   <- 'drift' is COMPOSITION, not learning")
    for M in [0, 100, 300, 1000, 3000]:
        m, b = sim(M, 0.5)
        print(f"  {M:6d} {m:18.3f} {b:11.3f}")

    print("\nREAD:")
    print("  A   subsample: mean FLAT — cannot leave the pool value. Predetermined.")
    print("  B1  simulate matched-composition: mean FLAT too — simulation does NOT move the")
    print("      estimate toward an unknown truth; only the SPREAD shrinks. That shrink is the")
    print("      ONLY thing it buys: a calibrated 'precision after M more' (power/design).")
    print("  B2  the balanced 'drift' 0.75->0.87 looks like learning but is NOT: 0.87 is")
    print("      itself fixed by the CURRENT rates s,f — you only changed the class mix.")
    print("  => Both subsample and simulation are functions of the data you already have.")
    print("     The estimate moves toward TRUTH only when REAL new curations arrive whose")
    print("     s,f DIFFER from today's — the one thing no resample/sim can conjure.")


if __name__ == "__main__":
    main()
