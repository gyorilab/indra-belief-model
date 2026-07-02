"""What does scoring a bigger gold buy us? Answer it for FREE from the n=61 we
already scored, before spending on a re-run.

Bootstraps gemma's existing rasmachine_v1 (gold_err, pred_err) pairs to get the
error-F1 confidence-interval-vs-n curve, projects it to larger n under the
1/sqrt(n) law, and reports the minimum detectable error-F1 gap at each n — i.e.
how big a model-to-model difference you could actually resolve. The whole point:
distinguish "tighter error bars on gemma's absolute score" (which scoring buys)
from "rank the top-8 frontier cluster" (which it almost certainly can't).

    PYTHONPATH=src .venv/bin/python scripts/bootstrap_precision.py
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

GOLD = ROOT / "data" / "benchmark" / "rasmachine_v1_gold.jsonl"
GEMMA = ROOT / "data" / "results" / "rasmachine_v1_bedrock-gemma.jsonl"
RNG = np.random.default_rng(20260630)


def errf1(pairs) -> float:
    return confusion_pr([(bool(g), bool(p)) for g, p in pairs])["f1"]


def ed_pairs():
    gold = [json.loads(l) for l in open(GOLD) if l.strip()]
    by_pair, by_sh = build_gold_index(gold)
    scored = [json.loads(l) for l in open(GEMMA) if l.strip()]
    joined, _, _ = join_model(scored, by_pair, by_sh)
    return [(not is_gold_correct(g["tag"]), s["verdict"] == "incorrect") for g, s in joined]


def boot_halfwidth(pairs, n, reps=4000) -> float:
    """95% CI half-width of error-F1 for a sample of size n drawn (with
    replacement) from `pairs` — the standard precision-at-n projection."""
    arr = np.array(pairs, dtype=bool)
    m = len(arr)
    stats = np.empty(reps)
    for i in range(reps):
        idx = RNG.integers(0, m, size=n)
        stats[i] = errf1(arr[idx])
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return (hi - lo) / 2.0


def main():
    pairs = ed_pairs()
    n0 = len(pairs)
    f1 = errf1(pairs)
    ng = sum(1 for g, _ in pairs if g)
    print(f"gemma rasmachine_v1: n={n0} ({ng} gold-incorrect / {n0-ng} correct)  error-F1={f1:.3f}")

    print("\n=== empirical bootstrap CI half-width vs n (resampled from the n=61) ===")
    print(f"{'n':>6} {'±halfwidth':>11} {'min detectable Δ':>18}")
    grid = [30, 61, 114, 200, 304, 500, 1000, 2000, 3000]
    hw61 = None
    for n in grid:
        hw = boot_halfwidth(pairs, n)
        if n == 61:
            hw61 = hw
        # min detectable paired Δ ≈ sqrt(2) * single-arm half-width (independent arms;
        # real paired tests do better via correlation, so this is conservative)
        mdd = (2 ** 0.5) * hw
        tag = "  <- current" if n == 61 else ""
        print(f"{n:6d} {hw:11.3f} {mdd:18.3f}{tag}")

    print("\n=== 1/sqrt(n) projection from n=61 (sanity check vs empirical) ===")
    for n in grid:
        proj = hw61 * (61.0 / n) ** 0.5
        print(f"  n={n:5d}: projected ±{proj:.3f}")

    print("\n=== n required to resolve a given error-F1 gap (95%, conservative paired) ===")
    for gap in (0.10, 0.05, 0.03, 0.02, 0.01):
        # need sqrt(2)*hw61*sqrt(61/n) < gap  ->  n > 2*hw61^2*61 / gap^2
        n_req = 2 * hw61 ** 2 * 61 / gap ** 2
        print(f"  gap {gap:.2f}  ->  n ≈ {n_req:,.0f}")


if __name__ == "__main__":
    main()
