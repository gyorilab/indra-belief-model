"""Fine-grained error-F1 learning curve for gemma vs nemotron on the v2 gold.

Answers two things the 3-point (60/114/304) sketch couldn't:
  1. MANY intervals — error-F1 + 95% band across a fine n-grid, per model, from
     the actually-scored v2 pairs (real point estimates, bootstrap CI at each n).
  2. DOES A LARGER CAP HELP — evaluate both models at the three balance/cap cuts
     (capped-64 -> stratified-114 -> full-304) and report the paired gemma-nemotron
     gap + CI at each, so we see whether relaxing the Complex cap (more n, more
     skew) actually buys discrimination or just precision on a shifting target.

    PYTHONPATH=src .venv/bin/python scripts/learning_curve_v2.py
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
BENCH = ROOT / "data" / "benchmark"
RUNS = {
    "gemma": RES / "rasmachine_v2_bedrock-gemma.jsonl",
    "nemotron": RES / "rasmachine_v2_bedrock-nemotron-nano-30b.jsonl",
}
GOLDS = {
    "capped-64": BENCH / "rasmachine_v2_balanced_capped_gold.jsonl",
    "stratified-114": BENCH / "rasmachine_v2_balanced_gold.jsonl",
    "full-304": BENCH / "rasmachine_v2_gold.jsonl",
}
RNG = np.random.default_rng(20260630)


def errf1(arr) -> float:
    return confusion_pr([(bool(g), bool(p)) for g, p in arr])["f1"]


UMASK = (1 << 64) - 1


def _key(g: dict):
    return (int(g["matches_hash"]) & UMASK, int(g["source_hash"]) & UMASK)


def ed_map(run_path: Path, gold_path: Path):
    """Map (matches_hash, source_hash) -> (gold_err, pred_err) so two models can be
    aligned by EVIDENCE, not by file order (concurrent runs write rows out of order;
    pairing by position silently mismatches evidences and inflates the paired CI)."""
    gold = [json.loads(l) for l in open(gold_path) if l.strip()]
    by_pair, by_sh = build_gold_index(gold)
    scored = [json.loads(l) for l in open(run_path) if l.strip()]
    joined, pnull, missed = join_model(scored, by_pair, by_sh)
    m = {_key(g): (not is_gold_correct(g["tag"]), s["verdict"] == "incorrect") for g, s in joined}
    return m, pnull, missed


def ed_pairs(run_path: Path, gold_path: Path):
    m, pnull, missed = ed_map(run_path, gold_path)
    return np.array(list(m.values()), dtype=bool), pnull, missed


def aligned_pairs(run_a: Path, run_b: Path, gold_path: Path):
    """Both models' (gold_err, pred_err) on the SHARED evidences, aligned by key."""
    ma, _, _ = ed_map(run_a, gold_path)
    mb, _, _ = ed_map(run_b, gold_path)
    keys = [k for k in ma if k in mb]
    pa = np.array([ma[k] for k in keys], dtype=bool)
    pb = np.array([mb[k] for k in keys], dtype=bool)
    return pa, pb


def boot(pairs, n, reps=4000):
    """error-F1 mean and 95% band for a sample of size n (with replacement)."""
    m = len(pairs)
    if m == 0:
        return float("nan"), float("nan"), float("nan")
    stats = np.array([errf1(pairs[RNG.integers(0, m, n)]) for _ in range(reps)])
    return float(stats.mean()), float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def paired_delta(pa, pb, n, reps=4000):
    """Bootstrap of (gemma_F1 - nemotron_F1) at sample size n, pairs aligned by
    index (same evidences). Returns point + 95% CI of the difference."""
    m = min(len(pa), len(pb))
    if m == 0:
        return float("nan"), float("nan"), float("nan")
    d = np.empty(reps)
    for i in range(reps):
        idx = RNG.integers(0, m, n)
        d[i] = errf1(pa[idx]) - errf1(pb[idx])
    return errf1(pa[:m]) - errf1(pb[:m]), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def main():
    # load full-304 joined pools per model (the curve's substrate)
    pools = {}
    print("=== join sanity (full-304 gold) ===")
    for name, path in RUNS.items():
        if not path.exists():
            print(f"  {name}: MISSING run file {path.name} — score it first"); return
        pairs, pnull, missed = ed_pairs(path, GOLDS["full-304"])
        pools[name] = pairs
        ninc = int(pairs[:, 0].sum())
        print(f"  {name:9s}: joined n={len(pairs)}  ({ninc} gold-incorrect)  "
              f"parser_null={pnull} missed={missed}  full-pool errF1={errf1(pairs):.3f}")

    # ── 1. fine-grained learning curve on the natural full-304 distribution ──
    pg, pn = pools["gemma"], pools["nemotron"]
    # paired Δ must use evidence-ALIGNED arrays, not per-model file order
    pa_al, pb_al = aligned_pairs(RUNS["gemma"], RUNS["nemotron"], GOLDS["full-304"])
    print(f"  aligned shared pairs for paired Δ: {len(pa_al)}")
    nmax = min(len(pg), len(pn))
    grid = [n for n in (20, 30, 40, 50, 60, 75, 90, 114, 140, 170, 200, 240, 280, nmax)
            if n <= nmax]
    grid = sorted(set(grid))
    print(f"\n=== error-F1 vs n (bootstrap 95% band, natural distribution, n<= {nmax}) ===")
    print(f"{'n':>5} | {'gemma F1 [95%]':^26} | {'nemotron F1 [95%]':^26} | {'Δ g-n [95%]':^24}")
    print("-" * 92)
    for n in grid:
        gm, glo, ghi = boot(pg, n)
        nm, nlo, nhi = boot(pn, n)
        dd, dlo, dhi = paired_delta(pa_al, pb_al, min(n, len(pa_al)))
        sig = "*" if (dlo > 0 or dhi < 0) else " "
        print(f"{n:5d} | {gm:.3f} [{glo:.3f},{ghi:.3f}] | {nm:.3f} [{nlo:.3f},{nhi:.3f}] | "
              f"{dd:+.3f} [{dlo:+.3f},{dhi:+.3f}]{sig}")
    print("  (* = paired Δ CI excludes 0 → models distinguishable at that n)")

    # ── 2. does a larger cap help? per-cut metrics + paired gap ──────────────
    print("\n=== cap relaxation: capped-64 -> stratified-114 -> full-304 ===")
    print(f"{'cut':>15} | {'n':>4} {'%Cmplx':>6} {'%inc':>5} | {'gemma F1':>16} | {'nemotron F1':>16} | {'Δ g-n [95%]':>22}")
    print("-" * 104)
    for cut, gpath in GOLDS.items():
        gold = [json.loads(l) for l in open(gpath) if l.strip()]
        n_gold = len(gold)
        pcx = 100 * sum(1 for r in gold if r["stmt_type"] == "Complex") / max(n_gold, 1)
        pinc = 100 * sum(1 for r in gold if r["gold"] == "incorrect") / max(n_gold, 1)
        pa, _, _ = ed_pairs(RUNS["gemma"], gpath)
        pb, _, _ = ed_pairs(RUNS["nemotron"], gpath)
        gm, glo, ghi = boot(pa, len(pa))
        nm, nlo, nhi = boot(pb, len(pb))
        pa_c, pb_c = aligned_pairs(RUNS["gemma"], RUNS["nemotron"], gpath)  # aligned by evidence
        dd, dlo, dhi = paired_delta(pa_c, pb_c, len(pa_c))
        print(f"{cut:>15} | {len(pa):4d} {pcx:5.0f}% {pinc:4.0f}% | "
              f"{gm:.3f}[{glo:.3f},{ghi:.3f}] | {nm:.3f}[{nlo:.3f},{nhi:.3f}] | "
              f"{dd:+.3f}[{dlo:+.3f},{dhi:+.3f}]")


if __name__ == "__main__":
    main()
