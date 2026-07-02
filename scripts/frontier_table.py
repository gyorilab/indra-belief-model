"""Cost x error-F1 frontier on the external curator gold (n=578), as a text table.

For every external_curator_v1_* run: error-F1 + bootstrap 95% CI (vs the gold) and
observed cost (call_log tokens x price). Sorted by F1, Pareto-optimal points flagged,
value champion called. The HTML plate is frontier_report.py; this is the numbers.

    PYTHONPATH=src .venv/bin/python scripts/frontier_table.py
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from eval_curation_compare import build_gold_index, join_model  # noqa: E402
from indra_belief.corpus.cost import price_for  # noqa: E402
from indra_belief.curation import is_gold_correct  # noqa: E402
from indra_belief.metrics import confusion_pr  # noqa: E402

GOLD = ROOT / "data" / "benchmark" / "external_curator_gold_v1.jsonl"
EV = 587
RNG = np.random.default_rng(20260630)


def f1(pairs):
    return confusion_pr([(bool(g), bool(p)) for g, p in pairs])["f1"]


def row(run: Path, by_pair, by_sh):
    scored = [json.loads(l) for l in run.open() if l.strip()]
    joined, _, _ = join_model(scored, by_pair, by_sh)
    pairs = np.array([(not is_gold_correct(g["tag"]), s["verdict"] == "incorrect") for g, s in joined], bool)
    if not len(pairs):
        return None
    pt = f1(pairs)
    bs = np.array([f1(pairs[RNG.integers(0, len(pairs), len(pairs))]) for _ in range(2000)])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    # cost
    ti = to = 0; mid = None
    for r in scored:
        for c in (r.get("call_log") or []):
            mid = c.get("model_id") or mid
            ti += c.get("prompt_tokens") or 0
            to += c.get("out_tokens") or 0
    p = price_for(mid or "")
    cost = (ti * p[0] + to * p[1]) / 1e6 if p else None
    name = run.stem.replace("external_curator_v1_bedrock-", "").replace("external_curator_v1_", "")
    return {"name": name, "f1": pt, "lo": lo, "hi": hi, "n": len(pairs),
            "cost": cost, "usd_1k": (cost / EV * 1000) if cost is not None else None}


def main():
    gold = [json.loads(l) for l in GOLD.open() if l.strip()]
    by_pair, by_sh = build_gold_index(gold)
    runs = [r for r in (row(Path(p), by_pair, by_sh)
            for p in sorted(glob.glob(str(ROOT / "data/results/external_curator_v1_*.jsonl")))
            if "progress" not in p) if r]
    runs.sort(key=lambda r: -r["f1"])

    # Pareto: a run is on the frontier if no other run is both cheaper-or-equal and >=F1
    priced = [r for r in runs if r["usd_1k"]]
    for r in priced:
        r["front"] = not any(s is not r and s["usd_1k"] <= r["usd_1k"] and s["f1"] >= r["f1"]
                             and (s["usd_1k"] < r["usd_1k"] or s["f1"] > r["f1"]) for s in priced)

    print(f"=== cost x error-F1 frontier — external-578 (n_gold={runs[0]['n'] if runs else 0}) ===")
    print(f"{'model':>26} {'error-F1 [95%]':>22} {'$/run':>8} {'$/1k ev':>9} {'Pareto':>7}")
    for r in runs:
        c = f"${r['cost']:.2f}" if r["cost"] is not None else "  ?"
        u = f"${r['usd_1k']:.2f}" if r["usd_1k"] is not None else "  ?"
        fr = "  ★" if r.get("front") else ""
        print(f"{r['name']:>26} {r['f1']:.3f} [{r['lo']:.3f},{r['hi']:.3f}] {c:>8} {u:>9} {fr:>7}")

    if priced:
        champ = max(priced, key=lambda r: r["f1"] / max(r["usd_1k"], 1e-6))
        best = max(priced, key=lambda r: r["f1"])
        print(f"\n  best F1:        {best['name']} {best['f1']:.3f} (${best['usd_1k']:.2f}/1k)")
        print(f"  value champion: {champ['name']} {champ['f1']:.3f} (${champ['usd_1k']:.2f}/1k)")
        print(f"  Pareto frontier: {', '.join(r['name'] for r in sorted([x for x in priced if x.get('front')], key=lambda r: r['usd_1k']))}")


if __name__ == "__main__":
    main()
