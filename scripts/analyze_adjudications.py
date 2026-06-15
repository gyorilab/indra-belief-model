#!/usr/bin/env python3
"""Reweight inter-model disagreement adjudications into a population answer.

Reads data/truth/adjudications_*.jsonl (written by /adjudicate) against the
frozen queue meta, and reports: when the two models disagree, which one does a
blinded human side with — overall (Horvitz-Thompson reweighted to the full
disagreement population) and per direction / bucket / stmt_type, with
cluster-robust bootstrap CIs (resampled over statements, the PSU).

The headline the corpus motivates: medpsy says "correct" where gemma says
"incorrect" ~7x more than the reverse. This tells you whether that leniency is
medpsy being *right* (gemma over-rejects) or medpsy *rubber-stamping*.

Usage::

    python scripts/analyze_adjudications.py
    python scripts/analyze_adjudications.py --queue data/truth/queue_disagree.jsonl
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
from collections import Counter, defaultdict


def _load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, center - half), min(1.0, center + half))


def _cluster_bootstrap(records, weights, value_fn, rng, n_boot=2000):
    """Reweighted estimate of mean(value_fn) with statement-cluster bootstrap.

    records: list of dicts (must carry 'stmt_hash' and 'direction').
    weights: dict direction -> HT weight (population share). If a direction is
    missing from the sample it simply drops out and the rest renormalize.
    """
    by_stmt = defaultdict(list)
    for r in records:
        by_stmt[r["stmt_hash"]].append(r)
    stmts = list(by_stmt.keys())

    def point(sample_records):
        per_dir = defaultdict(list)
        for r in sample_records:
            v = value_fn(r)
            if v is not None:
                per_dir[r["direction"]].append(v)
        dirs = [d for d in per_dir if per_dir[d]]
        if not dirs:
            return None
        wsum = sum(weights.get(d, 0) for d in dirs) or 1.0
        return sum((weights.get(d, 0) / wsum) * (sum(per_dir[d]) / len(per_dir[d])) for d in dirs)

    est = point(records)
    boot = []
    for _ in range(n_boot):
        draw = [s for s in (stmts[rng.randrange(len(stmts))] for _ in stmts)]
        sample = [r for s in draw for r in by_stmt[s]]
        p = point(sample)
        if p is not None:
            boot.append(p)
    boot.sort()
    lo = boot[int(0.025 * len(boot))] if boot else float("nan")
    hi = boot[int(0.975 * len(boot))] if boot else float("nan")
    return est, lo, hi


def _cohens_kappa(pairs):
    """pairs: list of (label_a, label_b). Returns kappa or None."""
    if not pairs:
        return None
    cats = sorted({x for p in pairs for x in p})
    n = len(pairs)
    po = sum(1 for a, b in pairs if a == b) / n
    pe = 0.0
    for c in cats:
        pa = sum(1 for a, _ in pairs if a == c) / n
        pb = sum(1 for _, b in pairs if b == c) / n
        pe += pa * pb
    if pe == 1.0:
        return 1.0
    return (po - pe) / (1 - pe)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--queue", default="data/truth/queue_disagree.jsonl")
    ap.add_argument("--labels-glob", default="data/truth/adjudications_*.jsonl")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()

    meta_path = args.queue.replace(".jsonl", ".meta.json")
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
    model_a = meta.get("model_a", "model_a")
    model_b = meta.get("model_b", "model_b")
    weights = meta.get("ht_weights", {})
    pop = meta.get("population_disagreements", {})

    files = sorted(glob.glob(args.labels_glob))
    records = []
    for fp in files:
        ann = os.path.basename(fp).removeprefix("adjudications_").removesuffix(".jsonl")
        for r in _load_jsonl(fp):
            r["_annotator"] = ann
            records.append(r)

    print(f"disagreement adjudication · {model_a} (A) vs {model_b} (B)")
    print(f"labels: {len(records)} from {len(files)} annotator file(s)")
    if not records:
        print("\n(no adjudications yet — label some at /adjudicate, then re-run)")
        return 0

    rng = random.Random(args.seed)

    # ---- headline: who does the human side with, reweighted to population ----
    # one record per (item, annotator); for the population estimate use the
    # primary pass only (first annotator listed in queue) to avoid double weight.
    primary = {it["item_id"]: it["annotators"][0] for it in _load_jsonl(args.queue)} if os.path.exists(args.queue) else {}
    prim = [r for r in records if primary.get(r["item_id"]) == r["_annotator"]] or records

    def sides_with_b(r):
        return 1.0 if r["sided_with"] == "model_b" else 0.0 if r["sided_with"] == "model_a" else None

    def sides_with_a(r):
        return 1.0 if r["sided_with"] == "model_a" else 0.0 if r["sided_with"] == "model_b" else None

    est_b, lo_b, hi_b = _cluster_bootstrap(prim, weights, sides_with_b, rng, args.n_boot)
    decided = [r for r in prim if r["sided_with"] in ("model_a", "model_b")]
    abstain = sum(1 for r in prim if r["sided_with"] == "neither")
    print(f"\namong decided disagreements (n={len(decided)}, abstained {abstain}):")
    print(f"  reweighted P(human sides with {model_b}) = {est_b:.1%}  [{lo_b:.1%}, {hi_b:.1%}]")
    if est_b == est_b:  # not nan
        print(f"  reweighted P(human sides with {model_a}) = {1-est_b:.1%}  [{1-hi_b:.1%}, {1-lo_b:.1%}]")

    # ---- per direction (the raw asymmetry, unweighted within each) ----
    print(f"\n{'direction':<26}{'pop':>7}{'n':>5}{'→'+model_b[:10]:>14}{'wilson 95%':>20}")
    by_dir = defaultdict(list)
    for r in decided:
        by_dir[r["direction"]].append(r)
    for dirn in ("a_correct_b_incorrect", "a_incorrect_b_correct"):
        rs = by_dir.get(dirn, [])
        kb = sum(1 for r in rs if r["sided_with"] == "model_b")
        lo, hi = _wilson(kb, len(rs))
        rate = f"{kb/len(rs):.0%}" if rs else "—"
        ci = f"[{lo:.0%}, {hi:.0%}]" if rs else ""
        print(f"{dirn:<26}{pop.get(dirn,0):>7}{len(rs):>5}{rate:>14}{ci:>20}")
    print(f"  (→{model_b} = fraction where the blinded human sided with {model_b})")

    # ---- interpretation hook for the dominant direction ----
    dom = by_dir.get("a_incorrect_b_correct", [])
    if dom:
        kb = sum(1 for r in dom if r["sided_with"] == "model_b") / len(dom)
        verdict = (f"{model_b} is right (→ {model_a} over-rejects)" if kb >= 0.6
                   else f"{model_a} is right (→ {model_b} rubber-stamps)" if kb <= 0.4
                   else "genuinely split")
        print(f"\non '{model_a}=incorrect, {model_b}=correct' (the {pop.get('a_incorrect_b_correct',0)}-case bulk): "
              f"{kb:.0%} side with {model_b} → {verdict}")

    # ---- cross-cuts ----
    for key, title in (("bucket_a", f"by {model_a} bucket"), ("stmt_type", "by stmt_type")):
        print(f"\n{title}:")
        groups = defaultdict(list)
        for r in decided:
            groups[r.get(key) or "—"].append(r)
        for g, rs in sorted(groups.items(), key=lambda kv: -len(kv[1])):
            kb = sum(1 for r in rs if r["sided_with"] == "model_b")
            print(f"  {g:<22} n={len(rs):<4} →{model_b}={kb/len(rs):.0%}")

    # ---- ambiguity + inter-annotator agreement ----
    amb = sum(1 for r in prim if r.get("ambiguous"))
    print(f"\nflagged genuinely ambiguous: {amb}/{len(prim)} ({amb/len(prim):.0%})")
    dbl = defaultdict(dict)
    for r in records:
        dbl[r["item_id"]][r["_annotator"]] = r["human_verdict"]
    pairs = [tuple(v.values()) for v in dbl.values() if len(v) == 2]
    if pairs:
        agree = sum(1 for a, b in pairs if a == b) / len(pairs)
        kappa = _cohens_kappa(pairs)
        print(f"double-labeled: {len(pairs)} items · raw agreement {agree:.0%} · Cohen's κ {kappa:.2f}")
    else:
        print("double-labeled: none yet (need a second annotator on shared items)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
