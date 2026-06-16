"""Calibration Stage C1 — soft-survival-weight fit + Tier-1/Tier-2 validation.

Interim, ZERO-COST validation (no new LLM scoring): fit the per-read weights on
a train split of eval_curation_v1 and test soft-vs-hard calibration on held-out
statements (group-split by pa_hash so none straddles). The authoritative
validation (C1.2) is a fresh holdout_cc run by both readers — see the data
dependency note in research/calibration_task_hypergraph.md. This stands in for
it and answers the core question: does the soft weight actually beat the hard
gate on calibration, before any production wiring?

The soft model (C2's mechanism, computed here in analysis only):

    per source s, per read j with verdict v_j:
        w_j = w_correct     if v_j == "correct"     (= P(read does NOT support | confirmed))
            = w_incorrect    if v_j == "incorrect"   (= P(read does NOT support | rejected))
            = rand_s         if v_j is None          (prior fallback)
        f_s = syst_s + (geomean_j w_j) ** n_eff,   n_eff = 1 + (n_s - 1)*kappa
        belief = 1 - prod_s f_s

CRITICAL: w_j is P(read does not support) = P(gold == incorrect | verdict). For a
REJECTED read that is (1 - rand_rej) ~ 0.87 (high w -> low belief), NOT rand_rej.
We fit both conditionals directly from the train cells to avoid the sign trap:
    w_correct   = P(gold incorrect | verdict correct)   [== rand_corr]
    w_incorrect = P(gold incorrect | verdict incorrect)  [== 1 - rand_rej]

Two variants reported:
  - replace : w is the pooled per-verdict wrong-rate (the plan's committed form;
              loses per-source granularity -> helps reach/sparser, can hurt the
              high-precision sources trips/signor/rlimsp).
  - guard   : w_correct = min(rand_corr, rand_s) [confirmation can only help],
              w_incorrect = max(1-rand_rej, rand_s) [rejection can only hurt] —
              preserves source ordering. A C2 design candidate.

PRIMARY for the G1 verdict = `replace` at kappa=1.0 (the plan's design, no test
tuning). Other variants/kappas are exploration that informs C2.

    PYTHONPATH=src python scripts/calibration_stage1.py
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import calibration_stage0 as c0  # noqa: E402  (reuse join + metrics)
from indra_belief.curation import aggregate_gold, is_gold_correct  # noqa: E402
from indra_belief.metrics import ece  # noqa: E402
from indra_belief.noise_model import (  # noqa: E402
    _DEFAULT_PRIOR,
    RECALIBRATED_PRIORS,
    compute_gated_belief,
)

PRIORS = RECALIBRATED_PRIORS


def statements_from_joined(joined) -> list[dict]:
    """Group joined (gold, scored) pairs into statements keyed by pa_hash."""
    by_stmt: dict[int, dict] = defaultdict(lambda: {"ev": [], "tags": []})
    for g, s in joined:
        rec = by_stmt[g["pa_hash"]]
        rec["ev"].append({
            "source_api": s.get("source_api") or g.get("source_api"),
            "verdict": s.get("verdict"),
            "ev_gold_correct": is_gold_correct(g["tag"]),
        })
        rec["tags"].append(g["tag"])
    out = []
    for pa, rec in by_stmt.items():
        gold = aggregate_gold(rec["tags"])
        if gold is None:
            continue
        out.append({"pa_hash": pa, "ev": rec["ev"],
                    "n_ev": len(rec["ev"]), "gold_correct": is_gold_correct(gold)})
    return out


def split_stratified(stmts, seed=0, frac_train=0.5):
    """Stratified train/test split by per-statement gold (deterministic)."""
    rng = random.Random(seed)
    pos = [s for s in stmts if s["gold_correct"]]
    neg = [s for s in stmts if not s["gold_correct"]]
    rng.shuffle(pos)
    rng.shuffle(neg)
    tr, te = [], []
    for group in (pos, neg):
        k = int(round(len(group) * frac_train))
        tr += group[:k]
        te += group[k:]
    return tr, te


def fit_weights(train_stmts) -> dict:
    """Fit the two per-read conditional wrong-rates from train EVIDENCES."""
    cc = ci = ic = ii = 0  # (verdict, ev_gold)
    for st in train_stmts:
        for ev in st["ev"]:
            gc = ev["ev_gold_correct"]
            if ev["verdict"] == "correct":
                cc += gc
                ci += (not gc)
            elif ev["verdict"] == "incorrect":
                ic += gc
                ii += (not gc)
    w_correct = ci / (cc + ci) if (cc + ci) else 0.5        # P(gold incorrect | correct) == rand_corr
    w_incorrect = ii / (ic + ii) if (ic + ii) else 0.5      # P(gold incorrect | incorrect) == 1 - rand_rej
    rand_corr = w_correct
    rand_rej = ic / (ic + ii) if (ic + ii) else float("nan")  # P(gold correct | incorrect)
    return {"w_correct": w_correct, "w_incorrect": w_incorrect,
            "rand_corr": rand_corr, "rand_rej": rand_rej,
            "cells": {"cc": cc, "ci": ci, "ic": ic, "ii": ii}}


def soft_belief(evidence, w_correct, w_incorrect, kappa, variant="replace", priors=PRIORS) -> float:
    by_source: dict[str, list[float]] = defaultdict(list)
    syst_of: dict[str, float] = {}
    for ev in evidence:
        src = (ev["source_api"] or "").lower()
        rand_s, syst_s = priors.get(src, _DEFAULT_PRIOR)
        syst_of[src] = syst_s
        v = ev["verdict"]
        if v == "correct":
            w = w_correct if variant == "replace" else min(w_correct, rand_s)
        elif v == "incorrect":
            w = w_incorrect if variant == "replace" else max(w_incorrect, rand_s)
        else:
            w = rand_s
        by_source[src].append(min(max(w, 1e-9), 1.0))
    p_inc = 1.0
    for src, ws in by_source.items():
        n = len(ws)
        geomean = math.exp(sum(math.log(w) for w in ws) / n)
        n_eff = 1.0 + (n - 1) * kappa
        f_s = min(1.0, syst_of[src] + geomean ** n_eff)
        p_inc *= f_s
    return max(0.0, min(1.0, 1.0 - p_inc))


def hard_belief(evidence) -> tuple[float, float]:
    ev = [{"source_api": e["source_api"], "included": e["verdict"] != "incorrect"} for e in evidence]
    res = compute_gated_belief(ev, priors=PRIORS)
    return res.belief, res.parametric_only


def metric_block(scores, labels) -> dict:
    b = c0.brier_murphy(scores, labels)
    return {
        "n": len(scores),
        "ece": ece(list(zip(scores, [bool(x) for x in labels]))),
        "brier": b["brier"], "resolution": b["resolution"], "reliability": b["reliability"],
        "auroc": c0.auroc(scores, labels), "auprc": c0.auprc(scores, labels),
        "mean_correct": float(np.mean([s for s, y in zip(scores, labels) if y])) if any(labels) else None,
        "mean_incorrect": float(np.mean([s for s, y in zip(scores, labels) if not y])) if not all(labels) else None,
    }


def evaluate(test_stmts, w) -> dict:
    labels = [s["gold_correct"] for s in test_stmts]
    methods: dict[str, list[float]] = {"hard": [], "parametric": []}
    kappas = [0.5, 1.0]
    variants = ["replace", "guard"]
    for var in variants:
        for k in kappas:
            methods[f"soft_{var}_k{k}"] = []
    for st in test_stmts:
        hb, pb = hard_belief(st["ev"])
        methods["hard"].append(hb)
        methods["parametric"].append(pb)
        for var in variants:
            for k in kappas:
                methods[f"soft_{var}_k{k}"].append(
                    soft_belief(st["ev"], w["w_correct"], w["w_incorrect"], k, var))
    out = {name: metric_block(scores, labels) for name, scores in methods.items()}
    bins = {name: c0.reliability_bins(scores, labels) for name, scores in methods.items()}
    return {"metrics": out, "bins": bins, "n_test": len(test_stmts),
            "base_rate": float(np.mean(labels))}


def analyze(name, joined, seed) -> dict:
    stmts = statements_from_joined(joined)
    tr, te = split_stratified(stmts, seed=seed)
    w = fit_weights(tr)
    ev = evaluate(te, w)
    # G1 verdict on the PRIMARY: soft_replace_k1.0 vs hard
    primary = ev["metrics"]["soft_replace_k1.0"]
    hard = ev["metrics"]["hard"]
    g1 = {
        "primary": "soft_replace_k1.0",
        "ece_soft": primary["ece"], "ece_hard": hard["ece"],
        "resolution_soft": primary["resolution"], "resolution_hard": hard["resolution"],
        "pass": bool(primary["ece"] < hard["ece"] and primary["resolution"] >= hard["resolution"] - 1e-9),
    }
    # best variant by test ECE (exploration, not the G1 gate)
    best = min((m for m in ev["metrics"] if m.startswith("soft")),
               key=lambda m: ev["metrics"][m]["ece"])
    return {"name": name, "n_statements": len(stmts), "n_train": len(tr), "n_test": len(te),
            "fitted_w": w, "eval": ev, "g1": g1, "best_variant_by_test_ece": best}


def _fmt(x, nd=3):
    if x is None or (isinstance(x, float) and (math.isnan(x))):
        return "—"
    return f"{x:.{nd}f}"


def render_md(results, seed) -> str:
    L = []
    e = L.append
    e("# Calibration Stage C1 — soft survival weight: fit + held-out validation")
    e("")
    e(f"Interim zero-cost validation: fit on a stratified train split of "
      f"eval_curation_v1 (by pa_hash, seed={seed}), test soft-vs-hard on held-out "
      f"statements. The authoritative C1.2 is a fresh holdout_cc run by both readers "
      f"(gated LLM spend). Per-read weight w = P(read does NOT support | verdict): "
      f"w_correct == rand_corr; w_incorrect == 1 - rand_rej. "
      f"PRIMARY for G1 = `soft_replace_k1.0` (plan's committed form, no test tuning). "
      f"Generated by `scripts/calibration_stage1.py`.")
    e("")
    e("## G1 — does the soft weight beat the hard gate on the held-out split?")
    e("")
    e("| reader | n test | ECE hard | ECE soft | Δ | resolution hard→soft | verdict |")
    e("|---|---|---|---|---|---|---|")
    for r in results:
        g = r["g1"]
        verdict = "**PASS**" if g["pass"] else "**FAIL**"
        e(f"| {r['name']} | {r['eval']['n_test']} | {_fmt(g['ece_hard'])} | {_fmt(g['ece_soft'])} | "
          f"{_fmt(g['ece_soft']-g['ece_hard'])} | {_fmt(g['resolution_hard'])}→{_fmt(g['resolution_soft'])} | {verdict} |")
    e("")
    e("> G1 PASS = ECE(soft) < ECE(hard) AND resolution not reduced, on the held-out split, "
      "for the primary `soft_replace_k1.0`. Other variants/kappas below are exploration for C2.")
    e("")
    for r in results:
        e(f"## {r['name']}")
        e("")
        w = r["fitted_w"]
        e(f"- statements {r['n_statements']} (train {r['n_train']} / test {r['n_test']}); base rate "
          f"{_fmt(r['eval']['base_rate'])}")
        e(f"- fitted on train: w_correct (rand_corr) {_fmt(w['w_correct'])}, "
          f"w_incorrect (=1-rand_rej) {_fmt(w['w_incorrect'])} [rand_rej {_fmt(w['rand_rej'])}]; "
          f"cells {w['cells']}")
        e(f"- best variant by test ECE: **{r['best_variant_by_test_ece']}**")
        e("")
        e("| method | ECE | Brier | resolution | AUROC | AUPRC | mean(correct/incorrect) |")
        e("|---|---|---|---|---|---|---|")
        order = ["hard", "parametric", "soft_replace_k0.5", "soft_replace_k1.0",
                 "soft_guard_k0.5", "soft_guard_k1.0"]
        for m in order:
            mm = r["eval"]["metrics"][m]
            e(f"| {m} | {_fmt(mm['ece'])} | {_fmt(mm['brier'])} | {_fmt(mm['resolution'])} | "
              f"{_fmt(mm['auroc'])} | {_fmt(mm['auprc'])} | {_fmt(mm['mean_correct'])}/{_fmt(mm['mean_incorrect'])} |")
        e("")
        e("Reliability — hard gate vs primary soft (`soft_replace_k1.0`):")
        e("")
        e("```")
        e("  HARD:")
        L.extend(c0.reliability_ascii(r["eval"]["bins"]["hard"]))
        e("  SOFT (replace, k=1.0):")
        L.extend(c0.reliability_ascii(r["eval"]["bins"]["soft_replace_k1.0"]))
        e("```")
        e("")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default="data/benchmark/eval_curation_v1.jsonl")
    ap.add_argument("--run", action="append", default=[])
    ap.add_argument("--name", action="append", default=[])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/results/calibration_stage1.md")
    ap.add_argument("--json", default="data/results/calibration_stage1.json")
    args = ap.parse_args()
    if not args.run:
        args.run = ["data/results/eval_curation_v1_medpsy.jsonl",
                    "data/results/eval_curation_v1_gemma.jsonl"]
        args.name = ["MedPsy-4B", "gemma-26B"]
    names = args.name + [f"run{i}" for i in range(len(args.name), len(args.run))]

    gold = c0.load_jsonl(ROOT / args.gold)
    by_pair, by_sh = c0.build_gold_index(gold)
    results = []
    for path, name in zip(args.run, names):
        rows = c0.load_jsonl(ROOT / path)
        joined, _, _ = c0.join_model(rows, by_pair, by_sh)
        results.append(analyze(name, joined, args.seed))

    md = render_md(results, args.seed)
    (ROOT / args.out).write_text(md)
    (ROOT / args.json).write_text(json.dumps(results, indent=2, default=float))
    print(md)
    print(f"\nWrote {args.out} and {args.json}")


if __name__ == "__main__":
    main()
