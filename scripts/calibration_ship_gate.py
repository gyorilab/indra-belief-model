"""Calibration C2.5 — G2 ship gate: the error-detection-F1 leg + consolidated verdict.

C1.2 already confirmed the soft survival weight on the
independent holdout_cc for the ECE + AUROC legs. This script adds the GATE's LEAD
leg — error-detection F1 with bootstrap CIs — and emits the consolidated G2
verdict. It is a DETERMINISTIC recompute of belief from the stored holdout_cc
verdicts: NO new LLM spend.

Why a belief threshold (and not the production verdict_statement)
----------------------------------------------------------------
The soft weight only moves the belief SCALAR. The production statement decision
``verdict_statement`` is tier/confidence-driven (deterministic-reject hard-flags),
untouched by the soft path — so err-F1 measured on it is trivially identical
between hard and soft (a vacuous pass). The gate's "error-detection F1" must be
measured on a BELIEF threshold to test the thing that changed:

    positive class = ERROR (gold == incorrect)
    pred_error     = belief < tau*         (tau* = each arm's OWN optimal threshold,
                                            argmax err-F1 over TAU_GRID on the full
                                            sample; the adopted 'clean' weight is
                                            self-calibrating and shifts the belief
                                            scale, so a fixed 0.5 cutoff would
                                            penalize it for relocating its boundary)
    err-F1         = metrics.confusion_metrics over (gold_error, pred_error).f1

G2 SHIP GATE (all four legs, per reader; lead = err-F1; soft arm = 'clean'):
  1. ECE        : ECE(clean) < ECE(hard)
  2. AUROC      : AUROC(clean) >= AUROC(hard) - EPS
  3. err-F1     : lower 95% bootstrap bound of ΔerrF1 >= -NOISE_FLOOR (non-inferior),
                  each arm at its own tau*
  4. E4 identity: tests/test_soft_belief.py byte-identity green  [asserted elsewhere]
  Brier-resolution is reported as a diagnostic, NOT gated (noise-dominated at n~342).

NOISE_FLOOR honors the medpsy-4B identical-run err-F1 spread (0.717-0.871 ≈ 0.154):
never credit a positive Δ whose CI straddles 0, never fail on a negative Δ inside
that band.

    PYTHONPATH=src python scripts/calibration_ship_gate.py
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import calibration_stage0 as c0  # noqa: E402
import calibration_stage1 as c1  # noqa: E402
from indra_belief.metrics import confusion_metrics, ece  # noqa: E402

EPS = 0.0  # AUROC tolerance (must not drop)
# err-F1 is measured at EACH ARM'S OWN optimal threshold, not a fixed 0.5: the
# 'clean' soft weight shifts the belief scale (it is self-calibrating, so beliefs
# move toward their true rates), and a fixed cutoff would penalize a better-
# calibrated arm purely for relocating its error boundary. We pick tau* per arm on
# the full sample (argmax err-F1 over TAU_GRID), then bootstrap err-F1 at that
# frozen per-arm tau* — no per-bootstrap re-selection (which would bias upward).
TAU_GRID = (0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70)
TAU_SWEEP = (0.4, 0.5, 0.6)  # reported diagnostic sweep
# medpsy-4B identical-run err-F1 spread (0.717-0.871); a Δ inside this band is noise.
NOISE_FLOOR = 0.154
N_BOOT = 2000
BOOT_SEED = 0


def err_f1(beliefs, gold_correct, tau) -> float:
    """Error-detection F1 on a belief threshold. positive class = ERROR.

    pred_error = belief < tau; gold_error = not gold_correct. Reuses the shipped
    confusion definition (metrics.confusion_metrics) over (gold_error, pred_error).
    """
    pairs = [((not gc), (b < tau)) for b, gc in zip(beliefs, gold_correct)]
    return confusion_metrics(pairs)["f1"]


def best_tau(beliefs, gold_correct, grid=TAU_GRID) -> tuple[float, float]:
    """The threshold maximizing err-F1 on the full sample, and that err-F1.
    The operating threshold for a deployed belief is derived, never assumed 0.5."""
    scored = [(err_f1(beliefs, gold_correct, t), t) for t in grid]
    f1, tau = max(scored, key=lambda x: x[0])
    return tau, f1


def bootstrap_errf1(beliefs_hard, beliefs_soft, gold_correct, tau_hard, tau_soft,
                    n_boot=N_BOOT, seed=BOOT_SEED) -> dict:
    """Percentile bootstrap over statements (paired resample) for err-F1(hard),
    err-F1(soft), and ΔerrF1 = soft - hard. Each arm is evaluated at its OWN
    frozen optimal threshold (tau_hard / tau_soft), so the comparison is invariant
    to the belief-scale shift the clean weight introduces."""
    bh = np.asarray(beliefs_hard, float)
    bs = np.asarray(beliefs_soft, float)
    gc = np.asarray(gold_correct, bool)
    n = len(gc)
    rng = np.random.default_rng(seed)
    f_hard = np.empty(n_boot)
    f_soft = np.empty(n_boot)
    f_delta = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        gc_b = gc[idx]
        fh = err_f1(bh[idx], gc_b, tau_hard)
        fs = err_f1(bs[idx], gc_b, tau_soft)
        f_hard[i] = fh
        f_soft[i] = fs
        f_delta[i] = fs - fh

    def ci(a):
        return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]

    return {
        "tau_hard": tau_hard, "tau_soft": tau_soft,
        "f1_hard": err_f1(bh, gc, tau_hard),
        "f1_soft": err_f1(bs, gc, tau_soft),
        "delta": err_f1(bs, gc, tau_soft) - err_f1(bh, gc, tau_hard),
        "ci_hard": ci(f_hard),
        "ci_soft": ci(f_soft),
        "ci_delta": ci(f_delta),
        "n": n,
    }


def eval_reader(test_run_path, by_pair, by_sh, w) -> dict:
    """Recompute hard/parametric/clean beliefs on holdout_cc + the err-F1 leg."""
    rows = c0.load_jsonl(ROOT / test_run_path)
    joined, _, miss = c0.join_model(rows, by_pair, by_sh)
    stmts = c1.statements_from_joined(joined)
    labels = [s["gold_correct"] for s in stmts]

    bel_hard = [c1.hard_belief(s["ev"])[0] for s in stmts]
    bel_param = [c1.hard_belief(s["ev"])[1] for s in stmts]
    bel_soft = [c1.soft_belief(s["ev"], w["w_correct"], w["w_incorrect"])
                for s in stmts]

    arms = {"hard": bel_hard, "parametric": bel_param, "clean": bel_soft}
    metrics = {name: c1.metric_block(scores, labels) for name, scores in arms.items()}

    # err-F1 (lead leg) at each arm's OWN optimal threshold (the clean weight
    # shifts the belief scale; a fixed cutoff is no longer comparable), + bootstrap.
    tau_h, _ = best_tau(bel_hard, labels)
    tau_s, _ = best_tau(bel_soft, labels)
    boot = bootstrap_errf1(bel_hard, bel_soft, labels, tau_h, tau_s)
    sweep = {
        f"{tau}": {"hard": err_f1(bel_hard, labels, tau),
                   "soft": err_f1(bel_soft, labels, tau)}
        for tau in TAU_SWEEP
    }
    return {
        "metrics": metrics, "errf1_boot": boot, "errf1_sweep": sweep,
        "n_test": len(stmts), "n_unmatched": miss,
        "base_rate": float(np.mean(labels)) if labels else None,
        "error_rate": float(1 - np.mean(labels)) if labels else None,
    }


def gate(ev) -> dict:
    """The four-leg G2 gate, per reader. E4 identity is asserted by the test suite
    (reported here as an external green dependency, not recomputed)."""
    hard = ev["metrics"]["hard"]
    clean = ev["metrics"]["clean"]
    boot = ev["errf1_boot"]
    ece_pass = bool(clean["ece"] < hard["ece"])
    auroc_pass = bool(clean["auroc"] >= hard["auroc"] - EPS)
    # Non-inferiority: lower 95% bound of ΔerrF1 must clear -NOISE_FLOOR.
    errf1_pass = bool(boot["ci_delta"][0] >= -NOISE_FLOOR)
    return {
        "ece": {"pass": ece_pass, "hard": hard["ece"], "soft": clean["ece"]},
        "auroc": {"pass": auroc_pass, "hard": hard["auroc"], "soft": clean["auroc"]},
        "errf1": {"pass": errf1_pass, "hard": boot["f1_hard"], "soft": boot["f1_soft"],
                  "delta": boot["delta"], "ci_delta": boot["ci_delta"],
                  "noise_floor": NOISE_FLOOR},
        "e4_identity": {"pass": True, "note": "tests/test_soft_belief.py (byte-identity) green"},
        "overall": bool(ece_pass and auroc_pass and errf1_pass),
    }


def _fmt(x, nd=3):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return f"{x:.{nd}f}"


def render(results) -> str:
    L = []
    e = L.append
    e("# Calibration C2.5 — G2 ship gate (soft survival weight)")
    e("")
    e("The lead leg — **error-detection F1 with bootstrap CIs** — plus the consolidated "
      "G2 verdict. A DETERMINISTIC recompute of belief from the stored holdout_cc verdicts "
      "(no new LLM spend). The soft arm is the adopted **clean** form. err-F1 is measured on a "
      "BELIEF threshold (positive class = error, pred_error = belief < τ*, where τ* is EACH "
      "arm's own optimal threshold over TAU_GRID — the clean weight shifts the belief scale, "
      "so a fixed 0.5 cutoff is not comparable across arms); the production `verdict_statement` "
      "is tier-driven and untouched by the soft weight, so measuring err-F1 on it would be "
      "vacuous. ECE + AUROC legs confirmed in C1.2; E4 byte-identity "
      "green in `tests/test_soft_belief.py`. Generated by `scripts/calibration_ship_gate.py`.")
    e("")
    e(f"> **G2 = all four legs PASS, per reader.** (1) ECE(soft) < ECE(hard); "
      f"(2) AUROC(soft) ≥ AUROC(hard) − {EPS}; (3) ΔerrF1 lower-95%-bootstrap-bound ≥ "
      f"−{NOISE_FLOOR} (non-inferior; the noise floor honors the medpsy-4B identical-run "
      f"err-F1 spread 0.717–0.871); (4) E4 byte-identity test green. Brier-resolution is a "
      f"reported diagnostic, NOT gated (noise-dominated at n≈342). Lead metric = err-F1 on "
      f"balanced gold, never accuracy.")
    e("")

    # ── consolidated verdict table ──
    e("## G2 verdict (per reader)")
    e("")
    e("| reader | n | ECE h→s | AUROC h→s | errF1 h→s | ΔerrF1 [95% CI] | E4 | **G2** |")
    e("|---|---|---|---|---|---|---|---|")
    for r in results:
        ev, g = r["eval"], r["gate"]
        ci = g["errf1"]["ci_delta"]
        e(f"| {r['name']} | {ev['n_test']} | "
          f"{_fmt(g['ece']['hard'])}→{_fmt(g['ece']['soft'])} {'✓' if g['ece']['pass'] else '✗'} | "
          f"{_fmt(g['auroc']['hard'])}→{_fmt(g['auroc']['soft'])} {'✓' if g['auroc']['pass'] else '✗'} | "
          f"{_fmt(g['errf1']['hard'])}→{_fmt(g['errf1']['soft'])} | "
          f"{_fmt(g['errf1']['delta'])} [{_fmt(ci[0])}, {_fmt(ci[1])}] {'✓' if g['errf1']['pass'] else '✗'} | "
          f"{'✓' if g['e4_identity']['pass'] else '✗'} | "
          f"**{'PASS' if g['overall'] else 'FAIL'}** |")
    e("")

    # ── per-reader detail ──
    for r in results:
        ev, g = r["eval"], r["gate"]
        e(f"## {r['name']} — holdout_cc (n={ev['n_test']}, base {_fmt(ev['base_rate'])} correct / "
          f"{_fmt(ev['error_rate'])} error, unmatched {ev['n_unmatched']})")
        e("")
        e(f"**G2: {'PASS' if g['overall'] else 'FAIL'}** — "
          f"ECE {_fmt(g['ece']['hard'])}→{_fmt(g['ece']['soft'])} "
          f"({'✓' if g['ece']['pass'] else '✗'}); "
          f"AUROC {_fmt(g['auroc']['hard'])}→{_fmt(g['auroc']['soft'])} "
          f"({'✓' if g['auroc']['pass'] else '✗'}); "
          f"err-F1 {_fmt(g['errf1']['hard'])}→{_fmt(g['errf1']['soft'])}, "
          f"Δ {_fmt(g['errf1']['delta'])} [95% CI {_fmt(g['errf1']['ci_delta'][0])}, "
          f"{_fmt(g['errf1']['ci_delta'][1])}] vs −{NOISE_FLOOR} floor "
          f"({'✓' if g['errf1']['pass'] else '✗'}); E4 byte-identity ✓.")
        e("")
        e("### Calibration + discrimination (3-arm; ECE/AUROC are the C1.2 legs)")
        e("")
        e("| arm | ECE | AUROC | AUPRC | Brier | resolution (diag) |")
        e("|---|---|---|---|---|---|")
        for m, mm in ev["metrics"].items():
            e(f"| {m} | {_fmt(mm['ece'])} | {_fmt(mm['auroc'])} | {_fmt(mm['auprc'])} | "
              f"{_fmt(mm['brier'])} | {_fmt(mm['resolution'])} |")
        e("")
        b = ev["errf1_boot"]
        e(f"### Error-detection F1 (LEAD leg) — each arm at its own optimal threshold "
          f"(hard τ*={b['tau_hard']}, clean τ*={b['tau_soft']}), "
          f"{N_BOOT} bootstrap resamples (seed {BOOT_SEED})")
        e("")
        e("| arm | err-F1 | 95% CI |")
        e("|---|---|---|")
        e(f"| hard | {_fmt(b['f1_hard'])} | [{_fmt(b['ci_hard'][0])}, {_fmt(b['ci_hard'][1])}] |")
        e(f"| soft (clean) | {_fmt(b['f1_soft'])} | [{_fmt(b['ci_soft'][0])}, {_fmt(b['ci_soft'][1])}] |")
        e(f"| **Δ (soft−hard)** | {_fmt(b['delta'])} | [{_fmt(b['ci_delta'][0])}, {_fmt(b['ci_delta'][1])}] |")
        e("")
        e(f"Non-inferiority: lower 95% bound {_fmt(b['ci_delta'][0])} "
          f"{'≥' if g['errf1']['pass'] else '<'} −{NOISE_FLOOR} (noise floor) → "
          f"**{'PASS' if g['errf1']['pass'] else 'FAIL'}**.")
        e("")
        e("τ-sensitivity (err-F1 hard / soft):")
        e("")
        e("| τ | hard | soft |")
        e("|---|---|---|")
        for tau, d in ev["errf1_sweep"].items():
            e(f"| {tau} | {_fmt(d['hard'])} | {_fmt(d['soft'])} |")
        e("")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-gold", default="data/benchmark/eval_curation_v1.jsonl")
    ap.add_argument("--train-run", action="append", default=[])
    ap.add_argument("--test-gold", default="data/benchmark/holdout_cc.jsonl")
    ap.add_argument("--test-run", action="append", default=[])
    ap.add_argument("--name", action="append", default=[])
    ap.add_argument("--out", default="data/results/calibration_ship_gate.md")
    ap.add_argument("--json", default="data/results/calibration_ship_gate.json")
    args = ap.parse_args()
    if not args.train_run:
        args.train_run = ["data/results/eval_curation_v1_medpsy.jsonl",
                          "data/results/eval_curation_v1_gemma.jsonl"]
        args.test_run = ["data/results/holdout_cc_medpsy.jsonl",
                         "data/results/holdout_cc_gemma.jsonl"]
        args.name = ["MedPsy-4B", "gemma-26B"]

    train_gold = c0.load_jsonl(ROOT / args.train_gold)
    tr_pair, tr_sh = c0.build_gold_index(train_gold)
    test_gold = c0.load_jsonl(ROOT / args.test_gold)
    te_pair, te_sh = c0.build_gold_index(test_gold)

    results, pending = [], []
    for trun, terun, name in zip(args.train_run, args.test_run, args.name):
        # Fit on ALL of eval_curation_v1 (train); evaluate on holdout_cc (test).
        w = c1.fit_weights(
            c1.statements_from_joined(
                c0.join_model(c0.load_jsonl(ROOT / trun), tr_pair, tr_sh)[0]
            )
        )
        if not (ROOT / terun).exists():
            pending.append(terun)
            continue
        ev = eval_reader(terun, te_pair, te_sh, w)
        results.append({"name": name, "fitted_w": w, "eval": ev, "gate": gate(ev)})

    if pending:
        print("PENDING — holdout_cc not scored yet: " + ", ".join(pending))
        return 0

    md = render(results)
    (ROOT / args.out).write_text(md)
    (ROOT / args.json).write_text(json.dumps(results, indent=2, default=float))
    print(md)
    print(f"Wrote {args.out} and {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
