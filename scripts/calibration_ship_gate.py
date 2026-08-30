"""Hybrid log-odds belief ship gate.

G2 is this repository's name for the four-leg gate below. Generated reports
print it, and ``src/indra_belief/calibration_gate.py`` and
``tests/test_local_reader_calibration.py`` cite it when referring to this
gate. This script derives a reader measurement profile from its training
confusion cells, recomputes the hybrid log-odds belief on independent gold, and
evaluates all four legs. Error-detection F1 with bootstrap CIs leads. This is a
DETERMINISTIC recompute from stored verdicts: no new LLM spend.

Why a belief threshold (and not the production verdict_statement)
----------------------------------------------------------------
The calibrated hybrid arm only moves the belief SCALAR. The production statement
decision ``verdict_statement`` is tier/confidence-driven (deterministic-reject
hard-flags), untouched by the calibrated path — so err-F1 measured on it is
trivially identical between hard and calibrated belief (a vacuous pass). The
gate's "error-detection F1" must be measured on a BELIEF threshold to test the
thing that changed:

    positive class = ERROR (gold == incorrect)
    pred_error     = belief < tau*         (tau* = each arm's OWN operating threshold,
                                            selected on eval_curation_v1 and then
                                            frozen before the independent test)
    err-F1         = metrics.confusion_metrics over (gold_error, pred_error).f1

G2 SHIP GATE (all four legs, per reader; lead = err-F1; calibrated arm key = 'clean'):
  1. ECE        : ECE(calibrated) < ECE(hard)
  2. AUROC      : AUROC(calibrated) >= AUROC(hard) - EPS
  3. err-F1     : lower 95% bootstrap bound of ΔerrF1 >= -NI_MARGIN (non-inferior),
                  each arm at its own train-selected tau*
  4. E4 identity: byte-identity of the soft path, locked by tests/test_soft_belief.py,
                  surfaced as CLI flag ``--e4-identity-pass`` and JSON key ``e4_identity``  [asserted elsewhere]
  Brier-resolution is reported as a diagnostic, NOT gated (noise-dominated at n~342).

The 0.154 non-inferiority margin is the historically observed medpsy-4B
identical-run err-F1 spread (0.717-0.871). It is deliberately disclosed as a
wide, empirical margin rather than described as sampling noise.

Exit-code contract (main): the md/json report is always written and printed,
then the process exits NON-ZERO if any requested test run is PENDING (evidence
missing) or any evaluated reader's gate.overall is False; it exits 0 only when
every evaluated reader PASSes and nothing is pending. A gate that returns success
on a FAIL or a missing run cannot stop a ship. Intentionally DISABLING a failed
candidate remains an explicit ``_PROFILE_META`` deployment_status decision in
``calibration_constants.py``: a FAIL exit forces a conscious override, it does
not auto-disable the reader.

    PYTHONPATH=src python -m pytest -q tests/test_soft_belief.py && \
      PYTHONPATH=src python scripts/calibration_ship_gate.py --e4-identity-pass
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import calibration_stage0 as c0  # noqa: E402
import calibration_stage1 as c1  # noqa: E402
from indra_belief.calibration_constants import reader_configuration_for_run  # noqa: E402
from indra_belief.curation import aggregate_gold, is_gold_correct  # noqa: E402
from indra_belief.metrics import confusion_metrics, ece  # noqa: E402
from indra_belief.noise_model import RECALIBRATED_PRIORS  # noqa: E402
from indra_belief.results import load_gold_map  # noqa: E402
from indra_belief.statement_belief import statement_belief  # noqa: E402

EPS = 0.0  # AUROC tolerance (must not drop)
# err-F1 is measured at EACH ARM'S OWN operating threshold, not a fixed 0.5.
# Thresholds are selected once on the fit set and frozen before evaluation on a
# disjoint test set. Bootstrap resamples never re-select them.
TAU_GRID = (0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70)
TAU_SWEEP = (0.4, 0.5, 0.6)  # reported diagnostic sweep
# Pre-specified non-inferiority margin based on the historically observed
# medpsy-4B identical-run err-F1 spread (0.717-0.871). This is intentionally
# wide and must not be described as a statistical noise estimate.
NI_MARGIN = 0.154
N_BOOT = 2000
BOOT_SEED = 0
HASH_MASK = (1 << 64) - 1


def file_sha256(path: str | Path) -> str:
    """Hash one gate input relative to the repository root."""
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    return hashlib.sha256(p.read_bytes()).hexdigest()


def production_profile_id(reader_configuration: str, fit_gold: str | Path) -> str:
    """Use the same compact display ID persisted by the production profile.

    ``reader_configuration`` remains the full model+prompt SHA identity; profile_id
    is its stable human-facing alias. Keeping both prevents the gate artifact from
    inventing a second profile identity for the same fitted configuration.
    """
    marker = "@prompt-sha256:"
    if marker not in reader_configuration:
        raise ValueError(f"unrecognized reader configuration: {reader_configuration!r}")
    model, prompt_sha256 = reader_configuration.split(marker, 1)
    if len(prompt_sha256) != 64:
        raise ValueError(f"invalid prompt SHA in reader configuration: {reader_configuration!r}")
    return f"{model}@prompt-{prompt_sha256[:12]}@{Path(fit_gold).stem}"


def err_f1(beliefs, gold_correct, tau) -> float:
    """Error-detection F1 on a belief threshold. positive class = ERROR.

    pred_error = belief < tau; gold_error = not gold_correct. Reuses the shipped
    confusion definition (metrics.confusion_metrics) over (gold_error, pred_error).
    """
    pairs = [((not gc), (b < tau)) for b, gc in zip(beliefs, gold_correct)]
    return confusion_metrics(pairs)["f1"]


def best_tau(beliefs, gold_correct, grid=TAU_GRID) -> tuple[float, float]:
    """Return the err-F1-maximizing threshold and score on a training sample."""
    scored = [(err_f1(beliefs, gold_correct, t), t) for t in grid]
    f1, tau = max(scored, key=lambda x: x[0])
    return tau, f1


def _ukey(value) -> int | None:
    try:
        return int(value) & HASH_MASK
    except (TypeError, ValueError):
        return None


def _run_statement_key(value) -> tuple[str, int | None]:
    """Return a stable display key and the unsigned INDRA matches hash.

    Scored runs persist ``stmt_hash`` as a 16-character hexadecimal string. The
    unsigned integer is needed for an exact ``(matches_hash, source_hash)`` gold
    join; the original string remains the production statement grouping key.
    """
    if value is None:
        return "", None
    display = str(value)
    try:
        return display, int(display, 16) & HASH_MASK
    except (TypeError, ValueError):
        return display, _ukey(value)


def _run_configuration(run_path: str | Path) -> dict:
    """Read the exact model+prompt identity persisted by a run."""
    return reader_configuration_for_run(ROOT / run_path)


def validate_configuration_pair(
    train_run: str | Path, test_run: str | Path,
) -> tuple[str | None, str | None]:
    """Reject accidental cross-configuration profile transfer.

    A profile is valid only for the serving/scorer configuration that produced
    its fit run. Missing or mixed prompt fingerprints cannot establish
    equivalence and are a hard provenance error, as are two different IDs.
    """
    train_configuration = _run_configuration(train_run)
    test_configuration = _run_configuration(test_run)
    if train_configuration["status"] != "identified":
        raise ValueError(
            f"training reader configuration is not identifiable: {train_run}: "
            f"{train_configuration}"
        )
    if test_configuration["status"] != "identified":
        raise ValueError(
            f"test reader configuration is not identifiable: {test_run}: "
            f"{test_configuration}"
        )
    if train_configuration["id"] != test_configuration["id"]:
        raise ValueError(
            "calibration profile transfer across reader configurations is forbidden: "
            f"train {train_run}={train_configuration['id']!r}, "
            f"test {test_run}={test_configuration['id']!r}"
        )
    return train_configuration["id"], test_configuration["id"]


def statements_for_run(run_path: str | Path, gold_path: str | Path) -> tuple[list[dict], dict]:
    """Join a run to gold and build production-grain statement records.

    Each row prefers its exact canonical pair and otherwise uses source-hash only
    when every context sharing that source agrees on truth. Statements are
    grouped by the run's ``stmt_hash`` (the production grain), and statement gold
    uses the shared conservative any-incorrect-wins rollup.
    """
    # Two runners persist different row identities. The monolithic runner writes
    # (stmt_i, evidence_i); run_vllm_gold_eval.py writes a flat `row_index`. Keying
    # only on the former collapsed EVERY row of such a run onto (None, None) — the
    # file read as a single row and the fit died on an empty confusion cell rather
    # than reporting an unreadable input.
    run_rows_by_position: dict[tuple, dict] = {}
    for position, row in enumerate(c0.load_jsonl(ROOT / run_path)):
        stmt_i, evidence_i = row.get("stmt_i"), row.get("evidence_i")
        if stmt_i is None and evidence_i is None:
            key = ("row_index", row.get("row_index", position))
        else:
            key = (stmt_i, evidence_i)
        run_rows_by_position[key] = row
    run_rows = list(run_rows_by_position.values())
    gold_rows = c0.load_jsonl(ROOT / gold_path)
    gold_map = load_gold_map(str(ROOT / gold_path))
    by_pair: dict[tuple[int, int], list[dict]] = defaultdict(list)
    by_source: dict[int, list[dict]] = defaultdict(list)
    for g in gold_rows:
        sh = _ukey(g.get("source_hash"))
        if sh is None:
            continue
        by_source[sh].append(g)
        mh = _ukey(g.get("matches_hash"))
        if mh is not None:
            by_pair[(mh, sh)].append(g)

    grouped: dict[str, dict] = defaultdict(
        lambda: {"ev": [], "tags": [], "stored_belief": None}
    )
    n_joined = n_unmatched = n_ambiguous = n_exact = n_source_fallback = 0
    for scored in run_rows:
        stmt_key, mh = _run_statement_key(scored.get("stmt_hash"))
        if mh is None:
            # Same statement identity, different encoding: the monolithic runner
            # persists it as `stmt_hash` (16-char hex), run_vllm_gold_eval.py as
            # `matches_hash` (already the unsigned integer). Without this every
            # row grouped under "" — one statement for the whole run.
            mh = _ukey(scored.get("matches_hash"))
            if mh is not None:
                stmt_key = str(mh)
        sh = _ukey(scored.get("source_hash"))
        exact_candidates = by_pair.get((mh, sh), [])
        candidates = exact_candidates or by_source.get(sh, [])
        gold = gold_map.for_row(mh, sh)
        if not candidates:
            n_unmatched += 1
            continue
        if gold is None:
            n_ambiguous += 1
            continue
        pair_gold = gold.get("verdict")
        if pair_gold is None:
            n_ambiguous += 1
            continue
        if exact_candidates:
            n_exact += 1
        else:
            n_source_fallback += 1
        g0 = candidates[0]
        rec = grouped[stmt_key]
        rec["ev"].append({
            "source_api": scored.get("source_api") or g0.get("source_api"),
            "verdict": scored.get("verdict"),
            "confidence": scored.get("confidence"),
            "tier": scored.get("tier"),
            "evidence_text": g0.get("evidence_text"),
            "evidence_hash": scored.get("evidence_hash"),
            "ev_gold_correct": is_gold_correct(pair_gold),
        })
        rec["tags"].append(pair_gold)
        if rec["stored_belief"] is None and isinstance(scored.get("belief"), (int, float)):
            rec["stored_belief"] = scored["belief"]
        n_joined += 1

    statements = []
    for key, rec in grouped.items():
        statement_gold = aggregate_gold(rec["tags"])
        if statement_gold is None:
            continue
        statements.append({
            "statement_key": key,
            "ev": rec["ev"],
            "gold_correct": is_gold_correct(statement_gold),
            "stored_belief": rec["stored_belief"],
        })
    diagnostics = {
        "join_mode": "per-row exact (matches_hash, source_hash) first; truth-safe source fallback",
        "n_run_rows": len(run_rows),
        "n_joined_rows": n_joined,
        "n_exact_joined_rows": n_exact,
        "n_source_fallback_rows": n_source_fallback,
        "n_unmatched_rows": n_unmatched,
        "n_ambiguous_rows": n_ambiguous,
        "n_grouped_statements": len(statements),
    }
    return statements, diagnostics


def score_statements(statements: list[dict], profile: dict) -> dict:
    """Score statements through the production hard and calibrated paths."""
    hard: list[float] = []
    parametric: list[float] = []
    calibrated: list[float] = []
    labels: list[bool] = []
    n_undefined = 0
    for stmt in statements:
        hard_result = statement_belief(stmt["ev"], RECALIBRATED_PRIORS)
        soft_result = statement_belief(stmt["ev"], RECALIBRATED_PRIORS, soft=profile)
        if hard_result.belief is None or soft_result.belief is None:
            n_undefined += 1
            continue
        hard.append(hard_result.belief)
        parametric.append(hard_result.parametric_only)
        calibrated.append(soft_result.belief)
        labels.append(stmt["gold_correct"])
    return {
        "hard": hard,
        "parametric": parametric,
        "calibrated": calibrated,
        "labels": labels,
        "n_undefined": n_undefined,
    }


def training_thresholds(statements: list[dict], profile: dict) -> dict:
    """Select each arm's operating threshold on training statements only."""
    scored = score_statements(statements, profile)
    if not scored["labels"]:
        raise ValueError("cannot select operating thresholds without scored training statements")
    tau_hard, f1_hard = best_tau(scored["hard"], scored["labels"])
    tau_soft, f1_soft = best_tau(scored["calibrated"], scored["labels"])
    return {
        "hard": tau_hard,
        "calibrated": tau_soft,
        "train_f1_hard": f1_hard,
        "train_f1_calibrated": f1_soft,
        "n_train": len(scored["labels"]),
        "n_train_undefined": scored["n_undefined"],
        "selection": "argmax error-detection F1 on train over TAU_GRID; frozen before test",
        "grid": list(TAU_GRID),
    }


def bootstrap_errf1(beliefs_hard, beliefs_soft, gold_correct, tau_hard, tau_soft,
                    n_boot=N_BOOT, seed=BOOT_SEED) -> dict:
    """Percentile bootstrap over statements (paired resample) for err-F1(hard),
    err-F1(soft), and ΔerrF1 = soft - hard. Each arm is evaluated at its OWN
    frozen train-selected threshold (tau_hard / tau_soft), so the test sample
    never chooses its own operating point."""
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


def eval_reader(test_statements, profile, thresholds, join_diagnostics=None) -> dict:
    """Evaluate production hard/calibrated beliefs at frozen train thresholds."""
    scored = score_statements(test_statements, profile)
    labels = scored["labels"]
    bel_hard = scored["hard"]
    bel_param = scored["parametric"]
    bel_soft = scored["calibrated"]
    arms = {"hard": bel_hard, "parametric": bel_param, "clean": bel_soft}
    metrics = {name: c1.metric_block(scores, labels) for name, scores in arms.items()}

    # Each arm's operating threshold was selected on the disjoint training gold.
    tau_h = thresholds["hard"]
    tau_s = thresholds["calibrated"]
    boot = bootstrap_errf1(bel_hard, bel_soft, labels, tau_h, tau_s)
    sweep = {
        f"{tau}": {"hard": err_f1(bel_hard, labels, tau),
                   "soft": err_f1(bel_soft, labels, tau)}
        for tau in TAU_SWEEP
    }
    return {
        "metrics": metrics, "errf1_boot": boot, "errf1_sweep": sweep,
        "thresholds": thresholds,
        "n_test": len(labels), "n_undefined": scored["n_undefined"],
        "join": join_diagnostics or {},
        "base_rate": float(np.mean(labels)) if labels else None,
        "error_rate": float(1 - np.mean(labels)) if labels else None,
    }


def gate(ev, *, e4_identity_pass: bool = False) -> dict:
    """The four-leg G2 gate, per reader.

    E4 is the external test-suite result for byte-identity of the soft path,
    locked by ``tests/test_soft_belief.py``. Callers assert it explicitly with
    ``--e4-identity-pass``, and the gate records the result under the
    ``e4_identity`` JSON key. The default is deliberately false: a standalone
    metrics recompute must never silently manufacture a green compatibility leg.
    """
    hard = ev["metrics"]["hard"]
    clean = ev["metrics"]["clean"]
    boot = ev["errf1_boot"]
    ece_pass = bool(clean["ece"] < hard["ece"])
    auroc_pass = bool(clean["auroc"] >= hard["auroc"] - EPS)
    # Non-inferiority: lower 95% bound of ΔerrF1 must clear -NI_MARGIN.
    errf1_pass = bool(boot["ci_delta"][0] >= -NI_MARGIN)
    return {
        "ece": {"pass": ece_pass, "hard": hard["ece"], "soft": clean["ece"]},
        "auroc": {"pass": auroc_pass, "hard": hard["auroc"], "soft": clean["auroc"]},
        "errf1": {"pass": errf1_pass, "hard": boot["f1_hard"], "soft": boot["f1_soft"],
                  "delta": boot["delta"], "ci_delta": boot["ci_delta"],
                  "noninferiority_margin": NI_MARGIN},
        "e4_identity": {
            "pass": bool(e4_identity_pass),
            "note": ("tests/test_soft_belief.py byte-identity verified"
                     if e4_identity_pass else
                     "not asserted; rerun after the byte-identity test and pass "
                     "--e4-identity-pass"),
        },
        "overall": bool(ece_pass and auroc_pass and errf1_pass and e4_identity_pass),
    }


def _fmt(x, nd=3):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return f"{x:.{nd}f}"


def render(results) -> str:
    L = []
    e = L.append
    e("# G2 ship gate — hybrid log-odds belief")
    e("")
    e("The lead leg — **error-detection F1 with bootstrap CIs** — plus the consolidated "
      "G2 verdict. A DETERMINISTIC recompute of production statement belief from stored "
      "reader verdicts (no new LLM spend). The calibrated arm is the configuration-"
      "specific **hybrid log-odds** model: reader evidence comes from confusion-derived "
      "likelihood ratios, while confirmations retain the separately fitted source-"
      "reliability floor (artifact key: `clean`). It is not a pure Bayesian posterior. "
      "err-F1 is measured on a "
      "BELIEF threshold (positive class = error, pred_error = belief < τ*). Each arm's τ* "
      "is selected on the training gold and frozen before the independent test; the "
      "production `verdict_statement` "
      "is tier-driven and untouched by the calibrated belief, so measuring err-F1 on it would be "
      "vacuous. E4 byte identity is independently locked by `tests/test_soft_belief.py`. "
      "Generated by `scripts/calibration_ship_gate.py`.")
    e("")
    e(f"> **G2 requires all four legs to pass, per reader.** (1) ECE(calibrated) < ECE(hard); "
      f"(2) AUROC(calibrated) ≥ AUROC(hard) − {EPS}; (3) ΔerrF1 lower-95%-bootstrap-bound ≥ "
      f"−{NI_MARGIN} (the pre-specified, historically observed medpsy-4B identical-run "
      f"err-F1 spread 0.717–0.871); (4) E4 byte-identity test green. Brier-resolution is a "
      f"reported diagnostic, not gated. Lead metric = err-F1, never accuracy.")
    e("")

    # ── consolidated verdict table ──
    e("## G2 verdict (per reader)")
    e("")
    e("| reader | n | ECE hard→cal | AUROC hard→cal | errF1 hard→cal | ΔerrF1 [95% CI] | E4 | **G2** |")
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
        join = ev.get("join") or {}
        e(f"## {r['name']} — {r['test_label']} (n={ev['n_test']}, "
          f"base {_fmt(ev['base_rate'])} correct / {_fmt(ev['error_rate'])} error, "
          f"undefined {ev['n_undefined']}, unmatched rows {join.get('n_unmatched_rows', 0)}, "
          f"ambiguous rows {join.get('n_ambiguous_rows', 0)})")
        e("")
        p = r["provenance"]
        e(f"Train: `{p['train_gold']}` + `{p['train_run']}`.  ")
        e(f"Test: `{p['test_gold']}` + `{p['test_run']}`.  ")
        e(f"Join: {join.get('join_mode', '—')}; statement grain = run `stmt_hash`; "
          "statement gold = any-incorrect-wins.")
        e("")
        e(f"**G2: {'PASS' if g['overall'] else 'FAIL'}** — "
          f"ECE {_fmt(g['ece']['hard'])}→{_fmt(g['ece']['soft'])} "
          f"({'✓' if g['ece']['pass'] else '✗'}); "
          f"AUROC {_fmt(g['auroc']['hard'])}→{_fmt(g['auroc']['soft'])} "
          f"({'✓' if g['auroc']['pass'] else '✗'}); "
          f"err-F1 {_fmt(g['errf1']['hard'])}→{_fmt(g['errf1']['soft'])}, "
          f"Δ {_fmt(g['errf1']['delta'])} [95% CI {_fmt(g['errf1']['ci_delta'][0])}, "
          f"{_fmt(g['errf1']['ci_delta'][1])}] vs −{NI_MARGIN} margin "
          f"({'✓' if g['errf1']['pass'] else '✗'}); E4 byte-identity "
          f"{'✓' if g['e4_identity']['pass'] else '✗'}.")
        e("")
        e("### Calibration + discrimination (3-arm; the ECE and AUROC legs)")
        e("")
        e("| arm | ECE | AUROC | AUPRC | Brier | resolution (diag) |")
        e("|---|---|---|---|---|---|")
        for m, mm in ev["metrics"].items():
            e(f"| {m} | {_fmt(mm['ece'])} | {_fmt(mm['auroc'])} | {_fmt(mm['auprc'])} | "
              f"{_fmt(mm['brier'])} | {_fmt(mm['resolution'])} |")
        e("")
        b = ev["errf1_boot"]
        e(f"### Error-detection F1 (LEAD leg) — frozen train-selected thresholds "
          f"(hard τ*={b['tau_hard']}, hybrid τ*={b['tau_soft']}; "
          f"training n={ev['thresholds']['n_train']}), "
          f"{N_BOOT} bootstrap resamples (seed {BOOT_SEED})")
        e("")
        e("| arm | err-F1 | 95% CI |")
        e("|---|---|---|")
        e(f"| hard | {_fmt(b['f1_hard'])} | [{_fmt(b['ci_hard'][0])}, {_fmt(b['ci_hard'][1])}] |")
        e(f"| calibrated (hybrid log-odds) | {_fmt(b['f1_soft'])} | [{_fmt(b['ci_soft'][0])}, {_fmt(b['ci_soft'][1])}] |")
        e(f"| **Δ (calibrated−hard)** | {_fmt(b['delta'])} | [{_fmt(b['ci_delta'][0])}, {_fmt(b['ci_delta'][1])}] |")
        e("")
        e(f"Non-inferiority: lower 95% bound {_fmt(b['ci_delta'][0])} "
          f"{'≥' if g['errf1']['pass'] else '<'} −{NI_MARGIN} (pre-specified margin) → "
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
    ap.add_argument("--test-gold", default="data/results/cc_holdout_cc/holdout_cc.jsonl")
    ap.add_argument("--test-run", action="append", default=[])
    ap.add_argument("--name", action="append", default=[])
    ap.add_argument("--test-label", default=None)
    ap.add_argument(
        "--e4-identity-pass", action="store_true",
        help=("assert that the byte-identity compatibility test passed in this "
              "verification run; otherwise the four-leg gate remains FAIL"),
    )
    ap.add_argument("--out", default="data/results/calibration_ship_gate.md")
    ap.add_argument("--json", default="data/results/calibration_ship_gate.json")
    args = ap.parse_args()
    if not args.train_run:
        args.train_run = ["data/results/eval_curation_v1_medpsy.jsonl",
                          "data/results/eval_curation_v1_gemma.jsonl"]
        args.test_run = ["data/results/holdout_cc_medpsy.jsonl",
                         "data/results/holdout_cc_gemma.jsonl"]
        args.name = ["MedPsy-4B", "gemma-26B"]
    if not (len(args.train_run) == len(args.test_run) == len(args.name)):
        ap.error("--train-run, --test-run, and --name must have the same number of values")
    test_label = args.test_label or Path(args.test_gold).stem

    results, pending = [], []
    for trun, terun, name in zip(args.train_run, args.test_run, args.name):
        if not (ROOT / terun).exists():
            pending.append((name, terun))
            continue
        train_configuration, test_configuration = validate_configuration_pair(trun, terun)
        # Fit the reader and select operating thresholds on train only.
        train_statements, train_join = statements_for_run(trun, args.train_gold)
        profile = c1.fit_reader_profile(train_statements)
        profile.update({
            "profile_id": production_profile_id(train_configuration, args.train_gold),
            "reader_configuration": train_configuration,
            "fit_run": trun,
            "fit_gold": args.train_gold,
            "fit_gold_sha256": file_sha256(args.train_gold),
            "fit_unique_pairs": sum(profile["confusion"].values()),
            "gold_rule": (
                "exact pair; multi-curator any-incorrect-wins; duplicate pairs removed"
            ),
        })
        thresholds = training_thresholds(train_statements, profile)
        test_statements, test_join = statements_for_run(terun, args.test_gold)
        ev = eval_reader(test_statements, profile, thresholds, test_join)
        results.append({
            "name": name,
            "test_label": test_label,
            "provenance": {
                "train_gold": args.train_gold,
                "train_gold_sha256": file_sha256(args.train_gold),
                "train_run": trun,
                "train_run_sha256": file_sha256(trun),
                "train_configuration": train_configuration,
                "test_gold": args.test_gold,
                "test_gold_sha256": file_sha256(args.test_gold),
                "test_run": terun,
                "test_run_sha256": file_sha256(terun),
                "test_configuration": test_configuration,
                "statement_grain": "run stmt_hash",
                "statement_gold": "any-incorrect-wins",
                "train_join": train_join,
                "test_join": test_join,
            },
            "reader_profile": profile,
            "eval": ev,
            "gate": gate(ev, e4_identity_pass=args.e4_identity_pass),
        })

    md = render(results)
    (ROOT / args.out).write_text(md)
    (ROOT / args.json).write_text(json.dumps(results, indent=2, default=float))
    print(md)
    print(f"Wrote {args.out} and {args.json}")

    # Exit-code contract: always report (above), then fail the process if any
    # requested test run is PENDING (evidence missing) or any evaluated reader's
    # gate.overall is False. Exit 0 only when every evaluated reader PASSes and
    # nothing is pending — a gate that exits success on a FAIL cannot stop a ship.
    passed = [r["name"] for r in results if r["gate"]["overall"]]
    failed = [r["name"] for r in results if not r["gate"]["overall"]]
    if pending:
        print("PENDING — test run not scored yet: "
              + ", ".join(f"{name} ({run})" for name, run in pending))
    print(
        "SHIP GATE — "
        f"PASS: {', '.join(passed) or '—'}; "
        f"FAIL: {', '.join(failed) or '—'}; "
        f"PENDING: {', '.join(name for name, _ in pending) or '—'}"
    )
    return gate_exit_code(results, pending)


def gate_exit_code(results: list[dict], pending: list[tuple[str, str]]) -> int:
    """Process contract: success only when every requested reader passes."""
    return 0 if (not pending and all(r["gate"]["overall"] for r in results)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
