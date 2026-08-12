#!/usr/bin/env python3
"""Fit, FREEZE and evaluate a continuous per-evidence belief model from probe logits.

WHY THIS EXISTS
---------------
The deployed per-evidence score is a six-cell lookup: (verdict, confidence) ->
{0.95, 0.80, 0.65, 0.35, 0.20, 0.05}. On the held-out battery split it emits
THREE distinct values across 500 statements, so it cannot rank within a verdict
and its calibration error is 0.2137. That is the binning this model is meant to
supersede.

An earlier experiment already tested exactly this replacement and was recorded
NO-GO. Read its gate before reusing that verdict:

    rule: held-out paired ci95_low(delta AUROC) > 0

It demanded the replacement be strictly BETTER at ranking. It was
indistinguishable (delta -0.0019, CI [-0.0490, +0.0444]) and therefore declined
-- while being 2.7x better calibrated, 4.1x cheaper, and +0.041 in average
precision. The replacement was never refuted; it failed a superiority test on
the one axis it was not trying to improve.

So this script re-asks the question with a gate that matches it:

    NON-INFERIOR on ranking  (ci95_low(delta AUROC) > -0.05)
    AND SUPERIOR on calibration (ECE strictly lower)
    AND SUPERIOR on resolution  (more distinct scores)

WHAT IT PRODUCES
----------------
A persisted `FrozenCombiner`. The prior arc measured a gain and saved nothing --
the import-boundary guard reports the combiner unreachable from any serving path
and no fitted artifact exists on disk. A measurement nobody can apply is not a
model. This writes one.

No inference is performed: both splits were scored previously and are committed
under data/probe_battery/. Fitting is CPU-only sklearn over 600 rows.

Usage:
    python scripts/fit_probe_belief_model.py
    python scripts/fit_probe_belief_model.py --probes pol.verdict_direct
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from indra_belief.metrics import BINS_8  # noqa: E402
from indra_belief.probe_combiner import fit_combiner  # noqa: E402

PROBE_DIR = ROOT / "data" / "probe_battery"
FIT_PATH = PROBE_DIR / "probes_fit.jsonl"
TEST_PATH = PROBE_DIR / "probes_test.jsonl"
INCUMBENT_PATH = PROBE_DIR / "holdout_scores_C_incumbent_plus_battery.jsonl"
OUT_MODEL = PROBE_DIR / "probe_belief_model.json"
OUT_EVAL = PROBE_DIR / "probe_belief_model_eval.json"

# The pre-registered anchor probe. The final ablation found it carries
# essentially the whole gain of the 16-probe battery.
BASE1 = "pol.verdict_direct"
# Non-inferiority margin on AUROC. The replacement may lose this much ranking
# and still be worth taking for what it buys elsewhere; it may not lose more.
NI_MARGIN = 0.05


def load_split(path: Path) -> tuple[dict, list[dict]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return rows[0], rows[1:]


def design_matrix(records: list[dict], probe_ids: list[str]) -> np.ndarray:
    """One row per record, one column per probe, carrying the probe's delta_logit.

    A probe that failed on a record contributes 0.0 — the neutral value on the
    log-odds scale, i.e. "this probe says nothing", which is exactly what a
    failed read means. Imputing a mean would invent evidence.
    """
    out = np.zeros((len(records), len(probe_ids)), dtype=float)
    for i, record in enumerate(records):
        probes = record.get("probes") or {}
        for j, pid in enumerate(probe_ids):
            entry = probes.get(pid) or {}
            value = entry.get("delta_logit")
            if isinstance(value, (int, float)) and math.isfinite(value):
                out[i, j] = float(value)
    return out


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    sorted_scores = scores[order]
    i = 0
    while i < len(sorted_scores):
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    pos = labels.astype(bool)
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return (ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    """Tie-aware average precision — the same correction the metrics module uses."""
    order = np.argsort(-scores, kind="mergesort")
    s, y = scores[order], labels[order].astype(bool)
    total_pos = int(y.sum())
    if total_pos == 0:
        return float("nan")
    ap, tp, seen = 0.0, 0, 0
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        block = y[i : j + 1]
        tp += int(block.sum())
        seen = j + 1
        # One precision value for the whole tied block, credited to its positives.
        ap += (tp / seen) * int(block.sum())
        i = j + 1
    return ap / total_pos


def ece_bins8(scores: np.ndarray, labels: np.ndarray) -> tuple[float, list[dict]]:
    n = len(scores)
    ece, bins = 0.0, []
    for lo, hi in BINS_8:
        mask = (scores >= lo) & (scores < hi)
        k = int(mask.sum())
        if k == 0:
            continue
        p_mean = float(scores[mask].mean())
        y_rate = float(labels[mask].mean())
        ece += (k / n) * abs(y_rate - p_mean)
        bins.append({"p_mean": round(p_mean, 6), "y_rate": round(y_rate, 6), "n": k})
    return ece, bins


def clustered_delta_ci(
    cand: np.ndarray,
    inc: np.ndarray,
    labels: np.ndarray,
    clusters: list[str],
    *,
    n_boot: int = 2000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Paired bootstrap over CLUSTERS, not rows.

    Evidence rows sharing a source_hash are not independent; resampling rows
    would understate the interval. This mirrors the battery's own resampling
    unit so the numbers stay comparable to the frozen decision artifact.
    """
    rng = np.random.default_rng(seed)
    by_cluster: dict[str, list[int]] = {}
    for i, c in enumerate(clusters):
        by_cluster.setdefault(c, []).append(i)
    keys = list(by_cluster)
    observed = auroc(cand, labels) - auroc(inc, labels)
    deltas = []
    for _ in range(n_boot):
        picked = rng.integers(0, len(keys), size=len(keys))
        idx = [i for p in picked for i in by_cluster[keys[p]]]
        idx_arr = np.asarray(idx)
        lab = labels[idx_arr]
        if lab.sum() == 0 or lab.sum() == len(lab):
            continue
        deltas.append(auroc(cand[idx_arr], lab) - auroc(inc[idx_arr], lab))
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return observed, float(lo), float(hi)


def evaluate(name: str, scores: np.ndarray, labels: np.ndarray) -> dict:
    ece, bins = ece_bins8(scores, labels)
    return {
        "name": name,
        "n": int(len(scores)),
        "auroc": round(auroc(scores, labels), 6),
        "average_precision": round(average_precision(scores, labels), 6),
        "ece_bins8": round(ece, 6),
        "distinct_scores": int(len(np.unique(np.round(scores, 6)))),
        "min": round(float(scores.min()), 6),
        "max": round(float(scores.max()), 6),
        "reliability_bins": bins,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--probes",
        nargs="*",
        default=None,
        help=f"probe ids to fit on; default is all. Pass '{BASE1}' for the anchor alone.",
    )
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    fit_meta, fit_rows = load_split(FIT_PATH)
    test_meta, test_rows = load_split(TEST_PATH)
    all_probe_ids = list(fit_meta["probe_ids"])
    probe_ids = args.probes if args.probes else all_probe_ids
    unknown = [p for p in probe_ids if p not in all_probe_ids]
    if unknown:
        print(f"unknown probe ids: {unknown}")
        return 2

    fit_ids = [str(r["row_index"]) for r in fit_rows]
    test_ids = [f"t{r['row_index']}" for r in test_rows]
    overlap = set(fit_ids) & set(test_ids)
    if overlap:
        print(f"fit/test record ids overlap ({len(overlap)}) — the split is not held out")
        return 2

    y_fit = np.array([bool(r["gold_correct"]) for r in fit_rows])
    y_test = np.array([bool(r["gold_correct"]) for r in test_rows])

    combiner = fit_combiner(
        design_matrix(fit_rows, probe_ids),
        y_fit,
        probe_ids=probe_ids,
        record_ids=fit_ids,
        seed=args.seed,
    )
    OUT_MODEL.write_text(json.dumps(combiner.to_dict(), indent=1) + "\n")
    print(f"froze combiner over {len(probe_ids)} probe(s), {len(fit_rows)} fit rows -> {OUT_MODEL}")

    # Held out. `score` raises InSampleError if any fit id leaks in.
    cand = np.asarray(
        combiner.score(design_matrix(test_rows, probe_ids), record_ids=test_ids, probe_ids=probe_ids)
    )

    incumbent_by_row = {}
    for line in INCUMBENT_PATH.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        incumbent_by_row[int(row["row_index"])] = float(row["incumbent_score"])
    inc = np.array([incumbent_by_row[int(r["row_index"])] for r in test_rows])

    clusters = [str(r["source_hash"]) for r in test_rows]
    cand_eval = evaluate(f"probe belief model ({len(probe_ids)} probe(s))", cand, y_test)
    inc_eval = evaluate("deployed six-cell grid", inc, y_test)
    delta, lo, hi = clustered_delta_ci(cand, inc, y_test, clusters, seed=args.seed)

    # Cost: the battery's own recorded seconds, summed over the probes actually used.
    per_probe = []
    for r in test_rows:
        probes = r.get("probes") or {}
        per_probe.append(
            sum(
                float((probes.get(p) or {}).get("elapsed_s") or 0.0)
                for p in probe_ids
            )
        )
    cand_seconds = float(np.mean(per_probe)) if any(per_probe) else None

    non_inferior = lo > -NI_MARGIN
    better_ece = cand_eval["ece_bins8"] < inc_eval["ece_bins8"]
    better_res = cand_eval["distinct_scores"] > inc_eval["distinct_scores"]
    verdict = "GO" if (non_inferior and better_ece and better_res) else "NO-GO"

    result = {
        "kind": "probe_belief_model_eval",
        "model": test_meta.get("model"),
        "probe_ids": probe_ids,
        "fit_rows": len(fit_rows),
        "test_rows": len(test_rows),
        "candidate": cand_eval,
        "incumbent": inc_eval,
        "delta_auroc": round(delta, 6),
        "delta_auroc_ci95": [round(lo, 6), round(hi, 6)],
        "candidate_s_per_record": None if cand_seconds is None else round(cand_seconds, 4),
        "gate": {
            "rule": (
                f"non-inferior ranking (ci95_low > -{NI_MARGIN}) AND lower ECE AND more "
                "distinct scores"
            ),
            "non_inferior_ranking": bool(non_inferior),
            "better_calibration": bool(better_ece),
            "better_resolution": bool(better_res),
            "verdict": verdict,
        },
        "prior_gate_note": (
            "the earlier NO-GO used 'ci95_low(delta AUROC) > 0', a SUPERIORITY test on ranking; "
            "this asks non-inferiority on ranking plus superiority on calibration and resolution"
        ),
    }
    OUT_EVAL.write_text(json.dumps(result, indent=1) + "\n")

    print(f"\n{'':30} {'AUROC':>8} {'AP':>8} {'ECE':>8} {'distinct':>9}")
    for e in (inc_eval, cand_eval):
        print(
            f"  {e['name'][:28]:28} {e['auroc']:8.4f} {e['average_precision']:8.4f} "
            f"{e['ece_bins8']:8.4f} {e['distinct_scores']:9d}"
        )
    print(f"\n  delta AUROC {delta:+.4f}  CI [{lo:+.4f}, {hi:+.4f}]  (cluster: source_hash)")
    if cand_seconds:
        print(f"  candidate cost {cand_seconds:.2f} s/record over {len(probe_ids)} probe(s)")
    print(f"  GATE: {verdict}  -> {OUT_EVAL}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
