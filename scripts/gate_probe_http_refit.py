#!/usr/bin/env python3
"""Refit the BASE1 probe combiner on HTTP-derived features, and gate it.

The published +0.045872 was measured with the probe read IN-PROCESS. That path
cannot ship (see scripts/refit_probe_over_http.py). The transport check showed
HTTP preserves RANK (AUROC 0.7036 vs 0.7121, CI spanning zero) but not
MAGNITUDE (pearson 0.935, sign agreement 79.4%, in-process ~2.3x wider), so the
frozen combiner's coefficients do not transfer and the model must be refitted on
the features the transport actually produces.

This reproduces the published ARM C — probe ADDED to the incumbent verdict —
against the same incumbent substrate the original used:

    incumbent = gemma-remote (the 29.30 s/record reasoning arm)
      fit  split: data/results/eval_curation_v1_gemma.jsonl
      test split: data/results/holdout_cc_gemma.jsonl

Both splits share that reader, so the comparison is not mixing substrates. Note
this is NOT the production configuration — prod would pair the probe with the
newly calibrated local-gemma-4-26b — but no local run exists on holdout_cc, and
reproducing the published number matters more here than pre-empting it.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from indra_belief.probe_combiner import fit_combiner  # noqa: E402
from indra_belief.verdict import grid_score  # noqa: E402
from refit_probe_over_http import auroc, clustered_ci  # noqa: E402

PROBE_DIR = ROOT / "data" / "probe_battery"
SCORES = PROBE_DIR / "http_base1_scores.json"
OUT = PROBE_DIR / "http_base1_gate.json"
INCUMBENT = {
    "fit": ROOT / "data/results/eval_curation_v1_gemma.jsonl",
    "test": ROOT / "data/results/holdout_cc_gemma.jsonl",
}
EPS = 1e-6


def incumbent_by_source_hash(path: Path) -> dict[str, float]:
    """source_hash -> the deployed six-cell grid score for that evidence."""
    out: dict[str, float] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        sh = r.get("source_hash")
        if sh is None:
            continue
        s = r.get("our_score")
        if s is None:
            s = grid_score(r.get("verdict"), r.get("confidence"))
        if s is None:
            continue
        out.setdefault(str(sh), float(s))
    return out


def logit(p: float) -> float:
    p = min(1 - EPS, max(EPS, p))
    return math.log(p / (1 - p))


def assemble(rows: list[dict], inc: dict[str, float]):
    X, y, ref, clusters = [], [], [], []
    for r in rows:
        sh = r["source_hash"]
        if sh not in inc:
            continue
        X.append([r["http_delta_logit"], logit(inc[sh])])
        y.append(bool(r["gold_correct"]))
        ref.append(inc[sh])
        clusters.append(sh)
    return np.array(X, float), np.array(y, bool), np.array(ref, float), clusters


def main() -> int:
    data = json.loads(SCORES.read_text())
    if "fit_rows" not in data:
        print("http_base1_scores.json has no fit_rows — rerun with --full")
        return 2

    inc_fit = incumbent_by_source_hash(INCUMBENT["fit"])
    inc_test = incumbent_by_source_hash(INCUMBENT["test"])
    print(f"  incumbent coverage: fit {len(inc_fit)} source_hashes, test {len(inc_test)}")

    Xf, yf, _, cf = assemble(data["fit_rows"], inc_fit)
    Xt, yt, inc_t, ct = assemble(data["test_rows"], inc_test)
    print(f"  usable: fit {len(yf)} rows, test {len(yt)} rows")
    if len(yf) < 100 or len(yt) < 100:
        print("  too few joined rows to fit or gate honestly")
        return 2

    fit_ids = [f"f{i}" for i in range(len(yf))]
    test_ids = [f"t{i}" for i in range(len(yt))]
    combiner = fit_combiner(Xf, yf, probe_ids=("base1", "incumbent"),
                            record_ids=fit_ids, seed=0)
    cand = np.asarray(combiner.score(Xt, record_ids=test_ids,
                                     probe_ids=("base1", "incumbent")))

    a_c, a_i = auroc(cand, yt), auroc(inc_t, yt)
    d, lo, hi = clustered_ci(cand, inc_t, yt, ct)
    published = 0.045872
    print("\n=== Q2: refitted on HTTP features, does the gain survive? ===")
    print(f"  AUROC incumbent alone        : {a_i:.4f}")
    print(f"  AUROC incumbent + HTTP probe : {a_c:.4f}")
    print(f"  delta                        : {d:+.4f}   95% CI [{lo:+.4f}, {hi:+.4f}]")
    print(f"  published (in-process BASE1) : {published:+.6f}  CI [+0.020968, +0.069776]")
    passed = lo > 0
    print(f"  GATE (ci95_low > 0)          : {'PASS' if passed else 'FAIL'}")

    OUT.write_text(json.dumps({
        "kind": "probe_http_refit_gate",
        "probe_id": "pol.verdict_direct",
        "incumbent": {k: str(v) for k, v in INCUMBENT.items()},
        "n_fit": int(len(yf)), "n_test": int(len(yt)),
        "auroc_incumbent": round(a_i, 6),
        "auroc_candidate": round(a_c, 6),
        "delta_auroc": round(d, 6),
        "delta_auroc_ci95": [round(lo, 6), round(hi, 6)],
        "published_inprocess_delta": published,
        "gate": {"rule": "held-out paired ci95_low(delta AUROC) > 0", "passed": bool(passed)},
    }, indent=1) + "\n")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
