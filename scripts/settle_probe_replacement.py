#!/usr/bin/env python3
"""Settle it: does a probe-only score SUPERSEDE the six-cell verdict grid?

THE QUESTION, AND WHY IT IS STILL OPEN
--------------------------------------
The deployed per-evidence score is a lookup on (verdict, confidence) -> six
values. On the battery's holdout it emitted THREE distinct scores across 500
statements, ECE 0.2137. Replacing it with a probe-only score was measured once
and recorded NO-GO:

    delta AUROC -0.0019, 95% CI [-0.0490, +0.0444]

That interval is ±0.047 wide. It cannot distinguish "equal" from "2 points
worse", so the honest reading was never "no" — it was "underpowered". Meanwhile
the same arm was 2.7x better calibrated (ECE 0.0802), 4.1x cheaper, and +0.041
in average precision. A wash on ranking that wins everywhere else is a real
candidate, and it deserves a test with enough power to resolve it.

THE DESIGN
----------
Test on the LARGER split, because test n is what buys resolution:

    fit  = B3, external_curator_gold_v1 scored by local-gemma-4-26b   (587)
    test = B2, eval_curation_v1        scored by local-gemma-4-26b   (1606)

At n=1606 the expected CI half-width is ~0.013, which clears a -0.02
non-inferiority margin. At n=500 it was ~0.023, which does not.

Both splits are the SAME reader (local-gemma-4-26b, production prompt) and the
probe runs over the SAME HTTP transport, so nothing here mixes substrates. The
incumbent is that reader's own grid score, so this compares two readings of
identical evidence rather than two different models.

GATE — stated before the run, and not moved afterwards:

    NON-INFERIOR ranking   ci95_low(delta AUROC) > -0.02
    AND better calibration ECE strictly lower
    AND better resolution  more distinct scores

Resumable: probe scores are checkpointed per record, so an interrupted run
continues rather than restarting three hours of inference.

Usage:
    python scripts/settle_probe_replacement.py            # score, then gate
    python scripts/settle_probe_replacement.py --gate-only
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_probe_battery as rpb  # noqa: E402
from indra_belief.metrics import BINS_8  # noqa: E402
from indra_belief.model_client import LOCAL_MODELS  # noqa: E402
from indra_belief.probe_combiner import fit_combiner  # noqa: E402
from indra_belief.probes.battery import probe_by_id, render  # noqa: E402
from indra_belief.curation import is_gold_correct  # noqa: E402
from refit_probe_over_http import auroc, clustered_ci  # noqa: E402

PROBE_DIR = ROOT / "data" / "probe_battery"
CKPT = PROBE_DIR / "settle_probe_scores.jsonl"
OUT = PROBE_DIR / "settle_probe_replacement.json"
NI_MARGIN = 0.02

SPLITS = {
    "fit": (ROOT / "data/results/external_curator_v1_local-gemma-4-26b.jsonl",
            ROOT / "data/benchmark/external_curator_gold_v1.jsonl"),
    "test": (ROOT / "data/results/eval_curation_v1_local-gemma-4-26b.jsonl",
             ROOT / "data/benchmark/eval_curation_v1.jsonl"),
}
PROBE_IDS = tuple(json.loads((PROBE_DIR / "probes_test.jsonl").read_text().splitlines()[0])["probe_ids"])


def endpoint():
    cfg = LOCAL_MODELS["local-gemma-4-26b"]
    return cfg["base_url"].rstrip("/") + "/chat/completions", cfg["model_id"], int(cfg["max_top_logprobs"])


def records(split: str):
    """Run rows joined to their gold, carrying evidence_text, incumbent score and label.

    The run row has no evidence_text (the probe needs it to render), and the gold
    has no incumbent score (the run does). Neither alone is enough.
    """
    run_path, gold_path = SPLITS[split]
    # Read the gold directly rather than through rpb.load_gold: that loader
    # demands a pa_hash which external_curator_gold_v1 does not carry, and this
    # comparison clusters on source_hash anyway. The LABEL still comes from the
    # canonical atom, indra_belief.curation.is_gold_correct.
    gmap: dict[str, dict] = {}
    for line in gold_path.read_text().splitlines():
        if not line.strip():
            continue
        g = json.loads(line)
        sh = g.get("source_hash")
        if sh is None:
            continue
        g["gold_correct"] = is_gold_correct(g.get("tag"))
        gmap.setdefault(str(sh), g)
    out = []
    for line in run_path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        sh = str(r.get("source_hash"))
        g = gmap.get(sh)
        if not g or not g.get("evidence_text"):
            continue
        if g.get("subject") != r.get("subject") or g.get("stmt_type") != r.get("stmt_type"):
            continue
        inc = r.get("our_score")
        label = g.get("gold_correct")
        if inc is None or label is None:
            continue
        out.append({
            "split": split, "source_hash": sh,
            "row": {k: g.get(k) for k in ("evidence_text", "subject", "object", "stmt_type")},
            "incumbent": float(inc), "gold_correct": bool(label),
        })
    # one record per source_hash keeps the bootstrap's cluster unit meaningful
    seen, uniq = set(), []
    for r in out:
        if r["source_hash"] in seen:
            continue
        seen.add(r["source_hash"]); uniq.append(r)
    return uniq


def probe_vector(url, model_id, k, row) -> dict[str, float] | None:
    """All 16 probe delta_logits for one record. None if any probe loses a label."""
    vec = {}
    for pid in PROBE_IDS:
        system, user, prefill = render(probe_by_id(pid), row)
        body = {"model": model_id, "max_tokens": 1, "temperature": 0.0, "logprobs": True,
                "top_logprobs": k,
                "messages": [{"role": "system", "content": system},
                             {"role": "user", "content": user},
                             {"role": "assistant", "content": prefill}],
                "chat_template_kwargs": {"enable_thinking": False},
                "continue_final_message": True, "add_generation_prompt": False}
        req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        payload = json.loads(urllib.request.urlopen(req, timeout=300).read())
        content = (payload["choices"][0].get("logprobs") or {}).get("content") or [{}]
        lp = {t["token"]: t["logprob"] for t in (content[0].get("top_logprobs") or [])}
        if "correct" not in lp or "incorrect" not in lp:
            return None
        vec[pid] = lp["correct"] - lp["incorrect"]
    return vec


def score_all():
    url, model_id, k = endpoint()
    done = set()
    if CKPT.exists():
        for line in CKPT.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done.add((r["split"], r["source_hash"]))
        print(f"  resuming: {len(done)} records already scored")
    todo = [r for s in ("fit", "test") for r in records(s) if (s, r["source_hash"]) not in done]
    total = len(todo)
    print(f"  to score: {total} records x {len(PROBE_IDS)} probes "
          f"= {total*len(PROBE_IDS)} calls, ~{total*len(PROBE_IDS)*0.33/3600:.1f} h", flush=True)
    t0 = time.time()
    with CKPT.open("a") as fh:
        for n, rec in enumerate(todo, 1):
            vec = probe_vector(url, model_id, k, rec["row"])
            if vec is None:
                continue
            fh.write(json.dumps({"split": rec["split"], "source_hash": rec["source_hash"],
                                 "incumbent": rec["incumbent"], "gold_correct": rec["gold_correct"],
                                 "probes": vec}) + "\n")
            fh.flush()
            if n % 25 == 0:
                el = time.time() - t0
                print(f"    {n}/{total}  {el/n:.2f} s/record  eta {(total-n)*el/n/3600:.2f} h", flush=True)


def ece_bins8(scores, labels):
    n = len(scores); e = 0.0
    for lo, hi in BINS_8:
        m = (scores >= lo) & (scores < hi)
        if not m.sum():
            continue
        e += (m.sum() / n) * abs(labels[m].mean() - scores[m].mean())
    return e


def gate():
    rows = [json.loads(l) for l in CKPT.read_text().splitlines() if l.strip()]
    fit = [r for r in rows if r["split"] == "fit"]
    test = [r for r in rows if r["split"] == "test"]
    print(f"  fit {len(fit)}  test {len(test)}")
    if len(fit) < 200 or len(test) < 800:
        print("  not enough scored records yet to settle anything")
        return 2
    X = lambda rs: np.array([[r["probes"][p] for p in PROBE_IDS] for r in rs], float)
    y = lambda rs: np.array([r["gold_correct"] for r in rs], bool)
    comb = fit_combiner(X(fit), y(fit), probe_ids=PROBE_IDS,
                        record_ids=[f"f{r['source_hash']}" for r in fit], seed=0)
    cand = np.asarray(comb.score(X(test), record_ids=[f"t{r['source_hash']}" for r in test],
                                 probe_ids=PROBE_IDS))
    inc = np.array([r["incumbent"] for r in test], float)
    yt = y(test)
    clusters = [r["source_hash"] for r in test]

    d, lo, hi = clustered_ci(cand, inc, yt, clusters)
    e_c, e_i = ece_bins8(cand, yt), ece_bins8(inc, yt)
    n_c, n_i = len(np.unique(np.round(cand, 6))), len(np.unique(np.round(inc, 6)))
    ni = lo > -NI_MARGIN
    verdict = "GO" if (ni and e_c < e_i and n_c > n_i) else "NO-GO"

    print(f"\n=== REPLACEMENT: probe-only vs the six-cell grid, n_test={len(test)} ===")
    print(f"  AUROC   grid {auroc(inc,yt):.4f}   probe {auroc(cand,yt):.4f}")
    print(f"  delta   {d:+.4f}   95% CI [{lo:+.4f}, {hi:+.4f}]   (margin -{NI_MARGIN})")
    print(f"  ECE     grid {e_i:.4f}   probe {e_c:.4f}")
    print(f"  scores  grid {n_i}        probe {n_c}")
    print(f"  non-inferior {ni} | better ECE {e_c<e_i} | better resolution {n_c>n_i}")
    print(f"  GATE: {verdict}")
    OUT.write_text(json.dumps({
        "kind": "probe_replacement_settlement", "n_fit": len(fit), "n_test": len(test),
        "auroc_incumbent": round(auroc(inc, yt), 6), "auroc_probe": round(auroc(cand, yt), 6),
        "delta_auroc": round(d, 6), "delta_auroc_ci95": [round(lo, 6), round(hi, 6)],
        "ece_incumbent": round(e_i, 6), "ece_probe": round(e_c, 6),
        "distinct_incumbent": n_i, "distinct_probe": n_c,
        "gate": {"rule": f"ci95_low > -{NI_MARGIN} AND lower ECE AND more distinct scores",
                 "non_inferior": bool(ni), "verdict": verdict},
    }, indent=1) + "\n")
    print(f"  wrote {OUT}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate-only", action="store_true")
    args = ap.parse_args()
    if not args.gate_only:
        score_all()
    return gate()


if __name__ == "__main__":
    raise SystemExit(main())
