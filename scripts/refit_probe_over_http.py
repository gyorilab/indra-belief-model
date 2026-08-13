#!/usr/bin/env python3
"""Re-derive the BASE1 probe over HTTP, and test whether the transport preserves it.

WHY
---
The probe battery read label logprobs IN-PROCESS via `mlx_lm.generate_step`,
which returns the full vocabulary, so it could index `logprobs[19448]` and
`logprobs[111863]` directly. That path cannot ship: `mlx_lm` is not importable
from the project venv, it loads its own 25 GB copy of the model, and it is
Apple-Silicon only — so it can never run on the vLLM/H200 substrate prod plans.

The same read IS possible over the ordinary OpenAI-compatible transport, given
two things:
  * `chat_template_kwargs={"enable_thinking": False}` to suppress the thought
    channel, plus an assistant prefill, so the next token is the verdict;
  * a `top_logprobs` cap high enough to contain BOTH label tokens. Stock
    mlx_lm.server hard-codes 11 and the losing label was measured at rank
    42/83/168 — see scripts/serve_mlx.sh for the local patch.

But a 40-record spot check showed the two paths are NOT interchangeable:
Pearson r 0.955 on `delta_logit`, yet HTTP compresses the range ~2.4x
(in-process mean +11.37, HTTP +2.54) and the sign disagrees on 10%. A combiner
fitted on in-process magnitudes cannot be fed HTTP magnitudes.

So this script re-derives the feature over HTTP and asks two questions:
  Q1 (transport): does HTTP `delta_logit` RANK as well as in-process on the same
     held-out records? Rank is what a combiner's discrimination rests on.
  Q2 (refit):     fitted on HTTP features, does the probe keep its measured gain?

Usage:
    python scripts/refit_probe_over_http.py            # test split only (Q1)
    python scripts/refit_probe_over_http.py --full     # both splits (Q1 + Q2)
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
from indra_belief.model_client import LOCAL_MODELS  # noqa: E402
from indra_belief.probes.battery import probe_by_id, render  # noqa: E402

PROBE_ID = "pol.verdict_direct"  # BASE1: the pre-registered anchor
PROBE_DIR = ROOT / "data" / "probe_battery"
OUT = PROBE_DIR / "http_base1_scores.json"


def endpoint() -> tuple[str, str, int]:
    cfg = LOCAL_MODELS["local-gemma-4-26b"]
    return (
        cfg["base_url"].rstrip("/") + "/chat/completions",
        cfg["model_id"],
        int(cfg["max_top_logprobs"]),
    )


def http_delta_logit(url: str, model_id: str, k: int, row: dict) -> float | None:
    """The probe's feature, read over the ordinary transport.

    Returns None when either label falls outside the top-k window — that is a
    real failure of the transport for this record, not a zero, and it must not
    be imputed.
    """
    probe = probe_by_id(PROBE_ID)
    system, user, prefill = render(probe, row)
    body = {
        "model": model_id,
        "max_tokens": 1,
        "temperature": 0.0,
        "logprobs": True,
        "top_logprobs": k,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": prefill},
        ],
        "chat_template_kwargs": {"enable_thinking": False},
        "continue_final_message": True,
        "add_generation_prompt": False,
    }
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    payload = json.loads(urllib.request.urlopen(req, timeout=300).read())
    content = (payload["choices"][0].get("logprobs") or {}).get("content") or [{}]
    top = content[0].get("top_logprobs") or []
    lp = {t["token"]: t["logprob"] for t in top}
    if "correct" not in lp or "incorrect" not in lp:
        return None
    return lp["correct"] - lp["incorrect"]


def joined(split_path: Path) -> list[tuple[dict, dict]]:
    """Probe rows paired with the gold row carrying their evidence text.

    The artifact stores no evidence_text, so it is rejoined from the gold the
    manifest names. Ambiguous joins are dropped rather than guessed: a row whose
    subject or statement type disagrees is not the same record.
    """
    rows = [json.loads(l) for l in split_path.read_text().splitlines() if l.strip()]
    manifest, records = rows[0], rows[1:]
    gold, _ = rpb.load_gold(manifest["gold_path"])
    gmap: dict[str, dict] = {}
    for g in gold:
        gmap.setdefault(str(g.get("source_hash")), g)
    out = []
    for r in records:
        g = gmap.get(str(r.get("source_hash")))
        if not g:
            continue
        if g.get("subject") != r.get("subject") or g.get("stmt_type") != r.get("stmt_type"):
            continue
        out.append((r, g))
    return out


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    s = scores[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    pos = labels.astype(bool)
    npos, nneg = int(pos.sum()), int((~pos).sum())
    if not npos or not nneg:
        return float("nan")
    return (ranks[pos].sum() - npos * (npos + 1) / 2.0) / (npos * nneg)


def clustered_ci(a, b, labels, clusters, seed=0, n_boot=2000):
    """Paired bootstrap over clusters of the AUROC difference a - b."""
    rng = np.random.default_rng(seed)
    by: dict[str, list[int]] = {}
    for i, c in enumerate(clusters):
        by.setdefault(c, []).append(i)
    keys = list(by)
    deltas = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(keys), size=len(keys))
        idx = np.asarray([i for p in pick for i in by[keys[p]]])
        lab = labels[idx]
        if lab.sum() in (0, len(lab)):
            continue
        deltas.append(auroc(a[idx], lab) - auroc(b[idx], lab))
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return auroc(a, labels) - auroc(b, labels), float(lo), float(hi)


def score_split(name: str, path: Path, limit: int | None) -> list[dict]:
    url, model_id, k = endpoint()
    pairs = joined(path)
    if limit:
        pairs = pairs[:limit]
    print(f"[{name}] {len(pairs)} cleanly joined records, top_logprobs={k}", flush=True)
    out, t0, dropped = [], time.time(), 0
    for n, (probe_row, gold_row) in enumerate(pairs, 1):
        d = http_delta_logit(url, model_id, k, gold_row)
        if d is None:
            dropped += 1
            continue
        ref = (probe_row.get("probes") or {}).get(PROBE_ID) or {}
        out.append(
            {
                "source_hash": str(probe_row.get("source_hash")),
                "gold_correct": bool(probe_row.get("gold_correct")),
                "http_delta_logit": d,
                "inprocess_delta_logit": ref.get("delta_logit"),
            }
        )
        if n % 50 == 0:
            el = time.time() - t0
            print(f"[{name}]   {n}/{len(pairs)}  {el/n:.2f} s/record  dropped {dropped}", flush=True)
    el = time.time() - t0
    print(f"[{name}] done: {len(out)} scored, {dropped} dropped (label outside top-k), "
          f"{el/max(len(out),1):.2f} s/record", flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="also score the fit split and refit")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    test = score_split("test", PROBE_DIR / "probes_test.jsonl", args.limit)
    both = [r for r in test if r["inprocess_delta_logit"] is not None]
    y = np.array([r["gold_correct"] for r in both])
    http = np.array([r["http_delta_logit"] for r in both])
    inproc = np.array([r["inprocess_delta_logit"] for r in both])
    clusters = [r["source_hash"] for r in both]

    print("\n=== Q1: does the TRANSPORT preserve the signal? ===")
    print(f"  n = {len(both)}   base rate {y.mean():.4f}")
    a_http, a_in = auroc(http, y), auroc(inproc, y)
    print(f"  AUROC, in-process delta_logit : {a_in:.4f}")
    print(f"  AUROC, HTTP       delta_logit : {a_http:.4f}")
    d, lo, hi = clustered_ci(http, inproc, y, clusters)
    print(f"  delta (HTTP - in-process)     : {d:+.4f}   95% CI [{lo:+.4f}, {hi:+.4f}]")
    r = float(np.corrcoef(http, inproc)[0, 1])
    sign = float(np.mean((http > 0) == (inproc > 0)))
    print(f"  pearson r {r:.4f}   sign agreement {sign:.1%}")
    print(f"  scale: in-process mean {inproc.mean():+.3f}, HTTP mean {http.mean():+.3f} "
          f"({inproc.std()/max(http.std(),1e-9):.2f}x wider)")

    result = {
        "kind": "probe_http_transport_check",
        "probe_id": PROBE_ID,
        "n": len(both),
        "auroc_inprocess": round(a_in, 6),
        "auroc_http": round(a_http, 6),
        "delta_auroc": round(d, 6),
        "delta_auroc_ci95": [round(lo, 6), round(hi, 6)],
        "pearson_r": round(r, 6),
        "sign_agreement": round(sign, 6),
        "test_rows": test,
    }
    if args.full:
        fit = score_split("fit", PROBE_DIR / "probes_fit.jsonl", args.limit)
        result["fit_rows"] = fit
    OUT.write_text(json.dumps(result, indent=1) + "\n")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
