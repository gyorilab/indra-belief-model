#!/usr/bin/env python3
"""Calibration-card data for Figure 6: per score-cell metadata + example rows.

For each of the 7 score values (verdict x confidence -> score) emit n, pct, the
(verdict, confidence) it derives from, and a diverse sample of example statements
(claim, source, pmid, verdict). The rare non-high cells are the interesting ones —
the handful of rows where the model expressed anything other than 'high' confidence.
Validates per-cell counts against the rendered SVG (data-count).
"""
import json, re, sys
from collections import defaultdict, Counter
sys.path.insert(0, "scripts")
from fig1_drilldown_data import corpus_evidence_text

SRC = "data/results/rasmachine_mono_gemma_remote_direct.jsonl"
HTML = "reports/rasmachine_belief_comparison.html"
GRID = {('correct', 'high'): "0.95", ('correct', 'medium'): "0.8", ('correct', 'low'): "0.65",
        ('incorrect', 'low'): "0.35", ('incorrect', 'medium'): "0.2", ('incorrect', 'high'): "0.05"}
CELLMETA = {"0.05": ("incorrect", "high"), "0.2": ("incorrect", "medium"), "0.35": ("incorrect", "low"),
            "0.5": ("null", "parse fail"), "0.65": ("correct", "low"), "0.8": ("correct", "medium"),
            "0.95": ("correct", "high")}


def main():
    rows = {}
    with open(SRC) as f:
        for line in f:
            d = json.loads(line)
            rows[(d["stmt_i"], d["evidence_i"])] = d

    CT = corpus_evidence_text()
    cells = defaultdict(lambda: {"n": 0, "cand": []})
    confs = defaultdict(lambda: {"n": 0, "verd": Counter(), "cand": []})
    for d in rows.values():
        v, c = d.get("verdict"), d.get("confidence")
        key = "0.5" if (v is None or c is None) else GRID.get((v, c), "0.5")
        obj = d.get("object")
        ct = CT.get((d["stmt_i"], d["evidence_i"]), "")   # authoritative sentence (not parsed reasoning)
        ex = {
            "claim": f'{d.get("subject")} [{d.get("stmt_type")}] {obj if obj not in (None, "") else "?"}',
            "src": d.get("source_api"), "pmid": d.get("pmid"), "verdict": v if v else "null",
            "text": ct[:200] + ("…" if len(ct) > 200 else "") if ct else "",
        }
        cell = cells[key]
        cell["n"] += 1
        cell["cand"].append(ex)
        ck = c if c in ("high", "medium", "low") else "null"
        cf = confs[ck]
        cf["n"] += 1
        cf["verd"][v if v else "null"] += 1
        cf["cand"].append(ex)

    def diverse(cands, k=12):
        seen, uniq = set(), []
        for c in cands:
            key = (c["claim"], c["src"])
            if key in seen:
                continue
            seen.add(key); uniq.append(c)
        if len(uniq) <= k:
            return uniq
        step = len(uniq) / k
        return [uniq[int(i * step)] for i in range(k)]

    total = sum(c["n"] for c in cells.values())
    out = {"total": total, "high_pct": round(1000 * sum(cells[k]["n"] for k in ["0.05", "0.95"]) / total) / 10, "cells": {}}
    for key, c in cells.items():
        v, cf = CELLMETA[key]
        out["cells"][key] = {
            "score": key, "verdict": v, "confidence": cf, "n": c["n"],
            "pct": round(1000 * c["n"] / total) / 10,
            "n_examples_total": c["n"], "examples": diverse(c["cand"]),
        }

    # per-confidence verdict split — the calibration punchline for the confidence-mix view:
    # "high" is applied to correct AND incorrect at similar rates, so it carries no signal.
    out["conf_cells"] = {}
    for ck in ("high", "medium", "low", "null"):
        cf = confs.get(ck)
        if not cf:
            continue
        vs = [{"verdict": k2, "n": n2, "pct_within": round(1000 * n2 / cf["n"]) / 10}
              for k2, n2 in cf["verd"].most_common()]
        out["conf_cells"][ck] = {
            "confidence": ck, "n": cf["n"], "pct": round(1000 * cf["n"] / total) / 10,
            "verdict_split": vs, "n_examples_total": cf["n"], "examples": diverse(cf["cand"]),
        }

    # validate vs rendered SVG data-count keyed by data-score
    html = open(HTML).read()
    svg = {}
    for m in re.finditer(r'data-score="([0-9.]+)"[^>]*data-count="(\d+)"', html):
        svg[m.group(1)] = int(m.group(2))
    mism = [(k, svg[k], cells.get(k, {}).get("n", 0)) for k in svg if cells.get(k, {}).get("n", 0) != svg[k]]
    print(f"total {total} | high+ correct/high share (2 cells): {out['high_pct']}%")
    for k in ["0.05", "0.2", "0.35", "0.5", "0.65", "0.8", "0.95"]:
        print(f"  score {k:<5} {CELLMETA[k][0]+'/'+CELLMETA[k][1]:<18} n={cells[k]['n']}")
    print(f"count mismatches vs SVG (if any already-spliced): {len(mism)}  {mism}")
    json.dump(out, open("/tmp/fig6_drill.json", "w"), ensure_ascii=False)
    print(f"wrote /tmp/fig6_drill.json ({len(json.dumps(out))/1024:.1f} KB)")


if __name__ == "__main__":
    main()
