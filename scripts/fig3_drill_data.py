#!/usr/bin/env python3
"""Per-cell drill-down data for Figure 3's hexbin (progressive disclosure).

Replicates fig3_refresh.py's EXACT filter + binning so each cell's statement set
matches the rendered cell's data-count, then emits, per cell (keyed "cx,cy" using
the SVG's data-cx 24..27 / data-cy 0..17), the source-DB + statement-type
composition and a diverse sample of real example statements (claim, source, pmid,
RasMachine belief, our mean belief). Validates counts against the live SVG.
"""
import json, re, sys
from collections import defaultdict, Counter
sys.path.insert(0, "scripts")
from fig1_drilldown_data import classify, split_preview, corpus_evidence_text
from fig_utils import SRC, NON_ARTIFACT  # noqa: F401

HTML = "reports/rasmachine_belief_comparison.html"
XLO, XHI, NX, NY = 0.857, 1.0, 4, 18
CX_MAP = {0: 24, 1: 25, 2: 26, 3: 27}
XLABEL = {24: "[0.857, 0.893]", 25: "[0.893, 0.929]", 26: "[0.929, 0.964]", 27: "[0.964, 1.000]"}
YLABEL = [f"[{i/NY:.2f}, {(i+1)/NY:.2f}]" for i in range(NY)]


def main():
    rows = {}
    with open(SRC) as f:
        for line in f:
            d = json.loads(line)
            rows[(d["stmt_i"], d["evidence_i"])] = d

    CT = corpus_evidence_text()
    by = defaultdict(lambda: {"belief": None, "na": [], "subj": None, "obj": None,
                              "stype": None, "src": None, "pmid": None,
                              "text": "", "nev": 0})
    for d in rows.values():
        sh = d["stmt_hash"]
        ev, reasoning = split_preview(d.get("raw_text_preview"), d.get("text_len") or 0)
        b = classify(d, ev, reasoning)
        s = by[sh]
        ct = CT.get((d["stmt_i"], d["evidence_i"]), "")   # authoritative sentence (not parsed reasoning)
        if ct:                       # first non-empty evidence sentence + count, for drill disclosure
            s["nev"] += 1
            if not s["text"]:
                s["text"] = ct[:150] + ("…" if len(ct) > 150 else "")
        if b in NON_ARTIFACT and isinstance(d.get("score"), (int, float)):
            s["na"].append(d["score"])
        if s["belief"] is None and isinstance(d.get("belief"), (int, float)):
            s["belief"] = d["belief"]
        if s["stype"] is None:
            s["subj"], s["obj"], s["stype"] = d.get("subject"), d.get("object"), d.get("stmt_type")
        if s["src"] is None:
            s["src"] = d.get("source_api")
        if s["pmid"] is None and d.get("pmid"):
            s["pmid"] = d.get("pmid")

    cells = defaultdict(lambda: {"n": 0, "src": Counter(), "type": Counter(), "cand": []})
    for sh, s in by.items():
        if not s["na"] or s["belief"] is None:
            continue
        ras, our = s["belief"], sum(s["na"]) / len(s["na"])
        cx = 0 if ras < XLO else min(int((ras - XLO) / (XHI - XLO) * NX), NX - 1)
        cy = min(int(our * NY), NY - 1)
        key = f"{CX_MAP[cx]},{cy}"
        c = cells[key]
        c["n"] += 1
        c["src"][s["src"]] += 1
        c["type"][s["stype"]] += 1
        obj = s["obj"]
        c["cand"].append({
            "claim": f'{s["subj"]} [{s["stype"]}] {obj if obj not in (None, "") else "?"}',
            "src": s["src"], "pmid": s["pmid"],
            "ras": round(ras, 3), "ours": round(our, 3),
            "text": s["text"], "ne": s["nev"],
        })

    def diverse(cands, k=16):
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

    TOPK = 6
    def topk(counter, total):
        items = counter.most_common()
        head = items[:TOPK]
        tail = sum(c for _, c in items[TOPK:])
        arr = [{"k": k or "—", "n": c} for k, c in head]
        if tail:
            arr.append({"k": "other", "n": tail})
        return arr

    out = {"xlabel": XLABEL, "ylabel": {str(i): YLABEL[i] for i in range(NY)}, "cells": {}}
    for key, c in cells.items():
        out["cells"][key] = {
            "n": c["n"], "by_source": topk(c["src"], c["n"]),
            "by_type": topk(c["type"], c["n"]),
            "n_examples_total": c["n"], "examples": diverse(c["cand"], 16),
        }

    # ---- VALIDATE against the rendered SVG's data-count ----
    with open(HTML) as f:
        html = f.read()
    svg_counts = {}
    for m in re.finditer(r'data-count="(\d+)"\s+data-cx="(\d+)"\s+data-cy="(\d+)"', html):
        svg_counts[f"{m.group(2)},{m.group(3)}"] = int(m.group(1))
    mism = []
    for k, sc in svg_counts.items():
        mine = cells.get(k, {}).get("n", 0)
        if mine != sc:
            mism.append((k, sc, mine))
    total = sum(c["n"] for c in cells.values())
    print(f"filtered statements: {total}  (expect 5,602)")
    print(f"cells with data: {len(cells)}   SVG cells: {len(svg_counts)}")
    print(f"count mismatches vs SVG data-count: {len(mism)}")
    for k, sc, mine in mism[:20]:
        print(f"   cell {k}: SVG={sc} recompute={mine}")
    sz = len(json.dumps(out, ensure_ascii=False))
    with open("/tmp/fig3_drill.json", "w") as f:
        json.dump(out, f, ensure_ascii=False)
    print(f"wrote /tmp/fig3_drill.json  ({sz/1024:.1f} KB)")


if __name__ == "__main__":
    main()
