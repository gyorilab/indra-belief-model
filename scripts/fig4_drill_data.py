#!/usr/bin/env python3
"""Per-bar drill-down data for Figure 4's per-source histograms.

Replicates fig4_gen_svg.py's EXACT population + binning (dedup by (stmt_i,evidence_i)
keep-last; per statement: RasMachine belief = first-seen `belief`, OUR belief = mean of
ALL scored evidence, source = first-seen source_api; only the 5 SHOWN sources). For each
bar (keyed "source|pred|bin", pred in {rasm,our}, bin 0..19 over [0,1] step 0.05) emits
the statement-type composition + a diverse sample of example statements (claim, pmid,
RasMachine belief, our belief). Validates every bar count against the rendered SVG.
"""
import json, re, sys
from collections import defaultdict, Counter
sys.path.insert(0, "scripts")
from fig1_drilldown_data import corpus_evidence_text

SRC = "data/results/rasmachine_mono_gemma_remote_direct.jsonl"
HTML = "reports/rasmachine_belief_comparison.html"
SHOWN = {"bel", "reach", "signor", "sparser", "eidos"}
NBINS = 20


def bin_index(x):
    return min(max(int(x * NBINS), 0), NBINS - 1)


def main():
    rows = {}
    with open(SRC) as f:
        for line in f:
            d = json.loads(line)
            rows[(d["stmt_i"], d["evidence_i"])] = d

    CT = corpus_evidence_text()
    belief, scores, source, meta = {}, defaultdict(list), {}, {}
    evtext, nev = {}, Counter()      # first non-empty evidence sentence + count per statement
    for d in rows.values():
        h = d["stmt_hash"]
        if h not in belief and isinstance(d.get("belief"), (int, float)):
            belief[h] = d["belief"]
        if h not in source:
            source[h] = d.get("source_api")
        if isinstance(d.get("score"), (int, float)):
            scores[h].append(d["score"])
        ct = CT.get((d["stmt_i"], d["evidence_i"]), "")   # authoritative sentence
        if ct:
            nev[h] += 1
            if h not in evtext:
                evtext[h] = ct[:150] + ("…" if len(ct) > 150 else "")
        if h not in meta:
            meta[h] = (d.get("subject"), d.get("object"), d.get("stmt_type"), d.get("pmid"))
        elif meta[h][3] is None and d.get("pmid"):
            s, o, t, _ = meta[h]
            meta[h] = (s, o, t, d.get("pmid"))

    cells = defaultdict(lambda: {"n": 0, "type": Counter(), "cand": []})

    def add(key, ex, stype):
        c = cells[key]
        c["n"] += 1
        c["type"][stype] += 1
        c["cand"].append(ex)

    for h, b in belief.items():
        if not scores.get(h):
            continue
        src = source[h]
        if src not in SHOWN:
            continue
        our = sum(scores[h]) / len(scores[h])
        subj, obj, stype, pmid = meta[h]
        ex = {
            "claim": f'{subj} [{stype}] {obj if obj not in (None, "") else "?"}',
            "pmid": pmid, "ras": round(b, 3), "ours": round(our, 3),
            "text": evtext.get(h, ""), "ne": nev.get(h, 0),
        }
        add(f"{src}|rasm|{bin_index(b)}", ex, stype)
        add(f"{src}|our|{bin_index(our)}", ex, stype)

    def diverse(cands, k=10):
        seen, uniq = set(), []
        for c in cands:
            key = (c["claim"], c["pmid"])
            if key in seen:
                continue
            seen.add(key); uniq.append(c)
        if len(uniq) <= k:
            return uniq
        step = len(uniq) / k
        return [uniq[int(i * step)] for i in range(k)]

    def topk(counter, n=6):
        items = counter.most_common()
        head, tail = items[:n], sum(c for _, c in items[n:])
        arr = [{"k": k or "—", "n": c} for k, c in head]
        if tail:
            arr.append({"k": "other", "n": tail})
        return arr

    out = {"cells": {}}
    for key, c in cells.items():
        out["cells"][key] = {
            "n": c["n"], "by_type": topk(c["type"]),
            "n_examples_total": c["n"], "examples": diverse(c["cand"]),
        }

    # ---- VALIDATE every bar against the rendered SVG ----
    html = open(HTML).read()
    svg_bars = {}
    for m in re.finditer(
            r'data-source="([^"]+)"\s+data-pred="([^"]+)"\s+data-bin="(\d+)"[^>]*>'
            r'<title>[^<]*·\s*n\s*=\s*(\d+)</title>', html):
        svg_bars[f"{m.group(1)}|{m.group(2)}|{m.group(3)}"] = int(m.group(4))
    mism = [(k, sc, cells.get(k, {}).get("n", 0)) for k, sc in svg_bars.items()
            if cells.get(k, {}).get("n", 0) != sc]
    print(f"bars in SVG: {len(svg_bars)}   cells computed: {len(cells)}")
    print(f"count mismatches vs SVG <title>: {len(mism)}")
    for k, sc, mine in mism[:20]:
        print(f"   {k}: SVG={sc} recompute={mine}")
    sz = len(json.dumps(out, ensure_ascii=False))
    json.dump(out, open("/tmp/fig4_drill.json", "w"), ensure_ascii=False)
    print(f"wrote /tmp/fig4_drill.json  ({sz/1024:.1f} KB)")


if __name__ == "__main__":
    main()
