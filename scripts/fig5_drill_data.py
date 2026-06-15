#!/usr/bin/env python3
"""Per-partition drill-down data for Figure 5's aggregation-policy lever.

Replicates fig5_refresh.py's exact partition: per statement (stmt_hash) tally scored
verdicts -> [n_correct, n_incorrect]; "touched" = >=1 scored; all-correct = ni==0 & nc>0;
all-incorrect = nc==0 & ni>0; mixed = both. For each class (key "all"/"mixed"/"none")
emit the source-DB + statement-type composition and a diverse sample of example statements
(claim, source, pmid, and the evidence split nc/total that explains the class membership).
Validates class counts against the rendered segment stats (2,296 / 2,079 / 4,266).
"""
import json, re, sys
from collections import defaultdict, Counter
sys.path.insert(0, "scripts")
from fig1_drilldown_data import corpus_evidence_text

SRC = "data/results/rasmachine_mono_gemma_remote_direct.jsonl"
HTML = "reports/rasmachine_belief_comparison.html"
EXPECT = {"all": 2296, "mixed": 2079, "none": 4266}


def main():
    rows = {}
    with open(SRC) as f:
        for line in f:
            d = json.loads(line)
            rows[(d["stmt_i"], d["evidence_i"])] = d

    CT = corpus_evidence_text()
    tally = defaultdict(lambda: [0, 0])     # stmt_hash -> [nc, ni]
    meta = {}                                # stmt_hash -> (subj, obj, type, src, pmid)
    evtext, nev = {}, Counter()              # first non-empty evidence sentence + count per statement
    for d in rows.values():
        h = d["stmt_hash"]
        v = d.get("verdict")
        if v == "correct":
            tally[h][0] += 1
        elif v == "incorrect":
            tally[h][1] += 1
        ct = CT.get((d["stmt_i"], d["evidence_i"]), "")   # authoritative sentence
        if ct:
            nev[h] += 1
            if h not in evtext:
                evtext[h] = ct[:150] + ("…" if len(ct) > 150 else "")
        if h not in meta:
            meta[h] = [d.get("subject"), d.get("object"), d.get("stmt_type"), d.get("source_api"), d.get("pmid")]
        else:
            if meta[h][3] is None and d.get("source_api"):
                meta[h][3] = d.get("source_api")
            if meta[h][4] is None and d.get("pmid"):
                meta[h][4] = d.get("pmid")

    classes = {k: {"n": 0, "src": Counter(), "type": Counter(), "cand": []} for k in EXPECT}
    for h, (nc, ni) in tally.items():
        if nc + ni == 0:
            continue
        cls = "all" if ni == 0 else ("none" if nc == 0 else "mixed")
        subj, obj, stype, src, pmid = meta[h]
        c = classes[cls]
        c["n"] += 1
        c["src"][src] += 1
        c["type"][stype] += 1
        c["cand"].append({
            "claim": f'{subj} [{stype}] {obj if obj not in (None, "") else "?"}',
            "src": src, "pmid": pmid, "nc": nc, "tot": nc + ni,
            "text": evtext.get(h, ""), "ne": nev.get(h, 0),
        })

    def diverse(cands, k=18):
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

    def topk(counter, n=6):
        items = counter.most_common()
        head, tail = items[:n], sum(c for _, c in items[n:])
        arr = [{"k": k or "—", "n": c} for k, c in head]
        if tail:
            arr.append({"k": "other", "n": tail})
        return arr

    total = sum(c["n"] for c in classes.values())
    out = {"total": total, "cells": {}}
    for k, c in classes.items():
        cands = c["cand"]
        # narrative-tailored composition: the evidence split the POLICY adjudicates,
        # not source/type. mixed -> how split (minority/even/majority support);
        # all-correct/all-incorrect -> depth (how many statements rest on a single vote).
        if k == "mixed":
            lo = sum(1 for x in cands if x["nc"] * 2 < x["tot"])
            even = sum(1 for x in cands if x["nc"] * 2 == x["tot"])
            hi = sum(1 for x in cands if x["nc"] * 2 > x["tot"])
            profile = {
                "label": "how split is the evidence",
                "headline": f"{round(100*lo/c['n'])}% have a minority of evidence supporting — yet any-correct counts every one of these {c['n']:,} as supported.",
                "bars": [{"k": "<50% support", "n": lo}, {"k": "even split", "n": even}, {"k": ">50% support", "n": hi}],
            }
        else:
            depth = Counter(min(x["tot"], 5) for x in cands)
            word = "supporting" if k == "all" else "refuting"
            profile = {
                "label": "depth of evidence",
                "headline": f"{round(100*depth.get(1,0)/c['n'])}% rest on a single {word} evidence — one verdict decides the statement.",
                "bars": [{"k": (f"{i} evidence" if i == 1 else f"{i} evidences" if i < 5 else "5+ evidences"), "n": depth[i]} for i in (1, 2, 3, 4, 5) if depth.get(i)],
            }
        out["cells"][k] = {
            "n": c["n"], "profile": profile,
            "n_examples_total": c["n"], "examples": diverse(cands),
        }

    # ---- validate against expected rendered segment counts ----
    mism = [(k, EXPECT[k], classes[k]["n"]) for k in EXPECT if classes[k]["n"] != EXPECT[k]]
    print(f"touched total: {total}  (expect {sum(EXPECT.values())})")
    for k in ["all", "mixed", "none"]:
        print(f"  {k:<6}: {classes[k]['n']}  (expect {EXPECT[k]})  top src {classes[k]['src'].most_common(2)}")
    print(f"count mismatches vs rendered segments: {len(mism)}  {mism}")
    sz = len(json.dumps(out, ensure_ascii=False))
    json.dump(out, open("/tmp/fig5_drill.json", "w"), ensure_ascii=False)
    print(f"wrote /tmp/fig5_drill.json  ({sz/1024:.1f} KB)")


if __name__ == "__main__":
    main()
