#!/usr/bin/env python3
"""Recompute Figure-1 bucket decomposition + drill-down data from the complete
rasmachine scoring run, and emit the inline JSON blob the report embeds.

Replaces the ad-hoc cells that produced the (deleted) reports/_figure1_data.json.
Buckets are a strict partition (each row in exactly one bucket) under the
precedence below; per-bucket source/type composition and real example rows are
computed for the progressive drill-down.

Predicates (from the report's methodology note):
  row_error        : error is not None / null verdict   (telemetry)
  no_evidence      : text_len == 0                       (schema)
  placeholder_text : 0 < text_len < 30                   (schema)
  incomplete_claim : stmt_type == 'ActiveForm' and object in {None,'?',''}  (schema)
  semantic_correct : verdict == 'correct'
  reader_halluc.   : verdict == 'incorrect' and HALLUC regex matches REASONING
  hedged_evidence  : verdict == 'incorrect' and HEDGE  regex matches EVIDENCE sentence
  semantic_incorrect: verdict == 'incorrect' otherwise
Precedence is top-to-bottom: telemetry > schema-shape > verdict-based.
"""
import json, re, sys
from collections import Counter, defaultdict

SRC = "data/results/rasmachine_mono_gemma_remote_direct.jsonl"

# The bucket taxonomy now lives in the package (single source of truth); the
# report scripts that `from fig1_drilldown_data import ...` keep working via this
# re-export. See src/indra_belief/results.py.
from indra_belief.results import (  # noqa: E402,F401
    META, ORDER, GROUP_NAME, split_preview, classify, corpus_evidence_text,
    HALLUC, HEDGE, EV_COLON, EV_TICK,
)

def main():
    rows = {}
    with open(SRC) as f:
        for line in f:
            d = json.loads(line)
            rows[(d["stmt_i"], d["evidence_i"])] = d

    buckets = defaultdict(lambda: {"n":0, "src":Counter(), "type":Counter(), "cand":[]})
    tl0_verdicts = Counter()
    for d in rows.values():
        ev, reasoning = split_preview(d.get("raw_text_preview"), d.get("text_len") or 0)
        b = classify(d, ev, reasoning)
        bk = buckets[b]
        bk["n"] += 1
        bk["src"][d.get("source_api")] += 1
        bk["type"][d.get("stmt_type")] += 1
        if (d.get("text_len") or 0) == 0:
            tl0_verdicts[d.get("verdict")] += 1
        obj = d.get("object")
        claim = f'{d.get("subject")} [{d.get("stmt_type")}] {obj if obj not in (None,"") else "?"}'
        bk["cand"].append({
            "claim": claim,
            "src": d.get("source_api"),
            "pmid": d.get("pmid"),
            "belief": round(d.get("belief"), 3) if isinstance(d.get("belief"),(int,float)) else None,
            "conf": d.get("confidence"),
            "verdict": d.get("verdict"),
            "text": (ev[:280] if ev else ""),
        })

    def diverse_sample(cands, k=30):
        """Evenly-spaced sample across the bucket, preferring evidence-bearing
        rows and de-duplicating repeated claims for a richer browse."""
        with_ev = [c for c in cands if c["text"]]
        pool = with_ev if len(with_ev) >= k else cands
        seen, uniq = set(), []
        for c in pool:                       # de-dup identical claims, keep order
            key = (c["claim"], c["src"])
            if key in seen: continue
            seen.add(key); uniq.append(c)
        if len(uniq) <= k:
            return uniq
        step = len(uniq) / k                 # even stride across the whole bucket
        return [uniq[int(i*step)] for i in range(k)]

    total = sum(b["n"] for b in buckets.values())
    print(f"total partitioned rows: {total}\n")
    print(f"{'bucket':<22}{'n':>8}{'pct':>8}   top sources")
    for name in ORDER:
        b = buckets[name]
        top = ", ".join(f"{s}:{c}" for s,c in b["src"].most_common(3))
        print(f"{name:<22}{b['n']:>8}{100*b['n']/total:>7.1f}%   {top}")
    print(f"\ntext_len==0 verdict distribution (overlap check): {dict(tl0_verdicts)}")

    # group rollups
    grp = Counter()
    GRP_OF = {n: META[n][0] for n in META}
    for name in ORDER:
        grp[GRP_OF[name]] += buckets[name]["n"]
    print(f"\ngroup totals: {dict(grp)}")

    # --- before-bar (naive two-tone, decomposition-aligned: correct stays, rest is 'incorrect') ---
    n_correct = buckets["semantic_correct"]["n"]
    n_error   = buckets["row_error"]["n"]
    n_incorrect = total - n_correct - n_error
    print(f"\nBEFORE bar (naive):  correct={n_correct} ({100*n_correct/total:.1f}%)  "
          f"incorrect={n_incorrect} ({100*n_incorrect/total:.1f}%)  error={n_error} ({100*n_error/total:.2f}%)")

    # --- Figure 2 waterfall: remove the four artifact classes from the 'incorrect' slab ---
    print("\nFIGURE-2 waterfall (cleanup):")
    rate0 = 100*n_incorrect/total
    print(f"  start: {n_incorrect}/{total} = {rate0:.1f}% incorrect")
    pop = total; inc = n_incorrect
    for label in ["no_evidence","incomplete_claim","placeholder_text","row_error","reader_hallucination"]:
        rem = buckets[label]["n"]
        pop -= rem
        if label != "row_error":   # row_error rows aren't 'incorrect'; they leave the pop only
            inc -= rem
        print(f"  − {label:<20} ({rem:>6}) → pop {pop:>6}, incorrect {inc:>6} = {100*inc/pop:5.1f}%")
    print(f"  cleaned population: {pop} rows, {100*inc/pop:.1f}% incorrect")

    # --- TL;DR characterization: shares of the naive 'incorrect' slab ---
    schema_art = buckets["no_evidence"]["n"]+buckets["incomplete_claim"]["n"]+buckets["placeholder_text"]["n"]
    halluc = buckets["reader_hallucination"]["n"]
    semantic = buckets["semantic_incorrect"]["n"]+buckets["hedged_evidence"]["n"]
    print(f"\nTL;DR shares of the {n_incorrect} 'incorrect' slab:  "
          f"schema {100*schema_art/n_incorrect:.0f}%  reader-halluc {100*halluc/n_incorrect:.0f}%  "
          f"semantic {100*semantic/n_incorrect:.0f}%")

    # emit JSON blob (compact) for inlining
    out = {"total": total, "buckets": {}}
    TOPK = 8
    for name in ORDER:
        b = buckets[name]
        def topk(counter):
            items = counter.most_common()
            head = items[:TOPK]
            tail = sum(c for _,c in items[TOPK:])
            arr = [{"k":k or "—","n":c} for k,c in head]
            if tail: arr.append({"k":"other","n":tail})
            return arr
        out["buckets"][name] = {
            "n": b["n"],
            "group": META[name][0],
            "remedy": META[name][1],
            "by_source": topk(b["src"]),
            "by_type": topk(b["type"]),
            "n_examples_total": b["n"],
            "examples": diverse_sample(b["cand"], 30),
        }
    with open("/tmp/fig1_drill.json","w") as f:
        json.dump(out, f, ensure_ascii=False)
    sz = len(json.dumps(out, ensure_ascii=False))
    print(f"\nwrote /tmp/fig1_drill.json  ({sz/1024:.1f} KB inline)")

if __name__ == "__main__":
    main()
