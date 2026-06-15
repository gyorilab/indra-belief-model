#!/usr/bin/env python3
"""Recompute every number cited in the TL;DR <aside> and the section-1 intro
prose of reports/rasmachine_belief_comparison.html from the COMPLETE run.

Streams the 486MB jsonl, dedups by (stmt_i, evidence_i) keeping the LAST
occurrence, and reuses classify()/split_preview() from fig1_drilldown_data.py so
the "bucket-aware" filtering matches Figures 1-3 exactly.

Prints OLD (snapshot value baked in the report) vs NEW for each cited value.
"""
import json
import math
import sys
from collections import defaultdict, Counter

sys.path.insert(0, "scripts")
from fig1_drilldown_data import classify, split_preview  # noqa: E402

# SRC (complete gemma run) + the Figure-3 / Figure-4-subtable "bucket-aware"
# filter (authoritative, matches scripts/fig3_refresh.py & fig4_refresh.py
# exactly): a statement is kept iff it has >=1 NON-ARTIFACT scored evidence, and
# OUR belief aggregates over the NON-ARTIFACT evidence only. NON-artifact = the
# three verdict-bearing buckets (reader_hallucination is treated as an artifact,
# NOT meaningful).
from fig_utils import SRC, NON_ARTIFACT  # noqa: E402, F401


def pearson(xs, ys):
    n = len(xs)
    if n < 2:
        return float("nan")
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    if sxx == 0 or syy == 0:
        return float("nan")
    return sxy / math.sqrt(sxx * syy)


def main():
    rows = {}
    n_lines = 0
    with open(SRC) as f:
        for line in f:
            n_lines += 1
            d = json.loads(line)
            rows[(d["stmt_i"], d["evidence_i"])] = d
    print(f"raw lines read           : {n_lines}")
    print(f"unique (stmt_i,evid_i)   : {len(rows)}   (OLD cited 47,434)\n")

    # ---- per-statement accumulation ----
    # scored = verdict is not None and not a row error (i.e. an actual score)
    by_stmt = defaultdict(lambda: {
        "belief": None,            # RasMachine per-statement belief (constant)
        "scores": [],              # all scored-row scores (verdict not null)
        "scores_clean": [],        # scores from rows in a "meaningful" bucket
        "n_correct": 0,            # any-correct counter
        "n_rows": 0,
        "src": None,               # primary source_api (statements may mix; take majority)
        "src_counter": Counter(),
    })

    n_scored_rows = 0              # rows with a non-null verdict
    conf_counter = Counter()
    # for per-source statement-level means we need each statement's source.

    for d in rows.values():
        h = d["stmt_hash"]
        s = by_stmt[h]
        if s["belief"] is None and isinstance(d.get("belief"), (int, float)):
            s["belief"] = d["belief"]
        if s["src"] is None:                       # first-seen source_api (matches fig4)
            s["src"] = d.get("source_api")
        s["src_counter"][d.get("source_api")] += 1

        verdict = d.get("verdict")
        score = d.get("score")
        if verdict is not None and isinstance(score, (int, float)):
            n_scored_rows += 1
            s["scores"].append(score)
            if verdict == "correct":
                s["n_correct"] += 1
            s["n_rows"] += 1
            conf = d.get("confidence")
            if conf is not None:
                conf_counter[conf] += 1

        # bucket-aware: keep this row's score for the cleaned aggregate only if
        # it is in a NON-ARTIFACT (verdict-bearing) bucket
        ev, reasoning = split_preview(d.get("raw_text_preview"), d.get("text_len") or 0)
        b = classify(d, ev, reasoning)
        if b in NON_ARTIFACT and isinstance(score, (int, float)):
            s["scores_clean"].append(score)

    n_unique_stmts = len(by_stmt)
    n_scored_stmts = sum(1 for s in by_stmt.values() if s["scores"])
    print(f"unique statements        : {n_unique_stmts}")
    print(f"scored statements        : {n_scored_stmts}   (OLD cited 8,716 scored / 8,724 corpus)")
    print(f"scored evidence rows     : {n_scored_rows}   (verdict not null)\n")

    # ============================================================
    # TL;DR item 1 — min RasMachine belief, mass below 0.50, Pearson
    # ============================================================
    beliefs = [s["belief"] for s in by_stmt.values() if s["belief"] is not None]
    min_belief = min(beliefs)
    print(f"[1] min RasMachine belief: {min_belief:.4f}   (OLD '>= 0.86')")

    # our per-statement belief (mean policy) over scored statements
    our_mean = {}
    for h, s in by_stmt.items():
        if s["scores"]:
            our_mean[h] = sum(s["scores"]) / len(s["scores"])
    below_50 = sum(1 for v in our_mean.values() if v < 0.50)
    print(f"    our-belief (mean) statements scored : {len(our_mean)}")
    print(f"    mass below 0.50 (mean-agg)          : {below_50} "
          f"({100*below_50/len(our_mean):.1f}% of scored)   (OLD: prose says 'substantial mass below 0.50')")

    # ---- Pearson on the bucket-aware filtered set (Figure 3 definition) ----
    # filtered statement = has >=1 meaningful row; RasMachine belief vs our
    # mean-aggregated belief over the CLEANED scores.
    xs, ys = [], []
    for h, s in by_stmt.items():
        if s["scores_clean"] and s["belief"] is not None:
            xs.append(s["belief"])
            ys.append(sum(s["scores_clean"]) / len(s["scores_clean"]))
    r_clean = pearson(xs, ys)
    print(f"    Figure-3 filtered set n             : {len(xs)}   (Fig3 SVG cites n=4,830; §3 intro/table cite 4,612)")
    print(f"    Pearson (mean-agg, cleaned scores)  : {r_clean:+.4f}   (OLD TL;DR +0.149; Fig3 refreshed +0.155)\n")

    # Also compute Pearson over the filtered set but using the FULL (uncleaned)
    # mean over all scored rows, in case the snapshot aggregated differently.
    xs2, ys2 = [], []
    kept = {h for h, s in by_stmt.items() if s["scores_clean"]}
    for h in kept:
        s = by_stmt[h]
        if s["scores"] and s["belief"] is not None:
            xs2.append(s["belief"])
            ys2.append(sum(s["scores"]) / len(s["scores"]))
    print(f"    (alt) Pearson filtered-stmts, mean over ALL scored rows: {pearson(xs2, ys2):+.4f} (n={len(xs2)})\n")

    # ============================================================
    # TL;DR item 2 — schema / hallucination / semantic split of naive incorrect
    # ============================================================
    buckets = Counter()
    for d in rows.values():
        ev, reasoning = split_preview(d.get("raw_text_preview"), d.get("text_len") or 0)
        buckets[classify(d, ev, reasoning)] += 1
    total = sum(buckets.values())
    n_correct = buckets["semantic_correct"]
    n_error = buckets["row_error"]
    n_incorrect = total - n_correct - n_error
    schema_art = buckets["no_evidence"] + buckets["incomplete_claim"] + buckets["placeholder_text"]
    halluc = buckets["reader_hallucination"]
    semantic = buckets["semantic_incorrect"] + buckets["hedged_evidence"]
    print(f"[2] naive 'incorrect' slab          : {n_incorrect}")
    print(f"    schema artifacts share          : {100*schema_art/n_incorrect:.1f}%  (OLD '~25%')")
    print(f"    reader-hallucination share      : {100*halluc/n_incorrect:.1f}%  (OLD '~25%')")
    print(f"    genuine-semantic share          : {100*semantic/n_incorrect:.1f}%  (OLD '~13%')")
    print(f"      (semantic_incorrect only      : {100*buckets['semantic_incorrect']/n_incorrect:.1f}%)\n")

    # ============================================================
    # TL;DR item 3 — per-source per-statement mean belief
    # ============================================================
    print("[3] per-source per-statement OUR mean belief:")
    print("    (TL;DR cites the §4 sub-table = BUCKET-AWARE filtered set, non-artifact agg)")
    # over ALL scored statements (Figure-4 histogram definition)
    src_means_all = defaultdict(list)
    for h, s in by_stmt.items():
        if s["scores"] and s["src"]:
            src_means_all[s["src"]].append(sum(s["scores"]) / len(s["scores"]))
    # over the FILTERED set, non-artifact aggregation (matches §4 sub-table the TL;DR cites)
    src_means_filt = defaultdict(list)
    for h in kept:
        s = by_stmt[h]
        if s["scores_clean"] and s["src"]:
            src_means_filt[s["src"]].append(sum(s["scores_clean"]) / len(s["scores_clean"]))

    OLD = {"reach": 0.74, "sparser": 0.56, "eidos": 0.42}
    for src in ["reach", "sparser", "eidos"]:
        a = src_means_all.get(src, [])
        fl = src_means_filt.get(src, [])
        ma = sum(a)/len(a) if a else float("nan")
        mf = sum(fl)/len(fl) if fl else float("nan")
        print(f"    {src:<8} fig4-hist n={len(a):>5} mean={ma:.4f}  | §4-table(filtered) n={len(fl):>5} mean={mf:.4f}  -> TL;DR={mf:.2f}  (OLD {OLD[src]})")
    print()

    # ============================================================
    # TL;DR item 4 — touched statements, any-correct, all-correct
    # ============================================================
    # "touched" = statements with >=1 scored row (a verdict was produced).
    touched = [s for s in by_stmt.values() if s["n_rows"] > 0]
    n_touched = len(touched)
    any_correct = sum(1 for s in touched if s["n_correct"] >= 1)
    all_correct = sum(1 for s in touched if s["n_correct"] == s["n_rows"])
    print(f"[4] touched statements (>=1 scored row): {n_touched}   (OLD 7,349)")
    print(f"    any-correct                         : {any_correct} ({100*any_correct/n_touched:.1f}%)  (OLD 48.0%)")
    print(f"    all-correct                         : {all_correct} ({100*all_correct/n_touched:.1f}%)  (OLD 23.1%)\n")

    # ============================================================
    # TL;DR item 5 — confidence calibration
    # ============================================================
    tot_conf = sum(conf_counter.values())
    hi = conf_counter.get("high", 0)
    print(f"[5] confidence distribution among scored rows: {dict(conf_counter)}")
    print(f"    high-confidence share               : {100*hi/tot_conf:.1f}%   (OLD 98.7%)\n")

    # ============================================================
    # §1 intro — corpus statement & evidence-row counts
    # ============================================================
    print("§1 intro cited counts:")
    print(f"    INDRA statements (unique stmt_hash) : {n_unique_stmts}   (OLD '8,724')")
    print(f"    evidence rows (unique stmt_i,ev_i)  : {len(rows)}   (OLD '47,434')")


if __name__ == "__main__":
    main()
