"""The plan we actually want: mock7ee (self, anchor) + a sample SPREAD across the
OTHER curators, to diversify curation DIRECTION instead of capturing one curator's.

Reads the cached keyed list-all and reports, for the recoverable (ev_json embedded
-> evidence text travels with the curation) subset:
  - mock7ee's own pool (anchor): n, balance, recoverable
  - the OTHER-curator pool (all minus ben/bachman/mock7ee): recoverable, balance,
    per-curator spread, after de-contamination
  - what a balanced, curator-spread gold could draw

    PYTHONPATH=src .venv/bin/python scripts/multicurator_pool.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from indra_belief.curation import aggregate_gold  # noqa: E402

UNIVERSE = ROOT / "data" / "results" / "curation_universe_all.jsonl"
BENCH = ROOT / "data" / "benchmark"
SELF = "mock7ee@gmail.com"
PAIRS = {"ben.gyori@gmail.com", "bachmanjohn@gmail.com", "ben.gyori", "bachmanjohn"}


def pair_of(d):
    mh, sh = d.get("pa_hash"), d.get("source_hash")
    if mh is None or sh is None:
        return None
    try:
        return (int(mh), int(sh))
    except (TypeError, ValueError):
        return None


def load_excl():
    """All (mh,sh) pairs in prior holdouts/evals/fewshots/benchmark/external_gold."""
    excl = set()
    pats = ("holdout*.jsonl", "eval_set*.jsonl", "eval_curation*.jsonl",
            "fewshot*.jsonl", "belief_benchmark.jsonl", "external_gold*.jsonl",
            "rasmachine*.jsonl", "probe_*.jsonl")
    for pat in pats:
        for p in BENCH.glob(pat):
            for line in p.open():
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if "pa_hash" in r and "source_hash" in r:
                    try:
                        excl.add((int(r["pa_hash"]), int(r["source_hash"])))
                    except (TypeError, ValueError):
                        pass
                elif "matches_hash" in r and "source_hash" in r:
                    try:
                        excl.add((int(r["matches_hash"]), int(r["source_hash"])))
                    except (TypeError, ValueError):
                        pass
    return excl


def summarize(name, curs):
    """curs: list of curation dicts. Group by pair, aggregate gold, count recoverable."""
    tags = defaultdict(list)
    rec = defaultdict(bool)
    cur_by_pair = defaultdict(set)
    for d in curs:
        k = pair_of(d)
        if k is None:
            continue
        tags[k].append(d.get("tag", ""))
        rec[k] = rec[k] or bool(d.get("ev_json"))
        if d.get("curator"):
            cur_by_pair[k].add(d["curator"])
    gold = {k: aggregate_gold(t) for k, t in tags.items()}
    recoverable = {k for k in tags if rec[k]}
    nc = sum(1 for k in recoverable if gold[k] == "correct")
    ni = sum(1 for k in recoverable if gold[k] == "incorrect")
    print(f"\n[{name}] pairs={len(tags)}  recoverable(ev_json)={len(recoverable)}  "
          f"({nc} correct / {ni} incorrect)  balanced-ceiling={2*min(nc,ni)}")
    return recoverable, gold, cur_by_pair


def main():
    data = [json.loads(l) for l in UNIVERSE.open() if l.strip()]
    excl = load_excl()
    print(f"universe curations: {len(data)}   prior-set pairs to exclude: {len(excl)}")

    mine = [d for d in data if (d.get("curator") or "").strip() == SELF]
    others = [d for d in data if (d.get("curator") or "").strip() not in PAIRS | {SELF, ""}]

    summarize("mock7ee (anchor)", mine)
    rec_o, gold_o, curby = summarize("OTHER curators (excl ben/bachman/self)", others)

    # de-contaminated other-curator recoverable pool
    clean = {k for k in rec_o if k not in excl}
    nc = sum(1 for k in clean if gold_o[k] == "correct")
    ni = sum(1 for k in clean if gold_o[k] == "incorrect")
    print(f"\n[OTHER, de-contaminated + recoverable] {len(clean)} pairs "
          f"({nc} correct / {ni} incorrect)  balanced-ceiling={2*min(nc,ni)}")

    # curator spread of the clean pool
    spread = Counter()
    for k in clean:
        for c in curby[k]:
            spread[c] += 1
    print(f"  spread over {len(spread)} curators; top: "
          f"{dict(spread.most_common(12))}")


if __name__ == "__main__":
    main()
