"""How far can Ben Gyori's curations extend the rasmachine gold beyond the n=60?

The current rasmachine_v1 gold is doubly narrow:
  (a) only 100 evidences were SAMPLED from the 47,434-evidence corpus, and
  (b) only curations with source='indra_belief_rasmachine' (made via our viewer)
      counted.

But the corpus file holds every evidence's text + source_hash, so ANY Ben
curation that joins to a corpus (matches_hash, source_hash) pair is fully
recoverable — no ev_json needed. This probe measures the join across three
widenings, independently:

  W1  same 100-ev sample, drop the source filter      (Ben, any source)
  W2  whole corpus, source='indra_belief_rasmachine'  (our-viewer curations we never sampled)
  W3  whole corpus, Ben any source                    (the full Ben-on-corpus gold)

Reports new-vs-current(60), balance, source + version-skew diagnostics.

    PYTHONPATH=src .venv/bin/python scripts/probe_ben_gold_extension.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from indra_belief.curation import aggregate_gold  # noqa: E402

BEN = {"ben.gyori@gmail.com", "ben.gyori"}
CORPUS = ROOT / "data" / "corpora" / "latest_statements_rasmachine.json"
UNIVERSE = ROOT / "data" / "results" / "curation_universe_all.jsonl"
SAMPLE_STMTS = ROOT / "data" / "corpora" / "rasmachine_v1_statements.json"
CUR_GOLD = ROOT / "data" / "benchmark" / "rasmachine_v1_gold.jsonl"


def corpus_pairs() -> tuple[set, set, dict]:
    """Return (all corpus (mh,sh) pairs, set of corpus matches_hashes,
    pair -> evidence_text)."""
    pairs, mhs, text = set(), set(), {}
    data = json.load(open(CORPUS))
    for s in data:
        mh = s.get("matches_hash")
        if mh is None:
            continue
        mh = int(mh)
        mhs.add(mh)
        for ev in s.get("evidence", []) or []:
            sh = ev.get("source_hash")
            if sh is None:
                continue
            try:
                k = (mh, int(sh))
            except (TypeError, ValueError):
                continue
            pairs.add(k)
            if ev.get("text"):
                text[k] = ev["text"]
    return pairs, mhs, text


def sample_mhs() -> set:
    """matches_hashes of the 91-statement v1 sample (the current gold's scope)."""
    mhs = set()
    if SAMPLE_STMTS.exists():
        d = json.load(open(SAMPLE_STMTS))
        stmts = d.get("statements", d) if isinstance(d, dict) else d
        for s in stmts:
            mh = s.get("matches_hash")
            if mh is not None:
                mhs.add(int(mh))
    return mhs


def current_gold_pairs() -> set:
    out = set()
    for line in open(CUR_GOLD):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        out.add((int(r["matches_hash"]), int(r["source_hash"])))
    return out


def ben_curations() -> list[dict]:
    out = []
    for line in open(UNIVERSE):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        if (d.get("curator") or "").strip() in BEN:
            out.append(d)
    return out


def pair_of(d):
    mh, sh = d.get("pa_hash"), d.get("source_hash")
    if mh is None or sh is None:
        return None
    try:
        return (int(mh), int(sh))
    except (TypeError, ValueError):
        return None


def widen(name, ben, cpairs, ctext, scope_mhs, source_filter, cur_pairs):
    """Join Ben's curations to the corpus under a scope/source filter; report."""
    rows = []
    for d in ben:
        if source_filter is not None and d.get("source") != source_filter:
            continue
        k = pair_of(d)
        if k is None:
            continue
        if scope_mhs is not None and k[0] not in scope_mhs:
            continue
        if k in cpairs:  # joins to corpus evidence -> recoverable
            rows.append((k, d.get("tag", ""), d.get("source")))
    by_pair = defaultdict(list)
    src = Counter()
    for k, tag, s in rows:
        by_pair[k].append(tag)
        src[s] += 1
    gold = {k: aggregate_gold(t) for k, t in by_pair.items()}
    nc = sum(1 for g in gold.values() if g == "correct")
    ni = sum(1 for g in gold.values() if g == "incorrect")
    joined = set(by_pair)
    new = joined - cur_pairs
    new_c = sum(1 for k in new if gold[k] == "correct")
    new_i = sum(1 for k in new if gold[k] == "incorrect")
    print(f"\n=== {name} ===", flush=True)
    print(f"  Ben curations joining corpus evidence: {len(rows)} over {len(joined)} pairs", flush=True)
    print(f"  gold balance: {nc} correct / {ni} incorrect", flush=True)
    print(f"  NEW beyond current 60-gold: {len(new)} pairs ({new_c} correct / {new_i} incorrect)", flush=True)
    print(f"  total gold if merged: {len(joined | cur_pairs)} pairs", flush=True)
    print(f"  by source: {dict(src.most_common())}", flush=True)
    return joined


def main():
    cpairs, cmhs, ctext = corpus_pairs()
    print(f"corpus: {len(cmhs)} statements, {len(cpairs)} (stmt,evidence) pairs with text", flush=True)
    smhs = sample_mhs()
    print(f"v1 sample scope: {len(smhs)} statement hashes", flush=True)
    cur = current_gold_pairs()
    print(f"current gold: {len(cur)} pairs", flush=True)

    ben = ben_curations()
    ben_pairs = {pair_of(d) for d in ben} - {None}
    ben_on_corpus = {k for k in ben_pairs if k in cpairs}
    ben_mh_on_corpus = {k[0] for k in ben_pairs if k[0] in cmhs}
    print(f"\nBen curations in universe: {len(ben)} over {len(ben_pairs)} pairs", flush=True)
    print(f"  Ben pairs whose STATEMENT is in corpus: "
          f"{sum(1 for k in ben_pairs if k[0] in cmhs)} (over {len(ben_mh_on_corpus)} statements)", flush=True)
    print(f"  Ben pairs that JOIN corpus exactly (mh,sh): {len(ben_on_corpus)}  "
          f"<- version-skew gate", flush=True)

    widen("W1  same 100-ev sample, ANY source", ben, cpairs, ctext, smhs, None, cur)
    widen("W2  whole corpus, source=indra_belief_rasmachine", ben, cpairs, ctext, None, "indra_belief_rasmachine", cur)
    widen("W3  whole corpus, ANY source (full Ben-on-corpus)", ben, cpairs, ctext, None, None, cur)


if __name__ == "__main__":
    main()
