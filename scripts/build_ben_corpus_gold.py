"""Extend the rasmachine gold from n=60 to the FULL set of Ben Gyori's curations
that join the corpus — no new human curation, single trusted curator.

The n=60 rasmachine_v1 gold was doubly narrow: only 100 evidences were sampled,
and only source='indra_belief_rasmachine' (our-viewer) curations counted. But Ben
has curated 6,684 (statement, evidence) pairs in the INDRA DB across many efforts
(EMMAA, bioexp, …); 284 of them join EXACTLY (matches_hash, source_hash) to our
rasmachine corpus, whose evidence text we already hold — so they are fully
recoverable gold with no ev_json dependency and no new labeling.

This builds, through the SAME code path as build_rasmachine_eval (so hashes and
the (mh,sh) join are identical to the working n=60 gold and to the runner):

  * rasmachine_v2_gold.jsonl       — one gold row per Ben-curated corpus evidence
  * rasmachine_v2_statements.json  — the corpus statements carrying >=1 gold row,
                                     as runner input (stmts_to_json)

Gold = aggregate_gold over ALL of Ben's curations on each pair (any-incorrect-wins
across efforts). v2 is a SUPERSET of the n=60 (those pairs reappear, possibly with
extra Ben curations folded in). Score the statements with the existing runner and
compare with eval_curation_compare --gold rasmachine_v2_gold.jsonl.

    PYTHONPATH=src .venv/bin/python scripts/build_ben_corpus_gold.py
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from indra.statements import stmts_from_json, stmts_to_json  # noqa: E402

from build_rasmachine_eval import gold_rows_for  # noqa: E402  (reuse exact gold path)
from indra_belief.curation import Curation, build_index, is_gold_correct  # noqa: E402

BEN = {"ben.gyori@gmail.com", "ben.gyori"}
CORPUS = ROOT / "data" / "corpora" / "latest_statements_rasmachine.json"
UNIVERSE = ROOT / "data" / "results" / "curation_universe_all.jsonl"
STMTS_OUT = ROOT / "data" / "corpora" / "rasmachine_v2_statements.json"
GOLD_OUT = ROOT / "data" / "benchmark" / "rasmachine_v2_gold.jsonl"


def ben_curations() -> list[Curation]:
    if not UNIVERSE.exists():
        raise SystemExit(f"no universe cache at {UNIVERSE} — run curation_pool_feasibility.py first")
    curs: list[Curation] = []
    for line in UNIVERSE.open():
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        if (d.get("curator") or "").strip() not in BEN:
            continue
        c = Curation.from_dict(d)  # keys on pa_hash + source_hash
        if c is not None:
            # carry the curation `source` so we can report the effort mix
            object.__setattr__(c, "_source", d.get("source"))
            curs.append(c)
    return curs


def main() -> int:
    curs = ben_curations()
    idx = build_index(curs)
    ben_stmt_hashes = {c.matches_hash for c in curs}
    print(f"Ben curations: {len(curs)} over {idx.n_evidences} evidences / "
          f"{len(ben_stmt_hashes)} statement hashes")

    # candidate corpus statements: those Ben curated (by stored matches_hash)
    raw = json.load(open(CORPUS))
    cand = []
    for s in raw:
        mh = s.get("matches_hash")
        if mh is None:
            continue
        try:
            if int(mh) in ben_stmt_hashes:
                cand.append(s)
        except (TypeError, ValueError):
            pass
    print(f"corpus statements Ben curated (by stored hash): {len(cand)}")

    stmts = stmts_from_json(cand)
    gold_rows = gold_rows_for(stmts, idx)  # recomputes get_hash(refresh=True)/get_source_hash()
    print(f"gold rows (recomputed-hash join to corpus evidence): {len(gold_rows)}")

    # statements carrying >=1 gold row -> runner input, with each statement's
    # evidence TRIMMED to just the curated pairs. Trimming evidence does not change
    # get_hash(refresh=True) (matches_hash is over agents/type, not evidence), so
    # the join is unaffected — but it cuts the scoring load from every-evidence
    # (~10k, mostly uncurated EMMAA Complex evidence) to exactly the gold set.
    gold_mh = {r["matches_hash"] for r in gold_rows}
    gold_pairs = {(r["matches_hash"], r["source_hash"]) for r in gold_rows}
    kept = []
    for s in stmts:
        mh = s.get_hash(refresh=True)
        if mh not in gold_mh:
            continue
        s.evidence = [ev for ev in (s.evidence or []) if (mh, ev.get_source_hash()) in gold_pairs]
        kept.append(s)
    n_scored_ev = sum(len(s.evidence or []) for s in kept)
    STMTS_OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(stmts_to_json(kept), open(STMTS_OUT, "w"))
    GOLD_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(GOLD_OUT, "w") as fh:
        for r in gold_rows:
            fh.write(json.dumps(r, default=str) + "\n")

    n = len(gold_rows)
    nc = sum(1 for r in gold_rows if is_gold_correct(r["tag"]))
    by_tag = collections.Counter(r["tag"] for r in gold_rows)
    by_type = collections.Counter(r["stmt_type"] for r in gold_rows)
    src = collections.Counter(getattr(c, "_source", None) for c in curs
                              if (c.matches_hash, c.source_hash) in
                              {(r["matches_hash"], r["source_hash"]) for r in gold_rows})
    print(f"\nwrote {len(kept)} statements ({n_scored_ev} evidences to score) -> {STMTS_OUT}")
    print(f"wrote {n} gold rows -> {GOLD_OUT}")
    print(f"\n  balance     : {nc} correct / {n - nc} incorrect")
    print(f"  balanced n  : {2 * min(nc, n - nc)} (forced 1:1 ceiling)")
    print(f"  by tag      : {dict(by_tag)}")
    print(f"  by stmt_type: {dict(by_type)}")
    print(f"  effort mix  : {dict(src.most_common())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
