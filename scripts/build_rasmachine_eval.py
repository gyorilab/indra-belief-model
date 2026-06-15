"""Build the rasmachine sampled-statements eval from an INDRA statement pickle.

Input is a pickled list of INDRA Statements sampled for human curation. This
emits the two artifacts the existing scoring tools consume:

  * a statements JSON (`stmts_to_json`) that run_rasmachine_monolithic.py scores;
  * a gold JSONL whose rows join to a scoring run on the (matches_hash,
    source_hash) pair and carry the curated tag — the schema
    eval_curation_compare.py reads via --gold.

Gold is resolved live from the public INDRA curation endpoint
(db.indra.bio/curation/list/<matches_hash>, no key) via
indra_belief.curation.fetch_curations, filtered to one curation effort's
`source`, and aggregated per evidence with the canonical gold rule
(aggregate_gold). The pulled curations are cached so a re-run can skip the
network with --no-fetch; otherwise every run re-pulls and picks up curations
added since. Coverage is reported, never assumed — an uncurated evidence simply
gets no gold row and is scored-but-unmeasured downstream.

    PYTHONPATH=src .venv/bin/python scripts/build_rasmachine_eval.py \
        --pkl data/corpora/sampled_statements_rasmachine_v1.pkl \
        --source indra_belief_rasmachine

Output names derive from the pkl stem with any "sampled_statements_" prefix
stripped, so the example above writes data/corpora/rasmachine_v1_statements.json
and data/benchmark/rasmachine_v1_gold.jsonl. Score with the existing runner and
compare with the existing harness:

    PYTHONPATH=src .venv/bin/python scripts/run_rasmachine_monolithic.py \
        --input  data/corpora/rasmachine_v1_statements.json \
        --model  gemma-remote --workers 4 \
        --output data/results/rasmachine_v1_gemma.jsonl
    # ... again with --model medpsy-remote ...
    PYTHONPATH=src .venv/bin/python scripts/eval_curation_compare.py \
        --gold data/benchmark/rasmachine_v1_gold.jsonl \
        --a data/results/rasmachine_v1_medpsy.jsonl --a-name MedPsy-4B \
        --b data/results/rasmachine_v1_gemma.jsonl  --b-name gemma-26B
"""
from __future__ import annotations

import argparse
import asyncio
import collections
import json
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from indra.statements import stmts_to_json  # noqa: E402

from indra_belief.curation import (  # noqa: E402
    Curation,
    build_index,
    fetch_curations,
    is_gold_correct,
)

DEFAULT_PKL = ROOT / "data/corpora/sampled_statements_rasmachine_v1.pkl"


def _stem(pkl: Path) -> str:
    """Dataset stem used to name sibling outputs (statements/gold/curations)."""
    name = pkl.stem
    return name[len("sampled_statements_"):] if name.startswith("sampled_statements_") else name


def _label(stmt) -> tuple[str, str]:
    agents = [a for a in (stmt.agent_list() or []) if a is not None]
    names = [a.name for a in agents]
    if not names:
        return "?", "?"
    return (names[0], names[1] if len(names) > 1 else "?")


def _representative_tag(gv) -> str:
    """A single tag for the evidence: the shared curation tag when curators
    agree, else the aggregated binary verdict. Preserves the specific error
    category (no_relation, wrong_relation, ...) for per-tag breakdowns while
    staying faithful to the any-incorrect-wins gold rule on disagreement."""
    tags = gv.tags
    return tags[0] if tags and all(t == tags[0] for t in tags) else gv.verdict


def _load_curations(stmts, source: str, cache: Path, fetch: bool) -> list[dict]:
    """Return raw curation dicts for the dataset's statements, filtered to
    `source`. Pulls live (and refreshes the cache) unless --no-fetch."""
    if not fetch:
        if not cache.exists():
            raise SystemExit(f"--no-fetch but no cached curations at {cache}")
        rows = [json.loads(l) for l in open(cache) if l.strip()]
        print(f"loaded {len(rows)} cached curations from {cache}")
    else:
        hashes = {s.get_hash(refresh=True) for s in stmts}
        rows, failed = asyncio.run(fetch_curations(hashes, concurrency=16))
        if failed:
            print(f"WARNING: {len(failed)} statement hashes failed to fetch", file=sys.stderr)
        cache.parent.mkdir(parents=True, exist_ok=True)
        with open(cache, "w") as fh:
            for r in rows:
                fh.write(json.dumps(r, default=str) + "\n")
        print(f"pulled {len(rows)} curations for {len(hashes)} statements -> {cache}")
    filtered = [r for r in rows if r.get("source") == source]
    if rows and not filtered:
        present = sorted({r.get("source") for r in rows})
        raise SystemExit(
            f"--source {source!r} matched 0 of {len(rows)} curations. "
            f"sources present: {present}"
        )
    return filtered


#: Keys every gold row carries. The first seven are what eval_curation_compare.py
#: reads ('gold' is the binary verdict it joins/sorts on, 'tag' the specific
#: category); the rest are provenance for the viewer and manual inspection.
GOLD_ROW_KEYS = (
    "matches_hash", "source_hash", "stmt_type", "subject", "object",
    "evidence_text", "pmid", "source_api", "tag", "gold", "n_curations", "curators",
)


def gold_rows_for(stmts, idx) -> list[dict]:
    """One gold row per curated evidence, joinable to a scoring run on the
    (matches_hash, source_hash) pair. Uncurated evidences are omitted, so the
    row count is the measurable subset, not the scored set."""
    rows: list[dict] = []
    for stmt in stmts:
        mh = stmt.get_hash(refresh=True)
        subj, obj = _label(stmt)
        stmt_type = type(stmt).__name__
        for ev in stmt.evidence or []:
            sh = ev.get_source_hash()
            gv = idx.gold_for(mh, sh)
            if gv is None:
                continue
            rows.append({
                "matches_hash": mh,
                "source_hash": sh,
                "stmt_type": stmt_type,
                "subject": subj,
                "object": obj,
                "evidence_text": ev.text or "",
                "pmid": ev.pmid,
                "source_api": ev.source_api or "",
                "tag": _representative_tag(gv),
                "gold": gv.verdict,
                "n_curations": gv.n,
                "curators": gv.curators,
            })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pkl", default=str(DEFAULT_PKL))
    ap.add_argument("--source", default="indra_belief_rasmachine",
                    help="curation `source` to accept as gold")
    ap.add_argument("--statements-out", default=None)
    ap.add_argument("--gold-out", default=None)
    ap.add_argument("--curations-cache", default=None,
                    help="raw pulled curations (default: sibling of the gold file)")
    ap.add_argument("--no-fetch", dest="fetch", action="store_false", default=True,
                    help="reuse the cached curations instead of re-pulling")
    args = ap.parse_args()

    pkl = Path(args.pkl)
    stem = _stem(pkl)
    statements_out = Path(args.statements_out or ROOT / f"data/corpora/{stem}_statements.json")
    gold_out = Path(args.gold_out or ROOT / f"data/benchmark/{stem}_gold.jsonl")
    cache = Path(args.curations_cache or ROOT / f"data/benchmark/{stem}_curations.jsonl")

    stmts = pickle.load(open(pkl, "rb"))
    n_ev = sum(len(s.evidence or []) for s in stmts)
    print(f"loaded {len(stmts)} statements / {n_ev} evidences from {pkl.name}")

    # 1. statements JSON for the runner
    statements_out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(stmts_to_json(stmts), open(statements_out, "w"))
    print(f"wrote runner input -> {statements_out}")

    # 2. gold from live curations, aggregated by the canonical rule
    raw = _load_curations(stmts, args.source, cache, args.fetch)
    curs = [c for c in (Curation.from_dict(r) for r in raw) if c is not None]
    idx = build_index(curs)
    print(f"source={args.source!r}: {len(curs)} curations over {idx.n_evidences} evidences")

    gold_rows = gold_rows_for(stmts, idx)
    gold_out.parent.mkdir(parents=True, exist_ok=True)
    with open(gold_out, "w") as fh:
        for r in gold_rows:
            fh.write(json.dumps(r, default=str) + "\n")

    n_gold = len(gold_rows)
    n_correct = sum(1 for r in gold_rows if is_gold_correct(r["tag"]))
    by_tag = collections.Counter(r["tag"] for r in gold_rows)
    by_type = collections.Counter(r["stmt_type"] for r in gold_rows)
    print(f"wrote gold -> {gold_out}")
    print(f"\ncoverage: {n_gold}/{n_ev} evidences carry {args.source} gold")
    print(f"  gold balance: {n_correct} correct / {n_gold - n_correct} incorrect")
    print(f"  by tag      : {dict(by_tag)}")
    print(f"  by stmt_type: {dict(by_type)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
