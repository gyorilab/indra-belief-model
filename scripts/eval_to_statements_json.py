"""Convert a flat curation-eval JSONL (belief_benchmark schema) into an INDRA
statements JSON that run_rasmachine_monolithic.py can score.

Each eval row -> one INDRA Statement carrying ONLY its curated Evidence,
resolved from the benchmark corpus by CorpusIndex (so agent db_refs survive and
the scorer's grounding tier behaves exactly as in production). One
single-evidence statement per row keeps source_hash unique per output row, so
the blind scorer output joins back to gold on (matches_hash, source_hash) with
no ambiguity and no version skew (same corpus the curations were made against).

    PYTHONPATH=src python scripts/eval_to_statements_json.py \
        --input data/benchmark/eval_curation_v1.jsonl \
        --output data/corpora/eval_curation_v1_statements.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from indra_belief.data.corpus import CorpusIndex  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default=str(ROOT / "data/benchmark/eval_curation_v1.jsonl"))
    ap.add_argument("--output", default=str(ROOT / "data/corpora/eval_curation_v1_statements.json"))
    ap.add_argument("--corpus", default=None, help="override benchmark corpus path")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.input) if l.strip()]
    idx = CorpusIndex(args.corpus) if args.corpus else CorpusIndex()

    stmts_json: list[dict] = []
    skipped: list[dict] = []
    for r in rows:
        res = idx.get(r["source_hash"], r.get("subject", ""), r.get("object", ""))
        if res is None:
            skipped.append({"source_hash": r["source_hash"],
                            "subject": r.get("subject"), "object": r.get("object")})
            continue
        stmt, ev = res
        stmt.evidence = [ev]  # score only the curated evidence
        stmts_json.append(stmt.to_json())

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(stmts_json, f)

    print(f"input rows:      {len(rows)}")
    print(f"resolved stmts:  {len(stmts_json)}")
    print(f"skipped (not in corpus): {len(skipped)}")
    if skipped:
        for s in skipped[:10]:
            print(f"    {s}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
