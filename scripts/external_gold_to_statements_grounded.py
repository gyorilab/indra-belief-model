"""Grounded rebuild of external_gold_v1 statements.

NAME-only statement reconstruction in external_gold_to_statements_json.py
(db_refs={"TEXT": name}) yields 0/308 grounded agents, leaving the scorer's
grounding/provenance tier INERT (not production parity). Its err-F1 of 0.774 is
uninterpretable against the 0.856/0.862 reference with 100% grounded agents.

This build recovers the GROUNDED pre-assembled statement (pa_json, which carries
real db_refs: HGNC/UP/CHEBI/MESH/...) for each gold row via the public per-hash
curation/list route, then attaches the gold row's evidence_text. File order is
preserved (output stmt_i == gold row index == line index in external_gold_v1.jsonl)
so the err-F1 join stays by stmt_i.

Usage:
  PYTHONPATH=src .venv/bin/python scripts/external_gold_to_statements_grounded.py \
    [--input data/benchmark/external_gold_v1.jsonl] \
    [--output data/corpora/external_gold_v1_statements_grounded.json] \
    [--cache /tmp/by_mh.json]   # optional pre-pulled {matches_hash: pa_json}
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from indra.statements import Evidence, stmts_from_json  # noqa: E402

from indra_belief.curation import fetch_curations  # noqa: E402

DEFAULT_INPUT = ROOT / "data" / "benchmark" / "external_gold_v1.jsonl"
DEFAULT_OUTPUT = ROOT / "data" / "corpora" / "external_gold_v1_statements_grounded.json"
DEFAULT_SIDECAR = ROOT / "data" / "corpora" / "external_gold_v1_statements_grounded.index.json"


def pull_pa_json(matches_hashes: list[int]) -> dict[int, dict]:
    """Pull grounded pa_json for each distinct matches_hash from the public DB."""
    out, failed = asyncio.run(fetch_curations(matches_hashes, concurrency=16))
    if failed:
        raise RuntimeError(f"failed to fetch {len(failed)} hashes: {failed[:5]}...")
    by_mh: dict[int, dict] = {}
    for c in out:
        mh = c["_matches_hash"]
        paj = c.get("pa_json")
        if isinstance(paj, str):
            paj = json.loads(paj)
        if paj:
            by_mh[mh] = paj
    return by_mh


def agents_of(paj: dict) -> list[dict]:
    if "members" in paj:
        return [a for a in paj["members"] if a]
    return [a for a in (paj.get("subj"), paj.get("obj")) if a]


def is_grounded(agent: dict) -> bool:
    refs = agent.get("db_refs", {}) or {}
    return any(k != "TEXT" for k in refs)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(DEFAULT_INPUT))
    ap.add_argument("--output", default=str(DEFAULT_OUTPUT))
    ap.add_argument("--sidecar", default=str(DEFAULT_SIDECAR))
    ap.add_argument("--cache", default="")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.input) if l.strip()]
    mhs = sorted({int(r["matches_hash"]) for r in rows})

    if args.cache and Path(args.cache).exists():
        by_mh = {int(k): v for k, v in json.load(open(args.cache)).items()}
        missing = [m for m in mhs if m not in by_mh]
        if missing:
            by_mh.update(pull_pa_json(missing))
    else:
        by_mh = pull_pa_json(mhs)

    out_json = []
    sidecar = []
    total_agents = grounded_agents = 0
    for i, row in enumerate(rows):
        mh = int(row["matches_hash"])
        paj = dict(by_mh[mh])  # shallow copy; we replace evidence below
        # the gold spine pins the evidence text/source for THIS (mh, source_hash) row;
        # pa_json's own evidence list is dropped so we score exactly the gold evidence.
        ev = Evidence(
            text=row["evidence_text"],
            source_api=row.get("source_api"),
            pmid=row.get("pmid"),
        )
        paj["evidence"] = [ev.to_json()]
        # round-trip through stmts_from_json to validate structure + reconstruct
        st = stmts_from_json([paj])[0]
        sj = st.to_json()
        out_json.append(sj)

        for a in agents_of(paj):
            total_agents += 1
            if is_grounded(a):
                grounded_agents += 1

        sidecar.append({
            "output_index": i,
            "gold_row_index": i,
            "matches_hash": row["matches_hash"],
            "source_hash": row.get("source_hash"),
            "gold": row.get("gold"),
            "stmt_type": row.get("stmt_type"),
            "n_agents": len(agents_of(paj)),
            "n_grounded": sum(1 for a in agents_of(paj) if is_grounded(a)),
        })

    assert len(out_json) == len(rows) == 154, f"expected 154, got {len(out_json)}"
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out_json))
    Path(args.sidecar).write_text(json.dumps(sidecar, indent=2))

    print(
        f"built {len(out_json)}/154 GROUNDED statements -> {args.output}\n"
        f"sidecar -> {args.sidecar}\n"
        f"agent grounding: {grounded_agents}/{total_agents} "
        f"= {100*grounded_agents/total_agents:.1f}% (confound was 0/308)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
