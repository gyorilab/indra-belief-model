#!/usr/bin/env python3
"""Preprocess the rasmachine inline-statements corpus into a curate-safe JSONL pool.

The /curate viewer samples a random (statement, evidence) pair from an INLINE
dataset entirely server-side (no network fetch). Its sampler JSON.parse's one line
at a time — and JS Number CANNOT hold a 64-bit INDRA hash: JSON.parse silently
rounds a bare integer past Number.MAX_SAFE_INTEGER (e.g. 325574632115105161 ->
325574632115105150), which would key a curation SUBMIT to the WRONG evidence.

Python's json reads those ints EXACTLY (arbitrary precision). This script reads the
raw corpus and re-emits every statement as one compact JSONL line where every
load-bearing hash is a QUOTED STRING (safe to JSON.parse in the viewer):
  - stmt['matches_hash']         -> str(...)   (already a string in the source)
  - each ev['source_hash']       -> str(...)   (a bare int in the source — the fix)

It also slims each record to only the fields renderStatement/evidenceAgents read,
which drops the nested big-int footguns (annotations.prior_hash,
annotations.indranet_edge.stmt_hash, epistemics) that would otherwise round on the
viewer's JSON.parse (harmless, since unused — but slim is cleaner and smaller).

Output: data/corpora/rasmachine_curate_pool.jsonl (gitignored, a reproducible
build artifact — regenerate on deploy, do NOT force-add).

Usage:  python scripts/build_curate_pool.py
"""

from __future__ import annotations

import json
import os

# statement-level keys the viewer's renderer reads (agent slots + modifiers).
# Every agent slot value is an agent dict {name, db_refs, ...}; db_refs/text_refs
# are all strings, so nothing at this level carries a 64-bit int to round.
STMT_KEEP = (
    "type",
    "belief",
    "matches_hash",
    "residue",
    "position",
    # agent slots (renderStatement / evidenceAgents)
    "enz",
    "sub",
    "subj",
    "obj",
    "members",
    "gef",
    "ras",
    "gap",
    "agent",
    "sub_obj",
    "obj_to",
)

SRC = os.path.join("data", "corpora", "latest_statements_rasmachine.json")
OUT = os.path.join("data", "corpora", "rasmachine_curate_pool.jsonl")


def slim_agent(a):
    """Keep only what agentName/dbRefs read: name + db_refs (all string values)."""
    if not isinstance(a, dict):
        return a
    out = {}
    if a.get("name") is not None:
        out["name"] = a["name"]
    if isinstance(a.get("db_refs"), dict):
        out["db_refs"] = a["db_refs"]
    return out


def slim_slot(v):
    if isinstance(v, list):
        return [slim_agent(x) for x in v]
    return slim_agent(v)


def slim_evidence(ev):
    """Keep source_api/pmid/text/text_refs + STRING source_hash + agents.raw_text."""
    out = {
        "source_api": ev.get("source_api"),
        "pmid": ev.get("pmid"),
        "text": ev.get("text"),
        "text_refs": ev.get("text_refs") or {},
        # THE FIX: bare 64-bit int -> exact-digit string (never a JS Number)
        "source_hash": str(ev["source_hash"]) if ev.get("source_hash") is not None else None,
    }
    ann = ev.get("annotations") or {}
    agents = ann.get("agents") or {}
    raw = agents.get("raw_text")
    if raw is not None:
        out["annotations"] = {"agents": {"raw_text": raw}}
    return out


def main() -> None:
    with open(SRC, encoding="utf-8") as fh:
        data = json.load(fh)  # arbitrary-precision ints — hashes are exact here

    written = 0
    skipped_no_ev = 0
    with open(OUT, "w", encoding="utf-8") as out:
        for stmt in data:
            evs = stmt.get("evidence") or []
            if not evs:
                skipped_no_ev += 1
                continue
            rec = {}
            for k in STMT_KEEP:
                if k not in stmt:
                    continue
                if k in ("enz", "sub", "subj", "obj", "members", "gef", "ras", "gap", "agent", "sub_obj", "obj_to"):
                    rec[k] = slim_slot(stmt[k])
                elif k == "matches_hash":
                    rec[k] = str(stmt[k])  # idempotent: already a string in source
                else:
                    rec[k] = stmt[k]
            rec["evidence"] = [slim_evidence(ev) for ev in evs]
            out.write(json.dumps(rec, ensure_ascii=False, separators=(",", ":")))
            out.write("\n")
            written += 1

    size_mb = os.path.getsize(OUT) / 1e6
    print(f"read {len(data)} statements from {SRC}")
    print(f"wrote {written} statements to {OUT} ({size_mb:.2f} MB)")
    print(f"skipped {skipped_no_ev} statements with no evidence")


if __name__ == "__main__":
    main()
