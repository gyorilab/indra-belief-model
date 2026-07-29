#!/usr/bin/env python3
"""Streaming uniform reservoir sample of CoGEx Evidence nodes — the corpus-
representative draw the /curate agent-pool sampling can't give.

nodes_Evidence.tsv.gz is one row per evidence of a GROUNDED, assembled statement
(EvidenceProcessor in indra_cogex). This reads it on stdin in ONE pass, keeps a
uniform random N rows (Algorithm R, seeded → reproducible), parses each evidence's
JSON column, and writes a JSONL of grounded evidences ready to curate/score:
  {stmt_hash, source_api, source_hash, pmid, text, evidence:<full json>}

Never holds the whole 4.87 GB — decompresses on the fly, keeps only N rows.
The checked-in representative-pool provenance was produced with N=5,000 from
the 44,944,056 evidence rows in the frozen 2025-09-16 dump.  The row-count
check is intentional: a truncated or different stdin must not silently claim
the identity of that frozen population.

    # via presigned URL (no creds needed on our end):
    curl -s '<presigned-url>' | python scripts/reservoir_sample_cogex.py
    # or with S3 creds configured:
    aws s3 cp s3://bigmech/indra-db/dumps/cogex_files/20250916/nodes_Evidence.tsv.gz - \
        | python scripts/reservoir_sample_cogex.py
"""
from __future__ import annotations

import csv
import gzip
import io
import json
import os
import random
import sys

DEFAULT_SAMPLE_N = 5_000
SOURCE_POPULATION_ROWS = 44_944_056

N = int(os.environ.get("SAMPLE_N", str(DEFAULT_SAMPLE_N)))
SEED = int(os.environ.get("SAMPLE_SEED", "20260701"))
OUT = os.environ.get("SAMPLE_OUT", "data/corpora/cogex_evidence_sample.jsonl")
# Override only when deliberately sampling a different source population.
EXPECTED_ROWS = int(os.environ.get("SAMPLE_EXPECTED_ROWS", str(SOURCE_POPULATION_ROWS)))


def open_stdin_text():
    """Read stdin as text, transparently gunzipping if it's gzip (magic 1f 8b)."""
    raw = sys.stdin.buffer
    head = raw.peek(2)[:2] if hasattr(raw, "peek") else b""
    if head == b"\x1f\x8b":
        return gzip.open(raw, "rt", encoding="utf-8", errors="replace")
    return io.TextIOWrapper(raw, encoding="utf-8", errors="replace")


def find_cols(header: list[str]) -> dict[str, int]:
    """Match the Neo4j-import header by base name (e.g. 'evidence:string')."""
    want = {"evidence", "stmt_hash", "source_api"}
    return {h.split(":")[0].strip().lower(): i for i, h in enumerate(header)
            if h.split(":")[0].strip().lower() in want}


def main() -> int:
    csv.field_size_limit(1 << 27)  # evidence JSON rows can be large
    reader = csv.reader(open_stdin_text(), delimiter="\t")
    header = next(reader)
    cols = find_cols(header)
    if "evidence" not in cols or "stmt_hash" not in cols:
        sys.exit(f"missing evidence/stmt_hash columns; header was: {header}")

    rng = random.Random(SEED)
    reservoir: list[list[str]] = []
    seen = 0
    for row in reader:
        seen += 1
        if len(reservoir) < N:
            reservoir.append(row)
        else:
            j = rng.randint(0, seen - 1)  # Algorithm R: uniform over all rows seen
            if j < N:
                reservoir[j] = row
        if seen % 5_000_000 == 0:
            print(f"  scanned {seen:,} evidences…", file=sys.stderr, flush=True)

    if seen != EXPECTED_ROWS:
        sys.exit(
            f"source population row-count mismatch: scanned {seen:,}, "
            f"expected {EXPECTED_ROWS:,}; set SAMPLE_EXPECTED_ROWS only for a "
            "deliberately different source dump"
        )

    os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
    written = 0
    with open(OUT, "w") as f:
        for row in reservoir:
            try:
                ev = json.loads(row[cols["evidence"]])
                stmt_hash = int(row[cols["stmt_hash"]])
            except (json.JSONDecodeError, IndexError, ValueError):
                continue
            tr = ev.get("text_refs") or {}
            f.write(json.dumps({
                "stmt_hash": stmt_hash,
                "source_api": ev.get("source_api"),
                "source_hash": ev.get("source_hash"),  # may be absent → recompute via INDRA downstream
                "pmid": ev.get("pmid") or tr.get("PMID"),
                "text": ev.get("text"),
                "evidence": ev,
            }) + "\n")
            written += 1

    print(f"scanned {seen:,} evidences → sampled {written} (N={N}, seed={SEED}) → {OUT}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
