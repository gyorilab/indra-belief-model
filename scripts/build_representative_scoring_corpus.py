#!/usr/bin/env python3
"""Materialize the frozen representative-curation gold as a blind scorer corpus.

The curation artifact embeds each statement without evidence.  The original,
full Evidence JSON is recovered from the exact CoGEx reservoir pair so
``source_id`` and reader metadata are preserved; reconstructing from text alone
changes the source hash for BEL and SIGNOR rows.

The builder fails closed unless every gold row is unique, every pair is present
in the reservoir, and INDRA round-tripping reproduces both hashes exactly.

Example::

    PYTHONPATH=src .venv/bin/python scripts/build_representative_scoring_corpus.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from indra.statements import stmts_from_json  # noqa: E402

DEFAULT_GOLD = ROOT / "data/benchmark/representative_indra_curations_400.jsonl"
DEFAULT_POOL = ROOT / "data/corpora/cogex_evidence_sample.jsonl"
DEFAULT_OUTPUT = (
    ROOT / "data/corpora/representative_indra_expanded_403_20260717_statements.json"
)
MASK = (1 << 64) - 1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open() if line.strip()]


def _pair(matches_hash: Any, source_hash: Any) -> tuple[int, int]:
    return int(matches_hash), int(source_hash)


def build_corpus(
    gold_path: Path,
    pool_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    gold = _load_jsonl(gold_path)
    pool = _load_jsonl(pool_path)

    gold_keys = [_pair(r["matches_hash"], r["source_hash"]) for r in gold]
    if len(set(gold_keys)) != len(gold_keys):
        duplicates = [k for k, n in Counter(gold_keys).items() if n > 1]
        raise ValueError(f"gold contains duplicate exact pairs: {duplicates[:5]}")

    pool_by_pair: dict[tuple[int, int], dict[str, Any]] = {}
    for row in pool:
        key = _pair(row["stmt_hash"], row["source_hash"])
        if key in pool_by_pair:
            raise ValueError(f"reservoir contains duplicate exact pair: {key}")
        pool_by_pair[key] = row

    statements_json: list[dict[str, Any]] = []
    source_api_counts: Counter[str] = Counter()
    for row, key in zip(gold, gold_keys, strict=True):
        pool_row = pool_by_pair.get(key)
        if pool_row is None:
            raise ValueError(f"gold pair absent from reservoir: {key}")

        evidence = pool_row.get("evidence")
        if not isinstance(evidence, dict):
            raise ValueError(f"reservoir pair has no full Evidence JSON: {key}")
        if (pool_row.get("text") or "") != (row.get("evidence_text") or ""):
            raise ValueError(f"evidence text differs for pair: {key}")
        if (pool_row.get("source_api") or "") != (row.get("source_api") or ""):
            raise ValueError(f"source_api differs for pair: {key}")

        statement = deepcopy(row.get("statement"))
        if not isinstance(statement, dict):
            raise ValueError(f"gold pair has no embedded statement: {key}")
        embedded_hash = statement.get("matches_hash")
        if embedded_hash is not None and int(embedded_hash) != key[0]:
            raise ValueError(f"embedded statement hash differs for pair: {key}")
        statement["matches_hash"] = str(key[0])
        statement["evidence"] = [deepcopy(evidence)]
        statements_json.append(statement)
        source_api_counts[str(row.get("source_api") or "unknown")] += 1

    statements = stmts_from_json(statements_json)
    if len(statements) != len(gold):
        raise ValueError(
            f"INDRA deserialized {len(statements)} statements for {len(gold)} gold rows"
        )
    for index, (row, statement) in enumerate(zip(gold, statements, strict=True)):
        expected = _pair(row["matches_hash"], row["source_hash"])
        observed = (
            int(statement.get_hash(shallow=True)),
            int(statement.evidence[0].get_source_hash()),
        )
        if (observed[0] & MASK, observed[1] & MASK) != (
            expected[0] & MASK,
            expected[1] & MASK,
        ):
            raise ValueError(
                f"hash round-trip failed at row {index}: expected={expected}, observed={observed}"
            )

    meta = {
        "schema_version": 1,
        "purpose": "blind monolithic scoring corpus for the expanded representative INDRA set",
        "gold": {"path": str(gold_path.relative_to(ROOT)), "sha256": _sha256(gold_path)},
        "reservoir": {"path": str(pool_path.relative_to(ROOT)), "sha256": _sha256(pool_path)},
        "counts": {
            "gold_rows": len(gold),
            "unique_exact_pairs": len(set(gold_keys)),
            "statements": len(statements_json),
            "evidences": len(statements_json),
            "gold_correct": sum(r.get("gold") == "correct" for r in gold),
            "gold_incorrect": sum(r.get("gold") == "incorrect" for r in gold),
        },
        "source_api_counts": dict(sorted(source_api_counts.items())),
        "validation": {
            "statement_hashes_reproduced": len(statements_json),
            "source_hashes_reproduced": len(statements_json),
            "exact_pair_join": True,
        },
    }
    return statements_json, meta


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--pool", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    statements_json, meta = build_corpus(args.gold, args.pool)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(statements_json, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    )
    meta["output"] = {
        "path": str(args.output.relative_to(ROOT)),
        "sha256": _sha256(args.output),
    }
    meta_path = args.output.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")

    counts = meta["counts"]
    print(
        f"wrote {counts['statements']} statements / {counts['evidences']} evidences "
        f"({counts['gold_correct']} correct, {counts['gold_incorrect']} incorrect)"
    )
    print(f"corpus: {args.output}")
    print(f"sha256: {meta['output']['sha256']}")
    print(f"meta:   {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
