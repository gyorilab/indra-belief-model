"""Streaming, filtering, cache, and atomic-shard invariants."""
from __future__ import annotations

import csv
import gzip
import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "build_processed_grounding_shards.py"
)
SPEC = importlib.util.spec_from_file_location(
    "build_processed_grounding_shards",
    SCRIPT,
)
assert SPEC and SPEC.loader
pipeline = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pipeline)


def test_processed_tsv_is_streamed_as_raw_statement_json(tmp_path):
    path = tmp_path / "processed.tsv.gz"
    with gzip.open(path, "wt", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t")
        for index in range(4):
            writer.writerow((100 + index, json.dumps({"type": f"T{index}"})))

    rows = list(pipeline.iter_processed_rows(path))

    assert rows == [
        (0, 100, '{"type": "T0"}'),
        (1, 101, '{"type": "T1"}'),
        (2, 102, '{"type": "T2"}'),
        (3, 103, '{"type": "T3"}'),
    ]


def test_empty_evidence_and_empty_text_are_filtered_separately():
    class Evidence:
        def __init__(self, text):
            self.text = text

    class Statement:
        def __init__(self, evidence):
            self.evidence = evidence

    selected, counts = pipeline.text_evidence_items(Statement([]))
    assert selected == []
    assert counts == {"statements_without_evidence": 1}

    no_text = Evidence(None)
    whitespace = Evidence("   ")
    supported = Evidence("supported sentence")
    selected, counts = pipeline.text_evidence_items(
        Statement([no_text, whitespace, supported])
    )
    assert selected == [(2, supported)]
    assert counts == {"evidences_without_text": 2}


@dataclass
class FakeEntity:
    name: str
    raw_text: str | None
    marker: int

    calls = 0

    @classmethod
    def resolve(cls, name, raw_text):
        cls.calls += 1
        return cls(name=name, raw_text=raw_text, marker=cls.calls)


def test_sqlite_grounding_cache_survives_reopen(tmp_path):
    cache_path = tmp_path / "gilda.sqlite3"
    FakeEntity.calls = 0

    cache = pipeline.GroundingCache(cache_path, "code-v1")
    first = cache.resolve_entity("MAPK1", "ERK2", FakeEntity)
    second = cache.resolve_entity("MAPK1", "ERK2", FakeEntity)
    cache.close()

    reopened = pipeline.GroundingCache(cache_path, "code-v1")
    third = reopened.resolve_entity("MAPK1", "ERK2", FakeEntity)
    reopened.close()

    assert first == second == third
    assert FakeEntity.calls == 1


def test_atomic_shard_is_only_published_on_commit(tmp_path):
    writer = pipeline.AtomicShardWriter(tmp_path, 3, compresslevel=1)
    writer.write({"job_id": "10:0"})

    assert not writer.final_path.exists()
    assert writer.tmp_path.exists()

    metadata = writer.commit()

    assert metadata == {
        "index": 3,
        "path": "grounded-000003.jsonl.gz",
        "jobs": 1,
    }
    assert writer.final_path.exists()
    assert not writer.tmp_path.exists()
    with gzip.open(writer.final_path, "rt") as fh:
        assert json.loads(fh.readline()) == {"job_id": "10:0"}
