"""Streaming, filtering, cache, and atomic-shard invariants."""
from __future__ import annotations

import csv
import gzip
import importlib.util
import json
import sys
import types
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
    assert selected == []
    assert counts == {"evidences_without_text": 1}

    selected, counts = pipeline.text_evidence_items(
        Statement([supported, whitespace])
    )
    assert selected == [(0, supported)]
    assert counts == {}


def test_all_evidence_keeps_each_text_bearing_evidence():
    class Evidence:
        def __init__(self, text):
            self.text = text

    class Statement:
        def __init__(self, evidence):
            self.evidence = evidence

    first = Evidence("first sentence")
    missing = Evidence(None)
    third = Evidence("third sentence")

    selected, counts = pipeline.text_evidence_items(
        Statement([first, missing, third]),
        all_evidence=True,
    )

    assert selected == [(0, first), (2, third)]
    assert counts == {"evidences_without_text": 1}


def test_entity_inputs_use_agent_text_db_refs(monkeypatch):
    statements_module = types.ModuleType("indra.statements")

    class SelfModification:
        pass

    statements_module.SelfModification = SelfModification
    indra_module = types.ModuleType("indra")
    indra_module.statements = statements_module
    monkeypatch.setitem(sys.modules, "indra", indra_module)
    monkeypatch.setitem(sys.modules, "indra.statements", statements_module)

    @dataclass
    class Agent:
        name: str
        db_refs: dict

    class Statement:
        def __init__(self, agents):
            self.agents = agents

        def agent_list(self):
            return self.agents

    statement = Statement([
        Agent("MAPK1", db_refs={"TEXT": "ERK2", "HGNC": "6871"}),
        Agent("ELK1", db_refs={"TEXT": "ELK-1", "HGNC": "3320"}),
    ])

    subject, obj = pipeline._entity_inputs(statement)

    assert subject == ("MAPK1", "ERK2")
    assert obj == ("ELK1", "ELK-1")


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


def test_a_committed_shard_is_durable_before_the_manifest_that_claims_it(
    tmp_path, monkeypatch
):
    """The resume manifest is ~2KB; the shard it accounts for is ~9MB.

    `commit_current_shard` renames the shard and then writes the manifest, and
    neither was fsynced. On a node crash between them the small file's data
    reaches disk first, so `input_rows_consumed` is durable while the shard's
    bytes are zeroes -- and `--resume` starts past those ~50,000 evidences for
    good. They are not recoverable by rerunning: they are simply absent from the
    corpus, and the identity check compares configuration, not shard contents.
    """
    events: list[str] = []
    real_fsync = pipeline.os.fsync
    real_replace = Path.replace

    def record_fsync(fd):
        events.append("fsync")
        return real_fsync(fd)

    def record_replace(self, target):
        events.append("rename")
        return real_replace(self, target)

    monkeypatch.setattr(pipeline.os, "fsync", record_fsync)
    monkeypatch.setattr(Path, "replace", record_replace)

    writer = pipeline.AtomicShardWriter(tmp_path, 4, compresslevel=1)
    writer.write({"job_id": "10:0"})
    writer.commit()
    assert events[:2] == ["fsync", "rename"], (
        f"the shard was published before its bytes were durable: {events}"
    )

    events.clear()
    pipeline._write_json_atomic(tmp_path / "manifest.json", {"shards": []})
    assert events[:2] == ["fsync", "rename"], (
        f"the manifest was published before its bytes were durable: {events}"
    )


def _empty_corpus_run(tmp_path, monkeypatch, *extra):
    """Drive main() over an empty input, which exercises identity and nothing else."""
    indra_db = types.ModuleType("indra_db")
    dumping = types.ModuleType("indra_db.readonly_dumping")
    util = types.ModuleType("indra_db.readonly_dumping.util")
    util.clean_json_loads = json.loads
    dumping.util = util
    indra_db.readonly_dumping = dumping
    monkeypatch.setitem(sys.modules, "indra_db", indra_db)
    monkeypatch.setitem(sys.modules, "indra_db.readonly_dumping", dumping)
    monkeypatch.setitem(sys.modules, "indra_db.readonly_dumping.util", util)

    source = tmp_path / "processed.tsv.gz"
    if not source.exists():
        with gzip.open(source, "wt"):
            pass
    monkeypatch.setattr(sys, "argv", [
        "build_processed_grounding_shards.py",
        "--input", str(source),
        "--output-dir", str(tmp_path / "prepared"),
        *extra,
    ])
    return pipeline.main()


def test_the_fingerprint_states_no_digest_that_writing_it_would_falsify():
    """A file cannot quote its own hash.

    The docstring carried a MEASURED pair, "7f6ede4cad6ac7d2 ->
    113841f047f6a030", to show that a durability-only edit moves the digest. The
    left-hand value was real; the right-hand one described no tree that ever
    existed, because the digest covers this file's OWN bytes and writing the
    result into the docstring changes it. Re-measuring and pasting today's value
    reproduces the defect on the next edit -- the number is stale the moment
    anyone touches the file it measures, and this codebase treats a wrong number
    in a comment as a defect.

    So the shape is pinned instead of the value: no 16-hex-digit literal in
    either docstring. A digest that must be current belongs in a manifest, which
    is where the pipeline already writes it.
    """
    import re

    for name, doc in (
        ("_code_fingerprint", pipeline._code_fingerprint.__doc__),
        ("test_a_durability_only_edit_does_not_have_to_orphan_a_prepared_corpus",
         test_a_durability_only_edit_does_not_have_to_orphan_a_prepared_corpus
         .__doc__),
    ):
        assert not re.search(r"\b[0-9a-f]{16}\b", doc), (
            f"{name} quotes a digest of the file it lives in; it is false as "
            "soon as it is true"
        )


def test_a_durability_only_edit_does_not_have_to_orphan_a_prepared_corpus(
    tmp_path, monkeypatch
):
    """`_code_fingerprint` hashes bytes, and the fsyncs changed the bytes.

    A digest over the four pinned paths moves for a change that cannot alter a
    single shard byte -- adding an fsync is the case this was written for. No
    before/after pair is quoted (see the test above: the hash covers the file
    the docstring lives in). Both consequences are silent-to-fatal: `--resume` dies with "resume configuration/input mismatch"
    and the alternative branch refuses too ("has shards but no manifest"), so a
    partially-prepared 60M corpus has no path forward; and `GroundingCache` is
    namespaced on the same digest, so the entire persistent Gilda cache is
    discarded and every lookup recomputed.
    """
    import pytest as _pytest

    namespaces: list[str] = []
    real_cache = pipeline.GroundingCache
    monkeypatch.setattr(
        pipeline, "GroundingCache",
        lambda path, namespace: namespaces.append(namespace) or real_cache(
            path, namespace),
    )

    assert _empty_corpus_run(tmp_path, monkeypatch) == 0
    manifest_path = tmp_path / "prepared" / "manifest.json"
    recorded = json.loads(manifest_path.read_text())["preparation_code_sha256"]

    # the fsync-only edit, in the only way the digest can see it
    monkeypatch.setattr(pipeline, "_code_fingerprint", lambda: "0" * 64)

    with _pytest.raises(SystemExit) as excinfo:
        _empty_corpus_run(tmp_path, monkeypatch)
    message = str(excinfo.value)
    assert "preparation_code_sha256" in message
    assert f"--accept-preparation-code {recorded}" in message, (
        f"the refusal names no way forward: {message}"
    )

    assert _empty_corpus_run(
        tmp_path, monkeypatch, "--accept-preparation-code", recorded) == 0
    manifest = json.loads(manifest_path.read_text())
    assert manifest["preparation_code_sha256"] == recorded
    assert manifest["preparation_code_accepted"][-1]["running"] == "0" * 64
    assert namespaces[-1] == recorded, (
        "the accepted resume renamespaced the Gilda cache, discarding it"
    )

    # an assertion about the WRONG digest is not an assertion
    with _pytest.raises(SystemExit):
        _empty_corpus_run(tmp_path, monkeypatch,
                          "--accept-preparation-code", "1" * 64)


# ── the job the whole corpus path is built on ─────────────────────────────────
#
# Everything above tests the machinery AROUND job preparation -- streaming,
# filtering, the cache, atomic publication -- and nothing called
# `prepare_statement_jobs` itself. So when a10df62 ("one semantic kernel for the
# live and batch scoring paths") replaced `ScoringRecord.format_user_message`
# with `execution_body()` + `ExecutionBody.render()` and did not move this
# builder onto it, the builder raised AttributeError for EVERY LLM-bound job and
# the suite stayed green.
#
# It stayed hidden because prepared shards already on disk were built before the
# refactor and still scored fine; the break only surfaces when someone
# regenerates them -- which is exactly what fitting a new serving stack's
# calibration requires.

import pytest  # noqa: E402

indra_statements = pytest.importorskip("indra.statements")


def _statement():
    Activation = indra_statements.Activation
    Agent = indra_statements.Agent
    Evidence = indra_statements.Evidence
    stmt = Activation(Agent("IL18", db_refs={"HGNC": "5986"}),
                      Agent("RAF1", db_refs={"HGNC": "9829"}))
    evidence = Evidence(source_api="reach", pmid="1",
                        text="IL-18 activates Raf-1 in NK cells.")
    stmt.evidence = [evidence]
    return stmt, evidence


def _jobs(tmp_path):
    stmt, evidence = _statement()
    cache = pipeline.GroundingCache(tmp_path / "cache.sqlite", "test-sha")
    out = pipeline.prepare_statement_jobs(
        statement=stmt, stmt_hash=1, input_row_index=0,
        cache=cache, selected_evidence=[(0, evidence)],
    )
    return (out[0] if isinstance(out, tuple) else out), stmt, evidence


def test_prepare_statement_jobs_actually_builds_a_job(tmp_path):
    """The regression. This is the call the corpus path cannot run without."""
    jobs, _, _ = _jobs(tmp_path)
    assert jobs, "no job was produced for an LLM-bound statement"
    assert jobs[0].get("user_message"), (
        "the job carries no user_message; the shard runner raises "
        "'LLM job has no user_message' on every row built this way"
    )


def test_the_batch_user_message_is_byte_identical_to_the_live_one(tmp_path):
    """The invariant a10df62 existed to create, asserted where it broke.

    The record owns the PARTS and `ExecutionBody.render` owns the JOIN precisely
    so the live and batch paths cannot drift. Nothing checked that the batch
    builder used it -- and it did not. A drift here is not cosmetic: a
    calibration profile is keyed on the prompt, so a batch path sending even one
    byte more would be scored against a profile fitted for a prompt it never
    sent.
    """
    from indra_belief.data.scoring_record import ScoringRecord

    jobs, stmt, evidence = _jobs(tmp_path)
    record = ScoringRecord(statement=stmt, evidence=evidence)
    record.resolve_entities()
    assert jobs[0]["user_message"] == record.execution_body().render(), (
        "the batch builder's user message has drifted from the live render"
    )


def test_the_builder_uses_the_shared_join_not_a_local_copy():
    """Structural, so a reintroduced local join fails even if it happens to
    produce the same bytes today."""
    source = SCRIPT.read_text()
    assert "execution_body()" in source
    # A CALL, not a mention: the comment at the call site names the deleted
    # method to explain the history, and a bare substring check flagged that
    # explanation as the defect it was documenting.
    assert ".format_user_message(" not in source, (
        "the builder is calling the deleted method again"
    )
