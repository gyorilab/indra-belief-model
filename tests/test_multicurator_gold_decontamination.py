"""De-contamination must key on the identity the pipeline joins on.

`excl_pairs` read `r.get("pa_hash", r.get("matches_hash"))` -- preferring
pa_hash, falling back to matches_hash. But everything downstream joins on
matches_hash, and MEASURED on eval_curation_v1 the two identities DIFFER for 105
of 1606 rows. Those pairs were therefore excluded under a name nothing else
uses, leaving them eligible for a newly built gold. One of them duly appeared in
the first build of external_curator_gold_v2.

De-contamination that keys on a different identity than the join is not
de-contamination, and it fails silently: the guard prints CLEAN because it is
checking the same wrong key.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location(
        "_gold_builder", ROOT / "scripts/build_multicurator_gold.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


builder = pytest.importorskip("indra") and _load()


def test_a_row_whose_two_hashes_differ_is_excluded_under_both(tmp_path, monkeypatch):
    bench = tmp_path / "bench"
    bench.mkdir()
    (bench / "eval_curation_probe.jsonl").write_text(json.dumps({
        "pa_hash": 111, "matches_hash": 222, "source_hash": 999,
    }) + "\n")
    monkeypatch.setattr(builder, "BENCH", bench)
    excluded = builder.excl_pairs(bench / "not-this-one.jsonl")
    assert (111, 999) in excluded, "the pa_hash identity was dropped"
    assert (222, 999) in excluded, (
        "the matches_hash identity was dropped — that is the one the pipeline "
        "joins on, so the pair stays eligible for a new gold"
    )


def test_the_output_being_written_is_still_skipped(tmp_path, monkeypatch):
    """Rebuilding a gold must not treat its own previous output as prior
    contamination, or the second build would be empty."""
    bench = tmp_path / "bench"
    bench.mkdir()
    own = bench / "external_curator_gold_v9.jsonl"
    own.write_text(json.dumps({"matches_hash": 5, "source_hash": 6}) + "\n")
    monkeypatch.setattr(builder, "BENCH", bench)
    assert (5, 6) not in builder.excl_pairs(own)
    assert (5, 6) in builder.excl_pairs(bench / "other.jsonl")


def test_output_paths_are_derived_from_the_name():
    gold, stmts = builder.output_paths("some_gold_v3")
    assert gold.name == "some_gold_v3.jsonl"
    assert stmts.name == "some_gold_v3_statements.json"
    assert gold.parent.name == "benchmark" and stmts.parent.name == "corpora"


@pytest.mark.skipif(
    not (ROOT / "data/benchmark/external_curator_gold_v2.jsonl").exists(),
    reason="needs the built gold",
)
def test_the_shipped_v2_is_disjoint_from_every_prior_set():
    """The property the fit depends on: v2 is the fit set and v1 plus
    eval_curation_v1 are independent validation, so a leak would make the
    validation self-referential."""
    def pairs(path):
        out = set()
        for line in path.open():
            row = json.loads(line)
            source = row.get("source_hash")
            if source is None:
                continue
            for stmt in (row.get("pa_hash"), row.get("matches_hash")):
                if stmt is not None:
                    out.add((str(stmt), str(source)))
        return out

    bench = ROOT / "data/benchmark"
    v2 = pairs(bench / "external_curator_gold_v2.jsonl")
    assert len(v2) > 500
    for other in ("external_curator_gold_v1", "eval_curation_v1", "holdout_large",
                  "archive/fewshot_pool"):
        path = bench / f"{other}.jsonl"
        if path.exists():
            assert not (v2 & pairs(path)), f"v2 overlaps {other}"
