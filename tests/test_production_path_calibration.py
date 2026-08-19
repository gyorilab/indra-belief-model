"""Calibration is fitted on the production path, not a parallel one.

WHY THIS EXISTS
---------------
A calibration is only valid for what it was fitted on, in two independent
senses, and until this arc only the first was addressed:

  PATH        which code produced the number. MEASURED on 300 identical
              evidences: the gold-eval path and the corpus shard path disagree
              on the VERDICT for 10% of them, their margins correlate at only
              r=0.874, and the SIGN differs on 30/290. The same shard path run
              twice agrees 300/300 with r=1.0000 and 290/290 identical margins
              -- so the divergence is the path, not server nondeterminism.

  POPULATION  which statements. `fit_prevalence` is baked into the artifact and
              anchors every weight (weight = logit(p_hat) - logit(fit_prevalence)),
              so a curve fitted on a balanced curated gold at 0.513 and applied
              to a corpus at 0.70 displaces every weight by +0.88 log-odds. The
              isotonic's knots sit where the FIT population's margins fell.

Byte-identical prompts do not fix either one. That was measured too, and it is
why "the prompts match, so gold-eval is fine" was the wrong conclusion.
"""
from __future__ import annotations

import gzip
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


builder = _load("_builder_tsv", "scripts/build_processed_grounding_shards.py")
fit = _load("_fit_prod", "scripts/fit_incall_calibration.py")


# ── the TSV that feeds the production builder ─────────────────────────────────

def _convert(tmp_path, corpus):
    """Round-trip a corpus through the builder's own adapter and reader.

    Deliberately asserts on what `iter_processed_rows` reads back rather than on
    the bytes written: the adapter exists only to feed that reader, so the
    contract between them is the thing worth pinning.
    """
    corpus_path = tmp_path / "corpus.json"
    corpus_path.write_text(json.dumps(corpus))
    out = tmp_path / "converted.tsv.gz"
    written = builder.corpus_json_to_tsv(corpus_path, out)
    return written, list(builder.iter_processed_rows(out))


def test_the_statement_hash_is_the_matches_hash(tmp_path):
    """Shards prepared from a labelled corpus must key exactly as shards
    prepared from the production dump, or the calibration is fitted against
    identities the corpus run does not use."""
    written, rows = _convert(tmp_path, [
        {"matches_hash": "123", "type": "Activation", "evidence": [{"text": "x"}]}])
    assert written == 1
    _, stmt_hash, _ = rows[0]
    assert stmt_hash == 123


def test_a_statement_without_a_matches_hash_is_skipped(tmp_path):
    written, rows = _convert(tmp_path, [
        {"type": "Activation", "evidence": [{"text": "x"}]},
        {"matches_hash": "7", "type": "Activation", "evidence": [{"text": "y"}]},
    ])
    assert written == 1 and len(rows) == 1


def test_a_corpus_with_no_usable_statement_refuses(tmp_path):
    corpus_path = tmp_path / "c.json"
    corpus_path.write_text(json.dumps([{"type": "Activation"}]))
    with pytest.raises(SystemExit, match="matches_hash"):
        builder.corpus_json_to_tsv(corpus_path, tmp_path / "o.tsv.gz")


def test_evidence_text_containing_tabs_survives_the_round_trip(tmp_path):
    """A raw tab or newline in the payload would split its own TSV row, and the
    reader would reject the line as having the wrong column count."""
    _, rows = _convert(tmp_path, [{
        "matches_hash": "1", "type": "Activation",
        "evidence": [{"text": "a\tb\nc"}],
    }])
    assert len(rows) == 1
    _, _, payload = rows[0]
    assert json.loads(payload)["evidence"][0]["text"] == "a\tb\nc"


# ── fitting from what the production path actually wrote ──────────────────────

@pytest.fixture
def shards(tmp_path):
    ind, outd = tmp_path / "in", tmp_path / "out"
    ind.mkdir(); outd.mkdir()
    jobs = [
        {"job_id": "0:0", "input_row_index": 0, "stmt_hash": 10, "source_hash": "s1",
         "source_api": "reach", "needs_llm": True, "tier1_result": None},
        {"job_id": "0:1", "input_row_index": 0, "stmt_hash": 10, "source_hash": "s2",
         "source_api": "reach", "needs_llm": True, "tier1_result": None},
        {"job_id": "1:0", "input_row_index": 1, "stmt_hash": 20, "source_hash": "s3",
         "source_api": "reach", "needs_llm": True, "tier1_result": None},
        {"job_id": "2:0", "input_row_index": 2, "stmt_hash": 30, "source_hash": "s4",
         "source_api": "reach", "needs_llm": True, "tier1_result": None},
    ]
    with gzip.open(ind / "grounded-000000.jsonl.gz", "wt") as fh:
        for job in jobs:
            fh.write(json.dumps(job) + "\n")
    with gzip.open(outd / "verdicts-000000.json.gz", "wt") as fh:
        json.dump({
            "10": {"s1": {"verdict": "correct", "probe_delta_logit": 8.0},
                   "s2": {"verdict": "incorrect"}},          # scored, no margin
            "20": {"s3": {"verdict": "correct", "probe_delta_logit": 2.0}},
            # s4 absent entirely: scored nothing
        }, fh)
    labels = tmp_path / "gold.jsonl"
    # Keyed on the PAIR. One Evidence can support several statements, so
    # source_hash alone is not a key -- see the conflicting-label test below.
    labels.write_text("".join(json.dumps(r) + "\n" for r in [
        {"matches_hash": "10", "source_hash": "s1", "gold": "correct"},
        {"matches_hash": "10", "source_hash": "s2", "gold": "incorrect"},
        {"matches_hash": "20", "source_hash": "s3", "gold": "incorrect"},
        {"matches_hash": "30", "source_hash": "s4", "gold": "correct"},
    ]))
    return ind, outd, labels


def test_rows_join_on_source_hash_not_position(shards):
    ind, outd, labels = shards
    rows, _ = fit.load_rows_from_shards(ind, outd, labels)
    by_id = {r["record_id"]: r for r in rows}
    assert by_id["10:s1"]["margin"] == 8.0 and by_id["10:s1"]["gold"] is True
    assert by_id["20:s3"]["margin"] == 2.0 and by_id["20:s3"]["gold"] is False


def test_a_scored_row_without_a_margin_is_counted_not_imputed(shards):
    ind, outd, labels = shards
    rows, skipped = fit.load_rows_from_shards(ind, outd, labels)
    assert "10:s2" not in {r["record_id"] for r in rows}
    assert skipped["no_margin"] == 1


def test_an_unscored_evidence_is_counted_separately_from_a_missing_label(shards):
    """Different failures with different remedies: one means the run did not
    cover it, the other means no curator did."""
    ind, outd, labels = shards
    _, skipped = fit.load_rows_from_shards(ind, outd, labels)
    assert skipped["unscored"] == 1          # s4 has a label, no scored cell
    assert skipped["no_gold"] == 0


def test_a_label_with_no_scored_row_never_becomes_a_row(shards):
    ind, outd, labels = shards
    rows, _ = fit.load_rows_from_shards(ind, outd, labels)
    assert "30:s4" not in {r["record_id"] for r in rows}


def test_gold_is_read_from_either_label_spelling(tmp_path, shards):
    """gold_correct (gold-eval) and gold (curation export) both appear in the
    tree; a loader that understood only one would silently fit on a subset."""
    ind, outd, _ = shards
    labels = tmp_path / "alt.jsonl"
    labels.write_text(json.dumps(
        {"matches_hash": "10", "source_hash": "s1", "gold_correct": True}) + "\n")
    rows, _ = fit.load_rows_from_shards(ind, outd, labels)
    assert [r["record_id"] for r in rows] == ["10:s1"] and rows[0]["gold"] is True


# ── the join key is the PAIR, and that is not a detail ────────────────────────
#
# MEASURED on eval_curation_v1: 1606 gold rows, 1604 distinct source_hash, and
# ONE of the two duplicates carries disagreeing labels. A dict keyed on
# source_hash resolves that curator disagreement by file order -- the same shape
# as the paired-by-file-order defect found in the v2 gold. The combiner also
# refuses duplicate record ids outright, so the bare key crashed the fit the
# first time a full corpus was run through it.

def test_the_same_evidence_under_two_statements_stays_two_rows(tmp_path):
    ind, outd = tmp_path / "in", tmp_path / "out"
    ind.mkdir(); outd.mkdir()
    with gzip.open(ind / "grounded-000000.jsonl.gz", "wt") as fh:
        for stmt in ("10", "20"):
            fh.write(json.dumps({
                "job_id": f"{stmt}:0", "input_row_index": 0, "stmt_hash": stmt,
                "source_hash": "shared", "source_api": "reach",
                "needs_llm": True, "tier1_result": None}) + "\n")
    with gzip.open(outd / "verdicts-000000.json.gz", "wt") as fh:
        json.dump({"10": {"shared": {"verdict": "correct", "probe_delta_logit": 5.0}},
                   "20": {"shared": {"verdict": "incorrect", "probe_delta_logit": -5.0}}}, fh)
    labels = tmp_path / "g.jsonl"
    labels.write_text("".join(json.dumps(r) + "\n" for r in [
        {"matches_hash": "10", "source_hash": "shared", "gold": "correct"},
        {"matches_hash": "20", "source_hash": "shared", "gold": "incorrect"},
    ]))
    rows, _ = fit.load_rows_from_shards(ind, outd, labels)
    assert {r["record_id"] for r in rows} == {"10:shared", "20:shared"}, (
        "one evidence supporting two statements collapsed to a single reading"
    )
    assert len({r["record_id"] for r in rows}) == len(rows), "ids must be unique"


def test_a_pair_whose_labels_conflict_is_dropped_not_decided_by_file_order(shards, tmp_path):
    ind, outd, _ = shards
    labels = tmp_path / "conflict.jsonl"
    labels.write_text("".join(json.dumps(r) + "\n" for r in [
        {"matches_hash": "10", "source_hash": "s1", "gold": "correct"},
        {"matches_hash": "10", "source_hash": "s1", "gold": "incorrect"},
    ]))
    rows, skipped = fit.load_rows_from_shards(ind, outd, labels)
    assert rows == [], "an unresolved curator disagreement was silently resolved"
    assert skipped["conflicting_labels"] == 1


def test_a_pair_repeated_in_the_corpus_is_read_once(shards, tmp_path):
    """The scoring path is deterministic (verified 290/290 identical margins
    across two runs), so a repeated pair carries no new information and would
    only weigh that evidence double in the fitted curve."""
    ind, outd, labels = shards
    with gzip.open(ind / "grounded-000001.jsonl.gz", "wt") as fh:
        fh.write(json.dumps({
            "job_id": "0:0", "input_row_index": 0, "stmt_hash": "10",
            "source_hash": "s1", "source_api": "reach",
            "needs_llm": True, "tier1_result": None}) + "\n")
    with gzip.open(outd / "verdicts-000001.json.gz", "wt") as fh:
        json.dump({"10": {"s1": {"verdict": "correct", "probe_delta_logit": 8.0}}}, fh)
    rows, skipped = fit.load_rows_from_shards(ind, outd, labels)
    assert [r["record_id"] for r in rows].count("10:s1") == 1
    assert skipped["duplicate_pair"] == 1
