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


emit = _load("_emit_tsv", "scripts/emit_statement_tsv.py")
fit = _load("_fit_prod", "scripts/fit_incall_calibration.py")


# ── the TSV that feeds the production builder ─────────────────────────────────

def test_the_statement_hash_is_the_matches_hash():
    """Shards prepared here must key exactly as shards from the real dump do."""
    rows, _ = emit.statement_rows([{"matches_hash": "123", "type": "Activation",
                                    "evidence": [{"text": "x"}]}])
    assert rows[0][0] == "123"


def test_a_statement_without_a_matches_hash_is_skipped_and_counted():
    rows, skipped = emit.statement_rows([{"type": "Activation", "evidence": [{}]}])
    assert rows == [] and skipped["no_matches_hash"] == 1


def test_the_emitted_json_cannot_split_its_own_row():
    """A raw tab or newline inside the payload would silently corrupt the TSV."""
    rows, _ = emit.statement_rows([{
        "matches_hash": "1", "type": "Activation",
        "evidence": [{"text": "a\tb\nc"}],
    }])
    payload = rows[0][1]
    assert "\t" not in payload and "\n" not in payload


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
    labels.write_text("".join(json.dumps(r) + "\n" for r in [
        {"source_hash": "s1", "gold": "correct"},
        {"source_hash": "s2", "gold": "incorrect"},
        {"source_hash": "s3", "gold": "incorrect"},
        {"source_hash": "s4", "gold": "correct"},
    ]))
    return ind, outd, labels


def test_rows_join_on_source_hash_not_position(shards):
    ind, outd, labels = shards
    rows, _ = fit.load_rows_from_shards(ind, outd, labels)
    by_id = {r["record_id"]: r for r in rows}
    assert by_id["s1"]["margin"] == 8.0 and by_id["s1"]["gold"] is True
    assert by_id["s3"]["margin"] == 2.0 and by_id["s3"]["gold"] is False


def test_a_scored_row_without_a_margin_is_counted_not_imputed(shards):
    ind, outd, labels = shards
    rows, skipped = fit.load_rows_from_shards(ind, outd, labels)
    assert "s2" not in {r["record_id"] for r in rows}
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
    assert "s4" not in {r["record_id"] for r in rows}


def test_gold_is_read_from_either_label_spelling(tmp_path, shards):
    """gold_correct (gold-eval) and gold (curation export) both appear in the
    tree; a loader that understood only one would silently fit on a subset."""
    ind, outd, _ = shards
    labels = tmp_path / "alt.jsonl"
    labels.write_text(json.dumps({"source_hash": "s1", "gold_correct": True}) + "\n")
    rows, _ = fit.load_rows_from_shards(ind, outd, labels)
    assert [r["record_id"] for r in rows] == ["s1"] and rows[0]["gold"] is True
