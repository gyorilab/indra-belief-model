"""Tests for the simplified processed-shard vLLM runner."""
from __future__ import annotations

import gzip
import importlib.util
import json
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_vllm_processed_shards.py"
SPEC = importlib.util.spec_from_file_location("run_vllm_processed_shards", SCRIPT)
assert SPEC and SPEC.loader
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def write_jobs(path: Path, jobs: list[dict]) -> None:
    with gzip.open(path, "wt") as fh:
        for job in jobs:
            fh.write(json.dumps(job) + "\n")


def test_limited_output_has_separate_name(tmp_path):
    final, partial = runner.output_paths(tmp_path, 12, 200)
    assert final.name == "verdicts-000012.limit-200.json.gz"
    assert partial.name == ".verdicts-000012.limit-200.partial.jsonl"


def test_partial_resume_uses_latest_valid_attempt_and_ignores_truncation(tmp_path):
    path = tmp_path / "partial.jsonl"
    path.write_text(
        '{"job_id":"1:0","verdict":null}\n'
        '{"job_id":"1:0","verdict":"correct","confidence":"high"}\n'
        '{"job_id":'
    )
    latest = runner.load_partial(path)
    assert latest["1:0"]["verdict"] == "correct"


def test_atomic_final_is_requested_dictionary(tmp_path):
    path = tmp_path / "verdicts.json.gz"
    payload = {
        "101": {"11": {"verdict": "correct", "confidence": "high"}}
    }
    runner.write_final_atomic(path, payload)
    with gzip.open(path, "rt") as fh:
        assert json.load(fh) == payload
    assert not (tmp_path / ".verdicts.json.gz.tmp").exists()


def test_finalize_builds_hash_dictionary(tmp_path):
    shard = tmp_path / "grounded-000000.jsonl.gz"
    write_jobs(
        shard,
        [
            {"job_id": "1:0", "stmt_hash": 101, "source_hash": 11},
            {"job_id": "2:0", "stmt_hash": -202, "source_hash": 22},
        ],
    )
    latest = {
        "1:0": {"verdict": "correct", "confidence": "high"},
        "2:0": {"verdict": "incorrect", "confidence": "medium"},
    }
    payload, missing = runner.finalize(shard, latest, None)
    assert payload == {
        "101": {"11": {"verdict": "correct", "confidence": "high"}},
        "-202": {
            "22": {"verdict": "incorrect", "confidence": "medium"}
        },
    }
    assert missing == []


def test_finalize_keeps_multiple_evidences_for_one_statement(tmp_path):
    shard = tmp_path / "grounded-000000.jsonl.gz"
    write_jobs(
        shard,
        [
            {"job_id": "1:0", "stmt_hash": 101, "source_hash": 11},
            {"job_id": "2:0", "stmt_hash": 101, "source_hash": 22},
        ],
    )
    latest = {
        "1:0": {"verdict": "incorrect", "confidence": "low"},
        "2:0": {"verdict": "correct", "confidence": "high"},
    }

    payload, missing = runner.finalize(shard, latest, None)

    assert payload == {
        "101": {
            "11": {"verdict": "incorrect", "confidence": "low"},
            "22": {"verdict": "correct", "confidence": "high"},
        }
    }
    assert missing == []


def test_tier1_does_not_call_model():
    class FailIfCalled:
        def post(self, *_args, **_kwargs):
            raise AssertionError("Tier 1 must not call vLLM")

    row = runner.score_job(
        {
            "job_id": "1:0",
            "stmt_hash": 101,
            "source_hash": 11,
            "needs_llm": False,
            "tier1_result": {"verdict": "incorrect", "confidence": "high"},
        },
        client=FailIfCalled(),
        prompt=object(),
        endpoint="unused",
        model_id="unused",
        max_tokens=1000,
        temperature=0.1,
    )
    assert row["source"] == "tier1"
    assert row["verdict"] == "incorrect"
