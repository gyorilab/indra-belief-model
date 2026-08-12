"""Tests for the simplified processed-shard vLLM runner."""
from __future__ import annotations

import gzip
import importlib.util
import json
import sys
import types
from concurrent.futures import ThreadPoolExecutor
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


def test_unparseable_response_keeps_diagnostic_preview():
    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "choices": [
                    {
                        "message": {"content": "I cannot decide."},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"completion_tokens": 7},
            }

    class Client:
        def post(self, *_args, **_kwargs):
            return Response()

    class Prompt:
        def request(self, _job):
            return "system", [{"role": "user", "content": "question"}]

        def parse(self, _content, _reasoning):
            return None, None

    row = runner.score_job(
        {
            "job_id": "1:0",
            "stmt_hash": 101,
            "source_hash": 11,
            "needs_llm": True,
        },
        client=Client(),
        prompt=Prompt(),
        endpoint="unused",
        model_id="unused",
        max_tokens=1000,
        temperature=0.1,
    )

    assert row["error"] == "unparseable model response"
    assert row["finish_reason"] == "stop"
    assert row["completion_tokens"] == 7
    assert "I cannot decide" in row["response_preview"]


def test_parser_accepts_gemma_plain_field_lines():
    """A real gemma reply: the disconfirm fields, one per line, no punctuation.

    The fixture text is unchanged from the scale_up branch — it is an observed
    reply from the vLLM-served model this runner drives, which is what makes it
    worth keeping. What changed is WHERE it is read: `_prompts.extract_verdict`
    no longer exists, because `indra_belief.verdict` became the single parser for
    the live path and the batch replay alike. The two patterns that make this
    reply readable came from this branch and now live there.
    """
    from indra_belief.verdict import parse_verdict

    text = (
        "relation_check way to a biological process is licensed\n"
        'support "Heart ischemia induces cardiac myocyte death"\n'
        "objection null\n"
        "verdict correct\n"
        "confidence high"
    )

    read = parse_verdict(text)
    assert read is not None, "the bare line-oriented reply must parse"
    assert (read.label, read.confidence) == ("correct", "high")


def test_the_runner_reads_that_reply_through_the_same_parser():
    """The runner's own `parse` must agree with the parser it delegates to.

    This is the property the delegation buys: a reply that scores here scores
    identically on the live scorer and the batch replay, because all three now
    read through `indra_belief.verdict`.
    """
    text = (
        "relation_check way to a biological process is licensed\n"
        'support "Heart ischemia induces cardiac myocyte death"\n'
        "objection null\n"
        "verdict correct\n"
        "confidence high"
    )
    assert runner.MonolithicPrompt().parse(text, "") == ("correct", "high")


def test_an_unreadable_reply_stays_absent_rather_than_becoming_a_number():
    """(None, None), never a fabricated score.

    `grid_score` returns None off-grid so an absent measurement stays absent;
    this runner must not be the path that reintroduces a default.
    """
    assert runner.MonolithicPrompt().parse("I cannot decide from this text.", "") == (
        None,
        None,
    )


def test_processed_runner_uses_commit_first_disconfirm_prompt():
    from indra_belief.scorers.monolithic._prompts_disconfirm import (
        DISCONFIRM_SYSTEM_PROMPT,
    )

    prompt = runner.MonolithicPrompt()

    assert prompt.system_prompt == DISCONFIRM_SYSTEM_PROMPT
    assert len(prompt.examples("Activation")) == 28  # 7 pairs / 14 examples


def test_complex_job_runs_relation_nature_before_verdict():
    class Response:
        def __init__(self, content):
            self.content = content

        def raise_for_status(self):
            return None

        def json(self):
            return {
                "choices": [
                    {"message": {"content": self.content}, "finish_reason": "stop"}
                ],
                "usage": {"completion_tokens": 20},
            }

    class Client:
        def __init__(self):
            self.calls = []
            self.responses = iter(
                [
                    Response(
                        '{"nature":"signaling_cascade","span":"activates Y"}'
                    ),
                    Response("verdict incorrect\nconfidence high"),
                ]
            )

        def post(self, _endpoint, json):
            self.calls.append(json)
            return next(self.responses)

    client = Client()
    row = runner.score_job(
        {
            "job_id": "1:0",
            "stmt_hash": 101,
            "source_hash": 11,
            "needs_llm": True,
            "stmt_type": "Complex",
            "subject": "X",
            "object": "Y",
            "evidence_text": "X activates Y.",
            "user_message": "CLAIM: X binds Y [Complex]\nEVIDENCE: X activates Y.",
            "subject_grounding": {"aliases": ["X alias"]},
            "object_grounding": {"aliases": ["Y alias"]},
        },
        client=client,
        prompt=runner.MonolithicPrompt(),
        endpoint="http://vllm/v1/chat/completions",
        model_id="served-model",
        max_tokens=1000,
        temperature=0.1,
    )

    assert row["verdict"] == "incorrect"
    assert row["confidence"] == "high"
    assert len(client.calls) == 2
    assert client.calls[0]["response_format"] == {"type": "json_object"}
    assert "X alias" in client.calls[0]["messages"][-1]["content"]
    assert "Relation nature (resolved)" in client.calls[1]["messages"][-1]["content"]


def test_offline_client_batches_llm_chat_calls(monkeypatch):
    instances = []

    class SamplingParams:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class Completion:
        text = "verdict correct\nconfidence high"
        token_ids = [1, 2, 3]
        finish_reason = "stop"

    class Output:
        outputs = [Completion()]
        prompt_token_ids = [10, 11]

    class LLM:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.batch_lengths = []
            instances.append(self)

        def chat(self, conversations, sampling_params, use_tqdm):
            self.batch_lengths.append(len(conversations))
            assert len(sampling_params) == len(conversations)
            assert use_tqdm is False
            return [Output() for _ in conversations]

    monkeypatch.setitem(
        sys.modules,
        "vllm",
        types.SimpleNamespace(LLM=LLM, SamplingParams=SamplingParams),
    )
    request = {
        "messages": [{"role": "user", "content": "question"}],
        "max_tokens": 100,
        "temperature": 0.1,
    }

    with runner.OfflineVllmClient(
        "model/path", batch_size=2, gpu_memory_utilization=0.8
    ) as client:
        with ThreadPoolExecutor(max_workers=2) as pool:
            responses = list(
                pool.map(lambda _i: client.post("", json=request), range(2))
            )

    assert instances[0].kwargs == {
        "model": "model/path",
        "enable_prefix_caching": True,
        "gpu_memory_utilization": 0.8,
    }
    assert instances[0].batch_lengths == [2]
    assert responses[0].json()["usage"] == {
        "prompt_tokens": 2,
        "completion_tokens": 3,
    }
