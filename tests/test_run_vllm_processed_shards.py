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
    final = runner.output_path(tmp_path, 12, 200)
    assert final.name == "verdicts-000012.limit-200.json.gz"


def test_iter_jobs_filters_hashes_before_applying_limit(tmp_path):
    shard = tmp_path / "grounded-000000.jsonl.gz"
    write_jobs(
        shard,
        [
            {"job_id": "1:0", "stmt_hash": 101, "source_hash": 11},
            {"job_id": "2:0", "stmt_hash": 202, "source_hash": 22},
            {"job_id": "3:0", "stmt_hash": 303, "source_hash": 33},
        ],
    )

    jobs = list(runner.iter_jobs(shard, limit=1, stmt_hashes={202, 303}))

    assert [job["stmt_hash"] for job in jobs] == [202]


def test_failed_job_is_retried_three_times_after_initial_attempt(monkeypatch):
    attempts = []

    def fake_score(job, **_kwargs):
        attempts.append(job["job_id"])
        if len(attempts) < 4:
            return {
                "job_id": job["job_id"],
                "verdict": None,
                "confidence": None,
                "error": "temporary",
            }
        return {
            "job_id": job["job_id"],
            "verdict": "correct",
            "confidence": "high",
        }

    monkeypatch.setattr(runner, "score_job", fake_score)
    row = runner.score_job_with_retries({"job_id": "1:0"}, retries=3)

    assert len(attempts) == 4
    assert row["verdict"] == "correct"


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
    payload = runner.finalize(shard, latest, None)
    assert payload == {
        "101": {"11": {"verdict": "correct", "confidence": "high"}},
        "-202": {
            "22": {"verdict": "incorrect", "confidence": "medium"}
        },
    }


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

    payload = runner.finalize(shard, latest, None)

    assert payload == {
        "101": {
            "11": {"verdict": "incorrect", "confidence": "low"},
            "22": {"verdict": "correct", "confidence": "high"},
        }
    }


def test_finalize_keeps_logit_and_exhausted_error(tmp_path):
    shard = tmp_path / "grounded-000000.jsonl.gz"
    write_jobs(
        shard,
        [
            {"job_id": "1:0", "stmt_hash": 101, "source_hash": 11},
            {"job_id": "2:0", "stmt_hash": 202, "source_hash": 22},
        ],
    )
    latest = {
        "1:0": {
            "verdict": "correct",
            "confidence": "high",
            "probe_delta_logit": 4.25,
        },
        "2:0": {
            "verdict": None,
            "confidence": None,
            "error": "unparseable model response",
            "attempts": 4,
            "response_preview": "bad reply",
        },
    }

    payload = runner.finalize(shard, latest, None)

    assert payload["101"]["11"]["probe_delta_logit"] == 4.25
    assert payload["202"]["22"] == {
        "verdict": "error",
        "confidence": None,
        "error": "unparseable model response",
        "attempts": 4,
        "response_preview": "bad reply",
    }


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
        # `variant` is not decoration: score_job reads the reasoning channel,
        # temperature and logprob window off it, because those three and the
        # prompt are one coherent set. A double that omits it is not standing in
        # for the real object.
        variant = runner.MonolithicPrompt(runner.DEFAULT_VARIANT).variant

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

    An unreadable categorical answer stays absent; this runner must not be the
    path that reintroduces a default measurement.
    """
    assert runner.MonolithicPrompt().parse("I cannot decide from this text.", "") == (
        None,
        None,
    )


def test_batch_runner_sends_byte_IDENTICAL_prompts_to_the_live_scorer():
    """WAS a pin on DISCONFIRM_SYSTEM_PROMPT, which conflated two things.

    The invariant worth having is PARITY: whatever variant this runner is asked
    for, it must send the same bytes the live scorer sends, because a
    calibration profile is keyed on (model, prompt sha) and a batch path that
    drifted by one character would silently score against a profile fitted for
    a prompt it never sent.

    Pinning one constant asserted parity only for the single variant the runner
    was hard-wired to, and in doing so it also froze the CHOICE of variant --
    which is what made the no-CoT path unreachable at corpus scale. Parity is
    the real property; the choice is a flag.

    HONEST ABOUT ITS OWN STRENGTH: the loop below is true BY CONSTRUCTION today,
    because MonolithicPrompt reads `system_prompt` straight off the variant. It
    cannot fail against the current implementation, and it is kept for the case
    that would make it fail -- someone reintroducing a local copy of a prompt in
    this file, which is exactly how the batch and live paths would drift apart.
    The examples() assertion below is a genuine value check.
    """
    from indra_belief.scorers.monolithic import scorer as mono

    for name, variant in mono.VARIANTS.items():
        if not name:
            continue
        assert runner.MonolithicPrompt(name).system_prompt == variant.system_prompt, (
            f"the batch runner's {name!r} prompt has drifted from the live one"
        )

    # The few-shot block is part of the request, so parity has to cover it too.
    disconfirm = runner.MonolithicPrompt("disconfirm_relnature_rf")
    assert len(disconfirm.examples("Activation")) == 28  # 7 pairs / 14 examples


def test_the_default_variant_sends_no_chain_of_thought():
    """The corpus-scale default. At 60M evidences an unasked-for CoT is the
    entire bill, and the previous default was 'whatever the chat template does'
    because the request carried no reasoning control at all."""
    assert runner.MonolithicPrompt().variant.reasoning_effort == "none"


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
        # Explicitly the relnature variant: the relation-nature step is a
        # property OF that variant, not of the runner. The default carries no
        # relation resolver and correctly skips the call.
        prompt=runner.MonolithicPrompt("disconfirm_relnature_rf"),
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


def test_main_reaches_shard_discovery_without_a_name_error(tmp_path, monkeypatch, capsys):
    """The corpus runner's entry point must actually import and run.

    REGRESSION. `main()` computed the pinned prompt's sha for the calibration
    banner using the bare name `DISCONFIRM_SYSTEM_PROMPT` — which this module
    imports INSIDE `MonolithicPrompt.__init__`, deliberately, so it stays
    importable without the scorer's dependency graph. The name was never in
    module scope, so every invocation died with NameError before touching a
    shard. Nothing caught it: the suite exercises this file's helpers, never its
    entry point.

    This drives main() far enough to prove the banner path executes, then lets
    it exit on an empty input directory. It asserts the failure mode, not a
    successful run — no server is involved.
    """
    import sys

    empty = tmp_path / "shards"
    empty.mkdir()
    monkeypatch.setattr(sys, "argv", [
        "run_vllm_processed_shards.py",
        "--input-dir", str(empty),
        "--output-dir", str(tmp_path / "out"),
        "--model", "vllm-local",
    ])
    import pytest as _pytest

    # An empty input directory is a SystemExit, not a return code. What is being
    # asserted is WHERE it dies: past the banner, at shard discovery.
    with _pytest.raises(SystemExit) as excinfo:
        runner.main()
    out = capsys.readouterr().out
    assert "calibration:" in out, (
        "the calibration banner did not print — main() failed before reaching it"
    )
    assert "no shards found" in str(excinfo.value), (
        f"expected to reach shard discovery, died earlier with: {excinfo.value}"
    )


def test_require_calibrated_refuses_an_unfitted_prompt(tmp_path, monkeypatch):
    """The pinned prompt has no fitted profile, so this must refuse rather than
    quietly publish hard-gate beliefs at corpus scale."""
    import sys

    import pytest as _pytest

    empty = tmp_path / "shards"
    empty.mkdir()
    monkeypatch.setattr(sys, "argv", [
        "run_vllm_processed_shards.py",
        "--input-dir", str(empty),
        "--output-dir", str(tmp_path / "out"),
        "--model", "vllm-local",
        "--require-calibrated",
    ])
    with _pytest.raises(SystemExit) as excinfo:
        runner.main()
    assert "no ship-approved profile" in str(excinfo.value)


def test_offline_client_preserves_requested_logprobs(monkeypatch):
    instances = []

    class SamplingParams:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class Logprob:
        def __init__(self, token, value):
            self.decoded_token = token
            self.logprob = value

    class Completion:
        text = "correct\nhigh"
        token_ids = [1]
        logprobs = [{1: Logprob("correct", -0.1), 2: Logprob("incorrect", -2.1)}]
        finish_reason = "stop"

    class Output:
        outputs = [Completion()]
        prompt_token_ids = [10, 11]

    class LLM:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.params = []
            instances.append(self)

        def chat(self, conversations, sampling_params, use_tqdm, **kwargs):
            self.params.extend(sampling_params)
            assert len(conversations) == 2
            assert use_tqdm is False
            assert kwargs == {"chat_template_kwargs": {"enable_thinking": False}}
            return [Output() for _ in conversations]

    monkeypatch.setitem(
        sys.modules,
        "vllm",
        types.SimpleNamespace(LLM=LLM, SamplingParams=SamplingParams),
    )
    request = {
        "messages": [{"role": "user", "content": "question"}],
        "max_tokens": 100,
        "temperature": 0,
        "logprobs": True,
        "top_logprobs": 128,
        "chat_template_kwargs": {"enable_thinking": False},
    }

    with runner.OfflineVllmClient(
        "model/path", batch_size=2, gpu_memory_utilization=0.8
    ) as client:
        with ThreadPoolExecutor(max_workers=2) as pool:
            responses = list(
                pool.map(lambda _i: client.post("", json=request), range(2))
            )

    assert instances[0].params[0].kwargs["logprobs"] == 128
    content = responses[0].json()["choices"][0]["logprobs"]["content"]
    assert content[0]["token"] == "correct"
    assert {row["token"] for row in content[0]["top_logprobs"]} == {
        "correct",
        "incorrect",
    }
